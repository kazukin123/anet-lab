#include "dqn_based_math.hpp"
#include <algorithm>
#include <charconv>
#include <tuple>
#include <cmath>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <set>
#include <utility>
#include <unordered_set>
#include "anet/log.hpp"
#include "anet/profile.hpp"
#include "anet/tensor_check.hpp"
#include "anet/tensor_util.hpp"
#include "anet/str_util.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/replay_experience.hpp"
#include "anet/image.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/nn_head.hpp"
#include "dqn_based_heads.hpp"

using namespace anet::rl::dqn;
namespace LOG = anet::log;

std::array<float, 3> anet::rl::dqn::MeasureNoisyQProbe(
    const torch::Tensor& mean_q, int num_draws, const std::function<torch::Tensor(int)>& draw)
{
    ANET_PROFILE_FUNC();
    torch::NoGradGuard no_grad;
    if (num_draws <= 0 || mean_q.dim() != 2 || mean_q.size(0) <= 0 || mean_q.size(1) <= 0
        || mean_q.scalar_type() != torch::kFloat32) {
        ANET_SYSTEM_ERROR("Noisy probe requires num_draws > 0 and float32 Q [B,A] with B,A > 0; num_draws="
            << num_draws << " shape=" << mean_q.sizes() << " dtype=" << c10::toString(mean_q.scalar_type()));
    }
    const auto baseline_actions = mean_q.argmax(1);
    auto absolute_diff = torch::zeros({}, mean_q.options());
    auto disagreements = torch::zeros({}, mean_q.options());
    auto counts = torch::zeros_like(mean_q);
    const auto increments = torch::ones({mean_q.size(0), 1}, mean_q.options());
    // 1 回に B 状態だけを評価し、三指標のための推論を共用して逐次集計する。
    for (int iteration = 0; iteration < num_draws; ++iteration) {
        const auto q = draw(iteration);
        if (q.sizes() != mean_q.sizes() || q.device() != mean_q.device() || q.scalar_type() != torch::kFloat32) {
            ANET_SYSTEM_ERROR("Noisy probe draw=" << iteration << " expected float32 Q shape=" << mean_q.sizes()
                << " device=" << mean_q.device() << "; actual shape=" << q.sizes() << " device=" << q.device()
                << " dtype=" << c10::toString(q.scalar_type()));
        }
        const auto actions = q.argmax(1);
        absolute_diff.add_((q - mean_q).abs().sum());
        disagreements.add_(actions.ne(baseline_actions).to(torch::kFloat32).sum());
        counts.scatter_add_(1, actions.unsqueeze(1), increments);
    }
    // 状態ごとの反復頻度から entropy を求め、最後に状態間で平均する。
    const auto frequencies = counts / num_draws;
    const auto terms = torch::where(frequencies.gt(0), frequencies * frequencies.log(), torch::zeros_like(frequencies));
    const auto packed = torch::stack({
        absolute_diff / (static_cast<double>(num_draws) * mean_q.numel()),
        disagreements / (static_cast<double>(num_draws) * mean_q.size(0)),
        -terms.sum(1).mean(),
    }).cpu();
    const auto values = packed.accessor<float, 1>();
    return {values[0], values[1], values[2]};
}

torch::Tensor anet::rl::dqn::MakeScaledLogPolicy(const torch::Tensor& score, float entropy_tau)
{
    // maxを引いてから指数化し、温度を掛けたlog-policyをFP32実空間で返す。
    const auto q = score.detach().to(torch::kFloat32);
    const auto centered = q - std::get<0>(q.max(1, true));
    return centered - entropy_tau * torch::logsumexp(centered / entropy_tau, 1, true);
}

MunchausenTargetTerms anet::rl::dqn::MakeMunchausenTargetTerms(
    const torch::Tensor& current_score, const torch::Tensor& next_score,
    const torch::Tensor& next_mean_q, const torch::Tensor& actions,
    const MunchausenValues& config, bool collect_diagnostics)
{
    ANET_PROFILE_FUNC();
    torch::NoGradGuard guard;
    
    if (current_score.dim() != 2 || current_score.size(1) < 1
        || next_score.sizes() != current_score.sizes() || next_mean_q.sizes() != next_score.sizes()
        || actions.dim() != 1 || actions.size(0) != current_score.size(0)) {
        ANET_SYSTEM_ERROR("Munchausen requires matching scores/mean Q [B,A] and actions [B]; current="
            << current_score.sizes() << " next=" << next_score.sizes() << " mean=" << next_mean_q.sizes()
            << " actions=" << actions.sizes());
    }

    // 方策スコアと価値用平均Qを分離し、risk方策でも平均Q基準のsoft gapを保持する。
    const auto current_log = MakeScaledLogPolicy(current_score, config.entropy_tau);
    const auto selected_log = current_log.gather(1, actions.unsqueeze(1)).squeeze(1);
    const auto bonus = config.alpha * selected_log.clamp(config.clip_value_min, 0.0f);
    const auto next_log = MakeScaledLogPolicy(next_score, config.entropy_tau);
    const auto next_policy = (next_log / config.entropy_tau).exp();
    torch::Tensor diagnostics;
    if (collect_diagnostics) {
        const auto mean_q = next_mean_q.detach().to(torch::kFloat32);
        const auto soft_value = (next_policy * (mean_q - next_log)).sum(1);
        diagnostics = StackDiagnosticValues<MunchausenMetric::Count,
        MunchausenMetric::ScaledLogPolicyMean, MunchausenMetric::ClipRatio, MunchausenMetric::BonusMean,
        MunchausenMetric::NextEntropy, MunchausenMetric::SoftGap>(
            selected_log.mean(),
            selected_log.lt(config.clip_value_min).to(torch::kFloat32).mean(),
            bonus.mean(),
            -(next_policy * next_log).sum(1).mean() / config.entropy_tau,
            (soft_value - std::get<0>(mean_q.max(1))).mean()
        );
    }
    return MunchausenTargetTerms{
        .bonus = bonus, .next_policy = next_policy, .next_scaled_log_policy = next_log, .diagnostics = diagnostics,
    };
}

static void ValidateTauGeneratorArguments(int64_t batch_size, int64_t num_taus, const std::string& sample_mode)
{
    if (batch_size < 0) {
        ANET_SYSTEM_ERROR("TauGenerator requires batch_size>=0, but batch_size=" << batch_size << ".");
    }
    if (num_taus <= 0) {
        ANET_SYSTEM_ERROR("TauGenerator requires num_taus>0, but num_taus=" << num_taus << ".");
    }
    if (sample_mode != "random" && sample_mode != "fixed"
        && sample_mode != "stratified" && sample_mode != "systematic"
        && sample_mode != "antithetic") {
        ANET_SYSTEM_ERROR("TauGenerator received sample_mode=" << sample_mode
            << "; expected one of: random, fixed, stratified, systematic, antithetic.");
    }
}

torch::Tensor anet::rl::dqn::MakeTauMidpointPositions(int64_t num_taus, const torch::TensorOptions& options)
{
    return (torch::arange(num_taus, options) + 0.5f) / static_cast<float>(num_taus);
}

static torch::Tensor MakeNormalizedTauPositions(
    int64_t batch_size,
    int64_t num_taus,
    const std::string& sample_mode,
    const torch::Device& device,
    anet::RandomGenerator& rnd)
{
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    if (sample_mode == "random") {
        // 乱数を対象device上で一括生成し、行・列ごとに独立な位置を返す。
        auto gen = rnd.GetTorchGenerator(device);
        return torch::rand({ batch_size, num_taus }, gen, options);
    }
    if (sample_mode == "stratified") {
        // 各stratumの層内位置を行ごとに一括生成し、列順の被覆を維持する。
        auto gen = rnd.GetTorchGenerator(device);
        const auto unit = torch::rand({ batch_size, num_taus }, gen, options);
        const auto stratum = torch::arange(num_taus, options).unsqueeze(0);
        return (stratum + unit) / static_cast<float>(num_taus);
    }
    if (sample_mode == "systematic") {
        // 行ごとに1つの位相を生成し、全stratumへbroadcastして等間隔を維持する。
        auto gen = rnd.GetTorchGenerator(device);
        const auto phase = torch::rand({ batch_size, 1 }, gen, options);
        const auto stratum = torch::arange(num_taus, options).unsqueeze(0);
        return (stratum + phase) / static_cast<float>(num_taus);
    }
    if (sample_mode == "antithetic") {
        // 前半の乱数を同じindex順で鏡映し、奇数Kでは最後の乱数を独立点として使う。
        const int64_t pair_count = num_taus / 2;
        const int64_t draw_count = (num_taus + 1) / 2;
        auto gen = rnd.GetTorchGenerator(device);
        const auto unit = torch::rand({ batch_size, draw_count }, gen, options);
        const auto first = unit.slice(1, 0, pair_count);
        const auto mirrored = 1.0f - first;
        if (num_taus % 2 == 0) {
            return torch::cat({ first, mirrored }, 1);
        }
        return torch::cat({ first, mirrored, unit.slice(1, pair_count, draw_count) }, 1);
    }

    // midpoint配置ではgeneratorに触れず、全batchへ同じ正規化位置を共有する。
    return MakeTauMidpointPositions(num_taus, options).unsqueeze(0).expand({ batch_size, num_taus });
}

torch::Tensor anet::rl::dqn::GenerateTaus(
    int64_t batch_size,
    int64_t num_taus,
    const std::string& sample_mode,
    float tau_min,
    float tau_max,
    const torch::Device& device,
    anet::RandomGenerator& rnd)
{
    ANET_PROFILE_FUNC();
    ValidateTauGeneratorArguments(batch_size, num_taus, sample_mode);

    // mode固有の正規化位置を、CPU同期を挟まず指定範囲へ写像する。
    const auto positions = MakeNormalizedTauPositions(batch_size, num_taus, sample_mode, device, rnd);
    return tau_min + positions * (tau_max - tau_min);
}

torch::Tensor anet::rl::dqn::GenerateTaus(
    const torch::Tensor& tau_min_per_env,
    float tau_max,
    int64_t num_taus,
    const std::string& sample_mode,
    anet::RandomGenerator& rnd)
{
    ANET_PROFILE_FUNC();
    if (tau_min_per_env.dim() != 1) {
        ANET_SYSTEM_ERROR("TauGenerator requires tau_min_per_env rank=1 (B), but shape="
            << tau_min_per_env.sizes() << ".");
    }
    ValidateTauGeneratorArguments(tau_min_per_env.size(0), num_taus, sample_mode);

    const auto device = tau_min_per_env.device();
    const auto lower = tau_min_per_env.to(torch::kFloat32).unsqueeze(1);
    // 共通範囲版と同じ正規化位置を使い、絶対位置だけをenvごとの下限で決める。
    const auto positions = MakeNormalizedTauPositions(
        tau_min_per_env.size(0), num_taus, sample_mode, device, rnd);
    return lower + positions * (tau_max - lower);
}

anet::rl::ReplayInitialPriorityMode anet::rl::dqn::ParseReplayInitialPriorityMode(const LearnerConfig& config)
{
    // 文字列表現をReplayBufferへ渡す内部modeへ変換し、未知値は設定ミスとして停止する。
    if (config.per_initial_priority_mode == "fixed") {
        return ReplayInitialPriorityMode::FIXED;
    } else if (config.per_initial_priority_mode == "max") {
        return ReplayInitialPriorityMode::MAX;
    } else if (config.per_initial_priority_mode == "actor_approx") {
        return ReplayInitialPriorityMode::ACTOR_APPROX;
    }

    ANET_SYSTEM_ERROR("Invalid learner.per_initial_priority_mode='" << config.per_initial_priority_mode
        << "'. Expected one of: fixed, max, actor_approx.");
    return ReplayInitialPriorityMode::FIXED; // ANET_SYSTEM_ERROR後のMSVC制御経路警告を抑止する。
}

void anet::rl::dqn::ValidateReplayPriorityConfig(
    const LearnerConfig& config, ReplayInitialPriorityMode initial_priority_mode)
{
    // modeを切り替えても同じconfigを再利用できるよう、関連する数値設定を一括検証する。
    if (!std::isfinite(config.per_initial_priority) || config.per_initial_priority < 0.0f) {
        ANET_SYSTEM_ERROR("Invalid learner.per_initial_priority=" << config.per_initial_priority
            << ". Expected a finite value >= 0; use learner.per_initial_priority_mode=max for maximum initialization.");
    }
    if (!std::isfinite(config.per_eps) || config.per_eps < 0.0f) {
        ANET_SYSTEM_ERROR("Invalid learner.per_eps=" << config.per_eps << ". Expected a finite value >= 0.");
    }
    if (config.use_per && config.per_eps == 0.0f) {
        LOG::warn_once() << "learner.per_eps=0 is allowed with learner.use_per=true, but zero-TD-error transitions may "
            << "receive zero sampling priority. Set learner.per_eps to a finite value > 0 to keep a positive priority floor.";
    }
    if (config.use_per && initial_priority_mode == ReplayInitialPriorityMode::ACTOR_APPROX && config.per_alpha == 0.0f) {
        LOG::warn_once() << "learner.per_initial_priority_mode=actor_approx with learner.per_alpha=0 is allowed, but Actor "
            << "priority does not affect sampling. Set learner.per_alpha to a finite value > 0 to enable prioritized sampling.";
    }
    if (config.use_per_prio_clip) {
        if (!std::isfinite(config.per_prio_clip_value) || config.per_prio_clip_value < 0.0f) {
            ANET_SYSTEM_ERROR("Invalid learner.per_prio_clip_value=" << config.per_prio_clip_value
                << ". Expected a finite value >= 0 when learner.use_per_prio_clip=true.");
        }
        if (config.per_prio_clip_value == 0.0f) {
            LOG::warn_once() << "learner.use_per_prio_clip=true with learner.per_prio_clip_value=0 is allowed, but all PER "
                << "priorities are clipped to zero. Set learner.per_prio_clip_value to a finite value > 0 or disable "
                << "learner.use_per_prio_clip.";
        } else if (config.per_prio_clip_value <= config.per_eps) {
            LOG::warn_once() << "learner.per_prio_clip_value=" << config.per_prio_clip_value
                << " <= learner.per_eps=" << config.per_eps
                << " may collapse priority differences. Set learner.per_prio_clip_value > learner.per_eps or disable "
                << "learner.use_per_prio_clip.";
        }
    }
}

torch::Tensor anet::rl::dqn::TransformH(const torch::Tensor& x, float epsilon)
{
    return x.sign() * (torch::sqrt(x.abs() + 1.0f) - 1.0f) + epsilon * x;
}

float anet::rl::dqn::TransformH(float x, float epsilon)
{
    const float sign = static_cast<float>((x > 0.0f) - (x < 0.0f));
    return sign * (std::sqrt(std::abs(x) + 1.0f) - 1.0f) + epsilon * x;
}

torch::Tensor anet::rl::dqn::TransformHInv(const torch::Tensor& x, float epsilon)
{
    auto abs_x = x.abs();
    auto inner = (torch::sqrt(1.0f + 4.0f * epsilon * (abs_x + 1.0f + epsilon)) - 1.0f) / (2.0f * epsilon);
    return x.sign() * (inner * inner - 1.0f);
}

float anet::rl::dqn::TransformHInv(float x, float epsilon)
{
    const float sign = static_cast<float>((x > 0.0f) - (x < 0.0f));
    const float inner = (std::sqrt(1.0f + 4.0f * epsilon * (std::abs(x) + 1.0f + epsilon)) - 1.0f)
        / (2.0f * epsilon);
    return sign * (inner * inner - 1.0f);
}

// 分解の和と差は桁落ちに敏感なので、同じ逆関数を引き算なしの形で評価する。
// combinedの既存系列は従来の逆変換を保つ。
torch::Tensor anet::rl::dqn::InverseDecomposedValue(const torch::Tensor& value, float epsilon)
{
    const float coefficient = 1.0f + 2.0f * epsilon;
    const auto absolute = value.abs();
    const auto z = (2.0f * absolute) / (torch::sqrt(coefficient * coefficient + 4.0f * epsilon * absolute) + coefficient);
    return value.sign() * z * (z + 2.0f);
}

float anet::rl::dqn::InverseDecomposedValue(float value, float epsilon)
{
    const float coefficient = 1.0f + 2.0f * epsilon;
    const float absolute = std::abs(value);
    const float z = (2.0f * absolute) / (std::sqrt(coefficient * coefficient + 4.0f * epsilon * absolute) + coefficient);
    return std::copysign(z * (z + 2.0f), value);
}

PerRawPriorityBatchResult anet::rl::dqn::MakePerRawPriorityBatch(
    const torch::Tensor& td_error, float per_eps, bool use_clip, float clip_value)
{
    // clip前の値で変更件数を判定し、上限と元から等しいpriorityは数えない。
    auto unclipped_priorities = td_error.abs() + per_eps;
    auto clipped_count = use_clip
        ? unclipped_priorities.gt(clip_value).sum()
        : torch::zeros({}, unclipped_priorities.options().dtype(torch::kInt64));
    auto priorities = use_clip
        ? unclipped_priorities.clamp_max(clip_value)
        : unclipped_priorities;
    return {
        .priorities = std::move(priorities),
        .clipped_count = std::move(clipped_count),
    };
}

torch::Tensor anet::rl::dqn::MakePerRawPriority(
    const torch::Tensor& td_error, float per_eps, bool use_clip, float clip_value)
{
    return MakePerRawPriorityBatch(td_error, per_eps, use_clip, clip_value).priorities;
}

float anet::rl::dqn::MakePerRawPriority(
    float td_error, float per_eps, bool use_clip, float clip_value)
{
    const float priority = std::abs(td_error) + per_eps;
    return use_clip ? std::min(priority, clip_value) : priority;
}

ActorQHintRow anet::rl::dqn::DecodeActorQHint(std::span<const float> payload)
{
    // ReplayBufferから渡されたopaque rowは、DQN推定器に入る時点でだけ列へdecodeする。
    if (payload.size() != kActorQHintColumnCount) {
        ANET_SYSTEM_ERROR("DQN Actor Q hint must have K=" << kActorQHintColumnCount << ". actual=" << payload.size());
    }
    return ActorQHintRow{
        .actor_q_sa = payload[kActorQSaColumn],
        .actor_state_value = payload[kActorStateValueColumn],
        .munchausen_term = payload[kActorMunchausenTermColumn],
        .intrinsic_q_sa = payload[kActorIntrinsicQSaColumn],
        .intrinsic_state_value = payload[kActorIntrinsicStateValueColumn],
    };
}

namespace {

class DqnInitialPriorityEstimator final : public anet::rl::InitialPriorityEstimator {
public:
    explicit DqnInitialPriorityEstimator(const LearnerConfig& config)
        : use_tbo_(config.use_tbo), tbo_epsilon_(config.tbo_epsilon), per_eps_(config.per_eps)
        , use_clip_(config.use_per_prio_clip), clip_value_(config.per_prio_clip_value)
        , decomposed_(config.intrinsic.decomposed), priority_lambda_(config.intrinsic.priority_lambda)
    {
    }

    bool ValidateHint(std::span<const float> hint) const override
    {
        const auto decoded = DecodeActorQHint(hint);
        return std::isfinite(decoded.actor_q_sa) && std::isfinite(decoded.actor_state_value)
            && std::isfinite(decoded.munchausen_term) && std::isfinite(decoded.intrinsic_q_sa)
            && std::isfinite(decoded.intrinsic_state_value);
    }

    std::optional<float> Estimate(const anet::rl::InitialPriorityEstimateInput& input) const override
    {
        if (!ValidateHint(input.start_hint)
            || (!input.terminal && !ValidateHint(input.bootstrap_hint))) {
            return std::nullopt;
        }
        const auto start_hint = DecodeActorQHint(input.start_hint);
        // TBOではbootstrap値を実空間へ戻してn-step収益と加算し、Learnerと同じh空間へ再変換する。
        float target = input.target_return + start_hint.munchausen_term;
        if (!input.terminal) {
            const float bootstrap_state_value = DecodeActorQHint(input.bootstrap_hint).actor_state_value;
            const float bootstrap = use_tbo_
                ? (decomposed_ ? InverseDecomposedValue(bootstrap_state_value, tbo_epsilon_) : TransformHInv(bootstrap_state_value, tbo_epsilon_))
                : bootstrap_state_value;
            target += input.discount * bootstrap;
        }
        if (use_tbo_) target = TransformH(target, tbo_epsilon_);
        // Learnerと同じraw priority確定policyをscalar overloadで適用する。
        float error = std::abs(target - start_hint.actor_q_sa);
        if (decomposed_) {
            if (input.extra_component_returns.size() != 1 || input.extra_component_discounts.size() != 1) {
                ANET_SYSTEM_ERROR("DQN decomposed priority requires one additional return and discount. returns="
                    << input.extra_component_returns.size() << " discounts=" << input.extra_component_discounts.size());
            }
            float intrinsic_target = input.extra_component_returns.front();
            if (!input.terminal) {
                const float value = DecodeActorQHint(input.bootstrap_hint).intrinsic_state_value;
                intrinsic_target += input.extra_component_discounts.front() * (use_tbo_ ? InverseDecomposedValue(value, tbo_epsilon_) : value);
            }
            if (use_tbo_) intrinsic_target = TransformH(intrinsic_target, tbo_epsilon_);
            error += priority_lambda_ * std::abs(intrinsic_target - start_hint.intrinsic_q_sa);
        }
        const float priority = MakePerRawPriority(error, per_eps_, use_clip_, clip_value_);
        if (!std::isfinite(priority)) return std::nullopt;
        return priority;
    }
private:
    bool use_tbo_;
    float tbo_epsilon_;
    float per_eps_;
    bool use_clip_;
    float clip_value_;
    bool decomposed_;
    float priority_lambda_;
};

} // namespace

std::unique_ptr<anet::rl::InitialPriorityEstimator> anet::rl::dqn::CreateInitialPriorityEstimator(
    const LearnerConfig& config)
{
    return std::make_unique<DqnInitialPriorityEstimator>(config);
}

QuantileTailStatistics anet::rl::dqn::CalculateQuantileTailStatistics(const torch::Tensor& tau_ordered_quantiles)
{
    if (!tau_ordered_quantiles.defined() || tau_ordered_quantiles.dim() < 1) {
        ANET_SYSTEM_ERROR("CalculateQuantileTailStatistics requires a defined rank>=1 tensor.");
    }

    // AMP/BF16入力も診断集約前にfloat32へ切り離し、学習graphへ接続しない。
    const auto quantiles = tau_ordered_quantiles.detach().to(torch::kFloat32);
    const int64_t num_quantiles = quantiles.size(-1);
    std::vector<int64_t> result_shape(
        quantiles.sizes().begin(), quantiles.sizes().end() - 1);
    const auto result_options = quantiles.options().dtype(torch::kFloat32);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    if (num_quantiles < 2) {
        return QuantileTailStatistics{
            .upper_truncated_std = torch::full(result_shape, nan, result_options),
            .lower_truncated_std = torch::full(result_shape, nan, result_options),
            .crossing_ratio = torch::full({}, nan, result_options),
        };
    }

    // tau順を維持したまま中央quantileを除外し、上下同数のtruncated stdを求める。
    const int64_t half = num_quantiles / 2;
    torch::Tensor median;
    if (num_quantiles % 2 == 0) {
        median = (quantiles.select(-1, half - 1) + quantiles.select(-1, half)) / 2.0f;
    } else {
        median = quantiles.select(-1, half);
    }
    const auto lower = quantiles.slice(-1, 0, half);
    const auto upper = quantiles.slice(-1, num_quantiles - half, num_quantiles);
    const auto upper_std = (upper - median.unsqueeze(-1)).square().mean(-1).sqrt();
    const auto lower_std = (median.unsqueeze(-1) - lower).square().mean(-1).sqrt();
    const auto crossing = quantiles.slice(-1, 0, num_quantiles - 1)
        .gt(quantiles.slice(-1, 1, num_quantiles))
        .to(torch::kFloat32).mean();
    return QuantileTailStatistics{
        .upper_truncated_std = upper_std,
        .lower_truncated_std = lower_std,
        .crossing_ratio = crossing,
    };
}

void DecomposedValueCombiner::Combine(anet::TensorDict& output, float beta, bool use_tbo, float epsilon)
{
    ANET_PROFILE_FUNC();
    const auto combine = [&](const torch::Tensor& external, const torch::Tensor& internal) {
        if (external.sizes() != internal.sizes())
            ANET_SYSTEM_ERROR("Decomposed values shape mismatch: extrinsic=" << external.sizes() << " intrinsic=" << internal.sizes());
        // β=0は外的の値をそのまま使い、hの往復の丸めでgreedy行動が変わるのを防ぐ。
        if (beta == 0.0f) return external;
        if (!use_tbo) return external + beta * internal;
        return dqn::TransformH(InverseDecomposedValue(external, epsilon) + beta * InverseDecomposedValue(internal, epsilon), epsilon);
    };
    if (output.Get(DecomposedHeadFactory::kExtrinsicQDistOutputKey)) {
        const auto distribution = combine(output.At(DecomposedHeadFactory::kExtrinsicQDistOutputKey), output.At(DecomposedHeadFactory::kIntrinsicQDistOutputKey));
        output.Set(HeadFactoryBase::kQDistOutputKey, distribution);
        output.Set(HeadFactoryBase::kQOutputKey, distribution.mean(-1));
    } else {
        output.Set(HeadFactoryBase::kQOutputKey, combine(output.At(DecomposedHeadFactory::kExtrinsicQOutputKey), output.At(DecomposedHeadFactory::kIntrinsicQOutputKey)));
    }
}

anet::TensorDict DqnForwardExecutor::Forward(const anet::TensorDict& input,
    const anet::nn::NetworkForwardCallInfo& info, anet::nn::NetworkForwardCapture* capture) const
{
    auto output = forward_.Forward(input, info, capture);
    if (beta_) DecomposedValueCombiner::Combine(output, *beta_, use_tbo_, epsilon_);
    return output;
}

torch::Tensor anet::rl::dqn::MakeRiskBiasedScore(float tau, const torch::Tensor& q_quantiles, bool use_tail_mean)
{

    // GPU同期を避けるため、sizes() からメタデータのみ取得
    const int64_t n_quantiles = q_quantiles.size(-1);

    // インデックス決定: floor(tau * (N-1)) clampして範囲外アクセスを防止
    int64_t tau_idx = static_cast<int64_t>(tau * (n_quantiles - 1));
    tau_idx = std::max<int64_t>(0, std::min<int64_t>(tau_idx, n_quantiles - 1));
    ANET_LOG_DEBUG("tau_idx=" << tau_idx);

    // 最後の次元(分位点)を確実に昇順(小さい順)にソートする
    torch::Tensor sorted_quantiles = std::get<0>(q_quantiles.sort(-1, /*descending=*/false));

    torch::Tensor uqe_values;
    if (use_tail_mean) {
        // 上位分位点すべての平均を使う場合

        // tau_idx から 最後まで (N-1) の範囲を切り出す
        auto tail_values = sorted_quantiles.slice(-1, tau_idx, n_quantiles);  // (B, A, N - tau_idx)
        //ANET_LOG_DEBUG("tail_values=" << anet::ToString(tail_values));

        // 切り出した範囲の平均をとる
        uqe_values = tail_values.mean(-1);
    } else {
        // 特定の分位点におけるQ値を取得
        uqe_values = sorted_quantiles.select(-1, tau_idx);  // (B, A, N) -> (B, A)
    }
    //ANET_LOG_DEBUG("uqe_values=" << anet::ToString(uqe_values));

    return uqe_values;
}

std::optional<int64_t> anet::rl::dqn::ParseAttentionEntropyMetric(std::string_view key)
{
    constexpr std::string_view prefix = "attention_entropy.";
    if (!key.starts_with(prefix)) return std::nullopt;
    const auto suffix = key.substr(prefix.size());
    int64_t layer = -1;
    bool valid = suffix.size() > 2 && suffix.front() == '[' && suffix.back() == ']';
    if (valid) {
        const auto text = suffix.substr(1, suffix.size() - 2);
        const auto result = std::from_chars(text.data(), text.data() + text.size(), layer);
        valid = result.ec == std::errc{} && result.ptr == text.data() + text.size() && layer >= 0;
    }
    if (!valid) ANET_SYSTEM_ERROR("Invalid scalar key='" << key << "'; expected attention_entropy.[n] with integer n>=0.");
    return layer;
}
