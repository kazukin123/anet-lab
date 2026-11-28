// default_dqn_agent.cpp

#include "default_dqn_agent.hpp"
#include <algorithm>
#include <atomic>
#include <random>
#include <functional>
#include <memory>
#include <numeric>
#include <limits>
#include <sstream>
#include <torch/torch.h>
#include <cmath>
#include "anet/profile.hpp"
#include "anet/str_util.hpp"
#include "anet/nn_util.hpp"
#include "anet/tensor_util.hpp"
#include "anet/tensor_store.hpp"
#include "anet/tensor_check.hpp"
#include "anet/log.hpp"
#include "anet/random.hpp"
#include "anet/config.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/stacker.hpp"
#include "anet/serialize.hpp"
#include "anet/intrinsic.hpp"
#include "anet/nn_head.hpp"
#include "dqn_heads.hpp"
#include "dqn_assembler.hpp"


using namespace anet::rl::dqn;
namespace LOG = anet::log;

constexpr char kPerceptionSeed[] = "perception";
constexpr char kPerceptionStructureTag[] = "perception.structure";
constexpr char kPerceptionDetailTag[] = "perception.detail";

// Agent全体の保存形式。第1版の識別子は既存primary checkpointのため保持する。
constexpr char kCheckpointFormatV1[] = "DefaultDQNAgent";
constexpr char kCheckpointFormatV2[] = "DefaultDQNAgent-2";
constexpr char kCheckpointIntrinsicV1[] = "DefaultDQNAgent;intrinsic";
constexpr char kCheckpointIntrinsicV2[] = "DefaultDQNAgent-2;intrinsic";
constexpr char kCheckpointFormatV3[] = "DefaultDQNAgent-3";
constexpr char kCheckpointIntrinsicV3[] = "DefaultDQNAgent-3;intrinsic";
constexpr char kIntrinsicPrefix[] = "intrinsic.";
constexpr char kIntrinsicRewardScale[] = "intrinsic.reward_scaler.scale";
constexpr char kIntrinsicBonus[] = "intrinsic.bonus";
constexpr char kIntrinsicError[] = "intrinsic.error";
constexpr char kIntrinsicBonusMean[] = "intrinsic.bonus_mean";
constexpr char kIntrinsicBonusMax[] = "intrinsic.bonus_max";
constexpr char kIntrinsicBonusShare[] = "intrinsic.bonus_share";
constexpr char kIntrinsicBonusMin[] = "intrinsic.bonus_min";
constexpr char kIntrinsicBonusPosRatio[] = "intrinsic.bonus_pos_ratio";
constexpr char kIntrinsicLearnCount[] = "intrinsic.learn_count";
constexpr char kIntrinsicLearnSeed[] = "intrinsic.learn";
constexpr char kIntrinsicRndGraphPrefix[] = "intrinsic.rnd.";
constexpr char kNetworkStructureSuffix[] = ".net.structure";
constexpr char kNetworkDetailSuffix[] = ".net.detail";
constexpr const char* kSkipStructureTag = "skip.net.structure";
constexpr const char* kSkipDetailTag = "skip.net.detail";
constexpr const char* kSkipNetworkSeed = "skip_network";

static ReplayFrameStack ResolveReplayFrameStack(const DefaultDQNAgentConfig& config, const anet::TensorSpecMap& frame_specs)
{
    const auto& stack = config.frame_stack;
    if (stack.count < 1) {
        ANET_SYSTEM_ERROR("Invalid " << config.GetFullKey("frame_stack.count") << "=" << stack.count << "; expected >= 1.");
    }
    // 入力の Frame spec に対して、重ねる対象を確定する。
    ReplayFrameStack resolved{ .count = stack.count };
    if (stack.keys.mode == FrameStackConfig::kAllKeys) {
        for (const auto& [key, spec] : frame_specs) resolved.keys.push_back(key);
        std::ranges::sort(resolved.keys);
    } else if (stack.keys.mode == FrameStackConfig::kListedKeys) {
        const auto field = config.GetFullKey("frame_stack.keys.listed.names");
        if (stack.keys.listed.names.empty()) {
            ANET_SYSTEM_ERROR("Invalid " << field << "=[]; expected at least one Frame key.");
        }
        for (const auto& key : stack.keys.listed.names) {
            if (!frame_specs.contains(key)) {
                ANET_SYSTEM_ERROR("Invalid " << field << " key='" << key << "'; expected a key in the Frame spec.");
            }
        }
        resolved.keys = stack.keys.listed.names;
    } else {
        ANET_SYSTEM_ERROR("Invalid " << config.GetFullKey("frame_stack.keys.mode") << "='" << stack.keys.mode
            << "'; expected " << FrameStackConfig::kAllKeys << " or " << FrameStackConfig::kListedKeys << ".");
    }
    // Actor が入力に足す開始の印と、本来の観測キーの衝突を拒む。
    if (frame_specs.contains(kPerceptionStartKey)) {
        ANET_SYSTEM_ERROR("Frame key='" << kPerceptionStartKey << "' conflicts with the reserved start key for "
            << anet::rl::kForwardKindPerceptionFromSequence << ".");
    }
    return resolved;
}

static std::shared_ptr<anet::SpecForwarder> BuildActorPerception(const anet::ForwarderBuilder& builder,
    const anet::TensorSpecMap& frame_specs, const ReplayFrameStack& stack, int row_count, anet::seed_t seed)
{
    auto input_specs = frame_specs;
    input_specs.emplace(kPerceptionStartKey, anet::TensorSpec{ .dtype = torch::kBool });
    const auto request = anet::rl::MakePerceptionFromSequenceRequest(row_count, seed, kPerceptionStartKey,
        stack.count, stack.keys);
    return CompletePerception(builder.BuildForwarder(request, input_specs), stack.keys, kPerceptionStartKey);
}

namespace {
    // 読み出し順はエポックごとに作る。分散時は、ためる側と学ぶ側の領域を交代で使う。
    class IntrinsicLearnRolloutBatcher {
    public:
        IntrinsicLearnRolloutBatcher(int64_t lanes, int64_t length, int64_t batch_size,
            int64_t steps, int64_t epochs, bool spread)
            : lanes_(lanes), length_(length), batch_size_(batch_size), total_steps_(steps),
              steps_per_epoch_(steps / epochs), spread_(spread), collecting_(length), training_(length) {}
    public:
        int64_t AppendAndLearn(const anet::TensorDict& observations,
            anet::rl::intrinsic::LifelongNoveltyEstimator& estimator, std::mt19937_64& random)
        {
            ANET_PROFILE_SCOPE(intrinsic_rollout_append);
            collecting_.Append(observations);
            ANET_PROFILE_SCOPE_END(intrinsic_rollout_append);
            if (!spread_) {
                // ため終えた intake の中で一括学習し、次の rollout に備えて空にする。
                if (collecting_.GetTimeCount() < length_) return 0;
                ANET_PROFILE_SCOPE(intrinsic_rollout_batch);
                const auto learned = LearnSteps(collecting_, total_steps_, estimator, random);
                collecting_.Clear();
                next_step_ = 0;
                return learned;
            }

            int64_t learned = 0;
            if (training_.GetTimeCount() != 0) {
                // 商と余りを分け、intake 数と step 数を掛けずに累計の整数部分を得る。
                spread_remainder_ += total_steps_ % length_;
                const auto steps = total_steps_ / length_ + spread_remainder_ / length_;
                spread_remainder_ %= length_;
                learned = LearnSteps(training_, steps, estimator, random);
            }
            if (collecting_.GetTimeCount() == length_) {
                // 前の rollout の学習を終えた領域を、次の蓄積に回す。
                training_.Clear();
                std::swap(collecting_, training_);
                next_step_ = 0;
                spread_remainder_ = 0;
            }
            return learned;
        }

        void Reset()
        {
            collecting_.Clear();
            training_.Clear();
            next_step_ = 0;
            spread_remainder_ = 0;
        }
    private:
        int64_t LearnSteps(const anet::LaneTimeTensorStore& source, int64_t steps,
            anet::rl::intrinsic::LifelongNoveltyEstimator& estimator, std::mt19937_64& random)
        {
            int64_t learned = 0;
            for (int64_t step = 0; step < steps; ++step, ++next_step_) {
                ANET_PROFILE_SCOPE(intrinsic_rollout_selection);
                // エポックの最初に全観測を並べ替え、以後は同じ並びを使う。
                const auto epoch_step = next_step_ % steps_per_epoch_;
                if (epoch_step == 0) {
                    order_.resize(lanes_ * length_);
                    std::iota(order_.begin(), order_.end(), int64_t{0});
                    std::shuffle(order_.begin(), order_.end(), random);
                }
                // 今回の区間だけを lane と時間の組に戻し、観測を取り出す。
                const auto first = epoch_step * batch_size_;
                batch_indices_.clear();
                batch_indices_.reserve(batch_size_);
                for (int64_t offset = 0; offset < batch_size_; ++offset) {
                    const auto index = order_.at(first + offset);
                    batch_indices_.push_back({.lane = index % lanes_, .time = index / lanes_});
                }
                const auto observations = source.Gather(batch_indices_);
                ANET_PROFILE_SCOPE_NEXT(intrinsic_learn);
                // 予定された step は消費し、optimizer が進んだ回数だけを返す。
                if (estimator.Learn(observations)) ++learned;
            }
            return learned;
        }
    private:
        const int64_t lanes_;
        const int64_t length_;
        const int64_t batch_size_;
        const int64_t total_steps_;
        const int64_t steps_per_epoch_;
        const bool spread_;
        anet::LaneTimeTensorStore collecting_;
        anet::LaneTimeTensorStore training_;
        std::vector<int64_t> order_;
        std::vector<anet::LaneTimeIndex> batch_indices_;
        int64_t next_step_ = 0;
        int64_t spread_remainder_ = 0;
    };
}

// Agent の内的報酬の State と処理をまとめ、設定は不変な値として保持する。
class DefaultDQNAgent::IntrinsicRewardCoordinator {
    using IntrinsicConfig = decltype(DefaultDQNAgentConfig::intrinsic);
public:
    IntrinsicRewardCoordinator(const IntrinsicConfig& config, bool learning_enabled,
        const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, const torch::Device& device,
        anet::SeedMaker& seed_maker, const std::string& prefix, const ConfigFullKeyTable& keys,
        float reward_gamma, const std::string& gamma_key)
        : config_(config), learning_enabled_(learning_enabled)
    {
        if (config_.enabled)
            intrinsic_learn_random_.emplace(seed_maker.MakeNamedSeed(kIntrinsicLearnSeed));

        // 鍵は生成側が宣言し、部品を作らない場合も既知の未成立値を保つ。
        for (const auto& [key, value] : anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::InitialDiagnostics())
            intrinsic_diagnostics_.try_emplace(std::string(kIntrinsicPrefix) + key, value);
        for (const auto* key : {kIntrinsicBonusMean, kIntrinsicBonusMax, kIntrinsicBonusShare,
            kIntrinsicBonusMin, kIntrinsicBonusPosRatio, kIntrinsicLearnCount})
            intrinsic_diagnostics_.try_emplace(key, std::numeric_limits<float>::quiet_NaN());

        // 内的報酬の推定器を初期化
        if (config_.enabled) {
            reward_scaler_ = RewardScalerFactory(config_.reward_scaler,
                prefix.substr(0, prefix.size() - std::string(".rnd").size()) + ".reward_scaler")
                .CreateRewardScaler(reward_gamma, batch_env_spec.num_envs, gamma_key);
            if (learning_enabled_) {
                if (config_.learn.mode == DefaultDQNAgentConfig::kIntrinsicLearnRollout) {
                    const auto& rollout = config_.learn.rollout;
                    const auto steps = ValidateRolloutSteps(batch_env_spec.num_envs, keys);
                    rollout_.emplace(batch_env_spec.num_envs, rollout.length, rollout.batch_size,
                        steps, rollout.epochs, rollout.use_spread);
                } else if (batch_env_spec.num_envs < config_.learn.intake.batch_size) {
                    ANET_SYSTEM_ERROR(keys.GetFullKey("intrinsic.learn.intake.batch_size") << "=" << config_.learn.intake.batch_size
                        << " exceeds num_envs=" << batch_env_spec.num_envs
                        << "; expected " << keys.GetFullKey("intrinsic.learn.intake.batch_size") << " <= run.train.num_envs when "
                        << keys.GetFullKey("learner.enabled") << "=true.");
                }
            }
            lifelong_novelty_ = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
                config_.rnd, prefix,
                env_spec.state_spec.obs_spec, device, seed_maker);
            intrinsic_beta_.emplace(config_.beta);
            // 学習の入口を通らない評価専用 Run にも、現在の Run の開始時点の β を用意する。
            last_intrinsic_beta_ = intrinsic_beta_->Evaluate(0);
            intrinsic_diagnostics_.at(kIntrinsicLearnCount).store(0.0f, std::memory_order_relaxed);
            if (!learning_enabled_)
                LOG::info_once() << keys.GetFullKey("intrinsic.enabled") << "=true with "
                    << keys.GetFullKey("learner.enabled") << "=false: intrinsic scoring and training are suspended.";
        }
    }

    bool IsEnabled() const { return static_cast<bool>(lifelong_novelty_); }
    void RecordNetworkStructure(const anet::nn::NetworkGraphVizConfig& detail) const
    {
        if (!lifelong_novelty_) return;
        const auto logger = anet::MetricsLogger::Instance();
        for (const auto& [role, graph] : lifelong_novelty_->MakeNetworkGraphs({}))
            logger->Log(kIntrinsicRndGraphPrefix + role + kNetworkStructureSuffix, *graph);
        for (const auto& [role, graph] : lifelong_novelty_->MakeNetworkGraphs(detail))
            logger->Log(kIntrinsicRndGraphPrefix + role + kNetworkDetailSuffix, *graph);
    }
    std::string ScalerStateKind() const { return reward_scaler_ ? reward_scaler_->StateKind() : RewardScaler::kConstantState; }

    void AdvanceBeta(const StepCounts& counts)
    {
        // 学習停止中も Run の予定を進める。推定器と診断値の更新は学習有効時に限る。
        if (lifelong_novelty_)
            last_intrinsic_beta_ = intrinsic_beta_->Evaluate(counts.exp_step);
    }

    bool IsLearningEnabled() const { return lifelong_novelty_ && learning_enabled_; }

    float Beta() const { return last_intrinsic_beta_.value_or(0.0f); }

    torch::Tensor ScoreBonus(const BatchExperience& batch_exp, const torch::Tensor& scaled_rewards, torch::Tensor* component = nullptr)
    {
        ANET_PROFILE_FUNC();

        // 生の観測を選んで採点する。次観測だけは終端のボーナスをマスクする。
        ANET_PROFILE_SCOPE(estimate);
        const auto& intrinsic_observations = SelectObservations(batch_exp);
        const bool suppress_bonus = lifelong_novelty_->SuppressesBonus();
        const auto alpha = lifelong_novelty_->Estimate(intrinsic_observations);
        ANET_PROFILE_SCOPE_NEXT(bonus);
        const auto unscaled = CalculateBonus(alpha, 1.0f, batch_exp.next_state.done,
            UsesDoneMask(), BonusMaskOperation::Copy);
        auto scaled = reward_scaler_->Scale(unscaled);
        if (suppress_bonus) scaled = torch::zeros_like(scaled);
        auto bonus = *last_intrinsic_beta_ * scaled;
        if (component) *component = scaled;

        // ボーナスの診断値を公開する。比率の分母には加算前の外的報酬を使う。
        ANET_PROFILE_SCOPE_NEXT(diagnostics);
        const float bonus_sum = bonus.abs().sum().item<float>();
        const float denominator = scaled_rewards.abs().sum().item<float>() + bonus_sum;
        intrinsic_diagnostics_.at(kIntrinsicBonusMean).store(bonus.mean().item<float>(), std::memory_order_relaxed);
        intrinsic_diagnostics_.at(kIntrinsicBonusMax).store(bonus.max().item<float>(), std::memory_order_relaxed);
        intrinsic_diagnostics_.at(kIntrinsicBonusMin).store(bonus.min().item<float>(), std::memory_order_relaxed);
        intrinsic_diagnostics_.at(kIntrinsicBonusPosRatio).store((bonus > 0.0f).to(torch::kFloat32).mean().item<float>(),
            std::memory_order_relaxed);
        intrinsic_diagnostics_.at(kIntrinsicBonusShare).store(denominator > 0.0f ? bonus_sum / denominator
            : std::numeric_limits<float>::quiet_NaN(), std::memory_order_relaxed);
        return bonus;
    }

    void Learn(const BatchExperience& batch_exp)
    {
        if (!lifelong_novelty_ || !learning_enabled_) return;
        const auto& intrinsic_observations = SelectObservations(batch_exp);
        if (rollout_)
            intrinsic_learn_count_ += rollout_->AppendAndLearn(intrinsic_observations, *lifelong_novelty_, *intrinsic_learn_random_);
        else
            LearnIntake(intrinsic_observations);

        // 補助学習後の推定器の診断値を公開する。
        intrinsic_diagnostics_.at(kIntrinsicLearnCount).store(static_cast<float>(intrinsic_learn_count_), std::memory_order_relaxed);
        for (const auto& [key, value] : lifelong_novelty_->Diagnostics())
            intrinsic_diagnostics_.at(std::string(kIntrinsicPrefix) + key).store(value, std::memory_order_relaxed);
    }

    void LearnIntake(const anet::TensorDict& intrinsic_observations)
    {
        // 経験件数に比例したcreditを貯め、同じintakeのlaneを重複なくk件ずつ渡す。
        const auto num_lanes = intrinsic_observations.Size(0);
        const auto batch_size = config_.learn.intake.batch_size;
        intrinsic_learn_credit_ += static_cast<float>(num_lanes) * config_.learn.intake.ratio / batch_size;
        if (intrinsic_learn_credit_ >= 1.0f) {
            ANET_PROFILE_SCOPE(intrinsic_selection);
            std::vector<int64_t> lanes(num_lanes);
            std::iota(lanes.begin(), lanes.end(), 0);
            std::shuffle(lanes.begin(), lanes.end(), *intrinsic_learn_random_);
            auto indices = torch::tensor(lanes, torch::kInt64);
            // indexのdevice転送はstepループの前にまとめ、CUDAにはpinnedから非同期で送る。
            if (intrinsic_observations.device().is_cuda())
                indices = indices.pin_memory().to(intrinsic_observations.device(), true);
            for (int64_t offset = 0; offset + batch_size <= num_lanes && intrinsic_learn_credit_ >= 1.0f;
                offset += batch_size) {
                ANET_PROFILE_SCOPE(intrinsic_learn);
                anet::TensorDict observations;
                const auto selected = indices.narrow(0, offset, batch_size);
                for (const auto& [key, value] : intrinsic_observations)
                    observations.Set(key, value.index_select(0, selected));
                if (lifelong_novelty_->Learn(observations)) ++intrinsic_learn_count_;
                // warm-upの未実行分も消費し、明けたときに追いつき学習をしない。
                intrinsic_learn_credit_ -= 1.0f;
            }
        }

    }

    int64_t Save(anet::OutputArchive& archive) const
    {
        int64_t total_size = 0;
        // RND本体に続けて、Coordinatorのcredit・lane抽選の乱数状態・累計成功回数を保存する。
        // 読込時にRND本体の書式検査で異なる形式を拒否できるよう、本体を先に置く。
        if (lifelong_novelty_) {
            total_size += lifelong_novelty_->Save(archive);
            total_size += archive.Write(intrinsic_learn_credit_);
            std::ostringstream random_state;
            random_state << *intrinsic_learn_random_;
            total_size += archive.Write(random_state.str());
            total_size += archive.Write(intrinsic_learn_count_);
            total_size += reward_scaler_->Save(archive);
        }
        return total_size;
    }

    void Load(anet::InputArchive& in)
    {
        // RND本体に続けて、Coordinatorのcredit・lane抽選の乱数状態・累計成功回数を復元し、診断へ反映する。
        // 異なる形式をCoordinatorの状態として読む前に拒否するため、RND本体の書式検査を先に行う。
        if (lifelong_novelty_) {
            lifelong_novelty_->Load(in);
            in.Read(intrinsic_learn_credit_);
            if (!std::isfinite(intrinsic_learn_credit_) || intrinsic_learn_credit_ < 0.0f)
                ANET_SYSTEM_ERROR("Invalid intrinsic learning credit=" << intrinsic_learn_credit_ << "; expected finite value >= 0.");
            std::string random_state;
            in.Read(random_state);
            std::istringstream random_stream(random_state);
            if (!(random_stream >> *intrinsic_learn_random_))
                ANET_SYSTEM_ERROR("Invalid intrinsic learning RNG state in checkpoint; expected mt19937_64 state.");
            in.Read(intrinsic_learn_count_);
            if (intrinsic_learn_count_ < 0)
                ANET_SYSTEM_ERROR("Invalid intrinsic learn_count=" << intrinsic_learn_count_ << "; expected value >= 0.");
            intrinsic_diagnostics_.at(kIntrinsicLearnCount).store(
                static_cast<float>(intrinsic_learn_count_), std::memory_order_relaxed);
            for (const auto& [key, value] : lifelong_novelty_->Diagnostics())
                intrinsic_diagnostics_.at(std::string(kIntrinsicPrefix) + key).store(value, std::memory_order_relaxed);
            reward_scaler_->Load(in);
            if (rollout_) rollout_->Reset();
        }
    }

    std::optional<float> GetScalar(const std::string& key) const
    {
        if (key == kIntrinsicRewardScale)
            return reward_scaler_ ? reward_scaler_->GetScalar(RewardScaler::kKeyScale) : std::optional<float>(std::numeric_limits<float>::quiet_NaN());
        const auto it = intrinsic_diagnostics_.find(key);
        if (it == intrinsic_diagnostics_.end()) return std::nullopt;
        if (!config_.enabled || !learning_enabled_) return std::numeric_limits<float>::quiet_NaN();
        return it->second.load(std::memory_order_relaxed);
    }

    static std::unique_ptr<anet::rl::AgentStepDiagnosticsProvider> CreateStepDiagnosticsProvider(
        const AgentStepDiagnosticsRequest& request,
        std::unique_ptr<anet::rl::AgentStepDiagnosticsProvider> base,
        std::shared_ptr<std::shared_mutex> mutex,
        std::shared_ptr<const IntrinsicRewardCoordinator> self);
private:
    int64_t ValidateRolloutSteps(int64_t lanes, const ConfigFullKeyTable& keys) const
    {
        // lane 数から、rollout 全体で必要になる実数の step 数を求める。
        const auto& rollout = config_.learn.rollout;
        const auto observations = lanes * rollout.length;
        const double steps = static_cast<double>(observations) * rollout.ratio / rollout.batch_size;
        const double nearest = std::round(steps);
        constexpr double kRelativeTolerance = 1e-6;
        const auto describe = [&] {
            std::ostringstream text;
            text << "num_envs=" << lanes << " * " << keys.GetFullKey("intrinsic.learn.rollout.length") << "=" << rollout.length
                << " * " << keys.GetFullKey("intrinsic.learn.rollout.ratio") << "=" << rollout.ratio
                << " / " << keys.GetFullKey("intrinsic.learn.rollout.batch_size") << "=" << rollout.batch_size
                << " gives " << steps << " step(s) per rollout";
            return text.str();
        };
        // 許容誤差内の正の整数に限って、実行する step 数として使う。
        if (nearest < 1.0 || nearest >= static_cast<double>(std::numeric_limits<int64_t>::max())
            || std::abs(steps - nearest) > steps * kRelativeTolerance)
            ANET_SYSTEM_ERROR(describe() << "; expected a positive integer step count representable by int64_t.");
        const auto count = static_cast<int64_t>(nearest);
        // エポックを均等に分け、各エポックで同じ観測を繰り返さずに済むことを確かめる。
        if (count % rollout.epochs != 0 || count / rollout.epochs > observations / rollout.batch_size)
            ANET_SYSTEM_ERROR(describe() << "; expected divisibility by "
                << keys.GetFullKey("intrinsic.learn.rollout.epochs") << "=" << rollout.epochs
                << " and at most " << observations << " observations per epoch.");
        return count;
    }

    bool UsesNextState() const
    {
        return config_.observation == DefaultDQNAgentConfig::kIntrinsicObservationNextState;
    }

    bool UsesDoneMask() const { return UsesNextState() && config_.use_done_mask; }

    const anet::TensorDict& SelectObservations(const BatchExperience& experience) const
    {
        return UsesNextState() ? experience.next_state.obs : experience.state.obs;
    }

    enum class BonusMaskOperation { Copy, InPlace };

    static torch::Tensor CalculateBonus(const torch::Tensor& multipliers, float beta,
        const torch::Tensor& done, bool mask, BonusMaskOperation operation)
    {
        auto bonus = beta * (multipliers - 1.0f);
        if (mask) {
            // 学習側の複製と Trace 側の書き換えを保ち、式と mask の判定を共有する。
            if (operation == BonusMaskOperation::Copy) bonus = bonus.masked_fill(done, 0.0f);
            else bonus.masked_fill_(done, 0.0f);
        }
        return bonus;
    }

    std::map<std::string, torch::Tensor> ScoreTrace(intrinsic::LifelongNoveltyEstimator& estimator,
        std::optional<float> beta, const BatchExperience& experience, torch::Tensor* component = nullptr) const
    {
        auto result = estimator.EstimateWithoutUpdate(SelectObservations(experience));
        std::map<std::string, torch::Tensor> values;
        for (auto& [key, value] : result.diagnostics)
            values.emplace(std::string(kIntrinsicPrefix) + key, std::move(value));
        auto bonus = torch::full_like(result.multipliers, std::numeric_limits<float>::quiet_NaN());
        if (beta) {
            const bool mask = UsesDoneMask();
            const auto raw = CalculateBonus(result.multipliers, 1.0f,
                mask ? experience.next_state.done.to(torch::kCPU) : torch::Tensor{}, mask, BonusMaskOperation::InPlace);
            auto scaled = reward_scaler_->ScaleWithoutUpdate(raw);
            if (estimator.SuppressesBonus()) scaled = torch::zeros_like(scaled);
            if (component) *component = scaled;
            bonus = *beta * scaled;
            bonus.masked_fill_(~torch::isfinite(values.at(kIntrinsicError)), std::numeric_limits<float>::quiet_NaN());
        }
        values.emplace(kIntrinsicBonus, std::move(bonus));
        return values;
    }

    const IntrinsicConfig config_;
    const bool learning_enabled_;
    std::unique_ptr<anet::rl::intrinsic::LifelongNoveltyEstimator> lifelong_novelty_;
    std::unique_ptr<RewardScaler> reward_scaler_;
    std::optional<anet::ProfiledValue<float>> intrinsic_beta_;
    std::optional<float> last_intrinsic_beta_; ///< 構築時は exp step 0、以後は学習の入口が直近に評価した値。checkpointには保存しない。
    float intrinsic_learn_credit_ = 0.0f; ///< Coordinatorが更新する補助学習の未消費credit。
    int64_t intrinsic_learn_count_ = 0; ///< optimizerが実際に進んだ累計step数。
    std::optional<std::mt19937_64> intrinsic_learn_random_; ///< 有効時だけ作るlane抽選のResource。
    std::optional<IntrinsicLearnRolloutBatcher> rollout_;
    std::map<std::string, std::atomic<float>> intrinsic_diagnostics_; ///< キーは構築時に固定。UI は値だけ読む。
};

void DefaultDQNAgent::IntrinsicRewardCoordinatorDeleter::operator()(IntrinsicRewardCoordinator* value) const
{
    delete value;
}

// ======================================================
// DefaultDQNAgent
// ======================================================

DefaultDQNAgent::DefaultDQNAgent(
    const DefaultDQNAgentConfig& config
    , const anet::nn::NetworkConfig& net_config
    , const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, const torch::Device device
    , std::optional<seed_t> seed)
    : AgentBase(device, batch_env_spec, env_spec, seed)
    , config_(config)
{
    // 直接DTOを渡す構築経路でも、有効化時は未指定のJを資源構築前に拒否する。
    if (config_.skip.enabled && config_.skip.max_length < 1) {
        ANET_SYSTEM_ERROR("Invalid " << config_.GetFullKey("skip.max_length") << "=" << config_.skip.max_length
            << "; expected explicit value >= 1 when " << config_.GetFullKey("skip.enabled") << "=true.");
    }
    ANET_LOG_DEBUG("seed=" << GetSeed());

    // ログ：パラメータ記録
    LOG::info() << "DefaultDQNAgent config=" << config_.ToString();
    anet::MetricsLogger::Instance()->Log(config_);
    anet::MetricsLogger::Instance()->Log("net.body", net_config.ToJson());

	// 割引率gammaと実効ホライズンのログ
    {
        // 実効ホライズン (Effective Horizon) = 1 / (1 - gamma)
        // gamma=0.9 -> 10 steps, 0.99 -> 100 steps, 0.995 -> 200 steps
        float g = anet::ProfiledValue<float>(config_.learner.gamma).Evaluate(0);
        float horizon = (std::abs(g - 1.0f) < 1e-6) ? -1.0f : (1.0f / (1.0f - g));

        //時系列の重みが約37%に減衰するまでのSTEP数
        if (horizon > 0) {
            LOG::info() << GetInitialProfiledValueKey(config_.learner.gamma) << "=" << g
                << " (EffectiveHorizon: " << static_cast<int>(horizon) << " steps)";
        } else {
            LOG::info() << GetInitialProfiledValueKey(config_.learner.gamma) << "=" << g << " (Effective Horizon: Infinite)";
        }
    }

    // seed
    anet::SeedMaker seed_maker(GetSeed());
    auto replay_seed = seed_maker.MakeNamedSeed("replaybuffer");
    auto learner_seed = seed_maker.MakeNamedSeed("learner");
    auto network_seed = seed_maker.MakeNamedSeed("network");
    const bool is_distributional = config_.quantile_mode != ActionPolicyConfig::kQuantileModeNone;

    // 無効時も既知の診断キーを保つ。可変の State は Coordinator に集める。
    const auto& intrinsic_gamma = config_.learner.GetIntrinsicGammaConfig();
    const float initial_intrinsic_gamma = anet::ProfiledValue<float>(intrinsic_gamma).Evaluate(0);
    intrinsic_.reset(new IntrinsicRewardCoordinator(config_.intrinsic, config_.learner.enabled,
        batch_env_spec, env_spec, device_, seed_maker, config_.GetConfigPrefix() + ".intrinsic.rnd",
        config_.GetFullKeys(), initial_intrinsic_gamma, GetInitialProfiledValueKey(intrinsic_gamma)));

    // RewardScaler生成
    anet::rl::RewardScalerFactory reward_scaler_factory(config_.reward_scaler);
    const auto& gamma = config_.learner.gamma;
    this->reward_scaler_ = reward_scaler_factory.CreateRewardScaler(anet::ProfiledValue<float>(gamma).Evaluate(0), batch_env_spec.num_envs,
        GetInitialProfiledValueKey(gamma));

    // Frame の spec と重ね方を、ReplayBuffer と Forwarder の依頼へ渡す。
    const auto frame_env_spec = env_spec;
    const auto& frame_specs = frame_env_spec.state_spec.obs_spec;
    replay_frame_stack_ = ResolveReplayFrameStack(config_, frame_specs);
    perception_ = std::make_unique<anet::ForwarderBuilder>(
        config_.perception_config, config_.GetConfigPrefix() + ".perception",
        seed_maker.MakeNamedSeed(kPerceptionSeed));
    // 取り込みの形と dtype を確認し、可視化の出力 spec を NN の基準にする。
    frame_intake_ = perception_->BuildForwarder(MakeFrameWithStatsUpdateRequest(), frame_specs);
    RequirePerceptionSpecs(frame_specs, frame_intake_->GetOutputSpecs(), kForwardKindFrameWithStatsUpdate);
    const auto raw_visualization = perception_->BuildForwarder(
        MakePerceptionFromFrameRequest(replay_frame_stack_.count, replay_frame_stack_.keys), frame_specs);
    visualization_perception_ = CompletePerception(raw_visualization, replay_frame_stack_.keys);
    // Actor が同じ spec を返すか、履歴を動かす前に照合する。
    const auto probe_actor = BuildActorPerception(*perception_, frame_specs, replay_frame_stack_, 1,
        seed_maker.MakeNamedSeed(kPerceptionSeed));
    RequirePerceptionSpecs(visualization_perception_->GetOutputSpecs(), probe_actor->GetOutputSpecs(),
        kForwardKindPerceptionFromSequence);


    // ------------------------------------------------------------
    // HeadFactory の準備
    // ------------------------------------------------------------

    auto head_factory = SelectHeadFactory();

    // DTO にある設定データから、用途が参照する不変 Resource を構築する。
    for (const auto& [key, data] : config_.nn_forward) {
        forward_configs_.emplace(key, std::make_shared<const anet::nn::NetworkForwardConfig>(
            config_.MakeForwardConfigKey(key), data));
    }
    auto network_obs_spec = visualization_perception_->GetOutputSpecs();
    // IQN の taus は Perception に含めず、Network の入力 spec に足す。
    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn) {
        anet::TensorSpec tau_spec;
        tau_spec.type = anet::SpaceType::Vector;
        tau_spec.shape = { config_.learner.iqn.current_taus.num_taus };
        tau_spec.dtype = torch::kFloat32;
        tau_spec.num_classes = 0;
        network_obs_spec[anet::rl::dqn::kKey_Taus] = tau_spec;
    }


    // NetworkModel生成
    std::optional<SkipNetworkDefinition> skip_definition;
    if (config_.skip.enabled) {
        skip_definition.emplace(SkipNetworkDefinition{
            .config = config_.skip,
            .network_config = anet::nn::NetworkConfig(config_.skip_net, "DefaultDQNAgent.skip.net"),
            .head_factory = std::make_shared<SkipHeadFactory>(n_actions_, config_.skip.max_length,
                config_.skip.network_mode, config_.quantile_mode, config_.qr.num_quantiles,
                config_.skip.head.init, config_.skip.head.projection),
            .seed = seed_maker.MakeNamedSeed(kSkipNetworkSeed)});
    }
    this->model_ = std::make_unique<NetworkModel>(
        config_.model, device_,
        net_config, network_obs_spec, n_actions_, head_factory,
        is_distributional, network_seed, skip_definition ? &*skip_definition : nullptr
    );

    // target用のランダム探索は全有効端点で禁止する。
    if (config_.target_policy.policy_type == "EpsilonGreedy") {
        auto epsilon = config_.target_policy.epsilon;
        epsilon.max_value = 0.0f;
        anet::detail::ValidateProfiledValueConfig(epsilon);
    }

    // ActionPolicy生成
    this->target_policy_ = DqnComponentAssembler::CreateActionPolicy(config_.target_policy, config_.quantile_mode, false, num_envs_, device_);

    InitializeLearner(batch_env_spec, frame_env_spec, replay_seed, learner_seed);
    RecordNetworkStructure(*raw_visualization);

    // load
    if (!config_.auto_load_file.empty()) {
        LOG::info() << "Auto-loading network from file: " << config_.auto_load_file;
        LoadNetwork(config_.auto_load_file);
	}
}

std::shared_ptr<anet::nn::NetworkHeadFactory> DefaultDQNAgent::SelectHeadFactory() const
{
    auto extrinsic = DqnComponentAssembler::CreateHeadFactory(config_.quantile_mode, config_.head.use_dueling,
        n_actions_, config_.qr.num_quantiles, config_.head.init, config_.head.projection);
    if (!config_.learner.intrinsic.decomposed) return extrinsic;
    const auto& head = config_.intrinsic.value.head;
    auto intrinsic = DqnComponentAssembler::CreateHeadFactory(config_.quantile_mode, head.use_dueling,
        n_actions_, config_.qr.num_quantiles, head.init, head.projection);
    return std::make_shared<DecomposedHeadFactory>(std::move(extrinsic), std::move(intrinsic),
        head.input_mode == DefaultDQNAgentConfig::kHeadInputDedicated, head.use_dueling, config_.GetFullKey("intrinsic.value.head.input_mode"));
}

void DefaultDQNAgent::RecordNetworkStructure(const anet::SpecForwarder& visualization) const
{
    auto structure = visualization.MakeGraphViz(false);
    auto detail = visualization.MakeGraphViz(true);
    if (!structure || !detail) {
        ANET_SYSTEM_ERROR(kForwardKindPerceptionFromFrame << ": expected both perception diagrams.");
    }
    anet::MetricsLogger::Instance()->Log(kPerceptionStructureTag, *structure);
    anet::MetricsLogger::Instance()->Log(kPerceptionDetailTag, *detail);
    // Network グラフ可視化
    {
        const auto& net = *model_->GetOnlineNetwork();
        auto structure_view = net.MakeGraphViz(anet::nn::NetworkGraphVizConfig{});
        anet::MetricsLogger::Instance()->Log("net.structure", *structure_view);
        auto detail_view = net.MakeGraphViz(config_.nn_viz);
        anet::MetricsLogger::Instance()->Log("net.detail", *detail_view);
        if (const auto skip = model_->GetSkipOnlineNetwork()) {
            anet::MetricsLogger::Instance()->Log(kSkipStructureTag, *skip->MakeGraphViz(anet::nn::NetworkGraphVizConfig{}));
            anet::MetricsLogger::Instance()->Log(kSkipDetailTag, *skip->MakeGraphViz(config_.nn_viz));
        }
        intrinsic_->RecordNetworkStructure(config_.nn_viz);
    }

}

void DefaultDQNAgent::InitializeLearner(const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec,
    anet::seed_t replay_seed, anet::seed_t learner_seed)
{
    // 解決済みのbootstrap方策をログへ示し、平均と経験分位のsoft targetを区別する。
    const auto risk_score = target_policy_->GetRiskScoreSpec();
    std::ostringstream munchausen_info;
    munchausen_info << " munchausen=" << (config_.learner.munchausen.enabled ? "on" : "off")
        << " log_policy_mode=" << config_.learner.munchausen.log_policy_mode
        << " score_source=" << (risk_score ? "risk_biased" : "mean");
    if (risk_score) munchausen_info << " risk_tau=" << risk_score->tau << " use_tail_mean=" << risk_score->use_tail_mean;

    // Noisy parameter がある Network にだけ、probe の固定用途を構築時に準備する。
    std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe_config;
    if (!model_->GetOnlineNetwork()->GetNoisyParameterEntries().empty()) {
        auto data = forward_configs_.at(config_.learner.current.nn_forward_key)->GetConfigData();
        data.Set("noisy.mode", "sample");
        data.Set("noisy.hold", "call");
        data.Set("noisy.share", "batch");
        noisy_probe_config = std::make_shared<const anet::nn::NetworkForwardConfig>(
            MakeNetworkForwardConfigKey("noisy_probe"), data);
    }

    // Learner は Replay の Sample spec で組み立て、可視化と同じ出力か照合する。
    const LearnerPerceptionSetup perception_setup{
        .builder = *perception_,
        .expected_specs = visualization_perception_->GetOutputSpecs(),
    };
    learner_ = DqnComponentAssembler::CreateLearner(config_.quantile_mode, config_.qr.num_quantiles, config_.learner,
        *model_, batch_env_spec, env_spec, device_, replay_seed, GetSeed(), target_policy_,
        &perception_setup, LearnerForwardResources{
            .current = *forward_configs_.at(config_.learner.current.nn_forward_key),
            .target = *forward_configs_.at(config_.learner.target.nn_forward_key)},
        replay_frame_stack_,
        learner_seed, noisy_probe_config, munchausen_info.str());
}

int64_t DefaultDQNAgent::Save(anet::OutputArchive& archive) const
{
    ANET_PROFILE_FUNC();
	std::shared_lock<std::shared_mutex> lock(*mutex_);

	int64_t total_size = 0;

    // ヘッダ
    anet::ArchiveHeader header;
    header.info = config_.skip.enabled ? kCheckpointFormatV2 : kCheckpointFormatV1;
    if (intrinsic_->IsEnabled())
        header.info = config_.skip.enabled ? kCheckpointIntrinsicV2 : kCheckpointIntrinsicV1;
    const bool has_scaler_state = reward_scaler_->StateKind() != RewardScaler::kConstantState;
    const bool extended = has_scaler_state || config_.learner.intrinsic.decomposed || intrinsic_->ScalerStateKind() != RewardScaler::kConstantState;
    if (extended) header.info = intrinsic_->IsEnabled() ? kCheckpointIntrinsicV3 : kCheckpointFormatV3;
    auto header_size = archive.Write(header);
	total_size += header_size;

    // Config
    auto config_size = archive.Write(this->config_.ToString());
	total_size += config_size;
    if (extended) {
        // Networkを読む前に検査できる構成の契約を、payloadから独立して保存する。
        total_size += archive.Write(config_.skip.enabled);
        total_size += archive.Write(config_.learner.intrinsic.decomposed);
        total_size += archive.Write(config_.learner.intrinsic.feature_gradient);
        total_size += archive.Write(reward_scaler_->StateKind());
        total_size += archive.Write(intrinsic_->ScalerStateKind());
    }

    /// @todo state_dim_やn_actions_の永続化対応(NN整合チェックで必要)
    /// @todo ObservationNormalizerの永続化対応

    // Network(policy_net/target_net)
	auto model_size = model_->Save(archive);
	total_size += model_size;

    // Learner(Adam)
    auto adam_size = learner_->Save(archive);
	total_size += adam_size;

    total_size += intrinsic_->Save(archive);
    if (extended) total_size += reward_scaler_->Save(archive);

    // ログ
    LOG::info() << "DefaultDQNAgent Serialized. total_size=" << anet::FormatWithCommas(total_size)
        << " config_size=" << anet::FormatWithCommas(config_size)
        << " adam_size=" << anet::FormatWithCommas(adam_size)
        << " model_size=" << anet::FormatWithCommas(model_size);

    return total_size;
}

void DefaultDQNAgent::LoadNetwork(const std::string& filename)
{
	std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        LOG::info() << "cwd=" << std::filesystem::current_path();
        ANET_SYSTEM_ERROR("Failed to open file for loading: " << filename);
	}
	anet::InputArchive in(ifs);

    // Header
    anet::ArchiveHeader header;
	in.Read(header);
	LOG::verbose() << "LoadNetwork: Archive Header: " << header.kMagicWord << " " << header.kFormatVersion << " "  << header.info;

    // Config
	std::string config_str;
	auto config_len = in.Read(config_str);
    LOG::info() << "LoadNetwork: config: len=" << config_len;
    LOG::verbose() << "LoadNetwork: config: config_str=\n" << config_str;

    const bool extended = header.info == kCheckpointFormatV3 || header.info == kCheckpointIntrinsicV3;
    const bool has_intrinsic = header.info == kCheckpointIntrinsicV1 || header.info == kCheckpointIntrinsicV2 || header.info == kCheckpointIntrinsicV3;
    bool has_skip = header.info == kCheckpointFormatV2 || header.info == kCheckpointIntrinsicV2;
    if (header.info != kCheckpointFormatV1 && header.info != kCheckpointFormatV2 && !has_intrinsic && !extended) {
        ANET_SYSTEM_ERROR("Invalid DefaultDQN checkpoint header='" << header.info
            << "'; expected " << kCheckpointFormatV1 << ", " << kCheckpointFormatV2
            << ", " << kCheckpointIntrinsicV1 << ", " << kCheckpointIntrinsicV2
            << ", " << kCheckpointFormatV3 << " or " << kCheckpointIntrinsicV3 << ".");
    }
    bool decomposed = false;
    std::string feature_gradient = LearnerConfig::kFeatureGradientMerge;
    std::string external_kind = RewardScaler::kConstantState, internal_kind = RewardScaler::kConstantState;
    if (extended) {
        in.Read(has_skip);
        in.Read(decomposed);
        in.Read(feature_gradient);
        in.Read(external_kind);
        in.Read(internal_kind);
    }
    if (decomposed != config_.learner.intrinsic.decomposed)
        ANET_SYSTEM_ERROR("Checkpoint value_mode=" << (decomposed ? DefaultDQNAgentConfig::kValueModeDecomposed : DefaultDQNAgentConfig::kValueModeCombined)
            << "; expected " << config_.GetFullKey("intrinsic.value_mode") << "=" << config_.intrinsic.value_mode << ".");
    if (decomposed && feature_gradient != config_.learner.intrinsic.feature_gradient)
        ANET_SYSTEM_ERROR("Checkpoint feature_gradient=" << feature_gradient
            << "; expected " << config_.GetFullKey("learner.intrinsic.feature_gradient") << "=" << config_.learner.intrinsic.feature_gradient << ".");
    if (external_kind != reward_scaler_->StateKind())
        ANET_SYSTEM_ERROR("Checkpoint reward_scaler kind=" << external_kind << "; expected " << reward_scaler_->StateKind() << ".");
    if (internal_kind != intrinsic_->ScalerStateKind())
        ANET_SYSTEM_ERROR("Checkpoint intrinsic reward scaler kind=" << internal_kind << "; expected " << intrinsic_->ScalerStateKind() << ".");
    if (has_intrinsic != config_.intrinsic.enabled)
        ANET_SYSTEM_ERROR("Checkpoint header='" << header.info << "' intrinsic=" << has_intrinsic
            << "; expected " << config_.GetFullKey("intrinsic.enabled") << "=" << config_.intrinsic.enabled << ".");
    if (has_skip && !config_.skip.enabled) ANET_SYSTEM_ERROR("Checkpoint contains skip networks but " << config_.GetFullKey("skip.enabled") << "=false.");

	// Network
	auto model_size = model_->LoadCheckpoint(in, has_skip);
    LOG::info() << "LoadNetwork: model_size=" << model_size;

	// Learner
	auto learner_size = learner_->LoadCheckpoint(in, has_skip);

    intrinsic_->Load(in);
    if (extended) reward_scaler_->Load(in);
    LOG::info() << "LoadNetwork: learner_size=" << learner_size;
    if (config_.skip.enabled && !has_skip) {
        LOG::warn_once(filename) << "Checkpoint has no skip networks; restored primary networks and optimizer state, initialized skip networks and optimizer groups. file=" << filename;
    }
}

std::optional<anet::TensorDictFunction> DefaultDQNAgent::GetTensorDictFunction(const std::string& key)
{
    // NetworkModel に委譲してベース関数を取得
    auto fn = model_->GetTensorDictFunction(key, device_);
    if (fn == std::nullopt) return std::nullopt;

    auto self = shared_from_this();
    auto network_fn = *fn;
    auto perception = visualization_perception_;

    return [self, network_fn, perception](const anet::TensorDict& obs) {
        // 学習中の Network と統計の更新から読み出しを保護する。
        std::shared_lock<std::shared_mutex> lock(*self->mutex_);
        torch::NoGradGuard grad_guard;
        auto output = network_fn(perception->Forward(obs.To(self->device_)));
        if (self->config_.learner.intrinsic.decomposed)
            DecomposedValueCombiner::Combine(output, self->intrinsic_->Beta(), self->config_.learner.use_tbo, self->config_.learner.tbo_epsilon);
        return output;
    };
}

std::optional<float> DefaultDQNAgent::GetScalar(const std::string& key, int64_t index) const
{
    if (key.starts_with(kIntrinsicPrefix)) return intrinsic_->GetScalar(key);
    if (const auto value = DqnComponentAssembler::GetScalar(
        key, index, *learner_, *target_policy_, *mutex_)) return value;
    if (key.find(RewardScaler::kKeyPrefix) == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return reward_scaler_->GetScalar(key);
    }
    std::shared_lock<std::shared_mutex> lock(*mutex_);
    return perception_->GetScalar(key);
}

void DefaultDQNAgent::ConfigureScalarMetricSubscriptions(
    const std::vector<ScalarMetricSubscription>& subscriptions)
{
    std::unique_lock<std::shared_mutex> lock(*mutex_);
    learner_->ConfigureScalarMetricSubscriptions(subscriptions);
}

std::optional<torch::Tensor> DefaultDQNAgent::GetTensor(const std::string& key, int64_t index) const
{
    if (const auto value = DqnComponentAssembler::GetTensor(key, *learner_, *mutex_)) return value;
    if (key.find(RewardScaler::kKeyPrefix) == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return reward_scaler_->GetTensor(key);
    }
    std::shared_lock<std::shared_mutex> lock(*mutex_);
    return perception_->GetTensor(key);
}

std::optional<std::vector<torch::Tensor>> DefaultDQNAgent::GetTensorVector(const std::string& key, int64_t index) const
{
    return DqnComponentAssembler::GetTensorVector(key, *learner_, *mutex_);
}

std::shared_ptr<anet::rl::Actor> DefaultDQNAgent::CreateActor(const ActorRequest& request) const
{
    ANET_PROFILE_FUNC();
    env_spec_.CheckSameStateActionSpec(request.env_spec);
    const auto& cfg = FindActorConfig(config_.actor, request.actor_key, config_.GetConfigPrefix());
    if (!forward_configs_.contains(cfg.nn_forward_key)) {
        ANET_SYSTEM_ERROR("Missing resolved " << cfg.full_keys.GetFullKey("nn_forward_key") << "; expected a declared forward configuration.");
    }
    ValidateActorDevice(cfg.clone_model, request.device);
    auto policy = DqnComponentAssembler::CreateActionPolicy(cfg.policy, config_.quantile_mode, cfg.policy.spatial.enabled,
        request.batch_env_spec.num_envs, request.device);
    std::optional<SkipActorResources> skip;
    if (config_.skip.enabled) {
        auto skip_source = cfg.network == "target" ? model_->GetSkipTargetNetwork() : model_->GetSkipOnlineNetwork();
        // greedy評価は両方をgreedyに揃え、既定では同じPolicyの現在値を使う。
        const bool shared_policy = !cfg.has_skip_policy || cfg.policy.policy_type == kActionPolicyTypeStr_Greedy;
        // モードは設定の選択だけを分け、skip の専用 seed と時計は Actor に任せる。
        const auto& skip_forward_key = cfg.skip.nn_forward_mode == DQNActorConfig::kForwardDedicated
            ? *cfg.skip.nn_forward_key : cfg.nn_forward_key;
        skip.emplace(SkipActorResources{
            .config = config_.skip,
            .policy = shared_policy ? policy : DqnComponentAssembler::CreateActionPolicy(cfg.skip_policy, config_.quantile_mode, cfg.skip_policy.spatial.enabled,
                request.batch_env_spec.num_envs, request.device),
            .network = skip_source, .source = skip_source,
            .forward_config = forward_configs_.at(skip_forward_key),
            .use_amp = cfg.policy.use_amp, .use_amp_bf16 = cfg.policy.use_amp_bf16,
            .shared_policy = shared_policy, .max_length_mode = cfg.skip.max_length_mode,
            .max_length = cfg.skip.max_length, .actor_keys = cfg.full_keys});
    }
    const ActorQHintConfig hint_config{
        .munchausen = config_.learner.munchausen, .use_tbo = config_.learner.use_tbo,
        .tbo_epsilon = config_.learner.tbo_epsilon, .gamma = config_.learner.gamma,
        .decomposed = config_.learner.intrinsic.decomposed,
        .beta = config_.learner.intrinsic.decomposed && cfg.intrinsic.action_beta_mode == DQNActorConfig::kActionBetaFixed
            ? anet::ProfiledValueConfig<float>{.value = cfg.intrinsic.action_beta.value(), .min_value = 0.0f} : config_.intrinsic.beta,
        .extrinsic_bootstrap_mode = config_.learner.intrinsic.extrinsic_bootstrap_mode,
        .intrinsic_gamma = config_.learner.GetIntrinsicGammaConfig()};
    const anet::SeedMaker seed_maker(request.seed);
    auto perception = BuildActorPerception(*perception_, env_spec_.state_spec.obs_spec, replay_frame_stack_,
        request.batch_env_spec.num_envs,
        seed_maker.MakeNamedSeed(kPerceptionSeed));
    return DqnComponentAssembler::CreateActor(cfg, config_.learner, request, *model_, mutex_, policy,
        *forward_configs_.at(cfg.nn_forward_key), std::move(perception),
        true, hint_config, cfg.clone_model ? cfg.sync_interval : std::nullopt, std::move(skip));
}

namespace {
    class IntrinsicSnapshot final : public anet::rl::ActorSyncFollower {
    public:
        using Estimator = anet::rl::intrinsic::LifelongNoveltyEstimator;
        using Capture = std::function<std::pair<std::unique_ptr<Estimator>, std::optional<float>>() >;
        explicit IntrinsicSnapshot(Capture capture) : capture_(std::move(capture)) {}
        void OnActorSync() override
        {
            ANET_PROFILE_FUNC();
            // Actor が既に取った source の読み取りロックの中でだけ写す。
            if (active_) {
                auto captured = capture_();
                estimator = std::move(captured.first);
                beta = captured.second;
            } else {
                estimator.reset();
                beta.reset();
            }
        }
        void Suspend() { active_ = false; }
        bool Resume() { active_ = true; return !estimator; }
        std::unique_ptr<Estimator> estimator;
        std::optional<float> beta;
    private:
        Capture capture_;
        bool active_ = true;
    };

    class IntrinsicStepDiagnosticsValues final : public anet::ModuleBase {
    public:
        IntrinsicStepDiagnosticsValues(std::unique_ptr<const anet::Module> base,
            std::map<std::string, torch::Tensor> values,
            std::shared_ptr<const std::vector<std::string>> keys, int64_t num_lanes)
            : base_(std::move(base)), values_(std::move(values)), keys_(std::move(keys)), num_lanes_(num_lanes) {}
        std::optional<float> GetScalar(const std::string& key, int64_t lane = -1) const override
        {
            const auto found = values_.find(key);
            if (found == values_.end()) {
                if (std::ranges::find(*keys_, key) == keys_->end())
                    return base_ ? base_->GetScalar(key, lane) : std::nullopt;
                ANET_CHECK_MSG(lane >= 0 && lane < num_lanes_,
                    "Invalid intrinsic diagnostic lane. key='" << key << "' lane=" << lane
                    << " expected=[0," << num_lanes_ << ").");
                return std::numeric_limits<float>::quiet_NaN();
            }
            ANET_CHECK_MSG(lane >= 0 && lane < found->second.size(0),
                "Invalid intrinsic diagnostic lane. key='" << key << "' lane=" << lane
                << " expected=[0," << found->second.size(0) << ").");
            return found->second[lane].item<float>();
        }
    private:
        std::unique_ptr<const anet::Module> base_;
        std::map<std::string, torch::Tensor> values_;
        std::shared_ptr<const std::vector<std::string>> keys_;
        int64_t num_lanes_;
    };

    class IntrinsicStepDiagnosticsProvider final : public anet::rl::AgentStepDiagnosticsProvider {
    public:
        using Score = std::function<std::map<std::string, torch::Tensor>(const anet::rl::BatchExperience&, torch::Tensor*)>;
        IntrinsicStepDiagnosticsProvider(std::unique_ptr<anet::rl::AgentStepDiagnosticsProvider> base,
            int64_t num_lanes, std::vector<std::string> keys, Score score,
            std::shared_ptr<IntrinsicSnapshot> snapshot, bool needs_component_rewards)
            : base_(std::move(base)), num_lanes_(num_lanes), keys_(std::make_shared<const std::vector<std::string>>(std::move(keys))),
              score_(std::move(score)), snapshot_(std::move(snapshot)), needs_component_rewards_(needs_component_rewards) {}
        anet::rl::AgentStepDiagnosticsResult Observe(const anet::rl::BatchExperience& experience) override
        {
            ANET_PROFILE_FUNC();
            // 既存の系列窓と遅延確定はそのまま使い、即時値だけを追加する。
            std::map<std::string, torch::Tensor> values;
            torch::Tensor component;
            if (score_) values = score_(experience, needs_component_rewards_ ? &component : nullptr);
            auto trace_experience = experience;
            if (needs_component_rewards_) {
                if (!component.defined()) component = torch::full_like(experience.reward, std::numeric_limits<float>::quiet_NaN());
                trace_experience.reward = torch::stack({experience.reward, component.to(experience.reward.device())}, 1);
            }
            auto result = base_->Observe(trace_experience);
            result.current = std::make_unique<IntrinsicStepDiagnosticsValues>(
                std::move(result.current), std::move(values), keys_, num_lanes_);
            return result;
        }
        void Reset() override { base_->Reset(); }
        void Suspend() override { if (snapshot_) snapshot_->Suspend(); }
        bool Resume() override { return snapshot_ && snapshot_->Resume(); }
    private:
        std::unique_ptr<anet::rl::AgentStepDiagnosticsProvider> base_;
        int64_t num_lanes_;
        std::shared_ptr<const std::vector<std::string>> keys_;
        Score score_;
        std::shared_ptr<IntrinsicSnapshot> snapshot_;
        bool needs_component_rewards_;
    };
} // namespace

std::unique_ptr<anet::rl::AgentStepDiagnosticsProvider> DefaultDQNAgent::CreateStepDiagnosticsProvider(
    const AgentStepDiagnosticsRequest& request) const
{
    auto self = shared_from_this();
    const bool needs_step_priority = !config_.learner.intrinsic.decomposed
        || std::ranges::find(request.keys, DqnComponentAssembler::kStepApproxPriorityKey) != request.keys.end();
    auto base = anet::rl::dqn::CreateStepDiagnosticsProvider(config_.learner, request.num_lanes,
        [self](const torch::Tensor& reward) {
            if (self->config_.learner.intrinsic.decomposed)
                return torch::stack({self->reward_scaler_->ScaleWithoutUpdate(reward.select(1, 0)), reward.select(1, 1)}, 1);
            return self->reward_scaler_->ScaleWithoutUpdate(reward);
        }, needs_step_priority);
    // Agent の所有する Coordinator を Agent と同じ寿命で借り、採点は Coordinator に委ねる。
    auto intrinsic = std::shared_ptr<const IntrinsicRewardCoordinator>(self, intrinsic_.get());
    return IntrinsicRewardCoordinator::CreateStepDiagnosticsProvider(
        request, std::move(base), mutex_, std::move(intrinsic));
}

std::unique_ptr<anet::rl::AgentStepDiagnosticsProvider> DefaultDQNAgent::IntrinsicRewardCoordinator::CreateStepDiagnosticsProvider(
    const AgentStepDiagnosticsRequest& request, std::unique_ptr<anet::rl::AgentStepDiagnosticsProvider> base,
    std::shared_ptr<std::shared_mutex> mutex, std::shared_ptr<const IntrinsicRewardCoordinator> self)
{
    // 既知の鍵と計算要求を分け、無効・未要求でも鍵はNaNとして認識する。
    std::vector<std::string> keys;
    for (const auto& key : intrinsic::LifelongNoveltyEstimatorFactory::LaneDiagnosticKeys())
        keys.push_back(std::string(kIntrinsicPrefix) + key);
    keys.emplace_back(kIntrinsicBonus);
    const auto requested = std::ranges::find_if(request.keys, [](const auto& key) { return key.starts_with(kIntrinsicPrefix); });
    IntrinsicStepDiagnosticsProvider::Score observe;
    std::shared_ptr<IntrinsicSnapshot> snapshot;
    const bool decomposed = self->config_.value_mode == DefaultDQNAgentConfig::kValueModeDecomposed;
    const bool needs_component_rewards = decomposed && std::ranges::find(request.keys, DqnComponentAssembler::kStepApproxPriorityKey) != request.keys.end();
    if (self->lifelong_novelty_ && (requested != request.keys.end() || needs_component_rewards)) {
        if (!request.actor)
            ANET_SYSTEM_ERROR("Step diagnostics key='" << (requested != request.keys.end() ? *requested : std::string(DqnComponentAssembler::kStepApproxPriorityKey))
                << "' requires an Actor; actual=null expected=Runner Actor.");
        snapshot = std::make_shared<IntrinsicSnapshot>([self] {
            return std::pair(self->lifelong_novelty_->Clone(), self->last_intrinsic_beta_);
        });
        if (!request.actor->AttachSyncFollower(snapshot)) snapshot.reset();
        observe = [self, snapshot, mutex](const BatchExperience& experience, torch::Tensor* component) {
            if (snapshot) {
                if (!snapshot->estimator) return std::map<std::string, torch::Tensor>{};
                return self->ScoreTrace(*snapshot->estimator, snapshot->beta, experience, component);
            }
            // 写しを持たないActorだけ、stepごとに学習側を読み取りロックで守る。
            std::shared_lock<std::shared_mutex> lock(*mutex);
            return self->ScoreTrace(*self->lifelong_novelty_, self->last_intrinsic_beta_, experience, component);
        };
    }
    return std::make_unique<IntrinsicStepDiagnosticsProvider>(std::move(base), request.num_lanes,
        std::move(keys), std::move(observe), std::move(snapshot), needs_component_rewards);
}

std::shared_ptr<anet::rl::Learner> DefaultDQNAgent::CreateLearner()
{
    return this->shared_from_this();
}

anet::rl::BatchUpdateResultList
DefaultDQNAgent::UpdateFromBatch(const StepCounts& counts, const anet::rl::BatchExperience& batch_exp)
{
    ANET_PROFILE_FUNC();

    BatchUpdateResultList result_list;

    {
        // 排他ロック（Writeロック)
        std::unique_lock<std::shared_mutex> lock(*mutex_);

        // RewardScaler
        auto scaled_rewards = this->reward_scaler_->Scale(batch_exp.reward);

        intrinsic_->AdvanceBeta(counts);
        torch::Tensor intrinsic_rewards;
        if (intrinsic_->IsLearningEnabled()) {
            ANET_PROFILE_SCOPE(intrinsic_reward);
            const auto bonus = intrinsic_->ScoreBonus(batch_exp, scaled_rewards,
                config_.learner.intrinsic.decomposed ? &intrinsic_rewards : nullptr);
            // 学習報酬へボーナスを加算する。
            scaled_rewards = config_.learner.intrinsic.decomposed ? torch::stack({scaled_rewards, intrinsic_rewards}, 1) : scaled_rewards + bonus;
            intrinsic_->Learn(batch_exp);
        }

        // 現在と次の観測を順に通し、統計に終端の観測も反映する。
        BatchExperience exp{
            batch_exp.state,
            batch_exp.action,
            scaled_rewards,
            batch_exp.next_state
        };
        exp.state.obs = frame_intake_->Forward(exp.state.obs);
        exp.next_state.obs = frame_intake_->Forward(exp.next_state.obs);

        // Update実行
        auto result = this->learner_->UpdateFromBatch(counts, exp, intrinsic_->Beta());
        result_list = std::move(result);

        // Update後処理
        target_policy_->UpdateSchedule(counts);
    }

    // BatchUpdateResultを返す
    return result_list;
}


// ======================================================
// DefaultDQNAgentFactory
// ======================================================

void DefaultDQNAgentFactory::ValidateQuantileNetworkContract(
    const DefaultDQNAgentConfig& config,
    const anet::nn::NetworkConfig& net_config,
    const std::string& net_config_prefix,
    const anet::ConfigData& config_data)
{
    // 各 branch の解析済み bind factor から、taus を直接要求する設定キーを列挙する。
    std::vector<std::string> taus_bind_keys;
    for (const auto& [branch_name, branch_config] : net_config.branches) {
        bool binds_taus = false;
        for (const auto& term : branch_config.bind_terms) {
            if (std::find(term.begin(), term.end(), anet::rl::dqn::kKey_Taus) != term.end()) {
                binds_taus = true;
                break;
            }
        }
        if (binds_taus) {
            taus_bind_keys.push_back(config_data.GetFullKey(net_config_prefix + ".branch.[" + branch_name + "].bind"));
        }
    }

    // quantile_mode と自分の net サブツリーとの局所契約を、モデル構築前に検証する。
    if (config.quantile_mode == ActionPolicyConfig::kQuantileModeIqn && taus_bind_keys.empty()) {
        ANET_SYSTEM_ERROR(
            "Invalid " << config.GetFullKey("quantile_mode") << "='" << config.quantile_mode
            << "': expected at least one 'taus' factor in "
            << net_config.config_prefix << ".branch.[*].bind.");
    }
    if (config.quantile_mode != ActionPolicyConfig::kQuantileModeIqn && !taus_bind_keys.empty()) {
        std::string joined_keys;
        for (const auto& key : taus_bind_keys) {
            if (!joined_keys.empty()) {
                joined_keys += ", ";
            }
            joined_keys += key;
        }
        ANET_SYSTEM_ERROR(
            "Invalid " << config.GetFullKey("quantile_mode") << "='" << config.quantile_mode
            << "': unexpected 'taus' factor in " << joined_keys << ".");
    }
}

std::shared_ptr<anet::rl::Agent> DefaultDQNAgentFactory::CreateAgent(
    const EnvSpec& env_spec, const BatchEnvSpec& batch_env_spec,
    const torch::Device& device, const anet::ConfigData& config_data,
    std::shared_ptr<anet::rl::Notifier> notifier, std::optional<anet::seed_t> seed) const
{
    DefaultDQNAgentConfig config(config_data);
    const auto net_config_prefix = GetTargetAgentClassId() + ".net";
    anet::nn::NetworkConfig net_config(config_data, net_config_prefix);
    ValidateQuantileNetworkContract(config, net_config, net_config_prefix, config_data);
    auto agent = std::make_shared<DefaultDQNAgent>(config, net_config, batch_env_spec, env_spec, device, seed);
    return agent;

    /// @todo 引数の順番を統一
}
