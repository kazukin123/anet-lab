#include "dqn_actor.hpp"
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
#include "dqn_heads.hpp"

using namespace anet::rl::dqn;
namespace LOG = anet::log;

constexpr const char* kActionQQuantilesAuxKey = "q_quantiles";

constexpr const char* kActionQValuesAuxKey = "q_values";

static constexpr char kValueSelectionDiagnostics[] = "value_selection_diagnostics";

static constexpr size_t kValueSelectionMetricCount = 4;

static constexpr std::array<const char*, kValueSelectionMetricCount> kValueSelectionLaneKeys{
    "extrinsic_q_argmax_disagreement", "extrinsic_q_regret", "extrinsic_q_gap", "intrinsic_q_gap"};

static constexpr std::array<const char*, kValueSelectionMetricCount> kValueSelectionMeanKeys{
    "extrinsic_q_argmax_disagreement_rate", "extrinsic_q_regret_mean", "extrinsic_q_gap_mean", "intrinsic_q_gap_mean"};

static constexpr std::array<const char*, kValueSelectionMetricCount> kValueSelectionSessionKeys{
    "extrinsic_q_session_argmax_disagreement_rate", "extrinsic_q_session_regret_mean", "extrinsic_q_session_gap_mean", "intrinsic_q_session_gap_mean"};

constexpr const char* kSkipForwardSeed = "skip_network_forward";

constexpr const char* kSkipPolicySeed = "skip_policy";

constexpr const char* kSkipSessionDecisionRate = "skip_session_decision_rate";

constexpr const char* kSkipMaxLength = "skip_max_length";

constexpr const char* kSkipSessionLengthMean = "skip_session_length_mean";

constexpr const char* kSkipSessionLengthRatePrefix = "skip_session_length_rate.";

constexpr const char* kSkipSessionLengthRateIndexPrefix = "skip_session_length_rate.[";

torch::Tensor anet::rl::dqn::PackActorQHint(
    const torch::Tensor& actor_q_sa, const torch::Tensor& actor_state_value, const torch::Tensor& munchausen_term,
    const torch::Tensor& intrinsic_q_sa, const torch::Tensor& intrinsic_state_value)
{
    // ActorとWithActionが同じ列順でpackできるよう、DQN payloadの構築を一元化する。
    if (!actor_q_sa.defined() || !actor_state_value.defined() || !munchausen_term.defined()) {
        ANET_SYSTEM_ERROR("PackActorQHint requires defined tensors.");
    }
    if (actor_q_sa.dim() != 1 || actor_state_value.dim() != 1 || munchausen_term.dim() != 1
        || actor_q_sa.sizes() != actor_state_value.sizes() || actor_q_sa.sizes() != munchausen_term.sizes()) {
        ANET_SYSTEM_ERROR("PackActorQHint expected matching rank-1 tensors. actor_q_sa="
            << actor_q_sa.sizes() << " actor_state_value=" << actor_state_value.sizes() << " munchausen_term=" << munchausen_term.sizes());
    }
    if (actor_q_sa.device() != actor_state_value.device() || actor_q_sa.device() != munchausen_term.device()) {
        ANET_SYSTEM_ERROR("PackActorQHint expected tensors on the same device. actor_q_sa="
            << actor_q_sa.device() << " actor_state_value=" << actor_state_value.device() << " munchausen_term=" << munchausen_term.device());
    }
    for (const auto& [name, value] : {std::pair{kStepIntrinsicQSaKey, intrinsic_q_sa},
        std::pair{kStepIntrinsicStateValueKey, intrinsic_state_value}}) {
        if (!value.defined() || value.sizes() != actor_q_sa.sizes() || value.device() != actor_q_sa.device()) {
            std::ostringstream actual;
            if (value.defined()) actual << "shape=" << value.sizes() << " device=" << value.device();
            else actual << "shape=undefined device=undefined";
            ANET_SYSTEM_ERROR("PackActorQHint requires matching intrinsic tensor. tensor=" << name
                << " " << actual.str() << " expected_shape=" << actor_q_sa.sizes()
                << " expected_device=" << actor_q_sa.device());
        }
    }
    return torch::stack({ actor_q_sa, actor_state_value, munchausen_term, intrinsic_q_sa, intrinsic_state_value }, 1)
        .to(torch::kFloat32).contiguous();
}

ActorQHintData anet::rl::dqn::MakeActorQHint(const torch::Tensor& q_values,
    const torch::Tensor& action, const ActorQHintValues& config)
{
    ANET_PROFILE_FUNC();
    // 既存Actorと同じgather/maxと平均Q近似を使い、forwardを追加しない。
    if (action.device() != q_values.device()) {
        ANET_SYSTEM_ERROR("actor_approx action and q_values must share a device. action="
            << action.device() << " q_values=" << q_values.device());
    }
    if (config.extrinsic_q.defined()) {
        // 分解経路ではcombined用のgather・方策・soft valueを重ねて計算しない。
        auto terms = torch::zeros_like(q_values);
        const auto real = [&](const torch::Tensor& value) {
            return config.use_tbo ? InverseDecomposedValue(value, config.tbo_epsilon) : value;
        };
        const auto transformed = [&](const torch::Tensor& value) {
            return config.use_tbo ? TransformH(value, config.tbo_epsilon) : value;
        };
        const auto gather = [&](const torch::Tensor& value, const torch::Tensor& indices) {
            return value.gather(1, indices.unsqueeze(1)).squeeze(1);
        };
        const auto external_real = real(config.extrinsic_q);
        const auto internal_real = real(config.intrinsic_q);
        const auto combined_real = real(config.combined_scores.defined() ? config.combined_scores : q_values);
        const auto external_scores = config.extrinsic_bootstrap
            ? real(config.extrinsic_scores.defined() ? config.extrinsic_scores : config.extrinsic_q) : combined_real;
        torch::Tensor external_value, internal_value;
        if (config.munchausen.enabled) {
            const auto external_log = MakeScaledLogPolicy(external_scores, config.munchausen.entropy_tau);
            const auto internal_log = MakeScaledLogPolicy(combined_real, config.munchausen.entropy_tau);
            external_value = ((external_log / config.munchausen.entropy_tau).exp() * (external_real - external_log)).sum(1);
            internal_value = ((internal_log / config.munchausen.entropy_tau).exp() * internal_real).sum(1);
            terms = config.munchausen.alpha * external_log.clamp(config.munchausen.clip_value_min, 0.0f);
        } else {
            external_value = gather(external_real, external_scores.argmax(1));
            internal_value = gather(internal_real, combined_real.argmax(1));
        }
        return {.payload = PackActorQHint(gather(config.extrinsic_q, action), transformed(external_value),
            gather(terms, action), gather(config.intrinsic_q, action), transformed(internal_value)), .munchausen_terms = terms};
    }
    auto q_sa = q_values.gather(1, action.to(torch::kInt64).unsqueeze(1)).squeeze(1);
    auto state_value = std::get<0>(q_values.max(1));
    auto terms = torch::zeros_like(q_values);
    if (config.munchausen.enabled) {
        // log_policy_modeはLearner専用。Actorは常に既存のaction scoreから近似する。
        const auto& munchausen = config.munchausen;
        const auto real_score = config.use_tbo ? TransformHInv(q_values, config.tbo_epsilon) : q_values;
        const auto scaled_log = MakeScaledLogPolicy(real_score, munchausen.entropy_tau);
        const auto soft_value = ((scaled_log / munchausen.entropy_tau).exp() * (real_score - scaled_log)).sum(1);
        state_value = config.use_tbo ? TransformH(soft_value, config.tbo_epsilon) : soft_value;
        terms = munchausen.alpha * scaled_log.clamp(munchausen.clip_value_min, 0.0f);
    }
    // OFFも第3列ゼロの同じschemaを運び、WithAction用の全行動bonusを保持する。
    const auto bonus = terms.gather(1, action.to(torch::kInt64).unsqueeze(1)).squeeze(1);
    return { .payload = PackActorQHint(q_sa, state_value, bonus, torch::zeros_like(q_sa), torch::zeros_like(q_sa)),
        .munchausen_terms = terms };
}

ActorQHintBatch anet::rl::dqn::DecodeActorQHint(const torch::Tensor& payload)
{
    // tensor carrierからDQN列を取り出す境界で、共通層が扱わないschemaを検証する。
    if (!payload.defined()) {
        ANET_SYSTEM_ERROR("DQN Actor Q hint payload must be defined.");
    }
    if (payload.dim() != 2 || payload.scalar_type() != torch::kFloat32
        || payload.size(1) != kActorQHintColumnCount) {
        ANET_SYSTEM_ERROR("DQN Actor Q hint must be float32 [B," << kActorQHintColumnCount << "]. actual=" << payload.sizes());
    }
    return ActorQHintBatch{
        .actor_q_sa = payload.select(1, kActorQSaColumn),
        .actor_state_value = payload.select(1, kActorStateValueColumn),
        .munchausen_term = payload.select(1, kActorMunchausenTermColumn),
        .intrinsic_q_sa = payload.select(1, kActorIntrinsicQSaColumn),
        .intrinsic_state_value = payload.select(1, kActorIntrinsicStateValueColumn),
    };
}

std::shared_ptr<anet::rl::BatchActionInfo> DQNActionInfo::To(torch::Device device) const
{
    auto result = std::make_shared<DQNActionInfo>(
        GetAction(device), info_.To(device), aux_, replay_initial_priority_hint_);
    result->train_actor_snapshot_metrics_ = train_actor_snapshot_metrics_;
    result->effective_gamma = effective_gamma;
    result->effective_intrinsic_gamma = effective_intrinsic_gamma;
    result->risk_num_taus = risk_num_taus;
    result->selection_taus = selection_taus;
    result->munchausen_alpha = munchausen_alpha;
    result->munchausen_entropy_tau = munchausen_entropy_tau;
    return result;
}

std::shared_ptr<anet::rl::BatchActionInfo> DQNActionInfo::WithAction(torch::Tensor action) const
{
    ANET_PROFILE_FUNC();
    auto hint = replay_initial_priority_hint_;
    if (hint.has_value()) {
        // 行動Qとbonusを同じ実行行動へ揃え、行動に依存しないstate valueは維持する。
        const auto it = aux_.find("q_values");
        if (it == aux_.end() || !it->second.defined()) {
            ANET_SYSTEM_ERROR(
                "DQNActionInfo::WithAction requires aux[q_values] when an Actor Q hint is present.");
        }
        const auto external = aux_.find(kExtrinsicQValues);
        const auto& q_values = external == aux_.end() ? it->second : external->second;
        const auto term_it = aux_.find("munchausen_terms");
        if (term_it == aux_.end() || !term_it->second.defined()) {
            ANET_SYSTEM_ERROR("DQNActionInfo::WithAction requires aux[munchausen_terms] when an Actor Q hint is present.");
        }
        const auto& terms = term_it->second;
        if (terms.sizes() != q_values.sizes() || terms.device() != q_values.device()) {
            ANET_SYSTEM_ERROR("DQNActionInfo::WithAction requires matching q_values/munchausen_terms shape and device. q_values="
                << q_values.sizes() << " on " << q_values.device() << " terms=" << terms.sizes() << " on " << terms.device());
        }
        // ヒントの計算だけで Q と同じ device の行動を使い、返す行動は呼び出し側の device に置く。
        const auto action_index = action.to(q_values.device(), torch::kInt64).unsqueeze(1);
        auto q_sa = q_values.gather(1, action_index).squeeze(1).to(torch::kFloat32);
        const auto decoded = DecodeActorQHint(hint->GetPayload());
        const auto bonus = terms.gather(1, action_index).squeeze(1);
        auto intrinsic_q_sa = decoded.intrinsic_q_sa;
        if (const auto intrinsic = aux_.find(kIntrinsicQValues); intrinsic != aux_.end()) {
            intrinsic_q_sa = intrinsic->second.gather(1, action_index).squeeze(1);
        }
        auto packed = PackActorQHint(q_sa, decoded.actor_state_value, bonus, intrinsic_q_sa, decoded.intrinsic_state_value);
        hint = ReplayInitialPriorityHint(std::move(packed));
    }
    auto result = std::make_shared<DQNActionInfo>(std::move(action), info_, aux_, std::move(hint));
    result->train_actor_snapshot_metrics_ = train_actor_snapshot_metrics_;
    result->effective_gamma = effective_gamma;
    result->effective_intrinsic_gamma = effective_intrinsic_gamma;
    result->risk_num_taus = risk_num_taus;
    result->selection_taus = selection_taus;
    result->munchausen_alpha = munchausen_alpha;
    result->munchausen_entropy_tau = munchausen_entropy_tau;
    return result;
}

static void AttachQuantileTailPolicyDiagnostics(
    const std::shared_ptr<DQNActionInfo>& action_info,
    const torch::Tensor& tau_ordered_quantiles)
{
    const auto stats = CalculateQuantileTailStatistics(tau_ordered_quantiles);
    const auto float_quantiles = tau_ordered_quantiles.detach().to(torch::kFloat32);
    torch::Tensor disagreement;
    if (tau_ordered_quantiles.size(-1) < 2) {
        disagreement = torch::full(
            {}, std::numeric_limits<float>::quiet_NaN(),
            float_quantiles.options().dtype(torch::kFloat32));
    } else {
        const auto full_q = float_quantiles.mean(-1);
        const auto lower_risk_score = full_q - stats.lower_truncated_std;
        disagreement = full_q.argmax(1).ne(lower_risk_score.argmax(1))
            .to(torch::kFloat32).mean();
    }

    auto& aux = action_info->GetAuxData();
    aux["quantile_tail_upper_std"] = stats.upper_truncated_std;
    aux["quantile_tail_lower_std"] = stats.lower_truncated_std;
    aux["quantile_tail_full_quantiles"] = tau_ordered_quantiles.detach();
    aux["quantile_tail_global_diagnostics"] = torch::stack({
        disagreement, stats.crossing_ratio,
    }).detach();
}

namespace {


    enum class ActionUqeScalarKind {
        kWinRate,
        kMargin,
    };

    struct ActionUqeScalarKey {
        ActionUqeScalarKind kind;
        int64_t action_index;
    };

} // namespace


static int64_t ParseActionScalarIndex(const std::string& key, const char* prefix)
{
    const std::string prefix_text(prefix);
    if (!anet::StartsWith(key, prefix_text) || !anet::EndsWith(key, "]")) {
        ANET_SYSTEM_ERROR("DQNActionInfo: invalid scalar key format: " << key);
    }

    const std::string index_text = key.substr(
        prefix_text.size(),
        key.size() - prefix_text.size() - 1);
    if (index_text.empty()) {
        ANET_SYSTEM_ERROR("DQNActionInfo: action index is empty in scalar key: " << key);
    }

    int64_t action_index = -1;
    try {
        size_t parsed_len = 0;
        action_index = std::stoll(index_text, &parsed_len);
        if (parsed_len != index_text.size()) {
            ANET_SYSTEM_ERROR("DQNActionInfo: invalid action index in scalar key: " << key);
        }
    } catch (const std::exception&) {
        ANET_SYSTEM_ERROR("DQNActionInfo: invalid action index in scalar key: " << key);
    }
    if (action_index < 0) {
        ANET_SYSTEM_ERROR("DQNActionInfo: action index must be non-negative in scalar key: " << key);
    }
    return action_index;
}

static std::optional<ActionUqeScalarKey> ParseActionUqeScalarKey(const std::string& key)
{
    static constexpr const char* kWinRatePrefix = "action_uqe_win_rate.[";
    static constexpr const char* kWinRateBase = "action_uqe_win_rate";
    static constexpr const char* kMarginPrefix = "action_uqe_margin.[";
    static constexpr const char* kMarginBase = "action_uqe_margin";

    if (anet::StartsWith(key, kWinRateBase)) {
        return ActionUqeScalarKey{ ActionUqeScalarKind::kWinRate, ParseActionScalarIndex(key, kWinRatePrefix) };
    }
    if (anet::StartsWith(key, kMarginBase)) {
        return ActionUqeScalarKey{ ActionUqeScalarKind::kMargin, ParseActionScalarIndex(key, kMarginPrefix) };
    }
    return std::nullopt;
}

namespace {


    enum class EpisodeStartActionMarginValueKind {
        kUqe,
        kQ,
    };

    struct EpisodeStartActionMarginKey {
        EpisodeStartActionMarginValueKind value_kind;
        int64_t action_index;
    };

} // namespace


static std::optional<EpisodeStartActionMarginKey> ParseEpisodeStartActionMarginKey(const std::string& key)
{
    static constexpr const char* kUqePrefix = "episode_start_action_uqe_margin.[";
    static constexpr const char* kUqeBase = "episode_start_action_uqe_margin";
    static constexpr const char* kQPrefix = "episode_start_action_q_margin.[";
    static constexpr const char* kQBase = "episode_start_action_q_margin";

    if (anet::StartsWith(key, kUqeBase)) {
        return EpisodeStartActionMarginKey{
            EpisodeStartActionMarginValueKind::kUqe,
            ParseActionScalarIndex(key, kUqePrefix)
        };
    }
    if (anet::StartsWith(key, kQBase)) {
        return EpisodeStartActionMarginKey{
            EpisodeStartActionMarginValueKind::kQ,
            ParseActionScalarIndex(key, kQPrefix)
        };
    }
    return std::nullopt;
}

static std::optional<int64_t> ParseActionFullQMarginKey(const std::string& key)
{
    static constexpr const char* kPrefix = "action_full_q_margin.[";
    static constexpr const char* kBase = "action_full_q_margin";
    if (!anet::StartsWith(key, kBase)) {
        return std::nullopt;
    }
    return ParseActionScalarIndex(key, kPrefix);
}

static torch::Tensor MakeValueSelectionDiagnostics(const anet::TensorDict& output, float beta, bool use_tbo, float epsilon)
{
    ANET_PROFILE_FUNC();
    const auto mean = [&](const std::string& scalar, const std::string& distribution) {
        const auto quantiles = output.Get(distribution);
        auto value = (quantiles ? *quantiles : output.At(scalar)).to(torch::kFloat32);
        if (use_tbo) value = InverseDecomposedValue(value, epsilon);
        return quantiles ? value.mean(2) : value;
    };
    const auto external = mean(DecomposedHeadFactory::kExtrinsicQOutputKey, DecomposedHeadFactory::kExtrinsicQDistOutputKey);
    const auto internal = beta * mean(DecomposedHeadFactory::kIntrinsicQOutputKey, DecomposedHeadFactory::kIntrinsicQDistOutputKey);
    const auto best_external = external.max(1);
    const auto best_combined_action = (external + internal).argmax(1);
    const auto disagreement = (best_combined_action != std::get<1>(best_external)).to(torch::kFloat32);
    const auto regret = std::get<0>(best_external) - external.gather(1, best_combined_action.unsqueeze(1)).squeeze(1);
    auto external_gap = torch::full_like(regret, std::numeric_limits<float>::quiet_NaN());
    auto internal_gap = external_gap;
    if (external.size(1) > 1) {
        const auto external_top = std::get<0>(external.topk(2, 1));
        const auto internal_top = std::get<0>(internal.topk(2, 1));
        external_gap = external_top.select(1, 0) - external_top.select(1, 1);
        internal_gap = internal_top.select(1, 0) - internal_top.select(1, 1);
    }
    return torch::stack({disagreement, regret, external_gap, internal_gap}, 1);
}

std::optional<float> DQNActionInfo::GetScalar(const std::string& key, int64_t index) const
{
    for (size_t metric = 0; metric < kValueSelectionMetricCount; ++metric) {
        const bool lane = key == kValueSelectionLaneKeys[metric];
        if (!lane && key != kValueSelectionMeanKeys[metric]) continue;
        const auto& aux = GetAuxData();
        const auto found = aux.find(kValueSelectionDiagnostics);
        if (found == aux.end()) return std::numeric_limits<float>::quiet_NaN();
        const auto& values = found->second;
        if (values.dim() != 2 || values.size(1) != kValueSelectionMetricCount)
            ANET_SYSTEM_ERROR("Invalid " << kValueSelectionDiagnostics << " shape=" << values.sizes()
                << "; expected [lanes," << kValueSelectionMetricCount << "].");
        if (lane && (index < 0 || index >= values.size(0)))
            ANET_SYSTEM_ERROR("Invalid value selection diagnostic lane: key=" << key << " lane=" << index
                << "; expected [0," << values.size(0) << ").");
        if (!value_selection_diagnostics_cpu_.defined()) value_selection_diagnostics_cpu_ = values.to(torch::kCPU);
        const auto column = value_selection_diagnostics_cpu_.select(1, static_cast<int64_t>(metric));
        return lane ? column[index].item<float>() : column.mean().item<float>();
    }
    if (key == kLearnerScheduleMetricKeys[LearnerScheduleMetric::Gamma]) return effective_gamma;
    if (key == kIntrinsicGamma) return effective_intrinsic_gamma;
    if (key == kLearnerScheduleMetricKeys[LearnerScheduleMetric::MunchausenAlpha]) return munchausen_alpha;
    if (key == kLearnerScheduleMetricKeys[LearnerScheduleMetric::MunchausenEntropyTau]) return munchausen_entropy_tau;

    const bool skip_rate = key.starts_with(DQNActionInfo::kSkipLengthRatePrefix);
    const bool skip_value = key.starts_with(DQNActionInfo::kSkipQPrefix);
    if (skip_rate || skip_value || key == DQNActionInfo::kSkipDecisionRate || key == DQNActionInfo::kSkipLengthMean
        || key == DQNActionInfo::kSkipQGap || key == DQNActionInfo::kSkipDecision || key == DQNActionInfo::kSkipLength) {
        const auto duration = skip_rate ? ParseActionScalarIndex(key, DQNActionInfo::kSkipLengthRateIndexPrefix)
            : skip_value ? ParseActionScalarIndex(key, DQNActionInfo::kSkipQIndexPrefix) : 1;
        if (duration < 1) ANET_SYSTEM_ERROR("DQNActionInfo: " << key << " requires duration >= 1.");
        const auto decision_it = aux_.find(DQNActionInfo::kSkipDecision);
        if (decision_it == aux_.end()) return std::numeric_limits<float>::quiet_NaN();
        const auto& decision = decision_it->second;
        const auto& lengths = aux_.at(DQNActionInfo::kSkipLength);
        const auto& values = aux_.at(DQNActionInfo::kSkipQValues);
        const auto batch = GetAction().size(0);
        if (decision.scalar_type() != torch::kBool || decision.dim() != 1 || decision.size(0) != batch
            || lengths.scalar_type() != torch::kInt64 || lengths.sizes() != decision.sizes()
            || values.dim() != 2 || values.size(0) != batch || values.size(1) < 1) {
            ANET_SYSTEM_ERROR("DQNActionInfo: invalid skip diagnostic payload. decision=" << decision.sizes()
                << " lengths=" << lengths.sizes() << " values=" << values.sizes() << " batch=" << batch);
        }
        const auto limit = values.size(1);
        if ((skip_rate || skip_value) && duration > limit) ANET_SYSTEM_ERROR("DQNActionInfo: " << key << " requires duration in [1," << limit << "].");
        const bool trace = skip_value || key == DQNActionInfo::kSkipDecision || key == DQNActionInfo::kSkipLength;
        if (trace && (index < 0 || index >= batch)) ANET_SYSTEM_ERROR("DQNActionInfo: " << key << " lane=" << index << " expected=[0," << batch - 1 << "].");
        if (!skip_diagnostics_cpu_.defined()) {
            ANET_PROFILE_SCOPE(skip_diagnostics);
            // 判断laneだけをdevice上で集約し、scalarとTraceの値を一度だけ転送する。
            const auto mask = decision.to(torch::kFloat32);
            const auto count = mask.sum();
            const auto selected = values.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1);
            const auto rates = lengths.unsqueeze(1).eq(torch::arange(1, limit + 1, lengths.options()))
                .to(torch::kFloat32).mul(mask.unsqueeze(1)).sum(0) / count;
            skip_diagnostics_cpu_ = torch::cat({mask.mean().reshape({1}),
                ((lengths * mask).sum() / count).reshape({1}),
                (((selected - values.select(1, 0)) * mask).sum() / count).reshape({1}), rates,
                mask, lengths.to(torch::kFloat32), values.reshape({-1})}).detach().to(torch::kCPU);
        }
        const int64_t offset = skip_value ? 3 + limit + 2 * batch + index * limit + duration - 1
            : skip_rate ? 3 + duration - 1
            : key == DQNActionInfo::kSkipDecisionRate ? 0 : key == DQNActionInfo::kSkipLengthMean ? 1
            : key == DQNActionInfo::kSkipQGap ? 2 : 3 + limit + index + (key == DQNActionInfo::kSkipLength ? batch : 0);
        return skip_diagnostics_cpu_[offset].item<float>();
    }
    if (key == "train_actor_snapshot_interval" || key == "train_actor_snapshot_age") {
        if (!train_actor_snapshot_metrics_.has_value()) {
            return std::nullopt;
        }
        return key == "train_actor_snapshot_interval"
            ? train_actor_snapshot_metrics_->interval
            : train_actor_snapshot_metrics_->age;
    }

    const bool is_iqn_policy_metric = key == "iqn_policy_margin_mc_ratio"
        || key == "iqn_uqe_full_q_argmax_disagreement";
    const auto full_q_margin_index = ParseActionFullQMarginKey(key);
    if (is_iqn_policy_metric || full_q_margin_index.has_value()) {
        const auto diagnostics_it = aux_.find("iqn_policy_diagnostics");
        if (diagnostics_it == aux_.end() || !diagnostics_it->second.defined()) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        // packed Tensorのshapeからaction契約を検証し、初回参照時だけCPUへ転送する。
        const auto& diagnostics = diagnostics_it->second;
        if (diagnostics.dim() != 1 || diagnostics.size(0) < 4) {
            ANET_SYSTEM_ERROR("DQNActionInfo: iqn_policy_diagnostics must have shape [2+A] with A >= 2. actual="
                << diagnostics.sizes());
        }
        const int64_t num_actions = diagnostics.size(0) - 2;
        int64_t diagnostics_index = key == "iqn_policy_margin_mc_ratio" ? 0 : 1;
        if (full_q_margin_index.has_value()) {
            if (*full_q_margin_index >= num_actions) {
                ANET_SYSTEM_ERROR("DQNActionInfo: action index out of range. key=" << key
                    << " index=" << *full_q_margin_index
                    << " valid_range=[0," << (num_actions - 1) << "]");
            }
            diagnostics_index = 2 + *full_q_margin_index;
        }
        if (!iqn_policy_diagnostics_cpu_.defined()) {
            iqn_policy_diagnostics_cpu_ = diagnostics.to(torch::kCPU);
        }
        return iqn_policy_diagnostics_cpu_[diagnostics_index].item<float>();
    }

    int64_t quantile_tail_index = -1;
    if (key == "policy_upper_truncated_std") quantile_tail_index = 0;
    else if (key == "policy_lower_truncated_std") quantile_tail_index = 1;
    else if (key == "lower_risk_full_q_argmax_disagreement") quantile_tail_index = 2;
    else if (key == "quantile_crossing_ratio") quantile_tail_index = 3;
    else if (key == "policy_selected_crossing_depth_p90_ratio") quantile_tail_index = 4;
    if (quantile_tail_index >= 0) {
        const auto upper_it = aux_.find("quantile_tail_upper_std");
        const auto lower_it = aux_.find("quantile_tail_lower_std");
        const auto quantiles_it = aux_.find("quantile_tail_full_quantiles");
        const auto global_it = aux_.find("quantile_tail_global_diagnostics");
        if (upper_it == aux_.end() || lower_it == aux_.end()
            || quantiles_it == aux_.end() || global_it == aux_.end()) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        // 最終実行actionへ追従した幅とcrossing深度、global診断を初回参照時だけ1本にpackする。
        if (!quantile_tail_diagnostics_cpu_.defined()) {
            const auto& upper = upper_it->second;
            const auto& lower = lower_it->second;
            const auto& quantiles = quantiles_it->second;
            const auto& global = global_it->second;
            const int64_t batch_size = GetAction().size(0);
            if (upper.dim() != 2 || lower.sizes() != upper.sizes()
                || upper.size(0) != batch_size || quantiles.dim() != 3
                || quantiles.size(0) != batch_size
                || quantiles.size(1) != upper.size(1)
                || global.dim() != 1 || global.size(0) != 2) {
                ANET_SYSTEM_ERROR("DQNActionInfo: invalid quantile tail diagnostic payload. upper="
                    << upper.sizes() << " lower=" << lower.sizes()
                    << " quantiles=" << quantiles.sizes() << " global=" << global.sizes());
            }
            const auto action = GetAction(upper.device()).to(torch::kInt64).unsqueeze(1);
            const auto selected_upper = upper.gather(1, action).squeeze(1).mean();
            const auto selected_lower = lower.gather(1, action).squeeze(1).mean();
            torch::Tensor selected_crossing_depth_p90;
            const int64_t num_quantiles = quantiles.size(2);
            if (num_quantiles < 2) {
                selected_crossing_depth_p90 = torch::full(
                    {}, std::numeric_limits<float>::quiet_NaN(), upper.options());
            } else {
                // positive crossingだけを母集団にし、lane別nearest-rank p90をdevice上で選ぶ。
                const auto quantile_action = action.unsqueeze(2).expand({ -1, 1, num_quantiles });
                const auto selected_quantiles = quantiles.gather(1, quantile_action)
                    .squeeze(1).to(torch::kFloat32);
                const auto depths = (selected_quantiles.slice(1, 0, num_quantiles - 1)
                    - selected_quantiles.slice(1, 1, num_quantiles)).clamp_min(0.0f);
                const auto ranges = std::get<0>(selected_quantiles.max(1))
                    - std::get<0>(selected_quantiles.min(1));
                const auto normalized_depths = depths / ranges.clamp_min(
                	std::numeric_limits<float>::epsilon()).unsqueeze(1);
                const auto positive_count = depths.gt(0.0f).sum(1);
                const auto nearest_rank = (positive_count.to(torch::kFloat32) * 0.9f)
                    .ceil().to(torch::kInt64);
                const int64_t pair_count = num_quantiles - 1;
                const auto sorted_depths = std::get<0>(normalized_depths.sort(1));
                const auto sorted_index = (pair_count - positive_count + nearest_rank - 1)
                    .clamp(0, pair_count - 1).unsqueeze(1);
                const auto lane_p90 = sorted_depths.gather(1, sorted_index).squeeze(1);
                selected_crossing_depth_p90 = torch::where(
                    positive_count.gt(0), lane_p90, torch::zeros_like(lane_p90)).mean();
            }
            quantile_tail_diagnostics_cpu_ = torch::cat({
                selected_upper.reshape({ 1 }),
                selected_lower.reshape({ 1 }),
                global,
                selected_crossing_depth_p90.reshape({ 1 }),
            }).detach().to(torch::kCPU);
        }
        return quantile_tail_diagnostics_cpu_[quantile_tail_index].item<float>();
    }

    const auto episode_start_key = ParseEpisodeStartActionMarginKey(key);
    if (episode_start_key.has_value()) {
        // 指定された価値表現を取得し、非UQE Policyだけは未対応値としてNaNを返す。
        const bool is_uqe = episode_start_key->value_kind == EpisodeStartActionMarginValueKind::kUqe;
        const char* values_key = is_uqe ? "uqe_values" : "q_values";
        const auto values_it = aux_.find(values_key);
        if (values_it == aux_.end() || !values_it->second.defined()) {
            if (is_uqe) {
                return std::numeric_limits<float>::quiet_NaN();
            }
            ANET_SYSTEM_ERROR("DQNActionInfo: required aux value is missing. key=" << key << " aux=" << values_key);
        }

        const auto& values = values_it->second;
        if (values.dim() != 2) {
            ANET_SYSTEM_ERROR("DQNActionInfo: " << values_key << " must have shape [B,A]. actual=" << values.sizes());
        }
        const int64_t batch_size = values.size(0);
        const int64_t num_actions = values.size(1);
        if (batch_size <= 0 || num_actions < 2) {
            ANET_SYSTEM_ERROR("DQNActionInfo: " << values_key
                << " must have non-empty batch and at least two actions. actual=" << values.sizes());
        }
        const int64_t action_index = episode_start_key->action_index;
        if (action_index >= num_actions) {
            ANET_SYSTEM_ERROR("DQNActionInfo: action index out of range. key=" << key << " num_actions=" << num_actions);
        }

        // 現在actionと同じbatchに対応するepisode開始maskを検証する。
        const auto episode_start_it = aux_.find("episode_start");
        if (episode_start_it == aux_.end() || !episode_start_it->second.defined()) {
            ANET_SYSTEM_ERROR("DQNActionInfo: required aux value is missing. key=" << key << " aux=episode_start");
        }
        const auto& episode_start = episode_start_it->second;
        if (episode_start.scalar_type() != torch::kBool) {
            ANET_SYSTEM_ERROR("DQNActionInfo: episode_start must have dtype bool. actual=" << episode_start.scalar_type());
        }
        if (episode_start.dim() != 1 || episode_start.size(0) != batch_size) {
            ANET_SYSTEM_ERROR("DQNActionInfo: episode_start must have shape [" << batch_size
                << "]. actual=" << episode_start.sizes());
        }

        // resetがない通常stepではdevice転送せず、Observerが出力をskipするNaNを返す。
        const auto episode_start_cpu = episode_start.device().is_cpu()
            ? episode_start
            : episode_start.to(torch::kCPU);
        if (!episode_start_cpu.any().item<bool>()) {
            return std::numeric_limits<float>::quiet_NaN();
        }

        // reset laneだけを抽出し、指定actionと最良の他actionとの差を平均する。
        const auto episode_start_on_device = episode_start.device() == values.device()
            ? episode_start
            : episode_start.to(values.device());
        const auto reset_values = values.index({ episode_start_on_device });
        const auto selected = reset_values.select(1, action_index);
        torch::Tensor other_values;
        if (action_index == 0) {
            other_values = reset_values.slice(1, 1, num_actions);
        } else if (action_index == num_actions - 1) {
            other_values = reset_values.slice(1, 0, num_actions - 1);
        } else {
            other_values = torch::cat({
                reset_values.slice(1, 0, action_index),
                reset_values.slice(1, action_index + 1, num_actions)
            }, 1);
        }
        const auto max_other = std::get<0>(other_values.max(1));
        return (selected - max_other).mean().item<float>();
    }

    const auto parsed_key = ParseActionUqeScalarKey(key);
    if (!parsed_key.has_value()) {
        return BatchActionInfo::GetScalar(key, index);
    }
    const int64_t action_index = parsed_key->action_index;

    auto it = aux_.find("uqe_values");
    if (it == aux_.end() || !it->second.defined()) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const auto& uqe_values = it->second;
    if (uqe_values.dim() != 2) {
        ANET_SYSTEM_ERROR("DQNActionInfo: uqe_values must have shape [B,A]. actual=" << uqe_values.sizes());
    }
    const int64_t batch_size = uqe_values.size(0);
    const int64_t num_actions = uqe_values.size(1);
    if (batch_size <= 0 || num_actions < 2) {
        ANET_SYSTEM_ERROR("DQNActionInfo: uqe_values must have non-empty batch and at least two actions. actual=" << uqe_values.sizes());
    }
    if (action_index >= num_actions) {
        ANET_SYSTEM_ERROR("DQNActionInfo: action index out of range. key=" << key << " num_actions=" << num_actions);
    }

    const auto selected = uqe_values.select(1, action_index);
    torch::Tensor other_values;
    if (action_index == 0) {
        other_values = uqe_values.slice(1, 1, num_actions);
    } else if (action_index == num_actions - 1) {
        other_values = uqe_values.slice(1, 0, num_actions - 1);
    } else {
        other_values = torch::cat({
            uqe_values.slice(1, 0, action_index),
            uqe_values.slice(1, action_index + 1, num_actions)
        }, 1);
    }

    const auto max_other = std::get<0>(other_values.max(1));
    if (parsed_key->kind == ActionUqeScalarKind::kWinRate) {
        const auto win_rate = selected.ge(max_other).to(torch::kFloat32).mean();
        return win_rate.item<float>();
    }
    const auto margin = (selected - max_other).mean();
    return margin.item<float>();
}

anet::rl::dqn::ActionPolicy::ActionPolicy(
    const ActionPolicyConfig& config, bool enable_spatial_exploration, int64_t num_envs, const torch::Device& device)
    : config_(config)
    , spatial_enabled_(enable_spatial_exploration)
    , spatial_num_envs_(enable_spatial_exploration ? num_envs : 0)
{
    if (!spatial_enabled_) return;

    if (num_envs <= 0) {
        ANET_SYSTEM_ERROR("ActionPolicy spatial exploration requires num_envs > 0. num_envs=" << num_envs);
    }
    if (num_envs < 32) {
        LOG::warn_once() << "ActionPolicy spatial exploration is enabled with num_envs < 32. num_envs=" << num_envs;
    }

    current_epsilon_ = std::numeric_limits<float>::quiet_NaN();
    current_uqe_tau_ = std::numeric_limits<float>::quiet_NaN();
}

anet::TensorDict anet::rl::dqn::ActionPolicy::ForwardForAction(
    const anet::TensorDict& obs, const DqnForwardExecutor& forward, const anet::nn::NetworkForwardCallInfo& info, anet::nn::NetworkForwardCapture* capture) const
{
    return forward.Forward(obs, info, capture);
}

anet::TensorDict anet::rl::dqn::ActionPolicy::ForwardForActionWithTaus(
    const anet::TensorDict& obs, const torch::Tensor& taus,
    const DqnForwardExecutor& forward, const anet::nn::NetworkForwardCallInfo& info, anet::nn::NetworkForwardCapture* capture,
    const torch::Tensor& additional_taus) const
{
    // 追加方策のqueryも同じforwardへ含め、両Networkが全く同じtausを参照できるようにする。
    const auto combined_taus = additional_taus.defined() ? torch::cat({taus, additional_taus}, 1) : taus;
    anet::TensorDict network_input = obs;
    network_input.Set(anet::rl::dqn::kKey_Taus, combined_taus);
    auto result = ForwardForAction(network_input, forward, info, capture);
    result.Set(anet::rl::dqn::kKey_Taus, combined_taus);
    return result;
}

static anet::ProfiledValueConfig<float> ValidateSpatialEndpoint(anet::ProfiledValueConfig<float> endpoint)
{
    // 単体構築のmetadataを補い、ProfiledValue構築より先に論理キー付きで全active端点を検証する。
    endpoint.min_value = 0.0f;
    endpoint.max_value = 1.0f;
    anet::detail::ValidateProfiledValueConfig(endpoint);
    return endpoint;
}

SpatialLadderSchedule::SpatialLadderSchedule(const ActionPolicyConfig::LadderConfig& config,
    std::string scale, int64_t lanes, torch::Device device, const std::string& scale_key)
    : first_(ValidateSpatialEndpoint(config.first))
    , last_(ValidateSpatialEndpoint(config.last))
    , scale_(std::move(scale)), lanes_(lanes), device_(device)
{
    ValidateSpatialScale(scale_, scale_key);
    if (lanes < 1)
        ANET_SYSTEM_ERROR("Invalid spatial ladder: lanes=" << lanes << " expected lanes>=1");
    Update(0);
}

void SpatialLadderSchedule::Update(step_t step)
{
    ANET_PROFILE_FUNC();
    const float first = first_.Evaluate(step), last = last_.Evaluate(step);
    if (first == current_first_ && last == current_last_) return;
    current_first_ = first;
    current_last_ = last;
    constexpr float kLogFloor = 1.0e-4f;
    const float a = std::max(first, kLogFloor), b = std::max(last, kLogFloor);
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    if (lanes_ == 1) {
        tensor_ = torch::full({1}, first, options);
        mean_ = minimum_ = maximum_ = first;
        return;
    }
    // 両端が下限以上の経路は従来と同じ式・演算順を維持する。
    const auto base = torch::linspace(0.0f, 1.0f, lanes_, options);
    if (scale_ == "linear" || a == b) {
        tensor_ = first + base * (last - first);
    } else if (first < kLogFloor || last < kLogFloor) {
        // 下限付近の微小差を直接引かず、D8の伸縮を正規化した重みで計算する。
        const double log_ratio = std::log(static_cast<double>(a) / static_cast<double>(b));
        const auto numerator = torch::expm1((1.0f - base) * log_ratio);
        // 同じdevice計算の先頭で割ると端点の重みが正確に1になり、readbackも不要。
        const auto weight = numerator / numerator.select(0, 0);
        tensor_ = first * weight + last * (1.0f - weight);
    } else {
        tensor_ = b * torch::exp((1.0f - base) * std::log(a / b));
    }
    tensor_ = tensor_.flip({0});
    // scalar診断はCPU上の端点から算出し、device同期を増やさない。
    double sum = 0.0;
    for (int64_t i = 0; i < lanes_; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(lanes_ - 1);
        float v = first + t * (last - first);
        if (scale_ == "log" && a != b) {
            if (first < kLogFloor || last < kLogFloor) {
                // Tensorと同じ正規化重みをCPUで評価し、診断用のdevice同期を避ける。
                const double log_ratio = std::log(static_cast<double>(a) / static_cast<double>(b));
                const double weight = std::expm1((1.0 - static_cast<double>(t)) * log_ratio) / std::expm1(log_ratio);
                v = static_cast<float>(first * weight + last * (1.0 - weight));
            } else {
                v = b * std::exp((1.0f - t) * std::log(a / b));
            }
        }
        sum += v;
    }
    mean_ = static_cast<float>(sum / lanes_);
    minimum_ = std::min(first, last);
    maximum_ = std::max(first, last);
}

static torch::Tensor PrepareSpatialTensorForAction(
    const torch::Tensor& tensor, int64_t expected_num_envs, int64_t actual_num_envs, const torch::Device& device, const char* name)
{
    if (!tensor.defined()) {
        ANET_SYSTEM_ERROR("Spatial exploration tensor is not initialized: " << name);
    }
    if (expected_num_envs != actual_num_envs) {
        ANET_SYSTEM_ERROR("Spatial exploration batch size mismatch for " << name
            << ". expected num_envs=" << expected_num_envs << " actual=" << actual_num_envs);
    }
    if (tensor.dim() != 1 || tensor.size(0) != actual_num_envs) {
        ANET_SYSTEM_ERROR("Spatial exploration tensor shape mismatch for " << name
            << ". expected=[" << actual_num_envs << "] actual=" << tensor.sizes());
    }
    return tensor.device() == device ? tensor : tensor.to(device);
}

torch::Tensor anet::rl::dqn::ActionPolicy::GetSpatialEpsilonTensor(int64_t num_envs, const torch::Device& device, bool is_uqe) const
{
    return PrepareSpatialTensorForAction(
        is_uqe ? spatial_uqe_eps_tensor_ : spatial_eps_tensor_,
        spatial_num_envs_,
        num_envs,
        device,
        is_uqe ? "uqe_epsilon" : "epsilon");
}

torch::Tensor anet::rl::dqn::ActionPolicy::GetSpatialTauTensor(int64_t num_envs, const torch::Device& device) const
{
    return PrepareSpatialTensorForAction(spatial_tau_tensor_, spatial_num_envs_, num_envs, device, "uqe_tau");
}

torch::Tensor anet::rl::dqn::ActionPolicy::MakeEpsilonGreedyAction(const torch::Tensor& greedy_action, float epsilon, int64_t num_envs, int64_t n_actions, std::shared_ptr<anet::RandomGenerator> rnd) const
{
    ANET_PROFILE_FUNC();

    // epsilongがゼロ(十分小さい)場合はGreedy（乱数生成影響を排除）
    if (epsilon <= std::numeric_limits<float>::epsilon()) {
        return greedy_action;
    }

    auto device = greedy_action.device();
    auto gen = rnd->GetTorchGenerator(device);

    // ランダム選択するActionをマスクとして選択（εを使って）
    auto mask = torch::rand({ num_envs }, gen, torch::TensorOptions().device(device)).lt(epsilon);    // mask: (N) bool, GPU上で生成

    // ランダム選択対象のアクションについて、乱数でアクション決定
    auto random_actions = torch::randint(/*low=*/0, /*high=*/n_actions, { num_envs }, gen, // random actions(N) int64
        torch::TensorOptions().dtype(torch::kInt64).device(device));

    // actions: where(mask, random_actions, greedy)
    auto actions = torch::where(mask, random_actions, greedy_action);

    return actions;
}

torch::Tensor anet::rl::dqn::ActionPolicy::MakeEpsilonGreedyAction(
    const torch::Tensor& greedy_action, const torch::Tensor& epsilon_tensor, int64_t num_envs, int64_t n_actions, std::shared_ptr<anet::RandomGenerator> rnd) const
{
    ANET_PROFILE_FUNC();

    if (epsilon_tensor.dim() != 1 || epsilon_tensor.size(0) != num_envs) {
        ANET_SYSTEM_ERROR("Vectorized epsilon shape mismatch. expected=[" << num_envs << "] actual=" << epsilon_tensor.sizes());
    }

    auto device = greedy_action.device();
    auto gen = rnd->GetTorchGenerator(device);
    auto epsilon = epsilon_tensor.device() == device ? epsilon_tensor : epsilon_tensor.to(device);

    auto mask = torch::rand({ num_envs }, gen, torch::TensorOptions().device(device)).lt(epsilon);
    auto random_actions = torch::randint(/*low=*/0, /*high=*/n_actions, { num_envs }, gen,
        torch::TensorOptions().dtype(torch::kInt64).device(device));
    return torch::where(mask, random_actions, greedy_action);
}

std::shared_ptr<DQNActionInfo> anet::rl::dqn::ActionPolicy::MakeActionInfo(const torch::Tensor& action_values, const torch::Tensor& q_values, const torch::Tensor& q_quantiles) const
{
    ANET_PROFILE_FUNC();

    auto action_info = std::make_shared<DQNActionInfo>(action_values);
    auto& aux = action_info->GetAuxData();

    // aux[max_q]
    auto max_pair = q_values.max(1);
    auto max_q = std::get<0>(max_pair).detach();
    aux["max_q"] = max_q;
    aux["q_values"] = q_values;
    aux["q_quantiles"] = q_quantiles;
    aux["raw_actions"] = action_values;

    // QRの固定quantile列はtau順のfull distributionとしてそのまま診断へ使う。
    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeQr && q_quantiles.defined()) {
        AttachQuantileTailPolicyDiagnostics(action_info, q_quantiles);
    }

    return action_info;
}

void anet::rl::dqn::ActionPolicy::UpdateEpsilon(step_t step, bool is_uqe)
{
    if (IsSpatialExplorationEnabled()) {
        auto& ladder = is_uqe ? uqe_epsilon_ladder_ : epsilon_ladder_;
        ladder->Update(step);
        (is_uqe ? spatial_uqe_eps_tensor_ : spatial_eps_tensor_) = ladder->Tensor();
        current_epsilon_ = ladder->Mean();
    } else {
        current_epsilon_ = epsilon_schedule_->Evaluate(step);
    }
}

std::optional<float> anet::rl::dqn::ActionPolicy::GetScalar(const std::string& key, int64_t index) const
{
    constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
    constexpr const char* kMetricKeys[] = {"epsilon", "uqe_tau", "uqe_epsilon", "epsilon_lane_min", "epsilon_lane_max", "uqe_epsilon_lane_min", "uqe_epsilon_lane_max", "uqe_tau_lane_min", "uqe_tau_lane_max"};
    const auto* epsilon = uses_uqe_ ? uqe_epsilon_ladder_.get() : epsilon_ladder_.get();
    if (key == kMetricKeys[0]) return current_epsilon_;
    if (key == kMetricKeys[1]) return thompson_ ? kNaN : current_uqe_tau_;
    if (key == kMetricKeys[2]) return uses_uqe_ ? current_epsilon_ : kNaN;
    if (key == kMetricKeys[3]) return epsilon ? epsilon->Min() : current_epsilon_;
    if (key == kMetricKeys[4]) return epsilon ? epsilon->Max() : current_epsilon_;
    if (key == kMetricKeys[5]) return uses_uqe_ ? (epsilon ? epsilon->Min() : current_epsilon_) : kNaN;
    if (key == kMetricKeys[6]) return uses_uqe_ ? (epsilon ? epsilon->Max() : current_epsilon_) : kNaN;
    if (key == kMetricKeys[7]) return tau_ladder_ ? tau_ladder_->Min() : (uses_uqe_ && !thompson_ ? current_uqe_tau_ : kNaN);
    if (key == kMetricKeys[8]) return tau_ladder_ ? tau_ladder_->Max() : (uses_uqe_ && !thompson_ ? current_uqe_tau_ : kNaN);
    return std::nullopt;
}

anet::rl::dqn::EpsilonGreedyActionPolicy::EpsilonGreedyActionPolicy(
    const ActionPolicyConfig& config, bool enable_spatial_exploration, int64_t num_envs, const torch::Device& device)
    : ActionPolicy(config, enable_spatial_exploration, num_envs, device)
{
    if (IsSpatialExplorationEnabled())
        epsilon_ladder_ = std::make_unique<SpatialLadderSchedule>(config.spatial.epsilon, config.spatial.scale, num_envs,
            device, config.full_keys.GetFullKey("spatial.scale"));
    if (!IsSpatialExplorationEnabled()) epsilon_schedule_.emplace(config.epsilon);
    UpdateEpsilon(0);
}

void anet::rl::dqn::EpsilonGreedyActionPolicy::UpdateSchedule(const StepCounts& counts)
{
    UpdateEpsilon(counts.exp_step);
}

std::shared_ptr<DQNActionInfo> EpsilonGreedyActionPolicy::SelectAction(const anet::TensorDict& obs, bool greedy_only,
    const DqnForwardExecutor& forward, std::shared_ptr<anet::RandomGenerator> rnd, const anet::nn::NetworkForwardCallInfo& info, anet::nn::NetworkForwardCapture* capture,
    const DiagnosticActionEvaluation* evaluation, const torch::Tensor& additional_taus) const
{
    ANET_PROFILE_FUNC();

    torch::NoGradGuard grad_guard;
    torch::ScalarType amp_dtype = config_.use_amp_bf16 ? torch::kBFloat16 : torch::kHalf;
    anet::Autocast cast_guard(obs.device(), config_.use_amp && evaluation == nullptr, amp_dtype);

    // IQNではpolicy自身のtau ruleでサンプルし、QR/TDは従来入力をそのまま使う。
    anet::TensorDict out;
    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn) {
        auto taus = evaluation
            ? MakeTauMidpointPositions(evaluation->num_taus, torch::TensorOptions().dtype(torch::kFloat32).device(obs.device()))
                .unsqueeze(0).expand({ obs.Size(0), evaluation->num_taus })
            : GenerateTaus(obs.Size(0), config_.tau_rule.num_taus, config_.tau_rule.sample_mode,
                0.0f, 1.0f, obs.device(), *rnd);
        out = ForwardForActionWithTaus(obs, taus, forward, info, capture, additional_taus);
    } else {
        out = ForwardForAction(obs, forward, info, capture);
    }
    auto q_values = out.At("q");
    torch::Tensor q_quantiles;
    if (out.Contains("q_dist")) {
        q_quantiles = out.At("q_dist");
        if (additional_taus.defined()) {
            q_quantiles = q_quantiles.narrow(-1, 0, q_quantiles.size(-1) - additional_taus.size(1));
            q_values = q_quantiles.mean(-1);
        }
    }

    // Q値GreedyなActionを生成
    auto greedy_action = q_values.argmax(1, /*keepdim=*/false);        // greedy = argmax(q_values, dim=1)

    // Greedy指定ならargmxを返す
    if (evaluation) {
        auto result = std::make_shared<DQNActionInfo>(greedy_action);
        result->forward_output = std::move(out);
        return result;
    }
    if (greedy_only) {
        auto result = MakeActionInfo(greedy_action, q_values, q_quantiles);
        result->forward_output = std::move(out);
        return result;
    }

    // EpsilonGreedy
    const int64_t N = q_values.sizes()[0];      // shape 読み取りは TensorOptions 経由で同期を回避
    const int64_t A = q_values.sizes()[1];
    torch::Tensor actions;
    if (IsSpatialExplorationEnabled()) {
        actions = MakeEpsilonGreedyAction(greedy_action, GetSpatialEpsilonTensor(N, q_values.device(), false), N, A, rnd);
    } else {
        actions = MakeEpsilonGreedyAction(greedy_action, current_epsilon_, N, A, rnd);
    }
    auto action_info = MakeActionInfo(actions, q_values, q_quantiles);
    action_info->forward_output = std::move(out);
    return action_info;
}

anet::rl::dqn::UQEActionPolicy::UQEActionPolicy(
    const ActionPolicyConfig& config, bool enable_spatial_exploration, int64_t num_envs, const torch::Device& device)
    : ActionPolicy(config, enable_spatial_exploration, num_envs, device)
{
    uses_uqe_ = true;
    if (IsSpatialExplorationEnabled()) {
        const auto scale_key = config.full_keys.GetFullKey("spatial.scale");
        uqe_epsilon_ladder_ = std::make_unique<SpatialLadderSchedule>(config.spatial.uqe.epsilon, config.spatial.scale, num_envs, device, scale_key);
        tau_ladder_ = std::make_unique<SpatialLadderSchedule>(config.spatial.uqe.tau, config.spatial.scale, num_envs, device, scale_key);
    }
    if (!IsSpatialExplorationEnabled()) {
        epsilon_schedule_.emplace(config.uqe.epsilon);
        if (!config.IsThompsonSampling()) tau_schedule_.emplace(config.uqe.tau);
    }
    if (!tau_schedule_ && !IsSpatialExplorationEnabled()) thompson_ = true;
    UpdateEpsilon(0, true);
    UpdateTau(0);
}

void anet::rl::dqn::UQEActionPolicy::UpdateTau(step_t step)
{
    if (IsSpatialExplorationEnabled()) {
        tau_ladder_->Update(step);
        spatial_tau_tensor_ = tau_ladder_->Tensor();
        current_uqe_tau_ = tau_ladder_->Mean();
    } else if (!thompson_) {
        current_uqe_tau_ = tau_schedule_->Evaluate(step);
    }
}

void anet::rl::dqn::UQEActionPolicy::UpdateSchedule(const StepCounts& counts)
{
    UpdateEpsilon(counts.exp_step, true);
    UpdateTau(counts.exp_step);
}

/// Q分布の上位tau%(上振れ時)Q値が最大になる行動を選択する。
/// tauが大きい(1.0に近い)程、不確実だけど化ける可能性がある行動が相対的に選択されやすくなる
torch::Tensor UQEActionPolicy::MakeUQEValues(float tau, const torch::Tensor& q_quantiles) const
{
    ANET_PROFILE_FUNC();
    
    // hard方策の演算順を維持し、soft方策とも同じ経験分位計算を共有する。
    return MakeRiskBiasedScore(tau, q_quantiles, config_.uqe.use_tail_mean);
}

torch::Tensor UQEActionPolicy::MakeVectorizedUQEValues(const torch::Tensor& tau_tensor, const torch::Tensor& q_quantiles) const
{
    // q_quantiles: (N, A, n_quantiles)
    // tau_tensor:  (N, 1)  (0.0 ~ 1.0)

    if (q_quantiles.dim() != 3) {
        ANET_SYSTEM_ERROR("UQEActionPolicy::MakeVectorizedUQEValues expected q_quantiles shape [N,A,Q], actual=" << q_quantiles.sizes());
    }

    const int64_t N = q_quantiles.size(0);
    const int64_t A = q_quantiles.size(1);
    const int64_t n_quantiles = q_quantiles.size(2);
    auto device = q_quantiles.device();

    if (!tau_tensor.defined()) {
        ANET_SYSTEM_ERROR("UQEActionPolicy::MakeVectorizedUQEValues received undefined tau_tensor.");
    }
    if (tau_tensor.dim() != 2 || tau_tensor.size(0) != N || tau_tensor.size(1) != 1) {
        ANET_SYSTEM_ERROR("UQEActionPolicy::MakeVectorizedUQEValues expected tau_tensor shape [" << N << ",1], actual="
            << tau_tensor.sizes());
    }
    auto tau = tau_tensor.device() == device ? tau_tensor : tau_tensor.to(device);
    if (tau.dtype() != q_quantiles.dtype()) {
        tau = tau.to(q_quantiles.dtype());
    }

    // 最後の次元(分位点)を確実に昇順(小さい順)にソートする
    torch::Tensor sorted_quantiles = std::get<0>(q_quantiles.sort(-1, /*descending=*/false));

    // インデックスの計算 (N, 1)
    auto tau_idx = (tau * (n_quantiles - 1)).to(torch::kLong);
    tau_idx = tau_idx.clamp(0, n_quantiles - 1);

    torch::Tensor uqe_values;

    if (config_.uqe.use_tail_mean) {
        // マスクを使って平均を計算する

        // range: (1, 1, n_quantiles) -> [0, 1, 2, ...]
        auto range = torch::arange(n_quantiles, device).view({ 1, 1, -1 });

        // mask: (N, 1, n_quantiles)
        // range >= tau_idx の部分が True (1.0)
        // tau_idx は (N, 1) なので (N, 1, 1) に view して broadcast
        auto mask = range.ge(tau_idx.view({ N, 1, 1 })).to(q_quantiles.dtype());

        // 平均計算: (sum(Q * mask) / sum(mask))
        // small_epsilon を足して 0除算防止
        uqe_values = (sorted_quantiles * mask).sum(-1) / (mask.sum(-1) + 1e-6);
    } else {
        // tau_idx を (N, A, 1) に拡張
        // バッチ(N)ごとに違うインデックスだが、Action(A)に対しては同じインデックスを使う
        auto gather_idx = tau_idx.view({ N, 1, 1 }).expand({ N, A, 1 });

        // gather: dim=-1 (quantiles次元) に沿って収集
        // output: (N, A, 1) -> squeeze -> (N, A)
        uqe_values = sorted_quantiles.gather(-1, gather_idx).squeeze(-1);
    }

    return uqe_values;
}

std::shared_ptr<DQNActionInfo> UQEActionPolicy::MakeUQEActionInfo(float tau, const torch::Tensor& tau_tensor, const anet::TensorDict& obs,
    bool greedy_only, const DqnForwardExecutor& forward, std::shared_ptr<anet::RandomGenerator> rnd, const anet::nn::NetworkForwardCallInfo& info, anet::nn::NetworkForwardCapture* capture,
    bool iqn_use_full_range, const DiagnosticActionEvaluation* evaluation, const torch::Tensor& additional_taus) const
{
    ANET_PROFILE_FUNC();

    torch::NoGradGuard grad_guard;
    torch::ScalarType amp_dtype = config_.use_amp_bf16 ? torch::kBFloat16 : torch::kHalf;
    anet::Autocast cast_guard(obs.device(), config_.use_amp && evaluation == nullptr, amp_dtype);

    const bool full_query = config_.full_distribution_query.enabled && evaluation == nullptr;
    // policy更新で変化する探索率と、呼び出し側が決めたrisk基準tauを今回のaction queryへ反映する。
    float effective_epsilon = current_epsilon_;
    float effective_tau = tau;
    bool use_vectorized_tau = tau_tensor.defined();

    //greedy_onlyが指定された場合は、ランダムノイズ(ε)を強制的にゼロにする
    // (Tauによる楽観的選択の基準は維持しつつ、デタラメな行動を防ぐ)
    if (greedy_only) {
        effective_epsilon = 0.0f;
    }

    // IQNでは行動scoreを作るrisk queryを先頭へ置き、任意のfull queryを後ろへ連結して1回のforwardで処理する。
    // QRではネットワークが固定quantileを返すため、tausを観測へ追加せず従来のforwardを使う。
    ANET_PROFILE_SCOPE(generate_taus);
    anet::TensorDict out;
    int64_t risk_num_taus = 0;
    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn) {
        const int64_t N = obs.Size(0);
        torch::Tensor risk_taus;
        if (evaluation) {
            // 診断は固定midpoint。point UQEでは同一の基準分位を1点だけ評価する。
            const int64_t count = config_.uqe.use_tail_mean ? evaluation->num_taus : 1;
            const auto unit = MakeTauMidpointPositions(count, torch::TensorOptions().dtype(torch::kFloat32).device(obs.device()));
            const auto lower = use_vectorized_tau ? tau_tensor.reshape({ N, 1 })
                : torch::full({ N, 1 }, effective_tau, unit.options());
            risk_taus = config_.uqe.use_tail_mean ? lower + (1.0f - lower) * unit.unsqueeze(0) : lower;
        } else if (iqn_use_full_range) {
            // ThompsonSamplingの非spatial IQNは、各環境でfull rangeをサンプルして分布から直接scoreを作る。
            risk_taus = GenerateTaus(
                N, config_.tau_rule.num_taus, config_.tau_rule.sample_mode,
                0.0f, 1.0f, obs.device(), *rnd);
        } else if (use_vectorized_tau) {
            // spatial UQEでは環境ごとの下限tauを使い、tail meanなら下限から1までをサンプルする。
            auto lower = tau_tensor.reshape({ N });
            if (config_.uqe.use_tail_mean) {
                risk_taus = GenerateTaus(
                    lower, 1.0f, config_.tau_rule.num_taus, config_.tau_rule.sample_mode, *rnd);
            } else {
                // point UQEは基準分位そのものを使う。full query併用時は重複計算を避けて1点だけ渡す。
                const int64_t point_count = full_query ? 1 : config_.tau_rule.num_taus;
                risk_taus = lower.unsqueeze(1).expand({ N, point_count });
            }
        } else if (config_.uqe.use_tail_mean) {
            // scalar tail meanは[effective_tau, 1]を問い合わせる。
            risk_taus = GenerateTaus(
                N, config_.tau_rule.num_taus, config_.tau_rule.sample_mode,
                effective_tau, 1.0f, obs.device(), *rnd);
        } else {
            // scalar point UQEはeffective_tauだけを問い合わせる。
            const int64_t point_count = full_query ? 1 : config_.tau_rule.num_taus;
            risk_taus = torch::full(
                { N, point_count }, effective_tau,
                torch::TensorOptions().dtype(torch::kFloat32).device(obs.device()));
        }
        risk_num_taus = risk_taus.size(1);

        // 可視化・統計用のfull rangeはrisk queryと混ぜず、後段で切り分けられる順序で連結する。
        torch::Tensor iqn_taus = risk_taus;
        if (full_query) {
            const auto& full_rule = config_.full_distribution_query.tau_rule;
            auto full_taus = GenerateTaus(
                N, full_rule.num_taus, full_rule.sample_mode,
                0.0f, 1.0f, obs.device(), *rnd);
            iqn_taus = torch::cat({ risk_taus, full_taus }, 1);
        }

        ANET_PROFILE_SCOPE_NEXT(forward);
        out = ForwardForActionWithTaus(obs, iqn_taus, forward, info, capture, additional_taus);
    } else {
        ANET_PROFILE_SCOPE_NEXT(forward);
        out = ForwardForAction(obs, forward, info, capture);
    }
    // Headのqは連結した全tausの平均になるため、full query有効時は使わず各領域から平均を再計算する。
    ANET_PROFILE_SCOPE_NEXT(split_outputs);
    auto q_values = out.At("q");
    auto q_quantiles = out.At("q_dist");
    // primaryの選択と診断には、追加方策のqueryを混ぜない。
    if (additional_taus.defined()) {
        q_quantiles = q_quantiles.narrow(-1, 0, q_quantiles.size(-1) - additional_taus.size(1));
        q_values = q_quantiles.mean(-1);
    }
    torch::Tensor full_q_values;
    torch::Tensor full_q_quantiles;
    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn && full_query) {
        full_q_quantiles = q_quantiles.slice(2, risk_num_taus);
        q_quantiles = q_quantiles.slice(2, 0, risk_num_taus);
        q_values = q_quantiles.mean(2);
        full_q_values = full_q_quantiles.mean(2);
    }

    // IQNのrisk部分は問い合わせたい分位だけで構成済みなので、その平均をそのまま行動scoreにする。
    // QRは固定quantile列から基準分位またはupper-tailを抽出する既存処理を維持する。
    torch::Tensor uqe_values;
    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn) {
        uqe_values = q_values;
    } else if (use_vectorized_tau) {
        // 学習時: バッチごとに異なるTauが渡されている場合（hompsonSampling向け）
        uqe_values = MakeVectorizedUQEValues(tau_tensor, q_quantiles);
    } else {
        // Eval時 または 学習時(固定Tau): スカラーTauを使う
        uqe_values = MakeUQEValues(effective_tau, q_quantiles);
    }
    auto uqe_action_values = uqe_values.argmax(1);
    if (evaluation) {
        auto result = std::make_shared<DQNActionInfo>(uqe_action_values);
        result->forward_output = std::move(out);
        result->risk_num_taus = risk_num_taus;
        result->selection_taus = tau_tensor;
        return result;
    }

    // risk-biased scoreでgreedy actionを決めた後、通常またはspatialなepsilon探索を適用する。
    const int64_t N = q_values.sizes()[0];      // shape 読み取りは TensorOptions 経由で同期を回避
    const int64_t A = q_values.sizes()[1];
    torch::Tensor actions;
    if (IsSpatialExplorationEnabled() && !greedy_only) {
        actions = MakeEpsilonGreedyAction(uqe_action_values, GetSpatialEpsilonTensor(N, q_values.device(), true), N, A, rnd);
    } else {
        actions = MakeEpsilonGreedyAction(uqe_action_values, effective_epsilon, N, A, rnd);
    }

    // q_quantilesは行動選択に使ったrisk分布、full_q_quantilesは独立した観測用分布として公開する。
    // どちらも同一forward由来なので、QValuePanelやmetricのための追加forwardは発生しない。
    auto action_info = MakeActionInfo(actions, q_values, q_quantiles);
    action_info->forward_output = std::move(out);
    action_info->risk_num_taus = risk_num_taus;
    action_info->selection_taus = tau_tensor;
    action_info->GetAuxData()["uqe_values"] = uqe_values.detach();
    if (full_q_quantiles.defined()) {
        action_info->GetAuxData()["full_q_values"] = full_q_values.detach();
        action_info->GetAuxData()["full_q_quantiles"] = full_q_quantiles.detach();
        if (config_.full_distribution_query.tau_rule.sample_mode == "fixed") {
            AttachQuantileTailPolicyDiagnostics(action_info, full_q_quantiles);
        }
    }

    // 行動決定済みのTensorだけを使い、Policy診断を1本のreadback単位へまとめる。
    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn) {
        ANET_PROFILE_SCOPE_NEXT(policy_diagnostics);
        const auto float_options = q_values.options().dtype(torch::kFloat32);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        torch::Tensor margin_ratio = torch::full({}, nan, float_options);
        if (risk_num_taus >= 2) {
            const auto risk_quantiles_float = q_quantiles.detach().to(torch::kFloat32);
            const auto uqe_values_float = uqe_values.detach().to(torch::kFloat32);
            const auto top2 = uqe_values_float.topk(/*k=*/2, /*dim=*/1);
            const auto top2_values = std::get<0>(top2);
            const auto top2_indices = std::get<1>(top2);
            const auto mc_scale = risk_quantiles_float.std(/*dim=*/2, /*unbiased=*/true)
                / std::sqrt(static_cast<float>(risk_num_taus));
            const auto top2_scale = mc_scale.gather(1, top2_indices);
            const auto denominator = (top2_scale.square().sum(1)).sqrt() + 1.0e-6f;
            margin_ratio = ((top2_values.select(1, 0) - top2_values.select(1, 1)) / denominator).mean();
        }

        torch::Tensor disagreement = torch::full({}, nan, float_options);
        torch::Tensor full_q_margins = torch::full({ A }, nan, float_options);
        if (full_q_values.defined()) {
            const auto uqe_values_float = uqe_values.detach().to(torch::kFloat32);
            const auto full_q_values_float = full_q_values.detach().to(torch::kFloat32);
            disagreement = uqe_values_float.argmax(1).ne(full_q_values_float.argmax(1))
                .to(torch::kFloat32).mean();

            const auto full_top2 = full_q_values_float.topk(/*k=*/2, /*dim=*/1);
            const auto top2_values = std::get<0>(full_top2);
            const auto top1_indices = std::get<1>(full_top2).select(1, 0).unsqueeze(1);
            const auto action_indices = torch::arange(A,
                torch::TensorOptions().dtype(torch::kInt64).device(full_q_values.device())).unsqueeze(0);
            const auto max_other = torch::where(
                action_indices.eq(top1_indices),
                top2_values.select(1, 1).unsqueeze(1),
                top2_values.select(1, 0).unsqueeze(1));
            full_q_margins = (full_q_values_float - max_other).mean(0);
        }
        action_info->GetAuxData()["iqn_policy_diagnostics"] = torch::cat({
            margin_ratio.reshape({ 1 }), disagreement.reshape({ 1 }), full_q_margins
        }).detach();
    }
    return action_info;
}

std::shared_ptr<DQNActionInfo> UQEActionPolicy::SelectAction(const anet::TensorDict& obs, bool greedy_only,
    const DqnForwardExecutor& forward, std::shared_ptr<anet::RandomGenerator> rnd, const anet::nn::NetworkForwardCallInfo& info, anet::nn::NetworkForwardCapture* capture,
    const DiagnosticActionEvaluation* evaluation, const torch::Tensor& additional_taus) const
{
    if (IsSpatialExplorationEnabled()) {
        const int64_t N = obs.Size(0);
        auto tau_tensor = GetSpatialTauTensor(N, obs.device()).view({ N, 1 });
        return MakeUQEActionInfo(0.0f, tau_tensor, obs, greedy_only, forward, rnd, info, capture, false, evaluation, additional_taus);
    }
    return MakeUQEActionInfo(current_uqe_tau_, torch::Tensor(), obs, greedy_only, forward, rnd, info, capture, false, evaluation, additional_taus);
}

anet::rl::dqn::ThompsonSamplingActionPolicy::ThompsonSamplingActionPolicy(
    const ActionPolicyConfig& config, bool enable_spatial_exploration, int64_t num_envs, const torch::Device& device)
    : UQEActionPolicy(config, enable_spatial_exploration, num_envs, device)
{
    thompson_ = true;
}

void anet::rl::dqn::ThompsonSamplingActionPolicy::UpdateSchedule(const StepCounts& counts)
{
    UpdateEpsilon(counts.exp_step, true);
    if (IsSpatialExplorationEnabled()) UpdateTau(counts.exp_step);
}

std::shared_ptr<DQNActionInfo> ThompsonSamplingActionPolicy::SelectAction(const anet::TensorDict& obs, bool greedy_only,
    const DqnForwardExecutor& forward, std::shared_ptr<anet::RandomGenerator> rnd, const anet::nn::NetworkForwardCallInfo& info, anet::nn::NetworkForwardCapture* capture,
    const DiagnosticActionEvaluation* evaluation, const torch::Tensor& additional_taus) const
{
    if (evaluation) ANET_SYSTEM_ERROR("Replay fit requires a deterministic target policy; ThompsonSampling is unsupported.");
    // ランダムな Tau をバッチサイズ分生成 (N, 1)
    const int64_t N = obs.Size(0);
    auto device = obs.device();
    if (IsSpatialExplorationEnabled()) {
        auto tau_tensor = GetSpatialTauTensor(N, device).view({ N, 1 });
        return MakeUQEActionInfo(0.0f, tau_tensor, obs, greedy_only, forward, rnd, info, capture, false, nullptr, additional_taus);
    }

    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn) {
        return MakeUQEActionInfo(
            0.0f, torch::Tensor(), obs, greedy_only, forward, rnd, info, capture,
            /*iqn_use_full_range=*/true, nullptr, additional_taus);
    }

    auto gen = rnd->GetTorchGenerator(device);
    auto tau_tensor = torch::rand({ N, 1 }, gen, torch::TensorOptions().device(device));

    // tau_tensor(ランダム)でUQE適用
    return MakeUQEActionInfo(0.0f, tau_tensor, obs, greedy_only, forward, rnd, info, capture, false, nullptr, additional_taus);
}

torch::Tensor ActionPolicy::MakeValueQueryTaus(int64_t batch, const torch::Device& device, anet::RandomGenerator& random) const
{
    if (config_.quantile_mode != ActionPolicyConfig::kQuantileModeIqn) return {};
    // 平均方策はprimaryのrisk範囲によらずfull rangeを問い合わせる。
    return GenerateTaus(batch, config_.tau_rule.num_taus, config_.tau_rule.sample_mode, 0.0f, 1.0f, device, random);
}

torch::Tensor UQEActionPolicy::MakeValueQueryTaus(int64_t batch, const torch::Device& device, anet::RandomGenerator& random) const
{
    if (config_.quantile_mode != ActionPolicyConfig::kQuantileModeIqn) return {};
    if (config_.IsThompsonSampling() && !IsSpatialExplorationEnabled()) {
        return ActionPolicy::MakeValueQueryTaus(batch, device, random);
    }
    // 専用方策の現在tauとlane配置からqueryを作る。point UQEは補間せず指定分位を評価する。
    const auto lower = IsSpatialExplorationEnabled() ? GetSpatialTauTensor(batch, device)
        : torch::full({batch}, current_uqe_tau_, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    return config_.uqe.use_tail_mean
        ? GenerateTaus(lower, 1.0f, config_.tau_rule.num_taus, config_.tau_rule.sample_mode, random)
        : lower.reshape({batch, 1}).expand({batch, config_.tau_rule.num_taus});
}

torch::Tensor ActionPolicy::MakeActionScores(const torch::Tensor& q, const torch::Tensor& distribution,
    bool iqn_risk_query, const torch::Tensor& selection_taus) const
{
    return q;
}

torch::Tensor ActionPolicy::SelectFromValues(const torch::Tensor& q, const torch::Tensor& distribution,
    std::shared_ptr<anet::RandomGenerator> random, bool iqn_risk_query, bool greedy_only, const torch::Tensor& selection_taus) const
{
    const auto greedy = q.argmax(1);
    if (greedy_only) return greedy;
    return IsSpatialExplorationEnabled()
        ? MakeEpsilonGreedyAction(greedy, GetSpatialEpsilonTensor(q.size(0), q.device(), false), q.size(0), q.size(1), random)
        : MakeEpsilonGreedyAction(greedy, current_epsilon_, q.size(0), q.size(1), random);
}

torch::Tensor UQEActionPolicy::MakeActionScores(const torch::Tensor& q, const torch::Tensor& distribution,
    bool iqn_risk_query, const torch::Tensor& selection_taus) const
{
    torch::Tensor score;
    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn && iqn_risk_query) {
        score = q;
    } else if (IsSpatialExplorationEnabled()) {
        score = MakeVectorizedUQEValues(GetSpatialTauTensor(q.size(0), q.device()).unsqueeze(1), distribution);
    } else if (config_.IsThompsonSampling()) {
        ANET_CHECK_MSG(selection_taus.defined(), "Thompson action scores require previously sampled selection_taus.");
        score = MakeVectorizedUQEValues(selection_taus, distribution);
    } else {
        score = MakeUQEValues(current_uqe_tau_, distribution);
    }
    return score;
}

torch::Tensor UQEActionPolicy::SelectFromValues(const torch::Tensor& q, const torch::Tensor& distribution,
    std::shared_ptr<anet::RandomGenerator> random, bool iqn_risk_query, bool greedy_only, const torch::Tensor& selection_taus) const
{
    auto taus = selection_taus;
    if (!taus.defined() && config_.IsThompsonSampling() && !IsSpatialExplorationEnabled()
        && !(config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn && iqn_risk_query))
        taus = torch::rand({q.size(0), 1}, random->GetTorchGenerator(q.device()), q.options());
    const auto greedy = MakeActionScores(q, distribution, iqn_risk_query, taus).argmax(1);
    if (greedy_only) return greedy;
    return IsSpatialExplorationEnabled()
        ? MakeEpsilonGreedyAction(greedy, GetSpatialEpsilonTensor(q.size(0), q.device(), true), q.size(0), q.size(1), random)
        : MakeEpsilonGreedyAction(greedy, current_epsilon_, q.size(0), q.size(1), random);
}

Actor::Actor(std::shared_ptr<ActionPolicy> policy,
    std::shared_ptr<anet::rl::ObservationNormalizer> obs_norm,
    std::unique_ptr<FrameStacker> stacker,
    torch::Device device,
    std::optional<seed_t> seed,
    std::shared_ptr<std::shared_mutex> mutex,
    std::shared_ptr<anet::nn::Network> network,
    std::shared_ptr<anet::nn::Network> src_network,
    const anet::nn::NetworkForwardConfig& forward_config,
    bool emit_actor_q_hint,
    std::optional<anet::ProfiledValueConfig<step_t>> snapshot_sync_interval,
    bool emit_snapshot_metrics,
    ActorQHintConfig actor_q_hint_config, std::optional<SkipActorResources> skip)
    : RandomHolder(seed), policy_(std::move(policy)), obs_norm_(std::move(obs_norm))
    , stacker_(std::move(stacker)), device_(device), mutex_(std::move(mutex))
    , network_(std::move(network)), src_network_(std::move(src_network))
    , emit_actor_q_hint_(emit_actor_q_hint)
    , actor_q_hint_config_(actor_q_hint_config)
    , gamma_(actor_q_hint_config.gamma)
    , intrinsic_gamma_(actor_q_hint_config.intrinsic_gamma)
    , intrinsic_beta_(actor_q_hint_config.beta)
    , munchausen_alpha_(actor_q_hint_config.munchausen.alpha)
    , munchausen_entropy_tau_(actor_q_hint_config.munchausen.entropy_tau)
    , emit_snapshot_metrics_(emit_snapshot_metrics)
    , skip_(std::move(skip))
{
    // 設定の実行可否は計算部品へ委譲し、既存のacting/τ RNGとは独立した箱を作る。
    actor_forward_ = network_->CreateForwardExecutor(forward_config, anet::nn::NetworkForwardClock::kContinuingRows,
            anet::SeedMaker(GetSeed()).MakeNamedSeed("network_forward"));
    if (skip_) {
        const anet::SeedMaker seeds(GetSeed());
        skip_forward_ = skip_->network->CreateForwardExecutor(*skip_->forward_config, anet::nn::NetworkForwardClock::kContinuingRows,
            seeds.MakeNamedSeed(kSkipForwardSeed));
        skip_random_ = std::make_shared<anet::RandomGenerator>(seeds.MakeNamedSeed(kSkipPolicySeed));
        if (skip_->max_length_mode == DQNActorConfig::kMaxLengthVariable) {
            if (!skip_->max_length) ANET_SYSTEM_ERROR("Skip Actor " << skip_->actor_keys.GetFullKey("skip.max_length_mode")
                << "=variable requires " << skip_->actor_keys.GetFullKey("skip.max_length") << ".");
            auto length = *skip_->max_length;
            length.min_value = 1;
            length.max_value = skip_->config.max_length;
            skip_max_length_.emplace(std::move(length));
        }
    }
    if (snapshot_sync_interval.has_value()) {
        snapshot_sync_interval_.emplace(std::move(*snapshot_sync_interval));
    }
}

bool Actor::AttachSyncFollower(std::weak_ptr<ActorSyncFollower> follower)
{
    // source と共有するActorは同期せず、呼び出し側が直接読む契約を選ぶ。
    if (network_ == src_network_) return false;
    sync_followers_.push_back(std::move(follower));
    return true;
}

void Actor::CopySourceNetwork() const
{
    if (network_ == src_network_) {
        return;
    }

    // Learner更新と同じAgent mutexでsourceを保護してsnapshotへ複製する。
    std::shared_lock<std::shared_mutex> lock(*mutex_);
    src_network_->CopyTo(*network_);
    network_->eval();
    if (skip_) {
        skip_->source->CopyTo(*skip_->network);
        skip_->network->eval();
    }
    // network と同じ時点のsourceを読ませ、寿命の切れた相手は取り除く。
    std::erase_if(sync_followers_, [](const auto& weak) {
        const auto follower = weak.lock();
        if (!follower) return true;
        follower->OnActorSync();
        return false;
    });
}

void Actor::UpdateSnapshot(const StepCounts& step) const
{
    if (!snapshot_sync_interval_.has_value()) {
        return;
    }

    // exp_stepで現在周期を評価してから、強制同期後のage基準を現在actionへ合わせる。
    snapshot_sync_interval_->Update(step.exp_step);
    if (reset_snapshot_age_on_next_action_) {
        last_snapshot_sync_train_step_ = step.train_step;
        reset_snapshot_age_on_next_action_ = false;
    }

    // 現在ageが評価済み周期へ到達したaction境界でsnapshotを更新する。
    const auto snapshot_age = step.train_step - last_snapshot_sync_train_step_;
    if (snapshot_age >= snapshot_sync_interval_->Value()) {
        CopySourceNetwork();
        ResetSessionDiagnostics();
        last_snapshot_sync_train_step_ = step.train_step;
    }
}

std::shared_ptr<anet::rl::BatchActionInfo> Actor::MakeAction(const StepCounts& step, const anet::rl::BatchState& state, const anet::rl::MakeActionOptions& options) const
{
    ANET_PROFILE_FUNC();
    torch::NoGradGuard ng;

    // 方策状態はActorの実行境界だけで更新する。
    policy_->UpdateSchedule(step);
    if (skip_ && skip_->policy != policy_) skip_->policy->UpdateSchedule(step);
    if (skip_max_length_) skip_max_length_->Update(step.exp_step);

    // action forwardより前にTrain Actor network snapshotを必要な場合だけ同期する。
    UpdateSnapshot(step);

    // frame stackとdevice転送（stacker無しならdevice転送のみ）を正規化前に行う。
    auto obs = stacker_ ? stacker_->Stack(state.obs, state.episode_start) : state.obs.To(device_);
    ANET_LOG_DEBUG("obs=" << obs.ToDefString());
    //ANET_LOG_DEBUG("obs=" << obs.ToString());

    // Observation正規化
    auto norm_obs = obs;
    if (obs_norm_ != nullptr) {
        norm_obs = obs_norm_->Normalize(obs);
    }
    //norm_obs = norm_obs.to(torch::kCPU);
    ANET_LOG_DEBUG("norm_obs=" << norm_obs.ToDefString());
    //ANET_LOG_DEBUG("norm_obs=" << norm_obs.ToString());

    // 行動選択の時計を進め、同じ選択内の全forwardへ不透明な実行情報を渡す。
    const anet::nn::NetworkForwardCallInfo forward_info({
        .clock = ++forward_clock_, .episode_start = state.episode_start});

    // 行動選択
    std::optional<anet::nn::NetworkForwardCapture> capture;
    if (options.nn_trace) capture.emplace(std::vector<std::string>{std::string(anet::nn::kNnTracePrefix)});
    const auto additional_taus = skip_ && !skip_->shared_policy
        ? skip_->policy->MakeValueQueryTaus(norm_obs.Size(0), device_, *skip_random_) : torch::Tensor();
    const auto dedicated_tau_count = additional_taus.defined() ? additional_taus.size(1) : 0;
    // skip 中も両網を全 lane で forward する（高速化は非ゴール）。Runner へ返す行動だけを継続値にする。
    std::shared_ptr<DQNActionInfo> act_info;
    const float action_beta = actor_q_hint_config_.decomposed ? intrinsic_beta_.Evaluate(step.exp_step) : 0.0f;
    const DqnForwardExecutor action_forward(*actor_forward_, actor_q_hint_config_.decomposed
        ? std::optional<float>(action_beta) : std::nullopt,
        actor_q_hint_config_.use_tbo, actor_q_hint_config_.tbo_epsilon);
    if (network_ != src_network_) {
        // Clone済み: 自分専用のネットワークなので排他不要
        act_info = policy_->SelectAction(norm_obs, false, action_forward, rnd_, forward_info, capture ? &*capture : nullptr, nullptr, additional_taus);
        // 行動 forward の直後に取り出し、skip 側の forward へ持ち越さない。
        if (capture) {
            anet::rl::AppendNnTraceAux(act_info->GetAuxData(), *capture);
            capture.reset();
        }
        if (skip_) act_info = ApplySkip(std::move(act_info), norm_obs, state.episode_start, forward_info, dedicated_tau_count);
    } else {
        // Clone無し（直列モード）: Learnerの更新と競合しないようSharedLock
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        act_info = policy_->SelectAction(norm_obs, false, action_forward, rnd_, forward_info, capture ? &*capture : nullptr, nullptr, additional_taus);
        // 行動 forward の直後に取り出し、skip 側の forward へ持ち越さない。
        if (capture) {
            anet::rl::AppendNnTraceAux(act_info->GetAuxData(), *capture);
            capture.reset();
        }
        if (skip_) act_info = ApplySkip(std::move(act_info), norm_obs, state.episode_start, forward_info, dedicated_tau_count);
    }
    if (actor_q_hint_config_.decomposed) {
        // IQNの追加full queryを、行動のrisk領域と初期優先度の近似へ混ぜない。
        const auto component = [&](const std::string& q_key, const std::string& dist_key) {
            auto distribution = act_info->forward_output.Contains(dist_key) ? act_info->forward_output.At(dist_key) : torch::Tensor();
            if (distribution.defined() && act_info->risk_num_taus > 0)
                distribution = distribution.narrow(2, 0, act_info->risk_num_taus);
            const auto values = distribution.defined() ? distribution.mean(2) : act_info->forward_output.At(q_key);
            return std::pair(values, distribution);
        };
        const auto [external_q, external_dist] = component(DecomposedHeadFactory::kExtrinsicQOutputKey, DecomposedHeadFactory::kExtrinsicQDistOutputKey);
        const auto [internal_q, internal_dist] = component(DecomposedHeadFactory::kIntrinsicQOutputKey, DecomposedHeadFactory::kIntrinsicQDistOutputKey);
        act_info->GetAuxData()[kExtrinsicQValues] = external_q;
        act_info->GetAuxData()[kIntrinsicQValues] = internal_q;
        act_info->GetAuxData()[kExtrinsicActionScores] = policy_->MakeActionScores(external_q, external_dist, true, act_info->selection_taus);
        act_info->GetAuxData()[kCombinedActionScores] = policy_->MakeActionScores(act_info->GetAuxData().at(kActionQValuesAuxKey),
            act_info->GetAuxData().at(kActionQQuantilesAuxKey), true, act_info->selection_taus);
        auto diagnostics = MakeValueSelectionDiagnostics(act_info->forward_output,
            action_beta, actor_q_hint_config_.use_tbo, actor_q_hint_config_.tbo_epsilon);
        act_info->GetAuxData()[kValueSelectionDiagnostics] = diagnostics;
        // Actorと同じdeviceでlane×stepを足し、session末の読出しまでCPUへ戻さない。
        const auto sums = diagnostics.sum(0);
        if (!value_selection_session_sums_.defined()) value_selection_session_sums_ = sums;
        else value_selection_session_sums_.add_(sums);
        value_selection_session_lane_steps_ += diagnostics.size(0);
        value_selection_session_cpu_ = torch::Tensor();
    }
    act_info->forward_output = {};
    act_info->effective_gamma = gamma_.Evaluate(step.exp_step);
    if (actor_q_hint_config_.decomposed) act_info->effective_intrinsic_gamma = intrinsic_gamma_.Evaluate(step.exp_step);
    act_info->munchausen_alpha = munchausen_alpha_.Evaluate(step.exp_step);
    act_info->munchausen_entropy_tau = munchausen_entropy_tau_.Evaluate(step.exp_step);

    // 上書き後の行動で既存 forward の平均 Q を gather し、学習 Actor の初期優先度ヒントを作る。
    if (emit_actor_q_hint_) {
        ANET_PROFILE_SCOPE(actor_q_hint);
        
        const auto q_it = act_info->GetAuxData().find("q_values");
        if (q_it == act_info->GetAuxData().end() || !q_it->second.defined()) {
            ANET_SYSTEM_ERROR("actor_approx requires ActionPolicy aux[q_values].");
        }
        const auto& q_values = q_it->second;
        auto hint = MakeActorQHint(q_values, act_info->GetAction(), ActorQHintValues{
            .munchausen = actor_q_hint_config_.munchausen.Resolve(
                act_info->munchausen_alpha, act_info->munchausen_entropy_tau),
            .use_tbo = actor_q_hint_config_.use_tbo, .tbo_epsilon = actor_q_hint_config_.tbo_epsilon,
            .extrinsic_q = actor_q_hint_config_.decomposed ? act_info->GetAuxData().at(kExtrinsicQValues) : torch::Tensor(),
            .intrinsic_q = actor_q_hint_config_.decomposed ? act_info->GetAuxData().at(kIntrinsicQValues) : torch::Tensor(),
            .extrinsic_scores = actor_q_hint_config_.decomposed ? act_info->GetAuxData().at(kExtrinsicActionScores) : torch::Tensor(),
            .combined_scores = actor_q_hint_config_.decomposed ? act_info->GetAuxData().at(kCombinedActionScores) : torch::Tensor(),
            .extrinsic_bootstrap = actor_q_hint_config_.extrinsic_bootstrap});
        act_info->GetAuxData()["munchausen_terms"] = hint.munchausen_terms;
        act_info->SetReplayInitialPriorityHint(ReplayInitialPriorityHint(std::move(hint.payload)));
    }

    // AuxData の詰め込み
    // 現在actionがepisode開始直後かを、device転送せず診断用auxへ保持する。
    act_info->GetAuxData()["episode_start"] = state.episode_start;
    act_info->GetAuxData()["raw_obs"] = anet::rl::ToUnifiedObservation(obs);
    if (obs_norm_ != nullptr) {
        act_info->GetAuxData()["norm_obs"] = anet::rl::ToUnifiedObservation(norm_obs);
    }

    // DefaultDQNでは無効時もNaN、周期snapshot有効時は同期後の現在値を公開する。
    if (emit_snapshot_metrics_) {
        TrainActorSnapshotMetrics metrics;
        if (snapshot_sync_interval_.has_value()) {
            metrics.interval = static_cast<float>(snapshot_sync_interval_->Value());
            metrics.age = static_cast<float>(step.train_step - last_snapshot_sync_train_step_);
        }
        act_info->SetTrainActorSnapshotMetrics(metrics);
    }

    return act_info;
}

std::shared_ptr<DQNActionInfo> Actor::ApplySkip(std::shared_ptr<DQNActionInfo> proposal,
    const anet::TensorDict& obs, const torch::Tensor& episode_start,
    const anet::nn::NetworkForwardCallInfo& info, int64_t dedicated_tau_count) const
{
    ANET_PROFILE_FUNC();

    const auto proposed_actions = proposal->GetAction();
    const bool single = skip_->max_length_mode == DQNActorConfig::kMaxLengthSingle;

    // lane ごとに、続けている行動・残り step 数・選んだ j を保持する。
    if (!single && !skip_remaining_.defined()) {
        skip_remaining_ = torch::zeros_like(proposed_actions);
        skip_actions_ = torch::zeros_like(proposed_actions);
        skip_lengths_ = torch::ones_like(proposed_actions);
    }

    // Action形状チェック
    if (!single && skip_remaining_.sizes() != proposed_actions.sizes()) {
        ANET_SYSTEM_ERROR("Skip Actor lane count changed: actual=" << proposed_actions.sizes() << " expected=" << skip_remaining_.sizes());
    }

    // episode 開始時と残り 0 の lane だけ、新しい行動と j を判断する。
    const auto decision = single ? torch::ones_like(proposed_actions, torch::kBool)
        : episode_start.to(proposed_actions.device()).logical_or(skip_remaining_.eq(0));
    // 継続中の lane は今回の提案を捨て、前に選んだ行動で上書きする。
    const auto actions = single ? proposed_actions : torch::where(decision, proposed_actions, skip_actions_);

    // SKIPネットワークでforward
    const auto input = MakeSkipInput(obs, proposal->forward_output, actions, true);
    anet::Autocast amp(proposed_actions.device(), skip_->use_amp, skip_->use_amp_bf16 ? torch::kBFloat16 : torch::kHalf);
    const auto output = skip_forward_->Forward(input, info);

    // SKIP-Qを取り出す
    auto values = ReadSkipValues(output, proposal->forward_output, actions, skip_->config);
    auto q = values.At("q");
    auto distribution = values.Get("q_dist").value_or(torch::Tensor());
    // IQN の専用 query は末尾、共有 risk query は先頭を選ぶ。Dueling も同じ分位で合成済み。
    if (dedicated_tau_count > 0) {
        distribution = distribution.narrow(-1, distribution.size(-1) - dedicated_tau_count, dedicated_tau_count);
        q = distribution.mean(-1);
    } else if (skip_->shared_policy && proposal->risk_num_taus > 0 && distribution.defined()) {
        distribution = distribution.narrow(-1, 0, proposal->risk_num_taus);
        q = distribution.mean(-1);
    }

    // SKIP-Qをもとに、skip_policyを使ってSKIP長を決定
    // 上限の絞りは選択用のviewだけに適用し、診断のJ本の値は保持する。
    const auto selection_q = skip_max_length_ ? q.narrow(1, 0, skip_max_length_->Value()) : q;
    const auto selection_distribution = skip_max_length_ && distribution.defined()
        ? distribution.narrow(1, 0, skip_max_length_->Value()) : distribution;
    const auto lengths = single ? torch::ones_like(proposed_actions)
        : skip_->policy->SelectFromValues(selection_q, selection_distribution, skip_random_) + 1;
    // 判断 lane だけ j を更新し、今回の 1 step を引いてから返す情報を組み立てる。
    if (!single) {
        skip_lengths_ = torch::where(decision, lengths, skip_lengths_);
        skip_remaining_ = torch::where(decision, lengths - 1, skip_remaining_ - 1);
        skip_actions_ = actions;
    }

    // 判断laneだけをdevice上で累積する。CPU転送はGetScalar要求時まで行わない。
    ANET_PROFILE_SCOPE(skip_session);
    const auto durations = torch::arange(1, skip_->config.max_length + 1, lengths.options());
    const auto counts = lengths.unsqueeze(1).eq(durations).logical_and(decision.unsqueeze(1)).sum(0);
    if (!skip_session_counts_.defined()) skip_session_counts_ = counts;
    else skip_session_counts_.add_(counts);
    skip_session_lane_steps_ += lengths.size(0);
    skip_session_counts_cpu_ = torch::Tensor();

    // info.decision は replay の判断境界、aux 3 つは scalar と Trace の診断へ渡す。
    auto result = std::make_shared<DQNActionInfo>(actions, proposal->GetInfo(), proposal->GetAuxData());
    result->GetInfo().Set(anet::rl::BatchActionInfo::kKeyDecision, decision);
    result->GetAuxData()[DQNActionInfo::kSkipDecision] = decision;
    result->GetAuxData()[DQNActionInfo::kSkipLength] = single ? lengths : skip_lengths_;
    result->GetAuxData()[DQNActionInfo::kSkipQValues] = q;
    return result;
}

void Actor::Sync()
{
    CopySourceNetwork();
    ResetSessionDiagnostics();
    if (snapshot_sync_interval_.has_value()) {
        // Syncはstepを受け取らないため、次actionのtrain_stepを新しいage基準にする。
        reset_snapshot_age_on_next_action_ = true;
    }
}

void Actor::ResetSessionDiagnostics() const
{
    skip_session_counts_ = torch::Tensor();
    skip_session_counts_cpu_ = torch::Tensor();
    skip_session_lane_steps_ = 0;
    value_selection_session_sums_ = torch::Tensor();
    value_selection_session_cpu_ = torch::Tensor();
    value_selection_session_lane_steps_ = 0;
}

std::optional<float> Actor::GetScalar(const std::string& key, int64_t index) const
{
    for (size_t metric = 0; metric < kValueSelectionMetricCount; ++metric) {
        if (key != kValueSelectionSessionKeys[metric]) continue;
        if (value_selection_session_lane_steps_ == 0) return std::numeric_limits<float>::quiet_NaN();
        if (!value_selection_session_cpu_.defined()) value_selection_session_cpu_ = value_selection_session_sums_.to(torch::kCPU);
        return value_selection_session_cpu_[static_cast<int64_t>(metric)].item<float>() / value_selection_session_lane_steps_;
    }
    if (key == kSkipMaxLength) {
        return skip_max_length_ ? static_cast<float>(skip_max_length_->Value())
            : std::numeric_limits<float>::quiet_NaN();
    }
    const bool length_rate = key.starts_with(kSkipSessionLengthRatePrefix);
    if (!length_rate && key != kSkipSessionDecisionRate && key != kSkipSessionLengthMean) {
        return policy_->GetScalar(key, index);
    }
    const auto duration = length_rate ? ParseActionScalarIndex(key, kSkipSessionLengthRateIndexPrefix) : 1;
    if (duration < 1 || (skip_ && duration > skip_->config.max_length)) {
        ANET_SYSTEM_ERROR("Actor: invalid key=" << key << " duration=" << duration
            << "; expected duration >= 1 and <= "
            << (skip_ ? skip_->config.full_keys.GetFullKey("max_length") : "max_length (not read from config)")
            << "=" << (skip_ ? skip_->config.max_length : 0) << ".");
    }
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    if (!skip_ || skip_session_lane_steps_ == 0) return nan;

    // 同じsessionの各指標は一度の読み戻しを共用し、CPU上で割合と平均にする。
    ANET_PROFILE_SCOPE(skip_session_readback);
    if (!skip_session_counts_cpu_.defined()) skip_session_counts_cpu_ = skip_session_counts_.to(torch::kCPU);
    const auto* counts = skip_session_counts_cpu_.data_ptr<int64_t>();
    int64_t decisions = 0, length_sum = 0;
    for (int64_t j = 0; j < skip_->config.max_length; ++j) {
        decisions += counts[j];
        length_sum += (j + 1) * counts[j];
    }
    if (key == kSkipSessionDecisionRate) return static_cast<float>(decisions) / skip_session_lane_steps_;
    if (decisions == 0) return nan;
    return static_cast<float>(length_rate ? counts[duration - 1] : length_sum) / decisions;
}
