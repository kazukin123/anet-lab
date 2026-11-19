#include "dqn_based_assembler.hpp"
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

constexpr char kStepQSaKey[] = "q_sa";

constexpr char kStepStateValueKey[] = "state_value";

constexpr char kStepExtrinsicQSaKey[] = "extrinsic_q_sa";

constexpr char kStepExtrinsicStateValueKey[] = "extrinsic_state_value";

namespace {

struct DqnDiagnosticRow {
    float q_sa = std::numeric_limits<float>::quiet_NaN();
    float state_value = std::numeric_limits<float>::quiet_NaN();
    float approx_priority = std::numeric_limits<float>::quiet_NaN();
    float extrinsic_q_sa = std::numeric_limits<float>::quiet_NaN();
    float intrinsic_q_sa = std::numeric_limits<float>::quiet_NaN();
    float extrinsic_state_value = std::numeric_limits<float>::quiet_NaN();
    float intrinsic_state_value = std::numeric_limits<float>::quiet_NaN();
};

// DQN固有のキーと値を所有し、共通の診断契約へはModuleだけを公開する。
class DqnAgentStepDiagnosticsValues final : public anet::ModuleBase {
public:
    explicit DqnAgentStepDiagnosticsValues(std::vector<DqnDiagnosticRow> rows) : rows_(std::move(rows)) {}
    std::optional<float> GetScalar(const std::string& key, int64_t lane = -1) const override
    {
        float DqnDiagnosticRow::* member = nullptr;
        if (key == kStepQSaKey) member = &DqnDiagnosticRow::q_sa;
        else if (key == kStepStateValueKey) member = &DqnDiagnosticRow::state_value;
        else if (key == DqnComponentAssembler::kStepApproxPriorityKey) member = &DqnDiagnosticRow::approx_priority;
        else if (key == kStepExtrinsicQSaKey) member = &DqnDiagnosticRow::extrinsic_q_sa;
        else if (key == kStepIntrinsicQSaKey) member = &DqnDiagnosticRow::intrinsic_q_sa;
        else if (key == kStepExtrinsicStateValueKey) member = &DqnDiagnosticRow::extrinsic_state_value;
        else if (key == kStepIntrinsicStateValueKey) member = &DqnDiagnosticRow::intrinsic_state_value;
        else return std::nullopt;
        ANET_CHECK_MSG(lane >= 0 && lane < static_cast<int64_t>(rows_.size()),
            "Invalid DQN diagnostic lane. key='" << key << "' lane=" << lane << " expected=[0," << rows_.size() << ").");
        return rows_[lane].*member;
    }
private:
    std::vector<DqnDiagnosticRow> rows_;
};

class DqnAgentStepDiagnosticsProvider final : public anet::rl::AgentStepDiagnosticsProvider {
    struct Pending {
        int64_t start;
        anet::rl::ReplayExperience experience;
        anet::rl::ExperienceSequence sequence;
        anet::rl::ReplayInitialPriorityHintRow hint;
    };
    struct Lane {
        int64_t step = 0;
        anet::rl::ExperienceQueue queue;
        std::deque<Pending> pending;
    };
public:
    DqnAgentStepDiagnosticsProvider(const LearnerConfig& config, int64_t num_lanes,
        std::function<torch::Tensor(const torch::Tensor&)> scale_reward, bool needs_step_priority)
        : config_(config), num_lanes_(num_lanes), scale_reward_(std::move(scale_reward)), needs_step_priority_(needs_step_priority)
        , controller_(anet::rl::CreateNStepQueueController(config.use_n_step ? config.n_step : 1))
        , builder_(anet::rl::CreateDefaultExperienceBuilder())
        , estimator_(CreateInitialPriorityEstimator(config))
    {
        ANET_CHECK_MSG(num_lanes > 0 && (!config.use_n_step || config.n_step > 0),
            "Step diagnostics require positive lanes and n_step. lanes=" << num_lanes << " n_step=" << config.n_step);
        Reset();
    }

    void Reset() override
    {
        // 再表示は途中のepisodeからでも新しい窓として扱う。
        lanes_.clear();
        lanes_.resize(static_cast<size_t>(num_lanes_));
    }

    anet::rl::AgentStepDiagnosticsResult Observe(const anet::rl::BatchExperience& experience) override
    {
        ANET_PROFILE_FUNC();
        torch::NoGradGuard no_grad;
        // Actorが行動時に評価したscalarを既存GetScalar経路で受け取り、収益とhintのγを揃える。
        const auto scalar = [&](size_t index) {
            const auto* key = kLearnerScheduleMetricKeys[index];
            const auto value = experience.action->GetScalar(key);
            ANET_CHECK_MSG(value && std::isfinite(*value), "DQN step diagnostics require an evaluated action scalar. key=" << key
                << " value=" << (value ? std::to_string(*value) : "unavailable") << " expected=finite");
            return *value;
        };
        const auto gamma = scalar(3);
        std::vector<float> gammas{gamma};
        if (config_.intrinsic.decomposed) {
            const auto value = experience.action->GetScalar(DQNActionInfo::kIntrinsicGamma);
            ANET_CHECK_MSG(value && std::isfinite(*value), "DQN step diagnostics require key=" << DQNActionInfo::kIntrinsicGamma
                << " value=" << (value ? std::to_string(*value) : "unavailable") << " expected=finite.");
            gammas.push_back(*value);
        }
        const bool scheduled = config_.gamma.type != kConstantScheduleType
            || (config_.intrinsic.decomposed && config_.intrinsic.discount_mode == LearnerConfig::kDiscountSeparate && config_.intrinsic.gamma.type != kConstantScheduleType);
        const auto& aux = experience.action->GetAuxData();
        const auto q = aux.find("q_values");
        ANET_CHECK_MSG(q != aux.end() && q->second.defined() && q->second.dim() == 2
            && q->second.size(0) == num_lanes_,
            "DQN step diagnostics require aux[q_values] with shape [lanes, actions]. expected_lanes=" << num_lanes_);
        // carrierの既存hintは使わず、強制後のactionと現在Qから必ず作り直す。
        const ActorQHintValues hint_config{ .munchausen = config_.munchausen.Resolve(
            scalar(4), scalar(5)),
            .use_tbo = config_.use_tbo, .tbo_epsilon = config_.tbo_epsilon };
        auto hint = MakeActorQHint(q->second, experience.action->GetAction(q->second.device()), hint_config)
            .payload.detach().to(torch::kCPU).contiguous();
        torch::Tensor component_hint;
        if (config_.intrinsic.decomposed) {
            auto component_config = hint_config;
            component_config.extrinsic_q = aux.at(kExtrinsicQValues);
            component_config.intrinsic_q = aux.at(kIntrinsicQValues);
            component_config.extrinsic_scores = aux.at(kExtrinsicActionScores);
            component_config.combined_scores = aux.at(kCombinedActionScores);
            component_config.extrinsic_bootstrap = config_.intrinsic.extrinsic_bootstrap_mode == LearnerConfig::kBootstrapExtrinsic;
            component_hint = MakeActorQHint(q->second, experience.action->GetAction(q->second.device()), component_config)
                .payload.detach().to(torch::kCPU).contiguous();
        }
        // Q列のみの要求では、未要求の報酬・系列窓・priorityを計算しない。
        auto rewards = needs_step_priority_ ? scale_reward_(experience.reward).to(torch::kCPU).contiguous() : torch::Tensor();
        const auto done = needs_step_priority_ ? experience.next_state.done.to(torch::kCPU).contiguous() : torch::Tensor();
        const auto truncated = needs_step_priority_ ? experience.next_state.truncated.to(torch::kCPU).contiguous() : torch::Tensor();
        const auto starts = needs_step_priority_ ? experience.state.episode_start.to(torch::kCPU).contiguous() : torch::Tensor();
        ANET_CHECK_MSG(!needs_step_priority_ || (rewards.dim() == (config_.intrinsic.decomposed ? 2 : 1) && rewards.size(0) == num_lanes_
            && (!config_.intrinsic.decomposed || rewards.size(1) == 2)),
            "DQN step diagnostics reward shape mismatch. expected_lanes=" << num_lanes_);
        anet::rl::AgentStepDiagnosticsResult result;
        std::vector<DqnDiagnosticRow> rows(static_cast<size_t>(num_lanes_));
        result.pending_keys.assign(static_cast<size_t>(num_lanes_), needs_step_priority_ ? std::vector<std::string>{DqnComponentAssembler::kStepApproxPriorityKey} : std::vector<std::string>{});
        for (int64_t i = 0; i < num_lanes_; ++i) {
            auto& lane = lanes_[i];
            anet::rl::ReplayInitialPriorityHintRow current;
            for (int col = 0; col < kActorQHintColumnCount; ++col) current.push_back(hint[i][col].item<float>());
            rows[i].q_sa = current[kActorQSaColumn];
            rows[i].state_value = current[kActorStateValueColumn];
            if (component_hint.defined()) {
                const auto values = component_hint.accessor<float, 2>();
                rows[i].extrinsic_q_sa = values[i][kActorQSaColumn];
                rows[i].intrinsic_q_sa = values[i][kActorIntrinsicQSaColumn];
                rows[i].extrinsic_state_value = values[i][kActorStateValueColumn];
                rows[i].intrinsic_state_value = values[i][kActorIntrinsicStateValueColumn];
                for (int col = 0; col < kActorQHintColumnCount; ++col) current[col] = values[i][col];
            }
            if (!needs_step_priority_) continue;
            const auto complete = [&](const Pending& pending, bool unavailable) {
                float priority = std::numeric_limits<float>::quiet_NaN();
                if (!unavailable) {
                    // 定数γは保存済み収益を使い、schedule γは現在のbootstrapと同じγで逆順和を再計算する。
                    auto effective = scheduled ? builder_->Build(pending.sequence, gammas) : pending.experience;
                    if (config_.gamma.type == kConstantScheduleType) effective.target_return = pending.experience.target_return;
                    if (config_.intrinsic.decomposed && (config_.intrinsic.discount_mode == LearnerConfig::kDiscountShared
                        ? config_.gamma.type == kConstantScheduleType : config_.intrinsic.gamma.type == kConstantScheduleType))
                        effective.extra_component_returns = pending.experience.extra_component_returns;
                    std::vector<float> discounts;
                    for (size_t component = 1; component < gammas.size(); ++component)
                        discounts.push_back(std::pow(gammas[component], effective.actual_n_steps));
                    const auto estimated = estimator_->Estimate(anet::rl::InitialPriorityEstimateInput{
                        .start_hint = std::span<const float>(pending.hint.data(), pending.hint.size()),
                        .bootstrap_hint = effective.terminal ? std::span<const float>{}
                            : std::span<const float>(current.data(), current.size()),
                        .target_return = effective.target_return,
                        .discount = static_cast<float>(std::pow(gamma, effective.actual_n_steps)),
                        .extra_component_returns = effective.extra_component_returns,
                        .extra_component_discounts = discounts,
                        .terminal = effective.terminal,
                        .actual_n_steps = effective.actual_n_steps });
                    if (estimated) priority = *estimated;
                }
                result.completed.push_back({ .lane = i, .steps_back = lane.step - pending.start,
                    .resolved_keys = {DqnComponentAssembler::kStepApproxPriorityKey}, .values = std::make_unique<DqnAgentStepDiagnosticsValues>(
                        std::vector<DqnDiagnosticRow>{{ .approx_priority = priority }}) });
            };
            // episode開始と将来Qの到着を、現在rewardを窓へ入れる前に処理する。
            if (starts[i].item<bool>()) {
                for (const auto& pending : lane.pending) complete(pending, true);
                lane.pending.clear();
                lane.queue.Clear();
            }
            while (!lane.pending.empty()) {
                const auto& pending = lane.pending.front();
                const auto expected = pending.start + pending.experience.actual_n_steps;
                if (expected > lane.step) break;
                ANET_CHECK_MSG(expected == lane.step, "Step diagnostics bootstrap mismatch. expected=" << expected << " actual=" << lane.step);
                complete(pending, false);
                lane.pending.pop_front();
            }
            // 系列の確定と逆順割引和はreplayの同じ部品へ委譲する。
            lane.queue.Push(anet::rl::QueueRecord{ .time_idx = lane.step, .logical_time_idx = lane.step,
                .reward = config_.intrinsic.decomposed ? rewards[i][0].item<float>() : rewards[i].item<float>(),
                .done = done[i].item<bool>(), .truncated = truncated[i].item<bool>(), .replay_initial_priority_hint = current,
                .extra_component_rewards = config_.intrinsic.decomposed ? std::vector<float>{rewards[i][1].item<float>()} : std::vector<float>{} });
            for (const auto& sequence : controller_->ExtractSequences(lane.queue)) {
                Pending pending{ .start = sequence.front().logical_time_idx,
                    .experience = builder_->Build(sequence, gammas),
                    // 後でγを評価し直すscheduleだけ系列を保持し、定数経路のコピーを増やさない。
                    .sequence = scheduled ? sequence : anet::rl::ExperienceSequence{}, .hint = *sequence.front().replay_initial_priority_hint };
                if (pending.experience.terminal || pending.experience.truncated) {
                    complete(pending, pending.experience.truncated);
                } else {
                    lane.pending.push_back(std::move(pending));
                }
            }
            ++lane.step;
        }
        result.current = std::make_unique<DqnAgentStepDiagnosticsValues>(std::move(rows));
        return result;
    }
private:
    LearnerConfig config_;
    int64_t num_lanes_;
    std::function<torch::Tensor(const torch::Tensor&)> scale_reward_;
    bool needs_step_priority_;
    std::unique_ptr<anet::rl::ExperienceQueueController> controller_;
    std::unique_ptr<anet::rl::ReplayExperienceBuilder> builder_;
    std::unique_ptr<anet::rl::InitialPriorityEstimator> estimator_;
    std::vector<Lane> lanes_;
};

} // namespace

std::unique_ptr<anet::rl::AgentStepDiagnosticsProvider> anet::rl::dqn::CreateStepDiagnosticsProvider(
    const LearnerConfig& config, int64_t num_lanes,
    std::function<torch::Tensor(const torch::Tensor&)> scale_reward, bool needs_step_priority)
{
    return std::make_unique<DqnAgentStepDiagnosticsProvider>(config, num_lanes, std::move(scale_reward), needs_step_priority);
}

std::shared_ptr<anet::nn::NetworkHeadFactory> DqnComponentAssembler::CreateHeadFactory(
    const std::string& quantile_mode, bool dueling, int64_t num_actions, int num_quantiles,
    const anet::nn::WeightInitConfig& init, const anet::ConfigData& projection)
{
    std::shared_ptr<anet::nn::NetworkHeadFactory> head_factory;
    // 分布表現とDuelingの直積でHeadを選択する。
    if (quantile_mode == ActionPolicyConfig::kQuantileModeIqn) {
        if (dueling) {
            head_factory = std::make_shared<IQNDuelingHeadFactory>(num_actions, init, projection);
            LOG::info() << "Network Head: IQN Dueling";
        } else {
            head_factory = std::make_shared<IQNHeadFactory>(num_actions, init, projection);
            LOG::info() << "Network Head: IQN Plain";
        }
    } else if (quantile_mode == ActionPolicyConfig::kQuantileModeQr) {
        if (dueling) {
            head_factory = std::make_shared<QuantileDuelingHeadFactory>(
                num_actions, num_quantiles, init, projection);
            LOG::info() << "Network Head: Quantile Dueling (N=" << num_quantiles << ")";
        } else {
            head_factory = std::make_shared<QuantileHeadFactory>(
                num_actions, num_quantiles, init, projection);
            LOG::info() << "Network Head: Quantile Plain (N=" << num_quantiles << ")";
        }
    } else {
        if (dueling) {
            head_factory = std::make_shared<DuelingHeadFactory>(
                num_actions, init, projection);
            LOG::info() << "Network Head: Dueling";
        } else {
            head_factory = std::make_shared<LinearHeadFactory>(
                num_actions, init, projection);
            LOG::info() << "Network Head: Plain Linear";
        }
    }

    return head_factory;
}

std::shared_ptr<ActionPolicy> DqnComponentAssembler::CreateActionPolicy(
    const ActionPolicyConfig& policy_config, const std::string& quantile_mode,
    bool enable_spatial_exploration, int64_t num_envs, const torch::Device& device)
{
    if (policy_config.policy_type == "EpsilonGreedy" || policy_config.policy_type == "0") {
        // ε-Greedy
        return std::make_shared<EpsilonGreedyActionPolicy>(policy_config, enable_spatial_exploration, num_envs, device);
    } else if (policy_config.policy_type == "UQE" || policy_config.policy_type == "1") {
        // UQE
        if (quantile_mode == ActionPolicyConfig::kQuantileModeNone) {
            ANET_SYSTEM_ERROR("Invalid action policy: policy_type='" << policy_config.policy_type
                << "' requires quantile_mode=" << ActionPolicyConfig::kQuantileModeQr
                << " or " << ActionPolicyConfig::kQuantileModeIqn << ", actual='" << quantile_mode << "'");
        }
        return std::make_shared<UQEActionPolicy>(policy_config, enable_spatial_exploration, num_envs, device);
    } else if (policy_config.IsThompsonSampling()) {
        //ThompsonSampling
        if (quantile_mode == ActionPolicyConfig::kQuantileModeNone) {
            ANET_SYSTEM_ERROR("Invalid action policy: policy_type='" << policy_config.policy_type
                << "' requires quantile_mode=" << ActionPolicyConfig::kQuantileModeQr
                << " or " << ActionPolicyConfig::kQuantileModeIqn << ", actual='" << quantile_mode << "'");
        }
        return std::make_shared<ThompsonSamplingActionPolicy>(policy_config, enable_spatial_exploration, num_envs, device);
    } else if (policy_config.policy_type == kActionPolicyTypeStr_Greedy || policy_config.policy_type == "3") {
        // Greedyは、EpsilonGreedyのノイズ0としてインスタンス化
        ActionPolicyConfig greedy_cfg = policy_config;
        greedy_cfg.epsilon = {.value = 0.0f, .min_value = 0.0f, .max_value = 1.0f};
        greedy_cfg.spatial.enabled = false;
        return std::make_shared<EpsilonGreedyActionPolicy>(greedy_cfg, false, num_envs, device);
    }

    // 不明なtype
    ANET_SYSTEM_ERROR("Unknown action policy type: " << policy_config.policy_type);
    return nullptr;
}

std::shared_ptr<Learner> DqnComponentAssembler::CreateLearner(
    const std::string& quantile_mode, int num_quantiles, const LearnerConfig& config,
    NetworkModel& model, RuntimeVars& vars,
    const BatchEnvSpec& batch, const EnvSpec& env, const torch::Device& device,
    anet::seed_t replay_seed, anet::seed_t agent_seed, std::shared_ptr<ActionPolicy> target_policy,
    std::shared_ptr<ObservationNormalizer> normalizer, const LearnerForwardResources& forwards,
    std::optional<StuckerConfig> stacker, std::optional<anet::seed_t> learner_seed,
    std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe, const std::string& log_suffix)
{
    // 型ごとの分岐からも同じ引数列を通し、構築時の入力を1か所にまとめる。
    const auto create = [&]<typename ConcreteLearner>() -> std::shared_ptr<Learner> {
        return std::make_shared<ConcreteLearner>(config, model, vars, normalizer, batch, env, device,
            replay_seed, agent_seed, target_policy, forwards, stacker, learner_seed, noisy_probe);
    };
    std::shared_ptr<Learner> learner;
    if (quantile_mode == ActionPolicyConfig::kQuantileModeIqn) {
        learner = create.operator()<IQNLearner>();
        LOG::info() << "Initialized IQNLearner (current_taus=" << config.iqn.current_taus.num_taus
            << ", target_taus=" << config.iqn.target_taus.num_taus << ")" << log_suffix;
    } else if (quantile_mode == ActionPolicyConfig::kQuantileModeQr) {
        learner = create.operator()<QRLearner>();
        LOG::info() << "Initialized QRLearner (Quantiles=" << num_quantiles << ")" << log_suffix;
    } else {
        learner = create.operator()<TDLearner>();
        LOG::info() << "Initialized TDLearner" << log_suffix;
    }
    return learner;
}

std::shared_ptr<Actor> DqnComponentAssembler::CreateActor(
    const DQNActorConfig& config, const LearnerConfig& learner_config,
    const ActorRequest& request, NetworkModel& model,
    const std::shared_ptr<std::shared_mutex>& mutex, std::shared_ptr<ActionPolicy> policy,
    const anet::nn::NetworkForwardConfig& forward,
    std::shared_ptr<ObservationNormalizer> normalizer, std::unique_ptr<FrameStacker> stacker,
    bool emit_snapshot_metrics, ActorQHintConfig hint,
    std::optional<anet::ProfiledValueConfig<step_t>> sync_interval,
    std::optional<SkipActorResources> skip)
{
    auto source = config.network == "target" ? model.GetTargetNetwork() : model.GetOnlineNetwork();
    auto network = source;
    if (config.clone_model) {
        // source の parameter と buffer を学習更新から保護して複製する。
        std::shared_lock lock(*mutex);
        network = source->Clone(request.device);
        network->eval();
        if (skip) {
            skip->network = skip->source->Clone(request.device);
            skip->network->eval();
        }
    }
    const bool emit_hint = learner_config.use_per
        && ParseReplayInitialPriorityMode(learner_config) == ReplayInitialPriorityMode::ACTOR_APPROX;
    return std::make_shared<Actor>(std::move(policy), std::move(normalizer), std::move(stacker),
        request.device, request.seed, mutex, network, source, forward, emit_hint,
        sync_interval, emit_snapshot_metrics, hint, std::move(skip));
}

std::optional<float> DqnComponentAssembler::GetScalar(
    const std::string& key, int64_t index, const Learner& learner, const RuntimeVars& vars,
    const ActionPolicy& target_policy, std::shared_mutex& mutex)
{
    constexpr char kTargetPolicyPrefix[] = "target_policy.";
    constexpr std::string_view kNoisyPrefix = "noisy_";
    if (key.starts_with(kTargetPolicyPrefix))
        return target_policy.GetScalar(anet::RemovePrefix(key, kTargetPolicyPrefix), index);
    for (const auto* schedule_key : kLearnerScheduleMetricKeys) {
        if (key == schedule_key) {
            std::shared_lock lock(mutex);
            return learner.GetScalar(key);
        }
    }
    if (std::ranges::find(kValueDecompositionMetricKeys, key) != kValueDecompositionMetricKeys.end()) {
        std::shared_lock lock(mutex);
        return learner.GetScalar(key, index);
    }
    if (key == "per_beta") {
        std::shared_lock lock(mutex);
        return vars.per_beta;
    }
    if (key.starts_with(kPlasticityProbePrefix) || key.starts_with(kNoisyPrefix)
        || key.starts_with(ReplayBuffer::kKeyPrefix)) {
        std::shared_lock lock(mutex);
        return learner.GetScalar(key);
    }
    return std::nullopt;
}

std::optional<torch::Tensor> DqnComponentAssembler::GetTensor(
    const std::string& key, const Learner& learner, std::shared_mutex& mutex)
{
    if (!key.starts_with(ReplayBuffer::kKeyPrefix)) return std::nullopt;
    std::shared_lock lock(mutex);
    return learner.GetTensor(key);
}

std::optional<std::vector<torch::Tensor>> DqnComponentAssembler::GetTensorVector(
    const std::string& key, const Learner& learner, std::shared_mutex& mutex)
{
    if (!key.starts_with(ReplayBuffer::kKeyPrefix)) return std::nullopt;
    std::shared_lock lock(mutex);
    return learner.GetTensorVector(key);
}
