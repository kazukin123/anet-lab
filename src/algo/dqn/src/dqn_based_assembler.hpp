#pragma once
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <limits>
#include <span>
#include <string_view>
#include <unordered_set>
#include <utility>
#include "anet/random.hpp"
#include "anet/stacker.hpp"
#include "anet/schedule.hpp"
#include "anet/nn_util.hpp"
#include "anet/nn.hpp"
#include "anet/rl.hpp"
#include "anet/scaler.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/agent.hpp"
#include "anet/transfer.hpp"
#include "dqn_based_learner.hpp"
#include "dqn_based_actor.hpp"
#include "dqn_based_network_model.hpp"
#include "dqn_based_math.hpp"

namespace anet::rl::dqn {

    std::unique_ptr<AgentStepDiagnosticsProvider> CreateStepDiagnosticsProvider(const LearnerConfig& config,
        int64_t num_lanes, std::function<torch::Tensor(const torch::Tensor&)> scale_reward, bool needs_step_priority = true);

    class DqnComponentAssembler {
    public:
        /// Agentのstep診断で要求・完了を対応付ける鍵。
        static constexpr char kStepApproxPriorityKey[] = "approx_priority";

        static std::shared_ptr<anet::nn::NetworkHeadFactory> CreateHeadFactory(
            const std::string& quantile_mode, bool dueling, int64_t num_actions, int num_quantiles,
            const anet::nn::WeightInitConfig& init,
            const anet::ConfigData& projection = anet::EmptyConfigData);
        static std::shared_ptr<ActionPolicy> CreateActionPolicy(
            const ActionPolicyConfig& config, const std::string& quantile_mode,
            bool spatial, int64_t num_envs, const torch::Device& device);
        static std::shared_ptr<Learner> CreateLearner(
            const std::string& quantile_mode, int num_quantiles, const LearnerConfig& config,
            NetworkModel& model, RuntimeVars& vars,
            const BatchEnvSpec& batch, const EnvSpec& env, const torch::Device& device,
            anet::seed_t replay_seed, anet::seed_t agent_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            std::shared_ptr<ObservationNormalizer> normalizer = nullptr,
            const LearnerForwardResources& forwards = {},
            std::optional<StuckerConfig> stacker = std::nullopt,
            std::optional<anet::seed_t> learner_seed = std::nullopt,
            std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe = nullptr,
            const std::string& log_suffix = "");
        static std::shared_ptr<Actor> CreateActor(
            const DQNActorConfig& config, const LearnerConfig& learner_config,
            const ActorRequest& request, NetworkModel& model,
            const std::shared_ptr<std::shared_mutex>& mutex, std::shared_ptr<ActionPolicy> policy,
            const anet::nn::NetworkForwardConfig& forward = anet::nn::NetworkForwardConfig::GetDefault(),
            std::shared_ptr<ObservationNormalizer> normalizer = nullptr,
            std::unique_ptr<FrameStacker> stacker = nullptr,
            bool emit_snapshot_metrics = false, ActorQHintConfig hint = {},
            std::optional<anet::ProfiledValueConfig<step_t>> sync_interval = std::nullopt,
            std::optional<SkipActorResources> skip = std::nullopt);
        static std::optional<float> GetScalar(const std::string& key, int64_t index,
            const Learner& learner, const RuntimeVars& vars, const ActionPolicy& target_policy,
            std::shared_mutex& mutex);
        static std::optional<torch::Tensor> GetTensor(
            const std::string& key, const Learner& learner, std::shared_mutex& mutex);
        static std::optional<std::vector<torch::Tensor>> GetTensorVector(
            const std::string& key, const Learner& learner, std::shared_mutex& mutex);
    };

} // namespace anet::rl::dqn
