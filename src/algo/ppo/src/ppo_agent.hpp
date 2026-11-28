#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "anet/random.hpp"
#include "anet/serialize.hpp"
#include "anet/rl.hpp"
#include "anet/nn.hpp"
#include "anet/agent.hpp"
#include "ppo_config.hpp"
#include "ppo_learner.hpp"
#include "ppo_network_model.hpp"

namespace anet::rl::ppo {

    /// 離散行動の PPO（Proximal Policy Optimization）の Agent。
    /// Network と optimizer を所有し、Actor（行動選択）と Learner（rollout の蓄積と更新）を作る。
    class PPOAgent final : public anet::rl::AgentBase {
    public:
        static constexpr char kCheckpointFormat[] = "PPOAgent-1"; ///< 保存の形式の名前。
    public:
        PPOAgent(const PPOAgentConfig& config, const anet::nn::NetworkConfig& network_config,
            const anet::rl::EnvSpec& env_spec, const anet::rl::BatchEnvSpec& batch_env_spec, torch::Device device,
            std::optional<anet::seed_t> seed = std::nullopt);

        std::shared_ptr<anet::rl::Actor> CreateActor(const ActorRequest& request) const override;
        std::shared_ptr<anet::rl::Learner> CreateLearner() override;
        void ConfigureScalarMetricSubscriptions(const std::vector<ScalarMetricSubscription>& subscriptions) override;
        int64_t Save(anet::OutputArchive& archive) const override;

        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override
            { return std::nullopt; }
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override
            { return std::nullopt; }
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override
            { return std::nullopt; }
    private:
        void LoadCheckpoint(const std::string& filename);
    private:
        const PPOAgentConfig config_;
        std::shared_ptr<NetworkModel> model_; ///< Learner が更新し、Actor が共有するか写す Network。
        std::shared_ptr<Learner> learner_;    ///< optimizer を保存できるよう、Agent と同じ寿命で持つ。
    };

    class PPOAgentFactory final : public anet::rl::AgentFactory {
    public:
        std::shared_ptr<Agent> CreateAgent(const EnvSpec& env_spec, const BatchEnvSpec& batch_env_spec,
            const torch::Device& device, const anet::ConfigData& config_data = EmptyConfigData,
            std::shared_ptr<anet::rl::Notifier> notifier = nullptr,
            std::optional<anet::seed_t> seed = std::nullopt) const override;
        std::string GetTargetAgentClassId() const override { return kAgentClassId; }
    };

} // namespace anet::rl::ppo
