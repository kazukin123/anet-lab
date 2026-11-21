#pragma once

#include <memory>
#include <torch/torch.h>

#include "anet/config.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/rl.hpp"
#include "dqn_config.hpp"
#include "anet/nn.hpp"

namespace anet::rl::dqn {

    struct RainbowAgentConfig : public anet::Config {

        struct {
            anet::nn::WeightInitConfig init;
        } head;
        NetworkModelConfig model;
        std::map<std::string, DQNActorConfig> actor;
        LearnerConfig learner;

        int num_quantiles = 51;
        bool use_dueling_net = true;
        bool use_qr = true;

        explicit RainbowAgentConfig(const ConfigData& config_data = EmptyConfigData) : anet::Config(config_data, "RainbowAgent") {
            head.init.nonlinearity = "linear";
            ANET_READ_CONFIG(config_data, head.init);

            ANET_READ_CONFIG(config_data, model.soft_update_tau);
            ANET_READ_CONFIG(config_data, model.hard_update_interval);

            // 方策とnetworkの選択は全Actorで同じカタログ契約を使う。
            for (const auto& [key, sub] : config_data.MakeSubConfigData("RainbowAgent.actor")) {
                auto& entry = actor[key];
                const auto prefix = "actor.[" + key + "]";
                const auto data = MakeReadScope(config_data, prefix);
                ReadConfig(data, prefix + ".policy", entry.policy);
                ReadConfig(data, prefix + ".network", entry.network);
                ReadConfig(data, prefix + ".clone_model", entry.clone_model);
                if (entry.network != "online" && entry.network != "target") {
                    ANET_SYSTEM_ERROR("Invalid RainbowAgent." << prefix << ".network='" << entry.network
                        << "'; expected online or target.");
                }
                data.RequireAllKeysRead(MakeDefaultConfigKey(prefix));
            }

            ANET_READ_CONFIG(config_data, learner.learning_rate);
            ANET_READ_CONFIG(config_data, learner.weight_decay);
            ANET_READ_CONFIG(config_data, learner.gamma);
            ANET_READ_CONFIG(config_data, learner.munchausen.alpha);
            ANET_READ_CONFIG(config_data, learner.munchausen.entropy_tau);
            ANET_READ_CONFIG(config_data, learner.use_grad_clip);
            ANET_READ_CONFIG(config_data, learner.grad_clip_tau);
            ANET_READ_CONFIG(config_data, learner.use_td_clip);
            ANET_READ_CONFIG(config_data, learner.td_clip_value);
            ANET_READ_CONFIG(config_data, learner.replay_capacity);
            ANET_READ_CONFIG(config_data, learner.replay_batch_size);
            ANET_READ_CONFIG(config_data, learner.update_warmup_steps);
            ANET_READ_CONFIG(config_data, learner.update_interval);
            ANET_READ_CONFIG(config_data, learner.n_step);
            ANET_READ_CONFIG(config_data, learner.per_alpha);
            ANET_READ_CONFIG(config_data, learner.per_beta);
            ANET_READ_CONFIG(config_data, learner.per_eps);
            ANET_READ_CONFIG(config_data, learner.per_initial_priority);
            ANET_READ_CONFIG(config_data, learner.per_initial_priority_mode);
            ANET_READ_CONFIG(config_data, learner.use_per_prio_clip);
            ANET_READ_CONFIG(config_data, learner.per_prio_clip_value);
            ANET_READ_CONFIG(config_data, learner.quantile_huber_kappa);
            ANET_READ_CONFIG(config_data, learner.use_double_dqn);
            ANET_READ_CONFIG(config_data, learner.use_n_step);
            ANET_READ_CONFIG(config_data, learner.use_per);

            ANET_READ_CONFIG(config_data, num_quantiles);
            ANET_READ_CONFIG(config_data, use_dueling_net);
            ANET_READ_CONFIG(config_data, use_qr);
            ValidateQuantileConfig();
            ValidatePolicyConfig();

            learner.num_quantiles = num_quantiles;
            learner.use_rb_prefetch = false;
            learner.use_tbo = false;
            learner.use_fused_optimizer = false;
        }

        void ValidatePolicyConfig() const
        {
            for (const auto& [key, entry] : actor) {
                const auto prefix = MakeDefaultConfigKey("actor.[" + key + "].policy");
                if (entry.policy.policy_type != kActionPolicyTypeStr_EpsilonGreedy && entry.policy.policy_type != "0") {
                    ANET_SYSTEM_ERROR("Invalid " << prefix << ".policy_type='" << entry.policy.policy_type
                        << "'; expected EpsilonGreedy (RainbowAgent supports epsilon-greedy only).");
                }
                ValidateActionPolicy(entry.policy, prefix);
            }
        }

        void ValidateQuantileConfig() const
        {
            if (use_qr && num_quantiles <= 1) {
                ANET_SYSTEM_ERROR("Invalid RainbowAgent.num_quantiles=" << num_quantiles
                    << " with RainbowAgent.use_qr=true; expected integer > 1.");
            }
        }
    };

    class RainbowAgent: public anet::rl::AgentBase, public anet::rl::Learner, public std::enable_shared_from_this<RainbowAgent> {
    public:
        RainbowAgent(
            const RainbowAgentConfig& config,
            const anet::nn::NetworkConfig& net_config,
            const anet::rl::BatchEnvSpec& batc_env_spec, const anet::rl::EnvSpec& env_spec, const torch::Device& device,
            std::optional<seed_t> seed = std::nullopt);

    public:
        std::shared_ptr<anet::rl::Actor> CreateActor(const ActorRequest& request) const override;
        std::shared_ptr<anet::rl::Learner> CreateLearner() override;
        std::unique_ptr<AgentStepDiagnosticsProvider> CreateStepDiagnosticsProvider(const AgentStepDiagnosticsRequest& request) const override;
    public:
        std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key) override;

        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override;
    private:
        BatchUpdateResultList UpdateFromBatch(const StepCounts& step, const anet::rl::BatchExperience& experience) override;
        RainbowAgentConfig config_;
        std::unique_ptr<anet::rl::dqn::RuntimeVars> vars_;
        std::unique_ptr<anet::rl::dqn::NetworkModel> model_;
        std::shared_ptr<anet::rl::dqn::ActionPolicy> target_policy_;
        std::shared_ptr<anet::rl::dqn::Learner> learner_;
    };

    class RainbowAgentFactory : public anet::rl::AgentFactory {
    public:
        RainbowAgentFactory() { }

        std::shared_ptr<anet::rl::Agent> CreateAgent(
            const EnvSpec& env_spec,
            const BatchEnvSpec& batch_env_spec,
            const torch::Device& device,
            const anet::ConfigData& config_data = EmptyConfigData,
            std::shared_ptr<anet::rl::Notifier> notifier = nullptr,
            std::optional<anet::seed_t> seed = std::nullopt
        ) const override;

        std::string GetTargetAgentClassId() const override { return "RainbowAgent"; }
    };

}// namespace anet::rl
