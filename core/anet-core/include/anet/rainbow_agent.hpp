#pragma once

#include <memory>
#include <torch/torch.h>

#include "anet/config.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/rl.hpp"
#include "anet/agent.hpp"
#include "anet/nn.hpp"

namespace anet::rl::dqn {

    struct RainbowAgentConfig : public anet::Config {

        anet::nn::WeightInitConfig head_init;
        NetworkModelConfig model;
        ActionPolicyConfig action_policy;
        LearnerConfig learner;

        int num_quantiles = 51;
        bool use_dueling_net = true;
        bool use_qr = true;

        explicit RainbowAgentConfig(const ConfigData& config_data = EmptyConfigData) : anet::Config(config_data, "RainbowAgent") {
            ANET_READ_CONFIG(config_data, head_init.mode);
            ANET_READ_CONFIG(config_data, head_init.manual_gain);
            head_init.nonlinearity = "linear";

            ANET_READ_CONFIG(config_data, model.soft_update_tau);
            ANET_READ_CONFIG(config_data, model.hard_update_interval);

            ANET_READ_CONFIG(config_data, action_policy.eps_start);
            ANET_READ_CONFIG(config_data, action_policy.eps_end);
            ANET_READ_CONFIG(config_data, action_policy.eps_decay_steps);

            ANET_READ_CONFIG(config_data, learner.alpha);
            ANET_READ_CONFIG(config_data, learner.gamma);
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
            ANET_READ_CONFIG(config_data, learner.per_beta_start);
            ANET_READ_CONFIG(config_data, learner.per_beta_end);
            ANET_READ_CONFIG(config_data, learner.per_beta_step);
            ANET_READ_CONFIG(config_data, learner.per_eps);
            ANET_READ_CONFIG(config_data, learner.per_initial_priority);
            ANET_READ_CONFIG(config_data, learner.use_per_prio_clip);
            ANET_READ_CONFIG(config_data, learner.per_prio_clip_value);
            ANET_READ_CONFIG(config_data, learner.quantile_huber_kappa);
            ANET_READ_CONFIG(config_data, learner.use_double_dqn);
            ANET_READ_CONFIG(config_data, learner.use_n_step);
            ANET_READ_CONFIG(config_data, learner.use_per);

            ANET_READ_CONFIG(config_data, num_quantiles);
            ANET_READ_CONFIG(config_data, use_dueling_net);
            ANET_READ_CONFIG(config_data, use_qr);

            learner.num_quantiles = num_quantiles;
            learner.use_rb_prefetch = false;
            learner.use_tbo = false;
            learner.use_fused_optimizer = false;
        }
    };

    class RainbowAgent: public anet::rl::AgentBase, public anet::rl::Learner, public std::enable_shared_from_this<RainbowAgent> {
    public:
        RainbowAgent(
            const RainbowAgentConfig& config,
            const anet::nn::NetworkConfig& net_config,
            const anet::rl::BatchEnvSpec& batc_env_spec, const anet::rl::EnvSpec& env_spec, const torch::Device& device,
            std::optional<seed_t> seed = std::nullopt);

        BatchUpdateResultList UpdateFromBatch(const StepCounts& step, const anet::rl::BatchExperience& exprience);
    public:
        std::shared_ptr<anet::rl::Actor> CreateActor(const anet::rl::BatchEnvSpec& batch_env_spec, anet::rl::RunMode run_mode, bool clone_model, std::optional<torch::Device> device = std::nullopt) const override;
        std::shared_ptr<anet::rl::Learner> CreateLearner() override;
    public:
        std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key) override;

        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override;
    private:
        std::shared_ptr<anet::rl::ActionContext> CreateActionContext(
            const BatchEnvSpec& batch_env_spec, RunMode run_mode, std::optional<torch::Device> device) const;
    private:
        RainbowAgentConfig config_;
        std::unique_ptr<anet::rl::dqn::RuntimeVars> vars_;
        std::unique_ptr<anet::rl::dqn::NetworkModel> model_;
        std::shared_ptr<anet::rl::dqn::ActionPolicy> action_policy_;
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
