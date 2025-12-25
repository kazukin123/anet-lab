// anet/default_dqn_agent.hpp
#pragma once

#include <memory>
#include <torch/torch.h>

#include "anet/config.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/rl.hpp"
#include "anet/agent.hpp"
#include "anet/reward_scaler.hpp"

namespace anet::rl::dqn {

    struct DefaultDQNAgentConfig : public anet::Config {

        QNetConfig qnet;
        NetworkConfig network;
        LearnerConfig learner;
        RewardScalerConfig reward_scaler;

        int num_quantiles = 51;
        bool use_dueling_net = true;
        bool use_qr = true;

        explicit DefaultDQNAgentConfig(const ConfigData& config_data = EmptyConfigData)
            : anet::Config(config_data, "DefaultDQNAgent")
        {
            ANET_READ_CONFIG(config_data, qnet.nn_init_mode);
            ANET_READ_CONFIG(config_data, qnet.nn_hidden1);
            ANET_READ_CONFIG(config_data, qnet.nn_hidden2);
            ANET_READ_CONFIG(config_data, qnet.num_quantiles);

            ANET_READ_CONFIG(config_data, network.soft_update_tau);
            ANET_READ_CONFIG(config_data, network.hard_update_interval);

            ANET_READ_CONFIG(config_data, learner.alpha);
            ANET_READ_CONFIG(config_data, learner.gamma);
            ANET_READ_CONFIG(config_data, learner.eps_max);
            ANET_READ_CONFIG(config_data, learner.eps_min);
            ANET_READ_CONFIG(config_data, learner.eps_decay_step);
            ANET_READ_CONFIG(config_data, learner.adam_eps);
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

            ANET_READ_CONFIG(config_data, reward_scaler.use_clip);
            ANET_READ_CONFIG(config_data, reward_scaler.clip_threshold);
            ANET_READ_CONFIG(config_data, reward_scaler.constant_scale);
            ANET_READ_CONFIG(config_data, reward_scaler.use_dynamic_scaling);
            ANET_READ_CONFIG(config_data, reward_scaler.scaling_epsilon);
            ANET_READ_CONFIG(config_data, reward_scaler.use_auto_post_scale);
            ANET_READ_CONFIG(config_data, reward_scaler.reference_q_std);
            ANET_READ_CONFIG(config_data, reward_scaler.manual_post_scale);

            ANET_READ_CONFIG(config_data, num_quantiles);
            ANET_READ_CONFIG(config_data, use_dueling_net);
            ANET_READ_CONFIG(config_data, use_qr);

            qnet.num_quantiles = num_quantiles;
            learner.num_quantiles = num_quantiles;
        }
    };

    class DefaultDQNAgent: public anet::rl::FlatStateAgent, public std::enable_shared_from_this<DefaultDQNAgent> {
    public:
        DefaultDQNAgent(
            const DefaultDQNAgentConfig& config,
            const anet::rl::BatchEnvSpec& batc_env_spec, const anet::rl::EnvSpec& env_spec, const torch::Device& device,
            std::shared_ptr<anet::rl::Notifier> notifier = nullptr,
            std::optional<seed_t> seed = std::nullopt);

        anet::rl::BatchActionInfo MakeAction(const StepCounts& step, const BatchState& state, RunMode mode = RunMode::Train) const override;
        std::shared_ptr<const anet::rl::BatchUpdateResult> UpdateFromBatch(
            const StepCounts& step, const anet::rl::BatchExperience& exprience, const anet::rl::Runner& trainer) override;
    public:
        std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key) override;

        std::optional<float> GetScalar(const std::string& key, int index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int index = -1) const override;
    private:
        DefaultDQNAgentConfig config_;
        std::unique_ptr<anet::rl::RewardScaler> reward_scaler_;
        std::unique_ptr<anet::rl::dqn::RuntimeVars> vars_;
        std::unique_ptr<anet::rl::dqn::Network> network_;
        std::shared_ptr<anet::rl::dqn::ActionPolicy> action_policy_;
        std::shared_ptr<anet::rl::dqn::Learner> learner_;
    };

    class DefaultDQNAgentFactory : public anet::rl::AgentFactory {
    public:
        DefaultDQNAgentFactory() {}

        std::shared_ptr<anet::rl::Agent> CreateAgent(
            const EnvSpec& env_spec,
            const BatchEnvSpec& batch_env_spec,
            const torch::Device& device,
            const anet::ConfigData& config_data = EmptyConfigData,
            std::shared_ptr<anet::rl::Notifier> notifier = nullptr,
            std::optional<anet::seed_t> seed = std::nullopt
        ) const override;

        std::string GetTargetAgentClassId() const override { return "DefaultDQNAgent"; }
    };

}// namespace anet::rl
