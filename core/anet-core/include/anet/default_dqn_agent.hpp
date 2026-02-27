// anet/default_dqn_agent.hpp
#pragma once

#include <memory>
#include <torch/torch.h>

#include "anet/config.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/rl.hpp"
#include "anet/agent.hpp"
#include "anet/scaler.hpp"
#include "anet/nn.hpp"

namespace anet::rl::dqn {

    struct DefaultDQNAgentConfig : public anet::Config {

        NetworkConfig network;
        StuckerConfig stucker;
        ActionPolicyConfig action_policy;
        LearnerConfig learner;
        RewardScalerConfig reward_scaler;
        ObservationNormalizerConfig obs_norm;
        anet::nn::WeightInitConfig head_init;

        int num_quantiles = 51;
        bool use_dueling_net = true;
        bool use_qr = true;

        explicit DefaultDQNAgentConfig(const ConfigData& config_data = EmptyConfigData)
            : anet::Config(config_data, "DefaultDQNAgent")
        {
            ANET_READ_CONFIG(config_data, head_init.mode);
            ANET_READ_CONFIG(config_data, head_init.manual_gain);
			head_init.nonlinearity = "linear";

            ANET_READ_CONFIG(config_data, stucker.use_stacker);
            ANET_READ_CONFIG(config_data, stucker.stack_count);

            ANET_READ_CONFIG(config_data, network.soft_update_tau);
            ANET_READ_CONFIG(config_data, network.hard_update_interval);

            ANET_READ_CONFIG(config_data, action_policy.policy_type);
            ANET_READ_CONFIG(config_data, action_policy.eps_max);
            ANET_READ_CONFIG(config_data, action_policy.eps_min);
            ANET_READ_CONFIG(config_data, action_policy.eps_decay_step);
            ANET_READ_CONFIG(config_data, action_policy.uqe_tau_max);
            ANET_READ_CONFIG(config_data, action_policy.uqe_tau_min);
            ANET_READ_CONFIG(config_data, action_policy.uqe_tau_decay_step);
            ANET_READ_CONFIG(config_data, action_policy.uqe_eps_max);
            ANET_READ_CONFIG(config_data, action_policy.uqe_eps_min);
            ANET_READ_CONFIG(config_data, action_policy.uqe_eps_decay_step);
            ANET_READ_CONFIG(config_data, action_policy.uqe_use_tail_mean);
            ANET_READ_CONFIG(config_data, action_policy.uqe_eval_tau);
            ANET_READ_CONFIG(config_data, action_policy.use_amp);
            ANET_READ_CONFIG(config_data, action_policy.use_amp_bf16);

            ANET_READ_CONFIG(config_data, learner.alpha);
            ANET_READ_CONFIG(config_data, learner.gamma);
            ANET_READ_CONFIG(config_data, learner.adam_eps);
            ANET_READ_CONFIG(config_data, learner.use_grad_clip);
            ANET_READ_CONFIG(config_data, learner.grad_clip_tau);
            ANET_READ_CONFIG(config_data, learner.use_td_clip);
            ANET_READ_CONFIG(config_data, learner.td_clip_value);
            ANET_READ_CONFIG(config_data, learner.replay_capacity);
            ANET_READ_CONFIG(config_data, learner.replay_batch_size);
            ANET_READ_CONFIG(config_data, learner.update_warmup_steps);
            ANET_READ_CONFIG(config_data, learner.update_interval);
            ANET_READ_CONFIG(config_data, learner.replay_ratio);
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
            ANET_READ_CONFIG(config_data, learner.use_amp);
            ANET_READ_CONFIG(config_data, learner.use_amp_bf16);

            ANET_READ_CONFIG(config_data, reward_scaler.use_clipping);
            ANET_READ_CONFIG(config_data, reward_scaler.clip_range);
            ANET_READ_CONFIG(config_data, reward_scaler.constant_scale);
            ANET_READ_CONFIG(config_data, reward_scaler.use_dynamic_scaling);
            ANET_READ_CONFIG(config_data, reward_scaler.epsilon);
            ANET_READ_CONFIG(config_data, reward_scaler.use_auto_post_scale);
            ANET_READ_CONFIG(config_data, reward_scaler.reference_q_std);
            ANET_READ_CONFIG(config_data, reward_scaler.manual_post_scale);

            ANET_READ_CONFIG(config_data, obs_norm.pass_through);
            ANET_READ_CONFIG(config_data, obs_norm.use_clipping);
            ANET_READ_CONFIG(config_data, obs_norm.clip_range);
            ANET_READ_CONFIG(config_data, obs_norm.use_dynamic_scaling);
            ANET_READ_CONFIG(config_data, obs_norm.use_centering);
            ANET_READ_CONFIG(config_data, obs_norm.epsilon);
            ANET_READ_CONFIG(config_data, obs_norm.constant_mean);
            ANET_READ_CONFIG(config_data, obs_norm.constant_std);
            ANET_READ_CONFIG(config_data, obs_norm.use_robust_update);
            ANET_READ_CONFIG(config_data, obs_norm.robust_warmup_count);
            ANET_READ_CONFIG(config_data, obs_norm.robust_std_threshold);
            ANET_READ_CONFIG(config_data, obs_norm.post_process_type);
            ANET_READ_CONFIG(config_data, obs_norm.post_process_threshold);

            ANET_READ_CONFIG(config_data, num_quantiles);
            ANET_READ_CONFIG(config_data, use_dueling_net);
            ANET_READ_CONFIG(config_data, use_qr);

            learner.num_quantiles = num_quantiles;
        }
    };

    class DefaultDQNAgent: public anet::rl::FlatStateAgent, public std::enable_shared_from_this<DefaultDQNAgent> {
    public:
        DefaultDQNAgent(
            const DefaultDQNAgentConfig& config,
			const anet::nn::NetworkConfig& net_config,
            const anet::rl::BatchEnvSpec& batc_env_spec, const anet::rl::EnvSpec& env_spec, const torch::Device& device,
            std::shared_ptr<anet::rl::Notifier> notifier = nullptr,
            std::optional<seed_t> seed = std::nullopt);

        std::shared_ptr<ActionContext> CreateActionContext(
            const BatchEnvSpec& batch_env_spec, RunMode run_mode = RunMode::Train) const override;
        anet::rl::BatchActionInfo MakeAction(const StepCounts& step, const BatchState& state, std::shared_ptr<ActionContext> ctx) const override;

        BatchUpdateResultList UpdateFromBatch(const StepCounts& step, const BatchExperience& expriences, std::shared_ptr<const anet::rl::Runner> runner) override;
    public:
        std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key) override;
        std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key) override;

        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override;

        int64_t Save(anet::OutputArchive& archive) const override;
    private:
        DefaultDQNAgentConfig config_;
        std::unique_ptr<anet::rl::RewardScaler> reward_scaler_;
        std::shared_ptr<anet::rl::ObservationNormalizer> obs_norm_;
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
