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


    // ======================================================
    // DefaultDQNAgentConfig
    // ======================================================

    struct DefaultDQNAgentConfig : public anet::Config {

        NetworkModelConfig model;
        StuckerConfig stucker;
        ActionPolicyConfig train_policy;
        ActionPolicyConfig eval_policy;
        ActionPolicyConfig target_policy;
        LearnerConfig learner;
        RewardScalerConfig reward_scaler;
        ObservationNormalizerConfig obs_norm;
        anet::nn::WeightInitConfig head_init;
        std::string auto_load_file;

        int num_quantiles = 51;
        bool use_dueling_net = true;
        bool use_qr = true;
        bool use_optimistic_target = false;

        explicit DefaultDQNAgentConfig(const ConfigData& config_data = EmptyConfigData)
            : anet::Config(config_data, "DefaultDQNAgent")
        {
            ANET_READ_CONFIG(config_data, head_init.mode);
            ANET_READ_CONFIG(config_data, head_init.manual_gain);
			head_init.nonlinearity = "linear";

            ANET_READ_CONFIG(config_data, stucker.use_stacker);
            ANET_READ_CONFIG(config_data, stucker.stack_count);
            ANET_READ_CONFIG(config_data, stucker.stack_keys);

            ANET_READ_CONFIG(config_data, model.soft_update_tau);
            ANET_READ_CONFIG(config_data, model.hard_update_interval);
            
            ANET_READ_CONFIG(config_data, use_optimistic_target);

            ANET_READ_CONFIG(config_data, train_policy.policy_type);
            ANET_READ_CONFIG(config_data, train_policy.eps_start);
            ANET_READ_CONFIG(config_data, train_policy.eps_end);
            ANET_READ_CONFIG(config_data, train_policy.eps_decay_steps);
            ANET_READ_CONFIG(config_data, train_policy.uqe_tau_start);
            ANET_READ_CONFIG(config_data, train_policy.uqe_tau_end);
            ANET_READ_CONFIG(config_data, train_policy.uqe_tau_decay_steps);
            ANET_READ_CONFIG(config_data, train_policy.uqe_use_tail_mean);
            ANET_READ_CONFIG(config_data, train_policy.uqe_eps_start);
            ANET_READ_CONFIG(config_data, train_policy.uqe_eps_end);
            ANET_READ_CONFIG(config_data, train_policy.uqe_eps_decay_steps);
            ANET_READ_CONFIG(config_data, train_policy.use_amp);
            ANET_READ_CONFIG(config_data, train_policy.use_amp_bf16);

            eval_policy.policy_type = "Greedy";     // デフォルトでGreedy
            eval_policy.eps_start = 0.0f;           // デフォルトでGreedy
            eval_policy.eps_end = 0.0f;
            eval_policy.eps_decay_steps = 0;       // Evalはデフォルトでアニーリングしないべき
            eval_policy.uqe_tau_decay_steps = 0;
            eval_policy.uqe_eps_decay_steps = 0;
            eval_policy.uqe_tau_start = train_policy.uqe_tau_end;   // デフォルト値としてTrainの「最終到達点」をコピーしておく
            eval_policy.uqe_tau_end = train_policy.uqe_tau_end;
            eval_policy.eps_start = train_policy.eps_end;
            eval_policy.eps_end = train_policy.eps_end;
            eval_policy.uqe_eps_start = 0.0f; // デフォルトでGreedy
            eval_policy.uqe_eps_end = 0.0f;   // デフォルトでGreedy
            //eval_policy.uqe_eps_start = train_policy.uqe_eps_end;
            //eval_policy.uqe_eps_end = train_policy.uqe_eps_end;
            ANET_READ_CONFIG(config_data, eval_policy.policy_type);
            ANET_READ_CONFIG(config_data, eval_policy.eps_start);
            ANET_READ_CONFIG(config_data, eval_policy.eps_end);
            ANET_READ_CONFIG(config_data, eval_policy.eps_decay_steps);
            ANET_READ_CONFIG(config_data, eval_policy.uqe_tau_start);
            ANET_READ_CONFIG(config_data, eval_policy.uqe_tau_end);
            ANET_READ_CONFIG(config_data, eval_policy.uqe_tau_decay_steps);
            ANET_READ_CONFIG(config_data, eval_policy.uqe_use_tail_mean);
            ANET_READ_CONFIG(config_data, eval_policy.uqe_eps_start);
            ANET_READ_CONFIG(config_data, eval_policy.uqe_eps_end);
            ANET_READ_CONFIG(config_data, eval_policy.uqe_eps_decay_steps);
            ANET_READ_CONFIG(config_data, eval_policy.use_amp);
            ANET_READ_CONFIG(config_data, eval_policy.use_amp_bf16);


            target_policy.policy_type = "Greedy";     // デフォルトは安全なGreedy
            target_policy.eps_start = 0.0f;
            target_policy.eps_end = 0.0f;

            if (use_optimistic_target) {        // 「use_optimistic_target = true」だった場合、target_policyのデフォルトはtrain_policyをベースとする
                target_policy = train_policy;   // Trainの設定を丸ごとコピー
                target_policy.eps_start = 0.0f; // ただしランダムノイズ(ε)はターゲット計算には絶対不要なので強制遮断
                target_policy.eps_end = 0.0f;
                target_policy.uqe_eps_start = 0.0f;
                target_policy.uqe_eps_end = 0.0f;

                // TrainがEpsilonGreedyだった場合は実質Greedyになるためタイプも変更
                if (target_policy.policy_type == "EpsilonGreedy" || target_policy.policy_type == "0") {
                    target_policy.policy_type = "Greedy";
                }
            } else {
                target_policy.policy_type = "Greedy";   // デフォルトは安全なGreedy
            }

            // target_policy.*の設定があれば、継承したかもしれないデフォルト値から上書き反映
            ANET_READ_CONFIG(config_data, target_policy.policy_type);
            ANET_READ_CONFIG(config_data, target_policy.eps_start);
            ANET_READ_CONFIG(config_data, target_policy.eps_end);
            ANET_READ_CONFIG(config_data, target_policy.eps_decay_steps);
            ANET_READ_CONFIG(config_data, target_policy.uqe_tau_start);
            ANET_READ_CONFIG(config_data, target_policy.uqe_tau_end);
            ANET_READ_CONFIG(config_data, target_policy.uqe_tau_decay_steps);
            ANET_READ_CONFIG(config_data, target_policy.uqe_use_tail_mean);
            ANET_READ_CONFIG(config_data, target_policy.uqe_eps_start);
            ANET_READ_CONFIG(config_data, target_policy.uqe_eps_end);
            ANET_READ_CONFIG(config_data, target_policy.uqe_eps_decay_steps);
            ANET_READ_CONFIG(config_data, target_policy.use_amp);
            ANET_READ_CONFIG(config_data, target_policy.use_amp_bf16);

            ANET_READ_CONFIG(config_data, learner.alpha);
            ANET_READ_CONFIG(config_data, learner.weight_decay);
            ANET_READ_CONFIG(config_data, learner.adam_eps);
            ANET_READ_CONFIG(config_data, learner.gamma);
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

            ANET_READ_CONFIG(config_data, auto_load_file);
            ANET_READ_CONFIG(config_data, num_quantiles);
            ANET_READ_CONFIG(config_data, use_dueling_net);
            ANET_READ_CONFIG(config_data, use_qr);

            learner.num_quantiles = num_quantiles;
        }
    };


    // ======================================================
    // DefaultDQNAgent
    // ======================================================

    class DefaultDQNAgent: public anet::rl::AgentBase, public anet::rl::Learner, public std::enable_shared_from_this<DefaultDQNAgent> {
    public:
        DefaultDQNAgent(
            const DefaultDQNAgentConfig& config,
			const anet::nn::NetworkConfig& net_config,
            const anet::rl::BatchEnvSpec& batc_env_spec, const anet::rl::EnvSpec& env_spec, const torch::Device device,
            std::optional<seed_t> seed = std::nullopt);

        std::shared_ptr<anet::rl::Actor> CreateActor(const BatchEnvSpec& batch_env_spec, RunMode mode, bool clone_model, std::optional<torch::Device> device = std::nullopt) const override;
        std::shared_ptr<anet::rl::Learner> CreateLearner() override;
    public:
        std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key) override;
        std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key) override;
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override;
    public:
        int64_t Save(anet::OutputArchive& archive) const override;
    private:
        std::shared_ptr<ActionContext> CreateActionContext(
            const BatchEnvSpec& batch_env_spec, RunMode run_mode = RunMode::Train, std::optional<torch::Device> device = std::nullopt) const;
        anet::rl::BatchActionInfo MakeAction(const StepCounts& step, const BatchState& state, std::shared_ptr<ActionContext> ctx) const;
        BatchUpdateResultList UpdateFromBatch(const StepCounts& step, const BatchExperience& expriences);
    private:
        std::shared_ptr<anet::rl::dqn::ActionPolicy> CreateActionPolicy(const ActionPolicyConfig& policy_config);
        void LoadNetwork(const std::string& filename);
    private:
        DefaultDQNAgentConfig config_;
        std::unique_ptr<anet::rl::dqn::RuntimeVars> vars_;
        std::unique_ptr<anet::rl::dqn::NetworkModel> model_;
        std::shared_ptr<anet::rl::ObservationNormalizer> obs_norm_ = nullptr;
        std::unique_ptr<anet::rl::RewardScaler> reward_scaler_ = nullptr;
        std::shared_ptr<anet::rl::dqn::ActionPolicy> train_policy_;     ///< 探索用ポリシー(RunMode=Train)
        std::shared_ptr<anet::rl::dqn::ActionPolicy> eval_policy_;      ///< 評価用ポリシー(RunMode=Eval/Eval1/Eval2)
        std::shared_ptr<anet::rl::dqn::ActionPolicy> target_policy_;    ///< 学習時ターゲット用ポリシー
        std::shared_ptr<anet::rl::dqn::Learner> learner_;
    private:
        seed_t action_context_seed_;
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
