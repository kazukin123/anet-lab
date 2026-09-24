// anet/default_dqn_agent.hpp
#pragma once

#include <cmath>
#include <memory>
#include <torch/torch.h>

#include "anet/config.hpp"
#include "anet/log.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/rl.hpp"
#include "anet/agent.hpp"
#include "anet/scaler.hpp"
#include "anet/nn.hpp"

namespace anet::rl::dqn {

    // ======================================================
    // DefaultDQNAgentConfig
    // ======================================================

    inline void ValidateTauRule(const TauRuleConfig& rule, const std::string& key)
    {
        if (rule.num_taus <= 0) {
            ANET_SYSTEM_ERROR("Invalid " << key << ".num_taus: value=" << rule.num_taus << " expected > 0");
        }
        if (rule.sample_mode != "random" && rule.sample_mode != "fixed"
            && rule.sample_mode != "stratified" && rule.sample_mode != "systematic"
            && rule.sample_mode != "antithetic") {
            ANET_SYSTEM_ERROR("Invalid " << key << ".sample_mode: value='" << rule.sample_mode
                << "' expected one of: random, fixed, stratified, systematic, antithetic");
        }
    }

    inline void ValidateActionPolicy(const ActionPolicyConfig& policy, const std::string& key)
    {
        const auto& quantile_mode = policy.quantile_mode;
        const auto is_uqe = [](const std::string& type) { return type == "UQE" || type == "1"; };
        const auto& type = policy.policy_type;
        if (type != "Greedy" && type != "EpsilonGreedy" && type != "0" && !is_uqe(type) && !policy.IsThompsonSampling()) {
            ANET_SYSTEM_ERROR("Invalid " << key << ".policy_type='" << type
                << "'; expected Greedy, EpsilonGreedy, UQE, or ThompsonSampling.");
        }
        if (policy.spatial_scale_type != "log" && policy.spatial_scale_type != "linear") {
            ANET_SYSTEM_ERROR("Invalid " << key << ".spatial_scale_type='" << policy.spatial_scale_type
                << "'; expected log or linear.");
        }
        ValidateTauRule(policy.tau_rule, key + ".tau_rule");
        ValidateTauRule(policy.full_distribution_query.tau_rule, key + ".full_distribution_query.tau_rule");

        if ((is_uqe(policy.policy_type) || policy.IsThompsonSampling()) && quantile_mode == "none") {
            ANET_SYSTEM_ERROR("Invalid " << key << ".policy_type: value='" << policy.policy_type
                << "' expected quantile_mode=qr or iqn");
        }
        // full queryはIQN専用の任意機能とし、他のquantile modeでは休眠設定として保持する。
        if (policy.full_distribution_query.enabled
            && quantile_mode == "iqn" && !is_uqe(policy.policy_type)) {
            ANET_SYSTEM_ERROR("Invalid " << key << ".full_distribution_query.enabled: value=true"
                << " expected policy_type=UQE when quantile_mode=iqn, actual quantile_mode='"
                << quantile_mode << "' policy_type='" << policy.policy_type << "'");
        }
        const bool uses_iqn_tau_range = quantile_mode == "iqn"
            && (is_uqe(policy.policy_type) || (policy.IsThompsonSampling() && policy.use_spatial_exploration));
        if (uses_iqn_tau_range
            && (!std::isfinite(policy.uqe_tau_start) || policy.uqe_tau_start < 0.0f || policy.uqe_tau_start > 1.0f
                || !std::isfinite(policy.uqe_tau_end) || policy.uqe_tau_end < 0.0f || policy.uqe_tau_end > 1.0f)) {
            ANET_SYSTEM_ERROR("Invalid " << key << ".uqe_tau_start/end: values="
                << policy.uqe_tau_start << "," << policy.uqe_tau_end << " expected finite values in [0,1]");
        }
    }

    struct DefaultDQNAgentConfig : public anet::Config {

        NetworkModelConfig model;
        StuckerConfig stucker;
        std::map<std::string, DQNActorConfig> actor;
        ActionPolicyConfig target_policy;
        LearnerConfig learner;
        RewardScalerConfig reward_scaler;
        ObservationNormalizerConfig obs_norm;
        anet::nn::WeightInitConfig head_init;
        anet::nn::NetworkGraphVizConfig nn_viz;
        std::string auto_load_file;

        struct QrConfig {
            int num_quantiles = 51;
        } qr;
        std::string quantile_mode = "qr";
        bool use_dueling_net = true;
        bool use_optimistic_target = false;

    public:
        explicit DefaultDQNAgentConfig(const ConfigData& config_data = EmptyConfigData)
            : anet::Config(config_data, "DefaultDQNAgent")
        {
            ANET_READ_CONFIG(config_data, head_init.mode);
            ANET_READ_CONFIG(config_data, head_init.manual_gain);
            ANET_READ_CONFIG(config_data, head_init.constant_val);
            ANET_READ_CONFIG(config_data, head_init.trunc_std);
            ANET_READ_CONFIG(config_data, head_init.trunc_a);
            ANET_READ_CONFIG(config_data, head_init.trunc_b);
			head_init.nonlinearity = "linear";

            ANET_READ_CONFIG(config_data, nn_viz.show_param_shapes);
            ANET_READ_CONFIG(config_data, nn_viz.show_param_count);
            ANET_READ_CONFIG(config_data, nn_viz.show_tensor_specs);
            ANET_READ_CONFIG(config_data, nn_viz.show_branch_config);
            ANET_READ_CONFIG(config_data, nn_viz.show_head_info);
            ANET_READ_CONFIG(config_data, nn_viz.layout);
            ANET_READ_CONFIG(config_data, nn_viz.cluster_branches);
            ANET_READ_CONFIG(config_data, nn_viz.float_precision);
            anet::nn::ValidateNetworkGraphVizConfig(nn_viz, "DefaultDQNAgent.nn_viz");

            ANET_READ_CONFIG(config_data, stucker.use_stacker);
            ANET_READ_CONFIG(config_data, stucker.stack_count);
            ANET_READ_CONFIG(config_data, stucker.stack_keys);

            ANET_READ_CONFIG(config_data, model.soft_update_tau);
            ANET_READ_CONFIG(config_data, model.hard_update_interval);

            ANET_READ_CONFIG(config_data, use_optimistic_target);

            // 全カタログ項目を構築時に型付き設定へ変換する。
            for (const auto& [key, sub] : config_data.MakeSubConfigData(MakeDefaultConfigKey("actor"))) {
                auto& entry = actor[key];
                const auto prefix = "actor.[" + key + "]";
                ReadActionPolicyConfig(config_data, prefix + ".policy", entry.policy);
                ReadConfig(config_data, prefix + ".network", entry.network);
                ReadConfig(config_data, prefix + ".clone_model", entry.clone_model);
                if (entry.network != "online" && entry.network != "target") {
                    ANET_SYSTEM_ERROR("Invalid " << MakeDefaultConfigKey(prefix + ".network")
                        << "='" << entry.network << "'; expected online or target.");
                }
                for (const auto& [leaf, value] : sub.Map()) {
                    if (leaf == "sync_interval" || leaf.starts_with("sync_interval.")) {
                        entry.sync_interval.emplace(anet::ProfiledValueConfig<step_t>{
                            .type = "constant", .value = 400, .min_value = 1});
                        ReadConfig(config_data, prefix + ".sync_interval", *entry.sync_interval);
                        break;
                    }
                }
            }

            target_policy.policy_type = "Greedy";     // デフォルトは安全なGreedy
            target_policy.eps_start = 0.0f;
            target_policy.eps_end = 0.0f;

            if (use_optimistic_target) {        // 「use_optimistic_target = true」だった場合、target_policyのデフォルトはactor.[train].policyをベースとする
                const auto train = actor.find("train");
                if (train == actor.end()) {
                    ANET_SYSTEM_ERROR("DefaultDQNAgent.use_optimistic_target=true requires actor.[train].policy.");
                }
                target_policy = train->second.policy;   // 明示されたコピー元から既定を作る
                const bool optimistic_source = target_policy.policy_type == "UQE" || target_policy.policy_type == "1"
                    || target_policy.IsThompsonSampling();
                target_policy.eps_start = 0.0f; // ただしランダムノイズ(ε)はターゲット計算には絶対不要なので強制遮断
                target_policy.eps_end = 0.0f;
				target_policy.eps_decay_steps = 0;
                target_policy.uqe_eps_start = 0.0f;
                target_policy.uqe_eps_end = 0.0f;
				target_policy.uqe_eps_decay_steps = 0;

                // TrainがEpsilonGreedyだった場合は実質Greedyになるためタイプも変更
                if (target_policy.policy_type == "EpsilonGreedy" || target_policy.policy_type == "0") {
                    target_policy.policy_type = "Greedy";
                }
                if (!optimistic_source) {
                    anet::log::warn() << "use_optimistic_target=true copies actor.[train].policy.policy_type='"
                        << train->second.policy.policy_type << "' to target_policy.policy_type='"
                        << target_policy.policy_type
                        << "' without optimism. Select UQE/ThompsonSampling or use_optimistic_target=false.";
                }
            } else {
                target_policy.policy_type = "Greedy";   // デフォルトは安全なGreedy
            }

            // 楽観policyのコピー対象にせず、target推定品質の既定を決定的な32点へ戻す。
            target_policy.tau_rule = TauRuleConfig{
                .sample_mode = "fixed",
                .num_taus = 32,
            };
            target_policy.full_distribution_query = FullDistributionQueryConfig{};

            // target_policy.*の設定があれば、継承したかもしれないデフォルト値から上書き反映
            ReadActionPolicyConfig(config_data, "target_policy", target_policy);
            target_policy.use_spatial_exploration = false;

            ANET_READ_CONFIG(config_data, learner.enabled);
            ANET_READ_CONFIG(config_data, learner.alpha);
            ANET_READ_CONFIG(config_data, learner.weight_decay);
            ANET_READ_CONFIG(config_data, learner.adam_eps);
            ANET_READ_CONFIG(config_data, learner.use_fused_optimizer);
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
            ANET_READ_CONFIG(config_data, learner.use_rb_prefetch);
            ANET_READ_CONFIG(config_data, learner.n_step);
            ANET_READ_CONFIG(config_data, learner.per_alpha);
            ANET_READ_CONFIG(config_data, learner.per_beta_start);
            ANET_READ_CONFIG(config_data, learner.per_beta_end);
            ANET_READ_CONFIG(config_data, learner.per_beta_step);
            ANET_READ_CONFIG(config_data, learner.per_eps);
            ANET_READ_CONFIG(config_data, learner.per_initial_priority);
            ANET_READ_CONFIG(config_data, learner.per_initial_priority_mode);
            ANET_READ_CONFIG(config_data, learner.use_per_prio_clip);
            ANET_READ_CONFIG(config_data, learner.per_prio_clip_value);
            ANET_READ_CONFIG(config_data, learner.quantile_huber_kappa);
            ANET_READ_CONFIG(config_data, learner.use_double_dqn);
            ANET_READ_CONFIG(config_data, learner.munchausen.enabled);
            ANET_READ_CONFIG(config_data, learner.munchausen.log_policy_mode);
            ANET_READ_CONFIG(config_data, learner.munchausen.alpha);
            ANET_READ_CONFIG(config_data, learner.munchausen.entropy_tau);
            ANET_READ_CONFIG(config_data, learner.munchausen.clip_value_min);

            // 休眠中も認識済みの設定値を検証し、ON/OFF切替で不正値を潜伏させない。
            const auto& munchausen = learner.munchausen;
            if (munchausen.log_policy_mode != "target" && munchausen.log_policy_mode != "online"
                && munchausen.log_policy_mode != "online_reuse") {
                ANET_SYSTEM_ERROR("Invalid learner.munchausen.log_policy_mode='" << munchausen.log_policy_mode
                    << "'; expected target, online, or online_reuse.");
            }
            if (!std::isfinite(munchausen.alpha) || munchausen.alpha < 0.0f || munchausen.alpha > 1.0f) {
                ANET_SYSTEM_ERROR("Invalid learner.munchausen.alpha=" << munchausen.alpha << "; expected finite [0,1].");
            }
            if (!std::isfinite(munchausen.entropy_tau) || munchausen.entropy_tau <= 0.0f) {
                ANET_SYSTEM_ERROR("Invalid learner.munchausen.entropy_tau=" << munchausen.entropy_tau << "; expected finite > 0.");
            }
            if (!std::isfinite(munchausen.clip_value_min) || munchausen.clip_value_min > 0.0f) {
                ANET_SYSTEM_ERROR("Invalid learner.munchausen.clip_value_min=" << munchausen.clip_value_min << "; expected finite <= 0.");
            }
            ANET_READ_CONFIG(config_data, learner.use_n_step);
            ANET_READ_CONFIG(config_data, learner.use_per);
            ANET_READ_CONFIG(config_data, learner.use_tbo);
            ANET_READ_CONFIG(config_data, learner.tbo_epsilon);
            ANET_READ_CONFIG(config_data, learner.use_amp);
            ANET_READ_CONFIG(config_data, learner.use_amp_bf16);
            ANET_READ_CONFIG(config_data, learner.iqn.current_taus.sample_mode);
            ANET_READ_CONFIG(config_data, learner.iqn.current_taus.num_taus);
            ANET_READ_CONFIG(config_data, learner.iqn.target_taus.sample_mode);
            ANET_READ_CONFIG(config_data, learner.iqn.target_taus.num_taus);
            ANET_READ_CONFIG(config_data, learner.plasticity.feature_key);
            ANET_READ_CONFIG(config_data, learner.plasticity.probe.batch_size);
            ANET_READ_CONFIG(config_data, learner.policy_churn.probe.batch_size);
            ANET_READ_CONFIG(config_data, learner.policy_churn.iqn.num_taus);
            ANET_READ_CONFIG(config_data, learner.replay_fit.probe.batch_size);
            ANET_READ_CONFIG(config_data, learner.replay_fit.iqn.num_taus);
            if (learner.replay_fit.probe.batch_size < 1) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.learner.replay_fit.probe.batch_size: value="
                    << learner.replay_fit.probe.batch_size << " expected integer >= 1");
            }
            if (learner.replay_fit.iqn.num_taus < 1) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.learner.replay_fit.iqn.num_taus: value="
                    << learner.replay_fit.iqn.num_taus << " expected integer >= 1");
            }
            if (learner.plasticity.probe.batch_size < 1) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.learner.plasticity.probe.batch_size: value="
                    << learner.plasticity.probe.batch_size << " expected integer >= 1");
            }
            if (learner.policy_churn.probe.batch_size < 1) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.learner.policy_churn.probe.batch_size: value="
                    << learner.policy_churn.probe.batch_size << " expected integer >= 1");
            }
            if (learner.policy_churn.iqn.num_taus < 1) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.learner.policy_churn.iqn.num_taus: value="
                    << learner.policy_churn.iqn.num_taus << " expected integer >= 1");
            }
            if (!std::isfinite(learner.tbo_epsilon) || learner.tbo_epsilon <= 0.0f) {
                ANET_SYSTEM_ERROR(
                    "Invalid DefaultDQNAgent.learner.tbo_epsilon: value=" << learner.tbo_epsilon
                    << " expected finite positive float");
            }

            ANET_READ_CONFIG(config_data, reward_scaler.use_clipping);
            ANET_READ_CONFIG(config_data, reward_scaler.clip_range);
            ANET_READ_CONFIG(config_data, reward_scaler.constant_scale);
            ANET_READ_CONFIG(config_data, reward_scaler.use_dynamic_scaling);
            ANET_READ_CONFIG(config_data, reward_scaler.epsilon);
            ANET_READ_CONFIG(config_data, reward_scaler.use_auto_post_scale);
            ANET_READ_CONFIG(config_data, reward_scaler.reference_q_std);
            ANET_READ_CONFIG(config_data, reward_scaler.manual_post_scale);
            if (learner.use_tbo && (reward_scaler.use_dynamic_scaling || reward_scaler.use_auto_post_scale)) {
                anet::log::warn()
                    << "learner.use_tbo is enabled together with reward_scaler.use_dynamic_scaling or "
                    << "reward_scaler.use_auto_post_scale; targets may be double-compressed.";
            }

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
            ANET_READ_CONFIG(config_data, quantile_mode);
            ANET_READ_CONFIG(config_data, qr.num_quantiles);
            ANET_READ_CONFIG(config_data, use_dueling_net);

            // 学習も読み込みもしないRunは初期重みの評価になる。意図的な構成もあり得るので止めず、書き忘れ検出用に1度だけWARNする。
            if (!learner.enabled && auto_load_file.empty()) {
                anet::log::warn()
                    << "learner.enabled=false without DefaultDQNAgent.auto_load_file; "
                    << "the run evaluates freshly initialized weights.";
            }

            // Agent直下の分布表現を3つのpolicyへ引き継ぎ、QR幅だけをQR learnerへ渡す。
            for (auto& [key, entry] : actor) entry.policy.quantile_mode = quantile_mode;
            target_policy.quantile_mode = quantile_mode;
            learner.quantile_mode = quantile_mode;
            learner.num_quantiles = qr.num_quantiles;

            // 現行の分布表現契約と全tau ruleを、利用前の設定境界で検証する。
            if (quantile_mode != "none" && quantile_mode != "qr" && quantile_mode != "iqn") {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.quantile_mode: value='" << quantile_mode
                    << "' expected one of: none, qr, iqn");
            }
            if (quantile_mode == "qr" && qr.num_quantiles <= 1) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.qr.num_quantiles: value=" << qr.num_quantiles
                    << " expected > 1 when quantile_mode=qr");
            }

            ValidateTauRule(target_policy.tau_rule, "DefaultDQNAgent.target_policy.tau_rule");
            ValidateTauRule(target_policy.full_distribution_query.tau_rule,
                "DefaultDQNAgent.target_policy.full_distribution_query.tau_rule");
            ValidateTauRule(learner.iqn.current_taus, "DefaultDQNAgent.learner.iqn.current_taus");
            ValidateTauRule(learner.iqn.target_taus, "DefaultDQNAgent.learner.iqn.target_taus");

            // コピーと明示overlayを解決した最終target policyに対して競合を検証する。
            if (learner.munchausen.enabled && learner.use_double_dqn) {
                ANET_SYSTEM_ERROR("learner.munchausen.enabled=true conflicts with learner.use_double_dqn=true; expected learner.use_double_dqn=false.");
            }
            // MunchausenとThompsonは共存付加
            if (learner.munchausen.enabled && target_policy.IsThompsonSampling()) {
                ANET_SYSTEM_ERROR("learner.munchausen.enabled=true conflicts with target_policy.policy_type='"
                    << target_policy.policy_type << "'; expected Greedy, EpsilonGreedy, or UQE. use_optimistic_target="
                    << (use_optimistic_target ? "true (actor.[train].policy copy before target overrides)" : "false"));
            }
            ValidateActionPolicy(target_policy, "DefaultDQNAgent.target_policy");
            for (auto& [key, entry] : actor) {
                ValidateActionPolicy(entry.policy, "DefaultDQNAgent.actor.[" + key + "].policy");
            }

            if (quantile_mode == "iqn"
                && (!std::isfinite(learner.quantile_huber_kappa) || learner.quantile_huber_kappa <= 0.0f)) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.learner.quantile_huber_kappa: value="
                    << learner.quantile_huber_kappa << " expected finite value > 0 when quantile_mode=iqn");
            }
        }
    private:
        void ReadActionPolicyConfig(const ConfigData& data, const std::string& prefix, ActionPolicyConfig& policy)
        {
            // ReadConfig経由で既定補完値もModule Configと保存用の記録へ残す。
            ReadConfig(data, prefix + ".policy_type", policy.policy_type);
            ReadConfig(data, prefix + ".eps_start", policy.eps_start);
            ReadConfig(data, prefix + ".eps_end", policy.eps_end);
            ReadConfig(data, prefix + ".eps_decay_steps", policy.eps_decay_steps);
            ReadConfig(data, prefix + ".use_spatial_exploration", policy.use_spatial_exploration);
            ReadConfig(data, prefix + ".spatial_scale_type", policy.spatial_scale_type);
            ReadConfig(data, prefix + ".uqe_tau_start", policy.uqe_tau_start);
            ReadConfig(data, prefix + ".uqe_tau_end", policy.uqe_tau_end);
            ReadConfig(data, prefix + ".uqe_tau_decay_steps", policy.uqe_tau_decay_steps);
            ReadConfig(data, prefix + ".uqe_use_tail_mean", policy.uqe_use_tail_mean);
            ReadConfig(data, prefix + ".uqe_eps_start", policy.uqe_eps_start);
            ReadConfig(data, prefix + ".uqe_eps_end", policy.uqe_eps_end);
            ReadConfig(data, prefix + ".uqe_eps_decay_steps", policy.uqe_eps_decay_steps);
            ReadConfig(data, prefix + ".use_amp", policy.use_amp);
            ReadConfig(data, prefix + ".use_amp_bf16", policy.use_amp_bf16);
            ReadConfig(data, prefix + ".tau_rule.sample_mode", policy.tau_rule.sample_mode);
            ReadConfig(data, prefix + ".tau_rule.num_taus", policy.tau_rule.num_taus);
            ReadConfig(data, prefix + ".full_distribution_query.enabled", policy.full_distribution_query.enabled);
            ReadConfig(data, prefix + ".full_distribution_query.tau_rule.sample_mode", policy.full_distribution_query.tau_rule.sample_mode);
            ReadConfig(data, prefix + ".full_distribution_query.tau_rule.num_taus", policy.full_distribution_query.tau_rule.num_taus);
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

        std::shared_ptr<anet::rl::Actor> CreateActor(const ActorRequest& request) const override;
        std::shared_ptr<anet::rl::Learner> CreateLearner() override;
        void ConfigureScalarMetricSubscriptions(
            const std::vector<ScalarMetricSubscription>& subscriptions) override;
    public:
        std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key) override;
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override;
    public:
        int64_t Save(anet::OutputArchive& archive) const override;
    private:
        std::shared_ptr<ActionContext> CreateActionContext(const ActorRequest& request) const;
        BatchUpdateResultList UpdateFromBatch(const StepCounts& step, const BatchExperience& expriences);
    private:
        std::shared_ptr<anet::rl::dqn::ActionPolicy> CreateActionPolicy(
            const ActionPolicyConfig& policy_config, bool enable_spatial_exploration, int64_t num_envs, const torch::Device& device) const;
        void LoadNetwork(const std::string& filename);
    private:
        DefaultDQNAgentConfig config_;
        std::unique_ptr<anet::rl::dqn::RuntimeVars> vars_;
        std::unique_ptr<anet::rl::dqn::NetworkModel> model_;
        std::shared_ptr<anet::rl::ObservationNormalizer> obs_norm_ = nullptr;
        std::unique_ptr<anet::rl::RewardScaler> reward_scaler_ = nullptr;
        std::shared_ptr<anet::rl::dqn::ActionPolicy> target_policy_;    ///< 学習時ターゲット用ポリシー
        std::shared_ptr<anet::RandomGenerator> plasticity_probe_random_; ///< plasticity probe 専用 Resource
        std::shared_ptr<anet::RandomGenerator> policy_churn_probe_random_; ///< policy churn probe 専用 Resource
        std::shared_ptr<anet::RandomGenerator> replay_fit_probe_random_; ///< 群抽出メトリクス購読時だけ作るResource
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

    private:
        static void ValidateQuantileNetworkContract(
            const DefaultDQNAgentConfig& config,
            const anet::nn::NetworkConfig& net_config,
            const std::string& net_config_prefix);
    };

}// namespace anet::rl
