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

namespace anet::rl {
    class FrameStacker;
}

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
        if (type != kActionPolicyTypeStr_Greedy && type != kActionPolicyTypeStr_EpsilonGreedy && type != "0" && !is_uqe(type) && !policy.IsThompsonSampling()) {
            ANET_SYSTEM_ERROR("Invalid " << key << ".policy_type='" << type
                << "'; expected " << kActionPolicyTypeStr_Greedy << ", " << kActionPolicyTypeStr_EpsilonGreedy
                << ", " << kActionPolicyTypeStr_UQE << ", or " << kActionPolicyTypeStr_ThompsonSampling << ".");
        }
        ValidateTauRule(policy.tau_rule, key + ".tau_rule");
        ValidateTauRule(policy.full_distribution_query.tau_rule, key + ".full_distribution_query.tau_rule");

        if ((is_uqe(policy.policy_type) || policy.IsThompsonSampling()) && quantile_mode == ActionPolicyConfig::kQuantileModeNone) {
            ANET_SYSTEM_ERROR("Invalid " << key << ".policy_type: value='" << policy.policy_type
                << "' expected quantile_mode=" << ActionPolicyConfig::kQuantileModeQr
                << " or " << ActionPolicyConfig::kQuantileModeIqn);
        }
        // full queryはIQN専用の任意機能とし、他のquantile modeでは休眠設定として保持する。
        if (policy.full_distribution_query.enabled
            && quantile_mode == ActionPolicyConfig::kQuantileModeIqn && !is_uqe(policy.policy_type)) {
            ANET_SYSTEM_ERROR("Invalid " << key << ".full_distribution_query.enabled: value=true"
                << " expected policy_type=UQE when quantile_mode=" << ActionPolicyConfig::kQuantileModeIqn
                << ", actual quantile_mode='"
                << quantile_mode << "' policy_type='" << policy.policy_type << "'");
        }

    }

    struct DefaultDQNAgentConfig : public anet::Config {
        std::map<std::string, anet::ConfigData> nn_forward;
        NetworkModelConfig model;
        StuckerConfig stucker;
        std::map<std::string, DQNActorConfig> actor;
        ActionPolicyConfig target_policy;
        LearnerConfig learner;
        RewardScalerConfig reward_scaler;
        ObservationNormalizerConfig obs_norm;
        struct {
            bool use_dueling = true;
            anet::nn::WeightInitConfig init;
            anet::ConfigData projection; ///< 構築設定の解釈は選択された Head の具象クラスが所有する。
        } head;
        SkipConfig skip;
        anet::ConfigData skip_net;
        anet::nn::NetworkGraphVizConfig nn_viz;
        std::string auto_load_file;

        struct QrConfig {
            int num_quantiles = 51;
        } qr;
        std::string quantile_mode = ActionPolicyConfig::kQuantileModeQr;
        bool use_optimistic_target = false;

    public:
        /// Agentの接頭辞を含む、選択されたNN実行設定の完全なキーを返す。
        std::string MakeForwardConfigKey(const std::string& key) const
        {
            return MakeDefaultConfigKey(MakeNetworkForwardConfigKey(key));
        }

        explicit DefaultDQNAgentConfig(const ConfigData& config_data = EmptyConfigData)
            : anet::Config(config_data, "DefaultDQNAgent")
        {
            // Head 射影の構築設定は opaque にコピーし、固有の項目を Agent で読み取らない。
            const auto projection_prefix = MakeDefaultConfigKey("head.projection") + ".";
            for (const auto& [key, value] : config_data.Map()) {
                if (!key.starts_with(projection_prefix)) continue;
                head.projection.Set(key.substr(projection_prefix.size()), value);
                const auto relative = key.substr(GetConfigPrefix().size() + 1);
                my_config_data_.Set(relative, value);
                my_config_json_[relative] = value;
            }

            ANET_READ_CONFIG(config_data, head.init.mode);
            ANET_READ_CONFIG(config_data, head.init.manual_gain);
            ANET_READ_CONFIG(config_data, head.init.constant_val);
            ANET_READ_CONFIG(config_data, head.init.trunc_std);
            ANET_READ_CONFIG(config_data, head.init.trunc_a);
            ANET_READ_CONFIG(config_data, head.init.trunc_b);
			head.init.nonlinearity = "linear";
            anet::nn::ValidateWeightInitConfig(head.init, MakeDefaultConfigKey("head.init"));

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

            // カタログの値はDTOへ保存し、使用する具象部品が実行準備で検証する。
            for (const auto& [key, sub] : config_data.MakeSubConfigData(MakeDefaultConfigKey("nn_forward"))) {
                nn_forward.emplace(key, sub);
                // 記録も opaque な値のまま保存し、型付きの解釈はここで追加しない。
                for (const auto& [leaf, value] : sub.Map()) {
                    const auto name = MakeNetworkForwardConfigKey(key) + "." + leaf;
                    my_config_data_.Set(name, value);
                    my_config_json_[name] = value;
                }
            }

            // 全カタログ項目を構築時に型付き設定へ変換する。
            for (const auto& [key, sub] : config_data.MakeSubConfigData(MakeDefaultConfigKey("actor"))) {
                auto& entry = actor[key];
                const auto prefix = "actor.[" + key + "]";
                for (const auto& [leaf, value] : sub.Map()) {
                    if (leaf.starts_with("skip.policy.")) entry.has_skip_policy = true;
                }
                ReadActionPolicyConfig(config_data, prefix + ".policy", entry.policy);
                ReadConfig(config_data, prefix + ".network", entry.network);
                ReadConfig(config_data, prefix + ".clone_model", entry.clone_model);
                ValidateForwardReference(config_data, prefix + ".nn_forward_key", entry.nn_forward_key);
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

            target_policy.policy_type = kActionPolicyTypeStr_Greedy;     // デフォルトは安全なGreedy
            target_policy.epsilon = {.value = 0.0f, .min_value = 0.0f, .max_value = 1.0f};
            if (use_optimistic_target) {        // 「use_optimistic_target = true」だった場合、target_policyのデフォルトはactor.[train].policyをベースとする
                const auto train = actor.find("train");
                if (train == actor.end()) {
                    ANET_SYSTEM_ERROR("DefaultDQNAgent.use_optimistic_target=true requires actor.[train].policy.");
                }
                target_policy = train->second.policy;
                target_policy.epsilon = {.value = 0.0f, .min_value = 0.0f, .max_value = 1.0f};
                target_policy.uqe.epsilon = {.value = 0.0f, .min_value = 0.0f, .max_value = 1.0f};   // 明示されたコピー元から既定を作る
                const bool optimistic_source = target_policy.policy_type == "UQE" || target_policy.policy_type == "1"
                    || target_policy.IsThompsonSampling();
                if (target_policy.policy_type == "EpsilonGreedy" || target_policy.policy_type == "0") {
                    target_policy.policy_type = kActionPolicyTypeStr_Greedy;
                }
                if (!optimistic_source) {
                    anet::log::warn() << "use_optimistic_target=true copies actor.[train].policy.policy_type='"
                        << train->second.policy.policy_type << "' to target_policy.policy_type='"
                        << target_policy.policy_type
                        << "' without optimism. Select UQE/ThompsonSampling or use_optimistic_target=false.";
                }
            } else {
                target_policy.policy_type = kActionPolicyTypeStr_Greedy;   // デフォルトは安全なGreedy
            }

            // 楽観policyのコピー対象にせず、target推定品質の既定を決定的な32点へ戻す。
            target_policy.tau_rule = TauRuleConfig{
                .sample_mode = "fixed",
                .num_taus = 32,
            };
            target_policy.full_distribution_query = FullDistributionQueryConfig{};

            // target_policy.*の設定があれば、継承したかもしれないデフォルト値から上書き反映
            ReadActionPolicyConfig(config_data, "target_policy", target_policy);
            target_policy.spatial.enabled = false;

            // 学習停止中も宣言された参照を解決する。保持や時計の適否は実際の利用時に委譲する。
            ValidateForwardReference(
                config_data, "learner.current.nn_forward_key", learner.current.nn_forward_key);
            ValidateForwardReference(
                config_data, "learner.target.nn_forward_key", learner.target.nn_forward_key);

            ANET_READ_CONFIG(config_data, learner.enabled);
            ANET_READ_CONFIG(config_data, learner.learning_rate);
            ANET_READ_CONFIG(config_data, learner.weight_decay);
            ANET_READ_CONFIG(config_data, learner.noisy_weight_decay);
            if (!std::isfinite(learner.noisy_weight_decay) || learner.noisy_weight_decay < 0.0f) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.learner.noisy_weight_decay: value="
                    << learner.noisy_weight_decay << " expected finite value >= 0");
            }
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
            ANET_READ_CONFIG(config_data, learner.per_beta);
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
            ANET_READ_CONFIG(config_data, learner.noisy_probe.probe.batch_size);
            ANET_READ_CONFIG(config_data, learner.noisy_probe.num_draws);
            ANET_READ_CONFIG(config_data, learner.noisy_probe.iqn.num_taus);
            if (learner.noisy_probe.probe.batch_size < 1) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.learner.noisy_probe.probe.batch_size: value="
                    << learner.noisy_probe.probe.batch_size << " expected integer >= 1");
            }
            if (learner.noisy_probe.num_draws < 1) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.learner.noisy_probe.num_draws: value="
                    << learner.noisy_probe.num_draws << " expected integer >= 1");
            }
            if (learner.noisy_probe.iqn.num_taus < 1) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.learner.noisy_probe.iqn.num_taus: value="
                    << learner.noisy_probe.iqn.num_taus << " expected integer >= 1");
            }
            // 値域・組み合わせの検証は、購読が決まる学習開始前の境界で行う。
            ANET_READ_CONFIG(config_data, learner.grad_noise.num_micro_batches);
            ANET_READ_CONFIG(config_data, learner.grad_noise.ema_alpha);
            ANET_READ_CONFIG(config_data, learner.final_feature_branch_key);
            ANET_READ_CONFIG(config_data, learner.attention_entropy.probe.batch_size);
            if (learner.attention_entropy.probe.batch_size < 1) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.learner.attention_entropy.probe.batch_size: value="
                    << learner.attention_entropy.probe.batch_size << " expected integer >= 1");
            }
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
            ANET_READ_CONFIG(config_data, head.use_dueling);

            ANET_READ_CONFIG(config_data, skip.enabled);
            const bool has_skip_limit = HasConfigValue(config_data, "skip.max_length");
            ANET_READ_CONFIG(config_data, skip.max_length);
            // 文字列の設定を境界で検証し、実行系へは構成を表すenumだけを渡す。
            std::string network_mode = SkipConfig::kNetworkModeActionOut;
            ReadConfig(config_data, "skip.network_mode", network_mode);
            if (network_mode == SkipConfig::kNetworkModeActionOut) skip.network_mode = SkipNetworkMode::ActionOut;
            else if (network_mode == SkipConfig::kNetworkModeActionIn) skip.network_mode = SkipNetworkMode::ActionIn;
            else ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.skip.network_mode='" << network_mode << "'; expected " << SkipConfig::kNetworkModeActionOut << " or " << SkipConfig::kNetworkModeActionIn << ".");
            ANET_READ_CONFIG(config_data, skip.head.use_dueling);
            ANET_READ_CONFIG(config_data, learner.skip.feature_gradient);
            ANET_READ_CONFIG(config_data, learner.skip.sample_weight);
            ANET_READ_CONFIG(config_data, learner.skip.loss_weight);
            ANET_READ_CONFIG(config_data, learner.skip.priority_lambda);
            if ((skip.enabled && !has_skip_limit) || (has_skip_limit && skip.max_length < 1)) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.skip.max_length=" << skip.max_length
                    << " specified=" << has_skip_limit << "; expected explicit integer >= 1 when skip.enabled=true.");
            }
            if (learner.skip.feature_gradient != LearnerConfig::kFeatureGradientStop
                && learner.skip.feature_gradient != LearnerConfig::kFeatureGradientMerge
                && learner.skip.feature_gradient != LearnerConfig::kFeatureGradientSplit) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.learner.skip.feature_gradient='" << learner.skip.feature_gradient << "'; expected " << LearnerConfig::kFeatureGradientStop
                    << ", " << LearnerConfig::kFeatureGradientMerge << " or " << LearnerConfig::kFeatureGradientSplit << ".");
            }
            if (learner.skip.sample_weight != LearnerConfig::kSampleWeightUniform
                && learner.skip.sample_weight != LearnerConfig::kSampleWeightKmax) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.learner.skip.sample_weight='" << learner.skip.sample_weight
                    << "'; expected " << LearnerConfig::kSampleWeightUniform << " or " << LearnerConfig::kSampleWeightKmax << ".");
            }
            for (const auto& [key, value] : std::initializer_list<std::pair<const char*, float>>{
                {"loss_weight", learner.skip.loss_weight}, {"priority_lambda", learner.skip.priority_lambda}}) {
                if (!std::isfinite(value) || value < 0.0f) {
                    ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.learner.skip." << key << "=" << value << "; expected finite value >= 0.");
                }
            }
            if (skip.enabled && learner.use_amp && !learner.use_amp_bf16) {
                ANET_SYSTEM_ERROR("DefaultDQNAgent.skip.enabled=true requires learner.use_amp=false or learner.use_amp_bf16=true.");
            }
            skip.head.projection = head.projection;
            for (const auto& [key, value] : config_data.Map()) {
                for (const auto& [prefix, destination] : std::initializer_list<std::pair<std::string, anet::ConfigData*>>{
                    {"skip.net", &skip_net}, {"skip.head.projection", &skip.head.projection}}) {
                    const auto full_prefix = MakeDefaultConfigKey(prefix) + ".";
                    if (!key.starts_with(full_prefix)) continue;
                    destination->Set(prefix == "skip.net" ? key : key.substr(full_prefix.size()), value);
                    const auto relative = key.substr(GetConfigPrefix().size() + 1);
                    my_config_data_.Set(relative, value);
                    my_config_json_[relative] = value;
                }
            }
            if (skip.enabled && skip_net.Map().begin() == skip_net.Map().end()) {
                ANET_SYSTEM_ERROR("Missing DefaultDQNAgent.skip.net; expected a network definition when skip.enabled=true.");
            }
            for (const auto& [key, value] : config_data.Map()) {
                if (key.starts_with("net.block.") || key.starts_with("net.config_profile.")) skip_net.Set(key, value);
            }
            skip.head.init = head.init;
            ANET_READ_CONFIG(config_data, skip.head.init.mode);
            ANET_READ_CONFIG(config_data, skip.head.init.manual_gain);
            ANET_READ_CONFIG(config_data, skip.head.init.constant_val);
            ANET_READ_CONFIG(config_data, skip.head.init.trunc_std);
            ANET_READ_CONFIG(config_data, skip.head.init.trunc_a);
            ANET_READ_CONFIG(config_data, skip.head.init.trunc_b);
            anet::nn::ValidateWeightInitConfig(skip.head.init, MakeDefaultConfigKey("skip.head.init"));
            anet::nn::ValidateLinearProjectionConfig(skip.head.projection, MakeDefaultConfigKey("skip.head.projection"));
            if (skip_net.Has("DefaultDQNAgent.skip.net.body.output.[features]")) {
                (void)anet::nn::NetworkConfig(skip_net, "DefaultDQNAgent.skip.net");
            }

            // 学習も読み込みもしないRunは初期重みの評価になる。意図的な構成もあり得るので止めず、書き忘れ検出用に1度だけWARNする。
            if (!learner.enabled && auto_load_file.empty()) {
                anet::log::warn()
                    << "learner.enabled=false without DefaultDQNAgent.auto_load_file; "
                    << "the run evaluates freshly initialized weights.";
            }

            // Agent直下の分布表現を3つのpolicyへ引き継ぎ、QR幅だけをQR learnerへ渡す。
            for (auto& [key, entry] : actor) {
                entry.policy.quantile_mode = quantile_mode;
                entry.skip_policy = entry.policy;
                ReadActionPolicyConfig(config_data, "actor.[" + key + "].skip.policy", entry.skip_policy);
            }
            target_policy.quantile_mode = quantile_mode;
            learner.quantile_mode = quantile_mode;
            learner.num_quantiles = qr.num_quantiles;

            // 現行の分布表現契約と全tau ruleを、利用前の設定境界で検証する。
            if (quantile_mode != ActionPolicyConfig::kQuantileModeNone && quantile_mode != ActionPolicyConfig::kQuantileModeQr && quantile_mode != ActionPolicyConfig::kQuantileModeIqn) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.quantile_mode: value='" << quantile_mode
                    << "' expected one of: " << ActionPolicyConfig::kQuantileModeNone << ", "
                    << ActionPolicyConfig::kQuantileModeQr << ", " << ActionPolicyConfig::kQuantileModeIqn);
            }
            if (quantile_mode == ActionPolicyConfig::kQuantileModeQr && qr.num_quantiles <= 1) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.qr.num_quantiles: value=" << qr.num_quantiles
                    << " expected > 1 when quantile_mode=" << ActionPolicyConfig::kQuantileModeQr);
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
                    << target_policy.policy_type << "'; expected " << kActionPolicyTypeStr_Greedy << ", "
                    << kActionPolicyTypeStr_EpsilonGreedy << ", or " << kActionPolicyTypeStr_UQE << ". use_optimistic_target="
                    << (use_optimistic_target ? "true (actor.[train].policy copy before target overrides)" : "false"));
            }
            ValidateActionPolicy(target_policy, "DefaultDQNAgent.target_policy");
            for (auto& [key, entry] : actor) {
                ValidateActionPolicy(entry.policy, "DefaultDQNAgent.actor.[" + key + "].policy");
                ValidateActionPolicy(entry.skip_policy, "DefaultDQNAgent.actor.[" + key + "].skip.policy");
            }

            if (quantile_mode == ActionPolicyConfig::kQuantileModeIqn
                && (!std::isfinite(learner.quantile_huber_kappa) || learner.quantile_huber_kappa <= 0.0f)) {
                ANET_SYSTEM_ERROR("Invalid DefaultDQNAgent.learner.quantile_huber_kappa: value="
                    << learner.quantile_huber_kappa << " expected finite value > 0 when quantile_mode="
                    << ActionPolicyConfig::kQuantileModeIqn);
            }
        }
    private:
        void ValidateForwardReference(
            const ConfigData& data, const std::string& reference, std::string& key)
        {
            // 参照の存在と接続先だけを確認し、Module 固有の実行設定には触れない。
            if (!HasConfigValue(data, reference)) {
                ANET_SYSTEM_ERROR("Missing " << MakeDefaultConfigKey(reference) << "; expected a declared nn_forward key.");
            }
            ReadConfig(data, reference, key);
            const auto it = nn_forward.find(key);
            if (it == nn_forward.end()) {
                ANET_SYSTEM_ERROR("Invalid " << MakeDefaultConfigKey(reference) << "='" << key
                    << "'; expected a declared nn_forward key.");
            }
        }

        void ReadActionPolicyConfig(const ConfigData& data, const std::string& prefix, ActionPolicyConfig& policy)
        {
            ReadConfig(data, prefix, policy);
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
        std::unique_ptr<AgentStepDiagnosticsProvider> CreateStepDiagnosticsProvider(int64_t num_lanes) const override;
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
        std::unique_ptr<FrameStacker> CreateFrameStacker(const ActorRequest& request) const;
        BatchUpdateResultList UpdateFromBatch(const StepCounts& step, const BatchExperience& expriences);
    private:
        std::shared_ptr<anet::rl::dqn::ActionPolicy> CreateActionPolicy(
            const ActionPolicyConfig& policy_config, bool enable_spatial_exploration, int64_t num_envs, const torch::Device& device) const;
        void LoadNetwork(const std::string& filename);
    private:
        DefaultDQNAgentConfig config_;
        std::map<std::string, std::shared_ptr<const anet::nn::NetworkForwardConfig>> forward_configs_; ///< Agent 所有の実行設定 Resource
        std::unique_ptr<anet::rl::dqn::RuntimeVars> vars_;
        std::unique_ptr<anet::rl::dqn::NetworkModel> model_;
        std::shared_ptr<anet::rl::ObservationNormalizer> obs_norm_ = nullptr;
        std::unique_ptr<anet::rl::RewardScaler> reward_scaler_ = nullptr;
        std::shared_ptr<anet::rl::dqn::ActionPolicy> target_policy_;    ///< 学習時ターゲット用ポリシー
        std::shared_ptr<anet::RandomGenerator> plasticity_probe_random_; ///< plasticity probe 専用 Resource
        std::shared_ptr<anet::RandomGenerator> policy_churn_probe_random_; ///< policy churn probe 専用 Resource
        std::shared_ptr<anet::RandomGenerator> noisy_probe_random_; ///< Noisy probe の一様抽選専用 Resource
        std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe_forward_config_; ///< 固定の probe 用途を選ぶ不変 Resource
        std::shared_ptr<anet::RandomGenerator> attention_entropy_probe_random_; ///< attention probeの一様抽選専用Resource
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
