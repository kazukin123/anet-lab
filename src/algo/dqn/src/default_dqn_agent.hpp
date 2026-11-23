// default_dqn_agent.hpp
#pragma once

#include <algorithm>
#include <initializer_list>
#include <string_view>
#include <cmath>
#include <memory>
#include <sstream>
#include <torch/torch.h>

#include "anet/config.hpp"
#include "anet/log.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/rl.hpp"
#include "dqn_config.hpp"
#include "anet/scaler.hpp"
#include "anet/nn.hpp"

namespace anet::rl {
    class FrameStacker;
}

namespace anet::rl::dqn {

    // ======================================================
    // DefaultDQNAgentConfig
    // ======================================================

    struct DefaultDQNAgentConfig : public anet::Config {
        static constexpr char kValueModeCombined[] = "combined";
        static constexpr char kValueModeDecomposed[] = "decomposed";
        static constexpr char kHeadInputShared[] = "shared";
        static constexpr char kHeadInputDedicated[] = "dedicated";
        static constexpr char kIntrinsicObservationNextState[] = "next_state";
        static constexpr char kIntrinsicObservationState[] = "state";
        static constexpr char kIntrinsicLearnIntake[] = "intake";
        static constexpr char kIntrinsicLearnRollout[] = "rollout";
        static constexpr bool kDefaultIntrinsicUseDoneMask = true;
        std::map<std::string, anet::ConfigData> nn_forward;
        NetworkModelConfig model;
        ConfigData perception_config;
        std::map<std::string, DQNActorConfig> actor;
        ActionPolicyConfig target_policy;
        LearnerConfig learner;
        RewardScalerConfig reward_scaler;
        struct {
            bool enabled = false;
            std::string value_mode = kValueModeCombined;
            anet::ProfiledValueConfig<float> beta{.value = 0.1f, .min_value = 0.0f};
            std::string observation = kIntrinsicObservationNextState;
            bool use_done_mask = kDefaultIntrinsicUseDoneMask;
            struct {
                std::string mode = kIntrinsicLearnIntake;
                struct {
                    float ratio = 0.25f;
                    int batch_size = 32;
                } intake;
                struct {
                    int length = 128;
                    int epochs = 4;
                    double ratio = 1.0;
                    int batch_size = 1024;
                    bool use_spread = false;
                } rollout;
            } learn;
            anet::ConfigData rnd; ///< Factory へ渡す opaque な構築設定（共有 NN カタログを含む）。
            RewardScalerConfig reward_scaler{.use_dynamic_scaling = false, .use_auto_post_scale = false};
            struct {
                struct {
                    std::string input_mode = kHeadInputShared;
                    bool use_dueling = true;
                    anet::nn::WeightInitConfig init;
                    anet::ConfigData projection;
                } head;
            } value;
        } intrinsic;
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
            ReadIntrinsic(config_data);
            ReadHeadAndCommon(config_data);
            ReadForwardCatalog(config_data);
            const auto actor_read_scopes = ReadActors(config_data);
            ReadTargetPolicy(config_data);
            ReadLearner(config_data);
            ReadRewardScaler(config_data);
            perception_config = MakeReadScope(config_data, "perception");
            ReadHeadSelection(config_data);
            ReadSkip(config_data);
            ReadValueDecomposition(config_data, actor_read_scopes);
            ReportFreshEvaluation();
            FinalizeActors(actor_read_scopes);
            ValidateCombinations();
            model.full_keys = GetFullKeys().Extract("model");
            skip.full_keys = GetFullKeys().Extract("skip");
            learner.full_keys = GetFullKeys().Extract("learner");
            learner.full_keys.CopyFieldFrom("quantile_mode", GetFullKeys(), "quantile_mode");
            learner.full_keys.CopyFieldFrom("num_quantiles", GetFullKeys(), "qr.num_quantiles");
        }
    private:
        // 元の接頭辞と記録先を明示し、opaque な値の型と受け取る部品のキー形式を保つ。
        void CopyAndRecordPrefix(const ConfigData& source, const std::string& source_prefix,
            const std::string& record_prefix, ConfigData* destination = nullptr, bool preserve_source_key = false)
        {
            for (const auto& [key, value] : source.Map()) {
                if (!key.starts_with(source_prefix)) continue;
                const auto suffix = key.substr(source_prefix.size());
                if (destination)
                    destination->CopyValueFrom(preserve_source_key ? key : suffix, source, key);
                RecordConfigValue(record_prefix + suffix, value);
            }
        }

        static void ValidateChoice(const std::string& key, const std::string& value,
            std::initializer_list<std::string_view> choices)
        {
            if (std::find(choices.begin(), choices.end(), value) != choices.end()) return;
            std::ostringstream expected;
            size_t index = 0;
            for (const auto choice : choices) {
                if (index > 0)
                    expected << (index + 1 == choices.size() ? (choices.size() == 2 ? " or " : ", or ") : ", ");
                expected << choice;
                ++index;
            }
            ANET_SYSTEM_ERROR("Invalid " << key << "='" << value << "'; expected " << expected.str() << ".");
        }

        template<typename T>
        static void ValidateRange(bool valid, const std::string& key, T value, const std::string& expected)
        {
            if (!valid)
                ANET_SYSTEM_ERROR("Invalid " << key << "=" << value << "; expected " << expected << ".");
        }

        template<typename T>
        void ReportIgnoredValue(const ConfigData& data, const std::string& key, const T& value,
            const T& default_value, bool used, const std::string& reason) const
        {
            if (!used && HasConfigValue(data, key) && value != default_value)
                anet::log::warn_once(GetFullKey(key)) << "Ignored " << GetFullKey(key)
                    << "=" << anet::json(value).dump() << "; default=" << anet::json(default_value).dump()
                    << "; reason=" << reason << ".";
        }

        void ReadIntrinsic(const ConfigData& config_data)
        {
            ANET_READ_CONFIG(config_data, intrinsic.value_mode);
            ValidateChoice(GetFullKey("intrinsic.value_mode"), intrinsic.value_mode, {kValueModeCombined, kValueModeDecomposed});
            ANET_READ_CONFIG(config_data, intrinsic.enabled);
            // 休眠中も型と列挙を検査し、利用する部品だけ組み合わせを検証する。
            // 組み合わせの検査だけを、Head・Actorを含む全キーの値の検証後へ残す。
            anet::ConfigReader<RewardScalerConfig>::ReadFields(*this, config_data, "intrinsic.reward_scaler", intrinsic.reward_scaler, intrinsic.enabled);
            if (intrinsic.enabled) {
                ANET_READ_CONFIG(config_data, intrinsic.beta);
                ANET_READ_CONFIG(config_data, intrinsic.observation);
                ANET_READ_CONFIG(config_data, intrinsic.use_done_mask);
                ValidateChoice(GetFullKey("intrinsic.observation"), intrinsic.observation,
                    {kIntrinsicObservationNextState, kIntrinsicObservationState});
                // state は終端の観測を採点しないので、この指定を使わない。
                if (intrinsic.observation == kIntrinsicObservationState
                    && intrinsic.use_done_mask != kDefaultIntrinsicUseDoneMask)
                    anet::log::warn_once() << "Ignored " << GetFullKey("intrinsic.use_done_mask") << "=" << anet::json(intrinsic.use_done_mask).dump()
                        << " with " << GetFullKey("intrinsic.observation") << "=" << intrinsic.observation
                        << "; default=" << anet::json(kDefaultIntrinsicUseDoneMask).dump() << ".";
                ANET_READ_CONFIG(config_data, intrinsic.learn.mode);
                ANET_READ_CONFIG(config_data, intrinsic.learn.intake.ratio);
                ANET_READ_CONFIG(config_data, intrinsic.learn.intake.batch_size);
                ANET_READ_CONFIG(config_data, intrinsic.learn.rollout.length);
                ANET_READ_CONFIG(config_data, intrinsic.learn.rollout.epochs);
                ANET_READ_CONFIG(config_data, intrinsic.learn.rollout.ratio);
                ANET_READ_CONFIG(config_data, intrinsic.learn.rollout.batch_size);
                ANET_READ_CONFIG(config_data, intrinsic.learn.rollout.use_spread);
                ValidateChoice(GetFullKey("intrinsic.learn.mode"), intrinsic.learn.mode,
                    {kIntrinsicLearnIntake, kIntrinsicLearnRollout});
                ValidateRange(std::isfinite(intrinsic.learn.intake.ratio) && intrinsic.learn.intake.ratio > 0.0f && intrinsic.learn.intake.ratio <= 1.0f,
                    GetFullKey("intrinsic.learn.intake.ratio"), intrinsic.learn.intake.ratio, "finite 0 < value <= 1");
                ValidateRange(intrinsic.learn.intake.batch_size >= 1,
                    GetFullKey("intrinsic.learn.intake.batch_size"), intrinsic.learn.intake.batch_size, "integer >= 1");
                const auto& rollout = intrinsic.learn.rollout;
                ValidateRange(rollout.length >= 1, GetFullKey("intrinsic.learn.rollout.length"),
                    rollout.length, "integer >= 1");
                ValidateRange(rollout.epochs >= 1, GetFullKey("intrinsic.learn.rollout.epochs"),
                    rollout.epochs, "integer >= 1");
                ValidateRange(rollout.batch_size >= 1, GetFullKey("intrinsic.learn.rollout.batch_size"),
                    rollout.batch_size, "integer >= 1");
                if (!std::isfinite(rollout.ratio) || rollout.ratio <= 0.0 || rollout.ratio > rollout.epochs)
                    ANET_SYSTEM_ERROR("Invalid " << GetFullKey("intrinsic.learn.rollout.ratio") << "=" << rollout.ratio
                        << " with " << GetFullKey("intrinsic.learn.rollout.epochs") << "=" << rollout.epochs
                        << "; expected finite 0 < ratio <= epochs.");
                intrinsic.rnd = config_data;
                CopyAndRecordPrefix(config_data, MakeDefaultConfigKey("intrinsic.rnd") + ".", "intrinsic.rnd.");
            } else {
                config_data.MarkInventory(MakeDefaultConfigKey("intrinsic"));
                anet::log::info_once() << "Inactive config prefix=DefaultDQNAgent.intrinsic; reason=DefaultDQNAgent.intrinsic.enabled=false.";
                // 無効時の checkpoint 内の設定文字列を従来と同一にする。
                my_config_json_.erase("intrinsic.enabled");
            }
        }

        void ReadValueDecomposition(const ConfigData& config_data,
            const std::map<std::string, ConfigData>& actor_read_scopes)
        {
            // 認識した値は休眠中も検証し、有効なときだけ構築へ使う。
            learner.intrinsic.decomposed = intrinsic.value_mode == kValueModeDecomposed;
            ANET_READ_CONFIG(config_data, learner.intrinsic.discount_mode);
            const auto gamma_key = MakeDefaultConfigKey("learner.intrinsic.gamma");
            const auto gamma_override = MakeOverrideConfigKey("learner.intrinsic.gamma");
            bool has_gamma = false;
            for (const auto& [key, value] : config_data.Map()) {
                if (key == gamma_key || key.starts_with(gamma_key + ".")
                    || (!GetOverridePrefix().empty()
                        && (key == gamma_override || key.starts_with(gamma_override + ".")))) {
                    has_gamma = true;
                    break;
                }
            }
            ANET_READ_CONFIG(config_data, learner.intrinsic.terminal_mode);
            ANET_READ_CONFIG(config_data, learner.intrinsic.loss_weight);
            ANET_READ_CONFIG(config_data, learner.intrinsic.priority_lambda);
            ANET_READ_CONFIG(config_data, learner.intrinsic.extrinsic_bootstrap_mode);
            ANET_READ_CONFIG(config_data, learner.intrinsic.feature_gradient);
            ValidateChoice(GetFullKey("learner.intrinsic.discount_mode"), learner.intrinsic.discount_mode, {LearnerConfig::kDiscountShared, LearnerConfig::kDiscountSeparate});
            ValidateChoice(GetFullKey("learner.intrinsic.terminal_mode"), learner.intrinsic.terminal_mode, {LearnerConfig::kTerminalEpisodic, LearnerConfig::kTerminalNonEpisodic});
            ValidateChoice(GetFullKey("learner.intrinsic.extrinsic_bootstrap_mode"), learner.intrinsic.extrinsic_bootstrap_mode, {LearnerConfig::kBootstrapCombined, LearnerConfig::kBootstrapExtrinsic});
            ValidateChoice(GetFullKey("learner.intrinsic.feature_gradient"), learner.intrinsic.feature_gradient,
                {LearnerConfig::kFeatureGradientMerge, LearnerConfig::kFeatureGradientSplit});
            ANET_READ_CONFIG(config_data, intrinsic.value.head.input_mode);
            ANET_READ_CONFIG(config_data, intrinsic.value.head.use_dueling);
            const anet::nn::WeightInitConfig init_defaults{.nonlinearity = "linear"};
            intrinsic.value.head.init = init_defaults;
            anet::ConfigReader<anet::nn::WeightInitConfig>::ReadFields(
                *this, config_data, "intrinsic.value.head.init", intrinsic.value.head.init);
            anet::nn::ValidateWeightInitConfigChoices(intrinsic.value.head.init);
            ValidateChoice(GetFullKey("intrinsic.value.head.input_mode"), intrinsic.value.head.input_mode, {kHeadInputShared, kHeadInputDedicated});
            CopyAndRecordPrefix(config_data, MakeDefaultConfigKey("intrinsic.value.head.projection") + ".",
                "intrinsic.value.head.projection.", &intrinsic.value.head.projection);
            const bool decomposed = learner.intrinsic.decomposed;
            // 全Actorの局所的な値を先に検証し、別Actorの未指定や機能の組み合わせより優先する。
            for (auto& [key, entry] : actor) {
                const auto prefix = "actor.[" + key + "].intrinsic";
                const auto& scope = actor_read_scopes.at(key);
                ReadConfig(scope, prefix + ".action_beta_mode", entry.intrinsic.action_beta_mode);
                ReadConfig(scope, prefix + ".action_beta", entry.intrinsic.action_beta);
                ValidateChoice(GetFullKey(prefix + ".action_beta_mode"), entry.intrinsic.action_beta_mode, {DQNActorConfig::kActionBetaLearner, DQNActorConfig::kActionBetaFixed});
            }
            // モード列挙をすべて確かめてから、数値とHead固有の構造キーを検証する。
            if (has_gamma) ANET_READ_CONFIG(config_data, learner.intrinsic.gamma);
            ValidateRange(std::isfinite(learner.intrinsic.loss_weight) && learner.intrinsic.loss_weight > 0,
                GetFullKey("learner.intrinsic.loss_weight"), learner.intrinsic.loss_weight, "finite > 0");
            ValidateRange(std::isfinite(learner.intrinsic.priority_lambda) && learner.intrinsic.priority_lambda >= 0,
                GetFullKey("learner.intrinsic.priority_lambda"), learner.intrinsic.priority_lambda, "finite >= 0");
            anet::nn::ValidateWeightInitConfig(intrinsic.value.head.init);
            anet::nn::ValidateLinearProjectionConfig(intrinsic.value.head.projection, MakeDefaultConfigKey("intrinsic.value.head.projection"));
            for (const auto& [key, entry] : actor) {
                if (entry.intrinsic.action_beta)
                    ValidateRange(std::isfinite(*entry.intrinsic.action_beta) && *entry.intrinsic.action_beta >= 0,
                        GetFullKey("actor.[" + key + "].intrinsic.action_beta"), *entry.intrinsic.action_beta, "finite >= 0");
            }
            // RFDの判定表に従い、値の検証後に組み合わせ、gamma、Actorの必須値を順に判定する。
            anet::rl::ValidateRewardScalerConfig(reward_scaler);
            anet::rl::ValidateRewardScalerConfig(intrinsic.reward_scaler, intrinsic.enabled);
            if (decomposed && (!intrinsic.enabled || skip.enabled))
                ANET_SYSTEM_ERROR("Invalid " << GetFullKey("intrinsic.value_mode") << "=" << intrinsic.value_mode
                    << "; expected " << GetFullKey("intrinsic.enabled") << "=true and " << GetFullKey("skip.enabled")
                    << "=false; actual " << GetFullKey("intrinsic.enabled") << "="
                    << intrinsic.enabled << " " << GetFullKey("skip.enabled") << "=" << skip.enabled << ".");
            if (decomposed && learner.intrinsic.discount_mode == LearnerConfig::kDiscountSeparate && !has_gamma)
                ANET_SYSTEM_ERROR("Missing " << gamma_key << "; expected explicit value for "
                    << GetFullKey("learner.intrinsic.discount_mode") << "=separate.");
            for (const auto& [key, entry] : actor) {
                if (decomposed && entry.intrinsic.action_beta_mode == DQNActorConfig::kActionBetaFixed && !entry.intrinsic.action_beta) {
                    const auto prefix = "actor.[" + key + "].intrinsic.";
                    const auto beta_key = config_data.GetFullKey(MakeDefaultConfigKey(prefix + "action_beta"));
                    ANET_SYSTEM_ERROR("Missing " << beta_key
                        << "; expected explicit value for " << GetFullKey(prefix + "action_beta_mode")
                        << "=" << DQNActorConfig::kActionBetaFixed << ".");
                }
            }
            const std::string inactive_reason = GetFullKey("intrinsic.value_mode") + "=combined";
            const LearnerConfig defaults;
            ReportIgnoredValue(config_data, "learner.intrinsic.discount_mode", learner.intrinsic.discount_mode,
                defaults.intrinsic.discount_mode, decomposed, inactive_reason);
            ReportIgnoredValue(config_data, "learner.intrinsic.terminal_mode", learner.intrinsic.terminal_mode,
                defaults.intrinsic.terminal_mode, decomposed, inactive_reason);
            ReportIgnoredValue(config_data, "learner.intrinsic.loss_weight", learner.intrinsic.loss_weight,
                defaults.intrinsic.loss_weight, decomposed, inactive_reason);
            ReportIgnoredValue(config_data, "learner.intrinsic.priority_lambda", learner.intrinsic.priority_lambda,
                defaults.intrinsic.priority_lambda, decomposed, inactive_reason);
            ReportIgnoredValue(config_data, "learner.intrinsic.extrinsic_bootstrap_mode", learner.intrinsic.extrinsic_bootstrap_mode,
                defaults.intrinsic.extrinsic_bootstrap_mode, decomposed, inactive_reason);
            ReportIgnoredValue(config_data, "learner.intrinsic.feature_gradient", learner.intrinsic.feature_gradient,
                defaults.intrinsic.feature_gradient, decomposed, inactive_reason);
            if (has_gamma && (!decomposed || learner.intrinsic.discount_mode == LearnerConfig::kDiscountShared))
                anet::log::warn_once(gamma_key) << "Ignored " << gamma_key << "; " << learner.intrinsic.gamma.full_keys.GetFullKey("type") << "=" << learner.intrinsic.gamma.type
                    << " value=" << learner.intrinsic.gamma.value
                    << "; expected " << GetFullKey("learner.intrinsic.discount_mode") << "=separate and "
                    << GetFullKey("intrinsic.value_mode") << "=decomposed; actual " << GetFullKey("learner.intrinsic.discount_mode") << "="
                    << learner.intrinsic.discount_mode << " " << GetFullKey("intrinsic.value_mode") << "=" << intrinsic.value_mode << ".";
            ReportIgnoredValue(config_data, "intrinsic.value.head.input_mode", intrinsic.value.head.input_mode,
                std::string(kHeadInputShared), decomposed, inactive_reason);
            ReportIgnoredValue(config_data, "intrinsic.value.head.use_dueling", intrinsic.value.head.use_dueling,
                true, decomposed, inactive_reason);
            const auto report_init = [&](const std::string& key, const auto& value, const auto& default_value) {
                ReportIgnoredValue(config_data, "intrinsic.value.head.init." + key, value, default_value, decomposed, inactive_reason);
            };
            report_init("mode", intrinsic.value.head.init.mode, init_defaults.mode);
            report_init("manual_gain", intrinsic.value.head.init.manual_gain, init_defaults.manual_gain);
            report_init("nonlinearity", intrinsic.value.head.init.nonlinearity, init_defaults.nonlinearity);
            report_init("constant_val", intrinsic.value.head.init.constant_val, init_defaults.constant_val);
            report_init("trunc_std", intrinsic.value.head.init.trunc_std, init_defaults.trunc_std);
            report_init("trunc_a", intrinsic.value.head.init.trunc_a, init_defaults.trunc_a);
            report_init("trunc_b", intrinsic.value.head.init.trunc_b, init_defaults.trunc_b);
            report_init("variance_scale", intrinsic.value.head.init.variance_scale, init_defaults.variance_scale);
            anet::nn::ReportUnusedLinearProjectionConfig(intrinsic.value.head.projection,
                MakeDefaultConfigKey("intrinsic.value.head.projection"), decomposed, inactive_reason);
            for (const auto& [key, entry] : actor) {
                const auto prefix = "actor.[" + key + "].intrinsic";
                const auto& scope = actor_read_scopes.at(key);
                ReportIgnoredValue(scope, prefix + ".action_beta_mode", entry.intrinsic.action_beta_mode,
                    std::string(DQNActorConfig::kActionBetaLearner), decomposed, inactive_reason);
                if (entry.intrinsic.action_beta && (!decomposed || entry.intrinsic.action_beta_mode == DQNActorConfig::kActionBetaLearner))
                    anet::log::warn_once(GetFullKey(prefix + ".action_beta"))
                        << "Ignored " << GetFullKey(prefix + ".action_beta") << "=" << *entry.intrinsic.action_beta
                        << "; expected " << GetFullKey("intrinsic.value_mode") << "=decomposed and "
                        << GetFullKey(prefix + ".action_beta_mode") << "=fixed; actual " << GetFullKey("intrinsic.value_mode") << "="
                        << intrinsic.value_mode << " " << GetFullKey(prefix + ".action_beta_mode") << "=" << entry.intrinsic.action_beta_mode << ".";
            }
        }

        void ReadHeadAndCommon(const ConfigData& config_data)
        {
            // Head 射影の構築設定は opaque にコピーし、固有の項目を Agent で読み取らない。
            CopyAndRecordPrefix(config_data, MakeDefaultConfigKey("head.projection") + ".",
                "head.projection.", &head.projection);

            head.init.nonlinearity = "linear";
            ANET_READ_CONFIG(config_data, head.init);

            ANET_READ_CONFIG(config_data, nn_viz.show_param_shapes);
            ANET_READ_CONFIG(config_data, nn_viz.show_param_count);
            ANET_READ_CONFIG(config_data, nn_viz.show_tensor_specs);
            ANET_READ_CONFIG(config_data, nn_viz.show_branch_config);
            ANET_READ_CONFIG(config_data, nn_viz.show_head_info);
            ANET_READ_CONFIG(config_data, nn_viz.layout);
            ANET_READ_CONFIG(config_data, nn_viz.cluster_branches);
            ANET_READ_CONFIG(config_data, nn_viz.float_precision);
            anet::nn::ValidateNetworkGraphVizConfig(nn_viz, "DefaultDQNAgent.nn_viz");

            ANET_READ_CONFIG(config_data, model.soft_update_tau);
            ANET_READ_CONFIG(config_data, model.hard_update_interval);

            ANET_READ_CONFIG(config_data, use_optimistic_target);
        }

        void ReadForwardCatalog(const ConfigData& config_data)
        {
            // カタログの値はDTOへ保存し、使用する具象部品が実行準備で検証する。
            for (const auto& [key, sub] : config_data.MakeSubConfigData(MakeDefaultConfigKey("nn_forward"))) {
                config_data.MarkInventory(MakeDefaultConfigKey(MakeNetworkForwardConfigKey(key)));
                nn_forward.emplace(key, sub);
                // 記録も opaque な値のまま保存し、型付きの解釈はここで追加しない。
                CopyAndRecordPrefix(sub, "", MakeNetworkForwardConfigKey(key) + ".");
            }
        }

        std::map<std::string, ConfigData> ReadActors(const ConfigData& config_data)
        {
            // 全カタログ項目を構築時に型付き設定へ変換する。
            std::map<std::string, ConfigData> actor_read_scopes;
            for (const auto& [key, sub] : config_data.MakeSubConfigData(MakeDefaultConfigKey("actor"))) {
                auto& entry = actor[key];
                const auto prefix = "actor.[" + key + "]";
                const auto& actor_data = actor_read_scopes.emplace(key, MakeReadScope(config_data, prefix)).first->second;
                for (const auto& [leaf, value] : sub.Map()) {
                    if (leaf.starts_with("skip.policy.")) entry.has_skip_policy = true;
                }
                ReadActionPolicyConfig(actor_data, prefix + ".policy", entry.policy);
                ReadConfig(actor_data, prefix + ".network", entry.network);
                ReadConfig(actor_data, prefix + ".clone_model", entry.clone_model);
                ValidateForwardReference(actor_data, prefix + ".nn_forward_key", entry.nn_forward_key);
                ValidateChoice(GetFullKey(prefix + ".network"), entry.network, {"online", "target"});
                for (const auto& [leaf, value] : sub.Map()) {
                    if (leaf == "sync_interval" || leaf.starts_with("sync_interval.")) {
                        entry.sync_interval.emplace(anet::ProfiledValueConfig<step_t>{
                            .type = "constant", .value = 400, .min_value = 1});
                        ReadConfig(actor_data, prefix + ".sync_interval", *entry.sync_interval);
                        break;
                    }
                }
            }
            return actor_read_scopes;
        }

        void ReadTargetPolicy(const ConfigData& config_data)
        {
            target_policy.policy_type = kActionPolicyTypeStr_Greedy;     // デフォルトは安全なGreedy
            target_policy.epsilon = {.value = 0.0f, .min_value = 0.0f, .max_value = 1.0f};
            if (use_optimistic_target) {        // 「use_optimistic_target = true」だった場合、target_policyのデフォルトはactor.[train].policyをベースとする
                const auto train = actor.find("train");
                if (train == actor.end()) {
                    ANET_SYSTEM_ERROR(GetFullKey("use_optimistic_target") << "=true requires " << MakeDefaultConfigKey("actor.[train].policy") << ".");
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
                    anet::log::warn_once() << GetFullKey("use_optimistic_target") << "=true copies " << GetFullKey("actor.[train].policy.policy_type") << "='"
                        << train->second.policy.policy_type << "' to " << config_data.GetFullKey(MakeDefaultConfigKey("target_policy.policy_type")) << "='"
                        << target_policy.policy_type
                        << "' without optimism. Select UQE/ThompsonSampling or " << GetFullKey("use_optimistic_target") << "=false.";
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
        }

        void ReadLearner(const ConfigData& config_data)
        {
            // 学習停止中も宣言された参照を解決する。保持や時計の適否は実際の利用時に委譲する。
            ValidateForwardReference(
                config_data, "learner.current.nn_forward_key", learner.current.nn_forward_key);
            ValidateForwardReference(
                config_data, "learner.target.nn_forward_key", learner.target.nn_forward_key);

            ANET_READ_CONFIG(config_data, learner.enabled);
            ANET_READ_CONFIG(config_data, learner.learning_rate);
            ANET_READ_CONFIG(config_data, learner.weight_decay);
            ANET_READ_CONFIG(config_data, learner.noisy_weight_decay);
            ValidateRange(std::isfinite(learner.noisy_weight_decay) && learner.noisy_weight_decay >= 0.0f,
                GetFullKey("learner.noisy_weight_decay"), learner.noisy_weight_decay, "finite value >= 0");
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
            ValidateChoice(GetFullKey("learner.munchausen.log_policy_mode"), munchausen.log_policy_mode,
                {MunchausenValues::kLogPolicyTarget, MunchausenValues::kLogPolicyOnline, MunchausenValues::kLogPolicyOnlineReuse});
            ValidateRange(std::isfinite(munchausen.clip_value_min) && munchausen.clip_value_min <= 0.0f,
                GetFullKey("learner.munchausen.clip_value_min"), munchausen.clip_value_min, "finite <= 0");
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
            RecordFullKey("learner.noisy_probe", config_data.GetFullKey(MakeDefaultConfigKey("learner.noisy_probe")));
            ValidateRange(learner.noisy_probe.probe.batch_size >= 1,
                GetFullKey("learner.noisy_probe.probe.batch_size"), learner.noisy_probe.probe.batch_size, "integer >= 1");
            ValidateRange(learner.noisy_probe.num_draws >= 1,
                GetFullKey("learner.noisy_probe.num_draws"), learner.noisy_probe.num_draws, "integer >= 1");
            ValidateRange(learner.noisy_probe.iqn.num_taus >= 1,
                GetFullKey("learner.noisy_probe.iqn.num_taus"), learner.noisy_probe.iqn.num_taus, "integer >= 1");
            // 個々の値域は購読前に検証し、batchやAMPとの条件は購読の準備時に扱う。
            ANET_READ_CONFIG(config_data, learner.grad_noise.num_micro_batches);
            ANET_READ_CONFIG(config_data, learner.grad_noise.ema_alpha);
            learner.grad_noise.Validate(GetFullKeys(), "learner.grad_noise.");
            ANET_READ_CONFIG(config_data, learner.final_feature_branch_key);
            ANET_READ_CONFIG(config_data, learner.attention_entropy.probe.batch_size);
            ValidateRange(learner.attention_entropy.probe.batch_size >= 1,
                GetFullKey("learner.attention_entropy.probe.batch_size"), learner.attention_entropy.probe.batch_size, "integer >= 1");
            ANET_READ_CONFIG(config_data, learner.plasticity.probe.batch_size);
            ANET_READ_CONFIG(config_data, learner.policy_churn.probe.batch_size);
            ANET_READ_CONFIG(config_data, learner.policy_churn.iqn.num_taus);
            ANET_READ_CONFIG(config_data, learner.replay_fit.probe.batch_size);
            ANET_READ_CONFIG(config_data, learner.replay_fit.iqn.num_taus);
            ValidateRange(learner.replay_fit.probe.batch_size >= 1,
                GetFullKey("learner.replay_fit.probe.batch_size"), learner.replay_fit.probe.batch_size, "integer >= 1");
            ValidateRange(learner.replay_fit.iqn.num_taus >= 1,
                GetFullKey("learner.replay_fit.iqn.num_taus"), learner.replay_fit.iqn.num_taus, "integer >= 1");
            ValidateRange(learner.plasticity.probe.batch_size >= 1,
                GetFullKey("learner.plasticity.probe.batch_size"), learner.plasticity.probe.batch_size, "integer >= 1");
            ValidateRange(learner.policy_churn.probe.batch_size >= 1,
                GetFullKey("learner.policy_churn.probe.batch_size"), learner.policy_churn.probe.batch_size, "integer >= 1");
            ValidateRange(learner.policy_churn.iqn.num_taus >= 1,
                GetFullKey("learner.policy_churn.iqn.num_taus"), learner.policy_churn.iqn.num_taus, "integer >= 1");
            ValidateRange(std::isfinite(learner.tbo_epsilon) && learner.tbo_epsilon > 0.0f,
                GetFullKey("learner.tbo_epsilon"), learner.tbo_epsilon, "finite positive float");
        }

        void ReadRewardScaler(const ConfigData& config_data)
        {
            // 内的用と同様に、表の行1・2の全キーを確かめてから行3の組み合わせを検証する。
            anet::ConfigReader<RewardScalerConfig>::ReadFields(*this, config_data, "reward_scaler", reward_scaler);
            if (learner.use_tbo && (reward_scaler.use_dynamic_scaling || reward_scaler.use_auto_post_scale)) {
                anet::log::warn_once()
                    << GetFullKey("learner.use_tbo") << " is enabled together with " << GetFullKey("reward_scaler.use_dynamic_scaling")
                    << " or " << GetFullKey("reward_scaler.use_auto_post_scale") << "; targets may be double-compressed.";
            }
        }

        void ReadHeadSelection(const ConfigData& config_data)
        {
            ANET_READ_CONFIG(config_data, auto_load_file);
            ANET_READ_CONFIG(config_data, quantile_mode);
            ANET_READ_CONFIG(config_data, qr.num_quantiles);
            ANET_READ_CONFIG(config_data, head.use_dueling);
        }

        void ReadSkip(const ConfigData& config_data)
        {
            ANET_READ_CONFIG(config_data, skip.enabled);
            if (!skip.enabled) {
                config_data.MarkInventory(MakeDefaultConfigKey("skip"));
                config_data.MarkInventory(MakeDefaultConfigKey("learner.skip"));
                anet::log::info_once() << "Inactive config prefixes=DefaultDQNAgent.skip,DefaultDQNAgent.learner.skip; reason=DefaultDQNAgent.skip.enabled=false.";
            }
            const bool has_skip_limit = HasConfigValue(config_data, "skip.max_length");
            ANET_READ_CONFIG(config_data, skip.max_length);
            // 文字列の設定を境界で検証し、実行系へは構成を表すenumだけを渡す。
            std::string network_mode = SkipConfig::kNetworkModeActionOut;
            ReadConfig(config_data, "skip.network_mode", network_mode);
            ValidateChoice(GetFullKey("skip.network_mode"), network_mode,
                {SkipConfig::kNetworkModeActionOut, SkipConfig::kNetworkModeActionIn});
            skip.network_mode = network_mode == SkipConfig::kNetworkModeActionOut
                ? SkipNetworkMode::ActionOut : SkipNetworkMode::ActionIn;
            ANET_READ_CONFIG(config_data, skip.head.use_dueling);
            ANET_READ_CONFIG(config_data, learner.skip.feature_gradient);
            ANET_READ_CONFIG(config_data, learner.skip.target_window);
            ANET_READ_CONFIG(config_data, learner.skip.sample_weight);
            ANET_READ_CONFIG(config_data, learner.skip.loss_weight);
            ANET_READ_CONFIG(config_data, learner.skip.priority_lambda);
            if ((skip.enabled && !has_skip_limit) || (has_skip_limit && skip.max_length < 1)) {
                ANET_SYSTEM_ERROR("Invalid " << GetFullKey("skip.max_length") << "=" << skip.max_length
                    << " specified=" << has_skip_limit << "; expected explicit integer >= 1 when "
                    << GetFullKey("skip.enabled") << "=true.");
            }
            ValidateChoice(GetFullKey("learner.skip.feature_gradient"), learner.skip.feature_gradient,
                {LearnerConfig::kFeatureGradientStop, LearnerConfig::kFeatureGradientMerge, LearnerConfig::kFeatureGradientSplit});
            ValidateChoice(GetFullKey("learner.skip.sample_weight"), learner.skip.sample_weight,
                {LearnerConfig::kSampleWeightUniform, LearnerConfig::kSampleWeightKmax});
            ValidateChoice(GetFullKey("learner.skip.target_window"), learner.skip.target_window,
                {LearnerConfig::kTargetWindowDecision, LearnerConfig::kTargetWindowActionChange, LearnerConfig::kTargetWindowEpisodeEnd});
            for (const auto& [key, value] : std::initializer_list<std::pair<const char*, float>>{
                {"loss_weight", learner.skip.loss_weight}, {"priority_lambda", learner.skip.priority_lambda}}) {
                ValidateRange(std::isfinite(value) && value >= 0.0f,
                    GetFullKey("learner.skip." + std::string(key)), value, "finite value >= 0");
            }
            if (skip.enabled && learner.use_amp && !learner.use_amp_bf16) {
                ANET_SYSTEM_ERROR(GetFullKey("skip.enabled") << "=true requires " << GetFullKey("learner.use_amp")
                    << "=false or " << GetFullKey("learner.use_amp_bf16") << "=true.");
            }
            skip.head.projection = head.projection;
            CopyAndRecordPrefix(config_data, MakeDefaultConfigKey("skip.head.projection") + ".",
                "skip.head.projection.", &skip.head.projection);
            CopyAndRecordPrefix(config_data, MakeDefaultConfigKey("skip.net") + ".",
                "skip.net.", &skip_net, true);
            const bool has_skip_net = skip_net.Map().begin() != skip_net.Map().end();
            if (has_skip_net) {
                RecordFullKey("skip.net", config_data.GetFullKey(MakeDefaultConfigKey("skip.net")));
            }
            if (skip.enabled && !has_skip_net) {
                ANET_SYSTEM_ERROR("Missing " << config_data.GetFullKey(MakeDefaultConfigKey("skip.net"))
                    << "; expected a network definition when " << GetFullKey("skip.enabled") << "=true.");
            }
            // skip NetworkConfig も共有の block・profile カタログを同じ完全キーで解決する。
            for (const auto& [key, value] : config_data.Map()) {
                if (key.starts_with("net.block.") || key.starts_with("net.config_profile.")) skip_net.CopyValueFrom(key, config_data, key);
            }
            skip.head.init = head.init;
            ANET_READ_CONFIG(config_data, skip.head.init);
            anet::nn::ValidateLinearProjectionConfig(skip.head.projection, MakeDefaultConfigKey("skip.head.projection"));
            if (skip_net.Has("DefaultDQNAgent.skip.net.body.output.[features]")) {
                (void)anet::nn::NetworkConfig(skip_net, "DefaultDQNAgent.skip.net");
            }
        }

        void ReportFreshEvaluation()
        {
            // 学習も読み込みもしないRunは初期重みの評価になる。意図的な構成もあり得るので止めず、書き忘れ検出用に1度だけWARNする。
            if (!learner.enabled && auto_load_file.empty()) {
                anet::log::warn_once()
                    << GetFullKey("learner.enabled") << "=false without " << GetFullKey("auto_load_file") << "; "
                    << "the run evaluates freshly initialized weights.";
            }
        }

        void FinalizeActors(const std::map<std::string, ConfigData>& actor_read_scopes)
        {
            // Agent直下の分布表現を3つのpolicyへ引き継ぎ、QR幅だけをQR learnerへ渡す。
            std::ostringstream missing_skip_forward;
            for (auto& [key, entry] : actor) {
                const auto& config_data = actor_read_scopes.at(key);
                entry.policy.quantile_mode = quantile_mode;
                const auto skip_prefix = "actor.[" + key + "].skip";
                ReadConfig(config_data, skip_prefix + ".nn_forward_mode", entry.skip.nn_forward_mode);
                ValidateChoice(GetFullKey(skip_prefix + ".nn_forward_mode"), entry.skip.nn_forward_mode,
                    {DQNActorConfig::kForwardShared, DQNActorConfig::kForwardDedicated});
                // 明示された参照だけを保存し、未指定を config dump に追加しない。
                const auto forward_key = skip_prefix + ".nn_forward_key";
                if (HasConfigValue(config_data, forward_key)) {
                    entry.skip.nn_forward_key.emplace();
                    ValidateForwardReference(config_data, forward_key, *entry.skip.nn_forward_key);
                }
                if (skip.enabled && entry.skip.nn_forward_mode == DQNActorConfig::kForwardDedicated
                    && !entry.skip.nn_forward_key) {
                    missing_skip_forward << "Missing " << MakeDefaultConfigKey(forward_key)
                        << "; expected a declared nn_forward key when " << GetFullKey(skip_prefix + ".nn_forward_mode")
                        << "=" << DQNActorConfig::kForwardDedicated << " and " << GetFullKey("skip.enabled") << "=true. ";
                }
                const auto mode_field = skip_prefix + ".max_length_mode";
                ReadConfig(config_data, mode_field, entry.skip.max_length_mode);
                const auto mode_key = GetFullKey(mode_field);
                ValidateChoice(mode_key, entry.skip.max_length_mode,
                    {DQNActorConfig::kMaxLengthSingle, DQNActorConfig::kMaxLengthFull, DQNActorConfig::kMaxLengthVariable});
                // Jが確定したあとで、bareと配下の指定を検出して共通の値域検証へ渡す。
                const auto length_key = skip_prefix + ".max_length";
                const auto default_key = MakeDefaultConfigKey(length_key);
                const auto override_key = MakeOverrideConfigKey(length_key);
                for (const auto& [name, value] : config_data.Map()) {
                    if (name == default_key || name.starts_with(default_key + ".")
                        || (!GetOverridePrefix().empty()
                            && (name == override_key || name.starts_with(override_key + ".")))) {
                        entry.skip.max_length.emplace();
                        entry.skip.max_length->min_value = 1;
                        if (skip.enabled) entry.skip.max_length->max_value = skip.max_length;
                        ReadConfig(config_data, length_key, *entry.skip.max_length);
                        break;
                    }
                }
                if (skip.enabled && entry.skip.max_length_mode == DQNActorConfig::kMaxLengthVariable
                    && !entry.skip.max_length) {
                    ANET_SYSTEM_ERROR("Missing " << default_key << "; expected an integer ProfiledValue in [1, "
                        << skip.max_length << "] when " << mode_key << "=" << DQNActorConfig::kMaxLengthVariable << ".");
                }
                if (skip.enabled && entry.skip.max_length && entry.skip.max_length_mode != DQNActorConfig::kMaxLengthVariable) {
                    const auto& length = *entry.skip.max_length;
                    const auto [lower, upper] = anet::detail::GetProfiledValueActiveRange(length);
                    const auto source_key = length.full_keys.GetFullKey(length.type == "constant" ? "value" : "type");
                    anet::log::warn_once(default_key) << source_key << " active_range=[" << lower << "," << upper
                        << "] is unused with " << mode_key << "=" << entry.skip.max_length_mode
                        << "; expected " << mode_key << "=" << DQNActorConfig::kMaxLengthVariable << " to use this value.";
                }
                entry.skip_policy = entry.policy;
                ReadActionPolicyConfig(config_data, "actor.[" + key + "].skip.policy", entry.skip_policy);
                entry.full_keys = GetFullKeys().Extract("actor.[" + key + "]");
                entry.policy.full_keys.CopyFieldFrom("quantile_mode", GetFullKeys(), "quantile_mode");
                entry.skip_policy.full_keys.CopyFieldFrom("quantile_mode", GetFullKeys(), "quantile_mode");
            }
            // 一つ目で止めず、宣言された全 Actor の不足を構築時にまとめて返す。
            if (!missing_skip_forward.str().empty()) {
                ANET_SYSTEM_ERROR(missing_skip_forward.str());
            }
            for (const auto& [key, data] : actor_read_scopes) {
                data.RequireAllKeysRead();
            }
            // shared の値は使わないことを明示し、意図された無効状態は静かに保つ。
            for (const auto& [key, entry] : actor) {
                if (skip.enabled && entry.skip.nn_forward_mode == DQNActorConfig::kForwardShared
                    && entry.skip.nn_forward_key) {
                    const auto forward_key = entry.full_keys.GetFullKey("skip.nn_forward_key");
                    anet::log::warn_once(forward_key) << forward_key
                        << "='" << *entry.skip.nn_forward_key << "' is unused with " << entry.full_keys.GetFullKey("skip.nn_forward_mode") << "="
                        << DQNActorConfig::kForwardShared << "; expected " << entry.full_keys.GetFullKey("skip.nn_forward_mode") << "="
                        << DQNActorConfig::kForwardDedicated << " to use this value.";
                }
            }
            target_policy.quantile_mode = quantile_mode;
            target_policy.full_keys.CopyFieldFrom("quantile_mode", GetFullKeys(), "quantile_mode");
        }

        void ValidateCombinations()
        {
            // 判断によらない教材は全Actorが常に長さ1のときだけ使い、用途名は解釈しない。
            if (skip.enabled && learner.enabled && learner.skip.target_window != LearnerConfig::kTargetWindowDecision) {
                std::ostringstream offending;
                for (const auto& [key, entry] : actor) {
                    bool invalid = entry.skip.max_length_mode == DQNActorConfig::kMaxLengthFull;
                    if (entry.skip.max_length_mode == DQNActorConfig::kMaxLengthVariable) {
                        auto bounds = *entry.skip.max_length;
                        bounds.max_value = 1;
                        invalid = anet::detail::FindProfiledValueBoundsError(bounds).has_value();
                    }
                    if (!invalid) continue;
                    offending << " " << entry.full_keys.GetFullKey("skip.max_length_mode") << "=" << entry.skip.max_length_mode;
                    if (entry.skip.max_length) {
                        const auto [lower, upper] = anet::detail::GetProfiledValueActiveRange(*entry.skip.max_length);
                        offending << " active_range=[" << lower << "," << upper << "]";
                    }
                    offending << ";";
                }
                if (!offending.str().empty()) {
                    ANET_SYSTEM_ERROR("Invalid " << GetFullKey("learner.skip.target_window") << "=" << learner.skip.target_window
                        << "; offending Actors:" << offending.str() << " expected every declared Actor to use "
                        << DQNActorConfig::kMaxLengthSingle << " or " << DQNActorConfig::kMaxLengthVariable
                        << " with active_range=[1,1]; biased skip outputs must not hold actions.");
                }
                if (learner.skip.feature_gradient == LearnerConfig::kFeatureGradientStop) {
                    anet::log::info_once() << GetFullKey("learner.skip.feature_gradient") << "=" << LearnerConfig::kFeatureGradientStop
                        << " with " << GetFullKey("learner.skip.target_window") << "=" << learner.skip.target_window
                        << ": skip loss gradients do not reach shared features, and outputs are not used for acting. "
                        << GetFullKey("learner.skip.priority_lambda") << "=" << learner.skip.priority_lambda
                        << "; 0 means no effect on Q learning (control arm); positive values affect Q learning through replay priorities.";
                }
            }
            learner.quantile_mode = quantile_mode;
            learner.num_quantiles = qr.num_quantiles;

            // 現行の分布表現契約と全tau ruleを、利用前の設定境界で検証する。
            ValidateChoice(GetFullKey("quantile_mode"), quantile_mode,
                {ActionPolicyConfig::kQuantileModeNone, ActionPolicyConfig::kQuantileModeQr, ActionPolicyConfig::kQuantileModeIqn});
            ValidateRange(quantile_mode != ActionPolicyConfig::kQuantileModeQr || qr.num_quantiles > 1,
                GetFullKey("qr.num_quantiles"), qr.num_quantiles, "> 1 when " + GetFullKey("quantile_mode") + "=" + ActionPolicyConfig::kQuantileModeQr);

            ValidateTauRule(target_policy.tau_rule, target_policy.full_keys, "tau_rule.");
            ValidateTauRule(target_policy.full_distribution_query.tau_rule, target_policy.full_keys, "full_distribution_query.tau_rule.");
            ValidateTauRule(learner.iqn.current_taus, GetFullKeys(), "learner.iqn.current_taus.");
            ValidateTauRule(learner.iqn.target_taus, GetFullKeys(), "learner.iqn.target_taus.");

            // コピーと明示overlayを解決した最終target policyに対して競合を検証する。
            if (learner.munchausen.enabled && learner.use_double_dqn) {
                ANET_SYSTEM_ERROR(GetFullKey("learner.munchausen.enabled") << "=true conflicts with " << GetFullKey("learner.use_double_dqn")
                    << "=true; expected " << GetFullKey("learner.use_double_dqn") << "=false.");
            }
            // MunchausenとThompsonは共存付加
            if (learner.munchausen.enabled && target_policy.IsThompsonSampling()) {
                ANET_SYSTEM_ERROR(GetFullKey("learner.munchausen.enabled") << "=true conflicts with " << target_policy.full_keys.GetFullKey("policy_type") << "='"
                    << target_policy.policy_type << "'; expected " << kActionPolicyTypeStr_Greedy << ", "
                    << kActionPolicyTypeStr_EpsilonGreedy << ", or " << kActionPolicyTypeStr_UQE << ". " << GetFullKey("use_optimistic_target") << "="
                    << (use_optimistic_target ? "true (" + MakeDefaultConfigKey("actor.[train].policy") + " copy before target overrides)" : "false"));
            }
            ValidateActionPolicy(target_policy);
            for (auto& [key, entry] : actor) {
                ValidateActionPolicy(entry.policy);
                ValidateActionPolicy(entry.skip_policy);
            }

            ValidateRange(quantile_mode != ActionPolicyConfig::kQuantileModeIqn || (std::isfinite(learner.quantile_huber_kappa) && learner.quantile_huber_kappa > 0.0f),
                GetFullKey("learner.quantile_huber_kappa"), learner.quantile_huber_kappa,
                "finite value > 0 when " + GetFullKey("quantile_mode") + "=" + ActionPolicyConfig::kQuantileModeIqn);
        }

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
                ANET_SYSTEM_ERROR("Invalid " << GetFullKey(reference) << "='" << key
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
        std::unique_ptr<AgentStepDiagnosticsProvider> CreateStepDiagnosticsProvider(const AgentStepDiagnosticsRequest& request) const override;
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
        BatchUpdateResultList UpdateFromBatch(const StepCounts& step, const BatchExperience& expriences);
    private:
        std::shared_ptr<anet::nn::NetworkHeadFactory> SelectHeadFactory() const;
        void RecordNetworkStructure() const;
        void InitializeLearner(const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec,
            anet::seed_t replay_seed, anet::seed_t learner_seed);
        void LoadNetwork(const std::string& filename);
    private:
        class IntrinsicRewardCoordinator;
        // 内部型の破棄を cpp に閉じ、Agent の公開の特殊メンバは変えない。
        struct IntrinsicRewardCoordinatorDeleter {
            void operator()(IntrinsicRewardCoordinator* value) const;
        };
        DefaultDQNAgentConfig config_;
        std::map<std::string, std::shared_ptr<const anet::nn::NetworkForwardConfig>> forward_configs_; ///< Agent 所有の実行設定 Resource
        std::unique_ptr<anet::rl::dqn::RuntimeVars> vars_;
        std::unique_ptr<anet::rl::dqn::NetworkModel> model_;
        std::unique_ptr<anet::ForwarderChain> perception_;
        std::shared_ptr<anet::Forwarder> frame_intake_;
        std::unique_ptr<anet::rl::RewardScaler> reward_scaler_ = nullptr;
        std::unique_ptr<IntrinsicRewardCoordinator, IntrinsicRewardCoordinatorDeleter> intrinsic_;
        std::shared_ptr<anet::rl::dqn::ActionPolicy> target_policy_;    ///< 学習時ターゲット用ポリシー
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
            const std::string& net_config_prefix,
            const ConfigData& config_data);
    };

}// namespace anet::rl
