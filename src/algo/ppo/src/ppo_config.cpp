#include "ppo_config.hpp"
#include <cmath>
#include <initializer_list>
#include "anet/diag.hpp"

using namespace anet::rl::ppo;

static void RequireChoice(const std::string& key, const std::string& value, std::initializer_list<const char*> choices)
{
    std::string expected;
    for (const auto* choice : choices) {
        if (value == choice) return;
        if (!expected.empty()) expected += ", ";
        expected += choice;
    }
    ANET_SYSTEM_ERROR("Invalid " << key << "='" << value << "'; expected one of [" << expected << "].");
}

template<typename T>
static void RequireRange(bool valid, const std::string& key, T value, const char* expected)
{
    if (!valid) ANET_SYSTEM_ERROR("Invalid " << key << "=" << value << "; expected " << expected << ".");
}

PPOAgentConfig::PPOAgentConfig(const anet::ConfigData& config_data, const std::string& config_prefix)
    : anet::Config(config_data, kAgentClassId, config_prefix)
{
    // Head の初期化の既定は、参照実装（CleanRL）に合わせる。
    // 方策の出力は小さく（gain 0.01）して、学習の初めの行動の確率をほぼ一様にする。価値の出力は gain 1。
    constexpr char kOrthogonalInit[] = "orthogonal";
    head.policy.init.mode = kOrthogonalInit;
    head.policy.init.manual_gain = 0.01;
    head.value.init.mode = kOrthogonalInit;
    head.value.init.manual_gain = 1.0;

    // rollout と Learner の設定を読む。
    ANET_READ_CONFIG(config_data, rollout.length);
    ANET_READ_CONFIG(config_data, learner.epochs);
    ANET_READ_CONFIG(config_data, learner.minibatch_count);
    ANET_READ_CONFIG(config_data, learner.gamma);
    ANET_READ_CONFIG(config_data, learner.gae_lambda);
    ANET_READ_CONFIG(config_data, learner.learning_rate);
    ANET_READ_CONFIG(config_data, learner.adam_eps);
    ANET_READ_CONFIG(config_data, learner.weight_decay);
    ANET_READ_CONFIG(config_data, learner.use_fused_optimizer);
    ANET_READ_CONFIG(config_data, learner.ratio_clip_range);
    ANET_READ_CONFIG(config_data, learner.use_value_clip);
    ANET_READ_CONFIG(config_data, learner.value_clip_range);
    ANET_READ_CONFIG(config_data, learner.entropy_coef);
    ANET_READ_CONFIG(config_data, learner.value_coef);
    ANET_READ_CONFIG(config_data, learner.use_grad_clip);
    ANET_READ_CONFIG(config_data, learner.grad_clip_max_norm);
    ANET_READ_CONFIG(config_data, learner.advantage_norm_mode);
    ANET_READ_CONFIG(config_data, learner.truncation_mode);
    ANET_READ_CONFIG(config_data, learner.kl_stop.mode);
    ANET_READ_CONFIG(config_data, learner.kl_stop.epoch.threshold);

    // Head の初期化、図、読み込むファイルを読む。初期化の設定は、読み口が値域も検証する。
    ANET_READ_CONFIG(config_data, head.policy.init);
    ANET_READ_CONFIG(config_data, head.value.init);
    ANET_READ_CONFIG(config_data, nn_viz.show_param_shapes);
    ANET_READ_CONFIG(config_data, nn_viz.show_param_count);
    ANET_READ_CONFIG(config_data, nn_viz.show_tensor_specs);
    ANET_READ_CONFIG(config_data, nn_viz.show_branch_config);
    ANET_READ_CONFIG(config_data, nn_viz.show_head_info);
    ANET_READ_CONFIG(config_data, nn_viz.layout);
    ANET_READ_CONFIG(config_data, nn_viz.cluster_branches);
    ANET_READ_CONFIG(config_data, nn_viz.float_precision);
    ANET_READ_CONFIG(config_data, auto_load_file);

    // Actor カタログを読む。項目ごとに、読まれない設定キーを構築時に止める。
    auto items = config_data.MakeSubConfigData(MakeDefaultConfigKey("actor"));
    if (!GetOverridePrefix().empty()) {
        for (const auto& [key, sub] : config_data.MakeSubConfigData(MakeOverrideConfigKey("actor"))) {
            items.try_emplace(key, sub);
        }
    }
    for (const auto& [key, sub] : items) {
        auto& entry = actor[key];
        const auto prefix = "actor.[" + key + "]";
        const auto data = MakeReadScope(config_data, prefix);
        ReadConfig(data, prefix + ".action_mode", entry.action_mode);
        ReadConfig(data, prefix + ".clone_model", entry.clone_model);
        data.RequireAllKeysRead();
        RequireChoice(GetFullKey(prefix + ".action_mode"), entry.action_mode,
            { ActorConfig::kActionModeSample, ActorConfig::kActionModeGreedy });
    }

    // 設定キーだけで決まる値域を検証する。使わない側のキー（clip を使わないときの幅など）も検証する。
    // モードや有効フラグを切り替えるだけで、関係のない設定を直させないためである。
    RequireRange(rollout.length >= 1, GetFullKey("rollout.length"), rollout.length, "an integer >= 1");
    RequireRange(learner.epochs >= 1, GetFullKey("learner.epochs"), learner.epochs, "an integer >= 1");
    RequireRange(learner.minibatch_count >= 1, GetFullKey("learner.minibatch_count"), learner.minibatch_count,
        "an integer >= 1");
    RequireRange(std::isfinite(learner.gamma) && learner.gamma >= 0.0f && learner.gamma <= 1.0f,
        GetFullKey("learner.gamma"), learner.gamma, "a value in [0, 1]");
    RequireRange(std::isfinite(learner.gae_lambda) && learner.gae_lambda >= 0.0f && learner.gae_lambda <= 1.0f,
        GetFullKey("learner.gae_lambda"), learner.gae_lambda, "a value in [0, 1]");
    RequireRange(std::isfinite(learner.adam_eps) && learner.adam_eps > 0.0, GetFullKey("learner.adam_eps"),
        learner.adam_eps, "a finite value > 0");
    RequireRange(std::isfinite(learner.weight_decay) && learner.weight_decay >= 0.0,
        GetFullKey("learner.weight_decay"), learner.weight_decay, "a finite value >= 0");
    RequireRange(std::isfinite(learner.value_clip_range) && learner.value_clip_range > 0.0f,
        GetFullKey("learner.value_clip_range"), learner.value_clip_range, "a finite value > 0");
    RequireRange(std::isfinite(learner.value_coef) && learner.value_coef >= 0.0f, GetFullKey("learner.value_coef"),
        learner.value_coef, "a finite value >= 0");
    RequireRange(std::isfinite(learner.grad_clip_max_norm) && learner.grad_clip_max_norm > 0.0f,
        GetFullKey("learner.grad_clip_max_norm"), learner.grad_clip_max_norm, "a finite value > 0");
    RequireRange(std::isfinite(learner.kl_stop.epoch.threshold) && learner.kl_stop.epoch.threshold > 0.0f,
        GetFullKey("learner.kl_stop.epoch.threshold"), learner.kl_stop.epoch.threshold, "a finite value > 0");
    RequireChoice(GetFullKey("learner.advantage_norm_mode"), learner.advantage_norm_mode,
        { LearnerConfig::kAdvantageNormNone, LearnerConfig::kAdvantageNormMinibatch });
    RequireChoice(GetFullKey("learner.truncation_mode"), learner.truncation_mode,
        { LearnerConfig::kTruncationBootstrap, LearnerConfig::kTruncationTerminal });
    RequireChoice(GetFullKey("learner.kl_stop.mode"), learner.kl_stop.mode,
        { LearnerConfig::kKlStopNone, LearnerConfig::kKlStopEpoch });
    anet::nn::ValidateNetworkGraphVizConfig(nn_viz, GetConfigPrefix() + ".nn_viz");
}

void PPOAgentConfig::ValidateLaneCount(int64_t lane_count) const
{
    // 1 epoch は、rollout の全部の遷移（長さ × lane 数）を、同じ大きさのミニバッチに分けて 1 回ずつ使う。
    // 端数を黙って捨てたり丸めたりしないよう、割り切れない組み合わせは止める。
    const int64_t transitions = static_cast<int64_t>(rollout.length) * lane_count;
    if (transitions % learner.minibatch_count != 0) {
        ANET_SYSTEM_ERROR("Invalid " << GetFullKey("learner.minibatch_count") << "=" << learner.minibatch_count
            << "; expected a divisor of rollout transitions=" << transitions << " (" << GetFullKey("rollout.length")
            << "=" << rollout.length << " x lanes=" << lane_count << ").");
    }
    // 件数が 1 のミニバッチは標準偏差を作れず、標準化した advantage が NaN になる。
    const int64_t minibatch_size = transitions / learner.minibatch_count;
    if (learner.advantage_norm_mode == LearnerConfig::kAdvantageNormMinibatch && minibatch_size < 2) {
        ANET_SYSTEM_ERROR("Invalid " << GetFullKey("learner.advantage_norm_mode") << "='" << learner.advantage_norm_mode
            << "' with minibatch size=" << minibatch_size << "; expected minibatch size >= 2 ("
            << GetFullKey("rollout.length") << "=" << rollout.length << " x lanes=" << lane_count << " / "
            << GetFullKey("learner.minibatch_count") << "=" << learner.minibatch_count << ") or '"
            << LearnerConfig::kAdvantageNormNone << "'.");
    }
}
