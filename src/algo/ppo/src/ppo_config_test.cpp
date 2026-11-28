#include "anet/test/catch.hpp"
#include <string>
#include "anet/config.hpp"
#include "ppo_config.hpp"
#include "ppo_fixture_test.hpp"

using namespace anet::rl::ppo;
using namespace anet::rl::ppo::test;

constexpr char kFalseValue[] = "false";

TEST_CASE("PPO config reads its defaults and overrides from the PPOAgent prefix", "[ppo][ppo_config][rfd0209]")
{
    // 何も書かない設定では、RFD0209 §7.4 の実装の既定になる（CleanRL の ppo.py と同じ値）。
    const PPOAgentConfig defaults{ anet::ConfigData() };
    CHECK(defaults.rollout.length == 128);
    CHECK(defaults.learner.epochs == 4);
    CHECK(defaults.learner.minibatch_count == 4);
    CHECK(defaults.learner.gamma == Catch::Approx(0.99f));
    CHECK(defaults.learner.gae_lambda == Catch::Approx(0.95f));
    CHECK(defaults.learner.learning_rate.value == Catch::Approx(2.5e-4));
    CHECK(defaults.learner.adam_eps == Catch::Approx(1e-5));
    CHECK(defaults.learner.weight_decay == 0.0);
    CHECK(defaults.learner.ratio_clip_range.value == Catch::Approx(0.2f));
    CHECK(defaults.learner.use_value_clip);
    CHECK(defaults.learner.value_clip_range == Catch::Approx(0.2f));
    CHECK(defaults.learner.entropy_coef.value == Catch::Approx(0.01f));
    CHECK(defaults.learner.value_coef == Catch::Approx(0.5f));
    CHECK(defaults.learner.use_grad_clip);
    CHECK(defaults.learner.grad_clip_max_norm == Catch::Approx(0.5f));
    CHECK(defaults.learner.advantage_norm_mode == LearnerConfig::kAdvantageNormMinibatch);
    CHECK(defaults.learner.truncation_mode == LearnerConfig::kTruncationBootstrap);
    CHECK(defaults.learner.kl_stop.mode == LearnerConfig::kKlStopNone);
    CHECK(defaults.learner.kl_stop.epoch.threshold == Catch::Approx(0.015f));
    CHECK(defaults.head.policy.init.mode == "orthogonal");
    CHECK(defaults.head.policy.init.manual_gain == Catch::Approx(0.01));
    CHECK(defaults.head.value.init.manual_gain == Catch::Approx(1.0));
    CHECK(defaults.actor.empty());

    // 設定に書いた値と、Actor カタログの項目を読む。
    auto data = MakeConfigData();
    data.Set("PPOAgent.rollout.length", 16);
    data.Set("PPOAgent.learner.kl_stop.mode", "epoch");
    data.Set("PPOAgent.head.policy.init.manual_gain", 0.5);
    const PPOAgentConfig config(data);
    CHECK(config.rollout.length == 16);
    CHECK(config.learner.kl_stop.mode == LearnerConfig::kKlStopEpoch);
    CHECK(config.head.policy.init.manual_gain == Catch::Approx(0.5));
    REQUIRE(config.actor.size() == 2);
    CHECK(config.actor.at(kTrainActorKey).action_mode == ActorConfig::kActionModeSample);
    CHECK_FALSE(config.actor.at(kTrainActorKey).clone_model);
    CHECK(config.actor.at(kEvalActorKey).action_mode == ActorConfig::kActionModeGreedy);
    CHECK(config.actor.at(kEvalActorKey).clone_model);
}

TEST_CASE("PPO config rejects out-of-range values with the full key and the given value",
    "[ppo][ppo_config][rfd0209]")
{
    // 設定キー、指定する値、メッセージに入るはずの「完全キー=値」。
    struct Case {
        const char* key;
        const char* value;
        const char* expected;
    };
    const auto test_case = GENERATE(
        Case{ .key = "PPOAgent.rollout.length", .value = "0", .expected = "PPOAgent.rollout.length=0" },
        Case{ .key = "PPOAgent.learner.epochs", .value = "0", .expected = "PPOAgent.learner.epochs=0" },
        Case{ .key = "PPOAgent.learner.minibatch_count", .value = "-1", .expected = "PPOAgent.learner.minibatch_count=-1" },
        Case{ .key = "PPOAgent.learner.gamma", .value = "1.5", .expected = "PPOAgent.learner.gamma=1.5" },
        Case{ .key = "PPOAgent.learner.gae_lambda", .value = "-0.5", .expected = "PPOAgent.learner.gae_lambda=-0.5" },
        Case{ .key = "PPOAgent.learner.adam_eps", .value = "0", .expected = "PPOAgent.learner.adam_eps=0" },
        Case{ .key = "PPOAgent.learner.weight_decay", .value = "-1", .expected = "PPOAgent.learner.weight_decay=-1" },
        Case{ .key = "PPOAgent.learner.value_coef", .value = "-1", .expected = "PPOAgent.learner.value_coef=-1" },
        Case{ .key = "PPOAgent.learner.advantage_norm_mode", .value = "batch", .expected = "PPOAgent.learner.advantage_norm_mode='batch'" },
        Case{ .key = "PPOAgent.learner.truncation_mode", .value = "ignore", .expected = "PPOAgent.learner.truncation_mode='ignore'" },
        Case{ .key = "PPOAgent.learner.kl_stop.mode", .value = "minibatch", .expected = "PPOAgent.learner.kl_stop.mode='minibatch'" },
        Case{ .key = "PPOAgent.actor.[train].action_mode", .value = "argmax", .expected = "PPOAgent.actor.[train].action_mode='argmax'" });
    INFO("key=" << test_case.key << " value=" << test_case.value);

    auto data = MakeConfigData();
    data.Set(test_case.key, std::string(test_case.value));
    CHECK_THROWS_WITH(PPOAgentConfig(data), Catch::Matchers::ContainsSubstring(test_case.expected));
}

TEST_CASE("PPO config validates the keys of an unused mode as well", "[ppo][ppo_config][rfd0209]")
{
    // clip や KL の判定を使わない設定でも、その幅や閾値の値域は検証する。
    // モードを切り替えるだけで動くようにし、切り替えたあとで別のキーの誤りが出ないようにするためである。
    struct Case {
        const char* switch_key;
        const char* switch_value;
        const char* key;
        const char* expected;
    };
    const auto test_case = GENERATE(
        Case{ .switch_key = "PPOAgent.learner.use_value_clip", .switch_value = kFalseValue, .key = "PPOAgent.learner.value_clip_range",
            .expected = "PPOAgent.learner.value_clip_range=0" },
        Case{ .switch_key = "PPOAgent.learner.use_grad_clip", .switch_value = kFalseValue, .key = "PPOAgent.learner.grad_clip_max_norm",
            .expected = "PPOAgent.learner.grad_clip_max_norm=0" },
        Case{ .switch_key = "PPOAgent.learner.kl_stop.mode", .switch_value = LearnerConfig::kKlStopNone, .key = "PPOAgent.learner.kl_stop.epoch.threshold",
            .expected = "PPOAgent.learner.kl_stop.epoch.threshold=0" });
    INFO("key=" << test_case.key);

    auto data = MakeConfigData();
    data.Set(test_case.switch_key, std::string(test_case.switch_value));
    data.Set(test_case.key, 0.0);
    CHECK_THROWS_WITH(PPOAgentConfig(data), Catch::Matchers::ContainsSubstring(test_case.expected));
}

TEST_CASE("PPO config rejects time-dependent values outside their bounds and unread Actor keys",
    "[ppo][ppo_config][rfd0209]")
{
    SECTION("negative learning rate")
    {
        auto data = MakeConfigData();
        data.Set("PPOAgent.learner.learning_rate", -0.001);
        CHECK_THROWS_WITH(PPOAgentConfig(data),
            Catch::Matchers::ContainsSubstring("PPOAgent.learner.learning_rate")
            && Catch::Matchers::ContainsSubstring("value=-0.001"));
    }

    SECTION("zero ratio clip range")
    {
        auto data = MakeConfigData();
        data.Set("PPOAgent.learner.ratio_clip_range", 0.0);
        CHECK_THROWS_WITH(PPOAgentConfig(data),
            Catch::Matchers::ContainsSubstring("PPOAgent.learner.ratio_clip_range")
            && Catch::Matchers::ContainsSubstring("value=0 "));
    }

    SECTION("negative entropy coefficient")
    {
        auto data = MakeConfigData();
        data.Set("PPOAgent.learner.entropy_coef", -0.01);
        CHECK_THROWS_WITH(PPOAgentConfig(data),
            Catch::Matchers::ContainsSubstring("PPOAgent.learner.entropy_coef")
            && Catch::Matchers::ContainsSubstring("value=-0.01"));
    }

    SECTION("a key the Actor catalog does not read")
    {
        auto data = MakeConfigData();
        data.Set("PPOAgent.actor.[train].epsilon", 0.1);
        CHECK_THROWS_WITH(PPOAgentConfig(data),
            Catch::Matchers::ContainsSubstring("PPOAgent.actor.[train].epsilon"));
    }
}

TEST_CASE("PPO config checks the rollout size against the lane count", "[ppo][ppo_config][rfd0209]")
{
    auto data = MakeConfigData();
    data.Set("PPOAgent.rollout.length", 3);
    data.Set("PPOAgent.learner.minibatch_count", 4);

    SECTION("transitions must be divisible by the minibatch count")
    {
        // 3 × 2 lane = 6 遷移は、4 つのミニバッチに等分できない。
        const PPOAgentConfig config(data);
        CHECK_THROWS_WITH(config.ValidateLaneCount(2),
            Catch::Matchers::ContainsSubstring("PPOAgent.learner.minibatch_count=4")
            && Catch::Matchers::ContainsSubstring("transitions=6")
            && Catch::Matchers::ContainsSubstring("PPOAgent.rollout.length=3"));
        // 3 × 4 lane = 12 遷移なら、3 件ずつに分けられる。
        CHECK_NOTHROW(config.ValidateLaneCount(4));
    }

    SECTION("minibatch normalization needs at least two rows")
    {
        // 4 遷移を 4 つに分けると 1 件ずつになり、標準偏差を作れない。
        data.Set("PPOAgent.rollout.length", 2);
        CHECK_THROWS_WITH(PPOAgentConfig(data).ValidateLaneCount(2),
            Catch::Matchers::ContainsSubstring("PPOAgent.learner.advantage_norm_mode='minibatch'")
            && Catch::Matchers::ContainsSubstring("minibatch size=1"));
        // 標準化をしなければ、1 件ずつでも更新できる。
        data.Set("PPOAgent.learner.advantage_norm_mode", LearnerConfig::kAdvantageNormNone);
        CHECK_NOTHROW(PPOAgentConfig(data).ValidateLaneCount(2));
    }

    SECTION("the Agent applies the check when it is built")
    {
        CHECK_THROWS_WITH(MakeRig(data, MakeEnvSpec(4, 2), /*lane_count=*/2, torch::Device(torch::kCPU)),
            Catch::Matchers::ContainsSubstring("PPOAgent.learner.minibatch_count=4"));
    }
}

TEST_CASE("PPO agent requires a discrete action space with at least two actions", "[ppo][ppo_config][rfd0209]")
{
    const auto data = MakeConfigData();
    const auto device = torch::Device(torch::kCPU);

    // 連続行動（2 次元）の Env。
    auto continuous = MakeEnvSpec(4, 2);
    continuous.action_spec.is_discrete = false;
    continuous.action_spec.value_labels.clear();
    continuous.action_spec.dims = {
        anet::rl::ActionDimInfo{ .min_value = -1.0f, .max_value = 1.0f, .name = "x" },
        anet::rl::ActionDimInfo{ .min_value = -1.0f, .max_value = 1.0f, .name = "y" } };
    CHECK_THROWS_WITH(MakeRig(data, continuous, 2, device), Catch::Matchers::ContainsSubstring("PPOAgent")
        && Catch::Matchers::ContainsSubstring("discrete"));
    // 行動が 1 つだけの Env は、選ぶものが無い。
    CHECK_THROWS_WITH(MakeRig(data, MakeEnvSpec(4, 1), 2, device), Catch::Matchers::ContainsSubstring("actions=1"));
}
