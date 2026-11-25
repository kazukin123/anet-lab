#include "anet/test/catch.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>
#include "anet/config.hpp"
#include "anet/observers.hpp"
#include "anet/test/nn.hpp"
#include "ppo_config.hpp"
#include "ppo_fixture_test.hpp"
#include "ppo_learner.hpp"

using namespace anet::rl::ppo;
using namespace anet::rl::ppo::test;

constexpr char kOutputDirectory[] = "out/test-tmp/rfd0202-shipped-config";
constexpr char kShippedConfigDirectory[] = "apps/runner/config";
constexpr char kUpdateResultTarget[] = "update_result";
constexpr size_t kMaxTagLength = 40;
constexpr char kEvalSampleActorKey[] = "eval_sample";
constexpr char kEvalPanelActorKey[] = "eval_panel";
constexpr char kOrthogonalInitMode[] = "orthogonal";

// 出荷の共通の設定（_main.txt）に、env の設定と追加の行を重ねて読む。
static anet::ConfigData LoadShippedConfig(const std::string& name, const std::string& env_file,
    const std::string& extra_lines)
{
    const auto repo = std::filesystem::current_path();
    const auto output = repo / kOutputDirectory;
    std::filesystem::create_directories(output);
    const auto input_path = output / (name + ".txt");
    {
        const auto config_root = (repo / kShippedConfigDirectory).generic_string();
        std::ofstream file(input_path, std::ios::binary);
        file << "$include <" << config_root << "/_main.txt>\n";
        if (!env_file.empty()) file << "$include <" << config_root << "/" << env_file << ">\n";
        file << extra_lines;
    }
    return anet::ConfigManager(input_path).GetConfigData();
}

static bool IsUpdateResultKey(const std::string& key)
{
    if (UpdateResult::IsDeviceStatKey(key)) return true;
    for (const auto* host_key : UpdateResult::GetHostStatKeys()) if (key == host_key) return true;
    return false;
}

TEST_CASE("Shipped agent.txt declares the RFD0202 defaults of PPOAgent", "[ppo][shipped_config][config][rfd0202]")
{
    // env の設定を含めない共通の設定で、agent.txt の PPOAgent.@baseline だけを確かめる。
    const auto data = LoadShippedConfig("main_only", "", "");

    // 既定は設定ファイルに書いてある（実装の既定に頼っていない）。
    for (const auto* key : { "PPOAgent.rollout.length", "PPOAgent.learner.epochs", "PPOAgent.learner.minibatch_count",
        "PPOAgent.learner.gamma", "PPOAgent.learner.gae_lambda", "PPOAgent.learner.learning_rate",
        "PPOAgent.learner.adam_eps", "PPOAgent.learner.weight_decay", "PPOAgent.learner.use_fused_optimizer",
        "PPOAgent.learner.ratio_clip_range", "PPOAgent.learner.use_value_clip", "PPOAgent.learner.value_clip_range",
        "PPOAgent.learner.entropy_coef", "PPOAgent.learner.value_coef", "PPOAgent.learner.use_grad_clip",
        "PPOAgent.learner.grad_clip_max_norm", "PPOAgent.learner.advantage_norm_mode",
        "PPOAgent.learner.truncation_mode", "PPOAgent.learner.kl_stop.mode",
        "PPOAgent.learner.kl_stop.epoch.threshold", "PPOAgent.head.policy.init.mode",
        "PPOAgent.head.policy.init.manual_gain", "PPOAgent.head.value.init.mode",
        "PPOAgent.head.value.init.manual_gain" }) {
        INFO("key=" << key);
        CHECK(data.Has(key));
    }

    // RFD0202 §7.4 の表の、出荷の既定。
    const PPOAgentConfig config(data);
    CHECK(config.rollout.length == 128);
    CHECK(config.learner.epochs == 4);
    CHECK(config.learner.minibatch_count == 4);
    CHECK(config.learner.gamma == Catch::Approx(0.99f));
    CHECK(config.learner.gae_lambda == Catch::Approx(0.95f));
    CHECK(config.learner.learning_rate.type == "constant");
    CHECK(config.learner.learning_rate.value == Catch::Approx(2.5e-4));
    CHECK(config.learner.adam_eps == Catch::Approx(1e-5));
    CHECK(config.learner.weight_decay == 0.0);
    CHECK(config.learner.use_fused_optimizer);
    CHECK(config.learner.ratio_clip_range.value == Catch::Approx(0.2f));
    CHECK(config.learner.use_value_clip);
    CHECK(config.learner.value_clip_range == Catch::Approx(0.2f));
    CHECK(config.learner.entropy_coef.value == Catch::Approx(0.01f));
    CHECK(config.learner.value_coef == Catch::Approx(0.5f));
    CHECK(config.learner.use_grad_clip);
    CHECK(config.learner.grad_clip_max_norm == Catch::Approx(0.5f));
    CHECK(config.learner.advantage_norm_mode == LearnerConfig::kAdvantageNormMinibatch);
    CHECK(config.learner.truncation_mode == LearnerConfig::kTruncationBootstrap);
    CHECK(config.learner.kl_stop.mode == LearnerConfig::kKlStopNone);
    CHECK(config.learner.kl_stop.epoch.threshold == Catch::Approx(0.015f));
    CHECK(config.head.policy.init.mode == kOrthogonalInitMode);
    CHECK(config.head.policy.init.manual_gain == Catch::Approx(0.01));
    CHECK(config.head.value.init.mode == kOrthogonalInitMode);
    CHECK(config.head.value.init.manual_gain == Catch::Approx(1.0));

    // Actor カタログ。学習は抽選で Network を共有し、評価は写しを持つ。EvalPanel は Greedy の評価と同じである。
    REQUIRE(config.actor.contains(kTrainActorKey));
    CHECK(config.actor.at(kTrainActorKey).action_mode == ActorConfig::kActionModeSample);
    CHECK_FALSE(config.actor.at(kTrainActorKey).clone_model);
    REQUIRE(config.actor.contains(kEvalActorKey));
    CHECK(config.actor.at(kEvalActorKey).action_mode == ActorConfig::kActionModeGreedy);
    CHECK(config.actor.at(kEvalActorKey).clone_model);
    REQUIRE(config.actor.contains(kEvalSampleActorKey));
    CHECK(config.actor.at(kEvalSampleActorKey).action_mode == ActorConfig::kActionModeSample);
    CHECK(config.actor.at(kEvalSampleActorKey).clone_model);
    REQUIRE(config.actor.contains(kEvalPanelActorKey));
    CHECK(config.actor.at(kEvalPanelActorKey).action_mode == ActorConfig::kActionModeGreedy);
    CHECK(config.actor.at(kEvalPanelActorKey).clone_model);
}

TEST_CASE("Shipped metrics.scalar.@ppo reads only keys the PPO update result knows, and covers all of them",
    "[ppo][shipped_config][config][rfd0202]")
{
    // プロファイルだけを選び、env の設定が足す行（行動ごとの鍵）は含めない。
    const auto data = LoadShippedConfig("metrics_ppo", "", "metrics.scalar.$ = metrics.scalar.@ppo\n");
    const anet::rl::ObserverFactory factory(data);

    std::set<std::string> update_result_keys;
    for (const auto& def : factory.GetScalarMetricDefs()) {
        INFO("tag=" << def.tag << " key=" << def.source_key);
        // tag は、Metrics Viewer の一覧で見切れない長さにする。
        CHECK(def.tag.size() <= kMaxTagLength);
        if (def.target != kUpdateResultTarget) continue;
        // 更新結果を読む行は、どれも更新結果が知っている鍵を、Train の LEARN で、exp step の横軸に書く。
        CHECK(IsUpdateResultKey(def.source_key));
        CHECK(def.event == "learn");
        CHECK(def.step_axis == "exp_step");
        CHECK(def.scope == anet::rl::RunnerScope::TRAIN);
        update_result_keys.insert(def.source_key);
    }

    // RFD0202 §7.5 の表の鍵は、このプロファイル 1 つでそろう。
    for (const auto* key : UpdateResult::GetMinibatchStatKeys()) {
        INFO("key=" << key);
        CHECK(update_result_keys.contains(key));
    }
    for (const auto* key : UpdateResult::GetRolloutStatKeys()) {
        INFO("key=" << key);
        CHECK(update_result_keys.contains(key));
    }
    for (const auto* key : UpdateResult::GetHostStatKeys()) {
        INFO("key=" << key);
        CHECK(update_result_keys.contains(key));
    }
}

TEST_CASE("Shipped CartPole_ppo.txt builds a PPOAgent that fills its metrics after one rollout",
    "[.][ppo_shipped_cartpole]")
{
    // Runner の Run を起動せずに、出荷の CartPole の設定で Agent を組めることを確かめる（env の設定の値は固定しない）。
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    anet::test::RegisterStandardNetworkModules();
    const auto data = LoadShippedConfig("cartpole_ppo", "CartPole_ppo.txt", "");
    CHECK(data.Get("agent.class_id") == kAgentClassId);
    const anet::rl::ObserverFactory factory(data);
    const PPOAgentConfig config(data);

    // CartPole の観測（4 要素の vector）と行動（2 つ）で Agent を作る。lane 数と rollout の長さは設定のまま使う。
    constexpr int64_t kCartPoleObsSize = 4;
    constexpr int kCartPoleActionCount = 2;
    const int lane_count = data.Get<int>("run.train.num_envs");
    auto rig = MakeRig(data, MakeEnvSpec(kCartPoleObsSize, kCartPoleActionCount), lane_count, device);
    for (const auto* actor_key : { kEvalActorKey, kEvalSampleActorKey, kEvalPanelActorKey }) {
        INFO("actor=" << actor_key);
        CHECK_NOTHROW(rig.agent->CreateActor(anet::rl::ActorRequest{ .batch_env_spec = rig.batch_env_spec,
            .env_spec = rig.env_spec, .device = device, .seed = 1, .actor_key = actor_key }));
    }

    // 出荷の metrics の行の購読を、そのまま Agent に渡す。
    std::vector<anet::rl::ScalarMetricSubscription> subscriptions;
    for (const auto& def : factory.GetScalarMetricDefs()) subscriptions.push_back(def.subscription);
    rig.agent->ConfigureScalarMetricSubscriptions(subscriptions);

    // rollout を 1 つ埋めて更新する。更新結果を読む行は、どれも有限の値を返す。
    anet::rl::BatchUpdateResultList results;
    for (int t = 0; t < config.rollout.length; ++t) {
        results = StepOnce(rig, MakeState(torch::rand({ lane_count, kCartPoleObsSize })), torch::rand({ lane_count }),
            MakeState(torch::rand({ lane_count, kCartPoleObsSize })));
    }
    REQUIRE(results.size() == static_cast<size_t>(config.learner.epochs * config.learner.minibatch_count));
    for (const auto& def : factory.GetScalarMetricDefs()) {
        if (def.target != kUpdateResultTarget) continue;
        INFO("tag=" << def.tag << " key=" << def.source_key);
        const auto value = results.back()->GetScalar(def.source_key);
        REQUIRE(value.has_value());
        // 方策と価値の Body を分ける設定では、両方の Head へ届く parameter が無いので、共有の組の勾配だけは値が成立しない。
        if (def.source_key == "grad_norm_shared") continue;
        CHECK(std::isfinite(*value));
    }
}
