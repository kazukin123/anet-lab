#include "anet/test/catch.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include "anet/serialize.hpp"
#include "anet/agent.hpp"
#include "anet/observers.hpp"
#include "anet/trainer.hpp"
#include "anet/algo/ppo.hpp"
#include "ppo_fixture_test.hpp"

using namespace anet::rl::ppo;
using namespace anet::rl::ppo::test;

constexpr char kCheckpointDirectory[] = "out/test-tmp/rfd0202";
constexpr char kOtherCheckpointFormat[] = "OtherAgent-1";
constexpr char kPolicyLagKey[] = "policy_lag";
constexpr char kMissingCheckpointName[] = "no_such_checkpoint.bin";

TEST_CASE("PPO agent collects a rollout and returns one update result per optimizer step", "[ppo][rfd0202]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    // 設定から Agent を作り、行動選択から rollout の更新までを 1 回通す。
    auto data = MakeConfigData();
    data.Set("PPOAgent.rollout.length", 2);
    data.Set("PPOAgent.learner.epochs", 1);
    data.Set("PPOAgent.learner.minibatch_count", 1);
    auto rig = MakeRig(data, MakeEnvSpec(4, 2), /*lane_count=*/2, device);
    rig.agent->ConfigureScalarMetricSubscriptions(MakeAllSubscriptions(2));

    const auto state = MakeState(torch::rand({ 2, 4 }));
    const auto reward = torch::ones({ 2 });
    // rollout の長さに届くまでは、更新結果が空である。
    CHECK(StepOnce(rig, state, reward, state).empty());
    // 届いた intake で、optimizer の step 1 回ぶんの結果が返る。
    const auto results = StepOnce(rig, state, reward, state);
    REQUIRE(results.size() == 1);
    const auto loss = results.front()->GetScalar("loss_total");
    REQUIRE(loss.has_value());
    CHECK(std::isfinite(*loss));
}

TEST_CASE("PPO initialization keeps its first Agent factory registration", "[ppo][init][idempotent]")
{
    anet::rl::algo::InitPpo();
    auto& repository = anet::rl::AgentRepository::Instance();
    const auto factory = repository.GetAgentFactory(kAgentClassId);
    REQUIRE(factory);
    CHECK_NOTHROW(anet::rl::algo::InitPpo());
    CHECK(repository.GetAgentFactory(kAgentClassId) == factory);
}

// 行動を全部の lane で強制して、乱数に左右されない経験を 1 step ぶん作る。
static anet::rl::BatchUpdateResultList StepWithForcedActions(Rig& rig, const torch::Tensor& obs,
    const torch::Tensor& next_obs, const torch::Tensor& actions, const torch::Tensor& reward)
{
    return StepOnce(rig, MakeState(obs), reward, MakeState(next_obs), {}, MakeForcedActionOptions(actions));
}

static std::filesystem::path MakeCheckpointPath(const std::string& name)
{
    const auto directory = std::filesystem::current_path() / kCheckpointDirectory;
    std::filesystem::create_directories(directory);
    return directory / name;
}

TEST_CASE("PPO agent restores the network and the optimizer from a saved checkpoint", "[ppo][ppo_save][rfd0202]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");
    const bool use_fused_optimizer = GENERATE(false, true);
    INFO("device=" << device.str() << " use_fused_optimizer=" << use_fused_optimizer);

    // 1 step で rollout が埋まる設定。ミニバッチが 1 つなので、更新は並べ替えの順に左右されない。
    constexpr int kLaneCount = 4;
    constexpr int64_t kObsSize = 4;
    auto data = MakeConfigData();
    data.Set("PPOAgent.rollout.length", 1);
    data.Set("PPOAgent.learner.epochs", 1);
    data.Set("PPOAgent.learner.minibatch_count", 1);
    data.Set("PPOAgent.learner.learning_rate", 0.01);
    data.Set("PPOAgent.learner.use_fused_optimizer", use_fused_optimizer);
    const auto env_spec = MakeEnvSpec(kObsSize, 2);
    const auto obs = torch::rand({ kLaneCount, kObsSize });
    const auto next_obs = torch::rand({ kLaneCount, kObsSize });
    const auto actions = torch::tensor({ 0, 1, 1, 0 }, torch::kInt64);
    const auto reward = torch::tensor({ 1.0f, 0.0f, -1.0f, 0.5f });

    // 1 回更新した Agent を保存する。optimizer は、1 回ぶんの moment を持っている。
    auto original = MakeRig(data, env_spec, kLaneCount, device);
    const auto initial_values = ReadValues(*original.actor, obs);
    const auto initial_log_probs = ReadLogProbs(*original.actor, obs, actions);
    REQUIRE(StepWithForcedActions(original, obs, next_obs, actions, reward).size() == 1);
    const auto path = MakeCheckpointPath(std::string("ppo_agent_") + (device.is_cuda() ? "cuda" : "cpu")
        + (use_fused_optimizer ? "_fused" : "_plain") + ".bin");
    {
        std::ofstream stream(path, std::ios::binary);
        anet::OutputArchive archive(stream, path.string());
        CHECK(original.agent->Save(archive) > 0);
    }
    // 方策と価値の Body を分ける設定なので、方策の側（log 確率）と価値の側（value）の両方が、更新で動いたことを確かめる。
    const auto saved_values = ReadValues(*original.actor, obs);
    const auto saved_log_probs = ReadLogProbs(*original.actor, obs, actions);
    REQUIRE_FALSE(torch::allclose(saved_values, initial_values, 0.0, 1e-5));
    REQUIRE_FALSE(torch::allclose(saved_log_probs, initial_log_probs, 0.0, 1e-6));

    // 同じ設定の新しい Agent が、構築時にファイルを読む。初期の重みではなく、保存した重みで、方策の log 確率と value を返す。
    data.Set("PPOAgent.auto_load_file", path.generic_string());
    auto restored = MakeRig(data, env_spec, kLaneCount, device);
    CHECK(torch::equal(ReadLogProbs(*restored.actor, obs, actions), saved_log_probs));
    CHECK(torch::equal(ReadValues(*restored.actor, obs), saved_values));

    // 同じ経験でもう 1 回ずつ更新すると、同じ重みになる。optimizer の moment も読めていることを示す
    // （読めていなければ、Adam の最初の 1 回の更新になり、保存した側の 2 回目の更新と違う重みになる）。
    REQUIRE(StepWithForcedActions(original, obs, next_obs, actions, reward).size() == 1);
    REQUIRE(StepWithForcedActions(restored, obs, next_obs, actions, reward).size() == 1);
    const auto continued_values = ReadValues(*original.actor, obs);
    REQUIRE_FALSE(torch::allclose(continued_values, saved_values, 0.0, 1e-5));
    CHECK(torch::allclose(ReadValues(*restored.actor, obs), continued_values, 1e-5, 1e-6));
    CHECK(torch::allclose(ReadLogProbs(*restored.actor, obs, actions), ReadLogProbs(*original.actor, obs, actions),
        1e-5, 1e-6));
}

TEST_CASE("PPO agent rejects a missing checkpoint and a checkpoint of another format", "[ppo][ppo_save][rfd0202]")
{
    const auto env_spec = MakeEnvSpec(4, 2);
    const auto device = torch::Device(torch::kCPU);

    SECTION("missing file")
    {
        auto data = MakeConfigData();
        data.Set("PPOAgent.auto_load_file",
            (std::filesystem::path(kCheckpointDirectory) / kMissingCheckpointName).generic_string());
        CHECK_THROWS_WITH(MakeRig(data, env_spec, 2, device),
            Catch::Matchers::ContainsSubstring("PPOAgent.auto_load_file")
            && Catch::Matchers::ContainsSubstring(kMissingCheckpointName));
    }

    SECTION("another format")
    {
        // 形式の名前が違うファイル（ほかの Agent の保存など）は、中身を読まずに止める。
        const auto path = MakeCheckpointPath("other_format.bin");
        {
            std::ofstream stream(path, std::ios::binary);
            anet::OutputArchive archive(stream, path.string());
            archive.Write(anet::ArchiveHeader(kOtherCheckpointFormat));
        }
        auto data = MakeConfigData();
        data.Set("PPOAgent.auto_load_file", path.generic_string());
        CHECK_THROWS_WITH(MakeRig(data, env_spec, 2, device),
            Catch::Matchers::ContainsSubstring(kOtherCheckpointFormat)
            && Catch::Matchers::ContainsSubstring(PPOAgent::kCheckpointFormat));
    }
}

TEST_CASE("PPO agent trains through the Serial and Pipeline Train Runners", "[ppo][ppo_runner][rfd0202]")
{
    // 製品の Train Runner に、行動選択・Env の step・経験の受け渡し・LearnEvent の通知を任せる。
    const bool use_pipeline = GENERATE(false, true);
    INFO("use_pipeline=" << use_pipeline);
    constexpr int kLaneCount = 4;
    constexpr int kRolloutLength = 8;
    constexpr int kEpochs = 2;
    constexpr int kMinibatchCount = 2;
    constexpr int kRolloutCount = 4;
    constexpr anet::seed_t kSeed = 123;
    auto data = MakeConfigData();
    data.Set("PPOAgent.rollout.length", kRolloutLength);
    data.Set("PPOAgent.learner.epochs", kEpochs);
    data.Set("PPOAgent.learner.minibatch_count", kMinibatchCount);

    const ScopedNoopMetricsLogger metrics_logger;
    const auto env = std::make_shared<CorridorBatchEnv>(kLaneCount);
    const auto device = torch::Device(torch::kCPU);
    const auto agent = PPOAgentFactory().CreateAgent(env->GetSpec(), env->GetBatchSpec(), device, data, nullptr, kSeed);
    agent->ConfigureScalarMetricSubscriptions(MakeAllSubscriptions(CorridorEnv::kActionCount));

    // LearnEvent ごとに、更新結果の件数と、方策の古さを控える。
    std::vector<size_t> result_counts;
    std::vector<float> policy_lags;
    const auto notifier = std::make_shared<anet::rl::Notifier>();
    notifier->Attach(std::make_shared<anet::rl::FunctionLearnObserver>([&](const anet::rl::LearnEvent& event) {
        result_counts.push_back(event.update_result_list.size());
        policy_lags.push_back(*event.update_result_list.front()->GetScalar(kPolicyLagKey));
    }));
    const anet::rl::ActorRequest request{ .batch_env_spec = env->GetBatchSpec(), .env_spec = env->GetSpec(),
        .device = device, .seed = kSeed, .actor_key = kTrainActorKey };
    std::shared_ptr<anet::rl::TrainRunner> runner;
    if (use_pipeline) runner = std::make_shared<anet::rl::PipelineTrainRunner>(env, agent, notifier, request);
    else runner = std::make_shared<anet::rl::SerialTrainRunner>(env, agent, notifier, request);

    // Pipeline は、経験の学習が 1 step 遅れ、その通知がさらに次の step の初めになるので、2 step 多く回す。
    const int step_count = kRolloutCount * kRolloutLength + (use_pipeline ? 2 : 0);
    const auto counts = runner->DoUpdateFrame(step_count);
    runner->Shutdown();

    // rollout が埋まるたびに LearnEvent が 1 回出て、learn step は、返した更新結果の件数だけ進む。
    REQUIRE(result_counts.size() == static_cast<size_t>(kRolloutCount));
    for (const auto result_count : result_counts) CHECK(result_count == static_cast<size_t>(kEpochs * kMinibatchCount));
    CHECK(counts.learn_step == kRolloutCount * kEpochs * kMinibatchCount);

    // 方策の古さ。Serial は、どの経験も今の方策で集まる。Pipeline は、rollout の更新と重なる 1 step だけ、
    // 更新の前の方策で行動が選ばれるので、2 つ目からの rollout で 1 step ぶん（1 / rollout の長さ）になる。
    CHECK(policy_lags.front() == 0.0f);
    for (size_t i = 1; i < policy_lags.size(); ++i) {
        INFO("rollout=" << i);
        CHECK(policy_lags[i] == Catch::Approx(use_pipeline ? 1.0f / kRolloutLength : 0.0f));
    }
}

// 一本道の課題で rollout を rollout_count 個ぶん学習し、rollout ごとの、終わった episode の return の平均を返す。
// episode が 1 つも終わらなかった rollout は NaN にする。
static std::vector<float> TrainOnCorridor(int rollout_count, anet::seed_t seed)
{
    constexpr int kLaneCount = 8;
    constexpr int kRolloutLength = 32;
    auto data = MakeConfigData();
    data.Set("PPOAgent.rollout.length", kRolloutLength);
    data.Set("PPOAgent.learner.epochs", 4);
    data.Set("PPOAgent.learner.minibatch_count", 4);
    data.Set("PPOAgent.learner.learning_rate", 3e-3);
    auto rig = MakeRig(data, MakeEnvSpec(CorridorEnv::kLength, CorridorEnv::kActionCount), kLaneCount,
        torch::Device(torch::kCPU), seed);

    CorridorEnv env(kLaneCount);
    auto state = env.Reset();
    std::vector<float> rollout_returns;
    anet::rl::StepCounts counts;
    for (int rollout = 0; rollout < rollout_count; ++rollout) {
        float return_sum = 0.0f;
        int episode_count = 0;
        for (int t = 0; t < kRolloutLength; ++t) {
            // Runner と同じ順: 行動選択 → Env の 1 step → 経験を Learner へ → 終わった lane を Reset した状態で続ける。
            const auto action_info = rig.actor->MakeAction(counts, state);
            auto step = env.Step(action_info->GetAction(torch::Device(torch::kCPU)));
            rig.learner->UpdateFromBatch(counts,
                anet::rl::BatchExperience(state, action_info, step.reward, step.next_state));
            for (const auto episode_return : step.finished_returns) {
                return_sum += episode_return;
                ++episode_count;
            }
            state = step.continue_state;
            counts.exp_step += kLaneCount;
        }
        rollout_returns.push_back(episode_count > 0 ? return_sum / static_cast<float>(episode_count)
            : std::numeric_limits<float>::quiet_NaN());
    }
    return rollout_returns;
}

// 並びの [first, last) の rollout の、return の平均（episode が終わらなかった rollout は除く）。
static float ComputeMeanReturn(const std::vector<float>& rollout_returns, size_t first, size_t last)
{
    float sum = 0.0f;
    int count = 0;
    for (size_t i = first; i < last; ++i) {
        if (std::isnan(rollout_returns[i])) continue;
        sum += rollout_returns[i];
        ++count;
    }
    REQUIRE(count > 0);
    return sum / static_cast<float>(count);
}

TEST_CASE("PPO agent improves the episode return on a small corridor task", "[ppo][ppo_learning][rfd0202]")
{
    // 40 個の rollout（約 1 万遷移）で、学習の終わりの return が、学習の初めより高いことを確かめる。
    constexpr size_t kRolloutCount = 40;
    constexpr size_t kWindow = 5;
    const auto returns = TrainOnCorridor(static_cast<int>(kRolloutCount), /*seed=*/123);
    const auto first = ComputeMeanReturn(returns, 0, kWindow);
    const auto last = ComputeMeanReturn(returns, kRolloutCount - kWindow, kRolloutCount);
    INFO("first=" << first << " last=" << last);
    CHECK(last > first);
}

TEST_CASE("PPO agent solves the small corridor task for every seed", "[.][ppo_learn]")
{
    // 200 個の rollout の学習のあと、最後の 20 個の rollout の return の平均が、最適な方策の return の 9 割以上になる。
    // 学習の Actor は抽選で行動を選ぶので、最適な方策の return（0.94）にはわずかに届かない。
    constexpr size_t kRolloutCount = 200;
    constexpr size_t kWindow = 20;
    constexpr float kRequiredRatio = 0.9f;
    const auto seed = GENERATE(anet::seed_t{ 1 }, anet::seed_t{ 2 }, anet::seed_t{ 3 });
    const auto returns = TrainOnCorridor(static_cast<int>(kRolloutCount), seed);
    const auto last = ComputeMeanReturn(returns, kRolloutCount - kWindow, kRolloutCount);
    WARN("seed=" << seed << " last=" << last << " optimal=" << CorridorEnv::kOptimalReturn);
    CHECK(last >= kRequiredRatio * CorridorEnv::kOptimalReturn);
}
