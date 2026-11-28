#include "anet/test/catch.hpp"
#include <cmath>
#include <memory>
#include <string>
#include "ppo_actor.hpp"
#include "ppo_fixture_test.hpp"

using namespace anet::rl::ppo;
using namespace anet::rl::ppo::test;

constexpr int64_t kObsSize = 4;
constexpr int kActionCount = 2;
constexpr char kUndefinedActorKey[] = "missing";
constexpr char kVectorInputKey[] = "input.vector";

static std::shared_ptr<anet::rl::Actor> MakeActor(const Rig& rig, const std::string& actor_key, torch::Device device)
{
    return rig.agent->CreateActor(anet::rl::ActorRequest{ .batch_env_spec = rig.batch_env_spec,
        .env_spec = rig.env_spec, .device = device, .seed = 321, .actor_key = actor_key });
}

// 方策の Head の出力を大きくして、行動の確率に差を付ける（既定の gain 0.01 では、ほぼ一様になる）。
static anet::ConfigData MakeDistinctPolicyConfigData()
{
    auto data = MakeConfigData();
    data.Set("PPOAgent.head.policy.init.manual_gain", 1.0);
    return data;
}

TEST_CASE("PPO Actor reports the log probability of the executed action, the value and the network input",
    "[ppo][ppo_actor][rfd0209]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    // 全部の lane に同じ観測を与え、行動だけを lane ごとに強制する（偶数の lane は 0、奇数の lane は 1）。
    constexpr int kLaneCount = 4;
    auto rig = MakeRig(MakeDistinctPolicyConfigData(), MakeEnvSpec(kObsSize, kActionCount), kLaneCount, device);
    const auto obs = torch::rand({ 1, kObsSize }).expand({ kLaneCount, kObsSize }).clone();
    const auto forced = torch::tensor({ 0, 1, 0, 1 }, torch::kInt64);
    const auto action_info = rig.actor->MakeAction({}, MakeState(obs), MakeForcedActionOptions(forced));

    // 実行される行動は、強制した行動である。
    CHECK(torch::equal(action_info->GetAction(torch::Device(torch::kCPU)), forced));

    // log 確率は、実行される行動のものである。同じ観測なので、2 つの行動の確率を足すと 1 になる。
    const auto& info = action_info->GetInfo();
    const auto log_probs = info.At(Actor::kLogProbKey);
    REQUIRE(log_probs.sizes() == torch::IntArrayRef({ kLaneCount }));
    CHECK(log_probs.device().type() == device.type());
    const auto probs = log_probs.cpu().exp();
    CHECK(probs[0].item<float>() + probs[1].item<float>() == Catch::Approx(1.0f).margin(1e-5));
    CHECK(probs[0].item<float>() == Catch::Approx(probs[2].item<float>()).margin(1e-6));
    CHECK(probs[0].item<float>() != Catch::Approx(probs[1].item<float>()).margin(1e-3));

    // value は lane ごとに 1 つで、同じ観測には同じ値になる。
    const auto values = info.At(Actor::kValueKey);
    REQUIRE(values.sizes() == torch::IntArrayRef({ kLaneCount }));
    CHECK(values.device().type() == device.type());
    CHECK(values.cpu()[0].item<float>() == Catch::Approx(values.cpu()[3].item<float>()).margin(1e-6));

    // Network へ渡した入力（stack 軸つき）を、Learner が使えるように載せる。device の上のままである。
    const auto input = info.At(kVectorInputKey);
    CHECK(input.sizes() == torch::IntArrayRef({ kLaneCount, 1, kObsSize }));
    CHECK(input.device().type() == device.type());
    CHECK(torch::equal(input.cpu().squeeze(1), obs));

    // 方策の世代は CPU のスカラーで、更新の前は 0 である。
    const auto version = info.At(Actor::kPolicyVersionKey);
    CHECK(version.device().is_cpu());
    CHECK(version.item<int64_t>() == 0);
}

TEST_CASE("PPO Actor samples with the policy probabilities and the greedy Actor selects the most probable action",
    "[ppo][ppo_actor][rfd0209]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    // 同じ観測を 4000 lane に並べ、1 回の行動選択で、行動の頻度を測る。
    constexpr int kLaneCount = 4000;
    auto rig = MakeRig(MakeDistinctPolicyConfigData(), MakeEnvSpec(kObsSize, kActionCount), kLaneCount, device);
    const auto obs = torch::rand({ 1, kObsSize }).expand({ kLaneCount, kObsSize }).clone();
    const auto state = MakeState(obs);

    // 行動 1 の確率を、行動を強制した 1 回で読む。
    const auto prob_1 = ReadLogProbs(*rig.actor, obs, torch::ones({ kLaneCount }, torch::kInt64))[0]
        .exp().item<float>();

    SECTION("sample")
    {
        // 抽選の頻度は、方策の確率に従う（4000 回の標準偏差は 0.008 以下）。
        const auto action_info = rig.actor->MakeAction({}, state);
        const auto actions = action_info->GetAction(torch::Device(torch::kCPU));
        CHECK(actions.to(torch::kFloat32).mean().item<float>() == Catch::Approx(prob_1).margin(0.04));
        // 載せる log 確率は、lane ごとに、その lane で選ばれた行動のものである。
        const auto expected = torch::where(actions == 1, torch::full({ kLaneCount }, std::log(prob_1)),
            torch::full({ kLaneCount }, std::log(1.0f - prob_1)));
        CHECK(torch::allclose(action_info->GetInfo().At(Actor::kLogProbKey).cpu(), expected, 1e-4, 1e-5));
    }

    SECTION("greedy")
    {
        // Greedy は、どの lane でも、確率が最大の行動を選ぶ。
        const auto eval_actor = MakeActor(rig, kEvalActorKey, device);
        const auto actions = eval_actor->MakeAction({}, state)->GetAction(torch::Device(torch::kCPU));
        const int64_t best_action = prob_1 > 0.5f ? 1 : 0;
        CHECK(torch::equal(actions, torch::full({ kLaneCount }, best_action, torch::kInt64)));
    }
}

TEST_CASE("PPO Actor with a cloned network follows the Learner only after Sync", "[ppo][ppo_actor][rfd0209]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    // 1 step で rollout が埋まり、1 回だけ更新する設定。学習率を大きくして、重みの変化を見やすくする。
    constexpr int kLaneCount = 2;
    auto data = MakeConfigData();
    data.Set("PPOAgent.rollout.length", 1);
    data.Set("PPOAgent.learner.epochs", 1);
    data.Set("PPOAgent.learner.minibatch_count", 1);
    data.Set("PPOAgent.learner.learning_rate", 0.05);
    auto rig = MakeRig(data, MakeEnvSpec(kObsSize, kActionCount), kLaneCount, device);
    const auto eval_actor = MakeActor(rig, kEvalActorKey, device);
    const auto read_version = [](const anet::rl::Actor& actor, const anet::rl::BatchState& state) {
        return actor.MakeAction({}, state)->GetInfo().At(Actor::kPolicyVersionKey).item<int64_t>();
    };

    // 更新の前は、写しも、Learner の Network と同じ value を返す。
    const auto obs = torch::rand({ kLaneCount, kObsSize });
    const auto state = MakeState(obs);
    const auto before = ReadValues(*rig.actor, obs);
    CHECK(torch::equal(ReadValues(*eval_actor, obs), before));

    // 1 回更新すると、共有の Actor は新しい重みを使い、写しは古い重みのまま残る。
    REQUIRE(StepOnce(rig, state, torch::ones({ kLaneCount }), state).size() == 1);
    const auto after = ReadValues(*rig.actor, obs);
    CHECK_FALSE(torch::allclose(after, before, 0.0, 1e-4));
    CHECK(torch::equal(ReadValues(*eval_actor, obs), before));
    CHECK(read_version(*rig.actor, state) == 1);
    CHECK(read_version(*eval_actor, state) == 0);

    // Sync で、写しが Learner の今の重みと、方策の世代に追いつく。
    eval_actor->Sync();
    CHECK(torch::equal(ReadValues(*eval_actor, obs), after));
    CHECK(read_version(*eval_actor, state) == 1);
}

TEST_CASE("PPO agent rejects an undefined Actor key", "[ppo][ppo_actor][rfd0209]")
{
    auto rig = MakeRig(MakeConfigData(), MakeEnvSpec(kObsSize, kActionCount), 2, torch::Device(torch::kCPU));
    // 設定のカタログに無いキーは、定義済みのキーを添えて止まる。
    CHECK_THROWS_WITH(MakeActor(rig, kUndefinedActorKey, torch::Device(torch::kCPU)),
        Catch::Matchers::ContainsSubstring(kUndefinedActorKey) && Catch::Matchers::ContainsSubstring(kTrainActorKey));
}
