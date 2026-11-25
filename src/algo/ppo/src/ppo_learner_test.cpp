#include "anet/test/catch.hpp"
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "ppo_actor.hpp"
#include "ppo_fixture_test.hpp"
#include "ppo_learner.hpp"

using namespace anet::rl::ppo;
using namespace anet::rl::ppo::test;

constexpr int64_t kObsSize = 4;
constexpr int kActionCount = 2;
constexpr char kStaleActorKey[] = "stale";

// 2 か所以上で読む、更新結果の鍵。
constexpr char kLossTotalKey[] = "loss_total";
constexpr char kEpochsRunKey[] = "epochs_run";
constexpr char kLearningRateKey[] = "learning_rate";
constexpr char kGradNormKey[] = "grad_norm";
constexpr char kPolicyLagKey[] = "policy_lag";
constexpr char kGradClippedKey[] = "grad_clipped";
constexpr char kValueClipFractionKey[] = "value_clip_fraction";
constexpr char kExplainedVarianceKey[] = "explained_variance";
constexpr char kGradNormSharedKey[] = "grad_norm_shared";
constexpr char kGradNormPolicyKey[] = "grad_norm_policy";
constexpr char kGradNormValueKey[] = "grad_norm_value";
constexpr char kActionProb0Key[] = "action_prob_0";

static anet::ConfigData MakeLearnerConfigData(int rollout_length, int epochs, int minibatch_count,
    bool shared_body = false)
{
    auto data = MakeConfigData(shared_body);
    data.Set("PPOAgent.rollout.length", rollout_length);
    data.Set("PPOAgent.learner.epochs", epochs);
    data.Set("PPOAgent.learner.minibatch_count", minibatch_count);
    return data;
}

// Network の写しを持ち、抽選で行動を選ぶ Actor を、カタログに足す（学習の Runner は、この Actor を写し直さない）。
static void AddStaleActor(anet::ConfigData& data)
{
    const auto prefix = std::string("PPOAgent.actor.[") + kStaleActorKey + "]";
    data.Set(prefix + ".action_mode", ActorConfig::kActionModeSample);
    data.Set(prefix + ".clone_model", true);
}

static std::shared_ptr<anet::rl::Actor> MakeStaleActor(const Rig& rig, torch::Device device)
{
    return rig.agent->CreateActor(anet::rl::ActorRequest{ .batch_env_spec = rig.batch_env_spec,
        .env_spec = rig.env_spec, .device = device, .seed = 5, .actor_key = kStaleActorKey });
}

static float ReadScalar(const anet::rl::BatchUpdateResultList& results, const std::string& key, size_t index = 0)
{
    const auto value = results.at(index)->GetScalar(key);
    REQUIRE(value.has_value());
    return *value;
}

// 乱数の観測と報酬で、rollout を 1 つ埋める。最後の intake の更新結果を返す。
static anet::rl::BatchUpdateResultList FillRollout(Rig& rig, int rollout_length, int lane_count,
    const anet::rl::StepCounts& counts = {})
{
    anet::rl::BatchUpdateResultList results;
    for (int t = 0; t < rollout_length; ++t) {
        // 届くまでの intake は、どれも空の結果を返す。
        CHECK(results.empty());
        results = StepOnce(rig, MakeState(torch::rand({ lane_count, kObsSize })), torch::rand({ lane_count }),
            MakeState(torch::rand({ lane_count, kObsSize })), counts);
    }
    return results;
}

TEST_CASE("PPO Learner returns one result per optimizer step when a rollout is full, and nothing before",
    "[ppo][ppo_learner][rfd0202]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    // rollout 3 × lane 4 = 12 遷移を、2 epoch、3 ミニバッチ（4 件ずつ）で使う。
    constexpr int kRolloutLength = 3;
    constexpr int kLaneCount = 4;
    constexpr int kEpochs = 2;
    constexpr int kMinibatchCount = 3;
    auto rig = MakeRig(MakeLearnerConfigData(kRolloutLength, kEpochs, kMinibatchCount),
        MakeEnvSpec(kObsSize, kActionCount), kLaneCount, device);
    rig.agent->ConfigureScalarMetricSubscriptions(MakeAllSubscriptions(kActionCount));

    // 2 つ目の rollout も、器を空にしてから同じに進む。
    for (int rollout = 0; rollout < 2; ++rollout) {
        INFO("rollout=" << rollout);
        const auto results = FillRollout(rig, kRolloutLength, kLaneCount);
        REQUIRE(results.size() == static_cast<size_t>(kEpochs * kMinibatchCount));
        for (size_t i = 0; i < results.size(); ++i) {
            CHECK(std::isfinite(ReadScalar(results, kLossTotalKey, i)));
            CHECK(ReadScalar(results, kEpochsRunKey, i) == kEpochs);
        }
        // Network を共有する Actor で集めた rollout は、最初のミニバッチの比が 1 で、方策の古さが 0 である。
        CHECK(ReadScalar(results, "first_ratio_deviation") <= 1e-5f);
        CHECK(ReadScalar(results, kPolicyLagKey) == 0.0f);
        // 最初のミニバッチは重みが変わる前なので、KL は 0 である。後のミニバッチでは、方策が動いている。
        CHECK(ReadScalar(results, "approx_kl", 0) == Catch::Approx(0.0f).margin(1e-6));
        CHECK(ReadScalar(results, "kl_last_epoch") >= 0.0f);
    }
}

TEST_CASE("PPO Learner rejects experiences that were not selected by a PPO Actor", "[ppo][ppo_learner][rfd0202]")
{
    constexpr int kLaneCount = 2;
    auto rig = MakeRig(MakeLearnerConfigData(2, 1, 1), MakeEnvSpec(kObsSize, kActionCount), kLaneCount,
        torch::Device(torch::kCPU));
    const auto state = MakeState(torch::rand({ kLaneCount, kObsSize }));
    const auto reward = torch::zeros({ kLaneCount });

    // 行動の情報が無い経験。
    CHECK_THROWS_WITH(rig.learner->UpdateFromBatch({}, anet::rl::BatchExperience(state, nullptr, reward, state)),
        Catch::Matchers::ContainsSubstring("action information"));
    // log 確率や value を載せていない行動（ほかの系統の Actor の行動など）。
    const auto plain_action = std::make_shared<anet::rl::BatchActionInfo>(torch::zeros({ kLaneCount }, torch::kInt64));
    CHECK_THROWS_WITH(rig.learner->UpdateFromBatch({}, anet::rl::BatchExperience(state, plain_action, reward, state)),
        Catch::Matchers::ContainsSubstring("PPO Actor"));
    // lane 数が Agent を作ったときと違う報酬。器に入れる前に、形を示して止まる。
    const auto action_info = rig.actor->MakeAction({}, state);
    CHECK_THROWS_WITH(
        rig.learner->UpdateFromBatch({}, anet::rl::BatchExperience(state, action_info, torch::zeros({ 3 }), state)),
        Catch::Matchers::ContainsSubstring("reward") && Catch::Matchers::ContainsSubstring("one value per lane"));
}

TEST_CASE("PPO Learner bootstraps truncated steps from the pre-reset observation unless configured as terminal",
    "[ppo][ppo_learner][rfd0202]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    // γ = λ = 1、報酬 0 にすると、value の目標は「その episode の先で bootstrap する value」そのものになる。
    //   続く lane       : rollout の最後の、次の観測の value
    //   終端した step    : 0
    //   打ち切りの step  : bootstrap なら Reset の前の観測の value、terminal なら 0
    struct Case {
        const char* name;
        const char* truncation_mode;
        bool done;       ///< lane 0 の最初の step の終端の印。
        bool truncated;  ///< lane 0 の最初の step の打ち切りの印。
        bool bootstraps; ///< lane 0 の最初の step の目標に、Reset の前の観測の value が入るか。
    };
    const auto test_case = GENERATE(
        Case{ .name = "bootstrap mode, truncated", .truncation_mode = LearnerConfig::kTruncationBootstrap,
            .done = false, .truncated = true, .bootstraps = true },
        Case{ .name = "terminal mode, truncated", .truncation_mode = LearnerConfig::kTruncationTerminal,
            .done = false, .truncated = true, .bootstraps = false },
        Case{ .name = "bootstrap mode, done and truncated", .truncation_mode = LearnerConfig::kTruncationBootstrap,
            .done = true, .truncated = true, .bootstraps = false },
        Case{ .name = "bootstrap mode, done", .truncation_mode = LearnerConfig::kTruncationBootstrap,
            .done = true, .truncated = false, .bootstraps = false });
    INFO(test_case.name);

    constexpr int kLaneCount = 2;
    auto data = MakeLearnerConfigData(2, 1, 1);
    data.Set("PPOAgent.learner.gamma", 1.0);
    data.Set("PPOAgent.learner.gae_lambda", 1.0);
    data.Set("PPOAgent.learner.advantage_norm_mode", LearnerConfig::kAdvantageNormNone);
    data.Set("PPOAgent.learner.truncation_mode", test_case.truncation_mode);
    auto rig = MakeRig(data, MakeEnvSpec(kObsSize, kActionCount), kLaneCount, device);
    rig.agent->ConfigureScalarMetricSubscriptions(MakeAllSubscriptions(kActionCount));

    // 更新の前の Network で、期待値に使う value を読んでおく。
    const auto first_next_obs = torch::rand({ kLaneCount, kObsSize });
    const auto last_next_obs = torch::rand({ kLaneCount, kObsSize });
    const auto truncation_value = ReadValues(*rig.actor, first_next_obs)[0].item<float>();
    const auto last_values = ReadValues(*rig.actor, last_next_obs);
    const auto last_0 = last_values[0].item<float>();
    const auto last_1 = last_values[1].item<float>();

    // lane 0 は最初の step で episode が終わり、lane 1 は rollout の終わりまで続く。
    const auto reward = torch::zeros({ kLaneCount });
    CHECK(StepOnce(rig, MakeState(torch::rand({ kLaneCount, kObsSize })), reward,
        MakeState(first_next_obs, { test_case.done, false }, { test_case.truncated, false })).empty());
    const auto results = StepOnce(rig, MakeState(torch::rand({ kLaneCount, kObsSize })), reward,
        MakeState(last_next_obs));
    REQUIRE(results.size() == 1);

    // value の目標 4 つ（時刻 2 × lane 2）の平均を、手計算と比べる。
    const float first_target = test_case.bootstraps ? truncation_value : 0.0f;
    const float expected_mean = (first_target + last_0 + last_1 + last_1) / 4.0f;
    CHECK(ReadScalar(results, "value_target_mean") == Catch::Approx(expected_mean).margin(1e-5));
    CHECK(ReadScalar(results, "bootstrap_value_mean") == Catch::Approx((last_0 + last_1) / 2.0f).margin(1e-5));

    // 終端と打ち切りの数え方。両方が真の step は、終端に数える。
    CHECK(ReadScalar(results, "terminated_ratio") == Catch::Approx(test_case.done ? 0.25f : 0.0f));
    CHECK(ReadScalar(results, "truncated_ratio") == Catch::Approx(test_case.done ? 0.0f : 0.25f));
    CHECK(ReadScalar(results, "episode_end_count") == 1.0f);
}

TEST_CASE("PPO Learner stops when a rollout holds actions from a policy two or more updates old",
    "[ppo][ppo_learner][rfd0202]")
{
    // Network の写しを持ち、写し直されない Actor を、学習の行動選択に混ぜる。
    constexpr int kRolloutLength = 2;
    constexpr int kLaneCount = 2;
    auto data = MakeLearnerConfigData(kRolloutLength, 1, 1);
    AddStaleActor(data);
    auto rig = MakeRig(data, MakeEnvSpec(kObsSize, kActionCount), kLaneCount, torch::Device(torch::kCPU));
    rig.agent->ConfigureScalarMetricSubscriptions(MakeAllSubscriptions(kActionCount));
    const auto shared_actor = rig.actor;
    const auto stale_actor = MakeStaleActor(rig, torch::Device(torch::kCPU));

    // rollout の最初の step だけを、写しの Actor で選ぶ。Pipeline の Runner の 1 step の遅れと同じ並びになる。
    const auto fill = [&]() {
        anet::rl::BatchUpdateResultList results;
        for (int t = 0; t < kRolloutLength; ++t) {
            rig.actor = t == 0 ? stale_actor : shared_actor;
            results = StepOnce(rig, MakeState(torch::rand({ kLaneCount, kObsSize })), torch::rand({ kLaneCount }),
                MakeState(torch::rand({ kLaneCount, kObsSize })));
        }
        return results;
    };

    // 1 つ目の rollout: 写しも最新なので、古さは 0。
    CHECK(ReadScalar(fill(), kPolicyLagKey) == 0.0f);
    // 2 つ目の rollout: 写しは 1 回ぶん古い。止まらず、古さの平均（2 step のうち 1 step が 1）が出る。
    CHECK(ReadScalar(fill(), kPolicyLagKey) == Catch::Approx(0.5f));
    // 3 つ目の rollout: 写しが 2 回ぶん古いので、学習に使わずに止める。
    CHECK_THROWS_WITH(fill(), Catch::Matchers::ContainsSubstring("2 updates old")
        && Catch::Matchers::ContainsSubstring("shares the Learner network"));
}

TEST_CASE("PPO Learner rejects Actor values that live on another device", "[ppo][ppo_learner][rfd0202]")
{
    if (!torch::cuda::is_available()) SKIP("CUDA is unavailable");

    // Agent は CPU に置き、写しを持つ Actor だけを CUDA に置く。その Actor が載せる値は、Learner の device に無い。
    constexpr int kLaneCount = 2;
    auto data = MakeLearnerConfigData(2, 1, 1);
    AddStaleActor(data);
    auto rig = MakeRig(data, MakeEnvSpec(kObsSize, kActionCount), kLaneCount, torch::Device(torch::kCPU));
    rig.actor = MakeStaleActor(rig, torch::Device(torch::kCUDA, 0));
    const auto state = MakeState(torch::rand({ kLaneCount, kObsSize }));
    CHECK_THROWS_WITH(StepOnce(rig, state, torch::zeros({ kLaneCount }), state),
        Catch::Matchers::ContainsSubstring("Actor values on device=cuda"));
}

TEST_CASE("PPO Learner reads its time-dependent values at the experience step of the full rollout",
    "[ppo][ppo_learner][rfd0202]")
{
    // 学習率を、exp step 100 で 0.01 から 0 へ線形に下げる。比の clip の幅と entropy の係数は定数にする。
    constexpr int kLaneCount = 2;
    auto data = MakeLearnerConfigData(1, 1, 1);
    data.Set("PPOAgent.learner.learning_rate.type", "linear");
    data.Set("PPOAgent.learner.learning_rate.start", 0.01);
    data.Set("PPOAgent.learner.learning_rate.end", 0.0);
    data.Set("PPOAgent.learner.learning_rate.steps", 100);
    data.Set("PPOAgent.learner.ratio_clip_range", 0.3);
    data.Set("PPOAgent.learner.entropy_coef", 0.02);
    auto rig = MakeRig(data, MakeEnvSpec(kObsSize, kActionCount), kLaneCount, torch::Device(torch::kCPU));

    // 時間の値は、統計の購読が無くても読める。
    const auto at_start = FillRollout(rig, 1, kLaneCount, anet::rl::StepCounts{ .exp_step = 0 });
    CHECK(ReadScalar(at_start, kLearningRateKey) == Catch::Approx(0.01f));
    CHECK(ReadScalar(at_start, "ratio_clip_range") == Catch::Approx(0.3f));
    CHECK(ReadScalar(at_start, "entropy_coef") == Catch::Approx(0.02f));
    const auto at_half = FillRollout(rig, 1, kLaneCount, anet::rl::StepCounts{ .exp_step = 50 });
    CHECK(ReadScalar(at_half, kLearningRateKey) == Catch::Approx(0.005f));
    const auto at_end = FillRollout(rig, 1, kLaneCount, anet::rl::StepCounts{ .exp_step = 400 });
    CHECK(ReadScalar(at_end, kLearningRateKey) == Catch::Approx(0.0f).margin(1e-9));
}

TEST_CASE("PPO Learner stops the remaining epochs when the approximate KL exceeds the threshold",
    "[ppo][ppo_learner][rfd0202]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    // 学習率を大きく、閾値を小さくして、最初の epoch の終わりで必ず閾値を超えるようにする。
    constexpr int kRolloutLength = 4;
    constexpr int kLaneCount = 2;
    constexpr int kEpochs = 4;
    constexpr int kMinibatchCount = 2;
    auto data = MakeLearnerConfigData(kRolloutLength, kEpochs, kMinibatchCount);
    data.Set("PPOAgent.learner.learning_rate", 0.05);
    data.Set("PPOAgent.learner.kl_stop.epoch.threshold", 1e-6);

    SECTION("mode=epoch stops after the first epoch")
    {
        data.Set("PPOAgent.learner.kl_stop.mode", LearnerConfig::kKlStopEpoch);
        auto rig = MakeRig(data, MakeEnvSpec(kObsSize, kActionCount), kLaneCount, device);
        // 統計の購読が無くても、判定に使う KL は計算する。返る件数は、実行したミニバッチの数である。
        const auto results = FillRollout(rig, kRolloutLength, kLaneCount);
        REQUIRE(results.size() == static_cast<size_t>(kMinibatchCount));
        CHECK(ReadScalar(results, kEpochsRunKey) == 1.0f);
    }

    SECTION("mode=none runs every epoch")
    {
        data.Set("PPOAgent.learner.kl_stop.mode", LearnerConfig::kKlStopNone);
        auto rig = MakeRig(data, MakeEnvSpec(kObsSize, kActionCount), kLaneCount, device);
        const auto results = FillRollout(rig, kRolloutLength, kLaneCount);
        REQUIRE(results.size() == static_cast<size_t>(kEpochs * kMinibatchCount));
        CHECK(ReadScalar(results, kEpochsRunKey) == static_cast<float>(kEpochs));
    }
}

TEST_CASE("PPO Learner statistics show whether gradient clipping and value clipping are in use",
    "[ppo][ppo_learner][rfd0202]")
{
    constexpr int kRolloutLength = 2;
    constexpr int kLaneCount = 2;
    const auto run = [&](const anet::ConfigData& data) {
        auto rig = MakeRig(data, MakeEnvSpec(kObsSize, kActionCount), kLaneCount, torch::Device(torch::kCPU));
        rig.agent->ConfigureScalarMetricSubscriptions(MakeAllSubscriptions(kActionCount));
        return FillRollout(rig, kRolloutLength, kLaneCount);
    };

    SECTION("gradient clipping")
    {
        // 上限が小さければ、どのミニバッチも clip され、大きければ clip されない。使わなければ、値が成立しない。
        auto data = MakeLearnerConfigData(kRolloutLength, 1, 1);
        data.Set("PPOAgent.learner.grad_clip_max_norm", 1e-6);
        CHECK(ReadScalar(run(data), kGradClippedKey) == 1.0f);
        data.Set("PPOAgent.learner.grad_clip_max_norm", 1e6);
        const auto loose = run(data);
        CHECK(ReadScalar(loose, kGradClippedKey) == 0.0f);
        CHECK(ReadScalar(loose, kGradNormKey) > 0.0f);
        data.Set("PPOAgent.learner.use_grad_clip", false);
        const auto off = run(data);
        CHECK(std::isnan(ReadScalar(off, kGradClippedKey)));
        CHECK(ReadScalar(off, kGradNormKey) > 0.0f);
    }

    SECTION("value clipping")
    {
        // 2 epoch めには value が動いているので、幅を小さくすると、clip が効いた割合が 0 より大きくなる。
        auto data = MakeLearnerConfigData(kRolloutLength, 2, 1);
        data.Set("PPOAgent.learner.learning_rate", 0.05);
        data.Set("PPOAgent.learner.value_clip_range", 1e-6);
        const auto on = run(data);
        CHECK(ReadScalar(on, kValueClipFractionKey, 0) == 0.0f);
        CHECK(ReadScalar(on, kValueClipFractionKey, 1) > 0.0f);
        data.Set("PPOAgent.learner.use_value_clip", false);
        CHECK(std::isnan(ReadScalar(run(data), kValueClipFractionKey, 1)));
    }
}

TEST_CASE("PPO update result answers every documented metric key and nothing else", "[ppo][ppo_learner][rfd0202]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    // 方策と価値が Body を共有し、勾配と value の clip を使う設定では、どの鍵も値を持つ。
    constexpr int kRolloutLength = 4;
    constexpr int kLaneCount = 4;
    auto data = MakeLearnerConfigData(kRolloutLength, 2, 2, /*shared_body=*/true);
    auto rig = MakeRig(data, MakeEnvSpec(kObsSize, kActionCount), kLaneCount, device);

    SECTION("with subscriptions")
    {
        rig.agent->ConfigureScalarMetricSubscriptions(MakeAllSubscriptions(kActionCount));
        const auto results = FillRollout(rig, kRolloutLength, kLaneCount);
        REQUIRE(results.size() == 4);
        for (const auto* key : UpdateResult::GetMinibatchStatKeys()) {
            INFO("key=" << key);
            CHECK(std::isfinite(ReadScalar(results, key, 3)));
        }
        for (const auto* key : UpdateResult::GetRolloutStatKeys()) {
            INFO("key=" << key);
            CHECK(std::isfinite(ReadScalar(results, key, 3)));
        }
        for (const auto* key : UpdateResult::GetHostStatKeys()) {
            INFO("key=" << key);
            CHECK(std::isfinite(ReadScalar(results, key, 3)));
        }

        // 行動ごとの鍵は、行動数の範囲で値を持ち、確率も頻度も足して 1 になる。
        CHECK(ReadScalar(results, kActionProb0Key) + ReadScalar(results, "action_prob_1")
            == Catch::Approx(1.0f).margin(1e-5));
        CHECK(ReadScalar(results, "action_freq_0") + ReadScalar(results, "action_freq_1")
            == Catch::Approx(1.0f).margin(1e-5));
        // 行動数を超える番号は、既知の鍵だが値が成立しない。
        CHECK(std::isnan(ReadScalar(results, "action_prob_2")));
        CHECK(std::isnan(ReadScalar(results, "action_freq_2")));
        // 番号でないものと、表に無い鍵は、値なしである。
        for (const auto* key : { "action_prob_", "action_prob_x", "action_prob_-1", "action_freq_1x", "unknown" }) {
            INFO("key=" << key);
            CHECK_FALSE(results.front()->GetScalar(key).has_value());
        }

        // rollout の単位の値は、同じ rollout のどの更新結果からも同じ値で読める。
        CHECK(ReadScalar(results, kExplainedVarianceKey, 0) == ReadScalar(results, kExplainedVarianceKey, 3));
        // entropy は ln(行動数) 以下で、その比は 1 以下である。
        CHECK(ReadScalar(results, "entropy") <= std::log(static_cast<float>(kActionCount)) + 1e-5f);
        CHECK(ReadScalar(results, "entropy_ratio") <= 1.0f + 1e-5f);
    }

    SECTION("without subscriptions")
    {
        // 購読が無ければ統計を計算しない。device の統計の鍵は値が成立せず、CPU で分かる値だけが読める。
        const auto results = FillRollout(rig, kRolloutLength, kLaneCount);
        REQUIRE(results.size() == 4);
        CHECK(std::isnan(ReadScalar(results, kLossTotalKey)));
        CHECK(std::isnan(ReadScalar(results, kExplainedVarianceKey)));
        CHECK(std::isnan(ReadScalar(results, kActionProb0Key)));
        CHECK(ReadScalar(results, kEpochsRunKey) == 2.0f);
        CHECK(std::isfinite(ReadScalar(results, kLearningRateKey)));
    }

    SECTION("subscriptions of other scopes and events do not enable statistics")
    {
        // 統計は、Train の LEARN の購読でだけ計算する。
        rig.agent->ConfigureScalarMetricSubscriptions({
            anet::rl::ScalarMetricSubscription{ .source_key = kLossTotalKey, .event = anet::rl::EventType::LEARN,
                .scope = anet::rl::RunnerScope::EVAL },
            anet::rl::ScalarMetricSubscription{ .source_key = kLossTotalKey, .event = anet::rl::EventType::TRAIN } });
        CHECK(std::isnan(ReadScalar(FillRollout(rig, kRolloutLength, kLaneCount), kLossTotalKey)));
    }

    SECTION("gradient norms per parameter group are computed only when subscribed")
    {
        rig.agent->ConfigureScalarMetricSubscriptions({ anet::rl::ScalarMetricSubscription{
            .source_key = kLossTotalKey, .event = anet::rl::EventType::LEARN } });
        const auto without_groups = FillRollout(rig, kRolloutLength, kLaneCount);
        CHECK(std::isfinite(ReadScalar(without_groups, kGradNormKey)));
        CHECK(std::isnan(ReadScalar(without_groups, kGradNormSharedKey)));

        rig.agent->ConfigureScalarMetricSubscriptions({ anet::rl::ScalarMetricSubscription{
            .source_key = kGradNormPolicyKey, .event = anet::rl::EventType::LEARN } });
        const auto with_groups = FillRollout(rig, kRolloutLength, kLaneCount);
        // 全体のノルムの二乗は、3 つの組のノルムの二乗の和である。
        const auto shared = ReadScalar(with_groups, kGradNormSharedKey);
        const auto policy = ReadScalar(with_groups, kGradNormPolicyKey);
        const auto value = ReadScalar(with_groups, kGradNormValueKey);
        const auto total = ReadScalar(with_groups, kGradNormKey);
        CHECK(shared * shared + policy * policy + value * value == Catch::Approx(total * total).epsilon(1e-3));
    }
}

TEST_CASE("PPO update result reports no shared gradient norm when the bodies are separate",
    "[ppo][ppo_learner][rfd0202]")
{
    constexpr int kRolloutLength = 2;
    constexpr int kLaneCount = 2;
    auto rig = MakeRig(MakeLearnerConfigData(kRolloutLength, 1, 1, /*shared_body=*/false),
        MakeEnvSpec(kObsSize, kActionCount), kLaneCount, torch::Device(torch::kCPU));
    rig.agent->ConfigureScalarMetricSubscriptions(MakeAllSubscriptions(kActionCount));
    const auto results = FillRollout(rig, kRolloutLength, kLaneCount);
    // 両方の Head へ届く parameter が無いので、共有の組の値は成立しない。方策と価値の組は値を持つ。
    CHECK(std::isnan(ReadScalar(results, kGradNormSharedKey)));
    CHECK(ReadScalar(results, kGradNormPolicyKey) > 0.0f);
    CHECK(ReadScalar(results, kGradNormValueKey) > 0.0f);
}
