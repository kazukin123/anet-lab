#include "anet/catch_test.hpp"
#include "muzero_rb.hpp"

TEST_CASE("PRD101 MuZero builder receives its constant discount explicitly", "[prd101][muzero]")
{
    using namespace anet::rl::muzero_proto;
    MuZeroExperienceQueue queue;
    for (int step = 0; step < 4; ++step) {
        queue.Push(MuZeroStepRecord{ .obs = {}, .action = 0, .reward = float(step + 1),
            .done = false, .truncated = false, .target_policy = torch::ones({1}), .root_value = 8.0f });
    }
    MuZeroReplayBuilder builder(3);
    const auto records = builder.ProcessQueue(queue, 0.5f);
    REQUIRE(records.size() == 1);
    CHECK(records.front().target_value == Catch::Approx(1.0f + 0.5f * 2.0f + 0.25f * 3.0f + 0.125f * 8.0f));
}

TEST_CASE("MuZero flushes terminal queues with the current discounted targets", "[prd122][muzero][terminal_targets]")
{
    using namespace anet::rl::muzero_proto;
    // n-step経路より短いキューを終わらせ、残る各位置の打ち切りを観測する。
    const bool truncated = GENERATE(false, true);
    constexpr float gamma = 0.25f;
    MuZeroExperienceQueue queue;
    queue.Push(MuZeroStepRecord{ .obs = {}, .action = 0, .reward = 2.0f,
        .done = false, .truncated = false, .target_policy = torch::ones({2}), .root_value = 100.0f });
    queue.Push(MuZeroStepRecord{ .obs = {}, .action = 1, .reward = 4.0f,
        .done = !truncated, .truncated = truncated, .target_policy = torch::ones({2}), .root_value = 8.0f });
    MuZeroReplayBuilder builder(3);
    const auto actual = builder.ProcessQueue(queue, gamma);
    REQUIRE(actual.size() == 2);
    CHECK(queue.Size() == 0);
    // truncatedの補いは最後のstep自身のroot_value。次状態とのずれの疑いは既存課題に残す。
    /// @todo issue-20261110-d21e5f: truncatedの補い値の時点は別PRDで検討する。本テストは現行値を固定する。
    const float tail_bootstrap = truncated ? 8.0f : 0.0f;
    CHECK(actual[0].target_value == Catch::Approx(2.0f + gamma * 4.0f + gamma * gamma * tail_bootstrap));
    CHECK(actual[1].target_value == Catch::Approx(4.0f + gamma * tail_bootstrap));
    CHECK(actual[0].action == 0);
    CHECK(actual[1].action == 1);
    CHECK_FALSE(actual[0].done);
    CHECK(actual[1].done);
}

TEST_CASE("MuZero unroll masks terminal crossings and lane tails", "[prd122][muzero][unroll_mask]")
{
    using namespace anet::rl::muzero_proto;
    // 終端の後に別episodeを置く場合と、記録そのものが尽きる場合を区別する。
    const bool terminal = GENERATE(false, true);
    MuZeroReplayStorage storage(12, 1);
    const int record_count = terminal ? 3 : 2;
    for (int i = 0; i < record_count; ++i) {
        anet::TensorDict obs;
        obs.Set(anet::rl::ObsKeys::kVector, torch::tensor({float(i)}));
        storage.Push(0, MuZeroReplayRecord{ .obs = obs, .action = i % 2, .reward = float(2 << i),
            .done = terminal && i == 1, .target_policy = torch::ones({2}) / 2.0f,
            .target_value = i == 0 ? 2.0f + 0.25f * 4.0f : i == 1 ? 4.0f : 8.0f });
    }
    MuZeroUniformSampler sampler(3, 122);
    const auto samples = sampler.Sample(storage, 32);
    const auto starts = samples.obs.At(anet::rl::ObsKeys::kVector);
    bool seen[3] = {false, false, false};
    const float expected_values[3][4] = {{3.0f, 4.0f, 0.0f, 0.0f},
        {4.0f, 0.0f, 0.0f, 0.0f}, {8.0f, 0.0f, 0.0f, 0.0f}};
    const float expected_masks[3][4] = {{1.0f, 1.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}};
    for (int row = 0; row < 32; ++row) {
        const int start = starts[row].item<int>();
        REQUIRE(start >= 0);
        REQUIRE(start < record_count);
        seen[start] = true;
        for (int k = 0; k <= 3; ++k) {
            CHECK(samples.target_values[row][k].item<float>() == expected_values[start][k]);
            CHECK(samples.masks[row][k].item<float>() == expected_masks[start][k]);
        }
    }
    // seedを固定し、起点の全場合を実際に引いたことも確認する。paddingの乱数行動は比べない。
    for (int i = 0; i < record_count; ++i) CHECK(seen[i]);
}

TEST_CASE("MuZero Push keeps each lane queue and target independent", "[prd122][muzero][lane_queues]")
{
    using namespace anet::rl;
    using namespace anet::rl::muzero_proto;
    constexpr const char* kTargetPolicy = "target_policy";
    constexpr const char* kRootValue = "root_value";
    ReplayBufferConfig config;
    config.capacity = 16;
    config.n_step = 3;
    config.unroll_steps = 1;
    MuZeroReplayBuffer buffer(config, 2, 2, 122);
    constexpr float gamma = 0.25f;
    for (int step = 0; step < 2; ++step) {
        // laneを観測値で識別し、報酬の桁を変えて混在を検出する。
        anet::TensorDict obs;
        obs.Set(ObsKeys::kVector, torch::tensor({{float(step)}, {float(100 + step)}}));
        const auto flags = torch::TensorOptions().dtype(torch::kBool);
        BatchState state(obs, torch::zeros({2}, flags), torch::zeros({2}, flags),
            torch::full({2}, step == 0, flags));
        BatchState next(obs, torch::full({2}, step == 1, flags),
            torch::zeros({2}, flags), torch::zeros({2}, flags));
        anet::TensorDict info;
        info.Set(kTargetPolicy, torch::ones({2, 2}) / 2.0f);
        info.Set(kRootValue, torch::zeros({2}));
        auto action = std::make_shared<BatchActionInfo>(torch::zeros({2}, torch::kInt64), info, AuxData());
        const BatchExperience experience(state, action,
            torch::tensor({float(step + 1), float(10 * (step + 1))}), next);
        buffer.Push(experience, gamma);
    }
    REQUIRE(buffer.Size() == 4);
    const auto samples = buffer.Sample(64, torch::Device(torch::kCPU), gamma);
    const auto obs = samples.obs.At(ObsKeys::kVector);
    bool seen[2][2] = {{false, false}, {false, false}};
    for (int row = 0; row < 64; ++row) {
        const int marker = obs[row].item<int>();
        const int lane = marker >= 100 ? 1 : 0;
        const int step = marker - lane * 100;
        REQUIRE((step >= 0 && step < 2));
        seen[lane][step] = true;
        const float first_reward = lane == 0 ? 1.0f : 10.0f;
        const float last_reward = lane == 0 ? 2.0f : 20.0f;
        const float expected = step == 0 ? first_reward + gamma * last_reward : last_reward;
        CHECK(samples.target_values[row][0].item<float>() == Catch::Approx(expected));
        CHECK(samples.masks[row][0].item<float>() == 1.0f);
        CHECK(samples.masks[row][1].item<float>() == (step == 0 ? 1.0f : 0.0f));
        CHECK(samples.target_values[row][1].item<float>() == (step == 0 ? last_reward : 0.0f));
    }
    for (const auto& lane : seen) for (const bool present : lane) CHECK(present);
}
