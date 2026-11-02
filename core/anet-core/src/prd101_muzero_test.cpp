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
