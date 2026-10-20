#include "GridMazeEnv.hpp"

#include <cmath>
#include <exception>
#include "anet/catch_test.hpp"
#include "anet/env.hpp"
#include "anet/test_util.hpp"

TEST_CASE("GridMaze Step statistics survive automatic reset and later transitions", "[gridmaze][prd097_gridmaze]")
{
    const int outcome = GENERATE(0, 1, 2);
    const bool goal = outcome == 1;
    const bool hole = outcome == 2;
    anet::ConfigData data;
    data.Set("GridMazeEnv.width", hole ? 3 : 2);
    data.Set("GridMazeEnv.height", 2);
    data.Set("GridMazeEnv.seed", 1);
    data.Set("GridMazeEnv.wall_density", 0.0);
    data.Set("GridMazeEnv.hole_density", hole ? 1.0 : 0.0);
    data.Set("GridMazeEnv.init_pos_x", 0);
    data.Set("GridMazeEnv.init_pos_y", 0);
    data.Set("GridMazeEnv.goal_pos_x", hole ? 2 : 1);
    data.Set("GridMazeEnv.goal_pos_y", 0);
    data.Set("GridMazeEnv.max_steps", 1);
    anet::rl::VectorizedDiscreteBatchEnv env(data,
        std::make_shared<anet::rl::env::GridMazeEnvFactory>(), "gridmaze-notification-reset", 1,
        torch::Device(torch::kCPU), 1);
    env.Reset();
    // 右のgoal/holeによるdoneと、左の壁衝突によるtruncatedを公開Actionで作る。
    const auto step = env.Step(std::make_shared<anet::rl::BatchActionInfo>(
        torch::tensor({(goal || hole) ? int64_t{3} : int64_t{1}}, torch::kInt64)));
    CHECK(step->next_state.done.item<bool>() == (goal || hole));
    CHECK(step->next_state.truncated.item<bool>() == (!goal && !hole));
    REQUIRE(step->GetScalar("reward_sum", 0).has_value());
    CHECK(step->GetScalar("reward_sum", 0) == step->reward.item<float>());
    CHECK(step->GetScalar("is_success", 0) == (goal ? 1.0f : 0.0f));
    CHECK(step->GetScalar("is_hole", 0) == (hole ? 1.0f : 0.0f));
    CHECK(step->GetScalar("wall_collisions", 0) == ((goal || hole) ? 0.0f : 1.0f));
    CHECK(step->continue_state.episode_start.item<bool>());
    // 次のエピソードの開始後に前回の完了値を再公開しない。
    for (const auto& key : {"reward_sum", "is_success", "is_hole", "wall_collisions"}) {
        CHECK_FALSE(env.GetScalar(key, 0).has_value());
    }
    CHECK_FALSE(env.GetScalar("unknown", 0).has_value());
    const auto before = step->GetScalar("reward_sum", 0);
    env.Step(std::make_shared<anet::rl::BatchActionInfo>(torch::tensor({int64_t{1}}, torch::kInt64)));
    CHECK(step->GetScalar("reward_sum", 0) == before);
}

int main(int argc, char* argv[])
{
    // 共通の非対話テスト設定を使い、失敗は終了コードへ届ける。
    anet::test::PreparedTestArgs test_args;
    try {
        test_args = anet::test::PrepareTestArgs(argc, argv);
    } catch (const std::exception& error) {
        return anet::test::ReportTestArgsError(error);
    }
    anet::test::SetupTestFailureDialog(test_args.failure_dialog_enabled);
    Catch::Session session;
    session.configData().showDurations = Catch::ShowDurations::Always;
    return session.run(test_args.Argc(), test_args.Argv());
}
