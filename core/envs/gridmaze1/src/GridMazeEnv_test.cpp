#include "GridMazeEnv.hpp"

#include <cmath>
#include <exception>
#include <algorithm>
#include <array>
#include <vector>
#include <unordered_set>
#include <future>
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
    data.Set("GridMazeEnv.seed", hole ? 58 : 1);
    data.Set("GridMazeEnv.wall_density", 0.0);
    data.Set("GridMazeEnv.hole_density", hole ? 0.5 : 0.0);
    data.Set("GridMazeEnv.init_pos_x", 0);
    data.Set("GridMazeEnv.init_pos_y", 0);
    data.Set("GridMazeEnv.goal_pos_x", hole ? 2 : 1);
    data.Set("GridMazeEnv.goal_pos_y", 0);
    data.Set("GridMazeEnv.max_steps", 1);
    anet::rl::VectorizedDiscreteBatchEnv env(data,
        std::make_shared<anet::rl::env::GridMazeEnvFactory>(), "gridmaze-notification-reset", 1,
        torch::Device(torch::kCPU), 1);
    const auto reset = env.Reset();
    if (hole) {
        // 右は穴、下段は通路で、終端のテストにも解ける迷路を使う。
        const auto map = reset->GetAuxDataList(0).at(0).at("map");
        CHECK(map[0][1].item<int>() == 2);
        for (int x = 0; x < 3; ++x) CHECK(map[1][x].item<int>() == 0);
    }
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

namespace {

anet::rl::env::GridMazeEnvConfig FixedMazeConfig(anet::seed_t seed = 2)
{
    anet::rl::env::GridMazeEnvConfig config;
    config.seed = seed;
    config.init_pos_x = 0;
    config.init_pos_y = 0;
    config.goal_pos_x = 4;
    config.goal_pos_y = 4;
    return config;
}

std::vector<int64_t> FindGoalPath(const anet::rl::AuxData& aux)
{
    // auxだけを読み、深さ優先で経路を復元する。productionの探索を呼ばない。
    const auto map = aux.at("map");
    const int width = static_cast<int>(map.size(1));
    const int height = static_cast<int>(map.size(0));
    const int start = aux.at("pos_y").item<int>() * width + aux.at("pos_x").item<int>();
    std::vector<int> pending{start};
    std::vector<int> parent(width * height, -1);
    std::vector<int64_t> action(width * height, -1);
    parent[start] = start;
    constexpr std::array<int, 4> dx{0, -1, 0, 1};
    constexpr std::array<int, 4> dy{-1, 0, 1, 0};
    while (!pending.empty()) {
        const int cell = pending.back();
        pending.pop_back();
        const int x = cell % width;
        const int y = cell / width;
        if (map[y][x].item<int>() == 3) {
            std::vector<int64_t> path;
            for (int at = cell; at != start; at = parent[at]) path.push_back(action[at]);
            std::reverse(path.begin(), path.end());
            return path;
        }
        for (int direction = 0; direction < 4; ++direction) {
            const int nx = x + dx[direction];
            const int ny = y + dy[direction];
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
            const int next = ny * width + nx;
            const int value = map[ny][nx].item<int>();
            if (parent[next] != -1 || value == 1 || value == 2) continue;
            parent[next] = cell;
            action[next] = direction;
            pending.push_back(next);
        }
    }
    return {};
}

} // namespace

TEST_CASE("GridMaze replaces unsolvable seed with a reachable maze", "[gridmaze][prd100][prd100_seed2]")
{
    anet::test::LogCaptureGuard logs;
    const auto config = FixedMazeConfig();
    anet::rl::env::GridMazeEnv env(config, torch::kCPU, "gridmaze-seed2", 1);
    const auto aux = env.Reset()->GetAuxData();
    CHECK(aux.at("seed").item<int64_t>() != 2);
    const auto path = FindGoalPath(aux);
    REQUIRE_FALSE(path.empty());
    // 観測した盤面の経路を実際のStepで進め、穴でなくゴール終端へ到達する。
    for (size_t i = 0; i < path.size(); ++i) {
        const auto step = env.Step(path[i]);
        CHECK(step->next_state.done == (i + 1 == path.size()));
        if (i + 1 == path.size()) CHECK(step->GetScalar("is_success") == 1.0f);
    }
}

TEST_CASE("GridMaze fixed seeds preserve maps and reproduce from the accepted seed", "[gridmaze][prd100][prd100_repro]")
{
    anet::test::LogCaptureGuard logs;
    const auto seed = GENERATE(anet::seed_t{2}, anet::seed_t{6545537233705677626ULL}, ~anet::seed_t{0});
    auto config = FixedMazeConfig(seed);
    if (seed == ~anet::seed_t{0}) config.wall_density = config.hole_density = 0.0f;
    anet::rl::env::GridMazeEnv first(config, torch::kCPU, "gridmaze-first", 1);
    const auto original = first.Reset()->GetAuxData();
    const auto accepted = static_cast<anet::seed_t>(original.at("seed").item<int64_t>());
    for (int i = 0; i < 3; ++i) {
        const auto next = first.Reset()->GetAuxData();
        CHECK(torch::equal(original.at("map"), next.at("map")));
        CHECK(next.at("seed").item<int64_t>() == original.at("seed").item<int64_t>());
    }
    anet::rl::env::GridMazeEnv second(config, torch::kCPU, "gridmaze-second", 999);
    CHECK(torch::equal(original.at("map"), second.Reset()->GetAuxData().at("map")));
    // 画面と同じ符号なし十進表記を、実際の設定読込へ戻して復元する。
    anet::ConfigData seed_data;
    seed_data.Set("GridMazeEnv.seed", std::to_string(accepted));
    config.seed = anet::rl::env::GridMazeEnvConfig(seed_data).seed;
    if (seed == ~anet::seed_t{0}) CHECK(accepted == seed);
    anet::rl::env::GridMazeEnv restored(config, torch::kCPU, "gridmaze-restored", 123);
    const auto restored_aux = restored.Reset()->GetAuxData();
    CHECK(torch::equal(original.at("map"), restored_aux.at("map")));
    CHECK(static_cast<anet::seed_t>(restored_aux.at("seed").item<int64_t>()) == accepted);
    if (seed == 6545537233705677626ULL) {
        // 既定seedの25セルと、生成後の乱数位置も変更しない。
        const auto golden = torch::tensor({{0.f,0.f,0.f,1.f,0.f}, {1.f,0.f,0.f,1.f,0.f},
            {0.f,0.f,0.f,0.f,1.f}, {0.f,1.f,2.f,0.f,0.f}, {0.f,2.f,0.f,0.f,3.f}});
        CHECK(torch::equal(original.at("map"), golden));
        CHECK(accepted == seed);
        anet::RandomGenerator reference(seed);
        for (int i = 0; i < 25; ++i) reference.Uniform01();
        CHECK(first.GetRandomGenerator()->RandUint64() == reference.RandUint64());
    }
}

TEST_CASE("GridMaze rejects identical fixed coordinates at construction", "[gridmaze][prd100][prd100_coordinate]")
{
    anet::test::LogCaptureGuard logs;
    auto config = FixedMazeConfig();
    config.goal_pos_x = 0;
    config.goal_pos_y = 0;
    try {
        anet::rl::env::GridMazeEnv env(config, torch::kCPU, "gridmaze-invalid", 1);
        FAIL("Identical fixed coordinates must fail");
    } catch (const std::exception& error) {
        // 生成上限まで試さず、指定した座標キーを示す構築エラーにする。
        CHECK(std::string(error.what()).find("GridMazeEnv.init_pos") != std::string::npos);
        CHECK(std::string(error.what()).find("max_retries=") == std::string::npos);
    }
}

TEST_CASE("GridMaze identifies an invalid board dimension before generation", "[gridmaze][prd100][prd100_dimension]")
{
    anet::test::LogCaptureGuard logs;
    auto config = FixedMazeConfig();
    const int dimension = GENERATE(0, 1);
    const int invalid = GENERATE(-1, 0);
    (dimension == 0 ? config.width : config.height) = invalid;
    try {
        anet::rl::env::GridMazeEnv env(config, torch::kCPU, "gridmaze-invalid-width", 1);
        FAIL("Negative board width must fail");
    } catch (const std::exception& error) {
        CHECK(std::string(error.what()).find(std::string(dimension == 0 ? "GridMazeEnv.width=" : "GridMazeEnv.height=") + std::to_string(invalid)) != std::string::npos);
        CHECK(std::string(error.what()).find("positive") != std::string::npos);
    }
}

TEST_CASE("GridMaze validates coordinate ranges before coordinate relationships", "[gridmaze][prd100][prd100_range]")
{
    anet::test::LogCaptureGuard logs;
    auto config = FixedMazeConfig();
    const int coordinate = GENERATE(-1, 0, 1, 2, 3);
    const int invalid = GENERATE(-2, 5, 9);
    const std::array<const char*, 4> keys{"GridMazeEnv.init_pos_x", "GridMazeEnv.init_pos_y", "GridMazeEnv.goal_pos_x", "GridMazeEnv.goal_pos_y"};
    const std::array<int*, 4> fields{&config.init_pos_x, &config.init_pos_y, &config.goal_pos_x, &config.goal_pos_y};
    if (coordinate < 0) {
        config.init_pos_x = config.goal_pos_x = invalid;
        config.init_pos_y = config.goal_pos_y = 0;
    } else {
        *fields[coordinate] = invalid;
    }
    try {
        anet::rl::env::GridMazeEnv env(config, torch::kCPU, "gridmaze-range", 1);
        FAIL("Out-of-range coordinates must fail");
    } catch (const std::exception& error) {
        CHECK(std::string(error.what()).find(std::string(keys[std::max(coordinate, 0)]) + "=" + std::to_string(invalid)) != std::string::npos);
        CHECK(std::string(error.what()).find("expected -1 or [0, 5)") != std::string::npos);
    }
}

TEST_CASE("GridMaze rejects an empty random-start candidate set", "[gridmaze][prd100][prd100_empty]")
{
    anet::test::LogCaptureGuard logs;
    auto config = FixedMazeConfig();
    config.width = config.height = 1;
    config.init_pos_x = -1;
    config.init_pos_y = 0;
    config.goal_pos_x = config.goal_pos_y = 0;
    try {
        anet::rl::env::GridMazeEnv env(config, torch::kCPU, "gridmaze-empty-start", 1);
        FAIL("An empty start candidate set must fail");
    } catch (const std::exception& error) {
        CHECK(std::string(error.what()).find("no start candidates") != std::string::npos);
        CHECK(std::string(error.what()).find("GridMazeEnv.init_pos_x=-1") != std::string::npos);
        CHECK(std::string(error.what()).find("max_retries=") == std::string::npos);
    }
}

TEST_CASE("GridMaze automatic seeds and random coordinates always produce reachable maps", "[gridmaze][prd100][prd100_random]")
{
    anet::test::LogCaptureGuard logs;
    const int mode = GENERATE(0, 1, 2, 3, 4, 5, 6, 7, 8);
    auto config = FixedMazeConfig();
    config.randomize_seed = true;
    if (mode == 1 || mode == 3 || mode == 5) config.init_pos_x = -1;
    if (mode == 2 || mode == 3 || mode == 5) config.init_pos_y = -1;
    if (mode == 4 || mode == 5) config.goal_pos_x = config.goal_pos_y = -1;
    if (mode == 6) {
        config.width = 1; config.height = 2;
        config.goal_pos_x = 0; config.goal_pos_y = -1;
    }
    if (mode == 7 || mode == 8) {
        config.width = 2; config.height = 1;
        config.goal_pos_x = mode == 7 ? -1 : 1; config.goal_pos_y = 0;
    }
    if (mode == 8) {
        // 密度の和が1を超えても、隣接する開始とゴールなら到達できる。
        config.wall_density = 0.9f; config.hole_density = 0.2f;
    }
    anet::rl::env::GridMazeEnv env(config, torch::kCPU, "gridmaze-random", 1);
    std::unordered_set<anet::seed_t> seeds;
    for (int i = 0; i < 50; ++i) {
        const auto reset = env.Reset();
        const auto aux = reset->GetAuxData();
        CHECK_FALSE(FindGoalPath(aux).empty());
        CHECK(seeds.insert(static_cast<anet::seed_t>(aux.at("seed").item<int64_t>())).second);
        CHECK(reset->state.episode_start);
        CHECK_FALSE(reset->state.done);
        CHECK_FALSE(reset->state.truncated);
    }
    for (const auto& record : logs.Records()) CHECK(record.message.find("GridMaze maze regenerated:") == std::string::npos);
}

TEST_CASE("GridMaze reports bounded failure with generation context", "[gridmaze][prd100][prd100_limit]")
{
    anet::test::LogCaptureGuard logs;
    auto config = FixedMazeConfig();
    config.wall_density = GENERATE(0.9f, 1.0f);
    config.hole_density = 0.1f;
    try {
        anet::rl::env::GridMazeEnv env(config, torch::kCPU, "gridmaze-impossible", 1);
        FAIL("Disconnected dense maps must exhaust retries");
    } catch (const std::exception& error) {
        for (const auto key : {"seed=2", "width=5", "height=5", "wall_density=", "hole_density=",
             "init_pos=(0,0)", "goal_pos=(4,4)", "max_retries=1000"}) {
            CHECK(std::string(error.what()).find(key) != std::string::npos);
        }
    }
}

TEST_CASE("GridMaze retry notice is emitted once across concurrent instances", "[prd100_log]")
{
    // 他のseed2テストで共有記録を先に埋めないよう、このタグは別プロセスで実行する。
    anet::test::LogCaptureGuard logs;
    const auto config = FixedMazeConfig();
    std::vector<std::future<anet::seed_t>> pending;
    for (int i = 0; i < 8; ++i) {
        pending.push_back(std::async(std::launch::async, [config, i] {
            anet::rl::env::GridMazeEnv env(config, torch::kCPU, "gridmaze-log-" + std::to_string(i), 1);
            const auto aux = env.Reset()->GetAuxData();
            env.Reset();
            return static_cast<anet::seed_t>(aux.at("seed").item<int64_t>());
        }));
    }
    const auto accepted = pending.front().get();
    CHECK(accepted != 2);
    for (size_t i = 1; i < pending.size(); ++i) CHECK(pending[i].get() == accepted);
    auto no_retry = FixedMazeConfig(6545537233705677626ULL);
    anet::rl::env::GridMazeEnv unchanged(no_retry, torch::kCPU, "gridmaze-no-retry", 1);
    no_retry.randomize_seed = true;
    anet::rl::env::GridMazeEnv automatic(no_retry, torch::kCPU, "gridmaze-auto-notice", 1);
    for (int i = 0; i < 5; ++i) automatic.Reset();
    logs.Flush();
    std::vector<std::string> notices;
    for (const auto& record : logs.Records()) {
        if (record.message.find("GridMaze maze regenerated:") != std::string::npos) {
            CHECK(record.level == wxLOG_Message);
            notices.push_back(record.message);
        }
    }
    REQUIRE(notices.size() == 1);
    CHECK(notices.front().find("gridmaze-log-") != std::string::npos);
    CHECK(notices.front().find("seed=2,") != std::string::npos);
    CHECK(notices.front().find("retries=") != std::string::npos);
    CHECK(notices.front().find("actual_seed=" + std::to_string(accepted)) != std::string::npos);
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
