#include "anet/catch_test.hpp"

#include "GridMazeEnv.hpp"

#include <cmath>
#include <exception>
#include <algorithm>
#include <array>
#include <vector>
#include <filesystem>
#include <fstream>
#include <future>
#include <latch>
#include <wx/cmdline.h>
#include "anet/init.hpp"
#include "anet/env.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/trainer.hpp"
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
    const auto seed = GENERATE(anet::seed_t{0}, anet::seed_t{2}, anet::seed_t{6545537233705677626ULL}, ~anet::seed_t{0});
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
    config.seed_mode = "episode";
    config.seed.reset();
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
    for (int i = 0; i < 50; ++i) {
        const auto reset = env.Reset();
        const auto aux = reset->GetAuxData();
        CHECK_FALSE(FindGoalPath(aux).empty());
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

TEST_CASE("GridMaze retry notice is emitted once across concurrent instances", "[.][prd100_log]")
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
    no_retry.seed_mode = "episode";
    no_retry.seed.reset();
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

TEST_CASE("GridMaze default process seed is logged once and reproduces the maze", "[.][prd102_process]")
{
    // 初回の構築から公開auxと共通通知までを独立プロセスで通す。
    anet::test::LogCaptureGuard logs;
    anet::ConfigData data;
    data.Set("GridMazeEnv.width", 3);
    data.Set("GridMazeEnv.height", 3);
    data.Set("GridMazeEnv.wall_density", 0.0);
    data.Set("GridMazeEnv.hole_density", 0.0);
    data.Set("GridMazeEnv.init_pos_x", 0);
    data.Set("GridMazeEnv.init_pos_y", 0);
    data.Set("GridMazeEnv.goal_pos_x", 2);
    data.Set("GridMazeEnv.goal_pos_y", 2);
    const anet::rl::env::GridMazeEnvConfig config(data);
    CHECK_FALSE(config.GetConfigData().Has("seed"));
    anet::rl::env::GridMazeEnv env(config, torch::kCPU, "process-default", 1);
    const auto initial = env.Reset()->GetAuxData();
    for (int i = 0; i < 3; ++i) {
        const auto next = env.Reset()->GetAuxData();
        CHECK(torch::equal(next.at("map"), initial.at("map")));
        CHECK(next.at("seed").item<int64_t>() == initial.at("seed").item<int64_t>());
    }
    logs.Flush();
    std::vector<std::string> notices;
    for (const auto& record : logs.Records()) {
        if (record.message.find("GridMaze process seed initialized:") != std::string::npos) {
            CHECK(record.level == wxLOG_Message);
            notices.push_back(record.message);
        }
    }
    REQUIRE(notices.size() == 1);
    const auto input_seed = std::stoull(notices.front().substr(notices.front().find("seed=") + 5));
    CHECK(static_cast<anet::seed_t>(initial.at("seed").item<int64_t>()) == input_seed);
    data.Set("GridMazeEnv.seed", std::to_string(input_seed));
    anet::rl::env::GridMazeEnv restored(anet::rl::env::GridMazeEnvConfig(data), torch::kCPU, "process-restored", 8);
    const auto restored_aux = restored.Reset()->GetAuxData();
    CHECK(torch::equal(restored_aux.at("map"), initial.at("map")));
    CHECK(restored_aux.at("seed").item<int64_t>() == initial.at("seed").item<int64_t>());
}

TEST_CASE("GridMaze shares the process seed across concurrent instances and recreation", "[.][prd102_concurrent]")
{
    // 同時に初回構築し、通知と入力seedの共有を独立プロセスで確かめる。
    anet::test::LogCaptureGuard logs;
    auto config = FixedMazeConfig();
    config.seed_mode = "process";
    config.seed.reset();
    config.wall_density = config.hole_density = 0.0f;
    std::latch start(8);
    std::vector<std::future<anet::rl::AuxData>> pending;
    for (int i = 0; i < 8; ++i) {
        pending.push_back(std::async(std::launch::async, [config, i, &start] {
            start.count_down();
            start.wait();
            anet::rl::env::GridMazeEnv env(config, torch::kCPU, "process-concurrent-" + std::to_string(i), i);
            const auto initial = env.Reset()->GetAuxData();
            env.Reset();
            return initial;
        }));
    }
    const auto initial = pending.front().get();
    for (size_t i = 1; i < pending.size(); ++i) {
        const auto next = pending[i].get();
        CHECK(next.at("seed").item<int64_t>() == initial.at("seed").item<int64_t>());
        CHECK(torch::equal(next.at("map"), initial.at("map")));
    }
    // 全個体が破棄されても、設定seedを変えて作る個体は同じ入力seedを使う。
    config.seed.reset();
    anet::rl::env::GridMazeEnv recreated(config, torch::kCPU, "process-recreated", 999);
    CHECK(recreated.Reset()->GetAuxData().at("seed").item<int64_t>() == initial.at("seed").item<int64_t>());
    logs.Flush();
    int notices = 0;
    for (const auto& record : logs.Records()) {
        if (record.message.find("GridMaze process seed initialized:") != std::string::npos) ++notices;
    }
    CHECK(notices == 1);
}

TEST_CASE("GridMaze episode mode resamples without a seed and reproduces every accepted maze", "[gridmaze][prd102_episode]")
{
    // 自動採番の重複を禁止せず、公開値から各盤面を復元できることを確かめる。
    anet::test::LogCaptureGuard logs;
    anet::ConfigData data;
    data.Set("GridMazeEnv.seed_mode", "episode");
    const anet::rl::env::GridMazeEnvConfig config(data);
    REQUIRE_FALSE(config.seed.has_value());
    anet::rl::env::GridMazeEnv env(config, torch::kCPU, "episode-automatic", 1);
    for (int i = 0; i < 50; ++i) {
        const auto aux = env.Reset()->GetAuxData();
        CHECK_FALSE(FindGoalPath(aux).empty());
        auto restored_config = config;
        restored_config.seed = static_cast<anet::seed_t>(aux.at("seed").item<int64_t>());
        anet::rl::env::GridMazeEnv restored(restored_config, torch::kCPU, "episode-seed-only", 99);
        const auto copy = restored.Reset()->GetAuxData();
        CHECK(torch::equal(copy.at("map"), aux.at("map")));
        CHECK(copy.at("pos_x").item<int>() == aux.at("pos_x").item<int>());
        CHECK(copy.at("pos_y").item<int>() == aux.at("pos_y").item<int>());
        CHECK(copy.at("seed").item<int64_t>() == aux.at("seed").item<int64_t>());
    }
    logs.Flush();
    for (const auto& record : logs.Records()) {
        CHECK(record.message.find("GridMaze process seed initialized:") == std::string::npos);
        CHECK(record.message.find("GridMaze maze regenerated:") == std::string::npos);
    }
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

TEST_CASE("GridMaze validates modes even with an explicit seed and rejects invalid seeds", "[.][prd102_settings]")
{
    anet::test::LogCaptureGuard logs;
    // 設定読込と、読込後にDTOを変えた直接構築の双方を検証する。
    for (const auto* mode : {"", "other", "PROCESS"}) {
        anet::ConfigData data;
        data.Set("GridMazeEnv.seed", "0");
        data.Set("GridMazeEnv.seed_mode", mode);
        CHECK_THROWS_WITH(anet::rl::env::GridMazeEnvConfig(data),
            Catch::Matchers::ContainsSubstring("GridMazeEnv.seed_mode='" + std::string(mode) + "': expected episode, process"));
    }
    auto invalid = FixedMazeConfig();
    invalid.seed_mode = "other";
    CHECK_THROWS_WITH(anet::rl::env::GridMazeEnv(invalid, torch::kCPU, "invalid-mode", 1),
        Catch::Matchers::ContainsSubstring("GridMazeEnv.seed_mode='other': expected episode, process"));
    // seedが存在すれば、どちらのモードでも0と最大値を通常の固定値として使う。
    for (const auto* mode : {"process", "episode"}) {
        for (const auto* seed : {"0", "18446744073709551615"}) {
            anet::ConfigData data;
            data.Set("GridMazeEnv.seed_mode", mode);
            data.Set("GridMazeEnv.seed", seed);
            data.Set("GridMazeEnv.wall_density", 0.0);
            data.Set("GridMazeEnv.hole_density", 0.0);
            const anet::rl::env::GridMazeEnvConfig config(data);
            anet::rl::env::GridMazeEnv env(config, torch::kCPU, "seed-priority", 1);
            for (int i = 0; i < 3; ++i) {
                const auto aux = env.Reset()->GetAuxData();
                CHECK(static_cast<anet::seed_t>(aux.at("seed").item<int64_t>()) == std::stoull(seed));
            }
        }
        for (const auto* seed : {"", "nan", "-1", "18446744073709551616", "1.5"}) {
            anet::ConfigData data;
            data.Set("GridMazeEnv.seed_mode", mode);
            data.Set("GridMazeEnv.seed", seed);
            CHECK_THROWS_WITH(anet::rl::env::GridMazeEnvConfig(data),
                Catch::Matchers::ContainsSubstring("GridMazeEnv.seed"));
        }
    }
    logs.Flush();
    for (const auto& record : logs.Records()) {
        CHECK(record.level != wxLOG_Warning);
        CHECK(record.message.find("GridMaze process seed initialized:") == std::string::npos);
    }
}

TEST_CASE("GridMaze initializes the process seed lazily and shares only the input seed", "[.][prd102_late]")
{
    anet::test::LogCaptureGuard logs;
    auto first = FixedMazeConfig();
    { anet::rl::env::GridMazeEnv fixed(first, torch::kCPU, "late-fixed", 1); }
    auto episode = first;
    episode.seed.reset();
    episode.seed_mode = "episode";
    { anet::rl::env::GridMazeEnv automatic(episode, torch::kCPU, "late-episode", 1); }
    logs.Flush();
    for (const auto& record : logs.Records()) CHECK(record.message.find("GridMaze process seed initialized:") == std::string::npos);
    // 異なる寸法・密度も同じ入力seedを使い、盤面と乱数器は別々に所有する。
    first.seed.reset();
    auto second = first;
    second.width = 7;
    second.height = 6;
    second.goal_pos_x = 6;
    second.goal_pos_y = 5;
    second.wall_density = 0.1f;
    anet::rl::env::GridMazeEnv a(first, torch::kCPU, "late-a", 1);
    anet::rl::env::GridMazeEnv b(second, torch::kCPU, "late-b", 999);
    logs.Flush();
    std::vector<std::string> notices;
    for (const auto& record : logs.Records()) {
        if (record.message.find("GridMaze process seed initialized:") != std::string::npos) notices.push_back(record.message);
    }
    REQUIRE(notices.size() == 1);
    const auto input = std::stoull(notices.front().substr(notices.front().find("seed=") + 5));
    const auto original_b = b.Reset()->GetAuxData();
    a.Step(3);
    for (int i = 0; i < 10; ++i) a.Reset();
    const auto next_b = b.Reset()->GetAuxData();
    CHECK(torch::equal(original_b.at("map"), next_b.at("map")));
    for (auto config : {first, second}) {
        auto& actual = config.width == first.width ? a : b;
        const auto actual_aux = actual.Reset()->GetAuxData();
        REQUIRE_FALSE(FindGoalPath(actual_aux).empty());
        config.seed = input;
        anet::rl::env::GridMazeEnv restored(config, torch::kCPU, "late-restored", 2);
        const auto aux = restored.Reset()->GetAuxData();
        CHECK(torch::equal(actual_aux.at("map"), aux.at("map")));
        CHECK(actual_aux.at("seed").item<int64_t>() == aux.at("seed").item<int64_t>());
    }
}

TEST_CASE("GridMaze configuration comparison resolves every manifest input", "[.][prd102_config_snapshot]")
{
    // Propertiesの情報通知もcaptureし、無人検証でダイアログを開かない。
    anet::test::LogCaptureGuard logs;
    const auto repo = std::filesystem::current_path();
    const auto output = repo / ".scratch" / "prd102";
    std::ifstream source(repo / "core/anet-core/testdata/prd102/manifest.json");
    REQUIRE(source.good());
    const auto manifest = anet::json::parse(source);
    std::filesystem::create_directories(output / "config-inputs");
    // 同じ17入力を、編集前の隔離設定と現用設定へ渡して全キーを保存する。
    for (const auto* stage : {"before", "after"}) {
        const auto config_root = std::string(stage) == "before" ? output / "baseline-config" : repo / "apps/runner/config";
        anet::json snapshots = anet::json::object();
        for (const auto& input : manifest.at("inputs")) {
            const auto id = input.at("id").get<std::string>();
            INFO(std::string(stage) + "/" + id);
            const auto path = output / "config-inputs" / (std::string(stage) + "-" + id + ".txt");
            {
                std::ofstream stream(path);
                stream << "$include <" << (config_root / "_main.txt").generic_string() << ">\n";
                stream << "$include <" << (config_root / (input.at("env").get<std::string>() + ".txt")).generic_string() << ">\n";
                REQUIRE(stream.good());
            }
            const wxCmdLineEntryDesc description[] = {
                {wxCMD_LINE_PARAM, nullptr, nullptr, "key=value", wxCMD_LINE_VAL_STRING,
                    wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE},
                {wxCMD_LINE_NONE}
            };
            const auto cli = input.at("cli").get<std::string>();
            wxCmdLineParser command_line(description, wxString::FromUTF8(cli));
            REQUIRE(command_line.Parse(false) == 0);
            const anet::ConfigManager manager(path, &command_line);
            snapshots[id] = manager.GetConfigData().ToJson();
        }
        std::ofstream stream(output / (std::string("config-") + stage + ".json"));
        stream << snapshots.dump(2) << '\n';
        REQUIRE(stream.good());
    }
}

namespace {

class GridMazeNoopMetricsBackend final : public anet::IBackend {
public:
    void Open(const std::filesystem::path& runs_dir, const std::string& run_name) override
    {
        std::filesystem::create_directories(runs_dir / run_name);
    }
    void WriteJsonl(const anet::json&) override {}
    void Flush() override {}
};

class GridMazeMetricsGuard final {
public:
    GridMazeMetricsGuard()
    {
        anet::MetricsLogger::Reset();
        anet::MetricsLoggerConfig config;
        config.run_name_tmpl = "gridmaze_process_test";
        anet::MetricsLogger::Init(std::make_unique<GridMazeNoopMetricsBackend>(), config,
            std::filesystem::current_path() / ".scratch/prd102/runner-metrics");
    }
    ~GridMazeMetricsGuard() { anet::MetricsLogger::Reset(); }
};

} // namespace

TEST_CASE("RunManager shares GridMaze process seed across training evaluation panels and Runs", "[.][prd102_runner]")
{
    anet::test::LogCaptureGuard logs;
    GridMazeMetricsGuard metrics;
    anet::rl::EnvRepository::Instance().Regist(std::make_shared<anet::rl::env::GridMazeEnvFactory>());
    const auto repo = std::filesystem::current_path();
    const auto path = repo / ".scratch/prd102/runner-input.txt";
    {
        std::ofstream stream(path);
        stream << "$include <" << (repo / "apps/runner/config/_main.txt").generic_string() << ">\n";
        stream << "$include <" << (repo / "apps/runner/config/GridMaze.txt").generic_string() << ">\n";
        REQUIRE(stream.good());
    }
    const auto resolved = anet::ConfigManager(path).GetConfigData();
    anet::ConfigData data;
    // 実験用seedを持つ現用設定は保持し、この試験だけ未指定のRunを作る。
    for (const auto& [key, value] : resolved.Map()) {
        if (key != "GridMazeEnv.seed") data.Set(key, value);
    }
    for (const auto* key : {"agent.device", "env.device", "run.eval_device"}) data.Set(key, "cpu");
    data.Set("run.seed", 1);
    data.Set("run.train.num_envs", 2);
    data.Set("run.train.runner_type", "serial");
    data.Set("DefaultDQNAgent.skip.enabled", false);
    data.Set("DefaultDQNAgent.learner.use_amp", false);
    data.Set("DefaultDQNAgent.learner.replay_batch_size", 16);
    data.Set("DefaultDQNAgent.learner.replay_capacity", 4096);
    data.Set("DefaultDQNAgent.learner.update_warmup_steps", 100000);
    data.Set("run.eval_schedule.[eval].interval", 100);
    data.Set("run.eval_schedule.[eval_target].interval", 0);
    data.Set("run.eval.[panel].run_mode", "eval");
    data.Set("run.eval.[panel].actor", "eval");
    // 通常起動と同じ公開初期化でNNの型とAgent factoryを登録する。
    anet::rl::InitRL(anet::rl::BackendConfig(data), data);
    anet::rl::AuxData original;
    // 1つ目のRunを全て破棄した後も、別Runの全用途で共通seedを使う。
    for (int run = 0; run < 2; ++run) {
        data.Set("run.seed", run + 1);
        anet::rl::RunManager manager(data);
        REQUIRE(manager.GetStatus() == anet::rl::RunnerStatus::RUNNING);
        const auto train = manager.GetTrainRunner()->GetBatchEnv();
        const auto eval = manager.GetEvalRunner("eval")->GetBatchEnv();
        const auto panel = manager.CreateEvalRunner("EvalPanel", "panel")->GetBatchEnv();
        for (const auto& env : {train, eval, panel}) {
            const auto reset = env->Reset();
            for (int lane = 0; lane < env->GetBatchSpec().num_envs; ++lane) {
                const auto aux = reset->GetAuxDataList(lane).at(0);
                if (original.empty()) original = aux;
                CHECK(torch::equal(original.at("map"), aux.at("map")));
                CHECK(original.at("seed").item<int64_t>() == aux.at("seed").item<int64_t>());
            }
        }
    }
    logs.Flush();
    int notices = 0;
    for (const auto& record : logs.Records()) {
        if (record.message.find("GridMaze process seed initialized:") != std::string::npos) ++notices;
    }
    CHECK(notices == 1);
}
