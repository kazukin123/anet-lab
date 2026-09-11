#include "AtariEnv.hpp"
#include "AtariPreprocess.hpp"

#include <clocale>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>

#include "anet/catch_test.hpp"
#include "anet/env/Atari.hpp"
#include "anet/test_util.hpp"
#include "anet/thread.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

    std::optional<std::filesystem::path> FindRom(const std::string& game)
    {
#ifdef _WIN32
        char* env_value = nullptr;
        size_t env_size = 0;
        if (_dupenv_s(&env_value, &env_size, "ATARI_ROM_DIR") != 0 || env_value == nullptr) {
            return std::nullopt;
        }
        const std::string rom_dir(env_value);
        std::free(env_value);
#else
        const char* rom_dir = std::getenv("ATARI_ROM_DIR");
        if (rom_dir == nullptr || std::string(rom_dir).empty()) {
            return std::nullopt;
        }
#endif
        if (rom_dir.empty()) return std::nullopt;
        const auto path = std::filesystem::path(rom_dir) / (game + ".bin");
        return std::filesystem::exists(path) ? std::optional(path) : std::nullopt;
    }

    void SetupUtf8Console()
    {
#ifdef _WIN32
        SetConsoleCP(CP_UTF8);
        SetConsoleOutputCP(CP_UTF8);
#endif
        std::setlocale(LC_CTYPE, ".UTF-8");
    }

    int64_t AuxInt64(const std::shared_ptr<const anet::rl::SingleStepResult>& result, const std::string& key)
    {
        const auto aux = result->GetAuxData();
        return aux.at(key).item<int64_t>();
    }

    float AuxFloat(const std::shared_ptr<const anet::rl::SingleStepResult>& result, const std::string& key)
    {
        const auto aux = result->GetAuxData();
        return aux.at(key).item<float>();
    }

} // namespace

TEST_CASE("InitAtari registers the public SingleDiscreteEnv factory", "[atari][factory]")
{
    anet::rl::env::InitAtari();

    const auto factory = anet::rl::EnvRepository::Instance()
        .GetSingleDiscreteEnvFactory("AtariEnv");

    REQUIRE(factory);
    CHECK(factory->GetTargetEnvClassId() == "AtariEnv");
    CHECK(factory->GetDefaultConfigPrefix() == "AtariEnv");
}

TEST_CASE("AtariEnvConfig exposes the v5 default contract", "[atari][config]")
{
    anet::ConfigData config_data;
    config_data.Set("AtariEnv.game", "pong");

    const anet::rl::env::AtariEnvConfig config(config_data);

    CHECK(config.game == "pong");
    CHECK(config.rom_dir.empty());
    CHECK(config.screen_size == 84);
    CHECK(config.frame_skip == 4);
    CHECK(config.max_pool);
    CHECK(config.repeat_action_probability == Catch::Approx(0.25f));
    CHECK(config.noop_max == 0);
    CHECK_FALSE(config.fire_reset);
    CHECK_FALSE(config.episodic_life);
    CHECK(config.reward_clip);
    CHECK_FALSE(config.full_action_space);
    CHECK(config.mode == -1);
    CHECK(config.difficulty == -1);
    CHECK(config.max_episode_frames == 108000);
    CHECK(config.retain_rgb_frame);
    CHECK_FALSE(config.display_screen);
    CHECK_FALSE(config.sound);
}

TEST_CASE("Atari.txt synthesizes v5_noop30 and classic presets through AutoMerge", "[atari][config]")
{
    const auto source_root = std::filesystem::path(ANET_SOURCE_DIR);
    const auto config_dir = source_root / "apps" / "runner" / "config";
    const auto temp_dir = source_root / "tmp" / "atari-config-preset-test";
    std::filesystem::create_directories(temp_dir);

    const auto load_preset = [&](const std::string& preset) {
        const auto overlay_path = temp_dir / (preset + ".txt");
        {
            std::ofstream overlay(overlay_path);
            overlay << "$include <Atari.txt>\n";
            overlay << "AtariEnv.$ = AtariEnv.@" << preset << "\n";
        }
        anet::ConfigManagerOptions options;
        options.config_search_dirs = { config_dir };
        options.overwrite_config_paths = { overlay_path };
        return anet::ConfigManager(config_dir / "_main.txt", nullptr, options).GetConfigData();
    };

    const auto v5 = load_preset("v5_noop30");
    CHECK(v5.Get<float>("AtariEnv.repeat_action_probability") == Catch::Approx(0.25f));
    CHECK(v5.Get<int>("AtariEnv.noop_max") == 30);
    CHECK_FALSE(v5.Get<bool>("AtariEnv.episodic_life"));
    CHECK_FALSE(v5.Get<bool>("AtariEnv.fire_reset"));

    const auto classic = load_preset("classic");
    CHECK(classic.Get<float>("AtariEnv.repeat_action_probability") == Catch::Approx(0.0f));
    CHECK(classic.Get<int>("AtariEnv.noop_max") == 30);
    CHECK(classic.Get<bool>("AtariEnv.episodic_life"));
    CHECK(classic.Get<bool>("AtariEnv.fire_reset"));

    std::filesystem::remove_all(temp_dir);
}

TEST_CASE("AtariEnv fails fast on invalid current config values", "[atari][config]")
{
    anet::rl::env::AtariEnvFactory factory;
    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto check_invalid = [&](const anet::ConfigData& config_data, const std::string& message) {
        CHECK_THROWS_WITH(
            factory.CreateSingleEnv(
                config_data, torch::Device(torch::kCPU), "invalid-atari", 1,
                anet::rl::RunMode::Train),
            Catch::Matchers::ContainsSubstring(message));
    };

    anet::ConfigData config_data;
    config_data.Set("AtariEnv.game", "Pong");
    check_invalid(config_data, "AtariEnv.game must be a non-empty snake_case ROM stem");

    config_data.Set("AtariEnv.game", "pong");
    config_data.Set("AtariEnv.screen_size", 0);
    check_invalid(config_data, "AtariEnv.screen_size must be > 0");

    config_data.Set("AtariEnv.screen_size", 84);
    config_data.Set("AtariEnv.frame_skip", 0);
    check_invalid(config_data, "AtariEnv.frame_skip must be >= 1");

    config_data.Set("AtariEnv.frame_skip", 4);
    config_data.Set("AtariEnv.repeat_action_probability", -0.01);
    check_invalid(config_data, "AtariEnv.repeat_action_probability must be in [0,1]");
    config_data.Set("AtariEnv.repeat_action_probability", 1.01);
    check_invalid(config_data, "AtariEnv.repeat_action_probability must be in [0,1]");
    config_data.Set("AtariEnv.repeat_action_probability", 0.25);
    config_data.Set("AtariEnv.noop_max", -1);
    check_invalid(config_data, "AtariEnv.noop_max must be >= 0");

    config_data.Set("AtariEnv.noop_max", 0);
    config_data.Set("AtariEnv.mode", -2);
    check_invalid(config_data, "AtariEnv.mode must be >= -1");

    config_data.Set("AtariEnv.mode", -1);
    config_data.Set("AtariEnv.difficulty", -2);
    check_invalid(config_data, "AtariEnv.difficulty must be >= -1");

    config_data.Set("AtariEnv.difficulty", -1);
    config_data.Set("AtariEnv.max_episode_frames", -1);
    check_invalid(config_data, "AtariEnv.max_episode_frames must be >= 0");
}

TEST_CASE("AtariEnv does not fall back from an invalid explicit ROM directory", "[atari][config][rom]")
{
    anet::ConfigData config_data;
    config_data.Set("AtariEnv.game", "pong");
    config_data.Set("AtariEnv.rom_dir", "missing-explicit-rom-directory");

    anet::rl::env::AtariEnvFactory factory;
    anet::test::LogCaptureGuard logs(wxLOG_Info);
    CHECK_THROWS_WITH(
        factory.CreateSingleEnv(
            config_data, torch::Device(torch::kCPU), "missing-atari-rom", 1,
            anet::rl::RunMode::Train),
        Catch::Matchers::ContainsSubstring("missing-explicit-rom-directory")
            && Catch::Matchers::ContainsSubstring("Set AtariEnv.rom_dir or ATARI_ROM_DIR"));
}

TEST_CASE("HumanNormalizedScore reproduces the Nature 2015 normalized DQN column", "[atari][hns]")
{
    using anet::rl::env::HnsBaseline;
    using anet::rl::env::HumanNormalizedScore;

    // Mnih et al. 2015 Extended Data Table 2 の DQN スコアを入れると、
    // 同表の "Normalized DQN (% Human)" 列が再現される。転記の検算。
    // 表の掲載値は小数第 1 位までの丸めなので margin を置く。
    struct Case { const char* game; float dqn; float expected; };
    const std::vector<Case> cases{
        { "breakout",          401.2f,  1327.2f },
        { "pong",               18.9f,   132.0f },
        { "boxing",             71.8f,  1707.9f },
        { "video_pinball",   42684.0f,  2539.4f },
        { "robotank",           51.6f,   509.0f },
        { "montezuma_revenge",   0.0f,     0.0f },
        { "venture",           380.0f,    32.0f },
        { "asteroids",        1629.0f,     7.3f },
        { "gravitar",          306.7f,     5.3f },
        { "double_dunk",       -18.1f,    17.1f },
    };
    for (const auto& c : cases) {
        const auto hns = HumanNormalizedScore(c.game, c.dqn, HnsBaseline::Nature49);
        REQUIRE(hns.has_value());
        INFO("game=" << c.game);
        CHECK(*hns == Catch::Approx(c.expected).margin(1.0));
    }
}

TEST_CASE("HumanNormalizedScore uses distinct baselines for the 57 and 49 game tables", "[atari][hns]")
{
    using anet::rl::env::HnsBaseline;
    using anet::rl::env::HumanNormalizedScore;

    // Pong の human は 57 表 14.6 / 49 表 9.3 で、同じ生スコアでも HNS が大きく違う。
    const auto pong57 = HumanNormalizedScore("pong", 7.0f, HnsBaseline::Dqn57);
    const auto pong49 = HumanNormalizedScore("pong", 7.0f, HnsBaseline::Nature49);
    REQUIRE(pong57.has_value());
    REQUIRE(pong49.has_value());
    CHECK(*pong57 == Catch::Approx(78.5f).margin(0.5));
    CHECK(*pong49 == Catch::Approx(92.3f).margin(0.5));

    // Breakout の human は 30.5 / 31.8。
    const auto breakout57 = HumanNormalizedScore("breakout", 119.0f, HnsBaseline::Dqn57);
    REQUIRE(breakout57.has_value());
    CHECK(*breakout57 == Catch::Approx(407.3f).margin(0.5));
}

TEST_CASE("HumanNormalizedScore reports missing games instead of guessing", "[atari][hns]")
{
    using anet::rl::env::HnsBaseline;
    using anet::rl::env::HumanNormalizedScore;

    // ALE には 104 ゲームあるが、標準表は 57 / 49 しか covers しない。
    CHECK_FALSE(HumanNormalizedScore("casino", 100.0f, HnsBaseline::Dqn57).has_value());
    CHECK_FALSE(HumanNormalizedScore("casino", 100.0f, HnsBaseline::Nature49).has_value());

    // 57 表のみに載る 8 ゲームは、49 表では未登録になる。
    for (const char* game : { "berzerk", "defender", "phoenix", "pitfall",
                              "skiing", "solaris", "surround", "yars_revenge" }) {
        INFO("game=" << game);
        CHECK(HumanNormalizedScore(game, 0.0f, HnsBaseline::Dqn57).has_value());
        CHECK_FALSE(HumanNormalizedScore(game, 0.0f, HnsBaseline::Nature49).has_value());
    }
}

TEST_CASE("PixelwiseMax returns the maximum value for every pixel", "[atari][preprocess]")
{
    const std::vector<uint8_t> first{ 0, 200, 30, 255 };
    const std::vector<uint8_t> second{ 10, 20, 40, 254 };
    std::vector<uint8_t> output;

    anet::rl::env::atari::PixelwiseMax(first, second, output);

    CHECK(output == std::vector<uint8_t>{ 10, 200, 40, 255 });
}

TEST_CASE("RollingMaxPool uses only the final two frames of one step", "[atari][preprocess]")
{
    anet::rl::env::atari::RollingMaxPool pool;
    pool.Push({ 250, 1 });
    pool.Push({ 2, 100 });
    pool.Push({ 3, 4 });
    std::vector<uint8_t> output;

    pool.Finish(output);

    CHECK(output == std::vector<uint8_t>{ 3, 100 });
}

TEST_CASE("RollingMaxPool uses its sole frame and fresh steps do not retain prior frames", "[atari][preprocess]")
{
    anet::rl::env::atari::RollingMaxPool first_step;
    first_step.Push({ 250, 200 });
    std::vector<uint8_t> first_output;
    first_step.Finish(first_output);

    anet::rl::env::atari::RollingMaxPool second_step;
    second_step.Push({ 1, 2 });
    std::vector<uint8_t> second_output;
    second_step.Finish(second_output);

    CHECK(first_output == std::vector<uint8_t>{ 250, 200 });
    CHECK(second_output == std::vector<uint8_t>{ 1, 2 });
}

TEST_CASE("ResizeGrayscale area-resizes and rounds to uint8", "[atari][preprocess]")
{
    const std::vector<uint8_t> source{
        0, 1, 10, 11,
        2, 2, 12, 12,
        100, 101, 200, 201,
        102, 103, 202, 204,
    };

    const auto output = anet::rl::env::atari::ResizeGrayscale(
        source.data(), 4, 4, 2);

    CHECK(output.sizes().vec() == std::vector<int64_t>{ 1, 2, 2 });
    CHECK(output.dtype() == torch::kUInt8);
    CHECK(output[0][0][0].item<uint8_t>() == 1);
    CHECK(output[0][0][1].item<uint8_t>() == 11);
    CHECK(output[0][1][0].item<uint8_t>() == 102);
    CHECK(output[0][1][1].item<uint8_t>() == 202);
}

TEST_CASE("InterleavedRgbToChw preserves adjacent RGB pixel channels", "[atari][preprocess][rgb]")
{
    // 1画素だけではplanar/interleavedを区別できないため、隣接2画素を固定する。
    const std::vector<uint8_t> source{
        10, 20, 30,
        40, 50, 60,
    };

    const auto output = anet::rl::env::atari::InterleavedRgbToChw(
        source.data(), 1, 2);

    REQUIRE(output.sizes().vec() == std::vector<int64_t>{ 3, 1, 2 });
    CHECK(output.dtype() == torch::kUInt8);
    CHECK(output.is_contiguous());
    CHECK(output[0][0][0].item<uint8_t>() == 10);
    CHECK(output[0][0][1].item<uint8_t>() == 40);
    CHECK(output[1][0][0].item<uint8_t>() == 20);
    CHECK(output[1][0][1].item<uint8_t>() == 50);
    CHECK(output[2][0][0].item<uint8_t>() == 30);
    CHECK(output[2][0][1].item<uint8_t>() == 60);
}

TEST_CASE("AtariEnv exposes Pong spec and reset observation through the public Env seam", "[atari][rom]")
{
    const auto rom = FindRom("pong");
    if (!rom.has_value()) {
        SKIP("pong.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }

    anet::ConfigData config_data;
    config_data.Set("AtariEnv.game", "pong");
    config_data.Set("AtariEnv.rom_dir", rom->parent_path().string());

    anet::rl::env::AtariEnvFactory factory;
    const auto env = factory.CreateSingleEnv(
        config_data, torch::Device(torch::kCPU), "atari-test", 123,
        anet::rl::RunMode::Train);

    const auto spec = env->GetSpec();
    REQUIRE(spec.action_spec.is_discrete);
    CHECK(spec.action_spec.value_labels.size() == 6);
    REQUIRE(spec.state_spec.obs_spec.contains(anet::rl::ObsKeys::kGrid));
    const auto& grid_spec = spec.state_spec.obs_spec.at(anet::rl::ObsKeys::kGrid);
    CHECK(grid_spec.shape == std::vector<int64_t>{ 1, 84, 84 });
    CHECK(grid_spec.dtype == torch::kUInt8);

    const auto reset = env->Reset();
    CHECK(reset->state.episode_start);
    CHECK(spec.state_spec.ValidateObservation(reset->state.obs, false));

    const auto step = env->Step(0);
    CHECK(spec.state_spec.ValidateObservation(step->next_state.obs, false));
    const auto aux = step->GetAuxData();
    REQUIRE(aux.contains("game_score"));
    REQUIRE(aux.contains("game_len"));
    REQUIRE(aux.contains("game_frames"));
    REQUIRE(aux.contains("lives"));
    CHECK(aux.at("game_score").dtype() == torch::kFloat32);
    CHECK(aux.at("game_len").dtype() == torch::kInt64);
    CHECK(aux.at("game_frames").dtype() == torch::kInt64);
    CHECK(aux.at("lives").dtype() == torch::kInt64);
    CHECK(env->GetScalar("lives").has_value());
}

TEST_CASE("AtariEnv full action space exposes all 18 player-A actions", "[atari][rom][actions]")
{
    const auto rom = FindRom("pong");
    if (!rom.has_value()) {
        SKIP("pong.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }

    anet::ConfigData config_data;
    config_data.Set("AtariEnv.game", "pong");
    config_data.Set("AtariEnv.rom_dir", rom->parent_path().string());
    config_data.Set("AtariEnv.full_action_space", true);
    anet::rl::env::AtariEnvFactory factory;
    const auto env = factory.CreateSingleEnv(
        config_data, torch::Device(torch::kCPU), "atari-full-actions", 123,
        anet::rl::RunMode::Train);

    const auto spec = env->GetSpec();
    REQUIRE(spec.action_spec.value_labels.size() == 18);
    CHECK(spec.action_spec.value_labels.front() == "NOOP");
    anet::test::LogCaptureGuard logs(wxLOG_Info);
    CHECK_THROWS(env->Step(-1));
    CHECK_THROWS(env->Step(18));
}

TEST_CASE("AtariEnv exposes an owned RGB frame when retention is enabled", "[atari][rom][rgb]")
{
    const auto rom = FindRom("pong");
    if (!rom.has_value()) {
        SKIP("pong.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }

    anet::ConfigData config_data;
    config_data.Set("AtariEnv.game", "pong");
    config_data.Set("AtariEnv.rom_dir", rom->parent_path().string());
    anet::rl::env::AtariEnvFactory factory;
    const auto env = factory.CreateSingleEnv(
        config_data, torch::Device(torch::kCPU), "atari-rgb", 123,
        anet::rl::RunMode::Train);

    env->Reset();
    const auto first = env->GetTensor("rgb_frame");
    REQUIRE(first.has_value());
    CHECK(first->sizes().vec() == std::vector<int64_t>{ 3, 210, 160 });
    CHECK(first->dtype() == torch::kUInt8);
    CHECK(first->is_contiguous());
    const auto snapshot = first->clone();

    env->Step(0);
    CHECK(torch::equal(*first, snapshot));
    CHECK(env->GetTensor("unknown").has_value() == false);
}

TEST_CASE("AtariEnv omits RGB frame when retention is disabled", "[atari][rom][rgb]")
{
    const auto rom = FindRom("pong");
    if (!rom.has_value()) {
        SKIP("pong.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }

    anet::ConfigData config_data;
    config_data.Set("AtariEnv.game", "pong");
    config_data.Set("AtariEnv.rom_dir", rom->parent_path().string());
    config_data.Set("AtariEnv.retain_rgb_frame", false);
    anet::rl::env::AtariEnvFactory factory;
    const auto env = factory.CreateSingleEnv(
        config_data, torch::Device(torch::kCPU), "atari-no-rgb", 123,
        anet::rl::RunMode::Train);

    env->Reset();
    CHECK_FALSE(env->GetTensor("rgb_frame").has_value());
}

TEST_CASE("AtariEnv rejects a mode or difficulty unavailable for the ROM", "[atari][rom][config]")
{
    const auto rom = FindRom("pong");
    if (!rom.has_value()) {
        SKIP("pong.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }

    anet::rl::env::AtariEnvFactory factory;
    anet::test::LogCaptureGuard logs(wxLOG_Info);
    anet::ConfigData mode_config;
    mode_config.Set("AtariEnv.game", "pong");
    mode_config.Set("AtariEnv.rom_dir", rom->parent_path().string());
    mode_config.Set("AtariEnv.mode", 999);
    CHECK_THROWS_WITH(
        factory.CreateSingleEnv(
            mode_config, torch::Device(torch::kCPU), "atari-bad-mode", 1,
            anet::rl::RunMode::Train),
        Catch::Matchers::ContainsSubstring("AtariEnv.mode=999")
            && Catch::Matchers::ContainsSubstring("available="));

    anet::ConfigData difficulty_config;
    difficulty_config.Set("AtariEnv.game", "pong");
    difficulty_config.Set("AtariEnv.rom_dir", rom->parent_path().string());
    difficulty_config.Set("AtariEnv.difficulty", 999);
    CHECK_THROWS_WITH(
        factory.CreateSingleEnv(
            difficulty_config, torch::Device(torch::kCPU), "atari-bad-difficulty", 1,
            anet::rl::RunMode::Train),
        Catch::Matchers::ContainsSubstring("AtariEnv.difficulty=999")
            && Catch::Matchers::ContainsSubstring("available="));
}

TEST_CASE("AtariEnv accepts current config boundary values", "[atari][rom][config]")
{
    const auto rom = FindRom("pong");
    if (!rom.has_value()) {
        SKIP("pong.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }

    anet::ConfigData config_data;
    config_data.Set("AtariEnv.game", "pong");
    config_data.Set("AtariEnv.rom_dir", rom->parent_path().string());
    config_data.Set("AtariEnv.screen_size", 1);
    config_data.Set("AtariEnv.frame_skip", 1);
    config_data.Set("AtariEnv.repeat_action_probability", 1.0);
    config_data.Set("AtariEnv.noop_max", 0);
    config_data.Set("AtariEnv.mode", -1);
    config_data.Set("AtariEnv.difficulty", -1);
    config_data.Set("AtariEnv.max_episode_frames", 0);

    anet::rl::env::AtariEnvFactory factory;
    const auto env = factory.CreateSingleEnv(
        config_data, torch::Device(torch::kCPU), "atari-config-boundaries", 1,
        anet::rl::RunMode::Train);

    CHECK(env->Reset()->state.obs.At(anet::rl::ObsKeys::kGrid).sizes().vec()
        == std::vector<int64_t>{ 1, 1, 1 });
}

TEST_CASE("AtariEnv is reproducible for the same seed and action sequence", "[atari][rom][seed]")
{
    const auto rom = FindRom("pong");
    if (!rom.has_value()) {
        SKIP("pong.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }

    anet::ConfigData config_data;
    config_data.Set("AtariEnv.game", "pong");
    config_data.Set("AtariEnv.rom_dir", rom->parent_path().string());
    anet::rl::env::AtariEnvFactory factory;
    const auto first = factory.CreateSingleEnv(
        config_data, torch::Device(torch::kCPU), "atari-seed-a", 9876,
        anet::rl::RunMode::Train);
    const auto second = factory.CreateSingleEnv(
        config_data, torch::Device(torch::kCPU), "atari-seed-b", 9876,
        anet::rl::RunMode::Train);

    const auto first_reset = first->Reset();
    const auto second_reset = second->Reset();
    CHECK(torch::equal(
        first_reset->state.obs.At(anet::rl::ObsKeys::kGrid),
        second_reset->state.obs.At(anet::rl::ObsKeys::kGrid)));

    const std::vector<int64_t> actions{ 0, 1, 2, 3, 4, 5, 0, 3, 2, 1 };
    for (const auto action : actions) {
        const auto first_step = first->Step(action);
        const auto second_step = second->Step(action);
        CHECK(first_step->reward == second_step->reward);
        CHECK(first_step->next_state.done == second_step->next_state.done);
        CHECK(first_step->next_state.truncated == second_step->next_state.truncated);
        CHECK(torch::equal(
            first_step->next_state.obs.At(anet::rl::ObsKeys::kGrid),
            second_step->next_state.obs.At(anet::rl::ObsKeys::kGrid)));
    }
}

TEST_CASE("AtariEnv reports an ALE-frame limit as truncation with completion metrics", "[atari][rom][terminal]")
{
    const auto rom = FindRom("pong");
    if (!rom.has_value()) {
        SKIP("pong.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }

    anet::ConfigData config_data;
    config_data.Set("AtariEnv.game", "pong");
    config_data.Set("AtariEnv.rom_dir", rom->parent_path().string());
    config_data.Set("AtariEnv.frame_skip", 1);
    config_data.Set("AtariEnv.max_episode_frames", 2);
    anet::rl::env::AtariEnvFactory factory;
    const auto env = factory.CreateSingleEnv(
        config_data, torch::Device(torch::kCPU), "atari-truncation", 123,
        anet::rl::RunMode::Train);

    env->Reset();
    const auto first_step = env->Step(0);
    CHECK_FALSE(first_step->next_state.done);
    CHECK_FALSE(first_step->next_state.truncated);
    const auto step = env->Step(0);

    CHECK_FALSE(step->next_state.done);
    CHECK(step->next_state.truncated);
    REQUIRE(env->GetScalar("game_frames").has_value());
    CHECK(*env->GetScalar("game_frames") >= 1.0f);
    CHECK(std::isfinite(*env->GetScalar("game_score")));
    CHECK(*env->GetScalar("game_len") == 2.0f);

    env->Reset();
    CHECK(std::isfinite(*env->GetScalar("game_score")));
    const auto next_episode_step = env->Step(0);
    CHECK_FALSE(next_episode_step->next_state.truncated);
    CHECK(std::isnan(*env->GetScalar("game_score")));
}

TEST_CASE("AtariEnv exposes game_score threshold indicators", "[atari][rom][terminal]")
{
    const auto rom = FindRom("pong");
    if (!rom.has_value()) {
        SKIP("pong.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }

    anet::ConfigData config_data;
    config_data.Set("AtariEnv.game", "pong");
    config_data.Set("AtariEnv.rom_dir", rom->parent_path().string());
    config_data.Set("AtariEnv.frame_skip", 1);
    config_data.Set("AtariEnv.max_episode_frames", 2);
    anet::rl::env::AtariEnvFactory factory;
    // 閾値不正の fail-fast を確認するので、wx ログを捕捉してダイアログを出さない。
    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto env = factory.CreateSingleEnv(
        config_data, torch::Device(torch::kCPU), "atari-score-threshold", 123,
        anet::rl::RunMode::Train);

    env->Reset();

    // 未確定 step は game_score と同じく NaN。0 を返すと集約の分母が完了 env 数でなくなる。
    const auto pending = env->GetScalar("game_score.ge.[0]");
    REQUIRE(pending.has_value());
    CHECK(std::isnan(*pending));

    env->Step(0);
    env->Step(0);
    const auto score = env->GetScalar("game_score");
    REQUIRE(score.has_value());
    REQUIRE(std::isfinite(*score));

    CHECK(*env->GetScalar("game_score.ge.[-1000]") == 1.0f);  // 負の閾値も許す(Pong は -21..21)
    CHECK(*env->GetScalar("game_score.ge.[1000]") == 0.0f);
    CHECK(*env->GetScalar("game_score.ge.[0]") == (*score >= 0.0f ? 1.0f : 0.0f));
    CHECK(*env->GetScalar("game_score.ge.[0.5]") == (*score >= 0.5f ? 1.0f : 0.0f));

    // 前方一致しないキーは未知キーのまま返し、GetScalar の既定経路へ落とす。
    CHECK_FALSE(env->GetScalar("game_score.ge").has_value());
    CHECK_FALSE(env->GetScalar("game_score_ge432").has_value());

    // 前方一致するのに閾値が読めないキーは黙って無視せず停止する。
    CHECK_THROWS_WITH(env->GetScalar("game_score.ge.[]"),
        Catch::Matchers::ContainsSubstring("threshold is empty"));
    CHECK_THROWS_WITH(env->GetScalar("game_score.ge.[abc]"),
        Catch::Matchers::ContainsSubstring("invalid threshold"));
    CHECK_THROWS_WITH(env->GetScalar("game_score.ge.[1x]"),
        Catch::Matchers::ContainsSubstring("invalid threshold"));
}

TEST_CASE("AtariEnv clips only the returned reward and keeps raw episode score", "[atari][rom][reward][terminal]")
{
    const auto rom = FindRom("pong");
    if (!rom.has_value()) {
        SKIP("pong.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }

    anet::ConfigData clipped_config;
    clipped_config.Set("AtariEnv.game", "pong");
    clipped_config.Set("AtariEnv.rom_dir", rom->parent_path().string());
    constexpr int64_t kFrameSkip = 100000;
    clipped_config.Set("AtariEnv.frame_skip", kFrameSkip);
    clipped_config.Set("AtariEnv.repeat_action_probability", 0.0);
    clipped_config.Set("AtariEnv.reward_clip", true);
    clipped_config.Set("AtariEnv.max_episode_frames", 0);
    auto raw_config = clipped_config;
    raw_config.Set("AtariEnv.reward_clip", false);

    anet::rl::env::AtariEnvFactory factory;
    const auto clipped = factory.CreateSingleEnv(
        clipped_config, torch::Device(torch::kCPU), "atari-clipped", 1357,
        anet::rl::RunMode::Train);
    const auto raw = factory.CreateSingleEnv(
        raw_config, torch::Device(torch::kCPU), "atari-raw", 1357,
        anet::rl::RunMode::Train);
    clipped->Reset();
    raw->Reset();

    // 十分大きいskip窓でreal game overまで進め、複数点の生報酬を1 Stepへ集約する。
    const auto clipped_step = clipped->Step(0);
    const auto raw_step = raw->Step(0);

    REQUIRE(clipped_step->next_state.done);
    REQUIRE(raw_step->next_state.done);
    CHECK_FALSE(clipped_step->next_state.truncated);
    CHECK_FALSE(raw_step->next_state.truncated);
    REQUIRE(std::abs(raw_step->reward) > 1.0f);
    CHECK(clipped_step->reward == std::copysign(1.0f, raw_step->reward));
    CHECK(AuxFloat(clipped_step, "game_score") == Catch::Approx(raw_step->reward));
    CHECK(AuxFloat(raw_step, "game_score") == Catch::Approx(raw_step->reward));
    CHECK(*clipped->GetScalar("game_score") == Catch::Approx(raw_step->reward));
    CHECK(*raw->GetScalar("game_score") == Catch::Approx(raw_step->reward));

    // real game overでskip窓を中断し、ALEが実際に進めたframe数を完了値として返す。
    const auto clipped_aux_frames = AuxInt64(clipped_step, "game_frames");
    const auto raw_aux_frames = AuxInt64(raw_step, "game_frames");
    REQUIRE(clipped_aux_frames > 0);
    CHECK(clipped_aux_frames < kFrameSkip);
    CHECK(raw_aux_frames == clipped_aux_frames);
    const auto clipped_scalar_frames = clipped->GetScalar("game_frames");
    const auto raw_scalar_frames = raw->GetScalar("game_frames");
    REQUIRE(clipped_scalar_frames.has_value());
    REQUIRE(raw_scalar_frames.has_value());
    CHECK(*clipped_scalar_frames == static_cast<float>(clipped_aux_frames));
    CHECK(*raw_scalar_frames == static_cast<float>(raw_aux_frames));
}

TEST_CASE("AtariEnv keeps vectorized and thread-pool batch semantics", "[atari][rom][batch]")
{
    const auto rom = FindRom("pong");
    if (!rom.has_value()) {
        SKIP("pong.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }

    anet::ConfigData config_data;
    config_data.Set("AtariEnv.game", "pong");
    config_data.Set("AtariEnv.rom_dir", rom->parent_path().string());
    config_data.Set("AtariEnv.max_episode_frames", 4);
    auto factory = std::make_shared<anet::rl::env::AtariEnvFactory>();
    anet::rl::VectorizedDiscreteBatchEnv vectorized(
        config_data, factory, "atari-vectorized", 2, torch::Device(torch::kCPU), 123);
    anet::rl::ThreadPoolDiscreteEnv threaded(
        config_data, factory, "atari-threaded", 2, torch::Device(torch::kCPU),
        std::make_shared<anet::PinnedThreadPool>(2, "atari-test-pool"), 123);

    const auto vectorized_reset = vectorized.Reset();
    const auto threaded_reset = threaded.Reset();
    CHECK(vectorized.GetSpec().state_spec.ValidateObservation(vectorized_reset->state.obs, true));
    CHECK(threaded.GetSpec().state_spec.ValidateObservation(threaded_reset->state.obs, true));

    auto actions = std::make_shared<anet::rl::BatchActionInfo>(
        torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kInt64)));
    const auto vectorized_step = vectorized.Step(actions);
    const auto threaded_step = threaded.Step(actions);
    CHECK(vectorized_step->n_transitions == 2);
    CHECK(threaded_step->n_transitions == 2);
    CHECK(torch::equal(vectorized_step->reward, threaded_step->reward));
    CHECK(torch::equal(
        vectorized_step->next_state.obs.At(anet::rl::ObsKeys::kGrid),
        threaded_step->next_state.obs.At(anet::rl::ObsKeys::kGrid)));
    CHECK(vectorized_step->next_state.truncated.all().item<bool>());
    CHECK(threaded_step->next_state.truncated.all().item<bool>());
    CHECK(vectorized_step->continue_state.episode_start.all().item<bool>());
    CHECK(threaded_step->continue_state.episode_start.all().item<bool>());
}

TEST_CASE("AtariEnv evaluates episodic life only after the full skip window", "[atari][rom][life]")
{
    const auto rom = FindRom("breakout");
    if (!rom.has_value()) {
        SKIP("breakout.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }

    anet::ConfigData skipped_config;
    skipped_config.Set("AtariEnv.game", "breakout");
    skipped_config.Set("AtariEnv.rom_dir", rom->parent_path().string());
    skipped_config.Set("AtariEnv.frame_skip", 4);
    skipped_config.Set("AtariEnv.repeat_action_probability", 0.0);
    skipped_config.Set("AtariEnv.fire_reset", true);
    skipped_config.Set("AtariEnv.episodic_life", true);
    skipped_config.Set("AtariEnv.reward_clip", false);
    skipped_config.Set("AtariEnv.max_episode_frames", 0);
    skipped_config.Set("AtariEnv.screen_size", 1);
    skipped_config.Set("AtariEnv.max_pool", false);
    skipped_config.Set("AtariEnv.retain_rgb_frame", false);

    auto reference_config = skipped_config;
    reference_config.Set("AtariEnv.frame_skip", 1);
    reference_config.Set("AtariEnv.episodic_life", false);

    anet::rl::env::AtariEnvFactory factory;
    {
        auto full_config = skipped_config;
        full_config.Set("AtariEnv.full_action_space", true);
        const auto full = factory.CreateSingleEnv(
            full_config, torch::Device(torch::kCPU), "atari-breakout-full", 2468,
            anet::rl::RunMode::Train);
        const auto full_spec = full->GetSpec();
        CHECK(full_spec.action_spec.value_labels.size() == 18);
        CHECK(full_spec.action_spec.value_labels[1] == "FIRE");
        CHECK(full_spec.state_spec.ValidateObservation(full->Reset()->state.obs, false));
    }

    const auto skipped = factory.CreateSingleEnv(
        skipped_config, torch::Device(torch::kCPU), "atari-life-skipped", 2468,
        anet::rl::RunMode::Train);
    const auto reference = factory.CreateSingleEnv(
        reference_config, torch::Device(torch::kCPU), "atari-life-reference", 2468,
        anet::rl::RunMode::Train);
    const auto skipped_reset = skipped->Reset();
    reference->Reset();
    const auto skipped_spec = skipped->GetSpec();
    CHECK(skipped_spec.action_spec.value_labels.size() == 4);
    CHECK(skipped_spec.action_spec.value_labels[1] == "FIRE");
    CHECK(skipped_spec.state_spec.ValidateObservation(skipped_reset->state.obs, false));

    bool observed_life_loss = false;
    for (int window = 0; window < 5000 && !observed_life_loss; ++window) {
        const int64_t action = window == 0 ? 1 : 0; // FIREで開始後はNOOPを維持する。
        const auto before_lives = *reference->GetScalar("lives");
        const auto skipped_step = skipped->Step(action);
        float reference_reward = 0.0f;
        std::shared_ptr<const anet::rl::SingleStepResult> reference_step;
        for (int frame = 0; frame < 4; ++frame) {
            reference_step = reference->Step(action);
            reference_reward += reference_step->reward;
            if (reference_step->next_state.done) break;
        }

        REQUIRE(reference_step);
        CHECK(skipped_step->reward == reference_reward);
        CHECK(AuxInt64(skipped_step, "game_frames") == AuxInt64(reference_step, "game_frames"));

        const auto after_lives = *reference->GetScalar("lives");
        if (!reference_step->next_state.done && after_lives < before_lives) {
            observed_life_loss = true;
            CHECK(skipped_step->next_state.done);
            CHECK_FALSE(skipped_step->next_state.truncated);
            const auto life_loss_frame = AuxInt64(skipped_step, "game_frames");
            const auto life_loss_score = AuxFloat(skipped_step, "game_score");
            CHECK(life_loss_frame % 4 == 2); // FIRE reset 2 frames (FIRE + action set 3番目) + 完走したskip窓
            CHECK(std::isnan(*skipped->GetScalar("game_score")));

            // life-loss後のsoft resetは実ゲームの集約を維持したままNOOP 1回を打つ。
            // fire_reset=true なので続けてFIRE + action set 3番目が入る（FireResetEnv 同手順、計3フレーム）。
            const auto soft_reset = skipped->Reset();
            const auto reset_aux = soft_reset->GetAuxData();
            CHECK(soft_reset->state.episode_start);
            CHECK(reset_aux.at("game_frames").item<int64_t>() == life_loss_frame + 3);
            CHECK(reset_aux.at("game_score").item<float>() == Catch::Approx(life_loss_score));
            CHECK(std::isnan(*skipped->GetScalar("game_score")));
        }
    }

    REQUIRE(observed_life_loss);

    // 以後もlife-lossだけでは完了値を確定せず、実ゲーム終了までsoft resetで継続する。
    bool observed_real_game_over = false;
    for (int window = 0; window < 25000 && !observed_real_game_over; ++window) {
        const auto step = skipped->Step(1); // FIREを維持して次のlifeを確実に開始する。
        const auto completed_score = *skipped->GetScalar("game_score");
        if (std::isfinite(completed_score)) {
            observed_real_game_over = true;
            REQUIRE(step->next_state.done);
            CHECK_FALSE(step->next_state.truncated);
            CHECK(completed_score == Catch::Approx(AuxFloat(step, "game_score")));
            break;
        }

        CHECK(std::isnan(completed_score));
        if (step->next_state.done) {
            CHECK_FALSE(step->next_state.truncated);
            const auto reset = skipped->Reset();
            CHECK(reset->state.episode_start);
            const auto reset_completed_score = *skipped->GetScalar("game_score");
            if (std::isfinite(reset_completed_score)) {
                // soft-reset NOOP中のreal game overも実ゲーム完了として扱う。
                observed_real_game_over = true;
            } else {
                CHECK(std::isnan(reset_completed_score));
            }
        } else {
            CHECK_FALSE(step->next_state.truncated);
        }
    }

    CHECK(observed_real_game_over);
}

TEST_CASE("AtariEnv soft reset applies the fire sequence only when fire_reset is enabled", "[atari][rom][life]")
{
    const auto rom = FindRom("breakout");
    if (!rom.has_value()) {
        SKIP("breakout.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }

    // fire_reset=false（v5 既定）では life-loss 後の soft reset は NOOP 1 回のまま。
    // true にすると FIRE + action set 3番目が続き（FireResetEnv 同手順）、計 3 フレームになる。
    const auto make = [&](bool fire_reset) {
        anet::ConfigData config_data;
        config_data.Set("AtariEnv.game", "breakout");
        config_data.Set("AtariEnv.rom_dir", rom->parent_path().string());
        config_data.Set("AtariEnv.frame_skip", 4);
        config_data.Set("AtariEnv.repeat_action_probability", 0.0);
        config_data.Set("AtariEnv.fire_reset", fire_reset);
        config_data.Set("AtariEnv.episodic_life", true);
        config_data.Set("AtariEnv.reward_clip", false);
        config_data.Set("AtariEnv.max_episode_frames", 0);
        config_data.Set("AtariEnv.screen_size", 1);
        config_data.Set("AtariEnv.max_pool", false);
        config_data.Set("AtariEnv.retain_rgb_frame", false);
        anet::rl::env::AtariEnvFactory factory;
        return factory.CreateSingleEnv(
            config_data, torch::Device(torch::kCPU),
            fire_reset ? "atari-soft-fire-on" : "atari-soft-fire-off", 2468,
            anet::rl::RunMode::Train);
    };

    for (const bool fire_reset : { false, true }) {
        const auto env = make(fire_reset);
        env->Reset();

        bool observed = false;
        for (int window = 0; window < 5000 && !observed; ++window) {
            const int64_t action = window == 0 ? 1 : 0;  // FIREで開始しNOOPを維持する
            const auto step = env->Step(action);
            if (!step->next_state.done) continue;

            observed = true;
            const auto before = AuxInt64(step, "game_frames");
            const auto reset_aux = env->Reset()->GetAuxData();
            const auto after = reset_aux.at("game_frames").item<int64_t>();
            INFO("fire_reset=" << fire_reset);
            CHECK(after == before + (fire_reset ? 3 : 1));
        }
        REQUIRE(observed);
    }
}

int main(int argc, char* argv[])
{
    SetupUtf8Console();

    anet::test::PreparedTestArgs test_args;
    try {
        test_args = anet::test::PrepareTestArgs(argc, argv);
    } catch (const std::exception& e) {
        return anet::test::ReportTestArgsError(e);
    }
    anet::test::SetupTestFailureDialog(test_args.failure_dialog_enabled);

    Catch::Session session;
    session.configData().showDurations = Catch::ShowDurations::Always;
    return session.run(test_args.Argc(), test_args.Argv());
}
