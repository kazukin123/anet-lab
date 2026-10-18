#include "AtariEnv.hpp"
#include "AtariPreprocess.hpp"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <wx/cmdline.h>

#include "anet/catch_test.hpp"
#include "anet/env/Atari.hpp"
#include "anet/observers.hpp"
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

    const auto load_preset = [&](const std::string& preset) {
        // Atari.txt の既定 run.$ は Run 層で AtariEnv.$ を選ぶので、プリセットはそれより強い CLI 層で選ぶ。
        const wxCmdLineEntryDesc description[] = {
            { wxCMD_LINE_PARAM, nullptr, nullptr, "key=value", wxCMD_LINE_VAL_STRING,
                wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE }, {wxCMD_LINE_NONE},
        };
        wxCmdLineParser cli(description, wxString::FromUTF8("AtariEnv.$=AtariEnv.@" + preset));
        REQUIRE(cli.Parse(false) == 0);
        anet::ConfigManagerOptions options;
        options.config_search_dirs = { config_dir };
        options.overwrite_config_paths = { config_dir / "Atari.txt" };
        return anet::ConfigManager(config_dir / "_main.txt", &cli, options).GetConfigData();
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

    CHECK(v5.Get("AtariEnv.ram_metric.[kung_fu_master].metrics") == "1:floor_clear 2:boss_kill 3:boss_hit");
    CHECK(v5.Get("AtariEnv.ram_metric.[qbert].metrics") == "1:round_clear");
    CHECK(v5.Get("AtariEnv.ram_metric.[phoenix].metrics") == "1:wave_clear 2:boss_kill");
    CHECK(v5.Get("AtariEnv.ram_metric.[breakout].metrics") == "1:wall_clear");
    const anet::rl::env::AtariEnvConfig ram_config(v5);
    CHECK(ram_config.known_ram_metric_numbers == std::set<int64_t>{1, 2, 3});
    CHECK(v5.Get("metrics.trace.[51_eval1/episode]").find("ram_metric.[1] ram_metric.[2] ram_metric.[3]") != std::string::npos);
    CHECK(v5.Get("metrics.trace.[42_env/episode]").find("ram_metric.[1] ram_metric.[2] ram_metric.[3]") != std::string::npos);
    CHECK(v5.Get("metrics.scalar.[42_env/50_ram_metric_1_mean]").find("mean.ram_metric.[1]") != std::string::npos);
    const anet::rl::ObserverFactory metric_factory(v5);
    const auto& trace_defs = metric_factory.GetTraceMetricDefs();
    const auto trace = std::find_if(trace_defs.begin(), trace_defs.end(), [](const auto& def) {
        return def.tag == "42_env/episode";
    });
    REQUIRE(trace != trace_defs.end());
    CHECK(trace->keys == std::vector<std::string>{
        "game_score", "game_len", "game_frames", "hns57",
        "ram_metric.[1]", "ram_metric.[2]", "ram_metric.[3]"});
    const auto& scalar_defs = metric_factory.GetScalarMetricDefs();
    const auto find_scalar = [&](const std::string& tag) {
        return std::find_if(scalar_defs.begin(), scalar_defs.end(), [&](const auto& def) {
            return def.tag == tag;
        });
    };
    const auto check_ram_scalar = [&](const std::string& tag, const std::string& source_key,
        bool has_ema, float ema_alpha, const std::string& reference_tag) {
        INFO(tag);
        const auto scalar = find_scalar(tag);
        const auto reference = find_scalar(reference_tag);
        REQUIRE(scalar != scalar_defs.end());
        REQUIRE(reference != scalar_defs.end());
        CHECK(scalar->source_key == source_key);
        CHECK(scalar->has_ema == has_ema);
        if (has_ema) CHECK(scalar->ema_alpha == Catch::Approx(ema_alpha));
        CHECK(scalar->step_axis == reference->step_axis);
        CHECK(scalar->runner == reference->runner);
        CHECK(scalar->event == reference->event);
        CHECK(scalar->target == reference->target);
        CHECK(scalar->scope == reference->scope);
        CHECK(scalar->eval_name == reference->eval_name);
        CHECK(scalar->interval == reference->interval);
    };
    struct RamScalarGroup {
        std::string name;
        bool has_ema;
        float ema_alpha;
    };
    const std::vector<RamScalarGroup> groups{
        {"42_env", true, 0.001f},
        {"51_eval1", true, 0.1f},
        {"52_eval2", true, 0.1f},
        {"53_evalg", false, 0.0f},
    };
    for (const auto& group : groups) {
        for (int number = 1; number <= 3; ++number) {
            const std::string stem = group.name + "/";
            const std::string key = "ram_metric.[" + std::to_string(number) + "]";
            const std::string suffix = "ram_metric_" + std::to_string(number);
            const std::string mean_tag = stem + std::to_string(48 + 2 * number) + "_" + suffix + "_mean";
            const std::string max_tag = stem + std::to_string(55 + number) + "_" + suffix + "_max";
            check_ram_scalar(mean_tag, "mean." + key, false, 0.0f, stem + "10_game_score_mean");
            check_ram_scalar(max_tag, "max." + key, false, 0.0f, stem + "16_game_score_max");
            if (group.has_ema) {
                const std::string ema_tag = stem + std::to_string(49 + 2 * number) + "_" + suffix + "_mean_ema";
                check_ram_scalar(ema_tag, "mean." + key, true, group.ema_alpha,
                    stem + "11_game_score_mean_ema");
            }
        }
    }
    CHECK(std::count_if(scalar_defs.begin(), scalar_defs.end(), [](const auto& def) {
        return def.tag.find("_ram_metric_") != std::string::npos;
    }) == 33);
    for (const std::string old_tag : {
        "42_env/50_stage_clear_mean", "42_env/51_boss_kill_mean", "42_env/52_boss_hit_mean",
        "53_evalg/51_ram_metric_1_mean_ema", "53_evalg/53_ram_metric_2_mean_ema",
        "53_evalg/55_ram_metric_3_mean_ema"}) {
        CHECK(find_scalar(old_tag) == scalar_defs.end());
        CHECK_FALSE(v5.Has("metrics.scalar.[" + old_tag + "]"));
    }
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

TEST_CASE("AtariEnv reports a configured RAM metric at game completion", "[atari][rom][ram_metric]")
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
    config_data.Set("AtariEnv.repeat_action_probability", 0.0f);
    config_data.Set("AtariEnv.ram_metric.[pong].[score_high]", "0x8D max_seen");
    config_data.Set("AtariEnv.ram_metric.[pong].metrics", "1:score_high");

    anet::rl::env::AtariEnvFactory factory;
    const auto env = factory.CreateSingleEnv(
        config_data, torch::Device(torch::kCPU), "atari-ram-pong", 123,
        anet::rl::RunMode::Train);
    const auto effective = env->GetConfigData();
    REQUIRE(effective.has_value());
    CHECK(effective->Get("AtariEnv.ram_metric.[pong].[score_high]") == "0x8D max_seen");
    CHECK(effective->Get("AtariEnv.ram_metric.[pong].metrics") == "1:score_high");

    const auto reset = env->Reset();
    REQUIRE(reset->GetAuxData().contains("ram_metric.score_high"));
    CHECK(reset->GetAuxData().at("ram_metric.score_high").dtype() == torch::kInt64);
    REQUIRE(env->GetScalar("ram_metric.[1]"));
    CHECK(std::isnan(*env->GetScalar("ram_metric.[1]")));

    const auto pending = env->Step(0);
    CHECK_FALSE(pending->next_state.truncated);
    CHECK(std::isnan(*env->GetScalar("ram_metric.[1]")));

    const auto completed = env->Step(0);
    REQUIRE(completed->next_state.truncated);
    REQUIRE(completed->GetAuxData().contains("ram_metric.score_high"));
    CHECK(*env->GetScalar("ram_metric.[1]") ==
        static_cast<float>(completed->GetAuxData().at("ram_metric.score_high").item<int64_t>()));
}

TEST_CASE("RAM metric reducers distinguish the initial value from transitions", "[atari][ram_metric]")
{
    using anet::rl::env::RamMetricDefinition;
    using anet::rl::env::RamMetricReducer;
    using anet::rl::env::RamMetricState;

    RamMetricState max_seen{.definition = RamMetricDefinition{.reducer = RamMetricReducer::MaxSeen}};
    max_seen.Begin(39);
    max_seen.Observe(35);
    max_seen.Observe(40);
    CHECK(max_seen.Value() == 40);

    RamMetricState min_seen{.definition = RamMetricDefinition{.reducer = RamMetricReducer::MinSeen}};
    min_seen.Begin(39);
    min_seen.Observe(35);
    min_seen.Observe(40);
    CHECK(min_seen.Value() == 35);

    RamMetricState inc_count{.definition = RamMetricDefinition{.reducer = RamMetricReducer::IncCount}};
    inc_count.Begin(39);
    inc_count.Observe(39);
    inc_count.Observe(40);
    inc_count.Observe(39);
    CHECK(inc_count.Value() == 1);

    RamMetricState dec_count{.definition = RamMetricDefinition{.reducer = RamMetricReducer::DecCount}};
    dec_count.Begin(39);
    dec_count.Observe(35);
    dec_count.Observe(0);
    dec_count.Observe(39);
    CHECK(dec_count.Value() == 2);

    RamMetricState reach_count{.definition = RamMetricDefinition{
        .reducer = RamMetricReducer::ReachCount, .target = 0}};
    reach_count.Begin(0);
    reach_count.Observe(0);
    reach_count.Observe(1);
    reach_count.Observe(0);
    reach_count.Observe(0);
    CHECK(reach_count.Value() == 1);
}

TEST_CASE("AtariEnvConfig reads all RAM reducers and shared metric numbers", "[atari][ram_metric][config]")
{
    anet::ConfigData data;
    data.Set("AtariEnv.game", "pong");
    data.Set("AtariEnv.ram_metric.[pong].[top]", "0x8D max_seen");
    data.Set("AtariEnv.ram_metric.[pong].[bottom]", "0x8E min_seen");
    data.Set("AtariEnv.ram_metric.[pong].[rise]", "0x8D inc_count");
    data.Set("AtariEnv.ram_metric.[pong].[fall]", "0x8E dec_count");
    data.Set("AtariEnv.ram_metric.[pong].[zero]", "0x8D reach_count:0");
    data.Set("AtariEnv.ram_metric.[pong].metrics", "1:top 2:bottom 3:rise 4:fall 5:zero");
    data.Set("AtariEnv.ram_metric.[qbert].[round]", "0xE3 inc_count");
    data.Set("AtariEnv.ram_metric.[qbert].metrics", "1:round");
    data.Set("AtariEnv.ram_metric.[breakout].[knowledge_only]", "0xFF min_seen");

    const anet::rl::env::AtariEnvConfig config(data);
    CHECK(config.known_ram_metric_numbers.size() == 5);
    CHECK(config.ram_metrics.at("pong").size() == 5);
    CHECK(config.ram_metrics.at("qbert").size() == 1);
    CHECK_FALSE(config.ram_metrics.contains("breakout"));
    CHECK(config.ram_metrics.at("pong").at(5).reducer == anet::rl::env::RamMetricReducer::ReachCount);
    CHECK(config.ram_metrics.at("pong").at(5).target == 0);
}

TEST_CASE("AtariEnvConfig reports accepted RAM declarations in its effective config", "[atari][ram_metric][config]")
{
    anet::ConfigData data;
    data.Set("AtariEnv.game", "pong");
    data.Set("AtariEnv.ram_metric.[pong].[score]", "0x8D max_seen");
    data.Set("AtariEnv.ram_metric.[pong].metrics", "1:score");
    data.Set("AtariEnv.ram_metric.[qbert].[knowledge_only]", "0xE3 min_seen");

    const anet::rl::env::AtariEnvConfig config(data, "run.eval.[eval_target].env");
    const auto effective = config.GetScopedConfigData();
    CHECK(effective.Get("run.eval.[eval_target].env.ram_metric.[pong].[score]") == "0x8D max_seen");
    CHECK(effective.Get("run.eval.[eval_target].env.ram_metric.[pong].metrics") == "1:score");
    CHECK(effective.Get("run.eval.[eval_target].env.ram_metric.[qbert].[knowledge_only]") == "0xE3 min_seen");
    CHECK(config.ToJson().at("ram_metric.[pong].[score]") == "0x8D max_seen");
    CHECK(config.ToJson().at("ram_metric.[pong].metrics") == "1:score");
}

TEST_CASE("AtariEnvConfig rejects malformed RAM definitions before ROM loading", "[atari][ram_metric][config]")
{
    anet::ConfigData base;
    base.Set("AtariEnv.game", "pong");
    base.Set("AtariEnv.ram_metric.[pong].[top]", "0x8D max_seen");
    base.Set("AtariEnv.ram_metric.[pong].metrics", "1:top");
    const std::vector<std::pair<std::string, std::string>> invalid_values{
        {"AtariEnv.ram_metric.[pong].[top]", "128 max_seen"},
        {"AtariEnv.ram_metric.[pong].[top]", "0x00 max_seen"},
        {"AtariEnv.ram_metric.[pong].[top]", "0xGG max_seen"},
        {"AtariEnv.ram_metric.[pong].[top]", "0x8D unknown"},
        {"AtariEnv.ram_metric.[pong].[top]", "0x8D reach_count:256"},
        {"AtariEnv.ram_metric.[pong].[top]", "0x8D reach_count:-1"},
        {"AtariEnv.ram_metric.[qbert].[bad]", "0x00 max_seen"},
        {"AtariEnv.ram_metric.[pong].metrics", "0:top"},
        {"AtariEnv.ram_metric.[pong].metrics", "1x:top"},
        {"AtariEnv.ram_metric.[pong].metrics", "1:top 1:top"},
        {"AtariEnv.ram_metric.[pong].metrics", "1:missing"},
        {"AtariEnv.ram_metric.[pong].metrics", ""},
        {"AtariEnv.ram_metric.[pong].typo", "value"},
        {"AtariEnv.ram_metric.[pong].[bad-label]", "0x8D max_seen"},
        {"AtariEnv.ram_metric.[BadGame].[top]", "0x8D max_seen"},
        {"AtariEnv.ram_metric.[pong].broken.[top]", "0x8D max_seen"},
        {"run.eval.[eval_target].env.ram_metric.[pong].[top]", "0x8D max_seen"},
    };
    anet::test::LogCaptureGuard logs(wxLOG_Info);
    for (const auto& [key, value] : invalid_values) {
        auto data = base;
        data.Set(key, value);
        INFO(key << " = " << value);
        CHECK_THROWS(anet::rl::env::AtariEnvConfig(data));
    }
}

TEST_CASE("AtariEnv distinguishes absent RAM numbers from malformed scalar keys", "[atari][rom][ram_metric]")
{
    const auto rom = FindRom("pong");
    if (!rom.has_value()) {
        SKIP("pong.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }
    anet::ConfigData data;
    data.Set("AtariEnv.game", "pong");
    data.Set("AtariEnv.rom_dir", rom->parent_path().string());
    data.Set("AtariEnv.frame_skip", 1);
    data.Set("AtariEnv.max_episode_frames", 1);
    data.Set("AtariEnv.ram_metric.[qbert].[round]", "0xE3 inc_count");
    data.Set("AtariEnv.ram_metric.[qbert].metrics", "2:round");
    anet::rl::env::AtariEnvFactory factory;
    const auto env = factory.CreateSingleEnv(data, torch::Device(torch::kCPU), "ram-absent", 123);
    REQUIRE(env->Step(0)->next_state.truncated);
    CHECK(std::isnan(*env->GetScalar("ram_metric.[2]")));
    CHECK(std::isnan(*env->GetScalar("ram_metric.[02]")));
    CHECK_FALSE(env->GetScalar("ram_metric.[3]").has_value());
    for (const std::string key : {"ram_metric.[]", "ram_metric.[0]", "ram_metric.[-1]",
            "ram_metric.[1x]", "ram_metric.[+1]", "ram_metric.[999999999999999999999]"}) {
        INFO(key);
        CHECK_THROWS_WITH(env->GetScalar(key), Catch::Matchers::ContainsSubstring("ram_metric"));
    }
}

TEST_CASE("AtariEnv appends numbered RAM values to the game completion log", "[atari][rom][ram_metric][log]")
{
    const auto rom = FindRom("pong");
    if (!rom.has_value()) {
        SKIP("pong.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }
    anet::ConfigData data;
    data.Set("AtariEnv.game", "pong");
    data.Set("AtariEnv.rom_dir", rom->parent_path().string());
    data.Set("AtariEnv.frame_skip", 1);
    data.Set("AtariEnv.max_episode_frames", 1);
    data.Set("AtariEnv.ram_metric.[pong].[later]", "0x8E max_seen");
    data.Set("AtariEnv.ram_metric.[pong].[first]", "0x8D max_seen");
    data.Set("AtariEnv.ram_metric.[pong].metrics", "3:later 1:first");
    anet::test::LogCaptureGuard logs(wxLOG_Info);
    anet::rl::env::AtariEnvFactory factory;
    const auto env = factory.CreateSingleEnv(data, torch::Device(torch::kCPU), "ram-log", 123);
    REQUIRE(env->Step(0)->next_state.truncated);
    logs.Flush();
    const auto suffix = " ram_metric: first="
        + std::to_string(static_cast<int64_t>(*env->GetScalar("ram_metric.[1]")))
        + " later=" + std::to_string(static_cast<int64_t>(*env->GetScalar("ram_metric.[3]")));
    bool found = false;
    for (const auto& record : logs.Records()) {
        if (record.message.starts_with("ram-log: Game truncated by max_episode_frames.")) {
            found = record.message.ends_with(suffix);
        }
    }
    CHECK(found);
}

TEST_CASE("AtariEnv reports Kung Fu Master progress from the configured RAM bytes", "[atari][rom][ram_metric]")
{
    const auto rom = FindRom("kung_fu_master");
    if (!rom.has_value()) {
        SKIP("kung_fu_master.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }
    anet::ConfigData data;
    data.Set("AtariEnv.game", "kung_fu_master");
    data.Set("AtariEnv.rom_dir", rom->parent_path().string());
    data.Set("AtariEnv.frame_skip", 1);
    data.Set("AtariEnv.max_episode_frames", 2);
    data.Set("AtariEnv.repeat_action_probability", 0.0f);
    data.Set("AtariEnv.ram_metric.[kung_fu_master].[floor_clear]", "0x9F inc_count");
    data.Set("AtariEnv.ram_metric.[kung_fu_master].[boss_kill]", "0xCC reach_count:0");
    data.Set("AtariEnv.ram_metric.[kung_fu_master].[boss_hit]", "0xCC dec_count");
    data.Set("AtariEnv.ram_metric.[kung_fu_master].[floor_max]", "0x9F max_seen");
    data.Set("AtariEnv.ram_metric.[kung_fu_master].[boss_hp_min]", "0xCC min_seen");
    data.Set("AtariEnv.ram_metric.[kung_fu_master].metrics",
        "1:floor_clear 2:boss_kill 3:boss_hit 4:floor_max 5:boss_hp_min");
    anet::rl::env::AtariEnvFactory factory;
    const auto env = factory.CreateSingleEnv(data, torch::Device(torch::kCPU), "ram-kfm", 123);
    REQUIRE(env->Step(0)->next_state.done == false);
    REQUIRE(env->Step(0)->next_state.truncated);
    CHECK(*env->GetScalar("ram_metric.[1]") == 0.0f);
    CHECK(*env->GetScalar("ram_metric.[2]") == 0.0f);
    CHECK(*env->GetScalar("ram_metric.[3]") == 0.0f);
    CHECK(*env->GetScalar("ram_metric.[4]") == 1.0f);
    CHECK(*env->GetScalar("ram_metric.[5]") == 39.0f);
}

TEST_CASE("AtariEnv counts RAM transitions inside one frame skip window", "[atari][rom][ram_metric]")
{
    const auto rom = FindRom("pong");
    if (!rom.has_value()) {
        SKIP("pong.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }
    anet::ConfigData data;
    data.Set("AtariEnv.game", "pong");
    data.Set("AtariEnv.rom_dir", rom->parent_path().string());
    data.Set("AtariEnv.frame_skip", 100000);
    data.Set("AtariEnv.max_episode_frames", 0);
    data.Set("AtariEnv.repeat_action_probability", 0.0f);
    data.Set("AtariEnv.ram_metric.[pong].[cpu_points]", "0x8D inc_count");
    data.Set("AtariEnv.ram_metric.[pong].metrics", "1:cpu_points");
    anet::rl::env::AtariEnvFactory factory;
    const auto env = factory.CreateSingleEnv(data, torch::Device(torch::kCPU), "ram-pong-count", 1357);
    const auto step = env->Step(0);
    REQUIRE(step->next_state.done);
    CHECK(*env->GetScalar("ram_metric.[1]") == 21.0f);
    CHECK(step->GetAuxData().at("ram_metric.cpu_points").item<int64_t>() == 21);
}

TEST_CASE("AtariEnv keeps RAM counts through life-loss reset and clears them on hard reset", "[atari][rom][ram_metric]")
{
    const auto rom = FindRom("kung_fu_master");
    if (!rom.has_value()) {
        SKIP("kung_fu_master.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }
    anet::ConfigData data;
    data.Set("AtariEnv.game", "kung_fu_master");
    data.Set("AtariEnv.rom_dir", rom->parent_path().string());
    data.Set("AtariEnv.frame_skip", 800);
    data.Set("AtariEnv.max_pool", false);
    data.Set("AtariEnv.retain_rgb_frame", false);
    data.Set("AtariEnv.screen_size", 1);
    data.Set("AtariEnv.max_episode_frames", 0);
    data.Set("AtariEnv.episodic_life", true);
    data.Set("AtariEnv.repeat_action_probability", 0.0f);
    data.Set("AtariEnv.ram_metric.[kung_fu_master].[life_loss]", "0x9D dec_count");
    data.Set("AtariEnv.ram_metric.[kung_fu_master].metrics", "1:life_loss");
    anet::rl::env::AtariEnvFactory factory;
    const auto env = factory.CreateSingleEnv(data, torch::Device(torch::kCPU), "ram-soft-reset", 123);

    const auto life_done = env->Step(0);
    REQUIRE(life_done->next_state.done);
    CHECK(std::isnan(*env->GetScalar("ram_metric.[1]")));
    const int64_t before = life_done->GetAuxData().at("ram_metric.life_loss").item<int64_t>();
    CHECK(before >= 1);

    const auto soft_reset = env->Reset();
    CHECK(soft_reset->GetAuxData().at("ram_metric.life_loss").item<int64_t>() == before);
    const auto hard_reset = env->Reset();
    CHECK(hard_reset->GetAuxData().at("ram_metric.life_loss").item<int64_t>() == 0);
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

TEST_CASE("AtariEnv logs one verbose line per completed game with the completion values", "[atari][rom][terminal][log]")
{
    const auto rom = FindRom("pong");
    if (!rom.has_value()) {
        SKIP("pong.bin is unavailable; set ATARI_ROM_DIR to run ROM-dependent tests.");
    }

    anet::ConfigData truncated_config;
    truncated_config.Set("AtariEnv.game", "pong");
    truncated_config.Set("AtariEnv.rom_dir", rom->parent_path().string());
    truncated_config.Set("AtariEnv.frame_skip", 1);
    truncated_config.Set("AtariEnv.max_episode_frames", 2);
    // 十分大きい skip 窓なら 1 Step で real game over まで進む。
    anet::ConfigData game_over_config;
    game_over_config.Set("AtariEnv.game", "pong");
    game_over_config.Set("AtariEnv.rom_dir", rom->parent_path().string());
    game_over_config.Set("AtariEnv.frame_skip", 100000);
    game_over_config.Set("AtariEnv.repeat_action_probability", 0.0);
    game_over_config.Set("AtariEnv.max_episode_frames", 0);

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    anet::rl::env::AtariEnvFactory factory;
    const auto truncated = factory.CreateSingleEnv(
        truncated_config, torch::Device(torch::kCPU), "atari-log-truncated", 123,
        anet::rl::RunMode::Train);
    const auto game_over = factory.CreateSingleEnv(
        game_over_config, torch::Device(torch::kCPU), "atari-log-game-over", 1357,
        anet::rl::RunMode::Train);

    const auto game_lines = [&logs](const std::string& name) {
        logs.Flush();
        std::vector<std::string> lines;
        for (const auto& record : logs.Records()) {
            if (!record.message.starts_with(name + ": Game ")) continue;
            CHECK(record.level == wxLOG_Info);
            lines.push_back(record.message);
        }
        return lines;
    };
    // ログの値は trace / metrics が読む完了値と一致しなければならない。
    const auto completion_line = [](const std::string& name, const std::string& head, const auto& env) {
        return name + ": " + head
            + " game_score=" + std::to_string(static_cast<int64_t>(*env->GetScalar("game_score")))
            + " game_len=" + std::to_string(static_cast<int64_t>(*env->GetScalar("game_len")))
            + " game_frames=" + std::to_string(static_cast<int64_t>(*env->GetScalar("game_frames")));
    };

    truncated->Reset();
    truncated->Step(0);
    CHECK(game_lines("atari-log-truncated").empty());
    REQUIRE(truncated->Step(0)->next_state.truncated);
    const std::vector<std::string> expected_truncated{
        completion_line("atari-log-truncated", "Game truncated by max_episode_frames.", truncated) };
    CHECK(game_lines("atari-log-truncated") == expected_truncated);
    // 次のゲームの途中では増えない。
    truncated->Reset();
    truncated->Step(0);
    CHECK(game_lines("atari-log-truncated") == expected_truncated);

    game_over->Reset();
    const auto game_over_step = game_over->Step(0);
    REQUIRE(game_over_step->next_state.done);
    REQUIRE_FALSE(game_over_step->next_state.truncated);
    CHECK(game_lines("atari-log-game-over") == std::vector<std::string>{
        completion_line("atari-log-game-over", "Game over.", game_over) });
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

    anet::test::StderrLogGuard log_target_guard;

    Catch::Session session;
    session.configData().showDurations = Catch::ShowDurations::Always;
    return session.run(test_args.Argc(), test_args.Argv());
}
