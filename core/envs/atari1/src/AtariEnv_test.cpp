#include "AtariEnv.hpp"
#include "AtariPreprocess.hpp"
#include <ale/ale_interface.hpp>

#include <algorithm>
#include <array>
#include <random>
#include <clocale>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <wx/cmdline.h>

#include <torch/nn/functional/upsampling.h>

#include "anet/catch_test.hpp"
#include "anet/profile.hpp"
#include "anet/env/Atari.hpp"
#include "anet/test_util.hpp"
#include "anet/thread.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

// 検証対象の振る舞いは公開APIで読み、friendはALEの初期条件設定に限定する。
class anet::rl::env::AtariEnvTestAccess {
public:
    void PrepareLifeLossNoopCompletion(AtariEnv& env) const
    {
        const auto actions = env.ale_->getMinimalActionSet();
        uint32_t state = 1;
        for (int frame = 0; frame < 300; ++frame) {
            state = 1664525u * state + 1013904223u;
            env.ale_->act(actions[state % actions.size()]);
            if (env.ale_->game_over(false)) env.ale_->reset_game();
        }
        env.ale_->setRAM(0x53, 1);
        env.ale_->setRAM(0x47, 3);
    }
};

namespace {

    // 2026-10-24の高速化前の実装を固定したテスト参照。式と積算順を変更しない。
    class PixelAreaReferenceResizer final {
    public:
        PixelAreaReferenceResizer(int source_height, int source_width, int destination_size,
            const std::string& mode);
        torch::Tensor Resize(const uint8_t* source) const;

    private:
        struct PixelWeight {
            int index;
            int overlap;
        };
        static std::vector<std::vector<PixelWeight>> MakeWeights(int source_size, int destination_size);

        int source_height_;
        int source_width_;
        int destination_size_;
        bool pixel_area_;
        std::vector<std::vector<PixelWeight>> horizontal_weights_;
        std::vector<std::vector<PixelWeight>> vertical_weights_;
    };

    PixelAreaReferenceResizer::PixelAreaReferenceResizer(
        int source_height, int source_width, int destination_size, const std::string& mode)
        : source_height_(source_height)
        , source_width_(source_width)
        , destination_size_(destination_size)
        , pixel_area_(mode == "pixel_area")
    {
        // 設定に依存する検証と重みの準備は構築時に完了させる。
        if (source_height <= 0 || source_width <= 0 || destination_size <= 0) {
            ANET_SYSTEM_ERROR("PixelAreaReferenceResizer requires positive dimensions. source_height="
                << source_height << ", source_width=" << source_width
                << ", destination_size=" << destination_size << ". Expected dimensions > 0.");
        }
        if (mode != "adaptive_average" && mode != "pixel_area") {
            ANET_SYSTEM_ERROR("PixelAreaReferenceResizer mode='" << mode
                << "'. Expected adaptive_average or pixel_area.");
        }
        if (pixel_area_) {
            if (destination_size > std::min(source_height, source_width)) {
                ANET_SYSTEM_ERROR("PixelAreaReferenceResizer pixel_area destination_size=" << destination_size
                    << ". Expected <= " << std::min(source_height, source_width) << ".");
            }
            horizontal_weights_ = MakeWeights(source_width, destination_size);
            vertical_weights_ = MakeWeights(source_height, destination_size);
        }
    }

    std::vector<std::vector<PixelAreaReferenceResizer::PixelWeight>> PixelAreaReferenceResizer::MakeWeights(
        int source_size, int destination_size)
    {
        // 境界を出力サイズ倍した整数格子へ写し、正規化前の重なり幅を誤差なく保持する。
        std::vector<std::vector<PixelWeight>> result(destination_size);
        for (int destination = 0; destination < destination_size; ++destination) {
            const int64_t begin = static_cast<int64_t>(destination) * source_size;
            const int64_t end = static_cast<int64_t>(destination + 1) * source_size;
            const int first = static_cast<int>(begin / destination_size);
            const int last = static_cast<int>((end + destination_size - 1) / destination_size);
            for (int source = first; source < last; ++source) {
                const int64_t overlap = std::min(end, static_cast<int64_t>(source + 1) * destination_size)
                    - std::max(begin, static_cast<int64_t>(source) * destination_size);
                result[destination].push_back({.index = source, .overlap = static_cast<int>(overlap)});
            }
        }
        return result;
    }

    torch::Tensor PixelAreaReferenceResizer::Resize(const uint8_t* source) const
    {
        ANET_PROFILE_SCOPE(resize);
        if (source == nullptr) {
            ANET_SYSTEM_ERROR("PixelAreaReferenceResizer source=null. Expected a non-null grayscale image.");
        }
        if (!pixel_area_) {
            // 現行方式のfloat32 area補間とuint8への戻し方をそのまま保つ。
            auto input = torch::from_blob(
                const_cast<uint8_t*>(source), {1, 1, source_height_, source_width_},
                torch::TensorOptions().dtype(torch::kUInt8));
            auto resized = torch::nn::functional::interpolate(
                input.to(torch::kFloat32),
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{destination_size_, destination_size_})
                    .mode(torch::kArea));
            return resized.squeeze(0).round().clamp(0, 255).to(torch::kUInt8).contiguous();
        }

        // 正規化前の整数の和を横→縦へ渡し、途中の除算と丸めを避ける。
        std::vector<int64_t> horizontal(static_cast<size_t>(source_height_) * destination_size_, 0);
        for (int y = 0; y < source_height_; ++y) {
            for (int x = 0; x < destination_size_; ++x) {
                int64_t value = 0;
                for (const auto& weight : horizontal_weights_[x]) {
                    value += static_cast<int64_t>(source[y * source_width_ + weight.index]) * weight.overlap;
                }
                horizontal[y * destination_size_ + x] = value;
            }
        }
        auto output = torch::empty({1, destination_size_, destination_size_},
            torch::TensorOptions().dtype(torch::kUInt8));
        auto* pixels = output.data_ptr<uint8_t>();
        // 各軸の重みの和は元の辺長。両軸分を最後に一度だけ正規化する。
        const int64_t divisor = static_cast<int64_t>(source_height_) * source_width_;
        for (int y = 0; y < destination_size_; ++y) {
            for (int x = 0; x < destination_size_; ++x) {
                int64_t value = 0;
                for (const auto& weight : vertical_weights_[y]) {
                    value += horizontal[weight.index * destination_size_ + x] * weight.overlap;
                }
                // 商と余りなら数学的な半整数を正確に識別でき、浮動小数誤差が入らない。
                const int64_t integer = value / divisor;
                const int64_t twice_remainder = 2 * (value % divisor);
                const int64_t rounded = integer + (twice_remainder > divisor
                    || (twice_remainder == divisor && integer % 2 != 0));
                pixels[y * destination_size_ + x] = static_cast<uint8_t>(rounded);
            }
        }
        return output;
    }


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

TEST_CASE("GrayscaleResizer preserves adaptive average and uint8 rounding", "[atari][preprocess]")
{
    const std::vector<uint8_t> source{
        0, 1, 10, 11,
        2, 2, 12, 12,
        100, 101, 200, 201,
        102, 103, 202, 204,
    };

    const anet::rl::env::atari::GrayscaleResizer resizer(4, 4, 2, "adaptive_average");
    const auto output = resizer.Resize(source.data());

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
    CHECK_FALSE(env->GetScalar("ram_metric.[1]").has_value());

    const auto pending = env->Step(0);
    CHECK_FALSE(pending->next_state.truncated);
    CHECK(pending->GetScalar("ram_metric.[1]") ==
        static_cast<float>(pending->GetAuxData().at("ram_metric.score_high").item<int64_t>()));

    const auto completed = env->Step(0);
    REQUIRE(completed->next_state.truncated);
    REQUIRE(completed->GetAuxData().contains("ram_metric.score_high"));
    CHECK(*completed->GetScalar("ram_metric.[1]") ==
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
    const auto step = env->Step(0);
    REQUIRE(step->next_state.truncated);
    CHECK(std::isnan(*step->GetScalar("ram_metric.[2]")));
    CHECK(std::isnan(*step->GetScalar("ram_metric.[02]")));
    CHECK_FALSE(step->GetScalar("ram_metric.[3]").has_value());
    for (const std::string key : {"ram_metric.[]", "ram_metric.[0]", "ram_metric.[-1]",
            "ram_metric.[1x]", "ram_metric.[+1]", "ram_metric.[999999999999999999999]"}) {
        INFO(key);
        CHECK_THROWS_WITH(step->GetScalar(key), Catch::Matchers::ContainsSubstring("ram_metric"));
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
    const auto step = env->Step(0);
    REQUIRE(step->next_state.truncated);
    logs.Flush();
    const auto suffix = " ram_metric: first="
        + std::to_string(static_cast<int64_t>(*step->GetScalar("ram_metric.[1]")))
        + " later=" + std::to_string(static_cast<int64_t>(*step->GetScalar("ram_metric.[3]")));
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
    const auto step = env->Step(0);
    REQUIRE(step->next_state.truncated);
    CHECK(*step->GetScalar("ram_metric.[1]") == 0.0f);
    CHECK(*step->GetScalar("ram_metric.[2]") == 0.0f);
    CHECK(*step->GetScalar("ram_metric.[3]") == 0.0f);
    CHECK(*step->GetScalar("ram_metric.[4]") == 1.0f);
    CHECK(*step->GetScalar("ram_metric.[5]") == 39.0f);
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
    CHECK(*step->GetScalar("ram_metric.[1]") == 21.0f);
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
    CHECK(std::isnan(*life_done->GetScalar("ram_metric.[1]")));
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
    REQUIRE(step->GetScalar("game_frames").has_value());
    CHECK(*step->GetScalar("game_frames") >= 1.0f);
    CHECK(std::isfinite(*step->GetScalar("game_score")));
    CHECK(*step->GetScalar("game_len") == 2.0f);

    env->Reset();
    CHECK_FALSE(env->GetScalar("game_score").has_value());
    const auto next_episode_step = env->Step(0);
    CHECK_FALSE(next_episode_step->next_state.truncated);
    CHECK(next_episode_step->GetScalar("game_score") == AuxFloat(next_episode_step, "game_score"));
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
    const auto completion_line = [](const std::string& name, const std::string& head, const auto& result) {
        return name + ": " + head
            + " game_score=" + std::to_string(static_cast<int64_t>(*result->GetScalar("game_score")))
            + " game_len=" + std::to_string(static_cast<int64_t>(*result->GetScalar("game_len")))
            + " game_frames=" + std::to_string(static_cast<int64_t>(*result->GetScalar("game_frames")));
    };

    truncated->Reset();
    truncated->Step(0);
    CHECK(game_lines("atari-log-truncated").empty());
    const auto truncated_step = truncated->Step(0);
    REQUIRE(truncated_step->next_state.truncated);
    const std::vector<std::string> expected_truncated{
        completion_line("atari-log-truncated", "Game truncated by max_episode_frames.", truncated_step) };
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
        completion_line("atari-log-game-over", "Game over.", game_over_step) });
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

    // 途中のstepでも、結果が保持する速報スコアから任意閾値を計算できる。
    const auto first = env->Step(0);
    const auto pending = first->GetScalar("game_score.ge.[0]");
    REQUIRE(pending.has_value());
    CHECK(*pending == (*first->GetScalar("game_score") >= 0.0f ? 1.0f : 0.0f));

    const auto step = env->Step(0);
    const auto score = step->GetScalar("game_score");
    REQUIRE(score.has_value());
    REQUIRE(std::isfinite(*score));

    CHECK(*step->GetScalar("game_score.ge.[-1000]") == 1.0f);  // 負の閾値も許す(Pong は -21..21)
    CHECK(*step->GetScalar("game_score.ge.[1000]") == 0.0f);
    CHECK(*step->GetScalar("game_score.ge.[0]") == (*score >= 0.0f ? 1.0f : 0.0f));
    CHECK(*step->GetScalar("game_score.ge.[0.5]") == (*score >= 0.5f ? 1.0f : 0.0f));

    // 前方一致しないキーは未知キーのまま返し、GetScalar の既定経路へ落とす。
    CHECK_FALSE(step->GetScalar("game_score.ge").has_value());
    CHECK_FALSE(step->GetScalar("game_score_ge432").has_value());

    // 前方一致するのに閾値が読めないキーは黙って無視せず停止する。
    CHECK_THROWS_WITH(step->GetScalar("game_score.ge.[]"),
        Catch::Matchers::ContainsSubstring("threshold is empty"));
    CHECK_THROWS_WITH(step->GetScalar("game_score.ge.[abc]"),
        Catch::Matchers::ContainsSubstring("invalid threshold"));
    CHECK_THROWS_WITH(step->GetScalar("game_score.ge.[1x]"),
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
    CHECK(*clipped_step->GetScalar("game_score") == Catch::Approx(raw_step->reward));
    CHECK(*raw_step->GetScalar("game_score") == Catch::Approx(raw_step->reward));

    // real game overでskip窓を中断し、ALEが実際に進めたframe数を完了値として返す。
    const auto clipped_aux_frames = AuxInt64(clipped_step, "game_frames");
    const auto raw_aux_frames = AuxInt64(raw_step, "game_frames");
    REQUIRE(clipped_aux_frames > 0);
    CHECK(clipped_aux_frames < kFrameSkip);
    CHECK(raw_aux_frames == clipped_aux_frames);
    const auto clipped_scalar_frames = clipped_step->GetScalar("game_frames");
    const auto raw_scalar_frames = raw_step->GetScalar("game_frames");
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
            CHECK(std::isnan(*skipped_step->GetScalar("game_score")));

            // life-loss後のNOOPはStepで済み、soft resetは実ゲームの集約を維持する。
            // fire_reset=true なので続けてFIRE + action set 3番目が入る（FireResetEnv 同手順、計3フレーム）。
            const auto soft_reset = skipped->Reset();
            const auto reset_aux = soft_reset->GetAuxData();
            CHECK(soft_reset->state.episode_start);
            CHECK(reset_aux.at("game_frames").item<int64_t>() == life_loss_frame + 3);
            CHECK(reset_aux.at("game_score").item<float>() == Catch::Approx(life_loss_score));
            CHECK_FALSE(skipped->GetScalar("game_score").has_value());
        }
    }

    REQUIRE(observed_life_loss);

    // 以後もlife-lossだけでは完了値を確定せず、実ゲーム終了までsoft resetで継続する。
    bool observed_real_game_over = false;
    for (int window = 0; window < 25000 && !observed_real_game_over; ++window) {
        const auto step = skipped->Step(1); // FIREを維持して次のlifeを確実に開始する。
        const auto completed_score = *step->GetScalar("game_score");
        if (step->next_state.done && std::isfinite(completed_score)) {
            observed_real_game_over = true;
            REQUIRE(step->next_state.done);
            CHECK_FALSE(step->next_state.truncated);
            CHECK(completed_score == Catch::Approx(AuxFloat(step, "game_score")));
            break;
        }

        if (step->next_state.done) CHECK(std::isnan(completed_score));
        else CHECK(std::isfinite(completed_score));
        if (step->next_state.done) {
            CHECK_FALSE(step->next_state.truncated);
            const auto reset = skipped->Reset();
            CHECK(reset->state.episode_start);
            CHECK_FALSE(skipped->GetScalar("game_score").has_value());
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

    // fire_reset=false（v5 既定）ではlife-loss後にStepのNOOP 1回だけが進む。
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

TEST_CASE("Atari life-loss NOOP result retains final game values after automatic reset", "[atari][rom][prd097_atari]")
{
    const auto rom = FindRom("asterix");
    // 必須受入なのでROMやC++ ALEの条件が欠けた場合はskipしない。
    REQUIRE(rom.has_value());
    anet::ConfigData data;
    data.Set("AtariEnv.game", "asterix");
    data.Set("AtariEnv.rom_dir", rom->parent_path().string());
    data.Set("AtariEnv.frame_skip", 1);
    data.Set("AtariEnv.repeat_action_probability", 0.0);
    data.Set("AtariEnv.noop_max", 0);
    data.Set("AtariEnv.fire_reset", false);
    data.Set("AtariEnv.episodic_life", true);
    data.Set("AtariEnv.max_episode_frames", 0);
    class FixtureFactory final : public anet::rl::SingleDiscreteEnvFactory {
    public:
        std::string GetTargetEnvClassId() const override { return "AtariEnv"; }
        std::shared_ptr<anet::rl::SingleDiscreteEnv> CreateSingleEnv(
            const anet::ConfigData& config, const torch::Device& device, const std::string& name,
            std::optional<anet::seed_t>, anet::rl::RunMode mode, const std::string&) override
        {
            // fixtureのALE seedはlane seed変換を経ず1へ固定する。
            child = std::make_shared<anet::rl::env::AtariEnv>(
                anet::rl::env::AtariEnvConfig(config), device, name, 1, mode);
            return child;
        }
        std::shared_ptr<anet::rl::env::AtariEnv> child;
    };
    auto factory = std::make_shared<FixtureFactory>();
    anet::rl::VectorizedDiscreteBatchEnv env(data, factory, "asterix-noop-completion", 1, torch::Device(torch::kCPU), 1);
    env.Reset();
    anet::rl::env::AtariEnvTestAccess{}.PrepareLifeLossNoopCompletion(*factory->child);
    const auto step = env.Step(std::make_shared<anet::rl::BatchActionInfo>(torch::zeros({1}, torch::kInt64)));
    REQUIRE(step->next_state.done.item<bool>());
    CHECK_FALSE(step->next_state.truncated.item<bool>());
    CHECK(step->n_episode_end == 1);
    CHECK(step->GetScalar("lives", 0) == 1.0f);
    CHECK(step->GetAuxDataList(0).at(0).at("game_frames").item<int64_t>() == 301);
    REQUIRE(step->GetScalar("game_score", 0).has_value());
    REQUIRE(std::isfinite(*step->GetScalar("game_score", 0)));
    CHECK(step->GetScalar("game_len", 0) == 1.0f);
    CHECK(step->GetScalar("game_frames", 0) == 302.0f);
    CHECK(step->continue_state.episode_start.item<bool>());
    CHECK_FALSE(env.GetScalar("game_score", 0).has_value());
    const auto aux = step->GetAuxDataList(0).at(0);
    REQUIRE(aux.contains("rgb_frame"));
    const auto rgb = aux.at("rgb_frame").clone();
    CHECK_FALSE(torch::equal(rgb, *env.GetTensor("rgb_frame", 0)));
    env.Step(std::make_shared<anet::rl::BatchActionInfo>(torch::zeros({1}, torch::kInt64)));
    CHECK(step->GetScalar("game_frames", 0) == 302.0f);
    CHECK(torch::equal(rgb, step->GetAuxDataList(0).at(0).at("rgb_frame")));
}

TEST_CASE("Atari Step results answer provisional derived and numbered RAM scalars", "[atari][rom][prd097_atari_derived]")
{
    const auto rom = FindRom("pong");
    REQUIRE(rom.has_value());
    anet::ConfigData data;
    data.Set("AtariEnv.game", "pong");
    data.Set("AtariEnv.rom_dir", rom->parent_path().string());
    data.Set("AtariEnv.frame_skip", 1);
    data.Set("AtariEnv.max_episode_frames", 2);
    data.Set("AtariEnv.repeat_action_probability", 0.0f);
    data.Set("AtariEnv.ram_metric.[pong].[score_high]", "0x8D max_seen");
    data.Set("AtariEnv.ram_metric.[pong].metrics", "1:score_high");
    data.Set("AtariEnv.ram_metric.[qbert].[round]", "0xE3 inc_count");
    data.Set("AtariEnv.ram_metric.[qbert].metrics", "2:round");
    anet::rl::env::AtariEnvFactory factory;
    const auto env = factory.CreateSingleEnv(data, torch::Device(torch::kCPU), "derived-result", 1);
    env->Reset();
    const auto first = env->Step(0);
    REQUIRE_FALSE((first->next_state.done || first->next_state.truncated));
    const auto score = first->GetAuxData().at("game_score").item<float>();
    REQUIRE(first->GetScalar("hns57").has_value());
    CHECK(first->GetScalar("hns57") == anet::rl::env::HumanNormalizedScore(
        "pong", score, anet::rl::env::HnsBaseline::Dqn57));
    CHECK(first->GetScalar("hns49") == anet::rl::env::HumanNormalizedScore(
        "pong", score, anet::rl::env::HnsBaseline::Nature49));
    CHECK(first->GetScalar("game_score.ge.[-1000]") == 1.0f);
    CHECK(first->GetScalar("game_score.ge.[1000]") == 0.0f);
    CHECK(first->GetScalar("ram_metric.[1]") ==
        static_cast<float>(first->GetAuxData().at("ram_metric.score_high").item<int64_t>()));
    REQUIRE(first->GetScalar("ram_metric.[2]").has_value());
    CHECK(std::isnan(*first->GetScalar("ram_metric.[02]")));
    CHECK_FALSE(first->GetScalar("ram_metric.[3]").has_value());
    const auto last = env->Step(0);
    REQUIRE(last->next_state.truncated);
    CHECK(last->GetScalar("game_frames") == 2.0f);
    env->Reset();
    CHECK(first->GetScalar("game_frames") == 1.0f);
    CHECK(last->GetScalar("game_frames") == 2.0f);
    for (const auto& key : {"game_score", "hns57", "game_score.ge.[0]", "ram_metric.[1]"})
        CHECK_FALSE(env->GetScalar(key).has_value());
}

TEST_CASE("Atari factory reports the default resize mode with its Reset observation", "[atari][prd099][resize-config]")
{
    // 工場からResetまで通し、実効設定と観測契約を同時に確認する。
    const auto rom = FindRom("pong");
    REQUIRE(rom.has_value());
    anet::ConfigData data;
    data.Set("AtariEnv.game", "pong");
    data.Set("AtariEnv.rom_dir", rom->parent_path().string());
    anet::rl::env::AtariEnvFactory factory;
    const auto env = factory.CreateSingleEnv(data, torch::Device(torch::kCPU), "prd099-default", 123);
    const auto reset = env->Reset();
    CHECK(env->GetSpec().state_spec.ValidateObservation(reset->state.obs, false));
    const auto effective = env->GetConfigData();
    REQUIRE(effective.has_value());
    REQUIRE(effective->Has("AtariEnv.resize_mode"));
    CHECK(effective->Get("AtariEnv.resize_mode") == "adaptive_average");
}

TEST_CASE("GrayscaleResizer uses pixel overlap for fractional downsampling", "[atari][prd099][resize-area]")
{
    // 3→2の境界をまたぐ画素で、等重みのadaptive平均との違いを固定する。
    const std::vector<uint8_t> source{0, 0, 90, 0, 0, 90, 90, 90, 90};
    const anet::rl::env::atari::GrayscaleResizer resizer(3, 3, 2, "pixel_area");
    const auto output = resizer.Resize(source.data());
    CHECK(output.sizes().vec() == std::vector<int64_t>{1, 2, 2});
    CHECK(output.dtype() == torch::kUInt8);
    CHECK(output.is_contiguous());
    CHECK(output.flatten().to(torch::kInt64).equal(torch::tensor({0, 60, 60, 80}, torch::kInt64)));
}

TEST_CASE("Pixel area matches each fixed OpenCV reference image", "[atari][prd099][resize-golden]")
{
    // 同じ入力と最終uint8期待値を読み、画像ごとに許容を判定する。
    const auto directory = std::filesystem::path(ANET_SOURCE_DIR) / "core/anet-core/testdata/prd099";
    const auto read = [](const std::filesystem::path& path) {
        std::ifstream stream(path, std::ios::binary);
        REQUIRE(stream.good());
        return std::vector<uint8_t>(std::istreambuf_iterator<char>(stream), {});
    };
    const auto frames = read(directory / "frames_210x160.u8");
    const auto expected = read(directory / "expected_84x84.u8");
    REQUIRE(frames.size() == 24 * 210 * 160);
    REQUIRE(expected.size() == 24 * 84 * 84);
    const anet::rl::env::atari::GrayscaleResizer area(210, 160, 84, "pixel_area");
    const anet::rl::env::atari::GrayscaleResizer adaptive(210, 160, 84, "adaptive_average");
    int maximum_error = 0;
    int total_mismatches = 0;
    bool differs_from_adaptive = false;
    for (int frame = 0; frame < 24; ++frame) {
        CAPTURE(frame);
        const auto actual = area.Resize(frames.data() + frame * 210 * 160);
        const auto existing = adaptive.Resize(frames.data() + frame * 210 * 160);
        const auto* pixels = actual.data_ptr<uint8_t>();
        const auto* old_pixels = existing.data_ptr<uint8_t>();
        int mismatches = 0;
        int error = 0;
        for (int pixel = 0; pixel < 84 * 84; ++pixel) {
            const int reference = expected[frame * 84 * 84 + pixel];
            error = std::max(error, std::abs(static_cast<int>(pixels[pixel]) - reference));
            mismatches += pixels[pixel] != reference;
            differs_from_adaptive |= old_pixels[pixel] != reference;
        }
        INFO("mismatches=" << mismatches << ", max_error=" << error);
        CHECK(error <= 1);
        CHECK(mismatches * 1000 <= 84 * 84);
        maximum_error = std::max(maximum_error, error);
        total_mismatches += mismatches;
    }
    CHECK(differs_from_adaptive);
    std::cout << "PRD099 golden: frames=24 max_error=" << maximum_error
        << " mismatches=" << total_mismatches << "/" << 24 * 84 * 84 << std::endl;
}

TEST_CASE("Atari resize configuration rejects unknown modes at reading", "[atari][prd099][resize-invalid-mode]")
{
    // 共通値と評価個別値の両方で、利用者が直すキーと指定値を診断する。
    for (const auto& prefix : {std::string("AtariEnv"), std::string("run.eval.[eval_panel].env")}) {
        for (const auto& value : {std::string("unknown"), std::string("")}) {
            const std::string key = prefix + ".resize_mode";
            CAPTURE(key, value);
            anet::ConfigData data;
            data.Set("AtariEnv.game", "pong");
            data.Set(key, value);
            anet::test::LogCaptureGuard logs(wxLOG_Info);
            CHECK_THROWS_WITH(anet::rl::env::AtariEnvConfig(data, prefix == "AtariEnv" ? "" : prefix),
                Catch::Matchers::ContainsSubstring(key)
                && Catch::Matchers::ContainsSubstring("'" + value + "'")
                && Catch::Matchers::ContainsSubstring("adaptive_average")
                && Catch::Matchers::ContainsSubstring("pixel_area"));
        }
    }
}

TEST_CASE("Pixel area rejects invalid sizes with the responsible configuration key", "[atari][prd099][resize-invalid-size]")
{
    // ROMを読む前に、型・範囲と評価個別の指定箇所を診断する。
    anet::rl::env::AtariEnvFactory factory;
    for (const auto& prefix : {std::string("AtariEnv"), std::string("run.eval.[eval_panel].env")}) {
        for (const auto& value : {std::string("0"), std::string("-1"), std::string("161"), std::string("84.5")}) {
            const std::string key = prefix + ".screen_size";
            CAPTURE(key, value);
            anet::ConfigData data;
            data.Set("AtariEnv.game", "pong");
            data.Set("AtariEnv.resize_mode", "pixel_area");
            data.Set(key, value);
            anet::test::LogCaptureGuard logs(wxLOG_Info);
            CHECK_THROWS_WITH(factory.CreateSingleEnv(data, torch::Device(torch::kCPU), "invalid-size", 1,
                anet::rl::RunMode::Train, prefix == "AtariEnv" ? "" : prefix),
                Catch::Matchers::ContainsSubstring(key) && Catch::Matchers::ContainsSubstring(value));
        }
    }
}

TEST_CASE("Atari Run profile selects pixel area while evaluation keeps explicit overrides", "[atari][prd099][resize-profile][shipped_config]")
{
    // 実際のAtari.txtを解決し、Runの共通値と評価個別の責務を確認する。
    const auto directory = std::filesystem::path(ANET_SOURCE_DIR) / "apps/runner/config";
    const wxCmdLineEntryDesc description[] = {
        {wxCMD_LINE_PARAM, nullptr, nullptr, "key=value", wxCMD_LINE_VAL_STRING,
            wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE}, {wxCMD_LINE_NONE},
    };
    wxCmdLineParser cli(description, wxString::FromUTF8(
        "run.$=run.@v5_iqn_impala_x2>run.@pixel_area run.eval.[eval_panel].env.resize_mode=adaptive_average"));
    REQUIRE(cli.Parse(false) == 0);
    anet::ConfigManagerOptions options;
    options.config_search_dirs = {directory};
    options.overwrite_config_paths = {directory / "Atari.txt"};
    const auto data = anet::ConfigManager(directory / "_main.txt", &cli, options).GetConfigData();
    CHECK(data.Get("AtariEnv.resize_mode") == "pixel_area");
    const anet::rl::env::AtariEnvConfig common(data);
    CHECK(common.resize_mode == "pixel_area");
    const anet::rl::env::AtariEnvConfig panel(data, "run.eval.[eval_panel].env");
    CHECK(panel.resize_mode == "adaptive_average");
    CHECK(panel.GetScopedConfigData().Get("run.eval.[eval_panel].env.resize_mode") == "adaptive_average");
    const anet::rl::env::AtariEnvConfig inherited(data, "run.eval.[eval_target].env");
    CHECK(inherited.resize_mode == "pixel_area");
}

TEST_CASE("Pixel area preserves numeric rules and all supported sizes", "[atari][prd099][resize-boundary]")
{
    using anet::rl::env::atari::GrayscaleResizer;
    // 半整数と途中丸めを、二進で正確な重みになる入力で固定する。
    const std::vector<std::vector<uint8_t>> examples{{0, 1, 0, 1}, {1, 2, 1, 2}, {2, 3, 2, 3}, {0, 1, 0, 2}};
    const std::vector<int> rounded{0, 2, 2, 1};
    const GrayscaleResizer half(2, 2, 1, "pixel_area");
    for (size_t i = 0; i < examples.size(); ++i) {
        CAPTURE(i);
        CHECK(half.Resize(examples[i].data()).item<uint8_t>() == rounded[i]);
    }

    // 元画面の大きさでも半整数を保つ。浮動小数の逐次積算では128.5が129へ化けた。
    std::vector<uint8_t> native_half(210 * 160);
    const GrayscaleResizer native_average(210, 160, 1, "pixel_area");
    for (int base : {0, 1, 2, 126, 127, 128, 129, 254}) {
        CAPTURE(base);
        for (bool horizontal : {false, true}) {
            CAPTURE(horizontal);
            for (int y = 0; y < 210; ++y) {
                for (int x = 0; x < 160; ++x) {
                    native_half[y * 160 + x] = static_cast<uint8_t>(
                        base + (horizontal ? x >= 80 : y >= 105));
                }
            }
            CHECK(native_average.Resize(native_half.data()).item<uint8_t>() == base + base % 2);
        }
    }
    const GrayscaleResizer native_fractional(210, 160, 84, "pixel_area");
    for (int base : {126, 127, 128, 129}) {
        CAPTURE(base);
        std::fill(native_half.begin(), native_half.end(), static_cast<uint8_t>(base));
        // 左上の重みは42/200と38/200。6×42−4×38=100で平均がbase+0.5。
        native_half[0] = static_cast<uint8_t>(base + 6);
        native_half[1] = static_cast<uint8_t>(base - 4);
        CHECK(native_fractional.Resize(native_half.data()).data_ptr<uint8_t>()[0] == base + base % 2);
    }

    // 小さい非整数比では、区間を整数格子へ拡大した独立な面積計算と照合する。
    std::vector<uint8_t> source(5 * 7);
    for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<uint8_t>((17 * i) % 256);
    for (int size = 1; size <= 5; ++size) {
        const GrayscaleResizer area(5, 7, size, "pixel_area");
        const auto actual = area.Resize(source.data());
        const auto* pixels = actual.data_ptr<uint8_t>();
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                int numerator = 0;
                for (int sy = 0; sy < 5; ++sy) {
                    const int height = std::max(0, std::min((y + 1) * 5, (sy + 1) * size) - std::max(y * 5, sy * size));
                    for (int sx = 0; sx < 7; ++sx) {
                        const int width = std::max(0, std::min((x + 1) * 7, (sx + 1) * size) - std::max(x * 7, sx * size));
                        numerator += source[sy * 7 + sx] * height * width;
                    }
                }
                const int expected = numerator / 35 + ((numerator % 35) * 2 > 35);
                CAPTURE(size, y, x);
                CHECK(pixels[y * size + x] == expected);
            }
        }
    }

    // 定数保持が正規化を、入力範囲の保持が非負の重みを外から確かめる。
    std::vector<uint8_t> constant(210 * 160, 17);
    for (int size = 1; size <= 160; ++size) {
        CAPTURE(size);
        const GrayscaleResizer area(210, 160, size, "pixel_area");
        const auto image = area.Resize(constant.data());
        CHECK(image.sizes().vec() == std::vector<int64_t>{1, size, size});
        CHECK(image.min().item<uint8_t>() == 17);
        CHECK(image.max().item<uint8_t>() == 17);
    }
    for (int size : {1, 84, 160}) {
        const GrayscaleResizer area(210, 160, size, "pixel_area");
        for (size_t i = 0; i < constant.size(); ++i) constant[i] = i % 2 ? 203 : 17;
        const auto image = area.Resize(constant.data());
        CHECK(image.min().item<uint8_t>() >= 17);
        CHECK(image.max().item<uint8_t>() <= 203);
    }
    const GrayscaleResizer existing(210, 160, 161, "adaptive_average");
    CHECK(existing.Resize(constant.data()).sizes().vec() == std::vector<int64_t>{1, 161, 161});

    // 整数倍の箱平均は既存方式と一致し、入力は借用しても結果は独立して所有する。
    const std::vector<uint8_t> box{0, 1, 10, 11, 2, 2, 12, 12, 100, 101, 200, 201, 102, 103, 202, 204};
    const GrayscaleResizer area_box(4, 4, 2, "pixel_area");
    const GrayscaleResizer existing_box(4, 4, 2, "adaptive_average");
    CHECK(torch::equal(area_box.Resize(box.data()), existing_box.Resize(box.data())));
    auto borrowed = std::vector<uint8_t>{7, 7, 7, 7};
    const auto owned = half.Resize(borrowed.data());
    std::fill(borrowed.begin(), borrowed.end(), 0);
    CHECK(owned.item<uint8_t>() == 7);
}

TEST_CASE("Atari Reset and Step apply resize mode without changing the emulator", "[atari][prd099][resize-env]")
{
    // 同じseedと行動を与え、公開RGBは一致したまま観測だけが方式で変わることを確認する。
    const auto rom = FindRom("qbert");
    REQUIRE(rom.has_value());
    anet::ConfigData data;
    data.Set("AtariEnv.game", "qbert");
    data.Set("AtariEnv.rom_dir", rom->parent_path().string());
    data.Set("AtariEnv.resize_mode", "pixel_area");
    data.Set("AtariEnv.frame_skip", 1);
    data.Set("AtariEnv.max_pool", false);
    data.Set("AtariEnv.noop_max", 30);
    data.Set("run.eval.[eval_panel].env.resize_mode", "adaptive_average");
    anet::rl::env::AtariEnvFactory factory;
    const auto area = factory.CreateSingleEnv(data, torch::Device(torch::kCPU), "prd099-area", 13);
    const auto panel = factory.CreateSingleEnv(data, torch::Device(torch::kCPU), "prd099-panel", 13,
        anet::rl::RunMode::Eval1, "run.eval.[eval_panel].env");
    CHECK(area->GetConfigData()->Get("AtariEnv.resize_mode") == "pixel_area");
    CHECK(panel->GetConfigData()->Get("run.eval.[eval_panel].env.resize_mode") == "adaptive_average");
    const auto rgb_equal = [&] {
        const auto a = area->GetTensor("rgb_frame");
        const auto b = panel->GetTensor("rgb_frame");
        REQUIRE(a.has_value());
        REQUIRE(b.has_value());
        CHECK(torch::equal(*a, *b));
    };
    bool reset_differs = false;
    for (int repeat = 0; repeat < 4; ++repeat) {
        const auto a = area->Reset();
        const auto b = panel->Reset();
        rgb_equal();
        reset_differs |= !torch::equal(a->state.obs.Get(anet::rl::ObsKeys::kGrid).value(),
            b->state.obs.Get(anet::rl::ObsKeys::kGrid).value());
    }
    CHECK(reset_differs);
    bool step_differs = false;
    for (int step = 0; step < 32; ++step) {
        const auto a = area->Step(0);
        const auto b = panel->Step(0);
        rgb_equal();
        CHECK(a->next_state.done == b->next_state.done);
        CHECK(a->next_state.truncated == b->next_state.truncated);
        step_differs |= !torch::equal(a->next_state.obs.Get(anet::rl::ObsKeys::kGrid).value(),
            b->next_state.obs.Get(anet::rl::ObsKeys::kGrid).value());
        if (a->next_state.done || a->next_state.truncated) {
            area->Reset();
            panel->Reset();
        }
    }
    CHECK(step_differs);
}

TEST_CASE("Pixel area optimization preserves the fixed integer reference", "[.][prd099-perf-exact]")
{
    using anet::rl::env::atari::GrayscaleResizer;
    // 各サイズの参照と実装を再利用し、呼び出しを繰り返したときの作業領域も検査する。
    std::vector<PixelAreaReferenceResizer> reference;
    std::vector<GrayscaleResizer> actual;
    reference.reserve(160);
    actual.reserve(160);
    for (int size = 1; size <= 160; ++size) {
        reference.emplace_back(210, 160, size, "pixel_area");
        actual.emplace_back(210, 160, size, "pixel_area");
    }
    int comparisons = 0;
    int64_t pixels = 0;
    const auto compare = [&](const uint8_t* input) {
        for (int size = 1; size <= 160; ++size) {
            CAPTURE(size);
            const auto expected = reference[size - 1].Resize(input);
            const auto output = actual[size - 1].Resize(input);
            REQUIRE(torch::equal(expected, output));
            ++comparisons;
            pixels += size * size;
        }
    };

    // 固定24枚はS=84だけでなく、対応する全160サイズで比較する。
    const auto path = std::filesystem::path(ANET_SOURCE_DIR) / "core/anet-core/testdata/prd099/frames_210x160.u8";
    std::ifstream stream(path, std::ios::binary);
    REQUIRE(stream.good());
    const std::vector<uint8_t> frames(std::istreambuf_iterator<char>(stream), {});
    REQUIRE(frames.size() == 24 * 210 * 160);
    for (int image = 0; image < 24; ++image) {
        CAPTURE(image);
        compare(frames.data() + image * 210 * 160);
    }

    // 乱数1000枚とS=1..160の直積をすべて比較し、特定の縮小率だけの一致にしない。
    std::mt19937 random(20261024);
    std::vector<uint8_t> input(210 * 160);
    for (int image = 0; image < 1000; ++image) {
        CAPTURE(image);
        for (auto& pixel : input) pixel = static_cast<uint8_t>(random());
        compare(input.data());
    }
    // 定数・縦縞・横縞・市松模様も全サイズで確認する。
    for (int pattern = 0; pattern < 5; ++pattern) {
        CAPTURE(pattern);
        for (int y = 0; y < 210; ++y) {
            for (int x = 0; x < 160; ++x) {
                const bool high = pattern == 1 || (pattern == 2 && x % 2)
                    || (pattern == 3 && y % 2) || (pattern == 4 && (x + y) % 2);
                input[y * 160 + x] = high ? 255 : 0;
            }
        }
        compare(input.data());
    }
    // 32bitに収まらない任意寸法も、従来の64bit積算で維持する。
    const std::vector<uint8_t> large(3000 * 3000, 255);
    const PixelAreaReferenceResizer wide_reference(3000, 3000, 1, "pixel_area");
    const GrayscaleResizer wide_actual(3000, 3000, 1, "pixel_area");
    REQUIRE(torch::equal(wide_reference.Resize(large.data()), wide_actual.Resize(large.data())));
    CHECK(wide_actual.Resize(large.data()).item<uint8_t>() == 255);
    std::cout << "PRD099 exact equivalence comparisons=" << comparisons << " pixels=" << pixels
        << " mismatches=0 random_images=1000 sizes=1..160 wide_case=passed" << std::endl;
}

TEST_CASE("PRD099 resize prototype timings", "[.][prd099-resize-probe]")
{
    // 同じ実行体の中で高速化前の参照・現在方式・既定を交互に測る。正式なEnv Step受入ではない。
    const auto path = std::filesystem::path(ANET_SOURCE_DIR) / "core/anet-core/testdata/prd099/frames_210x160.u8";
    std::ifstream stream(path, std::ios::binary);
    REQUIRE(stream.good());
    const std::vector<uint8_t> frames(std::istreambuf_iterator<char>(stream), {});
    REQUIRE(frames.size() == 24 * 210 * 160);
    const PixelAreaReferenceResizer baseline(210, 160, 84, "pixel_area");
    const anet::rl::env::atari::GrayscaleResizer current(210, 160, 84, "pixel_area");
    const anet::rl::env::atari::GrayscaleResizer adaptive(210, 160, 84, "adaptive_average");
    int checksum = 0;
    const auto time = [&](const auto& resizer, int count) {
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < count; ++i) {
            const auto output = resizer.Resize(frames.data() + (i % 24) * 210 * 160);
            checksum += output.template data_ptr<uint8_t>()[i % (84 * 84)];
        }
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / count;
    };
    time(baseline, 50);
    time(current, 50);
    time(adaptive, 50);
    std::array<std::vector<double>, 3> samples;
    const std::array<const char*, 3> names{"baseline", "current", "adaptive"};
    // 5巡で開始方式を回し、各方式2000回の平均を1試行として記録する。
    for (int round = 0; round < 5; ++round) {
        for (int offset = 0; offset < 3; ++offset) {
            const int variant = (round + offset) % 3;
            double elapsed = 0;
            if (variant == 0) elapsed = time(baseline, 2000);
            else if (variant == 1) elapsed = time(current, 2000);
            else elapsed = time(adaptive, 2000);
            CHECK(elapsed > 0);
            samples[variant].push_back(elapsed);
            std::cout << "PRD099 resize probe round=" << round << " variant=" << names[variant]
                << " us=" << elapsed << std::endl;
        }
    }
    for (auto& values : samples) std::sort(values.begin(), values.end());
    std::cout << "PRD099 resize probe median_baseline_us=" << samples[0][2]
        << " median_current_us=" << samples[1][2] << " median_adaptive_us=" << samples[2][2]
        << " speed_ratio=" << samples[0][2] / samples[1][2] << " checksum=" << checksum
        << " formal_env_acceptance=not_measured" << std::endl;
}
