// DropMergeEnv_test.cpp

#include "DropMergeEnv.hpp"

#include <cmath>
#include <clocale>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "anet/catch_test.hpp"
#include "anet/env.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/test_util.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

constexpr const char* kVectorKey = anet::rl::ObsKeys::kVector;
constexpr const char* kGridKey = anet::rl::ObsKeys::kGrid;

void SetupUtf8Console()
{
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::setlocale(LC_CTYPE, ".UTF-8");
}

class NoopMetricsBackend final : public anet::IBackend {
public:
    void Open(const std::filesystem::path&, const std::string&) override {}
    void WriteJsonl(const anet::json&) override {}
    void Flush() override {}
};

class ScopedNoopMetricsLogger final {
public:
    ScopedNoopMetricsLogger()
    {
        anet::MetricsLogger::Reset();
        anet::MetricsLoggerConfig config;
        config.run_name_tmpl = "dropmerge_env_test";
        anet::MetricsLogger::Init(std::make_unique<NoopMetricsBackend>(), config, "C:/tmp");
    }

    ~ScopedNoopMetricsLogger()
    {
        anet::MetricsLogger::Reset();
    }
};

torch::Tensor VectorObs(const anet::rl::SingleState& state)
{
    return state.obs.At(kVectorKey);
}

torch::Tensor GridObs(const anet::rl::SingleState& state)
{
    return state.obs.At(kGridKey);
}

void RequireFlatApprox(const torch::Tensor& tensor, const std::vector<float>& expected)
{
    const auto flat = tensor.detach().cpu().to(torch::kFloat32).reshape({ -1 }).contiguous();
    REQUIRE(flat.numel() == static_cast<int64_t>(expected.size()));
    for (int64_t i = 0; i < flat.numel(); ++i) {
        CHECK(flat[i].item<float>() == Catch::Approx(expected[static_cast<size_t>(i)]).margin(1.0e-6f));
    }
}

anet::rl::env::drop_merge::DropMergeEnvConfig MakeBlockedBoardConfig()
{
    anet::rl::env::drop_merge::DropMergeEnvConfig config;
    config.seed_mode = "fixed";
    config.action_mode = "direct_noop";
    config.drop_divisions = 1;
    config.box_width = 1.0f;
    config.box_height = 0.9f;
    config.fruit_radii.assign(anet::rl::env::drop_merge::kFruitTypeCount, 0.4f);
    config.drop_probs = { 1.0f };
    config.drop_noise = 0.0f;
    config.spin_noise = 0.0f;
    config.restitution = 0.0f;
    config.box_restitution = 0.0f;
    config.use_instant_drop = true;
    config.game_over_grace_step = 1000;
    return config;
}

int CountOccurrences(const std::string& value, const std::string& needle)
{
    int count = 0;
    std::string::size_type position = 0;
    while ((position = value.find(needle, position)) != std::string::npos) {
        ++count;
        position += needle.size();
    }
    return count;
}

} // namespace

TEST_CASE("DropMergeEnv appends the selected previous action trio", "[dropmerge][prev_action]")
{
    anet::rl::env::drop_merge::DropMergeEnvConfig default_config;
    CHECK_FALSE(default_config.ToJson().at("obs_include_prev_action").get<bool>());
    CHECK_FALSE(default_config.ToJson().at("obs_prev_drop_marker").get<bool>());
    CHECK(default_config.ToConfigString().find(
        "DropMergeEnv.obs_include_prev_action = false") != std::string::npos);
    CHECK(default_config.ToConfigString().find(
        "DropMergeEnv.obs_prev_drop_marker = false") != std::string::npos);

    anet::ConfigData config_data;
    config_data.Set("DropMergeEnv.obs_include_prev_action", true);
    config_data.Set("DropMergeEnv.obs_prev_drop_marker", false);
    config_data.Set("DropMergeEnv.action_mode", "direct_noop");
    config_data.Set("DropMergeEnv.drop_divisions", 4);
    const anet::rl::env::drop_merge::DropMergeEnvConfig config(config_data);
    CHECK(config.obs_include_prev_action);
    CHECK_FALSE(config.obs_prev_drop_marker);

    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-prev-action[0]", 123);
    const auto spec = env->GetSpec();
    const auto& vector_spec = spec.state_spec.obs_spec.at(kVectorKey);
    CHECK(vector_spec.shape == std::vector<int64_t>{ 7 });
    REQUIRE(vector_spec.labels.size() == 7);
    CHECK(vector_spec.labels[4] == "prev_valid");
    CHECK(vector_spec.labels[5] == "prev_noop");
    CHECK(vector_spec.labels[6] == "prev_drop_x");
    CHECK(vector_spec.min_values[4] == Catch::Approx(0.0));
    CHECK(vector_spec.min_values[5] == Catch::Approx(0.0));
    CHECK(vector_spec.min_values[6] == Catch::Approx(-1.0));
    CHECK(vector_spec.max_values[4] == Catch::Approx(1.0));
    CHECK(vector_spec.max_values[5] == Catch::Approx(1.0));
    CHECK(vector_spec.max_values[6] == Catch::Approx(1.0));

    const auto reset = env->Reset();
    CHECK(spec.state_spec.ValidateObservation(reset->state.obs, /*is_batched=*/false));
    RequireFlatApprox(VectorObs(reset->state).slice(/*dim=*/0, 4, 7), { 0.0f, 0.0f, 0.0f });

    const auto noop = env->Step(0);
    CHECK(spec.state_spec.ValidateObservation(noop->next_state.obs, /*is_batched=*/false));
    RequireFlatApprox(VectorObs(noop->next_state).slice(/*dim=*/0, 4, 7), { 1.0f, 1.0f, 0.0f });

    const auto drop = env->Step(1);
    CHECK(spec.state_spec.ValidateObservation(drop->next_state.obs, /*is_batched=*/false));
    RequireFlatApprox(VectorObs(drop->next_state).slice(/*dim=*/0, 4, 7), { 1.0f, 0.0f, -0.75f });
}

TEST_CASE("DropMergeEnv appends the previous action trio after the timeout ratio", "[dropmerge][prev_action][timeout]")
{
    anet::rl::env::drop_merge::DropMergeEnvConfig config;
    config.action_mode = "direct_noop";
    config.drop_divisions = 4;
    config.use_no_drop_timeout_gameover = true;
    config.obs_include_prev_action = true;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-prev-timeout[0]", 123);

    const auto spec = env->GetSpec();
    const auto& vector_spec = spec.state_spec.obs_spec.at(kVectorKey);
    CHECK(vector_spec.shape == std::vector<int64_t>{ 8 });
    REQUIRE(vector_spec.labels.size() == 8);
    CHECK(vector_spec.labels[4] == "no_drop_timeout_ratio");
    CHECK(vector_spec.labels[5] == "prev_valid");
    CHECK(vector_spec.labels[6] == "prev_noop");
    CHECK(vector_spec.labels[7] == "prev_drop_x");

    const auto reset = env->Reset();
    CHECK(spec.state_spec.ValidateObservation(
        reset->state.obs, /*is_batched=*/false));
    RequireFlatApprox(
        VectorObs(reset->state).slice(/*dim=*/0, 5, 8),
        { 0.0f, 0.0f, 0.0f });

    const auto drop = env->Step(1);
    CHECK(spec.state_spec.ValidateObservation(
        drop->next_state.obs, /*is_batched=*/false));
    RequireFlatApprox(
        VectorObs(drop->next_state).slice(/*dim=*/0, 5, 8),
        { 1.0f, 0.0f, -0.75f });
}

TEST_CASE("DropMergeEnv previous action observations require a direct action mode", "[dropmerge][prev_action][config]")
{
    anet::rl::env::drop_merge::DropMergeEnvConfig direct_config;
    direct_config.action_mode = "direct";
    direct_config.drop_divisions = 4;
    direct_config.obs_include_prev_action = true;
    auto direct_env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        direct_config, torch::Device(torch::kCPU), "dropmerge-prev-direct[0]", 123);
    direct_env->Reset();
    const auto direct_drop = direct_env->Step(3);
    RequireFlatApprox(
        VectorObs(direct_drop->next_state).slice(/*dim=*/0, 4, 7),
        { 1.0f, 0.0f, 0.75f });

    for (const std::string& action_mode : { "move", "move_fast" }) {
        for (const bool use_marker : { false, true }) {
            anet::rl::env::drop_merge::DropMergeEnvConfig invalid_config;
            invalid_config.action_mode = action_mode;
            invalid_config.obs_include_prev_action = !use_marker;
            invalid_config.obs_prev_drop_marker = use_marker;

            anet::test::LogCaptureGuard logs(wxLOG_Info);
            CHECK_THROWS_WITH(
                std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
                    invalid_config, torch::Device(torch::kCPU), "dropmerge-prev-invalid[0]", 123),
                Catch::Matchers::ContainsSubstring(
                    "obs_include_prev_action / obs_prev_drop_marker require "
                    "action_mode=direct or direct_noop")
                && Catch::Matchers::ContainsSubstring("actual=" + action_mode));
            logs.Flush();
        }
    }
}

TEST_CASE("DropMergeEnv draws the selected DROP column marker", "[dropmerge][prev_action][marker]")
{
    anet::rl::env::drop_merge::DropMergeEnvConfig config;
    config.action_mode = "direct_noop";
    config.drop_divisions = 2;
    config.grid_rows = 4;
    config.grid_cols = 4;
    config.use_instant_drop = true;
    config.obs_prev_drop_marker = true;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-prev-marker[0]", 123);

    const auto spec = env->GetSpec();
    const auto& grid_spec = spec.state_spec.obs_spec.at(kGridKey);
    CHECK(grid_spec.shape == std::vector<int64_t>{ 1, 4, 4 });
    CHECK(grid_spec.num_classes == anet::rl::env::drop_merge::kFruitTypeCount + 2);
    CHECK(grid_spec.max_values == std::vector<double>{
        static_cast<double>(anet::rl::env::drop_merge::kFruitTypeCount + 1) });

    const int marker_class = anet::rl::env::drop_merge::kFruitTypeCount + 1;
    const auto reset = env->Reset();
    CHECK(spec.state_spec.ValidateObservation(reset->state.obs, /*is_batched=*/false));
    CHECK(GridObs(reset->state).eq(marker_class).sum().item<int64_t>() == 0);

    // 2 分割の DROP 命令列 1 を、4 列 grid の対応する top-row 列 3 へ写像する。
    const auto drop = env->Step(2);
    CHECK(spec.state_spec.ValidateObservation(drop->next_state.obs, /*is_batched=*/false));
    CHECK(GridObs(drop->next_state).index({ 0, 3, 3 }).item<int>() == marker_class);
    CHECK(GridObs(drop->next_state).eq(marker_class).sum().item<int64_t>() == 1);

    const auto noop = env->Step(0);
    CHECK(spec.state_spec.ValidateObservation(noop->next_state.obs, /*is_batched=*/false));
    CHECK(GridObs(noop->next_state).eq(marker_class).sum().item<int64_t>() == 0);

    const auto reset_again = env->Reset();
    CHECK(spec.state_spec.ValidateObservation(reset_again->state.obs, /*is_batched=*/false));
    CHECK(GridObs(reset_again->state).eq(marker_class).sum().item<int64_t>() == 0);
}

TEST_CASE("DropMergeEnv previous action observations preserve the default state contract", "[dropmerge][prev_action][compat]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::rl::env::drop_merge::DropMergeEnvConfig default_config;
    default_config.action_mode = "direct_noop";
    default_config.drop_divisions = 4;
    default_config.use_instant_drop = true;
    default_config.max_step = 3;

    auto enabled_config = default_config;
    enabled_config.obs_include_prev_action = true;
    enabled_config.obs_prev_drop_marker = true;

    auto default_env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        default_config, torch::Device(torch::kCPU), "dropmerge-prev-default[0]", 123);
    auto enabled_env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        enabled_config, torch::Device(torch::kCPU), "dropmerge-prev-enabled[0]", 123);

    const auto default_spec = default_env->GetSpec();
    const auto enabled_spec = enabled_env->GetSpec();
    CHECK(default_spec.state_spec.obs_spec.at(kVectorKey).shape == std::vector<int64_t>{ 4 });
    CHECK(enabled_spec.state_spec.obs_spec.at(kVectorKey).shape == std::vector<int64_t>{ 7 });
    CHECK(default_spec.state_spec.obs_spec.at(kGridKey).shape ==
        enabled_spec.state_spec.obs_spec.at(kGridKey).shape);
    CHECK(default_spec.state_spec.obs_spec.at(kGridKey).num_classes ==
        enabled_spec.state_spec.obs_spec.at(kGridKey).num_classes);

    const auto default_reset = default_env->Reset();
    const auto enabled_reset = enabled_env->Reset();
    CHECK(torch::equal(
        VectorObs(default_reset->state),
        VectorObs(enabled_reset->state).slice(/*dim=*/0, 0, 4)));
    CHECK(torch::equal(GridObs(default_reset->state), GridObs(enabled_reset->state)));

    const int marker_class = anet::rl::env::drop_merge::kFruitTypeCount + 1;
    anet::test::LogCaptureGuard logs(wxLOG_Info);
    for (const int64_t action : { 2, 0, 3 }) {
        const auto default_step = default_env->Step(action);
        const auto enabled_step = enabled_env->Step(action);
        CHECK(default_spec.state_spec.ValidateObservation(
            default_step->next_state.obs, /*is_batched=*/false));
        CHECK(enabled_spec.state_spec.ValidateObservation(
            enabled_step->next_state.obs, /*is_batched=*/false));
        CHECK(torch::equal(
            VectorObs(default_step->next_state),
            VectorObs(enabled_step->next_state).slice(/*dim=*/0, 0, 4)));

        // marker class だけを除けば、同 seed・同 action の盤面は一致する。
        auto enabled_grid_without_marker = GridObs(enabled_step->next_state).clone();
        enabled_grid_without_marker.masked_fill_(
            enabled_grid_without_marker.eq(marker_class), 0);
        CHECK(torch::equal(
            GridObs(default_step->next_state), enabled_grid_without_marker));
        CHECK(default_step->reward == enabled_step->reward);
        CHECK(default_step->next_state.done == enabled_step->next_state.done);
        CHECK(default_step->next_state.truncated ==
            enabled_step->next_state.truncated);
    }
    logs.Flush();
}

TEST_CASE("DropMergeEnv previous action observations work through batch env prefixes", "[dropmerge][prev_action][batch]")
{
    anet::ConfigData config_data;
    config_data.Set("DropMergeEnv.action_mode", "direct_noop");
    config_data.Set("DropMergeEnv.drop_divisions", 4);
    config_data.Set("DropMergeEnv.use_instant_drop", true);
    config_data.Set("DropMergeEnv.obs_include_prev_action", true);
    config_data.Set("DropMergeEnv.obs_prev_drop_marker", true);

    const std::vector<std::string> prefixes = {
        "train.env",
        "train.eval.[eval1].env",
        "train.eval.[eval2].env",
    };
    const std::vector<anet::rl::RunMode> run_modes = {
        anet::rl::RunMode::Train,
        anet::rl::RunMode::Eval1,
        anet::rl::RunMode::Eval2,
    };
    auto factory =
        std::make_shared<anet::rl::env::drop_merge::DropMergeEnvFactory>();

    for (size_t i = 0; i < prefixes.size(); ++i) {
        anet::rl::VectorizedDiscreteBatchEnv env(
            config_data,
            factory,
            "dropmerge-prev-batch-" + std::to_string(i),
            /*num_envs=*/1,
            torch::Device(torch::kCPU),
            /*seed=*/123,
            run_modes[i],
            prefixes[i]);

        const auto spec = env.GetSpec();
        CHECK(spec.state_spec.obs_spec.at(kVectorKey).shape ==
            std::vector<int64_t>{ 7 });
        const auto reset = env.Reset();
        CHECK(spec.state_spec.ValidateObservation(
            reset->state.obs, /*is_batched=*/true));
        CHECK(reset->state.obs.At(kVectorKey).sizes().vec() ==
            std::vector<int64_t>{ 1, 7 });

        auto action_info = std::make_shared<anet::rl::BatchActionInfo>(
            torch::tensor(
                { 2 },
                torch::TensorOptions().dtype(torch::kInt64)));
        const auto step = env.Step(action_info);
        CHECK(spec.state_spec.ValidateObservation(
            step->next_state.obs, /*is_batched=*/true));
        RequireFlatApprox(
            step->next_state.obs.At(kVectorKey)[0].slice(
                /*dim=*/0, 4, 7),
            { 1.0f, 0.0f, -0.25f });
    }
}

TEST_CASE("DropMergeEnv reports the successive DROP column ratio", "[dropmerge][prev_action][ratio]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::rl::env::drop_merge::DropMergeEnvConfig config;
    config.action_mode = "direct_noop";
    config.drop_divisions = 6;
    config.use_instant_drop = true;
    config.max_step = 5;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-prev-ratio[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    for (const int64_t action : { 4, 4, 6, 6 }) {
        const auto step = env->Step(action);
        REQUIRE_FALSE(step->next_state.done);
        REQUIRE_FALSE(step->next_state.truncated);
        REQUIRE(env->GetScalar("ep_same_drop_col_ratio").has_value());
        CHECK(std::isnan(*env->GetScalar("ep_same_drop_col_ratio")));
    }
    const auto terminal = env->Step(6);
    REQUIRE(terminal->next_state.truncated);
    REQUIRE(env->GetScalar("ep_same_drop_col_ratio").has_value());
    CHECK(*env->GetScalar("ep_same_drop_col_ratio") ==
        Catch::Approx(3.0f / 4.0f));
    logs.Flush();
}

TEST_CASE("DropMergeEnv handles sparse DROP commands in the column ratio", "[dropmerge][prev_action][ratio]")
{
    ScopedNoopMetricsLogger metrics_logger;

    SECTION("fewer than two DROP commands reports zero")
    {
        anet::rl::env::drop_merge::DropMergeEnvConfig config;
        config.action_mode = "direct";
        config.drop_divisions = 4;
        config.use_instant_drop = true;
        config.max_step = 1;
        auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
            config, torch::Device(torch::kCPU), "dropmerge-prev-ratio-short[0]", 123);
        env->Reset();

        anet::test::LogCaptureGuard logs(wxLOG_Info);
        const auto terminal = env->Step(2);
        REQUIRE(terminal->next_state.truncated);
        REQUIRE(env->GetScalar("ep_same_drop_col_ratio").has_value());
        CHECK(*env->GetScalar("ep_same_drop_col_ratio") == Catch::Approx(0.0f));
        logs.Flush();
    }

    SECTION("NOOP preserves the previous DROP column")
    {
        anet::rl::env::drop_merge::DropMergeEnvConfig config;
        config.action_mode = "direct_noop";
        config.drop_divisions = 4;
        config.use_instant_drop = true;
        config.max_step = 3;
        auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
            config, torch::Device(torch::kCPU), "dropmerge-prev-ratio-noop[0]", 123);
        env->Reset();

        anet::test::LogCaptureGuard logs(wxLOG_Info);
        env->Step(2);
        env->Step(0);
        const auto terminal = env->Step(2);
        REQUIRE(terminal->next_state.truncated);
        REQUIRE(env->GetScalar("ep_same_drop_col_ratio").has_value());
        CHECK(*env->GetScalar("ep_same_drop_col_ratio") == Catch::Approx(1.0f));
        logs.Flush();
    }
}

TEST_CASE("DropMergeEnv column ratio is NaN for move action modes", "[dropmerge][prev_action][ratio]")
{
    ScopedNoopMetricsLogger metrics_logger;
    for (const std::string& action_mode : { "move", "move_fast" }) {
        anet::rl::env::drop_merge::DropMergeEnvConfig config;
        config.action_mode = action_mode;
        config.max_step = 1;
        auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
            config, torch::Device(torch::kCPU),
            "dropmerge-prev-ratio-" + action_mode + "[0]", 123);
        env->Reset();

        anet::test::LogCaptureGuard logs(wxLOG_Info);
        const auto terminal = env->Step(anet::rl::env::drop_merge::kActionNoop);
        REQUIRE(terminal->next_state.truncated);
        REQUIRE(env->GetScalar("ep_same_drop_col_ratio").has_value());
        CHECK(std::isnan(*env->GetScalar("ep_same_drop_col_ratio")));
        logs.Flush();
    }
}

TEST_CASE("DropMergeEnv observes DROP commands rejected while busy", "[dropmerge][prev_action][busy]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::rl::env::drop_merge::DropMergeEnvConfig config;
    config.action_mode = "direct_noop";
    config.drop_divisions = 4;
    config.grid_rows = 4;
    config.grid_cols = 4;
    config.use_instant_drop = false;
    config.reload_min_steps = 20;
    config.max_step = 3;
    config.obs_include_prev_action = true;
    config.obs_prev_drop_marker = true;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-prev-busy[0]", 123);
    const auto spec = env->GetSpec();
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto first = env->Step(1);
    REQUIRE_FALSE(first->next_state.truncated);
    CHECK(VectorObs(first->next_state)[3].item<float>() == Catch::Approx(1.0f));

    const auto rejected = env->Step(4);
    REQUIRE_FALSE(rejected->next_state.truncated);
    CHECK(VectorObs(rejected->next_state)[3].item<float>() == Catch::Approx(1.0f));

    const auto terminal = env->Step(4);
    REQUIRE(terminal->next_state.truncated);
    CHECK(spec.state_spec.ValidateObservation(
        terminal->next_state.obs, /*is_batched=*/false));
    RequireFlatApprox(
        VectorObs(terminal->next_state).slice(/*dim=*/0, 4, 7),
        { 1.0f, 0.0f, 0.75f });

    const int marker_class = anet::rl::env::drop_merge::kFruitTypeCount + 1;
    CHECK(GridObs(terminal->next_state).index({ 0, 3, 3 }).item<int>() ==
        marker_class);
    REQUIRE(env->GetScalar("ep_same_drop_col_ratio").has_value());
    CHECK(*env->GetScalar("ep_same_drop_col_ratio") ==
        Catch::Approx(1.0f / 2.0f));
    REQUIRE(env->GetScalar("ep_end_fruit_count").has_value());
    CHECK(*env->GetScalar("ep_end_fruit_count") == Catch::Approx(1.0f));
    logs.Flush();
}

TEST_CASE("DropMergeEnv prefixes maximum-step log with its name once", "[dropmerge][env_name]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::rl::env::drop_merge::DropMergeEnvConfig config;
    config.max_step = 1;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-log[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    env->Step(anet::rl::env::drop_merge::kActionNoop);
    logs.Flush();

    int matching_records = 0;
    for (const auto& record : logs.Records()) {
        if (record.message.find("Episode truncated. Maximum step count exceeded.") == std::string::npos) {
            continue;
        }
        ++matching_records;
        CHECK(record.level == wxLOG_Info);
        CHECK(record.message.find("dropmerge-log[0]: Episode truncated. Maximum step count exceeded.") == 0);
        CHECK(CountOccurrences(record.message, "dropmerge-log[0]: ") == 1);
    }
    CHECK(matching_records == 1);
}

TEST_CASE("DropMergeEnv reports DROP selected on a NoLegal candidate", "[dropmerge][no_legal_candidate]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeBlockedBoardConfig();
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-candidate[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto first_drop = env->Step(1);
    REQUIRE_FALSE(first_drop->next_state.done);
    REQUIRE_FALSE(first_drop->next_state.truncated);
    REQUIRE(env->GetScalar("blocked_drop_on_candidate").has_value());
    CHECK(std::isnan(*env->GetScalar("blocked_drop_on_candidate")));
    REQUIRE(env->GetScalar("no_drop_timeout_on_candidate").has_value());
    CHECK(std::isnan(*env->GetScalar("no_drop_timeout_on_candidate")));
    REQUIRE(env->GetScalar("ep_mean_blocked_frames").has_value());
    CHECK(std::isnan(*env->GetScalar("ep_mean_blocked_frames")));
    REQUIRE(env->GetScalar("ep_max_blocked_frames").has_value());
    CHECK(std::isnan(*env->GetScalar("ep_max_blocked_frames")));

    const auto blocked_drop = env->Step(1);
    CHECK(blocked_drop->next_state.done);
    CHECK_FALSE(blocked_drop->next_state.truncated);
    REQUIRE(env->GetScalar("term_reason_spawn_blocked").has_value());
    CHECK(*env->GetScalar("term_reason_spawn_blocked") == 1.0f);
    REQUIRE(env->GetScalar("blocked_drop_on_candidate").has_value());
    CHECK(*env->GetScalar("blocked_drop_on_candidate") == 1.0f);
    logs.Flush();
}

TEST_CASE("DropMergeEnv handles a fruit wider than its placement range", "[dropmerge][no_legal_candidate]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeBlockedBoardConfig();
    config.box_width = 0.5f;
    config.settle_velocity_threshold = 100.0f;
    config.settle_angular_threshold = 100.0f;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-no-placement-range[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto noop = env->Step(0);
    CHECK(noop->next_state.done);
    CHECK_FALSE(noop->next_state.truncated);
    REQUIRE(env->GetScalar("term_reason_no_legal_drop").has_value());
    CHECK(*env->GetScalar("term_reason_no_legal_drop") == 1.0f);
    logs.Flush();
}

TEST_CASE("blocked intervals report whether the entire range is covered", "[dropmerge][blocked_intervals]")
{
    const auto covers = [](std::vector<std::pair<float, float>> intervals) {
        return anet::rl::env::drop_merge::DoBlockedIntervalsCoverRange(intervals, -1.0f, 1.0f);
    };

    CHECK_FALSE(covers({}));
    CHECK(covers({ { -1.0f, 1.0f } }));
    CHECK_FALSE(covers({ { -0.9f, 1.0f } }));
    CHECK_FALSE(covers({ { -1.0f, 0.9f } }));
    CHECK_FALSE(covers({ { -1.0f, -0.1f }, { 0.1f, 1.0f } }));
    CHECK(covers({ { 0.0f, 1.0f }, { -1.0f, 0.5f } }));
    CHECK(covers({ { -1.0f, 0.0f }, { 0.0f, 1.0f } }));
}

TEST_CASE("DropMergeEnv keeps settled NoLegal NOOP termination unchanged", "[dropmerge][no_legal_candidate]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeBlockedBoardConfig();
    config.settle_velocity_threshold = 100.0f;
    config.settle_angular_threshold = 100.0f;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-settled[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto first_drop = env->Step(1);
    REQUIRE_FALSE(first_drop->next_state.done);

    const auto noop = env->Step(0);
    CHECK(noop->next_state.done);
    CHECK_FALSE(noop->next_state.truncated);
    REQUIRE(env->GetScalar("term_reason_no_legal_drop").has_value());
    CHECK(*env->GetScalar("term_reason_no_legal_drop") == 1.0f);
    logs.Flush();
}

TEST_CASE("DropMergeEnv reports terminal blocked persistence for settled NoLegal", "[dropmerge][phase1b]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeBlockedBoardConfig();
    config.settle_velocity_threshold = 100.0f;
    config.settle_angular_threshold = 100.0f;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-terminal-blocked[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto first_drop = env->Step(1);
    REQUIRE_FALSE(first_drop->next_state.done);
    REQUIRE(env->GetScalar("ep_terminal_blocked_frames").has_value());
    CHECK(std::isnan(*env->GetScalar("ep_terminal_blocked_frames")));
    REQUIRE(env->GetScalar("ep_blocked_run_count").has_value());
    CHECK(std::isnan(*env->GetScalar("ep_blocked_run_count")));

    const auto noop = env->Step(0);
    REQUIRE(noop->next_state.done);
    REQUIRE(env->GetScalar("ep_terminal_blocked_frames").has_value());
    CHECK(*env->GetScalar("ep_terminal_blocked_frames") > 0.0f);
    REQUIRE(env->GetScalar("ep_blocked_run_count").has_value());
    CHECK(*env->GetScalar("ep_blocked_run_count") == 0.0f);
    logs.Flush();
}

TEST_CASE("DropMergeEnv NoLegal adjudication config defaults to disabled with sixty frames", "[dropmerge][phase2][config]")
{
    const anet::rl::env::drop_merge::DropMergeEnvConfig config;

    CHECK_FALSE(config.use_no_legal_adjudication);
    CHECK(config.no_legal_min_blocked_frames == 60);
}

TEST_CASE("DropMergeEnv reads NoLegal adjudication config values", "[dropmerge][phase2][config]")
{
    anet::ConfigData config_data;
    config_data.Set("DropMergeEnv.use_no_legal_adjudication", true);
    config_data.Set("DropMergeEnv.no_legal_min_blocked_frames", 45);

    const anet::rl::env::drop_merge::DropMergeEnvConfig config(config_data);
    CHECK(config.use_no_legal_adjudication);
    CHECK(config.no_legal_min_blocked_frames == 45);
}

TEST_CASE("DropMergeEnv rejects a non-positive NoLegal adjudication horizon", "[dropmerge][phase2][config]")
{
    anet::rl::env::drop_merge::DropMergeEnvConfig config;
    config.no_legal_min_blocked_frames = 0;

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    CHECK_THROWS_WITH(
        std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
            config, torch::Device(torch::kCPU), "dropmerge-invalid-horizon[0]", 123),
        Catch::Matchers::ContainsSubstring("key=no_legal_min_blocked_frames")
        && Catch::Matchers::ContainsSubstring("value=0")
        && Catch::Matchers::ContainsSubstring("expected integer >= 1"));
    logs.Flush();
}

TEST_CASE("DropMergeEnv rejects a NoLegal horizon that cannot beat the timeout", "[dropmerge][phase2][config]")
{
    anet::rl::env::drop_merge::DropMergeEnvConfig config;
    config.use_no_legal_adjudication = true;
    config.use_no_drop_timeout_gameover = true;
    config.no_drop_timeout_steps = 60;
    config.no_legal_min_blocked_frames = 60;

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    CHECK_THROWS_WITH(
        std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
            config, torch::Device(torch::kCPU), "dropmerge-conflicting-horizon[0]", 123),
        Catch::Matchers::ContainsSubstring("key=no_legal_min_blocked_frames")
        && Catch::Matchers::ContainsSubstring("value=60")
        && Catch::Matchers::ContainsSubstring("expected < no_drop_timeout_steps=60"));
    logs.Flush();
}

TEST_CASE("DropMergeEnv reports timeout with a legal DROP available", "[dropmerge][no_legal_candidate]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::rl::env::drop_merge::DropMergeEnvConfig config;
    config.seed_mode = "fixed";
    config.action_mode = "direct_noop";
    config.use_no_legal_adjudication = true;
    config.no_legal_min_blocked_frames = 1;
    config.no_drop_timeout_steps = 1;
    config.max_step = 10;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-legal-timeout[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto timeout = env->Step(0);
    CHECK_FALSE(timeout->next_state.done);
    CHECK(timeout->next_state.truncated);
    REQUIRE(env->GetScalar("term_reason_no_drop_timeout").has_value());
    CHECK(*env->GetScalar("term_reason_no_drop_timeout") == 1.0f);
    REQUIRE(env->GetScalar("no_drop_timeout_on_candidate").has_value());
    CHECK(*env->GetScalar("no_drop_timeout_on_candidate") == 0.0f);
    REQUIRE(env->GetScalar("ep_terminal_blocked_frames").has_value());
    CHECK(*env->GetScalar("ep_terminal_blocked_frames") == 0.0f);
    REQUIRE(env->GetScalar("ep_blocked_run_count").has_value());
    CHECK(*env->GetScalar("ep_blocked_run_count") == 0.0f);
    logs.Flush();
}

TEST_CASE("DropMergeEnv reports timeout while an unsettled board remains blocked", "[dropmerge][no_legal_candidate]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeBlockedBoardConfig();
    config.no_drop_timeout_steps = 3;
    config.restitution = 1.0f;
    config.box_restitution = 1.0f;
    config.damping = 0.0f;
    config.settle_velocity_threshold = 0.0f;
    config.settle_angular_threshold = 0.0f;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-unsettled[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto first_drop = env->Step(1);
    REQUIRE_FALSE(first_drop->next_state.done);

    std::shared_ptr<const anet::rl::SingleStepResult> timeout;
    for (int i = 0; i < config.no_drop_timeout_steps; ++i) {
        timeout = env->Step(0);
    }

    REQUIRE(timeout != nullptr);
    CHECK_FALSE(timeout->next_state.done);
    CHECK(timeout->next_state.truncated);
    REQUIRE(env->GetScalar("term_reason_no_drop_timeout").has_value());
    CHECK(*env->GetScalar("term_reason_no_drop_timeout") == 1.0f);
    REQUIRE(env->GetScalar("no_drop_timeout_on_candidate").has_value());
    CHECK(*env->GetScalar("no_drop_timeout_on_candidate") == 1.0f);
    REQUIRE(env->GetScalar("ep_mean_blocked_frames").has_value());
    CHECK(*env->GetScalar("ep_mean_blocked_frames") == 0.0f);
    REQUIRE(env->GetScalar("ep_max_blocked_frames").has_value());
    CHECK(*env->GetScalar("ep_max_blocked_frames") == 0.0f);
    logs.Flush();
}

TEST_CASE("DropMergeEnv adjudicates persistent blocked frames without a penalty", "[dropmerge][phase2][adjudication]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeBlockedBoardConfig();
    config.use_no_legal_adjudication = true;
    config.no_legal_min_blocked_frames = 3;
    config.use_no_drop_timeout_gameover = true;
    config.no_drop_timeout_steps = 10;
    config.restitution = 1.0f;
    config.box_restitution = 1.0f;
    config.damping = 0.0f;
    config.settle_velocity_threshold = 0.0f;
    config.settle_angular_threshold = 0.0f;
    config.time_penalty = 0.0f;
    config.noop_penalty = 0.0f;
    config.game_over_penalty = -10.0f;
    config.no_drop_timeout_gameover_penalty = -10.0f;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-adjudicated[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto first_drop = env->Step(1);
    REQUIRE_FALSE(first_drop->next_state.done);

    const auto before_horizon = env->Step(0);
    CHECK_FALSE(before_horizon->next_state.done);
    CHECK_FALSE(before_horizon->next_state.truncated);

    const auto adjudicated = env->Step(0);
    CHECK(adjudicated->next_state.done);
    CHECK_FALSE(adjudicated->next_state.truncated);
    CHECK(adjudicated->reward == Catch::Approx(0.0f));
    REQUIRE(env->GetScalar("term_reason_no_legal_drop").has_value());
    CHECK(*env->GetScalar("term_reason_no_legal_drop") == 1.0f);
    REQUIRE(env->GetScalar("ep_terminal_blocked_frames").has_value());
    CHECK(*env->GetScalar("ep_terminal_blocked_frames") == 3.0f);

    logs.Flush();
    int persistence_log_count = 0;
    for (const auto& record : logs.Records()) {
        if (record.message.find("Episode done: no legal drop persisted for 3 frames.") != std::string::npos) {
            ++persistence_log_count;
        }
    }
    CHECK(persistence_log_count == 1);
}

TEST_CASE("DropMergeEnv prefers the settled NoLegal fast path", "[dropmerge][phase2][adjudication]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeBlockedBoardConfig();
    config.use_no_legal_adjudication = true;
    config.no_legal_min_blocked_frames = 1;
    config.settle_velocity_threshold = 100.0f;
    config.settle_angular_threshold = 100.0f;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-fast-path[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto first_drop = env->Step(1);
    REQUIRE_FALSE(first_drop->next_state.done);

    const auto noop = env->Step(0);
    CHECK(noop->next_state.done);
    CHECK_FALSE(noop->next_state.truncated);

    logs.Flush();
    int settled_log_count = 0;
    int persistence_log_count = 0;
    for (const auto& record : logs.Records()) {
        if (record.message.find("Episode done: no legal drop remains.") != std::string::npos) {
            ++settled_log_count;
        }
        if (record.message.find("Episode done: no legal drop persisted") != std::string::npos) {
            ++persistence_log_count;
        }
    }
    CHECK(settled_log_count == 1);
    CHECK(persistence_log_count == 0);
}

TEST_CASE("DropMergeEnv reports the length of a resolved blocked run", "[dropmerge][blocked_persistence]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::rl::env::drop_merge::DropMergeEnvConfig config;
    config.seed_mode = "fixed";
    config.action_mode = "direct_noop";
    config.drop_divisions = 1;
    config.box_width = 1.0f;
    config.box_height = 2.0f;
    config.fruit_radii.assign(anet::rl::env::drop_merge::kFruitTypeCount, 0.1f);
    config.drop_probs = { 1.0f };
    config.drop_noise = 0.0f;
    config.spin_noise = 0.0f;
    config.restitution = 0.0f;
    config.damping = 0.0f;
    config.use_instant_drop = true;
    config.reload_min_steps = 0;
    config.reload_max_steps = 1;
    config.use_no_legal_adjudication = true;
    config.no_legal_min_blocked_frames = 29;
    config.no_drop_timeout_steps = 30;
    config.max_step = 100;
    config.game_over_grace_step = 1000;
    config.settle_velocity_threshold = 0.0f;
    config.settle_angular_threshold = 0.0f;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-resolved-run[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto first_drop = env->Step(1);
    REQUIRE_FALSE(first_drop->next_state.done);

    std::shared_ptr<const anet::rl::SingleStepResult> timeout;
    for (int i = 0; i < config.no_drop_timeout_steps; ++i) {
        timeout = env->Step(0);
    }

    REQUIRE(timeout != nullptr);
    REQUIRE(timeout->next_state.truncated);
    REQUIRE(env->GetScalar("ep_mean_blocked_frames").has_value());
    REQUIRE(env->GetScalar("ep_max_blocked_frames").has_value());
    const float mean_blocked_frames = *env->GetScalar("ep_mean_blocked_frames");
    const float max_blocked_frames = *env->GetScalar("ep_max_blocked_frames");
    CHECK(mean_blocked_frames > 0.0f);
    CHECK(mean_blocked_frames == max_blocked_frames);
    CHECK(max_blocked_frames < static_cast<float>(config.no_drop_timeout_steps));
    REQUIRE(env->GetScalar("ep_terminal_blocked_frames").has_value());
    CHECK(*env->GetScalar("ep_terminal_blocked_frames") == 0.0f);
    REQUIRE(env->GetScalar("ep_blocked_run_count").has_value());
    CHECK(*env->GetScalar("ep_blocked_run_count") > 0.0f);
    logs.Flush();
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
