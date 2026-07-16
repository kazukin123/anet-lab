#include "anet/catch_test.hpp"

#include "anet/app_util.hpp"
#include "anet/config.hpp"
#include "anet/schedule.hpp"
#include "anet/test_util.hpp"

#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string_view>
#include <vector>

using namespace anet::test;

namespace {

TEST_CASE("Train Actor snapshot metrics are registered only in the full catalog", "[config][metrics][snapshot]")
{
    const auto repository_root = anet::GetExecutableRootDir().parent_path().parent_path();
    const auto catalog_path = repository_root / "apps" / "runner" / "config" / "metrics_scalar.txt";
    std::ifstream catalog(catalog_path);
    REQUIRE(catalog);

    int snapshot_metric_count = 0;
    std::string line;
    while (std::getline(catalog, line)) {
        if (line.find("train_actor_snapshot_") == std::string::npos) continue;

        ++snapshot_metric_count;
        CHECK(line.starts_with("metrics.scalar.full."));
        CHECK(line.find("@train $action_info") != std::string::npos);
    }
    CHECK(snapshot_metric_count == 2);
}

class ScopedCurrentPath final {
public:
    explicit ScopedCurrentPath(const std::filesystem::path& path)
        : old_path_(std::filesystem::current_path())
    {
        std::filesystem::current_path(path);
    }

    ~ScopedCurrentPath()
    {
        std::error_code ec;
        std::filesystem::current_path(old_path_, ec);
    }

    ScopedCurrentPath(const ScopedCurrentPath&) = delete;
    ScopedCurrentPath& operator=(const ScopedCurrentPath&) = delete;

private:
    std::filesystem::path old_path_;
};

anet::ConfigManagerOptions MakeConfigSearchOptions(std::vector<std::filesystem::path> dirs)
{
    anet::ConfigManagerOptions options;
    options.config_search_dirs = dirs;
    return options;
}

void WriteConfig(const std::filesystem::path& path, std::initializer_list<std::string_view> lines)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream ofs(path);
    for (const auto line : lines) {
        ofs << line << "\n";
    }
}

class ProfiledValueOwnerConfig final : public anet::Config {
public:
    anet::ProfiledValueConfig<double> learning_rate;

    explicit ProfiledValueOwnerConfig(
        const anet::ConfigData& config_data,
        const std::string& config_prefix = "")
        : anet::Config(config_data, "ImageClsAgent", config_prefix)
    {
        ANET_READ_CONFIG(config_data, learning_rate);
    }
};

class PositiveProfiledValueOwnerConfig final : public anet::Config {
public:
    anet::ProfiledValueConfig<double> interval{
        .type = "constant",
        .value = 400.0,
        .min_value = 1.0,
    };

    explicit PositiveProfiledValueOwnerConfig(
        const anet::ConfigData& config_data,
        const std::string& override_prefix = "")
        : anet::Config(config_data, "PositiveProfile", override_prefix)
    {
        ANET_READ_CONFIG(config_data, interval);
    }
};

anet::ProfiledValueConfig<double> MakeDefaultPhasedProfile()
{
    anet::ProfiledValueConfig<double> config{
        .type = "phased",
        .phases = { "main" },
    };
    config.phase.Set("main", anet::ProfiledValuePhaseConfig<double>{
        .type = "constant",
        .value = 2.0,
        .start = 4.0,
        .end = 2.0,
        .steps = 10,
    });
    config.phase.Set("dormant", anet::ProfiledValuePhaseConfig<double>{
        .type = "constant",
        .value = 0.0,
        .steps = 0,
    });
    return config;
}

class DefaultPhasedProfileOwnerConfig final : public anet::Config {
public:
    anet::ProfiledValueConfig<double> profile = MakeDefaultPhasedProfile();

    explicit DefaultPhasedProfileOwnerConfig(
        const anet::ConfigData& config_data,
        const std::string& override_prefix = "")
        : anet::Config(config_data, "DefaultProfile", override_prefix)
    {
        ANET_READ_CONFIG(config_data, profile);
    }
};

} // namespace

TEST_CASE("ConfigData Read rejects present invalid values", "[config]")
{
    anet::ConfigData config_data;
    config_data.Set("invalid_int", "not-int");
    config_data.Set("invalid_bool", "maybe");
    config_data.Set("empty_bool", "  ");
    config_data.Set("invalid_float_vector", "1.0 x");
    config_data.Set("valid_int", "42");

    SECTION("invalid int throws without replacing the current value")
    {
        int value = 5;
        CHECK_THROWS_WITH(
            config_data.Read("invalid_int", value, 9),
            Catch::Matchers::ContainsSubstring("key=invalid_int")
            && Catch::Matchers::ContainsSubstring("value=\"not-int\"")
            && Catch::Matchers::ContainsSubstring("expected=int"));
        CHECK(value == 5);
    }

    SECTION("invalid bool throws without replacing the current value")
    {
        bool value = true;
        CHECK_THROWS_WITH(
            config_data.Read("invalid_bool", value, false),
            Catch::Matchers::ContainsSubstring("key=invalid_bool")
            && Catch::Matchers::ContainsSubstring("value=\"maybe\"")
            && Catch::Matchers::ContainsSubstring("expected=bool"));
        CHECK(value);
    }

    SECTION("explicit empty bool is a format error")
    {
        bool value = true;
        CHECK_THROWS_WITH(
            config_data.Read("empty_bool", value, false),
            Catch::Matchers::ContainsSubstring("key=empty_bool")
            && Catch::Matchers::ContainsSubstring("value=\"  \"")
            && Catch::Matchers::ContainsSubstring("expected=bool"));
        CHECK(value);
    }

    SECTION("invalid vector throws without replacing the current value")
    {
        std::vector<float> value = { 9.0f };
        const std::vector<float> default_value = { 1.0f, 2.0f };
        CHECK_THROWS_WITH(
            config_data.Read("invalid_float_vector", value, default_value),
            Catch::Matchers::ContainsSubstring("key=invalid_float_vector")
            && Catch::Matchers::ContainsSubstring("value=\"1.0 x\"")
            && Catch::Matchers::ContainsSubstring("expected=float vector"));
        CHECK(value == std::vector<float>{ 9.0f });
    }

    SECTION("missing key keeps the existing no-warning behavior")
    {
        LogCaptureGuard logs;

        int value = 5;
        CHECK_FALSE(config_data.Read("missing_int", value, 9));
        CHECK(value == 9);

        logs.Flush();
        CHECK(CountRecords(logs.Records(), wxLOG_Warning) == 0);
    }

    SECTION("valid value does not warn")
    {
        LogCaptureGuard logs;

        int value = 5;
        CHECK(config_data.Read("valid_int", value, 9));
        CHECK(value == 42);

        logs.Flush();
        CHECK(CountRecords(logs.Records(), wxLOG_Warning) == 0);
    }
}

TEST_CASE("ConfigData numeric readers require complete finite values", "[config]")
{
    anet::ConfigData config_data;
    config_data.Set("trailing_int", "42x");
    config_data.Set("empty_int", "  ");
    config_data.Set("negative_unsigned", "-1");
    config_data.Set("overflow_unsigned", "18446744073709551616");
    config_data.Set("nonfinite_float", "nan");
    config_data.Set("nonfinite_double", "inf");
    config_data.Set("comma_int", " 1,2,3 ");
    config_data.Set("negative_int64", " -7 ");

    int int_value = 5;
    CHECK_THROWS(config_data.Read("trailing_int", int_value, 9));
    CHECK_THROWS(config_data.Read("empty_int", int_value, 9));

    uint64_t unsigned_value = 5;
    CHECK_THROWS(config_data.Read("negative_unsigned", unsigned_value, 9));
    CHECK_THROWS(config_data.Read("overflow_unsigned", unsigned_value, 9));

    float float_value = 5.0f;
    CHECK_THROWS(config_data.Read("nonfinite_float", float_value, 9.0f));

    double double_value = 5.0;
    CHECK_THROWS(config_data.Read("nonfinite_double", double_value, 9.0));

    CHECK(config_data.Read("comma_int", int_value, 9));
    CHECK(int_value == 123);

    int64_t int64_value = 5;
    CHECK(config_data.Read("negative_int64", int64_value, 9));
    CHECK(int64_value == -7);
}

TEST_CASE("ConfigData Get shares the Read fail-fast contract", "[config]")
{
    anet::ConfigData config_data;
    config_data.Set("invalid", "12x");

    CHECK(config_data.Get<int>("missing", 7) == 7);
    CHECK_THROWS(config_data.Get<int>("invalid", 7));
}

TEST_CASE("ConfigData accepts explicit empty strings and vectors", "[config]")
{
    anet::ConfigData config_data;
    config_data.Set("string", "");
    config_data.Set("float_vector", "  ");
    config_data.Set("int_vector", "");
    config_data.Set("string_vector", "　");

    std::string string_value = "default";
    CHECK(config_data.Read("string", string_value, "default"));
    CHECK(string_value.empty());

    std::vector<float> float_vector = { 1.0f };
    CHECK(config_data.Read("float_vector", float_vector, { 2.0f }));
    CHECK(float_vector.empty());

    std::vector<int64_t> int_vector = { 1 };
    CHECK(config_data.Read("int_vector", int_vector, { 2 }));
    CHECK(int_vector.empty());

    std::vector<std::string> string_vector = { "default" };
    CHECK(config_data.Read("string_vector", string_vector, { "fallback" }));
    CHECK(string_vector.empty());
}

TEST_CASE("ConfigData numeric vectors use strict scalar token parsing", "[config]")
{
    anet::ConfigData config_data;
    config_data.Set("float_vector", "1.0 inf");
    config_data.Set("int_vector", "-7 8x");
    config_data.Set("comma_vector", "1,2 3");

    std::vector<float> float_vector = { 9.0f };
    CHECK_THROWS(config_data.Read("float_vector", float_vector, { 1.0f }));
    CHECK(float_vector == std::vector<float>{ 9.0f });

    std::vector<int64_t> int_vector = { 9 };
    CHECK_THROWS(config_data.Read("int_vector", int_vector, { 1 }));
    CHECK(int_vector == std::vector<int64_t>{ 9 });

    CHECK(config_data.Read("comma_vector", int_vector, { 1 }));
    CHECK(int_vector == std::vector<int64_t>{ 12, 3 });
}

TEST_CASE("Config reads ProfiledValueConfig root fields and phases", "[config][profiled_value]")
{
    anet::ConfigData config_data;
    config_data.Set("ImageClsAgent.learning_rate.type", std::string("phased"));
    config_data.Set("ImageClsAgent.learning_rate.value", 0.5);
    config_data.Set("ImageClsAgent.learning_rate.start", 0.0);
    config_data.Set("ImageClsAgent.learning_rate.end", 1.0);
    config_data.Set("ImageClsAgent.learning_rate.steps", 100);
    config_data.Set("ImageClsAgent.learning_rate.cycle_mult", 2.0);
    config_data.Set("ImageClsAgent.learning_rate.phases", std::string("warmup main"));

    config_data.Set("ImageClsAgent.learning_rate.phase.[warmup].type", std::string("linear"));
    config_data.Set("ImageClsAgent.learning_rate.phase.[warmup].start", 0.0);
    config_data.Set("ImageClsAgent.learning_rate.phase.[warmup].end", 0.1);
    config_data.Set("ImageClsAgent.learning_rate.phase.[warmup].steps", 10);

    config_data.Set("ImageClsAgent.learning_rate.phase.[main].type", std::string("cosine"));
    config_data.Set("ImageClsAgent.learning_rate.phase.[main].start", 0.1);
    config_data.Set("ImageClsAgent.learning_rate.phase.[main].end", 0.01);
    config_data.Set("ImageClsAgent.learning_rate.phase.[main].steps", 90);

    ProfiledValueOwnerConfig config(config_data);

    CHECK(config.learning_rate.type == "phased");
    CHECK(config.learning_rate.value == Catch::Approx(0.5));
    CHECK(config.learning_rate.start == Catch::Approx(0.0));
    CHECK(config.learning_rate.end == Catch::Approx(1.0));
    CHECK(config.learning_rate.steps == 100);
    CHECK(config.learning_rate.cycle_mult == Catch::Approx(2.0));
    CHECK_FALSE(config.learning_rate.min_value.has_value());
    CHECK_FALSE(config.learning_rate.max_value.has_value());
    const std::vector<std::string> expected_phases = { "warmup", "main" };
    CHECK(config.learning_rate.phases == expected_phases);

    REQUIRE(config.learning_rate.phase.Has("warmup"));
    REQUIRE(config.learning_rate.phase.Has("main"));
    const auto& warmup = config.learning_rate.phase.Get("warmup");
    CHECK(warmup.type == "linear");
    CHECK(warmup.start == Catch::Approx(0.0));
    CHECK(warmup.end == Catch::Approx(0.1));
    CHECK(warmup.steps == 10);

    const auto& main = config.learning_rate.phase.Get("main");
    CHECK(main.type == "cosine");
    CHECK(main.start == Catch::Approx(0.1));
    CHECK(main.end == Catch::Approx(0.01));
    CHECK(main.steps == 90);

    const auto json = config.ToJson();
    CHECK(json.at("learning_rate.type") == "phased");
    CHECK(json.at("learning_rate.value") == 0.5);
    CHECK(json.at("learning_rate.phases") == anet::json::array({ "warmup", "main" }));
    CHECK(json.at("learning_rate.phase.[warmup].type") == "linear");
    CHECK(json.at("learning_rate.phase.[main].steps") == 90);
    CHECK_FALSE(json.contains("learning_rate.min_value"));
    CHECK_FALSE(json.contains("learning_rate.max_value"));

    const auto config_string = config.ToConfigString();
    CHECK(config_string.find("ImageClsAgent.learning_rate.type = phased") != std::string::npos);
    CHECK(config_string.find("ImageClsAgent.learning_rate.phases = warmup main") != std::string::npos);
    CHECK(config_string.find("ImageClsAgent.learning_rate.phase.[warmup].start = 0") != std::string::npos);
    CHECK(config_string.find("ImageClsAgent.learning_rate.phase.[main].steps = 90") != std::string::npos);
    CHECK(config_string.find("min_value") == std::string::npos);
    CHECK(config_string.find("max_value") == std::string::npos);

    // bounds未指定のImageCls学習率では、0を従来どおり有効値として扱う。
    const anet::ProfiledValue<double> learning_rate(config.learning_rate);
    CHECK(learning_rate.Value() == Catch::Approx(0.0));
}

TEST_CASE("Config rejects a ProfiledValueConfig that violates its bounds", "[config][profiled_value][bounds]")
{
    SECTION("default layer reports a layer-neutral logical key")
    {
        anet::ConfigData config_data;
        config_data.Set("PositiveProfile.interval.type", "constant");
        config_data.Set("PositiveProfile.interval.value", "0");

        CHECK_THROWS_WITH(
            PositiveProfiledValueOwnerConfig(config_data),
            Catch::Matchers::ContainsSubstring("key=interval.value value=0"));
    }

    SECTION("override layer reports the same layer-neutral logical key")
    {
        anet::ConfigData config_data;
        config_data.Set("PositiveProfile.interval.type", "constant");
        config_data.Set("PositiveProfile.interval.value", "400");
        config_data.Set("Trial.interval.value", "0");

        CHECK_THROWS_WITH(
            PositiveProfiledValueOwnerConfig(config_data, "Trial"),
            Catch::Matchers::ContainsSubstring("key=interval.value value=0"));
    }
}

TEST_CASE("Config override does not mask a malformed default layer", "[config][profiled_value]")
{
    anet::ConfigData config_data;
    config_data.Set("PositiveProfile.interval.value", "invalid");
    config_data.Set("Trial.interval.value", "2");

    CHECK_THROWS_WITH(
        PositiveProfiledValueOwnerConfig(config_data, "Trial"),
        Catch::Matchers::ContainsSubstring("key=PositiveProfile.interval.value")
        && Catch::Matchers::ContainsSubstring("value=\"invalid\"")
        && Catch::Matchers::ContainsSubstring("expected=double"));
}

TEST_CASE("Config rejects a listed ProfiledValue phase without a definition", "[config][profiled_value]")
{
    anet::ConfigData config_data;
    config_data.Set("ImageClsAgent.learning_rate.type", "phased");
    config_data.Set("ImageClsAgent.learning_rate.phases", "missing");

    CHECK_THROWS_WITH(
        ProfiledValueOwnerConfig(config_data),
        Catch::Matchers::ContainsSubstring("phase is listed but not defined")
        && Catch::Matchers::ContainsSubstring("value=missing"));
}

TEST_CASE("Config preserves programmatic ProfiledValue phase defaults", "[config][profiled_value]")
{
    anet::ConfigData config_data;
    config_data.Set("Trial.profile.phase.[main].type", "cosine");

    DefaultPhasedProfileOwnerConfig config(config_data, "Trial");
    const auto& main = config.profile.phase.Get("main");
    CHECK(main.type == "cosine");
    CHECK(main.start == Catch::Approx(4.0));
    CHECK(main.end == Catch::Approx(2.0));
    CHECK(main.steps == 10);
    CHECK(config.profile.phase.find("dormant") != config.profile.phase.end());
}

TEST_CASE("Config ProfiledValueConfig override can switch type and keep dormant fields", "[config][profiled_value]")
{
    anet::ConfigData config_data;
    config_data.Set("ImageClsAgent.learning_rate.type", std::string("phased"));
    config_data.Set("ImageClsAgent.learning_rate.phases", std::string("main"));

    config_data.Set("ImageClsAgent.learning_rate.phase.[main].type", std::string("constant"));
    config_data.Set("ImageClsAgent.learning_rate.phase.[main].value", 0.2);
    config_data.Set("ImageClsAgent.learning_rate.phase.[main].start", 0.1);
    config_data.Set("ImageClsAgent.learning_rate.phase.[main].end", 0.01);
    config_data.Set("ImageClsAgent.learning_rate.phase.[main].steps", 100);

    config_data.Set("Trial.learning_rate.phase.[main].type", std::string("cosine"));

    ProfiledValueOwnerConfig config(config_data, "Trial");

    REQUIRE(config.learning_rate.phase.Has("main"));
    const auto& phase = config.learning_rate.phase.Get("main");
    CHECK(phase.type == "cosine");
    CHECK(phase.value == Catch::Approx(0.2));
    CHECK(phase.start == Catch::Approx(0.1));
    CHECK(phase.end == Catch::Approx(0.01));
    CHECK(phase.steps == 100);

    anet::ProfiledValue<double> value(config.learning_rate);
    CHECK(value.Evaluate(50) == Catch::Approx(0.055));
}

TEST_CASE("ConfigManager loads trial main config with include and override", "[config]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "config-manager-trial-main-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    const auto base_path = root / "base.txt";
    const auto trial_path = root / "trial.txt";

    {
        std::ofstream ofs(base_path);
        ofs << "app.run_name = base_run\n";
        ofs << "net.branch.[main_feature].structure = BaseStructure\n";
        ofs << "net.branch.[main_feature].$ = net.branch.Base\n";
        ofs << "net.branch.Base.structure = BaseBranch\n";
        ofs << "net.branch.Trial.structure = TrialBranch\n";
    }
    {
        std::ofstream ofs(trial_path);
        ofs << "$include <base.txt>\n";
        ofs << "app.run_name = optuna_trial_00001\n";
        ofs << "net.branch.[main_feature].$ = net.branch.Trial\n";
    }

    anet::ConfigManager manager(trial_path.string(), nullptr);
    const auto config_data = manager.GetConfigData();

    CHECK(config_data.Get("app.run_name") == "optuna_trial_00001");
    CHECK(config_data.Get("net.branch.[main_feature].structure") == "TrialBranch");

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager AutoMerge only merges dot-delimited descendants", "[config]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "config-manager-merge-boundary-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "backend.$ = backend.deterministic",
        "backend.deterministic_algorithms = true",
        "backend.deterministic_warn_only = false",
        "backend.deterministic.cudnn_benchmark = false",
        "backend.deterministic.cudnn_deterministic = true",
        "backend.deterministic.deterministic_algorithms = true",
    });

    anet::ConfigManager manager(config_path.string(), nullptr);
    const auto config_data = manager.GetConfigData();

    // profile 配下の正規の子設定だけを backend 直下へ展開する。
    CHECK(config_data.Get("backend.cudnn_benchmark") == "false");
    CHECK(config_data.Get("backend.cudnn_deterministic") == "true");
    CHECK(config_data.Get("backend.deterministic_algorithms") == "true");
    CHECK(config_data.Get("backend.deterministic_warn_only") == "false");

    // profile 名と文字列 prefix が同じだけのキーから、不正な派生キーを作らない。
    CHECK_FALSE(config_data.Has("backend_algorithms"));
    CHECK_FALSE(config_data.Has("backend_warn_only"));

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager resolves include paths from parent before config search dirs", "[config]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "config-manager-include-order-test";
    std::filesystem::remove_all(root);

    const auto main_dir = root / "main";
    const auto config_dir = root / "config";
    WriteConfig(main_dir / "trial.txt", {
        "$include <base.txt>",
    });
    WriteConfig(main_dir / "base.txt", {
        "app.run_name = parent_base",
    });
    WriteConfig(config_dir / "base.txt", {
        "app.run_name = fallback_base",
    });

    anet::ConfigManager manager(
        (main_dir / "trial.txt").string(),
        nullptr,
        MakeConfigSearchOptions({ config_dir }));
    const auto config_data = manager.GetConfigData();

    CHECK(config_data.Get("app.run_name") == "parent_base");

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager resolves include paths from config search dirs", "[config]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "config-manager-include-fallback-test";
    std::filesystem::remove_all(root);

    const auto main_dir = root / "main";
    const auto config_dir = root / "config";
    WriteConfig(main_dir / "trial.txt", {
        "$include <fallback_only.txt>",
    });
    WriteConfig(config_dir / "fallback_only.txt", {
        "app.run_name = fallback_include",
    });

    anet::ConfigManager manager(
        (main_dir / "trial.txt").string(),
        nullptr,
        MakeConfigSearchOptions({ config_dir }));
    const auto config_data = manager.GetConfigData();

    CHECK(config_data.Get("app.run_name") == "fallback_include");

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager warns and continues when include is missing", "[config]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "config-manager-missing-include-test";
    std::filesystem::remove_all(root);

    const auto main_path = root / "trial.txt";
    WriteConfig(main_path, {
        "$include <missing.txt>",
        "app.run_name = after_missing_include",
    });

    LogCaptureGuard logs;
    anet::ConfigManager manager(
        main_path.string(),
        nullptr,
        MakeConfigSearchOptions(std::vector<std::filesystem::path>{}));
    const auto config_data = manager.GetConfigData();

    logs.Flush();
    CHECK(config_data.Get("app.run_name") == "after_missing_include");
    CHECK(HasRecordContaining(
        logs.Records(),
        wxLOG_Warning,
        { "Properties: Failed to open include file", "missing.txt" }));

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager resolves relative main config from cwd before config search dirs", "[config]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "config-manager-main-fallback-test";
    std::filesystem::remove_all(root);

    const auto cwd_dir = root / "cwd";
    const auto config_dir = root / "config";
    WriteConfig(cwd_dir / "main.txt", {
        "app.run_name = cwd_main",
    });
    WriteConfig(config_dir / "main.txt", {
        "app.run_name = fallback_main",
    });
    WriteConfig(config_dir / "fallback_only.txt", {
        "app.run_name = fallback_only_main",
    });

    {
        ScopedCurrentPath scoped_cwd(cwd_dir);
        const auto options = MakeConfigSearchOptions({ config_dir });

        anet::ConfigManager cwd_manager("main.txt", nullptr, options);
        CHECK(cwd_manager.GetConfigData().Get("app.run_name") == "cwd_main");

        anet::ConfigManager fallback_manager("fallback_only.txt", nullptr, options);
        CHECK(fallback_manager.GetConfigData().Get("app.run_name") == "fallback_only_main");

        CHECK_THROWS(anet::ConfigManager("missing_main.txt", nullptr, options));
    }

    std::filesystem::remove_all(root);
}
