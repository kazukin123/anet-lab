#include "anet/catch_test.hpp"

#include "anet/config.hpp"
#include "anet/schedule.hpp"
#include "anet/test_util.hpp"

#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string_view>
#include <vector>
#include <wx/cmdline.h>

using namespace anet::test;

namespace {

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

class ScopedSnapshotConfig final : public anet::Config {
public:
    int value = 1;
    std::string label = "default";

    explicit ScopedSnapshotConfig(
        const anet::ConfigData& config_data,
        const std::string& override_prefix = "")
        : anet::Config(config_data, "ScopedSnapshot", override_prefix)
    {
        ANET_READ_CONFIG(config_data, value);
        ANET_READ_CONFIG(config_data, label);
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

TEST_CASE("Config exposes resolved values under the injected scope", "[config][snapshot]")
{
    anet::ConfigData config_data;
    config_data.Set("ScopedSnapshot.value", 10);
    config_data.Set("ScopedSnapshot.label", "base");
    config_data.Set("train.eval.[eval1].env.value", 20);

    const ScopedSnapshotConfig config(config_data, "train.eval.[eval1].env");
    const auto snapshot = config.GetScopedConfigData();

    CHECK(snapshot.Get<int>("train.eval.[eval1].env.value") == 20);
    CHECK(snapshot.Get("train.eval.[eval1].env.label") == "base");
    CHECK_FALSE(snapshot.Has("ScopedSnapshot.value"));
}

TEST_CASE("ConfigData checked merge rejects conflicting effective values", "[config][snapshot]")
{
    anet::ConfigData merged;
    merged.Set("env.batch_size", 4);

    anet::ConfigData compatible;
    compatible.Set("env.batch_size", 4);
    compatible.Set("ImageClsEnv.train.dataset_key", "food101_train");
    merged.MergeFromChecked(compatible);

    CHECK(merged.Get<int>("env.batch_size") == 4);
    CHECK(merged.Get("ImageClsEnv.train.dataset_key") == "food101_train");

    anet::ConfigData conflicting;
    conflicting.Set("env.batch_size", 8);
    CHECK_THROWS(merged.MergeFromChecked(conflicting));
}

TEST_CASE("ConfigData saves Properties text and replaces an existing file", "[config][properties]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-data-save-properties-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto path = root / "history.txt";

    anet::ConfigData first;
    first.Set("workspace.history.0", R"(D:\Program Files\anet workspace)");
    first.Set("workspace.history.1", "_default");
    first.SaveProperties(path);

    CHECK(first.ToPropertiesString() ==
        "workspace.history.0 = D:\\Program Files\\anet workspace\n"
        "workspace.history.1 = _default\n");
    const auto loaded_first = anet::Properties(path.string()).ToConfigData();
    CHECK(loaded_first.Map().Size() == 2);
    CHECK(loaded_first.Get("workspace.history.0") == R"(D:\Program Files\anet workspace)");
    CHECK(loaded_first.Get("workspace.history.1") == "_default");

    for (const bool skip : { false, true }) {
        const std::string expected = std::string("workspace.dialog_skip = ")
            + (skip ? "true\n" : "false\n");

        anet::ConfigData second;
        second.Set("workspace.dialog_skip", skip);
        CHECK(second.Get("workspace.dialog_skip") == (skip ? "true" : "false"));
        CHECK(second.ToPropertiesString() == expected);
        second.SaveProperties(path);

        std::ifstream saved_file(path, std::ios::binary);
        std::stringstream saved_text;
        saved_text << saved_file.rdbuf();
        CHECK(saved_text.str() == expected);

        const auto loaded_second = anet::Properties(path.string()).ToConfigData();
        CHECK(loaded_second.Map().Size() == 1);
        CHECK(loaded_second.Get<bool>("workspace.dialog_skip") == skip);
    }
    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigData rejects Properties control tokens when saving", "[config][properties]")
{
    const auto path = std::filesystem::current_path() / "out" / "test-tmp" / "unsafe-properties.txt";

    for (const auto& value : { "value#comment", "value//comment", "value;", "value;  " }) {
        anet::ConfigData config_data;
        config_data.Set("workspace.value", value);
        CHECK_THROWS_WITH(
            config_data.SaveProperties(path),
            Catch::Matchers::ContainsSubstring("key=workspace.value")
            && Catch::Matchers::ContainsSubstring(value));
    }
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

TEST_CASE("ConfigManager applies injected and file overlays before AutoMerge", "[config][workspace]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-workspace-overlay-test";
    std::filesystem::remove_all(root);

    const auto config_dir = root / "config";
    const auto workspace_dir = root / "workspace";
    WriteConfig(config_dir / "base.txt", {
        "app.runs_dir = base-runs",
        "env.profile.value = base",
        "env.$ = env.profile",
    });
    WriteConfig(workspace_dir / "_main.txt", {
        "$include <base.txt>",
        "env.profile.value = workspace",
    });
    WriteConfig(workspace_dir / "override.txt", {
        "$include <overlay-include.txt>",
        "app.runs_dir = forbidden-override",
    });
    WriteConfig(config_dir / "overlay-include.txt", {
        "overlay.include = found",
    });

    anet::ConfigManagerOptions options;
    options.config_search_dirs = { config_dir };
    options.injected_config.Set("app.runs_dir", "workspace-runs");
    options.overwrite_config_paths = { workspace_dir / "override.txt" };

    anet::ConfigManager manager((workspace_dir / "_main.txt").string(), nullptr, options);
    const auto config_data = manager.GetConfigData();

    CHECK(config_data.Get("app.runs_dir") == "forbidden-override");
    CHECK(config_data.Get("overlay.include") == "found");
    CHECK(config_data.Get("env.value") == "workspace");

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

TEST_CASE("ConfigManager resolves relative material selection and records resolution", "[config][resolver]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-relative-material-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "AtariEnv.@baseline.repeat_action_probability = 0.25",
        "AtariEnv.$ = @baseline",
    });

    const anet::ConfigManager manager(config_path.string());
    const auto config_data = manager.GetConfigData();
    CHECK(config_data.Get("AtariEnv.repeat_action_probability") == "0.25");
    CHECK_FALSE(config_data.Has("AtariEnv.@baseline.repeat_action_probability"));

    const auto resolution = manager.GetResolutionJson();
    CHECK(resolution["schema_version"] == 1);
    REQUIRE(resolution["selections"].size() == 1);
    CHECK(resolution["selections"][0]["key"] == "AtariEnv.$");
    CHECK(resolution["selections"][0]["chain"][0]["term"] == "@baseline");
    CHECK(resolution["selections"][0]["chain"][0]["resolved"] == "AtariEnv.@baseline");
    CHECK(resolution["references"].empty());

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager expands a named trunk before root selections", "[config][resolver][trunk]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-named-trunk-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "run.@verify.env.class_id = AtariEnv",
        "run.@verify.AtariEnv.$ = @v5",
        "AtariEnv.@v5.repeat_action_probability = 0.25",
        "run.$ = run.@verify",
    });

    const anet::ConfigManager manager(config_path.string());
    const auto config_data = manager.GetConfigData();
    CHECK(config_data.Get("env.class_id") == "AtariEnv");
    CHECK(config_data.Get("AtariEnv.repeat_action_probability") == "0.25");
    CHECK_FALSE(config_data.Has("run.$"));
    CHECK_FALSE(config_data.Has("run.@verify.env.class_id"));

    const auto resolution = manager.GetResolutionJson();
    CHECK(resolution["schema_version"] == 1);
    REQUIRE(resolution["selections"].size() == 2);
    CHECK(resolution["selections"][0]["key"] == "run.$");
    CHECK(resolution["selections"][0]["chain"][0]["resolved"] == "run.@verify");
    CHECK(resolution["selections"][1]["key"] == "AtariEnv.$");

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager resolves a relative named trunk term", "[config][resolver][trunk]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-relative-trunk-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "run.@verify.env.class_id = AtariEnv",
        "run.$ = @verify",
    });

    const anet::ConfigManager manager(config_path.string());
    CHECK(manager.GetConfigData().Get("env.class_id") == "AtariEnv");
    CHECK(manager.GetResolutionJson()["selections"][0]["chain"][0]["resolved"]
        == "run.@verify");

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager applies named trunk terms from left to right", "[config][resolver][trunk]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-trunk-chain-order-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "run.@a.env.class_id = EnvA",
        "run.@b.env.class_id = EnvB",
        "run.$ = @a > @b",
    });

    const anet::ConfigManager manager(config_path.string());
    CHECK(manager.GetConfigData().Get("env.class_id") == "EnvB");

    const auto chain = manager.GetResolutionJson()["selections"][0]["chain"];
    REQUIRE(chain.size() == 2);
    CHECK(chain[0]["resolved"] == "run.@a");
    CHECK(chain[1]["resolved"] == "run.@b");

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager applies a named trunk as a file-tail overwrite", "[config][resolver][trunk]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-trunk-tail-overwrite-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "env.class_id = OldEnv",
        "Env.@a.shared = a",
        "Env.@a.only_a = stale",
        "Env.@b.shared = b",
        "Env.$ = @a",
        "run.@verify.env.class_id = NewEnv",
        "run.@verify.Env.$ = @b",
        "run.$ = @verify",
    });

    const anet::ConfigManager manager(config_path.string());
    const auto config_data = manager.GetConfigData();
    CHECK(config_data.Get("env.class_id") == "NewEnv");
    CHECK(config_data.Get("Env.shared") == "b");
    CHECK_FALSE(config_data.Has("Env.only_a"));

    const auto selections = manager.GetResolutionJson()["selections"];
    REQUIRE(selections.size() == 2);
    CHECK(selections[1]["key"] == "Env.$");
    CHECK(selections[1]["chain"][0]["term"] == "@b");

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager switches the named trunk from CLI phase one", "[config][resolver][trunk][cli]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-cli-trunk-switch-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "run.@a.env.class_id = EnvA",
        "run.@b.env.class_id = EnvB",
        "run.$ = @a",
    });

    const wxCmdLineEntryDesc description[] = {
        { wxCMD_LINE_PARAM, nullptr, nullptr, "key=value", wxCMD_LINE_VAL_STRING,
            wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE },
        { wxCMD_LINE_NONE },
    };
    wxCmdLineParser command_line(description, "run.$=run.@b");
    REQUIRE(command_line.Parse(false) == 0);

    const anet::ConfigManager manager(config_path.string(), &command_line);
    CHECK(manager.GetConfigData().Get("env.class_id") == "EnvB");
    CHECK(manager.GetResolutionJson()["selections"][0]["chain"][0]["term"] == "run.@b");

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager applies CLI leaf override after named trunk expansion", "[config][resolver][trunk][cli]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-cli-trunk-leaf-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "run.@verify.env.class_id = TrunkEnv",
        "run.$ = @verify",
    });

    const wxCmdLineEntryDesc description[] = {
        { wxCMD_LINE_PARAM, nullptr, nullptr, "key=value", wxCMD_LINE_VAL_STRING,
            wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE },
        { wxCMD_LINE_NONE },
    };
    wxCmdLineParser command_line(description, "env.class_id=CliEnv");
    REQUIRE(command_line.Parse(false) == 0);

    const anet::ConfigManager manager(config_path.string(), &command_line);
    CHECK(manager.GetConfigData().Get("env.class_id") == "CliEnv");

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager rejects a named trunk that selects another trunk", "[config][resolver][trunk][error]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-nested-trunk-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "run.@outer.run.$ = @inner",
        "run.@inner.env.class_id = NestedEnv",
        "run.$ = @outer",
    });

    CHECK_THROWS_WITH(
        anet::ConfigManager(config_path.string()),
        Catch::Matchers::ContainsSubstring("named trunk must not select another trunk")
        && Catch::Matchers::ContainsSubstring("material=run.@outer")
        && Catch::Matchers::ContainsSubstring("key=run.@outer.run.$"));

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager rejects an undefined named trunk material", "[config][resolver][trunk][error]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-undefined-trunk-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, { "run.$ = run.@missing" });

    CHECK_THROWS_WITH(
        anet::ConfigManager(config_path.string()),
        Catch::Matchers::ContainsSubstring("material selection target not found")
        && Catch::Matchers::ContainsSubstring("selection=run.$")
        && Catch::Matchers::ContainsSubstring("resolved=run.@missing"));

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager leaves ordinary run keys unchanged without a named trunk", "[config][resolver][trunk]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-no-trunk-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "run.foo = value",
        "env.class_id = AtariEnv",
    });

    const anet::ConfigManager manager(config_path.string());
    CHECK(manager.GetConfigData().Get("run.foo") == "value");
    CHECK(manager.GetConfigData().Get("env.class_id") == "AtariEnv");
    CHECK(manager.GetResolutionJson()["selections"].empty());

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager resolves nested selection copied from material", "[config][resolver]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-nested-material-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "DefaultDQNAgent.@iqn.quantile_mode = iqn",
        "DefaultDQNAgent.@iqn.net.$ = net.@iqn",
        "net.@iqn.body.output.[features] = iqn_fusion",
        "DefaultDQNAgent.$ = @iqn",
    });

    const anet::ConfigManager manager(config_path.string());
    const auto config_data = manager.GetConfigData();
    CHECK(config_data.Get("DefaultDQNAgent.quantile_mode") == "iqn");
    CHECK(config_data.Get("DefaultDQNAgent.net.body.output.[features]") == "iqn_fusion");
    CHECK_FALSE(config_data.Has("DefaultDQNAgent.net.$"));

    const auto selections = manager.GetResolutionJson()["selections"];
    REQUIRE(selections.size() == 2);
    CHECK(selections[0]["key"] == "DefaultDQNAgent.$");
    CHECK(selections[1]["key"] == "DefaultDQNAgent.net.$");
    CHECK(selections[1]["chain"][0]["resolved"] == "net.@iqn");

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager resolves nested selection at the same owner", "[config][resolver]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-same-owner-nested-material-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "Env.@a.$ = @b",
        "Env.@b.value = resolved",
        "Env.$ = @a",
    });

    const anet::ConfigManager manager(config_path.string());
    CHECK(manager.GetConfigData().Get("Env.value") == "resolved");

    const auto selections = manager.GetResolutionJson()["selections"];
    REQUIRE(selections.size() == 2);
    CHECK(selections[0]["chain"][0]["resolved"] == "Env.@a");
    CHECK(selections[1]["chain"][0]["resolved"] == "Env.@b");

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager applies CLI selection and material before effective leaf override", "[config][resolver][cli]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-cli-phases-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "Env.@a.value = file-a",
        "Env.@b.value = file-b",
        "Env.$ = @a",
    });

    const wxCmdLineEntryDesc description[] = {
        { wxCMD_LINE_PARAM, nullptr, nullptr, "key=value", wxCMD_LINE_VAL_STRING,
            wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE },
        { wxCMD_LINE_NONE },
    };
    wxCmdLineParser command_line(
        description,
        "Env.$=@b Env:@b.value=cli-material Env.value=cli-leaf");
    REQUIRE(command_line.Parse(false) == 0);

    const anet::ConfigManager manager(config_path.string(), &command_line);
    const auto config_data = manager.GetConfigData();
    CHECK(config_data.Get("Env.value") == "cli-leaf");
    CHECK_FALSE(config_data.Has("Env.$"));
    CHECK_FALSE(config_data.Has("Env.@b.value"));

    const auto selections = manager.GetResolutionJson()["selections"];
    REQUIRE(selections.size() == 1);
    CHECK(selections[0]["chain"][0]["term"] == "@b");

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager applies a CLI source-prefix leaf before selection", "[config][resolver][cli]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-cli-source-prefix-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "app.batchrun.exp_exit_step = 100",
        "app.$ = app.batchrun",
    });

    const wxCmdLineEntryDesc description[] = {
        { wxCMD_LINE_PARAM, nullptr, nullptr, "key=value", wxCMD_LINE_VAL_STRING,
            wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE },
        { wxCMD_LINE_NONE },
    };
    wxCmdLineParser command_line(description, "app.batchrun.exp_exit_step=200");
    REQUIRE(command_line.Parse(false) == 0);

    const anet::ConfigManager manager(config_path.string(), &command_line);
    const auto config_data = manager.GetConfigData();
    CHECK(config_data.Get("app.exp_exit_step") == "200");
    CHECK(config_data.Get("app.batchrun.exp_exit_step") == "200");

    std::filesystem::remove_all(root);
}

TEST_CASE("Runner app selection resolves the error dialog policy", "[config][resolver][cli]")
{
    const auto repo_root = std::filesystem::path(__FILE__)
        .parent_path().parent_path().parent_path().parent_path();
    const auto config_path = repo_root / "apps" / "runner" / "config" / "common.txt";
    REQUIRE(std::filesystem::exists(config_path));

    const wxCmdLineEntryDesc description[] = {
        { wxCMD_LINE_PARAM, nullptr, nullptr, "key=value", wxCMD_LINE_VAL_STRING,
            wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE },
        { wxCMD_LINE_NONE },
    };

    SECTION("online configuration enables dialogs")
    {
        wxCmdLineParser command_line(description, "app.$=app.online");
        REQUIRE(command_line.Parse(false) == 0);

        const anet::ConfigManager manager(config_path.string(), &command_line);
        CHECK(manager.GetConfigData().Get("app.show_error_dialog") == "true");
    }

    SECTION("batchrun configuration disables dialogs")
    {
        wxCmdLineParser command_line(description, "app.$=app.batchrun");
        REQUIRE(command_line.Parse(false) == 0);

        const anet::ConfigManager manager(config_path.string(), &command_line);
        CHECK(manager.GetConfigData().Get("app.show_error_dialog") == "false");
    }

    SECTION("source-prefix CLI override is applied before selection")
    {
        wxCmdLineParser command_line(
            description, "app.batchrun.show_error_dialog=true app.$=app.batchrun");
        REQUIRE(command_line.Parse(false) == 0);

        const anet::ConfigManager manager(config_path.string(), &command_line);
        const auto config_data = manager.GetConfigData();
        CHECK(config_data.Get("app.show_error_dialog") == "true");
        CHECK(config_data.Get("app.batchrun.show_error_dialog") == "true");
    }
}

TEST_CASE("ConfigManager keeps a literal CLI leaf after value-reference selection", "[config][resolver][reference][cli]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-cli-reference-source-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "@vars.max_exp_step = 100",
        "app.@batchrun.exp_exit_step = ${@vars.max_exp_step}",
        "app.$ = @batchrun",
    });

    const wxCmdLineEntryDesc description[] = {
        { wxCMD_LINE_PARAM, nullptr, nullptr, "key=value", wxCMD_LINE_VAL_STRING,
            wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE },
        { wxCMD_LINE_NONE },
    };
    wxCmdLineParser command_line(description, "app.exp_exit_step=200");
    REQUIRE(command_line.Parse(false) == 0);

    const anet::ConfigManager manager(config_path.string(), &command_line);
    CHECK(manager.GetConfigData().Get("app.exp_exit_step") == "200");
    CHECK(manager.GetResolutionJson()["references"].empty());

    std::filesystem::remove_all(root);
}

TEST_CASE("Properties normalizes config key whitespace and colon sugar", "[config][resolver][properties]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-colon-sugar-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "AtariEnv.@v5 : repeat_action_probability = 0.25",
        "AtariEnv .$ = @v5",
        "metrics.rule = ema_alpha:0.001",
    });

    const anet::ConfigManager manager(config_path.string());
    const auto config_data = manager.GetConfigData();
    CHECK(config_data.Get("AtariEnv.repeat_action_probability") == "0.25");
    CHECK(config_data.Get("metrics.rule") == "ema_alpha:0.001");

    const auto selections = manager.GetResolutionJson()["selections"];
    REQUIRE(selections.size() == 1);
    CHECK(selections[0]["key"] == "AtariEnv.$");
    CHECK(selections[0]["chain"][0]["resolved"] == "AtariEnv.@v5");

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager expands value references after CLI leaf override", "[config][resolver][reference][cli]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-value-reference-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "@vars.max_exp_step = 50000000",
        "app.online.exp_pause_step = ${@vars.max_exp_step}",
        "app.batchrun.exp_exit_step = ${@vars.max_exp_step}",
    });

    const wxCmdLineEntryDesc description[] = {
        { wxCMD_LINE_PARAM, nullptr, nullptr, "key=value", wxCMD_LINE_VAL_STRING,
            wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE },
        { wxCMD_LINE_NONE },
    };
    wxCmdLineParser command_line(description, "@vars.max_exp_step=75000000");
    REQUIRE(command_line.Parse(false) == 0);

    const anet::ConfigManager manager(config_path.string(), &command_line);
    const auto config_data = manager.GetConfigData();
    CHECK(config_data.Get("app.online.exp_pause_step") == "75000000");
    CHECK(config_data.Get("app.batchrun.exp_exit_step") == "75000000");
    CHECK_FALSE(config_data.Has("@vars.max_exp_step"));

    const auto references = manager.GetResolutionJson()["references"];
    REQUIRE(references.size() == 2);
    CHECK(references[0]["source"] == "app.online.exp_pause_step");
    CHECK(references[0]["target"] == "@vars.max_exp_step");
    CHECK(references[0]["value"] == "75000000");
    CHECK(references[1]["source"] == "app.batchrun.exp_exit_step");

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager rejects an undefined material selection", "[config][resolver][error]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-undefined-material-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, { "AtariEnv.$ = @missing" });

    CHECK_THROWS_WITH(
        anet::ConfigManager(config_path.string()),
        Catch::Matchers::ContainsSubstring("selection=AtariEnv.$")
        && Catch::Matchers::ContainsSubstring("term=@missing")
        && Catch::Matchers::ContainsSubstring("resolved=AtariEnv.@missing")
        && Catch::Matchers::ContainsSubstring("scope=AtariEnv"));

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager rejects a material selection cycle", "[config][resolver][error]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-material-cycle-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "Env.@a.$ = @b",
        "Env.@b.$ = @a",
        "Env.$ = @a",
    });

    CHECK_THROWS_WITH(
        anet::ConfigManager(config_path.string()),
        Catch::Matchers::ContainsSubstring("selection cycle detected")
        && Catch::Matchers::ContainsSubstring(
            "path=Env.$ -> Env.@a.$ -> Env.@b.$ -> Env.@a.$"));

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager rejects selection depth over ten", "[config][resolver][error]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-selection-depth-test";
    std::filesystem::remove_all(root);

    const auto config_path = root / "config.txt";
    WriteConfig(config_path, {
        "Env.@deep.child.$ = Layer1",
        "Layer1.child.$ = Layer2",
        "Layer2.child.$ = Layer3",
        "Layer3.child.$ = Layer4",
        "Layer4.child.$ = Layer5",
        "Layer5.child.$ = Layer6",
        "Layer6.child.$ = Layer7",
        "Layer7.child.$ = Layer8",
        "Layer8.child.$ = Layer9",
        "Layer9.child.$ = Layer10",
        "Layer10.child.$ = Layer11",
        "Layer11.value = done",
        "Env.$ = @deep",
    });

    CHECK_THROWS_WITH(
        anet::ConfigManager(config_path.string()),
        Catch::Matchers::ContainsSubstring("selection depth limit exceeded")
        && Catch::Matchers::ContainsSubstring("max=10")
        && Catch::Matchers::ContainsSubstring("Layer9.child.$"));

    std::filesystem::remove_all(root);
}

TEST_CASE("ConfigManager rejects invalid value references", "[config][resolver][reference][error]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-invalid-reference-test";
    std::filesystem::remove_all(root);

    SECTION("undefined target")
    {
        const auto config_path = root / "undefined.txt";
        WriteConfig(config_path, { "app.limit = ${@vars.missing}" });
        CHECK_THROWS_WITH(
            anet::ConfigManager(config_path.string()),
            Catch::Matchers::ContainsSubstring("source=app.limit")
            && Catch::Matchers::ContainsSubstring("target=@vars.missing"));
    }

    SECTION("chained target")
    {
        const auto config_path = root / "chained.txt";
        WriteConfig(config_path, {
            "@vars.base = 10",
            "@vars.indirect = ${@vars.base}",
            "app.limit = ${@vars.indirect}",
        });
        CHECK_THROWS_WITH(
            anet::ConfigManager(config_path.string()),
            Catch::Matchers::ContainsSubstring("chained value reference")
            && Catch::Matchers::ContainsSubstring("target=@vars.indirect"));
    }

    SECTION("earlier effective target is still treated as chained")
    {
        const auto config_path = root / "effective-chained.txt";
        WriteConfig(config_path, {
            "app.inner = ${@vars.base}",
            "app.outer = ${app.inner}",
            "@vars.base = 10",
        });
        CHECK_THROWS_WITH(
            anet::ConfigManager(config_path.string()),
            Catch::Matchers::ContainsSubstring("chained value reference")
            && Catch::Matchers::ContainsSubstring("target=app.inner"));
    }

    SECTION("unclosed token")
    {
        const auto config_path = root / "unclosed.txt";
        WriteConfig(config_path, { "app.limit = ${@vars.base" });
        CHECK_THROWS_WITH(
            anet::ConfigManager(config_path.string()),
            Catch::Matchers::ContainsSubstring("unresolved value reference token")
            && Catch::Matchers::ContainsSubstring("source=app.limit"));
    }

    std::filesystem::remove_all(root);
}

TEST_CASE("Properties rejects invalid colon sugar", "[config][resolver][properties][error]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "config-manager-invalid-colon-test";
    std::filesystem::remove_all(root);

    SECTION("multiple separators")
    {
        const auto config_path = root / "multiple.txt";
        WriteConfig(config_path, { "Env:material:value = 1" });
        CHECK_THROWS_WITH(
            anet::ConfigManager(config_path.string()),
            Catch::Matchers::ContainsSubstring("multiple ':' separators")
            && Catch::Matchers::ContainsSubstring("key=Env:material:value"));
    }

    SECTION("empty segment")
    {
        const auto config_path = root / "empty.txt";
        WriteConfig(config_path, { "Env: = value" });
        CHECK_THROWS_WITH(
            anet::ConfigManager(config_path.string()),
            Catch::Matchers::ContainsSubstring("empty ':' segment")
            && Catch::Matchers::ContainsSubstring("key=Env:"));
    }

    std::filesystem::remove_all(root);
}

// 凍結旧 AutoMerge との golden 比較は PH0/PH1a の移行検証として役目を終えた(素材 `@` 化後は
// 旧 AutoMerge が相対参照を解決できず oracle が成立しない)。以後の回帰は resolver 単体テスト群が守る。

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
