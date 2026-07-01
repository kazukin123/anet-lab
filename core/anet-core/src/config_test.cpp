#include "catch.hpp"

#include "anet/config.hpp"
#include "anet/test_util.hpp"

#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string_view>
#include <vector>

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

} // namespace

TEST_CASE("ConfigData Read warns for present invalid values only", "[config]")
{
    anet::ConfigData config_data;
    config_data.Set("invalid_int", "not-int");
    config_data.Set("invalid_bool", "maybe");
    config_data.Set("invalid_float_vector", "1.0 x");
    config_data.Set("valid_int", "42");

    SECTION("invalid int warns and returns the default")
    {
        LogCaptureGuard logs;

        int value = 5;
        CHECK_FALSE(config_data.Read("invalid_int", value, 9));
        CHECK(value == 9);

        logs.Flush();
        CHECK(CountRecords(logs.Records(), wxLOG_Warning) == 1);
        CHECK(HasRecordContaining(
            logs.Records(),
            wxLOG_Warning,
            { "ConfigData::Read failed", "key=invalid_int", "value=\"not-int\"", "expected=int" }));
    }

    SECTION("invalid bool warns and returns the default")
    {
        LogCaptureGuard logs;

        bool value = true;
        CHECK_FALSE(config_data.Read("invalid_bool", value, false));
        CHECK_FALSE(value);

        logs.Flush();
        CHECK(CountRecords(logs.Records(), wxLOG_Warning) == 1);
        CHECK(HasRecordContaining(
            logs.Records(),
            wxLOG_Warning,
            { "ConfigData::Read failed", "key=invalid_bool", "value=\"maybe\"", "expected=bool" }));
    }

    SECTION("invalid vector warns and returns the default")
    {
        LogCaptureGuard logs;

        std::vector<float> value = { 9.0f };
        const std::vector<float> default_value = { 1.0f, 2.0f };
        CHECK_FALSE(config_data.Read("invalid_float_vector", value, default_value));
        CHECK(value == default_value);

        logs.Flush();
        CHECK(CountRecords(logs.Records(), wxLOG_Warning) == 1);
        CHECK(HasRecordContaining(
            logs.Records(),
            wxLOG_Warning,
            { "ConfigData::Read failed", "key=invalid_float_vector", "value=\"1.0 x\"", "expected=float vector" }));
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
