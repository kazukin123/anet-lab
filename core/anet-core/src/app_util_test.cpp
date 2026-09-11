#include "anet/catch_test.hpp"

#include "anet/app_util.hpp"
#include "app_util_impl.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <wx/cmdline.h>

static void WriteWorkspaceTestFile(const std::filesystem::path& path, const std::string& text)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << text;
}

static std::string ReadWorkspaceTestFile(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    return { std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>() };
}

static std::string WorkspaceTestPathToUtf8(const std::filesystem::path& path)
{
    const auto text = path.u8string();
    return { reinterpret_cast<const char*>(text.data()), text.size() };
}

TEST_CASE("app_util exposes executable-based config directory", "[app_util]")
{
    const auto exe_path = anet::GetExecutablePath();

    CHECK_FALSE(exe_path.empty());
    CHECK(anet::GetExecutableDir() == exe_path.parent_path());
    CHECK(anet::GetExecutableConfigDir() == anet::GetExecutableRootDir() / "config");
}

TEST_CASE("app_util resolves portable and user app-data directories", "[app_util]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "app-util-data-dir-test";
    std::filesystem::remove_all(root);
    const auto executable_root = root / "runner";
    const auto user_root = root / "user";

    SECTION("existing portable appdata wins")
    {
        std::filesystem::create_directories(executable_root / "appdata");
        CHECK(anet::internal::ResolveAppDataDir(executable_root, user_root) == executable_root / "appdata");
        CHECK_FALSE(std::filesystem::exists(user_root));
    }

    SECTION("user mode creates the application directory")
    {
        const auto expected = user_root / "anet-lab" / "runner";
        CHECK(anet::internal::ResolveAppDataDir(executable_root, user_root) == expected);
        CHECK(std::filesystem::is_directory(expected));
    }

    SECTION("missing user root fails when portable mode is unavailable")
    {
        CHECK_THROWS_WITH(
            anet::internal::ResolveAppDataDir(executable_root, std::nullopt),
            Catch::Matchers::ContainsSubstring("user configuration directory is unavailable"));
    }

    std::filesystem::remove_all(root);
}

TEST_CASE("WorkspaceService trims creates and resolves a local workspace", "[workspace]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "app-util-workspace-resolve-test";
    std::filesystem::remove_all(root);
    WriteWorkspaceTestFile(root / "runner" / "config" / "_workspace_template.txt", "$include <DropMerge.txt>\n");
    anet::WorkspaceService service(root / "runner", root / "appdata");

    const auto workspace = service.Resolve("  trial  ", true);

    CHECK(workspace.input == "trial");
    CHECK(workspace.root == std::filesystem::absolute(root / "runner" / "workspaces" / "trial"));
    CHECK(workspace.runs_config_value == (std::filesystem::path("workspaces") / "trial" / "runs").string());
    CHECK(std::filesystem::is_regular_file(workspace.config_file));
    CHECK(std::filesystem::is_directory(workspace.runs_dir));
    std::filesystem::remove_all(root);
}

TEST_CASE("WorkspaceService validates new workspace names without side effects", "[workspace][validation]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "app-util-workspace-name-validation-test";
    std::filesystem::remove_all(root);
    anet::WorkspaceService service(root / "runner", root / "appdata");

    // 新規名として許可する単一の相対名は、検証だけでは filesystem を変更しない。
    const auto utf8_name = WorkspaceTestPathToUtf8(std::filesystem::path(u8"実験 workspace"));
    for (const auto& input : { std::string("trial"), std::string("  trial  "), utf8_name }) {
        CHECK_FALSE(service.GetNewWorkspaceNameError(input).has_value());
    }

    // workspace path の禁止形式と、新規作成で許可しない path 形式を同じ境界で拒否する。
    for (const auto& input : {
        std::string(""),
        std::string("   "),
        std::string("bad#name"),
        std::string("bad//name"),
        std::string("bad;"),
        std::string(R"(\\server\share)"),
        std::string(R"(//server/share)"),
        std::string("nested/name"),
        std::string(R"(nested\name)"),
        std::string("."),
        std::string(".."),
        std::string("C:relative"),
        WorkspaceTestPathToUtf8(std::filesystem::absolute(root / "missing")),
    }) {
        CHECK(service.GetNewWorkspaceNameError(input).has_value());
    }
    CHECK_FALSE(std::filesystem::exists(root));
}

TEST_CASE("WorkspaceService validates path and creation boundaries", "[workspace]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "app-util-workspace-validation-test";
    std::filesystem::remove_all(root);
    WriteWorkspaceTestFile(root / "runner" / "config" / "_workspace_template.txt", "env.class_id = DropMerge\n");
    anet::WorkspaceService service(root / "runner", root / "appdata");

    for (const auto& input : {
        "",
        "bad#name",
        "bad//name",
        "bad;",
        R"(\\server\share)",
        R"(//server/share)",
        R"(\/server\share)",
        R"(/\server/share)",
        "nested/name",
        R"(nested\name)",
        "C:relative",
    }) {
        CHECK_THROWS(service.Resolve(input, true));
    }
    CHECK_FALSE(std::filesystem::exists(root / "runner" / "workspaces" / "nested"));
    CHECK_THROWS(service.Resolve((root / "absolute-missing").string(), true));

    std::filesystem::remove_all(root);
}

TEST_CASE("Workspace availability checks do not create the runs directory", "[workspace][availability]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "app-util-workspace-availability-test";
    std::filesystem::remove_all(root);
    WriteWorkspaceTestFile(root / "runner" / "workspaces" / "existing" / "config" / "_main.txt", "# test\n");
    anet::WorkspaceService service(root / "runner", root / "appdata");

    CHECK(service.IsResolvable("existing"));
    CHECK_FALSE(std::filesystem::exists(root / "runner" / "workspaces" / "existing" / "runs"));
    std::filesystem::remove_all(root);
}

TEST_CASE("WorkspaceService initializes an existing archived local workspace", "[workspace][archive]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "app-util-workspace-archive-init-test";
    std::filesystem::remove_all(root);
    WriteWorkspaceTestFile(root / "runner" / "config" / "_workspace_template.txt", "env.class_id = DropMerge\n");
    const auto workspace_root = root / "runner" / "workspaces" / "archive";
    WriteWorkspaceTestFile(workspace_root / "runs" / "old_run" / "artifact.txt", "preserve\n");
    WriteWorkspaceTestFile(workspace_root / "config" / "notes.txt", "preserve config note\n");
    anet::WorkspaceService service(root / "runner", root / "appdata");

    REQUIRE(service.ScanLocalWorkspaces() == std::vector<std::string>{ "archive" });
    CHECK(service.IsResolvable("archive"));
    CHECK_FALSE(std::filesystem::exists(workspace_root / "config" / "_main.txt"));

    const auto workspace = service.Resolve("archive", true);

    CHECK(workspace.root == std::filesystem::absolute(workspace_root));
    CHECK(ReadWorkspaceTestFile(workspace.config_file) == "env.class_id = DropMerge\n");
    CHECK(ReadWorkspaceTestFile(workspace_root / "runs" / "old_run" / "artifact.txt") == "preserve\n");
    CHECK(ReadWorkspaceTestFile(workspace_root / "config" / "notes.txt") == "preserve config note\n");
    std::filesystem::remove_all(root);
}

TEST_CASE("WorkspaceService scans every local workspace directory", "[workspace][archive]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "app-util-workspace-scan-all-test";
    std::filesystem::remove_all(root);
    const auto workspaces_root = root / "runner" / "workspaces";
    WriteWorkspaceTestFile(workspaces_root / "complete" / "config" / "_main.txt", "# complete\n");
    WriteWorkspaceTestFile(workspaces_root / "archive" / "runs" / "old_run" / "metrics.jsonl", "{}\n");
    std::filesystem::create_directories(workspaces_root / "empty");
    WriteWorkspaceTestFile(workspaces_root / "note.txt", "not a workspace directory\n");
    anet::WorkspaceService service(root / "runner", root / "appdata");

    CHECK(service.ScanLocalWorkspaces()
        == std::vector<std::string>{ "archive", "complete", "empty" });
    std::filesystem::remove_all(root);
}

TEST_CASE("WorkspaceService initializes an existing absolute workspace", "[workspace][archive][utf8]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "app-util-workspace-absolute-init-test";
    std::filesystem::remove_all(root);
    WriteWorkspaceTestFile(root / "runner" / "config" / "_workspace_template.txt", "env.class_id = DropMerge\n");
    const auto workspace_root = root / std::filesystem::path(u8"過去 workspace");
    WriteWorkspaceTestFile(workspace_root / "archive.txt", "preserve\n");
    anet::WorkspaceService service(root / "runner", root / "appdata");
    const auto input = WorkspaceTestPathToUtf8(std::filesystem::absolute(workspace_root));

    CHECK(service.IsResolvable(input));
    CHECK_FALSE(std::filesystem::exists(workspace_root / "config" / "_main.txt"));
    CHECK_FALSE(std::filesystem::exists(workspace_root / "runs"));

    const auto workspace = service.Resolve(input, true);

    CHECK(workspace.root == std::filesystem::absolute(workspace_root));
    CHECK(ReadWorkspaceTestFile(workspace.config_file) == "env.class_id = DropMerge\n");
    CHECK(ReadWorkspaceTestFile(workspace_root / "archive.txt") == "preserve\n");
    CHECK(std::filesystem::is_directory(workspace.runs_dir));
    std::filesystem::remove_all(root);
}

TEST_CASE("WorkspaceService preserves an existing workspace config", "[workspace][archive]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "app-util-workspace-existing-config-test";
    std::filesystem::remove_all(root);
    WriteWorkspaceTestFile(root / "runner" / "config" / "_workspace_template.txt", "template.value = new\n");
    const auto workspace_root = root / "runner" / "workspaces" / "configured";
    WriteWorkspaceTestFile(workspace_root / "config" / "_main.txt", "workspace.value = existing\n");
    anet::WorkspaceService service(root / "runner", root / "appdata");

    const auto workspace = service.Resolve("configured", true);

    CHECK(ReadWorkspaceTestFile(workspace.config_file) == "workspace.value = existing\n");
    CHECK(std::filesystem::is_directory(workspace.runs_dir));
    std::filesystem::remove_all(root);
}

TEST_CASE("WorkspaceService rejects unsafe archive initialization without side effects", "[workspace][archive]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "app-util-workspace-archive-init-failure-test";
    std::filesystem::remove_all(root);
    const auto workspace_root = root / "runner" / "workspaces" / "archive";
    std::filesystem::create_directories(workspace_root);
    anet::WorkspaceService service(root / "runner", root / "appdata");

    SECTION("creation is disabled")
    {
        WriteWorkspaceTestFile(root / "runner" / "config" / "_workspace_template.txt", "env.class_id = DropMerge\n");

        CHECK_THROWS(service.Resolve("archive", false));
        CHECK_FALSE(std::filesystem::exists(workspace_root / "config"));
        CHECK_FALSE(std::filesystem::exists(workspace_root / "runs"));
    }

    SECTION("template is missing")
    {
        CHECK_FALSE(service.IsResolvable("archive"));
        CHECK_THROWS(service.Resolve("archive", true));
        CHECK_FALSE(std::filesystem::exists(workspace_root / "config"));
        CHECK_FALSE(std::filesystem::exists(workspace_root / "runs"));
    }

    SECTION("config path has an invalid type")
    {
        WriteWorkspaceTestFile(root / "runner" / "config" / "_workspace_template.txt", "env.class_id = DropMerge\n");
        std::filesystem::create_directories(workspace_root / "config" / "_main.txt");

        CHECK_FALSE(service.IsResolvable("archive"));
        CHECK_THROWS(service.Resolve("archive", true));
        CHECK_FALSE(std::filesystem::exists(workspace_root / "runs"));
    }

    std::filesystem::remove_all(root);
}

TEST_CASE("WorkspaceService persists MRU and last workspace", "[workspace]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "app-util-workspace-state-test";
    std::filesystem::remove_all(root);
    WriteWorkspaceTestFile(root / "runner" / "config" / "_workspace_template.txt", "env.class_id = DropMerge\n");
    anet::WorkspaceService service(root / "runner", root / "appdata");

    for (int i = 0; i < 12; ++i) {
        service.RecordHistory("workspace " + std::to_string(i));
    }
    service.RecordHistory("workspace 5");
    const auto history = service.LoadHistory();
    REQUIRE(history.size() == 10);
    CHECK(history.front() == "workspace 5");
    CHECK(std::count(history.begin(), history.end(), "workspace 5") == 1);

    const auto workspace = service.Resolve("current", true);
    service.SaveLastWorkspace(workspace);
    std::string saved;
    {
        std::ifstream ifs(root / "appdata" / "last_workspace.txt");
        std::getline(ifs, saved);
    }
    CHECK(std::filesystem::path(saved) == workspace.root);
    std::filesystem::remove_all(root);
}

TEST_CASE("WorkspaceService preserves UTF-8 absolute paths in config history and launcher state", "[workspace][utf8]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        std::filesystem::path(u8"app-util-workspace-日本語-test");
    std::filesystem::remove_all(root);
    const auto workspace_root = root / std::filesystem::path(u8"外部 workspace");
    WriteWorkspaceTestFile(workspace_root / "config" / "_main.txt", "workspace.value = valid\n");
    std::filesystem::create_directories(workspace_root / "runs");
    anet::WorkspaceService service(root / "runner", root / "appdata");

    const auto input = WorkspaceTestPathToUtf8(workspace_root);
    const auto workspace = service.Resolve(input, false);
    service.RecordHistory(input);
    service.SaveLastWorkspace(workspace);

    CHECK(workspace.root == std::filesystem::absolute(workspace_root));
    CHECK(workspace.runs_config_value == WorkspaceTestPathToUtf8(std::filesystem::absolute(workspace_root / "runs")));
    CHECK(service.LoadHistory() == std::vector<std::string>{ input });
    std::string saved;
    {
        std::ifstream ifs(root / "appdata" / "last_workspace.txt", std::ios::binary);
        saved.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    }
    CHECK(saved == WorkspaceTestPathToUtf8(std::filesystem::absolute(workspace_root)));
    std::filesystem::remove_all(root);
}

TEST_CASE("WorkspaceService skips malformed history entries without discarding valid entries", "[workspace][history]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "app-util-workspace-invalid-history-test";
    std::filesystem::remove_all(root);
    WriteWorkspaceTestFile(root / "appdata" / "history.txt",
        "workspace.history.0 = ;\n"
        "workspace.history.1 = valid\n"
        "workspace.history.2 = \\\\server\\share\n");
    anet::WorkspaceService service(root / "runner", root / "appdata");

    std::vector<std::string> history;
    CHECK_NOTHROW(history = service.LoadHistory());
    CHECK(history == std::vector<std::string>{ "valid" });
    std::filesystem::remove_all(root);
}

TEST_CASE("Workspace config enforces derived runs directory after AutoMerge", "[workspace]")
{
    anet::WorkspacePaths workspace{
        .input = "trial",
        .root = "trial",
        .config_file = "trial/config/_main.txt",
        .runs_dir = "trial/runs",
        .runs_config_value = R"(workspaces\trial\runs)",
    };
    anet::ConfigData config_data;
    config_data.Set("app.runs_dir", workspace.runs_config_value);
    CHECK_NOTHROW(anet::ValidateWorkspaceRunsDir(config_data, workspace));

    config_data.Set("app.runs_dir", "other-runs");
    CHECK_THROWS_WITH(
        anet::ValidateWorkspaceRunsDir(config_data, workspace),
        Catch::Matchers::ContainsSubstring("Workspace config changed app.runs_dir"));
}

TEST_CASE("Workspace config rejects direct indirect and CLI runs overrides", "[workspace]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        "app-util-workspace-runs-invariant-test";
    std::filesystem::remove_all(root);
    const auto common_dir = root / "runner" / "config";
    const auto workspace_root = root / "runner" / "workspaces" / "trial";
    const auto workspace_config = workspace_root / "config" / "_main.txt";
    WriteWorkspaceTestFile(common_dir / "_main.txt", "app.runs_dir = legacy-runs\n");
    WriteWorkspaceTestFile(workspace_config, "workspace.value = valid\n");
    std::filesystem::create_directories(workspace_root / "runs");

    const anet::WorkspacePaths workspace{
        .input = "trial",
        .root = std::filesystem::absolute(workspace_root),
        .config_file = std::filesystem::absolute(workspace_config),
        .runs_dir = std::filesystem::absolute(workspace_root / "runs"),
        .runs_config_value = (std::filesystem::path("workspaces") / "trial" / "runs").string(),
    };

    CHECK_NOTHROW(anet::CreateWorkspaceConfigManager(workspace, common_dir, nullptr));

    WriteWorkspaceTestFile(workspace_config, "app.runs_dir = direct-override\n");
    CHECK_THROWS(anet::CreateWorkspaceConfigManager(workspace, common_dir, nullptr));

    WriteWorkspaceTestFile(workspace_config,
        "app.$ = app.online\n"
        "app.online.runs_dir = indirect-override\n");
    CHECK_THROWS(anet::CreateWorkspaceConfigManager(workspace, common_dir, nullptr));

    WriteWorkspaceTestFile(workspace_config, "workspace.value = valid\n");
    const wxCmdLineEntryDesc command_line_description[] = {
        { wxCMD_LINE_PARAM, nullptr, nullptr, "key=value", wxCMD_LINE_VAL_STRING,
            wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE },
        { wxCMD_LINE_NONE },
    };
    wxCmdLineParser command_line(command_line_description, "app.runs_dir=cli-override");
    REQUIRE(command_line.Parse(false) == 0);
    CHECK_THROWS(anet::CreateWorkspaceConfigManager(workspace, common_dir, &command_line));

    std::filesystem::remove_all(root);
}

TEST_CASE("Application config mode bypasses workspace flow for direct config", "[workspace][cli]")
{
    using anet::AppConfigMode;
    using anet::DetermineAppConfigMode;

    CHECK(DetermineAppConfigMode(false, false, false) == AppConfigMode::WorkspaceFlow);
    CHECK(DetermineAppConfigMode(false, false, true) == AppConfigMode::WorkspaceFlow);
    CHECK(DetermineAppConfigMode(true, false, false) == AppConfigMode::DirectConfig);
    CHECK(DetermineAppConfigMode(false, true, false) == AppConfigMode::ExplicitWorkspace);
    CHECK_THROWS_WITH(
        DetermineAppConfigMode(true, true, false),
        Catch::Matchers::ContainsSubstring("--config, --workspace"));
    CHECK_THROWS_WITH(
        DetermineAppConfigMode(true, false, true),
        Catch::Matchers::ContainsSubstring("--config, --select-workspace"));
    CHECK_THROWS_WITH(
        DetermineAppConfigMode(false, true, true),
        Catch::Matchers::ContainsSubstring("--workspace, --select-workspace"));
}
