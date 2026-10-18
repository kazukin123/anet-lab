// anet/app_util.hpp

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "anet/config.hpp"

namespace anet {

    std::filesystem::path GetExecutablePath();
    std::filesystem::path GetExecutableDir();
    std::filesystem::path GetExecutableRootDir();
    std::filesystem::path GetExecutableConfigDir();
    std::filesystem::path GetAppDataDir();

    struct WorkspacePaths {
        std::string input;
        std::filesystem::path root;
        std::filesystem::path config_file;
        std::filesystem::path runs_dir;
        std::string runs_config_value;
    };

    enum class AppConfigMode {
        DirectConfig,
        ExplicitWorkspace,
        WorkspaceFlow,
    };

    AppConfigMode DetermineAppConfigMode(
        bool has_config,
        bool has_workspace,
        bool force_workspace_selection);

    class WorkspaceService final {
    public:
        WorkspaceService(
            std::filesystem::path runner_root,
            std::filesystem::path app_data_dir);

        std::optional<std::string> GetNewWorkspaceNameError(
            const std::string& raw_input) const;
        WorkspacePaths Resolve(const std::string& raw_input, bool allow_create) const;
        bool IsResolvable(const std::string& raw_input) const;
        std::vector<std::string> ScanLocalWorkspaces() const;

        std::vector<std::string> LoadHistory() const;
        void RecordHistory(const std::string& adopted_input) const;
        void SaveLastWorkspace(const WorkspacePaths& workspace) const;

        const std::filesystem::path& RunnerRoot() const { return runner_root_; }
        const std::filesystem::path& AppDataDir() const { return app_data_dir_; }

    private:
        std::string ValidateAndTrim(const std::string& raw_input) const;
        void CreateWorkspace(const std::filesystem::path& root) const;

        std::filesystem::path runner_root_;
        std::filesystem::path app_data_dir_;
    };

    std::unique_ptr<ConfigManager> CreateWorkspaceConfigManager(
        const WorkspacePaths& workspace,
        const std::filesystem::path& common_config_dir,
        const wxCmdLineParser* cmd_line);

    void ValidateWorkspaceRunsDir(
        const ConfigData& config_data,
        const WorkspacePaths& workspace);

    class StandardStreamLogger final {
    public:
        StandardStreamLogger();
        ~StandardStreamLogger();

        StandardStreamLogger(const StandardStreamLogger&) = delete;
        StandardStreamLogger& operator=(const StandardStreamLogger&) = delete;

        void Start(const std::filesystem::path& run_dir);
        void LogStatus() const;
        void Flush() const;
        void Stop();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace anet
