#pragma once

#include <filesystem>
#include <optional>
#include <string>

class wxWindow;

namespace anet {

    class WorkspaceService;

} // namespace anet

namespace anet::runner {

    struct WorkspaceDialogResult {
        std::string input;
        bool skip_dialog = false;
    };

    bool LoadWorkspaceDialogSkip(const std::filesystem::path& app_data_dir);
    void SaveWorkspaceDialogSkip(const std::filesystem::path& app_data_dir, bool skip);

    std::optional<WorkspaceDialogResult> ShowWorkspaceDialog(
        wxWindow* parent,
        const WorkspaceService& service);

} // namespace anet::runner
