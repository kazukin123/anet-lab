#pragma once

#include <filesystem>
#include <optional>

namespace anet::internal {

    std::filesystem::path ResolveAppDataDir(
        const std::filesystem::path& executable_root,
        const std::optional<std::filesystem::path>& user_config_root);

} // namespace anet::internal
