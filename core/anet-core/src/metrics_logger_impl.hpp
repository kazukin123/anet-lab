#pragma once

#include <filesystem>
#include <optional>
#include <ostream>
#include <string_view>

namespace anet::detail {

enum class JsonlIoOperation {
    kWrite,
    kFlush,
};

struct JsonlIoFailure {
    std::filesystem::path path;
    JsonlIoOperation operation;
    bool fail;
    bool bad;
};

std::optional<JsonlIoFailure> WriteJsonlLine(
    std::ostream& stream,
    std::string_view line,
    const std::filesystem::path& path,
    bool& error_reported);

std::optional<JsonlIoFailure> FlushJsonl(
    std::ostream& stream,
    const std::filesystem::path& path,
    bool& error_reported);

void LogJsonlIoFailure(const JsonlIoFailure& failure);

} // namespace anet::detail
