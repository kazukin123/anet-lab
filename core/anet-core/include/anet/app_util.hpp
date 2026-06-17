// anet/app_util.hpp

#pragma once

#include <filesystem>
#include <memory>

namespace anet {

std::filesystem::path GetExecutablePath();
std::filesystem::path GetExecutableDir();
std::filesystem::path GetExecutableRootDir();
std::filesystem::path GetExecutableConfigDir();

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
