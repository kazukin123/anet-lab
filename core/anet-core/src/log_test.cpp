#include "catch.hpp"

#include "anet/log.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

std::string ReadTextFile(const std::filesystem::path& path)
{
    std::ifstream ifs(path);
    return std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
}

class ScopedFileLogger final {
public:
    explicit ScopedFileLogger(const std::filesystem::path& path)
    {
        FILE* const file = std::fopen(path.string().c_str(), "wb");
        if (file == nullptr) {
            throw std::runtime_error("Failed to open test log file.");
        }

        old_log_level_ = wxLog::GetLogLevel();
        logger_ = new anet::log::FileLogger(file);
        logger_->SetFormatter(new anet::log::LogFormatter(/*enable_timestamp=*/false));
        old_target_ = wxLog::SetActiveTarget(logger_);
        wxLog::SetLogLevel(wxLOG_Max);
    }

    ~ScopedFileLogger()
    {
        wxLog::FlushActive();
        wxLog* const logger = wxLog::SetActiveTarget(old_target_);
        wxLog::SetLogLevel(old_log_level_);
        delete logger;
    }

    ScopedFileLogger(const ScopedFileLogger&) = delete;
    ScopedFileLogger& operator=(const ScopedFileLogger&) = delete;

private:
    wxLog* old_target_ = nullptr;
    wxLogLevel old_log_level_ = wxLOG_Max;
    anet::log::FileLogger* logger_ = nullptr;
};

} // namespace

TEST_CASE("FileLogger flushes main and worker thread info messages", "[log]")
{
    const auto case_id = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" /
        ("anet-core-file-logger-test-" + std::to_string(case_id));
    const auto log_path = root / "run.log";
    std::filesystem::create_directories(root);

    {
        ScopedFileLogger logger(log_path);

        anet::log::info() << "main-thread-marker";
        wxLog::FlushActive();
        CHECK(ReadTextFile(log_path).find("main-thread-marker") != std::string::npos);

        std::thread worker([] {
            anet::log::info() << "worker-thread-marker";
        });
        worker.join();

        wxLog::FlushActive();
        CHECK(ReadTextFile(log_path).find("worker-thread-marker") != std::string::npos);
    }

    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
}
