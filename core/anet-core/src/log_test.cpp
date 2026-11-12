#include "anet/catch_test.hpp"

#include "anet/log.hpp"
#include "anet/test_util.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

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

static_assert(!std::is_copy_assignable_v<anet::log::Logger>);
static_assert(!std::is_move_assignable_v<anet::log::Logger>);

TEST_CASE("Run notifications occur once per source and target across threads", "[log][prd117]")
{
    const auto path = std::filesystem::current_path() / ".scratch/prd117/log-once/run.log";
    std::filesystem::create_directories(path.parent_path());
    const auto emit = [](bool info, const std::string& key, int step) {
        if (info) {
            anet::log::info_once(key) << "once info key=" << key << " value=off step=" << step << " expected=off";
        } else {
            anet::log::warn_once(key) << "once warning key=" << key << " value=" << step << " expected=99";
        }
    };
    const auto emit_default = [](int value) {
        anet::log::warn_once() << "once default key=all value=" << value << " expected=99";
    };
    const auto count = [](const std::string& text, const std::string& marker) {
        size_t result = 0;
        size_t cursor = 0;
        while ((cursor = text.find(marker, cursor)) != std::string::npos) {
            ++result;
            cursor += marker.size();
        }
        return result;
    };
    ScopedFileLogger logger(path);
    anet::log::ResetNotifications();
    for (const auto& key : {std::string(), std::string("same"), std::string("other")}) {
        for (int i = 0; i < 3; ++i) {
            emit(false, key, i);
            emit(true, key, i);
        }
    }
    emit_default(1);
    emit_default(2);

    // 同じ出力先へ並行に届く通知も、Run の同じ対象として数える。
    std::vector<std::thread> workers;
    for (int i = 0; i < 8; ++i) {
        workers.emplace_back([emit, i] {
            emit(false, "threaded", i);
            emit(true, "threaded", i);
        });
    }
    for (auto& worker : workers) worker.join();
    wxLog::FlushActive();
    auto text = ReadTextFile(path);
    CHECK(count(text, "[W] ") == 5);
    CHECK(count(text, "[I] ") == 4);
    CHECK(count(text, "\n") == 9);

    // 出力先を使い続けても、次の Run では同じ箇所と対象が再び 1 件出る。
    anet::log::ResetNotifications();
    for (int i = 0; i < 2; ++i) {
        emit(false, "same", i);
        emit(true, "same", i);
        emit_default(i);
    }
    wxLog::FlushActive();
    text = ReadTextFile(path);
    CHECK(count(text, "[W] ") == 7);
    CHECK(count(text, "[I] ") == 5);
    CHECK(count(text, "\n") == 12);
}

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

TEST_CASE("Logger prefixes an info message exactly once", "[log][logger]")
{
    anet::test::LogCaptureGuard logs;
    anet::log::Logger logger("env.[train].[0]: ");

    logger.info() << "body";
    logs.Flush();

    REQUIRE(logs.Records().size() == 1);
    CHECK(logs.Records()[0].level == wxLOG_Message);
    CHECK(logs.Records()[0].message == "env.[train].[0]: body");
}

TEST_CASE("Logger level methods preserve wx levels and prefix", "[log][logger]")
{
    anet::test::LogCaptureGuard logs;
    anet::log::Logger logger("env.[eval]: ");

    logger.verbose() << "verbose-body";
    logger.warn() << "warn-body";
    logger.error() << "error-body";
    logs.Flush();

    REQUIRE(logs.Records().size() == 3);
    CHECK(logs.Records()[0].level == wxLOG_Info);
    CHECK(logs.Records()[0].message == "env.[eval]: verbose-body");
    CHECK(logs.Records()[1].level == wxLOG_Warning);
    CHECK(logs.Records()[1].message == "env.[eval]: warn-body");
    CHECK(logs.Records()[2].level == wxLOG_Error);
    CHECK(logs.Records()[2].message == "env.[eval]: error-body");
}

TEST_CASE("Logger keeps its construction prefix and defaults to no prefix", "[log][logger]")
{
    anet::log::Logger prefixed("env.[train]: ");
    anet::log::Logger unprefixed;

    CHECK(prefixed.prefix() == "env.[train]: ");
    CHECK(unprefixed.prefix().empty());

    anet::test::LogCaptureGuard logs;
    unprefixed.info() << "body";
    logs.Flush();

    REQUIRE(logs.Records().size() == 1);
    CHECK(logs.Records()[0].message == "body");
}
