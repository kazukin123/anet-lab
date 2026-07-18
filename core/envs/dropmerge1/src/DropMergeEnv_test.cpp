// DropMergeEnv_test.cpp

#include "DropMergeEnv.hpp"

#include <clocale>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>

#include "anet/catch_test.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/test_util.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

void SetupUtf8Console()
{
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::setlocale(LC_CTYPE, ".UTF-8");
}

class NoopMetricsBackend final : public anet::IBackend {
public:
    void Open(const std::filesystem::path&, const std::string&) override {}
    void WriteJsonl(const anet::json&) override {}
    void Flush() override {}
};

class ScopedNoopMetricsLogger final {
public:
    ScopedNoopMetricsLogger()
    {
        anet::MetricsLogger::Reset();
        anet::MetricsLoggerConfig config;
        config.run_name_tmpl = "dropmerge_env_test";
        anet::MetricsLogger::Init(std::make_unique<NoopMetricsBackend>(), config, "C:/tmp");
    }

    ~ScopedNoopMetricsLogger()
    {
        anet::MetricsLogger::Reset();
    }
};

int CountOccurrences(const std::string& value, const std::string& needle)
{
    int count = 0;
    std::string::size_type position = 0;
    while ((position = value.find(needle, position)) != std::string::npos) {
        ++count;
        position += needle.size();
    }
    return count;
}

} // namespace

TEST_CASE("DropMergeEnv prefixes maximum-step log with its name once", "[dropmerge][env_name]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::rl::env::drop_merge::DropMergeEnvConfig config;
    config.max_step = 1;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-log[0]", 123);
    env->Reset(anet::rl::RunMode::Train);

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    env->Step(anet::rl::env::drop_merge::kActionNoop, anet::rl::RunMode::Train);
    logs.Flush();

    int matching_records = 0;
    for (const auto& record : logs.Records()) {
        if (record.message.find("Episode truncated. Maximum step count exceeded.") == std::string::npos) {
            continue;
        }
        ++matching_records;
        CHECK(record.level == wxLOG_Info);
        CHECK(record.message.find("[dropmerge-log[0]] Episode truncated. Maximum step count exceeded.") == 0);
        CHECK(CountOccurrences(record.message, "[dropmerge-log[0]]") == 1);
    }
    CHECK(matching_records == 1);
}

int main(int argc, char* argv[])
{
    SetupUtf8Console();

    anet::test::PreparedTestArgs test_args;
    try {
        test_args = anet::test::PrepareTestArgs(argc, argv);
    } catch (const std::exception& e) {
        return anet::test::ReportTestArgsError(e);
    }
    anet::test::SetupTestFailureDialog(test_args.failure_dialog_enabled);

    Catch::Session session;
    session.configData().showDurations = Catch::ShowDurations::Always;
    return session.run(test_args.Argc(), test_args.Argv());
}
