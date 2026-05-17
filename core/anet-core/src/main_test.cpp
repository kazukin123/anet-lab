#include "catch.hpp"

#include "anet/log.hpp"

#ifdef _WIN32
#include <windows.h>
#endif
#include <clocale>
#include <cstdio>

namespace {

void SetupUtf8Console()
{
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::setlocale(LC_CTYPE, ".UTF-8");
}

class TestLogTargetGuard {
public:
    TestLogTargetGuard()
    {
        log_target_ = new wxLogStderr(stderr);
        delete log_target_->SetFormatter(new anet::log::LogFormatter(/*enable_timestamp=*/false));
        old_log_target_ = wxLog::SetActiveTarget(log_target_);

#if ANET_ENABLE_DEBUGINFO
        wxLog::SetLogLevel(wxLOG_Debug);
#else
        wxLog::SetLogLevel(wxLOG_Info);
#endif
    }

    ~TestLogTargetGuard()
    {
        wxLog::FlushActive();
        wxLog::SetActiveTarget(old_log_target_);
        delete log_target_;
    }

    TestLogTargetGuard(const TestLogTargetGuard&) = delete;
    TestLogTargetGuard& operator=(const TestLogTargetGuard&) = delete;

private:
    wxLog* old_log_target_ = nullptr;
    wxLogStderr* log_target_ = nullptr;
};

} // namespace

int main(int argc, char* argv[])
{
    SetupUtf8Console();
    TestLogTargetGuard log_target_guard;

    Catch::Session session;
    session.configData().showDurations = Catch::ShowDurations::Always;
    return session.run(argc, argv);
}
