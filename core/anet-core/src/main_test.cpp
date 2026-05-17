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
        wxLog::DontCreateOnDemand();
        log_target_ = new TestLogTarget(stderr);
        old_log_target_ = wxLog::SetActiveTarget(log_target_);
        wxLog::EnableLogging(true);

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
    class TestLogTarget final : public wxLog {
    public:
        explicit TestLogTarget(FILE* file)
            : file_(file)
        {
        }

    protected:
        void DoLogRecord(wxLogLevel level, const wxString& msg, const wxLogRecordInfo& info) override
        {
            anet::log::LogFormatter formatter(/*enable_timestamp=*/false);
            wxString formatted = formatter.Format(level, msg, info);
            auto utf8 = formatted.ToUTF8();
            if (utf8.data()) {
                std::fputs(utf8.data(), file_);
                std::fputc('\n', file_);
                std::fflush(file_);
            }
        }

    private:
        FILE* file_;
    };

    wxLog* old_log_target_ = nullptr;
    TestLogTarget* log_target_ = nullptr;
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
