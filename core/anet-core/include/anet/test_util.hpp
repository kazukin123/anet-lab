// anet/test_util.hpp

#pragma once

#include "anet/common.hpp"
#include "anet/log.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <wx/log.h>

#ifdef _WIN32
#include <windows.h>
#endif

#if defined(_MSC_VER)
#include <crtdbg.h>
#include <stdlib.h>
#endif

namespace anet::test {

    inline constexpr std::string_view kFailureDialogOption = "--anet-test-failure-dialog";
    inline constexpr const char* kFailureDialogEnv = "ANET_TEST_FAILURE_DIALOG";

    struct PreparedTestArgs {
        std::vector<std::string> storage;
        std::vector<char*> argv;
        bool failure_dialog_enabled = false;

        int Argc() const
        {
            return static_cast<int>(argv.size());
        }

        char** Argv()
        {
            return argv.data();
        }
    };

    namespace detail {

        inline std::optional<bool> ParseFailureDialogValue(std::string_view value)
        {
            if (value == "on") return true;
            if (value == "off") return false;
            return std::nullopt;
        }

        inline std::string FormatExpectedFailureDialogValue()
        {
            return "Expected on or off.";
        }

        inline bool StartsWith(std::string_view text, std::string_view prefix)
        {
            return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
        }

    } // namespace detail

    inline PreparedTestArgs PrepareTestArgs(int argc, char* argv[], const char* failure_dialog_env)
    {
        PreparedTestArgs result;
        result.storage.reserve(static_cast<size_t>(argc));

        if (failure_dialog_env != nullptr) {
            const auto value = detail::ParseFailureDialogValue(failure_dialog_env);
            if (!value.has_value()) {
                throw std::invalid_argument(
                    "Invalid " + std::string(kFailureDialogEnv) + " value: \"" + std::string(failure_dialog_env)
                    + "\". " + detail::FormatExpectedFailureDialogValue());
            }
            result.failure_dialog_enabled = *value;
        }

        const std::string option_prefix = std::string(kFailureDialogOption) + "=";
        for (int i = 0; i < argc; ++i) {
            const std::string_view arg = argv[i] != nullptr ? std::string_view(argv[i]) : std::string_view();

            if (arg == kFailureDialogOption) {
                throw std::invalid_argument(
                    "Missing value for " + std::string(kFailureDialogOption)
                    + ". Expected " + std::string(kFailureDialogOption) + "=on or =off.");
            }

            if (detail::StartsWith(arg, option_prefix)) {
                const auto raw_value = arg.substr(option_prefix.size());
                const auto value = detail::ParseFailureDialogValue(raw_value);
                if (!value.has_value()) {
                    throw std::invalid_argument(
                        "Invalid " + std::string(kFailureDialogOption) + " value: \""
                        + std::string(raw_value) + "\". " + detail::FormatExpectedFailureDialogValue());
                }
                result.failure_dialog_enabled = *value;
                continue;
            }

            if (detail::StartsWith(arg, kFailureDialogOption)) {
                throw std::invalid_argument(
                    "Invalid " + std::string(kFailureDialogOption)
                    + " format. Expected " + std::string(kFailureDialogOption) + "=on or =off.");
            }

            result.storage.emplace_back(arg);
        }

        result.argv.reserve(result.storage.size());
        for (auto& arg : result.storage) {
            result.argv.push_back(arg.data());
        }
        return result;
    }

    inline PreparedTestArgs PrepareTestArgs(int argc, char* argv[])
    {
#if defined(_MSC_VER)
        char* env_value = nullptr;
        size_t env_size = 0;
        if (_dupenv_s(&env_value, &env_size, kFailureDialogEnv) != 0 || env_value == nullptr) {
            return PrepareTestArgs(argc, argv, nullptr);
        }

        std::string env_storage(env_value);
        std::free(env_value);
        return PrepareTestArgs(argc, argv, env_storage.c_str());
#else
        return PrepareTestArgs(argc, argv, std::getenv(kFailureDialogEnv));
#endif
    }

    inline int ReportTestArgsError(const std::exception& e, FILE* file = stderr)
    {
        std::fprintf(file, "ANET test argument error: %s\n", e.what());
        std::fflush(file);
        return 2;
    }

    inline void SetupTestFailureDialog(const bool enable_failure_dialog)
    {
#ifdef _WIN32
        if (enable_failure_dialog) {
            return;
        }

        // Windows/CRT の failure UI を stderr/非対話終了へ寄せ、test runner の人手待ちを防ぐ。
        SetErrorMode(GetErrorMode()
            | SEM_FAILCRITICALERRORS
            | SEM_NOGPFAULTERRORBOX
            | SEM_NOOPENFILEERRORBOX);

#if defined(_MSC_VER)
        _set_error_mode(_OUT_TO_STDERR);
        _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);

#if defined(_DEBUG)
        _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
        _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
        _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
        _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
#endif
#else
        (void)enable_failure_dialog;
#endif
    }

    /**
     * @brief テストで捕捉した wxLog の 1 件分の記録。
     *
     * production code からではなく、テストコードでログ内容を検証するための utility。
     */
    struct CapturedLogRecord {
        wxLogLevel level;
        std::string message;
    };

    /**
     * @brief wxLog を一時的に差し替えてログをメモリへ捕捉する guard。
     */
    class LogCaptureGuard {
    public:
        explicit LogCaptureGuard(wxLogLevel level = wxLOG_Max)
        {
            wxLog::DontCreateOnDemand();
            old_log_level_ = wxLog::GetLogLevel();
            old_logging_enabled_ = wxLog::EnableLogging(true);
            target_ = new CapturingLogTarget();
            old_target_ = wxLog::SetActiveTarget(target_);
            wxLog::SetLogLevel(level);
        }

        ~LogCaptureGuard()
        {
            wxLog::FlushActive();
            wxLog::SetActiveTarget(old_target_);
            wxLog::SetLogLevel(old_log_level_);
            wxLog::EnableLogging(old_logging_enabled_);
            delete target_;
        }

        LogCaptureGuard(const LogCaptureGuard&) = delete;
        LogCaptureGuard& operator=(const LogCaptureGuard&) = delete;

        void Flush() const
        {
            wxLog::FlushActive();
        }

        const std::vector<CapturedLogRecord>& Records() const
        {
            return target_->Records();
        }

    private:
        class CapturingLogTarget final : public wxLog {
        public:
            const std::vector<CapturedLogRecord>& Records() const { return records_; }

        protected:
            void DoLogRecord(wxLogLevel level, const wxString& msg, const wxLogRecordInfo& info) override
            {
                auto utf8 = msg.ToUTF8();
                records_.push_back({ level, utf8.data() ? utf8.data() : "" });
            }

        private:
            std::vector<CapturedLogRecord> records_;
        };

        wxLog* old_target_ = nullptr;
        wxLogLevel old_log_level_ = wxLOG_Max;
        bool old_logging_enabled_ = true;
        CapturingLogTarget* target_ = nullptr;
    };

    inline wxLogLevel DefaultTestLogLevel()
    {
#if ANET_ENABLE_DEBUGINFO
        return wxLOG_Debug;
#else
        return wxLOG_Info;
#endif
    }

    /**
     * @brief wxLog を一時的に差し替えてテストログを FILE へ出力する guard。
     */
    class StderrLogGuard {
    public:
        explicit StderrLogGuard(FILE* file = stderr, wxLogLevel level = DefaultTestLogLevel())
        {
            wxLog::DontCreateOnDemand();
            old_log_level_ = wxLog::GetLogLevel();
            old_logging_enabled_ = wxLog::EnableLogging(true);
            log_target_ = new StderrLogTarget(file);
            old_log_target_ = wxLog::SetActiveTarget(log_target_);
            wxLog::SetLogLevel(level);
        }

        ~StderrLogGuard()
        {
            wxLog::FlushActive();
            wxLog::SetActiveTarget(old_log_target_);
            wxLog::SetLogLevel(old_log_level_);
            wxLog::EnableLogging(old_logging_enabled_);
            delete log_target_;
        }

        StderrLogGuard(const StderrLogGuard&) = delete;
        StderrLogGuard& operator=(const StderrLogGuard&) = delete;

    private:
        class StderrLogTarget final : public wxLog {
        public:
            explicit StderrLogTarget(FILE* file)
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
        wxLogLevel old_log_level_ = wxLOG_Max;
        bool old_logging_enabled_ = true;
        StderrLogTarget* log_target_ = nullptr;
    };

    inline int CountRecords(const std::vector<CapturedLogRecord>& records, wxLogLevel level)
    {
        int count = 0;
        for (const auto& record : records) {
            if (record.level == level) {
                ++count;
            }
        }
        return count;
    }

    namespace detail {

        inline bool ContainsAll(std::string_view message, std::initializer_list<std::string_view> snippets)
        {
            for (const auto snippet : snippets) {
                if (message.find(snippet) == std::string_view::npos) {
                    return false;
                }
            }
            return true;
        }

    } // namespace detail

    inline bool HasRecordContaining(
        const std::vector<CapturedLogRecord>& records,
        wxLogLevel level,
        std::initializer_list<std::string_view> snippets)
    {
        for (const auto& record : records) {
            if (record.level == level && detail::ContainsAll(record.message, snippets)) {
                return true;
            }
        }
        return false;
    }

} // namespace anet::test
