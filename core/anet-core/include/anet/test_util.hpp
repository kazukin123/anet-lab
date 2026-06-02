// anet/test_util.hpp

#pragma once

#include "anet/common.hpp"
#include "anet/log.hpp"

#include <cstdio>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>
#include <wx/log.h>

namespace anet::test {

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
