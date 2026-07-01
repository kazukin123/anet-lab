// anet/log.hpp

#pragma once

#include <ostream>
#include <sstream>
#include <string_view>
#include <string_view>
#include <algorithm>
#include <filesystem>
#include <cstdio>
#include <wx/log.h>
#include <wx/debug.h>
#include "anet/common.hpp"
#include "anet/profile.hpp"


#if defined(_MSC_VER) && _MSC_VER < 1930
    // consteval が不完全な時
#define ANET_CONSTEXTRACT constexpr
#else
#define ANET_CONSTEXTRACT consteval
#endif

#if ANET_ENABLE_DEBUGINFO

#ifndef ANET_ENABLE_DEBUG_LOG
#define ANET_ENABLE_DEBUG_LOG 1
#endif

#else

#ifndef ANET_ENABLE_DEBUG_LOG
#define ANET_ENABLE_DEBUG_LOG 0
#endif

#endif


namespace anet::log {


    // ============================================================
    // Null Stream (無効ログ用)
    // ============================================================

    class NullStream {
    public:
        using Manip = std::ostream& (*)(std::ostream&);
        NullStream& operator<<(Manip) { return *this; }
        template <typename T> NullStream& operator<<(const T&) { return *this; }
        template<typename T>
        constexpr const NullStream& operator<<(const T&) const noexcept { return *this; }
    };

    inline constexpr NullStream null_log_stream{};


    // ============================================================
    // WxLogStream
    // ============================================================

    class WxLogStream {
    public:
        WxLogStream(wxLogLevel level,
            std::string_view fn = {}, int line = 0, const char* func = nullptr)
            : level_(level), fn_(fn), line_(line), func_(func)
        {
            ;
        }
        WxLogStream(WxLogStream&&) noexcept = default;
        WxLogStream& operator=(WxLogStream&&) noexcept = default;
        WxLogStream(const WxLogStream&) = delete;
        WxLogStream& operator=(const WxLogStream&) = delete;

        ~WxLogStream() { flush(); }
    public:
        using Manip = std::ostream& (*)(std::ostream&);
        WxLogStream& operator<<(Manip manip) { manip(stream_); return *this; }
    private:
        void LogDebug(const std::string& msg)
        {
            if (wxIsDebuggerRunning()
                && wxLog::IsLevelEnabled(wxLOG_Debug, wxString::FromAscii(wxLOG_COMPONENT))) {
                wxMessageOutputDebug().Output(msg);
            }
		}
        void flush()
        {
            ANET_PROFILE_FUNC();

            const std::string body = stream_.str();
            if (body.empty()) return;

            std::ostringstream msg;

            // **** DEBUG のみヘッダを出力 ****
            if (level_ == wxLOG_Debug && !fn_.empty()) {
                msg << "" << fn_ << ":" << line_ << " " << func_ << "() ";
            }
            msg << body;

            switch (level_) {
            case wxLOG_Debug:   LogDebug(msg.str()); break;
            case wxLOG_Message: LogDebug("[MSG] " + msg.str()); wxLogMessage(msg.str());  break;
            case wxLOG_Info:    LogDebug("[INFO] " + msg.str()); wxLogInfo(msg.str());    break;
            case wxLOG_Warning: LogDebug("[WARN] " + msg.str()); wxLogWarning(msg.str()); break;
            case wxLOG_Error:   LogDebug("[ERROR] " + msg.str()); wxLogError(msg.str());  break;
            default:            LogDebug(msg.str()); wxLogMessage(msg.str()); break;
            }

            //wxLog::FlushActive();
        }
        wxLogLevel level_;
        std::string_view fn_;
        int line_;
        const char* func_;
    public:
        std::ostringstream stream_;
    };

    inline WxLogStream& operator<<(WxLogStream& s, const char* v)
    {
        s.stream_ << v;
        return s;
    }

    inline WxLogStream& operator<<(WxLogStream& s, const std::string& v)
    {
        s.stream_ << v;
        return s;
    }

    template<typename T>
    inline WxLogStream& operator<<(WxLogStream& s, const T& v)
    {
        s.stream_ << v;
        return s;
    }

    template<typename T>
    inline WxLogStream&& operator<<(WxLogStream&& s, const T& v)
    {
        s.stream_ << v;
        return std::move(s);
    }

    inline WxLogStream&& operator<<(WxLogStream&& s, WxLogStream::Manip manip)
    {
        manip(s.stream_);
        return std::move(s);
    }


    ANET_CONSTEXTRACT std::string_view ExtractSourceFileName(const char* filepath)
    {
        std::string_view fname = filepath;
        const auto pos = fname.find_last_of("/\\");
        return (pos == std::string_view::npos) ? fname : fname.substr(pos + 1);
    }

    static_assert(anet::log::ExtractSourceFileName("C:\\abc\\test.cpp") == "test.cpp");


    // ============================================================
    // Log API
    // ============================================================

    inline auto info() { return anet::log::WxLogStream(wxLOG_Message); }
    inline auto verbose() { return anet::log::WxLogStream(wxLOG_Info); }
    inline auto warn() { return anet::log::WxLogStream(wxLOG_Warning); }
    inline auto error() { return anet::log::WxLogStream(wxLOG_Error); }
    inline auto fatal() { return anet::log::WxLogStream(wxLOG_FatalError); }


    // ============================================================
    // FileLogger
    // ============================================================

    /// UTF-8でファイルに書き込むためのシンプルなカスタムロガー
    class FileLogger : public wxLog {
    public:
        FileLogger(FILE* file) : m_file(file) {}

        ~FileLogger()
        {
            if (m_file) {
                fclose(m_file);
            }
        }
    protected:
        void DoLogTextAtLevel(wxLogLevel level, const wxString& msg) override
        {
            ANET_PROFILE_FUNC();

            if (!m_file) return;
            wxString logStr = msg + wxT("\n");
            const wxCharBuffer buf = logStr.utf8_str(); // UTF-8エンコーディングのバイト列に変換
            fwrite(buf.data(), 1, buf.length(), m_file);

            if (level <= wxLOG_Warning) {
                fflush(m_file);
            }
        }
    private:
        FILE* m_file;
    };


    // ============================================================
    // LogFormatter
    // ============================================================

    class LogFormatter : public wxLogFormatter {
    public:
        LogFormatter(bool enable_timestamp = true)
            : enable_timestamp_(enable_timestamp)
        {
        }

        // Formatメソッドをオーバーライドして好みの文字列を生成する
        wxString Format(wxLogLevel level, const wxString& msg, const wxLogRecordInfo& info) const override
        {
            ANET_PROFILE_FUNC();

            const wxChar* level_str;
            switch (level) {
            case wxLOG_FatalError: level_str = wxT("[F] "); break;
            case wxLOG_Error:      level_str = wxT("[E] "); break;
            case wxLOG_Warning:    level_str = wxT("[W] "); break;
            case wxLOG_Message:    level_str = wxT("[I] "); break;
            case wxLOG_Info:       level_str = wxT("[V] "); break;
            case wxLOG_Debug:      level_str = wxT("[D] "); break;
            case wxLOG_Trace:      level_str = wxT("[T] "); break;
            default:               level_str = wxT("[L] "); break;
            }
            wxString result;
            result.Alloc(30 + msg.length());
            if (enable_timestamp_) {
                wxString time_str = FormatTimeMS(info.timestampMS);
                result.Append(time_str);
            }
            result.Append(level_str);
            result.Append(msg);
            return result;
        }
    private:
        const bool enable_timestamp_;
    };

} // namespace anet::log


// ============================================================
// マクロ 
// ============================================================

#define ANET_THIS_FILENAME anet::log::ExtractSourceFileName( __FILE__ )

#if ANET_ENABLE_DEBUG_LOG
#define ANET_LOG_DEBUG(expr)                                             \
    do {                                                                 \
        if (wxIsDebuggerRunning() &&                                     \
            wxLog::IsLevelEnabled(                                       \
            wxLOG_Debug, wxString::FromAscii(wxLOG_COMPONENT))) {        \
            auto _stream = anet::log::WxLogStream(wxLOG_Debug,           \
                                ANET_THIS_FILENAME, __LINE__, __func__); \
            _stream << expr;                                             \
        }                                                                \
    } while(0)

#else
#define ANET_LOG_DEBUG(expr) do {} while(0)
#endif

