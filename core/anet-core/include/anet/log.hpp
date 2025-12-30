// anet/log.hpp

#pragma once

#include <ostream>
#include <sstream>
#include <string_view>
#include <string_view>
#include <algorithm>
#include <wx/log.h>
#include <wx/debug.h>
#include "anet/common.hpp"

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

    // ====== Null Stream (無効ログ用) ======
    class NullStream {
    public:
        using Manip = std::ostream& (*)(std::ostream&);
        NullStream& operator<<(Manip) { return *this; }
        template <typename T> NullStream& operator<<(const T&) { return *this; }
        template<typename T>
        constexpr const NullStream& operator<<(const T&) const noexcept { return *this; }
    };
    
    // ====== wxLog ストリームラッパ ======
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
        void flush() {
            const std::string body = stream_.str();
            if (body.empty()) return;

            std::ostringstream msg;

            // **** DEBUG のみヘッダを出力 ****
            if (level_ == wxLOG_Debug && !fn_.empty()) {
                msg << "" << fn_ << ":" << line_ << " " << func_ << "() ";
            }
            msg << body;

            switch (level_) {
            case wxLOG_Debug:   wxLogDebug(msg.str());   break;
            case wxLOG_Info:    wxLogInfo(msg.str());    break;
            case wxLOG_Message: wxLogMessage(msg.str()); break;
            case wxLOG_Warning: wxLogWarning(msg.str()); break;
            case wxLOG_Error:   wxLogError(msg.str());   break;
            default:            wxLogMessage(msg.str()); break;
            }
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

    // ====== NullStream インスタンス ======

    inline constexpr NullStream null_log_stream{};

    // ====== log ======

    inline auto info() { return anet::log::WxLogStream(wxLOG_Message); }
    inline auto warn() { return anet::log::WxLogStream(wxLOG_Warning); }
    inline auto error() { return anet::log::WxLogStream(wxLOG_Error); }


    // ====== Logger ======

    //class Logger {
    //public:
    //    explicit Logger(const char* tag) : tag_(tag) {}

    //    auto info()  const { return WxLogStream(wxLOG_Info) << "[" << tag_ << "] "; }
    //    auto debug() const { return WxLogStream(wxLOG_Debug) << "[" << tag_ << "] "; }
    //    auto warn()  const { return WxLogStream(wxLOG_Warning) << "[" << tag_ << "] "; }
    //    auto error() const { return WxLogStream(wxLOG_Error) << "[" << tag_ << "] "; }

    //private:
    //    const char* tag_;
    //};

    //template<typename T>
    //struct ClassLog {
    //    static Logger logger;
    //};

    //template<typename T>
    //Logger ClassLog<T>::logger{ typeid(T).name() };


}

// ====== マクロ ======

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

