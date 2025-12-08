// anet/log.hpp

#pragma once

#include <ostream>
#include <sstream>
#include <wx/log.h>

// ====== Null Stream (無効ログ用) ======
class NullStream {
public:
    template <typename T> NullStream& operator<<(const T&) { return *this; }
    using Manip = std::ostream& (*)(std::ostream&);
    NullStream& operator<<(Manip) { return *this; }
};

// ====== wxLog ストリームラッパ ======
class WxLogStream {
public:
    WxLogStream(wxLogLevel level,
        const char* file = nullptr, int line = 0, const char* func = nullptr)
        : m_level(level), m_file(file), m_line(line), m_func(func) {
    }

    ~WxLogStream() { flush(); }

    template <typename T>
    WxLogStream& operator<<(const T& value) {
        m_stream << value;
        return *this;
    }

    using Manip = std::ostream& (*)(std::ostream&);
    WxLogStream& operator<<(Manip manip) { manip(m_stream); return *this; }

private:
    void flush() {
        const std::string body = m_stream.str();
        if (body.empty()) return;

        std::ostringstream msg;

        // **** DEBUG のみヘッダを出力 ****
        if (m_level == wxLOG_Debug && m_file) {
            msg << "[" << m_file << ":" << m_line << " " << m_func << "] ";
        }
        msg << body;

        switch (m_level) {
        case wxLOG_Debug:   wxLogDebug(msg.str());   break;
        case wxLOG_Info:    wxLogInfo(msg.str());    break;
        case wxLOG_Warning: wxLogWarning(msg.str()); break;
        case wxLOG_Error:   wxLogError(msg.str());   break;
        default:            wxLogMessage(msg.str()); break;
        }
    }

    wxLogLevel m_level;
    std::ostringstream m_stream;
    const char* m_file;
    int m_line;
    const char* m_func;
};

// ====== NullStream インスタンス ======
static NullStream null_log_stream;

// ====== マクロ ======

#if ANET_ENABLE_DEBUGINFO

#define ANET_LOG_DEBUG() null_log_stream
#define ANET_LOG_INFO()  WxLogStream(wxLOG_Info)
#define ANET_LOG_WARN()  WxLogStream(wxLOG_Warning)
#define ANET_LOG_ERROR() WxLogStream(wxLOG_Error)

#else            // ========== Debug ==========

#define ANET_LOG_DEBUG() WxLogStream(wxLOG_Debug, __FILE__, __LINE__, __func__)
#define ANET_LOG_INFO()  WxLogStream(wxLOG_Info)
#define ANET_LOG_WARN()  WxLogStream(wxLOG_Warning)
#define ANET_LOG_ERROR() WxLogStream(wxLOG_Error)

#endif
