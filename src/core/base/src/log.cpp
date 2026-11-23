// log.cpp

#include "anet/log.hpp"

#include <wx/time.h>

using namespace anet::log;

static constexpr char kNotificationKey[] = "anet.notification.key";
static constexpr char kNotificationReset[] = "anet.notification.reset";

bool NotificationFilter::Accept(const wxLogRecordInfo& info)
{
    // Run の制御用の記録は、通常の出力へ流さない。
    wxUIntPtr reset = 0;
    if (info.GetNumValue(kNotificationReset, &reset)) {
        std::lock_guard<std::mutex> lock(mutex_);
        notified_.clear();
        return false;
    }
    wxString key;
    if (!info.GetStrValue(kNotificationKey, &key)) return true;
    const auto utf8 = key.ToUTF8();
    // wxLog が thread 固有の出力先へ直接届ける場合も、同じ記録を共有する。
    std::lock_guard<std::mutex> lock(mutex_);
    return notified_.emplace(utf8.data(), utf8.length()).second;
}

void anet::log::ResetNotifications()
{
    // wxLogChain の各出力先へ同じ制御を届け、可変な global を持たない。
    wxLogRecordInfo info(__FILE__, __LINE__, __FUNCTION__, wxLOG_COMPONENT);
    info.timestampMS = wxGetUTCTimeMillis().GetValue();
    info.StoreValue(kNotificationReset, wxUIntPtr{1});
    wxLog::OnLog(wxLOG_Message, wxString(), info);
}

void WxLogStream::FlushNotificationRecord(const std::string& message)
{
    if (!wxLog::IsEnabled()
        || !wxLog::IsLevelEnabled(level_, wxString::FromAscii(wxLOG_COMPONENT))) return;

    // 本文の値や prefix は識別に使わず、呼び出し位置と対象名だけで決める。
    std::ostringstream key;
    key << notification_source_.file_name() << ':' << notification_source_.line()
        << ':' << notification_source_.column() << '|' << notification_target_;
    wxLogRecordInfo info(notification_source_.file_name(),
        static_cast<int>(notification_source_.line()), notification_source_.function_name(), wxLOG_COMPONENT);
    info.timestampMS = wxGetUTCTimeMillis().GetValue();
    info.StoreValue(kNotificationKey, wxString::FromUTF8(key.str()));
    wxLog::OnLog(level_, wxString::FromUTF8(message), info);
}
