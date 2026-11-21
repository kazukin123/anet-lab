// LogPanel.hpp

#pragma once

#include <deque>
#include <vector>
#include <wx/wx.h>
#include <wx/timer.h>
#include "anet/gui.hpp"

class wxAuiToolBar;
class wxSearchCtrl;
namespace anet { class ConfigData; }

class LogPanel final : public wxPanel {
public:
    /// 利用者がまだ開いていない WARN 以上の行の件数。FATAL は errors に含める。
    struct UnviewedAlerts {
        int warnings = 0;
        int errors = 0;

        int Total() const { return warnings + errors; }
        wxString Describe() const;
        bool operator==(const UnviewedAlerts&) const = default;
    };

    /// 表示 level と操作帯のトグルの既定を設定（app.log_panel.*）から読む。
    LogPanel(wxWindow* parent, const anet::ConfigData& config);
    void SetLogLevel(const std::string& level);
    wxLogLevel GetLogLevel() const;
    UnviewedAlerts GetUnviewedAlerts() const;
    virtual ~LogPanel();
private:
    class Target;
    /// 色を付けた 1 記録の範囲（時刻を含む）。通常の行は記録しない。
    struct TintedRange {
        long start = 0;
        long end = 0;
        wxLogLevel level = wxLOG_Message;
    };
    /// 時刻など、記録の中の一部の範囲。
    struct TextRange {
        long start = 0;
        long end = 0;
    };
    /// 追記や折り返しの前後で戻す表示位置。
    struct ViewPosition {
        long first_visible = 0;
        int horizontal = 0;
    };

    void SetupControls();
    void SetupHeader(wxSizer& layout);
    void ApplyToggleDefaults(const anet::ConfigData& config);
    void SetupLogTarget();
    void RestoreLogTarget();
    void AppendRecord(wxLogLevel level, const wxString& timestamp, const wxString& text);
    ViewPosition CaptureView() const;
    void RestoreView(const ViewPosition& view);
    bool IsEndVisible() const;
    void NoteUserScroll();
    void FollowUserScroll();
    void FollowView();
    void SetAutoScroll(bool enabled);
    void SetWrap(bool enabled);
    void SetIndent(bool enabled);
    template <typename Change> void ChangeKeepingView(Change change);
    wxTextAttr TextStyle(const wxColour& colour) const;
    void CopyAll();
    void ClearLog();
    void ShowNextUnviewedAlert();
    void MarkAllAlertsViewed();
    void ShowNextUnreadLines();
    void MarkVisibleAsRead();
    void FindMatch(bool older);
    void RevealRange(long start, long end);
    void SetSearchMiss(bool miss);
    void UpdateAlertTool();
    void UpdateUnreadTool();
    void UpdateCountTool(wxAuiToolBar& toolbar, int tool_id, const wxString& label);
    void UpdateAppearance();
private:
    wxTextCtrl* text_ctrl_ = nullptr;
    wxAuiToolBar* alert_toolbar_ = nullptr;
    wxAuiToolBar* scroll_toolbar_ = nullptr;
    wxAuiToolBar* unread_toolbar_ = nullptr;
    wxAuiToolBar* view_toolbar_ = nullptr;
    wxAuiToolBar* search_toolbar_ = nullptr;
    wxSearchCtrl* search_ctrl_ = nullptr;
    int alert_tool_id_ = wxID_NONE;
    int auto_scroll_tool_id_ = wxID_NONE;
    int unread_tool_id_ = wxID_NONE;
    int wrap_tool_id_ = wxID_NONE;
    int indent_tool_id_ = wxID_NONE;
    int older_tool_id_ = wxID_NONE;
    int newer_tool_id_ = wxID_NONE;
    int clear_tool_id_ = wxID_NONE;
    wxLog* old_log_target_ = nullptr; ///< 以前のログターゲット（復元用）
    Target* log_target_ = nullptr;

    std::vector<TintedRange> tinted_ranges_;    ///< 外観の変更で塗り直す行
    std::vector<TextRange> timestamp_ranges_;   ///< 外観の変更で塗り直す時刻（tinted_ranges_ の後に塗る）
    std::vector<size_t> alerts_;                ///< tinted_ranges_ のうち WARN 以上の添字（古い順）
    size_t first_unviewed_alert_ = 0;           ///< 未確認は古い順に開くため、常に alerts_ の後ろの部分になる
    int unviewed_errors_ = 0;
    std::deque<long> unread_lines_;             ///< Auto-scroll を止めてから届き、まだ画面に映っていない行の頭（古い順）
    int hanging_indent_ = 0;                    ///< 折り返した 2 行目以降の字下げ（1/10 mm、0 は字下げなし）
    wxTimer scroll_settle_timer_;               ///< 利用者のスクロールが止まるのを待つ
    bool end_visible_before_scroll_ = true;     ///< 一続きのスクロール操作の前に最下が見えていたか
    long settling_first_visible_ = 0;           ///< 止まったかを見る、表示の先頭の文字
    bool auto_scroll_ = true;
    bool search_miss_ = false;
};
