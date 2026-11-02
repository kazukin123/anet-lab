#include "LogPanel.hpp"
#include <algorithm>
#include <cmath>
#include <optional>
#include <wx/artprov.h>
#include <wx/aui/auibar.h>
#include <wx/clipbrd.h>
#include <wx/datetime.h>
#include <wx/font.h>
#include <wx/settings.h>
#include <wx/srchctrl.h>
#include <wx/textbuf.h>
#include <wx/utils.h>
#ifdef __WXMSW__
#include <wx/msw/wrapwin.h>
#include <richedit.h>
#endif
#include "anet/config.hpp"
#include "anet/log.hpp"

namespace LOG = anet::log;

namespace {
    // 行の種類ごとの色。本文として読める明るさの差（コントラスト比 4.5 以上）を背景ごとに満たす。
    // 時刻は本文より一段薄く、灰色の VERBOSE とは色相で見分けられる青みの灰色にする。
    struct TonePalette {
        wxColour timestamp;
        wxColour dim;
        wxColour warning;
        wxColour error;
        wxColour search_miss;
    };
}

const TonePalette kLightPalette{
    .timestamp = wxColour(0x5F, 0x78, 0x94),
    .dim = wxColour(0x6E, 0x6E, 0x6E),
    .warning = wxColour(0xA6, 0x54, 0x00),
    .error = wxColour(0xC6, 0x28, 0x28),
    .search_miss = wxColour(0xFF, 0xDA, 0xDA),
};
const TonePalette kDarkPalette{
    .timestamp = wxColour(0x7A, 0x8B, 0xA1),
    .dim = wxColour(0x9A, 0x9A, 0x9A),
    .warning = wxColour(0xF0, 0xB0, 0x40),
    .error = wxColour(0xFF, 0x6B, 0x6B),
    .search_miss = wxColour(0x5A, 0x2A, 0x2A),
};
const wxString kTimestampSeparator = " ";
const wxSize kToolBitmapSize(16, 16);
constexpr int kJumpContextLines = 3;    // 開いた行の上に見せる行数
constexpr int kMaxAlertCount = 999;     // 警告ボタンに数字で出す上限
constexpr int kMaxUnreadCount = 9999;   // 未読の行数に数字で出す上限
constexpr int kSearchMinWidthDip = 120;
constexpr int kScrollSettleMs = 100;    // スクロールが止まったとみなす、表示が動かない時間

static const TonePalette& SelectPalette(const wxColour& background)
{
    // 背景が暗ければダーク用の色を使う。
    return background.GetLuminance() < 0.5 ? kDarkPalette : kLightPalette;
}

static std::optional<wxColour> FindLevelColour(wxLogLevel level, const TonePalette& palette)
{
    // FATAL・ERROR・WARN・VERBOSE にだけ色を付け、INFO はシステムの文字色のままにする。
    if (level <= wxLOG_Error) return palette.error;
    if (level == wxLOG_Warning) return palette.warning;
    if (level == wxLOG_Info) return palette.dim;
    return std::nullopt;
}

static wxString FormatCappedCount(int count, int max_count)
{
    // 上限を超えたら「上限+」で表し、表示の幅を一定に保つ。
    return count > max_count ? wxString::Format("%d+", max_count) : wxString::Format("%d", count);
}

static wxBitmapBundle MakeTrashBitmap(const wxColour& colour)
{
    // wx の標準アイコンにゴミ箱が無いため、再生ボタン（gui::MakePlaybackBitmap）と同じくシステムの文字色の SVG で描く。
    const auto svg = wxString::Format(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 16 16\"><g fill=\"%s\">"
        "<path d=\"M6 1 H10 V2 H14 V4 H2 V2 H6 Z\"/>"
        "<path fill-rule=\"evenodd\" d=\"M3 5 H13 L12 15 H4 Z M5.5 7 V13 H6.5 V7 Z M7.5 7 V13 H8.5 V7 Z M9.5 7 V13 H10.5 V7 Z\"/>"
        "</g></svg>",
        colour.GetAsString(wxC2S_HTML_SYNTAX));
    return wxBitmapBundle::FromSVG(svg.utf8_str(), kToolBitmapSize);
}

static void ScrollToEnd(wxTextCtrl& text)
{
#ifdef __WXMSW__
    // MSW の ShowPosition は指定行を表示の先頭へ送るため、wxTextCtrl::AppendText と同じく最下端へのスクロールを直接送る。
    ::SendMessage(static_cast<HWND>(text.GetHWND()), WM_VSCROLL, SB_BOTTOM, 0);
#else
    text.ShowPosition(text.GetLastPosition());
#endif
}

static void SetHorizontalScroll(wxTextCtrl& text, int position)
{
#ifdef __WXMSW__
    // 追記や選択の復元で動いた横位置を、控えた位置へ戻す。
    ::SendMessage(static_cast<HWND>(text.GetHWND()), WM_HSCROLL, MAKEWPARAM(SB_THUMBPOSITION, position), 0);
#else
    wxUnusedVar(text);
    wxUnusedVar(position);
#endif
}

static void ApplyWrap(wxTextCtrl& text, bool wrap)
{
    // 作り直さずに折り返しだけを切り替え、中身・色・選択を保つ。
#ifdef __WXMSW__
    // RichEdit の折り返しは作成時のスタイルで決まるため、行幅の指定で切り替える（0=窓幅で折り返す、1=折り返さない）。
    const auto hwnd = static_cast<HWND>(text.GetHWND());
    ::SendMessage(hwnd, EM_SETTARGETDEVICE, 0, wrap ? 0 : 1);
    // wx は RICH2 に ES_DISABLENOSCROLL を付けるため、折り返している間は使わない横スクロールバーを隠す。
    ::SendMessage(hwnd, EM_SHOWSCROLLBAR, SB_HORZ, wrap ? FALSE : TRUE);
#else
    const long style = text.GetWindowStyleFlag();
    text.SetWindowStyleFlag(wrap ? style & ~wxTE_DONTWRAP : style | wxTE_DONTWRAP);
#endif
}


// ======================================================
// LogPanel::Target
// ======================================================

class LogPanel::Target final : public wxLog {
public:
    explicit Target(LogPanel& panel) : panel_(panel) {}

    void SetLogLevel(wxLogLevel level) { level_ = level; }
    wxLogLevel GetLogLevel() const { return level_; }

    /// 本文より前（時刻・区切り・種類）の文字数。書式は記録によらないため、空の本文の記録を 1 件整形して数える。
    size_t HeadLength() const
    {
        const wxString timestamp = FormatTimestamp(0);
        const size_t separator = timestamp.empty() ? 0 : kTimestampSeparator.length();
        return timestamp.length() + separator + body_formatter_.Format(wxLOG_Message, wxString(), wxLogRecordInfo()).length();
    }

protected:
    void DoLogRecord(wxLogLevel level, const wxString& msg, const wxLogRecordInfo& info) override {
        // 表示レベルより詳しい記録はパネルへ出さない。
        if (level > level_) return;

        // Debug・Trace は従来どおり、基底の整形を通してデバッグ出力へ回す。
        if (level == wxLOG_Debug || level == wxLOG_Trace) {
            wxLog::DoLogRecord(level, msg, info);
            return;
        }
        // 時刻だけを塗り分けるため、時刻と本文を分けて渡す。
        panel_.AppendRecord(level, FormatTimestamp(info.timestampMS), body_formatter_.Format(level, msg, info));
    }
private:
    static wxString FormatTimestamp(wxLongLong_t msec)
    {
        // wxLog::TimeStampMS と同じ書式で、後ろの ": " だけを付けない。
        const wxString& format = wxLog::GetTimestamp();
        return format.empty() ? wxString() : wxDateTime(wxLongLong(msec)).Format(format);
    }

    LogPanel& panel_;   ///< 所有者の LogPanel から借りる（LogPanel が作り、破棄の前に受け口を戻す）
    wxLogLevel level_ = wxLOG_Message;
    const LOG::LogFormatter body_formatter_{/*enable_timestamp=*/false};
};


// ======================================================
// LogPanel
// ======================================================

wxString LogPanel::UnviewedAlerts::Describe() const
{
    // ツールチップ用に、未確認の件数を英語の 1 文にまとめる。
    return wxString::Format("%d %s, %d %s not viewed",
        warnings, warnings == 1 ? "warning" : "warnings", errors, errors == 1 ? "error" : "errors");
}

LogPanel::LogPanel(wxWindow* parent, const anet::ConfigData& config)
    : wxPanel(parent)
{
    SetupControls();
    SetupLogTarget();
    SetLogLevel(config.Get<std::string>("app.log_panel.level", "info"));
    ApplyToggleDefaults(config);

    // 隠していた間に届いた行は、表示されたときに映った分だけ既読にする。
    Bind(wxEVT_SHOW, [this](wxShowEvent& event) {
        if (event.IsShown()) CallAfter([this] { MarkVisibleAsRead(); });
        event.Skip();
    });
}

void LogPanel::ApplyToggleDefaults(const anet::ConfigData& config)
{
    // 操作帯のトグルの既定を、利用者が押したときと同じ経路で反映する（字下げは受け口の書式で幅を測るので、受け口の後）。
    const bool wrap = config.Get<bool>("app.log_panel.wrap", true);
    const bool indent = config.Get<bool>("app.log_panel.indent", true);
    view_toolbar_->ToggleTool(wrap_tool_id_, wrap);
    SetWrap(wrap);
    view_toolbar_->ToggleTool(indent_tool_id_, indent);
    SetIndent(indent);
    SetAutoScroll(config.Get<bool>("app.log_panel.auto_scroll", true));
}

LogPanel::~LogPanel()
{
    RestoreLogTarget();
}

void LogPanel::SetLogLevel(const std::string& level)
{
    if (!log_target_) return;

    wxLogLevel log_level = wxLOG_Message; // default is info (Message)
    if (level == "verbose") log_level = wxLOG_Info;
    else if (level == "warn") log_level = wxLOG_Warning;
    else if (level == "error") log_level = wxLOG_Error;

    log_target_->SetLogLevel(log_level);
}

wxLogLevel LogPanel::GetLogLevel() const
{
    return log_target_ ? log_target_->GetLogLevel() : wxLOG_Message;
}

LogPanel::UnviewedAlerts LogPanel::GetUnviewedAlerts() const
{
    // 未確認は alerts_ の後ろの部分なので、件数は添字の差から出る。
    const int total = static_cast<int>(alerts_.size() - first_unviewed_alert_);
    return {.warnings = total - unviewed_errors_, .errors = unviewed_errors_};
}

void LogPanel::SetupControls()
{
    // レイアウト: 上に操作の帯、下にログ欄を Panel いっぱいに広げる
    wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);
    SetupHeader(*sizer);

    // テキストコントロールの作成
    // 読み取り専用、複数行、リッチテキスト（行の色分けのため）。
    // 選んだ行と検索の一致は、フォーカスが帯や検索欄にあっても見えるようにする。
    // 折り返しは切替で横スクロールを戻せるよう wxHSCROLL のまま作り、既定（ApplyToggleDefaults）で折り返す。
    long style = wxTE_MULTILINE | wxTE_READONLY | wxHSCROLL | wxTE_RICH2 | wxTE_NOHIDESEL;
    text_ctrl_ = new wxTextCtrl(this, wxID_ANY, "", wxDefaultPosition, wxDefaultSize, style);

    // フォント設定 (等幅フォント推奨)
    wxFont font = wxFont(10, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
    text_ctrl_->SetFont(font);

    // 利用者のスクロールで最下が見えなくなったら自動スクロールを止め、最下まで戻したら再開する。
    // 通知は native の処理の前に来るため、操作の前の様子を控えて、スクロールが止まってから確かめる。
    const auto follow_user_scroll = [this](wxEvent& event) {
        NoteUserScroll();
        event.Skip();
    };
    scroll_settle_timer_.Bind(wxEVT_TIMER, [this](wxTimerEvent&) { FollowUserScroll(); });
    text_ctrl_->Bind(wxEVT_MOUSEWHEEL, follow_user_scroll);
    text_ctrl_->Bind(wxEVT_KEY_DOWN, follow_user_scroll);
    text_ctrl_->Bind(wxEVT_LEFT_UP, follow_user_scroll);
    // 最下へのスクロール（SB_BOTTOM）は追記のたびに自ら送るので、スクロールバーの通知から外す。
    for (const auto type : {wxEVT_SCROLLWIN_TOP, wxEVT_SCROLLWIN_LINEUP, wxEVT_SCROLLWIN_LINEDOWN,
            wxEVT_SCROLLWIN_PAGEUP, wxEVT_SCROLLWIN_PAGEDOWN, wxEVT_SCROLLWIN_THUMBTRACK, wxEVT_SCROLLWIN_THUMBRELEASE}) {
        text_ctrl_->Bind(type, [follow_user_scroll](wxScrollWinEvent& event) {
            // 横の通知（横位置を戻すときに自ら送るものを含む）は見ない。
            if (event.GetOrientation() == wxVERTICAL) follow_user_scroll(event);
            else event.Skip();
        });
    }
    // 大きさや折り返しが変わって最下が外れても、止めたと取り違えないよう、自動スクロール中は最下を見せ続ける。
    // 止めている間は、広がって新たに映った行を既読にする。
    text_ctrl_->Bind(wxEVT_SIZE, [this](wxSizeEvent& event) {
        CallAfter([this] {
            if (auto_scroll_) ScrollToEnd(*text_ctrl_);
            else MarkVisibleAsRead();
        });
        event.Skip();
    });

    sizer->Add(text_ctrl_, 1, wxEXPAND | wxALL, 0);
    SetSizer(sizer);

    // 件数のツールバーは 0 件で始まるので、隠した状態で並べる。
    UpdateAlertTool();
    UpdateUnreadTool();

    // 色はシステムに合わせ、テーマの変更時に塗り直す。
    UpdateAppearance();
    Bind(wxEVT_SYS_COLOUR_CHANGED, [this](wxSysColourChangedEvent& event) { UpdateAppearance(); event.Skip(); });

    // Ctrl+F で検索欄へ移り、検索欄の Esc でログ欄へ戻る。ほかのキーは既定の処理へ回す。
    Bind(wxEVT_CHAR_HOOK, [this](wxKeyEvent& event) {
        if (event.GetKeyCode() == 'F' && event.GetModifiers() == wxMOD_CONTROL) {
            search_ctrl_->SetFocus();
            search_ctrl_->SelectAll();
            return;
        }
        if (event.GetKeyCode() == WXK_ESCAPE && search_ctrl_->IsDescendant(wxWindow::FindFocus())) {
            text_ctrl_->SetFocus();
            return;
        }
        event.Skip();
    });
}

void LogPanel::SetupHeader(wxSizer& layout)
{
    // 操作の帯は「ツールバー群・伸びる検索欄・ツールバー」を横に並べる（EvalTracePanel の操作帯と同じ形）。
    // 件数のツールバーは 0 件のとき隠す。未確認の警告は左端に置いて残りを左へ詰め、未読の行数は検索欄の左に置いて
    // 出入りで検索欄の幅だけが変わるようにする（Auto-scroll などのボタンの位置を動かさない）。
    const long style = wxAUI_TB_DEFAULT_STYLE | wxAUI_TB_HORZ_TEXT;
    const auto make_toolbar = [this, style] {
        auto* toolbar = new wxAuiToolBar(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, style);
        toolbar->SetOverflowVisible(false);
        return toolbar;
    };
    auto* header = new wxBoxSizer(wxHORIZONTAL);

    // 未確認の警告: 押すと一番古い未確認の行を開き、▾ で全部を見たことにする。区切りも一緒に隠す。
    alert_toolbar_ = make_toolbar();
    alert_tool_id_ = wxWindow::NewControlId();
    alert_toolbar_->AddTool(alert_tool_id_, wxEmptyString,
        wxArtProvider::GetBitmapBundle(wxART_WARNING, wxART_TOOLBAR, kToolBitmapSize), wxEmptyString);
    alert_toolbar_->SetToolDropDown(alert_tool_id_, true);
    alert_toolbar_->AddSeparator();
    alert_toolbar_->Realize();
    alert_toolbar_->Bind(wxEVT_TOOL, [this](wxCommandEvent&) { ShowNextUnviewedAlert(); }, alert_tool_id_);
    alert_toolbar_->Bind(wxEVT_AUITOOLBAR_TOOL_DROPDOWN, [this](wxAuiToolBarEvent& event) {
        // 本体の部分の押下は通常のツール操作へ回し、▾ だけでメニューを出す。
        if (!event.IsDropDownClicked()) {
            event.Skip();
            return;
        }
        wxMenu menu;
        const int mark_all_id = menu.Append(wxID_ANY, "Mark all as viewed")->GetId();
        if (alert_toolbar_->GetPopupMenuSelectionFromUser(menu, event.GetItemRect().GetBottomLeft()) == mark_all_id) {
            MarkAllAlertsViewed();
        }
    }, alert_tool_id_);
    header->Add(alert_toolbar_, 0, wxALIGN_CENTER_VERTICAL);

    // 自動スクロール。
    scroll_toolbar_ = make_toolbar();
    auto_scroll_tool_id_ = wxWindow::NewControlId();
    scroll_toolbar_->AddTool(auto_scroll_tool_id_, "Auto-scroll", wxBitmapBundle(), "Auto-scroll", wxITEM_CHECK);
    scroll_toolbar_->ToggleTool(auto_scroll_tool_id_, auto_scroll_);
    scroll_toolbar_->Realize();
    scroll_toolbar_->Bind(wxEVT_TOOL, [this](wxCommandEvent&) {
        SetAutoScroll(scroll_toolbar_->GetToolToggled(auto_scroll_tool_id_));
    }, auto_scroll_tool_id_);
    header->Add(scroll_toolbar_, 0, wxALIGN_CENTER_VERTICAL);

    // 表示: 折り返し、折り返した行の字下げ、全体のコピー。
    view_toolbar_ = make_toolbar();
    wrap_tool_id_ = wxWindow::NewControlId();
    indent_tool_id_ = wxWindow::NewControlId();
    view_toolbar_->AddTool(wrap_tool_id_, "Wrap", wxBitmapBundle(), "Wrap long lines", wxITEM_CHECK);
    view_toolbar_->AddTool(indent_tool_id_, "Indent", wxBitmapBundle(), "Indent wrapped lines", wxITEM_CHECK);
    view_toolbar_->AddSeparator();
    const int copy_tool_id = wxWindow::NewControlId();
    view_toolbar_->AddTool(copy_tool_id, wxEmptyString,
        wxArtProvider::GetBitmapBundle(wxART_COPY, wxART_TOOLBAR, kToolBitmapSize), "Copy all");
    view_toolbar_->AddSeparator();
    view_toolbar_->Realize();
    view_toolbar_->Bind(wxEVT_TOOL, [this](wxCommandEvent&) {
        SetWrap(view_toolbar_->GetToolToggled(wrap_tool_id_));
    }, wrap_tool_id_);
    view_toolbar_->Bind(wxEVT_TOOL, [this](wxCommandEvent&) {
        SetIndent(view_toolbar_->GetToolToggled(indent_tool_id_));
    }, indent_tool_id_);
    view_toolbar_->Bind(wxEVT_TOOL, [this](wxCommandEvent&) { CopyAll(); }, copy_tool_id);
    header->Add(view_toolbar_, 0, wxALIGN_CENTER_VERTICAL);

    // Auto-scroll を止めている間に、まだ映っていない行数。押すと次の未読の行へ進む。
    unread_toolbar_ = make_toolbar();
    unread_tool_id_ = wxWindow::NewControlId();
    unread_toolbar_->AddTool(unread_tool_id_, wxEmptyString, wxBitmapBundle(), "Show the next unread lines");
    unread_toolbar_->Realize();
    unread_toolbar_->Bind(wxEVT_TOOL, [this](wxCommandEvent&) { ShowNextUnreadLines(); }, unread_tool_id_);
    header->Add(unread_toolbar_, 0, wxALIGN_CENTER_VERTICAL);

    // 検索: 空いた幅をすべて使う。Enter は古い方へ、Shift+Enter は新しい方へ進み、× は語があるときだけ出す。
    search_ctrl_ = new wxSearchCtrl(this, wxID_ANY, wxEmptyString,
        wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER);
    search_ctrl_->SetDescriptiveText("Search log");
    search_ctrl_->ShowCancelButton(false);
    search_ctrl_->SetMinSize(wxSize(FromDIP(kSearchMinWidthDip), -1));
    search_ctrl_->Bind(wxEVT_SEARCH, [this](wxCommandEvent&) { FindMatch(!wxGetKeyState(WXK_SHIFT)); });
    search_ctrl_->Bind(wxEVT_SEARCH_CANCEL, [this](wxCommandEvent&) { search_ctrl_->Clear(); });
    search_ctrl_->Bind(wxEVT_TEXT, [this](wxCommandEvent& event) {
        // 語が無い間は × と上下の移動を押せなくする。
        const bool has_query = !search_ctrl_->IsEmpty();
        if (search_miss_) SetSearchMiss(false);
        search_ctrl_->ShowCancelButton(has_query);
        search_toolbar_->EnableTool(older_tool_id_, has_query);
        search_toolbar_->EnableTool(newer_tool_id_, has_query);
        search_toolbar_->Refresh(false);
        event.Skip();
    });
    header->Add(search_ctrl_, 1, wxALIGN_CENTER_VERTICAL | wxLEFT | wxRIGHT, FromDIP(2));

    // 一致の移動と、ログのクリア（誤って押さないよう右端へ離す）。
    search_toolbar_ = make_toolbar();
    older_tool_id_ = wxWindow::NewControlId();
    newer_tool_id_ = wxWindow::NewControlId();
    clear_tool_id_ = wxWindow::NewControlId();
    search_toolbar_->AddTool(older_tool_id_, wxEmptyString,
        wxArtProvider::GetBitmapBundle(wxART_GO_UP, wxART_TOOLBAR, kToolBitmapSize), "Older match");
    search_toolbar_->AddTool(newer_tool_id_, wxEmptyString,
        wxArtProvider::GetBitmapBundle(wxART_GO_DOWN, wxART_TOOLBAR, kToolBitmapSize), "Newer match");
    search_toolbar_->AddSeparator();
    search_toolbar_->AddTool(clear_tool_id_, wxEmptyString,
        MakeTrashBitmap(wxSystemSettings::GetColour(wxSYS_COLOUR_BTNTEXT)), "Clear log");
    search_toolbar_->EnableTool(older_tool_id_, false);
    search_toolbar_->EnableTool(newer_tool_id_, false);
    search_toolbar_->Realize();
    search_toolbar_->Bind(wxEVT_TOOL, [this](wxCommandEvent&) { FindMatch(true); }, older_tool_id_);
    search_toolbar_->Bind(wxEVT_TOOL, [this](wxCommandEvent&) { FindMatch(false); }, newer_tool_id_);
    search_toolbar_->Bind(wxEVT_TOOL, [this](wxCommandEvent&) { ClearLog(); }, clear_tool_id_);
    header->Add(search_toolbar_, 0, wxALIGN_CENTER_VERTICAL);

    layout.Add(header, 0, wxEXPAND);
}

void LogPanel::SetupLogTarget() {
    if (!text_ctrl_) return;

    // フィルタリング機能付きのカスタムロガーを使用（この整形はデバッグ出力へ回す Debug・Trace 用）
    log_target_ = new Target(*this);
    log_target_->SetFormatter(new anet::log::LogFormatter());

    // ターゲットを切り替え、古いターゲットを保存しておく
    old_log_target_ = wxLog::SetActiveTarget(log_target_);

    // タイムスタンプのフォーマット設定 (パネルの例: 12:34:56.789 [I] Message)
    wxLog::SetTimestamp("%H:%M:%S.%l");
}

void LogPanel::RestoreLogTarget() {
    // LogViewが破棄される前に、ログ先を元に戻す
    // これをしないと、LogPanel破棄後にログが出た瞬間にクラッシュする
    if (old_log_target_) {
        wxLog::SetActiveTarget(old_log_target_);
        old_log_target_ = nullptr;
    }
}

void LogPanel::AppendRecord(wxLogLevel level, const wxString& timestamp, const wxString& text)
{
    // 最下から離れていたら、追記の前に自動スクロールを止める。通知の来ないスクロール（選択のドラッグなど）もここで拾う。
    // 利用者のスクロールが止まるのを待っている間は、追記で最下が外れる前に、再開するかもここで決める。
    if (text_ctrl_->IsShownOnScreen() && (auto_scroll_ || scroll_settle_timer_.IsRunning())) FollowView();

    // 追記の前に、利用者の選択と表示位置を控える。自動スクロール中で選択も無いときは控えずに足す。
    long selection_from = 0;
    long selection_to = 0;
    text_ctrl_->GetSelection(&selection_from, &selection_to);
    const bool restores_view = !auto_scroll_ || selection_from != selection_to;
    ViewPosition view;
    if (restores_view) {
        view = CaptureView();
        text_ctrl_->Freeze();
    }

    // 区切りの改行は次の記録の前に置き、末尾に空行（カーソル行）を残さない。時刻と本文は色を分けて足す。
    const long previous_end = text_ctrl_->GetLastPosition();
    const long start = previous_end == 0 ? 0 : previous_end + 1;
    const auto& palette = SelectPalette(text_ctrl_->GetBackgroundColour());
    const auto colour = FindLevelColour(level, palette);
    wxString lead = previous_end == 0 ? wxString() : wxString(wxS("\n"));
    if (!timestamp.empty()) {
        text_ctrl_->SetDefaultStyle(TextStyle(palette.timestamp));
        text_ctrl_->AppendText(lead + timestamp);
        timestamp_ranges_.push_back({.start = start, .end = text_ctrl_->GetLastPosition()});
        lead = kTimestampSeparator;
    }
    text_ctrl_->SetDefaultStyle(TextStyle(colour.value_or(wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOWTEXT))));
    text_ctrl_->AppendText(lead + text);
    const long end = text_ctrl_->GetLastPosition();

    // 色を付けた記録（時刻を含む）と、未確認の WARN 以上の記録を控える。
    if (colour) {
        tinted_ranges_.push_back({.start = start, .end = end, .level = level});
        if (level <= wxLOG_Warning) {
            alerts_.push_back(tinted_ranges_.size() - 1);
            if (level <= wxLOG_Error) ++unviewed_errors_;
        }
    }

    // 選択と表示を戻す。Freeze 中の選択の復元はスクロールを伴わないので、表示位置は最後に決める。
    if (restores_view) {
        text_ctrl_->SetSelection(selection_from, selection_to);
        if (auto_scroll_) {
            ScrollToEnd(*text_ctrl_);
            SetHorizontalScroll(*text_ctrl_, view.horizontal);
        } else {
            RestoreView(view);
        }
        text_ctrl_->Thaw();
    } else {
        // 入力位置を追記した記録の先頭へ置き、横スクロールを左端に保つ。
        // 入力位置へのスクロールは、表示より高い記録では先頭へ戻すので、縦は最下へ戻す。
        text_ctrl_->SetInsertionPoint(start);
        ScrollToEnd(*text_ctrl_);
    }

    // 止めている間に届いた行の頭を未読として控え、映る位置に届いた行はすぐ既読にする。警告ボタンも更新する。
    if (!auto_scroll_) {
        unread_lines_.push_back(start);
        const long text_start = end - static_cast<long>(text.length());
        for (size_t i = text.find('\n'); i != wxString::npos; i = text.find('\n', i + 1)) {
            unread_lines_.push_back(text_start + static_cast<long>(i) + 1);
        }
        UpdateUnreadTool();
        MarkVisibleAsRead();
    }
    if (level <= wxLOG_Warning) UpdateAlertTool();
}

LogPanel::ViewPosition LogPanel::CaptureView() const
{
    // 表示の先頭の文字と横スクロール位置を控える。行の番号ではなく文字で持ち、折り返しの切替にも使う。
    ViewPosition view{.horizontal = text_ctrl_->GetScrollPos(wxHORIZONTAL)};
    text_ctrl_->HitTest(wxPoint(0, 0), &view.first_visible);
    return view;
}

void LogPanel::RestoreView(const ViewPosition& view)
{
    // 控えた文字の行を表示の先頭へ戻し、横位置も戻す。
    text_ctrl_->ShowPosition(view.first_visible);
    SetHorizontalScroll(*text_ctrl_, view.horizontal);
}

template <typename Change>
void LogPanel::ChangeKeepingView(Change change)
{
    // 行の区切りや色を変えても、自動スクロール中は最下を、止めている間は表示の先頭の文字を見せ続ける。
    const auto view = CaptureView();
    text_ctrl_->Freeze();
    change();
    if (auto_scroll_) {
        ScrollToEnd(*text_ctrl_);
        SetHorizontalScroll(*text_ctrl_, view.horizontal);
    } else {
        RestoreView(view);
    }
    text_ctrl_->Thaw();

    // 行の区切りが変わって新たに映った行を既読にする。
    MarkVisibleAsRead();
}

bool LogPanel::IsEndVisible() const
{
    // 表示の下端の行が最後の行なら、最下が見えている。行は折り返しを含む見た目の行で比べる。
    long bottom = 0;
    text_ctrl_->HitTest(wxPoint(0, text_ctrl_->GetClientSize().y - 1), &bottom);
    long column = 0;
    long bottom_line = 0;
    long last_line = 0;
    text_ctrl_->PositionToXY(bottom, &column, &bottom_line);
    text_ctrl_->PositionToXY(text_ctrl_->GetLastPosition(), &column, &last_line);
    return bottom_line >= last_line;
}

void LogPanel::NoteUserScroll()
{
    // 一続きの操作の最初に、最下が見えていたかと表示の先頭を控え、止まるのを待つ。
    // RichEdit はホイールのスクロールを通知の後も続けることがあるため、通知の直後には決めない。
    if (!scroll_settle_timer_.IsRunning()) {
        end_visible_before_scroll_ = IsEndVisible();
        settling_first_visible_ = CaptureView().first_visible;
    }
    scroll_settle_timer_.StartOnce(kScrollSettleMs);
}

void LogPanel::FollowUserScroll()
{
    // 表示がまだ動いていれば、止まるまで待ち直す。
    const long first_visible = CaptureView().first_visible;
    if (first_visible != settling_first_visible_) {
        settling_first_visible_ = first_visible;
        scroll_settle_timer_.StartOnce(kScrollSettleMs);
        return;
    }
    FollowView();

    // 止めたまま読み進めた分を既読にする。
    MarkVisibleAsRead();
}

void LogPanel::FollowView()
{
    // 最下が見えなくなったら止め、見えていなかった最下まで戻したら再開する。
    // 操作の前から最下が見えていたとき（最下にいるままのクリックやキー）は、手で止めた状態を保つ。
    const bool end_visible = IsEndVisible();
    if (auto_scroll_ && !end_visible) {
        SetAutoScroll(false);
    } else if (!auto_scroll_ && end_visible && !end_visible_before_scroll_) {
        SetAutoScroll(true);
    }
}

wxTextAttr LogPanel::TextStyle(const wxColour& colour) const
{
    // 既定の書式は前の属性と合成されるが、wx は字下げのうち 2 行目以降の分を引き継がないため、毎回そろえて渡す。
    wxTextAttr style(colour);
    style.SetLeftIndent(0, hanging_indent_);
    return style;
}

void LogPanel::SetAutoScroll(bool enabled)
{
    // ボタンの表示を状態に合わせる（ジャンプや検索から止めた場合も同じ）。
    scroll_toolbar_->ToggleTool(auto_scroll_tool_id_, enabled);
    scroll_toolbar_->Refresh(false);
    if (auto_scroll_ == enabled) return;

    // ON にした時点で末尾へ飛び、未読を消す（追従している間に届く行は、すべて映る）。
    auto_scroll_ = enabled;
    unread_lines_.clear();
    UpdateUnreadTool();
    if (enabled) ScrollToEnd(*text_ctrl_);
}

void LogPanel::SetWrap(bool enabled)
{
    // 折り返しで行の区切りが変わるため、表示の先頭の文字を基準に位置を戻す。
    // 字下げは折り返した行にだけ効くので、折り返さない間は押せなくする（状態は保つ）。
    ChangeKeepingView([&] { ApplyWrap(*text_ctrl_, enabled); });
    view_toolbar_->EnableTool(indent_tool_id_, enabled);
    view_toolbar_->Refresh(false);
}

void LogPanel::SetIndent(bool enabled)
{
    // 折り返した 2 行目以降を本文の頭（時刻と種類の後ろ）にそろえる。段落の書式なので、既にある行は 1 回の指定で、
    // これから足す行は TextStyle でそろう。幅は等幅フォントでの文字数の幅を、wx の単位（1/10 mm）に直したもの。
    const int head_width = text_ctrl_->GetTextExtent(wxString('0', log_target_->HeadLength())).x;
    hanging_indent_ = enabled ? static_cast<int>(std::lround(head_width * 254.0 / text_ctrl_->GetDPI().x)) : 0;
    wxTextAttr indent;
    indent.SetLeftIndent(0, hanging_indent_);
    ChangeKeepingView([&] { text_ctrl_->SetStyle(0, text_ctrl_->GetLastPosition(), indent); });
}

void LogPanel::CopyAll()
{
    // 表示している全行を、OS の改行に直してクリップボードへ渡す。
    wxClipboardLocker locker;
    if (!locker) {
        LOG::warn() << "Log copy failed: the clipboard could not be opened.";
        return;
    }
    wxTheClipboard->SetData(new wxTextDataObject(wxTextBuffer::Translate(text_ctrl_->GetValue())));
}

void LogPanel::ClearLog()
{
    // 表示だけを消す。Run のログファイルには残る。
    text_ctrl_->Clear();
    tinted_ranges_.clear();
    timestamp_ranges_.clear();
    alerts_.clear();
    first_unviewed_alert_ = 0;
    unviewed_errors_ = 0;
    unread_lines_.clear();
    UpdateUnreadTool();
    UpdateAlertTool();
}

void LogPanel::ShowNextUnviewedAlert()
{
    // 未確認のうち一番古い行を開き、見たことにする。行の頭から見えるよう横位置は左端にする。
    if (first_unviewed_alert_ >= alerts_.size()) return;
    const TintedRange range = tinted_ranges_[alerts_[first_unviewed_alert_]];
    ++first_unviewed_alert_;
    if (range.level <= wxLOG_Error) --unviewed_errors_;
    RevealRange(range.start, range.end);
    SetHorizontalScroll(*text_ctrl_, 0);
    UpdateAlertTool();
}

void LogPanel::MarkAllAlertsViewed()
{
    // 同じ警告が大量に出たときのために、開かずに全件を見たことにする。
    first_unviewed_alert_ = alerts_.size();
    unviewed_errors_ = 0;
    UpdateAlertTool();
}

void LogPanel::ShowNextUnreadLines()
{
    // 次の未読の行を、直前の数行（既読の最後）と一緒に表示の上の方へ出し、映った分だけを既読にする。
    // 自分のペースで読み進める操作なので、最下まで映っても Auto-scroll は止めたままにする。
    if (unread_lines_.empty()) return;
    long column = 0;
    long line = 0;
    text_ctrl_->PositionToXY(unread_lines_.front(), &column, &line);
    text_ctrl_->ShowPosition(text_ctrl_->XYToPosition(0, std::max(0L, line - kJumpContextLines)));
    SetHorizontalScroll(*text_ctrl_, 0);
    MarkVisibleAsRead();
}

void LogPanel::MarkVisibleAsRead()
{
    // 表示の下端の行までに頭が映った未読の行を既読にする。スクロールバーで飛ばした行も既読とし、上へ戻っても未読に戻さない。
    // Logs pane が隠れている間は映っていないので進めない。
    if (unread_lines_.empty() || !text_ctrl_->IsShownOnScreen()) return;
    long bottom = 0;
    text_ctrl_->HitTest(wxPoint(0, text_ctrl_->GetClientSize().y - 1), &bottom);
    const size_t unread = unread_lines_.size();
    while (!unread_lines_.empty() && unread_lines_.front() <= bottom) unread_lines_.pop_front();
    if (unread_lines_.size() != unread) UpdateUnreadTool();
}

void LogPanel::FindMatch(bool older)
{
    // 大文字小文字を区別せずに探す。RICH2 では文字の位置と GetValue の添字が一致する。
    const wxString query = search_ctrl_->GetValue().Lower();
    if (query.empty()) return;
    const wxString text = text_ctrl_->GetValue().Lower();

    // 起点は今の選択。選択が空なら末尾から探す（新しい行は下に増えるので、直近の一致から見る）。
    long from = 0;
    long to = 0;
    text_ctrl_->GetSelection(&from, &to);
    if (from == to) from = to = static_cast<long>(text.length());

    // 端まで探して無ければ、反対の端から続ける。
    size_t found = wxString::npos;
    if (older) {
        if (from > 0) found = text.rfind(query, static_cast<size_t>(from - 1));
        if (found == wxString::npos) found = text.rfind(query);
    } else {
        found = text.find(query, static_cast<size_t>(to));
        if (found == wxString::npos) found = text.find(query);
    }
    SetSearchMiss(found == wxString::npos);
    if (found == wxString::npos) return;
    RevealRange(static_cast<long>(found), static_cast<long>(found + query.length()));
}

void LogPanel::RevealRange(long start, long end)
{
    // 読んでいる間に末尾へ戻されないよう、自動スクロールを止めてから開く。
    SetAutoScroll(false);

    // 直前の流れも見えるよう、開く行の数行上を表示の先頭にする。
    long column = 0;
    long line = 0;
    text_ctrl_->PositionToXY(start, &column, &line);
    text_ctrl_->ShowPosition(text_ctrl_->XYToPosition(0, std::max(0L, line - kJumpContextLines)));

    // 選ぶと、まだ見えていない部分（狭い表示での行や、長い行の右側）だけが最小限スクロールされる。
    text_ctrl_->SetSelection(start, end);
    MarkVisibleAsRead();
}

void LogPanel::SetSearchMiss(bool miss)
{
    // 一致が無いことを検索欄の背景で示す（ステータスバーは使わない）。
    search_miss_ = miss;
    const auto background = wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOW);
    search_ctrl_->SetBackgroundColour(miss ? SelectPalette(background).search_miss : background);
    search_ctrl_->Refresh();
}

void LogPanel::UpdateAlertTool()
{
    // 未確認にエラーがあれば赤、WARN だけなら黄の印に件数を添える。無いときは欄ごと隠す。
    const auto alerts = GetUnviewedAlerts();
    const wxArtID icon = alerts.errors > 0 ? wxART_ERROR : wxART_WARNING;
    alert_toolbar_->SetToolBitmap(alert_tool_id_, wxArtProvider::GetBitmapBundle(icon, wxART_TOOLBAR, kToolBitmapSize));
    alert_toolbar_->SetToolShortHelp(alert_tool_id_, alerts.Describe());
    UpdateCountTool(*alert_toolbar_, alert_tool_id_,
        alerts.Total() > 0 ? FormatCappedCount(alerts.Total(), kMaxAlertCount) : wxString());
}

void LogPanel::UpdateUnreadTool()
{
    // 止めている間に、まだ映っていない行があるときだけ件数を出す。
    const int unread = static_cast<int>(unread_lines_.size());
    UpdateCountTool(*unread_toolbar_, unread_tool_id_, auto_scroll_ || unread == 0
        ? wxString() : "+" + FormatCappedCount(unread, kMaxUnreadCount));
}

void LogPanel::UpdateCountTool(wxAuiToolBar& toolbar, int tool_id, const wxString& label)
{
    // 空の件数は欄ごと隠して残りを左へ詰める。表示の有無か桁数が変わったときだけ並べ直す。
    const bool visible = !label.empty();
    const bool relayout = toolbar.IsShown() != visible || toolbar.FindTool(tool_id)->GetLabel().length() != label.length();
    toolbar.SetToolLabel(tool_id, label);
    if (!relayout) {
        toolbar.Refresh(false);
        return;
    }
    toolbar.Realize();
    toolbar.InvalidateBestSize();
    GetSizer()->Show(&toolbar, visible, true);
    Layout();
}

void LogPanel::UpdateAppearance()
{
    // 色をシステムに合わせ、新しく足す行の既定にする。
    const auto background = wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOW);
    const auto foreground = wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOWTEXT);
    text_ctrl_->SetBackgroundColour(background);
    text_ctrl_->SetForegroundColour(foreground);
    text_ctrl_->SetDefaultStyle(wxTextAttr(foreground, background, text_ctrl_->GetFont()));

    // 既にある行は、全体を通常の色に戻してから、背景の明るさに合う色で色付きの行、時刻の順に塗り直す。
    const long last = text_ctrl_->GetLastPosition();
    if (last > 0) {
        const auto& palette = SelectPalette(background);
        ChangeKeepingView([&] {
            text_ctrl_->SetStyle(0, last, wxTextAttr(foreground, background));
            for (const auto& range : tinted_ranges_) {
                text_ctrl_->SetStyle(range.start, range.end, wxTextAttr(*FindLevelColour(range.level, palette)));
            }
            for (const auto& range : timestamp_ranges_) {
                text_ctrl_->SetStyle(range.start, range.end, wxTextAttr(palette.timestamp));
            }
        });
    }
    SetSearchMiss(search_miss_);

    // 自前で描いたゴミ箱は文字色に合わせて描き直す。
    search_toolbar_->SetToolBitmap(clear_tool_id_, MakeTrashBitmap(wxSystemSettings::GetColour(wxSYS_COLOUR_BTNTEXT)));
    search_toolbar_->Refresh(false);
}
