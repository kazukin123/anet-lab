#include "LogPanel.hpp"

#include <wx/font.h>


// ======================================================
// FilteredLogTextCtrl
// ======================================================

class FilteredLogTextCtrl : public wxLogTextCtrl {
public:
    FilteredLogTextCtrl(wxTextCtrl* textctrl) : wxLogTextCtrl(textctrl), level_(wxLOG_Message) {}

    void SetLogLevel(wxLogLevel level) { level_ = level; }

protected:
    void DoLogRecord(wxLogLevel level, const wxString& msg, const wxLogRecordInfo& info) override {
        if (level <= level_) {
            wxLogTextCtrl::DoLogRecord(level, msg, info);
        }
    }
private:
    wxLogLevel level_;
};


// ======================================================
// LogPanel
// ======================================================

LogPanel::LogPanel(wxWindow* parent, const std::string& init_log_level)
    : wxPanel(parent)
{
    SetupControls();
    SetupLogTarget();
    SetLogLevel(init_log_level);
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

void LogPanel::SetupControls()
{
    // テキストコントロールの作成
    // 読み取り専用、複数行、リッチテキスト（色付け等のため）
    long style = wxTE_MULTILINE | wxTE_READONLY | wxHSCROLL | wxTE_RICH2;
    text_ctrl_ = new wxTextCtrl(this, wxID_ANY, "", wxDefaultPosition, wxDefaultSize, style);
    //text_ctrl_->Enable(false);

    // フォント設定 (等幅フォント推奨)
    wxFont font = wxFont(10, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
    text_ctrl_->SetFont(font);

    // 色をシステム初期に強制
    auto bg_color = wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOW);
    auto fg_color = wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOWTEXT);
    text_ctrl_->SetBackgroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOW));
    text_ctrl_->SetForegroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOWTEXT));

    // 色をシステム初期に強制
    wxTextAttr defaultStyle;
    defaultStyle.SetTextColour(fg_color);
    defaultStyle.SetBackgroundColour(bg_color);
    defaultStyle.SetFont(font);
    text_ctrl_->SetDefaultStyle(defaultStyle);

    // レイアウト (Sizerを使ってPanelいっぱいに広げる)
    wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);
    sizer->Add(text_ctrl_, 1, wxEXPAND | wxALL, 0);
    SetSizer(sizer);
}

void LogPanel::SetupLogTarget() {
    if (!text_ctrl_) return;

    // フィルタリング機能付きのカスタムロガーを使用
    log_target_ = new FilteredLogTextCtrl(text_ctrl_);
    log_target_->SetFormatter(new anet::log::LogFormatter());

    // ターゲットを切り替え、古いターゲットを保存しておく
    old_log_target_ = wxLog::SetActiveTarget(log_target_);

    // タイムスタンプのフォーマット設定 (例: [12:34:56.789] Message)
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

