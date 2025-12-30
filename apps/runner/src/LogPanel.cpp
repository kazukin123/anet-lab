#include "LogPanel.hpp"

#include <wx/font.h>


LogPanel::LogPanel(wxWindow* parent)
    : anet::rl::gui::Panel(parent)
{
    SetupControls();
    SetupLogTarget();
}

LogPanel::~LogPanel()
{
    RestoreLogTarget();
}

void LogPanel::SetupControls()
{
    // テキストコントロールの作成
    // 読み取り専用、複数行、リッチテキスト（色付け等のため）
    long style = wxTE_MULTILINE | wxTE_READONLY | wxHSCROLL | wxTE_RICH2;
    text_ctrl_ = new wxTextCtrl(this, wxID_ANY, "", wxDefaultPosition, wxDefaultSize, style);
    text_ctrl_->Enable(false);

    // フォント設定 (等幅フォント推奨)
    wxFont font = wxFont(10, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
    text_ctrl_->SetFont(font);

    // 背景色を少し暗くしてコンソールっぽくする（お好みで）
    // text_ctrl_->SetBackgroundColour(wxColour(30, 30, 30));
    // text_ctrl_->SetForegroundColour(wxColour(200, 200, 200));
    
    // text_ctrl_->SetBackgroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_LISTBOX));
    // text_ctrl_->SetForegroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_LISTBOXTEXT));

    // レイアウト (Sizerを使ってPanelいっぱいに広げる)
    wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);
    sizer->Add(text_ctrl_, 1, wxEXPAND | wxALL, 0);
    SetSizer(sizer);
}

void LogPanel::SetupLogTarget() {
    if (!text_ctrl_) return;

    // wxWidgets標準の「TextCtrlへ流すロガー」を使用
    // これにより wxLogMessage() 等が自動的にここに出るようになる
    wxLogTextCtrl* log_target = new wxLogTextCtrl(text_ctrl_);

    // ターゲットを切り替え、古いターゲットを保存しておく
    old_log_target_ = wxLog::SetActiveTarget(log_target);

    // タイムスタンプのフォーマット設定 (例: [12:34:56] Message)
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

