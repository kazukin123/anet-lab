// ErrorDialog.cpp

#include "ErrorDialog.hpp"
#include <wx/clipbrd.h>
#include <wx/artprov.h>
#include <wx/richmsgdlg.h>
#include "anet/log.hpp"

namespace LOG = anet::log;

void ReportError(const wxString& message, const wxString& detail, bool show_dialog)
{
    LOG::error() << "System error: " << message << "\n" << detail;
    wxLog::FlushActive();

    if (!show_dialog) {
        return;
    }

    // ダイアログ作成 (リサイズ可能にする)
    wxDialog dlg(NULL, wxID_ANY, wxT("System Error"),
        wxDefaultPosition, wxSize(650, 450), // 横幅を少し広げました
        wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER);

    // 全体の縦並びサイザー
    wxBoxSizer* v_sizer = new wxBoxSizer(wxVERTICAL);

    // 上部エリア (アイコン + メッセージ)
    wxBoxSizer* top_sizer = new wxBoxSizer(wxHORIZONTAL);

    // エラーアイコン (wxART_ERROR)
    wxBitmap error_icon = wxArtProvider::GetBitmap(wxART_ERROR, wxART_MESSAGE_BOX);
    wxStaticBitmap* icon_ctrl = new wxStaticBitmap(&dlg, wxID_ANY, error_icon);

    // アイコンを配置 (上揃え、全周囲に余白)
    top_sizer->Add(icon_ctrl, 0, wxALIGN_TOP | wxALL, 15);

    // エラーメッセージ
    wxTextCtrl* msg_ctrl = new wxTextCtrl(&dlg, wxID_ANY, message,
        wxDefaultPosition, wxSize(-1, 120),
        wxTE_MULTILINE | wxTE_READONLY | wxBORDER_THEME);

    // メッセージが短い場合は高さを制限、長い場合は広がるように調整
    top_sizer->Add(msg_ctrl, 1, wxEXPAND | wxTOP | wxBOTTOM | wxRIGHT, 5);

    // 上部エリアを全体に追加
    v_sizer->Add(top_sizer, 0, wxEXPAND);

    // wxTE_RICH を追加 (Windowsで背景色変更を確実にするため)
    wxTextCtrl* detail_text = new wxTextCtrl(&dlg, wxID_ANY, detail,
        wxDefaultPosition, wxDefaultSize,
        wxTE_MULTILINE | wxTE_READONLY | wxHSCROLL | wxTE_RICH);

    // 等幅フォント
    wxFont font(10, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
    detail_text->SetFont(font);

    v_sizer->Add(detail_text, 1, wxALL | wxEXPAND, 10);

    // --- ボタンエリア (下部) ---
    wxBoxSizer* btn_sizer = new wxBoxSizer(wxHORIZONTAL);

    // 「クリップボードにコピー」ボタン
    wxButton* copy_btn = new wxButton(&dlg, wxID_ANY, wxT("Copy to Clipboard"));
    wxString all_text = message + "\n----\n" + detail;
    copy_btn->Bind(wxEVT_BUTTON, [&](wxCommandEvent&) {
        if (wxTheClipboard->Open()) {
            wxTheClipboard->SetData(new wxTextDataObject(all_text));
            wxTheClipboard->Close();
            wxMessageBox(wxT("Copied to clipboard."), wxT("Info"), wxOK);
        }
        });

    // 「閉じる」ボタン
    wxButton* close_btn = new wxButton(&dlg, wxID_OK, wxT("Close"));
    close_btn->SetDefault(); // Enterキーで閉じるように設定

    btn_sizer->Add(copy_btn, 0, wxRIGHT, 10);
    btn_sizer->Add(close_btn, 0);

    v_sizer->Add(btn_sizer, 0, wxALL | wxALIGN_RIGHT, 10);

    dlg.SetSizer(v_sizer);
    dlg.Layout();
    dlg.Centre(); // 画面中央に表示
    dlg.ShowModal();
}
