#include "WorkspaceDialog.hpp"

#include <filesystem>
#include <utility>
#include <vector>

#include <wx/button.h>
#include <wx/checkbox.h>
#include <wx/colour.h>
#include <wx/dc.h>
#include <wx/dialog.h>
#include <wx/dirdlg.h>
#include <wx/listbox.h>
#include <wx/msgdlg.h>
#include <wx/odcombo.h>
#include <wx/settings.h>
#include <wx/sizer.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>

#include "anet/app_util.hpp"

using namespace anet;
using namespace anet::runner;

namespace anet::runner::detail {

class HistoryComboBox final : public wxOwnerDrawnComboBox {
public:
    HistoryComboBox(
        wxWindow* parent,
        const wxArrayString& choices,
        std::vector<bool> availability)
        : wxOwnerDrawnComboBox(
            parent, wxID_ANY, "", wxDefaultPosition, wxDefaultSize,
            choices, wxCB_READONLY)
        , availability_(std::move(availability))
    {
    }

protected:
    void OnDrawItem(wxDC& dc, const wxRect& rect, int item, int flags) const override
    {
        // popup と選択欄のどちらでも、解決不能な履歴だけをシステムの灰色で描画する。
        const auto original_colour = dc.GetTextForeground();
        if (item != wxNOT_FOUND
            && static_cast<size_t>(item) < availability_.size()
            && !availability_[static_cast<size_t>(item)]) {
            dc.SetTextForeground(wxSystemSettings::GetColour(wxSYS_COLOUR_GRAYTEXT));
        }
        wxOwnerDrawnComboBox::OnDrawItem(dc, rect, item, flags);
        dc.SetTextForeground(original_colour);
    }

private:
    std::vector<bool> availability_;
};

class WorkspaceDialog final : public wxDialog {
public:
    WorkspaceDialog(wxWindow* parent, const WorkspaceService& service)
        : wxDialog(parent, wxID_ANY, "Select Workspace", wxDefaultPosition, wxSize(620, 500))
        , service_(service)
        , history_(service.LoadHistory())
        , local_workspaces_(service.ScanLocalWorkspaces())
    {
        // 履歴と可用性を同じ index で保持し、表示色と確定時検証を一致させる。
        wxArrayString history_choices;
        std::vector<bool> history_availability;
        for (const auto& item : history_) {
            history_choices.Add(wxString::FromUTF8(item));
            history_availability.push_back(service_.IsResolvable(item));
        }

        // 履歴・ローカル一覧・任意パス・新規名を一つの選択画面へ配置する。
        auto* root = new wxBoxSizer(wxVERTICAL);
        root->Add(new wxStaticText(this, wxID_ANY, "Recent workspaces"), 0, wxLEFT | wxRIGHT | wxTOP, 12);
        history_combo_ = new HistoryComboBox(this, history_choices, std::move(history_availability));
        root->Add(history_combo_, 0, wxEXPAND | wxALL, 12);

        root->Add(new wxStaticText(this, wxID_ANY, "Local workspaces (click or press Enter to open)"),
            0, wxLEFT | wxRIGHT, 12);
        wxArrayString local_choices;
        for (const auto& item : local_workspaces_) {
            local_choices.Add(wxString::FromUTF8(item));
        }
        local_list_ = new wxListBox(this, wxID_ANY, wxDefaultPosition, wxSize(-1, 150), local_choices);
        root->Add(local_list_, 1, wxEXPAND | wxALL, 12);

        auto* browse_button = new wxButton(this, wxID_ANY, "Browse...");
        root->Add(browse_button, 0, wxLEFT | wxRIGHT | wxBOTTOM, 12);

        root->Add(new wxStaticText(this, wxID_ANY, "New workspace name"), 0, wxLEFT | wxRIGHT, 12);
        const auto initial_name = history_.empty() && local_workspaces_.empty() ? "_default" : "";
        new_name_ = new wxTextCtrl(this, wxID_ANY, initial_name);
        root->Add(new_name_, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 12);
        new_name_error_ = new wxStaticText(this, wxID_ANY, " ");
        new_name_error_->SetForegroundColour(*wxRED);
        root->Add(new_name_error_, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 12);

        skip_dialog_ = new wxCheckBox(this, wxID_ANY, "Do not show this dialog again");
        skip_dialog_->SetValue(LoadWorkspaceDialogSkip(service.AppDataDir()));
        root->Add(skip_dialog_, 0, wxLEFT | wxRIGHT | wxBOTTOM, 12);

        auto* buttons = new wxStdDialogButtonSizer();
        ok_button_ = new wxButton(this, wxID_OK);
        ok_button_->SetDefault();
        buttons->AddButton(ok_button_);
        buttons->AddButton(new wxButton(this, wxID_CANCEL));
        buttons->Realize();
        root->Add(buttons, 0, wxEXPAND | wxALL, 12);
        SetSizerAndFit(root);
        SetMinSize(wxSize(620, 500));

        // 初回は _default、新規作成済み環境ではMRU先頭を初期選択にする。
        if (!history_.empty()) {
            history_combo_->SetSelection(0);
            new_name_->Clear();
        }
        UpdateOkButton();

        // 入力源は相互排他的にし、ローカル一覧だけランチャーとして即確定する。
        history_combo_->Bind(wxEVT_COMBOBOX, [this](wxCommandEvent&) {
            new_name_->Clear();
            UpdateOkButton();
        });
        new_name_->Bind(wxEVT_TEXT, [this](wxCommandEvent&) {
            if (!new_name_->GetValue().empty()) {
                history_combo_->SetSelection(wxNOT_FOUND);
            }
            UpdateOkButton();
        });
        local_list_->Bind(wxEVT_LEFT_UP, [this](wxMouseEvent& event) {
            const int selection = local_list_->HitTest(event.GetPosition());
            if (selection == wxNOT_FOUND) {
                event.Skip();
                return;
            }
            local_list_->SetSelection(selection);
            AcceptLocalSelection();
        });
        local_list_->Bind(wxEVT_KEY_DOWN, [this](wxKeyEvent& event) {
            if (event.GetKeyCode() == WXK_RETURN || event.GetKeyCode() == WXK_NUMPAD_ENTER) {
                AcceptLocalSelection();
                return;
            }
            event.Skip();
        });
        browse_button->Bind(wxEVT_BUTTON, [this](wxCommandEvent&) { Browse(); });
        Bind(wxEVT_BUTTON, [this](wxCommandEvent&) { AcceptOkSelection(); }, wxID_OK);
    }

    WorkspaceDialogResult Result() const
    {
        return WorkspaceDialogResult{
            .input = selected_input_,
            .skip_dialog = skip_dialog_->GetValue(),
        };
    }

private:
    void ShowNewNameError(const std::optional<std::string>& error)
    {
        // 1行分の領域を維持しながら、入力中の修正理由だけを即時表示する。
        if (error.has_value()) {
            new_name_error_->SetLabel(wxString::FromUTF8("Error: " + *error));
        } else {
            new_name_error_->SetLabel(" ");
        }
        Layout();
    }

    void UpdateOkButton()
    {
        // 履歴選択中は新規名の診断を消し、それ以外は core の作成契約で判定する。
        if (history_combo_->GetSelection() != wxNOT_FOUND) {
            ShowNewNameError(std::nullopt);
            ok_button_->Enable(true);
            return;
        }
        if (new_name_->GetValue().empty()) {
            ShowNewNameError(std::nullopt);
            ok_button_->Enable(false);
            return;
        }

        const std::string input = new_name_->GetValue().utf8_string();
        const auto error = service_.GetNewWorkspaceNameError(input);
        ShowNewNameError(error);
        ok_button_->Enable(!error.has_value());
    }

    void AcceptOkSelection()
    {
        // 履歴は現在の可用性を再検証し、退避先消失時は選択画面へ留める。
        const int selection = history_combo_->GetSelection();
        if (selection != wxNOT_FOUND) {
            selected_input_ = history_[static_cast<size_t>(selection)];
            if (!service_.IsResolvable(selected_input_)) {
                wxMessageBox(
                    "The selected workspace is currently unavailable. Choose another workspace.",
                    "Workspace unavailable", wxOK | wxICON_ERROR, this);
                return;
            }
        } else {
            // 確定時にも再検証し、イベント順序に依存せず不正名をダイアログ内へ留める。
            const std::string input = new_name_->GetValue().utf8_string();
            if (const auto error = service_.GetNewWorkspaceNameError(input)) {
                ShowNewNameError(error);
                ok_button_->Enable(false);
                new_name_->SetFocus();
                new_name_->SelectAll();
                return;
            }
            selected_input_ = input;
        }
        EndModal(wxID_OK);
    }

    void AcceptLocalSelection()
    {
        // スキャン済みフォルダを即確定し、不足 config の補完は呼出側の Resolve に任せる。
        const int selection = local_list_->GetSelection();
        if (selection == wxNOT_FOUND) {
            return;
        }
        selected_input_ = local_workspaces_[static_cast<size_t>(selection)];
        EndModal(wxID_OK);
    }

    void Browse()
    {
        // native directory dialog の結果を UTF-8 workspace 入力へ変換する。
        wxDirDialog dialog(this, "Select workspace directory", "", wxDD_DIR_MUST_EXIST);
        if (dialog.ShowModal() == wxID_OK) {
            selected_input_ = dialog.GetPath().utf8_string();
            EndModal(wxID_OK);
        }
    }

    const WorkspaceService& service_;
    std::vector<std::string> history_;
    std::vector<std::string> local_workspaces_;
    HistoryComboBox* history_combo_ = nullptr;
    wxListBox* local_list_ = nullptr;
    wxTextCtrl* new_name_ = nullptr;
    wxStaticText* new_name_error_ = nullptr;
    wxCheckBox* skip_dialog_ = nullptr;
    wxButton* ok_button_ = nullptr;
    std::string selected_input_;
};

} // namespace anet::runner::detail

bool anet::runner::LoadWorkspaceDialogSkip(const std::filesystem::path& app_data_dir)
{
    // prefs が無い場合はダイアログ表示を既定とする。
    const auto path = app_data_dir / "prefs.txt";
    if (!std::filesystem::is_regular_file(path)) {
        return false;
    }
    return Properties(path).ToConfigData().Get<bool>("workspace.dialog_skip", false);
}

void anet::runner::SaveWorkspaceDialogSkip(
    const std::filesystem::path& app_data_dir,
    bool skip)
{
    // GUI 固有の選好を workspace 共通状態から分離して保存する。
    ConfigData data;
    data.Set("workspace.dialog_skip", skip);
    data.SaveProperties(app_data_dir / "prefs.txt");
}

std::optional<WorkspaceDialogResult> anet::runner::ShowWorkspaceDialog(
    wxWindow* parent,
    const WorkspaceService& service)
{
    // modal dialog の Cancel は workspace 未確定として呼出側へ返す。
    detail::WorkspaceDialog dialog(parent, service);
    if (dialog.ShowModal() != wxID_OK) {
        return std::nullopt;
    }
    return dialog.Result();
}
