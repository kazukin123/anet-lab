// LogPanel.hpp

#pragma once

#include <wx/wx.h>
#include "anet/gui.hpp"


class LogPanel final : public wxPanel {
public:
    explicit LogPanel(wxWindow* parent);
    virtual ~LogPanel();
private:
    void SetupControls();
    void SetupLogTarget();
    void RestoreLogTarget();
private:
    wxTextCtrl* text_ctrl_ = nullptr;
    wxLog* old_log_target_ = nullptr; ///< 以前のログターゲット（復元用）
};

