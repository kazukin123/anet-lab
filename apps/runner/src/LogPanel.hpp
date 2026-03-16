// LogPanel.hpp

#pragma once

#include <wx/wx.h>
#include "anet/gui.hpp"

class FilteredLogTextCtrl;

class LogPanel final : public wxPanel {
public:
    explicit LogPanel(wxWindow* parent, const std::string& init_log_level = "info");
    void SetLogLevel(const std::string& level);
    virtual ~LogPanel();
private:
    void SetupControls();
    void SetupLogTarget();
    void RestoreLogTarget();
private:
    wxTextCtrl* text_ctrl_ = nullptr;
    wxLog* old_log_target_ = nullptr; ///< 以前のログターゲット（復元用）
    FilteredLogTextCtrl* log_target_ = nullptr;
};

