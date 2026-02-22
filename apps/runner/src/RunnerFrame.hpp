// RunnerFrame.hpp

#pragma once

#include <memory>
#include <wx/wx.h>
#include <wx/aui/aui.h>
#include "anet/rl.hpp"
#include "LogPanel.hpp"
#include "TrainPanel.hpp"
#include "EvalPanel.hpp"
#include "QValuePanel.hpp"

class ModuleBrowser;
class RunPanel;

class RunnerFrame final : public wxFrame {
public:
    RunnerFrame(const wxString& title, const TrainPanelConfig& train_panel_config, const EvalPanelConfig& eval_panel_config);

    virtual ~RunnerFrame();

    void Initialize(std::shared_ptr<anet::rl::RunManager> run_manager);
protected:
    void OnExit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
    void OnClose(wxCloseEvent& event);

    void OnMouse(anet::rl::gui::ForwardedMouseEvent& event);
    void OnKey(anet::rl::gui::ForwardedKeyEvent& event);

    void OnHeatMap(wxCommandEvent& event);
    void OnConv2d(wxCommandEvent& event);
    void OnResetLayout(wxCommandEvent& event);
private:
    void SetupMenuBar();
    void CreateStatusBar();
    void SetupPanes(const TrainPanelConfig& train_panel_config, const EvalPanelConfig& eval_panel_config);
    void SetupEvents();
private:
    wxAuiPaneInfo PanelInfo(const wxString& name, const wxString& caption, const wxString& sub_caption = wxEmptyString);
private:
    wxAuiManager aui_mgr_;

    std::shared_ptr<anet::rl::EvalRunner> eval_runner_;

    ModuleBrowser* module_browser_ = nullptr;
    RunPanel* run_panel_ = nullptr;
    TrainPanel* train_panel_ = nullptr;
    EvalPanel* eval_panel_ = nullptr;
    LogPanel* log_panel_ = nullptr;
    QValuePanel* q_value_panel_ = nullptr;
};
