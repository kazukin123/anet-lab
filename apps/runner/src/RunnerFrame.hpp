// RunnerFrame.hpp

#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>
#include <wx/wx.h>
#include <wx/aui/aui.h>
#include <wx/aui/auibar.h>
#include "anet/rl.hpp"
#include "anet/gui.hpp"
#include "LogPanel.hpp"
#include "TrainPanel.hpp"
#include "EvalPanel.hpp"
#include "QValuePanel.hpp"

class RunnerFrame final : public anet::rl::gui::AuiLayoutFrame {
public:
    RunnerFrame(const wxString& title, const TrainPanelConfig& train_panel_config, const EvalPanelConfig& eval_panel_config);

    void Initialize(std::shared_ptr<anet::rl::RunManager> run_manager);
protected:
    // アプリ基本イベント
    void OnExit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
    void OnClose(wxCloseEvent& event);
    void OnSystemColourChanged(wxSysColourChangedEvent& event);

    // Env への入力転送 (Training トグル・Eval 操作)
    void OnMouse(anet::rl::gui::ForwardedMouseEvent& event);
    void OnKey(anet::rl::gui::ForwardedKeyEvent& event);

    // メニュー
    void OnHeatMap(wxCommandEvent& event);
    void OnConv2d(wxCommandEvent& event);
    void OnResetLayout(wxCommandEvent& event);
    void OnToggleTraining(wxCommandEvent& event);
    void OnToggleEval(wxCommandEvent& event);
    void OnEvalStep(wxCommandEvent& event);
    void OnSaveAgent(wxCommandEvent& event);
    void OnOpenRunFolder(wxCommandEvent& event);
    void OnUpdateTrainStatus(wxUpdateUIEvent& event);

    // AuiLayoutFrame フック
    void OnApplyLayoutPolicy() override;
private:
    enum class AlwaysOnTopMode {
        Off,
        Always,
        WhileRunning,
    };

    // 初期構築
    void SetupMenuBar();
    void CreateStatusBar();
    void SetupToolBars();
    void SetupPanes(const TrainPanelConfig& train_panel_config, const EvalPanelConfig& eval_panel_config);
    void SetupEvents();
private:
    void SetAlwaysOnTopMode(AlwaysOnTopMode mode);
    void ApplyAlwaysOnTopMode();
    void UpdateToggleBitmap(wxAuiToolBar* toolbar, int tool_id, bool running, std::optional<bool>& shown_as_running);
    void UpdateToolBarBitmaps();
    void UpdateTrainStatus();
    bool TrySaveAgent(const std::filesystem::path& file_path);
    void ShowEvalPaneIfHidden();
    void ToggleEvalPause();
    void AttachTrainStatusObserver(const std::shared_ptr<anet::rl::RunManager>& run_manager);
    void DetachTrainStatusObserver();
private:
    // 補助 pane 列 (HeatMap/Conv2d の動的追加と幅解決)
    void AddAuxPane(wxWindow* window, const wxAuiPaneInfo& pane_info);
    std::vector<wxAuiPaneInfo*> GetAuxPanes();
    bool IsAuxPane(const wxAuiPaneInfo& pane) const;
    int ResolveAuxDockWidth();
private:
    // 既定レイアウトへの復帰 (Reset Layout)
    void RestoreDefaultPanes();
    void HideAuxPanes();
    int GetDefaultQValueDockWidth() const;
private:
    struct TrainStatusSnapshot {
        anet::rl::StepCounts counts;
        std::optional<float> exp_sps;
        std::optional<float> train_sps;
        std::optional<float> elapsed_hour;
        std::chrono::steady_clock::time_point captured_at;
    };

    std::shared_ptr<anet::rl::EvalRunner> eval_runner_;
    std::shared_ptr<anet::rl::Notifier> train_status_notifier_;
    std::shared_ptr<anet::rl::TrainObserver> train_status_observer_;
    anet::rl::gui::UIDataStore<TrainStatusSnapshot> train_status_store_{0};
    std::optional<TrainStatusSnapshot> latest_train_status_;

    TrainPanel* train_panel_ = nullptr;
    EvalPanel* eval_panel_ = nullptr;
    LogPanel* log_panel_ = nullptr;
    QValuePanel* q_value_panel_ = nullptr;
    wxAuiToolBar* run_control_toolbar_ = nullptr;
    wxAuiToolBar* step_toolbar_ = nullptr;
    wxAuiToolBar* run_ops_toolbar_ = nullptr;
    wxAuiToolBar* panel_toolbar_ = nullptr;
    wxTextCtrl* exp_step_text_ = nullptr;
    wxTextCtrl* train_step_text_ = nullptr;

    // toggle bitmap へ反映済みの実行状態 (未反映は nullopt)。再生/一時停止の差し替え判定に使う。
    std::optional<bool> train_toggle_running_;
    std::optional<bool> eval_toggle_running_;
    AlwaysOnTopMode always_on_top_mode_ = AlwaysOnTopMode::Off;

    const float train_config_fps_;
    const float eval_config_fps_;
    bool initialized_ = false;
};
