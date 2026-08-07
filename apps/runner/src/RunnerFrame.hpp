// RunnerFrame.hpp

#pragma once

#include <memory>
#include <optional>
#include <vector>
#include <wx/wx.h>
#include <wx/aui/aui.h>
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

    // Env への入力転送 (Training トグル・Eval 操作)
    void OnMouse(anet::rl::gui::ForwardedMouseEvent& event);
    void OnKey(anet::rl::gui::ForwardedKeyEvent& event);

    // メニュー
    void OnHeatMap(wxCommandEvent& event);
    void OnConv2d(wxCommandEvent& event);
    void OnResetLayout(wxCommandEvent& event);

    // AuiLayoutFrame フック
    void OnApplyLayoutPolicy() override;
    void OnPaneHiding(wxAuiPaneInfo& pane) override;
private:
    // 初期構築
    void SetupMenuBar();
    void CreateStatusBar();
    void SetupPanes(const TrainPanelConfig& train_panel_config, const EvalPanelConfig& eval_panel_config);
    void SetupEvents();
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
    // Eval 非表示時のフレーム縮退/復元 (OnApplyLayoutPolicy から呼ぶ)
    void RestoreFrameSizeIfNeeded();
    void CompactFrameForHiddenEval();
private:
    std::shared_ptr<anet::rl::EvalRunner> eval_runner_;

    TrainPanel* train_panel_ = nullptr;
    EvalPanel* eval_panel_ = nullptr;
    LogPanel* log_panel_ = nullptr;
    QValuePanel* q_value_panel_ = nullptr;

    // Eval 非表示時のフレーム縮退状態 (縮退予約幅と復元サイズ)
    int pending_eval_compact_width_ = 0;
    std::optional<wxSize> compact_restore_size_;
};
