// HeatMapPanel.hpp

#pragma once

#include <memory>
#include <vector>
#include <wx/wx.h>
#include <wx/spinctrl.h>
#include <wx/glcanvas.h>
#include <wx/timer.h>
#include "anet/rl.hpp"
#include "anet/observers.hpp"
#include "anet/trainer.hpp"
#include "anet/gui.hpp"

struct SweepHeatMapSettings {
    int x;
    int y;
    wxString network_key;
    wxString extractor_name;
    int extractor_index;
    wxString tag;
};

class SweepHeatMapDialog final : public wxDialog {
public:
    SweepHeatMapDialog(wxWindow* parent, const anet::rl::EnvSpec& env_spec, int default_x = 0, int default_y = 1);

    // 入力値を構造体として取得
    SweepHeatMapSettings GetSettings() const;
private:
    void UpdateTag();
private:
    wxSpinCtrl* spin_x_ = nullptr;
    wxSpinCtrl* spin_y_ = nullptr;
    wxComboBox* network_combo_ = nullptr;
    wxComboBox* extractor_combo_ = nullptr;
    wxSpinCtrl* extractor_idx_ = nullptr;
    wxTextCtrl* tag_text_ = nullptr;
};

class SweepHeatMapPanel final : public anet::rl::gui::Panel
{
public:
    // observer: ヒートマップ生成ロジックを持つクラスの所有権を受け取る
    SweepHeatMapPanel(wxWindow* parent, const wxString& title,
        const SweepHeatMapSettings& settings, std::shared_ptr<anet::rl::DefaultTrainer> trainer);

    virtual ~SweepHeatMapPanel();
private:
    // イベントハンドラ
    void OnTimer(wxTimerEvent& event);
    void OnComboSelect(wxCommandEvent& event);
    void OnPaint(wxPaintEvent& event);
    void OnSize(wxSizeEvent& event);
    void OnRefreshButton(wxCommandEvent& event);

    // 描画処理
    void Render();

    // UI更新処理 (タイマー間隔の変更)
    void UpdateRefreshRate();
private:
    void CreateObserver(
        const SweepHeatMapSettings& settings, std::shared_ptr<anet::rl::DefaultTrainer> trainer, int log_interval);
private:
    // UIコンポーネント
    wxGLCanvas* canvas_ = nullptr;
    wxGLContext* context_ = nullptr;
    wxTextCtrl* step_text_ = nullptr;
    wxComboBox* refresh_combo_ = nullptr;
    wxTimer timer_;

    // ロジック
    std::shared_ptr<anet::rl::SweepedHeatMapObserver> observer_;

    // 描画用バッファ (Observerから受け取った画像データを保持)
    anet::rl::ImageData captured_;
};
