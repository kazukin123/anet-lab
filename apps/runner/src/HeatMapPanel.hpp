// HeatMapFrame.hpp
#pragma once
#include <memory>
#include <vector>
#include <wx/wx.h>
#include <wx/spinctrl.h>
#include <wx/glcanvas.h>
#include <wx/timer.h>
#include "anet/observers.hpp"
#include "anet/rl.hpp"
#include "anet/trainer.hpp"
#include "anet/gui.hpp"

struct SweepHeatMapSettings
{
    int x;
    int y;
    wxString network_key;
    wxString extractor_name;
    int extractor_index;
    wxString tag;
};

class SweepHeatMapDialog : public wxDialog
{
public:
    // コンストラクタ
    // max_x, max_y: 数値選択の最大値
    // value_choices: コンボボックスの選択肢リスト
    SweepHeatMapDialog(wxWindow* parent, const anet::rl::EnvSpec& env_spec, int default_x = 0, int default_y = 1);

    virtual ~SweepHeatMapDialog() {}

    // 入力値を構造体として取得するメソッド
    SweepHeatMapSettings GetSettings() const;
private:
    void UpdateTag();
private:
    // コントロールへのポインタ
    wxSpinCtrl* spin_x_;
    wxSpinCtrl* spin_y_;
    wxComboBox* network_combo_;
    wxComboBox* extractor_combo_;
    wxSpinCtrl* extractor_idx_;
    wxTextCtrl* tag_text_;
};

class SweepHeatMapPanel : public anet::rl::gui::Panel
{
public:
    // コンストラクタ
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
    void CreateObserver(const SweepHeatMapSettings& settings,
        std::shared_ptr<anet::rl::DefaultTrainer> trainer, int log_interval);
private:
    // UIコンポーネント
    wxGLCanvas* canvas_;
    wxGLContext* context_;
    wxTextCtrl* step_text_;
    wxComboBox* refresh_combo_;
    wxTimer timer_;

    // ロジック
    std::shared_ptr<anet::rl::SweepedHeatMapObserver> observer_;

    // 描画用バッファ (Observerから受け取った画像データを保持)
    //std::vector<unsigned char> rgb_data_;
    anet::rl::SweepedHeatMapObserver::ImageResult captured_;
    //int img_width_ = 0;
    //int img_height_ = 0;
};
