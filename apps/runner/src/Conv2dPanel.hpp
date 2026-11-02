// Conv2dPanel.hpp

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
#include "anet/config.hpp"

class Conv2dPanel final : public anet::rl::gui::Panel, public anet::rl::gui::TraceFollower
{
public:
    // observer: ヒートマップ生成ロジックを持つクラスの所有権を受け取る
    Conv2dPanel(
        wxWindow* parent, const wxString& title,
        anet::rl::RunManager& run_manager, std::shared_ptr<anet::rl::Runner> runner,
        const anet::ConfigData& config_data);

    virtual ~Conv2dPanel();
public:	// TraceFollower
    std::shared_ptr<const anet::rl::gui::TraceSnapshot> CreateTraceSnapshot(const anet::rl::TrainEvent& event) override;
    void ShowTraceSnapshot(std::shared_ptr<const anet::rl::gui::TraceSnapshot> snapshot) override;
    void ReturnToLive() override;
private:
    // イベントハンドラ
    //void OnTimer(wxTimerEvent& event);
    void OnPaint(wxPaintEvent& event);
    void OnSize(wxSizeEvent& event);

    // 描画処理
    void Render();

    // UI更新処理 (タイマー間隔の変更)
    void UpdateRefreshRate();
private:
    void CreateVisualizer();
    void RenderActivations(anet::rl::step_t step, const anet::TensorDict& activations);
    void CreateObserver(anet::rl::RunManager& run_manager, std::shared_ptr<anet::rl::Runner> runner);
private:
    class Config;
private:
    // 設定
    std::unique_ptr<Config> config_;

    // UIコンポーネント
    wxGLCanvas* canvas_ = nullptr;
    wxGLContext* context_ = nullptr;

    // データ
    std::mutex image_mutex_;
    wxImage current_image_;
    bool closed_ = false;
    bool trace_mode_ = false;
    std::optional<anet::rl::step_t> latest_step_;
    std::shared_ptr<const anet::TensorDict> latest_activations_;
    wxStaticText* no_data_label_ = nullptr;

    // ロジック
    std::unique_ptr<anet::rl::Conv2dVisualizer> visualizer_;
    std::shared_ptr<anet::rl::TrainObserver> observer_;
    std::shared_ptr<anet::rl::Notifier> notifier_;

};
