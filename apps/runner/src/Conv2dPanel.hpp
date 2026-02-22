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

class Conv2dPanel final : public anet::rl::gui::Panel
{
public:
    // observer: ヒートマップ生成ロジックを持つクラスの所有権を受け取る
    Conv2dPanel(wxWindow* parent, const wxString& title, anet::rl::RunManager& run_manager, std::shared_ptr<anet::rl::Runner> runner);

    virtual ~Conv2dPanel();
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
    void CreateVisualizer(std::shared_ptr<anet::rl::Runner> runner);
    void CreateObserver(anet::rl::RunManager& run_manager, std::shared_ptr<anet::rl::Runner> runner);

private:
    // UIコンポーネント
    wxGLCanvas* canvas_ = nullptr;
    wxGLContext* context_ = nullptr;

    // データ
    std::mutex image_mutex_;
    wxImage current_image_;
    bool closed_ = false;

    // ロジック
    std::unique_ptr<anet::rl::Conv2dVisualizer> visualizer_;
    std::shared_ptr<anet::rl::TrainObserver> observer_;
    std::shared_ptr<anet::rl::Notifier> notifier_;
    anet::TensorDictFunction vis_dict_fn_;

};
