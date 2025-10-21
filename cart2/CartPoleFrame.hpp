#pragma once
#include <wx/wx.h>
#include <wx/timer.h>
#include <memory>
#include <torch/torch.h>

#include "CartPoleEnv.hpp"
#include "RLAgent.hpp"
#include "CartPoleCanvas.hpp"
#include "PlotPanel.hpp"
#include "tb_logger.hpp"
#include "app.hpp"

//
// --- メインウィンドウ（Frame） ---
//
class CartPoleFrame : public wxFrame, wxLog {
public:
    CartPoleFrame(const wxString& title);
    ~CartPoleFrame();

    void ToggleTraining();

    virtual void DoLogText(const wxString& msg);
private:
    // GUI部品
    CartPoleCanvas* canvas = nullptr;
    PlotPanel* plotPanel = nullptr;
    wxTextCtrl* logBox = nullptr;

    // タイマー
    wxTimer timer;
    bool training_paused = false;

    // 強化学習関連
    std::unique_ptr<CartPoleEnv> env;
    std::unique_ptr<RLAgent> agent;
    at::Tensor state;

    // メトリクス
    int step_count = 0;
    int last_episode_step = 0;
    int episode_count = 0;

    // デバイス（CPU固定）
    torch::Device device;

    // イベントハンドラ
    void OnTimer(wxTimerEvent& event);
    void OnMouseClick(wxMouseEvent& event);

    wxDECLARE_EVENT_TABLE();
};
