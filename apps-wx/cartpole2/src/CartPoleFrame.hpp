#pragma once

#include <wx/wx.h>
#include <wx/timer.h>
#include <memory>
#include <torch/torch.h>

#include "CartPoleEnv.hpp"
#include "anet/dqn_agent.hpp"
#include "anet/observers.hpp"
#include "CartPoleCanvas.hpp"
#include "PlotPanel.hpp"

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
    // パラメータ
    struct Config;
    std::unique_ptr<Config> config_;

    // GUI部品
    CartPoleCanvas* canvas = nullptr;
    PlotPanel* plotPanel = nullptr;
    wxTextCtrl* logBox = nullptr;

    // タイマー
    wxTimer timer;
    bool training_paused = false;
    bool auto_pause_done_ = false;

    // 強化学習関連
    std::unique_ptr<CartPoleEnv> env;
    std::unique_ptr<anet::rl::Agent> agent;
    at::Tensor state;
    anet::rl::MetricsLogObserver metrics_obs_;
    anet::rl::HeatMapObserver heatmap_obs_;
    anet::rl::Notifier notifier_;

    // メトリクス
    int step_count = 0;
    int last_episode_step = 0;
    int episode_count = 0;
    int eval_count_ = 0;
    float train_total_reward = 0.0f;

    // デバイス（CPU固定）
    torch::Device device;

    // イベントハンドラ
    void OnTimer(wxTimerEvent& event);
    void OnMouseClick(wxMouseEvent& event);

    wxDECLARE_EVENT_TABLE();
};
