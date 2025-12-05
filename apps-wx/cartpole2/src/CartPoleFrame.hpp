#pragma once

#include <memory>
#include <torch/torch.h>
#include <wx/wx.h>
#include <wx/timer.h>

#include "anet/dqn_agent.hpp"
#include "anet/observers.hpp"
#include "PlotPanel.hpp"
#include "CartPoleCanvas.hpp"
#include "CartPoleEnv.hpp"

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
    void initImageLogObservers(const anet::rl::EnvSpec& env_spec);
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
    std::shared_ptr<anet::rl::BatchEnv> env_;
    std::shared_ptr<anet::rl::Agent> agent_;
    anet::rl::BatchState state_;
    anet::rl::Notifier notifier_;

    // メトリクス
    int step_count_ = 0;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point last_time_;
    anet::EmaFilter<float> train_reward_ema_;
    anet::EmaFilter<float> msec_per_step_ema_;

    // デバイス
    torch::Device device_agent_;

    // 乱数
    std::unique_ptr<anet::MasterSeedManager> master_seed_;

    // イベントハンドラ
    void OnTimer(wxTimerEvent& event);
    void OnMouseClick(wxMouseEvent& event);

    wxDECLARE_EVENT_TABLE();
};
