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
    // イベントハンドラ
    void OnTimer(wxTimerEvent& event);
    void OnMouseClick(wxMouseEvent& event);

    wxDECLARE_EVENT_TABLE();
private:
    // GUI部品
    CartPoleCanvas* canvas = nullptr;
    PlotPanel* plotPanel = nullptr;
    wxTextCtrl* logBox = nullptr;

    // タイマー
    wxTimer timer;
    bool training_paused = false;
    bool auto_pause_done_ = false;
private:
    void initImageLogObservers(const anet::rl::EnvSpec& env_spec);
private:
    // パラメータ
    struct Config;
    std::unique_ptr<Config> config_;

    // デバイス
    torch::Device device_agent_;

    // 乱数
    std::unique_ptr<anet::MasterSeedManager> master_seed_;

    // 強化学習関連
    anet::rl::StepCounts step_counts_;
    std::shared_ptr<anet::rl::BatchEnv> env_;
    std::shared_ptr<anet::rl::Agent> agent_;
    anet::rl::BatchState state_;

    // メトリクス
    anet::rl::Notifier notifier_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point last_time_;
    anet::rl::step_t last_exp_step_ = 0;
    anet::EmaFilter<float> train_reward_ema_;
    anet::EmaFilter<float> train_step_per_sec_ema_;
    anet::EmaFilter<float> exp_step_per_sec_ema_;
};
