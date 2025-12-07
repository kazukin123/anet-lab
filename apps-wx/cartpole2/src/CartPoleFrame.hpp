#pragma once

#include <memory>
#include <torch/torch.h>
#include <wx/wx.h>
#include <wx/timer.h>

#include "anet/trainer.hpp"
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

    void InitTrainer();

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
    struct Config;
    std::unique_ptr<Config> config_;
    std::unique_ptr<anet::rl::DefaultTrainer> trainer_;
private:
    void InitImageLogObservers();
};

