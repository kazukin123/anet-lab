#pragma once

#include <memory>
#include <torch/torch.h>
#include <wx/wx.h>
#include <wx/timer.h>

#include "anet/trainer.hpp"
#include "PlotPanel.hpp"
#include "CartPoleCanvas.hpp"
#include "CartPoleEnv.hpp"

/// メインウィンドウ（Frame）
class CartPoleFrame : public wxFrame, wxLog {
public:
    CartPoleFrame(const wxString& title, int timer_ms);
    ~CartPoleFrame();

    void AddPlotData(float reward);
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
    int timer_ms_;
};

