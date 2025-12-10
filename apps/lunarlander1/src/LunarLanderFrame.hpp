#pragma once

#include <memory>
#include <torch/torch.h>
#include <wx/wx.h>
#include <wx/timer.h>

#include "anet/trainer.hpp"
#include "PlotPanel.hpp"
#include "LunarLanderCanvas.hpp"
#include "LunarLanderEnv.hpp"

/// メインウィンドウ（Frame）
class LunarLanderFrame : public wxFrame, wxLog {
public:
    LunarLanderFrame(const wxString& title, int timer_ms);
    ~LunarLanderFrame();

    void AddPlotData(float reward);
    virtual void DoLogText(const wxString& msg);
private:
    // イベントハンドラ
    void OnTimer(wxTimerEvent& event);
    void OnMouseClick(wxMouseEvent& event);

    wxDECLARE_EVENT_TABLE();
private:
    // GUI部品
    LunarLanderCanvas* canvas = nullptr;
    PlotPanel* plotPanel = nullptr;
    wxTextCtrl* logBox = nullptr;

    // タイマー
    wxTimer timer;
    bool training_paused = false;
    bool auto_pause_done_ = false;
private:
    int timer_ms_;
};

