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
    LunarLanderFrame(const wxString& title, int train_timer_ms, int eval_timer_ms, int eval_step_per_frame);
    ~LunarLanderFrame();

    void AddPlotData(float reward);
    void DoLogText(const wxString& msg) override;
    void SetEvalRunner(std::shared_ptr<anet::rl::EvalRunner> eval_runner);
private:
    // イベントハンドラ
    void OnTrainTimer(wxTimerEvent& event);
    void OnEvalTimer(wxTimerEvent& event);
    void OnMouseLeftClick(wxMouseEvent& event);
    void OnMouseRightClick(wxMouseEvent& event);
    void OnKeyDown(wxKeyEvent& event);
    void OnClose(wxCloseEvent& event);
private:
    void ToggleEval();
private:
    LunarLanderCanvas* train_canvas_ = nullptr;
    LunarLanderCanvas* eval_canvas_ = nullptr;
    PlotPanel* plot_panel_ = nullptr;
    wxTextCtrl* log_box_ = nullptr;

    wxTimer train_timer_;
    wxTimer eval_timer_;

    bool is_eval_pause_ = false;
    float eval_total_reward_ = 0.0;

    int eval_step_per_frame_ = 1;
private:
    std::shared_ptr<anet::rl::EvalRunner> eval_runner_;
};

