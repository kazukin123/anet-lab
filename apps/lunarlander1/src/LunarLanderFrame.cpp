#include "LunarLanderFrame.hpp"
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <torch/torch.h>
#include <wx/log.h>
#include <wx/sizer.h>
#include "anet/tensor_utils.hpp"
#include "anet/profile.hpp"
#include "anet/log.hpp"
#include "anet/observers.hpp"
#include "anet/replay_buffer.hpp"
#include "LunarLanderCanvas.hpp"
#include "UISnapshot.hpp"
#include "LunarLanderApp.hpp"

namespace LOG = anet::log;


LunarLanderFrame::LunarLanderFrame(const wxString& title, int train_timer_ms, int eval_timer_ms, int eval_step_per_frame)
    : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxSize(800, 800))
    , train_timer_(this, wxID_ANY), eval_timer_(this, wxID_ANY), eval_step_per_frame_(eval_step_per_frame)
{
    // GUIレイアウト ---
    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);
    wxBoxSizer* hbox = new wxBoxSizer(wxHORIZONTAL);

    train_canvas_ = new LunarLanderCanvas(this);
    eval_canvas_ = new LunarLanderCanvas(this);
    plot_panel_ = new PlotPanel(this);
    log_box_ = new wxTextCtrl(this, wxID_ANY, wxEmptyString,
        wxDefaultPosition, wxSize(800, 150),
        wxTE_MULTILINE | wxTE_READONLY);


    plot_panel_->SetMinSize(wxSize(-1, 280));  // ← 上部の描画エリア固定高さ
    plot_panel_->SetMaxSize(wxSize(-1, 280));  // （上下方向のリサイズ禁止）

    log_box_->SetMinSize(wxSize(-1, 150));  // ← 下部ログ固定高さ
    log_box_->SetMaxSize(wxSize(-1, 150));

    hbox->Add(train_canvas_, 1, wxEXPAND | wxALL, 2);
    hbox->Add(eval_canvas_, 1, wxEXPAND | wxALL, 2);

    vbox->Add(hbox, 1, wxEXPAND | wxALL, 2);
    vbox->Add(plot_panel_, 1, wxEXPAND | wxALL, 2);
    vbox->Add(log_box_, 0, wxEXPAND | wxALL, 2);

    SetSizer(vbox);
    Layout();

    // ログレベル
#if ANET_ENABLE_DEBUGINFO
    wxLog::SetLogLevel(wxLOG_Debug);
#endif

    // ログ出力先をこのクラスに設定
    wxLog::SetActiveTarget(this);
    LOG::info() << "LunarLanderRLGUI started.";


    // Train終了イベント
    Bind(wxEVT_TRAINER_EXIT, [this](wxCommandEvent&) {
        LOG::info() << "Stop training requested. Exiting.";
        Close(true);    // Frameを閉じる
        });

    // クリックイベント
    Bind(wxEVT_LEFT_DOWN, &LunarLanderFrame::OnMouseLeftClick, this);
    Bind(wxEVT_RIGHT_DOWN, &LunarLanderFrame::OnMouseRightClick, this);

    // タイマー開始
    Bind(wxEVT_TIMER, &LunarLanderFrame::OnTrainTimer, this, train_timer_.GetId());
    Bind(wxEVT_TIMER, &LunarLanderFrame::OnEvalTimer, this, eval_timer_.GetId());
    train_timer_.Start(train_timer_ms);  // 学習＆描画更新
    eval_timer_.Start(eval_timer_ms);  // 学習＆描画更新
}

LunarLanderFrame::~LunarLanderFrame()
{
    wxLog::SetActiveTarget(NULL);
}

void LunarLanderFrame::SetEvalRunner(std::shared_ptr<anet::rl::EvalRunner> eval_runner)
{
    eval_runner_ = eval_runner;

    // 評価Canvas向けObserver準備
    eval_runner_->GetNotifier()->Attach<anet::rl::FunctionTrainObserver>(
        [this](const anet::rl::TrainEvent& event)
        {
            // Trainスナップショット取得
            auto snapshot = LunarLanderApp::CreateSnapshot(event);
            eval_canvas_->SetUISnapshot(snapshot);

        }, "LunarLanderEvalApp");
}

void LunarLanderFrame::OnTrainTimer(wxTimerEvent& event)
{
    anet::ProfileRange r("LunarLanderFrame::OnTrainTimer");

    // 再入防止
    train_timer_.Stop();

    // スナップショットから画面描画
    auto snapshot = wxGetApp().GetUISnapshot();
    if (snapshot.has_value()) {
        this->train_canvas_->SetUISnapshot(*snapshot);
    }

    // 画面表示更新
    train_canvas_->Refresh();

    // タイマー再開
    train_timer_.Start();
}

void LunarLanderFrame::OnEvalTimer(wxTimerEvent& event)
{
    if (is_eval_pause_) return;

    // 再入防止
    eval_timer_.Stop();

    // 評価エピソードを1ステップ回す（Observer経由でeval_canvas更新)
    if (eval_step_per_frame_ > 0) {
        anet::ProfileRange r("LunarLanderFrame::OnEvalTimer.DoUpdateFrame");
        eval_runner_->DoUpdateFrame(eval_step_per_frame_);
    }

    // 画面表示更新
    eval_canvas_->Refresh();

    // タイマー再開
    eval_timer_.Start();
}

void LunarLanderFrame::DoLogText(const wxString& msg)
{
    this->log_box_->AppendText(msg);
    this->log_box_->AppendText("\n");
}

void LunarLanderFrame::AddPlotData(float value)
{
    plot_panel_->AddData(value);
    plot_panel_->Refresh();
}

void LunarLanderFrame::OnMouseLeftClick(wxMouseEvent& event)
{
    wxGetApp().ToggleTraining();
}

void LunarLanderFrame::OnMouseRightClick(wxMouseEvent& event)
{
    is_eval_pause_ = !is_eval_pause_;
    LOG::info() << "Eval " << (is_eval_pause_ ? "paused." : " resumed.");
    event.Skip();
}
