#include "LunarLanderFrame.hpp"
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <torch/torch.h>
#include <wx/log.h>
#include <wx/sizer.h>
#include "anet/tensor_util.hpp"
#include "anet/profile.hpp"
#include "anet/log.hpp"
#include "anet/observers.hpp"
#include "anet/replay_buffer.hpp"
#include "LunarLanderCanvas.hpp"
#include "UISnapshot.hpp"
#include "LunarLanderApp.hpp"
#include "HeatMapFrame.hpp"

namespace LOG = anet::log;

enum
{
    ID_VIEW_HEAT_MAP = wxID_HIGHEST + 1
};

LunarLanderFrame::LunarLanderFrame(const wxString& title, int train_timer_ms, int eval_timer_ms, int eval_step_per_frame)
    : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxSize(900, 800))
    , train_timer_(this, wxID_ANY), eval_timer_(this, wxID_ANY), eval_step_per_frame_(eval_step_per_frame)
{
    // メニュー
    auto menu_file = new wxMenu();
    menu_file->Append(wxID_EXIT, "E&xit");
    auto menu_view = new wxMenu();
    menu_view->Append(ID_VIEW_HEAT_MAP, "&HeatMap");
    auto menu_bar = new wxMenuBar();
    menu_bar->Append(menu_file, "&File");
    menu_bar->Append(menu_view, "&View");
    SetMenuBar(menu_bar);

    // GUIパネル群
    train_canvas_ = new LunarLanderCanvas(this);
    eval_canvas_ = new LunarLanderCanvas(this);
    plot_panel_ = new PlotPanel(this);
    log_box_ = new wxTextCtrl(this, wxID_ANY, wxEmptyString,
        wxDefaultPosition, wxSize(800, 150), wxTE_MULTILINE | wxTE_READONLY);
    log_box_->SetCanFocus(false);

    // GUIレイアウト ---
    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);
    wxBoxSizer* hbox = new wxBoxSizer(wxHORIZONTAL);
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

    // UIイベントハンドラ
    Bind(wxEVT_LEFT_DOWN, &LunarLanderFrame::OnMouseLeftClick, this);
    Bind(wxEVT_RIGHT_DOWN, &LunarLanderFrame::OnMouseRightClick, this);
    Bind(wxEVT_KEY_DOWN, &LunarLanderFrame::OnKeyDown, this);
    Bind(wxEVT_CHAR_HOOK, &LunarLanderFrame::OnKeyDown, this);
    Bind(wxEVT_TIMER, &LunarLanderFrame::OnTrainTimer, this, train_timer_.GetId());
    Bind(wxEVT_TIMER, &LunarLanderFrame::OnEvalTimer, this, eval_timer_.GetId());
    Bind(wxEVT_CLOSE_WINDOW, &LunarLanderFrame::OnClose, this, eval_timer_.GetId());
    Bind(wxEVT_MENU, [=](wxCommandEvent&) { Close(true);}, wxID_EXIT);
    Bind(wxEVT_MENU, &LunarLanderFrame::OnViewHeatMap, this, ID_VIEW_HEAT_MAP);

    // タイマー開始
    train_timer_.Start(train_timer_ms); // 学習＆描画更新
    eval_timer_.Start(eval_timer_ms);   // 評価＆描画更新
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
            if (event.batch_exp.state.IsEpisodeStart()) {
                eval_total_reward_ = 0;
            }

            eval_total_reward_ += event.batch_exp.reward[0].item<float>();

            // スナップショットを生成
            auto snapshot = LunarLanderApp::CreateSnapshot(event);
            snapshot.total_reward = eval_total_reward_;

            // スナップショットをUIに設定
            eval_canvas_->SetUISnapshot(snapshot);

            // エピソード終了なら総報酬をリセット
            //if (event.batch_exp.next_state.IsDone() || event.batch_exp.next_state.IsTruncated())
            //    eval_total_reward_ = 0.0f;

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
    anet::ProfileRange r("LunarLanderFrame::OnEvalTimer");

    if (is_eval_pause_) return;
    if (eval_runner_ == nullptr) return;

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
    anet::ProfileRange r("LunarLanderFrame::DoLogText");

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
    ToggleEval();
}

void LunarLanderFrame::ToggleEval()
{
    is_eval_pause_ = !is_eval_pause_;
    LOG::info() << "Eval " << (is_eval_pause_ ? "paused." : " resumed.");
}

void LunarLanderFrame::OnKeyDown(wxKeyEvent& event)
{
    //LOG::info() << "KeyDown: key=" << event.GetKeyCode();
    ANET_LOG_DEBUG("KeyDown: key=" << event.GetKeyCode());

    int64_t action;

    switch (event.GetKeyCode()) {
    case WXK_UP: action = 0; break;     // NOOP
    case WXK_DOWN: action = 2; break;   // MAIN ENGINE
    case WXK_LEFT: action = 1; break;   // LEFT ENGINE
    case WXK_RIGHT: action = 3; break;  // RIGHT ENGINE

    case WXK_NUMPAD0: action = 0; break;
    case WXK_NUMPAD1: action = 1; break;
    case WXK_NUMPAD2: action = 2; break;
    case WXK_NUMPAD3: action = 3; break;
    case WXK_NUMPAD4: action = 4; break;
    case WXK_NUMPAD5: action = 5; break;
    case WXK_NUMPAD6: action = 6; break;
    case WXK_NUMPAD7: action = 7; break;
    case WXK_NUMPAD8: action = 8; break;
    case WXK_NUMPAD9: action = 9; break;

    case WXK_SHIFT:
        wxGetApp().ToggleTraining();
        return;
    case WXK_SPACE:
        ToggleEval();
        return;
    default:
        eval_runner_->DoStep();
        eval_canvas_->Refresh();
        event.Skip();
        return;
    }

    eval_runner_->DoStep(action);
    eval_canvas_->Refresh();

    event.Skip();
}

void LunarLanderFrame::OnClose(wxCloseEvent& event)
{
    wxGetApp().StopTraining();
}

void LunarLanderFrame::OnViewHeatMap(wxCommandEvent& event)
{
    auto trainer = wxGetApp().GetTrainer();
    auto env_spec = trainer->GetBatchEnv()->GetSpec();

    // ダイアログ生成
    SweepHeatMapDialog dialog(this, env_spec);

    if (dialog.ShowModal() == wxID_OK)
    {
        // 構造体でまとめて取得
        SweepHeatMapSettings s = dialog.GetSettings();


        auto frame = new SweepHeatMapFrame(this, s.tag, s, trainer);
        frame->Show();
    }
}
