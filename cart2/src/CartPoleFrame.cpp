#include "CartPoleFrame.hpp"
#include "CartPoleCanvas.hpp"
#include "app.hpp"
#include "EvaluateEnvironment.hpp"
#include <wx/sizer.h>
#include <wx/stattext.h>
#include <torch/torch.h>
#include <iomanip>
#include <sstream>

#include <filesystem>


const int timer_ms = 20;
const int step_per_frame = 10;

wxBEGIN_EVENT_TABLE(CartPoleFrame, wxFrame)
EVT_TIMER(wxID_ANY, CartPoleFrame::OnTimer)
EVT_LEFT_DOWN(CartPoleFrame::OnMouseClick)
wxEND_EVENT_TABLE()

CartPoleFrame::CartPoleFrame(const wxString& title)
    : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxSize(800, 800)),
    device(torch::kCPU),
    timer(this, wxID_ANY)
{
    // --- GUIレイアウト ---
    wxBoxSizer* vbox = new wxBoxSizer(wxVERTICAL);

    canvas = new CartPoleCanvas(this);
    plotPanel = new PlotPanel(this);
    logBox = new wxTextCtrl(this, wxID_ANY, wxEmptyString,
        wxDefaultPosition, wxSize(800, 150),
        wxTE_MULTILINE | wxTE_READONLY);

    canvas->SetMinSize(wxSize(-1, 280));  // ← 上部の描画エリア固定高さ
    canvas->SetMaxSize(wxSize(-1, 280));  // （上下方向のリサイズ禁止）

    logBox->SetMinSize(wxSize(-1, 150));  // ← 下部ログ固定高さ
    logBox->SetMaxSize(wxSize(-1, 150));

    vbox->Add(canvas, 0, wxEXPAND | wxALL, 5);
    vbox->Add(plotPanel, 1, wxEXPAND | wxALL, 5);
    vbox->Add(logBox, 0, wxEXPAND | wxALL, 5);
    SetSizer(vbox);
    Layout();

    // ログレベル
    wxLog::SetLogLevel(wxLOG_Debug);

	// --- ログ出力先をこのクラスに設定 ---
    wxLog::SetActiveTarget(this);

    // --- RL生成 ---
    env = std::make_unique<CartPoleEnv>();
	evaluateEnvironment(*env, /*num_actions=*/2, /*num_trials=*/100);   // ← ランダム方策で環境評価
    agent = std::make_unique<RLAgent>(4, 2, device);
    
    // --- 環境初期化 ---
    state = env->Reset();  // ← reset() は 初期状態 を返す

    // --- タイマー開始 ---
    Bind(wxEVT_TIMER, &CartPoleFrame::OnTimer, this);
    timer.Start(timer_ms);  // 学習＆描画更新

    wxLogInfo("CartPoleRLGUI started.\n");
}

CartPoleFrame::~CartPoleFrame() {
    wxLog::SetActiveTarget(NULL);
}

void CartPoleFrame::ToggleTraining() {
    training_paused = !training_paused;
    wxLogMessage(training_paused ? "Training paused" : "Training resumed");
}
void CartPoleFrame::DoLogText(const wxString& msg) {
    this->logBox->AppendText(msg);
    this->logBox->AppendText("\n");
}

void CartPoleFrame::OnMouseClick(wxMouseEvent& event) {
    ToggleTraining();
}

void CartPoleFrame::OnTimer(wxTimerEvent& event) {
    if (training_paused)
        return;  // ←停止中は一切処理しない

    // 再入防止
	//this->timer.Stop();

    // --- 学習ステップを複数回回す ---
    //auto action = agent->select_action(state);
    float last_reward = 0.0f;
    torch::Tensor action;
    for (int i = 0; i < step_per_frame; ++i) {
        action = agent->select_action(state);
        auto [next_state, reward, done, _] = env->DoStep(action);
        agent->update(state, action.item<int>(), next_state, reward, done);
        state = next_state.clone();
        last_reward = reward;

        //if (step_count % 100 == 0) {
        //    std::ostringstream ss;
        //    ss << "Step " << step_count
        //        << ", state=[" << env->get_x() << " " << env->get_theta() << " " << env->get_x_dot() << " " << env->get_theta_dot() <<  "]"
        //        << ", actions=" << action.item<int>()
        //        << ", reward=" << reward;
        //    wxLogInfo(ss.str());
        //}

        // ステップ数インプリメント（グローバルなステップ数）
        step_count++;

        if (done) {
            episode_count++;
            float total_reward = env->get_total_reward();

            // ログ
            std::ostringstream ss;
            ss << "Episode " << episode_count
                << " finished, step_count=" << step_count
                << " total_reward=" << std::fixed << std::setprecision(3) << total_reward
                << " episode_step=" << (step_count - last_episode_step);
            wxLogInfo(ss.str());

            // 統計ログ
			wxGetApp().logScalar("11_episode/total_reward", episode_count, total_reward);
            //wxGetApp().flushMetricsLog();

			// プロット更新
            plotPanel->AddReward(total_reward);

            // 環境リセット
            state = env->Reset();

            // 次のエピソード開始準備
            last_episode_step = step_count;
            break;
        }
    }

    // --- カート位置・角度の描画更新 ---
    canvas->SetState(env->get_x(), env->get_theta(), env->get_x_dot(), env->get_theta_dot());
    canvas->SetAction(action);
    canvas->SetReward(last_reward);
    canvas->Refresh();

    // --- 100 step ごとにログを出力 ---
    if (step_count % 10 == 0) {
        std::ostringstream ss;
        ss << "Step " << step_count
            << ", eps=" << std::fixed << std::setprecision(2) << agent->epsilon
            << ", loss_ema=" << agent->loss_ema;
        wxLogInfo(ss.str());
    }

    // タイマー再開
	//this->timer.Start();
}
