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
const int eval_interval = 1;

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


    // パラメータ記録
    nlohmann::json params = {
        {"eval_interval", eval_interval},
    };
    wxGetApp().logJson("train/params", params);
    wxGetApp().flushMetricsLog();

    // --- RL生成 ---
    env = std::make_unique<CartPoleEnv>();
    agent = std::make_unique<RLAgent>(4, 2, device);
    
    // ランダム方策で環境難易度評価
    evaluateEnvironment(*env, /*num_actions=*/2, /*num_trials=*/100);
    
    //for (int i = 0; i < 10; i++) {
    //    state = env->Reset(anet::rl::RunMode::Eval1);
    //    for (int j = 0; j < 10; j++) {
    //        wxLogInfo("%d %d -> %f %f %f %f", i, j, env->get_x(), env->get_theta(), env->get_x_dot(), env->get_theta_dot());
    //        auto [action, _, __] = agent->SelectAction(state, anet::rl::RunMode::Eval1);
    //        env->DoStep(action);
    //        state = env->GetState();
    //    }
    //}

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
	this->timer.Stop();

    // --- 学習ステップを複数回回す ---
    //auto action = agent->select_action(state);
    float last_reward = 0.0f;
    torch::Tensor action;
    for (int i = 0; i < step_per_frame; ++i) {
        auto [action, _, __] = agent->SelectAction(state);
        //auto [next_state, reward, done, _ ] = env->DoStep(action);
        auto env_result = env->DoStep(action);
        agent->Update({ state, action, env_result.next_state, env_result.reward, env_result.done });
        state = env_result.next_state.clone();
        last_reward = env_result.reward;

        // ステップ数インプリメント（グローバルなステップ数）
        step_count++;

        if (env_result.done) {
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
			wxGetApp().logScalar("11_train/01_total_reward", episode_count, total_reward);
            //wxGetApp().flushMetricsLog();

            // 学習状況評価
            if (episode_count % eval_interval == 0) {
                eval_count_++;
                {   // ターゲットネットワークによる評価
                    state = env->Reset(anet::rl::RunMode::Eval1);
                    bool done = false;
                    while (!done) {
                        auto [action, _, __] = agent->SelectAction(state, anet::rl::RunMode::Eval1);
                        auto env_result = env->DoStep(action);
                        state = env_result.next_state.clone();
                        done = env_result.done;
                    }
                    wxLogInfo("Evaluation after %d episodes:  eval_count_=%d total_reward=%.3f",
                        episode_count, eval_count_, env->get_total_reward());
                    wxGetApp().logScalar("10_eval/01_target_reward", eval_count_, env->get_total_reward());
                }
                {   // メインネットワークによる評価
                    state = env->Reset(anet::rl::RunMode::Eval1);
                    bool done = false;
                    while (!done) {
                        auto [action, _, __] = agent->SelectAction(state, anet::rl::RunMode::Eval2);
                        auto env_result = env->DoStep(action);
                        state = env_result.next_state.clone();
                        done = env_result.done;
                    }
                    wxLogInfo("Evaluation after %d episodes:  eval_count_=%d total_reward=%.3f",
                        episode_count, eval_count_, env->get_total_reward());
                    wxGetApp().logScalar("10_eval/02_policy_reward", eval_count_, env->get_total_reward());
                }
            }

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

    // タイマー再開
	this->timer.Start();
}
