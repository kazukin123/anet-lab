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


struct CartPoleFrame::Param {

    int timer_ms = 20;
    int step_per_frame = 10;
    int eval_interval = 1;
    int train_pause_step = 110000;
    int train_exit_step = -1; //110000;

    CartPoleFrame::Param(Properties* props) {
        if (props == NULL) return;
        std::string preset = props->Get("train.preset", "train");
        wxString preset_override;
        if (wxGetApp().GetCommandLine()->Found("t", &preset_override)) {
            preset = preset_override;
            props->Set("train.preset", preset);
        }
        ANET_READ_PROPS(props, preset, timer_ms);
        ANET_READ_PROPS(props, preset, step_per_frame);
        ANET_READ_PROPS(props, preset, eval_interval);
        ANET_READ_PROPS(props, preset, train_pause_step);
        ANET_READ_PROPS(props, preset, train_exit_step);
    }

};

wxBEGIN_EVENT_TABLE(CartPoleFrame, wxFrame)
EVT_TIMER(wxID_ANY, CartPoleFrame::OnTimer)
EVT_LEFT_DOWN(CartPoleFrame::OnMouseClick)
wxEND_EVENT_TABLE()

CartPoleFrame::CartPoleFrame(const wxString& title)
    : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxSize(800, 800)),
    param_(std::make_unique<Param>(wxGetApp().GetConfig())),
    //device(torch::kCPU),
    device(torch::kCUDA),
    timer(this, wxID_ANY)
{
    //test_heatmap_and_histgram();

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
    wxLogInfo("train.preset=%s", wxGetApp().GetConfig()->Get("train.preset", "train"));
    nlohmann::json params = {
        {"eval_interval", param_->eval_interval},
        {"train_pause_step", param_->train_pause_step},
    };
    wxGetApp().logJson("train/params", params);
    wxGetApp().flushMetricsLog();

    // --- RL生成 ---
    env = std::make_unique<CartPoleEnv>();
    agent = std::make_unique<RLAgent>(*env, 4, 2, device);

    // ランダム方策で環境難易度評価
    evaluateEnvironment(*env, /*num_actions=*/2, /*num_trials=*/100);
    
    // --- 環境初期化 ---
    state = env->Reset();  // ← reset() は 初期状態 を返す

    // --- タイマー開始 ---
    Bind(wxEVT_TIMER, &CartPoleFrame::OnTimer, this);
    timer.Start(param_->timer_ms);  // 学習＆描画更新

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
    for (int i = 0; i < param_->step_per_frame; ++i) {
        if ((param_->train_exit_step > 0) && (step_count >= param_->train_exit_step)) {
            wxGetApp().flushMetricsLog();
            wxGetApp().Exit();
        }

        // 行動選択と環境ステップ実行、更新
        auto [action, _, __] = agent->SelectAction(state);
        //auto [next_state, reward, done, _ ] = env->DoStep(action);
        auto env_result = env->DoStep(action);
        agent->Update({ state, action, env_result.next_state, env_result.reward, env_result.done });
        state = env_result.next_state.clone();
        last_reward = env_result.reward;
        train_total_reward += env_result.reward;

        // ステップ数インプリメント（グローバルなステップ数）
        step_count++;

        //エピソード終了判定
        if (env_result.done) {
            episode_count++;

            // ログ
            auto eps_step = step_count - last_episode_step;
            wxLogInfo("Episode finished. eps=%d step=%d total_reward=%f eps_step=%d", episode_count, step_count,train_total_reward, eps_step);

            // 統計ログ
			wxGetApp().logScalar("10_epsode/01_total_reward", episode_count, train_total_reward);

            // プロット更新
            plotPanel->AddReward(train_total_reward);

            // Canvas更新（エピソード終了）
            //canvas->SetState(env->get_x(), env->get_theta(), env->get_x_dot(), env->get_theta_dot());
            //canvas->SetAction(action);
            //canvas->SetReward(last_reward);

            // 学習状況評価
            if (episode_count % param_->eval_interval == 0) {
                eval_count_++;
                {   // ターゲットネットワークによる評価
                    state = env->Reset(anet::rl::RunMode::Eval1);
                    bool done = false;
                    auto total_reward = 0.0f;
                    while (!done) {
                        auto [action, _, __] = agent->SelectAction(state, anet::rl::RunMode::Eval1);
                        auto env_result = env->DoStep(action);
                        total_reward += env_result.reward;
                        state = env_result.next_state.clone();
                        done = env_result.done;
                    }
                    wxGetApp().logScalar("10_epsode/02_eval_reward", episode_count, total_reward);
                    wxGetApp().logScalar("11_eval/01_target_reward", step_count, total_reward);

                    // ターゲットネットワークによる評価の終了状態を描画
                    canvas->SetState(env->get_x(), env->get_theta(), env->get_x_dot(), env->get_theta_dot());
                    canvas->SetAction(action);
                    canvas->SetReward(env_result.reward);
                    //canvas->Refresh();
                }
                {   // メインネットワークによる評価
                    state = env->Reset(anet::rl::RunMode::Eval1);
                    bool done = false;
                    auto total_reward = 0.0f;
                    while (!done) {
                        auto [action, _, __] = agent->SelectAction(state, anet::rl::RunMode::Eval2);
                        auto env_result = env->DoStep(action);
                        total_reward += env_result.reward;
                        state = env_result.next_state.clone();
                        done = env_result.done;
                    }
                    wxGetApp().logScalar("11_eval/02_policy_reward", step_count, total_reward);
                }
            }

            // 環境リセット
            state = env->Reset();

            // エピソードが終わったので次エピソード準備
            last_episode_step = step_count;
            train_total_reward = 0.0f;
            //break;
        }
    }

    // --- カート位置・角度の描画更新 ---
    //canvas->SetState(env->get_x(), env->get_theta(), env->get_x_dot(), env->get_theta_dot());
    //canvas->SetAction(action);
    //canvas->SetReward(last_reward);
    canvas->Refresh();

    if ((param_->train_exit_step > 0) && (step_count >= param_->train_exit_step)) {
        wxGetApp().flushMetricsLog();
        wxGetApp().Exit();
    }
    if ((param_->train_pause_step > 0) && (step_count >= param_->train_pause_step) && !auto_pause_done_) {
        auto_pause_done_ = true;
        training_paused = true;
    }

    // タイマー再開
	this->timer.Start();
}
