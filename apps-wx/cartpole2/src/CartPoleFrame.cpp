#include "CartPoleFrame.hpp"
#include "CartPoleCanvas.hpp"
#include "app.hpp"
#include "anet/eval_env.hpp"
#include <wx/sizer.h>
#include <wx/stattext.h>
#include <torch/torch.h>
#include <iomanip>
#include <sstream>

#include <filesystem>
#include "anet/tensor_check.hpp"


struct CartPoleFrame::Config : public anet::Config {
    int timer_ms = 20;
    int step_per_frame = 10;
    int eval_interval = 1;
    int train_pause_step = 110000;
    int train_exit_step = -1; //110000;
	int canvas_mode = 0;    //  0:評価エピソードの終了状況を描画 1:学習エピソードの終了状態を描画 2:学習状況を描画 

    CartPoleFrame::Config(const anet::ConfigData& configData) : anet::Config(configData, "train", "CartPoleFrame") {
        ANET_APPLY_CONFIG(configData, timer_ms);
        ANET_APPLY_CONFIG(configData, step_per_frame);
        ANET_APPLY_CONFIG(configData, eval_interval);
        ANET_APPLY_CONFIG(configData, train_pause_step);
        ANET_APPLY_CONFIG(configData, train_exit_step);
    }
};

wxBEGIN_EVENT_TABLE(CartPoleFrame, wxFrame)
EVT_TIMER(wxID_ANY, CartPoleFrame::OnTimer)
EVT_LEFT_DOWN(CartPoleFrame::OnMouseClick)
wxEND_EVENT_TABLE()

CartPoleFrame::CartPoleFrame(const wxString& title)
    : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxSize(800, 800)),
    config_(std::make_unique<CartPoleFrame::Config>(wxGetApp().GetConfig("train"))),
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
    wxLogInfo("train.preset=%s confg=%s", wxGetApp().GetConfig("train").Get("preset"), config_->ToStdString());
    wxLogInfo("seed=%lld", rnd_.GetSeed());
    anet::MetricsLogger::Instance()->log_json("train/params", config_->ToJson());
    anet::MetricsLogger::Instance()->flush();

    // --- RL生成 ---
    env = std::make_unique<CartPoleEnv>(&rnd_);
    auto env_spec = env->GetSpec();
    wxLogInfo("env_spec=" + env_spec.ToString());
    anet::ConfigData agentConfig = wxGetApp().GetConfig("agent");
    agent = std::make_unique<anet::rl::DQNAgent>(agentConfig, env_spec, device, &rnd_);
    notifier_.AddObserver(&metrics_obs_);
    notifier_.AddObserver(&heatmap_obs_);

    // ランダム方策で環境難易度評価
    //evaluateEnvironment(*env, /*num_actions=*/2, /*num_trials=*/100);
    
    // --- 環境初期化 ---
    state_ = env->Reset();  // ← reset() は 初期状態 を返す
    ANET_CHECK_DEVICE_CPU_MSG(state_.obs, "Initial state");
    ANET_CHECK_SHAPE(state_.obs, { ANET_SHAPE_ANY, 4 });

    // --- タイマー開始 ---
    Bind(wxEVT_TIMER, &CartPoleFrame::OnTimer, this);
    timer.Start(config_->timer_ms);  // 学習＆描画更新

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
    //anet::rl::BatchStepResult step_result;
    for (int i = 0; i < config_->step_per_frame; ++i) {
        if ((config_->train_exit_step > 0) && (step_count >= config_->train_exit_step)) {
            anet::MetricsLogger::Instance()->flush();
            wxGetApp().Exit();
        }

        // 行動選択
        wxLogDebug("CartPoleFrame::OnTimer() step=%d state=%s", step_count, state_.ToString());
        auto action_info = agent->MakeAction(state_.obs);
        wxLogDebug("CartPoleFrame::OnTimer() step=%d action=%s", step_count, action_info.ToString());
        ANET_CHECK_DEVICE(action_info.action, device);

        // 環境ステップ実行
        anet::rl::BatchStepResult result = env->DoStep(action_info.action);    // next_state, reward, done, truncated
        wxLogDebug("CartPoleFrame::OnTimer() step=%d reward=%s", step_count, anet::ToString(result.reward));
        wxLogDebug("CartPoleFrame::OnTimer() step=%d next_state=%s", step_count, result.next_state.ToString());
        ANET_CHECK_DEVICE(result.next_state.obs, torch::kCPU);
        ANET_CHECK_DEVICE(result.next_state.done, torch::kCPU);
        ANET_CHECK_DEVICE(result.next_state.truncated, torch::kCPU);
        ANET_CHECK_DEVICE(result.reward, torch::kCPU);
        ANET_CHECK_SHAPE(result.next_state.obs, { 1, ANET_SHAPE_ENDANY });
        ANET_CHECK_SHAPE(result.next_state.done, { 1 });
        ANET_CHECK_SHAPE(result.next_state.truncated, { 1 });
        ANET_CHECK_SHAPE(result.reward, { 1 });
       
        // Agent更新
        anet::rl::BatchExperience exp({ state_, action_info, result.reward, result.next_state });
        auto update_result = agent->UpdateFromBatch(exp);

        // 更新後処理
        notifier_.Notify(update_result, exp, step_count);
        state_ = result.next_state.Clone();
        last_reward = result.reward.squeeze(0).item<float>();
        train_total_reward_ += last_reward;

        // ステップ数インプリメント（グローバルなステップ数）
        step_count++;

        canvas->SetState(env->get_x(), env->get_theta(), env->get_x_dot(), env->get_theta_dot());
        canvas->SetAction(action_info.action);
        canvas->SetReward(last_reward);
        canvas->Refresh();

        //エピソード終了判定
        if (result.next_state.IsDone() || result.next_state.IsTruncated()) {
            episode_count++;

            // プロット更新
            plotPanel->AddReward(train_total_reward_);

            // Canvas更新（エピソード終了）
            //canvas->SetState(env->get_x(), env->get_theta(), env->get_x_dot(), env->get_theta_dot());
            //canvas->SetAction(action);
            //canvas->SetReward(last_reward);
            //canvas->Refresh();

            // 学習状況評価
            float eval_total_reward = 0.0f;
            if (episode_count % config_->eval_interval == 0) {
                eval_count_++;
                {   // ターゲットネットワークによる評価
                    auto state = env->Reset(anet::rl::RunMode::Eval1);
                    auto total_reward = 0.0f;
                    bool done = false;
                    bool truncated = false;
                    do {
                        auto action = agent->MakeAction(state.obs, anet::rl::RunMode::Eval1);
                        auto env_result = env->DoStep(action.action);
                        total_reward += env_result.reward.squeeze(0).item<float>();
                        state = env_result.next_state.Clone();
                        done = env_result.next_state.IsDone();
                        truncated = env_result.next_state.IsTruncated();
                    } while (!done && !truncated);
                    eval_total_reward = total_reward;

                    // ログ
                    anet::MetricsLogger::Instance()->log_scalar("10_epsode/02_eval_reward", episode_count, total_reward);
                    anet::MetricsLogger::Instance()->log_scalar("11_eval/01_target_reward", step_count, total_reward);

                    // ターゲットネットワークによる評価の終了状態を描画
                    //canvas->SetState(env->get_x(), env->get_theta(), env->get_x_dot(), env->get_theta_dot());
                    //canvas->SetAction(action);
                    //canvas->SetReward(env_result.reward);
                    //canvas->Refresh();
                }
                {   // メインネットワークによる評価
                    auto state = env->Reset(anet::rl::RunMode::Eval2);
                    auto total_reward = 0.0f;
                    bool done = false;
                    bool truncated = false;
                    do {
                        auto action = agent->MakeAction(state.obs, anet::rl::RunMode::Eval2);
                        auto env_result = env->DoStep(action.action);
                        total_reward += env_result.reward.squeeze(0).item<float>();
                        state = env_result.next_state.Clone();
                        done = env_result.next_state.IsDone();
                        truncated = env_result.next_state.IsTruncated();
                    } while (!done && !truncated);
                    anet::MetricsLogger::Instance()->log_scalar("11_eval/02_policy_reward", step_count, total_reward);
                }
            }

            // ログ
            auto eps_step = step_count - last_episode_step;
            wxLogInfo("Episode finished. eps=%d total_step=%d  eps_step=%d train_reward=%f eval_reward=%f",
                episode_count, step_count, eps_step, train_total_reward_, eval_total_reward);
            anet::MetricsLogger::Instance()->log_scalar("10_epsode/01_total_reward", episode_count, train_total_reward_);

            // 環境リセット
            state_ = env->Reset();

            // エピソードが終わったので次エピソード準備
            last_episode_step = step_count;
            train_total_reward_ = 0.0f;
            //break;
        }
    }

    // --- カート位置・角度の描画更新 ---
    //canvas->SetState(env->get_x(), env->get_theta(), env->get_x_dot(), env->get_theta_dot());
    //canvas->SetAction(action);
    //canvas->SetReward(last_reward);
    //canvas->Refresh();

    if ((config_->train_exit_step > 0) && (step_count >= config_->train_exit_step)) {
        anet::MetricsLogger::Instance()->flush();
        wxGetApp().Exit();
    }
    if ((config_->train_pause_step > 0) && (step_count >= config_->train_pause_step) && !auto_pause_done_) {
        auto_pause_done_ = true;
        training_paused = true;
    }

    // タイマー再開
	this->timer.Start();
}
