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
    uint64_t seed = 0;

    CartPoleFrame::Config(const anet::ConfigData& configData) : anet::Config(configData, "train", "CartPoleFrame") {
        ANET_APPLY_CONFIG(configData, timer_ms);
        ANET_APPLY_CONFIG(configData, step_per_frame);
        ANET_APPLY_CONFIG(configData, eval_interval);
        ANET_APPLY_CONFIG(configData, train_pause_step);
        ANET_APPLY_CONFIG(configData, train_exit_step);
        ANET_APPLY_CONFIG(configData, seed);
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
    device_(torch::kCUDA),
    timer(this, wxID_ANY),
    msec_per_step_ema_(0.001)
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

    if (config_->seed == 0) {
        rnd_ = std::make_shared<anet::RandomGenerator>(true, true);
    }
    else {
        rnd_ = std::make_shared<anet::RandomGenerator>(config_->seed, true, true);
    }
    wxLogInfo("CartPoleRLGUI started.");

    // パラメータ記録
    wxLogInfo("seed=%lld", rnd_->GetSeed());
    wxLogInfo("train.preset=%s confg=%s", wxGetApp().GetConfig("train").Get("preset"), config_->ToStdString());
    anet::MetricsLogger::Instance()->LogJson("train/params", config_->ToJson());
    anet::MetricsLogger::Instance()->Flush();

    // ENV生成 ---
    env_ = std::make_unique<CartPoleEnv>(rnd_);
    auto env_spec = env_->GetSpec();
    wxLogInfo("env_spec=" + env_spec.ToString());

    // ランダム方策で環境難易度評価
    auto eval_result = anet::rl::EvaluateEnvironmentDifficulty(*env_, 100);
    anet::MetricsLogger::Instance()->LogJson("eval_env", eval_result.ToJson());

    // --- Agent生成 ---
    anet::ConfigData agentConfig = wxGetApp().GetConfig("agent");
    agent_ = std::make_shared<anet::rl::DQNAgent>(agentConfig, env_spec, device_, rnd_);

    // MetricsLogObserver
    auto metrics_obs = std::make_shared<anet::rl::MetricsLogObserver>();

    // HeatMapObserver
    auto flags = 
        //anet::HeatMapFlags::HM_LogScaleValue | 
        anet::HeatMapFlags::HM_AutoNormValue
        | anet::HeatMapFlags::HM_AutoScaleAxis
        //| anet::HeatMapFlags::HM_LogScaleAxis
        | anet::HeatMapFlags::HM_SumMode; // | anet::HeatMapFlags::HM_ShowZeroLine;

    anet::rl::HeatMapObserverConfig visit_heat_obs_config {
        256,    // width
        256,    // height
        100,    // log_interval 
        30000,  // max_points
        flags   // flags
        -1,     // image_width
        -1,     // image_height
    };
    auto visit_x_probe = std::make_shared<anet::StateAxisProbe>(0, &env_spec.state_spec, true);
    auto visit_theta_probe = std::make_shared<anet::StateAxisProbe>(2, &env_spec.state_spec, true);
    auto visit_theta_dot_probe = std::make_shared<anet::StateAxisProbe>(3, &env_spec.state_spec, true);
    auto visit_reward_probe = std::make_shared<anet::RewardProbe>(nullptr);

    auto rep_x_probe = std::make_shared<anet::AgentTensorVectorProbe>("replaybuffer.next_states", 0, &env_spec.state_spec);
    auto rep_theta_probe = std::make_shared<anet::AgentTensorVectorProbe>("replaybuffer.next_states", 2, &env_spec.state_spec);
    auto rep_theta_dot_probe = std::make_shared<anet::AgentTensorVectorProbe>("replaybuffer.next_states", 3, &env_spec.state_spec);
    auto rep_reward_probe = std::make_shared<anet::AgentTensorVectorProbe>("replaybuffer.rewards", -1, &env_spec.state_spec);

    auto visit_02_reward = std::make_shared<anet::rl::HeatMapObserver>(
        "43_agent_img/02_hm_visit_02", visit_heat_obs_config, visit_x_probe, visit_theta_probe, visit_reward_probe);
    auto visit_23_reward = std::make_shared<anet::rl::HeatMapObserver>(
        "43_agent_img/03_hm_visit_23", visit_heat_obs_config, visit_theta_probe, visit_theta_dot_probe, visit_reward_probe);

    auto replay_02_reward = std::make_shared<anet::rl::HeatMapVectorObserver>(
        "43_agent_img/12_hm_rep_02", visit_heat_obs_config, rep_x_probe, rep_theta_probe, rep_reward_probe);
    auto replay_23_reward = std::make_shared<anet::rl::HeatMapVectorObserver>(
        "43_agent_img/13_hm_rep_23", visit_heat_obs_config, rep_theta_probe, rep_theta_dot_probe, rep_reward_probe);


    // TimeHistogramObserver
    anet::rl::TimeHistogramObserverConfig q_hist_obs_config {
        256,    // x bins
        1920,   // y max_frames
        512,    // image_height
        1920,   // image_width
        anet::TimeFrameMode::Scale,             // mode
        flags | anet::HeatMapFlags::HM_FlipY,   // flags
        100,    // log_interval
        20,     // frame_interval
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        1.0f// alpha = 0.05f
    };
    auto q_probe = std::make_shared<anet::BatchUpdateResultTensorProbe>("max_q");
    auto q_hist_obs = std::make_shared<anet::rl::TimeHistogramObserver>(
        "44_agent_img/04_thg_t", q_hist_obs_config, q_probe);

    //SweepedHeatMapObserver(
    //    const std::string & tag,
    //    const SweepedHeatMapObserverConfig & config,
    //    std::shared_ptr<ISweepInputGenerator> input_gen,
    //    ApplyNNFn apply_nn_fn,
    //    std::shared_ptr<ISweepOutputExtractor> output_ext);

    anet::rl::SweepedHeatMapObserverConfig q_sweep_obs_config {
        100,    // log_interval
        flags,  // flags
        128,    // grid_width
        128,    // grid_height
        -1,     // image_width
        -1,     // image_height
    };
    auto proc_x_theta_qmax = std::make_shared<anet::RLStateSweepProcessor>(
        env_spec.state_spec,
        0,  // x_index = x
        2   // y_index = theta
    );
    auto proc_theta_thetadot_qmax = std::make_shared<anet::RLStateSweepProcessor>(
        env_spec.state_spec,
        2,  // x_index = theta
        3   // y_index = theta_dot
    );
    //RLStateSweepProcessor(
    //    const anet::rl::StateSpec& state_spec,
    //    int x_index,
    //    int y_index,
    //    ValueExtractFunction value_extract_fn = &extractor::MaxExtractor,
    //    const torch::Device& device = torch::kCUDA,
    //    std::optional<torch::Tensor> base_state = std::nullopt,
    //    std::optional<float> x_min_override = std::nullopt,
    //    std::optional<float> x_max_override = std::nullopt,
    //    std::optional<float> y_min_override = std::nullopt,
    //    std::optional<float> y_max_override = std::nullopt
    //);
    auto proc_theta_thetadot_qdiff = std::make_shared<anet::RLStateSweepProcessor>(
        env_spec.state_spec,
        0,  // x_index = x
        2,  // y_index = theta
        std::bind(
            &anet::extractor::DiffIndexExtractor,
            std::placeholders::_1,
            std::placeholders::_2,
            0,
            1)
    );
    auto proc_theta_thetadot_qdiff_mask = std::make_shared<anet::RLStateSweepProcessor>(
        env_spec.state_spec,
        0,  // x_index = x
        2,  // y_index = theta
        std::bind(
            &anet::extractor::BoundaryMaskFromQdiffAuto,
            std::placeholders::_1,
            std::placeholders::_2,
            env_spec.action_spec.ActionCount(),
            0,
            1)
    );
    /// policy_netとtarget_netの差分を見る
    auto proc_theta_thetadot_pair_qdelta = std::make_shared<anet::RLStateSweepProcessor>(
        env_spec.state_spec,
        0,  // x_index = x
        2,  // y_index = theta
        std::bind(
            &anet::extractor::PairDiffExtractor,
            std::placeholders::_1,
            std::placeholders::_2,
            env_spec.action_spec.ActionCount())
    );
    /// @brief QdeltaとQmaxの合成。
    /// <summary>
    //    Qdelta 高 × Qmax 高
    //    → 発散で地形が壊れて target 追従不能の領域
    //    Qdelta 高 × Qmax 低
    //    → target追従不足（でも発散ではない）
    //    Qdelta 低 × Qmax 高
    //    → Qの発散だけが起きている領域（target が遅れて青くなる前兆）
    //    両方低
    //    → 安定
    /// </summary>
    auto proc_theta_thetadot_pair_combo_qdeltaqmax = std::make_shared<anet::RLStateSweepProcessor>(
        env_spec.state_spec,
        0,  // x_index = x
        2,  // y_index = theta
        std::bind(
            &anet::extractor::QdeltaQmaxCombinedAuto,
            std::placeholders::_1,
            std::placeholders::_2,
            env_spec.action_spec.ActionCount()
        )
    );
    auto proc_theta_thetadot_pair_combo_qdelta_qdiff = std::make_shared<anet::RLStateSweepProcessor>(
        env_spec.state_spec,
        0,  // x_index = x
        2,  // y_index = theta
        std::bind(
            &anet::extractor::QdeltaQdiffCombinedAuto,
            std::placeholders::_1,
            std::placeholders::_2,
            env_spec.action_spec.ActionCount(),
            0,  // action_index_a (LEFT)
            1   // action_index_b (RIGHT)
        )
    );
    auto proc_theta_thetadot_pair_combo_qdelta_qdiffmasked = std::make_shared<anet::RLStateSweepProcessor>(
        env_spec.state_spec,
        0,  // x_index = x
        2,  // y_index = theta
        std::bind(
            &anet::extractor::BoundaryMaskedQdeltaAuto,
            std::placeholders::_1,
            std::placeholders::_2,
            env_spec.action_spec.ActionCount(),
            0,  // action_index_a (LEFT)
            1   // action_index_b (RIGHT)
        )
    );

    using StrMap = std::unordered_map<std::string, std::string>;


    anet::rl::TensorFunction policy_forward = agent_->GetTensorFunction("policy_net.forward");
    anet::rl::TensorFunction qpair_forward = agent_->GetTensorFunction("q_pair.forward");
    auto q_sweep_obs_02_qmax = std::make_shared<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/05_shm_02_qmax", q_sweep_obs_config, proc_x_theta_qmax, policy_forward, proc_x_theta_qmax);
    auto q_sweep_obs_23_qmax = std::make_shared<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/06_shm_23_qmax", q_sweep_obs_config, proc_theta_thetadot_qmax, policy_forward, proc_theta_thetadot_qmax);
    //q_sweep_obs_23_q0_ = std::make_shared<anet::rl::SweepedHeatMapObserver>(
    //    "43_agent_img/07_shm_23_q0", q_sweep_obs_config, proc_theta_thetadot_q0, policy_forward, proc_theta_thetadot_q0);
    auto q_sweep_obs_02_qdiff = std::make_shared<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/08_shm_02_qdiff", q_sweep_obs_config, proc_theta_thetadot_qdiff, policy_forward, proc_theta_thetadot_qdiff);
    auto q_sweep_obs_02_qdiff_mask = std::make_shared<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/09_shm_02_qdiff_mask", q_sweep_obs_config, proc_theta_thetadot_qdiff_mask, policy_forward, proc_theta_thetadot_qdiff_mask);
    auto q_sweep_obs_02_qdelta = std::make_shared<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/11_shm_02_qdelta", q_sweep_obs_config, proc_theta_thetadot_pair_qdelta, qpair_forward, proc_theta_thetadot_pair_qdelta);
    auto q_sweep_obs_02_combo_qdqmax = std::make_shared<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/12_shm_02_qdelta_qmax", q_sweep_obs_config, proc_theta_thetadot_pair_combo_qdeltaqmax, qpair_forward, proc_theta_thetadot_pair_combo_qdeltaqmax);
    auto q_sweep_obs_02_combo_qdqdiff = std::make_shared<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/13_shm_02_qdelta_qdiff", q_sweep_obs_config, proc_theta_thetadot_pair_combo_qdelta_qdiff, qpair_forward, proc_theta_thetadot_pair_combo_qdelta_qdiff);
    auto q_sweep_obs_02_combo_qdelta_qdiff_masked = std::make_shared<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/14_shm_02_qdelta_qdiff-masked", q_sweep_obs_config, proc_theta_thetadot_pair_combo_qdelta_qdiffmasked, qpair_forward, proc_theta_thetadot_pair_combo_qdelta_qdiffmasked,
        StrMap {
            { "46_agent_imgsc/02qdd_raw_qdelta_mean", "raw_qdelta_mean" },
            { "46_agent_imgsc/02qdd_raw_qdelta_max", "raw_qdelta_max" },
            { "46_agent_imgsc/02qdd_raw_boundary_mean", "raw_boundary_mean" },
            { "46_agent_imgsc/02qdd_boundary_area", "boundary_area"  },
            { "46_agent_imgsc/02qdd_combined_mean", "combined_mean" },
            { "46_agent_imgsc/02qdd_combined_max", "combined_max"  }
        });

    // --- Obserber登録 ---
    notifier_.AddObserver(metrics_obs);
    notifier_.AddObserver(visit_02_reward);
    notifier_.AddObserver(visit_23_reward);
    notifier_.AddObserver(replay_02_reward);
    notifier_.AddObserver(replay_23_reward);
    notifier_.AddObserver(q_hist_obs);
    notifier_.AddObserver(q_sweep_obs_02_qmax);
    notifier_.AddObserver(q_sweep_obs_23_qmax);
    //notifier_.AddObserver(q_sweep_obs_23_q0_);
    notifier_.AddObserver(q_sweep_obs_02_qdiff);
    notifier_.AddObserver(q_sweep_obs_02_qdiff_mask);
    notifier_.AddObserver(q_sweep_obs_02_qdelta);
    notifier_.AddObserver(q_sweep_obs_02_combo_qdqmax);
    notifier_.AddObserver(q_sweep_obs_02_combo_qdqdiff);
    notifier_.AddObserver(q_sweep_obs_02_combo_qdelta_qdiff_masked);

    // --- 環境初期化 ---
    state_ = env_->Reset();  // ← reset() は 初期状態 を返す
    ANET_CHECK_DEVICE_CPU_MSG(state_.obs, "Initial state");
    ANET_CHECK_SHAPE(state_.obs, { ANET_SHAPE_ANY, 4 });

    // --- タイマー開始 ---
    Bind(wxEVT_TIMER, &CartPoleFrame::OnTimer, this);
    timer.Start(config_->timer_ms);  // 学習＆描画更新
    //auto now = std::chrono::high_resolution_clock::now();
    //auto cnt = now.time_since_epoch().count();


    // 時間計測開始
    last_time_ = std::chrono::high_resolution_clock::now();
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

    auto env_spec = env_->GetSpec();

    // --- 学習ステップを複数回回す ---
    //auto action = agent->select_action(state);
    float last_reward = 0.0f;
    //anet::rl::BatchStepResult step_result;
    for (int i = 0; i < config_->step_per_frame; ++i) {
        if ((config_->train_exit_step > 0) && (step_count_ >= config_->train_exit_step)) {
            anet::MetricsLogger::Instance()->Flush();
            wxGetApp().Exit();
        }

        // Stateチェック
        wxLogDebug("CartPoleFrame::OnTimer() step=%d state=%s", step_count_, state_.ToString());
        ANET_ASSERT(env_spec.state_spec.MatchesShape(state_.obs));
        ANET_ASSERT(env_spec.state_spec.MatchesRange(state_.Flatten().obs));

        // 行動選択
        auto action_info = agent_->MakeAction(state_);
        wxLogDebug("CartPoleFrame::OnTimer() step=%d action=%s", step_count_, action_info.ToString());
        ANET_CHECK_DEVICE(action_info.action, device_);

        // 環境ステップ実行
        anet::rl::BatchStepResult result = env_->DoStep(action_info.action);    // next_state, reward, done, truncated
        wxLogDebug("CartPoleFrame::OnTimer() step=%d reward=%s", step_count_, anet::ToString(result.reward));
        wxLogDebug("CartPoleFrame::OnTimer() step=%d next_state=%s", step_count_, result.next_state.ToString());
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
        auto update_result = agent_->UpdateFromBatch(exp);

        // 更新後処理
        notifier_.Notify(step_count_, agent_, exp, update_result);
        state_ = result.next_state.Clone();
        last_reward = result.reward.squeeze(0).item<float>();
        train_total_reward_ += last_reward;

        // msec per step
        std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
        auto msec_diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time_).count();
        if (step_count_ != 0) { // 0ステップ目は誤差が大きいので
            msec_per_step_ema_.Update(msec_diff);
            anet::MetricsLogger::Instance()->LogScalar("90_train/01_msec_per_step", step_count_, msec_diff);
            anet::MetricsLogger::Instance()->LogScalar("90_train/01_msec_per_step_ema", step_count_, msec_per_step_ema_.Value());
        }
        last_time_ = now;

        // ステップ数インプリメント（グローバルなステップ数）
        step_count_++;

        canvas->SetState(env_->get_x(), env_->get_theta(), env_->get_x_dot(), env_->get_theta_dot());
        canvas->SetAction(action_info.action);
        canvas->SetReward(last_reward);
        canvas->Refresh();

        //エピソード終了判定
        if (result.next_state.IsDone() || result.next_state.IsTruncated()) {
            episode_count_++;

            // プロット更新
            plotPanel->AddReward(train_total_reward_);

            // Canvas更新（エピソード終了）
            //canvas->SetState(env->get_x(), env->get_theta(), env->get_x_dot(), env->get_theta_dot());
            //canvas->SetAction(action);
            //canvas->SetReward(last_reward);
            //canvas->Refresh();

            // 学習状況評価
            float eval_total_reward = 0.0f;
            if (episode_count_ % config_->eval_interval == 0) {
                eval_count_++;
                {   // ターゲットネットワークによる評価
                    auto state = env_->Reset(anet::rl::RunMode::Eval1);
                    auto total_reward = 0.0f;
                    bool done = false;
                    bool truncated = false;
                    do {
                        auto action = agent_->MakeAction(state, anet::rl::RunMode::Eval1);
                        auto env_result = env_->DoStep(action.action);
                        total_reward += env_result.reward.squeeze(0).item<float>();
                        state = env_result.next_state.Clone();
                        done = env_result.next_state.IsDone();
                        truncated = env_result.next_state.IsTruncated();
                    } while (!done && !truncated);
                    eval_total_reward = total_reward;

                    // ログ
                    anet::MetricsLogger::Instance()->LogScalar("10_epsode/02_eval_reward", episode_count_, total_reward);
                    anet::MetricsLogger::Instance()->LogScalar("11_eval/01_target_reward",step_count_, total_reward);

                    // ターゲットネットワークによる評価の終了状態を描画
                    //canvas->SetState(env->get_x(), env->get_theta(), env->get_x_dot(), env->get_theta_dot());
                    //canvas->SetAction(action);
                    //canvas->SetReward(env_result.reward);
                    //canvas->Refresh();
                }
                {   // メインネットワークによる評価
                    auto state = env_->Reset(anet::rl::RunMode::Eval2);
                    auto total_reward = 0.0f;
                    bool done = false;
                    bool truncated = false;
                    do {
                        auto action = agent_->MakeAction(state, anet::rl::RunMode::Eval2);
                        auto env_result = env_->DoStep(action.action);
                        total_reward += env_result.reward.squeeze(0).item<float>();
                        state = env_result.next_state.Clone();
                        done = env_result.next_state.IsDone();
                        truncated = env_result.next_state.IsTruncated();
                    } while (!done && !truncated);
                    anet::MetricsLogger::Instance()->LogScalar("11_eval/02_policy_reward", step_count_, total_reward);
                }

                anet::MetricsLogger::Instance()->Flush();
            }

            // ログ
            auto eps_step = step_count_ - last_episode_step_;
            wxLogInfo("Episode finished. eps=%d total_step=%d  eps_step=%d train_reward=%f eval_reward=%f",
                episode_count_, step_count_, eps_step, train_total_reward_, eval_total_reward);
            anet::MetricsLogger::Instance()->LogScalar("10_epsode/01_total_reward",
                episode_count_, train_total_reward_);

            // 環境リセット
            state_ = env_->Reset();

            // エピソードが終わったので次エピソード準備
            last_episode_step_ = step_count_;
            train_total_reward_ = 0.0f;
            //break;
        }
    }

    // --- カート位置・角度の描画更新 ---
    //canvas->SetState(env->get_x(), env->get_theta(), env->get_x_dot(), env->get_theta_dot());
    //canvas->SetAction(action);
    //canvas->SetReward(last_reward);
    //canvas->Refresh();

    if ((config_->train_exit_step > 0) && (step_count_ >= config_->train_exit_step)) {
        anet::MetricsLogger::Instance()->Flush();
        wxGetApp().Exit();
    }
    if ((config_->train_pause_step > 0) && (step_count_ >= config_->train_pause_step) && !auto_pause_done_) {
        auto_pause_done_ = true;
        training_paused = true;
    }

    // タイマー再開
	this->timer.Start();
}
