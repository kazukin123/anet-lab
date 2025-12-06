#include "CartPoleFrame.hpp"
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <torch/torch.h>
#include <wx/stattext.h>
#include <wx/sizer.h>

#include "anet/tensor_check.hpp"
#include "anet/tensor_utils.hpp"
#include "anet/profile.hpp"
#include "anet/eval_env.hpp"
#include "anet/env.hpp"
#include "CartPoleCanvas.hpp"
#include "app.hpp"


struct CartPoleFrame::Config : public anet::Config
{
    uint64_t seed = 0;
    int batch_size = 1;

    int timer_ms = 20;
    int step_per_frame = 10;
    int eval_interval = 50;
    int train_pause_step = -1;
    int train_exit_step = -1; //110000;

	bool enable_image_log = true;
    int perf_log_interval = 100;

    CartPoleFrame::Config(const anet::ConfigData& config_data) : anet::Config(config_data, "train")
    {
        ANET_READ_CONFIG(config_data, seed);
        ANET_READ_CONFIG(config_data, batch_size);
        ANET_READ_CONFIG(config_data, timer_ms);
        ANET_READ_CONFIG(config_data, step_per_frame);
        ANET_READ_CONFIG(config_data, eval_interval);
        ANET_READ_CONFIG(config_data, train_pause_step);
        ANET_READ_CONFIG(config_data, train_exit_step);
        ANET_READ_CONFIG(config_data, enable_image_log);
        ANET_READ_CONFIG(config_data, perf_log_interval);
    }
};

wxBEGIN_EVENT_TABLE(CartPoleFrame, wxFrame)
EVT_TIMER(wxID_ANY, CartPoleFrame::OnTimer)
EVT_LEFT_DOWN(CartPoleFrame::OnMouseClick)
wxEND_EVENT_TABLE()

//const torch::Device ENV_DEVICE = torch::kCPU;
const torch::Device AGENT_DEVICE = torch::kCUDA;

CartPoleFrame::CartPoleFrame(const wxString& title)
    : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxSize(800, 800)),
    device_agent_(AGENT_DEVICE),
    timer(this, wxID_ANY),
    train_reward_ema_(0.001),
    train_step_per_sec_ema_(0.001),
    exp_step_per_sec_ema_(0.001)
{
	// --- 設定読み込み ---
    auto config_data = wxGetApp().GetConfig();
    config_ = std::make_unique<CartPoleFrame::Config>(config_data);

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
#if ANET_ENABLE_DEBUGINFO
    wxLog::SetLogLevel(wxLOG_Debug);
#endif

    // --- ログ出力先をこのクラスに設定 ---
    wxLog::SetActiveTarget(this);

    if (config_->seed == 0) {
        master_seed_ = std::make_unique<anet::MasterSeedManager>();
    } else {
        master_seed_ = std::make_unique<anet::MasterSeedManager>(config_->seed);
    }
    wxLogInfo("CartPoleRLGUI started.");

    // seed値生成
    auto grobal_seed = master_seed_->GetMasterSeed();
    auto env_seed = master_seed_->GetGroupSeed("env");
    auto agent_seed = master_seed_->GetGroupSeed("agent");
    wxLogInfo("grobal_seed=%lld env_seed=%lld agent_seed=%lld", grobal_seed, env_seed, agent_seed);

    // パラメータ記録
    //wxLogInfo("train.preset=%s confg=%s", wxGetApp().GetConfig("train").Get("preset"), config_->ToString());
    anet::MetricsLogger::Instance()->LogJson("train/seed",
        { "grobal_seed", grobal_seed, "agent_seed", agent_seed, "env_seed", env_seed });
    anet::MetricsLogger::Instance()->LogJson("train/config", config_->ToJson());
    anet::MetricsLogger::Instance()->Flush();

    // ENV生成
    anet::rl::DefaultBatchEnvFactoryConfig env_config(config_data);
    wxLogInfo("env_config=%s", env_config.ToString());
    auto env_factory = anet::rl::DefaultBatchEnvFactory(env_config, config_data, config_->batch_size, env_seed);
    auto env_device = env_factory.GetDevice();
    auto single_env_factory = env_factory.GetSingleFactory();
    env_ = env_factory.CreateBatchEnv();
    if (env_ == nullptr) {
        wxLogError("Failed to create env.");
        return;
    }

    auto batch_env_spec = env_->GetBatchSpec();
    auto env_spec = env_->GetSpec();
    wxLogInfo("batch_env_spec=" + batch_env_spec.ToString());
    wxLogInfo("env_spec=" + env_spec.ToString());
    anet::MetricsLogger::Instance()->LogJson("env/batch_env_spec", batch_env_spec.ToJson());
    anet::MetricsLogger::Instance()->LogJson("env/env_spec", env_spec.ToJson());
    anet::MetricsLogger::Instance()->Flush();

    // ランダム方策で環境難易度評価
    /// @todo EvaluateEnvironmentDifficultyを復活
    //auto eval_result = anet::rl::EvaluateEnvironmentDifficulty(*env_, 100);
    //anet::MetricsLogger::Instance()->LogJson("eval_env", eval_result.ToJson());

    // Agent生成
    anet::rl::DQNAgentConfig agent_config(config_data);
    agent_ = std::make_shared<anet::rl::DQNAgent>(agent_config, batch_env_spec, env_spec, device_agent_, agent_seed);

    // EpisodeEvalObserver
    notifier_.Attach<anet::rl::EpisodeEvalObserver>(
        "11_eval/01_target_reward", single_env_factory, config_data, env_device, anet::rl::RunMode::Eval1,
        config_->eval_interval, config_->eval_interval);
    notifier_.Attach<anet::rl::EpisodeEvalObserver>(
        "11_eval/02_policy_reward", single_env_factory, config_data, env_device, anet::rl::RunMode::Eval2,
        config_->eval_interval, config_->eval_interval);

    // MetricsLogObserver
    notifier_.Attach<anet::rl::MetricsLogObserver>();

    // log
    auto plot_reward = std::make_shared<float>(0.0f);
    notifier_.Attach<anet::rl::FunctionObserver>(
        //[&, plot_reward](int step,
        [this, plot_reward](const anet::rl::PostUpdateEvent& event)
        {
            auto train_step = event.counts.train_step;

            float step_reward = event.batch_exp.reward.mean().item<float>();
            train_reward_ema_.Update(step_reward);
            anet::MetricsLogger::Instance()->LogScalar(
                "10_train/10_total_reward", train_step, train_reward_ema_.Value());

            *plot_reward += step_reward;
            if (event.counts.train_step % 10 == 0) {
                this->plotPanel->AddReward(*plot_reward);
                *plot_reward = 0;
            }
            if (event.counts.train_step % 100 == 0) {
                wxLogInfo("step=%llu  train_mean_reward=%f", train_step, train_reward_ema_.Value());
            }
        }
    );

    // 画像ベースのObserverを初期化
    initImageLogObservers(env_spec);

    // --- 環境初期化 ---
    state_ = env_->Reset();  // ← reset() は 初期状態 を返す
    ANET_CHECK_DEVICE_CPU_MSG(state_.obs, "Initial state");
    ANET_CHECK_SHAPE(state_.obs, { ANET_SHAPE_ANY, 4 });

    // --- タイマー開始 ---
    Bind(wxEVT_TIMER, &CartPoleFrame::OnTimer, this);
    timer.Start(config_->timer_ms);  // 学習＆描画更新

    // 時間計測開始
    start_time_ = std::chrono::high_resolution_clock::now();
    last_time_ = start_time_;
}

void CartPoleFrame::initImageLogObservers(const anet::rl::EnvSpec& env_spec)
{
    // 有効設定チェック
    if (!config_->enable_image_log)
        return;

    // flags
    auto flags =
        //anet::HeatMapFlags::HM_LogScaleValue | 
        anet::HeatMapFlags::HM_AutoNormValue
        | anet::HeatMapFlags::HM_AutoScaleAxis
        //| anet::HeatMapFlags::HM_LogScaleAxis
        | anet::HeatMapFlags::HM_SumMode; // | anet::HeatMapFlags::HM_ShowZeroLine;

    // ---- Experience visit ----

    //anet::rl::HeatMapObserverConfig visit_heat_obs_config{
    //    512,    // width
    //    512,    // height
    //    100,    // log_interval 
    //    30000,  // max_points
    //    flags   // flags
    //    - 1,     // image_width
    //    -1,     // image_height
    //};
    //auto visit_x_probe = std::make_shared<anet::rl::BatchExperienceStateProbe>(0, &env_spec.state_spec, true);
    //auto visit_theta_probe = std::make_shared<anet::rl::BatchExperienceStateProbe>(2, &env_spec.state_spec, true);
    //auto visit_theta_dot_probe = std::make_shared<anet::rl::BatchExperienceStateProbe>(3, &env_spec.state_spec, true);
    //auto visit_reward_probe = std::make_shared<anet::rl::BatchExperienceRewardProbe>(nullptr);
    
    //notifier_.Attach<anet::rl::HeatMapVectorObserver>(
    //    "43_agent_img/02_hm_visit_02", visit_heat_obs_config, visit_x_probe, visit_theta_probe, visit_reward_probe);
    //notifier_.Attach<anet::rl::HeatMapVectorObserver>(
    //    "43_agent_img/03_hm_visit_23", visit_heat_obs_config, visit_theta_probe, visit_theta_dot_probe, visit_reward_probe);


    // ---- ReplayBuffer ----

    auto rep_x_probe = std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 0, &env_spec.state_spec);
    auto rep_theta_probe = std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 2, &env_spec.state_spec);
    auto rep_theta_dot_probe = std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 3, &env_spec.state_spec);
    auto rep_reward_probe = std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::REWARD, -1, &env_spec.state_spec);

    anet::rl::HeatMapObserverConfig replay_heat_obs_config{
        512,    // width
        512,    // height
        100,    // log_interval 
        30000,  // max_points
        flags   // flags
        - 1,     // image_width
        -1,     // image_height
    };

    //auto auto_scale_mode = anet::rl::AgentTensorVectorProbe::AutoScaleMode::GLOBAL;   // サンプル値でmin/max調整
    auto auto_scale_mode = anet::rl::AgentTensorVectorProbe::AutoScaleMode::DISABLE;    // EnvSpecで固定
    std::vector<std::shared_ptr<anet::rl::VectorProbe>> probes_3axis = {
        std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 0, &env_spec.state_spec, nullptr, auto_scale_mode),
        std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 2, &env_spec.state_spec, nullptr, auto_scale_mode),
        std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 3, &env_spec.state_spec, nullptr, auto_scale_mode),
    };
    std::vector<std::shared_ptr<anet::rl::VectorProbe>> probes_4axis = {
        std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 0, &env_spec.state_spec, nullptr, auto_scale_mode),
        std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 1, &env_spec.state_spec, nullptr, auto_scale_mode),
        std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 2, &env_spec.state_spec, nullptr, auto_scale_mode),
        std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 3, &env_spec.state_spec, nullptr, auto_scale_mode),
    };

    notifier_.Attach<anet::rl::HeatMapVectorObserver>(
        "43_agent_img/12_hm_rep_02", replay_heat_obs_config, rep_x_probe, rep_theta_probe, rep_reward_probe);
    notifier_.Attach<anet::rl::HeatMapVectorObserver>(
        "43_agent_img/13_hm_rep_23", replay_heat_obs_config, rep_theta_probe, rep_theta_dot_probe, rep_reward_probe);
    notifier_.Attach<anet::rl::MultiPairHeatMapObserver>(
        "43_agent_img/21_hm_rep_multi3",
        replay_heat_obs_config,
        probes_3axis,
        rep_reward_probe);
    notifier_.Attach<anet::rl::MultiPairHeatMapObserver>(
        "43_agent_img/22_hm_rep_multi4",
        replay_heat_obs_config,
        probes_4axis,
        rep_reward_probe);

    // ---- TimeHistogram ----

    anet::rl::TimeHistogramObserverConfig q_hist_obs_config{
        256,    // x bins
        1920,   // y max_frames
        512,    // image_height
        1920,   // image_width
        anet::TimeFrameMode::Scale,             // mode
        flags | anet::HeatMapFlags::HM_FlipY | anet::HeatMapFlags::HM_LogScaleAxis,   // flags
        100,    // log_interval
        20,     // frame_interval
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        1.0f// alpha = 0.05f
    };
    auto q_probe = std::make_shared<anet::rl::BatchUpdateResultTensorToVectorProbe>("max_q");
    notifier_.Attach<anet::rl::TimeHistogramObserver>(
        "44_agent_img/04_thg_t", q_hist_obs_config, q_probe);

    // ---- SweepedHeatMap ----

    anet::rl::SweepedHeatMapObserverConfig q_sweep_obs_config{
        100,    // log_interval
        flags,  // flags
        128,    // grid_width
        128,    // grid_height
        -1,     // image_width
        -1,     // image_height
    };
    auto proc_x_theta_qmax = std::make_shared<anet::rl::StateSweepProcessor>(
        env_spec.state_spec,
        0,  // x_index = x
        2   // y_index = theta
    );
    auto proc_theta_thetadot_qmax = std::make_shared<anet::rl::StateSweepProcessor>(
        env_spec.state_spec,
        2,  // x_index = theta
        3   // y_index = theta_dot
    );
    auto proc_theta_thetadot_qdiff = std::make_shared<anet::rl::StateSweepProcessor>(
        env_spec.state_spec,
        0,  // x_index = x
        2,  // y_index = theta
        std::bind(
            &anet::rl::extractor::DiffIndexExtractor,
            std::placeholders::_1,
            std::placeholders::_2,
            0,
            1)
    );
    auto proc_theta_thetadot_qdiff_mask = std::make_shared<anet::rl::StateSweepProcessor>(
        env_spec.state_spec,
        0,  // x_index = x
        2,  // y_index = theta
        std::bind(
            &anet::rl::extractor::BoundaryMaskFromQdiffAuto,
            std::placeholders::_1,
            std::placeholders::_2,
            env_spec.action_spec.ActionCount(),
            0,
            1)
    );
    /// policy_netとtarget_netの差分を見る
    auto proc_theta_thetadot_pair_qdelta = std::make_shared<anet::rl::StateSweepProcessor>(
        env_spec.state_spec,
        0,  // x_index = x
        2,  // y_index = theta
        std::bind(
            &anet::rl::extractor::PairDiffExtractor,
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
    auto proc_theta_thetadot_pair_combo_qdeltaqmax = std::make_shared<anet::rl::StateSweepProcessor>(
        env_spec.state_spec,
        0,  // x_index = x
        2,  // y_index = theta
        std::bind(
            &anet::rl::extractor::QdeltaQmaxCombinedAuto,
            std::placeholders::_1,
            std::placeholders::_2,
            env_spec.action_spec.ActionCount()
        )
    );
    auto proc_theta_thetadot_pair_combo_qdelta_qdiff = std::make_shared<anet::rl::StateSweepProcessor>(
        env_spec.state_spec,
        0,  // x_index = x
        2,  // y_index = theta
        std::bind(
            &anet::rl::extractor::QdeltaQdiffCombinedAuto,
            std::placeholders::_1,
            std::placeholders::_2,
            env_spec.action_spec.ActionCount(),
            0,  // action_index_a (LEFT)
            1   // action_index_b (RIGHT)
        )
    );
    auto proc_theta_thetadot_pair_combo_qdelta_qdiffmasked = std::make_shared<anet::rl::StateSweepProcessor>(
        env_spec.state_spec,
        0,  // x_index = x
        2,  // y_index = theta
        std::bind(
            &anet::rl::extractor::BoundaryMaskedQdeltaAuto,
            std::placeholders::_1,
            std::placeholders::_2,
            env_spec.action_spec.ActionCount(),
            0,  // action_index_a (LEFT)
            1   // action_index_b (RIGHT)
        )
    );

    using StrMap = std::unordered_map<std::string, std::string>;

    anet::TensorFunction policy_forward = agent_->GetTensorFunction("policy_net.forward");
    anet::TensorFunction qpair_forward = agent_->GetTensorFunction("q_pair.forward");

    notifier_.Attach<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/05_shm_02_qmax", q_sweep_obs_config, proc_x_theta_qmax, policy_forward, proc_x_theta_qmax);
    notifier_.Attach<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/06_shm_23_qmax", q_sweep_obs_config, proc_theta_thetadot_qmax, policy_forward, proc_theta_thetadot_qmax);
    notifier_.Attach<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/08_shm_02_qdiff", q_sweep_obs_config, proc_theta_thetadot_qdiff, policy_forward, proc_theta_thetadot_qdiff);
    notifier_.Attach<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/09_shm_02_qdiff_mask", q_sweep_obs_config, proc_theta_thetadot_qdiff_mask, policy_forward, proc_theta_thetadot_qdiff_mask);
    notifier_.Attach<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/11_shm_02_qdelta", q_sweep_obs_config, proc_theta_thetadot_pair_qdelta, qpair_forward, proc_theta_thetadot_pair_qdelta);
    notifier_.Attach<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/12_shm_02_qdelta_qmax", q_sweep_obs_config, proc_theta_thetadot_pair_combo_qdeltaqmax, qpair_forward, proc_theta_thetadot_pair_combo_qdeltaqmax);
    notifier_.Attach<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/13_shm_02_qdelta_qdiff", q_sweep_obs_config, proc_theta_thetadot_pair_combo_qdelta_qdiff, qpair_forward, proc_theta_thetadot_pair_combo_qdelta_qdiff);
    notifier_.Attach<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/14_shm_02_qdelta_qdiff-masked", q_sweep_obs_config, proc_theta_thetadot_pair_combo_qdelta_qdiffmasked, qpair_forward, proc_theta_thetadot_pair_combo_qdelta_qdiffmasked,
        StrMap{
            { "46_agent_imgsc/02qdd_raw_qdelta_mean", "raw_qdelta_mean" },
            { "46_agent_imgsc/02qdd_raw_qdelta_max", "raw_qdelta_max" },
            { "46_agent_imgsc/02qdd_raw_boundary_mean", "raw_boundary_mean" },
            { "46_agent_imgsc/02qdd_boundary_area", "boundary_area"  },
            { "46_agent_imgsc/02qdd_combined_mean", "combined_mean" },
            { "46_agent_imgsc/02qdd_combined_max", "combined_max"  }
        });
}

CartPoleFrame::~CartPoleFrame() {
    wxLog::SetActiveTarget(NULL);
}

void CartPoleFrame::OnTimer(wxTimerEvent& event) {
    anet::ProfileRange r("CartPoleFrame::OnTimer");

    if (training_paused)
        return;  // ←停止中は一切処理しない

    // 再入防止
	this->timer.Stop();

    auto env_spec = env_->GetSpec();
    auto batch_env_spec = env_->GetBatchSpec();

    // --- 学習ステップを複数回回す ---
    float frame_total_reward = 0.0f;
    for (int i = 0; i < config_->step_per_frame; ++i) {
        anet::ProfileRange r("CartPoleFrame::OnTimer.step");

        auto train_step = step_counts_.train_step;

        if ((config_->train_exit_step > 0) && (train_step >= config_->train_exit_step)) {
            anet::MetricsLogger::Instance()->Flush();
            wxGetApp().Exit();
        }

        // Stateチェック
        wxLogDebug("CartPoleFrame::OnTimer() step=%llu state=%s", train_step, state_.ToString());
        ANET_ASSERT(env_spec.state_spec.MatchesShape(state_.obs));
        ANET_ASSERT(env_spec.state_spec.MatchesRange(state_.Flatten().obs));
        const int N = state_.obs.size(0);

        // 行動選択
        auto action_info = agent_->MakeAction(step_counts_, state_);
        wxLogDebug("CartPoleFrame::OnTimer() step=%llu action=%s", train_step, action_info.ToString());
        ANET_CHECK_DEVICE(action_info.action, device_agent_);
        ANET_CHECK_SHAPE(action_info.action, { N });

        // 環境ステップ実行
        anet::rl::BatchStepResult result = env_->Step(action_info.action);    // next_state, reward, done, truncated
        wxLogDebug("CartPoleFrame::OnTimer() step=%llu reward=%s", train_step, anet::ToString(result.reward));
        wxLogDebug("CartPoleFrame::OnTimer() step=%llu next_state=%s", train_step, result.next_state.ToString());
        ANET_CHECK_DEVICE(result.next_state.obs, torch::kCPU);
        ANET_CHECK_DEVICE(result.next_state.done, torch::kCPU);
        ANET_CHECK_DEVICE(result.next_state.truncated, torch::kCPU);
        ANET_CHECK_DEVICE(result.reward, torch::kCPU);
        ANET_CHECK_DEVICE(result.continue_state.obs, torch::kCPU);
        ANET_CHECK_DEVICE(result.continue_state.done, torch::kCPU);
        ANET_CHECK_DEVICE(result.continue_state.truncated, torch::kCPU);
        ANET_CHECK_SHAPE(result.next_state.obs, { N, ANET_SHAPE_ENDANY });
        ANET_CHECK_SHAPE(result.next_state.done, { N });
        ANET_CHECK_SHAPE(result.next_state.truncated, { N });
        ANET_CHECK_SHAPE(result.reward, { N });
        ANET_CHECK_SHAPE(result.continue_state.obs, { N, ANET_SHAPE_ENDANY });
        ANET_CHECK_SHAPE(result.continue_state.done, { N });
        ANET_CHECK_SHAPE(result.continue_state.truncated, { N });

        // カウント更新
        step_counts_.train_step++;
        step_counts_.exp_step += result.n_transitions;
        step_counts_.episode_count += result.n_done;

        // Agent更新
        anet::rl::BatchExperience exp({ state_, action_info, result.reward, result.next_state });
        auto update_result = agent_->UpdateFromBatch(step_counts_, exp);

        // カウント更新
        step_counts_.update_step++;
        step_counts_.learn_step += update_result->GetLearnStepDiff();

        // 更新後処理
        anet::rl::PostUpdateEvent  post_update_event { exp, step_counts_, agent_, update_result };
        notifier_.Notify(post_update_event);
        state_ = result.continue_state;
        float step_reward = result.reward.mean().item<float>();
        frame_total_reward += step_reward;

        // メトリクス算出（処理性能系）
        auto exp_step = step_counts_.exp_step;
		auto exp_step_delta = exp_step - last_exp_step_;
        std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
        auto msec_diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time_).count();
        if (msec_diff <= 0) msec_diff = 1;
        auto train_step_per_sec = 1.0f / static_cast<float>(msec_diff) * 1000.0f;
        auto exp_step_per_sec = static_cast<float>(exp_step_delta) / static_cast<float>(msec_diff) * 1000.0f;
        auto elapse_msec = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count());
        auto elapse_hour = elapse_msec / 1000.0f / 60.0f / 60.0f;


        // メトリクス出力（処理性能系）
        if (train_step != 0) { // 0ステップ目は誤差が大きいので
            train_step_per_sec_ema_.Update(train_step_per_sec);
            exp_step_per_sec_ema_.Update(exp_step_per_sec);

            if (train_step % config_->perf_log_interval == 0) {
                auto train_ema = train_step_per_sec_ema_.Value();
                auto exp_ema = exp_step_per_sec_ema_.Value();

                //anet::MetricsLogger::Instance()->LogScalar("90_perf/11_train_step_per_sec", train_step, train_step_per_sec);
                //anet::MetricsLogger::Instance()->LogScalar("90_perf/12_exp_step_per_sec", train_step, exp_step_per_sec);
                anet::MetricsLogger::Instance()->LogScalar("90_perf/21_train_step_per_sec_ema", train_step, train_ema);
                anet::MetricsLogger::Instance()->LogScalar("90_perf/22_exp_step_per_sec_ema", train_step, exp_ema);
                anet::MetricsLogger::Instance()->LogScalar("90_perf/82_exp_step", train_step, exp_step);
                anet::MetricsLogger::Instance()->LogScalar("90_perf/90_elapse_hour", train_step, elapse_hour);
            }
        }
        last_time_ = now;
		last_exp_step_ = exp_step;

        // 描画
        canvas->SetBatchExperience(exp);
        canvas->Refresh();

    }

    auto train_step = step_counts_.train_step;

    // AP終了判定
    if ((config_->train_exit_step > 0) && (train_step >= config_->train_exit_step)) {
        anet::MetricsLogger::Instance()->Flush();
        wxGetApp().Exit();
    }

	// Train一時停止判定
    if ((config_->train_pause_step > 0) && (train_step >= config_->train_pause_step) && !auto_pause_done_) {
        auto_pause_done_ = true;
        training_paused = true;
    }

    // タイマー再開
	this->timer.Start();
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
