#include "CartPoleFrame.hpp"
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <torch/torch.h>
#include <wx/stattext.h>
#include <wx/sizer.h>
#include "anet/tensor_utils.hpp"
#include "anet/profile.hpp"
#include "anet/observers.hpp"
#include "anet/replay_buffer.hpp"
#include "CartPoleCanvas.hpp"
#include "app.hpp"

wxBEGIN_EVENT_TABLE(CartPoleFrame, wxFrame)
EVT_TIMER(wxID_ANY, CartPoleFrame::OnTimer)
EVT_LEFT_DOWN(CartPoleFrame::OnMouseClick)
wxEND_EVENT_TABLE()

struct CartPoleFrame::Config : public anet::Config
{
    int timer_ms = 20;
    int step_per_frame = 10;
    int train_pause_step = -1;
    int train_exit_step = -1; //110000;
    bool enable_image_log = true;

    CartPoleFrame::Config(const anet::ConfigData& config_data = anet::EmptyConfigData)
        : anet::Config(config_data, "CartPoleFrame")
    {
        ANET_READ_CONFIG(config_data, timer_ms);
        ANET_READ_CONFIG(config_data, step_per_frame);
        ANET_READ_CONFIG(config_data, train_pause_step);
        ANET_READ_CONFIG(config_data, train_exit_step);
        ANET_READ_CONFIG(config_data, enable_image_log);
    }
};

CartPoleFrame::CartPoleFrame(const wxString& title)
    : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxSize(800, 800)),
    timer(this, wxID_ANY)
{
	// --- 設定読み込み ---
    auto config_data = wxGetApp().GetConfig();
    config_ = std::make_unique<Config>(config_data);

    // GUIレイアウト ---
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

    // ログ出力先をこのクラスに設定
    wxLog::SetActiveTarget(this);

    wxLogInfo("CartPoleRLGUI started.");

    // Trainer生成
    trainer_ = std::make_unique<anet::rl::DefaultTrainer>(config_data);
    InitTrainer();
    trainer_->GetNotifier()->LogObservers();
    InitImageLogObservers();

    // タイマー開始
    Bind(wxEVT_TIMER, &CartPoleFrame::OnTimer, this);
    timer.Start(config_->timer_ms);  // 学習＆描画更新
}

CartPoleFrame::~CartPoleFrame()
{
    wxLog::SetActiveTarget(NULL);
}

void CartPoleFrame::InitTrainer()
{
    // log
    trainer_->GetNotifier()->Attach<anet::rl::FunctionTrainObserver>(
        [this](const anet::rl::TrainEvent& event)
        {
            auto train_step = event.counts.train_step;

            canvas->SetBatchExperience(event.batch_exp);
            canvas->Refresh();

            if (event.counts.train_step % 10 == 0) {
                auto train_reward_ema = event.trainer.GetScalar(anet::rl::Trainer::TRAIN_REWARD_EMA);
                ANET_ASSERT(train_reward_ema.has_value());
                this->plotPanel->AddReward(*train_reward_ema);
            }

            if (event.counts.train_step % 100 == 0) {
                auto train_reward_ema = event.trainer.GetScalar(anet::rl::Trainer::TRAIN_REWARD_EMA);
                ANET_ASSERT(train_reward_ema.has_value());
                wxLogInfo("train_step=%llu train_mean_reward=%f", train_step, *train_reward_ema);
            }

            return false;
        }, "CartPoleFrame"
    );
    trainer_->GetNotifier()->Attach<anet::rl::FunctionLearnObserver>(
        //[&, plot_reward](int step,
        [this](const anet::rl::LearnEvent& event)
        {
            auto train_step = event.counts.train_step;
            auto learn_step = event.counts.learn_step;

            //float step_reward = event.batch_exp.reward.mean().item<float>();
            //train_reward_ema_.Update(step_reward);
            //anet::MetricsLogger::Instance()->LogScalar(
            //    "10_train/10_total_reward", learn_step, train_reward_ema_.Value());

            //*plot_reward += step_reward;
            //if (event.counts.train_step % 10 == 0) {
            //    this->plotPanel->AddReward(*plot_reward);
            //    *plot_reward = 0;
            //}

            if (event.counts.learn_step % 100 == 0) {
                wxLogInfo("train_step=%llu learn_step=%llu",
                    train_step, learn_step);
            }
            return false;
        }, "CartPoleFrame"
    );
}

void CartPoleFrame::OnTimer(wxTimerEvent& event)
{
    anet::ProfileRange r("CartPoleFrame::OnTimer");

    // 停止中は一切処理しない
    if (training_paused) return;

    // 再入防止
    timer.Stop();

    // RL frame
    trainer_->DoUpdateFrame(config_->step_per_frame,
        [this]()
        {
            // AP終了判定
            if ((config_->train_exit_step > 0) && (trainer_->GetCounts().train_step >= config_->train_exit_step)) {
                anet::MetricsLogger::Instance()->Flush();
                //wxGetApp().Exit();
				return true;    // Train終了
            }
            return false;
        }
    );

    // Train一時停止
    if ((config_->train_pause_step > 0) && (trainer_->GetCounts().train_step >= config_->train_pause_step) && !auto_pause_done_) {
        auto_pause_done_ = true;
        training_paused = true;
    }

    Refresh();

    // タイマー再開
    this->timer.Start();
}

void CartPoleFrame::ToggleTraining()
{
    training_paused = !training_paused;
    wxLogMessage(training_paused ? "Training paused" : "Training resumed");
}

void CartPoleFrame::DoLogText(const wxString& msg)
{
    this->logBox->AppendText(msg);
    this->logBox->AppendText("\n");
}

void CartPoleFrame::OnMouseClick(wxMouseEvent& event)
{
    ToggleTraining();
}

void CartPoleFrame::InitImageLogObservers()
{
    // 有効設定チェック
    if (!config_->enable_image_log)
        return;

	auto env_spec = trainer_->GetBatchEnv()->GetSpec();
    auto notifier = trainer_->GetNotifier();
    auto agent = trainer_->GetAgent();

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

    notifier->Attach<anet::rl::HeatMapVectorObserver>(
        "43_agent_img/12_hm_rep_02", replay_heat_obs_config, rep_x_probe, rep_theta_probe, rep_reward_probe);
    notifier->Attach<anet::rl::HeatMapVectorObserver>(
        "43_agent_img/13_hm_rep_23", replay_heat_obs_config, rep_theta_probe, rep_theta_dot_probe, rep_reward_probe);
    notifier->Attach<anet::rl::MultiPairHeatMapObserver>(
        "43_agent_img/21_hm_rep_multi3",
        replay_heat_obs_config,
        probes_3axis,
        rep_reward_probe);
    notifier->Attach<anet::rl::MultiPairHeatMapObserver>(
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
    notifier->Attach<anet::rl::TimeHistogramObserver>(
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

    anet::TensorFunction policy_forward = agent->GetTensorFunction("policy_net.forward");
    anet::TensorFunction qpair_forward = agent->GetTensorFunction("q_pair.forward");

    notifier->Attach<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/05_shm_02_qmax", q_sweep_obs_config, proc_x_theta_qmax, policy_forward, proc_x_theta_qmax);
    notifier->Attach<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/06_shm_23_qmax", q_sweep_obs_config, proc_theta_thetadot_qmax, policy_forward, proc_theta_thetadot_qmax);
    notifier->Attach<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/08_shm_02_qdiff", q_sweep_obs_config, proc_theta_thetadot_qdiff, policy_forward, proc_theta_thetadot_qdiff);
    notifier->Attach<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/09_shm_02_qdiff_mask", q_sweep_obs_config, proc_theta_thetadot_qdiff_mask, policy_forward, proc_theta_thetadot_qdiff_mask);
    notifier->Attach<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/11_shm_02_qdelta", q_sweep_obs_config, proc_theta_thetadot_pair_qdelta, qpair_forward, proc_theta_thetadot_pair_qdelta);
    notifier->Attach<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/12_shm_02_qdelta_qmax", q_sweep_obs_config, proc_theta_thetadot_pair_combo_qdeltaqmax, qpair_forward, proc_theta_thetadot_pair_combo_qdeltaqmax);
    notifier->Attach<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/13_shm_02_qdelta_qdiff", q_sweep_obs_config, proc_theta_thetadot_pair_combo_qdelta_qdiff, qpair_forward, proc_theta_thetadot_pair_combo_qdelta_qdiff);
    notifier->Attach<anet::rl::SweepedHeatMapObserver>(
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
