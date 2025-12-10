// app.cpp
#include "app.hpp"
#include <filesystem>
#include <wx/stdpaths.h>
#include <wx/cmdline.h>
#include <wx/filename.h>
#include "anet/metrics_logger.hpp"
#include "anet/init.hpp"
#include "anet/log.hpp"
#include "anet/observers.hpp"
#include "anet/replay_buffer.hpp"
#include "CartPoleFrame.hpp"
#include "UISnapshot.hpp"

namespace LOG = anet::log;

wxDEFINE_EVENT(wxEVT_TRAINER_EXIT, wxCommandEvent);
wxDEFINE_EVENT(wxEVT_APP_TRAINER_SHUTDOWN, wxThreadEvent);

struct CartPoleApp::Config : public anet::Config
{
    int train_pause_step = -1;
    int train_exit_step = -1; //110000;
    bool enable_image_log = true;
    int timer_ms = 10;

    CartPoleApp::Config(const anet::ConfigData& config_data) : anet::Config(config_data, "CartPoleApp")
    {
        ANET_READ_CONFIG(config_data, train_pause_step);
        ANET_READ_CONFIG(config_data, train_exit_step);
        ANET_READ_CONFIG(config_data, enable_image_log);
        ANET_READ_CONFIG(config_data, timer_ms);
    }
};

wxString GetExeDir() {
    wxStandardPaths& sp = wxStandardPaths::Get();
    wxString exe_path = sp.GetExecutablePath();      // フルパス (C:\proj\bin\myapp.exe 等)
    wxFileName fn(exe_path);
    return fn.GetPath();                            // ディレクトリ部分を返す
}

std::filesystem::path GetProjectRootDir()
{
    std::filesystem::path exePath = GetExeDir().ToStdString();  // 既存の GetExeDir を利用
    return exePath.parent_path().parent_path();    // exe の親ディレクトリを返す
}

std::string GetConfigFilePath() {
    return (GetProjectRootDir() / "config" / "CartPoleRLGUI.txt").string();  // パスを結合
}

std::string GetLogsPath() {
    return (GetProjectRootDir() / "runs").string();
}

static wxCmdLineEntryDesc desc[] = {
    // kind,              short-name, long-name, usage,      type,                  flags
    //{ wxCMD_LINE_SWITCH, "v",         "verbose", "エラー表示を饒舌に" }, // wxCMD_LINE_SWITCH:A boolean argument of the program;    e.g. -v to enable verbose mode.
    //{ wxCMD_LINE_OPTION, "f",         "file",    "設定ファイルのパス" }, // wxCMD_LINE_OPTION:An argument with an associated value; e.g. -o filename

    {
        wxCMD_LINE_PARAM,              // 種別：位置パラメータ
        nullptr,                       // 短いオプション名なし
        nullptr,                       // 長いオプション名なし
        "key=value pairs",             // 説明文
        wxCMD_LINE_VAL_STRING,         // 文字列として受け取る
        wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE      // 複数 OK
    },
    //{ wxCMD_LINE_PARAM,  NULL,        NULL,  "引数",     wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE },  // A parameter: a required program argument.
    { wxCMD_LINE_USAGE_TEXT, NULL,    NULL,    "CartPoleRLGUI.exe key1=value1 key2=value2" },     //  Additional usage text.
    { wxCMD_LINE_NONE } // 終了マーク
};

bool CartPoleApp::OnInit()
{
    // ライブラリ初期化
    wxInitAllImageHandlers();
    anet::rl::InitRL();
    
    // メインスレッドでのffmpeg実行準備
    Bind(wxEVT_APP_EXECUTE_START, [&](wxThreadEvent& event) {
        anet::ExecuteStarter* executer = event.GetPayload<anet::ExecuteStarter*>();
        ANET_LOG_DEBUG("Executing command on main thread. command=" << executer->GetCommand());
        executer->OnMainStart();   // wxExecute 内部呼び出し
        });

    // ConfigManager
    wxCmdLineParser cmdline_(desc, argc, (wchar_t**)argv);
    if (cmdline_.Parse(true))
        return false;
    config_mgr_ = std::make_unique<anet::ConfigManager>(GetConfigFilePath(), &cmdline_);
    auto config_data = config_mgr_->GetConfigData();

    // CartPoleAppConfig
    config_ = std::make_unique<CartPoleApp::Config>(config_data);

    // MetricsLogger
    anet::MetricsLogger::Init(std::make_unique<anet::JsonlBackend>(), GetLogsPath());

    // CartPoleFrame
    frame_ = new CartPoleFrame("CartPole RL", config_->timer_ms);
    frame_->Show(true);

    // Trainer
    trainer_ = std::make_unique<anet::rl::DefaultTrainer>(config_data);
    auto status = trainer_->Initialize(config_data);
    if (status != anet::rl::TrainerStatus::RUNNING) {
        LOG::error() << "Failed to initialize trainer.";
        return true;
    }

    // Trainer初期化
    InitTrainer();
    trainer_->GetNotifier()->LogObservers();
    InitImageLogObservers();

    // Trainerスレッド生成
    trainer_thread_ = std::make_unique<anet::rl::AsyncTrainerRunner>(
        trainer_,
        [this](const anet::rl::StepCounts& counts)   // pre_train_step_function
        {
            auto train_step = counts.train_step;

            // Train終了判定
            if ((config_->train_exit_step > 0) && (trainer_->GetCounts().train_step >= config_->train_exit_step)) {
                wxQueueEvent(wxTheApp->GetTopWindow(), new wxCommandEvent(wxEVT_TRAINER_EXIT)); // Mainスレッドに終了要求
                //LOG::info() << "Auto exit.";
                return anet::rl::ControlSignal::STOP;    // Train終了
            }

            // 自動Pause
            if ((config_->train_pause_step > 0) && (train_step >= config_->train_pause_step) && !auto_pause_done_) {
                auto_pause_done_ = true;    // 一回だけ自動
                trainer_thread_->Pause();
                LOG::info() << "Auto pause.";
                return anet::rl::ControlSignal::BREAK;
            }

            return anet::rl::ControlSignal::CONTINUE;
        });

    // Train開始！
    trainer_thread_->Start();

    return true;
}

void CartPoleApp::ToggleTraining()
{
    bool paused = trainer_thread_->IsPaused();

    if (paused) {
        trainer_thread_->Resume();
        LOG::info() << "Training resumed";
    } else {
        trainer_thread_->Pause();
        LOG::info() << "Training paused";
    }
}

void CartPoleApp::StopTraining()
{
    trainer_thread_->Stop();
}

int CartPoleApp::OnExit()
{
    trainer_thread_->Stop();
    anet::MetricsLogger::Reset();
    return 0;
}

void CartPoleApp::InitTrainer()
{
    // TrainObserver
    trainer_->GetNotifier()->Attach<anet::rl::FunctionTrainObserver>(
        [this](const anet::rl::TrainEvent& event)
        {
            auto train_step = event.counts.train_step;

            // Trainスナップショット取得
            if (event.counts.train_step % 10 == 0) {
                // 平均報酬をPlotデータ追加
                auto train_reward_ema = event.trainer.GetScalar(anet::rl::Trainer::TRAIN_REWARD_EMA);
                ANET_ASSERT(train_reward_ema.has_value());
                frame_->AddPlotData(*train_reward_ema);

                // Train状況のSnapshotを更新
                UISnapshot snapshot {
                    event.counts,
                    event.batch_exp,
                    //*train_reward_ema,
                    //event.agent
                };
                snapshot_store_.Update(snapshot);
            }

            // Trainログ
            if (event.counts.train_step % 100 == 0) {
                auto train_reward_ema = event.trainer.GetScalar(anet::rl::Trainer::TRAIN_REWARD_EMA);
                ANET_ASSERT(train_reward_ema.has_value());
                LOG::info() << "train_step=" << train_step << " train_mean_reward=" << *train_reward_ema;
            }

        }, "CartPoleApp");

    // LearnObserver
    trainer_->GetNotifier()->Attach<anet::rl::FunctionLearnObserver>(
        [](const anet::rl::LearnEvent& event)
        {
            auto train_step = event.counts.train_step;
            auto learn_step = event.counts.learn_step;

            // Learnログ
            if (event.counts.learn_step % 100 == 0) {
                LOG::info() << "train_step=" << train_step << " learn_step=" << learn_step;
            }

        }, "CartPoleApp");
}

void CartPoleApp::InitImageLogObservers()
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

wxIMPLEMENT_APP(CartPoleApp);

