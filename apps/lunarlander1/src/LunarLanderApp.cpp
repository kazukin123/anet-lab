// app.cpp
#include "LunarLanderApp.hpp"
#include <filesystem>
#include <wx/stdpaths.h>
#include <wx/cmdline.h>
#include <wx/filename.h>
#include "anet/profile.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/init.hpp"
#include "anet/log.hpp"
#include "anet/observers.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/rainbow_agent.hpp"
#include "LunarLanderFrame.hpp"
#include "UISnapshot.hpp"

namespace LOG = anet::log;

wxDEFINE_EVENT(wxEVT_TRAINER_EXIT, wxCommandEvent);
wxDEFINE_EVENT(wxEVT_APP_TRAINER_SHUTDOWN, wxThreadEvent);

struct LunarLanderApp::Config : public anet::Config
{
    bool train_auto_start = true;
    int train_pause_step = -1;
    int train_exit_step = -1; //110000;
    int train_timer_ms = 10;
    int eval_timer_ms = 10;
    int eval_step_per_frame = 1;
    int image_log_interval = 100;
    int image_log_interval_thm = 500;
    bool use_image_log = false;
    bool use_per_image_log = false;
    std::string run_name = "run_%t";

    LunarLanderApp::Config(const anet::ConfigData& config_data) : anet::Config(config_data, "LunarLanderApp")
    {
        ANET_READ_CONFIG(config_data, train_auto_start);
        ANET_READ_CONFIG(config_data, train_pause_step);
        ANET_READ_CONFIG(config_data, train_exit_step);
        ANET_READ_CONFIG(config_data, train_timer_ms);
        ANET_READ_CONFIG(config_data, eval_timer_ms);
        ANET_READ_CONFIG(config_data, eval_step_per_frame);
        ANET_READ_CONFIG(config_data, image_log_interval);
        ANET_READ_CONFIG(config_data, image_log_interval_thm);
        ANET_READ_CONFIG(config_data, use_image_log);
        ANET_READ_CONFIG(config_data, use_per_image_log);
        ANET_READ_CONFIG(config_data, run_name);
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
    return (GetProjectRootDir() / "config" / "LunarLanderRLGUI.txt").string();  // パスを結合
}

std::string GetRunsPath() {
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
    { wxCMD_LINE_USAGE_TEXT, NULL,    NULL,    "LunarLanderRLGUI.exe key1=value1 key2=value2" },     //  Additional usage text.
    { wxCMD_LINE_NONE } // 終了マーク
};

bool LunarLanderApp::OnInit()
{
    // ライブラリ初期化
    wxInitAllImageHandlers();
    anet::rl::InitRL();
    
    // メインスレッドでffmpeg実行する準備
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

    // LunarLanderAppConfig
    config_ = std::make_unique<LunarLanderApp::Config>(config_data);

    // MetricsLogger
    anet::MetricsLogger::Init(std::make_unique<anet::JsonlBackend>(), GetRunsPath(), config_->run_name);

    // LunarLanderAppConfigをダンプ
    anet::MetricsLogger::Instance()->LogJson("LunarLanderApp", config_->ToJson());
    anet::MetricsLogger::Instance()->Flush();

    // LunarLanderFrame
    frame_ = new LunarLanderFrame("LunarLander RL", config_->train_timer_ms, config_->eval_timer_ms, config_->eval_step_per_frame);
    frame_->Show();

    // Trainer生成
    trainer_ = std::make_unique<anet::rl::DefaultTrainer>(config_data);
    auto status = trainer_->Initialize(config_data);
    if (status != anet::rl::RunnerStatus::RUNNING) {
        LOG::error() << "Failed to initialize trainer.";
        return true;
    }

    // 評価Runner生成（描画用）
    auto eval_runner = trainer_->CreateEvalRunner();
    frame_->SetEvalRunner(std::move(eval_runner));

    // Trainer初期化
    InitTrainer();
    trainer_->GetNotifier()->LogObservers();
    if (config_->use_image_log) InitImageLogObservers();
    if (config_->use_per_image_log) InitPERImageLogObservers(config_data);

    // Trainerスレッド生成
    trainer_thread_ = std::make_unique<anet::rl::RunnerThread>(
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
    if (!config_->train_auto_start)
        trainer_thread_->Pause();
    trainer_thread_->Start();

    return true;
}

void LunarLanderApp::ToggleTraining()
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

void LunarLanderApp::StopTraining()
{
    trainer_thread_->Stop();
}

int LunarLanderApp::OnExit()
{
    trainer_thread_->Stop();
    anet::MetricsLogger::Reset();
    return 0;
}

UISnapshot LunarLanderApp::CreateSnapshot(anet::rl::TrainEvent event)
{
    anet::ProfileRange r1("CreateSnapshot");

    ANET_LOG_DEBUG("batch_step_result=" << event.batch_step_result->ToString());

    const int ENV_INDEX = 0;
    
    // RL由来情報
    auto train_step = event.counts.train_step;
    anet::rl::SingleState state = {
        event.batch_step_result->next_state.obs[ENV_INDEX],
        event.batch_step_result->next_state.done[ENV_INDEX].item<bool>(),
        event.batch_step_result->next_state.truncated[ENV_INDEX].item<bool>(),
        event.batch_step_result->next_state.episode_start[ENV_INDEX].item<bool>(),
    };
    auto action = event.batch_exp.action.GetAction(torch::kCPU)[ENV_INDEX].item<int64_t>();
    auto reward = event.batch_exp.reward[ENV_INDEX].item<float>();

    // aux情報
    auto auxs = event.batch_step_result->GetAuxDataList(ENV_INDEX);
    ANET_ASSERT(auxs.size() > 0);
    auto aux = auxs[0];

    // Snapshotを作る
    UISnapshot snapshot{ train_step, state, action, reward, aux };

    return snapshot;
}

void LunarLanderApp::InitTrainer()
{
    // TrainObserver
    trainer_->GetNotifier()->Attach<anet::rl::FunctionTrainObserver>(
        [this](const anet::rl::TrainEvent& event)
        {
            anet::ProfileRange r1("FunctionTrainObserver");
            auto train_step = event.counts.train_step;

            // Trainスナップショット取得
            if (snapshot_store_.IsDataRequest() || (train_step % 2000 == 0)) {
                ANET_LOG_DEBUG("UI data. train_step=" << train_step);

                // Plotデータ追加
                auto train_reward_ema = event.runner.GetScalar(anet::rl::Runner::TARGET_EVAL_REWARD);
                ANET_ASSERT(train_reward_ema.has_value());
                frame_->AddPlotData(*train_reward_ema);

                // UIスナップショットを生成＆更新
                auto snapshot = CreateSnapshot(event);
                snapshot_store_.Update(snapshot);
            }

            // Trainログ
            if (event.counts.train_step % 100 == 0) {
                auto train_reward_ema = event.runner.GetScalar(anet::rl::Runner::TRAIN_REWARD_EMA);
                ANET_ASSERT(train_reward_ema.has_value());
                LOG::info() << "train_step=" << train_step << " train_mean_reward=" << *train_reward_ema;
            }

        }, "LunarLanderApp");

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

        }, "LunarLanderApp");
}

void LunarLanderApp::InitImageLogObservers()
{
    auto notifier = trainer_->GetNotifier();
    auto env_spec = trainer_->GetBatchEnv()->GetSpec();
    auto agent = trainer_->GetAgent();

    // flags
    auto flags =
        //anet::HeatMapFlags::HM_LogScaleValue | 
        anet::HeatMapFlags::HM_AutoNormValue
        | anet::HeatMapFlags::HM_AutoScaleAxis	// ★
        //| anet::HeatMapFlags::HM_LogScaleAxis
        //| anet::HeatMapFlags::HM_MeanMode;
        | anet::HeatMapFlags::HM_SumMode;
        //| anet::HeatMapFlags::HM_ShowZeroLine;

    // ---- Visit ----

    anet::rl::HeatMapObserverConfig visit_heat_obs_config {
        512,    // width
        512,    // height
        config_->image_log_interval,    // log_interval 
        30000,  // max_points
        flags   // flags
        -1,     // image_width
        -1,     // image_height
    };
    auto visit_x_probe = std::make_shared<anet::rl::BatchExperienceStateProbe>(0, &env_spec.state_spec, true);
    auto visit_y_probe = std::make_shared<anet::rl::BatchExperienceStateProbe>(1, &env_spec.state_spec, true);
    //auto visit_reward_probe = std::make_shared<anet::rl::BatchExperienceRewardProbe>(nullptr);
    auto visit_q_probe = std::make_shared<anet::rl::BatchActionInfoToVectorProbe>("max_q");

    //notifier->Attach<anet::rl::HeatMapVectorObserver>(
    //    "43_agent_img/02_hm_visit_01_reward", visit_heat_obs_config, visit_x_probe, visit_y_probe, visit_reward_probe);
    //notifier->Attach<anet::rl::HeatMapVectorObserver>(
    //    "43_agent_img/03_hm_visit_01_maxq", visit_heat_obs_config, visit_x_probe, visit_y_probe, visit_q_probe);

    // ---- ReplayBuffer ----

    auto rep_x_probe = std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 0, &env_spec.state_spec);
    auto rep_y_probe = std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 1, &env_spec.state_spec);
    auto rep_theta_probe = std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 4, &env_spec.state_spec);
    auto rep_reward_probe = std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::REWARD, -1, &env_spec.state_spec);

    anet::rl::HeatMapObserverConfig replay_heat_obs_config {
        512,    // width
        512,    // height
        config_->image_log_interval,    // log_interval 
        60000,  // max_points
        flags   // flags
        -1,     // image_width
        -1,     // image_height
    };

    //auto auto_scale_mode = anet::rl::AgentTensorVectorProbe::AutoScaleMode::GLOBAL;   // サンプル値でmin/max調整
    auto auto_scale_mode = anet::rl::AgentTensorVectorProbe::AutoScaleMode::DISABLE;    // EnvSpecで固定
    std::vector<std::shared_ptr<anet::rl::VectorProbe>> probes_3axis = {
        std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 0, &env_spec.state_spec, nullptr, auto_scale_mode),  // X
        std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 1, &env_spec.state_spec, nullptr, auto_scale_mode),  // Y
        std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 4, &env_spec.state_spec, nullptr, auto_scale_mode),  // theta
    };

    //notifier->Attach<anet::rl::HeatMapVectorObserver>(
    //    "43_agent_img/12_hm_rep_01", replay_heat_obs_config, rep_x_probe, rep_y_probe, rep_reward_probe);
    //notifier->Attach<anet::rl::HeatMapVectorObserver>(
    //    "43_agent_img/13_hm_rep_04", replay_heat_obs_config, rep_x_probe, rep_theta_probe, rep_reward_probe);
    //notifier->Attach<anet::rl::MultiPairHeatMapObserver>(
    //    "43_agent_img/21_hm_rep_multi3",
    //    replay_heat_obs_config,
    //    probes_3axis,
    //    rep_reward_probe);

    // ---- SweepedHeatMap ----

    auto v_extractor =
        [](const torch::Tensor& t, const std::unordered_set<std::string>& req)
        {
            // あまり意味ないけどサンプルとして残しておく
            auto ret = anet::rl::extractor::MeanIdxExtractor(t, req, 0);  // V : index=0
            return ret;
        };

    anet::rl::SweepedHeatMapObserverConfig q_sweep_obs_config{
        config_->image_log_interval,    // log_interval
        flags | anet::HeatMapFlags::HM_AutoScaleAxis,  // flags
        128,    // grid_width
        128,    // grid_height
        -1,     // image_width
        -1,     // image_height
    };
    auto proc_x_y_qmax = std::make_shared<anet::rl::StateSweepProcessor>(
        env_spec.state_spec,
        0,  // x_index = x
        1,   // y_index = y
        anet::rl::extractor::MaxExtractor
    );
    auto proc_x_y_qmean = std::make_shared<anet::rl::StateSweepProcessor>(
        env_spec.state_spec,
        0,  // x_index = x
        1,   // y_index = y
        anet::rl::extractor::MeanExtractor
    );
    auto proc_x_y_v = std::make_shared<anet::rl::StateSweepProcessor>(
        env_spec.state_spec,
        0,  // x_index = x
        1,   // y_index = y
        v_extractor
    );


    using StrMap = std::unordered_map<std::string, std::string>;

    std::optional<anet::TensorFunction> policy_forward = agent->GetTensorFunction("policy-net.forward");
    std::optional<anet::TensorFunction> v_policy_forward = agent->GetTensorFunction("policy-net.forward.v");
    //std::optional<anet::TensorFunction> qpair_forward = agent->GetTensorFunction("q_pair.forward");
    ANET_ASSERT(policy_forward.has_value());


    notifier->Attach<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/05_shm_01_qmax", q_sweep_obs_config, proc_x_y_qmax, *policy_forward, proc_x_y_qmax);
    notifier->Attach<anet::rl::SweepedHeatMapObserver>(
        "45_agent_img/06_shm_02_qmean", q_sweep_obs_config, proc_x_y_qmean, *policy_forward, proc_x_y_qmean);
    if (v_policy_forward.has_value())
        notifier->Attach<anet::rl::SweepedHeatMapObserver>(
            "45_agent_img/07_shm_03_v", q_sweep_obs_config, proc_x_y_v, *v_policy_forward, proc_x_y_v);
}

void LunarLanderApp::InitPERImageLogObservers(const anet::ConfigData& config_data)
{
    anet::rl::dqn::RainbowAgentConfig agent_config(config_data);
    if (!agent_config.learner.use_per) {
        LOG::warn() << "PER agent config disabled. Skipping PER ImageLog observer.";
        return;
    }

    auto notifier = trainer_->GetNotifier();
    auto env_spec = trainer_->GetBatchEnv()->GetSpec();
    auto agent = trainer_->GetAgent();

    // flags
    auto flags =
        //anet::HeatMapFlags::HM_LogScaleValue | 
        anet::HeatMapFlags::HM_AutoNormValue
        | anet::HeatMapFlags::HM_AutoScaleAxis
        //| anet::HeatMapFlags::HM_LogScaleAxis
        | anet::HeatMapFlags::HM_SumMode; // | anet::HeatMapFlags::HM_ShowZeroLine;

    // ---- ReplayBuffer ----

    auto rep_x_probe = std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 0, &env_spec.state_spec);
    auto rep_y_probe = std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 1, &env_spec.state_spec);
    auto rep_prio_probe = std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::PER_DIST, -1);

    anet::rl::HeatMapObserverConfig replay_heat_obs_config {
        512,    // width
        512,    // height
        config_->image_log_interval,    // log_interval 
        100000,  // max_points
        flags   // flags
        -1,     // image_width
        -1,     // image_height
    };

    auto auto_scale_mode = anet::rl::AgentTensorVectorProbe::AutoScaleMode::DISABLE;    // EnvSpecで固定

    notifier->Attach<anet::rl::HeatMapVectorObserver>(
        "43_agent_img/52_per_hm_prio_01", replay_heat_obs_config, rep_x_probe, rep_y_probe, rep_prio_probe);


    anet::rl::TimeHistogramObserverConfig prio_hist_obs_config {
        256,    // x bins
        1920,   // y max_frames
        512,    // image_height
        1920,   // image_width
        anet::TimeFrameMode::Scale,             // mode
        flags | anet::HeatMapFlags::HM_FlipY | anet::HeatMapFlags::HM_LogScaleAxis,   // flags
        config_->image_log_interval_thm,    // log_interval
        20,     // frame_interval
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        1.0f// alpha = 0.05f
    };
    notifier->Attach<anet::rl::TimeHistogramObserver>(
        "44_agent_img/52_per_thg_prio", prio_hist_obs_config, rep_prio_probe);

}


wxIMPLEMENT_APP(LunarLanderApp);

