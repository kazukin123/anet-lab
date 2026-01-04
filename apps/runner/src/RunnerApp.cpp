// RunnerApp.cpp
#include "RunnerApp.hpp"
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
#include "anet/scaler.hpp"
#include "anet/exception.hpp"
#include "anet/profile.hpp"
#include "anet/env/LunarLander.hpp"
#include "anet/env/CartPole.hpp"
#include "ErrorDialog.hpp"
#include "RunnerFrame.hpp"
#include "TrainPanel.hpp"
#include "EvalPanel.hpp"

namespace LOG = anet::log;


wxDEFINE_EVENT(wxEVT_TRAINER_EXIT, wxCommandEvent);
wxDEFINE_EVENT(wxEVT_APP_TRAINER_SHUTDOWN, wxThreadEvent);


struct RunnerApp::Config : public anet::Config {
    std::string run_name = "run_{%t}";
    bool train_auto_start = true;
    int train_pause_step = -1;
    int train_exit_step = -1; //110000;
    int exp_pause_step = -1;
    int exp_exit_step = -1;

    TrainPanelConfig train_panel;
    EvalPanelConfig eval_panel;

    RunnerApp::Config(const anet::ConfigData& config_data) : anet::Config(config_data, "app")
    {
        ANET_READ_CONFIG(config_data, run_name);
        ANET_READ_CONFIG(config_data, train_auto_start);
        ANET_READ_CONFIG(config_data, train_pause_step);
        ANET_READ_CONFIG(config_data, train_exit_step);
        ANET_READ_CONFIG(config_data, exp_pause_step);
        ANET_READ_CONFIG(config_data, exp_exit_step);

        ANET_READ_CONFIG(config_data, train_panel.fps);
        ANET_READ_CONFIG(config_data, eval_panel.fps);
        ANET_READ_CONFIG(config_data, eval_panel.step_per_frame);
    }
};

wxString GetExeDir()
{
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

std::string GetConfigFilePath()
{
    return (GetProjectRootDir() / "config" / "_main.txt").string();  // パスを結合
}

std::string GetRunsPath()
{
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
    //{ wxCMD_LINE_USAGE_TEXT, NULL,    NULL,    "AnetRLRunner.exe key1=value1 key2=value2" },     //  Additional usage text.
    { wxCMD_LINE_NONE } // 終了マーク
};

bool RunnerApp::OnInit()
{
    anet::ProfileThreadName th("MainThread");

    // ライブラリ初期化
    wxInitAllImageHandlers();
    anet::rl::InitRL();
    anet::rl::env::InitLunarLander();
    anet::rl::env::InitCartPole();

    // 全体ログレベル設定
#if ANET_ENABLE_DEBUGINFO
    wxLog::SetLogLevel(wxLOG_Debug);
#endif

    // メインスレッドでffmpeg実行する準備
    Bind(wxEVT_APP_EXECUTE_START, [&](wxThreadEvent& event) {
        anet::ExecuteStarter* executer = event.GetPayload<anet::ExecuteStarter*>();
        ANET_LOG_DEBUG("Executing command on main thread. command=" << executer->GetCommand());
        executer->OnMainStart();   // wxExecute 内部呼び出し
        });

    // コマンドライン引数解析
    wxCmdLineParser cmdline_(desc, argc, (wchar_t**)argv);
    if (cmdline_.Parse(true)) {
        return false;
    }

    // ConfigManager
    config_mgr_ = std::make_unique<anet::ConfigManager>(GetConfigFilePath(), &cmdline_);
    auto config_data = config_mgr_->GetConfigData();

    // RunnerApp設定生成
    config_ = std::make_unique<RunnerApp::Config>(config_data);

    // MetricsLogger
    anet::MetricsLogger::Init(std::make_unique<anet::JsonlBackend>(), GetRunsPath(), config_->run_name);

    // RunnerFrame生成＆表示
    wxString frame_title = anet::MetricsLogger::Instance()->GetRunName() + " - ANET RL Runner";
    frame_ = new RunnerFrame(frame_title, config_->train_panel, config_->eval_panel);
    frame_->Show();

    // RunnerAppConfigをダンプ
    LOG::info() << "RunnerApp config=" << *config_;
    anet::MetricsLogger::Instance()->Log(*config_);

    // Trainer生成
    trainer_ = std::make_unique<anet::rl::DefaultTrainer>(config_data);
    auto status = trainer_->Initialize(config_data);
    if (status != anet::rl::RunnerStatus::RUNNING) {
        LOG::error() << "Failed to initialize trainer.";
        return true;
    }

    // DefaultViewFactory
    view_factory_ = std::make_unique<anet::rl::gui::DefaultViewFactory>(config_data);
    
    // Trainer初期化
    InitTrainer();

    // ImageProviderManager
    auto env_spec = trainer_->GetBatchEnv()->GetSpec();
    auto notifier = trainer_->GetNotifier();
    auto agent = trainer_->GetAgent();
    img_prov_mgr_ = std::make_unique<anet::rl::ImageProviderManager>(env_spec, notifier);
    img_prov_mgr_->CreateAndRegister(config_data, agent);

    // ManualObserver
    //if (config_->use_image_log) InitImageLogObservers();
    //if (config_->use_per_image_log) InitPERImageLogObservers(config_data);

    // Trainer初期化後のUI初期化
    frame_->Initialize(trainer_);

    // ログ
    trainer_->GetNotifier()->LogObservers();

    // Trainerスレッド生成
    trainer_thread_ = std::make_unique<anet::rl::RunnerThread>(
        trainer_,
        [this](const anet::rl::StepCounts& counts)   // pre_train_step_func
        {
            const auto& train_step = counts.train_step;
            const auto& exp_step = counts.exp_step;

            // Train終了判定
            if ((config_->train_exit_step > 0) && (train_step >= config_->train_exit_step)) {
                wxQueueEvent(wxTheApp->GetTopWindow(), new wxCommandEvent(wxEVT_TRAINER_EXIT)); // Mainスレッドに終了要求
                return anet::rl::ControlSignal::STOP;    // Train終了
            }
            if ((config_->exp_exit_step > 0) && (exp_step >= config_->exp_exit_step)) {
                wxQueueEvent(wxTheApp->GetTopWindow(), new wxCommandEvent(wxEVT_TRAINER_EXIT)); // Mainスレッドに終了要求
                return anet::rl::ControlSignal::STOP;    // Train終了
            }

            // 自動Pause
            if ((config_->train_pause_step > 0) && (train_step >= config_->train_pause_step) && !auto_pause_done_) {
                auto_pause_done_ = true;    // 一回だけ自動
                trainer_thread_->Pause();
                LOG::info() << "Auto pause.";
                return anet::rl::ControlSignal::BREAK;
            }
            if ((config_->exp_pause_step > 0) && (exp_step >= config_->exp_pause_step) && !auto_pause_done_) {
                auto_pause_done_ = true;    // 一回だけ自動
                trainer_thread_->Pause();
                LOG::info() << "Auto pause.";
                return anet::rl::ControlSignal::BREAK;
            }

            return anet::rl::ControlSignal::CONTINUE;
        },
        nullptr,    // post_train_step_func
        [this]() {  // exception_func
            wxGetApp().StoreCurrentException();
            wxGetApp().CallAfter([] { wxGetApp().RethrowStoredException(); });
        }
    );

    // Train開始！
    if (!config_->train_auto_start)
        trainer_thread_->Pause();
    trainer_thread_->Start();

    return true;
}

void RunnerApp::ToggleTraining()
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

void RunnerApp::StopTraining()
{
    trainer_thread_->Stop();
}

int RunnerApp::OnExit()
{
    trainer_thread_->Stop();
    anet::MetricsLogger::Reset();
    return 0;
}

void RunnerApp::showFatalError()
{
    try {
        // 現在発生している例外を再スローして型を特定する
        throw;
    } catch (const anet::AnetException& e1) {
        auto msg = wxString::FromUTF8(e1.what());
        auto detail = wxString::FromUTF8(e1.stack_trace());
        ShowErrorDialog(msg, detail);
    } catch (const std::exception& e) {
        auto msg = wxString::FromUTF8(e.what());
        ShowErrorDialog(msg);
    } catch (...) {
        wxMessageBox("System error", "System Error", wxICON_ERROR);
    }
}

bool RunnerApp::OnExceptionInMainLoop()
{
    showFatalError();
    return false;
}

void RunnerApp::OnUnhandledException()
{
    showFatalError();
}

void RunnerApp::InitTrainer()
{
    // TrainObserver
    trainer_->GetNotifier()->Attach<anet::rl::FunctionTrainObserver>(
        [this](const anet::rl::TrainEvent& event)
        {
            anet::ProfileRange r1("FunctionTrainObserver");
            auto train_step = event.counts.train_step;

            // Trainスナップショット取得
            //if (snapshot_store_.IsDataRequest() || (train_step % 2000 == 0)) {
            //    ANET_LOG_DEBUG("UI data. train_step=" << train_step);

                // Plotデータ追加
                //auto train_reward_ema = event.runner.GetScalar(anet::rl::Runner::TARGET_EVAL_REWARD);
                //ANET_ASSERT(train_reward_ema.has_value());
                //frame_->AddPlotData(*train_reward_ema);

                // UIスナップショットを生成＆更新
                //auto snapshot = CreateSnapshot(event);
                //snapshot_store_.Update(snapshot);
            //}

            // Trainログ
            if (event.counts.train_step % 100 == 0) {
                auto train_reward_ema = event.runner.GetScalar(anet::rl::Runner::TRAIN_REWARD_EMA);
                ANET_ASSERT(train_reward_ema.has_value());
                LOG::info() << "train_step=" << train_step << " train_mean_reward=" << *train_reward_ema;
            }

        }, "RunnerApp");

    // LearnObserver
    trainer_->GetNotifier()->Attach<anet::rl::FunctionLearnObserver>(
        [](const anet::rl::LearnEvent& event)
        {
            auto train_step = event.counts.train_step;
            auto learn_step = event.counts.learn_step;

            // Learnログ
            if (event.counts.learn_step % 100 == 0) {
                LOG::info() << "train_step=" << train_step << " learn_step=" << learn_step;

                auto obs_norm_stat_mean = event.agent->GetTensor(anet::rl::ObservationNormalizer::kKeyMean);
                auto obs_norm_stat_std = event.agent->GetTensor(anet::rl::ObservationNormalizer::kKeyStd);

                if (obs_norm_stat_mean.has_value() && obs_norm_stat_mean) {
                    LOG::info() << "obs_norm_stat_mean=" << anet::ToString(*obs_norm_stat_mean);
                }
                if (obs_norm_stat_std.has_value()) {
                    LOG::info() << "obs_norm_stat_std=" << anet::ToString(*obs_norm_stat_std);
                }
            }

        }, "RunnerApp");
}

std::shared_ptr<anet::rl::gui::View> RunnerApp::CreateExperinceView(wxWindow* parent)
{
    auto env_class_id = trainer_->GetEnvClassId();
    auto notifier = trainer_->GetNotifier();
    auto view = view_factory_->CreateView(parent, env_class_id, notifier);
    if (view == nullptr) {
        LOG::error() << "Failed to create view. env_class_id=" << env_class_id;
        return nullptr;
    }
    return view;
}


//void RunnerApp::InitPERImageLogObservers(const anet::ConfigData& config_data)
//{
//    anet::rl::dqn::RainbowAgentConfig agent_config(config_data);
//    if (!agent_config.learner.use_per) {
//        LOG::warn() << "PER agent config disabled. Skipping PER ImageLog observer.";
//        return;
//    }
//
//    auto notifier = trainer_->GetNotifier();
//    auto env_spec = trainer_->GetBatchEnv()->GetSpec();
//    auto agent = trainer_->GetAgent();
//
//    // flags
//    auto flags =
//        //anet::HeatMapFlags::HM_LogScaleValue | 
//        anet::HeatMapFlags::HM_AutoNormValue
//        | anet::HeatMapFlags::HM_AutoScaleAxis
//        //| anet::HeatMapFlags::HM_LogScaleAxis
//        | anet::HeatMapFlags::HM_SumMode; // | anet::HeatMapFlags::HM_ShowZeroLine;
//
//    // ---- ReplayBuffer ----
//    auto rep_x_probe = std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 0, &env_spec.state_spec);
//    auto rep_y_probe = std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::NEXT_STATE_OBS, 1, &env_spec.state_spec);
//    auto rep_prio_probe = std::make_shared<anet::rl::AgentTensorVectorProbe>(anet::rl::ReplayBuffer::PER_DIST, -1);
//
//    anet::rl::HeatMapObserverConfig replay_heat_obs_config{
//        512,    // width
//        512,    // height
//        config_->image_log_interval,    // log_interval 
//        //100000,  // max_points
//        agent_config.learner.replay_capacity,	// max_points
//        flags,   // flags
//        -1,     // image_width
//        -1,     // image_height
//    };
//
//    auto auto_scale_mode = anet::rl::AgentTensorVectorProbe::AutoScaleMode::DISABLE;    // EnvSpecで固定
//    notifier->Attach<anet::rl::HeatMapVectorObserver>(
//        "43_agent_img/52_per_hm_prio_01_", replay_heat_obs_config, rep_x_probe, rep_y_probe, rep_prio_probe);
//
//
//    anet::rl::TimeHistogramObserverConfig prio_hist_obs_config{
//        256,    // x bins
//        1920,   // y max_frames
//        512,    // image_height
//        1920,   // image_width
//        anet::TimeFrameMode::Scale,             // mode
//        flags | anet::HeatMapFlags::HM_FlipY | anet::HeatMapFlags::HM_LogScaleAxis,   // flags
//        config_->image_log_interval_thm,    // log_interval
//        20,     // frame_interval
//        std::numeric_limits<float>::quiet_NaN(),
//        std::numeric_limits<float>::quiet_NaN(),
//        1.0f// alpha = 0.05f
//    };
//    //notifier->Attach<anet::rl::TimeHistogramObserver>(
//    //    "44_agent_img/52_per_thg_prio", prio_hist_obs_config, rep_prio_probe);
//
//}


wxIMPLEMENT_APP(RunnerApp);

