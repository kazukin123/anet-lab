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
#include "LunarLanderFrame.hpp"
#include "UISnapshot.hpp"

namespace LOG = anet::log;

wxDEFINE_EVENT(wxEVT_TRAINER_EXIT, wxCommandEvent);
wxDEFINE_EVENT(wxEVT_APP_TRAINER_SHUTDOWN, wxThreadEvent);

struct LunarLanderApp::Config : public anet::Config
{
    int train_pause_step = -1;
    int train_exit_step = -1; //110000;
    bool enable_image_log = true;
    int train_timer_ms = 10;
    int eval_timer_ms = 10;
    int eval_step_per_frame = 1;

    LunarLanderApp::Config(const anet::ConfigData& config_data) : anet::Config(config_data, "LunarLanderApp")
    {
        ANET_READ_CONFIG(config_data, train_pause_step);
        ANET_READ_CONFIG(config_data, train_exit_step);
        ANET_READ_CONFIG(config_data, enable_image_log);
        ANET_READ_CONFIG(config_data, train_timer_ms);
        ANET_READ_CONFIG(config_data, eval_timer_ms);
        ANET_READ_CONFIG(config_data, eval_step_per_frame);
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
    { wxCMD_LINE_USAGE_TEXT, NULL,    NULL,    "LunarLanderRLGUI.exe key1=value1 key2=value2" },     //  Additional usage text.
    { wxCMD_LINE_NONE } // 終了マーク
};

bool LunarLanderApp::OnInit()
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

    // LunarLanderAppConfig
    config_ = std::make_unique<LunarLanderApp::Config>(config_data);

    // LunarLanderFrame
    frame_ = new LunarLanderFrame("LunarLander RL", config_->train_timer_ms, config_->eval_timer_ms, config_->eval_step_per_frame);
    frame_->Show(true);

    // MetricsLogger
    anet::MetricsLogger::Init(std::make_unique<anet::JsonlBackend>(), GetLogsPath());

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
    InitImageLogObservers();

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

    ANET_LOG_DEBUG("batch_step_result=" << event.batch_step_result.ToString());

    const int ENV_INDEX = 0;
    
    // RL由来情報
    auto train_step = event.counts.train_step;
    anet::rl::SingleState state = {
        event.batch_step_result.next_state.obs[ENV_INDEX],
        event.batch_step_result.next_state.done[ENV_INDEX].item<bool>(),
        event.batch_step_result.next_state.truncated[ENV_INDEX].item<bool>(),
        event.batch_step_result.next_state.episode_start[ENV_INDEX].item<bool>(),
    };
    auto action = event.batch_exp.action.action[ENV_INDEX].item<int64_t>();
    auto reward = event.batch_exp.reward[ENV_INDEX].item<float>();
    const auto& aux = event.batch_step_result.auxs[ENV_INDEX];

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
            if (train_step % 1 == 0) {
                // 平均報酬をPlotデータ追加
                auto train_reward_ema = event.runner.GetScalar(anet::rl::Runner::TRAIN_REWARD_EMA);
                ANET_ASSERT(train_reward_ema.has_value());
                frame_->AddPlotData(*train_reward_ema);

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
}

wxIMPLEMENT_APP(LunarLanderApp);

