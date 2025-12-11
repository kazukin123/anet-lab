// app.cpp
#include "LunarLanderApp.hpp"
#include <filesystem>
#include <wx/stdpaths.h>
#include <wx/cmdline.h>
#include <wx/filename.h>
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
    int timer_ms = 10;

    LunarLanderApp::Config(const anet::ConfigData& config_data) : anet::Config(config_data, "LunarLanderApp")
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

    // MetricsLogger
    anet::MetricsLogger::Init(std::make_unique<anet::JsonlBackend>(), GetLogsPath());

    // LunarLanderFrame
    frame_ = new LunarLanderFrame("LunarLander RL", config_->timer_ms);
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

void LunarLanderApp::InitTrainer()
{
    // TrainObserver
    trainer_->GetNotifier()->Attach<anet::rl::FunctionTrainObserver>(
        [this](const anet::rl::TrainEvent& event)
        {
            auto train_step = event.counts.train_step;

            // Trainスナップショット取得
            if (train_step % 1 == 0) {
                // 平均報酬をPlotデータ追加
                auto train_reward_ema = event.trainer.GetScalar(anet::rl::Trainer::TRAIN_REWARD_EMA);
                ANET_ASSERT(train_reward_ema.has_value());
                frame_->AddPlotData(*train_reward_ema);

                auto exps = event.batch_exp.ToExperienceList();
                ANET_ASSERT(exps.size() > 0);

                const int env_index = 0;

                auto exp = exps[env_index];

                // Train状況のSnapshotを更新
                UISnapshot snapshot { train_step, exps[0] };

                // ---- wind_x ----
                {
                    auto w = event.env->GetScalar("wind_x", env_index);
                    snapshot.wind_x = w.has_value() ? *w : 0.0f;  // fallback
                }

                // ---- pad ----
                {
                    auto t = event.env->GetTensor("pad", env_index);
                    if (t.has_value()) {
                        const auto& pad_tensor = *t;
                        ANET_ASSERT(pad_tensor.size(0) == 3);
                        snapshot.pad.x1 = pad_tensor[0].item<float>();
                        snapshot.pad.x2 = pad_tensor[1].item<float>();
                        snapshot.pad.y = pad_tensor[2].item<float>();
                    }
                }

                // ---- world bounds ----
                {
                    auto t = event.env->GetTensor("world_bounds", env_index);
                    if (t.has_value()) {
                        const auto& b = *t;
                        ANET_ASSERT(b.size(0) == 4);
                        snapshot.world_min_x = b[0].item<float>();
                        snapshot.world_max_x = b[1].item<float>();
                        snapshot.world_min_y = b[2].item<float>();
                        snapshot.world_max_y = b[3].item<float>();
                    }
                }

                // ---- terrain polyline ----
                //if (exp.state.episode_start) {    /// @todo episode_start判定
                {
                    auto tv = event.env->GetTensorVector("terrain", env_index);
                    if (tv.has_value()) {
                        TerrainPolyline poly;
                        for (auto& pt : *tv) {
                            ANET_ASSERT(pt.size(0) == 2);
                            TerrainPoint p{
                                pt[0].item<float>(),
                                pt[1].item<float>()
                            };
                            poly.points.push_back(p);
                        }
                        snapshot.terrain = poly;
                    }
                }
                snapshot_store_.Update(snapshot);
            }

            // Trainログ
            if (event.counts.train_step % 100 == 0) {
                auto train_reward_ema = event.trainer.GetScalar(anet::rl::Trainer::TRAIN_REWARD_EMA);
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

