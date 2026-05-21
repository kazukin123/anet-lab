// RunnerApp.cpp
#include "RunnerApp.hpp"
#include <filesystem>
#include <wx/stdpaths.h>
#include <wx/cmdline.h>
#include <wx/filename.h>
#include <wx/snglinst.h>
#include "anet/profile.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/init.hpp"
#include "anet/log.hpp"
#include "anet/observers.hpp"
#include "anet/scaler.hpp"
#include "anet/exception.hpp"
#include "anet/env/LunarLander.hpp"
#include "anet/env/CartPole.hpp"
#include "anet/env/DropMerge.hpp"
#include "anet/env/GridMaze.hpp"
#include "anet/env/ImageCls.hpp"
#include "ErrorDialog.hpp"
#include "RunnerFrame.hpp"
#include "TrainPanel.hpp"
#include "EvalPanel.hpp"

namespace LOG = anet::log;


wxDEFINE_EVENT(wxEVT_TRAINER_EXIT, wxCommandEvent);
wxDEFINE_EVENT(wxEVT_APP_TRAINER_SHUTDOWN, wxThreadEvent);


struct RunnerApp::Config : public anet::Config {
    std::string run_name = "run_{%t}";
    std::string log_level = "info";
    bool train_auto_start = true;
    int train_pause_step = -1;
    int train_exit_step = -1; //110000;
    int exp_pause_step = -1;
    int exp_exit_step = -1;

    anet::MetricsLoggerConfig metrics_logger;
    TrainPanelConfig train_panel;
    EvalPanelConfig eval_panel;

    Config(const anet::ConfigData& config_data) : anet::Config(config_data, "app")
    {
        ANET_READ_CONFIG(config_data, run_name);
        ANET_READ_CONFIG(config_data, log_level);

        ANET_READ_CONFIG(config_data, train_auto_start);
        ANET_READ_CONFIG(config_data, train_pause_step);
        ANET_READ_CONFIG(config_data, train_exit_step);
        ANET_READ_CONFIG(config_data, exp_pause_step);
        ANET_READ_CONFIG(config_data, exp_exit_step);

        ANET_READ_CONFIG(config_data, metrics_logger.runs_dir);
        metrics_logger.run_name_tmpl = run_name;
        ANET_READ_CONFIG(config_data, metrics_logger.video_codec);
        ANET_READ_CONFIG(config_data, metrics_logger.video_fps);
        ANET_READ_CONFIG(config_data, metrics_logger.use_png_dump);

        ANET_READ_CONFIG(config_data, train_panel.fps);
        ANET_READ_CONFIG(config_data, eval_panel.fps);
        ANET_READ_CONFIG(config_data, eval_panel.step_per_frame);
        ANET_READ_CONFIG(config_data, eval_panel.auto_start);
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

    // wxWidgets初期化
    wxInitAllImageHandlers();

    // DarkModeサポート
    SetAppearance(Appearance::System);

    // 全体ログレベル設定
#if ANET_ENABLE_DEBUGINFO
    wxLog::SetLogLevel(wxLOG_Debug);
#else
    wxLog::SetLogLevel(wxLOG_Info);
#endif

    // メインスレッドでffmpeg実行する準備
    Bind(wxEVT_APP_EXECUTE_START, [&](wxThreadEvent& event) {
        anet::ExecuteStarter* executer = event.GetPayload<anet::ExecuteStarter*>();
        ANET_LOG_DEBUG("Executing command on main thread. command=" << executer->GetCommand());
        executer->OnMainStart();   // wxExecute 内部呼び出し
        });

    // コマンドライン引数解析
    wxCmdLineParser cmdline(desc, argc, (wchar_t**)argv);
    if (cmdline.Parse(true)) {
        return false;
    }

    // ConfigManager
    config_mgr_ = std::make_unique<anet::ConfigManager>(GetConfigFilePath(), &cmdline);
    auto config_data = config_mgr_->GetConfigData();

    // RunnerApp設定生成
    config_ = std::make_unique<RunnerApp::Config>(config_data);

    // MetricsLogger
    anet::MetricsLogger::Init(std::make_unique<anet::JsonlBackend>(), config_->metrics_logger, GetProjectRootDir());
    anet::MetricsLogger::Instance()->Log("config_data", config_data.ToJson());

    // ライブラリ初期化
	anet::rl::BackendConfig backend_config(config_data);
    anet::rl::InitRL(backend_config);

	// ENV初期化
    anet::rl::env::InitLunarLander();
    anet::rl::env::InitCartPole();
    anet::rl::env::InitDropMerge();
    anet::rl::env::InitGridMaze();
    anet::rl::env::InitImageCls();

    // RunnerFrame生成
    wxString frame_title = anet::MetricsLogger::Instance()->GetRunName() + " - ANET RL Runner";
    frame_ = new RunnerFrame(frame_title, config_->train_panel, config_->eval_panel);

    // ログ初期化
    SetupLogging();

    // RunNameを記録
    //this->WriteLastRunName(anet::MetricsLogger::Instance()->GetRunName());

    // RunnerFrame表示
    frame_->Show();

    // Configをダンプ
    LOG::info() << "RunnerApp config=" << config_->ToString();
    anet::MetricsLogger::Instance()->Log(*config_);

    // RunManager生成
    run_manager_ = std::make_shared<anet::rl::RunManager>(config_data);
    auto status = run_manager_->GetStatus();
    if (status != anet::rl::RunnerStatus::RUNNING) {
        LOG::error() << "Failed to initialize RunManager.";
        return true;
    }

    // DefaultViewFactory
    view_factory_ = std::make_unique<anet::rl::gui::DefaultViewFactory>(config_data);

    // Trainer初期化
    InitTrainer();

    // ImageProviderManager
    auto notifier = run_manager_->GetNotifier();
    auto agent = run_manager_->GetAgent();
    auto env_spec = run_manager_->GetTrainRunner()->GetBatchEnv()->GetSpec();
    auto train_runner = run_manager_->GetTrainRunner();
    img_prov_mgr_ = std::make_unique<anet::rl::ImageProviderManager>(env_spec, notifier);
    img_prov_mgr_->CreateAndRegister(config_data, agent, train_runner);

    // Trainer初期化後のUI初期化
    frame_->Initialize(run_manager_);

    // ログ
    run_manager_->GetNotifier()->LogObservers();

    // Trainerスレッド生成
    trainer_thread_ = std::make_unique<anet::rl::RunnerThread>(
        "TrainThread",
        run_manager_->GetTrainRunner(),
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
                wxGetApp().GetMainFrame()->SetStatusText("Auto pause.");
                return anet::rl::ControlSignal::BREAK;
            }
            if ((config_->exp_pause_step > 0) && (exp_step >= config_->exp_pause_step) && !auto_pause_done_) {
                auto_pause_done_ = true;    // 一回だけ自動
                trainer_thread_->Pause();
                LOG::info() << "Auto pause.";
                wxGetApp().GetMainFrame()->SetStatusText("Auto pause.");
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

    // 永続化
    //SaveAgent("agent_init.anet");

    // Train開始！
    if (!config_->train_auto_start)
        trainer_thread_->Pause();
    trainer_thread_->Start();

    return true;
}

std::filesystem::path RunnerApp::GetRunDir()
{
    return anet::MetricsLogger::Instance()->GetRunDir();
}

std::ofstream RunnerApp::GetOutputStream(const std::string& file_name)
{
    auto file = GetRunDir() / file_name;
    return std::ofstream(file, std::ios::binary);
}

void RunnerApp::SetupLogging()
{
    std::filesystem::path run_dir = GetRunDir();
    auto run_name = anet::MetricsLogger::Instance()->GetRunName();
    std::filesystem::create_directories(run_dir); // 念のためディレクトリ作成
    std::filesystem::path log_file_path = run_dir / (run_name + ".log");

    // ファイルを追記モードで開く（パス自体に日本語が含まれても安全なようにwstringを使用）
    FILE* log_file = wxFopen(log_file_path.wstring(),"a");

    if (log_file) {
        // 標準の wxLogStderr ではなく、UTF-8専用ロガーを生成
        wxLog* file_logger = new anet::log::FileLogger(log_file);
        file_logger->SetFormatter(new anet::log::LogFormatter(/*enable_timestamp=*/true));

        // 既存のUIロガー(LogPanel等)を維持したまま、ファイルロガーをチェーンに追加
        new wxLogChain(file_logger);
    }
}

int64_t RunnerApp::SaveAgent(const std::string& file_name)
{
    wxBeginBusyCursor();

    auto agent = GetRunManager().GetAgent();
    auto os = wxGetApp().GetOutputStream(file_name);
    anet::OutputArchive archive(os, file_name);
    auto size = agent->Save(archive);
    os.close();
    
    wxEndBusyCursor();

    return size;
}

void RunnerApp::ToggleTraining()
{
    bool paused = trainer_thread_->IsPaused();
    std::string log_str;
    if (paused) {
        trainer_thread_->Resume();
        log_str = "Training resumed";
    } else {
        trainer_thread_->Pause();
        log_str = "Training paused";
    }
    LOG::info() << log_str;
    wxGetApp().GetMainFrame()->SetStatusText(log_str);
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
    run_manager_->GetNotifier()->Attach<anet::rl::FunctionTrainObserver>(
        [](const anet::rl::TrainEvent&)
        {
            anet::ProfileRange r1("FunctionTrainObserver");

        }, "RunnerApp");

    // LearnObserver
    run_manager_->GetNotifier()->Attach<anet::rl::FunctionLearnObserver>(
        [](const anet::rl::LearnEvent& event)
        {
            // Learnログ
            if (event.counts.learn_step % 100 == 0) {
                auto train_step = event.counts.train_step;
                auto learn_step = event.counts.learn_step;
                auto exp_step = event.counts.exp_step;

                LOG::info() << "train_step=" << train_step << " exp_step=" << exp_step << " learn_step=" << learn_step;

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
    auto env_class_id = run_manager_->GetEnvClassId();
    auto notifier = run_manager_->GetNotifier();
    auto view = view_factory_->CreateView(parent, env_class_id, notifier);
    if (view == nullptr) {
        LOG::error() << "Failed to create view. env_class_id=" << env_class_id;
        return nullptr;
    }
    return view;
}

bool RunnerApp::WriteLastRunName(const std::string& run_name) const
{
    // 保存先ディレクトリ（var）のパスを作成・取得
    //wxFileName dir(GetProjectRootDir().string(), "");
    //LOG::info() << "fn1=" << fn.GetFullPath().c_str();
    //fn.AppendDir("var");
    //LOG::info() << "dir=" << dir.GetFullPath().c_str();

    // ディレクトリがなければ作成
    //if (!dir.DirExists()) {
        //dir.Mkdir(wxPATH_MKDIR_FULL);
    //}

    // ファイル名を設定
    wxFileName fn(GetProjectRootDir().string(), "runname.txt");
    wxString file_path = fn.GetFullPath();
    //LOG::info() << "file_path=" << file_path;

    // ファイルごとに固有のロック名を作成
    wxSingleInstanceChecker lock(file_path);

    // 他のプロセスが掴んでいたら待機
    int attempts = 0;
    const int maxAttempts = 50; // 最大5秒待機 (100ms * 50)
    while (lock.IsAnotherRunning()) {
        if (++attempts > maxAttempts) {
			ANET_SYSTEM_ERROR("Failed to acquire lock for writing last run name after multiple attempts.");
            return false;
        }
        wxMilliSleep(100);
    }

    // ファイルを新規作成(無ければ)して開く
    wxFile file;
    if (file.Create(file_path, true)) {
        file.Write(run_name);
        file.Close();
    } else {
        ANET_SYSTEM_ERROR("Failed to create or open file for writing runname.txt.");
        return false;
	}
        
    return true;
}

wxIMPLEMENT_APP(RunnerApp);

