// RunnerApp.cpp
#include "RunnerApp.hpp"
#include <array>
#include <exception>
#include <filesystem>
#include <vector>
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
#include "WorkspaceDialog.hpp"

namespace LOG = anet::log;

namespace {

constexpr int kTextLogFlushTimerId = wxID_HIGHEST + 100;

} // namespace


wxDEFINE_EVENT(wxEVT_TRAINER_EXIT, wxCommandEvent);
wxDEFINE_EVENT(wxEVT_APP_TRAINER_SHUTDOWN, wxThreadEvent);


struct RunnerApp::Config : public anet::Config {
    std::string run_name = "run_{%t}";
    std::string runs_dir = "runs";
    std::string log_level = "info";
    int log_flush_interval_ms = 500;
    bool train_auto_start = true;
    int train_pause_step = -1;
    int train_exit_step = -1; //110000;
    int exp_pause_step = -1;
    int exp_exit_step = -1;

    anet::MetricsLoggerConfig metrics_logger;
    TrainPanelConfig train_panel;
    EvalPanelConfig eval_panel;
    std::vector<std::string> warnings;

    Config(const anet::ConfigData& config_data) : anet::Config(config_data, "app")
    {
        ANET_READ_CONFIG(config_data, run_name);
        ANET_READ_CONFIG(config_data, runs_dir);
        ANET_READ_CONFIG(config_data, log_level);
        ANET_READ_CONFIG(config_data, log_flush_interval_ms);

        ANET_READ_CONFIG(config_data, train_auto_start);
        ANET_READ_CONFIG(config_data, train_pause_step);
        ANET_READ_CONFIG(config_data, train_exit_step);
        ANET_READ_CONFIG(config_data, exp_pause_step);
        ANET_READ_CONFIG(config_data, exp_exit_step);

        WarnDeprecatedRunsDirConfig(config_data);
        ValidateRunsDir();
        ValidateLogFlushInterval();
        metrics_logger.runs_dir = runs_dir;
        metrics_logger.run_name_tmpl = run_name;
        ANET_READ_CONFIG(config_data, metrics_logger.video_codec);
        ANET_READ_CONFIG(config_data, metrics_logger.video_fps);
        ANET_READ_CONFIG(config_data, metrics_logger.use_png_dump);

        ANET_READ_CONFIG(config_data, train_panel.fps);
        ANET_READ_CONFIG(config_data, eval_panel.fps);
        ANET_READ_CONFIG(config_data, eval_panel.step_per_frame);
        ANET_READ_CONFIG(config_data, eval_panel.auto_start);
        ANET_READ_CONFIG(config_data, eval_panel.eval_config_tag);
        ANET_READ_CONFIG(config_data, eval_panel.model_sync.mode);
        ANET_READ_CONFIG(config_data, eval_panel.model_sync.frame_interval);
        ANET_READ_CONFIG(config_data, eval_panel.model_sync.time_interval_ms);
        ANET_READ_CONFIG(config_data, eval_panel.model_sync.episode_interval);
        eval_panel.model_sync.Validate();
    }

private:
    void WarnDeprecatedRunsDirConfig(const anet::ConfigData& config_data)
    {
        for (const char* key : std::array{
            "app.metrics_logger.runs_dir",
            "app.metrics_logger.root_dir",
        }) {
            if (!config_data.Has(key)) {
                continue;
            }

            const auto value = config_data.Get(key);
            warnings.push_back(
                "Config key " + std::string(key) + " is deprecated and ignored. value=\""
                + value + "\". Use app.runs_dir instead.");
        }
    }

    void ValidateRunsDir() const
    {
        if (runs_dir.empty()) {
            ANET_SYSTEM_ERROR("Invalid config key app.runs_dir: value must not be empty.");
        }
    }

    void ValidateLogFlushInterval() const
    {
        if (log_flush_interval_ms < 0) {
            ANET_SYSTEM_ERROR(
                "Invalid config key app.log_flush_interval_ms: value=" << log_flush_interval_ms
                << " (expected: >= 0; 0 disables periodic flushing).");
        }
    }
};

std::filesystem::path GetConfigFilePath()
{
    return anet::GetExecutableConfigDir() / "_main.txt";
}

std::filesystem::path GetConfigFilePath(const wxCmdLineParser& cmdline)
{
    wxString config_path_arg;
    if (!cmdline.Found("config", &config_path_arg)) {
        return GetConfigFilePath();
    }

    std::filesystem::path config_path(config_path_arg.ToStdWstring());
    if (config_path.empty()) {
        ANET_SYSTEM_ERROR("Invalid command line option --config: value must not be empty.");
    }

    return config_path;
}

static wxCmdLineEntryDesc desc[] = {
    // kind,              short-name, long-name, usage,      type,                  flags
    //{ wxCMD_LINE_SWITCH, "v",         "verbose", "エラー表示を饒舌に" }, // wxCMD_LINE_SWITCH:A boolean argument of the program;    e.g. -v to enable verbose mode.
    //{ wxCMD_LINE_OPTION, "f",         "file",    "設定ファイルのパス" }, // wxCMD_LINE_OPTION:An argument with an associated value; e.g. -o filename
    {
        wxCMD_LINE_OPTION,
        "c",
        "config",
        "main config file path",
        wxCMD_LINE_VAL_STRING,
        0
    },
    {
        wxCMD_LINE_OPTION,
        "w",
        "workspace",
        "workspace path (relative paths use runner/workspaces)",
        wxCMD_LINE_VAL_STRING,
        0
    },
    {
        wxCMD_LINE_SWITCH,
        nullptr,
        "select-workspace",
        "always show the workspace selection dialog",
        wxCMD_LINE_VAL_NONE,
        0
    },

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

    // 起動入力を排他的に解決し、workspace モードでは config と Run 出力先を同じ箱へ束ねる。
    wxString config_arg;
    wxString workspace_arg;
    const bool has_config = cmdline.Found("config", &config_arg);
    const bool has_workspace = cmdline.Found("workspace", &workspace_arg);
    const bool select_workspace = cmdline.Found("select-workspace");
    const auto config_mode = anet::DetermineAppConfigMode(
        has_config, has_workspace, select_workspace);
    std::string startup_warning;
    std::string startup_info;

    if (config_mode == anet::AppConfigMode::DirectConfig) {
        startup_info = "Workspace resolution bypassed because --config was specified.";
        config_mgr_ = std::make_unique<anet::ConfigManager>(GetConfigFilePath(cmdline), &cmdline);
    } else {
        anet::WorkspaceService workspace_service(
            anet::GetExecutableRootDir(), anet::GetAppDataDir());
        std::optional<anet::WorkspacePaths> workspace;

        if (config_mode == anet::AppConfigMode::ExplicitWorkspace) {
            workspace = workspace_service.Resolve(workspace_arg.utf8_string(), true);
        } else if (select_workspace
            || !anet::runner::LoadWorkspaceDialogSkip(workspace_service.AppDataDir())) {
            const auto selection = anet::runner::ShowWorkspaceDialog(nullptr, workspace_service);
            if (!selection.has_value()) {
                return false;
            }
            anet::runner::SaveWorkspaceDialogSkip(
                workspace_service.AppDataDir(), selection->skip_dialog);
            workspace = workspace_service.Resolve(selection->input, true);
        } else {
            const auto history = workspace_service.LoadHistory();
            if (!history.empty()) {
                try {
                    workspace = workspace_service.Resolve(history.front(), false);
                } catch (const std::exception& e) {
                    startup_warning = "Last workspace is unavailable; falling back to _default. path=\""
                        + history.front() + "\" error=" + e.what();
                }
            }
            if (!workspace.has_value()) {
                workspace = workspace_service.Resolve("_default", true);
            }
        }

        workspace_service.RecordHistory(workspace->input);
        workspace_service.SaveLastWorkspace(*workspace);
        config_mgr_ = anet::CreateWorkspaceConfigManager(
            *workspace, anet::GetExecutableConfigDir(), &cmdline);
        const auto workspace_root_text = workspace->root.u8string();
        startup_info = "Workspace selected. input=\"" + workspace->input
            + "\" path="
            + std::string(
                reinterpret_cast<const char*>(workspace_root_text.data()),
                workspace_root_text.size());
    }
    auto config_data = config_mgr_->GetConfigData();

    // RunnerApp設定生成
    config_ = std::make_unique<RunnerApp::Config>(config_data);

    // MetricsLogger
    anet::MetricsLogger::Init(std::make_unique<anet::JsonlBackend>(), config_->metrics_logger, anet::GetExecutableRootDir());
    standard_stream_logger_.Start(GetRunDir());
    anet::MetricsLogger::Instance()->Log("config_data", config_data.ToJson());
    anet::MetricsLogger::Instance()->Log("config_data", config_data);

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

    // workspace 確定中の通知は、wxLogGUI ではなく UI と Run file のログへ送る。
    if (!startup_warning.empty()) {
        LOG::warn() << startup_warning;
    }
    if (!startup_info.empty()) {
        LOG::info() << startup_info;
    }
    standard_stream_logger_.LogStatus();
    for (const auto& warning : config_->warnings) {
        LOG::warn() << warning;
    }

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
                wxGetApp().FlushRunOutputs();
                return anet::rl::ControlSignal::BREAK;
            }
            if ((config_->exp_pause_step > 0) && (exp_step >= config_->exp_pause_step) && !auto_pause_done_) {
                auto_pause_done_ = true;    // 一回だけ自動
                trainer_thread_->Pause();
                LOG::info() << "Auto pause.";
                wxGetApp().GetMainFrame()->SetStatusText("Auto pause.");
                wxGetApp().FlushRunOutputs();
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
    if (!config_->train_auto_start) {
        trainer_thread_->Pause();
        FlushRunOutputs();
    }
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
        run_log_chain_ = new wxLogChain(file_logger);

        // 表示FPSと独立した周期で、RunName.logだけを外部から読める状態へ近づける。
        if (config_->log_flush_interval_ms > 0) {
            text_log_flush_timer_.SetOwner(this, kTextLogFlushTimerId);
            Bind(wxEVT_TIMER, &RunnerApp::OnTextLogFlushTimer, this, kTextLogFlushTimerId);
            text_log_flush_timer_.Start(config_->log_flush_interval_ms);
        }
    }
}

void RunnerApp::FlushRunOutputs()
{
    // 標準出力とメトリクスは、pause/saveなどの明示的な即時境界でのみflushする。
    standard_stream_logger_.Flush();
    anet::MetricsLogger::Instance()->Flush();

    // RunName.logを含むwxLogは、メインスレッドで保留分までまとめて排出する。
    FlushTextLog();
}

void RunnerApp::FlushTextLog()
{
    if (wxThread::IsMain()) {
        wxLog::FlushActive();
        return;
    }

    CallAfter([] {
        wxLog::FlushActive();
    });
}

void RunnerApp::OnTextLogFlushTimer(wxTimerEvent& WXUNUSED(event))
{
    // timerはtext log専用とし、metrics/stdoutのI/O周期を増やさない。
    ANET_PROFILE_SCOPE(text_log_flush);
    FlushTextLog();
}

void RunnerApp::ShutdownRunLogging()
{
    // shutdown開始後に周期イベントがfile loggerへ触れないよう先に停止する。
    text_log_flush_timer_.Stop();

    if (run_log_chain_ == nullptr) {
        return;
    }

    // Save完了など、終了直前に発生した全出力をfile loggerのclose前に確定する。
    FlushRunOutputs();

    // LogPanelが破棄される前にactive targetを戻し、chainが所有するFileLoggerをcloseする。
    if (wxLog::GetActiveTarget() != run_log_chain_) {
        run_log_chain_ = nullptr;
        return;
    }

    wxLogChain* const run_log_chain = run_log_chain_;
    run_log_chain_ = nullptr;
    wxLog* const detached_chain = wxLog::SetActiveTarget(run_log_chain->GetOldLog());
    if (detached_chain == run_log_chain) {
        delete detached_chain;
    }
}

int64_t RunnerApp::SaveAgent(const std::string& file_name)
{
    const auto file_path = GetRunDir() / file_name;
    auto log_file_path = file_path.lexically_relative(anet::GetExecutableRootDir());
    if (log_file_path.empty()) {
        log_file_path = file_name;
    }
    const auto log_file_path_str = log_file_path.string();

    LOG::info() << "Started saving agent: file=" << log_file_path_str;
    FlushTextLog();

    wxBeginBusyCursor();

    int64_t size = 0;
    try {
        auto agent = GetRunManager().GetAgent();
        auto os = wxGetApp().GetOutputStream(file_name);
        anet::OutputArchive archive(os, file_name);
        size = agent->Save(archive);
        os.close();
    } catch (const std::exception& e) {
        wxEndBusyCursor();
        LOG::error() << "Failed to save agent: file=" << log_file_path_str << " error=" << e.what();
        throw;
    } catch (...) {
        wxEndBusyCursor();
        LOG::error() << "Failed to save agent: file=" << log_file_path_str;
        throw;
    }

    wxEndBusyCursor();

    LOG::info() << "Finished saving agent: file=" << log_file_path_str << " size=" << size << " bytes";
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
    FlushRunOutputs();
}

void RunnerApp::StopTraining()
{
    trainer_thread_->Stop();
}

int RunnerApp::OnExit()
{
    trainer_thread_->Stop();
    ShutdownRunLogging();
    anet::MetricsLogger::Reset();
    standard_stream_logger_.Flush();
    standard_stream_logger_.Stop();
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
            ANET_PROFILE_SCOPE_FULL(function_train_observer, "FunctionTrainObserver");

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
    //wxFileName dir(anet::GetExecutableRootDir().string(), "");
    //LOG::info() << "fn1=" << fn.GetFullPath().c_str();
    //fn.AppendDir("var");
    //LOG::info() << "dir=" << dir.GetFullPath().c_str();

    // ディレクトリがなければ作成
    //if (!dir.DirExists()) {
        //dir.Mkdir(wxPATH_MKDIR_FULL);
    //}

    // ファイル名を設定
    wxFileName fn(wxString(anet::GetAppDataDir().wstring()), "runname.txt");
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
