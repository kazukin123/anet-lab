// RunnerApp.hpp

#pragma once

#include <memory>
#include <optional>
#include <filesystem>
#include <wx/app.h>
#include <wx/log.h>
#include <wx/timer.h>
#include "anet/config.hpp"
#include "anet/trainer.hpp"
#include "anet/gui.hpp"
#include "anet/image.hpp"

#include "anet/app_util.hpp"
#include "RunnerFrame.hpp"


//bool IsDarkMode() {
//    // 背景色の明るさ(Luminance)を取得して判定
//    wxColour bg = wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOW);
//    // 平均輝度が0.5未満ならダークモードとみなす
//    return ((0.299 * bg.Red() + 0.587 * bg.Green() + 0.114 * bg.Blue()) / 255.0) < 0.5;
//}

wxDECLARE_EVENT(wxEVT_TRAINER_EXIT, wxCommandEvent);
wxDECLARE_EVENT(wxEVT_APP_TRAINER_SHUTDOWN, wxThreadEvent);

class RunnerApp final : public wxApp {
public:
    ~RunnerApp() override;
    bool OnInit() override;
    int OnRun() override;
    int OnExit() override;
    bool OnExceptionInMainLoop() override;
    void OnUnhandledException() override;
public:
    void ToggleTraining();
    void PauseTraining();
    void StopTraining();
    bool IsTrainingPaused() const;
    bool IsTrainingRunning() const;
    anet::ConfigData GetConfigData() const { return config_mgr_->GetConfigData(); }
    anet::rl::RunManager& GetRunManager() { return *run_manager_; }
    std::shared_ptr<anet::rl::gui::View> CreateExperinceView(wxWindow* parent);
	wxFrame* GetMainFrame() { return frame_; }
    
    std::filesystem::path GetRunDir();
    int64_t SaveAgent(const std::filesystem::path& file_path);
    void FlushRunOutputs();
    void ShutdownRunLogging();
    bool ShouldShowErrorDialog() const { return show_error_dialog_; }
private:
    void SetTrainingPaused(bool paused);
    void InitTrainer();
    void showFatalError();
    void SetupNonModalErrorLogging();
    void ShutdownNonModalErrorLogging();
    bool WriteLastRunName(const std::string& run_name) const;
    void SetupLogging();
    void FlushTextLog();
    void OnTextLogFlushTimer(wxTimerEvent& event);
private:
    std::unique_ptr<anet::ConfigManager> config_mgr_;
    struct Config;
    std::unique_ptr<Config> config_;
    
    std::shared_ptr<anet::rl::RunManager> run_manager_;
    std::unique_ptr<anet::rl::RunnerThread> trainer_thread_;
    std::unique_ptr<anet::rl::gui::DefaultViewFactory> view_factory_;
    std::unique_ptr<anet::rl::ImageProviderManager> img_prov_mgr_;
    anet::StandardStreamLogger standard_stream_logger_;
    std::unique_ptr<wxLogStderr> non_modal_log_target_;
    wxLog* previous_log_target_ = nullptr;
    wxTimer text_log_flush_timer_;
    wxLogChain* run_log_chain_ = nullptr;
    bool show_error_dialog_ = true;
    bool fatal_error_seen_ = false;
    bool auto_pause_done_ = false;
    RunnerFrame* frame_ = nullptr;
};

wxDECLARE_APP(RunnerApp);
