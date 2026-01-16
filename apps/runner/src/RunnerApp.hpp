// RunnerApp.hpp

#pragma once

#include <memory>
#include <optional>
#include <wx/app.h>
#include "anet/config.hpp"
#include "anet/trainer.hpp"
#include "anet/gui.hpp"
#include "anet/image.hpp"

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
    bool OnInit() override;
    int OnExit() override;
    bool OnExceptionInMainLoop() override;
    void OnUnhandledException() override;
public:
    void ToggleTraining();
    void StopTraining();
    anet::ConfigData GetConfigData() const { return config_mgr_->GetConfigData(); }
    std::shared_ptr<anet::rl::DefaultTrainer> GetTrainer() { return trainer_; }
    std::shared_ptr<anet::rl::gui::View> CreateExperinceView(wxWindow* parent);
	wxFrame* GetMainFrame() { return frame_; }
private:
    void InitTrainer();
    //void InitImageLogObservers();
    //void InitPERImageLogObservers(const anet::ConfigData& config_data);
    void showFatalError();
private:
    std::unique_ptr<anet::ConfigManager> config_mgr_;
    struct Config;
    std::unique_ptr<Config> config_;
    std::shared_ptr<anet::rl::DefaultTrainer> trainer_;
    std::unique_ptr<anet::rl::RunnerThread> trainer_thread_;
    std::unique_ptr<anet::rl::gui::DefaultViewFactory> view_factory_;
    std::unique_ptr<anet::rl::ImageProviderManager> img_prov_mgr_;
    bool auto_pause_done_ = false;
    RunnerFrame* frame_;
};

wxDECLARE_APP(RunnerApp);
