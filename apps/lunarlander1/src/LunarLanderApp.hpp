// LunarLanderApp.hpp
#pragma once

#include <memory>
#include <optional>
#include <wx/app.h>

#include "anet/config.hpp"
#include "anet/trainer.hpp"
#include "UISnapshot.hpp"
#include "LunarLanderFrame.hpp"

#define WX_APP_COMPATIBLE

wxDECLARE_EVENT(wxEVT_TRAINER_EXIT, wxCommandEvent);
wxDECLARE_EVENT(wxEVT_APP_TRAINER_SHUTDOWN, wxThreadEvent);

class LunarLanderApp : public wxApp {
public:
    virtual bool OnInit() override;
    virtual int OnExit() override;

    anet::ConfigData GetConfig() const { return config_mgr_->GetConfigData(); }
    std::optional<UISnapshot> GetUISnapshot() { return snapshot_store_.Get(); }
    std::shared_ptr<anet::rl::DefaultTrainer> GetTrainer() { return trainer_; }

    //void StartTraining();
    void ToggleTraining();
    void StopTraining();
public:
    static UISnapshot CreateSnapshot(anet::rl::TrainEvent event);
private:
    void InitTrainer();
    void InitImageLogObservers();
    void InitPERImageLogObservers(const anet::ConfigData& config_data);
private:
    std::unique_ptr<anet::ConfigManager> config_mgr_;
    struct Config;
    std::unique_ptr<Config> config_;
    UISnapshotStore snapshot_store_;
    std::shared_ptr<anet::rl::DefaultTrainer> trainer_;
    std::unique_ptr<anet::rl::RunnerThread> trainer_thread_;
    bool auto_pause_done_ = false;
private:
    LunarLanderFrame* frame_;
};

wxDECLARE_APP(LunarLanderApp);
