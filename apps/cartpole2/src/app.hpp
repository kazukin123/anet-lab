// app.h
#pragma once

#include <memory>
#include <optional>
#include <wx/app.h>

#include "anet/config.hpp"
#include "anet/trainer.hpp"
#include "UISnapshot.hpp"
#include "CartPoleFrame.hpp"

#define WX_APP_COMPATIBLE

wxDECLARE_EVENT(wxEVT_TRAINER_EXIT, wxCommandEvent);
wxDECLARE_EVENT(wxEVT_APP_TRAINER_SHUTDOWN, wxThreadEvent);

class CartPoleApp : public wxApp {
public:
    virtual bool OnInit() override;
    virtual int OnExit() override;

    anet::ConfigData GetConfig() const { return config_mgr_->GetConfigData(); }
    std::optional<UISnapshot> GetUISnapshot() { return snapshot_store_.Get(); }
    
    //void StartTraining();
    void ToggleTraining();
    void StopTraining();
private:
    void InitTrainer();
    void InitImageLogObservers(const anet::ConfigData& config_data);
private:
    std::unique_ptr<anet::ConfigManager> config_mgr_;
    struct Config;
    std::unique_ptr<Config> config_;
    UISnapshotStore snapshot_store_;
    std::shared_ptr<anet::rl::Runner> trainer_;
    std::unique_ptr<anet::rl::RunnerThread> trainer_thread_;
    bool auto_pause_done_ = false;
private:
    CartPoleFrame* frame_;
};

wxDECLARE_APP(CartPoleApp);
