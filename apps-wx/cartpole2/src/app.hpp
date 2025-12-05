// app.h
#pragma once

#include <wx/app.h>

#include "anet/config.hpp"

#define WX_APP_COMPATIBLE

class MyApp : public wxApp {
public:
    virtual bool OnInit() override;
    virtual int OnExit() override;

    anet::ConfigData GetConfig() const { return config_mgr_->GetConfigData(); }
private:
    std::unique_ptr<anet::ConfigManager> config_mgr_;
};

wxDECLARE_APP(MyApp);
