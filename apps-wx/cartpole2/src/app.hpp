#pragma once

#include <wx/wx.h>
#include <wx/cmdline.h>

#include "anet/config.hpp"
#include "anet/metrics_logger.hpp"

#define WX_APP_COMPATIBLE

class MyApp : public wxApp {
public:
    virtual bool OnInit() override;
    virtual int OnExit() override;

    const anet::ConfigData& GetConfig() const { return *config_data_; }
    wxCmdLineParser* GetCmdLineParser() const { return cmdline_.get(); }
private:
    std::unique_ptr<wxCmdLineParser> cmdline_;
    std::unique_ptr<anet::ConfigData> config_data_;
};

wxDECLARE_APP(MyApp);
