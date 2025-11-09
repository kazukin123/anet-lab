#pragma once

#include <wx/wx.h>
#include <wx/cmdline.h>

#include "anet/properties.hpp"
#include "anet/metrics_logger.hpp"

#define WX_APP_COMPATIBLE

class MyApp : public wxApp {
public:
    virtual bool OnInit() override;
    virtual int OnExit() override;

    anet::Properties* GetConfig() const { return properties_.get(); }
    wxCmdLineParser* GetCmdLineParser() const { return cmdline_.get(); }
private:
    std::unique_ptr<wxCmdLineParser> cmdline_;
    std::unique_ptr<anet::Properties> properties_;
};

wxDECLARE_APP(MyApp);
