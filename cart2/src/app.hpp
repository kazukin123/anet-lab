#pragma once

#include <wx/wx.h>
#include <wx/cmdline.h>

#include "properties.hpp"
#include "metrics_logger.hpp"

#define WX_APP_COMPATIBLE

class MyApp : public wxApp {
public:
    virtual bool OnInit();

    Properties* GetConfig() const { return properties_.get(); }
    wxCmdLineParser* GetCommandLine() const { return cmdline_.get(); }
    MetricsLogger* GetMetricsLogger() const { return mt_logger.get(); }

    void logScalar(const std::string& tag, int step, double value) {
        mt_logger->log_scalar(tag, step, value);
    }
    void logJson(const std::string& tag, const nlohmann::json& data) {
        mt_logger->log_json(tag, data);
    }
    void flushMetricsLog() {
        mt_logger->flush();
    }
private:
    std::unique_ptr<wxCmdLineParser> cmdline_;
    std::unique_ptr<Properties> properties_;
    std::unique_ptr<MetricsLogger> mt_logger;
};

wxDECLARE_APP(MyApp);
