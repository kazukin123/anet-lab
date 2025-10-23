#pragma once

#include <wx/wx.h>
#include "metrics_logger.hpp"

#define WX_APP_COMPATIBLE

class MyApp : public wxApp {
public:
    virtual bool OnInit();

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
    std::unique_ptr<MetricsLogger> mt_logger;

};

wxDECLARE_APP(MyApp);
