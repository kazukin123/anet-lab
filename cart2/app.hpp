#pragma once

#include <wx/wx.h>
#include "CartPoleFrame.hpp"
#include "tb_logger.hpp"

class MyApp : public wxApp {
public:
    virtual bool OnInit();

    void logScalar(const std::string& tag, double value, int64_t step) {
        tb_logger->scalar(tag, value, step);
    }
    void flushLog() {
        tb_logger->flush();
    }
private:
    std::unique_ptr<simplelog::JsonlLogger> tb_logger;

};

wxDECLARE_APP(MyApp);
