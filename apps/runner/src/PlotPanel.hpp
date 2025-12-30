// PlotPanel.hpp

#pragma once

#include <vector>
#include <mutex>
#include <wx/wx.h>
#include "anet/gui.hpp"

class PlotPanel : public anet::rl::gui::Panel {
public:
    PlotPanel(wxWindow* parent);

    void AddData(float value);
private:
    void OnPaint(wxPaintEvent& event);

    wxDECLARE_EVENT_TABLE();
private:
    std::vector<float> plot_data_;
    std::mutex plot_mutex_;
};
