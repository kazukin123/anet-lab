#pragma once
#include <vector>
#include <wx/wx.h>
#include <wx/dcbuffer.h>

class PlotPanel : public wxPanel {
public:
    PlotPanel(wxWindow* parent);

    void OnMouseClick(wxMouseEvent& event);

    void AddData(float value);
private:
    void OnPaint(wxPaintEvent& event);
    wxDECLARE_EVENT_TABLE();
private:
    std::vector<float> plot_data_;
    std::mutex plot_mutex_;
};
