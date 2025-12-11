#pragma once
#include <vector>
#include <mutex>
#include <wx/wx.h>
#include <wx/dcbuffer.h>

class PlotPanel : public wxPanel {
public:
    PlotPanel(wxWindow* parent);

    void AddData(float value);
private:
    void OnPaint(wxPaintEvent& event);
    void OnMouseLeftClick(wxMouseEvent& event);
    void OnMouseRightClick(wxMouseEvent& event);
    wxDECLARE_EVENT_TABLE();
private:
    std::vector<float> plot_data_;
    std::mutex plot_mutex_;
};
