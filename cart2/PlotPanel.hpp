#pragma once
#include <wx/wx.h>
#include <wx/dcbuffer.h>
#include <vector>

class PlotPanel : public wxPanel {
public:
    PlotPanel(wxWindow* parent);

    void OnMouseClick(wxMouseEvent& event);

    void AddReward(float reward);
private:
    std::vector<float> rewards;
    void OnPaint(wxPaintEvent& event);

    wxDECLARE_EVENT_TABLE();
};
