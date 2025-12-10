#include "PlotPanel.hpp"
#include <algorithm>

#include "CartPoleFrame.hpp"
#include "app.hpp"

wxBEGIN_EVENT_TABLE(PlotPanel, wxPanel)
EVT_PAINT(PlotPanel::OnPaint)
EVT_LEFT_DOWN(PlotPanel::OnMouseClick)
wxEND_EVENT_TABLE()

PlotPanel::PlotPanel(wxWindow* parent)
    : wxPanel(parent, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxBORDER_SIMPLE)
{
    SetBackgroundStyle(wxBG_STYLE_PAINT);
}

void PlotPanel::OnMouseClick(wxMouseEvent& event)
{
    //event.Skip();
    wxMouseEvent evt = event;
    GetParent()->GetEventHandler()->ProcessEvent(evt);
}

void PlotPanel::AddData(float value)
{
    std::lock_guard<std::mutex> lock(plot_mutex_);

    plot_data_.push_back(value);
    if (plot_data_.size() > 1000) // 最新1000点だけ保持
        plot_data_.erase(plot_data_.begin());
    //Refresh(false);            // 再描画要求（即時ではない）
}

void PlotPanel::OnPaint(wxPaintEvent&)
{
    wxAutoBufferedPaintDC dc(this);
    dc.Clear();

    std::vector<float> plot_data;
    {
        std::lock_guard<std::mutex> lock(plot_mutex_);
        plot_data = plot_data_; // UI描画用にコピー
    }

    if (plot_data.empty()) {
        dc.DrawText("No data yet...", 10, 10);
        return;
    }

    wxSize sz = GetClientSize();
    float w = static_cast<float>(sz.GetWidth());
    float h = static_cast<float>(sz.GetHeight() - 50);

    // スケーリング
    float max_val = *std::max_element(plot_data.begin(), plot_data.end());
    float min_val = *std::min_element(plot_data.begin(), plot_data.end());
    if (max_val == min_val) { max_val += 1.0f; min_val -= 1.0f; }

    dc.SetPen(wxPen(*wxBLUE, 2));

    const int n = (int)plot_data.size();
    for (int i = 1; i < n; i++) {
        float x1 = (w / (n - 1)) * (i - 1);
        float y1 = 40 + h - (plot_data[i - 1] - min_val) / (max_val - min_val) * h;
        float x2 = (w / (n - 1)) * i;
        float y2 = 40 + h - (plot_data[i] - min_val) / (max_val - min_val) * h;
        dc.DrawLine(wxPoint(x1, y1), wxPoint(x2, y2));
    }

    // 軸線
    dc.SetPen(wxPen(*wxBLACK, 1, wxPENSTYLE_DOT));
    dc.DrawLine(0, h / 2, w, h / 2);

    // 最新値
    wxString txt;
    txt.Printf("Latest: %.3f", plot_data.back());
    dc.DrawText(txt, 10, 10);
    txt.Printf("Min: %.3f", min_val);
    dc.DrawText(txt, 110, 10);
    txt.Printf("Max: %.3f", max_val);
    dc.DrawText(txt, 210, 10);
}
