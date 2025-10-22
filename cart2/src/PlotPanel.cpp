#include "PlotPanel.hpp"
#include <algorithm>

#include "CartPoleFrame.hpp"

wxBEGIN_EVENT_TABLE(PlotPanel, wxPanel)
EVT_PAINT(PlotPanel::OnPaint)
EVT_LEFT_DOWN(PlotPanel::OnMouseClick)
wxEND_EVENT_TABLE()

PlotPanel::PlotPanel(wxWindow* parent)
    : wxPanel(parent, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxBORDER_SIMPLE)
{
    SetBackgroundStyle(wxBG_STYLE_PAINT); // ← ダブルバッファ描画有効化
}

void PlotPanel::OnMouseClick(wxMouseEvent& event) {
    auto* frame = dynamic_cast<CartPoleFrame*>(wxGetTopLevelParent(this));
    if (frame) frame->ToggleTraining();
}

void PlotPanel::AddReward(float reward) {
    rewards.push_back(reward);
    if (rewards.size() > 1000)  // 最新1000点だけ保持
        rewards.erase(rewards.begin());
    Refresh(false);            // 再描画要求（即時ではない）
}

void PlotPanel::OnPaint(wxPaintEvent&) {
    wxAutoBufferedPaintDC dc(this);
    dc.Clear();

    if (rewards.empty()) {
        dc.DrawText("No data yet...", 10, 10);
        return;
    }

    wxSize sz = GetClientSize();
    float w = static_cast<float>(sz.GetWidth());
    float h = static_cast<float>(sz.GetHeight());

    // スケーリング
    float max_r = *std::max_element(rewards.begin(), rewards.end());
    float min_r = *std::min_element(rewards.begin(), rewards.end());
    if (max_r == min_r) { max_r += 1.0f; min_r -= 1.0f; }

    dc.SetPen(wxPen(*wxBLUE, 2));

    const int n = (int)rewards.size();
    for (int i = 1; i < n; i++) {
        float x1 = (w / (n - 1)) * (i - 1);
        float y1 = h - (rewards[i - 1] - min_r) / (max_r - min_r) * h;
        float x2 = (w / (n - 1)) * i;
        float y2 = h - (rewards[i] - min_r) / (max_r - min_r) * h;
        dc.DrawLine(wxPoint(x1, y1), wxPoint(x2, y2));
    }

    // 軸線
    dc.SetPen(wxPen(*wxBLACK, 1, wxPENSTYLE_DOT));
    dc.DrawLine(0, h / 2, w, h / 2);

    // 最新値
    wxString txt;
    txt.Printf("Latest: %.3f", rewards.back());
    dc.DrawText(txt, 10, 10);
}
