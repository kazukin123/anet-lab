#include "CartPoleCanvas.hpp"
#include <wx/dcbuffer.h>
#include <cmath>

wxBEGIN_EVENT_TABLE(CartPoleCanvas, wxPanel)
EVT_PAINT(CartPoleCanvas::OnPaint)
wxEND_EVENT_TABLE()

CartPoleCanvas::CartPoleCanvas(wxWindow* parent)
    : wxPanel(parent, wxID_ANY, wxDefaultPosition, wxSize(800, 400)),
    cart_x(0.0f),
    pole_theta(0.0f),
    cart_scale(80.0f),    // x=1.0 の時の画面スケール
    pole_length(120.0f)   // 棒のピクセル長
{
    SetBackgroundStyle(wxBG_STYLE_PAINT);
}

void CartPoleCanvas::SetState(float x, float theta) {
    cart_x = x;
    pole_theta = theta;
    Refresh();
}

void CartPoleCanvas::OnPaint(wxPaintEvent&) {
    wxAutoBufferedPaintDC dc(this);
    wxSize sz = GetClientSize();

    dc.SetBackground(*wxWHITE_BRUSH);
    dc.Clear();

    // 中央を原点に変換
    int centerX = sz.x / 2;
    int baseY = sz.y - 120;

    // カート位置
    int cart_px = centerX + static_cast<int>(cart_x * cart_scale);

    // カート描画
    wxRect cartRect(cart_px - 25, baseY - 20, 50, 20);
    dc.SetBrush(*wxBLUE_BRUSH);
    dc.DrawRectangle(cartRect);

    // 棒の先端
    float sin_t = std::sin(pole_theta);
    float cos_t = std::cos(pole_theta);
    int x2 = cart_px + static_cast<int>(pole_length * sin_t);
    int y2 = baseY - 20 - static_cast<int>(pole_length * cos_t);

    // 棒描画
    dc.SetPen(wxPen(*wxRED, 4));
    dc.DrawLine(cart_px, baseY - 20, x2, y2);

    // 軸線
    dc.SetPen(*wxLIGHT_GREY_PEN);
    dc.DrawLine(0, baseY, sz.x, baseY);
}
