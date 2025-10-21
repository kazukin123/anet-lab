#include <wx/dcbuffer.h>
#include <cmath>
#include "CartPoleCanvas.hpp"
#include "CartPoleFrame.hpp"

#define _USE_MATH_DEFINES // for C++
#include <cmath>

wxBEGIN_EVENT_TABLE(CartPoleCanvas, wxPanel)
EVT_PAINT(CartPoleCanvas::OnPaint)
EVT_LEFT_DOWN(CartPoleCanvas::OnMouseClick)
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

void CartPoleCanvas::OnMouseClick(wxMouseEvent& event) {
    auto* frame = dynamic_cast<CartPoleFrame*>(wxGetTopLevelParent(this));
    if (frame) frame->ToggleTraining();
}

void CartPoleCanvas::SetState(float x, float theta, float x_dot, float theta_dot) {
    cart_x = x;
    pole_theta = theta;
    cart_x_dot = x_dot;
    pole_theta_dot = theta_dot;

    Refresh();
}

void CartPoleCanvas::SetAction(const torch::Tensor& action) {
    this->action_ = action;

//    wxLogInfo("action=%s", action.toString());
}

void CartPoleCanvas::OnPaint(wxPaintEvent& event) {
    wxAutoBufferedPaintDC dc(this);
    dc.Clear();

    const wxSize size = GetClientSize();
    const int width = size.GetWidth();
    const int height = size.GetHeight();
    const float scale = width / 8.0f;
    const int groundY = height / 2;

    // 床線
    dc.SetPen(*wxBLACK_PEN);
    dc.DrawLine(0, groundY, width, groundY);

    // x_limit表示
    dc.SetPen(wxPen(wxColour(180, 180, 180), 1, wxPENSTYLE_DOT));
    int leftX = width / 2 + static_cast<int>(-2.4f * scale);
    int rightX = width / 2 + static_cast<int>(2.4f * scale);
    dc.DrawLine(leftX, 0, leftX, height);
    dc.DrawLine(rightX, 0, rightX, height);

    // カート位置
    float cartX = width / 2 + static_cast<int>(this->cart_x * scale);
    float cartY = groundY;
    float cartWidth = 50;
    float cartHeight = 20;

    // カート
    dc.SetBrush(*wxBLUE_BRUSH);
    dc.DrawRectangle(cartX - cartWidth / 2, cartY - cartHeight / 2, cartWidth, cartHeight);

    // ポール
    float poleLength = 100;
    float angle = -this->pole_theta;
    int poleX = cartX + static_cast<int>(std::sin(angle) * poleLength);
    int poleY = cartY - static_cast<int>(std::cos(angle) * poleLength);

    dc.SetPen(wxPen(wxColour(255, 128, 0), 3));
    dc.DrawLine(cartX, cartY - cartHeight / 2, poleX, poleY);

    // === 報酬バー ===
    // reward ∈ [0, 2] 程度を想定して正規化
    float clamped_reward = std::max(0.0f, std::min(reward, 2.0f));
    int bar_width = static_cast<int>((width / 3) * (clamped_reward / 2.0f));
    int bar_height = 12;
    int bar_x = 100;
    int bar_y = height - bar_height - 10;
    dc.SetPen(*wxTRANSPARENT_PEN);
    dc.SetBrush(wxBrush(wxColour(0, 220, 0)));  // 緑
    dc.DrawRectangle(bar_x, bar_y, bar_width, bar_height);

    // 報酬文字
    dc.SetTextForeground(*wxBLACK);
    dc.DrawText(wxString::Format("Reward: %.2f", reward), 10, bar_y - 2);// bar_x + bar_width + 8, bar_y - 2);

    // state文字
    dc.DrawText(wxString::Format("X = %.2f", this->cart_x), 10, 10);
    dc.DrawText(wxString::Format("θ = %.2f°", this->pole_theta * 180/ M_PI), 10, 30);
    dc.DrawText(wxString::Format("dotX = %.2f", this->cart_x_dot), 10, 50);
    dc.DrawText(wxString::Format("dotθ = %.2f", this->pole_theta_dot * 180 / M_PI), 10, 70);

    // 力の方向ベクトルを描画
    if (action_.defined()) {
        int act = action_.item<int>();
        float arrowLen = 40.0f;
        wxPoint start(cartX, cartY + cartHeight / 2 + 5);
        wxPoint end(cartX + (act == 1 ? arrowLen : -arrowLen), cartY + cartHeight / 2 + 5);

        dc.SetPen(wxPen(wxColour(255, 0, 0), 3));
        dc.DrawLine(start, end);

        // 矢印ヘッド
        int dir = (act == 1) ? 1 : -1;
        dc.DrawLine(end, wxPoint(end.x - dir * 8, end.y - 5));
        dc.DrawLine(end, wxPoint(end.x - dir * 8, end.y + 5));
    }
}


