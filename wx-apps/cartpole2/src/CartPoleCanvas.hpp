#pragma once

#include <torch/torch.h>

#include <wx/wx.h>

class CartPoleCanvas : public wxPanel {
public:
    CartPoleCanvas(wxWindow* parent);

    // カート位置・角度をセット
    void SetState(float x, float theta, float x_dot, float theta_dot);
    void SetAction(const torch::Tensor& action);
    void SetReward(float reward) { this->reward = reward; }

protected:
    void OnPaint(wxPaintEvent& event);
    void OnMouseClick(wxMouseEvent& event);

private:
    float cart_x;
    float pole_theta;
    float cart_x_dot;
    float pole_theta_dot;

    float reward;

    torch::Tensor action_;

    // 表示スケールなど
    float cart_scale;
    float pole_length;

    wxDECLARE_EVENT_TABLE();
};
