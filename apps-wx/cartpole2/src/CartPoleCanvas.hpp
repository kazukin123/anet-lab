#pragma once

#include <torch/torch.h>

#include <wx/wx.h>
#include "anet/rl.hpp"

class CartPoleCanvas : public wxPanel {
public:
    CartPoleCanvas(wxWindow* parent);

    // カート位置・角度をセット
    void SetState(const anet::rl::BatchState& state, anet::rl::StateSpec spec);
    void SetAction(const torch::Tensor& action);
    void SetReward(float reward) { this->reward = reward; }

protected:
    void OnPaint(wxPaintEvent& event);
    void OnMouseClick(wxMouseEvent& event);

private:
    float cart_x;
    float cart_x_dot;
    float pole_theta;
    float pole_theta_dot;

    float reward;

    torch::Tensor action_;

    // 表示スケールなど
    float cart_scale;
    float pole_length;

    wxDECLARE_EVENT_TABLE();
};
