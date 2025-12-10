#pragma once

#include <torch/torch.h>
#include <wx/wx.h>
#include "anet/rl.hpp"
#include "UISnapshot.hpp"

class CartPoleCanvas : public wxPanel {
public:
    CartPoleCanvas(wxWindow* parent);

    // UIデータ設定
    void SetUIData(const UISnapshot& snapshot);

protected:
    void OnPaint(wxPaintEvent& event);
    void OnMouseClick(wxMouseEvent& event);
private:
    anet::rl::step_t step_;

    float cart_x_;
    float cart_x_dot_;
    float pole_theta_;
    float pole_theta_dot_;

    float reward_ = 0.0f;

    int64_t action_ = 0;

    // 表示スケールなど
    float cart_scale_;
    float pole_length_;

    wxDECLARE_EVENT_TABLE();
};
