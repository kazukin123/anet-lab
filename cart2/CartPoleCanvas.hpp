#pragma once
#include <wx/wx.h>

class CartPoleCanvas : public wxPanel {
public:
    CartPoleCanvas(wxWindow* parent);

    // カート位置・角度をセット
    void SetState(float x, float theta);

protected:
    void OnPaint(wxPaintEvent& event);

private:
    float cart_x;
    float pole_theta;

    // 表示スケールなど
    float cart_scale;
    float pole_length;

    wxDECLARE_EVENT_TABLE();
};
