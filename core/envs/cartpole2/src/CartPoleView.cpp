// CartPoleView.cpp

#include "CartPoleView.hpp"

#include <cmath>
#include <wx/dcbuffer.h>
#include "anet/str_util.hpp"


// =============================================================
// CartPolePanel
// =============================================================

CartPolePanel::CartPolePanel(wxWindow* parent)
    : anet::rl::gui::Panel(parent)
    , cart_x_(0.0f)
    , cart_x_dot_(0.0f)
    , pole_theta_(0.0f)
    , pole_theta_dot_(0.0f)
    , cart_scale_(80.0f)    // x=1.0 の時の画面スケール
    , pole_length_(120.0f)   // 棒のピクセル長
{
    SetBackgroundStyle(wxBG_STYLE_PAINT);
    Bind(wxEVT_PAINT, &CartPolePanel::OnPaint, this);
}

void CartPolePanel::ApplyData(const CartPoleData& data)
{
    const int BATCH_POS = 0;

    // step
    step_ = data.counts.exp_step;

    // state
    auto exp = data.train_exp;
    torch::Tensor obs = exp.state.Flatten().obs[BATCH_POS];
    cart_x_ = obs[0].item<float>();
	cart_x_dot_ = obs[1].item<float>();
    pole_theta_ = obs[2].item<float>();
	pole_theta_dot_ = obs[3].item<float>();

    // action
    action_ = exp.action->GetAction(torch::kCPU)[BATCH_POS].item<int64_t>();

    // reward
    reward_ = exp.reward[BATCH_POS].item<float>();
}

void CartPolePanel::OnPaint(wxPaintEvent& event)
{
    wxAutoBufferedPaintDC dc(this);
    dc.Clear();

    const wxSize size = GetClientSize();
    const int width = size.GetWidth();
    const int height = size.GetHeight();
    const float scale = width / 8.0f;
    const int groundY = height / 2;

    // 床線
    //dc.SetPen(*wxBLACK_PEN);
    dc.DrawLine(0, groundY, width, groundY);

    // x_limit表示
    dc.SetPen(wxPen(wxColour(180, 180, 180), 1, wxPENSTYLE_DOT));
    int leftX = width / 2 + static_cast<int>(-2.4f * scale);
    int rightX = width / 2 + static_cast<int>(2.4f * scale);
    dc.DrawLine(leftX, 0, leftX, height);
    dc.DrawLine(rightX, 0, rightX, height);

    // カート位置
    float cartX = width / 2 + static_cast<int>(this->cart_x_ * scale);
    float cartY = groundY;
    float cartWidth = 50;
    float cartHeight = 20;

    // カート
    dc.SetBrush(*wxBLUE_BRUSH);
    dc.DrawRectangle(cartX - cartWidth / 2, cartY - cartHeight / 2, cartWidth, cartHeight);

    // ポール
    float poleLength = 100;
    float angle = -this->pole_theta_;
    int poleX = cartX + static_cast<int>(std::sin(angle) * poleLength);
    int poleY = cartY - static_cast<int>(std::cos(angle) * poleLength);

    dc.SetPen(wxPen(wxColour(255, 128, 0), 3));
    dc.DrawLine(cartX, cartY - cartHeight / 2, poleX, poleY);

    // === 報酬バー ===
    // reward ∈ [0, 2] 程度を想定して正規化
    float clamped_reward = std::max(0.0f, std::min(reward_, 2.0f));
    int bar_width = static_cast<int>((width / 3) * (clamped_reward / 2.0f));
    int bar_height = 12;
    int bar_x = 100;
    int bar_y = height - bar_height - 10;
    dc.SetPen(*wxTRANSPARENT_PEN);
    dc.SetBrush(wxBrush(wxColour(0, 220, 0)));  // 緑
    dc.DrawRectangle(bar_x, bar_y, bar_width, bar_height);

    // 報酬文字
    //dc.SetTextForeground(*wxBLACK);
    dc.DrawText(wxString::Format("Reward: %.2f", reward_), 10, bar_y - 2);// bar_x + bar_width + 8, bar_y - 2);

    // Step
    dc.DrawText(wxString::Format("Step: %s", anet::FormatWithCommas(step_)), 10, 10);

    // state文字
    dc.DrawText(wxString::Format("X = %.2f", this->cart_x_), 10, 40);
    dc.DrawText(wxString::Format("theta = %.2f(%.2f)", this->pole_theta_, this->pole_theta_ * 180/ M_PI), 10, 60);
    dc.DrawText(wxString::Format("dotX = %.2f", this->cart_x_dot_), 10, 80);
    dc.DrawText(wxString::Format("dotTheta = %.2f(%.2f)", this->pole_theta_dot_, this->pole_theta_dot_ * 180 / M_PI), 10, 100);

    // 力の方向ベクトルを描画
    float arrowLen = 40.0f;
    wxPoint start(cartX, cartY + cartHeight / 2 + 5);
    wxPoint end(cartX + (action_ == 1 ? -arrowLen : arrowLen), cartY + cartHeight / 2 + 5);

    dc.SetPen(wxPen(wxColour(255, 0, 0), 3));
    dc.DrawLine(start, end);

    // 矢印ヘッド
    int dir = (action_ == 1) ? -1 : 1;
    dc.DrawLine(end, wxPoint(end.x - dir * 8, end.y - 5));
    dc.DrawLine(end, wxPoint(end.x - dir * 8, end.y + 5));
}


// =============================================================
// CartPoleView
// =============================================================

CartPoleView::CartPoleView(wxWindow* parent)
    : ViewBaseType(parent)
{
    window_ = new CartPolePanel(parent_);
}

CartPoleData CartPoleView::CreateViewData(const anet::rl::TrainEvent& event) const
{
    return { event.counts, event.experience };
}
