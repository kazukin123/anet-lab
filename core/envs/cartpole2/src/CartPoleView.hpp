#pragma once

#include <torch/torch.h>
#include <wx/wx.h>
#include "anet/rl.hpp"
#include "anet/gui.hpp"


// =============================================================
// CartPoleData
// =============================================================

struct CartPoleData {
    anet::rl::StepCounts counts;                ///< 現在のステップ状況
    anet::rl::BatchExperience train_exp;        ///< Experience（描画対象）
};

// =============================================================
// CartPolePanel
// =============================================================

class CartPolePanel : public anet::rl::gui::Panel {
public:
    explicit CartPolePanel(wxWindow* parent);

    // UIデータ設定
    void ApplyData(const CartPoleData& data);
protected:
    void OnPaint(wxPaintEvent& event);
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
};


// =============================================================
// CartPoleView
// =============================================================

class CartPoleView final : public anet::rl::gui::ViewBase<CartPoleData, CartPolePanel> {
public:
    explicit CartPoleView(wxWindow* parent);
    CartPoleData CreateViewData(const anet::rl::TrainEvent& event) const override;
};


// =============================================================
// CartPoleViewCreator
// =============================================================

class CartPoleViewCreator final : public anet::rl::gui::ViewCreator {
public:
    CartPoleViewCreator() = default;

    std::shared_ptr<anet::rl::gui::View> CreateView(
        wxWindow* parent, const anet::ConfigData& config_data, std::shared_ptr<anet::rl::Notifier> notifier) const
    {
        return std::make_shared<CartPoleView>(parent);
    }

    std::string GetTargetClassId() const override
    {
        return "CartPoleEnv";
    }
};