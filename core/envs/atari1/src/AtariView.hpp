#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <torch/torch.h>
#include <wx/glcanvas.h>
#include <wx/wx.h>

#include "anet/gui.hpp"
#include "anet/rl.hpp"

namespace anet::rl::env {

    struct AtariViewData {
        torch::Tensor rgb_hwc;
        torch::Tensor grid_hwc;
        float game_score = 0.0f;
        int64_t lives = 0;
        int64_t game_len = 0;
        int64_t game_frames = 0;
        std::string action_name = "-";
        float reward = 0.0f;
    };

    class AtariPanel final : public anet::rl::gui::Panel {
    public:
        explicit AtariPanel(wxWindow* parent);
        ~AtariPanel() override;

        void ApplyData(const AtariViewData& data);

    private:
        void OnPaint(wxPaintEvent& event);
        void OnSize(wxSizeEvent& event);
        void OnChildMouse(wxMouseEvent& event);
        void Render();
        void DrawTensor(const torch::Tensor& image, int x, int y, int width, int height);

        wxStaticText* overlay_ = nullptr;
        wxGLCanvas* canvas_ = nullptr;
        wxGLContext* context_ = nullptr;
        AtariViewData data_;
    };

    class AtariView final : public anet::rl::gui::ViewBase<AtariViewData, AtariPanel> {
    public:
        explicit AtariView(wxWindow* parent);
        AtariViewData CreateViewData(const anet::rl::TrainEvent& event) const override;
    };

    class AtariViewCreator final : public anet::rl::gui::ViewCreator {
    public:
        std::shared_ptr<anet::rl::gui::View> CreateView(
            wxWindow* parent,
            const anet::ConfigData& config_data,
            std::shared_ptr<anet::rl::Notifier> notifier) const override;

        std::string GetTargetClassId() const override { return "AtariEnv"; }
    };

} // namespace anet::rl::env
