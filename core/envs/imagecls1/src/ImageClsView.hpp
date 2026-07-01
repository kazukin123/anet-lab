// ImageClsView.hpp
#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>
#include <wx/wx.h>
#include "anet/rl.hpp"
#include "anet/gui.hpp"

namespace anet::rl::env {

    struct ImageClsOutputValue {
        int64_t rank = -1;
        int64_t label_id = -1;
        std::string label_name;
        float probability = 0.0f;
        bool is_true = false;
        bool is_predicted = false;
    };

    struct ImageClsData {
        anet::rl::StepCounts counts;
        torch::Tensor image_tensor;     // [C, H, W]
        int64_t true_label = -1;
        int64_t predicted_label = -1;
        std::string true_label_name;
        std::string predicted_label_name;
        std::vector<ImageClsOutputValue> top_outputs;
        bool is_valid = false;
    };

    class ImageClsPanel : public anet::rl::gui::Panel {
    public:
        explicit ImageClsPanel(wxWindow* parent);
        void ApplyData(const ImageClsData& data);
    protected:
        void OnPaint(wxPaintEvent& event);
    private:
        void UpdateBitmap();

        ImageClsData data_;
        wxBitmap current_bitmap_;
    };

    class ImageClsView final : public anet::rl::gui::ViewBase<ImageClsData, ImageClsPanel> {
    public:
        explicit ImageClsView(wxWindow* parent);
        ImageClsData CreateViewData(const anet::rl::TrainEvent& event) const override;
    };

    class ImageClsViewCreator final : public anet::rl::gui::ViewCreator {
    public:
        ImageClsViewCreator() = default;

        // 引数を最新のインターフェースに合わせる
        std::shared_ptr<anet::rl::gui::View> CreateView(
            wxWindow* parent,
            const anet::ConfigData& config_data,
            std::shared_ptr<anet::rl::Notifier> notifier) const override;

        // メソッド名を GetTargetClassId に変更
        std::string GetTargetClassId() const override { return "ImageClsEnv"; }
    };

} // namespace anet::rl::env
