// GridMazeView.hpp
#pragma once

#include <torch/torch.h>
#include <wx/wx.h>
#include "anet/rl.hpp"
#include "anet/gui.hpp"

namespace anet::rl::env {

    struct GridMazeData {
        anet::rl::StepCounts counts;
        anet::rl::BatchExperience train_exp;
        int64_t current_seed = 0;
        int map_width = 5;
        int map_height = 5;
        int pos_x = 0;
        int pos_y = 0;
        float ep_reward_sum = 0.0f;
        torch::Tensor map_tensor;
    };

    class GridMazePanel : public anet::rl::gui::Panel {
    public:
        explicit GridMazePanel(wxWindow* parent);
        void ApplyData(const GridMazeData& data);
    protected:
        void OnPaint(wxPaintEvent& event);
    private:
        GridMazeData data_;
    };

    class GridMazeView final : public anet::rl::gui::ViewBase<GridMazeData, GridMazePanel> {
    public:
        explicit GridMazeView(wxWindow* parent);
        GridMazeData CreateViewData(const anet::rl::TrainEvent& event) const override;
    };

    class GridMazeViewCreator final : public anet::rl::gui::ViewCreator {
    public:
        GridMazeViewCreator() = default;
        std::shared_ptr<anet::rl::gui::View> CreateView(
            wxWindow* parent, const anet::ConfigData& config_data, std::shared_ptr<anet::rl::Notifier> notifier) const override
        {
            return std::make_shared<GridMazeView>(parent);
        }
        std::string GetTargetClassId() const override { return "GridMazeEnv"; }
    };

} // namespace anet::rl::env