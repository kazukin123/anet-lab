#include "anet/rl.hpp"
#include <stdexcept>

namespace anet::rl {

    std::unique_ptr<HeatMap> MakeStateHeatMapPtr(
        const anet::rl::StateSpaceInfo& info,
        int idx_x,
        int idx_y,
        int width,
        int height,
        size_t max_points,
        uint32_t flags)
    {
        if (!info.low.defined() || !info.high.defined())
            throw std::runtime_error("StateSpaceInfo.low/high are undefined.");

        auto dim = info.low.size(0);
        if (idx_x >= dim || idx_y >= dim)
            throw std::runtime_error("MakeStateHeatMapPtr: axis index out of range.");

        float x_min = info.low[idx_x].item<float>();
        float x_max = info.high[idx_x].item<float>();
        float y_min = info.low[idx_y].item<float>();
        float y_max = info.high[idx_y].item<float>();

        return std::make_unique<HeatMap>(
            width,
            height,
            x_min, x_max,
            y_min, y_max,
            max_points,
            flags);
    }

    std::unique_ptr<TimeHeatMap> MakeStateTimeHeatMapPtr(
        const anet::rl::StateSpaceInfo& info,
        int idx_x,
        int width, int height,
        size_t max_points,
        uint32_t flags,
        TimeFrameMode mode)
    {
        if (!info.low.defined() || !info.high.defined())
            throw std::runtime_error("StateSpaceInfo.low/high are undefined.");

        auto dim = info.low.size(0);
        if (idx_x >= dim)
            throw std::runtime_error("MakeStateHeatMapPtr: axis index out of range.");

        float x_min = info.low[idx_x].item<float>();
        float x_max = info.high[idx_x].item<float>();

        return std::make_unique<TimeHeatMap>(
            width,
            height,
            x_min, x_max,
            flags,
            max_points,
            mode);
    }

} // namespace anet::rl
