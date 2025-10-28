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

    ExperienceBatch ReplayBuffer::SampleBatch(size_t n, torch::Device device) const
    {
        auto exps = Sample(n); // ä˘ë∂ÇÃÉâÉìÉ_ÉÄÉTÉìÉvÉãéÊìæ

        std::vector<torch::Tensor> states;
        std::vector<torch::Tensor> next_states;
        std::vector<torch::Tensor> actions;
        std::vector<float> rewards;
        std::vector<bool> dones;

        states.reserve(n);
        next_states.reserve(n);
        actions.reserve(n);
        rewards.reserve(n);
        dones.reserve(n);

        for (auto& e : exps) {
            states.push_back(e.state);                         // (state_dim)
            next_states.push_back(e.response.next_state);      // (state_dim)
            actions.push_back(e.action);                       // (1)
            rewards.push_back(e.response.reward);              // float
            dones.push_back(e.response.done || e.response.truncated);
        }

        ExperienceBatch batch;

        batch.states = torch::stack(states).to(device);                      // (B, state_dim)
        batch.next_states = torch::stack(next_states).to(device);                // (B, state_dim)
        batch.actions = torch::stack(actions).to(device).to(torch::kLong).squeeze(-1); // (B,)
        batch.rewards = torch::tensor(rewards, torch::dtype(torch::kFloat32)).to(device); // (B,)

        // vector<bool> Å® vector<uint8_t> Ç…ïœä∑ÇµÇƒÇ©ÇÁ tensor âª
        std::vector<uint8_t> dones_u8;
        dones_u8.reserve(n);
        for (bool d : dones) dones_u8.push_back(d ? 1 : 0);
        batch.dones = torch::tensor(dones_u8, torch::kUInt8).to(device);

        return batch;
    }
} // namespace anet::rl
