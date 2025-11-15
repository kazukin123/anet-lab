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

    void ReplayBuffer::Push(const Experience& e)
    {
        if (!initialized_) {
            device_ = e.state.device();

            auto state_sizes = e.state.sizes();
            auto action_sizes = e.action.sizes();

            std::vector<int64_t> state_shape(1 + state_sizes.size());
            state_shape[0] = static_cast<int64_t>(capacity_);
            for (size_t i = 0; i < state_sizes.size(); ++i)
                state_shape[i + 1] = state_sizes[i];

            std::vector<int64_t> action_shape(1 + action_sizes.size());
            action_shape[0] = static_cast<int64_t>(capacity_);
            for (size_t i = 0; i < action_sizes.size(); ++i)
                action_shape[i + 1] = action_sizes[i];

            states_ = torch::empty(state_shape, e.state.options());
            actions_ = torch::empty(action_shape, e.action.options());
            next_states_ = torch::empty(state_shape, e.response.next_state.options());

            rewards_ = torch::empty({ static_cast<int64_t>(capacity_) },
                torch::TensorOptions().dtype(torch::kFloat32).device(device_));
            dones_ = torch::empty({ static_cast<int64_t>(capacity_) },
                torch::TensorOptions().dtype(torch::kFloat32).device(device_));
            truncateds_ = torch::empty({ static_cast<int64_t>(capacity_) },
                torch::TensorOptions().dtype(torch::kFloat32).device(device_));

            initialized_ = true;
        }

        states_[write_index_] = e.state;
        actions_[write_index_] = e.action;
        next_states_[write_index_] = e.response.next_state;

        rewards_[write_index_] = e.response.reward;
        dones_[write_index_] = e.response.done ? 1.0f : 0.0f;
        truncateds_[write_index_] = e.response.truncated ? 1.0f : 0.0f;

        write_index_ = (write_index_ + 1) % capacity_;
        size_ = std::min(size_ + 1, capacity_);
    }

    std::vector<Experience> ReplayBuffer::Sample(size_t n) const
    {
        n = std::min(n, size_);
        std::vector<Experience> out;
        out.reserve(n);

        std::uniform_int_distribution<size_t> dist(0, size_ - 1);

        for (size_t i = 0; i < n; ++i) {
            size_t idx = dist(engine_);

            Experience e;
            e.state = states_[idx];
            e.action = actions_[idx];
            e.response.next_state = next_states_[idx];
            e.response.reward = rewards_[idx].item<float>();
            e.response.done = dones_[idx].item<float>() > 0.5f;
            e.response.truncated = truncateds_[idx].item<float>() > 0.5f;

            out.push_back(e);
        }
        return out;
    }

    ExperienceBatch ReplayBuffer::SampleBatch(size_t n, torch::Device device) const
    {
        ExperienceBatch batch{};
        if (size_ == 0) return batch;

        n = std::min(n, size_);

        std::vector<int64_t> idx;
        idx.reserve(n);
        std::uniform_int_distribution<size_t> dist(0, size_ - 1);

        for (size_t i = 0; i < n; ++i)
            idx.push_back(static_cast<int64_t>(dist(engine_)));

        auto index_tensor = torch::tensor(idx,
            torch::TensorOptions().dtype(torch::kInt64).device(device_));

        batch.states = states_.index_select(0, index_tensor).to(device);
        batch.actions = actions_.index_select(0, index_tensor).to(device);
        batch.next_states = next_states_.index_select(0, index_tensor).to(device);
        batch.rewards = rewards_.index_select(0, index_tensor).to(device);
        batch.dones = dones_.index_select(0, index_tensor).to(device);
        batch.truncateds = truncateds_.index_select(0, index_tensor).to(device);

        return batch;
    }

} // namespace anet::rl
