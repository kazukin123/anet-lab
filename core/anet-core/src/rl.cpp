#include <stdexcept>
#include <wx/log.h>
#include "anet/rl.hpp"
#include "anet/common.hpp"
#include "anet/tensor_utils.hpp"
#include "anet/tensor_check.hpp"

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
        //auto& s = e.state;

        //// TODO: unbatch対応
        //if (s.dim() == 2) {
        //    int64_t N = s.size(0);

        //    for (int64_t i = 0; i < N; ++i) {
        //        EnvResponse resp({
        //                e.response.next_state[i],
        //                e.response.reward,
        //                e.response.done,
        //                e.response.truncated
        //            }
        //        );
        //        Experience ei{ s[i], e.action[i], resp };
        //        PushSingle_(ei);
        //    }
        //    return;
        //}

        // --- N=1 case ---
        PushSingle_(e);
    }

    void ReplayBuffer::PushSingle_(const Experience& e)
    {
        // --- Copy for preprocessing ---
        Experience exp = e;

        // ======================================
        // N = 1 前提の暫定処理：先頭次元が 1 の場合だけ除去
        // （Multi-Env は将来 unbatch に置き換え）
        // ======================================
        if (exp.state.dim() == 2 && exp.state.size(0) == 1)
            exp.state = exp.state.squeeze(0);

        if (exp.response.next_state.dim() == 2 && exp.response.next_state.size(0) == 1)
            exp.response.next_state = exp.response.next_state.squeeze(0);

        if (is_discrete_) {
            // 離散 Action = int64 scalar または size-1 tensor
            if (exp.action.dim() == 2 && exp.action.size(0) == 1)
                exp.action = exp.action.squeeze(0);

            ANET_ASSERT_MSG(
                exp.action.dtype() == torch::kInt64,
                "Discrete action must be int64."
            );

            ANET_ASSERT_MSG(
                exp.action.dim() == 0 || (exp.action.dim() == 1 && exp.action.size(0) == 1),
                "Discrete action must be scalar or size-1 tensor."
            );
        }

        // toCPU
        exp.state = exp.state.to(torch::kCPU);
        exp.action = exp.action.to(torch::kCPU);
        exp.response.next_state = exp.response.next_state.to(torch::kCPU);

        // ======================================
        // Device は CPU 固定（Trainer が to(device) を消したので必要）
        // ======================================
        ANET_CHECK_DEVICE_CPU(exp.state);
        ANET_CHECK_DEVICE_CPU(exp.action);
        ANET_CHECK_DEVICE_CPU(exp.response.next_state);

        // ======================================
        // 初回 push 時に内部テンソルを初期化
        // ======================================
        if (!initialized_) {
            //device_ = torch::kCPU;

            auto state_sizes = exp.state.sizes();   // e.g., [4]
            auto action_sizes = exp.action.sizes();  // e.g., [] or [1]

            // states_
            {
                std::vector<int64_t> shape(1 + state_sizes.size());
                shape[0] = static_cast<int64_t>(capacity_);
                for (size_t i = 0; i < state_sizes.size(); ++i)
                    shape[i + 1] = state_sizes[i];

                states_ = torch::empty(shape,
                    exp.state.options().device(torch::kCPU));
                next_states_ = torch::empty(shape,
                    exp.response.next_state.options().device(torch::kCPU));
            }

            // actions_
            {
                std::vector<int64_t> shape(1 + action_sizes.size());
                shape[0] = static_cast<int64_t>(capacity_);
                for (size_t i = 0; i < action_sizes.size(); ++i)
                    shape[i + 1] = action_sizes[i];

                actions_ = torch::empty(shape,
                    exp.action.options().device(torch::kCPU));
            }

            rewards_ = torch::empty(
                { static_cast<int64_t>(capacity_) },
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
            );

            dones_ = torch::empty(
                { static_cast<int64_t>(capacity_) },
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
            );

            truncateds_ = torch::empty(
                { static_cast<int64_t>(capacity_) },
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
            );

            initialized_ = true;
        }

        // ======================================
        // Ring-buffer 書き込み
        // ======================================
        states_[write_index_] = exp.state;
        actions_[write_index_] = exp.action;
        next_states_[write_index_] = exp.response.next_state;

        rewards_[write_index_] = exp.response.reward;
        dones_[write_index_] = exp.response.done ? 1.0f : 0.0f;
        truncateds_[write_index_] = exp.response.truncated ? 1.0f : 0.0f;

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

            wxLogDebug("ReplayBuffer::Sample() e.state=%s", anet::ToString(e.state));
            out.push_back(e);
        }
        return out;
    }

    ExperienceBatch ReplayBuffer::SampleBatch(size_t n, torch::Device device) const
    {
        ANET_ASSERT_MSG(n > 0, "SampleBatch: n must be > 0");

        ExperienceBatch batch{};
        if (size_ == 0) return batch;

        n = std::min(n, size_);

        std::vector<int64_t> idx;
        idx.reserve(n);
        std::uniform_int_distribution<size_t> dist(0, size_ - 1);

        for (size_t i = 0; i < n; ++i)
            idx.push_back(static_cast<int64_t>(dist(engine_)));

        auto index_tensor = torch::tensor(idx, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

        batch.states = states_.index_select(0, index_tensor).to(device);
        batch.actions = actions_.index_select(0, index_tensor).to(device);
        batch.next_states = next_states_.index_select(0, index_tensor).to(device);
        batch.rewards = rewards_.index_select(0, index_tensor).to(device);
        batch.dones = dones_.index_select(0, index_tensor).to(device);
        batch.truncateds = truncateds_.index_select(0, index_tensor).to(device);

        return batch;
    }

} // namespace anet::rl
