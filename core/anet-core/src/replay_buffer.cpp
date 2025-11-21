#include <stdexcept>
#include <algorithm>
#include <wx/log.h>
#include "anet/rl.hpp"
#include "anet/common.hpp"
#include "anet/tensor_utils.hpp"
#include "anet/tensor_check.hpp"
#include "anet/replay_buffer.hpp"

namespace anet::rl {

    ReplayBuffer::ReplayBuffer(const EnvSpec& spec, size_t capacity, anet::RandomGenerator* rnd)
        : RandomHolder(rnd),
        capacity_(capacity),
        state_dim_(spec.state.CalcStateDim()),
        action_dim_(spec.action.ActionCount()),
        device_(torch::kCPU)
    {
        is_discrete_ = spec.action.is_discrete;
        if (is_discrete_) {
            action_dim_ = 1;
        }

        ANET_ASSERT_MSG(state_dim_ > 0,
            "ReplayBuffer::ReplayBuffer(): invalid state_dim.");

        ANET_ASSERT_MSG(action_dim_ > 0,
            "ReplayBuffer::ReplayBuffer(): invalid action_dim.");

        states_ = torch::zeros({ static_cast<long>(capacity_), state_dim_ });
        next_states_ = torch::zeros({ static_cast<long>(capacity_), state_dim_ });
        rewards_ = torch::zeros({ static_cast<long>(capacity_) });
        dones_ = torch::zeros({ static_cast<long>(capacity_) }, torch::TensorOptions().dtype(torch::kBool));
        truncateds_ = torch::zeros({ static_cast<long>(capacity_) }, torch::TensorOptions().dtype(torch::kBool));

        if (is_discrete_) {
            actions_ = torch::zeros(
                { static_cast<long>(capacity_), 1 },
                torch::TensorOptions().dtype(torch::kInt64) // 離散アクションでは1次元かつint64固定
            );
        } else {
            actions_ = torch::zeros(
                { static_cast<long>(capacity_), action_dim_ },
                torch::TensorOptions().dtype(torch::kFloat32)
            );
        }
    }

    // replay_buffer.cpp
    void ReplayBuffer::Push(const BatchExperience& batch)
    {
        // shape チェック
        const int64_t B = batch.state.obs.size(0);

        ANET_CHECK_SHAPE(batch.state.obs, { B, state_dim_ });
        ANET_CHECK_SHAPE(batch.state.done, { B });
        ANET_CHECK_SHAPE(batch.state.truncated, { B });
        ANET_CHECK_SHAPE(batch.state.episode_start, { B });
        ANET_CHECK_SHAPE(batch.action.action, { B, action_dim_ });
        ANET_CHECK_SHAPE(batch.action.is_random, { B, action_dim_ });
        ANET_CHECK_SHAPE(batch.reward, { B });
        ANET_CHECK_SHAPE(batch.next_state.obs, { B, state_dim_ });
        ANET_CHECK_SHAPE(batch.next_state.done, { B });
        ANET_CHECK_SHAPE(batch.next_state.truncated, { B });
        ANET_CHECK_SHAPE(batch.next_state.episode_start, { B });

        ANET_CHECK_DTYPE(batch.state.obs, torch::kFloat32);
        ANET_CHECK_DTYPE(batch.state.done, torch::kBool);
        ANET_CHECK_DTYPE(batch.state.truncated, torch::kBool);
        ANET_CHECK_DTYPE(batch.state.episode_start, torch::kBool);
        ANET_CHECK_DTYPE(batch.action.action, is_discrete_ ? torch::kInt64 : torch::kFloat32);
        ANET_CHECK_DTYPE(batch.action.is_random, torch::kBool);
        ANET_CHECK_DTYPE(batch.reward, torch::kFloat32);
        ANET_CHECK_DTYPE(batch.next_state.obs, torch::kFloat32);
        ANET_CHECK_DTYPE(batch.next_state.done, torch::kBool);
        ANET_CHECK_DTYPE(batch.next_state.truncated, torch::kBool);
        ANET_CHECK_DTYPE(batch.next_state.episode_start, torch::kBool);

        // 1件ずつ circular buffer に書き込む
        for (int64_t i = 0; i < B; ++i) {
            const int64_t idx = write_index_;

            states_[idx].copy_(batch.state.obs[i]);
            next_states_[idx].copy_(batch.next_state.obs[i]);
            actions_[idx].copy_(batch.action.action[i]);
            rewards_[idx] = batch.reward[i].item<float>();
            dones_[idx] = batch.next_state.done[i].item<bool>();
            truncateds_[idx] = batch.next_state.truncated[i].item<bool>();

            write_index_ = (write_index_ + 1) % static_cast<int64_t>(capacity_);
            if (size_ < capacity_) size_++;
        }
    }

    void ReplayBuffer::Push(const std::vector<Experience>& exps)
    {
        size_t n = exps.size();
        if (n == 0) return;

        for (size_t i = 0; i < n; ++i) {
            const auto& e = exps[i];

            const int64_t idx = write_index_;

            auto flat_s = e.state.Flattened();
            auto flat_ns = e.next_state.Flattened();

            ANET_CHECK_SHAPE(flat_s, { state_dim_ });
            ANET_CHECK_SHAPE(flat_ns, { state_dim_ });
            ANET_CHECK_SHAPE(e.action, { action_dim_ });

            states_[idx].copy_(flat_s);
            next_states_[idx].copy_(flat_ns);
            actions_[idx].copy_(e.action);
            rewards_[idx] = e.reward;
            dones_[idx] = e.next_state.done ? 1.0f : 0.0f;
            truncateds_[idx] = e.next_state.truncated ? 1.0f : 0.0f;

            write_index_ = (write_index_ + 1) % static_cast<int64_t>(capacity_);
            if (size_ < capacity_) size_++;
        }
    }

    ExperienceSample ReplayBuffer::Sample(size_t n, torch::Device device) const
    {
        ANET_ASSERT_MSG(size_ > 0, "ReplayBuffer::Sample: buffer empty.");
        ANET_ASSERT_MSG(n > 0, "ReplayBuffer::Sample: n must be > 0.");
        ANET_ASSERT_MSG(n <= size_, "ReplayBuffer::Sample: n exceeds current size.");
        ANET_ASSERT_MSG(rng_ != nullptr,
            "ReplayBuffer::Sample: rng_ must not be null.");

        // ---- RNG を使って n 個のインデックスを取得 ----
        std::vector<int64_t> vec;
        vec.reserve(n);

        const int64_t max_i = static_cast<int64_t>(size_);

        for (size_t i = 0; i < n; ++i) {
            int64_t v = static_cast<int64_t>(rng_->RandIndex(size_ - 1));
            ANET_ASSERT_MSG(0 <= v && v < max_i,
                "ReplayBuffer::Sample: rng returned out-of-range index.");
            vec.push_back(v);
        }

        torch::Tensor idx = torch::tensor(
            vec, torch::TensorOptions().dtype(torch::kLong));

        ExperienceSample out;

        out.states = states_.index_select(0, idx).to(device);
        out.actions = actions_.index_select(0, idx).to(device);
        out.rewards = rewards_.index_select(0, idx).to(device);
        out.next_states = next_states_.index_select(0, idx).to(device);
        out.dones = dones_.index_select(0, idx).to(device);
        out.truncateds = truncateds_.index_select(0, idx).to(device);

        ANET_CHECK_SHAPE(out.states, { static_cast<int64_t>(n), state_dim_ });
        ANET_CHECK_SHAPE(out.actions,
            { static_cast<int64_t>(n), action_dim_ });
        ANET_CHECK_SHAPE(out.rewards, { static_cast<int64_t>(n) });
        ANET_CHECK_SHAPE(out.next_states,
            { static_cast<int64_t>(n), state_dim_ });
        ANET_CHECK_SHAPE(out.dones, { static_cast<int64_t>(n) });
        ANET_CHECK_SHAPE(out.truncateds, { static_cast<int64_t>(n) });

        return out;
    }

} // namespace anet::rl
