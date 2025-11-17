#include <stdexcept>
#include <wx/log.h>
#include "anet/rl.hpp"
#include "anet/common.hpp"
#include "anet/tensor_utils.hpp"
#include "anet/tensor_check.hpp"
#include "anet/replay_buffer.hpp"

namespace anet::rl {

    // batched Experience を検出したら N サンプルに自動分割して Push する
    static inline bool IsBatched(const torch::Tensor& t) {
        return (t.dim() >= 2 && t.size(0) >= 1);
    }

    void ReplayBuffer::Push(const Experience& e) {
        wxLogDebug("ReplayBuffer::Push() e.state=%s", anet::ToString(e.state));
        wxLogDebug("ReplayBuffer::Push() e.action=%s", anet::ToString(e.action));
        wxLogDebug("ReplayBuffer::Push() e.next_state=%s", anet::ToString(e.result.next_state));
        wxLogDebug("ReplayBuffer::Push() e.reward=%s", anet::ToString(e.result.reward));
        wxLogDebug("ReplayBuffer::Push() e.done=%s", anet::ToString(e.result.done));
        wxLogDebug("ReplayBuffer::Push() e.truncated=%s", anet::ToString(e.result.truncated));

        // -----------------------------------------------------
        // 1) batched Experience の自動展開 (N ≥ 1)
        // -----------------------------------------------------
        if (IsBatched(e.state)) {

            int64_t N = e.state.size(0);

            // action は shape が [N] または [N, A] の場合あり
            bool action_batched = IsBatched(e.action) ||
                (e.action.dim() == 1 && e.action.size(0) == N);

            for (int64_t i = 0; i < N; ++i) {
                Experience ei;
                ei.state = e.state[i];                         // [S]
                ei.result.next_state = e.result.next_state[i]; // [S]
                ei.result.reward = e.result.reward[i];         // []
                ei.result.done = e.result.done[i];             // []
                ei.result.truncated = e.result.truncated[i];   // []

                if (action_batched) {
                    // 離散なら action[i] は scalar / continuous なら [A]
                    ei.action = (e.action.dim() == 1) ? e.action[i] : e.action[i];
                } else {
                    ei.action = e.action;   // もともと scalar
                }

                // 再帰的に N=1 path に落とす
                Push(ei);
            }

            return;
        }

        // -----------------------------------------------------
        // 2) ここからは N=1 の単一 Experience 処理
        // -----------------------------------------------------
        if (is_discrete_) {
            ANET_ASSERT_MSG(
                e.action.dtype() == torch::kInt64,
                "Discrete action must be int64."
            );
            //ANET_ASSERT_MSG(
            //    exp.action.dim() == 0 || (exp.action.dim() == 1 && exp.action.size(0) == 1),
            //    "Discrete action must be scalar or size-1 tensor."
            //);
        }

        // CPU に統一
        torch::Tensor s = e.state.to(torch::kCPU);
        torch::Tensor ns = e.result.next_state.to(torch::kCPU);
        torch::Tensor a = e.action.to(torch::kCPU);
        torch::Tensor r = e.result.reward.to(torch::kCPU);
        torch::Tensor dn = e.result.done.to(torch::kCPU);
        torch::Tensor tr = e.result.truncated.to(torch::kCPU);

        wxLogDebug("ReplayBuffer::Push() state=%s", anet::ToString(s));
        wxLogDebug("ReplayBuffer::Push() next_state=%s", anet::ToString(a));
        wxLogDebug("ReplayBuffer::Push() reward=%s", anet::ToString(r));
        wxLogDebug("ReplayBuffer::Push() action=%s", anet::ToString(ns));
        wxLogDebug("ReplayBuffer::Push() done=%s", anet::ToString(dn));
        wxLogDebug("ReplayBuffer::Push() truncated=%s", anet::ToString(tr));

        // -----------------------------------------------------
        // 3) 初回のみテンソル確保
        // -----------------------------------------------------
        if (!initialized_) {

            auto S = s.sizes();   // [S]
            auto A = a.sizes();   // [] or [A]

            // states_: [capacity, S]
            {
                std::vector<int64_t> shp(1 + S.size());
                shp[0] = (int64_t)capacity_;
                for (size_t i = 0; i < S.size(); ++i)
                    shp[i + 1] = S[i];

                states_ = torch::empty(shp, s.options());
                next_states_ = torch::empty(shp, s.options());
            }

            // actions_: [capacity, A] または [capacity]
            {
                std::vector<int64_t> shp(1 + A.size());
                shp[0] = (int64_t)capacity_;
                for (size_t i = 0; i < A.size(); ++i)
                    shp[i + 1] = A[i];

                actions_ = torch::empty(shp, a.options());
            }

            // reward / done / truncated: [capacity]
            rewards_ = torch::empty({ (long)capacity_ }, r.options());
            dones_ = torch::empty({ (long)capacity_ }, dn.options());
            truncateds_ = torch::empty({ (long)capacity_ }, tr.options());

            wxLogDebug("ReplayBuffer::Push() initialized_ states_=%s", anet::ToDefString(states_));
            wxLogDebug("ReplayBuffer::Push() initialized_ actions_=%s", anet::ToDefString(actions_));
            wxLogDebug("ReplayBuffer::Push() initialized_ rewards_=%s", anet::ToDefString(rewards_));
            wxLogDebug("ReplayBuffer::Push() initialized_ dones_=%s", anet::ToDefString(dones_));
            wxLogDebug("ReplayBuffer::Push() initialized_ truncateds_=%s", anet::ToDefString(truncateds_));

            initialized_ = true;
        }

        // -----------------------------------------------------
        // 4) 単一サンプルの書き込み
        // -----------------------------------------------------
        states_[write_index_] = s;
        actions_[write_index_] = a;
        next_states_[write_index_] = ns;

        rewards_[write_index_] = r;
        dones_[write_index_] = dn;
        truncateds_[write_index_] = tr;

        write_index_ = (write_index_ + 1) % capacity_;
        size_ = std::min(size_ + 1, capacity_);
    }

    anet::rl::ExperienceSample ReplayBuffer::Sample(size_t n, torch::Device device) const {
        ANET_ASSERT_MSG(n > 0, "SampleBatch: n must be > 0");

        ExperienceSample batch{};
        if (size_ == 0) return batch;

        n = std::min(n, size_);

        std::vector<int64_t> idx;
        idx.reserve(n);
        std::uniform_int_distribution<size_t> dist(0, size_ - 1);

        for (size_t i = 0; i < n; ++i)
            idx.push_back(static_cast<int64_t>(rng_->RandIndex(size_ - 1)));

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
