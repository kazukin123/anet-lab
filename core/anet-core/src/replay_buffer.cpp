#include "anet/replay_buffer.hpp"
#include <stdexcept>
#include <algorithm>
#include <wx/log.h>
#include "anet/common.hpp"
#include "anet/tensor_util.hpp"
#include "anet/tensor_check.hpp"
#include "anet/profile.hpp"
#include "anet/rl.hpp"

namespace anet::rl {

    PlainReplayBuffer::PlainReplayBuffer(const EnvSpec& spec, size_t capacity, std::optional<seed_t> seed)
        : RandomHolder(seed),
        capacity_(capacity),
        state_dim_(spec.state_spec.CalcFlattenDim()),
        n_actions_(spec.action_spec.GetNumActions()),
        device_(torch::kCPU)
    {
        is_discrete_ = spec.action_spec.is_discrete;
        if (is_discrete_) {
            n_actions_ = 1;
        }

        ANET_ASSERT_MSG(state_dim_ > 0,
            "ReplayBuffer::ReplayBuffer(): invalid state_count_.");

        ANET_ASSERT_MSG(n_actions_ > 0,
            "ReplayBuffer::ReplayBuffer(): invalid action_count_.");

        states_ = torch::zeros({ static_cast<long>(capacity_), state_dim_ });
        next_states_ = torch::zeros({ static_cast<long>(capacity_), state_dim_ });
        rewards_ = torch::zeros({ static_cast<long>(capacity_) });
        terminals_ = torch::zeros({ static_cast<long>(capacity_) }, torch::TensorOptions().dtype(torch::kBool));

        if (is_discrete_) {
            actions_ = torch::zeros(
                { static_cast<long>(capacity_), 1 },
                torch::TensorOptions().dtype(torch::kInt64) // 離散アクションでは1次元かつint64固定
            );
        } else {
            actions_ = torch::zeros(
                { static_cast<long>(capacity_), n_actions_ },
                torch::TensorOptions().dtype(torch::kFloat32)
            );
        }
    }

    void PlainReplayBuffer::Push(const BatchExperience& batch)
    {
        anet::ProfileRange r1("PlainReplayBuffer::Push1");

        // shape チェック
        const int64_t N = batch.state.obs.size(0);

        ANET_CHECK_SHAPE(batch.state.obs, { N, state_dim_ });
        ANET_CHECK_SHAPE(batch.state.done, { N });
        ANET_CHECK_SHAPE(batch.state.truncated, { N });
        ANET_CHECK_SHAPE(batch.state.episode_start, { N });
        if (is_discrete_) {
            ANET_CHECK_SHAPE(batch.action.GetAction(), {N});
        } else {
            ANET_CHECK_SHAPE(batch.action.GetAction(), {N, ANET_SHAPE_ENDANY});
        }
        ANET_CHECK_SHAPE(batch.reward, { N });
        ANET_CHECK_SHAPE(batch.next_state.obs, { N, state_dim_ });
        ANET_CHECK_SHAPE(batch.next_state.done, { N });
        ANET_CHECK_SHAPE(batch.next_state.truncated, { N });
        ANET_CHECK_SHAPE(batch.next_state.episode_start, { N });

        ANET_CHECK_DTYPE(batch.state.obs, torch::kFloat32);
        ANET_CHECK_DTYPE(batch.state.done, torch::kBool);
        ANET_CHECK_DTYPE(batch.state.truncated, torch::kBool);
        ANET_CHECK_DTYPE(batch.state.episode_start, torch::kBool);
        ANET_CHECK_DTYPE(batch.action.GetAction(), is_discrete_ ? torch::kInt64 : torch::kFloat32);
        ANET_CHECK_DTYPE(batch.reward, torch::kFloat32);
        ANET_CHECK_DTYPE(batch.next_state.obs, torch::kFloat32);
        ANET_CHECK_DTYPE(batch.next_state.done, torch::kBool);
        ANET_CHECK_DTYPE(batch.next_state.truncated, torch::kBool);
        ANET_CHECK_DTYPE(batch.next_state.episode_start, torch::kBool);

        auto exps = batch.ToExperienceList();
        Push(exps);
    }

    void PlainReplayBuffer::Push(const std::vector<SingleExperience>& exps)
    {
        anet::ProfileRange r1("PlainReplayBuffer::Push2");

        size_t n = exps.size();
        if (n == 0) return;

        for (size_t i = 0; i < n; ++i) {
            const auto& e = exps[i];

            const int64_t idx = write_index_;

            ANET_CHECK_SHAPE(e.state.obs, { state_dim_ });
            ANET_CHECK_SHAPE(e.action, { });
            ANET_CHECK_SHAPE(e.next_state.obs, { state_dim_ });

            states_[idx].copy_(e.state.obs);
            next_states_[idx].copy_(e.next_state.obs);
            actions_[idx].copy_(e.action);
            rewards_[idx] = e.reward;
            terminals_[idx] = (e.next_state.done || e.next_state.truncated) ? 1.0f : 0.0f;

            write_index_ = (write_index_ + 1) % static_cast<int64_t>(capacity_);
            if (size_ < capacity_) size_++;
        }
    }

    ExperienceSamples PlainReplayBuffer::Sample(int64_t b, torch::Device device, float beta) const
    {
        anet::ProfileRange r1("PlainReplayBuffer::Sample");

        ANET_ASSERT_MSG(size_ > 0, "ReplayBuffer::Sample: buffer empty.");
        ANET_ASSERT_MSG(b > 0, "ReplayBuffer::Sample: n must be > 0.");
        //ANET_ASSERT_MSG(b <= size_, "ReplayBuffer::Sample: n exceeds current size.");

        // ---- RNG を使って n 個のインデックスを取得 ----
        std::vector<int64_t> idx_vec;
        idx_vec.reserve(b);

        const int64_t max_i = static_cast<int64_t>(size_);

        for (size_t i = 0; i < b; ++i) {
            int64_t v = static_cast<int64_t>(rnd_->RandIndex(size_));
            ANET_ASSERT_MSG(0 <= v && v < max_i,
                "ReplayBuffer::Sample: rng returned out-of-range index.");
            idx_vec.push_back(v);
        }

        torch::Tensor idx = torch::tensor(idx_vec, torch::TensorOptions().dtype(torch::kLong));

        ExperienceSamples out({
                states_.index_select(0, idx).to(device),        // obs
                actions_.index_select(0, idx).to(device),       // action
                rewards_.index_select(0, idx).to(device),       // reward
                {   // next_states
                    next_states_.index_select(0, idx).to(device),   // next_states.obs
                    terminals_.index_select(0, idx).to(device),     // next_states.terminals
                },
                torch::Tensor(), // n-steps
                torch::Tensor(),        // indices          (B,) kInt64
                torch::Tensor(),        // sampling_prob;   (B,) kFloat32
                torch::Tensor(),        // is_weights;      (B,) kFloat32
            });
        ANET_CHECK_SHAPE(out.obs, { b, state_dim_ });
        ANET_CHECK_SHAPE(out.actions,{ b, n_actions_ });
        ANET_CHECK_SHAPE(out.target_values, { b });
        ANET_CHECK_SHAPE(out.next_states.obs, { b, state_dim_ });
        ANET_CHECK_SHAPE(out.next_states.terminals, { b });

        return out;
    }

    std::vector<torch::Tensor> replay_in_order(
        const torch::Tensor& t,
        size_t size,
        size_t capacity,
        size_t write_index)
    {
        using Slice = torch::indexing::Slice;

        std::vector<torch::Tensor> out;
        if (size == 0) return out;

        out.reserve(2);

        size_t head = write_index;
        size_t tail = (head + capacity - size) % capacity;

        size_t first_len = std::min(size, capacity - tail);
        if (first_len > 0) {
            out.push_back(
                t.index({ Slice((int64_t)tail, (int64_t)(tail + first_len)) })
            );
        }

        size_t second_len = size - first_len;
        if (second_len > 0) {
            out.push_back(
                t.index({ Slice(0, (int64_t)second_len) })
            );
        }

        return out;
    }

    std::optional<float> PlainReplayBuffer::GetScalar(const std::string& key, int64_t index) const
    {
        return std::nullopt;
    }

    std::optional<torch::Tensor> PlainReplayBuffer::GetTensor(const std::string& key, int64_t index) const
    {
        return std::nullopt;
    }

    std::optional<std::vector<torch::Tensor>> PlainReplayBuffer::GetTensorVector(const std::string& key, int64_t index) const
    {
        anet::ProfileRange r1("PlainReplayBuffer::GetTensorVector");

        /// @todo index指定対応

        // ReplayBuffer は ring-buffer 構造のため、時系列順のデータは
        // メモリ上で最大 2 区間に分かれる。
        // 1 Tensor に連結すると memcpy が必要になり重いため、
        // コピーを避けるために vector<Tensor>（ビュー）で返す。
        // 各 Tensor は index による view で実データのコピーは発生しない。

        if (key == STATE_OBS)
            return replay_in_order(states_, size_, capacity_, write_index_);
        if (key == ACTION)
            return replay_in_order(actions_, size_, capacity_, write_index_);
        if (key == REWARD)
            return replay_in_order(rewards_, size_, capacity_, write_index_);
        if (key == NEXT_STATE_OBS)
            return replay_in_order(next_states_, size_, capacity_, write_index_);
        if (key == NEXT_STATE_TERMINAL)
            return replay_in_order(terminals_, size_, capacity_, write_index_);
        if (key == N_STEP)
            return std::nullopt;    // 非対応

        return std::nullopt;
    }


} // namespace anet::rl
