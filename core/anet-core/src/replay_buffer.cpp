#include <stdexcept>
#include <algorithm>
#include <wx/log.h>
#include "anet/rl.hpp"
#include "anet/common.hpp"
#include "anet/tensor_utils.hpp"
#include "anet/tensor_check.hpp"
#include "anet/replay_buffer.hpp"

namespace anet::rl {

    ReplayBuffer::ReplayBuffer(const EnvSpec& spec, size_t capacity, std::shared_ptr<anet::RandomGenerator> rnd)
        : RandomHolder(rnd),
        capacity_(capacity),
        state_count_(spec.state_spec.CalcFlattenSize()),
        n_actions_(spec.action_spec.ActionCount()),
        device_(torch::kCPU)
    {
        is_discrete_ = spec.action_spec.is_discrete;
        if (is_discrete_) {
            n_actions_ = 1;
        }

        ANET_ASSERT_MSG(state_count_ > 0,
            "ReplayBuffer::ReplayBuffer(): invalid state_count_.");

        ANET_ASSERT_MSG(n_actions_ > 0,
            "ReplayBuffer::ReplayBuffer(): invalid action_count_.");

        states_ = torch::zeros({ static_cast<long>(capacity_), state_count_ });
        next_states_ = torch::zeros({ static_cast<long>(capacity_), state_count_ });
        rewards_ = torch::zeros({ static_cast<long>(capacity_) });
        dones_ = torch::zeros({ static_cast<long>(capacity_) }, torch::TensorOptions().dtype(torch::kBool));
        truncateds_ = torch::zeros({ static_cast<long>(capacity_) }, torch::TensorOptions().dtype(torch::kBool));
        episode_start_ = torch::zeros({ static_cast<long>(capacity_) }, torch::TensorOptions().dtype(torch::kBool));

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

    void ReplayBuffer::Push(const BatchExperience& batch)
    {
        // shape チェック
        const int64_t N = batch.state.obs.size(0);

        ANET_CHECK_SHAPE(batch.state.obs, { N, state_count_ });
        ANET_CHECK_SHAPE(batch.state.done, { N });
        ANET_CHECK_SHAPE(batch.state.truncated, { N });
        ANET_CHECK_SHAPE(batch.state.episode_start, { N });
        ANET_CHECK_SHAPE(batch.action.action, { N, n_actions_ });
        ANET_CHECK_SHAPE(batch.action.is_random, { N, n_actions_ });
        ANET_CHECK_SHAPE(batch.reward, { N });
        ANET_CHECK_SHAPE(batch.next_state.obs, { N, state_count_ });
        ANET_CHECK_SHAPE(batch.next_state.done, { N });
        ANET_CHECK_SHAPE(batch.next_state.truncated, { N });
        ANET_CHECK_SHAPE(batch.next_state.episode_start, { N });

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
        for (int64_t i = 0; i < N; ++i) {   // N環境を回しながら取り出す事でunbatchする
            const int64_t idx = write_index_;

            states_[idx].copy_(batch.state.obs[i]);
            next_states_[idx].copy_(batch.next_state.obs[i]);
            actions_[idx].copy_(batch.action.action[i]);
            rewards_[idx] = batch.reward[i].item<float>();
            dones_[idx] = batch.next_state.done[i].item<bool>();
            truncateds_[idx] = batch.next_state.truncated[i].item<bool>();
            episode_start_[idx] = batch.next_state.episode_start[i].item<bool>();

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

            ANET_CHECK_SHAPE(e.state.obs, { state_count_ });
            ANET_CHECK_SHAPE(e.action, { n_actions_ });
            ANET_CHECK_SHAPE(e.next_state.obs, { state_count_ });

            states_[idx].copy_(e.state.obs);
            next_states_[idx].copy_(e.next_state.obs);
            actions_[idx].copy_(e.action);
            rewards_[idx] = e.reward;
            dones_[idx] = e.next_state.done ? 1.0f : 0.0f;
            truncateds_[idx] = e.next_state.truncated ? 1.0f : 0.0f;
            episode_start_[idx] = e.next_state.episode_start ? 1.0f : 0.0f;

            write_index_ = (write_index_ + 1) % static_cast<int64_t>(capacity_);
            if (size_ < capacity_) size_++;
        }
    }

    ExperienceSample ReplayBuffer::Sample(int64_t b, torch::Device device) const
    {
        ANET_ASSERT_MSG(size_ > 0, "ReplayBuffer::Sample: buffer empty.");
        ANET_ASSERT_MSG(b > 0, "ReplayBuffer::Sample: n must be > 0.");
        ANET_ASSERT_MSG(b <= size_, "ReplayBuffer::Sample: n exceeds current size.");

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

        ExperienceSample out({
                states_.index_select(0, idx).to(device),        // obs
                actions_.index_select(0, idx).to(device),       // action
                rewards_.index_select(0, idx).to(device),       // reward
                {   // next_states
                    next_states_.index_select(0, idx).to(device),   // next_states.obs
                    dones_.index_select(0, idx).to(device),         // next_states.dones
                    truncateds_.index_select(0, idx).to(device),    // next_states.truncateds
                    episode_start_.index_select(0, idx).to(device)  // next_states.episode_start
                }
            });

        ANET_CHECK_SHAPE(out.obs, { b, state_count_ });
        ANET_CHECK_SHAPE(out.actions,{ b, n_actions_ });
        ANET_CHECK_SHAPE(out.rewards, { b });
        ANET_CHECK_SHAPE(out.next_states.obs, { b, state_count_ });
        ANET_CHECK_SHAPE(out.next_states.dones, { b });
        ANET_CHECK_SHAPE(out.next_states.truncateds, { b });
        ANET_CHECK_SHAPE(out.next_states.episode_start, { b });

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

    std::optional<float> ReplayBuffer::GetScalar(const std::string& key) const
    {
        return std::nullopt;
    }

    std::optional<torch::Tensor> ReplayBuffer::GetTensor(const std::string& key) const
    {
        return std::nullopt;
    }

    std::optional<std::vector<torch::Tensor>> ReplayBuffer::GetTensorVector(const std::string& key) const
    {
        // ReplayBuffer は ring-buffer 構造のため、時系列順のデータは
        // メモリ上で最大 2 区間に分かれる。
        // 1 Tensor に連結すると memcpy が必要になり重いため、
        // コピーを避けるために vector<Tensor>（ビュー）で返す。
        // 各 Tensor は index による view で実データのコピーは発生しない。

        if (key == STATE_OBS)
            return replay_in_order(states_, size_, capacity_, write_index_);
        if (key == ACTION_ACTION)
            return replay_in_order(actions_, size_, capacity_, write_index_);
        if (key == REWARD)
            return replay_in_order(rewards_, size_, capacity_, write_index_);
        if (key == NEXT_STATE_OBS)
            return replay_in_order(next_states_, size_, capacity_, write_index_);
        if (key == NEXT_STATE_DONE)
            return replay_in_order(dones_, size_, capacity_, write_index_);
        if (key == NEXT_STATE_TRUNCATED)
            return replay_in_order(truncateds_, size_, capacity_, write_index_);
        if (key == NEXT_STATE_EPISODE_START)
            return replay_in_order(episode_start_, size_, capacity_, write_index_);

        return std::nullopt;
    }

    ExperienceSample ExperienceSample::Flatten() const {
        return ExperienceSample{
            obs.flatten(1), // obs
            actions,        // action
            rewards,        // reward
            {
                next_states.obs.flatten(1), // next_states.obs
                next_states.dones,          // next_states.dones
                next_states.truncateds,     // next_states.truncateds
                next_states.episode_start   // next_states.episode_start
            }
        };
    }

    std::string ExperienceSample::ToString() const
    {
        std::ostringstream oss;
        oss << "ExperienceSample{\n";
        oss << "  obs     = " << anet::ToString(obs) << "\n";
        oss << "  action  = " << anet::ToString(actions) << "\n";
        oss << "  reward  = " << anet::ToString(rewards) << "\n";
        oss << "  next_state.obs           = " << anet::ToString(next_states.obs) << "\n";
        oss << "  next_state.dones         = " << anet::ToString(next_states.dones) << "\n";
        oss << "  next_state.truncateds    = " << anet::ToString(next_states.truncateds) << "\n";
        oss << "  next_state.episode_start = " << anet::ToString(next_states.episode_start) << "\n";
        oss << "}";
        return oss.str();
    }

} // namespace anet::rl
