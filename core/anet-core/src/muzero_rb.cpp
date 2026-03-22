// muzero_rb.cpp

#include "muzero_rb.hpp"
#include "anet/profile.hpp"
#include "anet/tensor_check.hpp"


using namespace anet::rl::muzero_proto;


// ======================================================
// MuZeroExperienceSamples
// ======================================================

MuZeroExperienceSamples MuZeroExperienceSamples::To(torch::Device device) const
{
    anet::ProfileRange r("MuZeroExperienceSamples::To");

    return {
        obs.to(device, /*non_blocking=*/true),
        actions.to(device, true),
        target_rewards.to(device, true),
        target_values.to(device, true),
        target_policies.to(device, true),
        masks.to(device, true)
    };
}


// ======================================================
// MuZeroExperienceQueue
// ======================================================

void MuZeroExperienceQueue::Push(const MuZeroStepRecord& record)
{
    buffer_.push_back(record);
}

void MuZeroExperienceQueue::Pop()
{
    if (!buffer_.empty()) buffer_.pop_front();
}

void MuZeroExperienceQueue::Clear()
{
    buffer_.clear();
}

const MuZeroStepRecord& MuZeroExperienceQueue::Peek(size_t index) const
{
    ANET_ASSERT(index < buffer_.size());
    return buffer_[index];
}


// ======================================================
// MuZeroReplayBuilder
// ======================================================

MuZeroReplayBuilder::MuZeroReplayBuilder(int n_step, float gamma)
    : n_step_(n_step), gamma_(gamma)
{
    ;
}

std::vector<MuZeroReplayRecord> MuZeroReplayBuilder::ProcessQueue(MuZeroExperienceQueue& queue)
{
    anet::ProfileRange r("MuZeroReplayBuilder::ProcessQueue");

    std::vector<MuZeroReplayRecord> out_records;

    // N-Step先まで未来が見えた場合、先頭要素の価値を計算して確定
    if (queue.Size() >= static_cast<size_t>(n_step_ + 1)) {
        float target_value = 0.0f;
        float current_gamma = 1.0f;

        // 未来Nステップ分の報酬を足し込む
        for (int i = 0; i < n_step_; ++i) {
            target_value += current_gamma * queue.Peek(i).reward;
            current_gamma *= gamma_;
        }
        // Nステップ先の状態価値を足し込む
        target_value += current_gamma * queue.Peek(n_step_).root_value;

        // 確定した先頭レコードを生成
        const auto& head = queue.Peek(0);
        out_records.push_back({
            head.obs, head.action, head.reward, head.done,
            head.target_policy, target_value
            });
        queue.Pop();
    }

    // 最新のステップが done または truncated の場合にフラッシュする
    if (queue.Size() > 0) {
        const auto& last_step = queue.Peek(queue.Size() - 1);

        if (last_step.done || last_step.truncated) {
            size_t remaining = queue.Size();

            for (size_t k = 0; k < remaining; ++k) {
                float target_value = 0.0f;
                float current_gamma = 1.0f;
                size_t steps_to_done = remaining - k;

                // 終了までの報酬を足し込む
                for (size_t i = 0; i < steps_to_done; ++i) {
                    target_value += current_gamma * queue.Peek(k + i).reward;
                    current_gamma *= gamma_;
                }

                // truncatedの場合、残りの価値を0ではなく最後の状態の予測価値(root_value)でブートストラップして底上げする
                if (last_step.truncated) {
                    target_value += current_gamma * last_step.root_value;
                }

                const auto& head = queue.Peek(k);
                out_records.push_back({
                    head.obs, head.action, head.reward, (head.done || head.truncated), head.target_policy, target_value
                    });
            }
            queue.Clear();
        }
    }

    return out_records;
}


// ======================================================
// MuZeroReplayStorage
// ======================================================

MuZeroReplayStorage::MuZeroReplayStorage(int64_t capacity, int64_t num_envs)
    : capacity_per_env_(std::max<int64_t>(1, capacity / num_envs)), num_envs_(num_envs)
{
    env_buffers_.resize(num_envs_);
}

void MuZeroReplayStorage::Push(int64_t env_index, const MuZeroReplayRecord& record)
{
    ANET_ASSERT(env_index >= 0 && env_index < num_envs_);
    auto& buffer = env_buffers_[env_index];

    if (buffer.size() >= static_cast<size_t>(capacity_per_env_)) {
        buffer.pop_front();
        total_size_--;
    }

    buffer.push_back(record);
    total_size_++;
}

const MuZeroReplayRecord& MuZeroReplayStorage::GetRecord(int64_t env_index, int64_t t) const
{
    return env_buffers_[env_index][t];
}


// ======================================================
// MuZeroUniformSampler
// ======================================================

MuZeroUniformSampler::MuZeroUniformSampler(int unroll_steps, std::optional<anet::seed_t> seed)
    : anet::RandomHolder(seed), unroll_steps_(unroll_steps)
{
    ;
}

MuZeroExperienceSamples MuZeroUniformSampler::Sample(const MuZeroReplayStorage& storage, int64_t batch_size)
{
    anet::ProfileRange r("MuZeroUniformSampler::Sample");

    ANET_ASSERT_MSG(storage.GetTotalSize() > 0, "Cannot sample from an empty ReplayBuffer.");

    // バッチ分のTensorを構築するための配列
    std::vector<torch::Tensor> b_obs;
    std::vector<torch::Tensor> b_actions;
    std::vector<torch::Tensor> b_rewards;
    std::vector<torch::Tensor> b_values;
    std::vector<torch::Tensor> b_policies;
    std::vector<torch::Tensor> b_masks;

    b_obs.reserve(batch_size);
    b_actions.reserve(batch_size);
    b_rewards.reserve(batch_size);
    b_values.reserve(batch_size);
    b_policies.reserve(batch_size);
    b_masks.reserve(batch_size);

    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32);
    auto int_opts = torch::TensorOptions().dtype(torch::kInt64);

    for (int64_t b = 0; b < batch_size; ++b) {
        // 有効なデータを持つEnvをランダムに探す
        int64_t env_idx = 0;
        int64_t env_size = 0;
        while (true) {
            env_idx = rnd_->RandIndex(storage.GetNumEnvs());
            env_size = storage.GetEnvSize(env_idx);
            if (env_size > 0) break; // 最低1件あればサンプリング可能
        }

        // 抽出の起点(t)を決定
        int64_t t_start = rnd_->RandIndex(env_size);

        // 各Stepのデータを格納するローカルバッファ
        std::vector<int64_t> seq_actions(unroll_steps_, 0);
        std::vector<float> seq_rewards(unroll_steps_, 0.0f);
        std::vector<float> seq_values(unroll_steps_ + 1, 0.0f);
        std::vector<torch::Tensor> seq_policies;
        std::vector<float> seq_masks(unroll_steps_ + 1, 1.0f);

        // 起点となる t の観測(obs)を取得
        const auto& start_record = storage.GetRecord(env_idx, t_start);
        b_obs.push_back(start_record.obs);

        int64_t num_actions = start_record.target_policy.size(0);
        bool episode_ended = false;

        // 起点(t) から 未来(t + K) までのデータを収集
        for (int k = 0; k <= unroll_steps_; ++k) {
            int64_t t_current = t_start + k;

            if (episode_ended || t_current >= env_size) {
                // エピソード終端を超えたらゼロパディング
                seq_values[k] = 0.0f;
                seq_policies.push_back(torch::zeros({ num_actions }, float_opts));
                seq_masks[k] = 0.0f; // マスクを落とす
                if (k < unroll_steps_) {
                    seq_actions[k] = rnd_->RandIndex(num_actions); // ランダムアクションで埋める
                    seq_rewards[k] = 0.0f;
                }
            } else {
                const auto& rec = storage.GetRecord(env_idx, t_current);
                seq_values[k] = rec.target_value;
                seq_policies.push_back(rec.target_policy);

                if (k < unroll_steps_) {
                    seq_actions[k] = rec.action;
                    seq_rewards[k] = rec.reward;
                }

                if (rec.done) {
                    episode_ended = true; // このステップでエピソードが終了した
                }
            }
        }

        b_actions.push_back(torch::tensor(seq_actions, int_opts));
        b_rewards.push_back(torch::tensor(seq_rewards, float_opts));
        b_values.push_back(torch::tensor(seq_values, float_opts));
        b_policies.push_back(torch::stack(seq_policies, 0)); // [K+1, num_actions]
        b_masks.push_back(torch::tensor(seq_masks, float_opts));
    }

    // テンソルをスタックしてバッチ次元 [B, ...] を作成
    return {
        torch::stack(b_obs, 0),
        torch::stack(b_actions, 0),
        torch::stack(b_rewards, 0),
        torch::stack(b_values, 0),
        torch::stack(b_policies, 0),
        torch::stack(b_masks, 0)
    };
}


// ======================================================
// MuZeroReplayBuffer
// ======================================================

MuZeroReplayBuffer::MuZeroReplayBuffer(
    const ReplayBufferConfig& config, int64_t num_envs, int64_t num_actions, std::optional<anet::seed_t> seed)
    : anet::RandomHolder(seed), num_envs_(num_envs), num_actions_(num_actions)
{
    queues_.resize(num_envs_);
    builder_ = std::make_unique<MuZeroReplayBuilder>(config.n_step, config.gamma);
    storage_ = std::make_unique<MuZeroReplayStorage>(config.capacity, num_envs_);
    sampler_ = std::make_unique<MuZeroUniformSampler>(config.unroll_steps, seed);
}

void MuZeroReplayBuffer::Push(const anet::rl::BatchExperience& batch_exp)
{
    anet::ProfileRange r("MuZeroReplayBuffer::Push");

    // Actorから渡された info ディクショナリを取り出す
    const auto& info = batch_exp.action.GetInfo();
    auto policy_tensor = info.At("target_policy");
    auto value_tensor = info.At("root_value");

    for (int64_t b = 0; b < num_envs_; ++b) {
        // 生のステップデータを構築
        MuZeroStepRecord step;
        step.obs = batch_exp.state.obs[b].cpu();
        step.action = batch_exp.action.GetAction()[b].item<int64_t>();
        step.reward = batch_exp.reward[b].item<float>();
        step.done = batch_exp.next_state.done[b].item<bool>();
        step.truncated = batch_exp.next_state.truncated[b].item<bool>();

        step.target_policy = policy_tensor[b].cpu();
        step.root_value = value_tensor[b].item<float>();

        // Queueに追加
        auto& queue = queues_[b];
        queue.Push(step);

        // BuilderでN-Step計算済みレコードを抽出
        auto records = builder_->ProcessQueue(queue);

        // Storageに格納
        for (const auto& rec : records) {
            storage_->Push(b, rec);
        }
    }
}

int64_t MuZeroReplayBuffer::Size() const
{
    return storage_->GetTotalSize();
}

MuZeroExperienceSamples MuZeroReplayBuffer::Sample(int64_t batch_size, torch::Device device) const
{
    anet::ProfileRange r("MuZeroReplayBuffer::SampleMuZero");
    
    auto samples = sampler_->Sample(*storage_, batch_size);
    return samples.To(device); // GPU等のターゲットデバイスへ一括転送
}

