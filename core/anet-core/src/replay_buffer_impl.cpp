// replay_buffer_impl.cpp

#include "replay_buffer_impl.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/profile.hpp"

// ======================================================
// ReplayBuffer ExperienceQueue 
// ======================================================

void ExperienceQueue::Push(const SingleExperience& exp)
{
    buffer_.push_back(exp);
}

void ExperienceQueue::Pop(size_t k)
{
    if (k == 0 || buffer_.empty()) {
        return;
    }

    if (k >= buffer_.size()) {
        buffer_.clear();
        return;
    }

    buffer_.erase(buffer_.begin(), buffer_.begin() + static_cast<std::ptrdiff_t>(k));
}

std::vector<SingleExperience> ExperienceQueue::Peek(size_t k) const
{
    if (buffer_.empty() || k == 0) {
        return {};
    }

    if (k >= buffer_.size()) {
        return buffer_;
    }

    return std::vector<SingleExperience>(
        buffer_.begin(),
        buffer_.begin() + static_cast<std::ptrdiff_t>(k)
    );
}


// ======================================================
// PlainExperienceQueueController
// ======================================================

std::vector<ExperienceSequence>
PlainExperienceQueueController::ProcessSingleExperience(
    ExperienceQueue& queue,
    const SingleExperience& exp)
{
    anet::ProfileRange r1("PlainExperienceQueueController::ProcessSingleExperience");

    // 単一 Experience をそのまま 1 sequence にする
    ExperienceSequence seq;
    seq.reserve(1);
    seq.push_back(exp);

    return { std::move(seq) };
}


// ======================================================
// NStepExperienceQueueController
// ======================================================

NStepExperienceQueueController::NStepExperienceQueueController(size_t n_step)
    : n_step_(n_step)
{
    ANET_ASSERT(n_step_ > 1);
}

std::vector<ExperienceSequence>
NStepExperienceQueueController::ProcessSingleExperience(ExperienceQueue& queue, const SingleExperience& exp)
{
    anet::ProfileRange r1("NStepExperienceQueueController::ProcessSingleExperience");

    std::vector<ExperienceSequence> out_seq;

    // 新しい experience を queue に追加
    queue.Push(exp);

    // 通常ケース：n_step 分たまったら 1 sequence 吐く
    if (queue.Size() >= n_step_) {
        ExperienceSequence seq = queue.Peek(n_step_);
        out_seq.push_back(std::move(seq));
        queue.Pop(1);
    }

    // エピソード終端処理（done / truncated）
    if (exp.next_state.done || exp.next_state.truncated) {
        // 残っている要素をすべて flush
        while (queue.Size() > 0) {
            size_t k = queue.Size();
            ExperienceSequence seq = queue.Peek(k); // k個のSingleExperience列から一つのReplayExperienceを作る
            out_seq.push_back(std::move(seq));
            queue.Pop(1);    // 1個のExperienceSequenceから1個のReplayExperienceを作る
        }
    }

    return out_seq;
}


// ======================================================
// PlainReplayExperienceBuilder
// ======================================================

ReplayExperience PlainReplayExperienceBuilder::Build(const ExperienceSequence& sequence) const
{
    anet::ProfileRange r1("PlainReplayExperienceBuilder::Build");

    ANET_ASSERT(sequence.size() == 1);

    const auto& exp = sequence[0];

    return ReplayExperience {
        exp.state,
        exp.action,
        exp.reward,
        exp.next_state,
        exp.next_state.done || exp.next_state.truncated,
        1
    };
}


// ======================================================
// NStepReplayExperienceBuilder
// ======================================================

NStepReplayExperienceBuilder::NStepReplayExperienceBuilder(float gamma)
    : gamma_(gamma)
{
    ANET_ASSERT(gamma_ > 0.0f && gamma_ <= 1.0f);
}

ReplayExperience NStepReplayExperienceBuilder::Build(const ExperienceSequence& sequence) const
{
    anet::ProfileRange r1("NStepReplayExperienceBuilder::Build");

    ANET_ASSERT(!sequence.empty());

    const size_t n = sequence.size();

    float G = 0.0f;
    float gamma_pow = 1.0f;
    bool terminal = false;

    for (size_t i = 0; i < n; ++i) {
        const auto& exp = sequence[i];

        G += gamma_pow * exp.reward;
        gamma_pow *= gamma_;

        if (exp.next_state.done || exp.next_state.truncated) {
            terminal = true;
            break;
        }
    }

    const SingleExperience& first = sequence.front();
    const SingleExperience& last = sequence[n - 1];

    return ReplayExperience {
        first.state,        // state
        first.action,       // action
        G,                  // target_value
        last.next_state,    // next_state   TDのブートストラップ状態
        terminal,           // terminal
        static_cast<int>(n) // n_step
    };
}


// ======================================================
// ReplayExperienceStorage
// ======================================================

ReplayExperienceStorage::ReplayExperienceStorage(const EnvSpec& env_spec, int64_t capacity, torch::Device device)
    : device_(device), capacity_(capacity)
    , int64_opt_(torch::TensorOptions().dtype(torch::kInt64).device(device_))
{
    auto state_dim = env_spec.state_spec.CalcFlattenDim();
    auto action_dim = env_spec.action_spec.GetNumActions();
    ANET_ASSERT(state_dim > 0);
    ANET_ASSERT(action_dim > 0);
    ANET_ASSERT(capacity_ > 0);

    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_).pinned_memory(true);
    auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(device_).pinned_memory(true);
    auto b = torch::TensorOptions().dtype(torch::kBool).device(device_).pinned_memory(true);

    states_ = torch::zeros({ capacity_, state_dim }, f32);
    target_values_ = torch::zeros({ capacity_ }, f32);
    next_states_ = torch::zeros({ capacity_, state_dim }, f32);
    terminals_ = torch::zeros({ capacity_ }, b);
    n_steps_ = torch::zeros({ capacity_ }, i64);

    if (env_spec.action_spec.is_discrete) {
        actions_ = torch::zeros({ capacity_ }, i64); // 離散アクションでは1次元かつint64固定
    } else {
        actions_ = torch::zeros({ capacity_, action_dim }, f32);
    }
}

void ReplayExperienceStorage::Push(const ReplayExperience& exp)
{
    anet::ProfileRange r1("ReplayExperienceStorage::Push");

    const int64_t idx = write_index_;

    states_[idx].copy_(exp.state.obs.to(device_));
    actions_[idx].copy_(exp.action.to(device_));
    target_values_[idx] = exp.target_value;
    next_states_[idx].copy_(exp.next_state.obs.to(device_));
    terminals_[idx] = exp.terminal;
    n_steps_[idx] = exp.n_step;

    write_index_ = (write_index_ + 1) % capacity_;
    if (size_ < capacity_) size_++;
}

ExperienceSamples ReplayExperienceStorage::Gather(
    const std::vector<int64_t>& indices, std::optional<torch::Device> out_device) const
{
    anet::ProfileRange r1("ReplayExperienceStorage::Gather");

    // vector→Tensor変換
    auto index_tensor = torch::from_blob(
        const_cast<int64_t*>(indices.data()),{ static_cast<int64_t>(indices.size()) }, int64_opt_).clone();
    ANET_CHECK_DTYPE(index_tensor, torch::kInt64);

    // gather
    auto idx = index_tensor.to(device_);
    ExperienceSamples out {
        states_.index_select(0, idx),           // obs
        actions_.index_select(0, idx),          // actions
        target_values_.index_select(0, idx),    // target_values
        next_states_.index_select(0, idx),      // next_states.obs
        terminals_.index_select(0, idx),        // next_states.terminals
        n_steps_.index_select(0, idx)           // n_steps
    };

    // 必要に応じてdevice転送
    auto dst_device = out_device.value_or(device_);
    if (dst_device != device_) {
        out = out.To(dst_device, true);   // FlattenStates 後段想定
    }

    return out;
}

static std::vector<torch::Tensor> ring_view(
    const torch::Tensor& t, int64_t size, int64_t capacity, int64_t write_index)
{
    using Slice = torch::indexing::Slice;
    std::vector<torch::Tensor> out;
    if (size == 0) return out;

    int64_t head = write_index;
    int64_t tail = (head + capacity - size) % capacity;

    int64_t first_len = std::min(size, capacity - tail);
    if (first_len > 0)
        out.push_back(t.index({ Slice(tail, (tail + first_len)) }));

    int64_t second_len = size - first_len;
    if (second_len > 0)
        out.push_back(t.index({ Slice(0, second_len) }));
    return out;
}

std::optional<std::vector<torch::Tensor>>
ReplayExperienceStorage::GetTensorVector(const std::string& key, int index) const
{
    anet::ProfileRange r1("ReplayExperienceStorage::GetTensorVector");

    if (key == ReplayBuffer::STATE_OBS)
        return ring_view(states_, size_, capacity_, write_index_);
    if (key == ReplayBuffer::ACTION)
        return ring_view(actions_, size_, capacity_, write_index_);
    if (key == ReplayBuffer::REWARD)
        return ring_view(target_values_, size_, capacity_, write_index_);
    if (key == ReplayBuffer::NEXT_STATE_OBS)
        return ring_view(next_states_, size_, capacity_, write_index_);
    if (key == ReplayBuffer::NEXT_STATE_TERMINAL)
        return ring_view(terminals_, size_, capacity_, write_index_);
    if (key == ReplayBuffer::N_STEP)
        return ring_view(n_steps_, size_, capacity_, write_index_);
    return std::nullopt;
}

std::optional<float>
ReplayExperienceStorage::GetScalar(const std::string& key, int index) const
{
    return std::nullopt;
}

std::optional<torch::Tensor>
ReplayExperienceStorage::GetTensor(const std::string& key, int index) const
{
    return std::nullopt;
}


// ======================================================
// UniformReplayExperienceSampler
// ======================================================

UniformReplayExperienceSampler::UniformReplayExperienceSampler(anet::seed_t seed)
    : RandomHolder(seed)
    , opts_(torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
{
}

std::vector<int64_t> UniformReplayExperienceSampler::SampleIndices(
    const ReplayExperienceStorage& storage, int64_t minibatch_size)
{
    anet::ProfileRange r1("UniformReplayExperienceSampler::SampleIndices");

    const int64_t storaget_size = storage.Size();
    ANET_ASSERT(storaget_size > 0);
    ANET_ASSERT(minibatch_size > 0);

    // ---- RNG を使って n 個のインデックスを取得 ----
    std::vector<int64_t> indices(minibatch_size);
    for (int64_t i = 0; i < minibatch_size; ++i)
        indices[i] = rnd_->RandIndex(storaget_size);
    //auto indices = torch::from_blob(buf.data(), { minibatch_size }, opts_).clone();

    return indices;
}


// ======================================================
// DefaultReplayBuffer
// ======================================================

DefaultReplayBuffer::DefaultReplayBuffer(
    const EnvSpec& env_spec, int64_t capacity, int64_t num_envs,
    std::unique_ptr<ExperienceQueueController> queue_controller,
    std::unique_ptr<ReplayExperienceBuilder> replay_exp_builder,
    std::unique_ptr<ReplayExperienceSampler> sampler,
    torch::Device device, bool use_prefetch)
    : num_envs_(num_envs)
    , queues_(num_envs)
    , queue_controller_(std::move(queue_controller))
    , replay_exp_builder_(std::move(replay_exp_builder))
    , sampler_(std::move(sampler))
    , use_prefetch_(use_prefetch)
{
    ANET_ASSERT(num_envs_ > 0);
    storage_ = std::make_unique<ReplayExperienceStorage>(env_spec, capacity, device);
}

void DefaultReplayBuffer::Push(const BatchExperience& batch_exp)
{
    anet::ProfileRange r1("DefaultReplayBuffer::Push1");

    auto exps = batch_exp.ToExperienceList();
    const int64_t N = exps.size();
    ANET_ASSERT(N == num_envs_);

    int64_t i = 0;
    for (auto exp : exps) {
        auto& q = queues_[i];
        auto sequences = queue_controller_->ProcessSingleExperience(q, exp);

        for (const auto& seq : sequences) {
            ReplayExperience re = replay_exp_builder_->Build(seq);
            storage_->Push(re);
        }

        i++;
    }
}

void DefaultReplayBuffer::Push(const std::vector<SingleExperience>& exps)
{
    anet::ProfileRange r1("DefaultReplayBuffer::Push2");

    ANET_ASSERT(static_cast<int64_t>(exps.size()) == num_envs_);

    for (int64_t i = 0; i < num_envs_; ++i) {
        auto& q = queues_[i];
        auto sequences = queue_controller_->ProcessSingleExperience(q, exps[i]);

        for (const auto& seq : sequences) {
            ReplayExperience re = replay_exp_builder_->Build(seq);
            storage_->Push(re);
        }
    }
}

ExperienceSamples DefaultReplayBuffer::sampleInternal(int64_t minibatch_size, torch::Device device) const
{
    anet::ProfileRange r1("DefaultReplayBuffer::sampleInternal");

    ANET_ASSERT(storage_->Size() > 0);

    auto indices = sampler_->SampleIndices(*storage_, minibatch_size);
    return storage_->Gather(indices, device);
}

ExperienceSamples DefaultReplayBuffer::Sample(int64_t minibatch_size, torch::Device device) const
{
    anet::ProfileRange r1("DefaultReplayBuffer::Sample");

    ANET_ASSERT(storage_->Size() > 0);

    if (use_prefetch_) {
        if (!prefetch_cached_) {
            prefetch_result_ = sampleInternal(minibatch_size, device);
            prefetch_cached_ = true;
            return sampleInternal(minibatch_size, device);
        }
        auto result = prefetch_result_;
        prefetch_result_ = sampleInternal(minibatch_size, device);
        return result;
    }

    return sampleInternal(minibatch_size, device);
}

int64_t DefaultReplayBuffer::Size() const
{
    return storage_->Size();
}

std::optional<float> DefaultReplayBuffer::GetScalar(const std::string& key, int index) const
{
    return storage_->GetScalar(key, index);
}

std::optional<torch::Tensor> DefaultReplayBuffer::GetTensor(const std::string& key, int index) const
{
    return storage_->GetTensor(key, index);
}

std::optional<std::vector<torch::Tensor>>
DefaultReplayBuffer::GetTensorVector(const std::string& key, int index) const
{
    return storage_->GetTensorVector(key, index);
}

// ======================================================

ReplayBufferFactory::ReplayBufferFactory(const ReplayBufferConfig& config)
    : config_(config)
{
    ;
}

std::shared_ptr<ReplayBuffer> ReplayBufferFactory::Create(
    const EnvSpec& env_spec, torch::Device device, int batch_size, seed_t seed)
{
    std::unique_ptr<ExperienceQueueController> queue_controller;
    std::unique_ptr<ReplayExperienceBuilder> replay_builder;
    std::unique_ptr<ReplayExperienceSampler> sampler;

    switch (config_.type) {
    case ReplayBufferType::Plain:
        queue_controller = std::make_unique<PlainExperienceQueueController>();
        replay_builder = std::make_unique<PlainReplayExperienceBuilder>();
        sampler = std::make_unique<UniformReplayExperienceSampler>(seed);
        break;

    case ReplayBufferType::NStep:
        ANET_ASSERT(config_.n_step > 1);
        queue_controller = std::make_unique<NStepExperienceQueueController>(config_.n_step);
        replay_builder = std::make_unique<NStepReplayExperienceBuilder>(config_.gamma);
        sampler = std::make_unique<UniformReplayExperienceSampler>(seed);
        break;

    default:
        ANET_ASSERT(false);
    }

    return std::make_shared<DefaultReplayBuffer>(
        env_spec,
        config_.capacity,
        batch_size,
        std::move(queue_controller),
        std::move(replay_builder),
        std::move(sampler),
        device);
}
