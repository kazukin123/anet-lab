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

    // 現在のwrite_index_に書き込み
    states_[idx].copy_(exp.state.obs.to(device_));
    actions_[idx].copy_(exp.action.to(device_));
    target_values_[idx] = exp.target_value;
    next_states_[idx].copy_(exp.next_state.obs.to(device_));
    terminals_[idx] = exp.terminal;
    n_steps_[idx] = exp.n_step;

    // write_index_を更新
    write_index_ = (write_index_ + 1) % capacity_;
    bool overwrite = false;
    if (size_ < capacity_) size_++;
    else overwrite = true;

    // イベント通知
    StorageWriteEvent ev {
        overwrite ? StorageWriteEventCode::Overwrite : StorageWriteEventCode::Append, idx,
    };
    Notify(ev);
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
        n_steps_.index_select(0, idx),          // n_steps
        idx,                // indices (優先度更新用)
        torch::Tensor(),    // sampling_prob (Placeholder: 呼び出し元で設定)
        torch::Tensor(),    // is_weights (Placeholder: 呼び出し元で設定)
    };

    // 必要に応じてdevice転送
    auto dst_device = out_device.value_or(device_);
    if (dst_device != device_) {
        out = out.To(dst_device, true);   // FlattenStates 後段想定
    }

    return out;
}

void ReplayExperienceStorage::AttachEventHandler(EventHandler handler)
{
    event_handlers_.push_back(std::move(handler));
}

void ReplayExperienceStorage::Notify(const StorageWriteEvent& ev)
{
    for (const auto& handler : event_handlers_) {
        handler(ev);
    }
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

IndexSampleResult UniformReplayExperienceSampler::SampleIndices(
    const ReplayExperienceStorage& storage, int64_t minibatch_size, float beta)
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

    return { indices };
}


// ======================================================
// SumTree
// ======================================================

SumTree::SumTree(int64_t capacity)
    : capacity_(capacity)
{
    ANET_ASSERT_MSG(capacity_ > 0, "SumTree capacity must be positive");
    ANET_ASSERT_MSG(capacity_ <= static_cast<int64_t>(std::numeric_limits<size_t>::max() / 2),
        "SumTree capacity is too large for size_t");

    const size_t tree_size = static_cast<size_t>(capacity_) * 2;
    ANET_ASSERT_MSG(tree_size <= tree_.max_size(),"SumTree capacity exceeds vector::max_size()");
 
    tree_.assign(tree_size, 0.0f);
}

void SumTree::Update(int64_t index, float value)
{
    ANET_ASSERT_MSG(index >= 0 && index < capacity_, "SumTree::Update index out of range");

    // 対応する葉ノードに値を書き込む
    size_t node = static_cast<size_t>(index + capacity_);
    tree_[node] = value;

    // 親ノード方向へ累積和を更新する
    node >>= 1;
    while (node >= 1) {
        tree_[node] = tree_[node << 1] + tree_[(node << 1) | 1];
        node >>= 1;
    }
}

float SumTree::Get(int64_t index) const
{
    ANET_ASSERT_MSG(index >= 0 && index < capacity_, "SumTree::Get index out of range");

    // 対応する葉ノードの値を取得する
    return tree_[static_cast<size_t>(index + capacity_)];
}

int64_t SumTree::Sample(float value) const
{
    // @todo 分散低減のための区間分割サンプリング（stratified sampling）を検討

    // ルートノードから探索を開始
    int64_t node = 1;

    // 葉ノードに到達するまで木を降下する
    while (node < capacity_) {
        int64_t left = node << 1;
        float left_sum = tree_[static_cast<size_t>(left)];

        // 左部分木の累積和と比較し、対応する子ノードを選択する（右の子は left | 1）
        if (value <= left_sum) {
            node = left;
        } else {
            value -= left_sum;
            node = left | 1;
        }
    }

    // 葉ノードのインデックスを論理インデックスに変換する
    return node - capacity_;
}


// ======================================================
// PrioritizedReplayExperienceSampler
// ======================================================

PrioritizedReplayExperienceManager::PrioritizedReplayExperienceManager(int64_t capacity,float alpha, anet::seed_t seed)
    : RandomHolder(seed), sum_tree_(capacity), alpha_(alpha)
{
    ANET_ASSERT_MSG(alpha_ >= 0.0f, "alpha must be non-negative");
}

IndexSampleResult PrioritizedReplayExperienceManager::SampleIndices(
    const ReplayExperienceStorage& storage, int64_t minibatch_size, float beta)
{
    ANET_ASSERT_MSG(minibatch_size > 0, "minibatch_size must be positive");
    ANET_ASSERT_MSG(beta >= 0.0f && beta <= 1.0f, "beta must be in [0, 1]");

    IndexSampleResult result{ minibatch_size };

    // 優先度の総和（サンプリング空間の上限値）を取得
    const float total_priority = sum_tree_.Total();
    ANET_ASSERT_MSG(total_priority > 0.0f, "SumTree total priority must be positive");

    // 有効な experience 数（IS weight 計算に使用）
    const int64_t valid_size = storage.Size();
    ANET_ASSERT_MSG(valid_size > 0, "ReplayExperienceStorage is empty");

    float max_weight = 0.0f;
    // minibatch_size分回す
    for (int64_t i = 0; i < minibatch_size; ++i) {
        // @todo 分散低減のための区間分割サンプリング（stratified sampling）を検討

        // [0, total_priority) 上で一様乱数を生成し、累積和探索で index を選択
        const float u = rnd_->Uniform(0.0f, total_priority);
        const int64_t index = sum_tree_.Sample(u);

        // サンプリングされた論理インデックスを記録
        result.indices[i] = index;

        // 当該 index が選択される確率 P(i) を計算
        const float p = sum_tree_.Get(index) / total_priority;
        result.sampling_prob[i] = p;

        // Importance Sampling weight を計算（選択確率の偏り補正）
        float w = std::pow(1.0f / (static_cast<float>(valid_size) * p), beta);
        result.is_weights[i] = w;
        max_weight = std::max(max_weight, w);
    }

    // 勾配スケールの過大化を防ぐため、IS weight を最大値で正規化
    if (max_weight > 0.0f) {
        for (float& w : result.is_weights) {
            w /= max_weight;
        }
    }

    return result;
}

void PrioritizedReplayExperienceManager::UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities)
{
    ANET_ASSERT_MSG(indices.size() == priorities.size(),
        "indices and priorities size mismatch");

    const int64_t n = static_cast<int64_t>(indices.size());

    for (int64_t i = 0; i < n; ++i) {
        const int64_t index = indices[i];
        const float priority = priorities[i];

        ANET_ASSERT_MSG(priority > 0.0f, "priority must be positive");

        // α を適用した priority を SumTree に反映する
        // （ReplayStorage の論理 index と 1:1 に対応）
        sum_tree_.Update(index, std::pow(priority, alpha_));
    }
}

std::optional<float> PrioritizedReplayExperienceManager::GetScalar(
    const std::string& key, int index) const
{
    // priority/total : 全体のみ
    if (key == ReplayBuffer::PER_TOTAL) {
        return sum_tree_.Total();
    }

    return std::nullopt;
}

std::optional<torch::Tensor> PrioritizedReplayExperienceManager::GetTensor(const std::string& key, int index) const
{
    if (key == ReplayBuffer::PER_VALUES) {
        if (index < 0 || index >= sum_tree_.Capacity())
            return std::nullopt;
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        return torch::tensor(sum_tree_.Get(index), opts);
    }

    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>> PrioritizedReplayExperienceManager::GetTensorVector(
    const std::string& key, int index) const
{
    if (key == ReplayBuffer::PER_DIST) {
	    const int64_t capacity = sum_tree_.Capacity();
	    const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

        std::vector<float> buf;
        buf.reserve(capacity);
        for (int64_t i = 0; i < capacity; ++i) {
            auto prio = sum_tree_.Get(i);
            if (prio != 0.0)    /// @todo prioゼロ＝未設定ではなく明示的な判定を入れる？per_epsがゼロ出ない限りこれでも問題ないけど。
                buf.push_back(prio);
        }
        auto t = torch::from_blob(buf.data(), { static_cast<int64_t>(buf.size()), 1 }, opts).clone();
        return std::vector<torch::Tensor>{ std::move(t) };
    }

    return std::nullopt;
}


// ======================================================
// DefaultReplayBuffer
// ======================================================

DefaultReplayBuffer::DefaultReplayBuffer(
    const EnvSpec& env_spec, int64_t capacity, int64_t num_envs,
    std::unique_ptr<ExperienceQueueController> queue_controller,
    std::unique_ptr<ReplayExperienceBuilder> replay_exp_builder,
    std::shared_ptr<ReplayExperienceSampler> sampler,
    std::shared_ptr<ReplayPriorityController> prio_controller,
    torch::Device device, float initial_priority, bool use_prefetch)
    : num_envs_(num_envs)
    , queues_(num_envs)
    , queue_controller_(std::move(queue_controller))
    , replay_exp_builder_(std::move(replay_exp_builder))
    , sampler_(sampler)
    , prio_controller_(prio_controller)
    , use_prefetch_(use_prefetch)
    , initial_priority_(initial_priority)
{
    ANET_ASSERT(num_envs_ > 0);

    // Storage生成
    storage_ = std::make_unique<ReplayExperienceStorage>(env_spec, capacity, device);

    // Storage→prio_controllerの同期用イベント登録
    if (prio_controller_ != nullptr) {
        storage_->AttachEventHandler(
            [this](const StorageWriteEvent& ev)
            {
                prio_controller_->UpdatePriorities({ ev.index }, { initial_priority_ });
            }
        );
    }
}

void DefaultReplayBuffer::Push(const BatchExperience& batch_exp)
{
    anet::ProfileRange r1("DefaultReplayBuffer::Push1");

    auto exps = batch_exp.ToExperienceList();
    const int64_t N = exps.size();
    ANET_ASSERT(N == num_envs_);

    int64_t i = 0;
    for (auto exp : exps) {
        auto& queue = queues_[i];
        auto sequences = queue_controller_->ProcessSingleExperience(queue, exp);

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

ExperienceSamples DefaultReplayBuffer::sampleInternal(int64_t minibatch_size, torch::Device device, float beta) const
{
    anet::ProfileRange r1("DefaultReplayBuffer::sampleInternal");

    ANET_ASSERT(storage_->Size() > 0);

    // Samplerからインデックスと重みを取得
    auto indices_result = sampler_->SampleIndices(*storage_, minibatch_size, beta);

    // Storageからデータを収集 (この時点では prob, weights は空)
    auto samples = storage_->Gather(indices_result.indices, device);

    // Samplerが計算した prob, weights を Tensor化してセット
    if (!indices_result.sampling_prob.empty()) {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

        // vector -> Tensor
        auto prob_tensor = torch::from_blob(
            const_cast<float*>(indices_result.sampling_prob.data()),
            { minibatch_size }, opts).clone();

        auto weights_tensor = torch::from_blob(
            const_cast<float*>(indices_result.is_weights.data()),
            { minibatch_size }, opts).clone();

        /// @todo StorageではなくExperienceSamplesでpinned_memory出来るようにする（Storage全体をpinnedすると容量次第でマシン負荷に繋がる）

        // データのデバイスに合わせて転送
        samples.sampling_prob = prob_tensor.to(device);
        samples.is_weights = weights_tensor.to(device);
    }

    return samples;
}

ExperienceSamples DefaultReplayBuffer::Sample(int64_t minibatch_size, torch::Device device, float beta) const
{
    anet::ProfileRange r1("DefaultReplayBuffer::Sample");

    ANET_ASSERT(storage_->Size() > 0);

    if (use_prefetch_) {
        if (!prefetch_cached_) {
            prefetch_result_ = sampleInternal(minibatch_size, device, beta);
            prefetch_cached_ = true;
            return sampleInternal(minibatch_size, device, beta);
        }
        auto result = prefetch_result_;
        prefetch_result_ = sampleInternal(minibatch_size, device, beta);
        return result;
    }

    return sampleInternal(minibatch_size, device, beta);
}

int64_t DefaultReplayBuffer::Size() const
{
    return storage_->Size();
}

void DefaultReplayBuffer::UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities)
{
    if (prio_controller_ != nullptr)
        prio_controller_->UpdatePriorities(indices, priorities);
}

std::optional<float> DefaultReplayBuffer::GetScalar(const std::string& key, int index) const
{
    auto storage_ret = storage_->GetScalar(key, index);
    if (storage_ret.has_value()) return *storage_ret;
    if (prio_controller_ != nullptr) return prio_controller_->GetScalar(key, index);
    return std::nullopt;
}

std::optional<torch::Tensor> DefaultReplayBuffer::GetTensor(const std::string& key, int index) const
{
    auto storage_ret = storage_->GetTensor(key, index);
    if (storage_ret.has_value()) return *storage_ret;
    if (prio_controller_ != nullptr) return prio_controller_->GetTensor(key, index);
    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>>
DefaultReplayBuffer::GetTensorVector(const std::string& key, int index) const
{
    auto storage_ret = storage_->GetTensorVector(key, index);
    if (storage_ret.has_value()) return *storage_ret;
    if (prio_controller_ != nullptr) return prio_controller_->GetTensorVector(key, index);
    return std::nullopt;
}

// ======================================================

ReplayBufferFactory::ReplayBufferFactory(const ReplayBufferConfig& config)
    : config_(config)
{
    ;
}

std::shared_ptr<ReplayBuffer>
ReplayBufferFactory::Create(const EnvSpec& env_spec, torch::Device device, int batch_size, seed_t seed)
{
    ANET_ASSERT_MSG(config_.capacity > 0, "ReplayBuffer capacity must be positive");
    ANET_ASSERT_MSG(batch_size > 0, "batch_size must be positive");

    // -------------------------------------------------------------
    // ReplayExperienceBuilder
    // -------------------------------------------------------------
    std::unique_ptr<ExperienceQueueController> queue_controller;
    std::unique_ptr<ReplayExperienceBuilder> replay_exp_builder;

    switch (config_.type) {
    case ReplayBuilderType::PLAIN:
        queue_controller = std::make_unique<PlainExperienceQueueController>();
        replay_exp_builder = std::make_unique<PlainReplayExperienceBuilder>();
        break;
    case ReplayBuilderType::NSTEP:
        ANET_ASSERT_MSG(config_.n_step > 0, "n_step must be positive");
        queue_controller = std::make_unique<NStepExperienceQueueController>(config_.n_step);
        replay_exp_builder = std::make_unique<NStepReplayExperienceBuilder>(config_.gamma);
        break;
    default:
        ANET_ASSERT_MSG(false, "Unknown ReplayBuilderType");
        break;
    }

    // -------------------------------------------------------------
    // Sampler / PriorityController
    // -------------------------------------------------------------
    std::shared_ptr<ReplayExperienceSampler> sampler;
    std::shared_ptr<ReplayPriorityController> prio_controller;

    switch (config_.sampler_type) {
    case ReplaySamplerType::UNIFORM:
        sampler = std::make_shared<UniformReplayExperienceSampler>(seed);
        prio_controller = nullptr;
        break;
    case ReplaySamplerType::PRIOTIZED:
    {
        auto per_sampler = std::make_shared<PrioritizedReplayExperienceManager>(config_.capacity, config_.per_alpha, seed);
        sampler = per_sampler;
        prio_controller = per_sampler;
        break;
    }
    default:
        ANET_ASSERT_MSG(false, "Unknown ReplaySamplerType");
        break;
    }


    // -------------------------------------------------------------
    // ReplayBuffer 本体生成
    // -------------------------------------------------------------
    auto buffer = std::make_shared<DefaultReplayBuffer>(
        env_spec,
        config_.capacity,
        batch_size,
        std::move(queue_controller),
        std::move(replay_exp_builder),
        sampler,
        prio_controller,
        device,
        config_.per_initial_priority);

    return buffer;
}
