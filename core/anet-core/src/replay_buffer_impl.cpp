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
        exp.next_state.done,
        1,
        0
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

	// N-STEP TARGET VALUE 計算ループ
    for (size_t i = 0; i < n; ++i) {
        const auto& exp = sequence[i];

        G += gamma_pow * exp.reward;
        gamma_pow *= gamma_;

        if (exp.next_state.done) {
            terminal = true; // エピソード終了
            break;
        }
        if (exp.next_state.truncated) {
            terminal = false; // 時間切れはterminalじゃない
            break; // でもループ（N-step）はここで打ち切る
        }
        //if (exp.next_state.done || exp.next_state.truncated) {
        //    terminal = true;
        //    break;
        //}
    }

    const SingleExperience& first = sequence.front();
    const SingleExperience& last = sequence[n - 1];

    return ReplayExperience {
        first.state,        // state
        first.action,       // action
        G,                  // target_value
        last.next_state,    // next_state   TDのブートストラップ状態
        terminal,           // terminal
        static_cast<int>(n), // n_step
        0
    };
}


// ======================================================
// ReplayExperienceStorage
// ======================================================

ReplayExperienceStorage::ReplayExperienceStorage(const EnvSpec& env_spec, int64_t capacity, int64_t num_envs, torch::Device device)
    : device_(device), capacity_(capacity)
    , int64_opt_(torch::TensorOptions().dtype(torch::kInt64).device(device_))
{
    auto state_dim = env_spec.state_spec.CalcFlattenDim();
    auto action_dim = env_spec.action_spec.GetNumActions();
    ANET_ASSERT(state_dim > 0);
    ANET_ASSERT(action_dim > 0);
    ANET_ASSERT(capacity_ > 0);

    //auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_).pinned_memory(true);
    //auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(device_).pinned_memory(true);
    //auto b = torch::TensorOptions().dtype(torch::kBool).device(device_).pinned_memory(true);
    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device_).pinned_memory(false);
    auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(device_).pinned_memory(false);
    auto b = torch::TensorOptions().dtype(torch::kBool).device(device_).pinned_memory(false);

    /// @todo 複数環境のデータを分離して2Dバッファ化

    states_ = torch::zeros({ capacity_, state_dim }, f32);
    target_values_ = torch::zeros({ capacity_ }, f32);
    next_states_ = torch::zeros({ capacity_, state_dim }, f32);
    terminals_ = torch::zeros({ capacity_ }, b);
    n_steps_ = torch::zeros({ capacity_ }, i64);
    episode_starts_ = torch::zeros({ capacity_ }, b);

    prev_indices_ = torch::full({ capacity_ }, -1, i64);
    last_env_indices_.resize(num_envs, -1);

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
    next_states_[idx].copy_(exp.next_state.obs.to(device_));
    target_values_[idx].fill_(exp.target_value);
    terminals_[idx].fill_(exp.terminal);
    n_steps_[idx].fill_(exp.n_step);
    episode_starts_[idx].fill_(exp.state.episode_start);

    // リンクリストの構築：このデータの1つ前は、同じ環境の直近の書き込みインデックス
    prev_indices_[idx] = last_env_indices_[exp.env_index];

    // 最新のインデックスを更新
    last_env_indices_[exp.env_index] = idx;

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
    ANET_ASSERT_DTYPE(index_tensor, torch::kInt64);

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
ReplayExperienceStorage::GetTensorVector(const std::string& key, int64_t index) const
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
ReplayExperienceStorage::GetScalar(const std::string& key, int64_t index) const
{
    return std::nullopt;
}

std::optional<torch::Tensor>
ReplayExperienceStorage::GetTensor(const std::string& key, int64_t index) const
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

    const int64_t storaget_size = storage.GetSize();
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

    // サイズ 2Nの非再帰セグメント木を使用しているため、capacity が 2 のべき乗である必要はない

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
    /// @todo 分散低減のための区間分割サンプリング（stratified sampling）を検討

    int64_t node = 1;

    while (node < capacity_) {
        int64_t left = node << 1;
        float left_sum = tree_[static_cast<size_t>(left)];

        bool go_right = (value > left_sum);
        value -= go_right ? left_sum : 0.0f;
        node = left | static_cast<int64_t>(go_right);
    }

    int64_t data_idx = node - capacity_;
    return std::clamp<int64_t>(data_idx, 0, capacity_ - 1);
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
    const int64_t valid_size = storage.GetSize();
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
        const float priority_val = sum_tree_.Get(index);
        const float p = (total_priority > 0.0f) ? (priority_val / total_priority) : 0.0f;   //total_priority が 0 の場合のガード

        // Importance Sampling weight を計算（選択確率の偏り補正）
        const float safe_p = std::max(p, 1e-10f);
        const float w = std::pow(1.0f / (static_cast<float>(valid_size) * safe_p), beta);
        ANET_ASSERT(!std::isinf(w));
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

std::optional<float> PrioritizedReplayExperienceManager::GetScalar(const std::string& key, int64_t index) const
{
    // priority/total : 全体のみ
    if (key == ReplayBuffer::PER_TOTAL) {
        return sum_tree_.Total();
    }

    return std::nullopt;
}

std::optional<torch::Tensor> PrioritizedReplayExperienceManager::GetTensor(const std::string& key, int64_t index) const
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
    const std::string& key, int64_t index) const
{
    if (key == ReplayBuffer::PER_DIST) {
        const int64_t size = index;   /// @todo 無理やりindexをsize想定に
	    const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

        std::vector<float> buf;
        buf.resize(size);
        for (int64_t i = 0; i < size; ++i) {
            auto prio = sum_tree_.Get(i);
            buf[i] = prio;
        }
        auto t = torch::from_blob(buf.data(), { static_cast<int64_t>(buf.size()), 1 }, opts).clone();
        return std::vector<torch::Tensor>{ std::move(t) };
    }

    return std::nullopt;
}


// ======================================================
// ReplayExperienceStateStacker
// ======================================================

ReplayExperienceStateStacker::ReplayExperienceStateStacker(int stack_count, const std::vector<int64_t>& state_shape, torch::Device device)
    : stack_count_(stack_count)
    , state_shape_(state_shape)
    , device_(device)
{
    stacked_indices_opts_ = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
}

ExperienceSamples ReplayExperienceStateStacker::SampleBatch(
        const ReplayExperienceStorage& storage, const IndexSampleResult& index_result,
        int64_t minibatch_size, torch::Device target_device)
{
    anet::ProfileRange r("ReplayExperienceStateStacker::SampleBatch");

    // -----------------------------------------------------------------------
    // 準備
    // -----------------------------------------------------------------------
    anet::ProfileRange r1("ReplayExperienceStateStacker::SampleBatch.prepare");
    const std::vector<int64_t>& indices_vec = index_result.indices;
    const int64_t* indices_ptr = indices_vec.data();

    // Storage情報取得
    const int64_t write_idx = storage.GetWriteIndex();
    const int64_t capacity = storage.GetCapacity();
    const int64_t size = storage.GetSize();

    // EpisodeStarts (CPU Tensor) のポインタ取得
    const auto ep_tensor = storage.GetEpisodeStarts();
    const bool* ep_ptr = ep_tensor.data_ptr<bool>();

    // PrevIndices (CPU Tensor) のポインタ取得
    const auto prev_indices_tensor = storage.GetPrevIndices();
    const int64_t* prev_ptr = prev_indices_tensor.data_ptr<int64_t>();

    // 結果格納用インデックス (Flatten: B * S)
    torch::Tensor stacked_indices = torch::empty({ minibatch_size * stack_count_ }, stacked_indices_opts_);
    int64_t* out_ptr = stacked_indices.data_ptr<int64_t>();

    // -----------------------------------------------------------------------
    // 時系列を過去に遡りながら境界判定してstacked_indicesを埋める
    // -----------------------------------------------------------------------
    anet::ProfileRange r2("ReplayExperienceStateStacker::SampleBatch.idx", r1);
    for (int64_t b = 0; b < minibatch_size; ++b) {
        int64_t current_idx = indices_ptr[b];

        // 書き込み先頭ポインタ [t-(S-1), ..., t]
        int64_t* batch_head_ptr = out_ptr + (b * stack_count_);

        bool hit_boundary = false;
        int64_t padding_idx = -1;
        int64_t target_idx = current_idx; // k=0 の初期値

        // 最新 (k=0、Samplingされたidx) から 過去 (k=S-1) へ
        for (int k = 0; k < stack_count_; ++k) {
            int write_pos = stack_count_ - 1 - k;

            // エピソード開始に到達済みの場合、そのidxでパディングして終わり
            if (hit_boundary) {
                batch_head_ptr[write_pos] = padding_idx;
                continue;
            }

            // 過去フレームへ遡る (k=0の時はcurrent_idxをそのまま使う)
            if (k > 0) {
                int64_t old_target = target_idx; // 遡る前のインデックスを保持
                target_idx = prev_ptr[target_idx];

                bool is_valid_range = false;
                if (target_idx != -1) {
                    // write_idx から見た「データの古さ（age）」を計算
                    int64_t age_old = (write_idx - 1 - old_target + capacity) % capacity;
                    int64_t age_new = (write_idx - 1 - target_idx + capacity) % capacity;

                    // 過去のデータは、必ず現在のデータより古くなければならない (age_new > age_old)
                    // もし age_new が小さい場合、それはリングバッファを周回して上書きされた「新しいデータ」を意味する
                    is_valid_range = (age_new < size) && (age_new > age_old);
                }

                // 有効範囲外（まだデータがない、または周回して消えた、あるいは記録の始点）
                if (!is_valid_range) {
                    hit_boundary = true;
                    // padding_idx は更新せず、前回の有効値(または初期値-1)を使う
                    if (padding_idx == -1) padding_idx = current_idx; // 念のためのフォールバック
                    batch_head_ptr[write_pos] = padding_idx;
                    continue;
                }
            }

            // エピソード開始判定
            if (ep_ptr[target_idx]) {
                // エピソード開始に到達した場合、以降同じidxでパディングするためにidx保存
                hit_boundary = true;
                padding_idx = target_idx;
                batch_head_ptr[write_pos] = target_idx;
            } else {
                batch_head_ptr[write_pos] = target_idx;
            }
        }
    }

    // -----------------------------------------------------------------------
    // データ取得 (Gather) & Reshape
    // -----------------------------------------------------------------------
    anet::ProfileRange r3("ReplayExperienceStateStacker::SampleBatch.gather", r2);

    // Stacked Indices をデバイスへ転送
    auto stacked_indices_dev = stacked_indices.to(device_);

    // State (Stacked) : (B * S, state_dim...)
    auto flat_states = storage.GetStates().index_select(0, stacked_indices_dev);

    // Next State (Stacked) : (B * S, state_dim...)
    auto flat_next_states = storage.GetNextStates().index_select(0, stacked_indices_dev);

    // State出力用の出力形状情報を生成
    std::vector<int64_t> out_shape;
    out_shape.push_back(minibatch_size);
    out_shape.push_back(stack_count_);
    for (auto d : state_shape_) out_shape.push_back(d);

    // Reshape： (B * S, state_dim...) -> (B, S, state_dim...)
    const auto stacked_states = flat_states.view(out_shape);
    const auto stacked_next_obs = flat_next_states.view(out_shape);

    // その他 (Actions, TargetValues, Terminals)はindices (vector) を Tensor化してデバイス転送
    const auto indices_dev = torch::tensor(indices_vec, torch::TensorOptions().dtype(torch::kInt64)).to(device_);
    const auto actions = storage.GetActions().index_select(0, indices_dev);
    const auto target_values = storage.GetTargetValues().index_select(0, indices_dev);
    const auto terminals = storage.GetTerminals().index_select(0, indices_dev);
    const auto n_steps = storage.GetNSteps().index_select(0, indices_dev);

    // 結果を生成
    anet::ProfileRange r4("ReplayExperienceStateStacker::SampleBatch.result", r3);
    ExperienceSamples samples {
        stacked_states,    // obs
        actions,           // actions
        target_values,     // target_values
        {                  // next_states
            stacked_next_obs, // next_states.obs
            terminals,        // next_states.terminals   
        },
        n_steps,           // n_steps
        indices_dev,       // indices
        (!index_result.sampling_prob.empty()) ? // sampling_prob
            torch::tensor(index_result.sampling_prob, torch::TensorOptions().dtype(torch::kFloat32)).to(device_) : torch::Tensor(),
        (!index_result.is_weights.empty()) ?    // is_weights
            torch::tensor(index_result.is_weights, torch::TensorOptions().dtype(torch::kFloat32)).to(device_) : torch::Tensor(),
    };

    // device転送
    samples = samples.To(target_device, true);

    return samples;
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
    std::shared_ptr<ReplayExperienceStacker> stacker,
    torch::Device device, float initial_priority, bool use_prefetch)
    : num_envs_(num_envs)
    , queues_(num_envs)
    , queue_controller_(std::move(queue_controller))
    , replay_exp_builder_(std::move(replay_exp_builder))
    , sampler_(sampler)
    , prio_controller_(prio_controller)
	, stacker_(stacker)
    , use_prefetch_(use_prefetch)
    , initial_priority_(initial_priority)
{
    ANET_ASSERT(num_envs_ > 0);

    // Storage生成
    storage_ = std::make_unique<ReplayExperienceStorage>(env_spec, capacity, num_envs, device);

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
    anet::ProfileRange r("DefaultReplayBuffer::Push1");

    anet::ProfileRange r1("DefaultReplayBuffer::Push1.expand");
    auto exps = batch_exp.ToExperienceList();
    const int64_t N = exps.size();
    ANET_ASSERT(N == num_envs_);

    anet::ProfileRange r2("DefaultReplayBuffer::Push1.loop", r1);
    Push(exps);
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
            re.env_index = i;
            storage_->Push(re);
        }
    }
}

ExperienceSamples DefaultReplayBuffer::sampleInternal(int64_t minibatch_size, torch::Device device, float beta) const
{
    anet::ProfileRange r1("DefaultReplayBuffer::sampleInternal");
    torch::NoGradGuard ng;

    ANET_ASSERT(storage_->GetSize() > 0);

    // Samplerからインデックスと重みを取得
    auto indices_result = sampler_->SampleIndices(*storage_, minibatch_size, beta);

    ExperienceSamples samples;

    if (stacker_ == nullptr) {
        // Storageからデータを収集 (この時点では prob, weights は空)
        samples = storage_->Gather(indices_result.indices, device);

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
    } else {
		// Stackerでスタッキング
		samples = stacker_->SampleBatch(*storage_, indices_result, minibatch_size, device);
    }

    return samples;
}

ExperienceSamples DefaultReplayBuffer::Sample(int64_t minibatch_size, torch::Device device, float beta) const
{
    anet::ProfileRange r1("DefaultReplayBuffer::Sample");

    ANET_ASSERT(storage_->GetSize() > 0);

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
    return storage_->GetSize();
}

void DefaultReplayBuffer::UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities)
{
    if (prio_controller_ != nullptr)
        prio_controller_->UpdatePriorities(indices, priorities);
}

std::optional<float> DefaultReplayBuffer::GetScalar(const std::string& key, int64_t index) const
{
    auto storage_ret = storage_->GetScalar(key, index);
    if (storage_ret.has_value()) return *storage_ret;
    if (prio_controller_ != nullptr) return prio_controller_->GetScalar(key, index);
    return std::nullopt;
}

std::optional<torch::Tensor> DefaultReplayBuffer::GetTensor(const std::string& key, int64_t index) const
{
    auto storage_ret = storage_->GetTensor(key, index);
    if (storage_ret.has_value()) return *storage_ret;
    if (prio_controller_ != nullptr) return prio_controller_->GetTensor(key, index);
    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>>
DefaultReplayBuffer::GetTensorVector(const std::string& key, int64_t index) const
{
    auto storage_ret = storage_->GetTensorVector(key, index);
    if (storage_ret.has_value()) return *storage_ret;
    if (prio_controller_ != nullptr) return prio_controller_->GetTensorVector(key, storage_->GetSize());   // 無理やり
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
    // Stacker
    // -------------------------------------------------------------
    std::shared_ptr<ReplayExperienceStacker> stacker = nullptr;
    if (config_.use_stacker) {
        int stack_count = config_.stack_count;
        ANET_CHECK_MSG(stack_count > 0, "stack_count must be greater than 0");
        const auto& state_shape = env_spec.state_spec.shape;
        stacker = std::make_shared<ReplayExperienceStateStacker>(stack_count, state_shape, device);
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
        stacker,
        device,
        config_.per_initial_priority);

    return buffer;
}
