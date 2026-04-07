// replay_buffer_impl.cpp

#include "replay_buffer_impl.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include "anet/tensor_check.hpp"
#include "anet/profile.hpp"
#include "anet/log.hpp"
#include "anet/random.hpp"

using namespace anet::rl;
namespace LOG = anet::log;


// ===========================================================================
// Queue
// ===========================================================================

void ExperienceQueue::Push(const QueueRecord& record)
{
    buffer_.push_back(record);
}

void ExperienceQueue::Pop(size_t k)
{
    if (k >= buffer_.size()) {
        buffer_.clear();
    } else {
        buffer_.erase(buffer_.begin(), buffer_.begin() + k);
    }
}

std::vector<QueueRecord> ExperienceQueue::Peek(size_t k) const
{
    size_t count = std::min(k, buffer_.size());
    return std::vector<QueueRecord>(buffer_.begin(), buffer_.begin() + count);
}

size_t ExperienceQueue::Size() const
{
    return buffer_.size();
}

void ExperienceQueue::Clear()
{
    buffer_.clear();
}


// ===========================================================================
// Concrete Controller & Builder
// ===========================================================================

class NStepQueueController : public ExperienceQueueController {
public:
    explicit NStepQueueController(int n_step) : n_step_(n_step)
    {
    }

    std::vector<ExperienceSequence> ExtractSequences(ExperienceQueue& queue) override
    {
        std::vector<ExperienceSequence> sequences;
        while (queue.Size() > 0) {
            auto peeked = queue.Peek(n_step_);
            bool has_terminal = false;
            size_t term_idx = 0;

            // 系列内に終端(done/truncated)があるかチェック
            for (size_t i = 0; i < peeked.size(); ++i) {
                if (peeked[i].done || peeked[i].truncated) {
                    has_terminal = true;
                    term_idx = i;
                    break;
                }
            }

            if (has_terminal) {
                // 終端までで系列を確定し、先頭1要素分だけ進める(Pop1)
                ExperienceSequence seq(peeked.begin(), peeked.begin() + term_idx + 1);
                sequences.push_back(seq);
                queue.Pop(1);
            } else if (peeked.size() == static_cast<size_t>(n_step_)) {
                // N-Step揃ったので確定
                sequences.push_back(peeked);
                queue.Pop(1);
            } else {
                // まだ未来が揃っていないので待機
                break;
            }
        }
        return sequences;
    }
private:
    int n_step_;
};

class DefaultExperienceBuilder : public ReplayExperienceBuilder {
public:
    DefaultExperienceBuilder(float gamma) : gamma_(gamma) {}

    ReplayExperience Build(const ExperienceSequence& sequence) const override
    {
        ReplayExperience exp{};
        exp.actual_n_steps = static_cast<int>(sequence.size());
        exp.terminal = sequence.back().done; // Truncated時はBootstrappingするため terminal=false 扱い
        exp.target_return = 0.0f;

        // 割引報酬和の計算 (逆順)
        for (int i = exp.actual_n_steps - 1; i >= 0; --i) {
            exp.target_return = sequence[i].reward + gamma_ * exp.target_return;
        }
        return exp;
    }
private:
    float gamma_;
};


// ===========================================================================
// Valid Index Manager
// ===========================================================================

ValidIndexManager::ValidIndexManager(int64_t num_envs, int64_t capacity_per_env)
    : num_envs_(num_envs), capacity_per_env_(capacity_per_env)
{
    valid_cursors_.assign(num_envs, 0);
}

void ValidIndexManager::MarkWritten(int64_t env_idx, int64_t time_idx)
{
    // 物理的な書き込みカーソルはStorageが管理するため、ここでは何もしない。
}

void ValidIndexManager::MarkValid(int64_t env_idx, int64_t time_idx)
{
    // N-Step計算が完了し、サンプリング可能になった上限をインクリメント
    valid_cursors_[env_idx]++;
}

torch::Tensor ValidIndexManager::GetValidIndices1D(int stack_count, int unroll_steps) const
{
    std::vector<int64_t> valid_list;
    valid_list.reserve(num_envs_ * capacity_per_env_ / 2);

    // ここで作成される valid_list は strictly monotonically increasing (厳密な昇順) となるため、
    // 後述の PER での std::binary_search を用いた O(log N) 高速判定が保証される
    for (int64_t env = 0; env < num_envs_; ++env) {
        int64_t valid_count = std::min(valid_cursors_[env], capacity_per_env_);
        int64_t write_head = valid_cursors_[env] % capacity_per_env_;

        // バッファが1周していない場合
        if (valid_cursors_[env] < capacity_per_env_) {
            for (int64_t i = stack_count - 1; i <= valid_count - unroll_steps - 1; ++i) {
                valid_list.push_back(env * capacity_per_env_ + i);
            }
        }

        // バッファが1周以上している場合（書き込みヘッド付近が Invalid）
        else {
            for (int64_t i = 0; i < capacity_per_env_; ++i) {
                // カーソル位置(write_head)から、未来(unroll)と過去(stack)の禁止領域を計算
                int64_t dist_to_head = (write_head - i + capacity_per_env_) % capacity_per_env_;

                // 未来(unroll)と過去(stack)の禁止領域を計算
                if (dist_to_head >= unroll_steps && (capacity_per_env_ - dist_to_head) >= stack_count) {
                    valid_list.push_back(env * capacity_per_env_ + i);
                }
            }
        }
    }

    if (valid_list.empty()) {
        return torch::empty({ 0 }, torch::kInt64);
    }
    return torch::tensor(valid_list, torch::kInt64);
}

int64_t ValidIndexManager::GetValidCount() const
{
    int64_t total = 0;
    for (auto c : valid_cursors_) {
        total += std::min(c, capacity_per_env_);
    }
    return total;
}


// ===========================================================================
// ReplayExperienceStorage
// ===========================================================================

ReplayExperienceStorage::ReplayExperienceStorage(int64_t num_envs, int64_t capacity_per_env, const EnvSpec& spec, const ReplayBufferConfig& config, torch::Device device, bool pin_memory)
    : num_envs_(num_envs), capacity_per_env_(capacity_per_env), device_(device)
{
    write_cursors_.assign(num_envs, 0);

    auto options = torch::TensorOptions().device(device_).pinned_memory(pin_memory && device_.is_cpu());

    // Core配列を事前確保
    auto act_shape = spec.action_spec.GetShape();
    act_shape.insert(act_shape.begin(), capacity_per_env_);
    act_shape.insert(act_shape.begin(), num_envs_);

    actions_ = torch::empty(act_shape, options.dtype(spec.action_spec.GetDataType()));
    target_returns_ = torch::empty({ num_envs_, capacity_per_env_ }, options.dtype(torch::kFloat32));
    terminals_ = torch::empty({ num_envs_, capacity_per_env_ }, options.dtype(torch::kBool));
    truncates_ = torch::empty({ num_envs_, capacity_per_env_ }, options.dtype(torch::kBool));
    actual_n_steps_ = torch::empty({ num_envs_, capacity_per_env_ }, options.dtype(torch::kInt64));

    // obs_storage_ と info_storage_ は型の詳細が動的(Dict)なため、初回の Push 時に遅延アロケーションする
}

int64_t ReplayExperienceStorage::Push(int64_t env_idx, const anet::TensorDict& obs, const torch::Tensor& action, const anet::TensorDict& info)
{
    int64_t t = write_cursors_[env_idx] % capacity_per_env_;

    // 遅延アロケーション (初回のみ)
    if (obs_storage_.empty()) {
        for (const auto& kv : obs) {
            auto shape = kv.second.sizes().vec();
            shape.insert(shape.begin(), capacity_per_env_);
            shape.insert(shape.begin(), num_envs_);
            obs_storage_.Set(kv.first, torch::empty(shape, torch::TensorOptions().dtype(kv.second.dtype()).device(device_)));
        }
    }
    if (!info.empty() && info_storage_.empty()) {
        for (const auto& kv : info) {
            auto shape = kv.second.sizes().vec();
            shape.insert(shape.begin(), capacity_per_env_);
            shape.insert(shape.begin(), num_envs_);
            info_storage_.Set(kv.first, torch::empty(shape, torch::TensorOptions().dtype(kv.second.dtype()).device(device_)));
        }
    }

    // 重いテンソルの即時コピー
    for (const auto& kv : obs) {
        obs_storage_.At(kv.first)[env_idx][t].copy_(kv.second);
    }
    for (const auto& kv : info) {
        info_storage_.At(kv.first)[env_idx][t].copy_(kv.second);
    }
    actions_[env_idx][t].copy_(action);

    write_cursors_[env_idx]++;
    return t;
}

void ReplayExperienceStorage::Update(int64_t env_idx, int64_t time_idx, const ReplayExperience& exp)
{
    target_returns_[env_idx][time_idx] = exp.target_return;
    terminals_[env_idx][time_idx] = exp.terminal;
    actual_n_steps_[env_idx][time_idx] = exp.actual_n_steps;
}

void ReplayExperienceStorage::PushTerminalDummy(int64_t env_idx, const anet::TensorDict& terminal_obs)
{
    // 終端状態用のダミーステップ。Actionや報酬は無効値を入れる
    torch::Tensor dummy_action = torch::zeros_like(actions_[env_idx][0]);
    anet::TensorDict dummy_info; // infoも空
    int64_t t = Push(env_idx, terminal_obs, dummy_action, dummy_info);

    // ダミーの即時 Valid 化用メタデータ
    target_returns_[env_idx][t] = 0.0f;
    terminals_[env_idx][t] = true;
    actual_n_steps_[env_idx][t] = 0;
}

std::optional<float> ReplayExperienceStorage::GetScalar(const std::string& key, int64_t index) const
{
    return std::nullopt;
}

std::optional<torch::Tensor> ReplayExperienceStorage::GetTensor(const std::string& key, int64_t index) const
{
    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>> ReplayExperienceStorage::GetTensorVector(const std::string& key, int64_t index) const
{
    return std::nullopt;
}


// ===========================================================================
// SumTree
// ===========================================================================

SumTree::SumTree(int64_t capacity) : capacity_(capacity)
{
    tree_.assign(2 * capacity_ - 1, 0.0f);
}

void SumTree::Update(int64_t index, float priority)
{
    int64_t tree_idx = index + capacity_ - 1;
    float change = priority - tree_[tree_idx];
    tree_[tree_idx] = priority;
    while (tree_idx != 0) {
        tree_idx = (tree_idx - 1) / 2;
        tree_[tree_idx] += change;
    }
}

float SumTree::GetTotalPriority() const
{
    return tree_[0];
}

int64_t SumTree::Retrieve(float value) const
{
    int64_t tree_idx = 0;
    while (tree_idx < capacity_ - 1) {
        int64_t left = 2 * tree_idx + 1;
        int64_t right = left + 1;
        if (value <= tree_[left]) {
            tree_idx = left;
        } else {
            value -= tree_[left];
            tree_idx = right;
        }
    }
    return tree_idx - capacity_ + 1;
}

float SumTree::GetPriority(int64_t index) const
{
    return tree_[index + capacity_ - 1];
}


// ===========================================================================
// Sampler (具象クラス)
// ===========================================================================

namespace {
    // --- 内部ヘルパー関数 ---
    // リングバッファの物理境界を考慮して、安全にスライス(時間軸)を切り出す関数
    torch::Tensor RingSlice(const torch::Tensor& t, int64_t env_idx, int64_t logical_start, int64_t length, int64_t capacity, bool squeeze_time)
    {
        // C++の負数の剰余算を安全に行う
        int64_t start = (logical_start % capacity + capacity) % capacity;
        int64_t end = start + length;

        torch::Tensor res;
        if (end <= capacity) {
            // バッファ境界をまたがない場合（通常スライス）
            res = t[env_idx].slice(0, start, end);
        } else {
            // バッファ境界をまたぐ場合（末尾と先頭を結合）
            int64_t over = end - capacity;
            res = torch::cat({ t[env_idx].slice(0, start, capacity), t[env_idx].slice(0, 0, over) }, 0);
        }

        // stack_count=1 や unroll_steps=0 の場合、時間軸次元を消して [B, C, H, W] のようにする
        if (squeeze_time) {
            res = res.squeeze(0);
        }
        return res;
    }

    // TensorDict 版のリングバッファスライス
    anet::TensorDict RingSliceDict(const anet::TensorDict& dict, int64_t env_idx, int64_t logical_start, int64_t length, int64_t capacity, bool squeeze_time)
    {
        anet::TensorDict res;
        if (dict.empty()) return res;
        for (const auto& kv : dict) {
            res.Set(kv.first, RingSlice(kv.second, env_idx, logical_start, length, capacity, squeeze_time));
        }
        return res;
    }
} // namespace


class UniformSampler : public ReplayExperienceSampler, public anet::RandomHolder {
public:
    UniformSampler(std::optional<anet::seed_t> seed)
        : anet::RandomHolder(seed)
        , gen_(rnd_->GetTorchGenerator(torch::kCPU))
        , opt_long_(torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))    // idxはCPU固定
        , opt_float_(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)) // idxはCPU固定
    {
    }

    IndexSampleResult SampleIndices(int64_t batch_size, const torch::Tensor& valid_indices_1d, float beta) override
    {
        int64_t valid_count = valid_indices_1d.size(0);
        auto rand_idx = torch::randint(0, valid_count, { batch_size }, gen_, opt_long_);
        auto indices = valid_indices_1d.index_select(0, rand_idx);
        auto ones = torch::ones({ batch_size }, opt_float_);

        return { indices, ones / valid_count, ones };
    }
private:
    torch::Generator gen_;
    torch::TensorOptions opt_long_;
    torch::TensorOptions opt_float_;
};

class PrioritizedSampler : public ReplayExperienceSampler, public ReplayPriorityController, public anet::RandomHolder {
public:
    PrioritizedSampler(int64_t capacity, float alpha, float initial_priority, std::optional<anet::seed_t> seed)
        : anet::RandomHolder(seed)
        , tree_(capacity)
        , alpha_(alpha)
        , max_prio_(initial_priority)
        , gen_(rnd_->GetTorchGenerator(torch::kCPU))
        , opt_long_(torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
        , opt_float_(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
    {
    }

    IndexSampleResult SampleIndices(int64_t batch_size, const torch::Tensor& valid_indices_1d, float beta) override
    {
        int64_t valid_count = valid_indices_1d.size(0);
        const int64_t* valid_ptr = valid_indices_1d.data_ptr<int64_t>();

        std::vector<int64_t> sampled_indices(batch_size);
        std::vector<float> sampled_probs(batch_size);
        std::vector<float> sampled_weights(batch_size);

        float total_prio = tree_.GetTotalPriority();

        int64_t b = 0;
        int max_attempts = batch_size * 10;
        int attempts = 0;

        // valid_indices_1d は昇順ソートが保証されているため、std::binary_search による O(1)~O(log V) 判定が可能
        while (b < batch_size && attempts < max_attempts) {
            attempts++;
            float r = torch::rand({ 1 }, gen_, opt_float_).item<float>() * total_prio;
            int64_t idx = tree_.Retrieve(r);

            bool is_valid = std::binary_search(valid_ptr, valid_ptr + valid_count, idx);
            if (is_valid) {
                float p = tree_.GetPriority(idx);
                if (p <= 0.0f) continue; // 安全装置

                sampled_indices[b] = idx;
                float prob = p / total_prio;
                sampled_probs[b] = prob;

                float weight = std::pow(valid_count * prob, -beta);
                sampled_weights[b] = weight;
                b++;
            }
        }

        // フェイルセーフ（滅多に起きないが、ツリーが空に近い極初期など）
        if (b < batch_size) {
            LOG::warn() << "PER Rejection Sampling failed to fill batch. Falling back to uniform.";
            auto rand_idx = torch::randint(0, valid_count, { batch_size - b }, gen_, opt_long_);
            auto rand_acc = rand_idx.accessor<int64_t, 1>();
            for (int64_t i = 0; i < batch_size - b; ++i) {
                int64_t idx = valid_ptr[rand_acc[i]];
                sampled_indices[b + i] = idx;
                sampled_probs[b + i] = 1.0f / valid_count;
                sampled_weights[b + i] = 1.0f;
            }
        }

        auto idx_device = valid_indices_1d.device();
        auto indices_t = torch::tensor(sampled_indices, opt_long_).to(idx_device);
        auto probs_t = torch::tensor(sampled_probs, opt_float_).to(idx_device);
        auto weights_t = torch::tensor(sampled_weights, opt_float_).to(idx_device);

        weights_t /= weights_t.max(); // 正規化

        return { indices_t, probs_t, weights_t };
    }

    void UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities) override
    {
        for (size_t i = 0; i < indices.size(); ++i) {
            float p = priorities[i];

            if (p < 0.0f) {
                // 特殊フラグ (-1.0f): 新規データが必ず1回はサンプリングされるよう、現在の「最大優先度」を割り当てる
                tree_.Update(indices[i], max_prio_);
            } else if (p == 0.0f) {
                // 特殊フラグ (0.0f): 上書きに伴う無効化
                tree_.Update(indices[i], 0.0f);
            } else {
                // 通常の更新
                float adjusted_p = std::pow(p, alpha_);
                tree_.Update(indices[i], adjusted_p);
                if (adjusted_p > 0.0f) {
                    max_prio_ = std::max(max_prio_, adjusted_p);
                }
            }
        }
    }
private:
    SumTree tree_;
    const float alpha_;
    const torch::Generator gen_;
    const torch::TensorOptions opt_long_;
    const torch::TensorOptions opt_float_;
    float max_prio_;
};


// ===========================================================================
// Extractor (具象クラス)
// ===========================================================================

class DefaultSampleExtractor : public ExperienceSampleExtractor {
public:
    void ExtractSamples(ExperienceSamples& out, const ReplayExperienceStorage& storage, const IndexSampleResult& idx_result, int stack_count, int unroll_steps) const override
    {
		anet::ProfileRange r("DefaultSampleExtractor::ExtractSamples");

        int64_t B = idx_result.indices.size(0);
        auto indices_acc = idx_result.indices.accessor<int64_t, 1>();

        int64_t cap = storage.GetTargetReturns().size(1); // capacity_per_env

        // 次元スカラー化のフラグ (DQNのような1ステップ構成の場合は時間次元をSqueezeする)
        int unroll_len = (unroll_steps > 0) ? unroll_steps : 1;
        bool squeeze_unroll = (unroll_steps == 0);
        bool squeeze_stack = (stack_count == 1);

        // テンソル構築用の配列
        std::vector<torch::Tensor> batch_actions, batch_returns, batch_terminals, batch_truncates, batch_actual_n;
        std::vector<anet::TensorDict> batch_obs, batch_next_obs, batch_info;

        /// @todo [Performance] 現在はバッチサイズ(B)回数分のループで C++ 側からスライスと torch::stack を行っている。
        /// GPU上でストレージを持つ場合、Pythonの `tensor[batch_indices, time_indices]` のように
        /// 高度なインデックス演算 (Advanced Indexing) を用いてベクトル化された一括抽出にリファクタリングすることで更なる学習FPSの向上が見込める

        for (int64_t b = 0; b < B; ++b) {
            int64_t idx1d = indices_acc[b];
            int64_t env_idx = idx1d / cap;
            int64_t time_idx = idx1d % cap;
            int64_t actual_n = storage.GetActualNSteps()[env_idx][time_idx].item<int64_t>();

            // 未来方向 (Unroll) のスライス抽出 ---
            batch_actions.push_back(RingSlice(storage.GetActions(), env_idx, time_idx, unroll_len, cap, squeeze_unroll));
            batch_returns.push_back(RingSlice(storage.GetTargetReturns(), env_idx, time_idx, unroll_len, cap, squeeze_unroll));
            batch_terminals.push_back(RingSlice(storage.GetTerminals(), env_idx, time_idx, unroll_len, cap, squeeze_unroll));
            batch_truncates.push_back(RingSlice(storage.GetTruncates(), env_idx, time_idx, unroll_len, cap, squeeze_unroll));
            batch_actual_n.push_back(storage.GetActualNSteps()[env_idx][time_idx]);

            // MuZero用などの固有情報 (存在する場合のみ)
            if (!storage.GetInfo().empty()) {
                batch_info.push_back(RingSliceDict(storage.GetInfo(), env_idx, time_idx, unroll_len, cap, squeeze_unroll));
            }

            // 過去方向 (Frame Stacking) のスライス抽出
            int64_t obs_start = time_idx - stack_count + 1;
            batch_obs.push_back(RingSliceDict(storage.GetObs(), env_idx, obs_start, stack_count, cap, squeeze_stack));

            // N-Step先 (NextState) のスライス抽出
            int64_t next_obs_end = time_idx + actual_n + 1;
            int64_t next_obs_start = next_obs_end - stack_count;
            batch_next_obs.push_back(RingSliceDict(storage.GetObs(), env_idx, next_obs_start, stack_count, cap, squeeze_stack));
        }

        // バッチスタック
        out.actions = torch::stack(batch_actions, 0);
        out.target_returns = torch::stack(batch_returns, 0);
        out.next_state.terminals = torch::stack(batch_terminals, 0);
        out.next_state.truncates = torch::stack(batch_truncates, 0);
        out.n_steps = torch::stack(batch_actual_n, 0);

        out.obs = anet::TensorDict::Stack(batch_obs, 0);
        out.next_state.next_obs = anet::TensorDict::Stack(batch_next_obs, 0);
        if (!batch_info.empty()) {
            out.info = anet::TensorDict::Stack(batch_info, 0);
        }

        // --- PER等のメタデータ引き継ぎ ---
        out.indices = idx_result.indices;
        out.is_weights = idx_result.is_weights;
    }
};


// ===========================================================================
// DefaultReplayBuffer ( Facade )
// ===========================================================================

DefaultReplayBuffer::DefaultReplayBuffer(
    const ReplayBufferConfig& config, const EnvSpec& env_spec, int64_t num_envs,
    std::unique_ptr<ExperienceQueueController> queue_controller,
    std::unique_ptr<ReplayExperienceBuilder> builder,
    std::shared_ptr<ReplayExperienceSampler> sampler,
    std::shared_ptr<ReplayPriorityController> prio_controller,
    std::shared_ptr<ExperienceSampleExtractor> extractor,
    torch::Device device, bool pin_memory)
    : config_(config)
    , num_envs_(num_envs)
    , queue_controller_(std::move(queue_controller))
    , builder_(std::move(builder))
    , sampler_(sampler)
    , prio_controller_(prio_controller)
    , extractor_(extractor)
{
    capacity_per_env_ = config_.capacity / num_envs_;
    queues_.resize(num_envs_);

    storage_ = std::make_unique<ReplayExperienceStorage>(num_envs_, capacity_per_env_, env_spec, config_, device, pin_memory);
    index_manager_ = std::make_unique<ValidIndexManager>(num_envs_, capacity_per_env_);
}

void DefaultReplayBuffer::Push(const BatchExperience& batch)
{
    // 事前に action の info を取得しておく
    anet::TensorDict action_info = batch.action->GetInfo();

    for (int64_t b = 0; b < num_envs_; ++b) {
        // Step 1: Storage に重いデータを即時 Push (重複排除)

        // TensorDict に追加した operator[]() を使って単一バッチ要素のDictを構築
        anet::TensorDict single_obs = batch.state.obs[b];
        anet::TensorDict single_info = action_info.empty() ? anet::TensorDict() : action_info[b];

        int64_t time_idx = storage_->Push(b, single_obs, batch.action->GetAction()[b], single_info);
        index_manager_->MarkWritten(b, time_idx);

        // 上書きされたインデックスの優先度をゼロ(0.0f)にしてサンプリング対象から除外
        if (prio_controller_) {
            prio_controller_->UpdatePriorities({ b * capacity_per_env_ + time_idx }, { 0.0f });
        }

        // Truncatedのパラドックス対策 (ダミーステップの挿入)
        if (batch.next_state.truncated[b].item<bool>()) {
            anet::TensorDict terminal_obs = batch.next_state.obs[b];
            storage_->PushTerminalDummy(b, terminal_obs);
            index_manager_->MarkWritten(b, time_idx + 1);

            if (prio_controller_) {
                prio_controller_->UpdatePriorities({ b * capacity_per_env_ + time_idx + 1 }, { 0.0f });
            }

            // ダミーステップもQueueに入れることでカーソルの永遠のズレを防止
            QueueRecord dummy_rec;
            dummy_rec.time_idx = time_idx + 1;
            dummy_rec.reward = 0.0f;
            dummy_rec.done = true;
            dummy_rec.truncated = false;
            queues_[b].Push(dummy_rec);
        }

        // Step 2: Queue に軽量メタデータを Push
        QueueRecord rec;
        rec.time_idx = time_idx;
        rec.reward = batch.reward[b].item<float>();
        rec.done = batch.next_state.done[b].item<bool>();
        rec.truncated = batch.next_state.truncated[b].item<bool>();
        queues_[b].Push(rec);

        // Step 3: N-Step計算 と Valid 化の駆動
        ProcessQueue(b);
    }
}

void DefaultReplayBuffer::ProcessQueue(int64_t env_idx)
{
    auto sequences = queue_controller_->ExtractSequences(queues_[env_idx]);

    std::vector<int64_t> newly_valid;
    std::vector<float> init_prios;

    for (const auto& seq : sequences) {
        // Builderで割引報酬和を計算
        ReplayExperience exp = builder_->Build(seq);

        // 先頭の time_idx を取り出し、Storageを上書き(Update)
        int64_t time_idx = seq.front().time_idx;
        storage_->Update(env_idx, time_idx, exp);

        // 完全にサンプリング可能になったので封印解除
        index_manager_->MarkValid(env_idx, time_idx);

        newly_valid.push_back(env_idx * capacity_per_env_ + time_idx);
        init_prios.push_back(-1.0f); // 初期優先度を割り当てるための特殊フラグ
    }

    //  N-Step計算が完了し、安全になったデータをTreeに登録（初期優先度付与）
    if (prio_controller_ && !newly_valid.empty()) {
        prio_controller_->UpdatePriorities(newly_valid, init_prios);
    }
}

void DefaultReplayBuffer::Sample(ExperienceSamples& out_samples, int64_t minibatch_size, float beta) const
{
    auto valid_1d = index_manager_->GetValidIndices1D(config_.stack_count, config_.muzero.unroll_steps);
    ANET_ASSERT_MSG(valid_1d.size(0) >= minibatch_size, "Not enough valid samples in ReplayBuffer. size=" << valid_1d.size(0) << " minibatch_size=" << minibatch_size);

    auto idx_result = sampler_->SampleIndices(minibatch_size, valid_1d, beta);
    extractor_->ExtractSamples(out_samples, *storage_, idx_result, config_.stack_count, config_.muzero.unroll_steps);
}

int64_t DefaultReplayBuffer::Size() const
{
    return index_manager_->GetValidCount();
}

void DefaultReplayBuffer::UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities)
{
    if (prio_controller_) {
        prio_controller_->UpdatePriorities(indices, priorities);
    }
}

std::optional<float> DefaultReplayBuffer::GetScalar(const std::string& key, int64_t index) const
{
    return std::nullopt;
}

std::optional<torch::Tensor> DefaultReplayBuffer::GetTensor(const std::string& key, int64_t index) const
{
    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>> DefaultReplayBuffer::GetTensorVector(const std::string& key, int64_t index) const
{
    return std::nullopt;
}


// ===========================================================================
// Factory
// ===========================================================================

std::shared_ptr<ReplayBuffer> anet::rl::CreateReplayBuffer(
    const ReplayBufferConfig& config, const EnvSpec& env_spec, int64_t num_envs, torch::Device storage_device, bool pin_memory, std::optional<uint64_t> seed)
{
    // capacity の割り切れ補正
    int64_t capacity_per_env = config.capacity / num_envs;
    if (capacity_per_env * num_envs != config.capacity) {
        LOG::info() << "ReplayBuffer capacity adjusted to be divisible by num_envs."
            << " config.capacity=" << config.capacity << " actual_capacity=" << capacity_per_env * num_envs;
    }

    // 内部コンポーネントの構築
    auto queue_controller = std::make_unique<NStepQueueController>(config.n_step);
    auto builder = std::make_unique<DefaultExperienceBuilder>(config.gamma);

    std::shared_ptr<ReplayExperienceSampler> sampler;
    std::shared_ptr<ReplayPriorityController> prio;
    if (config.sampler_type == ReplaySamplerType::UNIFORM) {
        sampler = std::make_shared<UniformSampler>(seed);
        prio = nullptr;
    } else {
        auto per = std::make_shared<PrioritizedSampler>(config.capacity, config.per_alpha, config.per_initial_priority, seed);
        sampler = per;
        prio = per;
    }

    auto extractor = std::make_shared<DefaultSampleExtractor>();

    return std::make_shared<DefaultReplayBuffer>(
        config, env_spec, num_envs, std::move(queue_controller), std::move(builder), sampler, prio, extractor, storage_device, pin_memory);
}
