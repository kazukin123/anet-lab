// replay_buffer_impl.cpp

#include "replay_buffer_impl.hpp"
#include <cmath>
#include <algorithm>
#include <future>
#include <optional>
#include <utility>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include "anet/metrics_logger.hpp"
#include "anet/tensor_check.hpp"
#include "anet/profile.hpp"
#include "anet/log.hpp"
#include "anet/random.hpp"
#include "anet/thread.hpp"

using namespace anet::rl;
namespace LOG = anet::log;


/* ==============================================================================
 * [設計仕様] 時間軸の方向とエピソード境界処理のメカニズム
 * ==============================================================================
 *
 * ■ 1. 時間軸の方向と役割
 * ------------------------------------------------------------------------------
 * - Frame Stacking : 【過去方向 (Past)】
 * 　エージェントに「動き」や「速度」を認識させるため、現在から過去へ遡って観測(Obs)を重ねる。
 * - N-Step Return  : 【未来方向 (Future)】
 *  TD誤差（Target Q）をより正確に計算するため、現在から未来へ進んで得られた報酬を累積する。
 * - Unroll Steps   : 【未来方向 (Future)】
 * 役割: MuZero(RNN)等の系列学習のために、現在から未来へ向かって連続するステップを展開・抽出する。
 *
 *
 * ■ 2. エピソード境界の処理仕様
 * ------------------------------------------------------------------------------
 * 【エピソード開始時 (t=0 付近)】 -> 過去(Stack)の不足
 * - 症状: 過去のフレームが存在しないため、Stacking が物理的に不可能。
 * - 解決: Extractor サンプリング時に「一番古い利用可能なフレーム（t=0等）」を
 * 必要な回数だけ複製（コピーパディング）して補完する。
 *
 * 【エピソード終了時 (Done / Truncated)】 -> 未来(N-Step)の不足
 * - 症状: 未来のステップが存在しないため、N-Step分の報酬累積や未来状態の取得が不可能。
 * - 解決 (3段構え):
 * 1. 早期精算 (Flush) : N歩先を待たずに、キューに残留している未確定ステップを強制Valid化する。
 * 2. ダミーステップ   : 終了状態の「次」にダミーステップ(is_dummy=true)を挿入し、参照エラーを防ぐ。
 * 3. 未来価値の遮断   : ダミーには terminals=true フラグを立てる。
 * Target Q = Reward + γ^N * Q(Next) * (1.0 - terminals)
 * の数式により、終端を越えた未来の価値(Q)を数学的に完全にゼロにする。
 *
 * ■ 3. サンプリング有効判定 (Valid 条件) の大原則
 * ------------------------------------------------------------------------------
 * - 条件: 「未来方向 (N-Step / Unroll) が確定した瞬間」に Valid とする。
 * - 禁忌: 「過去方向 (Stacking) の物理的データが揃うまで待つ」というロジックは入れてはならない。
 * （これをやると、エピソード開始直後の重要な数ステップの経験が永遠にサンプリングされず消失する）
 * ============================================================================== */

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
        exp.target_return = 0.0f;
        exp.terminal = false;

        int n = static_cast<int>(sequence.size());

        // シーケンスの末尾がダミーなら、それは実データではないので長さに含めない
        if (sequence.back().is_dummy) {
            n -= 1;
            // ダミー起因（＝元のステップがTruncated）なので、ブートストラップを継続(terminal=false)
            exp.terminal = false;
        } else {
            // 本当のゲームオーバー
            exp.terminal = sequence.back().done;
        }

        exp.actual_n_steps = n;

        // 割引報酬和の計算 (逆順)
        for (int i = n - 1; i >= 0; --i) {
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
    write_cursors_.assign(num_envs, 0);
    is_dummy_.assign(num_envs * capacity_per_env, false);
}

void ValidIndexManager::MarkWritten(int64_t env_idx, int64_t time_idx)
{
    is_dummy_[env_idx * capacity_per_env_ + time_idx] = false;
}

void ValidIndexManager::MarkValid(int64_t env_idx)
{
    // N-Step計算が完了し、サンプリング可能になった上限をインクリメント
    valid_cursors_[env_idx]++;
}

void ValidIndexManager::MarkDummy(int64_t env_idx, int64_t time_idx)
{
    is_dummy_[env_idx * capacity_per_env_ + time_idx] = true; 
}

void ValidIndexManager::AdvanceWriteCursor(int64_t env_idx)
{
    write_cursors_[env_idx]++;
}

torch::Tensor ValidIndexManager::GetValidIndices1D(int stack_count, int unroll_steps, int n_step) const
{
    ANET_PROFILE_FUNC();

    std::vector<int64_t> valid_list;
    valid_list.reserve(num_envs_ * capacity_per_env_);

    for (int64_t env = 0; env < num_envs_; ++env) {
        ForEachSampleableIndex(env, stack_count, unroll_steps, n_step, [&](int64_t idx1d) {
            valid_list.push_back(idx1d);
        });
    }

    if (valid_list.empty()) return torch::empty({ 0 }, torch::kInt64);
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

int64_t ValidIndexManager::GetSampleableCount(int stack_count, int unroll_steps, int n_step) const
{
    ANET_PROFILE_FUNC();

    int64_t total = 0;
    for (int64_t env = 0; env < num_envs_; ++env) {
        ForEachSampleableIndex(env, stack_count, unroll_steps, n_step, [&](int64_t) {
            ++total;
        });
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
    actual_n_steps_ = torch::empty({ num_envs_, capacity_per_env_ }, options.dtype(torch::kInt64));

    target_returns_.fill_(0.0f);
    terminals_.fill_(true);
    actual_n_steps_.fill_(0);

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

void ReplayExperienceStorage::DumpToLog() const
{
    LOG::info() << "=== ReplayExperienceStorage Dump ===";
    for (int64_t e = 0; e < num_envs_; ++e) {
        LOG::info() << "Env " << e << " WriteCursor=" << write_cursors_[e];
        for (int64_t t = 0; t < capacity_per_env_; ++t) {
            float ret = target_returns_[e][t].item<float>();
            bool term = terminals_[e][t].item<bool>();
            int64_t n = actual_n_steps_[e][t].item<int64_t>();

            std::string obs_str = "";
            for (const auto& kv : obs_storage_) {
                float val = -999.0f; // 未初期化時のダミー
                if (kv.second.numel() > 0) {
                    val = kv.second[e][t].contiguous().view({ -1 })[0].item<float>();
                }
                obs_str += kv.first + ":" + std::to_string(val) + " ";
            }
            std::string act_str;
            if (actions_[e][t].dim() == 0)
                act_str = std::to_string(actions_[e][t].item<float>());
            else
                anet::ToString(actions_[e][t]);

            LOG::info() << "  [idx=" << t << "] ret=" << ret
                << " term=" << term << " n_steps=" << n
                << " obs={" << obs_str << "}"
                << " act={" << act_str << "}";
        }
    }
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
    //ANET_PROFILE_FUNC();

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
    //ANET_PROFILE_FUNC();

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
    anet::TensorDict RingSliceDict(
        const anet::TensorDict& dict, int64_t env_idx,
        int64_t valid_start, int64_t valid_len, int64_t pad_len,
        int64_t capacity, bool squeeze_time, const std::vector<std::string>& stack_keys)
    {
        anet::TensorDict res;
        if (dict.empty()) return res;

        for (const auto& kv : dict) {
            // stack_keys が指定されている場合、対象外のKeyは「最新の1フレームのみ」にする
            bool is_stacked = true;
            if (!stack_keys.empty()) {
                auto it = std::find(stack_keys.begin(), stack_keys.end(), kv.first);
                is_stacked = (it != stack_keys.end());
            }

            if (!is_stacked) {
                // Stack対象外: パディングは関係なく、常に一番未来の1フレームだけを取得
                int64_t latest_idx = valid_start + valid_len - 1;
                res.Set(kv.first, RingSlice(kv.second, env_idx, latest_idx, 1, capacity, true));
            } else {
                // Stack対象: まず安全な区間（valid_len）だけをスライス
                auto valid_tensor = RingSlice(kv.second, env_idx, valid_start, valid_len, capacity, false);

                // パディングが必要な場合、最古のフレーム(インデックス0)を複製して結合
                if (pad_len > 0) {
                    auto first_frame = valid_tensor[0].unsqueeze(0); // [1, C, H, W]
                    auto sizes = first_frame.sizes().vec();
                    sizes[0] = pad_len;
                    auto pad_tensor = first_frame.expand(sizes);     // [pad_len, C, H, W] (メモリ追加割当なしで高速)

                    // 過去方向(パディング) ＋ 現在方向(Valid) の順で結合
                    valid_tensor = torch::cat({ pad_tensor, valid_tensor }, /*dim=*/0);
                }

                if (squeeze_time && valid_tensor.size(0) == 1) {
                    valid_tensor = valid_tensor.squeeze(0);
                }
                res.Set(kv.first, valid_tensor);
            }
        }
        return res;
    }

    torch::Tensor FlattenRows(torch::Tensor tensor)
    {
        if (!tensor.defined()) return tensor;
        if (tensor.dim() == 0) return tensor.reshape({ 1, 1 });
        if (tensor.dim() == 1) return tensor.reshape({ tensor.size(0), 1 });
        if (tensor.dim() > 2) return tensor.flatten(1);
        return tensor;
    }

    torch::Tensor ToUnifiedRows(const anet::TensorDict& obs)
    {
        return FlattenRows(anet::rl::ToUnifiedObservation(obs));
    }

    torch::Tensor SelectRowIfRequested(torch::Tensor tensor, int64_t index)
    {
        if (index < 0 || !tensor.defined()) return tensor;
        if (tensor.size(0) <= index) return torch::Tensor();
        return tensor[index];
    }

    struct StorageTensorVectorKey {
        enum class Kind {
            kStateObs,
            kAction,
            kTargetReturn,
            kNextStateObs,
            kNextStateTerminal,
            kNStep,
        };

        Kind kind;
        std::string obs_subkey;
    };

    std::optional<std::string> ParseObservationSubKey(const std::string& key, const char* base_key)
    {
        const std::string base(base_key);
        if (key == base) return std::string();

        const std::string prefix = base + ".";
        if (key.rfind(prefix, 0) != 0) return std::nullopt;

        auto subkey = key.substr(prefix.size());
        if (subkey.empty()) {
            ANET_SYSTEM_ERROR("ReplayBuffer observation key requires a subkey after '" << prefix << "'.");
        }
        return subkey;
    }

    std::optional<StorageTensorVectorKey> ParseStorageTensorVectorKey(const std::string& key)
    {
        if (auto subkey = ParseObservationSubKey(key, ReplayBuffer::STATE_OBS)) {
            return StorageTensorVectorKey{ StorageTensorVectorKey::Kind::kStateObs, *subkey };
        }
        if (auto subkey = ParseObservationSubKey(key, ReplayBuffer::NEXT_STATE_OBS)) {
            return StorageTensorVectorKey{ StorageTensorVectorKey::Kind::kNextStateObs, *subkey };
        }
        if (key == ReplayBuffer::ACTION) {
            return StorageTensorVectorKey{ StorageTensorVectorKey::Kind::kAction, std::string() };
        }
        if (key == ReplayBuffer::TARGET_RETURN) {
            return StorageTensorVectorKey{ StorageTensorVectorKey::Kind::kTargetReturn, std::string() };
        }
        if (key == ReplayBuffer::NEXT_STATE_TERMINAL) {
            return StorageTensorVectorKey{ StorageTensorVectorKey::Kind::kNextStateTerminal, std::string() };
        }
        if (key == ReplayBuffer::N_STEP) {
            return StorageTensorVectorKey{ StorageTensorVectorKey::Kind::kNStep, std::string() };
        }
        return std::nullopt;
    }

    torch::Tensor GatherFlatRowsRaw(const torch::Tensor& storage, const torch::Tensor& indices)
    {
        if (!storage.defined()) return torch::Tensor();
        if (indices.numel() == 0) return torch::Tensor();
        ANET_ASSERT_MSG(storage.dim() >= 2, "ReplayBuffer storage tensor must have [env, time, ...] layout: " << anet::ToDefString(storage));

        std::vector<int64_t> flat_shape;
        flat_shape.reserve(static_cast<size_t>(storage.dim() - 1));
        flat_shape.push_back(storage.size(0) * storage.size(1));
        for (int64_t dim = 2; dim < storage.dim(); ++dim) {
            flat_shape.push_back(storage.size(dim));
        }

        auto gather_indices = indices.to(
            torch::TensorOptions().dtype(torch::kInt64).device(storage.device()),
            /*non_blocking=*/true);
        return storage.reshape(flat_shape).index_select(0, gather_indices);
    }

    torch::Tensor GatherFlatRows(const torch::Tensor& storage, const torch::Tensor& indices)
    {
        return FlattenRows(GatherFlatRowsRaw(storage, indices));
    }

    torch::Tensor GatherObservationRows(
    	const anet::TensorDict& obs_storage, const torch::Tensor& indices, const std::string& obs_subkey)
    {
        if (obs_storage.empty()) return torch::Tensor();

        if (!obs_subkey.empty()) {
            auto opt_tensor = obs_storage.Get(obs_subkey);
            if (!opt_tensor.has_value()) {
                ANET_SYSTEM_ERROR("ReplayBuffer observation storage key '" << obs_subkey << "' was not found.");
            }
            return FlattenRows(GatherFlatRowsRaw(*opt_tensor, indices).to(torch::kFloat32));
        }

        anet::TensorDict obs_rows;
        for (const auto& kv : obs_storage) {
            obs_rows.Set(kv.first, GatherFlatRowsRaw(kv.second, indices));
        }
        return ToUnifiedRows(obs_rows);
    }

    torch::Tensor MakeNextFlatIndices(const torch::Tensor& indices, const torch::Tensor& actual_n_steps, int64_t capacity_per_env)
    {
        auto actual_rows = GatherFlatRowsRaw(actual_n_steps, indices).to(torch::kCPU).contiguous();
        auto indices_cpu = indices.to(torch::kCPU).contiguous();
        auto actual_acc = actual_rows.accessor<int64_t, 1>();
        auto index_acc = indices_cpu.accessor<int64_t, 1>();

        std::vector<int64_t> next_indices(static_cast<size_t>(indices_cpu.size(0)));
        for (int64_t i = 0; i < indices_cpu.size(0); ++i) {
            const int64_t flat_index = index_acc[i];
            const int64_t env_idx = flat_index / capacity_per_env;
            const int64_t time_idx = flat_index % capacity_per_env;
            next_indices[static_cast<size_t>(i)] = env_idx * capacity_per_env + ((time_idx + actual_acc[i]) % capacity_per_env);
        }

        return torch::tensor(next_indices, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    }

    bool HasSameKeys(const anet::TensorDict& lhs, const anet::TensorDict& rhs)
    {
        if (lhs.size() != rhs.size()) return false;
        for (const auto& kv : rhs) {
            if (!lhs.Contains(kv.first)) return false;
        }
        return true;
    }

    void SetTensorDictBatchItem(anet::TensorDict& scratch, const anet::TensorDict& batch_dict, int64_t batch_index)
    {
        if (!HasSameKeys(scratch, batch_dict)) {
            scratch = anet::TensorDict();
        }
        for (const auto& kv : batch_dict) {
            scratch.Set(kv.first, kv.second[batch_index]);
        }
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

#if 1
    IndexSampleResult SampleIndices(int64_t batch_size, const torch::Tensor& valid_indices_1d, float beta) override
    {
        ANET_PROFILE_FUNC();

        int64_t valid_count = valid_indices_1d.size(0);
        auto rand_idx = torch::randint(0, valid_count, { batch_size }, gen_, opt_long_);
        auto indices = valid_indices_1d.index_select(0, rand_idx);
        auto ones = torch::ones({ batch_size }, opt_float_);

#if 0
        {
            // デバッグ用メトリクス計算
            torch::NoGradGuard no_grad; // 勾配計算から切り離す

            auto sorted_indices = std::get<0>(torch::sort(indices));
            auto unique_indices = std::get<0>(torch::unique_consecutive(sorted_indices));
            float unique_count = static_cast<float>(unique_indices.size(0));
            float unique_ratio = unique_count / static_cast<float>(batch_size);

            float std_val = indices.to(torch::kFloat32).std().item<float>();
            float std_norm = std_val / static_cast<float>(valid_count);

            static step_t log_step = 0;
            anet::MetricsLogger::Instance()->LogScalar("99_debug/rb_unique_ratio", log_step, unique_ratio);
            anet::MetricsLogger::Instance()->LogScalar("99_debug/rb_std_norm", log_step, std_norm);
            log_step++;

        }
#endif

        return { indices, ones / valid_count, ones };
    }
#else
    // V1互換
    IndexSampleResult SampleIndices(int64_t batch_size, const torch::Tensor& valid_indices_1d_2, float beta) override
    {
        auto shuffled_valid_indices = valid_indices_1d_2.index_select(0, torch::randperm(valid_indices_1d_2.size(0), opt_long_));

        int64_t valid_count = shuffled_valid_indices.size(0);

        std::vector<int64_t> cpp_rand_idx(batch_size);
        for (int64_t i = 0; i < batch_size; ++i) {
            // V1と同じ乱数生成器を使用する。
            // ※V1で storage_size から引いていたのと同じ挙動にするため、
            //   ここでは valid_count を上限として引く
            cpp_rand_idx[i] = rnd_->RandIndex(valid_count);
        }

        // 生成したC++の配列を torch::Tensor に変換 (メモリをコピーして独立させる)
        auto rand_idx_tensor = torch::from_blob(
            cpp_rand_idx.data(), { batch_size }, torch::TensorOptions().dtype(torch::kInt64)
        ).clone();

        // デバイスを合わせる（valid_indices_1d がGPU上にある場合のため）
        rand_idx_tensor = rand_idx_tensor.to(shuffled_valid_indices.device());

        // テンソルを使って物理インデックスを抽出
        auto indices = shuffled_valid_indices.index_select(0, rand_idx_tensor);
        auto ones = torch::ones({ batch_size }, opt_float_);

        {
            // デバッグ用メトリクス計算
            torch::NoGradGuard no_grad; // 勾配計算から切り離す

            auto sorted_indices = std::get<0>(torch::sort(indices));
            auto unique_indices = std::get<0>(torch::unique_consecutive(sorted_indices));
            float unique_count = static_cast<float>(unique_indices.size(0));
            float unique_ratio = unique_count / static_cast<float>(batch_size);

            float std_val = indices.to(torch::kFloat32).std().item<float>();
            float std_norm = std_val / static_cast<float>(valid_count);

			static step_t log_step = 0;
            anet::MetricsLogger::Instance()->LogScalar("99_debug/rb_unique_ratio", log_step, unique_ratio);
			anet::MetricsLogger::Instance()->LogScalar("99_debug/rb_std_norm", log_step, std_norm);
            log_step++;

        }

        return { indices, ones / valid_count, ones };
    }
#endif
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
        , max_prio_(initial_priority < 0.0f ? 1.0f : std::pow(initial_priority, alpha))
        , initial_priority_(initial_priority)
        , gen_(rnd_->GetTorchGenerator(torch::kCPU))
        , opt_long_(torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
        , opt_float_(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
    {
    }

#if 1
    IndexSampleResult SampleIndices(int64_t batch_size, const torch::Tensor& valid_indices_1d, float beta) override
    {
        ANET_PROFILE_FUNC();

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
#else
    // V1互換
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
            float r = rnd_->Uniform(0.0f, total_prio);

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
            for (int64_t i = 0; i < batch_size - b; ++i) {
                // ★ フェイルセーフ側もV1互換のRNGに統一
                int64_t rand_valid_idx = rnd_->RandIndex(valid_count);
                int64_t idx = valid_ptr[rand_valid_idx];

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
#endif

    void UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities) override
    {
        ANET_PROFILE_FUNC();

        //ANET_LOG_DEBUG("UpdatePriorities() indices=" << indices << " priorities=" << priorities);
        //LOG::info() << "UpdatePriorities() indices=" << indices << " priorities=" << priorities;

        for (size_t i = 0; i < indices.size(); ++i) {
            float p = priorities[i];

            if (p < 0.0f) {
                // 特殊フラグ (-1.0f): 初期優先度を設定
                if (initial_priority_ < 0.0f) {
					tree_.Update(indices[i], max_prio_);    // 最大優先度で初期化
                } else {
                    float adjusted_p = (initial_priority_ == 1.0) ? 1.0 : std::pow(initial_priority_, alpha_);
                    tree_.Update(indices[i], adjusted_p);	// 固定値で初期化
                }
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

    float GetTotalPriority() const
    {
        return tree_.GetTotalPriority();
    }

    std::optional<torch::Tensor> GetPriorityTensor(int64_t index) const
    {
        if (index < 0 || index >= tree_.Capacity()) return std::nullopt;
        return torch::tensor(tree_.GetPriority(index), opt_float_);
    }

    torch::Tensor GatherPriorityRows(const torch::Tensor& indices) const
    {
        ANET_PROFILE_FUNC();

        auto indices_cpu = indices.to(torch::kCPU).contiguous();
        auto acc = indices_cpu.accessor<int64_t, 1>();
        std::vector<float> priorities(static_cast<size_t>(indices_cpu.size(0)));
        for (int64_t i = 0; i < indices_cpu.size(0); ++i) {
            priorities[static_cast<size_t>(i)] = tree_.GetPriority(acc[i]);
        }
        return torch::tensor(priorities, opt_float_).reshape({ indices_cpu.size(0), 1 });
    }
private:
    SumTree tree_;
    const float alpha_;
    const torch::Generator gen_;
    const torch::TensorOptions opt_long_;
    const torch::TensorOptions opt_float_;
    float max_prio_;
    float initial_priority_;
};


// ===========================================================================
// Extractor (具象クラス)
// ===========================================================================

class DefaultSampleExtractor : public ExperienceSampleExtractor {
public:
    explicit DefaultSampleExtractor(std::vector<std::string> stack_keys)
        : stack_keys_(std::move(stack_keys))
    {
    }

    void ExtractSamples(ExperienceSamples& out, const ReplayExperienceStorage& storage, const IndexSampleResult& idx_result, int stack_count, int unroll_steps) const override
    {
        ANET_PROFILE_FUNC();

        ANET_PROFILE_SCOPE(prepare);

        int64_t B = idx_result.indices.size(0);
        auto indices_acc = idx_result.indices.accessor<int64_t, 1>();

        int64_t cap = storage.GetTargetReturns().size(1); // capacity_per_env

        // 次元スカラー化のフラグ (DQNのような1ステップ構成の場合は時間次元をSqueezeする)
        int unroll_len = (unroll_steps > 0) ? unroll_steps : 1;
        bool squeeze_unroll = (unroll_steps == 0);
        bool squeeze_stack = (stack_count == 1);

        // テンソル構築用の配列
        std::vector<torch::Tensor> batch_actions, batch_returns, batch_terminals, batch_actual_n;
        std::vector<anet::TensorDict> batch_obs, batch_next_obs, batch_info;

        // 境界チェック用に terminals (done) フラグを一括取得
        auto terminals_tensor = storage.GetTerminals();

        /// @todo [Performance] 現在はバッチサイズ(B)回数分のループで C++ 側からスライスと torch::stack を行っている。
        /// GPU上でストレージを持つ場合、Pythonの `tensor[batch_indices, time_indices]` のように
        /// 高度なインデックス演算 (Advanced Indexing) を用いてベクトル化された一括抽出にリファクタリングすることで更なる学習FPSの向上が見込める

        ANET_PROFILE_SCOPE_NEXT(extract);
        for (int64_t b = 0; b < B; ++b) {
            int64_t idx1d = indices_acc[b];
            int64_t env_idx = idx1d / cap;
            int64_t time_idx = idx1d % cap;
            int64_t actual_n = storage.GetActualNSteps()[env_idx][time_idx].item<int64_t>();

            // 未来方向 (Unroll) のスライス抽出 ---
            batch_actions.push_back(RingSlice(storage.GetActions(), env_idx, time_idx, unroll_len, cap, squeeze_unroll));
            batch_returns.push_back(RingSlice(storage.GetTargetReturns(), env_idx, time_idx, unroll_len, cap, squeeze_unroll));
            batch_terminals.push_back(RingSlice(storage.GetTerminals(), env_idx, time_idx, unroll_len, cap, squeeze_unroll));
            batch_actual_n.push_back(storage.GetActualNSteps()[env_idx][time_idx]);

            // MuZero用などの固有情報 (存在する場合のみ)
            if (!storage.GetInfo().empty()) {
                batch_info.push_back(RingSliceDict(storage.GetInfo(), env_idx, time_idx, unroll_len, 0, cap, squeeze_unroll, {}));
            }

            // 過去方向 (Frame Stacking) のスライス抽出と境界パディング
            int64_t obs_start = time_idx - stack_count + 1;
            int64_t obs_valid_start = obs_start;
            for (int64_t k = time_idx - 1; k >= obs_start; --k) {
                int64_t phys_k = (k % cap + cap) % cap; // 負数を安全にリングバッファの末尾に折り返す
                if (terminals_tensor[env_idx][phys_k].item<bool>()) {
                    obs_valid_start = k + 1;
                    break;
                }
            }
            int64_t obs_valid_len = time_idx - obs_valid_start + 1;
            int64_t obs_pad_len = stack_count - obs_valid_len;
            batch_obs.push_back(RingSliceDict(storage.GetObs(), env_idx, obs_valid_start, obs_valid_len, obs_pad_len, cap, squeeze_stack, stack_keys_));

#if 1
            // N-Step先 (NextState) のスライス抽出と境界パディング
            int64_t next_obs_end = time_idx + actual_n + 1;
            int64_t next_obs_start = next_obs_end - stack_count;
            int64_t next_obs_valid_start = next_obs_start;
            for (int64_t k = next_obs_end - 2; k >= next_obs_start; --k) {
                int64_t phys_k = (k % cap + cap) % cap; // 負数を安全にリングバッファの末尾に折り返す
                if (terminals_tensor[env_idx][phys_k].item<bool>()) {
                    next_obs_valid_start = k + 1;
                    break;
                }
            }
            int64_t next_obs_valid_len = next_obs_end - next_obs_valid_start;
            int64_t next_obs_pad_len = stack_count - next_obs_valid_len;
            batch_next_obs.push_back(RingSliceDict(storage.GetObs(), env_idx, next_obs_valid_start, next_obs_valid_len, next_obs_pad_len, cap, squeeze_stack, stack_keys_));
#else
            // -------------------------------------------------------------------------
            // N-Step先 (NextState) のスライス抽出と境界パディング (★V1バグ再現パッチ 修正版)
            // -------------------------------------------------------------------------

            // 本来の next_obs の最新フレーム
            int64_t next_obs_end = time_idx + actual_n + 1;

            // ★ V1バグの再現: 
            // V1は obs の抽出インデックスをそのまま next_obs のバッファ(next_states_)適用していた。
            // つまり、抽出の開始位置(過去)は obs と全く同じになり、
            // 抽出の終了位置(最新)も「1つ未来」ではなく「現在(obsと同じ)」になっていた。
            // それを再現するため、開始位置を obs と同じにする。
            int64_t bugged_start = time_idx - stack_count + 1;
            int64_t bugged_valid_start = bugged_start;

            for (int64_t k = time_idx - 1; k >= bugged_start; --k) {
                if (k < 0) {
                    bugged_valid_start = k + 1;
                    break;
                }
                if (terminals_tensor[env_idx][k % cap].item<bool>()) {
                    bugged_valid_start = k + 1;
                    break;
                }
            }

            // V1の抽出長は obs と同じ (= obs_valid_len)
            int64_t bugged_valid_len = time_idx - bugged_valid_start + 1;

            // ただし、一番最後（最新フレーム）だけは「Nextの画像」に差し替えるのが
            // 強化学習の基本であり、V1でも末尾の結合等で実現されていた挙動。
            // これをV2のDict構造で安全に再現するため、
            // 1. まず [t-3, t-2, t-1] (過去3フレーム) を RingSliceDict で取る
            // 2. 次に [t_next] (最新1フレーム) を RingSliceDict で取る
            // 3. それらを concat する
            // という手順を踏む。

            // 1. 過去部分の抽出 (長さは stack_count - 1 になるように調整)
            int64_t past_pad_len = stack_count - 1 - bugged_valid_len;
            if (past_pad_len < 0) {
                // パディング不要な場合（通常時）は、先頭を1つ削る
                bugged_valid_start += 1;
                bugged_valid_len -= 1;
                past_pad_len = 0;
            }

            auto past_dict = RingSliceDict(storage.GetObs(), env_idx, bugged_valid_start, bugged_valid_len, past_pad_len, cap, false, stack_keys_);

            // 2. 最新部分の抽出 (NextObs)
            auto latest_dict = RingSliceDict(storage.GetObs(), env_idx, next_obs_end - 1, 1, 0, cap, false, stack_keys_);

            // 3. 結合してバッチに追加
            anet::TensorDict bugged_next_obs_dict;
            for (const auto& kv : past_dict) {
                auto past_tensor = kv.second;
                auto latest_tensor = latest_dict.At(kv.first);

                // Stack対象のキー（またはGridMazeのような全てStack）の場合のみ結合
                if (past_tensor.dim() >= 2 && latest_tensor.dim() >= 2) {
                    auto combined = torch::cat({ past_tensor, latest_tensor }, 0);
                    if (squeeze_stack && combined.size(0) == 1) {
                        combined = combined.squeeze(0);
                    }
                    bugged_next_obs_dict.Set(kv.first, combined);
                } else {
                    bugged_next_obs_dict.Set(kv.first, latest_tensor); // Stack非対象は最新のみ
                }
            }

            batch_next_obs.push_back(bugged_next_obs_dict);
            // -------------------------------------------------------------------------
#endif
        }

        ANET_PROFILE_SCOPE_NEXT(stack);
        // バッチスタック
        out.actions = torch::stack(batch_actions, 0);
        out.target_returns = torch::stack(batch_returns, 0);
        out.next_state.terminals = torch::stack(batch_terminals, 0);
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
private:
    std::vector<std::string> stack_keys_;
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
    ANET_PROFILE_FUNC();

    std::unique_lock<std::shared_mutex> storage_lock(storage_mutex_);
    PreparePendingPriorityUpdates();

    // 事前に action の info を取得しておく
    const anet::TensorDict& action_info = batch.action->GetInfo();
    const torch::Tensor actions = batch.action->GetAction();
    const anet::TensorDict empty_info;
    anet::TensorDict single_obs;
    anet::TensorDict single_info;
    anet::TensorDict terminal_obs;

    for (int64_t b = 0; b < num_envs_; ++b) {
        // Step 1: Storage に重いデータを即時 Push (重複排除)

        // 単一バッチ要素のDictは scratch に view を上書きして再利用する
        SetTensorDictBatchItem(single_obs, batch.state.obs, b);
        const anet::TensorDict* single_info_ptr = &empty_info;
        if (!action_info.empty()) {
            SetTensorDictBatchItem(single_info, action_info, b);
            single_info_ptr = &single_info;
        }

        int64_t time_idx = storage_->Push(b, single_obs, actions[b], *single_info_ptr);
        {
            std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
            index_manager_->MarkWritten(b, time_idx);
            index_manager_->AdvanceWriteCursor(b); // カーソルを進める

            AddPendingPriorityUpdate(b * capacity_per_env_ + time_idx, 0.0f);
        }

        // 正常なステップを先に Queue に入れる (タイムトラベル防止)
        QueueRecord rec;
        rec.time_idx = time_idx;
        rec.reward = batch.reward[b].item<float>();
        const bool next_done = batch.next_state.done[b].item<bool>();
        const bool next_truncated = batch.next_state.truncated[b].item<bool>();
        rec.done = next_done;
        rec.truncated = next_truncated;
        queues_[b].Push(rec);

        // Truncatedのパラドックス対策 (ダミーステップの挿入)
        if (next_truncated) {
            SetTensorDictBatchItem(terminal_obs, batch.next_state.obs, b);
            storage_->PushTerminalDummy(b, terminal_obs);

            int64_t dummy_idx = (time_idx + 1) % capacity_per_env_;
            {
                std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
                index_manager_->MarkDummy(b, dummy_idx);
                index_manager_->AdvanceWriteCursor(b); // ダミー書き込み分もカーソルを進める

                AddPendingPriorityUpdate(b * capacity_per_env_ + dummy_idx, 0.0f);
            }

            QueueRecord dummy_rec;
            dummy_rec.time_idx = dummy_idx;
            dummy_rec.reward = 0.0f;
            dummy_rec.done = true;
            dummy_rec.truncated = false;
            dummy_rec.is_dummy = true;
            queues_[b].Push(dummy_rec);
        }

        ProcessQueue(b);
    }

    FlushPendingPriorityUpdates();
    InvalidateAccessorCacheForStorage();
}

void DefaultReplayBuffer::PreparePendingPriorityUpdates()
{
    pending_prio_indices_.clear();
    pending_prio_values_.clear();

    if (!prio_controller_) return;

    const auto expected_updates = static_cast<size_t>(
        num_envs_ * (std::max<int>(1, config_.n_step) + 2));
    if (pending_prio_indices_.capacity() < expected_updates) {
        pending_prio_indices_.reserve(expected_updates);
        pending_prio_values_.reserve(expected_updates);
    }
}

void DefaultReplayBuffer::AddPendingPriorityUpdate(int64_t index, float priority)
{
    if (!prio_controller_) return;

    pending_prio_indices_.push_back(index);
    pending_prio_values_.push_back(priority);
}

void DefaultReplayBuffer::FlushPendingPriorityUpdates()
{
    if (!prio_controller_ || pending_prio_indices_.empty()) return;

    std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
    prio_controller_->UpdatePriorities(pending_prio_indices_, pending_prio_values_);
}

void DefaultReplayBuffer::ProcessQueue(int64_t env_idx)
{
    ANET_PROFILE_FUNC();

    auto sequences = queue_controller_->ExtractSequences(queues_[env_idx]);

    std::vector<int64_t> valid_envs;
    valid_envs.reserve(sequences.size());

    for (const auto& seq : sequences) {
        // 念のため空チェック
        if (seq.empty()) continue;

        //ダミーステップ自身が起点(state)となっているシーケンスはStorageに登録しない
        if (seq[0].is_dummy) {
            valid_envs.push_back(env_idx);
            continue;
        }

        // Builderで割引報酬和を計算
        ReplayExperience exp = builder_->Build(seq);

        // 先頭の time_idx を取り出し、Storageを上書き(Update)
        int64_t time_idx = seq.front().time_idx;
        storage_->Update(env_idx, time_idx, exp);

        // 完全にサンプリング可能になったので封印解除
        valid_envs.push_back(env_idx);

        AddPendingPriorityUpdate(env_idx * capacity_per_env_ + time_idx, -1.0f); // 初期優先度を割り当てるための特殊フラグ
    }

    // N-Step計算が完了し、安全になったデータをサンプリング対象へ移す
    if (!valid_envs.empty()) {
        std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
        for (int64_t valid_env : valid_envs) {
            index_manager_->MarkValid(valid_env);
        }
    }
}

void DefaultReplayBuffer::Sample(ExperienceSamples& out_samples, int64_t minibatch_size, float beta) const
{
	ANET_PROFILE_FUNC();

    std::shared_lock<std::shared_mutex> storage_lock(storage_mutex_);

    IndexSampleResult idx_result;
    {
        std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
        auto valid_1d = index_manager_->GetValidIndices1D(config_.stack_count, config_.muzero.unroll_steps, config_.n_step);
        ANET_ASSERT_MSG(valid_1d.size(0) >= minibatch_size, "Not enough valid samples in ReplayBuffer. size=" << valid_1d.size(0) << " minibatch_size=" << minibatch_size);

        idx_result = sampler_->SampleIndices(minibatch_size, valid_1d, beta);
    }
    extractor_->ExtractSamples(out_samples, *storage_, idx_result, config_.stack_count, config_.muzero.unroll_steps);
}

int64_t DefaultReplayBuffer::Size() const
{
    ANET_PROFILE_FUNC();

    // 生のカウントではなく、Stack/Unrollを考慮して安全なサンプリング可能な数を返す
    std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
    return index_manager_->GetSampleableCount(config_.stack_count, config_.muzero.unroll_steps, config_.n_step);
}

void DefaultReplayBuffer::UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities)
{
    if (prio_controller_) {
        {
            std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
            prio_controller_->UpdatePriorities(indices, priorities);
        }
        InvalidateAccessorCacheForPriority();
    }
}

void DefaultReplayBuffer::InvalidateAccessorCacheForStorage()
{
    std::lock_guard<std::mutex> lock(accessor_cache_mutex_);
    ++accessor_storage_version_;
    ++accessor_priority_version_;
    tensor_vector_cache_.clear();
}

void DefaultReplayBuffer::InvalidateAccessorCacheForPriority()
{
    std::lock_guard<std::mutex> lock(accessor_cache_mutex_);
    ++accessor_priority_version_;
    tensor_vector_cache_.clear();
}

std::optional<std::vector<torch::Tensor>> DefaultReplayBuffer::TryGetCachedTensorVector(const std::string& key, int64_t index) const
{
    std::lock_guard<std::mutex> lock(accessor_cache_mutex_);
    for (const auto& entry : tensor_vector_cache_) {
        if (entry.key == key &&
            entry.index == index &&
            entry.storage_version == accessor_storage_version_ &&
            entry.priority_version == accessor_priority_version_) {
            return entry.value;
        }
    }
    return std::nullopt;
}

void DefaultReplayBuffer::StoreTensorVectorCache(const std::string& key, int64_t index, std::vector<torch::Tensor> value) const
{
    constexpr size_t kMaxCacheEntries = 8;

    std::lock_guard<std::mutex> lock(accessor_cache_mutex_);
    for (auto& entry : tensor_vector_cache_) {
        if (entry.key == key && entry.index == index) {
            entry.storage_version = accessor_storage_version_;
            entry.priority_version = accessor_priority_version_;
            entry.value = std::move(value);
            return;
        }
    }

    if (tensor_vector_cache_.size() >= kMaxCacheEntries) {
        tensor_vector_cache_.erase(tensor_vector_cache_.begin());
    }

    tensor_vector_cache_.push_back(TensorVectorCacheEntry{
        key,
        index,
        accessor_storage_version_,
        accessor_priority_version_,
        std::move(value)
    });
}

std::optional<float> DefaultReplayBuffer::GetScalar(const std::string& key, int64_t index) const
{
    if (key == PER_TOTAL) {
        std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
        auto prioritized = std::dynamic_pointer_cast<PrioritizedSampler>(sampler_);
        if (!prioritized) return std::nullopt;
        return prioritized->GetTotalPriority();
    }
    return std::nullopt;
}

std::optional<torch::Tensor> DefaultReplayBuffer::GetTensor(const std::string& key, int64_t index) const
{
    if (key == PER_VALUES) {
        std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
        auto prioritized = std::dynamic_pointer_cast<PrioritizedSampler>(sampler_);
        if (!prioritized) return std::nullopt;
        return prioritized->GetPriorityTensor(index);
    }

    auto opt_vec = GetTensorVector(key, -1);
    if (!opt_vec.has_value() || opt_vec->empty()) return std::nullopt;

    auto tensor = (*opt_vec)[0];
    if (!tensor.defined()) return std::nullopt;
    if (index >= 0) {
        tensor = SelectRowIfRequested(tensor, index);
        if (!tensor.defined()) return std::nullopt;
    }
    return tensor;
}

std::optional<std::vector<torch::Tensor>> DefaultReplayBuffer::GetTensorVector(const std::string& key, int64_t index) const
{
    ANET_PROFILE_FUNC();

    auto storage_key = ParseStorageTensorVectorKey(key);
    const bool is_per_vector_key = key == PER_VALUES || key == PER_DIST;
    if (!storage_key.has_value() && !is_per_vector_key) return std::nullopt;

    ANET_PROFILE_SCOPE(cache_lookup);
    if (auto cached = TryGetCachedTensorVector(key, index)) {
        ANET_PROFILE_SCOPE_NEXT(cache_hit);
        return cached;
    }

    ANET_PROFILE_SCOPE_NEXT(cache_miss);
    std::shared_lock<std::shared_mutex> storage_lock(storage_mutex_);

    ANET_PROFILE_SCOPE_NEXT(valid_indices);
    torch::Tensor valid_1d;
    {
        std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
        valid_1d = index_manager_->GetValidIndices1D(config_.stack_count, config_.muzero.unroll_steps, config_.n_step);
    }
    if (valid_1d.numel() == 0) {
        std::vector<torch::Tensor> result;
        StoreTensorVectorCache(key, index, result);
        return result;
    }

    torch::Tensor tensor;

    if (is_per_vector_key) {
        ANET_PROFILE_SCOPE_NEXT(gather_per);
        {
            std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
            auto prioritized = std::dynamic_pointer_cast<PrioritizedSampler>(sampler_);
            if (!prioritized) return std::nullopt;
            tensor = prioritized->GatherPriorityRows(valid_1d);
            if (key == PER_DIST) {
                // PER_DIST は正規化サンプリング確率 p/total を返す（SampleIndices の prob=p/total と同義）。
                const float total = prioritized->GetTotalPriority();
                if (total > 0.0f) tensor = tensor / total;  // total<=0（極初期）はゼロのまま
            }
        }
    } else {
        ANET_PROFILE_SCOPE_NEXT(direct_gather);

        switch (storage_key->kind) {
        case StorageTensorVectorKey::Kind::kStateObs:
            tensor = GatherObservationRows(storage_->GetObs(), valid_1d, storage_key->obs_subkey);
            break;
        case StorageTensorVectorKey::Kind::kAction:
            tensor = GatherFlatRows(storage_->GetActions(), valid_1d);
            break;
        case StorageTensorVectorKey::Kind::kTargetReturn:
            // target return（N-step 割引報酬和、bootstrap 前）を返す。
            tensor = GatherFlatRows(storage_->GetTargetReturns(), valid_1d);
            break;
        case StorageTensorVectorKey::Kind::kNextStateObs:
            tensor = GatherObservationRows(
                storage_->GetObs(),
                MakeNextFlatIndices(valid_1d, storage_->GetActualNSteps(), capacity_per_env_),
                storage_key->obs_subkey);
            break;
        case StorageTensorVectorKey::Kind::kNextStateTerminal:
            tensor = GatherFlatRows(storage_->GetTerminals(), valid_1d);
            break;
        case StorageTensorVectorKey::Kind::kNStep:
            tensor = GatherFlatRows(storage_->GetActualNSteps(), valid_1d);
            break;
        }
    }

    tensor = SelectRowIfRequested(tensor, index);
    if (!tensor.defined()) return std::nullopt;

    std::vector<torch::Tensor> result{ tensor };
    StoreTensorVectorCache(key, index, result);
    return result;
}

void DefaultReplayBuffer::DumpToLog() const
{
    std::shared_lock<std::shared_mutex> storage_lock(storage_mutex_);

    if (storage_) {
        storage_->DumpToLog();
    }
    torch::Tensor valid_1d;
    {
        std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
        valid_1d = index_manager_->GetValidIndices1D(config_.stack_count, config_.muzero.unroll_steps, config_.n_step);
    }

    // Valid Index を見やすく出力
    std::string valid_str = "[ ";
    if (valid_1d.numel() > 0) {
        auto acc = valid_1d.accessor<int64_t, 1>();
        for (int64_t i = 0; i < valid_1d.size(0); ++i) {
            valid_str += std::to_string(acc[i]) + " ";
        }
    }
    valid_str += "]";

    LOG::info() << "=== Valid Indices (1D) ===";
    LOG::info() << valid_str;
    LOG::info() << "Total Valid Count: " << valid_1d.size(0);
}


// ===========================================================================
// Prefetching ReplayBuffer
// ===========================================================================

static torch::Tensor PrefetchPinCpuTensor(const torch::Tensor& tensor)
{
    if (!tensor.defined()) return tensor;

    torch::Tensor cpu_tensor = tensor.device().is_cpu() ? tensor : tensor.to(torch::kCPU);
    if (cpu_tensor.is_pinned()) return cpu_tensor;

    auto options = cpu_tensor.options().device(torch::kCPU).pinned_memory(true);
    auto pinned = torch::empty_strided(cpu_tensor.sizes(), cpu_tensor.strides(), options);
    pinned.copy_(cpu_tensor);
    return pinned;
}

static anet::TensorDict PrefetchPinCpuTensorDict(const anet::TensorDict& dict)
{
    anet::TensorDict pinned;
    for (const auto& kv : dict) {
        pinned.Set(kv.first, PrefetchPinCpuTensor(kv.second));
    }
    return pinned;
}

static ExperienceSamples PrefetchPinCpuSamples(const ExperienceSamples& samples)
{
    return ExperienceSamples{
        .obs = PrefetchPinCpuTensorDict(samples.obs),
        .actions = PrefetchPinCpuTensor(samples.actions),
        .target_returns = PrefetchPinCpuTensor(samples.target_returns),
        .next_state = {
            PrefetchPinCpuTensorDict(samples.next_state.next_obs),
            PrefetchPinCpuTensor(samples.next_state.terminals)
        },
        .n_steps = PrefetchPinCpuTensor(samples.n_steps),
        .indices = PrefetchPinCpuTensor(samples.indices),
        .is_weights = PrefetchPinCpuTensor(samples.is_weights),
        .info = PrefetchPinCpuTensorDict(samples.info)
    };
}

struct PrefetchingReplayBuffer::PrefetchedBatch {
    ExperienceSamples pinned_cpu_samples;
    ExperienceSamples samples;
    std::unique_ptr<at::cuda::CUDAEvent> ready_event;
};

struct PrefetchingReplayBuffer::State {
    explicit State(torch::Device target_device)
        : pool(std::make_unique<anet::PinnedThreadPool>(1, "ReplayBufferPrefetch"))
    {
        if (target_device.is_cuda()) {
            copy_stream = at::cuda::getStreamFromPool(false);
        }
    }

    mutable std::mutex mutex;
    std::unique_ptr<anet::PinnedThreadPool> pool;
    std::optional<at::cuda::CUDAStream> copy_stream;
    std::future<PrefetchedBatch> future;
};

PrefetchingReplayBuffer::PrefetchingReplayBuffer(std::shared_ptr<ReplayBuffer> inner, torch::Device target_device)
    : inner_(std::move(inner))
    , target_device_(std::move(target_device))
    , state_(std::make_unique<State>(target_device_))
{
    ANET_ASSERT_MSG(inner_ != nullptr, "PrefetchingReplayBuffer requires a non-null inner ReplayBuffer.");
}

PrefetchingReplayBuffer::~PrefetchingReplayBuffer()
{
    StopPrefetch();
}

void PrefetchingReplayBuffer::Push(const BatchExperience& batch_exp)
{
    ANET_PROFILE_FUNC();

    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->future.valid()) {
        ANET_PROFILE_SCOPE(wait_prefetch);

        // Pushはstorageを書き換えるため、background Sample が終わってからinnerへ進める。
        WaitForPrefetchLocked();
    }
    inner_->Push(batch_exp);
}

void PrefetchingReplayBuffer::Sample(ExperienceSamples& out_samples, int64_t minibatch_size, float beta) const
{
    ANET_PROFILE_FUNC();

    PrefetchedBatch batch;
    {
        std::unique_lock<std::mutex> lock(state_->mutex);
        if (state_->future.valid()) {
            ANET_PROFILE_SCOPE(consume_wait);

            // 前回起動した1-deep prefetchをここで消費し、同じ排他区間で次のprefetchを起動する。
            // futureの有無をこの順序で更新することで、二重起動せず常に最大1件だけ先読みする。
            batch = state_->future.get();
            LaunchPrefetchLocked(minibatch_size, beta);
        } else {
            ANET_PROFILE_SCOPE(cold_fetch);

            // 初回はfutureが無いため、state mutex内で同期Sampleし、Push/UpdatePrioritiesとの順序を固定する。
            batch = Fetch(minibatch_size, beta);
            LaunchPrefetchLocked(minibatch_size, beta);
        }
    }

    if (batch.ready_event) {
        ANET_PROFILE_SCOPE(event_wait);

        // CUDA転送はcopy streamで済ませるため、利用側のcurrent streamからevent待ちしてから返す。
        batch.ready_event->block(at::cuda::getCurrentCUDAStream());
    }
    out_samples = std::move(batch.samples);
}

int64_t PrefetchingReplayBuffer::Size() const
{
    return inner_->Size();
}

void PrefetchingReplayBuffer::UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities)
{
    ANET_PROFILE_FUNC();

    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->future.valid()) {
        ANET_PROFILE_SCOPE(wait_prefetch);

        // background Sample と priority 更新の順序を固定する。
        WaitForPrefetchLocked();
    }
    inner_->UpdatePriorities(indices, priorities);
}

std::optional<float> PrefetchingReplayBuffer::GetScalar(const std::string& key, int64_t index) const
{
    return inner_->GetScalar(key, index);
}

std::optional<torch::Tensor> PrefetchingReplayBuffer::GetTensor(const std::string& key, int64_t index) const
{
    return inner_->GetTensor(key, index);
}

std::optional<std::vector<torch::Tensor>> PrefetchingReplayBuffer::GetTensorVector(const std::string& key, int64_t index) const
{
    return inner_->GetTensorVector(key, index);
}

PrefetchingReplayBuffer::PrefetchedBatch PrefetchingReplayBuffer::Fetch(int64_t minibatch_size, float beta) const
{
    ANET_PROFILE_SCOPE(sample);
    ExperienceSamples cpu_samples;
    inner_->Sample(cpu_samples, minibatch_size, beta);

    return TransferSamples(std::move(cpu_samples));
}

PrefetchingReplayBuffer::PrefetchedBatch PrefetchingReplayBuffer::TransferSamples(ExperienceSamples cpu_samples) const
{
    if (!target_device_.is_cuda()) {
        ANET_PROFILE_SCOPE_FULL(to, "PrefetchingReplayBuffer::Fetch.to");

        // CPU learnerではCUDA用のpin/stream/eventを使わず、抽出済みsampleだけをCPUへそろえる。
        return PrefetchedBatch{
            .pinned_cpu_samples = ExperienceSamples{},
            .samples = cpu_samples.To(target_device_),
            .ready_event = nullptr
        };
    }

    ANET_PROFILE_SCOPE_FULL(to, "PrefetchingReplayBuffer::Fetch.to");

    // CUDA learnerではworker側でCPU tensorをpinned化し、copy streamへH2D転送を積む。
    // pinned_cpu_samplesはevent待ちが終わるまで転送元の寿命を保つためbatchに保持する。
    auto pinned_samples = PrefetchPinCpuSamples(cpu_samples);

    const auto& copy_stream = state_->copy_stream.value();
    at::cuda::CUDAStreamGuard guard(copy_stream);
    auto dev_samples = pinned_samples.To(target_device_, /*non_blocking=*/true);

    auto ready_event = std::make_unique<at::cuda::CUDAEvent>();
    ready_event->record(copy_stream);

    return PrefetchedBatch{
        .pinned_cpu_samples = std::move(pinned_samples),
        .samples = std::move(dev_samples),
        .ready_event = std::move(ready_event)
    };
}

void PrefetchingReplayBuffer::WaitForPrefetchLocked() const
{
    if (state_->future.valid()) {
        state_->future.wait();
    }
}

void PrefetchingReplayBuffer::LaunchPrefetchLocked(int64_t minibatch_size, float beta) const
{
    if (state_->future.valid()) return;

    state_->future = state_->pool->EnqueueFuture(0, [this, minibatch_size, beta]() {
        return Fetch(minibatch_size, beta);
    });
}

void PrefetchingReplayBuffer::StopPrefetch() const
{
    if (!state_) return;

    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        WaitForPrefetchLocked();
    }

    if (state_->pool) {
        state_->pool->WaitAll();
        state_->pool->Stop();
    }
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

    auto extractor = std::make_shared<DefaultSampleExtractor>(config.stack_keys);

    return std::make_shared<DefaultReplayBuffer>(
        config, env_spec, num_envs, std::move(queue_controller), std::move(builder), sampler, prio, extractor, storage_device, pin_memory);
}
