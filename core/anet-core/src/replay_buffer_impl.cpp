// replay_buffer_impl.cpp

#include "replay_buffer_impl.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <future>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>
#include "anet/metrics_logger.hpp"
#include "anet/tensor_check.hpp"
#include "anet/profile.hpp"
#include "anet/log.hpp"
#include "anet/random.hpp"
#include "anet/thread.hpp"
#include "anet/transfer.hpp"

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
            exp.truncated = sequence.back().truncated && !sequence.back().done;
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

std::optional<ValidIndexManager::LogicalSampleableRange>
ValidIndexManager::GetLogicalSampleableRange(int64_t env_idx, int unroll_steps, int n_step) const
{
    const int64_t write_cursor = write_cursors_[env_idx];
    const int64_t valid_cursor = valid_cursors_[env_idx];
    const int64_t future_obs_lag = std::max<int64_t>(1, n_step);

    // リング上で未上書き、かつ必要な未来観測とunroll範囲が確定済みの論理区間を求める。
    const int64_t logical_start = std::max<int64_t>(0, write_cursor - capacity_per_env_);
    const int64_t max_safe_by_write = std::max<int64_t>(-1, write_cursor - future_obs_lag - 1);
    const int64_t max_safe_by_valid = valid_cursor - 1 - unroll_steps;
    const int64_t logical_end = std::min(max_safe_by_write, max_safe_by_valid);
    if (logical_end < logical_start) return std::nullopt;
    return LogicalSampleableRange{ .start = logical_start, .end = logical_end };
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

bool ValidIndexManager::IsOverwritingSampleable(int64_t env_idx, int64_t time_idx, int stack_count, int unroll_steps, int n_step) const
{
    (void)stack_count;

    const int64_t w_cursor = write_cursors_[env_idx];
    if (w_cursor < capacity_per_env_) return false;
    if (time_idx != w_cursor % capacity_per_env_) return false;

    const int64_t flat_idx = env_idx * capacity_per_env_ + time_idx;
    if (is_dummy_[flat_idx]) return false;

    const int64_t evicted_logical = w_cursor - capacity_per_env_;
    const auto range = GetLogicalSampleableRange(env_idx, unroll_steps, n_step);
    return range.has_value() && range->Contains(evicted_logical);
}

bool ValidIndexManager::IsLogicalSampleable(
    int64_t env_idx, int64_t logical_idx, int unroll_steps, int n_step) const
{
    // 単点判定でも列挙処理と同じ上書き境界、未来観測、unroll終端を適用する。
    const auto range = GetLogicalSampleableRange(env_idx, unroll_steps, n_step);
    return range.has_value() && range->Contains(logical_idx);
}


// ===========================================================================
// Initial Priority Completer
// ===========================================================================

InitialPriorityCompleter::InitialPriorityCompleter(
    Config config, int64_t num_envs,
    std::unique_ptr<InitialPriorityEstimator> estimator)
    : config_(config)
    , estimator_(std::move(estimator))
    , pending_(static_cast<size_t>(num_envs))
{
}

void InitialPriorityCompleter::Enqueue(int64_t env_idx, PendingEntry entry)
{
    pending_[static_cast<size_t>(env_idx)].push_back(std::move(entry));
}

void InitialPriorityCompleter::CompleteReady(
    int64_t env_idx,
    const std::optional<ReplayInitialPriorityHintRow>& current_hint,
    int64_t current_logical_idx,
    const ValidIndexManager& index_manager,
    ReplayPriorityStore& priority_store)
{
    ANET_PROFILE_FUNC();

    auto& pending = pending_[static_cast<size_t>(env_idx)];
    // FIFO先頭から、必要な未来観測とunroll範囲が確定してsampleableになった遷移だけを処理する。
    while (!pending.empty()
        && index_manager.IsLogicalSampleable(
            env_idx, pending.front().logical_time_idx, config_.unroll_steps, config_.n_step)) {
        const auto entry = pending.front();
        pending.pop_front();

        // fixed/maxはActorヒントを参照せず、同じsampleable化境界で初期sourceを確定する。
        if (config_.mode == ReplayInitialPriorityMode::FIXED) {
            priority_store.SetRawInitialPriority(
                entry.flat_slot_index, config_.fixed_raw_priority, ReplayPrioritySource::FIXED_INITIAL);
            continue;
        }
        if (config_.mode == ReplayInitialPriorityMode::MAX) {
            priority_store.SetMaxInitialPriority(entry.flat_slot_index);
            continue;
        }

        ++stats_.attempt_count;

        // 実遷移の開始stepには必ずschemaを満たすhint行が付随する契約を検証する。
        if (!entry.start_hint.has_value()) {
            ANET_SYSTEM_ERROR("actor_approx requires a Replay initial priority hint for every real start step."
                << " env=" << env_idx << " logical_time_idx=" << entry.logical_time_idx);
        }
        if (!estimator_) {
            ANET_SYSTEM_ERROR("actor_approx requires InitialPriorityEstimator.");
        }
        const auto start_hint = std::span<const float>(entry.start_hint->data(), entry.start_hint->size());
        // Actor由来の非finite値はDebugでは原因を表面化し、Release系ではmaxへ退避して学習を継続する。
        if (!estimator_->ValidateHint(start_hint)) {
#ifdef NDEBUG
            ++stats_.nonfinite_fallback_count;
            priority_store.SetMaxInitialPriority(entry.flat_slot_index);
            continue;
#else
            ANET_SYSTEM_ERROR("actor_approx received a non-finite start Replay initial priority hint.");
#endif
        }
        // truncatedは終端観測を推論しないため、bootstrapせずmax初期化へフォールバックする。
        if (entry.truncated) {
            ++stats_.truncation_fallback_count;
            priority_store.SetMaxInitialPriority(entry.flat_slot_index);
            continue;
        }

        std::span<const float> bootstrap_hint;
        if (!entry.terminal) {
            // 非終端では実n-step先の現在hintだけをbootstrapに使い、時系列ずれを契約違反として止める。
            const int64_t expected_logical = entry.logical_time_idx + entry.actual_n_steps;
            if (current_logical_idx != expected_logical || !current_hint.has_value()) {
                ANET_SYSTEM_ERROR("actor_approx bootstrap Replay initial priority hint mismatch. expected_logical="
                    << expected_logical << " actual_logical=" << current_logical_idx);
            }
            bootstrap_hint = std::span<const float>(current_hint->data(), current_hint->size());
            if (!estimator_->ValidateHint(bootstrap_hint)) {
#ifdef NDEBUG
                ++stats_.nonfinite_fallback_count;
                priority_store.SetMaxInitialPriority(entry.flat_slot_index);
                continue;
#else
                ANET_SYSTEM_ERROR("actor_approx received a non-finite bootstrap Replay initial priority hint.");
#endif
            }
        }

        // 確定済み収益とActor値をアルゴリズム固有Estimatorへ渡し、raw優先度を完成させる。
        const auto raw_priority = estimator_->Estimate(InitialPriorityEstimateInput{
            .start_hint = start_hint,
            .bootstrap_hint = bootstrap_hint,
            .target_return = entry.target_return,
            .discount = static_cast<float>(std::pow(config_.gamma, entry.actual_n_steps)),
            .terminal = entry.terminal,
            .actual_n_steps = entry.actual_n_steps,
        });
        if (!raw_priority.has_value() || !std::isfinite(*raw_priority)) {
#ifdef NDEBUG
            ++stats_.nonfinite_fallback_count;
            priority_store.SetMaxInitialPriority(entry.flat_slot_index);
            continue;
#else
            ANET_SYSTEM_ERROR("actor_approx estimator returned a non-finite priority.");
#endif
        }
        if (*raw_priority < 0.0f) {
            ANET_SYSTEM_ERROR("actor_approx estimator returned a negative priority. value=" << *raw_priority);
        }

        // per_alpha適用前のraw値として保存し、sourceをactor_initialへ遷移させる。
        priority_store.SetRawInitialPriority(
            entry.flat_slot_index, *raw_priority, ReplayPrioritySource::ACTOR_INITIAL);
        ++stats_.success_count;
    }
}

std::unique_ptr<InitialPriorityCompleter> anet::rl::CreateInitialPriorityCompleter(
    const ReplayBufferConfig& config,
    int64_t num_envs,
    std::unique_ptr<InitialPriorityEstimator> estimator)
{
    if (config.sampler_type == ReplaySamplerType::UNIFORM) return nullptr;

    return std::make_unique<InitialPriorityCompleter>(
        InitialPriorityCompleter::Config{
            .mode = config.per_initial_priority_mode,
            .fixed_raw_priority = config.per_initial_priority,
            .gamma = config.gamma,
            .n_step = config.n_step,
            .unroll_steps = config.muzero.unroll_steps,
        },
        num_envs,
        std::move(estimator));
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
                act_str = anet::ToString(actions_[e][t]);

            LOG::info() << "  [idx=" << t << "] ret=" << ret
                << " term=" << term << " n_steps=" << n
                << " obs={" << obs_str << "}"
                << " act={" << act_str << "}";
        }
    }
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
        ANET_PROFILE_FUNC();

        int64_t valid_count = valid_indices_1d.size(0);
        auto rand_idx = torch::randint(0, valid_count, { batch_size }, gen_, opt_long_);
        auto indices = valid_indices_1d.index_select(0, rand_idx);
        auto ones = torch::ones({ batch_size }, opt_float_);



        return { indices, ones / valid_count, ones, torch::Tensor() };
    }
private:
    torch::Generator gen_;
    torch::TensorOptions opt_long_;
    torch::TensorOptions opt_float_;
};

class PrioritizedSampler : public ReplayExperienceSampler, public ReplayPriorityStore, public anet::RandomHolder {
public:
    PrioritizedSampler(int64_t capacity, float alpha, std::optional<anet::seed_t> seed)
        : anet::RandomHolder(seed)
        , tree_(capacity)
        , sources_(static_cast<size_t>(capacity), ReplayPrioritySource::NONE)
        , alpha_(alpha)
        , gen_(rnd_->GetTorchGenerator(torch::kCPU))
        , opt_long_(torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
        , opt_float_(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
        , max_prio_(1.0f)
    {
    }

    IndexSampleResult SampleIndices(int64_t batch_size, const torch::Tensor& valid_indices_1d, float beta) override
    {
        ANET_PROFILE_FUNC();

        int64_t valid_count = valid_indices_1d.size(0);
        const int64_t* valid_ptr = valid_indices_1d.data_ptr<int64_t>();

        std::vector<int64_t> sampled_indices(batch_size);
        std::vector<float> sampled_probs(batch_size);
        std::vector<float> sampled_weights(batch_size);
        std::vector<int8_t> sampled_sources(batch_size);

        const double total_prio = tree_.GetTotalPriority();

        int64_t b = 0;
        int max_attempts = batch_size * 10;
        int attempts = 0;

        // valid_indices_1d は昇順ソートが保証されているため、std::binary_search による O(1)~O(log V) 判定が可能
        while (b < batch_size && attempts < max_attempts) {
            attempts++;
            const double r = static_cast<double>(torch::rand({ 1 }, gen_, opt_float_).item<float>()) * total_prio;
            int64_t idx = tree_.Retrieve(r);

            bool is_valid = std::binary_search(valid_ptr, valid_ptr + valid_count, idx);
            if (is_valid) {
                const double p = tree_.GetPriority(idx);
                if (p <= 0.0) continue; // 安全装置

                sampled_indices[b] = idx;
                const double prob = p / total_prio;
                sampled_probs[b] = static_cast<float>(prob);

                const double weight = std::pow(static_cast<double>(valid_count) * prob, -static_cast<double>(beta));
                sampled_weights[b] = static_cast<float>(weight);
                sampled_sources[b] = static_cast<int8_t>(sources_[static_cast<size_t>(idx)]);
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
                sampled_sources[b + i] = static_cast<int8_t>(sources_[static_cast<size_t>(idx)]);
            }
        }

        auto idx_device = valid_indices_1d.device();
        auto indices_t = torch::tensor(sampled_indices, opt_long_).to(idx_device);
        auto probs_t = torch::tensor(sampled_probs, opt_float_).to(idx_device);
        auto weights_t = torch::tensor(sampled_weights, opt_float_).to(idx_device);
        auto source_t = torch::tensor(sampled_sources, torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU));

        weights_t /= weights_t.max(); // 正規化

        return { indices_t, probs_t, weights_t, source_t };
    }

    void InvalidateSlot(int64_t flat_slot_index) override
    {
        UpdateTreePriority(flat_slot_index, 0.0f, ReplayPrioritySource::NONE);
    }

    void SetRawInitialPriority(
        int64_t flat_slot_index, float raw_priority, ReplayPrioritySource source) override
    {
        UpdateTreePriority(flat_slot_index, std::pow(raw_priority, alpha_), source);
    }

    void SetMaxInitialPriority(int64_t flat_slot_index) override
    {
        UpdateTreePriority(flat_slot_index, max_prio_, ReplayPrioritySource::MAX_INITIAL);
    }

    ReplayPriorityUpdateResult ApplyLearnerPriorities(
        const std::vector<int64_t>& flat_slot_indices,
        const std::vector<float>& raw_priorities) override
    {
        ANET_PROFILE_FUNC();
        ReplayPriorityUpdateResult result;
        std::vector<std::pair<float, float>> actor_pairs;
        actor_pairs.reserve(flat_slot_indices.size());

        for (size_t i = 0; i < flat_slot_indices.size(); ++i) {
            const int64_t index = flat_slot_indices[i];
            const float adjusted = std::pow(raw_priorities[i], alpha_);
            if (sources_[static_cast<size_t>(index)] == ReplayPrioritySource::ACTOR_INITIAL) {
                actor_pairs.emplace_back(static_cast<float>(tree_.GetPriority(index)), adjusted);
            }
            UpdateTreePriority(index, adjusted, ReplayPrioritySource::LEARNER_UPDATED);
            max_prio_ = std::max(max_prio_, adjusted);
            ++result.applied_count;
        }

        FillActorComparison(actor_pairs, &result);
        return result;
    }

    std::optional<float> GetScalar(const std::string& key) const override
    {
        if (key == ReplayBuffer::PER_TOTAL) return static_cast<float>(tree_.GetTotalPriority());
        const double total = tree_.GetTotalPriority();
        if (total <= 0.0) return std::numeric_limits<float>::quiet_NaN();
        if (key == ReplayBuffer::PER_INITIAL_MASS_RATIO) {
            const double mass = source_mass_[1] + source_mass_[2] + source_mass_[3];
            return static_cast<float>(std::clamp(mass, 0.0, total) / total);
        }
        if (key == ReplayBuffer::PER_FIXED_INITIAL_MASS_RATIO) return SourceMassRatio(ReplayPrioritySource::FIXED_INITIAL, total);
        if (key == ReplayBuffer::PER_MAX_INITIAL_MASS_RATIO) return SourceMassRatio(ReplayPrioritySource::MAX_INITIAL, total);
        if (key == ReplayBuffer::PER_ACTOR_INITIAL_MASS_RATIO) return SourceMassRatio(ReplayPrioritySource::ACTOR_INITIAL, total);
        return std::nullopt;
    }

    std::optional<torch::Tensor> GetPriorityTensor(int64_t index) const override
    {
        if (index < 0 || index >= tree_.Capacity()) return std::nullopt;
        return torch::tensor(static_cast<float>(tree_.GetPriority(index)), opt_float_);
    }

    torch::Tensor GatherPriorityRows(const torch::Tensor& indices) const override
    {
        ANET_PROFILE_FUNC();

        auto indices_cpu = indices.to(torch::kCPU).contiguous();
        auto acc = indices_cpu.accessor<int64_t, 1>();
        std::vector<float> priorities(static_cast<size_t>(indices_cpu.size(0)));
        for (int64_t i = 0; i < indices_cpu.size(0); ++i) {
            priorities[static_cast<size_t>(i)] = static_cast<float>(tree_.GetPriority(acc[i]));
        }
        return torch::tensor(priorities, opt_float_).reshape({ indices_cpu.size(0), 1 });
    }
private:
    static void FillActorComparison(
        const std::vector<std::pair<float, float>>& pairs, ReplayPriorityUpdateResult* result)
    {
        result->actor_learner_pair_count = static_cast<int64_t>(pairs.size());
        if (pairs.empty()) return;

        // 倍率統計は除算と対数が定義できる正値ペアだけを対象にする。
        std::vector<float> ratios;
        double log_sum = 0.0;
        for (const auto& [actor, learner] : pairs) {
            if (actor > 0.0f && learner > 0.0f) {
                ratios.push_back(actor / learner);
                log_sum += std::log(actor / learner);
            }
        }
        result->actor_learner_positive_pair_ratio =
            static_cast<float>(ratios.size()) / static_cast<float>(pairs.size());
        if (!ratios.empty()) {
            // 偶数件では中央2値を平均し、倍率の外れ値に強い代表値を作る。
            std::sort(ratios.begin(), ratios.end());
            const size_t mid = ratios.size() / 2;
            result->actor_learner_ratio_median = ratios.size() % 2 == 0
                ? (ratios[mid - 1] + ratios[mid]) * 0.5f
                : ratios[mid];
            result->actor_learner_log_ratio_mean = static_cast<float>(log_sum / ratios.size());
        }
        result->actor_learner_spearman = Spearman(pairs);
    }

    static float Spearman(const std::vector<std::pair<float, float>>& pairs)
    {
        if (pairs.size() < 2) return std::numeric_limits<float>::quiet_NaN();
        // 同値には占有順位の平均を割り当て、tieを含む通常のSpearman順位相関にする。
        auto make_ranks = [](const std::vector<float>& values) {
            std::vector<size_t> order(values.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return values[a] < values[b]; });
            std::vector<double> ranks(values.size());
            for (size_t i = 0; i < order.size();) {
                size_t j = i + 1;
                while (j < order.size() && values[order[j]] == values[order[i]]) ++j;
                const double rank = (static_cast<double>(i) + static_cast<double>(j - 1)) * 0.5;
                for (size_t k = i; k < j; ++k) ranks[order[k]] = rank;
                i = j;
            }
            return ranks;
        };
        std::vector<float> actor, learner;
        for (const auto& p : pairs) { actor.push_back(p.first); learner.push_back(p.second); }
        const auto ar = make_ranks(actor);
        const auto lr = make_ranks(learner);
        const double am = std::accumulate(ar.begin(), ar.end(), 0.0) / ar.size();
        const double lm = std::accumulate(lr.begin(), lr.end(), 0.0) / lr.size();
        double cov = 0.0, av = 0.0, lv = 0.0;
        for (size_t i = 0; i < ar.size(); ++i) {
            cov += (ar[i] - am) * (lr[i] - lm);
            av += (ar[i] - am) * (ar[i] - am);
            lv += (lr[i] - lm) * (lr[i] - lm);
        }
        // 片側が全tieなら順位の分散がなく相関を定義できないためNaNを維持する。
        if (av == 0.0 || lv == 0.0) return std::numeric_limits<float>::quiet_NaN();
        return static_cast<float>(cov / std::sqrt(av * lv));
    }

    float SourceMassRatio(ReplayPrioritySource source, double total) const
    {
        return static_cast<float>(std::clamp(source_mass_[static_cast<size_t>(source)], 0.0, total) / total);
    }

    void UpdateTreePriority(int64_t index, float adjusted_priority, ReplayPrioritySource source)
    {
        const size_t slot = static_cast<size_t>(index);
        const double old_priority = tree_.GetPriority(index);
        source_mass_[static_cast<size_t>(sources_[slot])] -= old_priority;

        tree_.Update(index, static_cast<double>(adjusted_priority));
        sources_[slot] = source;
        source_mass_[static_cast<size_t>(source)] += static_cast<double>(adjusted_priority);
    }

    SumTree<double> tree_;
    std::vector<ReplayPrioritySource> sources_;
    std::array<double, 5> source_mass_{};
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
    explicit DefaultSampleExtractor(std::vector<std::string> stack_keys)
        : stack_keys_(std::move(stack_keys))
    {
    }

    void ExtractSamples(ExperienceSamples& out, const ReplayExperienceStorage& storage, const IndexSampleResult& idx_result, int stack_count, int unroll_steps) const override
    {
        ANET_PROFILE_FUNC();

        ANET_PROFILE_SCOPE(prepare);

        int64_t B = idx_result.flat_slot_indices.size(0);
        auto indices_acc = idx_result.flat_slot_indices.accessor<int64_t, 1>();

        int64_t cap = storage.GetTargetReturns().size(1); // capacity_per_env

        // 次元スカラー化のフラグ (DQNのような1ステップ構成の場合は時間次元をSqueezeする)
        int unroll_len = (unroll_steps > 0) ? unroll_steps : 1;
        bool squeeze_unroll = (unroll_steps == 0);
        bool squeeze_stack = (stack_count == 1);

        // テンソル構築用の配列
        std::vector<torch::Tensor> batch_actions, batch_returns, batch_terminals, batch_actual_n;
        std::vector<anet::TensorDict> batch_obs, batch_next_obs, batch_info;

        // terminals_ は n-step return の終端到達も表すため、frame stack の境界には
        // 実エピソード終端、dummy、未書き込み slot だけを使う。
        auto terminals_tensor = storage.GetTerminals();
        auto actual_n_steps_tensor = storage.GetActualNSteps();
        auto terminals_acc = terminals_tensor.accessor<bool, 2>();
        auto actual_n_steps_acc = actual_n_steps_tensor.accessor<int64_t, 2>();
        auto is_episode_boundary = [&](int64_t env_idx, int64_t phys_idx) {
            return terminals_acc[env_idx][phys_idx] && actual_n_steps_acc[env_idx][phys_idx] <= 1;
        };

        /// @todo [Performance] 現在はバッチサイズ(B)回数分のループで C++ 側からスライスと torch::stack を行っている。
        /// GPU上でストレージを持つ場合、Pythonの `tensor[batch_indices, time_indices]` のように
        /// 高度なインデックス演算 (Advanced Indexing) を用いてベクトル化された一括抽出にリファクタリングすることで更なる学習FPSの向上が見込める

        ANET_PROFILE_SCOPE_NEXT(extract);
        for (int64_t b = 0; b < B; ++b) {
            int64_t idx1d = indices_acc[b];
            int64_t env_idx = idx1d / cap;
            int64_t time_idx = idx1d % cap;
            int64_t actual_n = actual_n_steps_acc[env_idx][time_idx];

            // 未来方向 (Unroll) のスライス抽出 ---
            batch_actions.push_back(RingSlice(storage.GetActions(), env_idx, time_idx, unroll_len, cap, squeeze_unroll));
            batch_returns.push_back(RingSlice(storage.GetTargetReturns(), env_idx, time_idx, unroll_len, cap, squeeze_unroll));
            batch_terminals.push_back(RingSlice(terminals_tensor, env_idx, time_idx, unroll_len, cap, squeeze_unroll));
            batch_actual_n.push_back(actual_n_steps_tensor[env_idx][time_idx]);

            // MuZero用などの固有情報 (存在する場合のみ)
            if (!storage.GetInfo().empty()) {
                batch_info.push_back(RingSliceDict(storage.GetInfo(), env_idx, time_idx, unroll_len, 0, cap, squeeze_unroll, {}));
            }

            // 過去方向 (Frame Stacking) のスライス抽出と境界パディング
            int64_t obs_start = time_idx - stack_count + 1;
            int64_t obs_valid_start = obs_start;
            for (int64_t k = time_idx - 1; k >= obs_start; --k) {
                int64_t phys_k = (k % cap + cap) % cap; // 負数を安全にリングバッファの末尾に折り返す
                if (is_episode_boundary(env_idx, phys_k)) {
                    obs_valid_start = k + 1;
                    break;
                }
            }
            int64_t obs_valid_len = time_idx - obs_valid_start + 1;
            int64_t obs_pad_len = stack_count - obs_valid_len;
            batch_obs.push_back(RingSliceDict(storage.GetObs(), env_idx, obs_valid_start, obs_valid_len, obs_pad_len, cap, squeeze_stack, stack_keys_));

            // N-Step先 (NextState) のスライス抽出と境界パディング
            int64_t next_obs_end = time_idx + actual_n + 1;
            int64_t next_obs_start = next_obs_end - stack_count;
            int64_t next_obs_valid_start = next_obs_start;
            for (int64_t k = next_obs_end - 2; k >= next_obs_start; --k) {
                int64_t phys_k = (k % cap + cap) % cap; // 負数を安全にリングバッファの末尾に折り返す
                if (is_episode_boundary(env_idx, phys_k)) {
                    next_obs_valid_start = k + 1;
                    break;
                }
            }
            int64_t next_obs_valid_len = next_obs_end - next_obs_valid_start;
            int64_t next_obs_pad_len = stack_count - next_obs_valid_len;
            batch_next_obs.push_back(RingSliceDict(storage.GetObs(), env_idx, next_obs_valid_start, next_obs_valid_len, next_obs_pad_len, cap, squeeze_stack, stack_keys_));
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
        out.is_weights = idx_result.is_weights;
        out.per_priority_sources = idx_result.per_priority_sources;
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
    std::shared_ptr<ReplayPriorityStore> priority_store,
    std::shared_ptr<ExperienceSampleExtractor> extractor,
    std::unique_ptr<InitialPriorityCompleter> initial_priority_completer,
    torch::Device device, bool pin_memory)
    : config_(config)
    , num_envs_(num_envs)
    , queue_controller_(std::move(queue_controller))
    , builder_(std::move(builder))
    , sampler_(sampler)
    , priority_store_(std::move(priority_store))
    , extractor_(extractor)
    , initial_priority_completer_(std::move(initial_priority_completer))
{
    capacity_per_env_ = config_.capacity / num_envs_;
    actual_capacity_ = capacity_per_env_ * num_envs_;
    queues_.resize(num_envs_);
    generations_.assign(static_cast<size_t>(actual_capacity_), 0);

    // PERではpriority storeとcompletion Moduleを対で構築し、uniformではどちらも持たない。
    if (static_cast<bool>(priority_store_) != static_cast<bool>(initial_priority_completer_)) {
        ANET_SYSTEM_ERROR("ReplayPriorityStore and InitialPriorityCompleter must be configured together.");
    }

    storage_ = std::make_unique<ReplayExperienceStorage>(num_envs_, capacity_per_env_, env_spec, config_, device, pin_memory);
    index_manager_ = std::make_unique<ValidIndexManager>(num_envs_, capacity_per_env_);
    sampled_once_.assign(static_cast<size_t>(num_envs_ * capacity_per_env_), 0);
}

void DefaultReplayBuffer::Push(const BatchExperience& batch)
{
    ANET_PROFILE_FUNC();

    std::unique_lock<std::shared_mutex> storage_lock(storage_mutex_);

    // 事前に action の info を取得しておく
    anet::TensorDict action_info = batch.action->GetInfo();
    const auto& replay_hint = batch.action->GetReplayInitialPriorityHint();
    torch::Tensor replay_hint_payload;
    if (replay_hint.has_value()) {
        ANET_PROFILE_SCOPE(replay_initial_priority_hint_cpu);
        replay_hint_payload = replay_hint->GetPayloadCpu();
        if (replay_hint_payload.size(0) != num_envs_) {
            ANET_SYSTEM_ERROR("ReplayInitialPriorityHint batch size mismatch. expected=" << num_envs_
                << " actual=" << replay_hint_payload.size(0));
        }
    }

    int64_t evicted_sampleable_count = 0;
    int64_t evicted_never_sampled_count = 0;

    for (int64_t b = 0; b < num_envs_; ++b) {
        // Step 1: Storage に重いデータを即時 Push (重複排除)

        // 単一バッチ要素のDictを構築
        anet::TensorDict single_obs = batch.state.obs[b];
        anet::TensorDict single_info = action_info.empty() ? anet::TensorDict() : action_info[b];

        const int64_t logical_time_idx = index_manager_->GetWriteCursor(b);
        int64_t time_idx = storage_->Push(b, single_obs, batch.action->GetAction()[b], single_info);
        const int64_t flat_slot_index = b * capacity_per_env_ + time_idx;
        {
            std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
            RecordEvictionIfSampleable(b, time_idx, &evicted_sampleable_count, &evicted_never_sampled_count);
            sampled_once_[static_cast<size_t>(flat_slot_index)] = 0;
            OnSlotWritten(flat_slot_index);
            index_manager_->MarkWritten(b, time_idx);
            index_manager_->AdvanceWriteCursor(b); // カーソルを進める

        }

        // 正常なステップを先に Queue に入れる (タイムトラベル防止)
        QueueRecord rec;
        rec.time_idx = time_idx;
        rec.logical_time_idx = logical_time_idx;
        rec.reward = batch.reward[b].item<float>();
        rec.done = batch.next_state.done[b].item<bool>();
        rec.truncated = batch.next_state.truncated[b].item<bool>();
        if (replay_hint.has_value()) {
            // 共通層では列を解釈せず、CPU化済みの1行をinline保持できる小配列へコピーする。
            auto payload = replay_hint_payload.accessor<float, 2>();
            ReplayInitialPriorityHintRow row;
            row.reserve(static_cast<size_t>(replay_hint_payload.size(1)));
            for (int64_t k = 0; k < replay_hint_payload.size(1); ++k) {
                row.push_back(payload[b][k]);
            }
            rec.replay_initial_priority_hint = std::move(row);
        }
        queues_[b].Push(rec);

        // Truncatedのパラドックス対策 (ダミーステップの挿入)
        if (batch.next_state.truncated[b].item<bool>()) {
            anet::TensorDict terminal_obs = batch.next_state.obs[b];
            storage_->PushTerminalDummy(b, terminal_obs);

            int64_t dummy_idx = (time_idx + 1) % capacity_per_env_;
            const int64_t dummy_logical_idx = index_manager_->GetWriteCursor(b);
            {
                std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
                RecordEvictionIfSampleable(b, dummy_idx, &evicted_sampleable_count, &evicted_never_sampled_count);
                sampled_once_[static_cast<size_t>(b * capacity_per_env_ + dummy_idx)] = 0;
                OnSlotWritten(b * capacity_per_env_ + dummy_idx);
                index_manager_->MarkDummy(b, dummy_idx);
                index_manager_->AdvanceWriteCursor(b); // ダミー書き込み分もカーソルを進める

            }

            QueueRecord dummy_rec;
            dummy_rec.time_idx = dummy_idx;
            dummy_rec.logical_time_idx = dummy_logical_idx;
            dummy_rec.reward = 0.0f;
            dummy_rec.done = true;
            dummy_rec.truncated = false;
            dummy_rec.is_dummy = true;
            queues_[b].Push(dummy_rec);
        }

        ProcessQueue(b, rec.replay_initial_priority_hint, rec.logical_time_idx);
    }

    StoreLastEvictionStats(evicted_sampleable_count, evicted_never_sampled_count);
    InvalidateAccessorCacheForStorage();
}

void DefaultReplayBuffer::MarkSampledOnce(const torch::Tensor& indices) const
{
    if (!indices.defined()) return;

    auto indices_cpu = indices.device().is_cpu() ? indices.contiguous() : indices.to(torch::kCPU).contiguous();
    auto acc = indices_cpu.accessor<int64_t, 1>();
    for (int64_t i = 0; i < indices_cpu.size(0); ++i) {
        const int64_t flat_idx = acc[i];
        if (flat_idx >= 0 && flat_idx < static_cast<int64_t>(sampled_once_.size())) {
            sampled_once_[static_cast<size_t>(flat_idx)] = 1;
        }
    }
}

void DefaultReplayBuffer::RecordEvictionIfSampleable(
    int64_t env_idx,
    int64_t time_idx,
    int64_t* evicted_sampleable_count,
    int64_t* evicted_never_sampled_count)
{
    if (!index_manager_->IsOverwritingSampleable(env_idx, time_idx, config_.stack_count, config_.muzero.unroll_steps, config_.n_step)) {
        return;
    }

    const int64_t flat_idx = env_idx * capacity_per_env_ + time_idx;
    ++(*evicted_sampleable_count);
    if (!sampled_once_[static_cast<size_t>(flat_idx)]) {
        ++(*evicted_never_sampled_count);
    }
}

void DefaultReplayBuffer::StoreLastEvictionStats(int64_t evicted_sampleable_count, int64_t evicted_never_sampled_count)
{
    std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
    last_evicted_sampleable_count_ = evicted_sampleable_count;
    last_evicted_never_sampled_count_ = evicted_never_sampled_count;
    last_evicted_never_sampled_ratio_ = evicted_sampleable_count > 0
        ? static_cast<float>(evicted_never_sampled_count) / static_cast<float>(evicted_sampleable_count)
        : std::numeric_limits<float>::quiet_NaN();
}

void DefaultReplayBuffer::ProcessQueue(
    int64_t env_idx,
    const std::optional<ReplayInitialPriorityHintRow>& current_hint,
    int64_t current_logical_idx)
{
    ANET_PROFILE_FUNC();

    auto sequences = queue_controller_->ExtractSequences(queues_[env_idx]);

    int64_t valid_count = 0;
    std::vector<InitialPriorityCompleter::PendingEntry> completion_entries;
    if (initial_priority_completer_) completion_entries.reserve(sequences.size());

    for (const auto& seq : sequences) {
        // 念のため空チェック
        if (seq.empty()) continue;

        //ダミーステップ自身が起点(state)となっているシーケンスはStorageに登録しない
        if (seq[0].is_dummy) {
            ++valid_count;
            continue;
        }

        // Builderで割引報酬和を計算
        ReplayExperience exp = builder_->Build(seq);

        // 先頭の time_idx を取り出し、Storageを上書き(Update)
        int64_t time_idx = seq.front().time_idx;
        storage_->Update(env_idx, time_idx, exp);

        ++valid_count;
        if (initial_priority_completer_) {
            // ExperienceQueueで系列が確定したreal transitionから、priority完成に必要な小さい情報だけを移す。
            completion_entries.push_back(InitialPriorityCompleter::PendingEntry{
                .logical_time_idx = seq.front().logical_time_idx,
                .flat_slot_index = env_idx * capacity_per_env_ + time_idx,
                .target_return = exp.target_return,
                .terminal = exp.terminal,
                .truncated = exp.truncated,
                .actual_n_steps = exp.actual_n_steps,
                .start_hint = seq.front().replay_initial_priority_hint,
            });
        }
    }

    if (valid_count == 0 && !initial_priority_completer_) return;

    // pending登録、sampleable化、priority/source適用を同じmetadata排他区間で公開する。
    std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
    if (initial_priority_completer_) {
        for (auto& entry : completion_entries) {
            initial_priority_completer_->Enqueue(env_idx, std::move(entry));
        }
    }
    // pendingを先に登録してから封印解除し、sampleableだがpriority未完成という状態を作らない。
    for (int64_t i = 0; i < valid_count; ++i) {
        index_manager_->MarkValid(env_idx);
    }
    if (initial_priority_completer_) {
        initial_priority_completer_->CompleteReady(
            env_idx, current_hint, current_logical_idx, *index_manager_, *priority_store_);
    }
}

int64_t DefaultReplayBuffer::OnSlotWritten(int64_t flat_slot_index)
{
    auto& generation = generations_[static_cast<size_t>(flat_slot_index)];
    if (generation == std::numeric_limits<int64_t>::max()) {
        ANET_SYSTEM_ERROR("Replay item generation overflow. flat_slot_index=" << flat_slot_index);
    }
    ++generation;
    if (priority_store_) priority_store_->InvalidateSlot(flat_slot_index);
    return generation;
}

int64_t DefaultReplayBuffer::EncodeReplayItemKey(int64_t flat_slot_index) const
{
    const int64_t generation = generations_[static_cast<size_t>(flat_slot_index)];
    if (generation <= 0 || generation > (std::numeric_limits<int64_t>::max() - flat_slot_index) / actual_capacity_) {
        ANET_SYSTEM_ERROR("Replay item key overflow. generation=" << generation
            << " actual_capacity=" << actual_capacity_ << " flat_slot_index=" << flat_slot_index);
    }
    return generation * actual_capacity_ + flat_slot_index;
}

void DefaultReplayBuffer::Sample(ExperienceSamples& out_samples, int64_t minibatch_size, float beta) const
{
    ANET_PROFILE_FUNC();

    ANET_PROFILE_SCOPE(storage_lock_wait);
    std::shared_lock<std::shared_mutex> storage_lock(storage_mutex_);
    ANET_PROFILE_SCOPE_END(storage_lock_wait);

    IndexSampleResult idx_result;

    ANET_PROFILE_SCOPE(metadata_lock_wait);
    std::unique_lock<std::mutex> metadata_lock(metadata_mutex_);
    ANET_PROFILE_SCOPE_END(metadata_lock_wait);

    ANET_PROFILE_SCOPE(valid_indices);
    auto valid_1d = index_manager_->GetValidIndices1D(config_.stack_count, config_.muzero.unroll_steps, config_.n_step);
    ANET_PROFILE_SCOPE_END(valid_indices);
    ANET_ASSERT_MSG(valid_1d.size(0) >= minibatch_size, "Not enough valid samples in ReplayBuffer. size=" << valid_1d.size(0) << " minibatch_size=" << minibatch_size);

    ANET_PROFILE_SCOPE(sample_indices);
    idx_result = sampler_->SampleIndices(minibatch_size, valid_1d, beta);
    ANET_PROFILE_SCOPE_END(sample_indices);
    MarkSampledOnce(idx_result.flat_slot_indices);
    metadata_lock.unlock();

    ANET_PROFILE_SCOPE(extract);
    extractor_->ExtractSamples(out_samples, *storage_, idx_result, config_.stack_count, config_.muzero.unroll_steps);
    auto flat_indices = idx_result.flat_slot_indices.to(torch::kCPU).contiguous();
    auto flat_acc = flat_indices.accessor<int64_t, 1>();
    std::vector<int64_t> item_keys(static_cast<size_t>(flat_indices.size(0)));
    for (int64_t i = 0; i < flat_indices.size(0); ++i) {
        item_keys[static_cast<size_t>(i)] = EncodeReplayItemKey(flat_acc[i]);
    }
    out_samples.replay_item_keys = torch::tensor(
        item_keys, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    ANET_PROFILE_SCOPE_END(extract);
}

int64_t DefaultReplayBuffer::Size() const
{
    ANET_PROFILE_FUNC();

    // 生のカウントではなく、Stack/Unrollを考慮して安全なサンプリング可能な数を返す
    std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
    return index_manager_->GetSampleableCount(config_.stack_count, config_.muzero.unroll_steps, config_.n_step);
}

ReplayPriorityUpdateResult DefaultReplayBuffer::UpdatePriorities(
    const std::vector<int64_t>& item_keys, const std::vector<float>& priorities)
{
    ANET_PROFILE_FUNC();

    ReplayPriorityUpdateResult result;
    if (!priority_store_) return result;
    if (item_keys.size() != priorities.size()) {
        ANET_SYSTEM_ERROR("ReplayBuffer::UpdatePriorities size mismatch. item_keys=" << item_keys.size()
            << " priorities=" << priorities.size());
    }

    std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
    std::vector<int64_t> valid_slots;
    std::vector<float> valid_priorities;
    valid_slots.reserve(item_keys.size());
    valid_priorities.reserve(priorities.size());

    // どのleafも変更する前にbatch全体を検証する。
    for (size_t i = 0; i < item_keys.size(); ++i) {
        const int64_t key = item_keys[i];
        const float priority = priorities[i];
        if (key < 0 || !std::isfinite(priority) || priority < 0.0f) {
            ANET_SYSTEM_ERROR("ReplayBuffer::UpdatePriorities invalid input. key=" << key
                << " priority=" << priority << " expected non-negative finite priority and key");
        }
        const int64_t generation = key / actual_capacity_;
        const int64_t flat_slot_index = key % actual_capacity_;
        if (generation <= 0) {
            ANET_SYSTEM_ERROR("ReplayBuffer::UpdatePriorities generation must be positive. key=" << key);
        }
        const int64_t current_generation = generations_[static_cast<size_t>(flat_slot_index)];
        if (generation > current_generation) {
            ANET_SYSTEM_ERROR("ReplayBuffer::UpdatePriorities future generation. key=" << key
                << " current_generation=" << current_generation);
        }
    }
    // 過去generationだけを要素単位で棄却し、現generationの更新を物理slotへ変換する。
    for (size_t i = 0; i < item_keys.size(); ++i) {
        const int64_t generation = item_keys[i] / actual_capacity_;
        const int64_t flat_slot_index = item_keys[i] % actual_capacity_;
        if (generation < generations_[static_cast<size_t>(flat_slot_index)]) {
            ++result.stale_count;
            continue;
        }
        valid_slots.push_back(flat_slot_index);
        valid_priorities.push_back(priorities[i]);
    }
    if (!valid_slots.empty()) {
        auto applied = priority_store_->ApplyLearnerPriorities(valid_slots, valid_priorities);
        applied.stale_count = result.stale_count;
        result = applied;
        InvalidateAccessorCacheForPriority();
    }
    priority_update_stale_drop_count_ += result.stale_count;
    return result;
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
    const auto ratio = [](int64_t numerator, int64_t denominator) {
        return denominator > 0
            ? static_cast<float>(numerator) / static_cast<float>(denominator)
            : std::numeric_limits<float>::quiet_NaN();
    };
    if (key == PER_ACTOR_COMPLETION_ATTEMPT_COUNT || key == PER_ACTOR_COMPLETION_SUCCESS_COUNT
        || key == PER_ACTOR_COMPLETION_SUCCESS_RATIO || key == PER_ACTOR_TRUNCATION_FALLBACK_COUNT
        || key == PER_ACTOR_TRUNCATION_FALLBACK_RATIO || key == PER_ACTOR_NONFINITE_FALLBACK_COUNT
        || key == PER_ACTOR_NONFINITE_FALLBACK_RATIO) {
        std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
        if (!initial_priority_completer_) return std::nullopt;
        const auto stats = initial_priority_completer_->GetStats();
        if (key == PER_ACTOR_COMPLETION_ATTEMPT_COUNT) return static_cast<float>(stats.attempt_count);
        if (key == PER_ACTOR_COMPLETION_SUCCESS_COUNT) return static_cast<float>(stats.success_count);
        if (key == PER_ACTOR_COMPLETION_SUCCESS_RATIO) {
            return ratio(stats.success_count, stats.attempt_count);
        }
        if (key == PER_ACTOR_TRUNCATION_FALLBACK_COUNT) return static_cast<float>(stats.truncation_fallback_count);
        if (key == PER_ACTOR_TRUNCATION_FALLBACK_RATIO) {
            return ratio(stats.truncation_fallback_count, stats.attempt_count);
        }
        if (key == PER_ACTOR_NONFINITE_FALLBACK_COUNT) return static_cast<float>(stats.nonfinite_fallback_count);
        if (key == PER_ACTOR_NONFINITE_FALLBACK_RATIO) {
            return ratio(stats.nonfinite_fallback_count, stats.attempt_count);
        }
    }
    if (key == PER_PRIORITY_UPDATE_STALE_DROP_COUNT) {
        std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
        if (!priority_store_) return std::nullopt;
        return static_cast<float>(priority_update_stale_drop_count_);
    }
    if (key == PER_LAST_EVICTED_NEVER_SAMPLED_RATIO) {
        std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
        return priority_store_ ? std::optional<float>(last_evicted_never_sampled_ratio_) : std::nullopt;
    }
    if (priority_store_) {
        std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
        return priority_store_->GetScalar(key);
    }
    return std::nullopt;
}

std::optional<torch::Tensor> DefaultReplayBuffer::GetTensor(const std::string& key, int64_t index) const
{
    if (key == PER_VALUES) {
        std::lock_guard<std::mutex> metadata_lock(metadata_mutex_);
        return priority_store_ ? priority_store_->GetPriorityTensor(index) : std::nullopt;
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
            if (!priority_store_) return std::nullopt;
            tensor = priority_store_->GatherPriorityRows(valid_1d);
            if (key == PER_DIST) {
                // PER_DIST は正規化サンプリング確率 p/total を返す（SampleIndices の prob=p/total と同義）。
                const float total = priority_store_->GetScalar(PER_TOTAL).value_or(0.0f);
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

struct PrefetchingReplayBuffer::State {
    using PrefetchedBatch = anet::transfer::DeviceTransfer<ExperienceSamples>;

    State(ReplayBuffer& inner, torch::Device target_device)
        : inner(inner)
        , target_device(std::move(target_device))
        , pool(std::make_unique<anet::PinnedThreadPool>(1, "ReplayBufferPrefetch"))
    {
        if (this->target_device.is_cuda()) {
            copy_stream = at::cuda::getStreamFromPool(false);
        }
    }

    /// inner からサンプルし transfer まで行って PrefetchedBatch を作る（prefetch worker 本体）。
    PrefetchedBatch Fetch(int64_t minibatch_size, float beta);

    /// CPU サンプルを pin→H2D→event record して DeviceTransfer 化する（CPU device 時は同期 To のみ）。
    PrefetchedBatch TransferSamples(ExperienceSamples cpu_samples);

    /// future 未起動なら次の 1-deep prefetch を worker に投入する（mutex 保持前提）。
    void LaunchPrefetchLocked(int64_t minibatch_size, float beta);

    /// in-flight prefetch future の完了を待つ（消費はしない。mutex 保持前提）。
    void WaitForPrefetchLocked();

    /// armed 後の Push を worker FIFO に write-behind 投入する（mutex 保持前提）。
    void EnqueueDelayedPushLocked(BatchExperience batch_exp);

    /// 完了済み delayed push を非ブロッキングで回収する（mutex 保持前提）。
    void PruneCompletedPushesLocked();

    /// 残る delayed push を全て完了待ちする（mutex 保持前提）。
    void WaitForQueuedPushesLocked();

    /// shutdown: prefetch/push 完了待ち→event_recycler.Drain→pool 停止まで行う（内部で lock）。
    void StopPrefetch();

    ReplayBuffer& inner;
    torch::Device target_device;
    std::mutex mutex;
    std::unique_ptr<anet::PinnedThreadPool> pool;
    std::optional<at::cuda::CUDAStream> copy_stream;
    anet::transfer::EventRecycler<ExperienceSamples> event_recycler;
    std::future<PrefetchedBatch> future;
    std::deque<std::future<void>> push_futures;
};

PrefetchingReplayBuffer::State::PrefetchedBatch PrefetchingReplayBuffer::State::Fetch(int64_t minibatch_size, float beta)
{
    ANET_PROFILE_SCOPE(sample);
    ExperienceSamples cpu_samples;
    inner.Sample(cpu_samples, minibatch_size, beta);

    return TransferSamples(std::move(cpu_samples));
}

PrefetchingReplayBuffer::State::PrefetchedBatch PrefetchingReplayBuffer::State::TransferSamples(ExperienceSamples cpu_samples)
{
    if (!target_device.is_cuda()) {
        ANET_PROFILE_SCOPE_FULL(to, "PrefetchingReplayBuffer::Fetch.to");

        // CPU learnerではCUDA用のpin/stream/eventを使わず、抽出済みsampleだけをCPUへそろえる。
        return PrefetchedBatch(std::move(cpu_samples), target_device);
    }

    ANET_PROFILE_SCOPE_FULL(to, "PrefetchingReplayBuffer::Fetch.to");

    // CUDA learnerではworker側でCPU tensorをpinned化し、copy streamへH2D転送を積む。
    // 転送元とeventはSample側でevent_recyclerへ渡し、完了後にまとめて回収・再利用する。
    return PrefetchedBatch(
        std::move(cpu_samples),
        target_device,
        copy_stream.value(),
        event_recycler);
}

void PrefetchingReplayBuffer::State::LaunchPrefetchLocked(int64_t minibatch_size, float beta)
{
    if (future.valid()) return;

    future = pool->EnqueueFuture(0, [this, minibatch_size, beta]() {
        return Fetch(minibatch_size, beta);
    });
}

void PrefetchingReplayBuffer::State::WaitForPrefetchLocked()
{
    if (future.valid()) {
        future.wait();
    }
}

void PrefetchingReplayBuffer::State::EnqueueDelayedPushLocked(BatchExperience batch_exp)
{
    push_futures.push_back(pool->EnqueueFuture(0, [this, batch_exp = std::move(batch_exp)]() mutable {
        ANET_PROFILE_SCOPE_FULL(push_worker, "PrefetchingReplayBuffer::Push.push_worker");
        inner.Push(batch_exp);
    }));
}

void PrefetchingReplayBuffer::State::PruneCompletedPushesLocked()
{
    while (!push_futures.empty()) {
        auto& front = push_futures.front();
        if (front.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
        front.get();
        push_futures.pop_front();
    }
}

void PrefetchingReplayBuffer::State::WaitForQueuedPushesLocked()
{
    while (!push_futures.empty()) {
        push_futures.front().get();
        push_futures.pop_front();
    }
}

void PrefetchingReplayBuffer::State::StopPrefetch()
{
    {
        std::lock_guard<std::mutex> lock(mutex);
        WaitForPrefetchLocked();
        WaitForQueuedPushesLocked();
        event_recycler.Drain();
    }

    if (pool) {
        pool->WaitAll();
        pool->Stop();
    }
}

PrefetchingReplayBuffer::PrefetchingReplayBuffer(std::shared_ptr<ReplayBuffer> inner, torch::Device target_device)
    : inner_(std::move(inner))
    , target_device_(std::move(target_device))
{
    ANET_ASSERT_MSG(inner_ != nullptr, "PrefetchingReplayBuffer requires a non-null inner ReplayBuffer.");
    state_ = std::make_unique<State>(*inner_, target_device_);
}

PrefetchingReplayBuffer::~PrefetchingReplayBuffer()
{
    if (state_) {
        state_->StopPrefetch();
    }
}

void PrefetchingReplayBuffer::Push(const BatchExperience& batch_exp)
{
    ANET_PROFILE_FUNC();

    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->future.valid()) {
        inner_->Push(batch_exp);
        return;
    }

    ANET_PROFILE_SCOPE(snapshot);
    // prefetch armed 後は現在のfutureの後、次のFetchの前にPushをFIFO投入する。
    // runner側でstable化済みのBatchExperienceを浅く保持し、Tensor storageの寿命だけ延ばす。
    BatchExperience snapshot = batch_exp;
    {
        ANET_PROFILE_SCOPE_NEXT(enqueue_push);
        state_->EnqueueDelayedPushLocked(std::move(snapshot));
    }
}

void PrefetchingReplayBuffer::Sample(ExperienceSamples& out_samples, int64_t minibatch_size, float beta) const
{
    ANET_PROFILE_FUNC();

    State::PrefetchedBatch batch;
    {
        std::unique_lock<std::mutex> lock(state_->mutex);
        if (state_->future.valid()) {
            ANET_PROFILE_SCOPE(consume_wait);

            // 前回起動した1-deep prefetchをここで消費し、同じ排他区間で次のprefetchを起動する。
            // futureの有無をこの順序で更新することで、二重起動せず常に最大1件だけ先読みする。
            batch = state_->future.get();
            state_->PruneCompletedPushesLocked();
            state_->LaunchPrefetchLocked(minibatch_size, beta);
        } else {
            ANET_PROFILE_SCOPE(cold_fetch);

            // 初回はfutureが無いため、state mutex内で同期Sampleし、Push/UpdatePrioritiesとの順序を固定する。
            batch = state_->Fetch(minibatch_size, beta);
            state_->LaunchPrefetchLocked(minibatch_size, beta);
        }
    }

    if (!batch.ready_event) {
        out_samples = std::move(batch.device_samples);
        return;
    }

    {
        ANET_PROFILE_SCOPE(event_wait);

        // CUDA転送はcopy streamで済ませるため、利用側のcurrent streamからevent待ちしてから返す。
        auto current_stream = at::cuda::getCurrentCUDAStream();
        batch.ready_event->block(current_stream);
        anet::transfer::RecordStreamOn(batch.device_samples, current_stream);
    }
    out_samples = std::move(batch.device_samples);

    {
        ANET_PROFILE_SCOPE(retire_ready_event);

        state_->event_recycler.Retire(std::move(*batch.ready_event), std::move(batch.retained_source));
    }
}

int64_t PrefetchingReplayBuffer::Size() const
{
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->PruneCompletedPushesLocked();
    return inner_->Size();
}

ReplayPriorityUpdateResult PrefetchingReplayBuffer::UpdatePriorities(
    const std::vector<int64_t>& item_keys, const std::vector<float>& priorities)
{
    ANET_PROFILE_FUNC();

    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->future.valid()) {
        ANET_PROFILE_SCOPE(wait_prefetch);

        // background Sample と priority 更新の順序を固定する。
        state_->WaitForPrefetchLocked();
        state_->WaitForQueuedPushesLocked();
    }
    return inner_->UpdatePriorities(item_keys, priorities);
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


// ===========================================================================
// Factory
// ===========================================================================

std::shared_ptr<ReplayBuffer> anet::rl::CreateReplayBuffer(
    const ReplayBufferConfig& config, const EnvSpec& env_spec, int64_t num_envs,
    torch::Device storage_device, bool pin_memory, std::optional<uint64_t> seed,
    std::unique_ptr<InitialPriorityEstimator> initial_priority_estimator)
{
    // capacity の割り切れ補正
    int64_t capacity_per_env = config.capacity / num_envs;
    const int64_t actual_capacity = capacity_per_env * num_envs;
    const int64_t required_capacity_per_env = std::max<int64_t>(1, config.n_step) + 1;
    if (capacity_per_env < required_capacity_per_env) {
        ANET_SYSTEM_ERROR("ReplayBuffer capacity per env is too small. replay_capacity=" << config.capacity
            << " num_envs=" << num_envs << " capacity_per_env=" << capacity_per_env
            << " required_min=" << required_capacity_per_env);
    }
    if (!std::isfinite(config.per_initial_priority) || config.per_initial_priority < 0.0f) {
        ANET_SYSTEM_ERROR("Invalid per_initial_priority: value=" << config.per_initial_priority
            << " expected finite value >= 0; use per_initial_priority_mode=max for maximum initialization");
    }
    if (config.sampler_type == ReplaySamplerType::UNIFORM
        && config.per_initial_priority_mode != ReplayInitialPriorityMode::FIXED) {
        ANET_SYSTEM_ERROR("per_initial_priority_mode requires prioritized replay when mode is not fixed.");
    }
    if (config.per_initial_priority_mode == ReplayInitialPriorityMode::ACTOR_APPROX
        && !initial_priority_estimator) {
        ANET_SYSTEM_ERROR("per_initial_priority_mode=actor_approx requires InitialPriorityEstimator.");
    }
    if (capacity_per_env * num_envs != config.capacity) {
        LOG::info() << "ReplayBuffer capacity adjusted to be divisible by num_envs."
            << " config.capacity=" << config.capacity << " actual_capacity=" << capacity_per_env * num_envs;
    }

    // 内部コンポーネントの構築
    auto queue_controller = std::make_unique<NStepQueueController>(config.n_step);
    auto builder = std::make_unique<DefaultExperienceBuilder>(config.gamma);

    std::shared_ptr<ReplayExperienceSampler> sampler;
    std::shared_ptr<ReplayPriorityStore> priority_store;
    if (config.sampler_type == ReplaySamplerType::UNIFORM) {
        sampler = std::make_shared<UniformSampler>(seed);
        priority_store = nullptr;
    } else {
        auto per = std::make_shared<PrioritizedSampler>(actual_capacity, config.per_alpha, seed);
        sampler = per;
        priority_store = per;
    }
    auto initial_priority_completer = CreateInitialPriorityCompleter(
        config, num_envs, std::move(initial_priority_estimator));

    auto extractor = std::make_shared<DefaultSampleExtractor>(config.stack_keys);

    return std::make_shared<DefaultReplayBuffer>(
        config, env_spec, num_envs, std::move(queue_controller), std::move(builder), sampler,
        priority_store, extractor, std::move(initial_priority_completer), storage_device, pin_memory);
}
