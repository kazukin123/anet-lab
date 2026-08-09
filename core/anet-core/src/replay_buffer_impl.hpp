// replay_buffer_impl.hpp

#pragma once

#include "anet/replay_buffer.hpp"
#include <algorithm>
#include <limits>
#include <vector>
#include <memory>
#include <deque>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <type_traits>
#include <c10/util/SmallVector.h>
#include <torch/torch.h>
#include "anet/tensor_util.hpp"

namespace anet::rl {

    struct DecodedReplayItemKey {
        int64_t generation = 0;      ///< slotへ最後に書き込まれた世代
        int64_t flat_slot_index = 0; ///< ReplayBuffer全体での物理slot index
    };

    /// generation付きopaque keyをoverflow検証付きで生成する内部helper。
    int64_t EncodeReplayItemKeyChecked(
        int64_t generation, int64_t flat_slot_index, int64_t actual_capacity);

    /// generation付きopaque keyを算術境界検証付きで分解する内部helper。
    DecodedReplayItemKey DecodeReplayItemKeyChecked(int64_t key, int64_t actual_capacity);

    /// Actor初期値とLearner更新値の比較ペアから呼び出し単位の診断統計を埋める。
    void FillActorLearnerComparison(
        const std::vector<std::pair<float, float>>& pairs, ReplayPriorityUpdateResult* result);


    // ======================================================
    // Queue
    // ======================================================

    /// ReplayBuffer内部でcompletionまで運ぶ、schema非依存のhint行。
    using ReplayInitialPriorityHintRow = c10::SmallVector<float, 4>;

    /// Storage上の重いデータ（Obs等）とは分離された、N-Step計算用の軽量メタデータ
    struct QueueRecord {
        int64_t time_idx;       ///< Storage上のインデックス (Facadeが管理するためのKey)
        int64_t logical_time_idx = 0;
        float reward;           ///< 即時報酬
        bool done;              ///< エピソード終了フラグ
        bool truncated;         ///< タイムアップ等による打ち切りフラグ
        bool is_dummy = false;  ///< Truncated時の終端計算用ダミーフラグ
        std::optional<ReplayInitialPriorityHintRow> replay_initial_priority_hint;
    };

    /// 各環境(Env)ごとの未処理メタデータを一時保持するキュー
    class ExperienceQueue {
    public:
        void Push(const QueueRecord& record);
        void Pop(size_t k);
        std::vector<QueueRecord> Peek(size_t k) const;
        size_t Size() const;
        void Clear();
    private:
        std::deque<QueueRecord> buffer_;
    };


    // ======================================================
    // Queue Controller & Builder
    // ======================================================

    /// N-Step計算用に切り出された時系列レコード
    using ExperienceSequence = std::vector<QueueRecord>;

    class ExperienceQueueController {
    public:
        /// Queueを操作し、計算準備が整った系列のリストを返す。使用済み要素はPopする。
        virtual std::vector<ExperienceSequence> ExtractSequences(ExperienceQueue& queue) = 0;
        virtual ~ExperienceQueueController() = default;
    };

    /// Storageに後から上書き(Update)するための、構築済み経験データ
    struct ReplayExperience {
        float target_return;    ///< 計算された割引報酬和など (Value)
        bool terminal;          ///< 最終的な終了フラグ
        int actual_n_steps;     ///< 実際に進んだステップ数
        bool truncated = false; ///< true terminalではないtruncation境界
    };

    class ReplayExperienceBuilder {
    public:
        /// 系列(Sequence)を受け取り、アルゴリズムに応じた ReplayExperience を構築する
        virtual ReplayExperience Build(const ExperienceSequence& sequence) const = 0;
        virtual ~ReplayExperienceBuilder() = default;
    };
   

    // ======================================================
    // Valid Index Manager
    // ======================================================

    /// Storageから安全にサンプリング可能なインデックス範囲を管理するクラス.
    /// Storageには随時データを書き込むが、完全にサンプリング可能になるまでに遅延が存在する事から必要となる。
    class ValidIndexManager {
    public:
        ValidIndexManager(int64_t num_envs, int64_t capacity_per_env);

        /// 指定位置にデータが書き込まれたことを通知 (この時点ではまだサンプリング封印状態)
        void MarkWritten(int64_t env_idx, int64_t time_idx);

        /// N-Step等を経て「未来」が担保され、完全にサンプリング可能になったことを通知 (封印解除)
        void MarkValid(int64_t env_idx);

        /// ダミーデータが書き込まれた事を通知
        void MarkDummy(int64_t env_idx, int64_t time_idx);

		/// 書き込みカーソルを進める
        void AdvanceWriteCursor(int64_t env_idx);

        /// Stack/Unroll 制約を考慮し、安全に引ける 1D インデックスのリストを返す
        torch::Tensor GetValidIndices1D(int stack_count, int unroll_steps, int n_step) const;

        int64_t GetValidCount() const;

        int64_t GetSampleableCount(int stack_count, int unroll_steps, int n_step) const;

        bool IsOverwritingSampleable(int64_t env_idx, int64_t time_idx, int stack_count, int unroll_steps, int n_step) const;
        int64_t GetWriteCursor(int64_t env_idx) const { return write_cursors_[env_idx]; }
        /// 指定logical indexが上書き境界、未来観測、unroll終端の全条件を満たすか判定する。
        bool IsLogicalSampleable(int64_t env_idx, int64_t logical_idx, int unroll_steps, int n_step) const;
    private:
        struct LogicalSampleableRange {
            int64_t start = 0; ///< 未上書きで保持されている最古のlogical index
            int64_t end = -1;  ///< 未来観測とunroll終端が確定済みの最新logical index

            bool Contains(int64_t logical_idx) const
            {
                return logical_idx >= start && logical_idx <= end;
            }
        };

        /**
         * @brief write、valid、unrollの制約を満たすlogical indexの閉区間を返す。
         *
         * dummyは論理的な時系列幅へ含まれるため、このrangeでは除外しない。
         * 列挙と上書き判定の各処理が物理slotのdummy状態を別途判断する。
         */
        std::optional<LogicalSampleableRange> GetLogicalSampleableRange(
            int64_t env_idx, int unroll_steps, int n_step) const;

        template <class Fn>
        void ForEachSampleableIndex(int64_t env, int stack_count, int unroll_steps, int n_step, Fn&& fn) const
        {
            (void)stack_count;

            const auto range = GetLogicalSampleableRange(env, unroll_steps, n_step);
            if (!range.has_value()) return;

            int64_t start_phys = range->start % capacity_per_env_;
            int64_t end_phys = range->end % capacity_per_env_;

            auto visit_range = [&](int64_t p_start, int64_t p_end) {
                for (int64_t p = p_start; p <= p_end; ++p) {
                    if (!is_dummy_[env * capacity_per_env_ + p]) {
                        fn(env * capacity_per_env_ + p);
                    }
                }
            };

            // 物理インデックス昇順にして、PER側の binary_search 前提を保つ。
            if (start_phys <= end_phys) {
                visit_range(start_phys, end_phys);
            } else {
                visit_range(0, end_phys);
                visit_range(start_phys, capacity_per_env_ - 1);
            }
        }

        int64_t num_envs_;
        int64_t capacity_per_env_;
        std::vector<int64_t> valid_cursors_;
        std::vector<int64_t> write_cursors_;
        std::vector<bool> is_dummy_;
    };


    // ======================================================
    // Storage (CPU/GPU対応)
    // ======================================================

    class ReplayExperienceStorage : public anet::Module {
    public:
        ReplayExperienceStorage(int64_t num_envs, int64_t capacity_per_env, const EnvSpec& spec, const ReplayBufferConfig& config, torch::Device device, bool pin_memory);

        /// 重いデータ（Dict等）を即時追加し、書き込まれた time_idx を返す
        int64_t Push(int64_t env_idx, const anet::TensorDict& obs, const torch::Tensor& action, const anet::TensorDict& info);

        /// Builderが構築したメタデータを、指定したインデックスに上書き(遅延反映)する
        void Update(int64_t env_idx, int64_t time_idx, const ReplayExperience& exp);

        /// 終端到達時の「ダミーステップ」を書き込む（パラドックス回避用）
        void PushTerminalDummy(int64_t env_idx, const anet::TensorDict& terminal_obs);

		/// デバッグ用: Storageの内容をログに出力する
        void DumpToLog() const;
    public:
        // 読み取りインターフェース (Extractor用) 
        const anet::TensorDict& GetObs() const { return obs_storage_; }
        const anet::TensorDict& GetInfo() const { return info_storage_; }
        const torch::Tensor& GetActions() const { return actions_; }
        const torch::Tensor& GetTargetReturns() const { return target_returns_; }
        const torch::Tensor& GetTerminals() const { return terminals_; }
        const torch::Tensor& GetActualNSteps() const { return actual_n_steps_; }
    public:
        // 可視化用
        std::optional<float> GetScalar(const std::string& key, int64_t index) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index) const override;
    private:
        int64_t num_envs_;
        int64_t capacity_per_env_;
        std::vector<int64_t> write_cursors_;
        torch::Device device_;

        // --- 物理テンソル群 ---
        anet::TensorDict obs_storage_;
        anet::TensorDict info_storage_;
        torch::Tensor actions_;
        torch::Tensor target_returns_;
        torch::Tensor terminals_;
        torch::Tensor actual_n_steps_;
    };
    

    // ======================================================
    // Sampler & Extractor (Controller)
    // ======================================================

    struct IndexSampleResult {
        torch::Tensor flat_slot_indices; ///< [B] 内部物理位置
        torch::Tensor sampling_prob; ///< [B] probabilities
        torch::Tensor is_weights;    ///< [B] importance sampling weights
        torch::Tensor per_priority_sources; ///< [B] ReplayPrioritySource
    };

    class ReplayExperienceSampler {
    public:
        virtual IndexSampleResult SampleIndices(int64_t batch_size, const torch::Tensor& valid_indices_1d, float beta) = 0;
        virtual ~ReplayExperienceSampler() = default;
    };

    class ExperienceSampleExtractor {
    public:
        /// サンプリングされたインデックスに基づき、StorageからStack/Unrollを考慮したサンプルを抽出する
        virtual void ExtractSamples(
            ExperienceSamples& out_samples,
            const ReplayExperienceStorage& storage,
            const IndexSampleResult& idx_result,
            int stack_count, int unroll_steps) const = 0;
        virtual ~ExperienceSampleExtractor() = default;
    };

    class ReplayPriorityStore {
    public:
        virtual void InvalidateSlot(int64_t flat_slot_index) = 0;
        virtual void SetRawInitialPriority(int64_t flat_slot_index, float raw_priority, ReplayPrioritySource source) = 0;
        virtual void SetMaxInitialPriority(int64_t flat_slot_index) = 0;
        virtual ReplayPriorityUpdateResult ApplyLearnerPriorities(
            const std::vector<int64_t>& flat_slot_indices, const std::vector<float>& raw_priorities) = 0;
        virtual std::optional<float> GetScalar(const std::string& key) const = 0;
        virtual std::optional<torch::Tensor> GetPriorityTensor(int64_t flat_slot_index) const = 0;
        virtual torch::Tensor GatherPriorityRows(const torch::Tensor& flat_slot_indices) const = 0;
        virtual ~ReplayPriorityStore() = default;
    };

    /**
     * @brief サンプリング可能化境界でPER初期優先度を完成させる内部Module。
     *
     * ExperienceQueueがn-step/系列を確定した後のpriority未完成entryをenv別FIFOで保持し、
     * ValidIndexManagerがsampleableと判定した順に初期優先度を適用する。
     * mode、Estimator呼び出し、fallback、初期source、completion counterを所有する一方、
     * mutex、ValidIndexManager、ReplayPriorityStore内のpriority/source実体は所有しない。
     *
     * @note EnqueueとCompleteReadyは、呼び出し側がmetadata_mutex_を保持した状態で実行する。
     * @note 現在は開始itemごとにscalar priorityを1つ完成させる。MuZero等で系列内の
     *       複数stepを集約する場合は、Estimator入力とPendingEntryの表現を拡張する。
     */
    class InitialPriorityCompleter {
    public:
        struct Config {
            ReplayInitialPriorityMode mode = ReplayInitialPriorityMode::FIXED; ///< 初期優先度の決定方式
            float fixed_raw_priority = 1.0f; ///< FIXEDで適用するper_alpha変換前の優先度
            float gamma = 0.99f;             ///< 実n-step数へ累乗する割引率
            int n_step = 1;                  ///< sampleable判定に使う未来観測幅
            int unroll_steps = 0;            ///< sampleable判定に使う未来系列幅
        };

        struct PendingEntry {
            int64_t logical_time_idx = 0; ///< 開始stepのenv内単調増加index
            int64_t flat_slot_index = 0;  ///< 初期優先度を適用する物理slot
            float target_return = 0.0f;   ///< ExperienceQueue通過後の確定済みtarget return
            bool terminal = false;        ///< bootstrapを行わない真の終端か
            bool truncated = false;       ///< 終端観測を推論せずmaxへfallbackする打切りか
            int actual_n_steps = 1;       ///< 終端短縮を反映した実n-step数
            std::optional<ReplayInitialPriorityHintRow> start_hint; ///< 開始stepのopaqueなhint行
        };

        struct Stats {
            int64_t attempt_count = 0;             ///< Actor近似completionを試みた累積件数
            int64_t success_count = 0;             ///< Actor近似priorityを適用できた累積件数
            int64_t truncation_fallback_count = 0; ///< truncatedを理由にmaxへfallbackした累積件数
            int64_t nonfinite_fallback_count = 0;  ///< nonfiniteを理由にmaxへfallbackした累積件数
        };

        InitialPriorityCompleter(
            Config config, int64_t num_envs,
            std::unique_ptr<InitialPriorityEstimator> estimator);

        /// target return確定済みのreal transitionを、env別の完成待ちFIFOへ追加する。
        void Enqueue(int64_t env_idx, PendingEntry entry);

        /// sampleableになったFIFO先頭から、初期優先度とsourceを同じ同期区間で適用する。
        void CompleteReady(
            int64_t env_idx,
            const std::optional<ReplayInitialPriorityHintRow>& current_hint,
            int64_t current_logical_idx,
            const ValidIndexManager& index_manager,
            ReplayPriorityStore& priority_store);

        /// public metricsへ変換するための累積counter snapshotを返す。
        Stats GetStats() const { return stats_; }

    private:
        Config config_;
        std::unique_ptr<InitialPriorityEstimator> estimator_;
        std::vector<std::deque<PendingEntry>> pending_;
        Stats stats_;
    };

    /// productionとテストで共用し、PER時だけcompletion Moduleを構築する。
    std::unique_ptr<InitialPriorityCompleter> CreateInitialPriorityCompleter(
        const ReplayBufferConfig& config,
        int64_t num_envs,
        std::unique_ptr<InitialPriorityEstimator> estimator);


    // ======================================================
    // SumTree (汎用部品)
    // ======================================================

    template <typename Value>
    class SumTree {
        static_assert(std::is_floating_point_v<Value>, "SumTree Value must be a floating-point type.");
    public:
        explicit SumTree(int64_t capacity) : capacity_(capacity)
        {
            tree_.assign(2 * capacity_ - 1, Value{ 0 });
        }

        void Update(int64_t index, Value priority)
        {
            int64_t tree_idx = index + capacity_ - 1;
            const Value change = priority - tree_[tree_idx];
            tree_[tree_idx] = priority;
            while (tree_idx != 0) {
                tree_idx = (tree_idx - 1) / 2;
                tree_[tree_idx] += change;
            }
        }

        Value GetTotalPriority() const
        {
            return tree_[0];
        }

        int64_t Retrieve(Value value) const
        {
            int64_t tree_idx = 0;
            while (tree_idx < capacity_ - 1) {
                const int64_t left = 2 * tree_idx + 1;
                const int64_t right = left + 1;
                if (value <= tree_[left]) {
                    tree_idx = left;
                } else {
                    value -= tree_[left];
                    tree_idx = right;
                }
            }
            return tree_idx - capacity_ + 1;
        }

        Value GetPriority(int64_t index) const
        {
            return tree_[index + capacity_ - 1];
        }

        int64_t Capacity() const { return capacity_; }
    private:
        int64_t capacity_;
        std::vector<Value> tree_;
    };


    // ======================================================
    // Facade (DefaultReplayBuffer)
    // ======================================================

    class DefaultReplayBuffer final : public ReplayBuffer {
    public:
        DefaultReplayBuffer(
            const ReplayBufferConfig& config,
            const EnvSpec& env_spec,
            int64_t num_envs,
            std::unique_ptr<ExperienceQueueController> queue_controller,
            std::unique_ptr<ReplayExperienceBuilder> builder,
            std::shared_ptr<ReplayExperienceSampler> sampler,
            std::shared_ptr<ReplayPriorityStore> priority_store,
            std::shared_ptr<ExperienceSampleExtractor> extractor,
            std::unique_ptr<InitialPriorityCompleter> initial_priority_completer,
            torch::Device device,
            bool pin_memory);

        void Push(const BatchExperience& batch_exp) override;
        void Sample(ExperienceSamples& out_samples, int64_t minibatch_size, float beta) const override;
        int64_t Size() const override;
        ReplayPriorityUpdateResult UpdatePriorities(
            const std::vector<int64_t>& item_keys, const std::vector<float>& priorities) override;

        std::optional<float> GetScalar(const std::string& key, int64_t index) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index) const override;

		/// デバッグ用: Storageの内容とValid Indexをログに出力する
        void DumpToLog() const;
    private:
        void ProcessQueue(
            int64_t env_idx,
            const std::optional<ReplayInitialPriorityHintRow>& current_hint,
            int64_t current_logical_idx); // 内部パイプラインの駆動
        int64_t OnSlotWritten(int64_t flat_slot_index);
        int64_t EncodeReplayItemKey(int64_t flat_slot_index) const;
        void InvalidateAccessorCacheForStorage();
        void InvalidateAccessorCacheForPriority();
        std::optional<std::vector<torch::Tensor>> TryGetCachedTensorVector(const std::string& key, int64_t index) const;
        void StoreTensorVectorCache(const std::string& key, int64_t index, std::vector<torch::Tensor> value) const;
        void MarkSampledOnce(const torch::Tensor& indices) const;
        void RecordEvictionIfSampleable(
            int64_t env_idx,
            int64_t time_idx,
            int64_t* evicted_sampleable_count,
            int64_t* evicted_never_sampled_count);
        void StoreLastEvictionStats(int64_t evicted_sampleable_count, int64_t evicted_never_sampled_count);
    private:
        struct TensorVectorCacheEntry {
            std::string key;
            int64_t index = -1;
            uint64_t storage_version = 0;
            uint64_t priority_version = 0;
            std::vector<torch::Tensor> value;
        };
    private:
        ReplayBufferConfig config_;
        int64_t num_envs_;
        int64_t capacity_per_env_;

        std::unique_ptr<ReplayExperienceStorage> storage_;
        std::unique_ptr<ValidIndexManager> index_manager_;

        std::unique_ptr<ExperienceQueueController> queue_controller_;
        std::unique_ptr<ReplayExperienceBuilder> builder_;
        std::shared_ptr<ReplayExperienceSampler> sampler_;
        std::shared_ptr<ReplayPriorityStore> priority_store_;
        std::shared_ptr<ExperienceSampleExtractor> extractor_;
        std::unique_ptr<InitialPriorityCompleter> initial_priority_completer_;

        std::vector<ExperienceQueue> queues_;
        std::vector<int64_t> generations_;
        int64_t actual_capacity_ = 0;
        int64_t priority_update_stale_drop_count_ = 0;

        mutable std::shared_mutex storage_mutex_;
        mutable std::mutex metadata_mutex_;
        mutable std::mutex accessor_cache_mutex_;
        mutable std::vector<uint8_t> sampled_once_;
        int64_t last_evicted_sampleable_count_ = 0;
        int64_t last_evicted_never_sampled_count_ = 0;
        float last_evicted_never_sampled_ratio_ = std::numeric_limits<float>::quiet_NaN();
        uint64_t accessor_storage_version_ = 0;
        uint64_t accessor_priority_version_ = 0;
        mutable std::vector<TensorVectorCacheEntry> tensor_vector_cache_;
    };

} // namespace anet::rl
