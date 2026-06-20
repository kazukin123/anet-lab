// replay_buffer_impl.hpp

#pragma once

#include "anet/replay_buffer.hpp"
#include <algorithm>
#include <vector>
#include <memory>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <torch/torch.h>
#include "anet/tensor_util.hpp"

namespace anet::rl {


    // ======================================================
    // Queue
    // ======================================================

    /// Storage上の重いデータ（Obs等）とは分離された、N-Step計算用の軽量メタデータ
    struct QueueRecord {
        int64_t time_idx;       ///< Storage上のインデックス (Facadeが管理するためのKey)
        float reward;           ///< 即時報酬
        bool done;              ///< エピソード終了フラグ
        bool truncated;         ///< タイムアップ等による打ち切りフラグ
        bool is_dummy = false;  ///< Truncated時の終端計算用ダミーフラグ
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
    private:
        template <class Fn>
        void ForEachSampleableIndex(int64_t env, int stack_count, int unroll_steps, int n_step, Fn&& fn) const
        {
            (void)stack_count;

            int64_t w_cursor = write_cursors_[env];
            int64_t v_cursor = valid_cursors_[env];
            int64_t future_obs_lag = std::max<int64_t>(1, n_step);

            int64_t logical_start = std::max<int64_t>(0, w_cursor - capacity_per_env_);
            int64_t max_safe_by_write = std::max<int64_t>(-1, w_cursor - future_obs_lag - 1);
            int64_t max_safe_by_valid = v_cursor - 1 - unroll_steps;
            int64_t logical_end = std::min(max_safe_by_write, max_safe_by_valid);

            if (logical_end < logical_start) return;

            int64_t start_phys = logical_start % capacity_per_env_;
            int64_t end_phys = logical_end % capacity_per_env_;

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
        torch::Tensor indices;       ///< [B] 1D indices
        torch::Tensor sampling_prob; ///< [B] probabilities
        torch::Tensor is_weights;    ///< [B] importance sampling weights
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


    // ======================================================
    // SumTree (汎用部品)
    // ======================================================

    class SumTree {
    public:
        explicit SumTree(int64_t capacity);
        void Update(int64_t index, float priority);
        float GetTotalPriority() const;
        int64_t Retrieve(float value) const;
        float GetPriority(int64_t index) const;
        int64_t Capacity() const { return capacity_; }
    private:
        int64_t capacity_;
        std::vector<float> tree_;
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
            std::shared_ptr<ReplayPriorityController> prio_controller,
            std::shared_ptr<ExperienceSampleExtractor> extractor,
            torch::Device device,
            bool pin_memory);

        void Push(const BatchExperience& batch_exp) override;
        void Sample(ExperienceSamples& out_samples, int64_t minibatch_size, float beta) const override;
        int64_t Size() const override;
        void UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities) override;

        std::optional<float> GetScalar(const std::string& key, int64_t index) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index) const override;

		/// デバッグ用: Storageの内容とValid Indexをログに出力する
        void DumpToLog() const;
    private:
        void ProcessQueue(int64_t env_idx); // 内部パイプラインの駆動
        void InvalidateAccessorCacheForStorage();
        void InvalidateAccessorCacheForPriority();
        std::optional<std::vector<torch::Tensor>> TryGetCachedTensorVector(const std::string& key, int64_t index) const;
        void StoreTensorVectorCache(const std::string& key, int64_t index, std::vector<torch::Tensor> value) const;
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
        std::shared_ptr<ReplayPriorityController> prio_controller_;
        std::shared_ptr<ExperienceSampleExtractor> extractor_;

        std::vector<ExperienceQueue> queues_;

        mutable std::shared_mutex storage_mutex_;
        mutable std::mutex metadata_mutex_;
        mutable std::mutex accessor_cache_mutex_;
        uint64_t accessor_storage_version_ = 0;
        uint64_t accessor_priority_version_ = 0;
        mutable std::vector<TensorVectorCacheEntry> tensor_vector_cache_;
    };

} // namespace anet::rl
