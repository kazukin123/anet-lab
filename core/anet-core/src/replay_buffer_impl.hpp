// replay_buffer_impl.hpp

#pragma once

#include "anet/replay_buffer.hpp"
#include <vector>
#include <memory>
#include <deque>
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
        void MarkValid(int64_t env_idx, int64_t time_idx);

        /// Stack/Unroll 制約を考慮し、安全に引ける 1D インデックスのリストを返す
        torch::Tensor GetValidIndices1D(int stack_count, int unroll_steps) const;

        int64_t GetValidCount() const;
    private:
        int64_t num_envs_;
        int64_t capacity_per_env_;
        std::vector<int64_t> valid_cursors_;
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
    public:
        // 読み取りインターフェース (Extractor用) 
        const anet::TensorDict& GetObs() const { return obs_storage_; }
        const anet::TensorDict& GetInfo() const { return info_storage_; }
        const torch::Tensor& GetActions() const { return actions_; }
        const torch::Tensor& GetTargetReturns() const { return target_returns_; }
        const torch::Tensor& GetTerminals() const { return terminals_; }
        const torch::Tensor& GetTruncates() const { return truncates_; }
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
        torch::Tensor truncates_;
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
        float TotalPriority() const;
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
        //void UpdatePriorities(const torch::Tensor& indices, const torch::Tensor& priorities) override;

        std::optional<float> GetScalar(const std::string& key, int64_t index) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index) const override;

    private:
        void ProcessQueue(int64_t env_idx); // 内部パイプラインの駆動

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
    };

} // namespace anet::rl
