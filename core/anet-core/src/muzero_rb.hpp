// muzero_rb.hpp
#pragma once

#include <deque>
#include <vector>
#include <memory>
#include <torch/torch.h>
#include "anet/rl.hpp"
#include "anet/random.hpp"
#include "anet/muzero_proto_agent.hpp"


namespace anet::rl::muzero_proto {


    // ======================================================
    // Config & Data Structures
    // ======================================================

    /// Learnerに渡すためのMuZero専用バッチデータ
    struct MuZeroExperienceSamples {
        torch::Tensor obs;             ///< (B, state_dim...) 起点の観測
        torch::Tensor actions;         ///< (B, K) 未来Kステップの行動
        torch::Tensor target_rewards;  ///< (B, K) 未来Kステップの即時報酬
        torch::Tensor target_values;   ///< (B, K+1) 起点+未来Kステップの価値(N-Step計算済)
        torch::Tensor target_policies; ///< (B, K+1, num_actions) 起点+未来Kステップの方策
        torch::Tensor masks;           ///< (B, K+1) エピソード終端を超えた場合のパディングマスク(1=有効, 0=無効)

        MuZeroExperienceSamples To(torch::Device device) const;
    };

    /// Actorから受け取った直後の1ステップの生データ
    struct MuZeroStepRecord {
        torch::Tensor obs;
        int64_t action;
        float reward;
        bool done;
        bool truncated;
        torch::Tensor target_policy;
        float root_value;
    };

    /// N-Step計算が完了し、Storageに保存される確定レコード
    struct MuZeroReplayRecord {
        torch::Tensor obs;
        int64_t action;
        float reward;       ///< 即時報酬
        bool done;
        torch::Tensor target_policy;
        float target_value; ///< N-Step割引報酬和で計算された目標価値
    };


    // ======================================================
    // Internal Components (Queue, Builder, Storage, Sampler)
    // ======================================================

    class MuZeroExperienceQueue {
    public:
        void Push(const MuZeroStepRecord& record);
        void Pop();
        void Clear();
        const MuZeroStepRecord& Peek(size_t index) const;
        size_t Size() const { return buffer_.size(); }
    private:
        std::deque<MuZeroStepRecord> buffer_;
    };

    class MuZeroReplayBuilder {
    public:
        MuZeroReplayBuilder(int n_step, float gamma);

        /// Queueの状態を見て、可能なら確定済みRecordを生成する
        std::vector<MuZeroReplayRecord> ProcessQueue(MuZeroExperienceQueue& queue);
    private:
        const int n_step_;
        const float gamma_;
    };

    class MuZeroReplayStorage {
    public:
        MuZeroReplayStorage(int64_t capacity, int64_t num_envs);

        void Push(int64_t env_index, const MuZeroReplayRecord& record);

        /// サンプラーが利用するための情報アクセス
        int64_t GetNumEnvs() const { return num_envs_; }
        int64_t GetEnvSize(int64_t env_index) const { return env_buffers_[env_index].size(); }
        const MuZeroReplayRecord& GetRecord(int64_t env_index, int64_t t) const;

        int64_t GetTotalSize() const { return total_size_; }

    private:
        const int64_t capacity_per_env_;
        const int64_t num_envs_;
        int64_t total_size_ = 0;

        // 環境(Env)ごとに独立したリングバッファ構造（std::dequeで代用）
        std::vector<std::deque<MuZeroReplayRecord>> env_buffers_;
    };

    class MuZeroUniformSampler : public anet::RandomHolder {
    public:
        MuZeroUniformSampler(int unroll_steps, std::optional<anet::seed_t> seed = std::nullopt);

        MuZeroExperienceSamples Sample(const MuZeroReplayStorage& storage, int64_t batch_size);
    private:
        const int unroll_steps_;
    };


    // ======================================================
    // MuZeroReplayBuffer (Main Facade)
    // ======================================================

    /// @brief 試作版専用の独立したReplayBuffer実装
    class MuZeroReplayBuffer : public anet::RandomHolder {  // public anet::rl::ReplayBuffer, 
    public:
        MuZeroReplayBuffer(
            const ReplayBufferConfig& config, int64_t num_envs, int64_t num_actions, std::optional<anet::seed_t> seed = std::nullopt);

        // --- anet::rl::ReplayBuffer Interfaces ---
        void Push(const anet::rl::BatchExperience& batch_exp);
        //void Push(const std::vector<anet::rl::SingleExperience>& exps) { /* Not Used */ }
        int64_t Size() const;
        //void UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities) {}

        /// @note 標準のSampleは使用不可。MuZero専用の SampleMuZero を呼び出すこと。
        //anet::rl::ExperienceSamples Sample(int64_t minibatch_size, torch::Device device, float beta = -1) const ;

        /// brief Learner向けにKステップ展開された時系列バッチをサンプリングする
        MuZeroExperienceSamples Sample(int64_t batch_size, torch::Device device) const;
    public:
        // dummy overrides
        //std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override { return std::nullopt; }
        //std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override { return std::nullopt; }
        //std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override { return std::nullopt; }
    private:
        const int64_t num_envs_;
        const int64_t num_actions_;

        std::vector<MuZeroExperienceQueue> queues_; // EnvごとのQueue
        std::unique_ptr<MuZeroReplayBuilder> builder_;
        std::unique_ptr<MuZeroReplayStorage> storage_;
        std::unique_ptr<MuZeroUniformSampler> sampler_;
    };

} // namespace anet::rl::muzero_proto
