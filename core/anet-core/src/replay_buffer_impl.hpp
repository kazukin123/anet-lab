// replay_buffer_impl.hpp
#pragma once

#include<vector>
#include<memory>
#include <cstddef>
#include "anet/rl.hpp"

using namespace anet::rl;


//BatchExperience
//  ↓（分解）
//SingleExperience
//  ↓ ExperienceQueueController
//ExperienceSequence
//  ↓ ReplayExperienceBuilder
//ReplayExperience
//  ↓
//ReplayExperienceStorage::Push

//ReplayExperienceStorage::Sample
//  ↓
//Sampler::SampleIndices
//  ↓
//ReplayExperienceStorage::Gather
//  ↓
//Learner


// ======================================================
// ReplayBuffer ExperienceQueue 
// ======================================================

class ExperienceQueue {
public:
    ExperienceQueue() = default;
    void Push(const SingleExperience& exp);             ///< 末尾に 1 Experience を追加
    void Pop(size_t k);                                 ///< 先頭から k 個を削除
    std::vector<SingleExperience> Peek(size_t k) const; ///< 先頭から k 個を取得（コピー）
    size_t Size() const { return buffer_.size(); }      ///< 現在の保持数
    void Clear() { buffer_.clear(); }                             ///< 空にする
private:
    std::vector<SingleExperience> buffer_;
};


// ======================================================
// ReplayBuffer ExperienceQueueController 
// ======================================================

using ExperienceSequence = std::vector<SingleExperience>;

class ExperienceQueueController {
public:
    virtual std::vector<ExperienceSequence> 
        ProcessSingleExperience(ExperienceQueue& queue, const SingleExperience& exp) = 0;
    virtual ~ExperienceQueueController() = default;

};

class PlainExperienceQueueController final : public ExperienceQueueController {
public:
    PlainExperienceQueueController() = default;

    std::vector<ExperienceSequence>
        ProcessSingleExperience(ExperienceQueue& queue, const SingleExperience& exp) override;
};

class NStepExperienceQueueController final : public ExperienceQueueController {
public:
    explicit NStepExperienceQueueController(size_t n_step);

    std::vector<ExperienceSequence>
        ProcessSingleExperience(ExperienceQueue& queue, const SingleExperience& exp) override;
private:
    const size_t n_step_;
};


// ======================================================
// ReplayBuffer ReplayExperienceBuilder
// ======================================================

struct ReplayExperience {
    SingleState state;
    torch::Tensor action;
    float target_value;      // r or G
    SingleState next_state;
    bool terminal;
    int n_step;           // この experience が何 step 分か
};

class ReplayExperienceBuilder {
public:
    virtual ReplayExperience Build(const ExperienceSequence& sequence) const = 0;;
    virtual ~ReplayExperienceBuilder() = default;
};

class PlainReplayExperienceBuilder final : public ReplayExperienceBuilder {
public:
    PlainReplayExperienceBuilder() = default;
    ReplayExperience Build(const ExperienceSequence& sequence) const override;
};

class NStepReplayExperienceBuilder final : public ReplayExperienceBuilder {
public:
    explicit NStepReplayExperienceBuilder(float gamma);
    ReplayExperience Build(const ExperienceSequence& sequence) const override;
private:
    const float gamma_;
};


// ======================================================
// ReplayBuffer ReplayExperienceStorage
// ======================================================

class ReplayExperienceStorage : public anet::DataExporter {
public:
    ReplayExperienceStorage(const EnvSpec& env_spec, int64_t capacity, torch::Device device);
    void Push(const ReplayExperience& exp);
    int64_t Size() const { return size_; }
    ExperienceSamples Gather(const std::vector<int64_t>& indices,std::optional<torch::Device> out_device = std::nullopt) const;
public: //---- DataExporter ----
    std::optional<float> GetScalar(const std::string& key, int index) const override;
    std::optional<torch::Tensor> GetTensor(const std::string& key, int index) const override;
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int index) const override;
private:
    const int64_t capacity_;
    const torch::Device device_;
    const torch::TensorOptions int64_opt_;
private:
    int64_t size_ = 0;
    int64_t write_index_ = 0;
    torch::Tensor states_;        // (N, state_dim)
    torch::Tensor actions_;       // (N, action_dim)
    torch::Tensor target_values_; // (N,)
    torch::Tensor next_states_;   // (N, state_dim)
    torch::Tensor terminals_;     // (N,) bool
    torch::Tensor n_steps_;       // (N,) int
};


// ======================================================
// ReplayBuffer ReplayExperienceSampler
// ======================================================

class ReplayExperienceSampler {
public:
    virtual std::vector<int64_t> SampleIndices(const ReplayExperienceStorage& storage, int64_t minibatch_size) = 0;
    virtual ~ReplayExperienceSampler() = default;
};

class UniformReplayExperienceSampler final : public ReplayExperienceSampler, public anet::RandomHolder {
public:
    explicit UniformReplayExperienceSampler(anet::seed_t seed);
    std::vector<int64_t> SampleIndices(const ReplayExperienceStorage& storage, int64_t minibatch_size) override;
private:
    const torch::TensorOptions opts_;
};


// ======================================================
// DefaultReplayBuffer
// ======================================================

class DefaultReplayBuffer : public ReplayBuffer {
public:
    DefaultReplayBuffer(
        const EnvSpec& env_spec, int64_t capacity, int64_t num_envs,
        std::unique_ptr<ExperienceQueueController> queue_controller,
        std::unique_ptr<ReplayExperienceBuilder> replay_exp_builder,
        std::unique_ptr<ReplayExperienceSampler> sampler,
        torch::Device device, bool use_prefetch = false);

    void Push(const BatchExperience& batch_exp) override;
    void Push(const std::vector<SingleExperience>& exps) override;
    ExperienceSamples Sample(int64_t minibatch_size, torch::Device device) const override;
    int64_t Size() const override;
public: // DataExporter
    std::optional<float> GetScalar(const std::string& key, int index) const override;
    std::optional<torch::Tensor> GetTensor(const std::string& key, int index) const override;
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int index) const override;
private:
    ExperienceSamples sampleInternal(int64_t minibatch_size, torch::Device device) const;
private:
    // 設定
    bool use_prefetch_;

    // N 環境分
    const int64_t num_envs_;
    std::vector<ExperienceQueue> queues_;

    // 共通
    std::unique_ptr<ExperienceQueueController> queue_controller_;
    std::unique_ptr<ReplayExperienceBuilder> replay_exp_builder_;
    std::unique_ptr<ReplayExperienceSampler> sampler_;
    std::unique_ptr<ReplayExperienceStorage> storage_;

    // Prefech
    mutable bool prefetch_cached_ = false;
    mutable ExperienceSamples prefetch_result_;
};

