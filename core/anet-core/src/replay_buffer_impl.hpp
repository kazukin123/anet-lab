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
//ReplayExperienceStacker
//  ↓
//ReplayExperienceStorage::Gather/Get
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

enum class StorageWriteEventCode {
    Append,
    Overwrite
};

struct StorageWriteEvent {
    StorageWriteEventCode code;
    int64_t index;
};

class ReplayExperienceStorage : public anet::Module {
public:
    using EventHandler = std::function<void(const StorageWriteEvent&)>;
public:
    ReplayExperienceStorage(const EnvSpec& env_spec, int64_t capacity, torch::Device device);
    void Push(const ReplayExperience& exp);
    ExperienceSamples Gather(const std::vector<int64_t>& indices, std::optional<torch::Device> out_device = std::nullopt) const;

    int64_t GetWriteIndex() const { return write_index_; }
    int64_t GetSize() const { return size_; }
    int64_t GetCapacity() const { return capacity_; }

    const torch::Tensor& GetStates() const { return states_; }
    const torch::Tensor& GetActions() const { return actions_; }
    const torch::Tensor& GetTargetValues() const { return target_values_; }
    const torch::Tensor& GetNextStates() const { return next_states_; }
    const torch::Tensor& GetTerminals() const { return terminals_; }
    const torch::Tensor& GetNSteps() const { return n_steps_; }
    const torch::Tensor& GetEpisodeStarts() const { return episode_starts_; }
public:
    void AttachEventHandler(EventHandler handler);
public:
    std::optional<float> GetScalar(const std::string& key, int64_t index) const override;
    std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index) const override;
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index) const override;
private:
    void Notify(const StorageWriteEvent& ev);
private:
    const int64_t capacity_;
    const torch::Device device_;
    const torch::TensorOptions int64_opt_;
    std::vector<EventHandler> event_handlers_;
private:
    int64_t size_ = 0;
    int64_t write_index_ = 0;
    torch::Tensor states_;        // (N, state_dim)
    torch::Tensor actions_;       // (N, action_dim)
    torch::Tensor target_values_; // (N,)
    torch::Tensor next_states_;   // (N, state_dim)
    torch::Tensor terminals_;     // (N,) bool
    torch::Tensor n_steps_;       // (N,) int
    torch::Tensor episode_starts_;// (N,) bool

    /// @todo state → obs
};


// ======================================================
// ReplayBuffer ReplayExperienceSampler
// ======================================================

struct IndexSampleResult {
    explicit IndexSampleResult(int64_t size)
        : indices(size), sampling_prob(size), is_weights(size) { }

    IndexSampleResult(
        const std::vector<int64_t>& indices_in, const std::vector<float>& sampling_prob_in, const std::vector<float>& is_weights_in)
        : indices(indices_in), sampling_prob(sampling_prob_in), is_weights(is_weights_in) { }

    IndexSampleResult(const std::vector<int64_t>& indices_in)
        : indices(indices_in) { }

    std::vector<int64_t> indices;
    std::vector<float> sampling_prob;
    std::vector<float> is_weights;
};

class ReplayExperienceSampler {
public:
    virtual IndexSampleResult SampleIndices(const ReplayExperienceStorage& storage, int64_t minibatch_size, float beta = -1) = 0;
    virtual ~ReplayExperienceSampler() = default;
};

class UniformReplayExperienceSampler final : public ReplayExperienceSampler, public anet::RandomHolder {
public:
    explicit UniformReplayExperienceSampler(anet::seed_t seed);
    IndexSampleResult SampleIndices(const ReplayExperienceStorage& storage, int64_t minibatch_size, float beta) override;
private:
    const torch::TensorOptions opts_;
};


// ======================================================
// SumTree
// ======================================================

/// @note This implementation is NOT thread-safe.
class SumTree {
public:
    explicit SumTree(int64_t capacity);

    void Update(int64_t index, float value);
    float Get(int64_t index) const;
    int64_t Sample(float value) const;
    int64_t Capacity() const noexcept { return capacity_; }
    float Total() const noexcept { return tree_[1]; }
private:
    int64_t capacity_;
    std::vector<float> tree_; // size = 2 * capacity_
};


// ======================================================
// ReplayPriorityController
// ======================================================

class PrioritizedReplayExperienceManager final
    : public ReplayExperienceSampler, public ReplayPriorityController, public anet::RandomHolder {
public:
    PrioritizedReplayExperienceManager(int64_t capacity, float alpha, anet::seed_t seed);

    IndexSampleResult SampleIndices(const ReplayExperienceStorage& storage, int64_t minibatch_size, float beta) override;
    void UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities) override;
public:
    std::optional<float> GetScalar(const std::string& key, int64_t index) const override;
    std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index) const override;
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index) const override;
private:
    SumTree sum_tree_;
    float alpha_;
};


// ======================================================
// ReplayExperienceStacker
// ======================================================

class ReplayExperienceStacker {
public:
    virtual ExperienceSamples SampleBatch(
        const ReplayExperienceStorage& storage, const IndexSampleResult& index_result, int64_t minibatch_size, torch::Device target_device) = 0;
    virtual ~ReplayExperienceStacker() = default;
};

class ReplayExperienceStateStacker final : public ReplayExperienceStacker {
public:
    ReplayExperienceStateStacker(int stack_count, int num_envs, const std::vector<int64_t>& state_shape, torch::Device device);

    ExperienceSamples SampleBatch(
        const ReplayExperienceStorage& storage, const IndexSampleResult& index_result, int64_t minibatch_size, torch::Device target_device) override;
private:
    const torch::Device device_;
    torch::TensorOptions stacked_indices_opts_;
    const int stack_count_;
    const int num_envs_;
    std::vector<int64_t> state_shape_;
};


// ======================================================
// DefaultReplayBuffer
// ======================================================

class DefaultReplayBuffer final : public ReplayBuffer {
public:
    DefaultReplayBuffer(
        const EnvSpec& env_spec, int64_t capacity, int64_t num_envs,
        std::unique_ptr<ExperienceQueueController> queue_controller,
        std::unique_ptr<ReplayExperienceBuilder> replay_exp_builder,
        std::shared_ptr<ReplayExperienceSampler> sampler,
        std::shared_ptr<ReplayPriorityController> prio_controller,
        std::shared_ptr<ReplayExperienceStacker> stacker,
        torch::Device device, float initial_priority = 1.0f, bool use_prefetch = false);

    void Push(const BatchExperience& batch_exp) override;
    void Push(const std::vector<SingleExperience>& exps) override;
    ExperienceSamples Sample(int64_t minibatch_size, torch::Device device, float beta) const override;
    int64_t Size() const override;

    void UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities) override;
public: // DataExporter
    std::optional<float> GetScalar(const std::string& key, int64_t index) const override;
    std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index) const override;
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index) const override;
private:
    ExperienceSamples sampleInternal(int64_t minibatch_size, torch::Device device, float beta) const;
private:
    // 設定
    bool use_prefetch_;
    float initial_priority_;

    // N 環境分
    const int64_t num_envs_;
    std::vector<ExperienceQueue> queues_;

    // 共通
    std::unique_ptr<ExperienceQueueController> queue_controller_;
    std::unique_ptr<ReplayExperienceBuilder> replay_exp_builder_;
    std::unique_ptr<ReplayExperienceStorage> storage_;
    std::shared_ptr<ReplayExperienceSampler> sampler_;
    std::shared_ptr<ReplayPriorityController> prio_controller_;
	std::shared_ptr<ReplayExperienceStacker> stacker_;

    // Prefech
    mutable bool prefetch_cached_ = false;
    mutable ExperienceSamples prefetch_result_;
};

