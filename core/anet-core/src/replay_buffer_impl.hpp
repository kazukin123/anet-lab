// replay_buffer_impl.hpp
#pragma once

#include<vector>
#include<memory>
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

class ExperienceQueue {
public:
    void Push(const SingleExperience& exp);
    void Pop(size_t k);
    std::vector<SingleExperience> Peek(size_t k) const;
    size_t Size() const;
    /// @todo impl
};

using ExperienceSequence = std::vector<SingleExperience>;

class ExperienceQueueController {
public:
    virtual std::vector<ExperienceSequence> 
        ProcessSingleExperience(ExperienceQueue& queue, const SingleExperience& exp) = 0;
    virtual ~ExperienceQueueController() = default;

};

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

class ReplayExperienceStorage {
public:
    void Push(const ReplayExperience& exp);
    size_t Size() const;
    ExperienceSamples Gather(const torch::Tensor& indices) const;
    const ReplayExperience& Get(size_t index) const;

    /// @todo impl
private:
    torch::Tensor states;        // (N, state_dim)
    torch::Tensor actions;       // (N, action_dim)
    torch::Tensor target_values; // (N,)
    torch::Tensor next_states;   // (N, state_dim)
    torch::Tensor terminals;     // (N,) bool
    torch::Tensor n_steps;       // (N,) int
};

class ReplayExperienceSampler {
public:
    virtual torch::Tensor SampleIndices(size_t minibatch_size) = 0;
    virtual ~ReplayExperienceSampler() = default;
};

class DefaultReplayBuffer : public ReplayBuffer {
public:
    /// @todo impl
    void Push(const BatchExperience& batch_exp);
    void Push(const std::vector<SingleExperience>& exps);
    ExperienceSamples Sample(int64_t minibatch_size, torch::Device device) const;
    size_t Size() const;
private:    // data
    std::unique_ptr<ExperienceQueue> queue_;
    std::unique_ptr<ReplayExperienceStorage> storage_;
private:    // logic
    std::unique_ptr<ExperienceQueueController> queue_controller_;
    std::unique_ptr<ReplayExperienceBuilder> replay_exp_builder_;
};

class ReplayBufferFactory {
public:
    /// @todo impl
    std::shared_ptr<ReplayBuffer> CreateReplayBuffer();
};


