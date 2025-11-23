#pragma once
#include <vector>
#include <random>
#include <torch/torch.h>
#include "anet/rl.hpp"
#include "anet/random.hpp"

namespace anet::rl {

    /// ReplayBatchから取り出したB個のサンプルデータ（「N環境」ではなく「Bサンプル」である事に注意）
    struct ExperienceSample {
        torch::Tensor obs;          // (B, state_dim...)
        torch::Tensor actions;      // (B, action_dim...)
        torch::Tensor rewards;      // (B,)
        struct {
            torch::Tensor obs;            // (B, state_dim...)
            torch::Tensor dones;          // (B,)
            torch::Tensor truncateds;     // (B,)
            torch::Tensor episode_start;  // (B,)
        } next_states;

        ExperienceSample Flatten() const;
        std::string ToString() const;
    };

    class ReplayBuffer : public RandomHolder {
    public:
        explicit ReplayBuffer(const EnvSpec& env_spec, size_t capacity = 10000, std::shared_ptr<anet::RandomGenerator> rnd = nullptr);

        void Push(const BatchExperience& batch);
        void Push(const std::vector<Experience>& exps);
        ExperienceSample Sample(int64_t b, torch::Device device) const;
        size_t Size() const { return size_; }
    private:
        void InitFromSpec(const EnvSpec& spec);

        size_t capacity_;
        int64_t state_dim_;  ///< state_dim_ は StateSpec.shape の総積（flatten 後次元）
        int64_t action_dim_;

        int64_t index_ = 0;
        size_t size_ = 0;
        int64_t write_index_ = 0;
        bool is_discrete_ = true;
        torch::Device device_ = torch::kCPU;

        torch::Tensor states_;
        torch::Tensor actions_;
        torch::Tensor next_states_;
        torch::Tensor rewards_;
        torch::Tensor dones_;
        torch::Tensor truncateds_;
        torch::Tensor episode_start_;
    };

} // namespace anet