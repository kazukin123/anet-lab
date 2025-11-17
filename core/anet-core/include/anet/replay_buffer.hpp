#pragma once
#include <vector>
#include <random>
#include <torch/torch.h>
#include "anet/rl.hpp"
#include "anet/random.hpp"

namespace anet::rl {

    struct ExperienceSample {
        torch::Tensor states;       // (B, state_dim...)
        torch::Tensor actions;      // (B, action_dim...) or (B,)
        torch::Tensor next_states;  // (B, state_dim...)
        torch::Tensor rewards;      // (B,)
        torch::Tensor dones;        // (B,)
        torch::Tensor truncateds;   // (B,)
    };

    class ReplayBuffer {
    public:
        explicit ReplayBuffer(size_t capacity = 10000)
            : capacity_(capacity), rng_(&RandomGenerator::Default()) { }

        // RNG 注入（null の場合は Default を使用）
        void SetRandomGenerator(RandomGenerator* rng) {
            rng_ = (rng ? rng : &RandomGenerator::Default());
        }

        void Push(const Experience& e);
        anet::rl::ExperienceSample Sample(size_t n, torch::Device) const;

        size_t Size() const { return size_; }

    private:
        size_t capacity_;
        size_t size_ = 0;
        size_t write_index_ = 0;

        bool initialized_ = false;
        bool is_discrete_ = true;
        torch::Device device_ = torch::kCPU;

        torch::Tensor states_;
        torch::Tensor actions_;
        torch::Tensor next_states_;
        torch::Tensor rewards_;
        torch::Tensor dones_;
        torch::Tensor truncateds_;

        mutable RandomGenerator* rng_ = nullptr;
    };

} // namespace anet