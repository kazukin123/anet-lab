#pragma once
#include <vector>
#include <random>
#include <torch/torch.h>
#include "anet/random.hpp"
#include "anet/rl.hpp"

namespace anet::rl {


    class PlainReplayBuffer : public ReplayBuffer, public RandomHolder {
    public:
        explicit PlainReplayBuffer(const EnvSpec& env_spec, size_t capacity = 10000, std::optional<seed_t> seed = std::nullopt);

        void Push(const BatchExperience& batch) override;
        void Push(const std::vector<Experience>& exps) override;
        ExperienceSample Sample(int64_t b, torch::Device device) const override;
        size_t Size() const  override { return size_; }

        std::optional<float> GetScalar(const std::string& key, int index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int index = -1) const override;
    private:
        void InitFromSpec(const EnvSpec& spec);

        size_t capacity_;
        int64_t state_dim_;  ///< state_dim_ は StateSpec.shape の総積（flatten 後次元）
        int64_t n_actions_;

        int64_t index_ = 0;
        size_t size_ = 0;
        int64_t write_index_ = 0;
        bool is_discrete_ = true;
        torch::Device device_ = torch::kCPU;

        torch::Tensor states_;          ///< cpu (capacity, state_count) kFloat32
        torch::Tensor actions_;         ///< cpu (capacity, n_actions_) kInt64 or kFloat32
        torch::Tensor next_states_;     ///< cpu (capacity, state_count) kFloat32
        torch::Tensor rewards_;         ///< cpu (capacity) kFloat32
        torch::Tensor dones_;           ///< cpu (capacity) kBool
        torch::Tensor truncateds_;      ///< cpu (capacity) kBool
        torch::Tensor episode_start_;   ///< cpu (capacity) kBool
    };

} // namespace anet