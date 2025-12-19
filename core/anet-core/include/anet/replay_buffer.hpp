#pragma once
#include <vector>
#include <random>
#include <torch/torch.h>
#include "anet/random.hpp"
#include "anet/rl.hpp"

namespace anet::rl {


    class PlainReplayBuffer final : public ReplayBuffer, public RandomHolder {
    public:
        explicit PlainReplayBuffer(const EnvSpec& env_spec, size_t capacity = 10000, std::optional<seed_t> seed = std::nullopt);

        void Push(const BatchExperience& batch) override;
        void Push(const std::vector<SingleExperience>& exps) override;
        ExperienceSamples Sample(int64_t b, torch::Device device, float beta) const override;
        int64_t Size() const  override { return size_; }
        void UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities) { }

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
        torch::Tensor terminals_;       ///< cpu (capacity) kBool
    };

    // ======================================================

    enum class ReplayBuilderType {
        PLAIN = 0,  ///< 1-step, uniform
        NSTEP,      ///< N-step, uniform
    };

    enum class ReplaySamplerType {
        UNIFORM = 0,
        PRIOTIZED   ///<! PER
    };

    struct ReplayBufferConfig {
        // 基本設定
        ReplayBuilderType type = ReplayBuilderType::PLAIN;
        ReplaySamplerType sampler_type = ReplaySamplerType::UNIFORM;

        int64_t capacity = 10000;   // 10K

        // N-step 用
        int n_step = 1;
        float gamma = 0.99f;

        // PER系
        float per_alpha = 0.5f;
        float per_initial_priority = 1.0f;
    };

    class ReplayBufferFactory {
    public:
        ReplayBufferFactory(const ReplayBufferConfig& config);
        std::shared_ptr<ReplayBuffer> Create(const EnvSpec& env_spec, torch::Device device, int batch_size, seed_t seed);
    private:
        ReplayBufferConfig config_;
    };

} // namespace anet
