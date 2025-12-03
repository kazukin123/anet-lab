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

    class ReplayBuffer : public RandomHolder, public DataExporter {
    public:
        explicit ReplayBuffer(const EnvSpec& env_spec, size_t capacity = 10000, std::optional<seed_t> seed = std::nullopt);

        void Push(const BatchExperience& batch);
        void Push(const std::vector<Experience>& exps);
        ExperienceSample Sample(int64_t b, torch::Device device) const;
        size_t Size() const { return size_; }

        std::optional<float> GetScalar(const std::string& key) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key) const override;
    public:
        static constexpr const char* STATE_OBS = "replaybuffer.state";
        static constexpr const char* ACTION_ACTION = "replaybuffer.action";
        static constexpr const char* REWARD = "replaybuffer.reward";
        static constexpr const char* NEXT_STATE_OBS = "replaybuffer.next_state";
        static constexpr const char* NEXT_STATE_DONE = "replaybuffer.done";
        static constexpr const char* NEXT_STATE_TRUNCATED = "replaybuffer.truncated";
        static constexpr const char* NEXT_STATE_EPISODE_START = "replaybuffer.episode_start";
    private:
        void InitFromSpec(const EnvSpec& spec);

        size_t capacity_;
        int64_t state_count_;  ///< state_dim_ は StateSpec.shape の総積（flatten 後次元）
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