//anet/replay_buffer.hpp

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <random>
#include <torch/torch.h>
#include "anet/random.hpp"
#include "anet/rl.hpp"

namespace anet::rl {


    // ======================================================
    // Config & Data Structures
    // ======================================================

    enum class ReplaySamplerType {
        UNIFORM = 0,
        PRIORITIZED   // PER
    };

    struct ReplayBufferConfig {

        int64_t capacity = 100000;              // ENV毎の容量ではなく、全ENVの総容量(1Dツリーサイズ)
        ReplaySamplerType sampler_type = ReplaySamplerType::UNIFORM;

        // N-step 用
        int n_step = 1;
        float gamma = 0.99f;

        // PER系
        float per_alpha = 0.5f;
        float per_initial_priority = 1.0f;

        // Stacking
        int stack_count = 1;                    ///< 過去方向へのスライス数 (Frame Stacking)
		std::vector<std::string> stack_keys;    ///< Stacking対象のDictキー。空の場合は全てのキーをスタッキングする

		// MuZero系
        struct {
            int unroll_steps = 0;                   // 未来方向へのスライス数 (MuZero Unroll 等)
        } muzero;   /// @todo MuZeroのキーワードは排除して「未来方向」で一般化？
    };


    // ======================================================
    // Factory
    // ======================================================

    std::shared_ptr<ReplayBuffer> CreateReplayBuffer(
        const ReplayBufferConfig& config,
        const EnvSpec& env_spec,
        int64_t num_envs,
        torch::Device storage_device,
        bool pin_memory = true,
        std::optional<uint64_t> seed = std::nullopt
    );

    /// ReplayBuffer を 1-deep で先読みする decorator。
    ///
    /// `Sample()` は cold start では同期 `Fetch()` を返し、その同じ排他区間で次の
    /// background `Fetch()` を起動する。以後は前回 future を `get()` で消費してから
    /// 次 future を起動するため、同時に in-flight になる prefetch は最大 1 本。
    ///
    /// `Push()` と `UpdatePriorities()` は in-flight future があれば完了まで `wait()` して
    /// から inner へ mutation を委譲する。この順序保証は、storage / priority を変更する
    /// 経路がこの wrapper を必ず通ることを前提にする。`wait()` は future を消費しないため、
    /// prefetch worker の例外は次回 `Sample()` が future を `get()` するときに表面化する。
    ///
    /// CPU path は background `Sample()+To(CPU)` のみを行う。CUDA path は CPU sample を
    /// pinned 化し、copy stream の non-blocking H2D と CUDAEvent 待ちで転送完了を保証する。
    class PrefetchingReplayBuffer final : public ReplayBuffer {
    public:
        PrefetchingReplayBuffer(std::shared_ptr<ReplayBuffer> inner, torch::Device target_device);
        ~PrefetchingReplayBuffer() override;

        void Push(const BatchExperience& batch_exp) override;
        void Sample(ExperienceSamples& out_samples, int64_t minibatch_size, float beta) const override;
        int64_t Size() const override;
        void UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities) override;

        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override;
    private:
        struct State;
        struct PrefetchedBatch;

        PrefetchedBatch Fetch(int64_t minibatch_size, float beta) const;
        PrefetchedBatch TransferSamples(ExperienceSamples cpu_samples) const;
        void WaitForPrefetchLocked() const;
        void LaunchPrefetchLocked(int64_t minibatch_size, float beta) const;
        void StopPrefetch() const;
    private:
        std::shared_ptr<ReplayBuffer> inner_;
        torch::Device target_device_;
        mutable std::unique_ptr<State> state_;
    };

} // namespace anet
