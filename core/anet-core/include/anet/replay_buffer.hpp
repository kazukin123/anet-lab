#pragma once
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

        // Stackeing
        int stack_count = 1;                    ///< 過去方向へのスライス数 (Frame Stacking)
		std::vector<std::string> stack_keys;    ///< Stacking対象のDictキー。空の場合は全てのキーをスタッキングする

		// MuZero系
        struct {
            int unroll_steps = 0;                   // 未来方向へのスライス数 (MuZero Unroll 等)
        } muzero;
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

} // namespace anet
