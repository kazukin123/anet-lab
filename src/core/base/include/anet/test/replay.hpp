#pragma once

#ifndef ANET_TEST_SUPPORT
#error "anet/test/replay.hpp requires ANET_TEST_SUPPORT"
#endif

#include "anet/replay_experience.hpp"
#include "anet/replay_buffer.hpp"

namespace anet::test {
    struct ReferenceInitialPriority {
        float raw;
        rl::ReplayPrioritySource source;
    };

    ReferenceInitialPriority CompleteReferenceInitialPriority(
        int n_step, int actual_n_steps, float target_return, float discount,
        bool terminal, bool truncated, const rl::ReplayInitialPriorityHintRow& start_hint,
        const rl::ReplayInitialPriorityHintRow& end_hint,
        std::unique_ptr<rl::InitialPriorityEstimator> estimator);
}
