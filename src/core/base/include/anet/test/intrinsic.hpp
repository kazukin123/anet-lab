#pragma once

#ifndef ANET_TEST_SUPPORT
#error "anet/test/intrinsic.hpp requires ANET_TEST_SUPPORT"
#endif

#include "anet/config.hpp"
#include "anet/nn.hpp"

namespace anet::test {
    struct RndConfig {
        struct ErrorNorm {
            std::string average_mode;
            int64_t ema_steps;
            bool use_centering;
        };

        bool use_clipping;
        std::string statistics;
        float multiplier_max;
        int64_t error_warmup_steps;
        int64_t predictor_warmup_steps;
        ErrorNorm standardize;
        float observation_clip_range;
        nn::WeightInitConfig head_init;
        double adam_eps;
    };

    RndConfig ReadRndConfig(const ConfigData& data, const std::string& prefix);
}
