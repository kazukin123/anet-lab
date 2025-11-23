#pragma once
#include "anet/rl.hpp"  // PostUpdateObserver, Experience, ActionInfo, UpdateResult

namespace anet::rl {

    class MetricsLogObserver : public anet::rl::PostUpdateObserver {
    public:
        MetricsLogObserver() = default;
        virtual ~MetricsLogObserver() = default;

        void OnPostUpdate(
            std::shared_ptr<const anet::rl::BatchUpdateResult> result,
            const anet::rl::BatchExperience& experiences,
            size_t step
        ) override;
    };

    class HeatMapObserver : public anet::rl::PostUpdateObserver {
    public:
        HeatMapObserver() = default;
        virtual ~HeatMapObserver() = default;

        void OnPostUpdate(
            std::shared_ptr<const anet::rl::BatchUpdateResult> result,
            const anet::rl::BatchExperience& experiences,
            size_t step) override;
    };
}

