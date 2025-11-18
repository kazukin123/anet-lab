#pragma once
#include "anet/rl.hpp"  // PostUpdateObserver, Experience, ActionInfo, UpdateResult

namespace anet::rl {

    class MetricsLogObserver : public anet::rl::PostUpdateObserver {
    public:
        MetricsLogObserver() = default;
        virtual ~MetricsLogObserver() = default;

        void OnPostUpdate(
            std::shared_ptr<const anet::rl::UpdateResult> result,
            const anet::rl::Experience& experience, const anet::rl::ActionInfo& action_info,
            size_t step
        ) override;
    };

    class HeatMapObserver : public anet::rl::PostUpdateObserver {
    public:
        HeatMapObserver() = default;
        virtual ~HeatMapObserver() = default;

        void OnPostUpdate(
            std::shared_ptr<const anet::rl::UpdateResult> result,
            const anet::rl::Experience& experience, const anet::rl::ActionInfo& action_info,
            size_t step) override;
    };
}

