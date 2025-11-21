
#include "anet/observers.hpp"
#include "anet/metrics_logger.hpp"

namespace anet::rl {

    void MetricsLogObserver::OnPostUpdate(
        std::shared_ptr<const anet::rl::UpdateResult> result,
        const anet::rl::BatchExperience& experiences,
        size_t step
    )
    {
        /// @todo Metrics 出力処理を実装する
        /// @todo result->GetMetricsMap() の値をログ・保存・GUI に反映する
        /// @todo 必要であれば時系列バッファへ保存する

        auto map = result->GetMetricsMap();
        for (const auto& [tag, value] : map) {
            MetricsLogger::Instance()->LogScalar(tag, step, value);
        }
    }

    void HeatMapObserver::OnPostUpdate(
        std::shared_ptr<const anet::rl::UpdateResult> result,
        const anet::rl::BatchExperience& experiences,
        size_t step
    )
    {
        /// @todo HeatMap / Histogram の更新処理を実装する
    }

}