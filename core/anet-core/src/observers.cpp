
#include "anet/observers.hpp"
#include "anet/metrics_logger.hpp"

namespace anet::rl {

    void MetricsLogObserver::OnPostUpdate(
        int step,
        const anet::rl::BatchExperience& experiences,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result
    )
    {
        auto map = result->GetMetricsMap();
        for (const auto& [tag, value] : map) {
            MetricsLogger::Instance()->LogScalar(tag, step, value);
        }
    }

    static float ResolveMin(
        bool override_flag, float override_v, std::optional<float> probe_v, float fallback)
    {
        if (override_flag) return override_v;
        if (probe_v.has_value()) return *probe_v;
        return fallback;
    }

    static float ResolveMax(
        bool override_flag, float override_v, std::optional<float> probe_v, float fallback)
    {
        if (override_flag) return override_v;
        if (probe_v.has_value()) return *probe_v;
        return fallback;
    }

    HeatMapObserver::HeatMapObserver(
        const std::string& tag,
        const HeatMapObserverConfig& config,
        IFloatProbe* x_probe, IFloatProbe* y_probe, IFloatProbe* value_probe)
        : config_(config), tag_(tag)
    {
        x_probe_.reset(x_probe);
        y_probe_.reset(y_probe);
        value_probe_.reset(value_probe);

        // Probe の min/max と config の override から HeatMap 範囲決定
        float xmin = ResolveMin(config_.override_xmin, config_.xmin, x_probe_->GetMin(), 0.0f);
        float xmax = ResolveMax(config_.override_xmax, config_.xmax, x_probe_->GetMax(), 1.0f);
        float ymin = ResolveMin(config_.override_ymin, config_.ymin, y_probe_->GetMin(), 0.0f);
        float ymax = ResolveMax(config_.override_ymax, config_.ymax, y_probe_->GetMax(), 1.0f);

        //HeatMap(int width, int height, float x_min = 0.0f, float x_max = 1.0f,
        //    float y_min = 0.0f, float y_max = 1.0f, size_t max_points = 0,
        //    uint32_t flags = HM_Default);

        heatmap_ = std::make_unique<anet::HeatMap>(
            config_.width,
            config_.height,
            xmin, xmax, ymin, ymax,
            config_.max_points,
            config_.flags
        );
    }

    void HeatMapObserver::OnPostUpdate(
        int step,
        const anet::rl::BatchExperience& batch_experience,
        std::shared_ptr<const anet::rl::BatchUpdateResult> result)
    {
        // N 環境ぶんループ（Probe が Update を受けてから TryGetFloat）
        //const int N = experience.state.size(0);

        auto exp_list = batch_experience.ToExperienceList();
        for (auto exp : exp_list) {
            // 生成： xv, yv, vv
            auto xv = x_probe_->GetFloat(step, exp, result);
            auto yv = y_probe_->GetFloat(step, exp, result);
            auto vv = value_probe_->GetFloat(step, exp, result);

            // 値が揃ってなかったらスキップ
            if (!xv.has_value() || !yv.has_value() || !vv.has_value())
                continue;

            // データ追加
            heatmap_->AddData(*xv, *yv, *vv);
        }

        if (step % config_.log_interval == 0) {
            MetricsLogger::Instance()->LogImage(
                tag_,
                step,
                *heatmap_,
                config_.image_width,
                config_.image_height
            );
        }
    }

}