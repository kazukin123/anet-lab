#pragma once
#include "anet/rl.hpp"  // PostUpdateObserver, Experience, ActionInfo, UpdateResult
#include "anet/probe.hpp"
#include "anet/metrics_logger.hpp"

namespace anet::rl {

    class MetricsLogObserver : public anet::rl::PostUpdateObserver {
    public:
        MetricsLogObserver() = default;
        virtual ~MetricsLogObserver() = default;

        void OnPostUpdate(
            int step,
            const anet::rl::BatchExperience& experiences,
            std::shared_ptr<const anet::rl::BatchUpdateResult> result
        ) override;
    };


    /**
     * @brief HeatMapObserver の設定
     */
    struct HeatMapObserverConfig {
        int width = 256;
        int height = 256;

        int log_interval = 100;
        size_t max_points = 30000;
        uint32_t flags = HM_Default;

        int image_width = -1;
        int image_height = -1;

        // HeatMap の軸定義（優先度：固定値 → Probe）
        bool override_xmin = false;
        bool override_xmax = false;
        bool override_ymin = false;
        bool override_ymax = false;

        float xmin = 0.0f;
        float xmax = 1.0f;
        float ymin = 0.0f;
        float ymax = 1.0f;
    };

    /**
     * @brief State/Reward/TensorProbe を使って HeatMap を生成する Observer
     */
    class HeatMapObserver : public anet::rl::PostUpdateObserver {
    public:
        HeatMapObserver(
            const std::string& tag,
            const HeatMapObserverConfig& config,
            IFloatProbe* x_probe,
            IFloatProbe* y_probe,
            IFloatProbe* value_probe);

        void OnPostUpdate(
            int step,
            const anet::rl::BatchExperience& batch_experience,
            std::shared_ptr<const anet::rl::BatchUpdateResult> result) override;

    private:
        HeatMapObserverConfig config_;
        std::string tag_;

        std::unique_ptr<IFloatProbe> x_probe_;
        std::unique_ptr<IFloatProbe> y_probe_;
        std::unique_ptr<IFloatProbe> value_probe_;

        std::unique_ptr<anet::HeatMap> heatmap_;  ///< @todo ptr外し
    };


    struct TimeHistogramObserverConfig {
        int bins = 128;
        int max_frames = 1000;
        int image_height = -1;
        int image_width = -1;
        TimeFrameMode mode = TimeFrameMode::Scale;
        uint32_t flags = anet::HeatMapFlags::HM_Default;
        int log_interval = 100;
        int frame_interval = 10;
        float base_min = std::numeric_limits<float>::quiet_NaN();
        float base_max = std::numeric_limits<float>::quiet_NaN();
        float alpha = 0.05f;
    };

    /**
    *@brief TimeHistogramObserver（Histogram＋Probe＋Logger を Observer 内に完結）
    *
    * ExtractTensorFn:
    *torch::Tensor func(const anet::rl::BatchUpdateResult& result);
    */
    class TimeHistogramObserver : public anet::rl::PostUpdateObserver {
    public:
        using ExtractTensorFn = std::function<std::optional<torch::Tensor>(std::shared_ptr<const anet::rl::BatchUpdateResult>, const std::string& key)>;

        static ExtractTensorFn DefaultExtractFn() {
                return [](auto r, const std::string& key) {
                    if (!r) return std::optional<torch::Tensor>{};
                    return r->GetTensor(key);
                    };
            }

        TimeHistogramObserver(
            const std::string& tag,
            const TimeHistogramObserverConfig& config,
            const std::string& key = "",
            ExtractTensorFn tensor_fn = DefaultExtractFn()
            ) : tag_(tag), config_(config), key_(key), extract_tensor_fn_(tensor_fn)
        {
            histogram_ = std::make_unique<anet::TimeHistogram>(
                config_.bins, config_.max_frames, config_.mode, config_.flags, config_.base_min, config_.base_max, config_.alpha);
            vec_probe_ = std::make_unique<TensorVectorProbe>();
        }

        void OnPostUpdate(
            int step,
            const anet::rl::BatchExperience& batch_experience,
            std::shared_ptr<const anet::rl::BatchUpdateResult> result) override
        {
            // 抽出関数で Tensor を取得
            auto t = extract_tensor_fn_(result, key_);
            if (t.has_value()) {
                // Probe にセット → vector に変換
                vec_probe_->UpdateTensor(*t);
                std::vector<float> values;
                if (vec_probe_->TryGetVector(values)) {
                    histogram_->AddBatch(values);
                }
            }

            // フレーム更新
            if (step % config_.frame_interval == 0) {
                histogram_->NextFrame();
            }

            // ログ出力
            if (step % config_.log_interval == 0) {
                MetricsLogger::Instance()->LogImage(tag_, step, *histogram_, config_.image_width, config_.image_height);
            }
        }
    private:
        TimeHistogramObserverConfig config_;
        ExtractTensorFn extract_tensor_fn_;
        std::string key_;
        std::string tag_;

        std::unique_ptr<anet::TimeHistogram> histogram_;    ///< @todo ptr外し
        std::unique_ptr<TensorVectorProbe> vec_probe_;      ///< @todo ptr外し
    };

}

