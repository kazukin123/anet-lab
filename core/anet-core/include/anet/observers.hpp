#pragma once
#include "anet/util.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/probe.hpp"
#include "anet/rl.hpp" 

namespace anet::rl {

    class MetricsLogObserver : public anet::rl::PostUpdateObserver {
    public:
        struct MetricDef {
            std::string tag;
            anet::rl::StepAxis step_axis;
		};
        using MetricsDefMap = std::unordered_map<std::string, std::vector<MetricDef>>;
    public:
        MetricsLogObserver();
        virtual ~MetricsLogObserver() = default;

        void OnPostUpdate(const PostUpdateEvent& event) override;
    private:
        MetricsDefMap metric_def_map_;
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

    //class HeatMapObserver : public anet::rl::PostUpdateObserver {
    //public:
    //    HeatMapObserver(
    //        const std::string& tag,
    //        const HeatMapObserverConfig& config,
    //        std::shared_ptr<IFloatProbe> x_probe,
    //        std::shared_ptr<IFloatProbe> y_probe,
    //        std::shared_ptr<IFloatProbe> value_probe);

    //    void OnPostUpdate(
    //        int step,
    //        std::shared_ptr<Agent> agent,
    //        const anet::rl::BatchExperience& batch_experience,
    //        std::shared_ptr<const anet::rl::BatchUpdateResult> result) override;
    //private:
    //    HeatMapObserverConfig config_;
    //    std::string tag_;

    //    std::shared_ptr<IFloatProbe> x_probe_;
    //    std::shared_ptr<IFloatProbe> y_probe_;
    //    std::shared_ptr<IFloatProbe> value_probe_;

    //    std::unique_ptr<anet::HeatMap> heatmap_;  ///< @todo ptr外し
    //};

    class HeatMapVectorObserver : public anet::rl::PostUpdateObserver {
    public:
        HeatMapVectorObserver(
            const std::string& tag,
            const HeatMapObserverConfig& config,
            std::shared_ptr<VectorProbe> x_probe,
            std::shared_ptr<VectorProbe> y_probe,
            std::shared_ptr<VectorProbe> value_probe);

        void OnPostUpdate(const PostUpdateEvent& event) override;
    private:
        HeatMapObserverConfig config_;
        std::string tag_;

        std::shared_ptr<VectorProbe> x_probe_;
        std::shared_ptr<VectorProbe> y_probe_;
        std::shared_ptr<VectorProbe> value_probe_;

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
        float alpha = 1.0f;
    };

    /**
    *@brief TimeHistogramObserver（Histogram＋Probe＋Logger を Observer 内に完結）
    *
    * ExtractTensorFn:
    *torch::Tensor func(const anet::rl::BatchUpdateResult& result);
    */
    class TimeHistogramObserver : public anet::rl::PostUpdateObserver {
    public:
        TimeHistogramObserver(
            const std::string& tag,
            const TimeHistogramObserverConfig& config,
            std::shared_ptr<VectorProbe> probe);

        void OnPostUpdate(const PostUpdateEvent& event) override;
    private:
        TimeHistogramObserverConfig config_;
        std::string tag_;
        std::unique_ptr<anet::TimeHistogram> histogram_;    ///< @todo ptr外し
        std::shared_ptr<VectorProbe> probe_;
    };

    class MultiPairHeatMapObserver : public anet::rl::PostUpdateObserver {
    public:
        MultiPairHeatMapObserver(
            const std::string& tag,
            const HeatMapObserverConfig& config,
            const std::vector<std::shared_ptr<VectorProbe>>& axis_probes,
            std::shared_ptr<VectorProbe> value_probe);

        void OnPostUpdate(const PostUpdateEvent& event) override;
    private:
        std::string tag_;
        HeatMapObserverConfig config_;
        std::vector<std::shared_ptr<VectorProbe>> axis_probes_;
        std::shared_ptr<VectorProbe> value_probe_;
        std::unique_ptr<anet::HeatMap> heatmap_;
    };

    struct SweepedHeatMapObserverConfig {
        int log_interval = 100;
        uint64_t flags = anet::HeatMapFlags::HM_Default;
        int grid_width = -1;    ///< 内部で生成するGridの幅。-1で指定なし(自動)
        int grid_height = -1;   ///< 内部で生成するGridの高さ。-1で指定なし(自動)
        int image_width = -1;   ///< 出力する画像の幅。-1で指定なし(自動)
        int image_height = -1;  ///< 出力する画像の高さ。-1で指定なし(自動)
    };

    /**
     * @brief NN 可視化のための 2D Sweep HeatMap Observer。
     *
     * - InputGenerator が grid_x, grid_y ごとの NN入力Tensor を生成
     * - Observer が全セル分まとめてバッチ forward 実行
     * - OutputExtractor が batched 出力から (grid_x, grid_y) の値を抽出
     * - HeatMap に追加して画像として出力
     */
    class SweepedHeatMapObserver : public anet::rl::PostUpdateObserver {
    public:
        SweepedHeatMapObserver(
            const std::string& heatmap_tag,
            const SweepedHeatMapObserverConfig& config,
            std::shared_ptr<ISweepInputGenerator> input_gen,
            TensorFunction tensor_fn_,
            std::shared_ptr<ISweepOutputExtractor> output_ext,
            const std::unordered_map<std::string, std::string>& scalar_tag_label_map = {});

        void OnPostUpdate(const PostUpdateEvent& event) override;
    private:
        std::string heatmap_tag_;
        SweepedHeatMapObserverConfig config_;
        std::unordered_map<std::string, std::string> scalar_label_tag_map_;

        std::shared_ptr<ISweepInputGenerator> input_gen_;
        TensorFunction tensor_fn_;
        std::shared_ptr<ISweepOutputExtractor> output_ext_;

        int grid_w_ = 0;
        int grid_h_ = 0;

        std::unique_ptr<anet::HeatMap> heatmap_;
    };

    class EpisodeEvalObserver : public anet::rl::PostUpdateObserver {
    public:
        EpisodeEvalObserver(
            const std::string& tag,
            std::shared_ptr<anet::rl::SingleDiscreteEnvFactory> eval_env_factory,
            const ConfigData& config_data,
            const torch::Device& device,
            anet::rl::RunMode runmode_ = anet::rl::RunMode::Eval,
            int log_interval = 10, int eval_interval = 10,
            float ema_decay = 1.00);

        void OnPostUpdate(const PostUpdateEvent& event) override;
    private:
        std::string tag_;
        std::unique_ptr<anet::rl::BatchEnv> env_;
        anet::rl::RunMode runmode_;
        int log_interval_;
        int eval_interval_;
        anet::EmaFilter<float> eval_total_reward_;
    };

    class FunctionObserver : public anet::rl::PostUpdateObserver {
    public:
        using Fn = std::function<void(const anet::rl::PostUpdateEvent& event)>;
    public:
        FunctionObserver(Fn fn) : fn_(std::move(fn)) { ; }

        void OnPostUpdate(const anet::rl::PostUpdateEvent& event) override
        {
            fn_(event);
        }
    private:
        Fn fn_;
    };
}

