#pragma once
#include "anet/util.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/probe.hpp"
#include "anet/rl.hpp" 

namespace anet::rl {

    // ==============================================================

    class TaggedObserver {
    public:
		TaggedObserver(const std::string& tag) : tag_(tag) {}
        virtual std::string GetClassName() const = 0;
        virtual ~TaggedObserver() = default;
    protected:
        std::string ToStringInternal() const { return GetClassName() + "[" + tag_ + "]"; }
        std::string tag_;
    };

    class TaggedTrainObserver : public TaggedObserver, public anet::rl::TrainObserver {
    public:
		TaggedTrainObserver(const std::string& tag) : TaggedObserver(tag) {}
        virtual std::string ToString() const override { return ToStringInternal(); }
        virtual ~TaggedTrainObserver() = default;
    };

    class TaggedLearnObserver : public TaggedObserver, public anet::rl::LearnObserver {
    public:
        TaggedLearnObserver(const std::string& tag) : TaggedObserver(tag) {}
        virtual std::string ToString() const override { return ToStringInternal(); }
        virtual ~TaggedLearnObserver() = default;
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

    class HeatMapVectorObserver : public anet::rl::TaggedTrainObserver {
    public:
        HeatMapVectorObserver(
            const std::string& tag,
            const HeatMapObserverConfig& config,
            std::shared_ptr<VectorProbe> x_probe,
            std::shared_ptr<VectorProbe> y_probe,
            std::shared_ptr<VectorProbe> value_probe);

        void OnTrain(const TrainEvent& event) override;
        std::string GetClassName() const override { return "HeatMapVectorObserver"; }
    private:
        HeatMapObserverConfig config_;

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
    class TimeHistogramObserver : public anet::rl::TaggedTrainObserver {
    public:
        TimeHistogramObserver(
            const std::string& tag,
            const TimeHistogramObserverConfig& config,
            std::shared_ptr<VectorProbe> probe);

        void OnTrain(const TrainEvent& event) override;
        std::string GetClassName() const override { return "TimeHistogramObserver"; }
    private:
        TimeHistogramObserverConfig config_;
        std::unique_ptr<anet::TimeHistogram> histogram_;    ///< @todo ptr外し
        std::shared_ptr<VectorProbe> probe_;
    };

    class MultiPairHeatMapObserver : public anet::rl::TaggedTrainObserver {
    public:
        MultiPairHeatMapObserver(
            const std::string& tag,
            const HeatMapObserverConfig& config,
            const std::vector<std::shared_ptr<VectorProbe>>& axis_probes,
            std::shared_ptr<VectorProbe> value_probe);

        void OnTrain(const TrainEvent& event) override;
        std::string GetClassName() const override { return "MultiPairHeatMapObserver"; }
    private:
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
    class SweepedHeatMapObserver : public anet::rl::TaggedTrainObserver {
    public:
        SweepedHeatMapObserver(
            const std::string& tag,
            const SweepedHeatMapObserverConfig& config,
            std::shared_ptr<ISweepInputGenerator> input_gen,
            TensorFunction tensor_fn_,
            std::shared_ptr<ISweepOutputExtractor> output_ext,
            const std::unordered_map<std::string, std::string>& scalar_tag_label_map = {});

        void OnTrain(const TrainEvent& event) override;
        std::string GetClassName() const override { return "SweepedHeatMapObserver"; }
    private:
        SweepedHeatMapObserverConfig config_;
        std::unordered_map<std::string, std::string> scalar_label_tag_map_;

        std::shared_ptr<ISweepInputGenerator> input_gen_;
        TensorFunction tensor_fn_;
        std::shared_ptr<ISweepOutputExtractor> output_ext_;

        int grid_w_ = 0;
        int grid_h_ = 0;

        std::unique_ptr<anet::HeatMap> heatmap_;
    };

    class EpisodeEvalObserver : public anet::rl::LearnObserver {
    public:
        using ReportFunction = std::function<void(float total_reward)>;
    public:
        EpisodeEvalObserver(
            ReportFunction report_function,
            std::shared_ptr<anet::rl::SingleDiscreteEnvFactory> eval_env_factory,
            const ConfigData& config_data,
            const torch::Device& device,
            anet::rl::RunMode runmode_ = anet::rl::RunMode::Eval,
            int log_interval = 10, int eval_interval = 10,
            std::optional<seed_t> seed = std::nullopt);

        void OnLearn(const LearnEvent& event) override;
        std::string ToString() const override;
    private:
        ReportFunction report_function_;
        std::unique_ptr<anet::rl::BatchEnv> env_;
        anet::rl::RunMode runmode_;
        int log_interval_;
        int eval_interval_;
        //anet::EmaFilter<float> eval_total_reward_;
    };

    class FunctionTrainObserver : public anet::rl::TrainObserver {
    public:
        using Fn = std::function<void(const anet::rl::TrainEvent& event)>;
    public:
        FunctionTrainObserver(Fn fn, std::optional<std::string> name = std::nullopt);
        void OnTrain(const TrainEvent& event) override { fn_(event); }
        std::string ToString() const override { return name_; }
    private:
        Fn fn_;
        std::string name_;
    };

    // ===========================================================

    class FunctionLearnObserver : public anet::rl::LearnObserver {
    public:
        using Fn = std::function<void(const anet::rl::LearnEvent& event)>;
    public:
        FunctionLearnObserver(Fn fn, std::optional<std::string> name = std::nullopt);
        void OnLearn(const LearnEvent& event) override { fn_(event); }
        std::string ToString() const override { return name_; }
    private:
        Fn fn_;
        std::string name_;
    };

    // ==============================================================

    class MetricsLogObserverBase : public TaggedObserver {
    public:
        MetricsLogObserverBase(
            const std::string& tag, const std::string& key,
            anet::rl::StepAxis step_axis, std::optional<anet::rl::EventField> event_field,
            int interval, bool is_ema, float ema_alpha);
        virtual ~MetricsLogObserverBase() = default;
    protected:
        void OnUpdate(const UpdateEvent& event);
    private:
        std::optional<float> GetScalar(const UpdateEvent& event, anet::rl::EventField event_field);
    protected:
        std::string key_;
        anet::rl::StepAxis step_axis_;
        std::optional<anet::rl::EventField> event_field_;
        bool is_ema_;
		int interval_;
		anet::EmaFilter<float> val_ema_;
    };

    class MetricsLogTrainObserver : public MetricsLogObserverBase, public TrainObserver {
    public:
        MetricsLogTrainObserver(const std::string& tag, const std::string& key,
            anet::rl::StepAxis step_axis, std::optional<anet::rl::EventField> event_field,
            int interval, bool is_ema, float ema_alpha);
        void OnTrain(const TrainEvent& event) override { OnUpdate(event); }
        std::string GetClassName() const override { return "MetricsLogTrainObserver"; }
        virtual std::string ToString() const override { return ToStringInternal(); }
    };

    class MetricsLogLearnObserver : public MetricsLogObserverBase, public LearnObserver {
    public:
        MetricsLogLearnObserver(const std::string& tag, const std::string& key,
            anet::rl::StepAxis step_axis, std::optional<anet::rl::EventField> event_field,
            int interval, bool is_ema, float ema_alpha);
        void OnLearn(const LearnEvent& event) override { OnUpdate(event); }
        std::string GetClassName() const override { return "MetricsLogLearnObserver"; }
        virtual std::string ToString() const override { return ToStringInternal(); }
    };

	// ===========================================================

    class ObserverFactory {
    public:
        ObserverFactory(const ConfigData& config_data);

        std::vector<std::shared_ptr<anet::rl::TrainObserver>> GetUpdateObservers() { return train_observers_; }
        std::vector<std::shared_ptr<anet::rl::LearnObserver>> GetLearnObservers() { return learn_observers_; }
    private:
        std::vector<std::shared_ptr<anet::rl::TrainObserver>> train_observers_;
        std::vector<std::shared_ptr<anet::rl::LearnObserver>> learn_observers_;
    };
}

