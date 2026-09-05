// anet/observers.hpp

#pragma once

#include "anet/thread.hpp"
#include "anet/util.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/probe.hpp"
#include "anet/rl.hpp" 
#include "anet/image.hpp"
#include "anet/nn.hpp"


namespace anet::rl {

    // ==============================================================

    class EvalRunner;

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

    // ==============================================================
    
    /**
     * @brief HeatMapObserver の設定
     */
    struct HeatMapObserverConfig {
        int width = 256;
        int height = 256;

        int log_interval = 100;
        int max_points = 30000;
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

    class HeatMapVectorObserver : public anet::rl::TaggedTrainObserver, public anet::rl::ImageProvider {
    public:
        HeatMapVectorObserver(
            const std::string& tag,
            const HeatMapObserverConfig& config,
            std::shared_ptr<VectorProbe> x_probe,
            std::shared_ptr<VectorProbe> y_probe,
            std::shared_ptr<VectorProbe> value_probe);

        void OnTrain(const TrainEvent& event) override;
        std::string GetClassName() const override { return "HeatMapVectorObserver"; }
    public:
        anet::rl::ImageData GetImageData(int width = -1, int height = -1) override;
    private:
        HeatMapObserverConfig config_;

        std::shared_ptr<VectorProbe> x_probe_;
        std::shared_ptr<VectorProbe> y_probe_;
        std::shared_ptr<VectorProbe> value_probe_;

        std::unique_ptr<anet::HeatMap> heatmap_;
    private:
        std::shared_mutex mutex_;
        step_t captured_step_ = 0;
    };
    
    // ==============================================================

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
    class TimeHistogramObserver : public anet::rl::TaggedLearnObserver, public anet::rl::ImageProvider {
    public:
        TimeHistogramObserver(
            const std::string& tag,
            const TimeHistogramObserverConfig& config,
            std::shared_ptr<VectorProbe> probe);

        void OnLearn(const LearnEvent& event) override;
        std::string GetClassName() const override { return "TimeHistogramObserver"; }
    public:
        anet::rl::ImageData GetImageData(int width = -1, int height = -1) override;
    private:
        TimeHistogramObserverConfig config_;
        std::unique_ptr<anet::TimeHistogram> histogram_;    ///< @todo ptr外し
        std::shared_ptr<VectorProbe> probe_;
        std::optional<anet::IntervalGate> frame_gate_;      ///< frame_interval <= 0 で無効
        std::optional<anet::IntervalGate> log_gate_;        ///< log_interval <= 0 で無効
    private:
        std::shared_mutex mutex_;
        step_t captured_step_ = 0;
    };
    
	// ==============================================================

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
    

    // -----------------------------------------------------------------
    // SweepedHeatMapObserver
    // -----------------------------------------------------------------

    struct SweepedHeatMapObserverConfig {
        int log_interval = 100;
        uint32_t flags = anet::HeatMapFlags::HM_Default;
        int grid_width = -1;    ///< 内部で生成するGridの幅。-1で指定なし(自動)
        int grid_height = -1;   ///< 内部で生成するGridの高さ。-1で指定なし(自動)
        int image_width = -1;   ///< 出力する画像の幅。-1で指定なし(自動)
        int image_height = -1;  ///< 出力する画像の高さ。-1で指定なし(自動)
        std::string output_key = "q"; ///< NN出力TensorDictから描画対象Tensorを選ぶkey
    };

    /**
     * @brief NN 可視化のための 2D Sweep HeatMap Observer。
     *
     * - InputGenerator が grid_x, grid_y ごとの NN入力Tensor を生成
     * - Observer が全セル分まとめてバッチ forward 実行
     * - OutputExtractor が batched 出力から (grid_x, grid_y) の値を抽出
     * - HeatMap に追加して画像として出力
     */
    class SweepedHeatMapObserver : public anet::rl::TaggedLearnObserver, public anet::rl::ImageProvider {
    public:
        SweepedHeatMapObserver(
            const std::string& tag,
            const SweepedHeatMapObserverConfig& config,
            std::shared_ptr<ISweepInputGenerator> input_gen,
            TensorDictFunction tensor_fn_,
            std::shared_ptr<ISweepOutputExtractor> output_ext,
            const std::unordered_map<std::string, std::string>& scalar_tag_label_map = {}
            );

        void OnLearn(const LearnEvent& event) override;
        std::string GetClassName() const override { return "SweepedHeatMapObserver"; }
    public:
        anet::rl::ImageData GetImageData(int width = -1, int height = -1) override;
    private:
        std::pair<ExtractResult, std::vector<torch::Tensor>> Render();
    private:
        SweepedHeatMapObserverConfig config_;
        std::unordered_map<std::string, std::string> scalar_label_tag_map_;

        std::shared_ptr<ISweepInputGenerator> input_gen_;
        TensorDictFunction tensor_fn_;
        std::shared_ptr<ISweepOutputExtractor> output_ext_;

        int grid_w_ = 0;
        int grid_h_ = 0;
        std::unique_ptr<anet::HeatMap> heatmap_;
        std::optional<anet::IntervalGate> log_gate_;    ///< log_interval <= 0 で無効
    private:
        std::shared_mutex mutex_;
        step_t captured_step_ = 0;
    };

    class EpisodeEvalObserver : public anet::rl::LearnObserver {
    public:
        EpisodeEvalObserver(
            std::shared_ptr<EvalRunner> eval_runner,
            int eval_interval,
            bool use_background);

        void OnLearn(const LearnEvent& event) override;
        std::string ToString() const override;

        ~EpisodeEvalObserver() override;
    private:
        void RunEvaluationSession(const StepCounts& event_counts);      ///< EvalRunnerを評価session完了まで駆動
        void RethrowCompletedBackgroundEval();                          ///< 完了済みのバックグラウンド評価失敗を呼び出し元へ伝播
        void WaitBackgroundEval();                                      ///< 前回のバックグラウンド評価を待ち、失敗していれば呼び出し元へ伝播
    private:
        std::shared_ptr<EvalRunner> eval_runner_;
        const bool use_background_;
        std::optional<anet::IntervalGate> eval_gate_;   ///< eval_interval <= 0 で無効

        std::unique_ptr<anet::PinnedThreadPool> eval_pool_;
        std::future<void> eval_future_;
    };


    // -----------------------------------------------------------------
    // FunctionObserver
    // -----------------------------------------------------------------

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


    // -----------------------------------------------------------------
    // Conv2dVisualizationObserver
    // -----------------------------------------------------------------

    class Conv2dVisualizationObserver : public TaggedTrainObserver, public anet::rl::ImageProvider {
    public:
        Conv2dVisualizationObserver(
            const std::string& tag,
            int episode_interval,
            const Conv2dVisualizerConfig& vis_config
        );

        void OnTrain(const TrainEvent& event) override;
        std::string GetClassName() const override { return "Conv2dVisualizationObserver"; }
        ImageData GetImageData(int width = -1, int height = -1) override;
    private:
        const int episode_interval_;
        const Conv2dVisualizer visualizer_;

        bool is_recording_ = false;
        bool is_first_record_ = true;
        int local_episode_count_ = 0; // env[0]専用のエピソードカウンタ
        mutable std::mutex image_mutex_;
        ImageData last_image_;
    };


    // -----------------------------------------------------------------
    // MetricsLogObserver
    // -----------------------------------------------------------------

    class MetricsLogObserverBase : public TaggedObserver {
    public:
        MetricsLogObserverBase(
            const std::string& tag, const std::string& key,
            anet::rl::StepAxis step_axis, std::optional<anet::rl::EventField> event_field,
            int interval, bool is_ema, float ema_alpha, std::optional<float> clip);
        virtual ~MetricsLogObserverBase() = default;
    protected:
        void OnUpdate(const UpdateEvent& event);
        void OnTrainUpdate(const TrainEvent& event);
        void OnGenericUpdate(
            const StepCounts& counts,
            std::shared_ptr<const Agent> agent,
            std::shared_ptr<const Runner> runner,
            std::shared_ptr<const BatchEnv> env,
            const BatchExperience* experience,
            const BatchUpdateResultList* update_result_list,
            std::shared_ptr<const BatchActionInfo> action_info = nullptr);
    private:
        using MetricsData = std::pair<step_t, std::optional<float>>;
        using MetricsDataList = std::vector<MetricsData>;
        struct UpdateResultMetricsLookup {
            bool recognized = false;
            MetricsDataList data_list;
        };
    private:
        MetricsDataList GetMetricsDataList(
            const StepCounts& counts,
            std::shared_ptr<const Agent> agent,
            std::shared_ptr<const Runner> runner,
            std::shared_ptr<const BatchEnv> env,
            const BatchExperience* experience,
            const BatchUpdateResultList* update_result_list,
            std::shared_ptr<const BatchActionInfo> action_info);

        MetricsData GetMetricsData(
            const StepCounts& counts,
            std::shared_ptr<const Agent> agent,
            std::shared_ptr<const Runner> runner,
            std::shared_ptr<const BatchEnv> env,
            const BatchExperience* experience,
            std::shared_ptr<const BatchActionInfo> action_info,
            EventField event_field);
        UpdateResultMetricsLookup GetMetricsDataListFromUpdateResultList(
            const StepCounts& counts,
            const BatchUpdateResultList* update_result_list);

        /// interval を検証して IntervalGate 用の値へ変換する（1 未満は ANET_SYSTEM_ERROR）
        static uint64_t ValidateMetricsInterval(const std::string& tag, int interval);
    protected:
        std::string key_;
        anet::rl::StepAxis step_axis_;
        std::optional<anet::rl::EventField> event_field_;
        bool is_ema_;
		anet::IntervalGate gate_;
		anet::EmaFilter<float> val_ema_;
        std::optional<float> clip_;
    };

    class MetricsLogTrainObserver : public MetricsLogObserverBase, public TrainObserver {
    public:
        MetricsLogTrainObserver(const std::string& tag, const std::string& key,
            anet::rl::StepAxis step_axis, std::optional<anet::rl::EventField> event_field,
            int interval, bool is_ema, float ema_alpha, std::optional<float> clip);
            
        void OnTrain(const TrainEvent& event) override { OnTrainUpdate(event); }
        std::string GetClassName() const override { return "MetricsLogTrainObserver"; }
        virtual std::string ToString() const override { return ToStringInternal(); }
    };

    class MetricsLogLearnObserver : public MetricsLogObserverBase, public LearnObserver {
    public:
        MetricsLogLearnObserver(const std::string& tag, const std::string& key,
            anet::rl::StepAxis step_axis, std::optional<anet::rl::EventField> event_field,
            int interval, bool is_ema, float ema_alpha, std::optional<float> clip);
            
        void OnLearn(const LearnEvent& event) override { OnUpdate(event); }
        std::string GetClassName() const override { return "MetricsLogLearnObserver"; }
        virtual std::string ToString() const override { return ToStringInternal(); }
    };

    class MetricsLogEpisodeEndObserver : public MetricsLogObserverBase, public EpisodeEndObserver {
    public:
        MetricsLogEpisodeEndObserver(const std::string& tag, const std::string& key,
            anet::rl::StepAxis step_axis, std::optional<anet::rl::EventField> event_field,
            int interval, bool is_ema, float ema_alpha, std::optional<float> clip);
            
        void OnEpisodeEnd(const EpisodeEndEvent& event) override
        {
            OnGenericUpdate(event.counts, event.agent, event.runner, event.env, nullptr, nullptr);
        }
        std::string GetClassName() const override { return "MetricsLogEpisodeEndObserver"; }
        virtual std::string ToString() const override { return ToStringInternal(); }
    };

    class MetricsLogSessionEndObserver : public MetricsLogObserverBase, public SessionEndObserver {
    public:
        MetricsLogSessionEndObserver(const std::string& tag, const std::string& key,
            anet::rl::StepAxis step_axis, std::optional<anet::rl::EventField> event_field,
            int interval, bool is_ema, float ema_alpha, std::optional<float> clip);

        void OnSessionEnd(const SessionEndEvent& event) override
        {
            OnGenericUpdate(event.counts, event.agent, event.runner, event.env, nullptr, nullptr);
        }
        std::string GetClassName() const override { return "MetricsLogSessionEndObserver"; }
        virtual std::string ToString() const override { return ToStringInternal(); }
    };


    class MetricsLogTraceObserver : public TaggedObserver, public EpisodeEndObserver {
    public:
        MetricsLogTraceObserver(const std::string& tag, std::vector<std::string> keys,
            StepAxis step_axis, EventField field);
        void OnEpisodeEnd(const EpisodeEndEvent& event) override;
        std::string GetClassName() const override { return "MetricsLogTraceObserver"; }
        std::string ToString() const override { return ToStringInternal(); }
    private:
        std::vector<std::string> keys_;
        StepAxis step_axis_;
        EventField field_;
    };

    // -----------------------------------------------------------------
    // GraphVizObserver
    // -----------------------------------------------------------------

    class GraphVizObserver : public TaggedTrainObserver {
    public:
        GraphVizObserver(
            const std::string& tag,
            int step_interval,           ///< 既存のステップ間隔 (-1で無効)
            int episode_interval,        ///< 新規のエピソード間隔 (-1で無効)
            const std::string& provider_key,
            std::optional<anet::rl::EventField> event_field);

        void OnTrain(const TrainEvent& event) override;
        std::string GetClassName() const override { return "GraphVizObserver"; }
    private:
        const anet::graphviz::GraphVizProvider* FindProvider(const TrainEvent& event) const;
    private:
        int step_interval_;
        int episode_interval_;
        std::string provider_key_;
        std::optional<anet::rl::EventField> event_field_;

        // エピソードキャプチャ用の状態
        bool is_recording_ = false;
        int local_episode_count_ = 0;
    };


    // -----------------------------------------------------------------
    // ObserverFactory
    // -----------------------------------------------------------------

    class ObserverFactory {
    public:
        struct ParsedTrainObserver {
            RunnerScope scope = RunnerScope::TRAIN;
            std::string eval_name;
            std::shared_ptr<anet::rl::TrainObserver> obs;
        };
        struct ParsedLearnObserver {
            RunnerScope scope = RunnerScope::TRAIN;
            std::string eval_name;
            std::shared_ptr<anet::rl::LearnObserver> obs;
        };
        struct ParsedEpisodeEndObserver {
            RunnerScope scope = RunnerScope::TRAIN;
            std::string eval_name;
            std::shared_ptr<anet::rl::EpisodeEndObserver> obs;
        };
        struct ParsedSessionEndObserver {
            RunnerScope scope = RunnerScope::TRAIN;
            std::string eval_name;
            std::shared_ptr<anet::rl::SessionEndObserver> obs;
        };

        /// scalar metric 1 件の解決済み定義。
        /// 設定の書き方ではなく、実際に構築した Observer の内容を表す。
        /// 解析側はこれを正本として読み、設定から step 座標系を再導出しない (ADR 0029)。
        struct ScalarMetricDef {
            std::string tag;
            std::string step_axis;      ///< train_step / exp_step など、config token と同じ表記
            std::string runner;         ///< step counter を所有する Runner。train または eval 名
            std::string event;          ///< train / learn / episode_end / session_end
            std::string target;         ///< agent / env / exp / update_result / runner / action_info。未指定は空
            std::string source_key;     ///< metric key として採用した token
            bool has_ema = false;
            float ema_alpha = 0.0f;
            int interval = 1;
            std::optional<float> clip;
            RunnerScope scope = RunnerScope::TRAIN;
            std::string eval_name;
            std::optional<int> eval_episodes;  ///< RunManager が付与するセッション採用予定数
            std::optional<int64_t> num_envs;   ///< 構築済み eval Env の lane 数。SHARED の group 数ではない
            ScalarMetricSubscription subscription;
        };
        /// trace の解決済み定義。keys の順序は値の取得順と同じ。
        struct TraceMetricDef {
            std::string tag;
            std::string step_axis;
            std::string runner;
            std::string event;
            std::string target;
            std::vector<std::string> keys;
            RunnerScope scope = RunnerScope::TRAIN;
            std::string eval_name;
            std::optional<int> eval_episodes;
            std::optional<int64_t> num_envs;
        };
    public:
        ObserverFactory(const ConfigData& config_data);

        std::vector<ParsedTrainObserver> GetUpdateObservers() { return train_observers_; }
        std::vector<ParsedLearnObserver> GetLearnObservers() { return learn_observers_; }
        std::vector<ParsedEpisodeEndObserver> GetEpisodeEndObservers() { return episode_end_observers_; }
        std::vector<ParsedSessionEndObserver> GetSessionEndObservers() { return session_end_observers_; }
        const std::vector<ScalarMetricDef>& GetScalarMetricDefs() const { return scalar_metric_defs_; }
        const std::vector<TraceMetricDef>& GetTraceMetricDefs() const { return trace_metric_defs_; }
    private:
        std::vector<ParsedTrainObserver> train_observers_;
        std::vector<ParsedLearnObserver> learn_observers_;
        std::vector<ParsedEpisodeEndObserver> episode_end_observers_;
        std::vector<ParsedSessionEndObserver> session_end_observers_;
        std::vector<ScalarMetricDef> scalar_metric_defs_;
        std::vector<TraceMetricDef> trace_metric_defs_;
    };

    /// scalar metric の解決済み定義を `metrics.scalar.defs` レコードの data 部へ変換する。
    /// tag をキーにした object を返す。未設定の target / EMA / clip と train scope の eval 情報は null。
    anet::json ScalarMetricDefsToJson(const std::vector<ObserverFactory::ScalarMetricDef>& defs);
    /// trace metric の定義を、宣言順の keys 配列を持つ object へ変換する。
    anet::json TraceMetricDefsToJson(const std::vector<ObserverFactory::TraceMetricDef>& defs);

}
