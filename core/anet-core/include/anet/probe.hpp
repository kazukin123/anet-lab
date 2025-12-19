#pragma once
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include "anet/rl.hpp"

namespace anet::rl {

    //==============================================================
    // Probe interface
    //==============================================================

    class ScalarProbe {
    public:
        virtual std::optional<float> GetFloat(const TrainEvent& event) const = 0;

        virtual std::string GetName() const = 0;
        virtual std::optional<float> GetMin() const = 0;
        virtual std::optional<float> GetMax() const = 0;

        virtual ~ScalarProbe() = default;
    };

    /**
     * @brief TimeHistogram 用の float ベクトル Probe
     */
    class VectorProbe {
    public:
        virtual std::optional<std::vector<float>> GetVector(const UpdateEvent& event) const = 0;
        virtual std::optional<std::vector<float>> GetVector(const TrainEvent& event) const { return GetVector((UpdateEvent)event); }

        virtual std::string GetName() const = 0;
        virtual std::optional<float> GetMin() const = 0;
        virtual std::optional<float> GetMax() const = 0;

        virtual ~VectorProbe() = default;
    };

    //==============================================================
    // ScalarProbe impl
    //==============================================================

    /**
     * @brief MetricsMap に格納された scalar を参照する Probe。
     */
    class MetricsScalarProbe : public ScalarProbe {
    public:
        explicit MetricsScalarProbe(std::string key, std::optional<std::string> name = std::nullopt)
            : key_(std::move(key)), name_(name.has_value() ? *name : "") {
        }

        std::optional<float> GetFloat(const TrainEvent& event) const override;

        std::string GetName() const override { return name_.empty() ? key_: name_; }
        std::optional<float> GetMin() const override { return std::nullopt; }
        std::optional<float> GetMax() const override { return std::nullopt; }

    private:
        std::string name_;
        std::string key_;
    };

    /**
     * @brief 固定値を返す Probe。
     */
    class StaticScalarProbe : public ScalarProbe {
    public:
        explicit StaticScalarProbe(float value, const std::string& name)
            : value_(value), name_(name) {
        }

        std::optional<float> GetFloat(const TrainEvent& event) const override;

        std::string GetName() const override { return name_; }
        std::optional<float> GetMin() const override { return value_; }
        std::optional<float> GetMax() const override { return value_; }
    private:
        std::string name_;
        float value_;
    };

    /**
     * @brief 関数オブジェクトから値を生成する汎用 Probe。
     */
    class FunctionScalarProbe : public ScalarProbe {
    public:
        using Fn = std::function<std::optional<float>(const TrainEvent& event)>;

        FunctionScalarProbe(
            const std::string& name,
            Fn fn,
            std::optional<float> min = std::nullopt,
            std::optional<float> max = std::nullopt)
            : fn_(std::move(fn)), min_(min), max_(max) {
        }

        std::optional<float> GetFloat(const TrainEvent& event) const override;

        std::string GetName() const override { return name_; }
        std::optional<float> GetMin() const override { return min_; }
        std::optional<float> GetMax() const override { return max_; }
    private:
        std::string name_;
        Fn fn_;
        std::optional<float> min_;
        std::optional<float> max_;
    };

    //==============================================================
    // VectorProbe impl
    //==============================================================

    class BatchExperienceBasedVectorProbe : public VectorProbe {
    public:
        virtual std::optional<std::vector<float>> GetVectorFromExperience(const UpdateEvent& event) const = 0;
    public:
        /**
         * @brief 観測情報からFloat値を生成
         */
		std::optional<std::vector<float>> GetVector(const UpdateEvent& event) const override
        {
			/// @todo step_counts のどれを使うか選択できるようにする
			auto step = event.counts.train_step;

            std::vector<float> ret;
            auto exps = event.batch_exp.ToExperienceList();
            for (auto e : exps) {
                auto vec = GetVectorFromExperience(event);
                if (vec.has_value()) {
                    ret.insert(ret.begin(), vec->begin(), vec->end());
                }
            }
            return ret;
        };
        virtual ~BatchExperienceBasedVectorProbe() = default;
    };

    /**
     * @brief EnvSpec.state_spec と index を用いて state から float を抽出する Probe。
     */
    class BatchExperienceStateProbe : public VectorProbe {
    public:
        /**
         * @param spec  値範囲抽出に使うSteteSpec。nullptrの場合は値範囲を定義しない。
         * @param index 抽出する state の次元（flatten 後の index）
         */
        BatchExperienceStateProbe(
            int64_t state_index, const anet::rl::StateSpec* spec = nullptr, bool for_next_state = true,
            const std::optional<std::string> name = std::nullopt);

        std::optional<std::vector<float>> GetVector(const UpdateEvent& event) const override;

        std::string GetName() const override { return name_; }
        std::optional<float> GetMin() const override { return min_; }
        std::optional<float> GetMax() const override { return max_; }
    private:
        std::string name_;
        int state_index_;
        bool for_next_state_;
        std::optional<float> min_;
        std::optional<float> max_;
    };

    class BatchExperienceRewardProbe : public VectorProbe {
    public:
        /**
         * @param spec 値範囲抽出に使うEnvSpec。nullの場合は値範囲を定義しない。
         */
        BatchExperienceRewardProbe(const anet::rl::EnvSpec* spec = nullptr, std::optional<std::string> name = std::nullopt);

        std::optional<std::vector<float>> GetVector(const UpdateEvent& event) const override;

        std::string GetName() const override { return name_; }
        std::optional<float> GetMin() const override { return min_; }
        std::optional<float> GetMax() const override { return max_; }
    private:
        std::string name_;
        std::optional<float> min_;
        std::optional<float> max_;
    };

    class BatchUpdateResultTensorToVectorProbe : public VectorProbe {
    public:
        BatchUpdateResultTensorToVectorProbe(const std::string& key, std::optional<float> min = std::nullopt, std::optional<float> max = std::nullopt);

        std::optional<std::vector<float>> GetVector(const UpdateEvent& event) const override;

        std::string GetName() const override { return name_; }
        std::optional<float> GetMin() const override { return min_; }
        std::optional<float> GetMax() const override { return max_; }
    private:
        std::string key_;
        std::string name_;
        std::optional<float> min_;
        std::optional<float> max_;
    };

    class BatchActionInfoToVectorProbe : public VectorProbe {
    public:
        BatchActionInfoToVectorProbe(const std::string& key, std::optional<float> min = std::nullopt, std::optional<float> max = std::nullopt);

        std::optional<std::vector<float>> GetVector(const UpdateEvent& event) const override { return std::nullopt; }
        std::optional<std::vector<float>> GetVector(const TrainEvent& event) const override;

        std::string GetName() const override { return name_; }
        std::optional<float> GetMin() const override { return min_; }
        std::optional<float> GetMax() const override { return max_; }
    private:
        std::string key_;
        std::string name_;
        std::optional<float> min_;
        std::optional<float> max_;
    };

    class AgentTensorVectorProbe : public VectorProbe {
    public:
        enum AutoScaleMode {
            DISABLE,
            PER_STEP,
            GLOBAL,
        };
    public:
        AgentTensorVectorProbe(
            const std::string& key,
            int index,
            const anet::rl::StateSpec* state_spec = nullptr,
            const anet::rl::ActionSpec* action_spec = nullptr,
            AutoScaleMode auto_scale_mode = AutoScaleMode::DISABLE,
            std::optional<float> min_override = std::nullopt,
            std::optional<float> max_override = std::nullopt,
            std::optional<std::string> name = std::nullopt);

        std::optional<std::vector<float>> GetVector(const UpdateEvent& event) const override;

        std::string GetName() const override { return name_; }
        std::optional<float> GetMin() const override { return min_; }
        std::optional<float> GetMax() const override { return max_; }
    private:
        std::string name_;
        std::string key_;
        int index_;
        AutoScaleMode auto_scale_mode_;
        mutable std::optional<float> min_;  ///< @todo mutable除去
        mutable std::optional<float> max_;  ///< @todo mutable除去
    };

    //==============================================================
    // Sweep
    //==============================================================

    class ISweepInputGenerator {
    public:
        // nullopt の axis は Generator が決めてよい
        /// @param grid_width gridの幅。-1は指定しないの意味
        /// @param grid_height gridの高さ。-1は指定しないの意味
        /// @brief Observer側からの希望Gridサイズを受け取る
        virtual void ApplyGridSize(int grid_width, int grid_height) = 0;

        /// @brief Generator が最終決定した Gridサイズ
        virtual std::pair<int, int> GetGridSize() const = 0;

        /// @return Tensor: [W*H, state_dim], Flatten済み、device上
        virtual torch::Tensor BuildInputTensor() = 0;

        /**
         * @brief flatten された state のサイズを返す
         */
        virtual int64_t GetFlattenSize() const = 0;

        virtual ~ISweepInputGenerator() = default;
    };

    struct ExtractResult {
        torch::Tensor grid;                          // HeatMap 用
        std::vector<std::string> labels;             // それぞれの scalar 名
        std::vector<torch::Tensor> scalars;          // 個別 scalar 値（GPU Tensor）
    };

    using ValueExtractFunction = std::function<
        ExtractResult(const torch::Tensor& t, const std::unordered_set<std::string>& required_labels) >;

    /**
     * @brief SweepedHeatMap の出力処理側インタフェース。
     * batched NN 出力から HeatMap の 1セル値を抽出する役割を持つ。
     *
     * GridSize は InputGenerator が決定し、Observer が確定値を通知する。
     */
    class ISweepOutputExtractor {
    public:
        virtual ~ISweepOutputExtractor() = default;

        /**
         * @brief Observer から確定された GridSize を受け取る。
         * @param grid_width  グリッド幅 (W)
         * @param grid_height グリッド高さ (H)
         */
        virtual void ApplyGridSize(int grid_width, int grid_height) = 0;

        /**
         * @brief OutputExtractor が保持している最終 GridSize を返す。
         * @return (width, height)
         */
        virtual std::pair<int, int> GetGridSize() const = 0;

        /// @brief batched output から W*H 個の値を GPU 上で抽出
        /// @return shape = [W*H], dtype=float32, device=device
        virtual ExtractResult Extract(const torch::Tensor& output,
            const std::unordered_set<std::string>& required_labels) = 0;
    };

    namespace extractor {
        ExtractResult MaxExtractor(const torch::Tensor& t, const std::unordered_set<std::string>& req);
        ExtractResult MeanExtractor(const torch::Tensor& t, const std::unordered_set<std::string>& req);
        ExtractResult IndexExtractor(const torch::Tensor& t, const std::unordered_set<std::string>& req, int idx);
        ExtractResult DiffIndexExtractor(const torch::Tensor& t, const std::unordered_set<std::string>& req, int plus_idx, int minus_idx);
        ExtractResult PairDiffExtractor(const torch::Tensor& t, const std::unordered_set<std::string>& req, int n_actions);
        ExtractResult QdeltaQmaxCombined(
            const torch::Tensor& t, const std::unordered_set<std::string>& req,
            int n_actions,
            float qdelta_scale,      // ex: 0.5f
            float qmax_threshold);    // ex: 20.0f;
        ExtractResult QdeltaQmaxCombinedAuto(
            const torch::Tensor& t, const std::unordered_set<std::string>& req,
            int n_actions);
        ExtractResult QdeltaQdiffCombinedAuto(
            const torch::Tensor& t, const std::unordered_set<std::string>& req,
            int n_actions,
            int action_index_a,
            int action_index_b);
        ExtractResult BoundaryMaskFromQdiffAuto(
            const torch::Tensor& t, const std::unordered_set<std::string>& req,
            int n_actions,
            int action_index_a,
            int action_index_b);
        ExtractResult BoundaryMaskedQdeltaAuto(
            const torch::Tensor& t, const std::unordered_set<std::string>& req,
            int n_actions,
            int action_index_a,
            int action_index_b);
    }

    /**
        * @brief
        * RL の State を 2軸で sweep し、NN 入力テンソルを生成する。
        * 同時に NN 出力から値抽出を行う。
        * ISweepInputGenerator / ISweepOutputExtractor を統合した処理クラス。
        */
    class StateSweepProcessor : public ISweepInputGenerator, public ISweepOutputExtractor {
    public:
        /**
            * @param env_spec      状態仕様
            * @param x_index       X 軸に割り当てる state flatten index
            * @param y_index       Y 軸に割り当てる state flatten index
            * @param base_state    Sweep ベースとなる state(flatten)。未指定なら zero state。
            * @param x_min_override  X sweep 範囲の最小値（未指定時は StateSpec に従う）
            * @param x_max_override  X sweep 範囲の最大値
            * @param y_min_override  Y sweep 範囲の最小値
            * @param y_max_override  Y sweep 範囲の最大値
            * @param value_extractor 出力抽出関数。デフォルト max(sample)。
            */
        StateSweepProcessor(
            const anet::rl::StateSpec& state_spec,
            int x_index,
            int y_index,
            ValueExtractFunction value_extract_fn = &extractor::MaxExtractor,
            const torch::Device& device = torch::kCUDA,
            std::optional<torch::Tensor> base_state = std::nullopt,
            std::optional<float> x_min_override = std::nullopt,
            std::optional<float> x_max_override = std::nullopt,
            std::optional<float> y_min_override = std::nullopt,
            std::optional<float> y_max_override = std::nullopt
        );

        ~StateSweepProcessor() override = default;
    public:
        void ApplyGridSize(int width, int height) override;
        std::pair<int, int> GetGridSize() const override;
        torch::Tensor BuildInputTensor() override;
        int64_t GetFlattenSize() const override;
        ExtractResult Extract(const torch::Tensor& batched_out,
            const std::unordered_set<std::string>& required_labels) override;
    private:
        const anet::rl::StateSpec state_spec_;
        const torch::Device device_;

        int x_index_;
        int y_index_;

        // Sweep 範囲
        float x_min_;
        float x_max_;
        float y_min_;
        float y_max_;

        bool x_min_overridden_ = false;
        bool x_max_overridden_ = false;
        bool y_min_overridden_ = false;
        bool y_max_overridden_ = false;

        // Base(flatten) state
        torch::Tensor base_flatten_;

        // GridSize（InputGenerator が決定して OutputExtractor に伝える）
        int grid_w_ = 256;
        int grid_h_ = 256;

        // NN 出力 → 値抽出
        ValueExtractFunction value_extract_fn_;
    };

} // namespace anet::rl