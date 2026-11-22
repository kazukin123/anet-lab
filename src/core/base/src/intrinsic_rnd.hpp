#pragma once

#include <cmath>
#include <functional>
#include <map>
#include <vector>
#include "anet/intrinsic.hpp"
#include "anet/nn.hpp"
#include "anet/scaler.hpp"
#include "scaler_impl.hpp"

namespace anet::rl::intrinsic {

    inline constexpr char kStatisticsWhiten[] = "whiten";
    inline constexpr char kStatisticsVarianceFeedback[] = "variance_feedback";
    inline constexpr char kErrorStandardize[] = "standardize";
    inline constexpr char kErrorNone[] = "none";
    inline constexpr char kErrorAverageCumulative[] = "cumulative";
    inline constexpr char kErrorAverageEma[] = "ema";

    struct RndNoveltyEstimatorConfig : public anet::Config {
        static constexpr float kDefaultMultiplierMax = 5.0f;
        static constexpr int64_t kDefaultWarmupSteps = 12800;
        static constexpr int64_t kDefaultEmaSteps = 1000000;
        struct {
            std::string mode = kStatisticsWhiten;
            struct {
                ObservationNormalizerConfig obs_norm;
            } whiten;
            struct {
                struct {
                    float clip_range = 5.0f;
                } obs_norm;
                bool use_clipping = false;
                float multiplier_max = kDefaultMultiplierMax;
            } variance_feedback;
        } statistics;
        struct {
            std::string mode = kErrorStandardize;
            struct {
                std::string average_mode = kErrorAverageCumulative;
                int64_t ema_steps = kDefaultEmaSteps;
                bool use_centering = true;
                int64_t warmup_steps = kDefaultWarmupSteps;
                bool use_clipping = true;
                float multiplier_max = kDefaultMultiplierMax;
            } standardize;
            struct {
                int64_t warmup_steps = kDefaultWarmupSteps;
                bool use_clipping = false;
                float multiplier_max = kDefaultMultiplierMax;
            } none;
        } error_norm;
        float learning_rate = 1e-4f;
        float weight_decay = 0.0f;
        double adam_eps = 1e-8;
        struct {
            int output_dim = 128;
            anet::nn::WeightInitConfig init;
        } head;
        struct {
            anet::nn::NetworkConfig net;
        } target;
        struct {
            anet::nn::NetworkConfig net;
            int64_t warmup_steps = 0;
        } predictor;

        RndNoveltyEstimatorConfig(const anet::ConfigData& data, const std::string& prefix);
        bool UsesRawError() const
        {
            return statistics.mode == kStatisticsWhiten && error_norm.mode == kErrorNone;
        }
    };

    class RndNoveltyEstimator final : public LifelongNoveltyEstimator {
    public:
        RndNoveltyEstimator(const RndNoveltyEstimatorConfig& config,
            const anet::TensorSpecMap& observation_spec, const torch::Device& device,
            const anet::SeedMaker& seed_maker);
        ~RndNoveltyEstimator() override;
        torch::Tensor Estimate(const anet::TensorDict& observations) override;
        LifelongNoveltyEstimateResult EstimateWithoutUpdate(const anet::TensorDict& observations) const override;
        std::unique_ptr<LifelongNoveltyEstimator> Clone() const override;
        bool Learn(const anet::TensorDict& observations) override;
        std::map<std::string, float> Diagnostics() const override;
        bool SuppressesBonus() const override;
        std::map<std::string, std::unique_ptr<anet::graphviz::GraphViz>> MakeNetworkGraphs(
            const anet::nn::NetworkGraphVizConfig& config) const override;
        int64_t Save(anet::OutputArchive& archive) const override;
        int64_t Load(anet::InputArchive& archive) override;
    private:
        explicit RndNoveltyEstimator(const RndNoveltyEstimator& source);

        // 誤差の統計の口。方式は構築時に選び、更新と保存は選んだ型へ委ねる。
        class RndErrorStatistics {
        public:
            virtual ~RndErrorStatistics() = default;
        public:
            virtual void Update(const torch::Tensor& errors) = 0;
            virtual bool HasData() const = 0;
            virtual float Mean() const = 0;
            virtual float Std() const = 0;
            virtual int64_t Save(anet::OutputArchive& archive) const = 0;
            virtual int64_t Load(anet::InputArchive& archive) = 0;
            virtual std::unique_ptr<RndErrorStatistics> Clone() const = 0;
        };
        class CumulativeErrorStatistics;
        class EmaErrorStatistics;

        enum class ForwardUse { Estimate, ReadOnly, Learn };
        struct ScoreResult {
            torch::Tensor target_features;
            torch::Tensor prediction_errors;
            torch::Tensor errors;
            torch::Tensor alpha;
        };
        using ScoreFunction = std::function<ScoreResult(const anet::TensorDict&)>;
        class RndStatistics;
        class WhitenStatistics;
        class VarianceFeedbackStatistics;

        template<ForwardUse use>
        ScoreResult ComputePredictionErrors(const anet::TensorDict& normalized) const;
        float TargetFeatureVariance() const;
        std::pair<bool, float> GetMultiplierClipping() const;
        void LogWarmup(const char* key, int64_t limit, int64_t previous_count) const;
        const RndNoveltyEstimatorConfig config_;
        const torch::Device device_;
        std::shared_ptr<anet::nn::Network> target_;
        std::shared_ptr<anet::nn::Network> predictor_;
        std::unique_ptr<torch::optim::AdamW> optimizer_;
        std::unique_ptr<RndStatistics> statistics_;
        RunningMeanStd target_feature_stats_;
        int64_t score_count_ = 0;
        std::map<std::string, float> diagnostics_;
    };

} // namespace anet::rl::intrinsic
