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

    inline constexpr char kMultiplierClipped[] = "clipped";
    inline constexpr char kMultiplierUnclipped[] = "unclipped";
    inline constexpr char kStatisticsWhiten[] = "whiten";
    inline constexpr char kStatisticsVarianceFeedback[] = "variance_feedback";
    inline constexpr char kWarmupLearnPredictor[] = "learn_predictor";
    inline constexpr char kWarmupCollectErrors[] = "collect_errors";
    inline constexpr char kErrorAverageCumulative[] = "cumulative";
    inline constexpr char kErrorAverageEma[] = "ema";

    struct RndNoveltyEstimatorConfig : public anet::Config {
        static constexpr float kDefaultMultiplierMax = 5.0f;
        static constexpr int64_t kDefaultEmaSteps = 1000000;
        static constexpr bool kDefaultErrorNormUseCentering = true;
        std::string multiplier_mode = kMultiplierClipped;
        std::string statistics_mode = kStatisticsWhiten;
        float multiplier_max = kDefaultMultiplierMax;
        int64_t warmup_steps = 12800;
        std::string warmup_mode = kWarmupLearnPredictor;
        struct {
            std::string average_mode = kErrorAverageCumulative;
            int64_t ema_steps = kDefaultEmaSteps;
            bool use_centering = kDefaultErrorNormUseCentering;
        } error_norm;
        float learning_rate = 1e-4f;
        float weight_decay = 0.0f;
        double adam_eps = 1e-8;
        struct {
            int output_dim = 128;
            anet::nn::WeightInitConfig init;
        } head;
        ObservationNormalizerConfig obs_norm;
        struct {
            anet::nn::NetworkConfig net;
        } target, predictor;

        RndNoveltyEstimatorConfig(const anet::ConfigData& data, const std::string& prefix);
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
