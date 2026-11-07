#pragma once

#include <cmath>
#include <map>
#include <optional>
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

    struct RndNoveltyEstimatorConfig : public anet::Config {
        int output_dim = 128;
        static constexpr float kDefaultMultiplierMax = 5.0f;
        std::string multiplier_mode = kMultiplierClipped;
        std::string statistics_mode = kStatisticsWhiten;
        float multiplier_max = kDefaultMultiplierMax;
        int64_t warmup_steps = 12800;
        float learning_rate = 1e-4f;
        float weight_decay = 0.0f;
        double adam_eps = 1e-8;
        struct {
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
        torch::Tensor Estimate(const anet::TensorDict& observations) override;
        bool Learn(const anet::TensorDict& observations) override;
        std::map<std::string, float> Diagnostics() const override;
        int64_t Save(anet::OutputArchive& archive) const override;
        int64_t Load(anet::InputArchive& archive) override;
    private:
        // 観測・誤差の再帰統計。共有の正規化部品とは独立。
        class VarianceFeedbackStatistics {
        public:
            explicit VarianceFeedbackStatistics(const anet::TensorSpecMap& observation_spec);
            anet::TensorDict Normalize(const anet::TensorDict& observations, float clip_range) const;
            std::pair<anet::TensorDict, torch::Tensor> NormalizeAndUpdate(
                const anet::TensorDict& observations, float clip_range);
            torch::Tensor UpdateErrors(const torch::Tensor& errors);
            double ErrorMean() const { return err_m_; }
            double ErrorStd() const { return std::sqrt(err_v_); }
            int64_t Count() const { return n_; }
            int64_t Save(anet::OutputArchive& archive) const;
            int64_t Load(anet::InputArchive& archive);
        private:
            void ValidateObservations(const anet::TensorDict& observations) const;
            std::map<std::string, torch::Tensor> obs_m_, obs_v_;
            double err_m_ = 0.0;
            double err_v_ = 1.0;
            int64_t n_ = 0;
        };

        float TargetFeatureVariance() const;
        const RndNoveltyEstimatorConfig config_;
        const torch::Device device_;
        std::vector<std::string> byte_observation_keys_;
        std::shared_ptr<anet::nn::Network> target_;
        std::shared_ptr<anet::nn::Network> predictor_;
        std::unique_ptr<torch::optim::AdamW> optimizer_;
        std::shared_ptr<ObservationNormalizer> normalizer_;
        RunningMeanStd error_stats_;
        RunningMeanStd target_feature_stats_;
        std::optional<VarianceFeedbackStatistics> feedback_stats_;
        int64_t score_count_ = 0;
        std::map<std::string, float> diagnostics_;
    };

} // namespace anet::rl::intrinsic
