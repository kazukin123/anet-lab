#pragma once

#include <random>
#include <vector>
#include "anet/intrinsic.hpp"
#include "anet/nn.hpp"
#include "anet/scaler.hpp"
#include "scaler_impl.hpp"

namespace anet::rl::intrinsic {

    struct RndNoveltyEstimatorConfig : public anet::Config {
        int output_dim = 128;
        float multiplier_max = 5.0f;
        int64_t warmup_steps = 100;
        float train_fraction = 0.25f;
        float learning_rate = 1e-4f;
        float weight_decay = 0.0f;
        ObservationNormalizerConfig obs_norm;
        anet::nn::NetworkConfig net;

        RndNoveltyEstimatorConfig(const anet::ConfigData& data, const std::string& prefix);
    };

    class RndNoveltyEstimator final : public LifelongNoveltyEstimator {
    public:
        RndNoveltyEstimator(const RndNoveltyEstimatorConfig& config,
            const anet::TensorSpecMap& observation_spec, const torch::Device& device,
            const anet::SeedMaker& seed_maker);
        torch::Tensor Multiplier(const anet::TensorDict& observations) override;
        void Update(const anet::TensorDict& observations) override;
        std::map<std::string, float> Diagnostics() const override;
        int64_t Save(anet::OutputArchive& archive) const override;
        int64_t Load(anet::InputArchive& archive) override;
    private:
        const RndNoveltyEstimatorConfig config_;
        const torch::Device device_;
        std::vector<std::string> byte_observation_keys_;
        std::shared_ptr<anet::nn::Network> target_;
        std::shared_ptr<anet::nn::Network> predictor_;
        std::unique_ptr<torch::optim::AdamW> optimizer_;
        std::mt19937_64 train_random_;
        torch::Tensor pending_errors_;
        torch::Tensor pending_errors_cpu_;
        std::shared_ptr<ObservationNormalizer> normalizer_;
        RunningMeanStd error_stats_;
        int64_t score_count_ = 0;
        std::map<std::string, float> diagnostics_;
    };

} // namespace anet::rl::intrinsic
