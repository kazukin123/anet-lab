// anet/image_cls_agent.hpp

#pragma once

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <torch/torch.h>
#include "anet/diag.hpp"
#include "anet/rl.hpp"
#include "anet/config.hpp"
#include "anet/schedule.hpp"
#include "anet/nn.hpp"
#include "anet/agent.hpp"
#include "anet/random.hpp"
#include "anet/serialize.hpp"

namespace anet::rl::img_cls {

    // ======================================================
    // Config
    // ======================================================
    struct ImageClsAgentConfig : public anet::Config {
        anet::ProfiledValueConfig<double> learning_rate;
        double weight_decay = 1e-4;
        bool use_fused_optimizer = true; ///< ATen fused AdamWを使うか。falseで従来AdamWへ戻す。
        double label_smoothing = 0.1;
        double grad_clip_max_norm = 1.0;
        int64_t learn_log_interval = 0;
        std::string auto_load_file;
        struct PlasticityConfig {
            std::string feature_key;
        } plasticity;

        struct MixupConfig {
            bool enabled = false;
            double mixup_alpha = 0.2;
            double cutmix_alpha = 1.0;
            double prob = 1.0;
            double switch_prob = 0.5;
        };

        struct Bf16Config {
            bool enabled = false;
            bool learner = true;
            bool actor = false;
        };

        MixupConfig mixup;
        Bf16Config bf16;

        anet::nn::NetworkGraphVizConfig nn_viz;

        ImageClsAgentConfig(const anet::ConfigData& config_data = anet::EmptyConfigData, const std::string& config_prefix = "")
            : anet::Config(config_data, "ImageClsAgent", config_prefix)
        {
            learning_rate.value = 1e-3;
            ANET_READ_CONFIG(config_data, learning_rate);
            ANET_READ_CONFIG(config_data, weight_decay);
            ANET_READ_CONFIG(config_data, use_fused_optimizer);
            ANET_READ_CONFIG(config_data, label_smoothing);
            ANET_READ_CONFIG(config_data, grad_clip_max_norm);
            ANET_READ_CONFIG(config_data, learn_log_interval);
            ANET_READ_CONFIG(config_data, auto_load_file);
            ANET_READ_CONFIG(config_data, plasticity.feature_key);

            ANET_READ_CONFIG(config_data, mixup.enabled);
            ANET_READ_CONFIG(config_data, mixup.mixup_alpha);
            ANET_READ_CONFIG(config_data, mixup.cutmix_alpha);
            ANET_READ_CONFIG(config_data, mixup.prob);
            ANET_READ_CONFIG(config_data, mixup.switch_prob);

            ANET_READ_CONFIG(config_data, bf16.enabled);
            ANET_READ_CONFIG(config_data, bf16.learner);
            ANET_READ_CONFIG(config_data, bf16.actor);

            ANET_READ_CONFIG(config_data, nn_viz.show_param_shapes);
            ANET_READ_CONFIG(config_data, nn_viz.show_param_count);
            ANET_READ_CONFIG(config_data, nn_viz.show_tensor_specs);
            ANET_READ_CONFIG(config_data, nn_viz.show_branch_config);
            ANET_READ_CONFIG(config_data, nn_viz.show_head_info);
            ANET_READ_CONFIG(config_data, nn_viz.layout);
            ANET_READ_CONFIG(config_data, nn_viz.cluster_branches);
            ANET_READ_CONFIG(config_data, nn_viz.float_precision);

            Validate();
        }

    private:
        void Validate() const
        {
            ValidateNonNegative("ImageClsAgent.mixup.mixup_alpha", mixup.mixup_alpha);
            ValidateNonNegative("ImageClsAgent.mixup.cutmix_alpha", mixup.cutmix_alpha);
            ValidateUnitInterval("ImageClsAgent.mixup.prob", mixup.prob);
            ValidateUnitInterval("ImageClsAgent.mixup.switch_prob", mixup.switch_prob);
            if (learn_log_interval < 0) {
                ANET_SYSTEM_ERROR("Invalid config ImageClsAgent.learn_log_interval=" << learn_log_interval
                    << ". Expected >= 0.");
            }
        }

        static void ValidateNonNegative(const char* key, double value)
        {
            if (value < 0.0) {
                ANET_SYSTEM_ERROR("Invalid config " << key << "=" << value << ". Expected >= 0.");
            }
        }

        static void ValidateUnitInterval(const char* key, double value)
        {
            if (value < 0.0 || value > 1.0) {
                ANET_SYSTEM_ERROR("Invalid config " << key << "=" << value << ". Expected range [0, 1].");
            }
        }
    };


    // ======================================================
    // Result (Metrics)
    // ======================================================
    class ImageClsUpdateResult final : public anet::rl::BatchUpdateResult {
    public:
        torch::Tensor loss;
        torch::Tensor accuracy;
        torch::Tensor target_prob_mix_norm;
        torch::Tensor accuracy_either;
        torch::Tensor pred_max_prob;
        torch::Tensor same_class_pair_ratio;
        torch::Tensor plasticity_features;
        torch::Tensor plasticity_weight_norms;
        std::shared_ptr<anet::nn::Network> plasticity_network;
        anet::nn::PlasticityMetricRequest plasticity_request;

        std::optional<float> GetScalar(const std::string& key, int64_t index) const override
        {
            // 必要になって初めて scalar tensor を CPU 同期し、同じ event 内の再読みに備えて cache する。
            if (key == "loss") return GetCachedScalar(loss, loss_cache_);
            if (key == "accuracy") return GetCachedScalar(accuracy, accuracy_cache_);
            if (key == "target_prob_mix_norm") return GetCachedScalar(target_prob_mix_norm, target_prob_mix_norm_cache_);
            if (key == "accuracy_either") return GetCachedScalar(accuracy_either, accuracy_either_cache_);
            if (key == "pred_max_prob") return GetCachedScalar(pred_max_prob, pred_max_prob_cache_);
            if (key == "same_class_pair_ratio") return GetCachedScalar(same_class_pair_ratio, same_class_pair_ratio_cache_);
            int64_t weight_norm_index = -1;
            if (key == "plasticity_weight_norm_feature") weight_norm_index = 0;
            else if (key == "plasticity_weight_norm_readout") weight_norm_index = 1;
            else if (key == "plasticity_weight_norm_feature_effective") weight_norm_index = 2;
            else if (key == "plasticity_weight_norm_readout_effective") weight_norm_index = 3;
            else if (key == "plasticity_spectral_sigma_feature") weight_norm_index = 4;
            else if (key == "plasticity_spectral_sigma_readout") weight_norm_index = 5;
            if (weight_norm_index >= 0) {
                if (!plasticity_weight_norms.defined()) {
                    return std::numeric_limits<float>::quiet_NaN();
                }
                if (!plasticity_weight_norms_cpu_.defined()) {
                    plasticity_weight_norms_cpu_ = plasticity_weight_norms.cpu();
                }
                if (plasticity_weight_norms_cpu_.numel() != 7) {
                    ANET_SYSTEM_ERROR("ImageCls plasticity weight norm pack has invalid size="
                        << plasticity_weight_norms_cpu_.numel() << " expected=7.");
                }
                ValidateSpectralNormSentinel(plasticity_weight_norms_cpu_[6].item<float>());
                return plasticity_weight_norms_cpu_[weight_norm_index].item<float>();
            }
            if (key.starts_with("plasticity_")) {
                const auto metric = anet::nn::ParsePlasticityMetricSuffix(
                    key.substr(std::string("plasticity_").size()));
                if (!metric.has_value()) return std::nullopt;
                if (!plasticity_request.Contains(*metric)) {
                    return std::numeric_limits<float>::quiet_NaN();
                }
                if (!plasticity_features.defined()) return std::numeric_limits<float>::quiet_NaN();
                if (!plasticity_metrics_cache_.has_value()) {
                    plasticity_metrics_cache_ = anet::nn::ComputePlasticityMetrics(
                        plasticity_features, plasticity_request);
                }
                const auto value = plasticity_metrics_cache_->Get(*metric);
                if (!value.has_value()) {
                    ANET_SYSTEM_ERROR("ImageClsUpdateResult plasticity cache is missing a requested metric.");
                }
                return value;
            }
            return std::nullopt;
        }

        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index) const override
        {
            return std::nullopt;
        }

        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index) const override
        {
            return std::nullopt;
        }

    private:
        void ValidateSpectralNormSentinel(float invalid_count) const
        {
            if (invalid_count < 1.0f) return;
            if (!plasticity_network) {
                ANET_SYSTEM_ERROR("ImageCls spectral normalization sentinel failed: invalid_count="
                    << invalid_count << " network handle is unavailable.");
            }
            for (const auto& entry : plasticity_network->GetSpectralNormEntries()) {
                const auto sigma = anet::nn::ComputeSpectralSigma(
                    entry.weight.reshape({ entry.weight.size(0), -1 }), entry.u, entry.v).item<float>();
                const bool valid = std::isfinite(sigma)
                    && (entry.mode == anet::nn::WeightNormMode::kSpectralCap || sigma > 0.0f);
                if (!valid) {
                    ANET_SYSTEM_ERROR("ImageCls spectral normalization is invalid: layer="
                        << entry.name << " sigma=" << sigma << ".");
                }
            }
            ANET_SYSTEM_ERROR("ImageCls spectral normalization sentinel mismatch: invalid_count="
                << invalid_count << ".");
        }

        static std::optional<float> GetCachedScalar(const torch::Tensor& tensor, std::optional<float>& cache)
        {
            if (!tensor.defined()) {
                return std::numeric_limits<float>::quiet_NaN();
            }
            if (!cache.has_value()) {
                cache = tensor.item<float>();
            }
            return cache;
        }

        mutable std::optional<float> loss_cache_;
        mutable std::optional<float> accuracy_cache_;
        mutable std::optional<float> target_prob_mix_norm_cache_;
        mutable std::optional<float> accuracy_either_cache_;
        mutable std::optional<float> pred_max_prob_cache_;
        mutable std::optional<float> same_class_pair_ratio_cache_;
        mutable torch::Tensor plasticity_weight_norms_cpu_;
        mutable std::optional<anet::nn::PlasticityMetrics> plasticity_metrics_cache_;
    };


    // ======================================================
    // Actor
    // ======================================================
    class ImageClsActor final : public anet::rl::Actor {
    public:
        ImageClsActor(
            const ImageClsAgentConfig& config,
            std::shared_ptr<std::shared_mutex> mutex,
            std::shared_ptr<anet::nn::Network> network,
            anet::rl::RunMode run_mode,
            torch::Device device,
            std::shared_ptr<anet::nn::Network> src_network = nullptr);

        std::shared_ptr<BatchActionInfo> MakeAction(const StepCounts& step, const anet::rl::BatchState& state) const override;
        void Sync() override;
    private:
        const ImageClsAgentConfig config_;
        const anet::rl::RunMode run_mode_;
        std::shared_ptr<std::shared_mutex> mutex_;
        std::shared_ptr<anet::nn::Network> network_;
        std::shared_ptr<anet::nn::Network> src_network_;
        torch::Device device_;
    };


    // ======================================================
    // Learner
    // ======================================================
    class ImageClsLearner final : public anet::rl::Learner, public anet::RandomHolder {
    public:
        ImageClsLearner(const ImageClsAgentConfig& config,
            std::shared_ptr<std::shared_mutex> mutex,
            std::shared_ptr<anet::nn::Network> network,
            std::shared_ptr<anet::ProfiledValue<double>> learning_rate,
            torch::Device device,
            std::optional<seed_t> seed = std::nullopt);

        anet::rl::BatchUpdateResultList UpdateFromBatch(
            const anet::rl::StepCounts& step,
            const anet::rl::BatchExperience& experiences) override;
        void ConfigureScalarMetricSubscriptions(
            const std::vector<ScalarMetricSubscription>& subscriptions);

        int64_t Save(anet::OutputArchive& archive) const;
        int64_t Load(anet::InputArchive& archive);
    private:
        struct CutMixBox;
        struct MixResult;
        struct PlasticityDemandEntry {
            anet::nn::PlasticityMetric metric;
            anet::IntervalGate gate;

            PlasticityDemandEntry(anet::nn::PlasticityMetric metric, int interval)
                : metric(metric), gate(static_cast<uint64_t>(interval)) {}
        };
    private:
        double SampleBeta(double alpha);
        bool Bernoulli(double probability);
        CutMixBox SampleCutMixBox(int64_t height, int64_t width, double lambda);

        MixResult ApplyMix(anet::TensorDict& obs, const torch::Tensor& targets);
        anet::nn::PlasticityMetricRequest MakePlasticityRequest(uint64_t step);
    private:
        const ImageClsAgentConfig config_;
        std::shared_ptr<std::shared_mutex> mutex_;
        std::shared_ptr<anet::nn::Network> network_;
        std::shared_ptr<anet::ProfiledValue<double>> learning_rate_;
        std::unique_ptr<torch::optim::Optimizer> optimizer_;
        torch::Device device_;
        bool plasticity_enabled_ = false;
        int plasticity_interval_ = 1;
        std::vector<PlasticityDemandEntry> plasticity_demands_;
        bool plasticity_weight_norm_enabled_ = false;
        int plasticity_weight_norm_interval_ = 1;
    };


    // ======================================================
    // Agent
    // ======================================================
    class ImageClsAgent final : public anet::rl::AgentBase {
    public:
        ImageClsAgent(const ImageClsAgentConfig& config, const anet::nn::NetworkConfig& network_config,
            const anet::rl::EnvSpec& env_spec, const anet::rl::BatchEnvSpec& batch_env_spec, torch::Device device,
            std::optional<seed_t> seed = std::nullopt);

        std::shared_ptr<anet::rl::Actor> CreateActor(
            const anet::rl::BatchEnvSpec& batch_env_spec,
            const anet::rl::EnvSpec& env_spec,
            anet::rl::RunMode run_mode,
            std::optional<bool> clone_model_override = std::nullopt,
            std::optional<torch::Device> device = std::nullopt) const override;

        std::shared_ptr<anet::rl::Learner> CreateLearner() override;
        void ConfigureScalarMetricSubscriptions(
            const std::vector<ScalarMetricSubscription>& subscriptions) override;

        int64_t Save(anet::OutputArchive& archive) const override;

        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override { return std::nullopt; }
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override { return std::nullopt; }

        std::optional<TensorDictFunction> GetTensorDictFunction(const std::string& key) override { return std::nullopt; }
    private:
        void LoadNetwork(const std::string& filename);
    private:
        const ImageClsAgentConfig config_;
        std::shared_ptr<std::shared_mutex> mutex_;
        std::shared_ptr<anet::nn::Network> network_;
        std::shared_ptr<anet::ProfiledValue<double>> learning_rate_;
        std::shared_ptr<ImageClsLearner> learner_;
    };


    // ======================================================
    // Factory
    // ======================================================
    class ImageClsAgentFactory final : public anet::rl::AgentFactory {
    public:
        std::shared_ptr<Agent> CreateAgent(
            const EnvSpec& env_spec,
            const BatchEnvSpec& batch_env_spec,
            const torch::Device& device,
            const anet::ConfigData& config_data = EmptyConfigData,
            std::shared_ptr<anet::rl::Notifier> notifier = nullptr,
            std::optional<anet::seed_t> seed = std::nullopt) const override;

        std::string GetTargetAgentClassId() const override { return "ImageClsAgent"; }
    };

} // namespace anet::rl::img_cls
