// anet/image_cls_agent.hpp

#pragma once

#include <memory>
#include <string>
#include <torch/torch.h>
#include "anet/rl.hpp"
#include "anet/config.hpp"
#include "anet/nn.hpp"
#include "anet/agent.hpp"

namespace anet::rl::img_cls {

    // ======================================================
    // Config
    // ======================================================
    struct ImageClsAgentConfig : public anet::Config {
        double learning_rate = 1e-3;
        double weight_decay = 1e-4;
        double label_smoothing = 0.1;
        double grad_clip_max_norm = 1.0;

        anet::nn::NetworkGraphVizConfig nn_viz;

        ImageClsAgentConfig(const anet::ConfigData& config_data = anet::EmptyConfigData, const std::string& config_prefix = "")
            : anet::Config(config_data, "ImageClsAgent", config_prefix)
        {
            ANET_READ_CONFIG(config_data, learning_rate);
            ANET_READ_CONFIG(config_data, weight_decay);
            ANET_READ_CONFIG(config_data, label_smoothing);
            ANET_READ_CONFIG(config_data, grad_clip_max_norm);

            ANET_READ_CONFIG(config_data, nn_viz.show_param_shapes);
            ANET_READ_CONFIG(config_data, nn_viz.show_param_count);
            ANET_READ_CONFIG(config_data, nn_viz.show_tensor_specs);
            ANET_READ_CONFIG(config_data, nn_viz.show_branch_config);
            ANET_READ_CONFIG(config_data, nn_viz.show_head_info);
            ANET_READ_CONFIG(config_data, nn_viz.layout);
            ANET_READ_CONFIG(config_data, nn_viz.cluster_branches);
            ANET_READ_CONFIG(config_data, nn_viz.float_precision);

        }
    };


    // ======================================================
    // Result (Metrics)
    // ======================================================
    class ImageClsUpdateResult : public anet::rl::BatchUpdateResult {
    public:
        float loss = 0.0f;
        float accuracy = 0.0f;

        std::optional<float> GetScalar(const std::string& key, int64_t index) const override
        {
            if (key == "loss") return loss;
            if (key == "accuracy") return accuracy;
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
    };


    // ======================================================
    // Actor
    // ======================================================
    class ImageClsActor : public anet::rl::Actor {
    public:
        ImageClsActor(
            std::shared_ptr<std::shared_mutex> mutex,
            std::shared_ptr<anet::nn::Network> network,
            anet::rl::RunMode run_mode,
            torch::Device device);

        std::shared_ptr<BatchActionInfo> MakeAction(const StepCounts& step, const anet::rl::BatchState& state) const override;
        void Sync() override {}
    private:
        const anet::rl::RunMode run_mode_;
        std::shared_ptr<std::shared_mutex> mutex_;
        std::shared_ptr<anet::nn::Network> network_;
        torch::Device device_;
    };


    // ======================================================
    // Learner
    // ======================================================
    class ImageClsLearner : public anet::rl::Learner {
    public:
        ImageClsLearner(const ImageClsAgentConfig& config,
            std::shared_ptr<std::shared_mutex> mutex,
            std::shared_ptr<anet::nn::Network> network,
            torch::Device device);

        anet::rl::BatchUpdateResultList UpdateFromBatch(
            const anet::rl::StepCounts& step,
            const anet::rl::BatchExperience& experiences) override;

    private:
        const ImageClsAgentConfig config_;
        std::shared_ptr<std::shared_mutex> mutex_;
        std::shared_ptr<anet::nn::Network> network_;
        std::unique_ptr<torch::optim::Optimizer> optimizer_;
        torch::Device device_;
    };


    // ======================================================
    // Agent
    // ======================================================
    class ImageClsAgent : public anet::rl::AgentBase {
    public:
        ImageClsAgent(const ImageClsAgentConfig& config, const anet::nn::NetworkConfig& network_config,
            const anet::rl::EnvSpec& env_spec, const anet::rl::BatchEnvSpec& batch_env_spec, torch::Device device,
            std::optional<seed_t> seed = std::nullopt);

        std::shared_ptr<anet::rl::Actor> CreateActor(
            const anet::rl::BatchEnvSpec& batch_env_spec,
            anet::rl::RunMode run_mode,
            bool clone_model,
            std::optional<torch::Device> device) const override;

        std::shared_ptr<anet::rl::Learner> CreateLearner() override;

        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override { return std::nullopt; }
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override { return std::nullopt; }
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override { return std::nullopt; }

        std::optional<TensorDictFunction> GetTensorDictFunction(const std::string& key) override { return std::nullopt; }
    private:
        const ImageClsAgentConfig config_;
        std::shared_ptr<std::shared_mutex> mutex_;
        std::shared_ptr<anet::nn::Network> network_;
    };


    // ======================================================
    // Factory
    // ======================================================
    class ImageClsAgentFactory : public anet::rl::AgentFactory {
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

