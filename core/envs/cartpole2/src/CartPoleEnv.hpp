// CartPoleEnv.hpp

#pragma once

#include <torch/torch.h>
#include "anet/random.hpp"
#include "anet/config.hpp"
#include "anet/env.hpp"


struct CartPoleEnvConfig : public anet::Config {
    int limit_step = 200;

    CartPoleEnvConfig(const anet::ConfigData& config_data = anet::EmptyConfigData, const std::string& config_prefix = "")
        : anet::Config(config_data, "CartPoleEnv", config_prefix) {
        ANET_READ_CONFIG(config_data, limit_step);
    }
};

class CartPoleEnv : public anet::rl::SingleDiscreteEnvBase, public anet::RandomHolder {
public:
    CartPoleEnv(
        const CartPoleEnvConfig& config,
        const torch::Device& device,
        const std::string& name,
        const std::optional<anet::seed_t> seed = std::nullopt,
        anet::rl::RunMode run_mode = anet::rl::RunMode::Train);

    anet::rl::EnvSpec GetSpec() const override;
    std::shared_ptr<const anet::rl::SingleResetResult> Reset() override;
    std::shared_ptr<const anet::rl::SingleStepResult> Step(int64_t action) override;

    std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override { return std::nullopt; }
    std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override { return std::nullopt; }

    float get_x() const { return x_; }
    float get_theta() const { return theta_; }
    float get_x_dot() const { return x_dot_; }
    float get_theta_dot() const { return theta_dot_; }
private:
    CartPoleEnvConfig config_;

    float x_, x_dot_, theta_, theta_dot_;
    bool done_ = false, truncated_ = false, episode_start_ = true;
    int step_count_ = 0;
    torch::TensorOptions obs_opt_;
};

class CartPoleEnvFactory : public anet::rl::SingleDiscreteEnvFactory {
public:
    CartPoleEnvFactory();

    std::string GetTargetEnvClassId() const override { return "CartPoleEnv"; }

    std::shared_ptr<anet::rl::SingleDiscreteEnv> CreateSingleEnv(
        const anet::ConfigData& config_data,
        const torch::Device& device, const std::string& name,
        std::optional<anet::seed_t> seed = std::nullopt,
        anet::rl::RunMode run_mode = anet::rl::RunMode::Train,
        const std::string& config_prefix = "") override;
};
