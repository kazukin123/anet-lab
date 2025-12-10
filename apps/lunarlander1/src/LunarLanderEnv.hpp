#pragma once
#include <torch/torch.h>
#include "anet/random.hpp"
#include "anet/config.hpp"
#include "anet/rl.hpp"

/// LunarLanderEnv実装（1環境固定）

struct LunarLanderEnvConfig : public anet::Config {
    int limit_step = 200;

    LunarLanderEnvConfig(const anet::ConfigData& config_data = anet::EmptyConfigData) : anet::Config(config_data, "LunarLanderEnv") {
        ANET_READ_CONFIG(config_data, limit_step);
    }
};

class LunarLanderEnv : public anet::rl::SingleDiscreteEnv, public anet::RandomHolder {
public:
    LunarLanderEnv(
        const LunarLanderEnvConfig& config,
        const torch::Device& device,
        const std::optional<anet::seed_t> seed = std::nullopt);

    anet::rl::EnvSpec GetSpec() const override;
    anet::rl::SingleState Reset(anet::rl::RunMode mode = anet::rl::RunMode::Train) override;
    anet::rl::SingleStepResult Step(int64_t action, anet::rl::RunMode mode = anet::rl::RunMode::Train) override;

    float get_x() const { return x_; }
    float get_theta() const { return theta_; }
    float get_x_dot() const { return x_dot_; }
    float get_theta_dot() const { return theta_dot_; }
private:
    LunarLanderEnvConfig config_;

    float x_, x_dot_, theta_, theta_dot_;
    bool done_ = false, truncated_ = false, episode_start_ = true;
    int step_count_ = 0;
    torch::TensorOptions obs_opt_;
};

class LunarLanderEnvFactory : public anet::rl::SingleDiscreteEnvFactory {
public:
    LunarLanderEnvFactory();

    std::string GetTargetEnvClassId() const override { return "LunarLanderEnv"; }

    std::unique_ptr<anet::rl::SingleDiscreteEnv> CreateSingleEnv(
        const anet::ConfigData& config_data,
        const torch::Device& device, std::optional<anet::seed_t> seed = std::nullopt) override;
};

