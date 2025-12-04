#pragma once
#include <torch/torch.h>
#include "anet/random.hpp"
#include "anet/rl.hpp"
#include "anet/env.hpp"

/// CartPole環境実装（1環境固定）
class CartPoleEnv : public anet::rl::SingleDiscreteEnv, public anet::RandomHolder {
public:
    CartPoleEnv(
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
    float x_, x_dot_, theta_, theta_dot_;
    bool done_ = false, truncated_ = false, episode_start_ = true;
    int step_count_ = 0;
    torch::TensorOptions obs_opt_;
};

class CartPoleEnvFactory : public anet::rl::SingleDiscreteEnvFactory {
public:
    CartPoleEnvFactory();

    std::string GetTargetEnvClassName() const { return "CartPole"; }

    std::unique_ptr<anet::rl::SingleDiscreteEnv> CreateSingleEnv(
        const torch::Device& device, std::optional<anet::seed_t> seed = std::nullopt) override;
};

