#pragma once
#include <torch/torch.h>
#include "anet/random.hpp"
#include "anet/rl.hpp"

/// CartPole環境実装（1環境固定）
class CartPoleEnv : public anet::rl::SingleDiscreteEnv, public anet::RandomHolder {
public:
    CartPoleEnv(std::shared_ptr<anet::RandomGenerator> rnd = nullptr);

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
};

class CartPoleEnvFactory : public anet::rl::SingleDiscreteEnvFactory, public anet::RandomHolder {
public:
    CartPoleEnvFactory(std::shared_ptr<anet::RandomGenerator> rnd = nullptr);
    std::unique_ptr<anet::rl::SingleDiscreteEnv> Create() override;
};
