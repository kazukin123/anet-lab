#pragma once
#include <torch/torch.h>

#include "anet/rl.hpp"

/// CartPoleŠÂ‹«À‘•i1ŠÂ‹«ŒÅ’èj
class CartPoleEnv : public anet::rl::BatchEnvironment, public anet::RandomHolder {
public:
    CartPoleEnv(anet::RandomGenerator* rnd = nullptr);

    anet::rl::EnvSpec GetSpec() const override;
    anet::rl::BatchState Reset(anet::rl::RunMode mode = anet::rl::RunMode::Train) override;
    anet::rl::BatchStepResult DoStep(const torch::Tensor& action, anet::rl::RunMode mode = anet::rl::RunMode::Train) override;   //  state, reward, done
    anet::rl::BatchState GetState() const override;

    float get_x() const { return x_; }
    float get_theta() const { return theta_; }
    float get_x_dot() const { return x_dot_; }
    float get_theta_dot() const { return theta_dot_; }
private:
    float x_, x_dot_, theta_, theta_dot_;
    bool done_ = false, truncated_ = false, episode_start_ = true;
    int step_count_ = 0;
};
