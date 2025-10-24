#pragma once
#include <torch/torch.h>

#include "anet/rl.hpp"

class CartPoleEnv : public anet::rl::Environment {
public:
    CartPoleEnv();

    torch::Tensor Reset(anet::rl::RunMode mode = anet::rl::RunMode::Train);
    anet::rl::EnvResponse DoStep(const torch::Tensor& action, anet::rl::RunMode mode = anet::rl::RunMode::Train);   //  state, reward, done
    torch::Tensor GetState() const;

    float get_x() const { return x; }
    float get_theta() const { return theta; }
    float get_x_dot() const { return x_dot; }
    float get_theta_dot() const { return theta_dot; }

    float get_total_reward() const { return total_reward; }
private:
    float x, x_dot, theta, theta_dot;
    float total_reward = 0.0f;
    int step_count = 0;
};
