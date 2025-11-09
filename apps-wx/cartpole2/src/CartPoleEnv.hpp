#pragma once
#include <torch/torch.h>

#include "anet/rl.hpp"

class CartPoleEnv : public anet::rl::Environment {
public:
    CartPoleEnv();

    anet::rl::StateSpaceInfo GetStateSpaceInfo() const override;

    torch::Tensor Reset(anet::rl::RunMode mode = anet::rl::RunMode::Train) override;
    anet::rl::EnvResponse DoStep(const torch::Tensor& action, anet::rl::RunMode mode = anet::rl::RunMode::Train) override;   //  state, reward, done
    torch::Tensor GetState() const override;

    float get_x() const { return x; }
    float get_theta() const { return theta; }
    float get_x_dot() const { return x_dot; }
    float get_theta_dot() const { return theta_dot; }
private:
    float x, x_dot, theta, theta_dot;
    int step_count = 0;
};
