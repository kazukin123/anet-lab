#pragma once
#include <torch/torch.h>

class CartPoleEnv {
public:
    CartPoleEnv();

    at::Tensor reset();  // © –ß‚è’l‚ğ at::Tensor ‚É
    std::tuple<at::Tensor, float, bool> step(int action);

    float get_x() const { return x; }
    float get_theta() const { return theta; }
    float get_total_reward() const { return total_reward; }
    torch::Tensor get_state() const;

private:
    float x, x_dot, theta, theta_dot;
    float total_reward = 0.0f;
    int step_count = 0;
    const int max_steps = 500;  // I—¹ğŒ
};
