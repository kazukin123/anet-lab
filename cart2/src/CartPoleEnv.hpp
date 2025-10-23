#pragma once
#include <torch/torch.h>


class Env {
public:
    virtual torch::Tensor reset() = 0;
    virtual std::tuple<torch::Tensor, float, bool> step(int action) = 0;   //  state, reward, done
    virtual torch::Tensor get_state() const = 0;
    virtual ~Env() = default;
};

class CartPoleEnv : public Env {
public:
    CartPoleEnv();

    torch::Tensor reset();
    std::tuple<torch::Tensor, float, bool> step(int action);   //  state, reward, done
    torch::Tensor get_state() const;

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
