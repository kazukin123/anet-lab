#pragma once
#include <torch/torch.h>

struct StepResult {
    torch::Tensor next_state;
    float reward;
    bool done;
    bool truncated = false;  // ŠÔ§ŒÀ‚â‹­§I—¹‚È‚Ç‚Ì‘Å‚¿Ø‚èƒtƒ‰ƒO
};

class Environment {
public:
    virtual torch::Tensor Reset() = 0;
    virtual StepResult DoStep(const torch::Tensor& action) = 0;   //  state, reward, done
    virtual torch::Tensor GetState() const = 0;
    virtual ~Environment() = default;
};

class CartPoleEnv : public Environment {
public:
    CartPoleEnv();

    torch::Tensor Reset();
    StepResult DoStep(const torch::Tensor& action);   //  state, reward, done
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
