#pragma once
#include <torch/torch.h>
#include <memory>

struct QNetImpl;
using QNet = std::shared_ptr<QNetImpl>;  // ← 明示的に定義（マクロ代替）

class RLAgent {
public:
    RLAgent(int state_dim, int n_actions, torch::Device device);

    torch::Tensor select_action(torch::Tensor state);
    void update(const torch::Tensor& state,
        const torch::Tensor& next_state,
        float reward, bool done);

    float epsilon;
    float latest_loss;

private:
    QNet policy_net;
    QNet target_net;
    torch::optim::Adam optimizer;
    torch::Device device;
    int n_actions_;
    int step_count;
};
