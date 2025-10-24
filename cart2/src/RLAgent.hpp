#pragma once
#include <torch/torch.h>
#include <memory>
#include "anet/rl.hpp"

struct QNetImpl;
using QNet = std::shared_ptr<QNetImpl>;  // ← 明示的に定義（マクロ代替）

class RLAgent : public anet::rl::Agent {
public:
    RLAgent(int state_dim, int n_actions, torch::Device device);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
        SelectAction(const torch::Tensor& state, anet::rl::RunMode mode = anet::rl::RunMode::Train) override;

    void Update(const anet::rl::Experience& exprience) override;
    void UpdateBatch(const anet::rl::BatchData&) override {} // 未対応

    float epsilon;
    float loss_ema;

private:
    void hard_update();
    void soft_update(float tau);

    QNet policy_net;
    QNet target_net;
    torch::optim::Adam optimizer;
    torch::Device device;
    int n_actions_;
    int train_step;
    float grad_norm_clipped_ema = 0.0f;
    float td_clip_ema = 0.0f;
};
