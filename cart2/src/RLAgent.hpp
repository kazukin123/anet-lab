#pragma once

#include <torch/torch.h>
#include <memory>
#include "anet/rl.hpp"
#include "anet/HeatMap.hpp"

struct QNetImpl;
using QNet = std::shared_ptr<QNetImpl>;  // ← 明示的に定義（マクロ代替）

class RLAgent : public anet::rl::Agent {
public:
    RLAgent(anet::rl::Environment& env, int state_dim, int n_actions, torch::Device device);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
        SelectAction(const torch::Tensor& state, anet::rl::RunMode mode = anet::rl::RunMode::Train) override;

    void Update(const anet::rl::Experience& exprience) override;
    void UpdateBatch(const anet::rl::BatchData&) override; // ReplayBuffer対応
private:
    void hard_update();
    void soft_update(float tau);
    void OptimizeSingle(const anet::rl::Experience& exprence);
    void OptimizeBatch(const std::vector<anet::rl::Experience>& batch);
private:
    // NN
    int n_actions_;
    QNet policy_net;
    QNet target_net;
    torch::Device device;

    // パラメータ
    struct Param;
    std::unique_ptr<Param> param_;

    // パラメータ依存オブジェクト
    torch::optim::Adam optimizer;
    anet::rl::ReplayBuffer replay_buffer;

    // 実行状態
    float epsilon;
    int train_step;

    // 統計情報
    float loss_ema = 0.0f;
    float grad_norm_clipped_ema = 0.0f;
    float td_clip_ema = 0.0f;
    std::unique_ptr<anet::HeatMap> heatmap_visit1_;
    std::unique_ptr<anet::HeatMap> heatmap_visit2_;
    std::unique_ptr<anet::HeatMap> heatmap_td_;
    std::unique_ptr<anet::TimeHeatMap> thmap_reward_;

    int thmap_reward_count_ = 0;
};

