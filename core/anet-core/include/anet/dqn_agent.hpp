#pragma once

#include <torch/torch.h>
#include <memory>
#include <wx/cmdline.h>

#include "anet/rl.hpp"
#include "anet/heat_map.hpp"
#include "anet/properties.hpp"

namespace anet::rl {

    class DQNAgent : public anet::rl::Agent {
    public:
        DQNAgent(const wxCmdLineParser& cmd_line, anet::Properties* props, anet::rl::Environment& env, int state_dim, int n_actions, torch::Device device);

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
            SelectAction(const torch::Tensor& state, anet::rl::RunMode mode = anet::rl::RunMode::Train) override;

        void Update(const anet::rl::Experience& exprience) override;
        void UpdateBatch(const anet::rl::BatchData&) override; // ReplayBuffer対応
    private:
        void hard_update();
        void soft_update();
        void OptimizeSingle(const anet::rl::Experience& exprence);
        void OptimizeBatch(const std::vector<anet::rl::Experience>& batch);
    private:
        struct QNetImpl;
        // NN
        int n_actions_;
        std::shared_ptr<QNetImpl> policy_net;
        std::shared_ptr<QNetImpl> target_net;
        torch::Device device;

        // パラメータ
        struct Param;
        std::unique_ptr<Param> param_;

        // パラメータ依存オブジェクト
        torch::optim::Adam optimizer;
        anet::rl::ReplayBuffer replay_buffer;

        // 学習変数
        float epsilon;
        float eps_reheat_floor_;
        float tau_;

        int train_step = 0;
        int last_unstable_step_ = 0;
        float eps_boost_ = 1.0f;

        bool  qstd_init_ = false;   // Q値の標準偏差EMA初期化済みフラグ
        float qstd_ema_ = 0.0f;     // Q値の標準偏差のEMA
        float qstd_ema2_ = 0.0f;    // Q値の標準偏差の2乗EMA 

        float unstable_ema_ = 0.0f;    // 連続不安定度
        float uema_alpha_ = 0.0f;      // ln(2)/half_life を前計算
        int post_warmup_steps_ = 0;

        // 統計情報
        float loss_ema = 0.0f;
        bool  loss_ema_init_ = false;
        float grad_norm_clipped_ema = 0.0f;
        float td_clip_ema = 0.0f;
        float action_ema = 0.5f;
        std::unique_ptr<anet::HeatMap> heatmap_visit1_;
        std::unique_ptr<anet::HeatMap> heatmap_visit2_;
        std::unique_ptr<anet::HeatMap> heatmap_td_;
        std::unique_ptr<anet::TimeHistogram> hist_action_;
        std::unique_ptr<anet::TimeHistogram> hist_q_;

    };
}// namespace anet::rl
