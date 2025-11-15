#pragma once

#include <torch/torch.h>
#include <memory>
#include <wx/cmdline.h>

#include "anet/rl.hpp"
#include "anet/heat_map.hpp"
#include "anet/config.hpp"

namespace anet::rl {

    struct DQNAgentConfig : public anet::Config {
        float alpha = 1e-3f;   // 学習率 1e-3 3e-3 1e-4 1e-4 3e-4 5e-4
        float gamma = 0.99f;   // 0.99f; 0.995f      γが高いほど「長期安定」を目指す
        float eps_max = 1.00f;
        float eps_min = 0.05f;    //0.1f 0.05f
        float eps_decay_step = 100000;
        float softupdate_tau = 0.01f;// 1.0f 0.004f  0.01f 0.005f;   // 大きいとターゲットネットワークからの反映が早くなる。小さいと遅く滑らかになる。0.005→半減期138step
        int hardupdate_step = 2000;// -1 5000; //200 500 1000
        bool use_grad_clip = true;
        float grad_clip_tau = 30.0f;   // 10~40 1f 5f 10f
        bool use_td_clip = true;
        float td_clip_value = 4.0f;
        int eps_zero_step = -1;// 120000;
        bool use_double_dqn = true;   // Double DQN 有効化フラグ（trueで有効）

        bool use_replay_buffer = false;   // ← ON/OFF切替
        int replay_capacity = 50000;
        int replay_batch_size = 64;
        int replay_warmup_steps = 1000;
        int replay_update_interval = 4;

        int heatmap_log_image_interval = 100;
        int heatmap_log_sweep_interval = 100;
        int heatmap_log_hist_interval = 10;

        bool use_as_dqn = false;             //Adaptive Stabilized DQN (AS-DQN)
        float qstd_alpha = 0.01f;           //Q値 std の EWMA 平滑率
        float q_z_threshold = 3.0f;         //z-score 崩壊判定閾値
        float q_cv_threshold = 0.5f;        //CV 崩壊判定閾値
        float q_niqr_threshold = 0.6f;      //NIQR 崩壊判定閾値
        float eps_boost_max = 2.0f;               //ε ブースト上限倍率
        int   eps_boost_half_life_hit = 300;      // 崩壊中、ε ブーストが2倍になるまでのstep数
        int   eps_boost_half_life_recover = 8000; // 安定後、ε ブーストの自然減衰半減期
        float eps_gain = 0.15f;
        float eps_reheat_floor = 0.20f;
        float tau_min = 0.0005f;            //τ の下限
        float tau_max = 0.001f;             //τ の上限
        float tau_half_life_hit = 200;      // 崩壊中、τが半減するまでのstep数
        float tau_half_life_recover = 4000; // 安定後、τが2倍に戻るまでのstep数
        int tau_recover_delay = 1000;       // 1000step安定していたら回復開始
        float act_bias_threshold = 0.85f;  // 行動偏り閾値 (|left_ratio - right_ratio| > 0.85 → 崩壊)

        bool  use_unstable_ema = false;    // 連続崩壊制御を使うか（切替用）
        float uema_half_life = 2000.0f;   // 半減期 [step]。ln2/半減期 が EMA係数
        float uema_u0 = 0.12f;            // 作動し始めの基準（無次元）
        float uema_k = 12.0f;            // シグモイドの傾き
        float uema_g1 = 0.02f;            // εブースト倍率のゲイン（相対）
        float uema_g2 = 0.05f;            // ε再加熱floorのゲイン（絶対上乗せ）
        float uema_g3 = 0.10f;            // τ減衰のゲイン（exp(-g3*s)）
        float uema_s_clip = 0.2f;
        float eps_reheat_base = 0.10f;    // 停滞時の軽い再加熱ベース
        float eps_reheat_half_life = 1000;
        float unstable_ema_s_threshold = 0.0f; // 連続崩壊度の閾値

        DQNAgentConfig() {}
        DQNAgentConfig(const ConfigData& configData) : anet::Config(configData, "agent", "DQNAgent") {
            ANET_APPLY_CONFIG(configData, alpha);
            ANET_APPLY_CONFIG(configData, gamma);
            ANET_APPLY_CONFIG(configData, eps_max);
            ANET_APPLY_CONFIG(configData, eps_min);
            ANET_APPLY_CONFIG(configData, eps_decay_step);
            ANET_APPLY_CONFIG(configData, softupdate_tau);
            ANET_APPLY_CONFIG(configData, hardupdate_step);
            ANET_APPLY_CONFIG(configData, use_grad_clip);
            ANET_APPLY_CONFIG(configData, grad_clip_tau);
            ANET_APPLY_CONFIG(configData, use_td_clip);
            ANET_APPLY_CONFIG(configData, td_clip_value);
            ANET_APPLY_CONFIG(configData, eps_zero_step);
            ANET_APPLY_CONFIG(configData, use_double_dqn);
            ANET_APPLY_CONFIG(configData, use_replay_buffer);
            ANET_APPLY_CONFIG(configData, replay_capacity);
            ANET_APPLY_CONFIG(configData, replay_batch_size);
            ANET_APPLY_CONFIG(configData, replay_warmup_steps);
            ANET_APPLY_CONFIG(configData, replay_update_interval);
            ANET_APPLY_CONFIG(configData, heatmap_log_image_interval);
            ANET_APPLY_CONFIG(configData, heatmap_log_sweep_interval);
            ANET_APPLY_CONFIG(configData, heatmap_log_hist_interval);
            ANET_APPLY_CONFIG(configData, use_as_dqn);
            ANET_APPLY_CONFIG(configData, qstd_alpha);
            ANET_APPLY_CONFIG(configData, q_z_threshold);
            ANET_APPLY_CONFIG(configData, q_cv_threshold);
            ANET_APPLY_CONFIG(configData, q_niqr_threshold);
            ANET_APPLY_CONFIG(configData, eps_boost_max);
            ANET_APPLY_CONFIG(configData, eps_boost_half_life_hit);
            ANET_APPLY_CONFIG(configData, eps_boost_half_life_recover);
            ANET_APPLY_CONFIG(configData, eps_gain);
            ANET_APPLY_CONFIG(configData, eps_reheat_floor);
            ANET_APPLY_CONFIG(configData, eps_reheat_half_life);
            ANET_APPLY_CONFIG(configData, tau_min);
            ANET_APPLY_CONFIG(configData, tau_max);
            ANET_APPLY_CONFIG(configData, tau_half_life_hit);
            ANET_APPLY_CONFIG(configData, tau_half_life_recover);
            ANET_APPLY_CONFIG(configData, tau_recover_delay);
            ANET_APPLY_CONFIG(configData, act_bias_threshold);
            ANET_APPLY_CONFIG(configData, use_unstable_ema);
            ANET_APPLY_CONFIG(configData, uema_half_life);
            ANET_APPLY_CONFIG(configData, uema_u0);
            ANET_APPLY_CONFIG(configData, uema_k);
            ANET_APPLY_CONFIG(configData, uema_g1);
            ANET_APPLY_CONFIG(configData, uema_g2);
            ANET_APPLY_CONFIG(configData, uema_g3);
            ANET_APPLY_CONFIG(configData, uema_s_clip);
            ANET_APPLY_CONFIG(configData, eps_reheat_base);
            ANET_APPLY_CONFIG(configData, unstable_ema_s_threshold);
        }
    };
    class DQNAgent : public anet::rl::Agent {
    public:
        DQNAgent(const DQNAgentConfig& config, anet::rl::Environment& env, int state_dim, int n_actions, torch::Device device);

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
        // 設定
        DQNAgentConfig config_;

        // NN
        struct QNetImpl;
        int n_actions_;
        std::shared_ptr<QNetImpl> policy_net;
        std::shared_ptr<QNetImpl> target_net;
        torch::Device device;

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
