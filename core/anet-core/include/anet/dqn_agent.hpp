#pragma once

#include <memory>
#include <torch/torch.h>

#include "anet/config.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/rl.hpp"

namespace anet::rl {

    struct DQNAgentConfig : public anet::Config {
        int nn_init_mode = 1;  // 0=default、1=XavierUniform、2=HeNormal

        float alpha = 1e-3f;   ///<  学習率 1e-3 3e-3 1e-4 1e-4 3e-4 5e-4
        float gamma = 0.99f;   ///<  0.99f; 0.995f      γが高いほど「長期安定」を目指す
        float eps_max = 1.00f;
        float eps_min = 0.05f;    ///< 0.1f 0.05f
        int eps_decay_step = 100000;
        int eps_sigmoid_step = -1;
        float softupdate_tau = 0.01f;  ///<  1.0f 0.004f  0.01f 0.005f;   // 大きいとターゲットネットワークからの反映が早くなる。小さいと遅く滑らかになる。0.005→半減期138step
        int hardupdate_interval = -1;
        bool use_grad_clip = true;
        float grad_clip_tau = 30.0f;
        bool use_td_clip = true;
        float td_clip_value = 4.0f;
        int eps_zero_step = -1;// 120000;

        bool use_double_dqn = true;   ///< Double DQN 有効化フラグ（trueで有効）

        bool use_replay_buffer = false;   ///< ReplayBuffer ON/OFF切替
        int replay_capacity = 50000;
        int replay_batch_size = 64;
        int replay_warmup_steps = 1000;
        int replay_update_interval = 10;

        /// @todo AS-DQN系メトリクスの計測か出力を無効化する設定を追加

        bool use_as_dqn = false;            ///< Adaptive Stabilized DQN (AS-DQN)
        float q_z_threshold = 3.0f;         ///< z-score 崩壊判定閾値
        float q_cv_threshold = 0.5f;        ///< CV 崩壊判定閾値
        float q_niqr_threshold = 0.6f;      ///< NIQR 崩壊判定閾値
        float eps_boost_max = 2.0f;               ///< ε ブースト上限倍率
        int   eps_boost_half_life_hit = 300;      ///<  崩壊中、ε ブーストが2倍になるまでのstep数
        int   eps_boost_half_life_recover = 8000; ///<  安定後、ε ブーストの自然減衰半減期
        float eps_gain = 0.15f;
        float eps_reheat_floor = 0.20f;
        float tau_min = 0.0005f;            ///< τ の下限
        float tau_max = 0.001f;             ///< τ の上限
        float tau_half_life_hit = 200;      ///< 崩壊中、τが半減するまでのstep数
        float tau_half_life_recover = 4000; ///< 安定後、τが2倍に戻るまでのstep数
        int tau_recover_delay = 1000;       ///< 1000step安定していたら回復開始
        float act_bias_threshold = 0.85f;   ///< 行動偏り閾値 (|left_ratio - right_ratio| > 0.85 → 崩壊)

        bool  use_unstable_ema = false;    ///< 連続崩壊制御を使うか（切替用）
        float uema_half_life = 2000.0f;    ///< 半減期 [step]。ln2/半減期 が EMA係数
        float uema_u0 = 0.12f;             ///< 作動し始めの基準（無次元）
        float uema_k = 12.0f;              ///< シグモイドの傾き
        float uema_g1 = 0.02f;             ///< εブースト倍率のゲイン（相対）
        float uema_g2 = 0.05f;             ///< ε再加熱floorのゲイン（絶対上乗せ）
        float uema_g3 = 0.10f;             ///< τ減衰のゲイン（exp(-g3*s)）
        float uema_s_clip = 0.2f;
        float eps_reheat_base = 0.10f;     ///<  停滞時の軽い再加熱ベース
        float eps_reheat_half_life = 1000;
        float unstable_ema_s_threshold = 0.0f; ///< 連続崩壊度の閾値

        DQNAgentConfig() : anet::Config("DQNAgent") {}
        DQNAgentConfig(const ConfigData& config_data) : anet::Config(config_data, "DQNAgent") {
            ANET_READ_CONFIG(config_data, nn_init_mode);
            ANET_READ_CONFIG(config_data, alpha);
            ANET_READ_CONFIG(config_data, gamma);
            ANET_READ_CONFIG(config_data, eps_max);
            ANET_READ_CONFIG(config_data, eps_min);
            ANET_READ_CONFIG(config_data, eps_decay_step);
            ANET_READ_CONFIG(config_data, eps_sigmoid_step);
            ANET_READ_CONFIG(config_data, softupdate_tau);
            ANET_READ_CONFIG(config_data, hardupdate_interval);
            ANET_READ_CONFIG(config_data, use_grad_clip);
            ANET_READ_CONFIG(config_data, grad_clip_tau);
            ANET_READ_CONFIG(config_data, use_td_clip);
            ANET_READ_CONFIG(config_data, td_clip_value);
            ANET_READ_CONFIG(config_data, eps_zero_step);
            ANET_READ_CONFIG(config_data, use_double_dqn);
            ANET_READ_CONFIG(config_data, use_replay_buffer);
            ANET_READ_CONFIG(config_data, replay_capacity);
            ANET_READ_CONFIG(config_data, replay_batch_size);
            ANET_READ_CONFIG(config_data, replay_warmup_steps);
            ANET_READ_CONFIG(config_data, replay_update_interval);
            ANET_READ_CONFIG(config_data, use_as_dqn);
            ANET_READ_CONFIG(config_data, q_z_threshold);
            ANET_READ_CONFIG(config_data, q_cv_threshold);
            ANET_READ_CONFIG(config_data, q_niqr_threshold);
            ANET_READ_CONFIG(config_data, eps_boost_max);
            ANET_READ_CONFIG(config_data, eps_boost_half_life_hit);
            ANET_READ_CONFIG(config_data, eps_boost_half_life_recover);
            ANET_READ_CONFIG(config_data, eps_gain);
            ANET_READ_CONFIG(config_data, eps_reheat_floor);
            ANET_READ_CONFIG(config_data, eps_reheat_half_life);
            ANET_READ_CONFIG(config_data, tau_min);
            ANET_READ_CONFIG(config_data, tau_max);
            ANET_READ_CONFIG(config_data, tau_half_life_hit);
            ANET_READ_CONFIG(config_data, tau_half_life_recover);
            ANET_READ_CONFIG(config_data, tau_recover_delay);
            ANET_READ_CONFIG(config_data, act_bias_threshold);
            ANET_READ_CONFIG(config_data, use_unstable_ema);
            ANET_READ_CONFIG(config_data, uema_half_life);
            ANET_READ_CONFIG(config_data, uema_u0);
            ANET_READ_CONFIG(config_data, uema_k);
            ANET_READ_CONFIG(config_data, uema_g1);
            ANET_READ_CONFIG(config_data, uema_g2);
            ANET_READ_CONFIG(config_data, uema_g3);
            ANET_READ_CONFIG(config_data, uema_s_clip);
            ANET_READ_CONFIG(config_data, eps_reheat_base);
            ANET_READ_CONFIG(config_data, unstable_ema_s_threshold);
        }
    };

    class DQNAgent : public anet::rl::StepBasedAgent<DQNAgentConfig> {
    public:
        DQNAgent(
            const DQNAgentConfig& config,
            anet::rl::BatchEnvSpec batc_env_spec, anet::rl::EnvSpec& env_spec,
            torch::Device device, std::optional<seed_t> seed = std::nullopt);

        anet::rl::BatchActionInfo MakeAction(const StepCounts& step, const anet::rl::BatchState& state, anet::rl::RunMode mode = anet::rl::RunMode::Train) override;
        std::shared_ptr<const anet::rl::BatchUpdateResult> UpdateFromBatch(
            const StepCounts& step, const anet::rl::BatchExperience& exprience) override;

        anet::TensorFunction GetTensorFunction(const std::string& key) const override;
        std::optional<float> GetScalar(const std::string& key) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key) const override;
    private:
        int state_count_;
        int n_actions_;
		int batch_size_;
    private:
        class BatchUpdateResult;
    private:
        struct RuntimeVars;         ///< Agent内部変数
        struct QNetImpl;            ///< NN
        class RuntimeVarsUpdater;   ///< 内部変数制御
        class ActionDecider;        ///< 行動選択（Policy相当）
        class ReplayScheduler;      ///< 学習更新タイミング管理
        class TargetUpdater;        ///< target_net の同期実行
        class StabilityMonitor;     ///< メトリクス情報管理
        class StabilityController;  ///< 安定制御
    private:
        // Resource（Agentが管理すべき内部データ）
        std::unique_ptr<RuntimeVars> vars_;
        std::shared_ptr<QNetImpl> policy_net_;
        std::shared_ptr<QNetImpl> target_net_;
        std::unique_ptr<torch::optim::Adam> optimizer_;
    private:
        std::unique_ptr<anet::rl::ReplayBuffer> replay_buffer_;
        std::unique_ptr<RuntimeVarsUpdater> vars_updater_;
        std::unique_ptr<ActionDecider> action_decider_;
        std::unique_ptr<ReplayScheduler> replay_scheduler_;
        std::unique_ptr<TargetUpdater> target_updater_;
        std::unique_ptr<StabilityMonitor> stability_monitor_;
        std::unique_ptr<StabilityController> stability_controller_;
    };
}// namespace anet::rl
