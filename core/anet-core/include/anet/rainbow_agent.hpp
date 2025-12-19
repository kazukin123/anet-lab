#pragma once

#include <memory>
#include <torch/torch.h>

#include "anet/config.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/rl.hpp"
#include "anet/agent.hpp"

namespace anet::rl {

    struct RainbowAgentConfig : public anet::Config {
        int nn_init_mode = 1;  // 0=default、1=XavierUniform、2=HeNormal
        int nn_hidden1 = 128;
        int nn_hidden2 = 128;

        float alpha = 1e-3f;   ///<  学習率 1e-3 3e-3 1e-4 1e-4 3e-4 5e-4
        float gamma = 0.99f;   ///<  0.99f; 0.995f      γが高いほど「長期安定」を目指す
        float eps_max = 1.00f;
        float eps_min = 0.05f;    ///< 0.1f 0.05f
        int eps_decay_step = 100000;
        //int eps_sigmoid_step = -1;
        float soft_update_tau = 0.01f;  ///<  1.0f 0.004f  0.01f 0.005f;   // 大きいとターゲットネットワークからの反映が早くなる。小さいと遅く滑らかになる。0.005→半減期138step
        int hard_update_interval = -1;
        bool use_grad_clip = true;
        float grad_clip_tau = 30.0f;
        bool use_td_clip = true;
        float td_clip_value = 4.0f;
        //int eps_zero_step = -1;// 120000;

        int replay_capacity = 10000;
        int replay_batch_size = 128;
        int update_warmup_steps = 1000;
        int update_interval = 10;

        int n_step = 3;

        float per_alpha = 0.6f;            ///< 優先度の反映度合い (0:uniform, 1:full)
        float per_beta_start = 0.4f;       ///< IS重みの補正度合い (初期値)
        float per_beta_end = 1.0f;         ///< IS重みの補正度合い (収束値)
        int   per_beta_step = 100000;      ///< betaを収束値まで線形変化させるステップ数
        float per_eps = 1e-6f;             ///< 優先度加算用微小値
        float per_initial_priority = 1.0f; ///< 新規データの初期優先度

        bool use_double_dqn = true;   ///< Double DQN 有効化フラグ
        bool use_dueling_net = true;  ///< Dueling Net 有効化フラグ
        bool use_n_step = true;       ///< N-STEPを使用するか
        bool use_per = true;          ///< PERを使用するか

        explicit RainbowAgentConfig(const ConfigData& config_data = EmptyConfigData) : anet::Config(config_data, "RainbowAgent") {
            ANET_READ_CONFIG(config_data, nn_init_mode);
            ANET_READ_CONFIG(config_data, nn_hidden1);
            ANET_READ_CONFIG(config_data, nn_hidden2);
            ANET_READ_CONFIG(config_data, alpha);
            ANET_READ_CONFIG(config_data, gamma);
            ANET_READ_CONFIG(config_data, eps_max);
            ANET_READ_CONFIG(config_data, eps_min);
            ANET_READ_CONFIG(config_data, eps_decay_step);
            ANET_READ_CONFIG(config_data, soft_update_tau);
            ANET_READ_CONFIG(config_data, hard_update_interval);
            ANET_READ_CONFIG(config_data, use_grad_clip);
            ANET_READ_CONFIG(config_data, grad_clip_tau);
            ANET_READ_CONFIG(config_data, use_td_clip);
            ANET_READ_CONFIG(config_data, td_clip_value);
            ANET_READ_CONFIG(config_data, replay_capacity);
            ANET_READ_CONFIG(config_data, replay_batch_size);
            ANET_READ_CONFIG(config_data, update_warmup_steps);
            ANET_READ_CONFIG(config_data, update_interval);
            ANET_READ_CONFIG(config_data, n_step);
            ANET_READ_CONFIG(config_data, per_alpha);
            ANET_READ_CONFIG(config_data, per_beta_start);
            ANET_READ_CONFIG(config_data, per_beta_end);
            ANET_READ_CONFIG(config_data, per_beta_step);
            ANET_READ_CONFIG(config_data, per_eps);
            ANET_READ_CONFIG(config_data, per_initial_priority);
            ANET_READ_CONFIG(config_data, use_double_dqn);
            ANET_READ_CONFIG(config_data, use_dueling_net);
            ANET_READ_CONFIG(config_data, use_n_step);
            ANET_READ_CONFIG(config_data, use_per);
        }
    };

    class RainbowAgent: public anet::rl::FlatStateAgent<RainbowAgentConfig>, public std::enable_shared_from_this<RainbowAgent> {
    public:
        RainbowAgent(
            const RainbowAgentConfig& config,
            const anet::rl::BatchEnvSpec& batc_env_spec, const anet::rl::EnvSpec& env_spec, const torch::Device& device,
            std::shared_ptr<anet::rl::Notifier> notifier = nullptr,
            std::optional<seed_t> seed = std::nullopt);

        BatchActionInfo MakeAction(const StepCounts& step, const BatchState& state, RunMode mode = RunMode::Train) const override;
        std::shared_ptr<const anet::rl::BatchUpdateResult> UpdateFromBatch(
            const StepCounts& step, const anet::rl::BatchExperience& exprience, const anet::rl::Runner& trainer) override;
    public:
        std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key) override;

        std::optional<float> GetScalar(const std::string& key, int index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int index = -1) const override;
    private:
        class BatchUpdateResult;
    private:
        struct RuntimeVars;         ///< Agent内部変数
        class Network;              ///< NN

        class ActionPolicy;         ///< 行動選択アルゴリズム

        class Learner;              ///< 学習アルゴリズム
        class TDLearner;
        //class QRLearner;
    private:
        std::unique_ptr<RuntimeVars> vars_;
        std::unique_ptr<Network> network_;
        //std::unique_ptr<ReplayBuffer> replay_buffer_;
    private:
        std::shared_ptr<ActionPolicy> action_policy_;
        std::shared_ptr<Learner> learner_;
    };

    class RainbowAgentFactory : public anet::rl::AgentFactory {
    public:
        RainbowAgentFactory() { }

        std::shared_ptr<Agent> CreateAgent(
            const EnvSpec& env_spec,
            const BatchEnvSpec& batch_env_spec,
            const torch::Device& device,
            const anet::ConfigData& config_data = EmptyConfigData,
            std::shared_ptr<anet::rl::Notifier> notifier = nullptr,
            std::optional<anet::seed_t> seed = std::nullopt
        ) const override;

        std::string GetTargetAgentClassId() const override { return "RainbowAgent"; }
    };

}// namespace anet::rl
