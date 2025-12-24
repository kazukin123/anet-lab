// anet/agent.hpp
#pragma once

#include <memory>
#include <optional>
#include <mutex>
#include "anet/rl.hpp"
#include "anet/config.hpp"

namespace anet::rl {

    // 環境のステップに同期して更新する Agent 基底クラス
    class FlatStateAgent : public Agent, public anet::RandomHolder {
    public:
        FlatStateAgent(torch::Device device,
            std::shared_ptr<anet::rl::Notifier> notifier,
            const BatchEnvSpec& batch_env_spec,
            const EnvSpec& env_spec,
            std::optional<seed_t> seed = std::nullopt)
            : RandomHolder(seed), notifier_(notifier),device_(device)
            , state_dim_(env_spec.state_spec.CalcFlattenDim())
            , n_actions_(env_spec.action_spec.GetNumActions())
            , batch_size_(batch_env_spec.batch_size)
        {
            mutex_ = std::make_shared<std::shared_mutex>();
        }

        virtual ~FlatStateAgent() = default;
    protected:
        std::shared_ptr<std::shared_mutex> mutex_;
        const torch::Device device_;
        const std::shared_ptr<anet::rl::Notifier> notifier_;
        int state_dim_;
        int n_actions_;
        int batch_size_;

    };


    // =============================================================
    // DQN
    // =============================================================

    namespace dqn {
        struct RuntimeVars;         ///< Agent内部変数
        class Network;              ///< NN
        class ActionPolicy;         ///< 行動選択アルゴリズム
        class Learner;              ///< 学習アルゴリズム
        class TDLearner;
        class QRLearner;
        class BatchUpdateResult;

        struct QNetConfig {
            int nn_init_mode = 1;  // 0=default、1=XavierUniform、2=HeNormal
            int nn_hidden1 = 128;
            int nn_hidden2 = 128;

            int num_quantiles = 51;
        };

        struct NetworkConfig {
            float soft_update_tau = 0.01f;
            int hard_update_interval = -1;
        };

        struct LearnerConfig {
            float alpha = 1e-3f;   ///<  学習率 1e-3 3e-3 1e-4 1e-4 3e-4 5e-4
            float gamma = 0.99f;   ///<  0.99f; 0.995f      γが高いほど「長期安定」を目指す
            float eps_max = 1.00f;
            float eps_min = 0.05f;    ///< 0.1f 0.05f
            int eps_decay_step = 100000;
            //int eps_sigmoid_step = -1;

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
            bool use_per_prio_clip = false;    ///< 優先度をクリッピングするか
            float per_prio_clip_value = 50.0f; ///< 優先度の上限値

            bool use_double_dqn = true;   ///< Double DQN 有効化フラグ
            bool use_n_step = true;       ///< N-STEPを使用するか
            bool use_per = true;          ///< PERを使用するか

            int num_quantiles = 51;         ///< 分位数 N (デフォルト51)
            float quantile_huber_kappa = 1.0f;///< Huber Loss の閾値 kappa
        };
    }

    // =============================================================

    struct DefaultAgentFactoryConfig : public anet::Config
    {
        std::string class_id;
        int device_type = 1;   ///< 0=cpu 1=cuda
        int device_index = -1; ///< GPU index -1=current device

        DefaultAgentFactoryConfig(const ConfigData& config_data = EmptyConfigData)
            : anet::Config(config_data, "agent")
        {
            ANET_READ_CONFIG(config_data, class_id);
            ANET_READ_CONFIG(config_data, device_type);
            ANET_READ_CONFIG(config_data, device_index);
        }
    };

    class DefaultAgentFactory {
    public:
        DefaultAgentFactory(
            const DefaultAgentFactoryConfig& config,
            const EnvSpec& env_spec,
			const BatchEnvSpec& batch_env_spec,
            const anet::ConfigData& config_data = anet::EmptyConfigData,
            std::optional<seed_t> seed = std::nullopt);

        std::shared_ptr<Agent> CreateAgent(std::shared_ptr<anet::rl::Notifier> notifier = nullptr) const;
        torch::Device GetDevice() const { return device_; }
    private:
        DefaultAgentFactoryConfig config_;
        anet::ConfigData config_data_;
        EnvSpec env_spec_;
		BatchEnvSpec batch_env_spec_;
        std::optional<seed_t> seed_;
        torch::Device device_;
    };

    // =============================================================

    class AgentRepository {
    public:
        static AgentRepository& Instance() {
            static AgentRepository inst;
            return inst;
        }

        void Register(std::shared_ptr<AgentFactory> factory);
        std::shared_ptr<AgentFactory> GetAgentFactory(const std::string& id) const;
    private:
        AgentRepository() = default;

        mutable std::mutex mtx_;
        std::unordered_map<std::string, std::shared_ptr<AgentFactory>> factories_;
    };

    template<typename T, class... Args>
    inline void RegisterAgentFactory(Args&&... args)
    {
        auto factory = std::make_shared<T>(std::forward<Args>(args)...);
        AgentRepository::Instance().Register(factory);
    }

} // namespace anet::rl

  // =============================================================

#define ANET_REGISTER_AGENT_FACTORY(FactoryType) \
    namespace { \
        struct FactoryType##AutoRegister { \
            FactoryType##AutoRegister() { \
                anet::rl::RegistAgentFactory(std::make_shared<FactoryType>()); \
            } \
        }; \
        static FactoryType##AutoRegister global_##FactoryType##_auto_register; \
    }
