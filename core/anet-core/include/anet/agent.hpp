// anet/agent.hpp
#pragma once

#include <memory>
#include <optional>
#include <mutex>
#include "anet/rl.hpp"
#include "anet/config.hpp"
#include "anet/scaler.hpp"
#include "anet/schedule.hpp"

namespace anet::rl {


    // ----------------------------------------------------------------------
    // ActionContext
    // ----------------------------------------------------------------------

    class ActionContext : public anet::RandomHolder {
    public:
        ActionContext(RunMode mode, std::optional<seed_t> seed = std::nullopt)
            : RandomHolder(seed)
            , run_mode_(mode)
        {
        }

        RunMode GetRunMode() const { return run_mode_; }

        /// @return 加工されたObservation
        virtual anet::TensorDict PushObservation(const anet::rl::BatchState& state) = 0;
        virtual void Reset() = 0;

        virtual ~ActionContext() = default;
    private:
        RunMode run_mode_;
    };


    // ----------------------------------------------------------------------
    // DefaultActionContext
    // ----------------------------------------------------------------------

    /// 加工を行わず、State内のobsをそのまま通過させるActionContext 
    class DefaultActionContext : public ActionContext {
    public:
        DefaultActionContext(RunMode run_mode, std::optional<seed_t> seed = std::nullopt, std::optional<torch::Device> device = std::nullopt)
            : ActionContext(run_mode, seed)
            , device_(device)
        {
        }

        anet::TensorDict PushObservation(const BatchState& state) override
        {
            ///< そのまま obs を返す
            if (device_.has_value()){
                return state.obs.To(device_.value());
            } else {
                return state.obs;
            }
        }
        void Reset() override { }

		virtual ~DefaultActionContext() = default;
    private:
        std::optional<torch::Device> device_;
    };


    // ----------------------------------------------------------------------
    // AgentBase
    // ----------------------------------------------------------------------

    // 環境のステップに同期して更新する Agent 基底クラス
    class AgentBase : public Agent, public ModuleBase, public anet::RandomHolder {
    public:
        AgentBase(torch::Device device,
            const BatchEnvSpec& batch_env_spec,
            const EnvSpec& env_spec,
            std::optional<seed_t> seed = std::nullopt);

        torch::Device GetDevice() const override { return device_; }
        virtual ~AgentBase() = default;
    protected:
        std::shared_ptr<anet::RandomGenerator> GetRandomGenerator(RunMode mode) const;
    protected:
        std::shared_ptr<std::shared_mutex> mutex_;
        const torch::Device device_;
        const EnvSpec env_spec_;
        int n_actions_;
        int num_envs_;
    private:
        mutable std::unordered_map<RunMode, std::shared_ptr<anet::RandomGenerator>> run_mode_rngs_;
        mutable std::mutex rng_mutex_;
        seed_t action_context_seed_;
    };


    // =============================================================
    // DQN
    // =============================================================

    namespace dqn {
        class Actor;
        struct RuntimeVars;         ///< Agent内部変数
        class NetworkModel;              ///< NN
        class ActionPolicy;         ///< 行動選択アルゴリズム
        class Learner;              ///< 学習アルゴリズム
        class TDLearner;
        class QRLearner;
        class BatchUpdateResult;

        struct NetworkModelConfig {
            float soft_update_tau = 0.01f;
            int hard_update_interval = -1;
        };

        static constexpr int kActionPolicyType_EpsilonGreedy = 0;
        static constexpr int kActionPolicyType_UQE = 1;
        static constexpr int kActionPolicyType_ThompsonSampling = 2;

        static constexpr const char* kActionPolicyTypeStr_EpsilonGreedy = "EpsilonGreedy";
        static constexpr const char* kActionPolicyTypeStr_UQE = "UQE";
        static constexpr const char* kActionPolicyTypeStr_ThompsonSampling = "ThompsonSampling";

        struct TauRuleConfig {
            std::string sample_mode = "random";
            int num_taus = 32;
        };

        struct FullDistributionQueryConfig {
            bool enabled = false;
            TauRuleConfig tau_rule{
                .sample_mode = "fixed",
                .num_taus = 32,
            };
        };

        struct ActionPolicyConfig {
            // Poilcy選択
            std::string policy_type = "EpsilonGreedy"; ///< "EpsilonGreedy", "UQE", "ThompsonSampling", "Greedy"

            // ==========================================
            // 純粋な EpsilonGreedy 用設定
            // ==========================================
            float eps_start = 1.0f;
            float eps_end = 0.05f;
            uint64_t eps_decay_steps = 100000;
            bool use_spatial_exploration = false;
            std::string spatial_scale_type = "log";

            // ==========================================
            // UQE / ThompsonSampling 用設定
            // ==========================================
            float uqe_tau_start = 0.5f;
            float uqe_tau_end = 0.5f;
            uint64_t uqe_tau_decay_steps = 0;
            bool uqe_use_tail_mean = true;

            float uqe_eps_start = 0.0f;
            float uqe_eps_end = 0.0f;
            uint64_t uqe_eps_decay_steps = 0;

            // ==========================================
            // 推論（Forward）時の精度設定
            // ==========================================
            bool use_amp = false;
            bool use_amp_bf16 = false;

            TauRuleConfig tau_rule;
            FullDistributionQueryConfig full_distribution_query;
            std::string quantile_mode = "none";
        };

        struct TrainActorConfig {
            bool clone_model = false;
            anet::ProfiledValueConfig<step_t> sync_interval{
                .type = "constant",
                .value = 400,
                .min_value = 1,
            };
        };

        struct StuckerConfig {
            bool use_stacker = false;
            int stack_count = 4;
			std::vector<std::string> stack_keys; // obs内のどのキーをスタックするか。空なら全てスタック
        };

        struct LearnerConfig {
            float alpha = 1e-3f;         ///< 学習率 1e-3 3e-3 1e-4 1e-4 3e-4 5e-4
            float weight_decay = 1e-2f;  ///< AdamWの重み減衰率
            float adam_eps = 1e-5;       ///< ゼロ除算防止項。LibTorchのデフォルトは1e-8。大きくすることで小さな勾配の変化に敏感になりすぎるのを防ぎ学習をマイルドに。
            bool use_fused_optimizer = true; ///< ATen fused AdamWを使うか。falseで従来AdamWへ戻す。

            float gamma = 0.99f;         ///< 0.99f; 0.995f      γが高いほど「長期安定」を目指す

            bool use_grad_clip = true;
            float grad_clip_tau = 30.0f;
            bool use_td_clip = true;
            float td_clip_value = 4.0f;

            int replay_capacity = 10000;
            int replay_batch_size = 128;
            int update_warmup_steps = 1000;
            int update_interval = 2;         ///< 何ステップに1回Updateするか。replay_ratioが正なら使われない。
            float replay_ratio = -1;         ///< 環境1ステップあたり平均何回の勾配更新を行うか。num_envsに依存しない。負数ではuppdate_intervalのみ使う
            bool use_rb_prefetch = false;    ///< ReplayBuffer Sample + H2Dを1バッチ先読みし、armed後のPushを遅延投入するか

            int n_step = 3;

            float per_alpha = 0.6f;            ///< 優先度の反映度合い (0:uniform, 1:full)
            float per_beta_start = 0.4f;       ///< IS重みの補正度合い (初期値)
            float per_beta_end = 1.0f;         ///< IS重みの補正度合い (収束値)
            int   per_beta_step = 100000;      ///< betaを収束値まで線形変化させるステップ数
            float per_eps = 1e-6f;             ///< 優先度加算用微小値
            float per_initial_priority = 1.0f; ///< 新規データの初期優先度
            std::string per_initial_priority_mode = "fixed"; ///< fixed / max / actor_approx
            bool use_per_prio_clip = false;    ///< 優先度をクリッピングするか
            float per_prio_clip_value = 50.0f; ///< 優先度の上限値

            bool use_double_dqn = true;   ///< Double DQN 有効化フラグ
            bool use_n_step = true;       ///< N-STEPを使用するか
            bool use_per = true;          ///< PERを使用するか

            bool use_tbo = false;         ///< Transformed Bellman Operatorを使用するか
            float tbo_epsilon = 1e-2f;    ///< TBO変換関数hの正則化項

            int num_quantiles = 51;         ///< 分位数 N (デフォルト51)
            float quantile_huber_kappa = 1.0f;///< Huber Loss の閾値 kappa

            struct IqnConfig {
                TauRuleConfig current_taus{
                    .sample_mode = "random",
                    .num_taus = 64,
                };
                TauRuleConfig target_taus{
                    .sample_mode = "random",
                    .num_taus = 64,
                };
            } iqn;

            struct PlasticityConfig {
                std::string feature_key;
                struct ProbeConfig {
                    int batch_size = 512;
                } probe;
            } plasticity;

            bool use_amp = false;
            bool use_amp_bf16 = false;
        };
    }


    // =============================================================
    // AgentRepository
    // =============================================================

    class AgentRepository {
    public:
        static AgentRepository& Instance() {
            static AgentRepository inst;
            return inst;
        }

        /// @todo AgentFactory → AgentCreator

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


    // =============================================================
    // DefaultAgentFactory
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

} // namespace anet::rl


#define ANET_REGISTER_AGENT_FACTORY(FactoryType) \
    namespace { \
        struct FactoryType##AutoRegister { \
            FactoryType##AutoRegister() { \
                anet::rl::RegisterAgentFactory<FactoryType>(); \
            } \
        }; \
        static FactoryType##AutoRegister global_##FactoryType##_auto_register; \
    }
