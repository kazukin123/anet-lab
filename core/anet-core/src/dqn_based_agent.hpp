// rainbow_agent_impl.hpp

#include <memory>
#include <optional>
#include "anet/agent.hpp"
#include "anet/rl.hpp"
#include "anet/scaler.hpp"
#include "anet/nn.hpp"


namespace anet::rl::dqn {

    const float MET_EMA_DECAY = 0.001f;  // 平滑化係数(メトリクス用)
    const float MET_EMA_DECAY_ACT = 0.0005f;  // 平滑化係数(メトリクス用)action_ema用


    // ======================================================
    //  データ構造
    // ======================================================

    /// ランタイム変数
    struct anet::rl::dqn::RuntimeVars {
        float epsilon = 1.0f;
        float uqe_tau = 0.9f;
        anet::rl::step_t learn_step = 0;
        float per_beta = 0.0f;  ///< PER用beta
    };


    // ======================================================
    // BatchUpdateResult 
    // ======================================================

    class BatchUpdateResult : public anet::rl::BatchUpdateResult {
    public:
        // grad/loss/td
        torch::Tensor grad_norm_tensor;
        std::optional<float> grad_norm;
        float grad_clip_ratio = 0.0f;
        torch::Tensor loss;
        torch::Tensor td_error;

		// Q Value Metrics Source Tensors
        torch::Tensor max_q;
        mutable torch::Tensor max_q_cpu;
        torch::Tensor q_gap;
        torch::Tensor q_gap_rel;

        // PER Metrics Source Tensors
        torch::Tensor per_is_weights;      ///< IS Weights (B,)
        torch::Tensor per_priorities;      ///< Updated Priorities (B,)
        torch::Tensor per_clipped_count;   ///< Clipped Count (scalar tensor)
        long per_minibatch_size = 0;       ///< Batch Size

        // QR-DQN Metrics
        torch::Tensor q_std; // 分布の標準偏差

    public:
        BatchUpdateResult() = default;

        std::optional<float> GetScalar(const std::string& key, int64_t index) const override
        {
            // 必要になって初めてCPUに転送する

            // loss/td/grad
            if (key == "loss") return loss.item<float>();
            if (key == "td_mean") return td_error.abs().mean().item<float>();
            if (key == "grad_norm") {
                if (grad_norm.has_value())
                    return *grad_norm;
                if (grad_norm_tensor.defined())
                    return grad_norm_tensor.item<float>();
                return std::nullopt;
            }
            if (key == "grad_clip_ratio") return grad_clip_ratio;

			// Q Values
            if (key == "q_max_max") {
                TransQToCpu();
                return max_q_cpu.defined() ? std::optional<float>(max_q_cpu.max().item<float>()) : std::nullopt;
            }
            if (key == "q_max_mean") {
                TransQToCpu();
                return max_q_cpu.defined() ? std::optional<float>(max_q_cpu.mean().item<float>()) : std::nullopt;
            }
            if (key == "q_max_std") {
                TransQToCpu();
                return max_q_cpu.defined() ? std::optional<float>(max_q_cpu.std(false).item<float>()) : std::nullopt;
            }
            if (key == "q_std") {
                if (q_std.defined()) return anet::ToFloat(q_std);
                return 0.0f;
            }
            if (key == "q_gap") {
                if (q_gap.defined()) return anet::ToFloat(q_gap);
                return 0.0f;
            }
            if (key == "q_gap_rel") {
                if (q_gap_rel.defined()) return anet::ToFloat(q_gap_rel);
                return 0.0f;
            }

            // PER Metrics
            if (key == "per_td_error_abs_max") {
                if (td_error.defined())
                    return td_error.abs().max().item<float>();
                return std::nullopt;
            }
            if (key == "per_prio_clip_ratio") {
                if (per_clipped_count.defined() && per_minibatch_size > 0)
                    return per_clipped_count.item<float>() / static_cast<float>(per_minibatch_size);
                return std::nullopt;
            }
            if (key == "per_prio_max") {
                if (per_priorities.defined())
                    return per_priorities.max().item<float>();
                return std::nullopt;
            }
            if (key == "per_batch_prio_mean") {
                if (per_priorities.defined())
                    return per_priorities.mean().item<float>();
                return std::nullopt;
            }
            if (key == "per_is_weight_mean") {
                if (per_is_weights.defined())
                    return per_is_weights.mean().item<float>();
                return std::nullopt;
            }

            return std::nullopt;
        }

        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index) const override
        {
            if (key == "max_q") return max_q;
            return std::nullopt;
        }

        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index) const override
        {
            return std::nullopt;
        }
    private:
        void TransQToCpu() const {
            if (max_q_cpu.defined()) return;
            max_q_cpu = max_q.cpu();
        }
    };


    // ======================================================
    //  Network
    // ======================================================

    class Network {
    public:
        Network(
            const NetworkConfig& config, const torch::Device& device,
            std::shared_ptr<anet::nn::Network> policy_net,
            std::shared_ptr<anet::nn::Network> target_net,
            int64_t n_actions,
            int64_t num_quantiles = 1);

        /// 行動選択・学習用：期待値Q (B, A) を返す
        /// QR-DQNの場合は分布の平均を計算して返す
        torch::Tensor Forward(const torch::Tensor& obs, bool use_target) const;

        /// QR判定
        bool IsDistributional(bool use_target) const;

        /// QR-DQN学習用：Quantile 出力 (B, A, N) を返す
        /// 内部で (B, A*N) -> (B, A, N) へのReshapeを行う
        torch::Tensor ForwardQuantiles(const torch::Tensor& obs, bool use_target) const;

        /// policy_netのパラメータ取得
        std::vector<torch::Tensor> GetPolicyParameters() const;

        /// target network 同期
        void UpdateTarget(anet::rl::step_t learn_step);

        /// メトリクス用：NN生出力
        std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key, const torch::Device& device);
    private:
        void SoftUpdate();
        void HardUpdate();
    private:
        const NetworkConfig config_;
        std::shared_ptr<anet::nn::Network> policy_net_;
        std::shared_ptr<anet::nn::Network> target_net_;

        int64_t n_actions_;
        int64_t num_quantiles_;
    };


    // ======================================================
    // ActionPolicy
    // ======================================================

    class ActionPolicy : public anet::RandomHolder {
    public:
        ActionPolicy(const ActionPolicyConfig& config,
            const anet::rl::dqn::Network& network, anet::rl::dqn::RuntimeVars& vars, anet::seed_t seed);

        virtual BatchActionInfo SelectAction(const torch::Tensor& obs, bool greedy_only, bool use_target) const = 0;
        virtual void OnLearn(const StepCounts& counts) { }

        virtual ~ActionPolicy() = default;
    protected:
        torch::Tensor MakeEpsilonGreedyAction(const torch::Tensor& greedy_action, float epsilon, int64_t batch_size, int64_t n_actions) const;
        BatchActionInfo MakeActionInfo(const torch::Tensor& action_values, const torch::Tensor& q_values, const torch::Tensor& q_quantiles) const;
        torch::Tensor GetQuantiles(const torch::Tensor& obs, bool use_target) const;
        void UpdateEpsilon(step_t step, bool is_uqe = false);
    protected:
        const ActionPolicyConfig config_;
        const anet::rl::dqn::Network& network_;
        RuntimeVars& vars_;
    };

    class EpsilonGreedyActionPolicy final : public ActionPolicy {
    public:
        EpsilonGreedyActionPolicy(const ActionPolicyConfig& config,
            const anet::rl::dqn::Network& network, anet::rl::dqn::RuntimeVars& vars, anet::seed_t seed);

        BatchActionInfo SelectAction(const torch::Tensor& obs, bool greedy_only, bool use_target) const;
        void OnLearn(const StepCounts& counts) override;
    };

    /**
     * UQE (Upper Quantile Exploration)
     * 分布型RLにおける楽観的探索 (Optimistic Exploration) を行うクラス。
     * 分布の右裾（上位Quantile）の情報を用いて、不確実性の高い行動を積極的に探索する。
     * * uqe_use_tail_meanにより、以下2つの標準的な手法に対応：
     * 1. Q-UCB: 特定の上位分位点(tau)の値を使用 (falseの場合)
     * 2. Upper CVaR: 上位tauから1.0までの平均値を使用 (trueの場合)
     */
    class UQEActionPolicy : public ActionPolicy {
    public:
        UQEActionPolicy(const ActionPolicyConfig& config,
            const anet::rl::dqn::Network& network, anet::rl::dqn::RuntimeVars& vars, anet::seed_t seed);

        BatchActionInfo SelectAction(const torch::Tensor& obs, bool greedy_only, bool use_target) const;
        void OnLearn(const StepCounts& counts) override;

        virtual ~UQEActionPolicy() = default;
    protected:
        anet::rl::BatchActionInfo MakeUQEActionInfo(float tau, const torch::Tensor& tau_tensor, const torch::Tensor& obs, bool greedy_only, bool use_target) const;
        void UpdateTau(step_t step);
    private:
        torch::Tensor MakeUQEAction(float tau, const torch::Tensor& q_quantiles) const;
        torch::Tensor MakeVectorizedUQEAction(const torch::Tensor& tau_tensor, const torch::Tensor& q_quantiles) const;
    };

    class ThompsonSamplingActionPolicy final : public UQEActionPolicy {
    public:
        ThompsonSamplingActionPolicy(const ActionPolicyConfig& config,
            const anet::rl::dqn::Network& network, anet::rl::dqn::RuntimeVars& vars, anet::seed_t seed);

        BatchActionInfo SelectAction(const torch::Tensor& obs, bool greedy_only, bool use_target) const;
        void OnLearn(const StepCounts& counts) override;
    };

    // ======================================================
    // Learner
    // ======================================================

    class Learner : public anet::rl::Learner, public anet::Module {
    public:
        Learner(const LearnerConfig& config, Network& network, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec,
            torch::Device device, anet::seed_t replay_seed,
            std::optional<StuckerConfig> stucker_config = std::nullopt);

        BatchUpdateResultList UpdateFromBatch(const StepCounts& step, const BatchExperience& expriences, std::shared_ptr<const anet::rl::Runner> runner) override;

        virtual ~Learner() = default;
    public:
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override;
    protected:
        // アルゴリズム固有の更新処理 (Loss計算, Backprop, Priority更新)
        virtual std::shared_ptr<const anet::rl::BatchUpdateResult> UpdateFromSamples(
            const anet::rl::ExperienceSamples& samples) = 0;
    protected:
        void SetupOptimizer();                  ///< 共通初期化処理（Optimizer生成など）
        void SetupReplayBuffer(const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec, anet::seed_t seed);
    private:
        bool CanUpdate(step_t update_step, step_t exp_step) const;
        void UpdatePerBeta(step_t step);
        void UpdateTargetNetwork(step_t step);
    protected:
        const torch::Device device_;
        int batch_size_;
        int state_dim_;
        int n_actions_;
        float earned_credit_;
        LearnerConfig config_;
        std::optional<StuckerConfig> stucker_config_;
        Network& network_;
        RuntimeVars& vars_;
        std::shared_ptr<ObservationNormalizer> obs_norm_;
        std::shared_ptr<anet::rl::ReplayBuffer> replay_buffer_;
        std::unique_ptr<torch::optim::Adam> optimizer_;
    protected:
        float update_credit_ = 0.0f;
    };

    class TDLearner final : public anet::rl::dqn::Learner {
    public:
        explicit TDLearner(const LearnerConfig& config, Network& network, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed,
            std::optional<StuckerConfig> stucker_config = std::nullopt);

        std::shared_ptr<const anet::rl::BatchUpdateResult> UpdateFromSamples(
            const anet::rl::ExperienceSamples& samples) override;
    };

    class QRLearner final : public anet::rl::dqn::Learner {
    public:
        explicit QRLearner(const LearnerConfig& config, Network& network, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed,
            std::optional<StuckerConfig> stucker_config = std::nullopt);

        std::shared_ptr<const anet::rl::BatchUpdateResult> UpdateFromSamples(
            const anet::rl::ExperienceSamples& samples) override;
    private:
        torch::Tensor ComputeQuantileHuberLoss(
            const torch::Tensor& current_dist, const torch::Tensor& target_dist) const;
    };


} // namespace anet::rl::dqn
