// rainbow_agent_impl.hpp

#include <memory>
#include <optional>
#include <limits>
#include "anet/agent.hpp"
#include "anet/rl.hpp"
#include "anet/scaler.hpp"
#include "anet/nn.hpp"
#include "anet/nn_util.hpp"


namespace anet::rl::dqn {

    const float MET_EMA_DECAY = 0.001f;  // 平滑化係数(メトリクス用)
    const float MET_EMA_DECAY_ACT = 0.0005f;  // 平滑化係数(メトリクス用)action_ema用


    // ======================================================
    //  データ構造
    // ======================================================

    /// ランタイム変数
    struct anet::rl::dqn::RuntimeVars {
        float epsilon = 1.0f;   // 互換性のために残す
        float uqe_tau = 0.9f;   // 互換性のために残す
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
        mutable torch::Tensor td_error_abs_cpu;
        float grad_clip_tau;

		// Q Value Metrics Source Tensors
        torch::Tensor max_q;
        mutable torch::Tensor max_q_cpu;
        torch::Tensor q_sa;
        mutable torch::Tensor q_sa_cpu;
        torch::Tensor q_gap;
        torch::Tensor q_gap_rel;

        // PER Metrics Source Tensors
        torch::Tensor per_is_weights;      ///< IS Weights (B,)
        torch::Tensor per_priorities;      ///< Updated Priorities (B,)
        torch::Tensor per_clipped_count;   ///< Clipped Count (scalar tensor)
        long per_minibatch_size = 0;       ///< Minibatch Size

        // QR-DQN Metrics
        torch::Tensor q_std; // 分布の標準偏差

    public:
        BatchUpdateResult() = default;

        std::optional<float> GetScalar(const std::string& key, int64_t index) const override
        {
            // 必要になって初めてCPUに転送する

            // loss/td/grad
            if (key == "loss") return loss.item<float>();
            if (key == "td_mean") {
                if (!td_error_abs_cpu.defined())
                    td_error_abs_cpu = td_error.abs().cpu();
                return td_error_abs_cpu.mean().item<float>();
            }
            if (key == "td_std") {
                if (!td_error_abs_cpu.defined())
                    td_error_abs_cpu = td_error.abs().cpu();
                return td_error_abs_cpu.std().item<float>();
            }
            if (key == "grad_norm") {
                if (grad_norm.has_value())
                    return *grad_norm;
                if (grad_norm_tensor.defined())
                    return grad_norm_tensor.item<float>();
                return std::nullopt;
            }
            if (key == "grad_clip_ratio") {
                if (!grad_norm_tensor.defined()) return 0.0f;
                return (grad_norm_tensor.item<float>() > grad_clip_tau) ? 1.0f : 0.0f;
            }

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
            if (key == "q_sa_mean") {
                if (!q_sa_cpu.defined() && q_sa.defined()) q_sa_cpu = q_sa.cpu();
                return q_sa_cpu.defined() ? std::optional<float>(q_sa_cpu.mean().item<float>()) : std::nullopt;
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
            // CV = 標準偏差 / 平均
            //  高: 特定の経験に優先度が集中しており、PERが「選別」を強く行っている状態。
            //  低(0に近い) : すべての経験が似たようなTD誤差を持っており、一様サンプリングに近い状態。
            // 対策: per_alpha を調整
            if (key == "per_prio_cv") {
                if (per_priorities.defined()) {
                    auto mean = per_priorities.mean();
                    auto std = per_priorities.std();
                    return (std / (mean + 1e-9)).item<float>();
                }
                return std::nullopt;
            }
            // 勾配更新の偏り(IS Weightsベース)
            if (key == "per_is_ess_ratio") {
                if (per_is_weights.defined() && per_minibatch_size > 0) {
                    auto w = per_is_weights;
                    auto sum_w = w.sum();
                    auto sum_w2 = (w * w).sum();
                    // ESS = (Σw)^2 / (Σw^2) / B
                    return ((sum_w * sum_w) / (static_cast<float>(per_minibatch_size) * sum_w2 + 1e-9)).item<float>();
                }
                return std::nullopt;
            }
            // 有効サンプルサイズ比率 (ESS Ratio) 実質的にバッチ内の何割のデータが学習に寄与しているか (0.0 ~ 1.0) 公式: (Σp)^2 / (B * Σp^2)
            //   1.0に近い: バッチ内のデータが均等に重要。
            //   0に近い : バッチ内の極一部のデータ（外れ値など）が支配的で、実質的な学習効率が低下している警告信号。 対策: per_beta を上げる（初期値を 0.6 -> 0.8 にするなど）か、per_alpha を下げる。
            // 理想: 0.5 〜 0.8 付近
            if (key == "per_prio_ess_ratio") {
                if (per_priorities.defined() && per_minibatch_size > 0) {
                    auto p = per_priorities;
                    auto sum_p = p.sum();
                    auto sum_p2 = (p * p).sum();
                    auto ess = (sum_p * sum_p) / (static_cast<float>(per_minibatch_size) * sum_p2 + 1e-9);
                    return ess.item<float>();
                }
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
        void TransQToCpu() const
        {
            if (max_q_cpu.defined()) return;
            max_q_cpu = max_q.cpu();
        }
    };

    struct NormalizedSampleObservations {
        anet::TensorDict obs;
        anet::TensorDict next_obs;
    };

    struct OptimizerStepResult {
        torch::Tensor grad_norm_tensor;
        std::optional<float> grad_norm;
        float grad_clip_ratio = 0.0f;
        float grad_clip_tau = std::numeric_limits<float>::infinity();
    };

    struct PerPriorityUpdateInfo {
        torch::Tensor per_clipped_count;
        torch::Tensor per_priorities;
        torch::Tensor per_is_weights;
        long per_minibatch_size = 0;
    };

    struct QuantileMetrics {
        torch::Tensor q_sa;
        torch::Tensor max_q;
        torch::Tensor q_std;
        torch::Tensor q_gap;
        torch::Tensor q_gap_rel;
    };


    // ======================================================
    //  NetworkModel
    // ======================================================

    class NetworkModel : public anet::Serializable {
    public:
        NetworkModel(
            const NetworkModelConfig& config,
            const torch::Device device,
            const anet::nn::NetworkConfig& network_config,
            const anet::TensorSpecMap& obs_spec,
            int64_t n_actions,
            std::shared_ptr<anet::nn::NetworkHeadFactory> head_factory,
            int64_t num_quantiles);
    protected:
        NetworkModel(
            const NetworkModelConfig& config,
            std::shared_ptr<anet::nn::Network> policy_net,
            std::shared_ptr<anet::nn::Network> target_net,
            int64_t n_actions,
            int64_t num_quantiles);
    public:

        /// 行動選択・学習用：期待値Q (B, A) を返す
        /// QR-DQNの場合は分布の平均を計算して返す
        anet::TensorDict Forward(const anet::TensorDict& obs, bool use_target) const;

        // Network取得
        std::shared_ptr<anet::nn::Network> GetMainNetwork() { return policy_net_; }
        std::shared_ptr<anet::nn::Network> GetTargetNetwork() { return target_net_; }

        /// QR判定
        bool IsDistributional(bool use_target) const;

        /// policy_netのパラメータ取得
        std::vector<torch::Tensor> GetPolicyParameters() const;

        torch::OrderedDict<std::string, torch::Tensor> GetPolicyNamedParameters() const;

        /// target network 同期
        void UpdateTarget(anet::rl::step_t learn_step);

        /// メトリクス用：NN生出力
        std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key, const torch::Device& device);

        /// メトリクス用：TensorDict
        std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key, const torch::Device& device);
    public:
        int64_t Save(OutputArchive& archive) const override;
        int64_t Load(InputArchive& archive) override;
    private:
        void SoftUpdate();
        void HardUpdate();
    private:
        const anet::rl::dqn::NetworkModelConfig config_;
        std::shared_ptr<anet::nn::Network> policy_net_;
        std::shared_ptr<anet::nn::Network> target_net_;

        int64_t n_actions_;
        int64_t num_quantiles_;
    };


    // ======================================================
    // ActionPolicy
    // ======================================================

    class DQNActionInfo final : public anet::rl::BatchActionInfo, public anet::ModuleBase {
    public:
        using anet::rl::BatchActionInfo::BatchActionInfo;

        std::shared_ptr<anet::rl::BatchActionInfo> To(torch::Device device) const override;
        std::shared_ptr<anet::rl::BatchActionInfo> WithAction(torch::Tensor action) const override;
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
    };

    class ActionPolicy : virtual public anet::ModuleBase {
    public:
        ActionPolicy(const ActionPolicyConfig& config,
        	bool enable_spatial_exploration = false, int64_t num_envs = 0,
            const torch::Device& device = torch::Device(torch::kCPU));

        virtual std::shared_ptr<BatchActionInfo> SelectAction(const anet::TensorDict& obs, bool greedy_only,
            std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd,
            const anet::TraceSink& sink = {}) const = 0;
        virtual void OnLearn(const StepCounts& counts) { }

        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;

        virtual ~ActionPolicy() = default;
    protected:
        anet::TensorDict ForwardForAction(const anet::TensorDict& obs, std::shared_ptr<anet::nn::Network> network, const anet::TraceSink& sink) const;
        torch::Tensor MakeEpsilonGreedyAction(const torch::Tensor& greedy_action, float epsilon, int64_t num_envs, int64_t n_actions, std::shared_ptr<anet::RandomGenerator> rnd) const;
        torch::Tensor MakeEpsilonGreedyAction(const torch::Tensor& greedy_action, const torch::Tensor& epsilon_tensor, int64_t num_envs, int64_t n_actions, std::shared_ptr<anet::RandomGenerator> rnd) const;
        std::shared_ptr<BatchActionInfo> MakeActionInfo(const torch::Tensor& action_values, const torch::Tensor& q_values, const torch::Tensor& q_quantiles) const;
        //torch::Tensor GetQuantiles(const torch::Tensor& obs, bool use_target) const;
        void UpdateEpsilon(step_t step, bool is_uqe = false);
        bool IsSpatialExplorationEnabled() const { return use_spatial_exploration_; }
        static torch::Tensor CreateSpatialTensor(int64_t num_envs, float start_val, float end_val, const std::string& scale_type, const torch::Device& device);
        torch::Tensor GetSpatialEpsilonTensor(int64_t num_envs, const torch::Device& device, bool is_uqe) const;
        torch::Tensor GetSpatialTauTensor(int64_t num_envs, const torch::Device& device) const;
    protected:
        const ActionPolicyConfig config_;
        bool use_spatial_exploration_ = false;
        int64_t spatial_num_envs_ = 0;
        torch::Tensor spatial_eps_tensor_;
        torch::Tensor spatial_uqe_eps_tensor_;
        torch::Tensor spatial_tau_tensor_;
        float current_epsilon_ = 0.0f;
        float current_uqe_tau_ = 0.0f;
    };

    class EpsilonGreedyActionPolicy final : public ActionPolicy {
    public:
        EpsilonGreedyActionPolicy(const ActionPolicyConfig& config,
            bool enable_spatial_exploration = false,
            int64_t num_envs = 0,
            const torch::Device& device = torch::Device(torch::kCPU));

        std::shared_ptr<BatchActionInfo> SelectAction(const anet::TensorDict& obs, bool greedy_only,
            std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd,
            const anet::TraceSink& sink) const;
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
            bool enable_spatial_exploration = false,
            int64_t num_envs = 0,
            const torch::Device& device = torch::Device(torch::kCPU));

        std::shared_ptr<BatchActionInfo> SelectAction(const anet::TensorDict& obs, bool greedy_only,
            std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd,
            const anet::TraceSink& sink) const;
        void OnLearn(const StepCounts& counts) override;

        virtual ~UQEActionPolicy() = default;
    protected:
        std::shared_ptr<anet::rl::BatchActionInfo> MakeUQEActionInfo(float tau, const torch::Tensor& tau_tensor, const anet::TensorDict& obs, bool greedy_only,
            std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd, const anet::TraceSink& sink) const;
        void UpdateTau(step_t step);
    private:
        torch::Tensor MakeUQEValues(float tau, const torch::Tensor& q_quantiles) const;
        torch::Tensor MakeVectorizedUQEValues(const torch::Tensor& tau_tensor, const torch::Tensor& q_quantiles) const;
    };

    class ThompsonSamplingActionPolicy final : public UQEActionPolicy {
    public:
        ThompsonSamplingActionPolicy(const ActionPolicyConfig& config,
            bool enable_spatial_exploration = false,
            int64_t num_envs = 0,
            const torch::Device& device = torch::Device(torch::kCPU));

        std::shared_ptr<BatchActionInfo> SelectAction(const anet::TensorDict& obs, bool greedy_only,
            std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd,
            const anet::TraceSink& sink) const;
        void OnLearn(const StepCounts& counts) override;
    };


    // ======================================================
    // Actor
    // ======================================================

    class Actor : public anet::rl::Actor {
    public:
        Actor(std::shared_ptr<ActionPolicy> policy,
            std::shared_ptr<anet::rl::ObservationNormalizer> obs_norm,
            std::shared_ptr<ActionContext> context,
            std::shared_ptr<std::shared_mutex> mutex,
            std::shared_ptr<anet::nn::Network> network,
            std::shared_ptr<anet::nn::Network> src_network);
        std::shared_ptr<BatchActionInfo> MakeAction(const StepCounts& step, const anet::rl::BatchState& state) const override;
        void Sync() override;
    private:
        std::shared_ptr<ActionPolicy> policy_;
        std::shared_ptr<anet::rl::ObservationNormalizer> obs_norm_;
        std::shared_ptr<ActionContext> context_;
        std::shared_ptr<std::shared_mutex> mutex_;
        std::shared_ptr<anet::nn::Network> network_;
        std::shared_ptr<anet::nn::Network> src_network_;
    };

    // ======================================================
    // Learner
    // ======================================================

    class Learner : public anet::rl::Learner, public anet::Module, public anet::Serializable, public anet::RandomHolder {
    public:
        Learner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec,
            torch::Device device, anet::seed_t replay_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            std::optional<StuckerConfig> stucker_config = std::nullopt,
            std::optional<anet::seed_t> target_seed = std::nullopt);

        BatchUpdateResultList UpdateFromBatch(const StepCounts& step, const BatchExperience& expriences) override;

        virtual ~Learner() = default;
    public:
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override;
    public:
        int64_t Save(OutputArchive& archive) const override;
        int64_t Load(InputArchive& archive) override;
    protected:
        // アルゴリズム固有の更新処理 (Loss計算, Backprop, Priority更新)
        virtual std::shared_ptr<const anet::rl::BatchUpdateResult> UpdateFromSamples(
            const anet::rl::ExperienceSamples& samples) = 0;
    protected:
        void SetupOptimizer();                  ///< 共通初期化処理（Optimizer生成など）
        void SetupReplayBuffer(const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec, anet::seed_t seed);
        NormalizedSampleObservations NormalizeSampleObservations(const anet::rl::ExperienceSamples& samples) const;
        OptimizerStepResult Optimize(const torch::Tensor& loss);
        PerPriorityUpdateInfo MakePerPriorityUpdateInfo(const anet::rl::ExperienceSamples& samples, const torch::Tensor& td_error) const;
        PerPriorityUpdateInfo UpdatePerPriorities(const anet::rl::ExperienceSamples& samples, const torch::Tensor& td_error);
        std::shared_ptr<anet::rl::dqn::BatchUpdateResult> MakeBatchUpdateResult(
            const torch::Tensor& loss,
            const torch::Tensor& td_error,
            const OptimizerStepResult& opt_result,
            const torch::Tensor& max_q,
            const torch::Tensor& q_sa,
            const PerPriorityUpdateInfo& per_info,
            const torch::Tensor& q_std = torch::Tensor(),
            const torch::Tensor& q_gap = torch::Tensor(),
            const torch::Tensor& q_gap_rel = torch::Tensor()) const;
    private:
        bool CanUpdate(step_t exp_step) const;
        void UpdatePerBeta(step_t step);
        void UpdateTargetNetwork(step_t step);
    protected:
        const torch::Device device_;
        int num_envs_;
        int n_actions_;
        float earned_credit_;
        LearnerConfig config_;
        std::optional<StuckerConfig> stucker_config_;
        std::shared_ptr<ActionPolicy> target_policy_;
        NetworkModel& model_;
        RuntimeVars& vars_;
        std::shared_ptr<ObservationNormalizer> obs_norm_;
        std::shared_ptr<anet::rl::ReplayBuffer> replay_buffer_;
        std::unique_ptr<torch::optim::Optimizer> optimizer_;
        anet::GradScaler grad_scaler_;
    protected:
        float update_credit_ = 0.0f;
    };

    class QuantileLearnerBase : public anet::rl::dqn::Learner {
    public:
        explicit QuantileLearnerBase(
        	const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            std::optional<StuckerConfig> stucker_config = std::nullopt,
            std::optional<anet::seed_t> target_seed = std::nullopt);

        virtual ~QuantileLearnerBase() = default;
    protected:
        torch::Tensor GatherActionQuantiles(const torch::Tensor& quantiles, const torch::Tensor& actions) const;
        torch::Tensor SelectTargetActions(const anet::TensorDict& next_obs);
        torch::Tensor CalcTargetQuantiles(const anet::rl::ExperienceSamples& samples, const torch::Tensor& next_dist) const;
        QuantileMetrics MakeQuantileMetrics(const torch::Tensor& current_dist, const torch::Tensor& q_values_mean) const;
        torch::Tensor ComputeQuantileHuberLoss(
        	const torch::Tensor& current_dist, const torch::Tensor& target_dist, const torch::Tensor& taus) const;
        static torch::Tensor ComputeQuantileHuberLoss(
            const torch::Tensor& current_dist, const torch::Tensor& target_dist, const torch::Tensor& taus, float kappa);
    };

    class TDLearner final : public anet::rl::dqn::Learner {
    public:
        explicit TDLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            std::optional<StuckerConfig> stucker_config = std::nullopt,
            std::optional<anet::seed_t> target_seed = std::nullopt);

        std::shared_ptr<const anet::rl::BatchUpdateResult> UpdateFromSamples(
            const anet::rl::ExperienceSamples& samples) override;
    };

    class QRLearner final : public anet::rl::dqn::QuantileLearnerBase {
    public:
        explicit QRLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            std::optional<StuckerConfig> stucker_config = std::nullopt,
            std::optional<anet::seed_t> target_seed = std::nullopt);

        std::shared_ptr<const anet::rl::BatchUpdateResult> UpdateFromSamples(
            const anet::rl::ExperienceSamples& samples) override;
    private:
        torch::Tensor tau_i_; // QuantileHuberLoss 算出用
    };


} // namespace anet::rl::dqn
