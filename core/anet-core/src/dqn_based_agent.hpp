// rainbow_agent_impl.hpp

#include <memory>
#include <optional>
#include "anet/agent.hpp"
#include "anet/rl.hpp"
#include "anet/scaler.hpp"


namespace anet::rl::dqn {

    const float MET_EMA_DECAY = 0.001f;  // 平滑化係数(メトリクス用)
    const float MET_EMA_DECAY_ACT = 0.0005f;  // 平滑化係数(メトリクス用)action_ema用


    // ======================================================
    //  データ構造
    // ======================================================

    /// ランタイム変数
    struct anet::rl::dqn::RuntimeVars {
        float epsilon = 1.0f;
        anet::rl::step_t learn_step = 0;
        float per_beta = 0.0f;  ///< PER用beta
    };


    // ======================================================
    // BatchUpdateResult 
    // ======================================================

    class BatchUpdateResult : public anet::rl::BatchUpdateResult {
    public:
        torch::Tensor grad_norm_tensor;
        std::optional<float> grad_norm;
        float grad_clip_ratio = 0.0f;
        torch::Tensor loss;
        torch::Tensor td_error;
        torch::Tensor max_q;
        mutable torch::Tensor max_q_cpu;

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

            if (key == "loss")
                return loss.item<float>();
            if (key == "td_mean")
                return td_error.abs().mean().item<float>();
            if (key == "grad_norm") {
                if (grad_norm.has_value())
                    return *grad_norm;
                if (grad_norm_tensor.defined())
                    return grad_norm_tensor.item<float>();
                return std::nullopt;
            }
            if (key == "grad_clip_ratio")
                return grad_clip_ratio;
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

            // PER Metrics (Lazy Evaluation)
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
            if (key == "q_std") {
                if (q_std.defined()) return anet::ToFloat(q_std);
                return 0.0f;
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
    //  QNet
    // ======================================================

    class QNet : public torch::nn::Module {
    public:
        QNet() = default;
        virtual ~QNet() = default;

        /// 観測からQ表現を出力する。
        virtual torch::Tensor Forward(const torch::Tensor& obs) = 0;

        /// QR-DQN専用：Quantile 出力 (B, A, N)
        virtual torch::Tensor ForwardQuantiles(const torch::Tensor& obs) {
            throw std::runtime_error("ForwardQuantiles not implemented for this QNet.");
        }

        /// アクションの個数（離散アクション前提）
        virtual int64_t GetNumActions() const = 0;

        /// Quantile 数を返す。
        virtual int64_t GetNumQuantiles() const { return 1; }

        /// このQNetが分布的表現を返すかどうか。
        virtual bool IsDistributional() const { return false; }

        /// メトリクス用
        virtual std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key, const torch::Device& device) = 0;
    };

    class BaseQNet : public QNet {
    public:
        BaseQNet() = delete;
        virtual ~BaseQNet() = default;

        int64_t GetNumActions() const override { return n_actions_; }
        bool IsDistributional() const override { return false; }
        int64_t GetNumQuantiles() const override { return 1; }
    protected:
        BaseQNet(const QNetConfig& config, int64_t state_dim, int64_t n_actions);

        void InitWeightsLinear(torch::nn::Linear& layer, int nn_init_mode, bool is_relu, float manual_gain = -1.0f);
    protected:
        int64_t state_dim_ = 0;
        int64_t n_actions_ = 0;
        torch::nn::Linear fc1_{ nullptr };
        torch::nn::Linear fc2_{ nullptr };
    };

    class PlainQNet final : public BaseQNet {
    public:
        explicit PlainQNet(const QNetConfig& config, int state_dim, int n_actions);
        torch::Tensor Forward(const torch::Tensor& obs) override;
        std::optional<anet::TensorFunction> GetTensorFunction(
            const std::string& key, const torch::Device& device) override;
    private:
        torch::nn::Linear fc3_{ nullptr };
    };

    class DuelingQNet : public BaseQNet {
    public:
        explicit DuelingQNet(const QNetConfig& config, int state_dim, int n_actions);
        torch::Tensor Forward(const torch::Tensor& obs);

        std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key, const torch::Device& device) override;
    private:
        torch::nn::Linear value_{ nullptr };  // (H -> 1)
        torch::nn::Linear adv_{ nullptr };    // (H -> A)
    };

    class QuantilePlainQNet final : public BaseQNet {
    public:
        explicit QuantilePlainQNet(const QNetConfig& config, int state_dim, int n_actions);

        // 平均値を返す (DQN互換)
        torch::Tensor Forward(const torch::Tensor& obs) override;

        // 分位数 (B, A, N) を返す
        torch::Tensor ForwardQuantiles(const torch::Tensor& obs) override;

        int64_t GetNumQuantiles() const override { return num_quantiles_; }
        bool IsDistributional() const override { return true; }

        std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key, const torch::Device& device) override;
    private:
        int64_t num_quantiles_;
        torch::nn::Linear fc3_{ nullptr };
    };

    class QuantileDuelingQNet final : public BaseQNet {
    public:
        explicit QuantileDuelingQNet(const QNetConfig& config, int state_dim, int n_actions);

        // 平均値を返す (DQN互換)
        torch::Tensor Forward(const torch::Tensor& obs) override;

        // 分位数 (B, A, N) を返す
        torch::Tensor ForwardQuantiles(const torch::Tensor& obs) override;

        int64_t GetNumQuantiles() const override { return num_quantiles_; }
        bool IsDistributional() const override { return true; }

        std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key, const torch::Device& device) override;
    private:
        int64_t num_quantiles_;
        torch::nn::Linear value_{ nullptr };  // (H -> 1*N)
        torch::nn::Linear adv_{ nullptr };    // (H -> A*N)
    };


    // ======================================================
    //  Network
    // ======================================================

    class Network {
    public:
        Network(
            const NetworkConfig& config, const torch::Device& device, std::shared_ptr<QNet> policy_net, std::shared_ptr<QNet> target_net);

        /// 行動選択用：期待値Q (B, A)
        torch::Tensor ForwardExpectation(const torch::Tensor& obs, bool use_target) const;

        /// Learner用：Q出力 DQN=(B, A) QR-DQN=(B, A, Nq)
        torch::Tensor Forward(const torch::Tensor& obs, bool use_target) const;


        /// QR-DQN専用：Quantile 出力
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
        std::shared_ptr<QNet> policy_net_;
        std::shared_ptr<QNet> target_net_;
    };


    // ======================================================
    // ActionPolicy 
    // ======================================================

    class ActionPolicy : public anet::RandomHolder {
    public:
        ActionPolicy(
            const anet::rl::dqn::Network& network, const anet::rl::dqn::RuntimeVars& vars, anet::seed_t seed);

        BatchActionInfo SelectAction(const torch::Tensor& obs, bool greedy_only, bool use_target) const;
    private:
        const anet::rl::dqn::Network& network_;
        const RuntimeVars& vars_;
    };


    // ======================================================
    // Learner
    // ======================================================

    class Learner : public anet::rl::Learner, public anet::DataExporter {
    public:
        Learner(const LearnerConfig& config, Network& network, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec,
            torch::Device device, anet::seed_t replay_seed);

        BatchUpdateResultList UpdateFromBatch(const StepCounts& step, const BatchExperience& expriences, const Runner& trainer) override;

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
        void UpdatePerBeta(step_t learn_step);
        void UpdateEpsilon(step_t learn_step);
        void UpdateTargetNetwork(step_t step);
    protected:
        const torch::Device device_;
        int batch_size_;
        int state_dim_;
        int n_actions_;
        LearnerConfig config_;
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
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed);

        std::shared_ptr<const anet::rl::BatchUpdateResult> UpdateFromSamples(
            const anet::rl::ExperienceSamples& samples) override;
    };

    class QRLearner final : public anet::rl::dqn::Learner {
    public:
        explicit QRLearner(const LearnerConfig& config, Network& network, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed);

        std::shared_ptr<const anet::rl::BatchUpdateResult> UpdateFromSamples(
            const anet::rl::ExperienceSamples& samples) override;
    private:
        torch::Tensor ComputeQuantileHuberLoss(
            const torch::Tensor& current_dist, const torch::Tensor& target_dist) const;
    };


} // namespace anet::rl::dqn
