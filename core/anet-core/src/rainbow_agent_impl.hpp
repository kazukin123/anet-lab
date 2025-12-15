// rainbow_agent_impl.hpp

#include "anet/rainbow_agent.hpp"
#include <memory>
#include <optional>

using namespace anet::rl;

const float MET_EMA_DECAY = 0.001f;  // 平滑化係数(メトリクス用)
const float MET_EMA_DECAY_ACT = 0.0005f;  // 平滑化係数(メトリクス用)action_ema用


// ======================================================
// RainbowAgent データ構造
// ======================================================

/// Agent内部変数
struct RainbowAgent::RuntimeVars {
    // ランタイム変数
    float epsilon = 1.0f;
    step_t learn_step = 0;
};

class RainbowAgent::BatchUpdateResult : public anet::rl::BatchUpdateResult {
public:
    float grad_norm = 0.0f;
    float grad_clip_ratio = 0.0f;
    torch::Tensor loss;
    torch::Tensor td_error;
    torch::Tensor max_q;
    mutable torch::Tensor max_q_cpu;
public:
    RainbowAgent::BatchUpdateResult(uint32_t learn_step_diff)
        : anet::rl::BatchUpdateResult(learn_step_diff)
    {
    }

    std::optional<float> GetScalar(const std::string& key, int index) const override
    {
        if (key == "loss") return loss.item<float>();
        if (key == "td_mean") return td_error.abs().mean().item<float>();
        if (key == "grad_norm") return grad_norm;
        if (key == "grad_clip_ratio") return grad_clip_ratio;
        if (key == "q_max") {
            TransQToCpu();
            return max_q_cpu.defined() ? std::optional<float>(max_q_cpu.max().item<float>()) : std::nullopt;
        }
        if (key == "q_mean") {
            TransQToCpu();
            return max_q_cpu.defined() ? std::optional<float>(max_q_cpu.mean().item<float>()) : std::nullopt;
        }
        if (key == "q_std") {
            TransQToCpu();
            return max_q_cpu.defined() ? std::optional<float>(max_q_cpu.std(false).item<float>()) : std::nullopt;
        }
        return std::nullopt;
    }

    std::optional<torch::Tensor> GetTensor(const std::string& key, int index) const override
    {
        if (key == "max_q") return max_q;
        return std::nullopt;
    }

    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int index) const override
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
// RainbowAgent QNet
// ======================================================

class QNet : public torch::nn::Module {
public:
    QNet() = default;
    virtual ~QNet() = default;

    /// 観測からQ表現を出力する。
    virtual torch::Tensor forward(const torch::Tensor& obs) = 0;

    /// アクションの個数（離散アクション前提）
    virtual int64_t GetNumActions() const = 0;

    /// Quantile 数を返す。
    virtual int64_t GetNumQuantiles() const { return 1; }

    /// このQNetが分布的表現を返すかどうか。
    virtual bool IsDistributional() const { return false; }

    /// @todo QNet全体共通要素をQNetに移す
};

class PlainQNet : public QNet {
public:
    explicit PlainQNet(const RainbowAgentConfig& config, int state_dim, int n_actions);

    torch::Tensor forward(const torch::Tensor& obs) override;

    int64_t GetNumActions() const override { return n_actions_; }
    bool IsDistributional() const override { return false; }
    int64_t GetNumQuantiles() const override { return 1; }
private:
    void InitWeights(int nn_init_mode);
private:
    int64_t state_dim_;
    int64_t n_actions_;

    torch::nn::Linear fc1_{ nullptr };
    torch::nn::Linear fc2_{ nullptr };
    torch::nn::Linear fc3_{ nullptr };
};


// ======================================================
// RainbowAgent Network
// ======================================================

class RainbowAgent::Network {
public:
    RainbowAgent::Network(
        const RainbowAgentConfig& config, std::shared_ptr<QNet> policy_net, std::shared_ptr<QNet> target_net);

    /// 行動選択用：期待値Q (B, A)
    torch::Tensor ForwardExpectation(const torch::Tensor& obs) const;

    /// Learner用：生の出力 DQN=(B, A) QR-DQN=(B, A, Nq)
    torch::Tensor ForwardRaw(const torch::Tensor& obs, bool use_target) const;

    /// QR-DQN専用：Quantile 出力
    torch::Tensor ForwardQuantiles(const torch::Tensor& obs, bool use_target) const;

    /// policy_netのパラメータ取得
    std::vector<torch::Tensor> GetPolicyParameters() const;

    /// target network 同期
    void UpdateTarget(step_t learn_step);
private:
    void SoftUpdate();
    void HardUpdate();
private:
    const RainbowAgentConfig& config_;
    std::shared_ptr<QNet> policy_net_;
    std::shared_ptr<QNet> target_net_;
};


// ======================================================
// RainbowAgent ActionPolicy 
// ======================================================

class RainbowAgent::ActionPolicy : public anet::RandomHolder {
public:
    ActionPolicy(
        const RainbowAgent::Network& network, const RainbowAgent::RuntimeVars& vars, seed_t seed);

    BatchActionInfo SelectAction(const torch::Tensor& obs, bool greedy_only) const;
private:
    const RainbowAgent::Network& network_;
    const RainbowAgent::RuntimeVars& vars_;
};

// ======================================================
// RainbowAgent Learner
// ======================================================

class RainbowAgent::Learner : public anet::rl::Learner, public DataExporter {
public:
    Learner(RainbowAgent& agent);

    std::optional<float> GetScalar(const std::string& key, int index = -1) const override;
    std::optional<torch::Tensor> GetTensor(const std::string& key, int index = -1) const override;
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int index = -1) const override;

    virtual ~Learner() = default;
protected:
    RainbowAgent& agent_;
    std::unique_ptr<anet::rl::ReplayBuffer> replay_buffer_;
};

class RainbowAgent::TDLearner : public RainbowAgent::Learner {
public:
    explicit TDLearner(RainbowAgent& agent, const EnvSpec& env_spec, seed_t replay_seed);

    std::shared_ptr<const anet::rl::BatchUpdateResult> UpdateFromBatch(
        const StepCounts& step, const BatchExperience& expriences, const Runner& trainer) override;

private:
    bool CanUpdate(step_t update_step) const;
    void UpdateEpsilon(step_t learn_step);
    void UpdateTargetNetwork(step_t step);
private:
    std::unique_ptr<torch::optim::Adam> optimizer_;
};


