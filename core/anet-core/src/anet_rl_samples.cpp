#include "anet/rl.hpp"
#include <random>

using namespace anet::rl;

// =============================================================
// DummyEnv：x[0] が ±2 を超えるとエピソード終了。
// =============================================================
class DummyEnv : public Environment {
public:
    torch::Tensor state = torch::zeros({ 4 });

    anet::rl::StateSpaceInfo GetStateSpaceInfo() const override {
        return {
            /*shape=*/torch::tensor({4}),
            /*low=*/torch::tensor( { 0.0f, 0.0f, 0.0f, -std::numeric_limits<float>::infinity()}),
            /*high=*/torch::tensor({ 2.0f, 0.0f, 0.0f,  std::numeric_limits<float>::infinity()})
        };
    }

    torch::Tensor Reset(anet::rl::RunMode mode = anet::rl::RunMode::Train) override {
        state = torch::randn({ 4 });
        return state;
    }

    EnvResponse DoStep(const torch::Tensor& action, anet::rl::RunMode mode = anet::rl::RunMode::Train) override {
        float act = action.item<float>();
        state = state + torch::tensor({ act * 0.1f, 0.0, 0.0, 0.0 });
        float reward = 1.0f - std::abs(state[0].item<float>());
        bool done = std::abs(state[0].item<float>()) > 2.0f;
        bool truncated = false;
        return { state.clone(), reward, done, truncated };
    }

    torch::Tensor GetState() const override { return state; }
};

// =============================================================
// DQN風エージェント（ReplayBuffer利用）
// =============================================================
class DQNStyleAgent : public Agent {
public:
    DQNStyleAgent(int state_dim, int action_dim)
        : policy(torch::nn::Linear(state_dim, action_dim)) {
        torch::nn::init::xavier_uniform_(policy->weight);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
        SelectAction(const torch::Tensor& state, RunMode mode = RunMode::Train) override {
        torch::NoGradGuard no_grad;
        auto q_values = policy->forward(state);
        int action_index;
        if (mode == RunMode::Train && ((float)rand() / RAND_MAX) < epsilon)
            action_index = rand() % q_values.size(0);
        else
            action_index = q_values.argmax().item<int>();
        return { torch::tensor(action_index, torch::kInt64), torch::Tensor(), torch::Tensor() };
    }

    void Update(const Experience& e) override { buffer_.Push(e); }

    void UpdateBatch(const BatchData&) override {
        if (buffer_.Size() < batch_size_) return;
        auto samples = buffer_.Sample(batch_size_);
        torch::Tensor loss = torch::zeros({ 1 });
        for (const auto& e : samples)
            loss += torch::pow(e.state.mean() - e.response.next_state.mean(), 2);
        std::cout << "[DQN] loss=" << loss.item<float>() << " (" << samples.size() << " samples)\n";
        epsilon = std::max(0.05f, epsilon * 0.99f);
    }

private:
    torch::nn::Linear policy;
    ReplayBuffer buffer_{ 5000 };
    float epsilon = 1.0f;
    const size_t batch_size_ = 32;
};

// =============================================================
// PPO風エージェント（OnPolicySession使用）
// =============================================================
class PPOStyleAgent : public Agent {
public:
    PPOStyleAgent(int state_dim, int action_dim)
        : policy(torch::nn::Linear(state_dim, action_dim)), value_net(torch::nn::Linear(state_dim, 1)) {
        torch::nn::init::xavier_uniform_(policy->weight);
        torch::nn::init::xavier_uniform_(value_net->weight);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
        SelectAction(const torch::Tensor& state, RunMode mode = RunMode::Train) override {
        auto logits = policy->forward(state);
        auto probs = torch::softmax(logits, -1);
        int action_index = probs.argmax().item<int>();
        auto log_prob = torch::log(probs[action_index]);
        auto value = value_net->forward(state).squeeze(-1);
        return { torch::tensor(action_index, torch::kInt64), log_prob, value };
    }

    void UpdateBatch(const BatchData& batch) override {
        std::cout << "[PPO] collected " << batch.Size() << " experiences\n";
    }

private:
    torch::nn::Linear policy;
    torch::nn::Linear value_net;
};

// =============================================================
// サンプル①：ReplayBuffer学習（DQN）
// =============================================================
void Sample_ReplayBufferTraining(Environment& env, DQNStyleAgent& agent) {
    std::cout << "\n=== ReplayBuffer Training ===\n";
    auto state = env.Reset();
    for (int t = 0; t < 200; ++t) {
        auto [action, _, __] = agent.SelectAction(state);
        auto resp = env.DoStep(action);
        agent.Update({ state, action, resp });
        state = resp.next_state;
        if (resp.done) env.Reset();
        if (t % 10 == 0) agent.UpdateBatch(BatchData());
    }
}

// =============================================================
// サンプル②：On-policy集約（PPO）
// =============================================================
void Sample_OnPolicySession(Environment& env, PPOStyleAgent& agent) {
    std::cout << "\n=== On-Policy Session ===\n";
    OnPolicySession session(agent);
    auto state = env.Reset();
    for (int t = 0; t < 100; ++t) {
        auto [action, logp, value] = agent.SelectAction(state);
        auto resp = env.DoStep(action);
        session.AddExperience({ state, action, resp });
        state = resp.next_state;
        if (resp.done) env.Reset();
    }
    session.Finalize();
}

// =============================================================
// サンプル③：ミックスモード（1000stepごとに評価）
// =============================================================
void Sample_MixedTrainingAndEval(Environment& env, DQNStyleAgent& agent) {
    std::cout << "\n=== Mixed Train+Eval ===\n";
    auto state = env.Reset();
    for (int t = 1; t <= 3000; ++t) {
        auto [action, _, __] = agent.SelectAction(state);
        auto resp = env.DoStep(action);
        agent.Update({ state, action, resp });
        state = resp.next_state;
        if (resp.done) env.Reset();
        if (t % 20 == 0) agent.UpdateBatch(BatchData());

        if (t % 1000 == 0) {
            DummyEnv eval_env;
            auto s = eval_env.Reset();
            float total_reward = 0.0f;
            for (int i = 0; i < 500; ++i) {
                auto [a, _, __2] = agent.SelectAction(s, RunMode::Eval1);
                auto r = eval_env.DoStep(a);
                total_reward += r.reward;
                s = r.next_state;
                if (r.done) break;
            }
            std::cout << "Eval after " << t << " steps: total_reward=" << total_reward << "\n";
        }
    }
}

// =============================================================
// 実行エントリ
// =============================================================
void RunAllSamples() {
    DummyEnv env;
    DQNStyleAgent dqn(4, 2);
    PPOStyleAgent ppo(4, 2);

    Sample_ReplayBufferTraining(env, dqn);
    Sample_OnPolicySession(env, ppo);
    Sample_MixedTrainingAndEval(env, dqn);
}

int main() { RunAllSamples(); }
