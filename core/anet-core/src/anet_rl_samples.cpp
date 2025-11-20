#include <random>
#include "anet/rl.hpp"
#include "anet/replay_buffer.hpp"

using namespace anet::rl;

//// =============================================================
//// DummyEnv：x[0] が ±2 を超えるとエピソード終了。
//// =============================================================
//class DummyEnv : public BatchEnvironment {
//public:
//    torch::Tensor state = torch::zeros({ 4 });
//
//        return {
//            /*shape=*/torch::tensor({4}),
//            /*low=*/torch::tensor( { 0.0f, 0.0f, 0.0f, -std::numeric_limits<float>::infinity()}),
//            /*high=*/torch::tensor({ 2.0f, 0.0f, 0.0f,  std::numeric_limits<float>::infinity()})
//        };
//    }
//
//    torch::Tensor Reset(anet::rl::RunMode mode = anet::rl::RunMode::Train) override {
//        state = torch::randn({ 4 });
//        return state;
//    }
//
//    ActionResult DoStep(const torch::Tensor& action, anet::rl::RunMode mode = anet::rl::RunMode::Train) override {
//        float act = action.item<float>();
//        state = state + torch::tensor({ act * 0.1f, 0.0, 0.0, 0.0 });
//        float reward = 1.0f - std::abs(state[0].item<float>());
//        bool done = std::abs(state[0].item<float>()) > 2.0f;
//        bool truncated = false;
//        return {
//            state.clone(),
//              torch::tensor({ reward }).unsqueeze(0),
//              torch::tensor({ done ? 1.0f : 0.0f }).unsqueeze(0),
//              torch::tensor({ truncated ? 1.0f : 0.0f }).unsqueeze(0)
//        };
//    }
//
//    torch::Tensor GetState() const override { return state; }
//};
//
//class DummyUpdateResult : public UpdateResult {
//public:
//    virtual MetricsMap GetMetricsMap() const override { return MetricsMap(); }
//};
//
//// =============================================================
//// DQN風エージェント（ReplayBuffer利用）
//// =============================================================
//class DQNStyleAgent : public Agent {
//public:
//    DQNStyleAgent(int state_dim, int action_dim)
//        : policy(torch::nn::Linear(state_dim, action_dim)) {
//        torch::nn::init::xavier_uniform_(policy->weight);
//    }
//
//    BatchActionInfo MakeAction(const torch::Tensor& state, RunMode mode = RunMode::Train) override {
//        torch::NoGradGuard no_grad;
//        auto q_values = policy->forward(state);
//        int action_index;
//        bool is_randomized;
//        if (mode == RunMode::Train && ((float)rand() / RAND_MAX) < epsilon)
//            action_index = rand() % q_values.size(0);
//        else
//            action_index = q_values.argmax().item<int>();
//        return { torch::tensor(action_index, torch::kInt64), torch::Tensor()};
//    }
//
//    std::shared_ptr<const UpdateResult> UpdateStep(const Experience& e) override {
//        buffer_.Push(e);
//        return std::make_shared<const DummyUpdateResult>();
//    }
//
//    std::shared_ptr<const UpdateResult> UpdateBatch(const BatchData&) override {
//        //if (buffer_.Size() < batch_size_) return std::make_shared<DummyUpdateResult>();
//        //auto samples = buffer_.Sample(batch_size_);
//        //torch::Tensor loss = torch::zeros({ 1 });
//        //for (const auto& e : samples)
//        //    loss += torch::pow(e.state.mean() - e.response.next_state.mean(), 2);
//        //std::cout << "[DQN] loss=" << loss.item<float>() << " (" << samples.size() << " samples)\n";
//        //epsilon = std::max(0.05f, epsilon * 0.99f);
//        return std::make_shared<DummyUpdateResult>();
//    }
//
//    void OnPostUpdate(const std::shared_ptr<UpdateResult>& result) {}
//private:
//    torch::nn::Linear policy;
//    ReplayBuffer buffer_{ 5000 };
//    float epsilon = 1.0f;
//    const size_t batch_size_ = 32;
//};
//
//
//// =============================================================
//// サンプル①：ReplayBuffer学習（DQN）
//// =============================================================
//void Sample_ReplayBufferTraining(Environment& env, DQNStyleAgent& agent) {
//    std::cout << "\n=== ReplayBuffer Training ===\n";
//    auto state = env.Reset();
//    for (int t = 0; t < 200; ++t) {
//        auto [action, _] = agent.MakeAction(state);
//        auto resp = env.DoStep(action);
//        agent.UpdateStep({ state, action, resp });
//        state = resp.next_state;
//        if (resp.IsDone()) env.Reset();
//        const Experience e;
//        if (t % 10 == 0) agent.UpdateStep(e);
//    }
//}
//
//// =============================================================
//// サンプル③：ミックスモード（1000stepごとに評価）
//// =============================================================
//void Sample_MixedTrainingAndEval(Environment& env, DQNStyleAgent& agent) {
//    std::cout << "\n=== Mixed Train+Eval ===\n";
//    //auto state = env.Reset();
//    //for (int t = 1; t <= 3000; ++t) {
//    //    auto [action, _, __] = agent.MakeAction(state);
//    //    auto resp = env.DoStep(action);
//    //    agent.UpdateStep({ state, action, resp });
//    //    state = resp.next_state;
//    //    if (resp.done) env.Reset();
//    //    const Experience e;
//
//    //    if (t % 1000 == 0) {
//    //        DummyEnv eval_env;
//    //        auto s = eval_env.Reset();
//    //        float total_reward = 0.0f;
//    //        for (int i = 0; i < 500; ++i) {
//    //            auto [a, _, __2] = agent.MakeAction(s, RunMode::Eval1);
//    //            auto r = eval_env.DoStep(a);
//    //            total_reward += r.reward;
//    //            s = r.next_state;
//    //            if (r.done) break;
//    //        }
//    //        std::cout << "Eval after " << t << " steps: total_reward=" << total_reward << "\n";
//    //    }
//    //}
//}
//
//// =============================================================
//// 実行エントリ
//// =============================================================
//void RunAllSamples() {
//    DummyEnv env;
//    DQNStyleAgent dqn(4, 2);
//
//    Sample_ReplayBufferTraining(env, dqn);
//    Sample_MixedTrainingAndEval(env, dqn);
//}
//
