#include "RLAgent.hpp"
#include <iostream>
#include <tuple>
#include "app.hpp"

const float gamma = 0.995f;// 0.99f; 0.995f      γが高いほど「長期安定」を目指す


// ======================================================
// QNet 定義（Impl を CPP に置く）
// ======================================================
struct QNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };

    QNetImpl(int state_dim, int n_actions) {
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, n_actions));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }
};

// ======================================================
// RLAgent 実装
// ======================================================
RLAgent::RLAgent(int state_dim, int n_actions, torch::Device device)
    : device(device),
    n_actions_(n_actions),
    policy_net(std::make_shared<QNetImpl>(state_dim, n_actions)),
    target_net(std::make_shared<QNetImpl>(state_dim, n_actions)),
    optimizer(policy_net->parameters(), torch::optim::AdamOptions(1e-4)),   // 学習率 1e-3 3e-3
    epsilon(1.0f),
    latest_loss(0.0f),
    avg_loss(0.0f),
    step_count(0)
{
    policy_net->to(device);
    target_net->to(device);
    target_net->eval();

    // ✅ Module 間コピーは torch::serialize を使う
    torch::serialize::OutputArchive archive;
    policy_net->save(archive);
    torch::serialize::InputArchive in;
    std::stringstream ss;
    archive.save_to(ss);
    in.load_from(ss);
    target_net->load(in);
}

torch::Tensor RLAgent::select_action(torch::Tensor state) {
    step_count++;
    epsilon = std::max(0.01f, 1.0f - step_count / 200000.0f);
    //epsilon = std::max(0.05f, epsilon * 0.9998f);
    //epsilon = 1.0f;

    if (((float)rand() / RAND_MAX) < epsilon) {
        // 確率εがヒットした場合はランダムでActionを決定
        int action = rand() % n_actions_;
        //int action = 0;
        return torch::tensor({ action }, torch::kLong).to(device);
    }
    else {
        // メインネットワークを元にActionを決定
        auto q_values = policy_net->forward(state.to(device));
        auto result = q_values.max(1);
        auto max_index = std::get<1>(result);
        return max_index.to(torch::kLong);
    }
}
void RLAgent::hard_update() {
    torch::NoGradGuard no_grad;

    auto src = policy_net->named_parameters();
    auto dst = target_net->named_parameters();

    for (auto& kv : src) {
        if (dst.contains(kv.key())) {
            dst[kv.key()].copy_(kv.value());
        }
    }
}

void RLAgent::soft_update(float tau) {
    torch::NoGradGuard no_grad;
    auto src = policy_net->named_parameters();
    auto dst = target_net->named_parameters();

    for (auto& kv : src) {
        if (dst.contains(kv.key())) {
            dst[kv.key()].mul_(1.0 - tau);
            dst[kv.key()].add_(kv.value() * tau);
        }
    }
}

void RLAgent::update(const torch::Tensor& state, const torch::Tensor& next_state, float reward, bool done)
{
    policy_net->train();

    // 現在の行動価値をメインネットワークから算出
    auto q_values = policy_net->forward(state.to(device));
    auto q_result = q_values.max(1);
    auto q_value = std::get<0>(q_result);

    // 教師信号としての次の行動価値をターゲットネットワークから算出
    // （ターゲットネットワークは随時更新度が低いので、教師信号としての安定性が高い）
    auto next_q_values = target_net->forward(next_state.to(device));
    auto next_result = next_q_values.max(1);
    auto max_next_q = std::get<0>(next_result);
    auto expected_q = torch::tensor({ reward }, torch::TensorOptions().device(device))
        + gamma * max_next_q * (done ? 0.0f : 1.0f);

    // 現在の行動価値と次の行動価値（教師信号）の差分を算出
    auto loss = torch::mse_loss(q_value, expected_q.detach());
    latest_loss = loss.item<float>();
    avg_loss = 0.9f * avg_loss + 0.1f * latest_loss;

    // メインネットワークに勾配反映
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    // --- soft update（毎回少し近づける） ---
    soft_update(0.01f);
    if (step_count % 200 == 0) {
        hard_update();      // --- hard update（200stepごとに完全同期） ---
    }

    // 低頻度でメインネットワークの内容をターゲットネットに上書き
    // (最新の学習結果であるメインネットワークの内容を教師データとして取り込む）
    //if (step_count % 50 == 0) { // 50stepに一回
    //    torch::serialize::OutputArchive archive;
    //    policy_net->save(archive);
    //    torch::serialize::InputArchive in;
    //    std::stringstream ss;
    //    archive.save_to(ss);
    //    in.load_from(ss);
    //    target_net->load(in);
    //}

	wxGetApp().logScalar("agent/loss", latest_loss, step_count);
    wxGetApp().logScalar("agent/q_value", q_value.item<double>(), step_count);
    wxGetApp().logScalar("agent/epsilon", epsilon, step_count);

}
