#include "RLAgent.hpp"
#include <iostream>
#include <tuple>

const float gamma = 0.99f;// 0.99f; 0.995f      γが高いほど「長期安定」を目指す


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
    optimizer(policy_net->parameters(), torch::optim::AdamOptions(1e-3)),
    epsilon(1.0f),
    latest_loss(0.0f),
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
    //epsilon = std::max(0.05f, 1.0f - step_count / 10000.0f);
    epsilon = std::max(0.05f, epsilon * 0.9998f);

    if (((float)rand() / RAND_MAX) < epsilon) {
        int action = rand() % n_actions_;
        return torch::tensor({ action }, torch::kLong).to(device);
    }
    else {
        auto q_values = policy_net->forward(state.to(device));
        auto result = q_values.max(1);
        auto max_index = std::get<1>(result);
        return max_index.to(torch::kLong);
    }
}

void RLAgent::update(const torch::Tensor& state,
    const torch::Tensor& next_state,
    float reward, bool done)
{
    policy_net->train();

    auto q_values = policy_net->forward(state.to(device));
    auto q_result = q_values.max(1);
    auto q_value = std::get<0>(q_result);

    auto next_q_values = target_net->forward(next_state.to(device));
    auto next_result = next_q_values.max(1);
    auto max_next_q = std::get<0>(next_result);

    auto expected_q = torch::tensor({ reward }, torch::TensorOptions().device(device))
        + gamma * max_next_q * (done ? 0.0f : 1.0f);

    auto loss = torch::mse_loss(q_value, expected_q.detach());
    latest_loss = loss.item<float>();

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    // ターゲットネット更新（serialize経由）
    if (step_count % 200 == 0) {
        torch::serialize::OutputArchive archive;
        policy_net->save(archive);
        torch::serialize::InputArchive in;
        std::stringstream ss;
        archive.save_to(ss);
        in.load_from(ss);
        target_net->load(in);
    }
}
