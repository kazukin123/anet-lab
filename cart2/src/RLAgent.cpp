#include "RLAgent.hpp"
#include <iostream>
#include <tuple>
#include "app.hpp"


const float alpha = 1e-4f;   // 学習率 1e-3 3e-3 1e-4 1e-4 5e-4
const float gamma = 0.995f;// 0.99f; 0.995f      γが高いほど「長期安定」を目指す
const float eps_max = 1.00f;
const float eps_min = 0.05f;
const float eps_decay_step = 100000;
const float tnetup_softupdate_tau = 0.01;// -1;// 0.01;
const int tnetup_hardupdate_step = 500; //200 1000
const float grad_clip_tau = 10.0f;   // 1 5 10

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
    optimizer(policy_net->parameters(), torch::optim::AdamOptions(alpha)),
    epsilon(1.0f),
    latest_loss(0.0f),
    step_count(0),
    grad_norm_clipped_ema(0.0f) {
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
    epsilon = std::max(eps_min, eps_max - step_count / eps_decay_step);
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

void RLAgent::update(const torch::Tensor& state, const torch::Tensor& next_state, float reward, bool done) {
    policy_net->train();

    // 現在の行動価値をメインネットワークから算出
    auto q_values = policy_net->forward(state.to(device));
    auto q_result = q_values.max(1);
    auto q_value = std::get<0>(q_result);

    // 教師信号としての次の行動価値をターゲットネットワークから算出
    // （ターゲットネットワークは随時更新度が低いので、教師信号としての安定性が高い）
    auto next_q_values = target_net->forward(next_state.to(device));
    auto next_result = next_q_values.max(1);
    auto max_next_q = std::get<0>(next_result).detach();
    auto reward_t = torch::full({ 1 }, reward, torch::TensorOptions().device(device));
    auto expected_q = reward_t + gamma * max_next_q * (done ? 0.0f : 1.0f);
    //auto expected_q = torch::tensor({ reward }, torch::TensorOptions().device(device))
    //    + gamma * max_next_q * (done ? 0.0f : 1.0f);

    // 現在の行動価値と次の行動価値（教師信号）の差分を算出
    //auto loss = torch::mse_loss(q_value, expected_q.detach());
    auto loss = torch::nn::functional::smooth_l1_loss(q_value, expected_q);
    latest_loss = loss.item<float>();

    // メインネットワークの勾配計算
    optimizer.zero_grad();
    loss.backward();

    // --- 勾配ノルム測定 ---
    float total_norm = 0.0f;
    for (auto& param : policy_net->parameters()) {
        if (param.grad().defined()) {
            total_norm += param.grad().data().pow(2).sum().item<float>();
        }
    }
    total_norm = std::sqrt(total_norm);

    // メインネットワークに勾配反映
    torch::nn::utils::clip_grad_norm_(policy_net->parameters(), grad_clip_tau);   // 勾配クリッピング（Gradient Clipping）
    optimizer.step();

    // --- soft update（毎回少し近づける） ---
    if (tnetup_softupdate_tau > 0) {
        soft_update(tnetup_softupdate_tau);
    }
    bool hard_update_done = false;
    if (tnetup_hardupdate_step > 0 && (step_count % tnetup_hardupdate_step) == 0) {
        hard_update();      // --- hard update（定期stepごとに完全同期） ---
        wxLogInfo("Hard update done. step=%d", step_count);
        hard_update_done = true;
    }

    // メトリクス生成・更新
    float td_error = (q_value - expected_q).item<float>();
    float grad_norm_clipped = (total_norm > grad_clip_tau) ? 1.0f : 0.0f;    // 勾配ノルムがクリッピングしきい値を超えたか
    grad_norm_clipped_ema = (0.9f * grad_norm_clipped_ema) + 0.1f * grad_norm_clipped;

	// メトリクス記録
    wxGetApp().logScalar("2_agent/loss",     step_count, latest_loss);
    wxGetApp().logScalar("2_agent/q_value",  step_count, q_value.item<double>());
    wxGetApp().logScalar("2_agent/epsilon",  step_count, epsilon);
    wxGetApp().logScalar("2_agent/grad_norm",step_count, total_norm);
    wxGetApp().logScalar("2_agent/grad_clip",step_count, grad_norm_clipped);
    wxGetApp().logScalar("2_agent/grad_clip_ema", step_count, grad_norm_clipped_ema);
    wxGetApp().logScalar("2_agent/reward",   step_count, reward);
    wxGetApp().logScalar("2_agent/td_error", step_count, td_error);
    //    wxGetApp().logScalar("2_agent/done",     step_count, done);
//    wxGetApp().logScalar("2_agent/hard_update_done",     step_count, hard_update_done);
}


