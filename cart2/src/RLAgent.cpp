#include "RLAgent.hpp"
#include <iostream>
#include <tuple>
#include "app.hpp"
#include "CartPoleEnv.hpp"


const float alpha = 3e-4f;   // 学習率 1e-3 3e-3 1e-4 1e-4 3e-4 5e-4
const float gamma = 0.99f;// 0.99f; 0.995f      γが高いほど「長期安定」を目指す
const float eps_max = 1.00f;
const float eps_min = 0.05f;    //0.1f 0.05f
const float eps_decay_step = 100000;
const float tnetup_softupdate_tau = 0.005f;// -1;// 0.01;
const int tnetup_hardupdate_step = -1;// 5000; //200 500 1000
const float grad_clip_tau = 30.0f;   // 1f 5f 10f
const bool use_td_clip = false;
const float td_clip_value = 3.0f;
const int eps_zero_step = 120000;

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
    loss_ema(0.0f),
    step_count(0),
    grad_norm_clipped_ema(0.0f) {
    policy_net->to(device);
    target_net->to(device);
    target_net->eval();

    // Module 間コピーは torch::serialize を使う
    torch::serialize::OutputArchive archive;
    policy_net->save(archive);
    torch::serialize::InputArchive in;
    std::stringstream ss;
    archive.save_to(ss);
    in.load_from(ss);
    target_net->load(in);

    // パラメータ記録
    nlohmann::json params = {
        {"alpha", alpha},
        {"gamma", gamma},
        {"eps_max", eps_max},
        {"eps_min", eps_min},
        {"eps_decay_step", eps_decay_step},
        {"eps_zero_step", eps_zero_step},
        {"tnetup_softupdate_tau", tnetup_softupdate_tau},
        {"tnetup_hardupdate_step", tnetup_hardupdate_step},
        {"grad_clip_tau", grad_clip_tau},
        {"td_clip_value", td_clip_value},
    };
    wxGetApp().logJson("agent/params", params);
    wxGetApp().flushMetricsLog();
}

torch::Tensor RLAgent::select_action(torch::Tensor state) {
    step_count++;

    if (eps_zero_step > 0 && (step_count > eps_zero_step)) {
        epsilon = 0.0f;
    }
    else {
        epsilon = std::max(eps_min, eps_max - step_count / eps_decay_step);
        //epsilon = std::max(0.05f, epsilon * 0.9998f);
        //epsilon = 1.0f;
    }

    if (((float)rand() / RAND_MAX) < epsilon) {
        // 確率εがヒットした場合はランダムでActionを決定
        int action = rand() % n_actions_;
        //int action = 0;
        return torch::tensor({ action }, torch::kLong).to(device);
    } else {
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

void RLAgent::update(const torch::Tensor& state, int action, const torch::Tensor& next_state, float reward, bool done) {
    policy_net->train();

    auto q_values = policy_net->forward(state.to(device));      // (B,A)
    auto action_t = torch::tensor({ action }, torch::dtype(torch::kLong).device(device));
    auto q_sa = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1);  // (B,)

    // 教師信号（ターゲットQ）
    auto next_q_targ = target_net->forward(next_state.to(device));  // (B,A)
    auto max_next_q = std::get<0>(next_q_targ.max(1)).detach();    // (B,)
    auto reward_t = torch::full({ 1 }, reward, torch::TensorOptions().device(device));
    auto expected_q = reward_t + gamma * max_next_q * (done ? 0.0f : 1.0f);

    // TD誤差
    auto td_raw = expected_q - q_sa;
    auto td = use_td_clip ? td_raw.clamp(-td_clip_value, td_clip_value) : td_raw;

    // --- 損失計算（正） ---
    //auto loss = torch::nn::functional::smooth_l1_loss(q_sa, expected_q.detach());
    auto loss = torch::nn::functional::smooth_l1_loss(q_sa, expected_q.detach(),
        torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kMean));

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
    const float ema_decay = 0.98f;  // 平滑化係数
    float grad_norm_clipped = (total_norm > grad_clip_tau) ? 1.0f : 0.0f;    // 勾配ノルムがクリッピングしきい値を超えたか
    grad_norm_clipped_ema = (0.9f * grad_norm_clipped_ema) + 0.1f * grad_norm_clipped;
    loss_ema = ema_decay * loss_ema + (1 - ema_decay) * loss.item<float>();

	// メトリクス記録
    wxGetApp().logScalar("21_agent/01_epsilon", step_count, epsilon);
    wxGetApp().logScalar("21_agent/02_reward", step_count, reward);
    wxGetApp().logScalar("21_agent/03_q_sa", step_count, q_sa.item<double>());
    if (use_td_clip)
        wxGetApp().logScalar("22_agent/04_td_error_raw", step_count, td_raw.item<float>());
    wxGetApp().logScalar("22_agent/05_td_error", step_count, td.item<float>());
    wxGetApp().logScalar("23_agent/06_loss", step_count, loss.item<float>());
    wxGetApp().logScalar("23_agent/07_loss_ema", step_count, loss_ema);
    wxGetApp().logScalar("24_agent/08_grad_norm", step_count, total_norm);
    wxGetApp().logScalar("24_agent/09_grad_clip", step_count, grad_norm_clipped);
    wxGetApp().logScalar("24_agent/10_grad_clip_ema", step_count, grad_norm_clipped_ema);
    //    wxGetApp().logScalar("2_agent/done",     step_count, done);
    //    wxGetApp().logScalar("2_agent/hard_update_done",     step_count, hard_update_done);
}


