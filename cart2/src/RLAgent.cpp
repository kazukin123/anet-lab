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
const float tnetup_softupdate_tau = 0.004f;//  0.01f 0.005f;   // 大きいとターゲットネットワークからの反映が早くなる。小さいと遅く滑らかになる。0.005→半減期138step
const int tnetup_hardupdate_step = -1;// 5000; //200 500 1000
const float grad_clip_tau = 30.0f;   // 1f 5f 10f
const bool use_td_clip = false;
const float td_clip_value = 3.0f;
const int eps_zero_step = -1;// 120000;

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
    train_step(0),
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
    target_net->eval(); // ターゲットネットは期待Q値の評価だけで更新はしないはず

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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RLAgent::SelectAction(const torch::Tensor& state, anet::rl::RunMode mode) {
    torch::NoGradGuard ng;
    policy_net->eval();

    if (anet::rl::IsEval(mode)) { // 評価モードではε-greedyを無効化し、常に最大Q値のActionを選択
        auto q_values = (mode == anet::rl::RunMode::Eval1)
            ? target_net->forward(state.to(device)):    // Eval1:ターゲットネットワークで評価
              policy_net->forward(state.to(device));    // Eval2:メインネットワークで評価
        auto result = q_values.max(1);
        auto max_index = std::get<1>(result);
        auto action = max_index.to(torch::kLong);
        return { action, torch::Tensor(), torch::Tensor() };
        //学習済みステップ数やepsilonの更新もしない
	}

    train_step++;

    if (eps_zero_step > 0 && (train_step > eps_zero_step)) {
        epsilon = 0.0f;
    }
    else {
        epsilon = std::max(eps_min, eps_max - train_step / eps_decay_step);
        //epsilon = std::max(0.05f, epsilon * 0.9998f);
        //epsilon = 1.0f;
    }

    if (((float)rand() / RAND_MAX) < epsilon) {
        // 確率εがヒットした場合はランダムでActionを決定
        int action_int = rand() % n_actions_;
        //int action = 0;
        auto action = torch::tensor({ action_int }, torch::kLong).to(device);
        return { action, torch::Tensor(), torch::Tensor() };
    }
    else {
        // メインネットワークを元にActionを決定
        auto q_values = policy_net->forward(state.to(device));
        auto result = q_values.max(1);
        auto max_index = std::get<1>(result);
        auto action = max_index.to(torch::kLong);
        return { action, torch::Tensor(), torch::Tensor() };
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

//struct StepData {
//    torch::Tensor state;
//    torch::Tensor action;
//    torch::Tensor next_state;
//    float reward;
//    bool done;
//    bool truncated;
//};

void RLAgent::Update(const anet::rl::Experience& exprence) {
    policy_net->train();

    // 簡易 LR スケジュール（終盤で半減）
    if (train_step == 120000 || train_step == 180000) { // 120k 180k で学習率半減
        for (auto& p : optimizer.param_groups()) {
            p.options().set_lr(p.options().get_lr() * 0.5);
        }
    }

    auto state_t = exprence.state;
    if (state_t.dim() == 1)
        state_t = state_t.unsqueeze(0);

    auto q_values = policy_net->forward(state_t.to(device));  // shape=(1,A)
    auto action_t = torch::tensor({ exprence.action.item<int>() }, torch::kLong).to(device);
    auto q_sa = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1).squeeze(0);


    // 教師信号（ターゲットQ）
    auto next_q_targ = target_net->forward(exprence.response.next_state.to(device));  // (B,A)
    auto max_next_q = std::get<0>(next_q_targ.max(1)).detach();    // (B,)
    auto reward_t = torch::full({ 1 }, exprence.response.reward, torch::TensorOptions().device(device));
    auto done_t = torch::tensor(exprence.response.done ? 0.0f : 1.0f, torch::TensorOptions().device(device));
    auto expected_q = reward_t + gamma * max_next_q * done_t;

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

    // メトリクス生成・更新
    const float ema_decay = 0.995f;  // 平滑化係数
    float grad_norm_clipped = (total_norm > grad_clip_tau) ? 1.0f : 0.0f;    // 勾配ノルムがクリッピングしきい値を超えたか
    grad_norm_clipped_ema = (0.9f * grad_norm_clipped_ema) + 0.1f * grad_norm_clipped;
    loss_ema = ema_decay * loss_ema + (1 - ema_decay) * loss.item<float>();
    auto q_targ = target_net->forward(state_t.to(device));
    auto q_diff = torch::mean(torch::abs(q_sa - q_targ.gather(1, action_t.unsqueeze(1)).squeeze(1)));

	// メトリクス記録
    wxGetApp().logScalar("21_agent/01_epsilon", train_step, epsilon);               //εグリーディーのε

    wxGetApp().logScalar("22_agent/02_q_sa", train_step, q_sa.item<double>());      // 
    wxGetApp().logScalar("22_agent/03_q_diff", train_step, q_diff.item<float>());   // policy と target の Q値乖離

    wxGetApp().logScalar("23_agent/04_reward", train_step, exprence.response.reward);   // 報酬
    wxGetApp().logScalar("23_agent/05_td_error", train_step, td.item<float>());         // TD誤差（TD誤差クリップ有りの場合はクリップ後）
    if (use_td_clip)
        wxGetApp().logScalar("23_agent/06_td_error_raw", train_step, td_raw.item<float>()); // TD誤差 クリップ前
    wxGetApp().logScalar("23_agent/07_loss", train_step, loss.item<float>());           // loss値
    wxGetApp().logScalar("23_agent/08_loss_ema", train_step, loss_ema);                 // loss値のEMA移動平均

    wxGetApp().logScalar("24_agent/09_grad_norm", train_step, total_norm);              // 勾配ノルム
    wxGetApp().logScalar("24_agent/10_grad_clip", train_step, grad_norm_clipped);       // 勾配ノルムがクリッピングされたか
    wxGetApp().logScalar("24_agent/11_grad_clip_ema", train_step, grad_norm_clipped_ema);   // 勾配ノルムのクリッピング率（EMA移動平均）
    //    wxGetApp().logScalar("2_agent/done",     train_step, done);
    //    wxGetApp().logScalar("2_agent/hard_update_done",     train_step, hard_update_done);

        // --- soft update（毎回少し近づける） ---
    if (tnetup_softupdate_tau > 0) {
        soft_update(tnetup_softupdate_tau);
    }
    bool hard_update_done = false;
    if (tnetup_hardupdate_step > 0 && (train_step % tnetup_hardupdate_step) == 0) {
        hard_update();      // --- hard update（定期stepごとに完全同期） ---
        wxLogInfo("Hard update done. step=%d", train_step);
        hard_update_done = true;
    }

}


