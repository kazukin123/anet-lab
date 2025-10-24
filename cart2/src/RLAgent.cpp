#include "RLAgent.hpp"
#include <iostream>
#include <tuple>
#include "app.hpp"
#include "CartPoleEnv.hpp"


const float alpha = 1e-3f;   // 学習率 1e-3 3e-3 1e-4 1e-4 3e-4 5e-4
const float gamma = 0.99f;// 0.99f; 0.995f      γが高いほど「長期安定」を目指す
const float eps_max = 1.00f;
const float eps_min = 0.05f;    //0.1f 0.05f
const float eps_decay_step = 100000;
const float softupdate_tau = 0.015f;// 1.0f 0.004f  0.01f 0.005f;   // 大きいとターゲットネットワークからの反映が早くなる。小さいと遅く滑らかになる。0.005→半減期138step
const int hardupdate_step = 2000;// -1 5000; //200 500 1000
const float grad_clip_tau = 30.0f;   // 10~40 1f 5f 10f
const bool use_td_clip = true;
const float td_clip_value = 4.0f;
const int eps_zero_step = -1;// 120000;
const bool use_double_dqn = true;   // Double DQN 有効化フラグ（trueで有効）

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
    train_step(0)
{
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
        {"softupdate_tau", softupdate_tau},
        {"hardupdate_step", hardupdate_step},
        {"grad_clip_tau", grad_clip_tau},
        {"use_td_clip", use_td_clip},
        {"td_clip_value", td_clip_value},
        {"use_double_dqn", use_double_dqn},
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

    if (eps_zero_step > 0 && (train_step > eps_zero_step)) {
        epsilon = 0.0f;
    } else {
        epsilon = std::max(eps_min, eps_max - (float)train_step / eps_decay_step);
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
    if (tau >= 1.0f) {
        hard_update();   // policyを完全コピー
        return;
    }

    torch::NoGradGuard no_grad;

    auto src = policy_net->named_parameters();
    auto dst = target_net->named_parameters();

    for (auto& kv : src) {
        if (dst.contains(kv.key())) {
            dst[kv.key()].copy_(dst[kv.key()] * (1 - tau) + kv.value() * tau);
        }
    }
}

//  B:バッチサイズ        A：アクション数
//state_t	    状態入力(x, x_dot, θ, θ_dot)   (1, 4)
//q_values    	各アクションのQ値                (1, 2)
//max_next_q	各サンプルの「最大Q値」          (B, )（通常(1, )）
//reward_t    	報酬（各サンプル）               (B, )
//nonterminal	「未終端なら1、終端なら0」マスク (B, )
//expected_q	教師信号Q値                      (B, )
//q_sa      	実際に選んだActionのQ値          (B, )

void RLAgent::Update(const anet::rl::Experience& exprence) {    // A=2
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

    auto q_values = policy_net->forward(state_t.to(device));                          // (1,A)
    auto action_t = torch::tensor({ exprence.action.item<int>() }, torch::kLong).to(device);
    auto q_sa = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1).squeeze(0);

    // 教師信号（ターゲットQ）

    // --- 期待Qの算出 ---
    auto next_state = exprence.response.next_state.to(device);
    torch::Tensor max_next_q;
    if (use_double_dqn) {
        // ===== Double DQN: 次状態は policy で行動選択し、target で評価 =====
        auto next_q_policy = policy_net->forward(next_state);               // (B, A)
        auto next_action = std::get<1>(next_q_policy.max(1));               // (B,)
        auto next_q_target = target_net->forward(next_state);               // (B, A)
        max_next_q = next_q_target.gather(1, next_action.unsqueeze(1)).squeeze(1);// (B,)
        max_next_q = max_next_q.detach();
    }
    else {
        // ===== 通常DQN: ターゲットネット単体で max_a' Q_target(s', a') =====
        auto next_q_targ = target_net->forward(next_state);                 // (B, A)
        max_next_q = std::get<0>(next_q_targ.max(1)).detach();              // (B,)
    }

    // デバイス＆dtypeオプション
    auto optsF = torch::TensorOptions().dtype(torch::kFloat).device(device);
    auto optsB = torch::TensorOptions().dtype(torch::kBool).device(device);

    // 報酬テンソル（(B,)に合うサイズで作成）
    auto reward_t = torch::full(max_next_q.sizes(), exprence.response.reward, optsF);   // (B,)

    // done / truncated を bool テンソル化（(B,)にブロードキャスト）
    auto done_b = torch::full(max_next_q.sizes(), exprence.response.done, optsB);       // (B,)
    auto trunc_b = torch::full(max_next_q.sizes(), exprence.response.truncated, optsB); // (B,)

    // 「吸収終端」= 失敗終了：done && !truncated
    auto absorbing_b = done_b & (~trunc_b);                                              // (B,)

    // 非終端マスク：吸収終端なら0、そうでなければ1（truncatedは1）
    auto nonterminal = torch::where(
        absorbing_b,
        torch::zeros_like(max_next_q, optsF),
        torch::ones_like(max_next_q, optsF)
    );                                                                                   // (B,)

    // 期待Q
    auto expected_q = reward_t + gamma * max_next_q * nonterminal;                       // (B,)

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

    // --- 勾配ノルム測定 & 勾配クリッピング（Gradient Clipping）---
    float total_norm = static_cast<float>(torch::nn::utils::clip_grad_norm_(policy_net->parameters(), grad_clip_tau));

    // メインネットワークに勾配反映
    optimizer.step();

    // メトリクス生成・更新
    const float ema_decay = 0.995f;  // 平滑化係数
    float grad_norm_clipped = (total_norm > grad_clip_tau) ? 1.0f : 0.0f;    // 勾配ノルムがクリッピングしきい値を超えたか
    grad_norm_clipped_ema = (0.9f * grad_norm_clipped_ema) + 0.1f * grad_norm_clipped;
    loss_ema = ema_decay * loss_ema + (1 - ema_decay) * loss.item<float>();
    auto q_targ = target_net->forward(state_t.to(device));
    auto q_diff = torch::mean(torch::abs(q_sa - q_targ.gather(1, action_t.unsqueeze(1)).squeeze(1)));
    float td_cliped = 0.0f;
    if (use_td_clip) {
        float abs_raw = std::abs(td_raw.item<float>());
        td_cliped = (abs_raw > td_clip_value) ? 1.0f : 0.0f;
        td_clip_ema = ema_decay * td_clip_ema + (1 - ema_decay) * td_cliped;
    }

	// メトリクス記録
    wxGetApp().logScalar("21_agent/01_epsilon", train_step, epsilon);               //εグリーディーのε

    wxGetApp().logScalar("22_agent/02_q_sa", train_step, q_sa.item<double>());      // 
    wxGetApp().logScalar("22_agent/03_q_diff", train_step, q_diff.item<float>());   // policy と target の Q値乖離

    if (use_td_clip) {
        wxGetApp().logScalar("23_agent/04_td_cliped_ema", train_step, td_clip_ema); // TD誤差 クリップ前
        wxGetApp().logScalar("23_agent/05_td_cliped", train_step, td_cliped); // TD誤差 クリップ前
        wxGetApp().logScalar("23_agent/06_td_error_raw", train_step, td_raw.item<float>()); // TD誤差 クリップ前
    }
    wxGetApp().logScalar("23_agent/07_reward", train_step, exprence.response.reward);   // 報酬
    wxGetApp().logScalar("23_agent/08_td_error", train_step, td.item<float>());         // TD誤差（TD誤差クリップ有りの場合はクリップ後）
    wxGetApp().logScalar("23_agent/09_loss", train_step, loss.item<float>());           // loss値
    wxGetApp().logScalar("23_agent/10_loss_ema", train_step, loss_ema);                 // loss値のEMA移動平均

    wxGetApp().logScalar("24_agent/11_grad_norm", train_step, total_norm);              // 勾配ノルム
    wxGetApp().logScalar("24_agent/12_grad_cliped", train_step, grad_norm_clipped);       // 勾配ノルムがクリッピングされたか
    wxGetApp().logScalar("24_agent/13_grad_cliped_ema", train_step, grad_norm_clipped_ema);   // 勾配ノルムのクリッピング率（EMA移動平均）
    //    wxGetApp().logScalar("2_agent/done",     train_step, done);
    //    wxGetApp().logScalar("2_agent/hard_update_done",     train_step, hard_update_done);


        // --- soft update（毎回少し近づける） ---
    if (softupdate_tau > 0) {
        soft_update(softupdate_tau);
    }
    bool hard_update_done = false;
    if (hardupdate_step > 0 && (train_step % hardupdate_step) == 0) {
        hard_update();      // --- hard update（定期stepごとに完全同期） ---
        wxLogInfo("Hard update done. step=%d", train_step);
        hard_update_done = true;
    }

    train_step++;

}


