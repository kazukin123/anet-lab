#include "RLAgent.hpp"
#include <iostream>
#include <tuple>
#include "app.hpp"
#include "CartPoleEnv.hpp"
#include "tensor_utils.hpp"

using namespace anet::util;

// ==== ハイパーパラメータ ====
struct RLAgent::Param {

    float alpha = 1e-3f;   // 学習率 1e-3 3e-3 1e-4 1e-4 3e-4 5e-4
    float gamma = 0.99f;   // 0.99f; 0.995f      γが高いほど「長期安定」を目指す
    float eps_max = 1.00f;
    float eps_min = 0.05f;    //0.1f 0.05f
    float eps_decay_step = 100000;
    float softupdate_tau = 0.015f;// 1.0f 0.004f  0.01f 0.005f;   // 大きいとターゲットネットワークからの反映が早くなる。小さいと遅く滑らかになる。0.005→半減期138step
    int hardupdate_step = 2000;// -1 5000; //200 500 1000
    float grad_clip_tau = 30.0f;   // 10~40 1f 5f 10f
    bool use_td_clip = true;
    float td_clip_value = 4.0f;
    int eps_zero_step = -1;// 120000;
    bool use_double_dqn = true;   // Double DQN 有効化フラグ（trueで有効）

    bool use_replay_buffer = false;   // ← ON/OFF切替
    int replay_capacity = 50000;
    int replay_batch_size = 64;
    int replay_warmup_steps = 1000;
    int replay_update_interval = 4;


    RLAgent::Param(Properties* props) {
        if (props == NULL) return;
        std::string group = props->Read("agent.preset", "agent");
        ANET_READ_PROPS(props, group, alpha);
        ANET_READ_PROPS(props, group, gamma);
        ANET_READ_PROPS(props, group, eps_max);
        ANET_READ_PROPS(props, group, eps_min);
        ANET_READ_PROPS(props, group, eps_decay_step);
        ANET_READ_PROPS(props, group, softupdate_tau);
        ANET_READ_PROPS(props, group, hardupdate_step);
        ANET_READ_PROPS(props, group, grad_clip_tau);
        ANET_READ_PROPS(props, group, use_td_clip);
        ANET_READ_PROPS(props, group, td_clip_value);
        ANET_READ_PROPS(props, group, eps_zero_step);
        ANET_READ_PROPS(props, group, use_double_dqn);
        ANET_READ_PROPS(props, group, use_replay_buffer);
        ANET_READ_PROPS(props, group, replay_capacity);
        ANET_READ_PROPS(props, group, replay_batch_size);
        ANET_READ_PROPS(props, group, replay_warmup_steps);
        ANET_READ_PROPS(props, group, replay_update_interval);
    }

    std::string toString() {

    }
};

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
RLAgent::RLAgent(int state_dim, int n_actions, torch::Device device) :
    n_actions_(n_actions),
    policy_net(std::make_shared<QNetImpl>(state_dim, n_actions)),
    target_net(std::make_shared<QNetImpl>(state_dim, n_actions)),
    device(device),

    param_(std::make_unique<RLAgent::Param>(wxGetApp().GetConfig())),   // 設定からパラメータを読み込み

    optimizer(policy_net->parameters(), torch::optim::AdamOptions(param_->alpha)),
    replay_buffer(param_->replay_capacity),

    epsilon(1.0f),
    train_step(0)
{
    // NN初期化
    policy_net->to(device);
    target_net->to(device);
    target_net->eval();

    // 初期同期：policy → target
    torch::serialize::OutputArchive archive;
    policy_net->save(archive);
    torch::serialize::InputArchive in;
    std::stringstream ss;
    archive.save_to(ss);
    in.load_from(ss);
    target_net->load(in);
    target_net->eval();

    // ログ：パラメータ記録
    nlohmann::json params = {
        {"alpha", param_->alpha},
        {"gamma", param_->gamma},
        {"eps_max", param_->eps_max},
        {"eps_min", param_->eps_min},
        {"eps_decay_step", param_->eps_decay_step},
        {"eps_zero_step", param_->eps_zero_step},
        {"softupdate_tau", param_->softupdate_tau},
        {"hardupdate_step", param_->hardupdate_step},
        {"grad_clip_tau", param_->grad_clip_tau},
        {"use_td_clip", param_->use_td_clip},
        {"td_clip_value", param_->td_clip_value},
        {"use_double_dqn", param_->use_double_dqn},
        {"use_replay_buffer", param_->use_replay_buffer},
        {"replay_capacity", param_->replay_capacity},
        {"replay_batch_size", param_->replay_batch_size},
        {"replay_warmup_steps", param_->replay_warmup_steps},
        {"replay_update_interval", param_->replay_update_interval},
    };
    wxGetApp().logJson("agent/params", params);
    wxGetApp().flushMetricsLog();
}

// ======================================================
// SelectAction：行動選択（ε-greedy）
// ======================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RLAgent::SelectAction(const torch::Tensor& state, anet::rl::RunMode mode) {
    torch::NoGradGuard ng;
    policy_net->eval();

    // 評価モード：常にgreedy
    if (anet::rl::IsEval(mode)) {
        auto q_values = (mode == anet::rl::RunMode::Eval1)
            ? target_net->forward(state.to(device))  // Eval1:ターゲットネットワークで評価
            : policy_net->forward(state.to(device)); // Eval2:メインネットワークで評価
        auto result = q_values.max(1);
        auto action = std::get<1>(result).to(torch::kLong);
        return { action, torch::Tensor(), torch::Tensor() };
    }

    // Train：ε-greedy
    if (param_->eps_zero_step > 0 && (train_step > param_->eps_zero_step)) {
        epsilon = 0.0f;
    }
    else {
        epsilon = std::max(param_->eps_min, param_->eps_max - static_cast<float>(train_step) / param_->eps_decay_step);
    }

    if (static_cast<float>(rand()) / RAND_MAX < epsilon) {
        // 確率εがヒットした場合はランダムでActionを決定
        int action_int = rand() % n_actions_;
        auto action = torch::tensor({ action_int }, torch::kLong).to(device);
        return { action, torch::Tensor(), torch::Tensor() };
    }
    else {
        // メインネットワークを元にActionを決定
        auto q_values = policy_net->forward(state.to(device));
        auto result = q_values.max(1);
        auto action = std::get<1>(result).to(torch::kLong);
        return { action, torch::Tensor(), torch::Tensor() };
    }
}

// ======================================================
// hard_update / soft_update（hard → soft の順）
// ======================================================
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
    if (tau >= 1.0f) { hard_update(); return; }  // policyを完全コピー
    torch::NoGradGuard no_grad;
    auto src = policy_net->named_parameters();
    auto dst = target_net->named_parameters();
    for (auto& kv : src) {
        if (dst.contains(kv.key())) {
            dst[kv.key()].copy_(dst[kv.key()] * (1.0f - tau) + kv.value() * tau);
        }
    }
}

//  B:バッチサイズ, A:アクション数
// state_t     入力状態 (1,4)
// q_values    Q値      (1,2)
// max_next_q  次状態の最大Q (B,)（通常(1,)）
// reward_t    報酬     (B,)
// nonterminal 未終端=1/終端=0 (B,)
// expected_q  教師Q    (B,)
// q_sa        選択行動のQ (B,)


// ======================================================
// Update：逐次更新 or ReplayBufferモード切替
// ======================================================
void RLAgent::Update(const anet::rl::Experience& exprence) {
    if (!param_->use_replay_buffer) {
        OptimizeSingle(exprence);
        train_step++;
        return;
    }

    replay_buffer.Push(exprence);
    if (replay_buffer.Size() < static_cast<size_t>(param_->replay_warmup_steps)) {
        train_step++;
        return;
    }

    if (train_step % param_->replay_update_interval == 0) {
        auto batch = replay_buffer.Sample(param_->replay_batch_size);
        OptimizeBatch(batch);
    }

    train_step++;
}

void RLAgent::UpdateBatch(const anet::rl::BatchData& batch) {
    OptimizeBatch(batch.Data());
}

// ======================================================
// OptimizeSingle：ReplayBuffer無効時（逐次学習）
// ======================================================
void RLAgent::OptimizeSingle(const anet::rl::Experience& exprence) {
    policy_net->train();

    // 簡易LRスケジュール
    if (train_step == 120000 || train_step == 180000) {
        for (auto& p : optimizer.param_groups()) {
            p.options().set_lr(p.options().get_lr() * 0.5);
        }
    }

    TensorContext ctx(device);

    // --- Q(s, a) の抽出
    auto state_t = exprence.state;
    if (state_t.dim() == 1) state_t = state_t.unsqueeze(0);
    auto q_values = policy_net->forward(state_t.to(device));                          // (1,A)
    auto action_t = torch::tensor({ exprence.action.item<int>() }, ctx.LongOpt());
    auto q_sa = q_values.gather(1, action_t.unsqueeze(1)).squeeze(1).squeeze(0);     // (B,)

    // --- 期待Qの算出（Double DQN対応／現状維持）
    auto next_state = exprence.response.next_state.to(device);
    torch::Tensor max_next_q;
    if (param_->use_double_dqn) {
        auto next_q_policy = policy_net->forward(next_state);                          // (B, A)
        auto next_action = std::get<1>(next_q_policy.max(1));                          // (B,)
        auto next_q_target = target_net->forward(next_state);                          // (B, A)
        max_next_q = next_q_target.gather(1, next_action.unsqueeze(1)).squeeze(1).detach(); // (B,)
    }
    else {
        auto next_q_targ = target_net->forward(next_state);                            // (B, A)
        max_next_q = std::get<0>(next_q_targ.max(1)).detach();                         // (B,)
    }

    // 報酬・終端マスク（テンソルでGPU対応）
    auto reward_t = FullLike(max_next_q, exprence.response.reward, ctx);               // (B,)
    auto done_b = BoolFullLike(max_next_q, exprence.response.done, ctx);               // (B,)
    auto trunc_b = BoolFullLike(max_next_q, exprence.response.truncated, ctx);         // (B,)
    auto absorbing_b = done_b & (~trunc_b);                                            // (B,)
    auto nonterminal = 1.0f - absorbing_b.to(torch::kFloat);                           // (B,)

    auto expected_q = reward_t + param_->gamma * max_next_q * nonterminal;                     // (B,)
    auto td_raw = expected_q - q_sa;                                                   // (B,)
    auto td = param_->use_td_clip ? td_raw.clamp(-param_->td_clip_value, param_->td_clip_value) : td_raw;      // (B,)

    // --- 損失計算（正） ---
    auto loss = torch::nn::functional::smooth_l1_loss(
        q_sa, expected_q.detach(),
        torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kMean)
    );

    // --- 逆伝播
    optimizer.zero_grad();
    loss.backward();

    // --- 勾配ノルム測定 & 勾配クリッピング（Gradient Clipping）---
    float total_norm = static_cast<float>(torch::nn::utils::clip_grad_norm_(policy_net->parameters(), param_->grad_clip_tau));

    // メインネットワークに勾配反映
    optimizer.step();

    // --- メトリクス生成・更新 ---
    const float ema_decay = 0.995f;  // 平滑化係数
    float grad_norm_clipped = (total_norm > param_->grad_clip_tau) ? 1.0f : 0.0f;    // クリップ発動
    grad_norm_clipped_ema = 0.9f * grad_norm_clipped_ema + 0.1f * grad_norm_clipped;
    loss_ema = ema_decay * loss_ema + (1 - ema_decay) * loss.item<float>();
    auto q_targ = target_net->forward(state_t.to(device));                                       // (1,A)
    auto q_diff = torch::mean(torch::abs(q_sa - q_targ.gather(1, action_t.unsqueeze(1)).squeeze(1)));

    float td_cliped = 0.0f;
    if (param_->use_td_clip) {
        float abs_raw = std::abs(td_raw.item<float>());
        td_cliped = (abs_raw > param_->td_clip_value) ? 1.0f : 0.0f;
        td_clip_ema = ema_decay * td_clip_ema + (1 - ema_decay) * td_cliped;
    }

	// メトリクス記録
    wxGetApp().logScalar("21_agent/01_epsilon", train_step, epsilon);               //εグリーディーのε

    wxGetApp().logScalar("22_agent/02_q_sa", train_step, q_sa.item<double>());      // 
    wxGetApp().logScalar("22_agent/03_q_diff", train_step, q_diff.item<float>());   // policy と target の Q値乖離

    if (param_->use_td_clip) {
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
    if (param_->softupdate_tau > 0) {
        soft_update(param_->softupdate_tau);
    }
    if (param_->hardupdate_step > 0 && (train_step % param_->hardupdate_step) == 0) {
        hard_update();      // --- hard update（定期stepごとに完全同期） ---
    }
}

// ======================================================
// OptimizeBatch：ReplayBuffer有効時（バッチ学習本体）
// ======================================================
void RLAgent::OptimizeBatch(const std::vector<anet::rl::Experience>& batch) {
    if (batch.empty()) return;

    policy_net->train();
    TensorContext ctx(device);

    // --- バッチ展開 ---
    std::vector<torch::Tensor> states, next_states, actions, rewards, dones, truncs;
    for (const auto& e : batch) {
        states.push_back(e.state);
        actions.push_back(e.action.view({ 1 }));  // (1,)
        next_states.push_back(e.response.next_state);
        // shape=(1,) 明示的に指定（tensorより安全）
        rewards.push_back(torch::full({ 1 }, e.response.reward, ctx.FloatOpt()));
        dones.push_back(torch::full({ 1 }, e.response.done ? 1.0f : 0.0f, ctx.FloatOpt()));
        truncs.push_back(torch::full({ 1 }, e.response.truncated ? 1.0f : 0.0f, ctx.FloatOpt()));
    }

    auto state_b = torch::cat(states).to(device);               // (B,state_dim)
    auto action_b = torch::cat(actions).view({ (int64_t)actions.size() }).to(device); // (B,)
    auto next_state_b = torch::cat(next_states).to(device);     // (B,state_dim)

    // squeeze() を安全に（次元指定なし）
    auto reward_b = torch::cat(rewards).squeeze().to(device);   // (B,)
    auto done_b = torch::cat(dones).squeeze().to(device);       // (B,)
    auto trunc_b = torch::cat(truncs).squeeze().to(device);     // (B,)
    auto nonterminal = 1.0f - (done_b * (1.0f - trunc_b));      // (B,)

    // --- Q(s,a) ---
    auto q_values = policy_net->forward(state_b);               // (B,A)
    auto q_sa = q_values.gather(1, action_b.unsqueeze(-1)).squeeze(-1); // (B,)


    // --- 期待Q（Double DQN対応） ---
    torch::Tensor max_next_q;
    if (param_->use_double_dqn) {
        auto next_q_policy = policy_net->forward(next_state_b);  // (B,A)
        auto next_action = std::get<1>(next_q_policy.max(1));    // (B,)
        auto next_q_target = target_net->forward(next_state_b);  // (B,A)
        max_next_q = next_q_target.gather(1, next_action.unsqueeze(1)).squeeze(1).detach(); // (B,)
    } else {
        auto next_q_targ = target_net->forward(next_state_b);    // (B,A)
        max_next_q = std::get<0>(next_q_targ.max(1)).detach();   // (B,)
    }

    auto expected_q = reward_b + param_->gamma * max_next_q * nonterminal; // (B,)
    auto td_raw = expected_q - q_sa;                               // (B,)
    auto td = param_->use_td_clip ? td_raw.clamp(-param_->td_clip_value, param_->td_clip_value) : td_raw; // (B,)

    // --- 損失計算 ---
    auto loss = torch::nn::functional::smooth_l1_loss(
        q_sa, expected_q.detach(),
        torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kMean)
    );

    // --- 逆伝播 ---
    optimizer.zero_grad();
    loss.backward();
    float total_norm =
        static_cast<float>(torch::nn::utils::clip_grad_norm_(policy_net->parameters(), param_->grad_clip_tau));
    optimizer.step();

    // --- 統計情報算出（バッチ平均ベース） ---
    const float ema_decay = 0.995f;
    auto q_targ = target_net->forward(state_b); // (B,A)
    auto q_diff = torch::mean(torch::abs(q_sa - q_targ.gather(1, action_b.unsqueeze(1)).squeeze(1)));
    float grad_norm_clipped = (total_norm > param_->grad_clip_tau) ? 1.0f : 0.0f;
    grad_norm_clipped_ema = 0.9f * grad_norm_clipped_ema + 0.1f * grad_norm_clipped;
    loss_ema = ema_decay * loss_ema + (1 - ema_decay) * loss.item<float>();
    float td_cliped = 0.0f;
    if (param_->use_td_clip) {
        auto abs_raw = torch::abs(td_raw);
        td_cliped = torch::mean((abs_raw > param_->td_clip_value).to(torch::kFloat)).item<float>();
        td_clip_ema = ema_decay * td_clip_ema + (1 - ema_decay) * td_cliped;
    }

    // --- メトリクス出力（バッチ平均） ---
    wxGetApp().logScalar("21_agent/01_epsilon", train_step, epsilon);
    wxGetApp().logScalar("22_agent/02_q_sa", train_step, q_sa.mean().item<double>());
    wxGetApp().logScalar("22_agent/03_q_diff", train_step, q_diff.item<float>());
    if (param_->use_td_clip) {
        wxGetApp().logScalar("23_agent/04_td_cliped_ema", train_step, td_clip_ema);
        wxGetApp().logScalar("23_agent/05_td_cliped", train_step, td_cliped);
        wxGetApp().logScalar("23_agent/06_td_error_raw", train_step, td_raw.mean().item<float>());
    }
    wxGetApp().logScalar("23_agent/07_reward", train_step, reward_b.mean().item<float>());
    wxGetApp().logScalar("23_agent/08_td_error", train_step, td.mean().item<float>());
    wxGetApp().logScalar("23_agent/09_loss", train_step, loss.item<float>());
    wxGetApp().logScalar("23_agent/10_loss_ema", train_step, loss_ema);
    wxGetApp().logScalar("24_agent/11_grad_norm", train_step, total_norm);
    wxGetApp().logScalar("24_agent/12_grad_cliped", train_step, grad_norm_clipped);
    wxGetApp().logScalar("24_agent/13_grad_cliped_ema", train_step, grad_norm_clipped_ema);
    wxGetApp().logScalar("25_replay/01_buffer_size", train_step, replay_buffer.Size());

    // --- soft/hard update ---
    if (param_->softupdate_tau > 0) soft_update(param_->softupdate_tau);
    if (param_->hardupdate_step > 0 && (train_step % param_->hardupdate_step) == 0) hard_update();
}
