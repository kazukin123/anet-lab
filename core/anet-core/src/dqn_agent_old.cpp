#include "anet/dqn_agent.hpp"
#include <iostream>
#include <tuple>
#include <wx/log.h>
#include "nlohmann/json.hpp"
#include "anet/tensor_utils.hpp"
#include "anet/tensor_check.hpp"
#include "anet/config.hpp"
#include "anet/metrics_logger.hpp"

using namespace anet::rl;

const float met_ema_decay = 0.995f;  // 平滑化係数(メトリクス用)
const float met_ema_decay_act = 0.9995f;  // 平滑化係数(メトリクス用)action_ema用
const float met_ema_decay_reward = 0.9995f;  // 平滑化係数(メトリクス用)action_ema用

namespace {
    static constexpr int64_t ANY = ANET_SHAPE_ANY;
}

// ======================================================
// QNet 定義（Impl を CPP に置く）
// ======================================================
struct anet::rl::DQNAgent::QNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };

    QNetImpl(int state_dim, int n_actions) {
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, 120));
        fc2 = register_module("fc2", torch::nn::Linear(120, 84));
        fc3 = register_module("fc3", torch::nn::Linear(84, n_actions));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }
};

// ======================================================
// DQNAgent 実装
// ======================================================
DQNAgent::DQNAgent(const DQNAgentConfig& config, anet::rl::Environment& env, int state_dim, int n_actions, torch::Device device) :
    state_dim_(state_dim),
    n_actions_(n_actions),
    policy_net(std::make_shared<QNetImpl>(state_dim, n_actions)),
    target_net(std::make_shared<QNetImpl>(state_dim, n_actions)),
    device(device),
    config_(config),
    optimizer(policy_net->parameters(), torch::optim::AdamOptions(config_.alpha)),
    replay_buffer(config_.replay_capacity)
{
    // 学習変数初期化
    tau_ = config_.softupdate_tau;
    epsilon = config_.eps_max;
    eps_reheat_floor_ = config_.eps_min;
    // EMA係数（半減期 → α = ln(2)/H）
    uema_alpha_ = (config_.uema_half_life > 0.0f)
        ? static_cast<float>(std::log(2.0) / config_.uema_half_life)
        : 1.0f; // 念のため（0割回避）

    // ヒートマップオブジェクトを生成
	auto nan = std::numeric_limits<float>::quiet_NaN();
    auto info = env.GetStateSpaceInfo();
    auto flags = anet::HeatMapFlags::HM_LogScaleValue | anet::HeatMapFlags::HM_AutoNormValue
		| anet::HeatMapFlags::HM_AutoScaleAxis | anet::HeatMapFlags::HM_LogScaleAxis | anet::HeatMapFlags::HM_ShowZeroLine;
    heatmap_visit1_ = anet::rl::MakeStateHeatMapPtr(info, 0, 2, 256, 256, 30000, flags | anet::HeatMapFlags::HM_SumMode);  // x vs theta → reward
    heatmap_visit2_ = anet::rl::MakeStateHeatMapPtr(info, 2, 3, 256, 256, 30000, flags | anet::HeatMapFlags::HM_SumMode);  // x vs theta → reward
    heatmap_td_     = anet::rl::MakeStateHeatMapPtr(info, 0, 2, 256, 256, 30000, flags | anet::HeatMapFlags::HM_MeanMode); // x vs theta → td
    hist_action_ = std::make_unique<anet::TimeHistogram>(
        2, 200, anet::TimeFrameMode::Scroll, flags, -1.0f, 1.0f, 0.05f);
    hist_q_ = std::make_unique<anet::TimeHistogram>(
        128, 1000, anet::TimeFrameMode::Unlimited, flags | anet::HeatMapFlags::HM_FlipY, 0.0f, nan, 0.05f);

    //TimeHistogram(int bins, int max_frames,
    //    TimeFrameMode mode = TimeFrameMode::Scroll,
    //    uint32_t flags = HM_AutoScaleAxis | HM_AutoNormValue,
    //    float alpha = 0.05f,
    //    float base_min = std::numeric_limits<float>::quiet_NaN(),
    //    float base_max = std::numeric_limits<float>::quiet_NaN()
    //);

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
    wxLogInfo("DQNAgent config=%s", config_.ToStdString());
    anet::MetricsLogger::Instance()->log_json("agent/params", config_.ToJson());
    anet::MetricsLogger::Instance()->flush();
}

// ======================================================
// SelectAction：行動選択（ε-greedy）
// ======================================================
anet::rl::ActionResult DQNAgent::SelectAction(const torch::Tensor& state, anet::rl::RunMode mode) {
    ANET_CHECK_DEVICE_CPU(state);
    ANET_CHECK_SHAPE(state, { ANY, 4 });

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
    if (!config_.use_as_dqn) {
        float eps_base;
        if (config_.eps_zero_step > 0 && (train_step > config_.eps_zero_step)) {
            eps_base = 0.0f;
        } else {
            eps_base = std::max(config_.eps_min, config_.eps_max - static_cast<float>(train_step) / config_.eps_decay_step);
        }
        epsilon = std::clamp(eps_base * eps_boost_, config_.eps_min, config_.eps_max);
    }

    torch::Tensor action;
    if (static_cast<float>(rand()) / RAND_MAX < epsilon) {
        // 確率εがヒットした場合はランダムでActionを決定
        int action_int = rand() % n_actions_;
        action = torch::tensor({ action_int }, torch::kLong).to(device);
    } else {
        // メインネットワークを元にActionを決定
        auto q_values = policy_net->forward(state.to(device));    // (1, A)
        auto result = q_values.max(1);      // tuple (values, indices)  // (B,) (B,)
        action = std::get<1>(result).to(torch::kLong); //indices を取得してlong型に変換  // (B,)
    }

    // 選択されたactionをメトリクスに記録（EMA）
    float action_float = (action.numel() == 1) ?
        static_cast<float>(action.item<int>()) :    // 単一
        action.to(torch::kFloat32).mean().item<float>();    // バッチ（将来用）
    action_ema = met_ema_decay_act * action_ema + (1 - met_ema_decay_act) * action_float;
    anet::MetricsLogger::Instance()->log_scalar("34_agent_as_a/02_action_ema", train_step, action_ema);

    wxLogDebug("SelectAction() action=%s", anet::ToString(action));

    return { action, torch::Tensor(), torch::Tensor() };
}

// ======================================================
// hard_update / soft_update（hard → soft の順）
// ======================================================
void DQNAgent::hard_update() {
    torch::NoGradGuard no_grad;
    auto src = policy_net->named_parameters();
    auto dst = target_net->named_parameters();
    for (auto& kv : src) {
        if (dst.contains(kv.key())) {
            dst[kv.key()].copy_(kv.value());
        }
    }
}

void DQNAgent::soft_update() {
    if (tau_ >= 1.0f) { hard_update(); return; }  // policyを完全コピー
    torch::NoGradGuard no_grad;
    auto src = policy_net->named_parameters();
    auto dst = target_net->named_parameters();
    for (auto& kv : src) {
        if (dst.contains(kv.key())) {
            dst[kv.key()].copy_(dst[kv.key()] * (1.0f - tau_) + kv.value() * tau_);
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
std::shared_ptr<anet::rl::UpdateResult> DQNAgent::UpdateStep(const anet::rl::Experience& exprence) {
    train_step++;

    if (!config_.use_replay_buffer) {
        OptimizeSingle(exprence);
    } else {
        replay_buffer.Push(exprence);
        if (replay_buffer.Size() < static_cast<size_t>(config_.replay_warmup_steps)) {
            //return;
        } else if (train_step % config_.replay_update_interval == 0) {
            auto batch = replay_buffer.Sample(config_.replay_batch_size);   // TODO: UpdateBatch へ完全移行したら Sample() は廃止する
            OptimizeBatch(batch);
        }
    }

    // --- εブースト自然減衰（毎step適用） ---
    if (config_.use_as_dqn) {
        // εブーストの自然減衰
        // eps_boost_half_life_recover に基づき実効step補正
        float decay = std::exp(-std::log(2.0f) * config_.replay_update_interval / config_.eps_boost_half_life_recover);
        eps_boost_ = 1.0f + (eps_boost_ - 1.0f) * decay;

        // --- ε下限の自然回復 ---
        if (eps_reheat_floor_ > config_.eps_min) {
            const float k = std::exp(-std::log(2.0f) / config_.eps_reheat_half_life);
            eps_reheat_floor_ = config_.eps_min + (eps_reheat_floor_ - config_.eps_min) * k;
        }

        // --- ε最終値算出（base × boost + add）---
        float eps_base;
        if (config_.eps_zero_step > 0 && (train_step > config_.eps_zero_step)) {
            eps_base = 0.0f;
        } else {
            eps_base = std::max(config_.eps_min, config_.eps_max - static_cast<float>(train_step) / config_.eps_decay_step);
        }
        float eps_mult = eps_base * eps_boost_;
        float eps_add = (eps_boost_ - 1.0f) * config_.eps_gain;
        float eps_raw = eps_mult + eps_add;

        // clamp時にreheat_floorを適用してepsilon値を生成
        epsilon = std::clamp(eps_raw, eps_reheat_floor_, config_.eps_max);

        // 安定状態が続いている場合はτをゆっくり回復
        float recover_factor = std::exp(std::log(2.0f) / config_.tau_half_life_recover);
        if (config_.use_unstable_ema) {
            if (unstable_ema_ < 0.3f) { // 閾値は経験的に 0.2〜0.4
                tau_ = std::min(config_.tau_max, tau_ * recover_factor);
            }
        } else {
            int stable_steps = train_step - last_unstable_step_;
            if (stable_steps > config_.tau_recover_delay) {
                tau_ = std::min(config_.tau_max, tau_ * recover_factor);
            }
        }
        tau_ = std::clamp(tau_, config_.tau_min, config_.tau_max);
    }

    // visitヒートマップ更新
    auto x = exprence.state[0][0].item<float>();
    auto theta = exprence.state[0][2].item<float>();
    auto theta_dot = exprence.state[0][3].item<float>();
    heatmap_visit1_->AddData(x, theta, exprence.response.reward);
    heatmap_visit2_->AddData(theta, theta_dot, exprence.response.reward);

    // ヒートマップ画像保存
    if (config_.heatmap_log_image_interval > 0 && train_step != 0 &&
        (train_step % config_.heatmap_log_image_interval) == 0) {
        anet::MetricsLogger::Instance()->log_image("43_agent_img/02_hm_visit1", train_step, *heatmap_visit1_);
        anet::MetricsLogger::Instance()->log_image("43_agent_img/03_hm_visit2", train_step, *heatmap_visit2_);
        if (!config_.use_replay_buffer)
            anet::MetricsLogger::Instance()->log_image("43_agent_img/04_hm_td", train_step, *heatmap_td_);
    }

    // Q値ヒートマップ生成・保存（CUDA無いと超遅い）
    if (config_.heatmap_log_sweep_interval > 0 && train_step != 0 &&
        (train_step % config_.heatmap_log_sweep_interval) == 0) {
        anet::SweepedHeatMap q_map = anet::SweepedHeatMap::EvaluateTensorFunction(
            128, 128,
            -2.4f, 2.4f,   // x range
            -0.21f, 0.21f, // theta range
            device,
            // forward関数: xy = [x, theta] → [x, x_dot, theta, theta_dot] = [x, 0, theta, 0]
            [&](const torch::Tensor& xy) {
                auto x = xy.index({ torch::indexing::Slice(), 0 }).unsqueeze(1);
                auto theta = xy.index({ torch::indexing::Slice(), 1 }).unsqueeze(1);
                auto x_dot = torch::zeros_like(x);
                auto theta_dot = torch::zeros_like(theta);
                auto s = torch::cat({ x, x_dot, theta, theta_dot }, 1).to(device);
                return policy_net->forward(s);
            },
            // value抽出関数: [Q_right, Q_left] → [ Q_right - Q_left ]
            [&](const torch::Tensor& out) {
                return out.index({ torch::indexing::Slice(), 1 }) -
                    out.index({ torch::indexing::Slice(), 0 });
            });
        anet::MetricsLogger::Instance()->log_image("43_agent_img/05_hm_q", train_step, q_map);
    }

    // メトリクス出力
// --- Warmup期間中にAS-DQN統計と同期して軸を揃える ---
    if (config_.use_as_dqn && replay_buffer.Size() < config_.replay_warmup_steps) {
        anet::MetricsLogger::Instance()->log_scalar("33_agent_as/01_eps_boost", train_step, eps_boost_);
        anet::MetricsLogger::Instance()->log_scalar("34_agent_as_a/01_action_diff", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("34_agent_as_a/03_action_unstable", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/01_q_unstable", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/02_q_mean", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/03_q_max", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/04_q_std", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/05_q_std_ema", train_step, qstd_ema_);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/06_q_z", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/07_q_cv", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/08_q_niqr", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/01_q_unstable", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("36_agent_as_uema/01_uema_unstable", train_step, unstable_ema_);
        anet::MetricsLogger::Instance()->log_scalar("36_agent_as_uema/02_uema_s", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("36_agent_as_uema/03_uema_e_t", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("36_agent_as_uema/04_uema_diff_s_e_t", train_step, 0.0f);
    }
    if (config_.use_as_dqn) {
        anet::MetricsLogger::Instance()->log_scalar("32_agent_dqn_base/01_tau", train_step, tau_);
        anet::MetricsLogger::Instance()->log_scalar("32_agent_dqn_base/02_tau_half_life", train_step, std::log(2.0f) / tau_);
        anet::MetricsLogger::Instance()->log_scalar("33_agent_as/01_eps_reheat_floor", train_step, eps_reheat_floor_);               //εグリーディーのε
    }
    {
        anet::MetricsLogger::Instance()->log_scalar("32_agent_dqn_base/03_epsilon", train_step, epsilon);               //εグリーディーのε
    }

	// 軸合わせ用にQ値ヒストグラムをダミー更新
    if (replay_buffer.Size() < static_cast<size_t>(config_.replay_warmup_steps)) {
		// ウォームアップ中なのでQ値ヒストグラムをダミー更新
        if (train_step % config_.replay_update_interval == 0) {
            // ミニバッチ間隔でフレーム更新
            hist_q_->NextFrame();
            if (config_.heatmap_log_hist_interval > 0 && (train_step % config_.heatmap_log_hist_interval) == 0) {
				// heatmap_log_hist_interval間隔でQ値ヒストグラム画像保存
                anet::MetricsLogger::Instance()->log_image("43_agent_img/06_th_q", train_step, *hist_q_);
            }
        }
    }
    return {};
}

void DQNAgent::OnPostUpdate(const std::shared_ptr<UpdateResult>& result) {
    ;
}

std::shared_ptr<anet::rl::UpdateResult> DQNAgent::UpdateBatch(const anet::rl::BatchData& batch) {
    OptimizeBatch(batch.Data());
    return {};
}

// ======================================================
// OptimizeSingle：ReplayBuffer無効時（逐次学習）
// ======================================================
void DQNAgent::OptimizeSingle(const anet::rl::Experience& exprence) {
    policy_net->train();

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
    if (config_.use_double_dqn) {
        auto next_q_policy = policy_net->forward(next_state);                          // (B, A)
        auto next_action = std::get<1>(next_q_policy.max(1));                          // (B,)
        auto next_q_target = target_net->forward(next_state);                          // (B, A)
        max_next_q = next_q_target.gather(1, next_action.unsqueeze(1)).squeeze(1).detach(); // (B,)
    } else {
        auto next_q_targ = target_net->forward(next_state);                            // (B, A)
        max_next_q = std::get<0>(next_q_targ.max(1)).detach();                         // (B,)
    }

    // 報酬・終端マスク（テンソルでGPU対応）
    auto reward_t = FullLike(max_next_q, exprence.response.reward, ctx);               // (B,)
    auto done_b = BoolFullLike(max_next_q, exprence.response.done, ctx);               // (B,)
    auto trunc_b = BoolFullLike(max_next_q, exprence.response.truncated, ctx);         // (B,)
    auto absorbing_b = done_b & (~trunc_b);                                            // (B,)
    auto nonterminal = 1.0f - absorbing_b.to(torch::kFloat);                           // (B,)

    auto expected_q = reward_t + config_.gamma * max_next_q * nonterminal;             // (B,)
    auto td_raw = expected_q - q_sa;                                                   // (B,)
    auto td = config_.use_td_clip ? td_raw.clamp(-config_.td_clip_value, config_.td_clip_value) : td_raw;      // (B,)

    // TODO: OptimizeSingle()の場合、use_td_clipが動作していない

    // --- 損失計算（正） ---
    auto loss = torch::nn::functional::smooth_l1_loss(
        q_sa, expected_q.detach(),
        torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kMean)
    );

    // --- 逆伝播
    optimizer.zero_grad();
    loss.backward();

    // --- 勾配ノルム測定 & 勾配クリッピング（Gradient Clipping）---
    float total_norm;
    if (config_.use_grad_clip) {
        total_norm = static_cast<float>(
            torch::nn::utils::clip_grad_norm_(policy_net->parameters(), config_.grad_clip_tau)
            );
    }
    else {
        // クリップしない場合は「そのままノルム計測のみ」
        total_norm = 0.0f;
        for (auto& p : policy_net->parameters()) {
            if (p.grad().defined()) {
                total_norm += p.grad().data().norm().item<float>();
            }
        }
    }

    // メインネットワークに勾配反映
    optimizer.step();

    // --- メトリクス生成・更新 ---
    float grad_norm_clipped = (config_.use_grad_clip && total_norm > config_.grad_clip_tau) ? 1.0f : 0.0f;    // クリップ発動
    grad_norm_clipped_ema = 0.9f * grad_norm_clipped_ema + 0.1f * grad_norm_clipped;
    loss_ema = loss_ema_init_ ? met_ema_decay * loss_ema + (1 - met_ema_decay) * loss.item<float>()
        : loss.item<float>();
    auto q_targ = target_net->forward(state_t.to(device));                                       // (1,A)
    auto q_diff = torch::mean(torch::abs(q_sa - q_targ.gather(1, action_t.unsqueeze(1)).squeeze(1)));
    if (!loss_ema_init_) loss_ema_init_ = true;

    float td_cliped = 0.0f;
    if (config_.use_td_clip) {
        float abs_raw = std::abs(td_raw.item<float>());
        td_cliped = (abs_raw > config_.td_clip_value) ? 1.0f : 0.0f;
        td_clip_ema = met_ema_decay * td_clip_ema + (1 - met_ema_decay) * td_cliped;
    }

    // メトリクス記録
    anet::MetricsLogger::Instance()->log_scalar("37_agent_dqn_qtd/01_q_sa", train_step, q_sa.item<double>());      //
    anet::MetricsLogger::Instance()->log_scalar("37_agent_dqn_qtd/02_q_diff", train_step, q_diff.item<float>());   // policy と target の Q値乖離

    if (config_.use_td_clip) {
        anet::MetricsLogger::Instance()->log_scalar("37_agent_dqn_qtd/05_td_cliped_ema", train_step, td_clip_ema); // TD誤差 クリップ前
        anet::MetricsLogger::Instance()->log_scalar("37_agent_dqn_qtd/06_td_cliped", train_step, td_cliped);       // TD誤差 クリップ前
        anet::MetricsLogger::Instance()->log_scalar("37_agent_dqn_qtd/04_td_error_raw", train_step, td_raw.item<float>()); // TD誤差 クリップ前
    }
    anet::MetricsLogger::Instance()->log_scalar("37_agent_dqn_qtd/03_td_error", train_step, td.item<float>());         // TD誤差（TD誤差クリップ有りの場合はクリップ後）
    anet::MetricsLogger::Instance()->log_scalar("38_agent_dqn_loss/01_loss", train_step, loss.item<float>());           // loss値
    anet::MetricsLogger::Instance()->log_scalar("38_agent_dqn_loss/02_loss_ema", train_step, loss_ema);                 // loss値のEMA移動平均

    anet::MetricsLogger::Instance()->log_scalar("39_agent_dqn_grad/01_grad_norm", train_step, total_norm);              // 勾配ノルム
    if (config_.use_grad_clip) {
        anet::MetricsLogger::Instance()->log_scalar("39_agent_dqn_grad/02_grad_cliped", train_step, grad_norm_clipped);       // 勾配ノルムがクリッピングされたか
        anet::MetricsLogger::Instance()->log_scalar("39_agent_dqn_grad/03_grad_cliped_ema", train_step, grad_norm_clipped_ema);   // 勾配ノルムのクリッピング率（EMA移動平均）
    }

    // ヒートマップ更新
    auto x = exprence.state[0][0].item<float>();
    auto theta = exprence.state[0][2].item<float>();
    heatmap_td_->AddData(x, theta, td.item<float>());

    // --- soft update（毎回少し近づける） ---
    if (config_.softupdate_tau > 0) {
        soft_update();
    }
    if (config_.hardupdate_step > 0 && (train_step % config_.hardupdate_step) == 0) {
        hard_update();      // --- hard update（定期stepごとに完全同期） ---
    }
}

// ======================================================
// OptimizeBatch：ReplayBuffer有効時（バッチ学習本体）
// ======================================================
void DQNAgent::OptimizeBatch(const std::vector<anet::rl::Experience>& batch) {
    if (replay_buffer.Size() < config_.replay_warmup_steps) {
        post_warmup_steps_ = 0;
        // （既存の0出力処理やreturnはそのまま）
        return;
    } else {
        post_warmup_steps_++;
    }

    if (batch.empty()) return;

    if (replay_buffer.Size() < config_.replay_warmup_steps) {
        // 初期化された統計をリセットして安全値へ
        unstable_ema_ = 0.1f;
        eps_boost_ = 1.0f;
        eps_reheat_floor_ = config_.eps_min;
        tau_ = config_.softupdate_tau;

        // --- Warmup期間中も0値を出力して可視化軸を統一 ---
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/02_q_mean", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/03_q_max", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/04_q_std", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/04_q_std_ema", train_step, qstd_ema_);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/06_q_z", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/07_q_cv", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/08_q_niqr", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/01_q_unstable", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("34_agent_as_a/03_action_unstable", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("34_agent_as_a/01_action_diff", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("36_agent_as_uema/01_uema_unstable", train_step, unstable_ema_);
        anet::MetricsLogger::Instance()->log_scalar("36_agent_as_uema/02_uema_s", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("36_agent_as_uema/03_uema_e_t", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("36_agent_as_uema/04_uema_diff_s_e_t", train_step, 0.0f);

        return;  // ← AS-DQN部分の更新スキップ
    }

    policy_net->train();
    TensorContext ctx(device);
    const int B = config_.replay_batch_size;

    if (replay_buffer.Size() >= config_.replay_batch_size) {    // バッチ数分以上のサンプルが溜まってる場合のみ計測

        // バッチサイズ分をまとめてサンプリング
        auto samples = replay_buffer.SampleBatch(config_.replay_batch_size, device);
        ANET_CHECK_DEVICE(samples.states, device);
        ANET_CHECK_DEVICE(samples.actions, device);
        ANET_CHECK_DEVICE(samples.rewards, device);
        ANET_CHECK_DEVICE(samples.next_states, device);
        ANET_CHECK_DEVICE(samples.dones, device);
        ANET_CHECK_DEVICE(samples.truncateds, device);
        ANET_ASSERT((samples.actions.dtype() == torch::kInt64));   //DQNでは離散アクション
        ANET_CHECK_SHAPE(samples.states,     { B, state_dim_ });    // (B, state_dim)
        ANET_CHECK_SHAPE(samples.actions,    { B, 1 });             // (B, 1)   DQNでは離散アクション
        ANET_CHECK_SHAPE(samples.rewards,    { B });             // (B,)
        ANET_CHECK_SHAPE(samples.next_states,{ B, state_dim_ });    // (B, state_dim)
        ANET_CHECK_SHAPE(samples.dones,      { B });              // (B,)
        ANET_CHECK_SHAPE(samples.truncateds, { B });              // (B,)

        // ---- action統計 ----
        auto actions_cpu = samples.actions.detach().cpu(); // (B,)
        int count_left = 0;
        int count_right = 0;
        const int a_num = actions_cpu.size(0);

        for (int i = 0; i < a_num; ++i) {
            int a = actions_cpu[i].item<int>();
            if (a == 0) count_left++;
            else if (a == 1) count_right++;                 // ※ それ以外の値は無視（離散2値前提）
        }

        float sum = static_cast<float>(count_left + count_right);
        // sum==0 のときは 0.5/0.5（視覚上ニュートラル）
        float left_ratio = (sum > 0.0f) ? (count_left / sum) : 0.5f;
        float right_ratio = 1.0f - left_ratio;
        float act_diff = right_ratio - left_ratio; // -1 〜 +1
        float diff_amp = std::clamp(act_diff, -1.0f, 1.0f);
        hist_action_->AddBatch({ diff_amp });
        hist_action_->NextFrame();

        // ---- Q値統計 ----
        // サンプリング状態群に対するQ値算出
        auto q = policy_net->forward(samples.states);                     // (B, A)
        auto max_q_dev = std::get<0>(q.max(1));                           // (B,)

        // 数値安定化：有限値のみ抽出（NaN/Inf除去）
        auto finite_mask = torch::isfinite(max_q_dev);
        auto max_q_finite = max_q_dev.index({ finite_mask });

        // 最低サンプル数チェック（分位点や分散計算の安定性確保）
        const int64_t n_finite = max_q_finite.numel();
        bool stats_ready = n_finite >= 4;

        if (stats_ready) {
            auto max_q_cpu = max_q_finite.detach().to(torch::kFloat32).contiguous().cpu(); // (N,)
            // 平均・標準偏差・最大
            float q_mean = max_q_cpu.mean().item<float>();
            float q_std = max_q_cpu.std(false).item<float>();
            float q_max = max_q_cpu.max().item<float>();

            // EMA更新（標準偏差と2乗）
            if (!qstd_init_) {
                qstd_init_ = true;
                qstd_ema_ = q_std;
                qstd_ema2_ = q_std * q_std;
            }
            else {
                qstd_ema_ = (1 - config_.qstd_alpha) * qstd_ema_ + config_.qstd_alpha * q_std;
                qstd_ema2_ = (1 - config_.qstd_alpha) * qstd_ema2_ + config_.qstd_alpha * (q_std * q_std);
            }

            // Z score
            float var = std::max(0.0f, qstd_ema2_ - qstd_ema_ * qstd_ema_);
            float sigma = std::sqrt(var);
            float q_z = (sigma > 1e-8f) ? (q_std - qstd_ema_) / sigma : 0.0f;   // q_stdスパイク時にだけ跳ねる

            // 長期安定補助指標
            float q_cv = q_std / (std::abs(q_mean) + 1e-6f);

            // NIQR（四分位範囲の正規化）
            auto sorted = std::get<0>(max_q_cpu.sort());
            int n = static_cast<int>(sorted.size(0));
            int i25 = static_cast<int>(0.25f * (n - 1));
            int i50 = static_cast<int>(0.50f * (n - 1));
            int i75 = static_cast<int>(0.75f * (n - 1));
            float q25 = sorted[i25].item<float>();
            float q50 = sorted[i50].item<float>();
            float q75 = sorted[i75].item<float>();
            float q_niqr = (q75 - q25) / (std::abs(q50) + 1e-6f);

            // 崩壊判定（Q値由来）
            bool qz_unstable = (q_z > config_.q_z_threshold);
            bool qcv_unstable = (q_cv > config_.q_cv_threshold);
            bool qniqr_unstable = (q_niqr > config_.q_niqr_threshold);
            bool q_unstable = qz_unstable || qcv_unstable || qniqr_unstable;

            // --- 連続不安定度（超過“度”をスカラー化） ---
            float e_z = std::max(0.0f, q_z / config_.q_z_threshold - 1.0f);
            float e_cv = std::max(0.0f, q_cv / config_.q_cv_threshold - 1.0f);
            float e_niqr = std::max(0.0f, q_niqr / config_.q_niqr_threshold - 1.0f);
            float e_t = std::max(e_z, std::max(e_cv, e_niqr));  // 0以上
            e_t = std::clamp(e_t, 0.0f, 1.0f);  //            // Q統計が不安定な初期ステップで異常値を防ぐ

            // --- 漏れ積分（EWMA） ---
            float alpha = std::clamp(uema_alpha_, 0.0f, 1.0f);
            unstable_ema_ = (1.0f - alpha) * unstable_ema_ + alpha * e_t;

            float s_raw = 1.0f / (1.0f + std::exp(-config_.uema_k * (unstable_ema_ - config_.uema_u0))); // 0..1

            // ベースライン( u=0 のときの出力 )を引き、0..1 に再マップ
            float s0 = 1.0f / (1.0f + std::exp(-config_.uema_k * (0.0f - config_.uema_u0)));
            float s = (s_raw - s0) / (1.0f - s0);
            s = std::clamp(s, 0.0f, 1.0f);

            // ---- 行動偏りによる崩壊検知 ----
            bool act_unstable = (std::abs(act_diff) > config_.act_bias_threshold);

            // --- unstable判定の統合 ---
            bool unstable = q_unstable || act_unstable;

            // ---- 崩壊検知時の適応制御 ----
            if (config_.use_as_dqn && config_.use_unstable_ema) {
                //float ramp = std::clamp(post_warmup_steps_ / 3000.0f, 0.0f, 1.0f); // 3kステップで完全有効化
                //float s_eff = s * ramp;

                if (s > config_.unstable_ema_s_threshold || act_unstable)
                        last_unstable_step_ = train_step;     // 崩壊判定の最後のステップ数を覚えておく

                // εブーストを強化（探索復帰、ReplayBuffer 多様性回復）
                float mult = 1.0f + config_.uema_g1 * s;
                eps_boost_ = std::min(config_.eps_boost_max, eps_boost_ * mult);

                // ε再加熱
                float floor_target = config_.eps_reheat_base + config_.uema_g2 * s;
                if (act_unstable) floor_target = std::max(floor_target, config_.eps_reheat_floor);
                eps_reheat_floor_ = std::max(eps_reheat_floor_, floor_target);

                // τ減衰（行動崩壊時は強制的に下限寄せ）
                float decay = 1.0f - config_.uema_g3 * s;
                if (act_unstable) decay *= 0.5f;
                tau_ = std::max(config_.tau_min, tau_ * decay);
            } else if (config_.use_as_dqn && unstable) {
                // 崩壊判定の最後のステップ数を覚えておく
                last_unstable_step_ = train_step;

                // εブーストを強化（探索復帰、ReplayBuffer 多様性回復）
                float boost_factor = std::exp(std::log(2.0f) /
                    (config_.eps_boost_half_life_hit / config_.replay_update_interval));
                eps_boost_ = std::min(config_.eps_boost_max, eps_boost_ * boost_factor);

                // ε再加熱: εの「下限」を動的に引き上げる（epsilon自体は触らない）
                eps_reheat_floor_ = std::max(eps_reheat_floor_, config_.eps_reheat_floor);

                // τ を減らす（ターゲット更新を滑らかに → 発散を防止）
                float decay_factor = std::exp(-std::log(2.0f) / 
                    (config_.tau_half_life_hit / config_.replay_update_interval));
                tau_ = std::max(config_.tau_min, tau_ * decay_factor);
            }

            // 安定時のτ回復とεブーストの自然減衰はUpdate()側でマイステップ実行

            // ---- メトリクス出力 ----
            anet::MetricsLogger::Instance()->log_scalar("33_agent_as/01_eps_boost", train_step, eps_boost_);
            anet::MetricsLogger::Instance()->log_scalar("34_agent_as_a/01_action_diff", train_step, act_diff);
            anet::MetricsLogger::Instance()->log_scalar("34_agent_as_a/03_action_unstable", train_step, act_unstable ? 1.0f : 0.0f);
            anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/01_q_unstable", train_step, q_unstable ? 1.0f : 0.0f);
            anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/02_q_mean", train_step, q_mean);
            anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/03_q_max", train_step, q_max);
            anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/04_q_std", train_step, q_std);
            anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/05_q_std_ema", train_step, qstd_ema_); // 優先度低い
            anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/06_q_z", train_step, q_z);
            anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/07_q_cv", train_step, q_cv);
            anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/08_q_niqr", train_step, q_niqr);
            anet::MetricsLogger::Instance()->log_scalar("36_agent_as_uema/01_uema_unstable", train_step, unstable_ema_);
            anet::MetricsLogger::Instance()->log_scalar("36_agent_as_uema/02_uema_s", train_step, s);
            anet::MetricsLogger::Instance()->log_scalar("36_agent_as_uema/03_uema_e_t", train_step, e_t);
            anet::MetricsLogger::Instance()->log_scalar("36_agent_as_uema/04_uema_diff_s_e_t", train_step, s - e_t);

            // Q値ヒストグラム更新
            float* p = max_q_cpu.data_ptr<float>();
            std::vector<float> vals(p, p + max_q_cpu.size(0));
            hist_q_->AddBatch(vals);
            hist_q_->NextFrame();   // TODO データ量が不十分なら追加でサンプリング評価
        }   // stats_ready
    } else {    // else: replay_buffer.Size() >= config_.replay_batch_size
        // ---- メトリクス出力(サンプル不足で評価未実施なので固定値) ----
        anet::MetricsLogger::Instance()->log_scalar("33_agent_as/01_eps_boost", train_step, eps_boost_);
        anet::MetricsLogger::Instance()->log_scalar("34_agent_as_a/01_action_diff", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("34_agent_as_a/03_action_unstable", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/01_q_unstable", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/02_q_mean", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/03_q_max", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/04_q_std", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/04_q_std_ema", train_step, qstd_ema_); // 優先度低い
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/06_q_z", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/07_q_cv", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("35_agent_as_q/08_q_niqr", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("36_agent_as_uema/01_uema_unstable", train_step, unstable_ema_);
        anet::MetricsLogger::Instance()->log_scalar("36_agent_as_uema/02_uema_s", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("36_agent_as_uema/03_uema_e_t", train_step, 0.0f);
        anet::MetricsLogger::Instance()->log_scalar("36_agent_as_uema/04_uema_diff_s_e_t", train_step, 0.0f);
    }

    // ReplayBufferモード時はReplayBufferの状態分布に基づくヒートマップを生成
    if (config_.heatmap_log_hist_interval > 0 && (train_step % config_.heatmap_log_hist_interval) == 0) {
        // ヒストグラム保存 (軸合わせのためサンプル数が足りない場合も出力)
		// TODO: hist_action_ が正しく出力されない
        //anet::MetricsLogger::Instance()->log_image("43_agent_img/01_th_action", train_step, *hist_action_, 50);
        anet::MetricsLogger::Instance()->log_image("43_agent_img/06_th_q", train_step, *hist_q_);
    }

    // --- バッチ展開 ---
    std::vector<torch::Tensor> states, next_states, actions, rewards, dones, truncs;
    for (const auto& e : batch) {
        wxLogDebug("OptimizeBatch() e.state=%s", anet::ToString(e.state));
        wxLogDebug("OptimizeBatch() e.action=%s", anet::ToString(e.action));
        wxLogDebug("OptimizeBatch() e.next_state=%s", anet::ToString(e.response.next_state));
        wxLogDebug("OptimizeBatch() e.reward=%f", e.response.reward);
        wxLogDebug("OptimizeBatch() e.done=%d", e.response.done);
        wxLogDebug("OptimizeBatch() e.trunc=%d", e.response.truncated);
        states.push_back(e.state);
        actions.push_back(e.action.view({ 1 }));  // (1,)
        next_states.push_back(e.response.next_state);
        // shape=(1,) 明示的に指定（tensorより安全）
        rewards.push_back(torch::full({ 1 }, e.response.reward, ctx.FloatOpt()));
        dones.push_back(torch::full({ 1 }, e.response.done ? 1.0f : 0.0f, ctx.FloatOpt()));
        truncs.push_back(torch::full({ 1 }, e.response.truncated ? 1.0f : 0.0f, ctx.FloatOpt()));
    }

    auto state_b = torch::stack(states).to(device);               // (B,state_dim)
    auto action_b = torch::cat(actions).view({ (int64_t)actions.size() }).to(device); // (B,1)
    auto next_state_b = torch::stack(next_states).to(device);     // (B,state_dim)
    wxLogDebug("OptimizeBatch() state_b=%s", anet::ToDefString(state_b));
    wxLogDebug("OptimizeBatch() action_b=%s", anet::ToDefString(action_b));
    wxLogDebug("OptimizeBatch() next_state_b=%s", anet::ToDefString(next_state_b));
    ANET_CHECK_SHAPE(state_b, { B, state_dim_ });
    ANET_CHECK_SHAPE(action_b, { B });
    ANET_CHECK_SHAPE(next_state_b, { B, state_dim_ });

    // squeeze() を安全に（次元指定なし）
    auto reward_b = torch::cat(rewards).squeeze().to(device);   // (B,)
    auto done_b = torch::cat(dones).squeeze().to(device);       // (B,)
    auto trunc_b = torch::cat(truncs).squeeze().to(device);     // (B,)
    auto nonterminal = 1.0f - (done_b * (1.0f - trunc_b));      // (B,)
    ANET_CHECK_SHAPE(reward_b, { B });
    ANET_CHECK_SHAPE(done_b, { B });
    ANET_CHECK_SHAPE(trunc_b, { B });
    ANET_CHECK_SHAPE(nonterminal, { B });

    // --- Q(s,a) ---
    auto q_values = policy_net->forward(state_b);               // (B,A)
    wxLogDebug("OptimizeBatch() q_values=%s", anet::ToDefString(q_values));
    ANET_CHECK_SHAPE(q_values, {B, this->n_actions_});
    auto q_sa = q_values.gather(1, action_b.unsqueeze(-1)).squeeze(-1); // (B,)

    // --- 期待Q（Double DQN対応） ---
    torch::Tensor max_next_q;
    if (config_.use_double_dqn) {
        auto next_q_policy = policy_net->forward(next_state_b);  // (B,A)
        auto next_action = std::get<1>(next_q_policy.max(1));    // (B,)
        auto next_q_target = target_net->forward(next_state_b);  // (B,A)
        max_next_q = next_q_target.gather(1, next_action.unsqueeze(1)).squeeze(1).detach(); // (B,)
    } else {
        auto next_q_targ = target_net->forward(next_state_b);    // (B,A)
        max_next_q = std::get<0>(next_q_targ.max(1)).detach();   // (B,)
    }

    // --- TD誤差と教師値の計算 ---
    auto expected_q_raw = reward_b + config_.gamma * max_next_q * nonterminal; // 教師値
    auto td_raw = expected_q_raw - q_sa;                                       // TD誤差

    torch::Tensor td;  // ← これが必要（スコープ内に定義）
    torch::Tensor expected_q_clamped;

    if (config_.use_td_clip) {
        auto td_clamped = td_raw.clamp(-config_.td_clip_value, config_.td_clip_value);
        td = td_clamped;                              // ← td をここでセット
        expected_q_clamped = q_sa + td_clamped;       // 教師値再構築
    } else {
        td = td_raw;                                  // ← clipしない場合も td に代入
        expected_q_clamped = expected_q_raw;
    }

    // --- 損失計算 ---
    auto loss = torch::nn::functional::smooth_l1_loss(
        q_sa,
        expected_q_clamped.detach(),
        torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kMean)
    );

    // --- 逆伝播 ---
    optimizer.zero_grad();
    loss.backward();

    // --- 勾配ノルム測定 & 勾配クリッピング（Gradient Clipping）---
    float total_norm;
    if (config_.use_grad_clip) {
        total_norm = static_cast<float>(
            torch::nn::utils::clip_grad_norm_(policy_net->parameters(), config_.grad_clip_tau)
            );
    } else {
        total_norm = 0.0f;
        for (auto& p : policy_net->parameters()) {
            if (p.grad().defined()) {
                total_norm += p.grad().data().norm().item<float>();
            }
        }
    }

    // メインネットワークに勾配反映
    optimizer.step();

    // --- 統計情報算出（バッチ平均ベース） ---
    auto q_targ = target_net->forward(state_b); // (B,A)
    auto q_diff = torch::mean(torch::abs(q_sa - q_targ.gather(1, action_b.unsqueeze(1)).squeeze(1)));
    float grad_norm_clipped = (config_.use_grad_clip && total_norm > config_.grad_clip_tau) ? 1.0f : 0.0f;
    grad_norm_clipped_ema = 0.9f * grad_norm_clipped_ema + 0.1f * grad_norm_clipped;
    loss_ema = met_ema_decay * loss_ema + (1 - met_ema_decay) * loss.item<float>();
    float td_cliped = 0.0f;
    if (config_.use_td_clip) {
        auto abs_raw = torch::abs(td_raw);
        td_cliped = torch::mean((abs_raw > config_.td_clip_value).to(torch::kFloat)).item<float>();
        td_clip_ema = met_ema_decay * td_clip_ema + (1 - met_ema_decay) * td_cliped;
    }
    auto td_mean = td.mean().detach().cpu().item<float>();

    // --- メトリクス出力（バッチ平均） ---
    anet::MetricsLogger::Instance()->log_scalar("37_agent_dqn_qtd/01_q_sa", train_step, q_sa.mean().item<double>());
    anet::MetricsLogger::Instance()->log_scalar("37_agent_dqn_qtd/02_q_diff", train_step, q_diff.item<float>());
    if (config_.use_td_clip) {
        anet::MetricsLogger::Instance()->log_scalar("37_agent_dqn_qtd/05_td_cliped_ema", train_step, td_clip_ema);
        anet::MetricsLogger::Instance()->log_scalar("37_agent_dqn_qtd/06_td_cliped", train_step, td_cliped);
        anet::MetricsLogger::Instance()->log_scalar("37_agent_dqn_qtd/04_td_error_raw", train_step, td_raw.mean().item<float>());
    }
    anet::MetricsLogger::Instance()->log_scalar("37_agent_dqn_qtd/03_td_error", train_step, td_mean);
    anet::MetricsLogger::Instance()->log_scalar("38_agent_dqn_loss/01_loss", train_step, loss.item<float>());
    anet::MetricsLogger::Instance()->log_scalar("38_agent_dqn_loss/02_loss_ema", train_step, loss_ema);
    anet::MetricsLogger::Instance()->log_scalar("39_agent_dqn_grad/01_grad_norm", train_step, total_norm);
    if (config_.use_grad_clip) {
        anet::MetricsLogger::Instance()->log_scalar("39_agent_dqn_grad/02_grad_cliped", train_step, grad_norm_clipped);
        anet::MetricsLogger::Instance()->log_scalar("39_agent_dqn_grad/03_grad_cliped_ema", train_step, grad_norm_clipped_ema);
    }

    // --- soft/hard update ---
    if (config_.softupdate_tau > 0) soft_update();
    if (config_.hardupdate_step > 0 && (train_step % config_.hardupdate_step) == 0) hard_update();
}
