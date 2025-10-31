#include "RLAgent.hpp"
#include <iostream>
#include <tuple>
#include "app.hpp"
#include "CartPoleEnv.hpp"
#include "tensor_utils.hpp"

using namespace anet::util;

const float met_ema_decay = 0.995f;  // 平滑化係数(メトリクス用)
const float met_ema_decay_act = 0.9995f;  // 平滑化係数(メトリクス用)action_ema用

// ==== ハイパーパラメータ ====
struct RLAgent::Param {

    float alpha = 1e-3f;   // 学習率 1e-3 3e-3 1e-4 1e-4 3e-4 5e-4
    float gamma = 0.99f;   // 0.99f; 0.995f      γが高いほど「長期安定」を目指す
    float eps_max = 1.00f;
    float eps_min = 0.05f;    //0.1f 0.05f
    float eps_decay_step = 100000;
    float softupdate_tau = 0.01f;// 1.0f 0.004f  0.01f 0.005f;   // 大きいとターゲットネットワークからの反映が早くなる。小さいと遅く滑らかになる。0.005→半減期138step
    int hardupdate_step = 2000;// -1 5000; //200 500 1000
    bool use_grad_clip = true;
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

    int heatmap_log_image_interval = 100;
    int heatmap_log_sweep_interval = 100;
    int heatmap_log_hist_interval = 10;

    bool use_as_dqn = false;             //Adaptive Stabilized DQN (AS-DQN)
    float qstd_alpha = 0.01f;           //Q値 std の EWMA 平滑率
    float q_z_threshold = 3.0f;         //z-score 崩壊判定閾値
    float q_cv_threshold = 0.5f;        //CV 崩壊判定閾値
    float q_niqr_threshold = 0.6f;      //NIQR 崩壊判定閾値
    float eps_boost_max = 2.0f;         //ε ブースト上限倍率
    int   eps_boost_half_life = 8000;  //ε ブーストの自然減衰半減期
    float eps_boost_up = 1.03f;
    float eps_gain = 0.15f;
    float eps_reheat_floor = 0.20f;
    float tau_min = 0.0005f;            //τ の下限
    float tau_max = 0.001f;             //τ の上限
    float tau_decay_on_hit = 0.98f;     // 不安定時のτ減衰率
    float tau_recover_rate = 0.002f;    // 1stepごとに+0.2%回復
    int tau_recover_delay = 1000;       // 1000step安定していたら回復開始

    bool  use_unstable_ema = false;    // 連続崩壊制御を使うか（切替用）
    float uema_half_life = 2000.0f;   // 半減期 [step]。ln2/半減期 が EMA係数
    float uema_u0 = 0.12f;            // 作動し始めの基準（無次元）
    float uema_k = 12.0f;            // シグモイドの傾き
    float uema_g1 = 0.02f;            // εブースト倍率のゲイン（相対）
    float uema_g2 = 0.05f;            // ε再加熱floorのゲイン（絶対上乗せ）
    float uema_g3 = 0.10f;            // τ減衰のゲイン（exp(-g3*s)）
    float uema_s_clip = 0.2f;
    float eps_reheat_base = 0.10f;    // 停滞時の軽い再加熱ベース
    float eps_reheat_half_life = 1000;
    float unstable_ema_s_threshold = 0.0f; // 連続崩壊度の閾値

    RLAgent::Param(Properties* props) {
        if (props == NULL) return;
        std::string preset = props->Get("agent.preset", "agent");
        wxString preset_override;
        if (wxGetApp().GetCommandLine()->Found("a", &preset_override)) {
            preset = preset_override;
            props->Set("agent.preset", preset);
        }
        wxLogInfo("agent.preset=%s", preset);

        ANET_READ_PROPS(props, preset, alpha);
        ANET_READ_PROPS(props, preset, gamma);
        ANET_READ_PROPS(props, preset, eps_max);
        ANET_READ_PROPS(props, preset, eps_min);
        ANET_READ_PROPS(props, preset, eps_decay_step);
        ANET_READ_PROPS(props, preset, softupdate_tau);
        ANET_READ_PROPS(props, preset, hardupdate_step);
        ANET_READ_PROPS(props, preset, use_grad_clip);
        ANET_READ_PROPS(props, preset, grad_clip_tau);
        ANET_READ_PROPS(props, preset, use_td_clip);
        ANET_READ_PROPS(props, preset, td_clip_value);
        ANET_READ_PROPS(props, preset, eps_zero_step);
        ANET_READ_PROPS(props, preset, use_double_dqn);
        ANET_READ_PROPS(props, preset, use_replay_buffer);
        ANET_READ_PROPS(props, preset, replay_capacity);
        ANET_READ_PROPS(props, preset, replay_batch_size);
        ANET_READ_PROPS(props, preset, replay_warmup_steps);
        ANET_READ_PROPS(props, preset, replay_update_interval);
        ANET_READ_PROPS(props, preset, heatmap_log_image_interval);
        ANET_READ_PROPS(props, preset, heatmap_log_sweep_interval);
        ANET_READ_PROPS(props, preset, heatmap_log_hist_interval);
        ANET_READ_PROPS(props, preset, use_as_dqn);
        ANET_READ_PROPS(props, preset, qstd_alpha);
        ANET_READ_PROPS(props, preset, q_z_threshold);
        ANET_READ_PROPS(props, preset, q_cv_threshold);
        ANET_READ_PROPS(props, preset, q_niqr_threshold);
        ANET_READ_PROPS(props, preset, eps_boost_max);
        ANET_READ_PROPS(props, preset, eps_boost_half_life);
        ANET_READ_PROPS(props, preset, eps_boost_up);
        ANET_READ_PROPS(props, preset, eps_gain);
        ANET_READ_PROPS(props, preset, eps_reheat_floor);
        ANET_READ_PROPS(props, preset, eps_reheat_half_life);
        ANET_READ_PROPS(props, preset, tau_min);
        ANET_READ_PROPS(props, preset, tau_max);
        ANET_READ_PROPS(props, preset, tau_decay_on_hit);
        ANET_READ_PROPS(props, preset, tau_recover_rate);
        ANET_READ_PROPS(props, preset, tau_recover_delay);
        ANET_READ_PROPS(props, preset, use_unstable_ema);
        ANET_READ_PROPS(props, preset, uema_half_life);
        ANET_READ_PROPS(props, preset, uema_u0);
        ANET_READ_PROPS(props, preset, uema_k);
        ANET_READ_PROPS(props, preset, uema_g1);
        ANET_READ_PROPS(props, preset, uema_g2);
        ANET_READ_PROPS(props, preset, uema_g3);
        ANET_READ_PROPS(props, preset, uema_s_clip);
        ANET_READ_PROPS(props, preset, eps_reheat_base);
        ANET_READ_PROPS(props, preset, unstable_ema_s_threshold);
    }

};

// ======================================================
// QNet 定義（Impl を CPP に置く）
// ======================================================
struct QNetImpl : torch::nn::Module {
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
// RLAgent 実装
// ======================================================
RLAgent::RLAgent(anet::rl::Environment& env, int state_dim, int n_actions, torch::Device device) :
    n_actions_(n_actions),
    policy_net(std::make_shared<QNetImpl>(state_dim, n_actions)),
    target_net(std::make_shared<QNetImpl>(state_dim, n_actions)),
    device(device),

    param_(std::make_unique<RLAgent::Param>(wxGetApp().GetConfig())),   // 設定からパラメータを読み込み

    optimizer(policy_net->parameters(), torch::optim::AdamOptions(param_->alpha)),
    replay_buffer(param_->replay_capacity)
{
    // 学習変数初期化
    tau_ = param_->softupdate_tau;
    epsilon = param_->eps_max;
    eps_reheat_floor_ = param_->eps_min;
    // EMA係数（半減期 → α = ln(2)/H）
    uema_alpha_ = (param_->uema_half_life > 0.0f)
        ? static_cast<float>(std::log(2.0) / param_->uema_half_life)
        : 1.0f; // 念のため（0割回避）

    // ヒートマップオブジェクトを生成
    auto info = env.GetStateSpaceInfo();
    auto flags = anet::HeatMapFlags::HM_LogScale | anet::HeatMapFlags::HM_AutoNorm;
    heatmap_visit1_ = anet::rl::MakeStateHeatMapPtr(info, 0, 2, 256, 256, 30000, flags | anet::HeatMapFlags::HM_SumMode);  // x vs theta → reward
    heatmap_visit2_ = anet::rl::MakeStateHeatMapPtr(info, 2, 3, 256, 256, 30000, flags | anet::HeatMapFlags::HM_SumMode);  // x vs theta → reward
    heatmap_td_     = anet::rl::MakeStateHeatMapPtr(info, 0, 2, 256, 256, 30000, flags | anet::HeatMapFlags::HM_MeanMode); // x vs theta → td
    hist_action_ = std::make_unique<anet::TimeHistogram>(2, 200, anet::TimeFrameMode::Scroll, false, false, -1.0f, 1.0f);
    hist_q_ = std::make_unique<anet::TimeHistogram>(64, 200, anet::TimeFrameMode::Scroll);

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
    wxLogInfo("agent.preset=%s", wxGetApp().GetConfig()->Get("agent.preset", "agent"));
    nlohmann::json params = {
        {"alpha", param_->alpha},
        {"gamma", param_->gamma},
        {"eps_max", param_->eps_max},
        {"eps_min", param_->eps_min},
        {"eps_decay_step", param_->eps_decay_step},
        {"eps_zero_step", param_->eps_zero_step},
        {"softupdate_tau", param_->softupdate_tau},
        {"hardupdate_step", param_->hardupdate_step},
        {"use_grad_clip", param_->use_grad_clip},
        {"grad_clip_tau", param_->grad_clip_tau},
        {"use_td_clip", param_->use_td_clip},
        {"td_clip_value", param_->td_clip_value},
        {"use_double_dqn", param_->use_double_dqn},

        {"use_replay_buffer", param_->use_replay_buffer},
        {"replay_capacity", param_->replay_capacity},
        {"replay_batch_size", param_->replay_batch_size},
        {"replay_warmup_steps", param_->replay_warmup_steps},
        {"replay_update_interval", param_->replay_update_interval},
        {"heatmap_log_hist_interval", param_->heatmap_log_hist_interval},

        {"use_as_dqn", param_->use_as_dqn},
        {"qstd_alpha", param_->qstd_alpha},
        {"q_z_threshold", param_->q_z_threshold},
        {"q_cv_threshold", param_->q_cv_threshold},
        {"q_niqr_threshold", param_->q_niqr_threshold},
        {"eps_boost_max", param_->eps_boost_max},
        {"eps_boost_half_life", param_->eps_boost_half_life},
        {"eps_boost_up", param_->eps_boost_up},
        {"eps_gain", param_->eps_gain},
        {"eps_reheat_floor", param_->eps_reheat_floor},
        {"tau_min", param_->tau_min},
        {"tau_max", param_->tau_max},
        {"tau_decay_on_hit", param_->tau_decay_on_hit},
        {"tau_recover_rate", param_->tau_recover_rate},
        {"tau_recover_delay", param_->tau_recover_delay},

        {"use_unstable_ema", param_->use_unstable_ema},
        {"uema_half_life", param_->uema_half_life},
        {"uema_u0", param_->uema_u0},
        {"uema_k", param_->uema_k},
        {"uema_g1", param_->uema_g1},
        {"uema_g2", param_->uema_g2},
        {"uema_g3", param_->uema_g3},
        {"uema_s_clip", param_->uema_s_clip},
        {"eps_reheat_base", param_->eps_reheat_base},
        {"eps_reheat_half_life", param_->eps_reheat_half_life},
        {"unstable_ema_s_threshold", param_->unstable_ema_s_threshold},
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
    //float eps_base;
    //if (param_->eps_zero_step > 0 && (train_step > param_->eps_zero_step)) {
    //    eps_base = 0.0f;
    //} else {
    //    eps_base = std::max(param_->eps_min, param_->eps_max - static_cast<float>(train_step) / param_->eps_decay_step);
    //}
    //epsilon = std::clamp(eps_base * eps_boost_, param_->eps_min, param_->eps_max);

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
    wxGetApp().GetMetricsLogger()->log_scalar("26_agent/01_action_ema", train_step, action_ema);

    return { action, torch::Tensor(), torch::Tensor() };
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

void RLAgent::soft_update() {
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
void RLAgent::Update(const anet::rl::Experience& exprence) {
    train_step++;

    if (!param_->use_replay_buffer) {
        OptimizeSingle(exprence);
    } else {
        replay_buffer.Push(exprence);
        if (replay_buffer.Size() < static_cast<size_t>(param_->replay_warmup_steps)) {
            //return;
        } else if (train_step % param_->replay_update_interval == 0) {
            auto batch = replay_buffer.Sample(param_->replay_batch_size);
            OptimizeBatch(batch);
        }
    }

    // --- εブースト自然減衰（毎step適用） ---
    if (param_->use_as_dqn) {
        // εブーストの自然減衰
        // replay_update_interval に基づき実効step補正
        float decay = std::exp(-std::log(2.0f) * param_->replay_update_interval / param_->eps_boost_half_life);
        eps_boost_ = 1.0f + (eps_boost_ - 1.0f) * decay;

        // --- ε下限の自然回復 ---
        if (eps_reheat_floor_ > param_->eps_min) {
            const float k = std::exp(-std::log(2.0f) / param_->eps_reheat_half_life); // 半減期5000step
            eps_reheat_floor_ = param_->eps_min + (eps_reheat_floor_ - param_->eps_min) * k;
        }

        // --- ε最終値算出（base × boost + add）---
        float eps_base;
        if (param_->eps_zero_step > 0 && (train_step > param_->eps_zero_step)) {
            eps_base = 0.0f;
        } else {
            eps_base = std::max(param_->eps_min, param_->eps_max - static_cast<float>(train_step) / param_->eps_decay_step);
        }
        float eps_mult = eps_base * eps_boost_;
        float eps_add = (eps_boost_ - 1.0f) * param_->eps_gain;
        float eps_raw = eps_mult + eps_add;

        // clamp時にreheat_floorを適用してepsilon値を生成
        epsilon = std::clamp(eps_raw, eps_reheat_floor_, param_->eps_max);

        // 安定状態が続いている場合はτをゆっくり回復
        if (param_->use_unstable_ema) {
            if (unstable_ema_ < 0.3f) { // 閾値は経験的に 0.2〜0.4
                tau_ = std::min(param_->tau_max, tau_ * (1.0f + param_->tau_recover_rate));
            }
        } else {
            int stable_steps = train_step - last_unstable_step_;
            if (stable_steps > param_->tau_recover_delay) {
                tau_ = std::min(param_->tau_max, tau_ * (1.0f + param_->tau_recover_rate));
            }
        }
        tau_ = std::clamp(tau_, param_->tau_min, param_->tau_max);
    }

    // visitヒートマップ更新
    auto x = exprence.state[0][0].item<float>();
    auto theta = exprence.state[0][2].item<float>();
    auto theta_dot = exprence.state[0][3].item<float>();
    heatmap_visit1_->AddData(x, theta, exprence.response.reward);
    heatmap_visit2_->AddData(theta, theta_dot, exprence.response.reward);

    // ヒートマップ画像保存
    if (param_->heatmap_log_image_interval > 0 && train_step != 0 &&
        (train_step % param_->heatmap_log_image_interval) == 0) {
        wxGetApp().GetMetricsLogger()->log_image("28_agent/01_heatmap_visit1", train_step, *heatmap_visit1_);
        wxGetApp().GetMetricsLogger()->log_image("28_agent/02_heatmap_visit2", train_step, *heatmap_visit2_);
        if (!param_->use_replay_buffer)
            wxGetApp().GetMetricsLogger()->log_image("28_agent/03_heatmap_td", train_step, *heatmap_td_);
    }

    // Q値ヒートマップ生成・保存（CUDA無いと遅い）
    if (param_->heatmap_log_sweep_interval > 0 && train_step != 0 &&
        (train_step % param_->heatmap_log_sweep_interval) == 0) {
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
        wxGetApp().GetMetricsLogger()->log_image("28_agent/04_heatmap_q", train_step, q_map);
    }

    // メトリクス出力
// --- Warmup期間中にAS-DQN統計と同期して軸を揃える ---
    if (param_->use_as_dqn && replay_buffer.Size() < param_->replay_warmup_steps) {
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/01_q_mean", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/02_q_max", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/03_q_std", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/03_q_std_ema", train_step, qstd_ema_);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/04_q_z", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/09_q_cv", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/10_q_niqr", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/14_q_unstable", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/15_unstable_ema", train_step, unstable_ema_);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/16_tau", train_step, tau_);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/17_eps_boost", train_step, eps_boost_);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/19_uema_s", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/20_e_t", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/21_diff_s_e_t", train_step, 0.0f);
    }
    wxGetApp().logScalar("21_agent/18_epsilon", train_step, epsilon);               //εグリーディーのε
    wxGetApp().logScalar("21_agent/17_eps_reheat_floor", train_step, eps_reheat_floor_);               //εグリーディーのε
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
    //if (train_step == 120000 || train_step == 180000) {
    //    for (auto& p : optimizer.param_groups()) {
    //        p.options().set_lr(p.options().get_lr() * 0.5);
    //    }
    //}

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

    auto expected_q = reward_t + param_->gamma * max_next_q * nonterminal;             // (B,)
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
    float total_norm;
    if (param_->use_grad_clip) {
        total_norm = static_cast<float>(
            torch::nn::utils::clip_grad_norm_(policy_net->parameters(), param_->grad_clip_tau)
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
    float grad_norm_clipped = (param_->use_grad_clip && total_norm > param_->grad_clip_tau) ? 1.0f : 0.0f;    // クリップ発動
    grad_norm_clipped_ema = 0.9f * grad_norm_clipped_ema + 0.1f * grad_norm_clipped;
    loss_ema = loss_ema_init_ ? met_ema_decay * loss_ema + (1 - met_ema_decay) * loss.item<float>()
        : loss.item<float>();
    auto q_targ = target_net->forward(state_t.to(device));                                       // (1,A)
    auto q_diff = torch::mean(torch::abs(q_sa - q_targ.gather(1, action_t.unsqueeze(1)).squeeze(1)));
    if (!loss_ema_init_) loss_ema_init_ = true;

    float td_cliped = 0.0f;
    if (param_->use_td_clip) {
        float abs_raw = std::abs(td_raw.item<float>());
        td_cliped = (abs_raw > param_->td_clip_value) ? 1.0f : 0.0f;
        td_clip_ema = met_ema_decay * td_clip_ema + (1 - met_ema_decay) * td_cliped;
    }

    // メトリクス記録
    wxGetApp().logScalar("22_agent/03_q_sa", train_step, q_sa.item<double>());      //
    wxGetApp().logScalar("22_agent/04_q_diff", train_step, q_diff.item<float>());   // policy と target の Q値乖離

    if (param_->use_td_clip) {
        wxGetApp().logScalar("23_agent/05_td_cliped_ema", train_step, td_clip_ema); // TD誤差 クリップ前
        wxGetApp().logScalar("23_agent/06_td_cliped", train_step, td_cliped);       // TD誤差 クリップ前
        wxGetApp().logScalar("23_agent/07_td_error_raw", train_step, td_raw.item<float>()); // TD誤差 クリップ前
    }
    wxGetApp().logScalar("23_agent/08_reward", train_step, exprence.response.reward);   // 報酬
    wxGetApp().logScalar("23_agent/09_td_error", train_step, td.item<float>());         // TD誤差（TD誤差クリップ有りの場合はクリップ後）
    wxGetApp().logScalar("23_agent/10_loss", train_step, loss.item<float>());           // loss値
    wxGetApp().logScalar("23_agent/11_loss_ema", train_step, loss_ema);                 // loss値のEMA移動平均

    wxGetApp().logScalar("24_agent/12_grad_norm", train_step, total_norm);              // 勾配ノルム
    if (param_->use_grad_clip) {
        wxGetApp().logScalar("24_agent/13_grad_cliped", train_step, grad_norm_clipped);       // 勾配ノルムがクリッピングされたか
        wxGetApp().logScalar("24_agent/14_grad_cliped_ema", train_step, grad_norm_clipped_ema);   // 勾配ノルムのクリッピング率（EMA移動平均）
    }
    //    wxGetApp().logScalar("2_agent/done",     train_step, done);
    //    wxGetApp().logScalar("2_agent/hard_update_done",     train_step, hard_update_done);

    // ヒートマップ更新
    auto x = exprence.state[0][0].item<float>();
    auto theta = exprence.state[0][2].item<float>();
    heatmap_td_->AddData(x, theta, td.item<float>());

    // --- soft update（毎回少し近づける） ---
    if (param_->softupdate_tau > 0) {
        soft_update();
    }
    if (param_->hardupdate_step > 0 && (train_step % param_->hardupdate_step) == 0) {
        hard_update();      // --- hard update（定期stepごとに完全同期） ---
    }
}

// ======================================================
// OptimizeBatch：ReplayBuffer有効時（バッチ学習本体）
// ======================================================
void RLAgent::OptimizeBatch(const std::vector<anet::rl::Experience>& batch) {
    if (replay_buffer.Size() < param_->replay_warmup_steps) {
        post_warmup_steps_ = 0;
        // （既存の0出力処理やreturnはそのまま）
        return;
    } else {
        post_warmup_steps_++;
    }

    if (batch.empty()) return;

    if (replay_buffer.Size() < param_->replay_warmup_steps) {
        // 初期化された統計をリセットして安全値へ
        unstable_ema_ = 0.1f;
        eps_boost_ = 1.0f;
        eps_reheat_floor_ = param_->eps_min;
        tau_ = param_->softupdate_tau;

        // --- Warmup期間中も0値を出力して可視化軸を統一 ---
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/01_q_mean", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/02_q_max", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/03_q_std", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/03_q_std_ema", train_step, qstd_ema_);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/04_q_z", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/09_q_cv", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/10_q_niqr", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/14_q_unstable", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/15_unstable_ema", train_step, unstable_ema_);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/16_tau", train_step, tau_);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/17_eps_boost", train_step, eps_boost_);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/18_eps_reheat_floor", train_step, eps_reheat_floor_);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/19_uema_s", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/20_e_t", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/21_diff_s_e_t", train_step, 0.0f);

        return;  // ← AS-DQN部分の更新スキップ
    }

    policy_net->train();
    TensorContext ctx(device);

    if (replay_buffer.Size() >= param_->replay_batch_size) {    // バッチ数分以上のサンプルが溜まってる場合のみ計測
        // バッチサイズ分をまとめてサンプリング
        auto samples = replay_buffer.SampleBatch(param_->replay_batch_size, device);    // (B, state_dim)

        // ---- action統計 ----
        auto actions_cpu = samples.actions.detach().cpu(); // (B,)
        int count_left = 0;
        int count_right = 0;
        const int B = actions_cpu.size(0);

        for (int i = 0; i < B; ++i) {
            int a = actions_cpu[i].item<int>();
            if (a == 0) count_left++;
            else if (a == 1) count_right++;                 // ※ それ以外の値は無視（離散2値前提）
        }

        float sum = static_cast<float>(count_left + count_right);
        // sum==0 のときは 0.5/0.5（視覚上ニュートラル）
        float left_ratio = (sum > 0.0f) ? (count_left / sum) : 0.5f;
        float right_ratio = 1.0f - left_ratio;
        float diff = left_ratio - right_ratio; // -1 〜 +1
        float amp = 3.0f;  // ← 強調倍率（実測に基づき可変推奨）
        float diff_amp = std::clamp(diff * amp, -1.0f, 1.0f);
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
                qstd_ema_ = (1 - param_->qstd_alpha) * qstd_ema_ + param_->qstd_alpha * q_std;
                qstd_ema2_ = (1 - param_->qstd_alpha) * qstd_ema2_ + param_->qstd_alpha * (q_std * q_std);
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
            bool qz_unstable = (q_z > param_->q_z_threshold);
            bool qcv_unstable = (q_cv > param_->q_cv_threshold);
            bool qniqr_unstable = (q_niqr > param_->q_niqr_threshold);
            bool q_unstable = qz_unstable || qcv_unstable || qniqr_unstable;

            // --- 連続不安定度（超過“度”をスカラー化） ---
            float e_z = std::max(0.0f, q_z / param_->q_z_threshold - 1.0f);
            float e_cv = std::max(0.0f, q_cv / param_->q_cv_threshold - 1.0f);
            float e_niqr = std::max(0.0f, q_niqr / param_->q_niqr_threshold - 1.0f);
            float e_t = std::max(e_z, std::max(e_cv, e_niqr));  // 0以上
            e_t = std::clamp(e_t, 0.0f, 1.0f);  //            // Q統計が不安定な初期ステップで異常値を防ぐ

            // --- 漏れ積分（EWMA） ---
            float alpha = std::clamp(uema_alpha_, 0.0f, 1.0f);
            unstable_ema_ = (1.0f - alpha) * unstable_ema_ + alpha * e_t;

            float s_raw = 1.0f / (1.0f + std::exp(-param_->uema_k * (unstable_ema_ - param_->uema_u0))); // 0..1

            // ベースライン( u=0 のときの出力 )を引き、0..1 に再マップ
            float s0 = 1.0f / (1.0f + std::exp(-param_->uema_k * (0.0f - param_->uema_u0)));
            float s = (s_raw - s0) / (1.0f - s0);
            s = std::clamp(s, 0.0f, 1.0f);

            // 崩壊判定結果
            bool unstable = q_unstable;

            // ---- 崩壊検知時の適応制御 ----
            if (param_->use_as_dqn && param_->use_unstable_ema) {
                //float ramp = std::clamp(post_warmup_steps_ / 3000.0f, 0.0f, 1.0f); // 3kステップで完全有効化
                //float s_eff = s * ramp;

                if (s > param_->unstable_ema_s_threshold) // 明確に上昇したとき
                    last_unstable_step_ = train_step;     // 崩壊判定の最後のステップ数を覚えておく

                // εブーストを強化（探索復帰、ReplayBuffer 多様性回復）
                float mult = 1.0f + param_->uema_g1 * s;
                eps_boost_ = std::min(param_->eps_boost_max, eps_boost_ * mult);

                // ε再加熱: εの「下限」を動的に引き上げる（epsilon自体は触らない）
                float floor_target = param_->eps_reheat_base + param_->uema_g2 * s;
                eps_reheat_floor_ = std::max(eps_reheat_floor_, floor_target);

                // unstable_ema の値 s (0..1) に応じて τ を減少させる
                float decay = 1.0f - param_->uema_g3 * s;  // uema_g3 = 0.05〜0.2 推奨
                tau_ = std::max(param_->tau_min, tau_ * decay);
            } else if (param_->use_as_dqn && unstable) {
                // 崩壊判定の最後のステップ数を覚えておく
                last_unstable_step_ = train_step;

                // εブーストを強化（探索復帰、ReplayBuffer 多様性回復）
                eps_boost_ = std::min(param_->eps_boost_max, eps_boost_ * param_->eps_boost_up);

                // ε再加熱: εの「下限」を動的に引き上げる（epsilon自体は触らない）
                eps_reheat_floor_ = std::max(eps_reheat_floor_, param_->eps_reheat_floor);

                // τ を減らす（ターゲット更新を滑らかに → 発散を防止）
                tau_ = std::max(param_->tau_min, tau_ * param_->tau_decay_on_hit);
            }

            // 安定時のτ回復とεブーストの自然減衰はUpdate()側でマイステップ実行

            // ---- メトリクス出力 ----
            wxGetApp().GetMetricsLogger()->log_scalar("21_agent/01_q_mean", train_step, q_mean);
            wxGetApp().GetMetricsLogger()->log_scalar("21_agent/02_q_max", train_step, q_max);
            wxGetApp().GetMetricsLogger()->log_scalar("21_agent/03_q_std", train_step, q_std);
            wxGetApp().GetMetricsLogger()->log_scalar("21_agent/03_q_std_ema", train_step, qstd_ema_); // 優先度低い
            wxGetApp().GetMetricsLogger()->log_scalar("21_agent/04_q_z", train_step, q_z);
            wxGetApp().GetMetricsLogger()->log_scalar("21_agent/09_q_cv", train_step, q_cv);
            wxGetApp().GetMetricsLogger()->log_scalar("21_agent/10_q_niqr", train_step, q_niqr);
            wxGetApp().GetMetricsLogger()->log_scalar("21_agent/13_q_unstable", train_step, q_unstable ? 1.0f : 0.0f);
            wxGetApp().GetMetricsLogger()->log_scalar("21_agent/15_unstable_ema", train_step, unstable_ema_);
            wxGetApp().GetMetricsLogger()->log_scalar("21_agent/16_tau", train_step, tau_);
            wxGetApp().GetMetricsLogger()->log_scalar("21_agent/17_eps_boost", train_step, eps_boost_);
            wxGetApp().GetMetricsLogger()->log_scalar("21_agent/19_uema_s", train_step, s);
            wxGetApp().GetMetricsLogger()->log_scalar("21_agent/20_e_t", train_step, e_t);
            wxGetApp().GetMetricsLogger()->log_scalar("21_agent/21_diff_s_e_t", train_step, s - e_t);

            // Q値ヒストグラム更新
            float* p = max_q_cpu.data_ptr<float>();
            std::vector<float> vals(p, p + max_q_cpu.size(0));
            hist_q_->AddBatch(vals);
            hist_q_->NextFrame();   // TODO データ量が不十分なら追加でサンプリング評価
        }   // stats_ready
    } else {    // else: replay_buffer.Size() >= param_->replay_batch_size
        // ---- メトリクス出力(サンプル不足で評価未実施なので固定値) ----
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/01_q_mean", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/02_q_max", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/03_q_std", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/03_q_std_ema", train_step, qstd_ema_); // 優先度低い
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/04_q_z", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/09_q_cv", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/10_q_niqr", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/13_q_unstable", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/15_unstable_ema", train_step, unstable_ema_);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/16_tau", train_step, tau_);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/17_eps_boost", train_step, eps_boost_);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/19_uema_s", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/20_e_t", train_step, 0.0f);
        wxGetApp().GetMetricsLogger()->log_scalar("21_agent/21_diff_s_e_t", train_step, 0.0f);
    }

    // ReplayBufferモード時はReplayBufferの状態分布に基づくヒートマップを生成
    if (param_->heatmap_log_hist_interval > 0 && (train_step % param_->heatmap_log_hist_interval) == 0) {
        // ヒストグラム保存 (軸合わせのためサンプル数が足りない場合も出力)
        wxGetApp().GetMetricsLogger()->log_image("22_agent/02_hist_action", train_step, *hist_action_, 50);
        wxGetApp().GetMetricsLogger()->log_image("28_agent/05_hist_q", train_step, *hist_q_);
    }

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

    // --- 勾配ノルム測定 & 勾配クリッピング（Gradient Clipping）---
    float total_norm;
    if (param_->use_grad_clip) {
        total_norm = static_cast<float>(
            torch::nn::utils::clip_grad_norm_(policy_net->parameters(), param_->grad_clip_tau)
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
    float grad_norm_clipped = (param_->use_grad_clip && total_norm > param_->grad_clip_tau) ? 1.0f : 0.0f;
    grad_norm_clipped_ema = 0.9f * grad_norm_clipped_ema + 0.1f * grad_norm_clipped;
    loss_ema = met_ema_decay * loss_ema + (1 - met_ema_decay) * loss.item<float>();
    float td_cliped = 0.0f;
    if (param_->use_td_clip) {
        auto abs_raw = torch::abs(td_raw);
        td_cliped = torch::mean((abs_raw > param_->td_clip_value).to(torch::kFloat)).item<float>();
        td_clip_ema = met_ema_decay * td_clip_ema + (1 - met_ema_decay) * td_cliped;
    }

    // --- メトリクス出力（バッチ平均） ---
    wxGetApp().logScalar("22_agent/03_q_sa", train_step, q_sa.mean().item<double>());
    wxGetApp().logScalar("22_agent/04_q_diff", train_step, q_diff.item<float>());
    if (param_->use_td_clip) {
        wxGetApp().logScalar("23_agent/05_td_cliped_ema", train_step, td_clip_ema);
        wxGetApp().logScalar("23_agent/06_td_cliped", train_step, td_cliped);
        wxGetApp().logScalar("23_agent/07_td_error_raw", train_step, td_raw.mean().item<float>());
    }
    wxGetApp().logScalar("23_agent/08_reward", train_step, reward_b.mean().item<float>());
    wxGetApp().logScalar("23_agent/09_td_error", train_step, td.mean().item<float>());
    wxGetApp().logScalar("23_agent/10_loss", train_step, loss.item<float>());
    wxGetApp().logScalar("23_agent/11_loss_ema", train_step, loss_ema);
    wxGetApp().logScalar("24_agent/12_grad_norm", train_step, total_norm);
    if (param_->use_grad_clip) {
        wxGetApp().logScalar("24_agent/13_grad_cliped", train_step, grad_norm_clipped);
        wxGetApp().logScalar("24_agent/14_grad_cliped_ema", train_step, grad_norm_clipped_ema);
    }
    //wxGetApp().logScalar("25_replay/01_buffer_size", train_step, replay_buffer.Size());

    // --- soft/hard update ---
    if (param_->softupdate_tau > 0) soft_update();
    if (param_->hardupdate_step > 0 && (train_step % param_->hardupdate_step) == 0) hard_update();
}
