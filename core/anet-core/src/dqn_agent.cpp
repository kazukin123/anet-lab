#include "anet/dqn_agent.hpp"
#include <iostream>
#include <tuple>
#include <wx/log.h>
#include "nlohmann/json.hpp"
#include "anet/tensor_utils.hpp"
#include "anet/tensor_check.hpp"
#include "anet/config.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/random.hpp"

using namespace anet::rl;

const float MET_EMA_DECAY = 0.001f;  // 平滑化係数(メトリクス用)
const float MET_EMA_DECAY_ACT = 0.0005f;  // 平滑化係数(メトリクス用)action_ema用

namespace {
    static constexpr int64_t ANY = ANET_SHAPE_ANY;
}

// ---- 内部モジュール達 ----

/// Agent内部変数
struct DQNAgent::RuntimeVars {
    float epsilon;
    float tau;
};

struct DQNUpdateResult : public UpdateResult {
    // A 群
    float td_mean = 0.0f;
    float td_ema = 0.0f;
    float loss = 0.0f;
    float loss_ema = 0.0f;
    float q_mean = 0.0f;
    float q_std = 0.0f;
    float grad_norm = 0.0f;
    float grad_clip_ratio = 0.0f;

    // ランタイムパラメータ
    float epsilon = 1.0f;
    float tau = 0.0f;

    // B 群（AS-DQN 連携用）
    float q_z = 0.0f;
    float q_cv = 0.0f;
    float q_niqr = 0.0f;
    float q_unstable = 0.0f;
    float q_unstable_ema = 0.0f;
    float act_diff_ema = 0.0f;
    float act_unstable = 0.0f;
    float act_unstable_ema = 0.0f;
    float e_t = 0.0f;
    float s = 0.0f;
    float unstable_ema = 0.0f;
    float collapse_s_ = 0.0f;
    float collapse_l_ = 0.0f;

    virtual MetricsMap GetMetricsMap() const override {
        MetricsMap map;

        // A 群：DQN基本事項
        map["32_agent_dqn_base/03_epsilon"] = epsilon;
        map["32_agent_dqn_base/04_tau"] = tau;
        map["37_agent_dqn_qtd/01_td_mean"] = td_mean;
        map["37_agent_dqn_qtd/03_td_error_ema"] = td_ema;
        map["37_agent_dqn_qtd/11_q_mean"] = q_mean;
        map["37_agent_dqn_qtd/12_q_std"] = q_std;
        map["37_agent_dqn_qtd/21_grad_norm"] = grad_norm;
        map["37_agent_dqn_qtd/22_grad_clip_ratio"] = grad_clip_ratio;
        map["38_agent_dqn_loss/01_loss"] = loss;
        map["38_agent_dqn_loss/02_loss_ema"] = loss_ema;

        // B 群：AS-DQN関連
        map["32_agent_dqn_base/10_collapse_s"] = collapse_s_;
        map["32_agent_dqn_base/11_collapse_l"] = collapse_l_;
        map["33_agent_dqn_action/01_act_diff_ema"] = act_diff_ema;
        map["33_agent_dqn_action/02_act_unstable"] = act_unstable;
        map["33_agent_dqn_action/03_act_unstable_ema"] = act_unstable_ema;
        map["37_agent_dqn_qtd/31_q_z"] = q_z;
        map["37_agent_dqn_qtd/32_q_cv"] = q_cv;
        map["37_agent_dqn_qtd/33_q_niqr"] = q_niqr;
        map["37_agent_dqn_qtd/34_q_unstable"] = q_unstable;
        map["37_agent_dqn_qtd/35_q_unstable_ema"] = q_unstable_ema;
        map["37_agent_dqn_qtd/41_e_t"] = e_t;
        map["37_agent_dqn_qtd/42_s"] = s;
        map["37_agent_dqn_qtd/43_unstable_ema"] = unstable_ema;

        /// @todo s_emaを追加。

        return map;
    }
};

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


// ===============================
// DQNAgent::ActionDecider
// ===============================
class DQNAgent::ActionDecider {
public:
    ActionDecider(DQNAgent& agent)
        : agent_(agent), config_(agent.config_), rnd_(agent.rnd_)
    {
    }

    /// q_values: (1, n_actions)
    std::tuple<int, bool> Decide(const torch::Tensor& q_values, size_t step, bool greedy_only)
    {
        // 常に greedy モード
        if (greedy_only) {
            return { GreedyAction(q_values), false };
        }
 
        // ε-greedy
        if (rnd_->Uniform01() < agent_.vars_->epsilon) {
            return { RandomAction(), true };
        } else {
            return { GreedyAction(q_values), false };
        }
    }
private:
    // ----------------------------------------------------
    // greedy
    // ----------------------------------------------------
    int GreedyAction(const torch::Tensor& q_values) const
    {
        auto max_idx = std::get<1>(q_values.max(1));
        return max_idx.item<int>();
    }

    // ----------------------------------------------------
    // random
    // ----------------------------------------------------
    int RandomAction() const
    {
        int n_actions = agent_.n_actions_;
        return rnd_->RandInt(0, n_actions - 1);
    }

private:
    DQNAgent& agent_;
    const DQNAgentConfig& config_;
    anet::RandomGenerator* rnd_;
};

// ===============================
// DQNAgent::ReplayScheduler
// ===============================
class DQNAgent::ReplayScheduler {
public:
    ReplayScheduler(const DQNAgentConfig& config)
        : config_(config) {
    }

    bool CanUpdate(size_t step, const ReplayBuffer& buf) const
    {
        // warmup（経験不足なら更新しない）
        if (buf.Size() < static_cast<size_t>(config_.replay_warmup_steps)) {
            return false;
        }

        // interval（毎 step=4 のような更新頻度）
        if (config_.replay_update_interval > 1 &&
            (step % config_.replay_update_interval) != 0)
        {
            return false;
        }

        return true;
    }

    int BatchSize() const {
        return config_.replay_batch_size;
    }
private:
    const DQNAgentConfig& config_;
};

// ===============================
// DQNAgent::TargetUpdater
// ===============================
class DQNAgent::TargetUpdater {
public:
    TargetUpdater(const DQNAgentConfig& config)
        : config_(config) {
    }

    /// policy_net → target_net
    void Sync(
        size_t step,
        float tau,
        const std::shared_ptr<const QNetImpl>& policy_net,
        const std::shared_ptr<QNetImpl>& target_net)
    {
        // Hard update
        if (config_.hardupdate_interval > 0 &&
            (step % config_.hardupdate_interval) == 0)
        {
            HardSync(policy_net, target_net);
            return;
        }

        // Soft update
        if (tau > 0.0f) {
            SoftSync(policy_net, target_net, tau);
            return;
        }

        // どちらも無効なら何もしない
    }

private:
    void HardSync(
        const std::shared_ptr<const DQNAgent::QNetImpl>& policy_net,
        const std::shared_ptr<DQNAgent::QNetImpl>& target_net)
    {
        torch::NoGradGuard no_grad;

        auto p_params = policy_net->named_parameters(true /*recurse*/);
        auto t_params = target_net->named_parameters(true /*recurse*/);

        for (auto& kv : t_params) {
            const std::string& name = kv.key();
            kv.value().copy_(p_params[name]);
        }
    }

    void SoftSync(
        const std::shared_ptr<const DQNAgent::QNetImpl>& policy_net,
        const std::shared_ptr<DQNAgent::QNetImpl>& target_net,
        float tau)
    {
        torch::NoGradGuard ng;

        wxLogDebug("SoftSync() tau=%f", tau);

        auto p_params = policy_net->named_parameters(true);
        auto t_params = target_net->named_parameters(true);

        for (auto& kv : t_params) {
            const std::string& name = kv.key();
            auto t = kv.value();
            auto p = p_params[name];

            t.mul_(1.0f - tau);
            t.add_(p, tau);
        }
    }
private:
    const DQNAgentConfig& config_;
};


/// Metrics出力および安定性評価のためのメトリクス情報を提供。Agent内部状態の変更は行わない。
class DQNAgent::StabilityMonitor {
public:
    explicit StabilityMonitor(const DQNAgentConfig& cfg)
        : cfg_(cfg),
        td_ema_(MET_EMA_DECAY),
        loss_ema_(MET_EMA_DECAY),
        q_std_ema_(cfg.qstd_alpha),
        grad_clip_ema_(MET_EMA_DECAY),
        q_unstable_ema_(MET_EMA_DECAY),
        act_diff_ema_(MET_EMA_DECAY_ACT),
        act_unstable_ema_(MET_EMA_DECAY)
    {
        act_diff_ema_.Set(0.0f);
    }

    // ------------------------------------------------------
    // Action 情報更新（B 群：行動偏り）
    // ------------------------------------------------------
    void UpdateActionStats(const anet::rl::BatchActionInfo& info)
    {
        torch::Tensor a = info.action;  // (N, action_dim)
        auto a_cpu = a.to(torch::kCPU).reshape({ -1 });
        const int64_t n = a_cpu.numel();

        int64_t cnt0 = 0;
        int64_t cnt1 = 0;
        const auto* p = a_cpu.data_ptr<int64_t>();
        for (int64_t i = 0; i < n; ++i) {
            if (p[i] == 0) {
                ++cnt0;
            }
            else if (p[i] == 1) {
                ++cnt1;
            }
        }

        const float total = static_cast<float>(cnt0 + cnt1);
        if (total <= 0.0f) {
            return;
        }

        const float ratio_left = static_cast<float>(cnt0) / total;
        const float ratio_right = static_cast<float>(cnt1) / total;
        const float diff = ratio_right - ratio_left; // [-1, 1]
        //wxLogDebug("UpdateActionStats() left=%f right=%f diff=%f", ratio_left, ratio_right, diff);

        // 生の偏りはそのまま保持
        act_diff_ = diff;

        // EMA を取る
        act_diff_ema_.Update(diff);

        // 閾値判定は EMAの絶対値 に対して行う
        act_unstable_ = (fabs(act_diff_ema_.Value()) > cfg_.act_bias_threshold) ? 1.0f : 0.0f;
        act_unstable_ema_.Update(act_unstable_);
    }

    // ------------------------------------------------------
    // Batch 学習更新（A 群メトリクス + B 群 Q 形状）
    // ------------------------------------------------------
    void UpdateBatchStats(
        const torch::Tensor& td_error,         // (B,)
        const torch::Tensor& loss_per_sample,  // (B,)
        const torch::Tensor& max_q,            // (B,) ← max_a Q(s,a)
        float grad_norm,
        float grad_clip_ratio)
    {
        // --- A 群 -------------------------------------------------

        // TD
        td_mean_ = td_error.mean().item<float>();
        td_ema_.Update(td_mean_);

        // Loss
        loss_mean_ = loss_per_sample.mean().item<float>();
        loss_ema_.Update(loss_mean_);

        // Q 統計
        const float q_mean = max_q.mean().item<float>();
        const float q_std = max_q.std(/*unbiased=*/false).item<float>();

        q_mean_ = q_mean;
        q_std_ = q_std;

        // Q std の EMA は「基準値」になる
        q_std_ema_.Update(q_std);

        // grad
        grad_norm_ = grad_norm;
        grad_clip_ema_.Update(grad_clip_ratio);
        grad_clip_ratio_ = grad_clip_ema_.Value();

        // --- B 群：AS-DQN（Q 形状） ------------------------------
        UpdateQShapeStats_(max_q, q_mean, q_std);
    }
private:

    // ------------------------------------------------------
    // B 群：AS-DQN Q 形状統計
    // ------------------------------------------------------
    void UpdateQShapeStats_(const torch::Tensor& max_q,
        float q_mean, float q_std)
    {
        auto sorted = std::get<0>(max_q.sort(0));
        int64_t n = sorted.size(0);
        if (n <= 1) return;

        // IQR
        float q1 = sorted[static_cast<int64_t>(n * 0.25)].item<float>();
        float q3 = sorted[static_cast<int64_t>(n * 0.75)].item<float>();
        float iqr = q3 - q1;

        // CV
        const float eps = 1e-6f;
        q_cv_ = q_std / (std::fabs(q_mean) + eps);

        // Z-score: 平常時stdとの比
        float std_ref = q_std_ema_.Value();
        q_z_ = q_std / (std_ref + eps);

        // NIQR: IQR を std_ref で正規化
        q_niqr_ = iqr / (std_ref + eps);

        // 閾値
        bool unstable_z = (q_z_ > cfg_.q_z_threshold);
        bool unstable_cv = (q_cv_ > cfg_.q_cv_threshold);
        bool unstable_niqr = (q_niqr_ > cfg_.q_niqr_threshold);

        q_unstable_ = (unstable_z || unstable_cv || unstable_niqr) ? 1.0f : 0.0f;
        q_unstable_ema_.Update(q_unstable_);

        // e_t: “崩壊度”
        float ez = std::max(0.0f, q_z_ - cfg_.q_z_threshold);
        float ecv = std::max(0.0f, q_cv_ - cfg_.q_cv_threshold);
        float eniqr = std::max(0.0f, q_niqr_ - cfg_.q_niqr_threshold);

        e_t_ = std::max({ ez, ecv, eniqr });

        // s: ロジスティック圧縮
        float k = cfg_.uema_k;     // 傾き
        float x0 = cfg_.uema_u0;    // 中心値
        s_ = 1.0f / (1.0f + std::exp(-k * (e_t_ - x0)));

        // unstable_ema: 半減期ベースのEMA
        float alpha = std::log(2.0f) / cfg_.uema_half_life;
        unstable_ema_ += alpha * (s_ - unstable_ema_);
    }

private:
    const DQNAgentConfig& cfg_;

    // ---------- A 群 ----------
    float td_mean_ = 0.0f;
    float loss_mean_ = 0.0f;
    float q_mean_ = 0.0f;
    float q_std_ = 0.0f;

    float grad_norm_ = 0.0f;
    float grad_clip_ratio_ = 0.0f;

    anet::EmaFilter<float> td_ema_;
    anet::EmaFilter<float> loss_ema_;
    anet::EmaFilter<float> q_std_ema_;
    anet::EmaFilter<float> grad_clip_ema_;

    // ---------- B 群（Q 形状） ----------
    float q_z_ = 0.0f;
    float q_cv_ = 0.0f;
    float q_niqr_ = 0.0f;
    float q_unstable_ = 0.0f;
    anet::EmaFilter<float> q_unstable_ema_;

    float e_t_ = 0.0f;
    float s_ = 0.0f;
    float unstable_ema_ = 0.0f;

    // ---------- B 群（行動） ----------
    float act_diff_ = 0.0f;
    anet::EmaFilter<float> act_diff_ema_;
    float act_unstable_ = 0.0f;
    anet::EmaFilter<float> act_unstable_ema_;
    float collapse_s_ = 0.0f;
    float collapse_l_ = 0.0f;
public:
    float GetActUnstableEma() const { return act_unstable_ema_; }
    float GetQUnstableEma() const { return q_unstable_ema_; }
    float GetQUnstable() const { return q_unstable_; }
    float GetActUnstable() const { return act_unstable_; }
    float GetUnstableEma() const { return unstable_ema_; }
    void SetCollapseStep(float collapse_s) { collapse_s_ = collapse_s; }
    void SetCollapseLearn(float collapse_l) { collapse_l_ = collapse_l; }

    // ------------------------------------------------------
    // UpdateResult へ書き込み
    // ------------------------------------------------------
    void FillUpdateResult(DQNUpdateResult& out) const
    {
        // A 群
        out.td_mean = td_mean_;
        out.td_ema = td_ema_.Value();
        out.loss = loss_mean_;
        out.loss_ema = loss_ema_.Value();
        out.q_mean = q_mean_;
        out.q_std = q_std_;
        out.grad_norm = grad_norm_;
        out.grad_clip_ratio = grad_clip_ratio_;

        // B 群（Q形状）
        out.q_z = q_z_;
        out.q_cv = q_cv_;
        out.q_niqr = q_niqr_;
        out.q_unstable = q_unstable_;
        out.q_unstable_ema = q_unstable_ema_.Value();
        out.e_t = e_t_;
        out.s = s_;
        out.unstable_ema = unstable_ema_;

        // B 群（行動）
        out.act_diff_ema = act_diff_ema_;
        out.act_unstable = act_unstable_;
        out.act_unstable_ema = act_unstable_ema_.Value();
        out.collapse_s_ = collapse_s_;
        out.collapse_l_ = collapse_l_;
    }
};

// ================================================
// DQNAgent::StabilityController  (AS-DQN 完全実装)
// ================================================
class DQNAgent::StabilityController {
public:
    explicit StabilityController(const DQNAgentConfig& cfg)
        : cfg_(cfg)
    {
    }

    // ------------------------------------------------------------
    // UpdateOnStep : 毎ステップ（環境 step）
    // ・行動偏りに基づく collapse を取得
    // ・epsilon ブースト（hit側）だけを担当
    // ------------------------------------------------------------
    void UpdateOnStep(RuntimeVars& vars, StabilityMonitor& mon, size_t step) const
    {
        float collapse_s = ComputeCollapseOnStep(mon);
        float eps_new = ComputeBoostEpsilon(vars.epsilon, collapse_s, step);
        mon.SetCollapseStep(collapse_s);

        if (cfg_.use_as_dqn) {
            vars.epsilon = eps_new;
        }
    }

    // ------------------------------------------------------------
    // UpdateOnLearn : replay_update_intervalごとの学習ステップ
    // ・Q統計などを含む collapse を取得
    // ・epsilon 減衰（recover側）
    // ・tau の更新（hit/recover 両方）
    // ------------------------------------------------------------
    void UpdateOnLearn(RuntimeVars& vars, StabilityMonitor& mon, size_t update_step_count) const
    {
        float collapse_l = ComputeCollapseOnLearn(mon);
        auto [eps_new, tau_new] = ComputeEpsilonTauLearn(vars.epsilon, vars.tau, collapse_l, update_step_count);
        mon.SetCollapseLearn(collapse_l);

        if (cfg_.use_as_dqn) {
            vars.epsilon = eps_new;
            vars.tau = tau_new;
        }
    }

private:
    const DQNAgentConfig& cfg_;

    // ============================================================
    // collapse（崩壊度）の設計
    // ============================================================

    // --- 行動偏りなどの Step 用 ---
    float ComputeCollapseOnStep(const StabilityMonitor& mon) const
    {
        // act_unstable_ema ∈ [0,1] のイメージ
        float a = mon.GetActUnstableEma();

        // UEMAも利用可能（GetUnstableEma）
        float u = 0;// mon.GetUnstableEma(); /// @todo unstable_emaを考慮

        // Q統計要素まで使うと過剰なので Step側は軽めに
        float c = std::max(a, u);
        return c;
    }

    // --- Q統計などを含めた Learn 用 ---
    float ComputeCollapseOnLearn(const StabilityMonitor& mon) const
    {
        float zu = mon.GetQUnstableEma();       // q_unstable（Z / CV / NIQR）
        float au = mon.GetActUnstableEma();     // act_unstable
        float uu = 0;// mon.GetUnstableEma();   // unstable_ema   /// @todo unstable_emaを考慮

        float c = std::max({ zu, au, uu });
        return c;
    }

    /// @todo config_.eps_gainは本来は別用途のConfig項目

    // ============================================================
    // ε 更新：Step 側（hit = ブースト）
    // ============================================================
    float ComputeBoostEpsilon(float eps, float collapse, size_t step) const
    {
        if (collapse < cfg_.eps_gain) {
            return eps; // ブーストしない
        }

        // ブースト用 half-life（崩壊時）
        float h = static_cast<float>(cfg_.eps_boost_half_life_hit);
        float alpha = std::log(2.0f) / h;

        float target = cfg_.eps_max * cfg_.eps_boost_max;
        float new_eps = eps + alpha * (target - eps);

        // 再加熱も加える
        new_eps = std::max(new_eps, cfg_.eps_reheat_floor);

        // clamp
        new_eps = std::clamp(new_eps, cfg_.eps_min, cfg_.eps_max);
        return new_eps;
    }

    // ============================================================
    // ε / τ 更新：Learn 側（recover + hit）
    // ============================================================
    std::tuple<float, float> ComputeEpsilonTauLearn(float eps, float tau,
            float collapse, size_t update_step_count) const
    {
        // ================================
        // epsilon 側（recover or hit）
        // ================================
        float eps_new = eps;

        if (collapse > cfg_.eps_gain) {
            //// ---- hit（崩壊中）：回復でなく減衰方向 ----
            //float h = static_cast<float>(cfg_.eps_boost_half_life_hit);
            //float alpha = std::log(2.0f) / h;

            //float target = cfg_.eps_max * cfg_.eps_boost_max;
            //eps_new += alpha * (target - eps_new);

            //// 再加熱成分
            //eps_new = std::max(eps_new, cfg_.eps_reheat_floor);
        } else {
            // ---- recover（安定側）----
            float h = static_cast<float>(cfg_.eps_boost_half_life_recover);
            float alpha = std::log(2.0f) / h;

            float target = cfg_.eps_min;
            eps_new += alpha * (target - eps_new);

            // 軽い再加熱
            eps_new = std::max(eps_new, cfg_.eps_reheat_base);
        }

        // clamp
        eps_new = std::clamp(eps_new, cfg_.eps_min, cfg_.eps_max);

        // ================================
        // tau 側
        // ================================
        float tau_new = tau;

        if (collapse > cfg_.eps_gain) {
            // ---- hit（崩壊方向）：tau を減衰させる ----
            float h = static_cast<float>(cfg_.tau_half_life_hit);
            float alpha = std::log(2.0f) / h;
            float target = cfg_.tau_min;
            tau_new += alpha * (target - tau_new);
        } else {
            // ---- recover：一定期間安定後、tau を回復 ----
            float h = static_cast<float>(cfg_.tau_half_life_recover);
            float alpha = std::log(2.0f) / h;
            float target = cfg_.tau_max;

            tau_new += alpha * (target - tau_new);
        }

        // clamp
        tau_new = std::clamp(tau_new, cfg_.tau_min, cfg_.tau_max);

        return { eps_new, tau_new };
    }
};


class DQNAgent::RuntimeVarsUpdater {
public:
    RuntimeVarsUpdater(const DQNAgentConfig& config) : config_(config) { }

    void Initilize(RuntimeVars& vars) const
    {
        vars.epsilon = 1.0f;
        vars.tau = config_.softupdate_tau;
    }

    float ComputeEpsilon(size_t step) const
    {
        // 強制ゼロ領域
        if (config_.eps_zero_step >= 0 &&
            static_cast<int>(step) >= config_.eps_zero_step)
        {
            return 0.0f;
        }

        // 自然減衰
        float decay = std::exp(-static_cast<float>(step) / config_.eps_decay_step);
        float eps = config_.eps_min + (config_.eps_max - config_.eps_min) * decay;

        // clamp
        eps = std::max(config_.eps_min, std::min(config_.eps_max, eps));
        return eps;
    }
private:
    const DQNAgentConfig& config_;
};

// ======================================================
// DQNAgent 本体
// ======================================================
DQNAgent::DQNAgent(const DQNAgentConfig& config, anet::rl::EnvSpec& env_spec, torch::Device device, anet::RandomGenerator* rnd) :
    StepBasedAgent(config, device, rnd),
    state_dim_(env_spec.state.CalcStateDim()),
    n_actions_(env_spec.action.ActionCount()),
    policy_net_(std::make_shared<QNetImpl>(state_dim_, n_actions_)),
    target_net_(std::make_shared<QNetImpl>(state_dim_, n_actions_))
{
    /// @todo  ヒートマップオブジェクト類をHeatMapObservberに移動
    //auto nan = std::numeric_limits<float>::quiet_NaN();
    //auto info = env.GetStateSpaceInfo();
    //auto flags = anet::HeatMapFlags::HM_LogScaleValue | anet::HeatMapFlags::HM_AutoNormValue
    //    | anet::HeatMapFlags::HM_AutoScaleAxis | anet::HeatMapFlags::HM_LogScaleAxis | anet::HeatMapFlags::HM_ShowZeroLine;
    //heatmap_visit1_ = anet::rl::MakeStateHeatMapPtr(info, 0, 2, 256, 256, 30000, flags | anet::HeatMapFlags::HM_SumMode);  // x vs theta → reward
    //heatmap_visit2_ = anet::rl::MakeStateHeatMapPtr(info, 2, 3, 256, 256, 30000, flags | anet::HeatMapFlags::HM_SumMode);  // x vs theta → reward
    //heatmap_td_ = anet::rl::MakeStateHeatMapPtr(info, 0, 2, 256, 256, 30000, flags | anet::HeatMapFlags::HM_MeanMode); // x vs theta → td
    //hist_action_ = std::make_unique<anet::TimeHistogram>(
    //    2, 200, anet::TimeFrameMode::Scroll, flags, -1.0f, 1.0f, 0.05f);
    //hist_q_ = std::make_unique<anet::TimeHistogram>(
    //    128, 1000, anet::TimeFrameMode::Unlimited, flags | anet::HeatMapFlags::HM_FlipY, 0.0f, nan, 0.05f);

    // use_replay_buffer=false の場合の強制
    if (!config_.use_replay_buffer) {
        config_.replay_capacity = 1;
        config_.replay_batch_size = 1;
        config_.replay_warmup_steps = 0;
        config_.replay_update_interval = 1;
    }

    // NN初期化
    policy_net_->to(device);
    target_net_->to(device);
    target_net_->eval();

    // 初期同期：policy → target
    torch::serialize::OutputArchive archive;
    policy_net_->save(archive);
    torch::serialize::InputArchive in;
    std::stringstream ss;
    archive.save_to(ss);
    in.load_from(ss);
    target_net_->load(in);
    target_net_->eval();

    // 内部変数を生成＆初期化
    this->vars_ = std::make_unique<RuntimeVars>();

    // 内部モジュール生成
    this->optimizer_ = std::make_unique<torch::optim::Adam>(policy_net_->parameters(), torch::optim::AdamOptions(config_.alpha));
    this->vars_updater_ = std::make_unique<RuntimeVarsUpdater>(config_);
    this->replay_buffer_ = std::make_unique<anet::rl::ReplayBuffer>(env_spec, config_.replay_capacity, rnd);
    this->action_decider_ = std::make_unique<ActionDecider>(*this);
    this->replay_scheduler_ = std::make_unique<ReplayScheduler>(this->config_);
    this->target_updater_ = std::make_unique<TargetUpdater>(this->config_);
    this->stability_monitor_ = std::make_unique<StabilityMonitor>(this->config_);
    this->stability_controller_ = std::make_unique<StabilityController>(this->config_);

    // 内部状態変数を初期化
    vars_updater_->Initilize(*this->vars_);

    // ログ：パラメータ記録
    wxLogInfo("DQNAgent config=%s", config_.ToStdString());
    anet::MetricsLogger::Instance()->LogJson("agent/params", config_.ToJson());
    anet::MetricsLogger::Instance()->Flush();
}


anet::rl::BatchActionInfo DQNAgent::MakeAction(const anet::rl::BatchState& state, anet::rl::RunMode mode)
{
    ANET_CHECK_SHAPE(state.obs, { ANY, state_dim_ });

    auto flat_state = state.Flatten();
    auto flat_obs = flat_state.obs.to(device_);

    torch::NoGradGuard ng;
    torch::Tensor q;
    if (mode == anet::rl::RunMode::Eval1) {
        q = target_net_->forward(flat_obs);
    }
    else {
        q = policy_net_->forward(flat_obs);
    }
    ANET_CHECK_SHAPE(q, { ANET_SHAPE_ANY, n_actions_ });  // (N, n_actions_)

    // Eval → greedy-only
    bool greedy_only = (mode == anet::rl::RunMode::Eval1 || mode == anet::rl::RunMode::Eval2);
    auto [ a, rand ] = action_decider_->Decide(q, step_count_, greedy_only);

    // (N=1) batched ActionInfo
    torch::Tensor action = torch::tensor({ { a } }, torch::kInt64).to(device_);
    torch::Tensor is_rand = torch::tensor({ { rand } }, torch::kBool).to(device_);
    anet::rl::BatchActionInfo act_info{ action, is_rand };

    // 行動統計の更新（学習安定性モニタ側）
    stability_monitor_->UpdateActionStats(act_info);

    // 崩壊情報更新
    stability_controller_->UpdateOnStep(*vars_, *stability_monitor_, step_count_);

    return act_info;
}

std::shared_ptr<const anet::rl::UpdateResult>
DQNAgent::UpdateFromBatch(const anet::rl::BatchExperience& batch_exp)
{
    // ReplayBuffer に push
    replay_buffer_->Push(batch_exp);

    // 戻り用変数
    auto result = std::make_shared<DQNUpdateResult>();

    // 学習タイミング判定
    const bool can_update = replay_scheduler_->CanUpdate(step_count_, *replay_buffer_);

    if (can_update) {
        const int B = config_.replay_batch_size;
        auto raw_samples = replay_buffer_->Sample(B, device_);

        // device / shape チェック（dtype は Push 時点で保証済み）
        ANET_CHECK_DEVICE(raw_samples.obs, device_);
        ANET_CHECK_DEVICE(raw_samples.actions, device_);
        ANET_CHECK_DEVICE(raw_samples.rewards, device_);
        ANET_CHECK_DEVICE(raw_samples.next_states.obs, device_);
        ANET_CHECK_DEVICE(raw_samples.next_states.dones, device_);
        ANET_CHECK_DEVICE(raw_samples.next_states.truncateds, device_);
        ANET_CHECK_DEVICE(raw_samples.next_states.episode_start, device_);
        ANET_CHECK_SHAPE(raw_samples.obs, { B, state_dim_ });
        ANET_CHECK_SHAPE(raw_samples.actions, { B, 1 });    // 離散アクション
        ANET_CHECK_SHAPE(raw_samples.rewards, { B });
        ANET_CHECK_SHAPE(raw_samples.next_states.obs, { B, state_dim_ });
        ANET_CHECK_SHAPE(raw_samples.next_states.dones, { B });
        ANET_CHECK_SHAPE(raw_samples.next_states.truncateds, { B });
        ANET_CHECK_SHAPE(raw_samples.next_states.episode_start, { B });
        ANET_CHECK_DTYPE(raw_samples.obs, torch::kFloat32);
        ANET_CHECK_DTYPE(raw_samples.actions, torch::kInt64);    // 離散アクション
        ANET_CHECK_DTYPE(raw_samples.rewards, torch::kFloat32);
        ANET_CHECK_DTYPE(raw_samples.next_states.dones, torch::kBool);
        ANET_CHECK_DTYPE(raw_samples.next_states.truncateds, torch::kBool);
        ANET_CHECK_DTYPE(raw_samples.next_states.episode_start, torch::kBool);

        wxLogDebug("ReplayBuffer batch OK: B=%lld", raw_samples.obs.size(0));

        // ReplayBufferから取り出した時点では生の多次元StateなのでFlattenする
        auto samples = raw_samples.Flatten();

        // Q(s, a) 生成
        torch::Tensor q_all = policy_net_->forward(samples.obs); // (B, n_actions_)
        ANET_CHECK_SHAPE(q_all, { B, n_actions_ });
        //wxLogDebug("q_all=%s", anet::ToString(q_all));

        // max_a Q(s,a)  (AS-DQN 用統計)
        torch::Tensor max_q = std::get<0>(q_all.max(1)); // (B,)

        // Q(s,a) for taken action
        torch::Tensor actions_b = samples.actions.view({ B, 1 });   // (B,1)
        ANET_CHECK_SHAPE(actions_b, { B, 1 });
        ANET_CHECK_DTYPE(actions_b, torch::kInt64);
        //wxLogDebug("actions_b=%s", anet::ToString(actions_b));
        torch::Tensor q_sa = q_all.gather(1, actions_b).squeeze(1); // (B,)
        ANET_CHECK_SHAPE(q_sa, { B });
        //wxLogDebug("q_sa=%s", anet::ToString(q_sa));

        // -------------------------------------------------
        // max_a' Q_target(s', a')（DQN / DoubleDQN 切替）
        // -------------------------------------------------
        torch::Tensor max_next_q;

        if (config_.use_double_dqn) {
            torch::NoGradGuard no_grad;

            // policy_net で argmax_a Q(s', a)
            torch::Tensor q_next_policy = policy_net_->forward(samples.next_states.obs); // (B, n_actions_)
            ANET_CHECK_SHAPE(q_next_policy, { B, n_actions_ });
            auto next_policy_pair = q_next_policy.max(1);
            torch::Tensor next_actions = std::get<1>(next_policy_pair); // (B,)

            // target_net で Q_target(s', argmax_a Q_online)
            torch::Tensor q_next_target = target_net_->forward(samples.next_states.obs); // (B, n_actions_)
            ANET_CHECK_SHAPE(q_next_target, { B, n_actions_ });
            torch::Tensor next_actions_b = next_actions.view({ B, 1 });             // (B,1)
            torch::Tensor q_next_selected =
                q_next_target.gather(1, next_actions_b).squeeze(1);                 // (B,)

            max_next_q = q_next_selected;
        } else {
            torch::NoGradGuard no_grad;

            // 通常 DQN: max_a' Q_target(s', a')
            torch::Tensor q_next_all = target_net_->forward(samples.next_states.obs); // (B, n_actions_)
            ANET_CHECK_SHAPE(q_next_all, { B, n_actions_ });
            max_next_q = std::get<0>(q_next_all.max(1));                         // (B,)
        }

        // -------------------------------------------------
        // TD target 計算
        //    td_target = r + (1 - terminal) * gamma * max_next_q
        // -------------------------------------------------
        torch::Tensor terminal = (samples.next_states.dones | samples.next_states.truncateds); // (B,) bool
        torch::Tensor not_terminal = 1.0f - terminal.to(torch::kFloat32); // (B,)
        torch::Tensor rewards = samples.rewards; // (B,)
        const float gamma = config_.gamma;
        torch::Tensor td_target = rewards + not_terminal * (gamma * max_next_q); // (B,)

        // -------------------------------------------------
        // TD 誤差と loss（td_clip, Huber）
        // -------------------------------------------------
        // 生の TD 誤差（監視用）
        torch::Tensor td_error_raw = q_sa - td_target.detach(); // (B,)

        // 学習に使う TD 誤差（必要なら clip）
        torch::Tensor td_error_for_loss = td_error_raw;
        if (config_.use_td_clip) {
            td_error_for_loss = torch::clamp(
                td_error_for_loss,
                -config_.td_clip_value,
                config_.td_clip_value
            );
        }

        // Smooth L1 (Huber, δ=1) を手動実装
        torch::Tensor abs_td = td_error_for_loss.abs();               // (B,)
        torch::Tensor quad = 0.5f * td_error_for_loss.pow(2);         // (B,)
        torch::Tensor linear = abs_td - 0.5f;                         // (B,)
        torch::Tensor per_sample_loss = torch::where(abs_td < 1.0f, quad, linear); // (B,)
        torch::Tensor loss_tensor = per_sample_loss.mean();           // scalar

        // optimizer step（grad clip 含む）
        optimizer_->zero_grad();
        loss_tensor.backward();

        float grad_norm = 0.0f;
        bool grad_clipped = false;

        if (config_.use_grad_clip) {
            // clip_grad_norm_ の戻り値は clip 前の全体ノルム
            double grad_norm_val =
                torch::nn::utils::clip_grad_norm_(
                    policy_net_->parameters(),
                    config_.grad_clip_tau
                );
            grad_norm = static_cast<float>(grad_norm_val);
            grad_clipped = (grad_norm_val > config_.grad_clip_tau);
        } else {
            // clip を使わない場合も全体ノルムだけは計算しておく
            torch::Tensor total_sq = torch::zeros({ 1 }, loss_tensor.options());
            for (auto& p : policy_net_->parameters()) {
                if (!p.grad().defined()) continue;
                total_sq += p.grad().data().pow(2).sum();
            }
            grad_norm = std::sqrt(total_sq.item<float>());
        }
        float grad_clip_ratio = grad_clipped ? 1.0f : 0.0f;

        // 更新
        optimizer_->step();

        // target_network更新
        target_updater_->Sync(step_count_, vars_->tau, policy_net_, target_net_);

        // StabilityMonitor 更新（TD / loss / Q / 勾配ノルム）
        float loss_scalar = loss_tensor.item<float>();
        stability_monitor_->UpdateBatchStats(
            td_error_raw,
            per_sample_loss,
            max_q,
            grad_norm,
            grad_clip_ratio);

        // 崩壊情報を更新
        stability_controller_->UpdateOnLearn(*vars_, *stability_monitor_, step_count_);
    }

    // 内部状態変数を更新
    if (!config_.use_as_dqn) {
        vars_->epsilon = vars_updater_->ComputeEpsilon(step_count_);
    }

    // BatchUpdateResultに統計情報を転写
    stability_monitor_->FillUpdateResult(*result);
    result->epsilon = vars_->epsilon;    // eps, tau は RuntimeVars から
    result->tau = vars_->tau;

    // step カウンタ更新
    step_count_++;

    return result;
}
