// dqn_agent.cpp

#include "anet/dqn_agent.hpp"
#include <memory>
#include <torch/torch.h>
#include <tuple>
#include "anet/nn_util.hpp"
#include "anet/tensor_util.hpp"
#include "anet/tensor_check.hpp"
#include "anet/log.hpp"
#include "anet/random.hpp"
#include "anet/config.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/profile.hpp"

using namespace anet::rl;

const float MET_EMA_DECAY = 0.001f;  // 平滑化係数(メトリクス用)
const float MET_EMA_DECAY_ACT = 0.0005f;  // 平滑化係数(メトリクス用)action_ema用

namespace {
    static constexpr int64_t ANY = ANET_SHAPE_ANY;
}

// ---- 内部モジュール達 ----

/// Agent内部変数
struct DQNAgent::RuntimeVars {

    // ランタイムパラメータ
    float epsilon = 1.0f;
    float tau = 0.0f;
	step_t learn_step = 0;

    /// @todo std::atomicに変更してロックフリーにするげ

    // --------------

    // Metrics：基本
    float td_mean = 0.0f;
    float loss = 0.0f;
    float q_mean = 0.0f;
    float q_max = 0.0f;
    float q_std = 0.0f;

    float grad_norm = 0.0f;
    float grad_clip_ratio = 0.0f;

    anet::EmaFilter<float> td_ema;
    anet::EmaFilter<float> loss_ema;
    anet::EmaFilter<float> q_std_ema;
    anet::EmaFilter<float> grad_clip_ema;

    // --------------

    // Metrics：崩壊制御
    float collapse_s = 0.0f;
    float collapse_l = 0.0f;
    float unstable_ema = 0.0f;

    // Metrics：崩壊制御：Q

    /// <summary>
    /// Coefficient of Variation：q_cv = std(Q) / |mean(Q)|
    /// 今回のサンプル群について、Q値がどれくらい散らばっているか（平均値基準）
    /// </summary>
    float q_cv = 0.0f;

    /// <summary>
    /// Std Growth Ratio：q_z = std(Q) / std_ema(Q)
    /// 今回の揺らぎが、直近過去と比べてどれだけ離れているかか
    /// </summary>
    float q_z = 0.0f;

    /// <summary>
    /// Normalized IQR：q_niqr = IQR(Q) / std_ema(Q)
    /// Q 分布の中心がどれだけ広がっているか。
    /// 今の分布の広がりが、過去の安定分布の何倍になっているか
    /// </summary>
    float q_niqr = 0.0f;

    float q_unstable = 0.0f;
    anet::EmaFilter<float> q_unstable_ema;
    float e_t = 0.0f;
    float s = 0.0f;
    anet::EmaFilter<float> s_ema;

    // Metrics：崩壊制御：A
    float act_diff = 0.0f;
    anet::EmaFilter<float> act_diff_ema;
    float act_unstable = 0.0f;
    anet::EmaFilter<float> act_unstable_ema;

    DQNAgent::RuntimeVars() :
        td_ema(MET_EMA_DECAY),
        loss_ema(MET_EMA_DECAY),
        q_std_ema(MET_EMA_DECAY),
        grad_clip_ema(MET_EMA_DECAY),
        q_unstable_ema(MET_EMA_DECAY),
        s_ema(MET_EMA_DECAY_ACT),
        act_diff_ema(MET_EMA_DECAY_ACT),
        act_unstable_ema(MET_EMA_DECAY)
    {
        act_diff_ema.Set(0.0f);
    }
};

class DQNAgent::BatchUpdateResult : public anet::rl::BatchUpdateResult {
private:
    MetricsMap map_;
    const std::optional<torch::Tensor> max_q_;

    //auto finite_mask = torch::isfinite(max_q_dev);
    //auto max_q_finite = max_q_dev.index({ finite_mask });
    //auto max_q_ = max_q_finite.detach().to(torch::kFloat32).contiguous().cpu(); // (N,)

public:
    DQNAgent::BatchUpdateResult(const DQNAgent::RuntimeVars& vars,
        const std::optional<torch::Tensor>& max_q, uint32_t learn_step_diff)
        : anet::rl::BatchUpdateResult(learn_step_diff), max_q_(max_q)
    {
        // A 群：DQN基本事項
        map_["epsilon"] = vars.epsilon;
        map_["tau"] = vars.tau;
        map_["td_mean"] = vars.td_mean;
        map_["td_ema"] = vars.td_ema;
        map_["q_mean"] = vars.q_mean;
        map_["q_max"] = vars.q_max;
        map_["q_std"] = vars.q_std;
        map_["grad_norm"] = vars.grad_norm;
        map_["grad_clip_ratio"] = vars.grad_clip_ratio;
        map_["loss"] = vars.loss;
        map_["loss_ema"] = vars.loss_ema;

        // B 群：AS-DQN関連
        map_["collapse_s"] = vars.collapse_s;
        map_["collapse_l"] = vars.collapse_l;
        map_["act_diff_ema"] = vars.act_diff_ema;
        map_["act_unstable"] = vars.act_unstable;
        map_["act_unstable_ema"] = vars.act_unstable_ema;
        map_["q_z"] = vars.q_z;
        map_["q_cv"] = vars.q_cv;
        map_["q_niqr"] = vars.q_niqr;
        map_["q_unstable"] = vars.q_unstable;
        map_["q_unstable_ema"] = vars.q_unstable_ema;
        map_["e_t"] = vars.e_t;
        map_["s"] = vars.s;
        map_["s_ema"] = vars.s_ema;
        map_["unstable_ema"] = vars.unstable_ema;
    }

    virtual std::optional<float> GetScalar(const std::string& key, int index) const override {
        auto itr = map_.find(key);
        if (itr == map_.end()) {
            return std::nullopt;
        }
        return itr->second;
    }
    virtual std::optional<torch::Tensor> GetTensor(const std::string& key, int index) const override {
        if (key == "max_q") return max_q_;
        return std::nullopt;
    }
    virtual std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int index) const override {
        return std::nullopt;
    }
};

// ======================================================
// QNet 定義（Impl を CPP に置く）
// ======================================================
struct anet::rl::DQNAgent::QNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{ nullptr };
    torch::nn::Linear fc2{ nullptr };
    torch::nn::Linear fc3{ nullptr };

    QNetImpl(int state_dim, int n_actions) {
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, 120));
        fc2 = register_module("fc2", torch::nn::Linear(120, 84));
        fc3 = register_module("fc3", torch::nn::Linear(84, n_actions));
    }
    ///  He正規分布で重み初期化
    void InitWeightsWithHeNormal() {
        anet::ApplyHeNormal(fc1);
        anet::ApplyHeNormal(fc2);
        //anet::ApplyHeNormal(fc3);
        anet::ApplyXavierUniform(fc3);
        /// @todo 出力層だけはXavier uniformが良い(ピクピクで安定してしまう)
    }
    void InitWeightsWithXavierUniform() {
        anet::ApplyXavierUniform(fc1);
        anet::ApplyXavierUniform(fc2);
        anet::ApplyXavierUniform(fc3);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }
};

std::optional<anet::TensorFunction> DQNAgent::GetTensorFunction(const std::string& key)
{
    if (key == "policy_net.forward") {
        anet::TensorFunction fn = [this](const torch::Tensor& t) {
            auto tdev = t.to(device_);
            std::shared_lock<std::shared_mutex> lock(*mutex_);
            return policy_net_->forward(tdev);
            };
        return fn;
    }
    if (key == "target_net.forward") {
        anet::TensorFunction fn = [this](const torch::Tensor& t) {
            auto tdev = t.to(device_);
            std::shared_lock<std::shared_mutex> lock(*mutex_);
            return target_net_->forward(tdev);
            };
        return fn;
    }
    if (key == "q_pair.forward") {
        anet::TensorFunction fn = [this](const torch::Tensor& t) {
            auto tdev = t.to(device_);
            std::shared_lock<std::shared_mutex> lock(*mutex_);
            auto q_online = policy_net_->forward(tdev);     // [N, A]
            auto q_target = target_net_->forward(tdev);     // [N, A]
            return torch::cat({ q_online, q_target }, 1);   // [N, 2*A]
            };
        return fn;
    }

    // default
    anet::TensorFunction fn = [this](const torch::Tensor& t) {
        auto tdev = t.to(device_);
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return policy_net_->forward(tdev);
    };

    return std::nullopt;
}

std::optional<float> DQNAgent::GetScalar(const std::string& key, int index) const
{
    if (key.find("replaybuffer.") == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return replay_buffer_->GetScalar(key);
    }
    if (key == "epsilon") {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return vars_->epsilon;
	} 
    if (key == "tau") {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return vars_->tau;
	}

    return std::nullopt;
}

std::optional<torch::Tensor> DQNAgent::GetTensor(const std::string& key, int index) const
{
    if (key.find("replaybuffer.") == 0)
        return replay_buffer_->GetTensor(key);

    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>> DQNAgent::GetTensorVector(const std::string& key, int index) const
{
    if (key.find("replaybuffer.") == 0)
        return replay_buffer_->GetTensorVector(key);

    return std::nullopt;
}

// ===============================
// DQNAgent::ActionDecider
// ===============================
class DQNAgent::ActionDecider : public anet::RandomHolder {
public:
    ActionDecider(DQNAgent& agent, seed_t seed)
        : agent_(agent), RandomHolder(seed)
    {
    }

    /**
     * @brief バッチ版 ε-greedy 行動選択
     * @param q_values (N, n_actions)
     * @param greedy_only ε-greedy を使わず常にgreedy
     */
    BatchActionInfo DecideBatch(const torch::Tensor& q_values, bool greedy_only)
    {
        ProfileRange  r("DQNAgent::DecideBatch");

        auto device = q_values.device();

        // shape 読み取りは TensorOptions 経由で同期を回避可能
        const int64_t N = q_values.sizes()[0];
        const int64_t A = q_values.sizes()[1];

        // greedy = argmax(q_values, dim=1)
        auto greedy = q_values.argmax(1, /*keepdim=*/false);

        if (greedy_only) {
            ProfileRange  r("DQNAgent::DecideBatch.greedy_only");
            auto zeros = torch::zeros({ N }, torch::TensorOptions().dtype(torch::kBool).device(device));
            return { greedy, zeros };
        }

        const float eps = agent_.vars_->epsilon;

        // mask: (N) bool, GPU上で生成
        auto mask = torch::rand({ N }, torch::TensorOptions().device(device)).lt(eps);    // GPUで完結

        // random actions (N) int64
        auto random_actions =torch::randint(/*low=*/0, /*high=*/A, { N },
                torch::TensorOptions().dtype(torch::kInt64).device(device));

        // actions: where(mask, random_actions, greedy)
        auto actions = torch::where(mask, random_actions, greedy);

        return {
            actions,        // (N) kInt64
            mask            // (N) kBool
        };
    }
private:
    DQNAgent& agent_;
};

// ===============================
// DQNAgent::ReplayScheduler
// ===============================
class DQNAgent::ReplayScheduler {
public:
    ReplayScheduler(const DQNAgentConfig& config)
        : config_(config) {
    }

    bool CanUpdate(step_t update_step, int batch_size, const ReplayBuffer& buf) const
    {
        // warmup（経験不足なら更新しない）
        if (buf.Size() < static_cast<size_t>(config_.replay_warmup_steps * batch_size)) {
            return false;
        }

        // interval（毎 step=4 のような更新頻度）
        if (config_.replay_update_interval > 0 &&
            (update_step % config_.replay_update_interval) != 0)
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
        anet::ProfileRange r("DQNAgent::Sync");

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

        ANET_LOG_DEBUG("SoftSync() tau=" << tau);

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
private:
    const DQNAgentConfig& config_;
public:
    explicit StabilityMonitor(const DQNAgentConfig& config) : config_(config){
    }

    // ------------------------------------------------------
    // Action 情報更新（B 群：行動偏り）
    // ------------------------------------------------------
    void UpdateActionStats(RuntimeVars& vars, const anet::rl::BatchActionInfo& info) const
    {
        anet::ProfileRange r("DQNAgent::UpdateActionStats");
        torch::Tensor a = info.action;  // (N, action_dim)
        auto a_cpu = a.to(torch::kCPU).reshape({ -1 }).contiguous();
        const int64_t n = a_cpu.numel();
        ANET_CHECK_DTYPE(a_cpu, torch::kInt64);

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
        //ANET_LOG_DEBUG("UpdateActionStats() left=" << ratio_left << " right=" << ratio_right << " diff=" << diff);

        // 生の偏りはそのまま保持
        vars.act_diff = diff;

        // EMA を取る
        vars.act_diff_ema.Update(diff);

        // 閾値判定は EMAの絶対値 に対して行う
        vars.act_unstable = (fabs(vars.act_diff_ema.Value()) > config_.act_bias_threshold) ? 1.0f : 0.0f;
        vars.act_unstable_ema.Update(vars.act_unstable);
    }

    // ------------------------------------------------------
    // Batch 学習更新（A 群メトリクス + B 群 Q 形状）
    // ------------------------------------------------------
    void UpdateBatchStats(
        RuntimeVars& vars,
        const torch::Tensor& td_error,         // (B,)
        const torch::Tensor& loss_per_sample,  // (B,)
        const torch::Tensor& max_q,            // (B,) ← max_a Q(s,a)
        float grad_norm, float grad_clip_ratio) const
    {
        anet::ProfileRange  r("DQNAgent::UpdateBatchStats");

        // --- A 群 -------------------------------------------------

        // TD
        vars.td_mean = td_error.mean().item<float>();
        vars.td_ema.Update(vars.td_mean);

        // Loss
        vars.loss = loss_per_sample.mean().item<float>();
        vars.loss_ema.Update(vars.loss);

        // Q 統計
        const float q_mean = max_q.mean().item<float>();
        const float q_max = max_q.max().item<float>();
        const float q_std = max_q.std(/*unbiased=*/false).item<float>();

        vars.q_mean = q_mean;
        vars.q_max = q_max;
        vars.q_std = q_std;

        // Q std の EMA は「基準値」になる
        vars.q_std_ema.Update(q_std);

        // grad
        vars.grad_norm = grad_norm;
        vars.grad_clip_ema.Update(grad_clip_ratio);
        vars.grad_clip_ratio = vars.grad_clip_ema.Value();

        // --- B 群：AS-DQN（Q 形状） ------------------------------
        UpdateQShapeStats_(vars, max_q, q_mean, q_std);
    }
private:

    // ------------------------------------------------------
    // B 群：AS-DQN Q 形状統計
    // ------------------------------------------------------
    void UpdateQShapeStats_(RuntimeVars& vars, const torch::Tensor& max_q, float q_mean, float q_std) const
    {
        auto sorted = std::get<0>(max_q.sort(0));
        int64_t n = sorted.size(0);
        if (n <= 1) return;

        // CV
        const float eps = 1e-6f;
        vars.q_cv = q_std / (std::fabs(q_mean) + eps);

        // Z-score: 平常時stdとの比
        float std_ref = vars.q_std_ema.Value();
        vars.q_z = q_std / (std_ref + eps);

        // NIQR: IQR を std_ref で正規化
        float q1 = sorted[static_cast<int64_t>(round(n * 0.25))].item<float>();
        float q3 = sorted[static_cast<int64_t>(round(n * 0.75))].item<float>();
        float iqr = q3 - q1;
        vars.q_niqr = iqr / (std_ref + eps);

        // 閾値
        bool unstable_z = (vars.q_z > config_.q_z_threshold);
        bool unstable_cv = (vars.q_cv > config_.q_cv_threshold);
        bool unstable_niqr = (vars.q_niqr > config_.q_niqr_threshold);

        vars.q_unstable = (unstable_z || unstable_cv || unstable_niqr) ? 1.0f : 0.0f;
        vars.q_unstable_ema.Update(vars.q_unstable);

        // e_t: “崩壊度”
        float ez = std::max(0.0f, vars.q_z - config_.q_z_threshold);
        float ecv = std::max(0.0f, vars.q_cv - config_.q_cv_threshold);
        float eniqr = std::max(0.0f, vars.q_niqr - config_.q_niqr_threshold);

        vars.e_t = std::max({ ez, ecv, eniqr });

        // s: ロジスティック圧縮
        float k = config_.uema_k;     // 傾き
        float x0 = config_.uema_u0;    // 中心値
        vars.s = 1.0f / (1.0f + std::exp(-k * (vars.e_t - x0)));
        vars.s_ema.Update(vars.s);

        // unstable_ema: 半減期ベースのEMA
        float alpha = std::log(2.0f) / config_.uema_half_life;
        vars.unstable_ema += alpha * (vars.s - vars.unstable_ema);
    }
};

// ================================================
// DQNAgent::StabilityController  (AS-DQN 完全実装)
// ================================================
class DQNAgent::StabilityController {
public:
    explicit StabilityController(const DQNAgentConfig& config)
        : config_(config)
    {
    }

    // ------------------------------------------------------------
    // UpdateOnStep : 毎ステップ（環境 step）
    // ・行動偏りに基づく collapse を取得
    // ・epsilon ブースト（hit側）だけを担当
    // ------------------------------------------------------------
    void UpdateOnStep(RuntimeVars& vars, StabilityMonitor& mon, step_t step) const
    {
        anet::ProfileRange  r("DQNAgent::UpdateOnStep");

        float collapse_s = ComputeCollapseOnStep(vars, mon);
        float eps_new = ComputeBoostEpsilon(vars.epsilon, collapse_s, step);
        vars.collapse_s = collapse_s;

        if (config_.use_as_dqn) {
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
        anet::ProfileRange  r("DQNAgent::UpdateOnLearn");

        float collapse_l = ComputeCollapseOnLearn(vars, mon);
        auto [eps_new, tau_new] = ComputeEpsilonTauLearn(vars.epsilon, vars.tau, collapse_l, update_step_count);
        vars.collapse_l = collapse_l;

        if (config_.use_as_dqn) {
            vars.epsilon = eps_new;
            vars.tau = tau_new;
        }
    }
private:
    const DQNAgentConfig& config_;

    /// @todo confg参照状況を整理

    // ============================================================
    // collapse（崩壊度）の設計
    // ============================================================

    // --- 行動偏りなどの Step 用 ---
    float ComputeCollapseOnStep(RuntimeVars& vars, const StabilityMonitor& mon) const
    {
        // act_unstable_ema ∈ [0,1] のイメージ
        float a = vars.act_unstable_ema.Value();

        // UEMAも利用可能（GetUnstableEma）
        float u = 0;// mon.GetUnstableEma(); /// @todo unstable_emaを考慮

        // Q統計要素まで使うと過剰なので Step側は軽めに
        float c = std::max(a, u);
        return c;
    }

    // --- Q統計などを含めた Learn 用 ---
    float ComputeCollapseOnLearn(RuntimeVars& vars, const StabilityMonitor& mon) const
    {
        float zu = vars.q_unstable_ema;       // q_unstable（Z / CV / NIQR）
        float au = vars.act_unstable_ema;     // act_unstable
        float uu = 0; // vars.unstable_ema;   // unstable_ema   /// @todo unstable_emaを考慮

        float c = std::max({ zu, au, uu });
        return c;
    }

    /// @todo config_.eps_gainは本来は別用途のConfig項目

    // ============================================================
    // ε 更新：Step 側（hit = ブースト）
    // ============================================================
    float ComputeBoostEpsilon(float eps, float collapse, step_t step) const
    {
        if (collapse < config_.eps_gain) {
            return eps; // ブーストしない
        }

        // ブースト用 half-life（崩壊時）
        float h = static_cast<float>(config_.eps_boost_half_life_hit);
        float alpha = std::log(2.0f) / h;

        float target = config_.eps_max * config_.eps_boost_max;
        float new_eps = eps + alpha * (target - eps);

        // 再加熱も加える
        new_eps = std::max(new_eps, config_.eps_reheat_floor);

        // clamp
        new_eps = std::clamp(new_eps, config_.eps_min, config_.eps_max);
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

        if (collapse > config_.eps_gain) {
            //// ---- hit（崩壊中）：回復でなく減衰方向 ----
            //float h = static_cast<float>(cfg_.eps_boost_half_life_hit);
            //float alpha = std::log(2.0f) / h;

            //float target = cfg_.eps_max * cfg_.eps_boost_max;
            //eps_new += alpha * (target - eps_new);

            //// 再加熱成分
            //eps_new = std::max(eps_new, cfg_.eps_reheat_floor);
        } else {
            // ---- recover（安定側）----
            float h = static_cast<float>(config_.eps_boost_half_life_recover);
            float alpha = std::log(2.0f) / h;

            float target = config_.eps_min;
            eps_new += alpha * (target - eps_new);

            // 軽い再加熱
            eps_new = std::max(eps_new, config_.eps_reheat_base);
        }

        // clamp
        eps_new = std::clamp(eps_new, config_.eps_min, config_.eps_max);

        // ================================
        // tau 側
        // ================================
        float tau_new = tau;

        if (collapse > config_.eps_gain) {
            // ---- hit（崩壊方向）：tau を減衰させる ----
            float h = static_cast<float>(config_.tau_half_life_hit);
            float alpha = std::log(2.0f) / h;
            float target = config_.tau_min;
            tau_new += alpha * (target - tau_new);
        } else {
            // ---- recover：一定期間安定後、tau を回復 ----
            float h = static_cast<float>(config_.tau_half_life_recover);
            float alpha = std::log(2.0f) / h;
            float target = config_.tau_max;

            tau_new += alpha * (target - tau_new);
        }

        // clamp
        tau_new = std::clamp(tau_new, config_.tau_min, config_.tau_max);

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

    float ComputeEpsilon(step_t step) const
    {
        anet::ProfileRange  r("DQNAgent::ComputeEpsilon");

        // 強制ゼロ領域
        if (config_.eps_zero_step >= 0 &&
            static_cast<int>(step) >= config_.eps_zero_step)
        {
            return 0.0f;
        }

        if (config_.eps_sigmoid_step > 0) {
            return ComputeEpsilonSigmoid(step);
        } else {
            return ComputeEpsilonDecay(step);
        }
    }
private:
    float ComputeEpsilonDecay(step_t step) const
    {
        // 自然減衰
        float decay = std::exp(-static_cast<float>(step) / config_.eps_decay_step);
        float eps = config_.eps_min + (config_.eps_max - config_.eps_min) * decay;

        // clamp
        eps = std::max(config_.eps_min, std::min(config_.eps_max, eps));
        return eps;
    }
    float ComputeEpsilonSigmoid(step_t step) const
    {
        int t_max = config_.eps_sigmoid_step;
        float eps_max = config_.eps_max;
        float eps_min = config_.eps_min;

        // int → float に自然に変換
        float mid = t_max * 0.5f;
        float k = 8.0f / t_max;   // steepness
        float s = 1.0f / (1.0f + std::exp(k * (step - mid)));

        return eps_min + (eps_max - eps_min) * s;
    }
    const DQNAgentConfig& config_;
};

// ======================================================
// DQNAgent 本体
// ======================================================
DQNAgent::DQNAgent(
    const DQNAgentConfig& config
    , const anet::rl::BatchEnvSpec& batch_env_spec, const anet::rl::EnvSpec& env_spec, const torch::Device& device
    , std::shared_ptr<anet::rl::Notifier> notifier
    , std::optional<seed_t> seed)
    : FlatStateAgent(config, device, notifier, batch_env_spec, env_spec, seed)
    , policy_net_(std::make_shared<QNetImpl>(state_dim_, n_actions_))
    , target_net_(std::make_shared<QNetImpl>(state_dim_, n_actions_))
{
    ANET_LOG_DEBUG("seed=" << GetSeed());

    //seed
    anet::SeedMaker seed_maker(GetSeed());
    auto replay_seed = seed_maker.MakeNamedSeed("replaybuffer");
    auto action_decider_seed = seed_maker.MakeNamedSeed("action_decider");

    // use_replay_buffer=false の場合の強制
    if (!config_.use_replay_buffer) {
        config_.replay_capacity = 1;
        config_.replay_batch_size = 1;
        config_.replay_warmup_steps = 0;
        config_.replay_update_interval = 1;
    }

    if (config_.nn_init_mode == 1) {
        policy_net_->InitWeightsWithXavierUniform();
    } else if (config_.nn_init_mode == 2) {
        policy_net_->InitWeightsWithHeNormal();
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
    this->replay_buffer_ = std::make_unique<anet::rl::PlainReplayBuffer>(env_spec, batch_size_ * config_.replay_capacity, replay_seed);
    this->action_decider_ = std::make_unique<ActionDecider>(*this, action_decider_seed);
    this->replay_scheduler_ = std::make_unique<ReplayScheduler>(this->config_);
    this->target_updater_ = std::make_unique<TargetUpdater>(this->config_);
    this->stability_monitor_ = std::make_unique<StabilityMonitor>(this->config_);
    this->stability_controller_ = std::make_unique<StabilityController>(this->config_);

    // 内部状態変数を初期化
    vars_updater_->Initilize(*this->vars_);

    // ログ：パラメータ記録
    anet::log::info() << "DQNAgent config=" << config_;
    anet::MetricsLogger::Instance()->LogJson("agent/params", config_.ToJson());
    anet::MetricsLogger::Instance()->Flush();
}


anet::rl::BatchActionInfo DQNAgent::MakeAction(
    const StepCounts& step, const anet::rl::BatchState& state, anet::rl::RunMode mode) const
{
    ProfileRange r1("DQNAgent::MakeAction");
    ANET_CHECK_SHAPE(state.obs, { ANY, state_dim_ });

    auto flat_state = state.Flatten();
    auto flat_obs = flat_state.obs.to(device_);
    bool greedy_only = anet::rl::IsEval(mode);

    torch::NoGradGuard ng;
    std::shared_lock<std::shared_mutex> lock(*mutex_);

    ProfileRange r2("DQNAgent::MakeAction.forward");

    torch::Tensor q;
    if (mode == anet::rl::RunMode::Eval1)
        q = target_net_->forward(flat_obs);
    else
        q = policy_net_->forward(flat_obs);
    //ANET_CHECK_SHAPE(q, { ANET_SHAPE_ANY, n_actions_ });  // (N, n_actions_)

    r2.End();

    auto act_info = action_decider_->DecideBatch(q, greedy_only);

    //ANET_CHECK_SHAPE(act_info.action, { state.obs.size(0) });
    //ANET_CHECK_SHAPE(act_info.is_random, { state.obs.size(0) });

    ProfileRange r3("DQNAgent::MakeAction.update");

    if (mode == anet::rl::RunMode::Train) {
        // 行動統計の更新
        stability_monitor_->UpdateActionStats(*vars_, act_info);

        // 崩壊情報更新
        stability_controller_->UpdateOnStep(*vars_, *stability_monitor_, vars_->learn_step);
    }

    return act_info;
}

std::shared_ptr<const anet::rl::BatchUpdateResult>
DQNAgent::UpdateFromBatch(const StepCounts& counts, const anet::rl::BatchExperience& batch_exp, const anet::rl::Runner& runner)
{
    ProfileRange r1("DQNAgent::UpdateFromBatch");

    bool can_update;

    std::shared_ptr<DQNAgent::BatchUpdateResult> result;
    {
        std::unique_lock<std::shared_mutex> lock(*mutex_);

        // ReplayBuffer に push
        replay_buffer_->Push(batch_exp);

        // 学習タイミング判定
        can_update = replay_scheduler_->CanUpdate(counts.update_step, batch_size_, *replay_buffer_);

        uint32_t learn_step_diff = 0;

        if (can_update) {
            const int B = config_.replay_batch_size;
            auto raw_samples = replay_buffer_->Sample(B, device_);

            // device / shape チェック（dtype は Push 時点で保証済み）
            ANET_CHECK_DEVICE(raw_samples.obs, device_);
            ANET_CHECK_DEVICE(raw_samples.actions, device_);
            ANET_CHECK_DEVICE(raw_samples.target_values, device_);
            ANET_CHECK_DEVICE(raw_samples.next_states.obs, device_);
            ANET_CHECK_DEVICE(raw_samples.next_states.terminals, device_);
            ANET_CHECK_SHAPE(raw_samples.obs, { B, state_dim_ });
            ANET_CHECK_SHAPE(raw_samples.actions, { B, 1 });    // 離散アクション
            ANET_CHECK_SHAPE(raw_samples.target_values, { B });
            ANET_CHECK_SHAPE(raw_samples.next_states.obs, { B, state_dim_ });
            ANET_CHECK_SHAPE(raw_samples.next_states.terminals, { B });
            ANET_CHECK_DTYPE(raw_samples.obs, torch::kFloat32);
            ANET_CHECK_DTYPE(raw_samples.actions, torch::kInt64);    // 離散アクション
            ANET_CHECK_DTYPE(raw_samples.target_values, torch::kFloat32);
            ANET_CHECK_DTYPE(raw_samples.next_states.terminals, torch::kBool);
            ANET_LOG_DEBUG("ReplayBuffer batch OK: B=" << raw_samples.obs.size(0));

            ProfileRange r2("DQNAgent::UpdateFromBatch.forward");

            // ReplayBufferから取り出した時点では生の多次元StateなのでFlattenする
            auto samples = raw_samples.FlattenStates();

            // Q(s, a) 生成
            torch::Tensor q_all = policy_net_->forward(samples.obs); // (B, n_actions_)
            ANET_CHECK_SHAPE(q_all, { B, n_actions_ });
            //ANET_LOG_DEBUG("q_all=" << anet::ToString(q_all));

            // max_a Q(s,a)  (AS-DQN 用統計)
            torch::Tensor max_q = std::get<0>(q_all.max(1)); // (B,)
            auto result_max_q = max_q;

            // Q(s,a) for taken action
            torch::Tensor actions_b = samples.actions.view({ B, 1 });   // (B,1)
            ANET_CHECK_SHAPE(actions_b, { B, 1 });
            ANET_CHECK_DTYPE(actions_b, torch::kInt64);
            //ANET_LOG_DEBUG("actions_b=" << anet::ToString(actions_b));
            torch::Tensor q_sa = q_all.gather(1, actions_b).squeeze(1); // (B,)
            ANET_CHECK_SHAPE(q_sa, { B });
            //ANET_LOG_DEBUG("q_sa=" << anet::ToString(q_sa));

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

            r2.End();   // forward

            anet::ProfileRange r3("DQNAgent::UpdateFromBatch.backward");

            // -------------------------------------------------
            // TD target 計算
            //    td_target = r + (1 - terminal) * gamma * max_next_q
            // -------------------------------------------------
            torch::Tensor not_terminal = 1.0f - samples.next_states.terminals.to(torch::kFloat32); // (B,)
            torch::Tensor rewards = samples.target_values; // (B,)
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

            r3.End();
            ProfileRange r4("DQNAgent::UpdateFromBatch.update");

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

            // optimizerを動かしたので step数を更新
            vars_->learn_step++;
            learn_step_diff++;

            // target_network更新
            target_updater_->Sync(vars_->learn_step, vars_->tau, policy_net_, target_net_);

            // StabilityMonitor 更新（TD / loss / Q / 勾配ノルム）
            float loss_scalar = loss_tensor.item<float>();
            stability_monitor_->UpdateBatchStats(
                *vars_,
                td_error_raw,
                per_sample_loss,
                max_q,
                grad_norm,
                grad_clip_ratio);

            // 崩壊情報を更新
            stability_controller_->UpdateOnLearn(*vars_, *stability_monitor_, vars_->learn_step);

            // 戻り値用オブジェクト生成
            result = std::make_shared<DQNAgent::BatchUpdateResult>(*vars_, result_max_q, learn_step_diff);

        }
        else {  // can_update
            result = std::make_shared<DQNAgent::BatchUpdateResult>(*vars_, std::nullopt, learn_step_diff);
        }

        // 内部状態変数を更新
        if (!config_.use_as_dqn) {
            vars_->epsilon = vars_updater_->ComputeEpsilon(vars_->learn_step);
        }
    }

    // LearnEvent通知
    if (notifier_ != nullptr && can_update) {
        anet::rl::LearnEvent event{ batch_exp, runner, counts, shared_from_this(), result };
        notifier_->Notify(event);
    }

    // 戻り用変数

    return result;
}

DQNAgentFactory::DQNAgentFactory()
{
    ;
}

std::shared_ptr<Agent> DQNAgentFactory::CreateAgent(
    const EnvSpec& env_spec, const BatchEnvSpec& batch_env_spec,
    const torch::Device& device, const anet::ConfigData& config_data,
    std::shared_ptr<anet::rl::Notifier> notifier, std::optional<anet::seed_t> seed) const
{
    DQNAgentConfig config(config_data);
    auto agent = std::make_shared<DQNAgent>(config, batch_env_spec, env_spec, device, notifier, seed);
    return agent;

    /// @todo 引数の順番を統一
}

//ANET_REGISTER_AGENT_FACTORY(DQNAgentFactory);

