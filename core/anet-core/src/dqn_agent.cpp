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

//const float met_ema_decay = 0.995f;  // 平滑化係数(メトリクス用)
//const float met_ema_decay_act = 0.9995f;  // 平滑化係数(メトリクス用)action_ema用
//const float met_ema_decay_reward = 0.9995f;  // 平滑化係数(メトリクス用)action_ema用

namespace {
    static constexpr int64_t ANY = ANET_SHAPE_ANY;
}


// ---- 内部モジュール達 ----


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
        : agent_(agent),
        config_(agent.config_),
        rnd_(agent.rnd_)
    {
    }

    /// q_values: (1, n_actions)
    std::tuple<int, bool> Decide(const torch::Tensor& q_values, size_t step, bool greedy_only)
    {
        // 1) 常に greedy モード
        if (greedy_only) {
            return { GreedyAction(q_values), false };
        }

        // 2) εを計算
        float eps = ComputeEpsilon(step);

        // 3) ε-greedy
        if (rnd_->Uniform01() < eps) {
            return { RandomAction(), true };
        } else {
            return { GreedyAction(q_values), false };
        }
    }

    // ----------------------------------------------------
    // εスケジューリング（標準DQN）
    // ----------------------------------------------------
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
        if (config_.softupdate_tau > 0.0f) {
            SoftSync(policy_net, target_net, config_.softupdate_tau);
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


// ===============================
// DQNAgent::StabilityMonitor
// ===============================
class DQNAgent::StabilityMonitor {
public:
    StabilityMonitor(const DQNAgentConfig& config) : config_(config) {}

    void Update(const torch::Tensor& td_error)
    {
        ANET_CHECK_SHAPE(td_error, { ANY }); // (B; )
        /// @todo Implement EMA update for TD-error stability
    }

    float GetTdErrorEma() const
    {
        return td_error_ema_;
    }
private:
    const DQNAgentConfig& config_;
    float td_error_ema_ = 0.0f;  // @todo Decide initial EMA value
}; 

struct DQNUpdateResult : public UpdateResult {
    float td_error_ema = 0.0f;
    float loss = 0.0f;
    float epsilon = 1.0f;
    /// @todo ラインナップ精査

    virtual MetricsMap GetMetricsMap() const override{
        MetricsMap map;
        map["37_agent_dqn_qtd/03_td_error"] = td_error_ema;
        map["38_agent_dqn_loss/01_loss"] = loss;
        map["32_agent_dqn_base/03_epsilon"] = epsilon;
        return map;
    }
};

// ======================================================
// DQNAgent 本体
// ======================================================
DQNAgent::DQNAgent(const DQNAgentConfig& config, anet::rl::EnvSpec& env_spec, torch::Device device, anet::RandomGenerator* rnd) :
    StepBasedAgent(config, device, rnd),
    state_dim_(env_spec.state.CalcStateDim()),
    n_actions_(env_spec.action.ActionCount()),
    policy_net_(std::make_shared<QNetImpl>(state_dim_, n_actions_)),
    target_net_(std::make_shared<QNetImpl>(state_dim_, n_actions_)),
    optimizer(policy_net_->parameters(), torch::optim::AdamOptions(config_.alpha))
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

    // 内部モジュール生成
    this->replay_buffer_ = std::make_unique<anet::rl::ReplayBuffer>(env_spec, config_.replay_capacity, rnd);
    this->action_decider_ = std::make_unique<ActionDecider>(*this);
    this->replay_scheduler_ = std::make_unique<ReplayScheduler>(this->config_);
    this->stability_monitor_ = std::make_unique<StabilityMonitor>(this->config_);
    this->target_updater_ = std::make_unique<TargetUpdater>(this->config_);

    // ログ：パラメータ記録
    wxLogInfo("DQNAgent config=%s", config_.ToStdString());
    anet::MetricsLogger::Instance()->log_json("agent/params", config_.ToJson());
    anet::MetricsLogger::Instance()->flush();
}


anet::rl::BatchActionInfo DQNAgent::MakeAction(const torch::Tensor& state, anet::rl::RunMode mode)
{
    ANET_CHECK_SHAPE(state, { ANY, state_dim_ });

    torch::Tensor s = state.to(device_);

    torch::NoGradGuard ng;
    torch::Tensor q;
    if (mode == anet::rl::RunMode::Eval1) {
        q = target_net_->forward(s);
    }
    else {
        q = policy_net_->forward(s);
    }

    // Eval → greedy-only
    bool greedy_only = (mode == anet::rl::RunMode::Eval1 || mode == anet::rl::RunMode::Eval2);
    auto [ a, rand ] = action_decider_->Decide(q, step_count_, greedy_only);

    // (N=1) batched ActionInfo
    torch::Tensor action = torch::tensor({ { a } }, torch::kInt64).to(device_);
    torch::Tensor is_rand = torch::tensor({ { rand } }, torch::kBool).to(device_);

    return { action, is_rand };
}

std::shared_ptr<const anet::rl::UpdateResult>
DQNAgent::UpdateFromBatch(const anet::rl::BatchExperience& batch_exp)
{
    // ReplayBuffer に push
    replay_buffer_->Push(batch_exp);

    // step カウンタ更新
    step_count_++;

    float loss_value = 0.0f;

    // 学習タイミング判定
    const bool can_update = replay_scheduler_->CanUpdate(step_count_, *replay_buffer_);

    if (can_update) {
        const int B = config_.replay_batch_size;
        auto samples = replay_buffer_->Sample(B, device_);

        // device / shape チェック（dtype は Push 時点で保証済み）
        ANET_CHECK_DEVICE(samples.states, device_);
        ANET_CHECK_DEVICE(samples.next_states, device_);
        ANET_CHECK_DEVICE(samples.rewards, device_);
        ANET_CHECK_DEVICE(samples.actions, device_);
        ANET_CHECK_DEVICE(samples.dones, device_);
        ANET_CHECK_DEVICE(samples.truncateds, device_);
        ANET_CHECK_SHAPE(samples.states, { B, state_dim_ });
        ANET_CHECK_SHAPE(samples.next_states, { B, state_dim_ });
        ANET_CHECK_SHAPE(samples.rewards, { B });
        ANET_CHECK_SHAPE(samples.actions, { B, 1 });
        ANET_CHECK_SHAPE(samples.dones, { B });
        ANET_CHECK_SHAPE(samples.truncateds, { B });
        ANET_CHECK_DTYPE(samples.states, torch::kFloat32);
        ANET_CHECK_DTYPE(samples.actions, torch::kInt64);
        ANET_CHECK_DTYPE(samples.rewards, torch::kFloat32);
        ANET_CHECK_DTYPE(samples.next_states, torch::kFloat32);
        ANET_CHECK_DTYPE(samples.dones, torch::kBool);
        ANET_CHECK_DTYPE(samples.truncateds, torch::kBool);

        wxLogDebug("ReplayBuffer batch OK: B=%lld", samples.states.size(0));

        // -------------------------------------------------
        // 4. Q(s, a) 抽出
        // -------------------------------------------------
        torch::Tensor q_all = policy_net_->forward(samples.states); // (B, n_actions_)
        ANET_CHECK_SHAPE(q_all, { B, n_actions_ });
        //wxLogDebug("q_all=%s", anet::ToString(q_all));

        torch::Tensor actions_b = samples.actions.view({ B, 1 });   // (B,1)
        ANET_CHECK_SHAPE(actions_b, { B, 1 });
        ANET_CHECK_DTYPE(actions_b, torch::kInt64);
        //wxLogDebug("actions_b=%s", anet::ToString(actions_b));

        torch::Tensor q_sa = q_all.gather(1, actions_b).squeeze(1); // (B,)
        ANET_CHECK_SHAPE(q_sa, { B });
        //wxLogDebug("q_sa=%s", anet::ToString(q_sa));

        // -------------------------------------------------
        // 5. max_a' Q_target(s', a')（DQN / DoubleDQN 切替）
        // -------------------------------------------------
        torch::Tensor max_next_q;

        if (config_.use_double_dqn) {
            torch::NoGradGuard no_grad;

            // 5-1) policy_net で argmax_a Q(s', a)
            torch::Tensor q_next_policy = policy_net_->forward(samples.next_states); // (B, n_actions_)
            ANET_CHECK_SHAPE(q_next_policy, { B, n_actions_ });
            auto next_policy_pair = q_next_policy.max(1);
            torch::Tensor next_actions = std::get<1>(next_policy_pair); // (B,)

            // 5-2) target_net で Q_target(s', argmax_a Q_online)
            torch::Tensor q_next_target = target_net_->forward(samples.next_states); // (B, n_actions_)
            ANET_CHECK_SHAPE(q_next_target, { B, n_actions_ });
            torch::Tensor next_actions_b = next_actions.view({ B, 1 });             // (B,1)
            torch::Tensor q_next_selected =
                q_next_target.gather(1, next_actions_b).squeeze(1);                 // (B,)

            max_next_q = q_next_selected;
        } else {
            torch::NoGradGuard no_grad;

            // 通常 DQN: max_a' Q_target(s', a')
            torch::Tensor q_next_all = target_net_->forward(samples.next_states); // (B, n_actions_)
            ANET_CHECK_SHAPE(q_next_all, { B, n_actions_ });
            max_next_q = std::get<0>(q_next_all.max(1));                         // (B,)
        }

        // -------------------------------------------------
        // 6. TD target 計算
        //    td_target = r + (1 - terminal) * gamma * max_next_q
        // -------------------------------------------------
        torch::Tensor terminal = (samples.dones | samples.truncateds); // (B,) bool
        torch::Tensor not_terminal = 1.0f - terminal.to(torch::kFloat32); // (B,)
        torch::Tensor rewards = samples.rewards; // (B,)
        const float gamma = config_.gamma;
        torch::Tensor td_target = rewards + not_terminal * (gamma * max_next_q); // (B,)

        // -------------------------------------------------
        // 7. TD 誤差と loss（td_clip, Huber）
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

        // -------------------------------------------------
        // 8. optimizer step（grad clip 含む）
        // -------------------------------------------------
        optimizer.zero_grad();
        loss_tensor.backward();
        if (config_.use_grad_clip) {
            // 勾配の L2 ノルムを制限
            torch::nn::utils::clip_grad_norm_(
                policy_net_->parameters(),
                config_.grad_clip_tau
            );
        }
        optimizer.step();

        // -------------------------------------------------
        // 9. StabilityMonitor 更新（生 TD 誤差で監視）
        // -------------------------------------------------
        stability_monitor_->Update(td_error_raw);

        // -------------------------------------------------
        // 10. TargetUpdater による同期
        // -------------------------------------------------
        target_updater_->Sync(step_count_, policy_net_, target_net_);

        // -------------------------------------------------
        // 11. loss スカラー取得
        // -------------------------------------------------
        loss_value = loss_tensor.item<float>();
    }

    // 12. UpdateResult
    auto result = std::make_shared<DQNUpdateResult>();
    result->td_error_ema = stability_monitor_->GetTdErrorEma();
    result->loss = loss_value;
    result->epsilon = action_decider_->ComputeEpsilon(step_count_);

    return result;
}


