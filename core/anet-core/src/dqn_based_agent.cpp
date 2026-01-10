
#include "dqn_based_agent.hpp"
#include <tuple>
#include <cmath>
#include "anet/log.hpp"
#include "anet/profile.hpp"
#include "anet/tensor_check.hpp"
#include "anet/tensor_util.hpp"
#include "anet/str_util.hpp"
#include "anet/replay_buffer.hpp"

using namespace anet::rl::dqn;
namespace LOG = anet::log;


// ======================================================
// Network
// ======================================================

Network::Network(
    const NetworkConfig& config, const torch::Device& device,
    std::shared_ptr<anet::nn::Network> policy_net, std::shared_ptr<anet::nn::Network> target_net,
    int64_t n_actions,int64_t num_quantiles)
    : config_(config), policy_net_(std::move(policy_net)), target_net_(std::move(target_net))
    , n_actions_(n_actions)
    , num_quantiles_(num_quantiles)
{
    ANET_ASSERT(policy_net_);
    ANET_ASSERT(target_net_);
    ANET_ASSERT(n_actions_ > 0);
    ANET_ASSERT(num_quantiles_ > 0);

    policy_net_->to(device);    /// @todo Agent側で実行されるので削除？
    target_net_->to(device);
    target_net_->eval();
}

torch::Tensor Network::Forward(const torch::Tensor& obs, bool use_target) const
{
    const auto& net = use_target ? target_net_ : policy_net_;

    // 生の出力:
    //  Plain Head    -> (B, A)
    //  Quantile Head -> (B, A*N)
    auto output = net->Forward(obs);

    if (num_quantiles_ > 1) {
        // QR-DQNの場合: 分布の平均を取って Q(s,a) を返す
        auto batch_size = output.size(0);
        // Reshape: (B, A*N) -> (B, A, N)
        auto reshaped = output.view({ batch_size, n_actions_, num_quantiles_ });
        // Mean: (B, A, N) -> (B, A)
        return reshaped.mean(2);
    } else {
        // DQNの場合: そのまま返す (B, A)
        return output;
    }
}

torch::Tensor Network::ForwardQuantiles(const torch::Tensor& obs, bool use_target) const
{
    // QR-DQNでない場合は呼んではいけない
    ANET_ASSERT(num_quantiles_ > 1);

    const auto& net = use_target ? target_net_ : policy_net_;
    auto output = net->Forward(obs); // (B, A*N)

    // Reshape: (B, A*N) -> (B, A, N)
    auto batch_size = output.size(0);
    return output.view({ batch_size, n_actions_, num_quantiles_ });
}

bool Network::IsDistributional(bool use_target) const
{
    return (num_quantiles_ > 1);
}

std::vector<torch::Tensor> Network::GetPolicyParameters() const
{
    auto params = policy_net_->parameters();
    return params;
}

void Network::UpdateTarget(step_t learn_step)
{
    if (config_.hard_update_interval > 0) {
        if (learn_step % config_.hard_update_interval == 0) {
            HardUpdate();
        }
        return;
    }
    SoftUpdate();
}

void Network::SoftUpdate()
{
    // @todo optimizer 非依存での soft update 実装
    auto p_params = policy_net_->parameters();
    auto t_params = target_net_->parameters();

    ANET_ASSERT(p_params.size() == t_params.size());

    for (size_t i = 0; i < p_params.size(); ++i) {
        t_params[i].data().mul_(1.0f - config_.soft_update_tau);
        t_params[i].data().add_(p_params[i].data(), config_.soft_update_tau);
    }
}

void Network::HardUpdate()
{
    auto p_params = policy_net_->parameters();
    auto t_params = target_net_->parameters();

    ANET_ASSERT(p_params.size() == t_params.size());

    for (size_t i = 0; i < p_params.size(); ++i) {
        t_params[i].data().copy_(p_params[i].data());
    }
}

/// メトリクス用：NN生出力
std::optional<anet::TensorFunction> Network::GetTensorFunction(const std::string& key, const torch::Device& device)
{
    static constexpr const char* POLICY_PREFIX = "policy-net.";
    static constexpr const char* TARGET_PREFIX = "target-net.";

    // policy net
    if (anet::StartsWith(key, POLICY_PREFIX)) {
        auto subkey = anet::RemovePrefix(key, POLICY_PREFIX);
        auto fn = policy_net_->GetTensorFunction(subkey);
        return fn;
    }

    // target net
    if (anet::StartsWith(key, TARGET_PREFIX)) {
        auto subkey = anet::RemovePrefix(key, TARGET_PREFIX);
        auto fn = target_net_->GetTensorFunction(subkey);
        return fn;
    }

    return std::nullopt;
}


// ======================================================
// ActionPolicy 
// ======================================================

anet::rl::dqn::ActionPolicy::ActionPolicy(const ActionPolicyConfig& config,
    const anet::rl::dqn::Network& network, RuntimeVars& vars, anet::seed_t seed)
    : config_(config), anet::RandomHolder(seed), network_(network), vars_(vars)
{
    // RandomHolderを継承しているが、基本GPU側で乱数生成しているので意味はない。ただ、マークとしてそのままにしておく。
}

torch::Tensor anet::rl::dqn::ActionPolicy::MakeEpsilonGreedyAction(const torch::Tensor& greedy_action, float epsilon, int64_t batch_size, int64_t n_actions) const
{
    ProfileRange  r("ActionPolicy::MakeEpsilonGreedyAction");

    auto device = greedy_action.device();

    // mask: (N) bool, GPU上で生成
    auto mask = torch::rand({ batch_size }, torch::TensorOptions().device(device)).lt(epsilon);    // GPUで完結

    // random actions (N) int64
    auto random_actions = torch::randint(/*low=*/0, /*high=*/n_actions, { batch_size },
        torch::TensorOptions().dtype(torch::kInt64).device(device));

    // actions: where(mask, random_actions, greedy)
    auto actions = torch::where(mask, random_actions, greedy_action);

    return actions;
}

torch::Tensor anet::rl::dqn::ActionPolicy::GetQuantiles(const torch::Tensor& obs, bool use_target) const
{
    if (network_.IsDistributional(use_target)) {
        return network_.ForwardQuantiles(obs, use_target);
    }
    return torch::Tensor();
}

anet::rl::BatchActionInfo anet::rl::dqn::ActionPolicy::MakeActionInfo(const torch::Tensor& action_values, const torch::Tensor& q_values, const torch::Tensor& q_quantiles) const
{
    ProfileRange  r("ActionPolicy::MakeActionInfo");

    BatchActionInfo action_info{ action_values };
    auto& aux = action_info.GetAuxData();

    // aux[max_q]
    auto max_pair = q_values.max(1);
    auto max_q = std::get<0>(max_pair).detach();
    aux["max_q"] = max_q;
    aux["q_values"] = q_values;
    aux["q_quantiles"] = q_quantiles;

    return action_info;
}

void anet::rl::dqn::ActionPolicy::UpdateEpsilon(step_t step, bool is_uqe)
{
    if (is_uqe) {
        if (config_.uqe_eps_decay_step <= 0) return;
        if (step >= config_.uqe_eps_decay_step) {
            vars_.epsilon = config_.uqe_eps_min;
        } else {
            const float t = static_cast<float>(step) / static_cast<float>(config_.uqe_eps_decay_step);
            vars_.epsilon = config_.uqe_eps_max + t * (config_.uqe_eps_min - config_.uqe_eps_max);
        }
    } else {
        if (config_.eps_decay_step <= 0) return;
        if (step >= config_.eps_decay_step) {
            vars_.epsilon = config_.eps_min;
        } else {
            const float t = static_cast<float>(step) / static_cast<float>(config_.eps_decay_step);
            vars_.epsilon = config_.eps_max + t * (config_.eps_min - config_.eps_max);
        }
    }
}

// ======================================================
// EpsilonGreedyActionPolicy 
// ======================================================

anet::rl::dqn::EpsilonGreedyActionPolicy::EpsilonGreedyActionPolicy(const ActionPolicyConfig& config,
    const anet::rl::dqn::Network& network, RuntimeVars& vars, anet::seed_t seed)
    : ActionPolicy(config, network, vars, seed)
{
    vars_.epsilon = config_.eps_max;
}

void anet::rl::dqn::EpsilonGreedyActionPolicy::OnLearn(const StepCounts& counts)
{
    UpdateEpsilon(counts.exp_step);
}

anet::rl::BatchActionInfo EpsilonGreedyActionPolicy::SelectAction(const torch::Tensor& obs, bool greedy_only, bool use_target) const
{
    ProfileRange r("EpsilonGreedyActionPolicy::SelectAction");

    torch::NoGradGuard guard;

    // Q値生成
    auto q_quantiles = GetQuantiles(obs, use_target);
    auto q_values = q_quantiles.defined() ? q_quantiles.mean(-1) : network_.Forward(obs, use_target);
    auto greedy_action = q_values.argmax(1, /*keepdim=*/false);        // greedy = argmax(q_values, dim=1)

    // Greedy指定ならargmxを返す
    if (greedy_only) return MakeActionInfo(greedy_action, q_values, q_quantiles);

    // EpsilonGreedy
    const int64_t N = q_values.sizes()[0];      // shape 読み取りは TensorOptions 経由で同期を回避
    const int64_t A = q_values.sizes()[1];
    auto actions = MakeEpsilonGreedyAction(greedy_action, vars_.epsilon, N, A);
    auto action_info = MakeActionInfo(actions, q_values, q_quantiles);
    return action_info;
}

// ======================================================
// UQEActionPolicy
// ======================================================

anet::rl::dqn::UQEActionPolicy::UQEActionPolicy(const ActionPolicyConfig& config,
    const anet::rl::dqn::Network& network, RuntimeVars& vars, anet::seed_t seed)
    : ActionPolicy(config, network, vars, seed)
{
    vars_.epsilon = config_.uqe_eps_max;
    vars_.uqe_tau = config_.uqe_tau_max;
}

void anet::rl::dqn::UQEActionPolicy::UpdateTau(step_t step)
{
    if (config_.uqe_tau_decay_step <= 0) return;
    if (step >= config_.uqe_tau_decay_step) {
        vars_.uqe_tau = config_.uqe_tau_min;
        return;
    }
    const float t = static_cast<float>(step) / static_cast<float>(config_.uqe_tau_decay_step);
    vars_.uqe_tau = config_.uqe_tau_max + t * (config_.uqe_tau_min - config_.uqe_tau_max);
}

void anet::rl::dqn::UQEActionPolicy::OnLearn(const StepCounts& counts)
{
    UpdateEpsilon(counts.exp_step, true);
    UpdateTau(counts.exp_step);
}

torch::Tensor UQEActionPolicy::MakeUQEAction(float tau, const torch::Tensor& q_quantiles) const
{
    ProfileRange r("UQEActionPolicy::MakeUQEAction");

    // GPU同期を避けるため、sizes() からメタデータのみ取得
    const int64_t n_quantiles = q_quantiles.size(-1);

    // インデックス決定: floor(tau * (N-1)) clampして範囲外アクセスを防止
    int64_t tau_idx = static_cast<int64_t>(tau * (n_quantiles - 1));
    tau_idx = std::max<int64_t>(0, std::min<int64_t>(tau_idx, n_quantiles - 1));
    ANET_LOG_DEBUG("tau_idx=" << tau_idx);

    torch::Tensor uqe_values;
    if (config_.uqe_use_tail_mean) {
        // 上位分位点すべての平均を使う場合

        // tau_idx から 最後まで (N-1) の範囲を切り出す
        auto tail_values = q_quantiles.slice(-1, tau_idx, n_quantiles);  // (B, A, N - tau_idx)
        ANET_LOG_DEBUG("tail_values=" << anet::ToString(tail_values));

        // 切り出した範囲の平均をとる
        uqe_values = tail_values.mean(-1);
    } else {
        // 特定の分位点におけるQ値を取得
        uqe_values = q_quantiles.select(-1, tau_idx);  // (B, A, N) -> (B, A)
    }
    ANET_LOG_DEBUG("uqe_values=" << anet::ToString(uqe_values));

    // UQE Actions: argmax(Q_tau)
    auto actions = uqe_values.argmax(1);
    return actions;
}

torch::Tensor UQEActionPolicy::MakeVectorizedUQEAction(const torch::Tensor& tau_tensor, const torch::Tensor& q_quantiles) const
{
    // q_quantiles: (N, A, n_quantiles)
    // tau_tensor:  (N, 1)  (0.0 ~ 1.0)

    const int64_t N = q_quantiles.size(0);
    const int64_t A = q_quantiles.size(1);
    const int64_t n_quantiles = q_quantiles.size(2);
    auto device = q_quantiles.device();

    // 1. インデックスの計算 (N, 1)
    // floor(tau * (n_q - 1))
    auto tau_idx = (tau_tensor * (n_quantiles - 1)).to(torch::kLong);
    tau_idx = tau_idx.clamp(0, n_quantiles - 1);

    torch::Tensor uqe_values;

    if (config_.uqe_use_tail_mean) {
        // 【難所】Tail Meanの場合 (バッチごとに開始位置が違うため slice は使えない)
        // マスクを使って平均を計算する

        // range: (1, 1, n_quantiles) -> [0, 1, 2, ...]
        auto range = torch::arange(n_quantiles, device).view({ 1, 1, -1 });

        // mask: (N, 1, n_quantiles)
        // range >= tau_idx の部分が True (1.0)
        // tau_idx は (N, 1) なので (N, 1, 1) に view して broadcast
        auto mask = range.ge(tau_idx.view({ N, 1, 1 })).to(q_quantiles.dtype());

        // 平均計算: (sum(Q * mask) / sum(mask))
        // small_epsilon を足して 0除算防止
        uqe_values = (q_quantiles * mask).sum(-1) / (mask.sum(-1) + 1e-6);
    } else {
        // tau_idx を (N, A, 1) に拡張
        // バッチ(N)ごとに違うインデックスだが、Action(A)に対しては同じインデックスを使う
        auto gather_idx = tau_idx.view({ N, 1, 1 }).expand({ N, A, 1 });

        // gather: dim=-1 (quantiles次元) に沿って収集
        // output: (N, A, 1) -> squeeze -> (N, A)
        uqe_values = q_quantiles.gather(-1, gather_idx).squeeze(-1);
    }

    return uqe_values.argmax(1);
}

anet::rl::BatchActionInfo UQEActionPolicy::MakeUQEActionInfo(float tau, const torch::Tensor& tau_tensor, const torch::Tensor& obs, bool greedy_only, bool use_target) const
{
    ProfileRange r("UQEActionPolicy::MakeUQEActionInfo");

    torch::NoGradGuard guard;

    auto q_quantiles = GetQuantiles(obs, use_target);
    auto q_values = q_quantiles.mean(-1);

    // greedy_only (評価モード) の場合、「平均値(リスク中立)」を使う
    if (greedy_only) {
        auto greedy_action = q_values.argmax(1, /*keepdim=*/false);        // greedy = argmax(q_values, dim=1)
        return MakeActionInfo(greedy_action, q_values, q_quantiles);
    }

    const float epsilon = vars_.epsilon;
    const int64_t N = q_values.sizes()[0];      // shape 読み取りは TensorOptions 経由で同期を回避
    const int64_t A = q_values.sizes()[1];

    torch::Tensor uqe_action;
    if (tau_tensor.defined()) {
        uqe_action = MakeVectorizedUQEAction(tau_tensor, q_quantiles);
    } else {
        uqe_action = MakeUQEAction(tau, q_quantiles);
    }
    auto actions = MakeEpsilonGreedyAction(uqe_action, epsilon, N, A);
    auto action_info = MakeActionInfo(actions, q_values, q_quantiles);
    return action_info;
}

anet::rl::BatchActionInfo UQEActionPolicy::SelectAction(const torch::Tensor& obs, bool greedy_only, bool use_target) const
{
    ANET_ASSERT(network_.IsDistributional(use_target)); // 念の為チェック

    return MakeUQEActionInfo(vars_.uqe_tau, torch::Tensor(), obs, greedy_only, use_target);
}


// ======================================================
// ThompsonSamplingActionPolicy
// ======================================================

anet::rl::dqn::ThompsonSamplingActionPolicy::ThompsonSamplingActionPolicy(const ActionPolicyConfig& config,
    const anet::rl::dqn::Network& network, RuntimeVars& vars, anet::seed_t seed)
    : UQEActionPolicy(config, network, vars, seed)
{
    ;
}

void anet::rl::dqn::ThompsonSamplingActionPolicy::OnLearn(const StepCounts& counts)
{
    UpdateEpsilon(counts.exp_step, true);
    //UpdateTau(counts.exp_step);
}

anet::rl::BatchActionInfo ThompsonSamplingActionPolicy::SelectAction(const torch::Tensor& obs, bool greedy_only, bool use_target) const
{
    ANET_ASSERT(network_.IsDistributional(use_target)); // 念の為チェック

    // ランダムな Tau をバッチサイズ分生成 (N, 1)
    const int64_t N = obs.size(0);
    auto tau_tensor = torch::rand({ N, 1 }, torch::TensorOptions().device(obs.device()));

    // tau_tensor(ランダム)でUQE適用
    return MakeUQEActionInfo(0.0f, tau_tensor, obs, greedy_only, use_target);
}


// ======================================================
// Learner
// ======================================================

Learner::Learner(const LearnerConfig& config, Network& network, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
    const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed)
    : config_(config), network_(network), vars_(vars), obs_norm_(obs_norm)
    , batch_size_(batch_env_spec.batch_size)
    , n_actions_(env_spec.action_spec.GetNumActions()), state_dim_(env_spec.state_spec.CalcFlattenDim())
    , device_(std::move(device))
{
    // Credit計算
    if (config_.replay_ratio > 0) {
        // RRモード:  U = (N * RR) / B
        earned_credit_ = static_cast<float>(batch_size_) * config_.replay_ratio / config_.replay_batch_size;
    } else {
        // Intervalモード: U = 1.0 / Interval
        earned_credit_ = 1.0f / static_cast<float>(std::max(1, config_.update_interval));
    }
    
    LOG::info() << "Learner: U = " << earned_credit_;
}

std::optional<float> Learner::GetScalar(const std::string& key, int64_t index) const
{
    if (key.find(ReplayBuffer::kKeyPrefix) == 0 && replay_buffer_ != nullptr) {
        return replay_buffer_->GetScalar(key);
    }

    return std::nullopt;
}

std::optional<torch::Tensor> Learner::GetTensor(const std::string& key, int64_t index) const
{
    if (key.find(ReplayBuffer::kKeyPrefix) == 0 && replay_buffer_ != nullptr)
        return replay_buffer_->GetTensor(key);

    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>> Learner::GetTensorVector(const std::string& key, int64_t index) const
{
    if (key.find(ReplayBuffer::kKeyPrefix) == 0 && replay_buffer_ != nullptr)
        return replay_buffer_->GetTensorVector(key);

    return std::nullopt;
}

void Learner::SetupOptimizer()
{
    auto params = torch::optim::AdamOptions(config_.alpha).eps(config_.adam_eps);
    ANET_LOG_DEBUG("lr=" << params.lr() << " eps=" << params.eps());
    this->optimizer_ = std::make_unique<torch::optim::Adam>(network_.GetPolicyParameters(), params);
}

void Learner::SetupReplayBuffer(const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec, seed_t seed)
{
    anet::rl::ReplayBufferConfig rep_config{};
    rep_config.capacity = config_.replay_capacity;
    rep_config.gamma = config_.gamma;
    rep_config.n_step = config_.n_step;
    rep_config.type = config_.use_n_step ? ReplayBuilderType::NSTEP : ReplayBuilderType::PLAIN;

    if (config_.use_per) {
        rep_config.sampler_type = ReplaySamplerType::PRIOTIZED;
        rep_config.per_alpha = config_.per_alpha;
        rep_config.per_initial_priority = config_.per_initial_priority;
        vars_.per_beta = config_.per_beta_start;
    } else {
        rep_config.sampler_type = ReplaySamplerType::UNIFORM;
    }

    anet::rl::ReplayBufferFactory rep_factory(rep_config);
    this->replay_buffer_ = rep_factory.Create(env_spec, torch::kCPU, batch_env_spec.batch_size, seed);
}

void Learner::UpdateTargetNetwork(step_t step)
{
    ProfileRange r("Learner::UpdateTargetNetwork");
    network_.UpdateTarget(step);
}

void Learner::UpdatePerBeta(step_t step)
{
    ProfileRange r("Learner::UpdatePerBeta");

    if (!config_.use_per) return;
    if (config_.per_beta_step <= 0) return;
    if (step < config_.per_beta_step) {
        float progress = static_cast<float>(step) / static_cast<float>(config_.per_beta_step);
        vars_.per_beta = config_.per_beta_start + progress * (config_.per_beta_end - config_.per_beta_start);
    } else {
        vars_.per_beta = config_.per_beta_end;
    }
}

bool Learner::CanUpdate(step_t update_step, step_t exp_step) const
{
    // warmup
    if (config_.update_warmup_steps > 0 && exp_step < config_.update_warmup_steps)
        return false;

    // ReplayBufferのサイズがminibatchサイズに満たない場合はスキップ（N-STEPであり得る）
    if (replay_buffer_->Size() < config_.replay_batch_size)
        return false;

    return true;
}

anet::rl::BatchUpdateResultList
Learner::UpdateFromBatch(const anet::rl::StepCounts& counts, const anet::rl::BatchExperience& experiences, const anet::rl::Runner& trainer)
{
    // ReplayBuffer へ push
    replay_buffer_->Push(experiences);

    // 戻り値のUpdateResultを準備
    BatchUpdateResultList result_list;

    // Update不可なら空の結果を返す
    if (!CanUpdate(counts.update_step, counts.exp_step)) {
        return result_list;  // 空配列
    }

    // Credit加算
    update_credit_ += earned_credit_;

    // update_credit が十分な間、学習ループを回す
    while (update_credit_ >= 1.0f) {
        if (!CanUpdate(counts.update_step, counts.exp_step))
            break;
            
        const int B = config_.replay_batch_size;
        const int S = state_dim_;

        // Sample
        float current_beta = config_.use_per ? vars_.per_beta : 0.0f;
        auto raw_samples = replay_buffer_->Sample(config_.replay_batch_size, device_, current_beta);

        // Check shapes & dtypes
        ANET_ASSERT_DEVICE(raw_samples.obs, device_);
        ANET_ASSERT_DEVICE(raw_samples.actions, device_);
        ANET_ASSERT_DEVICE(raw_samples.target_values, device_);
        ANET_ASSERT_DEVICE(raw_samples.next_states.obs, device_);
        ANET_ASSERT_DEVICE(raw_samples.next_states.terminals, device_);
        ANET_ASSERT_DEVICE(raw_samples.n_steps, device_);
        ANET_ASSERT_SHAPE(raw_samples.obs, { B, S });
        ANET_ASSERT_SHAPE(raw_samples.actions, { B });    // 離散アクション
        ANET_ASSERT_SHAPE(raw_samples.target_values, { B });
        ANET_ASSERT_SHAPE(raw_samples.next_states.obs, { B, S });
        ANET_ASSERT_SHAPE(raw_samples.next_states.terminals, { B });
        ANET_ASSERT_SHAPE(raw_samples.n_steps, { B });
        ANET_ASSERT_DTYPE(raw_samples.obs, torch::kFloat32);
        ANET_ASSERT_DTYPE(raw_samples.actions, torch::kInt64);    // 離散アクション
        ANET_ASSERT_DTYPE(raw_samples.target_values, torch::kFloat32);
        ANET_ASSERT_DTYPE(raw_samples.next_states.terminals, torch::kBool);
        ANET_ASSERT_DTYPE(raw_samples.n_steps, torch::kInt64);

        // 固有処理呼び出し
        auto samples = raw_samples.FlattenStates();
        auto result = UpdateFromSamples(samples);
        result_list.push_back(result);

        // 更新後処理
        UpdateTargetNetwork(vars_.learn_step);
        UpdatePerBeta(counts.exp_step);

        // カウント系更新
        vars_.learn_step++;
        update_credit_ -= 1.0f;
    }

    return result_list;
}

// ======================================================
// TDLearner
// ======================================================


TDLearner::TDLearner(const LearnerConfig& config, Network& network, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
    const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, seed_t replay_seed)
    : Learner(config, network, vars, obs_norm, batch_env_spec, env_spec, device, replay_seed)
{
    SetupReplayBuffer(batch_env_spec, env_spec, replay_seed);
    SetupOptimizer();
}

std::shared_ptr<const anet::rl::BatchUpdateResult>
TDLearner::UpdateFromSamples(const anet::rl::ExperienceSamples& samples)
{
    ProfileRange r("TDLearner::UpdateFromBatch");

    const int B = config_.replay_batch_size;
    const int S = state_dim_;
    const int A = n_actions_;
    const auto& target_values = samples.target_values;
    const auto& terminals = samples.next_states.terminals;

    // Observation正規化
    torch::Tensor obs = samples.obs;
    torch::Tensor next_obs = samples.next_states.obs;
    if (obs_norm_) {
        // 統計更新は Agent 側の収集フェーズで行うためここでは適用のみ(false)
        obs = obs_norm_->Normalize(samples.obs);
        next_obs = obs_norm_->Normalize(samples.next_states.obs);
    }

    // ------------------------------------------------------------
    // Q(s, a)
    // ------------------------------------------------------------
    auto q_all = network_.Forward(obs, /*use_target=*/false);      // (B,A)
    ANET_ASSERT_SHAPE(q_all, { B, A });

    torch::Tensor idx_actions = samples.actions.view({ B, 1 });   // (B,1)
    ANET_ASSERT_SHAPE(idx_actions, { B, 1 });
    ANET_ASSERT_DTYPE(idx_actions, torch::kInt64);

    auto q_sa = q_all.gather(1, idx_actions).squeeze(1);          // (B)
    ANET_ASSERT_SHAPE(q_sa, { B });
    ANET_ASSERT_DTYPE(q_sa, torch::kFloat32);

    torch::Tensor max_q = std::get<0>(q_all.max(1)).detach();     // (B)

    // ------------------------------------------------------------
    // max_a' Q(s', a')
    // ------------------------------------------------------------
    torch::Tensor max_next_q;

    if (config_.use_double_dqn) {
        torch::NoGradGuard no_grad;

        // Double DQN: policy_netで行動選択、target_netで価値計算
        auto next_q_policy = network_.Forward(next_obs, /*use_target=*/false);
        ANET_ASSERT_SHAPE(next_q_policy, { B, A });
        auto next_actions = std::get<1>(next_q_policy.max(1));
        ANET_ASSERT_SHAPE(next_actions, { B });

        // target_net で Q_target(s', argmax_a Q_online)
        auto next_q_target = network_.Forward(next_obs, /*use_target=*/true);
        ANET_ASSERT_SHAPE(next_q_target, { B, A });
        torch::Tensor next_actions_b = next_actions.view({ B, 1 });             // (B,1)
        max_next_q = next_q_target.gather(1, next_actions_b).squeeze(1);
    } else {
        torch::NoGradGuard;
        auto next_q_target = network_.Forward(next_obs, /*use_target=*/true);
        ANET_ASSERT_SHAPE(next_q_target, { B, A });
        max_next_q = std::get<0>(next_q_target.max(1));
    }
    ANET_ASSERT_SHAPE(max_next_q, { B });
    ANET_ASSERT_DTYPE(max_next_q, torch::kFloat32);

    // ------------------------------------------------------------
    // TD target & TD Error
    // ------------------------------------------------------------
    auto not_terminal = 1.0f - terminals.to(torch::kFloat32); // (B,)
    auto td_target = target_values + not_terminal * config_.gamma * max_next_q.detach(); // (B,)
    ANET_ASSERT_SHAPE(td_target, { B });
    ANET_ASSERT_DTYPE(td_target, torch::kFloat32);
    auto td_error = q_sa - td_target; // (B,)

    // ------------------------------------------------------------
    // PER Priority Update
    // ------------------------------------------------------------

    // PER Metrics用 Tensor
    torch::Tensor metric_per_clipped_count;
    torch::Tensor metric_per_priorities;
    torch::Tensor metric_per_is_weights;

    if (config_.use_per) {
        torch::NoGradGuard no_grad;

        // 優先度 = |td_error| + eps
        auto abs_td_error = td_error.abs().detach();
        auto new_priorities = abs_td_error + config_.per_eps;

        // Priority Clipping
        if (config_.use_per_prio_clip) {
            // [Metric Source] Clipped Count (Keep as Tensor)
            metric_per_clipped_count = (new_priorities > config_.per_prio_clip_value).sum(); // (Scalar Tensor)

            // クリップ実行
            new_priorities = torch::clamp(new_priorities, 0.0f, config_.per_prio_clip_value);
        } else {
            // クリップしない場合は0
            metric_per_clipped_count = torch::zeros({}, abs_td_error.options());
        }

        // [Metric Source] Priorities
        metric_per_priorities = new_priorities; // (B,)

        // Tensor -> vector (CPU)
        // ※ここでCPU転送(同期)が発生するのはSumTree(CPU)更新のため避けられない
        auto indices_cpu = samples.indices.cpu();
        auto indices_ptr = indices_cpu.data_ptr<int64_t>();
        std::vector<int64_t> indices_vec(indices_ptr, indices_ptr + B);

        auto prios_cpu = new_priorities.cpu();
        auto prios_ptr = prios_cpu.data_ptr<float>();
        std::vector<float> priorities_vec(prios_ptr, prios_ptr + B);

        // Priority更新
        replay_buffer_->UpdatePriorities(indices_vec, priorities_vec);

        // [Metric Source] IS Weights
        if (samples.is_weights.defined()) {
            metric_per_is_weights = samples.is_weights;
        }
    }

    // ------------------------------------------------------------
    // Loss Calculation
    // ------------------------------------------------------------

    torch::Tensor td_error_for_loss = td_error;
    if (config_.use_td_clip && config_.td_clip_value > 0.0f)
        td_error_for_loss = torch::clamp(td_error_for_loss, -config_.td_clip_value, config_.td_clip_value);

    torch::Tensor loss;

    if (config_.use_per) {
        // PER: IS Weights を loss に適用
        // element-wise loss (reduction=none)
        auto element_loss = torch::nn::functional::smooth_l1_loss(
            td_error_for_loss,
            torch::zeros_like(td_error_for_loss),
            torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kNone));

        // 重みを適用して平均
        auto weights = samples.is_weights.to(element_loss.device());
        loss = (element_loss * weights).mean();
    } else {
        // 通常: mean reduction
        loss = torch::nn::functional::smooth_l1_loss(
            td_error_for_loss,
            torch::zeros_like(td_error_for_loss),
            torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kMean));
    }

    // ------------------------------------------------------------
    //  backward & grad
    // ------------------------------------------------------------

    // backward
    optimizer_->zero_grad();
    loss.backward();

    // grad_clip
    torch::Tensor grad_norm_tensor;
    std::optional<float> grad_norm;
    bool grad_clipped = false;
    if (config_.use_grad_clip) {
        // clip_grad_norm_ の戻り値は clip 前の全体ノルム
        double grad_norm_val = torch::nn::utils::clip_grad_norm_(
                network_.GetPolicyParameters(), config_.grad_clip_tau);   // use_grad_clip=true では CPU同期は現状避けられない
        grad_norm = static_cast<float>(grad_norm_val);
        grad_clipped = (grad_norm_val > config_.grad_clip_tau);
    } else {
        torch::Tensor total_sq = torch::zeros({ 1 }, loss.options());
        auto params = network_.GetPolicyParameters();
        for (auto& p : params) {
            if (!p.grad().defined()) continue;
            total_sq += p.grad().detach().pow(2).sum();
        }
        // sqrt までは GPU 上でやる
        grad_norm_tensor = total_sq.sqrt();
    }
    float grad_clip_ratio = grad_clipped ? 1.0f : 0.0f;

    // ------------------------------------------------------------
    // optimize
    // ------------------------------------------------------------
    optimizer_->step();

    // ------------------------------------------------------------
    // UpdateResult
    // ------------------------------------------------------------
    auto result = std::make_shared<BatchUpdateResult>();
    result->loss = loss;
    result->td_error = td_error;
    result->grad_norm = grad_norm;
    result->grad_norm_tensor = grad_norm_tensor;
    result->grad_clip_ratio = grad_clip_ratio;
    result->max_q = max_q;
    if (config_.use_per) {
        result->per_minibatch_size = B;
        result->per_clipped_count = metric_per_clipped_count;
        result->per_priorities = metric_per_priorities;
        result->per_is_weights = metric_per_is_weights;
    }
    return result;
}


// ======================================================
// QRLearner
// ======================================================

QRLearner::QRLearner(const LearnerConfig& config, Network& network, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
    const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, seed_t replay_seed)
    : Learner(config, network, vars, obs_norm, batch_env_spec, env_spec, std::move(device), replay_seed)
{
    SetupReplayBuffer(batch_env_spec, env_spec, replay_seed);
    SetupOptimizer();
}

torch::Tensor QRLearner::ComputeQuantileHuberLoss(
    const torch::Tensor& current_dist, const torch::Tensor& target_dist) const
{
    ProfileRange r("QRLearner::ComputeQuantileHuberLoss");

    const int N = config_.num_quantiles;
    const float kappa = config_.quantile_huber_kappa;
    const auto& device = current_dist.device();
    const auto B = current_dist.size(0);

    // 入力チェック
    ANET_ASSERT_SHAPE(current_dist, { B, N });
    ANET_ASSERT_SHAPE(target_dist, { B, N });

    // current: (B, N) -> (B, N, 1)
    // target : (B, N) -> (B, 1, N)
    auto cur = current_dist.unsqueeze(2);
    auto tgt = target_dist.unsqueeze(1);
    ANET_ASSERT_SHAPE(cur, { B, N, 1 });
    ANET_ASSERT_SHAPE(tgt, { B, 1, N });

    // pair-wise差分: (B, N, N)
    auto diff = tgt - cur;
    ANET_ASSERT_SHAPE(diff, { B, N, N });

    // 分位数 tau_i = (i + 0.5) / N
    auto tau = torch::arange(0.5f / N, 1.0f, 1.0f / N, device).view({ 1, N, 1 });
    ANET_ASSERT_SHAPE(tau, { 1, N, 1 });

    // Huber Loss
    auto abs_diff = diff.abs();
    auto huber = torch::where(abs_diff < kappa, 0.5f * diff.pow(2), kappa * (abs_diff - 0.5f * kappa));
    ANET_ASSERT_SHAPE(huber, { B, N, N });

    // Quantile Regression Loss
    // rho_tau(u) = |tau - I(u<0)| * L_k(u)
    auto indicator = (diff.detach() < 0).to(torch::kFloat);
    auto quantile_weight = torch::abs(tau - indicator);	 // Broadcasting Check: (1, N, 1) - (B, N, N) -> (B, N, N)
    ANET_ASSERT_SHAPE(quantile_weight, { B, N, N });

    auto loss_per_pair = quantile_weight * huber; // (B, N, N)

    // ターゲット分位数(dim=2)で総和、現在の分位数(dim=1)で平均 "Loss per Batch element"
    auto element_wise_loss = loss_per_pair.sum(2).mean(1); // (B)
    ANET_ASSERT_SHAPE(element_wise_loss, { B });

    return element_wise_loss;
}

std::shared_ptr<const anet::rl::BatchUpdateResult>
QRLearner::UpdateFromSamples(const anet::rl::ExperienceSamples& samples)
{
    ProfileRange r("QRLearner::UpdateFromSamples");

    const int B = config_.replay_batch_size;
    const int A = n_actions_;
    const int N = config_.num_quantiles;

    // 入力チェック
    ANET_ASSERT_SHAPE(samples.actions, { B });
    ANET_ASSERT_SHAPE(samples.target_values, { B });
    ANET_ASSERT_SHAPE(samples.next_states.terminals, { B });

    // Observation正規化
    torch::Tensor obs = samples.obs;
    torch::Tensor next_obs = samples.next_states.obs;
    if (obs_norm_) {
        // 統計更新は Agent 側の収集フェーズで行うためここでは適用のみ(false)
        obs = obs_norm_->Normalize(samples.obs);
        next_obs = obs_norm_->Normalize(samples.next_states.obs);
    }
    ANET_ASSERT_NAN(obs);
    ANET_ASSERT_NAN(next_obs);

    // ------------------------------------------------------------
    // 分布計算
    // ------------------------------------------------------------

    // 現在の分布計算: Z(s, a)、ForwardQuantiles は (B, A, N) を返す
    auto current_dist_all = network_.ForwardQuantiles(obs, /*use_target=*/false);
    ANET_ASSERT_SHAPE(current_dist_all, { B, A, N });
    ANET_ASSERT_NAN(current_dist_all);

    // 選択された行動の分布を取得: (B, A, N) -> (B, N)
    torch::Tensor idx_actions = samples.actions.view({ B, 1, 1 }).expand({ B, 1, N });
    ANET_ASSERT_SHAPE(idx_actions, { B, 1, N });
    ANET_ASSERT_NAN(idx_actions);

    auto current_dist = current_dist_all.gather(1, idx_actions).squeeze(1); // (B, N)
    ANET_ASSERT_SHAPE(current_dist, { B, N });
    ANET_ASSERT_NAN(current_dist);

    // メトリクス用: 平均値をmax_qとして報告
    auto max_q = current_dist.mean(1).detach(); // (B)
    ANET_ASSERT_SHAPE(max_q, { B });

    // メトリクス用: Q Std (分布の標準偏差)、 GPUTensor (Scalar) のまま保持
    auto std_q_tensor = current_dist.std(1).mean().detach();
    ANET_ASSERT_SHAPE(std_q_tensor, {});


    // ------------------------------------------------------------
    // ターゲット分布計算: r + gamma * Z(s', a*)
    // ------------------------------------------------------------

    torch::Tensor target_dist;
    {
        torch::NoGradGuard no_grad;

        // 次状態のGreedy行動 a* = argmax E[Z(s', a')]
        torch::Tensor next_actions;
        if (config_.use_double_dqn) {
            auto next_q_policy = network_.Forward(next_obs, /*use_target=*/false); // (B, A)
            ANET_ASSERT_SHAPE(next_q_policy, { B, A });
            next_actions = std::get<1>(next_q_policy.max(1)); // (B)
        } else {
            auto next_q_target = network_.Forward(next_obs, /*use_target=*/true); // (B, A)
            ANET_ASSERT_SHAPE(next_q_target, { B, A });
            next_actions = std::get<1>(next_q_target.max(1)); // (B)
        }
        ANET_ASSERT_SHAPE(next_actions, { B });

        // 次状態のターゲット分布: Z_target(s', :)
        auto next_dist_all = network_.ForwardQuantiles(next_obs, /*use_target=*/true); // (B, A, N)
        ANET_ASSERT_SHAPE(next_dist_all, { B, A, N });

        // a* に対応する分布を選択: (B, A, N) -> (B, N)
        torch::Tensor idx_next_actions = next_actions.view({ B, 1, 1 }).expand({ B, 1, N });
        ANET_ASSERT_SHAPE(idx_next_actions, { B, 1, N });

        auto next_dist = next_dist_all.gather(1, idx_next_actions).squeeze(1); // (B, N)
        ANET_ASSERT_SHAPE(next_dist, { B, N });

        // ベルマン作用素適用: T = r + gamma * Z(s', a*)
        auto reward = samples.target_values.view({ B, 1 }); // (B, 1)
        auto not_terminal = (1.0f - samples.next_states.terminals.to(torch::kFloat32)).view({ B, 1 }); // (B, 1)
        ANET_ASSERT_SHAPE(reward, { B, 1 });
        ANET_ASSERT_SHAPE(not_terminal, { B, 1 });

        // (B, 1) + (B, 1) * (B, N) -> (B, N)
        target_dist = reward + config_.gamma * not_terminal * next_dist;
        ANET_ASSERT_SHAPE(target_dist, { B, N });
    }
    ANET_ASSERT_NAN(target_dist);

    // target_dist: (B, N) -> mean -> (B)
    auto target_mean = target_dist.mean(1).detach();

    // current_dist.mean(1) は max_q (detach済み) と同じ。 TDLearner: q_sa - td_target に合わせる
    auto td_error_tensor = max_q - target_mean;


    // ------------------------------------------------------------
    // Loss Calculation
    // ------------------------------------------------------------

    // 要素ごとのLoss (B) を取得  ※ここで重い計算を一回だけ行う
    auto element_loss = ComputeQuantileHuberLoss(current_dist, target_dist);
    ANET_ASSERT_SHAPE(element_loss, { B });
    ANET_ASSERT_NAN(element_loss);

    // 最適化用Loss(Scalar) ※ PERの重み (IS Weights) を適用
    torch::Tensor weights = config_.use_per ? samples.is_weights : torch::ones({ B }, device_);
    ANET_ASSERT_NAN(weights);
    auto loss = (element_loss * weights).mean();
    ANET_ASSERT_SHAPE(loss, {});
    ANET_ASSERT_NAN(loss);


    // ------------------------------------------------------------
    // Optimize
    // ------------------------------------------------------------
    optimizer_->zero_grad();
    loss.backward();

    // 勾配NaNチェック
#if ANET_ENABLE_TENSOR_NAN_CHECK
    for (auto& param : network_.GetPolicyParameters()) {
        if (param.grad().defined()) { // 勾配が存在する場合
            ANET_ASSERT_NAN(param.grad());
        }
    }
#endif

    // ------------------------------------------------------------
    // 勾配クリッピング
    // ------------------------------------------------------------
    torch::Tensor grad_norm_tensor;
    std::optional<float> grad_norm;
    bool grad_clipped = false;
    if (config_.use_grad_clip) {
        double grad_norm_val = torch::nn::utils::clip_grad_norm_(
            network_.GetPolicyParameters(), config_.grad_clip_tau);
        grad_norm = static_cast<float>(grad_norm_val);
        grad_clipped = (grad_norm_val > config_.grad_clip_tau);
    } else {
        torch::Tensor total_sq = torch::zeros({ 1 }, loss.options());
        for (auto& p : network_.GetPolicyParameters()) {
            if (!p.grad().defined()) continue;
            total_sq += p.grad().detach().pow(2).sum();
        }
        grad_norm_tensor = total_sq.sqrt();
    }
    float grad_clip_ratio = grad_clipped ? 1.0f : 0.0f;

    // パラメータ更新
    optimizer_->step();


    // ------------------------------------------------------------
    // PER優先度更新
    // ------------------------------------------------------------

    torch::Tensor metric_per_clipped_count;
    torch::Tensor metric_per_priorities;
    torch::Tensor metric_per_is_weights;

    if (config_.use_per) {
        torch::NoGradGuard no_grad;

        // 優先度用に要素ごとのLossを再計算（ブロードキャストを利用して全ペア差分を計算）
        auto tgt = target_dist.unsqueeze(1); // (B, 1, N)
        auto cur = current_dist.unsqueeze(2); // (B, N, 1)
        ANET_ASSERT_SHAPE(tgt, { B, 1, N });
        ANET_ASSERT_SHAPE(cur, { B, N, 1 });

        auto diff = tgt - cur; // (B, N, N)
        ANET_ASSERT_SHAPE(diff, { B, N, N });

        // HuberLoss 部分
        auto tau = torch::arange(0.5f / N, 1.0f, 1.0f / N, device_).view({ 1, N, 1 });
        auto huber_loss = torch::where(
            diff.abs() < config_.quantile_huber_kappa,
            0.5f * diff.pow(2),
            config_.quantile_huber_kappa * (diff.abs() - 0.5f * config_.quantile_huber_kappa)
        );

        // Quantile Loss 部分: rho_tau(u) = |tau - I(u<0)| * L_k(u)
        auto element_wise_loss = (torch::abs(tau - (diff.detach() < 0).to(torch::kFloat)) * huber_loss).sum(2).mean(1); // (B)
        ANET_ASSERT_SHAPE(element_wise_loss, { B });

        // Priority (element_loss を N で割ってスケーリング)
        auto new_priorities = (element_loss / static_cast<float>(N)) + config_.per_eps;
        ANET_ASSERT_SHAPE(new_priorities, { B });
        ANET_ASSERT_NAN(new_priorities);

        // PER clip
        if (config_.use_per_prio_clip) {
            metric_per_clipped_count = (new_priorities > config_.per_prio_clip_value).sum();
            new_priorities = torch::clamp(new_priorities, 0.0f, config_.per_prio_clip_value);
        } else {
            metric_per_clipped_count = torch::zeros({}, new_priorities.options());
        }
        metric_per_priorities = new_priorities;

        // Priorityをstd::vectorに詰める
        auto indices_cpu = samples.indices.cpu();
        auto indices_ptr = indices_cpu.data_ptr<int64_t>();
        std::vector<int64_t> indices_vec(indices_ptr, indices_ptr + B);

        auto prios_cpu = new_priorities.cpu();
        auto prios_ptr = prios_cpu.data_ptr<float>();
        std::vector<float> priorities_vec(prios_ptr, prios_ptr + B);
            /// @todo PERの優先度更新にはCPU値が必要なので、ここでは同期が発生する

        // Priority更新
        replay_buffer_->UpdatePriorities(indices_vec, priorities_vec);
        if (samples.is_weights.defined()) {
            metric_per_is_weights = samples.is_weights;
        }
    }


    // ------------------------------------------------------------
    // 結果生成
    // ------------------------------------------------------------

    auto result = std::make_shared<BatchUpdateResult>();
    result->loss = loss;
    result->td_error = td_error_tensor;
    result->grad_norm = grad_norm;
    result->grad_norm_tensor = grad_norm_tensor;
    result->grad_clip_ratio = grad_clip_ratio;
    result->max_q = max_q;
    if (config_.use_per) {
        result->per_minibatch_size = B;
        result->per_clipped_count = metric_per_clipped_count;
        result->per_priorities = metric_per_priorities;
        result->per_is_weights = metric_per_is_weights;
    }
    result->q_std = std_q_tensor;

    return result;
}

