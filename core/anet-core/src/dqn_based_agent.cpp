
#include "dqn_based_agent.hpp"
#include <tuple>
#include <cmath>
#include "anet/log.hpp"
#include "anet/profile.hpp"
#include "anet/tensor_check.hpp"
#include "anet/tensor_util.hpp"
#include "anet/str_util.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/image.hpp"

using namespace anet::rl::dqn;
namespace LOG = anet::log;


// ======================================================
// NetworkModel
// ======================================================

NetworkModel::NetworkModel(
    const NetworkModelConfig& config, const torch::Device& device,
    const anet::nn::NetworkConfig& network_config, const std::vector<int64_t>& input_shape, int64_t n_actions, std::shared_ptr<anet::nn::NetworkHeadFactory> head_factory,
    int64_t num_quantiles)
    : config_(config)
    , n_actions_(n_actions)
    , num_quantiles_(num_quantiles)
{
    ANET_ASSERT(n_actions_ > 0);

    // メインネットワークを作る
    policy_net_ = anet::nn::NetworkBuilder::BuildNetwork(network_config, input_shape, head_factory);
    policy_net_->to(device);

    // メインネットワークをコピーしてターゲットネットワークを作る
    target_net_ = policy_net_->Clone(device);
    target_net_->eval();
}

anet::TensorDict NetworkModel::Forward(const torch::Tensor& obs, bool use_target) const
{
    const auto& net = use_target ? target_net_ : policy_net_;
    return net->Forward(obs);
}

bool NetworkModel::IsDistributional(bool use_target) const
{
    return (num_quantiles_ > 1);
}

std::vector<torch::Tensor> NetworkModel::GetPolicyParameters() const
{
    auto params = policy_net_->parameters();
    return params;
}

void NetworkModel::UpdateTarget(step_t learn_step)
{
    torch::NoGradGuard grad_guard;

    if (config_.hard_update_interval > 0) {
        if (learn_step % config_.hard_update_interval == 0) {
            HardUpdate();
        }
        return;
    }
    SoftUpdate();
}

void NetworkModel::SoftUpdate()
{
    policy_net_->SoftCopyTo(*target_net_, config_.soft_update_tau);
}

void NetworkModel::HardUpdate()
{
    policy_net_->CopyTo(*target_net_);
}

std::optional<anet::TensorFunction> NetworkModel::GetTensorFunction(const std::string& key, const torch::Device& device)
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

std::optional<anet::TensorDictFunction> NetworkModel::GetTensorDictFunction(const std::string& key, const torch::Device& device)
{
    // Keyの指定に応じて、対象のネットワークを切り替える
    std::shared_ptr<anet::nn::Network> net = nullptr;

    if (key == "policy-net.conv2d") {
        net = policy_net_;
    } else if (key == "target-net.conv2d") {
        net = target_net_;
    }

    // 対象が見つからなければ nullopt
    if (!net) {
        return std::nullopt;
    }

    // デバイス転送と抽出処理をラップした関数を返す
    return [net, device](const torch::Tensor& obs) {
        return net->GetConv2dOutputs(obs.to(device));
        };
}

int64_t NetworkModel::Save(OutputArchive& archive) const
{
	int64_t size = 0;
    size += archive.WriteTorchObject(policy_net_);
    size += archive.WriteTorchObject(target_net_);
    return size;
}

int64_t NetworkModel::Load(InputArchive& archive)
{
    int64_t size = 0;
    size += archive.ReadTorchObject(policy_net_);
    size += archive.ReadTorchObject(target_net_);
    return size;
}


// ======================================================
// ActionPolicy 
// ======================================================

anet::rl::dqn::ActionPolicy::ActionPolicy(const ActionPolicyConfig& config)
    : config_(config)
{
}

torch::Tensor anet::rl::dqn::ActionPolicy::MakeEpsilonGreedyAction(const torch::Tensor& greedy_action, float epsilon, int64_t batch_size, int64_t n_actions, std::shared_ptr<anet::RandomGenerator> rnd) const
{
    ProfileRange  r("ActionPolicy::MakeEpsilonGreedyAction");

    // epsilongがゼロ(十分小さい)場合はGreedy（乱数生成影響を排除）
    if (epsilon <= std::numeric_limits<float>::epsilon()) {
        return greedy_action;
    }

    auto device = greedy_action.device();
    auto gen = rnd->GetTorchGenerator(device);

    // ランダム選択するActionをマスクとして選択（εを使って）
    auto mask = torch::rand({ batch_size }, gen, torch::TensorOptions().device(device)).lt(epsilon);    // mask: (N) bool, GPU上で生成

    // ランダム選択対象のアクションについて、乱数でアクション決定
    auto random_actions = torch::randint(/*low=*/0, /*high=*/n_actions, { batch_size }, gen, // random actions(N) int64
        torch::TensorOptions().dtype(torch::kInt64).device(device));

    // actions: where(mask, random_actions, greedy)
    auto actions = torch::where(mask, random_actions, greedy_action);

    return actions;
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
    aux["raw_actions"] = action_values;

    return action_info;
}

void anet::rl::dqn::ActionPolicy::UpdateEpsilon(step_t step, bool is_uqe)
{
    if (is_uqe) {
        if (config_.uqe_eps_decay_steps <= 0) return;
        if (step >= config_.uqe_eps_decay_steps) {
            current_epsilon_ = config_.uqe_eps_end;
        } else {
            const float t = static_cast<float>(step) / static_cast<float>(config_.uqe_eps_decay_steps);
            current_epsilon_ = config_.uqe_eps_start + t * (config_.uqe_eps_end - config_.uqe_eps_start);
        }
    } else {
        if (config_.eps_decay_steps <= 0) return;
        if (step >= config_.eps_decay_steps) {
            current_epsilon_ = config_.eps_end;
        } else {
            const float t = static_cast<float>(step) / static_cast<float>(config_.eps_decay_steps);
            current_epsilon_ = config_.eps_start + t * (config_.eps_end - config_.eps_start);
        }
    }
}

std::optional<float> anet::rl::dqn::ActionPolicy::GetScalar(const std::string& key, int64_t index) const
{
    if (key == "epsilon") return current_epsilon_;
    if (key == "uqe_tau") return current_uqe_tau_;
    return std::nullopt;
}


// ======================================================
// EpsilonGreedyActionPolicy 
// ======================================================

anet::rl::dqn::EpsilonGreedyActionPolicy::EpsilonGreedyActionPolicy(const ActionPolicyConfig& config)
    : ActionPolicy(config)
{
    current_epsilon_ = config_.eps_start;
}

void anet::rl::dqn::EpsilonGreedyActionPolicy::OnLearn(const StepCounts& counts)
{
    UpdateEpsilon(counts.exp_step);
}

anet::rl::BatchActionInfo EpsilonGreedyActionPolicy::SelectAction(const torch::Tensor& obs, bool greedy_only, std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd) const
{
    ProfileRange r("EpsilonGreedyActionPolicy::SelectAction");

    torch::NoGradGuard grad_guard;
    torch::ScalarType amp_dtype = config_.use_amp_bf16 ? torch::kBFloat16 : torch::kHalf;
    anet::Autocast cast_guard(torch::kCUDA, config_.use_amp, amp_dtype);

    // obsからQ値取得
    auto out = network->Forward(obs);
    auto q_values = out.At("q");
    torch::Tensor q_quantiles;
    if (out.Contains("q_dist")) {
        q_quantiles = out.At("q_dist");
    }

    // Q値GreedyなActionを生成
    auto greedy_action = q_values.argmax(1, /*keepdim=*/false);        // greedy = argmax(q_values, dim=1)

    // Greedy指定ならargmxを返す
    if (greedy_only) return MakeActionInfo(greedy_action, q_values, q_quantiles);

    // EpsilonGreedy
    const int64_t N = q_values.sizes()[0];      // shape 読み取りは TensorOptions 経由で同期を回避
    const int64_t A = q_values.sizes()[1];
    auto actions = MakeEpsilonGreedyAction(greedy_action, current_epsilon_, N, A, rnd);
    auto action_info = MakeActionInfo(actions, q_values, q_quantiles);
    return action_info;
}


// ======================================================
// UQEActionPolicy
// ======================================================

anet::rl::dqn::UQEActionPolicy::UQEActionPolicy(const ActionPolicyConfig& config)
    : ActionPolicy(config)
{
    current_epsilon_ = config_.uqe_eps_start;
    current_uqe_tau_ = config_.uqe_tau_start;
}

void anet::rl::dqn::UQEActionPolicy::UpdateTau(step_t step)
{
    if (config_.uqe_tau_decay_steps <= 0) return;
    if (step >= config_.uqe_tau_decay_steps) {
        current_uqe_tau_ = config_.uqe_tau_end;
        return;
    }
    const float t = static_cast<float>(step) / static_cast<float>(config_.uqe_tau_decay_steps);
    current_uqe_tau_ = config_.uqe_tau_start + t * (config_.uqe_tau_end - config_.uqe_tau_start);
}

void anet::rl::dqn::UQEActionPolicy::OnLearn(const StepCounts& counts)
{
    UpdateEpsilon(counts.exp_step, true);
    UpdateTau(counts.exp_step);
}

/// Q分布の上位tau%(上振れ時)Q値が最大になる行動を選択する。
/// tauが大きい(1.0に近い)程、不確実だけど化ける可能性がある行動が相対的に選択されやすくなる
torch::Tensor UQEActionPolicy::MakeUQEAction(float tau, const torch::Tensor& q_quantiles) const
{
    ProfileRange r("UQEActionPolicy::MakeUQEAction");

    // GPU同期を避けるため、sizes() からメタデータのみ取得
    const int64_t n_quantiles = q_quantiles.size(-1);

    // インデックス決定: floor(tau * (N-1)) clampして範囲外アクセスを防止
    int64_t tau_idx = static_cast<int64_t>(tau * (n_quantiles - 1));
    tau_idx = std::max<int64_t>(0, std::min<int64_t>(tau_idx, n_quantiles - 1));
    ANET_LOG_DEBUG("tau_idx=" << tau_idx);

    // 最後の次元(分位点)を確実に昇順(小さい順)にソートする
    torch::Tensor sorted_quantiles = std::get<0>(q_quantiles.sort(-1, /*descending=*/false));

    torch::Tensor uqe_values;
    if (config_.uqe_use_tail_mean) {
        // 上位分位点すべての平均を使う場合

        // tau_idx から 最後まで (N-1) の範囲を切り出す
        auto tail_values = sorted_quantiles.slice(-1, tau_idx, n_quantiles);  // (B, A, N - tau_idx)
        ANET_LOG_DEBUG("tail_values=" << anet::ToString(tail_values));

        // 切り出した範囲の平均をとる
        uqe_values = tail_values.mean(-1);
    } else {
        // 特定の分位点におけるQ値を取得
        uqe_values = sorted_quantiles.select(-1, tau_idx);  // (B, A, N) -> (B, A)
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

    // 最後の次元(分位点)を確実に昇順(小さい順)にソートする
    torch::Tensor sorted_quantiles = std::get<0>(q_quantiles.sort(-1, /*descending=*/false));

    // インデックスの計算 (N, 1)
    auto tau_idx = (tau_tensor * (n_quantiles - 1)).to(torch::kLong);
    tau_idx = tau_idx.clamp(0, n_quantiles - 1);

    torch::Tensor uqe_values;

    if (config_.uqe_use_tail_mean) {
        // マスクを使って平均を計算する

        // range: (1, 1, n_quantiles) -> [0, 1, 2, ...]
        auto range = torch::arange(n_quantiles, device).view({ 1, 1, -1 });

        // mask: (N, 1, n_quantiles)
        // range >= tau_idx の部分が True (1.0)
        // tau_idx は (N, 1) なので (N, 1, 1) に view して broadcast
        auto mask = range.ge(tau_idx.view({ N, 1, 1 })).to(q_quantiles.dtype());

        // 平均計算: (sum(Q * mask) / sum(mask))
        // small_epsilon を足して 0除算防止
        uqe_values = (sorted_quantiles * mask).sum(-1) / (mask.sum(-1) + 1e-6);
    } else {
        // tau_idx を (N, A, 1) に拡張
        // バッチ(N)ごとに違うインデックスだが、Action(A)に対しては同じインデックスを使う
        auto gather_idx = tau_idx.view({ N, 1, 1 }).expand({ N, A, 1 });

        // gather: dim=-1 (quantiles次元) に沿って収集
        // output: (N, A, 1) -> squeeze -> (N, A)
        uqe_values = sorted_quantiles.gather(-1, gather_idx).squeeze(-1);
    }

    return uqe_values.argmax(1);
}

anet::rl::BatchActionInfo UQEActionPolicy::MakeUQEActionInfo(float tau, const torch::Tensor& tau_tensor, const torch::Tensor& obs,
    bool greedy_only, std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd) const
{
    ProfileRange r("UQEActionPolicy::MakeUQEActionInfo");

    torch::NoGradGuard grad_guard;
    torch::ScalarType amp_dtype = config_.use_amp_bf16 ? torch::kBFloat16 : torch::kHalf;
    anet::Autocast cast_guard(torch::kCUDA, config_.use_amp, amp_dtype);

    // Q値を取得
    auto out = network->Forward(obs);
    auto q_values = out.At("q");
    auto q_quantiles = out.At("q_dist");

    // パラメータの決定 (Train vs Eval)
    float effective_epsilon = current_epsilon_;
    float effective_tau = tau;
    bool use_vectorized_tau = tau_tensor.defined();

    //greedy_onlyが指定された場合は、ランダムノイズ(ε)を強制的にゼロにする
    // (Tauによる楽観的選択の基準は維持しつつ、デタラメな行動を防ぐ)
    if (greedy_only) {
        effective_epsilon = 0.0f;
    }

    // UQE (楽観的Q値) の計算
    torch::Tensor uqe_action_values;
    if (use_vectorized_tau) {
        // 学習時: バッチごとに異なるTauが渡されている場合（hompsonSampling向け）
        uqe_action_values = MakeVectorizedUQEAction(tau_tensor, q_quantiles);
    } else {
        // Eval時 または 学習時(固定Tau): スカラーTauを使う
        uqe_action_values = MakeUQEAction(effective_tau, q_quantiles);
    }

    // EpsilonGreedy
    const int64_t N = q_values.sizes()[0];      // shape 読み取りは TensorOptions 経由で同期を回避
    const int64_t A = q_values.sizes()[1];
    auto actions = MakeEpsilonGreedyAction(uqe_action_values, effective_epsilon, N, A, rnd);

    // 情報詰め替え
    auto action_info = MakeActionInfo(actions, q_values, q_quantiles);
    return action_info;
}

anet::rl::BatchActionInfo UQEActionPolicy::SelectAction(const torch::Tensor& obs, bool greedy_only, std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd) const
{
    return MakeUQEActionInfo(current_uqe_tau_, torch::Tensor(), obs, greedy_only, network, rnd);
}


// ======================================================
// ThompsonSamplingActionPolicy
// ======================================================

anet::rl::dqn::ThompsonSamplingActionPolicy::ThompsonSamplingActionPolicy(const ActionPolicyConfig& config)
    : UQEActionPolicy(config)
{
    ;
}

void anet::rl::dqn::ThompsonSamplingActionPolicy::OnLearn(const StepCounts& counts)
{
    UpdateEpsilon(counts.exp_step, true);
    //UpdateTau(counts.exp_step);
}

anet::rl::BatchActionInfo ThompsonSamplingActionPolicy::SelectAction(const torch::Tensor& obs, bool greedy_only, std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd) const
{
    // ランダムな Tau をバッチサイズ分生成 (N, 1)
    const int64_t N = obs.size(0);
    auto device = obs.device();
    auto gen = rnd->GetTorchGenerator(device);
    auto tau_tensor = torch::rand({ N, 1 }, gen, torch::TensorOptions().device(device));

    // tau_tensor(ランダム)でUQE適用
    return MakeUQEActionInfo(0.0f, tau_tensor, obs, greedy_only, network, rnd);
}


// ======================================================
// Actor
// ======================================================

Actor::Actor(std::shared_ptr<ActionPolicy> policy,
    std::shared_ptr<anet::rl::ObservationNormalizer> obs_norm,
    std::shared_ptr<ActionContext> context,
    std::shared_ptr<std::shared_mutex> mutex,
    std::shared_ptr<anet::nn::Network> network,
    std::shared_ptr<anet::nn::Network> src_network)
    : policy_(std::move(policy)), obs_norm_(std::move(obs_norm)), context_(std::move(context)), mutex_(std::move(mutex))
    , network_(std::move(network)), src_network_(std::move(src_network))
{
    ;
}

anet::rl::BatchActionInfo Actor::MakeAction(const StepCounts& step, const anet::rl::BatchState& state) const
{
    ProfileRange r1("Actor::MakeAction");
    torch::NoGradGuard ng;

    // FrameStacking
    auto obs = state.obs;
    if (context_ != nullptr) {
        obs = context_->PushObservation(state);
    }
    ANET_LOG_DEBUG("obs=" << anet::ToDefString(obs));

    // Observation正規化
    torch::Tensor norm_obs = obs;
    if (obs_norm_ != nullptr) {
        norm_obs = obs_norm_->Normalize(obs);
    }
    //norm_obs = norm_obs.to(torch::kCPU);
    ANET_LOG_DEBUG("norm_obs=" << anet::ToDefString(norm_obs));

    // 行動選択
    auto rnd = context_->GetRandomGenerator();
    anet::rl::BatchActionInfo act_info;
    if (network_ != src_network_) {
        // Clone済み: 自分専用のネットワークなので排他不要
        act_info = policy_->SelectAction(norm_obs, false, network_, rnd);
    } else {
        // Clone無し（直列モード）: Learnerの更新と競合しないようSharedLock
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        act_info = policy_->SelectAction(norm_obs, false, network_, rnd);
    }

    // AuxData の詰め込み
    act_info.GetAuxData()["raw_obs"] = obs;
    if (obs_norm_ != nullptr) {
        act_info.GetAuxData()["norm_obs"] = norm_obs;
    }

    return act_info;
}

void Actor::Sync()
{
    if (network_ != src_network_) {
        // Clone中は排他が必要
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        src_network_->CopyTo(*network_);
    }
}


// ======================================================
// Learner
// ======================================================

Learner::Learner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
    const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed,
    std::shared_ptr<ActionPolicy> target_policy, std::optional<StuckerConfig> stucker_config, std::optional<anet::seed_t> target_seed)
    : RandomHolder(target_seed), config_(config), stucker_config_(stucker_config), model_(model), vars_(vars), obs_norm_(std::move(obs_norm))
    , batch_size_(batch_env_spec.batch_size)
    , n_actions_(env_spec.action_spec.GetNumActions()), state_dim_(env_spec.state_spec.CalcFlattenDim())
    , device_(std::move(device))
    , target_policy_(std::move(target_policy))
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
    this->optimizer_ = std::make_unique<torch::optim::Adam>(model_.GetPolicyParameters(), params);
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
    if (stucker_config_.has_value()) {
        rep_config.use_stacker = stucker_config_->use_stacker;
        rep_config.stack_count = stucker_config_->stack_count;
	}
    
    anet::rl::ReplayBufferFactory rep_factory(rep_config);
    this->replay_buffer_ = rep_factory.Create(env_spec, torch::kCPU, batch_env_spec.batch_size, seed);
}

void Learner::UpdateTargetNetwork(step_t step)
{
    ProfileRange r("Learner::UpdateTargetNetwork");
    model_.UpdateTarget(step);
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

bool Learner::CanUpdate(step_t exp_step) const
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
Learner::UpdateFromBatch(const anet::rl::StepCounts& counts, const anet::rl::BatchExperience& experiences)
{
    // ReplayBuffer へ push
    replay_buffer_->Push(experiences);

    // 戻り値のUpdateResultを準備
    BatchUpdateResultList result_list;

    // Update不可なら空の結果を返す
    if (!CanUpdate(counts.exp_step)) {
        return result_list;  // 空配列
    }

    // Credit加算
    update_credit_ += earned_credit_;

    // update_credit が十分な間、学習ループを回す
    while (update_credit_ >= 1.0f) {
        if (!CanUpdate(counts.exp_step))
            break;
            
        const int B = config_.replay_batch_size;
        const int S = state_dim_;

        // Sample
        float current_beta = config_.use_per ? vars_.per_beta : 0.0f;
        auto raw_samples = replay_buffer_->Sample(config_.replay_batch_size, device_, current_beta);
        //ANET_LOG_DEBUG("raw_samples.obs=" << anet::ToDefString(raw_samples.obs));
        //ANET_LOG_DEBUG("raw_samples.next_states.obs=" << anet::ToDefString(raw_samples.next_states.obs));

        // Check shapes & dtypes
        ANET_ASSERT_DEVICE(raw_samples.obs, device_);
        ANET_ASSERT_DEVICE(raw_samples.actions, device_);
        ANET_ASSERT_DEVICE(raw_samples.target_values, device_);
        ANET_ASSERT_DEVICE(raw_samples.next_states.obs, device_);
        ANET_ASSERT_DEVICE(raw_samples.next_states.terminals, device_);
        ANET_ASSERT_DEVICE(raw_samples.n_steps, device_);
        //ANET_ASSERT_SHAPE(raw_samples.obs, { B, S });
        ANET_ASSERT_SHAPE(raw_samples.actions, { B });    // 離散アクション
        ANET_ASSERT_SHAPE(raw_samples.target_values, { B });
        //ANET_ASSERT_SHAPE(raw_samples.next_states.obs, { B, S });
        ANET_ASSERT_SHAPE(raw_samples.next_states.terminals, { B });
        ANET_ASSERT_SHAPE(raw_samples.n_steps, { B });
        ANET_ASSERT_DTYPE(raw_samples.obs, torch::kFloat32);
        ANET_ASSERT_DTYPE(raw_samples.actions, torch::kInt64);    // 離散アクション
        ANET_ASSERT_DTYPE(raw_samples.target_values, torch::kFloat32);
        ANET_ASSERT_DTYPE(raw_samples.next_states.terminals, torch::kBool);
        ANET_ASSERT_DTYPE(raw_samples.n_steps, torch::kInt64);

        // 固有処理呼び出し
        //auto samples = raw_samples.FlattenStates();
        auto result = UpdateFromSamples(raw_samples);
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

int64_t Learner::Save(OutputArchive& archive) const
{
    int64_t size = 0;
    size += archive.WriteTorchObject(*optimizer_);
    return size;
}

int64_t Learner::Load(InputArchive& archive)
{
    int64_t size = 0;
    size += archive.ReadTorchObject(*optimizer_);
    return size;
}


// ======================================================
// TDLearner
// ======================================================


TDLearner::TDLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
    const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, seed_t replay_seed,
    std::shared_ptr<ActionPolicy> target_policy, std::optional<StuckerConfig> stucker_config, std::optional<anet::seed_t> target_seed)
    : Learner(config, model, vars, std::move(obs_norm), batch_env_spec, env_spec, device, replay_seed, std::move(target_policy), stucker_config, target_seed)
{
    SetupReplayBuffer(batch_env_spec, env_spec, replay_seed);
    SetupOptimizer();
}

std::shared_ptr<const anet::rl::BatchUpdateResult>
TDLearner::UpdateFromSamples(const anet::rl::ExperienceSamples& samples)
{
    ProfileRange r("TDLearner::UpdateFromBatch");

    /// @todo コード整理

    const int B = config_.replay_batch_size;
    const int S = state_dim_;
    const int A = n_actions_;
    const auto& target_values = samples.target_values;
    const auto& terminals = samples.next_states.terminals;
    torch::ScalarType amp_dtype = config_.use_amp_bf16 ? torch::kBFloat16 : torch::kHalf;

    // Observation正規化
    torch::Tensor obs = samples.obs;
    torch::Tensor next_obs = samples.next_states.obs;
    if (obs_norm_) {
        // 統計更新は Agent 側の収集フェーズで行うためここでは適用のみ(false)
        obs = obs_norm_->Normalize(samples.obs);
        next_obs = obs_norm_->Normalize(samples.next_states.obs);
    }

    // 結果変数（スコープ外で宣言）
    torch::Tensor loss;
    torch::Tensor td_error;
    torch::Tensor max_q;
    torch::Tensor q_sa;
    torch::Tensor gap_abs;
    torch::Tensor gap_rel;

    // ============================================================
    // Forward & Loss Calculation (Autocast Scope)
    // ============================================================
    {
        // AMP Guard (Forward計算のみfloat16化)
        anet::Autocast amp_guard(torch::kCUDA, config_.use_amp, amp_dtype);

        // ------------------------------------------------------------
        // Q(s, a)
        // ------------------------------------------------------------
        auto q_out = model_.Forward(obs, /*use_target=*/false);
        auto q_all = q_out.At("q"); // (B,A)
        ANET_ASSERT_SHAPE(q_all, { B, A });

        torch::Tensor idx_actions = samples.actions.view({ B, 1 });   // (B,1)
        ANET_ASSERT_SHAPE(idx_actions, { B, 1 });
        ANET_ASSERT_DTYPE(idx_actions, torch::kInt64);

        q_sa = q_all.gather(1, idx_actions).squeeze(1);          // (B)
        ANET_ASSERT_SHAPE(q_sa, { B });
        ANET_ASSERT_DTYPE(q_sa, torch::kFloat32);

        max_q = std::get<0>(q_all.max(1)).detach();     // (B)

        // Gap Metrics
        if (n_actions_ >= 2) {  // 念の為
            auto top2 = std::get<0>(q_all.topk(2, 1));  //  (B, 2) 上位2つのQ値
            auto q_best = top2.select(1, 0);       // 1位 (B)
            auto q_second = top2.select(1, 1);     // 2位 (B)

        // 絶対値差分
            auto gap_batch = q_best - q_second;
            gap_abs = gap_batch.mean().detach();

        // Relative Gap (相対差分: Gap / (|MaxQ| + eps))
        //   Q値は負になることもあるので abs() が必要
        //   学習初期は 0 になるので 1e-6 で割るのを防ぐ
            auto denom = q_best.abs() + 1e-6f;
            gap_rel = (gap_batch / denom).mean().detach();
        }

        // ------------------------------------------------------------
        // max_a' Q(s', a')
        // ------------------------------------------------------------
        torch::Tensor max_next_q;
        {
            torch::NoGradGuard no_grad;

            // 行動 a' を決定
            // DDQN有効の場合は PolicyNetで行動選択、DDQN無効の場合はTargetNetで行動選択
            auto network = (config_.use_double_dqn) ? model_.GetMainNetwork() : model_.GetTargetNetwork();

            // target_policy_ に行動を選ばせる
            // ※ここで greedy_only=false にすることで、UQE設定時は楽観的に選ばれる
            auto target_action_info = target_policy_->SelectAction(next_obs, /*greedy_only=*/true, network, this->GetRandomGenerator());
            torch::Tensor next_actions = target_action_info.GetAction(device_);

            // 選んだ行動の価値を TargetNet で評価する (価値評価は常に TargetNet)
            auto next_q_out = model_.Forward(next_obs, /*use_target=*/true);
            auto next_q_target = next_q_out.At("q"); // (B, A)
            ANET_ASSERT_SHAPE(next_q_target, { B, A });

            torch::Tensor next_actions_b = next_actions.view({ B, 1 }); // (B,1)
            max_next_q = next_q_target.gather(1, next_actions_b).squeeze(1);
        }
        ANET_ASSERT_SHAPE(max_next_q, { B });
        ANET_ASSERT_DTYPE(max_next_q, torch::kFloat32);

        // ------------------------------------------------------------
        // TD target & TD Error
        // ------------------------------------------------------------
        auto not_terminal = 1.0f - terminals.to(torch::kFloat32); // (B,)
        auto gamma_n = torch::pow(config_.gamma, samples.n_steps.to(torch::kFloat32)); // (B,)
        auto td_target = target_values.detach() + not_terminal * gamma_n * max_next_q.detach(); // (B,)
        td_error = q_sa - td_target; // (B,)
        ANET_ASSERT_SHAPE(td_error, { B });
        ANET_ASSERT_DTYPE(td_error, torch::kFloat32);

        // ------------------------------------------------------------
        // Loss Calculation
        // ------------------------------------------------------------
        torch::Tensor td_error_for_loss = td_error;
        if (config_.use_td_clip && config_.td_clip_value > 0.0f)
            td_error_for_loss = torch::clamp(td_error_for_loss, -config_.td_clip_value, config_.td_clip_value);

        if (config_.use_per) {
            auto element_loss = torch::nn::functional::smooth_l1_loss(
                td_error_for_loss,
                torch::zeros_like(td_error_for_loss),
                torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kNone));
            auto weights = samples.is_weights.to(element_loss.device());
            loss = (element_loss * weights).mean();
        } else {
            loss = torch::nn::functional::smooth_l1_loss(
                td_error_for_loss,
                torch::zeros_like(td_error_for_loss),
                torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kMean));
        }
    } // End of Autocast Scope

    // ------------------------------------------------------------
    // PER Priority Update (NoGrad & NoAmp usually)
    // ------------------------------------------------------------

    // PER Metrics用 Tensor
    torch::Tensor metric_per_clipped_count;
    torch::Tensor metric_per_priorities;
    torch::Tensor metric_per_is_weights;

    if (config_.use_per) {
        torch::NoGradGuard grad_grad;

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
    //  Backward & Optimize (GradScaler)
    // ------------------------------------------------------------

    optimizer_->zero_grad();

    if (config_.use_amp && config_.use_amp_bf16) {
        // backward
        loss.backward();

        // grad_clip
        torch::Tensor grad_norm_tensor;
        std::optional<float> grad_norm;
        bool grad_clipped = false;
        if (config_.use_grad_clip) {
            double grad_norm_val = torch::nn::utils::clip_grad_norm_(model_.GetPolicyParameters(), config_.grad_clip_tau);
            grad_norm = static_cast<float>(grad_norm_val);
            grad_clipped = (grad_norm_val > config_.grad_clip_tau);
        } else {
            // クリップしない場合もノルム計算
            torch::Tensor total_sq = torch::zeros({ 1 }, loss.options());
            for (auto& p : model_.GetPolicyParameters()) {
                if (p.grad().defined()) {
                    total_sq += p.grad().detach().pow(2).sum();
                }
            }
            grad_norm_tensor = total_sq.sqrt();
            grad_norm = grad_norm_tensor.item<float>(); // CPU同期して値取得
        }
        float grad_clip_ratio = grad_clipped ? 1.0f : 0.0f;

        // パラメータ更新
        optimizer_->step();

        auto result = std::make_shared<BatchUpdateResult>();
        result->loss = loss;
        result->td_error = td_error;
        result->grad_norm = grad_norm;
        result->grad_norm_tensor = grad_norm_tensor;
        result->grad_clip_ratio = grad_clip_ratio;
        result->grad_clip_tau = config_.use_grad_clip ? config_.grad_clip_tau : std::numeric_limits<float>::infinity();
        result->max_q = max_q;
        result->q_sa = q_sa.detach();       // 実際に行動したQ値
        result->q_gap = gap_abs;
        result->q_gap_rel = gap_rel;
        if (config_.use_per) {
            result->per_minibatch_size = B;
            result->per_clipped_count = metric_per_clipped_count;
            result->per_priorities = metric_per_priorities;
            result->per_is_weights = metric_per_is_weights;
        }
        return result;
    } else if (config_.use_amp) {
        // AMP Mode
        
        // backward
        grad_scaler_.Scale(loss).backward();

        // Unscale (Clip前)
        grad_scaler_.Unscale_(*optimizer_);

        // Inf/NaN チェック (Unscale後の生勾配を見る)
        bool found_inf = false;
        // 簡易チェック: 全パラメータの勾配を見る
        for (auto& p : model_.GetPolicyParameters()) {
             if (p.grad().defined() && !torch::isfinite(p.grad()).all().item<bool>()) { /// @todo AMP：コスト削減のため一部チェックにする？
                 found_inf = true;
                 break;
             }
        }

        // Clip (Unscaled gradients)
        torch::Tensor grad_norm_tensor;
        std::optional<float> grad_norm;
        bool grad_clipped = false;

        if (!found_inf && config_.use_grad_clip) {
            double grad_norm_val = torch::nn::utils::clip_grad_norm_(model_.GetPolicyParameters(), config_.grad_clip_tau);
            grad_norm = static_cast<float>(grad_norm_val);
            grad_clipped = (grad_norm_val > config_.grad_clip_tau);
        }
        float grad_clip_ratio = grad_clipped ? 1.0f : 0.0f;

        // Step & Update
        // found_inf=trueなら内部でスキップされる
        grad_scaler_.Step(*optimizer_, found_inf);
        grad_scaler_.Update();


        // Result生成 (AMP時)
        auto result = std::make_shared<BatchUpdateResult>();
        result->loss = loss;
        result->td_error = td_error;
        result->grad_norm = grad_norm;
        result->grad_clip_ratio = grad_clip_ratio;
        result->grad_clip_tau = config_.use_grad_clip ? config_.grad_clip_tau : std::numeric_limits<float>::infinity();
        result->max_q = max_q;              // すべての行動の最大Q値
        result->q_sa = q_sa.detach();       // 実際に行動したQ値（追加
        result->q_gap = gap_abs;
        result->q_gap_rel = gap_rel;
        if (config_.use_per) {
            result->per_minibatch_size = B;
            result->per_clipped_count = metric_per_clipped_count;
            result->per_priorities = metric_per_priorities;
            result->per_is_weights = metric_per_is_weights;
        }

        return result;
    } else {
        // FP32 Mode
        
        // backward
        optimizer_->zero_grad();
        loss.backward();

        // grad_clip
        torch::Tensor grad_norm_tensor;
        std::optional<float> grad_norm;
        bool grad_clipped = false;
        if (config_.use_grad_clip) {
             // clip_grad_norm_ の戻り値は clip 前の全体ノルム
            double grad_norm_val = torch::nn::utils::clip_grad_norm_(model_.GetPolicyParameters(), config_.grad_clip_tau);   // use_grad_clip=true では CPU同期は現状避けられない
            grad_norm = static_cast<float>(grad_norm_val);
            grad_clipped = (grad_norm_val > config_.grad_clip_tau);
        } else {
            torch::Tensor total_sq = torch::zeros({ 1 }, loss.options());
            auto params = model_.GetPolicyParameters();
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
        // UpdateResult(FP32)
        // ------------------------------------------------------------
        auto result = std::make_shared<BatchUpdateResult>();
        result->loss = loss;
        result->td_error = td_error;
        result->grad_norm = grad_norm;
        result->grad_norm_tensor = grad_norm_tensor;
        result->grad_clip_ratio = grad_clip_ratio;
        result->grad_clip_tau = config_.use_grad_clip ? config_.grad_clip_tau : std::numeric_limits<float>::infinity();
        result->max_q = max_q;              // すべての行動の最大Q値
        result->q_sa = q_sa.detach();       // 実際に行動したQ値（追加
        result->q_gap = gap_abs;
        result->q_gap_rel = gap_rel;
        if (config_.use_per) {
            result->per_minibatch_size = B;
            result->per_clipped_count = metric_per_clipped_count;
            result->per_priorities = metric_per_priorities;
            result->per_is_weights = metric_per_is_weights;
        }
        return result;
    }
}


// ======================================================
// QRLearner
// ======================================================

QRLearner::QRLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
    const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, seed_t replay_seed,
    std::shared_ptr<ActionPolicy> target_policy, std::optional<StuckerConfig> stucker_config, std::optional<anet::seed_t> target_seed)
    : Learner(config, model, vars, std::move(obs_norm), batch_env_spec, env_spec, std::move(device), replay_seed, std::move(target_policy), stucker_config, target_seed)
{
    SetupReplayBuffer(batch_env_spec, env_spec, replay_seed);
    SetupOptimizer();

    // tau_iの事前計算
    const int N = config_.num_quantiles;
    tau_i_ = torch::arange(0.5f / N, 1.0f, 1.0f / N, device).view({ 1, N, 1 });
    ANET_ASSERT_SHAPE(tau_i_, { 1, N, 1 });
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

    // Huber Loss
    auto abs_diff = diff.abs();
    auto huber = torch::where(abs_diff < kappa, 0.5f * diff.pow(2), kappa * (abs_diff - 0.5f * kappa));
    ANET_ASSERT_SHAPE(huber, { B, N, N });

    // Quantile Regression Loss
    // rho_tau(u) = |tau - I(u<0)| * L_k(u)
    auto indicator = (diff.detach() < 0).to(torch::kFloat);
    auto quantile_weight = torch::abs(tau_i_ - indicator);	 // Broadcasting Check: (1, N, 1) - (B, N, N) -> (B, N, N)
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

    /// @todo コード整理

    const int B = config_.replay_batch_size;
    const int A = n_actions_;
    const int N = config_.num_quantiles;
    torch::ScalarType amp_dtype = config_.use_amp_bf16 ? torch::kBFloat16 : torch::kHalf;

    // 入力チェック
    ANET_ASSERT_SHAPE(samples.actions, { B });
    ANET_ASSERT_SHAPE(samples.target_values, { B });
    ANET_ASSERT_SHAPE(samples.next_states.terminals, { B });
    //ANET_LOG_DEBUG("obs=" << anet::ToDefString(samples.obs));

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

    // 結果変数
    torch::Tensor loss;
    torch::Tensor td_error_tensor;
    torch::Tensor max_q;
    torch::Tensor q_sa_val;
    torch::Tensor std_q_tensor;
    torch::Tensor gap_abs;
    torch::Tensor gap_rel;

    // 変数定義 (PER用)
    torch::Tensor current_dist; // (B, N)
    torch::Tensor target_dist;  // (B, N)
    torch::Tensor element_loss; // (B)

    // ============================================================
    // Forward & Loss Calculation (Autocast Scope)
    // ============================================================
    {
        // AMP Guard
        anet::Autocast amp_guard(torch::kCUDA, config_.use_amp, amp_dtype);

        // ------------------------------------------------------------
        // 分布計算
        // ------------------------------------------------------------

        // 現在の分布計算: Z(s, a)、ForwardQuantiles は (B, A, N) を返す
        auto current_out = model_.Forward(obs, /*use_target=*/false);
        auto current_dist_all = current_out.At("q_dist"); // (B, A, N)
        ANET_ASSERT_SHAPE(current_dist_all, { B, A, N });
        ANET_ASSERT_NAN(current_dist_all);

        // 選択された行動の分布を取得: (B, A, N) -> (B, N)
        torch::Tensor idx_actions = samples.actions.view({ B, 1, 1 }).expand({ B, 1, N });
        ANET_ASSERT_SHAPE(idx_actions, { B, 1, N });
        ANET_ASSERT_NAN(idx_actions);

        current_dist = current_dist_all.gather(1, idx_actions).squeeze(1); // (B, N)
        ANET_ASSERT_SHAPE(current_dist, { B, N });
        ANET_ASSERT_NAN(current_dist);

        // メトリクス用: 平均値をmax_qとして報告
        q_sa_val = current_dist.mean(1).detach(); // 選ばれた行動のQ値
        auto q_values_mean = current_out.At("q"); // すでに計算済みの平均Q値 (B, A)
        max_q = std::get<0>(q_values_mean.max(1)).detach(); // 全行動の最大Q値
        ANET_ASSERT_SHAPE(max_q, { B });

        // メトリクス用: Q Std (分布の標準偏差)、 GPUTensor (Scalar) のまま保持
        std_q_tensor = current_dist.std(1).mean().detach();
        ANET_ASSERT_SHAPE(std_q_tensor, {});

        // メトリクス用: トップActionと2位ActionのQ値Gap平均、 GPUTensor (Scalar) のまま保持
        if (n_actions_ >= 2) { // 念の為
        auto top2 = std::get<0>(q_values_mean.topk(2, 1));  //  (B, 2) 上位2つのQ値
        auto q_best = top2.select(1, 0);       // 1位 (B)
        auto q_second = top2.select(1, 1);     // 2位 (B)

        // 絶対値差分
            auto gap_batch = q_best - q_second;
            gap_abs = gap_batch.mean().detach();

        // Relative Gap (相対差分: Gap / (|MaxQ| + eps))
        //   Q値は負になることもあるので abs() が必要
        //   学習初期は 0 になるので 1e-6 で割るのを防ぐ
            auto denom = q_best.abs() + 1e-6f;
            gap_rel = (gap_batch / denom).mean().detach();
        }

        // ------------------------------------------------------------
        // ターゲット分布計算: r + gamma * Z(s', a*)
        // ------------------------------------------------------------
        {
            torch::NoGradGuard grad_guard;

            // 行動 a' を決定
            auto network = (config_.use_double_dqn) ? model_.GetMainNetwork() : model_.GetTargetNetwork();
            auto target_action_info = target_policy_->SelectAction(next_obs, /*greedy_only=*/true, network, this->GetRandomGenerator());
            torch::Tensor next_actions = target_action_info.GetAction(device_);
            ANET_ASSERT_SHAPE(next_actions, { B });

            // 次状態のターゲット分布: Z_target(s', :)
            auto next_out = model_.Forward(next_obs, /*use_target=*/true);
            auto next_dist_all = next_out.At("q_dist"); // (B, A, N)
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

            // gammaの n_step 乗を計算し、ブロードキャスト用に shape(B, 1) に変形
            auto gamma_n = torch::pow(config_.gamma, samples.n_steps.to(torch::kFloat32)).view({ B, 1 });

            // (B, 1) + (B, 1) * (B, 1) * (B, N) -> (B, N)
            target_dist = reward + gamma_n * not_terminal * next_dist;
            ANET_ASSERT_SHAPE(target_dist, { B, N });
            ANET_ASSERT_NAN(target_dist);
        }

        // target_dist: (B, N) -> mean -> (B)
        auto target_mean = target_dist.mean(1).detach();

        // TD誤差
        td_error_tensor = q_sa_val - target_mean;

        // ------------------------------------------------------------
        // Loss Calculation
        // ------------------------------------------------------------

        // 要素ごとのLoss (B) を取得  ※ここで重い計算を一回だけ行う
        element_loss = ComputeQuantileHuberLoss(current_dist, target_dist); // (B)
        ANET_ASSERT_SHAPE(element_loss, { B });
        ANET_ASSERT_NAN(element_loss);

        // 最適化用Loss(Scalar) ※ PERの重み (IS Weights) を適用
        torch::Tensor weights = config_.use_per ? samples.is_weights : torch::ones({ B }, device_);
        ANET_ASSERT_NAN(weights);
        loss = (element_loss * weights).mean();
        ANET_ASSERT_SHAPE(loss, {});
        ANET_ASSERT_NAN(loss);

    } // End of Autocast Scope

    // ------------------------------------------------------------
    //  Backward & Optimize (GradScaler)
    // ------------------------------------------------------------
    optimizer_->zero_grad();

    torch::Tensor grad_norm_tensor;
    std::optional<float> grad_norm;
    float grad_clip_ratio = 0.0f;

	// AMP + BF16 Mode
    if (config_.use_amp && config_.use_amp_bf16) {
        // backward
        loss.backward();

		// grad_clip
        bool grad_clipped = false;

        if (config_.use_grad_clip) {
            double grad_norm_val = torch::nn::utils::clip_grad_norm_(model_.GetPolicyParameters(), config_.grad_clip_tau);
            grad_norm = static_cast<float>(grad_norm_val);
            grad_clipped = (grad_norm_val > config_.grad_clip_tau);
        } else {
            torch::Tensor total_sq = torch::zeros({ 1 }, loss.options());
            for (auto& p : model_.GetPolicyParameters()) {
                if (!p.grad().defined()) continue;
                total_sq += p.grad().detach().pow(2).sum();
            }
            grad_norm_tensor = total_sq.sqrt();
        }
        grad_clip_ratio = grad_clipped ? 1.0f : 0.0f;

        // パラメータ更新
        optimizer_->step();
    } else if (config_.use_amp) {
        // AMP Mode
        
        // Scaleしてbackword()
        grad_scaler_.Scale(loss).backward();

        // Unscale
        grad_scaler_.Unscale_(*optimizer_);

        // Clip
        bool found_inf = false; // 簡易実装: step内でチェックさせる
        bool grad_clipped = false;

        if (config_.use_grad_clip) {
            double grad_norm_val = torch::nn::utils::clip_grad_norm_(model_.GetPolicyParameters(), config_.grad_clip_tau);
            grad_norm = static_cast<float>(grad_norm_val);
            grad_clipped = (grad_norm_val > config_.grad_clip_tau);
        } else {
            torch::Tensor total_sq = torch::zeros({ 1 }, loss.options());
            for (auto& p : model_.GetPolicyParameters()) {
                if (!p.grad().defined()) continue;
                total_sq += p.grad().detach().pow(2).sum();
            }
            grad_norm_tensor = total_sq.sqrt();
        }
        grad_clip_ratio = grad_clipped ? 1.0f : 0.0f;

        grad_scaler_.Step(*optimizer_, found_inf);
        grad_scaler_.Update();
    } else {
        // FP32 Mode
        
        // backward
        loss.backward();

        // 勾配NaNチェック
#if ANET_ENABLE_TENSOR_NAN_CHECK
        for (auto& param : model_.GetPolicyParameters()) {
        if (param.grad().defined()) { // 勾配が存在する場合
                ANET_ASSERT_NAN(param.grad());
            }
        }
#endif

        // 勾配ノルム計算とクリッピング (非同期)
        torch::Tensor total_sq = torch::zeros({ 1 }, loss.options());
        auto params = model_.GetPolicyParameters();
        for (auto& p : params) {
            if (!p.grad().defined()) continue;
            total_sq += p.grad().detach().pow(2).sum();
        }

        // Result側で grad_norm_tensor にフォールバックさせるため nullopt にする
        grad_norm = std::nullopt;
        grad_norm_tensor = total_sq.sqrt();

        if (config_.use_grad_clip) {
            // テンソルのまま非同期でスケール計算と乗算を行う
            torch::Tensor tau_tensor = torch::full({ 1 }, config_.grad_clip_tau, loss.options());
            torch::Tensor scale = (tau_tensor / (grad_norm_tensor + 1e-6)).clamp_max(1.0);

            for (auto& p : params) {
                if (!p.grad().defined()) continue;
                p.grad().detach().mul_(scale);
            }
        }

        // パラメータ更新
        optimizer_->step();
    }

    // ------------------------------------------------------------
    // PER優先度更新 (Outside AMP)
    // ------------------------------------------------------------

    torch::Tensor metric_per_clipped_count;
    torch::Tensor metric_per_priorities;
    torch::Tensor metric_per_is_weights;

    if (config_.use_per) {
        torch::NoGradGuard grad_guard;

        // 分布の平均(Q値)のTD誤差の絶対値からPER優先度を算出
        auto abs_td_error = td_error_tensor.abs().detach();
        auto new_priorities = abs_td_error + config_.per_eps;
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
        auto indices_cpu = samples.indices.cpu();   /// @todo PERの優先度更新にはCPU値が必要なので、ここで同期が発生する
        auto indices_ptr = indices_cpu.data_ptr<int64_t>();
        std::vector<int64_t> indices_vec(indices_ptr, indices_ptr + B);

        auto prios_cpu = new_priorities.cpu();
        auto prios_ptr = prios_cpu.data_ptr<float>();
        std::vector<float> priorities_vec(prios_ptr, prios_ptr + B);

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
    result->grad_clip_tau = config_.use_grad_clip ? config_.grad_clip_tau : std::numeric_limits<float>::infinity();
    result->max_q = max_q;              // すべての行動の最大Q値
    result->q_sa = q_sa_val;            // 実際に行動したQ値（追加）
    result->q_std = std_q_tensor;
    result->q_gap = gap_abs;
    result->q_gap_rel = gap_rel;
    if (config_.use_per) {
        result->per_minibatch_size = B;
        result->per_clipped_count = metric_per_clipped_count;
        result->per_priorities = metric_per_priorities;
        result->per_is_weights = metric_per_is_weights;
    }

    return result;
}

