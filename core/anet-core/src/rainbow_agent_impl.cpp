
#include "rainbow_agent_impl.hpp"
#include "anet/profile.hpp"
#include "anet/tensor_check.hpp"
#include "anet/tensor_util.hpp"
#include "anet/str_util.hpp"

using namespace anet::rl;
namespace LOG = anet::log;


// ======================================================
// RainbowAgent BaseQNet
// ======================================================

BaseQNet::BaseQNet(const RainbowAgentConfig& config, int64_t state_dim, int64_t n_actions)
        : state_dim_(state_dim), n_actions_(n_actions)
{
    ANET_ASSERT(state_dim_ > 0);
    ANET_ASSERT(n_actions_ > 0);

    fc1_ = register_module("fc1", torch::nn::Linear(state_dim_, config.nn_hidden1));
    fc2_ = register_module("fc2", torch::nn::Linear(config.nn_hidden1, config.nn_hidden2));

    InitWeightsLinear(fc1_, config.nn_init_mode, /*is_relu=*/true);
    InitWeightsLinear(fc2_, config.nn_init_mode, /*is_relu=*/true);
}

void BaseQNet::InitWeightsLinear(torch::nn::Linear& layer, int nn_init_mode, bool is_relu)
{
    if (nn_init_mode == 1) {
        torch::nn::init::xavier_uniform_(layer->weight);
        if (layer->bias.defined()) {
            torch::nn::init::zeros_(layer->bias);
        }
    } else if (nn_init_mode == 2) {
        if (is_relu) {
            torch::nn::init::kaiming_normal_(layer->weight, 0.0, torch::kFanIn, torch::kReLU);
        } else {
            torch::nn::init::kaiming_normal_(layer->weight, 0.0, torch::kFanIn, torch::kLinear);
        }
        if (layer->bias.defined()) {
            torch::nn::init::zeros_(layer->bias);
        }
    }
}


// ======================================================
// RainbowAgent PlainQNet
// ======================================================

PlainQNet::PlainQNet(const RainbowAgentConfig& config, int state_dim, int n_actions)
    : BaseQNet(config, state_dim, n_actions)
{
    fc3_ = register_module("fc3", torch::nn::Linear(config.nn_hidden2, n_actions_));
    InitWeightsLinear(fc3_, config.nn_init_mode, /*is_relu=*/false);
}

torch::Tensor PlainQNet::Forward(const torch::Tensor& obs)
{
    auto x = obs;
    x = torch::relu(fc1_->forward(x));
    x = torch::relu(fc2_->forward(x));
    x = fc3_->forward(x);
    return x;
}

std::optional<anet::TensorFunction> PlainQNet::GetTensorFunction(const std::string& key, const torch::Device& device, std::shared_ptr<std::shared_mutex> smutex)
{
    if (key == "forward" || key == "forward.q") {
        anet::TensorFunction fn = [this, device, smutex](const torch::Tensor& t) {
            auto tdev = t.to(device);
            std::shared_lock<std::shared_mutex> lock(*smutex);
            return Forward(tdev);
            };
        return fn;
    }

    return std::nullopt;
}


// ======================================================
// RainbowAgent DuelingQNet
// ======================================================

DuelingQNet ::DuelingQNet(const RainbowAgentConfig& config, int state_dim, int n_actions)
        : BaseQNet(config, state_dim, n_actions)
{
    value_ = register_module("value", torch::nn::Linear(config.nn_hidden2, 1));
    adv_ = register_module("adv", torch::nn::Linear(config.nn_hidden2, n_actions_));

    InitWeightsLinear(value_, config.nn_init_mode, /*is_relu=*/false);
    InitWeightsLinear(adv_, config.nn_init_mode, /*is_relu=*/false);
}

torch::Tensor DuelingQNet::Forward(const torch::Tensor& obs)
{
    auto x = obs;
    x = torch::relu(fc1_->forward(x));
    x = torch::relu(fc2_->forward(x));

    auto v = value_->forward(x);   // (B, 1)
    auto a = adv_->forward(x);     // (B, A)

    auto a_mean = a.mean(/*dim=*/1, /*keepdim=*/true);  // (B, 1)
    auto q = v + (a - a_mean);                          // (B, A)
    return q;
}

std::optional<anet::TensorFunction> DuelingQNet::GetTensorFunction(const std::string& key, const torch::Device& device, std::shared_ptr<std::shared_mutex> smutex)
{
    if (key == "forward" || key == "forward.q") {
        anet::TensorFunction fn = [this, device, smutex](const torch::Tensor& t) {
            auto tdev = t.to(device);
            std::shared_lock<std::shared_mutex> lock(*smutex);
            auto q = Forward(tdev);
            return q;
            };
        return fn;
    }
    if (key == "forward.va") {
        anet::TensorFunction fn = [this, device, smutex](const torch::Tensor& t) {
            auto x = t.to(device);
            std::shared_lock<std::shared_mutex> lock(*smutex);
            x = torch::relu(fc1_->forward(x));
            x = torch::relu(fc2_->forward(x));

            auto v = value_->forward(x);        // (B, 1)
            auto a = adv_->forward(x);          // (B, A)
            auto va = torch::cat({ v, a }, 1);  // [B, 1 + A]
            ANET_CHECK_SHAPE(va, { ANET_SHAPE_ANY, 1 + n_actions_ });

            return va;

            };
        return fn;
    }

    return std::nullopt;
}


// ======================================================
// RainbowAgent Network
// ======================================================

RainbowAgent::Network::Network(
    const RainbowAgentConfig& config, const torch::Device& device, std::shared_ptr<QNet> policy_net, std::shared_ptr<QNet> target_net)
    : config_(config), policy_net_(std::move(policy_net)), target_net_(std::move(target_net))
{
    ANET_ASSERT(policy_net_);
    ANET_ASSERT(target_net_);

    policy_net_->to(device);
    target_net_->to(device);
    target_net_->eval();

}

torch::Tensor RainbowAgent::Network::ForwardExpectation(const torch::Tensor& obs, bool use_target) const
{
    ANET_ASSERT(!policy_net_->IsDistributional());    /// @todo QR-DQN対応

    const auto& net = use_target ? target_net_ : policy_net_;
    auto q = net->Forward(obs);
 
    ///// @todo QR-DQN の場合は quantile の平均を取る
    //if (policy_net_->IsDistributional()) {
    //    return q.mean(-1);
    //}

    return q;
}

torch::Tensor RainbowAgent::Network::Forward(const torch::Tensor& obs, bool use_target) const
{
    const auto& net = use_target ? target_net_ : policy_net_;
    return net->Forward(obs);
}

torch::Tensor RainbowAgent::Network::ForwardQuantiles(const torch::Tensor& obs, bool use_target) const
{
    ANET_ASSERT(target_net_->IsDistributional());
    ANET_ASSERT(policy_net_->IsDistributional());

    const auto& net = use_target ? target_net_ : policy_net_;
    return net->Forward(obs);
}

std::vector<torch::Tensor> RainbowAgent::Network::GetPolicyParameters() const
{
    auto params = policy_net_->parameters();
    return params;
}

void RainbowAgent::Network::UpdateTarget(step_t learn_step)
{
    if (config_.hard_update_interval > 0) {
        if (learn_step % config_.hard_update_interval == 0) {
            HardUpdate();
        }
        return;
    }
    SoftUpdate();
}

void RainbowAgent::Network::SoftUpdate()
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

void RainbowAgent::Network::HardUpdate()
{
    auto p_params = policy_net_->parameters();
    auto t_params = target_net_->parameters();

    ANET_ASSERT(p_params.size() == t_params.size());

    for (size_t i = 0; i < p_params.size(); ++i) {
        t_params[i].data().copy_(p_params[i].data());
    }
}

/// メトリクス用：NN生出力
std::optional<anet::TensorFunction> RainbowAgent::Network::GetTensorFunction(const std::string& key, const torch::Device& device, std::shared_ptr<std::shared_mutex> smutex)
{
    static constexpr const char* POLICY_PREFIX = "policy_net.";
    static constexpr const char* TARGET_PREFIX = "target_net.";

    // policy net
    if (anet::StartsWith(key, POLICY_PREFIX)) {
        auto subkey = anet::RemovePrefix(key, POLICY_PREFIX);
        auto fn = policy_net_->GetTensorFunction(subkey, device, smutex);
        return fn;
    }

    // target net
    if (anet::StartsWith(key, TARGET_PREFIX)) {
        auto subkey = anet::RemovePrefix(key, TARGET_PREFIX);
        auto fn = target_net_->GetTensorFunction(subkey, device, smutex);
        return fn;
    }

    return std::nullopt;
}


// ======================================================
// RainbowAgent ActionPolicy 
// ======================================================

RainbowAgent::ActionPolicy::ActionPolicy(
    const RainbowAgent::Network& network, const RainbowAgent::RuntimeVars& vars, seed_t seed)
    : RandomHolder(seed), network_(network), vars_(vars)
{
    // RandomHolderを継承しているが、GPU側でrand生成しているので意味はない。ただ、マークとしてそのままにしておく。
}

BatchActionInfo RainbowAgent::ActionPolicy::SelectAction(const torch::Tensor& obs, bool greedy_only, bool use_target) const
{
    ProfileRange  r("RainbowAgent::SelectAction");

    torch::NoGradGuard;

    // Q値生成
    auto q_values = network_.ForwardExpectation(obs, use_target);

    auto device = q_values.device();
    
    // shape: (N, A)

    // shape 読み取りは TensorOptions 経由で同期を回避
    const int64_t N = q_values.sizes()[0];
    const int64_t A = q_values.sizes()[1];

    // greedy = argmax(q_values, dim=1)
    auto greedy = q_values.argmax(1, /*keepdim=*/false);

    if (greedy_only) {
        ProfileRange  r("RainbowAgent::SelectAction.greedy_only");
        BatchActionInfo action_info{ greedy };

        // aux[max_q]
        auto max_pair = q_values.max(1);
        auto max_q = std::get<0>(max_pair).detach();
        action_info.GetAuxData()["max_q"] = max_q;

        return action_info;
    }

    const float eps = vars_.epsilon;

    // mask: (N) bool, GPU上で生成
    auto mask = torch::rand({ N }, torch::TensorOptions().device(device)).lt(eps);    // GPUで完結

    // random actions (N) int64
    auto random_actions = torch::randint(/*low=*/0, /*high=*/A, { N },
        torch::TensorOptions().dtype(torch::kInt64).device(device));

    // actions: where(mask, random_actions, greedy)
    auto actions = torch::where(mask, random_actions, greedy);

    BatchActionInfo action_info{ actions }; // (N) kInt64

    // aux[max_q]
    auto max_pair = q_values.max(1);
    auto max_q = std::get<0>(max_pair).detach();
    action_info.GetAuxData()["max_q"] = max_q;

    return action_info;
}


// ======================================================
// RainbowAgent Learner
// ======================================================

RainbowAgent::Learner::Learner(RainbowAgent& agent)
    : agent_(agent)
{
    ;
}

std::optional<float> RainbowAgent::Learner::GetScalar(const std::string& key, int index) const
{
    if (key.find("replaybuffer.") == 0 && replay_buffer_ != nullptr) {
        return replay_buffer_->GetScalar(key);
    }

    return std::nullopt;
}

std::optional<torch::Tensor> RainbowAgent::Learner::GetTensor(const std::string& key, int index) const
{
    if (key.find("replaybuffer.") == 0 && replay_buffer_ != nullptr)
        return replay_buffer_->GetTensor(key);

    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>> RainbowAgent::Learner::GetTensorVector(const std::string& key, int index) const
{
    if (key.find("replaybuffer.") == 0 && replay_buffer_ != nullptr)
        return replay_buffer_->GetTensorVector(key);

    return std::nullopt;
}


// ======================================================
// RainbowAgent TDLearner
// ======================================================

RainbowAgent::TDLearner::TDLearner(RainbowAgent& agent, const EnvSpec& env_spec, seed_t replay_seed)
    : Learner(agent)
{
    // ReplayBufferConfig
    anet::rl::ReplayBufferConfig rep_config{};
    rep_config.capacity = agent_.config_.replay_capacity;
    rep_config.gamma = agent_.config_.gamma;
    rep_config.n_step = agent_.config_.n_step;

    // N-Step 設定
    if (agent_.config_.use_n_step)
        rep_config.type = ReplayBuilderType::NSTEP;
    else
        rep_config.type = ReplayBuilderType::PLAIN;

    // PER 設定
    if (agent_.config_.use_per) {
        rep_config.sampler_type = ReplaySamplerType::PRIOTIZED;
        rep_config.per_alpha = agent_.config_.per_alpha;
        rep_config.per_initial_priority = agent_.config_.per_initial_priority;

        // RuntimeVarsのbeta初期化
        agent_.vars_->per_beta = agent_.config_.per_beta_start;
    } else {
        rep_config.sampler_type = ReplaySamplerType::UNIFORM;
    }

    // ReplayBuffer生成
    anet::rl::ReplayBufferFactory rep_factory(rep_config);
    //if (agent_.config_.use_n_step) {
        //this->replay_buffer_ = rep_factory.Create(env_spec, agent.device_, agent_.batch_size_, replay_seed);      // GPUだと遅くなる
        this->replay_buffer_ = rep_factory.Create(env_spec, torch::kCPU, agent_.batch_size_, replay_seed);
    //} else {
    //    this->replay_buffer_ = std::make_shared<anet::rl::PlainReplayBuffer>(env_spec, agent_.config_.replay_capacity, replay_seed);
    //}

    // Optimizer生成
    this->optimizer_ = std::make_unique<torch::optim::Adam>(agent_.network_->GetPolicyParameters(), torch::optim::AdamOptions(agent_.config_.alpha));
}

bool RainbowAgent::TDLearner::CanUpdate(step_t update_step) const
{
    //if (replay_buffer_->Size() < agent_.config_.replay_batch_size)
    //    return false;

    /// @todo ReplayBuffer充足チェックにexp_stepを使う？

    // warmup（batch_sizeが大きいとexpが早く溜まるので補正）
    if (update_step < agent_.config_.update_warmup_steps * agent_.batch_size_)
        return false;

    // update_interval間隔で更新
    if ((update_step % agent_.config_.update_interval) != 0)
        return false;

    return true;
}

void RainbowAgent::TDLearner::UpdateEpsilon(step_t learn_step)
{
    ProfileRange r("RainbowAgent::TDLearner::UpdateEpsilon");

    const auto& config = agent_.config_;
    auto& vars = *agent_.vars_;

    if (learn_step >= config.eps_decay_step) {
        vars.epsilon = config.eps_min;
        return;
    }

    const float t = static_cast<float>(learn_step) / static_cast<float>(config.eps_decay_step);

    vars.epsilon = config.eps_max + t * (config.eps_min - config.eps_max);
}

void RainbowAgent::TDLearner::UpdateTargetNetwork(step_t step)
{
    ProfileRange r("RainbowAgent::TDLearner::UpdateTargetNetwork");

    agent_.network_->UpdateTarget(step);
}

std::shared_ptr<const anet::rl::BatchUpdateResult>
RainbowAgent::TDLearner::UpdateFromBatch(const StepCounts& counts, const BatchExperience& experiences, const Runner& trainer)
{
    ProfileRange r("RainbowAgent::TDLearner::UpdateFromBatch");

    const auto& update_step = counts.update_step;
    const auto& config = agent_.config_;
    auto& network = *agent_.network_;
    auto& vars = *agent_.vars_;

    const int B = config.replay_batch_size;
    const int S = agent_.state_dim_;
    const int A = agent_.n_actions_;
    const torch::Device& device = agent_.device_;

    // ------------------------------------------------------------
    // ReplayBuffer へ push
    // ------------------------------------------------------------
    replay_buffer_->Push(experiences);

    // Update不可ならLeanStep差分無しで返す
    if (!CanUpdate(update_step)) {
        auto result = std::make_shared<anet::rl::RainbowAgent::BatchUpdateResult>(0);
        return result;
    }

    // ------------------------------------------------------------
    // Sample
    // ------------------------------------------------------------
    // PERの場合は現在のbetaを使用、それ以外は無視される
    float current_beta = config.use_per ? vars.per_beta : 0.0f;
    auto raw_samples = replay_buffer_->Sample(config.replay_batch_size, device, current_beta);

    // Check shapes & dtypes
    ANET_CHECK_DEVICE(raw_samples.obs, device);
    ANET_CHECK_DEVICE(raw_samples.actions, device);
    ANET_CHECK_DEVICE(raw_samples.target_values, device);
    ANET_CHECK_DEVICE(raw_samples.next_states.obs, device);
    ANET_CHECK_DEVICE(raw_samples.next_states.terminals, device);
    ANET_CHECK_DEVICE(raw_samples.n_steps, device);
    ANET_CHECK_SHAPE(raw_samples.obs, { B, S });
    ANET_CHECK_SHAPE(raw_samples.actions, { B });    // 離散アクション
    ANET_CHECK_SHAPE(raw_samples.target_values, { B });
    ANET_CHECK_SHAPE(raw_samples.next_states.obs, { B, S });
    ANET_CHECK_SHAPE(raw_samples.next_states.terminals, { B });
    ANET_CHECK_SHAPE(raw_samples.n_steps, { B });
    ANET_CHECK_DTYPE(raw_samples.obs, torch::kFloat32);
    ANET_CHECK_DTYPE(raw_samples.actions, torch::kInt64);    // 離散アクション
    ANET_CHECK_DTYPE(raw_samples.target_values, torch::kFloat32);
    ANET_CHECK_DTYPE(raw_samples.next_states.terminals, torch::kBool);
    ANET_CHECK_DTYPE(raw_samples.n_steps, torch::kInt64);

    auto samples = raw_samples.FlattenStates();

    const auto& obs = samples.obs;
    // const auto& actions = samples.actions; // 未使用
    const auto& target_values = samples.target_values;
    const auto& next_obs = samples.next_states.obs;
    const auto& terminals = samples.next_states.terminals;

    // ------------------------------------------------------------
    // Q(s, a)
    // ------------------------------------------------------------
    auto q_all = network.Forward(obs, /*use_target=*/false);      // (B,A)
    ANET_CHECK_SHAPE(q_all, { B, A });

    torch::Tensor idx_actions = samples.actions.view({ B, 1 });   // (B,1)
    ANET_CHECK_SHAPE(idx_actions, { B, 1 });
    ANET_CHECK_DTYPE(idx_actions, torch::kInt64);

    auto q_sa = q_all.gather(1, idx_actions).squeeze(1);          // (B)
    ANET_CHECK_SHAPE(q_sa, { B });
    ANET_CHECK_DTYPE(q_sa, torch::kFloat32);

    torch::Tensor max_q = std::get<0>(q_all.max(1)).detach();     // (B)

    // ------------------------------------------------------------
    // max_a' Q(s', a')
    // ------------------------------------------------------------
    torch::Tensor max_next_q;

    if (config.use_double_dqn) {
        torch::NoGradGuard no_grad;

        // Double DQN: policy_netで行動選択、target_netで価値計算
        auto next_q_policy = network.Forward(next_obs, /*use_target=*/false);
        ANET_CHECK_SHAPE(next_q_policy, { B, A });
        auto next_actions = std::get<1>(next_q_policy.max(1));
        ANET_CHECK_SHAPE(next_actions, { B });

        // target_net で Q_target(s', argmax_a Q_online)
        auto next_q_target = network.Forward(next_obs, /*use_target=*/true);
        ANET_CHECK_SHAPE(next_q_target, { B, A });
        torch::Tensor next_actions_b = next_actions.view({ B, 1 });             // (B,1)
        max_next_q = next_q_target.gather(1, next_actions_b).squeeze(1);
    } else {
        torch::NoGradGuard;
        auto next_q_target = network.Forward(next_obs, /*use_target=*/true);
        ANET_CHECK_SHAPE(next_q_target, { B, A });
        max_next_q = std::get<0>(next_q_target.max(1));
    }
    ANET_CHECK_SHAPE(max_next_q, { B });
    ANET_CHECK_DTYPE(max_next_q, torch::kFloat32);

    // ------------------------------------------------------------
    // TD target & TD Error
    // ------------------------------------------------------------
    auto not_terminal = 1.0f - terminals.to(torch::kFloat32); // (B,)
    auto td_target = target_values + not_terminal * config.gamma * max_next_q.detach(); // (B,)
    ANET_CHECK_SHAPE(td_target, { B });
    ANET_CHECK_DTYPE(td_target, torch::kFloat32);

    auto td_error = q_sa - td_target; // (B,)

    // ------------------------------------------------------------
    // PER Priority Update
    // ------------------------------------------------------------

    // PER Metrics用 Tensor
    torch::Tensor metric_per_clipped_count;
    torch::Tensor metric_per_priorities;
    torch::Tensor metric_per_is_weights;

    if (config.use_per) {
        torch::NoGradGuard no_grad;

        // 優先度 = |td_error| + eps
        auto abs_td_error = td_error.abs().detach();
        auto new_priorities = abs_td_error + config.per_eps;

        // Priority Clipping
        if (config.use_per_prio_clip) {
            // [Metric Source] Clipped Count (Keep as Tensor)
            metric_per_clipped_count = (new_priorities > config.per_prio_clip_value).sum(); // (Scalar Tensor)

            // クリップ実行
            new_priorities = torch::clamp(new_priorities, 0.0f, config.per_prio_clip_value);
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
    if (config.use_td_clip && config.td_clip_value > 0.0f)
        td_error_for_loss = torch::clamp(td_error_for_loss, -config.td_clip_value, config.td_clip_value);

    torch::Tensor loss;

    if (config.use_per) {
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
    if (config.use_grad_clip) {
        // clip_grad_norm_ の戻り値は clip 前の全体ノルム
        double grad_norm_val = torch::nn::utils::clip_grad_norm_(
                network.GetPolicyParameters(), config.grad_clip_tau);   // use_grad_clip=true では CPU同期は現状避けられない
        grad_norm = static_cast<float>(grad_norm_val);
        grad_clipped = (grad_norm_val > config.grad_clip_tau);
    } else {
        torch::Tensor total_sq = torch::zeros({ 1 }, loss.options());
        auto params = network.GetPolicyParameters();
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
    vars.learn_step++;

    // ------------------------------------------------------------
    // Post Updates
    // ------------------------------------------------------------
    UpdateTargetNetwork(vars.learn_step);
    UpdateEpsilon(vars.learn_step);

    // PER Beta Update
    if (config.use_per) {
        if (vars.learn_step < config.per_beta_step) {
            float progress = static_cast<float>(vars.learn_step) / static_cast<float>(config.per_beta_step);
            vars.per_beta = config.per_beta_start + progress * (config.per_beta_end - config.per_beta_start);
        } else {
            vars.per_beta = config.per_beta_end;
        }
    }

    // ------------------------------------------------------------
    // UpdateResult
    // ------------------------------------------------------------
    auto result = std::make_shared<anet::rl::RainbowAgent::BatchUpdateResult>(1);
    result->loss = loss;
    result->td_error = td_error;
    result->grad_norm = grad_norm;
    result->grad_norm_tensor = grad_norm_tensor;
    result->grad_clip_ratio = grad_clip_ratio;
    result->max_q = max_q;
    if (config.use_per) {
        result->per_minibatch_size = B;
        result->per_clipped_count = metric_per_clipped_count;
        result->per_priorities = metric_per_priorities;
        result->per_is_weights = metric_per_is_weights;
    }
    return result;
}
