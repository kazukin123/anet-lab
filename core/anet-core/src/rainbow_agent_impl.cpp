
#include "rainbow_agent_impl.hpp"
#include "anet/profile.hpp"
#include "anet/tensor_check.hpp"
#include "anet/tensor_util.hpp"

using namespace anet::rl;
namespace LOG = anet::log;

// ======================================================
// RainbowAgent Network
// ======================================================

RainbowAgent::Network::Network(
    const RainbowAgentConfig& config, std::shared_ptr<QNet> policy_net, std::shared_ptr<QNet> target_net)
    : config_(config), policy_net_(std::move(policy_net)), target_net_(std::move(target_net))
{
    ANET_ASSERT(policy_net_);
    ANET_ASSERT(target_net_);
}

torch::Tensor RainbowAgent::Network::ForwardExpectation(const torch::Tensor& obs) const
{
    ANET_ASSERT(!policy_net_->IsDistributional());    /// @todo QR-DQN対応

    auto q = policy_net_->forward(obs);
 
    ///// @todo QR-DQN の場合は quantile の平均を取る
    //if (policy_net_->IsDistributional()) {
    //    return q.mean(-1);
    //}

    return q;
}

torch::Tensor RainbowAgent::Network::ForwardRaw(const torch::Tensor& obs, bool use_target) const
{
    const auto& net = use_target ? target_net_ : policy_net_;
    return net->forward(obs);
}

torch::Tensor RainbowAgent::Network::ForwardQuantiles(const torch::Tensor& obs, bool use_target) const
{
    ANET_ASSERT(target_net_->IsDistributional());
    ANET_ASSERT(policy_net_->IsDistributional());

    const auto& net = use_target ? target_net_ : policy_net_;
    return net->forward(obs);
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

// ======================================================
// RainbowAgent PlainQNet
// ======================================================

PlainQNet::PlainQNet(const RainbowAgentConfig& config, int state_dim, int n_actions)
    : state_dim_(state_dim), n_actions_(n_actions)
{
    ANET_ASSERT(state_dim_ > 0);
    ANET_ASSERT(n_actions > 0);

    fc1_ = register_module("fc1", torch::nn::Linear(state_dim_, config.nn_hidden1));
    fc2_ = register_module("fc2", torch::nn::Linear(config.nn_hidden1, config.nn_hidden2));
    fc3_ = register_module("fc3", torch::nn::Linear(config.nn_hidden2, n_actions_));

    InitWeights(config.nn_init_mode);
}

torch::Tensor PlainQNet::forward(const torch::Tensor& obs)
{
    auto x = obs;
    x = torch::relu(fc1_->forward(x));
    x = torch::relu(fc2_->forward(x));
    x = fc3_->forward(x);
    return x;
}

void PlainQNet::InitWeights(int nn_init_mode)
{
    if (nn_init_mode == 1) {
        torch::nn::init::xavier_uniform_(fc1_->weight);
        torch::nn::init::xavier_uniform_(fc2_->weight);
        torch::nn::init::xavier_uniform_(fc3_->weight);
    } else if (nn_init_mode == 2) {
        torch::nn::init::kaiming_normal_(fc1_->weight, 0.0, torch::kFanIn, torch::kReLU);
        torch::nn::init::kaiming_normal_(fc2_->weight, 0.0, torch::kFanIn, torch::kReLU);
        torch::nn::init::kaiming_normal_(fc3_->weight, 0.0, torch::kFanIn, torch::kLinear);
    }
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

BatchActionInfo RainbowAgent::ActionPolicy::SelectAction(const torch::Tensor& obs, bool greedy_only) const
{
    ProfileRange  r("RainbowAgent::SelectAction");

    // Q値生成
    auto q_values = network_.ForwardExpectation(obs);

    auto device = q_values.device();
    
    // shape: (N, A)

    // shape 読み取りは TensorOptions 経由で同期を回避
    const int64_t N = q_values.sizes()[0];
    const int64_t A = q_values.sizes()[1];

    // greedy = argmax(q_values, dim=1)
    auto greedy = q_values.argmax(1, /*keepdim=*/false);

    if (greedy_only) {
        ProfileRange  r("RainbowAgent::SelectAction.greedy_only");
        auto zeros = torch::zeros({ N }, torch::TensorOptions().dtype(torch::kBool).device(device));
        BatchActionInfo action_info{ greedy, zeros };

        // aux[max_q]
        auto max_pair = q_values.max(1);
        auto max_q = std::get<0>(max_pair).detach();
        action_info.aux["max_q"] = max_q;

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

    BatchActionInfo action_info {
        actions,        // (N) kInt64
        mask            // (N) kBool
    };

    // aux[max_q]
    auto max_pair = q_values.max(1);
    auto max_q = std::get<0>(max_pair).detach();
    action_info.aux["max_q"] = max_q;

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
    // ReplayBuffer生成
    this->replay_buffer_ = std::make_unique<anet::rl::PlainReplayBuffer>(env_spec, agent_.config_.replay_capacity, replay_seed);
    /// @todo ReplayBufferFactoryを利用

    // Optimizer生成
    this->optimizer_ = std::make_unique<torch::optim::Adam>(agent_.network_->GetPolicyParameters(), torch::optim::AdamOptions(agent_.config_.alpha));
}

bool RainbowAgent::TDLearner::CanUpdate(step_t update_step) const
{
    //if (replay_buffer_->Size() < agent_.config_.replay_batch_size)
    //    return false;

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

    const auto& train_step = counts.train_step;
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
    auto raw_samples = replay_buffer_->Sample(config.replay_batch_size, device);
    ANET_CHECK_DEVICE(raw_samples.obs, device);
    ANET_CHECK_DEVICE(raw_samples.actions, device);
    ANET_CHECK_DEVICE(raw_samples.rewards, device);
    ANET_CHECK_DEVICE(raw_samples.next_states.obs, device);
    ANET_CHECK_DEVICE(raw_samples.next_states.dones, device);
    ANET_CHECK_DEVICE(raw_samples.next_states.truncateds, device);
    ANET_CHECK_DEVICE(raw_samples.next_states.episode_start, device);
    ANET_CHECK_SHAPE(raw_samples.obs, { B, S });
    ANET_CHECK_SHAPE(raw_samples.actions, { B, 1 });    // 離散アクション
    ANET_CHECK_SHAPE(raw_samples.rewards, { B });
    ANET_CHECK_SHAPE(raw_samples.next_states.obs, { B, S });
    ANET_CHECK_SHAPE(raw_samples.next_states.dones, { B });
    ANET_CHECK_SHAPE(raw_samples.next_states.truncateds, { B });
    ANET_CHECK_SHAPE(raw_samples.next_states.episode_start, { B });
    ANET_CHECK_DTYPE(raw_samples.obs, torch::kFloat32);
    ANET_CHECK_DTYPE(raw_samples.actions, torch::kInt64);    // 離散アクション
    ANET_CHECK_DTYPE(raw_samples.rewards, torch::kFloat32);
    ANET_CHECK_DTYPE(raw_samples.next_states.dones, torch::kBool);
    ANET_CHECK_DTYPE(raw_samples.next_states.truncateds, torch::kBool);
    ANET_CHECK_DTYPE(raw_samples.next_states.episode_start, torch::kBool);

    auto samples = raw_samples.Flatten();

    const auto& obs = samples.obs;
    const auto& actions = samples.actions;
    const auto& rewards = samples.rewards;
    const auto& next_obs = samples.next_states.obs;
    const auto& dones = samples.next_states.dones;
    const auto& truncateds = samples.next_states.truncateds;

    // ------------------------------------------------------------
    // Q(s, a)
    // ------------------------------------------------------------
    auto q_all = network.ForwardRaw(obs, /*use_target=*/false);
    ANET_CHECK_SHAPE(q_all, { B, A });

    torch::Tensor idx_actions = samples.actions.view({ B, 1 });   // (B,1)
    ANET_CHECK_SHAPE(idx_actions, { B, 1 });
    ANET_CHECK_DTYPE(idx_actions, torch::kInt64);

    auto q_sa = q_all.gather(1, idx_actions).squeeze(1);
    ANET_CHECK_SHAPE(q_sa, { B });
    ANET_CHECK_DTYPE(q_sa, torch::kFloat32);

    torch::Tensor max_q = std::get<0>(q_all.max(1)).detach(); // (B,)

    // ------------------------------------------------------------
    // max_a' Q(s', a')
    // ------------------------------------------------------------
    torch::Tensor max_next_q;

    if (config.use_double_dqn) {
        torch::NoGradGuard no_grad;

        // policy_net で argmax_a Q(s', a)
        auto next_q_policy = network.ForwardRaw(next_obs, /*use_target=*/false);
        ANET_CHECK_SHAPE(next_q_policy, { B, A });
        auto next_actions = std::get<1>(next_q_policy.max(1));
        ANET_CHECK_SHAPE(next_actions, { B });

        // target_net で Q_target(s', argmax_a Q_online)
        auto next_q_target = network.ForwardRaw(next_obs, /*use_target=*/true);
        ANET_CHECK_SHAPE(next_q_target, { B, A });
        torch::Tensor next_actions_b = next_actions.view({ B, 1 });             // (B,1)
        max_next_q = next_q_target.gather(1, next_actions_b).squeeze(1);
    } else {
        auto next_q_target = network.ForwardRaw(next_obs, /*use_target=*/true);
        ANET_CHECK_SHAPE(next_q_target, { B, A });
        max_next_q = std::get<0>(next_q_target.max(1));
    }
    ANET_CHECK_SHAPE(max_next_q, { B });
    ANET_CHECK_DTYPE(max_next_q, torch::kFloat32);

    // ------------------------------------------------------------
    // TD target
    // ------------------------------------------------------------
    auto terminal = (dones | truncateds); // (B,) bool
    auto not_terminal = 1.0f - terminal.to(torch::kFloat32); // (B,)
    auto sample_rewards = samples.rewards; // (B,)
    auto td_target = sample_rewards + not_terminal * config.gamma * max_next_q.detach(); // (B,)
    ANET_CHECK_SHAPE(td_target, { B });
    ANET_CHECK_DTYPE(td_target, torch::kFloat32);

    // ------------------------------------------------------------
    // TD loss
    // ------------------------------------------------------------
    auto td_error = q_sa - td_target;
    torch::Tensor td_error_for_loss = td_error;
    if (config.use_td_clip && config.td_clip_value > 0.0f)
        td_error_for_loss = torch::clamp(td_error_for_loss, -config.td_clip_value, config.td_clip_value);
    auto loss = torch::nn::functional::smooth_l1_loss(
        td_error_for_loss,
        torch::zeros_like(td_error_for_loss),
        torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kMean));

    // ------------------------------------------------------------
    //  backward
    // ------------------------------------------------------------
    optimizer_->zero_grad();
    loss.backward();

    // ------------------------------------------------------------
    // grad_clip
    // ------------------------------------------------------------
    float grad_norm = 0.0f;
    bool grad_clipped = false;
    if (config.use_grad_clip) {
        // clip_grad_norm_ の戻り値は clip 前の全体ノルム
        double grad_norm_val = torch::nn::utils::clip_grad_norm_(
                network.GetPolicyParameters(), config.grad_clip_tau);
        grad_norm = static_cast<float>(grad_norm_val);
        grad_clipped = (grad_norm_val > config.grad_clip_tau);
    } else {
        // clip を使わない場合も全体ノルムだけは計算しておく
        torch::Tensor total_sq = torch::zeros({ 1 }, loss.options());
        auto params = network.GetPolicyParameters();
        for (auto& p : params) {
            if (!p.grad().defined()) continue;
            total_sq += p.grad().data().pow(2).sum();
        }
        grad_norm = std::sqrt(total_sq.item<float>());
    }
    float grad_clip_ratio = grad_clipped ? 1.0f : 0.0f;
    /// @todo grad_normの遅延評価 or 評価無しによる性能検討

    // ------------------------------------------------------------
    // optimize
    // ------------------------------------------------------------
    optimizer_->step();
    vars.learn_step++;

    // ------------------------------------------------------------
    // target network 更新
    // ------------------------------------------------------------
    UpdateTargetNetwork(vars.learn_step);

    // ------------------------------------------------------------
    // epsilon 更新
    // ------------------------------------------------------------
    UpdateEpsilon(vars.learn_step);

    // ------------------------------------------------------------
    // UpdateResult
    // ------------------------------------------------------------
    auto result = std::make_shared<anet::rl::RainbowAgent::BatchUpdateResult>(1);
    result->loss = loss;
    result->td_error = td_error;
    result->grad_norm = grad_norm;
    result->grad_clip_ratio = grad_clip_ratio;
    result->max_q = max_q;

    return result;
}
