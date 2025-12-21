
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

std::optional<anet::TensorFunction> DuelingQNet::GetTensorFunction(
    const std::string& key, const torch::Device& device, std::shared_ptr<std::shared_mutex> smutex)
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
    if (key == "forward.v") {
        anet::TensorFunction fn = [this, device, smutex](const torch::Tensor& t) {
            auto x = t.to(device);
            std::shared_lock<std::shared_mutex> lock(*smutex);
            x = torch::relu(fc1_->forward(x));
            x = torch::relu(fc2_->forward(x));

            auto v = value_->forward(x);        // (B, 1)
            ANET_CHECK_SHAPE(v, { ANET_SHAPE_ANY, 1 });

            return v;
            };
        return fn;
    }

    return std::nullopt;
}


// ======================================================
// RainbowAgent QuantilePlainQNet
// ======================================================

QuantilePlainQNet::QuantilePlainQNet(const RainbowAgentConfig& config, int state_dim, int n_actions)
    : BaseQNet(config, state_dim, n_actions), num_quantiles_(config.num_quantiles)
{
    ANET_ASSERT(num_quantiles_ > 1);

    // 出力層: Actions * Quantiles
    fc3_ = register_module("fc3", torch::nn::Linear(config.nn_hidden2, n_actions_ * num_quantiles_));
    InitWeightsLinear(fc3_, config.nn_init_mode, /*is_relu=*/false);
}

torch::Tensor QuantilePlainQNet::Forward(const torch::Tensor& obs)
{
    // 分布を計算し、平均を取って期待値Qとする
    auto q_dist = ForwardQuantiles(obs); // (B, A, N)
    return q_dist.mean(2);               // (B, A) -> mean over quantiles
}

torch::Tensor QuantilePlainQNet::ForwardQuantiles(const torch::Tensor& obs)
{
    auto x = obs;
    x = torch::relu(fc1_->forward(x));
    x = torch::relu(fc2_->forward(x));
    x = fc3_->forward(x); // (B, A * N)

    // Reshape to (B, A, N)
    auto batch_size = x.size(0);
    x = x.view({ batch_size, n_actions_, num_quantiles_ });

    return x;
}

std::optional<anet::TensorFunction> QuantilePlainQNet::GetTensorFunction(const std::string& key, const torch::Device& device, std::shared_ptr<std::shared_mutex> smutex)
{
    // 既存の可視化用キー
    if (key == "forward" || key == "forward.q") {
        return [this, device, smutex](const torch::Tensor& t) {
            auto tdev = t.to(device);
            std::shared_lock<std::shared_mutex> lock(*smutex);
            return Forward(tdev);
            };
    }
    // 分布可視化用
    if (key == "forward.dist") {
        return [this, device, smutex](const torch::Tensor& t) {
            auto tdev = t.to(device);
            std::shared_lock<std::shared_mutex> lock(*smutex);
            return ForwardQuantiles(tdev);
            };
    }
    return std::nullopt;
}


// ======================================================
// RainbowAgent QuantileDuelingQNet
// ======================================================

QuantileDuelingQNet::QuantileDuelingQNet(const RainbowAgentConfig& config, int state_dim, int n_actions)
    : BaseQNet(config, state_dim, n_actions), num_quantiles_(config.num_quantiles)
{
    ANET_ASSERT(num_quantiles_ > 1);

    // Value stream: (Batch, 1 * N)
    value_ = register_module("value", torch::nn::Linear(config.nn_hidden2, 1 * num_quantiles_));

    // Advantage stream: (Batch, A * N)
    adv_ = register_module("adv", torch::nn::Linear(config.nn_hidden2, n_actions_ * num_quantiles_));

    InitWeightsLinear(value_, config.nn_init_mode, /*is_relu=*/false);
    InitWeightsLinear(adv_, config.nn_init_mode, /*is_relu=*/false);
}

torch::Tensor QuantileDuelingQNet::Forward(const torch::Tensor& obs)
{
    auto x = obs;
    x = torch::relu(fc1_->forward(x));
    x = torch::relu(fc2_->forward(x));

    // Value: (B, H) -> (B, N) -> mean -> (B, 1)
    auto v_dist = value_->forward(x);
    auto v_mean = v_dist.view({ -1, 1, num_quantiles_ }).mean(2); // (B, 1)

    // Advantage: (B, H) -> (B, A*N) -> (B, A, N) -> mean -> (B, A)
    auto a_dist = adv_->forward(x);
    auto a_mean = a_dist.view({ -1, n_actions_, num_quantiles_ }).mean(2); // (B, A)

    // Dueling scalar calculation
    auto a_mean_mean = a_mean.mean(1, true); // (B, 1)
    auto q = v_mean + (a_mean - a_mean_mean);

    return q;
}

torch::Tensor QuantileDuelingQNet::ForwardQuantiles(const torch::Tensor& obs)
{
    auto x = obs;
    x = torch::relu(fc1_->forward(x));
    x = torch::relu(fc2_->forward(x));

    // Value: (B, H) -> (B, N) -> (B, 1, N)
    auto v = value_->forward(x);
    auto batch_size = v.size(0);
    v = v.view({ batch_size, 1, num_quantiles_ });

    // Advantage: (B, H) -> (B, A * N) -> (B, A, N)
    auto a = adv_->forward(x);
    a = a.view({ batch_size, n_actions_, num_quantiles_ });

    // Mean Advantage across actions: (B, 1, N)
    auto a_mean = a.mean(/*dim=*/1, /*keepdim=*/true);

    // Q(s, a, tau) = V(s, tau) + (A(s, a, tau) - mean(A, dim=1))
    // Broadcasting: (B, 1, N) + (B, A, N) - (B, 1, N) => (B, A, N)
    auto q = v + (a - a_mean);

    return q;
}

std::optional<anet::TensorFunction> QuantileDuelingQNet::GetTensorFunction(const std::string& key, const torch::Device& device, std::shared_ptr<std::shared_mutex> smutex)
{
    // 平均値Q (Scalar)
    if (key == "forward" || key == "forward.q") {
        return [this, device, smutex](const torch::Tensor& t) {
            auto tdev = t.to(device);
            std::shared_lock<std::shared_mutex> lock(*smutex);
            return Forward(tdev);
            };
    }
    // 分布Q (Distribution)
    if (key == "forward.dist") {
        return [this, device, smutex](const torch::Tensor& t) {
            auto tdev = t.to(device);
            std::shared_lock<std::shared_mutex> lock(*smutex);
            return ForwardQuantiles(tdev);
            };
    }
    // Value & Advantage 分離出力 (B, 1+A, N)
    if (key == "forward.va") {
        return [this, device, smutex](const torch::Tensor& t) {
            auto x = t.to(device);
            std::shared_lock<std::shared_mutex> lock(*smutex);

            x = torch::relu(fc1_->forward(x));
            x = torch::relu(fc2_->forward(x));

            // Value: (B, H) -> (B, 1*N) -> (B, 1, N)
            auto v = value_->forward(x);
            auto batch_size = v.size(0);
            v = v.view({ batch_size, 1, num_quantiles_ });

            // Advantage: (B, H) -> (B, A*N) -> (B, A, N)
            auto a = adv_->forward(x);
            a = a.view({ batch_size, n_actions_, num_quantiles_ });

            // A正規化
            auto a_mean = a.mean(/*dim=*/1, /*keepdim=*/true); // (B, 1, N)
            auto a_centered = a - a_mean;

            // Concatenate: (B, 1 + A, N)
            // index 0: Value Distribution
            // index 1..A: Centered Advantage Distribution (実際にQ計算に使われる値)
            auto va = torch::cat({ v, a_centered }, 1);

            ANET_CHECK_SHAPE(va, { ANET_SHAPE_ANY, 1 + n_actions_, num_quantiles_ });

            return va;
            };
    }
    if (key == "forward.v") {
        return [this, device, smutex](const torch::Tensor& t) {
            auto tdev = t.to(device);
            std::shared_lock<std::shared_mutex> lock(*smutex);

            // 分布 Q(s, a, q) を取得
            auto q_dist = ForwardQuantiles(tdev); // (B, A, N)

            // 平均値 Q(s, a) を計算して、Greedy行動 a* を決定
            auto q_mean = q_dist.mean(2); // (B, A)
            auto best_actions = std::get<1>(q_mean.max(1)); // (B)

            // a* に対応する分布を抜き出す
            // (B, A, N) -> (B, 1, N)
            auto idx = best_actions.view({ -1, 1, 1 }).expand({ -1, 1, num_quantiles_ });
            auto v_true_dist = q_dist.gather(1, idx);

            ANET_CHECK_SHAPE(v_true_dist, { ANET_SHAPE_ANY, 1, num_quantiles_ });
            return v_true_dist;
            };
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
    const auto& net = use_target ? target_net_ : policy_net_;
    auto q = net->Forward(obs);
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
    return net->ForwardQuantiles(obs);
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

void RainbowAgent::Learner::SetupOptimizer()
{
    this->optimizer_ = std::make_unique<torch::optim::Adam>(
        agent_.network_->GetPolicyParameters(),
        torch::optim::AdamOptions(agent_.config_.alpha));
}

void RainbowAgent::Learner::SetupReplayBuffer(const EnvSpec& env_spec, seed_t seed)
{
    anet::rl::ReplayBufferConfig rep_config{};
    rep_config.capacity = agent_.config_.replay_capacity;
    rep_config.gamma = agent_.config_.gamma;
    rep_config.n_step = agent_.config_.n_step;
    rep_config.type = agent_.config_.use_n_step ? ReplayBuilderType::NSTEP : ReplayBuilderType::PLAIN;

    if (agent_.config_.use_per) {
        rep_config.sampler_type = ReplaySamplerType::PRIOTIZED;
        rep_config.per_alpha = agent_.config_.per_alpha;
        rep_config.per_initial_priority = agent_.config_.per_initial_priority;
        agent_.vars_->per_beta = agent_.config_.per_beta_start;
    } else {
        rep_config.sampler_type = ReplaySamplerType::UNIFORM;
    }

    anet::rl::ReplayBufferFactory rep_factory(rep_config);
    this->replay_buffer_ = rep_factory.Create(env_spec, torch::kCPU, agent_.batch_size_, seed);
}

bool RainbowAgent::Learner::CanUpdate(step_t update_step) const
{
    /// @todo ReplayBuffer充足チェックにexp_stepを使う？

    // warmup
    if (update_step < agent_.config_.update_warmup_steps * agent_.batch_size_)
        return false;
    // update_interval
    if ((update_step % agent_.config_.update_interval) != 0)
        return false;
    return true;
}

void RainbowAgent::Learner::UpdateEpsilon(step_t learn_step)
{
    ProfileRange r("RainbowAgent::Learner::UpdateEpsilon");

    const auto& config = agent_.config_;
    auto& vars = *agent_.vars_;

    if (learn_step >= config.eps_decay_step) {
        vars.epsilon = config.eps_min;
        return;
    }
    const float t = static_cast<float>(learn_step) / static_cast<float>(config.eps_decay_step);
    vars.epsilon = config.eps_max + t * (config.eps_min - config.eps_max);
}

void RainbowAgent::Learner::UpdateTargetNetwork(step_t step)
{
    ProfileRange r("RainbowAgent::Learner::UpdateTargetNetwork");
    agent_.network_->UpdateTarget(step);
}

void RainbowAgent::Learner::UpdatePerBeta(step_t learn_step)
{
    ProfileRange r("RainbowAgent::Learner::UpdatePerBeta");

    const auto& config = agent_.config_;
    auto& vars = *agent_.vars_;

    if (!config.use_per) return;

    if (learn_step < config.per_beta_step) {
        float progress = static_cast<float>(learn_step) / static_cast<float>(config.per_beta_step);
        vars.per_beta = config.per_beta_start + progress * (config.per_beta_end - config.per_beta_start);
    } else {
        vars.per_beta = config.per_beta_end;
    }
}

std::shared_ptr<const anet::rl::BatchUpdateResult> RainbowAgent::Learner::UpdateFromBatch(
    const StepCounts& counts, const BatchExperience& experiences, const Runner& trainer)
{
    // ReplayBuffer へ push
    replay_buffer_->Push(experiences);

    // Update不可なら空の結果を返す
    if (!CanUpdate(counts.update_step)) {
        return std::make_shared<anet::rl::RainbowAgent::BatchUpdateResult>(0);
    }

    const auto& config = agent_.config_;
    auto& vars = *agent_.vars_;
    const torch::Device& device = agent_.device_;
    const int B = config.replay_batch_size;
    const int S = agent_.state_dim_;
    //const int A = agent_.n_actions_;

    // Sample
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

    // 固有処理呼び出し
    auto result = UpdateFromSamples(samples);

    // 更新後処理
    UpdateTargetNetwork(vars.learn_step);
    UpdateEpsilon(vars.learn_step);
    UpdatePerBeta(vars.learn_step);

    // Post Updates (学習ステップ更新後に行う)
    vars.learn_step++;

    return result;
}

// ======================================================
// RainbowAgent TDLearner
// ======================================================

RainbowAgent::TDLearner::TDLearner(RainbowAgent& agent, const EnvSpec& env_spec, seed_t replay_seed)
    : Learner(agent)
{
    SetupReplayBuffer(env_spec, replay_seed);
    SetupOptimizer();
}

std::shared_ptr<const anet::rl::BatchUpdateResult>
RainbowAgent::TDLearner::UpdateFromSamples(const anet::rl::ExperienceSamples& samples)
{
    ProfileRange r("RainbowAgent::TDLearner::UpdateFromBatch");

    const auto& config = agent_.config_;
    auto& network = *agent_.network_;
    auto& vars = *agent_.vars_;
    const int B = config.replay_batch_size;
    const int S = agent_.state_dim_;
    const int A = agent_.n_actions_;
    const torch::Device& device = agent_.device_;
    const auto& obs = samples.obs;
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


// ======================================================
// RainbowAgent QRLearner (New Implementation)
// ======================================================

RainbowAgent::QRLearner::QRLearner(RainbowAgent& agent, const EnvSpec& env_spec, seed_t replay_seed)
    : Learner(agent)
{
    SetupReplayBuffer(env_spec, replay_seed);
    SetupOptimizer();
}

torch::Tensor RainbowAgent::QRLearner::ComputeQuantileHuberLoss(
    const torch::Tensor& current_dist, const torch::Tensor& target_dist) const
{
    ProfileRange r("RainbowAgent::QRLearner::ComputeQuantileHuberLoss");

    const int N = agent_.config_.num_quantiles;
    const float kappa = agent_.config_.quantile_huber_kappa;
    const auto& device = current_dist.device();
    const auto B = current_dist.size(0);

    // 入力チェック
    ANET_CHECK_SHAPE(current_dist, { B, N });
    ANET_CHECK_SHAPE(target_dist, { B, N });

    // current: (B, N) -> (B, N, 1)
    // target : (B, N) -> (B, 1, N)
    auto cur = current_dist.unsqueeze(2);
    auto tgt = target_dist.unsqueeze(1);
    ANET_CHECK_SHAPE(cur, { B, N, 1 });
    ANET_CHECK_SHAPE(tgt, { B, 1, N });

    // pair-wise差分: (B, N, N)
    auto diff = tgt - cur;
    ANET_CHECK_SHAPE(diff, { B, N, N });

    // 分位数 tau_i = (i + 0.5) / N
    auto tau = torch::arange(0.5f / N, 1.0f, 1.0f / N, device).view({ 1, N, 1 });
    ANET_CHECK_SHAPE(tau, { 1, N, 1 });

    // Huber Loss
    auto abs_diff = diff.abs();
    auto huber = torch::where(abs_diff < kappa, 0.5f * diff.pow(2), kappa * (abs_diff - 0.5f * kappa));
    ANET_CHECK_SHAPE(huber, { B, N, N });

    // Quantile Regression Loss
    // rho_tau(u) = |tau - I(u<0)| * L_k(u)
    auto indicator = (diff.detach() < 0).to(torch::kFloat);
    auto quantile_weight = torch::abs(tau - indicator);	 // Broadcasting Check: (1, N, 1) - (B, N, N) -> (B, N, N)
    ANET_CHECK_SHAPE(quantile_weight, { B, N, N });

    auto loss_per_pair = quantile_weight * huber; // (B, N, N)

    // ターゲット分位数(dim=2)で総和、現在の分位数(dim=1)で平均 "Loss per Batch element"
    auto element_wise_loss = loss_per_pair.sum(2).mean(1); // (B)
    ANET_CHECK_SHAPE(element_wise_loss, { B });

    return element_wise_loss;
}

std::shared_ptr<const anet::rl::BatchUpdateResult>
RainbowAgent::QRLearner::UpdateFromSamples(const anet::rl::ExperienceSamples& samples)
{
    ProfileRange r("RainbowAgent::QRLearner::UpdateFromSamples");

    const auto& config = agent_.config_;
    const int B = config.replay_batch_size;
    const int A = agent_.n_actions_;
    const int N = config.num_quantiles;
    const torch::Device& device = agent_.device_;

    // 入力チェック
    ANET_CHECK_SHAPE(samples.actions, { B });
    ANET_CHECK_SHAPE(samples.target_values, { B });
    ANET_CHECK_SHAPE(samples.next_states.terminals, { B });


    // ------------------------------------------------------------
    // 分布計算
    // ------------------------------------------------------------

    // 現在の分布計算: Z(s, a)、ForwardQuantiles は (B, A, N) を返す
    auto current_dist_all = agent_.network_->ForwardQuantiles(samples.obs, /*use_target=*/false);
    ANET_CHECK_SHAPE(current_dist_all, { B, A, N });

    // 選択された行動の分布を取得: (B, A, N) -> (B, N)
    torch::Tensor idx_actions = samples.actions.view({ B, 1, 1 }).expand({ B, 1, N });
    ANET_CHECK_SHAPE(idx_actions, { B, 1, N });

    auto current_dist = current_dist_all.gather(1, idx_actions).squeeze(1); // (B, N)
    ANET_CHECK_SHAPE(current_dist, { B, N });

    // メトリクス用: 平均値をmax_qとして報告
    auto max_q = current_dist.mean(1).detach(); // (B)
    ANET_CHECK_SHAPE(max_q, { B });

    // メトリクス用: Q Std (分布の標準偏差)、 GPUTensor (Scalar) のまま保持
    auto std_q_tensor = current_dist.std(1).mean().detach();
    ANET_CHECK_SHAPE(std_q_tensor, {});


    // ------------------------------------------------------------
    // ターゲット分布計算: r + gamma * Z(s', a*)
    // ------------------------------------------------------------

    torch::Tensor target_dist;
    {
        torch::NoGradGuard no_grad;

        // 次状態のGreedy行動 a* = argmax E[Z(s', a')]
        torch::Tensor next_actions;
        if (config.use_double_dqn) {
            auto next_q_policy = agent_.network_->Forward(samples.next_states.obs, /*use_target=*/false); // (B, A)
            ANET_CHECK_SHAPE(next_q_policy, { B, A });
            next_actions = std::get<1>(next_q_policy.max(1)); // (B)
        } else {
            auto next_q_target = agent_.network_->Forward(samples.next_states.obs, /*use_target=*/true); // (B, A)
            ANET_CHECK_SHAPE(next_q_target, { B, A });
            next_actions = std::get<1>(next_q_target.max(1)); // (B)
        }
        ANET_CHECK_SHAPE(next_actions, { B });

        // 次状態のターゲット分布: Z_target(s', :)
        auto next_dist_all = agent_.network_->ForwardQuantiles(samples.next_states.obs, /*use_target=*/true); // (B, A, N)
        ANET_CHECK_SHAPE(next_dist_all, { B, A, N });

        // a* に対応する分布を選択: (B, A, N) -> (B, N)
        torch::Tensor idx_next_actions = next_actions.view({ B, 1, 1 }).expand({ B, 1, N });
        ANET_CHECK_SHAPE(idx_next_actions, { B, 1, N });

        auto next_dist = next_dist_all.gather(1, idx_next_actions).squeeze(1); // (B, N)
        ANET_CHECK_SHAPE(next_dist, { B, N });

        // ベルマン作用素適用: T = r + gamma * Z(s', a*)
        auto reward = samples.target_values.view({ B, 1 }); // (B, 1)
        auto not_terminal = (1.0f - samples.next_states.terminals.to(torch::kFloat32)).view({ B, 1 }); // (B, 1)
        ANET_CHECK_SHAPE(reward, { B, 1 });
        ANET_CHECK_SHAPE(not_terminal, { B, 1 });

        // (B, 1) + (B, 1) * (B, N) -> (B, N)
        target_dist = reward + config.gamma * not_terminal * next_dist;
        ANET_CHECK_SHAPE(target_dist, { B, N });
    }

    // target_dist: (B, N) -> mean -> (B)
    auto target_mean = target_dist.mean(1).detach();

    // current_dist.mean(1) は max_q (detach済み) と同じ。 TDLearner: q_sa - td_target に合わせる
    auto td_error_tensor = max_q - target_mean;


    // ------------------------------------------------------------
    // Loss Calculation
    // ------------------------------------------------------------

    // 要素ごとのLoss (B) を取得  ※ここで重い計算を一回だけ行う
    auto element_loss = ComputeQuantileHuberLoss(current_dist, target_dist);
    ANET_CHECK_SHAPE(element_loss, { B });

    // 最適化用Loss(Scalar) ※ PERの重み (IS Weights) を適用
    torch::Tensor weights = config.use_per ? samples.is_weights : torch::ones({ B }, device);
    auto loss = (element_loss * weights).mean();
    ANET_CHECK_SHAPE(loss, {});
    

    // ------------------------------------------------------------
    // Optimize
    // ------------------------------------------------------------
    optimizer_->zero_grad();
    loss.backward();


    // ------------------------------------------------------------
    // 勾配クリッピング
    // ------------------------------------------------------------
    torch::Tensor grad_norm_tensor;
    std::optional<float> grad_norm;
    bool grad_clipped = false;
    if (config.use_grad_clip) {
        double grad_norm_val = torch::nn::utils::clip_grad_norm_(
            agent_.network_->GetPolicyParameters(), config.grad_clip_tau);
        grad_norm = static_cast<float>(grad_norm_val);
        grad_clipped = (grad_norm_val > config.grad_clip_tau);
    } else {
        torch::Tensor total_sq = torch::zeros({ 1 }, loss.options());
        for (auto& p : agent_.network_->GetPolicyParameters()) {
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

    if (config.use_per) {
        torch::NoGradGuard no_grad;

        // 優先度用に要素ごとのLossを再計算（ブロードキャストを利用して全ペア差分を計算）
        auto tgt = target_dist.unsqueeze(1); // (B, 1, N)
        auto cur = current_dist.unsqueeze(2); // (B, N, 1)
        ANET_CHECK_SHAPE(tgt, { B, 1, N });
        ANET_CHECK_SHAPE(cur, { B, N, 1 });

        auto diff = tgt - cur; // (B, N, N)
        ANET_CHECK_SHAPE(diff, { B, N, N });

        // HuberLoss 部分
        auto tau = torch::arange(0.5f / N, 1.0f, 1.0f / N, device).view({ 1, N, 1 });
        auto huber_loss = torch::where(
            diff.abs() < config.quantile_huber_kappa,
            0.5f * diff.pow(2),
            config.quantile_huber_kappa * (diff.abs() - 0.5f * config.quantile_huber_kappa)
        );

        // Quantile Loss 部分: rho_tau(u) = |tau - I(u<0)| * L_k(u)
        auto element_wise_loss = (torch::abs(tau - (diff.detach() < 0).to(torch::kFloat)) * huber_loss).sum(2).mean(1); // (B)
        ANET_CHECK_SHAPE(element_wise_loss, { B });

        // Priority (element_loss を N で割ってスケーリング)
        auto new_priorities = (element_loss / static_cast<float>(N)) + config.per_eps;
        ANET_CHECK_SHAPE(new_priorities, { B });

        // PER clip
        if (config.use_per_prio_clip) {
            metric_per_clipped_count = (new_priorities > config.per_prio_clip_value).sum();
            new_priorities = torch::clamp(new_priorities, 0.0f, config.per_prio_clip_value);
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

    auto result = std::make_shared<anet::rl::RainbowAgent::BatchUpdateResult>(1);
    result->loss = loss;
    result->td_error = td_error_tensor;
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
    result->q_std = std_q_tensor;

    return result;
}
