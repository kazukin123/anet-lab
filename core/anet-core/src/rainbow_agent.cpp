// dqn_agent.cpp

#include "anet/rainbow_agent.hpp"
#include "rainbow_agent_impl.hpp"
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
namespace LOG = anet::log;


// ======================================================
// RainbowAgent 本体
// ======================================================

RainbowAgent::RainbowAgent(
    const RainbowAgentConfig& config
    , const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, const torch::Device& device
    , std::shared_ptr<Notifier> notifier
    , std::optional<seed_t> seed)
    : StepBasedAgent(config, device, notifier, seed)
    , state_dim_(env_spec.state_spec.CalcFlattenSize())
    , n_actions_(env_spec.action_spec.GetNumActions())
    , batch_size_(batch_env_spec.batch_size)
{
    ANET_LOG_DEBUG("seed=" << GetSeed());

    //seed
    anet::SeedMaker seed_maker(GetSeed());
    auto replay_seed = seed_maker.MakeNamedSeed("replaybuffer");
    auto action_policy_seed = seed_maker.MakeNamedSeed("action_policy");

    // RuntimeVars生成
    this->vars_ = std::make_unique<RuntimeVars>();

    // NN生成＆初期化
    auto qnet_policy = std::make_shared<PlainQNet>(config_, state_dim_, n_actions_);
    auto qnet_target = std::make_shared<PlainQNet>(config_, state_dim_, n_actions_);
    qnet_policy->to(device_);
    qnet_target->to(device_);
    qnet_target->eval();
    this->network_ = std::make_unique<RainbowAgent::Network>(config_, qnet_policy, qnet_target);

    // ActionPolicy生成
    this->action_policy_ = std::make_unique<RainbowAgent::ActionPolicy>(*network_, *vars_, action_policy_seed);

    // Learner生成    /// @todo Learner暫定コンストラクタ
    this->learner_ = std::make_unique<RainbowAgent::TDLearner>(*this, env_spec, replay_seed);
}


std::optional<anet::TensorFunction> RainbowAgent::GetTensorFunction(const std::string& key) const
{
    if (key == "policy_net.forward") {
        anet::TensorFunction fn = [this](const torch::Tensor& t) {
            auto tdev = t.to(device_);
            std::shared_lock<std::shared_mutex> lock(mutex_);
            return network_->ForwardRaw(t, false).to(tdev);
            };
        return fn;
    }
    if (key == "target_net.forward") {
        anet::TensorFunction fn = [this](const torch::Tensor& t) {
            auto tdev = t.to(device_);
            std::shared_lock<std::shared_mutex> lock(mutex_);
            return network_->ForwardRaw(t, true).to(tdev);
            };
        return fn;
    }
    if (key == "q_pair.forward") {
        anet::TensorFunction fn = [this](const torch::Tensor& t) {
            auto tdev = t.to(device_);
            std::shared_lock<std::shared_mutex> lock(mutex_);
            auto q_online = network_->ForwardRaw(t, false).to(tdev);    // [N, A]
            auto q_target = network_->ForwardRaw(t, true).to(tdev);     // [N, A]
            return torch::cat({ q_online, q_target }, 1);               // [N, 2*A]
            };
        return fn;
    }

    return std::nullopt;
}

std::optional<float> RainbowAgent::GetScalar(const std::string& key, int index) const
{
    if (key == "epsilon") {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return vars_->epsilon;
    }
    if (key.find("replaybuffer.") == 0) {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return learner_->GetScalar(key);
    }

    return std::nullopt;
}

std::optional<torch::Tensor> RainbowAgent::GetTensor(const std::string& key, int index) const
{
    if (key.find("replaybuffer.") == 0) {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return learner_->GetTensor(key);
    }

    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>> RainbowAgent::GetTensorVector(const std::string& key, int index) const
{
    if (key.find("replaybuffer.") == 0) {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return learner_->GetTensorVector(key);
    }

    return std::nullopt;
}

BatchActionInfo RainbowAgent::MakeAction(const StepCounts& step, const BatchState& state, RunMode runmode) const
{
    ProfileRange r1("RainbowAgent::MakeAction");
    ANET_CHECK_SHAPE(state.obs, { ANET_SHAPE_ANY, state_dim_ });

    // 共有ロック＆Grad抑止
    std::shared_lock<std::shared_mutex> lock(mutex_);
    torch::NoGradGuard ng;

    // Flatなobsを生成
    auto flat_state = state.Flatten();
    auto flat_obs = flat_state.obs.to(device_);

    // 行動選択
    auto greedy_only = anet::rl::IsEval(runmode);
    auto act_info = this->action_policy_->SelectAction(flat_obs, greedy_only);

    // ActionInfoを返す
    return act_info;
}

std::shared_ptr<const anet::rl::BatchUpdateResult>
RainbowAgent::UpdateFromBatch(const StepCounts& counts, const anet::rl::BatchExperience& batch_exp, const anet::rl::Runner& runner)
{
    ProfileRange r1("RainbowAgent::UpdateFromBatch");

    std::shared_ptr<const anet::rl::BatchUpdateResult> update_result;

    {
        // 共有ロック
        std::unique_lock<std::shared_mutex> lock(mutex_);

        // Update実行
        update_result = this->learner_->UpdateFromBatch(counts, batch_exp, runner);
    }

    // LearnEvent通知
    if (update_result->GetLearnStepDiff() > 0 && notifier_ != nullptr) {
        anet::rl::LearnEvent event{ batch_exp, runner, counts, shared_from_this(), update_result };
        notifier_->Notify(event);
    }

    // BatchUpdateResultを返す
    return update_result;
}

// ======================================================
// RainbowAgentFactory
// ======================================================

std::shared_ptr<Agent> RainbowAgentFactory::CreateAgent(
    const EnvSpec& env_spec, const BatchEnvSpec& batch_env_spec,
    const torch::Device& device, const anet::ConfigData& config_data,
    std::shared_ptr<anet::rl::Notifier> notifier, std::optional<anet::seed_t> seed) const
{
    RainbowAgentConfig config(config_data);
    auto agent = std::make_shared<RainbowAgent>(config, batch_env_spec, env_spec, device, notifier, seed);
    return agent;

    /// @todo 引数の順番を統一
}

//ANET_REGISTER_AGENT_FACTORY(RainbowAgentFactory);

