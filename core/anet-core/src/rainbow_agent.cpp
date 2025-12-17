// dqn_agent.cpp

#include "anet/rainbow_agent.hpp"
#include "rainbow_agent_impl.hpp"
#include <memory>
#include <torch/torch.h>
#include <tuple>
#include "anet/str_util.hpp"
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
    : FlatStateAgent(config, device, notifier, batch_env_spec, env_spec, seed)
{
    ANET_LOG_DEBUG("seed=" << GetSeed());

    // ログ：パラメータ記録
    anet::log::info() << "RainbowAgent config=" << config_;
    anet::MetricsLogger::Instance()->LogJson("RainbowAgent", config_.ToJson());
    anet::MetricsLogger::Instance()->Flush();

    //seed
    anet::SeedMaker seed_maker(GetSeed());
    auto replay_seed = seed_maker.MakeNamedSeed("replaybuffer");
    auto action_policy_seed = seed_maker.MakeNamedSeed("action_policy");

    // RuntimeVars生成
    this->vars_ = std::make_unique<RuntimeVars>();

    // NN生成＆初期化
    if (config_.use_dueling_net) {
        auto policy_net = std::make_shared<DuelingQNet>(config_, state_dim_, n_actions_);
        auto target_qnet = std::make_shared<DuelingQNet>(config_, state_dim_, n_actions_);
        this->network_ = std::make_unique<RainbowAgent::Network>(config_, device_, policy_net, target_qnet);
    } else {
        auto policy_net = std::make_shared<PlainQNet>(config_, state_dim_, n_actions_);
        auto target_qnet= std::make_shared<PlainQNet>(config_, state_dim_, n_actions_);
        this->network_ = std::make_unique<RainbowAgent::Network>(config_, device_, policy_net, target_qnet);
    }

    // ActionPolicy生成
    this->action_policy_ = std::make_unique<RainbowAgent::ActionPolicy>(*network_, *vars_, action_policy_seed);

    // Learner生成    /// @todo Learner暫定コンストラクタ
    this->learner_ = std::make_unique<RainbowAgent::TDLearner>(*this, env_spec, replay_seed);
}


std::optional<anet::TensorFunction> RainbowAgent::GetTensorFunction(const std::string& key)
{
    return network_->GetTensorFunction(key, device_, mutex_);
}

std::optional<float> RainbowAgent::GetScalar(const std::string& key, int index) const
{
    if (key == "epsilon") {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return vars_->epsilon;
    }
    if (key.find("replaybuffer.") == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return learner_->GetScalar(key);
    }

    return std::nullopt;
}

std::optional<torch::Tensor> RainbowAgent::GetTensor(const std::string& key, int index) const
{
    if (key.find("replaybuffer.") == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return learner_->GetTensor(key);
    }

    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>> RainbowAgent::GetTensorVector(const std::string& key, int index) const
{
    if (key.find("replaybuffer.") == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return learner_->GetTensorVector(key);
    }

    return std::nullopt;
}

BatchActionInfo RainbowAgent::MakeAction(const StepCounts& step, const BatchState& state, RunMode runmode) const
{
    ProfileRange r1("RainbowAgent::MakeAction");
    ANET_CHECK_SHAPE(state.obs, { ANET_SHAPE_ANY, state_dim_ });

    // 共有ロック＆Grad抑止
    std::shared_lock<std::shared_mutex> lock(*mutex_);
    torch::NoGradGuard ng;

    // Flatなobsを生成
    auto flat_state = state.Flatten();
    auto flat_obs = flat_state.obs.to(device_);

    // 行動選択
    auto greedy_only = anet::rl::IsEval(runmode);
    auto use_target = (runmode == anet::rl::RunMode::Eval1);
    auto act_info = this->action_policy_->SelectAction(flat_obs, greedy_only, use_target);

    // ActionInfoを返す
    return act_info;
}

std::shared_ptr<const anet::rl::BatchUpdateResult>
RainbowAgent::UpdateFromBatch(const StepCounts& counts, const anet::rl::BatchExperience& batch_exp, const anet::rl::Runner& runner)
{
    ProfileRange r1("RainbowAgent::UpdateFromBatch");

    std::shared_ptr<const anet::rl::BatchUpdateResult> update_result;

    if (true) {
        // 排他ロック
        std::unique_lock<std::shared_mutex> lock(*mutex_);
        // Update実行
        update_result = this->learner_->UpdateFromBatch(counts, batch_exp, runner);
    } else {
        update_result = std::make_shared<anet::rl::RainbowAgent::BatchUpdateResult>(0);
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

