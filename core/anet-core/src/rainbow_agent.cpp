// dqn_agent.cpp

#include "anet/rainbow_agent.hpp"
#include "dqn_based_agent.hpp"
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


using namespace anet::rl::dqn;
namespace LOG = anet::log;


// ======================================================
// RainbowAgent 本体
// ======================================================

RainbowAgent::RainbowAgent(
    const RainbowAgentConfig& config
    , const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, const torch::Device& device
    , std::shared_ptr<Notifier> notifier
    , std::optional<seed_t> seed)
    : FlatStateAgent(device, notifier, batch_env_spec, env_spec, seed)
    , config_(config)
{
    ANET_LOG_DEBUG("seed=" << GetSeed());

    // ログ：パラメータ記録
    LOG::info() << "RainbowAgent config=" << config_;
    anet::MetricsLogger::Instance()->Log("RainbowAgent", config_.ToJson());
    anet::MetricsLogger::Instance()->Flush();

    //seed
    anet::SeedMaker seed_maker(GetSeed());
    auto replay_seed = seed_maker.MakeNamedSeed("replaybuffer");
    auto action_policy_seed = seed_maker.MakeNamedSeed("action_policy");

    // RuntimeVars生成
    this->vars_ = std::make_unique<dqn::RuntimeVars>();
    this->vars_->epsilon = config_.action_policy.eps_max;

    // QR-DQN設定確認 (use_qr フラグと num_quantiles の整合性)
    bool is_distributional = config_.use_qr;
    if (is_distributional && config_.num_quantiles <= 1) {
        LOG::warn() << "use_qr is true but num_quantiles <= 1. Treating as Scalar DQN.";
        is_distributional = false;
    }

    // NN生成＆初期化
    std::shared_ptr<anet::rl::dqn::QNet> policy_net;
    std::shared_ptr<anet::rl::dqn::QNet> target_net;
    if (config_.use_dueling_net) {
        if (is_distributional) {
            policy_net = std::make_shared<anet::rl::dqn::QuantileDuelingQNet>(config_.qnet, state_dim_, n_actions_);
            target_net = std::make_shared<anet::rl::dqn::QuantileDuelingQNet>(config_.qnet, state_dim_, n_actions_);
        } else {
            policy_net = std::make_shared<anet::rl::dqn::DuelingQNet>(config_.qnet, state_dim_, n_actions_);
            target_net = std::make_shared<anet::rl::dqn::DuelingQNet>(config_.qnet, state_dim_, n_actions_);
        }
    } else {
        if (is_distributional) {
            policy_net = std::make_shared<anet::rl::dqn::QuantilePlainQNet>(config_.qnet, state_dim_, n_actions_);
            target_net = std::make_shared<anet::rl::dqn::QuantilePlainQNet>(config_.qnet, state_dim_, n_actions_);
        } else {
            policy_net = std::make_shared<anet::rl::dqn::PlainQNet>(config_.qnet, state_dim_, n_actions_);
            target_net = std::make_shared<anet::rl::dqn::PlainQNet>(config_.qnet, state_dim_, n_actions_);
        }
    }
    
    // Network生成
    this->network_ = std::make_unique<dqn::Network>(config_.network, device_, policy_net, target_net);

    // ActionPolicy生成
    this->action_policy_ = std::make_unique<dqn::EpsilonGreedyActionPolicy>(config_.action_policy, *network_, *vars_, action_policy_seed);

    // Learner生成
    if (is_distributional) {
        this->learner_ = std::make_unique<dqn::QRLearner>(config_.learner, *network_, *vars_, nullptr, batch_env_spec, env_spec, device_, replay_seed);
        LOG::info() << "Initialized QRLearner (Quantiles=" << config_.num_quantiles << ")";
    } else {
        this->learner_ = std::make_unique<dqn::TDLearner>(config_.learner, *network_, *vars_, nullptr, batch_env_spec, env_spec, device_, replay_seed);
        LOG::info() << "Initialized TDLearner";
    }
}

std::optional<anet::TensorFunction> RainbowAgent::GetTensorFunction(const std::string& key)
{
    auto fn = network_->GetTensorFunction(key, device_);
    if (fn == std::nullopt) return fn;

    auto self = shared_from_this();
    auto network_fn = *fn;

    anet::TensorFunction norm_fn = [self, network_fn](const torch::Tensor& obs) {
        std::shared_lock<std::shared_mutex> lock(*(self->mutex_));
        auto out = network_fn(obs);
        return out;
        };

    return norm_fn;
}

std::optional<float> RainbowAgent::GetScalar(const std::string& key, int64_t index) const
{
    if (key == "epsilon") {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return vars_->epsilon;
    }
    if (key == "per_beta") {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return vars_->per_beta;
    }
    if (key.find("replaybuffer.") == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return learner_->GetScalar(key);
    }

    return std::nullopt;
}

std::optional<torch::Tensor> RainbowAgent::GetTensor(const std::string& key, int64_t index) const
{
    if (key.find("replaybuffer.") == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return learner_->GetTensor(key);
    }

    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>> RainbowAgent::GetTensorVector(const std::string& key, int64_t index) const
{
    if (key.find("replaybuffer.") == 0) {
        std::shared_lock<std::shared_mutex> lock(*mutex_);
        return learner_->GetTensorVector(key);
    }

    return std::nullopt;
}

anet::rl::BatchActionInfo RainbowAgent::MakeAction(const StepCounts& step, const BatchState& state, RunMode runmode) const
{
    ProfileRange r1("RainbowAgent::MakeAction");
    ANET_ASSERT_SHAPE(state.obs, { ANET_SHAPE_ANY, state_dim_ });

    // 共有ロック＆Grad抑止
    std::shared_lock<std::shared_mutex> lock(*mutex_);
    torch::NoGradGuard ng;

    // Flatなobsを生成
    auto flat_obs = state.To(device_).Flatten().obs;

    // 行動選択
    auto greedy_only = anet::rl::IsEval(runmode);
    auto use_target = (runmode == anet::rl::RunMode::Eval1);
    auto act_info = this->action_policy_->SelectAction(flat_obs, greedy_only, use_target);

    // ActionInfoを返す
    return act_info;
}

anet::rl::BatchUpdateResultList
RainbowAgent::UpdateFromBatch(const StepCounts& counts, const anet::rl::BatchExperience& batch_exp, const anet::rl::Runner& runner)
{
    ProfileRange r1("RainbowAgent::UpdateFromBatch");

    anet::rl::BatchUpdateResultList result_list;
    {
        // 排他ロック
        std::unique_lock<std::shared_mutex> lock(*mutex_);

        // Update実行
        auto result = this->learner_->UpdateFromBatch(counts, batch_exp, runner);
        result_list = std::move(result);

        // Update後処理
        action_policy_->OnLearn(counts);
    }

    // LearnEvent通知
    if (notifier_ != nullptr) {
        for (auto result : result_list) {
            anet::rl::LearnEvent event{ batch_exp, runner, counts, shared_from_this(), result_list };
            notifier_->Notify(event);
        }
    }

    // BatchUpdateResultListを返す
    return result_list;
}

// ======================================================
// RainbowAgentFactory
// ======================================================

std::shared_ptr<anet::rl::Agent> RainbowAgentFactory::CreateAgent(
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

