// dqn_agent.cpp

#include "rainbow_agent.hpp"
#include "dqn_assembler.hpp"
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
    , const anet::nn::NetworkConfig& net_config
    , const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, const torch::Device& device
    , std::optional<seed_t> seed)
    : AgentBase(device, batch_env_spec, env_spec, seed)
    , config_(config)
{
    config_.ValidateQuantileConfig();
    config_.ValidatePolicyConfig();
    ANET_LOG_DEBUG("seed=" << GetSeed());

    // ログ：パラメータ記録
    LOG::info() << "RainbowAgent config=" << config_.ToString();
    anet::MetricsLogger::Instance()->Log(config_);

    //seed
    anet::SeedMaker seed_maker(GetSeed());
    auto replay_seed = seed_maker.MakeNamedSeed("replaybuffer");
    auto network_seed = seed_maker.MakeNamedSeed("network");
    //auto action_policy_seed = seed_maker.MakeNamedSeed("action_policy");

    // RuntimeVars生成
    this->vars_ = std::make_unique<RuntimeVars>();

    const bool is_distributional = config_.use_qr;

    // ------------------------------------------------------------
	// HeadFactory の準備
	// ------------------------------------------------------------
	
    auto head_factory = DqnComponentAssembler::CreateHeadFactory(
        config_.use_qr ? ActionPolicyConfig::kQuantileModeQr : ActionPolicyConfig::kQuantileModeNone,
        config_.use_dueling_net, n_actions_, config_.num_quantiles, config_.head.init);

    // ------------------------------------------------------------
    // Network構築 (Builder)
    // ------------------------------------------------------------

    // NetworkModel生成
    this->model_ = std::make_unique<NetworkModel>(
        config_.model, device_, net_config, env_spec.state_spec.obs_spec, n_actions_, head_factory,
        is_distributional, network_seed);

    // ActionPolicy生成

    ActionPolicyConfig greedy_cfg;
    greedy_cfg.policy_type = kActionPolicyTypeStr_Greedy;
    this->target_policy_ = DqnComponentAssembler::CreateActionPolicy(
        greedy_cfg, is_distributional ? ActionPolicyConfig::kQuantileModeQr : ActionPolicyConfig::kQuantileModeNone,
        false, num_envs_, device_);

    learner_ = DqnComponentAssembler::CreateLearner(
        is_distributional ? ActionPolicyConfig::kQuantileModeQr : ActionPolicyConfig::kQuantileModeNone,
        config_.num_quantiles, config_.learner, *model_, *vars_, batch_env_spec, env_spec, device_, replay_seed, GetSeed(), target_policy_);
}

std::optional<anet::TensorDictFunction> RainbowAgent::GetTensorDictFunction(const std::string& key)
{
    auto fn = model_->GetTensorDictFunction(key, device_);
    if (fn == std::nullopt) return std::nullopt;

    auto self = shared_from_this();
    auto network_fn = *fn;

    anet::TensorDictFunction wrapped_fn = [self, network_fn](const anet::TensorDict& obs) {
        std::shared_lock<std::shared_mutex> lock(*(self->mutex_));
        torch::NoGradGuard grad_guard;
        auto out = network_fn(obs.To(self->device_));
        return out;
        };

    return wrapped_fn;
}

std::optional<float> RainbowAgent::GetScalar(const std::string& key, int64_t index) const
{
    return DqnComponentAssembler::GetScalar(key, index, *learner_, *vars_, *target_policy_, *mutex_);
}

std::optional<torch::Tensor> RainbowAgent::GetTensor(const std::string& key, int64_t index) const
{
    return DqnComponentAssembler::GetTensor(key, *learner_, *mutex_);
}

std::optional<std::vector<torch::Tensor>> RainbowAgent::GetTensorVector(const std::string& key, int64_t index) const
{
    return DqnComponentAssembler::GetTensorVector(key, *learner_, *mutex_);
}

std::shared_ptr<anet::rl::Actor> RainbowAgent::CreateActor(const ActorRequest& request) const
{
    ANET_PROFILE_FUNC();
    env_spec_.CheckSameStateActionSpec(request.env_spec);
    const auto& cfg = FindActorConfig(config_.actor, request.actor_key);
    ValidateActorDevice(cfg.clone_model, request.device);
    const auto quantile_mode = config_.use_qr
        ? ActionPolicyConfig::kQuantileModeQr : ActionPolicyConfig::kQuantileModeNone;
    auto policy = DqnComponentAssembler::CreateActionPolicy(cfg.policy, quantile_mode,
        cfg.policy.spatial.enabled, request.batch_env_spec.num_envs, request.device);
    return DqnComponentAssembler::CreateActor(cfg, config_.learner, request, *model_, mutex_, policy,
        anet::nn::NetworkForwardConfig::GetDefault(), nullptr, nullptr, false,
        ActorQHintConfig{.munchausen = config_.learner.munchausen, .gamma = config_.learner.gamma});
}

std::unique_ptr<anet::rl::AgentStepDiagnosticsProvider> RainbowAgent::CreateStepDiagnosticsProvider(const AgentStepDiagnosticsRequest& request) const
{
    // Rainbowの学習経路は報酬scalerを持たないので、診断も同じ無変換とする。
    return anet::rl::dqn::CreateStepDiagnosticsProvider(config_.learner, request.num_lanes,
        [](const torch::Tensor& reward) { return reward; });
}

std::shared_ptr<anet::rl::Learner> RainbowAgent::CreateLearner()
{
    return this->shared_from_this();
}

anet::rl::BatchUpdateResultList
RainbowAgent::UpdateFromBatch(const StepCounts& counts, const anet::rl::BatchExperience& batch_exp)
{
    ANET_PROFILE_FUNC();

    anet::rl::BatchUpdateResultList result_list;
    {
        // 排他ロック
        std::unique_lock<std::shared_mutex> lock(*mutex_);

        // Update実行
        auto result = this->learner_->UpdateFromBatch(counts, batch_exp);
        result_list = std::move(result);

        // Update後処理
        target_policy_->UpdateSchedule(counts);
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
    anet::nn::NetworkConfig net_config(config_data, GetTargetAgentClassId() + ".net");
    auto agent = std::make_shared<RainbowAgent>(config, net_config, batch_env_spec, env_spec, device, seed);
    return agent;

    /// @todo 引数の順番を統一
}
