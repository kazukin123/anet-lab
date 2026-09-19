// muzero_proto_agent.cpp

#include "anet/muzero_proto_agent.hpp"
#include "anet/log.hpp"
#include "anet/metrics_logger.hpp"
#include "muzero_based_agent.hpp"
#include "muzero_rb.hpp"

using namespace anet::rl::muzero_proto;
namespace LOG = anet::log;

namespace {

void LogNetworkGraphViz(const std::string& tag_prefix, const anet::nn::Network& network, const anet::nn::NetworkGraphVizConfig& config)
{
    auto structure_view = network.MakeGraphViz(anet::nn::NetworkGraphVizConfig{});
    anet::MetricsLogger::Instance()->Log(tag_prefix + ".structure", *structure_view);
    auto detail_view = network.MakeGraphViz(config);
    anet::MetricsLogger::Instance()->Log(tag_prefix + ".detail", *detail_view);
}

} // namespace


// ======================================================
// MuZeroAgent
// ======================================================

MuZeroAgent::MuZeroAgent(
    const MuZeroAgentConfig& config, const anet::ConfigData& config_data,
    const anet::rl::BatchEnvSpec& batch_env_spec, const anet::rl::EnvSpec& env_spec, const torch::Device device, std::optional<anet::seed_t> seed)
    : AgentBase(device, batch_env_spec, env_spec, seed) // 基底クラス初期化
    , config_(config), env_spec_(env_spec), batch_env_spec_(batch_env_spec), device_(device)
{
    ANET_LOG_DEBUG("seed=" << GetSeed());

    // ログ：パラメータ記録
    LOG::info() << "MuZeroAgent config=" << config_.ToString();
    anet::MetricsLogger::Instance()->Log(config_);

    // seed
    anet::SeedMaker seed_maker(GetSeed());
    auto rb_seed = seed_maker.MakeNamedSeed("ReplayBuffer");

    // MuZeroNetworkModel生成
    model_ = std::make_shared<MuZeroNetworkModel>(config_.model, config_data, env_spec_.state_spec, env_spec_.action_spec, GetSeed());
    model_->To(device_);
    model_->GetSuite()->eval();
    {
        auto suite = model_->GetSuite();
        LogNetworkGraphViz("net.rep", *suite->GetRepresentationNet(), config_.nn_viz);
        LogNetworkGraphViz("net.dyn", *suite->GetDynamicsNet(), config_.nn_viz);
        LogNetworkGraphViz("net.pred", *suite->GetPredictionNet(), config_.nn_viz);
    }

    // ReplayBuffer生成
    replay_buffer_ = std::make_shared<MuZeroReplayBuffer>(
        config_.buffer, batch_env_spec_.num_envs, env_spec_.action_spec.GetNumActions(), rb_seed);

    // メトリクス

}

std::shared_ptr<anet::rl::Actor> MuZeroAgent::CreateActor(const ActorRequest& request) const
{
    ANET_PROFILE_FUNC();
    env_spec_.CheckSameStateActionSpec(request.env_spec);
    const auto& cfg = FindActorConfig(config_.actor, request.actor_key);
    if (cfg.clone_model) ANET_SYSTEM_ERROR("MuZeroActor clone_model=true is unsupported; expected false.");
    ValidateActorDevice(false, request.device);
    return std::make_shared<MuZeroActor>(cfg, config_.mcts, mutex_, model_,
        request.env_spec.action_spec, request.device, request.seed);
}

std::shared_ptr<anet::rl::Learner> MuZeroAgent::CreateLearner()
{
    // Learnerを生成
    return std::make_shared<MuZeroLearner>(config_.learner, mutex_, model_, replay_buffer_, device_);
}

std::optional<float> MuZeroAgent::GetScalar(const std::string& key, int64_t index) const
{

    return std::nullopt;
}

// ======================================================
// MuZeroAgentFactory
// ======================================================

std::shared_ptr<anet::rl::Agent> MuZeroAgentFactory::CreateAgent(
    const anet::rl::EnvSpec& env_spec, const anet::rl::BatchEnvSpec& batch_env_spec, const torch::Device& device,
    const anet::ConfigData& config_data, std::shared_ptr<anet::rl::Notifier> notifier, std::optional<anet::seed_t> seed) const
{
    return std::make_shared<MuZeroAgent>(
        MuZeroAgentConfig(config_data), config_data, batch_env_spec, env_spec, device, seed);
}
