// muzero_proto_agent.cpp

#include "anet/muzero_proto_agent.hpp"
#include "anet/log.hpp"
#include "anet/metrics_logger.hpp"
#include "muzero_based_agent.hpp"
#include "muzero_rb.hpp"

using namespace anet::rl::muzero_proto;
namespace LOG = anet::log;


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
    model_ = std::make_shared<MuZeroNetworkModel>(config_.model, config_data, env_spec_.state_spec, env_spec_.action_spec);
    model_->To(device_);

    // ReplayBuffer生成
    replay_buffer_ = std::make_shared<MuZeroReplayBuffer>(
        config_.buffer, batch_env_spec_.num_envs, env_spec_.action_spec.GetNumActions(), rb_seed);

    // メトリクス
    latest_tau_ = std::make_shared<float>(config_.actor.temp_start);
}

std::shared_ptr<anet::rl::Actor> MuZeroAgent::CreateActor(
    const anet::rl::BatchEnvSpec& batch_env_spec, anet::rl::RunMode run_mode, bool clone_model, std::optional<torch::Device> device) const
{
    // RunMode毎の再現性のあるseedを取得
    auto rnd = GetRandomGenerator(run_mode);
    auto seed = rnd->RandUint64();

    // Actorを生成
    return std::make_shared<MuZeroActor>(
        config_.actor, config_.mcts, mutex_, latest_tau_, model_, env_spec_.action_spec, run_mode, device_, seed);
}

std::shared_ptr<anet::rl::Learner> MuZeroAgent::CreateLearner()
{
    // Learnerを生成
    return std::make_shared<MuZeroLearner>(config_.learner, mutex_, model_, replay_buffer_, device_);
}

std::optional<float> MuZeroAgent::GetScalar(const std::string& key, int64_t index) const
{
    if (key == "tau") {
        return *latest_tau_;
    }

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
