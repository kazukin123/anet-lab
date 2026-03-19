#include "anet/agent.hpp"
#include "anet/config.hpp"
#include "anet/tensor_util.hpp"
#include "anet/tensor_check.hpp"
#include "anet/profile.hpp"

using namespace anet::rl;


// =============================================================
// AgentBase
// =============================================================

AgentBase::AgentBase(torch::Device device,
    std::shared_ptr<anet::rl::Notifier> notifier,
    const BatchEnvSpec& batch_env_spec,
    const EnvSpec& env_spec,
    std::optional<seed_t> seed)
    : RandomHolder(seed), notifier_(notifier), device_(device)
    , state_dim_(env_spec.state_spec.CalcFlattenDim())
    , n_actions_(env_spec.action_spec.GetNumActions())
    , batch_size_(batch_env_spec.batch_size)
{
    mutex_ = std::make_shared<std::shared_mutex>();

    //  Actor(ActionContext)群のための専用ドメインシードを生成
    anet::SeedMaker seed_maker(this->GetSeed());
    action_context_seed_ = seed_maker.MakeNamedSeed("action_context");
}


std::shared_ptr<ActionContext> AgentBase::CreateActionContext(
    const BatchEnvSpec& batch_env_spec, RunMode run_mode, std::optional<torch::Device> device) const
{
    // 専用のRNGから1つシードを払い出してコンテキストに渡す
    auto rng = GetRandomGenerator(run_mode);
    seed_t ctx_seed = rng->RandUint64();
    return std::make_shared<DefaultActionContext>(run_mode, ctx_seed);
}

std::shared_ptr<anet::RandomGenerator> AgentBase::GetRandomGenerator(RunMode mode) const
{
    std::lock_guard<std::mutex> lock(rng_mutex_);

    auto it = run_mode_rngs_.find(mode);
    if (it != run_mode_rngs_.end()) {
        return it->second;
    }

    // ActionContext専用シードからRandomGeneratorを作る
    seed_t mode_base_seed = anet::splitmix64(action_context_seed_ ^ static_cast<uint64_t>(mode));
    auto new_rng = std::make_shared<anet::RandomGenerator>(mode_base_seed);
    run_mode_rngs_[mode] = new_rng;
    return new_rng;
}


// =============================================================
// AgentRepository
// =============================================================

void AgentRepository::Register(std::shared_ptr<AgentFactory> factory)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto class_id = factory->GetTargetAgentClassId();
    factories_[class_id] = factory;
}

std::shared_ptr<AgentFactory> AgentRepository::GetAgentFactory(const std::string& id) const
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = factories_.find(id);
    if (it == factories_.end()) return nullptr;
    return it->second;
}


// =============================================================
// DefaultAgentFactory
// =============================================================

DefaultAgentFactory::DefaultAgentFactory(
    const DefaultAgentFactoryConfig& config,
    const EnvSpec& env_spec,
    const BatchEnvSpec& batch_env_spec,
    const ConfigData& config_data,
    std::optional<seed_t> seed)
    : config_(config)
    , env_spec_(env_spec)
    , batch_env_spec_(batch_env_spec)
    , config_data_(config_data)
    , seed_(seed)
    , device_(anet::MakeDevice(config_.device_type, config_.device_index))
{
    ;
}

std::shared_ptr<Agent> DefaultAgentFactory::CreateAgent(
    std::shared_ptr<anet::rl::Notifier> notifier) const
{
    auto factory = AgentRepository::Instance().GetAgentFactory(config_.class_id);
    if (factory == nullptr)
        return nullptr;

    auto agent = factory->CreateAgent(
        env_spec_, batch_env_spec_, device_, config_data_, notifier, seed_);
    
    return agent;
}
