#include "anet/agent.hpp"
#include "anet/config.hpp"
#include "anet/tensor_util.hpp"

using namespace anet::rl;


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
    
    //const EnvSpec& env_spec,
    //    const BatchEnvSpec& batch_env_spec,
    //    const torch::Device& device,
    //    const anet::ConfigData& config_data = EmptyConfigData,
    //    std::shared_ptr<anet::rl::Notifier> notifier = nullptr,
    //    std::optional<anet::seed_t> seed = std::nullopt

    return agent;
}
