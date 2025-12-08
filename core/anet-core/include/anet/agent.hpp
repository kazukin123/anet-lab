// anet/agent.hpp
#pragma once

#include <memory>
#include <optional>
#include <mutex>
#include "anet/rl.hpp"
#include "anet/config.hpp"

namespace anet::rl {

    // 環境のステップに同期して更新する Agent 基底クラス
    template<typename ConfigT>
    class StepBasedAgent : public Agent, public anet::RandomHolder {
    public:
        StepBasedAgent(ConfigT config, torch::Device device,
            std::shared_ptr<anet::rl::Notifier> notifier,
            std::optional<seed_t> seed = std::nullopt)
            : RandomHolder(seed), config_(config), notifier_(notifier),device_(device)
        {
        }

        virtual ~StepBasedAgent() = default;
    protected:
        mutable std::shared_mutex mutex_;
    protected:
        // Resource（Agentが管理すべき領域）
        ConfigT config_;
        torch::Device device_;
        std::shared_ptr<anet::rl::Notifier> notifier_;

    };

    // =============================================================

    struct DefaultAgentFactoryConfig : public anet::Config
    {
        std::string class_id;
        int device_type = 1;   ///< 0=cpu 1=cuda
        int device_index = -1; ///< GPU index -1=current device

        DefaultAgentFactoryConfig(const ConfigData& config_data = EmptyConfigData)
            : anet::Config(config_data, "agent")
        {
            ANET_READ_CONFIG(config_data, class_id);
            ANET_READ_CONFIG(config_data, device_type);
            ANET_READ_CONFIG(config_data, device_index);
        }
    };

    class DefaultAgentFactory {
    public:
        DefaultAgentFactory(
            const DefaultAgentFactoryConfig& config,
            const EnvSpec& env_spec,
			const BatchEnvSpec& batch_env_spec,
            const anet::ConfigData& config_data = anet::EmptyConfigData,
            std::optional<seed_t> seed = std::nullopt);

        std::shared_ptr<Agent> CreateAgent(std::shared_ptr<anet::rl::Notifier> notifier = nullptr) const;
        torch::Device GetDevice() const { return device_; }
    private:
        DefaultAgentFactoryConfig config_;
        anet::ConfigData config_data_;
        EnvSpec env_spec_;
		BatchEnvSpec batch_env_spec_;
        std::optional<seed_t> seed_;
        torch::Device device_;
    };

    // =============================================================

    class AgentRepository {
    public:
        static AgentRepository& Instance() {
            static AgentRepository inst;
            return inst;
        }

        void Register(std::shared_ptr<AgentFactory> factory);
        std::shared_ptr<AgentFactory> GetAgentFactory(const std::string& id) const;
    private:
        AgentRepository() = default;

        mutable std::mutex mtx_;
        std::unordered_map<std::string, std::shared_ptr<AgentFactory>> factories_;
    };

    template<typename T, class... Args>
    inline void RegisterAgentFactory(Args&&... args)
    {
        auto factory = std::make_shared<T>(std::forward<Args>(args)...);
        AgentRepository::Instance().Register(factory);
    }

} // namespace anet::rl

  // =============================================================

#define ANET_REGISTER_AGENT_FACTORY(FactoryType) \
    namespace { \
        struct FactoryType##AutoRegister { \
            FactoryType##AutoRegister() { \
                anet::rl::RegistAgentFactory(std::make_shared<FactoryType>()); \
            } \
        }; \
        static FactoryType##AutoRegister global_##FactoryType##_auto_register; \
    }
