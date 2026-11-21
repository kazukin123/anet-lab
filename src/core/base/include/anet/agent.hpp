// anet/agent.hpp
#pragma once

#include <memory>
#include <limits>
#include <cmath>
#include <optional>
#include <mutex>
#include "anet/rl.hpp"
#include "anet/config.hpp"
#include "anet/scaler.hpp"
#include "anet/nn.hpp"
#include "anet/schedule.hpp"

namespace anet::nn {
    class NetworkForwardConfig;
}

namespace anet::rl {


    // ----------------------------------------------------------------------
    // AgentBase
    // ----------------------------------------------------------------------

    // 環境のステップに同期して更新する Agent 基底クラス
    class AgentBase : public Agent, public ModuleBase, public anet::RandomHolder {
    public:
        AgentBase(torch::Device device,
            const BatchEnvSpec& batch_env_spec,
            const EnvSpec& env_spec,
            std::optional<seed_t> seed = std::nullopt);

        torch::Device GetDevice() const override { return device_; }
        virtual ~AgentBase() = default;
    protected:
        void ValidateActorDevice(bool clone_model, const torch::Device& actor_device) const;
        template<typename T>
        const T& FindActorConfig(const std::map<std::string, T>& catalog, const std::string& key) const
        {
            const auto it = catalog.find(key);
            if (it == catalog.end()) {
                std::string available;
                for (const auto& [name, config] : catalog) {
                    if (!available.empty()) available += ", ";
                    available += name;
                }
                ANET_SYSTEM_ERROR("Undefined Actor key='" << key << "'. Defined keys=[" << available
                    << "]. Select an existing catalog entry with run.train.actor or run.eval.[tag].actor.");
            }
            return it->second;
        }
    protected:
        std::shared_ptr<std::shared_mutex> mutex_;
        const torch::Device device_;
        const EnvSpec env_spec_;
        int n_actions_;
        int num_envs_;
    };


    // =============================================================
    // DQN
    // =============================================================


    // =============================================================
    // AgentRepository
    // =============================================================

    class AgentRepository {
    public:
        static AgentRepository& Instance() {
            static AgentRepository inst;
            return inst;
        }

        /// @todo AgentFactory → AgentCreator

        void Register(std::shared_ptr<AgentFactory> factory);
        std::shared_ptr<AgentFactory> GetAgentFactory(const std::string& id) const;
        void MarkUnusedConfigInventory(const ConfigData& data, const std::string& active_class_id) const;
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


    // =============================================================
    // DefaultAgentFactory
    // =============================================================

    struct DefaultAgentFactoryConfig : public anet::Config
    {
        std::string class_id;
        std::string device = "auto";

        DefaultAgentFactoryConfig(const ConfigData& config_data = EmptyConfigData)
            : anet::Config(config_data, "agent")
        {
            ANET_READ_CONFIG(config_data, class_id);
            ANET_READ_CONFIG(config_data, device);
        }

        void RecordEffectiveDevice(const torch::Device& value) { my_config_json_["effective_device"] = value.str(); }
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

} // namespace anet::rl
