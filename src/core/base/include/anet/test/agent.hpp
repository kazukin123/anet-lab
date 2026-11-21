#pragma once

#ifndef ANET_TEST_SUPPORT
#error "anet/test/agent.hpp requires ANET_TEST_SUPPORT"
#endif

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "anet/rl.hpp"

namespace anet::test {

    class StubAgent final : public anet::rl::Agent {
    public:
        explicit StubAgent(torch::Device device);
    public:
        std::shared_ptr<anet::rl::Actor> CreateActor(const anet::rl::ActorRequest& request) const override;
        std::shared_ptr<anet::rl::Learner> CreateLearner() override;
        torch::Device GetDevice() const override;
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(
            const std::string& key, int64_t index = -1) const override;
    private:
        torch::Device device_;
    };

    class StubAgentFactory final : public anet::rl::AgentFactory {
    public:
        std::shared_ptr<anet::rl::Agent> CreateAgent(
            const anet::rl::EnvSpec& env_spec, const anet::rl::BatchEnvSpec& batch_env_spec,
            const torch::Device& device, const anet::ConfigData& config,
            std::shared_ptr<anet::rl::Notifier> notifier, std::optional<anet::seed_t> seed) const override;
        std::string GetTargetAgentClassId() const override;
    };
}
