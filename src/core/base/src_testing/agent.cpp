#include "anet/test/agent.hpp"
#include <utility>

using namespace anet::test;

namespace {

    // Run を組み立てるテストで、各 lane に有効な固定アクションを返す。
    class StubActor final : public anet::rl::Actor {
    public:
        StubActor(int64_t num_envs, torch::Device device)
            : num_envs_(num_envs), device_(std::move(device)) {}
    public:
        std::shared_ptr<anet::rl::BatchActionInfo> MakeAction(
            const anet::rl::StepCounts&, const anet::rl::BatchState&,
            const anet::rl::MakeActionOptions&) const override
        {
            return std::make_shared<anet::rl::BatchActionInfo>(
                torch::zeros({num_envs_}, torch::TensorOptions().dtype(torch::kInt64).device(device_)));
        }
        void Sync() override {}
    private:
        int64_t num_envs_;
        torch::Device device_;
    };

    // Run の更新経路を通る場合も、テストの外へ状態を持ち出さない。
    class StubLearner final : public anet::rl::Learner {
    public:
        anet::rl::BatchUpdateResultList UpdateFromBatch(
            const anet::rl::StepCounts&, const anet::rl::BatchExperience&) override
        {
            return {};
        }
    };

} // namespace

StubAgent::StubAgent(torch::Device device) : device_(std::move(device)) {}

std::shared_ptr<anet::rl::Actor> StubAgent::CreateActor(const anet::rl::ActorRequest& request) const
{
    return std::make_shared<StubActor>(request.batch_env_spec.num_envs, device_);
}

std::shared_ptr<anet::rl::Learner> StubAgent::CreateLearner()
{
    return std::make_shared<StubLearner>();
}

torch::Device StubAgent::GetDevice() const
{
    return device_;
}

std::optional<float> StubAgent::GetScalar(const std::string&, int64_t) const
{
    return std::nullopt;
}

std::optional<torch::Tensor> StubAgent::GetTensor(const std::string&, int64_t) const
{
    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>> StubAgent::GetTensorVector(const std::string&, int64_t) const
{
    return std::nullopt;
}

std::shared_ptr<anet::rl::Agent> StubAgentFactory::CreateAgent(
    const anet::rl::EnvSpec&, const anet::rl::BatchEnvSpec&, const torch::Device& device,
    const anet::ConfigData&, std::shared_ptr<anet::rl::Notifier>, std::optional<anet::seed_t>) const
{
    return std::make_shared<StubAgent>(device);
}

std::string StubAgentFactory::GetTargetAgentClassId() const
{
    return "StubAgent";
}
