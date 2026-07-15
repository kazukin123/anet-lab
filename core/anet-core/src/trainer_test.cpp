#include "anet/catch_test.hpp"

#include "anet/trainer.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace rl = anet::rl;

bool ContainsText(const std::string& text, const std::string& pattern)
{
    return text.find(pattern) != std::string::npos;
}

torch::Tensor BoolTensor(const std::vector<bool>& values)
{
    auto tensor = torch::empty({ static_cast<int64_t>(values.size()) }, torch::TensorOptions().dtype(torch::kBool));
    for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
        tensor[i].fill_(values[static_cast<size_t>(i)]);
    }
    return tensor;
}

anet::TensorDict MakeObs(int64_t num_envs)
{
    return anet::TensorDict{ { rl::ObsKeys::kVector, torch::zeros({ num_envs, 1 }, torch::kFloat32) } };
}

rl::BatchState MakeState(int64_t num_envs)
{
    const std::vector<bool> flags(static_cast<size_t>(num_envs), false);
    return rl::BatchState{ MakeObs(num_envs), BoolTensor(flags), BoolTensor(flags), BoolTensor(flags) };
}

rl::EnvSpec MakeEnvSpec()
{
    anet::TensorSpec vector_spec;
    vector_spec.type = anet::SpaceType::Vector;
    vector_spec.shape = { 1 };
    vector_spec.dtype = torch::kFloat32;

    rl::EnvSpec spec;
    spec.state_spec.obs_spec[rl::ObsKeys::kVector] = vector_spec;
    spec.action_spec.is_discrete = true;
    spec.action_spec.value_labels = { "noop" };
    spec.reward_range = { -1.0f, 1.0f };
    return spec;
}

class TestResetResult final : public rl::BatchResetResult {
public:
    explicit TestResetResult(int64_t num_envs)
        : rl::BatchResetResult(MakeState(num_envs))
        , num_envs_(num_envs)
    {
    }

    std::vector<rl::AuxData> GetAuxDataList(int env_index = -1) const override
    {
        if (env_index >= 0) return { rl::AuxData{} };
        return std::vector<rl::AuxData>(static_cast<size_t>(num_envs_));
    }

private:
    int64_t num_envs_ = 0;
};

class TestStepResult final : public rl::BatchStepResult {
public:
    explicit TestStepResult(int64_t num_envs)
        : rl::BatchStepResult(
            torch::zeros({ num_envs }, torch::kFloat32),
            MakeState(num_envs),
            MakeState(num_envs),
            static_cast<uint32_t>(num_envs),
            0)
        , num_envs_(num_envs)
    {
    }

    std::vector<rl::AuxData> GetAuxDataList(int env_index = -1) const override
    {
        if (env_index >= 0) return { rl::AuxData{} };
        return std::vector<rl::AuxData>(static_cast<size_t>(num_envs_));
    }

private:
    int64_t num_envs_ = 0;
};

class TestBatchEnv final : public rl::BatchEnv {
public:
    TestBatchEnv(int num_envs, torch::Device device)
        : batch_spec_{ num_envs, 1 }
        , device_(std::move(device))
    {
    }

    rl::EnvSpec GetSpec() const override { return MakeEnvSpec(); }
    rl::BatchEnvSpec GetBatchSpec() const override { return batch_spec_; }
    torch::Device GetDevice() const override { return device_; }

    std::shared_ptr<const rl::BatchResetResult> Reset(rl::RunMode = rl::RunMode::Train) override
    {
        return std::make_shared<TestResetResult>(batch_spec_.num_envs);
    }

    std::shared_ptr<const rl::BatchStepResult> Step(std::shared_ptr<rl::BatchActionInfo>, rl::RunMode = rl::RunMode::Train) override
    {
        return std::make_shared<TestStepResult>(batch_spec_.num_envs);
    }

    std::optional<float> GetScalar(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t = -1) const override { return std::nullopt; }

private:
    rl::BatchEnvSpec batch_spec_;
    torch::Device device_;
};

class TestActor final : public rl::Actor {
public:
    explicit TestActor(int64_t num_envs)
        : num_envs_(num_envs)
    {
    }

    std::shared_ptr<rl::BatchActionInfo> MakeAction(const rl::StepCounts&, const rl::BatchState&) const override
    {
        return std::make_shared<rl::BatchActionInfo>(torch::zeros({ num_envs_ }, torch::kInt64));
    }

    void Sync() override { ++sync_count_; }

    int GetSyncCount() const { return sync_count_; }

private:
    int64_t num_envs_ = 1;
    int sync_count_ = 0;
};

class TestLearner final : public rl::Learner {
public:
    rl::BatchUpdateResultList UpdateFromBatch(const rl::StepCounts&, const rl::BatchExperience&) override
    {
        return {};
    }
};

class TestAgent final : public rl::Agent {
public:
    explicit TestAgent(torch::Device device)
        : device_(std::move(device))
    {
    }

    std::shared_ptr<rl::Actor> CreateActor(
        const rl::BatchEnvSpec& batch_env_spec,
        rl::RunMode,
        std::optional<bool> clone_model_override = std::nullopt,
        std::optional<torch::Device> = std::nullopt) const override
    {
        last_clone_model_override_ = clone_model_override;
        last_actor_ = std::make_shared<TestActor>(batch_env_spec.num_envs);
        return last_actor_;
    }

    std::optional<bool> GetLastCloneModelOverride() const
    {
        return last_clone_model_override_;
    }

    std::shared_ptr<const TestActor> GetLastActor() const { return last_actor_; }

    std::shared_ptr<rl::Learner> CreateLearner() override
    {
        return std::make_shared<TestLearner>();
    }

    torch::Device GetDevice() const override { return device_; }

    std::optional<float> GetScalar(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t = -1) const override { return std::nullopt; }

private:
    torch::Device device_;
    mutable std::optional<bool> last_clone_model_override_ = true;
    mutable std::shared_ptr<TestActor> last_actor_;
};

} // namespace

TEST_CASE("TrainRunner delegates clone policy to Agent", "[trainer][actor]")
{
    auto env = std::make_shared<TestBatchEnv>(1, torch::Device(torch::kCPU));
    auto agent = std::make_shared<TestAgent>(torch::Device(torch::kCPU));

    auto runner = std::make_shared<rl::SerialTrainRunner>(env, agent, nullptr);

    CHECK_FALSE(agent->GetLastCloneModelOverride().has_value());
}

TEST_CASE("PipelineTrainRunner does not force actor synchronization every step", "[trainer][actor][pipeline]")
{
    auto env = std::make_shared<TestBatchEnv>(1, torch::Device(torch::kCPU));
    auto agent = std::make_shared<TestAgent>(torch::Device(torch::kCPU));
    auto notifier = std::make_shared<rl::Notifier>();
    auto runner = std::make_shared<rl::PipelineTrainRunner>(env, agent, notifier);

    runner->DoStep();

    REQUIRE(agent->GetLastActor());
    CHECK(agent->GetLastActor()->GetSyncCount() == 0);
    runner->Shutdown();
}

TEST_CASE("EvalRunner allows shared actor when actor device matches agent device", "[trainer][eval_runner]")
{
    auto env = std::make_shared<TestBatchEnv>(1, torch::Device(torch::kCPU));
    auto agent = std::make_shared<TestAgent>(torch::Device(torch::kCPU));

    REQUIRE_NOTHROW(std::make_shared<rl::EvalRunner>(
        env, agent, nullptr, rl::RunMode::Eval, false, torch::Device(torch::kCPU), "eval_cpu"));
}

TEST_CASE("EvalRunner allows cloned actor on different device", "[trainer][eval_runner]")
{
    auto env = std::make_shared<TestBatchEnv>(1, torch::Device(torch::kCPU));
    auto agent = std::make_shared<TestAgent>(torch::Device(torch::kCPU));

    REQUIRE_NOTHROW(std::make_shared<rl::EvalRunner>(
        env, agent, nullptr, rl::RunMode::Eval, true, torch::Device(torch::kCUDA, 0), "eval_cuda"));
}

TEST_CASE("EvalRunner rejects shared actor when actor device differs from agent device", "[trainer][eval_runner]")
{
    auto env = std::make_shared<TestBatchEnv>(1, torch::Device(torch::kCPU));
    auto agent = std::make_shared<TestAgent>(torch::Device(torch::kCPU));

    bool thrown = false;
    try {
        auto runner = std::make_shared<rl::EvalRunner>(
            env, agent, nullptr, rl::RunMode::Eval, false, torch::Device(torch::kCUDA, 0), "eval_mismatch");
        static_cast<void>(runner);
    } catch (const std::exception& e) {
        const std::string message = e.what();
        thrown = true;
        CHECK(ContainsText(message, "clone_model=false"));
        CHECK(ContainsText(message, "actor_device"));
        CHECK(ContainsText(message, "agent_device"));
    }

    CHECK(thrown);
}
