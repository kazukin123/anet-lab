#include "anet/catch_test.hpp"

#include "anet/agent.hpp"
#include "anet/env.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/test_util.hpp"
#include "anet/trainer.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace rl = anet::rl;

bool ContainsText(const std::string& text, const std::string& pattern)
{
    return text.find(pattern) != std::string::npos;
}

std::string ReadTextFile(const std::filesystem::path& path)
{
    std::ifstream ifs(path);
    return std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
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

rl::BatchState MakeResetState(int64_t num_envs)
{
    const std::vector<bool> flags(static_cast<size_t>(num_envs), false);
    const std::vector<bool> starts(static_cast<size_t>(num_envs), true);
    return rl::BatchState{ MakeObs(num_envs), BoolTensor(flags), BoolTensor(flags), BoolTensor(starts) };
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

class ThrowingRunner final : public rl::Runner {
public:
    rl::StepCounts DoStep() override
    {
        throw std::runtime_error("runner failure");
    }

    rl::StepCounts DoUpdateFrame(int, ControlFunction, ControlFunction) override
    {
        throw std::runtime_error("runner failure");
    }

    rl::RunnerStatus GetStatus() const override { return rl::RunnerStatus::RUNNING; }
    void Shutdown() override {}
    rl::StepCounts GetCounts() const override { return {}; }
    const std::string& GetName() const override { return name_; }
    std::shared_ptr<rl::BatchEnv> GetBatchEnv() const override { return nullptr; }
    std::shared_ptr<rl::Agent> GetAgent() const override { return nullptr; }
    std::shared_ptr<rl::Notifier> GetNotifier() const override { return nullptr; }
    std::optional<float> GetScalar(const std::string&, int64_t) const override { return std::nullopt; }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(
        const std::string&, int64_t) const override { return std::nullopt; }

private:
    std::string name_ = "ThrowingRunner";
};

class TestResetResult final : public rl::BatchResetResult {
public:
    explicit TestResetResult(int64_t num_envs)
        : rl::BatchResetResult(MakeResetState(num_envs))
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

class TestBatchEnv final : public rl::BatchEnvBase {
public:
    TestBatchEnv(
        const std::string& name,
        int num_envs,
        torch::Device device,
        rl::RunMode run_mode = rl::RunMode::Train,
        rl::EpisodeScope episode_scope = rl::EpisodeScope::PER_LANE)
        : rl::BatchEnvBase(name, num_envs, run_mode)
        , batch_spec_{
            .num_envs = num_envs,
            .num_threads = 1,
            .episode_scope = episode_scope,
        }
        , device_(std::move(device))
    {
    }

    rl::EnvSpec GetSpec() const override { return MakeEnvSpec(); }
    rl::BatchEnvSpec GetBatchSpec() const override { return batch_spec_; }
    torch::Device GetDevice() const override { return device_; }

    std::shared_ptr<const rl::BatchResetResult> Reset() override
    {
        return std::make_shared<TestResetResult>(batch_spec_.num_envs);
    }

    std::shared_ptr<const rl::BatchStepResult> Step(std::shared_ptr<rl::BatchActionInfo>) override
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
        auto payload = torch::arange(
            1, num_envs_ * 3 + 1, torch::TensorOptions().dtype(torch::kFloat32)).reshape({ num_envs_, 3 });
        return std::make_shared<rl::BatchActionInfo>(
            torch::zeros({ num_envs_ }, torch::kInt64),
            anet::TensorDict{},
            rl::AuxData{},
            rl::ReplayInitialPriorityHint(std::move(payload)));
    }

    void Sync() override { ++sync_count_; }

    int GetSyncCount() const { return sync_count_; }

private:
    int64_t num_envs_ = 1;
    int sync_count_ = 0;
};

class HintRecordingReplayBuffer final : public rl::ReplayBuffer {
public:
    void Push(const rl::BatchExperience& batch_exp) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto& hint = batch_exp.action->GetReplayInitialPriorityHint();
        if (hint.has_value()) {
            payload_ = hint->GetPayload().clone();
        }
        ++push_count_;
    }

    void Sample(rl::ExperienceSamples&, int64_t, float) const override {}
    bool SampleUniqueUniform(
        rl::ExperienceSamples&, int64_t, anet::RandomGenerator&) const override { return false; }
    int64_t Size() const override { return 0; }
    rl::ReplayPriorityUpdateResult UpdatePriorities(
        const std::vector<int64_t>&, const std::vector<float>&) override
    {
        return {};
    }
    std::optional<float> GetScalar(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(
        const std::string&, int64_t = -1) const override
    {
        return std::nullopt;
    }

    torch::Tensor GetPayload() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return payload_.defined() ? payload_.clone() : torch::Tensor();
    }

    int GetPushCount() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return push_count_;
    }

private:
    mutable std::mutex mutex_;
    torch::Tensor payload_;
    int push_count_ = 0;
};

class TestLearner final : public rl::Learner {
public:
    explicit TestLearner(std::shared_ptr<rl::ReplayBuffer> replay_buffer)
        : replay_buffer_(std::move(replay_buffer))
    {
    }

    rl::BatchUpdateResultList UpdateFromBatch(
        const rl::StepCounts&, const rl::BatchExperience& batch_exp) override
    {
        replay_buffer_->Push(batch_exp);
        return {};
    }

private:
    std::shared_ptr<rl::ReplayBuffer> replay_buffer_;
};

class TestAgent final : public rl::Agent {
public:
    explicit TestAgent(torch::Device device)
        : device_(std::move(device))
        , replay_buffer_(std::make_shared<HintRecordingReplayBuffer>())
    {
    }

    std::shared_ptr<rl::Actor> CreateActor(
        const rl::BatchEnvSpec& batch_env_spec,
        const rl::EnvSpec&,
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
        return std::make_shared<TestLearner>(replay_buffer_);
    }

    std::shared_ptr<const HintRecordingReplayBuffer> GetReplayBuffer() const { return replay_buffer_; }

    torch::Device GetDevice() const override { return device_; }

    std::optional<float> GetScalar(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t = -1) const override { return std::nullopt; }

private:
    torch::Device device_;
    mutable std::optional<bool> last_clone_model_override_ = true;
    mutable std::shared_ptr<TestActor> last_actor_;
    std::shared_ptr<HintRecordingReplayBuffer> replay_buffer_;
};

class RunManagerTestSingleEnv final : public rl::SingleDiscreteEnvBase {
public:
    explicit RunManagerTestSingleEnv(
        const std::string& name,
        rl::RunMode run_mode = rl::RunMode::Train,
        std::optional<anet::ConfigData> config_data = std::nullopt)
        : rl::SingleDiscreteEnvBase(name, run_mode, std::move(config_data))
    {
    }

    rl::EnvSpec GetSpec() const override { return MakeEnvSpec(); }

    std::shared_ptr<const rl::SingleResetResult> Reset() override
    {
        return std::make_shared<rl::SingleResetResult>(rl::SingleState{
            .obs = { rl::ObsKeys::kVector, torch::zeros({ 1 }, torch::kFloat32) },
            .done = false,
            .truncated = false,
            .episode_start = true,
        });
    }

    std::shared_ptr<const rl::SingleStepResult> Step(int64_t) override
    {
        return std::make_shared<rl::SingleStepResult>(0.0f, rl::SingleState{
            .obs = { rl::ObsKeys::kVector, torch::zeros({ 1 }, torch::kFloat32) },
            .done = false,
            .truncated = false,
            .episode_start = false,
        });
    }

    std::optional<float> GetScalar(const std::string&, int64_t) const override { return std::nullopt; }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(
        const std::string&, int64_t) const override
    {
        return std::nullopt;
    }
};

struct RunManagerEnvFactoryState {
    int creation_count = 0;
    std::optional<std::string> failing_name;
    std::vector<std::string> config_prefixes;
};

class RunManagerTestEnvFactory final : public rl::SingleDiscreteEnvFactory {
public:
    explicit RunManagerTestEnvFactory(std::shared_ptr<RunManagerEnvFactoryState> state)
        : state_(std::move(state))
    {
    }

    std::shared_ptr<rl::SingleDiscreteEnv> CreateSingleEnv(
        const anet::ConfigData&, const torch::Device&, const std::string& name,
        std::optional<anet::seed_t>, rl::RunMode run_mode, const std::string& config_prefix) override
    {
        ++state_->creation_count;
        state_->config_prefixes.push_back(config_prefix);
        if (state_->failing_name == name) {
            throw std::runtime_error("Requested RunManager test Env failure: " + name);
        }
        anet::ConfigData config_snapshot;
        config_snapshot.Set("RunManagerNameTestEnv.value", 1);
        return std::make_shared<RunManagerTestSingleEnv>(name, run_mode, config_snapshot);
    }

    std::string GetTargetEnvClassId() const override { return "RunManagerNameTestEnv"; }

private:
    std::shared_ptr<RunManagerEnvFactoryState> state_;
};

class RunManagerSharedBatchEnvFactory final : public rl::BatchEnvFactory {
public:
    std::shared_ptr<rl::BatchEnv> CreateBatchEnv(
        const anet::ConfigData&,
        const torch::Device& device,
        const std::string& name,
        std::optional<anet::seed_t>,
        int num_envs,
        rl::RunMode run_mode,
        const std::string&) override
    {
        const auto episode_scope = run_mode == rl::RunMode::Train
            ? rl::EpisodeScope::PER_LANE : rl::EpisodeScope::SHARED;
        return std::make_shared<TestBatchEnv>(
            name, num_envs, device, run_mode, episode_scope);
    }

    std::string GetTargetEnvClassId() const override
    {
        return "RunManagerSharedBatchEnv";
    }
};

class RunManagerTestAgentFactory final : public rl::AgentFactory {
public:
    std::shared_ptr<rl::Agent> CreateAgent(
        const rl::EnvSpec&, const rl::BatchEnvSpec&, const torch::Device& device,
        const anet::ConfigData&, std::shared_ptr<rl::Notifier>, std::optional<anet::seed_t>) const override
    {
        return std::make_shared<TestAgent>(device);
    }

    std::string GetTargetAgentClassId() const override { return "RunManagerNameTestAgent"; }
};

class RunManagerNoopMetricsBackend final : public anet::IBackend {
public:
    void Open(const std::filesystem::path&, const std::string&) override {}
    void WriteJsonl(const anet::json&) override {}
    void Flush() override {}
};

class ScopedRunManagerMetricsLogger final {
public:
    ScopedRunManagerMetricsLogger()
    {
        anet::MetricsLogger::Reset();
        anet::MetricsLoggerConfig logger_config;
        logger_config.run_name_tmpl = "run_manager_env_name_test";
        anet::MetricsLogger::Init(
            std::make_unique<RunManagerNoopMetricsBackend>(), logger_config, "C:/tmp");
    }

    ~ScopedRunManagerMetricsLogger()
    {
        anet::MetricsLogger::Reset();
    }
};

anet::ConfigData MakeRunManagerNameTestConfig()
{
    anet::ConfigData config;
    config.Set("env.class_id", "RunManagerNameTestEnv");
    config.Set("env.device_type", "0");
    config.Set("env.worker_type", "1");
    config.Set("agent.class_id", "RunManagerNameTestAgent");
    config.Set("agent.device_type", "0");
    config.Set("train.seed", "123");
    config.Set("train.num_envs", "1");
    config.Set("train.main_runner_type", "serial");
    config.Set("train.eval_device_type", "cpu");
    return config;
}

std::shared_ptr<RunManagerEnvFactoryState> RegisterRunManagerNameTestFactories()
{
    static auto state = std::make_shared<RunManagerEnvFactoryState>();
    static std::once_flag register_once;
    std::call_once(register_once, [&] {
        rl::EnvRepository::Instance().Regist(std::make_shared<RunManagerTestEnvFactory>(state));
        rl::EnvRepository::Instance().Regist(std::make_shared<RunManagerSharedBatchEnvFactory>());
        rl::AgentRepository::Instance().Register(std::make_shared<RunManagerTestAgentFactory>());
    });
    state->creation_count = 0;
    state->failing_name.reset();
    state->config_prefixes.clear();
    return state;
}

} // namespace

TEST_CASE("TrainRunner delegates clone policy to Agent", "[trainer][actor]")
{
    auto env = std::make_shared<TestBatchEnv>("serial-train", 1, torch::Device(torch::kCPU));
    auto agent = std::make_shared<TestAgent>(torch::Device(torch::kCPU));

    auto runner = std::make_shared<rl::SerialTrainRunner>(env, agent, nullptr);

    CHECK_FALSE(agent->GetLastCloneModelOverride().has_value());
}

TEST_CASE("PipelineTrainRunner does not force actor synchronization every step", "[trainer][actor][pipeline]")
{
    auto env = std::make_shared<TestBatchEnv>("pipeline-train", 1, torch::Device(torch::kCPU));
    auto agent = std::make_shared<TestAgent>(torch::Device(torch::kCPU));
    auto notifier = std::make_shared<rl::Notifier>();
    auto runner = std::make_shared<rl::PipelineTrainRunner>(env, agent, notifier);

    runner->DoStep();

    REQUIRE(agent->GetLastActor());
    CHECK(agent->GetLastActor()->GetSyncCount() == 0);
    runner->Shutdown();
}

TEST_CASE("Train runners preserve opaque K3 replay priority hints to the replay boundary", "[trainer][replay_hint]")
{
    auto run_and_get_payload = [](bool pipeline) {
        auto env = std::make_shared<TestBatchEnv>("replay-hint", 2, torch::Device(torch::kCPU));
        auto agent = std::make_shared<TestAgent>(torch::Device(torch::kCPU));
        auto notifier = std::make_shared<rl::Notifier>();

        if (pipeline) {
            auto runner = std::make_shared<rl::PipelineTrainRunner>(env, agent, notifier);
            // 1step目の経験を保存し、2step目で非同期Learnerへ渡す。
            runner->DoStep();
            runner->DoStep();
            runner->Shutdown();
        } else {
            auto runner = std::make_shared<rl::SerialTrainRunner>(env, agent, notifier);
            runner->DoStep();
            runner->Shutdown();
        }

        REQUIRE(agent->GetReplayBuffer()->GetPushCount() == 1);
        return agent->GetReplayBuffer()->GetPayload();
    };

    const auto expected = torch::tensor({ { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });

    SECTION("Serial")
    {
        auto payload = run_and_get_payload(false);
        REQUIRE(payload.defined());
        CHECK(torch::equal(payload, expected));
    }

    SECTION("Pipeline")
    {
        auto payload = run_and_get_payload(true);
        REQUIRE(payload.defined());
        CHECK(torch::equal(payload, expected));
    }
}

TEST_CASE("EvalRunner allows shared actor when actor device matches agent device", "[trainer][eval_runner]")
{
    auto env = std::make_shared<TestBatchEnv>("eval-cpu", 1, torch::Device(torch::kCPU));
    auto agent = std::make_shared<TestAgent>(torch::Device(torch::kCPU));

    REQUIRE_NOTHROW(std::make_shared<rl::EvalRunner>(
        env, agent, nullptr, rl::RunMode::Eval, false, torch::Device(torch::kCPU), "eval_cpu"));
}

TEST_CASE("EvalRunner allows cloned actor on different device", "[trainer][eval_runner]")
{
    auto env = std::make_shared<TestBatchEnv>("eval-cuda", 1, torch::Device(torch::kCPU));
    auto agent = std::make_shared<TestAgent>(torch::Device(torch::kCPU));

    REQUIRE_NOTHROW(std::make_shared<rl::EvalRunner>(
        env, agent, nullptr, rl::RunMode::Eval, true, torch::Device(torch::kCUDA, 0), "eval_cuda"));
}

TEST_CASE("EvalRunner rejects shared actor when actor device differs from agent device", "[trainer][eval_runner]")
{
    auto env = std::make_shared<TestBatchEnv>("eval-mismatch", 1, torch::Device(torch::kCPU));
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

TEST_CASE("RunManager rejects reserved configured Eval names before constructing Env", "[env_name][run_manager]")
{
    ScopedRunManagerMetricsLogger metrics_logger;
    SECTION("train") {
        auto factory_state = RegisterRunManagerNameTestFactories();
        auto config = MakeRunManagerNameTestConfig();
        config.Set("train.eval.[train].run_mode", "eval1");

        CHECK_THROWS_WITH(
            std::make_shared<rl::RunManager>(config),
            Catch::Matchers::ContainsSubstring("Duplicate Env name 'train' within Run"));
        CHECK(factory_state->creation_count == 0);
    }

    SECTION("EvalPanel") {
        auto factory_state = RegisterRunManagerNameTestFactories();
        auto config = MakeRunManagerNameTestConfig();
        config.Set("train.eval.[EvalPanel].run_mode", "eval1");

        CHECK_THROWS_WITH(
            std::make_shared<rl::RunManager>(config),
            Catch::Matchers::ContainsSubstring("Duplicate Env name 'EvalPanel' within Run"));
        CHECK(factory_state->creation_count == 0);
    }
}

TEST_CASE("RunManager propagates distinct Env names without interpreting them", "[env_name][run_manager]")
{
    ScopedRunManagerMetricsLogger metrics_logger;
    auto factory_state = RegisterRunManagerNameTestFactories();
    auto config = MakeRunManagerNameTestConfig();
    config.Set("train.eval.[eval_a].run_mode", "eval1");
    config.Set("train.eval.[Train].run_mode", "eval1");
    config.Set("train.eval_schedule.[eval_a].interval", "100");
    config.Set("train.eval_schedule.[Train].interval", "100");

    auto manager = std::make_shared<rl::RunManager>(config);

    REQUIRE(manager->GetStatus() == rl::RunnerStatus::RUNNING);
    CHECK(manager->GetTrainRunner()->GetBatchEnv()->GetName() == "train");
    CHECK(manager->GetEvalRunner("eval_a")->GetBatchEnv()->GetName() == "eval_a");
    CHECK(manager->GetEvalRunner("Train")->GetBatchEnv()->GetName() == "Train");
    CHECK(factory_state->creation_count == 3);

    const auto eval_panel = manager->CreateEvalRunner("EvalPanel");
    CHECK(eval_panel->GetBatchEnv()->GetName() == "EvalPanel");
    CHECK(factory_state->creation_count == 4);
    CHECK(factory_state->config_prefixes.back().empty());
    CHECK(std::ranges::find(
        factory_state->config_prefixes, "train.eval.[eval_a].env")
        != factory_state->config_prefixes.end());
}

TEST_CASE("RunManager writes each Env effective config to its own file", "[trainer][env_config]")
{
    ScopedRunManagerMetricsLogger metrics_logger;
    RegisterRunManagerNameTestFactories();
    const auto config_path = anet::MetricsLogger::Instance()->GetRunDir() /
        "config" / "env.train.txt";
    std::filesystem::remove(config_path);

    auto manager = std::make_shared<rl::RunManager>(MakeRunManagerNameTestConfig());

    REQUIRE(manager->GetStatus() == rl::RunnerStatus::RUNNING);
    REQUIRE(std::filesystem::exists(config_path));
    CHECK(ContainsText(ReadTextFile(config_path), "env.class_id = RunManagerNameTestEnv"));
    CHECK(ContainsText(ReadTextFile(config_path), "RunManagerNameTestEnv.value = 1"));
}

TEST_CASE("RunManager rejects distinct Env names that map to one config filename", "[trainer][env_config]")
{
    ScopedRunManagerMetricsLogger metrics_logger;
    RegisterRunManagerNameTestFactories();
    auto config = MakeRunManagerNameTestConfig();
    config.Set("train.eval.[eval/a].run_mode", "eval1");
    config.Set("train.eval.[eval-a].run_mode", "eval1");
    config.Set("train.eval_schedule.[eval/a].interval", "100");
    config.Set("train.eval_schedule.[eval-a].interval", "100");

    CHECK_THROWS_WITH(
        std::make_shared<rl::RunManager>(config),
        Catch::Matchers::ContainsSubstring("Env config filename collision"));
}

TEST_CASE("RunManager reserves dormant Eval tags without constructing an Env", "[trainer][dormant_eval]")
{
    ScopedRunManagerMetricsLogger metrics_logger;
    auto factory_state = RegisterRunManagerNameTestFactories();
    auto config = MakeRunManagerNameTestConfig();
    config.Set("train.eval.[sleep].run_mode", "eval1");
    config.Set("train.eval_schedule.[sleep].interval", "0");
    config.Set(
        "metrics.scalar.[sleep_reward]",
        "mean.episode_return $runner @episode_end $eval.[sleep]");

    auto manager = std::make_shared<rl::RunManager>(config);
    REQUIRE(manager->GetStatus() == rl::RunnerStatus::RUNNING);
    CHECK(factory_state->creation_count == 1);
    CHECK_THROWS_WITH(
        manager->CreateEvalRunner("sleep"),
        Catch::Matchers::ContainsSubstring("Duplicate Env name 'sleep' within Run"));
    CHECK(factory_state->creation_count == 1);
}

TEST_CASE("RunManager keeps definition-only Eval tags dormant", "[trainer][eval_schedule]")
{
    ScopedRunManagerMetricsLogger metrics_logger;
    anet::test::LogCaptureGuard logs(wxLOG_Info);
    auto factory_state = RegisterRunManagerNameTestFactories();
    auto config = MakeRunManagerNameTestConfig();
    config.Set("train.eval.[sleep].run_mode", "eval1");
    config.Set(
        "metrics.scalar.[sleep_reward_a]",
        "mean.episode_return $runner @episode_end $eval.[sleep]");
    config.Set(
        "metrics.scalar.[sleep_reward_b]",
        "mean.episode_return $runner @episode_end $eval.[sleep]");

    auto manager = std::make_shared<rl::RunManager>(config);
    logs.Flush();

    REQUIRE(manager->GetStatus() == rl::RunnerStatus::RUNNING);
    CHECK(factory_state->creation_count == 1);
    CHECK_THROWS_AS(manager->GetEvalRunner("sleep"), std::out_of_range);
    CHECK(anet::test::HasRecordContaining(
        logs.Records(), wxLOG_Message, { "eval tag 'sleep': definition-only" }));
    CHECK(std::ranges::count_if(logs.Records(), [](const auto& record) {
        return record.level == wxLOG_Warning
            && ContainsText(record.message, "Skipping metrics for unscheduled eval tag. tag='sleep'.");
    }) == 1);
}

TEST_CASE("RunManager rejects schedules for undefined Eval tags", "[trainer][eval_schedule]")
{
    ScopedRunManagerMetricsLogger metrics_logger;
    auto factory_state = RegisterRunManagerNameTestFactories();
    auto config = MakeRunManagerNameTestConfig();
    config.Set("train.eval_schedule.[ghost].interval", "10");

    CHECK_THROWS_WITH(
        std::make_shared<rl::RunManager>(config),
        Catch::Matchers::ContainsSubstring("train.eval.[ghost]"));
    CHECK(factory_state->creation_count == 0);
}

TEST_CASE("RunManager validates Eval session cardinality and ENV scalar prefixes", "[trainer][eval_session]")
{
    ScopedRunManagerMetricsLogger metrics_logger;

    SECTION("eval_episodes must be positive even for a definition-only tag") {
        auto factory_state = RegisterRunManagerNameTestFactories();
        auto config = MakeRunManagerNameTestConfig();
        config.Set("train.eval.[sleep].run_mode", "eval1");
        config.Set("train.eval.[sleep].eval_episodes", "0");
        CHECK_THROWS_WITH(
            std::make_shared<rl::RunManager>(config),
            Catch::Matchers::ContainsSubstring("train.eval.[sleep].eval_episodes=0"));
        CHECK(factory_state->creation_count == 1);
    }

    SECTION("multi-episode ENV metrics require an aggregation prefix") {
        auto factory_state = RegisterRunManagerNameTestFactories();
        auto config = MakeRunManagerNameTestConfig();
        config.Set("train.eval.[eval1].run_mode", "eval1");
        config.Set("train.eval.[eval1].eval_episodes", "2");
        config.Set("train.eval_schedule.[eval1].interval", "100");
        config.Set("metrics.scalar.[eval_score]", "score $env @episode_end $eval.[eval1]");
        CHECK_THROWS_WITH(
            std::make_shared<rl::RunManager>(config),
            Catch::Matchers::ContainsSubstring("requires an aggregation prefix"));
        CHECK(factory_state->creation_count == 1);
    }
}

TEST_CASE("RunManager decorates configured Eval but leaves EvalPanel step-driven", "[trainer][eval_session]")
{
    ScopedRunManagerMetricsLogger metrics_logger;
    anet::test::LogCaptureGuard logs(wxLOG_Info);
    auto factory_state = RegisterRunManagerNameTestFactories();
    auto config = MakeRunManagerNameTestConfig();
    config.Set("train.eval.[eval1].run_mode", "eval1");
    config.Set("train.eval.[eval1].eval_batch_size", "2");
    config.Set("train.eval.[eval1].eval_episodes", "1");
    config.Set("train.eval_schedule.[eval1].interval", "100");
    config.Set("metrics.scalar.[eval_score]", "mean.score $env @episode_end $eval.[eval1]");
    config.Set("train.eval.[eval2].run_mode", "eval2");
    config.Set("train.eval.[eval2].eval_batch_size", "2");
    config.Set("train.eval.[eval2].eval_episodes", "1");
    config.Set("train.eval_schedule.[eval2].interval", "100");

    auto manager = std::make_shared<rl::RunManager>(config);
    logs.Flush();
    REQUIRE(manager->GetStatus() == rl::RunnerStatus::RUNNING);
    CHECK(dynamic_cast<rl::EvalSessionEnv*>(
        manager->GetEvalRunner("eval1")->GetBatchEnv().get()) != nullptr);
    CHECK(std::ranges::count_if(logs.Records(), [](const auto& record) {
        return record.level == wxLOG_Warning
            && ContainsText(record.message, "eval_tag='eval1'")
            && ContainsText(record.message, "eval_episodes=1")
            && ContainsText(record.message, "group_count=2");
    }) == 1);
    CHECK(std::ranges::count_if(logs.Records(), [](const auto& record) {
        return record.level == wxLOG_Warning
            && ContainsText(record.message, "eval_tag='eval2'")
            && ContainsText(record.message, "eval_episodes=1")
            && ContainsText(record.message, "group_count=2");
    }) == 1);

    const int creation_count_before_panel = factory_state->creation_count;
    const auto eval_panel = manager->CreateEvalRunner("EvalPanel");
    CHECK(dynamic_cast<rl::EvalSessionEnv*>(eval_panel->GetBatchEnv().get()) == nullptr);
    CHECK(factory_state->creation_count == creation_count_before_panel + 1);
}

TEST_CASE("RunManager does not warn when shared Eval adopts its only group", "[trainer][eval_session]")
{
    ScopedRunManagerMetricsLogger metrics_logger;
    anet::test::LogCaptureGuard logs(wxLOG_Info);
    RegisterRunManagerNameTestFactories();
    auto config = MakeRunManagerNameTestConfig();
    config.Set("env.class_id", "RunManagerSharedBatchEnv");
    config.Set("train.eval.[shared].run_mode", "eval1");
    config.Set("train.eval.[shared].eval_batch_size", "2");
    config.Set("train.eval.[shared].eval_episodes", "1");
    config.Set("train.eval_schedule.[shared].interval", "100");

    auto manager = std::make_shared<rl::RunManager>(config);
    logs.Flush();

    REQUIRE(manager->GetStatus() == rl::RunnerStatus::RUNNING);
    CHECK(std::ranges::none_of(logs.Records(), [](const auto& record) {
        return record.level == wxLOG_Warning
            && ContainsText(record.message, "eval_tag='shared'")
            && ContainsText(record.message, "fewer adopted episodes");
    }));
}

TEST_CASE("RunManager requires an interval for every Eval schedule", "[trainer][eval_schedule]")
{
    ScopedRunManagerMetricsLogger metrics_logger;
    auto factory_state = RegisterRunManagerNameTestFactories();
    auto config = MakeRunManagerNameTestConfig();
    config.Set("train.eval.[scheduled].run_mode", "eval1");
    config.Set("train.eval_schedule.[scheduled].use_background", "false");

    CHECK_THROWS_WITH(
        std::make_shared<rl::RunManager>(config),
        Catch::Matchers::ContainsSubstring("train.eval_schedule.[scheduled].interval"));
    CHECK(factory_state->creation_count == 0);
}

TEST_CASE("RunManager rejects negative Eval schedule intervals before constructing Env", "[trainer][eval_schedule]")
{
    ScopedRunManagerMetricsLogger metrics_logger;
    auto factory_state = RegisterRunManagerNameTestFactories();
    auto config = MakeRunManagerNameTestConfig();
    config.Set("train.eval.[scheduled].run_mode", "eval1");
    config.Set("train.eval_schedule.[scheduled].interval", "-1");

    CHECK_THROWS_WITH(
        std::make_shared<rl::RunManager>(config),
        Catch::Matchers::ContainsSubstring(
            "Invalid train.eval_schedule.[scheduled].interval=-1"));
    CHECK(factory_state->creation_count == 0);
}

TEST_CASE("RunManager treats a zero interval Eval schedule as definition-only", "[trainer][eval_schedule]")
{
    ScopedRunManagerMetricsLogger metrics_logger;
    anet::test::LogCaptureGuard logs(wxLOG_Info);
    auto factory_state = RegisterRunManagerNameTestFactories();
    auto config = MakeRunManagerNameTestConfig();
    config.Set("train.eval.[sleep].run_mode", "eval1");
    config.Set("train.eval_schedule.[sleep].interval", "0");
    config.Set("train.eval_schedule.[sleep].use_background", "false");

    auto manager = std::make_shared<rl::RunManager>(config);
    logs.Flush();

    REQUIRE(manager->GetStatus() == rl::RunnerStatus::RUNNING);
    CHECK(factory_state->creation_count == 1);
    CHECK_THROWS_AS(manager->GetEvalRunner("sleep"), std::out_of_range);
    CHECK(anet::test::HasRecordContaining(
        logs.Records(), wxLOG_Message, { "eval tag 'sleep': definition-only" }));
}

TEST_CASE("RunManager creates Eval runners only for active schedules", "[trainer][eval_schedule]")
{
    ScopedRunManagerMetricsLogger metrics_logger;
    anet::test::LogCaptureGuard logs(wxLOG_Info);
    auto factory_state = RegisterRunManagerNameTestFactories();
    auto config = MakeRunManagerNameTestConfig();
    config.Set("train.eval.[scheduled].run_mode", "eval1");
    config.Set("train.eval_schedule.[scheduled].interval", "7");
    config.Set("train.eval_schedule.[scheduled].use_background", "false");

    auto manager = std::make_shared<rl::RunManager>(config);
    logs.Flush();

    REQUIRE(manager->GetStatus() == rl::RunnerStatus::RUNNING);
    REQUIRE(manager->GetEvalRunner("scheduled") != nullptr);
    CHECK(manager->GetEvalRunner("scheduled")->GetBatchEnv()->GetName() == "scheduled");
    CHECK(factory_state->creation_count == 2);
    CHECK(std::ranges::find(
        factory_state->config_prefixes, "train.eval.[scheduled].env")
        != factory_state->config_prefixes.end());
    CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message, {
        "eval tag 'scheduled': scheduled (interval=7, background=false)"
    }));
}

TEST_CASE("RunManager rejects dynamic duplicate Env names before construction", "[env_name][run_manager]")
{
    ScopedRunManagerMetricsLogger metrics_logger;
    auto factory_state = RegisterRunManagerNameTestFactories();
    auto config = MakeRunManagerNameTestConfig();
    config.Set("train.eval.[configured].run_mode", "eval1");
    auto manager = std::make_shared<rl::RunManager>(config);
    REQUIRE(manager->GetStatus() == rl::RunnerStatus::RUNNING);

    const int initial_creation_count = factory_state->creation_count;
    CHECK_THROWS_WITH(
        manager->CreateEvalRunner("train"),
        Catch::Matchers::ContainsSubstring("Duplicate Env name 'train' within Run"));
    CHECK_THROWS_WITH(
        manager->CreateEvalRunner("configured"),
        Catch::Matchers::ContainsSubstring("Duplicate Env name 'configured' within Run"));
    CHECK_THROWS_WITH(
        manager->CreateEvalRunner(""),
        Catch::Matchers::ContainsSubstring("Env name must not be empty"));
    CHECK(factory_state->creation_count == initial_creation_count);

    const auto first = manager->CreateEvalRunner("dynamic");
    const int after_first_creation = factory_state->creation_count;
    CHECK_THROWS_WITH(
        manager->CreateEvalRunner("dynamic"),
        Catch::Matchers::ContainsSubstring("Duplicate Env name 'dynamic' within Run"));
    CHECK(factory_state->creation_count == after_first_creation);
    CHECK(manager->GetEvalRunner("dynamic") == first);

    const auto first_eval_panel = manager->CreateEvalRunner("EvalPanel");
    const int after_eval_panel_creation = factory_state->creation_count;
    CHECK_THROWS_WITH(
        manager->CreateEvalRunner("EvalPanel"),
        Catch::Matchers::ContainsSubstring("Duplicate Env name 'EvalPanel' within Run"));
    CHECK(factory_state->creation_count == after_eval_panel_creation);
    CHECK(manager->GetEvalRunner("EvalPanel") == first_eval_panel);
}

TEST_CASE("RunManager reserves Env names only after successful construction", "[env_name][run_manager]")
{
    ScopedRunManagerMetricsLogger metrics_logger;
    auto factory_state = RegisterRunManagerNameTestFactories();
    auto config = MakeRunManagerNameTestConfig();
    auto first_manager = std::make_shared<rl::RunManager>(config);
    REQUIRE(first_manager->GetStatus() == rl::RunnerStatus::RUNNING);

    factory_state->failing_name = "retry[0]";
    CHECK_THROWS_WITH(
        first_manager->CreateEvalRunner("retry"),
        Catch::Matchers::ContainsSubstring("Requested RunManager test Env failure: retry[0]"));
    factory_state->failing_name.reset();
    CHECK(first_manager->CreateEvalRunner("retry")->GetBatchEnv()->GetName() == "retry");

    // registryはRunManagerローカルなので、別Runでは同じnameを再利用できる。
    auto second_manager = std::make_shared<rl::RunManager>(config);
    REQUIRE(second_manager->GetStatus() == rl::RunnerStatus::RUNNING);
    CHECK(second_manager->CreateEvalRunner("retry")->GetBatchEnv()->GetName() == "retry");
}

TEST_CASE("RunnerThread forwards a worker exception once and stops", "[trainer][thread]")
{
    auto runner = std::make_shared<ThrowingRunner>();
    std::atomic<int> callback_count = 0;
    std::exception_ptr callback_exception;
    std::promise<void> callback_called;
    auto callback_future = callback_called.get_future();

    rl::RunnerThread thread(
        "ThrowingRunnerThread",
        runner,
        nullptr,
        nullptr,
        [&] {
            // worker の catch 節にある現在例外を、main thread 側で検証できる形に保存する。
            callback_exception = std::current_exception();
            callback_count.fetch_add(1);
            callback_called.set_value();
        });

    // callback の通知を待ってから join し、停止状態と転送された例外を確認する。
    thread.Start();
    REQUIRE(callback_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    thread.Stop();

    CHECK(callback_count.load() == 1);
    CHECK_FALSE(thread.IsRunning());
    REQUIRE(callback_exception != nullptr);
    CHECK_THROWS_WITH(std::rethrow_exception(callback_exception), "runner failure");
}
