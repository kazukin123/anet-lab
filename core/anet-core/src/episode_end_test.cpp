#include "catch.hpp"

#include "anet/metrics_logger.hpp"
#include "anet/observers.hpp"
#include "anet/trainer.hpp"

#include <cmath>
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

namespace {

namespace rl = anet::rl;

torch::Tensor BoolTensor(const std::vector<bool>& values)
{
    auto tensor = torch::empty({ static_cast<int64_t>(values.size()) }, torch::TensorOptions().dtype(torch::kBool));
    for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
        tensor[i].fill_(values[static_cast<size_t>(i)]);
    }
    return tensor;
}

torch::Tensor FloatTensor(const std::vector<float>& values)
{
    return torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32));
}

anet::TensorDict MakeObs(int64_t num_envs)
{
    return anet::TensorDict{ { rl::ObsKeys::kVector, torch::zeros({ num_envs, 1 }, torch::kFloat32) } };
}

rl::BatchState MakeState(const std::vector<bool>& done, const std::vector<bool>& truncated)
{
    const int64_t num_envs = static_cast<int64_t>(done.size());
    return rl::BatchState{
        MakeObs(num_envs),
        BoolTensor(done),
        BoolTensor(truncated),
        BoolTensor(std::vector<bool>(static_cast<size_t>(num_envs), false))
    };
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
    spec.reward_range = { -100.0f, 100.0f };
    return spec;
}

class TestStepResult final : public rl::BatchStepResult {
public:
    TestStepResult(
        const std::vector<float>& rewards,
        const std::vector<bool>& done,
        const std::vector<bool>& truncated)
        : rl::BatchStepResult(
            FloatTensor(rewards),
            MakeState(done, truncated),
            MakeState(
                std::vector<bool>(done.size(), false),
                std::vector<bool>(done.size(), false)),
            static_cast<uint32_t>(rewards.size()),
            CountEpisodeEnds(done, truncated))
        , num_envs_(static_cast<int>(rewards.size()))
    {
    }

    std::vector<rl::AuxData> GetAuxDataList(int env_index = -1) const override
    {
        if (env_index >= 0) return { rl::AuxData{} };
        return std::vector<rl::AuxData>(static_cast<size_t>(num_envs_));
    }

private:
    static uint32_t CountEpisodeEnds(const std::vector<bool>& done, const std::vector<bool>& truncated)
    {
        uint32_t count = 0;
        for (size_t i = 0; i < done.size(); ++i) {
            if (done[i] || truncated[i]) count++;
        }
        return count;
    }

    int num_envs_ = 0;
};

class TestResetResult final : public rl::BatchResetResult {
public:
    explicit TestResetResult(int64_t num_envs)
        : rl::BatchResetResult(MakeState(
            std::vector<bool>(static_cast<size_t>(num_envs), false),
            std::vector<bool>(static_cast<size_t>(num_envs), false)))
        , num_envs_(static_cast<int>(num_envs))
    {
    }

    std::vector<rl::AuxData> GetAuxDataList(int env_index = -1) const override
    {
        if (env_index >= 0) return { rl::AuxData{} };
        return std::vector<rl::AuxData>(static_cast<size_t>(num_envs_));
    }

private:
    int num_envs_ = 0;
};

class TestBatchEnv final : public rl::BatchEnv {
public:
    explicit TestBatchEnv(int num_envs, float env_score = 0.0f)
        : batch_spec_{ num_envs, 1 }
        , env_score_(env_score)
    {
    }

    rl::EnvSpec GetSpec() const override { return MakeEnvSpec(); }
    rl::BatchEnvSpec GetBatchSpec() const override { return batch_spec_; }
    torch::Device GetDevice() const override { return torch::Device(torch::kCPU); }

    std::shared_ptr<const rl::BatchResetResult> Reset(rl::RunMode = rl::RunMode::Train) override
    {
        return std::make_shared<TestResetResult>(batch_spec_.num_envs);
    }

    std::shared_ptr<const rl::BatchStepResult> Step(std::shared_ptr<rl::BatchActionInfo> action_info, rl::RunMode = rl::RunMode::Train) override
    {
        last_action_ = action_info->GetAction().clone();
        return std::make_shared<TestStepResult>(
            std::vector<float>(static_cast<size_t>(batch_spec_.num_envs), 0.0f),
            std::vector<bool>(static_cast<size_t>(batch_spec_.num_envs), false),
            std::vector<bool>(static_cast<size_t>(batch_spec_.num_envs), false));
    }

    torch::Tensor GetLastAction() const { return last_action_; }

    std::optional<float> GetScalar(const std::string& key, int64_t = -1) const override
    {
        if (key == "mean.env_score") return env_score_;
        return std::nullopt;
    }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t = -1) const override { return std::nullopt; }

private:
    rl::BatchEnvSpec batch_spec_;
    float env_score_ = 0.0f;
    torch::Tensor last_action_;
};

class TestActionInfo final : public rl::BatchActionInfo, public anet::ModuleBase {
public:
    TestActionInfo(torch::Tensor action, const anet::TensorDict& info, const rl::AuxData& aux, float score)
        : rl::BatchActionInfo(std::move(action), info, aux)
        , score_(score)
    {
    }

    std::shared_ptr<rl::BatchActionInfo> WithAction(torch::Tensor action) const override
    {
        return std::make_shared<TestActionInfo>(std::move(action), info_, aux_, score_);
    }

    std::optional<float> GetScalar(const std::string& key, int64_t = -1) const override
    {
        if (key == "action_info_score") return score_;
        return std::nullopt;
    }

private:
    float score_ = 0.0f;
};

class TestActor final : public rl::Actor {
public:
    explicit TestActor(int64_t num_envs, bool use_action_info_scalar = false, float action_info_score = 0.0f)
        : num_envs_(num_envs)
        , use_action_info_scalar_(use_action_info_scalar)
        , action_info_score_(action_info_score)
    {
    }

    std::shared_ptr<rl::BatchActionInfo> MakeAction(const rl::StepCounts&, const rl::BatchState&) const override
    {
        auto action = torch::zeros({ num_envs_ }, torch::kInt64);
        if (use_action_info_scalar_) {
            return std::make_shared<TestActionInfo>(action, anet::TensorDict{}, rl::AuxData{}, action_info_score_);
        }
        return std::make_shared<rl::BatchActionInfo>(action);
    }

    void Sync() override {}

private:
    int64_t num_envs_ = 1;
    bool use_action_info_scalar_ = false;
    float action_info_score_ = 0.0f;
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
    explicit TestAgent(float agent_score = 0.0f, bool use_action_info_scalar = false, float action_info_score = 0.0f)
        : agent_score_(agent_score)
        , use_action_info_scalar_(use_action_info_scalar)
        , action_info_score_(action_info_score)
    {
    }

    std::shared_ptr<rl::Actor> CreateActor(
        const rl::BatchEnvSpec& batch_env_spec,
        rl::RunMode,
        bool,
        std::optional<torch::Device> = std::nullopt) const override
    {
        return std::make_shared<TestActor>(batch_env_spec.num_envs, use_action_info_scalar_, action_info_score_);
    }

    std::shared_ptr<rl::Learner> CreateLearner() override
    {
        return std::make_shared<TestLearner>();
    }

    std::optional<float> GetScalar(const std::string& key, int64_t = -1) const override
    {
        if (key == "agent_score") return agent_score_;
        return std::nullopt;
    }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<anet::TensorFunction> GetTensorFunction(const std::string&) override { return std::nullopt; }

private:
    float agent_score_ = 0.0f;
    bool use_action_info_scalar_ = false;
    float action_info_score_ = 0.0f;
};

class TestRunner final : public rl::RunnerBase, public std::enable_shared_from_this<TestRunner> {
public:
    TestRunner(
        std::shared_ptr<rl::BatchEnv> env,
        std::shared_ptr<rl::Agent> agent,
        std::shared_ptr<rl::Notifier> notifier,
        std::string name = "test")
        : rl::RunnerBase(env, agent, notifier, rl::RunMode::Train, false, std::nullopt, std::move(name))
    {
    }

    rl::StepCounts DoStep() override { return step_counts_; }
    void Shutdown() override {}

    bool FireEpisodeEnd(
        std::shared_ptr<const rl::BatchStepResult> result,
        const rl::StepCounts& event_counts)
    {
        return AccumulateAndNotifyEpisodeEnd(shared_from_this(), result, event_counts);
    }
};

class CountingEpisodeEndObserver final : public rl::EpisodeEndObserver {
public:
    void OnEpisodeEnd(const rl::EpisodeEndEvent& event) override
    {
        events.push_back(event);
    }

    std::string ToString() const override
    {
        return "CountingEpisodeEndObserver";
    }

    std::vector<rl::EpisodeEndEvent> events;
};

class CapturingBackend final : public anet::IBackend {
public:
    void Open(const std::filesystem::path&, const std::string&) override {}
    void WriteJsonl(const anet::json& obj) override { records.push_back(obj); }
    void Flush() override {}

    std::vector<anet::json> records;
};

bool HasScalarRecord(const CapturingBackend& backend, const std::string& tag, int64_t step, double value)
{
    for (const auto& record : backend.records) {
        if (!record.contains("type") || record["type"] != "scalar") continue;
        if (record["tag"] != tag) continue;
        if (record["step"] != step) continue;
        if (std::abs(record["value"].get<double>() - value) > 1e-5) continue;
        return true;
    }
    return false;
}

bool HasScalarTag(const CapturingBackend& backend, const std::string& tag)
{
    for (const auto& record : backend.records) {
        if (!record.contains("type") || record["type"] != "scalar") continue;
        if (record["tag"] != tag) continue;
        return true;
    }
    return false;
}

anet::ConfigData MakeScalarConfig(const std::string& value)
{
    anet::ConfigData config;
    config.Set("metrics.scalar.[test]", value);
    return config;
}

} // namespace

TEST_CASE("RunnerScopedEpisodeEndObserver only forwards target runner events", "[episode_end][observer]")
{
    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>();
    auto env = std::make_shared<TestBatchEnv>(1);
    auto target_runner = std::make_shared<TestRunner>(env, agent, notifier, "target");
    auto other_runner = std::make_shared<TestRunner>(env, agent, notifier, "other");
    auto real_observer = std::make_shared<CountingEpisodeEndObserver>();
    rl::RunnerScopedEpisodeEndObserver scoped(real_observer, target_runner);

    CHECK(target_runner->GetName() == "target");
    CHECK(other_runner->GetName() == "other");

    rl::StepCounts counts;
    rl::EpisodeEndEvent target_event{ target_runner, counts, agent, env, 0, 1.0f };
    rl::EpisodeEndEvent other_event{ other_runner, counts, agent, env, 0, 2.0f };

    scoped.OnEpisodeEnd(other_event);
    CHECK(real_observer->events.empty());

    scoped.OnEpisodeEnd(target_event);
    REQUIRE(real_observer->events.size() == 1);
    CHECK(real_observer->events[0].eps_total_reward == Catch::Approx(1.0f));
}

TEST_CASE("RunnerBase emits per-env EpisodeEndEvent with caller counts", "[episode_end][runner]")
{
    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>();
    auto env = std::make_shared<TestBatchEnv>(3);
    auto runner = std::make_shared<TestRunner>(env, agent, notifier);
    auto observer = std::make_shared<CountingEpisodeEndObserver>();
    notifier->Attach(observer);

    rl::StepCounts event_counts;
    event_counts.train_step = 123;
    event_counts.exp_step = 456;

    auto result = std::make_shared<TestStepResult>(
        std::vector<float>{ 1.0f, 2.0f, 3.0f },
        std::vector<bool>{ false, true, false },
        std::vector<bool>{ false, false, true });

    CHECK(runner->FireEpisodeEnd(result, event_counts));
    REQUIRE(observer->events.size() == 2);

    CHECK(observer->events[0].env_index == 1);
    CHECK(observer->events[0].eps_total_reward == Catch::Approx(2.0f));
    CHECK(observer->events[0].counts.train_step == 123);
    CHECK(observer->events[0].counts.exp_step == 456);
    CHECK(observer->events[1].env_index == 2);
    CHECK(observer->events[1].eps_total_reward == Catch::Approx(3.0f));

    auto last_reward = runner->GetScalar(rl::Runner::EPS_TOTAL_REWARD);
    REQUIRE(last_reward.has_value());
    CHECK(*last_reward == Catch::Approx(2.5f));

    auto non_terminal = std::make_shared<TestStepResult>(
        std::vector<float>{ 4.0f, 5.0f, 6.0f },
        std::vector<bool>{ false, false, false },
        std::vector<bool>{ false, false, false });
    CHECK_FALSE(runner->FireEpisodeEnd(non_terminal, event_counts));
    CHECK_FALSE(runner->LastStepHadEpisodeEnd());
}

TEST_CASE("MetricsLogEpisodeEndObserver logs runner and env scalars", "[episode_end][metrics]")
{
    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<CapturingBackend>();
    auto* backend_raw = backend.get();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "episode_end_test";
    anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");

    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>();
    auto env = std::make_shared<TestBatchEnv>(1, 42.0f);
    auto runner = std::make_shared<TestRunner>(env, agent, notifier);

    notifier->Attach(std::make_shared<rl::MetricsLogEpisodeEndObserver>(
        "runner_reward", rl::Runner::EPS_TOTAL_REWARD, rl::StepAxis::TRAIN,
        rl::EventField::RUNNER, 1, false, 0.01f, std::nullopt));
    notifier->Attach(std::make_shared<rl::MetricsLogEpisodeEndObserver>(
        "env_score", "mean.env_score", rl::StepAxis::TRAIN,
        rl::EventField::ENV, 1, false, 0.01f, std::nullopt));

    rl::StepCounts event_counts;
    event_counts.train_step = 77;
    auto result = std::make_shared<TestStepResult>(
        std::vector<float>{ 9.0f },
        std::vector<bool>{ true },
        std::vector<bool>{ false });

    runner->FireEpisodeEnd(result, event_counts);

    CHECK(HasScalarRecord(*backend_raw, "runner_reward", 77, 9.0));
    CHECK(HasScalarRecord(*backend_raw, "env_score", 77, 42.0));
    anet::MetricsLogger::Reset();
}

TEST_CASE("MetricsLogTrainObserver skips undefined action-info scalar", "[metrics][action_info]")
{
    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<CapturingBackend>();
    auto* backend_raw = backend.get();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "action_info_metrics_test";
    anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");

    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>();
    auto env = std::make_shared<TestBatchEnv>(1);
    auto runner = std::make_shared<TestRunner>(env, agent, notifier);
    auto observer = std::make_shared<rl::MetricsLogTrainObserver>(
        "noop_uqe_win_rate",
        "action_uqe_win_rate.[0]",
        rl::StepAxis::TRAIN,
        rl::EventField::ACTION_INFO,
        1,
        false,
        0.01f,
        std::nullopt);

    rl::StepCounts counts;
    counts.train_step = 7;
    auto action_info = std::make_shared<rl::BatchActionInfo>(
        torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kInt64)));
    auto state = MakeState({ false }, { false });
    auto result = std::make_shared<TestStepResult>(
        std::vector<float>{ 0.0f },
        std::vector<bool>{ false },
        std::vector<bool>{ false });
    rl::BatchExperience exp(state, action_info, FloatTensor({ 0.0f }), state);
    rl::TrainEvent event{ exp, runner, counts, agent, rl::BatchUpdateResultList(), env, result, action_info };

    observer->OnTrain(event);

    CHECK_FALSE(HasScalarTag(*backend_raw, "noop_uqe_win_rate"));
    anet::MetricsLogger::Reset();
}

TEST_CASE("EvalRunner forced action keeps derived action-info scalars", "[metrics][action_info][eval_runner]")
{
    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<CapturingBackend>();
    auto* backend_raw = backend.get();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "forced_action_info_test";
    anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");

    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>(0.0f, true, 12.5f);
    auto env = std::make_shared<TestBatchEnv>(1);
    auto runner = std::make_shared<rl::EvalRunner>(
        env,
        agent,
        notifier,
        rl::RunMode::Eval,
        false,
        std::nullopt,
        "eval1");
    notifier->Attach(std::make_shared<rl::MetricsLogTrainObserver>(
        "action_info_score",
        "action_info_score",
        rl::StepAxis::TRAIN,
        rl::EventField::ACTION_INFO,
        1,
        false,
        0.01f,
        std::nullopt));

    rl::StepCounts event_counts;
    runner->DoStep(5, event_counts);

    REQUIRE(env->GetLastAction().defined());
    REQUIRE(torch::equal(env->GetLastAction(), torch::tensor({ 5 }, torch::TensorOptions().dtype(torch::kInt64))));
    REQUIRE(HasScalarRecord(*backend_raw, "action_info_score", 1, 12.5));
    anet::MetricsLogger::Reset();
}

TEST_CASE("ObserverFactory parses episode-end scopes and rejects unsupported combinations", "[episode_end][observer_factory]")
{
    anet::ConfigData valid_config;
    valid_config.Set("metrics.scalar.[train_eps]", "eps_total_reward $runner @episode_end $train");
    valid_config.Set("metrics.scalar.[eval_eps]", "eps_total_reward $runner @episode_end $eval.[eval1]");
    valid_config.Set("metrics.scalar.[eval_action]", "action_uqe_win_rate.[0] $action_info @train $eval.[eval1]");
    valid_config.Set("metrics.scalar.[eval_action_margin]", "action_uqe_margin.[0] $action_info @train $eval.[eval1]");

    rl::ObserverFactory factory(valid_config);
    auto episode_end_obs = factory.GetEpisodeEndObservers();
    REQUIRE(episode_end_obs.size() == 2);
    CHECK(episode_end_obs[0].scope == rl::RunnerScope::TRAIN);
    CHECK(episode_end_obs[1].scope == rl::RunnerScope::EVAL);
    CHECK(episode_end_obs[1].eval_name == "eval1");
    auto train_obs = factory.GetUpdateObservers();
    REQUIRE(train_obs.size() == 2);
    CHECK(train_obs[0].scope == rl::RunnerScope::EVAL);
    CHECK(train_obs[0].eval_name == "eval1");
    CHECK(train_obs[1].scope == rl::RunnerScope::EVAL);
    CHECK(train_obs[1].eval_name == "eval1");

    CHECK_THROWS(rl::ObserverFactory(MakeScalarConfig("eps_total_reward $runner @train $eval.[eval1]")));
    CHECK_THROWS(rl::ObserverFactory(MakeScalarConfig("eps_total_reward $runner @learn $eval.[eval1]")));
    CHECK_THROWS(rl::ObserverFactory(MakeScalarConfig("action_uqe_win_rate.[0] $action_info @learn $train")));
    CHECK_THROWS(rl::ObserverFactory(MakeScalarConfig("action_uqe_win_rate.[0] $action_info @episode_end $train")));
    CHECK_THROWS(rl::ObserverFactory(MakeScalarConfig("action_uqe_margin.[0] $action_info @learn $train")));
    CHECK_THROWS(rl::ObserverFactory(MakeScalarConfig("action_uqe_margin.[0] $action_info @episode_end $train")));
    CHECK_THROWS(rl::ObserverFactory(MakeScalarConfig("eps_total_reward $exp @episode_end $train")));
    CHECK_THROWS(rl::ObserverFactory(MakeScalarConfig("eps_total_reward $update_result @episode_end $train")));
}
