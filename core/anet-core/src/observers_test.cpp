#include "anet/catch_test.hpp"

#include "anet/metrics_logger.hpp"
#include "anet/observers.hpp"
#include "anet/trainer.hpp"

#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

namespace rl = anet::rl;

constexpr const char* kVectorKey = rl::ObsKeys::kVector;

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

torch::Tensor FloatTensor(const std::vector<float>& values)
{
    return torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32));
}

torch::Tensor FloatTensor(const std::vector<float>& values, const std::vector<int64_t>& shape)
{
    return torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32)).reshape(shape);
}

anet::TensorDict MakeObs(int64_t num_envs)
{
    return anet::TensorDict{ { kVectorKey, torch::zeros({ num_envs, 1 }, torch::kFloat32) } };
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
    spec.state_spec.obs_spec[kVectorKey] = vector_spec;
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
    TestActor(
        int64_t num_envs,
        bool use_action_info_scalar = false,
        float action_info_score = 0.0f,
        std::string failure_message = {})
        : num_envs_(num_envs)
        , use_action_info_scalar_(use_action_info_scalar)
        , action_info_score_(action_info_score)
        , failure_message_(std::move(failure_message))
    {
    }

    std::shared_ptr<rl::BatchActionInfo> MakeAction(const rl::StepCounts&, const rl::BatchState&) const override
    {
        if (!failure_message_.empty()) {
            throw std::runtime_error(failure_message_);
        }

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
    std::string failure_message_;
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
    TestAgent(
        float agent_score = 0.0f,
        bool use_action_info_scalar = false,
        float action_info_score = 0.0f,
        std::string actor_failure_message = {})
        : agent_score_(agent_score)
        , use_action_info_scalar_(use_action_info_scalar)
        , action_info_score_(action_info_score)
        , actor_failure_message_(std::move(actor_failure_message))
    {
    }

    std::shared_ptr<rl::Actor> CreateActor(
        const rl::BatchEnvSpec& batch_env_spec,
        rl::RunMode,
        bool,
        std::optional<torch::Device> = std::nullopt) const override
    {
        return std::make_shared<TestActor>(
            batch_env_spec.num_envs,
            use_action_info_scalar_,
            action_info_score_,
            actor_failure_message_);
    }

    std::shared_ptr<rl::Learner> CreateLearner() override
    {
        return std::make_shared<TestLearner>();
    }

    torch::Device GetDevice() const override { return torch::Device(torch::kCPU); }

    std::optional<float> GetScalar(const std::string& key, int64_t = -1) const override
    {
        if (key == "agent_score") return agent_score_;
        return std::nullopt;
    }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t = -1) const override { return std::nullopt; }

private:
    float agent_score_ = 0.0f;
    bool use_action_info_scalar_ = false;
    float action_info_score_ = 0.0f;
    std::string actor_failure_message_;
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

class NoopMetricsBackend final : public anet::IBackend {
public:
    void Open(const std::filesystem::path&, const std::string&) override {}
    void WriteJsonl(const anet::json&) override {}
    void Flush() override {}
};

class FixedSweepInputGenerator final : public rl::ISweepInputGenerator {
public:
    void ApplyGridSize(int width, int height) override
    {
        grid_w_ = width;
        grid_h_ = height;
    }

    std::pair<int, int> GetGridSize() const override
    {
        return { grid_w_, grid_h_ };
    }

    anet::TensorDict BuildInputTensor() override
    {
        anet::TensorDict input;
        input.Set(kVectorKey, torch::zeros({ grid_w_ * grid_h_, 2 }, torch::kFloat32));
        return input;
    }

    int64_t GetFlattenSize() const override
    {
        return 2;
    }

private:
    int grid_w_ = 2;
    int grid_h_ = 2;
};

class RecordingSweepOutputExtractor final : public rl::ISweepOutputExtractor {
public:
    void ApplyGridSize(int width, int height) override
    {
        grid_w_ = width;
        grid_h_ = height;
    }

    std::pair<int, int> GetGridSize() const override
    {
        return { grid_w_, grid_h_ };
    }

    rl::ExtractResult Extract(
        const torch::Tensor& output,
        const std::unordered_set<std::string>&) override
    {
        last_output = output.detach().cpu().clone();

        rl::ExtractResult result;
        result.grid = output.reshape({ -1 }).to(torch::kFloat32);
        return result;
    }

    torch::Tensor last_output;

private:
    int grid_w_ = 2;
    int grid_h_ = 2;
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

void RequireFlatApprox(const torch::Tensor& tensor, const std::vector<float>& expected)
{
    auto flat = tensor.detach().cpu().reshape({ -1 }).contiguous();
    REQUIRE(flat.numel() == static_cast<int64_t>(expected.size()));
    auto acc = flat.accessor<float, 1>();
    for (int64_t i = 0; i < static_cast<int64_t>(expected.size()); ++i) {
        REQUIRE(acc[i] == Catch::Approx(expected[static_cast<size_t>(i)]).margin(1.0e-5f));
    }
}

anet::ConfigData MakeScalarConfig(const std::string& value)
{
    anet::ConfigData config;
    config.Set("metrics.scalar.[test]", value);
    return config;
}

} // namespace

TEST_CASE("RunnerScopedEpisodeEndObserver only forwards target runner events", "[episode_end][observers]")
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

TEST_CASE("MetricsLogEpisodeEndObserver logs runner and env scalars", "[episode_end][metrics][observers]")
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

TEST_CASE("MetricsLogTrainObserver skips undefined action-info scalar", "[metrics][action_info][observers]")
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

TEST_CASE("ObserverFactory parses episode-end scopes and rejects unsupported combinations", "[episode_end][observer_factory][observers]")
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

TEST_CASE("SweepedHeatMapObserver passes configured output_key tensor to extractor", "[probe][sweep][observers]")
{
    anet::MetricsLogger::Reset();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "sweep_output_key_test";
    anet::MetricsLogger::Init(std::make_unique<NoopMetricsBackend>(), logger_config, "out/test-tmp");

    rl::SweepedHeatMapObserverConfig config;
    config.log_interval = 1;
    config.grid_width = 2;
    config.grid_height = 2;
    config.output_key = "score";

    auto input_gen = std::make_shared<FixedSweepInputGenerator>();
    auto output_ext = std::make_shared<RecordingSweepOutputExtractor>();
    anet::TensorDictFunction tensor_fn = [](const anet::TensorDict&) {
        anet::TensorDict output;
        output.Set("q", torch::zeros({ 4, 1 }, torch::kFloat32));
        output.Set("score", FloatTensor({ 1.0f, 2.0f, 3.0f, 4.0f }, { 4, 1 }));
        return output;
    };

    rl::SweepedHeatMapObserver observer("test.sweep", config, input_gen, tensor_fn, output_ext);
    rl::BatchExperience experience;
    rl::StepCounts counts;
    counts.learn_step = 1;
    rl::LearnEvent event{ experience, nullptr, counts, nullptr, rl::BatchUpdateResultList{} };
    observer.OnLearn(event);

    RequireFlatApprox(output_ext->last_output, { 1.0f, 2.0f, 3.0f, 4.0f });
    anet::MetricsLogger::Reset();
}

TEST_CASE("SweepedHeatMapObserver rejects missing output_key", "[probe][sweep][observers]")
{
    rl::SweepedHeatMapObserverConfig config;
    config.log_interval = 1;
    config.grid_width = 2;
    config.grid_height = 2;
    config.output_key = "missing";

    auto input_gen = std::make_shared<FixedSweepInputGenerator>();
    auto output_ext = std::make_shared<RecordingSweepOutputExtractor>();
    anet::TensorDictFunction tensor_fn = [](const anet::TensorDict&) {
        anet::TensorDict output;
        output.Set("q", torch::zeros({ 4, 1 }, torch::kFloat32));
        return output;
    };

    rl::SweepedHeatMapObserver observer("test.sweep", config, input_gen, tensor_fn, output_ext);
    rl::BatchExperience experience;
    rl::StepCounts counts;
    counts.learn_step = 1;
    rl::LearnEvent event{ experience, nullptr, counts, nullptr, rl::BatchUpdateResultList{} };

    CHECK_THROWS(observer.OnLearn(event));
}

TEST_CASE("EpisodeEvalObserver rethrows background eval failure on next learn", "[observers][eval_runner]")
{
    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>(0.0f, false, 0.0f, "forced eval failure");
    auto env = std::make_shared<TestBatchEnv>(1);
    auto runner = std::make_shared<rl::EvalRunner>(
        env,
        agent,
        notifier,
        rl::RunMode::Eval,
        false,
        std::nullopt,
        "eval1");
    rl::EpisodeEvalObserver observer(runner, 1, true);

    rl::BatchExperience experience;
    rl::StepCounts first_counts;
    first_counts.learn_step = 1;
    rl::LearnEvent first_event{ experience, nullptr, first_counts, agent, rl::BatchUpdateResultList{} };
    observer.OnLearn(first_event);

    rl::StepCounts second_counts;
    second_counts.learn_step = 2;
    rl::LearnEvent second_event{ experience, nullptr, second_counts, agent, rl::BatchUpdateResultList{} };

    bool rethrown = false;
    try {
        observer.OnLearn(second_event);
    } catch (const std::exception& e) {
        rethrown = true;
        CHECK(ContainsText(e.what(), "forced eval failure"));
    }
    CHECK(rethrown);
}
