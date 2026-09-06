#include "anet/catch_test.hpp"

#include "anet/env.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/observers.hpp"
#include "anet/trainer.hpp"

#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

rl::BatchState MakeState(
    const std::vector<bool>& done,
    const std::vector<bool>& truncated,
    std::optional<std::vector<bool>> episode_start = std::nullopt)
{
    const int64_t num_envs = static_cast<int64_t>(done.size());
    return rl::BatchState{
        MakeObs(num_envs),
        BoolTensor(done),
        BoolTensor(truncated),
        BoolTensor(episode_start.value_or(
            std::vector<bool>(static_cast<size_t>(num_envs), false)))
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
                std::vector<bool>(done.size(), false),
                [&]() {
                    std::vector<bool> starts(done.size());
                    for (size_t i = 0; i < done.size(); ++i) starts[i] = done[i] || truncated[i];
                    return starts;
                }()),
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
            std::vector<bool>(static_cast<size_t>(num_envs), false),
            std::vector<bool>(static_cast<size_t>(num_envs), true)))
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

class TestBatchEnv : public rl::BatchEnvBase {
public:
    TestBatchEnv(const std::string& name, int num_envs, float env_score = 0.0f)
        : rl::BatchEnvBase(name, num_envs)
        , batch_spec_{ num_envs, 1 }
        , env_score_(env_score)
    {
    }

    rl::EnvSpec GetSpec() const override { return MakeEnvSpec(); }
    rl::BatchEnvSpec GetBatchSpec() const override { return batch_spec_; }
    torch::Device GetDevice() const override { return torch::Device(torch::kCPU); }

    std::shared_ptr<const rl::BatchResetResult> Reset() override
    {
        return std::make_shared<TestResetResult>(batch_spec_.num_envs);
    }

    std::shared_ptr<const rl::BatchStepResult> Step(std::shared_ptr<rl::BatchActionInfo> action_info) override
    {
        last_action_ = action_info->GetAction().clone();
        return std::make_shared<TestStepResult>(
            std::vector<float>(static_cast<size_t>(batch_spec_.num_envs), 0.0f),
            std::vector<bool>(static_cast<size_t>(batch_spec_.num_envs), false),
            std::vector<bool>(static_cast<size_t>(batch_spec_.num_envs), false));
    }

    std::optional<float> GetScalar(const std::string& key, int64_t = -1) const override
    {
        scalar_query_count_++;
        if (key == "mean.env_score") return env_score_;
        return std::nullopt;
    }
    int GetScalarQueryCount() const { return scalar_query_count_; }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t = -1) const override { return std::nullopt; }

private:
    rl::BatchEnvSpec batch_spec_;
    float env_score_ = 0.0f;
    torch::Tensor last_action_;
    mutable int scalar_query_count_ = 0;
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

class TestBatchUpdateResult final : public rl::BatchUpdateResult {
public:
    explicit TestBatchUpdateResult(std::unordered_map<std::string, float> scalars = {})
        : scalars_(std::move(scalars))
    {
    }

    std::optional<float> GetScalar(const std::string& key, int64_t = -1) const override
    {
        scalar_query_count_++;
        const auto it = scalars_.find(key);
        return it == scalars_.end() ? std::nullopt : std::optional<float>(it->second);
    }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(
        const std::string&, int64_t = -1) const override { return std::nullopt; }

    int GetScalarQueryCount() const { return scalar_query_count_; }

private:
    std::unordered_map<std::string, float> scalars_;
    mutable int scalar_query_count_ = 0;
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
        const rl::EnvSpec&,
        rl::RunMode,
        std::optional<bool> = std::nullopt,
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

    void SetAgentScore(float value) { agent_score_ = value; }

    std::optional<float> GetScalar(const std::string& key, int64_t = -1) const override
    {
        scalar_query_count_++;
        if (key == "agent_score") return agent_score_;
        return std::nullopt;
    }
    int GetScalarQueryCount() const { return scalar_query_count_; }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t = -1) const override { return std::nullopt; }

private:
    float agent_score_ = 0.0f;
    bool use_action_info_scalar_ = false;
    float action_info_score_ = 0.0f;
    std::string actor_failure_message_;
    mutable int scalar_query_count_ = 0;
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

    void SetScalar(std::string key, float value)
    {
        scalar_key_ = std::move(key);
        scalar_value_ = value;
    }

    std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override
    {
        scalar_query_count_++;
        if (key == scalar_key_) return scalar_value_;
        return RunnerBase::GetScalar(key, index);
    }

    int GetScalarQueryCount() const { return scalar_query_count_; }

    bool FireEpisodeEnd(
        std::shared_ptr<const rl::BatchStepResult> result,
        const rl::StepCounts& event_counts)
    {
        return AccumulateAndNotifyEpisodeEnd(shared_from_this(), result, event_counts);
    }

private:
    std::string scalar_key_;
    float scalar_value_ = 0.0f;
    mutable int scalar_query_count_ = 0;
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

bool HasScalarRecordAtStep(const CapturingBackend& backend, const std::string& tag, int64_t step)
{
    for (const auto& record : backend.records) {
        if (!record.contains("type") || record["type"] != "scalar") continue;
        if (record["tag"] == tag && record["step"] == step) return true;
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
    auto env = std::make_shared<TestBatchEnv>("observer-basic", 1);
    auto target_runner = std::make_shared<TestRunner>(env, agent, notifier, "target");
    auto other_runner = std::make_shared<TestRunner>(env, agent, notifier, "other");
    auto real_observer = std::make_shared<CountingEpisodeEndObserver>();
    rl::RunnerScopedEpisodeEndObserver scoped(real_observer, target_runner);

    CHECK(target_runner->GetName() == "target");
    CHECK(other_runner->GetName() == "other");

    rl::StepCounts counts;
    rl::EpisodeEndEvent target_event{ target_runner, counts, agent, env, 0 };
    rl::EpisodeEndEvent other_event{ other_runner, counts, agent, env, 0 };

    scoped.OnEpisodeEnd(other_event);
    CHECK(real_observer->events.empty());

    scoped.OnEpisodeEnd(target_event);
    REQUIRE(real_observer->events.size() == 1);
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
    auto env = std::make_shared<TestBatchEnv>("observer-scalar", 1, 42.0f);
    auto runner = std::make_shared<TestRunner>(env, agent, notifier);

    notifier->Attach(std::make_shared<rl::MetricsLogEpisodeEndObserver>(
        "runner_reward", "mean.episode_return", rl::StepAxis::TRAIN,
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
    auto env = std::make_shared<TestBatchEnv>("observer-event", 1);
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
    valid_config.Set("metrics.scalar.[train_eps]", "mean.episode_return $runner @episode_end $train");
    valid_config.Set("metrics.scalar.[eval_eps]", "mean.episode_return $runner @session_end $eval.[eval1]");
    valid_config.Set("metrics.scalar.[eval_action]", "action_uqe_win_rate.[0] $action_info @train $eval.[eval1]");
    valid_config.Set("metrics.scalar.[eval_action_margin]", "action_uqe_margin.[0] $action_info @train $eval.[eval1]");

    rl::ObserverFactory factory(valid_config);
    auto episode_end_obs = factory.GetEpisodeEndObservers();
    REQUIRE(episode_end_obs.size() == 1);
    CHECK(episode_end_obs[0].scope == rl::RunnerScope::TRAIN);
    const auto session_end_obs = factory.GetSessionEndObservers();
    REQUIRE(session_end_obs.size() == 1);
    CHECK(session_end_obs[0].scope == rl::RunnerScope::EVAL);
    CHECK(session_end_obs[0].eval_name == "eval1");
    auto train_obs = factory.GetUpdateObservers();
    REQUIRE(train_obs.size() == 2);
    CHECK(train_obs[0].scope == rl::RunnerScope::EVAL);
    CHECK(train_obs[0].eval_name == "eval1");
    CHECK(train_obs[1].scope == rl::RunnerScope::EVAL);
    CHECK(train_obs[1].eval_name == "eval1");

    CHECK_THROWS(rl::ObserverFactory(MakeScalarConfig("mean.episode_return $runner @train $eval.[eval1]")));
    CHECK_THROWS(rl::ObserverFactory(MakeScalarConfig("mean.episode_return $runner @learn $eval.[eval1]")));
    CHECK_THROWS(rl::ObserverFactory(MakeScalarConfig("action_uqe_win_rate.[0] $action_info @learn $train")));
    CHECK_THROWS(rl::ObserverFactory(MakeScalarConfig("action_uqe_win_rate.[0] $action_info @episode_end $train")));
    CHECK_THROWS(rl::ObserverFactory(MakeScalarConfig("action_uqe_margin.[0] $action_info @learn $train")));
    CHECK_THROWS(rl::ObserverFactory(MakeScalarConfig("action_uqe_margin.[0] $action_info @episode_end $train")));
    CHECK_THROWS(rl::ObserverFactory(MakeScalarConfig("mean.episode_return $exp @episode_end $train")));
    CHECK_THROWS(rl::ObserverFactory(MakeScalarConfig("mean.episode_return $update_result @episode_end $train")));
}

TEST_CASE("ObserverFactory records the resolved step coordinate space per scalar metric", "[observer_factory][metrics_defs][observers]")
{
    anet::ConfigData config;
    config.Set("metrics.scalar.[eval_episode]",
        "$eval.[eval1] @session_end $env $exp_step mean.ep_double_suika_created");
    config.Set("metrics.scalar.[eval_action]",
        "$eval.[eval1] @train $exp_step action_uqe_win_rate.[0] $action_info $ema ema_alpha:0.01 interval:100");
    config.Set("metrics.scalar.[train_plain]", "mean.ep_step $env @train");
    config.Set("metrics.scalar.[learn_td]", "td_mean @learn");
    config.Set("metrics.scalar.[learn_td_update]", "td_mean @learn $update_result");

    rl::ObserverFactory factory(config);
    auto defs = factory.GetScalarMetricDefs();
    REQUIRE(defs.size() == 5);

    std::unordered_map<std::string, rl::ObserverFactory::ScalarMetricDef> by_tag;
    for (const auto& def : defs) by_tag.emplace(def.tag, def);

    // 同じ $eval.[eval1] かつ $exp_step でも、counts を載せる Runner は @event で変わる。
    // @session_end は呼び出し元 (train runner) の counts、@train は eval runner 自身の counts。
    const auto& eval_episode = by_tag.at("eval_episode");
    CHECK(eval_episode.step_axis == "exp_step");
    CHECK(eval_episode.runner == "train");
    CHECK(eval_episode.event == "session_end");
    CHECK(eval_episode.target == "env");
    CHECK(eval_episode.source_key == "mean.ep_double_suika_created");

    const auto& eval_action = by_tag.at("eval_action");
    CHECK(eval_action.step_axis == "exp_step");
    CHECK(eval_action.runner == "eval1");
    CHECK(eval_action.event == "train");
    CHECK(eval_action.target == "action_info");
    CHECK(eval_action.source_key == "action_uqe_win_rate.[0]");
    CHECK(eval_action.has_ema);
    CHECK(eval_action.ema_alpha == Catch::Approx(0.01));
    CHECK(eval_action.interval == 100);
    CHECK(eval_action.subscription.source_key == "action_uqe_win_rate.[0]");
    CHECK(eval_action.subscription.event == rl::EventType::TRAIN);
    REQUIRE(eval_action.subscription.target.has_value());
    CHECK(*eval_action.subscription.target == rl::EventField::ACTION_INFO);
    CHECK(eval_action.subscription.interval == 100);
    CHECK(eval_action.subscription.scope == rl::RunnerScope::EVAL);
    CHECK(eval_action.subscription.eval_name == "eval1");

    // step 軸を省略した場合の既定は event で決まる。
    const auto& train_plain = by_tag.at("train_plain");
    CHECK(train_plain.step_axis == "train_step");
    CHECK(train_plain.runner == "train");
    CHECK(train_plain.interval == 1);
    CHECK_FALSE(train_plain.has_ema);

    const auto& learn_td = by_tag.at("learn_td");
    CHECK(learn_td.step_axis == "exp_step");
    CHECK(learn_td.runner == "train");
    CHECK(learn_td.event == "learn");
    CHECK(learn_td.target.empty());

    const auto& learn_td_update = by_tag.at("learn_td_update");
    CHECK(learn_td_update.target == "update_result");
    REQUIRE(learn_td_update.subscription.target.has_value());
    CHECK(*learn_td_update.subscription.target == rl::EventField::UPDATE_RESULT);
}

TEST_CASE("ObserverFactory preserves weight norm UpdateResult subscriptions", "[observer_factory][metrics_defs][plasticity][weight_norm]")
{
    anet::ConfigData config;
    config.Set("metrics.scalar.[34_agent_plasticity/61_weight_norm_feature]",
        "plasticity_weight_norm_feature @learn $learn_step $update_result interval:100");
    config.Set("metrics.scalar.[34_agent_plasticity/62_weight_norm_readout]",
        "plasticity_weight_norm_readout @learn $learn_step $update_result interval:100");

    rl::ObserverFactory factory(config);
    const auto defs = factory.GetScalarMetricDefs();

    REQUIRE(defs.size() == 2);
    for (const auto& def : defs) {
        CHECK(def.step_axis == "learn_step");
        CHECK(def.target == "update_result");
        CHECK(def.interval == 100);
        CHECK(def.subscription.event == rl::EventType::LEARN);
        REQUIRE(def.subscription.target.has_value());
        CHECK(*def.subscription.target == rl::EventField::UPDATE_RESULT);
        CHECK(def.subscription.interval == 100);
        CHECK(def.subscription.source_key == def.source_key);
    }
}

TEST_CASE("ObserverFactory preserves policy churn baseline subscriptions", "[observer_factory][metrics_defs][policy_churn]")
{
    const std::array<std::pair<const char*, const char*>, 7> metrics = { {
        { "35_agent_churn/01_action_churn_ratio", "policy_churn_action_ratio" },
        { "35_agent_churn/02_q_delta_abs_mean", "policy_churn_q_delta_abs_mean" },
        { "35_agent_churn/03_q_delta_signed_max", "policy_churn_q_delta_signed_max" },
        { "35_agent_churn/04_q_delta_signed_min", "policy_churn_q_delta_signed_min" },
        { "35_agent_churn/11_target_policy_disagreement", "policy_churn_target_policy_disagreement" },
        { "35_agent_churn/12_target_q_delta_abs_mean", "policy_churn_target_q_delta_abs_mean" },
        { "35_agent_churn/13_target_sync_age", "policy_churn_target_sync_age" },
    } };
    anet::ConfigData config;
    for (const auto& [tag, source_key] : metrics) {
        config.Set(
            std::string("metrics.scalar.[") + tag + "]",
            std::string(source_key) + " @learn $learn_step $update_result interval:503");
    }

    rl::ObserverFactory factory(config);
    const auto defs = factory.GetScalarMetricDefs();
    REQUIRE(defs.size() == metrics.size());
    for (const auto& [tag, source_key] : metrics) {
        const auto found = std::ranges::find_if(defs, [&](const auto& def) { return def.tag == tag; });
        REQUIRE(found != defs.end());
        CHECK(found->source_key == source_key);
        CHECK(found->step_axis == "learn_step");
        CHECK(found->event == "learn");
        CHECK(found->target == "update_result");
        CHECK(found->interval == 503);
        CHECK(found->subscription.scope == rl::RunnerScope::TRAIN);
        CHECK(found->subscription.event == rl::EventType::LEARN);
        REQUIRE(found->subscription.target.has_value());
        CHECK(*found->subscription.target == rl::EventField::UPDATE_RESULT);
    }
}

TEST_CASE("ObserverFactory preserves delta-specific plasticity subscriptions", "[observer_factory][metrics_defs][plasticity][demand]")
{
    anet::ConfigData config;
    config.Set("metrics.scalar.[34_agent_plasticity/06_srank_delta_005]",
        "plasticity_srank_delta_005 @learn $learn_step $update_result interval:1000");
    config.Set("metrics.scalar.[34_agent_plasticity/49_probe_srank_ratio_delta_020]",
        "$agent plasticity_probe_srank_ratio_delta_020 @learn $learn_step interval:100");

    rl::ObserverFactory factory(config);
    const auto defs = factory.GetScalarMetricDefs();

    REQUIRE(defs.size() == 2);
    const auto actual = std::ranges::find_if(defs, [](const auto& def) {
        return def.source_key == "plasticity_srank_delta_005";
    });
    REQUIRE(actual != defs.end());
    CHECK(actual->step_axis == "learn_step");
    CHECK(actual->target == "update_result");
    CHECK(actual->subscription.interval == 1000);
    REQUIRE(actual->subscription.target.has_value());
    CHECK(*actual->subscription.target == rl::EventField::UPDATE_RESULT);

    const auto probe = std::ranges::find_if(defs, [](const auto& def) {
        return def.source_key == "plasticity_probe_srank_ratio_delta_020";
    });
    REQUIRE(probe != defs.end());
    CHECK(probe->step_axis == "learn_step");
    CHECK(probe->target == "agent");
    CHECK(probe->subscription.interval == 100);
    REQUIRE(probe->subscription.target.has_value());
    CHECK(*probe->subscription.target == rl::EventField::AGENT);
}

TEST_CASE("Scalar EMA ignores nonfinite gaps and resumes from the last finite value", "[metrics][nan][observers]")
{
    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<CapturingBackend>();
    auto* backend_raw = backend.get();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "nan_ema_test";
    anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");

    auto agent = std::make_shared<TestAgent>(std::numeric_limits<float>::quiet_NaN());
    rl::MetricsLogLearnObserver observer(
        "agent_score_ema",
        "agent_score",
        rl::StepAxis::LEARN,
        rl::EventField::AGENT,
        1,
        true,
        0.5f,
        std::nullopt);
    rl::BatchExperience experience;

    auto fire = [&](int64_t step, float value) {
        agent->SetAgentScore(value);
        rl::StepCounts counts;
        counts.learn_step = step;
        rl::LearnEvent event{ experience, nullptr, counts, agent, rl::BatchUpdateResultList{} };
        observer.OnLearn(event);
    };
    fire(1, std::numeric_limits<float>::quiet_NaN());
    fire(2, 2.0f);
    fire(3, std::numeric_limits<float>::infinity());
    fire(4, 4.0f);

    CHECK_FALSE(HasScalarRecord(*backend_raw, "agent_score_ema", 1, 0.0));
    CHECK(HasScalarRecord(*backend_raw, "agent_score_ema", 2, 2.0));
    CHECK_FALSE(HasScalarRecord(*backend_raw, "agent_score_ema", 3, 0.0));
    CHECK(HasScalarRecord(*backend_raw, "agent_score_ema", 4, 10.0 / 3.0));
    anet::MetricsLogger::Reset();
}

TEST_CASE("Automatic UpdateResult lookup keeps recognized NaN from falling through", "[metrics][nan][observers]")
{
    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<CapturingBackend>();
    auto* backend_raw = backend.get();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "nan_update_result_lookup_test";
    anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");

    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>();
    auto env = std::make_shared<TestBatchEnv>("observer-update-result", 1);
    auto runner = std::make_shared<TestRunner>(env, agent, notifier);
    auto update_result = std::make_shared<TestBatchUpdateResult>(
        std::unordered_map<std::string, float>{
            { "q_max_real_mean", std::numeric_limits<float>::quiet_NaN() },
        });
    rl::MetricsLogLearnObserver observer(
        "q_max_real_mean",
        "q_max_real_mean",
        rl::StepAxis::EXP,
        std::nullopt,
        1,
        false,
        0.01f,
        std::nullopt);

    rl::StepCounts counts;
    counts.exp_step = 100;
    rl::BatchExperience experience;
    rl::LearnEvent event{
        experience,
        runner,
        counts,
        agent,
        rl::BatchUpdateResultList{ update_result },
    };
    observer.OnLearn(event);

    CHECK(update_result->GetScalarQueryCount() == 1);
    CHECK(agent->GetScalarQueryCount() == 0);
    CHECK(runner->GetScalarQueryCount() == 0);
    CHECK(env->GetScalarQueryCount() == 0);
    CHECK_FALSE(HasScalarTag(*backend_raw, "q_max_real_mean"));
    anet::MetricsLogger::Reset();
}

TEST_CASE("Automatic UpdateResult lookup averages only finite values", "[metrics][nan][observers]")
{
    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<CapturingBackend>();
    auto* backend_raw = backend.get();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "finite_update_result_average_test";
    anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");

    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>(99.0f);
    auto env = std::make_shared<TestBatchEnv>("observer-update-result-average", 1);
    auto runner = std::make_shared<TestRunner>(env, agent, notifier);
    rl::BatchUpdateResultList update_results{
        std::make_shared<TestBatchUpdateResult>(std::unordered_map<std::string, float>{
            { "agent_score", std::numeric_limits<float>::quiet_NaN() },
        }),
        std::make_shared<TestBatchUpdateResult>(std::unordered_map<std::string, float>{
            { "agent_score", 2.0f },
        }),
        std::make_shared<TestBatchUpdateResult>(std::unordered_map<std::string, float>{
            { "agent_score", std::numeric_limits<float>::infinity() },
        }),
        std::make_shared<TestBatchUpdateResult>(std::unordered_map<std::string, float>{
            { "agent_score", 4.0f },
        }),
    };
    rl::MetricsLogLearnObserver observer(
        "finite_update_result_average",
        "agent_score",
        rl::StepAxis::EXP,
        std::nullopt,
        1,
        false,
        0.01f,
        std::nullopt);

    rl::StepCounts counts;
    counts.exp_step = 200;
    rl::BatchExperience experience;
    rl::LearnEvent event{ experience, runner, counts, agent, update_results };
    observer.OnLearn(event);

    CHECK(HasScalarRecord(*backend_raw, "finite_update_result_average", 200, 3.0));
    CHECK(agent->GetScalarQueryCount() == 0);
    CHECK(runner->GetScalarQueryCount() == 0);
    CHECK(env->GetScalarQueryCount() == 0);
    anet::MetricsLogger::Reset();
}

TEST_CASE("Automatic UpdateResult lookup falls through for an unknown key", "[metrics][observers]")
{
    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<CapturingBackend>();
    auto* backend_raw = backend.get();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "unknown_update_result_lookup_test";
    anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");

    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>(7.5f);
    auto env = std::make_shared<TestBatchEnv>("observer-update-result-unknown", 1);
    auto runner = std::make_shared<TestRunner>(env, agent, notifier);
    auto update_result1 = std::make_shared<TestBatchUpdateResult>();
    auto update_result2 = std::make_shared<TestBatchUpdateResult>();
    rl::MetricsLogLearnObserver observer(
        "unknown_update_result_fallback",
        "agent_score",
        rl::StepAxis::EXP,
        std::nullopt,
        1,
        false,
        0.01f,
        std::nullopt);

    rl::StepCounts counts;
    counts.exp_step = 300;
    rl::BatchExperience experience;
    rl::LearnEvent event{
        experience,
        runner,
        counts,
        agent,
        rl::BatchUpdateResultList{ update_result1, update_result2 },
    };
    observer.OnLearn(event);

    CHECK(update_result1->GetScalarQueryCount() == 1);
    CHECK(update_result2->GetScalarQueryCount() == 1);
    CHECK(agent->GetScalarQueryCount() == 1);
    CHECK(runner->GetScalarQueryCount() == 0);
    CHECK(env->GetScalarQueryCount() == 0);
    CHECK(HasScalarRecord(*backend_raw, "unknown_update_result_fallback", 300, 7.5));
    anet::MetricsLogger::Reset();
}

TEST_CASE("Learn-axis UpdateResult lookup distinguishes unknown from unavailable", "[metrics][nan][observers]")
{
    SECTION("unknown key falls through to Agent")
    {
        anet::MetricsLogger::Reset();
        auto backend = std::make_unique<CapturingBackend>();
        auto* backend_raw = backend.get();
        anet::MetricsLoggerConfig logger_config;
        logger_config.run_name_tmpl = "learn_unknown_update_result_test";
        anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");

        auto notifier = std::make_shared<rl::Notifier>();
        auto agent = std::make_shared<TestAgent>(8.0f);
        auto env = std::make_shared<TestBatchEnv>("observer-learn-unknown", 1);
        auto runner = std::make_shared<TestRunner>(env, agent, notifier);
        auto update_result = std::make_shared<TestBatchUpdateResult>();
        rl::MetricsLogLearnObserver observer(
            "learn_unknown_update_result",
            "agent_score",
            rl::StepAxis::LEARN,
            std::nullopt,
            1,
            false,
            0.01f,
            std::nullopt);

        rl::StepCounts counts;
        counts.learn_step = 40;
        rl::BatchExperience experience;
        rl::LearnEvent event{
            experience,
            runner,
            counts,
            agent,
            rl::BatchUpdateResultList{ update_result },
        };
        observer.OnLearn(event);

        CHECK(update_result->GetScalarQueryCount() == 1);
        CHECK(agent->GetScalarQueryCount() == 1);
        CHECK(runner->GetScalarQueryCount() == 0);
        CHECK(env->GetScalarQueryCount() == 0);
        CHECK(HasScalarRecord(*backend_raw, "learn_unknown_update_result", 40, 8.0));
        anet::MetricsLogger::Reset();
    }

    SECTION("known NaN stops lookup without logging")
    {
        anet::MetricsLogger::Reset();
        auto backend = std::make_unique<CapturingBackend>();
        auto* backend_raw = backend.get();
        anet::MetricsLoggerConfig logger_config;
        logger_config.run_name_tmpl = "learn_nan_update_result_test";
        anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");

        auto notifier = std::make_shared<rl::Notifier>();
        auto agent = std::make_shared<TestAgent>(8.0f);
        auto env = std::make_shared<TestBatchEnv>("observer-learn-nan", 1);
        auto runner = std::make_shared<TestRunner>(env, agent, notifier);
        auto update_result = std::make_shared<TestBatchUpdateResult>(
            std::unordered_map<std::string, float>{
                { "agent_score", std::numeric_limits<float>::quiet_NaN() },
            });
        rl::MetricsLogLearnObserver observer(
            "learn_nan_update_result",
            "agent_score",
            rl::StepAxis::LEARN,
            std::nullopt,
            1,
            false,
            0.01f,
            std::nullopt);

        rl::StepCounts counts;
        counts.learn_step = 50;
        rl::BatchExperience experience;
        rl::LearnEvent event{
            experience,
            runner,
            counts,
            agent,
            rl::BatchUpdateResultList{ update_result },
        };
        observer.OnLearn(event);

        CHECK(update_result->GetScalarQueryCount() == 1);
        CHECK(agent->GetScalarQueryCount() == 0);
        CHECK(runner->GetScalarQueryCount() == 0);
        CHECK(env->GetScalarQueryCount() == 0);
        CHECK_FALSE(HasScalarTag(*backend_raw, "learn_nan_update_result"));
        anet::MetricsLogger::Reset();
    }
}

TEST_CASE("Learn-axis UpdateResult lookup preserves per-result step alignment", "[metrics][nan][observers]")
{
    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<CapturingBackend>();
    auto* backend_raw = backend.get();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "learn_update_result_steps_test";
    anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");

    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>(99.0f);
    auto env = std::make_shared<TestBatchEnv>("observer-learn-steps", 1);
    auto runner = std::make_shared<TestRunner>(env, agent, notifier);
    rl::BatchUpdateResultList update_results{
        std::make_shared<TestBatchUpdateResult>(
            std::unordered_map<std::string, float>{ { "agent_score", 1.0f } }),
        std::make_shared<TestBatchUpdateResult>(std::unordered_map<std::string, float>{
            { "agent_score", std::numeric_limits<float>::quiet_NaN() },
        }),
        std::make_shared<TestBatchUpdateResult>(
            std::unordered_map<std::string, float>{ { "agent_score", 3.0f } }),
    };
    rl::MetricsLogLearnObserver observer(
        "learn_update_result_steps",
        "agent_score",
        rl::StepAxis::LEARN,
        std::nullopt,
        1,
        false,
        0.01f,
        std::nullopt);

    rl::StepCounts counts;
    counts.learn_step = 1000;
    rl::BatchExperience experience;
    rl::LearnEvent event{ experience, runner, counts, agent, update_results };
    observer.OnLearn(event);

    CHECK(HasScalarRecord(*backend_raw, "learn_update_result_steps", 1000, 1.0));
    CHECK_FALSE(HasScalarRecordAtStep(*backend_raw, "learn_update_result_steps", 1001));
    CHECK(HasScalarRecord(*backend_raw, "learn_update_result_steps", 1002, 3.0));
    CHECK(agent->GetScalarQueryCount() == 0);
    CHECK(runner->GetScalarQueryCount() == 0);
    CHECK(env->GetScalarQueryCount() == 0);
    anet::MetricsLogger::Reset();
}

TEST_CASE("Explicit UpdateResult target never queries fallback sources", "[metrics][observers]")
{
    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<CapturingBackend>();
    auto* backend_raw = backend.get();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "explicit_update_result_test";
    anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");

    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>(9.0f);
    auto env = std::make_shared<TestBatchEnv>("observer-explicit-update-result", 1);
    auto runner = std::make_shared<TestRunner>(env, agent, notifier);
    auto update_result = std::make_shared<TestBatchUpdateResult>();
    rl::MetricsLogLearnObserver observer(
        "explicit_update_result",
        "agent_score",
        rl::StepAxis::EXP,
        rl::EventField::UPDATE_RESULT,
        1,
        false,
        0.01f,
        std::nullopt);

    rl::StepCounts counts;
    counts.exp_step = 400;
    rl::BatchExperience experience;
    rl::LearnEvent event{
        experience,
        runner,
        counts,
        agent,
        rl::BatchUpdateResultList{ update_result },
    };
    observer.OnLearn(event);

    CHECK(update_result->GetScalarQueryCount() == 1);
    CHECK(agent->GetScalarQueryCount() == 0);
    CHECK(runner->GetScalarQueryCount() == 0);
    CHECK(env->GetScalarQueryCount() == 0);
    CHECK_FALSE(HasScalarTag(*backend_raw, "explicit_update_result"));
    anet::MetricsLogger::Reset();
}

TEST_CASE("Automatic UpdateResult lookup has priority over later sources", "[metrics][observers]")
{
    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<CapturingBackend>();
    auto* backend_raw = backend.get();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "update_result_priority_test";
    anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");

    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>(9.0f);
    auto env = std::make_shared<TestBatchEnv>("observer-update-result-priority", 1);
    auto runner = std::make_shared<TestRunner>(env, agent, notifier);
    runner->SetScalar("agent_score", 10.0f);
    auto update_result = std::make_shared<TestBatchUpdateResult>(
        std::unordered_map<std::string, float>{ { "agent_score", 2.0f } });
    rl::MetricsLogLearnObserver observer(
        "update_result_priority",
        "agent_score",
        rl::StepAxis::EXP,
        std::nullopt,
        1,
        false,
        0.01f,
        std::nullopt);

    rl::StepCounts counts;
    counts.exp_step = 500;
    rl::BatchExperience experience;
    rl::LearnEvent event{
        experience,
        runner,
        counts,
        agent,
        rl::BatchUpdateResultList{ update_result },
    };
    observer.OnLearn(event);

    CHECK(HasScalarRecord(*backend_raw, "update_result_priority", 500, 2.0));
    CHECK(update_result->GetScalarQueryCount() == 1);
    CHECK(agent->GetScalarQueryCount() == 0);
    CHECK(runner->GetScalarQueryCount() == 0);
    CHECK(env->GetScalarQueryCount() == 0);
    anet::MetricsLogger::Reset();
}

TEST_CASE("Empty UpdateResult list falls through to Agent", "[metrics][observers]")
{
    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<CapturingBackend>();
    auto* backend_raw = backend.get();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "empty_update_result_test";
    anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");

    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>(11.0f);
    auto env = std::make_shared<TestBatchEnv>("observer-empty-update-result", 1);
    auto runner = std::make_shared<TestRunner>(env, agent, notifier);
    rl::MetricsLogLearnObserver observer(
        "empty_update_result",
        "agent_score",
        rl::StepAxis::EXP,
        std::nullopt,
        1,
        false,
        0.01f,
        std::nullopt);

    rl::StepCounts counts;
    counts.exp_step = 600;
    rl::BatchExperience experience;
    rl::LearnEvent event{ experience, runner, counts, agent, rl::BatchUpdateResultList{} };
    observer.OnLearn(event);

    CHECK(HasScalarRecord(*backend_raw, "empty_update_result", 600, 11.0));
    CHECK(agent->GetScalarQueryCount() == 1);
    CHECK(runner->GetScalarQueryCount() == 0);
    CHECK(env->GetScalarQueryCount() == 0);
    anet::MetricsLogger::Reset();
}

TEST_CASE("Automatic scalar lookup preserves Runner and Env fallback order", "[metrics][observers]")
{
    SECTION("Runner is selected before Env")
    {
        anet::MetricsLogger::Reset();
        auto backend = std::make_unique<CapturingBackend>();
        auto* backend_raw = backend.get();
        anet::MetricsLoggerConfig logger_config;
        logger_config.run_name_tmpl = "runner_fallback_test";
        anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");

        auto notifier = std::make_shared<rl::Notifier>();
        auto agent = std::make_shared<TestAgent>();
        auto env = std::make_shared<TestBatchEnv>("observer-runner-fallback", 1, 12.0f);
        auto runner = std::make_shared<TestRunner>(env, agent, notifier);
        runner->SetScalar("runner_score", 13.0f);
        auto update_result = std::make_shared<TestBatchUpdateResult>();
        rl::MetricsLogLearnObserver observer(
            "runner_fallback",
            "runner_score",
            rl::StepAxis::EXP,
            std::nullopt,
            1,
            false,
            0.01f,
            std::nullopt);

        rl::StepCounts counts;
        counts.exp_step = 700;
        rl::BatchExperience experience;
        rl::LearnEvent event{
            experience,
            runner,
            counts,
            agent,
            rl::BatchUpdateResultList{ update_result },
        };
        observer.OnLearn(event);

        CHECK(HasScalarRecord(*backend_raw, "runner_fallback", 700, 13.0));
        CHECK(update_result->GetScalarQueryCount() == 1);
        CHECK(agent->GetScalarQueryCount() == 1);
        CHECK(runner->GetScalarQueryCount() == 1);
        CHECK(env->GetScalarQueryCount() == 0);
        anet::MetricsLogger::Reset();
    }

    SECTION("Env is selected after earlier sources return unknown")
    {
        anet::MetricsLogger::Reset();
        auto backend = std::make_unique<CapturingBackend>();
        auto* backend_raw = backend.get();
        anet::MetricsLoggerConfig logger_config;
        logger_config.run_name_tmpl = "env_fallback_test";
        anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");

        auto notifier = std::make_shared<rl::Notifier>();
        auto agent = std::make_shared<TestAgent>();
        auto env = std::make_shared<TestBatchEnv>("observer-env-fallback", 1, 14.0f);
        auto runner = std::make_shared<TestRunner>(env, agent, notifier);
        auto update_result = std::make_shared<TestBatchUpdateResult>();
        rl::MetricsLogLearnObserver observer(
            "env_fallback",
            "mean.env_score",
            rl::StepAxis::EXP,
            std::nullopt,
            1,
            false,
            0.01f,
            std::nullopt);

        rl::StepCounts counts;
        counts.exp_step = 800;
        rl::BatchExperience experience;
        rl::LearnEvent event{
            experience,
            runner,
            counts,
            agent,
            rl::BatchUpdateResultList{ update_result },
        };
        observer.OnLearn(event);

        CHECK(HasScalarRecord(*backend_raw, "env_fallback", 800, 14.0));
        CHECK(update_result->GetScalarQueryCount() == 1);
        CHECK(agent->GetScalarQueryCount() == 1);
        CHECK(runner->GetScalarQueryCount() == 1);
        CHECK(env->GetScalarQueryCount() == 1);
        anet::MetricsLogger::Reset();
    }
}

TEST_CASE("Train event keeps recognized nonfinite UpdateResult from falling through", "[metrics][nan][observers]")
{
    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<CapturingBackend>();
    auto* backend_raw = backend.get();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "train_nan_update_result_test";
    anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");

    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>(15.0f);
    auto env = std::make_shared<TestBatchEnv>("observer-train-update-result", 1);
    auto runner = std::make_shared<TestRunner>(env, agent, notifier);
    auto update_result = std::make_shared<TestBatchUpdateResult>(
        std::unordered_map<std::string, float>{
            { "agent_score", std::numeric_limits<float>::infinity() },
        });
    rl::MetricsLogTrainObserver observer(
        "train_nan_update_result",
        "agent_score",
        rl::StepAxis::TRAIN,
        std::nullopt,
        1,
        false,
        0.01f,
        std::nullopt);

    rl::StepCounts counts;
    counts.train_step = 900;
    auto action_info = std::make_shared<rl::BatchActionInfo>(
        torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kInt64)));
    auto state = MakeState({ false }, { false });
    auto step_result = std::make_shared<TestStepResult>(
        std::vector<float>{ 0.0f },
        std::vector<bool>{ false },
        std::vector<bool>{ false });
    rl::BatchExperience experience(state, action_info, FloatTensor({ 0.0f }), state);
    rl::TrainEvent event{
        experience,
        runner,
        counts,
        agent,
        rl::BatchUpdateResultList{ update_result },
        env,
        step_result,
        action_info,
    };
    observer.OnTrain(event);

    CHECK(update_result->GetScalarQueryCount() == 1);
    CHECK(agent->GetScalarQueryCount() == 0);
    CHECK(runner->GetScalarQueryCount() == 0);
    CHECK(env->GetScalarQueryCount() == 0);
    CHECK_FALSE(HasScalarTag(*backend_raw, "train_nan_update_result"));
    anet::MetricsLogger::Reset();
}

TEST_CASE("ScalarMetricDefsToJson emits the metrics.scalar.defs payload", "[observer_factory][metrics_defs][observers]")
{
    anet::ConfigData config;
    config.Set("metrics.scalar.[eval_action]",
        "$eval.[eval1] @train $exp_step action_uqe_win_rate.[0] $action_info $ema ema_alpha:0.01 interval:100 clip:2 clip:3");
    config.Set("metrics.scalar.[train_plain]", "mean.ep_step $env @train $eval.[ignored] $train");

    rl::ObserverFactory factory(config);
    auto payload = rl::ScalarMetricDefsToJson(factory.GetScalarMetricDefs());

    REQUIRE(payload.is_object());
    REQUIRE(payload.size() == 2);

    const auto& eval_action = payload.at("eval_action");
    CHECK(eval_action.at("step_axis").get<std::string>() == "exp_step");
    CHECK(eval_action.at("runner").get<std::string>() == "eval1");
    CHECK(eval_action.at("event").get<std::string>() == "train");
    CHECK(eval_action.at("target").get<std::string>() == "action_info");
    CHECK(eval_action.at("source_key").get<std::string>() == "action_uqe_win_rate.[0]");
    CHECK(eval_action.at("ema_alpha").get<double>() == Catch::Approx(0.01));
    CHECK(eval_action.at("interval").get<int>() == 100);
    CHECK(eval_action.at("scope") == "eval");
    CHECK(eval_action.at("eval_name") == "eval1");
    CHECK(eval_action.at("clip") == 3.0f);
    // Factory だけでは構築後の eval 条件は分からない。
    CHECK(eval_action.at("eval_episodes").is_null());
    CHECK(eval_action.at("num_envs").is_null());

    // EMA を使わない metric は ema_alpha を null にし、既定値を混ぜない。
    const auto& train_plain = payload.at("train_plain");
    CHECK(train_plain.at("ema_alpha").is_null());
    CHECK(train_plain.at("interval").get<int>() == 1);
    CHECK(train_plain.at("target").get<std::string>() == "env");
    // scalar の後勝ちで train に戻った場合も eval 名を漏らさない。
    CHECK(train_plain.at("scope") == "train");
    CHECK(train_plain.at("eval_name").is_null());
    CHECK(train_plain.at("eval_episodes").is_null());
    CHECK(train_plain.at("num_envs").is_null());
    CHECK(train_plain.at("clip").is_null());

    // target 未指定は null。空文字と区別する。
    anet::ConfigData learn_config;
    learn_config.Set("metrics.scalar.[learn_td]", "td_mean @learn");
    learn_config.Set("metrics.scalar.[learn_td_update]", "td_mean @learn $update_result");
    rl::ObserverFactory learn_factory(learn_config);
    auto learn_payload = rl::ScalarMetricDefsToJson(learn_factory.GetScalarMetricDefs());
    CHECK(learn_payload.at("learn_td").at("target").is_null());
    CHECK(learn_payload.at("learn_td_update").at("target").get<std::string>() == "update_result");
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
    auto inner_env = std::make_shared<TestBatchEnv>("observer-scope", 1);
    auto env = std::make_shared<rl::EvalSessionEnv>(inner_env, 1, std::vector<std::string>{});
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

TEST_CASE("Trace DSL rejects missing duplicate unknown and forbidden declarations", "[trace][observer_factory]")
{
    const std::vector<std::string> invalid = {
        "$env score", "@episode_end score", "@episode_end $env",
        "@train $env score", "@learn $env score", "@session_end $env score",
        "event:train $env score", "event:learn $env score", "event:session_end $env score",
        "@episode_end $exp score", "@episode_end $update_result score", "@episode_end $action_info score",
        "@episode_end $env mean.score", "@episode_end $env max.score",
        "@episode_end $env min.score", "@episode_end $env std.score",
        "@episode_end $env score score", "@episode_end $env $ema score",
        "@episode_end $env ema_alpha:0.1 score", "@episode_end $env clip:1 score",
        "@episode_end $env interval:1 score", "@episode_end $env key:score",
        "@episode_end @episode_end $env score", "@episode_end event:episode_end $env score",
        "@train @episode_end $env score", "@episode_end @train $env score",
        "@episode_end $env $env score", "@episode_end $env target:env score",
        "@episode_end $agent $env score", "@episode_end $exp $env score",
        "@episode_end $env $train $train score", "@episode_end $env $eval.[x] $eval.[x] score",
        "@episode_end $env $train $eval.[x] score", "@episode_end $env $eval.[x] $train score",
        "@episode_end $env $exp_step step:exp score", "@episode_end $env step:exp step_axis:exp score",
        "@episode_end $env $exp_step $learn_step score", "@episode_end $env $unknown score",
        "@unknown @episode_end $env score", "event:unknown @episode_end $env score",
        "target:unknown @episode_end $env score", "@episode_end $env step:unknown $exp_step score",
        "@episode_end $env unknown:value score", "@episode_end $env interval:no interval:1 score",
        "@episode_end $env event: score", "@episode_end $env target:env:extra score",
        "@episode_end $env $eval.[] score", "@episode_end $env $eval.[x]extra score"
    };
    for (const auto& definition : invalid) {
        CAPTURE(definition);
        anet::ConfigData config;
        config.Set("metrics.trace.[test]", definition);
        CHECK_THROWS_WITH(rl::ObserverFactory(config), Catch::Matchers::ContainsSubstring("metrics.trace.[test]"));
    }
}

TEST_CASE("Scalar and trace enforce the scope event target matrix", "[trace][observer_factory]")
{
    for (const bool eval : { false, true }) {
        for (const auto& event : { "train", "learn", "episode_end", "session_end" }) {
            for (const auto& target : { "", "$env", "$agent", "$runner", "$exp", "$update_result", "$action_info" }) {
                const std::string event_name(event), target_name(target);
                const std::string definition = std::string(eval ? "$eval.[x] " : "$train ")
                    + "@" + event_name + " " + target_name + " score";
                CAPTURE(definition);
                const bool end_target = target_name != "$exp" && target_name != "$update_result" && target_name != "$action_info";
                const bool scalar_ok = eval
                    ? ((event_name == "train" && target_name == "$action_info") || (event_name == "session_end" && end_target))
                    : (event_name == "train" || (event_name == "learn" && target_name != "$action_info")
                        || (event_name == "episode_end" && end_target));
                anet::ConfigData scalar;
                scalar.Set("metrics.scalar.[test]", definition);
                if (scalar_ok) CHECK_NOTHROW(rl::ObserverFactory(scalar));
                else CHECK_THROWS(rl::ObserverFactory(scalar));
                anet::ConfigData trace;
                trace.Set("metrics.trace.[test]", definition);
                const bool trace_ok = event_name == "episode_end"
                    && (target_name == "$env" || target_name == "$agent" || target_name == "$runner");
                if (trace_ok) CHECK_NOTHROW(rl::ObserverFactory(trace));
                else CHECK_THROWS(rl::ObserverFactory(trace));
            }
        }
    }
}

TEST_CASE("Shared metric token classification preserves scalar last assignment and defaults", "[observer_factory][trace]")
{
    anet::ConfigData config;
    config.Set("metrics.scalar.[legacy]", "first key:second third $agent target:env @learn event:train $learn_step step:exp step:unknown unknown:value");
    rl::ObserverFactory factory(config);
    const auto& def = factory.GetScalarMetricDefs().at(0);
    CHECK(def.source_key == "third");
    CHECK(def.target == "env");
    CHECK(def.event == "train");
    CHECK(def.step_axis == "exp_step");
    CHECK(def.interval == 1);
}

TEST_CASE("Train trace reads keys in declaration order and preserves nonfinite fields", "[trace][observers]")
{
    class ProbeEnv final : public TestBatchEnv {
    public:
        ProbeEnv() : TestBatchEnv("probe", 2) {}
        mutable std::vector<std::pair<std::string, int64_t>> queries;
        std::optional<float> GetScalar(const std::string& key, int64_t lane = -1) const override
        {
            queries.emplace_back(key, lane);
            if (key == "z") return std::numeric_limits<float>::quiet_NaN();
            if (key == "a") return std::numeric_limits<float>::infinity();
            if (key == "m") return 42.0f;
            return std::nullopt;
        }
    };
    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<CapturingBackend>();
    auto* captured = backend.get();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "trace_probe";
    anet::MetricsLogger::Init(std::move(backend), logger_config,
        std::filesystem::current_path() / "out" / "test-tmp" / "prd069");
    struct LoggerReset { ~LoggerReset() { anet::MetricsLogger::Reset(); } } logger_reset;
    auto env = std::make_shared<ProbeEnv>();
    auto agent = std::make_shared<TestAgent>();
    auto notifier = std::make_shared<rl::Notifier>();
    auto runner = std::make_shared<TestRunner>(env, agent, notifier);
    anet::ConfigData config;
    config.Set("metrics.trace.[ordered]", "$train @episode_end $env z a m $learn_step");
    rl::ObserverFactory factory(config);
    notifier->AttachScoped(factory.GetEpisodeEndObservers()[0].obs, runner);
    rl::StepCounts counts;
    counts.learn_step = 123;
    auto result = std::make_shared<TestStepResult>(
        std::vector<float>{ 1, 2 }, std::vector<bool>{ false, true }, std::vector<bool>{ false, false });
    runner->FireEpisodeEnd(result, counts);
    REQUIRE(env->queries == std::vector<std::pair<std::string, int64_t>>{ { "z", 1 }, { "a", 1 }, { "m", 1 } });
    std::vector<anet::json> traces;
    for (const auto& record : captured->records) {
        if (record.value("type", "") == "trace") traces.push_back(record);
    }
    REQUIRE(traces.size() == 1);
    // 本番 backend と同じ JSON シリアライズで NaN/Inf の null 化を確認する。
    const auto row = anet::json::parse(traces[0].dump());
    CHECK(row.at("data").size() == 3);
    CHECK(row.at("data").at("z").is_null());
    CHECK(row.at("data").at("a").is_null());
    CHECK(row.at("data").at("m") == 42.0f);
    CHECK(row.at("step") == 123);
    CHECK(row.at("lane") == 1);
}
