#include "anet/catch_test.hpp"

#include "anet/env.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/observers.hpp"
#include "anet/trainer.hpp"
#include "anet/test_util.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <regex>
#include <thread>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
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
        const std::vector<bool>& truncated, bool shared = false)
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
            shared ? 1 : CountEpisodeEnds(done, truncated))
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

class TestBatchEnv final : public rl::BatchEnvBase {
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

class SessionRunnerEnv final : public rl::BatchEnvBase {
public:
    explicit SessionRunnerEnv(bool shared = false)
        : rl::BatchEnvBase("session-runner", 2, rl::RunMode::Eval), shared_(shared)
    {
    }

    rl::EnvSpec GetSpec() const override { return MakeEnvSpec(); }
    rl::BatchEnvSpec GetBatchSpec() const override
    {
        return { .num_envs = 2, .num_threads = 1,
            .episode_scope = shared_ ? rl::EpisodeScope::SHARED : rl::EpisodeScope::PER_LANE };
    }
    torch::Device GetDevice() const override { return torch::Device(torch::kCPU); }
    std::shared_ptr<const rl::BatchResetResult> Reset() override
    {
        return std::make_shared<TestResetResult>(2);
    }
    std::shared_ptr<const rl::BatchStepResult> Step(std::shared_ptr<rl::BatchActionInfo>) override
    {
        if (step_++ == 0) {
            scores_ = { 10.0f, 20.0f };
            return std::make_shared<TestStepResult>(
                std::vector<float>{ 1.0f, 2.0f },
                std::vector<bool>{ true, true },
                std::vector<bool>{ false, false }, shared_);
        }
        scores_ = { 30.0f, 40.0f };
        return std::make_shared<TestStepResult>(
            std::vector<float>{ 3.0f, 4.0f },
            std::vector<bool>{ true, true },
            std::vector<bool>{ false, false }, shared_);
    }
    std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override
    {
        if (key == "score" && shared_ && index == -1) return scores_[0];
        if (key == "score" && index >= 0) return scores_.at(static_cast<size_t>(index));
        return std::nullopt;
    }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override
    {
        return std::nullopt;
    }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(
        const std::string&, int64_t = -1) const override
    {
        return std::nullopt;
    }

private:
    bool shared_;
    int step_ = 0;
    std::vector<float> scores_;
};

class SlowSessionRunnerEnv final : public rl::BatchEnvBase {
public:
    SlowSessionRunnerEnv()
        : rl::BatchEnvBase("slow-session-runner", 2, rl::RunMode::Eval)
    {
    }

    rl::EnvSpec GetSpec() const override { return MakeEnvSpec(); }
    rl::BatchEnvSpec GetBatchSpec() const override
    {
        return {.num_envs = 2, .num_threads = 1, .episode_scope = rl::EpisodeScope::PER_LANE};
    }
    torch::Device GetDevice() const override { return torch::Device(torch::kCPU); }
    std::shared_ptr<const rl::BatchResetResult> Reset() override
    {
        return std::make_shared<TestResetResult>(2);
    }
    std::shared_ptr<const rl::BatchStepResult> Step(std::shared_ptr<rl::BatchActionInfo>) override
    {
        step_count_.fetch_add(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        return std::make_shared<TestStepResult>(
            std::vector<float>{1.0f, 2.0f},
            std::vector<bool>{true, true},
            std::vector<bool>{false, false});
    }
    std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override
    {
        if (key == "score" && index >= 0) return static_cast<float>(index + 1);
        return std::nullopt;
    }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override
    {
        return std::nullopt;
    }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(
        const std::string&, int64_t = -1) const override
    {
        return std::nullopt;
    }

    int GetStepCount() const { return step_count_.load(); }

private:
    std::atomic<int> step_count_ = 0;
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

    std::shared_ptr<rl::Actor> CreateActor(const rl::ActorRequest& request) const override
    {
        const auto& batch_env_spec = request.batch_env_spec;
        return std::make_shared<TestActor>(batch_env_spec.num_envs, use_action_info_scalar_, action_info_score_);
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
};

class TestRunner final : public rl::RunnerBase, public std::enable_shared_from_this<TestRunner> {
public:
    TestRunner(
        std::shared_ptr<rl::BatchEnv> env,
        std::shared_ptr<rl::Agent> agent,
        std::shared_ptr<rl::Notifier> notifier,
        std::string name = "test")
        : rl::RunnerBase(env, agent, notifier, rl::ActorRequest{.batch_env_spec = env->GetBatchSpec(), .env_spec = env->GetSpec(), .device = agent->GetDevice(), .seed = 123, .actor_key = "train"}, std::move(name))
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

} // namespace

TEST_CASE("RunnerBase emits per-env EpisodeEndEvent with caller counts", "[episode_end][runner]")
{
    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>();
    auto env = std::make_shared<TestBatchEnv>("episode-end-multi", 3);
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
    CHECK(observer->events[0].counts.train_step == 123);
    CHECK(observer->events[0].counts.exp_step == 456);
    CHECK(observer->events[1].env_index == 2);

    auto last_reward = runner->GetScalar("mean.episode_return");
    REQUIRE(last_reward.has_value());
    CHECK(*last_reward == Catch::Approx(2.5f));
    CHECK(runner->GetScalar("max.episode_return") == Catch::Approx(3.0f));
    CHECK(runner->GetScalar("min.episode_return") == Catch::Approx(2.0f));
    CHECK(runner->GetScalar("std.episode_return") == Catch::Approx(0.5f));

    auto non_terminal = std::make_shared<TestStepResult>(
        std::vector<float>{ 4.0f, 5.0f, 6.0f },
        std::vector<bool>{ false, false, false },
        std::vector<bool>{ false, false, false });
    CHECK_FALSE(runner->FireEpisodeEnd(non_terminal, event_counts));
    CHECK_FALSE(runner->LastStepHadEpisodeEnd());
}

TEST_CASE("RunnerBase counts completed episode steps across calls", "[episode_end][runner][episode_steps]")
{
    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>();
    auto env = std::make_shared<TestBatchEnv>("episode-steps", 2);
    auto runner = std::make_shared<TestRunner>(env, agent, notifier);
    auto observer = std::make_shared<CountingEpisodeEndObserver>();
    notifier->Attach(observer);
    const rl::StepCounts counts;

    // 未完了 Step も数え、done と truncated が同時に確定した群を集約する。
    auto pending = std::make_shared<TestStepResult>(
        std::vector<float>{ 1.0f, 2.0f }, std::vector<bool>{ false, false },
        std::vector<bool>{ false, false });
    CHECK_FALSE(runner->FireEpisodeEnd(pending, counts));
    auto completed = std::make_shared<TestStepResult>(
        std::vector<float>{ 3.0f, 4.0f }, std::vector<bool>{ true, false },
        std::vector<bool>{ false, true });
    CHECK(runner->FireEpisodeEnd(completed, counts));
    REQUIRE(observer->events.size() == 2);
    CHECK(runner->GetScalar("mean.episode_return") == Catch::Approx(5.0f));
    CHECK(runner->GetScalar("mean.episode_steps") == Catch::Approx(2.0f));
    CHECK(runner->GetScalar("max.episode_steps") == Catch::Approx(2.0f));
    CHECK(runner->GetScalar("min.episode_steps") == Catch::Approx(2.0f));
    CHECK(runner->GetScalar("std.episode_steps") == Catch::Approx(0.0f));
    CHECK_FALSE(runner->GetScalar("episode_steps").has_value());
    CHECK_FALSE(runner->GetScalar("unknown.episode_steps").has_value());

    // 完了後の次 episode はゼロから数え、未完了 Step に前回値を残さない。
    CHECK_FALSE(runner->FireEpisodeEnd(pending, counts));
    const auto unavailable = runner->GetScalar("mean.episode_steps");
    REQUIRE(unavailable.has_value());
    CHECK(std::isnan(*unavailable));
    CHECK(runner->FireEpisodeEnd(completed, counts));
    CHECK(runner->GetScalar("mean.episode_steps") == Catch::Approx(2.0f));
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
    auto env = std::make_shared<TestBatchEnv>("episode-end-single", 1);
    auto runner = std::make_shared<rl::EvalRunner>(env, agent, notifier, rl::ActorRequest{.batch_env_spec = env->GetBatchSpec(), .env_spec = env->GetSpec(), .device = agent->GetDevice(), .seed = 123, .actor_key = "eval"}, "eval1");
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

TEST_CASE("EvalRunner RunSession emits adopted episodes then one session event", "[episode_end][eval_session][runner]")
{
    const bool background = GENERATE(false, true);
    anet::test::LogCaptureGuard logs(wxLOG_Message);
    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>();
    auto inner = std::make_shared<SessionRunnerEnv>();
    auto env = std::make_shared<rl::EvalSessionEnv>(inner, 3, std::vector<std::string>{ "mean.score" });
    auto runner = std::make_shared<rl::EvalRunner>(env, agent, notifier, rl::ActorRequest{.batch_env_spec = env->GetBatchSpec(), .env_spec = env->GetSpec(), .device = agent->GetDevice(), .seed = 123, .actor_key = "eval"}, "eval1");
    auto observer = std::make_shared<CountingEpisodeEndObserver>();
    notifier->Attach(observer);

    class SessionRecorder final : public rl::SessionEndObserver {
    public:
        const CountingEpisodeEndObserver* episodes = nullptr;
        size_t episode_count_at_session = 0;
        std::vector<rl::SessionEndEvent> events;
        void OnSessionEnd(const rl::SessionEndEvent& event) override
        {
            episode_count_at_session = episodes->events.size();
            events.push_back(event);
            // 完了通知の処理時間もセッション所要時間に含まれることを確認する。
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
        std::string ToString() const override { return "SessionRecorder"; }
    };
    auto sessions = std::make_shared<SessionRecorder>();
    sessions->episodes = observer.get();
    notifier->Attach(sessions);

    rl::StepCounts event_counts;
    event_counts.train_step = 123;
    event_counts.exp_step = 456;
    {
        rl::EpisodeEvalObserver scheduler(runner, 1, background, true);
        rl::BatchExperience experience;
        event_counts.learn_step = 1;
        scheduler.OnLearn(rl::LearnEvent{ experience, nullptr, event_counts, agent, {} });
        // 次の発火で前セッションの完了通知を待ち、今回の train 座標で記録する。
        event_counts.learn_step = 2;
        event_counts.exp_step = 789;
        scheduler.OnLearn(rl::LearnEvent{ experience, nullptr, event_counts, agent, {} });
        scheduler.Shutdown(
            std::chrono::steady_clock::now() + std::chrono::seconds(5),
            rl::ShutdownMode::WAIT);
    }

    REQUIRE(observer->events.size() == 6);
    CHECK(observer->events[1].env_index == 1);
    CHECK(observer->events[2].env_index == 0);
    REQUIRE(sessions->events.size() == 2);
    CHECK(sessions->episode_count_at_session == 6);
    CHECK(sessions->events[0].env == env);
    CHECK(sessions->events[0].counts.exp_step == 456);
    CHECK(sessions->events[0].counts.learn_step == 1);
    CHECK(sessions->events[1].counts.exp_step == 789);
    CHECK(sessions->events[1].counts.learn_step == 2);
    CHECK(observer->events[3].counts.exp_step == 789);
    CHECK(observer->events[0].env == env);
    CHECK(observer->events[0].env_index == 0);
    CHECK(observer->events[0].counts.train_step == 123);
    CHECK(observer->events[0].counts.exp_step == 456);
    CHECK(runner->GetScalar("mean.episode_return") == Catch::Approx(10.0f / 3.0f));
    CHECK(runner->GetScalar("max.episode_return") == Catch::Approx(4.0f));
    CHECK(runner->GetScalar("mean.episode_steps") == Catch::Approx(1.0f));
    CHECK(env->GetScalar("mean.score") == Catch::Approx(100.0f / 3.0f));
    CHECK(runner->GetScalar("max.episode_steps") == Catch::Approx(1.0f));
    CHECK(runner->GetScalar("min.episode_steps") == Catch::Approx(1.0f));
    CHECK(runner->GetScalar("std.episode_steps") == Catch::Approx(0.0f));

    // worker 完了後に flush し、開始・終了の件数と train 側座標を突き合わせる。
    logs.Flush();
    int starts = 0;
    int ends = 0;
    int waits = 0;
    int wait_begins = 0;
    std::optional<size_t> last_start_index;
    std::optional<size_t> wait_begin_index;
    std::optional<size_t> waited_index;
    const auto& records = logs.Records();
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& record = records[i];
        if (record.message.find("eval.[eval1]: waiting for previous session") != std::string::npos) {
            ++wait_begins;
            wait_begin_index = i;
            CHECK(record.level == wxLOG_Message);
            CHECK(record.message.find("learn_step=2 exp_step=789") != std::string::npos);
        }
        if (record.message.find("eval.[eval1]: waited for previous session") != std::string::npos) {
            ++waits;
            waited_index = i;
            CHECK(record.level == wxLOG_Message);
            CHECK(record.message.find("learn_step=2 exp_step=789") != std::string::npos);
            std::smatch match;
            REQUIRE(std::regex_search(record.message, match, std::regex(R"(elapsed=([0-9]+\.[0-9]{2})s)")));
            CHECK(std::stod(match[1].str()) >= 0.03);
        }
        if (record.message.find("eval.[eval1]: session start") != std::string::npos) {
            ++starts;
            last_start_index = i;
            // 開始行は発火時点で出すので、background では前セッションの終了行を追い越す。
            if (background) {
                CHECK(ends <= starts - 1);
            } else {
                CHECK(ends == starts - 1);
            }
            CHECK(record.level == wxLOG_Message);
            CHECK(record.message.find(starts == 1 ? "learn_step=1 exp_step=456" : "learn_step=2 exp_step=789") != std::string::npos);
        }
        if (record.message.find("eval.[eval1]: session end") != std::string::npos) {
            ++ends;
            CHECK(ends <= starts);
            CHECK(record.level == wxLOG_Message);
            CHECK(record.message.find(ends == 1 ? "learn_step=1 exp_step=456" : "learn_step=2 exp_step=789") != std::string::npos);
            std::smatch mean;
            REQUIRE(std::regex_search(record.message, mean, std::regex(R"(mean.episode_return=(\S+))")));
            CHECK(std::stof(mean[1].str()) == Catch::Approx(ends == 1 ? 2.0f : 10.0f / 3.0f));
            CHECK(record.message.find(ends == 1 ? "max.episode_return=3" : "max.episode_return=4") != std::string::npos);
            CHECK(record.message.find("mean.episode_steps=1") != std::string::npos);
            CHECK(record.message.find("max.episode_steps=1") != std::string::npos);
            std::smatch match;
            REQUIRE(std::regex_search(record.message, match, std::regex(R"(elapsed=([0-9]+\.[0-9]{2})s)")));
            CHECK(std::stod(match[1].str()) >= 0.03);
        }
    }
    CHECK(starts == 2);
    CHECK(ends == 2);
    CHECK(waits == (background ? 1 : 0));
    CHECK(wait_begins == waits);
    if (background) {
        // 2 回目の発火の開始行、待ち始め、待ち終わりがこの順に並ぶ。いずれも train thread の出力。
        REQUIRE(last_start_index.has_value());
        REQUIRE(wait_begin_index.has_value());
        REQUIRE(waited_index.has_value());
        CHECK(*last_start_index < *wait_begin_index);
        CHECK(*wait_begin_index < *waited_index);
    }
}

TEST_CASE("Notifier shutdown drains a scoped background evaluation", "[prd076][eval_session][shutdown]")
{
    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<CapturingBackend>();
    auto* backend_raw = backend.get();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "prd076_shutdown";
    anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");
    struct LoggerReset { ~LoggerReset() { anet::MetricsLogger::Reset(); } } logger_reset;

    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>();
    auto env = std::make_shared<rl::EvalSessionEnv>(
        std::make_shared<SessionRunnerEnv>(), 3, std::vector<std::string>{ "mean.score" });
    auto runner = std::make_shared<rl::EvalRunner>(
        env, agent, notifier,
        rl::ActorRequest{.batch_env_spec = env->GetBatchSpec(), .env_spec = env->GetSpec(),
            .device = agent->GetDevice(), .seed = 123, .actor_key = "eval"},
        "eval1");

    class SessionRecorder final : public rl::SessionEndObserver {
    public:
        void OnSessionEnd(const rl::SessionEndEvent&) override { ++count; }
        std::string ToString() const override { return "SessionRecorder"; }
        int count = 0;
    };
    auto sessions = std::make_shared<SessionRecorder>();
    notifier->AttachScoped(sessions, runner);

    anet::ConfigData metric_config;
    metric_config.Set("metrics.scalar.[session_score]", "$eval.[eval1] @session_end $env mean.score");
    rl::ObserverFactory factory(metric_config);
    notifier->AttachScoped(factory.GetSessionEndObservers()[0].obs, runner);
    notifier->AttachScoped<rl::EpisodeEvalObserver>(runner, runner, 1, true, true);

    rl::StepCounts counts;
    counts.learn_step = 1;
    counts.exp_step = 456;
    rl::BatchExperience experience;
    notifier->Notify(rl::LearnEvent{ experience, runner, counts, agent, {} });
    notifier->Shutdown(
        std::chrono::steady_clock::now() + std::chrono::seconds(5),
        rl::ShutdownMode::WAIT);

    CHECK(sessions->count == 1);
    CHECK(HasScalarRecord(*backend_raw, "session_score", 456, 20.0));
}

TEST_CASE("Notifier cancellation stops a background evaluation at the next Step boundary", "[prd076][eval_session][shutdown]")
{
    anet::test::LogCaptureGuard logs(wxLOG_Message);
    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<CapturingBackend>();
    auto* backend_raw = backend.get();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "prd076_cancel";
    anet::MetricsLogger::Init(std::move(backend), logger_config, "C:/tmp");
    struct LoggerReset { ~LoggerReset() { anet::MetricsLogger::Reset(); } } logger_reset;

    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>();
    auto inner = std::make_shared<SlowSessionRunnerEnv>();
    auto env = std::make_shared<rl::EvalSessionEnv>(inner, 10, std::vector<std::string>{});
    auto runner = std::make_shared<rl::EvalRunner>(
        env, agent, notifier,
        rl::ActorRequest{.batch_env_spec = env->GetBatchSpec(), .env_spec = env->GetSpec(),
            .device = agent->GetDevice(), .seed = 123, .actor_key = "eval"},
        "eval1");

    auto episodes = std::make_shared<CountingEpisodeEndObserver>();
    notifier->AttachScoped(episodes, runner);
    anet::ConfigData metric_config;
    metric_config.Set("metrics.trace.[cancelled_episode]", "$eval.[eval1] @episode_end $env score");
    metric_config.Set("metrics.scalar.[cancelled_session]", "$eval.[eval1] @session_end $runner mean.episode_return");
    rl::ObserverFactory factory(metric_config);
    notifier->AttachScoped(factory.GetEpisodeEndObservers()[0].obs, runner);
    notifier->AttachScoped(factory.GetSessionEndObservers()[0].obs, runner);
    class SessionRecorder final : public rl::SessionEndObserver {
    public:
        void OnSessionEnd(const rl::SessionEndEvent&) override { ++count; }
        std::string ToString() const override { return "SessionRecorder"; }
        int count = 0;
    };
    auto sessions = std::make_shared<SessionRecorder>();
    notifier->AttachScoped(sessions, runner);
    notifier->AttachScoped<rl::EpisodeEvalObserver>(runner, runner, 1, true, true);

    rl::StepCounts counts;
    counts.learn_step = 7;
    counts.exp_step = 123;
    rl::BatchExperience experience;
    notifier->Notify(rl::LearnEvent{experience, runner, counts, agent, {}});
    const auto start_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (inner->GetStepCount() == 0 && std::chrono::steady_clock::now() < start_deadline) {
        std::this_thread::yield();
    }
    REQUIRE(inner->GetStepCount() == 1);

    notifier->Shutdown(std::chrono::steady_clock::now(), rl::ShutdownMode::CANCEL);

    CHECK(inner->GetStepCount() == 1);
    CHECK(episodes->events.size() == 2);
    CHECK(sessions->count == 0);
    CHECK(std::ranges::count_if(backend_raw->records, [](const auto& record) {
        return record.value("type", "") == "trace"
            && record.value("tag", "") == "cancelled_episode";
    }) == 2);
    CHECK_FALSE(std::ranges::any_of(backend_raw->records, [](const auto& record) {
        return record.value("type", "") == "scalar"
            && record.value("tag", "") == "cancelled_session";
    }));
    logs.Flush();
    CHECK(std::ranges::any_of(logs.Records(), [](const auto& record) {
        return record.message.find("eval.[eval1]: cancelling in-flight session on exit reason=close")
            != std::string::npos;
    }));
    CHECK(std::ranges::any_of(logs.Records(), [](const auto& record) {
        return record.message.find(
            "eval.[eval1]: session cancelled learn_step=7 exp_step=123") != std::string::npos
            && record.message.find("completed=2") != std::string::npos;
    }));
    CHECK(std::ranges::any_of(backend_raw->records, [](const auto& record) {
        return record.value("type", "") == "json"
            && record.value("tag", "") == "eval.[eval1].session_cancelled"
            && record.at("data").at("learn_step") == 7
            && record.at("data").at("exp_step") == 123
            && record.at("data").at("elapsed_sec").is_number()
            && record.at("data").at("elapsed_sec").get<double>() >= 0.0
            && record.at("data").at("completed") == 2;
    }));

    // 明示排水は完了後に複数回呼んでも副作用を繰り返さない。
    notifier->Shutdown(std::chrono::steady_clock::now(), rl::ShutdownMode::CANCEL);
}

TEST_CASE("Episode evaluation applies configured and timeout cancellation policies", "[prd076][eval_session][shutdown]")
{
    const bool timeout = GENERATE(false, true);
    anet::test::LogCaptureGuard logs(wxLOG_Message);
    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>();
    auto inner = std::make_shared<SlowSessionRunnerEnv>();
    auto env = std::make_shared<rl::EvalSessionEnv>(inner, 10, std::vector<std::string>{});
    auto runner = std::make_shared<rl::EvalRunner>(
        env, agent, notifier,
        rl::ActorRequest{.batch_env_spec = env->GetBatchSpec(), .env_spec = env->GetSpec(),
            .device = agent->GetDevice(), .seed = 123, .actor_key = "eval"},
        "eval1");
    rl::EpisodeEvalObserver observer(runner, 1, true, timeout);

    rl::StepCounts counts;
    counts.learn_step = 1;
    rl::BatchExperience experience;
    observer.OnLearn(rl::LearnEvent{experience, runner, counts, agent, {}});
    const auto start_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (inner->GetStepCount() == 0 && std::chrono::steady_clock::now() < start_deadline) {
        std::this_thread::yield();
    }
    REQUIRE(inner->GetStepCount() == 1);

    observer.Shutdown(
        timeout ? std::chrono::steady_clock::now() - std::chrono::seconds(1)
                : std::chrono::steady_clock::now() + std::chrono::seconds(5),
        rl::ShutdownMode::WAIT);

    CHECK(inner->GetStepCount() == 1);
    logs.Flush();
    const std::string expected_reason = timeout ? "reason=timeout" : "reason=config";
    CHECK(std::ranges::any_of(logs.Records(), [&](const auto& record) {
        return record.message.find("eval.[eval1]: cancelling in-flight session on exit")
            != std::string::npos
            && record.message.find(expected_reason) != std::string::npos;
    }));
}

TEST_CASE("All scoped observer wrappers forward shutdown queries and calls", "[prd076][shutdown][observer]")
{
    class MultiObserver final : public rl::TrainObserver,
        public rl::LearnObserver,
        public rl::EpisodeEndObserver,
        public rl::SessionEndObserver {
    public:
        void OnTrain(const rl::TrainEvent&) override {}
        void OnLearn(const rl::LearnEvent&) override {}
        void OnEpisodeEnd(const rl::EpisodeEndEvent&) override {}
        void OnSessionEnd(const rl::SessionEndEvent&) override {}
        void Shutdown(std::chrono::steady_clock::time_point, rl::ShutdownMode) override
        {
            ++shutdown_count;
        }
        bool WillBlockOnShutdown() const override { return true; }
        std::string ToString() const override { return "MultiObserver"; }
        int shutdown_count = 0;
    };

    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>();
    auto env = std::make_shared<TestBatchEnv>("observer-wrapper", 1);
    auto runner = std::make_shared<rl::EvalRunner>(
        env, agent, notifier,
        rl::ActorRequest{.batch_env_spec = env->GetBatchSpec(), .env_spec = env->GetSpec(),
            .device = agent->GetDevice(), .seed = 123, .actor_key = "eval"},
        "eval1");
    auto observer = notifier->AttachScoped<MultiObserver>(runner);

    CHECK(notifier->WillBlockOnShutdown());
    notifier->Shutdown(std::chrono::steady_clock::now(), rl::ShutdownMode::WAIT);
    CHECK(observer->shutdown_count == 4);
}

TEST_CASE("Episode evaluation reports whether shutdown wait will block", "[prd076][shutdown][observer]")
{
    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>();
    auto foreground_env = std::make_shared<rl::EvalSessionEnv>(
        std::make_shared<SessionRunnerEnv>(), 2, std::vector<std::string>{});
    auto foreground_runner = std::make_shared<rl::EvalRunner>(
        foreground_env, agent, notifier,
        rl::ActorRequest{.batch_env_spec = foreground_env->GetBatchSpec(),
            .env_spec = foreground_env->GetSpec(), .device = agent->GetDevice(),
            .seed = 123, .actor_key = "eval"},
        "foreground");
    rl::EpisodeEvalObserver foreground(foreground_runner, 1, false, true);
    CHECK_FALSE(foreground.WillBlockOnShutdown());
    foreground.Shutdown(
        std::chrono::steady_clock::now() + std::chrono::seconds(1),
        rl::ShutdownMode::WAIT);

    auto completed_env = std::make_shared<rl::EvalSessionEnv>(
        std::make_shared<SessionRunnerEnv>(), 2, std::vector<std::string>{});
    auto completed_runner = std::make_shared<rl::EvalRunner>(
        completed_env, agent, notifier,
        rl::ActorRequest{.batch_env_spec = completed_env->GetBatchSpec(),
            .env_spec = completed_env->GetSpec(), .device = agent->GetDevice(),
            .seed = 126, .actor_key = "eval"},
        "completed");
    rl::EpisodeEvalObserver completed(completed_runner, 1, true, true);
    rl::BatchExperience experience;
    rl::StepCounts counts;
    counts.learn_step = 1;
    completed.OnLearn(rl::LearnEvent{experience, completed_runner, counts, agent, {}});
    const auto completion_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (completed.WillBlockOnShutdown()
        && std::chrono::steady_clock::now() < completion_deadline) {
        std::this_thread::yield();
    }
    CHECK_FALSE(completed.WillBlockOnShutdown());
    completed.Shutdown(completion_deadline, rl::ShutdownMode::WAIT);

    auto slow_inner = std::make_shared<SlowSessionRunnerEnv>();
    auto slow_env = std::make_shared<rl::EvalSessionEnv>(
        slow_inner, 10, std::vector<std::string>{});
    auto slow_runner = std::make_shared<rl::EvalRunner>(
        slow_env, agent, notifier,
        rl::ActorRequest{.batch_env_spec = slow_env->GetBatchSpec(), .env_spec = slow_env->GetSpec(),
            .device = agent->GetDevice(), .seed = 124, .actor_key = "eval"},
        "slow");
    rl::EpisodeEvalObserver waiting(slow_runner, 1, true, true);
    waiting.OnLearn(rl::LearnEvent{experience, slow_runner, counts, agent, {}});
    const auto start_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (slow_inner->GetStepCount() == 0 && std::chrono::steady_clock::now() < start_deadline) {
        std::this_thread::yield();
    }
    REQUIRE(slow_inner->GetStepCount() == 1);
    CHECK(waiting.WillBlockOnShutdown());
    waiting.Shutdown(std::chrono::steady_clock::now(), rl::ShutdownMode::CANCEL);
    CHECK_FALSE(waiting.WillBlockOnShutdown());

    auto discard_inner = std::make_shared<SlowSessionRunnerEnv>();
    auto discard_env = std::make_shared<rl::EvalSessionEnv>(
        discard_inner, 10, std::vector<std::string>{});
    auto discard_runner = std::make_shared<rl::EvalRunner>(
        discard_env, agent, notifier,
        rl::ActorRequest{.batch_env_spec = discard_env->GetBatchSpec(), .env_spec = discard_env->GetSpec(),
            .device = agent->GetDevice(), .seed = 125, .actor_key = "eval"},
        "discard");
    rl::EpisodeEvalObserver discard(discard_runner, 1, true, false);
    discard.OnLearn(rl::LearnEvent{experience, discard_runner, counts, agent, {}});
    const auto discard_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (discard_inner->GetStepCount() == 0
        && std::chrono::steady_clock::now() < discard_deadline) {
        std::this_thread::yield();
    }
    REQUIRE(discard_inner->GetStepCount() == 1);
    CHECK_FALSE(discard.WillBlockOnShutdown());
    discard.Shutdown(std::chrono::steady_clock::now(), rl::ShutdownMode::WAIT);
}

TEST_CASE("Trace DSL records adopted episode values in JSONL before the next Step", "[trace][eval_session][metrics]")
{
    const bool background = GENERATE(false, true);
    const bool shared = GENERATE(false, true);
    anet::ConfigData config;
    config.Set("metrics.trace.[episode]", "$eval.[eval1] @episode_end $env score");
    config.Set("metrics.scalar.[episode]", "$eval.[eval1] @session_end $env mean.score");
    rl::ObserverFactory factory(config);
    const auto observers = factory.GetEpisodeEndObservers();
    REQUIRE(observers.size() == 1);

    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "prd069-trace";
    const auto jsonl_path = root / "runs" / "trace_test" / "metrics.jsonl";
    anet::MetricsLogger::Reset();
    std::filesystem::remove(jsonl_path);
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "trace_test";
    anet::MetricsLogger::Init(std::make_unique<anet::JsonlBackend>(), logger_config, root);
    struct LoggerReset { ~LoggerReset() { anet::MetricsLogger::Reset(); } } logger_reset;

    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>();
    auto env = std::make_shared<rl::EvalSessionEnv>(
        std::make_shared<SessionRunnerEnv>(shared), 3, std::vector<std::string>{ "mean.score" });
    auto runner = std::make_shared<rl::EvalRunner>(env, agent, notifier, rl::ActorRequest{.batch_env_spec = env->GetBatchSpec(), .env_spec = env->GetSpec(), .device = agent->GetDevice(), .seed = 123, .actor_key = "eval"}, "eval1");
    notifier->AttachScoped(observers[0].obs, runner);
    notifier->AttachScoped(factory.GetSessionEndObservers()[0].obs, runner);
    rl::StepCounts counts;
    counts.exp_step = 456;
    // 通常の終了経路と同様に明示排水してから出力を読む。
    {
        rl::EpisodeEvalObserver scheduler(runner, 1, background, true);
        rl::BatchExperience experience;
        counts.learn_step = 1;
        scheduler.OnLearn(rl::LearnEvent{ experience, nullptr, counts, agent, {} });
        scheduler.Shutdown(
            std::chrono::steady_clock::now() + std::chrono::seconds(5),
            rl::ShutdownMode::WAIT);
    }
    anet::MetricsLogger::Instance()->Flush();

    std::ifstream input(jsonl_path);
    std::vector<anet::json> rows;
    int scalar_count = 0;
    for (std::string line; std::getline(input, line);) {
        const auto row = anet::json::parse(line);
        if (row.at("type") == "trace") rows.push_back(row);
        if (row.at("type") == "scalar") {
            CHECK(rows.size() == 3);
            CHECK(row.at("tag") == "episode");
            CHECK(row.at("step") == 456);
            CHECK(row.at("value").get<float>() == Catch::Approx(shared ? 70.0f / 3 : 20.0f));
            scalar_count++;
        }
    }
    REQUIRE(rows.size() == 3);
    CHECK(scalar_count == 1);
    for (size_t i = 0; i < rows.size(); ++i) {
        CHECK(rows[i].size() == 5);
        CHECK(rows[i].at("tag") == "episode");
        CHECK(rows[i].at("step") == 456);
        CHECK(rows[i].at("lane") == (shared ? -1 : (i == 1 ? 1 : 0)));
        CHECK(rows[i].at("data").at("score") == static_cast<float>(shared ? (i == 0 ? 10 : 30) : (i + 1) * 10));
    }
    CHECK_FALSE(std::filesystem::exists(root / "runs" / "trace_test" / "json" / "episode_456.json"));
}

TEST_CASE("Background trace observer failure reaches the next learn callback", "[trace][eval_session][metrics]")
{
    auto notifier = std::make_shared<rl::Notifier>();
    auto agent = std::make_shared<TestAgent>();
    auto env = std::make_shared<rl::EvalSessionEnv>(
        std::make_shared<SessionRunnerEnv>(), 1, std::vector<std::string>{});
    auto runner = std::make_shared<rl::EvalRunner>(env, agent, notifier, rl::ActorRequest{.batch_env_spec = env->GetBatchSpec(), .env_spec = env->GetSpec(), .device = agent->GetDevice(), .seed = 123, .actor_key = "eval"}, "eval1");
    anet::ConfigData config;
    config.Set("metrics.trace.[broken]", "$eval.[eval1] @episode_end $env unknown_score");
    rl::ObserverFactory factory(config);
    notifier->AttachScoped(factory.GetEpisodeEndObservers()[0].obs, runner);
    rl::EpisodeEvalObserver scheduler(runner, 1, true, true);
    rl::BatchExperience experience;
    rl::StepCounts counts;
    counts.learn_step = 1;
    scheduler.OnLearn(rl::LearnEvent{ experience, nullptr, counts, agent, {} });
    counts.learn_step = 2;
    CHECK_THROWS_WITH(scheduler.OnLearn(rl::LearnEvent{ experience, nullptr, counts, agent, {} }),
        Catch::Matchers::ContainsSubstring("tag='broken' key='unknown_score' lane=0 target=env"));
}
