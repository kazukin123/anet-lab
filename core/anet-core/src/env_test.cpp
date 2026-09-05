#include "anet/catch_test.hpp"

#include "anet/env.hpp"
#include "anet/rl.hpp"
#include "anet/test_util.hpp"

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace {

namespace rl = anet::rl;

static_assert(std::is_abstract_v<rl::SingleDiscreteEnv>);
static_assert(std::is_abstract_v<rl::BatchEnv>);
static_assert(std::is_base_of_v<rl::SingleDiscreteEnv, rl::SingleDiscreteEnvBase>);
static_assert(std::is_base_of_v<rl::BatchEnv, rl::BatchEnvBase>);

rl::EnvSpec MakeNameTestEnvSpec()
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

class NameTestSingleEnv final : public rl::SingleDiscreteEnvBase {
public:
    explicit NameTestSingleEnv(
        const std::string& name,
        rl::RunMode run_mode = rl::RunMode::Train,
        std::optional<anet::ConfigData> config_data = std::nullopt)
        : rl::SingleDiscreteEnvBase(name, run_mode, std::move(config_data))
    {
    }

    rl::EnvSpec GetSpec() const override { return MakeNameTestEnvSpec(); }
    std::shared_ptr<const rl::SingleResetResult> Reset() override { return nullptr; }
    std::shared_ptr<const rl::SingleStepResult> Step(int64_t) override { return nullptr; }
    std::optional<float> GetScalar(const std::string&, int64_t) const override { return std::nullopt; }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(
        const std::string&, int64_t) const override
    {
        return std::nullopt;
    }

    void LogInfo(const std::string& body) { log.info() << body; }
    void LogInfoFromConst(const std::string& body) const { log.info() << body; }
};

class NameTestEnvConfig final : public anet::Config {
public:
    int limit_step = 0;

    NameTestEnvConfig(const anet::ConfigData& config_data, const std::string& config_prefix)
        : Config(config_data, "NameTestSingleEnv", config_prefix)
    {
        ANET_READ_CONFIG(config_data, limit_step);
    }
};

class RecordingSingleEnvFactory final : public rl::SingleDiscreteEnvFactory {
public:
    std::shared_ptr<rl::SingleDiscreteEnv> CreateSingleEnv(
        const anet::ConfigData& config_data, const torch::Device&, const std::string& name,
        std::optional<anet::seed_t>, rl::RunMode run_mode, const std::string& config_prefix) override
    {
        names_.push_back(name);
        const NameTestEnvConfig config(config_data, config_prefix);
        return std::make_shared<NameTestSingleEnv>(name, run_mode, config.GetScopedConfigData());
    }

    std::string GetTargetEnvClassId() const override { return "NameTestSingleEnv"; }
    const std::vector<std::string>& GetNames() const { return names_; }

private:
    std::vector<std::string> names_;
};

class NameTestBatchEnv final : public rl::BatchEnvBase {
public:
    NameTestBatchEnv(
        std::string name, int num_envs, rl::RunMode run_mode = rl::RunMode::Train)
        : rl::BatchEnvBase(std::move(name), num_envs, run_mode)
        , batch_spec_{ num_envs, 1 }
    {
    }

    rl::EnvSpec GetSpec() const override { return {}; }
    rl::BatchEnvSpec GetBatchSpec() const override { return batch_spec_; }
    torch::Device GetDevice() const override { return torch::Device(torch::kCPU); }

    std::shared_ptr<const rl::BatchResetResult> Reset() override { return nullptr; }
    std::shared_ptr<const rl::BatchStepResult> Step(
        std::shared_ptr<rl::BatchActionInfo>) override
    {
        return nullptr;
    }

    std::optional<float> GetScalar(const std::string&, int64_t) const override { return std::nullopt; }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(
        const std::string&, int64_t) const override
    {
        return std::nullopt;
    }

    void LogInfo(const std::string& body) { log.info() << body; }

private:
    rl::BatchEnvSpec batch_spec_;
};

class RecordingBatchEnvFactory final : public rl::BatchEnvFactory {
public:
    std::shared_ptr<rl::BatchEnv> CreateBatchEnv(
        const anet::ConfigData&, const torch::Device&, const std::string& name,
        std::optional<anet::seed_t>, int num_envs, rl::RunMode run_mode,
        const std::string&) override
    {
        return std::make_shared<NameTestBatchEnv>(name, num_envs, run_mode);
    }

    std::string GetTargetEnvClassId() const override { return "NameTestNativeBatchEnv"; }
};

rl::BatchState MakeEpisodeState(
    const std::vector<bool>& done,
    const std::vector<bool>& truncated,
    const std::vector<bool>& episode_start);
rl::PlainBatchStepResult MakeEpisodeStep(
    const std::vector<float>& rewards,
    const std::vector<bool>& done,
    const std::vector<bool>& truncated,
    const std::vector<bool>& continue_episode_start,
    uint32_t n_episode_end);

struct SessionStep {
    std::vector<float> rewards;
    std::vector<bool> done;
    std::vector<bool> truncated;
    std::vector<float> scores;
};

class ScriptedSessionEnv final : public rl::BatchEnvBase {
public:
    ScriptedSessionEnv(int num_envs, rl::EpisodeScope episode_scope, std::vector<SessionStep> steps)
        : rl::BatchEnvBase("session", num_envs, rl::RunMode::Eval)
        , batch_spec_{
            .num_envs = num_envs,
            .num_threads = 1,
            .episode_scope = episode_scope,
        }
        , steps_(std::move(steps))
    {
    }

    rl::EnvSpec GetSpec() const override { return MakeNameTestEnvSpec(); }
    rl::BatchEnvSpec GetBatchSpec() const override { return batch_spec_; }
    torch::Device GetDevice() const override { return torch::Device(torch::kCPU); }

    std::shared_ptr<const rl::BatchResetResult> Reset() override
    {
        reset_count_++;
        return std::make_shared<rl::PlainBatchResetResult>(
            MakeEpisodeState(
                std::vector<bool>(static_cast<size_t>(batch_spec_.num_envs), false),
                std::vector<bool>(static_cast<size_t>(batch_spec_.num_envs), false),
                std::vector<bool>(static_cast<size_t>(batch_spec_.num_envs), true)),
            std::vector<rl::AuxData>(static_cast<size_t>(batch_spec_.num_envs)));
    }

    std::shared_ptr<const rl::BatchStepResult> Step(std::shared_ptr<rl::BatchActionInfo>) override
    {
        REQUIRE(step_index_ < steps_.size());
        const auto& step = steps_[step_index_++];
        current_scores_ = step.scores;
        uint32_t completed = 0;
        if (batch_spec_.episode_scope == rl::EpisodeScope::PER_LANE) {
            for (size_t lane = 0; lane < step.done.size(); ++lane) {
                if (step.done[lane] || step.truncated[lane]) completed++;
            }
        } else if (step.done[0] || step.truncated[0]) {
            completed = 1;
        }
        std::vector<bool> episode_start(step.done.size());
        for (size_t lane = 0; lane < step.done.size(); ++lane) {
            episode_start[lane] = step.done[lane] || step.truncated[lane];
        }
        return std::make_shared<rl::PlainBatchStepResult>(MakeEpisodeStep(
            step.rewards, step.done, step.truncated, episode_start, completed));
    }

    std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override
    {
        if (key == "score") {
            if (index < 0) {
                if (current_scores_.empty()) return std::nullopt;
                float sum = 0.0f;
                for (const float score : current_scores_) sum += score;
                return sum / static_cast<float>(current_scores_.size());
            }
            return current_scores_.at(static_cast<size_t>(index));
        }
        if (key == "raw") return 42.0f + static_cast<float>(index);
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

    int GetResetCount() const { return reset_count_; }

private:
    rl::BatchEnvSpec batch_spec_;
    std::vector<SessionStep> steps_;
    size_t step_index_ = 0;
    std::vector<float> current_scores_;
    int reset_count_ = 0;
};

rl::BatchState MakeEpisodeState(
    const std::vector<bool>& done,
    const std::vector<bool>& truncated,
    const std::vector<bool>& episode_start)
{
    const auto num_envs = static_cast<int64_t>(done.size());
    auto make_bool_tensor = [](const std::vector<bool>& values) {
        auto tensor = torch::empty(
            { static_cast<int64_t>(values.size()) }, torch::TensorOptions().dtype(torch::kBool));
        for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
            tensor[i].fill_(values[static_cast<size_t>(i)]);
        }
        return tensor;
    };
    return {
        anet::TensorDict{ { rl::ObsKeys::kVector, torch::zeros({ num_envs, 1 }) } },
        make_bool_tensor(done),
        make_bool_tensor(truncated),
        make_bool_tensor(episode_start)
    };
}

rl::PlainBatchStepResult MakeEpisodeStep(
    const std::vector<float>& rewards,
    const std::vector<bool>& done,
    const std::vector<bool>& truncated,
    const std::vector<bool>& continue_episode_start,
    uint32_t n_episode_end)
{
    const auto num_envs = static_cast<int64_t>(rewards.size());
    return {
        torch::tensor(rewards, torch::TensorOptions().dtype(torch::kFloat32)),
        MakeEpisodeState(done, truncated, std::vector<bool>(rewards.size(), false)),
        MakeEpisodeState(
            std::vector<bool>(rewards.size(), false),
            std::vector<bool>(rewards.size(), false),
            continue_episode_start),
        static_cast<uint32_t>(num_envs),
        n_episode_end,
        std::vector<rl::AuxData>(rewards.size())
    };
}

} // namespace

TEST_CASE("BatchEnvSpec serializes episode scope", "[env][episode_scope]")
{
    CHECK(rl::BatchEnvSpec{ .num_envs = 2, .num_threads = 1 }.ToJson()["episode_scope"] == "per_lane");
    CHECK(rl::BatchEnvSpec{
        .num_envs = 2,
        .num_threads = 1,
        .episode_scope = rl::EpisodeScope::SHARED
    }.ToJson()["episode_scope"] == "shared");
}

TEST_CASE("Episode structure validation distinguishes per-lane and shared groups", "[env][episode_scope]")
{
    NameTestBatchEnv env("episode", 2);
    const rl::BatchEnvSpec per_lane{ .num_envs = 2, .num_threads = 1 };
    const rl::BatchEnvSpec shared{
        .num_envs = 2,
        .num_threads = 1,
        .episode_scope = rl::EpisodeScope::SHARED
    };

    auto both_flags = MakeEpisodeStep({ 1.0f, 2.0f }, { true, false }, { true, false }, { true, false }, 1);
    CHECK(rl::ValidateEpisodeStructure(env.GetName(), per_lane, both_flags) == std::vector<int64_t>{ 0 });

    auto shared_end = MakeEpisodeStep({ 1.0f, 2.0f }, { true, true }, { false, false }, { true, true }, 1);
    CHECK(rl::ValidateEpisodeStructure(env.GetName(), shared, shared_end) == std::vector<int64_t>{ 0 });

    auto both_terminal_flags = MakeEpisodeStep(
        { 1.0f, 2.0f }, { true, true }, { true, true }, { true, true }, 1);
    CHECK(rl::ValidateEpisodeStructure(env.GetName(), shared, both_terminal_flags)
        == std::vector<int64_t>{ 0 });

    rl::PlainBatchResetResult stale_reset(
        MakeEpisodeState({ false, false }, { false, false }, { true, false }),
        std::vector<rl::AuxData>(2));
    CHECK_THROWS_WITH(
        rl::ValidateEpisodeStructure(env.GetName(), per_lane, stale_reset),
        Catch::Matchers::ContainsSubstring("Reset must return a fresh episode group"));

    auto mismatched_mask = MakeEpisodeStep({ 1.0f, 2.0f }, { true, false }, { false, false }, { true, false }, 1);
    CHECK_THROWS_WITH(
        rl::ValidateEpisodeStructure(env.GetName(), shared, mismatched_mask),
        Catch::Matchers::ContainsSubstring("mask=done"));

    auto mismatched_continue = MakeEpisodeStep({ 1.0f, 2.0f }, { true, true }, { false, false }, { true, false }, 1);
    CHECK_THROWS_WITH(
        rl::ValidateEpisodeStructure(env.GetName(), shared, mismatched_continue),
        Catch::Matchers::ContainsSubstring("mask=continue_state.episode_start"));

    auto mismatched_per_lane_continue = MakeEpisodeStep(
        { 1.0f, 2.0f }, { false, true }, { false, false }, { false, false }, 1);
    CHECK_THROWS_WITH(
        rl::ValidateEpisodeStructure(env.GetName(), per_lane, mismatched_per_lane_continue),
        Catch::Matchers::ContainsSubstring("Episode continuation structure mismatch")
        && Catch::Matchers::ContainsSubstring("env='episode'")
        && Catch::Matchers::ContainsSubstring("group=1")
        && Catch::Matchers::ContainsSubstring("lane=1")
        && Catch::Matchers::ContainsSubstring("mask=continue_state.episode_start")
        && Catch::Matchers::ContainsSubstring("expected=1")
        && Catch::Matchers::ContainsSubstring("actual=0"));

    auto wrong_count = MakeEpisodeStep({ 1.0f, 2.0f }, { true, true }, { false, false }, { true, true }, 2);
    CHECK_THROWS_WITH(
        rl::ValidateEpisodeStructure(env.GetName(), shared, wrong_count),
        Catch::Matchers::ContainsSubstring("n_episode_end"));
}

TEST_CASE("EpisodeReturnAccumulator aggregates by episode group", "[env][episode_return]")
{
    rl::EpisodeReturnAccumulator per_lane({ .num_envs = 2, .num_threads = 1 });
    auto first = MakeEpisodeStep({ 1.0f, 10.0f }, { false, false }, { false, false }, { false, false }, 0);
    CHECK(per_lane.Add(first).empty());
    auto second = MakeEpisodeStep({ 2.0f, 20.0f }, { true, false }, { false, false }, { true, false }, 1);
    const auto completed = per_lane.Add(second);
    REQUIRE(completed.size() == 1);
    CHECK(completed[0].group_index == 0);
    CHECK(completed[0].episode_return == 3.0f);

    rl::EpisodeReturnAccumulator shared({
        .num_envs = 2,
        .num_threads = 1,
        .episode_scope = rl::EpisodeScope::SHARED
    });
    auto shared_step = MakeEpisodeStep({ 3.0f, 4.0f }, { true, true }, { false, false }, { true, true }, 1);
    const auto shared_completed = shared.Add(shared_step);
    REQUIRE(shared_completed.size() == 1);
    CHECK(shared_completed[0].group_index == 0);
    CHECK(shared_completed[0].episode_return == 7.0f);
}

TEST_CASE("EvalSessionEnv dynamically grants exactly N per-lane episodes", "[env][eval_session]")
{
    auto inner = std::make_shared<ScriptedSessionEnv>(
        2,
        rl::EpisodeScope::PER_LANE,
        std::vector<SessionStep>{
            { { 1.0f, 10.0f }, { true, true }, { false, false }, { 2.0f, 4.0f } },
            { { 2.0f, 20.0f }, { true, true }, { false, false }, { 6.0f, 8.0f } },
        });
    rl::EvalSessionEnv env(inner, 3, { "mean.score" });

    CHECK_FALSE(env.GetSessionResult().has_value());
    const auto reset = env.Reset();
    CHECK(inner->GetResetCount() == 1);
    CHECK_FALSE(env.GetSessionResult().has_value());
    env.Step(nullptr);
    CHECK(env.LastAdoptedGroups() == std::vector<int64_t>{ 0, 1 });
    CHECK_FALSE(env.GetSessionResult().has_value());
    env.Step(nullptr);
    CHECK(env.LastAdoptedGroups() == std::vector<int64_t>{ 0 });

    const auto result = env.GetSessionResult();
    REQUIRE(result.has_value());
    CHECK(result->episode_returns == std::vector<float>{ 1.0f, 10.0f, 2.0f });
    CHECK(env.GetScalar("mean.score") == 4.0f);
    CHECK(env.GetScalar("raw", 1) == 43.0f);

    // 全 group が fresh なら cached continue_state を空 AuxData とともに再利用する。
    const auto reused = env.Reset();
    CHECK(env.LastAdoptedGroups().empty());
    CHECK(inner->GetResetCount() == 1);
    CHECK(reused->GetAuxDataList().size() == 2);
    CHECK(reused->GetAuxDataList()[0].empty());
    CHECK(torch::equal(reused->state.episode_start, torch::tensor({ true, true })));
    CHECK_FALSE(env.GetSessionResult().has_value());
}

TEST_CASE("EvalSessionEnv supports shared episodes and resets a partially fresh batch", "[env][eval_session]")
{
    auto shared_inner = std::make_shared<ScriptedSessionEnv>(
        2,
        rl::EpisodeScope::SHARED,
        std::vector<SessionStep>{
            { { 1.0f, 2.0f }, { true, true }, { false, false }, { 0.25f, 0.75f } },
            { { 3.0f, 4.0f }, { true, true }, { false, false }, { 0.5f, 1.5f } },
        });
    rl::EvalSessionEnv shared(shared_inner, 2, { "mean.score" });
    shared.Reset();
    shared.Step(nullptr);
    CHECK(shared.LastAdoptedGroups() == std::vector<int64_t>{ 0 });
    shared.Step(nullptr);
    CHECK(shared.LastAdoptedGroups() == std::vector<int64_t>{ 0 });
    REQUIRE(shared.GetSessionResult().has_value());
    CHECK(shared.GetSessionResult()->episode_returns == std::vector<float>{ 3.0f, 7.0f });
    CHECK(shared.GetScalar("mean.score") == 0.75f);

    auto partial_inner = std::make_shared<ScriptedSessionEnv>(
        4,
        rl::EpisodeScope::PER_LANE,
        std::vector<SessionStep>{
            { { 1.0f, 2.0f, 3.0f, 4.0f },
              { true, true, false, false },
              { false, false, false, false },
              { 1.0f, 2.0f, 3.0f, 4.0f } },
        });
    rl::EvalSessionEnv partial(partial_inner, 2, {});
    partial.Reset();
    partial.Step(nullptr);
    REQUIRE(partial.GetSessionResult().has_value());
    partial.Reset();
    CHECK(partial_inner->GetResetCount() == 2);
}

TEST_CASE("EvalSessionEnv waits for every adopted per-lane episode", "[env][eval_session]")
{
    auto inner = std::make_shared<ScriptedSessionEnv>(
        4,
        rl::EpisodeScope::PER_LANE,
        std::vector<SessionStep>{
            { { 1.0f, 10.0f, 100.0f, 1000.0f },
              { true, false, true, false },
              { false, false, false, false },
              { 1.0f, 10.0f, 100.0f, 1000.0f } },
            { { 2.0f, 20.0f, 200.0f, 2000.0f },
              { false, false, false, true },
              { false, false, false, false },
              { 2.0f, 20.0f, 200.0f, 2000.0f } },
            { { 3.0f, 30.0f, 300.0f, 3000.0f },
              { false, true, false, false },
              { false, false, false, false },
              { 3.0f, 30.0f, 300.0f, 3000.0f } },
        });
    rl::EvalSessionEnv env(inner, 2, {});

    env.Reset();
    env.Step(nullptr);
    CHECK(env.LastAdoptedGroups() == std::vector<int64_t>{ 0 });
    CHECK_FALSE(env.GetSessionResult().has_value());
    env.Step(nullptr);
    CHECK(env.LastAdoptedGroups().empty());
    CHECK_FALSE(env.GetSessionResult().has_value());
    env.Step(nullptr);
    CHECK(env.LastAdoptedGroups() == std::vector<int64_t>{ 1 });

    REQUIRE(env.GetSessionResult().has_value());
    CHECK(env.GetSessionResult()->episode_returns == std::vector<float>{ 1.0f, 60.0f });
}

TEST_CASE("EvalSessionEnv keeps the N=1 single-lane trace and scalar identity", "[env][eval_session]")
{
    auto inner = std::make_shared<ScriptedSessionEnv>(
        1,
        rl::EpisodeScope::PER_LANE,
        std::vector<SessionStep>{
            { { 2.5f }, { true }, { false }, { 9.0f } },
        });
    rl::EvalSessionEnv env(inner, 1, { "score" });

    const auto reset = env.Reset();
    CHECK(inner->GetResetCount() == 1);
    CHECK(reset->state.obs.At(rl::ObsKeys::kVector).item<float>() == 0.0f);

    const auto step = env.Step(nullptr);
    CHECK(env.LastAdoptedGroups() == std::vector<int64_t>{ 0 });
    CHECK(step->reward.item<float>() == 2.5f);
    CHECK(step->next_state.done.item<bool>());
    CHECK_FALSE(step->next_state.truncated.item<bool>());
    CHECK(step->continue_state.episode_start.item<bool>());
    REQUIRE(env.GetSessionResult().has_value());
    CHECK(env.GetSessionResult()->episode_returns == std::vector<float>{ 2.5f });
    CHECK(env.GetScalar("score") == 9.0f);
}

TEST_CASE("BatchEnv exposes immutable human-readable lane names", "[env_name]")
{
    NameTestBatchEnv env("train", 3);

    CHECK(env.GetName() == "train");
    CHECK(env.GetEnvName(0) == "train[0]");
    CHECK(env.GetEnvName(1) == "train[1]");
    CHECK(env.GetEnvName(2) == "train[2]");

    CHECK_THROWS_WITH(env.GetEnvName(-1), Catch::Matchers::ContainsSubstring("lane_index=-1"));
    CHECK_THROWS_WITH(env.GetEnvName(3), Catch::Matchers::ContainsSubstring("lane_index=3"));
}

TEST_CASE("Env name and BatchEnv size fail fast when invalid", "[env_name]")
{
    CHECK_THROWS_WITH(NameTestBatchEnv("", 1), Catch::Matchers::ContainsSubstring("must not be empty"));
    CHECK_THROWS_WITH(NameTestBatchEnv("invalid", 0), Catch::Matchers::ContainsSubstring("num_envs=0"));
}

TEST_CASE("Batch wrappers pass stable lane names to every single Env", "[env_name]")
{
    const std::vector<std::string> expected_names = { "batch[0]", "batch[1]", "batch[2]" };

    auto vectorized_factory = std::make_shared<RecordingSingleEnvFactory>();
    rl::VectorizedDiscreteBatchEnv vectorized(
        anet::EmptyConfigData, vectorized_factory, "batch", 3, torch::Device(torch::kCPU), 1);
    CHECK(vectorized.GetName() == "batch");
    CHECK(vectorized_factory->GetNames() == expected_names);

    auto single_worker_factory = std::make_shared<RecordingSingleEnvFactory>();
    rl::ThreadPoolDiscreteEnv single_worker(
        anet::EmptyConfigData,
        single_worker_factory,
        "batch",
        3,
        torch::Device(torch::kCPU),
        std::make_shared<anet::PinnedThreadPool>(1, "env-name-single-worker"),
        1);
    CHECK(single_worker_factory->GetNames() == expected_names);

    auto multi_worker_factory = std::make_shared<RecordingSingleEnvFactory>();
    rl::ThreadPoolDiscreteEnv multi_worker(
        anet::EmptyConfigData,
        multi_worker_factory,
        "batch",
        3,
        torch::Device(torch::kCPU),
        std::make_shared<anet::PinnedThreadPool>(2, "env-name-multi-worker"),
        1);
    CHECK(multi_worker_factory->GetNames() == expected_names);
}

TEST_CASE("Env bases bind immutable names to protected loggers", "[env_name][logger]")
{
    NameTestSingleEnv single("train[2]");
    const NameTestSingleEnv& const_single = single;
    NameTestBatchEnv batch("eval", 2);
    anet::test::LogCaptureGuard logs;

    single.LogInfo("single-body");
    const_single.LogInfoFromConst("const-body");
    batch.LogInfo("batch-body");
    logs.Flush();

    REQUIRE(logs.Records().size() == 3);
    CHECK(logs.Records()[0].message == "train[2]: single-body");
    CHECK(logs.Records()[1].message == "train[2]: const-body");
    CHECK(logs.Records()[2].message == "eval: batch-body");
}

TEST_CASE("Env exposes its immutable injected config snapshot", "[env][config]")
{
    anet::ConfigData injected_config;
    injected_config.Set("NameTestSingleEnv.limit_step", 123);
    NameTestSingleEnv env("train[0]", rl::RunMode::Train, injected_config);

    const auto actual = env.GetConfigData();
    REQUIRE(actual.has_value());
    CHECK(actual->Get("NameTestSingleEnv.limit_step") == "123");
}

TEST_CASE("Batch wrapper exposes children-inclusive effective config", "[env][config]")
{
    anet::ConfigData config_data;
    config_data.Set("NameTestSingleEnv.limit_step", 10);
    config_data.Set("train.eval.[eval1].env.limit_step", 20);

    auto factory = std::make_shared<RecordingSingleEnvFactory>();
    rl::VectorizedDiscreteBatchEnv env(
        config_data, factory, "eval1", 2, torch::Device(torch::kCPU), 1,
        rl::RunMode::Eval1, "train.eval.[eval1].env");

    const auto actual = env.GetConfigData();
    REQUIRE(actual.has_value());
    CHECK(actual->Get<int>("train.eval.[eval1].env.limit_step") == 20);
    CHECK_FALSE(actual->Has("NameTestSingleEnv.limit_step"));
}

TEST_CASE("EnvSpec checks state and action contracts while ignoring reference info", "[env][spec]")
{
    auto expected = MakeNameTestEnvSpec();
    auto compatible = expected;
    expected.info["dataset"] = "train";
    expected.state_spec.info["note"] = "expected";
    expected.action_spec.info["note"] = "expected";
    compatible.info["dataset"] = "eval";
    compatible.state_spec.info["note"] = "compatible";
    compatible.action_spec.info["note"] = "compatible";
    compatible.reward_range = { -100.0f, 100.0f };

    CHECK_NOTHROW(expected.CheckSameStateActionSpec(compatible));

    auto incompatible_state = compatible;
    incompatible_state.state_spec.obs_spec.at(rl::ObsKeys::kVector).num_classes = 2;
    CHECK_THROWS(expected.CheckSameStateActionSpec(incompatible_state));

    auto incompatible_action = compatible;
    incompatible_action.action_spec.value_labels = { "left", "right" };
    CHECK_THROWS(expected.CheckSameStateActionSpec(incompatible_action));
}

TEST_CASE("EnvRepository dispatches a per-class native batch factory variant", "[env][factory_variant]")
{
    rl::EnvRepository::Instance().Regist(std::make_shared<RecordingBatchEnvFactory>());
    auto factory = rl::EnvRepository::Instance().GetBatchEnvFactory("NameTestNativeBatchEnv");
    REQUIRE(factory != nullptr);
    auto env = factory->CreateBatchEnv(
        anet::EmptyConfigData, torch::Device(torch::kCPU), "native",
        /*seed=*/1, /*num_envs=*/3, rl::RunMode::Eval1, "eval.override");
    REQUIRE(env != nullptr);
    CHECK(env->GetName() == "native");
    CHECK(env->GetBatchSpec().num_envs == 3);
    CHECK(env->GetRunMode() == rl::RunMode::Eval1);
    CHECK(rl::EnvRepository::Instance().GetSingleDiscreteEnvFactory("NameTestNativeBatchEnv") == nullptr);
}
