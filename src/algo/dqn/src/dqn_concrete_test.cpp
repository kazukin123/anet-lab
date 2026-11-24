#include "anet/test/catch.hpp"
#include <string_view>
#include "anet/test/nn.hpp"
#include "anet/test/replay.hpp"
#include "anet/agent.hpp"
#include "anet/algo/dqn.hpp"
#include "dqn_fixture_test.hpp"

constexpr char kDefaultDqnClassId[] = "DefaultDQNAgent";
constexpr char kMainFeatureInput[] = "main_feature/00_Input";
constexpr char kOmitted[] = "omitted";
constexpr char kScaled[] = "scaled";
constexpr char kRunning[] = "running";

constexpr char kRainbowClassId[] = "RainbowAgent";

TEST_CASE("DQN initialization keeps its first Agent factory registrations", "[init][idempotent]")
{
    anet::rl::algo::InitDqn();
    auto& repository = anet::rl::AgentRepository::Instance();
    const auto default_factory = repository.GetAgentFactory(kDefaultDqnClassId);
    const auto rainbow_factory = repository.GetAgentFactory(kRainbowClassId);
    REQUIRE(default_factory);
    REQUIRE(rainbow_factory);
    CHECK_NOTHROW(anet::rl::algo::InitDqn());
    CHECK(repository.GetAgentFactory(kDefaultDqnClassId) == default_factory);
    CHECK(repository.GetAgentFactory(kRainbowClassId) == rainbow_factory);
}

TEST_CASE("DQN initialization retains the original failure after partial registration", "[.][init_failure]")
{
    auto& repository = anet::rl::AgentRepository::Instance();
    const auto existing_default = std::make_shared<dqn::DefaultDQNAgentFactory>();
    repository.Register(existing_default);
    REQUIRE_FALSE(repository.GetAgentFactory(kRainbowClassId));
    const auto already_registered = Catch::Matchers::ContainsSubstring("already registered");
    const auto default_id = Catch::Matchers::ContainsSubstring(kDefaultDqnClassId);
    const auto duplicate_default = already_registered && default_id;

    CHECK_THROWS_WITH(anet::rl::algo::InitDqn(), duplicate_default);
    const auto rainbow = repository.GetAgentFactory(kRainbowClassId);
    REQUIRE(rainbow);
    CHECK_THROWS_WITH(anet::rl::algo::InitDqn(), duplicate_default);
    CHECK(repository.GetAgentFactory(kRainbowClassId) == rainbow);
    CHECK(repository.GetAgentFactory(kDefaultDqnClassId) == existing_default);
}

TEST_CASE("DefaultDQN skip requires an explicit positive duration limit", "[dqn][config][rfd0098]")
{
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.skip.enabled", true);
    REQUIRE_THROWS(dqn::DefaultDQNAgentConfig(data));
}

TEST_CASE("DQN replay priority validation warns once per Run", "[dqn][config][per]")
{
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.use_per = true;
    config.per_initial_priority_mode = "fixed";
    config.per_eps = 0.0f;

    anet::test::LogCaptureGuard logs;
    const auto mode = dqn::ParseReplayInitialPriorityMode(config);
    logs.Flush();
    CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 0);

    dqn::ValidateReplayPriorityConfig(config, mode);
    dqn::ValidateReplayPriorityConfig(config, mode);
    logs.Flush();

    int matching_warnings = 0;
    for (const auto& record : logs.Records()) {
        if (record.level != wxLOG_Warning || record.message.find("per_eps (not read from config)=0") == std::string::npos) continue;
        ++matching_warnings;
        CHECK(record.message.find("zero-TD-error transitions") != std::string::npos);
        CHECK(record.message.find("Set per_eps (not read from config)") != std::string::npos);
    }
    CHECK(matching_warnings == 1);
}

TEST_CASE("DQN replay priority warnings identify keys values reasons and alternatives", "[dqn][config][per]")
{
    auto require_warning = [](const dqn::LearnerConfig& config, const std::vector<std::string>& fragments) {
        anet::test::LogCaptureGuard logs;
        const auto mode = dqn::ParseReplayInitialPriorityMode(config);
        dqn::ValidateReplayPriorityConfig(config, mode);
        dqn::ValidateReplayPriorityConfig(config, mode);
        logs.Flush();

        REQUIRE(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 1);
        const auto& message = logs.Records().front().message;
        for (const auto& fragment : fragments) {
            CAPTURE(message, fragment);
            CHECK(message.find(fragment) != std::string::npos);
        }
    };

    SECTION("actor priority with alpha zero") {
        dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
        config.use_per = true;
        config.per_initial_priority_mode = "actor_approx";
        config.per_alpha = 0.0f;
        require_warning(config, {
            "per_initial_priority_mode (not read from config)=actor_approx",
            "per_alpha (not read from config)=0",
            "does not affect sampling",
            "Set per_alpha (not read from config)",
        });
    }
    SECTION("zero clip upper bound") {
        dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
        config.use_per_prio_clip = true;
        config.per_prio_clip_value = 0.0f;
        require_warning(config, {
            "use_per_prio_clip (not read from config)=true",
            "per_prio_clip_value (not read from config)=0",
            "clipped to zero",
            "disable use_per_prio_clip (not read from config)",
        });
    }
    SECTION("clip upper bound no larger than epsilon") {
        dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
        config.per_eps = 0.25f;
        config.use_per_prio_clip = true;
        config.per_prio_clip_value = 0.2f;
        require_warning(config, {
            "per_prio_clip_value (not read from config)=0.2",
            "per_eps (not read from config)=0.25",
            "collapse priority differences",
            "per_prio_clip_value (not read from config) > per_eps (not read from config)",
        });
    }
}

TEST_CASE("DQN replay priority validation preserves invalid configuration errors", "[dqn][config][per]")
{
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();

    config.per_initial_priority_mode = "unknown";
    CHECK_THROWS(dqn::ParseReplayInitialPriorityMode(config));

    config.per_initial_priority_mode = "fixed";
    const auto fixed_mode = dqn::ParseReplayInitialPriorityMode(config);
    for (const float invalid : {
        -1.0f,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity() }) {
        config.per_initial_priority = invalid;
        CHECK_THROWS(dqn::ValidateReplayPriorityConfig(config, fixed_mode));
    }

    config.per_initial_priority = 1.0f;
    for (const float invalid : {
        -1.0f,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity() }) {
        config.per_eps = invalid;
        CHECK_THROWS(dqn::ValidateReplayPriorityConfig(config, fixed_mode));
    }

    config.per_eps = 1.0e-6f;
    config.use_per = false;
    config.per_initial_priority_mode = "max";
    // PER無効時のmodeはPER有効化に備えた設定値であり正常ケース。エラーにもWARNにもしない。
    CHECK_NOTHROW(dqn::ValidateReplayPriorityConfig(
        config, dqn::ParseReplayInitialPriorityMode(config)));

    config.use_per = true;
    config.per_initial_priority_mode = "fixed";
    config.use_per_prio_clip = true;
    for (const float invalid : {
        -1.0f,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity() }) {
        config.per_prio_clip_value = invalid;
        CHECK_THROWS(dqn::ValidateReplayPriorityConfig(config, fixed_mode));
    }

    // clip無効時は未使用の値を検証しない既存契約を維持する。
    config.use_per_prio_clip = false;
    config.per_prio_clip_value = -1.0f;
    CHECK_NOTHROW(dqn::ValidateReplayPriorityConfig(config, fixed_mode));
}

TEST_CASE("AgentBase exposes configured device", "[agent][device]")
{
    DeviceOnlyAgent agent{ torch::Device(torch::kCPU) };

    CHECK(agent.GetDevice().is_cpu());
}

TEST_CASE("DefaultDQNAgent preserves the native stack axis for temporal Conv1d", "[dqn][native_stack]")
{
    ScopedNoopMetricsLogger metrics_logger;
    PrepareNativeStackNetworkTest();

    // Reshapeを使わない時間軸Conv1D構成とstack=4のAgentをpublic設定経路から構築する。
    auto config_data = MakeNativeStackConfigData();
    config_data.Set("net.block.[TemporalPermute].type", "Permute");
    config_data.Set("net.block.[TemporalPermute].permute.dims", "0 2 1");
    config_data.Set("net.block.[TemporalConv].type", "Conv1d");
    config_data.Set("net.block.[TemporalConv].conv.out_channels", 4);
    config_data.Set("net.block.[TemporalConv].conv.kernel_size", 2);
    config_data.Set("net.branch.[main_feature].bind", kVectorKey);
    config_data.Set("net.branch.[main_feature].structure", "TemporalPermute > TemporalConv > Flatten > Linear");

    anet::TensorSpec vector_spec{
        .type = anet::SpaceType::Vector,
        .shape = { 8 },
        .dtype = torch::kFloat32,
    };
    const auto env_spec = MakeNativeStackEnvSpec({ { kVectorKey, vector_spec } });

    // dummy forwardと同じNetworkをActorの実stack入力へ接続する。
    auto action_info = ActWithNativeStack(
        config_data,
        env_spec,
        anet::TensorDict{ { kVectorKey, torch::arange(16, torch::kFloat32).reshape({ 2, 8 }) } });

    // traceでbranch入口のstack軸とpublic出力を観測する。
    REQUIRE(action_info != nullptr);
    REQUIRE(ShapeOf(action_info->GetAction()) == std::vector<int64_t>{ 2 });
    REQUIRE(ShapeOf(action_info->GetAuxData().at("q_values")) == std::vector<int64_t>{ 2, 2 });
    SaveRfd0170Trace(action_info->GetAuxData(), "native-1");
    const auto trace = rl::ExtractNnTrace(action_info->GetAuxData());
    REQUIRE(trace.Contains("main_feature/00_Input"));
    CHECK(ShapeOf(trace.At("main_feature/00_Input")) == std::vector<int64_t>{ 1, 4, 8 });

    const auto expected = torch::arange(8, torch::kFloat32).reshape({ 1, 1, 8 }).expand({ 1, 4, 8 });
    CHECK(torch::equal(trace.At(kMainFeatureInput), expected));
}

TEST_CASE("DefaultDQNAgent uses a configured one-frame Perception in the Actor", "[dqn][perception]")
{
    ScopedNoopMetricsLogger metrics_logger;
    PrepareNativeStackNetworkTest();
    auto config = MakeNativeStackConfigData();
    config.Set("DefaultDQNAgent.perception.structure", "stack");
    config.Set("DefaultDQNAgent.perception.block.[stack].type", "FrameStack");
    config.Set("DefaultDQNAgent.frame_stack.count", 1);
    config.Set("net.branch.[main_feature].bind", kVectorKey);
    config.Set("net.branch.[main_feature].structure", "Flatten > Linear");

    const anet::TensorSpec spec{ .shape = { 8 } };
    const auto env_spec = MakeNativeStackEnvSpec({ { kVectorKey, spec } });
    const auto action = ActWithNativeStack(config, env_spec,
        anet::TensorDict{ { kVectorKey, torch::arange(16, torch::kFloat32).reshape({ 2, 8 }) } });
    REQUIRE(action);
    const auto trace = rl::ExtractNnTrace(action->GetAuxData());
    REQUIRE(trace.Contains(kMainFeatureInput));
    CHECK(ShapeOf(trace.At(kMainFeatureInput)) == std::vector<int64_t>{ 1, 1, 8 });
}

TEST_CASE("DefaultDQNAgent reports Perception normalization statistics after frame intake", "[dqn][perception]")
{
    ScopedNoopMetricsLogger metrics_logger;
    PrepareNativeStackNetworkTest();
    const std::string behavior = GENERATE(kRunning, kOmitted, kScaled);
    auto data = MakeNativeStackConfigData();
    data.Set("DefaultDQNAgent.learner.update_warmup_steps", 1000);
    data.Set("net.branch.[main_feature].bind", kVectorKey);
    data.Set("net.branch.[main_feature].structure", "Flatten > Linear");
    if (behavior != kOmitted) {
        data.Set("DefaultDQNAgent.perception.structure", "stack > norm");
        data.Set("DefaultDQNAgent.perception.block.[norm].type", "Normalize");
        data.Set("DefaultDQNAgent.perception.block.[norm].mode", kRunning);
    }
    const anet::TensorSpec spec{
        .shape = {8},
        .scale = behavior == kScaled ? anet::ValueScale::Scaled : anet::ValueScale::Unscaled,
    };
    const auto env = MakeNativeStackEnvSpec({{kVectorKey, spec}});
    const auto agent = std::make_shared<dqn::DefaultDQNAgent>(
        dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(data)), anet::nn::NetworkConfig(data),
        rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, env, torch::kCPU, 161);
    const auto reference = rl::ObservationNormalizerFactory(rl::ObservationNormalizerConfig{})
        .CreateObservationNormalizer(env.state_spec);
    const auto learner = agent->CreateLearner();
    for (rl::step_t step = 0; step < 3; ++step) {
        auto experience = MakeDeterminismExperience(step, 2);
        experience.state.obs.Set(kVectorKey, torch::full({2, 8}, static_cast<float>(step)));
        experience.next_state.obs.Set(kVectorKey, torch::full({2, 8}, static_cast<float>(step + 1)));
        reference->UpdateStats(experience.state.obs);
        reference->UpdateStats(experience.next_state.obs);
        learner->UpdateFromBatch({.exp_step = step * 2}, experience);
    }
    const auto count_key = rl::ObservationNormalizer::kKeyCount;
    const auto count = agent->GetScalar(count_key);
    if (behavior == kRunning) {
        REQUIRE(count);
        CHECK(std::isfinite(*count));
        CHECK(count == reference->GetScalar(count_key));
    } else {
        CHECK_FALSE(count);
    }
}

TEST_CASE("DefaultDQN supplies a one-frame axis for a Normalize-only perception pipeline", "[dqn][perception]")
{
    ScopedNoopMetricsLogger metrics_logger;
    PrepareNativeStackNetworkTest();
    auto data = MakeNativeStackConfigData();
    data.Set("DefaultDQNAgent.perception.structure", "norm");
    data.Set("DefaultDQNAgent.perception.block.[norm].type", "Normalize");
    data.Set("DefaultDQNAgent.perception.block.[norm].mode", "none");
    data.Set("DefaultDQNAgent.frame_stack.count", 1);
    data.Set("DefaultDQNAgent.learner.update_warmup_steps", 0);
    data.Set("DefaultDQNAgent.learner.use_rb_prefetch", false);
    data.Set("DefaultDQNAgent.learner.use_n_step", false);
    data.Set("DefaultDQNAgent.learner.update_interval", 1);
    data.Set("DefaultDQNAgent.learner.replay_ratio", -1.0f);
    data.Set("net.block.[StackMerge].type", "StackMerge");
    const std::string obs_key = rl::ObsKeys::kGrid;
    data.Set("net.branch.[main_feature].bind", obs_key);
    data.Set("net.branch.[main_feature].structure", "StackMerge > Flatten > Linear");
    const auto env = MakeNativeStackEnvSpec({
        { obs_key, { .type = anet::SpaceType::Grid, .shape = { 1, 4, 4 } } },
    });
    const anet::TensorDict frame{ { obs_key, torch::arange(32, torch::kFloat32).reshape({ 2, 1, 4, 4 }) } };
    const rl::BatchEnvSpec batch{ .num_envs = 2, .num_threads = 1 };
    const auto make_agent = [&] {
        return std::make_shared<dqn::DefaultDQNAgent>(
            dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(data)), anet::nn::NetworkConfig(data),
            batch, env, torch::kCPU, 123);
    };

    SECTION("Actor, visualization and Sample pass one-frame inputs through StackMerge") {
        // StackMerge の実行は [B, S, C, H, W] を要求する。
        const auto agent = make_agent();
        const auto actor = agent->CreateActor({ .batch_env_spec = batch, .env_spec = env, .device = torch::kCPU,
            .seed = 123, .actor_key = "train" });
        const auto flags = torch::zeros({ 2 }, torch::kBool);
        const auto action = actor->MakeAction({}, rl::BatchState(frame, flags, flags, flags), { .nn_trace = true });
        REQUIRE(action);
        CHECK(ShapeOf(rl::ExtractNnTrace(action->GetAuxData()).At(kMainFeatureInput))
            == std::vector<int64_t>{ 1, 1, 1, 4, 4 });

        const auto view = agent->GetTensorDictFunction(kPolicyForwardKey);
        REQUIRE(view);
        CHECK(torch::equal((*view)(frame).At(kQValueKey), action->GetAuxData().at("q_values")));

        const auto learner = agent->CreateLearner();
        rl::BatchUpdateResultList updates;
        for (rl::step_t step = 0; step < 4; ++step) {
            auto experience = MakeDeterminismExperience(step, 2);
            experience.state.obs = frame;
            experience.next_state.obs = frame;
            auto current = learner->UpdateFromBatch({ .exp_step = step * 2 }, experience);
            updates.insert(updates.end(), current.begin(), current.end());
        }
        CHECK_FALSE(updates.empty());
    }
    SECTION("Actor completion removes the start key before returning Perception") {
        const dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(data));
        const anet::ForwarderBuilder builder(config.perception_config, "", 123);
        auto input_specs = env.state_spec.obs_spec;
        input_specs.emplace(dqn::kPerceptionStartKey, anet::TensorSpec{ .dtype = torch::kBool });
        const auto request = rl::MakePerceptionFromSequenceRequest(2, 123, dqn::kPerceptionStartKey, 1, { obs_key });
        const auto completed = dqn::CompletePerception(builder.BuildForwarder(request, input_specs), { obs_key },
            dqn::kPerceptionStartKey);
        auto input = frame;
        input.Set(dqn::kPerceptionStartKey, torch::ones({ 2 }, torch::kBool));
        const auto output = completed->Forward(input);
        CHECK_FALSE(output.Contains(dqn::kPerceptionStartKey));
        CHECK_FALSE(completed->GetOutputSpecs().contains(dqn::kPerceptionStartKey));
        CHECK(torch::equal(output.At(obs_key), frame.At(obs_key).unsqueeze(1)));
    }
    SECTION("Four frames require a FrameStack module") {
        data.Set("DefaultDQNAgent.frame_stack.count", 4);
        CHECK_THROWS_WITH(make_agent(), Catch::Matchers::ContainsSubstring(anet::rl::kForwardKindPerceptionFromStack));
    }
}

TEST_CASE("DefaultDQN validates Agent-wide frame stacking before building perception", "[dqn][perception]")
{
    ScopedNoopMetricsLogger metrics_logger;
    PrepareNativeStackNetworkTest();
    auto data = MakeNativeStackConfigData();
    data.Set("net.branch.[main_feature].bind", kVectorKey);
    data.Set("net.branch.[main_feature].structure", "Flatten > Linear");
    auto env = MakeNativeStackEnvSpec({ { kVectorKey, { .shape = { 8 } } } });
    const auto construct = [&] {
        return std::make_shared<dqn::DefaultDQNAgent>(
            dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(data)), anet::nn::NetworkConfig(data),
            rl::BatchEnvSpec{ .num_envs = 2, .num_threads = 1 }, env, torch::kCPU, 161);
    };
    SECTION("Rejects zero and non-integral counts") {
        data.Set("DefaultDQNAgent.frame_stack.count", GENERATE("0", "2.5"));
        CHECK_THROWS_WITH(construct(), Catch::Matchers::ContainsSubstring("DefaultDQNAgent.frame_stack.count"));
    }
    SECTION("Rejects an invalid key mode") {
        data.Set("DefaultDQNAgent.frame_stack.keys.mode", "missing");
        CHECK_THROWS_WITH(construct(), Catch::Matchers::ContainsSubstring("DefaultDQNAgent.frame_stack.keys.mode"));
    }
    SECTION("Rejects empty lists and unknown keys") {
        data.Set("DefaultDQNAgent.frame_stack.keys.mode", "listed");
        data.Set("DefaultDQNAgent.frame_stack.keys.listed.names", GENERATE("", "missing"));
        CHECK_THROWS_WITH(construct(), Catch::Matchers::ContainsSubstring("DefaultDQNAgent.frame_stack.keys.listed.names"));
    }
    SECTION("Rejects a Frame key matching the start key") {
        env.state_spec.obs_spec.emplace(dqn::kPerceptionStartKey, anet::TensorSpec{ .shape = { 1 } });
        CHECK_THROWS_WITH(construct(), Catch::Matchers::ContainsSubstring("conflicts with the reserved start key"));
    }
}

TEST_CASE("DefaultDQN rejects mismatched intake and Sample specs from registered modules", "[dqn][perception]")
{
    constexpr char kType[] = "DqnTestDropObservationByKind";
    const std::string aux_key = "test_aux";
    class DropForwarder final : public anet::SpecForwarder {
    public:
        DropForwarder(const anet::TensorSpecMap& inputs, const std::string& key)
            : inputs_(inputs), outputs_(inputs), key_(key)
        {
            outputs_.erase(key_);
        }
    public:
        const anet::TensorSpecMap& GetInputSpecs() const override { return inputs_; }
        const anet::TensorSpecMap& GetOutputSpecs() const override { return outputs_; }
        anet::TensorDict Forward(const anet::TensorDict& input, anet::ForwardCapture*) override
        {
            anet::TensorDict output;
            for (const auto& [key, value] : input) if (key != key_) output.Set(key, value);
            return output;
        }
    private:
        anet::TensorSpecMap inputs_;
        anet::TensorSpecMap outputs_;
        std::string key_;
    };
    class DropModule final : public anet::ForwarderModule {
    public:
        DropModule(anet::ForwardKind kind, std::string key) : kind_(std::move(kind)), key_(std::move(key)) {}
    public:
        std::shared_ptr<anet::SpecForwarder> CreateForwarder(const anet::ForwarderRequest& request,
            const anet::TensorSpecMap& inputs) const override
        {
            if (request.GetForwardKind() == kind_) return std::make_shared<DropForwarder>(inputs, key_);
            return anet::MakePassThroughForwarder(inputs);
        }
    private:
        anet::ForwardKind kind_;
        std::string key_;
    };
    class Factory final : public anet::ForwarderModuleFactory {
    public:
        std::optional<std::vector<anet::ForwardKind>> GetForwardKinds() const override { return std::nullopt; }
        void ReadConfigKeys(const anet::ConfigData& config) const override
        {
            (void)config.Get("kind");
            (void)config.Get("key");
        }
        std::shared_ptr<anet::ForwarderModule> CreateModule(const anet::ConfigData& config,
            const anet::ForwarderModuleContext&) const override
        {
            return std::make_shared<DropModule>(config.Get("kind"), config.Get("key"));
        }
    };

    anet::RegisterForwarderModuleFactory(kType, std::make_shared<Factory>());
    ScopedNoopMetricsLogger metrics_logger;
    PrepareNativeStackNetworkTest();
    auto data = MakeNativeStackConfigData();
    data.Set("DefaultDQNAgent.perception.structure", "stack > test_drop");
    data.Set("DefaultDQNAgent.perception.block.[test_drop].type", kType);
    data.Set("DefaultDQNAgent.perception.block.[test_drop].key", aux_key);
    data.Set("net.branch.[main_feature].bind", kVectorKey);
    data.Set("net.branch.[main_feature].structure", "Flatten > Linear");
    const auto env = MakeNativeStackEnvSpec({ { kVectorKey, { .shape = { 8 } } },
        { aux_key, { .shape = { 2 } } } });
    for (const auto kind : { rl::kForwardKindFrameWithStatsUpdate, rl::kForwardKindPerceptionFromStack }) {
        data.Set("DefaultDQNAgent.perception.block.[test_drop].kind", kind);
        CAPTURE(kind);
        CHECK_THROWS_WITH((dqn::DefaultDQNAgent(
            dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(data)), anet::nn::NetworkConfig(data),
            rl::BatchEnvSpec{ .num_envs = 2, .num_threads = 1 }, env, torch::kCPU, 161)),
            Catch::Matchers::ContainsSubstring(kind) && Catch::Matchers::ContainsSubstring("output has 1 keys"));
    }
}

TEST_CASE("DefaultDQNAgent flattens native vector stacks for MLP branches", "[dqn][native_stack]")
{
    ScopedNoopMetricsLogger metrics_logger;
    PrepareNativeStackNetworkTest();

    // stack軸を保持したbranch入力をFlattenで従来の4*F特徴へ落とす。
    auto config_data = MakeNativeStackConfigData();
    config_data.Set("net.branch.[main_feature].bind", kVectorKey);
    config_data.Set("net.branch.[main_feature].structure", "Flatten > Linear");
    const auto env_spec = MakeNativeStackEnvSpec({
        { kVectorKey, anet::TensorSpec{
                          .type = anet::SpaceType::Vector,
                          .shape = { 8 },
                          .dtype = torch::kFloat32,
                      } },
    });

    auto action_info = ActWithNativeStack(
        config_data,
        env_spec,
        anet::TensorDict{ { kVectorKey, torch::arange(16, torch::kFloat32).reshape({ 2, 8 }) } });

    REQUIRE(action_info != nullptr);
    REQUIRE(ShapeOf(action_info->GetAuxData().at("q_values")) == std::vector<int64_t>{ 2, 2 });
    SaveRfd0170Trace(action_info->GetAuxData(), "native-2");
    const auto trace = rl::ExtractNnTrace(action_info->GetAuxData());
    REQUIRE(trace.Contains("main_feature/00_Input"));
    CHECK(ShapeOf(trace.At("main_feature/00_Input")) == std::vector<int64_t>{ 1, 4, 8 });
}

TEST_CASE("DefaultDQNAgent merges continuous Grid stacks before Conv2d", "[dqn][native_stack]")
{
    ScopedNoopMetricsLogger metrics_logger;
    PrepareNativeStackNetworkTest();

    // 連続Gridはbranch入口でstack軸を保持し、明示したStackMergeでchannelへ統合する。
    auto config_data = MakeNativeStackConfigData();
    config_data.Set("net.block.[StackMerge].type", "StackMerge");
    config_data.Set("net.block.[GridConv].type", "Conv2d");
    config_data.Set("net.block.[GridConv].conv.out_channels", 2);
    config_data.Set("net.block.[GridConv].conv.kernel_size", 1);
    config_data.Set("net.branch.[main_feature].bind", rl::ObsKeys::kGrid);
    config_data.Set("net.branch.[main_feature].structure", "StackMerge > GridConv > Flatten > Linear");
    const auto env_spec = MakeNativeStackEnvSpec({
        { rl::ObsKeys::kGrid, anet::TensorSpec{
                                  .type = anet::SpaceType::Grid,
                                  .shape = { 1, 4, 4 },
                                  .dtype = torch::kFloat32,
                              } },
    });

    auto action_info = ActWithNativeStack(
        config_data,
        env_spec,
        anet::TensorDict{ {
            rl::ObsKeys::kGrid,
            torch::arange(32, torch::kFloat32).reshape({ 2, 1, 4, 4 }),
        } });

    REQUIRE(action_info != nullptr);
    REQUIRE(ShapeOf(action_info->GetAuxData().at("q_values")) == std::vector<int64_t>{ 2, 2 });
    SaveRfd0170Trace(action_info->GetAuxData(), "native-3");
    const auto trace = rl::ExtractNnTrace(action_info->GetAuxData());
    REQUIRE(trace.Contains("main_feature/00_Input"));
    CHECK(ShapeOf(trace.At("main_feature/00_Input")) == std::vector<int64_t>{ 1, 4, 1, 4, 4 });
}

TEST_CASE("DefaultDQNAgent keeps discrete Grid one-hot channel merging", "[dqn][native_stack]")
{
    ScopedNoopMetricsLogger metrics_logger;
    PrepareNativeStackNetworkTest();

    // 離散Gridは既存のNN境界でone-hot化し、stackとclassをConv2dのchannelへ統合する。
    auto config_data = MakeNativeStackConfigData();
    config_data.Set("net.block.[GridConv].type", "Conv2d");
    config_data.Set("net.block.[GridConv].conv.out_channels", 2);
    config_data.Set("net.block.[GridConv].conv.kernel_size", 1);
    config_data.Set("net.branch.[main_feature].bind", rl::ObsKeys::kGrid);
    config_data.Set("net.branch.[main_feature].structure", "GridConv > Flatten > Linear");
    const auto env_spec = MakeNativeStackEnvSpec({
        { rl::ObsKeys::kGrid, anet::TensorSpec{
                                  .type = anet::SpaceType::Grid,
                                  .shape = { 1, 4, 4 },
                                  .dtype = torch::kInt8,
                                  .num_classes = 3,
                              } },
    });
    const auto grid = torch::arange(32, torch::kInt64).remainder(3).reshape({ 2, 1, 4, 4 });

    auto action_info = ActWithNativeStack(
        config_data,
        env_spec,
        anet::TensorDict{ { rl::ObsKeys::kGrid, grid } });

    REQUIRE(action_info != nullptr);
    REQUIRE(ShapeOf(action_info->GetAuxData().at("q_values")) == std::vector<int64_t>{ 2, 2 });
    SaveRfd0170Trace(action_info->GetAuxData(), "native-4");
    const auto trace = rl::ExtractNnTrace(action_info->GetAuxData());
    REQUIRE(trace.Contains("main_feature/00_Input"));
    CHECK(ShapeOf(trace.At("main_feature/00_Input")) == std::vector<int64_t>{ 1, 12, 4, 4 });
}

TEST_CASE("DefaultDQNAgent applies all and listed frame stack key modes", "[dqn][native_stack][perception]")
{
    ScopedNoopMetricsLogger metrics_logger;
    PrepareNativeStackNetworkTest();

    const bool listed = GENERATE(false, true);
    auto config_data = MakeNativeStackConfigData();
    config_data.Set("DefaultDQNAgent.frame_stack.keys.mode", listed ? "listed" : "all");
    config_data.Set(
        "DefaultDQNAgent.frame_stack.keys.listed.names",
        std::vector<std::string>{ kVectorKey });
    config_data.Set("net.branch.[main_feature].bind", kVectorKey);
    config_data.Set("net.branch.[main_feature].structure", "Flatten > Linear");
    config_data.Set("net.branch.[aux_feature].bind", "aux");
    config_data.Set("net.branch.[aux_feature].structure", listed ? "Linear" : "Flatten > Linear");
    const auto env_spec = MakeNativeStackEnvSpec({
        { kVectorKey, anet::TensorSpec{
                          .type = anet::SpaceType::Vector,
                          .shape = { 8 },
                          .dtype = torch::kFloat32,
                      } },
        { "aux", anet::TensorSpec{
                     .type = anet::SpaceType::Vector,
                     .shape = { 3 },
                     .dtype = torch::kFloat32,
                 } },
    });

    auto action_info = ActWithNativeStack(
        config_data,
        env_spec,
        anet::TensorDict{
            { kVectorKey, torch::arange(16, torch::kFloat32).reshape({ 2, 8 }) },
            { "aux", torch::arange(6, torch::kFloat32).reshape({ 2, 3 }) },
        });

    REQUIRE(action_info != nullptr);
    if (listed) SaveRfd0170Trace(action_info->GetAuxData(), "native-5");
    const auto trace = rl::ExtractNnTrace(action_info->GetAuxData());
    REQUIRE(trace.Contains("main_feature/00_Input"));
    REQUIRE(trace.Contains("aux_feature/00_Input"));
    CHECK(ShapeOf(trace.At("main_feature/00_Input")) == std::vector<int64_t>{ 1, 4, 8 });
    const auto aux_shape = listed ? std::vector<int64_t>{ 1, 3 } : std::vector<int64_t>{ 1, 4, 3 };
    CHECK(ShapeOf(trace.At("aux_feature/00_Input")) == aux_shape);
}

TEST_CASE("DefaultDQNAgent TensorDictFunction accepts CPU input on CUDA agent", "[dqn][network_model][device]")
{
    if (!torch::cuda::is_available()) return;

    ScopedNoopMetricsLogger metrics_logger;
    const torch::Device device(torch::kCUDA, 0);
    auto env_spec = MakeLearnerEnvSpec();
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(
        MakeDeviceForwardDefaultDqnConfig(),
        MakeAgentForwardNetworkConfig(),
        rl::BatchEnvSpec{ 1, 1 },
        env_spec,
        device,
        123);

    auto forward = agent->GetTensorDictFunction("policy-net.forward");
    REQUIRE(forward.has_value());

    auto obs = anet::TensorDict{
        { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)) },
    };
    auto out = (*forward)(obs);

    REQUIRE(out.Get("q").has_value());
    CHECK(out.At("q").device().type() == torch::kCUDA);
}

TEST_CASE("DefaultDQN constructs skip networks only when enabled", "[dqn][rfd0098][rfd0098-r2-disabled]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeDeviceForwardDefaultDqnConfig();
    config.skip.enabled = GENERATE(false, true);
    config.skip.max_length = 3;
    config.skip_net.Set("DefaultDQNAgent.skip.net.branch.[skip_feature].bind", "missing_feature");
    config.skip_net.Set("DefaultDQNAgent.skip.net.body.output.[features]", "skip_feature");
    const auto make = [&] {
        return std::make_shared<dqn::DefaultDQNAgent>(config, MakeAgentForwardNetworkConfig(),
            rl::BatchEnvSpec{.num_envs = 1, .num_threads = 1}, MakeLearnerEnvSpec(), torch::Device(torch::kCPU), 98);
    };
    // 無効の定義は構築しない。有効にした同じ不正配線だけを拒否する。
    if (config.skip.enabled) REQUIRE_THROWS(make());
    else REQUIRE_NOTHROW(make());
}

TEST_CASE("DefaultDQN learns skip values from replay windows", "[dqn][rfd0098]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeDeviceForwardDefaultDqnConfig();
    config.skip.enabled = true;
    config.skip.max_length = 3;
    config.skip_net.Set("DefaultDQNAgent.skip.net.branch.[skip_feature].bind", "features");
    config.skip_net.Set("DefaultDQNAgent.skip.net.body.output.[features]", "skip_feature");
    config.learner.quantile_mode = "none";
    config.learner.use_amp = false;
    config.learner.use_per = false;
    config.learner.use_n_step = false;
    config.learner.update_warmup_steps = 0;
    config.learner.update_interval = 1;
    config.learner.replay_ratio = -1.0f;
    const auto agent = std::make_shared<dqn::DefaultDQNAgent>(config, MakeAgentForwardNetworkConfig(),
        rl::BatchEnvSpec{.num_envs = 1, .num_threads = 1}, MakeLearnerEnvSpec(), torch::Device(torch::kCPU), 98);
    const auto learner = agent->CreateLearner();
    rl::BatchUpdateResultList results;
    for (rl::step_t step = 0; step < 20; ++step) {
        results = learner->UpdateFromBatch(rl::StepCounts{.exp_step = step}, MakeDeterminismExperience(step, 1));
    }
    REQUIRE_FALSE(results.empty());
    const auto loss = results.front()->GetScalar("skip_loss");
    REQUIRE(loss.has_value());
    CHECK(std::isfinite(*loss));
    CHECK(*results.front()->GetScalar("skip_sample_k_mean") == 1.0f);
}

TEST_CASE("Dedicated skip forward configuration reaches Actor values independently of Q", "[dqn][rfd0106][rfd0106-forward]")
{
    ScopedNoopMetricsLogger metrics_logger;
    const auto q_key = GENERATE(std::string("test_default"), std::string("sample"));
    const auto skip_key = GENERATE(std::string("shared"), std::string("test_default"), std::string("sample"));
    const bool clone = GENERATE(false, true);
    CAPTURE(q_key, skip_key, clone);
    // 公開特徴量と一つの行動を固定し、Q と skip の設定を独立に確かめる。
    auto data = MakeRfd0104ConfigData();
    data.Set("DefaultDQNAgent.head.projection.noisy.enabled", true);
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.mode", "sample");
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.hold", "call");
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.share", "batch");
    data.Set("DefaultDQNAgent.actor.[eval].nn_forward_key", q_key);
    data.Set("DefaultDQNAgent.actor.[eval].clone_model", clone);
    data.Set("DefaultDQNAgent.actor.[eval].skip.nn_forward_mode", skip_key == "shared" ? "shared" : "dedicated");
    if (skip_key != "shared") data.Set("DefaultDQNAgent.actor.[eval].skip.nn_forward_key", skip_key);
    const auto spec = MakeLearnerEnvSpec();
    const rl::BatchEnvSpec batch{.num_envs = 2, .num_threads = 1};
    torch::manual_seed(106);
    const auto agent = std::make_shared<dqn::DefaultDQNAgent>(dqn::DefaultDQNAgentConfig(data),
        MakeAgentForwardNetworkConfig(), batch, spec, torch::kCPU, 106);
    const auto actor = agent->CreateActor(rl::ActorRequest{.batch_env_spec = batch, .env_spec = spec,
        .device = torch::kCPU, .seed = 106, .actor_key = "eval"});
    const auto flags = torch::zeros({2}, torch::kBool);
    const rl::BatchState state(anet::TensorDict{{kVectorKey, torch::ones({2, 2})}}, flags, flags, flags);
    const auto first = actor->MakeAction({}, state);
    const auto second = actor->MakeAction({}, state);
    CHECK(torch::equal(first->GetAction(), second->GetAction()));
    CHECK(torch::equal(first->GetAuxData().at(kRfd0106QValuesKey), second->GetAuxData().at(kRfd0106QValuesKey)) == (q_key == "test_default"));
    const bool skip_sample = skip_key == "sample" || (skip_key == "shared" && q_key == "sample");
    CHECK(torch::equal(first->GetAuxData().at(dqn::DQNActionInfo::kSkipQValues), second->GetAuxData().at(dqn::DQNActionInfo::kSkipQValues)) == !skip_sample);
    if (skip_key == "shared") {
        // 同じ参照を dedicated で選んでも、重み・seed・時計の割り当ては変わらない。
        data.Set("DefaultDQNAgent.actor.[eval].skip.nn_forward_mode", "dedicated");
        data.Set("DefaultDQNAgent.actor.[eval].skip.nn_forward_key", q_key);
        torch::manual_seed(106);
        const auto dedicated_agent = std::make_shared<dqn::DefaultDQNAgent>(dqn::DefaultDQNAgentConfig(data),
            MakeAgentForwardNetworkConfig(), batch, spec, torch::kCPU, 106);
        const auto dedicated = dedicated_agent->CreateActor(rl::ActorRequest{.batch_env_spec = batch, .env_spec = spec,
            .device = torch::kCPU, .seed = 106, .actor_key = "eval"});
        CHECK(torch::equal(first->GetAuxData().at(dqn::DQNActionInfo::kSkipQValues), dedicated->MakeAction({}, state)->GetAuxData().at(dqn::DQNActionInfo::kSkipQValues)));
        CHECK(torch::equal(second->GetAuxData().at(dqn::DQNActionInfo::kSkipQValues), dedicated->MakeAction({}, state)->GetAuxData().at(dqn::DQNActionInfo::kSkipQValues)));
    }
}

TEST_CASE("Dedicated IQN skip taus separate fixed values from random Q values through fused features", "[dqn][rfd0106][rfd0106-tau]")
{
    ScopedNoopMetricsLogger metrics_logger;
    const bool q_random = GENERATE(false, true);
    auto data = MakeRfd0106IqnConfigData();
    data.Set("DefaultDQNAgent.actor.[train].policy.tau_rule.sample_mode", q_random ? "random" : "fixed");
    data.Set("DefaultDQNAgent.actor.[train].skip.policy.tau_rule.sample_mode", q_random ? "fixed" : "random");
    const auto spec = MakeIqnTracerEnvSpec();
    const rl::BatchEnvSpec batch{.num_envs = 16, .num_threads = 1};
    const auto agent = std::make_shared<dqn::DefaultDQNAgent>(dqn::DefaultDQNAgentConfig(data),
        anet::nn::NetworkConfig(data, "DefaultDQNAgent.net"), batch, spec, torch::kCPU, 106);
    const auto actor = agent->CreateActor(rl::ActorRequest{.batch_env_spec = batch, .env_spec = spec,
        .device = torch::kCPU, .seed = 106, .actor_key = "train"});
    const auto flags = torch::zeros({16}, torch::kBool);
    const rl::BatchState state(anet::TensorDict{{kVectorKey, torch::ones({16, 4})}}, flags, flags, flags);
    // single でも専用τを追加して skip を forward する。継続行動による上書きは入れない。
    const auto first = actor->MakeAction({}, state);
    const auto second = actor->MakeAction({}, state);
    const auto same_action = first->GetAction().eq(second->GetAction());
    REQUIRE(same_action.any().item<bool>());
    CHECK(first->GetInfo().At(rl::BatchActionInfo::kKeyDecision).all().item<bool>());
    CHECK(second->GetInfo().At(rl::BatchActionInfo::kKeyDecision).all().item<bool>());
    CHECK(torch::equal(first->GetAuxData().at(kRfd0106QValuesKey), second->GetAuxData().at(kRfd0106QValuesKey)) == !q_random);
    CHECK(torch::equal(first->GetAuxData().at(dqn::DQNActionInfo::kSkipQValues).index({same_action}),
        second->GetAuxData().at(dqn::DQNActionInfo::kSkipQValues).index({same_action})) == q_random);
}

TEST_CASE("Fixed IQN skip query records grid equivalence without making it an acceptance condition", "[dqn][rfd0106][rfd0106-grid]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto shared_data = MakeRfd0106IqnConfigData();
    shared_data.Set("DefaultDQNAgent.actor.[train].policy.tau_rule.sample_mode", "fixed");
    auto dedicated_data = shared_data;
    dedicated_data.Set("DefaultDQNAgent.actor.[train].skip.policy.tau_rule.sample_mode", "fixed");
    const auto spec = MakeIqnTracerEnvSpec();
    const rl::BatchEnvSpec batch{.num_envs = 16, .num_threads = 1};
    const auto make = [&](const anet::ConfigData& data) {
        torch::manual_seed(106);
        const auto agent = std::make_shared<dqn::DefaultDQNAgent>(dqn::DefaultDQNAgentConfig(data),
            anet::nn::NetworkConfig(data, "DefaultDQNAgent.net"), batch, spec, torch::kCPU, 106);
        return agent->CreateActor(rl::ActorRequest{.batch_env_spec = batch, .env_spec = spec,
            .device = torch::kCPU, .seed = 106, .actor_key = "train"});
    };
    const auto flags = torch::zeros({16}, torch::kBool);
    const rl::BatchState state(anet::TensorDict{{kVectorKey, torch::ones({16, 4})}}, flags, flags, flags);
    const auto shared = make(shared_data)->MakeAction({}, state);
    const auto dedicated = make(dedicated_data)->MakeAction({}, state);
    const auto same_action = shared->GetAction().eq(dedicated->GetAction());
    REQUIRE(same_action.any().item<bool>());
    const bool match = torch::allclose(shared->GetAuxData().at(dqn::DQNActionInfo::kSkipQValues).index({same_action}),
        dedicated->GetAuxData().at(dqn::DQNActionInfo::kSkipQValues).index({same_action}));
    // 不一致は経路の失敗ではなく、評価に G' が要るという観察として記録する。
    SUCCEED("RFD0106 shared_vs_dedicated_fixed_grid_equal=" << match);
}

TEST_CASE("Single skip configuration keeps every lane decision and learns one-step auxiliary targets", "[dqn][rfd0104][rfd0104-single]")
{
    ScopedNoopMetricsLogger metrics_logger;
    const auto data = MakeRfd0104ConfigData();
    const auto spec = MakeLearnerEnvSpec();
    const rl::BatchEnvSpec batch{.num_envs = 2, .num_threads = 1};
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(data)),
        MakeAgentForwardNetworkConfig(), batch, spec, torch::kCPU, 104);
    const auto actor = agent->CreateActor({.batch_env_spec = batch, .env_spec = spec,
        .device = torch::kCPU, .seed = 104, .actor_key = "train"});
    const auto learner = agent->CreateLearner();
    rl::BatchUpdateResultList results;
    // 公開Actorが返す判断の旗を、そのまま経験に載せてLearnerへ渡す。
    for (rl::step_t step = 0; step < 20; ++step) {
        auto experience = MakeDeterminismExperience(step, 2);
        const rl::StepCounts counts{.exp_step = step};
        experience.action = actor->MakeAction(counts, experience.state);
        CHECK(torch::equal(experience.action->GetAction(), torch::zeros({2}, torch::kLong)));
        CHECK(experience.action->GetInfo().At("decision").all().item<bool>());
        CHECK(experience.action->GetAuxData().at("skip_decision").all().item<bool>());
        CHECK(torch::equal(experience.action->GetAuxData().at("skip_length"), torch::ones({2}, torch::kLong)));
        CHECK(experience.action->GetAuxData().at("skip_q_values").sizes().vec() == std::vector<int64_t>{2, 3});
        results = learner->UpdateFromBatch(counts, experience);
    }
    REQUIRE_FALSE(results.empty());
    CHECK(std::isfinite(*results.front()->GetScalar("skip_loss")));
    CHECK(*results.front()->GetScalar("skip_sample_k_mean") == 1.0f);
    CHECK(*actor->GetScalar("skip_session_decision_rate") == 1.0f);
}

TEST_CASE("Action change auxiliary windows cross identical actions at every-step decisions", "[dqn][rfd0104][rfd0104-action-change]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto data = MakeRfd0104ConfigData();
    data.Set("DefaultDQNAgent.learner.skip.target_window", "action_change");
    const auto spec = MakeLearnerEnvSpec();
    const rl::BatchEnvSpec batch{.num_envs = 2, .num_threads = 1};
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(dqn::DefaultDQNAgentConfig(data),
        MakeAgentForwardNetworkConfig(), batch, spec, torch::kCPU, 104);
    const auto actor = agent->CreateActor({.batch_env_spec = batch, .env_spec = spec,
        .device = torch::kCPU, .seed = 104, .actor_key = "train"});
    const auto learner = agent->CreateLearner();
    rl::BatchUpdateResultList results;
    // 判断は毎stepでも、同じ行動なら上限Jまで教材の窓が伸びる。
    for (rl::step_t step = 0; step < 20; ++step) {
        auto experience = MakeDeterminismExperience(step, 2);
        experience.state.episode_start = torch::full({2}, step == 0, torch::kBool);
        experience.next_state.done = torch::zeros({2}, torch::kBool);
        experience.next_state.truncated = torch::zeros({2}, torch::kBool);
        const rl::StepCounts counts{.exp_step = step};
        experience.action = actor->MakeAction(counts, experience.state);
        results = learner->UpdateFromBatch(counts, experience);
    }
    REQUIRE_FALSE(results.empty());
    CHECK(std::isfinite(*results.front()->GetScalar("skip_loss")));
    CHECK(*results.front()->GetScalar("skip_kmax_mean") == 3.0f);
}

TEST_CASE("Episode end auxiliary windows ignore changing actions and decision flags", "[dqn][rfd0104][rfd0104-episode-end]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto data = MakeRfd0104ConfigData();
    data.Set("DefaultDQNAgent.learner.skip.target_window", "episode_end");
    auto spec = MakeLearnerEnvSpec();
    spec.action_spec.value_labels = {"a0", "a1"};
    const rl::BatchEnvSpec batch{.num_envs = 2, .num_threads = 1};
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(dqn::DefaultDQNAgentConfig(data),
        MakeAgentForwardNetworkConfig(), batch, spec, torch::kCPU, 104);
    const auto learner = agent->CreateLearner();
    rl::BatchUpdateResultList results;
    // 行動は毎step交互に変わるが、エピソードが続く限り窓はJまで伸びる。
    for (rl::step_t step = 0; step < 20; ++step) {
        auto experience = MakeDeterminismExperience(step, 2);
        experience.state.episode_start = torch::full({2}, step == 0, torch::kBool);
        experience.next_state.done = torch::zeros({2}, torch::kBool);
        experience.next_state.truncated = torch::zeros({2}, torch::kBool);
        experience.action = std::make_shared<rl::BatchActionInfo>(torch::full({2}, step % 2, torch::kLong),
            experience.action->GetInfo(), experience.action->GetAuxData(), experience.action->GetReplayInitialPriorityHint());
        experience.action->GetInfo().Set("decision", torch::ones({2}, torch::kBool));
        results = learner->UpdateFromBatch({.exp_step = step}, experience);
    }
    REQUIRE_FALSE(results.empty());
    CHECK(std::isfinite(*results.front()->GetScalar("skip_loss")));
    CHECK(*results.front()->GetScalar("skip_kmax_mean") == 3.0f);
}

TEST_CASE("Variable skip limit follows exp step and restricts only duration selection", "[dqn][rfd0104][rfd0104-variable]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto data = MakeRfd0104ConfigData();
    data.Set("DefaultDQNAgent.actor.[train].skip.max_length_mode", "variable");
    data.Set("DefaultDQNAgent.actor.[train].skip.max_length.type", "linear");
    data.Set("DefaultDQNAgent.actor.[train].skip.max_length.start", 3);
    data.Set("DefaultDQNAgent.actor.[train].skip.max_length.end", 1);
    data.Set("DefaultDQNAgent.actor.[train].skip.max_length.steps", 10);
    const auto spec = MakeLearnerEnvSpec();
    const rl::BatchEnvSpec batch{.num_envs = 2, .num_threads = 1};
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(dqn::DefaultDQNAgentConfig(data),
        MakeAgentForwardNetworkConfig(), batch, spec, torch::kCPU, 104);
    const auto actor = agent->CreateActor({.batch_env_spec = batch, .env_spec = spec,
        .device = torch::kCPU, .seed = 104, .actor_key = "train"});
    const auto initial = actor->GetScalar("skip_max_length");
    REQUIRE(initial.has_value());
    CHECK(*initial == 3.0f);
    torch::Tensor previous;
    // decisionの行だけを現在のcで制限し、継続行の長さとJ本のauxを維持する。
    for (rl::step_t step = 0; step < 20; ++step) {
        auto experience = MakeDeterminismExperience(step, 2);
        const auto action = actor->MakeAction({.exp_step = step}, experience.state);
        const int limit = step >= 10 ? 1 : static_cast<int>(3.0 - 2.0 * static_cast<double>(step) / 10.0);
        CHECK(*actor->GetScalar("skip_max_length") == static_cast<float>(limit));
        const auto decision = action->GetInfo().At("decision");
        const auto lengths = action->GetAuxData().at("skip_length");
        CHECK(lengths.masked_select(decision).le(limit).all().item<bool>());
        if (previous.defined()) {
            CHECK(torch::equal(lengths.masked_select(decision.logical_not()), previous.masked_select(decision.logical_not())));
        }
        CHECK(action->GetAuxData().at("skip_q_values").sizes().vec() == std::vector<int64_t>{2, 3});
        previous = lengths.clone();
    }
    for (const auto* mode : {"full", "single"}) {
        data.Set("DefaultDQNAgent.actor.[train].skip.max_length_mode", mode);
        auto other = std::make_shared<dqn::DefaultDQNAgent>(dqn::DefaultDQNAgentConfig(data),
            MakeAgentForwardNetworkConfig(), batch, spec, torch::kCPU, 104);
        const auto inactive = other->CreateActor({.batch_env_spec = batch, .env_spec = spec,
            .device = torch::kCPU, .seed = 104, .actor_key = "train"})->GetScalar("skip_max_length");
        REQUIRE(inactive.has_value());
        CHECK(std::isnan(*inactive));
    }
}

TEST_CASE("Auxiliary catalog checks use active fields for every ProfiledValue type", "[dqn][rfd0104][config]")
{
    const auto type = GENERATE(std::string("constant"), std::string("linear"), std::string("exponential"),
        std::string("cosine"), std::string("cosine_restart"), std::string("phased"));
    const bool one = GENERATE(false, true);
    CAPTURE(type, one);
    auto data = MakeRfd0104ConfigData();
    data.Set("DefaultDQNAgent.learner.skip.target_window", "episode_end");
    data.Set("DefaultDQNAgent.actor.[train].skip.max_length_mode", "variable");
    const std::string key = "DefaultDQNAgent.actor.[train].skip.max_length";
    data.Set(key + ".type", type);
    if (type == "constant") {
        data.Set(key, one ? 1 : 2);
        data.Set(key + ".start", 100);
    } else if (type == "phased") {
        data.Set(key, 100);
        data.Set(key + ".phases", "p");
        data.Set(key + ".phase.[p].type", "linear");
        data.Set(key + ".phase.[p].start", 1);
        data.Set(key + ".phase.[p].end", one ? 1 : 2);
        data.Set(key + ".phase.[p].steps", 2);
    } else {
        data.Set(key, 100);
        data.Set(key + ".start", 1);
        data.Set(key + ".end", one ? 1 : 2);
        data.Set(key + ".steps", 2);
    }
    anet::test::LogCaptureGuard logs;
    if (one) CHECK_NOTHROW(dqn::DefaultDQNAgentConfig(data));
    else CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data), Catch::Matchers::ContainsSubstring("active_range"));
}

TEST_CASE("Disabled skip has no required J or variable limit but still validates recognized values", "[dqn][rfd0104][config]")
{
    anet::ConfigData data;
    data.Set("DefaultDQNAgent.actor.[custom].network", "online");
    data.Set("DefaultDQNAgent.actor.[custom].skip.max_length_mode", "variable");
    data.Set("DefaultDQNAgent.learner.skip.target_window", "episode_end");
    data = WithDefaultForwardConfig(data);
    anet::test::LogCaptureGuard logs;
    CHECK_NOTHROW(dqn::DefaultDQNAgentConfig(data));
    data.Set("DefaultDQNAgent.actor.[custom].skip.max_length", 100);
    CHECK_NOTHROW(dqn::DefaultDQNAgentConfig(data));
    logs.Flush();
    CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 0);
    data.Set("DefaultDQNAgent.actor.[custom].skip.max_length", 0);
    CHECK_THROWS(dqn::DefaultDQNAgentConfig(data));
    data.Set("DefaultDQNAgent.actor.[custom].skip.max_length", 1);
    data.Set("DefaultDQNAgent.actor.[custom].skip.max_length_mode", "unknown");
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data), Catch::Matchers::ContainsSubstring("skip.max_length_mode"));
    data.Set("DefaultDQNAgent.actor.[custom].skip.max_length_mode", "single");
    data.Set("DefaultDQNAgent.learner.skip.target_window", "unknown");
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data), Catch::Matchers::ContainsSubstring("skip.target_window"));
}

TEST_CASE("Skip length diagnostics identify the configured mode", "[dqn][rfd0194]")
{
    auto data = MakeRfd0104ConfigData();
    const std::string prefix = "DefaultDQNAgent.actor.[train].skip.";
    const std::string mode_source = "Trial.actor.[train].skip.max_length_mode";
    anet::ConfigData source;
    source.Set(mode_source, "variable");
    data.CopyValueFrom(prefix + "max_length_mode", source, mode_source);
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data),
        Catch::Matchers::ContainsSubstring("Missing " + prefix + "max_length")
        && Catch::Matchers::ContainsSubstring("when " + mode_source + "=variable"));

    source.Set(mode_source, "single");
    const std::string length_source = "Trial.actor.[train].skip.max_length";
    source.Set(length_source, 1);
    data.CopyValueFrom(prefix + "max_length_mode", source, mode_source);
    data.CopyValueFrom(prefix + "max_length", source, length_source);
    anet::test::LogCaptureGuard logs;
    CHECK_NOTHROW(dqn::DefaultDQNAgentConfig(data));
    logs.Flush();
    CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Warning,
        {length_source, "is unused with " + mode_source + "=single", "expected " + mode_source + "=variable"}));
}

TEST_CASE("DefaultDQN checkpoint headers identify Agent serialization formats", "[dqn][serialize][rfd0098][checkpoint-format]")
{
    ScopedNoopMetricsLogger metrics_logger;
    // 定数倍率は既存の版、動的スケーラは状態を保存する第3版を使う。
    const bool enabled = GENERATE(false, true);
    const bool dynamic = GENERATE(false, true);
        INFO("skip.enabled=" << enabled);
        auto config = MakeDeviceForwardDefaultDqnConfig();
        config.skip.enabled = enabled;
        config.reward_scaler.use_dynamic_scaling = dynamic;
        config.learner.use_amp = false;
        if (enabled) {
            config.skip.max_length = 3;
            config.skip_net.Set("DefaultDQNAgent.skip.net.branch.[skip_feature].bind", "features");
            config.skip_net.Set("DefaultDQNAgent.skip.net.body.output.[features]", "skip_feature");
        }
        const auto agent = std::make_shared<dqn::DefaultDQNAgent>(config, MakeAgentForwardNetworkConfig(),
            rl::BatchEnvSpec{.num_envs = 1, .num_threads = 1}, MakeLearnerEnvSpec(), torch::kCPU, 98);
        std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
        anet::OutputArchive output(stream);
        agent->Save(output);
        stream.seekg(0);
        anet::InputArchive input(stream);
        anet::ArchiveHeader header;
        input.Read(header);
        CHECK(header.info == (dynamic ? "DefaultDQNAgent-3" : enabled ? "DefaultDQNAgent-2" : "DefaultDQNAgent"));
}

TEST_CASE("Skip checkpoint restores skip values across construction seeds", "[dqn][serialize][rfd0098]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeDeviceForwardDefaultDqnConfig();
    config.skip.enabled = true;
    config.skip.max_length = 3;
    config.skip_net.Set("DefaultDQNAgent.skip.net.branch.[skip_feature].bind", "features");
    config.skip_net.Set("DefaultDQNAgent.skip.net.body.output.[features]", "skip_feature");
    config.learner.use_amp = false;
    const auto env = MakeLearnerEnvSpec();
    const auto make_agent = [&](const dqn::DefaultDQNAgentConfig& settings, anet::seed_t seed) {
        return std::make_shared<dqn::DefaultDQNAgent>(settings, MakeAgentForwardNetworkConfig(),
            rl::BatchEnvSpec{.num_envs = 1, .num_threads = 1}, env, torch::Device(torch::kCPU), seed);
    };
    const auto source = make_agent(config, 98);
    const auto checkpoint = std::filesystem::temp_directory_path()
        / ("anet_rfd0098_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
    {
        std::ofstream stream(checkpoint, std::ios::binary);
        anet::OutputArchive archive(stream);
        source->Save(archive);
    }
    config.auto_load_file = checkpoint.string();
    const auto restored = make_agent(config, 99);
    const auto values = [&](const std::shared_ptr<dqn::DefaultDQNAgent>& agent) {
        const auto actor = agent->CreateActor(rl::ActorRequest{.batch_env_spec = {.num_envs = 1, .num_threads = 1}, .env_spec = env,
            .device = torch::Device(torch::kCPU), .seed = 98, .actor_key = "eval"});
        const auto flags = torch::zeros({1}, torch::kBool);
        const rl::BatchState state(anet::TensorDict{{kVectorKey, torch::ones({1, 2})}}, flags, flags, flags);
        return actor->MakeAction({}, state)->GetAuxData().at("skip_q_values").clone();
    };
    const bool same = torch::equal(values(source), values(restored));
    std::filesystem::remove(checkpoint);
    CHECK(same);
}

TEST_CASE("Skip checkpoint migration initializes only missing split moments and rejects dropping them", "[dqn][serialize][rfd0103]")
{
    ScopedNoopMetricsLogger metrics_logger;
    const std::string saved_mode = GENERATE(std::string("off"), std::string("stop"), std::string("merge"), std::string("split"));
    const std::string current_mode = GENERATE(std::string("stop"), std::string("merge"), std::string("split"));
    auto config = MakeDeviceForwardDefaultDqnConfig();
    config.skip.enabled = saved_mode != "off";
    config.skip.max_length = 3;
    config.skip_net.Set("DefaultDQNAgent.skip.net.branch.[skip_feature].bind", "features");
    config.skip_net.Set("DefaultDQNAgent.skip.net.body.output.[features]", "skip_feature");
    config.learner.skip.feature_gradient = saved_mode == "off" ? "stop" : saved_mode;
    config.learner.use_amp = false;
    const auto make_agent = [&](const dqn::DefaultDQNAgentConfig& settings) {
        return std::make_shared<dqn::DefaultDQNAgent>(settings, MakeAgentForwardNetworkConfig(),
            rl::BatchEnvSpec{.num_envs = 1, .num_threads = 1}, MakeLearnerEnvSpec(), torch::kCPU, 103);
    };
    const auto source = make_agent(config);
    const auto path = std::filesystem::temp_directory_path()
        / ("anet_rfd0103_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
    {
        std::ofstream stream(path, std::ios::binary);
        anet::OutputArchive archive(stream);
        source->Save(archive);
    }
    config.skip.enabled = true;
    config.learner.skip.feature_gradient = current_mode;
    auto scheduled = WithDefaultForwardConfig();
    scheduled.Set("DefaultDQNAgent.learner.learning_rate.type", "linear");
    scheduled.Set("DefaultDQNAgent.learner.learning_rate.start", 0.0003f);
    scheduled.Set("DefaultDQNAgent.learner.learning_rate.end", 0.0f);
    scheduled.Set("DefaultDQNAgent.learner.learning_rate.steps", 10);
    config.learner.learning_rate = dqn::DefaultDQNAgentConfig(scheduled).learner.learning_rate;
    config.learner.adam_eps = 0.0002f;
    config.learner.weight_decay.value = 0.02f;
    config.learner.update_warmup_steps = 1000;
    config.auto_load_file = path.string();
    anet::test::LogCaptureGuard logs;
    if (saved_mode == "split" && current_mode != "split") {
        CHECK_THROWS(make_agent(config));
    } else {
        std::shared_ptr<dqn::DefaultDQNAgent> restored;
        CHECK_NOTHROW(restored = make_agent(config));
        REQUIRE(restored);
        CHECK_NOTHROW(make_agent(config));
        logs.Flush();
        size_t initialized = 0;
        size_t missing_network = 0;
        for (const auto& record : logs.Records()) {
            if (record.message.find("checkpoint has no primary_skip optimizer role") != std::string::npos) {
                CHECK(record.level == wxLOG_Warning);
                ++initialized;
            }
            if (record.message.find("Checkpoint has no skip networks") != std::string::npos) {
                CHECK(record.level == wxLOG_Warning);
                ++missing_network;
            }
        }
        CHECK(initialized == (saved_mode != "split" && current_mode == "split" ? 1 : 0));
        CHECK(missing_network == (saved_mode == "off" ? 1 : 0));
        if (current_mode == "split") {
            CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
                {"group=4", "DefaultDQNAgent.learner.learning_rate.start=0.0003", "fixed weight decay 0", "adam_eps=0.0002"}));
        }
        if (saved_mode == "split" && current_mode == "split") {
            // 公開の更新境界でscheduleだけを進め、追加roleにも同じlrが保存されることを確かめる。
            restored->CreateLearner()->UpdateFromBatch({.exp_step = 5}, MakeDeterminismExperience(0, 1));
            {
                std::ofstream stream(path, std::ios::binary);
                anet::OutputArchive archive(stream);
                restored->Save(archive);
            }
            anet::log::ResetNotifications();
            CHECK_NOTHROW(make_agent(config));
            logs.Flush();
            CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
                {"role=primary_skip", "option=lr", "key=DefaultDQNAgent.learner.learning_rate.start",
                    "checkpoint=0.00015", "config=0.0003"}));
        }
    }
    std::filesystem::remove(path);
}

TEST_CASE("Skip learning supports distribution, target and feature-gradient combinations", "[dqn][rfd0098][rfd0103][skip_matrix]")
{
    ScopedNoopMetricsLogger metrics_logger;
    const bool split = GENERATE(false, true);
    for (const std::string mode : {"none", "qr", "iqn"}) {
        for (const std::string target : {"disabled", "target", "online", "online_reuse"}) {
            for (const bool conditional : {false, true}) {
                CAPTURE(mode, target, conditional, split);
                anet::ConfigData data;
                data.Set("DefaultDQNAgent.quantile_mode", mode);
                data.Set("DefaultDQNAgent.head.use_dueling", false);


                data.Set("DefaultDQNAgent.skip.enabled", true);
                data.Set("DefaultDQNAgent.skip.max_length", 3);
                data.Set("DefaultDQNAgent.skip.network_mode", conditional ? "action_in" : "action_out");
                data.Set("DefaultDQNAgent.skip.head.use_dueling", conditional);
                data.Set("DefaultDQNAgent.learner.skip.feature_gradient", split ? "split" : conditional ? "merge" : "stop");
                data.Set("DefaultDQNAgent.learner.skip.priority_lambda", 0.25f);
                data.Set("DefaultDQNAgent.learner.replay_capacity", 32);
                data.Set("DefaultDQNAgent.learner.replay_batch_size", 2);
                data.Set("DefaultDQNAgent.learner.use_fused_optimizer", split && conditional);
                data.Set("DefaultDQNAgent.learner.use_amp", conditional);
                data.Set("DefaultDQNAgent.learner.use_amp_bf16", true);
                data.Set("DefaultDQNAgent.learner.use_per", true);
                data.Set("DefaultDQNAgent.learner.use_tbo", conditional);
                data.Set("DefaultDQNAgent.learner.update_warmup_steps", 0);
                data.Set("DefaultDQNAgent.learner.update_interval", 1);
                data.Set("DefaultDQNAgent.learner.replay_ratio", -1.0f);
                data.Set("DefaultDQNAgent.learner.use_n_step", true);
                data.Set("DefaultDQNAgent.learner.n_step", 2);
                data.Set("DefaultDQNAgent.qr.num_quantiles", 4);
                data.Set("DefaultDQNAgent.learner.munchausen.enabled", target != "disabled");
                if (target != "disabled") {
                    data.Set("DefaultDQNAgent.learner.munchausen.log_policy_mode", target);
                    data.Set("DefaultDQNAgent.learner.use_double_dqn", false);
                }
                data.Set("DefaultDQNAgent.head.projection.noisy.enabled", conditional);
                data.Set("DefaultDQNAgent.actor.[train].network", "online");
                data.Set("DefaultDQNAgent.actor.[train].policy.epsilon", 0.0f);
                data.Set("DefaultDQNAgent.actor.[train].policy.epsilon.start", 0.0f);
                data.Set("DefaultDQNAgent.actor.[train].policy.epsilon.end", 0.0f);
                data.Set("net.block.[SkipTestLinear].type", "Linear");
                data.Set("net.block.[SkipTestLinear].linear.out_features", 4);
                data.Set("net.block.[Flatten].type", kFixtureFlatten);
                data.Set("DefaultDQNAgent.net.branch.[main].bind", "vector");
                data.Set("DefaultDQNAgent.net.branch.[main].structure", "Flatten > SkipTestLinear");
                data.Set("DefaultDQNAgent.net.body.output.[context]", "main");
                data.Set("DefaultDQNAgent.net.body.output.[features]", "main");
                const std::string prefix = "DefaultDQNAgent.skip.net.";
                data.Set(prefix + "branch.[main].bind", conditional ? "context, action" : "context");
                data.Set(prefix + "branch.[main].structure", "SkipTestLinear");
                data.Set(prefix + "body.output.[features]", "main");
                if (mode == "iqn") {
                    data.Set("net.block.[SkipTestCos].type", "CosineEmbedding");
                    data.Set("net.block.[SkipTestCos].cos.num_basis", 4);
                    for (const auto& net : {std::string("DefaultDQNAgent.net."), prefix}) {
                        data.Set(net + "branch.[tau].bind", "taus");
                        data.Set(net + "branch.[tau].structure", "SkipTestCos > SkipTestLinear");
                        data.Set(net + "branch.[fused].bind", "main * tau");
                        data.Set(net + "body.output.[features]", "fused");
                    }
                }
                data = WithDefaultForwardConfig(data);
                if (split && conditional) {
                    // 共有Bodyのsigmaにもskip勾配を流し、BF16/fusedの独立更新を通す。
                    data.Set("net.block.[SkipTestLinear].noisy.enabled", true);
                    data.Set("DefaultDQNAgent.nn_forward.[test_sample].noisy.mode", "sample");
                    data.Set("DefaultDQNAgent.nn_forward.[test_sample].noisy.hold", "call");
                    data.Set("DefaultDQNAgent.nn_forward.[test_sample].noisy.share", "sample");
                    data.Set("DefaultDQNAgent.learner.current.nn_forward_key", "test_sample");
                    data.Set("DefaultDQNAgent.learner.target.nn_forward_key", "test_sample");
                }
                const auto env = MakeIqnTracerEnvSpec();
                const auto agent = std::make_shared<dqn::DefaultDQNAgent>(dqn::DefaultDQNAgentConfig(data),
                    anet::nn::NetworkConfig(data, "DefaultDQNAgent.net"), rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, env, torch::kCPU, 98);
                const auto actor = agent->CreateActor(rl::ActorRequest{.batch_env_spec = {.num_envs = 2, .num_threads = 1}, .env_spec = env,
                    .device = torch::Device(torch::kCPU), .seed = 98, .actor_key = "train"});
                const auto learner = agent->CreateLearner();
                rl::BatchUpdateResultList updates;
                const auto flags = torch::zeros({2}, torch::kBool);
                for (rl::step_t step = 0; step < 12; ++step) {
                    const auto start = torch::full({2}, step == 0, torch::kBool);
                    rl::BatchState state(anet::TensorDict{{kVectorKey, torch::full({2, 4}, 0.1f * step)}}, flags, flags, start);
                    const auto action = actor->MakeAction({.exp_step = step}, state);
                    rl::BatchState next(anet::TensorDict{{kVectorKey, torch::full({2, 4}, 0.1f * (step + 1))}}, flags, flags, flags);
                    updates = learner->UpdateFromBatch({.exp_step = step}, rl::BatchExperience{state, action, torch::ones({2}), next});
                }
                REQUIRE_FALSE(updates.empty());
                for (const auto* key : {"skip_loss", "skip_td_abs_mean", "skip_sample_k_mean", "skip_kmax_mean", "skip_grad_norm"}) {
                    const auto value = updates.back()->GetScalar(key);
                    REQUIRE(value);
                    CHECK(std::isfinite(*value));
                }
                CHECK(*updates.back()->GetScalar("skip_sample_k_mean") >= 1.0f);
                CHECK(*updates.back()->GetScalar("skip_sample_k_mean") <= 3.0f);
                for (const auto* key : {"skip_primary_grad_norm", "skip_primary_clip_ratio"}) {
                    const auto value = updates.back()->GetScalar(key);
                    REQUIRE(value);
                    CHECK(std::isfinite(*value) == split);
                    if (!split) CHECK(std::isnan(*value));
                }
            }
        }
    }
}

TEST_CASE("Skip weighting and gradient enums are validated even while disabled", "[dqn][config][rfd0103]")
{
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.skip.enabled", false);
    data.Set("DefaultDQNAgent.learner.skip.feature_gradient", "split");
    CHECK(dqn::DefaultDQNAgentConfig(data).learner.skip.sample_weight == "uniform");
    data.Set("DefaultDQNAgent.learner.skip.sample_weight", "kmax");
    CHECK(dqn::DefaultDQNAgentConfig(data).learner.skip.sample_weight == "kmax");
    data.Set("DefaultDQNAgent.learner.skip.sample_weight", "invalid");
    CHECK_THROWS(dqn::DefaultDQNAgentConfig(data));
    data.Set("DefaultDQNAgent.learner.skip.sample_weight", "uniform");
    data.Set("DefaultDQNAgent.learner.skip.feature_gradient", "invalid");
    CHECK_THROWS(dqn::DefaultDQNAgentConfig(data));
}

TEST_CASE("Split gradient noise measures only Q loss while skip gradients change", "[dqn][rfd0103][gradnoise]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::ConfigData network;
    network.Set("net.block.[SharedLinear].type", "Linear");
    network.Set("net.block.[SharedLinear].linear.out_features", 2);
    network.Set("net.block.[Flatten].type", kFixtureFlatten);
    network.Set("net.branch.[shared].bind", "vector");
    network.Set("net.branch.[shared].structure", "Flatten > SharedLinear");
    network.Set("net.body.output.[features]", "shared");
    const auto make_agent = [&](float weight) {
        torch::manual_seed(103);
        auto config = MakeDeviceForwardDefaultDqnConfig();
        config.skip.enabled = true;
        config.skip.max_length = 3;
        config.skip_net.Set("DefaultDQNAgent.skip.net.branch.[skip_feature].bind", "features");
        config.skip_net.Set("DefaultDQNAgent.skip.net.body.output.[features]", "skip_feature");
        config.learner.skip.feature_gradient = "split";
        config.learner.skip.loss_weight = weight;
        config.learner.learning_rate.value = 0;
        config.learner.replay_capacity = 64;
        config.learner.replay_batch_size = 8;
        config.learner.final_feature_branch_key = "shared";
        config.learner.grad_noise.num_micro_batches = 4;
        config.learner.update_warmup_steps = 8;
        config.learner.update_interval = 1;
        config.learner.replay_ratio = -1;
        config.learner.use_n_step = false;
        config.learner.use_per = false;
        config.learner.use_amp = false;
        return std::make_shared<dqn::DefaultDQNAgent>(config, anet::nn::NetworkConfig(network, "net"),
            rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, MakeLearnerEnvSpec(), torch::kCPU, 103);
    };
    const auto zero = make_agent(0), weighted = make_agent(7);
    const auto first = zero->CreateLearner(), second = weighted->CreateLearner();
    for (const auto& agent : {zero, weighted}) {
        agent->ConfigureScalarMetricSubscriptions({
            {.source_key = "grad_noise_signal_raw", .event = rl::EventType::LEARN,
                .target = rl::EventField::UPDATE_RESULT, .interval = 1, .scope = rl::RunnerScope::TRAIN},
            {.source_key = "grad_noise_variance_raw", .event = rl::EventType::LEARN,
                .target = rl::EventField::UPDATE_RESULT, .interval = 1, .scope = rl::RunnerScope::TRAIN}});
    }
    bool measured = false, skip_gradient_seen = false;
    for (rl::step_t step = 0; step < 24; ++step) {
        const auto experience = MakeDeterminismExperience(step, 2);
        const auto a = first->UpdateFromBatch({.exp_step = step * 2}, experience);
        const auto b = second->UpdateFromBatch({.exp_step = step * 2}, experience);
        REQUIRE(a.size() == b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            for (const auto* key : {"grad_noise_signal_raw", "grad_noise_variance_raw"}) {
                REQUIRE(a[i]->GetScalar(key));
                REQUIRE(b[i]->GetScalar(key));
                REQUIRE(std::isfinite(*a[i]->GetScalar(key)));
                CHECK(*a[i]->GetScalar(key) == *b[i]->GetScalar(key));
            }
            CHECK(*a[i]->GetScalar("skip_primary_grad_norm") == 0.0f);
            CHECK(std::isfinite(*b[i]->GetScalar("skip_primary_grad_norm")));
            skip_gradient_seen |= *b[i]->GetScalar("skip_primary_grad_norm") > 0.0f;
            measured = true;
        }
    }
    CHECK(measured);
    CHECK(skip_gradient_seen);
}

TEST_CASE("RFD0103 convolution sharing arms construct and act through IQN", "[.][rfd0103-arms]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const bool separate = GENERATE(false, true);
    const auto root = std::filesystem::current_path();
    const auto output = root / ".scratch/rfd0103/arms";
    std::filesystem::create_directories(output);
    const auto input = output / (separate ? "separate.txt" : "shared.txt");
    {
        std::ofstream file(input);
        file << "$include <" << (root / "apps/runner/config/_main.txt").generic_string() << ">\n"
            << "$include <" << (root / "apps/runner/config/Atari.txt").generic_string() << ">\n"
            << "$include <" << (root / "src/core/base/testdata/rfd0103/arms.txt").generic_string() << ">\n"
            << "run.$ = run.@va_base > run.@skip > run.@skip_conv_" << (separate ? "separate" : "shared") << '\n';
    }
    const auto data = anet::ConfigManager(input).GetConfigData();
    auto config = dqn::DefaultDQNAgentConfig(data);
    config.learner.replay_capacity = 32;
    config.learner.replay_batch_size = 2;
    config.learner.use_rb_prefetch = false;
    CHECK(config.quantile_mode == "iqn");
    CHECK(config.learner.skip.feature_gradient == (separate ? "stop" : "merge"));
    CHECK(data.Get("DefaultDQNAgent.skip.net.branch.[skip_main_feature].bind") == (separate ? "grid" : "skip_context"));
    if (!separate) CHECK(data.Get("DefaultDQNAgent.net.body.output.[skip_context]") == "trunk");
    auto env = MakeLearnerEnvSpec();
    env.state_spec.obs_spec.clear();
    env.state_spec.obs_spec["grid"] = anet::TensorSpec{.type = anet::SpaceType::Grid,
        .shape = {1, 84, 84}, .dtype = torch::kUInt8, .min_values = {0.0}, .max_values = {255.0}};
    const rl::BatchEnvSpec batch{.num_envs = 2, .num_threads = 1};
    const auto agent = std::make_shared<dqn::DefaultDQNAgent>(config,
        anet::nn::NetworkConfig(data, "DefaultDQNAgent.net"), batch, env, torch::kCPU, 103);
    const auto actor = agent->CreateActor({.batch_env_spec = batch, .env_spec = env,
        .device = torch::kCPU, .seed = 103, .actor_key = "train"});
    const auto flags = torch::zeros({2}, torch::kBool);
    const rl::BatchState state(anet::TensorDict{{"grid", torch::zeros({2, 1, 84, 84}, torch::kUInt8)}}, flags, flags, ~flags);
    const auto action = actor->MakeAction({}, state);
    CHECK(action->GetAction().sizes().vec() == std::vector<int64_t>{2});
    const auto values = action->GetAuxData().at("skip_q_values");
    CHECK(values.sizes().vec() == std::vector<int64_t>{2, config.skip.max_length});
    CHECK(torch::isfinite(values).all().item<bool>());
    CHECK(*actor->GetScalar("skip_session_decision_rate") == 1.0f);
}

TEST_CASE("Primary checkpoint branches into skip with identical next primary optimizer update", "[dqn][serialize][rfd0098]")
{
    ScopedNoopMetricsLogger metrics_logger;
    for (const bool noisy : {false, true}) {
        auto config = MakeDeviceForwardDefaultDqnConfig();
        config.learner.use_per = false;
        config.learner.use_amp = false;
        config.learner.use_tbo = false;
        config.learner.use_n_step = false;
        config.learner.update_warmup_steps = 100;
        config.learner.update_interval = 1;
        config.learner.replay_ratio = -1;
        config.reward_scaler.use_dynamic_scaling = false;
        config.reward_scaler.use_auto_post_scale = false;
        config.head.projection.Set("noisy.enabled", noisy);
        const auto make = [&](const dqn::DefaultDQNAgentConfig& settings, anet::seed_t seed) {
            return std::make_shared<dqn::DefaultDQNAgent>(settings, MakeAgentForwardNetworkConfig(),
                rl::BatchEnvSpec{.num_envs = 1, .num_threads = 1}, MakeLearnerEnvSpec(), torch::kCPU, seed);
        };
        auto source = make(config, 98);
        rl::ExperienceSamples samples;
        samples.obs = anet::TensorDict{{kVectorKey, torch::ones({2, 2})}};
        samples.actions = torch::zeros({2}, torch::kInt64);
        samples.target_returns = torch::tensor({1.0f, 2.0f});
        samples.n_steps = torch::ones({2}, torch::kInt64);
        samples.next_state.next_obs = samples.obs;
        samples.next_state.terminals = torch::ones({2}, torch::kBool);
        samples.window.lengths = samples.n_steps;
        samples.window.max_lengths = samples.n_steps;
        samples.window.returns = samples.target_returns;
        samples.window.next_obs = samples.obs;
        samples.window.terminals = samples.next_state.terminals;
        samples.window.truncated = torch::zeros({2}, torch::kBool);
        const auto update = [&](const auto& agent) {
            const auto flags = torch::zeros({1}, torch::kBool);
            const rl::BatchState state(anet::TensorDict{{kVectorKey, torch::ones({1, 2})}}, flags, flags, flags);
            const auto action = std::make_shared<rl::BatchActionInfo>(torch::zeros({1}, torch::kInt64));
            rl::BatchExperience experience{state, action, torch::ones({1}), state};
            const auto learner = agent->CreateLearner();
            for (rl::step_t step = 0; step < 8; ++step) {
                experience.state.episode_start = torch::full({1}, step == 0, torch::kBool);
                CHECK(learner->UpdateFromBatch({.exp_step = step}, experience).empty());
            }
            REQUIRE(learner->UpdateFromBatch({.exp_step = 100}, experience).size() == 1);
        };
        update(source);
        const auto file = std::filesystem::temp_directory_path()
            / ("anet_rfd0098_branch_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
        { std::ofstream stream(file, std::ios::binary); anet::OutputArchive archive(stream); source->Save(archive); }
        config.auto_load_file = file.string();
        config.learner.learning_rate.value *= 0.5f;
        const auto off = make(config, 99);
        config.skip.enabled = true;
        config.skip.max_length = 3;
        config.skip_net.Set("DefaultDQNAgent.skip.net.branch.[main].bind", "features");
        config.skip_net.Set("DefaultDQNAgent.skip.net.body.output.[features]", "main");
        config.learner.skip.loss_weight = 0;
        config.skip.head.projection = config.head.projection;
        const auto on = make(config, 99);
        update(off);
        update(on);
        const auto left = (*off->GetTensorDictFunction("policy-net.forward"))(samples.obs).At("q");
        const auto right = (*on->GetTensorDictFunction("policy-net.forward"))(samples.obs).At("q");
        CHECK(torch::equal(left, right));
        { std::ofstream stream(file, std::ios::binary); anet::OutputArchive archive(stream); on->Save(archive); }
        config.skip.enabled = false;
        CHECK_THROWS(make(config, 100));
        std::filesystem::remove(file);
    }
}

TEST_CASE("Rainbow rejects invalid initialization during config loading", "[dqn][config][rfd0098][review-fix-rainbow]")
{
    // Networkを作る前に、認識した初期化modeの誤りを公開設定経路で拒否する。
    anet::ConfigData data;
    data.Set("RainbowAgent.head.init.mode", "unknown");
    REQUIRE_THROWS(dqn::RainbowAgentConfig(data));
}

TEST_CASE("Rainbow records all nested initialization settings", "[dqn][config][rfd0098][review-fix-rainbow]")
{
    // 外部の設定キー8件と設定記録が同じhead.init階層に揃うことを確かめる。
    anet::ConfigData data;
    data.Set("RainbowAgent.head.init.mode", "constant");
    data.Set("RainbowAgent.head.init.manual_gain", 1.5);
    data.Set("RainbowAgent.head.init.nonlinearity", "relu");
    data.Set("RainbowAgent.head.init.constant_val", 0.25);
    data.Set("RainbowAgent.head.init.trunc_std", 0.5);
    data.Set("RainbowAgent.head.init.trunc_a", -2.0);
    data.Set("RainbowAgent.head.init.trunc_b", 2.0);
    data.Set("RainbowAgent.head.init.variance_scale", 1.5);
    const dqn::RainbowAgentConfig config(data);
    CHECK(config.head.init.mode == "constant");
    CHECK(config.head.init.manual_gain == 1.5);
    CHECK(config.head.init.nonlinearity == "relu");
    CHECK(config.head.init.constant_val == 0.25);
    CHECK(config.head.init.trunc_std == 0.5);
    CHECK(config.head.init.trunc_a == -2.0);
    CHECK(config.head.init.trunc_b == 2.0);
    CHECK(config.head.init.variance_scale == 1.5);
    const auto& recorded = config.GetConfigData();
    CHECK(recorded.Get("head.init.mode") == "constant");
    CHECK(recorded.Get<double>("head.init.manual_gain") == 1.5);
    CHECK(recorded.Get("head.init.nonlinearity") == "relu");
    CHECK(recorded.Get<double>("head.init.constant_val") == 0.25);
    CHECK(recorded.Get<double>("head.init.trunc_std") == 0.5);
    CHECK(recorded.Get<double>("head.init.trunc_a") == -2.0);
    CHECK(recorded.Get<double>("head.init.trunc_b") == 2.0);
    CHECK(recorded.Get<double>("head.init.variance_scale") == 1.5);

    std::vector<std::string> recorded_keys;
    for (const auto& [key, value] : recorded.Map()) {
        if (key.starts_with("head.init.")) {
            recorded_keys.push_back(key);
        }
    }
    const std::vector<std::string> expected_keys{
        "head.init.mode", "head.init.manual_gain", "head.init.nonlinearity", "head.init.constant_val",
        "head.init.trunc_std", "head.init.trunc_a", "head.init.trunc_b", "head.init.variance_scale"};
    CHECK(recorded_keys == expected_keys);
}

TEST_CASE("DQN and Rainbow read Head nonlinearity with a linear default", "[dqn][config][rfd0193]")
{
    const bool explicit_relu = GENERATE(false, true);
    auto dqn_data = WithDefaultForwardConfig();
    anet::ConfigData rainbow_data;
    if (explicit_relu) {
        dqn_data.Set("DefaultDQNAgent.head.init.nonlinearity", "relu");
        rainbow_data.Set("RainbowAgent.head.init.nonlinearity", "relu");
    }
    const std::string expected = explicit_relu ? "relu" : "linear";
    const dqn::DefaultDQNAgentConfig dqn_config(dqn_data);
    const dqn::RainbowAgentConfig rainbow_config(rainbow_data);
    CHECK(dqn_config.head.init.nonlinearity == expected);
    CHECK(dqn_config.GetConfigData().Get("head.init.nonlinearity") == expected);
    CHECK(rainbow_config.head.init.nonlinearity == expected);
    CHECK(rainbow_config.GetConfigData().Get("head.init.nonlinearity") == expected);
}

TEST_CASE("Skip Head inherits variance scaling and can override its scale", "[dqn][config][rfd0193]")
{
    const bool override_scale = GENERATE(false, true);
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.head.init.mode", "variance_scaling");
    data.Set("DefaultDQNAgent.head.init.nonlinearity", "tanh");
    data.Set("DefaultDQNAgent.head.init.variance_scale", 0.5);
    if (override_scale) {
        data.Set("DefaultDQNAgent.skip.head.init.variance_scale", 2.0);
    }
    const dqn::DefaultDQNAgentConfig config(data);
    CHECK(config.head.init.variance_scale == 0.5);
    CHECK(config.skip.head.init.mode == "variance_scaling");
    CHECK(config.skip.head.init.nonlinearity == "tanh");
    CHECK(config.skip.head.init.variance_scale == (override_scale ? 2.0 : 0.5));
}

TEST_CASE("DQN and Rainbow reject a zero Head variance scale with full config keys", "[dqn][config][rfd0193]")
{
    const std::string key = GENERATE(
        "DefaultDQNAgent.head.init.variance_scale", "DefaultDQNAgent.skip.head.init.variance_scale",
        "DefaultDQNAgent.intrinsic.value.head.init.variance_scale", "RainbowAgent.head.init.variance_scale");
    auto data = WithDefaultForwardConfig();
    data.Set(key, 0.0);
    const auto error = Catch::Matchers::ContainsSubstring("Invalid " + key + "=0; expected > 0.");
    if (key.starts_with("RainbowAgent.")) {
        CHECK_THROWS_WITH(dqn::RainbowAgentConfig(data), error);
    } else {
        CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data), error);
    }
}

TEST_CASE("Disabled skip uses common projection validation", "[dqn][config][rfd0098][review-fix-projection]")
{
    // 無効のskipでも明示されたboolとsigmaを検証し、負値の診断に論理キーを残す。
    for (const auto& [key, value] : std::vector<std::pair<std::string, std::string>>{
        {"noisy.enabled", "sometimes"}, {"noisy.sigma0", "-0.5"},
        {"noisy.sigma0", "nan"}, {"noisy.sigma0", "inf"}}) {
        auto data = WithDefaultForwardConfig();
        data.Set("DefaultDQNAgent.skip.head.projection." + key, value);
        CAPTURE(key, value);
        CHECK_THROWS(dqn::DefaultDQNAgentConfig(data));
    }
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.skip.head.projection.noisy.sigma0", -0.5);
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data),
        Catch::Matchers::ContainsSubstring("DefaultDQNAgent.skip.head.projection.noisy.sigma0"));
    data.Set("DefaultDQNAgent.skip.head.projection.noisy.sigma0", 0.0);
    CHECK_NOTHROW(dqn::DefaultDQNAgentConfig(data));
}

TEST_CASE("Primary initialization is validated before skip overrides", "[dqn][config][rfd0098][weight-init-validation]")
{
    // primaryの未知modeは、skip側で上書きしても設定読込時に拒否する。
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.head.init.mode", "not_an_initializer");
    data.Set("DefaultDQNAgent.skip.head.init.mode", "constant");
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data),
        Catch::Matchers::ContainsSubstring("DefaultDQNAgent.head.init.mode"));
}

TEST_CASE("Head configuration uses nested init and projection keys with skip inheritance", "[dqn][config][rfd0098][rfd0098-config-head]")
{
    // 公開設定の読み込みで新しい階層と未指定値の継承を確かめる。
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.head.use_dueling", false);
    data.Set("DefaultDQNAgent.skip.network_mode", "action_in");
    data.Set("DefaultDQNAgent.head.init.mode", "constant");
    data.Set("DefaultDQNAgent.head.init.constant_val", 0.25);
    data.Set("DefaultDQNAgent.head.projection.noisy.enabled", true);
    data.Set("DefaultDQNAgent.head.projection.noisy.sigma0", 0.3);
    data.Set("DefaultDQNAgent.skip.head.init.constant_val", 0.5);
    data.Set("DefaultDQNAgent.skip.head.projection.noisy.sigma0", 0.2);
    const dqn::DefaultDQNAgentConfig config(data);
    CHECK_FALSE(config.head.use_dueling);
    CHECK(config.skip.network_mode == dqn::SkipNetworkMode::ActionIn);
    CHECK(config.head.init.mode == "constant");
    CHECK(config.head.init.constant_val == Catch::Approx(0.25));
    CHECK(config.head.projection.Get<bool>("noisy.enabled", false));
    CHECK(config.skip.head.init.mode == "constant");
    CHECK(config.skip.head.init.constant_val == Catch::Approx(0.5));
    CHECK(config.skip.head.projection.Get<bool>("noisy.enabled", false));
    CHECK(config.skip.head.projection.Get<double>("noisy.sigma0", 0) == Catch::Approx(0.2));
    for (const auto& key : {"DefaultDQNAgent.head.use_dueling", "DefaultDQNAgent.skip.network_mode"}) {
        auto invalid = data;
        invalid.Set(key, "unsupported");
        CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(invalid), Catch::Matchers::ContainsSubstring(key));
    }
}

TEST_CASE("Skip network mode and action binding must agree at Agent construction", "[dqn][config][rfd0098][rfd0098-config-bind]")
{
    ScopedNoopMetricsLogger metrics_logger;
    // 同じ公開構築経路の正例を先に通し、逆のbindだけを契約違反にする。
    for (const std::string mode : {"action_in", "action_out"}) {
        for (const bool enabled : {false, true}) {
            CAPTURE(mode, enabled);
            auto data = WithDefaultForwardConfig();
            data.Set("DefaultDQNAgent.quantile_mode", "none");
            data.Set("DefaultDQNAgent.head.use_dueling", false);


            data.Set("DefaultDQNAgent.learner.use_fused_optimizer", false);
            data.Set("DefaultDQNAgent.skip.enabled", enabled);
            data.Set("DefaultDQNAgent.skip.max_length", 3);
            data.Set("DefaultDQNAgent.skip.network_mode", mode);
            data.Set("DefaultDQNAgent.net.block.[Flatten].type", kFixtureFlatten);
            data.Set("DefaultDQNAgent.net.branch.[main].bind", kVectorKey);
            data.Set("DefaultDQNAgent.net.branch.[main].structure", kFixtureFlatten);
            data.Set("DefaultDQNAgent.net.body.output.[features]", "main");
            data.Set("DefaultDQNAgent.skip.net.body.output.[features]", "main");
            const auto bind_key = "DefaultDQNAgent.skip.net.branch.[main].bind";
            data.Set(bind_key, mode == "action_in" ? "features, action" : "features");
            const auto make = [&] {
                return std::make_shared<dqn::DefaultDQNAgent>(dqn::DefaultDQNAgentConfig(data),
                    anet::nn::NetworkConfig(data, "DefaultDQNAgent.net"), rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1},
                    MakeIqnTracerEnvSpec(), torch::kCPU, 98);
            };
            REQUIRE_NOTHROW(make());
            data.Set(bind_key, mode == "action_in" ? "features" : "features, action");
            if (enabled) CHECK_THROWS_WITH(make(), Catch::Matchers::ContainsSubstring("DefaultDQNAgent.skip.network_mode='" + mode + "'"));
            else REQUIRE_NOTHROW(make());
        }
    }
}

TEST_CASE("Disabled skip still validates explicit initialization settings", "[dqn][config][rfd0098]")
{
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.skip.head.init.mode", "not_an_initializer");
    CHECK_THROWS(dqn::DefaultDQNAgentConfig(data));
}

TEST_CASE("Skip recognized settings reject invalid values even when disabled", "[dqn][config][rfd0098]")
{
    for (const auto& [key, value] : std::vector<std::pair<std::string, std::string>>{
        {"skip.max_length", "0"}, {"skip.head.use_dueling", "average"}, {"learner.skip.feature_gradient", "other"},
        {"learner.skip.loss_weight", "-1"}, {"learner.skip.priority_lambda", "nan"},
        {"skip.head.projection.noisy.sigma0", "-0.5"}, {"skip.head.init.trunc_std", "0"},
        {"skip.head.init.variance_scale", "0"}, {"skip.head.init.nonlinearity", "selu"}}) {
        auto data = WithDefaultForwardConfig();
        data.Set("DefaultDQNAgent." + key, value);
        CAPTURE(key, value);
        CHECK_THROWS(dqn::DefaultDQNAgentConfig(data));
    }
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.skip.enabled", true);
    data.Set("DefaultDQNAgent.skip.max_length", 3);
    data.Set("DefaultDQNAgent.skip.net.body.output.[features]", "vector");
    data.Set("DefaultDQNAgent.learner.use_amp", true);
    data.Set("DefaultDQNAgent.learner.use_amp_bf16", false);
    CHECK_THROWS(dqn::DefaultDQNAgentConfig(data));
}

TEST_CASE("DefaultDQNAgent IQN acts through the public ConfigData path", "[dqn][iqn][tracer]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const auto config_data = MakeIqnTracerConfigData();
    const auto env_spec = MakeIqnTracerEnvSpec();
    const rl::BatchEnvSpec batch_env_spec{ 2, 1 };
    dqn::DefaultDQNAgentFactory factory;
    auto agent = factory.CreateAgent(
        env_spec,
        batch_env_spec,
        torch::Device(torch::kCPU),
        WithDefaultForwardConfig(config_data),
        nullptr,
        123);
    auto actor = agent->CreateActor(rl::ActorRequest{.batch_env_spec = batch_env_spec, .env_spec = env_spec, .device = torch::Device(torch::kCPU), .seed = 123, .actor_key = "train"});

    const auto flags = torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kBool));
    rl::BatchState state(
        anet::TensorDict{ { kVectorKey, torch::tensor({
            { 1.0f, 2.0f, 3.0f, 4.0f },
            { 4.0f, 3.0f, 2.0f, 1.0f },
        }) } },
        flags,
        flags,
        flags);
    REQUIRE_FALSE(state.obs.Contains(anet::rl::dqn::kKey_Taus));

    auto action_info = std::dynamic_pointer_cast<dqn::DQNActionInfo>(
        actor->MakeAction(rl::StepCounts{}, state, {.nn_trace = true}));

    REQUIRE(action_info != nullptr);
    SaveRfd0170Trace(action_info->GetAuxData(), "iqn");
    constexpr char kTauInputTrace[] = "tau_embedding/00_Input";
    const auto trace = rl::ExtractNnTrace(action_info->GetAuxData());
    REQUIRE(trace.Contains(kTauInputTrace));
    CHECK(trace.At(kTauInputTrace).dim() == 2);
    REQUIRE(ShapeOf(action_info->GetAction()) == std::vector<int64_t>{ 2 });
    const auto& aux = action_info->GetAuxData();
    REQUIRE(ShapeOf(aux.at("q_values")) == std::vector<int64_t>{ 2, 2 });
    REQUIRE(ShapeOf(aux.at("q_quantiles")) == std::vector<int64_t>{ 2, 2, 3 });
    CHECK(torch::isfinite(aux.at("q_values")).all().item<bool>());
    CHECK(torch::isfinite(aux.at("q_quantiles")).all().item<bool>());
    CHECK_FALSE(state.obs.Contains(anet::rl::dqn::kKey_Taus));
}

TEST_CASE("DefaultDQNAgentFactory rejects IQN without a taus bind", "[dqn][iqn][config]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    auto config_data = MakeIqnTracerConfigData();
    config_data.Set("DefaultDQNAgent.net.branch.[tau_embedding].bind", kVectorKey);

    dqn::DefaultDQNAgentFactory factory;
    CHECK_THROWS_WITH(
        factory.CreateAgent(
            MakeIqnTracerEnvSpec(),
            rl::BatchEnvSpec{ 2, 1 },
            torch::Device(torch::kCPU),
            WithDefaultForwardConfig(config_data),
            nullptr,
            123),
        Catch::Matchers::ContainsSubstring("DefaultDQNAgent.quantile_mode")
            && Catch::Matchers::ContainsSubstring("DefaultDQNAgent.net.branch.[*].bind"));
}

TEST_CASE("DefaultDQNAgentFactory rejects IQN without a selected tau branch", "[dqn][iqn][config]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.quantile_mode", "iqn");
    config_data.Set("DefaultDQNAgent.net.branch.[main].bind", kVectorKey);
    config_data.Set("DefaultDQNAgent.net.body.output.[features]", "main");

    dqn::DefaultDQNAgentFactory factory;
    CHECK_THROWS_WITH(
        factory.CreateAgent(
            MakeIqnTracerEnvSpec(),
            rl::BatchEnvSpec{ 2, 1 },
            torch::Device(torch::kCPU),
            WithDefaultForwardConfig(config_data),
            nullptr,
            123),
        Catch::Matchers::ContainsSubstring("DefaultDQNAgent.quantile_mode")
            && Catch::Matchers::ContainsSubstring("DefaultDQNAgent.net.branch.[*].bind"));
}

TEST_CASE("DefaultDQNAgentFactory rejects QR with a taus bind", "[dqn][qr][config]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config_data = MakeIqnTracerConfigData();
    anet::ConfigData source;
    source.Set("Trial.quantile_mode", "qr");
    source.Set("Trial.net.branch.[tau_embedding].bind", anet::rl::dqn::kKey_Taus);
    config_data.CopyValueFrom("DefaultDQNAgent.quantile_mode", source, "Trial.quantile_mode");
    config_data.CopyValueFrom("DefaultDQNAgent.net.branch.[tau_embedding].bind", source, "Trial.net.branch.[tau_embedding].bind");

    dqn::DefaultDQNAgentFactory factory;
    CHECK_THROWS_WITH(
        factory.CreateAgent(
            MakeIqnTracerEnvSpec(),
            rl::BatchEnvSpec{ 2, 1 },
            torch::Device(torch::kCPU),
            WithDefaultForwardConfig(config_data),
            nullptr,
            123),
        Catch::Matchers::ContainsSubstring("Trial.quantile_mode")
            && Catch::Matchers::ContainsSubstring(
                "Trial.net.branch.[tau_embedding].bind"));
}

TEST_CASE("DefaultDQNAgentFactory rejects non-quantile mode with a taus bind", "[dqn][config]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config_data = MakeIqnTracerConfigData();
    config_data.Set("DefaultDQNAgent.quantile_mode", "none");

    dqn::DefaultDQNAgentFactory factory;
    CHECK_THROWS_WITH(
        factory.CreateAgent(
            MakeIqnTracerEnvSpec(),
            rl::BatchEnvSpec{ 2, 1 },
            torch::Device(torch::kCPU),
            WithDefaultForwardConfig(config_data),
            nullptr,
            123),
        Catch::Matchers::ContainsSubstring("DefaultDQNAgent.quantile_mode")
            && Catch::Matchers::ContainsSubstring(
                "DefaultDQNAgent.net.branch.[tau_embedding].bind"));
}

TEST_CASE("DefaultDQNAgent IQN rejects a dead tau fusion branch", "[dqn][iqn][tracer]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    auto config_data = MakeIqnTracerConfigData();
    config_data.Set("DefaultDQNAgent.net.body.output.[features]", "main");
    const auto env_spec = MakeIqnTracerEnvSpec();

    CHECK_THROWS_WITH(
        dqn::DefaultDQNAgent(
            dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(config_data)),
            anet::nn::NetworkConfig(config_data, "DefaultDQNAgent.net"),
            rl::BatchEnvSpec{ 2, 1 },
            env_spec,
            torch::Device(torch::kCPU),
            123),
        Catch::Matchers::ContainsSubstring("IQNHead expected rank-3")
            && Catch::Matchers::ContainsSubstring("IQN fusion branch"));
}

TEST_CASE("DefaultDQNAgent IQN learner updates through the public learner path", "[dqn][iqn][learner]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();

    auto run_update = [](
        int64_t current_num_taus,
        int64_t target_num_taus = 3,
        bool use_per = true,
        bool use_tbo = false,
        torch::Device device = torch::Device(torch::kCPU),
        bool use_amp_bf16 = false) {
        auto config_data = MakeIqnTracerConfigData();
        config_data.Set("DefaultDQNAgent.learner.iqn.current_taus.num_taus", current_num_taus);
        config_data.Set("DefaultDQNAgent.learner.iqn.current_taus.sample_mode", "fixed");
        config_data.Set("DefaultDQNAgent.learner.iqn.target_taus.num_taus", target_num_taus);
        config_data.Set("DefaultDQNAgent.learner.iqn.target_taus.sample_mode", "fixed");
        config_data.Set("DefaultDQNAgent.target_policy.tau_rule.num_taus", 4);
        config_data.Set("DefaultDQNAgent.target_policy.tau_rule.sample_mode", "fixed");
        config_data.Set("DefaultDQNAgent.learner.update_warmup_steps", 0);
        config_data.Set("DefaultDQNAgent.learner.update_interval", 1);
        config_data.Set("DefaultDQNAgent.learner.use_n_step", false);
        config_data.Set("DefaultDQNAgent.learner.use_per", use_per);
        config_data.Set("DefaultDQNAgent.learner.use_tbo", use_tbo);
        config_data.Set("DefaultDQNAgent.learner.use_amp", use_amp_bf16);
        config_data.Set("DefaultDQNAgent.learner.use_amp_bf16", use_amp_bf16);
        config_data.Set("DefaultDQNAgent.learner.use_fused_optimizer", false);
        config_data.Set("DefaultDQNAgent.learner.policy_churn.probe.batch_size", 2);
        config_data.Set("DefaultDQNAgent.learner.policy_churn.iqn.num_taus", 5);

        const auto env_spec = MakeIqnTracerEnvSpec();
        const rl::BatchEnvSpec batch_env_spec{ 2, 1 };
        auto agent = std::make_shared<dqn::DefaultDQNAgent>(
            dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(config_data)),
            anet::nn::NetworkConfig(config_data, "DefaultDQNAgent.net"),
            batch_env_spec,
            env_spec,
            device,
            321);
        auto learner = agent->CreateLearner();
        std::vector<rl::ScalarMetricSubscription> churn_subscriptions;
        for (const auto* key : std::span(dqn::kPolicyChurnMetricKeys).first(dqn::PolicyChurnMetric::TargetSyncAge)) {
            churn_subscriptions.push_back(rl::ScalarMetricSubscription{
                .source_key = key,
                .event = rl::EventType::LEARN,
                .target = rl::EventField::UPDATE_RESULT,
                .interval = 1,
                .scope = rl::RunnerScope::TRAIN,
            });
        }
        agent->ConfigureScalarMetricSubscriptions(churn_subscriptions);

        const auto flags = torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kBool));
        const auto episode_start = torch::ones({ 2 }, torch::TensorOptions().dtype(torch::kBool));
        rl::BatchState state(
            anet::TensorDict{ { kVectorKey, torch::tensor({
                { 1.0f, 2.0f, 3.0f, 4.0f },
                { 4.0f, 3.0f, 2.0f, 1.0f },
            }) } },
            flags,
            flags,
            episode_start);
        rl::BatchState next_state(
            anet::TensorDict{ { kVectorKey, torch::tensor({
                { 1.5f, 2.5f, 3.5f, 4.5f },
                { 3.5f, 2.5f, 1.5f, 0.5f },
            }) } },
            flags,
            flags,
            flags);
        auto action_info = std::make_shared<rl::BatchActionInfo>(
            torch::tensor({ 0, 1 }, torch::TensorOptions().dtype(torch::kInt64)));
        rl::BatchExperience experience(
            state,
            action_info,
            torch::tensor({ 0.25f, -0.5f }),
            next_state);
        rl::BatchState final_state(
            anet::TensorDict{ { kVectorKey, torch::tensor({
                { 2.0f, 3.0f, 4.0f, 5.0f },
                { 3.0f, 2.0f, 1.0f, 0.0f },
            }) } },
            flags,
            flags,
            flags);
        rl::BatchExperience next_experience(
            next_state,
            action_info,
            torch::tensor({ -0.25f, 0.5f }),
            final_state);

        rl::StepCounts counts;
        counts.exp_step = 0;
        REQUIRE(learner->UpdateFromBatch(counts, experience).empty());
        counts.exp_step = 2;
        auto results = learner->UpdateFromBatch(counts, next_experience);

        REQUIRE(results.size() == 1);
        auto result = std::dynamic_pointer_cast<const dqn::BatchUpdateResult>(results.front());
        REQUIRE(result != nullptr);
        CHECK(ShapeOf(result->loss).empty());
        CHECK(ShapeOf(result->td_error) == std::vector<int64_t>{ 2 });
        CHECK(torch::isfinite(result->loss).all().item<bool>());
        CHECK(torch::isfinite(result->td_error).all().item<bool>());
        // stride-0へexpandしたfixed midpoint tausを実IQN headの最大3 forwardへ通す。
        for (const auto* key : std::span(dqn::kPolicyChurnMetricKeys).first(dqn::PolicyChurnMetric::TargetSyncAge)) {
            const auto churn = result->GetScalar(key);
            REQUIRE(churn.has_value());
            CHECK(std::isfinite(*churn));
        }
        if (use_per) {
            CHECK(ShapeOf(result->per_priorities) == std::vector<int64_t>{ 2 });
            CHECK(torch::isfinite(result->per_priorities).all().item<bool>());
            const auto expected_priorities = dqn::MakePerRawPriorityBatch(
                result->td_error, 1.0e-6f, false, 50.0f).priorities.to(torch::kFloat32).cpu();
            CHECK(torch::equal(result->per_priorities, expected_priorities));
            CHECK(result->per_update_result.applied_count == 2);
            CHECK(result->per_update_result.stale_count == 0);
        } else {
            CHECK_FALSE(result->per_priorities.defined());
        }
        CHECK(torch::isfinite(result->q_std).all().item<bool>());
        CHECK(torch::isfinite(result->max_q).all().item<bool>());
        CHECK(torch::isfinite(result->q_sa).all().item<bool>());
        CHECK(torch::isfinite(result->q_gap).all().item<bool>());
        CHECK(torch::isfinite(result->q_gap_rel).all().item<bool>());
        REQUIRE(result->iqn_diagnostics.defined());
        CHECK(result->iqn_diagnostics.device().is_cpu());
        CHECK(result->iqn_diagnostics.scalar_type() == torch::kFloat32);
        CHECK(ShapeOf(result->iqn_diagnostics) == std::vector<int64_t>{ 7 });
        const auto current_mc_scale = result->GetScalar("iqn_current_mc_scale", -1);
        REQUIRE(current_mc_scale.has_value());
        if (current_num_taus >= 2) {
            CHECK(std::isfinite(*current_mc_scale));
        } else {
            CHECK(std::isnan(*current_mc_scale));
        }
        const auto target_mc_scale = result->GetScalar("iqn_target_mc_scale", -1);
        REQUIRE(target_mc_scale.has_value());
        if (target_num_taus >= 2) {
            CHECK(std::isfinite(*target_mc_scale));
        } else {
            CHECK(std::isnan(*target_mc_scale));
        }
        const auto priority_mc_ratio = result->GetScalar("iqn_priority_mc_ratio", -1);
        REQUIRE(priority_mc_ratio.has_value());
        if (current_num_taus >= 2 && target_num_taus >= 2) {
            CHECK(std::isfinite(*priority_mc_ratio));
        } else {
            CHECK(std::isnan(*priority_mc_ratio));
        }
        const auto upper_tail_spearman = result->GetScalar("upper_tail_priority_spearman", -1);
        REQUIRE(upper_tail_spearman.has_value());
        if (use_per && current_num_taus >= 2) {
            CHECK(std::isfinite(*upper_tail_spearman));
        } else {
            CHECK(std::isnan(*upper_tail_spearman));
        }

        const auto initial_count = result->GetScalar("per_sample_initial_count", -1);
        REQUIRE(initial_count.has_value());
        CHECK(*initial_count == Catch::Approx(use_per ? 2.0f : 0.0f));
        for (const char* key : {
            "iqn_first_priority_mc_ratio",
            "iqn_first_pair_abs_td",
            "iqn_first_cancellation_ratio",
            "iqn_first_quantile_loss_norm",
        }) {
            const auto value = result->GetScalar(key, -1);
            REQUIRE(value.has_value());
            if (!use_per || (std::string_view(key) == "iqn_first_priority_mc_ratio"
                    && (current_num_taus < 2 || target_num_taus < 2))) {
                CHECK(std::isnan(*value));
            } else {
                CHECK(std::isfinite(*value));
            }
        }
        CHECK_FALSE(experience.state.obs.Contains(anet::rl::dqn::kKey_Taus));
        CHECK_FALSE(experience.next_state.obs.Contains(anet::rl::dqn::kKey_Taus));
        CHECK_FALSE(next_experience.state.obs.Contains(anet::rl::dqn::kKey_Taus));
        CHECK_FALSE(next_experience.next_state.obs.Contains(anet::rl::dqn::kKey_Taus));
        return result;
    };

    SECTION("current and target sample counts may differ")
    {
        const auto result = run_update(2);
        CHECK(result->q_std.item<float>() >= 0.0f);
    }

    SECTION("one current sample reports finite zero q_std")
    {
        const auto result = run_update(1);
        CHECK(result->q_std.item<float>() == Catch::Approx(0.0f));
    }

    SECTION("one target sample leaves only target-dependent ratios undefined")
    {
        run_update(2, 1);
    }

    SECTION("PER disabled keeps general IQN diagnostics and disables first-update diagnostics")
    {
        run_update(2, 3, false);
    }

    SECTION("TBO keeps IQN diagnostics in the learner priority path")
    {
        run_update(2, 3, true, true);
    }

    SECTION("BF16 diagnostics use one CPU float32 pack on each available device")
    {
        run_update(2, 3, true, false, torch::Device(torch::kCPU), true);
        if (torch::cuda::is_available()) {
            run_update(2, 3, true, false, torch::Device(torch::kCUDA, 0), true);
        }
    }
}

TEST_CASE("DefaultDQNAgent resolves named Actor snapshot schedules", "[dqn][actor][snapshot]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeDeviceForwardDefaultDqnConfig();
    config.actor["snapshot"] = config.actor.at("train");
    config.actor["eval_clone"] = config.actor.at("eval");
    auto& snapshot = config.actor["snapshot"];
    snapshot.clone_model = true;
    snapshot.sync_interval = anet::ProfiledValueConfig<rl::step_t>{};
    snapshot.sync_interval->value = 7;
    config.actor["eval_clone"].clone_model = true;
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(config,
        MakeAgentForwardNetworkConfig(), rl::BatchEnvSpec{1, 1}, MakeLearnerEnvSpec(),
        torch::Device(torch::kCPU), 123);
    auto flags = torch::zeros({1}, torch::TensorOptions().dtype(torch::kBool));
    rl::BatchState state(anet::TensorDict{{kVectorKey, torch::tensor({{1.0f, 2.0f}})}}, flags, flags, flags);
    const auto read_interval = [&](const std::string& key) {
        auto actor = agent->CreateActor(rl::ActorRequest{.batch_env_spec = {1, 1},
            .env_spec = MakeLearnerEnvSpec(), .device = torch::Device(torch::kCPU), .seed = 123, .actor_key = key});
        const auto info = std::dynamic_pointer_cast<dqn::DQNActionInfo>(actor->MakeAction(rl::StepCounts{}, state));
        REQUIRE(info != nullptr);
        const auto interval = info->GetScalar("train_actor_snapshot_interval");
        REQUIRE(interval.has_value());
        return *interval;
    };
    CHECK(std::isnan(read_interval("train")));
    CHECK(read_interval("snapshot") == Catch::Approx(7.0f));
    CHECK(std::isnan(read_interval("eval_clone")));
}

TEST_CASE("DefaultDQNAgent rejects an effective shared Actor on another device", "[dqn][actor][snapshot][device]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeDeviceForwardDefaultDqnConfig();
    config.actor["train"].clone_model = false;
    const auto batch_env_spec = rl::BatchEnvSpec{ 1, 1 };
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(
        config,
        MakeAgentForwardNetworkConfig(),
        batch_env_spec,
        MakeLearnerEnvSpec(),
        torch::Device(torch::kCPU),
        123);

    CHECK_THROWS(agent->CreateActor(rl::ActorRequest{.batch_env_spec = batch_env_spec, .env_spec = MakeLearnerEnvSpec(), .device = torch::Device(torch::kCUDA, 0), .seed = 123, .actor_key = "train"}));
}

TEST_CASE("DefaultDQNAgent decides whether an Eval EnvSpec is acceptable", "[dqn][actor][env_spec]")
{
    ScopedNoopMetricsLogger metrics_logger;
    const auto batch_env_spec = rl::BatchEnvSpec{ 1, 1 };
    const auto train_env_spec = MakeLearnerEnvSpec();
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(
        MakeDeviceForwardDefaultDqnConfig(),
        MakeAgentForwardNetworkConfig(),
        batch_env_spec,
        train_env_spec,
        torch::Device(torch::kCPU),
        123);

    auto incompatible_eval_spec = train_env_spec;
    incompatible_eval_spec.action_spec.value_labels.push_back("incompatible");
    CHECK_THROWS(agent->CreateActor(rl::ActorRequest{.batch_env_spec = batch_env_spec, .env_spec = incompatible_eval_spec, .device = torch::Device(torch::kCPU), .seed = 123, .actor_key = "eval"}));
}

TEST_CASE("DefaultDQNAgent creates the initial snapshot from an auto-loaded network", "[dqn][actor][snapshot][serialize][rfd0091]")
{
    ScopedNoopMetricsLogger metrics_logger;
    const auto env_spec = MakeLearnerEnvSpec();
    const auto batch_env_spec = rl::BatchEnvSpec{ 1, 1 };
    auto source_config = MakeDeviceForwardDefaultDqnConfig();
    source_config.learner.learning_rate.value = 1e-4f;
    auto source_agent = std::make_shared<dqn::DefaultDQNAgent>(
        source_config,
        MakeAgentForwardNetworkConfig(),
        batch_env_spec,
        env_spec,
        torch::Device(torch::kCPU),
        123);
    auto source_forward = source_agent->GetTensorDictFunction("policy-net.forward");
    REQUIRE(source_forward.has_value());
    const auto obs = anet::TensorDict{ { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }) } };
    const auto expected_q = (*source_forward)(obs).At("q").detach().clone();

    const auto unique_suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto checkpoint = std::filesystem::temp_directory_path()
        / ("anet_default_dqn_snapshot_" + std::to_string(unique_suffix) + ".bin");
    {
        std::ofstream stream(checkpoint, std::ios::binary);
        REQUIRE(stream);
        anet::OutputArchive archive(stream, checkpoint.string());
        source_agent->Save(archive);
    }

    const bool change_alpha = GENERATE(false, true);
    auto loaded_config = MakeDeviceForwardDefaultDqnConfig();
    loaded_config.learner.learning_rate.value = change_alpha ? 5e-5f : source_config.learner.learning_rate.value;
    loaded_config.auto_load_file = checkpoint.string();
    loaded_config.actor["train"].clone_model = true;
    std::shared_ptr<dqn::DefaultDQNAgent> loaded_agent;
    {
        // auto_load_file から Learner::Load へ到達し、snapshot 構築前に設定が適用される。
        anet::test::LogCaptureGuard logs;
        loaded_agent = std::make_shared<dqn::DefaultDQNAgent>(
            loaded_config,
            MakeAgentForwardNetworkConfig(),
            batch_env_spec,
            env_spec,
            torch::Device(torch::kCPU),
            456);
        std::ifstream reloaded_stream(checkpoint, std::ios::binary);
        REQUIRE(reloaded_stream);
        anet::InputArchive reloaded_archive(reloaded_stream, checkpoint.string());
        loaded_agent->Load(reloaded_archive);
        logs.Flush();
        const auto differences = std::count_if(logs.Records().begin(), logs.Records().end(), [](const auto& record) {
            return record.level == wxLOG_Message
                && record.message.find("Learner::Load: optimizer option differs") != std::string::npos;
        });
        CHECK(differences == (change_alpha ? 1 : 0));
        CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 0);
        if (change_alpha) {
            CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
                {"group=0", "option=lr", "key=DefaultDQNAgent.learner.learning_rate", "checkpoint=0.0001", "config=5e-05"}));
        }
        CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
            {"Learner::Load: optimizer options applied from config.", "group=0",
                change_alpha ? "DefaultDQNAgent.learner.learning_rate=5e-05" : "DefaultDQNAgent.learner.learning_rate=0.0001"}));
    }
    auto actor = loaded_agent->CreateActor(rl::ActorRequest{.batch_env_spec = batch_env_spec, .env_spec = env_spec, .device = torch::Device(torch::kCPU), .seed = 123, .actor_key = "train"});
    auto flags = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kBool));
    rl::BatchState state(obs, flags, flags, flags);
    const auto action_info = actor->MakeAction(rl::StepCounts{}, state);

    CHECK(torch::allclose(action_info->GetAuxData().at("q_values"), expected_q));
    std::filesystem::remove(checkpoint);
}

TEST_CASE("RainbowAgent TensorDictFunction accepts CPU input on CUDA agent", "[dqn][network_model][device]")
{
    if (!torch::cuda::is_available()) return;

    ScopedNoopMetricsLogger metrics_logger;
    const torch::Device device(torch::kCUDA, 0);
    auto env_spec = MakeLearnerEnvSpec();
    auto agent = std::make_shared<dqn::RainbowAgent>(
        MakeDeviceForwardRainbowConfig(),
        MakeAgentForwardNetworkConfig(),
        rl::BatchEnvSpec{ 1, 1 },
        env_spec,
        device,
        123);

    auto forward = agent->GetTensorDictFunction("policy-net.forward");
    REQUIRE(forward.has_value());

    auto obs = anet::TensorDict{
        { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)) },
    };
    auto out = (*forward)(obs);

    REQUIRE(out.Get("q").has_value());
    CHECK(out.At("q").device().type() == torch::kCUDA);
}

TEST_CASE("RainbowAgentFactory reads its class-owned net tree", "[dqn][rainbow][config][tracer]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    anet::ConfigData config_data;
    config_data.Set("RainbowAgent.use_dueling_net", false);
    config_data.Set("RainbowAgent.learner.replay_capacity", 16);
    config_data.Set("RainbowAgent.learner.replay_batch_size", 2);
    config_data.Set("RainbowAgent.net.branch.[main].bind", kVectorKey);
    config_data.Set("RainbowAgent.net.body.output.[features]", "main");

    dqn::RainbowAgentFactory factory;
    const auto agent = factory.CreateAgent(
        MakeIqnTracerEnvSpec(),
        rl::BatchEnvSpec{ 2, 1 },
        torch::Device(torch::kCPU),
        config_data,
        nullptr,
        123);

    REQUIRE(agent != nullptr);
}

TEST_CASE("Rainbow CPU learning preserves loss and linear parameter checksum", "[rfd0128][dqn][rainbow]")
{
    ScopedNoopMetricsLogger metrics_logger;
    const bool quantile = GENERATE(false, true);
    // Agent の named seed と、torch の重み初期化の seed をどちらも固定する。
    torch::manual_seed(128);
    auto config = MakeDeviceForwardRainbowConfig();
    config.use_qr = quantile;
    config.num_quantiles = quantile ? 4 : 1;
    config.learner.num_quantiles = config.num_quantiles;
    config.learner.update_warmup_steps = 0;
    config.learner.replay_ratio = -1.0f;
    config.learner.update_interval = 1;
    config.learner.use_per = false;
    config.learner.use_n_step = false;
    config.learner.use_amp = false;
    config.learner.use_rb_prefetch = false;
    auto agent = std::make_shared<dqn::RainbowAgent>(config, MakeAgentForwardNetworkConfig(),
        rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, MakeLearnerEnvSpec(), torch::kCPU, 128);
    const auto learner = agent->CreateLearner();
    std::vector<float> losses;
    for (rl::step_t step = 0; step < 8; ++step) {
        // 全 lane の同じ終端遷移を使い、抽選や target 方策の乱数に期待値を依存させない。
        auto experience = MakeDeterminismExperience(step, 2);
        experience.state.obs.Set(kVectorKey, torch::ones({2, 2}));
        experience.state.episode_start = torch::ones({2}, torch::kBool);
        experience.next_state.obs.Set(kVectorKey, torch::ones({2, 2}));
        experience.next_state.done = torch::ones({2}, torch::kBool);
        experience.next_state.truncated = torch::zeros({2}, torch::kBool);
        experience.reward = torch::ones({2});
        const auto results = learner->UpdateFromBatch({.exp_step = step * 2}, experience);
        for (const auto& result : results) {
            const auto loss = result->GetScalar(kLossMetricKey);
            REQUIRE(loss.has_value());
            REQUIRE(std::isfinite(*loss));
            losses.push_back(*loss);
        }
    }
    REQUIRE_FALSE(losses.empty());
    // Body が恒等写像なので、零点と基底 2 点への出力から Head の全 weight/bias を読む。
    const auto forward = agent->GetTensorDictFunction(quantile ? kPolicyForwardDistributionKey : kPolicyForwardKey);
    REQUIRE(forward.has_value());
    const auto output = (*forward)(anet::TensorDict{{kVectorKey, torch::tensor({{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}})}});
    const auto values = output.At(quantile ? kQDistributionKey : kQValueKey).reshape({3, -1});
    const auto checksum = (values[0].sum() + (values[1] - values[0]).sum()
        + (values[2] - values[0]).sum()).item<float>();
    // 実装前の CPU・seed 128 で採取した値を固定し、再実行同士だけの比較にしない。
    const std::vector<float> expected = quantile
        ? std::vector<float>{0.158297181f, 0.156962171f, 0.155644596f, 0.154344544f, 0.153062165f, 0.151797161f, 0.150549382f}
        : std::vector<float>{1.36429894f, 1.36129022f, 1.35828161f, 1.35527301f, 1.3522644f, 1.3492558f, 1.3462472f};
    CHECK(losses == expected);
    CHECK(checksum == (quantile ? 2.81837368f : -0.843238711f));
}

TEST_CASE("Agent diagnostic resubscription preserves the probe random sequence", "[rfd0128][dqn][probe-resubscribe]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.quantile_mode", "none");
    data.Set("DefaultDQNAgent.head.use_dueling", false);
    data.Set("DefaultDQNAgent.head.init.mode", "constant");
    data.Set("DefaultDQNAgent.head.init.constant_val", 0.1f);


    data.Set("DefaultDQNAgent.learner.replay_capacity", 64);
    data.Set("DefaultDQNAgent.learner.replay_batch_size", 2);
    data.Set("DefaultDQNAgent.learner.update_warmup_steps", 8);
    data.Set("DefaultDQNAgent.learner.replay_ratio", -1.0f);
    data.Set("DefaultDQNAgent.learner.update_interval", 1);
    data.Set("DefaultDQNAgent.learner.use_per", false);
    data.Set("DefaultDQNAgent.learner.use_n_step", false);
    data.Set("DefaultDQNAgent.learner.use_rb_prefetch", false);
    data.Set("DefaultDQNAgent.learner.use_amp", false);
    data.Set("DefaultDQNAgent.learner.use_fused_optimizer", false);
    data.Set("DefaultDQNAgent.learner.use_grad_clip", false);
    data.Set("DefaultDQNAgent.learner.use_td_clip", false);
    data.Set("DefaultDQNAgent.learner.learning_rate", 0.01f);
    data.Set("DefaultDQNAgent.learner.weight_decay", 0.0f);
    data.Set("DefaultDQNAgent.learner.gamma", 0.0f);
    data.Set("DefaultDQNAgent.learner.policy_churn.probe.batch_size", 2);
    const auto make_agent = [&] {
        torch::manual_seed(128);
        return std::make_shared<dqn::DefaultDQNAgent>(dqn::DefaultDQNAgentConfig(data),
            MakeAgentForwardNetworkConfig(), rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, MakeLearnerEnvSpec(), torch::kCPU, 128);
    };
    auto actual = make_agent();
    auto reference = make_agent();
    const std::string key = "policy_churn_q_delta_abs_mean";
    const std::vector<rl::ScalarMetricSubscription> subscriptions{{
        .source_key = key, .event = rl::EventType::LEARN, .target = rl::EventField::UPDATE_RESULT,
        .interval = 1, .scope = rl::RunnerScope::TRAIN}};
    actual->ConfigureScalarMetricSubscriptions(subscriptions);
    reference->ConfigureScalarMetricSubscriptions(subscriptions);
    size_t compared = 0;
    bool observed_update = false;
    for (rl::step_t step = 0; step < 15; ++step) {
        // 抽選を何度か消費したあとに片方だけ購読し直し、続く抽選の系列を比べる。
        if (step == 10) actual->ConfigureScalarMetricSubscriptions(subscriptions);
        auto experience = MakeDeterminismExperience(step, 2);
        const float x = static_cast<float>(step + 1);
        experience.state.obs.Set(kVectorKey, torch::tensor({{x, 1.0f}, {x + 1.0f, 1.0f}}));
        experience.state.episode_start = torch::ones({2}, torch::kBool);
        experience.next_state.done = torch::ones({2}, torch::kBool);
        experience.next_state.truncated = torch::zeros({2}, torch::kBool);
        const auto values = actual->CreateLearner()->UpdateFromBatch({.exp_step = step * 2}, experience);
        const auto expected = reference->CreateLearner()->UpdateFromBatch({.exp_step = step * 2}, experience);
        REQUIRE(values.size() == expected.size());
        if (step < 10) continue;
        for (size_t index = 0; index < values.size(); ++index) {
            const auto value = values[index]->GetScalar(key);
            const auto expected_value = expected[index]->GetScalar(key);
            REQUIRE(value.has_value());
            REQUIRE(expected_value.has_value());
            REQUIRE(std::isfinite(*value));
            CHECK(*value == *expected_value);
            CHECK(*values[index]->GetScalar(kLossMetricKey) == *expected[index]->GetScalar(kLossMetricKey));
            observed_update = observed_update || *value > 0.0f;
            ++compared;
        }
    }
    CHECK(compared >= 5);
    CHECK(observed_update);
}

TEST_CASE("RainbowAgent omits DefaultDQN snapshot diagnostics", "[dqn][actor][snapshot][rainbow]")
{
    ScopedNoopMetricsLogger metrics_logger;
    const auto env_spec = MakeLearnerEnvSpec();
    const auto batch_env_spec = rl::BatchEnvSpec{ 1, 1 };
    auto agent = std::make_shared<dqn::RainbowAgent>(
        MakeDeviceForwardRainbowConfig(),
        MakeAgentForwardNetworkConfig(),
        batch_env_spec,
        env_spec,
        torch::Device(torch::kCPU),
        123);
    auto actor = agent->CreateActor(rl::ActorRequest{.batch_env_spec = batch_env_spec, .env_spec = env_spec, .device = torch::Device(torch::kCPU), .seed = 123, .actor_key = "train"});
    auto flags = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kBool));
    rl::BatchState state(
        anet::TensorDict{ { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }) } },
        flags,
        flags,
        flags);
    auto action_info = std::dynamic_pointer_cast<dqn::DQNActionInfo>(
        actor->MakeAction(rl::StepCounts{}, state));

    REQUIRE(action_info != nullptr);
    CHECK(ShapeOf(action_info->GetAction()) == std::vector<int64_t>{1});
    CHECK_FALSE(action_info->GetScalar("train_actor_snapshot_interval").has_value());
    CHECK_FALSE(action_info->GetScalar("train_actor_snapshot_age").has_value());
    CHECK_THROWS(agent->CreateActor(rl::ActorRequest{.batch_env_spec = batch_env_spec, .env_spec = env_spec, .device = torch::Device(torch::kCUDA, 0), .seed = 123, .actor_key = "train"}));
}

TEST_CASE("Decomposition config rejects invalid explicit values even when inactive", "[rfd0121][dqn][config]")
{
    const bool intrinsic = GENERATE(false, true);
    const auto invalid = GENERATE(Catch::Generators::values<std::pair<std::string, std::string>>({
        {"intrinsic.value_mode", kRfd0121Invalid}, {"learner.intrinsic.discount_mode", kRfd0121Invalid},
        {"learner.intrinsic.gamma", "0"}, {"learner.intrinsic.gamma", "1"}, {"learner.intrinsic.gamma", kRfd0121Nan},
        {"learner.intrinsic.gamma.type", kRfd0121Invalid}, {"learner.intrinsic.terminal_mode", kRfd0121Invalid},
        {"learner.intrinsic.loss_weight", "0"}, {"learner.intrinsic.loss_weight", kRfd0121Inf},
        {"learner.intrinsic.priority_lambda", "-1"}, {"learner.intrinsic.priority_lambda", kRfd0121Nan},
        {"learner.intrinsic.extrinsic_bootstrap_mode", kRfd0121Invalid}, {"learner.intrinsic.feature_gradient", "stop"},
        {"intrinsic.value.head.input_mode", kRfd0121Invalid}, {"intrinsic.value.head.use_dueling", kRfd0121Invalid},
        {"intrinsic.value.head.init.mode", kRfd0121Invalid}, {"intrinsic.value.head.init.manual_gain", kRfd0121Nan},
        {"intrinsic.value.head.init.variance_scale", "0"}, {"intrinsic.value.head.init.nonlinearity", "selu"},
        {"intrinsic.value.head.projection.noisy.enabled", kRfd0121Invalid}, {"intrinsic.value.head.projection.noisy.sigma0", "-1"},
        {"actor.[train].intrinsic.action_beta_mode", kRfd0121Invalid}, {"actor.[train].intrinsic.action_beta", "-1"},
        {"actor.[train].intrinsic.action_beta", kRfd0121Inf}, {"reward_scaler.dynamic_mode", kRfd0121Invalid},
        {"intrinsic.reward_scaler.dynamic_mode", kRfd0121Invalid}}));
    anet::ConfigData base;
    base.Set("DefaultDQNAgent.actor.[train].network", kRfd0121OnlineNetworkMode);
    auto data = WithDefaultForwardConfig(base);
    data.Set("DefaultDQNAgent.intrinsic.enabled", intrinsic);
    data.Set("DefaultDQNAgent." + invalid.first, invalid.second);
    CAPTURE(intrinsic, invalid.first, invalid.second);
    const std::string projection_prefix = "intrinsic.value.head.projection.";
    if (invalid.first.starts_with(projection_prefix)) {
        CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data),
            Catch::Matchers::ContainsSubstring("intrinsic.value.head.projection")
            && Catch::Matchers::ContainsSubstring(invalid.first.substr(projection_prefix.size())));
    } else {
        CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data), Catch::Matchers::ContainsSubstring(invalid.first));
    }
}

TEST_CASE("Decomposition mode errors precede numeric errors in other components", "[rfd0121][dqn][config][rfd0121-review-config]")
{
    const auto numeric_key = GENERATE("learner.intrinsic.loss_weight", "learner.intrinsic.gamma", "actor.[a].intrinsic.action_beta");
    const auto mode_key = GENERATE(
        "actor.[z].intrinsic.action_beta_mode", "intrinsic.value.head.init.mode", "intrinsic.value.head.init.nonlinearity");
    anet::ConfigData base;
    base.Set("DefaultDQNAgent.actor.[a].network", kRfd0121OnlineNetworkMode);
    base.Set("DefaultDQNAgent.actor.[z].network", kRfd0121OnlineNetworkMode);
    auto data = WithDefaultForwardConfig(base);
    data.Set("DefaultDQNAgent." + std::string(numeric_key), -1);
    data.Set("DefaultDQNAgent." + std::string(mode_key),
        std::string_view(mode_key).ends_with(".nonlinearity") ? "selu" : kRfd0121Invalid);
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data),
        Catch::Matchers::ContainsSubstring(mode_key) && Catch::Matchers::ContainsSubstring("expected"));
}

TEST_CASE("Decomposition validates individual fields before scaler combinations", "[rfd0121][dqn][config][rfd0121-review-order]")
{
    const bool intrinsic_scaler = GENERATE(false, true);
    const int error = GENERATE(0, 1, 2, 3, 4, 5, 6, 7);
    anet::ConfigData base;
    base.Set("DefaultDQNAgent.actor.[z].network", kRfd0121OnlineNetworkMode);
    auto data = WithDefaultForwardConfig(base);
    data.Set("DefaultDQNAgent.intrinsic.enabled", true);
    const std::string scaler = intrinsic_scaler ? "intrinsic.reward_scaler" : "reward_scaler";
    const std::string other_scaler = intrinsic_scaler ? "reward_scaler" : "intrinsic.reward_scaler";
    data.Set("DefaultDQNAgent." + scaler + ".use_dynamic_scaling", true);
    data.Set("DefaultDQNAgent." + scaler + ".dynamic_mode", rl::RewardScaler::kReturnStdState);
    data.Set("DefaultDQNAgent." + scaler + ".use_auto_post_scale", true);
    const std::array<std::string, 8> keys{
        "actor.[z].intrinsic.action_beta_mode", "learner.intrinsic.terminal_mode",
        "intrinsic.value.head.init.mode", other_scaler + ".dynamic_mode",
        "learner.intrinsic.loss_weight", "learner.intrinsic.gamma",
        "actor.[z].intrinsic.action_beta", "intrinsic.value.head.init.manual_gain"};
    const auto& key = keys[error];
    data.Set("DefaultDQNAgent." + key, error < 4 ? kRfd0121Invalid : (error == 7 ? kRfd0121Nan : "-1"));
    CAPTURE(scaler, key);
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data),
        Catch::Matchers::ContainsSubstring(key) && Catch::Matchers::ContainsSubstring("expected"));
}

TEST_CASE("Decomposition still rejects each active scaler combination", "[rfd0121][dqn][config][rfd0121-review-order]")
{
    const bool intrinsic_scaler = GENERATE(false, true);
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.intrinsic.enabled", true);
    const std::string scaler = intrinsic_scaler ? "intrinsic.reward_scaler" : "reward_scaler";
    data.Set("DefaultDQNAgent." + scaler + ".use_dynamic_scaling", true);
    data.Set("DefaultDQNAgent." + scaler + ".dynamic_mode", rl::RewardScaler::kReturnStdState);
    data.Set("DefaultDQNAgent." + scaler + ".use_auto_post_scale", true);
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data),
        Catch::Matchers::ContainsSubstring("DefaultDQNAgent." + scaler)
        && Catch::Matchers::ContainsSubstring("expected DefaultDQNAgent." + scaler + ".use_auto_post_scale=false"));
}

TEST_CASE("Decomposition names the mode requiring a fixed actor beta", "[dqn][rfd0194]")
{
    anet::ConfigData data;
    data.Set("DefaultDQNAgent.intrinsic.enabled", true);
    data.Set("DefaultDQNAgent.intrinsic.value_mode", dqn::DefaultDQNAgentConfig::kValueModeDecomposed);
    anet::ConfigData source;
    source.Set("Trial.actor.[train].intrinsic.action_beta_mode", dqn::DQNActorConfig::kActionBetaFixed);
    data.CopyValueFrom("DefaultDQNAgent.actor.[train].intrinsic.action_beta_mode", source,
        "Trial.actor.[train].intrinsic.action_beta_mode");
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(data)),
        Catch::Matchers::ContainsSubstring("DefaultDQNAgent.actor.[train].intrinsic.action_beta")
        && Catch::Matchers::ContainsSubstring("Trial.actor.[train].intrinsic.action_beta_mode=fixed"));
}

TEST_CASE("Unused decomposition settings warn only for explicit nondefaults", "[rfd0121][dqn][config]")
{
    const int scenario = GENERATE(0, 1, 2, 3, 4, 5, 6);
    anet::ConfigData base;
    base.Set("DefaultDQNAgent.actor.[train].network", kRfd0121OnlineNetworkMode);
    auto data = WithDefaultForwardConfig(base);
    std::string key;
    if (scenario == 0 || scenario == 2) {
        key = "DefaultDQNAgent.intrinsic.value.head.input_mode";
        data.Set(key, scenario == 0 ? anet::rl::dqn::DefaultDQNAgentConfig::kHeadInputShared : anet::rl::dqn::DefaultDQNAgentConfig::kHeadInputDedicated);
    } else if (scenario == 1) {
        key = "DefaultDQNAgent.learner.intrinsic.loss_weight";
        data.Set(key, 2);
    } else if (scenario == 3 || scenario == 4) {
        data.Set("DefaultDQNAgent.intrinsic.enabled", true);
        data.Set("DefaultDQNAgent.intrinsic.value_mode", anet::rl::dqn::DefaultDQNAgentConfig::kValueModeDecomposed);
        key = scenario == 3 ? "DefaultDQNAgent.learner.intrinsic.gamma" : "DefaultDQNAgent.actor.[train].intrinsic.action_beta";
        data.Set(key, 0.5f);
    } else if (scenario == 5) {
        key = "DefaultDQNAgent.intrinsic.reward_scaler.constant_scale";
        data.Set(key, 2);
    } else {
        key = "DefaultDQNAgent.reward_scaler.dynamic_mode";
        data.Set("DefaultDQNAgent.reward_scaler.use_dynamic_scaling", false);
        data.Set(key, anet::rl::RewardScaler::kReturnStdState);
    }
    anet::test::LogCaptureGuard logs;
    const dqn::DefaultDQNAgentConfig config(data);
    logs.Flush();
    const auto count = std::count_if(logs.Records().begin(), logs.Records().end(), [&](const auto& record) {
        return record.level == wxLOG_Warning && record.message.find(key) != std::string::npos;
    });
    CAPTURE(scenario, key);
    CHECK(count == (scenario == 0 ? 0 : 1));
}

TEST_CASE("Inactive value decomposition warns once for explicit nondefault Head initialization settings", "[dqn][config][rfd0193]")
{
    const bool use_defaults = GENERATE(false, true);
    auto data = WithDefaultForwardConfig();
    const std::string prefix = "DefaultDQNAgent.intrinsic.value.head.init.";
    data.Set(prefix + "nonlinearity", use_defaults ? "linear" : "relu");
    data.Set(prefix + "variance_scale", use_defaults ? 1.0 : 2.0);
    anet::test::LogCaptureGuard logs;
    const dqn::DefaultDQNAgentConfig config(data);
    logs.Flush();
    for (const auto* field : {"nonlinearity", "variance_scale"}) {
        const auto count = std::count_if(logs.Records().begin(), logs.Records().end(), [&](const auto& record) {
            return record.level == wxLOG_Warning
                && record.message.find("Ignored " + prefix + field + "=") != std::string::npos;
        });
        CAPTURE(use_defaults, field);
        CHECK(count == (use_defaults ? 0 : 1));
    }
}

TEST_CASE("Combined Actor leaves inactive fixed beta unset", "[rfd0121][dqn][config]")
{
    ScopedNoopMetricsLogger logger;
    auto config = MakeDeviceForwardDefaultDqnConfig();
    config.actor[kRfd0121Eval].intrinsic.action_beta_mode = anet::rl::dqn::DQNActorConfig::kActionBetaFixed;
    const auto env = MakeLearnerEnvSpec();
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(config, MakeAgentForwardNetworkConfig(), rl::BatchEnvSpec{1, 1}, env, torch::kCPU, 121);
    CHECK_NOTHROW(agent->CreateActor({.batch_env_spec = {1, 1}, .env_spec = env,
        .device = torch::kCPU, .seed = 121, .actor_key = kRfd0121Eval}));
}

TEST_CASE("Decomposed config accepts an explicit gamma schedule without a constant value", "[rfd0121][dqn][config]")
{
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.intrinsic.enabled", true);
    data.Set("DefaultDQNAgent.intrinsic.value_mode", anet::rl::dqn::DefaultDQNAgentConfig::kValueModeDecomposed);
    data.Set("DefaultDQNAgent.learner.intrinsic.discount_mode", anet::rl::dqn::LearnerConfig::kDiscountSeparate);
    data.Set("DefaultDQNAgent.learner.intrinsic.gamma.type", kRfd0121Linear);
    data.Set("DefaultDQNAgent.learner.intrinsic.gamma.start", 0.9f);
    data.Set("DefaultDQNAgent.learner.intrinsic.gamma.end", 0.99f);
    data.Set("DefaultDQNAgent.learner.intrinsic.gamma.steps", 100);
    const dqn::DefaultDQNAgentConfig config(data);
    const anet::ProfiledValue<float> gamma(config.learner.intrinsic.gamma);
    CHECK(gamma.Evaluate(0) == Catch::Approx(0.9f));
    CHECK(gamma.Evaluate(100) == Catch::Approx(0.99f));
}

TEST_CASE("DefaultDQNAgentConfig defines IQN and QR sampling defaults", "[dqn][iqn][config]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.actor.[train].network", "online");
    config_data.Set("DefaultDQNAgent.actor.[eval].policy.tau_rule.sample_mode", "fixed");
    const dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(config_data));

    CHECK(config.quantile_mode == "qr");
    CHECK(config.qr.num_quantiles == 51);
    CHECK(config.actor.at("train").policy.tau_rule.num_taus == 32);
    CHECK(config.actor.at("train").policy.tau_rule.sample_mode == "random");
    CHECK(config.actor.at("eval").policy.tau_rule.num_taus == 32);
    CHECK(config.actor.at("eval").policy.tau_rule.sample_mode == "fixed");
    CHECK(config.target_policy.tau_rule.num_taus == 32);
    CHECK(config.target_policy.tau_rule.sample_mode == "fixed");
    CHECK_FALSE(config.actor.at("train").policy.full_distribution_query.enabled);
    CHECK(config.actor.at("train").policy.full_distribution_query.tau_rule.num_taus == 32);
    CHECK(config.actor.at("train").policy.full_distribution_query.tau_rule.sample_mode == "fixed");
    CHECK_FALSE(config.actor.at("eval").policy.full_distribution_query.enabled);
    CHECK(config.actor.at("eval").policy.full_distribution_query.tau_rule.num_taus == 32);
    CHECK(config.actor.at("eval").policy.full_distribution_query.tau_rule.sample_mode == "fixed");
    CHECK_FALSE(config.target_policy.full_distribution_query.enabled);
    CHECK(config.target_policy.full_distribution_query.tau_rule.num_taus == 32);
    CHECK(config.target_policy.full_distribution_query.tau_rule.sample_mode == "fixed");
    CHECK(config.learner.iqn.current_taus.num_taus == 64);
    CHECK(config.learner.iqn.current_taus.sample_mode == "random");
    CHECK(config.learner.iqn.target_taus.num_taus == 64);
    CHECK(config.learner.iqn.target_taus.sample_mode == "random");
}

TEST_CASE("DefaultDQNAgent config fixture resolves the IQN profile chain", "[dqn][iqn][config]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "default-dqn-iqn-profile-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    // live実験値から分離したprofile連鎖を作り、後段IQN設定の上書きを固定する。
    const auto config_path = root / "config.txt";
    {
        std::ofstream stream(config_path);
        REQUIRE(stream.is_open());
        stream << "DefaultDQNAgent.$ = DefaultDQNAgent.baseline > A > IQN\n";
        stream << "DefaultDQNAgent.baseline.quantile_mode = qr\n";
        stream << "A.learner.iqn.current_taus.num_taus = 64\n";
        stream << "A.learner.iqn.target_taus.num_taus = 64\n";
        stream << "IQN.quantile_mode = iqn\n";
        stream << "IQN.actor.[train].policy.policy_type = UQE\n";
        stream << "IQN.actor.[train].policy.tau_rule.num_taus = 32\n";
        stream << "IQN.actor.[train].policy.tau_rule.sample_mode = random\n";
        stream << "IQN.actor.[eval].policy.policy_type = UQE\n";
        stream << "IQN.actor.[eval].policy.uqe.use_tail_mean = true\n";
        stream << "IQN.actor.[eval].policy.tau_rule.num_taus = 32\n";
        stream << "IQN.actor.[eval].policy.tau_rule.sample_mode = fixed\n";
        stream << "IQN.actor.[eval].policy.full_distribution_query.enabled = true\n";
        stream << "IQN.actor.[eval].policy.full_distribution_query.tau_rule.num_taus = 32\n";
        stream << "IQN.actor.[eval].policy.full_distribution_query.tau_rule.sample_mode = fixed\n";
        stream << "IQN.learner.iqn.current_taus.num_taus = 32\n";
        stream << "IQN.learner.iqn.target_taus.num_taus = 32\n";
        stream << "net.$ = net.base > net.iqn\n";
        stream << "net.base.branch.[value_stream].bind = main_feature\n";
        stream << "net.base.branch.[adv_stream].bind = main_feature\n";
        stream << "net.iqn.block.[IQNTauProjFixture].linear.out_features = 2048\n";
        stream << "net.iqn.branch.[tau_embedding].bind = taus\n";
        stream << "net.iqn.branch.[iqn_fusion].bind = main_feature * tau_embedding\n";
        stream << "net.iqn.branch.[value_stream].bind = iqn_fusion\n";
        stream << "net.iqn.branch.[adv_stream].bind = iqn_fusion\n";
    }

    anet::ConfigManager config_manager(config_path.string());
    const auto config_data = config_manager.GetConfigData();
    const dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(config_data));

    // Agent設定とNN設定の両方で、profile連鎖の最終値が読み出せることを確認する。
    CHECK(config.quantile_mode == "iqn");
    CHECK(config.actor.at("train").policy.policy_type == "UQE");
    CHECK(config.actor.at("train").policy.tau_rule.num_taus == 32);
    CHECK(config.actor.at("train").policy.tau_rule.sample_mode == "random");
    CHECK_FALSE(config.actor.at("train").policy.full_distribution_query.enabled);
    CHECK(config.actor.at("eval").policy.policy_type == "UQE");
    CHECK(config.actor.at("eval").policy.uqe.use_tail_mean);
    CHECK(config.actor.at("eval").policy.tau_rule.num_taus == 32);
    CHECK(config.actor.at("eval").policy.tau_rule.sample_mode == "fixed");
    CHECK(config.actor.at("eval").policy.full_distribution_query.enabled);
    CHECK(config.actor.at("eval").policy.full_distribution_query.tau_rule.num_taus == 32);
    CHECK(config.actor.at("eval").policy.full_distribution_query.tau_rule.sample_mode == "fixed");
    CHECK_FALSE(config.target_policy.full_distribution_query.enabled);
    CHECK(config_data.Get("A.learner.iqn.current_taus.num_taus") == "64");
    CHECK(config.learner.iqn.current_taus.num_taus == 32);
    CHECK(config.learner.iqn.target_taus.num_taus == 32);
    CHECK(config_data.Get("net.block.[IQNTauProjFixture].linear.out_features") == "2048");
    CHECK(config_data.Get("net.branch.[tau_embedding].bind") == "taus");
    CHECK(config_data.Get("net.branch.[iqn_fusion].bind") == "main_feature * tau_embedding");
    CHECK(config_data.Get("net.branch.[value_stream].bind") == "iqn_fusion");
    CHECK(config_data.Get("net.branch.[adv_stream].bind") == "iqn_fusion");

    std::filesystem::remove_all(root);
}

TEST_CASE("DefaultDQNAgentConfig validates quantile modes and tau rules", "[dqn][iqn][config]")
{
    SECTION("mode and QR width")
    {
        anet::ConfigData invalid_mode;
        invalid_mode.Set("DefaultDQNAgent.quantile_mode", "invalid");
        CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(invalid_mode)));

        anet::ConfigData invalid_qr;
        invalid_qr.Set("DefaultDQNAgent.quantile_mode", "qr");
        invalid_qr.Set("DefaultDQNAgent.qr.num_quantiles", 1);
        CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(invalid_qr)));
    }

    SECTION("all tau rules")
    {
        const std::vector<std::string> prefixes{
            "DefaultDQNAgent.actor.[train].policy.tau_rule",
            "DefaultDQNAgent.actor.[eval].policy.tau_rule",
            "DefaultDQNAgent.target_policy.tau_rule",
            "DefaultDQNAgent.actor.[train].policy.full_distribution_query.tau_rule",
            "DefaultDQNAgent.actor.[eval].policy.full_distribution_query.tau_rule",
            "DefaultDQNAgent.target_policy.full_distribution_query.tau_rule",
            "DefaultDQNAgent.learner.iqn.current_taus",
            "DefaultDQNAgent.learner.iqn.target_taus",
        };
        for (const auto& prefix : prefixes) {
            INFO("prefix=" << prefix);

            for (const auto& mode : { "stratified", "systematic", "antithetic" }) {
                INFO("mode=" << mode);
                anet::ConfigData valid_mode;
                valid_mode.Set(prefix + ".sample_mode", mode);
                CHECK_NOTHROW(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(valid_mode)));
            }

            anet::ConfigData invalid_count;
            invalid_count.Set(prefix + ".num_taus", 0);
            CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(invalid_count)));

            anet::ConfigData invalid_mode;
            invalid_mode.Set(prefix + ".sample_mode", "sorted_random");
            CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(invalid_mode)));
        }
    }

    SECTION("full distribution query compatibility")
    {
        for (const auto& [mode, policy_type] : {
                 std::pair{ "none", "Greedy" },
                 std::pair{ "qr", "UQE" },
             }) {
            INFO("mode=" << mode);
            anet::ConfigData non_iqn;
            non_iqn.Set("DefaultDQNAgent.quantile_mode", mode);
            non_iqn.Set("DefaultDQNAgent.actor.[eval].policy.policy_type", policy_type);
            non_iqn.Set("DefaultDQNAgent.actor.[eval].policy.full_distribution_query.enabled", true);

            const dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(non_iqn));

            CHECK(config.actor.at("eval").policy.full_distribution_query.enabled);
        }

        anet::ConfigData non_uqe;
        non_uqe.Set("DefaultDQNAgent.quantile_mode", "iqn");
        non_uqe.Set("DefaultDQNAgent.actor.[eval].policy.policy_type", "Greedy");
        non_uqe.Set("DefaultDQNAgent.actor.[eval].policy.full_distribution_query.enabled", true);
        CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(non_uqe)));
    }

    SECTION("IQN kappa")
    {
        for (const std::string_view value : { "0", "-0.5", "nan", "inf" }) {
            INFO("kappa=" << value);
            anet::ConfigData config_data;
            config_data.Set("DefaultDQNAgent.quantile_mode", "iqn");
            constexpr char kappa_key[] = "DefaultDQNAgent.learner.quantile_huber_kappa";
            config_data.Set(kappa_key, value);
            const auto read = [&] { return dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(config_data)); };
            if (value == "nan" || value == "inf") {
                CHECK_THROWS_WITH(read(), Catch::Matchers::ContainsSubstring(kappa_key));
            } else {
                CHECK_THROWS_WITH(read(), Catch::Matchers::ContainsSubstring(kappa_key)
                    && Catch::Matchers::ContainsSubstring("when DefaultDQNAgent.quantile_mode=iqn"));
            }
        }
    }
}

TEST_CASE("DefaultDQNAgentConfig validates IQN risk ranges only when consumed", "[dqn][iqn][config]")
{
    for (const auto& type : { "UQE", "ThompsonSampling" }) {
        INFO("type=" << type);
        anet::ConfigData none_data;
        none_data.Set("DefaultDQNAgent.quantile_mode", "none");
        none_data.Set("DefaultDQNAgent.actor.[train].policy.policy_type", type);
        CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(none_data)));
    }

    anet::ConfigData uqe_data;
    uqe_data.Set("DefaultDQNAgent.quantile_mode", "iqn");
    uqe_data.Set("DefaultDQNAgent.actor.[train].policy.policy_type", "UQE");
    uqe_data.Set("DefaultDQNAgent.actor.[train].policy.uqe.tau", -0.1f);
    uqe_data.Set("DefaultDQNAgent.actor.[train].policy.uqe.tau.start", -0.1f);
    CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(uqe_data)));

    anet::ConfigData non_spatial_thompson;
    non_spatial_thompson.Set("DefaultDQNAgent.quantile_mode", "iqn");
    non_spatial_thompson.Set("DefaultDQNAgent.actor.[train].policy.policy_type", "ThompsonSampling");
    non_spatial_thompson.Set("DefaultDQNAgent.actor.[train].policy.spatial.enabled", false);
    non_spatial_thompson.Set("DefaultDQNAgent.actor.[train].policy.uqe.tau", -0.1f);
    non_spatial_thompson.Set("DefaultDQNAgent.actor.[train].policy.uqe.tau.start", -0.1f);
    non_spatial_thompson.Set("DefaultDQNAgent.actor.[train].policy.uqe.tau.end", 1.1f);
    CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(non_spatial_thompson)));

    auto spatial_thompson = non_spatial_thompson;
    spatial_thompson.Set("DefaultDQNAgent.actor.[train].policy.spatial.enabled", true);
    CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(spatial_thompson)));

    anet::ConfigData qr_uqe;
    qr_uqe.Set("DefaultDQNAgent.quantile_mode", "qr");
    qr_uqe.Set("DefaultDQNAgent.actor.[train].policy.policy_type", "UQE");
    qr_uqe.Set("DefaultDQNAgent.actor.[train].policy.uqe.tau", -0.1f);
    qr_uqe.Set("DefaultDQNAgent.actor.[train].policy.uqe.tau.start", -0.1f);
    qr_uqe.Set("DefaultDQNAgent.actor.[train].policy.uqe.tau.end", 1.1f);
    CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(qr_uqe)));
}

TEST_CASE("DefaultDQNAgentConfig requires a positive active snapshot interval", "[dqn][config][snapshot]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.actor.[train].clone_model", "false");
    config_data.Set("DefaultDQNAgent.actor.[train].sync_interval.type", "constant");
    config_data.Set("DefaultDQNAgent.actor.[train].sync_interval", "0");

    CHECK_THROWS_WITH(
        dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(config_data)),
        Catch::Matchers::ContainsSubstring("key=DefaultDQNAgent.actor.[train].sync_interval")
        && Catch::Matchers::ContainsSubstring("value=0")
        && Catch::Matchers::ContainsSubstring("expected=>=1"));
}

TEST_CASE("DefaultDQNAgentConfig strictly parses every explicit snapshot field", "[dqn][config][snapshot]")
{
    struct InvalidValue {
        std::string key;
        std::string value;
    };
    const std::vector<InvalidValue> invalid_values{
        { "DefaultDQNAgent.actor.[train].clone_model", "maybe" },
        { "DefaultDQNAgent.actor.[train].sync_interval.start", "-1" },
        { "DefaultDQNAgent.actor.[train].sync_interval.steps", "18446744073709551616" },
        { "DefaultDQNAgent.actor.[train].sync_interval.cycle_mult", "nan" },
        { "DefaultDQNAgent.actor.[train].sync_interval.cycle_mult", "inf" },
    };
    for (const auto& invalid : invalid_values) {
        INFO("key=" << invalid.key << " value=" << invalid.value);
        anet::ConfigData config_data;
        config_data.Set(invalid.key, invalid.value);
        CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(config_data)));
    }

    anet::ConfigData comma_data;
    comma_data.Set("DefaultDQNAgent.actor.[train].sync_interval", "1,000");
    const dqn::DefaultDQNAgentConfig comma_config(WithDefaultForwardConfig(comma_data));
    CHECK(comma_config.actor.at("train").sync_interval->value == 1000);
}

TEST_CASE("DefaultDQNAgentConfig validates phased snapshot profiles", "[dqn][config][snapshot]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.actor.[train].sync_interval.type", "phased");
    config_data.Set("DefaultDQNAgent.actor.[train].sync_interval.phases", "warm main");
    config_data.Set("DefaultDQNAgent.actor.[train].sync_interval.phase.[warm].type", "constant");
    config_data.Set("DefaultDQNAgent.actor.[train].sync_interval.phase.[warm].value", "4");
    config_data.Set("DefaultDQNAgent.actor.[train].sync_interval.phase.[warm].steps", "10");
    config_data.Set("DefaultDQNAgent.actor.[train].sync_interval.phase.[main].type", "linear");
    config_data.Set("DefaultDQNAgent.actor.[train].sync_interval.phase.[main].start", "4");
    config_data.Set("DefaultDQNAgent.actor.[train].sync_interval.phase.[main].end", "2");
    config_data.Set("DefaultDQNAgent.actor.[train].sync_interval.phase.[main].steps", "20");

    const dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(config_data));

    anet::ProfiledValue<rl::step_t> interval(*config.actor.at("train").sync_interval);
    CHECK(interval.Evaluate(0) == 4);
    CHECK(interval.Evaluate(10) == 4);
    CHECK(interval.Evaluate(30) == 2);

    anet::ConfigData undefined_phase_data;
    undefined_phase_data.Set("DefaultDQNAgent.actor.[train].sync_interval.type", "phased");
    undefined_phase_data.Set("DefaultDQNAgent.actor.[train].sync_interval.phases", "missing");
    CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(undefined_phase_data)));
}

TEST_CASE("DefaultDQNAgentConfig clears optimistic target spatial exploration", "[dqn][config][spatial]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.use_optimistic_target", "true");
    config_data.Set("DefaultDQNAgent.actor.[train].policy.policy_type", "UQE");
    config_data.Set("DefaultDQNAgent.actor.[train].policy.spatial.enabled", "true");
    config_data.Set("DefaultDQNAgent.actor.[train].policy.spatial.scale", "linear");

    dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(config_data));

    CHECK(config.actor.at("train").policy.spatial.enabled);
    CHECK_FALSE(config.target_policy.spatial.enabled);
    CHECK(config.target_policy.uqe.epsilon.value == Catch::Approx(0.0f));
    CHECK(config.target_policy.uqe.epsilon.end == Catch::Approx(0.0f));
}

TEST_CASE("DefaultDQNAgentConfig rejects invalid spatial scale type", "[dqn][config][spatial]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.actor.[train].policy.spatial.scale", "invalid");

    CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(config_data)));
}

TEST_CASE("DefaultDQNAgentConfig reads and validates TBO settings", "[dqn][config][tbo]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.learner.use_tbo", "true");
    config_data.Set("DefaultDQNAgent.learner.tbo_epsilon", "0.02");
    config_data.Set("DefaultDQNAgent.reward_scaler.use_dynamic_scaling", "false");
    config_data.Set("DefaultDQNAgent.reward_scaler.use_auto_post_scale", "false");

    dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(config_data));

    CHECK(config.learner.use_tbo);
    CHECK(config.learner.tbo_epsilon == Catch::Approx(0.02f));
}

TEST_CASE("DefaultDQNAgentConfig reads fused optimizer setting", "[dqn][config][optimizer]")
{
    dqn::DefaultDQNAgentConfig default_config(WithDefaultForwardConfig(anet::ConfigData{}));
    CHECK(default_config.learner.use_fused_optimizer);

    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.learner.use_fused_optimizer", "false");
    dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(config_data));
    CHECK_FALSE(config.learner.use_fused_optimizer);
}

TEST_CASE("DefaultDQNAgentConfig reads and validates plasticity probe settings", "[dqn][config][plasticity]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.learner.final_feature_branch_key", "main_feature");
    config_data.Set("DefaultDQNAgent.learner.plasticity.probe.batch_size", 512);
    const dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(config_data));
    CHECK(config.learner.final_feature_branch_key == "main_feature");
    CHECK(config.learner.plasticity.probe.batch_size == 512);

    config_data.Set("DefaultDQNAgent.learner.plasticity.probe.batch_size", 0);
    CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(config_data)));
}

TEST_CASE("DQN configs read sample prefetch setting", "[dqn][config][prefetch]")
{
    dqn::DefaultDQNAgentConfig default_config(WithDefaultForwardConfig(anet::ConfigData{}));
    CHECK_FALSE(default_config.learner.use_rb_prefetch);

    anet::ConfigData default_dqn_data;
    default_dqn_data.Set("DefaultDQNAgent.learner.use_rb_prefetch", "true");
    dqn::DefaultDQNAgentConfig default_dqn_config(WithDefaultForwardConfig(default_dqn_data));
    CHECK(default_dqn_config.learner.use_rb_prefetch);

    anet::ConfigData rainbow_data;
    rainbow_data.Set("RainbowAgent.learner.use_rb_prefetch", "true");
    dqn::RainbowAgentConfig rainbow_config(rainbow_data);
    CHECK_FALSE(rainbow_config.learner.use_rb_prefetch);
}

TEST_CASE("DefaultDQNAgentConfig rejects invalid TBO epsilon", "[dqn][config][tbo]")
{
    for (const auto& value : { "0", "-0.01", "nan", "inf" }) {
        INFO(value);
        anet::ConfigData config_data;
        config_data.Set("DefaultDQNAgent.learner.tbo_epsilon", value);
        CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(config_data)));
    }
}

TEST_CASE("RainbowAgentConfig keeps TBO disabled", "[dqn][config][tbo]")
{
    anet::ConfigData config_data;
    config_data.Set("RainbowAgent.learner.use_tbo", "true");

    dqn::RainbowAgentConfig config(config_data);

    CHECK_FALSE(config.learner.use_tbo);
}

TEST_CASE("RainbowAgentConfig keeps fused optimizer disabled", "[dqn][config][optimizer]")
{
    anet::ConfigData config_data;
    config_data.Set("RainbowAgent.learner.use_fused_optimizer", "true");

    dqn::RainbowAgentConfig config(config_data);

    CHECK_FALSE(config.learner.use_fused_optimizer);
}

TEST_CASE("DefaultDQNAgentConfig warns when TBO shares reward compression", "[dqn][config][tbo]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.learner.use_tbo", "true");
    config_data.Set("DefaultDQNAgent.reward_scaler.use_dynamic_scaling", "true");
    config_data.Set("DefaultDQNAgent.reward_scaler.use_auto_post_scale", "false");

    anet::test::LogCaptureGuard logs;
    dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(config_data));
    CHECK_NOTHROW(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(config_data)));
    logs.Flush();

    CHECK(config.learner.use_tbo);
    CHECK(config.reward_scaler.use_dynamic_scaling);
    CHECK_FALSE(config.reward_scaler.use_auto_post_scale);
    bool found_warning = false;
    for (const auto& record : logs.Records()) {
        if (record.message.find("learner.use_tbo") != std::string::npos
            && record.message.find("reward_scaler.use_dynamic_scaling") != std::string::npos
            && record.message.find("double-compressed") != std::string::npos) {
            CHECK(record.level == wxLOG_Warning);
            found_warning = true;
        }
    }
    CHECK(found_warning);
    CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 1);
}

TEST_CASE("DQN validates spectral soft update tau only for a soft update configuration", "[dqn][spectral_norm][config]")
{
    auto make_model = [](const dqn::NetworkModelConfig& model_config) {
        const anet::TensorSpec vector_spec{
            .type = anet::SpaceType::Vector,
            .shape = { 2 },
            .dtype = torch::kFloat32,
        };
        anet::ConfigData config_data;
        config_data.Set("net.block.[Linear].type", "Linear");
        config_data.Set("net.block.[Linear].linear.out_features", 2);
        config_data.Set("net.block.[Linear].weight_norm.mode", "spectral");
        config_data.Set("net.branch.[feature].bind", kVectorKey);
        config_data.Set("net.branch.[feature].structure", "Linear");
        config_data.Set("net.body.output.[feature]", "feature");
        return std::make_shared<dqn::NetworkModel>(
            model_config,
            torch::Device(torch::kCPU),
            anet::nn::NetworkConfig(config_data),
            anet::TensorSpecMap{ { kVectorKey, vector_spec } },
            1,
            nullptr,
            false,
            65065);
    };

    anet::test::RegisterStandardNetworkModules();
    CHECK_THROWS_WITH(
        make_model(dqn::NetworkModelConfig{
            .soft_update_tau = {.value = 0.5f},
            .hard_update_interval = -1,
        }),
        Catch::Matchers::ContainsSubstring("key=value (not read from config) value=0.5")
            && Catch::Matchers::ContainsSubstring("max_value=0.1"));
    CHECK_NOTHROW(make_model(dqn::NetworkModelConfig{
        .soft_update_tau = {.value = 0.1f},
        .hard_update_interval = -1,
    }));
    CHECK_NOTHROW(make_model(dqn::NetworkModelConfig{
        .soft_update_tau = {.value = 0.5f},
        .hard_update_interval = 100,
    }));
    CHECK_NOTHROW(make_model(dqn::NetworkModelConfig{ .soft_update_tau = {.value = 1.0f} }));
    CHECK_THROWS_WITH(make_model(dqn::NetworkModelConfig{
        .soft_update_tau = {.value = -0.1f}, .hard_update_interval = 100 }),
        Catch::Matchers::ContainsSubstring("key=value (not read from config) value=-0.1"));
    CHECK_THROWS_WITH(make_model(dqn::NetworkModelConfig{
        .soft_update_tau = {.type = "linear", .start = 0.05f, .end = 0.11f, .steps = 8} }),
        Catch::Matchers::ContainsSubstring("key=end (not read from config) value=0.11"));
    anet::ProfiledValueConfig<float> phased{.type = "phased", .phases = {"one"}};
    phased.phase.Set("one", anet::ProfiledValuePhaseConfig<float>{.value = 1.0f, .steps = 8});
    CHECK_THROWS_WITH(make_model(dqn::NetworkModelConfig{ .soft_update_tau = phased }),
        Catch::Matchers::ContainsSubstring("key=value (not read from config) value=1"));

}

TEST_CASE("Replay fit validates its configuration and resolved target subscription", "[dqn][replay_fit][config]")
{
    ScopedNoopMetricsLogger logger;
    anet::test::RegisterStandardNetworkModules();
    auto data = MakeIqnTracerConfigData();
    const dqn::DefaultDQNAgentConfig defaults(WithDefaultForwardConfig(data));
    CHECK(defaults.learner.replay_fit.probe.batch_size == 1024);
    CHECK(defaults.learner.replay_fit.iqn.num_taus == 32);
    for (const auto* field : { "probe.batch_size", "iqn.num_taus" }) {
        for (const auto* value : { "0", "-1", "0.5", "nan" }) {
            auto invalid = data;
            invalid.Set(std::string("DefaultDQNAgent.learner.replay_fit.") + field, value);
            CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(invalid)));
        }
    }
    for (const bool enabled : { false, true }) {
        for (const bool overridden : { false, true }) {
            auto config_data = data;
            config_data.Set("DefaultDQNAgent.learner.enabled", enabled);
            config_data.Set("DefaultDQNAgent.use_optimistic_target", true);
            config_data.Set("DefaultDQNAgent.actor.[train].policy.policy_type", "ThompsonSampling");
            if (overridden) config_data.Set("DefaultDQNAgent.target_policy.policy_type", "Greedy");
            auto agent = std::make_shared<dqn::DefaultDQNAgent>(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(config_data)),
                anet::nn::NetworkConfig(config_data, "DefaultDQNAgent.net"), rl::BatchEnvSpec{ 2, 1 },
                MakeIqnTracerEnvSpec(), torch::kCPU, 73032);
            const std::vector<rl::ScalarMetricSubscription> subscriptions{
                rl::ScalarMetricSubscription{ .source_key = "replay_fit_unsampled_count",
                    .event = rl::EventType::LEARN, .target = rl::EventField::UPDATE_RESULT,
                    .interval = 503, .scope = rl::RunnerScope::TRAIN } };
            if (enabled && !overridden) CHECK_THROWS_WITH(agent->ConfigureScalarMetricSubscriptions(subscriptions),
                Catch::Matchers::ContainsSubstring("ThompsonSampling"));
            else CHECK_NOTHROW(agent->ConfigureScalarMetricSubscriptions(subscriptions));
        }
    }
}

TEST_CASE("DQN plasticity feature key is validated only when subscribed", "[dqn][plasticity][config]")
{
    auto probe_state = std::make_shared<AutocastProbeState>();
    AutocastProbeNetworkModel model(probe_state, torch::Device(torch::kCPU));
    dqn::RuntimeVars vars;
    auto env_spec = MakeLearnerEnvSpec();
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.final_feature_branch_key.clear();
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, dqn::LearnerForwardResources{});
    CHECK_NOTHROW(learner.ConfigureScalarMetricSubscriptions({}));
    CHECK_THROWS(learner.ConfigureScalarMetricSubscriptions({ rl::ScalarMetricSubscription{
        .source_key = "plasticity_srank",
        .event = rl::EventType::LEARN,
    } }));
    CHECK_THROWS(learner.ConfigureScalarMetricSubscriptions({ rl::ScalarMetricSubscription{
        .source_key = "plasticity_weight_norm_feature",
        .event = rl::EventType::LEARN,
    } }));

    dqn::LearnerConfig unknown_config = WithDefaultLearnerForwardConfig();
    unknown_config.final_feature_branch_key = "missing";
    TestLearner unknown_learner(unknown_config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, dqn::LearnerForwardResources{});
    CHECK_THROWS_WITH(unknown_learner.ConfigureScalarMetricSubscriptions({ rl::ScalarMetricSubscription{
        .source_key = "plasticity_srank",
        .event = rl::EventType::LEARN,
    } }), Catch::Matchers::ContainsSubstring("available_branches"));
    CHECK_THROWS_WITH(unknown_learner.ConfigureScalarMetricSubscriptions({ rl::ScalarMetricSubscription{
        .source_key = "plasticity_weight_norm_readout",
        .event = rl::EventType::LEARN,
    } }), Catch::Matchers::ContainsSubstring("available_branches"));
}

TEST_CASE("Rainbow rejects quantile counts that contradict QR mode", "[rfd0117][dqn][rainbow][qr_config]")
{
    const int count = GENERATE(-1, 0, 1, 2);
    const bool use_qr = GENERATE(false, true);
    CAPTURE(count, use_qr);
    anet::ConfigData data;
    data.Set("RainbowAgent.use_qr", use_qr);
    data.Set("RainbowAgent.num_quantiles", count);
    if (use_qr && count <= 1) {
        CHECK_THROWS_WITH(dqn::RainbowAgentConfig(data),
            Catch::Matchers::ContainsSubstring("RainbowAgent.num_quantiles")
            && Catch::Matchers::ContainsSubstring("expected"));
    } else {
        CHECK_NOTHROW(dqn::RainbowAgentConfig(data));
    }
}

TEST_CASE("DQN feature diagnostics use the settings read by each agent config", "[rfd0194][rfd0194_dqn]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const auto env_spec = MakeLearnerEnvSpec();
    const rl::BatchEnvSpec batch_spec{.num_envs = 1, .num_threads = 1};
    const std::vector<rl::ScalarMetricSubscription> subscriptions{
        rl::ScalarMetricSubscription{.source_key = "plasticity_srank", .event = rl::EventType::LEARN}
    };

    auto default_config = MakeDeviceForwardDefaultDqnConfig();
    default_config.learner.final_feature_branch_key.clear();
    const auto default_agent = std::make_shared<dqn::DefaultDQNAgent>(default_config, MakeAgentForwardNetworkConfig(),
        batch_spec, env_spec, torch::kCPU, 123);
    CHECK_THROWS_WITH(default_agent->ConfigureScalarMetricSubscriptions(subscriptions),
        Catch::Matchers::ContainsSubstring("DefaultDQNAgent.learner.final_feature_branch_key")
        && !Catch::Matchers::ContainsSubstring("remove the metric subscription"));

    const auto rainbow_config = MakeDeviceForwardRainbowConfig();
    const auto rainbow_agent = std::make_shared<dqn::RainbowAgent>(rainbow_config, MakeAgentForwardNetworkConfig(),
        batch_spec, env_spec, torch::kCPU, 123);
    CHECK_NOTHROW(rainbow_agent->ConfigureScalarMetricSubscriptions(subscriptions));

    auto probe = std::make_shared<AutocastProbeState>();
    AutocastProbeNetworkModel model(probe, torch::Device(torch::kCPU));
    dqn::RuntimeVars vars;
    TestLearner rainbow_learner(rainbow_config.learner, model, vars, batch_spec, env_spec, dqn::LearnerForwardResources{});
    CHECK_THROWS_WITH(rainbow_learner.ConfigureScalarMetricSubscriptions(subscriptions),
        Catch::Matchers::ContainsSubstring("final_feature_branch_key (not read from config)")
        && Catch::Matchers::ContainsSubstring("remove the metric subscription")
        && !Catch::Matchers::ContainsSubstring("DefaultDQNAgent"));
}

TEST_CASE("Default DQN reader reports the full key that supplied an invalid setting", "[rfd0194]")
{
    auto data = WithDefaultForwardConfig();
    anet::ConfigData input;
    const std::string source_key = "Trial.learner.munchausen.clip_value_min";
    input.Set(source_key, 1.0f);
    data.CopyValueFrom("DefaultDQNAgent.learner.munchausen.clip_value_min", input, source_key);
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data),
        Catch::Matchers::ContainsSubstring(source_key + "=1")
        && Catch::Matchers::ContainsSubstring("expected finite <= 0"));
}

TEST_CASE("DQN configuration leaves root-level relative learner settings unread", "[rfd0194][rfd0194_dqn_config]")
{
    auto data = WithDefaultForwardConfig();
    data.Set("learner.per_alpha", 2.f);
    const auto default_config = dqn::DefaultDQNAgentConfig(data);
    const auto rainbow_config = dqn::RainbowAgentConfig(data);

    CHECK(default_config.learner.per_alpha == dqn::LearnerConfig{}.per_alpha);
    CHECK(rainbow_config.learner.per_alpha == dqn::LearnerConfig{}.per_alpha);
    CHECK(std::ranges::any_of(data.GetUnreadKeys(), [](const auto& item) {
        return item.first == "learner.per_alpha";
    }));
}

TEST_CASE("DQN Learners name the source of an invalid priority floor", "[rfd0194][rfd0194_per]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const auto env = MakeLearnerEnvSpec();
    const rl::BatchEnvSpec batch{.num_envs = 1, .num_threads = 1};
    anet::ConfigData data;
    const bool rainbow = GENERATE(false, true);
    const std::string key = std::string(rainbow ? "RainbowAgent" : "DefaultDQNAgent") + ".learner.per_eps";
    data.Set(key, -1.f);

    if (rainbow) {
        dqn::RainbowAgentConfig config(data);
        config.use_qr = false;
        config.use_dueling_net = false;
        CHECK_THROWS_WITH((dqn::RainbowAgent(config, MakeAgentForwardNetworkConfig(), batch, env, torch::kCPU, 123)),
            Catch::Matchers::ContainsSubstring(key + "=-1"));
    } else {
        dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(data));
        config.quantile_mode = "none";
        config.head.use_dueling = false;
        CHECK_THROWS_WITH((dqn::DefaultDQNAgent(config, MakeAgentForwardNetworkConfig(), batch, env, torch::kCPU, 123)),
            Catch::Matchers::ContainsSubstring(key + "=-1"));
    }
}

TEST_CASE("Rainbow validates directly modified quantile configuration before building networks", "[rfd0117][dqn][rainbow][qr_config]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeDeviceForwardRainbowConfig();
    config.use_qr = true;
    config.num_quantiles = GENERATE(0, 1);
    CAPTURE(config.num_quantiles);
    CHECK_THROWS_WITH(dqn::RainbowAgent(config, MakeAgentForwardNetworkConfig(),
        rl::BatchEnvSpec{1, 1}, MakeLearnerEnvSpec(), torch::kCPU, 123),
        Catch::Matchers::ContainsSubstring("RainbowAgent.num_quantiles")
        && Catch::Matchers::ContainsSubstring("expected"));
}

TEST_CASE("Rainbow Actor catalog selects its own epsilon", "[dqn][rfd0061][rainbow]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeDeviceForwardRainbowConfig();
    config.actor["train"].policy.epsilon.value = 1.0f;
    config.actor["eval"].policy.epsilon.value = 0.0f;
    config.actor["eval"].policy.epsilon.end = 0.0f;
    auto agent = std::make_shared<dqn::RainbowAgent>(config, MakeAgentForwardNetworkConfig(),
        rl::BatchEnvSpec{1, 1}, MakeLearnerEnvSpec(), torch::kCPU, 123);
    const auto actor = agent->CreateActor(rl::ActorRequest{.batch_env_spec = {1, 1},
        .env_spec = MakeLearnerEnvSpec(), .device = torch::kCPU, .seed = 456, .actor_key = "eval"});
    CHECK(*actor->GetScalar("epsilon") == Catch::Approx(0.0f));
}

TEST_CASE("DefaultDQN resolves opaque named forward settings for every purpose", "[dqn][noisynet][config]")
{
    anet::test::RegisterStandardNetworkModules();
    anet::ConfigData data;
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.mode", "sample");
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.hold", "call");
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.share", "batch");
    data.Set("DefaultDQNAgent.nn_forward.[mean].noisy.mode", "mu_only");
    data.Set("DefaultDQNAgent.nn_forward.[sample].custom.scale", 3);
    data.Set("DefaultDQNAgent.actor.[train].nn_forward_key", "sample");
    data.Set("DefaultDQNAgent.actor.[eval].nn_forward_key", "mean");
    data.Set("DefaultDQNAgent.learner.current.nn_forward_key", "sample");
    data.Set("DefaultDQNAgent.learner.target.nn_forward_key", "sample");
    const anet::rl::dqn::DefaultDQNAgentConfig config(data);
    REQUIRE(config.nn_forward.size() == 2);
    CHECK(config.actor.at("train").nn_forward_key == "sample");
    CHECK(config.actor.at("eval").nn_forward_key == "mean");
    CHECK(config.learner.current.nn_forward_key == "sample");
    CHECK(config.learner.target.nn_forward_key == "sample");
    CHECK(config.nn_forward.at("sample").Get<int>("custom.scale") == 3);
    // 元データの編集で解決済み設定が変わらないことも、利用側の型を持たずに確認する。
    data.Set("DefaultDQNAgent.nn_forward.[sample].custom.scale", 4);
    CHECK(config.nn_forward.at("sample").Get<int>("custom.scale") == 3);
}

TEST_CASE("DefaultDQN Actor owns Noisy draws and advances its action clock", "[dqn][noisynet][actor]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto data = WithDefaultForwardConfig(MakeNativeStackConfigData());

    data.Set("net.branch.[main_feature].bind", kVectorKey);
    data.Set("net.branch.[main_feature].structure", "Linear");
    const bool projection_only = GENERATE(false, true);
    CAPTURE(projection_only);
    data.Set("net.block.[Linear].noisy.enabled", !projection_only);
    data.Set("DefaultDQNAgent.head.projection.noisy.enabled", projection_only);
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.mode", "sample");
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.hold", "call");
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.share", "batch");
    data.Set("DefaultDQNAgent.actor.[train].nn_forward_key", "sample");
    data.Set("DefaultDQNAgent.actor.[train].policy.epsilon", 0.0f);
    data.Set("DefaultDQNAgent.actor.[train].policy.epsilon.start", 0.0f);
    data.Set("DefaultDQNAgent.actor.[train].policy.epsilon.end", 0.0f);
    data.Set("DefaultDQNAgent.actor.[eval].nn_forward_key", "test_default");
    const rl::BatchEnvSpec batch_spec{ 2, 1 };
    const auto env_spec = MakeIqnTracerEnvSpec();
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(dqn::DefaultDQNAgentConfig(data),
        anet::nn::NetworkConfig(data), batch_spec, env_spec, torch::kCPU, 101);
    auto create = [&](const std::string& key, anet::seed_t seed) {
        return agent->CreateActor(rl::ActorRequest{.batch_env_spec = batch_spec, .env_spec = env_spec,
            .device = torch::kCPU, .seed = seed, .actor_key = key});
    };
    auto first = create("train", 11);
    auto replay = create("train", 11);
    auto other = create("train", 12);
    auto eval = create("eval", 13);
    const auto flags = torch::zeros({ 2 }, torch::kBool);
    const rl::BatchState state(anet::TensorDict{{kVectorKey, torch::ones({2, 4})}}, flags, flags, flags);
    auto q = [&](const std::shared_ptr<rl::Actor>& actor) {
        return actor->MakeAction(rl::StepCounts{}, state)->GetAuxData().at("q_values").clone();
    };
    const auto a = q(first);
    CHECK(torch::equal(a, q(replay)));
    CHECK_FALSE(torch::equal(a, q(other)));
    const auto mean = q(eval);
    const auto b = q(first);
    CHECK_FALSE(torch::equal(a, b));
    // 他Actorと評価を挟んでも、同じseedの箱には同じ抽選列が残る。
    q(other);
    CHECK(torch::equal(b, q(replay)));
    CHECK(torch::equal(mean, q(eval)));
}

TEST_CASE("Default DQN numeric validation reports the key value and expected range", "[dqn][rfd0128][config-format]")
{
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.learner.noisy_probe.num_draws", 0);
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data), Catch::Matchers::ContainsSubstring(
        "Invalid DefaultDQNAgent.learner.noisy_probe.num_draws=0; expected integer >= 1."));
}

TEST_CASE("Default DQN choice validation lists three alternatives", "[dqn][rfd0128][config-format]")
{
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.quantile_mode", "unknown");
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data), Catch::Matchers::ContainsSubstring(
        "Invalid DefaultDQNAgent.quantile_mode='unknown'; expected none, qr, or iqn."));
}

TEST_CASE("Noisy probe sizes and sigma decay validate written values even when dormant", "[dqn][noisynet][config][validation]")
{
    for (const auto key : {"learner.noisy_probe.probe.batch_size", "learner.noisy_probe.num_draws", "learner.noisy_probe.iqn.num_taus"}) {
        for (const auto value : {"0", "-1", "1.5", "oops"}) {
            auto data = WithDefaultForwardConfig();
            data.Set("DefaultDQNAgent.learner.enabled", false);
            data.Set(std::string("DefaultDQNAgent.") + key, value);
            CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data), Catch::Matchers::ContainsSubstring(key));
        }
    }
    for (const auto value : {"-0.1", "nan", "inf", "oops"}) {
        auto data = WithDefaultForwardConfig();
        data.Set("DefaultDQNAgent.learner.enabled", false);
        data.Set("DefaultDQNAgent.learner.noisy_weight_decay", value);
        CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data), Catch::Matchers::ContainsSubstring("learner.noisy_weight_decay"));
    }
    const auto defaults = dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig()).learner;
    CHECK(defaults.noisy_weight_decay == 0.0f);
    CHECK(defaults.noisy_probe.probe.batch_size == 128);
    CHECK(defaults.noisy_probe.num_draws == 32);
    CHECK(defaults.noisy_probe.iqn.num_taus == 32);
}

TEST_CASE("RFD0084 profile punctuation preserves all resolved and typed values", "[.][rfd0084-config-equivalence]")
{
    anet::test::RegisterStandardNetworkModules();
    wxString before_directory, output_directory;
    REQUIRE(wxGetEnv("ANET_RFD0084_CONFIG_BEFORE", &before_directory));
    REQUIRE(wxGetEnv("ANET_RFD0084_CONFIG_OUTPUT", &output_directory));
    const auto before = std::filesystem::path(before_directory.ToStdWstring());
    const auto output = std::filesystem::path(output_directory.ToStdWstring());
    const auto after = std::filesystem::current_path() / "apps/runner/config";
    std::filesystem::create_directories(output);
    const std::string efficiency = "run.@v5_iqn_impala_x2>run.@a5>run.@a5_apex>run.@va_base>"
        "run.@hard500>run.@rr4>run.@munch>run.@rfit>run.@a5_metrics>run.@btrnet>run.@btrsn12>run.@cap2m>run.@eval2ch";
    struct Scenario {
        std::string key;
        std::string environment;
        std::string chain;
    };
    const std::vector<Scenario> scenarios{
        {.key="atari_default", .environment="Atari.txt", .chain=""},
        {.key="atari_efficiency", .environment="Atari.txt", .chain=efficiency},
        {.key="nz_learning", .environment="Atari.txt", .chain=efficiency + ">run.@nz_learning"},
        {.key="nz_reference", .environment="Atari.txt", .chain=efficiency + ">run.@nz_reference"},
        {.key="nz_reference_off", .environment="Atari.txt", .chain=efficiency + ">run.@nz_reference_off"},
        {.key="lunarlander_cpu", .environment="LunarLander.txt", .chain=""},
    };
    for (const auto& scenario : scenarios) {
        INFO(scenario.key);
        // 同じ選択を公開ConfigManagerへ渡し、実行資源を生成せず全キー・文字列値を採取する。
        const auto capture = [&](const std::filesystem::path& source, const std::string& suffix) {
            const auto input = output / (scenario.key + suffix + ".txt");
            {
                std::ofstream stream(input, std::ios::binary);
                REQUIRE(stream.good());
                stream << "$include <" << (source / "_main.txt").generic_string() << ">\n"
                    << "$include <" << (source / scenario.environment).generic_string() << ">\n";
            }
            const wxCmdLineEntryDesc description[] = {
                {wxCMD_LINE_PARAM, nullptr, nullptr, "key=value", wxCMD_LINE_VAL_STRING,
                    wxCMD_LINE_PARAM_OPTIONAL | wxCMD_LINE_PARAM_MULTIPLE}, {wxCMD_LINE_NONE},
            };
            wxCmdLineParser command_line(description,
                wxString::FromUTF8(scenario.chain.empty() ? std::string() : "run.$=" + scenario.chain));
            REQUIRE(command_line.Parse(false) == 0);
            const auto config = anet::ConfigManager(input, &command_line).GetConfigData();
            anet::json values = anet::json::object();
            for (const auto& [key, value] : config.Map()) values[key] = value;
            const anet::json result{{"values", values}, {"typed", dqn::DefaultDQNAgentConfig(config).ToJson()}};
            std::ofstream stream(output / (scenario.key + suffix + ".json"), std::ios::binary);
            REQUIRE(stream.good());
            stream << result.dump(2) << '\n';
            return result;
        };
        const auto old_values = capture(before, ".before");
        const auto new_values = capture(after, ".after");
        CHECK(old_values == new_values);
    }
}

TEST_CASE("Gradient noise validates subscribed settings and warns once for mixed intervals", "[dqn][gradnoise][config]")
{
    ScopedNoopMetricsLogger metrics_logger;
    const auto subscription = [](const char* key, int interval = 503) {
        return rl::ScalarMetricSubscription{ .source_key = key,
            .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT,
            .interval = interval,
            .scope = rl::RunnerScope::TRAIN };
    };
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.learner.grad_noise.num_micro_batches", 4);
    data.Set("DefaultDQNAgent.learner.grad_noise.ema_alpha", 0.25);
    const auto parsed = dqn::DefaultDQNAgentConfig(data).learner;
    CHECK(parsed.grad_noise.num_micro_batches == 4);
    CHECK(parsed.grad_noise.ema_alpha == Catch::Approx(0.25));
    const auto expect_invalid = [&](dqn::LearnerConfig config, const char* key, bool local = true) {
        TestNetworkModel model;
        dqn::RuntimeVars vars;
        if (local) {
            CHECK_THROWS_WITH((TestLearner(config, model, vars, rl::BatchEnvSpec{ 1, 1 },
                MakeLearnerEnvSpec(), dqn::LearnerForwardResources{})), Catch::Matchers::ContainsSubstring(key));
            return;
        }
        TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, MakeLearnerEnvSpec(), dqn::LearnerForwardResources{});
        CHECK_NOTHROW(learner.ConfigureScalarMetricSubscriptions({}));
        CHECK_THROWS_WITH(
            learner.ConfigureScalarMetricSubscriptions({ subscription("grad_noise_signal_raw") }), Catch::Matchers::ContainsSubstring(key));
    };
    auto config = WithDefaultLearnerForwardConfig();
    config.grad_noise.num_micro_batches = 1;
    expect_invalid(config, "num_micro_batches (not read from config)=1");
    config.grad_noise.num_micro_batches = 0;
    expect_invalid(config, "num_micro_batches (not read from config)=0");
    config.grad_noise.num_micro_batches = 3;
    expect_invalid(config, "replay_batch_size (not read from config)=128", false);
    config.grad_noise.num_micro_batches = 8;
    for (const auto alpha : { 0.0, -1.0, 1.1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity() }) {
        config.grad_noise.ema_alpha = alpha;
        expect_invalid(config, "grad_noise.ema_alpha (not read from config)");
    }
    config.grad_noise.ema_alpha = 0.05;
    config.use_amp = true;
    config.use_amp_bf16 = false;
    expect_invalid(config, "use_amp_bf16 (not read from config)=false", false);
    config.use_amp = false;
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, MakeLearnerEnvSpec(), dqn::LearnerForwardResources{});
    CHECK_NOTHROW(learner.ConfigureScalarMetricSubscriptions({ subscription("grad_noise_signal_raw") }));
    CHECK_THROWS_WITH(learner.ConfigureScalarMetricSubscriptions({ subscription("grad_noise_signal_adam_feature_raw") }),
        Catch::Matchers::ContainsSubstring("final_feature_branch_key"));
    config.final_feature_branch_key = "missing";
    TestLearner unknown(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, MakeLearnerEnvSpec(), dqn::LearnerForwardResources{});
    CHECK_THROWS_WITH(unknown.ConfigureScalarMetricSubscriptions({ subscription("grad_noise_scale_adam_feature") }),
        Catch::Matchers::ContainsSubstring("available_branches"));
    anet::test::LogCaptureGuard logs;
    learner.ConfigureScalarMetricSubscriptions({ subscription("grad_noise_signal_raw", 503) });
    logs.Flush();
    CHECK(std::ranges::none_of(logs.Records(),
        [](const auto& record) { return record.message.find("Gradient noise subscription intervals differ") != std::string::npos; }));
    for (int i = 0; i < 2; ++i)
        learner.ConfigureScalarMetricSubscriptions(
            { subscription("grad_noise_signal_raw", 503), subscription("grad_noise_variance_raw", 100) });
    logs.Flush();
    CHECK(std::ranges::count_if(logs.Records(), [](const auto& record) {
        return record.message.find("Gradient noise subscription intervals differ: intervals=[100,503]") != std::string::npos;
    }) == 1);
}

TEST_CASE("DQN step diagnostics rebuild forced action hints without PER and complete a bootstrap", "[dqn][rfd0094][step_diagnostics]")
{
    ScopedNoopMetricsLogger logger;
    auto config = MakeDeviceForwardDefaultDqnConfig();
    config.learner.use_per = false;
    config.learner.use_n_step = false;
    config.learner.n_step = 1;
    config.learner.gamma.value = 0.5f;
    config.learner.use_tbo = false;
    config.learner.per_eps = 0.1f;
    config.learner.use_per_prio_clip = false;
    config.learner.munchausen.enabled = false;
    config.reward_scaler.use_dynamic_scaling = false;
    config.reward_scaler.use_auto_post_scale = false;
    config.reward_scaler.constant_scale = 2.0f;
    config.reward_scaler.manual_post_scale = 1.0f;
    config.reward_scaler.use_clipping = true;
    config.reward_scaler.clip_range = 1.5f;
    auto diagnostic_spec = MakeLearnerEnvSpec();
    diagnostic_spec.action_spec.value_labels = { "a0", "a1" };
    std::shared_ptr<rl::Agent> agent = std::make_shared<dqn::DefaultDQNAgent>(config,
        MakeAgentForwardNetworkConfig(), rl::BatchEnvSpec{1, 1}, diagnostic_spec, torch::kCPU, 123);
    auto diagnostics = agent->CreateStepDiagnosticsProvider({.num_lanes = 1, .keys = {}, .actor = nullptr});
    REQUIRE(diagnostics);
    const auto make = [&](bool first, float reward, const torch::Tensor& q) {
        auto action = MakeDiagnosticActionInfo(torch::tensor({1}, torch::kInt64),
            anet::TensorDict{}, rl::AuxData{{"q_values", q}}, config.learner,
            rl::ReplayInitialPriorityHint(torch::full({1, dqn::kActorQHintColumnCount}, 100.0f)));
        return rl::BatchExperience{MakeDeterminismState(0, 1, {false}, {false}, {first}),
            action, torch::tensor({reward}), MakeDeterminismState(1, 1, {false}, {false}, {false})};
    };
    const auto first = diagnostics->Observe(make(true, 1.0f, torch::tensor({{4.0f, 5.0f}})));
    REQUIRE(first.current);
    CHECK(first.current->GetScalar("q_sa", 0) == 5.0f);
    CHECK(first.current->GetScalar("state_value", 0) == 5.0f);
    CHECK(first.pending_keys == std::vector<std::vector<std::string>>{{"approx_priority"}});
    CHECK(std::isnan(*first.current->GetScalar("approx_priority", 0)));
    CHECK_FALSE(first.current->GetScalar("unknown", 0));
    CHECK(first.completed.empty());
    const auto second = diagnostics->Observe(make(false, 0.0f, torch::tensor({{6.0f, 2.0f}})));
    REQUIRE(second.completed.size() == 1);
    CHECK(second.completed[0].lane == 0);
    CHECK(second.completed[0].steps_back == 1);
    // clip(2*r)=1.5、bootstrap=0.5*6、開始Q=5なので|4.5-5|+0.1。
    REQUIRE(second.completed[0].values);
    CHECK(second.completed[0].resolved_keys == std::vector<std::string>{"approx_priority"});
    CHECK(*second.completed[0].values->GetScalar("approx_priority", 0) == Catch::Approx(0.6f));
    CHECK(std::isnan(*agent->GetScalar("reward_scaler.clip_ratio")));
    diagnostics->Reset();
    CHECK(diagnostics->Observe(make(false, 1.0f, torch::tensor({{4.0f, 5.0f}}))).completed.empty());

    // 実Agentの部品を列生成器まで通し、同じ元行の重複列が一度だけ確定する。
    anet::ConfigData columns;
    columns.Set("app.eval_trace_panel.columns.[1]", "$agent_step approx_priority");
    columns.Set("app.eval_trace_panel.columns.[2]", "$agent_step approx_priority display:pct");
    rl::StepTraceBuilder trace(columns, diagnostic_spec, agent, 1, nullptr);
    std::vector<std::shared_ptr<rl::StepTrace>> rows;
    std::vector<int64_t> completed_ids;
    trace.SetHandlers([&](const auto& row, const rl::TrainEvent&) { rows.push_back(row); },
        [&](const auto& row, std::span<const size_t>) { completed_ids.push_back(row->id); });
    const auto add = [&](bool start, float reward, const torch::Tensor& q, rl::step_t step) {
        const auto experience = make(start, reward, q);
        auto result = std::make_shared<rl::PlainBatchStepResult>(experience.reward,
            experience.next_state, experience.next_state, 1, 0, std::vector<rl::AuxData>{rl::AuxData{}});
        trace.OnTrain(rl::TrainEvent{ {experience, nullptr, {.exp_step=step}, agent, {}},
            nullptr, result, experience.action });
    };
    add(true, 1.0f, torch::tensor({{4.0f, 5.0f}}), 1);
    REQUIRE(rows.size() == 1);
    const auto retained = rows.front();
    CHECK_FALSE(retained->complete);
    CHECK(std::isnan(retained->values[0]));
    add(false, 0.0f, torch::tensor({{6.0f, 2.0f}}), 2);
    CHECK(completed_ids == std::vector<int64_t>{retained->id});
    CHECK(retained->complete);
    CHECK(retained->values[0] == Catch::Approx(0.6f));
    CHECK(retained->values[0] == retained->values[1]);
    CHECK_FALSE(rows.back()->complete);
    trace.Reset();
    rows.clear();
    completed_ids.clear();
    add(false, 0.0f, torch::tensor({{2.0f, 1.0f}}), 3);
    CHECK(completed_ids.empty());

}

TEST_CASE("DQN step diagnostics match replay completion for n-step episode boundaries", "[dqn][rfd0094][diagnostic_replay]")
{
    const int n = GENERATE(1, 3);
    const int boundary = GENERATE(0, 1, 2, 3);
    const bool tbo = GENERATE(false, true);
    const bool munchausen = GENERATE(false, true);
    ScopedNoopMetricsLogger logger;
    auto config = MakeDeviceForwardDefaultDqnConfig();
    config.learner.use_n_step = true;
    config.learner.n_step = n;
    config.learner.gamma.value = 0.9f;
    config.learner.use_tbo = tbo;
    config.learner.munchausen.enabled = munchausen;
    config.learner.per_eps = 0.03f;
    config.learner.use_per_prio_clip = true;
    config.learner.per_prio_clip_value = 20.0f;
    config.reward_scaler.use_dynamic_scaling = false;
    config.reward_scaler.use_auto_post_scale = false;
    config.reward_scaler.manual_post_scale = 1.0f;
    config.reward_scaler.constant_scale = 1.25f;
    config.reward_scaler.use_clipping = true;
    config.reward_scaler.clip_range = 1.0f;
    auto spec = MakeLearnerEnvSpec();
    spec.action_spec.value_labels = {"a0", "a1"};
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(config, MakeAgentForwardNetworkConfig(),
        rl::BatchEnvSpec{1, 1}, spec, torch::kCPU, 345);
    auto diagnostics = agent->CreateStepDiagnosticsProvider({.num_lanes = 1, .keys = {}, .actor = nullptr});
    REQUIRE(diagnostics);
    const dqn::ActorQHintValues hint_config{ .munchausen = config.learner.munchausen.Resolve(
        config.learner.munchausen.alpha.value, config.learner.munchausen.entropy_tau.value),
        .use_tbo = tbo, .tbo_epsilon = config.learner.tbo_epsilon };
    std::vector<rl::ReplayInitialPriorityHintRow> hints;
    std::vector<float> rewards;
    for (int step = 0; step < 8; ++step) {
        const auto q = torch::tensor({{0.2f * step, 0.3f + 0.1f * step}});
        const auto action = torch::tensor({step % 2}, torch::kInt64);
        const auto payload = dqn::MakeActorQHint(q, action, hint_config);
        rl::ReplayInitialPriorityHintRow hint;
        for (int k = 0; k < dqn::kActorQHintColumnCount; ++k) hint.push_back(payload[0][k].item<float>());
        hints.push_back(hint);
        const float reward = 0.25f * (step + 1);
        rewards.push_back(std::clamp(reward * 1.25f, -1.0f, 1.0f));
        const bool done = step == 1 && (boundary == 1 || boundary == 3);
        const bool truncated = step == 1 && (boundary == 2 || boundary == 3);
        auto info = MakeDiagnosticActionInfo(action, anet::TensorDict{}, rl::AuxData{{"q_values", q}}, config.learner);
        const rl::BatchExperience experience{ MakeDeterminismState(step, 1, {false}, {false}, {step == 0 || (step == 2 && boundary != 0)}),
            info, torch::tensor({reward}), MakeDeterminismState(step + 1, 1, {done}, {truncated}, {false}) };
        const auto result = diagnostics->Observe(experience);
        for (const auto& completed : result.completed) {
            const int start = step - static_cast<int>(completed.steps_back);
            const bool crosses = boundary != 0 && start <= 1 && start + n > 1;
            const int actual_n = crosses ? 2 - start : n;
            const bool terminal = crosses && (boundary == 1 || boundary == 3);
            const bool cut = crosses && boundary == 2;
            float target_return = 0;
            for (int j = actual_n - 1; j >= 0; --j) target_return = rewards[start + j] + config.learner.gamma.value * target_return;
            const auto slot = anet::test::CompleteReferenceInitialPriority(
                n, actual_n, target_return, static_cast<float>(std::pow(config.learner.gamma.value, actual_n)),
                terminal, cut, hints[start], hints[step], dqn::CreateInitialPriorityEstimator(config.learner));
            if (cut) {
                CHECK(std::isnan(*completed.values->GetScalar("approx_priority", 0)));
                CHECK(slot.source == rl::ReplayPrioritySource::MAX_INITIAL);
            } else {
                CHECK(slot.source == rl::ReplayPrioritySource::ACTOR_INITIAL);
                CHECK(*completed.values->GetScalar("approx_priority", 0) == Catch::Approx(slot.raw).margin(1e-6));
            }
        }
    }
}

TEST_CASE("DQN diagnostics do not wait for the learner Agent mutex", "[dqn][rfd0094][diagnostic_no_wait]")
{
    class LockableAgent final : public dqn::DefaultDQNAgent {
    public:
        using dqn::DefaultDQNAgent::DefaultDQNAgent;
        std::shared_mutex& MutexForTest() { return *mutex_; }
    };
    const bool dynamic = GENERATE(false, true);
    ScopedNoopMetricsLogger logger;
    auto config = MakeDeviceForwardDefaultDqnConfig();
    config.reward_scaler.use_dynamic_scaling = dynamic;
    auto spec = MakeLearnerEnvSpec();
    spec.action_spec.value_labels = {"a0", "a1"};
    auto agent = std::make_shared<LockableAgent>(config, MakeAgentForwardNetworkConfig(),
        rl::BatchEnvSpec{1, 1}, spec, torch::kCPU, 123);
    auto diagnostics = agent->CreateStepDiagnosticsProvider({.num_lanes = 1, .keys = {}, .actor = nullptr});
    auto action = MakeDiagnosticActionInfo(torch::tensor({0}, torch::kInt64),
        anet::TensorDict{}, rl::AuxData{{"q_values", torch::tensor({{1.0f, 2.0f}})}}, config.learner);
    const rl::BatchExperience experience{MakeDeterminismState(0, 1, {false}, {false}, {true}),
        action, torch::tensor({1.0f}), MakeDeterminismState(1, 1, {false}, {false}, {false})};
    // 初回のTensor処理を先に済ませ、ロック待ちだけを制御する。
    diagnostics->Observe(experience);
    diagnostics->Reset();
    std::unique_lock lock(agent->MutexForTest());
    auto observed = std::async(std::launch::async, [&] { return diagnostics->Observe(experience); });
    const auto status = observed.wait_for(std::chrono::seconds(1));
    // REDでも必ず解放してからjoin相当のgetを行い、テストを停止可能にする。
    lock.unlock();
    const auto result = observed.get();
    CHECK(status == std::future_status::ready);
    REQUIRE(result.current);
    CHECK(result.current->GetScalar("q_sa", 0) == 1.0f);
}

TEST_CASE("DefaultDQN baseline uses Head booleans and leaves omitted skip length unset", "[dqn][rfd0098][rfd0098-r2-baseline]")
{
    ScopedNoopMetricsLogger metrics_logger;
    // 未指定は有効なJと区別し、明示した0は無効Runでも拒否する。
    auto data = WithDefaultForwardConfig();
    CHECK(dqn::DefaultDQNAgentConfig(data).skip.max_length == 0);
    for (const bool enabled : {false, true}) {
        auto invalid = data;
        invalid.Set("DefaultDQNAgent.skip.enabled", enabled);
        invalid.Set("DefaultDQNAgent.skip.max_length", 0);
        CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(invalid), Catch::Matchers::ContainsSubstring("skip.max_length"));
    }
    data.Set("DefaultDQNAgent.skip.enabled", true);
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data), Catch::Matchers::ContainsSubstring("skip.max_length"));

    // 新しいboolの2キーを設定からAgentへ通し、primaryとskipの観測出力で確認する。
    data.Set("DefaultDQNAgent.skip.max_length", 1);
    data.Set("DefaultDQNAgent.quantile_mode", "none");
    data.Set("DefaultDQNAgent.head.use_dueling", false);
    data.Set("DefaultDQNAgent.skip.head.use_dueling", true);
    data.Set("DefaultDQNAgent.head.init.mode", "constant");
    data.Set("DefaultDQNAgent.head.init.constant_val", 0.25);
    data.Set("DefaultDQNAgent.skip.head.init.constant_val", 0.0);


    data.Set("DefaultDQNAgent.learner.use_fused_optimizer", false);
    data.Set("DefaultDQNAgent.actor.[eval].network", "online");
    data.Set("DefaultDQNAgent.net.block.[Flatten].type", kFixtureFlatten);
    data.Set("DefaultDQNAgent.net.branch.[main].bind", kVectorKey);
    data.Set("DefaultDQNAgent.net.branch.[main].structure", kFixtureFlatten);
    data.Set("DefaultDQNAgent.net.body.output.[features]", "main");
    data.Set("DefaultDQNAgent.skip.net.branch.[main].bind", "features");
    data.Set("DefaultDQNAgent.skip.net.body.output.[features]", "main");
    data = WithDefaultForwardConfig(data);
    const auto env = MakeIqnTracerEnvSpec();
    const auto agent = std::make_shared<dqn::DefaultDQNAgent>(dqn::DefaultDQNAgentConfig(data),
        anet::nn::NetworkConfig(data, "DefaultDQNAgent.net"), rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, env, torch::kCPU, 98);
    const anet::TensorDict obs{{kVectorKey, torch::ones({2, 4})}};
    auto forward = agent->GetTensorDictFunction("policy-net.forward");
    REQUIRE(forward.has_value());
    const auto output = (*forward)(obs);
    CHECK_FALSE(agent->GetTensorDictFunction("policy-net.v_values").has_value());
    const auto actor = agent->CreateActor(rl::ActorRequest{.batch_env_spec={.num_envs = 2, .num_threads = 1}, .env_spec=env,
        .device=torch::Device(torch::kCPU), .seed=98, .actor_key="eval"});
    const auto flags = torch::zeros({2}, torch::kBool);
    const auto action = actor->MakeAction({}, rl::BatchState(obs, flags, flags, flags));
    const auto expected = output.At("q").gather(1, action->GetAction().unsqueeze(1));
    CHECK(torch::allclose(action->GetAuxData().at("skip_q_values"), expected));
}

TEST_CASE("Skip IQN consumes fused primary features with optional action input", "[dqn][rfd0098][rfd0098-r2-fused]")
{
    ScopedNoopMetricsLogger metrics_logger;
    const bool action_in = GENERATE(false, true);
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.quantile_mode", "iqn");
    data.Set("DefaultDQNAgent.head.use_dueling", false);
    data.Set("DefaultDQNAgent.skip.enabled", true);
    data.Set("DefaultDQNAgent.skip.max_length", 3);
    data.Set("DefaultDQNAgent.skip.network_mode", action_in ? "action_in" : "action_out");


    data.Set("DefaultDQNAgent.learner.use_fused_optimizer", false);
    data.Set("DefaultDQNAgent.actor.[eval].network", "online");
    data.Set("net.block.[R2Linear].type", "Linear");
    data.Set("net.block.[R2Linear].linear.out_features", 4);
    data.Set("net.block.[R2Cos].type", "CosineEmbedding");
    data.Set("net.block.[R2Cos].cos.num_basis", 4);
    data.Set("net.block.[Flatten].type", kFixtureFlatten);
    data.Set("DefaultDQNAgent.net.branch.[main].bind", kVectorKey);
    data.Set("DefaultDQNAgent.net.branch.[main].structure", "Flatten > R2Linear");
    data.Set("DefaultDQNAgent.net.branch.[tau].bind", "taus");
    data.Set("DefaultDQNAgent.net.branch.[tau].structure", "R2Cos > R2Linear");
    data.Set("DefaultDQNAgent.net.branch.[fused].bind", "main * tau");
    data.Set("DefaultDQNAgent.net.body.output.[features]", "fused");
    data.Set("DefaultDQNAgent.net.body.output.[skip_context]", "fused");
    data.Set("DefaultDQNAgent.skip.net.branch.[skip_feature].bind", action_in ? "skip_context, action" : "skip_context");
    data.Set("DefaultDQNAgent.skip.net.branch.[skip_feature].bind_concat_dim", -1);
    data.Set("DefaultDQNAgent.skip.net.branch.[skip_feature].structure", "R2Linear");
    data.Set("DefaultDQNAgent.skip.net.body.output.[features]", "skip_feature");
    data = WithDefaultForwardConfig(data);
    const auto env = MakeIqnTracerEnvSpec();
    const auto make = [&](const anet::ConfigData& settings) {
        return std::make_shared<dqn::DefaultDQNAgent>(dqn::DefaultDQNAgentConfig(settings),
            anet::nn::NetworkConfig(settings, "DefaultDQNAgent.net"), rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, env, torch::kCPU, 98);
    };
    // skipにtau枝が無くても、融合済みrank3特徴で構築・Actor実行できる。
    std::shared_ptr<dqn::DefaultDQNAgent> agent;
    REQUIRE_NOTHROW(agent = make(data));
    const auto actor = agent->CreateActor(rl::ActorRequest{.batch_env_spec={.num_envs = 2, .num_threads = 1}, .env_spec=env,
        .device=torch::Device(torch::kCPU), .seed=98, .actor_key="eval"});
    const auto flags = torch::zeros({2}, torch::kBool);
    const rl::BatchState state(anet::TensorDict{{kVectorKey, torch::ones({2,4})}}, flags, flags, flags);
    const auto action = actor->MakeAction({}, state);
    REQUIRE(action->GetAuxData().at("skip_q_values").sizes() == torch::IntArrayRef({2,3}));
    CHECK(torch::isfinite(action->GetAuxData().at("skip_q_values")).all().item<bool>());
    // 分位軸を落とした配線は、Head自身のrank契約が拒否する。
    auto invalid = data;
    invalid.Set("DefaultDQNAgent.net.body.output.[skip_context]", "main");
    CHECK_THROWS_WITH(make(invalid), Catch::Matchers::ContainsSubstring("rank=3"));
}

TEST_CASE("RFD0101 Actor values survive transfer and action replacement", "[rfd0101][dqn]")
{
    ScopedNoopMetricsLogger logger;
    auto config = MakeDeviceForwardDefaultDqnConfig();
    config.learner.use_per = false;
    auto set = [](anet::ProfiledValueConfig<float>& value, float end) {
        value.type = "linear"; value.start = 0.1f; value.end = end; value.steps = 8;
    };
    set(config.learner.gamma, 0.9f);
    set(config.learner.munchausen.alpha, 0.4f);
    set(config.learner.munchausen.entropy_tau, 0.7f);
    const auto env = MakeLearnerEnvSpec();
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(config, MakeAgentForwardNetworkConfig(),
        rl::BatchEnvSpec{1, 1}, env, torch::kCPU, 101);
    auto actor = agent->CreateActor(rl::ActorRequest{.batch_env_spec={1, 1}, .env_spec=env,
        .device=torch::kCPU, .seed=101, .actor_key="train"});
    const auto state = MakeDeterminismState(0, 1, {false}, {false}, {true});
    const auto first = actor->MakeAction({}, state);
    rl::StepCounts counts;
    counts.exp_step = 8;
    const auto current = actor->MakeAction(counts, state);
    const auto forced = actor->MakeAction(counts, state, {.forced_actions = rl::ForcedActions::ByLane(0, 0)});
    for (const auto& result : {current, current->To(torch::kCPU), forced}) {
        for (const auto& [key, expected] : std::vector<std::pair<std::string, float>>{
            {"gamma", 0.9f}, {"munchausen_alpha", 0.4f}, {"munchausen_entropy_tau", 0.7f}}) {
            REQUIRE(result->GetScalar(key).has_value());
            CHECK(*result->GetScalar(key) == Catch::Approx(expected));
        }
    }
    CHECK(*first->GetScalar("gamma") == Catch::Approx(0.1f));
}

TEST_CASE("DQN classifies unused forward entries and inactive owned prefixes", "[rfd0117][dqn][config_inventory]")
{
    anet::test::LogCaptureGuard logs;
    auto data = WithDefaultForwardConfig(anet::ConfigData{});
    data.Set("DefaultDQNAgent.intrinsic.enabled", false);
    data.Set("DefaultDQNAgent.intrinsic.unused", "unvalidated");
    data.Set("DefaultDQNAgent.skip.enabled", false);
    data.Set("DefaultDQNAgent.skip.unused", "unvalidated");
    data.Set("DefaultDQNAgent.learner.skip.unused", "unvalidated");
    data.Set("DefaultDQNAgent.nn_forward.[unused].unknown", "unvalidated");
    data.Set("DefaultDQNAgent.nn_forward.typo", "kept");
    const dqn::DefaultDQNAgentConfig config(data);
    const auto unread = data.GetUnreadKeys();
    REQUIRE(unread.size() == 1);
    CHECK(unread.front().first == "DefaultDQNAgent.nn_forward.typo");
    logs.Flush();
    CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
        {"DefaultDQNAgent.intrinsic", "enabled=false"}));
    CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
        {"DefaultDQNAgent.skip", "enabled=false"}));
}

TEST_CASE("DQN reads opaque projection values from their original keys", "[rfd0117][dqn][config_inventory]")
{
    auto data = WithDefaultForwardConfig(anet::ConfigData{});
    data.Set("DefaultDQNAgent.head.projection.noisy.sigma0", 0.7);
    data.Set("unread.sentinel", "kept");
    const dqn::DefaultDQNAgentConfig config(data);
    const auto unread = data.GetUnreadKeys();
    REQUIRE(unread.size() == 1);
    CHECK(unread.front().first == "unread.sentinel");
}

TEST_CASE("RFD0101 Agent publishes all learner schedule scalars", "[rfd0101][dqn]")
{
    ScopedNoopMetricsLogger logger;
    auto config = MakeDeviceForwardDefaultDqnConfig();
    config.learner.use_per = false;
    config.learner.update_warmup_steps = 100;
    auto set = [](anet::ProfiledValueConfig<float>& value, float end) {
        value.type = "linear"; value.start = 0.1f; value.end = end; value.steps = 8;
    };
    set(config.learner.learning_rate, 0.001f);
    set(config.learner.weight_decay, 0.04f);
    set(config.model.soft_update_tau, 0.1f);
    set(config.learner.gamma, 0.9f);
    set(config.learner.munchausen.alpha, 0.4f);
    set(config.learner.munchausen.entropy_tau, 0.7f);
    const auto env = MakeLearnerEnvSpec();
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(config, MakeAgentForwardNetworkConfig(),
        rl::BatchEnvSpec{1, 1}, env, torch::kCPU, 101);
    rl::StepCounts counts;
    counts.exp_step = 8;
    agent->CreateLearner()->UpdateFromBatch(counts, MakeDeterminismExperience(0, 1));
    const std::array<float, 6> expected{0.001f, 0.04f, 0.1f, 0.9f, 0.4f, 0.7f};
    for (size_t index = 0; index < expected.size(); ++index) {
        const auto key = dqn::kLearnerScheduleMetricKeys[index];
        REQUIRE(agent->GetScalar(key).has_value());
        CHECK(*agent->GetScalar(key) == Catch::Approx(expected[index]));
    }
}

TEST_CASE("DQN horizon logging names the field that supplied the initial discount", "[dqn][rfd0194]")
{
    ScopedNoopMetricsLogger logger;
    const auto initial_key = GENERATE(std::string("start"), std::string("phase.[warmup].value"));
    const std::string gamma = "DefaultDQNAgent.learner.gamma";
    auto data = WithDefaultForwardConfig();
    data.Set(gamma + ".type", initial_key == "start" ? "linear" : "phased");
    data.Set(gamma + "." + initial_key, 0.5f);
    if (initial_key == "start") {
        data.Set(gamma + ".steps", 4);
        data.Set(gamma + ".end", 0.9f);
    } else {
        data.Set(gamma + ".phases", "warmup");
        data.Set(gamma + ".phase.[warmup].steps", 4);
    }
    auto config = dqn::DefaultDQNAgentConfig(data);
    config.quantile_mode = "none";
    config.head.use_dueling = false;
    anet::test::LogCaptureGuard logs;
    dqn::DefaultDQNAgent agent(config, MakeAgentForwardNetworkConfig(), {1, 1}, MakeLearnerEnvSpec(), torch::kCPU, 101);
    logs.Flush();
    CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
        {gamma + "." + initial_key + "=0.5", "EffectiveHorizon: 2 steps"}));
}

TEST_CASE("RFD0101 Rainbow Actor honors spatial selection and moving endpoints", "[rfd0101][dqn][rainbow]")
{
    ScopedNoopMetricsLogger logger;
    const bool spatial = GENERATE(false, true);
    auto config = MakeDeviceForwardRainbowConfig();
    auto& policy = config.actor["train"].policy;
    policy.epsilon.value = 0.9f;
    policy.spatial.enabled = spatial;
    policy.spatial.scale = "linear";
    policy.spatial.epsilon.first = anet::ProfiledValueConfig<float>{
        .type = "linear", .start = 0.8f, .end = 0.4f, .steps = 8};
    policy.spatial.epsilon.last.value = 0.0f;
    const auto spec = MakeLearnerEnvSpec();
    const rl::BatchEnvSpec batch_spec{.num_envs = 4, .num_threads = 1};
    auto agent = std::make_shared<dqn::RainbowAgent>(config, MakeAgentForwardNetworkConfig(),
        batch_spec, spec, torch::kCPU, 123);
    auto actor = agent->CreateActor(rl::ActorRequest{.batch_env_spec = batch_spec,
        .env_spec = spec, .device = torch::kCPU, .seed = 456, .actor_key = "train"});
    CHECK(*actor->GetScalar("epsilon") == Catch::Approx(spatial ? 0.4f : 0.9f));
    CHECK(*actor->GetScalar("epsilon_lane_min") == Catch::Approx(spatial ? 0.0f : 0.9f));
    auto flags = torch::zeros({4}, torch::kBool);
    rl::BatchState state(anet::TensorDict{{kVectorKey, torch::ones({4, 2})}}, flags, flags, flags);
    actor->MakeAction(rl::StepCounts{.exp_step = 8}, state);
    CHECK(*actor->GetScalar("epsilon") == Catch::Approx(spatial ? 0.2f : 0.9f));
    CHECK(*actor->GetScalar("epsilon_lane_max") == Catch::Approx(spatial ? 0.4f : 0.9f));
}

TEST_CASE("Rainbow rejects intrinsic Trace columns at the first scalar read", "[rfd0127][step_diagnostics]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto agent = std::make_shared<dqn::RainbowAgent>(MakeDeviceForwardRainbowConfig(), MakeAgentForwardNetworkConfig(),
        rl::BatchEnvSpec{1, 1}, MakeLearnerEnvSpec(), torch::kCPU, 123);
    anet::ConfigData columns;
    columns.Set("app.eval_trace_panel.columns.[600]", "$agent_step intrinsic.error");
    rl::StepTraceBuilder trace(columns, MakeLearnerEnvSpec(), agent, 1, nullptr);
    auto flags = torch::zeros({1}, torch::kBool);
    rl::BatchState state(anet::TensorDict{{kVectorKey, torch::ones({1, 2})}}, flags, flags, flags);
    auto actor = agent->CreateActor(rl::ActorRequest{.batch_env_spec = {1, 1},
        .env_spec = MakeLearnerEnvSpec(), .device = torch::kCPU, .seed = 456, .actor_key = "eval"});
    auto action = actor->MakeAction({}, state);
    const rl::BatchExperience experience{state, action, torch::zeros({1}), state};
    auto result = std::make_shared<rl::PlainBatchStepResult>(experience.reward, state, state, 1, 0, std::vector<rl::AuxData>(1));
    CHECK_THROWS_WITH(trace.OnTrain({{experience, nullptr, {}, agent, {}}, nullptr, result, action}),
        Catch::Matchers::ContainsSubstring("column=600") && Catch::Matchers::ContainsSubstring("key='intrinsic.error'"));
}

TEST_CASE("RFD0128 Rainbow rejects unsupported actor policies in both construction paths",
    "[rfd0128][rainbow][policy-validation]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const std::string type = GENERATE("UQE", "1", "ThompsonSampling", "2", "Greedy", "3", "unknown", "Greedyy");
    const std::string message = "='" + type
        + "'; expected EpsilonGreedy (RainbowAgent supports epsilon-greedy only).";
    anet::ConfigData data;
    data.Set("RainbowAgent.actor.[train].policy.policy_type", type);
    CHECK_THROWS_WITH(dqn::RainbowAgentConfig(data),
        Catch::Matchers::ContainsSubstring("RainbowAgent.actor.[train].policy.policy_type" + message));

    dqn::RainbowAgentConfig config;
    config.use_qr = false;
    config.use_dueling_net = false;
    config.learner.replay_capacity = 16;
    config.learner.replay_batch_size = 2;
    config.actor["train"].policy.policy_type = type;
    CHECK_THROWS_WITH(std::make_shared<dqn::RainbowAgent>(config, MakeAgentForwardNetworkConfig(),
        rl::BatchEnvSpec{.num_envs = 1, .num_threads = 1}, MakeLearnerEnvSpec(), torch::kCPU, 128),
        Catch::Matchers::ContainsSubstring("policy_type (not read from config)" + message));
}

TEST_CASE("RFD0128 Rainbow validates actor tau rules in both construction paths",
    "[rfd0128][rainbow][tau-validation]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const bool full_query = GENERATE(false, true);
    const bool count = GENERATE(false, true);
    const std::string rule_key = full_query ? "full_distribution_query.tau_rule" : "tau_rule";
    const std::string field = count ? "num_taus" : "sample_mode";
    const std::string relative_key = rule_key + "." + field;
    const std::string key = "RainbowAgent.actor.[train].policy." + relative_key;
    anet::ConfigData data;
    if (count) data.Set(key, 0);
    else data.Set(key, "unknown");
    CHECK_THROWS_WITH(dqn::RainbowAgentConfig(data), Catch::Matchers::ContainsSubstring(key));

    dqn::RainbowAgentConfig config;
    config.use_qr = false;
    config.use_dueling_net = false;
    config.learner.replay_capacity = 16;
    config.learner.replay_batch_size = 2;
    auto& policy = config.actor["train"].policy;
    auto& rule = full_query ? policy.full_distribution_query.tau_rule : policy.tau_rule;
    if (count) rule.num_taus = 0;
    else rule.sample_mode = "unknown";
    CHECK_THROWS_WITH(std::make_shared<dqn::RainbowAgent>(config, MakeAgentForwardNetworkConfig(),
        rl::BatchEnvSpec{.num_envs = 1, .num_threads = 1}, MakeLearnerEnvSpec(), torch::kCPU, 128),
        Catch::Matchers::ContainsSubstring(relative_key + " (not read from config)"));
}

TEST_CASE("RFD0128 Rainbow accepts epsilon-greedy aliases and shared scalar routes",
    "[rfd0128][rainbow][scalar-routes]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const std::string type = GENERATE("EpsilonGreedy", "0");
    anet::ConfigData data;
    data.Set("RainbowAgent.actor.[train].policy.policy_type", type);
    CHECK_NOTHROW(dqn::RainbowAgentConfig(data));
    dqn::RainbowAgentConfig config(data);
    config.use_qr = false;
    config.use_dueling_net = false;
    config.learner.replay_capacity = 16;
    config.learner.replay_batch_size = 2;
    auto agent = std::make_shared<dqn::RainbowAgent>(config, MakeAgentForwardNetworkConfig(),
        rl::BatchEnvSpec{.num_envs = 1, .num_threads = 1}, MakeLearnerEnvSpec(), torch::kCPU, 128);
    const auto epsilon = agent->GetScalar("target_policy.epsilon");
    CHECK(epsilon.has_value());
    if (epsilon) CHECK(*epsilon == 0.0f);
    for (const auto* key : {"plasticity_probe_srank", kNoisyQAbsDiffKey}) {
        const auto value = agent->GetScalar(key);
        CHECK(value.has_value());
        if (value) CHECK(std::isnan(*value));
    }
    CHECK_FALSE(agent->GetScalar("unknown"));
}
