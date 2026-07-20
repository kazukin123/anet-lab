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

} // namespace

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
