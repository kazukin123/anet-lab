#include "catch.hpp"

#include "anet/image_cls_agent.hpp"
#include "anet/metrics_logger.hpp"
#include "nn_impl.hpp"

#include <filesystem>
#include <memory>
#include <shared_mutex>
#include <vector>

namespace {

class VisualTraceTestModule final : public anet::nn::NetworkModule {
public:
    torch::Tensor Forward(torch::Tensor input) override
    {
        return input + 1.0f;
    }

    bool IsConv2dVisualizable() const override { return true; }
};

class ImageClsTraceTestHead final : public anet::nn::NetworkHead {
public:
    anet::TensorDict Forward(const anet::TensorDict& feature_dict) override
    {
        auto feature = feature_dict.At("main_feature").flatten(1);
        auto logits = torch::stack({ feature.select(1, 0), feature.select(1, 1) }, 1);
        return anet::TensorDict{ { "logits", logits } };
    }
};

class ImageClsTrainableTestHead final : public anet::nn::NetworkHead {
public:
    ImageClsTrainableTestHead()
    {
        linear_ = register_module("linear", torch::nn::Linear(4, 2));
    }

    anet::TensorDict Forward(const anet::TensorDict& feature_dict) override
    {
        auto feature = feature_dict.At("main_feature").flatten(1);
        return anet::TensorDict{ { "logits", linear_->forward(feature) } };
    }

private:
    torch::nn::Linear linear_{ nullptr };
};

class NoopMetricsBackend final : public anet::IBackend {
public:
    void Open(const std::filesystem::path&, const std::string&) override {}
    void WriteJsonl(const anet::json&) override {}
    void Flush() override {}
};

class ScopedNoopMetricsLogger final {
public:
    ScopedNoopMetricsLogger()
    {
        anet::MetricsLogger::Reset();
        anet::MetricsLoggerConfig logger_config;
        logger_config.run_name_tmpl = "image_cls_agent_lr_test";
        anet::MetricsLogger::Init(std::make_unique<NoopMetricsBackend>(), logger_config, "out/test-tmp");
    }

    ~ScopedNoopMetricsLogger()
    {
        anet::MetricsLogger::Reset();
    }
};

void EnsureImageClsNnInitialized()
{
    static const bool initialized = [] {
        anet::nn::InitNN();
        return true;
    }();
    (void)initialized;
}

std::shared_ptr<anet::nn::Network> MakeImageClsTraceTestNetwork()
{
    anet::TensorSpec grid_spec;
    grid_spec.type = anet::SpaceType::Grid;
    grid_spec.shape = { 1, 2, 2 };
    grid_spec.dtype = torch::kFloat32;

    anet::TensorSpecMap input_specs;
    input_specs[anet::rl::ObsKeys::kGrid] = grid_spec;

    auto block = std::make_shared<anet::nn::NetworkBlock>(
        "Conv2d_0",
        std::make_shared<VisualTraceTestModule>());
    auto network_struct = std::make_shared<anet::nn::NetworkStruct>(
        std::vector<std::shared_ptr<anet::nn::NetworkBlock>>{ block });
    auto branch = std::make_shared<anet::nn::NetworkBranch>(
        "main_feature",
        std::vector<std::string>{ anet::rl::ObsKeys::kGrid },
        network_struct);

    anet::nn::NetworkConfig network_config;
    network_config.output_keys["main_feature"] = "main_feature";

    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{ branch },
        input_specs,
        std::vector<std::string>{},
        network_config.output_keys);

    return std::make_shared<anet::nn::Network>(
        network_config,
        input_specs,
        nullptr,
        body,
        std::make_shared<ImageClsTraceTestHead>());
}

std::shared_ptr<anet::nn::Network> MakeImageClsTrainableTestNetwork()
{
    anet::TensorSpec grid_spec;
    grid_spec.type = anet::SpaceType::Grid;
    grid_spec.shape = { 1, 2, 2 };
    grid_spec.dtype = torch::kFloat32;

    anet::TensorSpecMap input_specs;
    input_specs[anet::rl::ObsKeys::kGrid] = grid_spec;

    auto block = std::make_shared<anet::nn::NetworkBlock>(
        "Conv2d_0",
        std::make_shared<VisualTraceTestModule>());
    auto network_struct = std::make_shared<anet::nn::NetworkStruct>(
        std::vector<std::shared_ptr<anet::nn::NetworkBlock>>{ block });
    auto branch = std::make_shared<anet::nn::NetworkBranch>(
        "main_feature",
        std::vector<std::string>{ anet::rl::ObsKeys::kGrid },
        network_struct);

    anet::nn::NetworkConfig network_config;
    network_config.output_keys["main_feature"] = "main_feature";

    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{ branch },
        input_specs,
        std::vector<std::string>{},
        network_config.output_keys);

    return std::make_shared<anet::nn::Network>(
        network_config,
        input_specs,
        nullptr,
        body,
        std::make_shared<ImageClsTrainableTestHead>());
}

anet::rl::BatchState MakeImageClsBatchState(torch::Tensor grid)
{
    const int64_t batch_size = grid.size(0);
    auto bool_options = torch::TensorOptions().dtype(torch::kBool);
    return anet::rl::BatchState(
        anet::TensorDict{ { anet::rl::ObsKeys::kGrid, grid } },
        torch::zeros({ batch_size }, bool_options),
        torch::zeros({ batch_size }, bool_options),
        torch::zeros({ batch_size }, bool_options));
}

anet::rl::BatchState MakeImageClsLearningBatchState(torch::Tensor grid, torch::Tensor labels)
{
    const int64_t batch_size = grid.size(0);
    auto bool_options = torch::TensorOptions().dtype(torch::kBool);
    return anet::rl::BatchState(
        anet::TensorDict{
            { anet::rl::ObsKeys::kGrid, grid },
            { anet::rl::ObsKeys::kVector, labels.reshape({ batch_size, 1 }) }
        },
        torch::zeros({ batch_size }, bool_options),
        torch::zeros({ batch_size }, bool_options),
        torch::zeros({ batch_size }, bool_options));
}

anet::rl::BatchExperience MakeImageClsLearningExperience()
{
    auto grid = torch::arange(8, torch::kFloat32).view({ 2, 1, 2, 2 }).div(8.0f);
    auto labels = torch::tensor({ 0, 1 }, torch::TensorOptions().dtype(torch::kInt64));
    auto state = MakeImageClsLearningBatchState(grid, labels);
    auto action_info = std::make_shared<anet::rl::BatchActionInfo>(
        torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kInt64)));
    return anet::rl::BatchExperience(
        state,
        action_info,
        torch::zeros({ 2 }, torch::kFloat32),
        state.Clone());
}

anet::rl::img_cls::ImageClsAgentConfig MakeImageClsLearningRateTestConfig()
{
    anet::ConfigData config_data;
    config_data.Set("ImageClsAgent.learning_rate.type", std::string("linear"));
    config_data.Set("ImageClsAgent.learning_rate.start", 0.1);
    config_data.Set("ImageClsAgent.learning_rate.end", 0.01);
    config_data.Set("ImageClsAgent.learning_rate.steps", 10);
    config_data.Set("ImageClsAgent.learning_rate.value", 0.1);
    config_data.Set("ImageClsAgent.weight_decay", 0.0);
    config_data.Set("ImageClsAgent.label_smoothing", 0.0);
    config_data.Set("ImageClsAgent.grad_clip_max_norm", 10.0);
    return anet::rl::img_cls::ImageClsAgentConfig(config_data);
}

anet::rl::EnvSpec MakeImageClsEnvSpec()
{
    anet::TensorSpec grid_spec;
    grid_spec.type = anet::SpaceType::Grid;
    grid_spec.shape = { 1, 2, 2 };
    grid_spec.dtype = torch::kFloat32;

    anet::TensorSpec vector_spec;
    vector_spec.type = anet::SpaceType::Vector;
    vector_spec.shape = { 1 };
    vector_spec.dtype = torch::kInt64;

    anet::rl::EnvSpec spec;
    spec.state_spec.obs_spec[anet::rl::ObsKeys::kGrid] = grid_spec;
    spec.state_spec.obs_spec[anet::rl::ObsKeys::kVector] = vector_spec;
    spec.action_spec.is_discrete = true;
    spec.action_spec.value_labels = { "class0", "class1" };
    spec.reward_range = { 0.0f, 1.0f };
    return spec;
}

anet::nn::NetworkConfig MakeImageClsAgentNetworkConfig()
{
    anet::ConfigData config_data;
    config_data.Set("net.block.[Flatten].type", std::string("Flatten"));
    config_data.Set("net.block.[LinearOut].type", std::string("Linear"));
    config_data.Set("net.block.[LinearOut].linear.out_features", 2);
    config_data.Set("net.branch.[main_feature].bind", std::string(anet::rl::ObsKeys::kGrid));
    config_data.Set("net.branch.[main_feature].structure", std::string("Flatten > LinearOut"));
    config_data.Set("net.body.output.[logits]", std::string("main_feature"));
    return anet::nn::NetworkConfig(config_data);
}

} // namespace

TEST_CASE("ImageClsActor stores nn trace in action aux for Conv2dPanel", "[image_cls][trace]")
{
    auto network = MakeImageClsTraceTestNetwork();
    auto mutex = std::make_shared<std::shared_mutex>();
    anet::rl::img_cls::ImageClsActor actor(
        mutex,
        network,
        anet::rl::RunMode::Eval,
        torch::Device(torch::kCPU));

    auto grid = torch::arange(8, torch::kFloat32).view({ 2, 1, 2, 2 });
    auto expected_input_trace = grid.slice(0, 0, 1).clone();
    auto state = MakeImageClsBatchState(grid);

    auto action_info = actor.MakeAction(anet::rl::StepCounts{}, state);

    auto action = action_info->GetAction(torch::kCPU);
    REQUIRE(action.sizes().vec() == std::vector<int64_t>{ 2 });
    REQUIRE(torch::equal(action, torch::ones({ 2 }, torch::TensorOptions().dtype(torch::kInt64))));

    const auto& info = action_info->GetInfo();
    REQUIRE(info.Contains("probs"));
    const std::vector<int64_t> expected_probs_shape{ 2, 2 };
    REQUIRE(info.At("probs").sizes().vec() == expected_probs_shape);

    auto trace = anet::rl::ExtractNnTrace(action_info->GetAuxData());
    REQUIRE(trace.Contains("main_feature/00_Input"));
    REQUIRE(trace.Contains("main_feature/01_Conv2d_0"));

    auto input_trace = trace.At("main_feature/00_Input");
    auto conv_trace = trace.At("main_feature/01_Conv2d_0");
    const std::vector<int64_t> expected_trace_shape{ 1, 1, 2, 2 };
    REQUIRE(input_trace.sizes().vec() == expected_trace_shape);
    REQUIRE(conv_trace.sizes().vec() == expected_trace_shape);
    REQUIRE(input_trace.scalar_type() == torch::kFloat32);
    REQUIRE(conv_trace.scalar_type() == torch::kFloat32);
    REQUIRE(torch::equal(input_trace, expected_input_trace));
    REQUIRE(torch::equal(conv_trace, expected_input_trace + 1.0f));

    grid.fill_(-999.0f);
    REQUIRE(torch::equal(input_trace, expected_input_trace));
    REQUIRE(input_trace.data_ptr<float>() != grid.data_ptr<float>());
}

TEST_CASE("ImageClsLearner updates shared learning rate before optimizer step", "[image_cls][learning_rate]")
{
    auto network = MakeImageClsTrainableTestNetwork();
    auto mutex = std::make_shared<std::shared_mutex>();
    auto config = MakeImageClsLearningRateTestConfig();
    auto learning_rate = std::make_shared<anet::ProfiledValue<double>>(config.learning_rate);

    anet::rl::img_cls::ImageClsLearner learner(
        config,
        mutex,
        network,
        learning_rate,
        torch::Device(torch::kCPU));

    CHECK(learning_rate->Value() == Catch::Approx(0.1));

    anet::rl::StepCounts step;
    step.exp_step = 5;
    const auto result = learner.UpdateFromBatch(step, MakeImageClsLearningExperience());

    REQUIRE(result.size() == 1);
    CHECK(learning_rate->Value() == Catch::Approx(0.055));
}

TEST_CASE("ImageClsAgent exposes current learning rate scalar", "[image_cls][learning_rate]")
{
    EnsureImageClsNnInitialized();
    ScopedNoopMetricsLogger metrics_logger;

    auto config = MakeImageClsLearningRateTestConfig();
    auto env_spec = MakeImageClsEnvSpec();
    anet::rl::img_cls::ImageClsAgent agent(
        config,
        MakeImageClsAgentNetworkConfig(),
        env_spec,
        anet::rl::BatchEnvSpec{ 2, 1 },
        torch::Device(torch::kCPU),
        123);

    auto initial_learning_rate = agent.GetScalar("learning_rate");
    REQUIRE(initial_learning_rate.has_value());
    CHECK(*initial_learning_rate == Catch::Approx(0.1));

    auto learner = agent.CreateLearner();
    anet::rl::StepCounts step;
    step.exp_step = 5;
    learner->UpdateFromBatch(step, MakeImageClsLearningExperience());

    auto current_learning_rate = agent.GetScalar("learning_rate");
    REQUIRE(current_learning_rate.has_value());
    CHECK(*current_learning_rate == Catch::Approx(0.055));
    CHECK_FALSE(agent.GetScalar("unknown").has_value());
}
