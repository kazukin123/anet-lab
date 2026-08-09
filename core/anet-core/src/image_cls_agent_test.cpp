#include "anet/catch_test.hpp"

#include "anet/image_cls_agent.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/random.hpp"
#include "anet/serialize.hpp"
#include "anet/test_util.hpp"
#include "nn_impl.hpp"
#include "nn_heads.hpp"

#include <ATen/autocast_mode.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <shared_mutex>
#include <string>
#include <tuple>
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

class ImageClsRecordingTestModule final : public anet::nn::NetworkModule {
public:
    torch::Tensor Forward(torch::Tensor input) override
    {
        last_input = input.detach().to(torch::kCPU).clone();
        return input;
    }

    torch::Tensor last_input;
};

class ImageClsAutocastProbeTestModule final : public anet::nn::NetworkModule {
public:
    torch::Tensor Forward(torch::Tensor input) override
    {
        last_autocast_enabled = at::autocast::is_autocast_enabled(input.device().type());
        return input;
    }

    bool last_autocast_enabled = false;
};

class ImageClsFixedLinearTestHead final : public anet::nn::NetworkHead {
public:
    explicit ImageClsFixedLinearTestHead(int64_t feature_size)
    {
        auto options = torch::nn::LinearOptions(feature_size, 2).bias(false);
        linear_ = register_module("linear", torch::nn::Linear(options));
        torch::NoGradGuard no_grad;
        linear_->weight.zero_();
        linear_->weight.index_put_({ 0, 0 }, 1.0f);
        linear_->weight.index_put_({ 1, 1 }, 1.0f);
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
        std::vector<std::vector<std::string>>{ { anet::rl::ObsKeys::kGrid } },
        1,
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
        std::vector<std::vector<std::string>>{ { anet::rl::ObsKeys::kGrid } },
        1,
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

struct ImageClsRecordingNetworkFixture {
    std::shared_ptr<anet::nn::Network> network;
    std::shared_ptr<ImageClsRecordingTestModule> recorder;
};

struct ImageClsAutocastProbeNetworkFixture {
    std::shared_ptr<anet::nn::Network> network;
    std::shared_ptr<ImageClsAutocastProbeTestModule> probe;
};

ImageClsRecordingNetworkFixture MakeImageClsRecordingTestNetwork(
    torch::Dtype grid_dtype = torch::kUInt8,
    int64_t height = 2,
    int64_t width = 2)
{
    anet::TensorSpec grid_spec;
    grid_spec.type = anet::SpaceType::Grid;
    grid_spec.shape = { 1, height, width };
    grid_spec.dtype = grid_dtype;

    anet::TensorSpecMap input_specs;
    input_specs[anet::rl::ObsKeys::kGrid] = grid_spec;

    auto recorder = std::make_shared<ImageClsRecordingTestModule>();
    auto block = std::make_shared<anet::nn::NetworkBlock>("Record_0", recorder);
    auto network_struct = std::make_shared<anet::nn::NetworkStruct>(
        std::vector<std::shared_ptr<anet::nn::NetworkBlock>>{ block });
    auto branch = std::make_shared<anet::nn::NetworkBranch>(
        "main_feature",
        std::vector<std::vector<std::string>>{ { anet::rl::ObsKeys::kGrid } },
        1,
        network_struct);

    anet::nn::NetworkConfig network_config;
    network_config.output_keys["main_feature"] = "main_feature";

    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{ branch },
        input_specs,
        std::vector<std::string>{},
        network_config.output_keys);

    return {
        std::make_shared<anet::nn::Network>(
            network_config,
            input_specs,
            nullptr,
            body,
            std::make_shared<ImageClsFixedLinearTestHead>(height * width)),
        recorder
    };
}

ImageClsAutocastProbeNetworkFixture MakeImageClsAutocastProbeTestNetwork()
{
    anet::TensorSpec grid_spec;
    grid_spec.type = anet::SpaceType::Grid;
    grid_spec.shape = { 1, 2, 2 };
    grid_spec.dtype = torch::kFloat32;

    anet::TensorSpecMap input_specs;
    input_specs[anet::rl::ObsKeys::kGrid] = grid_spec;

    auto probe = std::make_shared<ImageClsAutocastProbeTestModule>();
    auto block = std::make_shared<anet::nn::NetworkBlock>("Probe_0", probe);
    auto network_struct = std::make_shared<anet::nn::NetworkStruct>(
        std::vector<std::shared_ptr<anet::nn::NetworkBlock>>{ block });
    auto branch = std::make_shared<anet::nn::NetworkBranch>(
        "main_feature",
        std::vector<std::vector<std::string>>{ { anet::rl::ObsKeys::kGrid } },
        1,
        network_struct);

    anet::nn::NetworkConfig network_config;
    network_config.output_keys["main_feature"] = "main_feature";

    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{ branch },
        input_specs,
        std::vector<std::string>{},
        network_config.output_keys);

    return {
        std::make_shared<anet::nn::Network>(
            network_config,
            input_specs,
            nullptr,
            body,
            std::make_shared<ImageClsFixedLinearTestHead>(4)),
        probe
    };
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

anet::rl::BatchExperience MakeImageClsLearningExperience(torch::Tensor grid, torch::Tensor labels);

anet::rl::BatchExperience MakeImageClsLearningExperience()
{
    auto grid = torch::arange(8, torch::kFloat32).view({ 2, 1, 2, 2 }).div(8.0f);
    auto labels = torch::tensor({ 0, 1 }, torch::TensorOptions().dtype(torch::kInt64));
    return MakeImageClsLearningExperience(grid, labels);
}

anet::rl::BatchExperience MakeImageClsLearningExperience(torch::Tensor grid, torch::Tensor labels)
{
    auto state = MakeImageClsLearningBatchState(grid, labels);
    const int64_t batch_size = grid.size(0);
    auto action_info = std::make_shared<anet::rl::BatchActionInfo>(
        torch::zeros({ batch_size }, torch::TensorOptions().dtype(torch::kInt64)));
    return anet::rl::BatchExperience(
        state,
        action_info,
        torch::zeros({ batch_size }, torch::kFloat32),
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

anet::ConfigData MakeImageClsMixTestConfigData()
{
    anet::ConfigData config_data;
    config_data.Set("ImageClsAgent.learning_rate.value", 0.01);
    config_data.Set("ImageClsAgent.weight_decay", 0.0);
    config_data.Set("ImageClsAgent.label_smoothing", 0.0);
    config_data.Set("ImageClsAgent.grad_clip_max_norm", 10.0);
    config_data.Set("ImageClsAgent.mixup.enabled", true);
    config_data.Set("ImageClsAgent.mixup.mixup_alpha", 0.4);
    config_data.Set("ImageClsAgent.mixup.cutmix_alpha", 1.0);
    config_data.Set("ImageClsAgent.mixup.prob", 1.0);
    config_data.Set("ImageClsAgent.mixup.switch_prob", 0.0);
    return config_data;
}

anet::ConfigData MakeImageClsSerializeTestConfigData()
{
    anet::ConfigData config_data;
    config_data.Set("ImageClsAgent.learning_rate.value", 0.01);
    config_data.Set("ImageClsAgent.weight_decay", 0.0);
    config_data.Set("ImageClsAgent.label_smoothing", 0.0);
    config_data.Set("ImageClsAgent.grad_clip_max_norm", 10.0);
    config_data.Set("ImageClsAgent.learn_log_interval", 0);
    config_data.Set("ImageClsAgent.mixup.enabled", false);
    return config_data;
}

anet::rl::img_cls::ImageClsAgentConfig MakeImageClsMixTestConfig(const anet::ConfigData& config_data)
{
    return anet::rl::img_cls::ImageClsAgentConfig(config_data);
}

anet::rl::EnvSpec MakeImageClsEnvSpec();
anet::nn::NetworkConfig MakeImageClsAgentNetworkConfig();

std::shared_ptr<const anet::rl::img_cls::ImageClsUpdateResult> RunImageClsRecordingUpdate(
    ImageClsRecordingNetworkFixture& fixture,
    const anet::rl::img_cls::ImageClsAgentConfig& config,
    const anet::rl::StepCounts& step,
    const anet::rl::BatchExperience& experience,
    std::optional<anet::seed_t> seed = 123)
{
    auto mutex = std::make_shared<std::shared_mutex>();
    auto learning_rate = std::make_shared<anet::ProfiledValue<double>>(config.learning_rate);
    anet::rl::img_cls::ImageClsLearner learner(
        config,
        mutex,
        fixture.network,
        learning_rate,
        torch::Device(torch::kCPU),
        seed);

    auto result_list = learner.UpdateFromBatch(step, experience);
    REQUIRE(result_list.size() == 1);
    auto result = std::dynamic_pointer_cast<const anet::rl::img_cls::ImageClsUpdateResult>(result_list[0]);
    REQUIRE(result != nullptr);
    return result;
}

std::shared_ptr<const anet::rl::img_cls::ImageClsUpdateResult> RunImageClsAgentMixUpdate(
    anet::seed_t agent_seed,
    torch::Tensor grid,
    torch::Tensor labels)
{
    torch::manual_seed(20240705);

    auto config = MakeImageClsMixTestConfig(MakeImageClsMixTestConfigData());
    auto env_spec = MakeImageClsEnvSpec();
    anet::rl::img_cls::ImageClsAgent agent(
        config,
        MakeImageClsAgentNetworkConfig(),
        env_spec,
        anet::rl::BatchEnvSpec{ static_cast<int>(grid.size(0)), 1 },
        torch::Device(torch::kCPU),
        agent_seed);

    auto learner = agent.CreateLearner();
    auto result_list = learner->UpdateFromBatch(
        anet::rl::StepCounts{},
        MakeImageClsLearningExperience(grid, labels));
    REQUIRE(result_list.size() == 1);
    auto result = std::dynamic_pointer_cast<const anet::rl::img_cls::ImageClsUpdateResult>(result_list[0]);
    REQUIRE(result != nullptr);
    return result;
}

bool Contains(const std::string& text, const std::string& pattern)
{
    return text.find(pattern) != std::string::npos;
}

std::filesystem::path MakeImageClsCheckpointPath(const std::string& filename)
{
    const auto dir = std::filesystem::path("out") / "test-tmp" / "image_cls_agent_checkpoint";
    std::filesystem::create_directories(dir);
    return dir / filename;
}

int64_t SaveImageClsAgent(anet::rl::img_cls::ImageClsAgent& agent, const std::filesystem::path& path)
{
    std::ofstream ofs(path, std::ios::binary);
    REQUIRE(ofs);
    anet::OutputArchive archive(ofs, path.string());
    return agent.Save(archive);
}

void WriteWrongImageClsHeaderArchive(const std::filesystem::path& path)
{
    std::ofstream ofs(path, std::ios::binary);
    REQUIRE(ofs);
    anet::OutputArchive archive(ofs, path.string());
    archive.Write(anet::ArchiveHeader("DefaultDQNAgent"));
    archive.Write(std::string("dummy config"));
}

torch::Tensor ForwardImageClsActorProbs(
    const std::shared_ptr<anet::rl::Actor>& actor,
    const torch::Tensor& grid)
{
    auto action_info = actor->MakeAction(anet::rl::StepCounts{}, MakeImageClsBatchState(grid.clone()));
    return action_info->GetInfo().At("probs").detach().to(torch::kCPU).clone();
}

torch::Tensor ForwardImageClsAgentProbs(
    anet::rl::img_cls::ImageClsAgent& agent,
    const torch::Tensor& grid)
{
    auto actor = agent.CreateActor(
        anet::rl::BatchEnvSpec{ static_cast<int>(grid.size(0)), 1 },
        MakeImageClsEnvSpec(),
        anet::rl::RunMode::Eval,
        false,
        torch::Device(torch::kCPU));
    return ForwardImageClsActorProbs(actor, grid);
}

void RequireTensorClose(const torch::Tensor& actual, const torch::Tensor& expected, double tolerance = 1e-5)
{
    REQUIRE(actual.sizes().vec() == expected.sizes().vec());
    CHECK(torch::allclose(actual, expected, tolerance, tolerance));
}

float RequireImageClsScalar(
    const std::shared_ptr<const anet::rl::img_cls::ImageClsUpdateResult>& result,
    const std::string& key)
{
    auto scalar = result->GetScalar(key, -1);
    REQUIRE(scalar.has_value());
    return *scalar;
}

void CheckImageClsScalar(
    const std::shared_ptr<const anet::rl::img_cls::ImageClsUpdateResult>& result,
    const std::string& key,
    float expected)
{
    CHECK(RequireImageClsScalar(result, key) == Catch::Approx(expected));
}

double TestSampleBeta(anet::RandomGenerator& rng, double alpha)
{
    const double a = rng.Gamma(static_cast<float>(alpha), 1.0f);
    const double b = rng.Gamma(static_cast<float>(alpha), 1.0f);
    return a / (a + b);
}

struct ExpectedMixResult {
    torch::Tensor perm;
    torch::Tensor targets_b;
    double lambda = 1.0;
    int64_t x1 = 0;
    int64_t y1 = 0;
    int64_t x2 = 0;
    int64_t y2 = 0;
};

struct ExpectedImageClsMetrics {
    float target_prob_mix_norm = 0.0f;
    float accuracy_either = 0.0f;
    float pred_max_prob = 0.0f;
    float same_class_pair_ratio = 0.0f;
};

ExpectedMixResult MakeExpectedMixupResult(
    const anet::rl::img_cls::ImageClsAgentConfig& config,
    int64_t batch_size,
    const torch::Tensor& labels,
    anet::seed_t seed)
{
    anet::RandomGenerator rng(seed);
    ExpectedMixResult expected;
    expected.lambda = TestSampleBeta(rng, config.mixup.mixup_alpha);
    auto gen = rng.GetTorchGenerator(torch::Device(torch::kCPU));
    expected.perm = torch::randperm(batch_size, gen, torch::TensorOptions().dtype(torch::kInt64));
    expected.targets_b = labels.index_select(0, expected.perm);
    return expected;
}

ExpectedMixResult MakeExpectedCutMixResult(
    const anet::rl::img_cls::ImageClsAgentConfig& config,
    int64_t batch_size,
    int64_t height,
    int64_t width,
    const torch::Tensor& labels,
    anet::seed_t seed)
{
    anet::RandomGenerator rng(seed);
    ExpectedMixResult expected;
    expected.lambda = TestSampleBeta(rng, config.mixup.cutmix_alpha);
    auto gen = rng.GetTorchGenerator(torch::Device(torch::kCPU));
    expected.perm = torch::randperm(batch_size, gen, torch::TensorOptions().dtype(torch::kInt64));
    expected.targets_b = labels.index_select(0, expected.perm);

    const double cut_ratio = std::sqrt(std::max(0.0, 1.0 - expected.lambda));
    const int64_t cut_w = static_cast<int64_t>(std::round(width * cut_ratio));
    const int64_t cut_h = static_cast<int64_t>(std::round(height * cut_ratio));
    const int64_t cx = rng.RandInt(0, static_cast<int>(width - 1));
    const int64_t cy = rng.RandInt(0, static_cast<int>(height - 1));
    const int64_t left_w = cut_w / 2;
    const int64_t top_h = cut_h / 2;
    expected.x1 = std::clamp<int64_t>(cx - left_w, 0, width);
    expected.x2 = std::clamp<int64_t>(cx + (cut_w - left_w), 0, width);
    expected.y1 = std::clamp<int64_t>(cy - top_h, 0, height);
    expected.y2 = std::clamp<int64_t>(cy + (cut_h - top_h), 0, height);
    const double area = static_cast<double>(expected.x2 - expected.x1) * static_cast<double>(expected.y2 - expected.y1);
    expected.lambda = 1.0 - area / static_cast<double>(height * width);
    return expected;
}

torch::Tensor MakeExpectedMixupGrid(const torch::Tensor& grid, const torch::Tensor& perm, double lambda)
{
    auto paired = grid.index_select(0, perm);
    return grid.to(torch::kFloat32).mul(lambda)
        .add(paired.to(torch::kFloat32).mul(1.0 - lambda))
        .round()
        .clamp(0, 255)
        .to(torch::kUInt8);
}

torch::Tensor MakeExpectedCutMixGrid(const torch::Tensor& grid, const ExpectedMixResult& expected)
{
    auto mixed = grid.clone();
    auto paired = grid.index_select(0, expected.perm);
    if (expected.x2 > expected.x1 && expected.y2 > expected.y1) {
        using torch::indexing::Slice;
        mixed.index({ Slice(), Slice(), Slice(expected.y1, expected.y2), Slice(expected.x1, expected.x2) })
            .copy_(paired.index({ Slice(), Slice(), Slice(expected.y1, expected.y2), Slice(expected.x1, expected.x2) }));
    }
    return mixed;
}

ExpectedImageClsMetrics MakeExpectedImageClsMetrics(
    const torch::Tensor& preprocessed_grid,
    const torch::Tensor& labels,
    const ExpectedMixResult& mix,
    double label_smoothing,
    bool mix_applied)
{
    auto feature = preprocessed_grid.flatten(1);
    auto logits = torch::stack({ feature.select(1, 0), feature.select(1, 1) }, 1);
    auto probs = torch::softmax(logits, 1);
    auto preds = logits.argmax(1);
    auto target_prob_a = probs.gather(1, labels.unsqueeze(1)).squeeze(1);
    auto target_prob_b = probs.gather(1, mix.targets_b.unsqueeze(1)).squeeze(1);
    auto target_prob_mix = target_prob_a.mul(mix.lambda).add(target_prob_b.mul(1.0 - mix.lambda));
    const double class_count = static_cast<double>(logits.size(1));
    const double ceiling = (1.0 - label_smoothing)
        * (mix.lambda * mix.lambda + (1.0 - mix.lambda) * (1.0 - mix.lambda))
        + label_smoothing / class_count;

    ExpectedImageClsMetrics metrics;
    metrics.target_prob_mix_norm = static_cast<float>(target_prob_mix.mean().item<float>() / ceiling);
    metrics.accuracy_either = (preds == labels)
        .logical_or(preds == mix.targets_b)
        .to(torch::kFloat32)
        .mean()
        .item<float>();
    metrics.pred_max_prob = std::get<0>(probs.max(/*dim=*/1)).mean().item<float>();
    if (mix_applied) {
        metrics.same_class_pair_ratio = (labels == mix.targets_b)
            .to(torch::kFloat32)
            .mean()
            .item<float>();
    }
    return metrics;
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
    config_data.Set("net.branch.[main_feature].bind", std::string(anet::rl::ObsKeys::kGrid));
    config_data.Set("net.branch.[main_feature].structure", std::string("Flatten"));
    config_data.Set(
        std::string("net.body.output.[") + anet::nn::kKey_DefaultOutput + "]",
        std::string("main_feature"));
    return anet::nn::NetworkConfig(config_data);
}

anet::nn::NetworkConfig MakeImageClsAgentHeadNetworkConfig()
{
    return MakeImageClsAgentNetworkConfig();
}

} // namespace

TEST_CASE("ImageCls mixup config defaults, round-trip and fail-fast validation", "[image_cls][mixup][config]")
{
    anet::rl::img_cls::ImageClsAgentConfig defaults;
    CHECK_FALSE(defaults.mixup.enabled);
    CHECK(defaults.mixup.mixup_alpha == Catch::Approx(0.2));
    CHECK(defaults.mixup.cutmix_alpha == Catch::Approx(1.0));
    CHECK(defaults.mixup.prob == Catch::Approx(1.0));
    CHECK(defaults.mixup.switch_prob == Catch::Approx(0.5));
    CHECK(defaults.learn_log_interval == 0);
    CHECK(defaults.auto_load_file.empty());
    CHECK(defaults.use_fused_optimizer);
    CHECK_FALSE(defaults.bf16.enabled);
    CHECK(defaults.bf16.learner);
    CHECK_FALSE(defaults.bf16.actor);

    auto config_data = MakeImageClsMixTestConfigData();
    config_data.Set("ImageClsAgent.mixup.cutmix_alpha", 2.0);
    config_data.Set("ImageClsAgent.mixup.prob", 0.75);
    config_data.Set("ImageClsAgent.mixup.switch_prob", 0.25);
    config_data.Set("ImageClsAgent.learn_log_interval", 100);
    config_data.Set("ImageClsAgent.auto_load_file", std::string("runs/image_cls/agent_close.anet"));
    config_data.Set("ImageClsAgent.use_fused_optimizer", false);
    config_data.Set("ImageClsAgent.bf16.enabled", true);
    config_data.Set("ImageClsAgent.bf16.learner", false);
    config_data.Set("ImageClsAgent.bf16.actor", true);
    auto config = MakeImageClsMixTestConfig(config_data);
    CHECK(config.mixup.enabled);
    CHECK(config.mixup.mixup_alpha == Catch::Approx(0.4));
    CHECK(config.mixup.cutmix_alpha == Catch::Approx(2.0));
    CHECK(config.mixup.prob == Catch::Approx(0.75));
    CHECK(config.mixup.switch_prob == Catch::Approx(0.25));
    CHECK(config.learn_log_interval == 100);
    CHECK(config.auto_load_file == "runs/image_cls/agent_close.anet");
    CHECK_FALSE(config.use_fused_optimizer);
    CHECK(config.bf16.enabled);
    CHECK_FALSE(config.bf16.learner);
    CHECK(config.bf16.actor);

    const auto config_string = config.ToConfigString();
    CHECK(Contains(config_string, "ImageClsAgent.mixup.enabled = true"));
    CHECK(Contains(config_string, "ImageClsAgent.mixup.mixup_alpha = 0.4"));
    CHECK(Contains(config_string, "ImageClsAgent.mixup.cutmix_alpha = 2"));
    CHECK(Contains(config_string, "ImageClsAgent.mixup.prob = 0.75"));
    CHECK(Contains(config_string, "ImageClsAgent.mixup.switch_prob = 0.25"));
    CHECK(Contains(config_string, "ImageClsAgent.learn_log_interval = 100"));
    CHECK(Contains(config_string, "ImageClsAgent.auto_load_file = runs/image_cls/agent_close.anet"));
    CHECK(Contains(config_string, "ImageClsAgent.use_fused_optimizer = false"));
    CHECK(Contains(config_string, "ImageClsAgent.bf16.enabled = true"));
    CHECK(Contains(config_string, "ImageClsAgent.bf16.learner = false"));
    CHECK(Contains(config_string, "ImageClsAgent.bf16.actor = true"));

    SECTION("prob must stay in [0, 1]")
    {
        auto invalid = MakeImageClsMixTestConfigData();
        invalid.Set("ImageClsAgent.mixup.prob", 1.1);
        CHECK_THROWS(anet::rl::img_cls::ImageClsAgentConfig(invalid));
    }

    SECTION("switch_prob must stay in [0, 1]")
    {
        auto invalid = MakeImageClsMixTestConfigData();
        invalid.Set("ImageClsAgent.mixup.switch_prob", -0.1);
        CHECK_THROWS(anet::rl::img_cls::ImageClsAgentConfig(invalid));
    }

    SECTION("alpha and log interval must be non-negative")
    {
        auto invalid_alpha = MakeImageClsMixTestConfigData();
        invalid_alpha.Set("ImageClsAgent.mixup.mixup_alpha", -0.1);
        CHECK_THROWS(anet::rl::img_cls::ImageClsAgentConfig(invalid_alpha));

        auto invalid_interval = MakeImageClsMixTestConfigData();
        invalid_interval.Set("ImageClsAgent.learn_log_interval", -1);
        CHECK_THROWS(anet::rl::img_cls::ImageClsAgentConfig(invalid_interval));
    }
}

TEST_CASE("ImageClsUpdateResult returns NaN for missing known scalars only", "[image_cls][mixup]")
{
    anet::rl::img_cls::ImageClsUpdateResult result;

    const auto loss = result.GetScalar("loss", -1);
    const auto accuracy = result.GetScalar("accuracy", -1);
    const auto target_prob_mix_norm = result.GetScalar("target_prob_mix_norm", -1);
    const auto accuracy_either = result.GetScalar("accuracy_either", -1);
    const auto pred_max_prob = result.GetScalar("pred_max_prob", -1);
    const auto same_class_pair_ratio = result.GetScalar("same_class_pair_ratio", -1);

    REQUIRE(loss.has_value());
    REQUIRE(accuracy.has_value());
    REQUIRE(target_prob_mix_norm.has_value());
    REQUIRE(accuracy_either.has_value());
    REQUIRE(pred_max_prob.has_value());
    REQUIRE(same_class_pair_ratio.has_value());
    CHECK(std::isnan(*loss));
    CHECK(std::isnan(*accuracy));
    CHECK(std::isnan(*target_prob_mix_norm));
    CHECK(std::isnan(*accuracy_either));
    CHECK(std::isnan(*pred_max_prob));
    CHECK(std::isnan(*same_class_pair_ratio));
    CHECK_FALSE(result.GetScalar("unknown", -1).has_value());
}

TEST_CASE("ImageClsLearner reports normalized diagnostic metrics when mixup is disabled", "[image_cls][mixup]")
{
    auto config_data = MakeImageClsMixTestConfigData();
    config_data.Set("ImageClsAgent.mixup.enabled", false);
    config_data.Set("ImageClsAgent.label_smoothing", 0.1);
    auto config = MakeImageClsMixTestConfig(config_data);

    auto fixture = MakeImageClsRecordingTestNetwork(torch::kUInt8);
    auto grid = torch::tensor(
        { 255, 0, 0, 0,
          0, 255, 0, 0 },
        torch::TensorOptions().dtype(torch::kUInt8)).view({ 2, 1, 2, 2 });
    auto labels = torch::tensor({ 0, 1 }, torch::TensorOptions().dtype(torch::kInt64));

    auto result = RunImageClsRecordingUpdate(
        fixture,
        config,
        anet::rl::StepCounts{},
        MakeImageClsLearningExperience(grid, labels),
        7);

    auto expected_input = grid.to(torch::kFloat32).div(255.0f);
    RequireTensorClose(fixture.recorder->last_input, expected_input);
    ExpectedMixResult no_mix;
    no_mix.targets_b = labels;
    no_mix.lambda = 1.0;
    const auto expected = MakeExpectedImageClsMetrics(expected_input, labels, no_mix, config.label_smoothing, false);
    CheckImageClsScalar(result, "accuracy", 1.0f);
    CheckImageClsScalar(result, "target_prob_mix_norm", expected.target_prob_mix_norm);
    CheckImageClsScalar(result, "accuracy_either", expected.accuracy_either);
    CheckImageClsScalar(result, "pred_max_prob", expected.pred_max_prob);
    CheckImageClsScalar(result, "same_class_pair_ratio", 0.0f);
    CHECK_FALSE(result->GetScalar("target_prob_mix", -1).has_value());
    CHECK_FALSE(result->GetScalar("unknown", -1).has_value());
}

TEST_CASE("ImageClsLearner bypasses mixup for small batches and prob zero", "[image_cls][mixup]")
{
    SECTION("batch size smaller than two")
    {
        auto config = MakeImageClsMixTestConfig(MakeImageClsMixTestConfigData());
        auto fixture = MakeImageClsRecordingTestNetwork(torch::kUInt8);
        auto grid = torch::tensor(
            { 64, 128, 0, 255 },
            torch::TensorOptions().dtype(torch::kUInt8)).view({ 1, 1, 2, 2 });
        auto labels = torch::tensor({ 0 }, torch::TensorOptions().dtype(torch::kInt64));

        RunImageClsRecordingUpdate(
            fixture,
            config,
            anet::rl::StepCounts{},
            MakeImageClsLearningExperience(grid, labels),
            11);

        RequireTensorClose(fixture.recorder->last_input, grid.to(torch::kFloat32).div(255.0f));
    }

    SECTION("probability zero")
    {
        auto config_data = MakeImageClsMixTestConfigData();
        config_data.Set("ImageClsAgent.mixup.prob", 0.0);
        auto config = MakeImageClsMixTestConfig(config_data);
        auto fixture = MakeImageClsRecordingTestNetwork(torch::kUInt8);
        auto grid = torch::tensor(
            { 255, 0, 0, 0,
              0, 255, 0, 0 },
            torch::TensorOptions().dtype(torch::kUInt8)).view({ 2, 1, 2, 2 });
        auto labels = torch::tensor({ 0, 1 }, torch::TensorOptions().dtype(torch::kInt64));

        RunImageClsRecordingUpdate(
            fixture,
            config,
            anet::rl::StepCounts{},
            MakeImageClsLearningExperience(grid, labels),
            11);

        RequireTensorClose(fixture.recorder->last_input, grid.to(torch::kFloat32).div(255.0f));
    }
}

TEST_CASE("ImageClsAgent builds logits head from action spec", "[image_cls][head]")
{
    EnsureImageClsNnInitialized();
    ScopedNoopMetricsLogger metrics_logger;

    auto config = anet::rl::img_cls::ImageClsAgentConfig(MakeImageClsSerializeTestConfigData());
    auto env_spec = MakeImageClsEnvSpec();
    anet::rl::img_cls::ImageClsAgent agent(
        config,
        MakeImageClsAgentHeadNetworkConfig(),
        env_spec,
        anet::rl::BatchEnvSpec{ 2, 1 },
        torch::Device(torch::kCPU),
        123);

    auto probs = ForwardImageClsAgentProbs(
        agent,
        torch::arange(8, torch::kFloat32).view({ 2, 1, 2, 2 }).div(16.0f));
    CHECK(probs.sizes() == torch::IntArrayRef({ 2, env_spec.action_spec.GetNumActions() }));
    CHECK(probs.dtype() == torch::kFloat32);
}

TEST_CASE("ImageCls actor and learner gate BF16 autocast around forward", "[image_cls][bf16]")
{
    const auto grid = torch::arange(8, torch::kFloat32).view({ 2, 1, 2, 2 }).div(16.0f);
    const auto labels = torch::tensor({ 0, 1 }, torch::TensorOptions().dtype(torch::kInt64));

    SECTION("actor uses BF16 autocast only when enabled for actor")
    {
        auto fixture = MakeImageClsAutocastProbeTestNetwork();
        auto mutex = std::make_shared<std::shared_mutex>();
        auto config = anet::rl::img_cls::ImageClsAgentConfig{};
        config.bf16.enabled = true;
        config.bf16.actor = true;
        anet::rl::img_cls::ImageClsActor actor(
            config,
            mutex,
            fixture.network,
            anet::rl::RunMode::Eval,
            torch::Device(torch::kCPU));

        auto action_info = actor.MakeAction(anet::rl::StepCounts{}, MakeImageClsBatchState(grid));

        REQUIRE(fixture.probe->last_autocast_enabled);
        CHECK_FALSE(at::autocast::is_autocast_enabled(torch::kCPU));
        REQUIRE(action_info->GetInfo().Contains("probs"));
        REQUIRE(action_info->GetInfo().At("probs").dtype() == torch::kFloat32);
    }

    SECTION("actor leaves autocast disabled by default")
    {
        auto fixture = MakeImageClsAutocastProbeTestNetwork();
        auto mutex = std::make_shared<std::shared_mutex>();
        auto config = anet::rl::img_cls::ImageClsAgentConfig{};
        config.bf16.enabled = true;
        anet::rl::img_cls::ImageClsActor actor(
            config,
            mutex,
            fixture.network,
            anet::rl::RunMode::Eval,
            torch::Device(torch::kCPU));

        actor.MakeAction(anet::rl::StepCounts{}, MakeImageClsBatchState(grid));

        CHECK_FALSE(fixture.probe->last_autocast_enabled);
        CHECK_FALSE(at::autocast::is_autocast_enabled(torch::kCPU));
    }

    SECTION("learner uses BF16 autocast only when enabled for learner")
    {
        auto fixture = MakeImageClsAutocastProbeTestNetwork();
        auto mutex = std::make_shared<std::shared_mutex>();
        auto config = MakeImageClsLearningRateTestConfig();
        config.bf16.enabled = true;
        config.bf16.learner = true;
        auto learning_rate = std::make_shared<anet::ProfiledValue<double>>(config.learning_rate);
        anet::rl::img_cls::ImageClsLearner learner(
            config,
            mutex,
            fixture.network,
            learning_rate,
            torch::Device(torch::kCPU),
            123);
        anet::rl::StepCounts step;

        learner.UpdateFromBatch(step, MakeImageClsLearningExperience(grid, labels));

        REQUIRE(fixture.probe->last_autocast_enabled);
        CHECK_FALSE(at::autocast::is_autocast_enabled(torch::kCPU));
    }

    SECTION("learner leaves autocast disabled when globally disabled")
    {
        auto fixture = MakeImageClsAutocastProbeTestNetwork();
        auto mutex = std::make_shared<std::shared_mutex>();
        auto config = MakeImageClsLearningRateTestConfig();
        auto learning_rate = std::make_shared<anet::ProfiledValue<double>>(config.learning_rate);
        anet::rl::img_cls::ImageClsLearner learner(
            config,
            mutex,
            fixture.network,
            learning_rate,
            torch::Device(torch::kCPU),
            123);
        anet::rl::StepCounts step;

        learner.UpdateFromBatch(step, MakeImageClsLearningExperience(grid, labels));

        CHECK_FALSE(fixture.probe->last_autocast_enabled);
        CHECK_FALSE(at::autocast::is_autocast_enabled(torch::kCPU));
    }
}

TEST_CASE("ImageClsLearner applies deterministic Mixup before network forward", "[image_cls][mixup]")
{
    constexpr anet::seed_t seed = 42;
    auto config = MakeImageClsMixTestConfig(MakeImageClsMixTestConfigData());
    auto fixture = MakeImageClsRecordingTestNetwork(torch::kUInt8);
    auto grid = torch::tensor(
        { 255, 0, 0, 0,
          0, 255, 0, 0,
          128, 64, 0, 0 },
        torch::TensorOptions().dtype(torch::kUInt8)).view({ 3, 1, 2, 2 });
    auto labels = torch::tensor({ 0, 1, 0 }, torch::TensorOptions().dtype(torch::kInt64));

    auto expected = MakeExpectedMixupResult(config, /*batch_size=*/3, labels, seed);
    CHECK(expected.lambda > 0.0);
    CHECK(expected.lambda < 1.0);
    auto expected_grid = MakeExpectedMixupGrid(grid, expected.perm, expected.lambda);
    auto expected_input = expected_grid.to(torch::kFloat32).div(255.0f);

    auto result = RunImageClsRecordingUpdate(
        fixture,
        config,
        anet::rl::StepCounts{},
        MakeImageClsLearningExperience(grid, labels),
        seed);

    RequireTensorClose(fixture.recorder->last_input, expected_input);
    const auto expected_metrics = MakeExpectedImageClsMetrics(
        expected_input, labels, expected, config.label_smoothing, true);
    CheckImageClsScalar(result, "target_prob_mix_norm", expected_metrics.target_prob_mix_norm);
    CheckImageClsScalar(result, "accuracy_either", expected_metrics.accuracy_either);
    CheckImageClsScalar(result, "pred_max_prob", expected_metrics.pred_max_prob);
    CheckImageClsScalar(result, "same_class_pair_ratio", expected_metrics.same_class_pair_ratio);
}

TEST_CASE("ImageClsLearner applies deterministic CutMix patch and corrected lambda", "[image_cls][mixup]")
{
    constexpr anet::seed_t seed = 99;
    constexpr int64_t height = 4;
    constexpr int64_t width = 4;
    auto config_data = MakeImageClsMixTestConfigData();
    config_data.Set("ImageClsAgent.mixup.cutmix_alpha", 10.0);
    config_data.Set("ImageClsAgent.mixup.switch_prob", 1.0);
    auto config = MakeImageClsMixTestConfig(config_data);
    auto fixture = MakeImageClsRecordingTestNetwork(torch::kUInt8, height, width);
    auto grid = torch::arange(3 * height * width, torch::TensorOptions().dtype(torch::kFloat32))
        .mul(5.0f)
        .remainder(256.0f)
        .to(torch::kUInt8)
        .view({ 3, 1, height, width });
    auto labels = torch::tensor({ 0, 1, 0 }, torch::TensorOptions().dtype(torch::kInt64));

    auto expected = MakeExpectedCutMixResult(config, /*batch_size=*/3, height, width, labels, seed);
    const int64_t patch_area = (expected.x2 - expected.x1) * (expected.y2 - expected.y1);
    REQUIRE(patch_area > 0);
    CHECK(expected.lambda == Catch::Approx(1.0 - static_cast<double>(patch_area) / static_cast<double>(height * width)));
    auto expected_grid = MakeExpectedCutMixGrid(grid, expected);
    auto expected_input = expected_grid.to(torch::kFloat32).div(255.0f);

    auto result = RunImageClsRecordingUpdate(
        fixture,
        config,
        anet::rl::StepCounts{},
        MakeImageClsLearningExperience(grid, labels),
        seed);

    RequireTensorClose(fixture.recorder->last_input, expected_input);
    const auto expected_metrics = MakeExpectedImageClsMetrics(
        expected_input, labels, expected, config.label_smoothing, true);
    CheckImageClsScalar(result, "target_prob_mix_norm", expected_metrics.target_prob_mix_norm);
    CheckImageClsScalar(result, "accuracy_either", expected_metrics.accuracy_either);
    CheckImageClsScalar(result, "pred_max_prob", expected_metrics.pred_max_prob);
    CheckImageClsScalar(result, "same_class_pair_ratio", expected_metrics.same_class_pair_ratio);
}

TEST_CASE("ImageClsLearner mixup seed is reproducible and isolated from global torch RNG", "[image_cls][mixup]")
{
    constexpr anet::seed_t seed = 1234;
    auto config = MakeImageClsMixTestConfig(MakeImageClsMixTestConfigData());
    auto grid = torch::tensor(
        { 255, 0, 0, 0,
          0, 255, 0, 0,
          128, 64, 0, 0 },
        torch::TensorOptions().dtype(torch::kUInt8)).view({ 3, 1, 2, 2 });
    auto labels = torch::tensor({ 0, 1, 0 }, torch::TensorOptions().dtype(torch::kInt64));

    auto fixture_a = MakeImageClsRecordingTestNetwork(torch::kUInt8);
    torch::manual_seed(999);
    auto result_a = RunImageClsRecordingUpdate(
        fixture_a,
        config,
        anet::rl::StepCounts{},
        MakeImageClsLearningExperience(grid, labels),
        seed);

    auto fixture_b = MakeImageClsRecordingTestNetwork(torch::kUInt8);
    torch::manual_seed(123);
    auto result_b = RunImageClsRecordingUpdate(
        fixture_b,
        config,
        anet::rl::StepCounts{},
        MakeImageClsLearningExperience(grid, labels),
        seed);

    RequireTensorClose(fixture_a.recorder->last_input, fixture_b.recorder->last_input);
    const float loss_a = RequireImageClsScalar(result_a, "loss");
    const float accuracy_a = RequireImageClsScalar(result_a, "accuracy");
    const float target_prob_mix_norm_a = RequireImageClsScalar(result_a, "target_prob_mix_norm");
    CHECK(loss_a == Catch::Approx(RequireImageClsScalar(result_b, "loss")));
    CHECK(accuracy_a == Catch::Approx(RequireImageClsScalar(result_b, "accuracy")));
    CHECK(target_prob_mix_norm_a == Catch::Approx(RequireImageClsScalar(result_b, "target_prob_mix_norm")));
}

TEST_CASE("ImageClsAgent derives learner mixup seed from agent seed", "[image_cls][mixup]")
{
    EnsureImageClsNnInitialized();
    ScopedNoopMetricsLogger metrics_logger;

    auto grid = torch::tensor(
        { 255, 0, 0, 0,
          0, 255, 0, 0,
          128, 64, 0, 0 },
        torch::TensorOptions().dtype(torch::kUInt8)).view({ 3, 1, 2, 2 });
    auto labels = torch::tensor({ 0, 1, 0 }, torch::TensorOptions().dtype(torch::kInt64));

    auto result_a = RunImageClsAgentMixUpdate(/*agent_seed=*/1234, grid, labels);
    auto result_b = RunImageClsAgentMixUpdate(/*agent_seed=*/1234, grid, labels);
    const float loss_a = RequireImageClsScalar(result_a, "loss");
    const float accuracy_a = RequireImageClsScalar(result_a, "accuracy");
    const float target_prob_mix_norm_a = RequireImageClsScalar(result_a, "target_prob_mix_norm");
    CHECK(loss_a == Catch::Approx(RequireImageClsScalar(result_b, "loss")));
    CHECK(accuracy_a == Catch::Approx(RequireImageClsScalar(result_b, "accuracy")));
    CHECK(target_prob_mix_norm_a == Catch::Approx(RequireImageClsScalar(result_b, "target_prob_mix_norm")));

    auto result_c = RunImageClsAgentMixUpdate(/*agent_seed=*/4321, grid, labels);
    const float loss_c = RequireImageClsScalar(result_c, "loss");
    const float target_prob_mix_norm_c = RequireImageClsScalar(result_c, "target_prob_mix_norm");
    const bool changed =
        std::abs(loss_a - loss_c) > 1e-6f ||
        std::abs(target_prob_mix_norm_a - target_prob_mix_norm_c) > 1e-6f;
    CHECK(changed);
}

TEST_CASE("ImageClsLearner verbose log follows learn_log_interval", "[image_cls][mixup]")
{
    auto config_data = MakeImageClsMixTestConfigData();
    config_data.Set("ImageClsAgent.mixup.enabled", false);
    config_data.Set("ImageClsAgent.learn_log_interval", 2);
    auto config = MakeImageClsMixTestConfig(config_data);
    auto fixture = MakeImageClsRecordingTestNetwork(torch::kUInt8);
    auto mutex = std::make_shared<std::shared_mutex>();
    auto learning_rate = std::make_shared<anet::ProfiledValue<double>>(config.learning_rate);
    anet::rl::img_cls::ImageClsLearner learner(
        config,
        mutex,
        fixture.network,
        learning_rate,
        torch::Device(torch::kCPU),
        5);

    auto grid = torch::tensor(
        { 255, 0, 0, 0,
          0, 255, 0, 0 },
        torch::TensorOptions().dtype(torch::kUInt8)).view({ 2, 1, 2, 2 });
    auto labels = torch::tensor({ 0, 1 }, torch::TensorOptions().dtype(torch::kInt64));

    anet::rl::StepCounts step;
    step.learn_step = 1;
    {
        anet::test::LogCaptureGuard logs(wxLOG_Info);
        learner.UpdateFromBatch(step, MakeImageClsLearningExperience(grid, labels));
        logs.Flush();
        CHECK_FALSE(anet::test::HasRecordContaining(logs.Records(), wxLOG_Info, { "ImageClsLearner update" }));
    }

    step.learn_step = 2;
    {
        anet::test::LogCaptureGuard logs(wxLOG_Info);
        learner.UpdateFromBatch(step, MakeImageClsLearningExperience(grid, labels));
        logs.Flush();
        CHECK(anet::test::HasRecordContaining(
            logs.Records(),
            wxLOG_Info,
            {
                "ImageClsLearner update",
                "learn_step=2",
                "target_prob_mix_norm=",
                "accuracy_either=",
                "pred_max_prob=",
                "mix_mode=none"
            }));
    }

    config_data.Set("ImageClsAgent.learn_log_interval", 0);
    auto disabled_config = MakeImageClsMixTestConfig(config_data);
    auto disabled_fixture = MakeImageClsRecordingTestNetwork(torch::kUInt8);
    auto disabled_lr = std::make_shared<anet::ProfiledValue<double>>(disabled_config.learning_rate);
    anet::rl::img_cls::ImageClsLearner disabled_learner(
        disabled_config,
        std::make_shared<std::shared_mutex>(),
        disabled_fixture.network,
        disabled_lr,
        torch::Device(torch::kCPU),
        5);
    step.learn_step = 0;
    {
        anet::test::LogCaptureGuard logs(wxLOG_Info);
        disabled_learner.UpdateFromBatch(step, MakeImageClsLearningExperience(grid, labels));
        logs.Flush();
        CHECK_FALSE(anet::test::HasRecordContaining(logs.Records(), wxLOG_Info, { "ImageClsLearner update" }));
    }
}

TEST_CASE("ImageClsAgent checkpoint restores network and optimizer state", "[image_cls][serialize]")
{
    EnsureImageClsNnInitialized();
    ScopedNoopMetricsLogger metrics_logger;

    torch::manual_seed(20240705);
    auto config = anet::rl::img_cls::ImageClsAgentConfig(MakeImageClsSerializeTestConfigData());
    auto env_spec = MakeImageClsEnvSpec();
    anet::rl::img_cls::ImageClsAgent saved_agent(
        config,
        MakeImageClsAgentNetworkConfig(),
        env_spec,
        anet::rl::BatchEnvSpec{ 2, 1 },
        torch::Device(torch::kCPU),
        123);

    // 1 update 済みの network/optimizer を checkpoint 化する
    auto saved_learner = saved_agent.CreateLearner();
    anet::rl::StepCounts first_step;
    first_step.exp_step = 1;
    first_step.learn_step = 1;
    saved_learner->UpdateFromBatch(first_step, MakeImageClsLearningExperience());

    const auto probe_grid = torch::arange(8, torch::kFloat32).view({ 2, 1, 2, 2 }).div(16.0f);
    const auto saved_probs = ForwardImageClsAgentProbs(saved_agent, probe_grid);

    const auto checkpoint_path = MakeImageClsCheckpointPath("roundtrip.anet");
    CHECK(SaveImageClsAgent(saved_agent, checkpoint_path) > 0);

    // 別 seed で作った Agent へ auto_load し、初期重みでなく checkpoint の重みに差し替わることを見る
    auto load_config_data = MakeImageClsSerializeTestConfigData();
    load_config_data.Set("ImageClsAgent.auto_load_file", checkpoint_path.string());
    torch::manual_seed(999);
    anet::rl::img_cls::ImageClsAgent loaded_agent(
        anet::rl::img_cls::ImageClsAgentConfig(load_config_data),
        MakeImageClsAgentNetworkConfig(),
        env_spec,
        anet::rl::BatchEnvSpec{ 2, 1 },
        torch::Device(torch::kCPU),
        999);

    RequireTensorClose(ForwardImageClsAgentProbs(loaded_agent, probe_grid), saved_probs);

    // 同じ追加 update をかけても一致することで、optimizer state も復元されていることを確認する
    anet::rl::StepCounts second_step;
    second_step.exp_step = 2;
    second_step.learn_step = 2;
    saved_learner->UpdateFromBatch(second_step, MakeImageClsLearningExperience());
    loaded_agent.CreateLearner()->UpdateFromBatch(second_step, MakeImageClsLearningExperience());

    RequireTensorClose(
        ForwardImageClsAgentProbs(loaded_agent, probe_grid),
        ForwardImageClsAgentProbs(saved_agent, probe_grid));
}

TEST_CASE("ImageClsAgent fused optimizer loads AdamW checkpoint", "[image_cls][serialize][optimizer]")
{
    EnsureImageClsNnInitialized();
    ScopedNoopMetricsLogger metrics_logger;

    auto save_config_data = MakeImageClsSerializeTestConfigData();
    save_config_data.Set("ImageClsAgent.use_fused_optimizer", false);
    torch::manual_seed(20240706);
    auto env_spec = MakeImageClsEnvSpec();
    anet::rl::img_cls::ImageClsAgent saved_agent(
        anet::rl::img_cls::ImageClsAgentConfig(save_config_data),
        MakeImageClsAgentNetworkConfig(),
        env_spec,
        anet::rl::BatchEnvSpec{ 2, 1 },
        torch::Device(torch::kCPU),
        123);

    // 従来AdamWで1 update済みの checkpoint を作る
    auto saved_learner = saved_agent.CreateLearner();
    anet::rl::StepCounts first_step;
    first_step.exp_step = 1;
    first_step.learn_step = 1;
    saved_learner->UpdateFromBatch(first_step, MakeImageClsLearningExperience());

    const auto probe_grid = torch::arange(8, torch::kFloat32).view({ 2, 1, 2, 2 }).div(16.0f);
    const auto saved_probs = ForwardImageClsAgentProbs(saved_agent, probe_grid);

    const auto checkpoint_path = MakeImageClsCheckpointPath("adamw_to_fused.anet");
    CHECK(SaveImageClsAgent(saved_agent, checkpoint_path) > 0);

    // default fused optimizer の Agent で、従来AdamWの optimizer state を読み込めることを見る
    auto load_config_data = MakeImageClsSerializeTestConfigData();
    load_config_data.Set("ImageClsAgent.auto_load_file", checkpoint_path.string());
    torch::manual_seed(999);
    anet::rl::img_cls::ImageClsAgent loaded_agent(
        anet::rl::img_cls::ImageClsAgentConfig(load_config_data),
        MakeImageClsAgentNetworkConfig(),
        env_spec,
        anet::rl::BatchEnvSpec{ 2, 1 },
        torch::Device(torch::kCPU),
        999);

    RequireTensorClose(ForwardImageClsAgentProbs(loaded_agent, probe_grid), saved_probs);

    // 同じ追加 update をかけ、FusedAdamW側で復元済み state が使われることを確認する
    anet::rl::StepCounts second_step;
    second_step.exp_step = 2;
    second_step.learn_step = 2;
    saved_learner->UpdateFromBatch(second_step, MakeImageClsLearningExperience());
    loaded_agent.CreateLearner()->UpdateFromBatch(second_step, MakeImageClsLearningExperience());

    RequireTensorClose(
        ForwardImageClsAgentProbs(loaded_agent, probe_grid),
        ForwardImageClsAgentProbs(saved_agent, probe_grid),
        1e-4);
}

TEST_CASE("ImageClsAgent auto-load rejects checkpoints from other agent types", "[image_cls][serialize]")
{
    EnsureImageClsNnInitialized();
    ScopedNoopMetricsLogger metrics_logger;

    const auto checkpoint_path = MakeImageClsCheckpointPath("wrong_agent_header.anet");
    WriteWrongImageClsHeaderArchive(checkpoint_path);

    auto config_data = MakeImageClsSerializeTestConfigData();
    config_data.Set("ImageClsAgent.auto_load_file", checkpoint_path.string());
    auto env_spec = MakeImageClsEnvSpec();
    auto construct_agent = [&]() {
        anet::rl::img_cls::ImageClsAgent agent(
            anet::rl::img_cls::ImageClsAgentConfig(config_data),
            MakeImageClsAgentNetworkConfig(),
            env_spec,
            anet::rl::BatchEnvSpec{ 2, 1 },
            torch::Device(torch::kCPU),
            123);
    };

    CHECK_THROWS(construct_agent());
}

TEST_CASE("ImageClsAgent cloned actor stays isolated until Sync", "[image_cls][actor]")
{
    EnsureImageClsNnInitialized();
    ScopedNoopMetricsLogger metrics_logger;

    auto config_data = MakeImageClsSerializeTestConfigData();
    config_data.Set("ImageClsAgent.learning_rate.value", 0.1);
    auto env_spec = MakeImageClsEnvSpec();
    torch::manual_seed(20240706);
    anet::rl::img_cls::ImageClsAgent agent(
        anet::rl::img_cls::ImageClsAgentConfig(config_data),
        MakeImageClsAgentNetworkConfig(),
        env_spec,
        anet::rl::BatchEnvSpec{ 2, 1 },
        torch::Device(torch::kCPU),
        123);

    const anet::rl::BatchEnvSpec batch_spec{ 2, 1 };
    auto shared_actor = agent.CreateActor(
        batch_spec,
        env_spec,
        anet::rl::RunMode::Eval,
        std::nullopt,
        torch::Device(torch::kCPU));
    auto cloned_actor = agent.CreateActor(
        batch_spec,
        env_spec,
        anet::rl::RunMode::Eval,
        true,
        torch::Device(torch::kCPU));

    const auto probe_grid = torch::arange(8, torch::kFloat32).view({ 2, 1, 2, 2 }).div(16.0f);
    const auto initial_shared = ForwardImageClsActorProbs(shared_actor, probe_grid);
    const auto initial_cloned = ForwardImageClsActorProbs(cloned_actor, probe_grid);
    RequireTensorClose(initial_cloned, initial_shared);

    auto learner = agent.CreateLearner();
    for (int64_t i = 1; i <= 3; ++i) {
        anet::rl::StepCounts step;
        step.exp_step = i;
        step.learn_step = i;
        learner->UpdateFromBatch(step, MakeImageClsLearningExperience());
    }

    const auto updated_shared = ForwardImageClsActorProbs(shared_actor, probe_grid);
    CHECK_FALSE(torch::allclose(updated_shared, initial_shared, 1e-5, 1e-5));

    // Clone actor は Sync まで作成時点の network を使い続ける
    const auto stale_cloned = ForwardImageClsActorProbs(cloned_actor, probe_grid);
    RequireTensorClose(stale_cloned, initial_cloned);

    cloned_actor->Sync();
    const auto synced_cloned = ForwardImageClsActorProbs(cloned_actor, probe_grid);
    RequireTensorClose(synced_cloned, updated_shared);
    RequireTensorClose(synced_cloned, ForwardImageClsAgentProbs(agent, probe_grid));
}

TEST_CASE("ImageClsActor stores nn trace in action aux for Conv2dPanel", "[image_cls][trace]")
{
    auto network = MakeImageClsTraceTestNetwork();
    auto mutex = std::make_shared<std::shared_mutex>();
    anet::rl::img_cls::ImageClsActor actor(
        anet::rl::img_cls::ImageClsAgentConfig{},
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
