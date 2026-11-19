#include "anet/test/catch.hpp"
#include <algorithm>
#include "default_dqn_agent.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/nn_util.hpp"
#include "anet/test/util.hpp"
#include "dqn_based_heads.hpp"
#include "anet/test/nn.hpp"
#include "anet/nn_head.hpp"
#include <ATen/autocast_mode.h>
#include <torch/csrc/autograd/profiler_legacy.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>


static anet::ConfigData SelectForwardConfigData(const anet::ConfigData& data, const std::string& prefix)
{
    anet::ConfigData selected;
    for (const auto& [key, value] : data.Map()) {
        if (key.starts_with(prefix + ".")) selected.Set(key.substr(prefix.size() + 1), value);
    }
    return selected;
}

static anet::ConfigData WithDefaultForwardConfig(anet::ConfigData data = {})
{
    anet::test::RegisterStandardNetworkModules();
    // 既存テストの実行目的を明示し、新しい必須参照を fixture 側で選択する。
    data.Set("DefaultDQNAgent.nn_forward.[test_default].noisy.mode", "mu_only");
    if (!data.Has("DefaultDQNAgent.learner.current.nn_forward_key"))
        data.Set("DefaultDQNAgent.learner.current.nn_forward_key", "test_default");
    if (!data.Has("DefaultDQNAgent.learner.target.nn_forward_key"))
        data.Set("DefaultDQNAgent.learner.target.nn_forward_key", "test_default");
    for (const auto& [key, entry] : data.MakeSubConfigData("DefaultDQNAgent.actor")) {
        const auto reference = "DefaultDQNAgent.actor.[" + key + "].nn_forward_key";
        if (!data.Has(reference)) data.Set(reference, "test_default");
    }
    return data;
}

namespace dqn = anet::rl::dqn;

static bool Contains(const std::string& text, const std::string& pattern)
{
    return text.find(pattern) != std::string::npos;
}

static void EnsureNNInitialized()
{
    static const bool initialized = [] {

        anet::test::RegisterStandardNetworkModules();
        return true;
    }();
    (void)initialized;
}

namespace {
    class DotTestModule final : public anet::nn::NetworkModule {
    public:
        DotTestModule()
        {
            weight_ = register_parameter("weight", torch::ones({ 2, 3 }));
        }
    
        torch::Tensor Forward(torch::Tensor input, const anet::nn::NetworkForwardContext& context) override
        {
            return input;
        }
    
        anet::ConfigData GetCurrentConfigData() const override
        {
            anet::ConfigData cd;
            cd.Set("scale", 1.23456);
            cd.Set("activation_mode", "pre");
            cd.Set("mode", "test");
            return cd;
        }
    
    private:
        torch::Tensor weight_;
    };
}

namespace {
    class DotTestHead final : public anet::nn::NetworkHead {
    public:
        anet::TensorDict Forward(const anet::TensorDict& feature_dict, const anet::nn::NetworkForwardContext& context) override
        {
            return feature_dict;
        }
    };
}

static std::shared_ptr<anet::nn::Network> MakeDotTestNetwork(
    std::shared_ptr<anet::nn::NetworkHead> head = nullptr,
    const std::string& head_key = "feature")
{
    anet::TensorSpec obs_spec;
    obs_spec.type = anet::SpaceType::Vector;
    obs_spec.shape = { 2 };
    obs_spec.dtype = torch::kFloat32;

    anet::TensorSpecMap input_specs;
    input_specs["obs"] = obs_spec;

    anet::nn::NetworkConfig network_config;
    anet::nn::NetworkBranchConfig branch_config;
    branch_config.name = "feature";
    branch_config.bind_terms = { { "obs" } };
    branch_config.raw_keys = { "obs" };
    branch_config.auto_format = false;
    branch_config.structure_str = "Scale_0";
    network_config.branches["feature"] = branch_config;
    network_config.output_keys[head_key] = "feature";

    return anet::test::MakeTestNetwork(network_config, input_specs, nullptr,
        head ? head : std::make_shared<DotTestHead>(),
        { { "feature", { { "obs" } }, 1, "Scale_0", std::make_shared<DotTestModule>() } });
}

static std::shared_ptr<anet::nn::NetworkHead> CreateDqnHead(const anet::nn::NetworkHeadFactory& factory)
{
    anet::TensorDict dummy_features;
    dummy_features.Set(anet::nn::kKey_DefaultOutput, torch::zeros({ 1, 4 }));
    return factory.CreateHead(dummy_features);
}

TEST_CASE("Network dot view emits head outputs and optional head info", "[nn][dot][head]")
{
    anet::nn::WeightInitConfig init_config;

    dqn::DuelingHeadFactory dueling_factory(3, init_config);
    auto dueling_network = MakeDotTestNetwork(
        CreateDqnHead(dueling_factory),
        anet::nn::kKey_DefaultOutput);

    const auto structure_dot = dueling_network->MakeGraphViz(anet::nn::NetworkGraphVizConfig{})->ToDotString();
    CHECK(Contains(structure_dot, "<B>DuelingHead</B>"));
    CHECK(Contains(structure_dot, "\"head\" -> \"head_output_q\""));
    CHECK(Contains(structure_dot, "\"head\" -> \"head_output_v\""));
    CHECK(Contains(structure_dot, "\"head\" -> \"head_output_a\""));
    CHECK(Contains(structure_dot, "\"head_output_q\" [shape=ellipse"));
    CHECK(Contains(structure_dot, "color=\"#C0392B\""));
    CHECK(Contains(structure_dot, "fillcolor=\"#FDEDEC\""));
    CHECK(Contains(structure_dot, "output: q"));
    CHECK_FALSE(Contains(structure_dot, ">outputs</TD>"));
    CHECK_FALSE(Contains(structure_dot, ">shape</TD>"));
    CHECK_FALSE(Contains(structure_dot, "streams"));

    dqn::QuantileDuelingHeadFactory quantile_dueling_factory(3, 5, init_config);
    auto quantile_dueling_network = MakeDotTestNetwork(
        CreateDqnHead(quantile_dueling_factory),
        anet::nn::kKey_DefaultOutput);

    anet::nn::NetworkGraphVizConfig detail_config;
    detail_config.show_head_info = true;
    const auto detail_dot = quantile_dueling_network->MakeGraphViz(detail_config)->ToDotString();
    CHECK(Contains(detail_dot, "<B>QuantileDuelingHead</B>"));
    CHECK(Contains(detail_dot, "\"head\" -> \"head_output_q_dist\""));
    CHECK(Contains(detail_dot, "output: q_dist"));
    CHECK(Contains(detail_dot, ">shape</TD>"));
    CHECK(Contains(detail_dot, "[3, 5]"));
    CHECK(Contains(detail_dot, "num_quantiles"));
}

TEST_CASE("Agent configs read nn_viz keys, ignore old keys, and reject invalid layout", "[nn][dot][config]")
{
    anet::ConfigData dqn_data;
    dqn_data.Set("DefaultDQNAgent.nn_viz.show_param_shapes", "true");
    dqn_data.Set("DefaultDQNAgent.nn_viz.show_branch_config", "true");
    dqn_data.Set("DefaultDQNAgent.nn_viz.show_head_info", "true");
    dqn_data.Set("DefaultDQNAgent.nn_viz.float_precision", "5");
    dqn_data.Set("DefaultDQNAgent.nn_dot.show_param_count", "true");
    dqn_data.Set("DefaultDQNAgent.dot.show_param_count", "true");

    dqn::DefaultDQNAgentConfig dqn_config(WithDefaultForwardConfig(dqn_data));
    CHECK(dqn_config.nn_viz.show_param_shapes);
    CHECK(dqn_config.nn_viz.show_branch_config);
    CHECK(dqn_config.nn_viz.show_head_info);
    CHECK_FALSE(dqn_config.nn_viz.show_param_count);
    CHECK(dqn_config.nn_viz.float_precision == 5);

    anet::ConfigData invalid_data;
    invalid_data.Set("DefaultDQNAgent.nn_viz.layout", "BT");
    CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(invalid_data)));
}

TEST_CASE("Network enumerates paired Noisy parameters with their owning branches", "[nn][noisynet][parameters]")
{
    EnsureNNInitialized();
    const bool enabled = GENERATE(false, true);
    anet::ConfigData data;
    data.Set("net.block.[Feature].type", "Linear");
    data.Set("net.block.[Feature].linear.out_features", 4);
    data.Set("net.block.[Feature].noisy.enabled", enabled);
    data.Set("net.block.[Readout].type", "Linear");
    data.Set("net.block.[Readout].linear.out_features", 3);
    data.Set("net.block.[Readout].linear.bias", false);
    data.Set("net.block.[Readout].noisy.enabled", enabled);
    data.Set("net.branch.[feature].bind", "obs");
    data.Set("net.branch.[feature].structure", "Feature");
    data.Set("net.branch.[readout].bind", "feature");
    data.Set("net.branch.[readout].structure", "Readout");
    data.Set("net.body.output.[features]", "readout");
    anet::ConfigData projection;
    projection.Set("noisy.enabled", enabled);
    const auto network = anet::nn::NetworkBuilder::BuildNetwork(anet::nn::NetworkConfig(data),
        anet::TensorSpecMap{{"obs", anet::TensorSpec{.type=anet::SpaceType::Vector, .shape={2}, .dtype=torch::kFloat32}}},
        std::make_shared<dqn::LinearHeadFactory>(2, anet::nn::WeightInitConfig{}, projection), 8441, torch::kCPU);
    const auto entries = network->GetNoisyParameterEntries();
    REQUIRE(entries.size() == (enabled ? 5 : 0));
    std::map<std::string, int> by_branch;
    const auto parameters = network->parameters();
    for (const auto& entry : entries) {
        ++by_branch[entry.branch];
        CHECK(entry.mean.sizes() == entry.sigma.sizes());
        CHECK(entry.mean.dim() == (entry.is_bias ? 1 : 2));
        CHECK(entry.mean.unsafeGetTensorImpl() != entry.sigma.unsafeGetTensorImpl());
        CHECK(std::ranges::count_if(parameters, [&](const auto& value) {
            return value.unsafeGetTensorImpl() == entry.mean.unsafeGetTensorImpl();
        }) == 1);
        CHECK(std::ranges::count_if(parameters, [&](const auto& value) {
            return value.unsafeGetTensorImpl() == entry.sigma.unsafeGetTensorImpl();
        }) == 1);
    }
    if (enabled) {
        CHECK(by_branch["feature"] == 2);
        CHECK(by_branch["readout"] == 1);
        CHECK(by_branch[""] == 2);
    }
}

TEST_CASE("Existing parameter norms exclude Noisy sigma independently of its magnitude", "[nn][noisynet][weight_norm]")
{
    EnsureNNInitialized();
    anet::ConfigData data;
    data.Set("net.block.[Linear].type", "Linear");
    data.Set("net.block.[Linear].linear.out_features", 3);
    data.Set("net.block.[Linear].noisy.enabled", true);
    data.Set("net.branch.[feature].bind", "obs");
    data.Set("net.branch.[feature].structure", "Linear");
    data.Set("net.body.output.[features]", "feature");
    anet::ConfigData projection;
    projection.Set("noisy.enabled", true);
    auto network = anet::nn::NetworkBuilder::BuildNetwork(anet::nn::NetworkConfig(data),
        anet::TensorSpecMap{{"obs", anet::TensorSpec{.type=anet::SpaceType::Vector, .shape={2}, .dtype=torch::kFloat32}}},
        std::make_shared<dqn::LinearHeadFactory>(2, anet::nn::WeightInitConfig{}, projection), 8451, torch::kCPU);
    torch::NoGradGuard guard;
    auto expected_feature = torch::zeros({});
    auto expected_readout = torch::zeros({});
    for (const auto& entry : network->GetNoisyParameterEntries()) {
        entry.sigma.fill_(100.0);
        (entry.branch.empty() ? expected_readout : expected_feature).add_(entry.mean.square().sum());
    }
    const auto norms = network->ComputeParameterNormSplit("feature");
    CHECK(torch::allclose(norms.feature, expected_feature.sqrt()));
    CHECK(torch::allclose(norms.readout, expected_readout.sqrt()));
    CHECK(torch::equal(norms.feature, norms.feature_effective));
    CHECK(torch::equal(norms.readout, norms.readout_effective));
}

TEST_CASE("Noisy statistics use paired means and effective spectral weights only", "[nn][noisynet][statistics]")
{
    EnsureNNInitialized();
    const bool noisy_feature = GENERATE(false, true);
    const auto norm_mode = GENERATE(std::string("none"), std::string("spectral"), std::string("spectral_cap"));
    const bool zero_mean = GENERATE(false, true);
    if (zero_mean && norm_mode != "none") return;
    anet::ConfigData data;
    data.Set("net.block.[Feature].type", "Linear");
    data.Set("net.block.[Feature].linear.out_features", 3);
    data.Set("net.block.[Feature].init.mode", "constant");
    data.Set("net.block.[Feature].init.constant_val", 3.0f);
    data.Set("net.block.[Feature].weight_norm.mode", norm_mode);
    data.Set("net.block.[Feature].noisy.enabled", noisy_feature);
    data.Set("net.block.[Plain].type", "Linear");
    data.Set("net.block.[Plain].linear.out_features", 4);
    data.Set("net.block.[Plain].init.mode", "constant");
    data.Set("net.block.[Plain].init.constant_val", 1000.0f);
    data.Set("net.branch.[feature].bind", "obs");
    data.Set("net.branch.[feature].structure", "Feature");
    data.Set("net.branch.[readout].bind", "feature");
    data.Set("net.branch.[readout].structure", "Plain");
    data.Set("net.body.output.[features]", "readout");
    anet::ConfigData projection;
    projection.Set("noisy.enabled", true);
    auto network = anet::nn::NetworkBuilder::BuildNetwork(anet::nn::NetworkConfig(data),
        anet::TensorSpecMap{{"obs", anet::TensorSpec{.type=anet::SpaceType::Vector, .shape={2}, .dtype=torch::kFloat32}}},
        std::make_shared<dqn::LinearHeadFactory>(2, anet::nn::WeightInitConfig{}, projection), 8461, torch::kCPU);
    torch::NoGradGuard guard;
    for (const auto& entry : network->GetNoisyParameterEntries()) {
        entry.mean.fill_(zero_mean ? 0.0 : 3.0);
        entry.sigma.fill_(entry.is_bias ? -2.0 : 2.0);
    }
    // [feature weight,bias,readout weight,bias] の mean-abs、続いて同順の L2 ratio。
    const auto stats = network->ComputeNoisyParameterStatistics("feature").cpu();
    REQUIRE(stats.sizes() == torch::IntArrayRef({8}));
    for (int64_t group = 0; group < 4; ++group) {
        const bool available = group >= 2 || noisy_feature;
        if (!available) {
            CHECK(std::isnan(stats[group].item<float>()));
            CHECK(std::isnan(stats[group+4].item<float>()));
            continue;
        }
        CHECK(stats[group].item<float>() == Catch::Approx(2.0f));
        if (zero_mean) {
            CHECK(std::isnan(stats[group+4].item<float>()));
        } else if (group == 0 && norm_mode != "none") {
            const auto entry = network->GetSpectralNormEntries().front();
            const auto raw = anet::nn::ComputeSpectralSigma(entry.weight, entry.u, entry.v);
            const auto denominator = norm_mode == "spectral_cap" ? raw.abs().clamp_min(1.0) : raw;
            const auto effective_mean = entry.weight / denominator;
            CHECK(stats[4].item<float>() == Catch::Approx(std::sqrt(6.0f) * 2.0f / effective_mean.norm().item<float>()).epsilon(1e-5));
        } else {
            CHECK(stats[group+4].item<float>() == Catch::Approx(2.0f / 3.0f));
        }
    }
    CHECK_THROWS(network->ComputeNoisyParameterStatistics("missing"));
}

TEST_CASE("Noisy parameters clone synchronize and serialize without resetting held draws", "[nn][noisynet][serialize][copy]")
{
    EnsureNNInitialized();
    const auto kind = GENERATE(std::string("td"), std::string("dueling"), std::string("qr"),
        std::string("qr_dueling"), std::string("iqn"), std::string("iqn_dueling"));
    const bool spectral = GENERATE(false, true);
    CAPTURE(kind, spectral);
    const bool iqn = kind.starts_with("iqn");
    anet::ConfigData data;
    data.Set("net.block.[Linear].type", "Linear");
    data.Set("net.block.[Linear].linear.out_features", 4);
    data.Set("net.block.[Linear].noisy.enabled", true);
    data.Set("net.block.[Linear].weight_norm.mode", spectral ? "spectral" : "none");
    data.Set("net.branch.[feature].bind", "obs");
    data.Set("net.branch.[feature].structure", "Linear");
    data.Set("net.body.output.[features]", "feature");
    data.Set("hold.noisy.mode", "sample");
    data.Set("hold.noisy.hold", "count");
    data.Set("hold.noisy.count", 32);
    data.Set("hold.noisy.share", "batch");
    anet::ConfigData projection;
    projection.Set("noisy.enabled", true);
    const anet::nn::WeightInitConfig init{};
    std::shared_ptr<anet::nn::NetworkHeadFactory> factory;
    if (kind == "td") factory = std::make_shared<dqn::LinearHeadFactory>(2, init, projection);
    else if (kind == "dueling") factory = std::make_shared<dqn::DuelingHeadFactory>(2, init, projection);
    else if (kind == "qr") factory = std::make_shared<dqn::QuantileHeadFactory>(2, 3, init, projection);
    else if (kind == "qr_dueling") factory = std::make_shared<dqn::QuantileDuelingHeadFactory>(2, 3, init, projection);
    else if (kind == "iqn") factory = std::make_shared<dqn::IQNHeadFactory>(2, init, projection);
    else factory = std::make_shared<dqn::IQNDuelingHeadFactory>(2, init, projection);
    const auto build = [&](anet::seed_t seed) {
        return anet::nn::NetworkBuilder::BuildNetwork(anet::nn::NetworkConfig(data),
            anet::TensorSpecMap{{"obs", anet::TensorSpec{.type=anet::SpaceType::Vector,
                .shape=iqn ? std::vector<int64_t>{3, 2} : std::vector<int64_t>{2}, .dtype=torch::kFloat32}}},
            factory, seed, torch::kCPU);
    };
    auto source = build(8471);
    auto target = build(8472);
    source->eval();
    target->eval();
    torch::NoGradGuard guard;
    for (const auto& entry : source->GetNoisyParameterEntries()) entry.sigma.fill_(-0.75f);
    const anet::nn::NetworkForwardConfig config("hold", SelectForwardConfigData(data, "hold"));
    const anet::TensorDict input{{"obs", torch::ones(iqn ? std::vector<int64_t>{2, 3, 2} : std::vector<int64_t>{2, 2})}};
    auto source_forward = source->CreateForwardExecutor(config, anet::nn::NetworkForwardClock::kContinuingRows, 901);
    const anet::nn::NetworkForwardCallInfo source_info({.clock = 1});
    auto target_forward = target->CreateForwardExecutor(config, anet::nn::NetworkForwardClock::kContinuingRows, 901);
    const anet::nn::NetworkForwardCallInfo target_info({.clock = 1});
    const auto expected = source_forward->Forward(input, source_info).At("q").clone();
    target_forward->Forward(input, target_info);
    source->CopyTo(*target);
    CHECK(torch::equal(target_forward->Forward(input, target_info).At("q"), expected));
    CHECK(source->buffers().size() == (spectral ? 2 : 0));
    auto check_parameters = [&](const auto& network) {
        const auto expected_parameters = source->named_parameters();
        const auto actual_parameters = network->named_parameters();
        REQUIRE(expected_parameters.size() == actual_parameters.size());
        REQUIRE(network->GetNoisyParameterEntries().size() == source->GetNoisyParameterEntries().size());
        for (const auto& entry : expected_parameters) CHECK(torch::equal(entry.value(), actual_parameters[entry.key()]));
    };
    check_parameters(target);
    auto cloned = source->Clone(torch::kCPU);
    check_parameters(cloned);
    auto restored = build(8473);
    std::stringstream archive(std::ios::in | std::ios::out | std::ios::binary);
    std::shared_ptr<torch::nn::Module> source_base = source;
    torch::save(source_base, archive);
    archive.seekg(0);
    std::shared_ptr<torch::nn::Module> restored_base = restored;
    torch::load(restored_base, archive);
    check_parameters(restored);
    restored->eval();
    auto restored_forward = restored->CreateForwardExecutor(config, anet::nn::NetworkForwardClock::kContinuingRows, 901);
    const anet::nn::NetworkForwardCallInfo restored_info({.clock = 1});
    CHECK(torch::equal(restored_forward->Forward(input, restored_info).At("q"), expected));
    // soft copy も sigma を含む学習 parameter 全体を同じ係数でブレンドする。
    std::map<std::string, torch::Tensor> before;
    for (const auto& entry : target->named_parameters()) before.emplace(entry.key(), entry.value().clone());
    for (const auto& parameter : source->parameters()) parameter.add_(0.5f);
    source->SoftCopyTo(*target, 0.1);
    const auto target_parameters = target->named_parameters();
    for (const auto& entry : source->named_parameters()) {
        CHECK(torch::allclose(target_parameters[entry.key()], before.at(entry.key()) * 0.9f + entry.value() * 0.1f));
    }
}

TEST_CASE("CUDA autocast preserves Body overrides and FP32 Head and spectral calculation", "[nn][noisynet][precision][cuda]")
{
    EnsureNNInitialized();
    if (!torch::cuda::is_available()) SKIP("CUDA unavailable");
    const bool force = GENERATE(false, true);
    const auto norm = GENERATE(std::string("none"), std::string("spectral"), std::string("spectral_cap"));
    const torch::Device device(torch::kCUDA, 0);
    CAPTURE(force, norm);
    anet::ConfigData data;
    data.Set("net.block.[FC].type", "Linear");
    data.Set("net.block.[FC].linear.out_features", 4);
    data.Set("net.block.[FC].noisy.enabled", true);
    data.Set("net.block.[FC].force_fp32", force);
    data.Set("net.block.[FC].weight_norm.mode", norm);
    data.Set("net.branch.[feature].bind", "obs");
    data.Set("net.branch.[feature].structure", "FC");
    data.Set("net.body.output.[features]", "feature");
    data.Set("sample.noisy.mode", "sample");
    data.Set("sample.noisy.hold", "call");
    data.Set("sample.noisy.share", "batch");
    anet::ConfigData projection;
    projection.Set("noisy.enabled", true);
    auto network = anet::nn::NetworkBuilder::BuildNetwork(anet::nn::NetworkConfig(data),
        anet::TensorSpecMap{{"obs", anet::TensorSpec{.type=anet::SpaceType::Vector, .shape={2}, .dtype=torch::kFloat32}}},
        std::make_shared<dqn::LinearHeadFactory>(3, anet::nn::WeightInitConfig{}, projection), 702, torch::kCPU);
    network->to(device);
    network->eval();
    torch::NoGradGuard guard;
    const anet::nn::NetworkForwardConfig config("sample", SelectForwardConfigData(data, "sample"));
    auto forward = network->CreateForwardExecutor(config, anet::nn::NetworkForwardClock::kContinuingRows, 703);
    const anet::nn::NetworkForwardCallInfo info({.clock = 0});
    const anet::TensorDict input{{"obs", torch::ones({2, 2}, torch::TensorOptions().device(device))}};
    anet::nn::NetworkForwardCapture capture({ "feature" });
    {
        anet::Autocast autocast(device, true, torch::kBFloat16);
        const auto output = forward->Forward(input, info, &capture);
        CHECK(output.At("q").scalar_type() == torch::kFloat32);
        CHECK(capture.Get("feature").scalar_type() == (force ? torch::kFloat32 : torch::kBFloat16));
        for (const auto& entry : network->GetSpectralNormEntries()) {
            CHECK(entry.u.scalar_type() == torch::kFloat32);
            CHECK(entry.v.scalar_type() == torch::kFloat32);
            CHECK(anet::nn::ComputeSpectralSigma(entry.weight, entry.u, entry.v).scalar_type() == torch::kFloat32);
        }
    }
}

TEST_CASE("Unreferenced forward contents remain dormant", "[nn][noisynet][simplify_catalog]")
{
    auto data = WithDefaultForwardConfig();
    data.Set("DefaultDQNAgent.nn_forward.[unused].noisy.mode", "invalid");
    data.Set("DefaultDQNAgent.nn_forward.[unused].noisy.count", "-1");
    CHECK_NOTHROW(dqn::DefaultDQNAgentConfig(data));
    data.Set("DefaultDQNAgent.learner.current.nn_forward_key", "missing");
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data), Catch::Matchers::ContainsSubstring("missing"));
}

TEST_CASE("Network-specific Noisy executors isolate synchronized draws for all Q Heads",
    "[nn][noisynet][prepared][target]")
{
    EnsureNNInitialized();
    const auto mode = GENERATE(std::string("td"), std::string("qr"), std::string("iqn"));
    const bool dueling = GENERATE(false, true);
    CAPTURE(mode, dueling);
    anet::ConfigData projection;
    projection.Set("noisy.enabled", true);
    std::shared_ptr<anet::nn::NetworkHeadFactory> factory;
    const anet::nn::WeightInitConfig init{};
    if (mode == "td") {
        if (dueling) factory = std::make_shared<dqn::DuelingHeadFactory>(2, init, projection);
        else factory = std::make_shared<dqn::LinearHeadFactory>(2, init, projection);
    } else if (mode == "qr") {
        if (dueling) factory = std::make_shared<dqn::QuantileDuelingHeadFactory>(2, 3, init, projection);
        else factory = std::make_shared<dqn::QuantileHeadFactory>(2, 3, init, projection);
    } else {
        if (dueling) factory = std::make_shared<dqn::IQNDuelingHeadFactory>(2, init, projection);
        else factory = std::make_shared<dqn::IQNHeadFactory>(2, init, projection);
    }
    anet::ConfigData data;
    data.Set("net.branch.[feature].bind", "obs");
    data.Set("net.body.output.[features]", "feature");
    const std::vector<int64_t> shape = mode == "iqn" ? std::vector<int64_t>{3, 4} : std::vector<int64_t>{4};
    const anet::TensorSpecMap specs{{"obs", {.type=anet::SpaceType::Vector, .shape=shape, .dtype=torch::kFloat32}}};
    auto online = anet::nn::NetworkBuilder::BuildNetwork(anet::nn::NetworkConfig(data), specs, factory, 981);
    auto target = online->Clone();
    anet::ConfigData selected;
    selected.Set("noisy.mode", "sample");
    selected.Set("noisy.hold", "call");
    selected.Set("noisy.share", "sample");
    const anet::nn::NetworkForwardConfig config("nn_forward.[target]", selected);
    const anet::SeedMaker seeds(982);
    auto online_forward = online->CreateForwardExecutor(config, anet::nn::NetworkForwardClock::kTransientRows,
        seeds.MakeNamedSeed("network_forward.target_online"));
    auto target_forward = target->CreateForwardExecutor(config, anet::nn::NetworkForwardClock::kTransientRows,
        seeds.MakeNamedSeed("network_forward.target"));
    auto target_reference = target->CreateForwardExecutor(config, anet::nn::NetworkForwardClock::kTransientRows,
        seeds.MakeNamedSeed("network_forward.target"));
    const auto tensor = mode == "iqn" ? torch::ones({2, 3, 4}) : torch::ones({2, 4});
    const anet::TensorDict input{{"obs", tensor}};
    const anet::nn::NetworkForwardCallInfo first({.clock=0});
    const anet::nn::NetworkForwardCallInfo next({.clock=1});
    const auto online_q = online_forward->Forward(input, first).At("q");
    const auto target_q = target_forward->Forward(input, first).At("q");
    CHECK_FALSE(torch::equal(online_q, target_q));
    CHECK(torch::equal(target_q, target_reference->Forward(input, first).At("q")));
    // onlineだけ進めても、targetの保持と次の抽選を変えない。
    online_forward->Forward(input, next);
    online->CopyTo(*target);
    CHECK(torch::equal(target_q, target_forward->Forward(input, first).At("q")));
    CHECK(torch::equal(target_forward->Forward(input, next).At("q"), target_reference->Forward(input, next).At("q")));
    online.reset();
    target.reset();
    CHECK(torch::equal(target_forward->Forward(input, next).At("q"), target_reference->Forward(input, next).At("q")));
}
