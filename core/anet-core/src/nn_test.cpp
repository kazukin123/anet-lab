#include "anet/catch_test.hpp"

#include "anet/default_dqn_agent.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/muzero_proto_agent.hpp"
#include "anet/nn_util.hpp"
#include "anet/test_util.hpp"
#include "dqn_based_heads.hpp"
#include "nn_impl.hpp"
#include "nn_heads.hpp"

#include <ATen/autocast_mode.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

namespace dqn = anet::rl::dqn;
namespace muzero = anet::rl::muzero_proto;

static_assert(!std::is_copy_constructible_v<anet::Autocast>);
static_assert(!std::is_move_constructible_v<anet::Autocast>);
static_assert(!std::is_constructible_v<anet::Autocast, torch::DeviceType, bool, torch::ScalarType>);
static_assert(std::is_constructible_v<anet::Autocast, torch::Device, bool, torch::ScalarType>);

bool Contains(const std::string& text, const std::string& pattern)
{
    return text.find(pattern) != std::string::npos;
}

void EnsureNNInitialized()
{
    static const bool initialized = [] {
        anet::nn::InitNN();
        // Debug版libtorchのoneDNNはCPU grouped conv(CNBlockのdepthwise)で
        // thread検証assert(nthr_==nthr)を起こすため、CPUテスト用にmkldnnを無効化する。
        // release版libtorch/本番GPUでは発生しないため本番コードには入れない。
        at::globalContext().setUserEnabledMkldnn(false);
        return true;
    }();
    (void)initialized;
}

std::vector<torch::Tensor> MakeAdamWTestParams(const torch::Device& device)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto p0 = (torch::arange(0, 6, options).reshape({ 2, 3 }) * 0.1f + 0.5f).clone();
    auto p1 = (torch::arange(0, 4, options).reshape({ 4 }) * -0.2f + 0.3f).clone();
    p0.requires_grad_(true);
    p1.requires_grad_(true);
    return { p0, p1 };
}

std::vector<torch::Tensor> CloneAdamWTestParams(const std::vector<torch::Tensor>& params)
{
    std::vector<torch::Tensor> cloned;
    cloned.reserve(params.size());
    for (const auto& param : params) {
        auto copy = param.detach().clone();
        copy.requires_grad_(true);
        cloned.push_back(copy);
    }
    return cloned;
}

torch::Tensor MakeAdamWTestGrad(const torch::Tensor& param, int step, size_t index)
{
    auto options = torch::TensorOptions().dtype(param.scalar_type()).device(param.device());
    auto values = torch::arange(0, param.numel(), options).reshape(param.sizes());
    const float offset = static_cast<float>((step + 1) * (index + 1)) * 0.13f;
    return torch::sin(values * 0.17f + offset);
}

void RunAdamWTestStep(torch::optim::Optimizer& optimizer, const std::vector<torch::Tensor>& params, int step)
{
    optimizer.zero_grad();
    auto loss = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(params.front().device()));
    for (size_t i = 0; i < params.size(); ++i) {
        loss = loss + (params[i] * MakeAdamWTestGrad(params[i], step, i)).sum();
    }
    loss.backward();
    optimizer.step();
}

const torch::optim::AdamWParamState& GetAdamWState(
    const torch::optim::Optimizer& optimizer,
    const torch::Tensor& param)
{
    const auto& state = optimizer.state();
    auto it = state.find(param.unsafeGetTensorImpl());
    REQUIRE(it != state.end());
    return static_cast<const torch::optim::AdamWParamState&>(*it->second);
}

void CheckAdamWParamsAndStateClose(
    const std::vector<torch::Tensor>& expected_params,
    const torch::optim::Optimizer& expected_optimizer,
    const std::vector<torch::Tensor>& actual_params,
    const torch::optim::Optimizer& actual_optimizer,
    double rtol = 1.0e-5,
    double atol = 1.0e-7)
{
    REQUIRE(expected_params.size() == actual_params.size());
    for (size_t i = 0; i < expected_params.size(); ++i) {
        INFO("param index=" << i);
        CHECK(torch::allclose(expected_params[i].detach().cpu(), actual_params[i].detach().cpu(), rtol, atol));

        const auto& expected_state = GetAdamWState(expected_optimizer, expected_params[i]);
        const auto& actual_state = GetAdamWState(actual_optimizer, actual_params[i]);
        CHECK(expected_state.step() == actual_state.step());
        CHECK(torch::allclose(expected_state.exp_avg().detach().cpu(), actual_state.exp_avg().detach().cpu(), rtol, atol));
        CHECK(torch::allclose(expected_state.exp_avg_sq().detach().cpu(), actual_state.exp_avg_sq().detach().cpu(), rtol, atol));
    }
}

void CheckTensorClose(
    const torch::Tensor& expected,
    const torch::Tensor& actual,
    double rtol = 1.0e-4,
    double atol = 1.0e-5)
{
    CHECK(torch::allclose(expected.detach().cpu(), actual.detach().cpu(), rtol, atol));
}

std::vector<double> GetDropoutRates(const std::shared_ptr<anet::nn::NetworkStruct>& network_struct)
{
    std::vector<double> rates;
    for (const auto& block : network_struct->GetBlocks()) {
        REQUIRE(block->GetModule());
        const auto config_data = block->GetModule()->GetCurrentConfigData();
        rates.push_back(std::stod(config_data.Get("dropout_rate")));
    }
    return rates;
}

std::shared_ptr<anet::nn::NetworkStruct> GetBranchNetworkStruct(
    const std::shared_ptr<anet::nn::NetworkBody>& body, const std::string& branch_name)
{
    for (const auto& branch : body->GetBranches()) {
        if (branch->GetName() == branch_name) {
            return branch->GetNetworkStruct();
        }
    }
    FAIL("Branch not found: " << branch_name);
    return nullptr;
}

anet::TensorSpec MakeConfigProfileVectorSpec()
{
    anet::TensorSpec spec;
    spec.type = anet::SpaceType::Vector;
    spec.shape = { 4 };
    spec.dtype = torch::kFloat32;
    return spec;
}

torch::Tensor LegacyMhaSelfAttention(torch::nn::MultiheadAttention& mha, const torch::Tensor& x)
{
    torch::Tensor x_t = x.transpose(0, 1);
    return std::get<0>(mha->forward(x_t, x_t, x_t)).transpose(0, 1);
}

torch::Tensor ManualScaledDotProductSelfAttention(const torch::nn::MultiheadAttention& mha, const torch::Tensor& x)
{
    namespace F = torch::nn::functional;

    const int64_t batch_size = x.size(0);
    const int64_t seq_len = x.size(1);
    const int64_t embed_dim = x.size(2);
    const int64_t num_heads = mha->options.num_heads();
    const int64_t head_dim = mha->head_dim;

    torch::Tensor qkv = F::linear(x, mha->in_proj_weight, mha->in_proj_bias);
    std::vector<torch::Tensor> chunks = qkv.chunk(3, /*dim=*/-1);

    auto to_heads = [&](const torch::Tensor& t) {
        return t.reshape({ batch_size, seq_len, num_heads, head_dim }).transpose(1, 2);
    };

    torch::Tensor q = to_heads(chunks[0]);
    torch::Tensor k = to_heads(chunks[1]);
    torch::Tensor v = to_heads(chunks[2]);

    torch::Tensor scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(head_dim));
    torch::Tensor weights = torch::softmax(scores, /*dim=*/-1);
    torch::Tensor attn = torch::matmul(weights, v);
    attn = attn.transpose(1, 2).reshape({ batch_size, seq_len, embed_dim });
    return F::linear(attn, mha->out_proj->weight, mha->out_proj->bias);
}

void CopyModuleState(torch::nn::Module& src, torch::nn::Module& dst)
{
    torch::NoGradGuard no_grad;

    auto src_params = src.named_parameters(true);
    auto dst_params = dst.named_parameters(true);
    REQUIRE(src_params.size() == dst_params.size());
    for (const auto& kv : src_params) {
        INFO("param name=" << kv.key());
        dst_params[kv.key()].copy_(kv.value());
    }

    auto src_buffers = src.named_buffers(true);
    auto dst_buffers = dst.named_buffers(true);
    REQUIRE(src_buffers.size() == dst_buffers.size());
    for (const auto& kv : src_buffers) {
        INFO("buffer name=" << kv.key());
        dst_buffers[kv.key()].copy_(kv.value());
    }
}

void CheckModuleGradientsClose(
    torch::nn::Module& expected,
    torch::nn::Module& actual,
    double rtol,
    double atol)
{
    auto expected_params = expected.named_parameters(true);
    auto actual_params = actual.named_parameters(true);
    REQUIRE(expected_params.size() == actual_params.size());
    for (const auto& kv : expected_params) {
        const auto& actual_param = actual_params[kv.key()];
        INFO("grad name=" << kv.key());
        REQUIRE(kv.value().grad().defined());
        REQUIRE(actual_param.grad().defined());
        CheckTensorClose(kv.value().grad(), actual_param.grad(), rtol, atol);
    }
}

torch::nn::Linear MakeWeightInitTestLinear()
{
    return torch::nn::Linear(torch::nn::LinearOptions(4, 3).bias(true));
}

void FillWeightInitTestLinear(torch::nn::Linear& layer, double weight_value, double bias_value)
{
    torch::NoGradGuard no_grad;
    layer->weight.fill_(weight_value);
    layer->bias.fill_(bias_value);
}

void CheckWeightInitMatchesDirect(
    const anet::nn::WeightInitConfig& config,
    int64_t seed,
    const std::function<void(torch::nn::Linear&)>& initialize_expected)
{
    auto actual = MakeWeightInitTestLinear();
    auto expected = MakeWeightInitTestLinear();

    torch::manual_seed(seed);
    anet::nn::WeightInitializer::Initialize(actual, config);

    torch::manual_seed(seed);
    {
        torch::NoGradGuard no_grad;
        initialize_expected(expected);
    }

    CheckTensorClose(expected->weight, actual->weight);
    CheckTensorClose(expected->bias, actual->bias);
}

template <typename Dict>
bool HasKey(const Dict& dict, const std::string& key)
{
    for (const auto& kv : dict) {
        if (kv.key() == key) {
            return true;
        }
    }
    return false;
}

std::shared_ptr<anet::nn::NetworkModule> MakeDropoutTestModule(double dropout_rate, bool set_old_p = false)
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    if (set_old_p) {
        config_data.Set("p", dropout_rate);
    } else {
        config_data.Set("dropout_rate", dropout_rate);
    }

    auto factory = anet::nn::NetworkModuleRepository::Instance().GetFactory("Dropout");
    return factory->CreateModule(config_data, anet::nn::ModuleContext{});
}

std::shared_ptr<anet::nn::NetworkModule> MakeCosineEmbeddingTestModule(int64_t num_basis = 64)
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("cos.num_basis", num_basis);
    auto factory = anet::nn::NetworkModuleRepository::Instance().GetFactory("CosineEmbedding");
    return factory->CreateModule(config_data, anet::nn::ModuleContext{});
}

std::shared_ptr<anet::nn::NetworkModule> MakeResBlockTestModule(
    double droppath_rate,
    double dropout_rate,
    const std::string& norm_type = "none",
    const std::string& activation_mode = "pre",
    std::optional<bool> norm_force_fp32 = std::nullopt)
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("res.channels", 3);
    config_data.Set("res.kernel_size", 3);
    config_data.Set("res.padding", 1);
    config_data.Set("res.activation", "relu");
    config_data.Set("res.activation_mode", activation_mode);
    config_data.Set("res.norm_type", norm_type);
    if (norm_force_fp32.has_value()) {
        config_data.Set("res.norm_force_fp32", *norm_force_fp32);
    }
    config_data.Set("res.droppath_rate", droppath_rate);
    config_data.Set("res.dropout_rate", dropout_rate);
    config_data.Set("init2.mode", "he");

    auto factory = anet::nn::NetworkModuleRepository::Instance().GetFactory("ResBlock");
    return factory->CreateModule(config_data, anet::nn::ModuleContext{});
}

std::shared_ptr<anet::nn::NetworkModule> MakeBatchNorm2dTestModule(
    int num_features,
    std::optional<bool> force_fp32 = std::nullopt)
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("num_features", num_features);
    if (force_fp32.has_value()) {
        config_data.Set("force_fp32", *force_fp32);
    }

    auto factory = anet::nn::NetworkModuleRepository::Instance().GetFactory("BatchNorm2d");
    return factory->CreateModule(config_data, anet::nn::ModuleContext{});
}

std::shared_ptr<anet::nn::NetworkModule> MakeLayerNormTestModule(
    int normalized_shape,
    std::optional<double> eps = std::nullopt,
    std::optional<bool> force_fp32 = std::nullopt)
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("normalized_shape", normalized_shape);
    if (eps.has_value()) {
        config_data.Set("eps", *eps);
    }
    if (force_fp32.has_value()) {
        config_data.Set("force_fp32", *force_fp32);
    }

    auto factory = anet::nn::NetworkModuleRepository::Instance().GetFactory("LayerNorm");
    return factory->CreateModule(config_data, anet::nn::ModuleContext{});
}

std::shared_ptr<anet::nn::NetworkModule> MakeLayerNorm2dTestModule(
    int num_channels,
    double eps = 1.0e-6,
    std::optional<bool> force_fp32 = std::nullopt)
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("num_channels", num_channels);
    config_data.Set("eps", eps);
    if (force_fp32.has_value()) {
        config_data.Set("force_fp32", *force_fp32);
    }

    auto factory = anet::nn::NetworkModuleRepository::Instance().GetFactory("LayerNorm2d");
    return factory->CreateModule(config_data, anet::nn::ModuleContext{});
}

std::shared_ptr<anet::nn::NetworkModule> MakeCNBlockTestModule(
    int channels = 3,
    double droppath_rate = 0.0,
    double layerscale_init = 1.0e-6,
    const std::string& norm_type = "layernorm2d",
    int kernel_size = 3,
    int ffn_expand_ratio = 2,
    bool constant_init = false,
    std::optional<bool> norm_force_fp32 = std::nullopt)
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("cn.channels", channels);
    config_data.Set("cn.kernel_size", kernel_size);
    config_data.Set("cn.ffn_expand_ratio", ffn_expand_ratio);
    config_data.Set("cn.layerscale_init", layerscale_init);
    config_data.Set("cn.droppath_rate", droppath_rate);
    config_data.Set("cn.norm_type", norm_type);
    if (norm_force_fp32.has_value()) {
        config_data.Set("cn.norm_force_fp32", *norm_force_fp32);
    }
    if (constant_init) {
        for (const std::string prefix : { "init_dw.", "init_pw1.", "init_pw2." }) {
            config_data.Set(prefix + "mode", std::string("constant"));
            config_data.Set(prefix + "constant_val", 0.05);
        }
    }

    auto factory = anet::nn::NetworkModuleRepository::Instance().GetFactory("CNBlock");
    return factory->CreateModule(config_data, anet::nn::ModuleContext{});
}

torch::Tensor GetNamedParameter(torch::nn::Module& module, const std::string& name)
{
    for (const auto& kv : module.named_parameters(true)) {
        if (kv.key() == name) {
            return kv.value();
        }
    }
    FAIL("Parameter not found: " << name);
    return torch::Tensor();
}

std::shared_ptr<anet::nn::NetworkModule> MakeTransformerTestModule(
    bool use_sdpa,
    bool norm_first,
    double hidden_dropout_rate = 0.0,
    double attn_dropout_rate = 0.0,
    double droppath_rate = 0.0)
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("tf.d_model", 16);
    config_data.Set("tf.nhead", 4);
    config_data.Set("tf.num_layers", 1);
    config_data.Set("tf.dim_feedforward", 32);
    config_data.Set("tf.norm_first", norm_first ? "true" : "false");
    config_data.Set("tf.use_sdpa", use_sdpa ? "true" : "false");
    config_data.Set("tf.activation", "gelu");
    config_data.Set("tf.hidden_dropout_rate", hidden_dropout_rate);
    config_data.Set("tf.attn_dropout_rate", attn_dropout_rate);
    config_data.Set("tf.droppath_rate", droppath_rate);

    auto factory = anet::nn::NetworkModuleRepository::Instance().GetFactory("TransformerEncoder");
    return factory->CreateModule(config_data, anet::nn::ModuleContext{});
}

void CheckTransformerSdpaMatchesLegacy(const torch::Device& device, bool norm_first)
{
    torch::manual_seed(2030 + (norm_first ? 1 : 0));
    auto sdpa_module = MakeTransformerTestModule(/*use_sdpa=*/true, norm_first);
    auto legacy_module = MakeTransformerTestModule(/*use_sdpa=*/false, norm_first);
    CopyModuleState(*sdpa_module, *legacy_module);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor sdpa_input = torch::randn({ 2, 5, 16 }, options);
    sdpa_input.requires_grad_(true);
    torch::Tensor legacy_input = sdpa_input.detach().clone();
    legacy_input.requires_grad_(true);

    torch::Tensor sdpa_output = sdpa_module->Forward(sdpa_input);
    torch::Tensor legacy_output = legacy_module->Forward(legacy_input);

    const double rtol = device.is_cuda() ? 1.0e-3 : 1.0e-4;
    const double atol = device.is_cuda() ? 1.0e-4 : 1.0e-5;
    CheckTensorClose(legacy_output, sdpa_output, rtol, atol);

    sdpa_output.pow(2).mean().backward();
    legacy_output.pow(2).mean().backward();

    CheckTensorClose(legacy_input.grad(), sdpa_input.grad(), rtol, atol);
    CheckModuleGradientsClose(*legacy_module, *sdpa_module, rtol, atol);
}

class DotTestModule final : public anet::nn::NetworkModule {
public:
    DotTestModule()
    {
        weight_ = register_parameter("weight", torch::ones({ 2, 3 }));
    }

    torch::Tensor Forward(torch::Tensor input) override
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

class SoftCopyTestModule final : public anet::nn::NetworkModule {
public:
    SoftCopyTestModule(float base, int64_t counter)
    {
        auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
        param0_ = register_parameter("param0", torch::tensor({ base, base + 1.0f }, float_options));
        param1_ = register_parameter("param1", torch::tensor({
            { base + 2.0f, base + 3.0f },
            { base + 4.0f, base + 5.0f },
        }, float_options));
        float_buffer_ = register_buffer("float_buffer", torch::tensor({ base + 6.0f, base + 7.0f }, float_options));
        int_buffer_ = register_buffer("int_buffer", torch::tensor({ counter }, torch::TensorOptions().dtype(torch::kInt64)));
    }

    torch::Tensor Forward(torch::Tensor input) override
    {
        return input;
    }

private:
    torch::Tensor param0_;
    torch::Tensor param1_;
    torch::Tensor float_buffer_;
    torch::Tensor int_buffer_;
};

class TraceTestModule final : public anet::nn::NetworkModule {
public:
    torch::Tensor Forward(torch::Tensor input) override
    {
        return input + 1.0f;
    }

    bool IsConv2dVisualizable() const override { return true; }
};

class DotTestHead final : public anet::nn::NetworkHead {
public:
    anet::TensorDict Forward(const anet::TensorDict& feature_dict) override
    {
        return feature_dict;
    }
};

std::shared_ptr<anet::nn::Network> MakeDotTestNetwork(
    std::shared_ptr<anet::nn::NetworkHead> head = nullptr,
    const std::string& head_key = "feature")
{
    anet::TensorSpec obs_spec;
    obs_spec.type = anet::SpaceType::Vector;
    obs_spec.shape = { 2 };
    obs_spec.dtype = torch::kFloat32;

    anet::TensorSpecMap input_specs;
    input_specs["obs"] = obs_spec;

    auto block = std::make_shared<anet::nn::NetworkBlock>(
        "Scale_0",
        std::make_shared<DotTestModule>());
    auto network_struct = std::make_shared<anet::nn::NetworkStruct>(
        std::vector<std::shared_ptr<anet::nn::NetworkBlock>>{ block });
    auto branch = std::make_shared<anet::nn::NetworkBranch>(
        "feature",
        std::vector<std::vector<std::string>>{ { "obs" } },
        1,
        network_struct);

    anet::nn::NetworkConfig network_config;
    anet::nn::NetworkBranchConfig branch_config;
    branch_config.name = "feature";
    branch_config.bind_terms = { { "obs" } };
    branch_config.raw_keys = { "obs" };
    branch_config.auto_format = false;
    branch_config.structure_str = "Scale_0";
    network_config.branches["feature"] = branch_config;
    network_config.output_keys[head_key] = "feature";

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
        head ? head : std::make_shared<DotTestHead>());
}

std::shared_ptr<anet::nn::Network> MakeSoftCopyTestNetwork(float base, int64_t counter)
{
    anet::TensorSpec obs_spec;
    obs_spec.type = anet::SpaceType::Vector;
    obs_spec.shape = { 2 };
    obs_spec.dtype = torch::kFloat32;

    anet::TensorSpecMap input_specs;
    input_specs["obs"] = obs_spec;

    auto block = std::make_shared<anet::nn::NetworkBlock>(
        "SoftCopy_0",
        std::make_shared<SoftCopyTestModule>(base, counter));
    auto network_struct = std::make_shared<anet::nn::NetworkStruct>(
        std::vector<std::shared_ptr<anet::nn::NetworkBlock>>{ block });
    auto branch = std::make_shared<anet::nn::NetworkBranch>(
        "feature",
        std::vector<std::vector<std::string>>{ { "obs" } },
        1,
        network_struct);

    anet::nn::NetworkConfig network_config;
    network_config.output_keys["feature"] = "feature";

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
        std::make_shared<DotTestHead>());
}

std::shared_ptr<anet::nn::Network> MakeBodyOnlyDotTestNetwork()
{
    anet::TensorSpec obs_spec;
    obs_spec.type = anet::SpaceType::Vector;
    obs_spec.shape = { 2 };
    obs_spec.dtype = torch::kFloat32;

    anet::TensorSpecMap input_specs;
    input_specs["obs"] = obs_spec;

    auto block = std::make_shared<anet::nn::NetworkBlock>(
        "Scale_0",
        std::make_shared<DotTestModule>());
    auto network_struct = std::make_shared<anet::nn::NetworkStruct>(
        std::vector<std::shared_ptr<anet::nn::NetworkBlock>>{ block });
    auto branch = std::make_shared<anet::nn::NetworkBranch>(
        "feature",
        std::vector<std::vector<std::string>>{ { "obs" } },
        1,
        network_struct);

    anet::nn::NetworkConfig network_config;
    network_config.output_keys["feature"] = "feature";

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
        nullptr);
}

std::shared_ptr<anet::nn::NetworkHead> CreateDqnHead(const anet::nn::NetworkHeadFactory& factory)
{
    anet::TensorDict dummy_features;
    dummy_features.Set(anet::nn::kKey_DefaultOutput, torch::zeros({ 1, 4 }));
    return factory.CreateHead(dummy_features);
}

TEST_CASE("LinearHeadFactory emits configurable output key", "[nn][head]")
{
    EnsureNNInitialized();

    anet::nn::WeightInitConfig init_config;
    init_config.mode = "he";
    anet::nn::LinearHeadFactory factory(3, "logits", init_config);

    anet::TensorDict dummy_features;
    dummy_features.Set(anet::nn::kKey_DefaultOutput, torch::ones({ 2, 4 }));

    auto head = factory.CreateHead(dummy_features);
    auto output = head->Forward(dummy_features);
    REQUIRE(output.Contains("logits"));
    CHECK_FALSE(output.Contains("q"));
    CHECK(output.At("logits").sizes() == torch::IntArrayRef({ 2, 3 }));
    CHECK(output.At("logits").dtype() == torch::kFloat32);
    CHECK(HasKey(head->named_parameters(true), "linear.weight"));
    CHECK(HasKey(head->named_parameters(true), "linear.bias"));

    auto logits_func = head->GetTensorDictFunction("logits");
    REQUIRE(logits_func.has_value());
    auto func_output = (*logits_func)(dummy_features);
    REQUIRE(func_output.Contains("logits"));
    CHECK(func_output.At("logits").sizes() == torch::IntArrayRef({ 2, 3 }));

    auto graph_info = head->GetGraphVizInfo();
    REQUIRE(graph_info.outputs.size() == 1);
    CHECK(graph_info.type == "LinearHead");
    CHECK(graph_info.outputs[0].name == "logits");
    CHECK(graph_info.outputs[0].shape == std::vector<int64_t>{ 3 });
}

TEST_CASE("Network keeps head output FP32 under CPU autocast", "[nn][head][bf16]")
{
    EnsureNNInitialized();

    anet::nn::WeightInitConfig init_config;
    init_config.mode = "he";
    auto head = std::make_shared<anet::nn::LinearHead>(
        2,
        3,
        "logits",
        init_config);
    auto network = MakeDotTestNetwork(head, anet::nn::kKey_DefaultOutput);

    anet::TensorDict input;
    input.Set("obs", torch::ones({ 2, 2 }, torch::TensorOptions().dtype(torch::kFloat32)));

    anet::TensorDict output;
    {
        anet::Autocast autocast_guard(torch::Device(torch::kCPU), true, torch::kBFloat16);
        output = network->Forward(input);
    }
    REQUIRE(output.Contains("logits"));
    REQUIRE(output.At("logits").dtype() == torch::kFloat32);

    auto logits_func = network->GetTensorDictFunction("logits");
    REQUIRE(logits_func.has_value());
    anet::TensorDict func_output;
    {
        anet::Autocast autocast_guard(torch::Device(torch::kCPU), true, torch::kBFloat16);
        func_output = (*logits_func)(input);
    }
    REQUIRE(func_output.Contains("logits"));
    REQUIRE(func_output.At("logits").dtype() == torch::kFloat32);
}

TEST_CASE("Network bind product fuses feature and tau tensors", "[nn][bind]")
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("net.branch.[fusion].bind", "features * tau_embedding");
    config_data.Set("net.branch.[fusion].structure", "");
    config_data.Set("net.body.output.[features]", "fusion");
    anet::nn::NetworkConfig config(config_data);

    anet::TensorSpecMap input_specs;
    input_specs["features"] = anet::TensorSpec{
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
    };
    input_specs["tau_embedding"] = anet::TensorSpec{
        .type = anet::SpaceType::Vector,
        .shape = { 3, 2 },
        .dtype = torch::kFloat32,
    };

    auto body = anet::nn::NetworkBodyBuilder::Build(config, input_specs);
    anet::TensorDict input;
    input.Set("features", torch::tensor({ { 2.0f, 3.0f }, { 5.0f, 7.0f } }));
    input.Set("tau_embedding", torch::tensor({
        { { 1.0f, 2.0f }, { 3.0f, 4.0f }, { 5.0f, 6.0f } },
        { { 2.0f, 3.0f }, { 4.0f, 5.0f }, { 6.0f, 7.0f } },
    }));

    const auto output = body->Forward(input);
    const auto expected = torch::tensor({
        { { 2.0f, 6.0f }, { 6.0f, 12.0f }, { 10.0f, 18.0f } },
        { { 10.0f, 21.0f }, { 20.0f, 35.0f }, { 30.0f, 49.0f } },
    });
    CheckTensorClose(expected, output.At("features"));
}

TEST_CASE("CosineEmbedding expands taus into cosine basis features", "[nn][iqn][cosine_embedding]")
{
    auto module = MakeCosineEmbeddingTestModule(4);
    const auto taus = torch::tensor(
        { { 0.0, 0.5, 1.0 }, { 0.25, 0.75, 0.0 } },
        torch::TensorOptions().dtype(torch::kFloat64));
    const auto output = module->Forward(taus);

    CHECK(output.sizes() == torch::IntArrayRef({ 2, 3, 4 }));
    CHECK(output.dtype() == torch::kFloat64);
    CHECK(output.device() == taus.device());
    CHECK(module->GetCurrentConfigData().Get("num_basis") == "4");
    CHECK(torch::allclose(output.select(-1, 0), torch::ones({ 2, 3 }, taus.options())));
    CHECK(torch::allclose(output[0][0], torch::ones({ 4 }, taus.options())));
    CHECK(torch::allclose(
        output[0][2],
        torch::tensor({ 1.0, -1.0, 1.0, -1.0 }, taus.options()),
        1.0e-12,
        1.0e-12));
}

TEST_CASE("CosineEmbedding validates its local input and config contracts", "[nn][iqn][cosine_embedding]")
{
    CHECK_THROWS_WITH(
        MakeCosineEmbeddingTestModule(0),
        Catch::Matchers::ContainsSubstring("cos.num_basis")
        && Catch::Matchers::ContainsSubstring("value=0")
        && Catch::Matchers::ContainsSubstring("expected=>0"));

    auto module = MakeCosineEmbeddingTestModule(4);
    CHECK_THROWS_WITH(
        module->Forward(torch::zeros({ 2, 3, 1 })),
        Catch::Matchers::ContainsSubstring("CosineEmbedding")
        && Catch::Matchers::ContainsSubstring("rank=3")
        && Catch::Matchers::ContainsSubstring("expected=2"));
}

TEST_CASE("Network bind config exposes product terms and concat dimension", "[nn][bind][config]")
{
    anet::ConfigData config_data;
    config_data.Set("net.branch.[fusion].bind", " a(raw) * b * c , d, , ");
    config_data.Set("net.branch.[fusion].bind_concat_dim", -1);
    config_data.Set("net.branch.[fusion].structure", "");

    const anet::nn::NetworkConfig config(config_data);
    const auto& branch = config.branches.at("fusion");
    CHECK(branch.bind_terms == std::vector<std::vector<std::string>>{ { "a", "b", "c" }, { "d" } });
    CHECK(branch.bind_concat_dim == -1);
    CHECK(branch.raw_keys == std::vector<std::string>{ "a" });

    const auto json = config.ToJson();
    const auto& branch_json = json.at("branches").at("fusion");
    CHECK(branch_json.at("bind_terms") == branch.bind_terms);
    CHECK(branch_json.at("bind_concat_dim") == -1);
    CHECK(branch_json.at("raw_keys") == branch.raw_keys);
    CHECK(branch_json.at("structure") == "");
    CHECK_FALSE(branch_json.contains("bind_keys"));
}

TEST_CASE("Network bind product rejects empty factors with context", "[nn][bind][config]")
{
    for (const std::string bind : { "a**b", "*a", "a*" }) {
        INFO("bind=" << bind);
        anet::ConfigData config_data;
        config_data.Set("net.branch.[fusion].bind", bind);
        config_data.Set("net.branch.[fusion].structure", "");
        CHECK_THROWS_WITH(
            anet::nn::NetworkConfig(config_data),
            Catch::Matchers::ContainsSubstring("branch 'fusion'")
            && Catch::Matchers::ContainsSubstring("bind=\"")
            && Catch::Matchers::ContainsSubstring(bind));
    }
}

TEST_CASE("NetworkBodyBuilder warns once only for directly unused inputs", "[nn][bind][warning]")
{
    anet::ConfigData config_data;
    config_data.Set("net.branch.[feature].bind", "used");
    config_data.Set("net.branch.[feature].structure", "");
    config_data.Set("net.body.output.[features]", "feature");
    config_data.Set("net.body.output.[direct]", "direct_input");
    const anet::nn::NetworkConfig config(config_data);

    anet::TensorSpec spec{
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
    };
    anet::TensorSpecMap input_specs{
        { "used", spec },
        { "direct_input", spec },
        { "unused", spec },
    };

    anet::test::LogCaptureGuard logs;
    (void)anet::nn::NetworkBodyBuilder::Build(config, input_specs);
    logs.Flush();

    int unused_warning_count = 0;
    for (const auto& record : logs.Records()) {
        if (record.level == wxLOG_Warning && Contains(record.message, "input key 'unused'")
            && Contains(record.message, "not bound")) {
            ++unused_warning_count;
        }
        CHECK_FALSE((Contains(record.message, "input key 'direct_input'") && Contains(record.message, "not bound")));
        CHECK_FALSE((Contains(record.message, "input key 'used'") && Contains(record.message, "not bound")));
    }
    CHECK(unused_warning_count == 1);
}

TEST_CASE("Network bind concat honors non-batch dimensions", "[nn][bind]")
{
    auto network_struct = std::make_shared<anet::nn::NetworkStruct>();
    anet::nn::NetworkBranch branch(
        "concat",
        std::vector<std::vector<std::string>>{ { "a" }, { "b" } },
        -1,
        network_struct);

    anet::TensorDict state;
    state.Set("a", torch::tensor({ { 1.0f, 2.0f }, { 3.0f, 4.0f } }));
    state.Set("b", torch::tensor({ { 5.0f }, { 6.0f } }));
    branch.Execute(state);

    CHECK(torch::equal(
        state.At("concat"),
        torch::tensor({ { 1.0f, 2.0f, 5.0f }, { 3.0f, 4.0f, 6.0f } })));
}

TEST_CASE("Network bind enforces product and concat batch contracts", "[nn][bind]")
{
    auto network_struct = std::make_shared<anet::nn::NetworkStruct>();

    SECTION("product factors must have matching batches")
    {
        anet::nn::NetworkBranch branch(
            "fusion", { { "a", "b" } }, 1, network_struct);
        anet::TensorDict state;
        state.Set("a", torch::ones({ 2, 3 }));
        state.Set("b", torch::ones({ 3, 1, 3 }));
        CHECK_THROWS_WITH(
            branch.Execute(state),
            Catch::Matchers::ContainsSubstring("NetworkBranch 'fusion'")
            && Catch::Matchers::ContainsSubstring("factor 'a'")
            && Catch::Matchers::ContainsSubstring("factor 'b'")
            && Catch::Matchers::ContainsSubstring("shape="));
    }

    SECTION("concat terms must have matching batches")
    {
        anet::nn::NetworkBranch branch(
            "concat", { { "a" }, { "b" } }, 1, network_struct);
        anet::TensorDict state;
        state.Set("a", torch::ones({ 2, 3 }));
        state.Set("b", torch::ones({ 3, 4 }));
        CHECK_THROWS_WITH(
            branch.Execute(state),
            Catch::Matchers::ContainsSubstring("NetworkBranch 'concat'")
            && Catch::Matchers::ContainsSubstring("batch size mismatch")
            && Catch::Matchers::ContainsSubstring("term_shapes="));
    }
}

TEST_CASE("Network bind rejects batch and out-of-range concat dimensions", "[nn][bind]")
{
    auto network_struct = std::make_shared<anet::nn::NetworkStruct>();
    for (const int64_t concat_dim : { int64_t{ 0 }, int64_t{ -2 }, int64_t{ 2 } }) {
        INFO("bind_concat_dim=" << concat_dim);
        anet::nn::NetworkBranch branch(
            "concat", { { "a" }, { "b" } }, concat_dim, network_struct);
        anet::TensorDict state;
        state.Set("a", torch::ones({ 2, 3 }));
        state.Set("b", torch::ones({ 2, 4 }));
        CHECK_THROWS_WITH(
            branch.Execute(state),
            Catch::Matchers::ContainsSubstring("NetworkBranch 'concat'")
            && Catch::Matchers::ContainsSubstring("value=" + std::to_string(concat_dim))
            && Catch::Matchers::ContainsSubstring("rank=2")
            && Catch::Matchers::ContainsSubstring("term_shapes="));
    }

    SECTION("concat dimension must exist in every term")
    {
        anet::nn::NetworkBranch branch(
            "concat", { { "a" }, { "b" } }, 2, network_struct);
        anet::TensorDict state;
        state.Set("a", torch::ones({ 2, 3, 4 }));
        state.Set("b", torch::ones({ 2, 5 }));
        CHECK_THROWS_WITH(
            branch.Execute(state),
            Catch::Matchers::ContainsSubstring("NetworkBranch 'concat'")
            && Catch::Matchers::ContainsSubstring("value=2")
            && Catch::Matchers::ContainsSubstring("term_rank=2")
            && Catch::Matchers::ContainsSubstring("term_shapes="));
    }
}

TEST_CASE("Network bind product factors participate in DAG validation", "[nn][bind]")
{
    anet::TensorSpec spec{
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
    };

    SECTION("factor dependency determines execution order")
    {
        anet::ConfigData config_data;
        config_data.Set("net.branch.[base].bind", "obs");
        config_data.Set("net.branch.[base].structure", "");
        config_data.Set("net.branch.[fusion].bind", "base * scale");
        config_data.Set("net.branch.[fusion].structure", "");
        config_data.Set("net.body.output.[features]", "fusion");
        const anet::nn::NetworkConfig config(config_data);
        auto body = anet::nn::NetworkBodyBuilder::Build(
            config, anet::TensorSpecMap{ { "obs", spec }, { "scale", spec } });
        REQUIRE(body->GetBranches().size() == 2);
        CHECK(body->GetBranches()[0]->GetName() == "base");
        CHECK(body->GetBranches()[1]->GetName() == "fusion");
    }

    SECTION("factor dependency cycles fail fast")
    {
        anet::ConfigData config_data;
        config_data.Set("net.branch.[a].bind", "obs * b");
        config_data.Set("net.branch.[a].structure", "");
        config_data.Set("net.branch.[b].bind", "a");
        config_data.Set("net.branch.[b].structure", "");
        const anet::nn::NetworkConfig config(config_data);
        CHECK_THROWS_WITH(
            anet::nn::NetworkBodyBuilder::Build(config, anet::TensorSpecMap{ { "obs", spec } }),
            Catch::Matchers::ContainsSubstring("Cycle detected"));
    }
}

TEST_CASE("BatchNorm2d runs in FP32 after BF16 autocast convolution", "[nn][batchnorm][bf16]")
{
    EnsureNNInitialized();

    std::vector<torch::Device> devices{ torch::Device(torch::kCPU) };
    if (torch::cuda::is_available()) {
        devices.emplace_back(torch::Device(torch::kCUDA, 0));
    }

    for (const auto& device : devices) {
        INFO("device=" << device.str());

        anet::ConfigData config_data;
        config_data.Set("net.block.[Conv].type", "Conv2d");
        config_data.Set("net.block.[Conv].conv.out_channels", 3);
        config_data.Set("net.block.[Conv].conv.kernel_size", 3);
        config_data.Set("net.block.[Conv].conv.stride", 1);
        config_data.Set("net.block.[Conv].conv.padding", 1);
        config_data.Set("net.block.[Conv].init.mode", "he");
        config_data.Set("net.block.[BN].type", "BatchNorm2d");
        config_data.Set("net.block.[BN].num_features", 3);
        config_data.Set("net.branch.[feature].bind", "obs");
        config_data.Set("net.branch.[feature].structure", "Conv > BN");
        config_data.Set("net.body.output.[feature]", "feature");

        anet::TensorSpec obs_spec;
        obs_spec.type = anet::SpaceType::Grid;
        obs_spec.shape = { 3, 4, 4 };
        obs_spec.dtype = torch::kFloat32;

        anet::TensorSpecMap input_specs;
        input_specs["obs"] = obs_spec;

        auto network_config = anet::nn::NetworkConfig(config_data);
        auto network = anet::nn::NetworkBuilder::BuildNetwork(
            network_config,
            input_specs,
            nullptr,
            device);

        anet::TensorDict input;
        input.Set("obs", torch::randn(
            { 2, 3, 4, 4 },
            torch::TensorOptions().dtype(torch::kFloat32).device(device)));

        anet::TensorDict output;
        {
            anet::Autocast autocast_guard(device, true, torch::kBFloat16);
            output = network->Forward(input);
        }

        REQUIRE(output.Contains("feature"));
        REQUIRE(output.At("feature").dtype() == torch::kFloat32);
        REQUIRE(output.At("feature").device().type() == device.type());
        CHECK_FALSE(at::autocast::is_autocast_enabled(device.type()));
    }
}

TEST_CASE("Norm modules can opt out of forced FP32 under BF16 input", "[nn][batchnorm][layernorm][layernorm2d][bf16]")
{
    EnsureNNInitialized();

    std::vector<torch::Device> devices{ torch::Device(torch::kCPU) };
    if (torch::cuda::is_available()) {
        devices.emplace_back(torch::Device(torch::kCUDA, 0));
    }

    for (const auto& device : devices) {
        INFO("device=" << device.str());

        {
            auto module = MakeBatchNorm2dTestModule(3);
            module->eval();
            auto input = torch::randn({ 2, 3, 4, 4 },
                torch::TensorOptions().dtype(torch::kBFloat16).device(device));
            auto output = module->Forward(input);
            REQUIRE(output.dtype() == torch::kFloat32);
            REQUIRE(GetNamedParameter(*module, "bn.weight").dtype() == torch::kFloat32);
            REQUIRE(module->GetCurrentConfigData().Get("force_fp32") == "true");
        }

        {
            auto module = MakeBatchNorm2dTestModule(3, false);
            module->eval();
            auto input = torch::randn({ 2, 3, 4, 4 },
                torch::TensorOptions().dtype(torch::kBFloat16).device(device));
            auto output = module->Forward(input);
            REQUIRE(output.dtype() == torch::kBFloat16);
            REQUIRE(GetNamedParameter(*module, "bn.weight").dtype() == torch::kBFloat16);
            REQUIRE(module->GetCurrentConfigData().Get("force_fp32") == "false");
        }

        {
            auto module = MakeLayerNormTestModule(4);
            auto input = torch::randn({ 2, 4 },
                torch::TensorOptions().dtype(torch::kBFloat16).device(device));
            auto output = module->Forward(input);
            REQUIRE(output.dtype() == torch::kFloat32);
            REQUIRE(GetNamedParameter(*module, "ln.weight").dtype() == torch::kFloat32);
            REQUIRE(module->GetCurrentConfigData().Get("force_fp32") == "true");
        }

        {
            auto module = MakeLayerNormTestModule(4, std::nullopt, false);
            auto input = torch::randn({ 2, 4 },
                torch::TensorOptions().dtype(torch::kBFloat16).device(device));
            auto output = module->Forward(input);
            REQUIRE(output.dtype() == torch::kBFloat16);
            REQUIRE(GetNamedParameter(*module, "ln.weight").dtype() == torch::kBFloat16);
            REQUIRE(module->GetCurrentConfigData().Get("force_fp32") == "false");
        }

        {
            auto module = MakeLayerNorm2dTestModule(3);
            auto input = torch::randn({ 2, 3, 4, 4 },
                torch::TensorOptions().dtype(torch::kBFloat16).device(device));
            auto output = module->Forward(input);
            REQUIRE(output.dtype() == torch::kFloat32);
            REQUIRE(GetNamedParameter(*module, "weight").dtype() == torch::kFloat32);
            REQUIRE(module->GetCurrentConfigData().Get("force_fp32") == "true");
        }

        {
            auto module = MakeLayerNorm2dTestModule(3, 1.0e-6, false);
            auto input = torch::randn({ 2, 3, 4, 4 },
                torch::TensorOptions().dtype(torch::kBFloat16).device(device));
            auto output = module->Forward(input);
            REQUIRE(output.dtype() == torch::kBFloat16);
            REQUIRE(GetNamedParameter(*module, "weight").dtype() == torch::kBFloat16);
            REQUIRE(module->GetCurrentConfigData().Get("force_fp32") == "false");
        }
    }
}

TEST_CASE("Internal norm force_fp32 config reaches ResBlock and CNBlock modules", "[nn][resblock][cnblock][bf16]")
{
    if (!torch::cuda::is_available()) {
        SKIP("CUDA is not available.");
    }

    const auto device = torch::Device(torch::kCUDA, 0);

    {
        auto module = MakeResBlockTestModule(
            /*droppath_rate=*/0.0,
            /*dropout_rate=*/0.0,
            "batch",
            "post");
        auto input = torch::randn({ 2, 3, 8, 8 },
            torch::TensorOptions().dtype(torch::kFloat32).device(device));
        {
            anet::Autocast autocast_guard(device, true, torch::kBFloat16);
            (void)module->Forward(input);
        }

        REQUIRE(module->GetCurrentConfigData().Get("norm_force_fp32") == "true");
        REQUIRE(GetNamedParameter(*module, "norm1.bn.weight").dtype() == torch::kFloat32);
    }

    {
        auto module = MakeResBlockTestModule(
            /*droppath_rate=*/0.0,
            /*dropout_rate=*/0.0,
            "batch",
            "post",
            false);
        auto input = torch::randn({ 2, 3, 8, 8 },
            torch::TensorOptions().dtype(torch::kFloat32).device(device));
        {
            anet::Autocast autocast_guard(device, true, torch::kBFloat16);
            (void)module->Forward(input);
        }

        REQUIRE(module->GetCurrentConfigData().Get("norm_force_fp32") == "false");
        REQUIRE(GetNamedParameter(*module, "norm1.bn.weight").dtype() == torch::kBFloat16);
    }

    {
        auto module = MakeCNBlockTestModule();
        auto input = torch::randn({ 2, 3, 8, 8 },
            torch::TensorOptions().dtype(torch::kFloat32).device(device));
        {
            anet::Autocast autocast_guard(device, true, torch::kBFloat16);
            (void)module->Forward(input);
        }

        REQUIRE(module->GetCurrentConfigData().Get("norm_force_fp32") == "true");
        REQUIRE(GetNamedParameter(*module, "norm.weight").dtype() == torch::kFloat32);
    }

    {
        auto module = MakeCNBlockTestModule(
            /*channels=*/3,
            /*droppath_rate=*/0.0,
            /*layerscale_init=*/1.0e-6,
            /*norm_type=*/"layernorm2d",
            /*kernel_size=*/3,
            /*ffn_expand_ratio=*/2,
            /*constant_init=*/false,
            false);
        auto input = torch::randn({ 2, 3, 8, 8 },
            torch::TensorOptions().dtype(torch::kFloat32).device(device));
        {
            anet::Autocast autocast_guard(device, true, torch::kBFloat16);
            (void)module->Forward(input);
        }

        REQUIRE(module->GetCurrentConfigData().Get("norm_force_fp32") == "false");
        REQUIRE(GetNamedParameter(*module, "norm.weight").dtype() == torch::kBFloat16);
    }
}

TEST_CASE("Autocast refreshes cached weight casts between scopes", "[nn][autocast][bf16]")
{
    if (!torch::cuda::is_available()) {
        SKIP("CUDA is not available.");
    }

    auto device = torch::Device(torch::kCUDA, 0);
    auto linear = torch::nn::Linear(torch::nn::LinearOptions(2, 1).bias(false));
    linear->to(device, torch::kFloat32);

    auto input = torch::ones({ 1, 2 }, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    torch::Tensor before_update;
    {
        torch::NoGradGuard no_grad;
        linear->weight.fill_(1.0f);
    }
    {
        anet::Autocast autocast_guard(device, true, torch::kBFloat16);
        before_update = linear->forward(input).detach().to(torch::kFloat32);
    }

    torch::Tensor after_update;
    {
        torch::NoGradGuard no_grad;
        linear->weight.fill_(2.0f);
    }
    {
        anet::Autocast autocast_guard(device, true, torch::kBFloat16);
        after_update = linear->forward(input).detach().to(torch::kFloat32);
    }

    REQUIRE(before_update.defined());
    REQUIRE(after_update.defined());
    const float before_value = before_update.cpu().item<float>();
    const float after_value = after_update.cpu().item<float>();
    REQUIRE(before_value == Catch::Approx(2.0f));
    REQUIRE(after_value == Catch::Approx(4.0f));
}

TEST_CASE("Autocast restores nested enabled and disabled scopes", "[nn][autocast][bf16]")
{
    std::vector<torch::DeviceType> device_types{ torch::kCPU };
    if (torch::cuda::is_available()) {
        device_types.push_back(torch::kCUDA);
    }

    for (const auto device_type : device_types) {
        INFO("device_type=" << c10::DeviceTypeName(device_type));
        const bool original_enabled = at::autocast::is_autocast_enabled(device_type);
        const auto original_dtype = at::autocast::get_autocast_dtype(device_type);

        {
            anet::Autocast outer(torch::Device(device_type), true, torch::kBFloat16);
            REQUIRE(at::autocast::is_autocast_enabled(device_type));
            REQUIRE(at::autocast::get_autocast_dtype(device_type) == torch::kBFloat16);

            {
                anet::Autocast inner(torch::Device(device_type), false, torch::kFloat32);
                CHECK_FALSE(at::autocast::is_autocast_enabled(device_type));
                REQUIRE(at::autocast::get_autocast_dtype(device_type) == torch::kFloat32);
            }

            REQUIRE(at::autocast::is_autocast_enabled(device_type));
            REQUIRE(at::autocast::get_autocast_dtype(device_type) == torch::kBFloat16);
        }

        REQUIRE(at::autocast::is_autocast_enabled(device_type) == original_enabled);
        REQUIRE(at::autocast::get_autocast_dtype(device_type) == original_dtype);
    }
}

std::shared_ptr<anet::nn::Network> MakeTraceTestNetwork()
{
    anet::TensorSpec obs_spec;
    obs_spec.type = anet::SpaceType::Grid;
    obs_spec.shape = { 1, 2, 2 };
    obs_spec.dtype = torch::kFloat32;

    anet::TensorSpecMap input_specs;
    input_specs["obs"] = obs_spec;

    auto block = std::make_shared<anet::nn::NetworkBlock>(
        "Conv2d_0",
        std::make_shared<TraceTestModule>());
    auto network_struct = std::make_shared<anet::nn::NetworkStruct>(
        std::vector<std::shared_ptr<anet::nn::NetworkBlock>>{ block });
    auto branch = std::make_shared<anet::nn::NetworkBranch>(
        "feature",
        std::vector<std::vector<std::string>>{ { "obs" } },
        1,
        network_struct);

    anet::nn::NetworkConfig network_config;
    network_config.output_keys["feature"] = "feature";

    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{ branch },
        input_specs,
        std::vector<std::string>{ "obs" },
        network_config.output_keys);

    return std::make_shared<anet::nn::Network>(
        network_config,
        input_specs,
        nullptr,
        body,
        std::make_shared<DotTestHead>());
}

class CapturingBackend final : public anet::IBackend {
public:
    void Open(const std::filesystem::path&, const std::string&) override {}
    void WriteJsonl(const anet::json& obj) override { records.push_back(obj); }
    void Flush() override {}

    std::vector<anet::json> records;
};

std::string ReadTextFile(const std::filesystem::path& path)
{
    std::ifstream ifs(path);
    return std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
}

} // namespace

TEST_CASE("LabelData AddAttr uses default and explicit float precision", "[graphviz][label]")
{
    anet::graphviz::LabelData label;
    label.SetTitle("precision")
        .AddAttr("default", 1.23456)
        .AddAttr("p1", 1.23456, 1);

    const auto dot_label = label.ToGraphvizLabel();
    CHECK(Contains(dot_label, ">1.235</TD>"));
    CHECK(Contains(dot_label, ">1.2</TD>"));
}

TEST_CASE("LabelData SetText emits plain quoted label", "[graphviz][label]")
{
    anet::graphviz::LabelData label;
    label.SetText("vector_feature");

    const auto dot_label = label.ToGraphvizLabel();
    CHECK(dot_label == "\"vector_feature\"");
    CHECK_FALSE(Contains(dot_label, "TABLE"));
    CHECK_FALSE(Contains(dot_label, "TD"));
}

TEST_CASE("GAP2D averages spatial dimensions", "[nn]")
{
    static const bool initialized = [] {
        anet::nn::InitNN();
        return true;
    }();
    (void)initialized;

    auto factory = anet::nn::NetworkModuleRepository::Instance().GetFactory("GAP2D");
    auto module = factory->CreateModule(anet::ConfigData{}, anet::nn::ModuleContext{});

    auto input = torch::arange(0, 48, torch::kFloat32).reshape({ 2, 3, 2, 4 });
    auto output = module->Forward(input);
    auto expected = input.mean(/*dims=*/{ 2, 3 }, /*keepdim=*/false);

    CHECK(output.sizes() == torch::IntArrayRef({ 2, 3 }));
    CHECK(torch::allclose(output, expected));
}

TEST_CASE("MaxPool2d pools spatial dimensions from config", "[nn][pool]")
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("pool.kernel_size", 2);
    config_data.Set("pool.stride", 2);
    config_data.Set("pool.padding", 0);

    auto factory = anet::nn::NetworkModuleRepository::Instance().GetFactory("MaxPool2d");
    auto module = factory->CreateModule(config_data, anet::nn::ModuleContext{});

    auto input = torch::arange(0, 16, torch::kFloat32).reshape({ 1, 1, 4, 4 });
    auto output = module->Forward(input);
    auto expected = torch::tensor({ 5.0f, 7.0f, 13.0f, 15.0f }).reshape({ 1, 1, 2, 2 });
    CHECK(torch::equal(output, expected));

    auto default_module = factory->CreateModule(anet::ConfigData{}, anet::nn::ModuleContext{});
    anet::ConfigData current = default_module->GetCurrentConfigData();
    CHECK(current.Get("kernel_size") == "3");
    CHECK(current.Get("stride") == "2");
    CHECK(current.Get("padding") == "1");
    CHECK(current.Get("dilation") == "1");
    CHECK(current.Get("ceil_mode") == "false");
}

TEST_CASE("NetworkBuilder builds MaxPool2d and GAP2D pipeline", "[nn][pool]")
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("net.block.[Pool].type", "MaxPool2d");
    config_data.Set("net.block.[Pool].pool.kernel_size", 2);
    config_data.Set("net.block.[Pool].pool.stride", 2);
    config_data.Set("net.block.[Pool].pool.padding", 0);
    config_data.Set("net.block.[GAP].type", "GAP2D");
    config_data.Set("net.branch.[feature].bind", "obs");
    config_data.Set("net.branch.[feature].structure", "Pool > GAP");
    config_data.Set("net.body.output.[feature]", "feature");

    anet::TensorSpec obs_spec;
    obs_spec.type = anet::SpaceType::Grid;
    obs_spec.shape = { 1, 4, 4 };
    obs_spec.dtype = torch::kFloat32;

    anet::TensorSpecMap input_specs;
    input_specs["obs"] = obs_spec;

    auto network_config = anet::nn::NetworkConfig(config_data);
    auto network = anet::nn::NetworkBuilder::BuildNetwork(network_config, input_specs, nullptr, torch::Device(torch::kCPU));

    anet::TensorDict input;
    input.Set("obs", torch::arange(0, 16, torch::kFloat32).reshape({ 1, 1, 4, 4 }));
    auto output = network->Forward(input);
    auto expected = torch::tensor({ 10.0f }).reshape({ 1, 1 });
    CHECK(torch::equal(output.At("feature"), expected));
}

TEST_CASE("Network config profile expands linear markers by branch order", "[nn][config_profile]")
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    const std::vector<std::string> block_names = { "A", "B", "C", "D" };
    for (const std::string& block_name : block_names) {
        const std::string prefix = "net.block.[" + block_name + "].";
        config_data.Set(prefix + "type", std::string("Dropout"));
        config_data.Set(prefix + "dropout_rate", std::string("@dp"));
    }
    config_data.Set("net.config_profile.[dp].type", std::string("linear"));
    config_data.Set("net.config_profile.[dp].start", 0.0);
    config_data.Set("net.config_profile.[dp].end", 0.1);
    config_data.Set("net.branch.[feature].bind", std::string("obs"));
    config_data.Set("net.branch.[feature].structure", std::string("A(*3) > B(*3) > C(*9) > D(*3)"));
    config_data.Set("net.body.output.[feature]", std::string("feature"));

    anet::nn::NetworkConfig config(config_data);
    const auto json = config.ToJson();
    CHECK(json.at("config_profiles").at("dp").at("type") == "linear");
    CHECK(json.at("config_profiles").at("dp").at("start") == 0.0);
    CHECK(json.at("config_profiles").at("dp").at("end") == 0.1);

    auto network_struct = anet::nn::NetworkStructBuilder::Build(
        config, config.branches.at("feature").structure_str);
    const auto rates = GetDropoutRates(network_struct);

    REQUIRE(rates.size() == 18);
    for (size_t i = 0; i < rates.size(); ++i) {
        const double expected = 0.1 * static_cast<double>(i) / static_cast<double>(rates.size() - 1);
        INFO("i=" << i);
        CHECK(rates[i] == Catch::Approx(expected).margin(1.0e-12));
    }
    CHECK(rates[3] > rates[2]);
    CHECK(rates[6] > rates[5]);
    CHECK(rates[15] > rates[14]);
}

TEST_CASE("NetworkConfig merges global catalogs into an agent-owned net tree", "[nn][config][agent_net]")
{
    anet::ConfigData config_data;
    config_data.Set("net.block.[Drop].type", "Dropout");
    config_data.Set("net.block.[Drop].dropout_rate", 0.25);
    config_data.Set("net.config_profile.[dp].type", "linear");
    config_data.Set("net.config_profile.[dp].start", 0.0);
    config_data.Set("net.config_profile.[dp].end", 0.5);
    config_data.Set("DefaultDQNAgent.net.block.[Drop].dropout_rate", 0.4);
    config_data.Set("DefaultDQNAgent.net.config_profile.[dp].end", 0.75);
    config_data.Set("DefaultDQNAgent.net.branch.[feature].bind", "obs");
    config_data.Set("DefaultDQNAgent.net.branch.[feature].structure", "Drop(*2)");
    config_data.Set("DefaultDQNAgent.net.body.output.[features]", "feature");

    const anet::nn::NetworkConfig config(config_data, "DefaultDQNAgent.net");

    REQUIRE(config.block_configs.contains("Drop"));
    CHECK(config.block_configs.at("Drop").type == "Dropout");
    CHECK(config.block_configs.at("Drop").config_data.Get("dropout_rate") == "0.4");
    REQUIRE(config.config_profiles.contains("dp"));
    CHECK(config.config_profiles.at("dp").start == Catch::Approx(0.0));
    CHECK(config.config_profiles.at("dp").end == Catch::Approx(0.75));
    REQUIRE(config.branches.contains("feature"));
    CHECK(config.output_keys.at("features") == "feature");
}

TEST_CASE("Network config profile returns start for a single marker", "[nn][config_profile]")
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("net.block.[One].type", std::string("Dropout"));
    config_data.Set("net.block.[One].dropout_rate", std::string("@single"));
    config_data.Set("net.config_profile.[single].type", std::string("linear"));
    config_data.Set("net.config_profile.[single].start", 0.25);
    config_data.Set("net.config_profile.[single].end", 0.75);
    config_data.Set("net.branch.[feature].bind", std::string("obs"));
    config_data.Set("net.branch.[feature].structure", std::string("One"));

    anet::nn::NetworkConfig config(config_data);
    auto network_struct = anet::nn::NetworkStructBuilder::Build(
        config, config.branches.at("feature").structure_str);
    const auto rates = GetDropoutRates(network_struct);

    REQUIRE(rates.size() == 1);
    CHECK(rates[0] == Catch::Approx(0.25).margin(1.0e-12));
}

TEST_CASE("Network config profile supports branch-local overrides", "[nn][config_profile]")
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("net.block.[Drop].type", std::string("Dropout"));
    config_data.Set("net.block.[Drop].dropout_rate", std::string("@dp"));
    config_data.Set("net.config_profile.[dp].type", std::string("linear"));
    config_data.Set("net.config_profile.[dp].start", 0.25);
    config_data.Set("net.config_profile.[dp].end", 0.5);
    config_data.Set("net.branch.[base].bind", std::string("obs"));
    config_data.Set("net.branch.[base].structure", std::string("Drop(*2)"));
    config_data.Set("net.branch.[wide].bind", std::string("obs"));
    config_data.Set("net.branch.[wide].structure", std::string("Drop(*2)"));
    config_data.Set("net.branch.[wide].config_profile.[dp].end", 0.75);

    anet::nn::NetworkConfig config(config_data);
    const auto json = config.ToJson();
    CHECK(json.at("branches").at("wide").at("config_profiles").at("dp").at("end") == 0.75);

    anet::TensorSpecMap input_specs;
    input_specs["obs"] = MakeConfigProfileVectorSpec();
    auto body = anet::nn::NetworkBodyBuilder::Build(config, input_specs);
    REQUIRE(body);

    const auto base_rates = GetDropoutRates(GetBranchNetworkStruct(body, "base"));
    REQUIRE(base_rates.size() == 2);
    CHECK(base_rates[0] == Catch::Approx(0.25).margin(1.0e-12));
    CHECK(base_rates[1] == Catch::Approx(0.5).margin(1.0e-12));

    const auto wide_rates = GetDropoutRates(GetBranchNetworkStruct(body, "wide"));
    REQUIRE(wide_rates.size() == 2);
    CHECK(wide_rates[0] == Catch::Approx(0.25).margin(1.0e-12));
    CHECK(wide_rates[1] == Catch::Approx(0.75).margin(1.0e-12));
}

TEST_CASE("Network config profile leaves marker-free branches on original config", "[nn][config_profile]")
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("net.block.[Drop].type", std::string("Dropout"));
    config_data.Set("net.block.[Drop].dropout_rate", 0.25);
    config_data.Set("net.config_profile.[unused].type", std::string("linear"));
    config_data.Set("net.config_profile.[unused].start", 0.0);
    config_data.Set("net.config_profile.[unused].end", 0.5);
    config_data.Set("net.branch.[feature].bind", std::string("obs"));
    config_data.Set("net.branch.[feature].structure", std::string("Drop"));

    anet::nn::NetworkConfig config(config_data);
    anet::TensorSpecMap input_specs;
    input_specs["obs"] = MakeConfigProfileVectorSpec();
    auto body = anet::nn::NetworkBodyBuilder::Build(config, input_specs);
    REQUIRE(body);

    auto network_struct = anet::nn::NetworkStructBuilder::Build(
        config, config.branches.at("feature").structure_str);
    const auto rates = GetDropoutRates(network_struct);

    REQUIRE(rates.size() == 1);
    CHECK(rates[0] == Catch::Approx(0.25).margin(1.0e-12));
}

TEST_CASE("Network config profile rejects invalid marker settings", "[nn][config_profile]")
{
    EnsureNNInitialized();

    anet::ConfigData undefined_group;
    undefined_group.Set("net.block.[Drop].type", std::string("Dropout"));
    undefined_group.Set("net.block.[Drop].dropout_rate", std::string("@missing"));
    undefined_group.Set("net.branch.[feature].bind", std::string("obs"));
    undefined_group.Set("net.branch.[feature].structure", std::string("Drop"));
    anet::nn::NetworkConfig undefined_config(undefined_group);
    CHECK_THROWS(anet::nn::NetworkStructBuilder::Build(
        undefined_config, undefined_config.branches.at("feature").structure_str));

    anet::ConfigData missing_end;
    missing_end.Set("net.config_profile.[dp].type", std::string("linear"));
    missing_end.Set("net.config_profile.[dp].start", 0.0);
    CHECK_THROWS(anet::nn::NetworkConfig(missing_end));

    anet::ConfigData legacy_minmax;
    legacy_minmax.Set("net.config_profile.[dp].type", std::string("linear"));
    legacy_minmax.Set("net.config_profile.[dp].min", 0.0);
    legacy_minmax.Set("net.config_profile.[dp].max", 1.0);
    CHECK_THROWS(anet::nn::NetworkConfig(legacy_minmax));

    anet::ConfigData unknown_type;
    unknown_type.Set("net.config_profile.[dp].type", std::string("cosine"));
    unknown_type.Set("net.config_profile.[dp].end", 1.0);
    CHECK_THROWS(anet::nn::NetworkConfig(unknown_type));
}

TEST_CASE("Network config profile expands same group independently per branch", "[nn][config_profile]")
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("net.block.[A].type", std::string("Dropout"));
    config_data.Set("net.block.[A].dropout_rate", std::string("@dp"));
    config_data.Set("net.block.[B].type", std::string("Dropout"));
    config_data.Set("net.block.[B].dropout_rate", std::string("@dp"));
    config_data.Set("net.config_profile.[dp].type", std::string("linear"));
    config_data.Set("net.config_profile.[dp].start", 0.0);
    config_data.Set("net.config_profile.[dp].end", 0.5);
    config_data.Set("net.branch.[feature_a].bind", std::string("obs_a"));
    config_data.Set("net.branch.[feature_a].structure", std::string("A(*2)"));
    config_data.Set("net.branch.[feature_b].bind", std::string("obs_b"));
    config_data.Set("net.branch.[feature_b].structure", std::string("B(*3)"));

    anet::nn::NetworkConfig config(config_data);
    anet::TensorSpecMap input_specs;
    input_specs["obs_a"] = MakeConfigProfileVectorSpec();
    input_specs["obs_b"] = MakeConfigProfileVectorSpec();

    auto body = anet::nn::NetworkBodyBuilder::Build(config, input_specs);
    REQUIRE(body);

    const auto a_rates = GetDropoutRates(GetBranchNetworkStruct(body, "feature_a"));
    REQUIRE(a_rates.size() == 2);
    CHECK(a_rates[0] == Catch::Approx(0.0).margin(1.0e-12));
    CHECK(a_rates[1] == Catch::Approx(0.5).margin(1.0e-12));

    const auto b_rates = GetDropoutRates(GetBranchNetworkStruct(body, "feature_b"));
    REQUIRE(b_rates.size() == 3);
    CHECK(b_rates[0] == Catch::Approx(0.0).margin(1.0e-12));
    CHECK(b_rates[1] == Catch::Approx(0.25).margin(1.0e-12));
    CHECK(b_rates[2] == Catch::Approx(0.5).margin(1.0e-12));
}

TEST_CASE("Network SoftCopyTo blends parameters and floating buffers", "[nn][soft-copy]")
{
    auto source = MakeSoftCopyTestNetwork(/*base=*/10.0f, /*counter=*/42);
    auto target = MakeSoftCopyTestNetwork(/*base=*/2.0f, /*counter=*/7);
    const double tau = 0.25;

    std::map<std::string, torch::Tensor> before_params;
    std::map<std::string, torch::Tensor> before_buffers;
    for (const auto& kv : target->named_parameters(true)) {
        before_params.emplace(kv.key(), kv.value().detach().clone());
    }
    for (const auto& kv : target->named_buffers(true)) {
        before_buffers.emplace(kv.key(), kv.value().detach().clone());
    }

    source->SoftCopyTo(*target, tau);

    auto src_params = source->named_parameters(true);
    auto dst_params = target->named_parameters(true);
    REQUIRE(src_params.size() == dst_params.size());
    for (const auto& kv : src_params) {
        INFO("param name=" << kv.key());
        const auto& before = before_params.at(kv.key());
        auto expected = before + (kv.value().detach() - before) * tau;
        CHECK(torch::equal(dst_params[kv.key()].detach(), expected));
    }

    auto src_buffers = source->named_buffers(true);
    auto dst_buffers = target->named_buffers(true);
    REQUIRE(src_buffers.size() == dst_buffers.size());
    for (const auto& kv : src_buffers) {
        INFO("buffer name=" << kv.key());
        if (kv.value().is_floating_point()) {
            const auto& before = before_buffers.at(kv.key());
            auto expected = before + (kv.value().detach() - before) * tau;
            CHECK(torch::equal(dst_buffers[kv.key()].detach(), expected));
        } else {
            CHECK(torch::equal(dst_buffers[kv.key()].detach(), kv.value().detach()));
        }
    }
}

TEST_CASE("Network forward trace captures branch-prefixed visual layers", "[nn][trace]")
{
    auto network = MakeTraceTestNetwork();
    auto input_tensor = torch::arange(8, torch::kFloat32).view({ 2, 1, 2, 2 });
    anet::TensorDict input{ { "obs", input_tensor } };
    anet::TensorDict trace;

    auto out = network->Forward(input, [&trace](std::string_view key, const torch::Tensor& tensor) {
        trace.Set(std::string(key), tensor.detach().clone());
    });

    REQUIRE(out.Contains("feature"));
    REQUIRE(trace.Contains("feature/00_Input"));
    REQUIRE(trace.Contains("feature/01_Conv2d_0"));
    CHECK(torch::equal(trace.At("feature/00_Input"), input_tensor));
    CHECK(torch::equal(trace.At("feature/01_Conv2d_0"), input_tensor + 1.0f));
}

TEST_CASE("Body-only Network exposes forward TensorDictFunction", "[nn][function]")
{
    auto network = MakeBodyOnlyDotTestNetwork();
    auto input_tensor = torch::tensor({ { 1.0f, 2.0f }, { 3.0f, 4.0f } });
    anet::TensorDict input{ { "obs", input_tensor } };

    auto forward_func = network->GetTensorDictFunction("forward");
    REQUIRE(forward_func.has_value());
    CHECK_FALSE(network->GetTensorDictFunction("feature").has_value());

    auto direct_out = network->Forward(input);
    auto func_out = (*forward_func)(input);

    REQUIRE(direct_out.Contains("feature"));
    REQUIRE(func_out.Contains("feature"));
    CHECK(torch::equal(func_out.At("feature"), direct_out.At("feature")));
}

TEST_CASE("WeightInitializer string modes match torch initializers", "[nn][init]")
{
    anet::nn::WeightInitConfig config;
    CheckWeightInitMatchesDirect(config, 1701, [](torch::nn::Linear& expected) {
        torch::nn::init::xavier_uniform_(expected->weight);
        torch::nn::init::constant_(expected->bias, 0.0);
    });

    config.mode = "he";
    config.nonlinearity = "relu";
    CheckWeightInitMatchesDirect(config, 1702, [](torch::nn::Linear& expected) {
        torch::nn::init::kaiming_normal_(expected->weight, 0.0, torch::kFanOut, torch::kReLU);
        torch::nn::init::constant_(expected->bias, 0.0);
    });

    config.mode = "orthogonal";
    config.nonlinearity = "linear";
    config.manual_gain = 0.0;
    CheckWeightInitMatchesDirect(config, 1703, [](torch::nn::Linear& expected) {
        const double gain = torch::nn::init::calculate_gain(torch::kLinear);
        torch::nn::init::orthogonal_(expected->weight, gain);
        torch::nn::init::constant_(expected->bias, 0.0);
    });

    config.mode = "constant";
    config.constant_val = 0.25;
    CheckWeightInitMatchesDirect(config, 1704, [](torch::nn::Linear& expected) {
        torch::nn::init::constant_(expected->weight, 0.25);
        torch::nn::init::constant_(expected->bias, 0.25);
    });
}

TEST_CASE("WeightInitializer preserves default mode and rejects unknown modes", "[nn][init]")
{
    anet::nn::WeightInitConfig config;
    config.mode = "default";
    auto layer = MakeWeightInitTestLinear();
    FillWeightInitTestLinear(layer, 0.25, -0.75);
    anet::nn::WeightInitializer::Initialize(layer, config);
    CheckTensorClose(torch::full_like(layer->weight, 0.25), layer->weight);
    CheckTensorClose(torch::full_like(layer->bias, -0.75), layer->bias);

    for (const std::string mode : { "unknown", "2" }) {
        INFO("mode=" << mode);
        config.mode = mode;
        auto invalid_layer = MakeWeightInitTestLinear();
        CHECK_THROWS(anet::nn::WeightInitializer::Initialize(invalid_layer, config));
    }
}

TEST_CASE("WeightInitializer trunc_normal clamps range and zeros bias", "[nn][init]")
{
    anet::nn::WeightInitConfig config;
    config.mode = "trunc_normal";
    config.trunc_std = 0.02;
    config.trunc_a = -0.04;
    config.trunc_b = 0.04;

    auto layer = torch::nn::Linear(torch::nn::LinearOptions(512, 512).bias(true));
    torch::manual_seed(1710);
    anet::nn::WeightInitializer::Initialize(layer, config);

    CHECK(layer->weight.min().item<double>() >= config.trunc_a - 1.0e-7);
    CHECK(layer->weight.max().item<double>() <= config.trunc_b + 1.0e-7);
    const double actual_std = layer->weight.std().item<double>();
    CHECK(actual_std > 0.01);
    CHECK(actual_std < 0.025);
    CheckTensorClose(torch::zeros_like(layer->bias), layer->bias);

    config.trunc_std = 0.0;
    CHECK_THROWS(anet::nn::WeightInitializer::Initialize(layer, config));

    config.trunc_std = 0.02;
    config.trunc_a = 0.1;
    config.trunc_b = 0.1;
    CHECK_THROWS(anet::nn::WeightInitializer::Initialize(layer, config));
}

TEST_CASE("LayerNorm exposes backward-compatible eps config", "[nn][layernorm]")
{
    auto default_module = MakeLayerNormTestModule(4);
    auto default_config = default_module->GetCurrentConfigData();
    CHECK(std::stod(default_config.Get("eps")) == Catch::Approx(1.0e-5));

    auto explicit_module = MakeLayerNormTestModule(4, 1.0e-6);
    auto explicit_config = explicit_module->GetCurrentConfigData();
    CHECK(std::stod(explicit_config.Get("eps")) == Catch::Approx(1.0e-6));

    CHECK_THROWS(MakeLayerNormTestModule(0));
}

TEST_CASE("LayerNorm2d normalizes channel axis only", "[nn][layernorm2d]")
{
    auto module = MakeLayerNorm2dTestModule(3, /*eps=*/0.0);
    torch::Tensor input = torch::arange(0, 24, torch::kFloat32).reshape({ 2, 3, 2, 2 });
    torch::Tensor output = module->Forward(input);

    CHECK(output.sizes() == input.sizes());

    auto mean = input.mean({ 1 }, /*keepdim=*/true);
    auto variance = (input - mean).pow(2).mean({ 1 }, /*keepdim=*/true);
    auto expected = (input - mean) / torch::sqrt(variance);
    CheckTensorClose(expected, output);
    CheckTensorClose(torch::zeros_like(output.mean({ 1 }, /*keepdim=*/true)), output.mean({ 1 }, /*keepdim=*/true));
    CheckTensorClose(torch::ones_like(output.pow(2).mean({ 1 }, /*keepdim=*/true)), output.pow(2).mean({ 1 }, /*keepdim=*/true));

    auto cd = module->GetCurrentConfigData();
    CHECK(cd.Get("num_channels") == "3");
    CHECK(std::stod(cd.Get("eps")) == Catch::Approx(0.0));
}

TEST_CASE("LayerNorm2d rejects invalid config and input", "[nn][layernorm2d]")
{
    CHECK_THROWS(MakeLayerNorm2dTestModule(0));

    auto module = MakeLayerNorm2dTestModule(3);
    CHECK_THROWS(module->Forward(torch::randn({ 2, 3, 4 }, torch::kFloat32)));
    CHECK_THROWS(module->Forward(torch::randn({ 2, 4, 2, 2 }, torch::kFloat32)));
}

TEST_CASE("CNBlock preserves shape and exposes config", "[nn][cnblock]")
{
    auto module = MakeCNBlockTestModule(
        /*channels=*/8,
        /*droppath_rate=*/0.25,
        /*layerscale_init=*/1.0e-6,
        /*norm_type=*/"layernorm2d",
        /*kernel_size=*/3,
        /*ffn_expand_ratio=*/2);

    torch::Tensor input = torch::randn({ 2, 8, 6, 6 }, torch::kFloat32);
    module->eval();
    torch::Tensor output = module->Forward(input);
    CHECK(output.sizes() == input.sizes());

    anet::ConfigData cd = module->GetCurrentConfigData();
    CHECK(cd.Get("channels") == "8");
    CHECK(cd.Get("kernel_size") == "3");
    CHECK(cd.Get("ffn_expand_ratio") == "2");
    CHECK(std::stod(cd.Get("layerscale_init")) == Catch::Approx(1.0e-6));
    CHECK(std::stod(cd.Get("droppath_rate")) == Catch::Approx(0.25));
    CHECK(cd.Get("norm_type") == "layernorm2d");
    CHECK(cd.Get("in_channels") == "8");

    auto gamma = GetNamedParameter(*module, "gamma");
    CheckTensorClose(torch::full_like(gamma, 1.0e-6), gamma);

    auto no_layerscale = MakeCNBlockTestModule(/*channels=*/8, /*droppath_rate=*/0.0, /*layerscale_init=*/0.0);
    (void)no_layerscale->Forward(input);
    CHECK_FALSE(HasKey(no_layerscale->named_parameters(true), "gamma"));
}

TEST_CASE("CNBlock DropPath is eval no-op and train can return shortcut", "[nn][cnblock]")
{
    auto baseline = MakeCNBlockTestModule(/*channels=*/8, /*droppath_rate=*/0.0, /*layerscale_init=*/1.0);
    auto droppath = MakeCNBlockTestModule(/*channels=*/8, /*droppath_rate=*/0.5, /*layerscale_init=*/1.0);
    torch::Tensor input = torch::randn({ 2, 8, 6, 6 }, torch::kFloat32);

    baseline->eval();
    droppath->eval();
    (void)baseline->Forward(input);
    (void)droppath->Forward(input);
    CopyModuleState(*baseline, *droppath);

    CheckTensorClose(baseline->Forward(input), droppath->Forward(input));

    auto shortcut_only = MakeCNBlockTestModule(/*channels=*/8, /*droppath_rate=*/0.999, /*layerscale_init=*/1.0);
    shortcut_only->train();
    (void)shortcut_only->Forward(input);

    bool saw_shortcut = false;
    for (int seed = 1500; seed < 1510; ++seed) {
        torch::manual_seed(seed);
        torch::Tensor output = shortcut_only->Forward(input);
        if (torch::allclose(input, output)) {
            saw_shortcut = true;
            break;
        }
    }
    CHECK(saw_shortcut);
}

TEST_CASE("CNBlock CPU fallback supports backward", "[nn][cnblock]")
{
    auto module = MakeCNBlockTestModule(
        /*channels=*/8,
        /*droppath_rate=*/0.0,
        /*layerscale_init=*/1.0,
        /*norm_type=*/"layernorm2d",
        /*kernel_size=*/3,
        /*ffn_expand_ratio=*/2);
    module->train();

    torch::Tensor input = torch::randn({ 2, 8, 5, 5 }, torch::kFloat32);
    input.requires_grad_(true);
    torch::Tensor loss = module->Forward(input).pow(2).mean();
    loss.backward();

    CHECK(input.grad().defined());
    CHECK(GetNamedParameter(*module, "gamma").grad().defined());
    CHECK(GetNamedParameter(*module, "dwconv.weight").grad().defined());
    CHECK(GetNamedParameter(*module, "pwconv1.weight").grad().defined());
    CHECK(GetNamedParameter(*module, "pwconv2.weight").grad().defined());
}

TEST_CASE("CNBlock supports disabling norm and validates invalid settings", "[nn][cnblock]")
{
    torch::Tensor input = torch::randn({ 2, 8, 5, 5 }, torch::kFloat32) + 0.5;

    auto with_norm = MakeCNBlockTestModule(
        /*channels=*/8,
        /*droppath_rate=*/0.0,
        /*layerscale_init=*/1.0,
        /*norm_type=*/"layernorm2d",
        /*kernel_size=*/3,
        /*ffn_expand_ratio=*/2,
        /*constant_init=*/true);
    auto without_norm = MakeCNBlockTestModule(
        /*channels=*/8,
        /*droppath_rate=*/0.0,
        /*layerscale_init=*/1.0,
        /*norm_type=*/"none",
        /*kernel_size=*/3,
        /*ffn_expand_ratio=*/2,
        /*constant_init=*/true);

    with_norm->eval();
    without_norm->eval();
    torch::Tensor norm_output = with_norm->Forward(input);
    torch::Tensor no_norm_output = without_norm->Forward(input);
    CHECK_FALSE(torch::allclose(norm_output, no_norm_output));
    CHECK(without_norm->GetCurrentConfigData().Get("norm_type") == "none");

    CHECK_THROWS(MakeCNBlockTestModule(/*channels=*/0));
    CHECK_THROWS(MakeCNBlockTestModule(/*channels=*/3, /*droppath_rate=*/1.0));
    CHECK_THROWS(MakeCNBlockTestModule(
        /*channels=*/3,
        /*droppath_rate=*/0.0,
        /*layerscale_init=*/1.0e-6,
        /*norm_type=*/"layernorm2d",
        /*kernel_size=*/2));
    CHECK_THROWS(MakeCNBlockTestModule(
        /*channels=*/3,
        /*droppath_rate=*/0.0,
        /*layerscale_init=*/1.0e-6,
        /*norm_type=*/"layernorm2d",
        /*kernel_size=*/3,
        /*ffn_expand_ratio=*/0));
    CHECK_THROWS(MakeCNBlockTestModule(
        /*channels=*/3,
        /*droppath_rate=*/0.0,
        /*layerscale_init=*/1.0e-6,
        /*norm_type=*/"batch"));

    auto channel_mismatch = MakeCNBlockTestModule(/*channels=*/3);
    CHECK_THROWS(channel_mismatch->Forward(torch::randn({ 2, 4, 5, 5 }, torch::kFloat32)));
}

TEST_CASE("Network dot view emits structure by default and configurable details", "[nn][dot]")
{
    auto network = MakeDotTestNetwork();

    const auto structure_dot = network->MakeGraphViz(anet::nn::NetworkGraphVizConfig{})->ToDotString();
    CHECK(Contains(structure_dot, "rankdir=\"LR\""));
    CHECK(Contains(structure_dot, "\"block_feature.Scale_0.0\" [shape=plain"));
    CHECK(Contains(structure_dot, "\"input_obs\" -> \"branch_feature\""));
    CHECK(Contains(structure_dot, "label=\"obs\""));
    CHECK(Contains(structure_dot, "\"branch_feature\" -> \"block_feature.Scale_0.0\""));
    CHECK(Contains(structure_dot, "\"output_feature\" -> \"head\""));
    CHECK(Contains(structure_dot, "<B>Head</B>"));
    CHECK_FALSE(Contains(structure_dot, "bind</TD>"));
    CHECK_FALSE(Contains(structure_dot, "outputs"));
    CHECK_FALSE(Contains(structure_dot, "head_output"));
    CHECK_FALSE(Contains(structure_dot, "scale"));
    CHECK_FALSE(Contains(structure_dot, "params"));
    CHECK_FALSE(Contains(structure_dot, "dtype"));
    CHECK_FALSE(Contains(structure_dot, "auto_format"));
    CHECK_FALSE(Contains(structure_dot, "raw_keys"));

    anet::nn::NetworkGraphVizConfig detail_config;
    detail_config.show_param_shapes = true;
    detail_config.show_param_count = true;
    detail_config.show_tensor_specs = true;
    detail_config.layout = "TB";
    detail_config.float_precision = 2;

    const auto detail_dot = network->MakeGraphViz(detail_config)->ToDotString();
    CHECK(Contains(detail_dot, "rankdir=\"TB\""));
    CHECK(Contains(detail_dot, "\"block_feature.Scale_0.0\" [shape=plain"));
    CHECK(Contains(detail_dot, "shape"));
    CHECK(Contains(detail_dot, "[2]"));
    CHECK(Contains(detail_dot, "dtype"));
    CHECK(Contains(detail_dot, "scale"));
    CHECK(Contains(detail_dot, "1.23"));
    CHECK(Contains(detail_dot, "activation_mode"));
    CHECK(Contains(detail_dot, ">pre</TD>"));
    CHECK(Contains(detail_dot, "params"));
    CHECK(Contains(detail_dot, "6"));
    CHECK_FALSE(Contains(detail_dot, "auto_format"));
    CHECK_FALSE(Contains(detail_dot, "raw_keys"));

    detail_config.show_branch_config = true;
    const auto branch_detail_dot = network->MakeGraphViz(detail_config)->ToDotString();
    CHECK(Contains(branch_detail_dot, "auto_format"));
    CHECK(Contains(branch_detail_dot, ">false</TD>"));
    CHECK(Contains(branch_detail_dot, "raw_keys"));
    CHECK(Contains(branch_detail_dot, ">obs</TD>"));
    CHECK(Contains(branch_detail_dot, "bind_concat_dim"));
    CHECK(Contains(branch_detail_dot, ">1</TD>"));
}

TEST_CASE("Network dot view emits an edge for every bind product factor", "[nn][bind][dot]")
{
    anet::ConfigData config_data;
    config_data.Set("net.branch.[fusion].bind", "features * tau_embedding");
    config_data.Set("net.branch.[fusion].structure", "");
    config_data.Set("net.body.output.[features]", "fusion");
    const anet::nn::NetworkConfig config(config_data);

    anet::TensorSpecMap input_specs;
    input_specs["features"] = anet::TensorSpec{
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
    };
    input_specs["tau_embedding"] = anet::TensorSpec{
        .type = anet::SpaceType::Vector,
        .shape = { 3, 2 },
        .dtype = torch::kFloat32,
    };
    auto network = anet::nn::NetworkBuilder::BuildNetwork(
        config, input_specs, nullptr, torch::Device(torch::kCPU));

    anet::nn::NetworkGraphVizConfig viz_config;
    viz_config.show_branch_config = true;
    const auto dot = network->MakeGraphViz(viz_config)->ToDotString();
    CHECK(Contains(dot, "\"input_features\" -> \"branch_fusion\""));
    CHECK(Contains(dot, "label=\"features\""));
    CHECK(Contains(dot, "\"input_tau_embedding\" -> \"branch_fusion\""));
    CHECK(Contains(dot, "label=\"tau_embedding\""));
    CHECK(Contains(dot, "bind_concat_dim"));
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

    dqn::DefaultDQNAgentConfig dqn_config(dqn_data);
    CHECK(dqn_config.nn_viz.show_param_shapes);
    CHECK(dqn_config.nn_viz.show_branch_config);
    CHECK(dqn_config.nn_viz.show_head_info);
    CHECK_FALSE(dqn_config.nn_viz.show_param_count);
    CHECK(dqn_config.nn_viz.float_precision == 5);

    anet::ConfigData muzero_data;
    muzero_data.Set("MuZeroAgent.nn_viz.show_tensor_specs", "true");
    muzero_data.Set("MuZeroAgent.nn_viz.show_branch_config", "true");
    muzero_data.Set("MuZeroAgent.nn_viz.show_head_info", "true");
    muzero_data.Set("MuZeroAgent.nn_dot.show_param_count", "true");
    muzero::MuZeroAgentConfig muzero_config(muzero_data);
    CHECK(muzero_config.nn_viz.show_tensor_specs);
    CHECK(muzero_config.nn_viz.show_branch_config);
    CHECK(muzero_config.nn_viz.show_head_info);
    CHECK_FALSE(muzero_config.nn_viz.show_param_count);

    anet::ConfigData invalid_data;
    invalid_data.Set("DefaultDQNAgent.nn_viz.layout", "BT");
    CHECK_THROWS(dqn::DefaultDQNAgentConfig(invalid_data));
}

TEST_CASE("DropPath applies sample-wise stochastic depth", "[nn][droppath]")
{
    torch::Tensor input = torch::ones({ 8, 3, 2, 2 }, torch::kFloat32);

    CHECK(torch::equal(anet::nn::DropPath(input, 0.5, /*training=*/false), input));
    CHECK(torch::equal(anet::nn::DropPath(input, 0.0, /*training=*/true), input));

    torch::Tensor output;
    bool found_mixed_mask = false;
    for (int seed = 1200; seed < 1230; ++seed) {
        torch::manual_seed(seed);
        output = anet::nn::DropPath(input, 0.5, /*training=*/true);

        bool saw_dropped = false;
        bool saw_kept = false;
        for (int64_t n = 0; n < output.size(0); ++n) {
            torch::Tensor sample = output[n];
            saw_dropped = saw_dropped || sample.eq(0.0).all().item<bool>();
            saw_kept = saw_kept || torch::allclose(sample, torch::full_like(sample, 2.0));
        }
        if (saw_dropped && saw_kept) {
            found_mixed_mask = true;
            break;
        }
    }

    REQUIRE(found_mixed_mask);
    CHECK(output.sizes() == input.sizes());
    for (int64_t n = 0; n < output.size(0); ++n) {
        torch::Tensor sample = output[n];
        const bool dropped = sample.eq(0.0).all().item<bool>();
        const bool kept = torch::allclose(sample, torch::full_like(sample, 2.0));
        CHECK((dropped || kept));
    }
}

TEST_CASE("DropoutModule uses dropout_rate config only", "[nn][dropout]")
{
    auto module = MakeDropoutTestModule(0.5);
    CHECK(module->GetCurrentConfigData().Get("dropout_rate") == "0.5");

    torch::Tensor input = torch::ones({ 64 }, torch::kFloat32);
    module->eval();
    CHECK(torch::equal(module->Forward(input), input));

    module->train();
    torch::Tensor output;
    bool saw_dropout = false;
    for (int seed = 1300; seed < 1310; ++seed) {
        torch::manual_seed(seed);
        output = module->Forward(input);
        if (!torch::equal(output, input)) {
            saw_dropout = true;
            break;
        }
    }
    CHECK(saw_dropout);
    CHECK(output.eq(0.0).any().item<bool>());

    auto old_key_module = MakeDropoutTestModule(0.9, /*set_old_p=*/true);
    old_key_module->train();
    torch::manual_seed(1301);
    CHECK(torch::equal(old_key_module->Forward(input), input));
    CHECK(old_key_module->GetCurrentConfigData().Get("dropout_rate") == "0");

    CHECK_THROWS(MakeDropoutTestModule(1.0));
}

TEST_CASE("ResBlock exposes dropout config and validates rates", "[nn][resblock][dropout]")
{
    auto module = MakeResBlockTestModule(/*droppath_rate=*/0.25, /*dropout_rate=*/0.125);
    anet::ConfigData cd = module->GetCurrentConfigData();
    CHECK(cd.Get("droppath_rate") == "0.25");
    CHECK(cd.Get("dropout_rate") == "0.125");

    CHECK_THROWS(MakeResBlockTestModule(/*droppath_rate=*/1.0, /*dropout_rate=*/0.0));
    CHECK_THROWS(MakeResBlockTestModule(/*droppath_rate=*/0.0, /*dropout_rate=*/-0.1));

    anet::test::LogCaptureGuard logs;
    auto warn_module = MakeResBlockTestModule(/*droppath_rate=*/0.0, /*dropout_rate=*/0.1, "batch");
    (void)warn_module;
    logs.Flush();
    CHECK(anet::test::HasRecordContaining(
        logs.Records(),
        wxLOG_Warning,
        { "ResBlock dropout_rate", "key=res.dropout_rate", "value=0.1", "BatchNorm", "droppath_rate" }));
}

TEST_CASE("ResBlock DropPath and Dropout2d affect training only", "[nn][resblock][dropout]")
{
    auto baseline = MakeResBlockTestModule(/*droppath_rate=*/0.0, /*dropout_rate=*/0.0);
    auto droppath = MakeResBlockTestModule(/*droppath_rate=*/0.5, /*dropout_rate=*/0.0);

    torch::Tensor input = torch::rand({ 2, 3, 8, 8 }, torch::kFloat32) + 0.1;
    baseline->eval();
    droppath->eval();
    (void)baseline->Forward(input);
    (void)droppath->Forward(input);
    CopyModuleState(*baseline, *droppath);

    torch::Tensor expected = baseline->Forward(input);
    torch::Tensor actual = droppath->Forward(input);
    CheckTensorClose(expected, actual);

    auto shortcut_only = MakeResBlockTestModule(/*droppath_rate=*/0.999999, /*dropout_rate=*/0.0);
    shortcut_only->train();
    (void)shortcut_only->Forward(input);
    torch::manual_seed(1401);
    torch::Tensor shortcut_output = shortcut_only->Forward(input);
    CheckTensorClose(input, shortcut_output);

    auto dropout2d_module = MakeResBlockTestModule(/*droppath_rate=*/0.0, /*dropout_rate=*/0.95);
    dropout2d_module->eval();
    torch::Tensor eval_output = dropout2d_module->Forward(input).detach();
    dropout2d_module->train();

    bool saw_train_difference = false;
    for (int seed = 1402; seed < 1412; ++seed) {
        torch::manual_seed(seed);
        torch::Tensor train_output = dropout2d_module->Forward(input).detach();
        if (!torch::allclose(eval_output, train_output)) {
            saw_train_difference = true;
            break;
        }
    }
    CHECK(saw_train_difference);
}

TEST_CASE("TransformerEncoder dropout config is eval no-op", "[nn][transformer][dropout]")
{
    torch::Tensor input = torch::randn({ 2, 5, 16 }, torch::kFloat32);

    for (bool use_sdpa : { true, false }) {
        for (bool norm_first : { true, false }) {
            INFO("use_sdpa=" << use_sdpa << " norm_first=" << norm_first);
            auto baseline = MakeTransformerTestModule(use_sdpa, norm_first);
            auto dropout = MakeTransformerTestModule(
                use_sdpa,
                norm_first,
                /*hidden_dropout_rate=*/0.5,
                /*attn_dropout_rate=*/0.4,
                /*droppath_rate=*/0.5);
            CopyModuleState(*baseline, *dropout);
            baseline->eval();
            dropout->eval();

            CheckTensorClose(baseline->Forward(input), dropout->Forward(input));

            anet::ConfigData cd = dropout->GetCurrentConfigData();
            CHECK(cd.Get("hidden_dropout_rate") == "0.5");
            CHECK(cd.Get("attn_dropout_rate") == "0.4");
            CHECK(cd.Get("droppath_rate") == "0.5");
        }
    }
}

TEST_CASE("TransformerEncoder dropout changes train output", "[nn][transformer][dropout]")
{
    torch::Tensor input = torch::randn({ 2, 5, 16 }, torch::kFloat32);

    for (bool use_sdpa : { true, false }) {
        for (bool norm_first : { true, false }) {
            INFO("use_sdpa=" << use_sdpa << " norm_first=" << norm_first);
            auto module = MakeTransformerTestModule(
                use_sdpa,
                norm_first,
                /*hidden_dropout_rate=*/0.6,
                /*attn_dropout_rate=*/0.0,
                /*droppath_rate=*/0.5);
            module->train();

            torch::manual_seed(1501);
            torch::Tensor first = module->Forward(input).detach();
            torch::Tensor second = module->Forward(input).detach();
            CHECK_FALSE(torch::allclose(first, second));
        }
    }
}

TEST_CASE("SdpaSelfAttention matches legacy MHA and manual reference", "[nn][transformer][sdpa]")
{
    torch::manual_seed(1201);
    torch::nn::MultiheadAttention mha(torch::nn::MultiheadAttentionOptions(16, 4));
    torch::Tensor x = torch::randn({ 2, 5, 16 }, torch::TensorOptions().dtype(torch::kFloat32));

    torch::Tensor sdpa_output = anet::nn::SdpaSelfAttention(mha, x);
    torch::Tensor legacy_output = LegacyMhaSelfAttention(mha, x);
    torch::Tensor manual_output = ManualScaledDotProductSelfAttention(mha, x);

    CheckTensorClose(legacy_output, sdpa_output);
    CheckTensorClose(manual_output, sdpa_output);
}

TEST_CASE("SdpaSelfAttention rejects unsupported MHA options", "[nn][transformer][sdpa]")
{
    torch::nn::MultiheadAttentionOptions options(16, 4);
    options.add_bias_kv(true);
    torch::nn::MultiheadAttention mha(options);
    torch::Tensor x = torch::randn({ 2, 5, 16 }, torch::TensorOptions().dtype(torch::kFloat32));

    CHECK_THROWS(anet::nn::SdpaSelfAttention(mha, x));
}

TEST_CASE("TransformerEncoder SDPA path matches legacy MHA path", "[nn][transformer][sdpa]")
{
    std::vector<torch::Device> devices{ torch::Device(torch::kCPU) };
    if (torch::cuda::is_available()) {
        devices.emplace_back(torch::Device(torch::kCUDA, 0));
    }

    for (const auto& device : devices) {
        for (bool norm_first : { true, false }) {
            INFO("device=" << device.str() << " norm_first=" << norm_first);
            CheckTransformerSdpaMatchesLegacy(device, norm_first);
        }
    }
}

TEST_CASE("TransformerEncoder SDPA preserves self_attn checkpoint names", "[nn][transformer][sdpa][serialize]")
{
    torch::manual_seed(1202);
    auto save_module = MakeTransformerTestModule(/*use_sdpa=*/true, /*norm_first=*/true);
    auto loaded_module = MakeTransformerTestModule(/*use_sdpa=*/true, /*norm_first=*/true);
    torch::Tensor input = torch::randn({ 2, 5, 16 }, torch::TensorOptions().dtype(torch::kFloat32));

    torch::Tensor before = save_module->Forward(input).detach().clone();
    auto params = save_module->named_parameters(true);
    CHECK(HasKey(params, "layer_0.self_attn.in_proj_weight"));
    CHECK(HasKey(params, "layer_0.self_attn.in_proj_bias"));
    CHECK(HasKey(params, "layer_0.self_attn.out_proj.weight"));
    CHECK(HasKey(params, "layer_0.self_attn.out_proj.bias"));

    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    std::shared_ptr<torch::nn::Module> save_base = save_module;
    torch::save(save_base, buffer);

    buffer.seekg(0);
    std::shared_ptr<torch::nn::Module> loaded_base = loaded_module;
    torch::load(loaded_base, buffer);

    torch::Tensor after = loaded_module->Forward(input).detach();
    CheckTensorClose(before, after);

    auto loaded_params = loaded_module->named_parameters(true);
    CHECK(HasKey(loaded_params, "layer_0.self_attn.in_proj_weight"));
    CHECK(HasKey(loaded_params, "layer_0.self_attn.in_proj_bias"));
    CHECK(HasKey(loaded_params, "layer_0.self_attn.out_proj.weight"));
    CHECK(HasKey(loaded_params, "layer_0.self_attn.out_proj.bias"));
}

TEST_CASE("FusedAdamW matches AdamW on deterministic gradients", "[nn][optimizer]")
{
    std::vector<torch::Device> devices{ torch::Device(torch::kCPU) };
    if (torch::cuda::is_available()) {
        devices.emplace_back(torch::Device(torch::kCUDA, 0));
    }

    for (const auto& device : devices) {
        for (double weight_decay : { 0.0, 1.0e-2 }) {
            INFO("device=" << device.str() << " weight_decay=" << weight_decay);
            auto adam_params = MakeAdamWTestParams(device);
            auto fused_params = CloneAdamWTestParams(adam_params);

            auto options = torch::optim::AdamWOptions(1.0e-2).weight_decay(weight_decay).eps(1.0e-8);
            torch::optim::AdamW adam(adam_params, options);
            anet::FusedAdamW fused(fused_params, options);

            for (int step = 0; step < 10; ++step) {
                RunAdamWTestStep(adam, adam_params, step);
                RunAdamWTestStep(fused, fused_params, step);
            }

            CheckAdamWParamsAndStateClose(adam_params, adam, fused_params, fused);
        }
    }
}

TEST_CASE("FusedAdamW checkpoint round-trip rebuilds device step tensors", "[nn][optimizer][serialize]")
{
    auto device = torch::Device(torch::kCPU);
    auto continuous_params = MakeAdamWTestParams(device);
    auto save_params = CloneAdamWTestParams(continuous_params);
    auto options = torch::optim::AdamWOptions(1.0e-2).weight_decay(1.0e-2).eps(1.0e-8);
    anet::FusedAdamW continuous_optimizer(continuous_params, options);
    anet::FusedAdamW save_optimizer(save_params, options);

    for (int step = 0; step < 3; ++step) {
        RunAdamWTestStep(continuous_optimizer, continuous_params, step);
        RunAdamWTestStep(save_optimizer, save_params, step);
    }

    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    torch::optim::Optimizer& save_base = save_optimizer;
    torch::save(save_base, buffer);

    auto loaded_params = CloneAdamWTestParams(save_params);
    anet::FusedAdamW loaded_optimizer(loaded_params, options);
    buffer.seekg(0);
    torch::optim::Optimizer& loaded_base = loaded_optimizer;
    torch::load(loaded_base, buffer);

    for (int step = 3; step < 8; ++step) {
        RunAdamWTestStep(continuous_optimizer, continuous_params, step);
        RunAdamWTestStep(loaded_optimizer, loaded_params, step);
    }

    CheckAdamWParamsAndStateClose(continuous_params, continuous_optimizer, loaded_params, loaded_optimizer);
}

TEST_CASE("FusedAdamW loads AdamW checkpoint and continues updates", "[nn][optimizer][serialize]")
{
    auto device = torch::Device(torch::kCPU);
    auto continuous_params = MakeAdamWTestParams(device);
    auto save_params = CloneAdamWTestParams(continuous_params);
    auto options = torch::optim::AdamWOptions(1.0e-2).weight_decay(1.0e-2).eps(1.0e-8);
    torch::optim::AdamW continuous_optimizer(continuous_params, options);
    torch::optim::AdamW save_optimizer(save_params, options);

    for (int step = 0; step < 3; ++step) {
        RunAdamWTestStep(continuous_optimizer, continuous_params, step);
        RunAdamWTestStep(save_optimizer, save_params, step);
    }

    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
    torch::optim::Optimizer& save_base = save_optimizer;
    torch::save(save_base, buffer);

    auto fused_params = CloneAdamWTestParams(save_params);
    anet::FusedAdamW fused_optimizer(fused_params, options);
    buffer.seekg(0);
    torch::optim::Optimizer& fused_base = fused_optimizer;
    torch::load(fused_base, buffer);

    for (int step = 3; step < 8; ++step) {
        RunAdamWTestStep(continuous_optimizer, continuous_params, step);
        RunAdamWTestStep(fused_optimizer, fused_params, step);
    }

    CheckAdamWParamsAndStateClose(continuous_params, continuous_optimizer, fused_params, fused_optimizer);
}

TEST_CASE("GradScaler skips step and backs off scale when unscale finds inf", "[nn][optimizer][amp]")
{
    auto param = torch::ones({ 2 }, torch::TensorOptions().dtype(torch::kFloat32));
    param.requires_grad_(true);
    torch::optim::SGD optimizer({ param }, torch::optim::SGDOptions(0.1));
    anet::GradScaler scaler(8.0, 2.0, 0.5, 2);

    auto inf = torch::full_like(param, std::numeric_limits<float>::infinity());
    auto loss = (param * inf).sum();
    scaler.Scale(loss).backward();

    scaler.Unscale_(optimizer);
    auto before = param.detach().clone();
    scaler.Step(optimizer);
    CHECK(torch::allclose(param.detach(), before));

    scaler.Update();
    CHECK(scaler.Scale(torch::ones({})).item<float>() == Catch::Approx(4.0f));
}

TEST_CASE("Foreach gradient helpers match manual norm and clipping", "[nn][optimizer]")
{
    auto grad0 = torch::tensor({ 3.0f, 4.0f });
    auto grad1 = torch::tensor({ 1.0f, 2.0f, 2.0f });
    std::vector<torch::Tensor> grads{ grad0.clone(), grad1.clone() };

    auto expected_norm = (grad0.pow(2).sum() + grad1.pow(2).sum()).sqrt();
    auto actual_norm = anet::ForeachGradNorm(grads);
    CHECK(torch::allclose(actual_norm, expected_norm));

    const float tau = 5.0f;
    auto scale = (torch::full({}, tau) / (expected_norm + 1e-6)).clamp_max(1.0);
    anet::ForeachClipGradNorm_(grads, actual_norm, tau);

    CHECK(torch::allclose(grads[0], grad0 * scale));
    CHECK(torch::allclose(grads[1], grad1 * scale));
}

TEST_CASE("MetricsLogger writes step-less GraphViz dot file", "[metrics][dot]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "anet-core-nn-dot-test";
    std::filesystem::remove_all(root);

    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<CapturingBackend>();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "nn_viz_test";
    anet::MetricsLogger::Init(std::move(backend), logger_config, root);

    auto network = MakeDotTestNetwork();
    auto graph = network->MakeGraphViz(anet::nn::NetworkGraphVizConfig{});
    anet::MetricsLogger::Instance()->Log("net:detail", *graph);

    const auto dot_path = root / "runs" / "nn_viz_test" / "dot" / "net-detail.dot";
    REQUIRE(std::filesystem::exists(dot_path));
    CHECK(Contains(ReadTextFile(dot_path), "digraph \"Network\""));

    anet::MetricsLogger::Reset();
    std::filesystem::remove_all(root);
}
