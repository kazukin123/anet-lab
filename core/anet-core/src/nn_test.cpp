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

namespace {
// テストの既定経路も、製品と同じ統合factoryで生成する。
std::shared_ptr<anet::nn::NetworkForwardExecutor> MakeDefaultExecutor(
    const std::shared_ptr<anet::nn::Network>& network)
{
    return network->CreateForwardExecutor(anet::nn::NetworkForwardConfig::GetDefault(),
        anet::nn::NetworkForwardClock::kTransientRows, 0);
}

// テスト用の生データから選択項目だけを取り出し、公開 constructor へ渡す。
anet::ConfigData SelectForwardConfigData(const anet::ConfigData& data, const std::string& prefix)
{
    anet::ConfigData selected;
    for (const auto& [key, value] : data.Map()) {
        if (key.starts_with(prefix + ".")) selected.Set(key.substr(prefix.size() + 1), value);
    }
    return selected;
}


anet::ConfigData WithDefaultForwardConfig(anet::ConfigData data = {})
{
    anet::nn::InitNN();
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
namespace muzero = anet::rl::muzero_proto;

static_assert(!std::is_copy_constructible_v<anet::Autocast>);
static_assert(!std::is_move_constructible_v<anet::Autocast>);
static_assert(!std::is_constructible_v<anet::Autocast, torch::DeviceType, bool, torch::ScalarType>);
static_assert(std::is_constructible_v<anet::Autocast, torch::Device, bool, torch::ScalarType>);

bool Contains(const std::string& text, const std::string& pattern)
{
    return text.find(pattern) != std::string::npos;
}

size_t CountProfileOperator(const std::function<void()>& operation, const std::string& operator_name)
{
    std::ostringstream output;
    {
        torch::autograd::profiler::RecordProfile profile(output);
        operation();
    }

    const std::string trace = output.str();
    const std::string needle = "\"name\": \"" + operator_name + "\"";
    size_t count = 0;
    for (size_t offset = 0; (offset = trace.find(needle, offset)) != std::string::npos; offset += needle.size()) {
        ++count;
    }
    return count;
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

std::shared_ptr<anet::nn::ModuleRandomSource> MakeTestModuleRandomSource()
{
    return std::make_shared<anet::nn::ModuleRandomSource>(0);
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

    torch::Tensor sdpa_output = sdpa_module->Forward(sdpa_input, anet::nn::NetworkForwardContext::GetDefault());
    torch::Tensor legacy_output = legacy_module->Forward(legacy_input, anet::nn::NetworkForwardContext::GetDefault());

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

    torch::Tensor Forward(torch::Tensor input, const anet::nn::NetworkForwardContext& context) override
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
    torch::Tensor Forward(torch::Tensor input, const anet::nn::NetworkForwardContext& context) override
    {
        return input + 1.0f;
    }

    bool IsConv2dVisualizable() const override { return true; }
};

class DotTestHead final : public anet::nn::NetworkHead {
public:
    anet::TensorDict Forward(const anet::TensorDict& feature_dict, const anet::nn::NetworkForwardContext& context) override
    {
        return feature_dict;
    }
};

class ParameterNormTestModule final : public anet::nn::NetworkModule {
public:
    explicit ParameterNormTestModule(
        const torch::Tensor& parameter,
        std::shared_ptr<int> forward_count = nullptr)
        : forward_count_(std::move(forward_count))
    {
        parameter_ = register_parameter("parameter", parameter.clone());
    }

    torch::Tensor Forward(torch::Tensor input, const anet::nn::NetworkForwardContext& context) override
    {
        if (forward_count_) ++*forward_count_;
        return input;
    }

private:
    torch::Tensor parameter_;
    std::shared_ptr<int> forward_count_;
};

class ParameterNormTestHead final : public anet::nn::NetworkHead {
public:
    explicit ParameterNormTestHead(const torch::Tensor& parameter)
    {
        parameter_ = register_parameter("parameter", parameter.clone());
    }

    anet::TensorDict Forward(const anet::TensorDict& feature_dict, const anet::nn::NetworkForwardContext& context) override
    {
        return feature_dict;
    }

private:
    torch::Tensor parameter_;
};

class ParameterNormAffineTestModule final : public anet::nn::NetworkModule {
public:
    ParameterNormAffineTestModule()
    {
        weight_ = register_parameter("weight", torch::tensor({ 3.0f, 4.0f }));
        bias_ = register_parameter("bias", torch::tensor({ 5.0f }));
        norm_ = register_module(
            "norm",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({ 2 })));
        torch::NoGradGuard no_grad;
        norm_->weight.copy_(torch::tensor({ 6.0f, 8.0f }));
        norm_->bias.copy_(torch::tensor({ 12.0f, 0.0f }));
    }

    torch::Tensor Forward(torch::Tensor input, const anet::nn::NetworkForwardContext& context) override
    {
        return norm_->forward(input);
    }

private:
    torch::Tensor weight_;
    torch::Tensor bias_;
    torch::nn::LayerNorm norm_{ nullptr };
};

std::shared_ptr<anet::nn::Network> MakeParameterNormTestNetwork(
    std::shared_ptr<int> forward_count = nullptr)
{
    const anet::TensorSpec spec{
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
    };
    const auto make_branch = [&forward_count](
        const std::string& name,
        const std::string& bind,
        const torch::Tensor& parameter) {
        auto block = std::make_shared<anet::nn::NetworkBlock>(
            name + "_block",
            std::make_shared<ParameterNormTestModule>(parameter, forward_count));
        auto structure = std::make_shared<anet::nn::NetworkStruct>(
            std::vector<std::shared_ptr<anet::nn::NetworkBlock>>{ block });
        return std::make_shared<anet::nn::NetworkBranch>(
            name,
            std::vector<std::vector<std::string>>{ { bind } },
            1,
            structure);
    };

    auto main_feature = make_branch("main_feature", "obs", torch::tensor({ 3.0f, 4.0f }));
    auto value_stream = make_branch("value_stream", "main_feature", torch::tensor({ 12.0f }));
    anet::nn::NetworkConfig config;
    config.output_keys["features"] = "value_stream";
    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{ main_feature, value_stream },
        anet::TensorSpecMap{ { "obs", spec } },
        std::vector<std::string>{},
        config.output_keys);
    return std::make_shared<anet::nn::Network>(
        config,
        anet::TensorSpecMap{ { "obs", spec } },
        nullptr,
        body,
        std::make_shared<ParameterNormTestHead>(torch::tensor({ 5.0f })));
}

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
    auto output = head->Forward(dummy_features, anet::nn::NetworkForwardContext::GetDefault());
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

    auto body = anet::nn::NetworkBodyBuilder::Build(config, input_specs, MakeTestModuleRandomSource());
    anet::TensorDict input;
    input.Set("features", torch::tensor({ { 2.0f, 3.0f }, { 5.0f, 7.0f } }));
    input.Set("tau_embedding", torch::tensor({
        { { 1.0f, 2.0f }, { 3.0f, 4.0f }, { 5.0f, 6.0f } },
        { { 2.0f, 3.0f }, { 4.0f, 5.0f }, { 6.0f, 7.0f } },
    }));

    const auto output = body->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
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
    const auto output = module->Forward(taus, anet::nn::NetworkForwardContext::GetDefault());

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
        module->Forward(torch::zeros({ 2, 3, 1 }), anet::nn::NetworkForwardContext::GetDefault()),
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
    (void)anet::nn::NetworkBodyBuilder::Build(config, input_specs, MakeTestModuleRandomSource());
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
    branch.Execute(state, anet::nn::NetworkForwardContext::GetDefault());

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
            branch.Execute(state, anet::nn::NetworkForwardContext::GetDefault()),
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
            branch.Execute(state, anet::nn::NetworkForwardContext::GetDefault()),
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
            branch.Execute(state, anet::nn::NetworkForwardContext::GetDefault()),
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
            branch.Execute(state, anet::nn::NetworkForwardContext::GetDefault()),
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
            config, anet::TensorSpecMap{ { "obs", spec }, { "scale", spec } },
            MakeTestModuleRandomSource());
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
            anet::nn::NetworkBodyBuilder::Build(
                config, anet::TensorSpecMap{ { "obs", spec } }, MakeTestModuleRandomSource()),
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
            0,
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
            auto output = module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
            REQUIRE(output.dtype() == torch::kFloat32);
            REQUIRE(GetNamedParameter(*module, "bn.weight").dtype() == torch::kFloat32);
            REQUIRE(module->GetCurrentConfigData().Get("force_fp32") == "true");
        }

        {
            auto module = MakeBatchNorm2dTestModule(3, false);
            module->eval();
            auto input = torch::randn({ 2, 3, 4, 4 },
                torch::TensorOptions().dtype(torch::kBFloat16).device(device));
            auto output = module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
            REQUIRE(output.dtype() == torch::kBFloat16);
            REQUIRE(GetNamedParameter(*module, "bn.weight").dtype() == torch::kBFloat16);
            REQUIRE(module->GetCurrentConfigData().Get("force_fp32") == "false");
        }

        {
            auto module = MakeLayerNormTestModule(4);
            auto input = torch::randn({ 2, 4 },
                torch::TensorOptions().dtype(torch::kBFloat16).device(device));
            auto output = module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
            REQUIRE(output.dtype() == torch::kFloat32);
            REQUIRE(GetNamedParameter(*module, "ln.weight").dtype() == torch::kFloat32);
            REQUIRE(module->GetCurrentConfigData().Get("force_fp32") == "true");
        }

        {
            auto module = MakeLayerNormTestModule(4, std::nullopt, false);
            auto input = torch::randn({ 2, 4 },
                torch::TensorOptions().dtype(torch::kBFloat16).device(device));
            auto output = module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
            REQUIRE(output.dtype() == torch::kBFloat16);
            REQUIRE(GetNamedParameter(*module, "ln.weight").dtype() == torch::kBFloat16);
            REQUIRE(module->GetCurrentConfigData().Get("force_fp32") == "false");
        }

        {
            auto module = MakeLayerNorm2dTestModule(3);
            auto input = torch::randn({ 2, 3, 4, 4 },
                torch::TensorOptions().dtype(torch::kBFloat16).device(device));
            auto output = module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
            REQUIRE(output.dtype() == torch::kFloat32);
            REQUIRE(GetNamedParameter(*module, "weight").dtype() == torch::kFloat32);
            REQUIRE(module->GetCurrentConfigData().Get("force_fp32") == "true");
        }

        {
            auto module = MakeLayerNorm2dTestModule(3, 1.0e-6, false);
            auto input = torch::randn({ 2, 3, 4, 4 },
                torch::TensorOptions().dtype(torch::kBFloat16).device(device));
            auto output = module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
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
            (void)module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
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
            (void)module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
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
            (void)module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
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
            (void)module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
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
    auto output = module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
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
    auto output = module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
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
    auto network = anet::nn::NetworkBuilder::BuildNetwork(
        network_config, input_specs, nullptr, 0, torch::Device(torch::kCPU));

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
        config, config.branches.at("feature").structure_str, MakeTestModuleRandomSource());
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
        config, config.branches.at("feature").structure_str, MakeTestModuleRandomSource());
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
    auto body = anet::nn::NetworkBodyBuilder::Build(config, input_specs, MakeTestModuleRandomSource());
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
    auto body = anet::nn::NetworkBodyBuilder::Build(config, input_specs, MakeTestModuleRandomSource());
    REQUIRE(body);

    auto network_struct = anet::nn::NetworkStructBuilder::Build(
        config, config.branches.at("feature").structure_str, MakeTestModuleRandomSource());
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
        undefined_config, undefined_config.branches.at("feature").structure_str,
        MakeTestModuleRandomSource()));

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

    auto body = anet::nn::NetworkBodyBuilder::Build(config, input_specs, MakeTestModuleRandomSource());
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
    torch::Tensor output = module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());

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
    CHECK_THROWS(module->Forward(torch::randn({ 2, 3, 4 }, torch::kFloat32), anet::nn::NetworkForwardContext::GetDefault()));
    CHECK_THROWS(module->Forward(torch::randn({ 2, 4, 2, 2 }, torch::kFloat32), anet::nn::NetworkForwardContext::GetDefault()));
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
    torch::Tensor output = module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
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
    (void)no_layerscale->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
    CHECK_FALSE(HasKey(no_layerscale->named_parameters(true), "gamma"));
}

TEST_CASE("CNBlock DropPath is eval no-op and train can return shortcut", "[nn][cnblock]")
{
    auto baseline = MakeCNBlockTestModule(/*channels=*/8, /*droppath_rate=*/0.0, /*layerscale_init=*/1.0);
    auto droppath = MakeCNBlockTestModule(/*channels=*/8, /*droppath_rate=*/0.5, /*layerscale_init=*/1.0);
    torch::Tensor input = torch::randn({ 2, 8, 6, 6 }, torch::kFloat32);

    baseline->eval();
    droppath->eval();
    (void)baseline->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
    (void)droppath->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
    CopyModuleState(*baseline, *droppath);

    CheckTensorClose(baseline->Forward(input, anet::nn::NetworkForwardContext::GetDefault()), droppath->Forward(input, anet::nn::NetworkForwardContext::GetDefault()));

    auto shortcut_only = MakeCNBlockTestModule(/*channels=*/8, /*droppath_rate=*/0.999, /*layerscale_init=*/1.0);
    shortcut_only->train();
    (void)shortcut_only->Forward(input, anet::nn::NetworkForwardContext::GetDefault());

    bool saw_shortcut = false;
    for (int seed = 1500; seed < 1510; ++seed) {
        torch::manual_seed(seed);
        torch::Tensor output = shortcut_only->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
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
    torch::Tensor loss = module->Forward(input, anet::nn::NetworkForwardContext::GetDefault()).pow(2).mean();
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
    torch::Tensor norm_output = with_norm->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
    torch::Tensor no_norm_output = without_norm->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
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
    CHECK_THROWS(channel_mismatch->Forward(torch::randn({ 2, 4, 5, 5 }, torch::kFloat32), anet::nn::NetworkForwardContext::GetDefault()));
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
        config, input_specs, nullptr, 0, torch::Device(torch::kCPU));

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

TEST_CASE("Network partial forward executes only the target dependency closure", "[nn][partial_forward][plasticity]")
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("net.branch.[base].bind", "obs");
    config_data.Set("net.branch.[base].structure", "");
    config_data.Set("net.branch.[target].bind", "base");
    config_data.Set("net.branch.[target].structure", "");
    config_data.Set("net.branch.[unrelated].bind", "aux");
    config_data.Set("net.branch.[unrelated].structure", "");
    config_data.Set("net.body.output.[features]", "target");

    const anet::TensorSpec spec{
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
    };
    auto network = anet::nn::NetworkBuilder::BuildNetwork(
        anet::nn::NetworkConfig(config_data),
        anet::TensorSpecMap{ { "obs", spec }, { "aux", spec } },
        nullptr,
        0,
        torch::Device(torch::kCPU));

    anet::TensorDict input;
    input.Set("obs", torch::tensor({ { 1.0f, 2.0f } }));
    input.Set("aux", torch::tensor({ { 3.0f, 4.0f } }));

    const auto state = network->ForwardUpTo(input, "target");

    CHECK(state.Contains("obs"));
    CHECK(state.Contains("aux"));
    CHECK(state.Contains("base"));
    CHECK(state.Contains("target"));
    CHECK_FALSE(state.Contains("unrelated"));
    const auto branch_names = network->GetBranchNames();
    CHECK(std::set<std::string>(branch_names.begin(), branch_names.end())
        == std::set<std::string>{ "base", "target", "unrelated" });
}

TEST_CASE("Linear spectral normalization is constructed from config", "[nn][spectral_norm]")
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("net.block.[Linear].type", "Linear");
    config_data.Set("net.block.[Linear].linear.out_features", 2);
    config_data.Set("net.block.[Linear].init.mode", "xavier");
    config_data.Set("net.block.[Linear].weight_norm.mode", "spectral");
    config_data.Set("net.branch.[feature].bind", "obs");
    config_data.Set("net.branch.[feature].structure", "Linear");
    config_data.Set("net.body.output.[feature]", "feature");

    const anet::TensorSpec spec{
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
    };
    auto network = anet::nn::NetworkBuilder::BuildNetwork(
        anet::nn::NetworkConfig(config_data),
        anet::TensorSpecMap{ { "obs", spec } },
        nullptr,
        65065,
        torch::Device(torch::kCPU));

    const auto entries = network->GetSpectralNormEntries();
    REQUIRE(entries.size() == 1);
    CHECK(entries.front().name == "feature.Linear_0.linear");
    CHECK(entries.front().u.dim() == 1);
    CHECK(entries.front().v.dim() == 1);
    bool has_sn_u_linear = false;
    bool has_sn_v_linear = false;
    for (const auto& item : network->named_buffers(true)) {
        has_sn_u_linear |= item.key().ends_with("sn_u_linear");
        has_sn_v_linear |= item.key().ends_with("sn_v_linear");
    }
    CHECK(has_sn_u_linear);
    CHECK(has_sn_v_linear);

    network->eval();
    anet::TensorDict input;
    const auto x = torch::tensor({ { 1.0f, -2.0f } });
    input.Set("obs", x);
    const auto output = network->Forward(input).At("feature");

    const auto& entry = entries.front();
    const auto sigma = anet::nn::ComputeSpectralSigma(
        entry.weight.reshape({ entry.weight.size(0), -1 }), entry.u, entry.v);
    torch::Tensor bias;
    for (const auto& item : network->named_parameters(true)) {
        if (item.key().ends_with("linear.bias")) bias = item.value();
    }
    REQUIRE(bias.defined());
    const auto expected = torch::nn::functional::linear(x, entry.weight / sigma, bias);
    CHECK(torch::allclose(output, expected, 1.0e-5, 1.0e-6));
}

TEST_CASE("Spectral normalization sigma and gradient match an analytic diagonal matrix", "[nn][spectral_norm][math]")
{
    auto weight = torch::tensor(
        { { 3.0f, 0.0f }, { 0.0f, 2.0f } },
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
    anet::nn::SpectralNormState state{
        .u = torch::tensor({ 1.0f, 0.0f }),
        .v = torch::tensor({ 1.0f, 0.0f }),
    };

    const auto sigma = anet::nn::ComputeSpectralSigma(weight, state.u, state.v);
    CHECK(sigma.scalar_type() == torch::kFloat32);
    CHECK(sigma.item<float>() == Catch::Approx(3.0f));

    const auto normalized = anet::nn::MakeSpectralNormalizedWeight(
        weight, anet::nn::WeightNormMode::kSpectral, state, /*update_state=*/false);
    normalized.sum().backward();
    const auto expected_gradient = torch::tensor(
        { { -2.0f / 9.0f, 1.0f / 3.0f }, { 1.0f / 3.0f, 1.0f / 3.0f } });
    CHECK(torch::allclose(weight.grad(), expected_gradient, 1.0e-6, 1.0e-6));
}

TEST_CASE("Spectral normalization gradient matches central difference", "[nn][spectral_norm][math]")
{
    auto weight = torch::tensor(
        { { 2.0f, 0.5f }, { -0.25f, 1.0f } },
        torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
    anet::nn::SpectralNormState state{
        .u = torch::tensor({ 1.0f, 0.0f }),
        .v = torch::tensor({ 1.0f, 0.0f }),
    };
    const auto loss = anet::nn::MakeSpectralNormalizedWeight(
        weight, anet::nn::WeightNormMode::kSpectral, state, /*update_state=*/false).square().sum();
    loss.backward();
    const float analytic = weight.grad()[0][1].item<float>();

    constexpr float kEps = 1.0e-3f;
    const auto evaluate = [&](float delta) {
        auto perturbed = weight.detach().clone();
        perturbed[0][1].add_(delta);
        return anet::nn::MakeSpectralNormalizedWeight(
            perturbed, anet::nn::WeightNormMode::kSpectral, state,
            /*update_state=*/false).square().sum().item<float>();
    };
    const float numeric = (evaluate(kEps) - evaluate(-kEps)) / (2.0f * kEps);
    CHECK(analytic == Catch::Approx(numeric).margin(2.0e-3f));
}

TEST_CASE("Spectral normalization computes sigma in FP32 with autocast enabled", "[nn][spectral_norm][autocast]")
{
    auto weight = torch::tensor(
        { { 3.0f, 0.0f }, { 0.0f, 2.0f } },
        torch::TensorOptions().dtype(torch::kBFloat16));
    const auto u = torch::tensor({ 1.0f, 0.0f });
    const auto v = torch::tensor({ 1.0f, 0.0f });

    anet::Autocast autocast(torch::Device(torch::kCPU), true, torch::kBFloat16);
    const auto sigma = anet::nn::ComputeSpectralSigma(weight, u, v);

    CHECK(sigma.scalar_type() == torch::kFloat32);
    CHECK(sigma.item<float>() == Catch::Approx(3.0f));
}

TEST_CASE("Spectral cap preserves zero initialization and activates after a nonzero transition", "[nn][spectral_norm][spectral_cap]")
{
    auto weight = torch::zeros(
        { 2, 2 }, torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true));
    anet::RandomGenerator rnd(65065);
    auto state = anet::nn::MakeSpectralNormState(
        weight, anet::nn::WeightNormMode::kSpectralCap, "zero", rnd);
    const auto u_before = state.u.clone();
    const auto v_before = state.v.clone();

    auto effective = anet::nn::MakeSpectralNormalizedWeight(
        weight, anet::nn::WeightNormMode::kSpectralCap, state, /*update_state=*/true);
    CHECK(torch::equal(effective, weight));
    CHECK(torch::equal(state.u, u_before));
    CHECK(torch::equal(state.v, v_before));
    effective.sum().backward();
    CHECK(torch::equal(weight.grad(), torch::ones_like(weight)));

    {
        torch::NoGradGuard no_grad;
        weight[0][0].fill_(2.0f);
    }
    effective = anet::nn::MakeSpectralNormalizedWeight(
        weight, anet::nn::WeightNormMode::kSpectralCap, state, /*update_state=*/true);
    const auto sigma = anet::nn::ComputeSpectralSigma(weight, state.u, state.v).abs();
    CHECK(sigma.item<float>() == Catch::Approx(2.0f).margin(1.0e-5f));
    CHECK(effective[0][0].item<float>() == Catch::Approx(1.0f).margin(1.0e-5f));
}

TEST_CASE("Spectral normalization rejects a degenerate spectral initialization", "[nn][spectral_norm][config]")
{
    auto weight = torch::zeros({ 2, 2 }, torch::kFloat32);
    anet::RandomGenerator rnd(65065);

    CHECK_THROWS_WITH(
        anet::nn::MakeSpectralNormState(
            weight, anet::nn::WeightNormMode::kSpectral, "zero_linear", rnd),
        Catch::Matchers::ContainsSubstring("zero_linear")
            && Catch::Matchers::ContainsSubstring("spectral_cap"));
    CHECK_THROWS_WITH(
        anet::nn::ParseWeightNormMode("mystery"),
        Catch::Matchers::ContainsSubstring("weight_norm.mode")
            && Catch::Matchers::ContainsSubstring("mystery")
            && Catch::Matchers::ContainsSubstring("none, spectral, spectral_cap"));
}

TEST_CASE("Spectral normalization requires an explicit network random source", "[nn][spectral_norm][seed]")
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("net.block.[Linear].type", "Linear");
    config_data.Set("net.block.[Linear].linear.out_features", 3);
    config_data.Set("net.block.[Linear].weight_norm.mode", "spectral");
    config_data.Set("net.branch.[feature].bind", "obs");
    config_data.Set("net.branch.[feature].structure", "Linear");
    const anet::TensorSpec spec{
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
    };

    CHECK_THROWS_WITH(
        anet::nn::NetworkBodyBuilder::Build(
            anet::nn::NetworkConfig(config_data),
            anet::TensorSpecMap{ { "obs", spec } }, nullptr),
        Catch::Matchers::ContainsSubstring("Linear")
            && Catch::Matchers::ContainsSubstring("Network-scoped ModuleRandomSource"));
}

TEST_CASE("Spectral normalization updates buffers only in training with GradMode", "[nn][spectral_norm][buffer]")
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("net.block.[Linear].type", "Linear");
    config_data.Set("net.block.[Linear].linear.out_features", 3);
    config_data.Set("net.block.[Linear].weight_norm.mode", "spectral");
    config_data.Set("net.branch.[feature].bind", "obs");
    config_data.Set("net.branch.[feature].structure", "Linear");
    config_data.Set("net.body.output.[feature]", "feature");
    const anet::TensorSpec spec{
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
    };
    auto network = anet::nn::NetworkBuilder::BuildNetwork(
        anet::nn::NetworkConfig(config_data),
        anet::TensorSpecMap{ { "obs", spec } }, nullptr, 65065, torch::Device(torch::kCPU));
    anet::TensorDict input{ { "obs", torch::tensor({ { 0.25f, -1.5f } }) } };
    {
        torch::NoGradGuard no_grad;
        auto entry = network->GetSpectralNormEntries().front();
        entry.u.copy_(torch::tensor({ 1.0f, 0.0f, 0.0f }));
        entry.v.copy_(torch::tensor({ 1.0f, 0.0f }));
    }
    const auto initial_u = network->GetSpectralNormEntries().front().u.clone();

    network->eval();
    (void)network->Forward(input);
    CHECK(torch::equal(network->GetSpectralNormEntries().front().u, initial_u));

    network->train();
    {
        torch::NoGradGuard no_grad;
        (void)network->Forward(input);
    }
    CHECK(torch::equal(network->GetSpectralNormEntries().front().u, initial_u));

    (void)network->Forward(input);
    CHECK_FALSE(torch::equal(network->GetSpectralNormEntries().front().u, initial_u));
}

TEST_CASE("Spectral normalization covers convolution residual ConvNeXt and Transformer weights", "[nn][spectral_norm][modules]")
{
    EnsureNNInitialized();
    const auto create = [](const std::string& type, anet::ConfigData config_data) {
        config_data.Set("weight_norm.mode", "spectral");
        anet::nn::ModuleContext context{
            .random_source = std::make_shared<anet::nn::ModuleRandomSource>(65065),
        };
        return anet::nn::NetworkModuleRepository::Instance().GetFactory(type)->CreateModule(config_data, context);
    };
    const auto check_module = [](const auto& module, const std::set<std::string>& expected) {
        std::set<std::string> actual;
        for (const auto& entry : module->GetSpectralNormEntries()) actual.insert(entry.name);
        CHECK(actual == expected);
        CHECK(module->GetCurrentConfigData().Get("weight_norm.mode") == "spectral");
    };

    SECTION("Conv1d") {
        anet::ConfigData config;
        config.Set("conv.out_channels", 4);
        config.Set("conv.kernel_size", 3);
        auto module = create("Conv1d", config);
        auto input = torch::randn({ 2, 3, 8 });
        CHECK(module->Forward(input, anet::nn::NetworkForwardContext::GetDefault()).sizes() == torch::IntArrayRef({ 2, 4, 6 }));
        check_module(module, { "conv" });
    }
    SECTION("Conv2d") {
        anet::ConfigData config;
        config.Set("conv.out_channels", 4);
        config.Set("conv.kernel_size", 3);
        auto module = create("Conv2d", config);
        auto input = torch::randn({ 2, 3, 8, 8 });
        CHECK(module->Forward(input, anet::nn::NetworkForwardContext::GetDefault()).sizes() == torch::IntArrayRef({ 2, 4, 6, 6 }));
        check_module(module, { "conv2d" });
    }
    SECTION("ResBlock") {
        anet::ConfigData config;
        config.Set("res.channels", 4);
        config.Set("res.kernel_size", 3);
        config.Set("res.padding", 1);
        config.Set("res.stride", 2);
        config.Set("init2.mode", "he");
        auto module = create("ResBlock", config);
        auto input = torch::randn({ 2, 3, 8, 8 });
        CHECK(module->Forward(input, anet::nn::NetworkForwardContext::GetDefault()).sizes() == torch::IntArrayRef({ 2, 4, 4, 4 }));
        check_module(module, { "conv1", "conv2", "downsample" });
    }
    SECTION("CNBlock") {
        anet::ConfigData config;
        config.Set("cn.channels", 3);
        config.Set("cn.kernel_size", 3);
        config.Set("cn.ffn_expand_ratio", 2);
        auto module = create("CNBlock", config);
        auto input = torch::randn({ 2, 3, 8, 8 });
        CHECK(module->Forward(input, anet::nn::NetworkForwardContext::GetDefault()).sizes() == input.sizes());
        check_module(module, { "dwconv", "pwconv1", "pwconv2" });
    }
    SECTION("TransformerEncoder") {
        anet::ConfigData config;
        config.Set("tf.d_model", 4);
        config.Set("tf.nhead", 2);
        config.Set("tf.num_layers", 2);
        config.Set("tf.dim_feedforward", 8);
        config.Set("tf.use_sdpa", true);
        auto module = create("TransformerEncoder", config);
        auto input = torch::randn({ 2, 3, 4 });
        input.requires_grad_(true);
        const auto output = module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
        CHECK(output.sizes() == input.sizes());
        check_module(module, {
            "layer_0.q", "layer_0.k", "layer_0.v", "layer_0.out_proj", "layer_0.linear1", "layer_0.linear2",
            "layer_1.q", "layer_1.k", "layer_1.v", "layer_1.out_proj", "layer_1.linear1", "layer_1.linear2" });
        output.backward(torch::randn_like(output));
        const auto packed_grad = GetNamedParameter(*module, "layer_0.self_attn.in_proj_weight").grad();
        REQUIRE(packed_grad.defined());
        const auto qkv_grad = packed_grad.chunk(3, 0);
        CHECK(qkv_grad[0].abs().sum().item<float>() > 0.0f);
        CHECK(qkv_grad[1].abs().sum().item<float>() > 0.0f);
        CHECK(qkv_grad[2].abs().sum().item<float>() > 0.0f);
    }
}

TEST_CASE("Transformer spectral normalization requires SDPA", "[nn][spectral_norm][transformer][config]")
{
    EnsureNNInitialized();
    anet::ConfigData config;
    config.Set("tf.d_model", 4);
    config.Set("tf.nhead", 2);
    config.Set("tf.use_sdpa", false);
    config.Set("weight_norm.mode", "spectral");
    anet::nn::ModuleContext context{
        .random_source = std::make_shared<anet::nn::ModuleRandomSource>(65065),
    };

    CHECK_THROWS_WITH(
        anet::nn::NetworkModuleRepository::Instance().GetFactory("TransformerEncoder")
            ->CreateModule(config, context),
        Catch::Matchers::ContainsSubstring("weight_norm.mode=spectral")
            && Catch::Matchers::ContainsSubstring("tf.use_sdpa=true"));
}

TEST_CASE("Spectral normalization uses reproducible purpose-local randomness", "[nn][spectral_norm][seed]")
{
    anet::nn::ModuleRandomSource first(65065);
    const auto first_spectral = first.Get("spectral_norm");
    const auto spectral_value = first_spectral->RandUint64();
    const auto other_value = first.Get("other")->RandUint64();
    CHECK(first.Get("spectral_norm") == first_spectral);

    anet::nn::ModuleRandomSource reordered(65065);
    CHECK(reordered.Get("other")->RandUint64() == other_value);
    CHECK(reordered.Get("spectral_norm")->RandUint64() == spectral_value);
}

TEST_CASE("Spectral normalization seed does not perturb parameter initialization", "[nn][spectral_norm][seed]")
{
    EnsureNNInitialized();
    const auto build = [](const std::string& mode, anet::seed_t seed) {
        anet::ConfigData config_data;
        config_data.Set("net.block.[Linear].type", "Linear");
        config_data.Set("net.block.[Linear].linear.out_features", 3);
        config_data.Set("net.block.[Linear].weight_norm.mode", mode);
        config_data.Set("net.branch.[feature].bind", "obs");
        config_data.Set("net.branch.[feature].structure", "Linear");
        config_data.Set("net.body.output.[feature]", "feature");
        const anet::TensorSpec spec{
            .type = anet::SpaceType::Vector,
            .shape = { 2 },
            .dtype = torch::kFloat32,
        };
        return anet::nn::NetworkBuilder::BuildNetwork(
            anet::nn::NetworkConfig(config_data),
            anet::TensorSpecMap{ { "obs", spec } }, nullptr, seed, torch::Device(torch::kCPU));
    };

    torch::manual_seed(9123);
    auto without_sn = build("none", 1);
    torch::manual_seed(9123);
    auto with_sn = build("spectral", 65065);
    const auto without_params = without_sn->named_parameters(true);
    const auto with_params = with_sn->named_parameters(true);
    REQUIRE(without_params.size() == with_params.size());
    for (const auto& item : without_params) {
        INFO(item.key());
        CHECK(torch::equal(item.value(), with_params[item.key()]));
    }

    torch::manual_seed(9123);
    auto same_seed = build("spectral", 65065);
    const auto first_entries = with_sn->GetSpectralNormEntries();
    const auto same_entries = same_seed->GetSpectralNormEntries();
    REQUIRE(first_entries.size() == same_entries.size());
    CHECK(torch::equal(first_entries.front().u, same_entries.front().u));
    CHECK(torch::equal(first_entries.front().v, same_entries.front().v));

    without_sn->eval();
    anet::TensorDict input{ { "obs", torch::tensor({ { 0.5f, -2.0f } }) } };
    torch::NoGradGuard no_grad;
    const auto direct = without_sn->Forward(input).At("feature");
    torch::manual_seed(9123);
    auto explicit_none = build("none", 999);
    explicit_none->eval();
    CHECK(torch::equal(direct, explicit_none->Forward(input).At("feature")));
}

TEST_CASE("Network parameter norm replaces only spectral weights with effective values", "[nn][spectral_norm][plasticity]")
{
    EnsureNNInitialized();
    anet::ConfigData config_data;
    config_data.Set("net.block.[Linear].type", "Linear");
    config_data.Set("net.block.[Linear].linear.out_features", 2);
    config_data.Set("net.block.[Linear].weight_norm.mode", "spectral");
    config_data.Set("net.branch.[feature].bind", "obs");
    config_data.Set("net.branch.[feature].structure", "Linear");
    config_data.Set("net.body.output.[feature]", "feature");
    const anet::TensorSpec spec{
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
    };
    auto network = anet::nn::NetworkBuilder::BuildNetwork(
        anet::nn::NetworkConfig(config_data),
        anet::TensorSpecMap{ { "obs", spec } }, nullptr, 65065, torch::Device(torch::kCPU));
    auto entry = network->GetSpectralNormEntries().front();
    torch::Tensor bias;
    for (const auto& item : network->named_parameters(true)) {
        if (item.key().ends_with("linear.bias")) bias = item.value();
    }
    REQUIRE(bias.defined());
    {
        torch::NoGradGuard no_grad;
        entry.weight.copy_(torch::tensor({ { 3.0f, 0.0f }, { 0.0f, 2.0f } }));
        bias.copy_(torch::tensor({ 4.0f, 0.0f }));
        entry.u.copy_(torch::tensor({ 1.0f, 0.0f }));
        entry.v.copy_(torch::tensor({ 1.0f, 0.0f }));
    }

    const auto norms = network->ComputeParameterNormSplit("feature");
    CHECK(norms.feature.item<float>() == Catch::Approx(std::sqrt(29.0f)));
    CHECK(norms.feature_effective.item<float>() == Catch::Approx(std::sqrt(157.0f / 9.0f)));
    CHECK(norms.sigma_feature_max.item<float>() == Catch::Approx(3.0f));
    CHECK(norms.readout.item<float>() == 0.0f);
    CHECK(norms.readout_effective.item<float>() == 0.0f);
    CHECK(std::isnan(norms.sigma_readout_max.item<float>()));
    CHECK(norms.invalid_count.item<float>() == 0.0f);
}

TEST_CASE("Spectral normalization buffers survive clone copy and serialization", "[nn][spectral_norm][seed][serialize]")
{
    EnsureNNInitialized();
    const auto build = [](anet::seed_t seed) {
        anet::ConfigData config_data;
        config_data.Set("net.block.[Linear].type", "Linear");
        config_data.Set("net.block.[Linear].linear.out_features", 3);
        config_data.Set("net.block.[Linear].weight_norm.mode", "spectral");
        config_data.Set("net.branch.[feature].bind", "obs");
        config_data.Set("net.branch.[feature].structure", "Linear");
        config_data.Set("net.body.output.[feature]", "feature");
        const anet::TensorSpec spec{
            .type = anet::SpaceType::Vector,
            .shape = { 2 },
            .dtype = torch::kFloat32,
        };
        return anet::nn::NetworkBuilder::BuildNetwork(
            anet::nn::NetworkConfig(config_data),
            anet::TensorSpecMap{ { "obs", spec } }, nullptr, seed, torch::Device(torch::kCPU));
    };

    torch::manual_seed(1001);
    auto source = build(65065);
    auto cloned = source->Clone(torch::Device(torch::kCPU));
    const auto source_entry = source->GetSpectralNormEntries().front();
    const auto cloned_entry = cloned->GetSpectralNormEntries().front();
    CHECK(torch::equal(source_entry.u, cloned_entry.u));
    CHECK(torch::equal(source_entry.v, cloned_entry.v));
    CHECK(source_entry.u.data_ptr<float>() != cloned_entry.u.data_ptr<float>());

    torch::manual_seed(1002);
    auto copied = build(123);
    source->CopyTo(*copied);
    CHECK(torch::equal(source_entry.u, copied->GetSpectralNormEntries().front().u));
    CHECK(torch::equal(source_entry.v, copied->GetSpectralNormEntries().front().v));

    torch::manual_seed(1003);
    auto restored = build(456);
    std::stringstream archive(std::ios::in | std::ios::out | std::ios::binary);
    std::shared_ptr<torch::nn::Module> source_base = source;
    torch::save(source_base, archive);
    archive.seekg(0);
    std::shared_ptr<torch::nn::Module> restored_base = restored;
    torch::load(restored_base, archive);
    CHECK(torch::equal(source_entry.u, restored->GetSpectralNormEntries().front().u));
    CHECK(torch::equal(source_entry.v, restored->GetSpectralNormEntries().front().v));
}

TEST_CASE("Spectral normalization soft copy validates tau before mutation and renormalizes buffers", "[nn][spectral_norm][soft-copy]")
{
    EnsureNNInitialized();
    const auto build = [](anet::seed_t seed) {
        anet::ConfigData config_data;
        config_data.Set("net.block.[Linear].type", "Linear");
        config_data.Set("net.block.[Linear].linear.out_features", 3);
        config_data.Set("net.block.[Linear].weight_norm.mode", "spectral");
        config_data.Set("net.branch.[feature].bind", "obs");
        config_data.Set("net.branch.[feature].structure", "Linear");
        config_data.Set("net.body.output.[feature]", "feature");
        const anet::TensorSpec spec{
            .type = anet::SpaceType::Vector,
            .shape = { 2 },
            .dtype = torch::kFloat32,
        };
        return anet::nn::NetworkBuilder::BuildNetwork(
            anet::nn::NetworkConfig(config_data),
            anet::TensorSpecMap{ { "obs", spec } }, nullptr, seed, torch::Device(torch::kCPU));
    };

    torch::manual_seed(111);
    auto source = build(65065);
    torch::manual_seed(222);
    auto target = build(65066);
    std::map<std::string, torch::Tensor> target_before;
    for (const auto& item : target->named_parameters(true)) {
        target_before.emplace(item.key(), item.value().detach().clone());
    }
    for (const auto& item : target->named_buffers(true)) {
        target_before.emplace(item.key(), item.value().detach().clone());
    }

    for (const double invalid_tau : {
        -0.001, 0.100001, 0.5, std::numeric_limits<double>::quiet_NaN() }) {
        CHECK_THROWS(source->SoftCopyTo(*target, invalid_tau));
        for (const auto& item : target->named_parameters(true)) {
            CHECK(torch::equal(item.value(), target_before.at(item.key())));
        }
        for (const auto& item : target->named_buffers(true)) {
            CHECK(torch::equal(item.value(), target_before.at(item.key())));
        }
    }

    for (const double valid_tau : { 0.0, static_cast<double>(0.1f), 1.0 }) {
        CHECK_NOTHROW(source->SoftCopyTo(*target, valid_tau));
        for (const auto& entry : target->GetSpectralNormEntries()) {
            CHECK(entry.u.norm().item<float>() == Catch::Approx(1.0f).margin(1.0e-5f));
            CHECK(entry.v.norm().item<float>() == Catch::Approx(1.0f).margin(1.0e-5f));
        }
    }
}

TEST_CASE("Network splits parameter norm at the feature dependency closure", "[nn][plasticity][weight_norm]")
{
    auto network = MakeParameterNormTestNetwork();

    const auto norms = network->ComputeParameterNormSplit("main_feature");

    CHECK(norms.feature.dim() == 0);
    CHECK(norms.readout.dim() == 0);
    CHECK(norms.feature.scalar_type() == torch::kFloat32);
    CHECK(norms.readout.scalar_type() == torch::kFloat32);
    CHECK(norms.feature.item<float>() == Catch::Approx(5.0f));
    CHECK(norms.readout.item<float>() == Catch::Approx(13.0f));
    CHECK(norms.feature_effective.item<float>() == Catch::Approx(5.0f));
    CHECK(norms.readout_effective.item<float>() == Catch::Approx(13.0f));
    CHECK(std::isnan(norms.sigma_feature_max.item<float>()));
    CHECK(std::isnan(norms.sigma_readout_max.item<float>()));
    CHECK(norms.invalid_count.item<float>() == 0.0f);
}

TEST_CASE("Network parameter norm includes weight bias and normalization affine parameters", "[nn][plasticity][weight_norm]")
{
    const anet::TensorSpec spec{
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
    };
    auto block = std::make_shared<anet::nn::NetworkBlock>(
        "affine_block",
        std::make_shared<ParameterNormAffineTestModule>());
    auto branch = std::make_shared<anet::nn::NetworkBranch>(
        "main_feature",
        std::vector<std::vector<std::string>>{ { "obs" } },
        1,
        std::make_shared<anet::nn::NetworkStruct>(
            std::vector<std::shared_ptr<anet::nn::NetworkBlock>>{ block }));
    anet::nn::NetworkConfig config;
    config.output_keys["features"] = "main_feature";
    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{ branch },
        anet::TensorSpecMap{ { "obs", spec } },
        std::vector<std::string>{},
        config.output_keys);
    auto network = std::make_shared<anet::nn::Network>(
        config,
        anet::TensorSpecMap{ { "obs", spec } },
        nullptr,
        body,
        nullptr);

    const auto norms = network->ComputeParameterNormSplit("main_feature");

    CHECK(norms.feature.item<float>() == Catch::Approx(std::sqrt(294.0f)));
    CHECK(norms.readout.item<float>() == 0.0f);
}

TEST_CASE("Network parameter norm returns CPU zeros when the network has no tensors", "[nn][plasticity][weight_norm]")
{
    const anet::TensorSpec spec{
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
    };
    auto branch = std::make_shared<anet::nn::NetworkBranch>(
        "main_feature",
        std::vector<std::vector<std::string>>{ { "obs" } },
        1,
        std::make_shared<anet::nn::NetworkStruct>(
            std::vector<std::shared_ptr<anet::nn::NetworkBlock>>{}));
    anet::nn::NetworkConfig config;
    config.output_keys["features"] = "main_feature";
    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{ branch },
        anet::TensorSpecMap{ { "obs", spec } },
        std::vector<std::string>{},
        config.output_keys);
    auto network = std::make_shared<anet::nn::Network>(
        config,
        anet::TensorSpecMap{ { "obs", spec } },
        nullptr,
        body,
        nullptr);

    const auto norms = network->ComputeParameterNormSplit("main_feature");

    CHECK(norms.feature.device().is_cpu());
    CHECK(norms.readout.device().is_cpu());
    CHECK(norms.feature.item<float>() == 0.0f);
    CHECK(norms.readout.item<float>() == 0.0f);
}

TEST_CASE("Network computes BF16 parameter norm in FP32 without changing parameters", "[nn][plasticity][weight_norm][bf16]")
{
    auto network = MakeParameterNormTestNetwork();
    network->to(torch::kBFloat16);

    const auto norms = network->ComputeParameterNormSplit("main_feature");

    CHECK(norms.feature.scalar_type() == torch::kFloat32);
    CHECK(norms.readout.scalar_type() == torch::kFloat32);
    CHECK(norms.feature.item<float>() == Catch::Approx(5.0f));
    CHECK(norms.readout.item<float>() == Catch::Approx(13.0f));
    for (const auto& parameter : network->parameters()) {
        CHECK(parameter.scalar_type() == torch::kBFloat16);
    }
}

TEST_CASE("Network excludes frozen parameters and returns zero for empty groups", "[nn][plasticity][weight_norm]")
{
    auto network = MakeParameterNormTestNetwork();
    for (auto& item : network->named_parameters()) {
        if (item.key().find("value_stream") != std::string::npos
            || item.key().starts_with("head.")) {
            item.value().requires_grad_(false);
        }
    }

    const auto readout_empty = network->ComputeParameterNormSplit("main_feature");
    CHECK(readout_empty.feature.item<float>() == Catch::Approx(5.0f));
    CHECK(readout_empty.readout.item<float>() == 0.0f);

    for (auto& item : network->named_parameters()) {
        item.value().requires_grad_(false);
    }
    const auto both_empty = network->ComputeParameterNormSplit("main_feature");
    CHECK(both_empty.feature.item<float>() == 0.0f);
    CHECK(both_empty.readout.item<float>() == 0.0f);
}

TEST_CASE("Network keeps IQN tau fusion and readout branches outside the feature norm group", "[nn][plasticity][weight_norm][iqn]")
{
    const anet::TensorSpec spec{
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
    };
    const auto make_parameter_branch = [](
        const std::string& name,
        std::vector<std::vector<std::string>> bind_terms,
        const torch::Tensor& parameter) {
        auto block = std::make_shared<anet::nn::NetworkBlock>(
            name + "_block",
            std::make_shared<ParameterNormTestModule>(parameter));
        auto structure = std::make_shared<anet::nn::NetworkStruct>(
            std::vector<std::shared_ptr<anet::nn::NetworkBlock>>{ block });
        return std::make_shared<anet::nn::NetworkBranch>(
            name, std::move(bind_terms), 1, structure);
    };
    auto main_feature = make_parameter_branch(
        "main_feature", { { "obs" } }, torch::tensor({ 3.0f, 4.0f }));
    auto tau_embedding = make_parameter_branch(
        "tau_embedding", { { "taus" } }, torch::tensor({ 6.0f, 8.0f }));
    auto fusion = std::make_shared<anet::nn::NetworkBranch>(
        "fusion",
        std::vector<std::vector<std::string>>{ { "main_feature", "tau_embedding" } },
        1,
        std::make_shared<anet::nn::NetworkStruct>(
            std::vector<std::shared_ptr<anet::nn::NetworkBlock>>{}));
    auto value_stream = make_parameter_branch(
        "value_stream", { { "fusion" } }, torch::tensor({ 12.0f }));
    anet::nn::NetworkConfig config;
    config.output_keys["features"] = "value_stream";
    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{
            main_feature, tau_embedding, fusion, value_stream },
        anet::TensorSpecMap{ { "obs", spec }, { "taus", spec } },
        std::vector<std::string>{},
        config.output_keys);
    auto network = std::make_shared<anet::nn::Network>(
        config,
        anet::TensorSpecMap{ { "obs", spec }, { "taus", spec } },
        nullptr,
        body,
        std::make_shared<ParameterNormTestHead>(torch::tensor({ 5.0f })));

    const auto norms = network->ComputeParameterNormSplit("main_feature");

    CHECK(norms.feature.item<float>() == Catch::Approx(5.0f));
    CHECK(norms.readout.item<float>() == Catch::Approx(std::sqrt(269.0f)));
    CHECK_THROWS_WITH(network->ComputeParameterNormSplit("missing"),
        Catch::Matchers::ContainsSubstring("available_branches"));
}

TEST_CASE("Network parameter norm does not run forward or consume RNG", "[nn][plasticity][weight_norm]")
{
    auto forward_count = std::make_shared<int>(0);
    auto network = MakeParameterNormTestNetwork(forward_count);
    torch::manual_seed(32063);
    const auto expected_random = torch::randn({ 8 });

    torch::manual_seed(32063);
    const auto norms = network->ComputeParameterNormSplit("main_feature");
    const auto actual_random = torch::randn({ 8 });

    CHECK(norms.feature.item<float>() == Catch::Approx(5.0f));
    CHECK(*forward_count == 0);
    CHECK(torch::equal(actual_random, expected_random));
}

TEST_CASE("Network partial forward stops before IQN tau fusion without taus input", "[nn][partial_forward][plasticity][iqn]")
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("net.branch.[main_feature].bind", "obs");
    config_data.Set("net.branch.[main_feature].structure", "");
    config_data.Set("net.branch.[tau_embedding].bind", "taus");
    config_data.Set("net.branch.[tau_embedding].structure", "");
    config_data.Set("net.branch.[fusion].bind", "main_feature * tau_embedding");
    config_data.Set("net.branch.[fusion].structure", "");
    config_data.Set("net.body.output.[features]", "fusion");

    auto network = anet::nn::NetworkBuilder::BuildNetwork(
        anet::nn::NetworkConfig(config_data),
        anet::TensorSpecMap{
            { "obs", anet::TensorSpec{
                .type = anet::SpaceType::Vector,
                .shape = { 2 },
                .dtype = torch::kFloat32,
            } },
            { "taus", anet::TensorSpec{
                .type = anet::SpaceType::Vector,
                .shape = { 3, 2 },
                .dtype = torch::kFloat32,
            } },
        },
        nullptr,
        0,
        torch::Device(torch::kCPU));

    anet::TensorDict input;
    input.Set("obs", torch::tensor({ { 1.0f, 2.0f } }));
    const auto state = network->ForwardUpTo(input, "main_feature");

    CHECK(torch::equal(state.At("main_feature"), input.At("obs")));
    CHECK_FALSE(state.Contains("taus"));
    CHECK_FALSE(state.Contains("tau_embedding"));
    CHECK_FALSE(state.Contains("fusion"));
}

TEST_CASE("Network partial forward gives input keys precedence over same-named branches", "[nn][partial_forward][plasticity]")
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("net.branch.[obs].bind", "aux");
    config_data.Set("net.branch.[obs].structure", "");
    config_data.Set("net.branch.[target].bind", "obs");
    config_data.Set("net.branch.[target].structure", "");
    config_data.Set("net.body.output.[features]", "target");
    const anet::TensorSpec spec{
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
    };
    auto network = anet::nn::NetworkBuilder::BuildNetwork(
        anet::nn::NetworkConfig(config_data),
        anet::TensorSpecMap{ { "obs", spec }, { "aux", spec } },
        nullptr,
        0,
        torch::Device(torch::kCPU));

    anet::TensorDict input;
    input.Set("obs", torch::tensor({ { 1.0f, 2.0f } }));
    input.Set("aux", torch::tensor({ { 9.0f, 9.0f } }));
    const auto state = network->ForwardUpTo(input, "target");

    CHECK(torch::equal(state.At("target"), input.At("obs")));
    CHECK_THROWS_WITH(network->ForwardUpTo(input, "missing"),
        Catch::Matchers::ContainsSubstring("available_branches"));
}

TEST_CASE("Network branch capture returns the actual forward feature without changing outputs", "[nn][branch_capture][plasticity]")
{
    EnsureNNInitialized();

    anet::ConfigData config_data;
    config_data.Set("net.branch.[feature].bind", "obs");
    config_data.Set("net.branch.[feature].structure", "");
    config_data.Set("net.body.output.[features]", "feature");
    const anet::TensorSpec spec{
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
    };
    auto network = anet::nn::NetworkBuilder::BuildNetwork(
        anet::nn::NetworkConfig(config_data),
        anet::TensorSpecMap{ { "obs", spec } },
        nullptr,
        0,
        torch::Device(torch::kCPU));

    anet::TensorDict input;
    input.Set("obs", torch::tensor({ { 1.0f, 2.0f }, { 3.0f, 4.0f } }));
    std::vector<std::pair<std::string, torch::Tensor>> expected_trace;
    const auto expected = network->Forward(input, [&](std::string_view name, const torch::Tensor& value) {
        expected_trace.emplace_back(name, value.detach().clone());
    });

    anet::nn::NetworkForwardCapture capture({ "feature" });
    std::vector<std::pair<std::string, torch::Tensor>> actual_trace;
    const auto actual = network->Forward(input, [&](std::string_view name, const torch::Tensor& value) {
        actual_trace.emplace_back(name, value.detach().clone());
    }, &capture);

    CHECK(torch::equal(actual.At("features"), expected.At("features")));
    REQUIRE(actual_trace.size() == expected_trace.size());
    for (size_t i = 0; i < expected_trace.size(); ++i) {
        CHECK(actual_trace[i].first == expected_trace[i].first);
        CHECK(torch::equal(actual_trace[i].second, expected_trace[i].second));
    }
    REQUIRE(capture.Get("feature").defined());
    CHECK(torch::equal(capture.Get("feature"), expected.At("features")));
    CHECK_FALSE(capture.Get("feature").requires_grad());
}

TEST_CASE("Plasticity metrics describe a full-rank feature matrix", "[nn][plasticity_metrics]")
{
    const auto features = torch::eye(2, torch::TensorOptions().dtype(torch::kFloat32));

    const auto metrics = anet::nn::ComputePlasticityMetrics(
        features, anet::nn::PlasticityMetricRequest::All());

    CHECK(metrics.srank[0] == 2.0f);
    CHECK(metrics.srank_ratio[0].value() == Catch::Approx(1.0f));
    CHECK(metrics.dormant_ratio.value() == Catch::Approx(0.0f));
    CHECK(metrics.dead_ratio.value() == Catch::Approx(0.0f));
    CHECK(metrics.feature_norm.value() == Catch::Approx(1.0f));
}

TEST_CASE("Plasticity metrics compute BF16 features through FP32", "[nn][plasticity_metrics][bf16]")
{
    const auto features = torch::eye(
        2, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU));

    const auto metrics = anet::nn::ComputePlasticityMetrics(
        features, anet::nn::PlasticityMetricRequest::All());

    CHECK(features.scalar_type() == torch::kBFloat16);
    CHECK(metrics.srank[0] == 2.0f);
    CHECK(metrics.srank_ratio[0].value() == Catch::Approx(1.0f));
    CHECK(metrics.dormant_ratio.value() == Catch::Approx(0.0f));
    CHECK(metrics.dead_ratio.value() == Catch::Approx(0.0f));
    CHECK(metrics.feature_norm.value() == Catch::Approx(1.0f));
}

TEST_CASE("Plasticity metrics handle rank-one and all-zero features", "[nn][plasticity_metrics]")
{
    const auto rank_one = torch::tensor({ { 1.0f, 2.0f }, { 2.0f, 4.0f } });
    const auto rank_one_metrics = anet::nn::ComputePlasticityMetrics(
        rank_one, anet::nn::PlasticityMetricRequest::All());
    CHECK(rank_one_metrics.srank[0] == 1.0f);
    CHECK(rank_one_metrics.srank_ratio[0].value() == Catch::Approx(0.5f));

    const auto all_zero = torch::zeros({ 4, 3 }, torch::TensorOptions().dtype(torch::kFloat32));
    const auto zero_metrics = anet::nn::ComputePlasticityMetrics(
        all_zero, anet::nn::PlasticityMetricRequest::All());
    CHECK(zero_metrics.srank[0] == 0.0f);
    CHECK(zero_metrics.srank_ratio[0] == 0.0f);
    CHECK(zero_metrics.dormant_ratio.value() == 1.0f);
    CHECK(zero_metrics.dead_ratio.value() == 1.0f);
    CHECK(zero_metrics.feature_norm.value() == 0.0f);
}

TEST_CASE("Plasticity metrics reject a non-matrix input", "[nn][plasticity_metrics]")
{
    CHECK_THROWS(anet::nn::ComputePlasticityMetrics(
        torch::ones({ 3 }), anet::nn::PlasticityMetricRequest::All()));
}

TEST_CASE("Plasticity metrics compute three srank deltas from one SVD", "[nn][plasticity_metrics][srank]")
{
    const auto features = torch::diag(torch::tensor({ 70.0f, 20.0f, 6.0f, 4.0f }));
    anet::nn::PlasticityMetricRequest request;
    request.Add(anet::nn::PlasticityMetric::SRANK_DELTA_001);
    request.Add(anet::nn::PlasticityMetric::SRANK_DELTA_005);
    request.Add(anet::nn::PlasticityMetric::SRANK_DELTA_020);
    request.Add(anet::nn::PlasticityMetric::SRANK_RATIO_DELTA_001);
    request.Add(anet::nn::PlasticityMetric::SRANK_RATIO_DELTA_005);
    request.Add(anet::nn::PlasticityMetric::SRANK_RATIO_DELTA_020);

    anet::nn::PlasticityMetrics metrics;
    const size_t svd_count = CountProfileOperator([&] {
        metrics = anet::nn::ComputePlasticityMetrics(features, request);
    }, "aten::linalg_svdvals");

    CHECK(svd_count == 1);
    CHECK(metrics.srank[0] == 4.0f);
    CHECK(metrics.srank[1] == 3.0f);
    CHECK(metrics.srank[2] == 2.0f);
    CHECK(metrics.srank_ratio[0].value() == Catch::Approx(1.0f));
    CHECK(metrics.srank_ratio[1].value() == Catch::Approx(0.75f));
    CHECK(metrics.srank_ratio[2].value() == Catch::Approx(0.5f));
}

TEST_CASE("Plasticity metrics leave unrequested fields unavailable", "[nn][plasticity_metrics][demand]")
{
    const auto features = torch::eye(3, torch::TensorOptions().dtype(torch::kFloat32));
    anet::nn::PlasticityMetricRequest request;
    request.Add(anet::nn::PlasticityMetric::FEATURE_NORM);

    anet::nn::PlasticityMetrics metrics;
    const size_t svd_count = CountProfileOperator([&] {
        metrics = anet::nn::ComputePlasticityMetrics(features, request);
    }, "aten::linalg_svdvals");

    CHECK(svd_count == 0);
    CHECK_FALSE(metrics.srank[0].has_value());
    CHECK_FALSE(metrics.srank_ratio[0].has_value());
    CHECK_FALSE(metrics.dormant_ratio.has_value());
    CHECK_FALSE(metrics.dead_ratio.has_value());
    REQUIRE(metrics.feature_norm.has_value());
    CHECK(*metrics.feature_norm == Catch::Approx(1.0f));
}

TEST_CASE("Plasticity metrics empty request performs no tensor statistics", "[nn][plasticity_metrics][demand]")
{
    const auto features = torch::eye(3, torch::TensorOptions().dtype(torch::kFloat32));
    const anet::nn::PlasticityMetricRequest request;

    const auto metrics = anet::nn::ComputePlasticityMetrics(features, request);

    CHECK_FALSE(metrics.feature_norm.has_value());
    CHECK_FALSE(metrics.srank[0].has_value());
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
    CHECK_THROWS(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(invalid_data)));
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
    CHECK(torch::equal(module->Forward(input, anet::nn::NetworkForwardContext::GetDefault()), input));

    module->train();
    torch::Tensor output;
    bool saw_dropout = false;
    for (int seed = 1300; seed < 1310; ++seed) {
        torch::manual_seed(seed);
        output = module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
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
    CHECK(torch::equal(old_key_module->Forward(input, anet::nn::NetworkForwardContext::GetDefault()), input));
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
    (void)baseline->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
    (void)droppath->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
    CopyModuleState(*baseline, *droppath);

    torch::Tensor expected = baseline->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
    torch::Tensor actual = droppath->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
    CheckTensorClose(expected, actual);

    auto shortcut_only = MakeResBlockTestModule(/*droppath_rate=*/0.999999, /*dropout_rate=*/0.0);
    shortcut_only->train();
    (void)shortcut_only->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
    torch::manual_seed(1401);
    torch::Tensor shortcut_output = shortcut_only->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
    CheckTensorClose(input, shortcut_output);

    auto dropout2d_module = MakeResBlockTestModule(/*droppath_rate=*/0.0, /*dropout_rate=*/0.95);
    dropout2d_module->eval();
    torch::Tensor eval_output = dropout2d_module->Forward(input, anet::nn::NetworkForwardContext::GetDefault()).detach();
    dropout2d_module->train();

    bool saw_train_difference = false;
    for (int seed = 1402; seed < 1412; ++seed) {
        torch::manual_seed(seed);
        torch::Tensor train_output = dropout2d_module->Forward(input, anet::nn::NetworkForwardContext::GetDefault()).detach();
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

            CheckTensorClose(baseline->Forward(input, anet::nn::NetworkForwardContext::GetDefault()), dropout->Forward(input, anet::nn::NetworkForwardContext::GetDefault()));

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
            torch::Tensor first = module->Forward(input, anet::nn::NetworkForwardContext::GetDefault()).detach();
            torch::Tensor second = module->Forward(input, anet::nn::NetworkForwardContext::GetDefault()).detach();
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

    torch::Tensor before = save_module->Forward(input, anet::nn::NetworkForwardContext::GetDefault()).detach().clone();
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

    torch::Tensor after = loaded_module->Forward(input, anet::nn::NetworkForwardContext::GetDefault()).detach();
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

TEST_CASE("Explicit forward contexts preserve a configured ordinary network", "[nn][noisynet][context]")
{
    EnsureNNInitialized();

    // 設定から Body と Head を構築し、既存の μ の演算を比較基準にする。
    anet::ConfigData data;
    data.Set("net.block.[Linear].type", "Linear");
    data.Set("net.block.[Linear].linear.out_features", 3);
    data.Set("net.block.[Linear].init.mode", "constant");
    data.Set("net.block.[Linear].init.constant_val", 0.25);
    data.Set("net.branch.[feature].bind", "obs");
    data.Set("net.branch.[feature].structure", "Linear");
    data.Set("net.body.output.[features]", "feature");
    data.Set("nn_forward.[test].noisy.mode", "sample");
    data.Set("nn_forward.[test].noisy.hold", "call");
    data.Set("nn_forward.[test].noisy.share", "batch");
    const anet::TensorSpec spec{
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
    };
    const anet::nn::WeightInitConfig head_init{
        .mode = "constant",
        .constant_val = 0.5,
    };
    auto head_factory = std::make_shared<anet::nn::LinearHeadFactory>(2, "logits", head_init);
    auto network = anet::nn::NetworkBuilder::BuildNetwork(
        anet::nn::NetworkConfig(data), anet::TensorSpecMap{ { "obs", spec } },
        head_factory, 17, torch::Device(torch::kCPU));
    std::vector<torch::Tensor> before;
    for (const auto& parameter : network->parameters()) before.push_back(parameter.detach().clone());

    // 不透明な箱を持つ sample と、箱を持たない μ-only を同じ公開経路へ渡す。
    const anet::nn::NetworkForwardConfig config("nn_forward.[test]", SelectForwardConfigData(data, "nn_forward.[test]"));
    auto forward = network->CreateForwardExecutor(config, anet::nn::NetworkForwardClock::kContinuingRows, 123);
    const anet::nn::NetworkForwardCallInfo info({.clock = 0});
    anet::TensorDict input;
    input.Set("obs", torch::ones({ 2, 2 }));
    const auto sample = forward->Forward(input, info);
    const auto mean = network->Forward(input);
    const auto partial = network->ForwardUpTo(input, "feature");
    CHECK(torch::equal(sample.At("logits"), torch::full({ 2, 2 }, 1.625f)));
    CHECK(torch::equal(sample.At("logits"), mean.At("logits")));
    CHECK(torch::equal(network->Forward(input).At("logits"), mean.At("logits")));
    CHECK(torch::equal(partial.At("feature"), torch::full({ 2, 3 }, 0.75f)));
    const auto after = network->parameters();
    REQUIRE(after.size() == before.size());
    for (size_t i = 0; i < after.size(); ++i) CHECK(torch::equal(after[i], before[i]));
}

TEST_CASE("Concrete modules consume independent forward settings through existing network paths", "[nn][noisynet][context]")
{
    class ConfiguredModule final : public anet::nn::NetworkModule {
    public:
        torch::Tensor Forward(torch::Tensor input, const anet::nn::NetworkForwardContext&) override
        { return input; }
        std::unique_ptr<anet::nn::NetworkModuleExecutor> CreateExecutor(
            const anet::nn::NetworkForwardConfig& request, anet::nn::NetworkForwardClock) override
        { return std::make_unique<Execution>(request); }
    private:
        class Execution final : public anet::nn::NetworkModuleExecutor {
        public:
            explicit Execution(const anet::nn::NetworkForwardConfig& request)
                : scale_(request.IsDefault() ? 1.0f : request.GetConfigData().Get<float>("test_affine.scale")),
                  shift_(request.IsDefault() ? 0.0f : request.GetConfigData().Get<float>("test_affine.shift")) {}
            torch::Tensor Forward(torch::Tensor input, const anet::nn::NetworkForwardContext&) override
            { return input * scale_ + shift_; }
        private:
            const float scale_;
            const float shift_;
        };
    };
    class ConfiguredHead final : public anet::nn::NetworkHead {
    public:
        anet::TensorDict Forward(const anet::TensorDict& features, const anet::nn::NetworkForwardContext&) override
        { return anet::TensorDict{{"logits", features.At("features")}}; }
        std::unique_ptr<anet::nn::NetworkHeadExecutor> CreateExecutor(
            const anet::nn::NetworkForwardConfig& request, anet::nn::NetworkForwardClock) override
        { return std::make_unique<Execution>(request); }
    private:
        class Execution final : public anet::nn::NetworkHeadExecutor {
        public:
            explicit Execution(const anet::nn::NetworkForwardConfig& request)
                : offset_(request.IsDefault() ? 0.0f : request.GetConfigData().Get<float>("test_head.offset")) {}
            anet::TensorDict Forward(const anet::TensorDict& features, const anet::nn::NetworkForwardContext&) override
            { return anet::TensorDict{{"logits", features.At("features") + offset_}}; }
        private:
            const float offset_;
        };
    };

    // 既存 Module と新しい Module を、既存の Block・branch・Body で接続する。
    auto ordinary = std::make_shared<anet::nn::NetworkBlock>("ordinary", std::make_shared<TraceTestModule>());
    auto configured = std::make_shared<anet::nn::NetworkBlock>("configured", std::make_shared<ConfiguredModule>());
    auto structure = std::make_shared<anet::nn::NetworkStruct>(
        std::vector<std::shared_ptr<anet::nn::NetworkBlock>>{ ordinary, configured });
    auto branch = std::make_shared<anet::nn::NetworkBranch>(
        "feature", std::vector<std::vector<std::string>>{ { "obs" } }, 1, structure);
    const anet::TensorSpec spec{ .type = anet::SpaceType::Vector, .shape = { 2 }, .dtype = torch::kFloat32 };
    const anet::TensorSpecMap specs{ { "obs", spec } };
    anet::nn::NetworkConfig network_config;
    network_config.output_keys["features"] = "feature";
    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{ branch }, specs,
        std::vector<std::string>{}, network_config.output_keys);
    auto network = std::make_shared<anet::nn::Network>(
        network_config, specs, nullptr, body, std::make_shared<ConfiguredHead>());

    // 選んだ項目だけを不変に保持し、後から元の設定を変更しても結果を変えない。
    anet::ConfigData data;
    data.Set("nn_forward.[custom].test_affine.scale", 2.0f);
    data.Set("nn_forward.[custom].test_affine.shift", 3.0f);
    data.Set("nn_forward.[custom].test_head.offset", 5.0f);
    data.Set("nn_forward.[other].test_affine.scale", 99.0f);
    const anet::nn::NetworkForwardConfig config("nn_forward.[custom]", SelectForwardConfigData(data, "nn_forward.[custom]"));
    data.Set("nn_forward.[custom].test_affine.scale", 100.0f);
    auto forward = network->CreateForwardExecutor(config, anet::nn::NetworkForwardClock::kContinuingRows, 0);
    const anet::nn::NetworkForwardCallInfo info({.clock = 0});
    anet::TensorDict input;
    input.Set("obs", torch::ones({ 2, 2 }));
    CHECK(torch::equal(forward->Forward(input, info).At("logits"), torch::full({ 2, 2 }, 12.0f)));
    CHECK(torch::equal(network->ForwardUpTo(input, "feature").At("feature"), torch::full({ 2, 2 }, 2.0f)));
    CHECK(config.GetConfigData().Map().Size() == 3);
    CHECK(config.key == "nn_forward.[custom]");

    // 明示的な既定要求では、具象 Module と Head がそれぞれ既定値を選ぶ。
    CHECK(torch::equal(network->Forward(input).At("logits"), torch::full({ 2, 2 }, 2.0f)));

    // 新しい設定の検証を追加しても、共通 carrier と既存 Module の変更を要求しない。
    CHECK_NOTHROW(network->CreateForwardExecutor(config, anet::nn::NetworkForwardClock::kContinuingRows, 0));
    auto invalid_data = data;
    invalid_data.Set("nn_forward.[custom].test_affine.scale", "oops");
    const anet::nn::NetworkForwardConfig invalid_module("nn_forward.[custom]", SelectForwardConfigData(invalid_data, "nn_forward.[custom]"));
    CHECK_THROWS_WITH(network->CreateForwardExecutor(invalid_module, anet::nn::NetworkForwardClock::kContinuingRows, 0),
        Catch::Matchers::ContainsSubstring("test_affine.scale"));
    invalid_data = data;
    invalid_data.Set("nn_forward.[custom].test_head.offset", "oops");
    const anet::nn::NetworkForwardConfig invalid_head("nn_forward.[custom]", SelectForwardConfigData(invalid_data, "nn_forward.[custom]"));
    CHECK_THROWS_WITH(network->CreateForwardExecutor(invalid_head, anet::nn::NetworkForwardClock::kContinuingRows, 0),
        Catch::Matchers::ContainsSubstring("test_head.offset"));

}

TEST_CASE("Default forward context is an explicit module-neutral request", "[nn][noisynet][context]")
{
    const auto context = anet::nn::NetworkForwardContext::GetDefault();
    CHECK(context.GetState() == nullptr);
    CHECK_FALSE(context.GetEpisodeStart().defined());
    CHECK_FALSE(context.GetRowMapping().has_value());

    // 名前付き設定が空でも、明示的な既定動作の要求へ読み替えない。
    const anet::nn::NetworkForwardConfig named("nn_forward.[empty]", SelectForwardConfigData(anet::EmptyConfigData, "nn_forward.[empty]"));
    CHECK_FALSE(named.IsDefault());
}

TEST_CASE("Noisy Linear registers sigma without changing the mean-only computation", "[nn][noisynet][initialization]")
{
    EnsureNNInitialized();
    anet::ConfigData data;
    data.Set("linear.out_features", 3);
    data.Set("init.mode", "constant");
    data.Set("init.constant_val", 0.125f);
    data.Set("noisy.enabled", true);
    data.Set("noisy.sigma0", 0.5f);
    auto factory = anet::nn::NetworkModuleRepository::Instance().GetFactory("Linear");
    auto module = factory->CreateModule(data, anet::nn::ModuleContext{ .random_source = MakeTestModuleRandomSource() });
    auto input = torch::ones({ 2, 4 }).requires_grad_(true);

    // μ の名前・初期化と通常の Linear 出力を保持し、σ だけを追加する。
    const auto output = module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
    const auto params = module->named_parameters();
    REQUIRE(params.contains("linear.sigma_weight"));
    REQUIRE(params.contains("linear.sigma_bias"));
    CHECK(torch::equal(params["linear.weight"], torch::full({ 3, 4 }, 0.125f)));
    CHECK(torch::equal(params["linear.bias"], torch::full({ 3 }, 0.125f)));
    CHECK(torch::equal(params["linear.sigma_weight"], torch::full({ 3, 4 }, 0.25f)));
    CHECK(torch::equal(params["linear.sigma_bias"], torch::full({ 3 }, 0.25f)));
    CHECK(torch::equal(output, torch::full({ 2, 3 }, 0.625f)));

    // μ-only では σ を計算グラフへ入れず、入力と μ だけに勾配を作る。
    output.sum().backward();
    CHECK(input.grad().defined());
    CHECK(params["linear.weight"].grad().defined());
    CHECK_FALSE(params["linear.sigma_weight"].grad().defined());
    CHECK_FALSE(params["linear.sigma_bias"].grad().defined());
}

TEST_CASE("Noisy Linear factorized computation matches an explicit weight and its gradients", "[nn][noisynet][arithmetic]")
{
    EnsureNNInitialized();
    const auto with_bias = GENERATE(false, true);
    const auto sigma0 = GENERATE(0.0, 0.5);
    const auto negative_sigma = GENERATE(false, true);
    CAPTURE(with_bias, sigma0, negative_sigma);
    anet::ConfigData data;
    data.Set("linear.out_features", 3);
    data.Set("linear.bias", with_bias);
    data.Set("noisy.sigma0", sigma0);
    data.Set("init.mode", "constant");
    data.Set("init.constant_val", 0.125f);
    data.Set("noisy.enabled", true);
    data.Set("nn_forward.[sample].noisy.mode", "sample");
    data.Set("nn_forward.[sample].noisy.hold", "call");
    data.Set("nn_forward.[sample].noisy.share", "batch");
    auto module = anet::nn::NetworkModuleRepository::Instance().GetFactory("Linear")->CreateModule(
        data, anet::nn::ModuleContext{ .random_source = MakeTestModuleRandomSource() });
    auto input = (torch::arange(8, torch::kFloat32).reshape({ 2, 4 }) / 7.0f).requires_grad_(true);
    module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
    const auto params = module->named_parameters();
    if (negative_sigma) {
        torch::NoGradGuard no_grad;
        params["linear.sigma_weight"].mul_(-1.0);
        if (with_bias) params["linear.sigma_bias"].mul_(-1.0);
    }
    CHECK(params.contains("linear.sigma_bias") == with_bias);
    auto weight = params["linear.weight"].detach().clone().requires_grad_(true);
    auto bias = with_bias ? params["linear.bias"].detach().clone().requires_grad_(true) : torch::Tensor{};
    auto sigma_weight = params["linear.sigma_weight"].detach().clone().requires_grad_(true);
    auto sigma_bias = with_bias ? params["linear.sigma_bias"].detach().clone().requires_grad_(true) : torch::Tensor{};
    auto reference_input = input.detach().clone().requires_grad_(true);

    // 専用 RNG と同じ factor を抽選し、テストの参照だけで重みを明示合成する。
    constexpr anet::seed_t seed = 317;
    anet::RandomGenerator reference_random(seed);
    const auto generator = reference_random.GetTorchGenerator(input.device());
    const auto raw_in = torch::randn({ 4 }, generator, input.options());
    const auto raw_out = torch::randn({ 3 }, generator, input.options());
    const auto epsilon_in = raw_in.sign() * raw_in.abs().sqrt();
    const auto epsilon_out = raw_out.sign() * raw_out.abs().sqrt();
    const auto expected = torch::nn::functional::linear(reference_input,
        weight + sigma_weight * epsilon_out.unsqueeze(1) * epsilon_in.unsqueeze(0),
        with_bias ? bias + sigma_bias * epsilon_out : torch::Tensor{});
    auto state_owner = MakeDotTestNetwork(nullptr, anet::nn::kKey_DefaultOutput);
    auto state = std::make_shared<anet::nn::NetworkForwardState>(seed);
    const anet::nn::NetworkForwardConfig config("nn_forward.[sample]", SelectForwardConfigData(data, "nn_forward.[sample]"));
    const anet::nn::NetworkForwardCallInfo context_info({
        .clock = 0,
    });
    const anet::nn::NetworkForwardContext context(state.get(), context_info);
    auto owned_module_forward = module->CreateExecutor(config, anet::nn::NetworkForwardClock::kContinuingRows);
    auto* module_forward = owned_module_forward ? owned_module_forward.get() : module.get();
    const auto actual = module_forward->Forward(input, context);
    CHECK(torch::allclose(actual, expected, 1e-5, 1e-6));

    // 出力だけでなく入力・μ・σ の全勾配を独立の参照グラフと比較する。
    actual.square().sum().backward();
    expected.square().sum().backward();
    CHECK(torch::allclose(input.grad(), reference_input.grad(), 1e-5, 1e-6));
    CHECK(torch::allclose(params["linear.weight"].grad(), weight.grad(), 1e-5, 1e-6));
    if (with_bias) {
        CHECK(torch::allclose(params["linear.bias"].grad(), bias.grad(), 1e-5, 1e-6));
        REQUIRE(params["linear.sigma_bias"].grad().defined());
        CHECK(torch::allclose(params["linear.sigma_bias"].grad(), sigma_bias.grad(), 1e-5, 1e-6));
    }
    REQUIRE(params["linear.sigma_weight"].grad().defined());
    CHECK(torch::allclose(params["linear.sigma_weight"].grad(), sigma_weight.grad(), 1e-5, 1e-6));
}

TEST_CASE("Noisy Linear sample factors stay shared across rank-three quantiles", "[nn][noisynet][sample]")
{
    EnsureNNInitialized();
    anet::ConfigData data;
    data.Set("linear.out_features", 3);
    data.Set("init.mode", "constant");
    data.Set("init.constant_val", 0.125f);
    data.Set("noisy.enabled", true);
    data.Set("nn_forward.[sample].noisy.mode", "sample");
    data.Set("nn_forward.[sample].noisy.hold", "call");
    data.Set("nn_forward.[sample].noisy.share", "sample");
    auto module = anet::nn::NetworkModuleRepository::Instance().GetFactory("Linear")->CreateModule(
        data, anet::nn::ModuleContext{ .random_source = MakeTestModuleRandomSource() });
    auto input = torch::ones({ 3, 2, 4 }).requires_grad_(true);
    module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
    const auto params = module->named_parameters();
    auto weight = params["linear.weight"].detach().clone().requires_grad_(true);
    auto bias = params["linear.bias"].detach().clone().requires_grad_(true);
    auto sigma_weight = params["linear.sigma_weight"].detach().clone().requires_grad_(true);
    auto sigma_bias = params["linear.sigma_bias"].detach().clone().requires_grad_(true);
    auto reference_input = input.detach().clone().requires_grad_(true);
    constexpr anet::seed_t seed = 419;
    anet::RandomGenerator random(seed);
    const auto generator = random.GetTorchGenerator(input.device());
    const auto raw_input = torch::randn({ 3, 4 }, generator, input.options());
    const auto raw_output = torch::randn({ 3, 3 }, generator, input.options());
    const auto epsilon_in = raw_input.sign() * raw_input.abs().sqrt();
    const auto epsilon_out = raw_output.sign() * raw_output.abs().sqrt();

    // 参照では遷移ごとの重みを合成し、同一遷移の全 τ へ同じ ε を適用する。
    const auto noisy_weight = weight.unsqueeze(0)
        + sigma_weight.unsqueeze(0) * epsilon_out.unsqueeze(2) * epsilon_in.unsqueeze(1);
    const auto noisy_bias = bias.unsqueeze(0) + sigma_bias.unsqueeze(0) * epsilon_out;
    const auto expected = torch::bmm(reference_input, noisy_weight.transpose(1, 2)) + noisy_bias.unsqueeze(1);
    auto owner = MakeDotTestNetwork();
    auto state = std::make_shared<anet::nn::NetworkForwardState>(seed);
    const anet::nn::NetworkForwardConfig config("nn_forward.[sample]", SelectForwardConfigData(data, "nn_forward.[sample]"));
    const anet::nn::NetworkForwardCallInfo context_info({
        .clock = 0,
    });
    const anet::nn::NetworkForwardContext context(state.get(), context_info);
    auto owned_module_forward = module->CreateExecutor(config, anet::nn::NetworkForwardClock::kTransientRows);
    auto* module_forward = owned_module_forward ? owned_module_forward.get() : module.get();
    const auto actual = module_forward->Forward(input, context);
    CHECK(torch::allclose(actual, expected, 1e-5, 1e-6));
    CHECK(torch::equal(actual.select(1, 0), actual.select(1, 1)));
    CHECK_FALSE(torch::equal(actual.select(0, 0), actual.select(0, 1)));
    actual.square().sum().backward();
    expected.square().sum().backward();
    CHECK(torch::allclose(input.grad(), reference_input.grad(), 1e-5, 1e-6));
    CHECK(torch::allclose(params["linear.weight"].grad(), weight.grad(), 1e-5, 1e-6));
    CHECK(torch::allclose(params["linear.bias"].grad(), bias.grad(), 1e-5, 1e-6));
    CHECK(torch::allclose(params["linear.sigma_weight"].grad(), sigma_weight.grad(), 1e-5, 1e-6));
    CHECK(torch::allclose(params["linear.sigma_bias"].grad(), sigma_bias.grad(), 1e-5, 1e-6));
}

TEST_CASE("Noisy Linear count holding ignores episode boundaries and redraws at its period", "[nn][noisynet][count]")
{
    EnsureNNInitialized();
    anet::ConfigData data;
    data.Set("linear.out_features", 3);
    data.Set("init.mode", "constant");
    data.Set("init.constant_val", 0.125f);
    data.Set("noisy.enabled", true);
    data.Set("nn_forward.[count].noisy.mode", "sample");
    data.Set("nn_forward.[count].noisy.hold", "count");
    data.Set("nn_forward.[count].noisy.share", "sample");
    data.Set("nn_forward.[count].noisy.count", 3);
    auto module = anet::nn::NetworkModuleRepository::Instance().GetFactory("Linear")->CreateModule(
        data, anet::nn::ModuleContext{ .random_source = MakeTestModuleRandomSource() });
    auto owner = MakeDotTestNetwork();
    auto state = std::make_shared<anet::nn::NetworkForwardState>(521);
    const anet::nn::NetworkForwardConfig config("nn_forward.[count]", SelectForwardConfigData(data, "nn_forward.[count]"));
    anet::nn::NetworkForwardCallInfo::Params context_params{
        .clock = 0,
    };
    auto owned_module_forward = module->CreateExecutor(config, anet::nn::NetworkForwardClock::kContinuingRows);
    auto* module_forward = owned_module_forward ? owned_module_forward.get() : module.get();
    const auto input = torch::ones({ 2, 4 });
    const auto initial = module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params))).detach().clone();
    CHECK(torch::equal(module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params))), initial));

    // episode をまたいでも抽選時計からの3回を保ち、同じ時計でも保持する。
    context_params.episode_start = torch::ones({ 2 }, torch::kBool);
    context_params.clock = 1;
    CHECK(torch::equal(module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params))), initial));
    context_params.clock = 2;
    CHECK(torch::equal(module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params))), initial));
    context_params.clock = 3;
    const auto second = module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params))).detach().clone();
    CHECK_FALSE(torch::equal(second, initial));
    CHECK(torch::equal(module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params))), second));
    context_params.clock = 4;
    CHECK(torch::equal(module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params))), second));
    context_params.clock = 6;
    CHECK_FALSE(torch::equal(module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params))), second));
}

TEST_CASE("Noisy Linear episode masks update each lane once per clock", "[nn][noisynet][episode]")
{
    EnsureNNInitialized();
    anet::ConfigData data;
    data.Set("linear.out_features", 3);
    data.Set("init.mode", "constant");
    data.Set("init.constant_val", 0.125f);
    data.Set("noisy.enabled", true);
    data.Set("nn_forward.[episode].noisy.mode", "sample");
    data.Set("nn_forward.[episode].noisy.hold", "episode");
    data.Set("nn_forward.[episode].noisy.share", "sample");
    auto module = anet::nn::NetworkModuleRepository::Instance().GetFactory("Linear")->CreateModule(
        data, anet::nn::ModuleContext{ .random_source = MakeTestModuleRandomSource() });
    auto owner = MakeDotTestNetwork();
    auto state = std::make_shared<anet::nn::NetworkForwardState>(587);
    const anet::nn::NetworkForwardConfig config("nn_forward.[episode]", SelectForwardConfigData(data, "nn_forward.[episode]"));
    anet::nn::NetworkForwardCallInfo::Params context_params{
        .clock = 0,
        .episode_start = torch::ones({ 2 }, torch::kBool),
    };
    auto owned_module_forward = module->CreateExecutor(config, anet::nn::NetworkForwardClock::kContinuingRows);
    auto* module_forward = owned_module_forward ? owned_module_forward.get() : module.get();
    const auto input = torch::ones({ 2, 4 });
    const auto initial = module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params))).detach().clone();
    CHECK(torch::equal(module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params))), initial));

    // 境界の lane だけを更新し、同じ時計の再 forward は mask が違っても再抽選しない。
    context_params.clock = 1;
    context_params.episode_start = torch::tensor({ false, true }, torch::kBool);
    const auto second = module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params))).detach().clone();
    CHECK(torch::equal(second.select(0, 0), initial.select(0, 0)));
    CHECK_FALSE(torch::equal(second.select(0, 1), initial.select(0, 1)));
    context_params.episode_start = torch::tensor({ true, false }, torch::kBool);
    CHECK(torch::equal(module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params))), second));
    context_params.clock = 2;
    context_params.episode_start = torch::zeros({ 2 }, torch::kBool);
    CHECK(torch::equal(module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params))), second));
    context_params.clock = 3;
    context_params.episode_start = torch::tensor({ true, false }, torch::kBool);
    const auto third = module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params))).detach().clone();
    CHECK_FALSE(torch::equal(third.select(0, 0), second.select(0, 0)));
    CHECK(torch::equal(third.select(0, 1), second.select(0, 1)));

    // μ の snapshot 相当の変更は、保持中の ε を再抽選する境界にしない。
    {
        torch::NoGradGuard no_grad;
        module->named_parameters()["linear.weight"].add_(0.25f);
    }
    CHECK(torch::allclose(module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params))), third + 1.0f, 1e-5, 1e-6));
}

TEST_CASE("Noisy Linear row mapping reuses transition factors in a joint batch", "[nn][noisynet][row_mapping]")
{
    EnsureNNInitialized();
    anet::ConfigData data;
    data.Set("linear.out_features", 3);
    data.Set("init.mode", "constant");
    data.Set("init.constant_val", 0.125f);
    data.Set("noisy.enabled", true);
    data.Set("nn_forward.[sample].noisy.mode", "sample");
    data.Set("nn_forward.[sample].noisy.hold", "call");
    data.Set("nn_forward.[sample].noisy.share", "sample");
    auto module = anet::nn::NetworkModuleRepository::Instance().GetFactory("Linear")->CreateModule(
        data, anet::nn::ModuleContext{ .random_source = MakeTestModuleRandomSource() });
    auto owner = MakeDotTestNetwork();
    auto state = std::make_shared<anet::nn::NetworkForwardState>(631);
    const anet::nn::NetworkForwardConfig config("nn_forward.[sample]", SelectForwardConfigData(data, "nn_forward.[sample]"));
    anet::nn::NetworkForwardCallInfo::Params single_params{
        .clock = 0,
    };
    auto owned_module_forward = module->CreateExecutor(config, anet::nn::NetworkForwardClock::kTransientRows);
    auto* module_forward = owned_module_forward ? owned_module_forward.get() : module.get();
    const auto input = torch::ones({ 2, 4 });
    const auto separate = module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(single_params))).detach().clone();
    auto joint_params = single_params;
    joint_params.row_mapping.emplace(torch::tensor({ 0, 1, 0, 1 }, torch::kLong), 2, torch::Device(torch::kCPU));
    const auto joined = torch::cat({ input, input }, 0);

    // 結合 batch の current/next 行を元遷移へ対応させ、同じ時計の ε を再利用する。
    CHECK(torch::equal(module_forward->Forward(joined, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(joint_params))), torch::cat({ separate, separate }, 0)));
    CHECK(torch::equal(module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(single_params))), separate));
    joint_params.clock = 1;
    const auto next = module_forward->Forward(joined, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(joint_params))).detach().clone();
    CHECK_FALSE(torch::equal(next, torch::cat({ separate, separate }, 0)));
    CHECK(torch::equal(next.slice(0, 0, 2), next.slice(0, 2, 4)));
    joint_params.row_mapping.emplace(torch::tensor({ 1, 0, 1, 0 }, torch::kLong), 2, torch::Device(torch::kCPU));
    const auto reordered = module_forward->Forward(joined, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(joint_params)));
    CHECK(torch::equal(reordered.select(0, 0), next.select(0, 1)));
    CHECK(torch::equal(reordered.select(0, 1), next.select(0, 0)));
}

TEST_CASE("Linear force_fp32 protects both default and sampled projections independently of Noisy", "[nn][noisynet][precision]")
{
    EnsureNNInitialized();
    const auto device_type = GENERATE(torch::kCPU, torch::kCUDA);
    if (device_type == torch::kCUDA && !torch::cuda::is_available()) SKIP("CUDA unavailable");
    const torch::Device device(device_type, device_type == torch::kCUDA ? 0 : -1);
    const auto noisy = GENERATE(false, true);
    const auto force = GENERATE(false, true);
    anet::ConfigData data;
    data.Set("linear.out_features", 3);
    data.Set("init.mode", "constant");
    data.Set("init.constant_val", 0.125f);
    data.Set("noisy.enabled", noisy);
    data.Set("force_fp32", force);
    data.Set("nn_forward.[sample].noisy.mode", "sample");
    data.Set("nn_forward.[sample].noisy.hold", "call");
    data.Set("nn_forward.[sample].noisy.share", "batch");
    auto module = anet::nn::NetworkModuleRepository::Instance().GetFactory("Linear")->CreateModule(
        data, anet::nn::ModuleContext{ .random_source = MakeTestModuleRandomSource() });
    auto owner = MakeDotTestNetwork();
    auto state = std::make_shared<anet::nn::NetworkForwardState>(1009);
    const anet::nn::NetworkForwardConfig config("nn_forward.[sample]", SelectForwardConfigData(data, "nn_forward.[sample]"));
    const anet::nn::NetworkForwardCallInfo context_info({
        .clock = 0,
    });
    const anet::nn::NetworkForwardContext context(state.get(), context_info);
    auto owned_module_forward = module->CreateExecutor(config, anet::nn::NetworkForwardClock::kTransientRows);
    auto* module_forward = owned_module_forward ? owned_module_forward.get() : module.get();
    const auto input = torch::ones({ 2, 4 }, torch::TensorOptions().device(device));
    module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
    module->to(device);
    const auto expected_dtype = force ? torch::kFloat32 : torch::kBFloat16;
    CAPTURE(device, noisy, force);
    {
        // 外側の AMP を継承する層と、FP32 を明示する層を同じ入力で確認する。
        anet::Autocast autocast(device, true, torch::kBFloat16);
        CHECK(module->Forward(input, anet::nn::NetworkForwardContext::GetDefault()).scalar_type() == expected_dtype);
        CHECK(module_forward->Forward(input, context).scalar_type() == expected_dtype);
        if (force) {
            CHECK(module_forward->Forward(input.to(torch::kBFloat16), context).scalar_type() == torch::kFloat32);
        }
    }
    CHECK(module->named_parameters()["linear.weight"].scalar_type() == torch::kFloat32);
    CHECK(module->GetCurrentConfigData().Get<bool>("force_fp32") == force);
}

TEST_CASE("Noisy spectral Linear normalizes only the mean and preserves spectral buffers", "[nn][noisynet][spectral_norm]")
{
    EnsureNNInitialized();
    const auto mode = GENERATE(std::string("spectral"), std::string("spectral_cap"));
    const auto independent = GENERATE(false, true);
    anet::ConfigData data;
    data.Set("linear.out_features", 3);
    data.Set("init.mode", "constant");
    data.Set("init.constant_val", 1.0);
    data.Set("weight_norm.mode", mode);
    data.Set("noisy.enabled", true);
    data.Set("nn_forward.[sample].noisy.mode", "sample");
    data.Set("nn_forward.[sample].noisy.hold", "call");
    data.Set("nn_forward.[sample].noisy.share", independent ? "sample" : "batch");
    auto module = anet::nn::NetworkModuleRepository::Instance().GetFactory("Linear")->CreateModule(
        data, anet::nn::ModuleContext{ .random_source = MakeTestModuleRandomSource() });
    const auto input = torch::arange(32, torch::kFloat32).reshape({ 2, 4, 4 }) * 0.03;
    module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
    module->eval();
    const auto entry = module->GetSpectralNormEntries().front();
    const auto initial_u = entry.u.detach().clone();
    const auto initial_v = entry.v.detach().clone();
    const auto denominator = anet::nn::ComputeSpectralSigma(entry.weight, entry.u, entry.v);
    const auto effective_denominator = mode == "spectral_cap" ? denominator.abs().clamp_min(1.0) : denominator;
    auto parameters = module->named_parameters();
    const auto mean = torch::linear(input, parameters["linear.weight"] / effective_denominator, parameters["linear.bias"]);
    const anet::seed_t seed = 1051;
    anet::RandomGenerator reference_random(seed);
    const auto generator = reference_random.GetTorchGenerator(input.device());
    const auto input_shape = independent ? std::vector<int64_t>{ 2, 4 } : std::vector<int64_t>{ 4 };
    const auto output_shape = independent ? std::vector<int64_t>{ 2, 3 } : std::vector<int64_t>{ 3 };
    const auto raw_input = torch::randn(input_shape, generator, input.options());
    const auto raw_output = torch::randn(output_shape, generator, input.options());
    const auto epsilon_input = raw_input.sign() * raw_input.abs().sqrt();
    const auto epsilon_output = raw_output.sign() * raw_output.abs().sqrt();
    const auto mean_weight = parameters["linear.weight"] / effective_denominator;
    torch::Tensor reference;
    if (independent) {
        // 参照だけで [B,q,p] を合成し、σ と bias が SN 分母で割られないことを検証する。
        const auto weight = mean_weight.unsqueeze(0) + parameters["linear.sigma_weight"].unsqueeze(0)
            * epsilon_output.unsqueeze(2) * epsilon_input.unsqueeze(1);
        const auto bias = parameters["linear.bias"].unsqueeze(0) + parameters["linear.sigma_bias"].unsqueeze(0) * epsilon_output;
        reference = torch::bmm(input, weight.transpose(1, 2)) + bias.unsqueeze(1);
    } else {
        reference = torch::linear(input,
            mean_weight + parameters["linear.sigma_weight"] * epsilon_output.unsqueeze(1) * epsilon_input.unsqueeze(0),
            parameters["linear.bias"] + parameters["linear.sigma_bias"] * epsilon_output);
    }
    auto owner = MakeDotTestNetwork();
    auto state = std::make_shared<anet::nn::NetworkForwardState>(seed);
    const anet::nn::NetworkForwardConfig config("nn_forward.[sample]", SelectForwardConfigData(data, "nn_forward.[sample]"));
    anet::nn::NetworkForwardCallInfo::Params context_params{
        .clock = 0,
    };
    auto owned_module_forward = module->CreateExecutor(config, anet::nn::NetworkForwardClock::kTransientRows);
    auto* module_forward = owned_module_forward ? owned_module_forward.get() : module.get();
    CAPTURE(mode, independent);
    const auto output = module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params)));
    CHECK(torch::allclose(output, reference, 1e-5, 1e-6));
    CHECK(torch::allclose(module->Forward(input, anet::nn::NetworkForwardContext::GetDefault()), mean, 1e-5, 1e-6));
    context_params.clock = 1;
    CHECK_FALSE(torch::equal(module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params))), output));
    CHECK(torch::equal(entry.u, initial_u));
    CHECK(torch::equal(entry.v, initial_v));
    CHECK(torch::equal(anet::nn::ComputeSpectralSigma(entry.weight, entry.u, entry.v), denominator));
    {
        anet::Autocast autocast(torch::Device(torch::kCPU), true, torch::kBFloat16);
        CHECK(module_forward->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(context_params))).scalar_type() == torch::kBFloat16);
        CHECK(anet::nn::ComputeSpectralSigma(entry.weight, entry.u, entry.v).scalar_type() == torch::kFloat32);
    }
}

TEST_CASE("Network validates local forward requirements before a configured computation", "[nn][noisynet][validation]")
{
    EnsureNNInitialized();
    anet::ConfigData data;
    data.Set("net.block.[Linear].type", "Linear");
    data.Set("net.block.[Linear].linear.out_features", 3);
    data.Set("net.block.[Linear].noisy.enabled", true);
    data.Set("net.branch.[feature].bind", "obs");
    data.Set("net.branch.[feature].structure", "Linear");
    data.Set("net.body.output.[features]", "feature");
    const anet::TensorSpec spec{ .type = anet::SpaceType::Vector, .shape = { 2 }, .dtype = torch::kFloat32 };
    auto network = anet::nn::NetworkBuilder::BuildNetwork(
        anet::nn::NetworkConfig(data), anet::TensorSpecMap{ { "obs", spec } }, nullptr, 1093);
    CHECK_NOTHROW(MakeDefaultExecutor(network));
    data.Set("nn_forward.[missing].noisy.mode", "sample");
    const anet::nn::NetworkForwardConfig missing("nn_forward.[missing]", SelectForwardConfigData(data, "nn_forward.[missing]"));
    CHECK_THROWS_WITH(network->CreateForwardExecutor(missing, anet::nn::NetworkForwardClock::kContinuingRows, 0),
        Catch::Matchers::ContainsSubstring("nn_forward.[missing]") && Catch::Matchers::ContainsSubstring("noisy.hold"));
    data.Set("nn_forward.[valid].noisy.mode", "sample");
    data.Set("nn_forward.[valid].noisy.hold", "call");
    data.Set("nn_forward.[valid].noisy.share", "batch");
    const anet::nn::NetworkForwardConfig valid("nn_forward.[valid]", SelectForwardConfigData(data, "nn_forward.[valid]"));
    CHECK_NOTHROW(network->CreateForwardExecutor(valid, anet::nn::NetworkForwardClock::kTransientRows, 0));
    data.Set("nn_forward.[actor].noisy.mode", "sample");
    data.Set("nn_forward.[actor].noisy.hold", "count");
    data.Set("nn_forward.[actor].noisy.count", 3);
    data.Set("nn_forward.[actor].noisy.share", "sample");
    const anet::nn::NetworkForwardConfig actor("nn_forward.[actor]", SelectForwardConfigData(data, "nn_forward.[actor]"));
    CHECK_NOTHROW(network->CreateForwardExecutor(actor, anet::nn::NetworkForwardClock::kContinuingRows, 0));
    CHECK_THROWS_WITH(network->CreateForwardExecutor(actor, anet::nn::NetworkForwardClock::kTransientRows, 0),
        Catch::Matchers::ContainsSubstring("nn_forward.[actor]") && Catch::Matchers::ContainsSubstring("continuing rows"));
    data.Set("nn_forward.[mean].noisy.mode", "mu_only");
    const anet::nn::NetworkForwardConfig mean("nn_forward.[mean]", SelectForwardConfigData(data, "nn_forward.[mean]"));
    CHECK_NOTHROW(network->CreateForwardExecutor(mean, anet::nn::NetworkForwardClock::kTransientRows, 0));
}

TEST_CASE("Used forward values are validated by the concrete preparation", "[nn][noisynet][config]")
{
    EnsureNNInitialized();
    anet::ConfigData structural;
    structural.Set("linear.out_features", 3);
    structural.Set("noisy.enabled", true);
    auto module = anet::nn::NetworkModuleRepository::Instance().GetFactory("Linear")->CreateModule(
        structural, anet::nn::ModuleContext{.random_source=MakeTestModuleRandomSource()});
    module->Forward(torch::ones({2, 4}), anet::nn::NetworkForwardContext::GetDefault());
    const auto validate = [&](const anet::ConfigData& data) {
        const anet::nn::NetworkForwardConfig config("nn_forward.[used]", data);
        module->CreateExecutor(config, anet::nn::NetworkForwardClock::kContinuingRows);
    };
    anet::ConfigData data;
    data.Set("noisy.mode", "mu_only");
    CHECK_NOTHROW(validate(data));
    for (const auto& [leaf, value] : std::vector<std::pair<std::string, std::string>>{
        {"mode", "other"}, {"hold", "other"}, {"share", "other"}, {"count", "0"},
        {"count", "-1"}, {"count", "1.5"},
    }) {
        auto invalid = data;
        invalid.Set("noisy." + leaf, value);
        CAPTURE(leaf, value);
        CHECK_THROWS_WITH(validate(invalid), Catch::Matchers::ContainsSubstring("noisy." + leaf));
    }
    data.Set("noisy.mode", "sample");
    data.Set("noisy.hold", "episode");
    data.Set("noisy.share", "batch");
    CHECK_THROWS_WITH(validate(data), Catch::Matchers::ContainsSubstring("episode") && Catch::Matchers::ContainsSubstring("sample"));
    data.Set("noisy.share", "sample");
    CHECK_NOTHROW(validate(data));
    data.Set("noisy.hold", "count");
    CHECK_THROWS_WITH(validate(data), Catch::Matchers::ContainsSubstring("noisy.count"));
    // 同じ項目も、その設定群を利用しない通常部品では休眠する。
    structural.Set("noisy.enabled", false);
    auto ordinary = anet::nn::NetworkModuleRepository::Instance().GetFactory("Linear")->CreateModule(
        structural, anet::nn::ModuleContext{.random_source=MakeTestModuleRandomSource()});
    anet::ConfigData unused;
    unused.Set("noisy.mode", "invalid");
    CHECK_FALSE(ordinary->CreateExecutor(anet::nn::NetworkForwardConfig("unused", unused),
        anet::nn::NetworkForwardClock::kTransientRows));
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

TEST_CASE("Forward row mapping owns validated source indices", "[nn][noisynet][row_mapping]")
{
    // 呼び出し側の入力を変更しても、一度確定した対応は変わらない。
    auto indices = torch::tensor({ 0, 1, 0, 1 }, torch::kInt64);
    const anet::nn::NetworkForwardRowMapping mapping(indices, 2, torch::Device(torch::kCPU));
    indices.fill_(0);
    CHECK(mapping.GetSourceRows() == 2);
    CHECK(torch::equal(mapping.GetIndices(), torch::tensor({ 0, 1, 0, 1 }, torch::kInt64)));
}

TEST_CASE("Forward row mapping rejects invalid source declarations", "[nn][noisynet][row_mapping]")
{
    // CPU の宣言で判断できる誤りは、転送や射影まで潜伏させない。
    const auto invalid = GENERATE(-1, 2);
    CHECK_THROWS(anet::nn::NetworkForwardRowMapping(
        torch::tensor({0, invalid}, torch::kInt64), 2, torch::Device(torch::kCPU)));
    CHECK_THROWS(anet::nn::NetworkForwardRowMapping(
        torch::tensor({0, 1}, torch::kInt64), 0, torch::Device(torch::kCPU)));
    CHECK_THROWS(anet::nn::NetworkForwardRowMapping(
        torch::ones({2}, torch::kFloat32), 2, torch::Device(torch::kCPU)));
    CHECK_THROWS(anet::nn::NetworkForwardRowMapping(
        torch::ones({1, 2}, torch::kInt64), 2, torch::Device(torch::kCPU)));
}

TEST_CASE("Prepared projection owns same-named settings independently", "[nn][noisynet][prepared]")
{
    anet::ConfigData structural;
    structural.Set("noisy.enabled", true);
    anet::nn::LinearProjectionModule projection(structural);
    projection.Bind(torch::nn::Linear(4, 3));
    auto state = std::make_shared<anet::nn::NetworkForwardState>(701);
    const auto input = torch::ones({2, 4});
    const auto mean = projection.Forward(input, anet::nn::NetworkForwardContext::GetDefault());
    std::unique_ptr<anet::nn::LinearProjectionModule::Executor> sample;
    {
        anet::ConfigData data;
        data.Set("noisy.mode", "sample");
        data.Set("noisy.hold", "call");
        data.Set("noisy.share", "batch");
        const anet::nn::NetworkForwardConfig config("same", data);
        sample = projection.CreateProjectionExecutor(config, anet::nn::NetworkForwardClock::kTransientRows);
        data.Set("noisy.mode", "mu_only");
    }
    // 設定元の寿命と変更から独立した値を、別々の準備結果が保持する。
    const anet::nn::NetworkForwardCallInfo context_info({.clock=0});
    const anet::nn::NetworkForwardContext context(state.get(), context_info);
    const auto noisy = sample->Forward(input, context);
    anet::ConfigData data;
    data.Set("noisy.mode", "mu_only");
    const anet::nn::NetworkForwardConfig same_key("same", data);
    CHECK_FALSE(projection.CreateProjectionExecutor(same_key, anet::nn::NetworkForwardClock::kTransientRows));
    CHECK(torch::equal(mean, projection.Forward(input, anet::nn::NetworkForwardContext::GetDefault())));
    CHECK_FALSE(torch::equal(mean, noisy));
    CHECK(torch::equal(noisy, sample->Forward(input, context)));
}

TEST_CASE("Forward row mapping inspection cannot mutate held correspondence", "[nn][noisynet][row_mapping]")
{
    // 検査用に取得した Tensor は独立した値であり、共有する対応を変更しない。
    const anet::nn::NetworkForwardRowMapping mapping(
        torch::tensor({0, 1, 0, 1}, torch::kInt64), 2, torch::Device(torch::kCPU));
    auto inspection = mapping.GetIndices();
    inspection.fill_(0);
    CHECK(torch::equal(mapping.GetIndices(), torch::tensor({0, 1, 0, 1}, torch::kInt64)));
}


TEST_CASE("Prepared projection permits concurrent callers with private states", "[nn][noisynet][prepared][concurrency]")
{
    EnsureNNInitialized();
    anet::ConfigData structural;
    structural.Set("noisy.enabled", true);
    anet::nn::LinearProjectionModule projection(structural);
    projection.Bind(torch::nn::Linear(4, 3));
    auto owner = MakeDotTestNetwork();
    const auto input = torch::ones({2, 4});
    std::vector<std::future<torch::Tensor>> futures;
    // 重みだけを共有し、各呼び出し側の準備結果・箱・RNG は独立させる。
    for (int i = 0; i < 4; ++i) {
        auto state = std::make_shared<anet::nn::NetworkForwardState>(701 + i / 2);
        futures.push_back(std::async(std::launch::async,
            [&, state = std::move(state)] {
                torch::NoGradGuard guard;
                anet::ConfigData data;
                data.Set("local.noisy.mode", "sample");
                data.Set("local.noisy.hold", "call");
                data.Set("local.noisy.share", "batch");
                const anet::nn::NetworkForwardConfig config("local", SelectForwardConfigData(data, "local"));
                auto prepared = projection.CreateProjectionExecutor(config, anet::nn::NetworkForwardClock::kContinuingRows);
                const anet::nn::NetworkForwardCallInfo context_info({.clock=0});
                const anet::nn::NetworkForwardContext context(state.get(), context_info);
                const auto first = prepared->Forward(input, context);
                if (!torch::equal(first, prepared->Forward(input, context))) {
                    throw std::runtime_error("Concurrent caller changed its held draw");
                }
                return first;
            }));
    }
    std::vector<torch::Tensor> outputs;
    for (auto& future : futures) outputs.push_back(future.get());
    CHECK(torch::equal(outputs[0], outputs[1]));
    CHECK(torch::equal(outputs[2], outputs[3]));
    CHECK_FALSE(torch::equal(outputs[0], outputs[2]));
}

TEST_CASE("Validated row mapping reuses device indices without transfer or scalar extraction", "[nn][noisynet][row_mapping][profile]")
{
    EnsureNNInitialized();
    const auto device_type = GENERATE(torch::kCPU, torch::kCUDA);
    if (device_type == torch::kCUDA && !torch::cuda::is_available()) SKIP("CUDA unavailable");
    const torch::Device device(device_type, device_type == torch::kCUDA ? 0 : -1);
    const bool rank3 = GENERATE(false, true);
    CAPTURE(device, rank3);
    torch::NoGradGuard guard;
    anet::ConfigData structural;
    structural.Set("noisy.enabled", true);
    anet::nn::LinearProjectionModule projection(structural);
    auto native = torch::nn::Linear(4, 3);
    projection.Bind(native);
    native->to(device);
    auto owner = MakeDotTestNetwork();
    auto state = std::make_shared<anet::nn::NetworkForwardState>(702);
    anet::ConfigData data;
    data.Set("sample.noisy.mode", "sample");
    data.Set("sample.noisy.hold", "call");
    data.Set("sample.noisy.share", "sample");
    const anet::nn::NetworkForwardConfig config("sample", SelectForwardConfigData(data, "sample"));
    auto prepared = projection.CreateProjectionExecutor(config, anet::nn::NetworkForwardClock::kTransientRows);
    const auto input = torch::arange(rank3 ? 24 : 8,
        torch::TensorOptions().dtype(torch::kFloat32).device(device))
        .reshape(rank3 ? std::vector<int64_t>{2, 3, 4} : std::vector<int64_t>{2, 4});
    const anet::nn::NetworkForwardCallInfo separate_info({.clock=0});
    const anet::nn::NetworkForwardContext separate(state.get(), separate_info);
    const auto expected = prepared->Forward(input, separate);
    anet::nn::NetworkForwardCallInfo::Params joint_params{.clock=0};
    joint_params.row_mapping.emplace(torch::tensor({0, 1, 0, 1}, torch::kInt64), 2, device);
    const auto joined = torch::cat({input, input}, 0);
    CHECK(torch::allclose(prepared->Forward(joined, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(joint_params))), torch::cat({expected, expected}, 0), 1e-5, 1e-6));
    // 宣言の転送を終えた後の forward に GPU->CPU 転送や min/max の scalar 取得がない。
    CHECK(CountProfileOperator([&] { prepared->Forward(joined, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(joint_params))); }, "aten::_to_copy") == 0);
    CHECK(CountProfileOperator([&] { prepared->Forward(joined, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(joint_params))); }, "aten::_local_scalar_dense") == 0);
    CHECK_THROWS(prepared->Forward(input, anet::nn::NetworkForwardContext(state.get(), anet::nn::NetworkForwardCallInfo(joint_params))));
    CHECK_THROWS(joint_params.row_mapping->SelectRows(torch::ones({3, 4}, input.options())));
    if (device_type == torch::kCUDA) {
        CHECK_THROWS(joint_params.row_mapping->SelectRows(torch::ones({2, 4})));
        CHECK_THROWS(anet::nn::NetworkForwardRowMapping(
            torch::tensor({0, 1}, torch::kInt64).to(device), 2, device));
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
        const auto output = forward->Forward(input, info, {}, &capture);
        CHECK(output.At("q").scalar_type() == torch::kFloat32);
        CHECK(capture.Get("feature").scalar_type() == (force ? torch::kFloat32 : torch::kBFloat16));
        for (const auto& entry : network->GetSpectralNormEntries()) {
            CHECK(entry.u.scalar_type() == torch::kFloat32);
            CHECK(entry.v.scalar_type() == torch::kFloat32);
            CHECK(anet::nn::ComputeSpectralSigma(entry.weight, entry.u, entry.v).scalar_type() == torch::kFloat32);
        }
    }
}


TEST_CASE("Internal context references immutable call info", "[nn][noisynet][forward_context]")
{
    using Context = anet::nn::NetworkForwardContext;
    static_assert(!std::is_copy_assignable_v<Context>);
    static_assert(!std::is_move_assignable_v<Context>);
    anet::nn::NetworkForwardState state(17);
    anet::nn::NetworkForwardCallInfo::Params params{
        .clock = 7,
        .episode_start = torch::tensor({true, false}, torch::kBool),
        .row_mapping = anet::nn::NetworkForwardRowMapping(torch::tensor({0, 1}, torch::kInt64), 2, torch::kCPU),
    };
    const anet::nn::NetworkForwardCallInfo info(params);
    const Context context(&state, info);
    // 元のParamsは変更できても、不変CallInfoとContextの参照先は変えない。
    params.clock = 8;
    params.episode_start = torch::zeros({2}, torch::kBool);
    CHECK(context.GetClock() == 7);
    CHECK(context.GetState() == &state);
    CHECK(&context.GetEpisodeStart() == &info.GetEpisodeStart());
    CHECK(&context.GetRowMapping() == &info.GetRowMapping());
    const Context copied(context);
    CHECK(&copied.GetEpisodeStart() == &info.GetEpisodeStart());
    const auto fallback = Context::GetDefault();
    CHECK(fallback.GetState() == nullptr);
    CHECK(fallback.GetClock() == 0);
}

TEST_CASE("Network forward call info replaces only row mapping in its copy", "[nn][noisynet][forward_context]")
{
    EnsureNNInitialized();
    using Context = anet::nn::NetworkForwardCallInfo;
    anet::ConfigData data;
    data.Set("custom.value", 3);
    const anet::nn::NetworkForwardConfig config("custom", SelectForwardConfigData(data, "custom"));
    const auto mask = torch::tensor({ true, false }, torch::kBool);
    const Context source({
        .clock = 19,
        .episode_start = mask,
        .row_mapping = anet::nn::NetworkForwardRowMapping(
            torch::tensor({ 0, 1 }, torch::kInt64), 2, torch::Device(torch::kCPU)),
    });

    // 差し替え以外の情報は設定・状態の同一性を含めてそのまま保持する。
    const auto replaced = source.WithRowMapping(anet::nn::NetworkForwardRowMapping(
        torch::tensor({ 1, 0, 1, 0 }, torch::kInt64), 2, torch::Device(torch::kCPU)));
    CHECK(replaced.GetClock() == 19);
    CHECK(replaced.GetEpisodeStart().is_same(mask));
    REQUIRE(replaced.GetRowMapping().has_value());
    CHECK(torch::equal(replaced.GetRowMapping()->GetIndices(), torch::tensor({ 1, 0, 1, 0 }, torch::kInt64)));
    CHECK(torch::equal(source.GetRowMapping()->GetIndices(), torch::tensor({ 0, 1 }, torch::kInt64)));

    // 行対応を外した複製も元の Context と差し替え済みの Context を変更しない。
    const auto cleared = replaced.WithRowMapping(std::nullopt);
    CHECK_FALSE(cleared.GetRowMapping().has_value());
    CHECK(cleared.GetClock() == 19);
    CHECK(replaced.GetRowMapping()->GetRows() == 4);
}

TEST_CASE("Prepared forward interprets concrete settings once and owns network lifetime", "[nn][noisynet][prepared]")
{
    class ScalingModule final : public anet::nn::NetworkModule {
    public:
        explicit ScalingModule(std::shared_ptr<int> preparations) : preparations_(std::move(preparations)) {}
        torch::Tensor Forward(torch::Tensor input, const anet::nn::NetworkForwardContext&) override
        { return input; }
        std::unique_ptr<anet::nn::NetworkModuleExecutor> CreateExecutor(
            const anet::nn::NetworkForwardConfig& config, anet::nn::NetworkForwardClock) override
        {
            ++*preparations_;
            // 型付き設定と準備結果は、この具象 Module の内部だけに置く。
            return std::make_unique<Execution>(config.IsDefault() ? 1.0f
                : config.GetConfigData().Get<float>("independent.scale"));
        }
    private:
        class Execution final : public anet::nn::NetworkModuleExecutor {
        public:
            explicit Execution(float scale) : scale_(scale) {}
            torch::Tensor Forward(torch::Tensor input, const anet::nn::NetworkForwardContext&) override
            { return input * scale_; }
        private:
            const float scale_;
        };
        const std::shared_ptr<int> preparations_;
    };
    const auto preparations = std::make_shared<int>(0);
    auto module = std::make_shared<ScalingModule>(preparations);
    auto block = std::make_shared<anet::nn::NetworkBlock>("scale", module);
    auto structure = std::make_shared<anet::nn::NetworkStruct>(
        std::vector<std::shared_ptr<anet::nn::NetworkBlock>>{block});
    auto branch = std::make_shared<anet::nn::NetworkBranch>(
        "feature", std::vector<std::vector<std::string>>{{"obs"}}, 1, structure);
    const anet::TensorSpecMap specs{{"obs", {.type=anet::SpaceType::Vector, .shape={2}, .dtype=torch::kFloat32}}};
    anet::nn::NetworkConfig structure_config;
    structure_config.output_keys["features"] = "feature";
    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{branch}, specs,
        std::vector<std::string>{}, structure_config.output_keys);
    auto network = std::make_shared<anet::nn::Network>(structure_config, specs, nullptr, body, nullptr);

    anet::ConfigData data;
    data.Set("independent.scale", 3.0f);
    auto config = std::make_unique<const anet::nn::NetworkForwardConfig>("custom", data);
    auto execution = network->CreateForwardExecutor(*config, anet::nn::NetworkForwardClock::kTransientRows, 991);
    CHECK(*preparations == 1);
    auto independent = network->CreateForwardExecutor(*config, anet::nn::NetworkForwardClock::kTransientRows, 992);
    config.reset();
    data.Set("independent.scale", 99.0f);
    network.reset();
    const anet::TensorDict input{{"obs", torch::ones({2, 2})}};
    for (uint64_t clock = 0; clock < 3; ++clock) {
        const anet::nn::NetworkForwardCallInfo info({.clock=clock});
        CHECK(torch::equal(execution->Forward(input, info).At("features"), torch::full({2, 2}, 3.0f)));
        CHECK(torch::equal(independent->Forward(input, info).At("features"), torch::full({2, 2}, 3.0f)));
        CHECK(torch::equal(execution->Forward(input, info).At("features"), torch::full({2, 2}, 3.0f)));
    }
    CHECK(*preparations == 2);
}

TEST_CASE("Prepared Noisy forward samples once at the supplied clock", "[nn][noisynet][prepared][sample]")
{
    EnsureNNInitialized();
    anet::ConfigData data;
    data.Set("net.block.[Linear].type", "Linear");
    data.Set("net.block.[Linear].linear.out_features", 3);
    data.Set("net.block.[Linear].noisy.enabled", true);
    data.Set("net.block.[Linear].init.mode", "constant");
    data.Set("net.block.[Linear].init.constant_val", 0.125f);
    data.Set("net.branch.[feature].bind", "obs");
    data.Set("net.branch.[feature].structure", "Linear");
    data.Set("net.body.output.[features]", "feature");
    const anet::TensorSpecMap specs{{"obs", {.type=anet::SpaceType::Vector, .shape={4}, .dtype=torch::kFloat32}}};
    auto network = anet::nn::NetworkBuilder::BuildNetwork(anet::nn::NetworkConfig(data), specs, nullptr, 810);
    anet::ConfigData selected;
    selected.Set("noisy.mode", "sample");
    selected.Set("noisy.hold", "call");
    selected.Set("noisy.share", "batch");
    const anet::nn::NetworkForwardConfig config("selected", selected);
    auto execution = network->CreateForwardExecutor(config, anet::nn::NetworkForwardClock::kTransientRows, 811);
    const anet::TensorDict input{{"obs", torch::ones({2, 4})}};
    const anet::nn::NetworkForwardCallInfo info({.clock=0});
    const auto first = execution->Forward(input, info).At("features");
    CHECK_FALSE(torch::equal(first, network->Forward(input).At("features")));
    CHECK(torch::equal(first, execution->Forward(input, info).At("features")));
    const anet::nn::NetworkForwardCallInfo next({.clock=1});
    CHECK_FALSE(torch::equal(first, execution->Forward(input, next).At("features")));
}

TEST_CASE("NN prepared Head composes configured and ordinary children",
    "[nn][noisynet][prepared-common]")
{
    class ScalingModule final : public anet::nn::NetworkModule {
    public:
        explicit ScalingModule(float initial)
            : weight(register_parameter("weight", torch::tensor(initial))) {}
        torch::Tensor Forward(torch::Tensor input, const anet::nn::NetworkForwardContext&) override
        { return input * weight; }
        std::unique_ptr<anet::nn::NetworkModuleExecutor> CreateExecutor(
            const anet::nn::NetworkForwardConfig& config, anet::nn::NetworkForwardClock) override
        {
            if (config.IsDefault()) return nullptr;
            ++interpretations;
            return std::make_unique<Execution>(*this, config.GetConfigData().Get<float>("independent.scale"));
        }
        torch::Tensor weight;
        int interpretations = 0;
    private:
        class Execution final : public anet::nn::NetworkModuleExecutor {
        public:
            Execution(ScalingModule& owner, float scale) : owner_(owner), scale_(scale) {}
            torch::Tensor Forward(torch::Tensor input, const anet::nn::NetworkForwardContext& context) override
            { return owner_.Forward(std::move(input), context) * scale_; }
        private:
            ScalingModule& owner_;
            const float scale_;
        };
    };
    class OrdinaryModule final : public anet::nn::NetworkModule {
    public:
        OrdinaryModule() : weight(register_parameter("weight", torch::tensor(3.0f))) {}
        torch::Tensor Forward(torch::Tensor input, const anet::nn::NetworkForwardContext&) override
        { return input * weight; }
        torch::Tensor weight;
    };
    class SumHead final : public anet::nn::NetworkHead {
    public:
        SumHead()
            : configured(register_module("configured", std::make_shared<ScalingModule>(2.0f))),
              ordinary(register_module("ordinary", std::make_shared<OrdinaryModule>())) {}
        anet::TensorDict Combine(const anet::TensorDict& input,
            const anet::nn::NetworkForwardContext& context,
            anet::nn::NetworkModuleExecutor& first, anet::nn::NetworkModuleExecutor& second)
        {
            // 子の準備済み処理と本体を同じ計算経路へ渡し、Head固有の合成だけを行う。
            return {{"output", first.Forward(input.At("input"), context)
                + second.Forward(input.At("input"), context)}};
        }
        anet::TensorDict Forward(const anet::TensorDict& input,
            const anet::nn::NetworkForwardContext& context) override
        { return Combine(input, context, *configured, *ordinary); }
        std::unique_ptr<anet::nn::NetworkHeadExecutor> CreateExecutor(
            const anet::nn::NetworkForwardConfig& config, anet::nn::NetworkForwardClock clock) override
        {
            return anet::nn::MakePreparedHeadExecutor<SumHead, &SumHead::Combine>(*this,
                std::array<anet::nn::NetworkModule*, 2>{configured.get(), ordinary.get()}, config, clock);
        }
        const std::shared_ptr<ScalingModule> configured;
        const std::shared_ptr<OrdinaryModule> ordinary;
    };
    SumHead head;
    anet::ConfigData data;
    data.Set("independent.scale", 5.0f);
    const anet::nn::NetworkForwardConfig config("custom", data);
    auto prepared = head.CreateExecutor(config, anet::nn::NetworkForwardClock::kTransientRows);
    REQUIRE(prepared);
    const auto context = anet::nn::NetworkForwardContext::GetDefault();
    const auto input = torch::tensor({1.0f, 2.0f}, torch::requires_grad());
    // 設定元の変更後も準備済み値を使い、通常の子だけは本体へfallbackする。
    data.Set("independent.scale", 99.0f);
    for (int repeat = 0; repeat < 3; ++repeat) {
        CHECK(torch::equal(prepared->Forward({{"input", input}}, context).At("output"), input * 13.0f));
    }
    CHECK(head.configured->interpretations == 1);
    // 同じ計算本体を通した出力と、入力・両子のparameterの勾配を確認する。
    prepared->Forward({{"input", input}}, context).At("output").sum().backward();
    CHECK(torch::equal(input.grad(), torch::full({2}, 13.0f)));
    CHECK(head.configured->weight.grad().item<float>() == 15.0f);
    CHECK(head.ordinary->weight.grad().item<float>() == 3.0f);
    // 全子が既定なら準備済みHeadを作らず、本体への直接経路を残す。
    CHECK_FALSE(head.CreateExecutor(anet::nn::NetworkForwardConfig::GetDefault(),
        anet::nn::NetworkForwardClock::kTransientRows));
    CHECK(torch::equal(head.Forward({{"input", input}}, context).At("output"), input * 5.0f));
    CHECK(head.configured->interpretations == 1);
}

TEST_CASE("Default execution bypasses concrete preparation", "[nn][noisynet][simplify_default]")
{
    class CountingModule final : public anet::nn::NetworkModule {
    public:
        int preparations = 0;
        torch::Tensor Forward(torch::Tensor input, const anet::nn::NetworkForwardContext&) override { return input * 2; }
        std::unique_ptr<anet::nn::NetworkModuleExecutor> CreateExecutor(
            const anet::nn::NetworkForwardConfig&, anet::nn::NetworkForwardClock) override
        { ++preparations; return nullptr; }
    };
    class CountingHead final : public anet::nn::NetworkHead {
    public:
        int preparations = 0;
        anet::TensorDict Forward(const anet::TensorDict& input, const anet::nn::NetworkForwardContext&) override
        { return {{"result", input.At("features") * 3}}; }
        std::unique_ptr<anet::nn::NetworkHeadExecutor> CreateExecutor(
            const anet::nn::NetworkForwardConfig&, anet::nn::NetworkForwardClock) override
        { ++preparations; return nullptr; }
    };
    auto module = std::make_shared<CountingModule>();
    auto head = std::make_shared<CountingHead>();
    auto block = std::make_shared<anet::nn::NetworkBlock>("count", module);
    auto structure = std::make_shared<anet::nn::NetworkStruct>(std::vector<std::shared_ptr<anet::nn::NetworkBlock>>{block});
    auto branch = std::make_shared<anet::nn::NetworkBranch>("feature", std::vector<std::vector<std::string>>{{"obs"}}, 1, structure);
    const anet::TensorSpecMap specs{{"obs", {.type=anet::SpaceType::Vector, .shape={2}, .dtype=torch::kFloat32}}};
    anet::nn::NetworkConfig config;
    config.output_keys["features"] = "feature";
    auto body = std::make_shared<anet::nn::NetworkBody>(std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{branch}, specs,
        std::vector<std::string>{}, config.output_keys);
    auto network = std::make_shared<anet::nn::Network>(config, specs, nullptr, body, head);
    auto execution = network->CreateForwardExecutor(anet::nn::NetworkForwardConfig::GetDefault(),
        anet::nn::NetworkForwardClock::kTransientRows, 0);
    CHECK(module->preparations == 0);
    CHECK(head->preparations == 0);
    auto independent = MakeDefaultExecutor(network);
    CHECK(module->preparations == 0);
    CHECK(head->preparations == 0);
    const auto input = torch::ones({2, 2}, torch::requires_grad());
    const anet::TensorDict observations{{"obs", input}};
    const anet::nn::NetworkForwardCallInfo info({.clock=7});
    const auto actual = execution->Forward(observations, info).At("result");
    CHECK(torch::equal(actual, network->Forward(observations).At("result")));
    CHECK(torch::equal(actual, independent->Forward(observations, info).At("result")));
    actual.sum().backward();
    CHECK(torch::equal(input.grad(), torch::full({2, 2}, 6.0f)));
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

TEST_CASE("Forward capture enforces requested unique keys and preserves write order", "[nn][capture][prd093]")
{
    anet::nn::NetworkForwardCapture capture({ "X/", "branch" });
    const auto value = torch::ones({ 2 });
    CHECK(capture.Wants("X/z"));
    CHECK_FALSE(capture.Wants("X"));
    CHECK_FALSE(capture.Wants("XY/z"));
    CHECK_THROWS_WITH(capture.Add("X", value), Catch::Matchers::ContainsSubstring("unrequested"));
    capture.Add("X/z", value);
    capture.Add("X/a", value + 1);
    CHECK_THROWS_WITH(capture.Add("X/z", value), Catch::Matchers::ContainsSubstring("duplicate"));
    REQUIRE(capture.Unfulfilled() == std::vector<std::string>{ "branch" });
    capture.Add("branch", value);
    CHECK(capture.Unfulfilled().empty());
    const auto all = capture.GetAll("X/");
    REQUIRE(all.size() == 2);
    CHECK(all[0].first == "X/z");
    CHECK(all[1].first == "X/a");
    CHECK(torch::equal(capture.Get("X/a"), value + 1));
    CHECK_FALSE(capture.Get("missing").defined());
    capture.Reset();
    CHECK_FALSE(capture.Get("branch").defined());
    CHECK(capture.GetAll("").empty());
    CHECK(capture.Wants("branch"));
    CHECK(capture.Unfulfilled().size() == 2);
}

TEST_CASE("Network partial and prepared forwards reset capture and reject unmet requests", "[nn][capture][prd093]")
{
    EnsureNNInitialized();
    anet::ConfigData data;
    data.Set("net.branch.[feature].bind", "obs");
    data.Set("net.branch.[feature].structure", "");
    data.Set("net.body.output.[features]", "feature");
    const anet::TensorSpec spec{ .type = anet::SpaceType::Vector, .shape = { 2 }, .dtype = torch::kFloat32 };
    auto network = anet::nn::NetworkBuilder::BuildNetwork(anet::nn::NetworkConfig(data),
        anet::TensorSpecMap{ { "obs", spec } }, nullptr, 0, torch::kCPU);
    anet::TensorDict input{ { "obs", torch::ones({ 2, 2 }, torch::TensorOptions().requires_grad(true)) } };
    anet::nn::NetworkForwardCapture capture({ "feature" });
    network->ForwardUpTo(input, "feature", &capture);
    CHECK_FALSE(capture.Get("feature").requires_grad());
    input.Set("obs", torch::zeros({ 2, 2 }));
    auto executor = network->CreateForwardExecutor(anet::nn::NetworkForwardConfig::GetDefault(),
        anet::nn::NetworkForwardClock::kTransientRows, 0);
    const anet::nn::NetworkForwardCallInfo info(anet::nn::NetworkForwardCallInfo::Params{});
    executor->Forward(input, info, {}, &capture);
    CHECK(torch::equal(capture.Get("feature"), input.At("obs")));
    for (const auto& request : { "missing", "TransformerEncoder/attention_entropy/" }) {
        anet::nn::NetworkForwardCapture missing({ request });
        CHECK_THROWS_WITH(network->Forward(input, {}, &missing), Catch::Matchers::ContainsSubstring(request));
        CHECK_THROWS_WITH(network->Forward(input, {}, &missing), Catch::Matchers::ContainsSubstring("reached_branch='feature'"));
        CHECK_THROWS_WITH(network->ForwardUpTo(input, "feature", &missing), Catch::Matchers::ContainsSubstring("reached_branch='feature'"));
    }
    data.Set("net.branch.[bad/name].bind", "obs");
    data.Set("net.branch.[bad/name].structure", "");
    CHECK_THROWS_WITH(anet::nn::NetworkBuilder::BuildNetwork(anet::nn::NetworkConfig(data),
        anet::TensorSpecMap{ { "obs", spec } }, nullptr, 0, torch::kCPU), Catch::Matchers::ContainsSubstring("must not contain '/'") );
}

TEST_CASE("Attention entropy normalizes uniform concentration and singleton sequences", "[nn][attention][prd093]")
{
    const auto input = torch::eye(4).unsqueeze(0).requires_grad_(true);
    const auto zero = torch::zeros({ 8, 4 });
    const auto uniform = anet::ComputeAttentionEntropy(input, zero, {}, 1);
    CHECK(uniform.item<float>() == Catch::Approx(1.0f).margin(1e-6));
    CHECK(uniform.scalar_type() == torch::kFloat32);
    CHECK_FALSE(uniform.requires_grad());
    const auto identity = torch::cat({ torch::eye(4), torch::eye(4) }, 0) * 20;
    CHECK(anet::ComputeAttentionEntropy(input, identity, {}, 1).item<float>() < 1e-5f);
    CHECK(std::isnan(anet::ComputeAttentionEntropy(input.narrow(1, 0, 1), identity, {}, 1).item<float>()));
}

TEST_CASE("Transformer entropy matches independent per-head MHA probabilities", "[nn][attention][prd093][reference]")
{
    EnsureNNInitialized();
    const bool norm_first = GENERATE(false, true);
    const bool cuda = GENERATE(false, true);
    const bool use_sdpa = GENERATE(false, true);
    const auto norm_mode = GENERATE(std::string("none"), std::string("spectral"), std::string("spectral_cap"));
    if (!use_sdpa && norm_mode != "none") return;
    if (cuda && !torch::cuda::is_available()) SKIP("CUDA unavailable; attention BF16 acceptance remains unverified");
    const torch::Device device(cuda ? torch::kCUDA : torch::kCPU);
    anet::ConfigData data;
    data.Set("tf.d_model", 16); data.Set("tf.nhead", 4); data.Set("tf.num_layers", 1);
    data.Set("tf.dim_feedforward", 32); data.Set("tf.norm_first", norm_first);
    data.Set("tf.use_sdpa", use_sdpa); data.Set("weight_norm.mode", norm_mode);
    anet::nn::ModuleContext construction{ .random_source = std::make_shared<anet::nn::ModuleRandomSource>(9301) };
    auto module = anet::nn::NetworkModuleRepository::Instance().GetFactory("TransformerEncoder")->CreateModule(data, construction);
    module->to(device); module->eval();
    torch::NoGradGuard no_grad;
    auto input = torch::randn({ 2, 5, 16 }, torch::TensorOptions().device(device));
    // 非ゼロbiasも参照へ写し、Q/K biasの欠落を検出する。
    GetNamedParameter(*module, "layer_0.self_attn.in_proj_bias").copy_(torch::linspace(-0.3, 0.3, 48, input.options()));
    std::map<std::string, torch::Tensor> buffers;
    for (const auto& item : module->named_buffers()) buffers.emplace(item.key(), item.value().clone());
    const auto key = std::string(anet::nn::kTransformerEncoderAttentionEntropyCaptureKey);
    anet::nn::NetworkForwardCapture capture({ key });
    const auto context = anet::nn::NetworkForwardContext::GetDefault().WithCapture(&capture);
    torch::Tensor actual_output, ordinary_output;
    {
        anet::Autocast amp(device, cuda, torch::kBFloat16);
        ordinary_output = module->Forward(input, anet::nn::NetworkForwardContext::GetDefault());
        actual_output = module->Forward(input, context);
    }
    CHECK(torch::equal(ordinary_output, actual_output));
    REQUIRE(capture.Get(key).sizes() == torch::IntArrayRef({ 1 }));
    CHECK(capture.Get(key).scalar_type() == torch::kFloat32);
    CHECK(capture.Get(key).device() == input.device());
    CHECK_FALSE(capture.Get(key).requires_grad());

    // 本体のQ/K射影コードは使わず、libtorch MHAのhead別確率を独立参照にする。
    anet::Autocast disable_amp(device, false, torch::kFloat32);
    auto weight = GetNamedParameter(*module, "layer_0.self_attn.in_proj_weight").clone();
    if (norm_mode != "none") {
        for (const auto& entry : module->GetSpectralNormEntries()) {
            const int64_t offset = entry.name == "layer_0.q" ? 0 : entry.name == "layer_0.k" ? 16 : -1;
            if (offset < 0) continue;
            const auto raw = anet::nn::ComputeSpectralSigma(entry.weight, entry.u, entry.v);
            const auto denominator = norm_mode == "spectral_cap" ? raw.abs().clamp_min(1.0) : raw;
            weight.narrow(0, offset, 16).copy_(entry.weight / denominator);
        }
    }
    torch::nn::MultiheadAttention mha(torch::nn::MultiheadAttentionOptions(16, 4).dropout(0));
    mha->to(device); mha->eval();
    mha->in_proj_weight.copy_(weight);
    mha->in_proj_bias.copy_(GetNamedParameter(*module, "layer_0.self_attn.in_proj_bias"));
    auto attention_input = input;
    if (norm_first) {
        attention_input = torch::nn::functional::layer_norm(input,
            torch::nn::functional::LayerNormFuncOptions({ 16 })
                .weight(GetNamedParameter(*module, "layer_0.norm1.weight"))
                .bias(GetNamedParameter(*module, "layer_0.norm1.bias")));
    }
    const auto sequence = attention_input.transpose(0, 1);
    const auto probability = std::get<1>(mha->forward(sequence, sequence, sequence, {}, true, {}, false));
    REQUIRE(probability.sizes() == torch::IntArrayRef({ 2, 4, 5, 5 }));
    const auto terms = torch::where(probability.gt(0), probability * probability.log(), torch::zeros_like(probability));
    const auto expected = -terms.sum(-1).mean() / std::log(5.0);
    CHECK(capture.Get(key)[0].item<float>() == Catch::Approx(expected.item<float>()).margin(2e-6));
    for (const auto& item : module->named_buffers()) CHECK(torch::equal(item.value(), buffers.at(item.key())));
}

TEST_CASE("Repeated Transformer captures have stable positions and execution-order layers", "[nn][attention][capture][prd093]")
{
    EnsureNNInitialized();
    anet::ConfigData data;
    data.Set("net.block.[TF].type", "TransformerEncoder");
    data.Set("net.block.[TF].tf.d_model", 4); data.Set("net.block.[TF].tf.nhead", 2);
    data.Set("net.block.[TF].tf.num_layers", 2); data.Set("net.block.[TF].tf.dim_feedforward", 8);
    data.Set("net.branch.[feature].bind", "obs");
    data.Set("net.branch.[feature].structure", "TF(*2) > Flatten");
    data.Set("net.block.[Flatten].type", "Flatten");
    data.Set("net.body.output.[features]", "feature");
    const anet::TensorSpec spec{ .type = anet::SpaceType::Vector, .shape = { 3, 4 }, .dtype = torch::kFloat32 };
    auto network = anet::nn::NetworkBuilder::BuildNetwork(anet::nn::NetworkConfig(data),
        anet::TensorSpecMap{ { "obs", spec } }, nullptr, 93, torch::kCPU);
    network->eval();
    anet::TensorDict input{ { "obs", torch::randn({ 2, 3, 4 }).requires_grad_(true) } };
    const auto prefix = std::string(anet::nn::kTransformerEncoderAttentionEntropyCaptureKey) + "/";
    anet::nn::NetworkForwardCapture capture({ prefix, "feature" });
    const auto ordinary = network->Forward(input);
    const auto partial = network->ForwardUpTo(input, "feature", &capture);
    const auto entries = capture.GetAll(prefix);
    REQUIRE(entries.size() == 2);
    CHECK(entries[0].first == prefix + "feature/TF_0");
    CHECK(entries[1].first == prefix + "feature/TF_1");
    CHECK(torch::equal(partial.At("feature"), ordinary.At("features")));
    CHECK(torch::equal(capture.Get("feature"), partial.At("feature")));
    for (const auto& [key, value] : entries) {
        CHECK(value.sizes() == torch::IntArrayRef({ 2 }));
        CHECK_FALSE(value.requires_grad());
    }
    const auto layers = torch::cat({ entries[0].second, entries[1].second });
    REQUIRE(layers.numel() == 4);
    const auto info = anet::nn::NetworkForwardCallInfo(anet::nn::NetworkForwardCallInfo::Params{});
    auto executor = network->CreateForwardExecutor(anet::nn::NetworkForwardConfig::GetDefault(),
        anet::nn::NetworkForwardClock::kTransientRows, 0);
    executor->Forward(input, info, {}, &capture);
    const auto prepared = capture.GetAll(prefix);
    REQUIRE(prepared.size() == entries.size());
    CHECK(torch::equal(torch::cat({ prepared[0].second, prepared[1].second }), layers));
}
