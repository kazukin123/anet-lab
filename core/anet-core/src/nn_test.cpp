#include "catch.hpp"

#include "anet/default_dqn_agent.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/muzero_proto_agent.hpp"
#include "anet/nn_util.hpp"
#include "dqn_based_heads.hpp"
#include "nn_impl.hpp"
#include "nn_heads.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace dqn = anet::rl::dqn;
namespace muzero = anet::rl::muzero_proto;

bool Contains(const std::string& text, const std::string& pattern)
{
    return text.find(pattern) != std::string::npos;
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

std::shared_ptr<anet::nn::NetworkModule> MakeTransformerTestModule(bool use_sdpa, bool norm_first)
{
    static const bool initialized = [] {
        anet::nn::InitNN();
        return true;
    }();
    (void)initialized;

    anet::ConfigData config_data;
    config_data.Set("tf.d_model", 16);
    config_data.Set("tf.nhead", 4);
    config_data.Set("tf.num_layers", 1);
    config_data.Set("tf.dim_feedforward", 32);
    config_data.Set("tf.norm_first", norm_first ? "true" : "false");
    config_data.Set("tf.use_sdpa", use_sdpa ? "true" : "false");
    config_data.Set("tf.activation", "gelu");

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
        std::vector<std::string>{ "obs" },
        network_struct);

    anet::nn::NetworkConfig network_config;
    anet::nn::NetworkBranchConfig branch_config;
    branch_config.name = "feature";
    branch_config.bind_keys = { "obs" };
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
        std::vector<std::string>{ "obs" },
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
        std::vector<std::string>{ "obs" },
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
