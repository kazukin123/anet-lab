#include "catch.hpp"

#include "anet/default_dqn_agent.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/muzero_proto_agent.hpp"
#include "dqn_based_heads.hpp"
#include "nn_impl.hpp"
#include "nn_heads.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
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
