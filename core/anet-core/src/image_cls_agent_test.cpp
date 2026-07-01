#include "catch.hpp"

#include "anet/image_cls_agent.hpp"
#include "nn_impl.hpp"

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
