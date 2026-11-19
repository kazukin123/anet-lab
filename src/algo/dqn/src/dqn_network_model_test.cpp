#include "anet/test/catch.hpp"
#include "test/dqn_fixture_test.hpp"

TEST_CASE("NetworkModel mode-specific forwards preserve training modes", "[dqn][network_model]")
{
    TestNetworkModel model;
    const auto online_forward = MakeDefaultExecutor(model.GetOnlineNetwork());
    const anet::nn::NetworkForwardCallInfo info({});
    auto obs = anet::TensorDict{ { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }) } };

    REQUIRE_FALSE(model.GetOnlineNetwork()->is_training());
    REQUIRE_FALSE(model.GetTargetNetwork()->is_training());

    model.ForwardOnlineWithTrain(obs, *online_forward, info);
    CHECK_FALSE(model.GetOnlineNetwork()->is_training());
    CHECK_FALSE(model.GetTargetNetwork()->is_training());

    model.GetOnlineNetwork()->train();
    model.ForwardOnlineWithTrain(obs, *online_forward, info);
    CHECK(model.GetOnlineNetwork()->is_training());
    CHECK_FALSE(model.GetTargetNetwork()->is_training());
}

TEST_CASE("NetworkModel routes TensorDictFunction by network side and function key", "[dqn][network_model]")
{
    TestNetworkModel model;
    {
        torch::NoGradGuard no_grad;
        model.GetOnlineNetwork()->parameters()[0].fill_(1.0f);
        model.GetTargetNetwork()->parameters()[0].fill_(2.0f);
    }
    auto obs = anet::TensorDict{ { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }) } };

    auto policy_forward = model.GetTensorDictFunction("policy-net.forward", torch::kCPU);
    auto policy_forward_q = model.GetTensorDictFunction("policy-net.forward.q", torch::kCPU);
    auto target_forward = model.GetTensorDictFunction("target-net.forward", torch::kCPU);
    REQUIRE(policy_forward.has_value());
    REQUIRE(policy_forward_q.has_value());
    REQUIRE(target_forward.has_value());

    CHECK(TensorToFloatVector((*policy_forward)(obs).At("q")) == std::vector<float>{ 3.0f });
    CHECK(TensorToFloatVector((*policy_forward_q)(obs).At("q")) == std::vector<float>{ 3.0f });
    CHECK(TensorToFloatVector((*target_forward)(obs).At("q")) == std::vector<float>{ 6.0f });

    CHECK_FALSE(model.GetTensorDictFunction("policy-net.forward.dist", torch::kCPU).has_value());
    CHECK_FALSE(model.GetTensorDictFunction("unknown-net.forward", torch::kCPU).has_value());
}

TEST_CASE("NetworkModel exposes the resolved distributional mode", "[dqn][network_model]")
{
    TestNetworkModel scalar_model(false);
    TestNetworkModel distributional_model(true);

    CHECK_FALSE(scalar_model.IsDistributional());
    CHECK(distributional_model.IsDistributional());
}

TEST_CASE("NetworkModel requires distinct online and target Networks", "[dqn][noisynet][network_model]")
{
    class ExplicitNetworks final : public dqn::NetworkModel {
    public:
        explicit ExplicitNetworks(const std::shared_ptr<anet::nn::Network>& network)
            : NetworkModel(dqn::NetworkModelConfig{}, network, network, 1, false) {}
    };
    const auto network = MakeLinearNetwork(1);
    CHECK_THROWS_WITH(ExplicitNetworks(network), Catch::Matchers::ContainsSubstring("distinct online and target Networks"));
}
