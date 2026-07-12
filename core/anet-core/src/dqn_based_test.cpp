#include "anet/catch_test.hpp"

#include "dqn_based_heads.hpp"
#include "nn_heads.hpp"

#include <memory>
#include <string>
#include <vector>

namespace {

namespace dqn = anet::rl::dqn;

std::shared_ptr<anet::nn::NetworkHead> CreateSharedDqnHead(const anet::nn::NetworkHeadFactory& factory)
{
    anet::TensorDict dummy_features;
    dummy_features.Set(anet::nn::kKey_DefaultOutput, torch::zeros({ 1, 4 }));
    return factory.CreateHead(dummy_features);
}

std::shared_ptr<anet::nn::NetworkHead> CreateBranchedDqnHead(const anet::nn::NetworkHeadFactory& factory)
{
    anet::TensorDict dummy_features;
    dummy_features.Set("value_feature", torch::zeros({ 1, 5 }));
    dummy_features.Set("adv_feature", torch::zeros({ 1, 7 }));
    return factory.CreateHead(dummy_features);
}

bool HasDetail(const anet::nn::HeadGraphVizInfo& info, const std::string& key, const std::string& value)
{
    for (const auto& detail : info.details) {
        if (detail.first == key && detail.second == value) {
            return true;
        }
    }
    return false;
}

} // namespace

TEST_CASE("Dueling heads use shared or branched body outputs", "[dqn][head][dueling]")
{
    anet::nn::WeightInitConfig init_config;

    dqn::DuelingHeadFactory dueling_factory(3, init_config);
    auto shared_dueling_head = CreateSharedDqnHead(dueling_factory);
    anet::TensorDict shared_features;
    shared_features.Set(anet::nn::kKey_DefaultOutput, torch::zeros({ 2, 4 }));

    auto shared_out = shared_dueling_head->Forward(shared_features);
    CHECK(shared_out.At("q").sizes() == torch::IntArrayRef({ 2, 3 }));
    CHECK(shared_out.At("v").sizes() == torch::IntArrayRef({ 2, 1 }));
    CHECK(shared_out.At("a").sizes() == torch::IntArrayRef({ 2, 3 }));

    auto branched_dueling_head = CreateBranchedDqnHead(dueling_factory);
    anet::TensorDict branched_features;
    branched_features.Set("value_feature", torch::zeros({ 2, 5 }));
    branched_features.Set("adv_feature", torch::zeros({ 2, 7 }));

    auto branched_out = branched_dueling_head->Forward(branched_features);
    CHECK(branched_out.At("q").sizes() == torch::IntArrayRef({ 2, 3 }));
    CHECK(branched_out.At("v").sizes() == torch::IntArrayRef({ 2, 1 }));
    CHECK(branched_out.At("a").sizes() == torch::IntArrayRef({ 2, 3 }));

    auto forward_v = branched_dueling_head->GetTensorDictFunction("forward.v");
    auto forward_a = branched_dueling_head->GetTensorDictFunction("forward.a");
    REQUIRE(forward_v.has_value());
    REQUIRE(forward_a.has_value());
    CHECK((*forward_v)(branched_features).At("v").sizes() == torch::IntArrayRef({ 2, 1 }));
    CHECK((*forward_a)(branched_features).At("a").sizes() == torch::IntArrayRef({ 2, 3 }));

    anet::TensorDict partial_features;
    partial_features.Set("value_feature", torch::zeros({ 1, 5 }));
    CHECK_THROWS(dueling_factory.CreateHead(partial_features));
}

TEST_CASE("Quantile dueling heads use shared or branched body outputs", "[dqn][head][dueling][qr]")
{
    anet::nn::WeightInitConfig init_config;

    dqn::QuantileDuelingHeadFactory quantile_dueling_factory(3, 5, init_config);
    auto shared_head = CreateSharedDqnHead(quantile_dueling_factory);
    anet::TensorDict shared_features;
    shared_features.Set(anet::nn::kKey_DefaultOutput, torch::zeros({ 2, 4 }));

    auto shared_out = shared_head->Forward(shared_features);
    CHECK(shared_out.At("q").sizes() == torch::IntArrayRef({ 2, 3 }));
    CHECK(shared_out.At("q_dist").sizes() == torch::IntArrayRef({ 2, 3, 5 }));
    CHECK(shared_out.At("v_dist").sizes() == torch::IntArrayRef({ 2, 1, 5 }));
    CHECK(shared_out.At("a_dist").sizes() == torch::IntArrayRef({ 2, 3, 5 }));

    auto branched_head = CreateBranchedDqnHead(quantile_dueling_factory);
    anet::TensorDict branched_features;
    branched_features.Set("value_feature", torch::zeros({ 2, 5 }));
    branched_features.Set("adv_feature", torch::zeros({ 2, 7 }));

    auto branched_out = branched_head->Forward(branched_features);
    CHECK(branched_out.At("q").sizes() == torch::IntArrayRef({ 2, 3 }));
    CHECK(branched_out.At("q_dist").sizes() == torch::IntArrayRef({ 2, 3, 5 }));
    CHECK(branched_out.At("v_dist").sizes() == torch::IntArrayRef({ 2, 1, 5 }));
    CHECK(branched_out.At("a_dist").sizes() == torch::IntArrayRef({ 2, 3, 5 }));

    auto forward_q = branched_head->GetTensorDictFunction("forward");
    auto forward_dist = branched_head->GetTensorDictFunction("forward.dist");
    auto forward_v = branched_head->GetTensorDictFunction("forward.v");
    auto forward_a = branched_head->GetTensorDictFunction("forward.a");
    REQUIRE(forward_q.has_value());
    REQUIRE(forward_dist.has_value());
    REQUIRE(forward_v.has_value());
    REQUIRE(forward_a.has_value());
    auto q_only = (*forward_q)(branched_features);
    CHECK(q_only.At("q").sizes() == torch::IntArrayRef({ 2, 3 }));
    CHECK_FALSE(q_only.Get("q_dist").has_value());
    CHECK((*forward_dist)(branched_features).At("q_dist").sizes() == torch::IntArrayRef({ 2, 3, 5 }));
    CHECK((*forward_v)(branched_features).At("v_dist").sizes() == torch::IntArrayRef({ 2, 1, 5 }));
    CHECK((*forward_a)(branched_features).At("a_dist").sizes() == torch::IntArrayRef({ 2, 3, 5 }));

    anet::TensorDict partial_features;
    partial_features.Set("adv_feature", torch::zeros({ 1, 7 }));
    CHECK_THROWS(quantile_dueling_factory.CreateHead(partial_features));
}

TEST_CASE("Dueling head graph info shows shared and branched input keys", "[dqn][head][dueling][dot]")
{
    anet::nn::WeightInitConfig init_config;
    dqn::DuelingHeadFactory dueling_factory(3, init_config);

    auto shared_info = CreateSharedDqnHead(dueling_factory)->GetGraphVizInfo();
    CHECK(HasDetail(shared_info, "mode", "shared"));
    CHECK(HasDetail(shared_info, "value_input_key", "features"));
    CHECK(HasDetail(shared_info, "adv_input_key", "features"));

    auto branched_info = CreateBranchedDqnHead(dueling_factory)->GetGraphVizInfo();
    CHECK(HasDetail(branched_info, "mode", "branched"));
    CHECK(HasDetail(branched_info, "value_input_key", "value_feature"));
    CHECK(HasDetail(branched_info, "adv_input_key", "adv_feature"));
}
