#include "anet/catch_test.hpp"

#include "dqn_based_heads.hpp"
#include "dqn_based_agent.hpp"
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

bool HasOutput(const anet::nn::HeadGraphVizInfo& info, const std::string& name, const std::vector<int64_t>& shape)
{
    for (const auto& output : info.outputs) {
        if (output.name == name && output.shape == shape) {
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

TEST_CASE("IQN head returns action values from dynamic tau samples", "[dqn][head][iqn]")
{
    anet::nn::WeightInitConfig init_config;
    dqn::IQNHeadFactory factory(3, init_config);

    anet::TensorDict dummy_features;
    dummy_features.Set(anet::nn::kKey_DefaultOutput, torch::zeros({ 1, 2, 4 }));
    auto head = factory.CreateHead(dummy_features);

    anet::TensorDict features;
    features.Set(anet::nn::kKey_DefaultOutput, torch::randn({ 2, 5, 4 }));
    const auto output = head->Forward(features);

    const auto q_dist = output.At("q_dist");
    CHECK(q_dist.sizes() == torch::IntArrayRef({ 2, 3, 5 }));
    CHECK(output.At("q").sizes() == torch::IntArrayRef({ 2, 3 }));
    CHECK(torch::allclose(output.At("q"), q_dist.mean(2)));
    CHECK(q_dist.is_contiguous());
}

TEST_CASE("IQN dueling head combines shared and branched tau features", "[dqn][head][iqn][dueling]")
{
    anet::nn::WeightInitConfig init_config;
    dqn::IQNDuelingHeadFactory factory(3, init_config);

    anet::TensorDict shared_dummy;
    shared_dummy.Set(anet::nn::kKey_DefaultOutput, torch::zeros({ 1, 2, 4 }));
    auto shared_head = factory.CreateHead(shared_dummy);
    anet::TensorDict shared_features;
    shared_features.Set(anet::nn::kKey_DefaultOutput, torch::randn({ 2, 5, 4 }));
    const auto shared_output = shared_head->Forward(shared_features);
    CHECK(shared_output.At("q_dist").sizes() == torch::IntArrayRef({ 2, 3, 5 }));

    anet::TensorDict branched_dummy;
    branched_dummy.Set("value_feature", torch::zeros({ 1, 2, 5 }));
    branched_dummy.Set("adv_feature", torch::zeros({ 1, 2, 7 }));
    auto branched_head = factory.CreateHead(branched_dummy);
    anet::TensorDict branched_features;
    branched_features.Set("value_feature", torch::randn({ 2, 5, 5 }));
    branched_features.Set("adv_feature", torch::randn({ 2, 5, 7 }));
    const auto output = branched_head->Forward(branched_features);

    const auto q_dist = output.At("q_dist");
    const auto v_dist = output.At("v_dist");
    const auto a_dist = output.At("a_dist");
    CHECK(q_dist.sizes() == torch::IntArrayRef({ 2, 3, 5 }));
    CHECK(v_dist.sizes() == torch::IntArrayRef({ 2, 1, 5 }));
    CHECK(a_dist.sizes() == torch::IntArrayRef({ 2, 3, 5 }));
    CHECK(torch::allclose(q_dist, v_dist + (a_dist - a_dist.mean(1, true))));
    CHECK(torch::allclose(output.At("q"), q_dist.mean(2)));
    CHECK(q_dist.is_contiguous());
}

TEST_CASE("IQN heads expose quantile diagnostic functions and graph metadata", "[dqn][head][iqn][function][dot]")
{
    anet::nn::WeightInitConfig init_config;

    dqn::IQNHeadFactory plain_factory(3, init_config);
    anet::TensorDict plain_dummy;
    plain_dummy.Set(anet::nn::kKey_DefaultOutput, torch::zeros({ 1, 2, 4 }));
    auto plain_head = plain_factory.CreateHead(plain_dummy);
    anet::TensorDict plain_features;
    plain_features.Set(anet::nn::kKey_DefaultOutput, torch::zeros({ 2, 5, 4 }));

    for (const std::string key : { "forward", "forward.q", "q_values" }) {
        const auto function = plain_head->GetTensorDictFunction(key);
        REQUIRE(function.has_value());
        const auto output = (*function)(plain_features);
        CHECK(output.At("q").sizes() == torch::IntArrayRef({ 2, 3 }));
        CHECK_FALSE(output.Get("q_dist").has_value());
    }
    for (const std::string key : { "forward.dist", "distributions" }) {
        const auto function = plain_head->GetTensorDictFunction(key);
        REQUIRE(function.has_value());
        CHECK((*function)(plain_features).At("q_dist").sizes() == torch::IntArrayRef({ 2, 3, 5 }));
    }
    CHECK_FALSE(plain_head->GetTensorDictFunction("unknown").has_value());
    const auto plain_info = plain_head->GetGraphVizInfo();
    CHECK(plain_info.type == "IQNHead");
    CHECK(HasOutput(plain_info, "q", { 3 }));
    CHECK(HasOutput(plain_info, "q_dist", { 3, -1 }));
    CHECK(HasDetail(plain_info, "action_dim", "3"));

    dqn::IQNDuelingHeadFactory dueling_factory(3, init_config);
    anet::TensorDict dueling_dummy;
    dueling_dummy.Set("value_feature", torch::zeros({ 1, 2, 5 }));
    dueling_dummy.Set("adv_feature", torch::zeros({ 1, 2, 7 }));
    auto dueling_head = dueling_factory.CreateHead(dueling_dummy);
    anet::TensorDict dueling_features;
    dueling_features.Set("value_feature", torch::zeros({ 2, 5, 5 }));
    dueling_features.Set("adv_feature", torch::zeros({ 2, 5, 7 }));

    for (const std::string key : { "forward", "forward.q", "q_values" }) {
        const auto function = dueling_head->GetTensorDictFunction(key);
        REQUIRE(function.has_value());
        const auto output = (*function)(dueling_features);
        CHECK(output.At("q").sizes() == torch::IntArrayRef({ 2, 3 }));
        CHECK_FALSE(output.Get("q_dist").has_value());
    }
    for (const std::string key : { "forward.dist", "distributions" }) {
        const auto function = dueling_head->GetTensorDictFunction(key);
        REQUIRE(function.has_value());
        CHECK((*function)(dueling_features).At("q_dist").sizes() == torch::IntArrayRef({ 2, 3, 5 }));
    }
    for (const std::string key : { "forward.v", "v_values" }) {
        const auto function = dueling_head->GetTensorDictFunction(key);
        REQUIRE(function.has_value());
        CHECK((*function)(dueling_features).At("v_dist").sizes() == torch::IntArrayRef({ 2, 1, 5 }));
    }
    for (const std::string key : { "forward.a", "a_values" }) {
        const auto function = dueling_head->GetTensorDictFunction(key);
        REQUIRE(function.has_value());
        CHECK((*function)(dueling_features).At("a_dist").sizes() == torch::IntArrayRef({ 2, 3, 5 }));
    }
    const auto dueling_info = dueling_head->GetGraphVizInfo();
    CHECK(dueling_info.type == "IQNDuelingHead");
    CHECK(HasOutput(dueling_info, "q", { 3 }));
    CHECK(HasOutput(dueling_info, "q_dist", { 3, -1 }));
    CHECK(HasOutput(dueling_info, "v_dist", { 1, -1 }));
    CHECK(HasOutput(dueling_info, "a_dist", { 3, -1 }));
    CHECK(HasDetail(dueling_info, "mode", "branched"));
    CHECK(HasDetail(dueling_info, "value_input_key", "value_feature"));
    CHECK(HasDetail(dueling_info, "adv_input_key", "adv_feature"));
}

TEST_CASE("IQN head factories reject incompatible feature shapes", "[dqn][head][iqn][validation]")
{
    anet::nn::WeightInitConfig init_config;

    SECTION("plain head requires rank three") {
        dqn::IQNHeadFactory factory(3, init_config);
        anet::TensorDict features;
        features.Set(anet::nn::kKey_DefaultOutput, torch::zeros({ 2, 4 }));
        CHECK_THROWS_WITH(
            factory.CreateHead(features),
            Catch::Matchers::ContainsSubstring("IQNHead")
                && Catch::Matchers::ContainsSubstring("rank-3")
                && Catch::Matchers::ContainsSubstring("fusion"));
    }

    SECTION("plain head validates each runtime input") {
        dqn::IQNHeadFactory factory(3, init_config);
        anet::TensorDict dummy_features;
        dummy_features.Set(anet::nn::kKey_DefaultOutput, torch::zeros({ 1, 2, 4 }));
        auto head = factory.CreateHead(dummy_features);
        anet::TensorDict features;
        features.Set(anet::nn::kKey_DefaultOutput, torch::zeros({ 2, 4 }));
        CHECK_THROWS_WITH(
            head->Forward(features),
            Catch::Matchers::ContainsSubstring("IQNHead")
                && Catch::Matchers::ContainsSubstring("rank-3"));
    }

    SECTION("dueling head requires both streams") {
        dqn::IQNDuelingHeadFactory factory(3, init_config);
        anet::TensorDict features;
        features.Set("value_feature", torch::zeros({ 1, 2, 5 }));
        CHECK_THROWS_WITH(
            factory.CreateHead(features),
            Catch::Matchers::ContainsSubstring("both 'value_feature' and 'adv_feature'"));
    }

    SECTION("dueling streams require rank three") {
        dqn::IQNDuelingHeadFactory factory(3, init_config);
        anet::TensorDict features;
        features.Set("value_feature", torch::zeros({ 1, 2, 5 }));
        features.Set("adv_feature", torch::zeros({ 1, 7 }));
        CHECK_THROWS_WITH(
            factory.CreateHead(features),
            Catch::Matchers::ContainsSubstring("IQNDuelingHead")
                && Catch::Matchers::ContainsSubstring("rank-3")
                && Catch::Matchers::ContainsSubstring("fusion"));
    }

    SECTION("dueling streams require matching batch dimensions") {
        dqn::IQNDuelingHeadFactory factory(3, init_config);
        anet::TensorDict features;
        features.Set("value_feature", torch::zeros({ 2, 4, 5 }));
        features.Set("adv_feature", torch::zeros({ 3, 4, 7 }));
        CHECK_THROWS_WITH(
            factory.CreateHead(features),
            Catch::Matchers::ContainsSubstring("matching B and K")
                && Catch::Matchers::ContainsSubstring("value_shape")
                && Catch::Matchers::ContainsSubstring("advantage_shape"));
    }

    SECTION("dueling streams require matching tau dimensions") {
        dqn::IQNDuelingHeadFactory factory(3, init_config);
        anet::TensorDict features;
        features.Set("value_feature", torch::zeros({ 2, 4, 5 }));
        features.Set("adv_feature", torch::zeros({ 2, 5, 7 }));
        CHECK_THROWS_WITH(
            factory.CreateHead(features),
            Catch::Matchers::ContainsSubstring("matching B and K")
                && Catch::Matchers::ContainsSubstring("value_shape")
                && Catch::Matchers::ContainsSubstring("advantage_shape"));
    }

    SECTION("dueling head validates runtime batch and tau dimensions") {
        dqn::IQNDuelingHeadFactory factory(3, init_config);
        anet::TensorDict dummy_features;
        dummy_features.Set("value_feature", torch::zeros({ 1, 2, 5 }));
        dummy_features.Set("adv_feature", torch::zeros({ 1, 2, 7 }));
        auto head = factory.CreateHead(dummy_features);
        anet::TensorDict features;
        features.Set("value_feature", torch::zeros({ 2, 4, 5 }));
        features.Set("adv_feature", torch::zeros({ 2, 5, 7 }));
        CHECK_THROWS_WITH(
            head->Forward(features),
            Catch::Matchers::ContainsSubstring("matching B and K"));
    }
}

TEST_CASE("TauGenerator samples reproducible taus within a common range", "[dqn][iqn][tau]")
{
    anet::RandomGenerator first_rng(123);
    anet::RandomGenerator second_rng(123);

    const auto first = dqn::GenerateTaus(
        3, 7, "random", 0.2f, 0.8f, torch::Device(torch::kCPU), first_rng);
    const auto second = dqn::GenerateTaus(
        3, 7, "random", 0.2f, 0.8f, torch::Device(torch::kCPU), second_rng);

    CHECK(first.sizes() == torch::IntArrayRef({ 3, 7 }));
    CHECK(first.scalar_type() == torch::kFloat32);
    CHECK(first.device().is_cpu());
    CHECK(torch::all(first >= 0.2).item<bool>());
    CHECK(torch::all(first < 0.8).item<bool>());
    CHECK(torch::equal(first, second));
    CHECK(torch::any(
        first.slice(1, 1, first.size(1)) < first.slice(1, 0, first.size(1) - 1)).item<bool>());
}

TEST_CASE("TauGenerator fixed mode places midpoint grids without consuming RNG", "[dqn][iqn][tau]")
{
    anet::RandomGenerator sampled_after_fixed(456);
    anet::RandomGenerator sampled_directly(456);

    const auto fixed = dqn::GenerateTaus(
        2, 4, "fixed", 0.2f, 1.0f, torch::Device(torch::kCPU), sampled_after_fixed);
    const auto expected_row = torch::tensor({ 0.3f, 0.5f, 0.7f, 0.9f });
    CHECK(torch::allclose(fixed, expected_row.unsqueeze(0).expand({ 2, 4 })));

    const auto lower = torch::tensor({ 0.0f, 0.5f });
    const auto per_env_fixed = dqn::GenerateTaus(
        lower, 1.0f, 2, "fixed", sampled_after_fixed);
    const auto per_env_expected = torch::tensor({ { 0.25f, 0.75f }, { 0.625f, 0.875f } });
    CHECK(torch::allclose(per_env_fixed, per_env_expected));

    const auto after_fixed = dqn::GenerateTaus(
        2, 4, "random", 0.0f, 1.0f, torch::Device(torch::kCPU), sampled_after_fixed);
    const auto direct = dqn::GenerateTaus(
        2, 4, "random", 0.0f, 1.0f, torch::Device(torch::kCPU), sampled_directly);
    CHECK(torch::equal(after_fixed, direct));
}

TEST_CASE("TauGenerator supports per-environment lower bounds", "[dqn][iqn][tau]")
{
    const auto lower = torch::tensor({ 0.0f, 0.5f });
    anet::RandomGenerator fixed_rng(10);
    const auto fixed = dqn::GenerateTaus(lower, 1.0f, 2, "fixed", fixed_rng);
    const auto expected = torch::tensor({ { 0.25f, 0.75f }, { 0.625f, 0.875f } });
    CHECK(torch::allclose(fixed, expected));

    anet::RandomGenerator random_rng(10);
    anet::RandomGenerator second_random_rng(10);
    const auto random = dqn::GenerateTaus(lower, 1.0f, 9, "random", random_rng);
    const auto second_random = dqn::GenerateTaus(lower, 1.0f, 9, "random", second_random_rng);
    CHECK(random.sizes() == torch::IntArrayRef({ 2, 9 }));
    CHECK(torch::all(random[0] >= 0.0).item<bool>());
    CHECK(torch::all(random[0] < 1.0).item<bool>());
    CHECK(torch::all(random[1] >= 0.5).item<bool>());
    CHECK(torch::all(random[1] < 1.0).item<bool>());
    CHECK(torch::equal(random, second_random));
}

TEST_CASE("TauGenerator rejects unknown placement modes", "[dqn][iqn][tau][validation]")
{
    anet::RandomGenerator rng(1);
    CHECK_THROWS_WITH(
        dqn::GenerateTaus(1, 2, "invalid", 0.0f, 1.0f, torch::Device(torch::kCPU), rng),
        Catch::Matchers::ContainsSubstring("sample_mode=invalid")
            && Catch::Matchers::ContainsSubstring("random or fixed"));

    const auto lower = torch::tensor({ 0.0f });
    CHECK_THROWS_WITH(
        dqn::GenerateTaus(lower, 1.0f, 2, "invalid", rng),
        Catch::Matchers::ContainsSubstring("sample_mode=invalid")
            && Catch::Matchers::ContainsSubstring("random or fixed"));
}

TEST_CASE("TauGenerator creates taus on CUDA when available", "[dqn][iqn][tau][cuda]")
{
    if (!torch::cuda::is_available()) {
        return;
    }

    const torch::Device device(torch::kCUDA, 0);
    anet::RandomGenerator rng(789);
    const auto taus = dqn::GenerateTaus(2, 3, "random", 0.0f, 1.0f, device, rng);
    CHECK(taus.sizes() == torch::IntArrayRef({ 2, 3 }));
    CHECK(taus.device() == device);

    const auto lower = torch::tensor(
        { 0.2f, 0.6f }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    const auto per_env_random = dqn::GenerateTaus(lower, 1.0f, 3, "random", rng);
    const auto per_env_fixed = dqn::GenerateTaus(lower, 1.0f, 3, "fixed", rng);
    CHECK(per_env_random.sizes() == torch::IntArrayRef({ 2, 3 }));
    CHECK(per_env_fixed.sizes() == torch::IntArrayRef({ 2, 3 }));
    CHECK(per_env_random.device() == device);
    CHECK(per_env_fixed.device() == device);
}
