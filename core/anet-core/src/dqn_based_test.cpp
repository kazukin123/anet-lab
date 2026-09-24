#include "anet/catch_test.hpp"

#include "dqn_based_heads.hpp"
#include "dqn_based_agent.hpp"
#include "nn_heads.hpp"

#include <cmath>
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

TEST_CASE("TauGenerator stratified mode covers every stratum reproducibly", "[dqn][iqn][tau]")
{
    anet::RandomGenerator first_rng(321);
    anet::RandomGenerator second_rng(321);

    // 公開生成経路から各列が対応stratum内に入り、同seedで再現されることを確認する。
    const auto first = dqn::GenerateTaus(
        3, 8, "stratified", 0.2f, 0.8f, torch::Device(torch::kCPU), first_rng);
    const auto second = dqn::GenerateTaus(
        3, 8, "stratified", 0.2f, 0.8f, torch::Device(torch::kCPU), second_rng);
    const auto normalized = (first - 0.2f) / 0.6f;
    const auto stratum_begin = torch::arange(8, first.options()) / 8.0f;
    const auto stratum_end = (torch::arange(8, first.options()) + 1.0f) / 8.0f;

    CHECK(first.sizes() == torch::IntArrayRef({ 3, 8 }));
    CHECK(first.scalar_type() == torch::kFloat32);
    CHECK(first.device().is_cpu());
    CHECK(torch::all(normalized >= stratum_begin).item<bool>());
    CHECK(torch::all(normalized <= stratum_end).item<bool>());
    CHECK(torch::all(first.slice(1, 1) > first.slice(1, 0, -1)).item<bool>());
    CHECK(torch::equal(first, second));
}

TEST_CASE("TauGenerator systematic mode uses an independent phase per row", "[dqn][iqn][tau]")
{
    anet::RandomGenerator first_rng(987);
    anet::RandomGenerator second_rng(987);

    // 行ごとに1つの位相を持ちながら、行内の点間隔が一定になることを確認する。
    const auto first = dqn::GenerateTaus(
        4, 5, "systematic", 0.2f, 0.8f, torch::Device(torch::kCPU), first_rng);
    const auto second = dqn::GenerateTaus(
        4, 5, "systematic", 0.2f, 0.8f, torch::Device(torch::kCPU), second_rng);
    const auto gaps = first.slice(1, 1) - first.slice(1, 0, -1);

    CHECK(first.sizes() == torch::IntArrayRef({ 4, 5 }));
    CHECK(torch::allclose(gaps, torch::full_like(gaps, 0.12f)));
    CHECK(torch::all(first.slice(1, 1) > first.slice(1, 0, -1)).item<bool>());
    CHECK_FALSE(torch::equal(first[0], first[1]));
    CHECK(torch::equal(first, second));
}

TEST_CASE("TauGenerator systematic mode maps per-environment ranges and consumes one draw per row", "[dqn][iqn][tau]")
{
    const auto lower = torch::tensor({ 0.0f, 0.5f, 1.0f });
    anet::RandomGenerator sampled_after_systematic(988);
    anet::RandomGenerator sampled_after_direct_draw(988);

    // per-env写像と退化範囲を確認し、同形の直接乱数生成後と後続系列が一致することを確認する。
    const auto taus = dqn::GenerateTaus(
        lower, 1.0f, 4, "systematic", sampled_after_systematic);
    auto direct_gen = sampled_after_direct_draw.GetTorchGenerator(torch::Device(torch::kCPU));
    static_cast<void>(torch::rand({ 3, 1 }, direct_gen, lower.options()));
    const auto after_systematic = dqn::GenerateTaus(
        2, 3, "random", 0.0f, 1.0f, torch::Device(torch::kCPU), sampled_after_systematic);
    const auto after_direct_draw = dqn::GenerateTaus(
        2, 3, "random", 0.0f, 1.0f, torch::Device(torch::kCPU), sampled_after_direct_draw);
    const auto gaps = taus.slice(0, 0, 2).slice(1, 1) - taus.slice(0, 0, 2).slice(1, 0, -1);
    const auto expected_gaps = torch::tensor({ { 0.25f, 0.25f, 0.25f }, { 0.125f, 0.125f, 0.125f } });

    CHECK(torch::allclose(gaps, expected_gaps));
    CHECK(torch::equal(taus[2], torch::ones({ 4 }, taus.options())));
    CHECK(torch::equal(after_systematic, after_direct_draw));
}

TEST_CASE("TauGenerator antithetic mode lays out mirrored pairs and an odd tail sample", "[dqn][iqn][tau]")
{
    anet::RandomGenerator even_rng(741);
    anet::RandomGenerator odd_rng(742);
    anet::RandomGenerator single_rng(743);

    // 前半と後半を同じindexの鏡映ペアにし、奇数Kの末尾だけを独立サンプルとして残す。
    const auto even = dqn::GenerateTaus(
        3, 4, "antithetic", 0.2f, 0.8f, torch::Device(torch::kCPU), even_rng);
    const auto odd = dqn::GenerateTaus(
        3, 5, "antithetic", 0.2f, 0.8f, torch::Device(torch::kCPU), odd_rng);
    const auto single = dqn::GenerateTaus(
        3, 1, "antithetic", 0.2f, 0.8f, torch::Device(torch::kCPU), single_rng);

    CHECK(even.sizes() == torch::IntArrayRef({ 3, 4 }));
    CHECK(torch::allclose(even.slice(1, 0, 2) + even.slice(1, 2, 4), torch::ones({ 3, 2 })));
    CHECK(torch::allclose(odd.slice(1, 0, 2) + odd.slice(1, 2, 4), torch::ones({ 3, 2 })));
    CHECK(torch::all(even >= 0.2f).item<bool>());
    CHECK(torch::all(even <= 0.8f).item<bool>());
    CHECK(torch::all(odd.slice(1, 4) >= 0.2f).item<bool>());
    CHECK(torch::all(odd.slice(1, 4) <= 0.8f).item<bool>());
    CHECK(single.sizes() == torch::IntArrayRef({ 3, 1 }));
    CHECK(torch::all(single >= 0.2f).item<bool>());
    CHECK(torch::all(single <= 0.8f).item<bool>());
}

TEST_CASE("TauGenerator antithetic mode mirrors per-environment ranges and consumes ceil half draws", "[dqn][iqn][tau]")
{
    const auto lower = torch::tensor({ 0.0f, 0.5f, 1.0f });
    anet::RandomGenerator sampled_after_antithetic(744);
    anet::RandomGenerator sampled_after_direct_draw(744);

    // 各行の範囲中点で同じindexを鏡映し、奇数Kでも乱数消費をceil(K/2)へ固定する。
    const auto taus = dqn::GenerateTaus(
        lower, 1.0f, 5, "antithetic", sampled_after_antithetic);
    auto direct_gen = sampled_after_direct_draw.GetTorchGenerator(torch::Device(torch::kCPU));
    static_cast<void>(torch::rand({ 3, 3 }, direct_gen, lower.options()));
    const auto after_antithetic = dqn::GenerateTaus(
        2, 3, "random", 0.0f, 1.0f, torch::Device(torch::kCPU), sampled_after_antithetic);
    const auto after_direct_draw = dqn::GenerateTaus(
        2, 3, "random", 0.0f, 1.0f, torch::Device(torch::kCPU), sampled_after_direct_draw);
    const auto pair_sum = taus.slice(1, 0, 2) + taus.slice(1, 2, 4);
    const auto expected_sum = (lower + 1.0f).unsqueeze(1).expand({ 3, 2 });

    CHECK(torch::allclose(pair_sum, expected_sum));
    CHECK(torch::equal(taus[2], torch::ones({ 5 }, taus.options())));
    CHECK(torch::equal(after_antithetic, after_direct_draw));
}

TEST_CASE("TauGenerator randomized placement modes follow their seeds", "[dqn][iqn][tau]")
{
    for (const auto& mode : { "stratified", "systematic", "antithetic" }) {
        INFO("mode=" << mode);
        anet::RandomGenerator first_rng(745);
        anet::RandomGenerator second_rng(745);
        anet::RandomGenerator different_rng(746);

        // 同一seedと呼出し順では再現し、異なるseedでは確率的位置が変わることを確認する。
        const auto first = dqn::GenerateTaus(
            4, 5, mode, 0.0f, 1.0f, torch::Device(torch::kCPU), first_rng);
        const auto second = dqn::GenerateTaus(
            4, 5, mode, 0.0f, 1.0f, torch::Device(torch::kCPU), second_rng);
        const auto different = dqn::GenerateTaus(
            4, 5, mode, 0.0f, 1.0f, torch::Device(torch::kCPU), different_rng);

        CHECK(torch::equal(first, second));
        CHECK_FALSE(torch::equal(first, different));
    }
}

TEST_CASE("TauGenerator ordered modes allow float32 ties in narrow ranges", "[dqn][iqn][tau]")
{
    for (const auto& mode : { "stratified", "systematic" }) {
        INFO("mode=" << mode);
        anet::RandomGenerator rng(747);

        // float32で隣接値を区別できない正幅でも補正せず、列順を非減少に保つ。
        const auto taus = dqn::GenerateTaus(
            2, 8, mode, 1.0f, std::nextafter(1.0f, 2.0f), torch::Device(torch::kCPU), rng);

        CHECK(torch::all(taus.slice(1, 1) >= taus.slice(1, 0, -1)).item<bool>());
        CHECK(torch::all(taus >= 1.0f).item<bool>());
        CHECK(torch::all(taus <= std::nextafter(1.0f, 2.0f)).item<bool>());
    }
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

TEST_CASE("TauGenerator stratified mode maps per-environment ranges", "[dqn][iqn][tau]")
{
    const auto lower = torch::tensor({ 0.0f, 0.5f, 1.0f });
    anet::RandomGenerator rng(654);
    anet::RandomGenerator different_rng(655);

    // 行ごとの範囲へ同じstratum規則を写像し、幅0の行は全点を下限へ保つ。
    const auto taus = dqn::GenerateTaus(lower, 1.0f, 4, "stratified", rng);
    const auto different = dqn::GenerateTaus(lower, 1.0f, 4, "stratified", different_rng);
    const auto normalized = (taus.slice(0, 0, 2) - lower.slice(0, 0, 2).unsqueeze(1))
        / (1.0f - lower.slice(0, 0, 2).unsqueeze(1));
    const auto stratum_begin = torch::arange(4, taus.options()) / 4.0f;
    const auto stratum_end = (torch::arange(4, taus.options()) + 1.0f) / 4.0f;

    CHECK(taus.sizes() == torch::IntArrayRef({ 3, 4 }));
    CHECK(torch::all(normalized >= stratum_begin).item<bool>());
    CHECK(torch::all(normalized <= stratum_end).item<bool>());
    CHECK(torch::all(taus.slice(0, 0, 2).slice(1, 1)
        > taus.slice(0, 0, 2).slice(1, 0, -1)).item<bool>());
    CHECK(torch::equal(taus[2], torch::ones({ 4 }, taus.options())));
    CHECK_FALSE(torch::equal(taus.slice(0, 0, 2), different.slice(0, 0, 2)));
}

TEST_CASE("TauGenerator stratified mode consumes one draw per row and stratum", "[dqn][iqn][tau]")
{
    anet::RandomGenerator sampled_after_stratified(656);
    anet::RandomGenerator sampled_after_direct_draw(656);

    // `(B,K)`の一括生成と同じ乱数列だけを消費することを後続randomから確認する。
    static_cast<void>(dqn::GenerateTaus(
        3, 4, "stratified", 0.0f, 1.0f, torch::Device(torch::kCPU), sampled_after_stratified));
    auto direct_gen = sampled_after_direct_draw.GetTorchGenerator(torch::Device(torch::kCPU));
    static_cast<void>(torch::rand(
        { 3, 4 }, direct_gen, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)));
    const auto after_stratified = dqn::GenerateTaus(
        2, 3, "random", 0.0f, 1.0f, torch::Device(torch::kCPU), sampled_after_stratified);
    const auto after_direct_draw = dqn::GenerateTaus(
        2, 3, "random", 0.0f, 1.0f, torch::Device(torch::kCPU), sampled_after_direct_draw);

    CHECK(torch::equal(after_stratified, after_direct_draw));
}

TEST_CASE("TauGenerator rejects unknown placement modes", "[dqn][iqn][tau][validation]")
{
    anet::RandomGenerator rng(1);
    CHECK_THROWS_WITH(
        dqn::GenerateTaus(1, 2, "invalid", 0.0f, 1.0f, torch::Device(torch::kCPU), rng),
        Catch::Matchers::ContainsSubstring("sample_mode=invalid")
            && Catch::Matchers::ContainsSubstring("random, fixed, stratified, systematic, antithetic"));

    const auto lower = torch::tensor({ 0.0f });
    CHECK_THROWS_WITH(
        dqn::GenerateTaus(lower, 1.0f, 2, "invalid", rng),
        Catch::Matchers::ContainsSubstring("sample_mode=invalid")
            && Catch::Matchers::ContainsSubstring("random, fixed, stratified, systematic, antithetic"));
}

TEST_CASE("TauGenerator creates taus on CUDA when available", "[dqn][iqn][tau][cuda]")
{
    if (!torch::cuda::is_available()) {
        return;
    }

    const torch::Device device(torch::kCUDA, 0);
    anet::RandomGenerator rng(789);
    const auto lower = torch::tensor(
        { 0.2f, 0.6f }, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    for (const auto& mode : { "random", "fixed", "stratified", "systematic", "antithetic" }) {
        INFO("mode=" << mode);
        // 両overloadが全modeでCUDA上のfloat32 `(B,K)` を維持することを確認する。
        const auto common = dqn::GenerateTaus(2, 3, mode, 0.0f, 1.0f, device, rng);
        const auto per_env = dqn::GenerateTaus(lower, 1.0f, 3, mode, rng);

        CHECK(common.sizes() == torch::IntArrayRef({ 2, 3 }));
        CHECK(per_env.sizes() == torch::IntArrayRef({ 2, 3 }));
        CHECK(common.scalar_type() == torch::kFloat32);
        CHECK(per_env.scalar_type() == torch::kFloat32);
        CHECK(common.device() == device);
        CHECK(per_env.device() == device);
    }
}
