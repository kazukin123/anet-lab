#include "anet/catch_test.hpp"

#include "dqn_based_heads.hpp"
#include "dqn_based_agent.hpp"
#include "nn_heads.hpp"
#include "nn_test.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace {
// テスト用の生データから選択項目だけを取り出し、公開 constructor へ渡す。
anet::ConfigData SelectForwardConfigData(const anet::ConfigData& data, const std::string& prefix)
{
    anet::ConfigData selected;
    for (const auto& [key, value] : data.Map()) {
        if (key.starts_with(prefix + ".")) selected.Set(key.substr(prefix.size() + 1), value);
    }
    return selected;
}


namespace dqn = anet::rl::dqn;

std::shared_ptr<anet::nn::NetworkHeadFactory> CreateConfiguredProjectionHeadFactory(
    int kind, const anet::nn::WeightInitConfig& init, const anet::ConfigData& data)
{
    switch (kind) {
    case 0: return std::make_shared<dqn::DuelingHeadFactory>(3, init, data);
    case 1: return std::make_shared<dqn::QuantileHeadFactory>(3, 5, init, data);
    case 2: return std::make_shared<dqn::QuantileDuelingHeadFactory>(3, 5, init, data);
    case 3: return std::make_shared<dqn::IQNHeadFactory>(3, init, data);
    case 4: return std::make_shared<dqn::IQNDuelingHeadFactory>(3, init, data);
    default: return std::make_shared<dqn::LinearHeadFactory>(3, init, data);
    }
}

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

    auto shared_out = shared_dueling_head->Forward(shared_features, anet::nn::NetworkForwardContext::GetDefault());
    CHECK(shared_out.At("q").sizes() == torch::IntArrayRef({ 2, 3 }));
    CHECK(shared_out.At("v").sizes() == torch::IntArrayRef({ 2, 1 }));
    CHECK(shared_out.At("a").sizes() == torch::IntArrayRef({ 2, 3 }));

    auto branched_dueling_head = CreateBranchedDqnHead(dueling_factory);
    anet::TensorDict branched_features;
    branched_features.Set("value_feature", torch::zeros({ 2, 5 }));
    branched_features.Set("adv_feature", torch::zeros({ 2, 7 }));

    auto branched_out = branched_dueling_head->Forward(branched_features, anet::nn::NetworkForwardContext::GetDefault());
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

    auto shared_out = shared_head->Forward(shared_features, anet::nn::NetworkForwardContext::GetDefault());
    CHECK(shared_out.At("q").sizes() == torch::IntArrayRef({ 2, 3 }));
    CHECK(shared_out.At("q_dist").sizes() == torch::IntArrayRef({ 2, 3, 5 }));
    CHECK(shared_out.At("v_dist").sizes() == torch::IntArrayRef({ 2, 1, 5 }));
    CHECK(shared_out.At("a_dist").sizes() == torch::IntArrayRef({ 2, 3, 5 }));

    auto branched_head = CreateBranchedDqnHead(quantile_dueling_factory);
    anet::TensorDict branched_features;
    branched_features.Set("value_feature", torch::zeros({ 2, 5 }));
    branched_features.Set("adv_feature", torch::zeros({ 2, 7 }));

    auto branched_out = branched_head->Forward(branched_features, anet::nn::NetworkForwardContext::GetDefault());
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
    const auto output = head->Forward(features, anet::nn::NetworkForwardContext::GetDefault());

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
    const auto shared_output = shared_head->Forward(shared_features, anet::nn::NetworkForwardContext::GetDefault());
    CHECK(shared_output.At("q_dist").sizes() == torch::IntArrayRef({ 2, 3, 5 }));

    anet::TensorDict branched_dummy;
    branched_dummy.Set("value_feature", torch::zeros({ 1, 2, 5 }));
    branched_dummy.Set("adv_feature", torch::zeros({ 1, 2, 7 }));
    auto branched_head = factory.CreateHead(branched_dummy);
    anet::TensorDict branched_features;
    branched_features.Set("value_feature", torch::randn({ 2, 5, 5 }));
    branched_features.Set("adv_feature", torch::randn({ 2, 5, 7 }));
    const auto output = branched_head->Forward(branched_features, anet::nn::NetworkForwardContext::GetDefault());

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
            head->Forward(features, anet::nn::NetworkForwardContext::GetDefault()),
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
            head->Forward(features, anet::nn::NetworkForwardContext::GetDefault()),
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

TEST_CASE("DQN Linear Head registers projection sigma and preserves default output", "[dqn][head][noisynet][initialization]")
{
    anet::ConfigData projection;
    projection.Set("noisy.enabled", true);
    projection.Set("noisy.sigma0", 0.5f);
    const anet::nn::WeightInitConfig init{ .mode = "constant", .constant_val = 0.125 };
    dqn::LinearHeadFactory factory(3, init, projection);
    auto head = CreateSharedDqnHead(factory);
    const auto parameters = head->named_parameters();
    REQUIRE(parameters.contains("linear.sigma_weight"));
    REQUIRE(parameters.contains("linear.sigma_bias"));
    CHECK(torch::equal(parameters["linear.weight"], torch::full({ 3, 4 }, 0.125f)));
    CHECK(torch::equal(parameters["linear.bias"], torch::full({ 3 }, 0.125f)));
    CHECK(torch::equal(parameters["linear.sigma_weight"], torch::full({ 3, 4 }, 0.25f)));
    CHECK(torch::equal(parameters["linear.sigma_bias"], torch::full({ 3 }, 0.25f)));
    anet::TensorDict features;
    features.Set(anet::nn::kKey_DefaultOutput, torch::ones({ 2, 4 }));

    // 最終射影の μ と既定の Q を保ち、σ を既定 forward のグラフへ入れない。
    const auto output = head->Forward(features, anet::nn::NetworkForwardContext::GetDefault()).At("q");
    CHECK(torch::equal(output, torch::full({ 2, 3 }, 0.625f)));
    output.sum().backward();
    CHECK_FALSE(parameters["linear.sigma_weight"].grad().defined());
    CHECK_FALSE(parameters["linear.sigma_bias"].grad().defined());
    const auto function = head->GetTensorDictFunction("forward.q");
    REQUIRE(function.has_value());
    CHECK(torch::equal((*function)(features).At("q"), output));
}

TEST_CASE("DQN Linear Head sample matches factorized reference and sigma gradients", "[dqn][head][noisynet][arithmetic]")
{
    anet::ConfigData data;
    data.Set("noisy.enabled", true);
    data.Set("nn_forward.[sample].noisy.mode", "sample");
    data.Set("nn_forward.[sample].noisy.hold", "call");
    data.Set("nn_forward.[sample].noisy.share", "batch");
    const anet::nn::WeightInitConfig init{ .mode = "constant", .constant_val = 0.125 };
    dqn::LinearHeadFactory factory(3, init, data);
    auto head = CreateSharedDqnHead(factory);
    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{}, anet::TensorSpecMap{},
        std::vector<std::string>{}, std::map<std::string, std::string>{});
    anet::nn::Network owner(anet::nn::NetworkConfig{}, anet::TensorSpecMap{}, nullptr, body, head);
    const anet::seed_t seed = 769;
    auto state = anet::test::MakeTestForwardState(seed);
    const anet::nn::NetworkForwardConfig config("nn_forward.[sample]", SelectForwardConfigData(data, "nn_forward.[sample]"));
    anet::nn::NetworkForwardContext::Params context_params{
        .state = state.get(), .clock_kind = anet::nn::NetworkForwardClock::kTransientRows, .clock = 0,
    };
    auto owned_head_forward = head->PrepareForward(config, anet::nn::NetworkForwardClock::kTransientRows);
    auto* head_forward = owned_head_forward ? owned_head_forward.get() : head.get();
    anet::TensorDict features;
    auto input = torch::arange(8, torch::kFloat32).reshape({ 2, 4 }).requires_grad_(true);
    features.Set(anet::nn::kKey_DefaultOutput, input);
    auto parameters = head->named_parameters();
    auto ref_input = input.detach().clone().requires_grad_(true);
    auto ref_weight = parameters["linear.weight"].detach().clone().requires_grad_(true);
    auto ref_bias = parameters["linear.bias"].detach().clone().requires_grad_(true);
    auto ref_sigma_weight = parameters["linear.sigma_weight"].detach().clone().requires_grad_(true);
    auto ref_sigma_bias = parameters["linear.sigma_bias"].detach().clone().requires_grad_(true);

    // 専用 seed から作った明示的な noisy weight と、出力・全勾配を比較する。
    anet::RandomGenerator reference_random(seed);
    const auto generator = reference_random.GetTorchGenerator(input.device());
    const auto raw_input = torch::randn({ 4 }, generator, input.options());
    const auto raw_output = torch::randn({ 3 }, generator, input.options());
    const auto epsilon_input = raw_input.sign() * raw_input.abs().sqrt();
    const auto epsilon_output = raw_output.sign() * raw_output.abs().sqrt();
    const auto reference = torch::linear(ref_input,
        ref_weight + ref_sigma_weight * epsilon_output.unsqueeze(1) * epsilon_input.unsqueeze(0),
        ref_bias + ref_sigma_bias * epsilon_output);
    const auto output = head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q");
    CHECK(torch::allclose(output, reference, 1e-5, 1e-6));
    output.square().sum().backward();
    reference.square().sum().backward();
    REQUIRE(parameters["linear.sigma_weight"].grad().defined());
    REQUIRE(parameters["linear.sigma_bias"].grad().defined());
    CHECK(torch::allclose(input.grad(), ref_input.grad(), 1e-5, 1e-6));
    CHECK(torch::allclose(parameters["linear.weight"].grad(), ref_weight.grad(), 1e-5, 1e-6));
    CHECK(torch::allclose(parameters["linear.bias"].grad(), ref_bias.grad(), 1e-5, 1e-6));
    CHECK(torch::allclose(parameters["linear.sigma_weight"].grad(), ref_sigma_weight.grad(), 1e-5, 1e-6));
    CHECK(torch::allclose(parameters["linear.sigma_bias"].grad(), ref_sigma_bias.grad(), 1e-5, 1e-6));
    CHECK(torch::equal(head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q"), output));
    context_params.clock = 1;
    CHECK_FALSE(torch::equal(head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q"), output));
}

TEST_CASE("All DQN projection heads register sigma with their existing mean parameter names", "[dqn][head][noisynet][initialization]")
{
    const auto kind = GENERATE(0, 1, 2, 3, 4);
    anet::ConfigData data;
    data.Set("noisy.enabled", true);
    data.Set("noisy.sigma0", 0.5);
    const anet::nn::WeightInitConfig init{ .mode = "constant", .constant_val = 0.125 };
    const auto factory = CreateConfiguredProjectionHeadFactory(kind, init, data);
    anet::TensorDict features;
    features.Set(anet::nn::kKey_DefaultOutput,
        kind >= 3 ? torch::ones({ 2, 7, 4 }) : torch::ones({ 2, 4 }));
    auto head = factory->CreateHead(features);
    const auto parameters = head->named_parameters();
    const bool dueling = kind == 0 || kind == 2 || kind == 4;
    const std::vector<std::string> projections = dueling
        ? std::vector<std::string>{ "value", "adv" } : std::vector<std::string>{ "linear" };
    for (const auto& projection : projections) {
        CAPTURE(kind, projection);
        REQUIRE(parameters.contains(projection + ".sigma_weight"));
        REQUIRE(parameters.contains(projection + ".sigma_bias"));
        CHECK(torch::equal(parameters[projection + ".weight"], torch::full_like(parameters[projection + ".weight"], 0.125)));
        CHECK(torch::equal(parameters[projection + ".bias"], torch::full_like(parameters[projection + ".bias"], 0.125)));
        CHECK(torch::equal(parameters[projection + ".sigma_weight"], torch::full_like(parameters[projection + ".weight"], 0.25)));
        CHECK(torch::equal(parameters[projection + ".sigma_bias"], torch::full_like(parameters[projection + ".bias"], 0.25)));
    }
    // 全方式で既定の Q を保持し、最終射影の σ に勾配を発生させない。
    const auto output = head->Forward(features, anet::nn::NetworkForwardContext::GetDefault()).At("q");
    CHECK(torch::allclose(output, torch::full({ 2, 3 }, 0.625f)));
    output.sum().backward();
    for (const auto& projection : projections) {
        CHECK_FALSE(parameters[projection + ".sigma_weight"].grad().defined());
        CHECK_FALSE(parameters[projection + ".sigma_bias"].grad().defined());
    }
}

TEST_CASE("All DQN projection heads sample the configured noise before distribution reduction", "[dqn][head][noisynet][arithmetic]")
{
    const auto kind = GENERATE(0, 1, 2, 3, 4);
    anet::ConfigData data;
    data.Set("noisy.enabled", true);
    data.Set("nn_forward.[sample].noisy.mode", "sample");
    data.Set("nn_forward.[sample].noisy.hold", "call");
    data.Set("nn_forward.[sample].noisy.share", "batch");
    const anet::nn::WeightInitConfig init{ .mode = "constant", .constant_val = 0.125 };
    const auto factory = CreateConfiguredProjectionHeadFactory(kind, init, data);
    anet::TensorDict features;
    const auto input = kind >= 3 ? torch::arange(56, torch::kFloat32).reshape({ 2, 7, 4 }) * 0.02
        : torch::arange(8, torch::kFloat32).reshape({ 2, 4 }) * 0.2;
    features.Set(anet::nn::kKey_DefaultOutput, input);
    auto head = factory->CreateHead(features);
    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{}, anet::TensorSpecMap{},
        std::vector<std::string>{}, std::map<std::string, std::string>{});
    anet::nn::Network owner(anet::nn::NetworkConfig{}, anet::TensorSpecMap{}, nullptr, body, head);
    const anet::seed_t seed = 811;
    auto state = anet::test::MakeTestForwardState(seed);
    const anet::nn::NetworkForwardConfig config("nn_forward.[sample]", SelectForwardConfigData(data, "nn_forward.[sample]"));
    anet::nn::NetworkForwardContext::Params context_params{
        .state = state.get(), .clock_kind = anet::nn::NetworkForwardClock::kTransientRows, .clock = 0,
    };
    auto owned_head_forward = head->PrepareForward(config, anet::nn::NetworkForwardClock::kTransientRows);
    auto* head_forward = owned_head_forward ? owned_head_forward.get() : head.get();
    auto parameters = head->named_parameters();
    anet::RandomGenerator reference_random(seed);
    const auto generator = reference_random.GetTorchGenerator(input.device());
    const auto reference_projection = [&](const std::string& name) {
        const auto weight = parameters[name + ".weight"].detach();
        const auto raw_input = torch::randn({ weight.size(1) }, generator, input.options());
        const auto raw_output = torch::randn({ weight.size(0) }, generator, input.options());
        const auto epsilon_input = raw_input.sign() * raw_input.abs().sqrt();
        const auto epsilon_output = raw_output.sign() * raw_output.abs().sqrt();
        return torch::linear(input,
            weight + parameters[name + ".sigma_weight"].detach() * epsilon_output.unsqueeze(1) * epsilon_input.unsqueeze(0),
            parameters[name + ".bias"].detach() + parameters[name + ".sigma_bias"].detach() * epsilon_output);
    };
    torch::Tensor reference;
    if (kind == 0 || kind == 2 || kind == 4) {
        auto value = reference_projection("value");
        auto advantage = reference_projection("adv");
        if (kind == 2) {
            value = value.view({ 2, 1, 5 });
            advantage = advantage.view({ 2, 3, 5 });
        }
        reference = value + advantage - advantage.mean(kind == 4 ? 2 : 1, true);
    } else {
        reference = reference_projection("linear");
        if (kind == 1) reference = reference.view({ 2, 3, 5 });
    }
    if (kind >= 3) reference = reference.permute({ 0, 2, 1 }).contiguous();
    const auto output = head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params));
    const auto q = reference.dim() == 3 ? reference.mean(2) : reference;
    CAPTURE(kind);
    CHECK(torch::allclose(output.At("q"), q, 1e-5, 1e-6));
    if (kind != 0) CHECK(torch::allclose(output.At("q_dist"), reference, 1e-5, 1e-6));
    output.At("q").square().sum().backward();
    for (const auto& parameter : parameters) {
        if (parameter.key().find("sigma_") != std::string::npos) REQUIRE(parameter.value().grad().defined());
    }
    CHECK(torch::equal(head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q"), output.At("q")));
    context_params.clock = 1;
    CHECK_FALSE(torch::equal(head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q"), output.At("q")));
}

TEST_CASE("DQN Head episode noise updates each lane once per Actor clock", "[dqn][head][noisynet][episode]")
{
    const auto kind = GENERATE(0, 1, 2, 3, 4, 5);
    anet::ConfigData data;
    data.Set("noisy.enabled", true);
    data.Set("nn_forward.[episode].noisy.mode", "sample");
    data.Set("nn_forward.[episode].noisy.hold", "episode");
    data.Set("nn_forward.[episode].noisy.share", "sample");
    const anet::nn::WeightInitConfig init{ .mode = "constant", .constant_val = 0.125 };
    const auto factory = CreateConfiguredProjectionHeadFactory(kind, init, data);
    anet::TensorDict features;
    features.Set(anet::nn::kKey_DefaultOutput,
        kind == 3 || kind == 4 ? torch::ones({ 2, 7, 4 }) : torch::ones({ 2, 4 }));
    auto head = factory->CreateHead(features);
    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{}, anet::TensorSpecMap{},
        std::vector<std::string>{}, std::map<std::string, std::string>{});
    anet::nn::Network owner(anet::nn::NetworkConfig{}, anet::TensorSpecMap{}, nullptr, body, head);
    auto state = anet::test::MakeTestForwardState(853);
    const anet::nn::NetworkForwardConfig config("nn_forward.[episode]", SelectForwardConfigData(data, "nn_forward.[episode]"));
    anet::nn::NetworkForwardContext::Params context_params{
        .state = state.get(), .clock_kind = anet::nn::NetworkForwardClock::kContinuingRows, .clock = 0,
        .episode_start = torch::ones({ 2 }, torch::kBool),
    };
    auto owned_head_forward = head->PrepareForward(config, anet::nn::NetworkForwardClock::kContinuingRows);
    auto* head_forward = owned_head_forward ? owned_head_forward.get() : head.get();
    const auto initial = head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q").detach().clone();
    CHECK(torch::equal(head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q"), initial));
    context_params.clock = 1;
    context_params.episode_start = torch::tensor({ false, true }, torch::kBool);
    const auto next = head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q").detach().clone();
    CAPTURE(kind);
    CHECK(torch::equal(next.select(0, 0), initial.select(0, 0)));
    CHECK_FALSE(torch::equal(next.select(0, 1), initial.select(0, 1)));

    // 同じ時計の mask の再評価を抑え、IQN の tau 次元にも同じ ε を使う。
    context_params.episode_start = torch::tensor({ true, false }, torch::kBool);
    CHECK(torch::equal(head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q"), next));
    context_params.clock = 2;
    context_params.episode_start = torch::zeros({ 2 }, torch::kBool);
    CHECK(torch::equal(head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q"), next));
    context_params.clock = 3;
    context_params.episode_start = torch::tensor({ true, false }, torch::kBool);
    const auto third = head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q");
    CHECK_FALSE(torch::equal(third.select(0, 0), next.select(0, 0)));
    CHECK(torch::equal(third.select(0, 1), next.select(0, 1)));
}

TEST_CASE("Noisy consumers accept a named mu-only setting without holding fields", "[dqn][nn][noisynet][config]")
{
    anet::nn::InitNN();
    const auto kind = GENERATE(0, 1, 2, 3, 4, 5, 6);
    anet::ConfigData data;
    data.Set("noisy.enabled", true);
    data.Set("linear.out_features", 3);
    data.Set("init.mode", "constant");
    data.Set("init.constant_val", 0.125);
    data.Set("nn_forward.[mean].noisy.mode", "mu_only");
    const anet::nn::WeightInitConfig init{ .mode = "constant", .constant_val = 0.125 };
    const anet::nn::NetworkForwardConfig config("nn_forward.[mean]", SelectForwardConfigData(data, "nn_forward.[mean]"));
    const anet::nn::NetworkForwardContext context({});
    anet::TensorDict features;
    const auto input = kind == 3 || kind == 4 ? torch::ones({ 2, 7, 4 }) : torch::ones({ 2, 4 });
    features.Set(anet::nn::kKey_DefaultOutput, input);
    std::shared_ptr<torch::nn::Module> module;
    torch::Tensor output;
    if (kind == 6) {
        auto linear = anet::nn::NetworkModuleRepository::Instance().GetFactory("Linear")->CreateModule(data, anet::nn::ModuleContext{});
        module = linear;
        linear->Forward(input, context);
        auto prepared = linear->PrepareForward(config, anet::nn::NetworkForwardClock::kNone);
        CHECK_FALSE(prepared);
        output = linear->Forward(input, context);
    } else {
        auto head = CreateConfiguredProjectionHeadFactory(kind, init, data)->CreateHead(features);
        module = head;
        auto prepared = head->PrepareForward(config, anet::nn::NetworkForwardClock::kNone);
        CHECK_FALSE(prepared);
        output = head->Forward(features, context).At("q");
    }
    // μ-only は保持方式・共有方式・箱・時計を要求しない。
    CHECK(torch::allclose(output, torch::full({ 2, 3 }, 0.625f)));
    output.sum().backward();
    for (const auto& parameter : module->named_parameters()) {
        if (parameter.key().find("sigma_") != std::string::npos) CHECK_FALSE(parameter.value().grad().defined());
    }
}

TEST_CASE("DQN Head count holding follows Actor actions across episode boundaries", "[dqn][head][noisynet][count]")
{
    const auto kind = GENERATE(0, 1, 2, 3, 4, 5);
    anet::ConfigData data;
    data.Set("noisy.enabled", true);
    data.Set("nn_forward.[count].noisy.mode", "sample");
    data.Set("nn_forward.[count].noisy.hold", "count");
    data.Set("nn_forward.[count].noisy.count", 3);
    data.Set("nn_forward.[count].noisy.share", "batch");
    const anet::nn::WeightInitConfig init{ .mode = "constant", .constant_val = 0.125 };
    const auto factory = CreateConfiguredProjectionHeadFactory(kind, init, data);
    anet::TensorDict features;
    features.Set(anet::nn::kKey_DefaultOutput,
        kind == 3 || kind == 4 ? torch::ones({ 2, 7, 4 }) : torch::ones({ 2, 4 }));
    auto head = factory->CreateHead(features);
    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{}, anet::TensorSpecMap{},
        std::vector<std::string>{}, std::map<std::string, std::string>{});
    anet::nn::Network owner(anet::nn::NetworkConfig{}, anet::TensorSpecMap{}, nullptr, body, head);
    auto state = anet::test::MakeTestForwardState(907);
    const anet::nn::NetworkForwardConfig config("nn_forward.[count]", SelectForwardConfigData(data, "nn_forward.[count]"));
    anet::nn::NetworkForwardContext::Params context_params{
        .state = state.get(), .clock_kind = anet::nn::NetworkForwardClock::kContinuingRows, .clock = 0,
        .episode_start = torch::ones({ 2 }, torch::kBool),
    };
    auto owned_head_forward = head->PrepareForward(config, anet::nn::NetworkForwardClock::kContinuingRows);
    auto* head_forward = owned_head_forward ? owned_head_forward.get() : head.get();
    const auto initial = head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q").detach().clone();
    for (const auto clock : { 0, 1, 2 }) {
        context_params.clock = clock;
        CHECK(torch::equal(head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q"), initial));
    }
    // episode mask を周期に混ぜず、再抽選後の同一時計でも ε を維持する。
    context_params.clock = 3;
    const auto next = head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q").detach().clone();
    CHECK_FALSE(torch::equal(next, initial));
    CHECK(torch::equal(head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q"), next));
    context_params.clock = 4;
    CHECK(torch::equal(head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q"), next));
    context_params.clock = 6;
    CHECK_FALSE(torch::equal(head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q"), next));
}

TEST_CASE("DQN Head joint batch preserves the noise of each original transition", "[dqn][head][noisynet][row_mapping]")
{
    const auto kind = GENERATE(0, 1, 2, 3, 4, 5);
    anet::ConfigData data;
    data.Set("noisy.enabled", true);
    data.Set("nn_forward.[sample].noisy.mode", "sample");
    data.Set("nn_forward.[sample].noisy.hold", "call");
    data.Set("nn_forward.[sample].noisy.share", "sample");
    const anet::nn::WeightInitConfig init{ .mode = "constant", .constant_val = 0.125 };
    const auto factory = CreateConfiguredProjectionHeadFactory(kind, init, data);
    anet::TensorDict features;
    const auto input = kind == 3 || kind == 4 ? torch::ones({ 2, 7, 4 }) : torch::ones({ 2, 4 });
    features.Set(anet::nn::kKey_DefaultOutput, input);
    auto head = factory->CreateHead(features);
    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{}, anet::TensorSpecMap{},
        std::vector<std::string>{}, std::map<std::string, std::string>{});
    anet::nn::Network owner(anet::nn::NetworkConfig{}, anet::TensorSpecMap{}, nullptr, body, head);
    auto state = anet::test::MakeTestForwardState(953);
    const anet::nn::NetworkForwardConfig config("nn_forward.[sample]", SelectForwardConfigData(data, "nn_forward.[sample]"));
    anet::nn::NetworkForwardContext::Params context_params{
        .state = state.get(), .clock_kind = anet::nn::NetworkForwardClock::kTransientRows, .clock = 0,
    };
    auto owned_head_forward = head->PrepareForward(config, anet::nn::NetworkForwardClock::kTransientRows);
    auto* head_forward = owned_head_forward ? owned_head_forward.get() : head.get();
    const auto initial = head_forward->Forward(features, anet::nn::NetworkForwardContext(context_params)).At("q").detach().clone();
    anet::TensorDict joint_features;
    joint_features.Set(anet::nn::kKey_DefaultOutput, torch::cat({ input, input }, 0));
    context_params.row_mapping.emplace(torch::tensor({ 0, 1, 0, 1 }, torch::kLong), 2, torch::Device(torch::kCPU));
    CAPTURE(kind);
    CHECK(torch::equal(head_forward->Forward(joint_features, anet::nn::NetworkForwardContext(context_params)).At("q"), torch::cat({ initial, initial }, 0)));

    // current/next の結合行を元遷移へ戻し、tau 次元では ε を共有する。
    context_params.clock = 1;
    const auto next = head_forward->Forward(joint_features, anet::nn::NetworkForwardContext(context_params)).At("q").detach().clone();
    CHECK_FALSE(torch::equal(next, torch::cat({ initial, initial }, 0)));
    CHECK(torch::equal(next.slice(0, 0, 2), next.slice(0, 2, 4)));
    context_params.row_mapping.emplace(torch::tensor({ 1, 0, 1, 0 }, torch::kLong), 2, torch::Device(torch::kCPU));
    const auto reordered = head_forward->Forward(joint_features, anet::nn::NetworkForwardContext(context_params)).At("q");
    CHECK(torch::equal(reordered.select(0, 0), next.select(0, 1)));
    CHECK(torch::equal(reordered.select(0, 1), next.select(0, 0)));
}

TEST_CASE("DQN Head validates forward requirements locally while disabled heads stay dormant", "[dqn][head][noisynet][validation]")
{
    const auto kind = GENERATE(0, 1, 2, 3, 4, 5);
    const auto enabled = GENERATE(false, true);
    const auto sigma0 = GENERATE(0.0, 0.5);
    anet::ConfigData data;
    data.Set("noisy.enabled", enabled);
    data.Set("noisy.sigma0", sigma0);
    data.Set("nn_forward.[missing].noisy.mode", "sample");
    const anet::nn::WeightInitConfig init{ .mode = "constant", .constant_val = 0.125 };
    anet::TensorDict features;
    features.Set(anet::nn::kKey_DefaultOutput,
        kind == 3 || kind == 4 ? torch::ones({ 2, 7, 4 }) : torch::ones({ 2, 4 }));
    auto head = CreateConfiguredProjectionHeadFactory(kind, init, data)->CreateHead(features);
    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{}, anet::TensorSpecMap{},
        std::vector<std::string>{}, std::map<std::string, std::string>{});
    anet::nn::Network network(anet::nn::NetworkConfig{}, anet::TensorSpecMap{}, nullptr, body, head);
    const anet::nn::NetworkForwardConfig missing("nn_forward.[missing]", SelectForwardConfigData(data, "nn_forward.[missing]"));
    CAPTURE(kind, enabled, sigma0);
    if (enabled) {
        CHECK_THROWS_WITH(head->PrepareForward(missing, anet::nn::NetworkForwardClock::kContinuingRows),
            Catch::Matchers::ContainsSubstring("nn_forward.[missing]") && Catch::Matchers::ContainsSubstring("noisy.hold"));
    } else {
        CHECK_NOTHROW(head->PrepareForward(missing, anet::nn::NetworkForwardClock::kContinuingRows));
    }
    data.Set("nn_forward.[count].noisy.mode", "sample");
    data.Set("nn_forward.[count].noisy.hold", "count");
    data.Set("nn_forward.[count].noisy.count", 3);
    data.Set("nn_forward.[count].noisy.share", "batch");
    const anet::nn::NetworkForwardConfig count("nn_forward.[count]", SelectForwardConfigData(data, "nn_forward.[count]"));
    CHECK_NOTHROW(head->PrepareForward(count, anet::nn::NetworkForwardClock::kContinuingRows));
    if (enabled) {
        CHECK_THROWS_WITH(head->PrepareForward(count, anet::nn::NetworkForwardClock::kTransientRows),
            Catch::Matchers::ContainsSubstring("nn_forward.[count]") && Catch::Matchers::ContainsSubstring("continuing rows"));
    } else {
        CHECK_NOTHROW(head->PrepareForward(count, anet::nn::NetworkForwardClock::kTransientRows));
    }
    data.Set("nn_forward.[mean].noisy.mode", "mu_only");
    const anet::nn::NetworkForwardConfig mean("nn_forward.[mean]", SelectForwardConfigData(data, "nn_forward.[mean]"));
    CHECK_NOTHROW(head->PrepareForward(mean, anet::nn::NetworkForwardClock::kNone));
}

TEST_CASE("Network output reuse checks noise modes only in consuming components", "[nn][dqn][noisynet][online_reuse]")
{
    anet::nn::InitNN();
    const auto kind = GENERATE(0, 1, 2, 3, 4, 5, 6, 7);
    const auto enabled = GENERATE(false, true);
    const auto sigma0 = GENERATE(0.0, 0.5);
    anet::ConfigData data;
    data.Set("noisy.enabled", enabled);
    data.Set("noisy.sigma0", sigma0);
    data.Set("nn_forward.[sample].noisy.mode", "sample");
    data.Set("nn_forward.[sample].noisy.hold", "call");
    data.Set("nn_forward.[sample].noisy.share", "batch");
    data.Set("nn_forward.[mean].noisy.mode", "mu_only");
    const anet::nn::WeightInitConfig init{ .mode = "constant", .constant_val = 0.125 };
    std::shared_ptr<anet::nn::Network> network;
    if (kind >= 6) {
        data.Set("net.block.[Linear].type", "Linear");
        data.Set("net.block.[Linear].linear.out_features", 4);
        data.Set("net.block.[Linear].noisy.enabled", enabled);
        data.Set("net.block.[Linear].noisy.sigma0", sigma0);
        data.Set("net.branch.[feature].bind", "obs");
        data.Set("net.branch.[feature].structure", "Linear");
        data.Set("net.body.output.[features]", "feature");
        const anet::TensorSpec spec{ .type = anet::SpaceType::Vector, .shape = { 4 }, .dtype = torch::kFloat32 };
        network = anet::nn::NetworkBuilder::BuildNetwork(
            anet::nn::NetworkConfig(data), anet::TensorSpecMap{ { "obs", spec } },
            kind == 7 ? CreateConfiguredProjectionHeadFactory(5, init, data) : nullptr, 1151);
    } else {
        anet::TensorDict features;
        features.Set(anet::nn::kKey_DefaultOutput,
            kind == 3 || kind == 4 ? torch::ones({ 2, 7, 4 }) : torch::ones({ 2, 4 }));
        auto head = CreateConfiguredProjectionHeadFactory(kind, init, data)->CreateHead(features);
        auto body = std::make_shared<anet::nn::NetworkBody>(
            std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{}, anet::TensorSpecMap{},
            std::vector<std::string>{}, std::map<std::string, std::string>{});
        network = std::make_shared<anet::nn::Network>(anet::nn::NetworkConfig{}, anet::TensorSpecMap{}, nullptr, body, head);
    }
    const anet::nn::NetworkForwardConfig sample("nn_forward.[sample]", SelectForwardConfigData(data, "nn_forward.[sample]"));
    const anet::nn::NetworkForwardConfig mean("nn_forward.[mean]", SelectForwardConfigData(data, "nn_forward.[mean]"));
    CAPTURE(kind, enabled, sigma0);
    CHECK_NOTHROW(network->ValidateForwardReuse(sample, sample));
    CHECK_NOTHROW(network->ValidateForwardReuse(mean, mean));
    CHECK_NOTHROW(network->ValidateForwardReuse(anet::nn::NetworkForwardConfig::GetDefault(), mean));
    if (enabled) {
        // σ0=0 でも要求した方式の意味を変更せず、sample↔μ-only の再利用を拒否する。
        CHECK_THROWS_WITH(network->ValidateForwardReuse(sample, mean),
            Catch::Matchers::ContainsSubstring("online_reuse") && Catch::Matchers::ContainsSubstring("nn_forward.[sample]")
            && Catch::Matchers::ContainsSubstring("nn_forward.[mean]"));
        CHECK_THROWS_WITH(network->ValidateForwardReuse(mean, sample), Catch::Matchers::ContainsSubstring("online_reuse"));
    } else {
        CHECK_NOTHROW(network->ValidateForwardReuse(sample, mean));
        CHECK_NOTHROW(network->ValidateForwardReuse(mean, sample));
    }
}

TEST_CASE("DQN Head factories delegate declaration validation to their concrete heads", "[dqn][head][noisynet][config]")
{
    const auto kind = GENERATE(0, 1, 2, 3, 4, 5);
    const anet::nn::WeightInitConfig init{ .mode = "constant", .constant_val = 0.125 };
    const auto factory = CreateConfiguredProjectionHeadFactory(kind, init, anet::EmptyConfigData);
    const auto validate = [&factory](const anet::ConfigData& data) {
        const anet::nn::NetworkForwardConfig config("nn_forward.[head_declaration]", SelectForwardConfigData(data, "nn_forward.[head_declaration]"));
        factory->ValidateForwardConfigDeclaration(config);
    };
    anet::ConfigData data;
    data.Set("nn_forward.[head_declaration].noisy.mode", "mu_only");
    CHECK_NOTHROW(validate(data));
    for (const auto& [leaf, value] : std::vector<std::pair<std::string, std::string>>{
        { "mode", "other" }, { "hold", "other" }, { "share", "other" }, { "count", "0" },
        { "count", "-1" }, { "count", "1.5" },
    }) {
        auto invalid = data;
        invalid.Set("nn_forward.[head_declaration].noisy." + leaf, value);
        CAPTURE(kind, leaf, value);
        CHECK_THROWS_WITH(validate(invalid), Catch::Matchers::ContainsSubstring("noisy." + leaf));
    }
    data.Set("nn_forward.[head_declaration].noisy.mode", "sample");
    data.Set("nn_forward.[head_declaration].noisy.hold", "episode");
    data.Set("nn_forward.[head_declaration].noisy.share", "batch");
    CHECK_THROWS_WITH(validate(data), Catch::Matchers::ContainsSubstring("episode") && Catch::Matchers::ContainsSubstring("sample"));
    data.Set("nn_forward.[head_declaration].noisy.share", "sample");
    CHECK_NOTHROW(validate(data));
    data.Set("nn_forward.[head_declaration].noisy.hold", "count");
    CHECK_NOTHROW(validate(data));
}
