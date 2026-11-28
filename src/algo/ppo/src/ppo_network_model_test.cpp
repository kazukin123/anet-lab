#include "anet/test/catch.hpp"
#include <memory>
#include <string>
#include "anet/config.hpp"
#include "anet/nn.hpp"
#include "ppo_fixture_test.hpp"
#include "ppo_network_model.hpp"

using namespace anet::rl::ppo;
using namespace anet::rl::ppo::test;

constexpr char kNetworkConfigPrefix[] = "PPOAgent.net";
constexpr char kCriticObsKey[] = "critic";
constexpr char kFlattenBlock[] = "Flatten";
constexpr int64_t kObsSize = 4;
constexpr int64_t kCriticObsSize = 3;
constexpr int64_t kActionCount = 3;
constexpr anet::seed_t kSeed = 7;

static anet::TensorSpecMap MakeObsSpecs(bool with_critic_obs = false)
{
    anet::TensorSpecMap specs;
    specs[anet::rl::ObsKeys::kVector] = anet::TensorSpec{
        .type = anet::SpaceType::Vector, .shape = { kObsSize }, .dtype = torch::kFloat32 };
    if (with_critic_obs) {
        specs[kCriticObsKey] = anet::TensorSpec{
            .type = anet::SpaceType::Vector, .shape = { kCriticObsSize }, .dtype = torch::kFloat32 };
    }
    return specs;
}

static std::shared_ptr<NetworkModel> MakeModel(const anet::ConfigData& data, const anet::TensorSpecMap& obs_specs,
    torch::Device device = torch::Device(torch::kCPU))
{
    const anet::nn::NetworkConfig network_config(data, kNetworkConfigPrefix);
    return std::make_shared<NetworkModel>(network_config, obs_specs, kActionCount, anet::nn::WeightInitConfig{},
        anet::nn::WeightInitConfig{}, device, kSeed);
}

TEST_CASE("PPO network shares or separates the policy and value bodies by the branch configuration alone",
    "[ppo][ppo_network][rfd0209]")
{
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");

    const bool shared_body = GENERATE(true, false);
    INFO("shared_body=" << shared_body);
    const auto model = MakeModel(MakeConfigData(shared_body), MakeObsSpecs(), device);

    // 出力は、方策の logits [行, 行動数] と、価値 [行, 1] である。
    const auto obs = anet::TensorDict({ { anet::rl::ObsKeys::kVector, torch::rand({ 5, kObsSize }).to(device) } });
    const auto outputs = model->GetNetwork()->Forward(NetworkModel::MakeNetworkInput(obs));
    CHECK(outputs.At(NetworkModel::kLogitsKey).sizes() == torch::IntArrayRef({ 5, kActionCount }));
    CHECK(outputs.At(NetworkModel::kValueKey).sizes() == torch::IntArrayRef({ 5, 1 }));

    // 両方の Head へ届く parameter は、共有の設定にだけある。
    const auto groups = model->MakeParameterGroups();
    CHECK(groups.shared.empty() == !shared_body);
    CHECK_FALSE(groups.policy.empty());
    CHECK_FALSE(groups.value.empty());
    // 3 つの組は、学習する parameter の全部を、重なりなく分ける。
    size_t trainable_count = 0;
    for (const auto& parameter : model->GetNetwork()->parameters()) {
        if (parameter.requires_grad()) ++trainable_count;
    }
    CHECK(groups.shared.size() + groups.policy.size() + groups.value.size() == trainable_count);
}

TEST_CASE("PPO network with separate bodies can feed the value Head from an observation the policy does not read",
    "[ppo][ppo_network][rfd0209]")
{
    // 方策は vector、価値は critic の観測だけを読む（Actor と Critic の入力を分ける構成）。
    auto data = MakeConfigData(/*shared_body=*/false);
    data.Set("PPOAgent.net.branch.[value_body].bind", kCriticObsKey);
    const auto model = MakeModel(data, MakeObsSpecs(/*with_critic_obs=*/true));

    const auto vector = torch::rand({ 2, kObsSize });
    const auto forward = [&model, &vector](const torch::Tensor& critic) {
        const auto obs = anet::TensorDict({ { anet::rl::ObsKeys::kVector, vector }, { kCriticObsKey, critic } });
        return model->GetNetwork()->Forward(NetworkModel::MakeNetworkInput(obs));
    };
    const auto first = forward(torch::zeros({ 2, kCriticObsSize }));
    const auto second = forward(torch::ones({ 2, kCriticObsSize }));

    // 価値だけが読む観測を変えると、value は変わり、logits は変わらない。
    CHECK(torch::equal(first.At(NetworkModel::kLogitsKey), second.At(NetworkModel::kLogitsKey)));
    CHECK_FALSE(torch::allclose(first.At(NetworkModel::kValueKey), second.At(NetworkModel::kValueKey)));
}

TEST_CASE("PPO network input adds a stack axis of length 1 to every observation key", "[ppo][ppo_network][rfd0209]")
{
    // spec は先頭に長さ 1 の stack 軸を持ち、値は batch 軸の次に同じ軸を持つ。
    const auto specs = NetworkModel::MakeInputSpecs(MakeObsSpecs(/*with_critic_obs=*/true));
    CHECK(specs.at(anet::rl::ObsKeys::kVector).shape == std::vector<int64_t>{ 1, kObsSize });
    CHECK(specs.at(anet::rl::ObsKeys::kVector).has_stack_axis);
    CHECK(specs.at(kCriticObsKey).shape == std::vector<int64_t>{ 1, kCriticObsSize });

    const auto vector = torch::rand({ 2, kObsSize });
    const auto input = NetworkModel::MakeNetworkInput(anet::TensorDict({ { anet::rl::ObsKeys::kVector, vector } }));
    CHECK(input.At(anet::rl::ObsKeys::kVector).sizes() == torch::IntArrayRef({ 2, 1, kObsSize }));
    CHECK(torch::equal(input.At(anet::rl::ObsKeys::kVector).squeeze(1), vector));
}

TEST_CASE("PPO network rejects configurations whose Heads cannot produce logits and a value per row",
    "[ppo][ppo_network][rfd0209]")
{
    SECTION("a body that keeps the stack axis")
    {
        // Flatten が無いと、stack 軸が Head の出力まで残り、logits が [行, 1, 行動数] になる。
        auto data = MakeConfigData(/*shared_body=*/true);
        data.Set("PPOAgent.net.branch.[shared_body].structure", "Hidden > Tanh");
        CHECK_THROWS_WITH(MakeModel(data, MakeObsSpecs()), Catch::Matchers::ContainsSubstring(kNetworkConfigPrefix)
            && Catch::Matchers::ContainsSubstring(kFlattenBlock));
    }

    SECTION("a Head input key that no branch feeds")
    {
        // 価値の Head が受け取るキーを、どの branch にも向けていない。
        anet::test::RegisterStandardNetworkModules();
        anet::ConfigData data;
        data.Set("net.block.[Flatten].type", kFlattenBlock);
        data.Set("PPOAgent.net.branch.[policy_body].bind", anet::rl::ObsKeys::kVector);
        data.Set("PPOAgent.net.branch.[policy_body].structure", kFlattenBlock);
        data.Set("PPOAgent.net.body.output.[policy_feature]", "policy_body");
        CHECK_THROWS_WITH(MakeModel(data, MakeObsSpecs()),
            Catch::Matchers::ContainsSubstring("PPOAgent.net.body.output.[value_feature]"));
    }
}

TEST_CASE("PPO network model counts policy versions and clones keep the weights", "[ppo][ppo_network][rfd0209]")
{
    const auto model = MakeModel(MakeConfigData(), MakeObsSpecs());
    CHECK(model->GetPolicyVersion() == 0);
    model->AdvancePolicyVersion();
    CHECK(model->GetPolicyVersion() == 1);

    // 写しは、同じ入力に同じ出力を返す別の Network である。
    const auto clone = model->CloneNetwork(torch::Device(torch::kCPU));
    CHECK(clone != model->GetNetwork());
    const auto input = NetworkModel::MakeNetworkInput(
        anet::TensorDict({ { anet::rl::ObsKeys::kVector, torch::rand({ 2, kObsSize }) } }));
    CHECK(torch::equal(clone->Forward(input).At(NetworkModel::kLogitsKey),
        model->GetNetwork()->Forward(input).At(NetworkModel::kLogitsKey)));
}
