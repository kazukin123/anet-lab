#include "catch.hpp"

#include "dqn_based_agent.hpp"
#include "nn_impl.hpp"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace rl = anet::rl;
namespace dqn = anet::rl::dqn;

struct QuantileLearnerBaseAccess : public dqn::QuantileLearnerBase {
    using dqn::QuantileLearnerBase::ComputeQuantileHuberLoss;
};

constexpr const char* kFeatureKey = "feature";
constexpr const char* kVectorKey = rl::ObsKeys::kVector;

class TestLinearHead final : public anet::nn::NetworkHead {
public:
    explicit TestLinearHead(int64_t in_features, int64_t out_features)
    {
        linear_ = register_module(
            "linear",
            torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(false)));
    }

    anet::TensorDict Forward(const anet::TensorDict& feature_dict) override
    {
        return anet::TensorDict{ { "q", linear_->forward(feature_dict.At(kFeatureKey)) } };
    }

private:
    torch::nn::Linear linear_{ nullptr };
};

std::shared_ptr<anet::nn::Network> MakeLinearNetwork()
{
    anet::TensorSpec vector_spec;
    vector_spec.type = anet::SpaceType::Vector;
    vector_spec.shape = { 2 };
    vector_spec.dtype = torch::kFloat32;

    anet::TensorSpecMap input_specs;
    input_specs[kVectorKey] = vector_spec;

    anet::nn::NetworkConfig network_config;
    network_config.output_keys[kFeatureKey] = kVectorKey;

    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{},
        input_specs,
        std::vector<std::string>{},
        network_config.output_keys);
    auto head = std::make_shared<TestLinearHead>(2, 1);

    return std::make_shared<anet::nn::Network>(
        network_config,
        input_specs,
        nullptr,
        body,
        head);
}

class TestNetworkModel final : public dqn::NetworkModel {
public:
    TestNetworkModel()
        : dqn::NetworkModel(
            dqn::NetworkModelConfig{},
            MakeLinearNetwork(),
            MakeLinearNetwork(),
            1,
            1)
    {
        GetMainNetwork()->CopyTo(*GetTargetNetwork());
        GetTargetNetwork()->eval();
    }
};

class TestLearner final : public dqn::Learner {
public:
    TestLearner(
        const dqn::LearnerConfig& config,
        dqn::NetworkModel& model,
        dqn::RuntimeVars& vars,
        const rl::BatchEnvSpec& batch_env_spec,
        const rl::EnvSpec& env_spec)
        : dqn::Learner(
            config,
            model,
            vars,
            nullptr,
            batch_env_spec,
            env_spec,
            torch::Device(torch::kCPU),
            123,
            std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}),
            std::nullopt,
            456)
    {
    }

    using dqn::Learner::MakePerPriorityUpdateInfo;
    using dqn::Learner::Optimize;

    void UseSgd(float lr)
    {
        optimizer_ = std::make_unique<torch::optim::SGD>(
            model_.GetPolicyParameters(),
            torch::optim::SGDOptions(lr));
    }

    std::shared_ptr<const anet::rl::BatchUpdateResult> UpdateFromSamples(
        const anet::rl::ExperienceSamples& samples) override
    {
        return nullptr;
    }
};

std::vector<int64_t> ShapeOf(const torch::Tensor& tensor)
{
    return tensor.sizes().vec();
}

rl::EnvSpec MakeLearnerEnvSpec()
{
    anet::TensorSpec vector_spec;
    vector_spec.type = anet::SpaceType::Vector;
    vector_spec.shape = { 2 };
    vector_spec.dtype = torch::kFloat32;

    rl::EnvSpec spec;
    spec.state_spec.obs_spec[kVectorKey] = vector_spec;
    spec.action_spec.is_discrete = true;
    spec.action_spec.value_labels = { "a0" };
    spec.reward_range = { -1000.0f, 1000.0f };
    return spec;
}

std::shared_ptr<anet::nn::Network> MakePassthroughNetwork(int64_t n_actions, int64_t n_quantiles)
{
    anet::TensorSpec q_spec;
    q_spec.shape = { n_actions };
    q_spec.dtype = torch::kFloat32;

    anet::TensorSpec q_dist_spec;
    q_dist_spec.shape = { n_actions, n_quantiles };
    q_dist_spec.dtype = torch::kFloat32;

    anet::TensorSpecMap input_specs;
    input_specs["q"] = q_spec;
    input_specs["q_dist"] = q_dist_spec;

    anet::nn::NetworkConfig network_config;
    network_config.output_keys["q"] = "q";
    network_config.output_keys["q_dist"] = "q_dist";

    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{},
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

anet::TensorDict MakePolicyInput()
{
    auto q_values = torch::tensor({
        { 0.0f, 10.0f, 5.0f },
        { 2.0f, 6.0f, 4.0f },
    });

    auto q_quantiles = torch::tensor({
        {
            { 0.0f, 0.0f, 0.0f, 0.0f },
            { 10.0f, 10.0f, 10.0f, 10.0f },
            { 5.0f, 5.0f, 5.0f, 5.0f },
        },
        {
            { 2.0f, 2.0f, 2.0f, 2.0f },
            { 6.0f, 6.0f, 6.0f, 6.0f },
            { 4.0f, 4.0f, 4.0f, 4.0f },
        },
    });

    return anet::TensorDict{
        { "q", q_values },
        { "q_dist", q_quantiles },
    };
}

} // namespace

TEST_CASE("Quantile huber loss matches known QR-DQN inputs", "[dqn][quantile]")
{
    auto current_dist = torch::tensor({ { 1.0f, 3.0f } });
    auto target_dist = torch::tensor({ { 2.0f, 4.0f } });
    auto taus = torch::tensor({ 0.25f, 0.75f }).view({ 1, 2, 1 });

    auto loss = QuantileLearnerBaseAccess::ComputeQuantileHuberLoss(
        current_dist,
        target_dist,
        taus,
        1.0f);

    REQUIRE(ShapeOf(loss) == std::vector<int64_t>{ 1 });
    REQUIRE(loss.item<float>() == Catch::Approx(0.625f).margin(1.0e-6f));
}

TEST_CASE("PER priority helper applies epsilon and clipping", "[dqn][per]")
{
    dqn::LearnerConfig config;
    config.use_per = true;
    config.per_eps = 0.1f;
    config.use_per_prio_clip = true;
    config.per_prio_clip_value = 1.0f;

    auto env_spec = MakeLearnerEnvSpec();
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    rl::BatchEnvSpec batch_env_spec{ 1, 1 };
    TestLearner learner(config, model, vars, batch_env_spec, env_spec);

    rl::ExperienceSamples samples;
    samples.is_weights = torch::tensor({ 0.25f, 0.75f });

    auto td_error = torch::tensor({ -0.2f, 2.0f });
    auto result = learner.MakePerPriorityUpdateInfo(samples, td_error);

    REQUIRE(result.per_priorities.defined());
    CHECK(torch::allclose(result.per_priorities, torch::tensor({ 0.3f, 1.0f })));
    REQUIRE(result.per_clipped_count.defined());
    CHECK(result.per_clipped_count.item<int64_t>() == 1);
    CHECK(result.per_minibatch_size == 2);
    REQUIRE(result.per_is_weights.defined());
    CHECK(torch::allclose(result.per_is_weights, samples.is_weights));
}

TEST_CASE("Optimizer helper keeps QR-DQN FP32 grad clip result contract", "[dqn][optimizer]")
{
    dqn::LearnerConfig config;
    config.use_amp = false;
    config.use_amp_bf16 = false;
    config.use_grad_clip = true;
    config.grad_clip_tau = 0.5f;

    auto env_spec = MakeLearnerEnvSpec();
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    rl::BatchEnvSpec batch_env_spec{ 1, 1 };
    TestLearner learner(config, model, vars, batch_env_spec, env_spec);
    learner.UseSgd(0.1f);

    auto weight_before = model.GetPolicyParameters()[0].detach().clone().cpu();
    auto obs = anet::TensorDict{ { kVectorKey, torch::tensor({ { 3.0f, 4.0f } }) } };
    auto loss = model.GetMainNetwork()->Forward(obs).At("q").sum();
    auto result = learner.Optimize(loss);

    CHECK_FALSE(result.grad_norm.has_value());
    REQUIRE(result.grad_norm_tensor.defined());
    CHECK(result.grad_norm_tensor.item<float>() == Catch::Approx(5.0f).margin(1.0e-5f));
    CHECK(result.grad_clip_tau == Catch::Approx(0.5f).margin(1.0e-6f));
    CHECK(result.grad_clip_ratio == Catch::Approx(0.0f).margin(1.0e-6f));

    auto weight_delta = model.GetPolicyParameters()[0].detach().cpu() - weight_before;
    CHECK(weight_delta[0][0].item<float>() == Catch::Approx(-0.03f).margin(1.0e-5f));
    CHECK(weight_delta[0][1].item<float>() == Catch::Approx(-0.04f).margin(1.0e-5f));
}

TEST_CASE("ActionPolicy variants preserve action info keys and shapes", "[dqn][action_policy]")
{
    auto network = MakePassthroughNetwork(3, 4);
    auto obs = MakePolicyInput();

    dqn::ActionPolicyConfig config;
    std::vector<std::pair<std::string, std::shared_ptr<dqn::ActionPolicy>>> policies;
    policies.emplace_back("epsilon-greedy", std::make_shared<dqn::EpsilonGreedyActionPolicy>(config));
    policies.emplace_back("uqe", std::make_shared<dqn::UQEActionPolicy>(config));
    policies.emplace_back("thompson-sampling", std::make_shared<dqn::ThompsonSamplingActionPolicy>(config));

    auto expected_actions = torch::tensor({ 1, 1 }, torch::TensorOptions().dtype(torch::kInt64));
    auto expected_max_q = torch::tensor({ 10.0f, 6.0f });

    for (const auto& [name, policy] : policies) {
        INFO(name);
        auto rnd = std::make_shared<anet::RandomGenerator>(123);
        auto action_info = policy->SelectAction(obs, /*greedy_only=*/true, network, rnd);

        auto action = action_info.GetAction();
        REQUIRE(ShapeOf(action) == std::vector<int64_t>{ 2 });
        CHECK(torch::equal(action.cpu(), expected_actions));

        const auto& aux = action_info.GetAuxData();
        REQUIRE(aux.count("max_q") == 1);
        REQUIRE(aux.count("q_values") == 1);
        REQUIRE(aux.count("q_quantiles") == 1);
        REQUIRE(aux.count("raw_actions") == 1);

        CHECK(ShapeOf(aux.at("max_q")) == std::vector<int64_t>{ 2 });
        CHECK((ShapeOf(aux.at("q_values")) == std::vector<int64_t>{ 2, 3 }));
        CHECK((ShapeOf(aux.at("q_quantiles")) == std::vector<int64_t>{ 2, 3, 4 }));
        CHECK(ShapeOf(aux.at("raw_actions")) == std::vector<int64_t>{ 2 });

        CHECK(torch::allclose(aux.at("max_q"), expected_max_q));
        CHECK(torch::allclose(aux.at("q_values"), obs.At("q")));
        CHECK(torch::allclose(aux.at("q_quantiles"), obs.At("q_dist")));
        CHECK(torch::equal(aux.at("raw_actions").cpu(), expected_actions));
    }
}
