#include "catch.hpp"

#include "anet/default_dqn_agent.hpp"
#include "anet/rainbow_agent.hpp"
#include "anet/test_util.hpp"
#include "dqn_based_agent.hpp"
#include "nn_impl.hpp"

#include <cmath>
#include <limits>
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

struct ActionPolicyAccess : public dqn::ActionPolicy {
    ActionPolicyAccess()
        : dqn::ActionPolicy(dqn::ActionPolicyConfig{})
    {
    }

    using dqn::ActionPolicy::CreateSpatialTensor;

    std::shared_ptr<anet::rl::BatchActionInfo> SelectAction(const anet::TensorDict&, bool, std::shared_ptr<anet::nn::Network>,
        std::shared_ptr<anet::RandomGenerator>, const anet::TraceSink&) const override
    {
        return std::make_shared<anet::rl::BatchActionInfo>();
    }
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

    using dqn::Learner::MakeBatchUpdateResult;
    using dqn::Learner::MakePerPriorityUpdateInfo;
    using dqn::Learner::Optimize;
    using dqn::Learner::TransformH;
    using dqn::Learner::TransformHInv;

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

anet::TensorDict MakeSpatialUQEInput()
{
    auto q_values = torch::zeros({ 2, 2 });
    auto q_quantiles = torch::tensor({
        {
            { 5.0f, 5.0f },
            { 0.0f, 100.0f },
        },
        {
            { 0.0f, 0.0f },
            { 10.0f, 10.0f },
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

TEST_CASE("TBO transform is monotonic and invertible on representative values", "[dqn][tbo]")
{
    auto env_spec = MakeLearnerEnvSpec();
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    rl::BatchEnvSpec batch_env_spec{ 1, 1 };

    for (float epsilon : { 1.0e-2f, 1.0e-3f }) {
        INFO(epsilon);
        dqn::LearnerConfig config;
        config.tbo_epsilon = epsilon;
        TestLearner learner(config, model, vars, batch_env_spec, env_spec);

        auto values = torch::tensor({ -1000.0f, -10.0f, -1.0f, 0.0f, 1.0f, 10.0f, 1000.0f });
        auto transformed = learner.TransformH(values);
        auto restored_from_values = learner.TransformHInv(transformed);
        auto restored_from_transformed = learner.TransformH(learner.TransformHInv(values));

        CHECK(torch::allclose(restored_from_values, values, 1.0e-4, 1.0e-4));
        CHECK(torch::allclose(restored_from_transformed, values, 1.0e-4, 1.0e-4));

        auto diffs = transformed.slice(0, 1) - transformed.slice(0, 0, -1);
        CHECK(torch::all(diffs.gt(0)).item<bool>());
    }
}

TEST_CASE("TBO real-space q scalars are exposed from batch update result", "[dqn][tbo][metrics]")
{
    dqn::LearnerConfig config;
    config.use_tbo = true;
    config.tbo_epsilon = 1.0e-2f;

    auto env_spec = MakeLearnerEnvSpec();
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    rl::BatchEnvSpec batch_env_spec{ 1, 1 };
    TestLearner learner(config, model, vars, batch_env_spec, env_spec);

    auto raw_max_q = torch::tensor({ -10.0f, 0.0f, 100.0f });
    auto raw_q_sa = torch::tensor({ -1.0f, 10.0f, 1000.0f });
    auto max_q = learner.TransformH(raw_max_q);
    auto q_sa = learner.TransformH(raw_q_sa);

    dqn::OptimizerStepResult opt_result;
    dqn::PerPriorityUpdateInfo per_info;
    auto result = learner.MakeBatchUpdateResult(
        torch::tensor(0.0f),
        torch::zeros({ 3 }),
        opt_result,
        max_q,
        q_sa,
        per_info);

    auto max_mean = result->GetScalar("q_max_real_mean", -1);
    auto max_max = result->GetScalar("q_max_real_max", -1);
    auto max_std = result->GetScalar("q_max_real_std", -1);
    auto sa_mean = result->GetScalar("q_sa_real_mean", -1);
    REQUIRE(max_mean.has_value());
    REQUIRE(max_max.has_value());
    REQUIRE(max_std.has_value());
    REQUIRE(sa_mean.has_value());
    CHECK(*max_mean == Catch::Approx(raw_max_q.mean().item<float>()).margin(1.0e-4f));
    CHECK(*max_max == Catch::Approx(raw_max_q.max().item<float>()).margin(1.0e-4f));
    CHECK(*max_std == Catch::Approx(raw_max_q.std(false).item<float>()).margin(1.0e-4f));
    CHECK(*sa_mean == Catch::Approx(raw_q_sa.mean().item<float>()).margin(1.0e-4f));

    dqn::LearnerConfig off_config;
    TestLearner off_learner(off_config, model, vars, batch_env_spec, env_spec);
    auto off_result = off_learner.MakeBatchUpdateResult(
        torch::tensor(0.0f),
        torch::zeros({ 3 }),
        opt_result,
        raw_max_q,
        raw_q_sa,
        per_info);
    CHECK(off_result->GetScalar("q_max_real_mean", -1).has_value());
    CHECK(off_result->GetScalar("q_sa_real_mean", -1).has_value());
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

        auto action = action_info->GetAction();
        REQUIRE(ShapeOf(action) == std::vector<int64_t>{ 2 });
        CHECK(torch::equal(action.cpu(), expected_actions));

        const auto& aux = action_info->GetAuxData();
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

        auto scalar_target = dynamic_cast<const anet::Module*>(action_info.get());
        REQUIRE(scalar_target != nullptr);
        auto uqe_win_rate = scalar_target->GetScalar("action_uqe_win_rate.[0]");
        auto uqe_margin = scalar_target->GetScalar("action_uqe_margin.[0]");
        REQUIRE(uqe_win_rate.has_value());
        REQUIRE(uqe_margin.has_value());
        if (name == "epsilon-greedy") {
            CHECK(std::isnan(*uqe_win_rate));
            CHECK(std::isnan(*uqe_margin));
        } else {
            REQUIRE(aux.count("uqe_values") == 1);
            CHECK((ShapeOf(aux.at("uqe_values")) == std::vector<int64_t>{ 2, 3 }));
            CHECK(torch::allclose(aux.at("uqe_values"), obs.At("q")));
            CHECK(*uqe_win_rate == Catch::Approx(0.0f));
            CHECK(*uqe_margin == Catch::Approx(-7.0f));
        }
    }
}

TEST_CASE("DQNActionInfo exposes action UQE scalar metrics", "[dqn][action_policy][metrics]")
{
    auto make_info = [](const torch::Tensor& uqe_values) {
        rl::AuxData aux;
        aux["uqe_values"] = uqe_values;
        return dqn::DQNActionInfo(
            torch::zeros({ uqe_values.size(0) }, torch::TensorOptions().dtype(torch::kInt64)),
            anet::TensorDict{},
            aux);
    };

    auto win_info = make_info(torch::tensor({
        { 5.0f, 1.0f, 0.0f },
        { 7.0f, 6.0f, 5.0f },
    }));
    auto win = win_info.GetScalar("action_uqe_win_rate.[0]");
    REQUIRE(win.has_value());
    CHECK(*win == Catch::Approx(1.0f));
    auto win_margin = win_info.GetScalar("action_uqe_margin.[0]");
    REQUIRE(win_margin.has_value());
    CHECK(*win_margin == Catch::Approx(2.5f));

    auto loss = win_info.GetScalar("action_uqe_win_rate.[1]");
    REQUIRE(loss.has_value());
    CHECK(*loss == Catch::Approx(0.0f));
    auto loss_margin = win_info.GetScalar("action_uqe_margin.[1]");
    REQUIRE(loss_margin.has_value());
    CHECK(*loss_margin == Catch::Approx(-2.5f));

    auto tie_info = make_info(torch::tensor({
        { 5.0f, 5.0f, 0.0f },
        { 1.0f, 1.0f, 0.0f },
    }));
    auto tie = tie_info.GetScalar("action_uqe_win_rate.[0]");
    REQUIRE(tie.has_value());
    CHECK(*tie == Catch::Approx(1.0f));
    auto tie_margin = tie_info.GetScalar("action_uqe_margin.[0]");
    REQUIRE(tie_margin.has_value());
    CHECK(*tie_margin == Catch::Approx(0.0f));

    dqn::DQNActionInfo non_uqe(torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kInt64)));
    auto undefined = non_uqe.GetScalar("action_uqe_win_rate.[0]");
    REQUIRE(undefined.has_value());
    CHECK(std::isnan(*undefined));
    auto undefined_margin = non_uqe.GetScalar("action_uqe_margin.[0]");
    REQUIRE(undefined_margin.has_value());
    CHECK(std::isnan(*undefined_margin));

    auto replaced = win_info.WithAction(torch::tensor({ 2, 1 }, torch::TensorOptions().dtype(torch::kInt64)));
    CHECK(torch::equal(replaced->GetAction(), torch::tensor({ 2, 1 }, torch::TensorOptions().dtype(torch::kInt64))));
    auto scalar_target = dynamic_cast<const anet::Module*>(replaced.get());
    REQUIRE(scalar_target != nullptr);
    auto replaced_win = scalar_target->GetScalar("action_uqe_win_rate.[0]");
    REQUIRE(replaced_win.has_value());
    CHECK(*replaced_win == Catch::Approx(1.0f));
    auto replaced_margin = scalar_target->GetScalar("action_uqe_margin.[0]");
    REQUIRE(replaced_margin.has_value());
    CHECK(*replaced_margin == Catch::Approx(2.5f));

    CHECK_THROWS(non_uqe.GetScalar("action_uqe_win_rate"));
    CHECK_THROWS(non_uqe.GetScalar("action_uqe_win_rate.[x]"));
    CHECK_THROWS(win_info.GetScalar("action_uqe_win_rate.[3]"));
    CHECK_THROWS(non_uqe.GetScalar("action_uqe_margin"));
    CHECK_THROWS(non_uqe.GetScalar("action_uqe_margin.[x]"));
    CHECK_THROWS(win_info.GetScalar("action_uqe_margin.[3]"));
}

TEST_CASE("ActionPolicy spatial tensor generation handles supported scale types", "[dqn][action_policy][spatial]")
{
    auto device = torch::Device(torch::kCPU);

    auto linear = ActionPolicyAccess::CreateSpatialTensor(3, 1.0f, 0.0f, "linear", device);
    CHECK(ShapeOf(linear) == std::vector<int64_t>{ 3 });
    CHECK(torch::allclose(linear, torch::tensor({ 1.0f, 0.5f, 0.0f })));

    auto log = ActionPolicyAccess::CreateSpatialTensor(3, 1.0f, 0.01f, "log", device);
    CHECK(torch::allclose(log, torch::tensor({ 1.0f, 0.1f, 0.01f }), 1.0e-5, 1.0e-5));

    auto clamped = ActionPolicyAccess::CreateSpatialTensor(2, 0.0f, 0.0f, "log", device);
    CHECK(torch::allclose(clamped, torch::tensor({ 1.0e-4f, 1.0e-4f })));

    auto single = ActionPolicyAccess::CreateSpatialTensor(1, 0.25f, 0.75f, "linear", device);
    CHECK(ShapeOf(single) == std::vector<int64_t>{ 1 });
    CHECK(single[0].item<float>() == Catch::Approx(0.25f).margin(1.0e-6f));

    CHECK_THROWS(ActionPolicyAccess::CreateSpatialTensor(2, 1.0f, 0.0f, "invalid", device));
}

TEST_CASE("DefaultDQNAgentConfig keeps spatial exploration train-only", "[dqn][config][spatial]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.train_policy.use_spatial_exploration", "true");
    config_data.Set("DefaultDQNAgent.train_policy.spatial_scale_type", "linear");
    config_data.Set("DefaultDQNAgent.eval_policy.use_spatial_exploration", "true");
    config_data.Set("DefaultDQNAgent.target_policy.use_spatial_exploration", "true");

    dqn::DefaultDQNAgentConfig config(config_data);

    CHECK(config.train_policy.use_spatial_exploration);
    CHECK_FALSE(config.eval_policy.use_spatial_exploration);
    CHECK_FALSE(config.target_policy.use_spatial_exploration);
}

TEST_CASE("DefaultDQNAgentConfig clears optimistic target spatial exploration", "[dqn][config][spatial]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.use_optimistic_target", "true");
    config_data.Set("DefaultDQNAgent.train_policy.policy_type", "UQE");
    config_data.Set("DefaultDQNAgent.train_policy.use_spatial_exploration", "true");
    config_data.Set("DefaultDQNAgent.train_policy.spatial_scale_type", "linear");

    dqn::DefaultDQNAgentConfig config(config_data);

    CHECK(config.train_policy.use_spatial_exploration);
    CHECK_FALSE(config.target_policy.use_spatial_exploration);
    CHECK(config.target_policy.uqe_eps_start == Catch::Approx(0.0f));
    CHECK(config.target_policy.uqe_eps_end == Catch::Approx(0.0f));
}

TEST_CASE("DefaultDQNAgentConfig rejects invalid spatial scale type", "[dqn][config][spatial]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.train_policy.spatial_scale_type", "invalid");

    CHECK_THROWS(dqn::DefaultDQNAgentConfig(config_data));
}

TEST_CASE("DefaultDQNAgentConfig reads and validates TBO settings", "[dqn][config][tbo]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.learner.use_tbo", "true");
    config_data.Set("DefaultDQNAgent.learner.tbo_epsilon", "0.02");
    config_data.Set("DefaultDQNAgent.reward_scaler.use_dynamic_scaling", "false");
    config_data.Set("DefaultDQNAgent.reward_scaler.use_auto_post_scale", "false");

    dqn::DefaultDQNAgentConfig config(config_data);

    CHECK(config.learner.use_tbo);
    CHECK(config.learner.tbo_epsilon == Catch::Approx(0.02f));
}

TEST_CASE("DefaultDQNAgentConfig rejects invalid TBO epsilon", "[dqn][config][tbo]")
{
    for (const auto& value : { "0", "-0.01", "nan", "inf" }) {
        INFO(value);
        anet::ConfigData config_data;
        config_data.Set("DefaultDQNAgent.learner.tbo_epsilon", value);
        CHECK_THROWS(dqn::DefaultDQNAgentConfig(config_data));
    }
}

TEST_CASE("RainbowAgentConfig keeps TBO disabled", "[dqn][config][tbo]")
{
    anet::ConfigData config_data;
    config_data.Set("RainbowAgent.learner.use_tbo", "true");

    dqn::RainbowAgentConfig config(config_data);

    CHECK_FALSE(config.learner.use_tbo);
}

TEST_CASE("DefaultDQNAgentConfig warns when TBO shares reward compression", "[dqn][config][tbo]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.learner.use_tbo", "true");
    config_data.Set("DefaultDQNAgent.reward_scaler.use_dynamic_scaling", "true");
    config_data.Set("DefaultDQNAgent.reward_scaler.use_auto_post_scale", "false");

    anet::test::LogCaptureGuard logs;
    dqn::DefaultDQNAgentConfig config(config_data);
    logs.Flush();

    CHECK(config.learner.use_tbo);
    CHECK(config.reward_scaler.use_dynamic_scaling);
    CHECK_FALSE(config.reward_scaler.use_auto_post_scale);
    bool found_warning = false;
    for (const auto& record : logs.Records()) {
        if (record.message.find("learner.use_tbo") != std::string::npos
            && record.message.find("reward_scaler.use_dynamic_scaling") != std::string::npos
            && record.message.find("double-compressed") != std::string::npos) {
            found_warning = true;
        }
    }
    CHECK(found_warning);
}

TEST_CASE("Spatial exploration keeps scalar metrics as NaN across policy updates", "[dqn][action_policy][spatial]")
{
    dqn::ActionPolicyConfig config;
    config.use_spatial_exploration = true;
    config.spatial_scale_type = "linear";
    config.eps_start = 1.0f;
    config.eps_end = 0.1f;
    config.uqe_eps_start = 0.2f;
    config.uqe_eps_end = 0.0f;
    config.uqe_tau_start = 0.0f;
    config.uqe_tau_end = 1.0f;

    dqn::EpsilonGreedyActionPolicy eps_policy(config, true, 2, torch::Device(torch::kCPU));
    dqn::UQEActionPolicy uqe_policy(config, true, 2, torch::Device(torch::kCPU));

    rl::StepCounts counts;
    counts.exp_step = 1000000;
    eps_policy.OnLearn(counts);
    uqe_policy.OnLearn(counts);

    auto eps = eps_policy.GetScalar("epsilon");
    auto uqe_eps = uqe_policy.GetScalar("epsilon");
    auto uqe_tau = uqe_policy.GetScalar("uqe_tau");
    REQUIRE(eps.has_value());
    REQUIRE(uqe_eps.has_value());
    REQUIRE(uqe_tau.has_value());
    CHECK(std::isnan(*eps));
    CHECK(std::isnan(*uqe_eps));
    CHECK(std::isnan(*uqe_tau));
}

TEST_CASE("Spatial UQE policies use per-env tau tensor", "[dqn][action_policy][spatial]")
{
    auto network = MakePassthroughNetwork(2, 2);
    auto obs = MakeSpatialUQEInput();

    dqn::ActionPolicyConfig config;
    config.use_spatial_exploration = true;
    config.spatial_scale_type = "linear";
    config.uqe_use_tail_mean = false;
    config.uqe_eps_start = 0.0f;
    config.uqe_eps_end = 0.0f;
    config.uqe_tau_start = 0.0f;
    config.uqe_tau_end = 1.0f;

    auto expected_actions = torch::tensor({ 0, 1 }, torch::TensorOptions().dtype(torch::kInt64));

    std::vector<std::pair<std::string, std::shared_ptr<dqn::ActionPolicy>>> policies;
    policies.emplace_back("uqe", std::make_shared<dqn::UQEActionPolicy>(config, true, 2, torch::Device(torch::kCPU)));
    policies.emplace_back("thompson-sampling", std::make_shared<dqn::ThompsonSamplingActionPolicy>(config, true, 2, torch::Device(torch::kCPU)));

    for (const auto& [name, policy] : policies) {
        INFO(name);
        auto rnd = std::make_shared<anet::RandomGenerator>(123);
        auto action_info = policy->SelectAction(obs, /*greedy_only=*/false, network, rnd);
        CHECK(torch::equal(action_info->GetAction().cpu(), expected_actions));
    }
}
