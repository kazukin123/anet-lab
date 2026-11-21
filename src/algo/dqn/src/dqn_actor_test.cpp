#include "anet/test/catch.hpp"
#include "anet/test/nn.hpp"
#include "dqn_fixture_test.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <torch/cuda.h>

TEST_CASE("Skip Actor retains an action and resets duration on episode start", "[dqn][prd098]")
{
    anet::test::RegisterStandardNetworkModules();
    torch::NoGradGuard no_grad;
    const bool cloned = GENERATE(false, true);
    const auto spec = MakeLearnerEnvSpec().state_spec.obs_spec;
    auto network_config = MakeAgentForwardNetworkConfig();
    anet::nn::WeightInitConfig init;
    const auto primary = anet::nn::NetworkBuilder::BuildNetwork(network_config, spec,
        std::make_shared<dqn::LinearHeadFactory>(2, init), 98, torch::kCPU);
    const auto skip = anet::nn::NetworkBuilder::BuildNetwork(network_config, spec,
        std::make_shared<dqn::SkipHeadFactory>(2, 3, anet::rl::dqn::SkipNetworkMode::ActionOut, "none", 1, init), 99, torch::kCPU);
    for (auto& parameter : primary->parameters()) parameter.zero_();
    primary->named_parameters()["head.linear.bias"].copy_(torch::tensor({3.0f, 1.0f}));
    for (auto& parameter : skip->parameters()) parameter.zero_();
    skip->named_parameters()["head.linear.bias"].copy_(torch::tensor({0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f}));
    dqn::ActionPolicyConfig policy_config;
    policy_config.epsilon.value = 0.0f;
    policy_config.epsilon.start = 0.0f;
    policy_config.spatial.epsilon.first.value = 0.0f;
    policy_config.epsilon.end = 0.0f;
    policy_config.spatial.epsilon.last.value = 0.0f;
    const auto policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(policy_config);
    dqn::SkipConfig skip_config;
    skip_config.enabled = true;
    skip_config.max_length = 3;
    const auto actor_primary = cloned ? primary->Clone() : primary;
    const auto actor_skip = cloned ? skip->Clone() : skip;
    dqn::Actor actor(policy, nullptr, nullptr, torch::kCPU, 98, std::make_shared<std::shared_mutex>(),
        actor_primary, primary, anet::nn::NetworkForwardConfig::GetDefault(), true, std::nullopt, false, {},
        dqn::SkipActorResources{.config = skip_config, .policy = policy, .network = actor_skip, .source = skip,
            .forward_config = std::make_shared<const anet::nn::NetworkForwardConfig>(anet::nn::NetworkForwardConfig::GetDefault())});
    const auto flags = torch::zeros({1}, torch::kBool);
    rl::BatchState state(anet::TensorDict{{kVectorKey, torch::ones({1, 2})}}, flags, flags, flags);
    auto first = actor.MakeAction({}, state);
    CHECK(first->GetInfo().At("decision").item<bool>());
    CHECK(first->GetAction().item<int64_t>() == 0);
    CHECK(first->GetAuxData().at("skip_length").item<int64_t>() == 3);
    // primaryの提案を反転させても継続行動0を返し、hintは上書き後のactionから取る。
    primary->named_parameters()["head.linear.bias"].copy_(torch::tensor({1.0f, 4.0f}));
    skip->named_parameters()["head.linear.bias"].copy_(torch::tensor({2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f}));
    const auto continuing = actor.MakeAction({}, state);
    CHECK_FALSE(continuing->GetInfo().At("decision").item<bool>());
    CHECK(continuing->GetAction().item<int64_t>() == 0);
    REQUIRE(continuing->GetReplayInitialPriorityHint());
    CHECK(dqn::DecodeActorQHint(continuing->GetReplayInitialPriorityHint()->GetPayload()).actor_q_sa.item<float>() == (cloned ? 3.0f : 1.0f));
    // Syncはprimaryとskipを同時に更新する。episode境界では両方の新しい値で判断する。
    actor.Sync();
    state.episode_start = torch::ones({1}, torch::kBool);
    const auto reset = actor.MakeAction({}, state);
    CHECK(reset->GetInfo().At("decision").item<bool>());
    CHECK(reset->GetAction().item<int64_t>() == 1);
    CHECK(reset->GetAuxData().at("skip_length").item<int64_t>() == 1);
}

TEST_CASE("Skip Actor session statistics count decision lanes and reset on Sync", "[dqn][prd103][skip_session]")
{
    anet::test::RegisterStandardNetworkModules();
    torch::NoGradGuard no_grad;
    const bool enabled = GENERATE(false, true);
    const auto spec = MakeLearnerEnvSpec().state_spec.obs_spec;
    const auto network_config = MakeAgentForwardNetworkConfig();
    const anet::nn::WeightInitConfig init;
    const auto primary = anet::nn::NetworkBuilder::BuildNetwork(network_config, spec,
        std::make_shared<dqn::LinearHeadFactory>(1, init), 98, torch::kCPU);
    const auto skip = anet::nn::NetworkBuilder::BuildNetwork(network_config, spec,
        std::make_shared<dqn::SkipHeadFactory>(1, 3, dqn::SkipNetworkMode::ActionOut, "none", 1, init), 99, torch::kCPU);
    for (auto& parameter : skip->parameters()) parameter.zero_();
    skip->named_parameters()["head.linear.bias"].copy_(torch::tensor({0.0f, 0.0f, 1.0f}));
    dqn::ActionPolicyConfig policy_config;
    policy_config.epsilon.value = policy_config.epsilon.start = policy_config.epsilon.end = 0;
    const auto policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(policy_config);
    std::optional<dqn::SkipActorResources> resources;
    if (enabled) resources = dqn::SkipActorResources{.config = {.enabled = true, .max_length = 3},
        .policy = policy, .network = skip, .source = skip,
        .forward_config = std::make_shared<const anet::nn::NetworkForwardConfig>(anet::nn::NetworkForwardConfig::GetDefault())};
    dqn::Actor actor(policy, nullptr, nullptr, torch::kCPU, 98, std::make_shared<std::shared_mutex>(),
        primary, primary, anet::nn::NetworkForwardConfig::GetDefault(), false, std::nullopt, false, {}, resources);
    for (const auto* key : {"skip_session_decision_rate", "skip_session_length_mean", "skip_session_length_rate.[1]"}) {
        const auto value = actor.GetScalar(key);
        REQUIRE(value.has_value());
        CHECK(std::isnan(*value));
    }
    CHECK_FALSE(actor.GetScalar("unknown_session_metric").has_value());
    const auto flags = torch::zeros({2}, torch::kBool);
    rl::BatchState state(anet::TensorDict{{kVectorKey, torch::ones({2, 2})}}, flags, flags, flags);
    actor.MakeAction({}, state);
    if (!enabled) {
        CHECK(std::isnan(*actor.GetScalar("skip_session_decision_rate")));
        return;
    }
    CHECK(*actor.GetScalar("skip_session_decision_rate") == 1.0f);
    state.episode_start = torch::tensor({false, true});
    actor.MakeAction({}, state);
    CHECK(*actor.GetScalar("skip_session_decision_rate") == 0.75f);
    state.episode_start = flags;
    actor.MakeAction({}, state);
    CHECK(*actor.GetScalar("skip_session_decision_rate") == 0.5f);
    CHECK(*actor.GetScalar("skip_session_length_mean") == 3.0f);
    CHECK(*actor.GetScalar("skip_session_length_rate.[3]") == 1.0f);
    CHECK(*actor.GetScalar("skip_session_length_rate.[1]") == 0.0f);
    CHECK_THROWS(actor.GetScalar("skip_session_length_rate.[0]"));
    CHECK_THROWS(actor.GetScalar("skip_session_length_rate.[4]"));
    actor.Sync();
    CHECK(std::isnan(*actor.GetScalar("skip_session_length_mean")));
    skip->named_parameters()["head.linear.bias"].copy_(torch::tensor({1.0f, 0.0f, 0.0f}));
    actor.MakeAction({}, state);
    CHECK(*actor.GetScalar("skip_session_decision_rate") == 0.5f);
    CHECK(*actor.GetScalar("skip_session_length_mean") == 1.0f);
    CHECK(*actor.GetScalar("skip_session_length_rate.[1]") == 1.0f);
}

TEST_CASE("Dedicated IQN skip policy queries its own mean or risk range in one shared forward", "[dqn][prd098][skip_policy]")
{
    // primaryは0.9分位、skipは平均または0.2分位。primaryの分位だけでは長さが逆転する。
    const auto kind = GENERATE(std::string("EpsilonGreedy"), std::string("UQE"), std::string("ThompsonSampling"));
    const bool spatial = GENERATE(false, true);
    const bool single = GENERATE(false, true);
    auto primary_state = std::make_shared<TauEchoState>();
    auto skip_state = std::make_shared<TauEchoState>();
    auto primary = MakeTauEchoNetwork(8, primary_state);
    auto skip = MakeTauEchoNetwork(8, skip_state, true);
    dqn::ActionPolicyConfig primary_config;
    primary_config.policy_type = "UQE";
    primary_config.quantile_mode = "iqn";
    primary_config.uqe.tau.value = 0.9f;
    primary_config.uqe.tau.start = 0.9f;
    primary_config.spatial.uqe.tau.first.value = 0.9f;
    primary_config.uqe.tau.end = 0.9f;
    primary_config.spatial.uqe.tau.last.value = 0.9f;
    primary_config.uqe.use_tail_mean = false;
    primary_config.uqe.epsilon.value = 0;
    primary_config.uqe.epsilon.start = 0;
    primary_config.spatial.uqe.epsilon.first.value = 0;
    primary_config.uqe.epsilon.end = 0;
    primary_config.spatial.uqe.epsilon.last.value = 0;
    primary_config.tau_rule.num_taus = 8;
    primary_config.tau_rule.sample_mode = "fixed";
    auto policy = std::make_shared<dqn::UQEActionPolicy>(primary_config);
    auto dedicated_config = primary_config;
    dedicated_config.policy_type = kind;
    dedicated_config.epsilon.value = 0;
    dedicated_config.epsilon.start = 0;
    dedicated_config.spatial.epsilon.first.value = 0;
    dedicated_config.epsilon.end = 0;
    dedicated_config.spatial.epsilon.last.value = 0;
    dedicated_config.uqe.tau.value = 0.2f;
    dedicated_config.uqe.tau.start = 0.2f;
    dedicated_config.spatial.uqe.tau.first.value = 0.2f;
    dedicated_config.uqe.tau.end = spatial ? 0.9f : 0.2f;
    dedicated_config.spatial.uqe.tau.last.value = spatial ? 0.9f : 0.2f;
    std::shared_ptr<dqn::ActionPolicy> dedicated;
    if (kind == "EpsilonGreedy") dedicated = std::make_shared<dqn::EpsilonGreedyActionPolicy>(dedicated_config, spatial, 2);
    else if (kind == "UQE") dedicated = std::make_shared<dqn::UQEActionPolicy>(dedicated_config, spatial, 2);
    else dedicated = std::make_shared<dqn::ThompsonSamplingActionPolicy>(dedicated_config, spatial, 2);
    dqn::Actor actor(policy, nullptr, nullptr, torch::kCPU, 98, std::make_shared<std::shared_mutex>(),
        primary, primary, anet::nn::NetworkForwardConfig::GetDefault(), false, std::nullopt, false, {},
        dqn::SkipActorResources{.config = {.enabled = true, .max_length = 2, .network_mode = dqn::SkipNetworkMode::ActionIn},
        .policy = dedicated, .network = skip, .source = skip,
        .forward_config = std::make_shared<const anet::nn::NetworkForwardConfig>(anet::nn::NetworkForwardConfig::GetDefault()), .shared_policy = false,
        .max_length_mode = single ? dqn::DQNActorConfig::kMaxLengthSingle : dqn::DQNActorConfig::kMaxLengthFull});
    const auto flags = torch::zeros({2}, torch::kBool);
    const auto result = actor.MakeAction({}, rl::BatchState(MakeTauEchoObservation(), flags, flags, flags));
    CHECK(torch::equal(result->GetAction(), torch::zeros({2}, torch::kLong)));
    CHECK(torch::allclose(result->GetAuxData().at("q_values").select(1, 0), torch::full({2}, 0.9f)));
    const auto expected = single ? torch::ones({2}, torch::kLong)
        : torch::tensor({spatial && kind != "EpsilonGreedy" ? 1 : 2, 2}, torch::kLong);
    CHECK(torch::equal(result->GetAuxData().at("skip_length"), expected));
    CHECK(primary_state->forward_count == 1);
    CHECK(skip_state->forward_count == 1);
    CHECK(torch::equal(primary_state->last_taus, skip_state->last_taus));
    CHECK(primary_state->last_taus.size(1) > 8);
}

TEST_CASE("Skip forward samples only its execution config with fixed observation features and action", "[dqn][prd106][prd106-fixed-input]")
{
    anet::test::RegisterStandardNetworkModules();
    const bool sample = GENERATE(false, true);
    anet::ConfigData projection;
    projection.Set("noisy.enabled", true);
    const anet::nn::WeightInitConfig init;
    const auto primary = anet::nn::NetworkBuilder::BuildNetwork(MakeAgentForwardNetworkConfig(),
        MakeLearnerEnvSpec().state_spec.obs_spec, std::make_shared<dqn::LinearHeadFactory>(2, init, projection), 106, torch::kCPU);
    anet::ConfigData body;
    body.Set("test.branch.[main].bind", anet::nn::kKey_DefaultOutput);
    body.Set("test.body.output.[features]", "main");
    const auto feature_spec = MakeLearnerEnvSpec().state_spec.obs_spec.at(kVectorKey);
    const auto skip = anet::nn::NetworkBuilder::BuildNetwork(anet::nn::NetworkConfig(body, "test"),
        anet::TensorSpecMap{{anet::nn::kKey_DefaultOutput, feature_spec}},
        std::make_shared<dqn::SkipHeadFactory>(2, 3, dqn::SkipNetworkMode::ActionOut, "none", 1, init, projection), 107, torch::kCPU);
    anet::ConfigData execution;
    execution.Set("noisy.mode", sample ? "sample" : "mu_only");
    execution.Set("noisy.hold", "call");
    execution.Set("noisy.share", "batch");
    const auto policy = std::make_shared<FixedSkipInputPolicy>(torch::ones({2, 2}));
    dqn::Actor actor(policy, nullptr, nullptr, torch::kCPU, 106, std::make_shared<std::shared_mutex>(), primary, primary,
        anet::nn::NetworkForwardConfig::GetDefault(), false, std::nullopt, false, {},
        dqn::SkipActorResources{.config = {.enabled = true, .max_length = 3}, .policy = policy, .network = skip, .source = skip,
            .forward_config = std::make_shared<const anet::nn::NetworkForwardConfig>("fixed_skip", execution), .max_length_mode = "single"});
    const auto flags = torch::zeros({2}, torch::kBool);
    const rl::BatchState state(anet::TensorDict{{kVectorKey, torch::ones({2, 2})}}, flags, flags, flags);
    const auto first = actor.MakeAction({}, state);
    const auto second = actor.MakeAction({}, state);
    CHECK(torch::equal(first->GetAuxData().at(dqn::DQNActionInfo::kSkipQValues), second->GetAuxData().at(dqn::DQNActionInfo::kSkipQValues)) == !sample);
}

TEST_CASE("Skip target windows validate the entire Actor catalog and warn for intentional controls", "[dqn][prd104][prd104-validation]")
{
    const bool enabled = GENERATE(false, true);
    const bool learning = GENERATE(false, true);
    const auto window = GENERATE(std::string("decision"), std::string("action_change"), std::string("episode_end"));
    const auto width = GENERATE(std::string("single"), std::string("full"), std::string("unused_single"),
        std::string("unused_full"), std::string("variable_one"), std::string("variable_two"),
        std::string("linear_one"), std::string("linear_two"), std::string("phased_one"), std::string("phased_two"));
    const auto gradient = GENERATE(std::string("merge"), std::string("stop"));
    CAPTURE(enabled, learning, window, width, gradient);
    auto data = MakePrd104ConfigData();
    data.Set("DefaultDQNAgent.skip.enabled", enabled);
    data.Set("DefaultDQNAgent.learner.enabled", learning);
    data.Set("DefaultDQNAgent.learner.skip.target_window", window);
    data.Set("DefaultDQNAgent.learner.skip.feature_gradient", gradient);
    const bool unused = width.starts_with("unused_");
    const bool variable = !unused && width != "single" && width != "full";
    const bool exceeds_one = width == "full" || width == "unused_full" || width.ends_with("_two");
    for (const auto* actor_key : {"train", "eval"}) {
        const auto prefix = std::string("DefaultDQNAgent.actor.[") + actor_key + "].skip.";
        const auto mode = variable ? "variable" : width.ends_with("single") ? "single" : "full";
        data.Set(prefix + "max_length_mode", mode);
        if (unused || variable) {
            const int value = width.ends_with("_two") ? 2 : 1;
            if (width.starts_with("linear_")) {
                data.Set(prefix + "max_length.type", "linear");
                data.Set(prefix + "max_length.start", 1);
                data.Set(prefix + "max_length.end", value);
                data.Set(prefix + "max_length.steps", 2);
            } else if (width.starts_with("phased_")) {
                data.Set(prefix + "max_length.type", "phased");
                data.Set(prefix + "max_length.phases", "p");
                data.Set(prefix + "max_length.phase.[p].value", value);
                data.Set(prefix + "max_length.phase.[p].steps", 2);
                data.Set(prefix + "max_length.start", 100);
            } else {
                data.Set(prefix + "max_length", value);
                data.Set(prefix + "max_length.start", 100);
            }
        }
    }
    anet::test::LogCaptureGuard logs;
    const bool invalid = enabled && learning && window != "decision" && exceeds_one;
    if (invalid) {
        std::string diagnostic;
        try { (void)dqn::DefaultDQNAgentConfig(data); }
        catch (const std::exception& error) { diagnostic = error.what(); }
        CHECK_FALSE(diagnostic.empty());
        CHECK(diagnostic.find("actor.[train]") != std::string::npos);
        CHECK(diagnostic.find("actor.[eval]") != std::string::npos);
        CHECK(diagnostic.find("target_window") != std::string::npos);
        logs.Flush();
        for (const auto& record : logs.Records()) CHECK(record.message.find("ProfiledValue:") == std::string::npos);
    } else {
        REQUIRE_NOTHROW(dqn::DefaultDQNAgentConfig(data));
        REQUIRE_NOTHROW(dqn::DefaultDQNAgentConfig(data));
        logs.Flush();
        int unused_warnings = 0, control_warnings = 0;
        for (const auto& record : logs.Records()) {
            unused_warnings += record.level == wxLOG_Warning
                && record.message.find(".skip.max_length") != std::string::npos;
            control_warnings += record.level == wxLOG_Message
                && record.message.find("learner.skip.feature_gradient=stop") != std::string::npos;
        }
        CHECK(unused_warnings == (enabled && unused ? 2 : 0));
        for (const auto* actor : {"train", "eval"}) {
            const auto key = std::string("DefaultDQNAgent.actor.[") + actor + "].skip.max_length";
            CHECK(std::count_if(logs.Records().begin(), logs.Records().end(), [&](const auto& record) {
                return record.level == wxLOG_Warning && record.message.find(key) != std::string::npos;
            }) == (enabled && unused ? 1 : 0));
        }
        CHECK(control_warnings == (enabled && learning && window != "decision" && gradient == "stop" ? 1 : 0));
    }
}

TEST_CASE("Variable Actor preserves a held action when its scheduled limit drops", "[dqn][prd104][prd104-held]")
{
    anet::test::RegisterStandardNetworkModules();
    torch::NoGradGuard no_grad;
    const auto spec = MakeLearnerEnvSpec().state_spec.obs_spec;
    const auto network_config = MakeAgentForwardNetworkConfig();
    const anet::nn::WeightInitConfig init;
    const auto primary = anet::nn::NetworkBuilder::BuildNetwork(network_config, spec,
        std::make_shared<dqn::LinearHeadFactory>(2, init), 104, torch::kCPU);
    const auto skip = anet::nn::NetworkBuilder::BuildNetwork(network_config, spec,
        std::make_shared<dqn::SkipHeadFactory>(2, 3, dqn::SkipNetworkMode::ActionOut, "none", 1, init), 105, torch::kCPU);
    for (auto& p : primary->parameters()) p.zero_();
    for (auto& p : skip->parameters()) p.zero_();
    primary->named_parameters()["head.linear.bias"].copy_(torch::tensor({3.0f, 1.0f}));
    skip->named_parameters()["head.linear.bias"].copy_(torch::tensor({0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 2.0f}));
    dqn::ActionPolicyConfig policy_config;
    policy_config.epsilon.value = policy_config.epsilon.start = policy_config.epsilon.end = 0;
    const auto policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(policy_config);
    dqn::Actor actor(policy, nullptr, nullptr, torch::kCPU, 104, std::make_shared<std::shared_mutex>(),
        primary, primary, anet::nn::NetworkForwardConfig::GetDefault(), false, std::nullopt, false, {},
        dqn::SkipActorResources{.config = {.enabled = true, .max_length = 3}, .policy = policy,
            .network = skip, .source = skip,
            .forward_config = std::make_shared<const anet::nn::NetworkForwardConfig>(anet::nn::NetworkForwardConfig::GetDefault()), .max_length_mode = "variable",
            .max_length = anet::ProfiledValueConfig<int>{.type = "linear", .start = 3, .end = 1, .steps = 1}});
    const auto flags = torch::zeros({2}, torch::kBool);
    rl::BatchState state(anet::TensorDict{{kVectorKey, torch::ones({2, 2})}}, flags, flags, flags);
    const auto first = actor.MakeAction({}, state);
    CHECK(torch::equal(first->GetAuxData().at("skip_length"), torch::full({2}, 3, torch::kLong)));
    // cが1へ落ちてもlane0の3stepは続け、episodeが始まるlane1だけは新しく判断する。
    primary->named_parameters()["head.linear.bias"].copy_(torch::tensor({1.0f, 4.0f}));
    state.episode_start = torch::tensor({false, true});
    const auto second = actor.MakeAction({.exp_step = 1}, state);
    CHECK(*actor.GetScalar("skip_max_length") == 1.0f);
    CHECK(torch::equal(second->GetAction(), torch::tensor({0, 1}, torch::kLong)));
    CHECK(torch::equal(second->GetAuxData().at("skip_length"), torch::tensor({3, 1}, torch::kLong)));
    CHECK(torch::equal(second->GetInfo().At("decision"), torch::tensor({false, true})));
    state.episode_start = flags;
    const auto third = actor.MakeAction({.exp_step = 2}, state);
    CHECK(third->GetAction()[0].item<int>() == 0);
    const auto fourth = actor.MakeAction({.exp_step = 3}, state);
    CHECK(torch::equal(fourth->GetAction(), torch::ones({2}, torch::kLong)));
    CHECK(torch::equal(fourth->GetAuxData().at("skip_length"), torch::ones({2}, torch::kLong)));
}

TEST_CASE("Variable IQN skip selection keeps full auxiliary values for each policy", "[dqn][prd104][prd104-distribution]")
{
    const auto kind = GENERATE(std::string("EpsilonGreedy"), std::string("UQE"), std::string("ThompsonSampling"));
    auto primary_state = std::make_shared<TauEchoState>();
    auto skip_state = std::make_shared<TauEchoState>();
    auto primary = MakeTauEchoNetwork(8, primary_state);
    auto skip = MakeTauEchoNetwork(8, skip_state, true);
    dqn::ActionPolicyConfig config;
    config.policy_type = kind;
    config.quantile_mode = "iqn";
    config.epsilon.value = config.epsilon.start = config.epsilon.end = 0;
    config.uqe.epsilon.value = config.uqe.epsilon.start = config.uqe.epsilon.end = 0;
    config.tau_rule.num_taus = 8;
    config.tau_rule.sample_mode = "fixed";
    std::shared_ptr<dqn::ActionPolicy> policy;
    if (kind == "EpsilonGreedy") policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(config, false, 2);
    else if (kind == "UQE") policy = std::make_shared<dqn::UQEActionPolicy>(config, false, 2);
    else policy = std::make_shared<dqn::ThompsonSamplingActionPolicy>(config, false, 2);
    dqn::Actor actor(policy, nullptr, nullptr, torch::kCPU, 104, std::make_shared<std::shared_mutex>(),
        primary, primary, anet::nn::NetworkForwardConfig::GetDefault(), false, std::nullopt, false, {},
        dqn::SkipActorResources{.config = {.enabled = true, .max_length = 2, .network_mode = dqn::SkipNetworkMode::ActionIn},
            .policy = policy, .network = skip, .source = skip,
            .forward_config = std::make_shared<const anet::nn::NetworkForwardConfig>(anet::nn::NetworkForwardConfig::GetDefault()), .max_length_mode = "variable",
            .max_length = anet::ProfiledValueConfig<int>{.value = 1}});
    const auto flags = torch::zeros({2}, torch::kBool);
    const auto action = actor.MakeAction({}, rl::BatchState(MakeTauEchoObservation(), flags, flags, flags));
    CHECK(torch::equal(action->GetAuxData().at("skip_length"), torch::ones({2}, torch::kLong)));
    CHECK(action->GetAuxData().at("skip_q_values").sizes().vec() == std::vector<int64_t>{2, 2});
    CHECK(primary_state->forward_count == 1);
    CHECK(skip_state->forward_count == 1);
    CHECK(torch::equal(primary_state->last_taus, skip_state->last_taus));
}

TEST_CASE("Skip action scalars distinguish decisions, durations and inactive values", "[dqn][prd098][prd098-r2-qtrace]")
{
    const auto actions = torch::zeros({3}, torch::kInt64);
    dqn::DQNActionInfo info(actions, anet::TensorDict{}, rl::AuxData{
        {"skip_decision", torch::tensor({true, false, true})},
        {"skip_length", torch::tensor({1, 2, 3}, torch::kInt64)},
        {"skip_q_values", torch::tensor({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}})},
    });
    REQUIRE(info.GetScalar("skip_decision_rate"));
    CHECK(*info.GetScalar("skip_decision_rate") == Catch::Approx(2.0f / 3.0f));
    CHECK(*info.GetScalar("skip_length_mean") == 2.0f);
    CHECK(*info.GetScalar("skip_length_rate.[1]") == 0.5f);
    CHECK(*info.GetScalar("skip_length_rate.[2]") == 0.0f);
    CHECK(*info.GetScalar("skip_length_rate.[3]") == 0.5f);
    CHECK(*info.GetScalar("skip_q_gap") == 1.0f);
    // 同じcacheからlaneごとのQ列を読み、jとlaneの境界を検証する。
    for (int64_t lane = 0; lane < 3; ++lane) {
        for (int64_t j = 1; j <= 3; ++j) {
            const auto value = info.GetScalar("skip_q.[" + std::to_string(j) + "]", lane);
            REQUIRE(value.has_value());
            CHECK(*value == Catch::Approx(lane * 3.0f + j));
        }
    }
    CHECK_THROWS(info.GetScalar("skip_q.[0]", 0));
    CHECK_THROWS(info.GetScalar("skip_q.[4]", 0));
    CHECK_THROWS(info.GetScalar("skip_q.[1]", 3));

    CHECK(*info.GetScalar("skip_decision", 1) == 0.0f);
    CHECK(*info.GetScalar("skip_length", 1) == 2.0f);
    CHECK_THROWS(info.GetScalar("skip_length_rate.[0]"));
    CHECK_THROWS(info.GetScalar("skip_length_rate.[4]"));
    CHECK_THROWS(info.GetScalar("skip_length", 3));
    auto continuing_aux = info.GetAuxData();
    continuing_aux["skip_decision"] = torch::zeros({3}, torch::kBool);
    const dqn::DQNActionInfo continuing(actions, anet::TensorDict{}, continuing_aux);
    CHECK(std::isnan(*continuing.GetScalar("skip_length_mean")));
    dqn::DQNActionInfo inactive(actions);
    REQUIRE(inactive.GetScalar("skip_decision_rate"));
    CHECK(std::isnan(*inactive.GetScalar("skip_decision_rate")));
    REQUIRE(inactive.GetScalar("skip_q.[1]", 0).has_value());
    CHECK(std::isnan(*inactive.GetScalar("skip_q.[1]", 0)));
    CHECK_FALSE(inactive.GetScalar("skip_unknown"));
    anet::ConfigData columns;
    columns.Set("app.eval_trace_panel.columns.[1]", "$action_info skip_decision");
    columns.Set("app.eval_trace_panel.columns.[2]", "$action_info skip_length");
    for (int64_t j = 1; j <= 3; ++j)
        columns.Set("app.eval_trace_panel.columns.[" + std::to_string(2+j) + "]", "$action_info skip_q.[" + std::to_string(j) + "]");
    rl::StepTraceBuilder trace(columns, MakeLearnerEnvSpec(), nullptr, 3, nullptr);
    std::vector<std::shared_ptr<rl::StepTrace>> rows;
    trace.SetHandlers([&](const auto& row, const rl::TrainEvent&) { rows.push_back(row); }, {});
    const auto flags = torch::zeros({3}, torch::kBool);
    const rl::BatchState state(anet::TensorDict{{kVectorKey, torch::zeros({3, 2})}}, flags, flags, flags);
    const auto action = std::make_shared<dqn::DQNActionInfo>(info);
    const auto step_result = std::make_shared<rl::PlainBatchStepResult>(torch::ones({3}), state, state,
        1, 0, std::vector<rl::AuxData>(3));
    const rl::BatchExperience experience{state, action, step_result->reward, state};
    trace.OnTrain(rl::TrainEvent{{experience, nullptr, {}, nullptr, {}}, nullptr, step_result, action});
    REQUIRE(rows.size() == 3);
    CHECK(rows[1]->values == std::vector<float>{0.0f, 2.0f, 4.0f, 5.0f, 6.0f});
    CHECK(rows[2]->values == std::vector<float>{1.0f, 3.0f, 7.0f, 8.0f, 9.0f});
}

TEST_CASE("Actor sync leaves cloned network in eval mode", "[dqn][actor]")
{
    auto src_network = MakeLinearNetwork();
    auto clone_network = MakeLinearNetwork();
    auto mutex = std::make_shared<std::shared_mutex>();
    dqn::Actor actor(nullptr, nullptr, nullptr, torch::kCPU, std::nullopt, mutex, clone_network, src_network, anet::nn::NetworkForwardConfig::GetDefault());

    src_network->eval();
    clone_network->train();

    actor.Sync();

    CHECK_FALSE(src_network->is_training());
    CHECK_FALSE(clone_network->is_training());
}

TEST_CASE("DQN Actor keeps a Train network snapshot until the sync interval", "[dqn][actor][snapshot]")
{
    auto src_network = MakeLinearNetwork();
    {
        torch::NoGradGuard no_grad;
        src_network->parameters()[0].fill_(1.0f);
    }
    auto snapshot_network = MakeLinearNetwork();
    {
        torch::NoGradGuard no_grad;
        snapshot_network->parameters()[0].fill_(1.0f);
    }
    auto policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{});
    auto mutex = std::make_shared<std::shared_mutex>();
    anet::ProfiledValueConfig<rl::step_t> sync_interval;
    sync_interval.value = 2;
    dqn::Actor actor(
        policy, nullptr, nullptr, torch::kCPU, 123, mutex, snapshot_network, src_network, anet::nn::NetworkForwardConfig::GetDefault(), false, sync_interval, true);

    {
        torch::NoGradGuard no_grad;
        src_network->parameters()[0].fill_(2.0f);
    }
    auto flags = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kBool));
    rl::BatchState state(
        anet::TensorDict{ { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }) } },
        flags,
        flags,
        flags);
    auto make_action = [&](rl::step_t train_step) {
        rl::StepCounts step;
        step.train_step = train_step;
        step.exp_step = train_step;
        return actor.MakeAction(step, state);
    };

    const auto action0 = make_action(0);
    const auto action1 = make_action(1);
    const auto action2 = make_action(2);
    CHECK(action0->GetAuxData().at("q_values").item<float>() == Catch::Approx(3.0f));
    CHECK(action1->GetAuxData().at("q_values").item<float>() == Catch::Approx(3.0f));
    CHECK(action2->GetAuxData().at("q_values").item<float>() == Catch::Approx(6.0f));

    const auto metrics0 = std::dynamic_pointer_cast<dqn::DQNActionInfo>(action0);
    const auto metrics1 = std::dynamic_pointer_cast<dqn::DQNActionInfo>(action1);
    const auto metrics2 = std::dynamic_pointer_cast<dqn::DQNActionInfo>(action2);
    REQUIRE(metrics0 != nullptr);
    REQUIRE(metrics1 != nullptr);
    REQUIRE(metrics2 != nullptr);
    CHECK(metrics0->GetScalar("train_actor_snapshot_interval") == 2.0f);
    CHECK(metrics0->GetScalar("train_actor_snapshot_age") == 0.0f);
    CHECK(metrics1->GetScalar("train_actor_snapshot_age") == 1.0f);
    CHECK(metrics2->GetScalar("train_actor_snapshot_age") == 0.0f);

    const auto moved = std::dynamic_pointer_cast<dqn::DQNActionInfo>(metrics2->To(torch::kCPU));
    const auto replaced = std::dynamic_pointer_cast<dqn::DQNActionInfo>(
        metrics2->WithAction(torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kInt64))));
    REQUIRE(moved != nullptr);
    REQUIRE(replaced != nullptr);
    CHECK(moved->GetScalar("train_actor_snapshot_interval") == 2.0f);
    CHECK(replaced->GetScalar("train_actor_snapshot_age") == 0.0f);
}

TEST_CASE("DQN Actor forced Sync resets snapshot age without a duplicate copy", "[dqn][actor][snapshot]")
{
    auto src_network = MakeLinearNetwork();
    auto snapshot_network = MakeLinearNetwork();
    {
        torch::NoGradGuard no_grad;
        src_network->parameters()[0].fill_(1.0f);
        snapshot_network->parameters()[0].fill_(1.0f);
    }
    auto policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{});
    auto mutex = std::make_shared<std::shared_mutex>();
    anet::ProfiledValueConfig<rl::step_t> sync_interval;
    sync_interval.value = 5;
    dqn::Actor actor(
        policy, nullptr, nullptr, torch::kCPU, 123, mutex, snapshot_network, src_network, anet::nn::NetworkForwardConfig::GetDefault(), false, sync_interval, true);

    {
        torch::NoGradGuard no_grad;
        src_network->parameters()[0].fill_(2.0f);
    }
    actor.Sync();
    {
        torch::NoGradGuard no_grad;
        src_network->parameters()[0].fill_(3.0f);
    }

    auto flags = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kBool));
    rl::BatchState state(
        anet::TensorDict{ { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }) } },
        flags,
        flags,
        flags);
    rl::StepCounts step;
    step.train_step = 3;
    step.exp_step = 3;
    auto action_info = std::dynamic_pointer_cast<dqn::DQNActionInfo>(actor.MakeAction(step, state));

    REQUIRE(action_info != nullptr);
    CHECK(action_info->GetAuxData().at("q_values").item<float>() == Catch::Approx(6.0f));
    CHECK(action_info->GetScalar("train_actor_snapshot_age") == 0.0f);
}

TEST_CASE("DQN Actor applies snapshot interval shortening and extension at action boundaries", "[dqn][actor][snapshot]")
{
    const auto run_profile = [](const anet::ProfiledValueConfig<rl::step_t>& sync_interval,
                                const std::vector<rl::StepCounts>& steps) {
        auto src_network = MakeLinearNetwork();
        auto snapshot_network = MakeLinearNetwork();
        {
            torch::NoGradGuard no_grad;
            src_network->parameters()[0].fill_(1.0f);
            snapshot_network->parameters()[0].fill_(1.0f);
        }
        auto policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{});
        auto mutex = std::make_shared<std::shared_mutex>();
        dqn::Actor actor(
            policy, nullptr, nullptr, torch::kCPU, 123, mutex, snapshot_network, src_network, anet::nn::NetworkForwardConfig::GetDefault(), false, sync_interval, true);
        {
            torch::NoGradGuard no_grad;
            src_network->parameters()[0].fill_(2.0f);
        }
        auto flags = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kBool));
        rl::BatchState state(
            anet::TensorDict{ { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }) } },
            flags,
            flags,
            flags);

        std::vector<std::shared_ptr<dqn::DQNActionInfo>> results;
        for (const auto& step : steps) {
            results.push_back(std::dynamic_pointer_cast<dqn::DQNActionInfo>(actor.MakeAction(step, state)));
        }
        return results;
    };

    anet::ProfiledValueConfig<rl::step_t> shortening{
        .type = "linear",
        .start = 4,
        .end = 2,
        .steps = 2,
    };
    auto shortened = run_profile(shortening, {
        rl::StepCounts{ .train_step = 1, .exp_step = 1 },
        rl::StepCounts{ .train_step = 2, .exp_step = 2 },
    });
    REQUIRE(shortened[0] != nullptr);
    REQUIRE(shortened[1] != nullptr);
    CHECK(shortened[0]->GetAuxData().at("q_values").item<float>() == Catch::Approx(3.0f));
    CHECK(shortened[1]->GetAuxData().at("q_values").item<float>() == Catch::Approx(6.0f));

    anet::ProfiledValueConfig<rl::step_t> extension{
        .type = "linear",
        .start = 2,
        .end = 4,
        .steps = 2,
    };
    auto extended = run_profile(extension, {
        rl::StepCounts{ .train_step = 2, .exp_step = 2 },
        rl::StepCounts{ .train_step = 4, .exp_step = 2 },
    });
    REQUIRE(extended[0] != nullptr);
    REQUIRE(extended[1] != nullptr);
    CHECK(extended[0]->GetAuxData().at("q_values").item<float>() == Catch::Approx(3.0f));
    CHECK(extended[0]->GetScalar("train_actor_snapshot_age") == 2.0f);
    CHECK(extended[1]->GetAuxData().at("q_values").item<float>() == Catch::Approx(6.0f));
}

TEST_CASE("DQN Actor exposes NaN snapshot metrics when periodic sync is disabled", "[dqn][actor][snapshot]")
{
    auto network = MakeLinearNetwork();
    auto policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{});
    auto mutex = std::make_shared<std::shared_mutex>();
    dqn::Actor actor(policy, nullptr, nullptr, torch::kCPU, 123, mutex, network, network, anet::nn::NetworkForwardConfig::GetDefault(), false, std::nullopt, true);
    auto flags = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kBool));
    rl::BatchState state(
        anet::TensorDict{ { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }) } },
        flags,
        flags,
        flags);

    auto action_info = std::dynamic_pointer_cast<dqn::DQNActionInfo>(
        actor.MakeAction(rl::StepCounts{}, state));

    REQUIRE(action_info != nullptr);
    REQUIRE(action_info->GetScalar("train_actor_snapshot_interval").has_value());
    REQUIRE(action_info->GetScalar("train_actor_snapshot_age").has_value());
    CHECK(std::isnan(*action_info->GetScalar("train_actor_snapshot_interval")));
    CHECK(std::isnan(*action_info->GetScalar("train_actor_snapshot_age")));
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
        const anet::nn::NetworkForwardCallInfo call_info_0({});
        auto action_info = policy->SelectAction(obs, /*greedy_only=*/true, *MakeDefaultExecutor(network), rnd, call_info_0);

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

TEST_CASE("QR policy exposes quantile tail diagnostics through action info", "[dqn][qr][action_policy][metrics]")
{
    auto network = MakePassthroughNetwork(2, 4);
    const auto q_quantiles = torch::tensor({
        {
            { 0.0f, 1.0f, 2.0f, 3.0f },
            { 0.0f, 0.0f, 0.0f, 4.0f },
        },
        {
            { 0.0f, 0.0f, 0.0f, 0.0f },
            { -4.0f, 0.0f, 0.0f, 0.0f },
        },
    });
    const auto obs = anet::TensorDict{
        { "q", q_quantiles.mean(2) },
        { "q_dist", q_quantiles },
    };

    dqn::ActionPolicyConfig config;
    config.quantile_mode = "qr";
    dqn::EpsilonGreedyActionPolicy policy(config);
    auto rnd = std::make_shared<anet::RandomGenerator>(123);
    const anet::nn::NetworkForwardCallInfo call_info_1({});
    auto action_info = policy.SelectAction(obs, /*greedy_only=*/true, *MakeDefaultExecutor(network), rnd, call_info_1, {});

    const float selected_tail_mean = std::sqrt(1.25f) / 2.0f;
    const auto crossing_depth_p90 =
        action_info->GetScalar("policy_selected_crossing_depth_p90_ratio");
    const auto upper = action_info->GetScalar("policy_upper_truncated_std");
    const auto lower = action_info->GetScalar("policy_lower_truncated_std");
    const auto disagreement = action_info->GetScalar("lower_risk_full_q_argmax_disagreement");
    const auto crossing = action_info->GetScalar("quantile_crossing_ratio");
    REQUIRE(crossing_depth_p90.has_value());
    REQUIRE(upper.has_value());
    REQUIRE(lower.has_value());
    REQUIRE(disagreement.has_value());
    REQUIRE(crossing.has_value());
    CHECK(*upper == Catch::Approx(selected_tail_mean));
    CHECK(*lower == Catch::Approx(selected_tail_mean));
    CHECK(*disagreement == Catch::Approx(0.5f));
    CHECK(*crossing == Catch::Approx(0.0f));
    CHECK(*crossing_depth_p90 == Catch::Approx(0.0f));
    CHECK(*action_info->GetScalar("policy_selected_crossing_depth_p90_ratio")
        == Catch::Approx(0.0f));

    // action差し替え後は上下幅だけを最終actionから再集約し、global診断は維持する。
    auto replaced = std::dynamic_pointer_cast<dqn::DQNActionInfo>(action_info->WithAction(torch::ones(
        { 2 }, torch::TensorOptions().dtype(torch::kInt64))));
    REQUIRE(replaced != nullptr);
    CHECK(*replaced->GetScalar("policy_upper_truncated_std") == Catch::Approx(std::sqrt(2.0f)));
    CHECK(*replaced->GetScalar("policy_lower_truncated_std") == Catch::Approx(std::sqrt(2.0f)));
    CHECK(*replaced->GetScalar("lower_risk_full_q_argmax_disagreement") == Catch::Approx(0.5f));
    CHECK(*replaced->GetScalar("quantile_crossing_ratio") == Catch::Approx(0.0f));
    CHECK(*replaced->GetScalar("policy_selected_crossing_depth_p90_ratio") == Catch::Approx(0.0f));
}

TEST_CASE("QR policy exposes selected action crossing depth p90", "[dqn][qr][action_policy][metrics]")
{
    const auto q_quantiles = torch::tensor({ { { 0.0f, 2.0f, 1.0f, 4.0f, 2.0f } } });
    const auto obs = anet::TensorDict{
        { "q", q_quantiles.mean(2) },
        { "q_dist", q_quantiles },
    };

    dqn::ActionPolicyConfig config;
    config.quantile_mode = "qr";
    dqn::EpsilonGreedyActionPolicy policy(config);
    const anet::nn::NetworkForwardCallInfo call_info_2({});
    auto action_info = policy.SelectAction(
        obs, /*greedy_only=*/true, *MakeDefaultExecutor(MakePassthroughNetwork(1, 5)),
        std::make_shared<anet::RandomGenerator>(123), call_info_2, {});

    const auto crossing_depth_p90 =
        action_info->GetScalar("policy_selected_crossing_depth_p90_ratio");
    REQUIRE(crossing_depth_p90.has_value());
    CHECK(*crossing_depth_p90 == Catch::Approx(0.5f));
}

TEST_CASE("QR policy reports zero crossing depth without positive crossings", "[dqn][qr][action_policy][metrics]")
{
    const auto q_quantiles = torch::tensor({
        { { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f } },
        { { 2.0f, 2.0f, 2.0f, 2.0f, 2.0f } },
    });
    const auto obs = anet::TensorDict{
        { "q", q_quantiles.mean(2) },
        { "q_dist", q_quantiles },
    };

    dqn::ActionPolicyConfig config;
    config.quantile_mode = "qr";
    dqn::EpsilonGreedyActionPolicy policy(config);
    const anet::nn::NetworkForwardCallInfo call_info_3({});
    auto action_info = policy.SelectAction(
        obs, /*greedy_only=*/true, *MakeDefaultExecutor(MakePassthroughNetwork(1, 5)),
        std::make_shared<anet::RandomGenerator>(123), call_info_3, {});

    CHECK(*action_info->GetScalar("policy_selected_crossing_depth_p90_ratio")
        == Catch::Approx(0.0f));
}

TEST_CASE("QR selected crossing depth follows the final action per lane", "[dqn][qr][action_policy][metrics]")
{
    const auto q_quantiles = torch::tensor({
        {
            { 10.0f, 12.0f, 11.0f, 14.0f, 12.0f },
            { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f },
        },
        {
            { 10.0f, 11.0f, 12.0f, 13.0f, 14.0f },
            { 0.0f, 3.0f, 0.0f, 4.0f, 0.0f },
        },
    });
    const auto obs = anet::TensorDict{
        { "q", q_quantiles.mean(2) },
        { "q_dist", q_quantiles },
    };

    dqn::ActionPolicyConfig config;
    config.quantile_mode = "qr";
    dqn::EpsilonGreedyActionPolicy policy(config);
    const anet::nn::NetworkForwardCallInfo call_info_5({});
    auto action_info = policy.SelectAction(
        obs, /*greedy_only=*/true, *MakeDefaultExecutor(MakePassthroughNetwork(2, 5)),
        std::make_shared<anet::RandomGenerator>(123), call_info_5, {});
    CHECK(*action_info->GetScalar("policy_selected_crossing_depth_p90_ratio")
        == Catch::Approx(0.25f));

    auto replaced = std::dynamic_pointer_cast<dqn::DQNActionInfo>(action_info->WithAction(torch::ones(
        { 2 }, torch::TensorOptions().dtype(torch::kInt64))));
    REQUIRE(replaced != nullptr);
    CHECK(*replaced->GetScalar("policy_selected_crossing_depth_p90_ratio")
        == Catch::Approx(0.5f));
}

TEST_CASE("QR policy tail diagnostics exclude the odd median and reject K below two", "[dqn][qr][action_policy][metrics]")
{
    dqn::ActionPolicyConfig config;
    config.quantile_mode = "qr";
    dqn::EpsilonGreedyActionPolicy policy(config);
    auto rnd = std::make_shared<anet::RandomGenerator>(123);

    const auto odd_quantiles = torch::tensor({ { { 0.0f, 1.0f, 5.0f } } });
    const auto odd_obs = anet::TensorDict{
        { "q", odd_quantiles.mean(2) },
        { "q_dist", odd_quantiles },
    };
    const anet::nn::NetworkForwardCallInfo call_info_6({});
    auto odd_info = policy.SelectAction(
        odd_obs, /*greedy_only=*/true, *MakeDefaultExecutor(MakePassthroughNetwork(1, 3)), rnd, call_info_6, {});
    CHECK(*odd_info->GetScalar("policy_upper_truncated_std") == Catch::Approx(4.0f));
    CHECK(*odd_info->GetScalar("policy_lower_truncated_std") == Catch::Approx(1.0f));

    const auto one_quantile = torch::tensor({ { { 2.0f } } });
    const auto one_obs = anet::TensorDict{
        { "q", one_quantile.mean(2) },
        { "q_dist", one_quantile },
    };
    const anet::nn::NetworkForwardCallInfo call_info_7({});
    auto one_info = policy.SelectAction(
        one_obs, /*greedy_only=*/true, *MakeDefaultExecutor(MakePassthroughNetwork(1, 1)), rnd, call_info_7, {});
    CHECK(std::isnan(*one_info->GetScalar("policy_upper_truncated_std")));
    CHECK(std::isnan(*one_info->GetScalar("policy_lower_truncated_std")));
    CHECK(std::isnan(*one_info->GetScalar("lower_risk_full_q_argmax_disagreement")));
    CHECK(std::isnan(*one_info->GetScalar("quantile_crossing_ratio")));
    CHECK(std::isnan(*one_info->GetScalar("policy_selected_crossing_depth_p90_ratio")));
}

TEST_CASE("IQN action policies inject their tau rules without mutating observations", "[dqn][iqn][action_policy]")
{
    auto run_policy = [](
        const std::shared_ptr<dqn::ActionPolicy>& policy,
        const torch::Tensor& expected_taus) {
        auto state = std::make_shared<TauEchoState>();
        auto network = MakeTauEchoNetwork(expected_taus.size(1), state);
        auto obs = MakeTauEchoObservation();
        auto rnd = std::make_shared<anet::RandomGenerator>(123);

        REQUIRE_FALSE(obs.Contains(anet::rl::dqn::kKey_Taus));
        const anet::nn::NetworkForwardCallInfo call_info_8({});
        auto action_info = policy->SelectAction(obs, /*greedy_only=*/true, *MakeDefaultExecutor(network), rnd, call_info_8);

        REQUIRE(state->forward_count == 1);
        REQUIRE(state->last_taus.defined());
        CHECK(torch::allclose(state->last_taus, expected_taus, 1.0e-6, 1.0e-6));
        CHECK_FALSE(obs.Contains(anet::rl::dqn::kKey_Taus));
        const auto& aux = action_info->GetAuxData();
        REQUIRE(aux.count("q_values") == 1);
        REQUIRE(aux.count("q_quantiles") == 1);
        CHECK(aux.count("full_q_values") == 0);
        CHECK(aux.count("full_q_quantiles") == 0);
        CHECK(torch::allclose(aux.at("q_values"), aux.at("q_quantiles").mean(2)));
        return action_info;
    };

    SECTION("epsilon greedy and greedy score the full interval")
    {
        dqn::ActionPolicyConfig config;
        config.quantile_mode = "iqn";
        config.epsilon.value = 0.0f;
        config.epsilon.start = 0.0f;
        config.spatial.epsilon.first.value = 0.0f;
        config.epsilon.end = 0.0f;
        config.spatial.epsilon.last.value = 0.0f;
        config.tau_rule.num_taus = 4;
        config.tau_rule.sample_mode = "fixed";
        auto expected = torch::tensor({ 0.125f, 0.375f, 0.625f, 0.875f }).repeat({ 2, 1 });

        run_policy(std::make_shared<dqn::EpsilonGreedyActionPolicy>(config), expected);
    }

    SECTION("UQE tail mean scores the sampled upper tail")
    {
        dqn::ActionPolicyConfig config;
        config.quantile_mode = "iqn";
        config.uqe.tau.value = 0.5f;
        config.uqe.tau.start = 0.5f;
        config.spatial.uqe.tau.first.value = 0.5f;
        config.uqe.tau.end = 0.5f;
        config.spatial.uqe.tau.last.value = 0.5f;
        config.uqe.use_tail_mean = true;
        config.uqe.epsilon.value = 0.0f;
        config.uqe.epsilon.start = 0.0f;
        config.spatial.uqe.epsilon.first.value = 0.0f;
        config.uqe.epsilon.end = 0.0f;
        config.spatial.uqe.epsilon.last.value = 0.0f;
        config.tau_rule.num_taus = 2;
        config.tau_rule.sample_mode = "fixed";
        auto expected = torch::tensor({ 0.625f, 0.875f }).repeat({ 2, 1 });

        auto action_info = run_policy(std::make_shared<dqn::UQEActionPolicy>(config), expected);
        CHECK(torch::allclose(
            action_info->GetAuxData().at("q_values"),
            action_info->GetAuxData().at("uqe_values")));
    }

    SECTION("UQE point score uses the decayed Z tau")
    {
        dqn::ActionPolicyConfig config;
        config.quantile_mode = "iqn";
        config.uqe.tau.value = 0.2f;
        config.uqe.tau.start = 0.2f;
        config.spatial.uqe.tau.first.value = 0.2f;
        config.uqe.tau.end = 0.6f;
        config.spatial.uqe.tau.last.value = 0.6f;
        config.uqe.tau.type = "linear";
        config.uqe.tau.steps = 10;
        config.uqe.use_tail_mean = false;
        config.uqe.epsilon.value = 0.0f;
        config.uqe.epsilon.start = 0.0f;
        config.spatial.uqe.epsilon.first.value = 0.0f;
        config.uqe.epsilon.end = 0.0f;
        config.spatial.uqe.epsilon.last.value = 0.0f;
        config.tau_rule.num_taus = 3;
        config.tau_rule.sample_mode = "fixed";
        auto policy = std::make_shared<dqn::UQEActionPolicy>(config);
        policy->UpdateSchedule(rl::StepCounts{ .exp_step = 5 });
        auto expected = torch::full({ 2, 3 }, 0.4f);

        auto action_info = run_policy(policy, expected);
        CHECK(torch::allclose(
            action_info->GetAuxData().at("q_values"),
            action_info->GetAuxData().at("uqe_values")));
    }

    SECTION("non spatial Thompson sampling scores the full interval")
    {
        dqn::ActionPolicyConfig config;
        config.quantile_mode = "iqn";
        config.tau_rule.num_taus = 2;
        config.tau_rule.sample_mode = "fixed";
        auto expected = torch::tensor({ 0.25f, 0.75f }).repeat({ 2, 1 });

        run_policy(std::make_shared<dqn::ThompsonSamplingActionPolicy>(config), expected);
    }

    SECTION("spatial UQE and Thompson sampling use per-env lower bounds")
    {
        dqn::ActionPolicyConfig config;
        config.quantile_mode = "iqn";
        config.spatial.enabled = true;
        config.spatial.scale = "linear";
        config.uqe.tau.value = 0.0f;
        config.uqe.tau.start = 0.0f;
        config.spatial.uqe.tau.first.value = 0.0f;
        config.uqe.tau.end = 0.5f;
        config.spatial.uqe.tau.last.value = 0.5f;
        config.uqe.use_tail_mean = true;
        config.uqe.epsilon.value = 0.0f;
        config.uqe.epsilon.start = 0.0f;
        config.spatial.uqe.epsilon.first.value = 0.0f;
        config.uqe.epsilon.end = 0.0f;
        config.spatial.uqe.epsilon.last.value = 0.0f;
        config.tau_rule.num_taus = 2;
        config.tau_rule.sample_mode = "fixed";
        auto expected = torch::tensor({
            { 0.625f, 0.875f },
            { 0.25f, 0.75f },
        });

        run_policy(
            std::make_shared<dqn::UQEActionPolicy>(config, true, 2, torch::Device(torch::kCPU)),
            expected);
        run_policy(
            std::make_shared<dqn::ThompsonSamplingActionPolicy>(config, true, 2, torch::Device(torch::kCPU)),
            expected);
    }
}

TEST_CASE("IQN UQE point query exports a full distribution from the same forward", "[dqn][iqn][action_policy]")
{
    dqn::ActionPolicyConfig config;
    config.quantile_mode = "iqn";
    config.uqe.tau.value = 0.4f;
    config.uqe.tau.start = 0.4f;
    config.spatial.uqe.tau.first.value = 0.4f;
    config.uqe.tau.end = 0.4f;
    config.spatial.uqe.tau.last.value = 0.4f;
    config.uqe.use_tail_mean = false;
    config.uqe.epsilon.value = 0.0f;
    config.uqe.epsilon.start = 0.0f;
    config.spatial.uqe.epsilon.first.value = 0.0f;
    config.uqe.epsilon.end = 0.0f;
    config.spatial.uqe.epsilon.last.value = 0.0f;
    config.tau_rule.num_taus = 3;
    config.tau_rule.sample_mode = "fixed";
    config.full_distribution_query.enabled = true;
    config.full_distribution_query.tau_rule.num_taus = 4;
    config.full_distribution_query.tau_rule.sample_mode = "fixed";

    auto state = std::make_shared<TauEchoState>();
    auto network = MakeTauEchoNetwork(5, state);
    auto obs = MakeTauEchoObservation();
    auto rnd = std::make_shared<anet::RandomGenerator>(123);
    auto policy = std::make_shared<dqn::UQEActionPolicy>(config);

    REQUIRE_FALSE(obs.Contains(anet::rl::dqn::kKey_Taus));
    const anet::nn::NetworkForwardCallInfo call_info_9({});
    auto action_info = policy->SelectAction(obs, /*greedy_only=*/true, *MakeDefaultExecutor(network), rnd, call_info_9, {});

    const auto expected_taus = torch::tensor({ 0.4f, 0.125f, 0.375f, 0.625f, 0.875f }).repeat({ 2, 1 });
    REQUIRE(state->forward_count == 1);
    CHECK(torch::allclose(state->last_taus, expected_taus, 1.0e-6, 1.0e-6));
    CHECK_FALSE(obs.Contains(anet::rl::dqn::kKey_Taus));

    const auto& aux = action_info->GetAuxData();
    REQUIRE(aux.count("full_q_values") == 1);
    REQUIRE(aux.count("full_q_quantiles") == 1);
    CHECK((ShapeOf(aux.at("q_quantiles")) == std::vector<int64_t>{ 2, 2, 1 }));
    CHECK((ShapeOf(aux.at("full_q_quantiles")) == std::vector<int64_t>{ 2, 2, 4 }));
    CHECK(torch::allclose(aux.at("q_values"), aux.at("q_quantiles").mean(2)));
    CHECK(torch::allclose(aux.at("uqe_values"), aux.at("q_values")));
    CHECK(torch::allclose(aux.at("full_q_values"), aux.at("full_q_quantiles").mean(2)));
    CHECK(torch::equal(action_info->GetAction().cpu(), torch::ones({ 2 }, torch::TensorOptions().dtype(torch::kInt64))));

    // point queryのK=1ではscaleを定義せず、full Qとの選択差だけを公開する。
    const auto margin_ratio = action_info->GetScalar("iqn_policy_margin_mc_ratio");
    REQUIRE(margin_ratio.has_value());
    CHECK(std::isnan(*margin_ratio));
    const auto disagreement = action_info->GetScalar("iqn_uqe_full_q_argmax_disagreement");
    REQUIRE(disagreement.has_value());
    CHECK(*disagreement == Catch::Approx(1.0f));
}

TEST_CASE("IQN UQE tail score excludes the optional full distribution", "[dqn][iqn][action_policy]")
{
    dqn::ActionPolicyConfig config;
    config.quantile_mode = "iqn";
    config.uqe.tau.value = 0.5f;
    config.uqe.tau.start = 0.5f;
    config.spatial.uqe.tau.first.value = 0.5f;
    config.uqe.tau.end = 0.5f;
    config.spatial.uqe.tau.last.value = 0.5f;
    config.uqe.use_tail_mean = true;
    config.uqe.epsilon.value = 0.0f;
    config.uqe.epsilon.start = 0.0f;
    config.spatial.uqe.epsilon.first.value = 0.0f;
    config.uqe.epsilon.end = 0.0f;
    config.spatial.uqe.epsilon.last.value = 0.0f;
    config.tau_rule.num_taus = 2;
    config.tau_rule.sample_mode = "fixed";
    config.full_distribution_query.enabled = true;
    config.full_distribution_query.tau_rule.num_taus = 2;
    config.full_distribution_query.tau_rule.sample_mode = "fixed";

    auto state = std::make_shared<TauEchoState>();
    auto network = MakeTauEchoNetwork(4, state);
    auto obs = MakeTauEchoObservation();
    auto rnd = std::make_shared<anet::RandomGenerator>(123);
    dqn::UQEActionPolicy policy(config);

    const anet::nn::NetworkForwardCallInfo call_info_10({});
    auto action_info = policy.SelectAction(obs, /*greedy_only=*/true, *MakeDefaultExecutor(network), rnd, call_info_10, {});

    const auto expected_taus = torch::tensor({ 0.625f, 0.875f, 0.25f, 0.75f }).repeat({ 2, 1 });
    REQUIRE(state->forward_count == 1);
    CHECK(torch::allclose(state->last_taus, expected_taus, 1.0e-6, 1.0e-6));

    const auto& aux = action_info->GetAuxData();
    CHECK((ShapeOf(aux.at("q_quantiles")) == std::vector<int64_t>{ 2, 2, 2 }));
    CHECK((ShapeOf(aux.at("full_q_quantiles")) == std::vector<int64_t>{ 2, 2, 2 }));
    CHECK(torch::allclose(aux.at("q_values"), torch::tensor({ { 0.75f, 0.25f }, { 0.75f, 0.25f } })));
    CHECK(torch::allclose(aux.at("full_q_values"), torch::full({ 2, 2 }, 0.5f)));
    CHECK(torch::equal(action_info->GetAction().cpu(), torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kInt64))));

    // 同じforwardのrisk/full quantileからPolicy診断を公開する。
    const auto margin_ratio = action_info->GetScalar("iqn_policy_margin_mc_ratio");
    REQUIRE(margin_ratio.has_value());
    CHECK(*margin_ratio == Catch::Approx(2.828411f));
    const auto disagreement = action_info->GetScalar("iqn_uqe_full_q_argmax_disagreement");
    REQUIRE(disagreement.has_value());
    CHECK(*disagreement == Catch::Approx(0.0f));
    const auto noop_margin = action_info->GetScalar("action_full_q_margin.[0]");
    REQUIRE(noop_margin.has_value());
    CHECK(*noop_margin == Catch::Approx(0.0f));
    CHECK(*action_info->GetScalar("policy_upper_truncated_std") == Catch::Approx(0.25f));
    CHECK(*action_info->GetScalar("policy_lower_truncated_std") == Catch::Approx(0.25f));
    CHECK(*action_info->GetScalar("lower_risk_full_q_argmax_disagreement") == Catch::Approx(0.0f));
    CHECK(*action_info->GetScalar("quantile_crossing_ratio") == Catch::Approx(0.5f));
    CHECK(*action_info->GetScalar("policy_selected_crossing_depth_p90_ratio") == Catch::Approx(0.0f));
    CHECK_THROWS_WITH(
        action_info->GetScalar("action_full_q_margin.[2]"),
        Catch::Matchers::ContainsSubstring("index=2")
            && Catch::Matchers::ContainsSubstring("valid_range=[0,1]"));
    CHECK(state->forward_count == 1);

    // full queryを無効化してもrisk診断は残り、full依存診断だけNaNになる。
    config.full_distribution_query.enabled = false;
    auto risk_only_state = std::make_shared<TauEchoState>();
    const anet::nn::NetworkForwardCallInfo call_info_11({});
    auto risk_only = dqn::UQEActionPolicy(config).SelectAction(
        obs, /*greedy_only=*/true, *MakeDefaultExecutor(MakeTauEchoNetwork(2, risk_only_state)), rnd, call_info_11, {});
    CHECK_FALSE(std::isnan(*risk_only->GetScalar("iqn_policy_margin_mc_ratio")));
    CHECK(std::isnan(*risk_only->GetScalar("iqn_uqe_full_q_argmax_disagreement")));
    CHECK(std::isnan(*risk_only->GetScalar("action_full_q_margin.[0]")));
    CHECK(std::isnan(*risk_only->GetScalar("policy_upper_truncated_std")));
    CHECK(std::isnan(*risk_only->GetScalar("policy_lower_truncated_std")));
    CHECK(std::isnan(*risk_only->GetScalar("lower_risk_full_q_argmax_disagreement")));
    CHECK(std::isnan(*risk_only->GetScalar("quantile_crossing_ratio")));
    CHECK(std::isnan(*risk_only->GetScalar("policy_selected_crossing_depth_p90_ratio")));
    CHECK(risk_only_state->forward_count == 1);

    // random full queryはtau順が保証されないため、tail診断payloadを作らない。
    config.full_distribution_query.enabled = true;
    config.full_distribution_query.tau_rule.sample_mode = "random";
    auto random_full_state = std::make_shared<TauEchoState>();
    const anet::nn::NetworkForwardCallInfo call_info_12({});
    auto random_full = dqn::UQEActionPolicy(config).SelectAction(
        obs, /*greedy_only=*/true, *MakeDefaultExecutor(MakeTauEchoNetwork(4, random_full_state)), rnd, call_info_12, {});
    CHECK(std::isnan(*random_full->GetScalar("policy_upper_truncated_std")));
    CHECK(std::isnan(*random_full->GetScalar("policy_lower_truncated_std")));
    CHECK(std::isnan(*random_full->GetScalar("lower_risk_full_q_argmax_disagreement")));
    CHECK(std::isnan(*random_full->GetScalar("quantile_crossing_ratio")));
    CHECK(std::isnan(*random_full->GetScalar("policy_selected_crossing_depth_p90_ratio")));
    CHECK(random_full_state->forward_count == 1);
}

TEST_CASE("DQN action policy BF16 autocast follows observation device", "[dqn][action_policy][amp][bf16]")
{
    std::vector<torch::Device> devices{ torch::Device(torch::kCPU) };
    if (torch::cuda::is_available()) {
        devices.emplace_back(torch::Device(torch::kCUDA, 0));
    }

    for (const auto& device : devices) {
        INFO("device=" << device.str());

        auto probe_state = std::make_shared<AutocastProbeState>();
        auto network = MakeAutocastProbeNetwork(probe_state, device);
        auto obs = MakeAutocastProbePolicyInput(device);

        dqn::ActionPolicyConfig config;
        config.use_amp = true;
        config.use_amp_bf16 = true;
        dqn::EpsilonGreedyActionPolicy policy(config, false, 0, device);

        const auto device_type = device.type();
        const bool original_enabled = at::autocast::is_autocast_enabled(device_type);
        auto rnd = std::make_shared<anet::RandomGenerator>(123);
        const anet::nn::NetworkForwardCallInfo call_info_13({});
        auto action_info = policy.SelectAction(obs, /*greedy_only=*/true, *MakeDefaultExecutor(network), rnd, call_info_13, {});

        REQUIRE(action_info->GetAction().device().type() == device_type);
        REQUIRE(probe_state->forward_count > 0);
        CHECK(probe_state->enabled_count == probe_state->forward_count);
        CHECK(probe_state->last_device_type == device_type);
        CHECK(at::autocast::is_autocast_enabled(device_type) == original_enabled);
    }
}

TEST_CASE("DQN Actor emits a packed priority hint without another forward", "[dqn][actor][per][actor_initial]")
{
    auto make_action = [](bool emit_hint) {
        auto probe_state = std::make_shared<AutocastProbeState>();
        auto network = MakeAutocastProbeNetwork(probe_state, torch::kCPU);
        auto policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{});
        auto mutex = std::make_shared<std::shared_mutex>();
        dqn::Actor actor(policy, nullptr, nullptr, torch::kCPU, 123, mutex, network, network, anet::nn::NetworkForwardConfig::GetDefault(), emit_hint);

        auto flags = torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kBool));
        auto episode_start = torch::tensor({ true, false }, torch::TensorOptions().dtype(torch::kBool));
        rl::BatchState state(MakeAutocastProbePolicyInput(torch::kCPU), flags, flags, episode_start);
        auto action_info = actor.MakeAction(rl::StepCounts{}, state);
        return std::pair(probe_state->forward_count, std::move(action_info));
    };

    const auto [plain_forward_count, plain] = make_action(false);
    const auto [hint_forward_count, hinted] = make_action(true);

    CHECK(plain_forward_count == hint_forward_count);
    CHECK_FALSE(plain->GetReplayInitialPriorityHint().has_value());
    CHECK(torch::equal(
        plain->GetAuxData().at("episode_start"),
        torch::tensor({ true, false }, torch::TensorOptions().dtype(torch::kBool))));
    REQUIRE(hinted->GetReplayInitialPriorityHint().has_value());
    const auto decoded = dqn::DecodeActorQHint(hinted->GetReplayInitialPriorityHint()->GetPayload());
    const auto& q_values = hinted->GetAuxData().at("q_values");
    const auto expected_q_sa = q_values.gather(
        1, hinted->GetAction(q_values.device()).to(torch::kInt64).unsqueeze(1)).squeeze(1).to(torch::kFloat32);
    const auto expected_state_value = std::get<0>(q_values.max(1)).to(torch::kFloat32);
    CHECK(torch::equal(decoded.actor_q_sa, expected_q_sa));
    CHECK(torch::equal(decoded.actor_state_value, expected_state_value));
}

TEST_CASE("DQN Actor snapshot synchronization performs one forward per action", "[dqn][actor][snapshot]")
{
    auto source_probe = std::make_shared<AutocastProbeState>();
    auto snapshot_probe = std::make_shared<AutocastProbeState>();
    auto source_network = MakeAutocastProbeNetwork(source_probe, torch::kCPU);
    auto snapshot_network = MakeAutocastProbeNetwork(snapshot_probe, torch::kCPU);
    auto policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{});
    auto mutex = std::make_shared<std::shared_mutex>();
    anet::ProfiledValueConfig<rl::step_t> sync_interval;
    sync_interval.value = 1;
    dqn::Actor actor(
        policy, nullptr, nullptr, torch::kCPU, 123, mutex, snapshot_network, source_network, anet::nn::NetworkForwardConfig::GetDefault(), false, sync_interval, true);

    auto flags = torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kBool));
    rl::BatchState state(MakeAutocastProbePolicyInput(torch::kCPU), flags, flags, flags);
    for (rl::step_t train_step = 0; train_step < 3; ++train_step) {
        rl::StepCounts step;
        step.train_step = train_step;
        step.exp_step = train_step;
        actor.MakeAction(step, state);
    }

    CHECK(source_probe->forward_count == 0);
    CHECK(snapshot_probe->forward_count == 3);
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
    CHECK(std::isnan(*non_uqe.GetScalar("iqn_policy_margin_mc_ratio")));
    CHECK(std::isnan(*non_uqe.GetScalar("iqn_uqe_full_q_argmax_disagreement")));
    CHECK(std::isnan(*non_uqe.GetScalar("action_full_q_margin.[0]")));

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

TEST_CASE("DQNActionInfo exposes episode-start action margin scalar metrics", "[dqn][action_policy][metrics]")
{
    const auto make_info = [](const torch::Tensor& q_values, const torch::Tensor& uqe_values, const torch::Tensor& episode_start) {
        rl::AuxData aux;
        aux["q_values"] = q_values;
        aux["uqe_values"] = uqe_values;
        aux["episode_start"] = episode_start;
        return dqn::DQNActionInfo(
            torch::zeros({ q_values.size(0) }, torch::TensorOptions().dtype(torch::kInt64)),
            anet::TensorDict{},
            aux);
    };

    const auto q_values = torch::tensor({
        { 6.0f, 2.0f, 1.0f },
        { 9.0f, 1.0f, 0.0f },
        { 2.0f, 5.0f, 0.0f },
    });
    const auto uqe_values = torch::tensor({
        { 5.0f, 1.0f, 0.0f },
        { 8.0f, 2.0f, 1.0f },
        { 3.0f, 4.0f, 2.0f },
    });
    const auto episode_start = torch::tensor({ true, false, true }, torch::TensorOptions().dtype(torch::kBool));
    auto info = make_info(q_values, uqe_values, episode_start);

    auto uqe_margin = info.GetScalar("episode_start_action_uqe_margin.[0]");
    REQUIRE(uqe_margin.has_value());
    CHECK(*uqe_margin == Catch::Approx(1.5f));
    auto q_margin = info.GetScalar("episode_start_action_q_margin.[0]");
    REQUIRE(q_margin.has_value());
    CHECK(*q_margin == Catch::Approx(0.5f));

    const auto no_episode_start = torch::zeros({ 3 }, torch::TensorOptions().dtype(torch::kBool));
    auto no_reset_info = make_info(q_values, uqe_values, no_episode_start);
    auto no_reset_uqe = no_reset_info.GetScalar("episode_start_action_uqe_margin.[0]");
    auto no_reset_q = no_reset_info.GetScalar("episode_start_action_q_margin.[0]");
    REQUIRE(no_reset_uqe.has_value());
    REQUIRE(no_reset_q.has_value());
    CHECK(std::isnan(*no_reset_uqe));
    CHECK(std::isnan(*no_reset_q));

    rl::AuxData non_uqe_aux;
    non_uqe_aux["q_values"] = q_values;
    non_uqe_aux["episode_start"] = episode_start;
    dqn::DQNActionInfo non_uqe(
        torch::zeros({ 3 }, torch::TensorOptions().dtype(torch::kInt64)),
        anet::TensorDict{},
        non_uqe_aux);
    auto non_uqe_margin = non_uqe.GetScalar("episode_start_action_uqe_margin.[0]");
    REQUIRE(non_uqe_margin.has_value());
    CHECK(std::isnan(*non_uqe_margin));

    auto replaced = info.WithAction(torch::tensor({ 2, 1, 0 }, torch::TensorOptions().dtype(torch::kInt64)));
    auto replaced_scalar_target = dynamic_cast<const anet::Module*>(replaced.get());
    REQUIRE(replaced_scalar_target != nullptr);
    auto replaced_q_margin = replaced_scalar_target->GetScalar("episode_start_action_q_margin.[0]");
    REQUIRE(replaced_q_margin.has_value());
    CHECK(*replaced_q_margin == Catch::Approx(0.5f));

    rl::AuxData missing_mask_aux;
    missing_mask_aux["q_values"] = q_values;
    dqn::DQNActionInfo missing_mask(
        torch::zeros({ 3 }, torch::TensorOptions().dtype(torch::kInt64)),
        anet::TensorDict{},
        missing_mask_aux);
    CHECK_THROWS(missing_mask.GetScalar("episode_start_action_q_margin.[0]"));

    rl::AuxData missing_q_aux;
    missing_q_aux["episode_start"] = episode_start;
    dqn::DQNActionInfo missing_q(
        torch::zeros({ 3 }, torch::TensorOptions().dtype(torch::kInt64)),
        anet::TensorDict{},
        missing_q_aux);
    CHECK_THROWS(missing_q.GetScalar("episode_start_action_q_margin.[0]"));

    auto invalid_mask_dtype = make_info(q_values, uqe_values, episode_start.to(torch::kInt64));
    CHECK_THROWS(invalid_mask_dtype.GetScalar("episode_start_action_q_margin.[0]"));
    auto invalid_mask_shape = make_info(q_values, uqe_values, episode_start.unsqueeze(1));
    CHECK_THROWS(invalid_mask_shape.GetScalar("episode_start_action_q_margin.[0]"));
    auto invalid_q_shape = make_info(q_values.flatten(), uqe_values, episode_start);
    CHECK_THROWS(invalid_q_shape.GetScalar("episode_start_action_q_margin.[0]"));
    CHECK_THROWS(info.GetScalar("episode_start_action_q_margin"));
    CHECK_THROWS(info.GetScalar("episode_start_action_q_margin.[x]"));
    CHECK_THROWS(info.GetScalar("episode_start_action_q_margin.[3]"));
}

TEST_CASE("Decomposed Actor hints reuse policy selection and exclude full queries", "[prd121][dqn][decomposed_hint_policy]")
{
    const bool iqn = GENERATE(false, true);
    const auto external = iqn ? torch::tensor({{{1.0f, 200.0f, 200.0f, 200.0f, 200.0f}, {2.0f, 100.0f, 100.0f, 100.0f, 100.0f}}})
        : torch::tensor({{{0.0f, 0.0f, 12.0f}, {6.0f, 6.0f, 6.0f}}});
    const auto internal = iqn ? torch::tensor({{{3.0f, 50.0f, 50.0f, 50.0f, 50.0f}, {4.0f, 100.0f, 100.0f, 100.0f, 100.0f}}})
        : torch::zeros_like(external);
    anet::TensorDict observations{{kPrd121ExtrinsicQDist, external}, {kPrd121IntrinsicQDist, internal},
        {kPrd121ExtrinsicQ, external.mean(2)}, {kPrd121IntrinsicQ, internal.mean(2)}};
    anet::TensorSpecMap specs;
    anet::nn::NetworkConfig net;
    for (const auto& [key, value] : observations) {
        specs[key] = {.shape = value.sizes().slice(1).vec(), .dtype = torch::kFloat32};
        net.output_keys[key] = key;
    }
    auto network = anet::test::MakeTestNetwork(net, specs, nullptr, nullptr);
    dqn::ActionPolicyConfig config;
    config.policy_type = "UQE";
    config.quantile_mode = iqn ? anet::rl::dqn::ActionPolicyConfig::kQuantileModeIqn : anet::rl::dqn::ActionPolicyConfig::kQuantileModeQr;
    config.uqe.tau.value = 1;
    config.uqe.use_tail_mean = false;
    config.uqe.epsilon.value = 0;
    config.tau_rule = {.sample_mode = kPrd121FixedTauMode, .num_taus = 1};
    config.full_distribution_query.enabled = iqn;
    config.full_distribution_query.tau_rule = {.sample_mode = kPrd121FixedTauMode, .num_taus = 4};
    auto policy = dqn::DqnComponentAssembler::CreateActionPolicy(config, config.quantile_mode, false, 1, torch::kCPU);
    dqn::ActorQHintConfig hint;
    hint.decomposed = true;
    hint.beta.value = 0.5f;
    hint.extrinsic_bootstrap = true;
    dqn::Actor actor(policy, nullptr, nullptr, torch::kCPU, 121, std::make_shared<std::shared_mutex>(), network, network,
        anet::nn::NetworkForwardConfig::GetDefault(), true, std::nullopt, false, hint);
    const auto flags = torch::zeros({1}, torch::kBool);
    const auto info = actor.MakeAction({}, rl::BatchState(observations, flags, flags, ~flags));
    REQUIRE(info->GetReplayInitialPriorityHint());
    const auto values = dqn::DecodeActorQHint(info->GetReplayInitialPriorityHint()->GetPayload());
    CHECK(values.actor_q_sa.item<float>() == Catch::Approx(iqn ? 2 : 4));
    CHECK(values.actor_state_value.item<float>() == Catch::Approx(iqn ? 2 : 4));
    CHECK(values.intrinsic_q_sa.item<float>() == Catch::Approx(iqn ? 4 : 0));
    const auto replaced = info->WithAction(torch::tensor({0}, torch::kInt64));
    const auto replacement = dqn::DecodeActorQHint(replaced->GetReplayInitialPriorityHint()->GetPayload());
    CHECK(replacement.actor_q_sa.item<float>() == Catch::Approx(iqn ? 1 : 4));
    CHECK(replacement.intrinsic_q_sa.item<float>() == Catch::Approx(iqn ? 3 : 0));
}

TEST_CASE("Extrinsic bootstrap preserves UQE risk selection in learning and diagnostics", "[prd121][dqn][decomposed_risk]")
{
    ScopedNoopMetricsLogger logger;
    anet::test::RegisterStandardNetworkModules();
    const bool tail = GENERATE(false, true);
    auto env = MakeLearnerEnvSpec();
    env.action_spec.value_labels = {"a0", "a1"};
    anet::nn::NetworkConfig net;
    net.output_keys[kPrd121Features] = kVectorKey;
    const anet::nn::WeightInitConfig init;
    auto factory = std::make_shared<dqn::QuantileHeadFactory>(2, 3, init);
    auto head = std::make_shared<dqn::DecomposedHeadFactory>(factory, factory, false, false);
    dqn::NetworkModel model({}, torch::kCPU, net, env.state_spec.obs_spec, 2, head, true, 121);
    const auto set_biases = [](const auto& network, const torch::Tensor& external, const torch::Tensor& internal) {
        torch::NoGradGuard guard;
        for (auto& entry : network->named_parameters()) {
            entry.value().zero_();
            if (entry.key() == kPrd121HeadExtrinsicLinearBias) entry.value().copy_(external.flatten());
            if (entry.key() == kPrd121HeadIntrinsicLinearBias) entry.value().copy_(internal.flatten());
        }
    };
    const auto external = tail ? torch::tensor({{-100.0f, 0.0f, 0.0f}, {-1.0f, -1.0f, -1.0f}})
        : torch::tensor({{0.0f, 0.0f, 100.0f}, {5.0f, 5.0f, 5.0f}});
    const auto internal = tail ? torch::tensor({{0.0f, 0.0f, 0.0f}, {20.0f, 20.0f, 20.0f}})
        : torch::tensor({{20.0f, 20.0f, 20.0f}, {0.0f, 0.0f, 0.0f}});
    set_biases(model.GetOnlineNetwork(), external, internal);
    set_biases(model.GetTargetNetwork(), torch::tensor({{4.0f, 4.0f, 4.0f}, {6.0f, 6.0f, 6.0f}}),
        torch::tensor({{1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f}}));
    auto config = WithDefaultLearnerForwardConfig();
    config.quantile_mode = anet::rl::dqn::ActionPolicyConfig::kQuantileModeQr;
    config.num_quantiles = 3;
    config.intrinsic.decomposed = true;
    config.intrinsic.extrinsic_bootstrap_mode = anet::rl::dqn::LearnerConfig::kBootstrapExtrinsic;
    config.gamma.value = 0.5f;
    config.use_double_dqn = true;
    config.use_per = false;
    config.use_tbo = false;
    config.learning_rate.value = 0;
    config.replay_batch_size = 2;
    config.replay_capacity = 16;
    config.use_n_step = false;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = -1;
    config.replay_fit.probe.batch_size = 1;
    dqn::ActionPolicyConfig policy_config;
    policy_config.quantile_mode = anet::rl::dqn::ActionPolicyConfig::kQuantileModeQr;
    policy_config.uqe.tau.value = tail ? 0.5f : 0;
    policy_config.uqe.use_tail_mean = tail;
    auto policy = std::make_shared<dqn::UQEActionPolicy>(policy_config);
    dqn::RuntimeVars vars;
    dqn::QRLearner learner(config, model, vars, nullptr, {2, 1}, env, torch::kCPU, 121, kTestAgentSeed, policy, {});
    auto experience = MakeDeterminismExperience(0, 2);
    experience.reward = torch::zeros({2, 2});
    experience.next_state.done.zero_();
    experience.next_state.truncated.zero_();
    learner.UpdateFromBatch({}, experience, 0.5f);
    rl::ExperienceSamples samples;
    samples.obs = experience.state.obs;
    samples.next_state.next_obs = samples.obs;
    samples.actions = torch::zeros({2}, torch::kInt64);
    samples.target_returns = torch::zeros({2, 2});
    samples.n_steps = torch::ones({2}, torch::kInt64);
    samples.next_state.terminals = torch::zeros({2}, torch::kBool);
    samples.next_state.truncated = torch::zeros({2}, torch::kBool);
    const auto result = learner.UpdateFromSamples(samples);
    const float expected = std::abs(external[0].mean().item<float>() - (tail ? 2.0f : 3.0f))
        + std::abs(internal[0].mean().item<float>() - (tail ? 1.0f : 0.5f));
    CHECK(torch::allclose(result->td_error, torch::full({2}, expected), 1e-5, 1e-5));
    learner.ConfigureScalarMetricSubscriptions({{.source_key = kPrd121ReplayFitUniformTdMean, .event = rl::EventType::LEARN,
        .target = rl::EventField::UPDATE_RESULT, .interval = 1, .scope = rl::RunnerScope::TRAIN}});
    experience.state.episode_start.zero_();
    const auto updates = learner.UpdateFromBatch({.exp_step = 2}, experience, 0.5f);
    REQUIRE_FALSE(updates.empty());
    REQUIRE(updates.back()->GetScalar(kPrd121ReplayFitUniformTdMean));
    CHECK(std::isfinite(*updates.back()->GetScalar(kPrd121ReplayFitUniformTdMean)));
}

TEST_CASE("Decomposition validates all Actor and Head values before combinations", "[prd121][dqn][config][prd121-review-config]")
{
    const int combination = GENERATE(0, 1, 2, 3);
    const auto invalid = GENERATE(Catch::Generators::values<std::pair<std::string, std::string>>({
        {"intrinsic.value.head.input_mode", kPrd121Invalid},
        {"intrinsic.value.head.init.manual_gain", kPrd121Nan},
        {"intrinsic.value.head.projection.noisy.sigma0", "-1"},
        {"actor.[z].intrinsic.action_beta_mode", kPrd121Invalid},
        {"actor.[z].intrinsic.action_beta", "-1"}}));
    anet::ConfigData base;
    base.Set("DefaultDQNAgent.actor.[a].network", kPrd121OnlineNetworkMode);
    base.Set("DefaultDQNAgent.actor.[z].network", kPrd121OnlineNetworkMode);
    auto data = WithDefaultForwardConfig(base);
    data.Set("DefaultDQNAgent.intrinsic.value_mode", dqn::DefaultDQNAgentConfig::kValueModeDecomposed);
    data.Set("DefaultDQNAgent.intrinsic.enabled", combination != 0);
    if (combination == 1) {
        data.Set("DefaultDQNAgent.skip.enabled", true);
        data.Set("DefaultDQNAgent.skip.max_length", 2);
        data.Set("DefaultDQNAgent.skip.net.branch.[shared].bind", kPrd121Features);
        data.Set("DefaultDQNAgent.skip.net.body.output.[features]", kPrd121SharedBranchKey);
    }
    if (combination == 2)
        data.Set("DefaultDQNAgent.learner.intrinsic.discount_mode", dqn::LearnerConfig::kDiscountSeparate);
    if (combination == 3)
        data.Set("DefaultDQNAgent.actor.[a].intrinsic.action_beta_mode", dqn::DQNActorConfig::kActionBetaFixed);
    data.Set("DefaultDQNAgent." + invalid.first, invalid.second);
    CAPTURE(combination, invalid.first);
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(data), Catch::Matchers::ContainsSubstring(
        invalid.first.starts_with("intrinsic.value.head.projection.") ? "intrinsic.value.head.projection" : invalid.first));
}

TEST_CASE("DQN Actor Q hint schema packs and decodes five columns", "[dqn][per][actor_initial][hint]")
{
    auto q_sa = torch::tensor({ 2.0f, 3.0f });
    auto state_value = torch::tensor({ 5.0f, 7.0f });

    auto packed = dqn::PackActorQHint(q_sa, state_value, torch::zeros_like(q_sa), torch::zeros_like(q_sa), torch::zeros_like(q_sa));
    CHECK(packed.scalar_type() == torch::kFloat32);
    CHECK(packed.sizes() == torch::IntArrayRef({ 2, dqn::kActorQHintColumnCount }));

    const auto batch = dqn::DecodeActorQHint(packed);
    CHECK(torch::equal(batch.actor_q_sa, q_sa));
    CHECK(torch::equal(batch.actor_state_value, state_value));

    const std::array<float, dqn::kActorQHintColumnCount> row{ 11.0f, 13.0f, 0.0f };
    const auto decoded_row = dqn::DecodeActorQHint(std::span<const float>(row));
    CHECK(decoded_row.actor_q_sa == Catch::Approx(11.0f));
    CHECK(decoded_row.actor_state_value == Catch::Approx(13.0f));

    CHECK_THROWS(dqn::DecodeActorQHint(torch::zeros({ 1, 2 }, torch::kFloat32)));
    CHECK_THROWS(dqn::DecodeActorQHint(std::span<const float>(row.data(), 1)));
}

TEST_CASE("Actor Q hint names invalid intrinsic tensors and their actual metadata", "[prd121][dqn][hint][prd121-review-pack]")
{
    const bool state_value = GENERATE(false, true);
    const int invalid = GENERATE(0, 1, 2);
    const auto primary = torch::zeros({2});
    auto intrinsic_q = torch::zeros_like(primary);
    auto intrinsic_value = torch::zeros_like(primary);
    torch::Tensor bad;
    if (invalid == 1) bad = torch::zeros({3});
    if (invalid == 2) bad = torch::empty({2}, torch::TensorOptions().device(c10::DeviceType::Meta));
    (state_value ? intrinsic_value : intrinsic_q) = bad;
    const auto name = state_value ? "intrinsic_state_value" : "intrinsic_q_sa";
    const auto shape = invalid == 0 ? "shape=undefined" : invalid == 1 ? "shape=[3]" : "shape=[2]";
    const auto device = invalid == 0 ? "device=undefined" : invalid == 1 ? "device=cpu" : "device=meta";
    CAPTURE(state_value, invalid);
    CHECK_THROWS_WITH(dqn::PackActorQHint(primary, primary, primary, intrinsic_q, intrinsic_value),
        Catch::Matchers::ContainsSubstring(name)
        && Catch::Matchers::ContainsSubstring(shape)
        && Catch::Matchers::ContainsSubstring(device)
        && Catch::Matchers::ContainsSubstring("expected_shape=[2]")
        && Catch::Matchers::ContainsSubstring("expected_device=cpu"));
}

TEST_CASE("DQNActionInfo regathers Actor Q hint after action replacement", "[dqn][per][actor_initial]")
{
    auto q_values = torch::tensor({
        { 1.0f, 5.0f, 2.0f },
        { 7.0f, 3.0f, 4.0f },
    });
    rl::AuxData aux;
    aux["q_values"] = q_values;
    aux["munchausen_terms"] = torch::zeros_like(q_values);
    auto packed = torch::tensor({ { 5.0f, 5.0f, 0.0f, 0.0f, 0.0f }, { 7.0f, 7.0f, 0.0f, 0.0f, 0.0f } });
    dqn::DQNActionInfo info(
        torch::tensor({ 1, 0 }, torch::TensorOptions().dtype(torch::kInt64)),
        {},
        aux,
        rl::ReplayInitialPriorityHint(packed));

    auto replaced = info.WithAction(torch::tensor({ 2, 1 }, torch::TensorOptions().dtype(torch::kInt64)));
    REQUIRE(replaced->GetReplayInitialPriorityHint().has_value());
    CHECK(torch::equal(
        replaced->GetReplayInitialPriorityHint()->GetPayload(),
        torch::tensor({ { 2.0f, 5.0f, 0.0f, 0.0f, 0.0f }, { 3.0f, 7.0f, 0.0f, 0.0f, 0.0f } })));
    const auto& first_cpu = replaced->GetReplayInitialPriorityHint()->GetPayloadCpu();
    const auto& second_cpu = replaced->GetReplayInitialPriorityHint()->GetPayloadCpu();
    CHECK(first_cpu.unsafeGetTensorImpl() == second_cpu.unsafeGetTensorImpl());
}

TEST_CASE("DefaultDQNAgentConfig accepts spatial exploration per Actor and disables it for targets", "[dqn][config][spatial]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.actor.[train].policy.spatial.enabled", "true");
    config_data.Set("DefaultDQNAgent.actor.[train].policy.spatial.scale", "linear");
    config_data.Set("DefaultDQNAgent.actor.[eval].policy.spatial.enabled", "true");
    config_data.Set("DefaultDQNAgent.target_policy.spatial.enabled", "true");

    dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(config_data));

    CHECK(config.actor.at("train").policy.spatial.enabled);
    CHECK(config.actor.at("eval").policy.spatial.enabled);
    CHECK_FALSE(config.target_policy.spatial.enabled);
}

TEST_CASE("DefaultDQNAgentConfig propagates quantile mode to every policy", "[dqn][iqn][config]")
{
    for (const auto& mode : { "none", "qr", "iqn" }) {
        INFO("mode=" << mode);
        anet::ConfigData config_data;
        config_data.Set("DefaultDQNAgent.quantile_mode", mode);
        config_data.Set("DefaultDQNAgent.actor.[train].network", "online");
        config_data.Set("DefaultDQNAgent.actor.[eval].network", "online");
        const dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(config_data));

        CHECK(config.actor.at("train").policy.quantile_mode == mode);
        CHECK(config.actor.at("eval").policy.quantile_mode == mode);
        CHECK(config.target_policy.quantile_mode == mode);
    }
}

TEST_CASE("DefaultDQNAgentConfig reads an optional IQN UQE full distribution query", "[dqn][iqn][config]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.actor.[train].network", "online");
    config_data.Set("DefaultDQNAgent.quantile_mode", "iqn");
    config_data.Set("DefaultDQNAgent.actor.[eval].policy.policy_type", "UQE");
    config_data.Set("DefaultDQNAgent.actor.[eval].policy.full_distribution_query.enabled", true);
    config_data.Set("DefaultDQNAgent.actor.[eval].policy.full_distribution_query.tau_rule.num_taus", 5);
    config_data.Set("DefaultDQNAgent.actor.[eval].policy.full_distribution_query.tau_rule.sample_mode", "random");

    const dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(config_data));

    CHECK_FALSE(config.actor.at("train").policy.full_distribution_query.enabled);
    CHECK(config.actor.at("eval").policy.full_distribution_query.enabled);
    CHECK(config.actor.at("eval").policy.full_distribution_query.tau_rule.num_taus == 5);
    CHECK(config.actor.at("eval").policy.full_distribution_query.tau_rule.sample_mode == "random");
    CHECK_FALSE(config.target_policy.full_distribution_query.enabled);
}

TEST_CASE("DefaultDQNAgentConfig restores deterministic target taus after optimistic policy copy", "[dqn][iqn][config]")
{
    anet::ConfigData inherited_data;
    inherited_data.Set("DefaultDQNAgent.quantile_mode", "iqn");
    inherited_data.Set("DefaultDQNAgent.use_optimistic_target", true);
    inherited_data.Set("DefaultDQNAgent.actor.[train].policy.policy_type", "UQE");
    inherited_data.Set("DefaultDQNAgent.actor.[train].policy.tau_rule.num_taus", 7);
    inherited_data.Set("DefaultDQNAgent.actor.[train].policy.tau_rule.sample_mode", "random");
    inherited_data.Set("DefaultDQNAgent.actor.[train].policy.full_distribution_query.enabled", true);
    inherited_data.Set("DefaultDQNAgent.actor.[train].policy.full_distribution_query.tau_rule.num_taus", 7);
    inherited_data.Set("DefaultDQNAgent.actor.[train].policy.full_distribution_query.tau_rule.sample_mode", "random");
    const dqn::DefaultDQNAgentConfig inherited(WithDefaultForwardConfig(inherited_data));

    CHECK(inherited.target_policy.policy_type == "UQE");
    CHECK(inherited.target_policy.tau_rule.num_taus == 32);
    CHECK(inherited.target_policy.tau_rule.sample_mode == "fixed");
    CHECK_FALSE(inherited.target_policy.full_distribution_query.enabled);
    CHECK(inherited.target_policy.full_distribution_query.tau_rule.num_taus == 32);
    CHECK(inherited.target_policy.full_distribution_query.tau_rule.sample_mode == "fixed");

    auto explicit_data = inherited_data;
    explicit_data.Set("DefaultDQNAgent.target_policy.tau_rule.num_taus", 5);
    explicit_data.Set("DefaultDQNAgent.target_policy.tau_rule.sample_mode", "random");
    explicit_data.Set("DefaultDQNAgent.target_policy.full_distribution_query.enabled", true);
    explicit_data.Set("DefaultDQNAgent.target_policy.full_distribution_query.tau_rule.num_taus", 5);
    explicit_data.Set("DefaultDQNAgent.target_policy.full_distribution_query.tau_rule.sample_mode", "random");
    const dqn::DefaultDQNAgentConfig explicit_target(WithDefaultForwardConfig(explicit_data));

    CHECK(explicit_target.target_policy.tau_rule.num_taus == 5);
    CHECK(explicit_target.target_policy.tau_rule.sample_mode == "random");
    CHECK(explicit_target.target_policy.full_distribution_query.enabled);
    CHECK(explicit_target.target_policy.full_distribution_query.tau_rule.num_taus == 5);
    CHECK(explicit_target.target_policy.full_distribution_query.tau_rule.sample_mode == "random");
}

TEST_CASE("DefaultDQNAgentConfig requires explicit Actor entries and optional snapshot schedules", "[dqn][config][snapshot]")
{
    const dqn::DefaultDQNAgentConfig defaults(WithDefaultForwardConfig());
    CHECK(defaults.actor.empty());
    anet::ConfigData data;
    data.Set("DefaultDQNAgent.actor.[train].clone_model", false);
    const dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(data));
    CHECK_FALSE(config.actor.at("train").clone_model);
    CHECK_FALSE(config.actor.at("train").sync_interval.has_value());
}

TEST_CASE("DefaultDQNAgentConfig rejects malformed Train Actor snapshot values", "[dqn][config][snapshot]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.actor.[train].clone_model", "false");
    config_data.Set("DefaultDQNAgent.actor.[train].sync_interval", "400x");

    CHECK_THROWS_WITH(
        dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(config_data)),
        Catch::Matchers::ContainsSubstring("DefaultDQNAgent.actor.[train].sync_interval")
            && Catch::Matchers::ContainsSubstring("400x")
            && Catch::Matchers::ContainsSubstring("expected=uint64_t"));
}

TEST_CASE("DefaultDQNAgentConfig reads and validates policy churn settings", "[dqn][config][policy_churn]")
{
    const dqn::DefaultDQNAgentConfig default_config(WithDefaultForwardConfig(anet::ConfigData{}));
    CHECK(default_config.learner.policy_churn.probe.batch_size == 1024);
    CHECK(default_config.learner.policy_churn.iqn.num_taus == 32);
    CHECK(default_config.learner.quantile_mode == default_config.quantile_mode);

    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.learner.policy_churn.probe.batch_size", 64);
    config_data.Set("DefaultDQNAgent.learner.policy_churn.iqn.num_taus", 8);
    const dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(config_data));
    CHECK(config.learner.policy_churn.probe.batch_size == 64);
    CHECK(config.learner.policy_churn.iqn.num_taus == 8);

    config_data.Set("DefaultDQNAgent.learner.policy_churn.probe.batch_size", 0);
    CHECK_THROWS_WITH(
        dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(config_data)),
        Catch::Matchers::ContainsSubstring("DefaultDQNAgent.learner.policy_churn.probe.batch_size=0; expected integer >= 1."));

    config_data.Set("DefaultDQNAgent.learner.policy_churn.probe.batch_size", 64);
    config_data.Set("DefaultDQNAgent.learner.policy_churn.iqn.num_taus", 0);
    CHECK_THROWS_WITH(
        dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(config_data)),
        Catch::Matchers::ContainsSubstring("DefaultDQNAgent.learner.policy_churn.iqn.num_taus=0; expected integer >= 1."));
}

TEST_CASE("Spatial exploration reports scalar means across policy updates", "[dqn][action_policy][spatial]")
{
    dqn::ActionPolicyConfig config;
    config.spatial.enabled = true;
    config.spatial.scale = "linear";
    config.epsilon.value = 1.0f;
    config.epsilon.start = 1.0f;
    config.spatial.epsilon.first.value = 1.0f;
    config.epsilon.end = 0.1f;
    config.spatial.epsilon.last.value = 0.1f;
    config.uqe.epsilon.value = 0.2f;
    config.uqe.epsilon.start = 0.2f;
    config.spatial.uqe.epsilon.first.value = 0.2f;
    config.uqe.epsilon.end = 0.0f;
    config.spatial.uqe.epsilon.last.value = 0.0f;
    config.uqe.tau.value = 0.0f;
    config.uqe.tau.start = 0.0f;
    config.spatial.uqe.tau.first.value = 0.0f;
    config.uqe.tau.end = 1.0f;
    config.spatial.uqe.tau.last.value = 1.0f;

    anet::test::LogCaptureGuard logs;
    dqn::EpsilonGreedyActionPolicy eps_policy(config, true, 2, torch::Device(torch::kCPU));
    dqn::UQEActionPolicy uqe_policy(config, true, 2, torch::Device(torch::kCPU));
    logs.Flush();
    CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 1);
    CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Warning,
        {"spatial exploration is enabled with num_envs < 32", "num_envs=2"}));

    rl::StepCounts counts;
    counts.exp_step = 1000000;
    eps_policy.UpdateSchedule(counts);
    uqe_policy.UpdateSchedule(counts);

    auto eps = eps_policy.GetScalar("epsilon");
    auto uqe_eps = uqe_policy.GetScalar("epsilon");
    auto uqe_tau = uqe_policy.GetScalar("uqe_tau");
    REQUIRE(eps.has_value());
    REQUIRE(uqe_eps.has_value());
    REQUIRE(uqe_tau.has_value());
    CHECK(*eps == Catch::Approx(0.55f));
    CHECK(*uqe_eps == Catch::Approx(0.1f));
    CHECK(*uqe_tau == Catch::Approx(0.5f));
}

TEST_CASE("Spatial UQE policies use per-env tau tensor", "[dqn][action_policy][spatial]")
{
    auto network = MakePassthroughNetwork(2, 2);
    auto obs = MakeSpatialUQEInput();

    dqn::ActionPolicyConfig config;
    config.spatial.enabled = true;
    config.spatial.scale = "linear";
    config.uqe.use_tail_mean = false;
    config.uqe.epsilon.value = 0.0f;
    config.uqe.epsilon.start = 0.0f;
    config.spatial.uqe.epsilon.first.value = 0.0f;
    config.uqe.epsilon.end = 0.0f;
    config.spatial.uqe.epsilon.last.value = 0.0f;
    config.uqe.tau.value = 0.0f;
    config.uqe.tau.start = 0.0f;
    config.spatial.uqe.tau.first.value = 0.0f;
    config.uqe.tau.end = 1.0f;
    config.spatial.uqe.tau.last.value = 1.0f;

    auto expected_actions = torch::tensor({ 1, 0 }, torch::TensorOptions().dtype(torch::kInt64));

    std::vector<std::pair<std::string, std::shared_ptr<dqn::ActionPolicy>>> policies;
    policies.emplace_back("uqe", std::make_shared<dqn::UQEActionPolicy>(config, true, 2, torch::Device(torch::kCPU)));
    policies.emplace_back("thompson-sampling", std::make_shared<dqn::ThompsonSamplingActionPolicy>(config, true, 2, torch::Device(torch::kCPU)));

    for (const auto& [name, policy] : policies) {
        INFO(name);
        auto rnd = std::make_shared<anet::RandomGenerator>(123);
        const anet::nn::NetworkForwardCallInfo call_info_14({});
        auto action_info = policy->SelectAction(obs, /*greedy_only=*/false, *MakeDefaultExecutor(network), rnd, call_info_14);
        CHECK(torch::equal(action_info->GetAction().cpu(), expected_actions));
    }
}

TEST_CASE("Replay fit policy evaluation suppresses inner AMP and stochastic auxiliary queries", "[dqn][replay_fit][policy]")
{
    std::vector<torch::Device> devices{ torch::kCPU };
    if (torch::cuda::is_available()) devices.emplace_back(torch::kCUDA, 0);
    for (const auto& device : devices) {
        const auto probe = std::make_shared<AutocastProbeState>();
        const auto network = MakeAutocastProbeNetwork(probe, device);
        dqn::ActionPolicyConfig config;
        config.use_amp = true;
        config.use_amp_bf16 = true;
        dqn::EpsilonGreedyActionPolicy greedy(config);
        auto random = std::make_shared<anet::RandomGenerator>(73031);
        const dqn::DiagnosticActionEvaluation evaluation{ .num_taus = 4 };
        {
            anet::Autocast outer(device, true, torch::kBFloat16);
            const anet::nn::NetworkForwardCallInfo call_info_15({});
            greedy.SelectAction(MakeAutocastProbePolicyInput(device), true, *MakeDefaultExecutor(network), random, call_info_15, {}, &evaluation);
        }
        CHECK(probe->forward_count == 1);
        CHECK(probe->enabled_count == 0);
        for (const bool tail : { false, true }) {
            const auto echo = std::make_shared<TauEchoState>();
            const auto iqn_network = MakeTauEchoNetwork(9, echo);
            iqn_network->to(device);
            config.quantile_mode = "iqn";
            config.uqe.use_tail_mean = tail;
            config.uqe.tau.value = 0.5f;
            config.uqe.tau.start = 0.5f;
            config.spatial.uqe.tau.first.value = 0.5f;
            config.uqe.tau.end = 0.5f;
            config.spatial.uqe.tau.last.value = 0.5f;
            config.tau_rule.num_taus = 9;
            config.tau_rule.sample_mode = "random";
            config.full_distribution_query.enabled = true;
            config.full_distribution_query.tau_rule.num_taus = 7;
            dqn::UQEActionPolicy policy(config);
            const auto rng_before = random->GetTorchGenerator(device).get_state();
            const anet::nn::NetworkForwardCallInfo call_info_16({});
            const auto result = policy.SelectAction(MakeTauEchoObservation().To(device), true, *MakeDefaultExecutor(iqn_network), random, call_info_16, {}, &evaluation);
            CHECK(echo->forward_count == 1);
            CHECK(echo->last_taus.size(1) == (tail ? 4 : 1));
            const auto expected = tail ? torch::tensor({ 0.5625f, 0.6875f, 0.8125f, 0.9375f }) : torch::tensor({ 0.5f });
            CHECK(torch::equal(echo->last_taus[0].cpu(), expected));
            CHECK(torch::equal(random->GetTorchGenerator(device).get_state(), rng_before));
            CHECK(result->GetAuxData().empty());
        }
    }
}

TEST_CASE("DQN policy churn observes target after hard synchronization", "[dqn][policy_churn][target]")
{
    // learn_step 1 は非 sync、learn_step 2 は hard sync となる network を準備する。
    TestNetworkModel model(
        /*distributional=*/false,
        /*n_actions=*/2,
        dqn::NetworkModelConfig{ .hard_update_interval = 2 });
    {
        torch::NoGradGuard no_grad;
        model.GetOnlineParameters().front().copy_(torch::tensor({
            { 1.0f, 0.0f },
            { 0.0f, 0.0f },
        }));
        model.GetOnlineNetwork()->CopyTo(*model.GetTargetNetwork());
    }
    dqn::RuntimeVars vars;
    vars.learn_step = 1;
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.replay_batch_size = 1;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = -1.0f;
    config.use_per = false;
    auto env_spec = MakeLearnerEnvSpec();
    env_spec.action_spec.value_labels = { "a0", "a1" };
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, dqn::LearnerForwardResources{});
    learner.UseSgd(1.0f);
    learner.EnablePolicyChurnOptimizerUpdate();
    auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
    replay_buffer->SetSizeValues({ 1 });
    replay_buffer->unique_sample_enabled = true;
    replay_buffer->vector_value = 1.0f;
    learner.UseReplayBuffer(replay_buffer);
    learner.ConfigureScalarMetricSubscriptions({
        rl::ScalarMetricSubscription{
            .source_key = "policy_churn_target_policy_disagreement",
            .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT,
            .interval = 1,
            .scope = rl::RunnerScope::TRAIN,
        },
        rl::ScalarMetricSubscription{
            .source_key = "policy_churn_target_q_delta_abs_mean",
            .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT,
            .interval = 1,
            .scope = rl::RunnerScope::TRAIN,
        },
        rl::ScalarMetricSubscription{
            .source_key = "policy_churn_target_sync_age",
            .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT,
            .interval = 1,
            .scope = rl::RunnerScope::TRAIN,
        },
    });

    // 非 sync update では既知差と age=1、次の sync update では厳密なゼロ点を返す。
    const auto non_sync = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{}).front();
    CHECK(*non_sync->GetScalar("policy_churn_target_policy_disagreement") == Catch::Approx(1.0f));
    CHECK(*non_sync->GetScalar("policy_churn_target_q_delta_abs_mean") == Catch::Approx(1.0f));
    CHECK(*non_sync->GetScalar("policy_churn_target_sync_age") == Catch::Approx(1.0f));

    const auto sync = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{}).front();
    CHECK(*sync->GetScalar("policy_churn_target_policy_disagreement") == Catch::Approx(0.0f));
    CHECK(*sync->GetScalar("policy_churn_target_q_delta_abs_mean") == Catch::Approx(0.0f));
    CHECK(*sync->GetScalar("policy_churn_target_sync_age") == Catch::Approx(0.0f));
}

TEST_CASE("DQN policy churn keeps unavailable known keys distinct from unknown keys", "[dqn][policy_churn][gate]")
{
    const auto make_learner = [](TestNetworkModel& model, dqn::RuntimeVars& vars) {
        dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
        config.replay_batch_size = 1;
        config.update_warmup_steps = 0;
        config.update_interval = 1;
        config.replay_ratio = -1.0f;
        config.use_per = false;
        auto env_spec = MakeLearnerEnvSpec();
        env_spec.action_spec.value_labels = { "a0", "a1" };
        return std::make_unique<TestLearner>(
            config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, dqn::LearnerForwardResources{});
    };

    SECTION("age-only on soft target performs no probe sampling") {
        TestNetworkModel model(/*distributional=*/false, /*n_actions=*/2);
        dqn::RuntimeVars vars;
        auto learner = make_learner(model, vars);
        learner->UseSgd(0.0f);
        learner->EnablePolicyChurnOptimizerUpdate();
        auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
        replay_buffer->SetSizeValues({ 1 });
        learner->UseReplayBuffer(replay_buffer);
        learner->ConfigureScalarMetricSubscriptions({ rl::ScalarMetricSubscription{
            .source_key = "policy_churn_target_sync_age",
            .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT,
            .interval = 1,
            .scope = rl::RunnerScope::TRAIN,
        } });

        const auto result = learner->UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{}).front();
        REQUIRE(result->GetScalar("policy_churn_target_sync_age").has_value());
        CHECK(std::isnan(*result->GetScalar("policy_churn_target_sync_age")));
        CHECK(replay_buffer->unique_sample_count == 0);
    }

    SECTION("incomplete probe publishes NaN and unknown key remains nullopt") {
        TestNetworkModel model(/*distributional=*/false, /*n_actions=*/2);
        dqn::RuntimeVars vars;
        auto learner = make_learner(model, vars);
        learner->UseSgd(0.0f);
        learner->EnablePolicyChurnOptimizerUpdate();
        auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
        replay_buffer->SetSizeValues({ 1 });
        replay_buffer->unique_sample_enabled = false;
        learner->UseReplayBuffer(replay_buffer);
        learner->ConfigureScalarMetricSubscriptions({ rl::ScalarMetricSubscription{
            .source_key = "policy_churn_action_ratio",
            .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT,
            .interval = 1,
            .scope = rl::RunnerScope::TRAIN,
        } });

        const auto result = learner->UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{}).front();
        REQUIRE(result->GetScalar("policy_churn_action_ratio").has_value());
        CHECK(std::isnan(*result->GetScalar("policy_churn_action_ratio")));
        CHECK_FALSE(result->GetScalar("policy_churn_unknown").has_value());
        CHECK(replay_buffer->unique_sample_count == 1);
    }
}

TEST_CASE("DQN policy churn uses post-soft-update target values", "[dqn][policy_churn][target][soft]")
{
    // soft tau=0.01 の update 後 target と online-after を比較する。
    TestNetworkModel model(/*distributional=*/false, /*n_actions=*/2);
    {
        torch::NoGradGuard no_grad;
        model.GetOnlineParameters().front().copy_(torch::tensor({
            { 1.0f, 0.0f },
            { 0.0f, 0.0f },
        }));
        model.GetOnlineNetwork()->CopyTo(*model.GetTargetNetwork());
    }
    dqn::RuntimeVars vars;
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.replay_batch_size = 1;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = -1.0f;
    config.use_per = false;
    auto env_spec = MakeLearnerEnvSpec();
    env_spec.action_spec.value_labels = { "a0", "a1" };
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, dqn::LearnerForwardResources{});
    learner.UseSgd(1.0f);
    learner.EnablePolicyChurnOptimizerUpdate();
    auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
    replay_buffer->SetSizeValues({ 1 });
    replay_buffer->unique_sample_enabled = true;
    replay_buffer->vector_value = 1.0f;
    learner.UseReplayBuffer(replay_buffer);
    learner.ConfigureScalarMetricSubscriptions({
        rl::ScalarMetricSubscription{
            .source_key = "policy_churn_target_policy_disagreement",
            .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT,
            .interval = 1,
            .scope = rl::RunnerScope::TRAIN,
        },
        rl::ScalarMetricSubscription{
            .source_key = "policy_churn_target_q_delta_abs_mean",
            .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT,
            .interval = 1,
            .scope = rl::RunnerScope::TRAIN,
        },
        rl::ScalarMetricSubscription{
            .source_key = "policy_churn_target_sync_age",
            .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT,
            .interval = 1,
            .scope = rl::RunnerScope::TRAIN,
        },
    });

    const auto result = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{}).front();
    CHECK(*result->GetScalar("policy_churn_target_policy_disagreement") == Catch::Approx(1.0f));
    CHECK(*result->GetScalar("policy_churn_target_q_delta_abs_mean") == Catch::Approx(0.99f));
    CHECK(std::isnan(*result->GetScalar("policy_churn_target_sync_age")));
    CHECK(replay_buffer->unique_sample_count == 1);
}

TEST_CASE("DQN policy churn warns once for a single hard-target phase", "[dqn][policy_churn][warning]")
{
    const auto make_subscription = [](int interval) {
        return rl::ScalarMetricSubscription{
            .source_key = "policy_churn_target_sync_age",
            .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT,
            .interval = interval,
            .scope = rl::RunnerScope::TRAIN,
        };
    };

    SECTION("single phase emits one aggregated warning") {
        TestNetworkModel model(
            /*distributional=*/false,
            /*n_actions=*/2,
            dqn::NetworkModelConfig{ .hard_update_interval = 503 });
        dqn::RuntimeVars vars;
        dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
        auto env_spec = MakeLearnerEnvSpec();
        env_spec.action_spec.value_labels = { "a0", "a1" };
        TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, dqn::LearnerForwardResources{});
        anet::test::LogCaptureGuard logs;
        learner.ConfigureScalarMetricSubscriptions({ make_subscription(503) });
        learner.ConfigureScalarMetricSubscriptions({ make_subscription(503) });
        logs.Flush();

        int matches = 0;
        for (const auto& record : logs.Records()) {
            if (record.level != wxLOG_Warning
                || record.message.find("Policy churn metrics observe one hard-target phase") == std::string::npos) {
                continue;
            }
            ++matches;
            CHECK(record.message.find("target_sync_age=0") != std::string::npos);
            CHECK(record.message.find("metrics_interval=503") != std::string::npos);
            CHECK(record.message.find("hard_update_interval (not read from config)=503") != std::string::npos);
        }
        CHECK(matches == 1);
    }

    SECTION("multiple phases emit no warning") {
        TestNetworkModel model(
            /*distributional=*/false,
            /*n_actions=*/2,
            dqn::NetworkModelConfig{ .hard_update_interval = 500 });
        dqn::RuntimeVars vars;
        dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
        auto env_spec = MakeLearnerEnvSpec();
        env_spec.action_spec.value_labels = { "a0", "a1" };
        TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, dqn::LearnerForwardResources{});
        anet::test::LogCaptureGuard logs;
        learner.ConfigureScalarMetricSubscriptions({ make_subscription(503) });
        logs.Flush();

        CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 0);
    }
}

TEST_CASE("DQN policy churn disables outer BF16 autocast for probe forwards", "[dqn][policy_churn][amp][bf16]")
{
    const auto device = torch::Device(torch::kCPU);
    auto probe_state = std::make_shared<AutocastProbeState>();
    AutocastProbeNetworkModel model(probe_state, device, /*n_actions=*/2);
    {
        torch::NoGradGuard no_grad;
        // BF16では更新前後とも両actionが1.0へ丸まる近接Qを作る。
        model.GetOnlineParameters().at(0).copy_(torch::tensor({
            { 0.5010f, 0.5010f },
            { 0.5005f, 0.5005f },
        }));
        model.GetOnlineNetwork()->CopyTo(*model.GetTargetNetwork());
    }
    dqn::RuntimeVars vars;
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.use_amp = true;
    config.use_amp_bf16 = true;
    config.use_grad_clip = false;
    config.replay_batch_size = 1;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = -1.0f;
    config.use_per = false;
    auto env_spec = MakeLearnerEnvSpec();
    env_spec.action_spec.value_labels = { "a0", "a1" };
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, dqn::LearnerForwardResources{});
    learner.UseSgd(0.001f);
    learner.EnablePolicyChurnOptimizerUpdate();
    auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
    replay_buffer->SetSizeValues({ 1 });
    replay_buffer->unique_sample_enabled = true;
    replay_buffer->vector_value = 1.0f;
    learner.UseReplayBuffer(replay_buffer);
    std::vector<rl::ScalarMetricSubscription> subscriptions;
    for (const auto* key : std::span(dqn::kPolicyChurnMetricKeys).first(dqn::PolicyChurnMetric::TargetPolicyDisagreement)) {
        subscriptions.push_back(rl::ScalarMetricSubscription{
            .source_key = key,
            .event = rl::EventType::LEARN,
            .target = rl::EventField::UPDATE_RESULT,
            .interval = 1,
            .scope = rl::RunnerScope::TRAIN,
        });
    }
    learner.ConfigureScalarMetricSubscriptions(subscriptions);

    anet::Autocast outer_autocast(device, true, torch::kBFloat16);
    const auto result = learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{}).front();
    // FP32では [1.002, 1.001] -> [1.002, 1.003] と反転し、BF16なら差が消える。
    CHECK(*result->GetScalar("policy_churn_action_ratio") == Catch::Approx(1.0f));
    CHECK(*result->GetScalar("policy_churn_q_delta_abs_mean") == Catch::Approx(0.001f).margin(1.0e-6f));
    CHECK(*result->GetScalar("policy_churn_q_delta_signed_max") == Catch::Approx(0.002f).margin(1.0e-6f));
    CHECK(*result->GetScalar("policy_churn_q_delta_signed_min") == Catch::Approx(0.0f).margin(1.0e-6f));
    CHECK(probe_state->forward_count == 3);
    CHECK(probe_state->enabled_count == 1);
}

TEST_CASE("DQN policy churn follows subscription cadence and stays inactive without demand", "[dqn][policy_churn][gate]")
{
    const auto run_case = [](std::optional<int> interval) {
        TestNetworkModel model(/*distributional=*/false, /*n_actions=*/2);
        dqn::RuntimeVars vars;
        dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
        config.replay_batch_size = 1;
        config.update_warmup_steps = 0;
        config.update_interval = 1;
        config.replay_ratio = -1.0f;
        config.use_per = false;
        auto env_spec = MakeLearnerEnvSpec();
        env_spec.action_spec.value_labels = { "a0", "a1" };
        TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec, dqn::LearnerForwardResources{});
        learner.UseSgd(0.0f);
        learner.EnablePolicyChurnOptimizerUpdate();
        auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
        replay_buffer->SetSizeValues({ 1 });
        replay_buffer->unique_sample_enabled = true;
        replay_buffer->vector_value = 1.0f;
        learner.UseReplayBuffer(replay_buffer);
        if (interval.has_value()) {
            learner.ConfigureScalarMetricSubscriptions({ rl::ScalarMetricSubscription{
                .source_key = "policy_churn_action_ratio",
                .event = rl::EventType::LEARN,
                .target = rl::EventField::UPDATE_RESULT,
                .interval = *interval,
                .scope = rl::RunnerScope::TRAIN,
            } });
        } else {
            learner.ConfigureScalarMetricSubscriptions({});
        }

        std::vector<std::shared_ptr<const rl::BatchUpdateResult>> results;
        for (int i = 0; i < 4; ++i) {
            results.push_back(learner.UpdateFromBatch(rl::StepCounts{}, rl::BatchExperience{}).front());
        }
        return std::pair{ replay_buffer->unique_sample_count, std::move(results) };
    };

    const auto [sample_count, cadence_results] = run_case(3);
    CHECK(sample_count == 2);
    CHECK(*cadence_results[0]->GetScalar("policy_churn_action_ratio") == Catch::Approx(0.0f));
    CHECK(std::isnan(*cadence_results[1]->GetScalar("policy_churn_action_ratio")));
    CHECK(std::isnan(*cadence_results[2]->GetScalar("policy_churn_action_ratio")));
    CHECK(*cadence_results[3]->GetScalar("policy_churn_action_ratio") == Catch::Approx(0.0f));

    const auto [inactive_sample_count, inactive_results] = run_case(std::nullopt);
    CHECK(inactive_sample_count == 0);
    CHECK(std::isnan(*inactive_results.front()->GetScalar("policy_churn_action_ratio")));
}

TEST_CASE("QR UQE preserves sorted point and tail scores", "[dqn][qr][uqe][munchausen]")
{
    // 非整列の固定分位点から、既存hard経路の点選択とtail平均を固定する。
    const auto quantiles = torch::tensor({ { { 7.0f, 1.0f, 3.0f, 5.0f }, { 0.0f, 20.0f, 10.0f, 2.0f } } });
    const anet::TensorDict obs{ { "q", quantiles.mean(-1) }, { "q_dist", quantiles } };
    for (const bool tail_mean : { false, true }) {
        dqn::ActionPolicyConfig config;
        config.quantile_mode = "qr";
        config.uqe.tau.value = 0.5f;
        config.uqe.tau.start = 0.5f;
        config.spatial.uqe.tau.first.value = 0.5f;
        config.uqe.tau.end = 0.5f;
        config.spatial.uqe.tau.last.value = 0.5f;
        config.uqe.use_tail_mean = tail_mean;
        dqn::UQEActionPolicy policy(config);
        const anet::nn::NetworkForwardCallInfo call_info_17({});
        const auto result = policy.SelectAction(obs, true, *MakeDefaultExecutor(MakePassthroughNetwork(2, 4)),
            std::make_shared<anet::RandomGenerator>(67001), call_info_17, {});
        const auto expected = tail_mean ? torch::tensor({ { 5.0f, 32.0f / 3.0f } })
                                        : torch::tensor({ { 3.0f, 2.0f } });
        CHECK(torch::equal(result->GetAuxData().at("uqe_values"), expected));
        CHECK(result->GetAction().item<int64_t>() == (tail_mean ? 1 : 0));
    }
}

TEST_CASE("DefaultDQNAgent forwards Munchausen settings to Actor approximation", "[dqn][munchausen][actor_munchausen]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    const bool tbo = GENERATE(false, true);
    const bool tail_mean = GENERATE(false, true);
    CAPTURE(tbo, tail_mean);
    const auto make_action = [&](bool enabled) {
        auto data = MakeIqnTracerConfigData();
        data.Set("DefaultDQNAgent.learner.munchausen.enabled", enabled);
        data.Set("DefaultDQNAgent.learner.munchausen.entropy_tau", 0.7f);
        data.Set("DefaultDQNAgent.learner.munchausen.clip_value_min", -2.0f);
        data.Set("DefaultDQNAgent.learner.use_double_dqn", false);
        data.Set("DefaultDQNAgent.learner.use_tbo", tbo);
        data.Set("DefaultDQNAgent.learner.tbo_epsilon", 0.001f);
        data.Set("DefaultDQNAgent.learner.use_per", true);
        data.Set("DefaultDQNAgent.learner.per_initial_priority_mode", "actor_approx");
        data.Set("DefaultDQNAgent.actor.[train].policy.policy_type", "UQE");
        data.Set("DefaultDQNAgent.actor.[train].policy.uqe.tau", 0.65f);
        data.Set("DefaultDQNAgent.actor.[train].policy.uqe.tau.start", 0.65f);
        data.Set("DefaultDQNAgent.actor.[train].policy.uqe.tau.end", 0.65f);
        data.Set("DefaultDQNAgent.actor.[train].policy.uqe.use_tail_mean", tail_mean);
        data.Set("DefaultDQNAgent.actor.[train].policy.full_distribution_query.enabled", true);
        data.Set("DefaultDQNAgent.actor.[train].policy.full_distribution_query.tau_rule.num_taus", 5);
        dqn::DefaultDQNAgentFactory factory;
        const auto spec = MakeIqnTracerEnvSpec();
        const rl::BatchEnvSpec batch_spec{ 2, 1 };
        const auto agent = factory.CreateAgent(spec, batch_spec, torch::kCPU, WithDefaultForwardConfig(data), nullptr, 67020);
        const auto actor = agent->CreateActor(rl::ActorRequest{.batch_env_spec = batch_spec, .env_spec = spec, .device = torch::kCPU, .seed = 123, .actor_key = "train"});
        const auto flags = torch::zeros({ 2 }, torch::kBool);
        const rl::BatchState state(anet::TensorDict{ { kVectorKey,
            torch::tensor({ { 1.0f, 2.0f, 3.0f, 4.0f }, { 4.0f, 3.0f, 2.0f, 1.0f } }) } }, flags, flags, flags);
        return actor->MakeAction(rl::StepCounts{}, state);
    };
    const auto off = make_action(false);
    const auto on = make_action(true);
    REQUIRE(off->GetReplayInitialPriorityHint().has_value());
    REQUIRE(on->GetReplayInitialPriorityHint().has_value());
    CHECK(torch::equal(off->GetReplayInitialPriorityHint()->GetPayload().select(1, 2), torch::zeros({ 2 })));
    const auto payload = on->GetReplayInitialPriorityHint()->GetPayload();
    const auto scores = on->GetAuxData().at("q_values").to(torch::kFloat64);
    REQUIRE(on->GetAuxData().contains("full_q_values"));
    const auto h = [](double x) { return std::copysign(std::sqrt(std::abs(x) + 1.0) - 1.0, x) + 0.001 * x; };
    const auto h_inv = [](double x) {
        const double t = (std::sqrt(1.0 + 0.004 * (std::abs(x) + 1.001)) - 1.0) / 0.002;
        return std::copysign(t * t - 1.0, x);
    };
    // full queryの平均ではなく、既存risk scoreを実空間へ戻した近似を独立式で検証する。
    for (int b = 0; b < 2; ++b) {
        std::array<double, 2> real{};
        for (int a = 0; a < 2; ++a) real[a] = tbo ? h_inv(scores[b][a].item<double>()) : scores[b][a].item<double>();
        const double maximum = std::max(real[0], real[1]);
        const double log_z = std::log(std::exp((real[0] - maximum) / 0.7) + std::exp((real[1] - maximum) / 0.7));
        const double soft = maximum + 0.7 * log_z;
        const int64_t action = on->GetAction()[b].item<int64_t>();
        const double bonus = 0.9 * std::clamp(real[action] - maximum - 0.7 * log_z, -2.0, 0.0);
        CHECK(payload[b][0].item<float>() == Catch::Approx(scores[b][action].item<double>()));
        CHECK(payload[b][1].item<float>() == Catch::Approx(tbo ? h(soft) : soft).margin(3e-4));
        CHECK(payload[b][2].item<float>() == Catch::Approx(bonus).margin(3e-4));
    }
}

TEST_CASE("IQN Actor catalogs are independent of train and of other Actor clocks and RNGs", "[dqn][prd061][actor_catalog]")
{
    const bool clone = GENERATE(false, true);
    ScopedNoopMetricsLogger metrics_logger;
    anet::test::RegisterStandardNetworkModules();
    anet::ConfigData data;
    const auto original = MakeIqnTracerConfigData();
    for (const auto& [key, value] : original.Map()) {
        if (!key.starts_with("DefaultDQNAgent.actor.")) data.Set(key, value);
    }
    data.Set("DefaultDQNAgent.learner.enabled", false);
    data.Set("DefaultDQNAgent.learner.iqn.current_taus.num_taus", 7);
    for (const auto& key : {"small", "large"}) {
        const auto prefix = std::string("DefaultDQNAgent.actor.[") + key + "]";
        data.Set(prefix + ".clone_model", clone);
        data.Set(prefix + ".policy.policy_type", "EpsilonGreedy");
        data.Set(prefix + ".policy.epsilon.type", "linear");
        data.Set(prefix + ".policy.epsilon.start", 1.0f);
        data.Set(prefix + ".policy.epsilon.end", 0.0f);
        data.Set(prefix + ".policy.epsilon.steps", 100);
        data.Set(prefix + ".policy.tau_rule.sample_mode", "random");
        data.Set(prefix + ".policy.tau_rule.num_taus", std::string(key) == "small" ? 3 : 5);
    }
    const dqn::DefaultDQNAgentConfig typed(WithDefaultForwardConfig(data));
    CHECK_FALSE(typed.actor.contains("train"));
    CHECK(typed.GetConfigData().Get("actor.[small].network") == "online");
    dqn::DefaultDQNAgentFactory factory;
    auto agent = factory.CreateAgent(MakeIqnTracerEnvSpec(), {2, 1}, torch::kCPU, WithDefaultForwardConfig(data), nullptr, 123);
    const auto create = [&](const std::string& key, anet::seed_t seed) {
        return agent->CreateActor(rl::ActorRequest{.batch_env_spec = {2, 1}, .env_spec = MakeIqnTracerEnvSpec(),
            .device = torch::kCPU, .seed = seed, .actor_key = key});
    };
    CHECK_THROWS_WITH(create("undefined", 1),
        Catch::Matchers::ContainsSubstring("Defined keys=[DefaultDQNAgent.actor.[large], DefaultDQNAgent.actor.[small]]"));
    auto first = create("small", 456);
    auto replay = create("small", 456);
    auto other = create("large", 789);
    auto flags = torch::zeros({2}, torch::kBool);
    rl::BatchState state(anet::TensorDict{{kVectorKey, torch::ones({2, 4})}}, flags, flags, flags);
    rl::StepCounts counts;
    counts.exp_step = 25;
    auto first_info = std::dynamic_pointer_cast<dqn::DQNActionInfo>(first->MakeAction(counts, state));
    REQUIRE(first_info != nullptr);
    CHECK(ShapeOf(first_info->GetAuxData().at("q_quantiles")) == std::vector<int64_t>{2, 2, 3});
    CHECK(*first->GetScalar("epsilon") == Catch::Approx(0.75f));
    CHECK_FALSE(first->GetScalar("unknown").has_value());
    counts.exp_step = 100;
    auto other_info = std::dynamic_pointer_cast<dqn::DQNActionInfo>(other->MakeAction(counts, state));
    CHECK(ShapeOf(other_info->GetAuxData().at("q_quantiles")) == std::vector<int64_t>{2, 2, 5});
    CHECK(*other->GetScalar("epsilon") == Catch::Approx(0.0f));
    CHECK(*first->GetScalar("epsilon") == Catch::Approx(0.75f));
    counts.exp_step = 25;
    auto replay_info = std::dynamic_pointer_cast<dqn::DQNActionInfo>(replay->MakeAction(counts, state));
    CHECK(torch::equal(first_info->GetAction(), replay_info->GetAction()));
    CHECK(torch::equal(first_info->GetAuxData().at("q_values"), replay_info->GetAuxData().at("q_values")));
    for (int i = 0; i < 16; ++i) {
        auto extra = create("large", 1000 + i);
        extra->MakeAction(counts, state);
        auto left = std::dynamic_pointer_cast<dqn::DQNActionInfo>(first->MakeAction(counts, state));
        other->MakeAction(counts, state);
        auto right = std::dynamic_pointer_cast<dqn::DQNActionInfo>(replay->MakeAction(counts, state));
        CHECK(torch::equal(left->GetAction(), right->GetAction()));
        CHECK(torch::equal(left->GetAuxData().at("q_values"), right->GetAuxData().at("q_values")));
    }
    data.Set("DefaultDQNAgent.use_optimistic_target", true);
    CHECK_THROWS_WITH(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(data)), Catch::Matchers::ContainsSubstring("requires DefaultDQNAgent.actor.[train].policy"));
}

TEST_CASE("Optimistic target warns for a non optimistic Actor source and preserves explicit overrides", "[dqn][prd061][config]")
{
    anet::ConfigData data;
    data.Set("DefaultDQNAgent.use_optimistic_target", true);
    data.Set("DefaultDQNAgent.actor.[train].policy.policy_type", "EpsilonGreedy");
    {
        anet::test::LogCaptureGuard logs;
        const dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(data));
        CHECK_NOTHROW(dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(data)));
        logs.Flush();
        CHECK(config.target_policy.policy_type == "Greedy");
        CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 1);
        CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Warning,
            {"use_optimistic_target=true", "actor.[train].policy.policy_type", "EpsilonGreedy", "UQE/ThompsonSampling"}));
    }
    data.Set("DefaultDQNAgent.actor.[train].policy.policy_type", "UQE");
    data.Set("DefaultDQNAgent.target_policy.uqe.tau", 0.7f);
    data.Set("DefaultDQNAgent.target_policy.uqe.tau.start", 0.7f);
    {
        anet::test::LogCaptureGuard logs;
        const dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(data));
        logs.Flush();
        CHECK(config.target_policy.policy_type == "UQE");
        CHECK(config.target_policy.uqe.tau.value == Catch::Approx(0.7f));
        CHECK_FALSE(anet::test::HasRecordContaining(logs.Records(), wxLOG_Warning, {"use_optimistic_target=true"}));
    }
}

TEST_CASE("Noisy probe metrics share draws and average action entropy per state", "[dqn][noisynet][probe]")
{
    const auto mean = torch::tensor({{3.0f, 1.0f}, {1.0f, 2.0f}});
    const std::vector<torch::Tensor> draws{
        torch::tensor({{4.0f, 0.0f}, {1.0f, 3.0f}}),
        torch::tensor({{0.0f, 2.0f}, {4.0f, 1.0f}})};
    int calls = 0;
    const auto values = dqn::MeasureNoisyQProbe(mean, 4, [&](int draw) {
        CHECK(draw == calls++);
        return draws[draw % 2];
    });
    CHECK(calls == 4);
    CHECK(values[0] == Catch::Approx(1.375f));
    CHECK(values[1] == Catch::Approx(0.5f));
    CHECK(values[2] == Catch::Approx(std::log(2.0f)));
    // 状態間で行動が異なっても、各状態で反復行動が固定なら entropy はゼロ。
    const auto fixed = dqn::MeasureNoisyQProbe(mean, 3, [&](int) { return mean; });
    CHECK(fixed[0] == 0.0f);
    CHECK(fixed[1] == 0.0f);
    CHECK(fixed[2] == 0.0f);
}

TEST_CASE("PRD101 policy reads a phased exponential epsilon and turns exploration off", "[dqn][prd101]")
{
    anet::ConfigData data;
    data.Set("DefaultDQNAgent.nn_forward.[mean].noisy.mode", "mu_only");
    data.Set("DefaultDQNAgent.actor.[train].nn_forward_key", "mean");
    data.Set("DefaultDQNAgent.learner.current.nn_forward_key", "mean");
    data.Set("DefaultDQNAgent.learner.target.nn_forward_key", "mean");
    const std::string key = "DefaultDQNAgent.actor.[train].policy.epsilon";
    data.Set(key + ".type", "phased");
    data.Set(key + ".phases", "decay off");
    data.Set(key + ".phase.[decay].type", "exponential");
    data.Set(key + ".phase.[decay].start", "1");
    data.Set(key + ".phase.[decay].end", "0.01");
    data.Set(key + ".phase.[decay].time_constant", "10");
    data.Set(key + ".phase.[decay].steps", "25");
    data.Set(key + ".phase.[off].value", "0");
    data.Set(key + ".phase.[off].steps", "1");
    const anet::rl::dqn::DefaultDQNAgentConfig config(data);
    anet::rl::dqn::EpsilonGreedyActionPolicy policy(config.actor.at("train").policy);
    policy.UpdateSchedule(anet::rl::StepCounts{.exp_step = 10});
    REQUIRE(policy.GetScalar("epsilon").has_value());
    CHECK(*policy.GetScalar("epsilon") == Catch::Approx(0.01 + 0.99 * std::exp(-1.0)));
    policy.UpdateSchedule(anet::rl::StepCounts{.exp_step = 25});
    CHECK(*policy.GetScalar("epsilon") == 0.0f);
}

TEST_CASE("PRD101 spatial log ladder preserves zero endpoints and follows its schedules", "[dqn][prd101][spatial]")
{
    anet::ConfigData data;
    const std::string key = "DefaultDQNAgent.actor.[train].policy.";
    data.Set(key + "spatial.enabled", "true");
    data.Set(key + "spatial.scale", "log");
    data.Set(key + "spatial.epsilon.first.type", "linear");
    data.Set(key + "spatial.epsilon.first.start", "0.4");
    data.Set(key + "spatial.epsilon.first.end", "0");
    data.Set(key + "spatial.epsilon.first.steps", "10");
    data.Set(key + "spatial.epsilon.last", "0");
    const dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(data));
    dqn::EpsilonGreedyActionPolicy policy(config.actor.at("train").policy, true, 4);
    policy.UpdateSchedule(rl::StepCounts{.exp_step = 0});
    REQUIRE(policy.GetScalar("epsilon_lane_min").has_value());
    CHECK(*policy.GetScalar("epsilon_lane_min") == 0.0f);
    CHECK(*policy.GetScalar("epsilon_lane_max") == Catch::Approx(0.4));
    CHECK(*policy.GetScalar("epsilon") == Catch::Approx((0.4 + 0.02510469717 + 0.001487772995) / 4));
    policy.UpdateSchedule(rl::StepCounts{.exp_step = 10});
    CHECK(*policy.GetScalar("epsilon") == 0.0f);
}

// D8の列生成と線形伸縮を倍精度で独立計算する（本体のexpm1式は使わない）。
static double D8LadderValue(double first, double last, double t)
{
    const double a = std::max(first, static_cast<double>(1e-4f));
    const double b = std::max(last, static_cast<double>(1e-4f));
    if (a == b) return first + t * (last - first);
    const double raw = b * std::exp((1.0 - t) * std::log(a / b));
    return last + (first - last) * (raw - b) / (a - b);
}

TEST_CASE("PRD101 spatial endpoints enforce bounds without caller metadata", "[prd101][dqn]")
{
    dqn::ActionPolicyConfig::LadderConfig config;
    config.first.value = 0.4f;
    config.last.value = 0.01f;
    SECTION("negative first") { config.first.value = -0.1f; }
    SECTION("last above one") { config.last.value = 1.1f; }
    SECTION("negative phase endpoint") {
        config.last.type = "phased";
        config.last.phases = {"bad"};
        config.last.phase.Set("bad", anet::ProfiledValuePhaseConfig<float>{.value = -0.1f, .steps = 8});
    }
    CHECK_THROWS(dqn::SpatialLadderSchedule(config, "log", 4, torch::kCPU));
}

TEST_CASE("PRD101 spatial constants reuse storage and moving endpoints replace it", "[prd101][dqn]")
{
    dqn::ActionPolicyConfig::LadderConfig config;
    config.first.value = 0.4f;
    config.last.value = 0.01f;
    dqn::SpatialLadderSchedule constant(config, "log", 4, torch::kCPU);
    const auto old_constant = constant.Tensor();
    constant.Update(100);
    CHECK(constant.Tensor().data_ptr() == old_constant.data_ptr());
    config.first.type = "linear";
    config.first.start = 0.4f;
    config.first.end = 0.2f;
    config.first.steps = 8;
    dqn::SpatialLadderSchedule moving(config, "log", 4, torch::kCPU);
    const auto old_moving = moving.Tensor();
    moving.Update(4);
    CHECK(moving.Tensor().data_ptr() != old_moving.data_ptr());
    CHECK_FALSE(torch::equal(moving.Tensor(), old_moving));
}

TEST_CASE("PRD101 spatial log endpoints remain continuous across the floor", "[prd101][dqn]")
{
    dqn::ActionPolicyConfig::LadderConfig config;
    config.first.value = 0.4f;
    config.last.type = "linear";
    config.last.start = 1e-4f - 1e-9f;
    config.last.end = 1e-4f + 1e-9f;
    config.last.steps = 1;
    dqn::SpatialLadderSchedule ladder(config, "log", 8, torch::kCPU);
    const auto before = ladder.Tensor();
    ladder.Update(1);
    CHECK((ladder.Tensor() - before).abs().max().item<float>() < 1e-6f);
}

TEST_CASE("PRD101 spatial first crosses the log floor without cancellation", "[prd101][dqn]")
{
    const auto check_device = [&](torch::Device device) {
        dqn::ActionPolicyConfig::LadderConfig config;
        config.last.value = 0.0f;
        config.first.type = "phased";
        config.first.phases = {"above", "floor", "below"};
        config.first.phase.Set("above", anet::ProfiledValuePhaseConfig<float>{.value = 1.0001e-4f, .steps = 1});
        config.first.phase.Set("floor", anet::ProfiledValuePhaseConfig<float>{.value = 1e-4f, .steps = 1});
        config.first.phase.Set("below", anet::ProfiledValuePhaseConfig<float>{.value = 5e-5f, .steps = 1});
        dqn::SpatialLadderSchedule ladder(config, "log", 3, device);
        const float firsts[] = {1.0001e-4f, 1e-4f, 5e-5f};
        float previous_middle = 0.0f;
        for (int64_t step = 0; step < 3; ++step) {
            CAPTURE(device, step);
            ladder.Update(step);
            const auto values = ladder.Tensor().to(torch::kCPU);
            CHECK(values[0].item<float>() == 0.0f);
            CHECK(values[2].item<float>() == firsts[step]);
            const double expected = D8LadderValue(firsts[step], 0.0f, 0.5);
            const double tolerance = 32.0 * std::numeric_limits<float>::epsilon() * expected;
            const float middle = values[1].item<float>();
            CHECK(std::abs(middle - expected) <= tolerance);
            if (step == 1) CHECK(std::abs(middle - previous_middle) < 1e-8f);
            previous_middle = middle;
        }
    };
    SECTION("CPU") { check_device(torch::kCPU); }
    SECTION("CUDA") {
        if (!torch::cuda::is_available()) { SKIP("CUDA is unavailable"); }
        check_device(torch::kCUDA);
    }
}

TEST_CASE("PRD101 spatial log ladder preserves a tiny last endpoint", "[prd101][dqn]")
{
    const auto check_device = [](torch::Device device) {
        dqn::ActionPolicyConfig::LadderConfig config;
        config.first.value = 0.4f;
        config.last.value = 5e-5f;
        dqn::SpatialLadderSchedule ladder(config, "log", 4, device);
        const auto values = ladder.Tensor().to(torch::kCPU);
        // 既存の逆順lane割当てを維持し、指定端点とD8から求めた中間値を確認する。
        CHECK(values[0].item<float>() == config.last.value);
        CHECK(values[3].item<float>() == config.first.value);
        double expected_sum = 0.0;
        for (int64_t i = 0; i < 4; ++i) {
            CAPTURE(device, i);
            const double expected = D8LadderValue(config.first.value, config.last.value, i / 3.0);
            const double tolerance = 32.0 * std::numeric_limits<float>::epsilon() * expected;
            CHECK(std::abs(values[3 - i].item<float>() - expected) <= tolerance);
            expected_sum += expected;
        }
        CHECK(std::abs(ladder.Mean() - expected_sum / 4.0)
            <= 32.0 * std::numeric_limits<float>::epsilon() * expected_sum / 4.0);
    };
    SECTION("CPU") { check_device(torch::kCPU); }
    SECTION("CUDA") {
        if (!torch::cuda::is_available()) { SKIP("CUDA is unavailable"); }
        check_device(torch::kCUDA);
    }
}

TEST_CASE("PRD101 spatial validation reports keys values and expected ranges", "[prd101][config][dqn]")
{
    anet::ConfigData data;
    dqn::ActionPolicyConfig policy;
    std::string key, value, expected;
    SECTION("invalid scale") {
        key = "spatial.scale"; value = "invalid"; expected = "expected=log|linear";
        data.Set("policy." + key, value);
        CHECK_THROWS_WITH(dqn::SpatialLadderSchedule(policy.spatial.epsilon, value, 4, torch::kCPU),
            Catch::Matchers::ContainsSubstring(key) && Catch::Matchers::ContainsSubstring(value)
            && Catch::Matchers::ContainsSubstring(expected));
    }
    SECTION("negative first") {
        key = "spatial.epsilon.first"; value = "-0.1"; expected = "expected=>=0";
        policy.spatial.epsilon.first.value = -0.1f;
    }
    SECTION("last above one") {
        key = "spatial.epsilon.last"; value = "1.1"; expected = "expected=<=1";
        policy.spatial.epsilon.last.value = 1.1f;
    }
    SECTION("negative phase start") {
        key = "spatial.epsilon.first.phase.[bad].start"; value = "-0.1"; expected = "expected=>=0";
        policy.spatial.epsilon.first.type = "phased";
        policy.spatial.epsilon.first.phases = {"bad"};
        policy.spatial.epsilon.first.phase.Set("bad", anet::ProfiledValuePhaseConfig<float>{
            .type = "linear", .start = -0.1f, .end = 0.1f, .steps = 2});
        data.Set("policy.spatial.epsilon.first.type", "phased");
        data.Set("policy.spatial.epsilon.first.phases", "bad");
        data.Set("policy.spatial.epsilon.first.phase.[bad].type", "linear");
        data.Set("policy.spatial.epsilon.first.phase.[bad].end", "0.1");
        data.Set("policy.spatial.epsilon.first.phase.[bad].steps", "2");
    }
    SECTION("phase end above one") {
        key = "spatial.epsilon.first.phase.[bad].end"; value = "1.1"; expected = "expected=<=1";
        policy.spatial.epsilon.first.type = "phased";
        policy.spatial.epsilon.first.phases = {"bad"};
        policy.spatial.epsilon.first.phase.Set("bad", anet::ProfiledValuePhaseConfig<float>{
            .type = "linear", .start = 0.1f, .end = 1.1f, .steps = 2});
        data.Set("policy.spatial.epsilon.first.type", "phased");
        data.Set("policy.spatial.epsilon.first.phases", "bad");
        data.Set("policy.spatial.epsilon.first.phase.[bad].type", "linear");
        data.Set("policy.spatial.epsilon.first.phase.[bad].start", "0.1");
        data.Set("policy.spatial.epsilon.first.phase.[bad].steps", "2");
    }
    data.Set("policy." + key, value);
    if (key != "spatial.scale") {
        const auto standalone_key = key.ends_with(".end") ? "end (not read from config)"
            : key.ends_with(".start") ? "start (not read from config)" : "value (not read from config)";
        CHECK_THROWS_WITH(dqn::SpatialLadderSchedule(policy.spatial.epsilon, "log", 4, torch::kCPU),
            Catch::Matchers::ContainsSubstring(standalone_key) && Catch::Matchers::ContainsSubstring("value=" + value)
            && Catch::Matchers::ContainsSubstring(expected));
    }
    // 公開ConfigReaderを通し、単体構築とは別の設定読込境界でも同じ不正値を拒否する。
    anet::Config owner("");
    dqn::ActionPolicyConfig parsed;
    CHECK_THROWS_WITH(anet::ConfigReader<dqn::ActionPolicyConfig>::Read(owner, data, "policy", parsed),
        Catch::Matchers::ContainsSubstring("policy." + key) && Catch::Matchers::ContainsSubstring(value)
        && Catch::Matchers::ContainsSubstring(expected));
}

TEST_CASE("Decomposed Actor reports the intrinsic effect on action selection", "[prd121][dqn]")
{
    // 非零値は相対1e-5、零値だけ絶対1e-6で検査する。
    const auto approximate = [](float expected) {
        return Catch::Approx(expected).epsilon(1e-5).margin(expected == 0.0f ? 1e-6 : 0.0);
    };
    // 2 lane・3行動の既知の値をNetworkの公開口へ渡し、Actorが作る診断を読む。
    const auto mode = GENERATE(std::string(anet::rl::dqn::ActionPolicyConfig::kQuantileModeNone), std::string(anet::rl::dqn::ActionPolicyConfig::kQuantileModeQr), std::string(anet::rl::dqn::ActionPolicyConfig::kQuantileModeIqn));
    const bool tbo = GENERATE(false, true);
    const bool zero_beta = GENERATE(false, true);
    const bool single_action = GENERATE(false, true);
    const int64_t actions = single_action ? 1 : 3;
    constexpr int64_t quantiles = 5;
    constexpr const char* external_key = kPrd121ExtrinsicQ;
    constexpr const char* internal_key = kPrd121IntrinsicQ;
    constexpr const char* external_dist_key = kPrd121ExtrinsicQDist;
    constexpr const char* internal_dist_key = kPrd121IntrinsicQDist;
    const auto external = torch::tensor({{1.0f, 1.02f, 0.9f}, {0.5f, 0.6f, 0.4f}}).narrow(1, 0, actions);
    const auto internal = torch::tensor({{8.0f, 5.0f, 12.0f}, {0.0f, 1.0f, 2.0f}}).narrow(1, 0, actions);
    anet::TensorDict observations;
    observations.Set(external_key, tbo ? dqn::TransformH(external, 0.01f) : external);
    observations.Set(internal_key, tbo ? dqn::TransformH(internal, 0.01f) : internal);
    observations.Set(external_dist_key, observations.At(external_key).unsqueeze(2).expand({2, actions, quantiles}));
    observations.Set(internal_dist_key, observations.At(internal_key).unsqueeze(2).expand({2, actions, quantiles}));
    anet::TensorSpecMap specs;
    anet::nn::NetworkConfig network_config;
    for (const auto* key : {external_key, internal_key, external_dist_key, internal_dist_key}) {
        if (mode == anet::rl::dqn::ActionPolicyConfig::kQuantileModeNone && (std::string(key) == external_dist_key || std::string(key) == internal_dist_key)) continue;
        specs[key] = {.shape = observations.At(key).sizes().slice(1).vec(), .dtype = torch::kFloat32};
        network_config.output_keys[key] = key;
    }
    auto network = anet::test::MakeTestNetwork(network_config, specs, nullptr, nullptr);
    dqn::ActionPolicyConfig policy_config;
    policy_config.policy_type = dqn::kActionPolicyTypeStr_Greedy;
    policy_config.tau_rule.num_taus = quantiles;
    auto policy = dqn::DqnComponentAssembler::CreateActionPolicy(policy_config, mode, false, 2, torch::kCPU);
    dqn::ActorQHintConfig hint;
    hint.use_tbo = tbo;
    hint.decomposed = true;
    hint.beta.value = zero_beta ? 0.0f : 0.01f;
    auto actor = std::make_shared<dqn::Actor>(policy, nullptr, nullptr, torch::kCPU, 121,
        std::make_shared<std::shared_mutex>(), network, network, anet::nn::NetworkForwardConfig::GetDefault(),
        true, std::nullopt, false, hint);
    const auto flags = torch::zeros({2}, torch::kBool);
    const auto info = actor->MakeAction({}, rl::BatchState(observations, flags, flags, ~flags));
    REQUIRE(info->GetReplayInitialPriorityHint());
    const auto values = dqn::DecodeActorQHint(info->GetReplayInitialPriorityHint()->GetPayload());
    const auto selected = info->GetAction().unsqueeze(1);
    if (zero_beta) CHECK(torch::equal(info->GetAction(), external.argmax(1)));
    CHECK(torch::allclose(values.actor_q_sa, observations.At(external_key).gather(1, selected).squeeze(1), 1e-5, 1e-6));
    CHECK(torch::allclose(values.intrinsic_q_sa, observations.At(internal_key).gather(1, selected).squeeze(1), 1e-5, 1e-6));
    const auto replaced = std::dynamic_pointer_cast<dqn::DQNActionInfo>(info->WithAction(torch::zeros({2}, torch::kInt64)));
    const auto replaced_values = dqn::DecodeActorQHint(replaced->GetReplayInitialPriorityHint()->GetPayload());
    CHECK(torch::allclose(replaced_values.actor_q_sa, observations.At(external_key).select(1, 0), 1e-5, 1e-6));
    CHECK(torch::allclose(replaced_values.intrinsic_q_sa, observations.At(internal_key).select(1, 0), 1e-5, 1e-6));
    CHECK(torch::equal(replaced_values.intrinsic_state_value, values.intrinsic_state_value));
    struct Expected {
        const char* lane_key;
        const char* mean_key;
        float first;
        float second;
    };
    for (const auto& expected : std::array<Expected, 4>{{
        {.lane_key = kPrd121ExtrinsicQArgmaxDisagreement, .mean_key = kPrd121ExtrinsicQArgmaxDisagreementRate, .first = 1.0f, .second = 0.0f},
        {.lane_key = kPrd121ExtrinsicQRegret, .mean_key = kPrd121ExtrinsicQRegretMean, .first = 0.02f, .second = 0.0f},
        {.lane_key = kPrd121ExtrinsicQGap, .mean_key = kPrd121ExtrinsicQGapMean, .first = 0.02f, .second = 0.1f},
        {.lane_key = kPrd121IntrinsicQGap, .mean_key = kPrd121IntrinsicQGapMean, .first = 0.04f, .second = 0.01f}}}) {
        CAPTURE(mode, tbo, expected.lane_key);
        const auto first = info->GetScalar(expected.lane_key, 0);
        const auto second = info->GetScalar(expected.lane_key, 1);
        const auto average = info->GetScalar(expected.mean_key);
        REQUIRE(first);
        REQUIRE(second);
        REQUIRE(average);
        CHECK_THROWS(info->GetScalar(expected.lane_key, 2));
        CHECK_THROWS(info->GetScalar(expected.lane_key, -1));
        if (single_action && std::string(expected.lane_key).ends_with(kPrd121Gap)) {
            CHECK(std::isnan(*first));
            CHECK(std::isnan(*second));
            CHECK(std::isnan(*average));
        } else {
            const bool zero = single_action || (zero_beta && std::string(expected.lane_key) != kPrd121ExtrinsicQGap);
            CHECK(*first == approximate(zero ? 0 : expected.first));
            CHECK(*second == approximate(zero ? 0 : expected.second));
            CHECK(*average == approximate(zero ? 0 : (expected.first + expected.second) / 2));
            CHECK(replaced->GetScalar(expected.mean_key) == average);
        }
    }
    auto next_observations = observations;
    next_observations.Set(internal_key, torch::zeros_like(observations.At(internal_key)));
    next_observations.Set(internal_dist_key, torch::zeros_like(observations.At(internal_dist_key)));
    actor->MakeAction({.exp_step = 2}, rl::BatchState(next_observations, flags, flags, flags));
    const std::array<std::pair<const char*, float>, 4> session_values{{
        {kPrd121ExtrinsicQSessionArgmaxDisagreementRate, 0.25f},
        {kPrd121ExtrinsicQSessionRegretMean, 0.005f},
        {kPrd121ExtrinsicQSessionGapMean, 0.06f},
        {kPrd121IntrinsicQSessionGapMean, 0.0125f}}};
    for (const auto& [key, expected] : session_values) {
        const auto value = actor->GetScalar(key);
        CAPTURE(key);
        REQUIRE(value);
        if (single_action && std::string(key).ends_with(kPrd121GapMean)) CHECK(std::isnan(*value));
        else {
            const bool zero = single_action || (zero_beta && std::string(key) != kPrd121ExtrinsicQSessionGapMean);
            CHECK(*value == approximate(zero ? 0 : expected));
        }
    }
    actor->Sync();
    for (const auto& [key, expected] : session_values) {
        const auto value = actor->GetScalar(key);
        REQUIRE(value);
        CHECK(std::isnan(*value));
    }
}

TEST_CASE("DQN action without a request does not create nn_trace", "[prd170][dqn][no_trace]")
{
    ScopedNoopMetricsLogger metrics_logger;
    PrepareNativeStackNetworkTest();
    auto data = MakeNativeStackConfigData();
    data.Set("net.branch.[main_feature].bind", kVectorKey);
    data.Set("net.branch.[main_feature].structure", "Flatten > Linear");
    const auto spec = MakeNativeStackEnvSpec({{kVectorKey, anet::TensorSpec{
        .type = anet::SpaceType::Vector, .shape = {8}, .dtype = torch::kFloat32}}});
    const auto action = ActWithNativeStack(data, spec,
        anet::TensorDict{{kVectorKey, torch::arange(16, torch::kFloat32).reshape({2, 8})}}, false);
    const auto& aux = action->GetAuxData();
    CHECK(std::none_of(aux.begin(), aux.end(), [](const auto& entry) {
        return entry.first.starts_with(anet::nn::kNnTracePrefix);
    }));
}
