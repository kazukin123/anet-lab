#include "anet/test/catch.hpp"
#include "test/dqn_fixture_test.hpp"

TEST_CASE("TBO transform is monotonic and invertible on representative values", "[dqn][tbo]")
{
    auto env_spec = MakeLearnerEnvSpec();
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    rl::BatchEnvSpec batch_env_spec{ 1, 1 };

    for (float epsilon : { 1.0e-2f, 1.0e-3f }) {
        INFO(epsilon);
        dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
        config.tbo_epsilon = epsilon;
        TestLearner learner(config, model, vars, batch_env_spec, env_spec, dqn::LearnerForwardResources{});

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

TEST_CASE("DQN scalar raw priority matches tensor clipping policy", "[dqn][per][actor_initial][math]")
{
    struct Case {
        float td_error;
        float per_eps;
        bool use_clip;
        float clip_value;
    };
    const std::array<Case, 8> cases{
        Case{ 0.0f, 0.0f, false, 0.0f },
        Case{ -0.25f, 0.1f, false, 0.0f },
        Case{ 0.25f, 0.1f, false, 0.0f },
        Case{ 0.25f, 0.1f, true, 0.5f },
        Case{ 0.4f, 0.1f, true, 0.5f },
        Case{ 2.0f, 0.1f, true, 0.5f },
        Case{ -2.0f, 0.1f, true, 0.5f },
        Case{ 2.0f, 0.0f, true, 0.0f },
    };

    for (const auto& test_case : cases) {
        INFO("td_error=" << test_case.td_error << " per_eps=" << test_case.per_eps
            << " use_clip=" << test_case.use_clip << " clip_value=" << test_case.clip_value);
        const float tensor_priority = dqn::MakePerRawPriority(
            torch::tensor(test_case.td_error),
            test_case.per_eps,
            test_case.use_clip,
            test_case.clip_value).item<float>();
        CHECK(dqn::MakePerRawPriority(
            test_case.td_error,
            test_case.per_eps,
            test_case.use_clip,
            test_case.clip_value) == Catch::Approx(tensor_priority).margin(1.0e-7f));
    }
}

TEST_CASE("DQN initial priority estimator completes a one-step bootstrap", "[dqn][per][actor_initial][estimator]")
{
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.use_tbo = false;
    config.per_eps = 0.1f;
    config.use_per_prio_clip = false;
    auto estimator = dqn::CreateInitialPriorityEstimator(config);

    const std::array<float, dqn::kActorQHintColumnCount> start_hint{ 4.0f, 5.0f, 0.0f };
    const std::array<float, dqn::kActorQHintColumnCount> bootstrap_hint{ 6.0f, 2.0f, 0.0f };
    const auto priority = estimator->Estimate(rl::InitialPriorityEstimateInput{
        .start_hint = start_hint,
        .bootstrap_hint = bootstrap_hint,
        .target_return = 1.0f,
        .discount = 0.9f,
        .terminal = false,
        .actual_n_steps = 1,
    });

    REQUIRE(priority.has_value());
    CHECK(*priority == Catch::Approx(1.3f).margin(1.0e-6f));
}

TEST_CASE("DQN initial priority estimator distinguishes n-step bootstrap and true terminal", "[dqn][per][actor_initial][estimator]")
{
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.use_tbo = false;
    config.per_eps = 0.0f;
    config.use_per_prio_clip = false;
    auto estimator = dqn::CreateInitialPriorityEstimator(config);
    const std::array<float, dqn::kActorQHintColumnCount> start_hint{ 4.0f, 99.0f, 0.0f };
    const std::array<float, dqn::kActorQHintColumnCount> bootstrap_hint{ 88.0f, 3.0f, 0.0f };

    const auto n_step_priority = estimator->Estimate(rl::InitialPriorityEstimateInput{
        .start_hint = start_hint,
        .bootstrap_hint = bootstrap_hint,
        .target_return = 1.0f,
        .discount = 0.729f,
        .terminal = false,
        .actual_n_steps = 3,
    });
    REQUIRE(n_step_priority.has_value());
    CHECK(*n_step_priority == Catch::Approx(0.813f).margin(1.0e-6f));

    const auto terminal_priority = estimator->Estimate(rl::InitialPriorityEstimateInput{
        .start_hint = start_hint,
        .bootstrap_hint = {},
        .target_return = 1.0f,
        .discount = 0.729f,
        .terminal = true,
        .actual_n_steps = 3,
    });
    REQUIRE(terminal_priority.has_value());
    CHECK(*terminal_priority == Catch::Approx(3.0f).margin(1.0e-6f));
}

TEST_CASE("DQN initial priority estimator applies TBO to QR mean Q hints", "[dqn][tbo][per][actor_initial][estimator]")
{
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.use_tbo = true;
    config.tbo_epsilon = 1.0e-2f;
    config.per_eps = 0.05f;
    config.use_per_prio_clip = false;
    auto estimator = dqn::CreateInitialPriorityEstimator(config);

    const float actor_q_sa = torch::tensor({ 2.0f, 4.0f }).mean().item<float>();
    const float bootstrap_state_value = torch::tensor({ 1.0f, 5.0f }).mean().item<float>();
    const std::array<float, dqn::kActorQHintColumnCount> start_hint{ actor_q_sa, 0.0f, 0.0f };
    const std::array<float, dqn::kActorQHintColumnCount> bootstrap_hint{ 0.0f, bootstrap_state_value, 0.0f };
    const auto priority = estimator->Estimate(rl::InitialPriorityEstimateInput{
        .start_hint = start_hint,
        .bootstrap_hint = bootstrap_hint,
        .target_return = 2.0f,
        .discount = 0.81f,
        .terminal = false,
        .actual_n_steps = 2,
    });

    const float target = dqn::TransformH(
        2.0f + 0.81f * dqn::TransformHInv(bootstrap_state_value, config.tbo_epsilon),
        config.tbo_epsilon);
    const float expected = dqn::MakePerRawPriority(
        target - actor_q_sa,
        config.per_eps,
        config.use_per_prio_clip,
        config.per_prio_clip_value);
    REQUIRE(priority.has_value());
    CHECK(*priority == Catch::Approx(expected).epsilon(1.0e-6f).margin(1.0e-6f));
}

TEST_CASE("DQN initial priority estimator preserves zero and clip boundaries", "[dqn][per][actor_initial][estimator]")
{
    const std::array<float, dqn::kActorQHintColumnCount> start_hint{ 2.0f, 0.0f, 0.0f };

    dqn::LearnerConfig zero_config = WithDefaultLearnerForwardConfig();
    zero_config.per_eps = 0.0f;
    zero_config.use_per_prio_clip = false;
    auto zero_estimator = dqn::CreateInitialPriorityEstimator(zero_config);
    const auto zero_priority = zero_estimator->Estimate(rl::InitialPriorityEstimateInput{
        .start_hint = start_hint,
        .bootstrap_hint = {},
        .target_return = 2.0f,
        .discount = 0.0f,
        .terminal = true,
        .actual_n_steps = 1,
    });
    REQUIRE(zero_priority.has_value());
    CHECK(*zero_priority == 0.0f);

    dqn::LearnerConfig clip_config = WithDefaultLearnerForwardConfig();
    clip_config.per_eps = 0.1f;
    clip_config.use_per_prio_clip = true;
    clip_config.per_prio_clip_value = 0.5f;
    auto clip_estimator = dqn::CreateInitialPriorityEstimator(clip_config);
    const auto clipped_priority = clip_estimator->Estimate(rl::InitialPriorityEstimateInput{
        .start_hint = start_hint,
        .bootstrap_hint = {},
        .target_return = 4.0f,
        .discount = 0.0f,
        .terminal = true,
        .actual_n_steps = 1,
    });
    REQUIRE(clipped_priority.has_value());
    CHECK(*clipped_priority == Catch::Approx(0.5f).margin(1.0e-7f));
}

TEST_CASE("DQN initial priority estimator distinguishes schema errors and non-finite values", "[dqn][per][actor_initial][estimator]")
{
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    auto estimator = dqn::CreateInitialPriorityEstimator(config);
    const std::array<float, dqn::kActorQHintColumnCount> finite_hint{ 1.0f, 2.0f, 0.0f };
    const std::array<float, dqn::kActorQHintColumnCount> nan_hint{ std::numeric_limits<float>::quiet_NaN(), 2.0f, 0.0f };
    const std::array<float, dqn::kActorQHintColumnCount> inf_hint{ 1.0f, std::numeric_limits<float>::infinity(), 0.0f };

    CHECK(estimator->ValidateHint(finite_hint));
    CHECK_FALSE(estimator->ValidateHint(nan_hint));
    CHECK_FALSE(estimator->ValidateHint(inf_hint));
    CHECK_THROWS(estimator->ValidateHint(std::span<const float>(finite_hint.data(), 1)));

    const auto nonfinite_start = estimator->Estimate(rl::InitialPriorityEstimateInput{
        .start_hint = nan_hint,
        .bootstrap_hint = {},
        .target_return = 1.0f,
        .discount = 0.0f,
        .terminal = true,
        .actual_n_steps = 1,
    });
    CHECK_FALSE(nonfinite_start.has_value());

    const auto nonfinite_bootstrap = estimator->Estimate(rl::InitialPriorityEstimateInput{
        .start_hint = finite_hint,
        .bootstrap_hint = inf_hint,
        .target_return = 1.0f,
        .discount = 0.9f,
        .terminal = false,
        .actual_n_steps = 1,
    });
    CHECK_FALSE(nonfinite_bootstrap.has_value());
}

TEST_CASE("DQN initial priority estimator matches scalar learner TD priority", "[dqn][learner][tbo][per][actor_initial][math]")
{
    for (bool use_tbo : { false, true }) {
        INFO("use_tbo=" << use_tbo);
        dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
        config.learning_rate.value = 0.0f;
        config.use_fused_optimizer = false;
        config.use_grad_clip = false;
        config.use_td_clip = false;
        config.use_per = false;
        config.replay_capacity = 8;
        config.replay_batch_size = 1;
        config.gamma.value = 0.9f;
        config.use_tbo = use_tbo;
        config.tbo_epsilon = 1.0e-2f;
        config.per_eps = 0.05f;
        config.use_per_prio_clip = true;
        config.per_prio_clip_value = 10.0f;

        auto env_spec = MakeLearnerEnvSpec();
        TestNetworkModel model;
        dqn::RuntimeVars vars;
        rl::BatchEnvSpec batch_env_spec{ 1, 1 };
        auto target_policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{});
        dqn::TDLearner learner(
            config,
            model,
            vars,
            nullptr,
            batch_env_spec,
            env_spec,
            torch::kCPU,
            123,
            kTestAgentSeed,
            target_policy,
            dqn::LearnerForwardResources{},
            std::nullopt,
            456);

        rl::ExperienceSamples samples;
        samples.obs = anet::TensorDict{ { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }) } };
        samples.actions = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kInt64));
        samples.target_returns = torch::tensor({ 0.75f });
        samples.next_state.next_obs = anet::TensorDict{
            { kVectorKey, torch::tensor({ { 3.0f, 4.0f } }) },
        };
        samples.next_state.terminals = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kBool));
        samples.n_steps = torch::tensor({ 3 }, torch::TensorOptions().dtype(torch::kInt64));

        float bootstrap_state_value = 0.0f;
        {
            torch::NoGradGuard no_grad;
            bootstrap_state_value = model.GetTargetNetwork()->Forward(samples.next_state.next_obs).At("q").item<float>();
        }
        const auto base_result = learner.UpdateFromSamples(samples);
        const auto result = std::dynamic_pointer_cast<const dqn::BatchUpdateResult>(base_result);
        REQUIRE(result != nullptr);
        REQUIRE(result->td_error.numel() == 1);
        REQUIRE(result->q_sa.numel() == 1);

        const std::array<float, dqn::kActorQHintColumnCount> start_hint{ result->q_sa.item<float>(), 0.0f, 0.0f };
        const std::array<float, dqn::kActorQHintColumnCount> bootstrap_hint{ 0.0f, bootstrap_state_value, 0.0f };
        auto estimator = dqn::CreateInitialPriorityEstimator(config);
        const auto actor_priority = estimator->Estimate(rl::InitialPriorityEstimateInput{
            .start_hint = start_hint,
            .bootstrap_hint = bootstrap_hint,
            .target_return = samples.target_returns.item<float>(),
            .discount = static_cast<float>(std::pow(config.gamma.value, samples.n_steps.item<int64_t>())),
            .terminal = false,
            .actual_n_steps = samples.n_steps.item<int>(),
        });
        const float learner_priority = dqn::MakePerRawPriority(
            result->td_error.detach(),
            config.per_eps,
            config.use_per_prio_clip,
            config.per_prio_clip_value).item<float>();

        REQUIRE(actor_priority.has_value());
        CHECK(*actor_priority == Catch::Approx(learner_priority).epsilon(2.0e-5f).margin(5.0e-6f));
    }
}

TEST_CASE("PER raw priority batch counts strict pre-clip changes on CPU and CUDA", "[dqn][per][clip]")
{
    std::vector<torch::Device> devices{ torch::Device(torch::kCPU) };
    if (torch::cuda::is_available()) {
        devices.emplace_back(torch::Device(torch::kCUDA, 0));
    }

    for (const auto& device : devices) {
        INFO("device=" << device.str());

        // priorityがclip上限未満・等値・超過のとき、超過分だけを件数へ含める。
        auto td_error = torch::tensor({ -0.2f, 0.9f, 2.0f }, torch::TensorOptions().device(device));
        auto clipped = dqn::MakePerRawPriorityBatch(td_error, 0.1f, true, 1.0f);
        CHECK(torch::allclose(clipped.priorities.cpu(), torch::tensor({ 0.3f, 1.0f, 1.0f })));
        CHECK(clipped.clipped_count.item<int64_t>() == 1);

        auto unclipped = dqn::MakePerRawPriorityBatch(td_error, 0.1f, false, 1.0f);
        CHECK(torch::allclose(unclipped.priorities.cpu(), torch::tensor({ 0.3f, 1.0f, 2.1f })));
        CHECK(unclipped.clipped_count.item<int64_t>() == 0);

        auto zero_clip = dqn::MakePerRawPriorityBatch(td_error, 0.0f, true, 0.0f);
        CHECK(torch::allclose(zero_clip.priorities.cpu(), torch::zeros({ 3 })));
        CHECK(zero_clip.clipped_count.item<int64_t>() == 3);
    }
}
