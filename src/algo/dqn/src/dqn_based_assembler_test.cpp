#include "anet/test/catch.hpp"
#include "test/dqn_fixture_test.hpp"

TEST_CASE("PRD101 step diagnostics use the carried gamma at bootstrap completion", "[prd101][dqn][step_diagnostics]")
{
    auto config = WithDefaultLearnerForwardConfig();
    config.gamma.type = "linear";
    config.gamma.start = 0.5f;
    config.gamma.end = 0.9f;
    config.gamma.steps = 2;
    config.use_n_step = true;
    config.n_step = 2;
    config.per_eps = 0.1f;
    config.use_tbo = false;
    config.munchausen.enabled = false;
    auto diagnostics = dqn::CreateStepDiagnosticsProvider(config, 1, [](const torch::Tensor& reward) { return reward; });
    for (int step = 0; step < 3; ++step) {
        auto action = std::make_shared<dqn::DQNActionInfo>(torch::zeros({1}, torch::kInt64), anet::TensorDict{},
            rl::AuxData{{"q_values", step == 2 ? torch::tensor({{6.0f, 2.0f}}) : torch::tensor({{1.0f, 4.0f}})}});
        action->effective_gamma = step == 2 ? 0.9f : 0.5f;
        action->munchausen_alpha = 0.9f;
        action->munchausen_entropy_tau = 0.03f;
        const auto result = diagnostics->Observe(rl::BatchExperience{
            MakeDeterminismState(step, 1, {false}, {false}, {step == 0}), action,
            torch::tensor({float(step + 1)}), MakeDeterminismState(step + 1, 1, {false}, {false}, {false})});
        if (step < 2) CHECK(result.completed.empty());
        else {
            REQUIRE(result.completed.size() == 1);
            CHECK(*result.completed[0].values->GetScalar("approx_priority", 0) == Catch::Approx(6.76f));
        }
    }
}
