// LunarLanderEnv_test.cpp

#include "LunarLanderEnv.hpp"

#include <clocale>
#include <exception>
#include <memory>
#include <string>
#include <vector>
#include "anet/catch_test.hpp"
#include "anet/test_util.hpp"
#include "anet/env.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

constexpr const char* kVectorKey = anet::rl::ObsKeys::kVector;

void SetupUtf8Console()
{
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::setlocale(LC_CTYPE, ".UTF-8");
}

anet::rl::env::LunarLanderEnvConfig MakeConfig(bool obs_include_action, int limit_step = 1000)
{
    anet::rl::env::LunarLanderEnvConfig config;
    config.obs_include_action = obs_include_action;
    config.limit_step = limit_step;
    config.terrain_point_count = 2;
    config.terrain_noise_height = 0.0f;
    config.init.x_range = 0.0f;
    config.init.y_range = 0.0f;
    config.init.x_velocity_range = 0.0f;
    config.init.y_velocity_range = 0.0f;
    config.init.angle_range = 0.0f;
    config.init.angular_velocity_range = 0.0f;
    return config;
}

anet::ConfigData MakeConfigData(bool obs_include_action, int limit_step = 1000)
{
    anet::ConfigData config_data;
    config_data.Set("LunarLanderEnv.obs_include_action", obs_include_action ? "true" : "false");
    config_data.Set("LunarLanderEnv.limit_step", std::to_string(limit_step));
    config_data.Set("LunarLanderEnv.terrain_point_count", "2");
    config_data.Set("LunarLanderEnv.terrain_noise_height", "0");
    config_data.Set("LunarLanderEnv.init.x_range", "0");
    config_data.Set("LunarLanderEnv.init.y_range", "0");
    config_data.Set("LunarLanderEnv.init.x_velocity_range", "0");
    config_data.Set("LunarLanderEnv.init.y_velocity_range", "0");
    config_data.Set("LunarLanderEnv.init.angle_range", "0");
    config_data.Set("LunarLanderEnv.init.angular_velocity_range", "0");
    return config_data;
}

std::shared_ptr<anet::rl::env::LunarLanderEnv> MakeEnv(
    bool obs_include_action, int limit_step = 1000, anet::seed_t seed = 1)
{
    return std::make_shared<anet::rl::env::LunarLanderEnv>(
        MakeConfig(obs_include_action, limit_step), torch::Device(torch::kCPU), seed);
}

std::vector<float> TensorToFloatVector(const torch::Tensor& tensor)
{
    const auto flat = tensor.detach().cpu().to(torch::kFloat32).reshape({ -1 }).contiguous();
    std::vector<float> values;
    values.reserve(static_cast<size_t>(flat.numel()));
    for (int64_t i = 0; i < flat.numel(); ++i) {
        values.push_back(flat[i].item<float>());
    }
    return values;
}

void RequireFlatApprox(const torch::Tensor& tensor, const std::vector<float>& expected)
{
    const auto values = TensorToFloatVector(tensor);
    REQUIRE(values.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(values[i] == Catch::Approx(expected[i]).margin(1.0e-6f));
    }
}

torch::Tensor VectorObs(const anet::rl::SingleState& state)
{
    return state.obs.At(kVectorKey);
}

} // namespace

TEST_CASE("LunarLanderEnv keeps default observation contract", "[lunarlander][obs_include_action]")
{
    anet::rl::env::LunarLanderEnvConfig default_config;
    CHECK(default_config.ToJson().at("obs_include_action").get<bool>() == false);
    CHECK(default_config.ToConfigString().find("LunarLanderEnv.obs_include_action = false") != std::string::npos);

    auto env = MakeEnv(/*obs_include_action=*/false);
    const auto spec = env->GetSpec().state_spec.obs_spec.at(kVectorKey);
    const std::vector<int64_t> expected_shape{ 8 };
    CHECK(spec.shape == expected_shape);

    const auto reset_result = env->Reset(anet::rl::RunMode::Train);
    CHECK(VectorObs(reset_result->state).numel() == 8);

    const auto step_result = env->Step(anet::rl::env::kActionNoop, anet::rl::RunMode::Train);
    CHECK(VectorObs(step_result->next_state).numel() == 8);
}

TEST_CASE("LunarLanderEnv extends spec and reset observation with previous action block", "[lunarlander][obs_include_action]")
{
    auto env = MakeEnv(/*obs_include_action=*/true);
    const auto spec = env->GetSpec().state_spec.obs_spec.at(kVectorKey);
    const std::vector<int64_t> expected_shape{ 12 };
    CHECK(spec.shape == expected_shape);
    REQUIRE(spec.labels.size() == 12);
    REQUIRE(spec.min_values.size() == 12);
    REQUIRE(spec.max_values.size() == 12);
    CHECK(spec.labels[8] == "a_noop");
    CHECK(spec.labels[9] == "a_left");
    CHECK(spec.labels[10] == "a_main");
    CHECK(spec.labels[11] == "a_right");
    CHECK(spec.min_values[8] == Catch::Approx(0.0));
    CHECK(spec.min_values[11] == Catch::Approx(0.0));
    CHECK(spec.max_values[8] == Catch::Approx(1.0));
    CHECK(spec.max_values[11] == Catch::Approx(1.0));

    const auto reset_result = env->Reset(anet::rl::RunMode::Train);
    const auto obs = VectorObs(reset_result->state);
    CHECK(obs.numel() == 12);
    RequireFlatApprox(obs.slice(/*dim=*/0, 8, 12), { 0.0f, 0.0f, 0.0f, 0.0f });
}

TEST_CASE("LunarLanderEnv reports the action that produced the next observation", "[lunarlander][obs_include_action]")
{
    const std::vector<std::pair<int64_t, std::vector<float>>> cases = {
        { anet::rl::env::kActionNoop, { 1.0f, 0.0f, 0.0f, 0.0f } },
        { anet::rl::env::kActionLeft, { 0.0f, 1.0f, 0.0f, 0.0f } },
        { anet::rl::env::kActionMain, { 0.0f, 0.0f, 1.0f, 0.0f } },
        { anet::rl::env::kActionRight, { 0.0f, 0.0f, 0.0f, 1.0f } },
    };

    for (const auto& [action, expected] : cases) {
        auto env = MakeEnv(/*obs_include_action=*/true);
        env->Reset(anet::rl::RunMode::Train);

        const auto step_result = env->Step(action, anet::rl::RunMode::Train);
        const auto obs = VectorObs(step_result->next_state);
        CHECK(obs.numel() == 12);
        RequireFlatApprox(obs.slice(/*dim=*/0, 8, 12), expected);
    }
}

TEST_CASE("LunarLanderEnv keeps the first eight observation values unchanged", "[lunarlander][obs_include_action]")
{
    auto env_without_action = MakeEnv(/*obs_include_action=*/false, /*limit_step=*/1000, /*seed=*/123);
    auto env_with_action = MakeEnv(/*obs_include_action=*/true, /*limit_step=*/1000, /*seed=*/123);

    auto reset_without_action = env_without_action->Reset(anet::rl::RunMode::Train);
    auto reset_with_action = env_with_action->Reset(anet::rl::RunMode::Train);
    CHECK(torch::allclose(
        VectorObs(reset_without_action->state),
        VectorObs(reset_with_action->state).slice(/*dim=*/0, 0, 8)));

    const std::vector<int64_t> actions = {
        anet::rl::env::kActionNoop,
        anet::rl::env::kActionLeft,
        anet::rl::env::kActionMain,
        anet::rl::env::kActionRight,
        anet::rl::env::kActionNoop,
    };

    for (const auto action : actions) {
        const auto step_without_action = env_without_action->Step(action, anet::rl::RunMode::Train);
        const auto step_with_action = env_with_action->Step(action, anet::rl::RunMode::Train);
        CHECK(torch::allclose(
            VectorObs(step_without_action->next_state),
            VectorObs(step_with_action->next_state).slice(/*dim=*/0, 0, 8)));
    }
}

TEST_CASE("LunarLanderEnv action observation works through batch env prefixes", "[lunarlander][obs_include_action]")
{
    auto config_data = MakeConfigData(/*obs_include_action=*/true, /*limit_step=*/1000);
    config_data.Set("train.eval.[test1].env.limit_step", "1");

    auto factory = std::make_shared<anet::rl::env::LunarLanderEnvFactory>();
    anet::rl::VectorizedDiscreteBatchEnv env(
        config_data,
        factory,
        /*num_envs=*/1,
        torch::Device(torch::kCPU),
        /*seed=*/1,
        /*config_prefix=*/"train.eval.[test1].env");

    const auto spec = env.GetSpec();
    const std::vector<int64_t> expected_single_shape{ 12 };
    const std::vector<int64_t> expected_batch_shape{ 1, 12 };
    CHECK(spec.state_spec.obs_spec.at(kVectorKey).shape == expected_single_shape);

    const auto reset_result = env.Reset(anet::rl::RunMode::Train);
    CHECK(spec.state_spec.ValidateObservation(reset_result->state.obs, /*is_batched=*/true));
    CHECK(reset_result->state.obs.At(kVectorKey).sizes().vec() == expected_batch_shape);
    RequireFlatApprox(reset_result->state.obs.At(kVectorKey)[0].slice(/*dim=*/0, 8, 12), { 0.0f, 0.0f, 0.0f, 0.0f });

    auto action_info = std::make_shared<anet::rl::BatchActionInfo>(
        torch::tensor({ anet::rl::env::kActionRight }, torch::TensorOptions().dtype(torch::kInt64)));
    const auto step_result = env.Step(action_info, anet::rl::RunMode::Train);
    CHECK(spec.state_spec.ValidateObservation(step_result->next_state.obs, /*is_batched=*/true));
    CHECK(step_result->next_state.obs.At(kVectorKey).sizes().vec() == expected_batch_shape);
    RequireFlatApprox(step_result->next_state.obs.At(kVectorKey)[0].slice(/*dim=*/0, 8, 12), { 0.0f, 0.0f, 0.0f, 1.0f });
    CHECK(step_result->next_state.truncated[0].item<bool>() == true);
}

int main(int argc, char* argv[])
{
    SetupUtf8Console();

    anet::test::PreparedTestArgs test_args;
    try {
        test_args = anet::test::PrepareTestArgs(argc, argv);
    } catch (const std::exception& e) {
        return anet::test::ReportTestArgsError(e);
    }
    anet::test::SetupTestFailureDialog(test_args.failure_dialog_enabled);

    Catch::Session session;
    session.configData().showDurations = Catch::ShowDurations::Always;
    return session.run(test_args.Argc(), test_args.Argv());
}
