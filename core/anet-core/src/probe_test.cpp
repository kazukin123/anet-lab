#include "catch.hpp"

#include "anet/probe.hpp"

#include <string>
#include <unordered_set>
#include <vector>

namespace {

namespace rl = anet::rl;

constexpr const char* kVectorKey = rl::ObsKeys::kVector;
constexpr const char* kGridKey = rl::ObsKeys::kGrid;

torch::Tensor FloatTensor(const std::vector<float>& values, const std::vector<int64_t>& shape)
{
    return torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32)).reshape(shape);
}

torch::Tensor BoolVector(int64_t size, bool value = false)
{
    return torch::full({ size }, value, torch::TensorOptions().dtype(torch::kBool));
}

void RequireFlatApprox(const torch::Tensor& tensor, const std::vector<float>& expected)
{
    auto flat = tensor.detach().cpu().reshape({ -1 }).contiguous();
    REQUIRE(flat.numel() == static_cast<int64_t>(expected.size()));
    auto acc = flat.accessor<float, 1>();
    for (int64_t i = 0; i < static_cast<int64_t>(expected.size()); ++i) {
        REQUIRE(acc[i] == Catch::Approx(expected[static_cast<size_t>(i)]).margin(1.0e-5f));
    }
}

void RequireVectorApprox(const std::vector<float>& actual, const std::vector<float>& expected)
{
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(actual[i] == Catch::Approx(expected[i]).margin(1.0e-5f));
    }
}

anet::TensorDict MakeObs(
    const std::vector<float>& vector_values,
    const std::vector<float>& grid_values)
{
    anet::TensorDict obs;
    obs.Set(kVectorKey, FloatTensor(vector_values, { 2, 2 }));
    obs.Set(kGridKey, FloatTensor(grid_values, { 2, 1, 2 }));
    return obs;
}

rl::BatchExperience MakeExperience()
{
    rl::BatchState state(
        MakeObs({ -1.0f, -2.0f, -3.0f, -4.0f }, { -10.0f, -11.0f, -12.0f, -13.0f }),
        BoolVector(2),
        BoolVector(2),
        BoolVector(2));

    rl::BatchState next_state(
        MakeObs({ 1.0f, 2.0f, 3.0f, 4.0f }, { 10.0f, 11.0f, 12.0f, 13.0f }),
        BoolVector(2),
        BoolVector(2),
        BoolVector(2));

    rl::AuxData aux;
    aux["max_q"] = FloatTensor({ 10.0f, 20.0f }, { 2 });

    auto actions = torch::tensor({ 0, 1 }, torch::TensorOptions().dtype(torch::kInt64));
    auto action_info = std::make_shared<rl::BatchActionInfo>(actions, anet::TensorDict{}, aux);
    auto rewards = FloatTensor({ 0.5f, 1.5f }, { 2 });

    return rl::BatchExperience(state, action_info, rewards, next_state);
}

rl::StateSpec MakeLunarLikeStateSpec()
{
    anet::TensorSpec vector_spec;
    vector_spec.type = anet::SpaceType::Vector;
    vector_spec.shape = { 8 };
    vector_spec.dtype = torch::kFloat32;
    vector_spec.labels = { "x", "y", "vx", "vy", "angle", "v_angle", "leg_l", "leg_r" };
    vector_spec.min_values = { -1.5, -0.5, -5.0, -5.0, -3.14, -5.0, 0.0, 0.0 };
    vector_spec.max_values = { 1.5, 2.0, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0 };

    rl::StateSpec state_spec;
    state_spec.obs_spec[kVectorKey] = vector_spec;
    return state_spec;
}

rl::StateSpec MakeMultiKeyStateSpec()
{
    anet::TensorSpec vector_spec;
    vector_spec.type = anet::SpaceType::Vector;
    vector_spec.shape = { 3 };
    vector_spec.dtype = torch::kFloat32;
    vector_spec.min_values = { -1.0 };
    vector_spec.max_values = { 1.0 };

    anet::TensorSpec aux_vector_spec;
    aux_vector_spec.type = anet::SpaceType::Vector;
    aux_vector_spec.shape = { 2 };
    aux_vector_spec.dtype = torch::kFloat32;
    aux_vector_spec.min_values = { -10.0, -20.0 };
    aux_vector_spec.max_values = { 10.0, 20.0 };

    anet::TensorSpec grid_spec;
    grid_spec.type = anet::SpaceType::Grid;
    grid_spec.shape = { 1, 2, 2 };
    grid_spec.dtype = torch::kFloat32;

    rl::StateSpec state_spec;
    state_spec.obs_spec[kVectorKey] = vector_spec;
    state_spec.obs_spec["aux_vector"] = aux_vector_spec;
    state_spec.obs_spec[kGridKey] = grid_spec;
    return state_spec;
}

rl::ValueExtractFunction MakeSweepExtractor()
{
    return [](const torch::Tensor& t, const std::unordered_set<std::string>& req) {
        return rl::extractor::MeanIdxExtractor(t, req, -1);
    };
}

} // namespace

TEST_CASE("BatchExperience observation keys expose unified and subkey tensors", "[probe][experience]")
{
    auto experience = MakeExperience();

    auto unified = experience.GetTensor(rl::BatchExperience::NEXT_STATE_OBS);
    REQUIRE(unified.has_value());
    REQUIRE(unified->sizes().vec() == std::vector<int64_t>{ 2, 4 });
    RequireFlatApprox(*unified, { 1.0f, 2.0f, 10.0f, 11.0f, 3.0f, 4.0f, 12.0f, 13.0f });

    auto vector = experience.GetTensor(std::string(rl::BatchExperience::NEXT_STATE_OBS) + ".vector");
    REQUIRE(vector.has_value());
    REQUIRE(vector->sizes().vec() == std::vector<int64_t>{ 2, 2 });
    RequireFlatApprox(*vector, { 1.0f, 2.0f, 3.0f, 4.0f });

    auto max_q = experience.GetTensor("action.max_q");
    REQUIRE(max_q.has_value());
    REQUIRE(max_q->sizes().vec() == std::vector<int64_t>{ 2 });
    RequireFlatApprox(*max_q, { 10.0f, 20.0f });

    CHECK_THROWS(experience.GetTensor(std::string(rl::BatchExperience::NEXT_STATE_OBS) + ".unknown"));
}

TEST_CASE("BatchExperienceVectorProbe extracts observation columns and action aux tensors", "[probe][experience]")
{
    auto experience = MakeExperience();
    rl::UpdateEvent event{
        experience,
        nullptr,
        rl::StepCounts{},
        nullptr,
        rl::BatchUpdateResultList{}
    };

    rl::BatchExperienceVectorProbe x_probe(rl::BatchExperience::NEXT_STATE_OBS, 0);
    auto x_values = x_probe.GetVector(event);
    REQUIRE(x_values.has_value());
    RequireVectorApprox(*x_values, { 1.0f, 3.0f });

    rl::BatchExperienceVectorProbe y_probe(rl::BatchExperience::NEXT_STATE_OBS, 1);
    auto y_values = y_probe.GetVector(event);
    REQUIRE(y_values.has_value());
    RequireVectorApprox(*y_values, { 2.0f, 4.0f });

    rl::BatchExperienceVectorProbe vector_y_probe(std::string(rl::BatchExperience::NEXT_STATE_OBS) + ".vector", 1);
    auto vector_y_values = vector_y_probe.GetVector(event);
    REQUIRE(vector_y_values.has_value());
    RequireVectorApprox(*vector_y_values, { 2.0f, 4.0f });

    rl::BatchExperienceVectorProbe max_q_probe("action.max_q", -1);
    auto max_q_values = max_q_probe.GetVector(event);
    REQUIRE(max_q_values.has_value());
    RequireVectorApprox(*max_q_values, { 10.0f, 20.0f });
}

TEST_CASE("AgentTensorVectorProbe resolves ReplayBuffer observation subkey specs", "[probe][agent]")
{
    auto state_spec = MakeLunarLikeStateSpec();

    rl::AgentTensorVectorProbe next_x_probe(
        std::string(rl::ReplayBuffer::NEXT_STATE_OBS) + ".vector",
        0,
        &state_spec);
    REQUIRE(next_x_probe.GetMin().has_value());
    REQUIRE(next_x_probe.GetMax().has_value());
    REQUIRE(*next_x_probe.GetMin() == Catch::Approx(-1.5f));
    REQUIRE(*next_x_probe.GetMax() == Catch::Approx(1.5f));
    REQUIRE(next_x_probe.GetName().find("(x)") != std::string::npos);

    rl::AgentTensorVectorProbe next_y_probe(
        std::string(rl::ReplayBuffer::NEXT_STATE_OBS) + ".vector",
        1,
        &state_spec);
    REQUIRE(next_y_probe.GetMin().has_value());
    REQUIRE(next_y_probe.GetMax().has_value());
    REQUIRE(*next_y_probe.GetMin() == Catch::Approx(-0.5f));
    REQUIRE(*next_y_probe.GetMax() == Catch::Approx(2.0f));
    REQUIRE(next_y_probe.GetName().find("(y)") != std::string::npos);

    rl::AgentTensorVectorProbe unified_x_probe(
        rl::ReplayBuffer::NEXT_STATE_OBS,
        0,
        &state_spec);
    REQUIRE(unified_x_probe.GetMin().has_value());
    REQUIRE(unified_x_probe.GetMax().has_value());
    REQUIRE(*unified_x_probe.GetMin() == Catch::Approx(-1.5f));
    REQUIRE(*unified_x_probe.GetMax() == Catch::Approx(1.5f));
}

TEST_CASE("StateSweepProcessor sweeps selected vector observation and zero-fills other keys", "[probe][sweep]")
{
    auto state_spec = MakeMultiKeyStateSpec();
    rl::StateSweepProcessor processor(
        state_spec,
        0,
        1,
        MakeSweepExtractor(),
        "aux_vector",
        torch::kCPU);
    processor.ApplyGridSize(2, 2);

    auto input = processor.BuildInputTensor();
    auto aux = input.At("aux_vector");
    REQUIRE(aux.sizes().vec() == std::vector<int64_t>{ 4, 2 });
    RequireFlatApprox(aux, {
        -10.0f, -20.0f,
        10.0f, -20.0f,
        -10.0f, 20.0f,
        10.0f, 20.0f,
    });

    auto vector = input.At(kVectorKey);
    REQUIRE(vector.sizes().vec() == std::vector<int64_t>{ 4, 3 });
    RequireFlatApprox(vector, std::vector<float>(12, 0.0f));

    auto grid = input.At(kGridKey);
    REQUIRE(grid.sizes().vec() == std::vector<int64_t>{ 4, 1, 2, 2 });
    RequireFlatApprox(grid, std::vector<float>(16, 0.0f));
}

TEST_CASE("StateSweepProcessor allows same X/Y index and keeps Y overwrite order", "[probe][sweep]")
{
    auto state_spec = MakeMultiKeyStateSpec();
    rl::StateSweepProcessor processor(
        state_spec,
        0,
        0,
        MakeSweepExtractor(),
        "aux_vector",
        torch::kCPU);
    processor.ApplyGridSize(2, 2);

    auto aux = processor.BuildInputTensor().At("aux_vector");
    REQUIRE(aux.sizes().vec() == std::vector<int64_t>{ 4, 2 });
    RequireFlatApprox(aux, {
        -10.0f, 0.0f,
        -10.0f, 0.0f,
        10.0f, 0.0f,
        10.0f, 0.0f,
    });
}

TEST_CASE("StateSweepProcessor rejects invalid sweep observation settings", "[probe][sweep]")
{
    auto state_spec = MakeMultiKeyStateSpec();

    CHECK_THROWS(([&] {
        rl::StateSweepProcessor processor(
            state_spec,
            0,
            1,
            MakeSweepExtractor(),
            "missing",
            torch::kCPU);
    }()));

    CHECK_THROWS(([&] {
        rl::StateSweepProcessor processor(
            state_spec,
            0,
            1,
            MakeSweepExtractor(),
            kGridKey,
            torch::kCPU);
    }()));

    CHECK_THROWS(([&] {
        rl::StateSweepProcessor processor(
            state_spec,
            2,
            0,
            MakeSweepExtractor(),
            "aux_vector",
            torch::kCPU);
    }()));
}
