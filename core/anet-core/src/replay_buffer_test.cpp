#include "catch.hpp"

#include "anet/replay_buffer.hpp"
#include "replay_buffer_impl.hpp"

#include <cmath>
#include <exception>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

namespace rl = anet::rl;

constexpr const char* kVectorKey = rl::ObsKeys::kVector;
constexpr const char* kMaskKey = rl::ObsKeys::kActionMask;

struct TestBuffer {
    std::shared_ptr<rl::ReplayBuffer> rb;
    rl::ReplayBufferConfig config;
    int64_t num_envs;
    int64_t capacity_per_env;
};

float StateValue(int64_t env_idx, int64_t time_idx)
{
    return static_cast<float>(env_idx * 100 + time_idx);
}

float TerminalStateValue(int64_t env_idx, int64_t time_idx)
{
    return static_cast<float>(env_idx * 100 + 1000 + time_idx);
}

float RewardValue(int64_t env_idx, int64_t time_idx)
{
    return static_cast<float>((env_idx + 1) * 10 + time_idx);
}

std::vector<float> StateValues(int64_t num_envs, int64_t time_idx)
{
    std::vector<float> values;
    values.reserve(static_cast<size_t>(num_envs));
    for (int64_t env = 0; env < num_envs; ++env) {
        values.push_back(StateValue(env, time_idx));
    }
    return values;
}

std::vector<float> TerminalStateValues(int64_t num_envs, int64_t time_idx)
{
    std::vector<float> values;
    values.reserve(static_cast<size_t>(num_envs));
    for (int64_t env = 0; env < num_envs; ++env) {
        values.push_back(TerminalStateValue(env, time_idx));
    }
    return values;
}

std::vector<float> RewardValues(int64_t num_envs, int64_t time_idx)
{
    std::vector<float> rewards;
    rewards.reserve(static_cast<size_t>(num_envs));
    for (int64_t env = 0; env < num_envs; ++env) {
        rewards.push_back(RewardValue(env, time_idx));
    }
    return rewards;
}

std::vector<bool> BoolValues(int64_t num_envs, bool value)
{
    return std::vector<bool>(static_cast<size_t>(num_envs), value);
}

torch::Tensor FloatVector(const std::vector<float>& values)
{
    return torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32));
}

torch::Tensor FloatColumn(const std::vector<float>& values)
{
    return FloatVector(values).reshape({ static_cast<int64_t>(values.size()), 1 });
}

torch::Tensor MaskTensor(const std::vector<float>& values)
{
    auto mask = torch::empty({ static_cast<int64_t>(values.size()), 2 }, torch::TensorOptions().dtype(torch::kFloat32));
    auto acc = mask.accessor<float, 2>();
    for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
        acc[i][0] = values[static_cast<size_t>(i)];
        acc[i][1] = values[static_cast<size_t>(i)] + 0.25f;
    }
    return mask;
}

torch::Tensor BoolTensor(const std::vector<bool>& values)
{
    auto tensor = torch::empty({ static_cast<int64_t>(values.size()) }, torch::TensorOptions().dtype(torch::kBool));
    auto acc = tensor.accessor<bool, 1>();
    for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
        acc[i] = values[static_cast<size_t>(i)];
    }
    return tensor;
}

anet::TensorDict MakeObs(const std::vector<float>& values, bool include_mask = false)
{
    anet::TensorDict obs;
    obs.Set(kVectorKey, FloatColumn(values));
    if (include_mask) {
        obs.Set(kMaskKey, MaskTensor(values));
    }
    return obs;
}

rl::EnvSpec MakeEnvSpec(bool include_mask = false)
{
    anet::TensorSpec vector_spec;
    vector_spec.type = anet::SpaceType::Vector;
    vector_spec.shape = { 1 };
    vector_spec.dtype = torch::kFloat32;

    rl::EnvSpec spec;
    spec.state_spec.obs_spec[kVectorKey] = vector_spec;

    if (include_mask) {
        anet::TensorSpec mask_spec;
        mask_spec.type = anet::SpaceType::Vector;
        mask_spec.shape = { 2 };
        mask_spec.dtype = torch::kFloat32;
        spec.state_spec.obs_spec[kMaskKey] = mask_spec;
    }

    spec.action_spec.is_discrete = true;
    spec.action_spec.value_labels = { "a0", "a1" };
    spec.reward_range = { -1000.0f, 1000.0f };
    return spec;
}

rl::BatchExperience MakeBatch(
    const std::vector<float>& state_values,
    const std::vector<float>& next_values,
    const std::vector<float>& rewards,
    const std::vector<bool>& next_done,
    const std::vector<bool>& next_truncated,
    const std::vector<bool>& episode_start,
    bool include_mask = false)
{
    const int64_t num_envs = static_cast<int64_t>(state_values.size());

    rl::BatchState state(
        MakeObs(state_values, include_mask),
        BoolTensor(BoolValues(num_envs, false)),
        BoolTensor(BoolValues(num_envs, false)),
        BoolTensor(episode_start));

    rl::BatchState next_state(
        MakeObs(next_values, include_mask),
        BoolTensor(next_done),
        BoolTensor(next_truncated),
        BoolTensor(BoolValues(num_envs, false)));

    auto actions = torch::zeros({ num_envs }, torch::TensorOptions().dtype(torch::kInt64));
    auto action_info = std::make_shared<rl::BatchActionInfo>(actions);

    return rl::BatchExperience(state, action_info, FloatVector(rewards), next_state);
}

rl::ReplayBufferConfig MakeConfig(
    int64_t capacity,
    int n_step = 1,
    float gamma = 0.99f,
    int stack_count = 1,
    rl::ReplaySamplerType sampler_type = rl::ReplaySamplerType::PRIORITIZED)
{
    rl::ReplayBufferConfig config;
    config.capacity = capacity;
    config.n_step = n_step;
    config.gamma = gamma;
    config.stack_count = stack_count;
    config.sampler_type = sampler_type;
    config.per_initial_priority = 1.0f;
    return config;
}

TestBuffer MakeBuffer(
    const rl::ReplayBufferConfig& config,
    int64_t num_envs,
    bool include_mask = false,
    uint64_t seed = 123)
{
    TestBuffer out;
    out.config = config;
    out.num_envs = num_envs;
    out.capacity_per_env = config.capacity / num_envs;
    out.rb = rl::CreateReplayBuffer(config, MakeEnvSpec(include_mask), num_envs, torch::kCPU, false, seed);
    return out;
}

void PushTime(
    const TestBuffer& buffer,
    int64_t time_idx,
    std::vector<bool> next_done = {},
    std::vector<bool> next_truncated = {},
    std::vector<bool> episode_start = {},
    std::vector<float> next_values = {},
    bool include_mask = false)
{
    if (next_done.empty()) {
        next_done = BoolValues(buffer.num_envs, false);
    }
    if (next_truncated.empty()) {
        next_truncated = BoolValues(buffer.num_envs, false);
    }
    if (episode_start.empty()) {
        episode_start = BoolValues(buffer.num_envs, false);
    }
    if (next_values.empty()) {
        next_values = StateValues(buffer.num_envs, time_idx + 1);
    }

    buffer.rb->Push(MakeBatch(
        StateValues(buffer.num_envs, time_idx),
        next_values,
        RewardValues(buffer.num_envs, time_idx),
        next_done,
        next_truncated,
        episode_start,
        include_mask));
}

int64_t IndexOf(const TestBuffer& buffer, int64_t env_idx, int64_t physical_time_idx)
{
    return env_idx * buffer.capacity_per_env + physical_time_idx;
}

void RequireShape(const torch::Tensor& tensor, const std::vector<int64_t>& expected)
{
    REQUIRE(tensor.sizes().vec() == expected);
}

void RequireFlatApprox(const torch::Tensor& tensor, const std::vector<float>& expected)
{
    auto flat = tensor.detach().cpu().to(torch::kFloat32).reshape({ -1 }).contiguous();
    REQUIRE(flat.numel() == static_cast<int64_t>(expected.size()));
    auto acc = flat.accessor<float, 1>();
    for (int64_t i = 0; i < static_cast<int64_t>(expected.size()); ++i) {
        REQUIRE(acc[i] == Catch::Approx(expected[static_cast<size_t>(i)]).margin(1.0e-5));
    }
}

torch::Tensor RequireSingleTensorVector(const std::optional<std::vector<torch::Tensor>>& opt_vec)
{
    REQUIRE(opt_vec.has_value());
    REQUIRE(opt_vec->size() == 1);
    REQUIRE((*opt_vec)[0].defined());
    return (*opt_vec)[0];
}

void RequireSampleIndex(const rl::ExperienceSamples& samples, int64_t expected_index)
{
    RequireShape(samples.indices, { 1 });
    REQUIRE(samples.indices.scalar_type() == torch::kInt64);
    REQUIRE(samples.indices[0].item<int64_t>() == expected_index);
}

void RequireSampleMeta(
    const rl::ExperienceSamples& samples,
    int64_t expected_index,
    float expected_return,
    bool expected_terminal,
    int64_t expected_n_steps)
{
    RequireSampleIndex(samples, expected_index);
    RequireShape(samples.target_returns, { 1 });
    RequireShape(samples.next_state.terminals, { 1 });
    RequireShape(samples.n_steps, { 1 });
    RequireShape(samples.is_weights, { 1 });

    REQUIRE(samples.target_returns[0].item<float>() == Catch::Approx(expected_return).margin(1.0e-5));
    REQUIRE(samples.next_state.terminals[0].item<bool>() == expected_terminal);
    REQUIRE(samples.n_steps[0].item<int64_t>() == expected_n_steps);
    REQUIRE(samples.is_weights[0].item<float>() == Catch::Approx(1.0f).margin(1.0e-5));
}

rl::ExperienceSamples SampleOnlyIndex(const TestBuffer& buffer, int64_t target_index)
{
    const int64_t total_capacity = buffer.capacity_per_env * buffer.num_envs;
    std::vector<int64_t> indices(static_cast<size_t>(total_capacity));
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<float> zeros(static_cast<size_t>(total_capacity), 0.0f);

    buffer.rb->UpdatePriorities(indices, zeros);
    buffer.rb->UpdatePriorities({ target_index }, { 1.0f });

    rl::ExperienceSamples samples;
    buffer.rb->Sample(samples, 1, 0.4f);
    RequireSampleIndex(samples, target_index);
    return samples;
}

float DiscountedReturn(int64_t env_idx, int64_t start_time, int n_step, float gamma)
{
    float result = 0.0f;
    float scale = 1.0f;
    for (int i = 0; i < n_step; ++i) {
        result += scale * RewardValue(env_idx, start_time + i);
        scale *= gamma;
    }
    return result;
}

void RequireOneStepSample(const rl::ExperienceSamples& samples, const TestBuffer& buffer, int64_t env_idx, int64_t time_idx)
{
    const int64_t index = IndexOf(buffer, env_idx, time_idx);
    RequireSampleMeta(samples, index, RewardValue(env_idx, time_idx), false, 1);
    RequireFlatApprox(samples.obs.At(kVectorKey)[0], { StateValue(env_idx, time_idx) });
    RequireFlatApprox(samples.next_state.next_obs.At(kVectorKey)[0], { StateValue(env_idx, time_idx + 1) });
}

} // namespace

TEST_CASE("ReplayBufferConfig has default values", "[replay_buffer][config]")
{
    rl::ReplayBufferConfig config;

    REQUIRE(config.capacity == 100000);
    REQUIRE(config.sampler_type == rl::ReplaySamplerType::UNIFORM);
    REQUIRE(config.n_step == 1);
    REQUIRE(config.gamma == Catch::Approx(0.99f));
    REQUIRE(config.stack_count == 1);
    REQUIRE(config.stack_keys.empty());
    REQUIRE(config.muzero.unroll_steps == 0);
}

TEST_CASE("ReplayBuffer sampled indices are valid sampleable storage indices", "[replay_buffer][indices]")
{
    constexpr int64_t num_envs = 2;
    constexpr int n_step = 3;
    constexpr float gamma = 0.5f;

    auto buffer = MakeBuffer(
        MakeConfig(40, n_step, gamma, 2, rl::ReplaySamplerType::UNIFORM),
        num_envs);

    for (int64_t t = 0; t <= 5; ++t) {
        PushTime(buffer, t);
    }

    REQUIRE(buffer.rb->Size() == 6);

    rl::ExperienceSamples samples;
    buffer.rb->Sample(samples, buffer.rb->Size(), 0.4f);

    RequireShape(samples.indices, { buffer.rb->Size() });
    REQUIRE(samples.indices.scalar_type() == torch::kInt64);

    auto idx_acc = samples.indices.accessor<int64_t, 1>();
    for (int64_t b = 0; b < samples.indices.size(0); ++b) {
        const int64_t idx = idx_acc[b];
        const int64_t env_idx = idx / buffer.capacity_per_env;
        const int64_t time_idx = idx % buffer.capacity_per_env;

        CAPTURE(idx, env_idx, time_idx);
        REQUIRE(idx >= 0);
        REQUIRE(idx < buffer.capacity_per_env * num_envs);
        REQUIRE(env_idx >= 0);
        REQUIRE(env_idx < num_envs);
        REQUIRE(time_idx >= 0);
        REQUIRE(time_idx <= 2);

        REQUIRE(samples.n_steps[b].item<int64_t>() == n_step);
        REQUIRE(samples.target_returns[b].item<float>() == Catch::Approx(DiscountedReturn(env_idx, time_idx, n_step, gamma)).margin(1.0e-5));
        REQUIRE(samples.next_state.terminals[b].item<bool>() == false);
        RequireFlatApprox(samples.obs.At(kVectorKey)[b], {
            StateValue(env_idx, time_idx - 1 < 0 ? time_idx : time_idx - 1),
            StateValue(env_idx, time_idx)
        });
    }
}

TEST_CASE("ReplayBuffer samples one-step transitions for each env", "[replay_buffer][basic][multi_env]")
{
    auto buffer = MakeBuffer(MakeConfig(20), 2);

    PushTime(buffer, 0, {}, {}, BoolValues(2, true));
    PushTime(buffer, 1);

    REQUIRE(buffer.rb->Size() == 2);

    RequireOneStepSample(SampleOnlyIndex(buffer, IndexOf(buffer, 0, 0)), buffer, 0, 0);
    RequireOneStepSample(SampleOnlyIndex(buffer, IndexOf(buffer, 1, 0)), buffer, 1, 0);
}

TEST_CASE("ReplayExperienceStorage initializes unwritten slots as episode boundaries", "[replay_buffer][storage][frame_stack]")
{
    constexpr int64_t num_envs = 2;
    constexpr int64_t capacity_per_env = 5;

    auto config = MakeConfig(num_envs * capacity_per_env, 1, 0.99f, 3);
    rl::ReplayExperienceStorage storage(
        num_envs,
        capacity_per_env,
        MakeEnvSpec(),
        config,
        torch::kCPU,
        false);

    RequireShape(storage.GetTargetReturns(), { num_envs, capacity_per_env });
    RequireShape(storage.GetTerminals(), { num_envs, capacity_per_env });
    RequireShape(storage.GetActualNSteps(), { num_envs, capacity_per_env });

    REQUIRE(storage.GetTargetReturns().eq(0.0f).all().item<bool>());
    REQUIRE(storage.GetTerminals().all().item<bool>());
    REQUIRE(storage.GetActualNSteps().eq(0).all().item<bool>());
}

TEST_CASE("ReplayBuffer visualization accessors expose V1-compatible storage keys", "[replay_buffer][visualization]")
{
    constexpr int64_t num_envs = 2;
    constexpr int n_step = 3;
    constexpr float gamma = 0.5f;

    auto buffer = MakeBuffer(MakeConfig(40, n_step, gamma), num_envs);

    for (int64_t t = 0; t <= 3; ++t) {
        PushTime(buffer, t, {}, {}, t == 0 ? BoolValues(num_envs, true) : BoolValues(num_envs, false));
    }

    REQUIRE(buffer.rb->Size() == num_envs);

    auto state = RequireSingleTensorVector(buffer.rb->GetTensorVector(rl::ReplayBuffer::STATE_OBS));
    RequireShape(state, { num_envs, 1 });
    RequireFlatApprox(state, { StateValue(0, 0), StateValue(1, 0) });

    auto next_state = RequireSingleTensorVector(buffer.rb->GetTensorVector(rl::ReplayBuffer::NEXT_STATE_OBS));
    RequireShape(next_state, { num_envs, 1 });
    RequireFlatApprox(next_state, { StateValue(0, 3), StateValue(1, 3) });

    auto action = RequireSingleTensorVector(buffer.rb->GetTensorVector(rl::ReplayBuffer::ACTION));
    RequireShape(action, { num_envs, 1 });
    RequireFlatApprox(action, { 0.0f, 0.0f });

    auto reward = RequireSingleTensorVector(buffer.rb->GetTensorVector(rl::ReplayBuffer::TARGET_RETURN));
    RequireShape(reward, { num_envs, 1 });
    RequireFlatApprox(reward, {
        DiscountedReturn(0, 0, n_step, gamma),
        DiscountedReturn(1, 0, n_step, gamma)
    });

    auto terminal = RequireSingleTensorVector(buffer.rb->GetTensorVector(rl::ReplayBuffer::NEXT_STATE_TERMINAL));
    RequireShape(terminal, { num_envs, 1 });
    RequireFlatApprox(terminal, { 0.0f, 0.0f });

    auto n_steps = RequireSingleTensorVector(buffer.rb->GetTensorVector(rl::ReplayBuffer::N_STEP));
    RequireShape(n_steps, { num_envs, 1 });
    RequireFlatApprox(n_steps, { static_cast<float>(n_step), static_cast<float>(n_step) });

    auto reward_tensor = buffer.rb->GetTensor(rl::ReplayBuffer::TARGET_RETURN);
    REQUIRE(reward_tensor.has_value());
    RequireShape(*reward_tensor, { num_envs, 1 });
    RequireFlatApprox(*reward_tensor, {
        DiscountedReturn(0, 0, n_step, gamma),
        DiscountedReturn(1, 0, n_step, gamma)
    });
}

TEST_CASE("ReplayBuffer visualization accessors expose single frames when samples use frame stack", "[replay_buffer][visualization][frame_stack]")
{
    constexpr int64_t num_envs = 1;
    constexpr int stack_count = 4;

    auto buffer = MakeBuffer(MakeConfig(20, 1, 0.99f, stack_count), num_envs);

    for (int64_t t = 0; t <= 4; ++t) {
        PushTime(buffer, t, {}, {}, t == 0 ? BoolValues(num_envs, true) : BoolValues(num_envs, false));
    }

    REQUIRE(buffer.rb->Size() == 4);

    auto samples = SampleOnlyIndex(buffer, IndexOf(buffer, 0, 3));
    RequireFlatApprox(samples.obs.At(kVectorKey)[0], {
        StateValue(0, 0),
        StateValue(0, 1),
        StateValue(0, 2),
        StateValue(0, 3)
    });

    auto state = RequireSingleTensorVector(buffer.rb->GetTensorVector(rl::ReplayBuffer::STATE_OBS));
    RequireShape(state, { 4, 1 });
    RequireFlatApprox(state, {
        StateValue(0, 0),
        StateValue(0, 1),
        StateValue(0, 2),
        StateValue(0, 3)
    });

    auto next_state = RequireSingleTensorVector(buffer.rb->GetTensorVector(rl::ReplayBuffer::NEXT_STATE_OBS));
    RequireShape(next_state, { 4, 1 });
    RequireFlatApprox(next_state, {
        StateValue(0, 1),
        StateValue(0, 2),
        StateValue(0, 3),
        StateValue(0, 4)
    });
}

TEST_CASE("ReplayBuffer visualization accessors expose observation subkeys", "[replay_buffer][visualization]")
{
    constexpr int64_t num_envs = 1;
    auto buffer = MakeBuffer(MakeConfig(20), num_envs, true);

    PushTime(buffer, 0, {}, {}, BoolValues(num_envs, true), {}, true);
    PushTime(buffer, 1, {}, {}, {}, {}, true);

    REQUIRE(buffer.rb->Size() == 1);

    const std::string next_vector_key = std::string(rl::ReplayBuffer::NEXT_STATE_OBS) + ".vector";
    auto next_vector = RequireSingleTensorVector(buffer.rb->GetTensorVector(next_vector_key));
    RequireShape(next_vector, { 1, 1 });
    RequireFlatApprox(next_vector, { StateValue(0, 1) });

    const std::string next_mask_key = std::string(rl::ReplayBuffer::NEXT_STATE_OBS) + ".action_mask";
    auto next_mask = RequireSingleTensorVector(buffer.rb->GetTensorVector(next_mask_key));
    RequireShape(next_mask, { 1, 2 });
    RequireFlatApprox(next_mask, { StateValue(0, 1), StateValue(0, 1) + 0.25f });

    const std::string missing_key = std::string(rl::ReplayBuffer::NEXT_STATE_OBS) + ".missing";
    REQUIRE_THROWS(buffer.rb->GetTensorVector(missing_key));
}

TEST_CASE("ReplayBuffer visualization accessors expose PER priorities", "[replay_buffer][visualization][per]")
{
    constexpr int64_t num_envs = 2;
    auto buffer = MakeBuffer(MakeConfig(40), num_envs);

    PushTime(buffer, 0, {}, {}, BoolValues(num_envs, true));
    PushTime(buffer, 1);

    REQUIRE(buffer.rb->Size() == num_envs);

    const int64_t env0_index = IndexOf(buffer, 0, 0);
    const int64_t env1_index = IndexOf(buffer, 1, 0);
    buffer.rb->UpdatePriorities({ env0_index, env1_index }, { 4.0f, 9.0f });

    const float env0_priority = std::sqrt(4.0f);
    const float env1_priority = std::sqrt(9.0f);

    auto total = buffer.rb->GetScalar(rl::ReplayBuffer::PER_TOTAL);
    REQUIRE(total.has_value());
    REQUIRE(*total == Catch::Approx(env0_priority + env1_priority + 2.0f).margin(1.0e-5));

    auto values = RequireSingleTensorVector(buffer.rb->GetTensorVector(rl::ReplayBuffer::PER_VALUES));
    RequireShape(values, { num_envs, 1 });
    RequireFlatApprox(values, { env0_priority, env1_priority });

    // PER_DIST は正規化サンプリング確率 p/total を返す。
    const float per_total = env0_priority + env1_priority + 2.0f;
    auto distribution = RequireSingleTensorVector(buffer.rb->GetTensorVector(rl::ReplayBuffer::PER_DIST));
    RequireShape(distribution, { num_envs, 1 });
    RequireFlatApprox(distribution, { env0_priority / per_total, env1_priority / per_total });

    auto env0_value = buffer.rb->GetTensor(rl::ReplayBuffer::PER_VALUES, env0_index);
    REQUIRE(env0_value.has_value());
    RequireFlatApprox(*env0_value, { env0_priority });
}

TEST_CASE("ReplayBuffer PER visualization keys are unavailable for uniform sampling", "[replay_buffer][visualization][per]")
{
    auto buffer = MakeBuffer(MakeConfig(20, 1, 0.99f, 1, rl::ReplaySamplerType::UNIFORM), 1);

    PushTime(buffer, 0, {}, {}, BoolValues(1, true));
    PushTime(buffer, 1);

    REQUIRE(buffer.rb->Size() == 1);
    REQUIRE_FALSE(buffer.rb->GetScalar(rl::ReplayBuffer::PER_TOTAL).has_value());
    REQUIRE_FALSE(buffer.rb->GetTensor(rl::ReplayBuffer::PER_VALUES, 0).has_value());
    REQUIRE_FALSE(buffer.rb->GetTensorVector(rl::ReplayBuffer::PER_VALUES).has_value());
    REQUIRE_FALSE(buffer.rb->GetTensorVector(rl::ReplayBuffer::PER_DIST).has_value());
}

TEST_CASE("ReplayBuffer samples while push and priority update run concurrently", "[replay_buffer][thread]")
{
    auto buffer = MakeBuffer(MakeConfig(32, 1, 0.99f, 1, rl::ReplaySamplerType::PRIORITIZED), 2);

    for (int64_t t = 0; t < 10; ++t) {
        PushTime(buffer, t);
    }

    std::exception_ptr worker_error;
    std::thread worker([&]() {
        try {
            for (int i = 0; i < 100; ++i) {
                rl::ExperienceSamples samples;
                buffer.rb->Sample(samples, 4, 0.4f);

                if (samples.indices.sizes().vec() != std::vector<int64_t>{ 4 }) {
                    throw std::runtime_error("Unexpected sample index shape.");
                }
                if (samples.actions.sizes().vec() != std::vector<int64_t>{ 4 }) {
                    throw std::runtime_error("Unexpected sample action shape.");
                }

                auto indices_cpu = samples.indices.cpu().contiguous();
                auto indices_ptr = indices_cpu.data_ptr<int64_t>();
                std::vector<int64_t> indices(indices_ptr, indices_ptr + indices_cpu.size(0));
                std::vector<float> priorities(indices.size(), 1.0f + static_cast<float>(i % 7));
                buffer.rb->UpdatePriorities(indices, priorities);
            }
        } catch (...) {
            worker_error = std::current_exception();
        }
    });

    for (int64_t t = 10; t < 80; ++t) {
        PushTime(buffer, t);
    }

    worker.join();
    if (worker_error) {
        std::rethrow_exception(worker_error);
    }

    rl::ExperienceSamples samples;
    buffer.rb->Sample(samples, 4, 0.4f);
    RequireShape(samples.indices, { 4 });
    RequireShape(samples.actions, { 4 });
}

TEST_CASE("ReplayBuffer computes n-step returns independently for each env", "[replay_buffer][n_step][multi_env]")
{
    constexpr int64_t num_envs = 2;
    constexpr int n_step = 3;
    constexpr float gamma = 0.5f;

    auto buffer = MakeBuffer(MakeConfig(40, n_step, gamma), num_envs);

    for (int64_t t = 0; t <= 3; ++t) {
        PushTime(buffer, t, {}, {}, t == 0 ? BoolValues(num_envs, true) : BoolValues(num_envs, false));
    }

    REQUIRE(buffer.rb->Size() == 2);

    for (int64_t env = 0; env < num_envs; ++env) {
        auto samples = SampleOnlyIndex(buffer, IndexOf(buffer, env, 0));
        RequireSampleMeta(samples, IndexOf(buffer, env, 0), DiscountedReturn(env, 0, n_step, gamma), false, n_step);
        RequireFlatApprox(samples.obs.At(kVectorKey)[0], { StateValue(env, 0) });
        RequireFlatApprox(samples.next_state.next_obs.At(kVectorKey)[0], { StateValue(env, 3) });
    }
}

TEST_CASE("ReplayBuffer flushes n-step returns at done terminals", "[replay_buffer][n_step][done][multi_env]")
{
    constexpr int64_t num_envs = 2;
    constexpr int n_step = 3;
    constexpr float gamma = 0.5f;

    auto buffer = MakeBuffer(MakeConfig(40, n_step, gamma), num_envs);

    PushTime(buffer, 0, {}, {}, BoolValues(num_envs, true));
    PushTime(buffer, 1, BoolValues(num_envs, true));
    PushTime(buffer, 2, {}, {}, BoolValues(num_envs, true));
    PushTime(buffer, 3);
    PushTime(buffer, 4);

    REQUIRE(buffer.rb->Size() == 4);

    for (int64_t env = 0; env < num_envs; ++env) {
        auto first = SampleOnlyIndex(buffer, IndexOf(buffer, env, 0));
        RequireSampleMeta(
            first,
            IndexOf(buffer, env, 0),
            RewardValue(env, 0) + gamma * RewardValue(env, 1),
            true,
            2);

        auto terminal = SampleOnlyIndex(buffer, IndexOf(buffer, env, 1));
        RequireSampleMeta(terminal, IndexOf(buffer, env, 1), RewardValue(env, 1), true, 1);
    }
}

TEST_CASE("ReplayBuffer treats truncated transitions as bootstrapable n-step boundaries", "[replay_buffer][n_step][truncated][multi_env]")
{
    constexpr int64_t num_envs = 2;
    constexpr int n_step = 3;
    constexpr float gamma = 0.5f;

    auto buffer = MakeBuffer(MakeConfig(40, n_step, gamma), num_envs);

    PushTime(buffer, 0, {}, {}, BoolValues(num_envs, true));
    PushTime(
        buffer,
        1,
        BoolValues(num_envs, false),
        BoolValues(num_envs, true),
        BoolValues(num_envs, false),
        TerminalStateValues(num_envs, 1));
    PushTime(buffer, 2, {}, {}, BoolValues(num_envs, true));
    PushTime(buffer, 3);
    PushTime(buffer, 4);

    REQUIRE(buffer.rb->Size() == 4);

    for (int64_t env = 0; env < num_envs; ++env) {
        auto first = SampleOnlyIndex(buffer, IndexOf(buffer, env, 0));
        RequireSampleMeta(
            first,
            IndexOf(buffer, env, 0),
            RewardValue(env, 0) + gamma * RewardValue(env, 1),
            false,
            2);
        RequireFlatApprox(first.next_state.next_obs.At(kVectorKey)[0], { TerminalStateValue(env, 1) });

        auto truncated = SampleOnlyIndex(buffer, IndexOf(buffer, env, 1));
        RequireSampleMeta(truncated, IndexOf(buffer, env, 1), RewardValue(env, 1), false, 1);
        RequireFlatApprox(truncated.next_state.next_obs.At(kVectorKey)[0], { TerminalStateValue(env, 1) });
    }
}

TEST_CASE("ReplayBuffer frame stacking pads the beginning of an episode", "[replay_buffer][frame_stack][episode_boundary]")
{
    auto buffer = MakeBuffer(MakeConfig(20, 1, 0.99f, 3), 1);

    PushTime(buffer, 0, {}, {}, BoolValues(1, true));
    PushTime(buffer, 1);

    REQUIRE(buffer.rb->Size() == 1);

    auto samples = SampleOnlyIndex(buffer, IndexOf(buffer, 0, 0));
    RequireSampleMeta(samples, IndexOf(buffer, 0, 0), RewardValue(0, 0), false, 1);
    RequireFlatApprox(samples.obs.At(kVectorKey)[0], { StateValue(0, 0), StateValue(0, 0), StateValue(0, 0) });
    RequireFlatApprox(samples.next_state.next_obs.At(kVectorKey)[0], { StateValue(0, 0), StateValue(0, 0), StateValue(0, 1) });
}

TEST_CASE("ReplayBuffer frame stacking pads the initial sample for nonzero env values", "[replay_buffer][frame_stack][episode_boundary][multi_env]")
{
    constexpr int64_t num_envs = 2;

    auto buffer = MakeBuffer(MakeConfig(40, 1, 0.99f, 3), num_envs);

    PushTime(buffer, 0, {}, {}, BoolValues(num_envs, true));
    PushTime(buffer, 1);

    REQUIRE(buffer.rb->Size() == num_envs);

    auto samples = SampleOnlyIndex(buffer, IndexOf(buffer, 1, 0));
    RequireSampleMeta(samples, IndexOf(buffer, 1, 0), RewardValue(1, 0), false, 1);
    RequireFlatApprox(samples.obs.At(kVectorKey)[0], { StateValue(1, 0), StateValue(1, 0), StateValue(1, 0) });
    RequireFlatApprox(samples.next_state.next_obs.At(kVectorKey)[0], { StateValue(1, 0), StateValue(1, 0), StateValue(1, 1) });
}

TEST_CASE("ReplayBuffer frame stacking does not cross done boundaries per env", "[replay_buffer][frame_stack][done][multi_env]")
{
    constexpr int64_t num_envs = 2;

    auto buffer = MakeBuffer(MakeConfig(40, 1, 0.99f, 3), num_envs);

    PushTime(buffer, 0, {}, {}, BoolValues(num_envs, true));
    PushTime(buffer, 1, { true, false });
    PushTime(buffer, 2, {}, {}, { true, false });
    PushTime(buffer, 3);

    REQUIRE(buffer.rb->Size() == 6);

    auto env0 = SampleOnlyIndex(buffer, IndexOf(buffer, 0, 2));
    RequireSampleMeta(env0, IndexOf(buffer, 0, 2), RewardValue(0, 2), false, 1);
    RequireFlatApprox(env0.obs.At(kVectorKey)[0], { StateValue(0, 2), StateValue(0, 2), StateValue(0, 2) });
    RequireFlatApprox(env0.next_state.next_obs.At(kVectorKey)[0], { StateValue(0, 2), StateValue(0, 2), StateValue(0, 3) });

    auto env1 = SampleOnlyIndex(buffer, IndexOf(buffer, 1, 2));
    RequireSampleMeta(env1, IndexOf(buffer, 1, 2), RewardValue(1, 2), false, 1);
    RequireFlatApprox(env1.obs.At(kVectorKey)[0], { StateValue(1, 0), StateValue(1, 1), StateValue(1, 2) });
    RequireFlatApprox(env1.next_state.next_obs.At(kVectorKey)[0], { StateValue(1, 1), StateValue(1, 2), StateValue(1, 3) });
}

TEST_CASE("ReplayBuffer stack_keys leaves non-stacked observations at latest frame", "[replay_buffer][frame_stack][stack_keys]")
{
    auto config = MakeConfig(20, 1, 0.99f, 3);
    config.stack_keys = { kVectorKey };
    auto buffer = MakeBuffer(config, 1, true);

    for (int64_t t = 0; t <= 3; ++t) {
        PushTime(buffer, t, {}, {}, t == 0 ? BoolValues(1, true) : BoolValues(1, false), {}, true);
    }

    REQUIRE(buffer.rb->Size() == 3);

    auto samples = SampleOnlyIndex(buffer, IndexOf(buffer, 0, 2));
    RequireSampleMeta(samples, IndexOf(buffer, 0, 2), RewardValue(0, 2), false, 1);

    RequireFlatApprox(samples.obs.At(kVectorKey)[0], { StateValue(0, 0), StateValue(0, 1), StateValue(0, 2) });
    RequireFlatApprox(samples.next_state.next_obs.At(kVectorKey)[0], { StateValue(0, 1), StateValue(0, 2), StateValue(0, 3) });

    RequireShape(samples.obs.At(kMaskKey), { 1, 2 });
    RequireShape(samples.next_state.next_obs.At(kMaskKey), { 1, 2 });
    RequireFlatApprox(samples.obs.At(kMaskKey)[0], { StateValue(0, 2), StateValue(0, 2) + 0.25f });
    RequireFlatApprox(samples.next_state.next_obs.At(kMaskKey)[0], { StateValue(0, 3), StateValue(0, 3) + 0.25f });
}
