#include "catch.hpp"

#include "anet/replay_buffer.hpp"
#include "anet/transfer.hpp"
#include "replay_buffer_impl.hpp"

#include <torch/cuda.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

// ============================================================================
// ReplayBuffer API test coverage model
//
// このファイルの [replay_buffer] テストは、内部 storage 表現ではなく public API
// Push / Sample / Size / UpdatePriorities / visualization accessors の契約を検証する。
// テストを追加するときは、以下の「観点」「条件」「確認事項」の組み合わせを確認し、
// 全直積ではなく、仕様上リスクの高い相互作用を代表シナリオで覆う。
//
// テスト観点:
// - Core transition alignment: Sample が同一 env/time の obs, action, return,
//   next_obs, terminal, n_steps, indices を一貫して返す。
// - Temporal horizons: n_step return と bootstrap next_obs が正しい時点を指す。
// - Episode boundaries: done, truncated, episode_start, unwritten slot, ring wrap を
//   sampleable range と frame stack 境界として正しく扱う。
// - Frame stacking: stack_count と stack_keys に従って過去フレームを積み、境界では
//   crossing せず padding する。
// - Isolation and capacity: multi-env 間で境界や時系列が混ざらず、capacity / wrap 後も
//   有効 index だけを sample する。
// - Priority and decorators: PER metadata / UpdatePriorities / prefetch / transfer が
//   ReplayBuffer の sample 契約を壊さない。
//
// テスト条件:
// - env: single-env / multi-env
// - horizon: n_step=1 / n_step>1 / unroll_steps=0 / unroll_steps>0
// - stack: stack_count=1 / stack_count>1 / stack_keys 指定あり
// - sampler: uniform / PER / fixed-index sampling
// - boundary: 通常遷移 / done / truncated / episode_start / unwritten slot / ring wrap
// - execution: direct buffer / prefetch / CPU transfer / CUDA transfer where available
//
// 確認事項:
// - values: obs, action, target_returns, next_obs, terminal, n_steps に関する 入出力と時系列の整合
// - shapes: stack/unroll/squeeze/TensorDict key ごとの shape 契約
// - sampleability: Size, sampled indices, overwritten slot を sample しないこと
// - boundaries: padding, no-crossing, bootstrap 可否, env 間 isolation
// - metadata: is_weights, per_is_initial_priority, priority visualization/accessors
//
// 運用:
// - TEST_CASE 名と tags は主条件を表す。例: [frame_stack][n_step][done]
// - fixed-index sampling を使える仕様テストでは、乱択ではなく IndexOf で対象 transition を固定する。
// - 1つの代表シナリオで複数の確認事項を assert し、時系列整合をまとめて検証する。
// ============================================================================

namespace {

namespace rl = anet::rl;

constexpr const char* kVectorKey = rl::ObsKeys::kVector;
constexpr const char* kMaskKey = rl::ObsKeys::kActionMask;
constexpr const char* kPolicyInfoKey = "policy_info";

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

float PolicyInfoValue(int64_t env_idx, int64_t time_idx)
{
    return static_cast<float>(env_idx * 100 + 500 + time_idx);
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

std::vector<float> PolicyInfoValues(int64_t num_envs, int64_t time_idx)
{
    std::vector<float> values;
    values.reserve(static_cast<size_t>(num_envs));
    for (int64_t env = 0; env < num_envs; ++env) {
        values.push_back(PolicyInfoValue(env, time_idx));
    }
    return values;
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
    bool include_mask = false,
    anet::TensorDict action_info_dict = anet::TensorDict())
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
    auto action_info = std::make_shared<rl::BatchActionInfo>(actions, action_info_dict);

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

std::vector<int64_t> TensorToInt64Vector(const torch::Tensor& tensor)
{
    auto cpu = tensor.cpu().contiguous();
    auto ptr = cpu.data_ptr<int64_t>();
    return std::vector<int64_t>(ptr, ptr + cpu.numel());
}

std::vector<float> TensorToFloatVector(const torch::Tensor& tensor)
{
    auto cpu = tensor.detach().cpu().to(torch::kFloat32).reshape({ -1 }).contiguous();
    auto ptr = cpu.data_ptr<float>();
    return std::vector<float>(ptr, ptr + cpu.numel());
}

bool WaitForFlag(const std::atomic<bool>& flag, std::chrono::milliseconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!flag.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return flag.load();
}

class BlockingReplayBuffer final : public rl::ReplayBuffer {
public:
    explicit BlockingReplayBuffer(bool block_first_sample = false, bool block_second_sample = true)
        : block_first_sample(block_first_sample)
        , block_second_sample(block_second_sample)
    {
    }

    void Push(const rl::BatchExperience& batch_exp) override
    {
        std::lock_guard<std::mutex> lock(mutex);
        push_called_before_first_sample_finished = first_sample_started && !first_sample_finished;
        push_called_before_second_sample_finished = second_sample_started && !second_sample_finished;
        last_pushed_reward = batch_exp.reward.defined() ? batch_exp.reward.clone() : torch::Tensor();
        last_pushed_state_obs = batch_exp.state.obs.Contains(kVectorKey)
            ? batch_exp.state.obs.At(kVectorKey).clone()
            : torch::Tensor();
        ++push_count;
        cv.notify_all();
    }

    void Sample(rl::ExperienceSamples& out_samples, int64_t minibatch_size, float) const override
    {
        int call_index = 0;
        {
            std::unique_lock<std::mutex> lock(mutex);
            call_index = ++sample_count;
            push_count_at_sample_start.push_back(push_count);
            cv.notify_all();
            if (call_index == 1 && block_first_sample) {
                first_sample_started = true;
                cv.notify_all();
                cv.wait(lock, [&] { return release_first_sample; });
                first_sample_finished = true;
                cv.notify_all();
            }
            if (call_index == 2 && block_second_sample) {
                second_sample_started = true;
                cv.notify_all();
                cv.wait(lock, [&] { return release_second_sample; });
                second_sample_finished = true;
                cv.notify_all();
            }
        }

        anet::TensorDict obs;
        obs.Set(kVectorKey, torch::zeros({ minibatch_size, 1 }, torch::TensorOptions().dtype(torch::kFloat32)));
        out_samples.obs = obs;
        out_samples.actions = torch::zeros({ minibatch_size }, torch::TensorOptions().dtype(torch::kInt64));
        out_samples.target_returns = torch::zeros({ minibatch_size }, torch::TensorOptions().dtype(torch::kFloat32));
        out_samples.next_state.next_obs = obs;
        out_samples.next_state.terminals = torch::zeros({ minibatch_size }, torch::TensorOptions().dtype(torch::kBool));
        out_samples.n_steps = torch::ones({ minibatch_size }, torch::TensorOptions().dtype(torch::kInt64));
        out_samples.indices = torch::full({ minibatch_size }, call_index, torch::TensorOptions().dtype(torch::kInt64));
        out_samples.is_weights = torch::ones({ minibatch_size }, torch::TensorOptions().dtype(torch::kFloat32));
    }

    int64_t Size() const override
    {
        return size_value;
    }

    void UpdatePriorities(const std::vector<int64_t>&, const std::vector<float>&) override
    {
        std::lock_guard<std::mutex> lock(mutex);
        update_called_before_second_sample_finished = second_sample_started && !second_sample_finished;
        ++update_count;
        cv.notify_all();
    }

    std::optional<float> GetScalar(const std::string& key, int64_t) const override
    {
        if (key == rl::ReplayBuffer::PER_TOTAL) return scalar_value;
        return std::nullopt;
    }

    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t) const override
    {
        return tensor_value;
    }

    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t) const override
    {
        return std::vector<torch::Tensor>{ tensor_value };
    }

    void WaitForFirstSampleStarted() const
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return first_sample_started; });
    }

    void WaitForSecondSampleStarted() const
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return second_sample_started; });
    }

    void WaitForSampleCount(int expected_count) const
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return sample_count >= expected_count; });
    }

    void WaitForPushCount(int expected_count) const
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return push_count >= expected_count; });
    }

    bool WaitForPushCount(int expected_count, std::chrono::milliseconds timeout) const
    {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, timeout, [&] { return push_count >= expected_count; });
    }

    void ReleaseFirstSample() const
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            release_first_sample = true;
        }
        cv.notify_all();
    }

    void ReleaseSecondSample() const
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            release_second_sample = true;
        }
        cv.notify_all();
    }

    int UpdateCount() const
    {
        std::lock_guard<std::mutex> lock(mutex);
        return update_count;
    }

    int PushCount() const
    {
        std::lock_guard<std::mutex> lock(mutex);
        return push_count;
    }

    int PushCountAtSampleStart(int sample_index) const
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (sample_index <= 0 || sample_index > static_cast<int>(push_count_at_sample_start.size())) {
            return -1;
        }
        return push_count_at_sample_start[static_cast<size_t>(sample_index - 1)];
    }

    torch::Tensor LastPushedReward() const
    {
        std::lock_guard<std::mutex> lock(mutex);
        return last_pushed_reward.defined() ? last_pushed_reward.clone() : torch::Tensor();
    }

    torch::Tensor LastPushedStateObs() const
    {
        std::lock_guard<std::mutex> lock(mutex);
        return last_pushed_state_obs.defined() ? last_pushed_state_obs.clone() : torch::Tensor();
    }

    bool UpdateWasCalledBeforeSecondSampleFinished() const
    {
        std::lock_guard<std::mutex> lock(mutex);
        return update_called_before_second_sample_finished;
    }

    bool PushWasCalledBeforeFirstSampleFinished() const
    {
        std::lock_guard<std::mutex> lock(mutex);
        return push_called_before_first_sample_finished;
    }

    bool PushWasCalledBeforeSecondSampleFinished() const
    {
        std::lock_guard<std::mutex> lock(mutex);
        return push_called_before_second_sample_finished;
    }

    bool block_first_sample = false;
    bool block_second_sample = true;
    mutable std::mutex mutex;
    mutable std::condition_variable cv;
    mutable int sample_count = 0;
    mutable std::vector<int> push_count_at_sample_start;
    mutable bool first_sample_started = false;
    mutable bool release_first_sample = false;
    mutable bool first_sample_finished = false;
    mutable bool second_sample_started = false;
    mutable bool release_second_sample = false;
    mutable bool second_sample_finished = false;
    mutable int update_count = 0;
    mutable bool update_called_before_second_sample_finished = false;
    mutable int push_count = 0;
    mutable bool push_called_before_first_sample_finished = false;
    mutable bool push_called_before_second_sample_finished = false;
    torch::Tensor last_pushed_reward;
    torch::Tensor last_pushed_state_obs;
    int64_t size_value = 7;
    float scalar_value = 3.5f;
    torch::Tensor tensor_value = torch::tensor({ 2.0f }, torch::TensorOptions().dtype(torch::kFloat32));
};

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

void PushTimeWithActionInfo(
    const TestBuffer& buffer,
    int64_t time_idx,
    const anet::TensorDict& action_info,
    std::vector<bool> episode_start = {})
{
    if (episode_start.empty()) {
        episode_start = BoolValues(buffer.num_envs, false);
    }

    buffer.rb->Push(MakeBatch(
        StateValues(buffer.num_envs, time_idx),
        StateValues(buffer.num_envs, time_idx + 1),
        RewardValues(buffer.num_envs, time_idx),
        BoolValues(buffer.num_envs, false),
        BoolValues(buffer.num_envs, false),
        episode_start,
        false,
        action_info));
}

rl::ExperienceSamples MakeTransferSamples()
{
    return rl::ExperienceSamples{
        .obs = MakeObs({ 1.0f, 2.0f }),
        .actions = torch::tensor({ 1, 2 }, torch::TensorOptions().dtype(torch::kInt64)),
        .target_returns = FloatVector({ 3.0f, 4.0f }),
        .next_state = {
            .next_obs = MakeObs({ 5.0f, 6.0f }),
            .terminals = BoolTensor({ false, true }),
        },
        .n_steps = torch::tensor({ 1, 2 }, torch::TensorOptions().dtype(torch::kInt64)),
        .indices = torch::tensor({ 7, 8 }, torch::TensorOptions().dtype(torch::kInt64)),
        .is_weights = FloatVector({ 0.5f, 0.25f }),
        .per_is_initial_priority = BoolTensor({ true, false }),
        .info = anet::TensorDict("info", FloatVector({ 9.0f, 10.0f }))
    };
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

void RequireFlatApproxUnordered(const torch::Tensor& tensor, std::vector<float> expected)
{
    auto actual = TensorToFloatVector(tensor);
    REQUIRE(actual.size() == expected.size());

    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());

    for (size_t i = 0; i < actual.size(); ++i) {
        REQUIRE(actual[i] == Catch::Approx(expected[i]).margin(1.0e-5));
    }
}

void RequireIndicesIn(const torch::Tensor& tensor, const std::vector<int64_t>& allowed)
{
    auto indices = TensorToInt64Vector(tensor);
    for (const int64_t idx : indices) {
        CAPTURE(idx);
        REQUIRE(std::find(allowed.begin(), allowed.end(), idx) != allowed.end());
    }
}

TEST_CASE("SumTree supports explicit floating-point value types", "[replay_buffer][per][precision]")
{
    // float / double のどちらでも基本的な更新と重み付き探索が成立することを確認する。
    auto verify = []<typename Value>() {
        rl::SumTree<Value> tree(4);
        tree.Update(0, static_cast<Value>(1.0));
        tree.Update(1, static_cast<Value>(2.0));
        tree.Update(2, static_cast<Value>(3.0));
        tree.Update(3, static_cast<Value>(4.0));

        REQUIRE(tree.GetTotalPriority() == Catch::Approx(10.0).margin(1.0e-6));
        REQUIRE(tree.GetPriority(2) == Catch::Approx(3.0).margin(1.0e-6));
        REQUIRE(tree.Retrieve(static_cast<Value>(0.5)) == 0);
        REQUIRE(tree.Retrieve(static_cast<Value>(1.5)) == 1);
        REQUIRE(tree.Retrieve(static_cast<Value>(4.0)) == 2);
        REQUIRE(tree.Retrieve(static_cast<Value>(9.5)) == 3);
    };

    verify.operator()<float>();
    verify.operator()<double>();
}

TEST_CASE("SumTree double aggregation preserves small priority updates", "[replay_buffer][per][precision]")
{
    constexpr int64_t capacity = 65536;
    const float initial_priority = std::pow(0.3f, 0.2f);
    const float updated_priority = initial_priority + 0.001f;
    rl::SumTree<double> tree(capacity);

    // total が大きくなった後も、各 leaf の小さな差分を失わず集計できることを確認する。
    for (int64_t index = 0; index < capacity; ++index) {
        tree.Update(index, static_cast<double>(initial_priority));
    }
    for (int64_t index = 0; index < capacity; ++index) {
        tree.Update(index, static_cast<double>(updated_priority));
    }

    const double expected_total = static_cast<double>(capacity) * static_cast<double>(updated_priority);
    REQUIRE(tree.GetTotalPriority() == Catch::Approx(expected_total).margin(1.0e-6));
}

TEST_CASE("ExperienceSamples ForEachTensor visits defined tensors", "[transfer]")
{
    auto samples = MakeTransferSamples();
    samples.info.Set("undefined", torch::Tensor());

    int visit_count = 0;
    samples.ForEachTensor([&](torch::Tensor& tensor) {
        ++visit_count;
        if (tensor.dtype() == torch::kFloat32) {
            tensor = tensor + 1.0f;
        }
    });

    CHECK(visit_count == 10);
    RequireFlatApprox(samples.obs.At(kVectorKey), { 2.0f, 3.0f });
    RequireFlatApprox(samples.target_returns, { 4.0f, 5.0f });
    RequireFlatApprox(samples.next_state.next_obs.At(kVectorKey), { 6.0f, 7.0f });
    RequireFlatApprox(samples.is_weights, { 1.5f, 1.25f });
    RequireFlatApprox(samples.per_is_initial_priority, { 1.0f, 0.0f });
    RequireFlatApprox(samples.info.At("info"), { 10.0f, 11.0f });
    CHECK_FALSE(samples.info.At("undefined").defined());
}

TEST_CASE("DeviceTransfer CPU constructor keeps transfer synchronous", "[transfer]")
{
    anet::transfer::DeviceTransfer<rl::ExperienceSamples> transfer(MakeTransferSamples(), torch::kCPU);

    CHECK_FALSE(transfer.ready_event.has_value());
    CHECK(transfer.device_samples.actions.device().is_cpu());
    CHECK(transfer.device_samples.indices.device().is_cpu());
    CHECK(transfer.device_samples.per_is_initial_priority.device().is_cpu());
    CHECK_FALSE(transfer.retained_source.actions.defined());
    RequireFlatApprox(transfer.device_samples.target_returns, { 3.0f, 4.0f });
}

TEST_CASE("HostReadback CPU constructor keeps tensor immediately readable", "[transfer]")
{
    anet::transfer::HostReadback readback(FloatVector({ 1.5f, 2.5f }));

    CHECK_FALSE(readback.source_ready_event.has_value());
    CHECK_FALSE(readback.ready_event.has_value());
    CHECK(readback.pinned_result.device().is_cpu());
    RequireFlatApprox(readback.pinned_result, { 1.5f, 2.5f });
}

TEST_CASE("EventRecycler reuses completed unrecorded events", "[transfer]")
{
    anet::transfer::EventRecycler<torch::Tensor> event_recycler;

    auto event = event_recycler.Acquire();
    CHECK(event_recycler.PendingCount() == 0);
    CHECK(event_recycler.FreeCount() == 0);

    event_recycler.Retire(std::move(event), torch::ones({ 1 }, torch::TensorOptions().dtype(torch::kFloat32)));
    CHECK(event_recycler.PendingCount() == 1);
    CHECK(event_recycler.FreeCount() == 0);

    event_recycler.Poll();
    CHECK(event_recycler.PendingCount() == 0);
    CHECK(event_recycler.FreeCount() == 1);

    auto reused_event = event_recycler.Acquire();
    CHECK(event_recycler.PendingCount() == 0);
    CHECK(event_recycler.FreeCount() == 0);

    event_recycler.Retire(std::move(reused_event), torch::ones({ 1 }, torch::TensorOptions().dtype(torch::kFloat32)));
    event_recycler.Drain();
    CHECK(event_recycler.PendingCount() == 0);
    CHECK(event_recycler.FreeCount() == 1);
}

TEST_CASE("DeviceTransfer CUDA constructor copies and records stream when available", "[transfer][cuda]")
{
    if (!torch::cuda::is_available()) {
        SUCCEED("CUDA is not available.");
        return;
    }

    auto samples = MakeTransferSamples();
    auto strided = torch::arange(12, torch::TensorOptions().dtype(torch::kFloat32)).reshape({ 3, 4 }).transpose(0, 1);
    const auto strided_sizes = strided.sizes().vec();
    const auto strided_strides = strided.strides().vec();
    samples.info.Set("strided", strided);

    anet::transfer::EventRecycler<rl::ExperienceSamples> event_recycler;
    auto copy_stream = at::cuda::getStreamFromPool(false);
    anet::transfer::DeviceTransfer<rl::ExperienceSamples> transfer(
        std::move(samples),
        torch::Device(torch::kCUDA),
        copy_stream,
        event_recycler);

    REQUIRE(transfer.ready_event.has_value());
    auto consumer_stream = at::cuda::getCurrentCUDAStream();
    transfer.ready_event->block(consumer_stream);
    anet::transfer::RecordStreamOn(transfer.device_samples, consumer_stream);

    CHECK(transfer.device_samples.actions.is_cuda());
    CHECK(transfer.device_samples.obs.At(kVectorKey).is_cuda());
    CHECK(transfer.device_samples.indices.device().is_cpu());
    CHECK(transfer.device_samples.per_is_initial_priority.device().is_cpu());
    RequireFlatApprox(transfer.device_samples.per_is_initial_priority, { 1.0f, 0.0f });
    const auto expected_indices = std::vector<int64_t>{ 7, 8 };
    CHECK(TensorToInt64Vector(transfer.device_samples.indices) == expected_indices);
    RequireFlatApprox(transfer.device_samples.target_returns, { 3.0f, 4.0f });

    const auto& retained_strided = transfer.retained_source.info.At("strided");
    CHECK(retained_strided.is_pinned());
    CHECK(retained_strided.sizes().vec() == strided_sizes);
    CHECK(retained_strided.strides().vec() == strided_strides);
    CHECK_FALSE(retained_strided.is_contiguous());

    const auto& device_strided = transfer.device_samples.info.At("strided");
    CHECK(device_strided.is_cuda());
    CHECK(device_strided.sizes().vec() == strided_sizes);
    CHECK(device_strided.strides().vec() == strided_strides);
    RequireFlatApprox(device_strided, { 0.0f, 4.0f, 8.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f, 10.0f, 3.0f, 7.0f, 11.0f });

    event_recycler.Retire(std::move(*transfer.ready_event), std::move(transfer.retained_source));
    event_recycler.Drain();
    CHECK(event_recycler.PendingCount() == 0);
    CHECK(event_recycler.FreeCount() == 1);
}

TEST_CASE("ExperienceSamples To CUDA keeps sampled indices on CPU", "[transfer][cuda]")
{
    if (!torch::cuda::is_available()) {
        SUCCEED("CUDA is not available.");
        return;
    }

    auto samples = MakeTransferSamples();
    auto device_samples = samples.To(torch::Device(torch::kCUDA));

    CHECK(device_samples.actions.is_cuda());
    CHECK(device_samples.obs.At(kVectorKey).is_cuda());
    CHECK(device_samples.target_returns.is_cuda());
    CHECK(device_samples.next_state.terminals.is_cuda());
    CHECK(device_samples.n_steps.is_cuda());
    CHECK(device_samples.is_weights.is_cuda());
    CHECK(device_samples.info.At("info").is_cuda());
    CHECK(device_samples.indices.device().is_cpu());
    const auto expected_indices = std::vector<int64_t>{ 7, 8 };
    CHECK(TensorToInt64Vector(device_samples.indices) == expected_indices);
}

TEST_CASE("HostReadback CUDA constructor copies device tensor to pinned CPU", "[transfer][cuda]")
{
    if (!torch::cuda::is_available()) {
        SUCCEED("CUDA is not available.");
        return;
    }

    anet::transfer::EventRecycler<torch::Tensor> event_recycler;
    auto copy_stream = at::cuda::getStreamFromPool(false);
    auto producer_stream = at::cuda::getCurrentCUDAStream();
    auto source = torch::tensor({ 4.0f, 5.0f }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    anet::transfer::HostReadback readback(source, copy_stream, producer_stream, event_recycler);
    REQUIRE(readback.source_ready_event.has_value());
    REQUIRE(readback.ready_event.has_value());

    readback.Wait();
    CHECK(readback.pinned_result.device().is_cpu());
    CHECK(readback.pinned_result.is_pinned());
    RequireFlatApprox(readback.pinned_result, { 4.0f, 5.0f });

    readback.RetireEvents(event_recycler);
    event_recycler.Drain();
    CHECK(event_recycler.PendingCount() == 0);
    CHECK(event_recycler.FreeCount() == 2);
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

    auto initial_ratio_before = buffer.rb->GetScalar(rl::ReplayBuffer::PER_INITIAL_MASS_RATIO);
    REQUIRE(initial_ratio_before.has_value());
    CHECK(*initial_ratio_before == Catch::Approx(1.0f).margin(1.0e-5));

    const int64_t env0_index = IndexOf(buffer, 0, 0);
    const int64_t env1_index = IndexOf(buffer, 1, 0);
    buffer.rb->UpdatePriorities({ env0_index, env1_index }, { 4.0f, 9.0f });

    const float env0_priority = std::sqrt(4.0f);
    const float env1_priority = std::sqrt(9.0f);

    auto total = buffer.rb->GetScalar(rl::ReplayBuffer::PER_TOTAL);
    REQUIRE(total.has_value());
    REQUIRE(*total == Catch::Approx(env0_priority + env1_priority + 2.0f).margin(1.0e-5));

    auto initial_ratio = buffer.rb->GetScalar(rl::ReplayBuffer::PER_INITIAL_MASS_RATIO);
    REQUIRE(initial_ratio.has_value());
    CHECK(*initial_ratio == Catch::Approx(2.0f / *total).margin(1.0e-5));

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

TEST_CASE("ReplayBuffer PER initial mass ratio remains accurate with many slots", "[replay_buffer][per][precision]")
{
    constexpr int64_t capacity = 2048;
    constexpr int64_t updated_count = capacity / 2;
    auto config = MakeConfig(capacity);
    config.per_alpha = 1.0f;
    config.per_initial_priority = 0.3f;
    auto buffer = MakeBuffer(config, 1);

    // 全 slot を初期優先度で埋め、半数だけ Learner 優先度へ置き換える。
    for (int64_t time_idx = 0; time_idx < capacity; ++time_idx) {
        PushTime(buffer, time_idx, {}, {}, time_idx == 0 ? BoolValues(1, true) : std::vector<bool>());
    }

    std::vector<int64_t> indices(static_cast<size_t>(updated_count));
    std::iota(indices.begin(), indices.end(), 0);
    buffer.rb->UpdatePriorities(indices, std::vector<float>(static_cast<size_t>(updated_count), 0.02f));

    auto initial_mass_ratio = buffer.rb->GetScalar(rl::ReplayBuffer::PER_INITIAL_MASS_RATIO);
    REQUIRE(initial_mass_ratio.has_value());
    REQUIRE(*initial_mass_ratio == Catch::Approx(0.9375f).margin(1.0e-6));
}

TEST_CASE("ReplayBuffer PER samples expose initial-priority flags", "[replay_buffer][per]")
{
    auto buffer = MakeBuffer(MakeConfig(20), 1, false, 777);

    PushTime(buffer, 0, {}, {}, BoolValues(1, true));
    PushTime(buffer, 1);

    REQUIRE(buffer.rb->Size() == 1);

    rl::ExperienceSamples first;
    buffer.rb->Sample(first, 1, 0.4f);
    RequireShape(first.per_is_initial_priority, { 1 });
    CHECK(first.per_is_initial_priority[0].item<bool>());

    auto sampled_indices = TensorToInt64Vector(first.indices);
    buffer.rb->UpdatePriorities(sampled_indices, { 2.0f });

    rl::ExperienceSamples second;
    buffer.rb->Sample(second, 1, 0.4f);
    RequireShape(second.per_is_initial_priority, { 1 });
    CHECK_FALSE(second.per_is_initial_priority[0].item<bool>());
}

TEST_CASE("ReplayBuffer PER normalizes IS weights from priority probabilities", "[replay_buffer][per][is_weight]")
{
    auto config = MakeConfig(20);
    config.per_alpha = 1.0f;
    auto buffer = MakeBuffer(config, 1, false, 777);

    PushTime(buffer, 0, {}, {}, BoolValues(1, true));
    PushTime(buffer, 1);
    PushTime(buffer, 2);

    REQUIRE(buffer.rb->Size() == 2);

    const int64_t low_priority_index = IndexOf(buffer, 0, 0);
    const int64_t high_priority_index = IndexOf(buffer, 0, 1);
    buffer.rb->UpdatePriorities({ low_priority_index, high_priority_index }, { 1.0f, 4.0f });

    bool checked_mixed_batch = false;
    for (int attempt = 0; attempt < 128 && !checked_mixed_batch; ++attempt) {
        rl::ExperienceSamples samples;
        buffer.rb->Sample(samples, 2, 0.5f);
        RequireShape(samples.indices, { 2 });
        RequireShape(samples.is_weights, { 2 });

        auto indices = TensorToInt64Vector(samples.indices);
        auto weights = TensorToFloatVector(samples.is_weights);

        const bool saw_low = std::find(indices.begin(), indices.end(), low_priority_index) != indices.end();
        const bool saw_high = std::find(indices.begin(), indices.end(), high_priority_index) != indices.end();
        if (!saw_low || !saw_high) continue;

        checked_mixed_batch = true;
        for (size_t i = 0; i < indices.size(); ++i) {
            CAPTURE(indices[i], weights[i]);
            if (indices[i] == low_priority_index) {
                REQUIRE(weights[i] == Catch::Approx(1.0f).margin(1.0e-5));
            } else if (indices[i] == high_priority_index) {
                REQUIRE(weights[i] == Catch::Approx(0.5f).margin(1.0e-5));
            } else {
                FAIL("PER sampled an index outside the two sampleable priorities.");
            }
        }
    }
    REQUIRE(checked_mixed_batch);
}

TEST_CASE("ReplayBuffer PER initializes new samples from max priority when configured", "[replay_buffer][per]")
{
    auto config = MakeConfig(20);
    config.per_alpha = 0.5f;
    config.per_initial_priority = -1.0f;
    auto buffer = MakeBuffer(config, 1);

    PushTime(buffer, 0, {}, {}, BoolValues(1, true));

    const int64_t first_index = IndexOf(buffer, 0, 0);
    buffer.rb->UpdatePriorities({ first_index }, { 9.0f });

    PushTime(buffer, 1);

    const int64_t second_index = IndexOf(buffer, 0, 1);
    auto first_priority = buffer.rb->GetTensor(rl::ReplayBuffer::PER_VALUES, first_index);
    auto second_priority = buffer.rb->GetTensor(rl::ReplayBuffer::PER_VALUES, second_index);
    REQUIRE(first_priority.has_value());
    REQUIRE(second_priority.has_value());
    RequireFlatApprox(*first_priority, { 3.0f });
    RequireFlatApprox(*second_priority, { 3.0f });
}

TEST_CASE("ReplayBuffer PER tracks last evicted sampleable slots that were never sampled", "[replay_buffer][per]")
{
    auto buffer = MakeBuffer(MakeConfig(4), 1);

    PushTime(buffer, 0, {}, {}, BoolValues(1, true));
    PushTime(buffer, 1);
    PushTime(buffer, 2);
    PushTime(buffer, 3);

    auto ratio_before_overwrite = buffer.rb->GetScalar(rl::ReplayBuffer::PER_LAST_EVICTED_NEVER_SAMPLED_RATIO);
    REQUIRE(ratio_before_overwrite.has_value());
    CHECK(std::isnan(*ratio_before_overwrite));

    SampleOnlyIndex(buffer, IndexOf(buffer, 0, 0));

    PushTime(buffer, 4);
    auto sampled_eviction_ratio = buffer.rb->GetScalar(rl::ReplayBuffer::PER_LAST_EVICTED_NEVER_SAMPLED_RATIO);
    REQUIRE(sampled_eviction_ratio.has_value());
    CHECK(*sampled_eviction_ratio == Catch::Approx(0.0f).margin(1.0e-5));

    PushTime(buffer, 5);
    auto never_sampled_eviction_ratio = buffer.rb->GetScalar(rl::ReplayBuffer::PER_LAST_EVICTED_NEVER_SAMPLED_RATIO);
    REQUIRE(never_sampled_eviction_ratio.has_value());
    CHECK(*never_sampled_eviction_ratio == Catch::Approx(1.0f).margin(1.0e-5));
}

TEST_CASE("ReplayBuffer PER visualization keys are unavailable for uniform sampling", "[replay_buffer][visualization][per]")
{
    auto buffer = MakeBuffer(MakeConfig(20, 1, 0.99f, 1, rl::ReplaySamplerType::UNIFORM), 1);

    PushTime(buffer, 0, {}, {}, BoolValues(1, true));
    PushTime(buffer, 1);

    REQUIRE(buffer.rb->Size() == 1);
    REQUIRE_FALSE(buffer.rb->GetScalar(rl::ReplayBuffer::PER_TOTAL).has_value());
    REQUIRE_FALSE(buffer.rb->GetScalar(rl::ReplayBuffer::PER_INITIAL_MASS_RATIO).has_value());
    REQUIRE_FALSE(buffer.rb->GetScalar(rl::ReplayBuffer::PER_LAST_EVICTED_NEVER_SAMPLED_RATIO).has_value());
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

TEST_CASE("PrefetchingReplayBuffer samples deterministically on CPU", "[replay_buffer][prefetch]")
{
    auto MakePrefetchingBuffer = [] {
        auto buffer = MakeBuffer(MakeConfig(32, 1, 0.99f, 1, rl::ReplaySamplerType::PRIORITIZED), 2, false, 777);
        for (int64_t t = 0; t < 10; ++t) {
            PushTime(buffer, t, {}, {}, t == 0 ? BoolValues(buffer.num_envs, true) : BoolValues(buffer.num_envs, false));
        }
        return std::make_shared<rl::PrefetchingReplayBuffer>(buffer.rb, torch::kCPU);
    };

    auto CollectIndices = [](const std::shared_ptr<rl::ReplayBuffer>& rb) {
        std::vector<int64_t> out;
        for (int i = 0; i < 4; ++i) {
            rl::ExperienceSamples samples;
            rb->Sample(samples, 4, 0.4f);
            RequireShape(samples.indices, { 4 });
            RequireShape(samples.actions, { 4 });
            CHECK(samples.actions.device().is_cpu());

            auto indices = TensorToInt64Vector(samples.indices);
            out.insert(out.end(), indices.begin(), indices.end());
            rb->UpdatePriorities(indices, std::vector<float>(indices.size(), 1.0f + static_cast<float>(i)));
        }
        return out;
    };

    auto lhs = CollectIndices(MakePrefetchingBuffer());
    auto rhs = CollectIndices(MakePrefetchingBuffer());

    CHECK(lhs == rhs);
}

TEST_CASE("PrefetchingReplayBuffer forwards ReplayBuffer accessors on CPU", "[replay_buffer][prefetch]")
{
    auto inner_state = std::make_shared<BlockingReplayBuffer>();
    rl::PrefetchingReplayBuffer rb(inner_state, torch::kCPU);

    rb.Push(rl::BatchExperience{});
    CHECK(inner_state->PushCount() == 1);
    CHECK(rb.Size() == inner_state->size_value);

    auto scalar = rb.GetScalar(rl::ReplayBuffer::PER_TOTAL);
    REQUIRE(scalar.has_value());
    CHECK(*scalar == Catch::Approx(inner_state->scalar_value));

    auto tensor = rb.GetTensor(rl::ReplayBuffer::PER_VALUES);
    REQUIRE(tensor.has_value());
    RequireFlatApprox(*tensor, { 2.0f });

    auto tensor_vector = rb.GetTensorVector(rl::ReplayBuffer::PER_VALUES);
    REQUIRE(tensor_vector.has_value());
    REQUIRE(tensor_vector->size() == 1);
    RequireFlatApprox((*tensor_vector)[0], { 2.0f });
}

TEST_CASE("PrefetchingReplayBuffer waits for in-flight sample before priority update", "[replay_buffer][prefetch][thread]")
{
    auto inner_state = std::make_shared<BlockingReplayBuffer>();
    rl::PrefetchingReplayBuffer rb(inner_state, torch::kCPU);

    rl::ExperienceSamples samples;
    rb.Sample(samples, 1, 0.0f);
    CHECK(TensorToInt64Vector(samples.indices) == std::vector<int64_t>{ 1 });

    inner_state->WaitForSecondSampleStarted();

    std::thread updater([&] {
        rb.UpdatePriorities({ 0 }, { 1.0f });
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(inner_state->UpdateCount() == 0);

    inner_state->ReleaseSecondSample();
    updater.join();

    CHECK(inner_state->UpdateCount() == 1);
    CHECK_FALSE(inner_state->UpdateWasCalledBeforeSecondSampleFinished());
}

TEST_CASE("PrefetchingReplayBuffer waits for cold sample before push", "[replay_buffer][prefetch][thread]")
{
    const bool block_first_sample = true;
    const bool block_second_sample = false;
    auto inner_state = std::make_shared<BlockingReplayBuffer>(block_first_sample, block_second_sample);
    rl::PrefetchingReplayBuffer rb(inner_state, torch::kCPU);

    rl::ExperienceSamples samples;
    std::thread sampler([&] {
        rb.Sample(samples, 1, 0.0f);
    });

    inner_state->WaitForFirstSampleStarted();

    std::thread pusher([&] {
        rb.Push(rl::BatchExperience{});
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(inner_state->PushCount() == 0);

    inner_state->ReleaseFirstSample();
    sampler.join();
    pusher.join();

    CHECK(TensorToInt64Vector(samples.indices) == std::vector<int64_t>{ 1 });
    REQUIRE(inner_state->WaitForPushCount(1, std::chrono::milliseconds(1000)));
    CHECK(inner_state->PushCount() == 1);
    CHECK_FALSE(inner_state->PushWasCalledBeforeFirstSampleFinished());
}

TEST_CASE("PrefetchingReplayBuffer delays armed push on worker FIFO", "[replay_buffer][prefetch][thread]")
{
    auto inner_state = std::make_shared<BlockingReplayBuffer>();
    rl::PrefetchingReplayBuffer rb(inner_state, torch::kCPU);

    rl::ExperienceSamples samples;
    rb.Sample(samples, 1, 0.0f);
    CHECK(TensorToInt64Vector(samples.indices) == std::vector<int64_t>{ 1 });

    inner_state->WaitForSecondSampleStarted();

    std::atomic<bool> push_returned = false;
    std::thread pusher([&] {
        rb.Push(rl::BatchExperience{});
        push_returned = true;
    });

    const bool returned_before_release = WaitForFlag(push_returned, std::chrono::milliseconds(1000));
    if (!returned_before_release) {
        inner_state->ReleaseSecondSample();
    }
    pusher.join();
    REQUIRE(returned_before_release);
    CHECK(inner_state->PushCount() == 0);

    inner_state->ReleaseSecondSample();

    rl::ExperienceSamples second_samples;
    rb.Sample(second_samples, 1, 0.0f);
    CHECK(TensorToInt64Vector(second_samples.indices) == std::vector<int64_t>{ 2 });
    inner_state->WaitForSampleCount(3);

    CHECK(inner_state->PushCount() == 1);
    CHECK_FALSE(inner_state->PushWasCalledBeforeSecondSampleFinished());
    CHECK(inner_state->PushCountAtSampleStart(3) == 1);
}

TEST_CASE("PrefetchingReplayBuffer keeps shallow armed push alive after caller scope", "[replay_buffer][prefetch][thread]")
{
    auto inner_state = std::make_shared<BlockingReplayBuffer>();
    rl::PrefetchingReplayBuffer rb(inner_state, torch::kCPU);

    rl::ExperienceSamples samples;
    rb.Sample(samples, 1, 0.0f);
    CHECK(TensorToInt64Vector(samples.indices) == std::vector<int64_t>{ 1 });

    inner_state->WaitForSecondSampleStarted();

    std::atomic<bool> push_returned = false;
    {
        auto batch = MakeBatch(
            { 1.0f },
            { 2.0f },
            { 3.0f },
            { false },
            { false },
            { false });

        std::thread pusher([&] {
            rb.Push(batch);
            push_returned = true;
        });

        const bool returned_before_release = WaitForFlag(push_returned, std::chrono::milliseconds(1000));
        if (!returned_before_release) {
            inner_state->ReleaseSecondSample();
        }
        pusher.join();
        REQUIRE(returned_before_release);
    }

    CHECK(inner_state->PushCount() == 0);

    inner_state->ReleaseSecondSample();
    rb.UpdatePriorities({}, {});
    inner_state->WaitForPushCount(1);

    RequireFlatApprox(inner_state->LastPushedReward(), { 3.0f });
    RequireFlatApprox(inner_state->LastPushedStateObs(), { 1.0f });
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

TEST_CASE("ReplayBuffer unroll samples aligned future transition fields", "[replay_buffer][unroll][n_step]")
{
    constexpr int64_t num_envs = 1;
    constexpr int n_step = 2;
    constexpr int unroll_steps = 3;
    constexpr float gamma = 0.5f;

    auto config = MakeConfig(40, n_step, gamma);
    config.muzero.unroll_steps = unroll_steps;
    auto buffer = MakeBuffer(config, num_envs);

    for (int64_t t = 0; t <= 6; ++t) {
        PushTimeWithActionInfo(
            buffer,
            t,
            anet::TensorDict(kPolicyInfoKey, FloatColumn(PolicyInfoValues(num_envs, t))),
            t == 0 ? BoolValues(num_envs, true) : BoolValues(num_envs, false));
    }

    REQUIRE(buffer.rb->Size() == 3);

    auto samples = SampleOnlyIndex(buffer, IndexOf(buffer, 0, 1));
    RequireSampleIndex(samples, IndexOf(buffer, 0, 1));

    RequireShape(samples.actions, { 1, unroll_steps });
    RequireShape(samples.target_returns, { 1, unroll_steps });
    RequireShape(samples.next_state.terminals, { 1, unroll_steps });
    RequireShape(samples.n_steps, { 1 });
    RequireShape(samples.info.At(kPolicyInfoKey), { 1, unroll_steps, 1 });

    RequireFlatApprox(samples.actions, { 0.0f, 0.0f, 0.0f });
    RequireFlatApprox(samples.target_returns, {
        DiscountedReturn(0, 1, n_step, gamma),
        DiscountedReturn(0, 2, n_step, gamma),
        DiscountedReturn(0, 3, n_step, gamma)
    });
    RequireFlatApprox(samples.next_state.terminals, { 0.0f, 0.0f, 0.0f });
    RequireFlatApprox(samples.info.At(kPolicyInfoKey), {
        PolicyInfoValue(0, 1),
        PolicyInfoValue(0, 2),
        PolicyInfoValue(0, 3)
    });
    REQUIRE(samples.n_steps[0].item<int64_t>() == n_step);
    RequireFlatApprox(samples.obs.At(kVectorKey)[0], { StateValue(0, 1) });
    RequireFlatApprox(samples.next_state.next_obs.At(kVectorKey)[0], { StateValue(0, 3) });
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

TEST_CASE("ReplayBuffer n-step returns stop at episode_start without done", "[replay_buffer][n_step][episode_start]")
{
    constexpr int64_t num_envs = 1;
    constexpr int n_step = 3;
    constexpr float gamma = 0.5f;

    auto buffer = MakeBuffer(MakeConfig(40, n_step, gamma), num_envs);

    PushTime(buffer, 0, {}, {}, BoolValues(num_envs, true));
    PushTime(buffer, 1);
    PushTime(buffer, 2);
    PushTime(buffer, 3, {}, {}, BoolValues(num_envs, true));
    PushTime(buffer, 4);
    PushTime(buffer, 5);

    auto before_reset = SampleOnlyIndex(buffer, IndexOf(buffer, 0, 1));
    RequireSampleMeta(
        before_reset,
        IndexOf(buffer, 0, 1),
        RewardValue(0, 1) + gamma * RewardValue(0, 2),
        true,
        2);
}

TEST_CASE("ReplayBuffer frame stacking keeps pre-terminal frames with n-step done flush", "[replay_buffer][frame_stack][n_step][done]")
{
    constexpr int64_t num_envs = 1;
    constexpr int n_step = 3;
    constexpr int stack_count = 4;
    constexpr float gamma = 0.5f;

    auto buffer = MakeBuffer(MakeConfig(20, n_step, gamma, stack_count), num_envs);

    PushTime(buffer, 0, {}, {}, BoolValues(num_envs, true));
    PushTime(buffer, 1);
    PushTime(buffer, 2);
    PushTime(buffer, 3);
    PushTime(buffer, 4);
    PushTime(buffer, 5, BoolValues(num_envs, true));
    PushTime(buffer, 6, {}, {}, BoolValues(num_envs, true));
    PushTime(buffer, 7);
    PushTime(buffer, 8);

    REQUIRE(buffer.rb->Size() == 6);

    auto bootstrap = SampleOnlyIndex(buffer, IndexOf(buffer, 0, 2));
    RequireSampleMeta(bootstrap, IndexOf(buffer, 0, 2), DiscountedReturn(0, 2, n_step, gamma), false, n_step);
    RequireFlatApprox(bootstrap.next_state.next_obs.At(kVectorKey)[0], {
        StateValue(0, 2),
        StateValue(0, 3),
        StateValue(0, 4),
        StateValue(0, 5)
    });

    auto terminal = SampleOnlyIndex(buffer, IndexOf(buffer, 0, 5));
    RequireSampleMeta(terminal, IndexOf(buffer, 0, 5), RewardValue(0, 5), true, 1);
    RequireFlatApprox(terminal.obs.At(kVectorKey)[0], {
        StateValue(0, 2),
        StateValue(0, 3),
        StateValue(0, 4),
        StateValue(0, 5)
    });
}

TEST_CASE("ReplayBuffer frame stacking keeps truncated n-step boundary aligned", "[replay_buffer][frame_stack][n_step][truncated]")
{
    constexpr int64_t num_envs = 1;
    constexpr int n_step = 3;
    constexpr int stack_count = 4;
    constexpr float gamma = 0.5f;

    auto buffer = MakeBuffer(MakeConfig(20, n_step, gamma, stack_count), num_envs);

    PushTime(buffer, 0, {}, {}, BoolValues(num_envs, true));
    PushTime(buffer, 1);
    PushTime(buffer, 2);
    PushTime(buffer, 3);
    PushTime(buffer, 4);
    PushTime(
        buffer,
        5,
        BoolValues(num_envs, false),
        BoolValues(num_envs, true),
        BoolValues(num_envs, false),
        TerminalStateValues(num_envs, 5));
    PushTime(buffer, 6, {}, {}, BoolValues(num_envs, true));
    PushTime(buffer, 7);

    REQUIRE(buffer.rb->Size() == 6);

    auto bootstrap = SampleOnlyIndex(buffer, IndexOf(buffer, 0, 2));
    RequireSampleMeta(bootstrap, IndexOf(buffer, 0, 2), DiscountedReturn(0, 2, n_step, gamma), false, n_step);
    RequireFlatApprox(bootstrap.next_state.next_obs.At(kVectorKey)[0], {
        StateValue(0, 2),
        StateValue(0, 3),
        StateValue(0, 4),
        StateValue(0, 5)
    });

    auto truncated = SampleOnlyIndex(buffer, IndexOf(buffer, 0, 5));
    RequireSampleMeta(truncated, IndexOf(buffer, 0, 5), RewardValue(0, 5), false, 1);
    RequireFlatApprox(truncated.obs.At(kVectorKey)[0], {
        StateValue(0, 2),
        StateValue(0, 3),
        StateValue(0, 4),
        StateValue(0, 5)
    });
    RequireFlatApprox(truncated.next_state.next_obs.At(kVectorKey)[0], {
        StateValue(0, 3),
        StateValue(0, 4),
        StateValue(0, 5),
        TerminalStateValue(0, 5)
    });
}

TEST_CASE("ReplayBuffer frame stacking and n-step next_obs survive ring wrap", "[replay_buffer][frame_stack][n_step][wrap]")
{
    constexpr int64_t num_envs = 1;
    constexpr int64_t capacity = 8;
    constexpr int n_step = 2;
    constexpr int stack_count = 3;
    constexpr float gamma = 0.5f;

    auto buffer = MakeBuffer(MakeConfig(capacity, n_step, gamma, stack_count), num_envs);

    for (int64_t t = 0; t <= 10; ++t) {
        PushTime(buffer, t, {}, {}, t == 0 ? BoolValues(num_envs, true) : BoolValues(num_envs, false));
    }

    auto samples = SampleOnlyIndex(buffer, IndexOf(buffer, 0, 0));
    RequireSampleMeta(
        samples,
        IndexOf(buffer, 0, 0),
        RewardValue(0, 8) + gamma * RewardValue(0, 9),
        false,
        n_step);
    RequireFlatApprox(samples.obs.At(kVectorKey)[0], {
        StateValue(0, 6),
        StateValue(0, 7),
        StateValue(0, 8)
    });
    RequireFlatApprox(samples.next_state.next_obs.At(kVectorKey)[0], {
        StateValue(0, 8),
        StateValue(0, 9),
        StateValue(0, 10)
    });
}

TEST_CASE("ReplayBuffer excludes wrapped samples whose frame stack would read overwritten frames", "[replay_buffer][frame_stack][wrap][sampleability]")
{
    constexpr int64_t num_envs = 1;
    constexpr int64_t capacity = 8;
    constexpr int n_step = 2;
    constexpr int stack_count = 3;
    constexpr float gamma = 0.5f;

    auto buffer = MakeBuffer(MakeConfig(capacity, n_step, gamma, stack_count), num_envs);

    for (int64_t t = 0; t <= 10; ++t) {
        PushTime(buffer, t, {}, {}, t == 0 ? BoolValues(num_envs, true) : BoolValues(num_envs, false));
    }

    REQUIRE(buffer.rb->Size() == 4);

    auto state = RequireSingleTensorVector(buffer.rb->GetTensorVector(rl::ReplayBuffer::STATE_OBS));
    RequireShape(state, { 4, 1 });
    RequireFlatApproxUnordered(state, {
        StateValue(0, 5),
        StateValue(0, 6),
        StateValue(0, 7),
        StateValue(0, 8)
    });

    auto oldest_safe = SampleOnlyIndex(buffer, IndexOf(buffer, 0, 5));
    RequireSampleMeta(
        oldest_safe,
        IndexOf(buffer, 0, 5),
        RewardValue(0, 5) + gamma * RewardValue(0, 6),
        false,
        n_step);
    RequireFlatApprox(oldest_safe.obs.At(kVectorKey)[0], {
        StateValue(0, 3),
        StateValue(0, 4),
        StateValue(0, 5)
    });
    RequireFlatApprox(oldest_safe.next_state.next_obs.At(kVectorKey)[0], {
        StateValue(0, 5),
        StateValue(0, 6),
        StateValue(0, 7)
    });
}

TEST_CASE("ReplayBuffer keeps wrapped oldest samples sampleable when frame stack is disabled", "[replay_buffer][wrap][sampleability]")
{
    constexpr int64_t num_envs = 1;
    constexpr int64_t capacity = 8;
    constexpr int n_step = 2;
    constexpr float gamma = 0.5f;

    auto buffer = MakeBuffer(MakeConfig(capacity, n_step, gamma), num_envs);

    for (int64_t t = 0; t <= 10; ++t) {
        PushTime(buffer, t, {}, {}, t == 0 ? BoolValues(num_envs, true) : BoolValues(num_envs, false));
    }

    REQUIRE(buffer.rb->Size() == 6);

    auto oldest = SampleOnlyIndex(buffer, IndexOf(buffer, 0, 3));
    RequireSampleMeta(
        oldest,
        IndexOf(buffer, 0, 3),
        RewardValue(0, 3) + gamma * RewardValue(0, 4),
        false,
        n_step);
    RequireFlatApprox(oldest.obs.At(kVectorKey)[0], { StateValue(0, 3) });
    RequireFlatApprox(oldest.next_state.next_obs.At(kVectorKey)[0], { StateValue(0, 5) });
}

TEST_CASE("ReplayBuffer PER samples only safe wrapped frame-stack indices", "[replay_buffer][per][frame_stack][wrap][sampleability]")
{
    constexpr int64_t num_envs = 1;
    constexpr int64_t capacity = 8;
    constexpr int n_step = 2;
    constexpr int stack_count = 3;
    constexpr float gamma = 0.5f;

    auto buffer = MakeBuffer(MakeConfig(capacity, n_step, gamma, stack_count), num_envs, false, 777);

    for (int64_t t = 0; t <= 10; ++t) {
        PushTime(buffer, t, {}, {}, t == 0 ? BoolValues(num_envs, true) : BoolValues(num_envs, false));
    }

    std::vector<int64_t> all_indices(static_cast<size_t>(capacity));
    std::iota(all_indices.begin(), all_indices.end(), 0);
    buffer.rb->UpdatePriorities(all_indices, std::vector<float>(all_indices.size(), 0.0f));
    buffer.rb->UpdatePriorities({ IndexOf(buffer, 0, 3), IndexOf(buffer, 0, 4) }, { 10.0f, 10.0f });

    for (int attempt = 0; attempt < 16; ++attempt) {
        rl::ExperienceSamples samples;
        buffer.rb->Sample(samples, 1, 0.4f);
        RequireShape(samples.indices, { 1 });
        RequireIndicesIn(samples.indices, {
            IndexOf(buffer, 0, 5),
            IndexOf(buffer, 0, 6),
            IndexOf(buffer, 0, 7),
            IndexOf(buffer, 0, 0)
        });
    }
}

TEST_CASE("ReplayBuffer wrapped sampleability honors both frame stack and unroll horizons", "[replay_buffer][frame_stack][wrap][unroll][sampleability]")
{
    constexpr int64_t num_envs = 1;
    constexpr int64_t capacity = 10;
    constexpr int n_step = 2;
    constexpr int stack_count = 3;
    constexpr int unroll_steps = 2;
    constexpr float gamma = 0.5f;

    auto config = MakeConfig(capacity, n_step, gamma, stack_count);
    config.muzero.unroll_steps = unroll_steps;
    auto buffer = MakeBuffer(config, num_envs);

    for (int64_t t = 0; t <= 12; ++t) {
        PushTime(buffer, t, {}, {}, t == 0 ? BoolValues(num_envs, true) : BoolValues(num_envs, false));
    }

    REQUIRE(buffer.rb->Size() == 5);

    auto state = RequireSingleTensorVector(buffer.rb->GetTensorVector(rl::ReplayBuffer::STATE_OBS));
    RequireShape(state, { 5, 1 });
    RequireFlatApproxUnordered(state, {
        StateValue(0, 5),
        StateValue(0, 6),
        StateValue(0, 7),
        StateValue(0, 8),
        StateValue(0, 9)
    });

    auto oldest_safe = SampleOnlyIndex(buffer, IndexOf(buffer, 0, 5));
    RequireSampleIndex(oldest_safe, IndexOf(buffer, 0, 5));
    RequireShape(oldest_safe.actions, { 1, unroll_steps });
    RequireShape(oldest_safe.target_returns, { 1, unroll_steps });
    RequireShape(oldest_safe.next_state.terminals, { 1, unroll_steps });
    RequireFlatApprox(oldest_safe.target_returns, {
        RewardValue(0, 5) + gamma * RewardValue(0, 6),
        RewardValue(0, 6) + gamma * RewardValue(0, 7)
    });
    RequireFlatApprox(oldest_safe.next_state.terminals, { 0.0f, 0.0f });
    RequireFlatApprox(oldest_safe.obs.At(kVectorKey)[0], {
        StateValue(0, 3),
        StateValue(0, 4),
        StateValue(0, 5)
    });
    RequireFlatApprox(oldest_safe.next_state.next_obs.At(kVectorKey)[0], {
        StateValue(0, 5),
        StateValue(0, 6),
        StateValue(0, 7)
    });
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

TEST_CASE("ReplayBuffer frame-stacked truncated next_obs keeps the truncation frame", "[replay_buffer][frame_stack][truncated]")
{
    constexpr int64_t num_envs = 1;
    constexpr int stack_count = 4;

    auto buffer = MakeBuffer(MakeConfig(20, 1, 0.99f, stack_count), num_envs);

    PushTime(buffer, 0, {}, {}, BoolValues(num_envs, true));
    PushTime(
        buffer,
        1,
        BoolValues(num_envs, false),
        BoolValues(num_envs, true),
        BoolValues(num_envs, false),
        TerminalStateValues(num_envs, 1));
    PushTime(buffer, 2, {}, {}, BoolValues(num_envs, true));

    REQUIRE(buffer.rb->Size() == 2);

    auto samples = SampleOnlyIndex(buffer, IndexOf(buffer, 0, 1));
    RequireSampleMeta(samples, IndexOf(buffer, 0, 1), RewardValue(0, 1), false, 1);
    RequireFlatApprox(samples.obs.At(kVectorKey)[0], {
        StateValue(0, 0),
        StateValue(0, 0),
        StateValue(0, 0),
        StateValue(0, 1)
    });
    RequireFlatApprox(samples.next_state.next_obs.At(kVectorKey)[0], {
        StateValue(0, 0),
        StateValue(0, 0),
        StateValue(0, 1),
        TerminalStateValue(0, 1)
    });
}

TEST_CASE("ReplayBuffer frame stacking starts a new stack at episode_start without done", "[replay_buffer][frame_stack][episode_start]")
{
    constexpr int64_t num_envs = 1;
    constexpr int stack_count = 3;

    auto buffer = MakeBuffer(MakeConfig(20, 1, 0.99f, stack_count), num_envs);

    PushTime(buffer, 0, {}, {}, BoolValues(num_envs, true));
    PushTime(buffer, 1);
    PushTime(buffer, 2);
    PushTime(buffer, 3, {}, {}, BoolValues(num_envs, true));
    PushTime(buffer, 4);

    REQUIRE(buffer.rb->Size() == 4);

    auto samples = SampleOnlyIndex(buffer, IndexOf(buffer, 0, 3));
    RequireSampleMeta(samples, IndexOf(buffer, 0, 3), RewardValue(0, 3), false, 1);
    RequireFlatApprox(samples.obs.At(kVectorKey)[0], {
        StateValue(0, 3),
        StateValue(0, 3),
        StateValue(0, 3)
    });
    RequireFlatApprox(samples.next_state.next_obs.At(kVectorKey)[0], {
        StateValue(0, 3),
        StateValue(0, 3),
        StateValue(0, 4)
    });
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
