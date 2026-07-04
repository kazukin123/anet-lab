#include "catch.hpp"

#include "anet/default_dqn_agent.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/rainbow_agent.hpp"
#include "anet/test_util.hpp"
#include "anet/trainer.hpp"
#include "dqn_based_agent.hpp"
#include "nn_impl.hpp"
#include "nn_heads.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
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

    std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key) override
    {
        if (key != "forward" && key != "forward.q") return std::nullopt;

        return [this](const anet::TensorDict& feature_dict) {
            torch::NoGradGuard no_grad;
            return Forward(feature_dict);
        };
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
    explicit TestNetworkModel(int64_t num_quantiles = 1)
        : dqn::NetworkModel(
            dqn::NetworkModelConfig{},
            MakeLinearNetwork(),
            MakeLinearNetwork(),
            1,
            num_quantiles)
    {
        GetOnlineNetwork()->CopyTo(*GetTargetNetwork());
        GetOnlineNetwork()->eval();
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
    using dqn::Learner::Optimize;
    using dqn::Learner::ApplyPerPriorityUpdate;
    using dqn::Learner::PreparePerPriorityUpdate;
    using dqn::Learner::TransformH;
    using dqn::Learner::TransformHInv;

    void UseSgd(float lr)
    {
        optimizer_ = std::make_unique<torch::optim::SGD>(
            model_.GetOnlineParameters(),
            torch::optim::SGDOptions(lr));
    }

    void UseReplayBuffer(std::shared_ptr<rl::ReplayBuffer> replay_buffer)
    {
        replay_buffer_ = std::move(replay_buffer);
    }

    std::shared_ptr<const anet::rl::BatchUpdateResult> UpdateFromSamples(
        const anet::rl::ExperienceSamples& samples) override
    {
        return nullptr;
    }
};

class RecordingReplayBuffer final : public rl::ReplayBuffer {
public:
    void Push(const rl::BatchExperience&) override
    {
        ++push_count;
    }

    void Sample(rl::ExperienceSamples& out_samples, int64_t minibatch_size, float) const override
    {
        ++sample_count;

        out_samples.obs = anet::TensorDict{
            { kVectorKey, torch::zeros({ minibatch_size, 2 }, torch::TensorOptions().dtype(torch::kFloat32)) },
        };
        out_samples.actions = torch::zeros({ minibatch_size }, torch::TensorOptions().dtype(torch::kInt64));
        out_samples.target_returns = torch::zeros({ minibatch_size }, torch::TensorOptions().dtype(torch::kFloat32));
        out_samples.next_state.next_obs = anet::TensorDict{
            { kVectorKey, torch::zeros({ minibatch_size, 2 }, torch::TensorOptions().dtype(torch::kFloat32)) },
        };
        out_samples.next_state.terminals = torch::zeros({ minibatch_size }, torch::TensorOptions().dtype(torch::kBool));
        out_samples.n_steps = torch::ones({ minibatch_size }, torch::TensorOptions().dtype(torch::kInt64));
        out_samples.indices = torch::arange(minibatch_size, torch::TensorOptions().dtype(torch::kInt64));
        out_samples.is_weights = torch::ones({ minibatch_size }, torch::TensorOptions().dtype(torch::kFloat32));
        out_samples.per_is_initial_priority = torch::zeros({ minibatch_size }, torch::TensorOptions().dtype(torch::kBool));
    }

    int64_t Size() const override
    {
        ++size_count;
        if (size_values.empty()) return 0;
        if (size_index >= size_values.size()) return size_values.back();
        return size_values[size_index++];
    }

    void UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities) override
    {
        last_indices = indices;
        last_priorities = priorities;
        ++update_count;
    }

    std::optional<float> GetScalar(const std::string&, int64_t = -1) const override
    {
        return std::nullopt;
    }

    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override
    {
        return std::nullopt;
    }

    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t = -1) const override
    {
        return std::nullopt;
    }

    void SetSizeValues(std::vector<int64_t> values)
    {
        size_values = std::move(values);
        size_index = 0;
        size_count = 0;
    }

    std::vector<int64_t> last_indices;
    std::vector<float> last_priorities;
    int push_count = 0;
    mutable int sample_count = 0;
    mutable int size_count = 0;
    int update_count = 0;

private:
    std::vector<int64_t> size_values;
    mutable size_t size_index = 0;
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

anet::nn::NetworkConfig MakeAgentForwardNetworkConfig()
{
    anet::nn::NetworkConfig config;
    config.output_keys[anet::nn::kKey_DefaultOutput] = kVectorKey;
    return config;
}

dqn::DefaultDQNAgentConfig MakeDeviceForwardDefaultDqnConfig()
{
    dqn::DefaultDQNAgentConfig config;
    config.use_qr = false;
    config.use_dueling_net = false;
    config.stucker.use_stacker = false;
    config.obs_norm.pass_through = true;
    config.learner.replay_capacity = 16;
    config.learner.replay_batch_size = 2;
    config.learner.use_fused_optimizer = false;
    return config;
}

dqn::RainbowAgentConfig MakeDeviceForwardRainbowConfig()
{
    dqn::RainbowAgentConfig config;
    config.use_qr = false;
    config.use_dueling_net = false;
    config.learner.replay_capacity = 16;
    config.learner.replay_batch_size = 2;
    config.learner.use_fused_optimizer = false;
    return config;
}

class DeviceOnlyAgent final : public rl::AgentBase {
public:
    explicit DeviceOnlyAgent(torch::Device device)
        : rl::AgentBase(device, rl::BatchEnvSpec{ 1, 1 }, MakeLearnerEnvSpec(), 123)
    {
    }

    std::shared_ptr<rl::Actor> CreateActor(
        const rl::BatchEnvSpec&,
        rl::RunMode,
        bool,
        std::optional<torch::Device> = std::nullopt) const override
    {
        return nullptr;
    }

    std::shared_ptr<rl::Learner> CreateLearner() override
    {
        return nullptr;
    }

    std::optional<float> GetScalar(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t = -1) const override { return std::nullopt; }
};

enum class DeterminismJitterPhase {
    BeforeUpdateFromBatch = 0,
    BeforeUpdateFromSamples,
    BeforePerUpdate,
    AfterPerUpdate,
    EnvStep,
    Count,
};

constexpr size_t kDeterminismJitterPhaseCount = static_cast<size_t>(DeterminismJitterPhase::Count);

class DeterminismJitterSchedule {
public:
    DeterminismJitterSchedule(uint64_t seed, int max_sleep_us, size_t entries_per_phase, size_t sleep_stride)
        : sleep_stride_(std::max<size_t>(1, sleep_stride))
    {
        const int max_delay = std::max(0, max_sleep_us);
        for (size_t phase = 0; phase < kDeterminismJitterPhaseCount; ++phase) {
            counters_[phase].store(0, std::memory_order_relaxed);
            delays_[phase].reserve(entries_per_phase);
            std::mt19937_64 rng(seed ^ (0x9e3779b97f4a7c15ull + phase * 0xbf58476d1ce4e5b9ull));
            std::uniform_int_distribution<int> dist(0, max_delay);
            for (size_t i = 0; i < entries_per_phase; ++i) {
                delays_[phase].push_back(dist(rng));
            }
        }
    }

    void Sleep(DeterminismJitterPhase phase) const
    {
        const auto phase_idx = static_cast<size_t>(phase);
        const auto& delays = delays_[phase_idx];
        if (delays.empty()) return;

        const size_t counter = counters_[phase_idx].fetch_add(1, std::memory_order_relaxed);
        const int delay_us = delays[counter % delays.size()];
        if ((counter % sleep_stride_) != 0) return;
        if (delay_us <= 0) return;

        std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
    }

private:
    std::array<std::vector<int>, kDeterminismJitterPhaseCount> delays_;
    mutable std::array<std::atomic<size_t>, kDeterminismJitterPhaseCount> counters_;
    size_t sleep_stride_;
};

struct DeterminismTraceEntry {
    std::vector<int64_t> indices;
    std::vector<int64_t> n_steps;
    std::vector<float> target_returns;
    std::vector<float> is_weights;
    std::vector<float> priorities;

    bool operator==(const DeterminismTraceEntry&) const = default;
};

class NoopMetricsBackend final : public anet::IBackend {
public:
    void Open(const std::filesystem::path&, const std::string&) override {}
    void WriteJsonl(const anet::json&) override {}
    void Flush() override {}
};

class ScopedNoopMetricsLogger final {
public:
    ScopedNoopMetricsLogger()
    {
        anet::MetricsLogger::Reset();
        anet::MetricsLoggerConfig logger_config;
        logger_config.run_name_tmpl = "dqn_agent_device_test";
        anet::MetricsLogger::Init(std::make_unique<NoopMetricsBackend>(), logger_config, "out/test-tmp");
    }

    ~ScopedNoopMetricsLogger()
    {
        anet::MetricsLogger::Reset();
    }
};

std::vector<int64_t> TensorToInt64Vector(const torch::Tensor& tensor)
{
    if (!tensor.defined()) return {};
    auto cpu = tensor.detach().to(torch::kCPU).to(torch::kInt64).contiguous();
    const auto* ptr = cpu.data_ptr<int64_t>();
    return std::vector<int64_t>(ptr, ptr + cpu.numel());
}

std::vector<float> TensorToFloatVector(const torch::Tensor& tensor)
{
    if (!tensor.defined()) return {};
    auto cpu = tensor.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    const auto* ptr = cpu.data_ptr<float>();
    return std::vector<float>(ptr, ptr + cpu.numel());
}

torch::Tensor DeterminismBoolTensor(const std::vector<bool>& values)
{
    auto tensor = torch::empty({ static_cast<int64_t>(values.size()) }, torch::TensorOptions().dtype(torch::kBool));
    auto acc = tensor.accessor<bool, 1>();
    for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
        acc[i] = values[static_cast<size_t>(i)];
    }
    return tensor;
}

std::vector<bool> DeterminismBoolValues(int64_t num_envs, bool value)
{
    return std::vector<bool>(static_cast<size_t>(num_envs), value);
}

anet::TensorDict MakeDeterminismObs(int64_t step, int64_t num_envs)
{
    auto obs = torch::empty({ num_envs, 2 }, torch::TensorOptions().dtype(torch::kFloat32));
    auto acc = obs.accessor<float, 2>();
    for (int64_t env = 0; env < num_envs; ++env) {
        acc[env][0] = static_cast<float>(step);
        acc[env][1] = static_cast<float>(env);
    }
    return anet::TensorDict{ { kVectorKey, obs } };
}

std::vector<bool> MakeDeterminismDoneFlags(int64_t step, int64_t num_envs)
{
    std::vector<bool> flags(static_cast<size_t>(num_envs), false);
    for (int64_t env = 0; env < num_envs; ++env) {
        flags[static_cast<size_t>(env)] = ((step + env * 3 + 5) % 17) == 0;
    }
    return flags;
}

std::vector<bool> MakeDeterminismTruncatedFlags(
    int64_t step,
    int64_t num_envs,
    const std::vector<bool>& done)
{
    std::vector<bool> flags(static_cast<size_t>(num_envs), false);
    for (int64_t env = 0; env < num_envs; ++env) {
        flags[static_cast<size_t>(env)] = !done[static_cast<size_t>(env)]
            && ((step + env * 5 + 7) % 23) == 0;
    }
    return flags;
}

std::vector<bool> MakeDeterminismEpisodeStartFlags(int64_t step, int64_t num_envs)
{
    if (step <= 0) return DeterminismBoolValues(num_envs, true);

    auto prev_done = MakeDeterminismDoneFlags(step - 1, num_envs);
    auto prev_truncated = MakeDeterminismTruncatedFlags(step - 1, num_envs, prev_done);

    std::vector<bool> flags(static_cast<size_t>(num_envs), false);
    for (int64_t env = 0; env < num_envs; ++env) {
        flags[static_cast<size_t>(env)] =
            prev_done[static_cast<size_t>(env)] || prev_truncated[static_cast<size_t>(env)];
    }
    return flags;
}

rl::BatchState MakeDeterminismState(
    int64_t step,
    int64_t num_envs,
    const std::vector<bool>& done,
    const std::vector<bool>& truncated,
    const std::vector<bool>& episode_start)
{
    return rl::BatchState{
        MakeDeterminismObs(step, num_envs),
        DeterminismBoolTensor(done),
        DeterminismBoolTensor(truncated),
        DeterminismBoolTensor(episode_start),
    };
}

torch::Tensor MakeDeterminismRewards(int64_t step, int64_t num_envs)
{
    auto rewards = torch::empty({ num_envs }, torch::TensorOptions().dtype(torch::kFloat32));
    auto acc = rewards.accessor<float, 1>();
    for (int64_t env = 0; env < num_envs; ++env) {
        const int64_t centered = ((step + env * 2) % 13) - 6;
        acc[env] = static_cast<float>(centered) * 0.25f + static_cast<float>(env) * 0.05f;
    }
    return rewards;
}

rl::BatchExperience MakeDeterminismExperience(int64_t step, int64_t num_envs)
{
    const auto state_done = DeterminismBoolValues(num_envs, false);
    const auto state_truncated = DeterminismBoolValues(num_envs, false);
    const auto state_start = MakeDeterminismEpisodeStartFlags(step, num_envs);
    auto next_done = MakeDeterminismDoneFlags(step, num_envs);
    auto next_truncated = MakeDeterminismTruncatedFlags(step, num_envs, next_done);

    auto action = torch::zeros({ num_envs }, torch::TensorOptions().dtype(torch::kInt64));
    return rl::BatchExperience(
        MakeDeterminismState(step, num_envs, state_done, state_truncated, state_start),
        std::make_shared<rl::BatchActionInfo>(action),
        MakeDeterminismRewards(step, num_envs),
        MakeDeterminismState(
            step + 1,
            num_envs,
            next_done,
            next_truncated,
            DeterminismBoolValues(num_envs, false)));
}

dqn::LearnerConfig MakeDeterminismLearnerConfig()
{
    dqn::LearnerConfig config;
    config.use_rb_prefetch = true;
    config.use_per = true;
    config.use_n_step = true;
    config.n_step = 3;
    config.replay_batch_size = 8;
    config.replay_capacity = 64;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = -1.0f;
    config.per_beta_start = 0.4f;
    config.per_beta_end = 1.0f;
    config.per_beta_step = 128;
    config.use_per_prio_clip = true;
    config.per_prio_clip_value = 10.0f;
    return config;
}

class CapturingLearner final : public dqn::Learner {
public:
    CapturingLearner(
        const dqn::LearnerConfig& config,
        dqn::NetworkModel& model,
        dqn::RuntimeVars& vars,
        const rl::BatchEnvSpec& batch_env_spec,
        const rl::EnvSpec& env_spec,
        anet::seed_t replay_seed,
        std::shared_ptr<DeterminismJitterSchedule> jitter)
        : dqn::Learner(
            config,
            model,
            vars,
            nullptr,
            batch_env_spec,
            env_spec,
            torch::Device(torch::kCPU),
            replay_seed,
            std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}),
            std::nullopt,
            replay_seed + 1)
        , jitter_(std::move(jitter))
    {
        SetupReplayBuffer(batch_env_spec, env_spec, replay_seed);
    }

    rl::BatchUpdateResultList UpdateFromBatch(
        const rl::StepCounts& step,
        const rl::BatchExperience& experiences) override
    {
        if (jitter_) jitter_->Sleep(DeterminismJitterPhase::BeforeUpdateFromBatch);
        return dqn::Learner::UpdateFromBatch(step, experiences);
    }

    std::vector<DeterminismTraceEntry> Trace() const
    {
        std::lock_guard<std::mutex> lock(trace_mutex_);
        return trace_;
    }

protected:
    std::shared_ptr<const anet::rl::BatchUpdateResult> UpdateFromSamples(
        const anet::rl::ExperienceSamples& samples) override
    {
        torch::NoGradGuard no_grad;

        if (jitter_) jitter_->Sleep(DeterminismJitterPhase::BeforeUpdateFromSamples);

        auto td_error = samples.target_returns.detach().to(torch::kFloat32) * 0.125f
            + samples.n_steps.to(torch::kFloat32) * 0.05f
            + samples.indices.to(torch::kFloat32) * 0.001f
            + 0.01f;

        if (jitter_) jitter_->Sleep(DeterminismJitterPhase::BeforePerUpdate);
        auto per_info = UpdatePerPriorities(samples, td_error);
        if (jitter_) jitter_->Sleep(DeterminismJitterPhase::AfterPerUpdate);

        DeterminismTraceEntry entry{
            .indices = TensorToInt64Vector(samples.indices),
            .n_steps = TensorToInt64Vector(samples.n_steps),
            .target_returns = TensorToFloatVector(samples.target_returns),
            .is_weights = TensorToFloatVector(samples.is_weights),
            .priorities = TensorToFloatVector(per_info.per_priorities),
        };
        {
            std::lock_guard<std::mutex> lock(trace_mutex_);
            trace_.push_back(std::move(entry));
        }

        return std::make_shared<dqn::BatchUpdateResult>();
    }

private:
    std::shared_ptr<DeterminismJitterSchedule> jitter_;
    mutable std::mutex trace_mutex_;
    std::vector<DeterminismTraceEntry> trace_;
};

std::vector<DeterminismTraceEntry> RunLearnerDeterminismTrial(
    anet::seed_t replay_seed,
    uint64_t jitter_seed,
    int64_t num_steps,
    int max_sleep_us,
    size_t sleep_stride)
{
    constexpr int64_t kNumEnv = 4;
    const size_t jitter_entries = static_cast<size_t>(num_steps) * 8 + 1024;
    auto jitter = std::make_shared<DeterminismJitterSchedule>(
        jitter_seed,
        max_sleep_us,
        jitter_entries,
        sleep_stride);

    auto config = MakeDeterminismLearnerConfig();
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    auto env_spec = MakeLearnerEnvSpec();
    rl::BatchEnvSpec batch_env_spec{ kNumEnv, 1 };
    CapturingLearner learner(config, model, vars, batch_env_spec, env_spec, replay_seed, jitter);

    for (int64_t step = 0; step < num_steps; ++step) {
        rl::StepCounts counts;
        counts.train_step = step;
        counts.exp_step = step * kNumEnv;
        counts.learn_step = vars.learn_step;
        learner.UpdateFromBatch(counts, MakeDeterminismExperience(step, kNumEnv));
    }

    return learner.Trace();
}

bool DeterminismTracesEqual(
    const std::vector<DeterminismTraceEntry>& lhs,
    const std::vector<DeterminismTraceEntry>& rhs)
{
    return lhs == rhs;
}

bool CurrentCatchFilterMentionsStress()
{
    const auto* config = Catch::getCurrentContext().getConfig();
    if (config == nullptr) return false;

    std::ostringstream oss;
    oss << config->testSpec();
    return oss.str().find("stress") != std::string::npos;
}

void RequireLearnerDeterminismPairs(int trial_pairs, int64_t num_steps, int max_sleep_us, size_t sleep_stride)
{
    for (int trial = 0; trial < trial_pairs; ++trial) {
        const auto replay_seed = static_cast<anet::seed_t>(10000 + trial * 37);
        const auto jitter_seed = static_cast<uint64_t>(20000 + trial * 101);
        INFO("trial=" << trial << " replay_seed=" << replay_seed << " jitter_seed=" << jitter_seed);

        auto first = RunLearnerDeterminismTrial(replay_seed, jitter_seed, num_steps, max_sleep_us, sleep_stride);
        auto second = RunLearnerDeterminismTrial(replay_seed, jitter_seed, num_steps, max_sleep_us, sleep_stride);

        REQUIRE_FALSE(first.empty());
        REQUIRE(DeterminismTracesEqual(first, second));
    }
}

class DeterminismResetResult final : public rl::BatchResetResult {
public:
    explicit DeterminismResetResult(int64_t num_envs)
        : rl::BatchResetResult(MakeDeterminismState(
            0,
            num_envs,
            DeterminismBoolValues(num_envs, false),
            DeterminismBoolValues(num_envs, false),
            DeterminismBoolValues(num_envs, true)))
        , num_envs_(num_envs)
    {
    }

    std::vector<rl::AuxData> GetAuxDataList(int env_index = -1) const override
    {
        if (env_index >= 0) return { rl::AuxData{} };
        return std::vector<rl::AuxData>(static_cast<size_t>(num_envs_));
    }

private:
    int64_t num_envs_;
};

uint32_t CountDeterminismEpisodeEnds(const rl::BatchState& state)
{
    auto done = state.done.to(torch::kCPU).contiguous();
    auto truncated = state.truncated.to(torch::kCPU).contiguous();
    auto done_acc = done.accessor<bool, 1>();
    auto truncated_acc = truncated.accessor<bool, 1>();

    uint32_t count = 0;
    for (int64_t i = 0; i < done.size(0); ++i) {
        if (done_acc[i] || truncated_acc[i]) ++count;
    }
    return count;
}

class DeterminismStepResult final : public rl::BatchStepResult {
public:
    DeterminismStepResult(
        torch::Tensor reward,
        rl::BatchState next_state,
        rl::BatchState continue_state,
        int64_t num_envs,
        uint32_t episode_end_count)
        : rl::BatchStepResult(
            std::move(reward),
            std::move(next_state),
            std::move(continue_state),
            static_cast<uint32_t>(num_envs),
            episode_end_count)
        , num_envs_(num_envs)
    {
    }

    std::vector<rl::AuxData> GetAuxDataList(int env_index = -1) const override
    {
        if (env_index >= 0) return { rl::AuxData{} };
        return std::vector<rl::AuxData>(static_cast<size_t>(num_envs_));
    }

private:
    int64_t num_envs_;
};

class JitterBatchEnv final : public rl::BatchEnv {
public:
    JitterBatchEnv(int64_t num_envs, std::shared_ptr<DeterminismJitterSchedule> jitter)
        : batch_spec_{ static_cast<int>(num_envs), 1 }
        , jitter_(std::move(jitter))
    {
    }

    rl::EnvSpec GetSpec() const override { return MakeLearnerEnvSpec(); }
    rl::BatchEnvSpec GetBatchSpec() const override { return batch_spec_; }
    torch::Device GetDevice() const override { return torch::Device(torch::kCPU); }

    std::shared_ptr<const rl::BatchResetResult> Reset(rl::RunMode = rl::RunMode::Train) override
    {
        step_ = 0;
        return std::make_shared<DeterminismResetResult>(batch_spec_.num_envs);
    }

    std::shared_ptr<const rl::BatchStepResult> Step(
        std::shared_ptr<rl::BatchActionInfo>,
        rl::RunMode = rl::RunMode::Train) override
    {
        if (jitter_) jitter_->Sleep(DeterminismJitterPhase::EnvStep);

        const int64_t num_envs = batch_spec_.num_envs;
        auto next_done = MakeDeterminismDoneFlags(step_, num_envs);
        auto next_truncated = MakeDeterminismTruncatedFlags(step_, num_envs, next_done);
        auto next_state = MakeDeterminismState(
            step_ + 1,
            num_envs,
            next_done,
            next_truncated,
            DeterminismBoolValues(num_envs, false));
        auto continue_state = MakeDeterminismState(
            step_ + 1,
            num_envs,
            DeterminismBoolValues(num_envs, false),
            DeterminismBoolValues(num_envs, false),
            MakeDeterminismEpisodeStartFlags(step_ + 1, num_envs));
        auto reward = MakeDeterminismRewards(step_, num_envs);
        const uint32_t episode_end_count = CountDeterminismEpisodeEnds(next_state);
        ++step_;

        return std::make_shared<DeterminismStepResult>(
            std::move(reward),
            std::move(next_state),
            std::move(continue_state),
            num_envs,
            episode_end_count);
    }

    std::optional<float> GetScalar(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t = -1) const override { return std::nullopt; }

private:
    rl::BatchEnvSpec batch_spec_;
    std::shared_ptr<DeterminismJitterSchedule> jitter_;
    int64_t step_ = 0;
};

class TraceActor final : public rl::Actor {
public:
    explicit TraceActor(int64_t num_envs)
        : num_envs_(num_envs)
    {
    }

    std::shared_ptr<rl::BatchActionInfo> MakeAction(const rl::StepCounts&, const rl::BatchState&) const override
    {
        return std::make_shared<rl::BatchActionInfo>(
            torch::zeros({ num_envs_ }, torch::TensorOptions().dtype(torch::kInt64)));
    }

    void Sync() override {}

private:
    int64_t num_envs_;
};

class TraceAgent final : public rl::Agent {
public:
    TraceAgent(
        const dqn::LearnerConfig& config,
        const rl::BatchEnvSpec& batch_env_spec,
        const rl::EnvSpec& env_spec,
        anet::seed_t replay_seed,
        std::shared_ptr<DeterminismJitterSchedule> jitter)
        : model_(std::make_shared<TestNetworkModel>())
        , vars_(std::make_shared<dqn::RuntimeVars>())
        , learner_(std::make_shared<CapturingLearner>(
            config,
            *model_,
            *vars_,
            batch_env_spec,
            env_spec,
            replay_seed,
            std::move(jitter)))
    {
    }

    std::shared_ptr<rl::Actor> CreateActor(
        const rl::BatchEnvSpec& batch_env_spec,
        rl::RunMode,
        bool,
        std::optional<torch::Device> = std::nullopt) const override
    {
        return std::make_shared<TraceActor>(batch_env_spec.num_envs);
    }

    std::shared_ptr<rl::Learner> CreateLearner() override
    {
        return learner_;
    }

    torch::Device GetDevice() const override { return torch::Device(torch::kCPU); }

    std::vector<DeterminismTraceEntry> Trace() const
    {
        return learner_->Trace();
    }

    std::optional<float> GetScalar(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t = -1) const override { return std::nullopt; }

private:
    std::shared_ptr<TestNetworkModel> model_;
    std::shared_ptr<dqn::RuntimeVars> vars_;
    std::shared_ptr<CapturingLearner> learner_;
};

struct RunnerDeterminismTrace {
    rl::StepCounts counts;
    std::vector<DeterminismTraceEntry> learner_trace;
};

RunnerDeterminismTrace RunPipelineDeterminismTrial(
    anet::seed_t replay_seed,
    uint64_t jitter_seed,
    int64_t num_steps,
    int max_sleep_us,
    size_t sleep_stride)
{
    constexpr int64_t kNumEnv = 4;
    anet::MetricsLogger::Reset();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "dqn_prefetch_determinism_test";
    anet::MetricsLogger::Init(std::make_unique<NoopMetricsBackend>(), logger_config, "C:/tmp");

    const size_t jitter_entries = static_cast<size_t>(num_steps) * 8 + 1024;
    auto jitter = std::make_shared<DeterminismJitterSchedule>(
        jitter_seed,
        max_sleep_us,
        jitter_entries,
        sleep_stride);
    auto env = std::make_shared<JitterBatchEnv>(kNumEnv, jitter);
    auto agent = std::make_shared<TraceAgent>(
        MakeDeterminismLearnerConfig(),
        env->GetBatchSpec(),
        env->GetSpec(),
        replay_seed,
        jitter);
    auto notifier = std::make_shared<rl::Notifier>();
    std::shared_ptr<rl::TrainRunner> runner = std::make_shared<rl::PipelineTrainRunner>(env, agent, notifier);

    auto counts = runner->DoUpdateFrame(static_cast<int>(num_steps));
    runner->Shutdown();
    auto trace = agent->Trace();
    anet::MetricsLogger::Reset();

    return RunnerDeterminismTrace{
        .counts = counts,
        .learner_trace = std::move(trace),
    };
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

TEST_CASE("Learner stops polling replay buffer size after minibatch threshold is reached", "[dqn][replay_buffer][performance]")
{
    dqn::LearnerConfig config;
    config.replay_batch_size = 2;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = -1.0f;

    auto env_spec = MakeLearnerEnvSpec();
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    rl::BatchEnvSpec batch_env_spec{ 1, 1 };
    TestLearner learner(config, model, vars, batch_env_spec, env_spec);
    auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
    replay_buffer->SetSizeValues({ 0, config.replay_batch_size, 0 });
    learner.UseReplayBuffer(replay_buffer);

    auto make_counts = [&vars](rl::step_t step) {
        rl::StepCounts counts;
        counts.train_step = step;
        counts.exp_step = step;
        counts.learn_step = vars.learn_step;
        return counts;
    };

    auto cold_result = learner.UpdateFromBatch(make_counts(0), MakeDeterminismExperience(0, 1));
    REQUIRE(cold_result.empty());
    REQUIRE(replay_buffer->size_count == 1);
    REQUIRE(replay_buffer->sample_count == 0);

    auto first_update = learner.UpdateFromBatch(make_counts(1), MakeDeterminismExperience(1, 1));
    REQUIRE(first_update.size() == 1);
    REQUIRE(replay_buffer->size_count == 2);
    REQUIRE(replay_buffer->sample_count == 1);

    auto latched_update = learner.UpdateFromBatch(make_counts(2), MakeDeterminismExperience(2, 1));
    REQUIRE(latched_update.size() == 1);
    REQUIRE(replay_buffer->size_count == 2);
    REQUIRE(replay_buffer->sample_count == 2);
    REQUIRE(replay_buffer->push_count == 3);
}

TEST_CASE("PER priority prepare/apply updates replay buffer from CPU materialized priorities", "[dqn][per]")
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
    auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
    learner.UseReplayBuffer(replay_buffer);

    rl::ExperienceSamples samples;
    samples.indices = torch::tensor({ 3, 5 }, torch::TensorOptions().dtype(torch::kInt64));
    samples.is_weights = torch::tensor({ 0.25f, 0.75f });
    samples.per_is_initial_priority = DeterminismBoolTensor({ true, false });

    auto td_error = torch::tensor({ -0.2f, 2.0f });
    auto pending = learner.PreparePerPriorityUpdate(samples, td_error);

    REQUIRE(pending.enabled);
    const auto expected_indices = std::vector<int64_t>{ 3, 5 };
    CHECK(pending.indices == expected_indices);
    REQUIRE(pending.per_sample_initial_count.defined());
    CHECK(pending.per_sample_initial_count.item<float>() == Catch::Approx(1.0f).margin(1.0e-6f));

    auto result = learner.ApplyPerPriorityUpdate(std::move(pending));

    CHECK(replay_buffer->update_count == 1);
    CHECK(replay_buffer->last_indices == expected_indices);
    REQUIRE(replay_buffer->last_priorities.size() == 2);
    CHECK(replay_buffer->last_priorities[0] == Catch::Approx(0.3f).margin(1.0e-6f));
    CHECK(replay_buffer->last_priorities[1] == Catch::Approx(1.0f).margin(1.0e-6f));

    REQUIRE(result.per_priorities.defined());
    CHECK(result.per_priorities.device().is_cpu());
    CHECK(torch::allclose(result.per_priorities, torch::tensor({ 0.3f, 1.0f })));
    REQUIRE(result.per_clipped_count.defined());
    CHECK(result.per_clipped_count.device().is_cpu());
    CHECK(result.per_clipped_count.item<int64_t>() == 1);
    CHECK(result.per_minibatch_size == 2);
    REQUIRE(result.per_is_weights.defined());
    CHECK(torch::allclose(result.per_is_weights, samples.is_weights));
    REQUIRE(result.per_sample_initial_count.defined());
    CHECK(result.per_sample_initial_count.item<float>() == Catch::Approx(1.0f).margin(1.0e-6f));

    dqn::OptimizerStepResult opt_result;
    auto batch_result = learner.MakeBatchUpdateResult(
        torch::tensor(0.0f),
        td_error,
        opt_result,
        torch::zeros({ 2 }),
        torch::zeros({ 2 }),
        result);
    auto sample_initial_ratio = batch_result->GetScalar("per_sample_initial_ratio", -1);
    REQUIRE(sample_initial_ratio.has_value());
    CHECK(*sample_initial_ratio == Catch::Approx(0.5f).margin(1.0e-6f));
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

    auto weight_before = model.GetOnlineParameters()[0].detach().clone().cpu();
    auto obs = anet::TensorDict{ { kVectorKey, torch::tensor({ { 3.0f, 4.0f } }) } };
    auto loss = model.GetOnlineNetwork()->Forward(obs).At("q").sum();
    auto result = learner.Optimize(loss);

    CHECK_FALSE(result.grad_norm.has_value());
    REQUIRE(result.grad_norm_tensor.defined());
    CHECK(result.grad_norm_tensor.item<float>() == Catch::Approx(5.0f).margin(1.0e-5f));
    CHECK(result.grad_clip_tau == Catch::Approx(0.5f).margin(1.0e-6f));
    CHECK(result.grad_clip_ratio == Catch::Approx(0.0f).margin(1.0e-6f));

    auto weight_delta = model.GetOnlineParameters()[0].detach().cpu() - weight_before;
    CHECK(weight_delta[0][0].item<float>() == Catch::Approx(-0.03f).margin(1.0e-5f));
    CHECK(weight_delta[0][1].item<float>() == Catch::Approx(-0.04f).margin(1.0e-5f));
}

TEST_CASE("NetworkModel mode-specific forwards preserve training modes", "[dqn][network_model]")
{
    TestNetworkModel model;
    auto obs = anet::TensorDict{ { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }) } };

    REQUIRE_FALSE(model.GetOnlineNetwork()->is_training());
    REQUIRE_FALSE(model.GetTargetNetwork()->is_training());

    model.ForwardOnline(obs);
    CHECK_FALSE(model.GetOnlineNetwork()->is_training());
    CHECK_FALSE(model.GetTargetNetwork()->is_training());

    model.ForwardTarget(obs);
    CHECK_FALSE(model.GetOnlineNetwork()->is_training());
    CHECK_FALSE(model.GetTargetNetwork()->is_training());

    model.ForwardOnlineWithTrain(obs);
    CHECK_FALSE(model.GetOnlineNetwork()->is_training());
    CHECK_FALSE(model.GetTargetNetwork()->is_training());

    model.GetOnlineNetwork()->train();
    model.ForwardOnlineWithTrain(obs);
    CHECK(model.GetOnlineNetwork()->is_training());
    CHECK_FALSE(model.GetTargetNetwork()->is_training());
}

TEST_CASE("AgentBase exposes configured device", "[agent][device]")
{
    DeviceOnlyAgent agent{ torch::Device(torch::kCPU) };

    CHECK(agent.GetDevice().is_cpu());
}

TEST_CASE("DefaultDQNAgent TensorDictFunction accepts CPU input on CUDA agent", "[dqn][network_model][device]")
{
    if (!torch::cuda::is_available()) return;

    ScopedNoopMetricsLogger metrics_logger;
    const torch::Device device(torch::kCUDA, 0);
    auto env_spec = MakeLearnerEnvSpec();
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(
        MakeDeviceForwardDefaultDqnConfig(),
        MakeAgentForwardNetworkConfig(),
        rl::BatchEnvSpec{ 1, 1 },
        env_spec,
        device,
        123);

    auto forward = agent->GetTensorDictFunction("policy-net.forward");
    REQUIRE(forward.has_value());

    auto obs = anet::TensorDict{
        { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)) },
    };
    auto out = (*forward)(obs);

    REQUIRE(out.Get("q").has_value());
    CHECK(out.At("q").device().type() == torch::kCUDA);
}

TEST_CASE("RainbowAgent TensorDictFunction accepts CPU input on CUDA agent", "[dqn][network_model][device]")
{
    if (!torch::cuda::is_available()) return;

    ScopedNoopMetricsLogger metrics_logger;
    const torch::Device device(torch::kCUDA, 0);
    auto env_spec = MakeLearnerEnvSpec();
    auto agent = std::make_shared<dqn::RainbowAgent>(
        MakeDeviceForwardRainbowConfig(),
        MakeAgentForwardNetworkConfig(),
        rl::BatchEnvSpec{ 1, 1 },
        env_spec,
        device,
        123);

    auto forward = agent->GetTensorDictFunction("policy-net.forward");
    REQUIRE(forward.has_value());

    auto obs = anet::TensorDict{
        { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)) },
    };
    auto out = (*forward)(obs);

    REQUIRE(out.Get("q").has_value());
    CHECK(out.At("q").device().type() == torch::kCUDA);
}

TEST_CASE("NetworkModel routes TensorDictFunction by network side and function key", "[dqn][network_model]")
{
    TestNetworkModel model;
    {
        torch::NoGradGuard no_grad;
        model.GetOnlineNetwork()->parameters()[0].fill_(1.0f);
        model.GetTargetNetwork()->parameters()[0].fill_(2.0f);
    }
    auto obs = anet::TensorDict{ { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }) } };

    auto policy_forward = model.GetTensorDictFunction("policy-net.forward", torch::kCPU);
    auto policy_forward_q = model.GetTensorDictFunction("policy-net.forward.q", torch::kCPU);
    auto target_forward = model.GetTensorDictFunction("target-net.forward", torch::kCPU);
    REQUIRE(policy_forward.has_value());
    REQUIRE(policy_forward_q.has_value());
    REQUIRE(target_forward.has_value());

    CHECK(TensorToFloatVector((*policy_forward)(obs).At("q")) == std::vector<float>{ 3.0f });
    CHECK(TensorToFloatVector((*policy_forward_q)(obs).At("q")) == std::vector<float>{ 3.0f });
    CHECK(TensorToFloatVector((*target_forward)(obs).At("q")) == std::vector<float>{ 6.0f });

    CHECK_FALSE(model.GetTensorDictFunction("policy-net.forward.dist", torch::kCPU).has_value());
    CHECK_FALSE(model.GetTensorDictFunction("unknown-net.forward", torch::kCPU).has_value());
}

TEST_CASE("NetworkModel distributionality depends only on quantile count", "[dqn][network_model]")
{
    TestNetworkModel dqn_model(1);
    TestNetworkModel qr_model(8);

    CHECK_FALSE(dqn_model.IsDistributional());
    CHECK(qr_model.IsDistributional());
}

TEST_CASE("Actor sync leaves cloned network in eval mode", "[dqn][actor]")
{
    auto src_network = MakeLinearNetwork();
    auto clone_network = MakeLinearNetwork();
    auto mutex = std::make_shared<std::shared_mutex>();
    dqn::Actor actor(nullptr, nullptr, nullptr, mutex, clone_network, src_network);

    src_network->eval();
    clone_network->train();

    actor.Sync();

    CHECK_FALSE(src_network->is_training());
    CHECK_FALSE(clone_network->is_training());
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

TEST_CASE("DefaultDQNAgentConfig reads fused optimizer setting", "[dqn][config][optimizer]")
{
    dqn::DefaultDQNAgentConfig default_config(anet::ConfigData{});
    CHECK(default_config.learner.use_fused_optimizer);

    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.learner.use_fused_optimizer", "false");
    dqn::DefaultDQNAgentConfig config(config_data);
    CHECK_FALSE(config.learner.use_fused_optimizer);
}

TEST_CASE("DQN configs read sample prefetch setting", "[dqn][config][prefetch]")
{
    dqn::DefaultDQNAgentConfig default_config(anet::ConfigData{});
    CHECK_FALSE(default_config.learner.use_rb_prefetch);

    anet::ConfigData default_dqn_data;
    default_dqn_data.Set("DefaultDQNAgent.learner.use_rb_prefetch", "true");
    dqn::DefaultDQNAgentConfig default_dqn_config(default_dqn_data);
    CHECK(default_dqn_config.learner.use_rb_prefetch);

    anet::ConfigData rainbow_data;
    rainbow_data.Set("RainbowAgent.learner.use_rb_prefetch", "true");
    dqn::RainbowAgentConfig rainbow_config(rainbow_data);
    CHECK_FALSE(rainbow_config.learner.use_rb_prefetch);
}

TEST_CASE("Learner allows sample prefetch on CPU device", "[dqn][prefetch]")
{
    dqn::LearnerConfig config;
    config.use_rb_prefetch = true;
    config.replay_batch_size = 1;

    TestNetworkModel model;
    dqn::RuntimeVars vars;
    auto env_spec = MakeLearnerEnvSpec();
    rl::BatchEnvSpec batch_env_spec{ 1, 1 };

    CHECK_NOTHROW(TestLearner(config, model, vars, batch_env_spec, env_spec));
}

TEST_CASE("DQN learner prefetch path is deterministic under fixed jitter", "[dqn][prefetch][determinism]")
{
    constexpr int kTrialPairs = 48;
    constexpr int64_t kNumSteps = 256;
    constexpr int kMaxSleepUs = 300;
    constexpr size_t kSleepStride = 64;
    RequireLearnerDeterminismPairs(kTrialPairs, kNumSteps, kMaxSleepUs, kSleepStride);
}

TEST_CASE("PipelineTrainRunner prefetch path is deterministic under fixed jitter", "[dqn][prefetch][determinism][runner]")
{
    constexpr int kTrialPairs = 12;
    constexpr int64_t kNumSteps = 256;
    constexpr int kMaxSleepUs = 300;
    constexpr size_t kSleepStride = 64;

    for (int trial = 0; trial < kTrialPairs; ++trial) {
        const auto replay_seed = static_cast<anet::seed_t>(30000 + trial * 43);
        const auto jitter_seed = static_cast<uint64_t>(40000 + trial * 109);
        INFO("trial=" << trial << " replay_seed=" << replay_seed << " jitter_seed=" << jitter_seed);

        auto first = RunPipelineDeterminismTrial(replay_seed, jitter_seed, kNumSteps, kMaxSleepUs, kSleepStride);
        auto second = RunPipelineDeterminismTrial(replay_seed, jitter_seed, kNumSteps, kMaxSleepUs, kSleepStride);

        REQUIRE_FALSE(first.learner_trace.empty());
        REQUIRE(first.counts.train_step == second.counts.train_step);
        REQUIRE(first.counts.exp_step == second.counts.exp_step);
        REQUIRE(first.counts.learn_step == second.counts.learn_step);
        REQUIRE(first.learner_trace.size() == second.learner_trace.size());
        REQUIRE(DeterminismTracesEqual(first.learner_trace, second.learner_trace));
    }
}

TEST_CASE("DQN learner prefetch path is deterministic under extended jitter stress", "[.][dqn][prefetch][determinism][stress]")
{
    if (!CurrentCatchFilterMentionsStress()) {
        SUCCEED("Skipping extended stress unless [stress] is explicitly selected.");
        return;
    }

    constexpr int kTrialPairs = 200;
    constexpr int64_t kNumSteps = 512;
    constexpr int kMaxSleepUs = 1000;
    constexpr size_t kSleepStride = 64;
    RequireLearnerDeterminismPairs(kTrialPairs, kNumSteps, kMaxSleepUs, kSleepStride);
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

TEST_CASE("RainbowAgentConfig keeps fused optimizer disabled", "[dqn][config][optimizer]")
{
    anet::ConfigData config_data;
    config_data.Set("RainbowAgent.learner.use_fused_optimizer", "true");

    dqn::RainbowAgentConfig config(config_data);

    CHECK_FALSE(config.learner.use_fused_optimizer);
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
