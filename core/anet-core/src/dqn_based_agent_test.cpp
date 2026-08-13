#include "anet/catch_test.hpp"

#include "anet/default_dqn_agent.hpp"
#include "anet/env.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/rainbow_agent.hpp"
#include "anet/test_util.hpp"
#include "anet/trainer.hpp"
#include "dqn_based_agent.hpp"
#include "nn_impl.hpp"
#include "nn_heads.hpp"

#include <ATen/autocast_mode.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace rl = anet::rl;
namespace dqn = anet::rl::dqn;

struct AutocastProbeState {
    void Record(torch::DeviceType device_type)
    {
        ++forward_count;
        if (at::autocast::is_autocast_enabled(device_type)) {
            ++enabled_count;
        }
        last_device_type = device_type;
    }

    int forward_count = 0;
    int enabled_count = 0;
    torch::DeviceType last_device_type = torch::kCPU;
};

class AutocastProbeModule final : public anet::nn::NetworkModule {
public:
    explicit AutocastProbeModule(std::shared_ptr<AutocastProbeState> state)
        : state_(std::move(state))
    {
    }

    torch::Tensor Forward(torch::Tensor input) override
    {
        state_->Record(input.device().type());
        return input;
    }

private:
    std::shared_ptr<AutocastProbeState> state_;
};

struct QuantileLearnerBaseAccess : public dqn::QuantileLearnerBase {
    using dqn::QuantileLearnerBase::ComputeQuantileHuberLoss;
    using dqn::QuantileLearnerBase::ComputeIqnQuantileHuberLoss;
};

struct ActionPolicyAccess : public dqn::ActionPolicy {
    ActionPolicyAccess()
        : dqn::ActionPolicy(dqn::ActionPolicyConfig{})
    {
    }

    using dqn::ActionPolicy::CreateSpatialTensor;
    using dqn::ActionPolicy::CreateSpatialLaneTensor;

    std::shared_ptr<dqn::DQNActionInfo> SelectAction(const anet::TensorDict&, bool, std::shared_ptr<anet::nn::Network>,
        std::shared_ptr<anet::RandomGenerator>, const anet::TraceSink&) const override
    {
        return std::make_shared<dqn::DQNActionInfo>();
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

std::shared_ptr<anet::nn::Network> MakeAutocastProbeNetwork(
    const std::shared_ptr<AutocastProbeState>& probe_state,
    torch::Device device)
{
    anet::TensorSpec vector_spec;
    vector_spec.type = anet::SpaceType::Vector;
    vector_spec.shape = { 2 };
    vector_spec.dtype = torch::kFloat32;

    anet::TensorSpecMap input_specs;
    input_specs[kVectorKey] = vector_spec;

    anet::nn::NetworkConfig network_config;
    network_config.output_keys[kFeatureKey] = kFeatureKey;

    auto probe = std::make_shared<AutocastProbeModule>(probe_state);
    auto block = std::make_shared<anet::nn::NetworkBlock>("probe", probe);
    auto network_struct = std::make_shared<anet::nn::NetworkStruct>(
        std::vector<std::shared_ptr<anet::nn::NetworkBlock>>{ block });
    auto branch = std::make_shared<anet::nn::NetworkBranch>(
        kFeatureKey,
        std::vector<std::vector<std::string>>{ { kVectorKey } },
        1,
        network_struct);
    auto body = std::make_shared<anet::nn::NetworkBody>(
        std::vector<std::shared_ptr<anet::nn::NetworkBranch>>{ branch },
        input_specs,
        std::vector<std::string>{},
        network_config.output_keys);
    auto head = std::make_shared<TestLinearHead>(2, 1);

    auto network = std::make_shared<anet::nn::Network>(
        network_config,
        input_specs,
        nullptr,
        body,
        head);
    network->to(device);
    return network;
}

class TestNetworkModel final : public dqn::NetworkModel {
public:
    explicit TestNetworkModel(bool distributional = false)
        : dqn::NetworkModel(
            dqn::NetworkModelConfig{},
            MakeLinearNetwork(),
            MakeLinearNetwork(),
            1,
            distributional)
    {
        GetOnlineNetwork()->CopyTo(*GetTargetNetwork());
        GetOnlineNetwork()->eval();
        GetTargetNetwork()->eval();
    }
};

class AutocastProbeNetworkModel final : public dqn::NetworkModel {
public:
    AutocastProbeNetworkModel(
        const std::shared_ptr<AutocastProbeState>& probe_state,
        torch::Device device)
        : dqn::NetworkModel(
            dqn::NetworkModelConfig{},
            MakeAutocastProbeNetwork(probe_state, device),
            MakeAutocastProbeNetwork(probe_state, device),
            1,
            false)
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
        out_samples.replay_item_keys = torch::arange(minibatch_size, torch::TensorOptions().dtype(torch::kInt64));
        out_samples.is_weights = torch::ones({ minibatch_size }, torch::TensorOptions().dtype(torch::kFloat32));
        out_samples.per_priority_sources = torch::zeros({ minibatch_size }, torch::TensorOptions().dtype(torch::kInt8));
    }

    int64_t Size() const override
    {
        ++size_count;
        if (size_values.empty()) return 0;
        if (size_index >= size_values.size()) return size_values.back();
        return size_values[size_index++];
    }

    rl::ReplayPriorityUpdateResult UpdatePriorities(
        const std::vector<int64_t>& indices, const std::vector<float>& priorities) override
    {
        last_indices = indices;
        last_priorities = priorities;
        ++update_count;
        if (priority_update_result.has_value()) {
            return *priority_update_result;
        }
        rl::ReplayPriorityUpdateResult result;
        result.applied_count = static_cast<int64_t>(indices.size());
        return result;
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
    std::optional<rl::ReplayPriorityUpdateResult> priority_update_result;

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

rl::EnvSpec MakeIqnTracerEnvSpec()
{
    anet::TensorSpec vector_spec;
    vector_spec.type = anet::SpaceType::Vector;
    vector_spec.shape = { 4 };
    vector_spec.dtype = torch::kFloat32;

    rl::EnvSpec spec;
    spec.state_spec.obs_spec[kVectorKey] = vector_spec;
    spec.action_spec.is_discrete = true;
    spec.action_spec.value_labels = { "a0", "a1" };
    spec.reward_range = { -1.0f, 1.0f };
    return spec;
}

anet::ConfigData MakeIqnTracerConfigData()
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.quantile_mode", "iqn");
    config_data.Set("DefaultDQNAgent.use_dueling_net", false);
    config_data.Set("DefaultDQNAgent.stucker.use_stacker", false);
    config_data.Set("DefaultDQNAgent.obs_norm.pass_through", true);
    config_data.Set("DefaultDQNAgent.train_policy.eps_start", 0.0f);
    config_data.Set("DefaultDQNAgent.train_policy.eps_end", 0.0f);
    config_data.Set("DefaultDQNAgent.train_policy.tau_rule.num_taus", 3);
    config_data.Set("DefaultDQNAgent.train_policy.tau_rule.sample_mode", "fixed");
    config_data.Set("DefaultDQNAgent.learner.replay_capacity", 16);
    config_data.Set("DefaultDQNAgent.learner.replay_batch_size", 2);
    config_data.Set("DefaultDQNAgent.learner.use_fused_optimizer", false);

    config_data.Set("net.block.[CosEmb].type", "CosineEmbedding");
    config_data.Set("net.block.[CosEmb].cos.num_basis", 4);
    config_data.Set("net.block.[TauProj].type", "Linear");
    config_data.Set("net.block.[TauProj].linear.out_features", 4);
    config_data.Set("net.branch.[main].bind", kVectorKey);
    config_data.Set("net.branch.[tau_embedding].bind", anet::nn::kKey_Taus);
    config_data.Set("net.branch.[tau_embedding].structure", "CosEmb > TauProj");
    config_data.Set("net.branch.[fusion].bind", "main * tau_embedding");
    config_data.Set("net.body.output.[features]", "fusion");
    return config_data;
}

dqn::DefaultDQNAgentConfig MakeDeviceForwardDefaultDqnConfig()
{
    dqn::DefaultDQNAgentConfig config;
    config.quantile_mode = "none";
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
        const rl::EnvSpec&,
        rl::RunMode,
        std::optional<bool> = std::nullopt,
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
            + samples.replay_item_keys.to(torch::kFloat32) * 0.001f
            + 0.01f;

        if (jitter_) jitter_->Sleep(DeterminismJitterPhase::BeforePerUpdate);
        auto per_info = UpdatePerPriorities(samples, td_error);
        if (jitter_) jitter_->Sleep(DeterminismJitterPhase::AfterPerUpdate);

        DeterminismTraceEntry entry{
            .indices = TensorToInt64Vector(samples.replay_item_keys),
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

class JitterBatchEnv final : public rl::BatchEnvBase {
public:
    JitterBatchEnv(
        const std::string& name, int64_t num_envs, std::shared_ptr<DeterminismJitterSchedule> jitter)
        : rl::BatchEnvBase(name, static_cast<int>(num_envs))
        , batch_spec_{ static_cast<int>(num_envs), 1 }
        , jitter_(std::move(jitter))
    {
    }

    rl::EnvSpec GetSpec() const override { return MakeLearnerEnvSpec(); }
    rl::BatchEnvSpec GetBatchSpec() const override { return batch_spec_; }
    torch::Device GetDevice() const override { return torch::Device(torch::kCPU); }

    std::shared_ptr<const rl::BatchResetResult> Reset() override
    {
        step_ = 0;
        return std::make_shared<DeterminismResetResult>(batch_spec_.num_envs);
    }

    std::shared_ptr<const rl::BatchStepResult> Step(
        std::shared_ptr<rl::BatchActionInfo>) override
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
        const rl::EnvSpec&,
        rl::RunMode,
        std::optional<bool> = std::nullopt,
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
    auto env = std::make_shared<JitterBatchEnv>("determinism-jitter", kNumEnv, jitter);
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

anet::TensorDict MakeAutocastProbePolicyInput(torch::Device device)
{
    return anet::TensorDict{
        { kVectorKey, torch::tensor(
            {
                { 1.0f, 2.0f },
                { 3.0f, 4.0f },
            },
            torch::TensorOptions().dtype(torch::kFloat32).device(device)) },
    };
}

rl::ExperienceSamples MakeAutocastProbeSamples(torch::Device device)
{
    rl::ExperienceSamples samples;
    samples.obs = MakeAutocastProbePolicyInput(device);
    samples.next_state.next_obs = anet::TensorDict{
        { kVectorKey, torch::tensor(
            {
                { 0.5f, 1.0f },
                { 1.5f, 2.0f },
            },
            torch::TensorOptions().dtype(torch::kFloat32).device(device)) },
    };
    samples.actions = torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kInt64).device(device));
    samples.target_returns = torch::tensor({ 0.1f, 0.2f }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    samples.next_state.terminals = torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kBool).device(device));
    samples.n_steps = torch::ones({ 2 }, torch::TensorOptions().dtype(torch::kInt64).device(device));
    samples.replay_item_keys = torch::arange(2, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    return samples;
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
            { 5.0f, 5.0f },
            { 0.0f, 100.0f },
        },
    });

    return anet::TensorDict{
        { "q", q_values },
        { "q_dist", q_quantiles },
    };
}

struct TauEchoState {
    int forward_count = 0;
    torch::Tensor last_taus;
};

class TauEchoHead final : public anet::nn::NetworkHead {
public:
    explicit TauEchoHead(std::shared_ptr<TauEchoState> state)
        : state_(std::move(state))
    {
    }

    anet::TensorDict Forward(const anet::TensorDict& feature_dict) override
    {
        const auto taus = feature_dict.At(anet::nn::kKey_Taus);
        ++state_->forward_count;
        state_->last_taus = taus.detach().clone();

        auto q_dist = torch::stack({ taus, 1.0f - taus }, 1);
        return anet::TensorDict{
            { "q", q_dist.mean(2) },
            { "q_dist", q_dist },
        };
    }

    std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string&) override
    {
        return std::nullopt;
    }

private:
    std::shared_ptr<TauEchoState> state_;
};

std::shared_ptr<anet::nn::Network> MakeTauEchoNetwork(
    int64_t nominal_num_taus,
    const std::shared_ptr<TauEchoState>& state)
{
    anet::TensorSpec vector_spec;
    vector_spec.shape = { 1 };
    vector_spec.dtype = torch::kFloat32;

    anet::TensorSpec tau_spec;
    tau_spec.shape = { nominal_num_taus };
    tau_spec.dtype = torch::kFloat32;

    anet::TensorSpecMap input_specs{
        { kVectorKey, vector_spec },
        { anet::nn::kKey_Taus, tau_spec },
    };
    anet::nn::NetworkConfig network_config;
    network_config.output_keys[anet::nn::kKey_Taus] = anet::nn::kKey_Taus;
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
        std::make_shared<TauEchoHead>(state));
}

anet::TensorDict MakeTauEchoObservation()
{
    return anet::TensorDict{
        { kVectorKey, torch::zeros({ 2, 1 }) },
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

TEST_CASE("IQN quantile huber loss sums current samples and averages target samples", "[dqn][iqn][loss]")
{
    auto current_dist = torch::tensor({ { 1.0f, 3.0f } });
    auto target_dist = torch::tensor({ { 2.0f, 3.25f, 4.0f } });
    auto taus = torch::tensor({ 0.25f, 0.75f }).view({ 1, 2, 1 });

    auto result = QuantileLearnerBaseAccess::ComputeIqnQuantileHuberLoss(
        current_dist,
        target_dist,
        taus,
        0.5f);

    REQUIRE(ShapeOf(result.element_loss) == std::vector<int64_t>{ 1 });
    REQUIRE(result.element_loss.item<float>() == Catch::Approx(139.0f / 192.0f).margin(1.0e-6f));
    CHECK(result.pair_abs_td.item<float>() == Catch::Approx(17.0f / 12.0f).margin(1.0e-6f));
    CHECK(result.cancellation_ratio.item<float>() == Catch::Approx(4.0f / 17.0f).margin(1.0e-6f));
}

TEST_CASE("IQN and QR quantile huber losses agree when sample counts and kappa are one", "[dqn][iqn][loss]")
{
    auto current_dist = torch::tensor({ { 1.0f, 3.0f }, { -1.0f, 2.0f } });
    auto target_dist = torch::tensor({ { 2.0f, 4.0f }, { 0.5f, 3.0f } });
    auto taus = torch::tensor({ { 0.25f, 0.75f }, { 0.2f, 0.8f } }).unsqueeze(2);

    auto qr_loss = QuantileLearnerBaseAccess::ComputeQuantileHuberLoss(
        current_dist, target_dist, taus, 1.0f);
    auto iqn_loss = QuantileLearnerBaseAccess::ComputeIqnQuantileHuberLoss(
        current_dist, target_dist, taus, 1.0f);

    REQUIRE(torch::allclose(iqn_loss.element_loss, qr_loss, 1.0e-6, 1.0e-6));
}

TEST_CASE("IQN quantile huber loss is finite for one current sample", "[dqn][iqn][loss]")
{
    auto current_dist = torch::tensor({ { 1.0f } });
    auto target_dist = torch::tensor({ { 0.5f, 2.0f, 4.0f } });
    auto taus = torch::tensor({ 0.5f }).view({ 1, 1, 1 });

    auto loss = QuantileLearnerBaseAccess::ComputeIqnQuantileHuberLoss(
        current_dist, target_dist, taus, 0.5f);

    REQUIRE(ShapeOf(loss.element_loss) == std::vector<int64_t>{ 1 });
    REQUIRE(torch::isfinite(loss.element_loss).all().item<bool>());
}

TEST_CASE("IQN diagnostics preserve cancellation and N-normalized loss contracts", "[dqn][iqn][loss][metrics]")
{
    const auto target_dist = torch::tensor({ { -1.0f, 1.0f } });
    const auto one_current = torch::tensor({ { 0.0f } });
    const auto one_tau = torch::tensor({ 0.5f }).view({ 1, 1, 1 });
    const auto cancelling = QuantileLearnerBaseAccess::ComputeIqnQuantileHuberLoss(
        one_current, target_dist, one_tau, 1.0f);
    CHECK(cancelling.pair_abs_td.item<float>() == Catch::Approx(1.0f));
    CHECK(cancelling.cancellation_ratio.item<float>() == Catch::Approx(1.0f));

    const auto same_sign_target = torch::tensor({ { 1.0f, 1.0f } });
    const auto same_sign = QuantileLearnerBaseAccess::ComputeIqnQuantileHuberLoss(
        one_current, same_sign_target, one_tau, 1.0f);
    CHECK(same_sign.cancellation_ratio.item<float>() == Catch::Approx(0.0f).margin(1.0e-6f));

    // current quantileを同じ値で2本に増やすとsample lossは2倍になるが、/N後は一致する。
    const auto two_current = torch::tensor({ { 0.0f, 0.0f } });
    const auto two_taus = torch::tensor({ 0.5f, 0.5f }).view({ 1, 2, 1 });
    const auto two_samples = QuantileLearnerBaseAccess::ComputeIqnQuantileHuberLoss(
        two_current, same_sign_target, two_taus, 1.0f);
    CHECK(two_samples.element_loss.item<float>() / 2.0f
        == Catch::Approx(same_sign.element_loss.item<float>()).margin(1.0e-6f));
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

TEST_CASE("DQN TBO scalar transforms match tensor transforms", "[dqn][tbo][per][actor_initial][math]")
{
    const std::array<float, 9> values{
        -1000.0f, -10.0f, -1.0f, -0.0f, 0.0f, 1.0e-6f, 1.0f, 10.0f, 1000.0f,
    };

    for (float epsilon : { 1.0e-2f, 1.0e-3f }) {
        INFO("epsilon=" << epsilon);
        for (float value : values) {
            INFO("value=" << value);
            const float tensor_h = dqn::TransformH(torch::tensor(value), epsilon).item<float>();
            const float tensor_h_inv = dqn::TransformHInv(torch::tensor(value), epsilon).item<float>();
            CHECK(dqn::TransformH(value, epsilon)
                == Catch::Approx(tensor_h).epsilon(1.0e-5f).margin(1.0e-6f));
            CHECK(dqn::TransformHInv(value, epsilon)
                == Catch::Approx(tensor_h_inv).epsilon(1.0e-5f).margin(1.0e-6f));
        }
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
    dqn::LearnerConfig config;
    config.use_tbo = false;
    config.per_eps = 0.1f;
    config.use_per_prio_clip = false;
    auto estimator = dqn::CreateInitialPriorityEstimator(config);

    const std::array<float, 2> start_hint{ 4.0f, 5.0f };
    const std::array<float, 2> bootstrap_hint{ 6.0f, 2.0f };
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

TEST_CASE("DQN replay priority validation warns independently for each learner", "[dqn][config][per]")
{
    dqn::LearnerConfig config;
    config.use_per = true;
    config.per_initial_priority_mode = "fixed";
    config.per_eps = 0.0f;

    anet::test::LogCaptureGuard logs;
    const auto mode = dqn::ParseReplayInitialPriorityMode(config);
    logs.Flush();
    CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 0);

    dqn::ValidateReplayPriorityConfig(config, mode);
    dqn::ValidateReplayPriorityConfig(config, mode);
    logs.Flush();

    int matching_warnings = 0;
    for (const auto& record : logs.Records()) {
        if (record.level != wxLOG_Warning || record.message.find("learner.per_eps=0") == std::string::npos) continue;
        ++matching_warnings;
        CHECK(record.message.find("zero-TD-error transitions") != std::string::npos);
        CHECK(record.message.find("Set learner.per_eps") != std::string::npos);
    }
    CHECK(matching_warnings == 2);
}

TEST_CASE("DQN replay priority warnings identify keys values reasons and alternatives", "[dqn][config][per]")
{
    auto require_warning = [](const dqn::LearnerConfig& config, const std::vector<std::string>& fragments) {
        anet::test::LogCaptureGuard logs;
        const auto mode = dqn::ParseReplayInitialPriorityMode(config);
        dqn::ValidateReplayPriorityConfig(config, mode);
        logs.Flush();

        REQUIRE(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 1);
        const auto& message = logs.Records().front().message;
        for (const auto& fragment : fragments) {
            CAPTURE(message, fragment);
            CHECK(message.find(fragment) != std::string::npos);
        }
    };

    SECTION("actor priority with alpha zero") {
        dqn::LearnerConfig config;
        config.use_per = true;
        config.per_initial_priority_mode = "actor_approx";
        config.per_alpha = 0.0f;
        require_warning(config, {
            "learner.per_initial_priority_mode=actor_approx",
            "learner.per_alpha=0",
            "does not affect sampling",
            "Set learner.per_alpha",
        });
    }
    SECTION("zero clip upper bound") {
        dqn::LearnerConfig config;
        config.use_per_prio_clip = true;
        config.per_prio_clip_value = 0.0f;
        require_warning(config, {
            "learner.use_per_prio_clip=true",
            "learner.per_prio_clip_value=0",
            "clipped to zero",
            "disable learner.use_per_prio_clip",
        });
    }
    SECTION("clip upper bound no larger than epsilon") {
        dqn::LearnerConfig config;
        config.per_eps = 0.25f;
        config.use_per_prio_clip = true;
        config.per_prio_clip_value = 0.2f;
        require_warning(config, {
            "learner.per_prio_clip_value=0.2",
            "learner.per_eps=0.25",
            "collapse priority differences",
            "learner.per_prio_clip_value > learner.per_eps",
        });
    }
}

TEST_CASE("DQN replay priority validation preserves invalid configuration errors", "[dqn][config][per]")
{
    dqn::LearnerConfig config;

    config.per_initial_priority_mode = "unknown";
    CHECK_THROWS(dqn::ParseReplayInitialPriorityMode(config));

    config.per_initial_priority_mode = "fixed";
    const auto fixed_mode = dqn::ParseReplayInitialPriorityMode(config);
    for (const float invalid : {
        -1.0f,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity() }) {
        config.per_initial_priority = invalid;
        CHECK_THROWS(dqn::ValidateReplayPriorityConfig(config, fixed_mode));
    }

    config.per_initial_priority = 1.0f;
    for (const float invalid : {
        -1.0f,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity() }) {
        config.per_eps = invalid;
        CHECK_THROWS(dqn::ValidateReplayPriorityConfig(config, fixed_mode));
    }

    config.per_eps = 1.0e-6f;
    config.use_per = false;
    config.per_initial_priority_mode = "max";
    CHECK_THROWS(dqn::ValidateReplayPriorityConfig(
        config, dqn::ParseReplayInitialPriorityMode(config)));

    config.per_initial_priority_mode = "fixed";
    config.use_per_prio_clip = true;
    for (const float invalid : {
        -1.0f,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity() }) {
        config.per_prio_clip_value = invalid;
        CHECK_THROWS(dqn::ValidateReplayPriorityConfig(config, fixed_mode));
    }

    // clip無効時は未使用の値を検証しない既存契約を維持する。
    config.use_per_prio_clip = false;
    config.per_prio_clip_value = -1.0f;
    CHECK_NOTHROW(dqn::ValidateReplayPriorityConfig(config, fixed_mode));
}

TEST_CASE("DQN initial priority estimator distinguishes n-step bootstrap and true terminal", "[dqn][per][actor_initial][estimator]")
{
    dqn::LearnerConfig config;
    config.use_tbo = false;
    config.per_eps = 0.0f;
    config.use_per_prio_clip = false;
    auto estimator = dqn::CreateInitialPriorityEstimator(config);
    const std::array<float, 2> start_hint{ 4.0f, 99.0f };
    const std::array<float, 2> bootstrap_hint{ 88.0f, 3.0f };

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
    dqn::LearnerConfig config;
    config.use_tbo = true;
    config.tbo_epsilon = 1.0e-2f;
    config.per_eps = 0.05f;
    config.use_per_prio_clip = false;
    auto estimator = dqn::CreateInitialPriorityEstimator(config);

    const float actor_q_sa = torch::tensor({ 2.0f, 4.0f }).mean().item<float>();
    const float bootstrap_state_value = torch::tensor({ 1.0f, 5.0f }).mean().item<float>();
    const std::array<float, 2> start_hint{ actor_q_sa, 0.0f };
    const std::array<float, 2> bootstrap_hint{ 0.0f, bootstrap_state_value };
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
    const std::array<float, 2> start_hint{ 2.0f, 0.0f };

    dqn::LearnerConfig zero_config;
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

    dqn::LearnerConfig clip_config;
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
    dqn::LearnerConfig config;
    auto estimator = dqn::CreateInitialPriorityEstimator(config);
    const std::array<float, 2> finite_hint{ 1.0f, 2.0f };
    const std::array<float, 2> nan_hint{ std::numeric_limits<float>::quiet_NaN(), 2.0f };
    const std::array<float, 2> inf_hint{ 1.0f, std::numeric_limits<float>::infinity() };

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
        dqn::LearnerConfig config;
        config.alpha = 0.0f;
        config.use_fused_optimizer = false;
        config.use_grad_clip = false;
        config.use_td_clip = false;
        config.use_per = false;
        config.replay_capacity = 8;
        config.replay_batch_size = 1;
        config.gamma = 0.9f;
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
            target_policy,
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
            bootstrap_state_value = model.ForwardTarget(samples.next_state.next_obs).At("q").item<float>();
        }
        const auto base_result = learner.UpdateFromSamples(samples);
        const auto result = std::dynamic_pointer_cast<const dqn::BatchUpdateResult>(base_result);
        REQUIRE(result != nullptr);
        REQUIRE(result->td_error.numel() == 1);
        REQUIRE(result->q_sa.numel() == 1);

        const std::array<float, 2> start_hint{ result->q_sa.item<float>(), 0.0f };
        const std::array<float, 2> bootstrap_hint{ 0.0f, bootstrap_state_value };
        auto estimator = dqn::CreateInitialPriorityEstimator(config);
        const auto actor_priority = estimator->Estimate(rl::InitialPriorityEstimateInput{
            .start_hint = start_hint,
            .bootstrap_hint = bootstrap_hint,
            .target_return = samples.target_returns.item<float>(),
            .discount = static_cast<float>(std::pow(config.gamma, samples.n_steps.item<int64_t>())),
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

TEST_CASE("PER priority prepare/apply counts only priorities changed by clipping", "[dqn][per][clip]")
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
    replay_buffer->priority_update_result = rl::ReplayPriorityUpdateResult{
        .applied_count = 2,
        .stale_count = 1,
        .actor_learner_pair_count = 4,
        .actor_learner_positive_pair_ratio = 0.75f,
        .actor_learner_ratio_median = 1.25f,
        .actor_learner_log_ratio_mean = -0.5f,
        .actor_learner_spearman = 0.6f,
    };
    learner.UseReplayBuffer(replay_buffer);

    rl::ExperienceSamples samples;
    samples.replay_item_keys = torch::tensor({ 3, 4, 5 }, torch::TensorOptions().dtype(torch::kInt64));
    samples.is_weights = torch::tensor({ 0.2f, 0.3f, 0.5f });
    samples.per_priority_sources = torch::tensor(
        { static_cast<int8_t>(rl::ReplayPrioritySource::FIXED_INITIAL),
          static_cast<int8_t>(rl::ReplayPrioritySource::LEARNER_UPDATED),
          static_cast<int8_t>(rl::ReplayPrioritySource::LEARNER_UPDATED) },
        torch::TensorOptions().dtype(torch::kInt8));

    // clip前priorityが上限未満・等値・超過となる3境界を同時に検証する。
    auto td_error = torch::tensor({ -0.2f, 0.9f, 2.0f });
    auto upper_tail_std = torch::tensor({ 1.0f, 2.0f, 3.0f });
    auto pending = learner.PreparePerPriorityUpdate(
        samples, td_error, torch::Tensor(), upper_tail_std);

    REQUIRE(pending.enabled);
    const auto expected_indices = std::vector<int64_t>{ 3, 4, 5 };
    CHECK(pending.indices == expected_indices);
    REQUIRE(pending.per_sample_initial_count.defined());
    CHECK(pending.per_sample_initial_count.item<float>() == Catch::Approx(1.0f).margin(1.0e-6f));

    auto result = learner.ApplyPerPriorityUpdate(std::move(pending));

    CHECK(replay_buffer->update_count == 1);
    CHECK(replay_buffer->last_indices == expected_indices);
    REQUIRE(replay_buffer->last_priorities.size() == 3);
    CHECK(replay_buffer->last_priorities[0] == Catch::Approx(0.3f).margin(1.0e-6f));
    CHECK(replay_buffer->last_priorities[1] == Catch::Approx(1.0f).margin(1.0e-6f));
    CHECK(replay_buffer->last_priorities[2] == Catch::Approx(1.0f).margin(1.0e-6f));

    REQUIRE(result.per_priorities.defined());
    CHECK(result.per_priorities.device().is_cpu());
    CHECK(torch::allclose(result.per_priorities, torch::tensor({ 0.3f, 1.0f, 1.0f })));
    REQUIRE(result.per_clipped_count.defined());
    CHECK(result.per_clipped_count.device().is_cpu());
    CHECK(result.per_clipped_count.item<int64_t>() == 1);
    CHECK(result.per_minibatch_size == 3);
    REQUIRE(result.per_is_weights.defined());
    CHECK(torch::allclose(result.per_is_weights, samples.is_weights));
    REQUIRE(result.per_sample_initial_count.defined());
    CHECK(result.per_sample_initial_count.item<float>() == Catch::Approx(1.0f).margin(1.0e-6f));
    CHECK(result.per_update_result.applied_count == 2);
    CHECK(result.per_update_result.stale_count == 1);
    CHECK(result.per_update_result.actor_learner_pair_count == 4);
    CHECK(result.per_update_result.actor_learner_positive_pair_ratio == Catch::Approx(0.75f));
    CHECK(result.per_update_result.actor_learner_ratio_median == Catch::Approx(1.25f));
    CHECK(result.per_update_result.actor_learner_log_ratio_mean == Catch::Approx(-0.5f));
    CHECK(result.per_update_result.actor_learner_spearman == Catch::Approx(0.6f));

    dqn::OptimizerStepResult opt_result;
    auto batch_result = learner.MakeBatchUpdateResult(
        torch::tensor(0.0f),
        td_error,
        opt_result,
        torch::zeros({ 3 }),
        torch::zeros({ 3 }),
        result);
    auto sample_initial_ratio = batch_result->GetScalar("per_sample_initial_ratio", -1);
    REQUIRE(sample_initial_ratio.has_value());
    CHECK(*sample_initial_ratio == Catch::Approx(1.0f / 3.0f).margin(1.0e-6f));
    CHECK(batch_result->GetScalar("per_priority_update_stale_ratio", -1).value()
        == Catch::Approx(1.0f / 3.0f).margin(1.0e-6f));
    CHECK(batch_result->GetScalar("per_actor_learner_pair_count", -1).value() == Catch::Approx(4.0f));
    CHECK(batch_result->GetScalar("per_actor_learner_positive_pair_ratio", -1).value() == Catch::Approx(0.75f));
    CHECK(batch_result->GetScalar("per_actor_learner_ratio_median", -1).value() == Catch::Approx(1.25f));
    CHECK(batch_result->GetScalar("per_actor_learner_log_ratio_mean", -1).value() == Catch::Approx(-0.5f));
    CHECK(batch_result->GetScalar("per_actor_learner_spearman", -1).value() == Catch::Approx(0.6f));
    const auto upper_tail_spearman = batch_result->GetScalar("upper_tail_priority_spearman", -1);
    REQUIRE(upper_tail_spearman.has_value());
    CHECK(*upper_tail_spearman == Catch::Approx(std::sqrt(3.0f) / 2.0f).margin(1.0e-6f));
}

TEST_CASE("Upper-tail priority Spearman handles rank direction and undefined cases", "[dqn][per][quantile][metrics]")
{
    auto calculate = [](const torch::Tensor& td_error, const torch::Tensor& upper_tail_std, bool use_per) {
        dqn::LearnerConfig config;
        config.use_per = use_per;
        config.per_eps = 0.0f;

        auto env_spec = MakeLearnerEnvSpec();
        TestNetworkModel model;
        dqn::RuntimeVars vars;
        TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec);
        auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
        learner.UseReplayBuffer(replay_buffer);

        const int64_t batch_size = td_error.size(0);
        rl::ExperienceSamples samples;
        samples.replay_item_keys = torch::arange(
            batch_size, torch::TensorOptions().dtype(torch::kInt64));
        auto pending = learner.PreparePerPriorityUpdate(
            samples, td_error, torch::Tensor(), upper_tail_std);
        auto per_result = learner.ApplyPerPriorityUpdate(std::move(pending));
        auto batch_result = learner.MakeBatchUpdateResult(
            torch::tensor(0.0f),
            td_error,
            dqn::OptimizerStepResult{},
            torch::zeros({ batch_size }),
            torch::zeros({ batch_size }),
            per_result);
        return batch_result->GetScalar("upper_tail_priority_spearman", -1).value();
    };

    // raw priorityと同順・逆順なら、それぞれ順位相関は+1・-1になる。
    const auto increasing_priority = torch::tensor({ 1.0f, 2.0f, 3.0f });
    CHECK(calculate(increasing_priority, torch::tensor({ 1.0f, 2.0f, 3.0f }), true)
        == Catch::Approx(1.0f));
    CHECK(calculate(increasing_priority, torch::tensor({ 3.0f, 2.0f, 1.0f }), true)
        == Catch::Approx(-1.0f));

    // 順位分散がない場合、batch不足、PER無効は相関を定義しない。
    CHECK(std::isnan(calculate(torch::tensor({ 1.0f }), torch::tensor({ 2.0f }), true)));
    CHECK(std::isnan(calculate(increasing_priority, torch::ones({ 3 }), true)));
    CHECK(std::isnan(calculate(torch::tensor({ 1.0f, -1.0f, 1.0f }),
        torch::tensor({ 1.0f, 2.0f, 3.0f }), true)));
    CHECK(std::isnan(calculate(increasing_priority, torch::tensor({ 1.0f, 2.0f, 3.0f }), false)));
}

TEST_CASE("PER IQN diagnostics classify only explicit initial priority sources", "[dqn][per][iqn][metrics]")
{
    dqn::LearnerConfig config;
    config.use_per = true;
    config.per_eps = 0.0f;

    auto env_spec = MakeLearnerEnvSpec();
    TestNetworkModel model;
    dqn::RuntimeVars vars;
    TestLearner learner(config, model, vars, rl::BatchEnvSpec{ 1, 1 }, env_spec);
    auto replay_buffer = std::make_shared<RecordingReplayBuffer>();
    learner.UseReplayBuffer(replay_buffer);

    rl::ExperienceSamples samples;
    samples.replay_item_keys = torch::arange(5, torch::TensorOptions().dtype(torch::kInt64));
    samples.is_weights = torch::ones({ 5 });
    samples.per_priority_sources = torch::tensor(
        { static_cast<int8_t>(rl::ReplayPrioritySource::FIXED_INITIAL),
          static_cast<int8_t>(rl::ReplayPrioritySource::MAX_INITIAL),
          static_cast<int8_t>(rl::ReplayPrioritySource::ACTOR_INITIAL),
          static_cast<int8_t>(rl::ReplayPrioritySource::NONE),
          static_cast<int8_t>(rl::ReplayPrioritySource::LEARNER_UPDATED) },
        torch::TensorOptions().dtype(torch::kInt8));
    const auto diagnostics = torch::tensor({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f });

    auto pending = learner.PreparePerPriorityUpdate(samples, torch::zeros({ 5 }), diagnostics);
    CHECK(pending.per_sample_initial_count.item<float>() == Catch::Approx(3.0f));
    CHECK(pending.per_sample_fixed_initial_count.item<float>() == Catch::Approx(1.0f));
    CHECK(pending.per_sample_max_initial_count.item<float>() == Catch::Approx(1.0f));
    CHECK(pending.per_sample_actor_initial_count.item<float>() == Catch::Approx(1.0f));

    const auto result = learner.ApplyPerPriorityUpdate(std::move(pending));
    CHECK(replay_buffer->update_count == 1);
    REQUIRE(result.iqn_diagnostics.defined());
    CHECK(torch::equal(result.iqn_diagnostics, diagnostics));
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

TEST_CASE("DQN learner BF16 autocast follows learner device", "[dqn][learner][amp][bf16]")
{
    std::vector<torch::Device> devices{ torch::Device(torch::kCPU) };
    if (torch::cuda::is_available()) {
        devices.emplace_back(torch::Device(torch::kCUDA, 0));
    }

    for (const auto& device : devices) {
        INFO("device=" << device.str());

        auto probe_state = std::make_shared<AutocastProbeState>();
        AutocastProbeNetworkModel model(probe_state, device);
        dqn::RuntimeVars vars;
        auto env_spec = MakeLearnerEnvSpec();
        rl::BatchEnvSpec batch_env_spec{ 2, 1 };

        dqn::LearnerConfig config;
        config.use_amp = true;
        config.use_amp_bf16 = true;
        config.use_fused_optimizer = false;
        config.use_grad_clip = false;
        config.use_per = false;
        config.replay_batch_size = 2;
        config.replay_capacity = 8;

        dqn::ActionPolicyConfig target_policy_config;
        target_policy_config.use_amp = true;
        target_policy_config.use_amp_bf16 = true;
        auto target_policy = std::make_shared<dqn::EpsilonGreedyActionPolicy>(
            target_policy_config,
            false,
            0,
            device);

        const auto device_type = device.type();
        const bool original_enabled = at::autocast::is_autocast_enabled(device_type);
        dqn::TDLearner learner(
            config,
            model,
            vars,
            nullptr,
            batch_env_spec,
            env_spec,
            device,
            123,
            target_policy,
            std::nullopt,
            456);

        auto result = learner.UpdateFromSamples(MakeAutocastProbeSamples(device));

        REQUIRE(result != nullptr);
        REQUIRE(probe_state->forward_count > 0);
        CHECK(probe_state->enabled_count == probe_state->forward_count);
        CHECK(probe_state->last_device_type == device_type);
        CHECK(at::autocast::is_autocast_enabled(device_type) == original_enabled);
    }
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

TEST_CASE("DefaultDQNAgent IQN acts through the public ConfigData path", "[dqn][iqn][tracer]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::nn::InitNN();
    const auto config_data = MakeIqnTracerConfigData();
    const auto env_spec = MakeIqnTracerEnvSpec();
    const rl::BatchEnvSpec batch_env_spec{ 2, 1 };
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(
        dqn::DefaultDQNAgentConfig(config_data),
        anet::nn::NetworkConfig(config_data),
        batch_env_spec,
        env_spec,
        torch::Device(torch::kCPU),
        123);
    auto actor = agent->CreateActor(
        batch_env_spec, env_spec, rl::RunMode::Train, std::nullopt, torch::Device(torch::kCPU));

    const auto flags = torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kBool));
    rl::BatchState state(
        anet::TensorDict{ { kVectorKey, torch::tensor({
            { 1.0f, 2.0f, 3.0f, 4.0f },
            { 4.0f, 3.0f, 2.0f, 1.0f },
        }) } },
        flags,
        flags,
        flags);
    REQUIRE_FALSE(state.obs.Contains(anet::nn::kKey_Taus));

    auto action_info = std::dynamic_pointer_cast<dqn::DQNActionInfo>(
        actor->MakeAction(rl::StepCounts{}, state));

    REQUIRE(action_info != nullptr);
    REQUIRE(ShapeOf(action_info->GetAction()) == std::vector<int64_t>{ 2 });
    const auto& aux = action_info->GetAuxData();
    REQUIRE(ShapeOf(aux.at("q_values")) == std::vector<int64_t>{ 2, 2 });
    REQUIRE(ShapeOf(aux.at("q_quantiles")) == std::vector<int64_t>{ 2, 2, 3 });
    CHECK(torch::isfinite(aux.at("q_values")).all().item<bool>());
    CHECK(torch::isfinite(aux.at("q_quantiles")).all().item<bool>());
    CHECK_FALSE(state.obs.Contains(anet::nn::kKey_Taus));
}

TEST_CASE("DefaultDQNAgent IQN rejects a dead tau fusion branch", "[dqn][iqn][tracer]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::nn::InitNN();
    auto config_data = MakeIqnTracerConfigData();
    config_data.Set("net.body.output.[features]", "main");
    const auto env_spec = MakeIqnTracerEnvSpec();

    CHECK_THROWS_WITH(
        dqn::DefaultDQNAgent(
            dqn::DefaultDQNAgentConfig(config_data),
            anet::nn::NetworkConfig(config_data),
            rl::BatchEnvSpec{ 2, 1 },
            env_spec,
            torch::Device(torch::kCPU),
            123),
        Catch::Matchers::ContainsSubstring("IQNHead expected rank-3")
            && Catch::Matchers::ContainsSubstring("IQN fusion branch"));
}

TEST_CASE("DefaultDQNAgent IQN learner updates through the public learner path", "[dqn][iqn][learner]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::nn::InitNN();

    auto run_update = [](
        int64_t current_num_taus,
        int64_t target_num_taus = 3,
        bool use_per = true,
        bool use_tbo = false,
        torch::Device device = torch::Device(torch::kCPU),
        bool use_amp_bf16 = false) {
        auto config_data = MakeIqnTracerConfigData();
        config_data.Set("DefaultDQNAgent.learner.iqn.current_taus.num_taus", current_num_taus);
        config_data.Set("DefaultDQNAgent.learner.iqn.current_taus.sample_mode", "fixed");
        config_data.Set("DefaultDQNAgent.learner.iqn.target_taus.num_taus", target_num_taus);
        config_data.Set("DefaultDQNAgent.learner.iqn.target_taus.sample_mode", "fixed");
        config_data.Set("DefaultDQNAgent.target_policy.tau_rule.num_taus", 4);
        config_data.Set("DefaultDQNAgent.target_policy.tau_rule.sample_mode", "fixed");
        config_data.Set("DefaultDQNAgent.learner.update_warmup_steps", 0);
        config_data.Set("DefaultDQNAgent.learner.update_interval", 1);
        config_data.Set("DefaultDQNAgent.learner.use_n_step", false);
        config_data.Set("DefaultDQNAgent.learner.use_per", use_per);
        config_data.Set("DefaultDQNAgent.learner.use_tbo", use_tbo);
        config_data.Set("DefaultDQNAgent.learner.use_amp", use_amp_bf16);
        config_data.Set("DefaultDQNAgent.learner.use_amp_bf16", use_amp_bf16);
        config_data.Set("DefaultDQNAgent.learner.use_fused_optimizer", false);

        const auto env_spec = MakeIqnTracerEnvSpec();
        const rl::BatchEnvSpec batch_env_spec{ 2, 1 };
        auto agent = std::make_shared<dqn::DefaultDQNAgent>(
            dqn::DefaultDQNAgentConfig(config_data),
            anet::nn::NetworkConfig(config_data),
            batch_env_spec,
            env_spec,
            device,
            321);
        auto learner = agent->CreateLearner();

        const auto flags = torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kBool));
        const auto episode_start = torch::ones({ 2 }, torch::TensorOptions().dtype(torch::kBool));
        rl::BatchState state(
            anet::TensorDict{ { kVectorKey, torch::tensor({
                { 1.0f, 2.0f, 3.0f, 4.0f },
                { 4.0f, 3.0f, 2.0f, 1.0f },
            }) } },
            flags,
            flags,
            episode_start);
        rl::BatchState next_state(
            anet::TensorDict{ { kVectorKey, torch::tensor({
                { 1.5f, 2.5f, 3.5f, 4.5f },
                { 3.5f, 2.5f, 1.5f, 0.5f },
            }) } },
            flags,
            flags,
            flags);
        auto action_info = std::make_shared<rl::BatchActionInfo>(
            torch::tensor({ 0, 1 }, torch::TensorOptions().dtype(torch::kInt64)));
        rl::BatchExperience experience(
            state,
            action_info,
            torch::tensor({ 0.25f, -0.5f }),
            next_state);
        rl::BatchState final_state(
            anet::TensorDict{ { kVectorKey, torch::tensor({
                { 2.0f, 3.0f, 4.0f, 5.0f },
                { 3.0f, 2.0f, 1.0f, 0.0f },
            }) } },
            flags,
            flags,
            flags);
        rl::BatchExperience next_experience(
            next_state,
            action_info,
            torch::tensor({ -0.25f, 0.5f }),
            final_state);

        rl::StepCounts counts;
        counts.exp_step = 0;
        REQUIRE(learner->UpdateFromBatch(counts, experience).empty());
        counts.exp_step = 2;
        auto results = learner->UpdateFromBatch(counts, next_experience);

        REQUIRE(results.size() == 1);
        auto result = std::dynamic_pointer_cast<const dqn::BatchUpdateResult>(results.front());
        REQUIRE(result != nullptr);
        CHECK(ShapeOf(result->loss).empty());
        CHECK(ShapeOf(result->td_error) == std::vector<int64_t>{ 2 });
        CHECK(torch::isfinite(result->loss).all().item<bool>());
        CHECK(torch::isfinite(result->td_error).all().item<bool>());
        if (use_per) {
            CHECK(ShapeOf(result->per_priorities) == std::vector<int64_t>{ 2 });
            CHECK(torch::isfinite(result->per_priorities).all().item<bool>());
            const auto expected_priorities = dqn::MakePerRawPriorityBatch(
                result->td_error, 1.0e-6f, false, 50.0f).priorities.to(torch::kFloat32).cpu();
            CHECK(torch::equal(result->per_priorities, expected_priorities));
            CHECK(result->per_update_result.applied_count == 2);
            CHECK(result->per_update_result.stale_count == 0);
        } else {
            CHECK_FALSE(result->per_priorities.defined());
        }
        CHECK(torch::isfinite(result->q_std).all().item<bool>());
        CHECK(torch::isfinite(result->max_q).all().item<bool>());
        CHECK(torch::isfinite(result->q_sa).all().item<bool>());
        CHECK(torch::isfinite(result->q_gap).all().item<bool>());
        CHECK(torch::isfinite(result->q_gap_rel).all().item<bool>());
        REQUIRE(result->iqn_diagnostics.defined());
        CHECK(result->iqn_diagnostics.device().is_cpu());
        CHECK(result->iqn_diagnostics.scalar_type() == torch::kFloat32);
        CHECK(ShapeOf(result->iqn_diagnostics) == std::vector<int64_t>{ 7 });
        const auto current_mc_scale = result->GetScalar("iqn_current_mc_scale", -1);
        REQUIRE(current_mc_scale.has_value());
        if (current_num_taus >= 2) {
            CHECK(std::isfinite(*current_mc_scale));
        } else {
            CHECK(std::isnan(*current_mc_scale));
        }
        const auto target_mc_scale = result->GetScalar("iqn_target_mc_scale", -1);
        REQUIRE(target_mc_scale.has_value());
        if (target_num_taus >= 2) {
            CHECK(std::isfinite(*target_mc_scale));
        } else {
            CHECK(std::isnan(*target_mc_scale));
        }
        const auto priority_mc_ratio = result->GetScalar("iqn_priority_mc_ratio", -1);
        REQUIRE(priority_mc_ratio.has_value());
        if (current_num_taus >= 2 && target_num_taus >= 2) {
            CHECK(std::isfinite(*priority_mc_ratio));
        } else {
            CHECK(std::isnan(*priority_mc_ratio));
        }
        const auto upper_tail_spearman = result->GetScalar("upper_tail_priority_spearman", -1);
        REQUIRE(upper_tail_spearman.has_value());
        if (use_per && current_num_taus >= 2) {
            CHECK(std::isfinite(*upper_tail_spearman));
        } else {
            CHECK(std::isnan(*upper_tail_spearman));
        }

        const auto initial_count = result->GetScalar("per_sample_initial_count", -1);
        REQUIRE(initial_count.has_value());
        CHECK(*initial_count == Catch::Approx(use_per ? 2.0f : 0.0f));
        for (const char* key : {
            "iqn_first_priority_mc_ratio",
            "iqn_first_pair_abs_td",
            "iqn_first_cancellation_ratio",
            "iqn_first_quantile_loss_norm",
        }) {
            const auto value = result->GetScalar(key, -1);
            REQUIRE(value.has_value());
            if (!use_per || (std::string_view(key) == "iqn_first_priority_mc_ratio"
                    && (current_num_taus < 2 || target_num_taus < 2))) {
                CHECK(std::isnan(*value));
            } else {
                CHECK(std::isfinite(*value));
            }
        }
        CHECK_FALSE(experience.state.obs.Contains(anet::nn::kKey_Taus));
        CHECK_FALSE(experience.next_state.obs.Contains(anet::nn::kKey_Taus));
        CHECK_FALSE(next_experience.state.obs.Contains(anet::nn::kKey_Taus));
        CHECK_FALSE(next_experience.next_state.obs.Contains(anet::nn::kKey_Taus));
        return result;
    };

    SECTION("current and target sample counts may differ")
    {
        const auto result = run_update(2);
        CHECK(result->q_std.item<float>() >= 0.0f);
    }

    SECTION("one current sample reports finite zero q_std")
    {
        const auto result = run_update(1);
        CHECK(result->q_std.item<float>() == Catch::Approx(0.0f));
    }

    SECTION("one target sample leaves only target-dependent ratios undefined")
    {
        run_update(2, 1);
    }

    SECTION("PER disabled keeps general IQN diagnostics and disables first-update diagnostics")
    {
        run_update(2, 3, false);
    }

    SECTION("TBO keeps IQN diagnostics in the learner priority path")
    {
        run_update(2, 3, true, true);
    }

    SECTION("BF16 diagnostics use one CPU float32 pack on each available device")
    {
        run_update(2, 3, true, false, torch::Device(torch::kCPU), true);
        if (torch::cuda::is_available()) {
            run_update(2, 3, true, false, torch::Device(torch::kCUDA, 0), true);
        }
    }
}

TEST_CASE("DefaultDQNAgent resolves Train Actor snapshot clone overrides", "[dqn][actor][snapshot]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto make_agent = [](bool clone_model) {
        auto config = MakeDeviceForwardDefaultDqnConfig();
        config.train_actor.clone_model = clone_model;
        config.train_actor.sync_interval.value = 7;
        auto env_spec = MakeLearnerEnvSpec();
        return std::make_shared<dqn::DefaultDQNAgent>(
            config,
            MakeAgentForwardNetworkConfig(),
            rl::BatchEnvSpec{ 1, 1 },
            env_spec,
            torch::Device(torch::kCPU),
            123);
    };
    auto flags = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kBool));
    rl::BatchState state(
        anet::TensorDict{ { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }) } },
        flags,
        flags,
        flags);
    const auto read_interval = [&](const std::shared_ptr<dqn::DefaultDQNAgent>& agent,
                                   rl::RunMode mode,
                                   std::optional<bool> clone_override) {
        auto actor = agent->CreateActor(
            rl::BatchEnvSpec{ 1, 1 }, MakeLearnerEnvSpec(), mode, clone_override,
            torch::Device(torch::kCPU));
        auto info = std::dynamic_pointer_cast<dqn::DQNActionInfo>(
            actor->MakeAction(rl::StepCounts{}, state));
        REQUIRE(info != nullptr);
        const auto interval = info->GetScalar("train_actor_snapshot_interval");
        REQUIRE(interval.has_value());
        return *interval;
    };

    auto shared_default = make_agent(false);
    CHECK(std::isnan(read_interval(shared_default, rl::RunMode::Train, std::nullopt)));
    CHECK(read_interval(shared_default, rl::RunMode::Train, true) == Catch::Approx(7.0f));

    auto snapshot_default = make_agent(true);
    CHECK(read_interval(snapshot_default, rl::RunMode::Train, std::nullopt) == Catch::Approx(7.0f));
    CHECK(std::isnan(read_interval(snapshot_default, rl::RunMode::Train, false)));
    CHECK(std::isnan(read_interval(snapshot_default, rl::RunMode::Eval, true)));
}

TEST_CASE("DefaultDQNAgent rejects an effective shared Actor on another device", "[dqn][actor][snapshot][device]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeDeviceForwardDefaultDqnConfig();
    config.train_actor.clone_model = false;
    const auto batch_env_spec = rl::BatchEnvSpec{ 1, 1 };
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(
        config,
        MakeAgentForwardNetworkConfig(),
        batch_env_spec,
        MakeLearnerEnvSpec(),
        torch::Device(torch::kCPU),
        123);

    CHECK_THROWS(agent->CreateActor(
        batch_env_spec,
        MakeLearnerEnvSpec(),
        rl::RunMode::Train,
        std::nullopt,
        torch::Device(torch::kCUDA, 0)));
}

TEST_CASE("DefaultDQNAgent decides whether an Eval EnvSpec is acceptable", "[dqn][actor][env_spec]")
{
    ScopedNoopMetricsLogger metrics_logger;
    const auto batch_env_spec = rl::BatchEnvSpec{ 1, 1 };
    const auto train_env_spec = MakeLearnerEnvSpec();
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(
        MakeDeviceForwardDefaultDqnConfig(),
        MakeAgentForwardNetworkConfig(),
        batch_env_spec,
        train_env_spec,
        torch::Device(torch::kCPU),
        123);

    auto incompatible_eval_spec = train_env_spec;
    incompatible_eval_spec.action_spec.value_labels.push_back("incompatible");
    CHECK_THROWS(agent->CreateActor(
        batch_env_spec,
        incompatible_eval_spec,
        rl::RunMode::Eval,
        std::nullopt,
        torch::Device(torch::kCPU)));
}

TEST_CASE("DefaultDQNAgent creates the initial snapshot from an auto-loaded network", "[dqn][actor][snapshot][serialize]")
{
    ScopedNoopMetricsLogger metrics_logger;
    const auto env_spec = MakeLearnerEnvSpec();
    const auto batch_env_spec = rl::BatchEnvSpec{ 1, 1 };
    auto source_config = MakeDeviceForwardDefaultDqnConfig();
    auto source_agent = std::make_shared<dqn::DefaultDQNAgent>(
        source_config,
        MakeAgentForwardNetworkConfig(),
        batch_env_spec,
        env_spec,
        torch::Device(torch::kCPU),
        123);
    auto source_forward = source_agent->GetTensorDictFunction("policy-net.forward");
    REQUIRE(source_forward.has_value());
    const auto obs = anet::TensorDict{ { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }) } };
    const auto expected_q = (*source_forward)(obs).At("q").detach().clone();

    const auto unique_suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto checkpoint = std::filesystem::temp_directory_path()
        / ("anet_default_dqn_snapshot_" + std::to_string(unique_suffix) + ".bin");
    {
        std::ofstream stream(checkpoint, std::ios::binary);
        REQUIRE(stream);
        anet::OutputArchive archive(stream, checkpoint.string());
        source_agent->Save(archive);
    }

    auto loaded_config = MakeDeviceForwardDefaultDqnConfig();
    loaded_config.auto_load_file = checkpoint.string();
    loaded_config.train_actor.clone_model = true;
    auto loaded_agent = std::make_shared<dqn::DefaultDQNAgent>(
        loaded_config,
        MakeAgentForwardNetworkConfig(),
        batch_env_spec,
        env_spec,
        torch::Device(torch::kCPU),
        456);
    auto actor = loaded_agent->CreateActor(
        batch_env_spec, env_spec, rl::RunMode::Train, std::nullopt, torch::Device(torch::kCPU));
    auto flags = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kBool));
    rl::BatchState state(obs, flags, flags, flags);
    const auto action_info = actor->MakeAction(rl::StepCounts{}, state);

    CHECK(torch::allclose(action_info->GetAuxData().at("q_values"), expected_q));
    std::filesystem::remove(checkpoint);
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

TEST_CASE("RainbowAgent omits DefaultDQN snapshot diagnostics", "[dqn][actor][snapshot][rainbow]")
{
    ScopedNoopMetricsLogger metrics_logger;
    const auto env_spec = MakeLearnerEnvSpec();
    const auto batch_env_spec = rl::BatchEnvSpec{ 1, 1 };
    auto agent = std::make_shared<dqn::RainbowAgent>(
        MakeDeviceForwardRainbowConfig(),
        MakeAgentForwardNetworkConfig(),
        batch_env_spec,
        env_spec,
        torch::Device(torch::kCPU),
        123);
    auto actor = agent->CreateActor(
        batch_env_spec, env_spec, rl::RunMode::Train, std::nullopt, torch::Device(torch::kCPU));
    auto flags = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kBool));
    rl::BatchState state(
        anet::TensorDict{ { kVectorKey, torch::tensor({ { 1.0f, 2.0f } }) } },
        flags,
        flags,
        flags);
    auto action_info = std::dynamic_pointer_cast<dqn::DQNActionInfo>(
        actor->MakeAction(rl::StepCounts{}, state));

    REQUIRE(action_info != nullptr);
    CHECK_FALSE(action_info->GetScalar("train_actor_snapshot_interval").has_value());
    CHECK_FALSE(action_info->GetScalar("train_actor_snapshot_age").has_value());
    CHECK_THROWS(agent->CreateActor(
        batch_env_spec,
        env_spec,
        rl::RunMode::Train,
        std::nullopt,
        torch::Device(torch::kCUDA, 0)));
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

TEST_CASE("NetworkModel exposes the resolved distributional mode", "[dqn][network_model]")
{
    TestNetworkModel scalar_model(false);
    TestNetworkModel distributional_model(true);

    CHECK_FALSE(scalar_model.IsDistributional());
    CHECK(distributional_model.IsDistributional());
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
    auto context = std::make_shared<rl::DefaultActionContext>(rl::RunMode::Train, 123);
    auto mutex = std::make_shared<std::shared_mutex>();
    anet::ProfiledValueConfig<rl::step_t> sync_interval;
    sync_interval.value = 2;
    dqn::Actor actor(
        policy, nullptr, context, mutex, snapshot_network, src_network, false, sync_interval, true);

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
    auto context = std::make_shared<rl::DefaultActionContext>(rl::RunMode::Train, 123);
    auto mutex = std::make_shared<std::shared_mutex>();
    anet::ProfiledValueConfig<rl::step_t> sync_interval;
    sync_interval.value = 5;
    dqn::Actor actor(
        policy, nullptr, context, mutex, snapshot_network, src_network, false, sync_interval, true);

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
        auto context = std::make_shared<rl::DefaultActionContext>(rl::RunMode::Train, 123);
        auto mutex = std::make_shared<std::shared_mutex>();
        dqn::Actor actor(
            policy, nullptr, context, mutex, snapshot_network, src_network, false, sync_interval, true);
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
    auto context = std::make_shared<rl::DefaultActionContext>(rl::RunMode::Train, 123);
    auto mutex = std::make_shared<std::shared_mutex>();
    dqn::Actor actor(policy, nullptr, context, mutex, network, network, false, std::nullopt, true);
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
    auto action_info = policy.SelectAction(obs, /*greedy_only=*/true, network, rnd, {});

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
    auto action_info = policy.SelectAction(
        obs, /*greedy_only=*/true, MakePassthroughNetwork(1, 5),
        std::make_shared<anet::RandomGenerator>(123), {});

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
    auto action_info = policy.SelectAction(
        obs, /*greedy_only=*/true, MakePassthroughNetwork(1, 5),
        std::make_shared<anet::RandomGenerator>(123), {});

    CHECK(*action_info->GetScalar("policy_selected_crossing_depth_p90_ratio")
        == Catch::Approx(0.0f));
}

TEST_CASE("QR selected crossing depth aggregates BF16 quantiles in float32", "[dqn][qr][action_policy][metrics][bf16]")
{
    const auto q_quantiles = torch::tensor({ { { 0.0f, 2.0f, 1.0f, 4.0f, 2.0f } } })
        .to(torch::kBFloat16);
    const auto obs = anet::TensorDict{
        { "q", q_quantiles.mean(2) },
        { "q_dist", q_quantiles },
    };

    dqn::ActionPolicyConfig config;
    config.quantile_mode = "qr";
    dqn::EpsilonGreedyActionPolicy policy(config);
    auto action_info = policy.SelectAction(
        obs, /*greedy_only=*/true, MakePassthroughNetwork(1, 5),
        std::make_shared<anet::RandomGenerator>(123), {});

    CHECK(*action_info->GetScalar("policy_selected_crossing_depth_p90_ratio")
        == Catch::Approx(0.5f));
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
    auto action_info = policy.SelectAction(
        obs, /*greedy_only=*/true, MakePassthroughNetwork(2, 5),
        std::make_shared<anet::RandomGenerator>(123), {});
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
    auto odd_info = policy.SelectAction(
        odd_obs, /*greedy_only=*/true, MakePassthroughNetwork(1, 3), rnd, {});
    CHECK(*odd_info->GetScalar("policy_upper_truncated_std") == Catch::Approx(4.0f));
    CHECK(*odd_info->GetScalar("policy_lower_truncated_std") == Catch::Approx(1.0f));

    const auto one_quantile = torch::tensor({ { { 2.0f } } });
    const auto one_obs = anet::TensorDict{
        { "q", one_quantile.mean(2) },
        { "q_dist", one_quantile },
    };
    auto one_info = policy.SelectAction(
        one_obs, /*greedy_only=*/true, MakePassthroughNetwork(1, 1), rnd, {});
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

        REQUIRE_FALSE(obs.Contains(anet::nn::kKey_Taus));
        auto action_info = policy->SelectAction(obs, /*greedy_only=*/true, network, rnd);

        REQUIRE(state->forward_count == 1);
        REQUIRE(state->last_taus.defined());
        CHECK(torch::allclose(state->last_taus, expected_taus, 1.0e-6, 1.0e-6));
        CHECK_FALSE(obs.Contains(anet::nn::kKey_Taus));
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
        config.eps_start = 0.0f;
        config.eps_end = 0.0f;
        config.tau_rule.num_taus = 4;
        config.tau_rule.sample_mode = "fixed";
        auto expected = torch::tensor({ 0.125f, 0.375f, 0.625f, 0.875f }).repeat({ 2, 1 });

        run_policy(std::make_shared<dqn::EpsilonGreedyActionPolicy>(config), expected);
    }

    SECTION("UQE tail mean scores the sampled upper tail")
    {
        dqn::ActionPolicyConfig config;
        config.quantile_mode = "iqn";
        config.uqe_tau_start = 0.5f;
        config.uqe_tau_end = 0.5f;
        config.uqe_use_tail_mean = true;
        config.uqe_eps_start = 0.0f;
        config.uqe_eps_end = 0.0f;
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
        config.uqe_tau_start = 0.2f;
        config.uqe_tau_end = 0.6f;
        config.uqe_tau_decay_steps = 10;
        config.uqe_use_tail_mean = false;
        config.uqe_eps_start = 0.0f;
        config.uqe_eps_end = 0.0f;
        config.tau_rule.num_taus = 3;
        config.tau_rule.sample_mode = "fixed";
        auto policy = std::make_shared<dqn::UQEActionPolicy>(config);
        policy->OnLearn(rl::StepCounts{ .exp_step = 5 });
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
        config.use_spatial_exploration = true;
        config.spatial_scale_type = "linear";
        config.uqe_tau_start = 0.0f;
        config.uqe_tau_end = 0.5f;
        config.uqe_use_tail_mean = true;
        config.uqe_eps_start = 0.0f;
        config.uqe_eps_end = 0.0f;
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
    config.uqe_tau_start = 0.4f;
    config.uqe_tau_end = 0.4f;
    config.uqe_use_tail_mean = false;
    config.uqe_eps_start = 0.0f;
    config.uqe_eps_end = 0.0f;
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

    REQUIRE_FALSE(obs.Contains(anet::nn::kKey_Taus));
    auto action_info = policy->SelectAction(obs, /*greedy_only=*/true, network, rnd, {});

    const auto expected_taus = torch::tensor({ 0.4f, 0.125f, 0.375f, 0.625f, 0.875f }).repeat({ 2, 1 });
    REQUIRE(state->forward_count == 1);
    CHECK(torch::allclose(state->last_taus, expected_taus, 1.0e-6, 1.0e-6));
    CHECK_FALSE(obs.Contains(anet::nn::kKey_Taus));

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
    config.uqe_tau_start = 0.5f;
    config.uqe_tau_end = 0.5f;
    config.uqe_use_tail_mean = true;
    config.uqe_eps_start = 0.0f;
    config.uqe_eps_end = 0.0f;
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

    auto action_info = policy.SelectAction(obs, /*greedy_only=*/true, network, rnd, {});

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
    auto risk_only = dqn::UQEActionPolicy(config).SelectAction(
        obs, /*greedy_only=*/true, MakeTauEchoNetwork(2, risk_only_state), rnd, {});
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
    auto random_full = dqn::UQEActionPolicy(config).SelectAction(
        obs, /*greedy_only=*/true, MakeTauEchoNetwork(4, random_full_state), rnd, {});
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
        auto action_info = policy.SelectAction(obs, /*greedy_only=*/true, network, rnd, {});

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
        auto context = std::make_shared<rl::DefaultActionContext>(rl::RunMode::Train, 123);
        auto mutex = std::make_shared<std::shared_mutex>();
        dqn::Actor actor(policy, nullptr, context, mutex, network, network, emit_hint);

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
    auto context = std::make_shared<rl::DefaultActionContext>(rl::RunMode::Train, 123);
    auto mutex = std::make_shared<std::shared_mutex>();
    anet::ProfiledValueConfig<rl::step_t> sync_interval;
    sync_interval.value = 1;
    dqn::Actor actor(
        policy, nullptr, context, mutex, snapshot_network, source_network, false, sync_interval, true);

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

TEST_CASE("DQN Actor Q hint schema packs and decodes two columns", "[dqn][per][actor_initial][hint]")
{
    auto q_sa = torch::tensor({ 2.0f, 3.0f });
    auto state_value = torch::tensor({ 5.0f, 7.0f });

    auto packed = dqn::PackActorQHint(q_sa, state_value);
    CHECK(packed.scalar_type() == torch::kFloat32);
    CHECK(packed.sizes() == torch::IntArrayRef({ 2, dqn::kActorQHintColumnCount }));

    const auto batch = dqn::DecodeActorQHint(packed);
    CHECK(torch::equal(batch.actor_q_sa, q_sa));
    CHECK(torch::equal(batch.actor_state_value, state_value));

    const std::array<float, 2> row{ 11.0f, 13.0f };
    const auto decoded_row = dqn::DecodeActorQHint(std::span<const float>(row));
    CHECK(decoded_row.actor_q_sa == Catch::Approx(11.0f));
    CHECK(decoded_row.actor_state_value == Catch::Approx(13.0f));

    CHECK_THROWS(dqn::DecodeActorQHint(torch::zeros({ 1, 3 }, torch::kFloat32)));
    CHECK_THROWS(dqn::DecodeActorQHint(std::span<const float>(row.data(), 1)));
}

TEST_CASE("DQNActionInfo regathers Actor Q hint after action replacement", "[dqn][per][actor_initial]")
{
    auto q_values = torch::tensor({
        { 1.0f, 5.0f, 2.0f },
        { 7.0f, 3.0f, 4.0f },
    });
    rl::AuxData aux;
    aux["q_values"] = q_values;
    auto packed = torch::tensor({ { 5.0f, 5.0f }, { 7.0f, 7.0f } });
    dqn::DQNActionInfo info(
        torch::tensor({ 1, 0 }, torch::TensorOptions().dtype(torch::kInt64)),
        {},
        aux,
        rl::ReplayInitialPriorityHint(packed));

    auto replaced = info.WithAction(torch::tensor({ 2, 1 }, torch::TensorOptions().dtype(torch::kInt64)));
    REQUIRE(replaced->GetReplayInitialPriorityHint().has_value());
    CHECK(torch::equal(
        replaced->GetReplayInitialPriorityHint()->GetPayload(),
        torch::tensor({ { 2.0f, 5.0f }, { 3.0f, 7.0f } })));
    const auto& first_cpu = replaced->GetReplayInitialPriorityHint()->GetPayloadCpu();
    const auto& second_cpu = replaced->GetReplayInitialPriorityHint()->GetPayloadCpu();
    CHECK(first_cpu.unsafeGetTensorImpl() == second_cpu.unsafeGetTensorImpl());
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

TEST_CASE("ActionPolicy reverses spatial parameter tuples for env lane assignment", "[dqn][action_policy][spatial]")
{
    auto device = torch::Device(torch::kCPU);

    // 補間結果を厳密に反転し、値の集合を変えずにend側をenv[0]へ割り当てる。
    auto linear = ActionPolicyAccess::CreateSpatialTensor(3, 1.0f, 0.0f, "linear", device);
    auto lane_linear = ActionPolicyAccess::CreateSpatialLaneTensor(3, 1.0f, 0.0f, "linear", device);
    CHECK(torch::equal(lane_linear, linear.flip({ 0 })));
    CHECK(torch::equal(lane_linear, torch::tensor({ 0.0f, 0.5f, 1.0f })));

    auto log = ActionPolicyAccess::CreateSpatialTensor(3, 1.0f, 0.01f, "log", device);
    auto lane_log = ActionPolicyAccess::CreateSpatialLaneTensor(3, 1.0f, 0.01f, "log", device);
    CHECK(torch::equal(lane_log, log.flip({ 0 })));

    // epsilonとtauを同じ向きで反転し、パラメータの組の集合を維持する。
    auto original_pairs = torch::stack({
        ActionPolicyAccess::CreateSpatialTensor(3, 0.6f, 0.0f, "log", device),
        ActionPolicyAccess::CreateSpatialTensor(3, 0.95f, 0.85f, "log", device),
    }, 1);
    auto lane_pairs = torch::stack({
        ActionPolicyAccess::CreateSpatialLaneTensor(3, 0.6f, 0.0f, "log", device),
        ActionPolicyAccess::CreateSpatialLaneTensor(3, 0.95f, 0.85f, "log", device),
    }, 1);
    CHECK(torch::equal(lane_pairs, original_pairs.flip({ 0 })));
    CHECK(lane_pairs[0][0].item<float>() == Catch::Approx(1.0e-4f).margin(1.0e-7f));
    CHECK(lane_pairs[2][0].item<float>() == Catch::Approx(0.6f).margin(1.0e-6f));

    // laneが1つなら反転はno-opとなり、従来どおりstart値を使う。
    auto single = ActionPolicyAccess::CreateSpatialLaneTensor(1, 0.25f, 0.75f, "linear", device);
    CHECK(single[0].item<float>() == Catch::Approx(0.25f).margin(1.0e-6f));
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

TEST_CASE("DefaultDQNAgentConfig defines IQN and QR sampling defaults", "[dqn][iqn][config]")
{
    const dqn::DefaultDQNAgentConfig config(anet::ConfigData{});

    CHECK(config.quantile_mode == "qr");
    CHECK(config.qr.num_quantiles == 51);
    CHECK(config.train_policy.tau_rule.num_taus == 32);
    CHECK(config.train_policy.tau_rule.sample_mode == "random");
    CHECK(config.eval_policy.tau_rule.num_taus == 32);
    CHECK(config.eval_policy.tau_rule.sample_mode == "fixed");
    CHECK(config.target_policy.tau_rule.num_taus == 32);
    CHECK(config.target_policy.tau_rule.sample_mode == "fixed");
    CHECK_FALSE(config.train_policy.full_distribution_query.enabled);
    CHECK(config.train_policy.full_distribution_query.tau_rule.num_taus == 32);
    CHECK(config.train_policy.full_distribution_query.tau_rule.sample_mode == "fixed");
    CHECK_FALSE(config.eval_policy.full_distribution_query.enabled);
    CHECK(config.eval_policy.full_distribution_query.tau_rule.num_taus == 32);
    CHECK(config.eval_policy.full_distribution_query.tau_rule.sample_mode == "fixed");
    CHECK_FALSE(config.target_policy.full_distribution_query.enabled);
    CHECK(config.target_policy.full_distribution_query.tau_rule.num_taus == 32);
    CHECK(config.target_policy.full_distribution_query.tau_rule.sample_mode == "fixed");
    CHECK(config.learner.iqn.current_taus.num_taus == 64);
    CHECK(config.learner.iqn.current_taus.sample_mode == "random");
    CHECK(config.learner.iqn.target_taus.num_taus == 64);
    CHECK(config.learner.iqn.target_taus.sample_mode == "random");
}

TEST_CASE("DefaultDQNAgentConfig propagates quantile mode to every policy", "[dqn][iqn][config]")
{
    for (const auto& mode : { "none", "qr", "iqn" }) {
        INFO("mode=" << mode);
        anet::ConfigData config_data;
        config_data.Set("DefaultDQNAgent.quantile_mode", mode);
        const dqn::DefaultDQNAgentConfig config(config_data);

        CHECK(config.train_policy.quantile_mode == mode);
        CHECK(config.eval_policy.quantile_mode == mode);
        CHECK(config.target_policy.quantile_mode == mode);
    }
}

TEST_CASE("DefaultDQNAgentConfig reads an optional IQN UQE full distribution query", "[dqn][iqn][config]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.quantile_mode", "iqn");
    config_data.Set("DefaultDQNAgent.eval_policy.policy_type", "UQE");
    config_data.Set("DefaultDQNAgent.eval_policy.full_distribution_query.enabled", true);
    config_data.Set("DefaultDQNAgent.eval_policy.full_distribution_query.tau_rule.num_taus", 5);
    config_data.Set("DefaultDQNAgent.eval_policy.full_distribution_query.tau_rule.sample_mode", "random");

    const dqn::DefaultDQNAgentConfig config(config_data);

    CHECK_FALSE(config.train_policy.full_distribution_query.enabled);
    CHECK(config.eval_policy.full_distribution_query.enabled);
    CHECK(config.eval_policy.full_distribution_query.tau_rule.num_taus == 5);
    CHECK(config.eval_policy.full_distribution_query.tau_rule.sample_mode == "random");
    CHECK_FALSE(config.target_policy.full_distribution_query.enabled);
}

TEST_CASE("DefaultDQNAgent config fixture resolves the IQN profile chain", "[dqn][iqn][config]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "default-dqn-iqn-profile-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    // live実験値から分離したprofile連鎖を作り、後段IQN設定の上書きを固定する。
    const auto config_path = root / "config.txt";
    {
        std::ofstream stream(config_path);
        REQUIRE(stream.is_open());
        stream << "DefaultDQNAgent.$ = DefaultDQNAgent.baseline > A > IQN\n";
        stream << "DefaultDQNAgent.baseline.quantile_mode = qr\n";
        stream << "A.learner.iqn.current_taus.num_taus = 64\n";
        stream << "A.learner.iqn.target_taus.num_taus = 64\n";
        stream << "IQN.quantile_mode = iqn\n";
        stream << "IQN.train_policy.policy_type = UQE\n";
        stream << "IQN.train_policy.tau_rule.num_taus = 32\n";
        stream << "IQN.train_policy.tau_rule.sample_mode = random\n";
        stream << "IQN.eval_policy.policy_type = UQE\n";
        stream << "IQN.eval_policy.uqe_use_tail_mean = true\n";
        stream << "IQN.eval_policy.tau_rule.num_taus = 32\n";
        stream << "IQN.eval_policy.tau_rule.sample_mode = fixed\n";
        stream << "IQN.eval_policy.full_distribution_query.enabled = true\n";
        stream << "IQN.eval_policy.full_distribution_query.tau_rule.num_taus = 32\n";
        stream << "IQN.eval_policy.full_distribution_query.tau_rule.sample_mode = fixed\n";
        stream << "IQN.learner.iqn.current_taus.num_taus = 32\n";
        stream << "IQN.learner.iqn.target_taus.num_taus = 32\n";
        stream << "net.$ = net.base > net.iqn\n";
        stream << "net.base.branch.[value_stream].bind = main_feature\n";
        stream << "net.base.branch.[adv_stream].bind = main_feature\n";
        stream << "net.iqn.block.[IQNTauProjFixture].linear.out_features = 2048\n";
        stream << "net.iqn.branch.[tau_embedding].bind = taus\n";
        stream << "net.iqn.branch.[iqn_fusion].bind = main_feature * tau_embedding\n";
        stream << "net.iqn.branch.[value_stream].bind = iqn_fusion\n";
        stream << "net.iqn.branch.[adv_stream].bind = iqn_fusion\n";
    }

    anet::ConfigManager config_manager(config_path.string());
    const auto config_data = config_manager.GetConfigData();
    const dqn::DefaultDQNAgentConfig config(config_data);

    // Agent設定とNN設定の両方で、profile連鎖の最終値が読み出せることを確認する。
    CHECK(config.quantile_mode == "iqn");
    CHECK(config.train_policy.policy_type == "UQE");
    CHECK(config.train_policy.tau_rule.num_taus == 32);
    CHECK(config.train_policy.tau_rule.sample_mode == "random");
    CHECK_FALSE(config.train_policy.full_distribution_query.enabled);
    CHECK(config.eval_policy.policy_type == "UQE");
    CHECK(config.eval_policy.uqe_use_tail_mean);
    CHECK(config.eval_policy.tau_rule.num_taus == 32);
    CHECK(config.eval_policy.tau_rule.sample_mode == "fixed");
    CHECK(config.eval_policy.full_distribution_query.enabled);
    CHECK(config.eval_policy.full_distribution_query.tau_rule.num_taus == 32);
    CHECK(config.eval_policy.full_distribution_query.tau_rule.sample_mode == "fixed");
    CHECK_FALSE(config.target_policy.full_distribution_query.enabled);
    CHECK(config_data.Get("A.learner.iqn.current_taus.num_taus") == "64");
    CHECK(config.learner.iqn.current_taus.num_taus == 32);
    CHECK(config.learner.iqn.target_taus.num_taus == 32);
    CHECK(config_data.Get("net.block.[IQNTauProjFixture].linear.out_features") == "2048");
    CHECK(config_data.Get("net.branch.[tau_embedding].bind") == "taus");
    CHECK(config_data.Get("net.branch.[iqn_fusion].bind") == "main_feature * tau_embedding");
    CHECK(config_data.Get("net.branch.[value_stream].bind") == "iqn_fusion");
    CHECK(config_data.Get("net.branch.[adv_stream].bind") == "iqn_fusion");

    std::filesystem::remove_all(root);
}

TEST_CASE("DefaultDQNAgentConfig restores deterministic target taus after optimistic policy copy", "[dqn][iqn][config]")
{
    anet::ConfigData inherited_data;
    inherited_data.Set("DefaultDQNAgent.quantile_mode", "iqn");
    inherited_data.Set("DefaultDQNAgent.use_optimistic_target", true);
    inherited_data.Set("DefaultDQNAgent.train_policy.policy_type", "UQE");
    inherited_data.Set("DefaultDQNAgent.train_policy.tau_rule.num_taus", 7);
    inherited_data.Set("DefaultDQNAgent.train_policy.tau_rule.sample_mode", "random");
    inherited_data.Set("DefaultDQNAgent.train_policy.full_distribution_query.enabled", true);
    inherited_data.Set("DefaultDQNAgent.train_policy.full_distribution_query.tau_rule.num_taus", 7);
    inherited_data.Set("DefaultDQNAgent.train_policy.full_distribution_query.tau_rule.sample_mode", "random");
    const dqn::DefaultDQNAgentConfig inherited(inherited_data);

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
    const dqn::DefaultDQNAgentConfig explicit_target(explicit_data);

    CHECK(explicit_target.target_policy.tau_rule.num_taus == 5);
    CHECK(explicit_target.target_policy.tau_rule.sample_mode == "random");
    CHECK(explicit_target.target_policy.full_distribution_query.enabled);
    CHECK(explicit_target.target_policy.full_distribution_query.tau_rule.num_taus == 5);
    CHECK(explicit_target.target_policy.full_distribution_query.tau_rule.sample_mode == "random");
}

TEST_CASE("DefaultDQNAgentConfig validates quantile modes and tau rules", "[dqn][iqn][config]")
{
    SECTION("mode and QR width")
    {
        anet::ConfigData invalid_mode;
        invalid_mode.Set("DefaultDQNAgent.quantile_mode", "invalid");
        CHECK_THROWS(dqn::DefaultDQNAgentConfig(invalid_mode));

        anet::ConfigData invalid_qr;
        invalid_qr.Set("DefaultDQNAgent.quantile_mode", "qr");
        invalid_qr.Set("DefaultDQNAgent.qr.num_quantiles", 1);
        CHECK_THROWS(dqn::DefaultDQNAgentConfig(invalid_qr));
    }

    SECTION("all tau rules")
    {
        const std::vector<std::string> prefixes{
            "DefaultDQNAgent.train_policy.tau_rule",
            "DefaultDQNAgent.eval_policy.tau_rule",
            "DefaultDQNAgent.target_policy.tau_rule",
            "DefaultDQNAgent.train_policy.full_distribution_query.tau_rule",
            "DefaultDQNAgent.eval_policy.full_distribution_query.tau_rule",
            "DefaultDQNAgent.target_policy.full_distribution_query.tau_rule",
            "DefaultDQNAgent.learner.iqn.current_taus",
            "DefaultDQNAgent.learner.iqn.target_taus",
        };
        for (const auto& prefix : prefixes) {
            INFO("prefix=" << prefix);

            for (const auto& mode : { "stratified", "systematic", "antithetic" }) {
                INFO("mode=" << mode);
                anet::ConfigData valid_mode;
                valid_mode.Set(prefix + ".sample_mode", mode);
                CHECK_NOTHROW(dqn::DefaultDQNAgentConfig(valid_mode));
            }

            anet::ConfigData invalid_count;
            invalid_count.Set(prefix + ".num_taus", 0);
            CHECK_THROWS(dqn::DefaultDQNAgentConfig(invalid_count));

            anet::ConfigData invalid_mode;
            invalid_mode.Set(prefix + ".sample_mode", "sorted_random");
            CHECK_THROWS(dqn::DefaultDQNAgentConfig(invalid_mode));
        }
    }

    SECTION("full distribution query compatibility")
    {
        for (const auto& [mode, policy_type] : {
                 std::pair{ "none", "Greedy" },
                 std::pair{ "qr", "UQE" },
             }) {
            INFO("mode=" << mode);
            anet::ConfigData non_iqn;
            non_iqn.Set("DefaultDQNAgent.quantile_mode", mode);
            non_iqn.Set("DefaultDQNAgent.eval_policy.policy_type", policy_type);
            non_iqn.Set("DefaultDQNAgent.eval_policy.full_distribution_query.enabled", true);

            const dqn::DefaultDQNAgentConfig config(non_iqn);

            CHECK(config.eval_policy.full_distribution_query.enabled);
        }

        anet::ConfigData non_uqe;
        non_uqe.Set("DefaultDQNAgent.quantile_mode", "iqn");
        non_uqe.Set("DefaultDQNAgent.eval_policy.policy_type", "Greedy");
        non_uqe.Set("DefaultDQNAgent.eval_policy.full_distribution_query.enabled", true);
        CHECK_THROWS(dqn::DefaultDQNAgentConfig(non_uqe));
    }

    SECTION("IQN kappa")
    {
        for (const auto& value : { "0", "-0.5", "nan", "inf" }) {
            INFO("kappa=" << value);
            anet::ConfigData config_data;
            config_data.Set("DefaultDQNAgent.quantile_mode", "iqn");
            config_data.Set("DefaultDQNAgent.learner.quantile_huber_kappa", value);
            CHECK_THROWS(dqn::DefaultDQNAgentConfig(config_data));
        }
    }
}

TEST_CASE("DefaultDQNAgentConfig validates IQN risk ranges only when consumed", "[dqn][iqn][config]")
{
    for (const auto& type : { "UQE", "ThompsonSampling" }) {
        INFO("type=" << type);
        anet::ConfigData none_data;
        none_data.Set("DefaultDQNAgent.quantile_mode", "none");
        none_data.Set("DefaultDQNAgent.train_policy.policy_type", type);
        CHECK_THROWS(dqn::DefaultDQNAgentConfig(none_data));
    }

    anet::ConfigData uqe_data;
    uqe_data.Set("DefaultDQNAgent.quantile_mode", "iqn");
    uqe_data.Set("DefaultDQNAgent.train_policy.policy_type", "UQE");
    uqe_data.Set("DefaultDQNAgent.train_policy.uqe_tau_start", -0.1f);
    CHECK_THROWS(dqn::DefaultDQNAgentConfig(uqe_data));

    anet::ConfigData non_spatial_thompson;
    non_spatial_thompson.Set("DefaultDQNAgent.quantile_mode", "iqn");
    non_spatial_thompson.Set("DefaultDQNAgent.train_policy.policy_type", "ThompsonSampling");
    non_spatial_thompson.Set("DefaultDQNAgent.train_policy.use_spatial_exploration", false);
    non_spatial_thompson.Set("DefaultDQNAgent.train_policy.uqe_tau_start", -0.1f);
    non_spatial_thompson.Set("DefaultDQNAgent.train_policy.uqe_tau_end", 1.1f);
    CHECK_NOTHROW(dqn::DefaultDQNAgentConfig(non_spatial_thompson));

    auto spatial_thompson = non_spatial_thompson;
    spatial_thompson.Set("DefaultDQNAgent.train_policy.use_spatial_exploration", true);
    CHECK_THROWS(dqn::DefaultDQNAgentConfig(spatial_thompson));

    anet::ConfigData qr_uqe;
    qr_uqe.Set("DefaultDQNAgent.quantile_mode", "qr");
    qr_uqe.Set("DefaultDQNAgent.train_policy.policy_type", "UQE");
    qr_uqe.Set("DefaultDQNAgent.train_policy.uqe_tau_start", -0.1f);
    qr_uqe.Set("DefaultDQNAgent.train_policy.uqe_tau_end", 1.1f);
    CHECK_NOTHROW(dqn::DefaultDQNAgentConfig(qr_uqe));
}

TEST_CASE("DefaultDQNAgentConfig defaults Train Actor snapshot to shared mode", "[dqn][config][snapshot]")
{
    dqn::DefaultDQNAgentConfig config;

    CHECK_FALSE(config.train_actor.clone_model);
    CHECK(config.train_actor.sync_interval.type == "constant");
    CHECK(config.train_actor.sync_interval.value == 400);
    REQUIRE(config.train_actor.sync_interval.min_value.has_value());
    CHECK(*config.train_actor.sync_interval.min_value == 1);
}

TEST_CASE("DefaultDQNAgentConfig rejects malformed Train Actor snapshot values", "[dqn][config][snapshot]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.train_actor.clone_model", "false");
    config_data.Set("DefaultDQNAgent.train_actor.sync_interval.value", "400x");

    CHECK_THROWS_WITH(
        dqn::DefaultDQNAgentConfig(config_data),
        Catch::Matchers::ContainsSubstring("DefaultDQNAgent.train_actor.sync_interval.value")
            && Catch::Matchers::ContainsSubstring("400x")
            && Catch::Matchers::ContainsSubstring("expected=uint64_t"));
}

TEST_CASE("DefaultDQNAgentConfig requires a positive active snapshot interval", "[dqn][config][snapshot]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.train_actor.clone_model", "false");
    config_data.Set("DefaultDQNAgent.train_actor.sync_interval.type", "constant");
    config_data.Set("DefaultDQNAgent.train_actor.sync_interval.value", "0");

    CHECK_THROWS_WITH(
        dqn::DefaultDQNAgentConfig(config_data),
        Catch::Matchers::ContainsSubstring("key=train_actor.sync_interval.value")
        && Catch::Matchers::ContainsSubstring("value=0")
        && Catch::Matchers::ContainsSubstring("expected=>=1"));
}

TEST_CASE("DefaultDQNAgentConfig strictly parses every explicit snapshot field", "[dqn][config][snapshot]")
{
    struct InvalidValue {
        std::string key;
        std::string value;
    };
    const std::vector<InvalidValue> invalid_values{
        { "DefaultDQNAgent.train_actor.clone_model", "maybe" },
        { "DefaultDQNAgent.train_actor.sync_interval.start", "-1" },
        { "DefaultDQNAgent.train_actor.sync_interval.steps", "18446744073709551616" },
        { "DefaultDQNAgent.train_actor.sync_interval.cycle_mult", "nan" },
        { "DefaultDQNAgent.train_actor.sync_interval.cycle_mult", "inf" },
    };
    for (const auto& invalid : invalid_values) {
        INFO("key=" << invalid.key << " value=" << invalid.value);
        anet::ConfigData config_data;
        config_data.Set(invalid.key, invalid.value);
        CHECK_THROWS(dqn::DefaultDQNAgentConfig(config_data));
    }

    anet::ConfigData comma_data;
    comma_data.Set("DefaultDQNAgent.train_actor.sync_interval.value", "1,000");
    const dqn::DefaultDQNAgentConfig comma_config(comma_data);
    CHECK(comma_config.train_actor.sync_interval.value == 1000);
}

TEST_CASE("DefaultDQNAgentConfig validates phased snapshot profiles", "[dqn][config][snapshot]")
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.train_actor.sync_interval.type", "phased");
    config_data.Set("DefaultDQNAgent.train_actor.sync_interval.phases", "warm main");
    config_data.Set("DefaultDQNAgent.train_actor.sync_interval.phase.[warm].type", "constant");
    config_data.Set("DefaultDQNAgent.train_actor.sync_interval.phase.[warm].value", "4");
    config_data.Set("DefaultDQNAgent.train_actor.sync_interval.phase.[warm].steps", "10");
    config_data.Set("DefaultDQNAgent.train_actor.sync_interval.phase.[main].type", "linear");
    config_data.Set("DefaultDQNAgent.train_actor.sync_interval.phase.[main].start", "4");
    config_data.Set("DefaultDQNAgent.train_actor.sync_interval.phase.[main].end", "2");
    config_data.Set("DefaultDQNAgent.train_actor.sync_interval.phase.[main].steps", "20");

    const dqn::DefaultDQNAgentConfig config(config_data);

    anet::ProfiledValue<rl::step_t> interval(config.train_actor.sync_interval);
    CHECK(interval.Evaluate(0) == 4);
    CHECK(interval.Evaluate(10) == 4);
    CHECK(interval.Evaluate(30) == 2);

    anet::ConfigData undefined_phase_data;
    undefined_phase_data.Set("DefaultDQNAgent.train_actor.sync_interval.type", "phased");
    undefined_phase_data.Set("DefaultDQNAgent.train_actor.sync_interval.phases", "missing");
    CHECK_THROWS(dqn::DefaultDQNAgentConfig(undefined_phase_data));
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

    auto expected_actions = torch::tensor({ 1, 0 }, torch::TensorOptions().dtype(torch::kInt64));

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
