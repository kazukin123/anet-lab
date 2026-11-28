#pragma once

#include "anet/test/catch.hpp"
#include "default_dqn_agent.hpp"
#include "anet/env.hpp"
#include "anet/metrics_logger.hpp"
#include "rainbow_agent.hpp"
#include "anet/test/util.hpp"
#include "anet/trainer.hpp"
#include "anet/metrics.hpp"
#include "dqn_heads.hpp"
#include "anet/test/nn.hpp"
#include "anet/nn_head.hpp"
#include "anet/replay_experience.hpp"
#include <ATen/autocast_mode.h>
#include <torch/csrc/autograd/profiler_legacy.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <span>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <wx/cmdline.h>
#include <wx/utils.h>
#include "anet/tensor_check.hpp"
#include "dqn_math.hpp"
#include "dqn_network_model.hpp"
#include "dqn_actor.hpp"
#include "dqn_learner.hpp"
#include "dqn_assembler.hpp"

constexpr char kFixtureFlatten[] = "Flatten";
constexpr char kFixtureMainBranch[] = "main";

namespace {

static void SaveRfd0170Trace(const anet::rl::AuxData& aux, const char* name)
{
    constexpr const char* kOutputEnv = "ANET_RFD0170_TRACE_DIR";
    const auto* directory = std::getenv(kOutputEnv);
    if (!directory) return;
    // 比較用の全要素を辞書順で保存し、表示の丸めや省略を介さずに突き合わせる。
    anet::json snapshot = anet::json::object();
    for (const auto& [key, value] : aux) {
        if (!key.starts_with(anet::nn::kNnTracePrefix)) continue;
        const auto tensor = value.detach().to(torch::kCPU).contiguous();
        REQUIRE(tensor.scalar_type() == torch::kFloat32);
        snapshot[key] = {{"shape", tensor.sizes().vec()}, {"dtype", "float32"},
            {"values", std::vector<float>(tensor.data_ptr<float>(), tensor.data_ptr<float>() + tensor.numel())}};
    }
    REQUIRE_FALSE(snapshot.empty());
    std::filesystem::create_directories(directory);
    std::ofstream output(std::filesystem::path(directory) / (std::string(name) + ".json"));
    REQUIRE(output.good());
    output << snapshot.dump(2) << '\n';
}

constexpr const char* kRfd0121AdvFeature = "adv_feature";

constexpr const char* kRfd0121ExtrinsicQArgmaxDisagreement = "extrinsic_q_argmax_disagreement";

constexpr const char* kRfd0121ExtrinsicQArgmaxDisagreementRate = "extrinsic_q_argmax_disagreement_rate";

constexpr const char* kRfd0121ExtrinsicQGapMean = "extrinsic_q_gap_mean";

constexpr const char* kRfd0121ExtrinsicQRegret = "extrinsic_q_regret";

constexpr const char* kRfd0121ExtrinsicQRegretMean = "extrinsic_q_regret_mean";

constexpr const char* kRfd0121ExtrinsicQSessionArgmaxDisagreementRate = "extrinsic_q_session_argmax_disagreement_rate";

constexpr const char* kRfd0121ExtrinsicQSessionRegretMean = "extrinsic_q_session_regret_mean";

constexpr const char* kRfd0121Gap = "_gap";

constexpr const char* kRfd0121GapMean = "gap_mean";

constexpr const char* kRfd0121Intrinsic = "intrinsic";

constexpr const char* kRfd0121IntrinsicPrefix = "intrinsic_";

constexpr const char* kRfd0121IntrinsicQGap = "intrinsic_q_gap";

constexpr const char* kRfd0121IntrinsicQGapMean = "intrinsic_q_gap_mean";

constexpr const char* kRfd0121IntrinsicQSessionGapMean = "intrinsic_q_session_gap_mean";

constexpr const char* kRfd0121OnlineReuse = "online_reuse";

constexpr const char* kRfd0121PlasticityTargetSrank = "plasticity_target_srank";

constexpr const char* kRfd0121ValueFeature = "value_feature";

// RFD0121の入力・公開識別子を、同じ意味の検査で共有する。
constexpr const char* kRfd0121Eval = "eval";

constexpr const char* kRfd0121ExtrinsicGradNorm = "extrinsic_grad_norm";

constexpr const char* kRfd0121ExtrinsicHead = "extrinsic";

constexpr const char* kRfd0121ExtrinsicLoss = "extrinsic_loss";

constexpr const char* kRfd0121ExtrinsicQ = "extrinsic_q";

constexpr const char* kRfd0121ExtrinsicQDist = "extrinsic_q_dist";

constexpr const char* kRfd0121ExtrinsicQGap = "extrinsic_q_gap";

constexpr const char* kRfd0121ExtrinsicQSessionGapMean = "extrinsic_q_session_gap_mean";

constexpr const char* kRfd0121Features = "features";

constexpr const char* kRfd0121FixedTauMode = "fixed";

constexpr const char* kRfd0121HeadExtrinsicLinearBias = "head.extrinsic.linear.bias";

constexpr const char* kRfd0121HeadIntrinsicLinearBias = "head.intrinsic.linear.bias";

constexpr const char* kRfd0121Inf = "inf";

constexpr const char* kRfd0121IntrinsicAdvFeature = "intrinsic_adv_feature";

constexpr const char* kRfd0121IntrinsicFeatures = "intrinsic_features";

constexpr const char* kRfd0121IntrinsicGradNorm = "intrinsic_grad_norm";

constexpr const char* kRfd0121IntrinsicLoss = "intrinsic_loss";

constexpr const char* kRfd0121IntrinsicQ = "intrinsic_q";

constexpr const char* kRfd0121IntrinsicQDist = "intrinsic_q_dist";

constexpr const char* kRfd0121IntrinsicValueFeature = "intrinsic_value_feature";

constexpr const char* kRfd0121Invalid = "invalid";

constexpr const char* kRfd0121Linear = "linear";

constexpr const char* kRfd0121Nan = "nan";

constexpr const char* kRfd0121Off = "off";

constexpr const char* kRfd0121OnlineLogPolicyMode = "online";

constexpr const char* kRfd0121OnlineNetworkMode = "online";

constexpr const char* kRfd0121Q = "q";

constexpr const char* kRfd0121QDist = "q_dist";

constexpr const char* kRfd0121ReplayFitUniformTdMean = "replay_fit_uniform_td_mean";

constexpr const char* kRfd0121SharedBranchKey = "shared";

constexpr const char* kRfd0121TargetLogPolicyMode = "target";

static constexpr anet::seed_t kTestAgentSeed = 6000;

static constexpr const char* kQValueKey = "q";

static constexpr const char* kQDistributionKey = "q_dist";

static constexpr const char* kLossMetricKey = "loss";

static constexpr const char* kPolicyForwardKey = "policy-net.forward";

static constexpr const char* kPolicyForwardDistributionKey = "policy-net.forward.dist";

static constexpr const char* kNoisyQAbsDiffKey = "noisy_q_abs_diff";

// テストの既定経路も、製品と同じ統合factoryで生成する。
std::shared_ptr<anet::nn::NetworkForwardExecutor> MakeDefaultExecutor(
    const std::shared_ptr<anet::nn::Network>& network)
{
    return network->CreateForwardExecutor(anet::nn::NetworkForwardConfig::GetDefault(),
        anet::nn::NetworkForwardClock::kTransientRows, 0);
}

// テスト用の生データから選択項目だけを取り出し、公開 constructor へ渡す。
anet::ConfigData SelectForwardConfigData(const anet::ConfigData& data, const std::string& prefix)
{
    anet::ConfigData selected;
    for (const auto& [key, value] : data.Map()) {
        if (key.starts_with(prefix + ".")) selected.Set(key.substr(prefix.size() + 1), value);
    }
    return selected;
}

anet::rl::dqn::LearnerConfig WithDefaultLearnerForwardConfig()
{
    // 名前付きカタログを使わない既存の直接構築 fixture は、既定動作を明示する。
    anet::rl::dqn::LearnerConfig config;
    return config;
}

anet::rl::dqn::LearnerForwardResources SelectLearnerForwardResources(const anet::ConfigData& data)
{
    const anet::rl::dqn::DefaultDQNAgentConfig config(data);
    return {
        .current = anet::nn::NetworkForwardConfig("DefaultDQNAgent.nn_forward.[" + config.learner.current.nn_forward_key + "]",
            config.nn_forward.at(config.learner.current.nn_forward_key)),
        .target = anet::nn::NetworkForwardConfig("DefaultDQNAgent.nn_forward.[" + config.learner.target.nn_forward_key + "]",
            config.nn_forward.at(config.learner.target.nn_forward_key)),
    };
}

anet::ConfigData WithDefaultForwardConfig(anet::ConfigData data = {})
{
    anet::test::RegisterStandardNetworkModules();
    // 既存テストの実行目的を明示し、新しい必須参照を fixture 側で選択する。
    data.Set("DefaultDQNAgent.nn_forward.[test_default].noisy.mode", "mu_only");
    if (!data.Has("DefaultDQNAgent.learner.current.nn_forward_key"))
        data.Set("DefaultDQNAgent.learner.current.nn_forward_key", "test_default");
    if (!data.Has("DefaultDQNAgent.learner.target.nn_forward_key"))
        data.Set("DefaultDQNAgent.learner.target.nn_forward_key", "test_default");
    for (const auto& [key, entry] : data.MakeSubConfigData("DefaultDQNAgent.actor")) {
        const auto reference = "DefaultDQNAgent.actor.[" + key + "].nn_forward_key";
        if (!data.Has(reference)) data.Set(reference, "test_default");
    }
    return data;
}

namespace rl = anet::rl;

namespace dqn = anet::rl::dqn;

size_t CountProfileOperator(const std::function<void()>& operation, const std::string& operator_name)
{
    std::ostringstream output;
    {
        torch::autograd::profiler::RecordProfile profile(output);
        operation();
    }

    const std::string trace = output.str();
    const std::string needle = "\"name\": \"" + operator_name + "\"";
    size_t count = 0;
    for (size_t offset = 0; (offset = trace.find(needle, offset)) != std::string::npos; offset += needle.size()) {
        ++count;
    }
    return count;
}

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

    torch::Tensor Forward(torch::Tensor input, const anet::nn::NetworkForwardContext& context) override
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

    static torch::Tensor BuildLaneTensor(int64_t lanes, float first, float last,
        const std::string& scale, const torch::Device& device)
    {
        const dqn::ActionPolicyConfig::LadderConfig config{
            .first = {.value = first, .min_value = 0.0f, .max_value = 1.0f},
            .last = {.value = last, .min_value = 0.0f, .max_value = 1.0f}};
        return dqn::SpatialLadderSchedule(config, scale, lanes, device,
            anet::ConfigFullKeyTable::FormatKeyNotReadFromConfig("spatial.scale")).Tensor();
    }
    static torch::Tensor BuildTensor(int64_t lanes, float first, float last,
        const std::string& scale, const torch::Device& device)
    {
        return BuildLaneTensor(lanes, first, last, scale, device).flip({0});
    }

    std::shared_ptr<dqn::DQNActionInfo> SelectAction(const anet::TensorDict&, bool, const dqn::DqnForwardExecutor&,
        std::shared_ptr<anet::RandomGenerator>, const anet::nn::NetworkForwardCallInfo&, anet::ForwardCapture*,
        const dqn::DiagnosticActionEvaluation*, const torch::Tensor&) const override
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

    anet::TensorDict Forward(const anet::TensorDict& feature_dict, const anet::nn::NetworkForwardContext& context) override
    {
        return anet::TensorDict{ { "q", linear_->forward(feature_dict.At(kFeatureKey)) } };
    }

    std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key) override
    {
        if (key != "forward" && key != "forward.q") return std::nullopt;

        return [this](const anet::TensorDict& feature_dict) {
            torch::NoGradGuard no_grad;
            return Forward(feature_dict, anet::nn::NetworkForwardContext::GetDefault());
        };
    }

private:
    torch::nn::Linear linear_{ nullptr };
};

std::shared_ptr<anet::nn::Network> MakeLinearNetwork(int64_t n_actions = 1)
{
    anet::TensorSpec vector_spec;
    vector_spec.type = anet::SpaceType::Vector;
    vector_spec.shape = { 2 };
    vector_spec.dtype = torch::kFloat32;

    anet::TensorSpecMap input_specs;
    input_specs[kVectorKey] = vector_spec;

    anet::nn::NetworkConfig network_config;
    network_config.output_keys[kFeatureKey] = kVectorKey;

    auto head = std::make_shared<TestLinearHead>(2, n_actions);

    return anet::test::MakeTestNetwork(
        network_config,
        input_specs,
        nullptr,
        head);
}

std::shared_ptr<anet::nn::Network> MakeSpectralLinearNetwork(anet::seed_t seed = 65065)
{
    anet::test::RegisterStandardNetworkModules();
    anet::ConfigData config_data;
    config_data.Set("net.block.[Linear].type", "Linear");
    config_data.Set("net.block.[Linear].linear.out_features", 2);
    config_data.Set("net.block.[Linear].weight_norm.mode", "spectral");
    config_data.Set("net.branch.[feature].bind", kVectorKey);
    config_data.Set("net.branch.[feature].structure", "Linear");
    config_data.Set("net.body.output.[feature]", "feature");
    const anet::TensorSpec vector_spec{
        .type = anet::SpaceType::Vector,
        .shape = { 2 },
        .dtype = torch::kFloat32,
    };
    return anet::nn::NetworkBuilder::BuildNetwork(
        anet::nn::NetworkConfig(config_data),
        anet::TensorSpecMap{ { kVectorKey, vector_spec } },
        nullptr,
        seed,
        torch::Device(torch::kCPU));
}

std::shared_ptr<anet::nn::Network> MakeAutocastProbeNetwork(
    const std::shared_ptr<AutocastProbeState>& probe_state,
    torch::Device device,
    int64_t n_actions = 1)
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
    const anet::test::TestNetworkBranch branch{
        kFeatureKey, std::vector<std::vector<std::string>>{ { kVectorKey } }, 1, "probe", probe,
    };
    auto head = std::make_shared<TestLinearHead>(2, n_actions);

    auto network = anet::test::MakeTestNetwork(
        network_config,
        input_specs,
        nullptr,
        head, { branch });
    network->to(device);
    return network;
}

class TestNetworkModel final : public dqn::NetworkModel {
public:
    explicit TestNetworkModel(
        bool distributional = false,
        int64_t n_actions = 1,
        dqn::NetworkModelConfig config = {})
        : dqn::NetworkModel(
            config,
            MakeLinearNetwork(n_actions),
            MakeLinearNetwork(n_actions),
            n_actions,
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
        torch::Device device,
        int64_t n_actions = 1)
        : dqn::NetworkModel(
            dqn::NetworkModelConfig{},
            MakeAutocastProbeNetwork(probe_state, device, n_actions),
            MakeAutocastProbeNetwork(probe_state, device, n_actions),
            n_actions,
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
        const rl::BatchEnvSpec& batch_env_spec,
        const rl::EnvSpec& env_spec,
        const dqn::LearnerForwardResources& forward_resources,
        std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe_config = nullptr)
        : dqn::Learner(
            config,
            model,
            batch_env_spec,
            env_spec,
            torch::Device(torch::kCPU),
            123,
            kTestAgentSeed,
            std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}),
            forward_resources,
            std::nullopt,
            456,
            std::move(noisy_probe_config))
    {
    }

    using dqn::Learner::MakeBatchUpdateResult;
    using dqn::Learner::Optimize;
    using dqn::Learner::ApplyPerPriorityUpdate;
    using dqn::Learner::PreparePerPriorityUpdate;
    using dqn::Learner::TransformH;
    using dqn::Learner::TransformHInv;

    // 実際の構築・読み込み経路を使い、テスト内だけで optimizer の状態を観測する。
    void SetupAdamW() { SetupOptimizer(); }
    torch::optim::Optimizer& Optimizer() { return *optimizer_; }
    torch::optim::Optimizer& SkipPrimaryOptimizer() { return *skip_primary_optimizer_; }

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

    void EnablePlasticityForward(bool include_target)
    {
        run_plasticity_forward_ = true;
        run_plasticity_target_forward_ = include_target;
    }

    void MutateOnlineParametersDuringUpdate(float delta)
    {
        parameter_mutation_delta_ = delta;
    }

    void EnablePolicyChurnOptimizerUpdate()
    {
        run_policy_churn_optimizer_update_ = true;
    }

    void SetOptimizerElementLossScale(torch::Tensor scale)
    {
        optimizer_element_loss_scale_ = std::move(scale);
    }

    std::shared_ptr<anet::rl::dqn::BatchUpdateResult> UpdateFromSamples(
        const anet::rl::ExperienceSamples& samples) override
    {
        const auto forward_info = MakeCurrentForwardInfo();
        if (run_policy_churn_optimizer_update_) {
            // action 1 の Q を上げる決定論的な 1 update を作り、公開結果から churn を観測する。
            auto q_values = model_.ForwardOnlineWithTrain(samples.obs, *current_forward_, forward_info, OnlineBranchCapture()).At("q");
            const int64_t optimized_action = n_actions_ > 1 ? 1 : 0;
            auto element_loss = -q_values.select(1, optimized_action);
            // 行ごとの係数で小batch勾配を制御し、未指定の既存テストは元の損失を使う。
            if (optimizer_element_loss_scale_.defined()) element_loss = element_loss * optimizer_element_loss_scale_;
            auto loss = element_loss.mean();
            auto opt_result = Optimize(loss, element_loss);
            dqn::PerPriorityUpdateInfo per_info;
            const auto batch_size = samples.actions.size(0);
            return MakeBatchUpdateResult(
                loss,
                torch::zeros({ batch_size }),
                opt_result,
                std::get<0>(q_values.detach().max(1)),
                q_values.detach().select(1, 0),
                per_info);
        }
        if (run_plasticity_forward_) {
            model_.ForwardOnlineWithTrain(samples.obs, *current_forward_, forward_info, OnlineBranchCapture());
            if (run_plasticity_target_forward_) target_forwards_.target->Forward(samples.next_state.next_obs, forward_info, TargetBranchCapture());
            if (parameter_mutation_delta_.has_value()) {
                torch::NoGradGuard no_grad;
                for (auto& parameter : model_.GetOnlineParameters()) {
                    parameter.add_(*parameter_mutation_delta_);
                }
            }
            dqn::OptimizerStepResult opt_result;
            dqn::PerPriorityUpdateInfo per_info;
            const auto batch_size = samples.actions.size(0);
            return MakeBatchUpdateResult(
                torch::tensor(0.0f),
                torch::zeros({ batch_size }),
                opt_result,
                torch::zeros({ batch_size }),
                torch::zeros({ batch_size }),
                per_info);
        }
        return nullptr;
    }

private:
    bool run_plasticity_forward_ = false;
    bool run_plasticity_target_forward_ = false;
    bool run_policy_churn_optimizer_update_ = false;
    torch::Tensor optimizer_element_loss_scale_;
    std::optional<float> parameter_mutation_delta_;
};

class RecordingReplayBuffer final : public rl::ReplayBuffer {
public:
    rl::SamplingHistoryProbeResult ProbeSamplingHistory(
        const rl::SamplingHistoryProbeRequest& request, anet::RandomGenerator*, std::span<const float> gammas) const override
    {
        ++history_probe_count;
        last_history_request = request;
        auto result = history_result;
        if (!request.unsampled_batch_size) result.unsampled.reset();
        if (!request.sampled_batch_size) result.sampled.reset();
        return result;
    }
    mutable std::vector<float> sampled_betas;
    std::vector<float> pushed_gammas;
    mutable std::vector<float> sampled_gammas;
    mutable int history_probe_count = 0;
    mutable rl::SamplingHistoryProbeRequest last_history_request;
    rl::SamplingHistoryProbeResult history_result;

    void Push(const rl::BatchExperience&, std::span<const float> gammas) override
    {
        pushed_gammas.push_back(gammas.front());
        ++push_count;
    }

    void Sample(rl::ExperienceSamples& out_samples, int64_t minibatch_size, float beta, std::span<const float> gammas) const override
    {
        sampled_betas.push_back(beta);
        sampled_gammas.push_back(gammas.front());
        ++sample_count;

        out_samples.obs = anet::TensorDict{
            { kVectorKey, torch::full(
                { minibatch_size, 2 }, vector_value,
                torch::TensorOptions().dtype(torch::kFloat32)) },
        };
        out_samples.actions = torch::zeros({ minibatch_size }, torch::TensorOptions().dtype(torch::kInt64));
        out_samples.target_returns = torch::zeros({ minibatch_size }, torch::TensorOptions().dtype(torch::kFloat32));
        out_samples.next_state.next_obs = anet::TensorDict{
            { kVectorKey, torch::full(
                { minibatch_size, 2 }, vector_value,
                torch::TensorOptions().dtype(torch::kFloat32)) },
        };
        out_samples.next_state.terminals = torch::zeros({ minibatch_size }, torch::TensorOptions().dtype(torch::kBool));
        out_samples.n_steps = torch::ones({ minibatch_size }, torch::TensorOptions().dtype(torch::kInt64));
        out_samples.replay_item_keys = torch::arange(minibatch_size, torch::TensorOptions().dtype(torch::kInt64));
        out_samples.is_weights = torch::ones({ minibatch_size }, torch::TensorOptions().dtype(torch::kFloat32));
        out_samples.per_priority_sources = torch::zeros({ minibatch_size }, torch::TensorOptions().dtype(torch::kInt8));
    }

    bool SampleUniqueUniform(
        rl::ExperienceSamples& out_samples, int64_t batch_size, anet::RandomGenerator&, std::span<const float> gammas) const override
    {
        ++unique_sample_count;
        if (!unique_sample_enabled) return false;
        Sample(out_samples, batch_size, 0.0f, gammas);
        return true;
    }

    const anet::TensorSpecMap& GetSampleObsSpecs() const override { return sample_obs_specs; }

    anet::TensorSpecMap sample_obs_specs;

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
    mutable int unique_sample_count = 0;
    bool unique_sample_enabled = false;
    float vector_value = 0.0f;
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
    config.block_configs[kFixtureFlatten].type = kFixtureFlatten;
    config.branches[kFixtureMainBranch] = {.name = kFixtureMainBranch, .bind_terms = {{kVectorKey}}, .structure_str = kFixtureFlatten};
    config.output_keys[anet::nn::kKey_DefaultOutput] = kFixtureMainBranch;
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
    config_data.Set("DefaultDQNAgent.head.use_dueling", false);


    config_data.Set("DefaultDQNAgent.actor.[train].policy.epsilon", 0.0f);
    config_data.Set("DefaultDQNAgent.actor.[train].policy.epsilon.start", 0.0f);
    config_data.Set("DefaultDQNAgent.actor.[train].policy.epsilon.end", 0.0f);
    config_data.Set("DefaultDQNAgent.actor.[train].policy.tau_rule.num_taus", 3);
    config_data.Set("DefaultDQNAgent.actor.[train].policy.tau_rule.sample_mode", "fixed");
    config_data.Set("DefaultDQNAgent.learner.replay_capacity", 16);
    config_data.Set("DefaultDQNAgent.learner.replay_batch_size", 2);
    config_data.Set("DefaultDQNAgent.learner.use_fused_optimizer", false);

    config_data.Set("net.block.[CosEmb].type", "CosineEmbedding");
    config_data.Set("net.block.[CosEmb].cos.num_basis", 4);
    config_data.Set("net.block.[TauProj].type", "Linear");
    config_data.Set("net.block.[TauProj].linear.out_features", 4);
    config_data.Set("net.block.[Flatten].type", kFixtureFlatten);
    config_data.Set("DefaultDQNAgent.net.branch.[main].bind", kVectorKey);
    config_data.Set("DefaultDQNAgent.net.branch.[main].structure", kFixtureFlatten);
    config_data.Set("DefaultDQNAgent.net.branch.[tau_embedding].bind", anet::rl::dqn::kKey_Taus);
    config_data.Set("DefaultDQNAgent.net.branch.[tau_embedding].structure", "CosEmb > TauProj");
    config_data.Set("DefaultDQNAgent.net.branch.[fusion].bind", "main * tau_embedding");
    config_data.Set("DefaultDQNAgent.net.body.output.[features]", "fusion");
    return config_data;
}

void PrepareNativeStackNetworkTest()
{
    // mkldnnはmain_test.cppのmainが全テストの前に切るので、ここでは切らない（RFD0124）。
    anet::test::RegisterStandardNetworkModules();
}

anet::ConfigData MakeNativeStackConfigData()
{
    anet::ConfigData config_data;
    config_data.Set("DefaultDQNAgent.quantile_mode", "none");
    config_data.Set("DefaultDQNAgent.head.use_dueling", false);
    config_data.Set("DefaultDQNAgent.perception.structure", "stack");
    config_data.Set("DefaultDQNAgent.perception.block.[stack].type", "FrameStack");
    config_data.Set("DefaultDQNAgent.frame_stack.count", 4);

    config_data.Set("DefaultDQNAgent.learner.replay_capacity", 16);
    config_data.Set("DefaultDQNAgent.learner.replay_batch_size", 2);
    config_data.Set("DefaultDQNAgent.learner.use_fused_optimizer", false);
    config_data.Set("net.block.[Flatten].type", "Flatten");
    config_data.Set("net.block.[Linear].type", "Linear");
    config_data.Set("net.block.[Linear].linear.out_features", 8);
    config_data.Set("net.body.output.[features]", "main_feature");
    config_data.Set("DefaultDQNAgent.actor.[train].network", "online");
    return config_data;
}

rl::EnvSpec MakeNativeStackEnvSpec(anet::TensorSpecMap obs_spec)
{
    rl::EnvSpec env_spec;
    env_spec.state_spec.obs_spec = std::move(obs_spec);
    env_spec.action_spec.is_discrete = true;
    env_spec.action_spec.value_labels = { "a0", "a1" };
    env_spec.reward_range = { -1.0f, 1.0f };
    return env_spec;
}

std::shared_ptr<dqn::DQNActionInfo> ActWithNativeStack(
    const anet::ConfigData& config_data,
    const rl::EnvSpec& env_spec,
    anet::TensorDict raw_obs, bool nn_trace = true)
{
    const rl::BatchEnvSpec batch_env_spec{ 2, 1 };
    auto agent = std::make_shared<dqn::DefaultDQNAgent>(
        dqn::DefaultDQNAgentConfig(WithDefaultForwardConfig(config_data)),
        anet::nn::NetworkConfig(config_data),
        batch_env_spec,
        env_spec,
        torch::Device(torch::kCPU),
        123);
    auto actor = agent->CreateActor(rl::ActorRequest{.batch_env_spec = batch_env_spec, .env_spec = env_spec, .device = torch::Device(torch::kCPU), .seed = 123, .actor_key = "train"});

    const auto flags = torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kBool));
    rl::BatchState state(std::move(raw_obs), flags, flags, flags);
    if (!nn_trace) {
        const auto implicit = actor->MakeAction(rl::StepCounts{}, state);
        for (const auto& [key, value] : implicit->GetAuxData()) CHECK_FALSE(key.starts_with(anet::nn::kNnTracePrefix));
    }
    return std::dynamic_pointer_cast<dqn::DQNActionInfo>(
        actor->MakeAction(rl::StepCounts{}, state, {.nn_trace = nn_trace}));
}

dqn::DefaultDQNAgentConfig MakeDeviceForwardDefaultDqnConfig()
{
    anet::ConfigData data;
    for (const auto* key : { "train", "eval", "eval_target" }) {
        data.Set(std::string("DefaultDQNAgent.actor.[") + key + "].network", "online");
    }
    dqn::DefaultDQNAgentConfig config(WithDefaultForwardConfig(data));
    config.quantile_mode = "none";
    config.head.use_dueling = false;


    config.learner.replay_capacity = 16;
    config.learner.replay_batch_size = 2;
    config.learner.use_fused_optimizer = false;
    config.actor["eval_target"].network = "target";
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
    config.actor["train"] = dqn::DQNActorConfig{};
    config.actor["eval"] = dqn::DQNActorConfig{};
    config.actor["eval_target"] = dqn::DQNActorConfig{.network = "target"};
    return config;
}

class DeviceOnlyAgent final : public rl::AgentBase {
public:
    explicit DeviceOnlyAgent(torch::Device device)
        : rl::AgentBase(device, rl::BatchEnvSpec{ 1, 1 }, MakeLearnerEnvSpec(), 123)
    {
    }

    std::shared_ptr<rl::Actor> CreateActor(const rl::ActorRequest& request) const override
    {
        const auto& batch_env_spec = request.batch_env_spec;
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
    dqn::LearnerConfig config = WithDefaultLearnerForwardConfig();
    config.use_rb_prefetch = true;
    config.use_per = true;
    config.use_n_step = true;
    config.n_step = 3;
    config.replay_batch_size = 8;
    config.replay_capacity = 64;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = -1.0f;
    config.per_beta.start = 0.4f;
    config.per_beta.end = 1.0f;
    config.per_beta.type = "linear";
    config.per_beta.steps = 128;
    config.use_per_prio_clip = true;
    config.per_prio_clip_value = 10.0f;
    return config;
}

class CapturingLearner final : public dqn::Learner {
public:
    CapturingLearner(
        const dqn::LearnerConfig& config,
        dqn::NetworkModel& model,
        const rl::BatchEnvSpec& batch_env_spec,
        const rl::EnvSpec& env_spec,
        anet::seed_t replay_seed,
        std::shared_ptr<DeterminismJitterSchedule> jitter)
        : dqn::Learner(
            config,
            model,
            batch_env_spec,
            env_spec,
            torch::Device(torch::kCPU),
            replay_seed,
            kTestAgentSeed,
            std::make_shared<dqn::EpsilonGreedyActionPolicy>(dqn::ActionPolicyConfig{}),
            dqn::LearnerForwardResources{},
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

    void UseReplayBuffer(std::shared_ptr<rl::ReplayBuffer> replay) { replay_buffer_ = std::move(replay); }

    std::vector<DeterminismTraceEntry> Trace() const
    {
        std::lock_guard<std::mutex> lock(trace_mutex_);
        return trace_;
    }

protected:
    std::shared_ptr<anet::rl::dqn::BatchUpdateResult> UpdateFromSamples(
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
    auto env_spec = MakeLearnerEnvSpec();
    rl::BatchEnvSpec batch_env_spec{ kNumEnv, 1 };
    CapturingLearner learner(config, model, batch_env_spec, env_spec, replay_seed, jitter);

    for (int64_t step = 0; step < num_steps; ++step) {
        rl::StepCounts counts;
        counts.train_step = step;
        counts.exp_step = step * kNumEnv;
        counts.learn_step = learner.GetLearnStep();
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

    std::shared_ptr<rl::BatchActionInfo> MakeAction(const rl::StepCounts&, const rl::BatchState&, const anet::rl::MakeActionOptions& options = {}) const override
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
        , learner_(std::make_shared<CapturingLearner>(
            config,
            *model_,
            batch_env_spec,
            env_spec,
            replay_seed,
            std::move(jitter)))
    {
    }

    std::shared_ptr<rl::Actor> CreateActor(const rl::ActorRequest& request) const override
    {
        const auto& batch_env_spec = request.batch_env_spec;
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
    std::shared_ptr<rl::TrainRunner> runner = std::make_shared<rl::PipelineTrainRunner>(env, agent, notifier, rl::ActorRequest{.batch_env_spec = env->GetBatchSpec(), .env_spec = env->GetSpec(), .device = agent->GetDevice(), .seed = 123, .actor_key = "train"});

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


    return anet::test::MakeTestNetwork(
        network_config,
        input_specs,
        nullptr,
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
    explicit TauEchoHead(std::shared_ptr<TauEchoState> state, bool skip = false)
        : state_(std::move(state)), skip_(skip)
    {
    }

    anet::TensorDict Forward(const anet::TensorDict& feature_dict, const anet::nn::NetworkForwardContext& context) override
    {
        const auto taus = feature_dict.At(anet::rl::dqn::kKey_Taus);
        ++state_->forward_count;
        state_->last_taus = taus.detach().clone();

        auto q_dist = torch::stack({ taus, skip_ ? torch::full_like(taus, 0.7f) : 1.0f - taus }, 1);
        return anet::TensorDict{
            { skip_ ? "skip_q" : "q", q_dist.mean(2) },
            { skip_ ? "skip_q_dist" : "q_dist", q_dist },
        };
    }

    std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string&) override
    {
        return std::nullopt;
    }

private:
    std::shared_ptr<TauEchoState> state_;
    bool skip_;
};

std::shared_ptr<anet::nn::Network> MakeTauEchoNetwork(
    int64_t nominal_num_taus,
    const std::shared_ptr<TauEchoState>& state, bool skip = false)
{
    anet::TensorSpec vector_spec;
    vector_spec.shape = { 1 };
    vector_spec.dtype = torch::kFloat32;

    anet::TensorSpec tau_spec;
    tau_spec.shape = { nominal_num_taus };
    tau_spec.dtype = torch::kFloat32;

    anet::TensorSpecMap input_specs{
        { kVectorKey, vector_spec },
        { anet::rl::dqn::kKey_Taus, tau_spec },
    };
    anet::nn::NetworkConfig network_config;
    network_config.output_keys[anet::rl::dqn::kKey_Taus] = anet::rl::dqn::kKey_Taus;
    return anet::test::MakeTestNetwork(
        network_config,
        input_specs,
        nullptr,
        std::make_shared<TauEchoHead>(state, skip));
}

anet::TensorDict MakeTauEchoObservation()
{
    return anet::TensorDict{
        { kVectorKey, torch::zeros({ 2, 1 }) },
    };
}

anet::ConfigData MakeRfd0104ConfigData()
{
    auto data = MakeDeviceForwardDefaultDqnConfig().GetScopedConfigData();
    data.Set("DefaultDQNAgent.quantile_mode", "none");
    data.Set("DefaultDQNAgent.head.use_dueling", false);

    data.Set("DefaultDQNAgent.skip.enabled", true);
    data.Set("DefaultDQNAgent.skip.max_length", 3);
    data.Set("DefaultDQNAgent.skip.net.branch.[skip_feature].bind", "features");
    data.Set("DefaultDQNAgent.skip.net.body.output.[features]", "skip_feature");
    data.Set("DefaultDQNAgent.actor.[train].skip.max_length_mode", "single");
    data.Set("DefaultDQNAgent.actor.[eval].skip.max_length_mode", "single");
    data.Set("DefaultDQNAgent.actor.[eval_target].skip.max_length_mode", "single");
    data.Set("DefaultDQNAgent.actor.[train].policy.epsilon", 1.0f);
    data.Set("DefaultDQNAgent.learner.use_amp", false);
    data.Set("DefaultDQNAgent.learner.use_per", false);
    data.Set("DefaultDQNAgent.learner.use_n_step", false);
    data.Set("DefaultDQNAgent.learner.use_fused_optimizer", false);
    data.Set("DefaultDQNAgent.learner.use_rb_prefetch", false);
    data.Set("DefaultDQNAgent.learner.replay_capacity", 32);
    data.Set("DefaultDQNAgent.learner.replay_batch_size", 2);
    data.Set("DefaultDQNAgent.learner.update_warmup_steps", 0);
    data.Set("DefaultDQNAgent.learner.update_interval", 1);
    data.Set("DefaultDQNAgent.learner.replay_ratio", -1.0f);
    return WithDefaultForwardConfig(data);
}

constexpr const char* kRfd0106QValuesKey = "q_values";

anet::ConfigData MakeRfd0106IqnConfigData()
{
    auto data = MakeIqnTracerConfigData();
    data.Set("DefaultDQNAgent.skip.enabled", true);
    data.Set("DefaultDQNAgent.skip.max_length", 3);
    data.Set("DefaultDQNAgent.learner.replay_capacity", 128);
    data.Set("DefaultDQNAgent.net.body.output.[skip_context]", "fusion");
    data.Set("DefaultDQNAgent.skip.net.branch.[skip_feature].bind", "skip_context");
    data.Set("DefaultDQNAgent.skip.net.body.output.[features]", "skip_feature");
    data.Set("DefaultDQNAgent.actor.[train].policy.policy_type", "EpsilonGreedy");
    data.Set("DefaultDQNAgent.actor.[train].policy.tau_rule.num_taus", 8);
    data.Set("DefaultDQNAgent.actor.[train].skip.max_length_mode", "single");
    return WithDefaultForwardConfig(data);
}

// 行動選択の実装を通さず、同じ行動と公開特徴量を skip の入力へ渡す。
class FixedSkipInputPolicy final : public dqn::ActionPolicy {
public:
    explicit FixedSkipInputPolicy(torch::Tensor features)
        : ActionPolicy(dqn::ActionPolicyConfig{}), features_(std::move(features)) {}
    std::shared_ptr<dqn::DQNActionInfo> SelectAction(const anet::TensorDict& obs, bool,
        const dqn::DqnForwardExecutor&, std::shared_ptr<anet::RandomGenerator>,
        const anet::nn::NetworkForwardCallInfo&, anet::ForwardCapture*,
        const dqn::DiagnosticActionEvaluation*, const torch::Tensor&) const override
    {
        auto result = std::make_shared<dqn::DQNActionInfo>(torch::zeros({obs.Size(0)}, torch::kLong));
        result->forward_output.Set(anet::nn::kKey_DefaultOutput, features_);
        return result;
    }
private:
    torch::Tensor features_;
};

struct ForwardContextRecord {
    std::string key;
    uint64_t clock;
    const anet::nn::NetworkForwardState* state;
    bool training;
    torch::Tensor row_mapping;
    torch::Tensor features;
};

class ForwardContextRecordingHead final : public anet::nn::NetworkHead {
public:
    ForwardContextRecordingHead(std::shared_ptr<anet::nn::NetworkHead> inner,
        std::shared_ptr<std::vector<ForwardContextRecord>> records)
        : inner_(register_module("projection", std::move(inner))), records_(std::move(records)) {}
    anet::TensorDict Forward(const anet::TensorDict& input, const anet::nn::NetworkForwardContext& context) override
    { return Run(input, context, "default", 1.0f, *inner_); }
    std::unique_ptr<anet::nn::NetworkHeadExecutor> CreateExecutor(
        const anet::nn::NetworkForwardConfig& config, anet::nn::NetworkForwardClock clock) override
    { return std::make_unique<Prepared>(*this, config, clock); }
    std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key) override
    { return inner_->GetTensorDictFunction(key); }
    void ValidateForwardReuse(const anet::nn::NetworkForwardConfig& source, const anet::nn::NetworkForwardConfig& requested) const override
    { inner_->ValidateForwardReuse(source, requested); }
private:
    class Prepared final : public anet::nn::NetworkHeadExecutor {
    public:
        Prepared(ForwardContextRecordingHead& owner, const anet::nn::NetworkForwardConfig& config,
            anet::nn::NetworkForwardClock clock)
            : owner_(owner), key_(config.key),
              scale_(config.IsDefault() ? 1.0f : config.GetConfigData().Get<float>("probe_test.scale", 1.0f)),
              owned_(owner.inner_->CreateExecutor(config, clock)), inner_(owned_ ? owned_.get() : owner.inner_.get()) {}
        anet::TensorDict Forward(const anet::TensorDict& input, const anet::nn::NetworkForwardContext& context) override
        { return owner_.Run(input, context, key_, scale_, *inner_); }
    private:
        ForwardContextRecordingHead& owner_;
        const std::string key_;
        const float scale_;
        const std::unique_ptr<anet::nn::NetworkHeadExecutor> owned_;
        anet::nn::NetworkHeadExecutor* const inner_;
    };
    anet::TensorDict Run(const anet::TensorDict& input, const anet::nn::NetworkForwardContext& context,
        const std::string& key, float scale, anet::nn::NetworkHeadExecutor& inner)
    {
        // 設定は準備側に閉じ、公開の Head 境界で用途・箱・時計だけを観測する。
        records_->push_back({key, context.GetClock(), context.GetState(), is_training(),
            context.GetRowMapping() ? context.GetRowMapping()->GetIndices() : torch::Tensor(),
            input.At(anet::nn::kKey_DefaultOutput).detach().clone()});
        auto output = inner.Forward(input, context);
        output.Set("q", output.At("q") * scale);
        return output;
    }
    std::shared_ptr<anet::nn::NetworkHead> inner_;
    std::shared_ptr<std::vector<ForwardContextRecord>> records_;
};

class ForwardContextRecordingHeadFactory final : public anet::nn::NetworkHeadFactory {
public:
    explicit ForwardContextRecordingHeadFactory(std::shared_ptr<std::vector<ForwardContextRecord>> records)
        : records_(std::move(records)), inner_(1, anet::nn::WeightInitConfig{}) {}
    std::shared_ptr<anet::nn::NetworkHead> CreateHead(const anet::TensorDict& input) const override
    { return std::make_shared<ForwardContextRecordingHead>(inner_.CreateHead(input), records_); }
private:
    std::shared_ptr<std::vector<ForwardContextRecord>> records_;
    dqn::LinearHeadFactory inner_;
};

// 3つのoptimizer契約は、同じ公開設定と実際のTD更新経路で確認する。
anet::ConfigData MakeNoisyOptimizerTestData()
{
    auto data = WithDefaultForwardConfig();
    data.Set("net.block.[Linear].type", "Linear");
    data.Set("net.block.[Linear].linear.out_features", 2);
    data.Set("net.block.[Linear].noisy.enabled", true);
    data.Set("net.branch.[feature].bind", kVectorKey);
    data.Set("net.branch.[feature].structure", "Linear");
    data.Set("net.body.output.[features]", "feature");
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.mode", "sample");
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.hold", "call");
    data.Set("DefaultDQNAgent.nn_forward.[sample].noisy.share", "batch");
    data.Set("DefaultDQNAgent.learner.current.nn_forward_key", "sample");
    return data;
}

dqn::LearnerConfig MakeNoisyOptimizerLearnerConfig(const anet::ConfigData& data)
{
    auto config = dqn::DefaultDQNAgentConfig(data).learner;
    // 少量の遷移から1サンプルを1回更新し、gamma=0でbootstrapの影響を除く。
    config.replay_batch_size = 1;
    config.replay_capacity = 8;
    config.update_warmup_steps = 0;
    config.update_interval = 1;
    config.replay_ratio = -1.0f;
    config.gamma.value = 0.0f;
    config.use_n_step = false;
    config.use_per = false;
    config.use_rb_prefetch = false;
    config.use_amp = false;
    config.use_grad_clip = false;
    config.use_td_clip = false;
    config.use_tbo = false;
    return config;
}

rl::BatchExperience MakeNoisyOptimizerExperience(rl::step_t step, float reward)
{
    auto experience = MakeDeterminismExperience(step, 1);
    experience.state.obs = anet::TensorDict{{kVectorKey, torch::ones({1, 2})}};
    experience.next_state.obs = experience.state.obs.Clone();
    experience.reward = torch::full({1}, reward);
    return experience;
}

std::shared_ptr<anet::nn::Network> MakeAttentionProbeTestNetwork(bool transformer = true)
{
    // productionのbuilderを使い、vector観測から二つのTransformerへ接続する。
    anet::test::RegisterStandardNetworkModules();
    anet::ConfigData data;
    data.Set("net.block.[Expand].type", "Linear");
    data.Set("net.block.[Expand].linear.out_features", 4);
    data.Set("net.block.[Tokens].type", "Reshape");
    data.Set("net.block.[Tokens].reshape.dims", "2 2");
    data.Set("net.block.[TF].type", "TransformerEncoder");
    data.Set("net.block.[TF].tf.d_model", 2); data.Set("net.block.[TF].tf.nhead", 1);
    data.Set("net.block.[TF].tf.num_layers", 2); data.Set("net.block.[TF].tf.dim_feedforward", 4);
    data.Set("net.block.[TF].tf.norm_first", false);
    data.Set("net.block.[Flatten].type", "Flatten");
    data.Set("net.branch.[feature].bind", kVectorKey);
    data.Set("net.branch.[feature].structure", transformer ? "Expand > Tokens > TF(*2) > Flatten" : "Expand > Flatten");
    data.Set("net.body.output.[features]", "feature");
    const anet::TensorSpec spec{ .type = anet::SpaceType::Vector, .shape = { 2 }, .dtype = torch::kFloat32 };
    return anet::nn::NetworkBuilder::BuildNetwork(anet::nn::NetworkConfig(data),
        anet::TensorSpecMap{ { kVectorKey, spec } },
        std::make_shared<dqn::LinearHeadFactory>(1, anet::nn::WeightInitConfig{}), 9302, torch::kCPU);
}

class AttentionProbeTestNetworkModel final : public dqn::NetworkModel {
public:
    AttentionProbeTestNetworkModel(std::shared_ptr<anet::nn::Network> online, std::shared_ptr<anet::nn::Network> target)
        : dqn::NetworkModel(dqn::NetworkModelConfig{}, std::move(online), std::move(target), 1, false) {}
};

rl::ScalarMetricSubscription AttentionSubscription(int64_t layer, int interval = 1)
{
    return { .source_key = "attention_entropy.[" + std::to_string(layer) + "]", .event = rl::EventType::LEARN,
        .target = rl::EventField::UPDATE_RESULT, .interval = interval, .scope = rl::RunnerScope::TRAIN };
}

// 手作りの行動結果にも、Actorが公開する同じ評価済み値を明示する。
std::shared_ptr<rl::BatchActionInfo> MakeDiagnosticActionInfo(torch::Tensor action, anet::TensorDict info,
    rl::AuxData aux, const dqn::LearnerConfig& config, std::optional<rl::ReplayInitialPriorityHint> hint = std::nullopt)
{
    auto result = std::make_shared<dqn::DQNActionInfo>(std::move(action), std::move(info), std::move(aux), std::move(hint));
    result->effective_gamma = anet::ProfiledValue<float>(config.gamma).Evaluate(0);
    result->munchausen_alpha = anet::ProfiledValue<float>(config.munchausen.alpha).Evaluate(0);
    result->munchausen_entropy_tau = anet::ProfiledValue<float>(config.munchausen.entropy_tau).Evaluate(0);
    return result;
}

} // namespace
