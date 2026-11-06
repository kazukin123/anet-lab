#include "anet/catch_test.hpp"
#include <cmath>
#include <sstream>
#include <fstream>
#include <chrono>
#include <ATen/CPUGeneratorImpl.h>
#include "nn_impl.hpp"
#include "nn_heads.hpp"
#include "anet/intrinsic.hpp"
#include "intrinsic_rnd.hpp"
#include "anet/default_dqn_agent.hpp"
#include "dqn_based_agent.hpp"
#include "anet/metrics_logger.hpp"

constexpr char kObservation[] = "obs";
constexpr char kErrorMean[] = "error_mean";
constexpr char kErrorStd[] = "error_std";
constexpr char kErrorStep[] = "error_step";
constexpr char kPredictorLoss[] = "predictor_loss";
constexpr char kBonusMean[] = "intrinsic.bonus_mean";
constexpr char kBonusShare[] = "intrinsic.bonus_share";
constexpr char kIntrinsicErrorStep[] = "intrinsic.error_step";
constexpr char kNovelty[] = "novelty";
constexpr char kTdMean[] = "td_mean";
constexpr char kSkipContext[] = "skip_context";

static anet::ConfigData RndTestConfig()
{
    anet::ConfigData data;
    data.Set("rnd.net.branch.[feature].bind", kObservation);
    data.Set("rnd.net.branch.[feature].structure", "");
    data.Set("rnd.net.body.output.[features]", "feature");
    data.Set("rnd.output_dim", 4);
    data.Set("rnd.warmup_steps", 2);
    return data;
}

static anet::TensorSpecMap RndTestSpec();
static anet::TensorDict RndTestObservation();

namespace {
    class IntrinsicMetricsBackend final : public anet::IBackend {
    public:
        void Open(const std::filesystem::path&, const std::string&) override {}
        void WriteJsonl(const anet::json&) override {}
        void Flush() override {}
    };
    struct IntrinsicMetricsScope {
        IntrinsicMetricsScope() {
            anet::MetricsLogger::Reset();
            anet::MetricsLoggerConfig config;
            config.run_name_tmpl = "prd116_test";
            anet::MetricsLogger::Init(std::make_unique<IntrinsicMetricsBackend>(), config, "out/test-tmp");
            anet::nn::InitNN();
        }
        ~IntrinsicMetricsScope() { anet::MetricsLogger::Reset(); }
    };
}

static anet::ConfigData IntrinsicAgentData(bool enabled)
{
    anet::ConfigData data;
    data.Set("DefaultDQNAgent.nn_forward.[test].noisy.mode", "mu_only");
    data.Set("DefaultDQNAgent.learner.current.nn_forward_key", "test");
    data.Set("DefaultDQNAgent.learner.target.nn_forward_key", "test");
    data.Set("DefaultDQNAgent.quantile_mode", "none");
    data.Set("DefaultDQNAgent.head.use_dueling", false);
    data.Set("DefaultDQNAgent.stucker.use_stacker", false);
    data.Set("DefaultDQNAgent.obs_norm.pass_through", true);
    data.Set("DefaultDQNAgent.learner.replay_capacity", 16);
    data.Set("DefaultDQNAgent.learner.replay_batch_size", 2);
    data.Set("DefaultDQNAgent.learner.use_fused_optimizer", false);
    data.Set("DefaultDQNAgent.reward_scaler.use_dynamic_scaling", false);
    data.Set("DefaultDQNAgent.reward_scaler.use_auto_post_scale", false);
    data.Set("DefaultDQNAgent.intrinsic.enabled", enabled);
    const auto rnd_data = RndTestConfig();
    for (const auto& [key, value] : rnd_data.Map())
        data.Set("DefaultDQNAgent.intrinsic." + key, value);
    return data;
}
static std::shared_ptr<anet::rl::dqn::DefaultDQNAgent> MakeIntrinsicAgent(const anet::ConfigData& data)
{
    anet::rl::EnvSpec spec;
    spec.state_spec.obs_spec = RndTestSpec();
    spec.action_spec.is_discrete = true;
    spec.action_spec.value_labels = {"a0"};
    spec.reward_range = {-1000.0f, 1000.0f};
    anet::nn::NetworkConfig net;
    net.output_keys["features"] = kObservation;
    if (data.Get("DefaultDQNAgent.skip.enabled", "false") == "true")
        net.output_keys[kSkipContext] = kObservation;
    return std::make_shared<anet::rl::dqn::DefaultDQNAgent>(
        anet::rl::dqn::DefaultDQNAgentConfig(data), net,
        anet::rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, spec, torch::kCPU, 123);
}
static anet::rl::BatchExperience IntrinsicExperience(float scale, bool done, bool truncated, bool start = false)
{
    const auto flags = torch::zeros({2}, torch::kBool);
    return {anet::rl::BatchState(RndTestObservation(), flags, flags, torch::full({2}, start, torch::kBool)),
        std::make_shared<anet::rl::BatchActionInfo>(torch::zeros({2}, torch::kLong)),
        torch::ones({2}),
        anet::rl::BatchState(anet::TensorDict(kObservation, RndTestObservation().At(kObservation) * scale),
            torch::full({2}, done, torch::kBool), torch::full({2}, truncated, torch::kBool), flags)};
}

TEST_CASE("DQN intrinsic validates beta only when enabled", "[prd116][intrinsic][config]")
{
    auto data = IntrinsicAgentData(false);
    data.Set("DefaultDQNAgent.intrinsic.beta", -1.0f);
    data.Set("DefaultDQNAgent.intrinsic.rnd.output_dim", -1);
    CHECK_NOTHROW(anet::rl::dqn::DefaultDQNAgentConfig(data));
    data.Set("DefaultDQNAgent.intrinsic.enabled", true);
    CHECK_THROWS_WITH(anet::rl::dqn::DefaultDQNAgentConfig(data),
        Catch::Matchers::ContainsSubstring("intrinsic.beta"));
}

TEST_CASE("DQN intrinsic masks done but scores truncated lanes", "[prd116][intrinsic]")
{
    IntrinsicMetricsScope metrics;
    const bool done = GENERATE(false, true);
    const bool truncated = !done;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.beta", 0.5f);
    data.Set("DefaultDQNAgent.intrinsic.rnd.obs_norm.pass_through", true);
    auto agent = MakeIntrinsicAgent(data);
    auto learner = agent->CreateLearner();
    REQUIRE(agent->GetScalar(kBonusMean).has_value());
    for (anet::rl::step_t step = 0; step < 2; ++step) {
        learner->UpdateFromBatch({.exp_step = step}, IntrinsicExperience(1, false, false, step == 0));
        CHECK(*agent->GetScalar(kBonusMean) == 0.0f);
    }
    learner->UpdateFromBatch({.exp_step = 2}, IntrinsicExperience(100, done, truncated));
    CHECK(*agent->GetScalar(kBonusMean) == Catch::Approx(done ? 0.0f : 2.0f));
    CHECK(*agent->GetScalar(kBonusShare) == Catch::Approx(done ? 0.0f : 2.0f / 3.0f));
    CHECK(std::isfinite(*agent->GetScalar(kIntrinsicErrorStep)));
}

TEST_CASE("DQN intrinsic disabled and evaluation-only modes stay inactive", "[prd116][intrinsic]")
{
    IntrinsicMetricsScope metrics;
    auto disabled = MakeIntrinsicAgent(IntrinsicAgentData(false));
    CHECK_FALSE(disabled->GetScalar(kBonusMean).has_value());
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.learner.enabled", false);
    auto evaluation = MakeIntrinsicAgent(data);
    evaluation->CreateLearner()->UpdateFromBatch({}, IntrinsicExperience(1, false, false));
    REQUIRE(evaluation->GetScalar(kIntrinsicErrorStep).has_value());
    CHECK(std::isnan(*evaluation->GetScalar(kIntrinsicErrorStep)));
}

static anet::TensorSpecMap RndTestSpec()
{
    return {{kObservation, {.type = anet::SpaceType::Vector, .shape = {2}, .dtype = torch::kFloat32}}};
}

static anet::TensorDict RndTestObservation()
{
    anet::TensorDict obs;
    obs.Set(kObservation, torch::tensor({{1.0f, 2.0f}, {3.0f, -1.0f}}));
    return obs;
}

TEST_CASE("RND factory scores observations and warms up statistics", "[prd116][intrinsic]")
{
    anet::nn::InitNN();
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        RndTestConfig(), "rnd", RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    const auto obs = RndTestObservation();
    for (int step = 0; step < 2; ++step) {
        const auto alpha = estimator->Multiplier(obs);
        CHECK(torch::equal(alpha, torch::ones({2})));
        estimator->Update(obs);
        const auto diagnostics = estimator->Diagnostics();
        CHECK(std::isfinite(diagnostics.at(kErrorMean)));
        CHECK(std::isfinite(diagnostics.at(kErrorStd)));
        CHECK(std::isnan(diagnostics.at(kPredictorLoss)));
    }
}

TEST_CASE("RND predictor learns repeated observations", "[prd116][intrinsic]")
{
    anet::nn::InitNN();
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
    data.Set("rnd.train_fraction", 1.0f);
    data.Set("rnd.learning_rate", 0.03f);
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, "rnd", RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    const auto obs = RndTestObservation();
    estimator->Multiplier(obs);
    const float initial = estimator->Diagnostics().at(kErrorStep);
    estimator->Update(obs);
    for (int step = 0; step < 100; ++step) {
        const auto alpha = estimator->Multiplier(obs);
        CHECK(torch::isfinite(alpha).all().item<bool>());
        CHECK((alpha >= 1.0f).all().item<bool>());
        CHECK((alpha <= 5.0f).all().item<bool>());
        estimator->Update(obs);
    }
    CHECK(std::isfinite(estimator->Diagnostics().at(kPredictorLoss)));
    CHECK(estimator->Diagnostics().at(kErrorStep) < initial * 0.1f);
}

TEST_CASE("RND checkpoint preserves normalization warmup and training continuation", "[prd116][intrinsic]")
{
    anet::nn::InitNN();
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) { SKIP("CUDA is unavailable"); }
    auto data = RndTestConfig();
    data.Set("rnd.learning_rate", 0.01f);
    const auto make = [&] {
        return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            data, "rnd", RndTestSpec(), device, anet::SeedMaker(123));
    };
    auto first = make();
    const auto obs = RndTestObservation();
    first->Multiplier(obs);
    first->Update(obs);
    std::stringstream bytes(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(bytes);
    first->Save(output);
    auto restored = make();
    anet::InputArchive input(bytes);
    restored->Load(input);
    CHECK(restored->Diagnostics().at(kErrorMean) == first->Diagnostics().at(kErrorMean));
    CHECK(restored->Diagnostics().at(kErrorStd) == first->Diagnostics().at(kErrorStd));
    for (int step = 0; step < 12; ++step) {
        const anet::TensorDict varied(kObservation, obs.At(kObservation) * (step + 1.0f));
        CHECK(torch::equal(first->Multiplier(varied), restored->Multiplier(varied)));
        first->Update(varied);
        restored->Update(varied);
        const auto a = first->Diagnostics();
        const auto b = restored->Diagnostics();
        for (const auto& [key, value] : a)
            CHECK((value == b.at(key) || (std::isnan(value) && std::isnan(b.at(key)))));
    }
}

TEST_CASE("RND rejects invalid configuration before building networks", "[prd116][intrinsic][config]")
{
    const auto invalid = GENERATE(Catch::Generators::values<std::pair<std::string, std::string>>({
        {"output_dim", "0"},
        {"warmup_steps", "-1"}, {"multiplier_max", "1"}, {"multiplier_max", "nan"},
        {"train_fraction", "0"}, {"train_fraction", "1.1"}, {"train_fraction", "nan"},
        {"learning_rate", "0"}, {"learning_rate", "inf"}, {"weight_decay", "-1"},
        {"obs_norm.clip_range", "0"}, {"obs_norm.epsilon", "-1"},
        {"obs_norm.post_process_type", "99"}}));
    auto data = RndTestConfig();
    data.Set("rnd." + invalid.first, invalid.second);
    CAPTURE(invalid.first, invalid.second);
    CHECK_THROWS_WITH(anet::rl::intrinsic::RndNoveltyEstimatorConfig(data, "rnd"),
        Catch::Matchers::ContainsSubstring("rnd." + invalid.first)
        && Catch::Matchers::ContainsSubstring("expected"));
}

TEST_CASE("RND rejects features with extra axes at construction", "[prd116][intrinsic][config]")
{
    anet::nn::InitNN();
    auto spec = RndTestSpec();
    spec.at(kObservation).shape = {2, 2};
    CHECK_THROWS_WITH(anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        RndTestConfig(), "rnd", spec, torch::kCPU, anet::SeedMaker(123)),
        Catch::Matchers::ContainsSubstring("RND features")
        && Catch::Matchers::ContainsSubstring("expected [B, F]"));
}

TEST_CASE("RND whitens byte observations like float observations", "[prd116][intrinsic]")
{
    anet::nn::InitNN();
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) { SKIP("CUDA is unavailable"); }
    auto byte_spec = RndTestSpec();
    byte_spec.at(kObservation).dtype = torch::kUInt8;
    const auto make = [&](const anet::TensorSpecMap& spec) {
        return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            RndTestConfig(), "rnd", spec, device, anet::SeedMaker(123));
    };
    auto bytes = make(byte_spec);
    auto floats = make(RndTestSpec());
    for (int step = 0; step < 4; ++step) {
        const auto values = torch::tensor({{16.0f, 32.0f}, {128.0f, 255.0f}});
        anet::TensorDict byte_obs(kObservation, values.to(torch::kUInt8));
        anet::TensorDict float_obs(kObservation, values);
        CHECK(torch::equal(bytes->Multiplier(byte_obs), floats->Multiplier(float_obs)));
        CHECK(bytes->Diagnostics().at(kErrorStep) == floats->Diagnostics().at(kErrorStep));
        bytes->Update(byte_obs);
        floats->Update(float_obs);
    }
}

TEST_CASE("RND keeps its seeded target fixed and uses preceding error statistics", "[prd116][intrinsic]")
{
    anet::nn::InitNN();
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
    data.Set("rnd.train_fraction", 1.0f);
    const auto initial_random = at::detail::getDefaultCPUGenerator().get_state();
    const auto make = [&] {
        return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            data, "rnd", RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    };
    auto estimator = make();
    const auto networks = [](const auto& instance) {
        std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
        anet::OutputArchive output(stream);
        instance->Save(output);
        anet::InputArchive input(stream);
        std::string format, target, predictor;
        input.Read(format);
        input.Read(target);
        input.Read(predictor);
        return std::pair(target, predictor);
    };
    const auto before = networks(estimator);
    CHECK(before.first != before.second);
    CHECK(before == networks(make()));
    CHECK(torch::equal(initial_random, at::detail::getDefaultCPUGenerator().get_state()));
    for (int step = 0; step < 12; ++step) {
        const auto previous = estimator->Diagnostics();
        const anet::TensorDict obs(kObservation, torch::tensor({{step + 1.0f, step * 2.0f}}));
        const auto alpha = estimator->Multiplier(obs).item<float>();
        const auto error = estimator->Diagnostics().at(kErrorStep);
        const auto std = previous.at(kErrorStd);
        const auto expected = std > 0 ? std::clamp(1 + (error - previous.at(kErrorMean)) / std, 1.0f, 5.0f) : 1.0f;
        CHECK(alpha == Catch::Approx(expected));
        estimator->Update(obs);
    }
    const auto after = networks(estimator);
    CHECK(before.first == after.first);
    CHECK(before.second != after.second);

    // 同じ入力を100回学習しても、保存したtargetの重みと出力は変わらない。
    const auto repeated = RndTestObservation();
    const auto target_output = [&](const std::string& saved_target) {
        auto network = anet::nn::NetworkBuilder::BuildNetwork(
            anet::nn::NetworkConfig(data, "rnd.net"), RndTestSpec(),
            std::make_shared<anet::nn::LinearHeadFactory>(4, kNovelty, anet::nn::WeightInitConfig{}),
            0, torch::kCPU);
        std::istringstream saved(saved_target, std::ios::binary);
        torch::load(network, saved);
        network->eval();
        torch::NoGradGuard guard;
        return network->Forward(repeated).At(kNovelty).clone();
    };
    const auto initial_target_output = target_output(after.first);
    for (int step = 0; step < 100; ++step) {
        estimator->Multiplier(repeated);
        estimator->Update(repeated);
    }
    const auto repeated_after = networks(estimator);
    CHECK(after.first == repeated_after.first);
    CHECK(torch::equal(initial_target_output, target_output(repeated_after.first)));
}

TEST_CASE("RND rejects nonfinite prediction errors before publishing statistics", "[prd116][intrinsic]")
{
    anet::nn::InitNN();
    auto data = RndTestConfig();
    data.Set("rnd.obs_norm.pass_through", true);
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, "rnd", RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    const anet::TensorDict obs(kObservation, torch::full({2, 2}, std::numeric_limits<float>::quiet_NaN()));
    CHECK_THROWS_WITH(estimator->Multiplier(obs), Catch::Matchers::ContainsSubstring("prediction_error")
        && Catch::Matchers::ContainsSubstring("expected finite"));
}

TEST_CASE("DQN learner consumes the scaled external reward plus intrinsic bonus", "[prd116][intrinsic]")
{
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.obs_norm.pass_through", true);
    data.Set("DefaultDQNAgent.intrinsic.beta", 0.5f);
    data.Set("DefaultDQNAgent.head.init.mode", "constant");
    data.Set("DefaultDQNAgent.head.init.constant_val", 0.0f);
    data.Set("DefaultDQNAgent.learner.learning_rate", 0.0f);
    data.Set("DefaultDQNAgent.learner.gamma", 0.0f);
    data.Set("DefaultDQNAgent.learner.use_n_step", false);
    data.Set("DefaultDQNAgent.learner.use_per", false);
    data.Set("DefaultDQNAgent.learner.update_warmup_steps", 0);
    data.Set("DefaultDQNAgent.learner.update_interval", 1);
    data.Set("DefaultDQNAgent.learner.replay_capacity", 4);
    data.Set("DefaultDQNAgent.learner.use_td_clip", false);
    data.Set("DefaultDQNAgent.reward_scaler.constant_scale", 2.0f);
    auto agent = MakeIntrinsicAgent(data);
    auto learner = agent->CreateLearner();
    learner->UpdateFromBatch({}, IntrinsicExperience(1, false, false, true));
    learner->UpdateFromBatch({.exp_step = 2}, IntrinsicExperience(1, false, false));
    learner->UpdateFromBatch({.exp_step = 4}, IntrinsicExperience(100, false, false));
    const auto results = learner->UpdateFromBatch({.exp_step = 6}, IntrinsicExperience(10000, false, true));
    REQUIRE_FALSE(results.empty());
    REQUIRE(results.back()->GetScalar(kTdMean).has_value());
    CHECK(*results.back()->GetScalar(kTdMean) == Catch::Approx(4.0f));
}

TEST_CASE("DQN checkpoint requires matching intrinsic presence and resumes scoring", "[prd116][intrinsic][checkpoint]")
{
    IntrinsicMetricsScope metrics;
    const bool enabled = GENERATE(false, true);
    const bool skip_enabled = GENERATE(false, true);
    auto data = IntrinsicAgentData(enabled);
    if (skip_enabled) {
        data.Set("DefaultDQNAgent.skip.enabled", true);
        data.Set("DefaultDQNAgent.skip.max_length", 2);
        data.Set("DefaultDQNAgent.skip.net.branch.[skip_feature].bind", kSkipContext);
        data.Set("DefaultDQNAgent.skip.net.branch.[skip_feature].structure", "");
        data.Set("DefaultDQNAgent.skip.net.body.output.[features]", "skip_feature");
    }
    auto original = MakeIntrinsicAgent(data);
    original->CreateLearner()->UpdateFromBatch({}, IntrinsicExperience(1, false, false, true));
    const auto path = std::filesystem::temp_directory_path() / ("anet-prd116-"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".anet");
    struct RemoveCheckpoint {
        std::filesystem::path path;
        ~RemoveCheckpoint() { std::error_code ignored; std::filesystem::remove(path, ignored); }
    } cleanup{.path = path};
    {
        std::ofstream stream(path, std::ios::binary);
        anet::OutputArchive archive(stream);
        original->Save(archive);
    }
    {
        std::ifstream stream(path, std::ios::binary);
        anet::InputArchive archive(stream);
        anet::ArchiveHeader header;
        archive.Read(header);
        const std::string expected = skip_enabled ? "DefaultDQNAgent-2" : "DefaultDQNAgent";
        CHECK(header.info == expected + (enabled ? ";intrinsic" : ""));
    }
    data.Set("DefaultDQNAgent.auto_load_file", path.string());
    auto restored = MakeIntrinsicAgent(data);
    // Replay は保存対象外なので、復元側の最初の経験は新しい episode とする。
    for (anet::rl::step_t step = 1; step < 5; ++step) {
        original->CreateLearner()->UpdateFromBatch({.exp_step = step}, IntrinsicExperience(10, false, false));
        restored->CreateLearner()->UpdateFromBatch({.exp_step = step}, IntrinsicExperience(10, false, false, step == 1));
        CHECK(original->GetScalar(kBonusMean) == restored->GetScalar(kBonusMean));
    }
    data.Set("DefaultDQNAgent.intrinsic.enabled", !enabled);
    CHECK_THROWS_WITH(MakeIntrinsicAgent(data), Catch::Matchers::ContainsSubstring("Checkpoint header=")
        && Catch::Matchers::ContainsSubstring("intrinsic.enabled"));
    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        anet::OutputArchive archive(stream);
        anet::ArchiveHeader unknown;
        unknown.info = "unknown-prd116-header";
        archive.Write(unknown);
        archive.Write(std::string("{}"));
    }
    CHECK_THROWS_WITH(MakeIntrinsicAgent(data),
        Catch::Matchers::ContainsSubstring("Invalid DefaultDQN checkpoint header='unknown-prd116-header'")
        && Catch::Matchers::ContainsSubstring("expected"));
}

TEST_CASE("RND checkpoint applies current optimizer options", "[prd116][intrinsic][checkpoint]")
{
    anet::nn::InitNN();
    const auto changed_key = GENERATE("learning_rate", "weight_decay");
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
    data.Set("rnd.train_fraction", 1.0f);
    data.Set("rnd.obs_norm.pass_through", true);
    const auto make = [](const anet::ConfigData& config) {
        return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            config, "rnd", RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    };
    auto original = make(data);
    const auto obs = RndTestObservation();
    for (int step = 0; step < 3; ++step) { original->Multiplier(obs); original->Update(obs); }
    std::stringstream bytes(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(bytes);
    original->Save(output);
    data.Set(std::string("rnd.") + changed_key, 0.5f);
    auto changed = make(data);
    anet::InputArchive input(bytes);
    changed->Load(input);
    CHECK(torch::equal(original->Multiplier(obs), changed->Multiplier(obs)));
    original->Update(obs);
    changed->Update(obs);
    original->Multiplier(obs);
    changed->Multiplier(obs);
    CHECK(original->Diagnostics().at(kErrorStep) != changed->Diagnostics().at(kErrorStep));
}
