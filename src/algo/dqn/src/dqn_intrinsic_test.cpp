#include "anet/test/catch.hpp"
#include <array>
#include <cmath>
#include <sstream>
#include <tuple>
#include <fstream>
#include <future>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include "anet/test/nn.hpp"
#include "anet/nn_head.hpp"
#include "anet/env.hpp"
#include "anet/intrinsic.hpp"
#include "anet/test/intrinsic.hpp"
#include "default_dqn_agent.hpp"
#include "dqn_learner.hpp"
#include "anet/metrics.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/observers.hpp"
#include "anet/test/util.hpp"


constexpr char kTraceActionLabel[] = "a0";
constexpr char kTraceActorKey[] = "trace";
constexpr char kTraceForwardKey[] = "test";
constexpr char kStepIntrinsicError[] = "intrinsic.error";
constexpr char kStepIntrinsicNormalizedError[] = "intrinsic.normalized_error";
constexpr char kStepIntrinsicBonus[] = "intrinsic.bonus";
constexpr char kObservation[] = "obs";
constexpr char kBonusMean[] = "intrinsic.bonus_mean";
constexpr char kBonusShare[] = "intrinsic.bonus_share";
constexpr char kBonusMin[] = "intrinsic.bonus_min";
constexpr char kBonusPosRatio[] = "intrinsic.bonus_pos_ratio";
constexpr char kIntrinsicErrorStep[] = "intrinsic.error_step";
constexpr char kNovelty[] = "novelty";
constexpr char kTdMean[] = "td_mean";
constexpr char kSkipContext[] = "skip_context";
constexpr char kRndPrefix[] = "rnd";
constexpr char kIntrinsicLearnCount[] = "intrinsic.learn_count";
constexpr char kIntrinsicPredictorLoss[] = "intrinsic.predictor_loss";
constexpr char kReferenceGrid[] = "grid";
constexpr char kReferenceFeature[] = "feature";
constexpr char kTargetRole[] = "target";
constexpr char kPredictorRole[] = "predictor";
constexpr char kConstantInit[] = "constant";
constexpr char kReluBlock[] = "ReLU";
constexpr char kInvalidConfigValue[] = "unknown";
constexpr char kFalseConfigValue[] = "false";
constexpr char kRndCheckpointFormat[] = "RndNoveltyEstimator-3";
constexpr char kRndEmaCheckpointFormat[] = "RndNoveltyEstimator-4";
constexpr char kCheckpointExtension[] = ".anet";
constexpr char kIntrinsicPrefix[] = "DefaultDQNAgent.intrinsic";
constexpr char kUnknownIntrinsicScalarKey[] = "intrinsic.unknown_key";
constexpr char kRawErrorMode[] = "none";

constexpr const char* kRfd0121ExtrinsicGradNorm = "extrinsic_grad_norm";

constexpr const char* kRfd0121ExtrinsicQSaMean = "extrinsic_q_sa_mean";

constexpr const char* kRfd0121IntrinsicGradNorm = "intrinsic_grad_norm";

constexpr const char* kRfd0121IntrinsicQSaMean = "intrinsic_q_sa_mean";

constexpr const char* kRfd0121IntrinsicQShare = "intrinsic_q_share";

constexpr const char* kRfd0121ApproxPriority = "approx_priority";

constexpr const char* kRfd0121ExtrinsicHeadGradNorm = "extrinsic_head_grad_norm";

constexpr const char* kRfd0121ExtrinsicLoss = "extrinsic_loss";

constexpr const char* kRfd0121ExtrinsicQ = "extrinsic_q";

constexpr const char* kRfd0121ExtrinsicQSa = "extrinsic_q_sa";

constexpr const char* kRfd0121ExtrinsicStateValue = "extrinsic_state_value";

constexpr const char* kRfd0121FixedActorKey = "fixed";

constexpr const char* kRfd0121IntrinsicHeadGradNorm = "intrinsic_head_grad_norm";

constexpr const char* kRfd0121IntrinsicLoss = "intrinsic_loss";

constexpr const char* kRfd0121IntrinsicQ = "intrinsic_q";

constexpr const char* kRfd0121IntrinsicQSa = "intrinsic_q_sa";

constexpr const char* kRfd0121IntrinsicRewardScalerScale = "intrinsic.reward_scaler.scale";

constexpr const char* kRfd0121IntrinsicStateValue = "intrinsic_state_value";

constexpr const char* kRfd0121Learner = "learner";

constexpr const char* kRfd0121NoisyQAbsDiff = "noisy_q_abs_diff";

constexpr const char* kRfd0121Off = "off";

constexpr const char* kRfd0121PolicyChurnQDeltaAbsMean = "policy_churn_q_delta_abs_mean";

constexpr const char* kRfd0121PolicyNetForward = "policy-net.forward";

constexpr const char* kRfd0121Q = "q";

constexpr const char* kRfd0121ReplayFitUniformTdMean = "replay_fit_uniform_td_mean";

constexpr std::array<const char*, 13> kAllIntrinsicScalarKeys{
    kBonusMean, "intrinsic.bonus_max", kBonusMin, kBonusPosRatio,
    kBonusShare, kIntrinsicLearnCount, "intrinsic.error_mean", "intrinsic.error_std",
    kIntrinsicErrorStep, "intrinsic.clip_ratio", kIntrinsicPredictorLoss,
    "intrinsic.target_feature_var", "intrinsic.target_feature_max"};

static anet::ConfigData RndTestConfig()
{
    anet::ConfigData data;
    for (const auto* role : {kTargetRole, kPredictorRole}) {
        const auto prefix = std::string(kRndPrefix) + "." + role + ".net";
        data.Set(prefix + ".branch.[feature].bind", kObservation);
        data.Set(prefix + ".branch.[feature].structure", "");
        data.Set(prefix + ".body.output.[features]", kReferenceFeature);
    }
    data.Set("rnd.head.output_dim", 4);
    data.Set("rnd.error_norm.standardize.warmup_steps", 4);
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
}

namespace {
    struct IntrinsicMetricsScope {
            explicit IntrinsicMetricsScope(std::string run_name = "rfd0116_test") {
                anet::MetricsLogger::Reset();
                anet::MetricsLoggerConfig config;
                config.run_name_tmpl = std::move(run_name);
                anet::MetricsLogger::Init(std::make_unique<IntrinsicMetricsBackend>(), config, "out/test-tmp");
                anet::test::RegisterStandardNetworkModules();
            }
            ~IntrinsicMetricsScope() { anet::MetricsLogger::Reset(); }
        };
}

namespace {
    struct IntrinsicCheckpointCleanup {
            std::filesystem::path path;
            ~IntrinsicCheckpointCleanup() {
                std::error_code ignored;
                std::filesystem::remove(path, ignored);
            }
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


    data.Set("DefaultDQNAgent.learner.replay_capacity", 16);
    data.Set("DefaultDQNAgent.learner.replay_batch_size", 2);
    data.Set("DefaultDQNAgent.learner.use_fused_optimizer", false);
    data.Set("DefaultDQNAgent.reward_scaler.use_dynamic_scaling", false);
    data.Set("DefaultDQNAgent.reward_scaler.use_auto_post_scale", false);
    data.Set("DefaultDQNAgent.intrinsic.enabled", enabled);
    data.Set("DefaultDQNAgent.intrinsic.learn.intake.batch_size", 2);
    const auto rnd_data = RndTestConfig();
    for (const auto& [key, value] : rnd_data.Map())
        data.Set("DefaultDQNAgent.intrinsic." + key, value);
    return data;
}

static anet::nn::NetworkConfig MakeIntrinsicNetworkConfig()
{
    anet::nn::NetworkConfig net;
    net.block_configs["Flatten"].type = "Flatten";
    net.branches["main"] = {.name = "main", .bind_terms = {{kObservation}}, .structure_str = "Flatten"};
    net.output_keys["features"] = "main";
    return net;
}

static std::shared_ptr<anet::rl::dqn::DefaultDQNAgent> MakeIntrinsicAgent(const anet::ConfigData& data, int num_envs = 2,
    torch::Device device = torch::kCPU)
{
    anet::rl::EnvSpec spec;
    spec.state_spec.obs_spec = RndTestSpec();
    spec.action_spec.is_discrete = true;
    spec.action_spec.value_labels = {"a0"};
    spec.reward_range = {-1000.0f, 1000.0f};
    auto net = MakeIntrinsicNetworkConfig();
    if (data.Get("DefaultDQNAgent.skip.enabled", "false") == "true")
        net.output_keys[kSkipContext] = "main";
    return std::make_shared<anet::rl::dqn::DefaultDQNAgent>(
        anet::rl::dqn::DefaultDQNAgentConfig(data), net,
        anet::rl::BatchEnvSpec{.num_envs = num_envs, .num_threads = 1}, spec, device, 123);
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

TEST_CASE("DQN records target and predictor network graphs when RND is enabled", "[rfd0191][intrinsic][rnd_graphs]")
{
    const bool enabled = GENERATE(false, true);
    IntrinsicMetricsScope metrics(enabled ? "rfd0191_graphs_on_{t}" : "rfd0191_graphs_off_{t}");
    const auto agent = MakeIntrinsicAgent(IntrinsicAgentData(enabled));
    const auto dot_dir = anet::MetricsLogger::Instance()->GetRunDir() / "dot";
    REQUIRE(std::filesystem::is_regular_file(dot_dir / "net.structure.dot"));
    for (const auto* filename : {"intrinsic.rnd.target.net.structure.dot", "intrinsic.rnd.target.net.detail.dot",
        "intrinsic.rnd.predictor.net.structure.dot", "intrinsic.rnd.predictor.net.detail.dot"}) {
        CAPTURE(filename);
        CHECK(std::filesystem::is_regular_file(dot_dir / filename) == enabled);
    }
}

TEST_CASE("RND shipped Atari networks and Taiga settings match their contracts",
    "[rfd0116][shipped_config][rnd_reference_shipped]")
{
    IntrinsicMetricsScope metrics;
    const auto root = std::filesystem::current_path();
    const auto path = root / "out/test-tmp/rfd0116-reference-config.txt";
    std::filesystem::create_directories(path.parent_path());
    const auto resolve = [&](const std::string& chain) {
        std::ofstream file(path);
        file << "$include <" << (root / "apps/runner/config/_main.txt").generic_string() << ">\n";
        file << "$include <" << (root / "apps/runner/config/Atari.txt").generic_string() << ">\n";
        file << "run.$ = " << chain << '\n';
        file.close();
        return anet::ConfigManager(path).GetConfigData();
    };
    auto data = resolve("run.@rnd");
    const auto prefix = std::string(kIntrinsicPrefix) + ".rnd";
    const auto rnd_config = anet::test::ReadRndConfig(data, prefix);
    for (const auto* key : {"intrinsic.use_done_mask", "intrinsic.rnd.predictor.warmup_steps",
        "intrinsic.rnd.error_norm.standardize.average_mode", "intrinsic.rnd.error_norm.standardize.ema_steps",
        "intrinsic.rnd.error_norm.standardize.use_centering"})
        REQUIRE(data.Has(std::string("DefaultDQNAgent.") + key));
    const auto rnd_defaults = anet::test::ReadRndConfig(RndTestConfig(), kRndPrefix);
    CHECK(rnd_config.predictor_warmup_steps == rnd_defaults.predictor_warmup_steps);
    CHECK(rnd_config.standardize.average_mode == rnd_defaults.standardize.average_mode);
    CHECK(rnd_config.standardize.ema_steps == rnd_defaults.standardize.ema_steps);
    CHECK(rnd_config.standardize.use_centering == rnd_defaults.standardize.use_centering);
    bool use_done_mask = false;
    data.Read("DefaultDQNAgent.intrinsic.use_done_mask", use_done_mask, false);
    const anet::rl::dqn::DefaultDQNAgentConfig facade_defaults(IntrinsicAgentData(true));
    CHECK(use_done_mask == facade_defaults.intrinsic.use_done_mask);
    CHECK(rnd_config.multiplier_max == 5.0f);
    CHECK(rnd_config.observation_clip_range == 5.0f);
    CHECK(rnd_config.head_init.mode == "orthogonal");
    CHECK(rnd_config.head_init.manual_gain == Catch::Approx(std::sqrt(2.0)));
    data.Set(prefix + ".error_norm.standardize.warmup_steps", 0);
    const anet::TensorSpecMap image_spec{{kReferenceGrid,
        {.type = anet::SpaceType::Grid, .shape = {1, 84, 84}, .dtype = torch::kUInt8}}};
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, prefix, image_spec, torch::kCPU, anet::SeedMaker(42));
    const anet::TensorDict image(kReferenceGrid, torch::zeros({1, 1, 84, 84}, torch::kUInt8));
    estimator->Estimate(image);
    REQUIRE(estimator->Learn(image));
    std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(saved);
    estimator->Save(output);

    // 本体のpublic checkpointから、実際に作った両NetworkのTensorを数える。
    anet::InputArchive input(saved);
    std::string format, mode, target, predictor;
    input.Read(format);
    input.Read(mode);
    REQUIRE(format == kRndCheckpointFormat);
    input.Read(target);
    input.Read(predictor);
    const auto count_parameters = [](const std::string& bytes) {
        torch::serialize::InputArchive archive;
        std::istringstream stream(bytes, std::ios::binary);
        archive.load_from(stream, torch::kCPU);
        const auto count = [](auto&& self, torch::serialize::InputArchive& current) -> int64_t {
            int64_t total = 0;
            for (const auto& key : current.keys()) {
                c10::IValue value;
                if (current.try_read(key, value) && value.isTensor()) {
                    total += value.toTensor().numel();
                } else {
                    torch::serialize::InputArchive child;
                    if (current.try_read(key, child)) total += self(self, child);
                }
            }
            return total;
        };
        return count(count, archive);
    };
    CHECK(count_parameters(target) == 1677984);
    CHECK(count_parameters(predictor) == 2203296);
    CHECK(saved.str().size() >= (1677984 + 2203296 * 3) * sizeof(float));
    CHECK(saved.str().size() < 35000000);

    // 共通の1D/2D部品は宣言どおりの幅でFlattenされる。
    for (const bool image_input : {false, true}) {
        CAPTURE(image_input);
        auto catalog = data;
        catalog.Set("catalog.net.branch.[feature].bind", kReferenceGrid);
        catalog.Set("catalog.net.branch.[feature].structure",
            image_input ? "Conv2d_S > ReLU > Conv2d_M > ReLU > Flatten"
                : "Conv1d_S > ReLU > Conv1d_M > ReLU > Flatten");
        catalog.Set("catalog.net.body.output.[features]", kReferenceFeature);
        const anet::TensorSpec spec{.type = image_input ? anet::SpaceType::Grid : anet::SpaceType::Sequence,
            .shape = image_input ? std::vector<int64_t>{1, 84, 84} : std::vector<int64_t>{1, 84},
            .dtype = torch::kFloat32};
        const anet::nn::WeightInitConfig init;
        auto head = std::make_shared<anet::nn::LinearHeadFactory>(128, kNovelty, init);
        auto network = anet::nn::NetworkBuilder::BuildNetwork(
            anet::nn::NetworkConfig(catalog, "catalog.net"), {{kReferenceGrid, spec}}, head, 42, torch::kCPU);
        const anet::TensorDict obs(kReferenceGrid, image_input
            ? torch::zeros({1, 1, 84, 84}) : torch::zeros({1, 1, 84}));
        CHECK(network->ForwardUpTo(obs, kReferenceFeature).At(kReferenceFeature).size(1) == (image_input ? 6400 : 640));
        if (image_input) {
            int64_t count = 0;
            for (const auto& parameter : network->parameters()) count += parameter.numel();
            CHECK(count == 854240);
        }
    }

    // 出荷の Taiga の連鎖を解決し、両方の設定の読み口を通す。
    const auto taiga = resolve("run.@rainbow_like > run.@rnd > run.@rnd_taiga");
    CHECK_NOTHROW(anet::rl::dqn::DefaultDQNAgentConfig(taiga));
    CHECK_NOTHROW(anet::test::ReadRndConfig(taiga, prefix));
}

TEST_CASE("DQN intrinsic validates beta only when enabled", "[rfd0116][intrinsic][config]")
{
    auto data = IntrinsicAgentData(false);
    data.Set("DefaultDQNAgent.intrinsic.beta", -1.0f);
    data.Set("DefaultDQNAgent.intrinsic.rnd.head.output_dim", -1);
    CHECK_NOTHROW(anet::rl::dqn::DefaultDQNAgentConfig(data));
    data.Set("DefaultDQNAgent.intrinsic.enabled", true);
    CHECK_THROWS_WITH(anet::rl::dqn::DefaultDQNAgentConfig(data),
        Catch::Matchers::ContainsSubstring("intrinsic.beta"));
}

TEST_CASE("DQN collects RND errors while predictor learning waits", "[rfd0191][intrinsic][rnd_settings]")
{
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.rnd.predictor.warmup_steps", 2);
    data.Set("DefaultDQNAgent.intrinsic.learn.intake.ratio", 1.0f);
    data.Set("DefaultDQNAgent.intrinsic.learn.intake.batch_size", 2);
    auto agent = MakeIntrinsicAgent(data);
    auto learner = agent->CreateLearner();
    learner->UpdateFromBatch({.exp_step = 2}, IntrinsicExperience(1.0f, false, false, true));
    CHECK(std::isfinite(agent->GetScalar("intrinsic.error_mean").value()));
    CHECK(agent->GetScalar(kIntrinsicLearnCount).value() == 0.0f);
    learner->UpdateFromBatch({.exp_step = 4}, IntrinsicExperience(2.0f, false, false));
    CHECK(agent->GetScalar(kIntrinsicLearnCount).value() == 1.0f);
}

TEST_CASE("DQN intrinsic state scoring ignores next observations and terminal masks", "[rfd0116][rnd_reference_observation]")
{
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.observation", anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationState);
    data.Set("DefaultDQNAgent.intrinsic.beta", 0.5f);
    data.Set("DefaultDQNAgent.intrinsic.learn.intake.ratio", 1e-6f);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.rnd.statistics.mode", "variance_feedback");
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.use_clipping", false);
    data.Set("DefaultDQNAgent.intrinsic.rnd.head.init.mode", kConstantInit);
    data.Set("DefaultDQNAgent.intrinsic.rnd.head.init.constant_val", 1.0f);
    data.Set("DefaultDQNAgent.intrinsic.rnd.predictor.net.branch.[feature].structure", kReluBlock);
    data.Set("DefaultDQNAgent.intrinsic.rnd.predictor.net.block.[ReLU].type", kReluBlock);
    auto terminal = MakeIntrinsicAgent(data);
    auto continuing = MakeIntrinsicAgent(data);
    auto ended = IntrinsicExperience(100.0f, true, false, true);
    auto active = IntrinsicExperience(0.01f, false, false, true);
    const anet::TensorDict observations(kObservation, torch::tensor({{-1.0f, 2.0f}, {2.0f, -1.0f}}));
    ended.state.obs = observations;
    active.state.obs = observations;

    // 行動した観測は同じ。次観測とdoneを変えても内的報酬は同じである。
    terminal->CreateLearner()->UpdateFromBatch({}, ended);
    continuing->CreateLearner()->UpdateFromBatch({}, active);
    CHECK(*terminal->GetScalar(kBonusMean) == Catch::Approx(0.15f));
    CHECK(terminal->GetScalar(kBonusMean) == continuing->GetScalar(kBonusMean));
    REQUIRE(terminal->GetScalar(kBonusMin).has_value());
    REQUIRE(terminal->GetScalar(kBonusPosRatio).has_value());
    CHECK(*terminal->GetScalar(kBonusMin) == 0.0f);
    CHECK(*terminal->GetScalar(kBonusPosRatio) == 0.5f);

    // 次のエピソードで誤差0の観測を渡す。負のボーナスでもshareは絶対値の比になる。
    auto familiar = IntrinsicExperience(100.0f, true, false, true);
    familiar.state.obs = anet::TensorDict(kObservation, torch::tensor({{1.5f, 0.0f}, {1.5f, 0.0f}}));
    terminal->CreateLearner()->UpdateFromBatch({.exp_step = 2}, familiar);
    REQUIRE(*terminal->GetScalar(kBonusMean) < 0.0f);
    CHECK(*terminal->GetScalar(kBonusMin) < 0.0f);
    CHECK(*terminal->GetScalar(kBonusPosRatio) == 0.0f);
    const auto share = *terminal->GetScalar(kBonusShare);
    CHECK(std::isfinite(share));
    CHECK(share > 0.0f);
    CHECK(share <= 1.0f);
}

TEST_CASE("DQN intrinsic learning cadence follows experience count", "[rfd0116][intrinsic][rnd_schedule]")
{
    IntrinsicMetricsScope metrics;
    const int num_envs = GENERATE(2, 4);
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.learn.intake.ratio", 0.25f);
    data.Set("DefaultDQNAgent.intrinsic.learn.intake.batch_size", 2);
    auto agent = MakeIntrinsicAgent(data, num_envs);
    auto learner = agent->CreateLearner();
    REQUIRE(agent->GetScalar(kIntrinsicLearnCount).has_value());
    // 同じ32経験を、2 laneと4 laneのintakeに分けて流す。
    for (int count = num_envs; count <= 32; count += num_envs) {
        const auto flags = torch::zeros({num_envs}, torch::kBool);
        const anet::TensorDict obs(kObservation, RndTestObservation().At(kObservation).repeat({num_envs / 2, 1}));
        const anet::rl::BatchExperience experience{
            anet::rl::BatchState(obs, flags, flags, torch::full({num_envs}, count == num_envs, torch::kBool)),
            std::make_shared<anet::rl::BatchActionInfo>(torch::zeros({num_envs}, torch::kLong)),
            torch::ones({num_envs}), anet::rl::BatchState(obs, flags, flags, flags)};
        learner->UpdateFromBatch({.exp_step = static_cast<anet::rl::step_t>(count)}, experience);
        CHECK(*agent->GetScalar(kIntrinsicLearnCount) == static_cast<float>(count / 8));
    }
    CHECK(*agent->GetScalar(kIntrinsicLearnCount) == 4.0f);
    // 抽選テストで1 stepの全入力が2件であることも重みから照合する。
    CHECK(*agent->GetScalar(kIntrinsicLearnCount) * 2 == 8.0f);
}

TEST_CASE("DQN learns complete RND rollouts on the selected schedule", "[rfd0191][intrinsic][rnd_rollout]")
{
    IntrinsicMetricsScope metrics;
    const bool spread = GENERATE(false, true);
    const int ratio = GENERATE(1, 2);
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.learn.mode", anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicLearnRollout);
    data.Set("DefaultDQNAgent.intrinsic.learn.rollout.length", 4);
    data.Set("DefaultDQNAgent.intrinsic.learn.rollout.epochs", 2);
    data.Set("DefaultDQNAgent.intrinsic.learn.rollout.ratio", ratio);
    data.Set("DefaultDQNAgent.intrinsic.learn.rollout.batch_size", 8);
    data.Set("DefaultDQNAgent.intrinsic.learn.rollout.use_spread", spread);
    auto agent = MakeIntrinsicAgent(data, 4);
    auto learner = agent->CreateLearner();
    const anet::SeedMaker seed_maker(123);
    auto reference = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        RndTestConfig(), kRndPrefix, RndTestSpec(), torch::kCPU, seed_maker);
    std::mt19937_64 random(seed_maker.MakeNamedSeed("intrinsic.learn"));
    const auto all_lanes = torch::arange(8, torch::kFloat32).reshape({4, 2});
    const auto observations_at = [&](int intake) { return all_lanes + intake * 16; };
    std::vector<int64_t> order(16);
    int previous_total = 0;
    const int rollout_steps = 2 * ratio;
    for (int intake = 1; intake <= 8; ++intake) {
        const auto flags = torch::zeros({4}, torch::kBool);
        const anet::TensorDict obs(kObservation, observations_at(intake));
        const anet::rl::BatchExperience experience{
            anet::rl::BatchState(obs, flags, flags, torch::full({4}, intake == 1, torch::kBool)),
            std::make_shared<anet::rl::BatchActionInfo>(torch::zeros({4}, torch::kLong)),
            torch::ones({4}), anet::rl::BatchState(obs, flags, flags, flags)};
        learner->UpdateFromBatch({.exp_step = static_cast<anet::rl::step_t>(intake * 4)}, experience);
        CAPTURE(intake);
        const auto expected = spread ? std::max(0, intake - 4) * rollout_steps / 4
            : (intake / 4) * rollout_steps;
        CHECK(*agent->GetScalar(kIntrinsicLearnCount) == expected);
        reference->Estimate(obs);
        if (expected > previous_total) {
            const int first_intake = spread ? 1 : intake - 3;
            const auto chronological = torch::cat({observations_at(first_intake), observations_at(first_intake + 1),
                observations_at(first_intake + 2), observations_at(first_intake + 3)});
            for (int step = previous_total; step < expected; ++step) {
                const int batch = step % ratio;
                if (batch == 0) {
                    std::iota(order.begin(), order.end(), int64_t{0});
                    std::shuffle(order.begin(), order.end(), random);
                }
                const auto first = order.begin() + 8 * batch;
                const auto indices = torch::tensor(std::vector<int64_t>(first, first + 8), torch::kLong);
                REQUIRE(reference->Learn({kObservation, chronological.index_select(0, indices)}));
            }
            CHECK(agent->GetScalar(kIntrinsicPredictorLoss).value()
                == Catch::Approx(reference->Diagnostics().at("predictor_loss")));
        }
        previous_total = expected;
    }
}

TEST_CASE("DQN intrinsic masks done but scores truncated lanes", "[rfd0116][intrinsic]")
{
    IntrinsicMetricsScope metrics;
    const bool done = GENERATE(false, true);
    const bool truncated = !done;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.rnd.predictor.warmup_steps", 4);
    data.Set("DefaultDQNAgent.intrinsic.beta", 0.5f);
    data.Set("DefaultDQNAgent.intrinsic.rnd.statistics.whiten.obs_norm.pass_through", true);
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

TEST_CASE("DQN intrinsic known scalar keys stay NaN while disabled or suspended", "[rfd0116][intrinsic][rfd0117]")
{
    IntrinsicMetricsScope metrics;
    const auto mode = GENERATE(std::pair{false, true}, std::pair{false, false}, std::pair{true, false});
    const auto [enabled, learner_enabled] = mode;
    CAPTURE(enabled, learner_enabled);
    auto data = IntrinsicAgentData(enabled);
    data.Set("DefaultDQNAgent.learner.enabled", learner_enabled);
    anet::test::LogCaptureGuard logs;
    auto agent = MakeIntrinsicAgent(data);
    logs.Flush();
    CHECK(std::ranges::count_if(logs.Records(), [](const auto& record) {
        return record.level == wxLOG_Message
            && record.message.find("intrinsic scoring and training are suspended") != std::string::npos;
    }) == (enabled && !learner_enabled ? 1 : 0));
    for (int intake = 0; intake < 2; ++intake) {
        for (const auto* key : kAllIntrinsicScalarKeys) {
            CAPTURE(key, intake);
            const auto value = agent->GetScalar(key);
            REQUIRE(value.has_value());
            CHECK(std::isnan(*value));
        }
        CHECK_FALSE(agent->GetScalar(kUnknownIntrinsicScalarKey).has_value());
        agent->CreateLearner()->UpdateFromBatch({}, IntrinsicExperience(1, false, false, intake == 0));
    }
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

TEST_CASE("DQN validates the selected intrinsic observation only when enabled", "[rfd0116][intrinsic][config]")
{
    auto data = IntrinsicAgentData(false);
    data.Set("DefaultDQNAgent.intrinsic.observation", kInvalidConfigValue);
    CHECK_NOTHROW(anet::rl::dqn::DefaultDQNAgentConfig(data));
    data.Set("DefaultDQNAgent.intrinsic.enabled", true);
    CHECK_THROWS_WITH(anet::rl::dqn::DefaultDQNAgentConfig(data),
        Catch::Matchers::ContainsSubstring("intrinsic.observation")
        && Catch::Matchers::ContainsSubstring(kInvalidConfigValue));
}

TEST_CASE("DQN scales the intrinsic reward before combining it with external reward", "[rfd0121][intrinsic]")
{
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.statistics.whiten.obs_norm.pass_through", true);
    data.Set("DefaultDQNAgent.intrinsic.beta", 0.5f);
    data.Set("DefaultDQNAgent.intrinsic.reward_scaler.constant_scale", 2.0f);
    data.Set("DefaultDQNAgent.head.init.mode", kConstantInit);
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
    CHECK(*results.back()->GetScalar(kTdMean) == Catch::Approx(6.0f));
}

TEST_CASE("Intrinsic return scaling masks before statistics and stores components before beta", "[rfd0121][intrinsic][return_intake]")
{
    IntrinsicMetricsScope metrics;
    const bool decomposed = GENERATE(false, true);
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.value_mode", decomposed ? anet::rl::dqn::DefaultDQNAgentConfig::kValueModeDecomposed : anet::rl::dqn::DefaultDQNAgentConfig::kValueModeCombined);
    data.Set("DefaultDQNAgent.intrinsic.beta", 0.5f);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.rnd.statistics.whiten.obs_norm.pass_through", true);
    data.Set("DefaultDQNAgent.head.init.mode", kConstantInit);
    data.Set("DefaultDQNAgent.head.init.constant_val", 0.0f);
    data.Set("DefaultDQNAgent.intrinsic.value.head.init.mode", kConstantInit);
    data.Set("DefaultDQNAgent.intrinsic.value.head.init.constant_val", 0.0f);
    data.Set("DefaultDQNAgent.learner.learning_rate", 0.0f);
    data.Set("DefaultDQNAgent.learner.gamma", 0.5f);
    data.Set("DefaultDQNAgent.learner.use_n_step", false);
    data.Set("DefaultDQNAgent.learner.use_per", false);
    data.Set("DefaultDQNAgent.learner.update_warmup_steps", 0);
    data.Set("DefaultDQNAgent.learner.update_interval", 1);
    data.Set("DefaultDQNAgent.learner.replay_capacity", 4);
    data.Set("DefaultDQNAgent.learner.use_td_clip", false);
    data.Set("DefaultDQNAgent.reward_scaler.constant_scale", 2.0f);
    auto identity = MakeIntrinsicAgent(data);
    data.Set("DefaultDQNAgent.intrinsic.reward_scaler.use_dynamic_scaling", true);
    data.Set("DefaultDQNAgent.intrinsic.reward_scaler.dynamic_mode", anet::rl::RewardScaler::kReturnStdState);
    data.Set("DefaultDQNAgent.intrinsic.reward_scaler.epsilon", 0.01f);
    auto scaled = MakeIntrinsicAgent(data);
    anet::rl::RewardScalerConfig config;
    config.dynamic_mode = anet::rl::RewardScaler::kReturnStdState;
    config.use_auto_post_scale = false;
    config.epsilon = 0.01f;
    auto reference = anet::rl::RewardScalerFactory(config).CreateRewardScaler(0.5f, 2,
        anet::ConfigFullKeyTable::FormatKeyNotReadFromConfig("gamma"));
    torch::Tensor previous;
    bool previous_done = false;
    for (anet::rl::step_t step = 0; step < 8; ++step) {
        const bool done = step >= 4 && step % 2 == 0;
        auto experience = IntrinsicExperience(1, false, false, step == 0);
        experience.state.episode_start = torch::tensor({step == 0 || previous_done, step == 0}, torch::kBool);
        experience.next_state.obs.Set(kObservation, torch::ones({2, 2}) * std::pow(10.0f, float(step)));
        experience.next_state.done = torch::tensor({done, false}, torch::kBool);
        identity->CreateLearner()->UpdateFromBatch({.exp_step = step * 2}, experience);
        const float raw = *identity->GetScalar(kBonusMean) / (done ? 0.25f : 0.5f);
        const auto expected = reference->Scale(torch::tensor({done ? 0.0f : raw, raw}));
        const auto updates = scaled->CreateLearner()->UpdateFromBatch({.exp_step = step * 2}, experience);
        CHECK(*scaled->GetScalar(kBonusMean) == Catch::Approx(0.5f * expected.mean().item<float>()));
        CHECK(scaled->GetScalar(kRfd0121IntrinsicRewardScalerScale) == reference->GetScalar(anet::rl::RewardScaler::kKeyScale));
        if (step == 3 || step == 4) {
            REQUIRE_FALSE(updates.empty());
            CHECK(*updates.back()->GetScalar(kTdMean) == Catch::Approx(2.0f + (decomposed ? 1.0f : 0.5f) * previous.mean().item<float>()));
        }
        previous = expected;
        previous_done = done;
    }
}

TEST_CASE("DQN warms the intrinsic return scaler with raw RND errors", "[rfd0191][intrinsic][rnd_raw_reward]")
{
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.mode", kRawErrorMode);
    data.Set("DefaultDQNAgent.intrinsic.rnd.statistics.whiten.obs_norm.pass_through", true);
    data.Set("DefaultDQNAgent.intrinsic.reward_scaler.use_dynamic_scaling", true);
    data.Set("DefaultDQNAgent.intrinsic.reward_scaler.dynamic_mode", anet::rl::RewardScaler::kReturnStdState);
    data.Set("DefaultDQNAgent.intrinsic.reward_scaler.use_auto_post_scale", false);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.none.warmup_steps", 0);
    auto immediate = MakeIntrinsicAgent(data);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.none.warmup_steps", 4);
    auto delayed = MakeIntrinsicAgent(data);
    for (anet::rl::step_t step = 2; step <= 6; step += 2) {
        const auto experience = IntrinsicExperience(static_cast<float>(step), false, false, step == 2);
        immediate->CreateLearner()->UpdateFromBatch({.exp_step = step}, experience);
        delayed->CreateLearner()->UpdateFromBatch({.exp_step = step}, experience);
        CHECK(delayed->GetScalar(kRfd0121IntrinsicRewardScalerScale)
            == immediate->GetScalar(kRfd0121IntrinsicRewardScalerScale));
        const auto bonus = immediate->GetScalar(kBonusMean).value();
        CHECK(bonus > 0.0f);
        CHECK(delayed->GetScalar(kBonusMean).value() == (step <= 4 ? 0.0f : bonus));
    }
}

TEST_CASE("DQN learner consumes the scaled external reward plus intrinsic bonus", "[rfd0116][intrinsic]")
{
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.statistics.whiten.obs_norm.pass_through", true);
    data.Set("DefaultDQNAgent.intrinsic.beta", 0.5f);
    data.Set("DefaultDQNAgent.head.init.mode", kConstantInit);
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

TEST_CASE("DQN intrinsic done mask controls the actual learning reward", "[rfd0123][intrinsic]")
{
    SECTION("validates enabled values and notifies ignored state masking once") {
        const std::string key = "DefaultDQNAgent.intrinsic.use_done_mask";
        auto data = IntrinsicAgentData(false);
        data.Set(key, kInvalidConfigValue);
        CHECK_NOTHROW(anet::rl::dqn::DefaultDQNAgentConfig(data));
        data.Set("DefaultDQNAgent.intrinsic.enabled", true);
        CHECK_THROWS_WITH(anet::rl::dqn::DefaultDQNAgentConfig(data),
            Catch::Matchers::ContainsSubstring(key) && Catch::Matchers::ContainsSubstring(kInvalidConfigValue));
        const std::string source_key = "Trial.intrinsic.use_done_mask";
        anet::ConfigData source;
        source.Set(source_key, false);
        data.CopyValueFrom(key, source, source_key);
        data.Set("DefaultDQNAgent.intrinsic.observation", anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationState);
        anet::test::LogCaptureGuard logs;
        const anet::rl::dqn::DefaultDQNAgentConfig config(data);
        logs.Flush();
        CHECK(std::count_if(logs.Records().begin(), logs.Records().end(), [&](const auto& record) {
            return record.level == wxLOG_Warning && record.message.find(source_key) != std::string::npos;
        }) == 1);
        CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Warning,
            {source_key, kFalseConfigValue, anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationState}));
    }
    SECTION("the learner receives masked or unmasked bonuses on done and truncated lanes") {
        IntrinsicMetricsScope metrics;
        const bool state = GENERATE(false, true);
        const bool mask = GENERATE(false, true);
        const bool done = GENERATE(false, true);
        auto data = IntrinsicAgentData(true);
        data.Set("DefaultDQNAgent.intrinsic.use_done_mask", mask);
        data.Set("DefaultDQNAgent.intrinsic.observation", state ? anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationState
            : anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationNextState);
        data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
        data.Set("DefaultDQNAgent.intrinsic.rnd.statistics.whiten.obs_norm.pass_through", true);
        data.Set("DefaultDQNAgent.intrinsic.beta", 0.5f);
        data.Set("DefaultDQNAgent.head.init.mode", kConstantInit);
        data.Set("DefaultDQNAgent.head.init.constant_val", 0.0f);
        data.Set("DefaultDQNAgent.learner.learning_rate", 0.0f);
        data.Set("DefaultDQNAgent.learner.gamma", 0.0f);
        data.Set("DefaultDQNAgent.learner.use_n_step", false);
        data.Set("DefaultDQNAgent.learner.use_per", false);
        data.Set("DefaultDQNAgent.learner.update_warmup_steps", 0);
        data.Set("DefaultDQNAgent.learner.update_interval", 1);
        data.Set("DefaultDQNAgent.learner.replay_capacity", 8);
        data.Set("DefaultDQNAgent.learner.use_td_clip", false);
        data.Set("DefaultDQNAgent.reward_scaler.constant_scale", 2.0f);
        auto agent = MakeIntrinsicAgent(data);
        auto learner = agent->CreateLearner();
        const float bonus = !state && mask && done ? 0.0f : 2.0f;
        // 末尾の全経験を上限倍率へ揃え、初期の倍率1の経験を Replay から押し出す。
        // Q=0・gamma=0 なので、公開 TD 平均は Learner の受け取った報酬そのものになる。
        for (int step = 0; step < 8; ++step) {
            auto experience = IntrinsicExperience(std::pow(10.0f, static_cast<float>(step)), done, !done, true);
            if (state) experience.state.obs = experience.next_state.obs;
            const auto results = learner->UpdateFromBatch({.exp_step = static_cast<anet::rl::step_t>(step * 2)}, experience);
            if (step == 7) {
                REQUIRE_FALSE(results.empty());
                REQUIRE(results.back()->GetScalar(kTdMean).has_value());
                CHECK(*results.back()->GetScalar(kTdMean) == Catch::Approx(2.0f + bonus));
            }
        }
        CHECK(*agent->GetScalar(kBonusMean) == Catch::Approx(bonus));
    }
}

TEST_CASE("DQN checkpoint resumes dynamic reward scaler statistics", "[rfd0121][intrinsic][checkpoint]")
{
    // Agentの公開保存・自動読み込みを通し、統計が初期化されず続くことを確かめる。
    IntrinsicMetricsScope metrics;
    const auto outer_mode = GENERATE(std::string(anet::rl::RewardScaler::kRewardStdState), std::string(anet::rl::RewardScaler::kReturnStdState));
    const auto inner_mode = GENERATE(std::string(kRfd0121Off), std::string(anet::rl::RewardScaler::kConstantState), std::string(anet::rl::RewardScaler::kRewardStdState), std::string(anet::rl::RewardScaler::kReturnStdState));
    const bool decomposed = GENERATE(false, true);
    if (decomposed && inner_mode == kRfd0121Off) return;
    CAPTURE(outer_mode, inner_mode, decomposed);
    auto data = IntrinsicAgentData(inner_mode != kRfd0121Off);
    data.Set("DefaultDQNAgent.intrinsic.value_mode", decomposed ? anet::rl::dqn::DefaultDQNAgentConfig::kValueModeDecomposed : anet::rl::dqn::DefaultDQNAgentConfig::kValueModeCombined);
    data.Set("DefaultDQNAgent.reward_scaler.use_dynamic_scaling", true);
    data.Set("DefaultDQNAgent.reward_scaler.use_auto_post_scale", false);
    data.Set("DefaultDQNAgent.reward_scaler.dynamic_mode", outer_mode);
    data.Set("DefaultDQNAgent.intrinsic.reward_scaler.use_dynamic_scaling", inner_mode == anet::rl::RewardScaler::kRewardStdState || inner_mode == anet::rl::RewardScaler::kReturnStdState);
    if (inner_mode == anet::rl::RewardScaler::kRewardStdState || inner_mode == anet::rl::RewardScaler::kReturnStdState)
        data.Set("DefaultDQNAgent.intrinsic.reward_scaler.dynamic_mode", inner_mode);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    data.Set("DefaultDQNAgent.learner.update_warmup_steps", 100000);
    auto original = MakeIntrinsicAgent(data);
    auto first = IntrinsicExperience(1, false, false, true);
    first.reward = torch::tensor({1.0f, 3.0f});
    original->CreateLearner()->UpdateFromBatch({}, first);
    const auto path = std::filesystem::temp_directory_path() / ("anet-rfd0121-"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + kCheckpointExtension);
    IntrinsicCheckpointCleanup cleanup{.path = path};
    {
        std::ofstream stream(path, std::ios::binary);
        anet::OutputArchive archive(stream);
        original->Save(archive);
    }
    data.Set("DefaultDQNAgent.auto_load_file", path.string());
    {
        auto mismatch = data;
        mismatch.Set("DefaultDQNAgent.reward_scaler.use_dynamic_scaling", false);
        CHECK_THROWS_WITH(MakeIntrinsicAgent(mismatch), Catch::Matchers::ContainsSubstring("reward_scaler kind"));
        mismatch = data;
        mismatch.Set("DefaultDQNAgent.reward_scaler.dynamic_mode", outer_mode == anet::rl::RewardScaler::kRewardStdState ? anet::rl::RewardScaler::kReturnStdState : anet::rl::RewardScaler::kRewardStdState);
        CHECK_THROWS_WITH(MakeIntrinsicAgent(mismatch), Catch::Matchers::ContainsSubstring("reward_scaler kind"));
        if (inner_mode != kRfd0121Off) {
            mismatch = data;
            mismatch.Set("DefaultDQNAgent.intrinsic.value_mode", decomposed ? anet::rl::dqn::DefaultDQNAgentConfig::kValueModeCombined : anet::rl::dqn::DefaultDQNAgentConfig::kValueModeDecomposed);
            CHECK_THROWS_WITH(MakeIntrinsicAgent(mismatch), Catch::Matchers::ContainsSubstring("intrinsic.value_mode"));
            mismatch = data;
            mismatch.Set("DefaultDQNAgent.intrinsic.reward_scaler.use_dynamic_scaling", inner_mode == anet::rl::RewardScaler::kConstantState);
            CHECK_THROWS_WITH(MakeIntrinsicAgent(mismatch), Catch::Matchers::ContainsSubstring("intrinsic reward scaler kind"));
        }
        if (decomposed) {
            mismatch = data;
            mismatch.Set("DefaultDQNAgent.learner.intrinsic.feature_gradient", anet::rl::dqn::LearnerConfig::kFeatureGradientSplit);
            CHECK_THROWS_WITH(MakeIntrinsicAgent(mismatch), Catch::Matchers::ContainsSubstring("feature_gradient"));
        }
    }
    auto restored = MakeIntrinsicAgent(data);
    const auto key = anet::rl::RewardScaler::kKeyCount;
    REQUIRE(original->GetScalar(key));
    REQUIRE(restored->GetScalar(key));
    CHECK(*restored->GetScalar(key) == *original->GetScalar(key));
    if (inner_mode != kRfd0121Off)
        CHECK(original->GetScalar(kRfd0121IntrinsicRewardScalerScale) == restored->GetScalar(kRfd0121IntrinsicRewardScalerScale));
    auto next = IntrinsicExperience(1, false, false);
    next.reward = torch::tensor({2.0f, 6.0f});
    original->CreateLearner()->UpdateFromBatch({.exp_step = 2}, next);
    next.state.episode_start.fill_(true);
    restored->CreateLearner()->UpdateFromBatch({.exp_step = 2}, next);
    CHECK(original->GetScalar(anet::rl::RewardScaler::kKeyScale) == restored->GetScalar(anet::rl::RewardScaler::kKeyScale));
    if (inner_mode != kRfd0121Off)
        CHECK(original->GetScalar(kRfd0121IntrinsicRewardScalerScale) == restored->GetScalar(kRfd0121IntrinsicRewardScalerScale));
}

TEST_CASE("DQN checkpoint requires matching intrinsic presence and resumes scoring", "[rfd0116][intrinsic][checkpoint]")
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
    const auto path = std::filesystem::temp_directory_path() / ("anet-rfd0116-"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".anet");
    IntrinsicCheckpointCleanup cleanup{.path = path};
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
        const auto original_bonus = original->GetScalar(kBonusMean);
        const auto restored_bonus = restored->GetScalar(kBonusMean);
        REQUIRE(original_bonus.has_value());
        REQUIRE(restored_bonus.has_value());
        if (enabled) CHECK(*original_bonus == *restored_bonus);
        else {
            CHECK(std::isnan(*original_bonus));
            CHECK(std::isnan(*restored_bonus));
        }
    }
    if (enabled) {
        data.Set("DefaultDQNAgent.learner.enabled", false);
        anet::test::LogCaptureGuard logs;
        // checkpointを1 laneで読み、batch数を縮めず、同じ通知を2回起こす。
        for (int attempt = 0; attempt < 2; ++attempt) {
            const auto suspended = MakeIntrinsicAgent(data, 1);
            const anet::rl::BatchExperience experience{};
            const anet::rl::LearnEvent event{{.experience = experience,
                .counts = {.exp_step = 1}, .agent = suspended}};
            for (const auto* key : kAllIntrinsicScalarKeys) {
                CAPTURE(key, attempt);
                const auto value = suspended->GetScalar(key);
                REQUIRE(value.has_value());
                CHECK(std::isnan(*value));
                anet::rl::MetricsLogLearnObserver observer(key, key, anet::rl::StepAxis::EXP,
                    anet::rl::EventField::AGENT, 1, false, 0.01f, std::nullopt);
                observer.OnLearn(event);
            }
            CHECK_FALSE(suspended->GetScalar(kUnknownIntrinsicScalarKey).has_value());
        }
        logs.Flush();
        CHECK(std::ranges::count_if(logs.Records(), [](const auto& record) {
            return record.level == wxLOG_Message
                && record.message.find("intrinsic scoring and training are suspended") != std::string::npos;
        }) == 1);
        CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 0);
        data.Set("DefaultDQNAgent.learner.enabled", true);
    }
    data.Set("DefaultDQNAgent.intrinsic.enabled", !enabled);
    CHECK_THROWS_WITH(MakeIntrinsicAgent(data), Catch::Matchers::ContainsSubstring("Checkpoint header=")
        && Catch::Matchers::ContainsSubstring("intrinsic.enabled"));
    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        anet::OutputArchive archive(stream);
        anet::ArchiveHeader unknown;
        unknown.info = "unknown-rfd0116-header";
        archive.Write(unknown);
        archive.Write(std::string("{}"));
    }
    CHECK_THROWS_WITH(MakeIntrinsicAgent(data),
        Catch::Matchers::ContainsSubstring("Invalid DefaultDQN checkpoint header='unknown-rfd0116-header'")
        && Catch::Matchers::ContainsSubstring("expected"));
}

TEST_CASE("DQN intrinsic checkpoint resumes fractional credit and selected observations", "[rfd0116][intrinsic][rnd_schedule_checkpoint]")
{
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.learn.intake.ratio", 0.75f);
    data.Set("DefaultDQNAgent.intrinsic.learn.intake.batch_size", 1);
    auto original = MakeIntrinsicAgent(data);
    original->CreateLearner()->UpdateFromBatch({}, IntrinsicExperience(1, false, false, true));
    REQUIRE(*original->GetScalar(kIntrinsicLearnCount) == 1.0f);
    // 1.5 creditのうち1回を消費し、0.5と抽選済みの乱数状態を保存する。
    const auto path = std::filesystem::temp_directory_path() / ("anet-rfd0116-schedule-"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".anet");
    IntrinsicCheckpointCleanup cleanup{.path = path};
    {
        std::ofstream stream(path, std::ios::binary);
        anet::OutputArchive archive(stream);
        original->Save(archive);
    }
    data.Set("DefaultDQNAgent.auto_load_file", path.string());
    auto restored = MakeIntrinsicAgent(data);
    CHECK(original->GetScalar(kIntrinsicLearnCount) == restored->GetScalar(kIntrinsicLearnCount));
    for (anet::rl::step_t step = 1; step <= 8; ++step) {
        original->CreateLearner()->UpdateFromBatch({.exp_step = step * 2}, IntrinsicExperience(step + 1.0f, false, false));
        restored->CreateLearner()->UpdateFromBatch({.exp_step = step * 2}, IntrinsicExperience(step + 1.0f, false, false, step == 1));
        CHECK(original->GetScalar(kIntrinsicLearnCount) == restored->GetScalar(kIntrinsicLearnCount));
        CHECK(*original->GetScalar(kIntrinsicLearnCount) == static_cast<float>((step + 1) * 3 / 2));
        CHECK(original->GetScalar(kIntrinsicPredictorLoss) == restored->GetScalar(kIntrinsicPredictorLoss));
        CHECK(original->GetScalar(kIntrinsicErrorStep) == restored->GetScalar(kIntrinsicErrorStep));
        CHECK(original->GetScalar(kBonusMean) == restored->GetScalar(kBonusMean));
    }
}

TEST_CASE("DQN intrinsic consumes warmup credit and holds the last loss", "[rfd0116][intrinsic][rnd_schedule_edges]")
{
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.rnd.predictor.warmup_steps", 8);
    data.Set("DefaultDQNAgent.intrinsic.learn.intake.ratio", 0.5f);
    data.Set("DefaultDQNAgent.intrinsic.learn.intake.batch_size", 2);
    auto agent = MakeIntrinsicAgent(data);
    auto learner = agent->CreateLearner();
    CHECK(std::isnan(*agent->GetScalar(kIntrinsicPredictorLoss)));
    float last_loss = 0;
    for (anet::rl::step_t step = 1; step <= 8; ++step) {
        learner->UpdateFromBatch({.exp_step = step * 2}, IntrinsicExperience(step, false, false, step == 1));
        CHECK(*agent->GetScalar(kIntrinsicLearnCount) == static_cast<float>(step <= 4 ? 0 : (step - 4) / 2));
        const auto loss = *agent->GetScalar(kIntrinsicPredictorLoss);
        if (step < 6) CHECK(std::isnan(loss));
        else {
            CHECK(std::isfinite(loss));
            if (step == 7) CHECK(loss == last_loss);
        }
        last_loss = loss;
    }
}

TEST_CASE("DQN intrinsic validates schedule settings only when enabled", "[rfd0116][intrinsic][config][rnd_schedule_edges]")
{
    const auto invalid = GENERATE(Catch::Generators::values<std::pair<std::string, std::string>>({
        {"learn.intake.ratio", "0"}, {"learn.intake.ratio", "-0.1"}, {"learn.intake.ratio", "1.01"},
        {"learn.intake.ratio", "nan"}, {"learn.intake.ratio", "inf"},
        {"learn.intake.batch_size", "0"}, {"learn.intake.batch_size", "-1"}, {"learn.intake.batch_size", "1.5"}}));
    auto data = IntrinsicAgentData(false);
    const auto key = std::string(kIntrinsicPrefix) + "." + invalid.first;
    data.Set(key, invalid.second);
    CAPTURE(key, invalid.second);
    CHECK_NOTHROW(anet::rl::dqn::DefaultDQNAgentConfig(data));
    data.Set("DefaultDQNAgent.intrinsic.enabled", true);
    CHECK_THROWS_WITH(anet::rl::dqn::DefaultDQNAgentConfig(data),
        Catch::Matchers::ContainsSubstring("Invalid " + key + "=" + invalid.second + "; expected ")
        || Catch::Matchers::ContainsSubstring("key=" + key + " value=\"" + invalid.second + "\" expected="));
}

static std::string IntrinsicCheckpointPayload(const anet::rl::dqn::DefaultDQNAgent& agent,
    const std::string& format = kRndCheckpointFormat)
{
    std::ostringstream saved(std::ios::binary);
    anet::OutputArchive output(saved);
    agent.Save(output);
    std::ostringstream marker(std::ios::binary);
    anet::OutputArchive marker_archive(marker);
    marker_archive.Write(format);
    const auto bytes = saved.str();
    const auto offset = bytes.find(marker.str());
    REQUIRE(offset != std::string::npos);
    return bytes.substr(offset);
}

TEST_CASE("RND rollout checkpoint resumes the count and starts filling an empty rollout",
    "[rfd0191][intrinsic][rnd_rollout_checkpoint]")
{
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.learn.mode", anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicLearnRollout);
    data.Set("DefaultDQNAgent.intrinsic.learn.rollout.length", 4);
    data.Set("DefaultDQNAgent.intrinsic.learn.rollout.epochs", 2);
    data.Set("DefaultDQNAgent.intrinsic.learn.rollout.batch_size", 4);
    data.Set("DefaultDQNAgent.intrinsic.learn.rollout.use_spread", true);
    auto original = MakeIntrinsicAgent(data);
    for (anet::rl::step_t step = 1; step <= 6; ++step)
        original->CreateLearner()->UpdateFromBatch({.exp_step = step * 2},
            IntrinsicExperience(static_cast<float>(step), false, false, step == 1));
    REQUIRE(original->GetScalar(kIntrinsicLearnCount).value() == 1.0f);
    const auto path = std::filesystem::temp_directory_path() / ("anet-rfd0191-rollout-"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + kCheckpointExtension);
    IntrinsicCheckpointCleanup cleanup{.path = path};
    {
        std::ofstream stream(path, std::ios::binary);
        anet::OutputArchive archive(stream);
        original->Save(archive);
    }
    data.Set("DefaultDQNAgent.auto_load_file", path.string());
    auto restored = MakeIntrinsicAgent(data);
    CHECK(restored->GetScalar(kIntrinsicLearnCount) == original->GetScalar(kIntrinsicLearnCount));
    const auto read_schedule = [&](const auto& agent) {
        std::istringstream stream(IntrinsicCheckpointPayload(agent), std::ios::binary);
        anet::InputArchive archive(stream);
        auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            RndTestConfig(), kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
        estimator->Load(archive);
        float credit;
        std::string random_state;
        int64_t learn_count;
        archive.Read(credit);
        archive.Read(random_state);
        archive.Read(learn_count);
        return std::tuple{credit, random_state, learn_count};
    };
    CHECK(read_schedule(*restored) == read_schedule(*original));
    CHECK(std::get<0>(read_schedule(*restored)) == 0.0f);
    for (anet::rl::step_t step = 1; step <= 8; ++step) {
        restored->CreateLearner()->UpdateFromBatch({.exp_step = step * 2},
            IntrinsicExperience(static_cast<float>(step + 6), false, false, step == 1));
        CHECK(restored->GetScalar(kIntrinsicLearnCount).value()
            == 1.0f + static_cast<float>(step <= 4 ? 0 : (step - 4) / 2));
    }
}

static std::string RndPredictorBytes(std::istream& stream)
{
    anet::InputArchive archive(stream);
    std::string format, saved_mode, target, predictor;
    archive.Read(format);
    REQUIRE(format == kRndCheckpointFormat);
    archive.Read(saved_mode);
    archive.Read(target);
    archive.Read(predictor);
    return predictor;
}

TEST_CASE("DQN validates RND rollout settings and only checks lane combinations when selected",
    "[rfd0191][intrinsic][rnd_rollout_config]")
{
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    SECTION("Each configured leaf is validated even under intake")
    {
        const auto [key, value] = GENERATE(
            std::pair{"learn.mode", "invalid"}, std::pair{"learn.rollout.length", "0"},
            std::pair{"learn.rollout.epochs", "0"}, std::pair{"learn.rollout.batch_size", "-1"},
            std::pair{"learn.rollout.ratio", "nan"}, std::pair{"learn.rollout.ratio", "4.01"});
        const auto full_key = std::string(kIntrinsicPrefix) + "." + key;
        data.Set(full_key, value);
        CHECK_THROWS_WITH(MakeIntrinsicAgent(data),
            Catch::Matchers::ContainsSubstring(full_key) && Catch::Matchers::ContainsSubstring(value));
        data.Set("DefaultDQNAgent.intrinsic.enabled", false);
        CHECK_NOTHROW(MakeIntrinsicAgent(data));
    }
    SECTION("Selected rollout must yield an integer number of steps per epoch")
    {
        const bool fractional = GENERATE(false, true);
        data.Set("DefaultDQNAgent.intrinsic.learn.rollout.length", 4);
        data.Set("DefaultDQNAgent.intrinsic.learn.rollout.epochs", 3);
        data.Set("DefaultDQNAgent.intrinsic.learn.rollout.batch_size", fractional ? 3 : 8);
        CHECK_NOTHROW(MakeIntrinsicAgent(data, 4));
        data.Set("DefaultDQNAgent.intrinsic.learn.mode", anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicLearnRollout);
        CHECK_THROWS_WITH(MakeIntrinsicAgent(data, 4),
            Catch::Matchers::ContainsSubstring("num_envs=4")
            && Catch::Matchers::ContainsSubstring("DefaultDQNAgent.intrinsic.learn.rollout.length=4")
            && Catch::Matchers::ContainsSubstring("DefaultDQNAgent.intrinsic.learn.rollout.ratio=1")
            && Catch::Matchers::ContainsSubstring(fractional ? "learn.rollout.batch_size=3 gives 5.333"
                : "learn.rollout.batch_size=8 gives 2")
            && Catch::Matchers::ContainsSubstring(fractional ? "positive integer"
                : "expected divisibility by DefaultDQNAgent.intrinsic.learn.rollout.epochs=3"));
        data.Set("DefaultDQNAgent.learner.enabled", false);
        CHECK_NOTHROW(MakeIntrinsicAgent(data, 4));
    }
    SECTION("Rollout validates the unused intake ratio but accepts a batch larger than the lane count")
    {
        data.Set("DefaultDQNAgent.intrinsic.learn.rollout.length", 4);
        data.Set("DefaultDQNAgent.intrinsic.learn.rollout.epochs", 2);
        data.Set("DefaultDQNAgent.intrinsic.learn.rollout.batch_size", 4);
        data.Set("DefaultDQNAgent.intrinsic.learn.intake.batch_size", 32);
        CHECK_THROWS_WITH(MakeIntrinsicAgent(data, 2),
            Catch::Matchers::ContainsSubstring("DefaultDQNAgent.intrinsic.learn.intake.batch_size=32")
            && Catch::Matchers::ContainsSubstring("num_envs=2"));
        data.Set("DefaultDQNAgent.intrinsic.learn.mode", anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicLearnRollout);
        CHECK_NOTHROW(MakeIntrinsicAgent(data, 2));
        data.Set("DefaultDQNAgent.intrinsic.learn.intake.ratio", 0);
        CHECK_THROWS_WITH(MakeIntrinsicAgent(data, 2),
            Catch::Matchers::ContainsSubstring("DefaultDQNAgent.intrinsic.learn.intake.ratio=0"));
    }
}

TEST_CASE("DQN intrinsic learns complete lane groups and rejects oversize batches", "[rfd0116][intrinsic][rnd_sampling][rfd0117]")
{
    IntrinsicMetricsScope metrics;
    const auto input_device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (input_device.is_cuda() && !torch::cuda::is_available()) { SKIP("CUDA is unavailable"); }
    const auto dimensions = GENERATE(Catch::Generators::values<std::pair<int, int>>({
        {2, 32}, {2, 2}, {3, 2}, {4, 2}, {4, 1}}));
    const auto [num_lanes, configured_batch] = dimensions;
    const int batch_size = configured_batch;
    CAPTURE(num_lanes, configured_batch, input_device);
    auto data = IntrinsicAgentData(true);
    const auto observation = GENERATE(anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationNextState, anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationState);
    data.Set("DefaultDQNAgent.intrinsic.observation", observation);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.learn.intake.ratio", 1.0f);
    data.Set("DefaultDQNAgent.intrinsic.learn.intake.batch_size", configured_batch);
    anet::test::LogCaptureGuard logs;
    if (configured_batch > num_lanes) {
        CHECK_THROWS_WITH(MakeIntrinsicAgent(data, num_lanes),
            Catch::Matchers::ContainsSubstring("DefaultDQNAgent.intrinsic.learn.intake.batch_size=32")
            && Catch::Matchers::ContainsSubstring("num_envs=2"));
        data.Set("DefaultDQNAgent.learner.enabled", false);
        CHECK_NOTHROW(MakeIntrinsicAgent(data, num_lanes));
        data.Set("DefaultDQNAgent.intrinsic.enabled", false);
        data.Set("DefaultDQNAgent.learner.enabled", true);
        CHECK_NOTHROW(MakeIntrinsicAgent(data, num_lanes));
        logs.Flush();
        CHECK(std::ranges::none_of(logs.Records(), [](const auto& record) {
            return record.level == wxLOG_Warning && record.message.find("exceeds num_envs=") != std::string::npos;
        }));
        return;
    }
    auto agent = MakeIntrinsicAgent(data, num_lanes);
    const auto flags = torch::zeros({num_lanes}, torch::kBool);
    const anet::TensorDict obs(kObservation,
        torch::tensor({{1.0f, 2.0f}, {3.0f, -1.0f}, {-4.0f, 6.0f}, {8.0f, 3.0f}}).narrow(0, 0, num_lanes));
    const anet::rl::BatchExperience experience{
        anet::rl::BatchState(anet::TensorDict(kObservation, obs.At(kObservation) * 10.0f),
            flags, flags, torch::ones({num_lanes}, torch::kBool)),
        std::make_shared<anet::rl::BatchActionInfo>(torch::zeros({num_lanes}, torch::kLong)),
        torch::ones({num_lanes}), anet::rl::BatchState(obs, flags, flags, flags)};
    agent->CreateLearner()->UpdateFromBatch({}, experience.To(input_device));
    CHECK(*agent->GetScalar(kIntrinsicLearnCount) == static_cast<float>(num_lanes / batch_size));
    std::istringstream actual_stream(IntrinsicCheckpointPayload(*agent), std::ios::binary);
    const auto actual_predictor = RndPredictorBytes(actual_stream);
    const auto& selected_obs = data.Get("DefaultDQNAgent.intrinsic.observation", anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationNextState) == anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationState
        ? experience.state.obs : experience.next_state.obs;

    // 抽選アルゴリズムやseed列を写さず、許される全順列の重みと照合する。
    // k件全ての入力・step間の非重複・端数laneの不使用を同時に確かめる。
    std::vector<int64_t> lanes(num_lanes);
    std::iota(lanes.begin(), lanes.end(), 0);
    bool matched = false;
    do {
        auto reference = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            data, std::string(kIntrinsicPrefix) + ".rnd", RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
        reference->Estimate(selected_obs);
        for (int offset = 0; offset + batch_size <= num_lanes; offset += batch_size) {
            const std::vector<int64_t> selected(lanes.begin() + offset, lanes.begin() + offset + batch_size);
            REQUIRE(reference->Learn(anet::TensorDict(kObservation,
                selected_obs.At(kObservation).index_select(0, torch::tensor(selected, torch::kInt64)))));
        }
        std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
        anet::OutputArchive archive(saved);
        reference->Save(archive);
        matched = actual_predictor == RndPredictorBytes(saved);
        if (matched) break;
    } while (std::next_permutation(lanes.begin(), lanes.end()));
    CHECK(matched);

    // N=3,k=2では毎回1.5獲得して1だけ消費し、整数分も保存する。
    auto continuation = experience;
    continuation.state.episode_start = flags;
    for (int step = 1; step < 3; ++step)
        agent->CreateLearner()->UpdateFromBatch({}, continuation.To(input_device));
    std::istringstream saved(IntrinsicCheckpointPayload(*agent), std::ios::binary);
    anet::InputArchive archive(saved);
    auto restored = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, std::string(kIntrinsicPrefix) + ".rnd", RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    restored->Load(archive);
    float credit;
    archive.Read(credit);
    CHECK(credit == (num_lanes == 3 ? 1.5f : 0.0f));
    CHECK(*agent->GetScalar(kIntrinsicLearnCount) == static_cast<float>(3 * (num_lanes / batch_size)));
    logs.Flush();
    const auto warnings = std::count_if(logs.Records().begin(), logs.Records().end(), [](const auto& record) {
        return record.message.find("exceeds num_envs=") != std::string::npos;
    });
    CHECK(warnings == 0);
}

static std::shared_ptr<anet::rl::Actor> MakeIntrinsicTraceActor(
    const std::shared_ptr<anet::rl::dqn::DefaultDQNAgent>& agent, int num_lanes = 2)
{
    anet::rl::EnvSpec spec;
    spec.state_spec.obs_spec = RndTestSpec();
    spec.action_spec.is_discrete = true;
    spec.action_spec.value_labels = {kTraceActionLabel};
    spec.reward_range = {-1000.0f, 1000.0f};
    return agent->CreateActor({.batch_env_spec = {.num_envs = num_lanes, .num_threads = 1},
        .env_spec = spec, .device = torch::kCPU, .seed = 42, .actor_key = kTraceActorKey});
}

TEST_CASE("Raw RND wait keeps both reward paths at zero after updating the scaler",
    "[rfd0191][intrinsic][rnd_raw_trace]")
{
    IntrinsicMetricsScope metrics;
    const bool decomposed = GENERATE(false, true);
    constexpr float kBeta = 0.25f;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.value_mode", decomposed ? "decomposed" : "combined");
    data.Set("DefaultDQNAgent.intrinsic.beta", kBeta);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.mode", kRawErrorMode);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.none.warmup_steps", 4);
    data.Set("DefaultDQNAgent.intrinsic.rnd.statistics.whiten.obs_norm.pass_through", true);
    data.Set("DefaultDQNAgent.intrinsic.learn.intake.ratio", 1e-6f);
    data.Set("DefaultDQNAgent.intrinsic.use_done_mask", false);
    data.Set("DefaultDQNAgent.intrinsic.reward_scaler.use_dynamic_scaling", true);
    data.Set("DefaultDQNAgent.intrinsic.reward_scaler.dynamic_mode", anet::rl::RewardScaler::kReturnStdState);
    data.Set("DefaultDQNAgent.intrinsic.reward_scaler.use_auto_post_scale", false);
    data.Set("DefaultDQNAgent.learner.use_n_step", false);
    data.Set("DefaultDQNAgent.learner.use_per", false);
    data.Set("DefaultDQNAgent.actor.[trace].nn_forward_key", kTraceForwardKey);
    data.Set("DefaultDQNAgent.actor.[trace].clone_model", true);
    auto agent = MakeIntrinsicAgent(data);
    auto actor = MakeIntrinsicTraceActor(agent);
    auto provider = agent->CreateStepDiagnosticsProvider({.num_lanes = 2,
        .keys = {kStepIntrinsicError, kStepIntrinsicNormalizedError, kStepIntrinsicBonus}, .actor = actor});
    const auto learner = agent->CreateLearner();
    std::vector<float> expected;
    for (anet::rl::step_t step = 1; step <= 3; ++step) {
        auto sample = IntrinsicExperience(static_cast<float>(step + 1), false, false, step == 1);
        sample.action = actor->MakeAction({}, sample.state);
        actor->Sync();
        const auto before = provider->Observe(sample);
        for (int lane = 0; lane < 2; ++lane) {
            const auto error = before.current->GetScalar(kStepIntrinsicError, lane).value();
            CHECK(before.current->GetScalar(kStepIntrinsicNormalizedError, lane).value() == error);
            if (step <= 2) CHECK(before.current->GetScalar(kStepIntrinsicBonus, lane).value() == 0.0f);
        }
        learner->UpdateFromBatch({.exp_step = step * 2}, sample);
        if (step <= 2) {
            CHECK(agent->GetScalar(kBonusMean).value() == 0.0f);
        } else {
            actor->Sync();
            const auto after = provider->Observe(sample);
            const auto scale = agent->GetScalar(kRfd0121IntrinsicRewardScalerScale).value();
            for (int lane = 0; lane < 2; ++lane) {
                const auto error = before.current->GetScalar(kStepIntrinsicError, lane).value();
                const auto scaled = ((1.0f + error) - 1.0f) * scale;
                CHECK(after.current->GetScalar(kStepIntrinsicBonus, lane).value() == Catch::Approx(kBeta * scaled));
                expected.push_back(decomposed ? scaled : kBeta * scaled);
            }
        }
    }
    // 末尾を閉じて、wait 中の 2 intake と、明けた直後の 1 intake を各 lane から読む。
    auto tail = IntrinsicExperience(8.0f, true, false);
    tail.action = actor->MakeAction({}, tail.state);
    learner->UpdateFromBatch({.exp_step = 8}, tail);
    const auto returns = agent->GetTensor(anet::rl::ReplayBuffer::TARGET_RETURN);
    REQUIRE(returns);
    const auto reward = decomposed ? returns->select(-1, 1) : *returns - 1.0f;
    CHECK(reward.eq(0.0f).sum().item<int64_t>() == 4);
    for (const auto value : expected) {
        CAPTURE(value);
        CHECK((reward - value).abs().lt(1e-5).any().item<bool>());
    }
}

TEST_CASE("Decomposed Q-only Trace avoids RND followers and scoring", "[rfd0121][intrinsic][rfd0121-review-trace]")
{
    IntrinsicMetricsScope metrics;
    const bool with_actor = GENERATE(false, true);
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.value_mode", anet::rl::dqn::DefaultDQNAgentConfig::kValueModeDecomposed);
    data.Set("DefaultDQNAgent.actor.[trace].nn_forward_key", kTraceForwardKey);
    data.Set("DefaultDQNAgent.actor.[trace].clone_model", true);
    data.Set("DefaultDQNAgent.learner.update_warmup_steps", 100000);
    auto agent = MakeIntrinsicAgent(data);
    auto actor = MakeIntrinsicTraceActor(agent);
    // Actorを渡さないQ専用の要求も通り、RNDのsync followerを接続しない。
    auto trace = agent->CreateStepDiagnosticsProvider({.num_lanes = 2,
        .keys = {kRfd0121ExtrinsicQSa, kRfd0121IntrinsicQSa, kRfd0121ExtrinsicStateValue, kRfd0121IntrinsicStateValue},
        .actor = with_actor ? actor : nullptr});
    CHECK_FALSE(trace->Resume());
    for (anet::rl::step_t step = 0; step < 4; ++step) {
        auto experience = IntrinsicExperience(2.0f, false, false, step == 0);
        experience.action = actor->MakeAction({.train_step = step}, experience.state);
        const auto result = trace->Observe(experience);
        for (int lane = 0; lane < 2; ++lane) {
            CHECK(std::isfinite(*result.current->GetScalar(kRfd0121ExtrinsicQSa, lane)));
            CHECK(std::isfinite(*result.current->GetScalar(kRfd0121IntrinsicQSa, lane)));
            CHECK(std::isnan(*result.current->GetScalar(kStepIntrinsicError, lane)));
        }
        // 未要求の内的報酬からpriorityを作るために採点しない。
        CHECK(result.completed.empty());
    }
}

TEST_CASE("Decomposed AMP updates both heads with merge and split", "[rfd0121][intrinsic][amp]")
{
    if (!torch::cuda::is_available()) {
        WARN("CUDA unavailable; decomposed AMP validation not executed.");
        return;
    }
    IntrinsicMetricsScope metrics;
    const bool bf16 = GENERATE(false, true);
    const bool split = GENERATE(false, true);
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.value_mode", anet::rl::dqn::DefaultDQNAgentConfig::kValueModeDecomposed);
    data.Set("DefaultDQNAgent.learner.intrinsic.feature_gradient", split ? anet::rl::dqn::LearnerConfig::kFeatureGradientSplit : anet::rl::dqn::LearnerConfig::kFeatureGradientMerge);
    data.Set("DefaultDQNAgent.learner.use_amp", true);
    data.Set("DefaultDQNAgent.learner.use_amp_bf16", bf16);
    data.Set("DefaultDQNAgent.learner.use_fused_optimizer", true);
    data.Set("DefaultDQNAgent.learner.use_n_step", false);
    data.Set("DefaultDQNAgent.learner.use_per", false);
    data.Set("DefaultDQNAgent.learner.update_warmup_steps", 0);
    data.Set("DefaultDQNAgent.learner.update_interval", 1);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    const torch::Device device(torch::kCUDA, 0);
    auto agent = MakeIntrinsicAgent(data, 2, device);
    const auto forward = agent->GetTensorDictFunction(kRfd0121PolicyNetForward);
    REQUIRE(forward);
    const auto observations = RndTestObservation().To(device);
    const auto before = (*forward)(observations).Clone();
    for (anet::rl::step_t step = 0; step < 12; ++step)
        agent->CreateLearner()->UpdateFromBatch({.exp_step = step * 2}, IntrinsicExperience(1 + float(step), false, false, step == 0));
    const auto after = (*forward)(observations);
    CAPTURE(bf16, split);
    CHECK_FALSE(torch::equal(before.At(kRfd0121ExtrinsicQ), after.At(kRfd0121ExtrinsicQ)));
    CHECK_FALSE(torch::equal(before.At(kRfd0121IntrinsicQ), after.At(kRfd0121IntrinsicQ)));
    for (const auto* key : {kRfd0121ExtrinsicLoss, kRfd0121IntrinsicLoss, kRfd0121ExtrinsicHeadGradNorm, kRfd0121IntrinsicHeadGradNorm}) {
        const auto value = agent->GetScalar(key);
        REQUIRE(value);
        CHECK(std::isfinite(*value));
    }
    for (const auto* key : {kRfd0121ExtrinsicGradNorm, kRfd0121IntrinsicGradNorm}) {
        const auto value = agent->GetScalar(key);
        REQUIRE(value);
        CHECK(std::isfinite(*value) == split);
    }
}

TEST_CASE("Decomposed Agent selects fixed-zero and learner beta per Actor", "[rfd0121][intrinsic][actor_beta]")
{
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.value_mode", anet::rl::dqn::DefaultDQNAgentConfig::kValueModeDecomposed);
    data.Set("DefaultDQNAgent.intrinsic.value.head.use_dueling", false);
    data.Set("DefaultDQNAgent.intrinsic.beta", 10.0f);
    for (const auto* name : {kRfd0121FixedActorKey, kRfd0121Learner}) {
        const auto prefix = std::string("DefaultDQNAgent.actor.[") + name + "]";
        data.Set(prefix + ".nn_forward_key", kTraceForwardKey);
        data.Set(prefix + ".network", "online");
        data.Set(prefix + ".clone_model", false);
        data.Set(prefix + ".policy.policy_type", anet::rl::dqn::kActionPolicyTypeStr_Greedy);
    }
    data.Set("DefaultDQNAgent.actor.[fixed].intrinsic.action_beta_mode", anet::rl::dqn::DQNActorConfig::kActionBetaFixed);
    data.Set("DefaultDQNAgent.actor.[fixed].intrinsic.action_beta", 0.0f);
    anet::rl::EnvSpec spec;
    spec.state_spec.obs_spec = RndTestSpec();
    spec.action_spec.is_discrete = true;
    spec.action_spec.value_labels = {"a0", "a1"};
    spec.reward_range = {-1000.0f, 1000.0f};
    const auto net = MakeIntrinsicNetworkConfig();
    auto agent = std::make_shared<anet::rl::dqn::DefaultDQNAgent>(anet::rl::dqn::DefaultDQNAgentConfig(data),
        net, anet::rl::BatchEnvSpec{2, 1}, spec, torch::kCPU, 123);
    const auto angles = torch::arange(64, torch::kFloat32) * (6.283185307f / 64);
    const anet::TensorDict observations{{kObservation, torch::stack({angles.cos(), angles.sin()}, 1)}};
    const auto forward = agent->GetTensorDictFunction(kRfd0121PolicyNetForward);
    REQUIRE(forward);
    const auto output = (*forward)(observations);
    const auto external = output.At(kRfd0121ExtrinsicQ).argmax(1);
    const auto combined = output.At(kRfd0121Q).argmax(1);
    REQUIRE(external.ne(combined).any().item<bool>());
    for (const auto* name : {kRfd0121FixedActorKey, kRfd0121Learner}) {
        const auto actor = agent->CreateActor({.batch_env_spec = {64, 1}, .env_spec = spec,
            .device = torch::kCPU, .seed = 123, .actor_key = name});
        const auto flags = torch::zeros({64}, torch::kBool);
        const auto info = actor->MakeAction({}, anet::rl::BatchState(observations, flags, flags, ~flags));
        CHECK(torch::equal(info->GetAction(), std::string(name) == kRfd0121FixedActorKey ? external : combined));
    }
}

TEST_CASE("DQN decomposed values learn both heads through Actor and Replay", "[rfd0121][intrinsic]")
{
    // 公開の設定・Actor・取り込み・Replay・更新をつないだ最初の検査。
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.value_mode", anet::rl::dqn::DefaultDQNAgentConfig::kValueModeDecomposed);
    data.Set("DefaultDQNAgent.intrinsic.value.head.use_dueling", false);
    data.Set("DefaultDQNAgent.intrinsic.value.head.projection.noisy.enabled", true);
    data.Set("DefaultDQNAgent.actor.[trace].nn_forward_key", kTraceForwardKey);
    data.Set("DefaultDQNAgent.actor.[trace].policy.policy_type", anet::rl::dqn::kActionPolicyTypeStr_Greedy);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    data.Set("DefaultDQNAgent.learner.update_warmup_steps", 0);
    data.Set("DefaultDQNAgent.learner.update_interval", 1);
    data.Set("DefaultDQNAgent.learner.use_n_step", false);
    data.Set("DefaultDQNAgent.learner.use_per", false);
    data.Set("DefaultDQNAgent.learner.replay_fit.probe.batch_size", 1);
    data.Set("DefaultDQNAgent.learner.policy_churn.probe.batch_size", 1);
    data.Set("DefaultDQNAgent.learner.noisy_probe.probe.batch_size", 1);
    data.Set("DefaultDQNAgent.learner.noisy_probe.num_draws", 2);
    auto agent = MakeIntrinsicAgent(data);
    auto actor = MakeIntrinsicTraceActor(agent);
    auto learner = agent->CreateLearner();
    agent->ConfigureScalarMetricSubscriptions({anet::rl::ScalarMetricSubscription{
        .source_key = kRfd0121ReplayFitUniformTdMean, .event = anet::rl::EventType::LEARN,
        .target = anet::rl::EventField::UPDATE_RESULT, .interval = 1, .scope = anet::rl::RunnerScope::TRAIN},
        anet::rl::ScalarMetricSubscription{.source_key = kRfd0121PolicyChurnQDeltaAbsMean, .event = anet::rl::EventType::LEARN,
        .target = anet::rl::EventField::UPDATE_RESULT, .interval = 1, .scope = anet::rl::RunnerScope::TRAIN},
        anet::rl::ScalarMetricSubscription{.source_key = kRfd0121NoisyQAbsDiff, .event = anet::rl::EventType::LEARN,
        .target = anet::rl::EventField::UPDATE_RESULT, .interval = 1, .scope = anet::rl::RunnerScope::TRAIN}});
    float replay_fit = std::numeric_limits<float>::quiet_NaN();
    float churn = std::numeric_limits<float>::quiet_NaN();
    float noisy = std::numeric_limits<float>::quiet_NaN();
    for (int step = 0; step < 6; ++step) {
        const anet::rl::StepCounts counts{.exp_step = static_cast<anet::rl::step_t>(step * 2)};
        auto experience = IntrinsicExperience(1.0f + step, false, false, step == 0);
        experience.action = actor->MakeAction(counts, experience.state);
        const auto updates = learner->UpdateFromBatch(counts, experience);
        if (!updates.empty()) {
            const auto value = updates.back()->GetScalar(kRfd0121ReplayFitUniformTdMean);
            if (value) replay_fit = *value;
            const auto churn_value = updates.back()->GetScalar(kRfd0121PolicyChurnQDeltaAbsMean);
            if (churn_value) churn = *churn_value;
            const auto noisy_value = updates.back()->GetScalar(kRfd0121NoisyQAbsDiff);
            if (noisy_value) noisy = *noisy_value;
        }
    }
    CHECK(std::isfinite(replay_fit));
    CHECK(std::isfinite(churn));
    CHECK(std::isfinite(noisy));
    CHECK(noisy > 0.0f);
    const auto forward = agent->GetTensorDictFunction(kRfd0121PolicyNetForward);
    REQUIRE(forward);
    const auto values = (*forward)(IntrinsicExperience(2.0f, false, false).state.obs);
    CHECK(torch::allclose(values.At(kRfd0121Q), values.At(kRfd0121ExtrinsicQ) + 0.1f * values.At(kRfd0121IntrinsicQ)));
    auto trace = agent->CreateStepDiagnosticsProvider({.num_lanes = 2,
        .keys = {kRfd0121ApproxPriority, kRfd0121ExtrinsicQSa, kRfd0121IntrinsicQSa,
            kRfd0121ExtrinsicStateValue, kRfd0121IntrinsicStateValue}, .actor = actor});
    auto experience = IntrinsicExperience(2.0f, false, false, true);
    experience.action = actor->MakeAction({}, experience.state);
    const auto row = trace->Observe(experience);
    for (const auto* key : {kRfd0121ExtrinsicQSa, kRfd0121IntrinsicQSa, kRfd0121ExtrinsicStateValue, kRfd0121IntrinsicStateValue}) {
        const auto value = row.current->GetScalar(key, 0);
        CAPTURE(key);
        REQUIRE(value);
        CHECK(std::isfinite(*value));
    }
    auto next = IntrinsicExperience(3.0f, false, false, false);
    next.action = actor->MakeAction({.exp_step = 2}, next.state);
    const auto following = trace->Observe(next);
    REQUIRE_FALSE(following.completed.empty());
    for (const auto& completed : following.completed) {
        REQUIRE(completed.values->GetScalar(kRfd0121ApproxPriority, 0));
        CHECK(std::isfinite(*completed.values->GetScalar(kRfd0121ApproxPriority, 0)));
    }
    for (const auto* key : {kRfd0121ExtrinsicLoss, kRfd0121IntrinsicLoss, kRfd0121ExtrinsicQSaMean, kRfd0121IntrinsicQSaMean,
        kRfd0121ExtrinsicHeadGradNorm, kRfd0121IntrinsicHeadGradNorm, kRfd0121IntrinsicQShare}) {
        const auto value = agent->GetScalar(key);
        CAPTURE(key);
        REQUIRE(value);
        CHECK(std::isfinite(*value));
    }
}

TEST_CASE("DQN trace scores its Actor snapshot after synchronization", "[rfd0127][rfd0129][intrinsic]")
{
    // 公開のAgent・Actor・step診断を通して、同期から列の読み出しまでを接続する。
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.actor.[trace].nn_forward_key", kTraceForwardKey);
    data.Set("DefaultDQNAgent.actor.[trace].policy.policy_type", anet::rl::dqn::kActionPolicyTypeStr_Greedy);
    data.Set("DefaultDQNAgent.actor.[trace].clone_model", true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    data.Set("DefaultDQNAgent.learner.update_warmup_steps", 100000);
    auto agent = MakeIntrinsicAgent(data);
    auto actor = MakeIntrinsicTraceActor(agent);
    auto diagnostics = agent->CreateStepDiagnosticsProvider({.num_lanes = 2,
        .keys = {kStepIntrinsicError, kStepIntrinsicNormalizedError, kStepIntrinsicBonus}, .actor = actor});
    auto experience = IntrinsicExperience(2.0f, false, false, true);
    experience.action = actor->MakeAction({}, experience.state);
    auto before = diagnostics->Observe(experience);
    CHECK(std::isnan(*before.current->GetScalar(kStepIntrinsicError, 0)));
    actor->Sync();
    auto initial = diagnostics->Observe(experience);
    REQUIRE(initial.current->GetScalar(kStepIntrinsicError, 0));
    CHECK(std::isfinite(*initial.current->GetScalar(kStepIntrinsicError, 0)));
    CHECK(std::isnan(*initial.current->GetScalar(kStepIntrinsicNormalizedError, 0)));
    CHECK(*initial.current->GetScalar(kStepIntrinsicBonus, 0) == 0.0f);
    agent->CreateLearner()->UpdateFromBatch({.exp_step = 2}, experience);
    actor->Sync();
    auto learned = diagnostics->Observe(experience);
    CHECK(std::isfinite(*learned.current->GetScalar(kStepIntrinsicNormalizedError, 0)));
    CHECK(std::isfinite(*learned.current->GetScalar(kStepIntrinsicBonus, 0)));
    CHECK_FALSE(learned.current->GetScalar("intrinsic.unknown", 0));
}

TEST_CASE("Trace forwards requested intrinsic keys and Actor lifetime controls", "[rfd0127][rfd0129][intrinsic]")
{
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.actor.[trace].nn_forward_key", kTraceForwardKey);
    data.Set("DefaultDQNAgent.actor.[trace].policy.policy_type", anet::rl::dqn::kActionPolicyTypeStr_Greedy);
    data.Set("DefaultDQNAgent.actor.[trace].clone_model", true);
    auto agent = MakeIntrinsicAgent(data);
    auto actor = MakeIntrinsicTraceActor(agent);
    anet::ConfigData columns;
    columns.Set("app.eval_trace_panel.columns.[235]", "$agent_step intrinsic.bonus");
    columns.Set("app.eval_trace_panel.columns.[600]", "$agent_step intrinsic.error");
    columns.Set("app.eval_trace_panel.columns.[605]", "$agent_step intrinsic.error display:pct");
    columns.Set("app.eval_trace_panel.columns.[610]", "$agent_step intrinsic.normalized_error");
    anet::rl::StepTraceBuilder trace(columns, {}, agent, 2, actor);
    std::vector<std::shared_ptr<anet::rl::StepTrace>> rows;
    trace.SetHandlers([&](const auto& row, const auto&) { rows.push_back(row); }, {});
    CHECK(trace.Resume());
    actor->Sync();
    auto experience = IntrinsicExperience(2.0f, false, false, true);
    experience.action = actor->MakeAction({}, experience.state);
    const auto result = std::make_shared<anet::rl::PlainBatchStepResult>(experience.reward,
        experience.next_state, experience.next_state, 2, 0,
        std::vector<anet::rl::AuxData>(2));
    trace.OnTrain({{experience, nullptr, {}, agent, {}}, nullptr, result, experience.action});
    REQUIRE(rows.size() == 2);
    CHECK(rows[0]->values[0] == 0.0f);
    CHECK(std::isfinite(rows[0]->values[1]));
    CHECK(rows[0]->values[1] == rows[0]->values[2]);
    const auto old_error = rows[0]->values[1];
    trace.Suspend();
    CHECK_FALSE(trace.Resume());
    trace.Suspend();
    actor->Sync();
    CHECK(trace.Resume());
    actor->Sync();
    CHECK(rows[0]->values[1] == old_error);
}

TEST_CASE("Intrinsic diagnostics honor snapshot synchronization and suspension", "[rfd0127][intrinsic]")
{
    IntrinsicMetricsScope metrics;
    const bool enabled = GENERATE(false, true);
    const bool requested = GENERATE(false, true);
    const bool cloned = GENERATE(false, true);
    auto data = IntrinsicAgentData(enabled);
    data.Set("DefaultDQNAgent.actor.[trace].nn_forward_key", kTraceForwardKey);
    data.Set("DefaultDQNAgent.actor.[trace].policy.policy_type", anet::rl::dqn::kActionPolicyTypeStr_Greedy);
    data.Set("DefaultDQNAgent.actor.[trace].clone_model", cloned);
    if (cloned) data.Set("DefaultDQNAgent.actor.[trace].sync_interval", 1);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.learn.intake.ratio", 1.0f);
    data.Set("DefaultDQNAgent.learner.update_warmup_steps", 100000);
    auto agent = MakeIntrinsicAgent(data);
    auto actor = MakeIntrinsicTraceActor(agent);
    const std::vector<std::string> keys = requested ? std::vector<std::string>{kStepIntrinsicError}
        : std::vector<std::string>{"q_sa"};
    auto diagnostics = agent->CreateStepDiagnosticsProvider({.num_lanes = 2, .keys = keys, .actor = actor});
    const bool follower = enabled && requested && cloned;
    CHECK(diagnostics->Resume() == follower);
    auto experience = IntrinsicExperience(2.0f, false, false, true);
    experience.action = actor->MakeAction({}, experience.state);
    const auto read = [&] {
        return *diagnostics->Observe(experience).current->GetScalar(kStepIntrinsicError, 0);
    };
    if (!enabled || !requested || cloned) CHECK(std::isnan(read()));
    else CHECK(std::isfinite(read()));
    auto learner = agent->CreateLearner();
    learner->UpdateFromBatch({.exp_step = 2}, experience);
    actor->Sync();
    const auto captured = read();
    if (enabled && requested) CHECK(std::isfinite(captured));
    else CHECK(std::isnan(captured));
    auto changed = IntrinsicExperience(5.0f, false, false, false);
    changed.action = experience.action;
    learner->UpdateFromBatch({.exp_step = 4}, changed);
    if (follower) CHECK(read() == captured);
    // 明示Syncのあとの初回でageを合わせ、次のMakeActionで周期同期を通す。
    actor->MakeAction({.train_step = 1, .exp_step = 4}, experience.state);
    actor->MakeAction({.train_step = 2, .exp_step = 6}, experience.state);
    if (enabled && requested) CHECK(read() != captured);
    const auto before_suspend = read();
    diagnostics->Suspend();
    CHECK_FALSE(diagnostics->Resume());
    if (enabled && requested) CHECK(read() == before_suspend);
    diagnostics->Suspend();
    actor->Sync();
    CHECK(diagnostics->Resume() == follower);
    if (follower) CHECK(std::isnan(read()));
    actor->Sync();
    if (enabled && requested) CHECK(std::isfinite(read()));
    // 相手を所有しないActorは、部品が消えたあとの同期でも安全に進む。
    diagnostics.reset();
    CHECK_NOTHROW(actor->Sync());
    if (enabled && requested) {
        CHECK_THROWS_WITH(agent->CreateStepDiagnosticsProvider({.num_lanes = 2, .keys = keys, .actor = nullptr}),
            Catch::Matchers::ContainsSubstring(kStepIntrinsicError) && Catch::Matchers::ContainsSubstring("Actor"));
    } else {
        auto absent_actor = agent->CreateStepDiagnosticsProvider({.num_lanes = 2, .keys = keys, .actor = nullptr});
        CHECK_FALSE(absent_actor->Resume());
        CHECK(std::isnan(*absent_actor->Observe(experience).current->GetScalar(kStepIntrinsicBonus, 0)));
    }
}

TEST_CASE("Intrinsic diagnostics wait only for an un-cloned requested estimator", "[rfd0127][intrinsic]")
{
    class LockableAgent final : public anet::rl::dqn::DefaultDQNAgent {
    public:
        using DefaultDQNAgent::DefaultDQNAgent;
        std::shared_mutex& MutexForTest() { return *mutex_; }
    };
    IntrinsicMetricsScope metrics;
    const bool cloned = GENERATE(false, true);
    const bool requested = GENERATE(false, true);
    const bool synchronized = GENERATE(false, true);
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.actor.[trace].nn_forward_key", kTraceForwardKey);
    data.Set("DefaultDQNAgent.actor.[trace].policy.policy_type", anet::rl::dqn::kActionPolicyTypeStr_Greedy);
    data.Set("DefaultDQNAgent.actor.[trace].clone_model", cloned);
    anet::rl::EnvSpec spec;
    spec.state_spec.obs_spec = RndTestSpec();
    spec.action_spec.is_discrete = true;
    spec.action_spec.value_labels = {kTraceActionLabel};
    spec.reward_range = {-1000.0f, 1000.0f};
    const auto network = MakeIntrinsicNetworkConfig();
    auto agent = std::make_shared<LockableAgent>(anet::rl::dqn::DefaultDQNAgentConfig(data), network,
        anet::rl::BatchEnvSpec{.num_envs = 2, .num_threads = 1}, spec, torch::kCPU, 123);
    auto actor = MakeIntrinsicTraceActor(agent);
    auto provider = agent->CreateStepDiagnosticsProvider({.num_lanes = 2,
        .keys = requested ? std::vector<std::string>{kStepIntrinsicError} : std::vector<std::string>{}, .actor = actor});
    if (synchronized) actor->Sync();
    auto experience = IntrinsicExperience(1.0f, false, false, true);
    experience.action = actor->MakeAction({}, experience.state);
    provider->Observe(experience);
    // Tensorの初回準備を済ませ、学習側の書き込みロックだけを競合させる。
    std::unique_lock lock(agent->MutexForTest());
    auto observed = std::async(std::launch::async, [&] { return provider->Observe(experience); });
    const auto status = observed.wait_for(std::chrono::seconds(1));
    lock.unlock();
    const auto result = observed.get();
    CHECK(status == (!cloned && requested ? std::future_status::timeout : std::future_status::ready));
    REQUIRE(result.current);
    if (requested && (!cloned || synchronized))
        CHECK(std::isfinite(*result.current->GetScalar(kStepIntrinsicError, 0)));
    else CHECK(std::isnan(*result.current->GetScalar(kStepIntrinsicError, 0)));
}

TEST_CASE("Trace bonus equals the reward consumed by the real learner", "[rfd0127][intrinsic]")
{
    IntrinsicMetricsScope metrics;
    const auto mode = GENERATE(0, 1, 2);
    const bool state = GENERATE(false, true);
    const bool done = GENERATE(false, true);
    const bool mask = GENERATE(false, true);
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.actor.[trace].nn_forward_key", kTraceForwardKey);
    data.Set("DefaultDQNAgent.actor.[trace].policy.policy_type", anet::rl::dqn::kActionPolicyTypeStr_Greedy);
    data.Set("DefaultDQNAgent.actor.[trace].clone_model", true);
    data.Set("DefaultDQNAgent.head.init.mode", kConstantInit);
    data.Set("DefaultDQNAgent.head.init.constant_val", 0.0f);
    data.Set("DefaultDQNAgent.intrinsic.beta", 0.5f);
    data.Set("DefaultDQNAgent.intrinsic.learn.intake.batch_size", 1);
    data.Set("DefaultDQNAgent.intrinsic.learn.intake.ratio", 1.0f);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.use_clipping", false);
    data.Set("DefaultDQNAgent.intrinsic.observation", state ? anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationState : anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationNextState);
    data.Set("DefaultDQNAgent.intrinsic.use_done_mask", mask);
    if (mode == 1) data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.average_mode", "ema");
    if (mode == 2) data.Set("DefaultDQNAgent.intrinsic.rnd.statistics.mode", "variance_feedback");
    data.Set("DefaultDQNAgent.learner.use_n_step", false);
    data.Set("DefaultDQNAgent.learner.use_per", false);
    data.Set("DefaultDQNAgent.learner.update_warmup_steps", 100000);
    data.Set("DefaultDQNAgent.reward_scaler.constant_scale", 2.0f);
    auto agent = MakeIntrinsicAgent(data, 1);
    auto actor = MakeIntrinsicTraceActor(agent, 1);
    auto provider = agent->CreateStepDiagnosticsProvider({.num_lanes = 1,
        .keys = {kStepIntrinsicError, kStepIntrinsicNormalizedError, kStepIntrinsicBonus}, .actor = actor});
    const auto make = [&](float scale, bool terminal, bool start) {
        const auto flags = torch::zeros({1}, torch::kBool);
        const auto obs = RndTestObservation().At(kObservation).narrow(0, 0, 1);
        anet::rl::BatchState current(anet::TensorDict(kObservation, obs), flags, flags,
            torch::full({1}, start, torch::kBool));
        anet::rl::BatchState next(anet::TensorDict(kObservation, obs * scale),
            torch::full({1}, terminal, torch::kBool), flags, flags);
        return anet::rl::BatchExperience{current, actor->MakeAction({}, current), torch::ones({1}), next};
    };
    auto learner = agent->CreateLearner();
    for (int step = 0; step < 6; ++step)
        learner->UpdateFromBatch({.exp_step = static_cast<anet::rl::step_t>(step)}, make(step + 2.0f, true, true));
    actor->Sync();
    auto probe = make(100.0f, done, true);
    const auto values = provider->Observe(probe);
    const auto bonus = *values.current->GetScalar(kStepIntrinsicBonus, 0);
    REQUIRE(std::isfinite(bonus));
    if (!state && mask && done) CHECK(bonus == 0.0f);
    learner->UpdateFromBatch({.exp_step = 6}, probe);
    auto tail = make(200.0f, true, done);
    tail.state.obs = probe.next_state.obs;
    learner->UpdateFromBatch({.exp_step = 7}, tail);
    // n_step=falseの公開Replay収益は、Learnerが受け取った各遷移の報酬そのものである。
    // 偽のLearnerを注入せず、終端の後続を加えて非終端probeも有効な保存行として読む。
    const auto rewards = agent->GetTensor(anet::rl::ReplayBuffer::TARGET_RETURN);
    REQUIRE(rewards);
    const auto flat = rewards->flatten();
    REQUIRE(flat.numel() >= 7);
    CHECK(flat[6].item<float>() == Catch::Approx(2.0f + bonus).epsilon(1e-5));
}

TEST_CASE("Interleaved Trace leaves learner rewards and saved intrinsic state unchanged", "[rfd0127][intrinsic]")
{
    IntrinsicMetricsScope metrics;
    const bool cloned = GENERATE(false, true);
    const bool state = GENERATE(false, true);
    const auto mode = GENERATE(0, 1, 2);
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.actor.[trace].nn_forward_key", kTraceForwardKey);
    data.Set("DefaultDQNAgent.actor.[trace].policy.policy_type", anet::rl::dqn::kActionPolicyTypeStr_Greedy);
    data.Set("DefaultDQNAgent.actor.[trace].clone_model", cloned);
    data.Set("DefaultDQNAgent.head.init.mode", kConstantInit);
    data.Set("DefaultDQNAgent.head.init.constant_val", 0.0f);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.learn.intake.ratio", 1.0f);
    data.Set("DefaultDQNAgent.intrinsic.use_done_mask", false);
    data.Set("DefaultDQNAgent.intrinsic.observation", state ? anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationState : anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationNextState);
    if (mode == 1) data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.average_mode", "ema");
    if (mode == 2) data.Set("DefaultDQNAgent.intrinsic.rnd.statistics.mode", "variance_feedback");
    data.Set("DefaultDQNAgent.learner.use_n_step", false);
    data.Set("DefaultDQNAgent.learner.use_per", false);
    data.Set("DefaultDQNAgent.learner.update_warmup_steps", 100000);
    auto reference = MakeIntrinsicAgent(data);
    auto observed = MakeIntrinsicAgent(data);
    auto actor = MakeIntrinsicTraceActor(observed);
    auto diagnostics = observed->CreateStepDiagnosticsProvider({.num_lanes = 2,
        .keys = {kStepIntrinsicError, kStepIntrinsicNormalizedError, kStepIntrinsicBonus}, .actor = actor});
    auto reference_learner = reference->CreateLearner();
    auto observed_learner = observed->CreateLearner();
    for (int step = 0; step < 6; ++step) {
        const anet::rl::StepCounts counts{.train_step = static_cast<anet::rl::step_t>(step),
            .exp_step = static_cast<anet::rl::step_t>(step * 2)};
        auto experience = IntrinsicExperience(step + 2.0f, true, false, true);
        experience.state.obs.Set(kObservation, RndTestObservation().At(kObservation) * (step + 1.0f));
        experience.action = actor->MakeAction(counts, experience.state);
        if (step % 2 == 0) actor->Sync();
        diagnostics->Observe(experience);
        reference_learner->UpdateFromBatch(counts, experience);
        observed_learner->UpdateFromBatch(counts, experience);
        for (const auto* key : kAllIntrinsicScalarKeys) {
            const auto expected = *reference->GetScalar(key);
            const auto actual = *observed->GetScalar(key);
            if (std::isnan(expected)) CHECK(std::isnan(actual));
            else CHECK(actual == expected);
        }
        const auto expected = reference->GetTensor(anet::rl::ReplayBuffer::TARGET_RETURN);
        const auto actual = observed->GetTensor(anet::rl::ReplayBuffer::TARGET_RETURN);
        REQUIRE(expected.has_value() == actual.has_value());
        if (expected) CHECK(torch::equal(*expected, *actual));
    }
    // 保存データの不透明なTensor archiveのバイト表現ではなく、公開Load後の継続を比較する。
    const auto restore = [&](const auto& agent) {
        const auto format = mode == 1 ? kRndEmaCheckpointFormat : kRndCheckpointFormat;
        std::istringstream saved(IntrinsicCheckpointPayload(*agent, format), std::ios::binary);
        anet::InputArchive archive(saved);
        auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            data, std::string(kIntrinsicPrefix) + ".rnd", RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
        estimator->Load(archive);
        float credit;
        std::string random_state;
        int64_t learn_count;
        archive.Read(credit); archive.Read(random_state); archive.Read(learn_count);
        return std::tuple(std::move(estimator), credit, std::move(random_state), learn_count);
    };
    auto [expected_estimator, expected_credit, expected_random, expected_count] = restore(reference);
    auto [actual_estimator, actual_credit, actual_random, actual_count] = restore(observed);
    CHECK(expected_credit == actual_credit);
    CHECK(expected_random == actual_random);
    CHECK(expected_count == actual_count);
    for (int step = 0; step < 4; ++step) {
        const anet::TensorDict obs(kObservation, RndTestObservation().At(kObservation) * (-step - 1.0f));
        CHECK(torch::equal(expected_estimator->Estimate(obs), actual_estimator->Estimate(obs)));
        CHECK(expected_estimator->Learn(obs) == actual_estimator->Learn(obs));
        for (const auto& [key, expected] : expected_estimator->Diagnostics()) {
            const auto actual = actual_estimator->Diagnostics().at(key);
            if (std::isnan(expected)) CHECK(std::isnan(actual));
            else CHECK(actual == expected);
        }
    }
}

TEST_CASE("Checkpoint trace scores bonus before the first learning intake", "[rfd0127][rfd0129][intrinsic]")
{
    IntrinsicMetricsScope metrics;
    const bool learning_enabled = GENERATE(false, true);
    constexpr float beta = 0.3f;
    constexpr float multiplier_max = 5.0f;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.beta", beta);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.use_clipping", true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.multiplier_max", multiplier_max);
    data.Set("DefaultDQNAgent.actor.[trace].nn_forward_key", kTraceForwardKey);
    data.Set("DefaultDQNAgent.actor.[trace].policy.policy_type", anet::rl::dqn::kActionPolicyTypeStr_Greedy);
    data.Set("DefaultDQNAgent.actor.[trace].clone_model", true);
    auto trained = MakeIntrinsicAgent(data);
    for (anet::rl::step_t step = 0; step < 4; ++step)
        trained->CreateLearner()->UpdateFromBatch({.exp_step = step * 2},
            IntrinsicExperience(2.0f + step, false, false, step == 0));
    const auto path = std::filesystem::temp_directory_path() / ("anet-rfd0127-"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".anet");
    IntrinsicCheckpointCleanup cleanup{.path = path};
    {
        std::ofstream stream(path, std::ios::binary);
        anet::OutputArchive archive(stream);
        trained->Save(archive);
    }
    data.Set("DefaultDQNAgent.auto_load_file", path.string());
    data.Set("DefaultDQNAgent.learner.enabled", learning_enabled);
    auto evaluated = MakeIntrinsicAgent(data);
    auto actor = MakeIntrinsicTraceActor(evaluated);
    auto diagnostics = evaluated->CreateStepDiagnosticsProvider({.num_lanes = 2,
        .keys = {kStepIntrinsicError, kStepIntrinsicNormalizedError, kStepIntrinsicBonus}, .actor = actor});
    actor->Sync();
    auto experience = IntrinsicExperience(3.0f, false, false, true);
    experience.action = actor->MakeAction({}, experience.state);
    const auto result = diagnostics->Observe(experience);
    REQUIRE(result.current);
    CHECK(std::isfinite(*result.current->GetScalar(kStepIntrinsicError, 0)));
    CHECK(std::isfinite(*result.current->GetScalar(kStepIntrinsicNormalizedError, 0)));
    // checkpoint の読込後、学習の入口を通さずに設定の β で採点できる。
    const auto normalized_error = *result.current->GetScalar(kStepIntrinsicNormalizedError, 0);
    const auto bonus = *result.current->GetScalar(kStepIntrinsicBonus, 0);
    CHECK(std::isfinite(bonus));
    CHECK(bonus == Catch::Approx(beta * std::clamp(normalized_error, 0.0f, multiplier_max - 1.0f)));
}

TEST_CASE("Disabled learning intake advances the Trace beta schedule", "[rfd0129][intrinsic]")
{
    IntrinsicMetricsScope metrics;
    const bool clone_model = GENERATE(false, true);
    constexpr float kBetaStart = 0.2f;
    constexpr float multiplier_max = 5.0f;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.rnd.statistics.whiten.obs_norm.pass_through", true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.use_clipping", true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.multiplier_max", multiplier_max);
    data.Set("DefaultDQNAgent.actor.[trace].nn_forward_key", kTraceForwardKey);
    data.Set("DefaultDQNAgent.actor.[trace].policy.policy_type", anet::rl::dqn::kActionPolicyTypeStr_Greedy);
    data.Set("DefaultDQNAgent.actor.[trace].clone_model", clone_model);
    auto trained = MakeIntrinsicAgent(data);
    for (anet::rl::step_t step = 0; step < 4; ++step)
        trained->CreateLearner()->UpdateFromBatch({.exp_step = step * 2},
            IntrinsicExperience(2.0f + step, false, false, step == 0));
    const auto path = std::filesystem::temp_directory_path() / ("anet-rfd0129-schedule-"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + kCheckpointExtension);
    IntrinsicCheckpointCleanup cleanup{.path = path};
    {
        std::ofstream stream(path, std::ios::binary);
        anet::OutputArchive archive(stream);
        trained->Save(archive);
    }
    data.Set("DefaultDQNAgent.auto_load_file", path.string());
    data.Set("DefaultDQNAgent.learner.enabled", false);
    data.Set("DefaultDQNAgent.intrinsic.beta.type", "linear");
    data.Set("DefaultDQNAgent.intrinsic.beta.start", kBetaStart);
    data.Set("DefaultDQNAgent.intrinsic.beta.end", 0.6f);
    data.Set("DefaultDQNAgent.intrinsic.beta.steps", 10);
    auto evaluated = MakeIntrinsicAgent(data);
    auto actor = MakeIntrinsicTraceActor(evaluated);
    auto diagnostics = evaluated->CreateStepDiagnosticsProvider({.num_lanes = 2,
        .keys = {kStepIntrinsicError, kStepIntrinsicNormalizedError, kStepIntrinsicBonus}, .actor = actor});
    auto experience = IntrinsicExperience(1000.0f, false, false, true);
    experience.action = actor->MakeAction({}, experience.state);

    // 同じ checkpoint の同じ観測を採点し、β の予定だけが進んだことを照合する。
    for (const auto [exp_step, beta] : {std::pair{0, kBetaStart}, std::pair{5, 0.4f}}) {
        if (exp_step != 0)
            evaluated->CreateLearner()->UpdateFromBatch({.exp_step = static_cast<anet::rl::step_t>(exp_step)}, experience);
        actor->Sync();
        const auto result = diagnostics->Observe(experience);
        REQUIRE(result.current);
        const auto normalized_error = *result.current->GetScalar(kStepIntrinsicNormalizedError, 0);
        REQUIRE(std::isfinite(normalized_error));
        REQUIRE(normalized_error > 0.0f);
        const auto bonus = *result.current->GetScalar(kStepIntrinsicBonus, 0);
        CHECK(bonus == Catch::Approx(beta * std::clamp(normalized_error, 0.0f, multiplier_max - 1.0f)));
    }
}

TEST_CASE("Disabled learning intake preserves all intrinsic scalar diagnostics", "[rfd0129][intrinsic]")
{
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    auto trained = MakeIntrinsicAgent(data);
    for (anet::rl::step_t step = 0; step < 4; ++step)
        trained->CreateLearner()->UpdateFromBatch({.exp_step = step * 2},
            IntrinsicExperience(2.0f + step, false, false, step == 0));
    REQUIRE(*trained->GetScalar(kIntrinsicLearnCount) > 0.0f);
    const auto path = std::filesystem::temp_directory_path() / ("anet-rfd0129-diagnostics-"
        + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + kCheckpointExtension);
    IntrinsicCheckpointCleanup cleanup{.path = path};
    {
        std::ofstream stream(path, std::ios::binary);
        anet::OutputArchive archive(stream);
        trained->Save(archive);
    }
    data.Set("DefaultDQNAgent.auto_load_file", path.string());
    data.Set("DefaultDQNAgent.learner.enabled", false);
    auto evaluated = MakeIntrinsicAgent(data);
    std::array<float, kAllIntrinsicScalarKeys.size()> before;
    for (size_t index = 0; index < kAllIntrinsicScalarKeys.size(); ++index) {
        const auto value = evaluated->GetScalar(kAllIntrinsicScalarKeys[index]);
        REQUIRE(value.has_value());
        before[index] = *value;
    }

    // 公開の学習の入口を 3 回通しても、群56の全キーは読込後の値を維持する。
    for (anet::rl::step_t step = 0; step < 3; ++step) {
        evaluated->CreateLearner()->UpdateFromBatch({.exp_step = step * 2},
            IntrinsicExperience(20.0f + step, false, false, step == 0));
        for (size_t index = 0; index < kAllIntrinsicScalarKeys.size(); ++index) {
            const auto value = evaluated->GetScalar(kAllIntrinsicScalarKeys[index]);
            REQUIRE(value.has_value());
            if (std::isnan(before[index])) CHECK(std::isnan(*value));
            else CHECK(*value == before[index]);
        }
    }
}

TEST_CASE("Read-only nonfinite errors remain unavailable and warn once", "[rfd0127][intrinsic]")
{
    IntrinsicMetricsScope metrics;
    anet::test::LogCaptureGuard logs;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.error_norm.standardize.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.rnd.statistics.whiten.obs_norm.pass_through", true);
    data.Set("DefaultDQNAgent.actor.[trace].nn_forward_key", kTraceForwardKey);
    data.Set("DefaultDQNAgent.actor.[trace].policy.policy_type", anet::rl::dqn::kActionPolicyTypeStr_Greedy);
    auto agent = MakeIntrinsicAgent(data);
    agent->CreateLearner()->UpdateFromBatch({}, IntrinsicExperience(2.0f, false, false, true));
    auto actor = MakeIntrinsicTraceActor(agent);
    auto diagnostics = agent->CreateStepDiagnosticsProvider({.num_lanes = 2,
        .keys = {kStepIntrinsicError, kStepIntrinsicNormalizedError, kStepIntrinsicBonus}, .actor = actor});
    actor->Sync();
    auto experience = IntrinsicExperience(2.0f, false, false, true);
    experience.action = actor->MakeAction({}, experience.state);
    experience.next_state.obs.At(kObservation).fill_(std::numeric_limits<float>::quiet_NaN());
    for (int step = 0; step < 3; ++step) {
        const auto result = diagnostics->Observe(experience);
        for (const auto* key : {kStepIntrinsicError, kStepIntrinsicNormalizedError, kStepIntrinsicBonus})
            CHECK(std::isnan(*result.current->GetScalar(key, 0)));
        actor->Sync();
    }
    logs.Flush();
    CHECK(std::count_if(logs.Records().begin(), logs.Records().end(), [](const auto& record) {
        return record.level == wxLOG_Warning && record.message.find("read-only prediction_error") != std::string::npos;
    }) == 1);
    CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Warning,
        {"read-only prediction_error", "lane=0", "value=nan"}));
}

namespace {
    class IntrinsicTraceRunner final : public anet::rl::Runner {
        public:
            IntrinsicTraceRunner(std::shared_ptr<anet::rl::Agent> agent, std::shared_ptr<anet::rl::Actor> actor)
                : agent_(std::move(agent)), actor_(std::move(actor)) {}
            anet::rl::StepCounts DoStep(const anet::rl::MakeActionOptions& options = {}) override { return {}; }
            anet::rl::StepCounts DoUpdateFrame(int, ControlFunction, ControlFunction, const anet::rl::MakeActionOptions& options = {}) override { return {}; }
            anet::rl::RunnerStatus GetStatus() const override { return anet::rl::RunnerStatus{}; }
            void Shutdown() override {}
            anet::rl::StepCounts GetCounts() const override { return {}; }
            const std::string& GetName() const override { static const std::string name = "IntrinsicTraceRunner"; return name; }
            std::shared_ptr<anet::rl::BatchEnv> GetBatchEnv() const override { return nullptr; }
            std::shared_ptr<anet::rl::Agent> GetAgent() const override { return agent_; }
            std::shared_ptr<anet::rl::Actor> GetActor() const override { return actor_; }
            std::shared_ptr<anet::rl::Notifier> GetNotifier() const override { return nullptr; }
            std::optional<float> GetScalar(const std::string&, int64_t = -1) const override { return 0.0f; }
            std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
            std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t = -1) const override { return std::nullopt; }
        private:
            std::shared_ptr<anet::rl::Agent> agent_;
            std::shared_ptr<anet::rl::Actor> actor_;
        };
}

TEST_CASE("Shipped RND profiles expose readable intrinsic Trace columns only when selected", "[rfd0127][shipped_config]")
{
    IntrinsicMetricsScope metrics;
    constexpr char kShippedLunarConfig[] = "LunarLander";
    anet::rl::EnvSpec trace_spec;
    trace_spec.action_spec.is_discrete = true;
    trace_spec.action_spec.value_labels = {kTraceActionLabel};
    const auto root = std::filesystem::current_path() / "apps/runner/config";
    for (const auto* env : {"Atari", "GridMaze", kShippedLunarConfig}) {
        CAPTURE(env);
        for (const bool selected : {false, true}) {
            // 値や本数の写しを固定せず、選択した機能の欄がその機能の提供する鍵で読めるかを確かめる。
            const auto path = std::filesystem::temp_directory_path() / ("anet-rfd0127-config-"
                + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".txt");
            struct RemoveInput {
                std::filesystem::path path;
                ~RemoveInput() { std::error_code ignored; std::filesystem::remove(path, ignored); }
            } cleanup{.path = path};
            {
                std::ofstream file(path);
                file << "$include <" << (root / "_main.txt").generic_string() << ">\n"
                    << "$include <" << (root / (std::string(env) + ".txt")).generic_string() << ">\n"
                    << "run.$ = " << (selected ? "run.@rnd" : "") << '\n';
            }
            const auto resolved = anet::ConfigManager(path).GetConfigData();
            const auto columns = anet::rl::StepTraceColumnSpec::Parse(resolved, trace_spec);
            const auto intrinsic = [](const auto& column) {
                return column.target == "$agent_step" && column.key.starts_with("intrinsic.");
            };
            CHECK(std::any_of(columns.begin(), columns.end(), intrinsic) == selected);
            if (!selected || std::string(env) != kShippedLunarConfig) continue;
            auto data = IntrinsicAgentData(true);
            data.Set("DefaultDQNAgent.actor.[trace].nn_forward_key", kTraceForwardKey);
            data.Set("DefaultDQNAgent.actor.[trace].policy.policy_type", anet::rl::dqn::kActionPolicyTypeStr_Greedy);
            auto agent = MakeIntrinsicAgent(data);
            auto actor = MakeIntrinsicTraceActor(agent);
            auto runner = std::make_shared<IntrinsicTraceRunner>(agent, actor);
            anet::rl::StepTraceBuilder trace(resolved, trace_spec, agent, 2, actor);
            int added = 0;
            trace.SetHandlers([&](const auto& row, const auto&) {
                ++added;
                CHECK(row->values.size() == columns.size());
                for (size_t index = 0; index < columns.size(); ++index)
                    if (intrinsic(columns[index])) CHECK(row->value_ready[index]);
            }, {});
            actor->Sync();
            auto experience = IntrinsicExperience(2.0f, false, false, true);
            experience.action = actor->MakeAction({}, experience.state);
            const auto result = std::make_shared<anet::rl::PlainBatchStepResult>(experience.reward,
                experience.next_state, experience.next_state, 2, 0, std::vector<anet::rl::AuxData>(2));
            CHECK_NOTHROW(trace.OnTrain({{experience, runner, {}, agent, {}}, nullptr, result, experience.action}));
            CHECK(added == 2);
        }
    }
}
