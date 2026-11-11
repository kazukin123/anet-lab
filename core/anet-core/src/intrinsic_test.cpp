#include "anet/catch_test.hpp"
#include <array>
#include <cmath>
#include <sstream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <ATen/CPUGeneratorImpl.h>
#include "nn_impl.hpp"
#include "nn_heads.hpp"
#include "anet/intrinsic.hpp"
#include "intrinsic_rnd.hpp"
#include "anet/default_dqn_agent.hpp"
#include "dqn_based_agent.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/test_util.hpp"

constexpr char kObservation[] = "obs";
constexpr char kErrorMean[] = "error_mean";
constexpr char kErrorStd[] = "error_std";
constexpr char kErrorStep[] = "error_step";
constexpr char kClipRatio[] = "clip_ratio";
constexpr char kPredictorLoss[] = "predictor_loss";
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
constexpr char kTargetFeatureVar[] = "target_feature_var";
constexpr char kTargetFeatureMax[] = "target_feature_max";
constexpr char kReferenceGrid[] = "grid";
constexpr char kReferenceFeature[] = "feature";
constexpr char kTargetRole[] = "target";
constexpr char kPredictorRole[] = "predictor";
constexpr char kCategory[] = "category";
constexpr char kConstantInit[] = "constant";
constexpr char kReluBlock[] = "ReLU";
constexpr char kInvalidConfigValue[] = "unknown";
constexpr char kFalseConfigValue[] = "false";
constexpr char kRndCheckpointFormat[] = "RndNoveltyEstimator-3";
constexpr char kRndEmaCheckpointFormat[] = "RndNoveltyEstimator-4";
constexpr char kIntrinsicPrefix[] = "DefaultDQNAgent.intrinsic";

static anet::ConfigData RndTestConfig()
{
    anet::ConfigData data;
    for (const auto* role : {kTargetRole, kPredictorRole}) {
        const auto prefix = std::string(kRndPrefix) + "." + role + ".net";
        data.Set(prefix + ".branch.[feature].bind", kObservation);
        data.Set(prefix + ".branch.[feature].structure", "");
        data.Set(prefix + ".body.output.[features]", kReferenceFeature);
    }
    data.Set("rnd.output_dim", 4);
    data.Set("rnd.warmup_steps", 4);
    return data;
}

TEST_CASE("RND requires independent target and predictor configurations", "[prd116][rnd_reference_network]")
{
    // 2つのBodyを明示して公開Factoryから採点・学習まで通す。
    anet::nn::InitNN();
    anet::ConfigData data;
    for (const auto* role : {kTargetRole, kPredictorRole}) {
        const auto prefix = std::string(kRndPrefix) + "." + role + ".net";
        data.Set(prefix + ".branch.[feature].bind", kObservation);
        data.Set(prefix + ".branch.[feature].structure", "");
        data.Set(prefix + ".body.output.[features]", kReferenceFeature);
    }
    data.Set("rnd.output_dim", 4);
    data.Set("rnd.warmup_steps", 0);
    data.Set("rnd.predictor.net.branch.[feature].structure", "Projection");
    data.Set("rnd.predictor.net.block.[Projection].type", "Linear");
    data.Set("rnd.predictor.net.block.[Projection].linear.out_features", 3);
    anet::TensorSpecMap spec;
    spec[kObservation] = {.type = anet::SpaceType::Vector, .shape = {2}, .dtype = torch::kFloat32};
    anet::TensorDict obs;
    obs.Set(kObservation, torch::ones({2, 2}));
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, kRndPrefix, spec, torch::kCPU, anet::SeedMaker(42));
    CHECK(estimator->Estimate(obs).sizes() == torch::IntArrayRef({2}));
    CHECK(estimator->Learn(obs));
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
} // namespace

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
static std::shared_ptr<anet::rl::dqn::DefaultDQNAgent> MakeIntrinsicAgent(const anet::ConfigData& data, int num_envs = 2)
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
        anet::rl::BatchEnvSpec{.num_envs = num_envs, .num_threads = 1}, spec, torch::kCPU, 123);
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

TEST_CASE("RND shipped Atari networks and Taiga settings match their contracts",
    "[prd116][shipped_config][rnd_reference_shipped]")
{
    IntrinsicMetricsScope metrics;
    const auto root = std::filesystem::current_path();
    const auto path = root / "out/test-tmp/prd116-reference-config.txt";
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
    const anet::rl::intrinsic::RndNoveltyEstimatorConfig rnd_config(data, prefix);
    for (const auto* key : {"intrinsic.use_done_mask", "intrinsic.rnd.warmup_mode",
        "intrinsic.rnd.error_norm.average_mode", "intrinsic.rnd.error_norm.ema_steps",
        "intrinsic.rnd.error_norm.use_centering"})
        REQUIRE(data.Has(std::string("DefaultDQNAgent.") + key));
    const anet::rl::intrinsic::RndNoveltyEstimatorConfig rnd_defaults(RndTestConfig(), kRndPrefix);
    CHECK(rnd_config.warmup_mode == rnd_defaults.warmup_mode);
    CHECK(rnd_config.error_norm.average_mode == rnd_defaults.error_norm.average_mode);
    CHECK(rnd_config.error_norm.ema_steps == rnd_defaults.error_norm.ema_steps);
    CHECK(rnd_config.error_norm.use_centering == rnd_defaults.error_norm.use_centering);
    bool use_done_mask = false;
    data.Read("DefaultDQNAgent.intrinsic.use_done_mask", use_done_mask, false);
    const anet::rl::dqn::DefaultDQNAgentConfig facade_defaults(IntrinsicAgentData(true));
    CHECK(use_done_mask == facade_defaults.intrinsic.use_done_mask);
    CHECK(rnd_config.multiplier_max == anet::rl::intrinsic::RndNoveltyEstimatorConfig::kDefaultMultiplierMax);
    CHECK(rnd_config.obs_norm.clip_range == 5.0f);
    CHECK(rnd_config.head.init.mode == "orthogonal");
    CHECK(rnd_config.head.init.manual_gain == Catch::Approx(std::sqrt(2.0)));
    data.Set(prefix + ".warmup_steps", 0);
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

    // 選択済みのRun層へTaigaの8葉が届くことをtyped configで確認する。
    const auto taiga = resolve("run.@rainbow_like > run.@rnd > run.@rnd_taiga");
    const anet::rl::dqn::DefaultDQNAgentConfig agent(taiga);
    const anet::rl::intrinsic::RndNoveltyEstimatorConfig rnd(taiga, prefix);
    CHECK(agent.intrinsic.beta.value == Catch::Approx(0.00005f));
    CHECK(agent.intrinsic.observation == anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationState);
    CHECK(agent.intrinsic.learn_ratio == 1.0f);
    CHECK(agent.intrinsic.learn_batch_size == 1);
    CHECK(rnd.multiplier_mode == anet::rl::intrinsic::kMultiplierUnclipped);
    CHECK(rnd.statistics_mode == anet::rl::intrinsic::kStatisticsVarianceFeedback);
    CHECK(rnd.warmup_steps == 0);
    CHECK(rnd.adam_eps == Catch::Approx(1e-5));
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

TEST_CASE("DQN intrinsic state scoring ignores next observations and terminal masks", "[prd116][rnd_reference_observation]")
{
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.observation", anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationState);
    data.Set("DefaultDQNAgent.intrinsic.beta", 0.5f);
    data.Set("DefaultDQNAgent.intrinsic.learn_ratio", 1e-6f);
    data.Set("DefaultDQNAgent.intrinsic.rnd.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.rnd.statistics_mode", anet::rl::intrinsic::kStatisticsVarianceFeedback);
    data.Set("DefaultDQNAgent.intrinsic.rnd.multiplier_mode", anet::rl::intrinsic::kMultiplierUnclipped);
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

TEST_CASE("DQN intrinsic learning cadence follows experience count", "[prd116][intrinsic][rnd_schedule]")
{
    IntrinsicMetricsScope metrics;
    const int num_envs = GENERATE(2, 4);
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.learn_ratio", 0.25f);
    data.Set("DefaultDQNAgent.intrinsic.learn_batch_size", 2);
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

TEST_CASE("DQN intrinsic masks done but scores truncated lanes", "[prd116][intrinsic]")
{
    IntrinsicMetricsScope metrics;
    const bool done = GENERATE(false, true);
    const bool truncated = !done;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.warmup_mode", anet::rl::intrinsic::kWarmupCollectErrors);
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

TEST_CASE("RND warmup counts observations and reports optimizer progress", "[prd116][intrinsic][rnd_warmup_count]")
{
    anet::nn::InitNN();
    const int num_lanes = GENERATE(2, 4);
    auto data = RndTestConfig();
    data.Set("rnd.warmup_mode", anet::rl::intrinsic::kWarmupCollectErrors);
    data.Set("rnd.warmup_steps", 4);
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    const anet::TensorDict obs(kObservation, RndTestObservation().At(kObservation).repeat({num_lanes / 2, 1}));
    CHECK_FALSE(estimator->Learn(obs));
    for (int count = num_lanes; count <= 12; count += num_lanes) {
        const auto alpha = estimator->Estimate(obs);
        if (count <= 4) CHECK(torch::equal(alpha, torch::ones({num_lanes})));
        CHECK(estimator->Learn(obs) == (count > 4));
        CHECK(std::isfinite(estimator->Diagnostics().at(kErrorMean)));
        CHECK(std::isfinite(estimator->Diagnostics().at(kPredictorLoss)) == (count > 4));
    }
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

TEST_CASE("RND learns during warmup without collecting errors", "[prd123][intrinsic]")
{
    // 公開 Factory から採点と補助学習を通し、新しい既定の warm-up を確認する。
    anet::nn::InitNN();
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", GENERATE(3, 4));
    data.Set("rnd.multiplier_mode", anet::rl::intrinsic::kMultiplierUnclipped);
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    const auto obs = RndTestObservation();
    CHECK(torch::equal(estimator->Estimate(obs), torch::ones({2})));
    CHECK(std::isnan(estimator->Diagnostics().at(kErrorMean)));
    CHECK(std::isnan(estimator->Diagnostics().at(kErrorStd)));
    CHECK(estimator->Learn(obs));
    CHECK(std::isfinite(estimator->Diagnostics().at(kPredictorLoss)));
    CHECK(torch::equal(estimator->Estimate(obs), torch::ones({2})));
    CHECK(std::isnan(estimator->Diagnostics().at(kErrorMean)));
    CHECK(estimator->Learn(obs));
    // 閾値を越えた最初の batch は統計が空なので倍率1で、次からその統計を使う。
    const anet::TensorDict varied(kObservation, torch::tensor({{0.0f, 0.0f}, {100.0f, -30.0f}}));
    CHECK(torch::equal(estimator->Estimate(varied), torch::ones({2})));
    const auto before = estimator->Diagnostics();
    REQUIRE(std::isfinite(before.at(kErrorMean)));
    REQUIRE(before.at(kErrorStd) > 0.0f);
    const auto alpha = estimator->Estimate(anet::TensorDict(kObservation, obs.At(kObservation) * 2.0f));
    const auto expected = 1.0f + (estimator->Diagnostics().at(kErrorStep) - before.at(kErrorMean)) / before.at(kErrorStd);
    CHECK(alpha.mean().item<float>() == Catch::Approx(expected).margin(1e-5));
}

TEST_CASE("RND zero warmup makes both warmup choices equivalent", "[prd123][intrinsic]")
{
    anet::nn::InitNN();
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
    const auto make = [&] {
        return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    };
    auto learn = make();
    data.Set("rnd.warmup_mode", anet::rl::intrinsic::kWarmupCollectErrors);
    auto collect = make();
    for (int step = 0; step < 3; ++step) {
        const anet::TensorDict obs(kObservation, RndTestObservation().At(kObservation) * (step + 1.0f));
        CHECK(torch::equal(learn->Estimate(obs), collect->Estimate(obs)));
        REQUIRE(learn->Learn(obs));
        REQUIRE(collect->Learn(obs));
        CHECK(learn->Diagnostics().at(kPredictorLoss) == collect->Diagnostics().at(kPredictorLoss));
    }
}

TEST_CASE("RND EMA weights scored observations independently of batch splitting", "[prd123][intrinsic]")
{
    anet::nn::InitNN();
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
    data.Set("rnd.obs_norm.pass_through", true);
    data.Set("rnd.error_norm.average_mode", anet::rl::intrinsic::kErrorAverageEma);
    data.Set("rnd.error_norm.ema_steps", 2);
    data.Set("rnd.head.init.mode", kConstantInit);
    data.Set("rnd.head.init.constant_val", 1.0f);
    data.Set("rnd.predictor.net.block.[ReLU].type", kReluBlock);
    data.Set("rnd.predictor.net.branch.[feature].structure", kReluBlock);
    const auto make = [&] {
        return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(42));
    };
    auto singles = make();
    auto batched = make();
    const auto inputs = torch::tensor({{-1.0f, 0.0f}, {-2.0f, 0.0f}, {-3.0f, 0.0f}, {-4.0f, 0.0f}});
    // 誤差 1,4,9,16 と重み 1/8,1/4,1/2,1 の手計算。
    for (int64_t i = 0; i < 4; ++i) {
        singles->Estimate(anet::TensorDict(kObservation, inputs.narrow(0, i, 1)));
        CHECK(singles->Diagnostics().at(kErrorStep) == Catch::Approx((i + 1) * (i + 1)));
    }
    CHECK(torch::equal(batched->Estimate(anet::TensorDict(kObservation, inputs)), torch::ones({4})));
    CHECK(singles->Diagnostics().at(kErrorMean) == Catch::Approx(173.0 / 15.0));
    CHECK(singles->Diagnostics().at(kErrorStd) == Catch::Approx(std::sqrt(6146.0) / 15.0));
    CHECK(batched->Diagnostics().at(kErrorMean) == singles->Diagnostics().at(kErrorMean));
    CHECK(batched->Diagnostics().at(kErrorStd) == singles->Diagnostics().at(kErrorStd));
    data.Set("rnd.error_norm.ema_steps", 1);
    auto last = make();
    for (int repeat = 0; repeat < 2; ++repeat)
        CHECK(torch::equal(last->Estimate(anet::TensorDict(kObservation, inputs)), torch::ones({4})));
    CHECK(last->Diagnostics().at(kErrorMean) == 16.0f);
    CHECK(last->Diagnostics().at(kErrorStd) == 0.0f);
}

TEST_CASE("RND centering and clipping are independent scoring choices", "[prd123][intrinsic]")
{
    anet::nn::InitNN();
    const bool center = GENERATE(false, true);
    const bool clipped = GENERATE(false, true);
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
    data.Set("rnd.obs_norm.pass_through", true);
    data.Set("rnd.error_norm.use_centering", center);
    data.Set("rnd.multiplier_mode", clipped ? anet::rl::intrinsic::kMultiplierClipped : anet::rl::intrinsic::kMultiplierUnclipped);
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(42));
    estimator->Estimate(RndTestObservation());
    // 非ゼロの誤差と更新前の公開統計から、中心化と上下限の4通りを照合する。
    for (const float scale : {1.0f, 100.0f}) {
        const auto before = estimator->Diagnostics();
        const anet::TensorDict obs(kObservation, torch::ones({1, 2}) * scale);
        const auto alpha = estimator->Estimate(obs).item<float>();
        const auto error = estimator->Diagnostics().at(kErrorStep);
        auto expected = 1.0f + (error - (center ? before.at(kErrorMean) : 0.0f)) / before.at(kErrorStd);
        if (clipped) expected = std::clamp(expected, 1.0f, 5.0f);
        CHECK(alpha == Catch::Approx(expected));
    }
}

TEST_CASE("RND validates new settings before notifying unused values", "[prd123][intrinsic][config]")
{
    anet::nn::InitNN();
    // モードが使わない値も、通知より先に検証する。
    for (const auto& [key, value] : std::vector<std::pair<std::string, std::string>>{
        {"warmup_mode", kInvalidConfigValue}, {"error_norm.average_mode", kInvalidConfigValue},
        {"error_norm.ema_steps", "0"}, {"error_norm.ema_steps", "1.5"},
        {"error_norm.use_centering", kInvalidConfigValue}}) {
        auto data = RndTestConfig();
        data.Set("rnd.statistics_mode", anet::rl::intrinsic::kStatisticsVarianceFeedback);
        data.Set("rnd." + key, value);
        CAPTURE(key, value);
        CHECK_THROWS_WITH(anet::rl::intrinsic::RndNoveltyEstimatorConfig(data, kRndPrefix),
            Catch::Matchers::ContainsSubstring("rnd." + key) && Catch::Matchers::ContainsSubstring(value));
    }
    struct IgnoredSetting {
        std::string key, value, mode;
        int warmup;
        wxLogLevel level;
    };
    constexpr char kIgnoredEmaSteps[] = "7";
    for (const auto& entry : std::vector<IgnoredSetting>{
        {.key = "warmup_mode", .value = anet::rl::intrinsic::kWarmupCollectErrors,
            .mode = anet::rl::intrinsic::kStatisticsVarianceFeedback, .warmup = 4, .level = wxLOG_Warning},
        {.key = "warmup_mode", .value = anet::rl::intrinsic::kWarmupCollectErrors,
            .mode = anet::rl::intrinsic::kStatisticsWhiten, .warmup = 0, .level = wxLOG_Message},
        {.key = "error_norm.average_mode", .value = anet::rl::intrinsic::kErrorAverageEma,
            .mode = anet::rl::intrinsic::kStatisticsVarianceFeedback, .warmup = 4, .level = wxLOG_Warning},
        {.key = "error_norm.ema_steps", .value = kIgnoredEmaSteps,
            .mode = anet::rl::intrinsic::kStatisticsVarianceFeedback, .warmup = 4, .level = wxLOG_Warning},
        {.key = "error_norm.ema_steps", .value = kIgnoredEmaSteps,
            .mode = anet::rl::intrinsic::kStatisticsWhiten, .warmup = 4, .level = wxLOG_Warning},
        {.key = "error_norm.use_centering", .value = kFalseConfigValue,
            .mode = anet::rl::intrinsic::kStatisticsVarianceFeedback, .warmup = 4, .level = wxLOG_Warning}}) {
        auto data = RndTestConfig();
        data.Set("rnd.statistics_mode", entry.mode);
        data.Set("rnd.warmup_steps", entry.warmup);
        // baseline の既定のままなら通知しない。
        {
            anet::test::LogCaptureGuard logs;
            const anet::rl::intrinsic::RndNoveltyEstimatorConfig defaults(data, kRndPrefix);
            logs.Flush();
            CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 0);
            CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Info) == 0);
            CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Message) == 0);
        }
        const auto baseline_data = data;
        data.Set("rnd." + entry.key, entry.value);
        anet::test::LogCaptureGuard logs;
        auto baseline = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            baseline_data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
        auto changed = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
        // 通知だけでなく、使われない値が採点・統計・学習を変えないことを確かめる。
        for (int step = 0; step < 4; ++step) {
            const anet::TensorDict obs(kObservation, RndTestObservation().At(kObservation) * (step + 1.0f));
            CHECK(torch::equal(baseline->Estimate(obs), changed->Estimate(obs)));
            CHECK(baseline->Learn(obs) == changed->Learn(obs));
            const auto expected = baseline->Diagnostics();
            const auto actual = changed->Diagnostics();
            for (const auto& [key, value] : expected)
                CHECK((value == actual.at(key) || (std::isnan(value) && std::isnan(actual.at(key)))));
        }
        // variance_feedback で ema を指定しても、保存する書式は変わらない。
        const auto checkpoint_format = [](const auto& estimator) {
            std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
            anet::OutputArchive output(saved);
            estimator->Save(output);
            anet::InputArchive input(saved);
            std::string format;
            input.Read(format);
            return format;
        };
        CHECK(checkpoint_format(baseline) == checkpoint_format(changed));
        logs.Flush();
        CAPTURE(entry.key, entry.mode, entry.warmup);
        CHECK(std::count_if(logs.Records().begin(), logs.Records().end(), [&](const auto& record) {
            return record.message.find("Ignored rnd." + entry.key + "=") != std::string::npos;
        }) == 1);
        CHECK(anet::test::HasRecordContaining(logs.Records(), entry.level, {"rnd." + entry.key, entry.value}));
    }
}

TEST_CASE("RND checkpoint format follows error statistics and resumes scoring", "[prd123][intrinsic][checkpoint]")
{
    anet::nn::InitNN();
    const auto choice = GENERATE(Catch::Generators::values<std::pair<std::string, std::string>>({
        {anet::rl::intrinsic::kStatisticsWhiten, anet::rl::intrinsic::kErrorAverageCumulative},
        {anet::rl::intrinsic::kStatisticsWhiten, anet::rl::intrinsic::kErrorAverageEma},
        {anet::rl::intrinsic::kStatisticsVarianceFeedback, anet::rl::intrinsic::kErrorAverageCumulative}}));
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
    data.Set("rnd.statistics_mode", choice.first);
    data.Set("rnd.error_norm.average_mode", choice.second);
    const auto make = [&] {
        return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    };
    auto original = make();
    const auto obs = RndTestObservation();
    original->Estimate(obs);
    REQUIRE(original->Learn(obs));
    std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(saved);
    original->Save(output);
    const auto* expected = choice.second == anet::rl::intrinsic::kErrorAverageEma ? kRndEmaCheckpointFormat : kRndCheckpointFormat;
    std::istringstream header(saved.str(), std::ios::binary);
    anet::InputArchive header_input(header);
    std::string format;
    header_input.Read(format);
    CHECK(format == expected);
    auto restored = make();
    anet::InputArchive input(saved);
    restored->Load(input);
    for (int step = 0; step < 3; ++step) {
        const anet::TensorDict varied(kObservation, obs.At(kObservation) * (step + 2.0f));
        CHECK(torch::equal(original->Estimate(varied), restored->Estimate(varied)));
        CHECK(original->Diagnostics().at(kErrorMean) == restored->Diagnostics().at(kErrorMean));
        CHECK(original->Diagnostics().at(kErrorStd) == restored->Diagnostics().at(kErrorStd));
        REQUIRE(original->Learn(varied));
        REQUIRE(restored->Learn(varied));
        CHECK(original->Diagnostics().at(kPredictorLoss) == restored->Diagnostics().at(kPredictorLoss));
    }
    // 識別子だけの入力で、payload を読む前に両方の値を知らせて拒否する。
    const auto* unexpected = choice.second == anet::rl::intrinsic::kErrorAverageEma ? kRndCheckpointFormat : kRndEmaCheckpointFormat;
    std::stringstream incompatible(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive wrong_output(incompatible);
    wrong_output.Write(std::string(unexpected));
    anet::InputArchive wrong_input(incompatible);
    CHECK_THROWS_WITH(restored->Load(wrong_input),
        Catch::Matchers::ContainsSubstring(expected) && Catch::Matchers::ContainsSubstring(unexpected));
}

TEST_CASE("RND unclipped multiplier preserves negative bonuses", "[prd116][rnd_reference_multiplier]")
{
    // 既知の平均より小さい誤差でも、unclippedは倍率を1へ切り上げない。
    anet::nn::InitNN();
    auto data = RndTestConfig();
    data.Set("rnd.multiplier_mode", anet::rl::intrinsic::kMultiplierUnclipped);
    data.Set("rnd.warmup_steps", 0);
    data.Set("rnd.obs_norm.pass_through", true);
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(42));
    estimator->Estimate(RndTestObservation());
    const auto before = estimator->Diagnostics();
    anet::TensorDict zero;
    zero.Set(kObservation, torch::zeros({1, 2}));
    const auto alpha = estimator->Estimate(zero).item<float>();
    const auto expected = 1.0f + (estimator->Diagnostics().at(kErrorStep) - before.at(kErrorMean))
        / before.at(kErrorStd);
    REQUIRE(expected < 1.0f);
    CHECK(alpha == Catch::Approx(expected));
    CHECK(estimator->Diagnostics().at(kClipRatio) == 0.0f);
}

TEST_CASE("RND variance feedback follows sequential moments and resumes them", "[prd116][rnd_reference_feedback]")
{
    anet::nn::InitNN();
    auto data = RndTestConfig();
    data.Set("rnd.statistics_mode", anet::rl::intrinsic::kStatisticsVarianceFeedback);
    data.Set("rnd.multiplier_mode", anet::rl::intrinsic::kMultiplierUnclipped);
    data.Set("rnd.warmup_steps", 0);
    data.Set("rnd.head.init.mode", kConstantInit);
    data.Set("rnd.head.init.constant_val", 1.0f);
    data.Set("rnd.predictor.net.block.[ReLU].type", kReluBlock);
    data.Set("rnd.predictor.net.branch.[feature].structure", kReluBlock);
    const auto make = [&](const anet::ConfigData& config) {
        return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            config, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(42));
    };
    auto estimator = make(data);

    // 単位重み・bias=1のHeadで、target=sum(p)+1、predictor=sum(ReLU(p))+1。
    // 入力3件を有理数で手計算した期待値。統計のコードをテスト内に写さない。
    struct Expected {
        double alpha, error, mean, std, target_variance, target_max;
    };
    const std::array<Expected, 3> expected{{
        {.alpha = 1.0, .error = 1.0, .mean = 1.0, .std = 0.8660254037844386,
            .target_variance = 0.0, .target_max = 2.0},
        {.alpha = 1.6, .error = 4.0, .mean = 2.5, .std = 1.5811388300841898,
            .target_variance = 0.25, .target_max = 3.0},
        {.alpha = 0.4758041514894615, .error = 0.33406278932749767,
            .mean = 1.7780209297758325, .std = 1.6597035127118214,
            .target_variance = 0.9342019377399479, .target_max = 0.6415305437458044}
    }};
    const auto inputs = torch::tensor({{-1.0f, 2.0f}, {2.0f, -1.0f}, {-2.0f, 1.0f}});
    for (int64_t i = 0; i < 3; ++i) {
        CAPTURE(i);
        const auto alpha = estimator->Estimate(anet::TensorDict(kObservation, inputs.narrow(0, i, 1)));
        const auto diagnostics = estimator->Diagnostics();
        CHECK(alpha.item<float>() == Catch::Approx(expected[i].alpha).margin(1e-5));
        CHECK(diagnostics.at(kErrorStep) == Catch::Approx(expected[i].error).margin(1e-5));
        CHECK(diagnostics.at(kErrorMean) == Catch::Approx(expected[i].mean).margin(1e-5));
        CHECK(diagnostics.at(kErrorStd) == Catch::Approx(expected[i].std).margin(1e-5));
        CHECK(diagnostics.at(kTargetFeatureVar) == Catch::Approx(expected[i].target_variance).margin(1e-5));
        CHECK(diagnostics.at(kTargetFeatureMax) == Catch::Approx(expected[i].target_max).margin(1e-5));
    }

    // 2 laneでも単独呼出しと同じ順に統計を更新する。
    auto batched = make(data);
    const auto alpha = batched->Estimate(anet::TensorDict(kObservation, inputs.narrow(0, 0, 2)));
    CHECK(alpha[0].item<float>() == Catch::Approx(expected[0].alpha));
    CHECK(alpha[1].item<float>() == Catch::Approx(expected[1].alpha));
    CHECK(batched->Diagnostics().at(kErrorMean) == Catch::Approx(expected[1].mean));

    // 保存した統計で続きが一致し、Learnは採点統計を変えない。
    std::stringstream bytes(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(bytes);
    estimator->Save(output);
    auto restored = make(data);
    anet::InputArchive input(bytes);
    restored->Load(input);
    const anet::TensorDict obs(kObservation, inputs);
    CHECK(torch::equal(estimator->Estimate(obs), restored->Estimate(obs)));
    const auto before = restored->Diagnostics();
    CHECK(estimator->Learn(obs));
    CHECK(restored->Learn(obs));
    CHECK(restored->Diagnostics().at(kErrorMean) == before.at(kErrorMean));
    CHECK(restored->Diagnostics().at(kErrorStd) == before.at(kErrorStd));
    CHECK(torch::equal(estimator->Estimate(obs), restored->Estimate(obs)));

    // 書式の直後の統計モードが違えば、payloadを読む前に拒否する。
    data.Set("rnd.statistics_mode", anet::rl::intrinsic::kStatisticsWhiten);
    auto different_mode = make(data);
    bytes.seekg(0);
    anet::InputArchive incompatible(bytes);
    CHECK_THROWS_WITH(different_mode->Load(incompatible),
        Catch::Matchers::ContainsSubstring(anet::rl::intrinsic::kStatisticsVarianceFeedback) && Catch::Matchers::ContainsSubstring(anet::rl::intrinsic::kStatisticsWhiten));
}

TEST_CASE("RND factory scores observations and warms up statistics", "[prd116][intrinsic]")
{
    anet::nn::InitNN();
    auto data = RndTestConfig();
    data.Set("rnd.warmup_mode", anet::rl::intrinsic::kWarmupCollectErrors);
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, "rnd", RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    const auto obs = RndTestObservation();
    for (int step = 0; step < 2; ++step) {
        const auto alpha = estimator->Estimate(obs);
        CHECK(torch::equal(alpha, torch::ones({2})));
        estimator->Learn(obs);
        const auto diagnostics = estimator->Diagnostics();
        CHECK(std::isfinite(diagnostics.at(kErrorMean)));
        CHECK(std::isfinite(diagnostics.at(kErrorStd)));
        CHECK(std::isnan(diagnostics.at(kPredictorLoss)));
    }
}

TEST_CASE("RND learns supplied observations without preceding scoring", "[prd116][intrinsic][rnd_independent]")
{
    anet::nn::InitNN();
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
    data.Set("rnd.obs_norm.pass_through", true);
    data.Set("rnd.learning_rate", 0.03f);
    const auto make = [&] {
        return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    };
    auto estimator = make();
    auto reference = make();
    const auto obs = RndTestObservation();
    reference->Estimate(obs);
    const auto expected = reference->Diagnostics().at(kErrorStep);
    estimator->Learn(obs);
    REQUIRE(estimator->Diagnostics().at(kPredictorLoss) == Catch::Approx(expected));
    CHECK(std::isnan(estimator->Diagnostics().at(kErrorMean)));
    for (int step = 0; step < 100; ++step) estimator->Learn(obs);
    CHECK(estimator->Diagnostics().at(kPredictorLoss) < expected * 0.1f);
}

TEST_CASE("RND learning is independent of repeated scoring and leaves statistics unchanged", "[prd116][intrinsic][rnd_independent]")
{
    anet::nn::InitNN();
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
    const auto make = [&] {
        return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    };
    auto estimator = make();
    auto reference = make();
    const auto scored = RndTestObservation();
    const anet::TensorDict learned(kObservation,
        torch::tensor({{-4.0f, 1.0f}, {2.0f, 6.0f}, {8.0f, -3.0f}}));
    // 採点の連続呼び出しはoptimizerを進めず、採点と異なる件数の観測で学習できる。
    for (int step = 0; step < 2; ++step) {
        CHECK(torch::equal(estimator->Estimate(scored), reference->Estimate(scored)));
        CHECK(std::isnan(estimator->Diagnostics().at(kPredictorLoss)));
    }
    const auto before = estimator->Diagnostics();
    reference->Estimate(learned);
    const auto expected = reference->Diagnostics().at(kErrorStep);
    REQUIRE(expected != Catch::Approx(before.at(kErrorStep)));

    // checkpointの公開保存順から観測正規化を復元し、Learnが統計を変えないことを確かめる。
    const auto normalized_probe = [&] {
        std::stringstream bytes(std::ios::in | std::ios::out | std::ios::binary);
        anet::OutputArchive output(bytes);
        estimator->Save(output);
        anet::InputArchive input(bytes);
        std::string format, statistics_mode, target, predictor, optimizer;
        input.Read(format);
        input.Read(statistics_mode);
        input.Read(target);
        input.Read(predictor);
        input.Read(optimizer);
        const anet::rl::StateSpec spec{ .obs_spec = RndTestSpec() };
        const anet::rl::intrinsic::RndNoveltyEstimatorConfig config(data, kRndPrefix);
        const auto normalizer = anet::rl::ObservationNormalizerFactory(config.obs_norm).CreateObservationNormalizer(spec);
        normalizer->Load(input);
        return normalizer->Normalize(learned).At(kObservation).clone();
    };
    const auto normalized_before = normalized_probe();
    estimator->Learn(learned);
    CHECK(estimator->Diagnostics().at(kPredictorLoss) == Catch::Approx(expected));
    CHECK(torch::equal(normalized_before, normalized_probe()));
    for (const auto key : {kErrorMean, kErrorStd, kErrorStep})
        CHECK(estimator->Diagnostics().at(key) == before.at(key));
}

TEST_CASE("RND returns estimates on the input device", "[prd116][intrinsic][rnd_device]")
{
    anet::nn::InitNN();
    if (!torch::cuda::is_available()) { SKIP("CUDA is unavailable"); }
    const auto compute_device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    const auto statistics_mode = GENERATE(anet::rl::intrinsic::kStatisticsWhiten, anet::rl::intrinsic::kStatisticsVarianceFeedback);
    auto data = RndTestConfig();
    data.Set("rnd.statistics_mode", statistics_mode);
    data.Set("rnd.warmup_steps", 0);
    const auto make = [&] {
        return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            data, kRndPrefix, RndTestSpec(), compute_device, anet::SeedMaker(123));
    };
    auto cpu_input = make();
    auto cuda_input = make();
    for (int step = 0; step < 4; ++step) {
        const anet::TensorDict cpu_obs(kObservation, RndTestObservation().At(kObservation) * (step + 1.0f));
        const auto cuda_obs = cpu_obs.To(torch::Device(torch::kCUDA, 0));
        const auto cpu_alpha = cpu_input->Estimate(cpu_obs);
        const auto cuda_alpha = cuda_input->Estimate(cuda_obs);
        CHECK(cpu_alpha.device() == cpu_obs.device());
        REQUIRE(cuda_alpha.device() == cuda_obs.device());
        CHECK(torch::equal(cpu_alpha, cuda_alpha.cpu()));
        CHECK_FALSE(cuda_alpha.requires_grad());
        cpu_input->Learn(cpu_obs);
        cuda_input->Learn(cuda_obs);
        CHECK(cpu_input->Diagnostics().at(kPredictorLoss) == cuda_input->Diagnostics().at(kPredictorLoss));
    }
}

TEST_CASE("RND predictor learns repeated observations", "[prd116][intrinsic]")
{
    anet::nn::InitNN();
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
    data.Set("rnd.learning_rate", 0.03f);
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, "rnd", RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    const auto obs = RndTestObservation();
    estimator->Estimate(obs);
    const float initial = estimator->Diagnostics().at(kErrorStep);
    estimator->Learn(obs);
    for (int step = 0; step < 100; ++step) {
        const auto alpha = estimator->Estimate(obs);
        CHECK(torch::isfinite(alpha).all().item<bool>());
        CHECK((alpha >= 1.0f).all().item<bool>());
        CHECK((alpha <= 5.0f).all().item<bool>());
        estimator->Learn(obs);
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
    data.Set("rnd.warmup_mode", anet::rl::intrinsic::kWarmupCollectErrors);
    data.Set("rnd.learning_rate", 0.01f);
    const auto make = [&] {
        return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            data, "rnd", RndTestSpec(), device, anet::SeedMaker(123));
    };
    auto first = make();
    const auto obs = RndTestObservation();
    first->Estimate(obs);
    first->Learn(obs);
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
        CHECK(torch::equal(first->Estimate(varied), restored->Estimate(varied)));
        first->Learn(varied);
        restored->Learn(varied);
        const auto a = first->Diagnostics();
        const auto b = restored->Diagnostics();
        for (const auto& [key, value] : a)
            CHECK((value == b.at(key) || (std::isnan(value) && std::isnan(b.at(key)))));
    }
}

TEST_CASE("RND checkpoint can resume at every operation boundary", "[prd116][intrinsic][rnd_boundary]")
{
    anet::nn::InitNN();
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) { SKIP("CUDA is unavailable"); }
    const int boundary = GENERATE(0, 1, 2);
    CAPTURE(device, boundary);
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
    const auto make = [&] {
        return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            data, kRndPrefix, RndTestSpec(), device, anet::SeedMaker(123));
    };
    auto original = make();
    const auto obs = RndTestObservation();
    // 初回前・採点直後・補助学習直後のそれぞれで保存する。
    if (boundary >= 1) original->Estimate(obs);
    if (boundary >= 2) original->Learn(obs);
    std::stringstream bytes(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(bytes);
    original->Save(output);
    auto restored = make();
    anet::InputArchive input(bytes);
    restored->Load(input);
    int learned_steps = 0;
    for (int step = 0; step < 12; ++step) {
        const anet::TensorDict varied(kObservation, obs.At(kObservation) * (step + 2.0f));
        // 復元後は先にLearnする。採点のやり直しで隠れる状態欠落を検出する。
        original->Learn(varied);
        restored->Learn(varied);
        const auto expected = original->Diagnostics().at(kPredictorLoss);
        const auto actual = restored->Diagnostics().at(kPredictorLoss);
        CHECK((expected == actual || (std::isnan(expected) && std::isnan(actual))));
        if (std::isfinite(expected)) ++learned_steps;
        CHECK(torch::equal(original->Estimate(varied), restored->Estimate(varied)));
        CHECK(original->Diagnostics().at(kErrorStep) == restored->Diagnostics().at(kErrorStep));
        CHECK(original->Diagnostics().at(kErrorMean) == restored->Diagnostics().at(kErrorMean));
        CHECK(original->Diagnostics().at(kErrorStd) == restored->Diagnostics().at(kErrorStd));
    }
    CHECK(learned_steps > 0);
}

TEST_CASE("RND rejects invalid configuration before building networks", "[prd116][intrinsic][config]")
{
    const auto invalid = GENERATE(Catch::Generators::values<std::pair<std::string, std::string>>({
        {"output_dim", "0"},
        {"warmup_steps", "-1"}, {"multiplier_max", "1"}, {"multiplier_max", "nan"},
        {"learning_rate", "0"}, {"learning_rate", "inf"}, {"weight_decay", "-1"},
        {"adam_eps", "0"}, {"adam_eps", "-1"}, {"adam_eps", "nan"}, {"adam_eps", "inf"},
        {"multiplier_mode", kInvalidConfigValue}, {"statistics_mode", kInvalidConfigValue},
        {"head.init.mode", kInvalidConfigValue}, {"head.init.manual_gain", "nan"},
        {"obs_norm.clip_range", "0"}, {"obs_norm.epsilon", "-1"},
        {"obs_norm.post_process_type", "99"}}));
    auto data = RndTestConfig();
    data.Set("rnd." + invalid.first, invalid.second);
    CAPTURE(invalid.first, invalid.second);
    CHECK_THROWS_WITH(anet::rl::intrinsic::RndNoveltyEstimatorConfig(data, "rnd"),
        Catch::Matchers::ContainsSubstring("rnd." + invalid.first)
        && Catch::Matchers::ContainsSubstring("expected"));
}

TEST_CASE("RND feedback preserves category IDs at the network boundary", "[prd116][rnd_reference_category]")
{
    // カテゴリと連続値の混在でも、IDのone-hot化はNetwork境界に任せる。
    anet::nn::InitNN();
    const auto dtype = GENERATE(torch::kInt64, torch::kUInt8, torch::kFloat32);
    const bool mixed = GENERATE(false, true);
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
    data.Set("rnd.obs_norm.pass_through", true);
    for (const auto* role : {kTargetRole, kPredictorRole})
        data.Set(std::string(kRndPrefix) + "." + role + ".net.branch.[feature].bind", kCategory);
    auto spec = mixed ? RndTestSpec() : anet::TensorSpecMap{};
    spec[kCategory] = {.type = anet::SpaceType::Vector, .shape = {1}, .dtype = dtype, .num_classes = 3};
    const auto create = [&] {
        return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            data, kRndPrefix, spec, torch::kCPU, anet::SeedMaker(123));
    };
    auto reference = create();
    data.Set("rnd.obs_norm.pass_through", false);
    data.Set("rnd.statistics_mode", anet::rl::intrinsic::kStatisticsVarianceFeedback);
    auto feedback = create();
    auto obs = mixed ? RndTestObservation() : anet::TensorDict{};
    for (const auto [first, second] : {std::pair(2, 0), std::pair(0, 1), std::pair(1, 2)}) {
        const auto categories = torch::tensor({first, second}, torch::kInt64).reshape({2, 1}).to(dtype);
        obs.Set(kCategory, categories);
        reference->Estimate(obs);
        REQUIRE(torch::isfinite(feedback->Estimate(obs)).all().item<bool>());
        CHECK(feedback->Diagnostics().at(kTargetFeatureMax) == reference->Diagnostics().at(kTargetFeatureMax));
        CHECK(feedback->Diagnostics().at(kTargetFeatureVar) == reference->Diagnostics().at(kTargetFeatureVar));
        REQUIRE(reference->Learn(obs));
        REQUIRE(feedback->Learn(obs));
        CHECK(feedback->Diagnostics().at(kPredictorLoss) == reference->Diagnostics().at(kPredictorLoss));
        CHECK(torch::equal(obs.At(kCategory), categories));
    }
    std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(saved);
    feedback->Save(output);
    auto restored = create();
    anet::InputArchive input(saved);
    restored->Load(input);
    CHECK(torch::equal(feedback->Estimate(obs), restored->Estimate(obs)));
    REQUIRE(feedback->Learn(obs));
    REQUIRE(restored->Learn(obs));
    CHECK(feedback->Diagnostics().at(kPredictorLoss) == restored->Diagnostics().at(kPredictorLoss));
}

TEST_CASE("RND identifies each missing network role", "[prd116][intrinsic][config]")
{
    const auto role = GENERATE(kTargetRole, kPredictorRole);
    const auto missing = std::string(kRndPrefix) + "." + role + ".net";
    const auto complete = RndTestConfig();
    anet::ConfigData data;
    for (const auto& [key, value] : complete.Map())
        if (!key.starts_with(missing)) data.Set(key, value);
    CHECK_THROWS_WITH(anet::rl::intrinsic::RndNoveltyEstimatorConfig(data, kRndPrefix),
        Catch::Matchers::ContainsSubstring(missing));
}

TEST_CASE("RND applies head initialization to each independently seeded network", "[prd116][intrinsic][config]")
{
    anet::nn::InitNN();
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
    const auto create = [&] {
        return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    };
    auto xavier = create();
    data.Set("rnd.head.init.mode", kConstantInit);
    data.Set("rnd.head.init.constant_val", 0.0f);
    auto zero = create();
    const auto obs = RndTestObservation();
    xavier->Estimate(obs);
    zero->Estimate(obs);
    CHECK(zero->Diagnostics().at(kTargetFeatureMax) == 0.0f);
    CHECK(xavier->Diagnostics().at(kTargetFeatureMax) > 0.0f);
    CHECK(zero->Diagnostics().at(kErrorStep) == 0.0f);
    CHECK(xavier->Diagnostics().at(kErrorStep) > 0.0f);
    // このfixtureのBodyはidentityなので、targetの保存値の差はHeadの重みの差である。
    const auto target_bytes = [](const auto& estimator) {
        std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
        anet::OutputArchive output(saved);
        estimator->Save(output);
        anet::InputArchive input(saved);
        std::string format, mode, target;
        input.Read(format);
        input.Read(mode);
        input.Read(target);
        return target;
    };
    CHECK(target_bytes(xavier) != target_bytes(zero));
}

TEST_CASE("RND warns once per ignored resolved leaf", "[prd116][intrinsic][rnd_reference_warnings]")
{
    anet::nn::InitNN();
    auto data = RndTestConfig();
    data.Set("rnd.multiplier_mode", anet::rl::intrinsic::kMultiplierUnclipped);
    data.Set("rnd.statistics_mode", anet::rl::intrinsic::kStatisticsVarianceFeedback);
    const bool changed = GENERATE(false, true);
    data.Set("rnd.multiplier_max", changed ? 10 : 5);
    data.Set("rnd.obs_norm.use_centering", !changed);
    data.Set("rnd.obs_norm.clip_range", 3);
    anet::test::LogCaptureGuard logs;
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    for (int step = 0; step < 3; ++step) estimator->Estimate(RndTestObservation());
    logs.Flush();
    const auto warnings = [&](const std::string& key) {
        return std::count_if(logs.Records().begin(), logs.Records().end(), [&](const auto& record) {
            return record.message.find("Ignored rnd." + key + "=") != std::string::npos;
        });
    };
    CHECK(warnings("multiplier_max") == (changed ? 1 : 0));
    CHECK(warnings("obs_norm.use_centering") == (changed ? 1 : 0));
    CHECK(warnings("obs_norm.clip_range") == 0);
}

TEST_CASE("DQN validates the selected intrinsic observation only when enabled", "[prd116][intrinsic][config]")
{
    auto data = IntrinsicAgentData(false);
    data.Set("DefaultDQNAgent.intrinsic.observation", kInvalidConfigValue);
    CHECK_NOTHROW(anet::rl::dqn::DefaultDQNAgentConfig(data));
    data.Set("DefaultDQNAgent.intrinsic.enabled", true);
    CHECK_THROWS_WITH(anet::rl::dqn::DefaultDQNAgentConfig(data),
        Catch::Matchers::ContainsSubstring("intrinsic.observation")
        && Catch::Matchers::ContainsSubstring(kInvalidConfigValue));
}

TEST_CASE("RND feedback warmup ends by observation count", "[prd116][intrinsic][rnd_reference_feedback]")
{
    anet::nn::InitNN();
    auto data = RndTestConfig();
    data.Set("rnd.statistics_mode", anet::rl::intrinsic::kStatisticsVarianceFeedback);
    data.Set("rnd.multiplier_mode", anet::rl::intrinsic::kMultiplierUnclipped);
    data.Set("rnd.warmup_steps", 2);
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    const anet::TensorDict one(kObservation, torch::tensor({{1.0f, 2.0f}}));
    for (int step = 0; step < 2; ++step) {
        CHECK(estimator->Estimate(one).item<float>() == 1.0f);
        CHECK_FALSE(estimator->Learn(one));
    }
    CHECK(torch::isfinite(estimator->Estimate(one)).all().item<bool>());
    CHECK(estimator->Learn(one));
    CHECK(estimator->Diagnostics().at(kErrorMean) >= 0.0f);
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
        CHECK(torch::equal(bytes->Estimate(byte_obs), floats->Estimate(float_obs)));
        CHECK(bytes->Diagnostics().at(kErrorStep) == floats->Diagnostics().at(kErrorStep));
        bytes->Learn(byte_obs);
        floats->Learn(float_obs);
    }
}

TEST_CASE("RND keeps its seeded target fixed and uses preceding error statistics", "[prd116][intrinsic]")
{
    anet::nn::InitNN();
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
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
        std::string format, statistics_mode, target, predictor;
        input.Read(format);
        input.Read(statistics_mode);
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
        const auto alpha = estimator->Estimate(obs).item<float>();
        const auto error = estimator->Diagnostics().at(kErrorStep);
        const auto std = previous.at(kErrorStd);
        const auto expected = std > 0 ? std::clamp(1 + (error - previous.at(kErrorMean)) / std, 1.0f, 5.0f) : 1.0f;
        CHECK(alpha == Catch::Approx(expected));
        estimator->Learn(obs);
    }
    const auto after = networks(estimator);
    CHECK(before.first == after.first);
    CHECK(before.second != after.second);

    // 同じ入力を100回学習しても、保存したtargetの重みと出力は変わらない。
    const auto repeated = RndTestObservation();
    const auto target_output = [&](const std::string& saved_target) {
        auto network = anet::nn::NetworkBuilder::BuildNetwork(
            anet::nn::NetworkConfig(data, "rnd.target.net"), RndTestSpec(),
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
        estimator->Estimate(repeated);
        estimator->Learn(repeated);
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
    data.Set("rnd.warmup_steps", 0);
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, "rnd", RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    const anet::TensorDict obs(kObservation, torch::full({2, 2}, std::numeric_limits<float>::quiet_NaN()));
    constexpr char kPredictionError[] = "prediction_error";
    constexpr char kExpectedFinite[] = "expected finite";
    CHECK_THROWS_WITH(estimator->Estimate(obs), Catch::Matchers::ContainsSubstring(kPredictionError)
        && Catch::Matchers::ContainsSubstring(kExpectedFinite));
    CHECK_THROWS_WITH(estimator->Learn(obs), Catch::Matchers::ContainsSubstring(kPredictionError)
        && Catch::Matchers::ContainsSubstring(kExpectedFinite));
    CHECK(std::isnan(estimator->Diagnostics().at(kPredictorLoss)));
}

TEST_CASE("DQN learner consumes the scaled external reward plus intrinsic bonus", "[prd116][intrinsic]")
{
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.obs_norm.pass_through", true);
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

TEST_CASE("DQN intrinsic done mask controls the actual learning reward", "[prd123][intrinsic]")
{
    SECTION("validates enabled values and notifies ignored state masking once") {
        const std::string key = "DefaultDQNAgent.intrinsic.use_done_mask";
        auto data = IntrinsicAgentData(false);
        data.Set(key, kInvalidConfigValue);
        CHECK_NOTHROW(anet::rl::dqn::DefaultDQNAgentConfig(data));
        data.Set("DefaultDQNAgent.intrinsic.enabled", true);
        CHECK_THROWS_WITH(anet::rl::dqn::DefaultDQNAgentConfig(data),
            Catch::Matchers::ContainsSubstring(key) && Catch::Matchers::ContainsSubstring(kInvalidConfigValue));
        data.Set(key, false);
        data.Set("DefaultDQNAgent.intrinsic.observation", anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationState);
        anet::test::LogCaptureGuard logs;
        const anet::rl::dqn::DefaultDQNAgentConfig config(data);
        logs.Flush();
        CHECK(std::count_if(logs.Records().begin(), logs.Records().end(), [&](const auto& record) {
            return record.level == wxLOG_Warning && record.message.find(key) != std::string::npos;
        }) == 1);
        CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Warning,
            {key, kFalseConfigValue, anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationState}));
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
        data.Set("DefaultDQNAgent.intrinsic.rnd.warmup_steps", 0);
        data.Set("DefaultDQNAgent.intrinsic.rnd.obs_norm.pass_through", true);
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
    const auto changed_key = GENERATE("learning_rate", "weight_decay", "adam_eps");
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
    data.Set("rnd.obs_norm.pass_through", true);
    const auto make = [](const anet::ConfigData& config) {
        return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
            config, "rnd", RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    };
    auto original = make(data);
    const auto obs = RndTestObservation();
    for (int step = 0; step < 3; ++step) { original->Estimate(obs); original->Learn(obs); }
    std::stringstream bytes(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(bytes);
    original->Save(output);
    data.Set(std::string("rnd.") + changed_key, 0.5f);
    auto changed = make(data);
    anet::InputArchive input(bytes);
    changed->Load(input);
    CHECK(torch::equal(original->Estimate(obs), changed->Estimate(obs)));
    original->Learn(obs);
    changed->Learn(obs);
    original->Estimate(obs);
    changed->Estimate(obs);
    CHECK(original->Diagnostics().at(kErrorStep) != changed->Diagnostics().at(kErrorStep));
}

TEST_CASE("DQN intrinsic checkpoint resumes fractional credit and selected observations", "[prd116][intrinsic][rnd_schedule_checkpoint]")
{
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.learn_ratio", 0.75f);
    data.Set("DefaultDQNAgent.intrinsic.learn_batch_size", 1);
    auto original = MakeIntrinsicAgent(data);
    original->CreateLearner()->UpdateFromBatch({}, IntrinsicExperience(1, false, false, true));
    REQUIRE(*original->GetScalar(kIntrinsicLearnCount) == 1.0f);
    // 1.5 creditのうち1回を消費し、0.5と抽選済みの乱数状態を保存する。
    const auto path = std::filesystem::temp_directory_path() / ("anet-prd116-schedule-"
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

TEST_CASE("DQN intrinsic consumes warmup credit and holds the last loss", "[prd116][intrinsic][rnd_schedule_edges]")
{
    IntrinsicMetricsScope metrics;
    auto data = IntrinsicAgentData(true);
    data.Set("DefaultDQNAgent.intrinsic.rnd.warmup_mode", anet::rl::intrinsic::kWarmupCollectErrors);
    data.Set("DefaultDQNAgent.intrinsic.rnd.warmup_steps", 8);
    data.Set("DefaultDQNAgent.intrinsic.learn_ratio", 0.5f);
    data.Set("DefaultDQNAgent.intrinsic.learn_batch_size", 2);
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

TEST_CASE("DQN intrinsic validates schedule settings only when enabled", "[prd116][intrinsic][config][rnd_schedule_edges]")
{
    const auto invalid = GENERATE(Catch::Generators::values<std::pair<std::string, std::string>>({
        {"learn_ratio", "0"}, {"learn_ratio", "-0.1"}, {"learn_ratio", "1.01"},
        {"learn_ratio", "nan"}, {"learn_ratio", "inf"},
        {"learn_batch_size", "0"}, {"learn_batch_size", "-1"}, {"learn_batch_size", "1.5"}}));
    auto data = IntrinsicAgentData(false);
    const auto key = std::string(kIntrinsicPrefix) + "." + invalid.first;
    data.Set(key, invalid.second);
    CAPTURE(key, invalid.second);
    CHECK_NOTHROW(anet::rl::dqn::DefaultDQNAgentConfig(data));
    data.Set("DefaultDQNAgent.intrinsic.enabled", true);
    CHECK_THROWS_WITH(anet::rl::dqn::DefaultDQNAgentConfig(data), Catch::Matchers::ContainsSubstring(key));
}

// 公開checkpointのRND節を読む。private stateや偽推定器をテスト用に公開しない。
static std::string IntrinsicCheckpointPayload(const anet::rl::dqn::DefaultDQNAgent& agent)
{
    std::ostringstream saved(std::ios::binary);
    anet::OutputArchive output(saved);
    agent.Save(output);
    std::ostringstream marker(std::ios::binary);
    anet::OutputArchive marker_archive(marker);
    marker_archive.Write(std::string(kRndCheckpointFormat));
    const auto bytes = saved.str();
    const auto offset = bytes.find(marker.str());
    REQUIRE(offset != std::string::npos);
    return bytes.substr(offset);
}

static std::string RndPredictorBytes(std::istream& stream)
{
    anet::InputArchive archive(stream);
    std::string format, statistics_mode, target, predictor;
    archive.Read(format);
    REQUIRE(format == kRndCheckpointFormat);
    archive.Read(statistics_mode);
    archive.Read(target);
    archive.Read(predictor);
    return predictor;
}

TEST_CASE("RND rejects mismatched checkpoint formats before reading payload", "[prd116][intrinsic][rnd_schedule_edges]")
{
    anet::nn::InitNN();
    const std::string format = GENERATE("RndNoveltyEstimator-1", "RndNoveltyEstimator-2", "unknown-rnd-format");
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        RndTestConfig(), kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(saved);
    output.Write(format);
    anet::InputArchive input(saved);
    CHECK_THROWS_WITH(estimator->Load(input),
        Catch::Matchers::ContainsSubstring(format) && Catch::Matchers::ContainsSubstring(kRndCheckpointFormat));
}

TEST_CASE("DQN intrinsic learns distinct complete lane groups and warns once when shrinking", "[prd116][intrinsic][rnd_sampling]")
{
    IntrinsicMetricsScope metrics;
    const auto input_device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (input_device.is_cuda() && !torch::cuda::is_available()) { SKIP("CUDA is unavailable"); }
    const auto dimensions = GENERATE(Catch::Generators::values<std::pair<int, int>>({
        {2, 32}, {3, 2}, {4, 2}, {4, 1}}));
    const auto [num_lanes, configured_batch] = dimensions;
    const int batch_size = std::min(num_lanes, configured_batch);
    CAPTURE(num_lanes, configured_batch, input_device);
    auto data = IntrinsicAgentData(true);
    const auto observation = GENERATE(anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationNextState, anet::rl::dqn::DefaultDQNAgentConfig::kIntrinsicObservationState);
    data.Set("DefaultDQNAgent.intrinsic.observation", observation);
    data.Set("DefaultDQNAgent.intrinsic.rnd.warmup_steps", 0);
    data.Set("DefaultDQNAgent.intrinsic.learn_ratio", 1.0f);
    data.Set("DefaultDQNAgent.intrinsic.learn_batch_size", configured_batch);
    anet::test::LogCaptureGuard logs;
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
    CHECK(warnings == (num_lanes < configured_batch ? 1 : 0));
}
