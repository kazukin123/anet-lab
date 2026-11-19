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
#include "nn_impl.hpp"
#include "anet/nn_head.hpp"
#include "anet/env.hpp"
#include "anet/intrinsic.hpp"
#include "intrinsic_rnd.hpp"
#include "anet/metrics.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/observers.hpp"
#include "anet/test/util.hpp"


// PRD121の入力・公開識別子を、同じ意味の検査で共有する。

constexpr char kTraceActionLabel[] = "a0";
constexpr char kTraceActorKey[] = "trace";
constexpr char kTraceForwardKey[] = "test";
constexpr char kStepIntrinsicError[] = "intrinsic.error";
constexpr char kStepIntrinsicNormalizedError[] = "intrinsic.normalized_error";
constexpr char kStepIntrinsicBonus[] = "intrinsic.bonus";
constexpr char kLaneError[] = "error";
constexpr char kLaneNormalizedError[] = "normalized_error";
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
constexpr char kCheckpointExtension[] = ".anet";
constexpr char kIntrinsicPrefix[] = "DefaultDQNAgent.intrinsic";
constexpr char kUnknownIntrinsicScalarKey[] = "intrinsic.unknown_key";

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
    data.Set("rnd.head.output_dim", 4);
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
} // namespace







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
        {"head.output_dim", "0"},
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
    CHECK_NOTHROW(anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123)));
    logs.Flush();
    const auto warnings = [&](const std::string& key) {
        return std::count_if(logs.Records().begin(), logs.Records().end(), [&](const auto& record) {
            return record.level == wxLOG_Warning
                && record.message.find("Ignored rnd." + key + "=") != std::string::npos;
        });
    };
    CHECK(warnings("multiplier_max") == (changed ? 1 : 0));
    CHECK(warnings("obs_norm.use_centering") == (changed ? 1 : 0));
    CHECK(warnings("obs_norm.clip_range") == 0);
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

TEST_CASE("RND mixed observations follow double precision reference statistics", "[prd130][intrinsic]")
{
    // CPU の倍精度で観測と誤差の統計を作り、公開の採点と保存状態に照らす。
    // 観測の統計の差を測るため、行列積は先行テストの TF32 設定によらず FP32 に揃える。
    struct MatmulPrecisionGuard {
        bool allow_tf32;
        ~MatmulPrecisionGuard() { at::globalContext().setAllowTF32CuBLAS(allow_tf32); }
    };
    const MatmulPrecisionGuard precision{.allow_tf32 = at::globalContext().allowTF32CuBLAS()};
    at::globalContext().setAllowTF32CuBLAS(false);
    anet::nn::InitNN();
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");
    const auto mode = GENERATE(0, 1, 2);
    auto data = RndTestConfig();
    data.Set("rnd.error_norm.ema_steps", 4);
    if (mode == 1) data.Set("rnd.error_norm.average_mode", anet::rl::intrinsic::kErrorAverageEma);
    else data.Set("rnd.error_norm.ema_steps", anet::rl::intrinsic::RndNoveltyEstimatorConfig::kDefaultEmaSteps);
    if (mode == 2) data.Set("rnd.warmup_mode", anet::rl::intrinsic::kWarmupCollectErrors);
    for (const auto* role : {kTargetRole, kPredictorRole})
        data.Set(std::string(kRndPrefix) + "." + role + ".net.branch.[feature].bind",
            std::string(kReferenceGrid) + ", " + kObservation + ", " + kCategory);
    auto spec = RndTestSpec();
    spec[kReferenceGrid] = {.type = anet::SpaceType::Vector, .shape = {2}, .dtype = torch::kUInt8};
    spec[kCategory] = {.type = anet::SpaceType::Vector, .shape = {1}, .dtype = torch::kInt64, .num_classes = 3};
    auto actual = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, kRndPrefix, spec, device, anet::SeedMaker(123));
    auto reference_data = data;
    reference_data.Set("rnd.obs_norm.pass_through", true);
    auto reference_spec = spec;
    reference_spec.at(kReferenceGrid).dtype = torch::kFloat32;
    auto reference = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        reference_data, kRndPrefix, reference_spec, torch::kCPU, anet::SeedMaker(123));
    std::map<std::string, std::vector<torch::Tensor>> history;
    std::vector<double> errors;
    double ema_weight = 0, ema_mean = 0, ema_m2 = 0;
    for (int step = 0; step < 10; ++step) {
        const float scale = step % 3 == 2 ? 8.0f : 1.0f;
        anet::TensorDict observations(kObservation,
            torch::tensor({{(step + 1.25f) * scale, (step + 2.5f) * scale},
                           {(step + 3.75f) * scale, (step + 5.0f) * scale}}));
        observations.Set(kReferenceGrid, torch::tensor({{step + 1, step + 3}, {step + 7, step + 11}}, torch::kUInt8));
        observations.Set(kCategory, torch::tensor({{step % 3}, {(step + 1) % 3}}, torch::kInt64));
        auto normalized = observations;
        for (const auto* key : {kObservation, kReferenceGrid}) {
            const auto values = observations.At(key).to(torch::kDouble);
            auto mean = torch::zeros({2}, torch::kDouble);
            auto std = torch::full({2}, 2.0, torch::kDouble);
            if (!history[key].empty()) {
                const auto all = torch::cat(history[key], 0);
                mean = all.mean(0);
                std = torch::var(all, {0}, true).sqrt() + 1.0;
            }
            normalized.Set(key, ((values - mean) / std).clamp(-5.0, 5.0).to(torch::kFloat32));
            history[key].push_back(values);
        }
        const auto prediction_errors = reference->EstimateWithoutUpdate(normalized).diagnostics.at(kLaneError);
        const auto read_only = actual->EstimateWithoutUpdate(observations);
        CHECK(torch::allclose(read_only.diagnostics.at(kLaneError), prediction_errors, 1e-4, 1e-5));
        auto expected = torch::ones({2});
        if (step >= 2 && !errors.empty()) {
            double mean = 0, variance = 0;
            for (const auto value : errors) mean += value / errors.size();
            for (const auto value : errors) variance += (value - mean) * (value - mean) / (errors.size() - 1);
            const auto std = mode == 1 ? std::sqrt(ema_m2 / ema_weight) : std::sqrt(variance);
            const auto center = mode == 1 ? ema_mean : mean;
            if (std > 0)
                expected = (1.0f + (prediction_errors - static_cast<float>(center)) / static_cast<float>(std)).clamp(1.0f, 5.0f);
        }
        const auto multipliers = actual->Estimate(observations).cpu();
        CHECK(torch::allclose(multipliers, expected, 0, 1e-3));
        if (step >= 2 || mode == 2) {
            for (int lane = 0; lane < 2; ++lane) {
                const auto value = prediction_errors[lane].item<double>();
                errors.push_back(value);
                const auto previous_weight = ema_weight * 0.75;
                ema_weight = previous_weight + 1.0;
                const auto difference = value - ema_mean;
                ema_mean += difference / ema_weight;
                ema_m2 = 0.75 * ema_m2 + difference * difference * previous_weight / ema_weight;
            }
        }
        // 公開 checkpoint の観測統計を読み、全観測の倍精度の平均・標準偏差と比較する。
        std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
        anet::OutputArchive output(saved);
        actual->Save(output);
        anet::InputArchive input(saved);
        std::string ignored;
        for (int field = 0; field < 6; ++field) input.Read(ignored);
        int64_t count;
        input.Read(count);
        REQUIRE(count == 2);
        for (int key_index = 0; key_index < count; ++key_index) {
            std::string key;
            input.Read(key);
            input.Read(ignored);
            anet::rl::RunningMeanStd statistics({2}, 1.0);
            statistics.Load(input);
            float clip_ratio, outlier_ratio;
            input.Read(clip_ratio);
            input.Read(outlier_ratio);
            const auto all = torch::cat(history.at(key), 0);
            CHECK(torch::allclose(statistics.GetMean(), all.mean(0), 1e-4, 0));
            CHECK(torch::allclose(statistics.GetStd(), torch::var(all, {0}, true).sqrt() + 1.0, 1e-4, 0));
        }
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
    anet::test::LogCaptureGuard logs;
    anet::InputArchive input(bytes);
    changed->Load(input);
    bytes.clear();
    bytes.seekg(0);
    anet::InputArchive again(bytes);
    changed->Load(again);
    logs.Flush();
    CHECK(std::ranges::count_if(logs.Records(), [](const auto& record) {
        return record.level == wxLOG_Message
            && record.message.find("RND checkpoint optimizer options replaced") != std::string::npos;
    }) == 1);
    CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 0);
    CHECK(torch::equal(original->Estimate(obs), changed->Estimate(obs)));
    original->Learn(obs);
    changed->Learn(obs);
    original->Estimate(obs);
    changed->Estimate(obs);
    CHECK(original->Diagnostics().at(kErrorStep) != changed->Diagnostics().at(kErrorStep));
}




// 公開checkpointのRND節を読む。private stateや偽推定器をテスト用に公開しない。


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


TEST_CASE("RND mixed observation snapshots and checkpoints preserve subsequent learning", "[prd130][intrinsic][checkpoint]")
{
    // 混合辞書の公開口で、warm-up 境界のあとも保存・Clone・採点・補助学習が同じ結果になる。
    anet::nn::InitNN();
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");
    const auto mode = GENERATE(0, 1, 2, 3);
    auto data = RndTestConfig();
    if (mode == 1) data.Set("rnd.error_norm.average_mode", anet::rl::intrinsic::kErrorAverageEma);
    if (mode == 2) data.Set("rnd.statistics_mode", anet::rl::intrinsic::kStatisticsVarianceFeedback);
    if (mode == 3) data.Set("rnd.warmup_mode", anet::rl::intrinsic::kWarmupCollectErrors);
    for (const auto* role : {kTargetRole, kPredictorRole})
        data.Set(std::string(kRndPrefix) + "." + role + ".net.branch.[feature].bind",
            std::string(kReferenceGrid) + ", " + kObservation + ", " + kCategory);
    auto spec = RndTestSpec();
    spec[kReferenceGrid] = {.type = anet::SpaceType::Vector, .shape = {2}, .dtype = torch::kUInt8};
    spec[kCategory] = {.type = anet::SpaceType::Vector, .shape = {1}, .dtype = torch::kInt64, .num_classes = 3};
    const auto create = [&] { return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, kRndPrefix, spec, device, anet::SeedMaker(123)); };
    const auto observation = [](int step) {
        auto obs = RndTestObservation();
        obs.Set(kObservation, obs.At(kObservation) * (step + 1.0f));
        obs.Set(kReferenceGrid, torch::tensor({{step + 1, step + 3}, {step + 7, step + 11}}, torch::kUInt8));
        obs.Set(kCategory, torch::tensor({{step % 3}, {(step + 1) % 3}}, torch::kInt64));
        return obs;
    };
    auto source = create();
    for (int step = 0; step < 3; ++step) {
        source->Estimate(observation(step));
        source->Learn(observation(step));
    }
    auto snapshot = source->Clone();
    std::stringstream state(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(state);
    source->Save(output);
    auto restored = create();
    anet::InputArchive input(state);
    restored->Load(input);
    for (int step = 3; step < 8; ++step) {
        const auto obs = observation(step);
        const auto expected = source->Estimate(obs);
        CHECK(torch::equal(expected, snapshot->Estimate(obs)));
        CHECK(torch::equal(expected, restored->Estimate(obs)));
        const auto learned = source->Learn(obs);
        CHECK(snapshot->Learn(obs) == learned);
        CHECK(restored->Learn(obs) == learned);
        const auto score = source->EstimateWithoutUpdate(obs);
        for (const auto* copy : {snapshot.get(), restored.get()}) {
            const auto actual = copy->EstimateWithoutUpdate(obs);
            CHECK(torch::equal(score.multipliers, actual.multipliers));
            for (const auto& [key, value] : score.diagnostics)
                CHECK(torch::allclose(value, actual.diagnostics.at(key), 0, 0, true));
            for (const auto& [key, value] : source->Diagnostics()) {
                if (std::isnan(value)) CHECK(std::isnan(copy->Diagnostics().at(key)));
                else CHECK(value == copy->Diagnostics().at(key));
            }
        }
    }
}

TEST_CASE("RND read-only scoring leaves subsequent learning unchanged", "[prd127][intrinsic]")
{
    // 公開Factoryから読むだけの採点を挟み、学習の返値と全診断が一致することを確かめる。
    anet::nn::InitNN();
    auto data = RndTestConfig();
    const auto mode = GENERATE(0, 1, 2);
    if (mode == 1) data.Set("rnd.error_norm.average_mode", anet::rl::intrinsic::kErrorAverageEma);
    if (mode == 2) data.Set("rnd.statistics_mode", anet::rl::intrinsic::kStatisticsVarianceFeedback);
    data.Set("rnd.warmup_steps", 0);
    const auto create = [&] { return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123)); };
    auto reference = create();
    auto observed = create();
    for (int step = 0; step < 6; ++step) {
        const anet::TensorDict obs(kObservation, RndTestObservation().At(kObservation) * (step + 1.0f));
        const auto score = observed->EstimateWithoutUpdate(obs);
        REQUIRE(score.multipliers.sizes() == torch::IntArrayRef({2}));
        REQUIRE(torch::isfinite(score.diagnostics.at(kLaneError)).all().item<bool>());
        CHECK(torch::equal(reference->Estimate(obs), observed->Estimate(obs)));
        CHECK(reference->Learn(obs) == observed->Learn(obs));
        for (const auto& [key, value] : reference->Diagnostics()) {
            const auto actual = observed->Diagnostics().at(key);
            if (std::isnan(value)) CHECK(std::isnan(actual));
            else CHECK(actual == value);
        }
    }
}

TEST_CASE("RND scoring snapshots remain independent of training", "[prd127][intrinsic]")
{
    // 方式ごとに公開Factoryで育て、Clone の採点と独立した補助学習を確かめる。
    anet::nn::InitNN();
    const auto compute_device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (compute_device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");
    const auto mode = GENERATE(0, 1, 2);
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
    if (mode == 1) data.Set("rnd.error_norm.average_mode", anet::rl::intrinsic::kErrorAverageEma);
    if (mode == 2) data.Set("rnd.statistics_mode", anet::rl::intrinsic::kStatisticsVarianceFeedback);
    const auto create = [&] { return anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, kRndPrefix, RndTestSpec(), compute_device, anet::SeedMaker(123)); };
    auto source = create();
    auto reference = create();
    for (int step = 0; step < 3; ++step) {
        const anet::TensorDict obs(kObservation, RndTestObservation().At(kObservation) * (step + 1.0f));
        source->Estimate(obs); reference->Estimate(obs);
        source->Learn(obs); reference->Learn(obs);
    }
    const auto rng_before = at::detail::getDefaultCPUGenerator().get_state().clone();
    const auto device_rng_before = at::globalContext().defaultGenerator(compute_device).get_state().clone();
    auto snapshot = source->Clone();
    CHECK(torch::equal(rng_before, at::detail::getDefaultCPUGenerator().get_state()));
    CHECK(torch::equal(device_rng_before, at::globalContext().defaultGenerator(compute_device).get_state()));
    const auto obs = RndTestObservation();
    const auto expected = source->EstimateWithoutUpdate(obs);
    for (int step = 0; step < 3; ++step) {
        const auto actual = snapshot->EstimateWithoutUpdate(obs);
        CHECK(torch::equal(expected.multipliers, actual.multipliers));
        for (const auto& [key, value] : expected.diagnostics)
            CHECK(torch::allclose(value, actual.diagnostics.at(key), 0, 0, true));
    }
    snapshot->Estimate(obs);
    snapshot->Learn(obs);
    CHECK(torch::equal(source->Estimate(obs), reference->Estimate(obs)));
    CHECK(source->Learn(obs) == reference->Learn(obs));
    for (const auto& [key, value] : source->Diagnostics()) {
        const auto actual = reference->Diagnostics().at(key);
        if (std::isnan(value)) CHECK(std::isnan(actual));
        else CHECK(actual == value);
    }
}








TEST_CASE("RND read-only formula follows current statistics and warmup", "[prd127][intrinsic]")
{
    anet::nn::InitNN();
    const auto mode = GENERATE(0, 1, 2);
    const bool centered = GENERATE(false, true);
    const bool clipped = GENERATE(false, true);
    const bool warmup = GENERATE(false, true);
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", warmup ? 12 : 0);
    data.Set("rnd.multiplier_mode", clipped ? "clipped" : anet::rl::intrinsic::kMultiplierUnclipped);
    if (warmup && mode != 2) data.Set("rnd.warmup_mode", "collect_errors");
    if (mode != 2) data.Set("rnd.error_norm.use_centering", centered);
    if (mode == 1) data.Set("rnd.error_norm.average_mode", anet::rl::intrinsic::kErrorAverageEma);
    if (mode == 2) data.Set("rnd.statistics_mode", anet::rl::intrinsic::kStatisticsVarianceFeedback);
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    const auto empty = estimator->EstimateWithoutUpdate(RndTestObservation());
    if (mode != 2) {
        CHECK(torch::isnan(empty.diagnostics.at(kLaneNormalizedError)).all().item<bool>());
        CHECK(torch::equal(empty.multipliers, torch::ones({2})));
    }
    for (int step = 0; step < 3; ++step)
        estimator->Estimate(anet::TensorDict(kObservation, RndTestObservation().At(kObservation) * (step + 1.0f)));
    const auto moments = estimator->Diagnostics();
    const auto mean = static_cast<double>(moments.at(kErrorMean));
    const auto std = static_cast<double>(moments.at(kErrorStd));
    const auto obs = RndTestObservation();
    const auto result = estimator->EstimateWithoutUpdate(obs);
    // 公開した統計から式を手計算し、各laneを次の1件としてEstimateした倍率とも比較する。
    for (int64_t lane = 0; lane < 2; ++lane) {
        const auto error = result.diagnostics.at(kLaneError)[lane].item<double>();
        auto expected = (error - (centered ? mean : 0.0)) / std;
        if (mode == 2) {
            const auto difference = error - mean;
            const auto next_mean = mean + difference / 7.0;
            const auto next_variance = (std * std * 7.0 + difference * difference * 7.0 / 8.0) / 8.0;
            expected = (error - next_mean) / next_variance;
        }
        CHECK(result.diagnostics.at(kLaneNormalizedError)[lane].item<float>() == Catch::Approx(expected).epsilon(1e-4));
        const auto multiplier = warmup ? 1.0 : clipped ? std::clamp(1.0 + expected, 1.0, 5.0) : 1.0 + expected;
        CHECK(result.multipliers[lane].item<float>() == Catch::Approx(multiplier).epsilon(1e-4));
        auto copy = estimator->Clone();
        const anet::TensorDict single(kObservation, obs.At(kObservation).slice(0, lane, lane + 1));
        CHECK(result.multipliers[lane].item<float>() == Catch::Approx(copy->Estimate(single)[0].item<float>()).epsilon(1e-5));
    }
}









namespace {
    // 出荷列の$runner欄を供給する。採点・Actor・行生成は本物の公開経路を使う。
}


TEST_CASE("Read-only normalization leaves zero variances unavailable", "[prd127][intrinsic]")
{
    anet::nn::InitNN();
    const auto mode = GENERATE(0, 1, 2);
    CAPTURE(mode);
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
    data.Set("rnd.head.init.mode", kConstantInit);
    data.Set("rnd.head.init.constant_val", 0.0f);
    data.Set("rnd.obs_norm.pass_through", true);
    if (mode != 0) data.Set("rnd.statistics_mode", anet::rl::intrinsic::kStatisticsVarianceFeedback);
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    const auto observation = RndTestObservation();
    if (mode == 0) estimator->Estimate(observation);
    if (mode != 0) {
        // 公開保存形式から、誤差または観測の一方の分散だけが0の状態を復元する。
        const double error = estimator->EstimateWithoutUpdate(observation).diagnostics.at(kLaneError)[0].item<float>();
        std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
        anet::OutputArchive archive(saved);
        estimator->Save(archive);
        anet::InputArchive input(saved);
        std::stringstream replaced(std::ios::in | std::ios::out | std::ios::binary);
        anet::OutputArchive output(replaced);
        std::string text;
        for (int field = 0; field < 5; ++field) { input.Read(text); output.Write(text); }
        int64_t count;
        double mean, variance;
        input.Read(count); input.Read(mean); input.Read(variance);
        output.Write(count); output.Write(error); output.Write(mode == 1 ? 0.0 : 1.0);
        input.Read(text); output.Write(text);
        torch::Tensor tensor;
        input.Read(tensor); output.Write(tensor);
        input.Read(tensor); output.Write(mode == 2 ? torch::zeros_like(tensor) : tensor);
        replaced << saved.rdbuf();
        anet::InputArchive restored(replaced);
        estimator->Load(restored);
    }
    const auto result = estimator->EstimateWithoutUpdate(observation);
    CHECK(torch::isfinite(result.diagnostics.at(kLaneError)).all().item<bool>());
    CHECK(torch::isnan(result.diagnostics.at(kLaneNormalizedError)).all().item<bool>());
    CHECK(torch::equal(result.multipliers, torch::ones({2})));
}

TEST_CASE("Read-only multiplier preserves overflow exactly as training does", "[prd127][prd127_overflow]")
{
    anet::nn::InitNN();
    const bool clipped = GENERATE(false, true);
    auto data = RndTestConfig();
    data.Set("rnd.warmup_steps", 0);
    data.Set("rnd.obs_norm.pass_through", true);
    data.Set("rnd.error_norm.average_mode", anet::rl::intrinsic::kErrorAverageEma);
    data.Set("rnd.multiplier_mode", clipped ? anet::rl::intrinsic::kMultiplierClipped : anet::rl::intrinsic::kMultiplierUnclipped);
    auto estimator = anet::rl::intrinsic::LifelongNoveltyEstimatorFactory::Create(
        data, kRndPrefix, RndTestSpec(), torch::kCPU, anet::SeedMaker(123));
    // 正当なEMA checkpointで、小さい正の除数を設定してfloatのoverflowを通す。
    std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive archive(saved);
    estimator->Save(archive);
    anet::InputArchive input(saved);
    std::stringstream replaced(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(replaced);
    std::string text;
    for (int field = 0; field < 5; ++field) { input.Read(text); output.Write(text); }
    anet::rl::StateSpec state_spec;
    state_spec.obs_spec = RndTestSpec();
    auto normalizer = anet::rl::ObservationNormalizerFactory(
        anet::rl::intrinsic::RndNoveltyEstimatorConfig(data, kRndPrefix).obs_norm).CreateObservationNormalizer(state_spec);
    normalizer->Load(input);
    normalizer->Save(output);
    double moment;
    for (int field = 0; field < 3; ++field) input.Read(moment);
    output.Write(1.0); output.Write(0.0); output.Write(4e-76);
    replaced << saved.rdbuf();
    anet::InputArchive restored(replaced);
    estimator->Load(restored);
    auto training = estimator->Clone();
    const anet::TensorDict observation(kObservation, RndTestObservation().At(kObservation) * 1e6f);
    const auto result = estimator->EstimateWithoutUpdate(observation);
    REQUIRE(torch::isinf(result.diagnostics.at(kLaneNormalizedError)).all().item<bool>());
    CHECK(torch::equal(result.multipliers, training->Estimate(observation)));
}
