#include "anet/test/catch.hpp"
#include "anet/scaler.hpp"

#include <cmath>
#include <array>
#include <sstream>

#include "anet/test/util.hpp"
#include "scaler_impl.hpp"

// RFD0121の入力・公開識別子を、同じ意味の検査で共有する。
constexpr const char* kRfd0121IntrinsicRewardScaler = "intrinsic.reward_scaler";
const std::string kDirectGammaKey = anet::ConfigFullKeyTable::FormatKeyNotReadFromConfig("gamma");

TEST_CASE("Return reward scaling follows per-lane discounted returns", "[rfd0121][scaler]")
{
    // 2 laneの割引和の累積分散を、公開factoryの変換結果から確かめる。
    anet::rl::RewardScalerConfig config;
    config.dynamic_mode = anet::rl::RewardScaler::kReturnStdState;
    config.use_auto_post_scale = false;
    config.epsilon = 0.01f;
    auto scaler = anet::rl::RewardScalerFactory(config).CreateRewardScaler(0.5f, 2, kDirectGammaKey);
    const auto first = torch::tensor({1.0f, 3.0f});
    CHECK(torch::allclose(scaler->Scale(first), first / (std::sqrt(2.0f) + config.epsilon)));
    const auto second = torch::ones({2});
    // 既存の統計部品と同じく、標本数-1で割る標準偏差を使う。
    const float expected_std = std::sqrt(2.5f / 3.0f) + config.epsilon;
    CHECK(torch::allclose(scaler->Scale(second), second / expected_std));
    const auto count = scaler->GetScalar(anet::rl::RewardScaler::kKeyCount);
    CHECK(torch::allclose(scaler->ScaleWithoutUpdate(first), first / expected_std));
    CHECK(scaler->GetScalar(anet::rl::RewardScaler::kKeyCount) == count);
}

TEST_CASE("Return scaler checkpoints retain statistics and reset only changed lanes", "[rfd0121][scaler][checkpoint]")
{
    const int lanes = GENERATE(2, 3);
    anet::rl::RewardScalerConfig config;
    config.dynamic_mode = anet::rl::RewardScaler::kReturnStdState;
    config.use_auto_post_scale = false;
    config.epsilon = 0.01f;
    auto source = anet::rl::RewardScalerFactory(config, kRfd0121IntrinsicRewardScaler).CreateRewardScaler(0.5f, 2, kDirectGammaKey);
    source->Scale(torch::tensor({1.0f, 3.0f}));
    source->Scale(torch::tensor({2.0f, 1.0f}));
    std::stringstream checkpoint(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(checkpoint);
    source->Save(output);
    auto restored = anet::rl::RewardScalerFactory(config, kRfd0121IntrinsicRewardScaler).CreateRewardScaler(0.5f, lanes, kDirectGammaKey);
    anet::test::LogCaptureGuard logs;
    anet::InputArchive input(checkpoint);
    restored->Load(input);
    CHECK(restored->GetScalar(anet::rl::RewardScaler::kKeyCount) == source->GetScalar(anet::rl::RewardScaler::kKeyCount));
    CHECK(torch::equal(restored->ScaleWithoutUpdate(torch::ones({lanes})), source->ScaleWithoutUpdate(torch::ones({lanes}))));
    const auto reward = lanes == 2 ? torch::tensor({1.0f, 2.0f}) : torch::tensor({1.0f, 2.0f, 4.0f});
    const auto actual = restored->Scale(reward);
    if (lanes == 2) {
        CHECK(torch::equal(actual, source->Scale(reward)));
    } else {
        // 保存前の割引和4件に、初期化された3 laneの報酬を足して標本分散を計算する。
        const std::array<double, 7> returns{1, 3, 2.5, 2.5, 1, 2, 4};
        double mean = 0, variance = 0;
        for (const double value : returns) mean += value / returns.size();
        for (const double value : returns) variance += (value - mean) * (value - mean) / (returns.size() - 1);
        CHECK(torch::allclose(actual, reward / (std::sqrt(variance) + config.epsilon)));
    }
    logs.Flush();
    CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == (lanes == 2 ? 0 : 1));
    auto wrong_config = config;
    wrong_config.dynamic_mode = anet::rl::RewardScaler::kRewardStdState;
    auto wrong = anet::rl::RewardScalerFactory(wrong_config).CreateRewardScaler(0.5f, 2, kDirectGammaKey);
    checkpoint.clear();
    checkpoint.seekg(0);
    anet::InputArchive wrong_input(checkpoint);
    CHECK_THROWS_WITH(wrong->Load(wrong_input), Catch::Matchers::ContainsSubstring("kind"));
}

TEST_CASE("Constant reward scaler exposes a defined clipping ratio", "[scaler][metrics]")
{
    // factory の公開経路から、初回は未成立、非クリップの Scale 後はゼロとなる契約を確認する。
    anet::rl::RewardScalerConfig config;
    config.use_dynamic_scaling = false;
    config.use_clipping = false;
    config.use_auto_post_scale = false;
    config.constant_scale = 2.0f;
    anet::rl::RewardScalerFactory factory(config);
    auto scaler = factory.CreateRewardScaler(0.99f, 2, kDirectGammaKey);
    const auto initial = scaler->GetScalar(anet::rl::RewardScaler::kKeyClipRatio);
    REQUIRE(initial.has_value());
    CHECK(std::isnan(*initial));
    const auto result = scaler->Scale(torch::tensor({ -2.0f, 3.0f }));
    CHECK(torch::equal(result, torch::tensor({ -4.0f, 6.0f })));
    CHECK(scaler->GetScalar(anet::rl::RewardScaler::kKeyClipRatio) == 0.0f);
}

TEST_CASE("Reward diagnostic scaling preserves statistics and clipping metrics", "[scaler][rfd0094][diagnostic_scaling]")
{
    const bool dynamic = GENERATE(false, true);
    anet::rl::RewardScalerConfig config;
    config.use_dynamic_scaling = dynamic;
    config.use_clipping = true;
    config.clip_range = 1.5f;
    config.use_auto_post_scale = false;
    config.manual_post_scale = 1.0f;
    config.constant_scale = 2.0f;
    auto scaler = anet::rl::RewardScalerFactory(config).CreateRewardScaler(0.99f, 2, kDirectGammaKey);
    scaler->Scale(torch::tensor({ -1.0f, 3.0f }));
    const auto count = scaler->GetScalar(anet::rl::RewardScaler::kKeyCount).value();
    const auto ratio = scaler->GetScalar(anet::rl::RewardScaler::kKeyClipRatio).value();
    const auto scale = scaler->GetScalar(anet::rl::RewardScaler::kKeyScale).value();
    const auto rewards = torch::tensor({ -4.0f, 2.0f });
    const anet::rl::RewardScaler& read_only = *scaler;
    const auto actual = read_only.ScaleWithoutUpdate(rewards);
    CHECK(torch::allclose(actual, torch::clamp(rewards * scale, -1.5f, 1.5f)));
    CHECK(torch::equal(rewards, torch::tensor({ -4.0f, 2.0f })));
    if (dynamic) CHECK(scaler->GetScalar(anet::rl::RewardScaler::kKeyCount) == count);
    else CHECK(std::isnan(*scaler->GetScalar(anet::rl::RewardScaler::kKeyCount)));
    CHECK(scaler->GetScalar(anet::rl::RewardScaler::kKeyClipRatio) == ratio);
}

TEST_CASE("Published reward transform follows statistics and Reset", "[scaler][rfd0094][diagnostic_scaling]")
{
    anet::rl::RewardScalerConfig config;
    config.use_dynamic_scaling = true;
    config.use_clipping = true;
    config.clip_range = 3.0f;
    config.epsilon = 0.25f;
    config.use_auto_post_scale = false;
    config.manual_post_scale = 2.0f;
    auto scaler = anet::rl::RewardScalerFactory(config).CreateRewardScaler(0.99f, 2, kDirectGammaKey);
    const auto probe = torch::tensor({-0.5f, 0.5f});
    CHECK(torch::allclose(scaler->ScaleWithoutUpdate(probe), probe / 1.25f * 2.0f));
    for (const auto samples : {torch::tensor({-1.0f, 3.0f}), torch::tensor({-10.0f, 10.0f})}) {
        scaler->Scale(samples);
        const auto std = scaler->GetScalar(anet::rl::RewardScaler::kKeyStd).value();
        CHECK(torch::allclose(scaler->ScaleWithoutUpdate(probe), torch::clamp(probe / std * 2.0f, -3.0f, 3.0f)));
    }
    scaler->Reset();
    CHECK(scaler->GetScalar(anet::rl::RewardScaler::kKeyCount) == 0.0f);
    CHECK(torch::allclose(scaler->ScaleWithoutUpdate(probe), probe / 1.25f * 2.0f));
}

TEST_CASE("Reward auto post scale uses the discounted reference Q scale", "[rfd0122][scaler][auto_post_scale]")
{
    // gammaと1-gammaを取り違えると異なる値になる条件で、公開の変換を通す。
    constexpr float gamma = 0.8f;
    anet::rl::RewardScalerConfig config;
    config.use_dynamic_scaling = false;
    config.use_clipping = false;
    config.use_auto_post_scale = true;
    config.reference_q_std = 8.0f;
    config.constant_scale = 1.5f;
    auto scaler = anet::rl::RewardScalerFactory(config).CreateRewardScaler(gamma, 2, kDirectGammaKey);
    const auto rewards = torch::tensor({-2.0f, 3.0f});
    const float expected_scale = config.constant_scale * config.reference_q_std * (1.0f - gamma);
    CHECK(torch::allclose(scaler->Scale(rewards), rewards * expected_scale));
}

static std::shared_ptr<anet::rl::ObservationNormalizer> MakeScalerObservationNormalizer(
    const anet::rl::ObservationNormalizerConfig& config, int64_t width)
{
    // 製品のfactoryを通し、格子でなく通常のvector観測を正規化する。
    anet::rl::StateSpec spec;
    spec.obs_spec[anet::rl::ObsKeys::kVector] = anet::TensorSpec{
        .type = anet::SpaceType::Vector, .shape = {width}, .dtype = torch::kFloat32 };
    return anet::rl::ObservationNormalizerFactory(config).CreateObservationNormalizer(spec);
}

TEST_CASE("Robust normalization admits warmup rows and rejects only outlier rows", "[rfd0122][scaler][robust_update]")
{
    anet::test::LogCaptureGuard logs;
    // 件数は行数で数える。境目の直前の外れ値は採用し、境目の直後は除外する。
    anet::rl::ObservationNormalizerConfig config;
    config.use_clipping = false;
    config.use_robust_update = true;
    config.robust_warmup_count = 2;
    config.robust_std_threshold = 1.0f;
    config.post_process_type = anet::rl::kPostProcessNone;
    auto normalizer = MakeScalerObservationNormalizer(config, 2);
    // warmupの0と10から、閾値の基準になる平均と標本標準偏差を手で求める。
    const float expected_mean = (0.0f + 10.0f) / 2.0f;
    const float expected_std = std::sqrt(50.0f) + config.epsilon;
    const auto update = [&](const torch::Tensor& values) {
        anet::TensorDict obs;
        obs.Set(anet::rl::ObsKeys::kVector, values);
        normalizer->NormalizeAndUpdateStats(obs);
    };
    update(torch::tensor({{0.0f, 0.0f}}));
    CHECK(*normalizer->GetScalar(anet::rl::ObservationNormalizer::kKeyCount) == 1.0f);
    update(torch::tensor({{10.0f, 10.0f}}));
    CHECK(*normalizer->GetScalar(anet::rl::ObservationNormalizer::kKeyCount) == 2.0f);
    CHECK(*normalizer->GetScalar(anet::rl::ObservationNormalizer::kKeyMeanMean) == expected_mean);
    CHECK(*normalizer->GetScalar(anet::rl::ObservationNormalizer::kKeyStdMean)
        == Catch::Approx(expected_std).margin(1e-6));
    update(torch::tensor({{100.0f, 5.0f}}));
    CHECK(*normalizer->GetScalar(anet::rl::ObservationNormalizer::kKeyCount) == 2.0f);
    // 片方の要素だけが外れ値でも行全体を落とし、正常な別の行だけを採る。
    update(torch::tensor({{5.0f, 5.0f}, {100.0f, 9.0f}}));
    CHECK(*normalizer->GetScalar(anet::rl::ObservationNormalizer::kKeyCount) == 3.0f);
    CHECK(*normalizer->GetScalar(anet::rl::ObservationNormalizer::kKeyMeanMean) == expected_mean);
    CHECK(*normalizer->GetScalar(anet::rl::ObservationNormalizer::kKeyRobustOutlierRatio) == 0.25f);
    logs.Flush();
    CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 0);
}

TEST_CASE("Observation SymLog keeps its positive non-unit scale", "[rfd0122][scaler][symlog]")
{
    // 統計未更新の標準偏差1+epsilonから手で計算し、scaleの掛け戻しを検出する。
    anet::rl::ObservationNormalizerConfig config;
    config.use_clipping = false;
    config.use_robust_update = false;
    config.epsilon = 0.25f;
    config.post_process_type = anet::rl::kPostProcessSymLog;
    config.post_process_threshold = 2.0f;
    auto normalizer = MakeScalerObservationNormalizer(config, 1);
    anet::TensorDict obs;
    obs.Set(anet::rl::ObsKeys::kVector, torch::tensor({{-5.0f}, {0.0f}, {5.0f}}));
    const float magnitude = 2.0f * std::log1p((5.0f / 1.25f) / 2.0f);
    const auto actual = normalizer->Normalize(obs).At(anet::rl::ObsKeys::kVector);
    CHECK(torch::allclose(actual, torch::tensor({{-magnitude}, {0.0f}, {magnitude}})));
}

TEST_CASE("Observation statistics update without normalizing a mixed dictionary", "[rfd0130][scaler]")
{
    // Factory から統計更新専用の口を通し、通常更新と同じ次回の正規化になることを確かめる。
    constexpr char kCategoryKey[] = "category";
    anet::rl::StateSpec spec;
    spec.obs_spec[anet::rl::ObsKeys::kVector] = {
        .type = anet::SpaceType::Vector, .shape = {2}, .dtype = torch::kFloat32};
    spec.obs_spec[anet::rl::ObsKeys::kGrid] = {
        .type = anet::SpaceType::Vector, .shape = {2}, .dtype = torch::kUInt8};
    spec.obs_spec[kCategoryKey] = {
        .type = anet::SpaceType::Vector, .shape = {1}, .dtype = torch::kInt64, .num_classes = 3};
    anet::rl::ObservationNormalizerConfig config;
    config.use_robust_update = false;
    config.use_centering = true;
    config.clip_range = 0.25f;
    config.post_process_type = anet::rl::kPostProcessNone;
    auto observed = anet::rl::ObservationNormalizerFactory(config).CreateObservationNormalizer(spec);
    auto reference = anet::rl::ObservationNormalizerFactory(config).CreateObservationNormalizer(spec);
    anet::TensorDict values(anet::rl::ObsKeys::kVector, torch::tensor({{1.0f, 3.0f}, {5.0f, 7.0f}}));
    values.Set(anet::rl::ObsKeys::kGrid, torch::tensor({{0, 255}, {128, 16}}, torch::kUInt8));
    values.Set(kCategoryKey, torch::tensor({{0}, {2}}, torch::kInt64));
    const auto before_clip = observed->GetScalar(anet::rl::ObservationNormalizer::kKeyClipRatio);
    observed->UpdateStats(values);
    reference->NormalizeAndUpdateStats(values);
    CHECK(observed->GetScalar(anet::rl::ObservationNormalizer::kKeyCount) == 2.0f);
    CHECK(observed->GetScalar(anet::rl::ObservationNormalizer::kKeyMeanMean)
        == reference->GetScalar(anet::rl::ObservationNormalizer::kKeyMeanMean));
    CHECK(observed->GetScalar(anet::rl::ObservationNormalizer::kKeyStdMean)
        == reference->GetScalar(anet::rl::ObservationNormalizer::kKeyStdMean));
    CHECK(observed->GetScalar(anet::rl::ObservationNormalizer::kKeyClipRatio) == before_clip);
    const auto actual = observed->Normalize(values);
    const auto expected = reference->Normalize(values);
    for (const auto& [key, tensor] : actual) CHECK(torch::equal(tensor, expected.At(key)));
    CHECK(torch::equal(actual.At(kCategoryKey), values.At(kCategoryKey)));
    CHECK(actual.At(anet::rl::ObsKeys::kGrid).scalar_type() == torch::kUInt8);
}

TEST_CASE("Statistics-only update rejects a mismatched observation before mutation", "[rfd0130][scaler]")
{
    // broadcast で計算できても、宣言した feature 軸と異なる入力は統計を進めない。
    anet::rl::ObservationNormalizerConfig config;
    config.use_robust_update = false;
    auto normalizer = MakeScalerObservationNormalizer(config, 2);
    const anet::TensorDict invalid(anet::rl::ObsKeys::kVector, torch::ones({2, 1}));
    CHECK_THROWS(normalizer->UpdateStats(invalid));
    CHECK(normalizer->GetScalar(anet::rl::ObservationNormalizer::kKeyCount) == 0.0f);
}

TEST_CASE("Statistics-only update preserves robust and constant normalization behavior", "[rfd0130][scaler]")
{
    // 外れ値を含む行、全行除外、素通し、固定値を通常の更新と対比する。
    const auto mode = GENERATE(0, 1, 2);
    anet::rl::ObservationNormalizerConfig config;
    config.use_clipping = false;
    config.post_process_type = anet::rl::kPostProcessNone;
    config.robust_warmup_count = 2;
    config.robust_std_threshold = 1.0f;
    config.pass_through = mode == 1;
    config.use_dynamic_scaling = mode != 2;
    auto observed = MakeScalerObservationNormalizer(config, 2);
    auto reference = MakeScalerObservationNormalizer(config, 2);
    for (const auto& values : {torch::tensor({{0.0f, 0.0f}, {10.0f, 10.0f}}),
                              torch::tensor({{5.0f, 5.0f}, {100.0f, 9.0f}}),
                              torch::tensor({{1000.0f, 1000.0f}, {1000.0f, 1000.0f}})}) {
        const anet::TensorDict obs(anet::rl::ObsKeys::kVector, values);
        observed->UpdateStats(obs);
        reference->NormalizeAndUpdateStats(obs);
        CHECK(observed->GetScalar(anet::rl::ObservationNormalizer::kKeyCount)
            == reference->GetScalar(anet::rl::ObservationNormalizer::kKeyCount));
        CHECK(observed->GetScalar(anet::rl::ObservationNormalizer::kKeyRobustOutlierRatio)
            == reference->GetScalar(anet::rl::ObservationNormalizer::kKeyRobustOutlierRatio));
        CHECK(torch::equal(observed->Normalize(obs).At(anet::rl::ObsKeys::kVector),
                           reference->Normalize(obs).At(anet::rl::ObsKeys::kVector)));
    }
}

TEST_CASE("Read-only normalization uses current statistics after updates load and reset", "[rfd0130][scaler]")
{
    // 異なる統計で育てた部品へ Load しても、保存元と同じ値を使う。
    anet::rl::ObservationNormalizerConfig config;
    config.use_robust_update = false;
    config.use_clipping = true;
    config.clip_range = 0.5f;
    config.use_centering = true;
    config.post_process_type = anet::rl::kPostProcessNone;
    auto observed = MakeScalerObservationNormalizer(config, 2);
    auto reference = MakeScalerObservationNormalizer(config, 2);
    const anet::TensorDict values(anet::rl::ObsKeys::kVector, torch::tensor({{1.0f, 3.0f}, {5.0f, 9.0f}}));
    reference->NormalizeAndUpdateStats(values);
    observed->UpdateStats(anet::TensorDict(anet::rl::ObsKeys::kVector, torch::tensor({{100.0f, 300.0f}, {500.0f, 900.0f}})));
    std::stringstream checkpoint(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(checkpoint);
    reference->Save(output);
    anet::InputArchive input(checkpoint);
    observed->Load(input);
    CHECK(torch::equal(observed->Normalize(values).At(anet::rl::ObsKeys::kVector),
                       reference->Normalize(values).At(anet::rl::ObsKeys::kVector)));
    const auto clip = observed->GetScalar(anet::rl::ObservationNormalizer::kKeyClipRatio);
    observed->Normalize(anet::TensorDict(anet::rl::ObsKeys::kVector, torch::zeros({2, 2})));
    CHECK(observed->GetScalar(anet::rl::ObservationNormalizer::kKeyClipRatio) == clip);
    observed->UpdateStats(values);
    reference->NormalizeAndUpdateStats(values);
    CHECK(torch::equal(observed->Normalize(values).At(anet::rl::ObsKeys::kVector),
                       reference->Normalize(values).At(anet::rl::ObsKeys::kVector)));
    observed->Reset();
    reference->Reset();
    CHECK(torch::equal(observed->Normalize(values).At(anet::rl::ObsKeys::kVector),
                       reference->Normalize(values).At(anet::rl::ObsKeys::kVector)));
}

TEST_CASE("Normalization accepts a different device than its statistics update", "[rfd0130][scaler]")
{
    if (!torch::cuda::is_available()) SKIP("CUDA is unavailable");
    // 更新に使った device の写しがあっても、別の device で同じ正規化を行う。
    const auto update_device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    anet::rl::ObservationNormalizerConfig config;
    config.use_centering = true;
    config.use_robust_update = false;
    config.post_process_type = anet::rl::kPostProcessNone;
    auto normalizer = MakeScalerObservationNormalizer(config, 2);
    const auto values = torch::tensor({{1.0f, 3.0f}, {5.0f, 9.0f}});
    normalizer->UpdateStats(anet::TensorDict(anet::rl::ObsKeys::kVector, values.to(update_device)));
    const auto expected = normalizer->Normalize(anet::TensorDict(anet::rl::ObsKeys::kVector, values));
    const auto actual = normalizer->Normalize(anet::TensorDict(anet::rl::ObsKeys::kVector, values.to(torch::kCUDA)));
    CHECK(torch::allclose(actual.At(anet::rl::ObsKeys::kVector).cpu(), expected.At(anet::rl::ObsKeys::kVector), 1e-6, 1e-6));
    CHECK(normalizer->GetScalar(anet::rl::ObservationNormalizer::kKeyCount) == 2.0f);
}

TEST_CASE("Batch moments agree with an independent double precision reference", "[rfd0130][scaler]")
{
    // batch の分割、空の入力、1 件の更新を含め、累積の不偏分散を手計算する。
    const auto device = GENERATE(torch::Device(torch::kCPU), torch::Device(torch::kCUDA, 0));
    if (device.is_cuda() && !torch::cuda::is_available()) SKIP("CUDA is unavailable");
    anet::rl::RunningMeanStd statistics({2}, 0.0);
    const auto values = torch::tensor({{1.25f, 3.5f}, {2.5f, 5.0f}, {7.75f, 11.0f}});
    statistics.Update(values.slice(0, 0, 0).to(device));
    CHECK(statistics.GetCount() == 0);
    statistics.Update(values.slice(0, 0, 2).to(device));
    statistics.Update(values.slice(0, 2, 3).to(device));
    const std::array<std::array<double, 3>, 2> columns{{{1.25, 2.5, 7.75}, {3.5, 5.0, 11.0}}};
    for (int column = 0; column < 2; ++column) {
        double mean = 0, variance = 0;
        for (const auto value : columns[column]) mean += value / 3.0;
        for (const auto value : columns[column]) variance += (value - mean) * (value - mean) / 2.0;
        CHECK(statistics.GetMean()[column].item<double>() == Catch::Approx(mean).epsilon(1e-4));
        CHECK(statistics.GetVar()[column].item<double>() == Catch::Approx(variance).epsilon(1e-4));
    }
    CHECK(statistics.GetCount() == 3);
}
