#include "anet/catch_test.hpp"
#include "anet/scaler.hpp"

#include <cmath>

TEST_CASE("Constant reward scaler exposes a defined clipping ratio", "[scaler][metrics]")
{
    // factory の公開経路から、初回は未成立、非クリップの Scale 後はゼロとなる契約を確認する。
    anet::rl::RewardScalerConfig config;
    config.use_dynamic_scaling = false;
    config.use_clipping = false;
    config.use_auto_post_scale = false;
    config.constant_scale = 2.0f;
    anet::rl::RewardScalerFactory factory(config);
    auto scaler = factory.CreateRewardScaler(0.99f);
    const auto initial = scaler->GetScalar(anet::rl::RewardScaler::kKeyClipRatio);
    REQUIRE(initial.has_value());
    CHECK(std::isnan(*initial));
    const auto result = scaler->Scale(torch::tensor({ -2.0f, 3.0f }));
    CHECK(torch::equal(result, torch::tensor({ -4.0f, 6.0f })));
    CHECK(scaler->GetScalar(anet::rl::RewardScaler::kKeyClipRatio) == 0.0f);
}

TEST_CASE("Reward diagnostic scaling preserves statistics and clipping metrics", "[scaler][prd094][diagnostic_scaling]")
{
    const bool dynamic = GENERATE(false, true);
    anet::rl::RewardScalerConfig config;
    config.use_dynamic_scaling = dynamic;
    config.use_clipping = true;
    config.clip_range = 1.5f;
    config.use_auto_post_scale = false;
    config.manual_post_scale = 1.0f;
    config.constant_scale = 2.0f;
    auto scaler = anet::rl::RewardScalerFactory(config).CreateRewardScaler(0.99f);
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

TEST_CASE("Published reward transform follows statistics and Reset", "[scaler][prd094][diagnostic_scaling]")
{
    anet::rl::RewardScalerConfig config;
    config.use_dynamic_scaling = true;
    config.use_clipping = true;
    config.clip_range = 3.0f;
    config.epsilon = 0.25f;
    config.use_auto_post_scale = false;
    config.manual_post_scale = 2.0f;
    auto scaler = anet::rl::RewardScalerFactory(config).CreateRewardScaler(0.99f);
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

TEST_CASE("Reward auto post scale uses the discounted reference Q scale", "[prd122][scaler][auto_post_scale]")
{
    // gammaと1-gammaを取り違えると異なる値になる条件で、公開の変換を通す。
    constexpr float gamma = 0.8f;
    anet::rl::RewardScalerConfig config;
    config.use_dynamic_scaling = false;
    config.use_clipping = false;
    config.use_auto_post_scale = true;
    config.reference_q_std = 8.0f;
    config.constant_scale = 1.5f;
    auto scaler = anet::rl::RewardScalerFactory(config).CreateRewardScaler(gamma);
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

TEST_CASE("Robust normalization admits warmup rows and rejects only outlier rows", "[prd122][scaler][robust_update]")
{
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
}

TEST_CASE("Observation SymLog keeps its positive non-unit scale", "[prd122][scaler][symlog]")
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
