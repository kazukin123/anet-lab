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
