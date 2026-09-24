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
