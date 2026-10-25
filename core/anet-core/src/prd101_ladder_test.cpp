#include "anet/catch_test.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <torch/cuda.h>

#include "dqn_based_agent.hpp"

using namespace anet;
using namespace anet::rl::dqn;

// D8の列生成と線形伸縮を倍精度で独立計算する（本体のexpm1式は使わない）。
static double D8LadderValue(double first, double last, double t)
{
    const double a = std::max(first, static_cast<double>(1e-4f));
    const double b = std::max(last, static_cast<double>(1e-4f));
    if (a == b) return first + t * (last - first);
    const double raw = b * std::exp((1.0 - t) * std::log(a / b));
    return last + (first - last) * (raw - b) / (a - b);
}

TEST_CASE("PRD101 spatial endpoints enforce bounds without caller metadata", "[prd101][dqn]")
{
    ActionPolicyConfig::LadderConfig config;
    config.first.value = 0.4f;
    config.last.value = 0.01f;
    SECTION("negative first") { config.first.value = -0.1f; }
    SECTION("last above one") { config.last.value = 1.1f; }
    SECTION("negative phase endpoint") {
        config.last.type = "phased";
        config.last.phases = {"bad"};
        config.last.phase.Set("bad", ProfiledValuePhaseConfig<float>{.value = -0.1f, .steps = 8});
    }
    CHECK_THROWS(SpatialLadderSchedule(config, "log", 4, torch::kCPU));
}

TEST_CASE("PRD101 spatial constants reuse storage and moving endpoints replace it", "[prd101][dqn]")
{
    ActionPolicyConfig::LadderConfig config;
    config.first.value = 0.4f;
    config.last.value = 0.01f;
    SpatialLadderSchedule constant(config, "log", 4, torch::kCPU);
    const auto old_constant = constant.Tensor();
    constant.Update(100);
    CHECK(constant.Tensor().data_ptr() == old_constant.data_ptr());
    config.first.type = "linear";
    config.first.start = 0.4f;
    config.first.end = 0.2f;
    config.first.steps = 8;
    SpatialLadderSchedule moving(config, "log", 4, torch::kCPU);
    const auto old_moving = moving.Tensor();
    moving.Update(4);
    CHECK(moving.Tensor().data_ptr() != old_moving.data_ptr());
    CHECK_FALSE(torch::equal(moving.Tensor(), old_moving));
}

TEST_CASE("PRD101 spatial log endpoints remain continuous across the floor", "[prd101][dqn]")
{
    ActionPolicyConfig::LadderConfig config;
    config.first.value = 0.4f;
    config.last.type = "linear";
    config.last.start = 1e-4f - 1e-9f;
    config.last.end = 1e-4f + 1e-9f;
    config.last.steps = 1;
    SpatialLadderSchedule ladder(config, "log", 8, torch::kCPU);
    const auto before = ladder.Tensor();
    ladder.Update(1);
    CHECK((ladder.Tensor() - before).abs().max().item<float>() < 1e-6f);
}

TEST_CASE("PRD101 spatial first crosses the log floor without cancellation", "[prd101][dqn]")
{
    const auto check_device = [&](torch::Device device) {
        ActionPolicyConfig::LadderConfig config;
        config.last.value = 0.0f;
        config.first.type = "phased";
        config.first.phases = {"above", "floor", "below"};
        config.first.phase.Set("above", ProfiledValuePhaseConfig<float>{.value = 1.0001e-4f, .steps = 1});
        config.first.phase.Set("floor", ProfiledValuePhaseConfig<float>{.value = 1e-4f, .steps = 1});
        config.first.phase.Set("below", ProfiledValuePhaseConfig<float>{.value = 5e-5f, .steps = 1});
        SpatialLadderSchedule ladder(config, "log", 3, device);
        const float firsts[] = {1.0001e-4f, 1e-4f, 5e-5f};
        float previous_middle = 0.0f;
        for (int64_t step = 0; step < 3; ++step) {
            CAPTURE(device, step);
            ladder.Update(step);
            const auto values = ladder.Tensor().to(torch::kCPU);
            CHECK(values[0].item<float>() == 0.0f);
            CHECK(values[2].item<float>() == firsts[step]);
            const double expected = D8LadderValue(firsts[step], 0.0f, 0.5);
            const double tolerance = 32.0 * std::numeric_limits<float>::epsilon() * expected;
            const float middle = values[1].item<float>();
            CHECK(std::abs(middle - expected) <= tolerance);
            if (step == 1) CHECK(std::abs(middle - previous_middle) < 1e-8f);
            previous_middle = middle;
        }
    };
    SECTION("CPU") { check_device(torch::kCPU); }
    SECTION("CUDA") {
        if (!torch::cuda::is_available()) { SKIP("CUDA is unavailable"); }
        check_device(torch::kCUDA);
    }
}

TEST_CASE("PRD101 spatial log ladder preserves a tiny last endpoint", "[prd101][dqn]")
{
    const auto check_device = [](torch::Device device) {
        ActionPolicyConfig::LadderConfig config;
        config.first.value = 0.4f;
        config.last.value = 5e-5f;
        SpatialLadderSchedule ladder(config, "log", 4, device);
        const auto values = ladder.Tensor().to(torch::kCPU);
        // 既存の逆順lane割当てを維持し、指定端点とD8から求めた中間値を確認する。
        CHECK(values[0].item<float>() == config.last.value);
        CHECK(values[3].item<float>() == config.first.value);
        double expected_sum = 0.0;
        for (int64_t i = 0; i < 4; ++i) {
            CAPTURE(device, i);
            const double expected = D8LadderValue(config.first.value, config.last.value, i / 3.0);
            const double tolerance = 32.0 * std::numeric_limits<float>::epsilon() * expected;
            CHECK(std::abs(values[3 - i].item<float>() - expected) <= tolerance);
            expected_sum += expected;
        }
        CHECK(std::abs(ladder.Mean() - expected_sum / 4.0)
            <= 32.0 * std::numeric_limits<float>::epsilon() * expected_sum / 4.0);
    };
    SECTION("CPU") { check_device(torch::kCPU); }
    SECTION("CUDA") {
        if (!torch::cuda::is_available()) { SKIP("CUDA is unavailable"); }
        check_device(torch::kCUDA);
    }
}

TEST_CASE("PRD101 spatial validation reports keys values and expected ranges", "[prd101][config][dqn]")
{
    ConfigData data;
    ActionPolicyConfig policy;
    std::string key, value, expected;
    SECTION("invalid scale") {
        key = "spatial.scale"; value = "invalid"; expected = "expected=log|linear";
        data.Set("policy." + key, value);
        CHECK_THROWS_WITH(SpatialLadderSchedule(policy.spatial.epsilon, value, 4, torch::kCPU),
            Catch::Matchers::ContainsSubstring(key) && Catch::Matchers::ContainsSubstring(value)
            && Catch::Matchers::ContainsSubstring(expected));
    }
    SECTION("negative first") {
        key = "spatial.epsilon.first"; value = "-0.1"; expected = "expected=>=0";
        policy.spatial.epsilon.first.value = -0.1f;
    }
    SECTION("last above one") {
        key = "spatial.epsilon.last"; value = "1.1"; expected = "expected=<=1";
        policy.spatial.epsilon.last.value = 1.1f;
    }
    SECTION("negative phase start") {
        key = "spatial.epsilon.first.phase.[bad].start"; value = "-0.1"; expected = "expected=>=0";
        policy.spatial.epsilon.first.type = "phased";
        policy.spatial.epsilon.first.phases = {"bad"};
        policy.spatial.epsilon.first.phase.Set("bad", ProfiledValuePhaseConfig<float>{
            .type = "linear", .start = -0.1f, .end = 0.1f, .steps = 2});
        data.Set("policy.spatial.epsilon.first.type", "phased");
        data.Set("policy.spatial.epsilon.first.phases", "bad");
        data.Set("policy.spatial.epsilon.first.phase.[bad].type", "linear");
        data.Set("policy.spatial.epsilon.first.phase.[bad].end", "0.1");
        data.Set("policy.spatial.epsilon.first.phase.[bad].steps", "2");
    }
    SECTION("phase end above one") {
        key = "spatial.epsilon.first.phase.[bad].end"; value = "1.1"; expected = "expected=<=1";
        policy.spatial.epsilon.first.type = "phased";
        policy.spatial.epsilon.first.phases = {"bad"};
        policy.spatial.epsilon.first.phase.Set("bad", ProfiledValuePhaseConfig<float>{
            .type = "linear", .start = 0.1f, .end = 1.1f, .steps = 2});
        data.Set("policy.spatial.epsilon.first.type", "phased");
        data.Set("policy.spatial.epsilon.first.phases", "bad");
        data.Set("policy.spatial.epsilon.first.phase.[bad].type", "linear");
        data.Set("policy.spatial.epsilon.first.phase.[bad].start", "0.1");
        data.Set("policy.spatial.epsilon.first.phase.[bad].steps", "2");
    }
    data.Set("policy." + key, value);
    if (key != "spatial.scale") {
        const auto standalone_key = key == "spatial.epsilon.last" ? "spatial.last" : "spatial.first";
        CHECK_THROWS_WITH(SpatialLadderSchedule(policy.spatial.epsilon, "log", 4, torch::kCPU),
            Catch::Matchers::ContainsSubstring(standalone_key) && Catch::Matchers::ContainsSubstring("value=" + value)
            && Catch::Matchers::ContainsSubstring(expected));
    }
    // 公開ConfigReaderを通し、単体構築とは別の設定読込境界でも同じ不正値を拒否する。
    Config owner("");
    ActionPolicyConfig parsed;
    CHECK_THROWS_WITH(ConfigReader<ActionPolicyConfig>::Read(owner, data, "policy", parsed),
        Catch::Matchers::ContainsSubstring("policy." + key) && Catch::Matchers::ContainsSubstring(value)
        && Catch::Matchers::ContainsSubstring(expected));
}
