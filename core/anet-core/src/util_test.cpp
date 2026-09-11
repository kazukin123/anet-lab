#include "anet/catch_test.hpp"

#include "anet/util.hpp"

#include <cmath>
#include <limits>

TEST_CASE("ScalarSampleAccumulator preserves unknown and unavailable semantics", "[util][scalar_aggregation]")
{
    const auto parsed = anet::ParseScalarAggregationKey("std.game_score");
    REQUIRE(parsed.has_value());
    CHECK(parsed->aggregation == anet::ScalarAggregation::STD);
    CHECK(parsed->base_key == "game_score");
    CHECK_FALSE(anet::ParseScalarAggregationKey("game_score").has_value());

    anet::ScalarSampleAccumulator values;
    values.Add(1.0f);
    values.Add(std::numeric_limits<float>::quiet_NaN());
    values.Add(3.0f);
    CHECK(values.Get(anet::ScalarAggregation::MEAN) == 2.0f);
    CHECK(values.Get(anet::ScalarAggregation::MAX) == 3.0f);
    CHECK(values.Get(anet::ScalarAggregation::MIN) == 1.0f);
    CHECK(values.Get(anet::ScalarAggregation::STD) == 1.0f);

    anet::ScalarSampleAccumulator unavailable;
    unavailable.Add(std::numeric_limits<float>::quiet_NaN());
    for (const auto aggregation : {
        anet::ScalarAggregation::MEAN,
        anet::ScalarAggregation::MAX,
        anet::ScalarAggregation::MIN,
        anet::ScalarAggregation::STD,
    }) {
        const auto value = unavailable.Get(aggregation);
        REQUIRE(value.has_value());
        CHECK(std::isnan(*value));
    }

    anet::ScalarSampleAccumulator single;
    single.Add(7.0f);
    REQUIRE(single.Get(anet::ScalarAggregation::STD).has_value());
    CHECK(std::isnan(*single.Get(anet::ScalarAggregation::STD)));

    anet::ScalarSampleAccumulator poisoned;
    poisoned.Add(1.0f);
    poisoned.Add(std::nullopt);
    for (const auto aggregation : {
        anet::ScalarAggregation::MEAN,
        anet::ScalarAggregation::MAX,
        anet::ScalarAggregation::MIN,
        anet::ScalarAggregation::STD,
    }) {
        CHECK_FALSE(poisoned.Get(aggregation).has_value());
    }
}

TEST_CASE("EmaFilter debiases initial observations", "[util][ema]")
{
    anet::EmaFilter<float> filter(0.25f);

    CHECK_FALSE(filter.IsInitialized());

    filter.Update(10.0f);

    CHECK(filter.IsInitialized());
    CHECK(filter.Value() == Catch::Approx(10.0f));
    CHECK(static_cast<float>(filter) == Catch::Approx(10.0f));

    filter.Update(2.0f);

    CHECK(filter.Value() == Catch::Approx(9.5f / 1.75f).margin(1.0e-6f));
}

TEST_CASE("EmaFilter preserves debiasing when decay changes", "[util][ema]")
{
    anet::EmaFilter<float> filter(0.5f);
    filter.Update(10.0f);

    filter.SetDecay(0.25f);
    filter.Update(2.0f);

    CHECK(filter.Value() == Catch::Approx(6.8f).margin(1.0e-6f));
}

TEST_CASE("EmaFilter converges to the conventional EMA", "[util][ema]")
{
    constexpr float kDecay = 0.05f;
    anet::EmaFilter<float> filter(kDecay);
    double conventional_ema = 0.0;

    for (int i = 0; i < 2000; ++i) {
        const float sample = static_cast<float>((i % 7) - 3);
        filter.Update(sample);

        if (i == 0) {
            conventional_ema = sample;
        } else {
            conventional_ema += kDecay * (sample - conventional_ema);
        }
    }

    CHECK(filter.Value() == Catch::Approx(conventional_ema).margin(1.0e-5));
}

TEST_CASE("EmaFilter explicit values skip debiasing warmup", "[util][ema]")
{
    SECTION("Set")
    {
        anet::EmaFilter<float> filter(0.5f);
        filter.Set(6.0f);

        CHECK(filter.IsInitialized());
        CHECK(filter.Value() == Catch::Approx(6.0f));

        filter.Update(2.0f);

        CHECK(filter.Value() == Catch::Approx(4.0f));
    }

    SECTION("value constructor")
    {
        anet::EmaFilter<float> filter(0.5f, 6.0f);

        CHECK(filter.IsInitialized());
        CHECK(filter.Value() == Catch::Approx(6.0f));

        filter.Update(2.0f);

        CHECK(filter.Value() == Catch::Approx(4.0f));
    }
}

TEST_CASE("EmaFilter Restart begins a new debiasing warmup", "[util][ema]")
{
    anet::EmaFilter<float> filter(0.5f);
    filter.Update(10.0f);
    filter.Update(2.0f);
    const float retained_value = filter.Value();

    filter.Restart();

    CHECK_FALSE(filter.IsInitialized());
    CHECK(filter.Value() == Catch::Approx(retained_value));

    filter.Update(4.0f);

    CHECK(filter.IsInitialized());
    CHECK(filter.Value() == Catch::Approx(4.0f));

    filter.Update(0.0f);

    CHECK(filter.Value() == Catch::Approx(4.0f / 3.0f).margin(1.0e-6f));
}

TEST_CASE("EmaFilter skips nonfinite observations without advancing weights", "[util][ema]")
{
    anet::EmaFilter<float> filter(0.25f);

    filter.Update(std::numeric_limits<float>::quiet_NaN());
    filter.Update(std::numeric_limits<float>::infinity());
    CHECK_FALSE(filter.IsInitialized());

    filter.Update(8.0f);
    const float first_value = filter.Value();

    filter.Update(std::numeric_limits<float>::quiet_NaN());
    filter.Update(std::numeric_limits<float>::infinity());
    filter.Update(-std::numeric_limits<float>::infinity());

    CHECK(filter.Value() == Catch::Approx(first_value));

    filter.Update(4.0f);

    CHECK(filter.Value() == Catch::Approx(10.0f / 1.75f).margin(1.0e-6f));
}

TEST_CASE("EmaFilter validates decay and keeps an integral default", "[util][ema]")
{
    const auto expected_range = Catch::Matchers::ContainsSubstring(
        "expected=finite value in (0, 1]");

    CHECK_THROWS_WITH(
        anet::EmaFilter<float>(0.0f),
        Catch::Matchers::ContainsSubstring("decay=0") && expected_range);
    CHECK_THROWS(anet::EmaFilter<float>(-0.1f));
    CHECK_THROWS(anet::EmaFilter<float>(1.1f));
    CHECK_THROWS(anet::EmaFilter<float>(std::numeric_limits<float>::quiet_NaN()));
    CHECK_THROWS(anet::EmaFilter<float>(std::numeric_limits<float>::infinity()));
    CHECK_THROWS(anet::EmaFilter<float>(0.0f, 1.0f));

    anet::EmaFilter<float> filter(0.5f);
    filter.Update(8.0f);
    CHECK_THROWS_WITH(filter.SetDecay(0.0f), expected_range);
    filter.Update(4.0f);
    CHECK(filter.Value() == Catch::Approx(16.0f / 3.0f).margin(1.0e-6f));

    anet::EmaFilter<int> integral_filter;
    integral_filter.Update(3);
    CHECK(integral_filter.Value() == 3);
    integral_filter.Update(7);
    CHECK(integral_filter.Value() == 7);
}

TEST_CASE("IntervalGate fires once per bucket for a coarse step stride", "[util][interval_gate]")
{
    // num_envs=512 相当の J=16 刻み。剰余判定では LCM(16,100)=400 ごとにしか発火しない。
    constexpr uint64_t kInterval = 100;
    constexpr uint64_t kStride = 16;
    anet::IntervalGate gate(kInterval);

    std::vector<uint64_t> fired_steps;
    for (uint64_t step = 0; step < 1000; step += kStride) {
        if (gate.ShouldFire(step)) fired_steps.push_back(step);
    }

    // 到達する step は 0〜992 なので、跨ぐバケットは 0〜9 の 10 個。各バケットで 1 回ずつ発火する。
    REQUIRE(fired_steps.size() == 10);
    for (size_t i = 0; i < fired_steps.size(); ++i) {
        CHECK(fired_steps[i] / kInterval == static_cast<uint64_t>(i));
    }
}

TEST_CASE("IntervalGate fires on the first call regardless of step", "[util][interval_gate]")
{
    SECTION("step=0")
    {
        anet::IntervalGate gate(100);
        CHECK(gate.ShouldFire(0));
    }

    SECTION("バケット途中の step から開始")
    {
        anet::IntervalGate gate(100);
        CHECK(gate.ShouldFire(1234));
        CHECK_FALSE(gate.ShouldFire(1250));
        CHECK(gate.ShouldFire(1300));
    }
}

TEST_CASE("IntervalGate fires once per bucket even when calls or strides vary", "[util][interval_gate]")
{
    SECTION("同一バケット内の複数呼び出しは 1 回だけ")
    {
        anet::IntervalGate gate(100);
        CHECK(gate.ShouldFire(0));
        CHECK_FALSE(gate.ShouldFire(1));
        CHECK_FALSE(gate.ShouldFire(50));
        CHECK_FALSE(gate.ShouldFire(99));
        CHECK(gate.ShouldFire(100));
    }

    SECTION("1 回の呼び出しで複数バケットを跨いでも 1 回だけ（catch-up しない）")
    {
        anet::IntervalGate gate(100);
        CHECK(gate.ShouldFire(0));
        CHECK(gate.ShouldFire(1000));
        CHECK_FALSE(gate.ShouldFire(1001));
        CHECK(gate.ShouldFire(1100));
    }

    SECTION("刻みが interval より大きい場合は毎回発火（毎 round 発火へ丸まる）")
    {
        anet::IntervalGate gate(10);
        for (uint64_t step = 0; step < 1000; step += 64) {
            CHECK(gate.ShouldFire(step));
        }
    }

    SECTION("非整数的な刻み（3,3,4,3,…）でも欠落しない")
    {
        // update_credit が float の構成（num_envs=100 相当）を模した刻み。
        constexpr uint64_t kInterval = 100;
        anet::IntervalGate gate(kInterval);

        uint64_t step = 0;
        uint64_t fire_count = 0;
        for (int i = 0; i < 400; ++i) {
            if (gate.ShouldFire(step)) ++fire_count;
            step += (i % 8 == 7) ? 4 : 3;   // 平均 3.125 刻み
        }

        // 最終 step までに跨いだバケット数と発火数が一致する（欠落なし）。
        const uint64_t last_step = step - ((399 % 8 == 7) ? 4 : 3);
        CHECK(fire_count == last_step / kInterval + 1);
    }

    SECTION("step が減少しても発火せず基準バケットも動かさない")
    {
        anet::IntervalGate gate(100);
        CHECK(gate.ShouldFire(500));
        CHECK_FALSE(gate.ShouldFire(100));
        CHECK_FALSE(gate.ShouldFire(500));
        CHECK(gate.ShouldFire(600));
    }
}

TEST_CASE("IntervalGate validates interval and restarts after Reset", "[util][interval_gate]")
{
    CHECK_THROWS_WITH(
        anet::IntervalGate(0),
        Catch::Matchers::ContainsSubstring("interval=0")
        && Catch::Matchers::ContainsSubstring("expected=1 or greater"));

    anet::IntervalGate gate(100);
    CHECK(gate.ShouldFire(0));
    CHECK_FALSE(gate.ShouldFire(50));

    gate.Reset();

    CHECK(gate.ShouldFire(50));
    CHECK_FALSE(gate.ShouldFire(99));
    CHECK(gate.ShouldFire(100));
}

TEST_CASE("EmaFilter time-weighted mode averages by elapsed time", "[util][ema]")
{
    constexpr float kTau = 10.0f;

    SECTION("一定値は dt 列によらずその値のまま")
    {
        auto filter = anet::EmaFilter<float>::TimeWeighted(kTau);

        CHECK_FALSE(filter.IsInitialized());

        filter.Update(7.0f, 0.2f);
        CHECK(filter.IsInitialized());
        CHECK(filter.Value() == Catch::Approx(7.0f));

        filter.Update(7.0f, 3.6f);
        filter.Update(7.0f, 0.001f);
        CHECK(filter.Value() == Catch::Approx(7.0f));
    }

    SECTION("可変 dt では経過時間で重み付けされる")
    {
        auto filter = anet::EmaFilter<float>::TimeWeighted(kTau);

        constexpr float kDt1 = 0.2f;
        constexpr float kDt2 = 3.6f;
        const float alpha1 = 1.0f - std::exp(-kDt1 / kTau);
        const float alpha2 = 1.0f - std::exp(-kDt2 / kTau);

        filter.Update(2000.0f, kDt1);
        filter.Update(0.0f, kDt2);

        // 重み付き平均: w1 = alpha1 * (1 - alpha2), w2 = alpha2
        const float w1 = alpha1 * (1.0f - alpha2);
        const float expected = (w1 * 2000.0f + alpha2 * 0.0f) / (w1 + alpha2);

        CHECK(filter.Value() == Catch::Approx(expected).margin(1.0e-3f));

        // 長い stall 側が支配的になるため、サンプル平均 1000 より大きく下振れする。
        CHECK(filter.Value() < 500.0f);
    }

    SECTION("tau に対して十分大きい dt では最新値へ収束する")
    {
        auto filter = anet::EmaFilter<float>::TimeWeighted(kTau);

        filter.Update(100.0f, 1.0f);
        filter.Update(4.0f, kTau * 100.0f);

        CHECK(filter.Value() == Catch::Approx(4.0f).margin(1.0e-4f));
    }

    SECTION("極小 dt でも alpha が下限クランプされ例外にならない")
    {
        auto filter = anet::EmaFilter<float>::TimeWeighted(kTau);

        CHECK_NOTHROW(filter.Update(5.0f, 1.0e-9f));
        CHECK(filter.IsInitialized());
        CHECK(filter.Value() == Catch::Approx(5.0f));

        CHECK_NOTHROW(filter.Update(5.0f, 1.0e-9f));
        CHECK(filter.Value() == Catch::Approx(5.0f));
    }

    SECTION("Set / Restart の意味は両モードで変わらない")
    {
        auto filter = anet::EmaFilter<float>::TimeWeighted(kTau);

        filter.Set(6.0f);
        CHECK(filter.IsInitialized());
        CHECK(filter.Value() == Catch::Approx(6.0f));

        filter.Restart();
        CHECK_FALSE(filter.IsInitialized());

        filter.Update(3.0f, 0.2f);
        CHECK(filter.Value() == Catch::Approx(3.0f));
    }
}

TEST_CASE("EmaFilter enforces the mode contract", "[util][ema]")
{
    // 非浮動小数点型での TimeWeighted() は static_assert のため実行時テストは書けない。

    SECTION("tau は正の有限値のみ")
    {
        const auto expected_range = Catch::Matchers::ContainsSubstring(
            "expected=finite positive value");

        CHECK_THROWS_WITH(
            anet::EmaFilter<float>::TimeWeighted(0.0f),
            Catch::Matchers::ContainsSubstring("tau_sec=0") && expected_range);
        CHECK_THROWS(anet::EmaFilter<float>::TimeWeighted(-1.0f));
        CHECK_THROWS(anet::EmaFilter<float>::TimeWeighted(
            std::numeric_limits<float>::quiet_NaN()));
        CHECK_THROWS(anet::EmaFilter<float>::TimeWeighted(
            std::numeric_limits<float>::infinity()));
    }

    SECTION("dt は正の有限値のみ")
    {
        auto filter = anet::EmaFilter<float>::TimeWeighted(10.0f);
        const auto expected_range = Catch::Matchers::ContainsSubstring(
            "expected=finite positive value");

        CHECK_THROWS_WITH(
            filter.Update(1.0f, 0.0f),
            Catch::Matchers::ContainsSubstring("dt_sec=0") && expected_range);
        CHECK_THROWS(filter.Update(1.0f, -0.1f));
        CHECK_THROWS(filter.Update(1.0f, std::numeric_limits<float>::quiet_NaN()));
        CHECK_THROWS(filter.Update(1.0f, std::numeric_limits<float>::infinity()));
    }

    SECTION("時定数モードでは Update(x) 単体と SetDecay() を弾く")
    {
        auto filter = anet::EmaFilter<float>::TimeWeighted(10.0f);

        CHECK_THROWS_WITH(
            filter.Update(1.0f),
            Catch::Matchers::ContainsSubstring("Update(x)")
            && Catch::Matchers::ContainsSubstring("time-weighted mode"));
        CHECK_THROWS_WITH(
            filter.SetDecay(0.5f),
            Catch::Matchers::ContainsSubstring("SetDecay()")
            && Catch::Matchers::ContainsSubstring("time-weighted mode"));
    }

    SECTION("サンプル重みモードでは Update(x, dt) を弾く")
    {
        anet::EmaFilter<float> filter(0.25f);

        CHECK_THROWS_WITH(
            filter.Update(1.0f, 0.2f),
            Catch::Matchers::ContainsSubstring("requires the time-weighted mode"));
    }
}
