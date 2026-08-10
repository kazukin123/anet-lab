#include "anet/catch_test.hpp"

#include "anet/util.hpp"

#include <limits>

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
