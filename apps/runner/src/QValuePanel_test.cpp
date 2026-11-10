#include "anet/catch_test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include "QValuePanel.hpp"

// 入力8値の平均は-11/8、分散は527/64。Histogramの左端は平均-標準偏差となる。
const float kRawHistogramMinimum = static_cast<float>(-11.0 / 8.0 - std::sqrt(527.0 / 64.0));
const float kRawHistogramGuide = -kRawHistogramMinimum / (4.0f - kRawHistogramMinimum);

static QValueData UnitQValueData(const std::vector<float>& values)
{
    return QValueData{ .stats = {}, .selected_action = -1,
        .width = static_cast<int>(values.size() / 2), .height = 2, .vv = values };
}

static QValueHeatMapRequest UnitHeatMapRequest(bool hist, bool adv, bool log, bool auto_range,
    bool smooth, std::span<const float> weights = {})
{
    return QValueHeatMapRequest{ .histogram = hist, .advantage = adv, .log_scale = log,
        .auto_range = auto_range, .hist_bins = 5, .hist_range_k = 1.0f,
        .use_hist_smooth = smooth, .smooth_radius = 1.0f, .hist_weights = weights };
}

static void RequireHeatMapResult(const QValueHeatMapResult& actual, int width,
    const std::vector<float>& expected, float guide)
{
    CHECK(actual.width == width);
    CHECK(actual.height == 2);
    CHECK(actual.guide_line_ratio == Catch::Approx(guide).margin(1e-6));
    REQUIRE(actual.vv.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        INFO(i);
        CHECK(actual.vv[i] == Catch::Approx(expected[i]).margin(1e-6));
    }
}

TEST_CASE("Q heat map raw and advantage processing is independent of windows", "[prd122][qvalue_heatmap]")
{
    // wxのwindowを作らず、共有snapshotを変更しない配列の加工だけを呼ぶ。
    const bool advantage = GENERATE(false, true);
    const std::vector<float> values{-4.0f, -4.0f, -4.0f, -4.0f, 0.0f, 0.0f, 1.0f, 4.0f};
    const auto input = UnitQValueData(values);
    QValueHeatMapRangeState range;
    const auto actual = ComputeQValueHeatMap(input, UnitHeatMapRequest(false, advantage, false, false, false), range);
    const std::vector<float> expected = advantage
        ? std::vector<float>{11.0f/43, 11.0f/43, 11.0f/43, 11.0f/43, 27.0f/43, 27.0f/43, 31.0f/43, 1.0f}
        : std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.625f, 1.0f};
    RequireHeatMapResult(actual, 4, expected, -1.0f);
    CHECK(input.vv == values);
}

TEST_CASE("Q heat map histogram mode matrix matches hand counted bins", "[prd122][qvalue_heatmap]")
{
    const int mode = GENERATE(0, 1, 2, 3, 4, 5, 6, 7);
    const bool smooth = GENERATE(false, true);
    const bool advantage = (mode & 1) != 0;
    const bool log = (mode & 2) != 0;
    const bool auto_range = (mode & 4) != 0;
    CAPTURE(mode, smooth);
    const float w = std::exp(-2.0f);
    const std::array<float, 3> weights{{w, 1.0f, w}};
    const auto input = UnitQValueData({-4.0f, -4.0f, -4.0f, -4.0f, 0.0f, 0.0f, 1.0f, 4.0f});
    QValueHeatMapRangeState range;
    const auto actual = ComputeQValueHeatMap(input,
        UnitHeatMapRequest(true, advantage, log, auto_range, smooth, weights), range);

    // 各行のbin件数と中央の最大値は手で数えた値。端の4件を色の基準へ混ぜない。
    std::vector<float> expected;
    float peak;
    if (advantage) {
        expected = smooth ? std::vector<float>{4*w, 4, 4*w, 0, 0, 0, 0, 3*w, 3+w, 1+3*w}
            : std::vector<float>{0, 4, 0, 0, 0, 0, 0, 0, 3, 1};
        peak = 4.0f;
    } else {
        expected = smooth ? std::vector<float>{4, 4*w, 0, 0, 0, 0, 2*w, 2+w, 1+3*w, 1+w}
            : std::vector<float>{4, 0, 0, 0, 0, 0, 0, 2, 1, 1};
        peak = smooth ? 2.0f + w : 2.0f;
    }
    for (auto& value : expected) {
        value = std::clamp(log ? std::log1p(value) / std::log1p(peak) : value / peak, 0.0f, 1.0f);
    }
    RequireHeatMapResult(actual, 5, expected, advantage ? 0.5f : kRawHistogramGuide);
    CHECK(range.minimum == Catch::Approx(advantage ? -5.375f : kRawHistogramMinimum).margin(1e-6));
    CHECK(range.maximum == Catch::Approx(advantage ? 5.375f : 4.0f).margin(1e-6));
}

TEST_CASE("Q heat map handles constant values in raw and histogram modes", "[prd122][qvalue_heatmap]")
{
    const bool histogram = GENERATE(false, true);
    const bool advantage = GENERATE(false, true);
    const auto input = UnitQValueData({2.0f, 2.0f, 2.0f, 2.0f});
    QValueHeatMapRangeState range;
    const auto actual = ComputeQValueHeatMap(input,
        UnitHeatMapRequest(histogram, advantage, false, false, false), range);
    const std::vector<float> expected = histogram ? std::vector<float>{1,0,0,0,0,1,0,0,0,0}
        : std::vector<float>{0.5f,0.5f,0.5f,0.5f};
    RequireHeatMapResult(actual, histogram ? 5 : 2, expected, histogram && advantage ? 0.0f : -1.0f);
}

TEST_CASE("Q heat map accumulated range grows immediately and shrinks only in auto", "[prd122][qvalue_heatmap]")
{
    const bool auto_range = GENERATE(false, true);
    const std::vector<float> values{-4.0f, -4.0f, -4.0f, -4.0f, 0.0f, 0.0f, 1.0f, 4.0f};
    const float factors[] = {1.0f, 0.5f, 2.0f};
    QValueHeatMapRangeState range;
    for (int step = 0; step < 3; ++step) {
        CAPTURE(auto_range, step);
        auto input_values = values;
        for (auto& value : input_values) value *= factors[step];
        const auto actual = ComputeQValueHeatMap(UnitQValueData(input_values),
            UnitHeatMapRequest(true, false, false, auto_range, false), range);
        const std::vector<float> expected = step == 1 ? std::vector<float>{0,1,0,0,0,0,0,0.75f,0.25f,0}
            : std::vector<float>{1,0,0,0,0,0,0,1,0.5f,0.5f};
        RequireHeatMapResult(actual, 5, expected, kRawHistogramGuide);
        const float factor = step == 2 ? 2.0f : step == 1 && auto_range ? 0.975f : 1.0f;
        CHECK(range.minimum == Catch::Approx(kRawHistogramMinimum * factor).margin(1e-6));
        CHECK(range.maximum == Catch::Approx(4.0f * factor).margin(1e-6));
    }
}
