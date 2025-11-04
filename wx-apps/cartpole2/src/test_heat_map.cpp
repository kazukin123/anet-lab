#include <gtest/gtest.h>
#include <wx/image.h>
#include <filesystem>
#include "anet/HeatMap.hpp"

using namespace anet;

// ============================================================
// テスト前初期化
// ============================================================
class HeatMapTest : public ::testing::Test {
protected:
    void SetUp() override {
        wxInitAllImageHandlers();
        std::filesystem::create_directories("test_output");
    }
};

// ============================================================
// HeatMap : 黒背景・描画テスト
// ============================================================
TEST_F(HeatMapTest, BasicHeatMap_BlackBackground) {
    HeatMap map(64, 64, -1.0f, 1.0f, -1.0f, 1.0f, 0, HM_AutoNormValue);
    // 散布：原点近くを集中加熱
    for (int i = 0; i < 1000; ++i) {
        float x = (float(rand()) / RAND_MAX) * 2.0f - 1.0f;
        float y = (float(rand()) / RAND_MAX) * 2.0f - 1.0f;
        float v = std::exp(-4.0f * (x * x + y * y));
        map.AddData(x, y, v);
    }
    wxImage img = map.RenderRaw();
    EXPECT_TRUE(img.IsOk());
    img.SaveFile("test_output/heatmap_black.png", wxBITMAP_TYPE_PNG);
}

// ============================================================
// TimeHeatMap : フレーム進行テスト
// ============================================================
TEST_F(HeatMapTest, TimeHeatMap_ScrollBehavior) {
    TimeHeatMap th(64, 32, -1.0f, 1.0f, HM_AutoNormValue | HM_ShowZeroLine, 0, TimeFrameMode::Scroll);

    for (int frame = 0; frame < 100; ++frame) {
        float val = std::sin(frame * 0.1f);
        for (int i = 0; i < 32; ++i) {
            float y = -1.0f + 2.0f * i / 31.0f;
            float heat = std::exp(-8.0f * std::pow(y - val, 2));
            th.AddData(y, heat);
        }
        th.NextFrame();
    }

    wxImage img = th.RenderRaw();
    EXPECT_TRUE(img.IsOk());
    img.SaveFile("test_output/time_heatmap_scroll.png", wxBITMAP_TYPE_PNG);
}

// ============================================================
// TimeHistogram : 対数軸＋自動スケール＋ゼロライン
// ============================================================
TEST_F(HeatMapTest, TimeHistogram_LogScale_ZeroLine) {
    const int bins = 64;
    const int frames = 100;
    TimeHistogram hist(bins, frames, TimeFrameMode::Scroll,
        HM_AutoScaleAxis | HM_LogScaleAxis | HM_ShowZeroLine,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        0.05f);

    for (int step = 0; step < 200; ++step) {
        std::vector<float> values;
        for (int i = 0; i < 512; ++i) {
            float sign = (rand() % 2 == 0) ? 1.0f : -1.0f;
            float v = sign * std::pow(float(rand()) / RAND_MAX, 2.0f) * 10.0f;
            values.push_back(v);
        }
        hist.AddBatch(values);
        hist.NextFrame();
    }

    wxImage img = hist.RenderRaw();
    EXPECT_TRUE(img.IsOk());
    img.SaveFile("test_output/time_histogram_log_zero.png", wxBITMAP_TYPE_PNG);
}
