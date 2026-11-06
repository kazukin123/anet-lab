#include "anet/catch_test.hpp"

#include <memory>
#include <string>
#include <string_view>
#include "anet/heat_map.hpp"

namespace {

constexpr int kImageWidth = 7;
constexpr int kImageHeight = 5;
constexpr const char* kTimeHeatMapSubtype = "timed_heat_map";
constexpr const char* kHistogramSubtype = "histgram";
constexpr const char* kSweepSubtype = "sweeped_map";
constexpr const char* kTimeHistogramSubtype = "time_histgram";

template<class Factory>
void RequireRetainedImage(Factory factory, int width, int height,
    const std::string& subtype, std::string_view expected)
{
    wxImage retained;
    {
        const auto source = factory();
        CHECK(source->GetImageSubType() == subtype);
        const auto image = source->Render();
        REQUIRE(image.IsOk());
        CHECK(image.GetWidth() == width);
        CHECK(image.GetHeight() == height);
        retained = image;
    }

    // 生成元と元画像の破棄後も、最後の画像所有者から同じ全画素を読める。
    REQUIRE(retained.IsOk());
    const auto* pixels = retained.GetData();
    REQUIRE(pixels != nullptr);
    REQUIRE(expected.size() == static_cast<size_t>(width * height * 6));
    for (int i = 0; i < width * height * 3; ++i) {
        CAPTURE(i);
        CHECK(static_cast<unsigned>(pixels[i]) ==
            std::stoul(std::string(expected.substr(i * 2, 2)), nullptr, 16));
    }
}

} // namespace

TEST_CASE("TimeHeatMap retains its empty black Scale image", "[image][heat_map][prd115]")
{
    RequireRetainedImage([] {
        return std::make_unique<anet::TimeHeatMap>(kImageWidth, kImageHeight, 0.0f, 5.0f,
            anet::HM_Default, 0, anet::TimeFrameMode::Scale);
    }, 1, 1, kTimeHeatMapSubtype, "000000");
}

TEST_CASE("TimeHeatMap preserves every Scale pixel and image ownership", "[image][heat_map][prd115]")
{
    // 修正前の独立probeで採取した固定入力のRGB列。
    constexpr std::string_view kExpected =
        "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a"
        "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a7f0000"
        "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a"
        "00007f00007f0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a"
        "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a";
    RequireRetainedImage([] {
        auto source = std::make_unique<anet::TimeHeatMap>(kImageWidth, kImageHeight, 0.0f, 5.0f,
            anet::HM_Default, 0, anet::TimeFrameMode::Scale);
        source->AddData(1.0f, 1.0f);
        source->NextFrame();
        source->AddData(3.0f, 2.0f);
        source->NextFrame();
        return source;
    }, kImageWidth, kImageHeight, kTimeHeatMapSubtype, kExpected);
}

TEST_CASE("Histgram preserves empty and populated pixels after its source is destroyed", "[image][heat_map][prd115]")
{
    SECTION("empty image is black")
    {
        RequireRetainedImage([] {
            return std::make_unique<anet::Histgram>(3, 0.0f, 3.0f, kImageWidth, kImageHeight);
        }, kImageWidth, kImageHeight, kHistogramSubtype, std::string(kImageWidth * kImageHeight * 6, '0'));
    }
    SECTION("populated image matches the baseline")
    {
        constexpr std::string_view kExpected =
            "000000000000000000000000ffffffffffffffffff"
            "000000000000000000000000ffffffffffffffffff"
            "000000000000000000000000ffffffffffffffffff"
            "ffffffffffff000000000000ffffffffffffffffff"
            "ffffffffffff000000000000ffffffffffffffffff";
        RequireRetainedImage([] {
            auto source = std::make_unique<anet::Histgram>(3, 0.0f, 3.0f, kImageWidth, kImageHeight);
            source->AddData(0.5f);
            source->AddData(2.5f);
            source->AddData(2.5f);
            return source;
        }, kImageWidth, kImageHeight, kHistogramSubtype, kExpected);
    }
}

TEST_CASE("SweepedHeatMap preserves reset and evaluated pixels", "[image][heat_map][prd115]")
{
    SECTION("reset image has the original minimum color")
    {
        std::string expected;
        for (int i = 0; i < kImageWidth * kImageHeight; ++i) expected += "00007f";
        RequireRetainedImage([] {
            auto source = std::make_unique<anet::SweepedHeatMap>(
                kImageWidth, kImageHeight, 0.0f, 1.0f, 0.0f, 1.0f);
            source->Reset();
            return source;
        }, kImageWidth, kImageHeight, kSweepSubtype, expected);
    }
    SECTION("evaluated image matches the baseline")
    {
        constexpr std::string_view kExpected =
            "7fff7fd4ff2affd400ff7f00ff2a00d400007f0000"
            "00ffff55ffa9aaff54ffff00ffa900ff5500ff0000"
            "007fff00d4ff2affd47fff7fd4ff2affd400ff7f00"
            "0000ff0055ff00aaff00ffff55ffa9a9ff55ffff00"
            "00007f0000d4002aff007fff00d4ff2affd47fff7f";
        RequireRetainedImage([] {
            auto source = std::make_unique<anet::SweepedHeatMap>(
                kImageWidth, kImageHeight, 0.0f, 1.0f, 0.0f, 1.0f);
            source->Evaluate([](float x, float y) { return x + y; });
            return source;
        }, kImageWidth, kImageHeight, kSweepSubtype, kExpected);
    }
}

TEST_CASE("TimeHistogram retains empty and populated Scale images", "[image][heat_map][prd115]")
{
    SECTION("empty Scale image is black")
    {
        RequireRetainedImage([] {
            return std::make_unique<anet::TimeHistogram>(
                3, 3, anet::TimeFrameMode::Scale, anet::HM_None, 0.0f, 3.0f);
        }, 1, 1, kTimeHistogramSubtype, "000000");
    }
    SECTION("fixed bins have independently calculated pixels")
    {
        // 値1は中間色、値2は上限色、隣の値0の補間画素は青、未観測セルは背景色10。
        constexpr std::string_view kExpected =
            "0a0a0a0a0a0a0a0a0a"
            "7fff7f00007f0a0a0a"
            "7f000000007f0a0a0a";
        RequireRetainedImage([] {
            auto source = std::make_unique<anet::TimeHistogram>(
                3, 3, anet::TimeFrameMode::Scale, anet::HM_None, 0.0f, 3.0f);
            source->AddBatch({ 1.5f, 2.5f, 2.5f });
            source->NextFrame();
            return source;
        }, 3, 3, kTimeHistogramSubtype, kExpected);
    }
}
