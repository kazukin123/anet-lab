#include "anet/catch_test.hpp"

#include "anet/image.hpp"

#include <vector>

namespace {

anet::TensorDict MakeConv2dLayoutTrace()
{
    auto first = torch::zeros({ 1, 4, 2, 3 }, torch::kFloat32);
    auto second = torch::ones({ 1, 4, 2, 3 }, torch::kFloat32);
    return anet::TensorDict{
        { "layer0", first },
        { "layer1", second },
    };
}

anet::rl::Conv2dVisualizerConfig MakeConv2dLayoutConfig(int layer_margin_y)
{
    anet::rl::Conv2dVisualizerConfig config;
    config.channels_per_row = 2;
    config.margin_x = 0;
    config.margin_y = 5;
    config.layer_margin_y = layer_margin_y;
    config.min_block_size = 1;
    config.scale_factor = 1;
    config.colormap = "gray";
    return config;
}

int JsonInt(const anet::json& obj, const char* key)
{
    return obj.at(key).get<int>();
}

void RequireRgb(const wxImage& image, int x, int y, unsigned char r, unsigned char g, unsigned char b)
{
    REQUIRE(image.IsOk());
    const unsigned char* data = image.GetData();
    REQUIRE(data != nullptr);
    const int idx = (y * image.GetWidth() + x) * 3;
    CHECK(data[idx] == r);
    CHECK(data[idx + 1] == g);
    CHECK(data[idx + 2] == b);
}

} // namespace

TEST_CASE("Conv2dVisualizer separates channel row margin from layer margin", "[image][conv2d]")
{
    const auto trace = MakeConv2dLayoutTrace();

    anet::rl::Conv2dVisualizer compat_visualizer(MakeConv2dLayoutConfig(/*layer_margin_y=*/-1));
    auto compat_result = compat_visualizer.Visualize(/*step=*/1, trace);
    const auto& compat_layout = compat_result.second;
    REQUIRE(compat_layout.at("layers").size() == 2);
    REQUIRE(JsonInt(compat_layout.at("layers").at(0), "offset_y") == 0);
    REQUIRE(JsonInt(compat_layout.at("layers").at(1), "offset_y") == 14);
    REQUIRE(JsonInt(compat_layout.at("layers").at(0), "layer_margin_y") == 5);
    REQUIRE(JsonInt(compat_layout, "image_height") == 28);

    anet::rl::Conv2dVisualizer separated_visualizer(MakeConv2dLayoutConfig(/*layer_margin_y=*/17));
    auto separated_result = separated_visualizer.Visualize(/*step=*/1, trace);
    const auto& separated_layout = separated_result.second;
    REQUIRE(separated_layout.at("layers").size() == 2);
    REQUIRE(JsonInt(separated_layout.at("layers").at(0), "offset_y") == 0);
    REQUIRE(JsonInt(separated_layout.at("layers").at(1), "offset_y") == 26);
    REQUIRE(JsonInt(separated_layout.at("layers").at(0), "margin_y") == 5);
    REQUIRE(JsonInt(separated_layout.at("layers").at(0), "layer_margin_y") == 17);
    REQUIRE(JsonInt(separated_layout, "image_width") == 6);
    REQUIRE(JsonInt(separated_layout, "image_height") == 52);
    RequireRgb(separated_result.first, /*x=*/0, /*y=*/9, /*r=*/96, /*g=*/96, /*b=*/96);
    RequireRgb(separated_result.first, /*x=*/5, /*y=*/10, /*r=*/56, /*g=*/56, /*b=*/56);
    RequireRgb(separated_result.first, /*x=*/5, /*y=*/16, /*r=*/180, /*g=*/180, /*b=*/180);
    RequireRgb(separated_result.first, /*x=*/5, /*y=*/17, /*r=*/180, /*g=*/180, /*b=*/180);
    RequireRgb(separated_result.first, /*x=*/5, /*y=*/18, /*r=*/180, /*g=*/180, /*b=*/180);
    RequireRgb(separated_result.first, /*x=*/5, /*y=*/25, /*r=*/96, /*g=*/96, /*b=*/96);
}

TEST_CASE("Conv2dVisualizer rejects invalid layer margin", "[image][conv2d]")
{
    anet::rl::Conv2dVisualizerConfig config = MakeConv2dLayoutConfig(/*layer_margin_y=*/-2);
    anet::rl::Conv2dVisualizer visualizer(config);
    CHECK_THROWS(visualizer.Visualize(/*step=*/1, MakeConv2dLayoutTrace()));
}
