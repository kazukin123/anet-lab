#include "anet/catch_test.hpp"

#include "anet/image.hpp"
#include "anet/metrics_logger.hpp"

#include <vector>

namespace {

class ImageTestMetricsBackend final : public anet::IBackend {
public:
    void Open(const std::filesystem::path&, const std::string&) override {}
    void WriteJsonl(const anet::json&) override {}
    void Flush() override {}
};

class ScopedImageMetricsLogger final {
public:
    ScopedImageMetricsLogger()
    {
        anet::MetricsLogger::Reset();
        anet::MetricsLoggerConfig config;
        config.run_name_tmpl = "image_registry_test";
        anet::MetricsLogger::Init(std::make_unique<ImageTestMetricsBackend>(), config, "out/test-tmp");
    }
    ~ScopedImageMetricsLogger() { anet::MetricsLogger::Reset(); }
};

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

TEST_CASE("ImageProviderManager rejects duplicate tags without replacing the provider", "[prd117][image][image_registry]")
{
    ScopedImageMetricsLogger metrics_logger;
    anet::rl::ImageProviderManager manager(anet::rl::EnvSpec{}, nullptr);
    anet::rl::ImageProviderConfig config;
    config.type = anet::rl::kImageType_Conv2d;
    manager.CreateAndRegister("layers", config, nullptr, nullptr);
    const auto first = manager.List();
    REQUIRE(first.size() == 1);
    CHECK_THROWS_WITH(manager.CreateAndRegister("layers", config, nullptr, nullptr),
        Catch::Matchers::ContainsSubstring("layers") && Catch::Matchers::ContainsSubstring("already exists"));
    const auto after = manager.List();
    REQUIRE(after.size() == 1);
    CHECK(after[0].provider == first[0].provider);
    manager.CreateAndRegister("other_layers", config, nullptr, nullptr);
    CHECK(manager.List().size() == 2);
}

TEST_CASE("Image catalog rejects unknown fields even after another reader consumed them", "[prd117][image][image_catalog]")
{
    ScopedImageMetricsLogger metrics_logger;
    anet::rl::ImageProviderManager manager(anet::rl::EnvSpec{}, nullptr);
    anet::ConfigData data;
    data.Set("metrics.image.[layers].type", anet::rl::kImageType_Conv2d);
    const std::string key = "metrics.image.[layers].conv2d.mergin_x";
    data.Set(key, 9);
    (void)data.Get(key);
    CHECK_THROWS_WITH(manager.CreateAndRegister(data, nullptr, nullptr),
        Catch::Matchers::ContainsSubstring("metrics.image.[layers]")
        && Catch::Matchers::ContainsSubstring("conv2d.mergin_x")
        && Catch::Matchers::ContainsSubstring("9"));
}

TEST_CASE("Image catalog reads known fields outside the selected image type", "[prd117][image][image_catalog]")
{
    ScopedImageMetricsLogger metrics_logger;
    anet::rl::ImageProviderManager manager(anet::rl::EnvSpec{}, nullptr);
    anet::ConfigData data;
    data.Set("metrics.image.[layers].type", anet::rl::kImageType_Conv2d);
    const std::string key = "metrics.image.[layers].heatmap.width";
    data.Set(key, "invalid");
    CHECK_THROWS_WITH(manager.CreateAndRegister(data, nullptr, nullptr),
        Catch::Matchers::ContainsSubstring(key) && Catch::Matchers::ContainsSubstring("expected"));
}

TEST_CASE("Image defaults are read without creating image providers", "[prd117][image][image_catalog]")
{
    ScopedImageMetricsLogger metrics_logger;
    anet::rl::ImageProviderManager manager(anet::rl::EnvSpec{}, nullptr);
    anet::ConfigData data;
    data.Set("metrics.image.histgram.frame_interval", 10);
    data.Set("metrics.image.conv2d.margin_x", 4);
    data.Set("some.setting", 9);
    manager.CreateAndRegister(data, nullptr, nullptr);
    CHECK(manager.List().empty());
    const auto unread = data.GetUnreadKeys();
    REQUIRE(unread.size() == 1);
    CHECK(unread[0].first == "some.setting");
}

TEST_CASE("Image catalog merges typed defaults and retains original read records", "[prd117][image][image_catalog]")
{
    ScopedImageMetricsLogger metrics_logger;
    anet::rl::ImageProviderManager manager(anet::rl::EnvSpec{}, nullptr);
    anet::ConfigData data;
    data.Set("metrics.image.interval", 77);
    data.Set("metrics.image.histgram.frame_interval", 10);
    data.Set("metrics.image.[layers].type", anet::rl::kImageType_Conv2d);
    data.Set("metrics.image.[layers].interval", 5);
    data.Set("metrics.image.[layers].heatmap.width", 512);
    data.Set("metrics.image.[layers].histgram.alpha", 0.5f);
    data.Set("some.setting", 9);
    manager.CreateAndRegister(data, nullptr, nullptr);
    const auto providers = manager.List();
    REQUIRE(providers.size() == 1);
    CHECK(providers[0].config.interval == 5);
    CHECK(providers[0].config.heatmap.width == 512);
    CHECK(providers[0].config.histgram.frame_interval == 10);
    CHECK(providers[0].config.histgram.alpha == 0.5f);
    const auto unread = data.GetUnreadKeys();
    REQUIRE(unread.size() == 1);
    CHECK(unread[0].first == "some.setting");
}
