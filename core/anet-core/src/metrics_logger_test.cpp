#include "catch.hpp"

#include "anet/metrics_logger.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>

namespace {

class NullBackend final : public anet::IBackend {
public:
    void Open(const std::filesystem::path&, const std::string&) override {}
    void WriteJsonl(const anet::json&) override {}
    void Flush() override {}
};

std::string ReadTextFile(const std::filesystem::path& path)
{
    std::ifstream ifs(path);
    return std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
}

} // namespace

TEST_CASE("MetricsLogger writes ConfigData text file", "[metrics][config]")
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "anet-core-config-data-test";
    std::filesystem::remove_all(root);

    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<NullBackend>();
    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = "config_data_test";
    anet::MetricsLogger::Init(std::move(backend), logger_config, root);

    anet::ConfigData config_data;
    config_data.Set("app.run_name", "run_{t}");
    config_data.Set("train.num_envs", "8");
    config_data.Set("DefaultDQNAgent.batch_size", "128");
    anet::MetricsLogger::Instance()->Log("config_data", config_data);

    const auto config_path = root / "runs" / "config_data_test" / "config" / "config_data.txt";
    REQUIRE(std::filesystem::exists(config_path));
    CHECK(ReadTextFile(config_path) ==
        "app.run_name = run_{t}\n"
        "train.num_envs = 8\n"
        "DefaultDQNAgent.batch_size = 128\n");

    anet::MetricsLogger::Reset();
    std::filesystem::remove_all(root);
}
