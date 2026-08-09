#include "anet/catch_test.hpp"

#include "anet/init.hpp"
#include "anet/metrics_logger.hpp"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr const char* kCudaLaunchBlockingEnv = "CUDA_LAUNCH_BLOCKING";

class CapturingBackend final : public anet::IBackend {
public:
    void Open(const std::filesystem::path&, const std::string&) override {}
    void WriteJsonl(const anet::json& obj) override { records.push_back(obj); }
    void Flush() override {}

    std::vector<anet::json> records;
};

std::optional<std::string> GetEnvValue(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr) return std::nullopt;
    return std::string(value);
}

int SetEnvValue(const char* name, const std::optional<std::string>& value)
{
#ifdef _WIN32
    return _putenv_s(name, value.has_value() ? value->c_str() : "");
#else
    return value.has_value() ? setenv(name, value->c_str(), 1) : unsetenv(name);
#endif
}

class EnvVarGuard {
public:
    explicit EnvVarGuard(const char* name)
        : name_(name)
        , old_value_(GetEnvValue(name))
    {
    }

    ~EnvVarGuard()
    {
        (void)SetEnvValue(name_, old_value_);
    }

    EnvVarGuard(const EnvVarGuard&) = delete;
    EnvVarGuard& operator=(const EnvVarGuard&) = delete;

private:
    const char* name_;
    std::optional<std::string> old_value_;
};

anet::rl::BackendConfig MakeBackendConfig(const std::string& value)
{
    anet::ConfigData config_data;
    config_data.Set("backend.cuda_launch_blocking", value);
    return anet::rl::BackendConfig(config_data);
}

void InitCapturingLogger(CapturingBackend*& backend_raw, const std::string& run_name)
{
    const auto root = std::filesystem::current_path() / "out" / "test-tmp" / "anet-core-init-test";
    std::filesystem::remove_all(root);

    anet::MetricsLogger::Reset();
    auto backend = std::make_unique<CapturingBackend>();
    backend_raw = backend.get();

    anet::MetricsLoggerConfig logger_config;
    logger_config.run_name_tmpl = run_name;
    anet::MetricsLogger::Init(std::move(backend), logger_config, root);
}

bool ContainsText(const std::string& text, const std::string& pattern)
{
    return text.find(pattern) != std::string::npos;
}

} // namespace

TEST_CASE("BackendConfig normalizes cuda_launch_blocking values", "[backend][config]")
{
    CHECK(anet::rl::BackendConfig().cuda_launch_blocking == "inherit");

    const std::vector<std::pair<std::string, std::string>> cases = {
        {"on", "on"},
        {"true", "on"},
        {"1", "on"},
        {"off", "off"},
        {"false", "off"},
        {"0", "off"},
        {"inherit", "inherit"},
        {" ON ", "on"},
        {" FALSE ", "off"},
    };

    for (const auto& [input, expected] : cases) {
        const auto config = MakeBackendConfig(input);
        CHECK(config.cuda_launch_blocking == expected);
        CHECK(config.ToJson().at("cuda_launch_blocking") == expected);
    }
}

TEST_CASE("BackendConfig rejects invalid cuda_launch_blocking values", "[backend][config]")
{
    try {
        (void)MakeBackendConfig("debug");
        FAIL("BackendConfig should reject invalid backend.cuda_launch_blocking");
    } catch (const std::exception& e) {
        const std::string message = e.what();
        CHECK(ContainsText(message, "backend.cuda_launch_blocking"));
        CHECK(ContainsText(message, "debug"));
        CHECK(ContainsText(message, "expected on, off, inherit"));
    }
}

TEST_CASE("InitRL applies cuda_launch_blocking off and records the effective environment", "[backend][env]")
{
    EnvVarGuard guard(kCudaLaunchBlockingEnv);
    REQUIRE(SetEnvValue(kCudaLaunchBlockingEnv, std::string("1")) == 0);

    CapturingBackend* backend_raw = nullptr;
    InitCapturingLogger(backend_raw, "cuda_launch_blocking_off_test");

    anet::ConfigData config_data;
    config_data.Set("backend.cuda_launch_blocking", "off");
    anet::rl::InitRL(anet::rl::BackendConfig(config_data));

    REQUIRE(GetEnvValue(kCudaLaunchBlockingEnv).has_value());
    CHECK(*GetEnvValue(kCudaLaunchBlockingEnv) == "0");

    anet::MetricsLogger::Reset();
}

TEST_CASE("InitRL leaves CUDA_LAUNCH_BLOCKING unchanged for inherit", "[backend][env]")
{
    EnvVarGuard guard(kCudaLaunchBlockingEnv);
    REQUIRE(SetEnvValue(kCudaLaunchBlockingEnv, std::string("1")) == 0);

    CapturingBackend* backend_raw = nullptr;
    InitCapturingLogger(backend_raw, "cuda_launch_blocking_inherit_test");

    anet::rl::InitRL(anet::rl::BackendConfig());

    REQUIRE(GetEnvValue(kCudaLaunchBlockingEnv).has_value());
    CHECK(*GetEnvValue(kCudaLaunchBlockingEnv) == "1");

    anet::MetricsLogger::Reset();
}
