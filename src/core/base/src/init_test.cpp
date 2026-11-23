#include "anet/test/catch.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "anet/tensor_util.hpp"
#include "anet/init.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/agent.hpp"
#include "anet/gui.hpp"
#include "anet/test/util.hpp"
#include "nn_impl.hpp"

constexpr char kDuplicateAgentClassId[] = "DuplicateRegistrationTestAgent";
constexpr char kDuplicateViewClassId[] = "DuplicateRegistrationTestView";
constexpr char kFlattenModuleType[] = "Flatten";

namespace {

    class DuplicateRegistrationAgentFactory final : public anet::rl::AgentFactory {
    public:
        std::shared_ptr<anet::rl::Agent> CreateAgent(
            const anet::rl::EnvSpec&, const anet::rl::BatchEnvSpec&, const torch::Device&,
            const anet::ConfigData&, std::shared_ptr<anet::rl::Notifier>, std::optional<anet::seed_t>) const override
        {
            return nullptr;
        }
        std::string GetTargetAgentClassId() const override { return kDuplicateAgentClassId; }
    };

    class DuplicateRegistrationViewCreator final : public anet::rl::gui::ViewCreator {
    public:
        std::shared_ptr<anet::rl::gui::View> CreateView(
            wxWindow*, const anet::ConfigData&, std::shared_ptr<anet::rl::Notifier>) const override
        {
            return nullptr;
        }
        std::string GetTargetClassId() const override { return kDuplicateViewClassId; }
    };

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

TEST_CASE("AgentRepository rejects duplicate names and retains the first factory", "[repository][duplicate]")
{
    auto& repository = anet::rl::AgentRepository::Instance();
    const auto first = std::make_shared<DuplicateRegistrationAgentFactory>();
    repository.Register(first);
    CHECK_THROWS_WITH(repository.Register(std::make_shared<DuplicateRegistrationAgentFactory>()),
        Catch::Matchers::ContainsSubstring("already registered")
        && Catch::Matchers::ContainsSubstring(kDuplicateAgentClassId));
    CHECK(repository.GetAgentFactory(kDuplicateAgentClassId) == first);
}

TEST_CASE("ViewRepository rejects duplicate names and retains the first creator", "[repository][duplicate]")
{
    auto& repository = anet::rl::gui::ViewRepository::Instance();
    const auto first = std::make_shared<DuplicateRegistrationViewCreator>();
    repository.Register(first);
    CHECK_THROWS_WITH(repository.Register(std::make_shared<DuplicateRegistrationViewCreator>()),
        Catch::Matchers::ContainsSubstring("already registered")
        && Catch::Matchers::ContainsSubstring(kDuplicateViewClassId));
    CHECK(repository.GetViewCreator(kDuplicateViewClassId) == first);
}

TEST_CASE("NetworkModuleRepository rejects duplicate names and retains the first factory", "[repository][duplicate]")
{
    constexpr char kDuplicateModuleType[] = "DuplicateRegistrationTestModule";
    anet::nn::InitNN();
    auto& repository = anet::nn::NetworkModuleRepository::Instance();
    const auto first = repository.GetFactory(kFlattenModuleType);
    repository.Register(kDuplicateModuleType, first);
    CHECK_THROWS_WITH(repository.Register(kDuplicateModuleType, repository.GetFactory("ReLU")),
        Catch::Matchers::ContainsSubstring("already registered")
        && Catch::Matchers::ContainsSubstring(kDuplicateModuleType));
    CHECK(repository.GetFactory(kDuplicateModuleType) == first);
}

TEST_CASE("InitNN preserves its first factory registrations when called again", "[init][idempotent]")
{
    anet::nn::InitNN();
    auto& repository = anet::nn::NetworkModuleRepository::Instance();
    const auto factory = repository.GetFactory(kFlattenModuleType);
    REQUIRE(factory);
    CHECK_NOTHROW(anet::nn::InitNN());
    CHECK(repository.GetFactory(kFlattenModuleType) == factory);
}

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

TEST_CASE("RFD0070 parses device specifications and preserves current CUDA index", "[device][rfd0070]")
{
    // 値の文法とautoの選択を公開パーサ経由で確認する。
    constexpr auto* key = "agent.device";
    CHECK(anet::ParseDevice(" CPU ", key).str() == "cpu");
    CHECK(anet::ParseDevice("CuDa", key).is_cuda());
    CHECK(anet::ParseDevice("CuDa", key).index() == -1);
    CHECK(anet::ParseDevice(" CUDA:1 ", key).index() == 1);
    CHECK(anet::ParseDevice("auto", key).is_cuda() == torch::cuda::is_available());

    for (const auto* invalid : {"", "cpu:0", "cuda:", "cuda:-1", "cuda:+1", "cuda:x", "mps"}) {
        INFO(invalid);
        CHECK_THROWS(anet::ParseDevice(invalid, key));
    }
    CHECK_THROWS(anet::ParseDevice("cpu", ""));
}

TEST_CASE("InitRL reports the original value of every explicit CUDA key", "[device][rfd0070][rfd0194][rfd0194_cuda]")
{
    if (torch::cuda::is_available()) SKIP("CUDA is available");

    CapturingBackend* backend_raw = nullptr;
    InitCapturingLogger(backend_raw, "explicit_cuda_unavailable_test");
    anet::ConfigData config_data;
    config_data.Set("agent.device", "cuda:1");
    config_data.Set("env.device", "cuda:2");
    config_data.Set("run.eval_device", "cuda");
    try {
        anet::rl::InitRL(anet::rl::BackendConfig(config_data), config_data);
        FAIL("Explicit CUDA must fail before NN initialization");
    } catch (const std::exception& e) {
        const std::string message = e.what();
        CHECK(ContainsText(message, "agent.device='cuda:1', env.device='cuda:2', run.eval_device='cuda'"));
        CHECK(ContainsText(message, "auto"));
    }
    CHECK(config_data.Get("agent.device") == "cuda:1");
    CHECK(config_data.Get("env.device") == "cuda:2");
    anet::MetricsLogger::Reset();
}

TEST_CASE("InitRL applies cuda_launch_blocking off and records the effective environment", "[backend][env]")
{
    EnvVarGuard guard(kCudaLaunchBlockingEnv);
    REQUIRE(SetEnvValue(kCudaLaunchBlockingEnv, std::string("1")) == 0);

    CapturingBackend* backend_raw = nullptr;
    InitCapturingLogger(backend_raw, "cuda_launch_blocking_off_test");

    anet::ConfigData config_data;
    config_data.Set("backend.cuda_launch_blocking", "off");
    anet::test::LogCaptureGuard logs;
    anet::rl::InitRL(anet::rl::BackendConfig(config_data), config_data);
    // 1回目の上書き後に同じ不一致を作り直し、通知だけが重複しないことを確認する。
    REQUIRE(SetEnvValue(kCudaLaunchBlockingEnv, std::string("1")) == 0);
    anet::rl::InitRL(anet::rl::BackendConfig(config_data), config_data);
    logs.Flush();
    CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message,
        {"CUDA_LAUNCH_BLOCKING overridden", "previous=1", "requested=off", "effective=0"}));
    CHECK(std::count_if(logs.Records().begin(), logs.Records().end(), [](const auto& record) {
        return record.level == wxLOG_Message && record.message.find("CUDA_LAUNCH_BLOCKING overridden") != std::string::npos;
    }) == 1);
    CHECK(anet::test::CountRecords(logs.Records(), wxLOG_Warning) == 0);

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

    anet::rl::InitRL(anet::rl::BackendConfig(), anet::EmptyConfigData);

    REQUIRE(GetEnvValue(kCudaLaunchBlockingEnv).has_value());
    CHECK(*GetEnvValue(kCudaLaunchBlockingEnv) == "1");

    anet::MetricsLogger::Reset();
}

TEST_CASE("InitRL rejects an invalid device when only that key is specified", "[rfd0122][rfd0070][device]")
{
    // 未指定の2キーを足さず、書かれた1キーの失敗が識別できることを確認する。
    const auto* key = GENERATE("agent.device", "env.device", "run.eval_device");
    CapturingBackend* backend = nullptr;
    InitCapturingLogger(backend, "rfd0122_explicit_device");
    anet::ConfigData data;
    data.Set(key, "invalid-device");
    CHECK_THROWS_WITH(anet::rl::InitRL(anet::rl::BackendConfig(), data),
        Catch::Matchers::ContainsSubstring(key));
    anet::MetricsLogger::Reset();
}

TEST_CASE("InitRL logs one fatal with the full key of an invalid device", "[device][rfd0194]")
{
    const auto* key = GENERATE("agent.device", "env.device", "run.eval_device");
    anet::ConfigData data;
    data.Set(key, "cuda:x");
    const auto expected = std::string(key) + "='cuda:x'";

    anet::test::LogCaptureGuard logs;
    CHECK_THROWS_WITH(anet::rl::InitRL(anet::rl::BackendConfig(data), data),
        Catch::Matchers::ContainsSubstring(expected));
    logs.Flush();
    // LOG::fatal は wxWidgets のプロセス終了を避けるため Message として届ける。
    CHECK(std::count_if(logs.Records().begin(), logs.Records().end(), [](const auto& record) {
        return record.level == wxLOG_Message && record.message.starts_with("Exception: ");
    }) == 1);
    CHECK(anet::test::HasRecordContaining(logs.Records(), wxLOG_Message, {"Exception: ", expected}));
}
