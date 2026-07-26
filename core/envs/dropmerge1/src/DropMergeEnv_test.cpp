// DropMergeEnv_test.cpp

#include "DropMergeEnv.hpp"

#include <cmath>
#include <clocale>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>

#include "anet/catch_test.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/test_util.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

void SetupUtf8Console()
{
#ifdef _WIN32
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::setlocale(LC_CTYPE, ".UTF-8");
}

class NoopMetricsBackend final : public anet::IBackend {
public:
    void Open(const std::filesystem::path&, const std::string&) override {}
    void WriteJsonl(const anet::json&) override {}
    void Flush() override {}
};

class ScopedNoopMetricsLogger final {
public:
    ScopedNoopMetricsLogger()
    {
        anet::MetricsLogger::Reset();
        anet::MetricsLoggerConfig config;
        config.run_name_tmpl = "dropmerge_env_test";
        anet::MetricsLogger::Init(std::make_unique<NoopMetricsBackend>(), config, "C:/tmp");
    }

    ~ScopedNoopMetricsLogger()
    {
        anet::MetricsLogger::Reset();
    }
};

anet::rl::env::drop_merge::DropMergeEnvConfig MakeBlockedBoardConfig()
{
    anet::rl::env::drop_merge::DropMergeEnvConfig config;
    config.seed_mode = "fixed";
    config.action_mode = "direct_noop";
    config.drop_divisions = 1;
    config.box_width = 1.0f;
    config.box_height = 0.9f;
    config.fruit_radii.assign(anet::rl::env::drop_merge::kFruitTypeCount, 0.4f);
    config.drop_probs = { 1.0f };
    config.drop_noise = 0.0f;
    config.spin_noise = 0.0f;
    config.restitution = 0.0f;
    config.box_restitution = 0.0f;
    config.use_instant_drop = true;
    config.game_over_grace_step = 1000;
    return config;
}

int CountOccurrences(const std::string& value, const std::string& needle)
{
    int count = 0;
    std::string::size_type position = 0;
    while ((position = value.find(needle, position)) != std::string::npos) {
        ++count;
        position += needle.size();
    }
    return count;
}

} // namespace

TEST_CASE("DropMergeEnv prefixes maximum-step log with its name once", "[dropmerge][env_name]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::rl::env::drop_merge::DropMergeEnvConfig config;
    config.max_step = 1;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-log[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    env->Step(anet::rl::env::drop_merge::kActionNoop);
    logs.Flush();

    int matching_records = 0;
    for (const auto& record : logs.Records()) {
        if (record.message.find("Episode truncated. Maximum step count exceeded.") == std::string::npos) {
            continue;
        }
        ++matching_records;
        CHECK(record.level == wxLOG_Info);
        CHECK(record.message.find("dropmerge-log[0]: Episode truncated. Maximum step count exceeded.") == 0);
        CHECK(CountOccurrences(record.message, "dropmerge-log[0]: ") == 1);
    }
    CHECK(matching_records == 1);
}

TEST_CASE("DropMergeEnv reports DROP selected on a NoLegal candidate", "[dropmerge][no_legal_candidate]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeBlockedBoardConfig();
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-candidate[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto first_drop = env->Step(1);
    REQUIRE_FALSE(first_drop->next_state.done);
    REQUIRE_FALSE(first_drop->next_state.truncated);
    REQUIRE(env->GetScalar("blocked_drop_on_candidate").has_value());
    CHECK(std::isnan(*env->GetScalar("blocked_drop_on_candidate")));
    REQUIRE(env->GetScalar("no_drop_timeout_on_candidate").has_value());
    CHECK(std::isnan(*env->GetScalar("no_drop_timeout_on_candidate")));
    REQUIRE(env->GetScalar("ep_mean_blocked_frames").has_value());
    CHECK(std::isnan(*env->GetScalar("ep_mean_blocked_frames")));
    REQUIRE(env->GetScalar("ep_max_blocked_frames").has_value());
    CHECK(std::isnan(*env->GetScalar("ep_max_blocked_frames")));

    const auto blocked_drop = env->Step(1);
    CHECK(blocked_drop->next_state.done);
    CHECK_FALSE(blocked_drop->next_state.truncated);
    REQUIRE(env->GetScalar("term_reason_spawn_blocked").has_value());
    CHECK(*env->GetScalar("term_reason_spawn_blocked") == 1.0f);
    REQUIRE(env->GetScalar("blocked_drop_on_candidate").has_value());
    CHECK(*env->GetScalar("blocked_drop_on_candidate") == 1.0f);
    logs.Flush();
}

TEST_CASE("DropMergeEnv handles a fruit wider than its placement range", "[dropmerge][no_legal_candidate]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeBlockedBoardConfig();
    config.box_width = 0.5f;
    config.settle_velocity_threshold = 100.0f;
    config.settle_angular_threshold = 100.0f;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-no-placement-range[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto noop = env->Step(0);
    CHECK(noop->next_state.done);
    CHECK_FALSE(noop->next_state.truncated);
    REQUIRE(env->GetScalar("term_reason_no_legal_drop").has_value());
    CHECK(*env->GetScalar("term_reason_no_legal_drop") == 1.0f);
    logs.Flush();
}

TEST_CASE("blocked intervals report whether the entire range is covered", "[dropmerge][blocked_intervals]")
{
    const auto covers = [](std::vector<std::pair<float, float>> intervals) {
        return anet::rl::env::drop_merge::DoBlockedIntervalsCoverRange(intervals, -1.0f, 1.0f);
    };

    CHECK_FALSE(covers({}));
    CHECK(covers({ { -1.0f, 1.0f } }));
    CHECK_FALSE(covers({ { -0.9f, 1.0f } }));
    CHECK_FALSE(covers({ { -1.0f, 0.9f } }));
    CHECK_FALSE(covers({ { -1.0f, -0.1f }, { 0.1f, 1.0f } }));
    CHECK(covers({ { 0.0f, 1.0f }, { -1.0f, 0.5f } }));
    CHECK(covers({ { -1.0f, 0.0f }, { 0.0f, 1.0f } }));
}

TEST_CASE("DropMergeEnv keeps settled NoLegal NOOP termination unchanged", "[dropmerge][no_legal_candidate]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeBlockedBoardConfig();
    config.settle_velocity_threshold = 100.0f;
    config.settle_angular_threshold = 100.0f;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-settled[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto first_drop = env->Step(1);
    REQUIRE_FALSE(first_drop->next_state.done);

    const auto noop = env->Step(0);
    CHECK(noop->next_state.done);
    CHECK_FALSE(noop->next_state.truncated);
    REQUIRE(env->GetScalar("term_reason_no_legal_drop").has_value());
    CHECK(*env->GetScalar("term_reason_no_legal_drop") == 1.0f);
    logs.Flush();
}

TEST_CASE("DropMergeEnv reports terminal blocked persistence for settled NoLegal", "[dropmerge][phase1b]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeBlockedBoardConfig();
    config.settle_velocity_threshold = 100.0f;
    config.settle_angular_threshold = 100.0f;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-terminal-blocked[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto first_drop = env->Step(1);
    REQUIRE_FALSE(first_drop->next_state.done);
    REQUIRE(env->GetScalar("ep_terminal_blocked_frames").has_value());
    CHECK(std::isnan(*env->GetScalar("ep_terminal_blocked_frames")));
    REQUIRE(env->GetScalar("ep_blocked_run_count").has_value());
    CHECK(std::isnan(*env->GetScalar("ep_blocked_run_count")));

    const auto noop = env->Step(0);
    REQUIRE(noop->next_state.done);
    REQUIRE(env->GetScalar("ep_terminal_blocked_frames").has_value());
    CHECK(*env->GetScalar("ep_terminal_blocked_frames") > 0.0f);
    REQUIRE(env->GetScalar("ep_blocked_run_count").has_value());
    CHECK(*env->GetScalar("ep_blocked_run_count") == 0.0f);
    logs.Flush();
}

TEST_CASE("DropMergeEnv NoLegal adjudication config defaults to disabled with sixty frames", "[dropmerge][phase2][config]")
{
    const anet::rl::env::drop_merge::DropMergeEnvConfig config;

    CHECK_FALSE(config.use_no_legal_adjudication);
    CHECK(config.no_legal_min_blocked_frames == 60);
}

TEST_CASE("DropMergeEnv reads NoLegal adjudication config values", "[dropmerge][phase2][config]")
{
    anet::ConfigData config_data;
    config_data.Set("DropMergeEnv.use_no_legal_adjudication", true);
    config_data.Set("DropMergeEnv.no_legal_min_blocked_frames", 45);

    const anet::rl::env::drop_merge::DropMergeEnvConfig config(config_data);
    CHECK(config.use_no_legal_adjudication);
    CHECK(config.no_legal_min_blocked_frames == 45);
}

TEST_CASE("DropMergeEnv rejects a non-positive NoLegal adjudication horizon", "[dropmerge][phase2][config]")
{
    anet::rl::env::drop_merge::DropMergeEnvConfig config;
    config.no_legal_min_blocked_frames = 0;

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    CHECK_THROWS_WITH(
        std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
            config, torch::Device(torch::kCPU), "dropmerge-invalid-horizon[0]", 123),
        Catch::Matchers::ContainsSubstring("key=no_legal_min_blocked_frames")
        && Catch::Matchers::ContainsSubstring("value=0")
        && Catch::Matchers::ContainsSubstring("expected integer >= 1"));
    logs.Flush();
}

TEST_CASE("DropMergeEnv rejects a NoLegal horizon that cannot beat the timeout", "[dropmerge][phase2][config]")
{
    anet::rl::env::drop_merge::DropMergeEnvConfig config;
    config.use_no_legal_adjudication = true;
    config.use_no_drop_timeout_gameover = true;
    config.no_drop_timeout_steps = 60;
    config.no_legal_min_blocked_frames = 60;

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    CHECK_THROWS_WITH(
        std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
            config, torch::Device(torch::kCPU), "dropmerge-conflicting-horizon[0]", 123),
        Catch::Matchers::ContainsSubstring("key=no_legal_min_blocked_frames")
        && Catch::Matchers::ContainsSubstring("value=60")
        && Catch::Matchers::ContainsSubstring("expected < no_drop_timeout_steps=60"));
    logs.Flush();
}

TEST_CASE("DropMergeEnv reports timeout with a legal DROP available", "[dropmerge][no_legal_candidate]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::rl::env::drop_merge::DropMergeEnvConfig config;
    config.seed_mode = "fixed";
    config.action_mode = "direct_noop";
    config.use_no_legal_adjudication = true;
    config.no_legal_min_blocked_frames = 1;
    config.no_drop_timeout_steps = 1;
    config.max_step = 10;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-legal-timeout[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto timeout = env->Step(0);
    CHECK_FALSE(timeout->next_state.done);
    CHECK(timeout->next_state.truncated);
    REQUIRE(env->GetScalar("term_reason_no_drop_timeout").has_value());
    CHECK(*env->GetScalar("term_reason_no_drop_timeout") == 1.0f);
    REQUIRE(env->GetScalar("no_drop_timeout_on_candidate").has_value());
    CHECK(*env->GetScalar("no_drop_timeout_on_candidate") == 0.0f);
    REQUIRE(env->GetScalar("ep_terminal_blocked_frames").has_value());
    CHECK(*env->GetScalar("ep_terminal_blocked_frames") == 0.0f);
    REQUIRE(env->GetScalar("ep_blocked_run_count").has_value());
    CHECK(*env->GetScalar("ep_blocked_run_count") == 0.0f);
    logs.Flush();
}

TEST_CASE("DropMergeEnv reports timeout while an unsettled board remains blocked", "[dropmerge][no_legal_candidate]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeBlockedBoardConfig();
    config.no_drop_timeout_steps = 3;
    config.restitution = 1.0f;
    config.box_restitution = 1.0f;
    config.damping = 0.0f;
    config.settle_velocity_threshold = 0.0f;
    config.settle_angular_threshold = 0.0f;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-unsettled[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto first_drop = env->Step(1);
    REQUIRE_FALSE(first_drop->next_state.done);

    std::shared_ptr<const anet::rl::SingleStepResult> timeout;
    for (int i = 0; i < config.no_drop_timeout_steps; ++i) {
        timeout = env->Step(0);
    }

    REQUIRE(timeout != nullptr);
    CHECK_FALSE(timeout->next_state.done);
    CHECK(timeout->next_state.truncated);
    REQUIRE(env->GetScalar("term_reason_no_drop_timeout").has_value());
    CHECK(*env->GetScalar("term_reason_no_drop_timeout") == 1.0f);
    REQUIRE(env->GetScalar("no_drop_timeout_on_candidate").has_value());
    CHECK(*env->GetScalar("no_drop_timeout_on_candidate") == 1.0f);
    REQUIRE(env->GetScalar("ep_mean_blocked_frames").has_value());
    CHECK(*env->GetScalar("ep_mean_blocked_frames") == 0.0f);
    REQUIRE(env->GetScalar("ep_max_blocked_frames").has_value());
    CHECK(*env->GetScalar("ep_max_blocked_frames") == 0.0f);
    logs.Flush();
}

TEST_CASE("DropMergeEnv adjudicates persistent blocked frames without a penalty", "[dropmerge][phase2][adjudication]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeBlockedBoardConfig();
    config.use_no_legal_adjudication = true;
    config.no_legal_min_blocked_frames = 3;
    config.use_no_drop_timeout_gameover = true;
    config.no_drop_timeout_steps = 10;
    config.restitution = 1.0f;
    config.box_restitution = 1.0f;
    config.damping = 0.0f;
    config.settle_velocity_threshold = 0.0f;
    config.settle_angular_threshold = 0.0f;
    config.time_penalty = 0.0f;
    config.noop_penalty = 0.0f;
    config.game_over_penalty = -10.0f;
    config.no_drop_timeout_gameover_penalty = -10.0f;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-adjudicated[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto first_drop = env->Step(1);
    REQUIRE_FALSE(first_drop->next_state.done);

    const auto before_horizon = env->Step(0);
    CHECK_FALSE(before_horizon->next_state.done);
    CHECK_FALSE(before_horizon->next_state.truncated);

    const auto adjudicated = env->Step(0);
    CHECK(adjudicated->next_state.done);
    CHECK_FALSE(adjudicated->next_state.truncated);
    CHECK(adjudicated->reward == Catch::Approx(0.0f));
    REQUIRE(env->GetScalar("term_reason_no_legal_drop").has_value());
    CHECK(*env->GetScalar("term_reason_no_legal_drop") == 1.0f);
    REQUIRE(env->GetScalar("ep_terminal_blocked_frames").has_value());
    CHECK(*env->GetScalar("ep_terminal_blocked_frames") == 3.0f);

    logs.Flush();
    int persistence_log_count = 0;
    for (const auto& record : logs.Records()) {
        if (record.message.find("Episode done: no legal drop persisted for 3 frames.") != std::string::npos) {
            ++persistence_log_count;
        }
    }
    CHECK(persistence_log_count == 1);
}

TEST_CASE("DropMergeEnv prefers the settled NoLegal fast path", "[dropmerge][phase2][adjudication]")
{
    ScopedNoopMetricsLogger metrics_logger;
    auto config = MakeBlockedBoardConfig();
    config.use_no_legal_adjudication = true;
    config.no_legal_min_blocked_frames = 1;
    config.settle_velocity_threshold = 100.0f;
    config.settle_angular_threshold = 100.0f;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-fast-path[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto first_drop = env->Step(1);
    REQUIRE_FALSE(first_drop->next_state.done);

    const auto noop = env->Step(0);
    CHECK(noop->next_state.done);
    CHECK_FALSE(noop->next_state.truncated);

    logs.Flush();
    int settled_log_count = 0;
    int persistence_log_count = 0;
    for (const auto& record : logs.Records()) {
        if (record.message.find("Episode done: no legal drop remains.") != std::string::npos) {
            ++settled_log_count;
        }
        if (record.message.find("Episode done: no legal drop persisted") != std::string::npos) {
            ++persistence_log_count;
        }
    }
    CHECK(settled_log_count == 1);
    CHECK(persistence_log_count == 0);
}

TEST_CASE("DropMergeEnv reports the length of a resolved blocked run", "[dropmerge][blocked_persistence]")
{
    ScopedNoopMetricsLogger metrics_logger;
    anet::rl::env::drop_merge::DropMergeEnvConfig config;
    config.seed_mode = "fixed";
    config.action_mode = "direct_noop";
    config.drop_divisions = 1;
    config.box_width = 1.0f;
    config.box_height = 2.0f;
    config.fruit_radii.assign(anet::rl::env::drop_merge::kFruitTypeCount, 0.1f);
    config.drop_probs = { 1.0f };
    config.drop_noise = 0.0f;
    config.spin_noise = 0.0f;
    config.restitution = 0.0f;
    config.damping = 0.0f;
    config.use_instant_drop = true;
    config.reload_min_steps = 0;
    config.reload_max_steps = 1;
    config.use_no_legal_adjudication = true;
    config.no_legal_min_blocked_frames = 29;
    config.no_drop_timeout_steps = 30;
    config.max_step = 100;
    config.game_over_grace_step = 1000;
    config.settle_velocity_threshold = 0.0f;
    config.settle_angular_threshold = 0.0f;
    auto env = std::make_shared<anet::rl::env::drop_merge::DropMergeEnv>(
        config, torch::Device(torch::kCPU), "dropmerge-resolved-run[0]", 123);
    env->Reset();

    anet::test::LogCaptureGuard logs(wxLOG_Info);
    const auto first_drop = env->Step(1);
    REQUIRE_FALSE(first_drop->next_state.done);

    std::shared_ptr<const anet::rl::SingleStepResult> timeout;
    for (int i = 0; i < config.no_drop_timeout_steps; ++i) {
        timeout = env->Step(0);
    }

    REQUIRE(timeout != nullptr);
    REQUIRE(timeout->next_state.truncated);
    REQUIRE(env->GetScalar("ep_mean_blocked_frames").has_value());
    REQUIRE(env->GetScalar("ep_max_blocked_frames").has_value());
    const float mean_blocked_frames = *env->GetScalar("ep_mean_blocked_frames");
    const float max_blocked_frames = *env->GetScalar("ep_max_blocked_frames");
    CHECK(mean_blocked_frames > 0.0f);
    CHECK(mean_blocked_frames == max_blocked_frames);
    CHECK(max_blocked_frames < static_cast<float>(config.no_drop_timeout_steps));
    REQUIRE(env->GetScalar("ep_terminal_blocked_frames").has_value());
    CHECK(*env->GetScalar("ep_terminal_blocked_frames") == 0.0f);
    REQUIRE(env->GetScalar("ep_blocked_run_count").has_value());
    CHECK(*env->GetScalar("ep_blocked_run_count") > 0.0f);
    logs.Flush();
}

int main(int argc, char* argv[])
{
    SetupUtf8Console();

    anet::test::PreparedTestArgs test_args;
    try {
        test_args = anet::test::PrepareTestArgs(argc, argv);
    } catch (const std::exception& e) {
        return anet::test::ReportTestArgsError(e);
    }
    anet::test::SetupTestFailureDialog(test_args.failure_dialog_enabled);

    Catch::Session session;
    session.configData().showDurations = Catch::ShowDurations::Always;
    return session.run(test_args.Argc(), test_args.Argv());
}
