#include "anet/catch_test.hpp"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <thread>
#include <wx/evtloop.h>
#include <wx/textctrl.h>
#include <wx/weakref.h>
#include "anet/log.hpp"
#include "anet/agent.hpp"
#include "RunnerApp.hpp"

namespace {
using Clock = std::chrono::steady_clock;
constexpr char kViewMenu[] = "View";
constexpr char kHeatMapMenu[] = "HeatMap";
constexpr char kCloseAgentPath[] = "startup/agent_close.anet";
constexpr char kResolvedBeforeDirectory[] = "resolved-before";
constexpr char kSlowInitTestAgentClassId[] = "SlowInitTestAgent";

class StartupTestActor final : public anet::rl::Actor {
public:
    std::shared_ptr<anet::rl::BatchActionInfo> MakeAction(
        const anet::rl::StepCounts&, const anet::rl::BatchState&) const override
    { return std::make_shared<anet::rl::BatchActionInfo>(torch::zeros({1}, torch::kInt64)); }
    void Sync() override {}
};

class SlowInitTestAgent final : public anet::rl::Agent {
private:
    struct Config final : anet::Config {
        int construction_delay_ms = 0;
        bool construction_fails = false;
        explicit Config(const anet::ConfigData& config_data) : anet::Config(config_data, "agent")
        {
            ANET_READ_CONFIG(config_data, construction_delay_ms);
            ANET_READ_CONFIG(config_data, construction_fails);
            if (construction_delay_ms < 0) {
                ANET_SYSTEM_ERROR("Invalid agent.construction_delay_ms: value=" << construction_delay_ms
                    << " (expected: >= 0)");
            }
        }
    };
public:
    explicit SlowInitTestAgent(const anet::ConfigData& config)
    {
        const Config settings(config);
        constexpr auto kLogInterval = std::chrono::milliseconds(200);
        const auto start = Clock::now();
        const auto end = start + std::chrono::milliseconds(settings.construction_delay_ms);
        for (auto next = start + kLogInterval; next <= end; next += kLogInterval) {
            std::this_thread::sleep_until(next);
            anet::log::info() << "SlowInitTestAgent waiting. elapsed_ms="
                << std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
        }
        std::this_thread::sleep_until(end);
        if (settings.construction_fails) ANET_SYSTEM_ERROR("SlowInitTestAgent failed by agent.construction_fails.");
    }
    std::shared_ptr<anet::rl::Actor> CreateActor(const anet::rl::ActorRequest&) const override
    { return std::make_shared<StartupTestActor>(); }
    std::shared_ptr<anet::rl::Learner> CreateLearner() override { return nullptr; }
    torch::Device GetDevice() const override { return torch::kCPU; }
    std::optional<float> GetScalar(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t = -1) const override
    { return std::nullopt; }
};

class SlowInitTestAgentFactory final : public anet::rl::AgentFactory {
public:
    std::shared_ptr<anet::rl::Agent> CreateAgent(const anet::rl::EnvSpec&, const anet::rl::BatchEnvSpec&,
        const torch::Device&, const anet::ConfigData& config, std::shared_ptr<anet::rl::Notifier>,
        std::optional<anet::seed_t>) const override
    { return std::make_shared<SlowInitTestAgent>(config); }
    std::string GetTargetAgentClassId() const override { return kSlowInitTestAgentClassId; }
};

struct StartupFixture {
    wxEventLoop loop;
    wxEventLoopActivator activator{&loop};
    const std::filesystem::path output = std::filesystem::temp_directory_path()
        / ("anet-prd109-" + std::to_string(wxGetProcessId()));
    wxWeakRef<wxFrame> frame;
    bool exited = false;

    void Start(const std::string& overrides = "")
    {
        std::filesystem::create_directories(output);
        constexpr char kConfigFile[] = "startup.txt";
        const auto path = output / kConfigFile;
        {
            std::ofstream config(path);
            config << "env.class_id = CartPoleEnv\nenv.worker_type = 1\nenv.device = cpu\n"
                "agent.class_id = " << kSlowInitTestAgentClassId << "\nagent.device = cpu\nrun.eval_device = cpu\n"
                "agent.construction_delay_ms = 3000\nrun.seed = 1\nrun.train.num_envs = 1\n"
                "run.train.runner_type = serial\nrun.eval.[panel].run_mode = eval\n"
                "app.eval_panel.eval_config_tag = panel\napp.train_auto_start = false\n"
                "app.save_agent_on_close = true\napp.show_error_dialog = false\n"
                "app.train_panel.fps = 0\napp.eval_panel.fps = 0\napp.eval_panel.auto_start = false\n"
                "backend.torch_num_threads = 1\napp.run_name = startup\napp.runs_dir = "
                << output.generic_string() << "\n" << overrides;
        }
        auto executable = wxTheApp->argv[0].ToStdWstring();
        auto config = path.wstring();
        constexpr auto kConfigOption = L"--config";
        std::wstring option(kConfigOption);
        wchar_t* argv[] = {executable.data(), option.data(), config.data()};
        wxTheApp->argc = 3;
        wxTheApp->argv.Init(3, argv);
        wxTheApp->SetExitOnFrameDelete(false);
        REQUIRE(wxTheApp->OnInit());
        frame = wxGetApp().GetMainFrame();
        REQUIRE(frame);
    }

    template<class Predicate> bool Wait(Predicate done)
    {
        // timerとCallAfterを配送しながら、承認済み計画の30秒を上限に待つ。
        const auto end = Clock::now() + std::chrono::seconds(30);
        while (!done() && Clock::now() < end) {
            while (loop.Pending()) loop.Dispatch();
            wxTheApp->ProcessPendingEvents();
            loop.ProcessIdle();
            wxMilliSleep(2);
        }
        return done();
    }

    std::string Log() const
    {
        constexpr char kRunLog[] = "startup/startup.log";
        std::ifstream input(output / kRunLog);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    ~StartupFixture()
    {
        if (exited) return;
        if (frame) frame->Close(true);
        wxTheApp->Yield(true);
        wxTheApp->OnExit();
    }
};
}

// GUI test の入口からだけ呼び、製品の AgentRepository へは登録しない。
void RegisterStartupTestAgents()
{
    anet::rl::AgentRepository::Instance().Register(std::make_shared<SlowInitTestAgentFactory>());
}

TEST_CASE("Runner construction keeps the event loop and logs active", "[.][prd109_startup][prd109_ready]")
{
    StartupFixture app;
    app.Start();
    CHECK_FALSE(wxGetApp().IsRunInitialized());
    const auto returned = Clock::now();
    wxTextCtrl* text = nullptr;
    for (auto* child : app.frame->GetChildren()) {
        if (auto* log = dynamic_cast<LogPanel*>(child)) {
            for (auto* control : log->GetChildren())
                if (auto* candidate = dynamic_cast<wxTextCtrl*>(control)) text = candidate;
        }
    }
    REQUIRE(text);
    int ticks = 0;
    bool logged_before_ready = false;
    wxTimer timer;
    timer.Bind(wxEVT_TIMER, [&](wxTimerEvent&) {
        ++ticks;
        if (!wxGetApp().IsRunInitialized() && text->GetValue().Contains("SlowInitTestAgent waiting."))
            logged_before_ready = true;
    });
    timer.Start(100);
    REQUIRE(app.Wait([] { return wxGetApp().IsRunInitialized(); }));
    timer.Stop();
    CHECK(Clock::now() - returned >= std::chrono::seconds(2));
    CHECK(ticks >= 10);
    CHECK(logged_before_ready);
}

TEST_CASE("Runner startup keeps panes usable and auxiliary views unavailable", "[.][prd109_startup][prd109_ui]")
{
    StartupFixture app;
    app.Start();
    auto* bar = app.frame->GetMenuBar();
    const int heatmap = bar->FindMenuItem(kViewMenu, kHeatMapMenu);
    REQUIRE(heatmap != wxNOT_FOUND);
    wxUpdateUIEvent update(heatmap);
    app.frame->GetEventHandler()->ProcessEvent(update);
    REQUIRE_FALSE(update.GetEnabled());
    auto* manager = wxAuiManager::GetManager(app.frame);
    REQUIRE(manager);
    const auto count = manager->GetAllPanes().GetCount();
    for (const auto* name : {kHeatMapMenu, "Conv2d"}) {
        wxCommandEvent event(wxEVT_MENU, bar->FindMenuItem(kViewMenu, name));
        app.frame->GetEventHandler()->ProcessEvent(event);
    }
    CHECK(manager->GetAllPanes().GetCount() == count);
    for (const auto* name : {"Log View", "Evaluation View", "Evaluation QValue View", "Evaluation Trace"}) {
        const int id = bar->FindMenuItem(kViewMenu, name);
        REQUIRE(id != wxNOT_FOUND);
        for (const bool show : {true, false}) {
            wxCommandEvent event(wxEVT_MENU, id);
            event.SetInt(show);
            app.frame->GetEventHandler()->ProcessEvent(event);
        }
    }
    // Train は現行UIでは固定の中央paneでmenuが無いので、公開AUI操作で往復する。
    for (auto& pane : manager->GetAllPanes()) {
        if (dynamic_cast<TrainPanel*>(pane.window)) {
            pane.Hide(); manager->Update();
            pane.Show(); manager->Update();
        }
    }
    app.frame->SetSize(1000, 650);
    app.frame->SetSize(1100, 700);
    REQUIRE(app.Wait([] { return wxGetApp().IsRunInitialized(); }));
    wxGetApp().FlushRunOutputs();
    const auto log = app.Log();
    const std::regex elapsed(R"(Run initialized\. elapsed_sec=(\d+\.\d{2})(\r?\n))");
    std::smatch match;
    REQUIRE(std::regex_search(log, match, elapsed));
    CHECK(std::stod(match[1]) >= 3.0);
    const auto occurrences = [&log](const std::string& text) {
        size_t count = 0;
        for (size_t at = 0; (at = log.find(text, at)) != std::string::npos; at += text.size()) ++count;
        return count;
    };
    CHECK(occurrences("Initializing run in background.") == 1);
    CHECK(occurrences("Run initialized.") == 1);
}

TEST_CASE("Runner defers repeated close requests until construction completes", "[.][prd109_startup][prd109_close]")
{
    StartupFixture app;
    app.Start();
    CHECK_FALSE(wxGetApp().PrepareRunClose(true));
    CHECK_FALSE(app.frame->Close());
    CHECK_FALSE(app.frame->Close());
    CHECK(app.frame);
    wxGetApp().FlushRunOutputs();
    const auto requested = app.Log();
    const std::string message = "Close requested during initialization.";
    const auto first = requested.find(message);
    REQUIRE(first != std::string::npos);
    CHECK(requested.find(message, first + message.size()) == std::string::npos);
    REQUIRE(app.Wait([&] { return !app.frame; }));
    CHECK_FALSE(wxGetApp().IsRunInitialized());
    CHECK_FALSE(wxGetApp().IsTrainingRunning());
    CHECK_THROWS_AS(wxGetApp().GetRunManager().GetEvalRunner("EvalPanel"), std::out_of_range);
    CHECK(app.Log().find("Run initialized.") == std::string::npos);
    CHECK(std::filesystem::exists(app.output / kCloseAgentPath));
}

TEST_CASE("Runner waits for construction without vetoing session queries", "[.][prd109_startup][prd109_query]")
{
    StartupFixture app;
    app.Start();
    wxCloseEvent query(wxEVT_QUERY_END_SESSION);
    query.SetCanVeto(true);
    query.SetLoggingOff(true);
    const auto start = Clock::now();
    wxTheApp->ProcessEvent(query);
    CHECK_FALSE(query.GetVeto());
    CHECK(Clock::now() - start >= std::chrono::seconds(2));
    REQUIRE(app.Wait([&] { return !app.frame; }));
    CHECK_FALSE(wxGetApp().IsRunInitialized());
    CHECK_FALSE(wxGetApp().IsTrainingRunning());
    CHECK(app.Log().find("Run initialized.") == std::string::npos);
    CHECK(std::filesystem::exists(app.output / kCloseAgentPath));
}

TEST_CASE("Runner waits on forced close before construction finishes", "[.][prd109_startup][prd109_forced_close]")
{
    StartupFixture app;
    app.Start();
    const auto start = Clock::now();
    CHECK(app.frame->Close(true));
    CHECK(Clock::now() - start >= std::chrono::seconds(2));
    REQUIRE(app.Wait([&] { return !app.frame; }));
    CHECK_FALSE(wxGetApp().IsRunInitialized());
    CHECK_FALSE(wxGetApp().IsTrainingRunning());
    CHECK(app.Log().find("Run initialized.") == std::string::npos);
    CHECK(std::filesystem::exists(app.output / kCloseAgentPath));
}

TEST_CASE("Runner reports construction failure and still closes forcibly", "[.][prd109_startup][prd109_close_failure]")
{
    StartupFixture app;
    app.Start("agent.construction_fails = true\n");
    bool threw = false;
    try { app.frame->Close(true); }
    catch (...) { threw = true; }
    CHECK_FALSE(threw);
    if (threw) {
        app.exited = true;
        wxTheApp->CallOnExit();
        return;
    }
    REQUIRE(app.Wait([&] { return !app.frame; }));
    CHECK_FALSE(wxGetApp().IsRunInitialized());
    CHECK(app.Log().find("SlowInitTestAgent failed by agent.construction_fails.") != std::string::npos);
    CHECK_FALSE(std::filesystem::exists(app.output / kCloseAgentPath));
    app.exited = true;
    CHECK(wxTheApp->OnExit() == 1);
}

TEST_CASE("Runner can close an incomplete RunManager after construction", "[.][prd109_startup][prd109_incomplete_after]")
{
    StartupFixture app;
    app.Start("env.class_id = Prd109UnregisteredEnv\n");
    REQUIRE(app.Wait([&] { return app.Log().find("Failed to initialize RunManager.") != std::string::npos; }));
    REQUIRE(app.frame);
    CHECK_FALSE(wxGetApp().IsRunInitialized());
    CHECK(app.frame->Close());
    REQUIRE(app.Wait([&] { return !app.frame; }));
    CHECK(app.Log().find("Failed to create env.") != std::string::npos);
    CHECK_FALSE(std::filesystem::exists(app.output / kCloseAgentPath));
    wxTheApp->CallAfter([] { wxTheApp->ExitMainLoop(); });
    CHECK(wxTheApp->OnRun() == 0);
    app.exited = true;
    CHECK(wxTheApp->OnExit() == 0);
}

TEST_CASE("Runner joins construction after CallOnExit deletes its windows", "[.][prd109_startup][prd109_exit]")
{
    StartupFixture app;
    app.Start();
    app.exited = true;
    const auto start = Clock::now();
    CHECK(wxTheApp->CallOnExit() == 0);
    CHECK(Clock::now() - start >= std::chrono::seconds(2));
    CHECK_FALSE(app.frame);
    CHECK_FALSE(wxGetApp().IsRunInitialized());
    const auto log = app.Log();
    const std::regex elapsed(R"(SlowInitTestAgent waiting\. elapsed_ms=(\d+))");
    int last_elapsed = 0;
    for (auto it = std::sregex_iterator(log.begin(), log.end(), elapsed); it != std::sregex_iterator(); ++it)
        last_elapsed = std::stoi((*it)[1]);
    CHECK(last_elapsed >= 3000);
    CHECK(app.Log().find("Run initialized.") == std::string::npos);
    CHECK_FALSE(std::filesystem::exists(app.output / kCloseAgentPath));
    // 遅れて届いた完了通知はGUIへ触れない。
    wxTheApp->ProcessPendingEvents();
}

TEST_CASE("Runner closes an incomplete RunManager requested before delivery", "[.][prd109_startup][prd109_incomplete_before]")
{
    StartupFixture app;
    app.Start("env.class_id = Prd109UnregisteredEnv\n");
    // 完了通知を処理する前なので、構築自体の速さによらず初期化中である。
    CHECK_FALSE(app.frame->Close());
    REQUIRE(app.frame);
    REQUIRE(app.Wait([&] { return !app.frame; }));
    CHECK_FALSE(wxGetApp().IsRunInitialized());
    CHECK(app.Log().find("Failed to create env.") != std::string::npos);
    CHECK(app.Log().find("Failed to initialize RunManager.") != std::string::npos);
    CHECK_FALSE(std::filesystem::exists(app.output / kCloseAgentPath));
}

TEST_CASE("Runner returns failure after windows are deleted during construction", "[.][prd109_startup][prd109_exit_failure]")
{
    StartupFixture app;
    app.Start("agent.construction_fails = true\n");
    app.exited = true;
    const auto start = Clock::now();
    CHECK(wxTheApp->CallOnExit() == 1);
    CHECK(Clock::now() - start >= std::chrono::seconds(2));
    CHECK_FALSE(app.frame);
    CHECK_FALSE(wxGetApp().IsRunInitialized());
    CHECK_FALSE(std::filesystem::exists(app.output / kCloseAgentPath));
    wxTheApp->ProcessPendingEvents();
}

TEST_CASE("PRD109 captures configuration from frozen inputs", "[.][prd109_config]")
{
    constexpr char kOutputVariable[] = "PRD109_CONFIG_OUTPUT";
    const char* output_env = std::getenv(kOutputVariable);
    REQUIRE(output_env);
    const std::filesystem::path output(output_env);
    constexpr char kBaselineConfigDirectory[] = "baseline-config";
    const auto root = output / kBaselineConfigDirectory;
    std::filesystem::create_directories(output / kResolvedBeforeDirectory);
    for (const auto* env : {"GridMaze", "LunarLander", "Atari", "DropMerge"}) {
        const auto config = output / (std::string(env) + "-capture.txt");
        {
            std::ofstream stream(config);
            stream << "$include <" << root.generic_string() << "/_main.txt>\n"
                << "$include <" << root.generic_string() << '/' << env << ".txt>\n";
        }
        const anet::ConfigManager manager(config);
        const auto config_data = manager.GetConfigData();
        anet::json values = anet::json::object();
        for (const auto& [key, value] : config_data.Map()) values[key] = value;
        std::ofstream stream(output / kResolvedBeforeDirectory / (std::string(env) + ".json"));
        stream << values.dump(2) << '\n';
        REQUIRE(stream.good());
    }
}
