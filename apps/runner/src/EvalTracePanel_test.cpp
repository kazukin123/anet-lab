#include "anet/catch_test.hpp"
#include <fstream>
#include <string_view>
#include "anet/test_util.hpp"
#include "EvalTracePanel.hpp"
#include "LogPanel.hpp"
#include "QValuePanel.hpp"
#include "RunnerApp.hpp"
#include <wx/app.h>
#include <wx/modalhook.h>
#include <wx/textdlg.h>

#include "anet/agent.hpp"
#include "anet/env/CartPole.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/observers.hpp"
#include <wx/dcclient.h>
#include <wx/settings.h>
#include <numeric>
#include <iostream>
#include <wx/dcmemory.h>
#include <wx/evtloop.h>
#include <wx/stattext.h>
#include <wx/utils.h>

namespace {
std::filesystem::path TraceUiOutputDirectory()
{
    // カレントディレクトリから独立し、同時実行するprocessごとに出力を分ける。
    return std::filesystem::temp_directory_path() / ("anet-eval-trace-" + std::to_string(wxGetProcessId()));
}

class TraceUiScalars final : public anet::ModuleBase {
public:
    explicit TraceUiScalars(float value) : value_(value) {}
    std::optional<float> GetScalar(const std::string& key, int64_t = -1) const override
    { return key == "later" ? std::optional(value_) : std::nullopt; }
private:
    float value_;
};
class TraceUiDiagnosticsProvider final : public anet::rl::AgentStepDiagnosticsProvider {
public:
    anet::rl::AgentStepDiagnosticsResult Observe(const anet::rl::BatchExperience&) override
    {
        anet::rl::AgentStepDiagnosticsResult result;
        result.current = std::make_unique<TraceUiScalars>(std::numeric_limits<float>::quiet_NaN());
        result.pending_keys = {{"later"}};
        if (step_++ > 0) result.completed.push_back({.lane=0, .steps_back=1, .resolved_keys={"later"},
            .values=std::make_unique<TraceUiScalars>(2000000000.0f)});
        return result;
    }
    void Reset() override { step_ = 0; }
private:
    int step_ = 0;
};
// 実際のEval step数を測るときも、NNや学習を起動せず一定の行動を返す。
class TraceUiActor final : public anet::rl::Actor {
public:
    std::shared_ptr<anet::rl::BatchActionInfo> MakeAction(
        const anet::rl::StepCounts&, const anet::rl::BatchState&) const override
    { return std::make_shared<anet::rl::BatchActionInfo>(torch::zeros({1}, torch::kInt64)); }
    void Sync() override {}
};
class TraceUiAgent final : public anet::rl::Agent {
public:
    std::unique_ptr<anet::rl::AgentStepDiagnosticsProvider> CreateStepDiagnosticsProvider(int64_t) const override
    { return std::make_unique<TraceUiDiagnosticsProvider>(); }
    std::shared_ptr<anet::rl::Actor> CreateActor(const anet::rl::ActorRequest&) const override { return std::make_shared<TraceUiActor>(); }
    std::shared_ptr<anet::rl::Learner> CreateLearner() override { return nullptr; }
    torch::Device GetDevice() const override { return torch::kCPU; }
    std::optional<float> GetScalar(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<torch::Tensor> GetTensor(const std::string&, int64_t = -1) const override { return std::nullopt; }
    std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string&, int64_t = -1) const override { return std::nullopt; }
};
class TraceUiAgentFactory final : public anet::rl::AgentFactory {
public:
    std::shared_ptr<anet::rl::Agent> CreateAgent(const anet::rl::EnvSpec&, const anet::rl::BatchEnvSpec&,
        const torch::Device&, const anet::ConfigData&, std::shared_ptr<anet::rl::Notifier>, std::optional<anet::seed_t>) const override
    { return std::make_shared<TraceUiAgent>(); }
    std::string GetTargetAgentClassId() const override { return "TraceUiAgent"; }
};
class TraceUiMetricsBackend final : public anet::IBackend {
public:
    void Open(const std::filesystem::path&, const std::string&) override {}
    void WriteJsonl(const anet::json&) override {}
    void Flush() override {}
};
// Traceの解除を公開のfollower通知から観測し、破棄後のメモリには触れない。
class TraceCloseFollowerProbe final : public wxPanel {
public:
    explicit TraceCloseFollowerProbe(wxWindow* parent) : wxPanel(parent) {}
    std::shared_ptr<const anet::rl::gui::TraceSnapshot> CreateTraceSnapshot(const anet::rl::TrainEvent&)
    { return std::make_shared<anet::rl::gui::TraceSnapshot>(); }
    void ShowTraceSnapshot(std::shared_ptr<const anet::rl::gui::TraceSnapshot>) { ++show_calls; }
    void ReturnToLive() { ++live_calls; }
    int show_calls = 0;
    int live_calls = 0;
};
// 実Runnerと同じ基底を使い、managerメンバが子より先に破棄される順序を通す。
class TraceCloseFrameFixture final : public anet::rl::gui::AuiLayoutFrame {
public:
    TraceCloseFrameFixture() : AuiLayoutFrame("Trace close acceptance", wxSize(900, 400)) { CreateStatusBar(); }
    wxAuiManager& Manager() { return aui_mgr_; }
};
struct TraceUiFixture {
    wxFrame frame{nullptr, wxID_ANY, "Evaluation Trace acceptance", wxDefaultPosition, wxSize(900, 400)};
    wxAuiManager manager{&frame};
    std::shared_ptr<anet::rl::RunManager> run;
    std::shared_ptr<anet::rl::EvalRunner> runner;
    EvalPanel* eval = nullptr;
    EvalTracePanel* panel = nullptr;
    wxGrid* grid = nullptr;
    wxAuiToolBar* toolbar = nullptr;
    explicit TraceUiFixture(anet::ConfigData config = {})
    {
        frame.CreateStatusBar();
        // 実RunManagerの通知を使い、学習やゲームの進行は起動しない。
        anet::MetricsLogger::Reset();
        anet::MetricsLoggerConfig logger_config;
        logger_config.run_name_tmpl = "trace_ui_test";
        anet::MetricsLogger::Init(std::make_unique<TraceUiMetricsBackend>(), logger_config,
            TraceUiOutputDirectory() / "metrics");
        config.Set("env.class_id", "CartPoleEnv");
        config.Set("env.worker_type", "1");
        config.Set("env.device", "cpu");
        config.Set("agent.class_id", "TraceUiAgent");
        config.Set("agent.device", "cpu");
        config.Set("run.eval_device", "cpu");
        config.Set("run.seed", "1");
        config.Set("run.train.num_envs", "1");
        config.Set("run.train.runner_type", "serial");
        config.Set("run.eval.[panel].run_mode", "eval");
        if (!config.Has("app.eval_trace_panel.columns.[1]"))
            config.Set("app.eval_trace_panel.columns.[1]", "$env_step reward label:r");
        run = std::make_shared<anet::rl::RunManager>(config);
        runner = run->CreateEvalRunner("EvalPanel", "panel");
        eval = new EvalPanel(&frame, EvalPanelConfig{}, [] {});
        eval->SetFps(14.5f);
        panel = new EvalTracePanel(frame, manager, *eval, config);
        manager.AddPane(panel, wxAuiPaneInfo().CenterPane().Name("trace"));
        panel->Initialize(run, runner);
        manager.Update();
        for (auto* child : panel->GetChildren()) {
            if (auto* candidate = dynamic_cast<wxGrid*>(child)) grid = candidate;
            for (auto* nested : child->GetChildren())
                if (auto* candidate = dynamic_cast<wxAuiToolBar*>(nested)) toolbar = candidate;
        }
    }
    ~TraceUiFixture()
    {
        manager.UnInit();
        delete panel;
        delete eval;
        runner.reset();
        run.reset();
        anet::MetricsLogger::Reset();
    }
    void Add(float reward, bool done = false, bool truncated = false)
    {
        // 公開のNotifierとStep結果から表へ届ける。内部の行やhandlerには触れない。
        using namespace anet::rl;
        const BatchState state{anet::TensorDict{}, torch::tensor({done}), torch::tensor({truncated}), torch::zeros({1}, torch::kBool)};
        auto action = std::make_shared<BatchActionInfo>(torch::zeros({1}, torch::kInt64));
        const BatchExperience experience{state, action, torch::tensor({reward}), state};
        auto result = std::make_shared<PlainBatchStepResult>(experience.reward, state, state, 1, 0, std::vector<AuxData>(1));
        TrainEvent event{{experience, runner, {.exp_step=++step}, run->GetAgent(), {}}, runner->GetBatchEnv(), result, action};
        const auto begin = std::chrono::steady_clock::now();
        run->GetNotifier()->Notify(event);
        last_notify_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    }
    void Tool(size_t index)
    {
        wxCommandEvent event(wxEVT_TOOL, toolbar->FindToolByIndex(index)->GetId());
        toolbar->GetEventHandler()->ProcessEvent(event);
    }
    anet::rl::step_t step = 0;
    double last_notify_ms = 0;
};
}
TEST_CASE("Trace ignores show notifications once destruction starts", "[prd094][trace_ui_close]")
{
    TraceUiFixture ui;
    auto* follower = new TraceCloseFollowerProbe(&ui.frame);
    ui.manager.AddPane(follower, wxAuiPaneInfo().Right().Name("close-probe").Caption("Probe"));
    ui.panel->RegisterFollower(follower);
    ui.manager.Update();
    ui.frame.Show();
    wxTheApp->Yield(true);
    ui.Add(1);
    ui.Tool(2);
    REQUIRE(ui.grid->GetNumberRows() == 1);
    REQUIRE(follower->show_calls > 0);
    const int live_before = follower->live_calls;

    // SendDestroyEvent直後ならmanagerと子は生存している。非表示通知だけを決定的に再現する。
    bool observed_destroy = false;
    ui.manager.GetPane(ui.panel).Hide();
    ui.panel->Bind(wxEVT_DESTROY, [&](wxWindowDestroyEvent& event) {
        if (event.GetEventObject() == ui.panel) {
            observed_destroy = true;
            CHECK(ui.panel->IsBeingDeleted());
            wxShowEvent hidden(ui.panel->GetId(), false);
            hidden.SetEventObject(ui.panel);
            ui.panel->GetEventHandler()->ProcessEvent(hidden);
            CHECK(follower->live_calls == live_before);
            CHECK(ui.grid->GetNumberRows() == 1);
            ui.manager.DetachPane(ui.panel);
        }
        event.Skip();
    });
    ui.panel->Destroy();
    ui.panel = nullptr;
    CHECK(observed_destroy);
    CHECK(follower->live_calls == live_before);
}

TEST_CASE("Trace grid follows numeric sorting and preserves row selection and playback", "[prd105][trace_ui_sort]")
{
    anet::ConfigData config;
    config.Set("app.eval_trace_panel.max_rows", "3");
    config.Set("app.eval_trace_panel.columns.[1]", "$env_step reward label:r");
    config.Set("app.eval_trace_panel.columns.[2]", "$runner exp_step label:step");
    TraceUiFixture ui(config);
    ui.Add(9);
    ui.Add(100);
    ui.Add(10);
    const auto header = [&](int col) {
        wxGridEvent event(ui.grid->GetId(), wxEVT_GRID_LABEL_LEFT_CLICK, ui.grid, -1, col);
        ui.grid->GetEventHandler()->ProcessEvent(event);
    };
    wxGridEvent select(ui.grid->GetId(), wxEVT_GRID_CELL_LEFT_CLICK, ui.grid, 0, 0);
    ui.grid->GetEventHandler()->ProcessEvent(select);
    REQUIRE(ui.panel->IsTraceMode());
    header(0);
    CHECK(ui.grid->GetCellValue(0, 0) == "100");
    CHECK(ui.grid->GetCellValue(1, 0) == "10");
    CHECK(ui.grid->GetGridCursorRow() == 1);
    CHECK(ui.grid->GetSortingColumn() == 0);
    CHECK_FALSE(ui.grid->IsSortOrderAscending());
    header(0);
    CHECK(ui.grid->GetCellValue(0, 0) == "9");
    CHECK(ui.grid->GetGridCursorRow() == 1);
    CHECK(ui.grid->IsSortOrderAscending());
    ui.Tool(0);
    header(1);
    CHECK(ui.grid->GetGridCursorRow() == 0);
    CHECK(ui.toolbar->FindToolByIndex(0)->GetShortHelp() == "Pause");
    ui.panel->ReturnToLive();
    header(0);
    ui.Add(-1);
    CHECK_FALSE(ui.panel->IsTraceMode());
    CHECK(ui.grid->GetNumberRows() == 3);
    CHECK(ui.grid->GetCellValue(0, 0) == "100");
    CHECK(ui.grid->GetCellValue(2, 0) == "-1");
    ui.panel->SetActive(false);
    ui.panel->SetActive(true);
    ui.Add(7);
    ui.Add(1);
    CHECK(ui.grid->GetCellValue(0, 0) == "1");
    CHECK(ui.grid->GetSortingColumn() == 1);
    CHECK_FALSE(ui.grid->IsSortOrderAscending());
}

TEST_CASE("Trace header paints the current sorting direction", "[prd105][trace_ui_sort_icon]")
{
    anet::ConfigData config;
    config.Set("app.eval_trace_panel.columns.[1]", "$env_step reward label:r");
    config.Set("app.eval_trace_panel.columns.[2]", "$runner exp_step label:step");
    TraceUiFixture ui(config);
    ui.frame.Show();
    ui.frame.Raise();
    wxTheApp->Yield(true);
    // プロパティだけでなく、実際の見出しに描かれた印の有無と向きを比べる。
    wxClientDC label_dc(ui.grid);
    label_dc.SetFont(ui.grid->GetLabelFont());
    CHECK(ui.grid->GetColSize(1) >= label_dc.GetTextExtent("step").x + ui.grid->FromDIP(24));
    constexpr const char* kCaptureDirEnv = "PRD105_SORT_CAPTURE_DIR";
    constexpr const char* kCapturePrefix = "sort-header-";
    const char* capture_directory = std::getenv(kCaptureDirEnv);
    int capture_number = 0;
    const auto pixels = [&] {
        auto* header = ui.grid->GetGridColLabelWindow();
        header->Refresh();
        header->Update();
        wxBitmap bitmap(header->GetClientSize());
        wxMemoryDC destination(bitmap);
        wxClientDC source(header);
        destination.Blit(wxPoint(0, 0), bitmap.GetSize(), &source, wxPoint(0, 0));
        destination.SelectObject(wxNullBitmap);
        const auto image = bitmap.ConvertToImage();
        if (capture_directory) {
            const std::filesystem::path directory(capture_directory);
            std::filesystem::create_directories(directory);
            const auto path = directory / (std::string(kCapturePrefix) + std::to_string(capture_number++) + ".png");
            CHECK(image.SaveFile(wxString::FromUTF8(path.string()), wxBITMAP_TYPE_PNG));
        }
        return std::vector<unsigned char>(image.GetData(), image.GetData() + image.GetWidth() * image.GetHeight() * 3);
    };
    CHECK(ui.grid->GetSortingColumn() == 1);
    CHECK_FALSE(ui.grid->IsSortOrderAscending());
    const auto descending = pixels();
    const auto highlight = wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT);
    const auto highlight_pixels = [&](const std::vector<unsigned char>& values) {
        size_t count = 0;
        for (size_t i = 0; i + 2 < values.size(); i += 3)
            if (values[i] == highlight.Red() && values[i + 1] == highlight.Green() && values[i + 2] == highlight.Blue()) ++count;
        return count;
    };
    CHECK(highlight_pixels(descending) > 0);
    ui.grid->UnsetSortingColumn();
    const auto unsorted = pixels();
    CHECK((unsorted != descending));
    CHECK(highlight_pixels(unsorted) == 0);
    ui.grid->SetSortingColumn(1, false);
    wxGridEvent ascending(ui.grid->GetId(), wxEVT_GRID_LABEL_LEFT_CLICK, ui.grid, -1, 1);
    ui.grid->GetEventHandler()->ProcessEvent(ascending);
    CHECK(ui.grid->IsSortOrderAscending());
    CHECK((pixels() != descending));
    wxGridEvent restored(ui.grid->GetId(), wxEVT_GRID_LABEL_LEFT_CLICK, ui.grid, -1, 1);
    ui.grid->GetEventHandler()->ProcessEvent(restored);
    CHECK((pixels() == descending));
}

TEST_CASE("Trace row retains its notification until eviction and dumps it on double click", "[prd105][trace_ui_dump]")
{
    anet::ConfigData config;
    config.Set("app.eval_trace_panel.max_rows", "2");
    TraceUiFixture ui(config);
    std::vector<std::weak_ptr<const anet::rl::BatchActionInfo>> actions;
    std::vector<std::weak_ptr<const anet::rl::BatchStepResult>> results;
    auto observer = std::make_shared<anet::rl::FunctionTrainObserver>([&](const anet::rl::TrainEvent& event) {
        actions.push_back(event.action_info);
        results.push_back(event.step_result);
    });
    ui.run->GetNotifier()->Attach(observer);
    ui.Add(7);
    REQUIRE(actions.size() == 1);
    REQUIRE_FALSE(actions.front().expired());
    REQUIRE_FALSE(results.front().expired());
    class CapturedLog final : public wxLog {
    public:
        std::vector<std::pair<wxLogLevel, wxString>> dumps;
    protected:
        void DoLogTextAtLevel(wxLogLevel level, const wxString& text) override
        {
            if (text.Contains("Trace dump step=")) dumps.emplace_back(level, text);
        }
    } log;
    struct RestoreLog {
        wxLog* previous;
        wxLogLevel level;
        bool verbose;
        ~RestoreLog() {
            wxLog::SetActiveTarget(previous);
            wxLog::SetLogLevel(level);
            wxLog::SetVerbose(verbose);
        }
    } restore{.previous=wxLog::SetActiveTarget(&log), .level=wxLog::GetLogLevel(), .verbose=wxLog::GetVerbose()};
    wxLog::SetLogLevel(wxLOG_Max);
    wxLog::SetVerbose(true);
    // 後続の異なる遷移を保持してから、過去の行を公開イベントで繰り返しダンプする。
    ui.Add(8);
    REQUIRE_FALSE(actions.front().expired());
    REQUIRE_FALSE(results.front().expired());
    for (int count = 1; count <= 2; ++count) {
        wxGridEvent event(ui.grid->GetId(), wxEVT_GRID_CELL_LEFT_DCLICK, ui.grid, 1, 0);
        ui.grid->GetEventHandler()->ProcessEvent(event);
        REQUIRE(log.dumps.size() == static_cast<size_t>(count * 2));
        CHECK(ui.panel->IsTraceMode());
    }
    CHECK(log.dumps[0].first == wxLOG_Message);
    CHECK(log.dumps[1].first == wxLOG_Info);
    CHECK(log.dumps[0].second.Contains("runner.forced=0"));
    CHECK(log.dumps[0].second.Contains("env=CartPoleEnv"));
    CHECK(log.dumps[0].second.Contains("Trace dump step=1 lane=0:"));
    CHECK(log.dumps[0].second.Contains("r=7"));
    CHECK(log.dumps[0].second.Contains("step_result.reward=7"));
    CHECK_FALSE(log.dumps[0].second.Contains("step_result.reward=8"));
    CHECK(log.dumps[1].second.Contains("BatchState{"));
    CHECK(log.dumps[1].second.Contains("BatchStepResult{"));
    CHECK(log.dumps[0].second == log.dumps[2].second);
    CHECK(log.dumps[1].second == log.dumps[3].second);
    ui.Add(9);
    CHECK(actions.front().expired());
    CHECK(results.front().expired());
    CHECK_FALSE(actions.back().expired());
    ui.panel->SetActive(false);
    CHECK(actions.back().expired());
    CHECK(results.back().expired());
    ui.run->GetNotifier()->Detach(observer);
}

TEST_CASE("A percentile of step sorts by its displayed value without a default step marker", "[prd105][trace_ui_step_pct]")
{
    anet::ConfigData config;
    config.Set("app.eval_trace_panel.columns.[1]", "$runner exp_step display:pct label:rank");
    TraceUiFixture ui(config);
    ui.runner->DoStep();
    ui.runner->DoStep();
    ui.runner->DoStep();
    CHECK(ui.grid->GetSortingColumn() == wxNOT_FOUND);
    wxGridEvent first(ui.grid->GetId(), wxEVT_GRID_LABEL_LEFT_CLICK, ui.grid, -1, 0);
    ui.grid->GetEventHandler()->ProcessEvent(first);
    CHECK_FALSE(ui.grid->IsSortOrderAscending());
    CHECK(ui.grid->GetCellValue(0, 0) == "83.33");
    wxGridEvent second(ui.grid->GetId(), wxEVT_GRID_LABEL_LEFT_CLICK, ui.grid, -1, 0);
    ui.grid->GetEventHandler()->ProcessEvent(second);
    CHECK(ui.grid->IsSortOrderAscending());
    CHECK(ui.grid->GetCellValue(0, 0) == "50");
}

TEST_CASE("Visible Trace survives deferred frame destruction", "[prd094][trace_ui_close]")
{
    // 通知資源はフレームの外で生存させ、実フレームの所有順序だけを検証する。
    anet::ConfigData config;
    config.Set("app.eval_trace_panel.columns.[1]", "$env_step reward label:r");
    TraceUiFixture source(config);
    auto frame = std::make_unique<TraceCloseFrameFixture>();
    auto& manager = frame->Manager();
    auto* eval = new EvalPanel(frame.get(), EvalPanelConfig{}, [] {});
    auto* panel = new EvalTracePanel(*frame, manager, *eval, config);
    auto* follower = new TraceCloseFollowerProbe(frame.get());
    manager.AddPane(panel, wxAuiPaneInfo().CenterPane().Name("trace"));
    manager.AddPane(follower, wxAuiPaneInfo().Right().Name("probe").Caption("Probe"));
    panel->RegisterFollower(follower);
    panel->Initialize(source.run, source.runner);
    manager.Update();
    frame->Show();
    wxTheApp->Yield(true);
    source.Add(1);
    wxGrid* grid = nullptr;
    for (auto* child : panel->GetChildren()) if (auto* candidate = dynamic_cast<wxGrid*>(child)) grid = candidate;
    REQUIRE(grid != nullptr);
    REQUIRE(grid->GetNumberRows() == 1);
    wxGridEvent selected(grid->GetId(), wxEVT_GRID_SELECT_CELL, grid, 0, 0);
    grid->GetEventHandler()->ProcessEvent(selected);
    REQUIRE(follower->show_calls > 0);

    // フレーム外に観測値を保存する。破棄済みpanelやfollowerは読み返さない。
    const int live_before = follower->live_calls;
    int live_at_destroy = -1;
    bool panel_destroyed = false;
    panel->Bind(wxEVT_DESTROY, [&](wxWindowDestroyEvent& event) {
        if (event.GetEventObject() == panel) {
            panel_destroyed = true;
            CHECK(panel->IsBeingDeleted());
        }
        event.Skip();
    });
    follower->Bind(wxEVT_DESTROY, [&](wxWindowDestroyEvent& event) {
        if (event.GetEventObject() == follower) live_at_destroy = follower->live_calls;
        event.Skip();
    });
    const wxWeakRef<TraceCloseFrameFixture> weak(frame.get());
    REQUIRE(frame->Destroy());
    frame.release(); // Destroy以降はwxの保留削除が所有する。
    for (int i = 0; weak && i < 20; ++i) {
        wxTheApp->ProcessIdle();
        wxTheApp->Yield(true);
    }
    CHECK_FALSE(weak);
    CHECK(panel_destroyed);
    CHECK(live_at_destroy == live_before);
}

TEST_CASE("Evaluation Trace presents six native playback tools", "[prd094][trace_ui_controls]")
{
    // 実際のwxの子ウィンドウと公開toolbar APIから操作列を確認する。
    wxFrame frame(nullptr, wxID_ANY, "Evaluation Trace acceptance");
    wxAuiManager manager(&frame);
    EvalPanel eval(&frame, EvalPanelConfig{}, [] {});
    EvalTracePanel panel(frame, manager, eval, anet::ConfigData{});
    wxAuiToolBar* toolbar = nullptr;
    for (auto* child : panel.GetChildren()) {
        for (auto* nested : child->GetChildren()) {
            if (auto* candidate = dynamic_cast<wxAuiToolBar*>(nested)) toolbar = candidate;
        }
    }
    REQUIRE(toolbar != nullptr);
    REQUIRE(toolbar->GetToolCount() == 6);
    const std::array<const char*, 6> help{"Play", "Stop and return to Live", "Step back", "Step forward", "Slower", "Faster"};
    for (size_t i = 0; i < help.size(); ++i) {
        const auto* tool = toolbar->FindToolByIndex(i);
        REQUIRE(tool != nullptr);
        CHECK(tool->GetShortHelp() == help[i]);
        CHECK_FALSE(toolbar->GetToolEnabled(tool->GetId()));
        CHECK(tool->GetBitmapFor(toolbar).IsOk());
    }
    manager.UnInit();
}

TEST_CASE("Trace grid fits headers and grows only for longer displayed values", "[prd094][trace_ui_width]")
{
    TraceUiFixture ui;
    REQUIRE(ui.grid != nullptr);
    wxClientDC dc(ui.grid);
    dc.SetFont(ui.grid->GetLabelFont());
    const int initial = ui.grid->GetColSize(0);
    CHECK(initial >= dc.GetTextExtent("r").x + ui.grid->FromDIP(6));
    CHECK(initial < ui.grid->FromDIP(50));
    ui.Add(-123456789.0f);
    dc.SetFont(ui.grid->GetDefaultCellFont());
    const int wide = ui.grid->GetColSize(0);
    CHECK(wide >= dc.GetTextExtent(ui.grid->GetCellValue(0, 0)).x + ui.grid->FromDIP(6));
    ui.Add(1.0f);
    CHECK(ui.grid->GetColSize(0) == wide);
}
TEST_CASE("Trace grid preserves manual widths and resets them on reopen", "[prd094][trace_ui_manual]")
{
    anet::ConfigData config;
    config.Set("app.eval_trace_panel.max_rows", "1");
    TraceUiFixture ui(config);
    const int initial = ui.grid->GetColSize(0);
    ui.Add(-123456789.0f);
    const int wide = ui.grid->GetColSize(0);
    ui.Add(1.0f);
    CHECK(ui.grid->GetColSize(0) == wide);
    ui.grid->SetColSize(0, ui.grid->FromDIP(20));
    wxGridSizeEvent resized(ui.grid->GetId(), wxEVT_GRID_COL_SIZE, ui.grid, 0);
    ui.grid->GetEventHandler()->ProcessEvent(resized);
    const int manual = ui.grid->GetColSize(0);
    ui.Add(-2000000000.0f);
    CHECK(ui.grid->GetColSize(0) == manual);
    ui.panel->SetActive(false);
    ui.panel->SetActive(true);
    CHECK(ui.grid->GetNumberRows() == 0);
    CHECK(ui.grid->GetColSize(0) == initial);
}

TEST_CASE("Trace grid expands the retained row when delayed diagnostics resolve", "[prd094][trace_ui_late]")
{
    anet::ConfigData config;
    config.Set("app.eval_trace_panel.columns.[1]", "$env_step reward label:r");
    config.Set("app.eval_trace_panel.columns.[2]", "$agent_step later label:p");
    TraceUiFixture ui(config);
    const int initial = ui.grid->GetColSize(1);
    ui.Add(0);
    CHECK(ui.grid->GetCellValue(0, 1).empty());
    ui.Add(1);
    CHECK(ui.grid->GetCellValue(1, 1) == "2000000000");
    CHECK(ui.grid->GetColSize(1) > initial);
    wxClientDC dc(ui.grid);
    dc.SetFont(ui.grid->GetDefaultCellFont());
    CHECK(ui.grid->GetColSize(1) >= dc.GetTextExtent("2000000000").x + ui.grid->FromDIP(6));
}

TEST_CASE("Trace toolbar toggles playback and enters Live-relative frames", "[prd094][trace_ui_toggle]")
{
    TraceUiFixture ui;
    ui.Add(1); ui.Add(2); ui.Add(3);
    // 速度も実際のtoolイベントから変更し、表示との同期を確認する。
    wxStaticText* speed = nullptr;
    for (auto* child : ui.toolbar->GetParent()->GetChildren())
        if (auto* candidate = dynamic_cast<wxStaticText*>(child)) speed = candidate;
    REQUIRE(speed != nullptr);
    ui.Tool(5);
    CHECK(speed->GetLabel() == "x2");
    ui.Tool(4);
    CHECK(speed->GetLabel() == "x1");
    ui.Tool(4);
    CHECK(speed->GetLabel() == "x0.5");
    ui.Tool(5);
    ui.Tool(0);
    CHECK(ui.toolbar->FindToolByIndex(0)->GetShortHelp() == "Pause");
    CHECK(ui.grid->GetGridCursorRow() == 2);
    CHECK(ui.eval->IsPaused());
    ui.Tool(0);
    CHECK(ui.toolbar->FindToolByIndex(0)->GetShortHelp() == "Play");
    ui.Tool(3);
    CHECK(ui.grid->GetGridCursorRow() == 1);
    ui.Tool(1);
    CHECK_FALSE(ui.grid->IsSelection());
    ui.Tool(2);
    CHECK(ui.grid->GetGridCursorRow() == 1);
    ui.Tool(1);
    ui.Tool(3);
    CHECK(ui.grid->GetGridCursorRow() == 0);
    ui.Tool(0);
    ui.eval->SetFps(0);
    CHECK(ui.toolbar->GetToolEnabled(ui.toolbar->FindToolByIndex(0)->GetId()));
    ui.Tool(0);
    CHECK(ui.toolbar->FindToolByIndex(0)->GetShortHelp() == "Play");
    CHECK_FALSE(ui.toolbar->GetToolEnabled(ui.toolbar->FindToolByIndex(0)->GetId()));
}

TEST_CASE("Trace minimum width keeps playback controls and the action column visible", "[prd094][trace_ui_minimum]")
{
    anet::ConfigData config;
    config.Set("app.eval_trace_panel.columns.[1]", "$runner exp_step label:step");
    config.Set("app.eval_trace_panel.columns.[2]", "$runner step_in_episode label:ep_step");
    config.Set("app.eval_trace_panel.columns.[3]", "$action_info action");
    config.Set("app.eval_trace_panel.columns.[4]", "$env_step reward");
    TraceUiFixture ui(config);
    ui.Add(1);
    auto* controls = ui.toolbar->GetParent();
    const int action_edge = ui.grid->GetColSize(0) + ui.grid->GetColSize(1) + ui.grid->GetColSize(2);
    REQUIRE(ui.panel->GetMinSize().x >= controls->GetSizer()->CalcMin().x);
    CHECK(ui.panel->GetMinSize().x >= action_edge + wxSystemSettings::GetMetric(wxSYS_VSCROLL_X, ui.grid));
    CHECK(ui.manager.GetPane(ui.panel).min_size.x >= ui.panel->GetMinSize().x);

    // 実際のdockを最小幅へ縮め、操作列と行動列の右端が切れないことを確認する。
    auto* center = new wxPanel(&ui.frame);
    ui.manager.GetPane(ui.panel).Right().BestSize(1, 300);
    ui.manager.AddPane(center, wxAuiPaneInfo().CenterPane().Name("minimum-center"));
    ui.frame.SetClientSize(wxSize(700, 450));
    ui.frame.Show();
    ui.manager.Update();
    wxTheApp->Yield(true);
    CHECK(ui.panel->GetClientSize().x >= ui.panel->GetMinSize().x);
    CHECK(ui.toolbar->GetRect().GetRight() < controls->GetClientSize().x);
    CHECK(action_edge <= ui.grid->GetGridWindow()->GetClientSize().x);

    // 手動で幅を広げたaction列とフォント変更も、最小幅へ反映される。
    const int before = ui.panel->GetMinSize().x;
    ui.grid->SetColSize(2, ui.grid->GetColSize(2) + before);
    wxGridSizeEvent resized(ui.grid->GetId(), wxEVT_GRID_COL_SIZE, ui.grid, 2);
    ui.grid->GetEventHandler()->ProcessEvent(resized);
    CHECK(ui.panel->GetMinSize().x > before);
    const int manual = ui.panel->GetMinSize().x;
    auto font = ui.panel->GetFont();
    font.SetPointSize(font.GetPointSize() + 4);
    ui.panel->SetFont(font);
    wxTheApp->Yield(true);
    CHECK(ui.panel->GetMinSize().x >= manual);
    ui.panel->SetActive(false);
    ui.panel->SetActive(true);
    CHECK(ui.panel->GetMinSize().x < manual);
    // フロート時もnativeの最小幅制約を通し、最後にdockへ戻して後始末する。
    ui.manager.GetPane(ui.panel).Float().FloatingSize(1, 350);
    ui.manager.Update();
    wxTheApp->Yield(true);
    CHECK(ui.panel->GetClientSize().x >= ui.panel->GetMinSize().x);
    ui.manager.GetPane(ui.panel).Dock().Right();
    ui.manager.Update();
    wxTheApp->Yield(true);
}

TEST_CASE("Trace without an action column reserves only the playback controls", "[prd094][trace_ui_minimum]")
{
    TraceUiFixture ui;
    const int before = ui.panel->GetMinSize().x;
    REQUIRE(before >= ui.toolbar->GetParent()->GetSizer()->CalcMin().x);
    ui.grid->SetColSize(0, before + 200);
    wxGridSizeEvent resized(ui.grid->GetId(), wxEVT_GRID_COL_SIZE, ui.grid, 0);
    ui.grid->GetEventHandler()->ProcessEvent(resized);
    CHECK(ui.panel->GetMinSize().x == before);
}

TEST_CASE("Trace selection preserves the column header appearance", "[prd094][trace_ui_header]")
{
    anet::ConfigData config;
    config.Set("app.eval_trace_panel.columns.[1]", "$env_step reward label:reward");
    config.Set("app.eval_trace_panel.columns.[2]", "$env_step truncated label:trunc");
    TraceUiFixture ui(config);
    ui.Add(0); ui.Add(1);
    ui.frame.Show();
    ui.frame.Raise();
    wxTheApp->Yield(true);
    const auto capture = [&] {
        auto* header = ui.grid->GetGridColLabelWindow();
        header->Refresh();
        header->Update();
        wxBitmap bitmap(header->GetClientSize());
        wxMemoryDC target(bitmap);
        wxClientDC source(header);
        target.Blit(wxPoint(0, 0), bitmap.GetSize(), &source, wxPoint(0, 0));
        target.SelectObject(wxNullBitmap);
        return bitmap.ConvertToImage();
    };
    const auto live = capture();
    ui.Tool(2);
    wxTheApp->Yield(true);
    REQUIRE(ui.grid->IsSelection());
    CHECK(ui.grid->GetColLabelValue(1) == "trunc");
    for (int col = 0; col < ui.grid->GetNumberCols(); ++col) CHECK_FALSE(ui.grid->IsColLabelHighlighted(col));
    const auto trace = capture();
    REQUIRE(live.GetSize() == trace.GetSize());
    const size_t bytes = static_cast<size_t>(live.GetWidth()) * live.GetHeight() * 3;
    CHECK(std::equal(live.GetData(), live.GetData() + bytes, trace.GetData()));
    CHECK(ui.grid->GetSelectionBackground() == wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT));
    CHECK(ui.grid->GetSelectionForeground() == wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHTTEXT));
}

TEST_CASE("Trace grid uses native theme colours and terminal tint precedence", "[prd094][trace_ui_palette]")
{
    TraceUiFixture ui;
    CHECK(ui.grid->GetDefaultCellBackgroundColour() == wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOW));
    CHECK(ui.grid->GetDefaultCellTextColour() == wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOWTEXT));
    CHECK(ui.grid->GetLabelBackgroundColour() == wxSystemSettings::GetColour(wxSYS_COLOUR_BTNFACE));
    CHECK(ui.grid->GetLabelTextColour() == wxSystemSettings::GetColour(wxSYS_COLOUR_BTNTEXT));
    CHECK(ui.grid->GetGridLineColour() == wxSystemSettings::GetColour(wxSYS_COLOUR_3DSHADOW));
    CHECK(ui.grid->GetSelectionBackground() == wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT));
    CHECK(ui.grid->GetSelectionForeground() == wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHTTEXT));
    ui.Add(0); ui.Add(1); ui.Add(0, true); ui.Add(1, false, true);
    CHECK(ui.grid->GetCellBackgroundColour(0, 0) == ui.grid->GetCellBackgroundColour(1, 0));
    CHECK(ui.grid->GetCellBackgroundColour(2, 0) != ui.grid->GetCellBackgroundColour(1, 0));
    for (int row = 0; row < 4; ++row)
        CHECK(ui.grid->GetCellTextColour(row, 0) == wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOWTEXT));
}

TEST_CASE("Trace grid follows a font change without overriding a manual column", "[prd094][trace_ui_font]")
{
    TraceUiFixture ui;
    ui.Add(-123456789.0f);
    const int original = ui.grid->GetColSize(0);
    auto font = ui.panel->GetFont();
    font.SetPointSize(font.GetPointSize() * 2);
    ui.panel->SetFont(font);
    wxTheApp->ProcessPendingEvents();
    CHECK(ui.grid->GetColSize(0) > original);
    ui.grid->SetColSize(0, 25);
    wxGridSizeEvent resized(ui.grid->GetId(), wxEVT_GRID_COL_SIZE, ui.grid, 0);
    ui.grid->GetEventHandler()->ProcessEvent(resized);
    font.SetPointSize(font.GetPointSize() + 2);
    ui.panel->SetFont(font);
    wxTheApp->ProcessPendingEvents();
    CHECK(ui.grid->GetColSize(0) == 25);
}

TEST_CASE("Trace UI twelve columns and two delayed values measurement", "[.][prd094][trace_ui_perf]")
{
    anet::ConfigData config;
    for (int i = 1; i <= 10; ++i)
        config.Set("app.eval_trace_panel.columns.[" + std::to_string(i) + "]", "$env_step reward label:c" + std::to_string(i));
    config.Set("app.eval_trace_panel.columns.[11]", "$agent_step later label:p");
    config.Set("app.eval_trace_panel.columns.[12]", "$agent_step later display:pct label:pct");
    TraceUiFixture ui(config);
    std::vector<double> samples;
    for (int i = 0; i < 2000; ++i) {
        ui.Add(i % 64 == 0 ? -2000000000.0f : static_cast<float>(i));
        if (i >= 1000) samples.push_back(ui.last_notify_ms);
    }
    std::sort(samples.begin(), samples.end());
    const double median = samples[samples.size() / 2];
    const double p95 = samples[samples.size() * 95 / 100];
    std::cout << "TRACE_UI_PERF C=12 K=2 rows=1000 samples=" << samples.size()
        << " median_ms=" << median << " p95_ms=" << p95
        << " budget_ms=" << 1000.0 / 14.5 * 0.025 << "\n";
    CHECK(ui.grid->GetNumberRows() == 1000);
}

TEST_CASE("Trace control bar renders a stable border across playback Live and theme changes", "[.][prd094][trace_ui_visual]")
{
    anet::ConfigData config;
    const std::array<const char*, 12> definitions{
        "$env_step reward label:step", "$env_step reward label:ep_step", "$action_info action",
        "$env_step reward label:Q_sa", "$env_step reward label:V_s", "$env_step reward",
        "$env_step done", "$env_step truncated label:trunc", "$agent_step later label:prio",
        "$agent_step later display:pct label:prio_pct", "$env_step reward label:score", "$env_step reward label:lives"};
    for (size_t col = 0; col < definitions.size(); ++col)
        config.Set("app.eval_trace_panel.columns.[" + std::to_string(col + 1) + "]", definitions[col]);
    TraceUiFixture ui(config);
    ui.Add(0); ui.Add(1); ui.Add(0, true); ui.Add(1, false, true);
    ui.frame.Show();
    ui.frame.Raise();
    wxTheApp->Yield(true);
    const char* capture_directory = std::getenv("PRD094_UI_CAPTURE_DIR");
    const auto directory = capture_directory ? std::filesystem::path(capture_directory) : TraceUiOutputDirectory() / "captures";
    std::filesystem::create_directories(directory);
    std::cout << "TRACE_UI_CAPTURE_DIR " << directory.string() << '\n';
    const std::string theme = std::getenv("PRD094_UI_DARK") ? "dark" : "light";
    const auto capture = [&](const std::string& name) {
        ui.panel->Refresh();
        ui.panel->Update();
        wxTheApp->Yield(true);
        wxBitmap bitmap(ui.panel->GetClientSize());
        wxMemoryDC destination(bitmap);
        wxClientDC source(ui.panel);
        destination.Blit(wxPoint(0, 0), bitmap.GetSize(), &source, wxPoint(0, 0));
        destination.SelectObject(wxNullBitmap);
        CHECK(bitmap.SaveFile(wxString::FromUTF8((directory / (theme + "-" + name + ".png")).string()), wxBITMAP_TYPE_PNG));
        const auto image = bitmap.ConvertToImage();
        int red = 0;
        const int height = ui.toolbar->GetParent()->GetSize().y;
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < image.GetWidth(); ++x)
                if (image.GetRed(x, y) > 180 && image.GetGreen(x, y) < 100 && image.GetBlue(x, y) < 100) ++red;
        return red;
    };
    const auto rect = ui.toolbar->GetRect();
    CHECK(capture("live") == 0);
    ui.Tool(2);
    CHECK(capture("trace") > 0);
    CHECK(ui.toolbar->GetRect() == rect);
    ui.Tool(0);
    CHECK(ui.toolbar->FindToolByIndex(0)->GetShortHelp() == "Pause");
    ui.Tool(0);
    CHECK(capture("paused") > 0);
    auto* slider = dynamic_cast<wxSlider*>(ui.toolbar->GetParent()->GetChildren().Item(1)->GetData());
    REQUIRE(slider != nullptr);
    slider->SetValue(1);
    wxCommandEvent moved(wxEVT_SLIDER, slider->GetId());
    slider->GetEventHandler()->ProcessEvent(moved);
    CHECK(ui.grid->GetGridCursorRow() == 2);
    ui.Tool(1);
    CHECK(capture("stopped") == 0);
    ui.Tool(0);
    // 実イベントループでTimerを配送し、描画用Yieldだけに依存しない。
    wxEventLoop loop;
    wxEventLoopActivator active(&loop);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (std::chrono::steady_clock::now() < deadline) {
        while (loop.Pending()) loop.Dispatch();
        wxTheApp->ProcessPendingEvents();
        wxMilliSleep(2);
    }
    CHECK(ui.toolbar->FindToolByIndex(0)->GetShortHelp() == "Play");
    CHECK(capture("ended") > 0);
    wxKeyEvent escape(wxEVT_CHAR_HOOK);
    escape.m_keyCode = WXK_ESCAPE;
    ui.grid->GetEventHandler()->ProcessEvent(escape);
    CHECK(capture("escape") == 0);
    ui.Tool(2);
    // wxMSWは既存windowへのSetAppearanceを拒否するため、各テーマは別processで確認する。
    // 開いたままの再反映は、通常のsystem-colour通知経路を通す。
    wxSysColourChangedEvent changed;
    ui.panel->GetEventHandler()->ProcessEvent(changed);
    CHECK(capture("theme-notified") > 0);
    CHECK(ui.grid->GetCellTextColour(0, 0) == wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOWTEXT));
    // 狭いpaneでも列幅を縮めず、横スクロールで全列へ到達できる。
    std::vector<int> widths;
    for (int col = 0; col < ui.grid->GetNumberCols(); ++col) widths.push_back(ui.grid->GetColSize(col));
    ui.frame.SetClientSize(ui.frame.FromDIP(wxSize(400, 350)));
    ui.manager.Update();
    wxTheApp->Yield(true);
    for (int col = 0; col < ui.grid->GetNumberCols(); ++col) CHECK(ui.grid->GetColSize(col) == widths[col]);
    CHECK(ui.grid->GetGridWindow()->GetVirtualSize().x > ui.grid->GetGridWindow()->GetClientSize().x);
    CHECK(capture("narrow") > 0);
    ui.panel->SetActive(false);
    ui.panel->SetActive(true);
    CHECK(ui.grid->GetNumberRows() == 0);
    CHECK(capture("reopened") == 0);
}

TEST_CASE("Eval runtime steps preserve startup config and reject nonpositive values", "[eval_steps]")
{
    wxFrame frame(nullptr, wxID_ANY, "Eval steps");
    EvalPanelConfig config;
    config.step_per_frame = 7;
    config.auto_start = false;
    EvalPanel eval(&frame, config, [] { FAIL("Changing steps must not return to Live"); });
    CHECK(eval.GetStepsPerFrame() == 7);
    eval.SetFps(14.5f);
    eval.SetStepsPerFrame(4);
    CHECK(eval.GetPlaybackRate() == 58.0);
    CHECK(eval.GetConfig().step_per_frame == 7);
    CHECK(eval.IsPaused());
    for (const int invalid : {0, -1, std::numeric_limits<int>::min()}) {
        CHECK_THROWS(eval.SetStepsPerFrame(invalid));
        CHECK(eval.GetStepsPerFrame() == 4);
        config.step_per_frame = invalid;
        CHECK_THROWS(config.Validate());
    }
    eval.SetStepsPerFrame(std::numeric_limits<int>::max());
    CHECK(eval.GetPlaybackRate() == 14.5 * std::numeric_limits<int>::max());
    eval.SetFps(0);
    config.step_per_frame = std::numeric_limits<int>::max();
    CHECK_NOTHROW(config.Validate());
}

TEST_CASE("Changing Eval steps retains the selected Trace row", "[eval_steps]")
{
    TraceUiFixture ui;
    ui.Add(1); ui.Add(2);
    ui.Tool(2);
    const int selected = ui.grid->GetGridCursorRow();
    REQUIRE(ui.eval->IsPaused());
    REQUIRE(ui.grid->IsSelection());
    ui.eval->SetStepsPerFrame(20);
    CHECK(ui.eval->GetPlaybackRate() == 290.0);
    CHECK(ui.eval->IsPaused());
    CHECK(ui.grid->IsSelection());
    CHECK(ui.grid->GetGridCursorRow() == selected);
    ui.eval->SetFps(0);
}

// RunnerAppの一度だけの起動を含むため、通常のTrace fixtureとは別processで実行する。
TEST_CASE("Runner menu changes Eval steps and preserves Custom cancellation", "[.][eval_steps_menu]")
{
    const auto directory = TraceUiOutputDirectory();
    std::filesystem::create_directories(directory);
    const auto config_path = directory / "eval_steps.txt";
    {
        std::ofstream config(config_path);
        config << "env.class_id = CartPoleEnv\nenv.worker_type = 1\nenv.device = cpu\n"
            "agent.class_id = TraceUiAgent\nagent.device = cpu\nrun.eval_device = cpu\n"
            "run.seed = 1\nrun.train.num_envs = 1\nrun.train.runner_type = serial\n"
            "run.eval.[panel].run_mode = eval\napp.eval_panel.eval_config_tag = panel\n"
            "app.train_auto_start = false\napp.save_agent_on_close = false\n"
            "app.train_panel.fps = 0\napp.eval_panel.fps = 0\n"
            "app.eval_panel.auto_start = false\napp.eval_panel.step_per_frame = 7\n"
            "app.run_name = eval_steps_menu\napp.runs_dir = " << directory.generic_string() << '\n';
    }
    auto executable = wxTheApp->argv[0].ToStdWstring();
    auto config_arg = config_path.wstring();
    wchar_t option[] = L"--config";
    wchar_t* arguments[] = {executable.data(), option, config_arg.data()};
    wxTheApp->argc = 3;
    wxTheApp->argv.Init(3, arguments);
    wxTheApp->SetExitOnFrameDelete(false);
    struct AppCleanup {
        ~AppCleanup() {
            if (auto* frame = wxGetApp().GetMainFrame()) frame->Close(true);
            wxTheApp->Yield(true);
            wxTheApp->OnExit();
        }
    } cleanup;
    REQUIRE(wxTheApp->OnInit());
    auto* frame = wxGetApp().GetMainFrame();
    // 初期化前の別Frameでは、ConfigとCustomを含む全項目が操作不可となる。
    {
        RunnerFrame uninitialized("Uninitialized Eval menu", TrainPanelConfig{}, EvalPanelConfig{});
        auto* bar = uninitialized.GetMenuBar();
        auto* menu = bar->FindItem(bar->FindMenuItem("View", "Eval Steps per Frame"))->GetSubMenu();
        for (auto* item : menu->GetMenuItems()) {
            wxUpdateUIEvent update(item->GetId());
            uninitialized.GetEventHandler()->ProcessEvent(update);
            CHECK_FALSE(update.GetEnabled());
        }
    }
    EvalPanel* eval = nullptr;
    for (auto* child : frame->GetChildren())
        if (auto* candidate = dynamic_cast<EvalPanel*>(child)) eval = candidate;
    REQUIRE(eval != nullptr);
    auto* bar = frame->GetMenuBar();
    const int menu_id = bar->FindMenuItem("View", "Eval Steps per Frame");
    REQUIRE(menu_id != wxNOT_FOUND);
    auto* menu = bar->FindItem(menu_id)->GetSubMenu();
    REQUIRE(menu != nullptr);
    CHECK(menu->GetMenuItemCount() == 9);
    auto* config_item = menu->FindItemByPosition(0);
    auto* custom_item = menu->FindItemByPosition(8);
    CHECK(config_item->GetItemLabelText() == "Config (7)");
    CHECK(config_item->IsChecked());
    const auto select = [&](wxMenuItem* item) {
        // nativeのradio切替を再現してから、公開のmenuイベントを配送する。
        item->Check(true);
        wxCommandEvent event(wxEVT_MENU, item->GetId());
        frame->GetEventHandler()->ProcessEvent(event);
    };
    for (size_t i = 0; const int value : {1, 2, 4, 10, 20, 50, 100}) {
        auto* item = menu->FindItemByPosition(++i);
        CHECK(item->GetItemLabelText() == wxString::Format("%d", value));
        select(item);
        CHECK(eval->GetStepsPerFrame() == value);
        CHECK(item->IsChecked());
    }
    struct DialogAnswers final : wxModalDialogHook {
        std::vector<std::optional<wxString>> answers;
        std::vector<wxString> initial_values;
        int errors = 0;
        int Enter(wxDialog* dialog) override {
            if (auto* input = dynamic_cast<wxTextEntryDialog*>(dialog)) {
                initial_values.push_back(input->GetValue());
                if (answers.empty()) return wxID_CANCEL;
                auto answer = answers.front();
                answers.erase(answers.begin());
                if (!answer) return wxID_CANCEL;
                input->SetValue(*answer);
                return wxID_OK;
            }
            ++errors;
            return wxID_OK;
        }
    } dialogs;
    dialogs.Register();
    dialogs.answers = {"4"};
    select(custom_item);
    CHECK(dialogs.initial_values.back() == "100");
    CHECK(eval->GetStepsPerFrame() == 4);
    CHECK(custom_item->GetItemLabelText() == "Custom (4)...");
    CHECK(custom_item->IsChecked());
    dialogs.answers = {std::nullopt};
    select(custom_item);
    CHECK(custom_item->IsChecked());
    select(config_item);
    CHECK(eval->GetStepsPerFrame() == 7);
    dialogs.answers = {std::nullopt};
    select(custom_item);
    CHECK(config_item->IsChecked());
    CHECK(eval->GetStepsPerFrame() == 7);
    for (const auto* invalid : {"0", "-1", "2147483648", "999999999999999999999999", "abc", "1.5"}) {
        dialogs.answers = {wxString(invalid), std::nullopt};
        select(custom_item);
        CHECK(config_item->IsChecked());
        CHECK(eval->GetStepsPerFrame() == 7);
    }
    CHECK(dialogs.errors == 6);
    dialogs.answers = {"2147483647"};
    select(custom_item);
    CHECK(eval->GetStepsPerFrame() == std::numeric_limits<int>::max());
    CHECK(eval->IsPaused());
    CHECK(eval->GetConfig().step_per_frame == 7);
    CHECK(wxGetApp().GetConfigData().Get<int>("app.eval_panel.step_per_frame") == 7);
    select(menu->FindItemByPosition(3));
    eval->SetFps(14.5f);
    CHECK(eval->GetPlaybackRate() == 58.0);
    eval->SetFps(0);

    // UI timerの配送1回あたりのstep数と、pause・手動stepの独立性を通知で観測する。
    int steps = 0;
    auto notifier = wxGetApp().GetRunManager().GetNotifier();
    auto observer = notifier->Attach<anet::rl::FunctionTrainObserver>(
        [&steps](const anet::rl::TrainEvent& event) {
            if (event.runner->GetName() == "EvalPanel") ++steps;
        }, "EvalStepsAcceptance");
    eval->DoStep();
    CHECK(steps == 1);
    CHECK(eval->IsPaused());
    // 実timerの公開イベントからIDを得て、以降は同じ配送を1回ずつ行う。
    auto timer_id = std::make_shared<int>(wxID_ANY);
    eval->Bind(wxEVT_TIMER, [timer_id](wxTimerEvent& event) {
        *timer_id = event.GetId();
        event.Skip();
    });
    eval->SetFps(1000);
    {
        wxEventLoop loop;
        wxEventLoopActivator active(&loop);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (*timer_id == wxID_ANY && std::chrono::steady_clock::now() < deadline) {
            while (loop.Pending()) loop.Dispatch();
            wxTheApp->ProcessPendingEvents();
            wxMilliSleep(2);
        }
    }
    eval->SetFps(0);
    REQUIRE(*timer_id != wxID_ANY);
    eval->TogglePause();
    wxTimer timer(eval, *timer_id);
    wxTimerEvent tick(timer);
    eval->GetEventHandler()->ProcessEvent(tick);
    CHECK(steps == 5);
    select(menu->FindItemByPosition(2));
    eval->GetEventHandler()->ProcessEvent(tick);
    CHECK(steps == 7);
    eval->PauseForTrace();
    select(config_item);
    eval->GetEventHandler()->ProcessEvent(tick);
    CHECK(steps == 7);
    CHECK(eval->IsPaused());
    notifier->Detach(observer);
    dialogs.Unregister();
}

// RunnerAppの起動を含むため、このタグだけを別processで実行する。
TEST_CASE("Runner forwards Trace arrows without steps and reports successful forced actions", "[.][prd105_ui_keys]")
{
    const auto directory = TraceUiOutputDirectory();
    std::filesystem::create_directories(directory);
    constexpr const char* kRealConfigEnv = "PRD105_UI_CONFIG";
    const char* real_config = std::getenv(kRealConfigEnv);
    const auto config_path = real_config ? std::filesystem::path(real_config) : directory / "prd105_keys.txt";
    if (!real_config) {
        std::ofstream config(config_path);
        config << "env.class_id = CartPoleEnv\nenv.worker_type = 1\nenv.device = cpu\n"
            "agent.class_id = TraceUiAgent\nagent.device = cpu\nrun.eval_device = cpu\n"
            "run.seed = 1\nrun.train.num_envs = 1\nrun.train.runner_type = serial\n"
            "run.eval.[panel].run_mode = eval\napp.eval_panel.eval_config_tag = panel\n"
            "app.train_auto_start = false\napp.save_agent_on_close = false\napp.show_error_dialog = false\n"
            "app.train_panel.fps = 0\napp.eval_panel.fps = 0\napp.eval_panel.auto_start = false\n"
            "backend.torch_num_threads = 1\napp.run_name = run_{t}_prd105_keys\n"
            "app.eval_trace_panel.columns.[1] = $runner exp_step label:step\n"
            "app.eval_trace_panel.columns.[2] = $runner forced label:forced\n"
            "app.eval_trace_panel.columns.[3] = $action_info action\n"
            "app.runs_dir = " << directory.generic_string() << '\n';
    }
    auto executable = wxTheApp->argv[0].ToStdWstring();
    auto config_arg = config_path.wstring();
    wchar_t option[] = L"--config";
    wchar_t* arguments[] = {executable.data(), option, config_arg.data()};
    wxTheApp->argc = 3;
    wxTheApp->argv.Init(3, arguments);
    wxTheApp->SetExitOnFrameDelete(false);
    struct AppCleanup {
        ~AppCleanup() {
            if (auto* frame = wxGetApp().GetMainFrame()) frame->Close(true);
            wxTheApp->Yield(true);
            wxTheApp->OnExit();
        }
    } cleanup;
    REQUIRE(wxTheApp->OnInit());
    auto* frame = wxGetApp().GetMainFrame();
    EvalPanel* eval = nullptr;
    EvalTracePanel* panel = nullptr;
    TrainPanel* train = nullptr;
    for (auto* child : frame->GetChildren()) {
        if (auto* candidate = dynamic_cast<EvalPanel*>(child)) eval = candidate;
        if (auto* candidate = dynamic_cast<EvalTracePanel*>(child)) panel = candidate;
        if (auto* candidate = dynamic_cast<TrainPanel*>(child)) train = candidate;
    }
    REQUIRE(eval);
    REQUIRE(panel);
    REQUIRE(train);
    wxGrid* grid = nullptr;
    for (auto* child : panel->GetChildren())
        if (auto* candidate = dynamic_cast<wxGrid*>(child)) grid = candidate;
    REQUIRE(grid);
    const int trace_menu = frame->GetMenuBar()->FindMenuItem("View", "Evaluation Trace");
    REQUIRE(trace_menu != wxNOT_FOUND);
    const auto show_trace = [&](bool show) {
        wxCommandEvent event(wxEVT_MENU, trace_menu);
        event.SetInt(show ? 1 : 0);
        frame->GetEventHandler()->ProcessEvent(event);
    };
    show_trace(true);
    auto runner = wxGetApp().GetRunManager().GetEvalRunner("EvalPanel");
    for (int i = 0; i < 22; ++i) eval->DoStep();
    REQUIRE(grid->GetNumberRows() == 22);
    const auto select_first = [&] {
        wxGridEvent event(grid->GetId(), wxEVT_GRID_CELL_LEFT_CLICK, grid, 0, 0);
        grid->GetEventHandler()->ProcessEvent(event);
    };
    const auto key = [&](int code) {
        wxKeyEvent raw(wxEVT_KEY_DOWN);
        raw.m_keyCode = code;
        anet::rl::gui::ForwardedKeyEvent event;
        event.SetKeyEvent(raw);
        event.SetWindow(train);
        frame->GetEventHandler()->ProcessEvent(event);
    };
    select_first();
    REQUIRE(panel->IsTraceMode());
    const auto before = runner->GetCounts().exp_step;
    auto* manager = wxAuiManager::GetManager(frame);
    REQUIRE(manager);
    manager->MaximizePane(manager->GetPane(train));
    manager->Update();
    REQUIRE(panel->IsTraceMode());
    for (int i = 0; i < 20; ++i) key(WXK_DOWN);
    REQUIRE(grid->GetGridCursorRow() == 20);
    key(WXK_LEFT);
    key(WXK_RIGHT);
    CHECK(grid->GetGridCursorRow() == 20);
    CHECK(runner->GetCounts().exp_step == before);
    CHECK(grid->GetNumberRows() == 22);
    manager->RestorePane(manager->GetPane(train));
    manager->Update();
    for (int i = 0; i < 20; ++i) key(WXK_UP);
    CHECK(grid->GetGridCursorRow() == 0);
    CHECK(runner->GetCounts().exp_step == before);
    panel->ReturnToLive();
    key(WXK_UP);
    CHECK(runner->GetCounts().exp_step == before + 1);
    CHECK(grid->GetCellValue(0, 1) == "1");
    CHECK(frame->GetStatusBar()->GetStatusText(0).ToStdString()
        == "Forced action: " + grid->GetCellValue(0, 2).ToStdString());
    select_first();
    key(WXK_NUMPAD1);
    CHECK_FALSE(panel->IsTraceMode());
    CHECK(runner->GetCounts().exp_step == before + 2);
    CHECK(grid->GetCellValue(0, 1) == "1");
    CHECK(frame->GetStatusBar()->GetStatusText(0).ToStdString()
        == "Forced action: " + grid->GetCellValue(0, 2).ToStdString());
    select_first();
    key(WXK_CONTROL);
    CHECK_FALSE(panel->IsTraceMode());
    CHECK(runner->GetCounts().exp_step == before + 3);
    CHECK(grid->GetCellValue(0, 1) == "0");
    show_trace(false);
    key(WXK_UP);
    CHECK_FALSE(panel->IsTraceMode());
    CHECK(runner->GetCounts().exp_step == before + 4);
    CHECK(runner->GetScalar("forced", 0) == 1.0f);
    CHECK(frame->GetStatusBar()->GetStatusText(0).StartsWith("Forced action: 0:"));
    if (real_config) {
        // 実Env/Agentでも同じ公開操作を通し、ログ欄への反映までを測る。
        show_trace(true);
        for (int i = 0; i < 22; ++i) eval->DoStep();
        REQUIRE(grid->GetNumberRows() == 22);
        const auto normal_step = runner->GetCounts().exp_step;
        const auto header = [&](int column) {
            wxGridEvent event(grid->GetId(), wxEVT_GRID_LABEL_LEFT_CLICK, grid, -1, column);
            grid->GetEventHandler()->ProcessEvent(event);
        };
        header(1);
        CHECK_FALSE(grid->IsSortOrderAscending());
        header(1);
        CHECK(grid->IsSortOrderAscending());
        int priority = wxNOT_FOUND;
        for (int col = 0; col < grid->GetNumberCols(); ++col)
            if (grid->GetColLabelValue(col) == "prio") priority = col;
        REQUIRE(priority != wxNOT_FOUND);
        header(priority);
        CHECK_FALSE(grid->IsSortOrderAscending());
        header(priority);
        CHECK(grid->IsSortOrderAscending());
        wxAuiToolBar* trace_toolbar = nullptr;
        for (auto* child : panel->GetChildren())
            for (auto* nested : child->GetChildren())
                if (auto* candidate = dynamic_cast<wxAuiToolBar*>(nested)) trace_toolbar = candidate;
        REQUIRE(trace_toolbar);
        const auto tool = [&](size_t index) {
            wxCommandEvent event(wxEVT_TOOL, trace_toolbar->FindToolByIndex(index)->GetId());
            trace_toolbar->GetEventHandler()->ProcessEvent(event);
        };
        eval->SetFps(14.5f);
        tool(0);
        CHECK(grid->GetGridCursorRow() == grid->GetNumberRows() - 1);
        tool(3);
        CHECK(grid->GetGridCursorRow() == grid->GetNumberRows() - 2);
        CHECK(runner->GetCounts().exp_step == normal_step);
        tool(1);
        header(0);
        CHECK_FALSE(grid->IsSortOrderAscending());
        panel->ReturnToLive();
        key(WXK_DOWN);
        CHECK(runner->GetCounts().exp_step == normal_step + 1);
        CHECK(frame->GetStatusBar()->GetStatusText(0).StartsWith("Forced action: 2:"));
        CHECK(grid->GetCellValue(0, 1) == "1");
        select_first();
        key(WXK_NUMPAD2);
        CHECK_FALSE(panel->IsTraceMode());
        CHECK(runner->GetScalar("forced", 0) == 1.0f);
        // QValueのクリックも実際の行動handlerへ配送する。
        QValuePanel* qvalue = nullptr;
        LogPanel* log_panel = nullptr;
        for (auto* child : frame->GetChildren()) {
            if (auto* candidate = dynamic_cast<QValuePanel*>(child)) qvalue = candidate;
            if (auto* candidate = dynamic_cast<LogPanel*>(child)) log_panel = candidate;
        }
        REQUIRE(qvalue);
        manager->GetPane(qvalue).Show();
        manager->Update();
        wxTheApp->Yield(true);
        wxGrid* qgrid = nullptr;
        for (auto* child : qvalue->GetChildren())
            if (auto* candidate = dynamic_cast<wxGrid*>(child)) qgrid = candidate;
        REQUIRE(qgrid);
        REQUIRE(qgrid->GetNumberRows() > 2);
        select_first();
        const auto before_click = runner->GetCounts().exp_step;
        wxGridEvent action_click(qgrid->GetId(), wxEVT_GRID_CELL_LEFT_CLICK, qgrid, 2, 0);
        qgrid->GetEventHandler()->ProcessEvent(action_click);
        CHECK_FALSE(panel->IsTraceMode());
        CHECK(runner->GetCounts().exp_step == before_click + 1);
        CHECK(frame->GetStatusBar()->GetStatusText(0).StartsWith("Forced action: 2:"));
        REQUIRE(log_panel);
        wxTextCtrl* log_text = nullptr;
        for (auto* child : log_panel->GetChildren())
            if (auto* candidate = dynamic_cast<wxTextCtrl*>(child)) log_text = candidate;
        REQUIRE(log_text);
        // 1件目はinfoの表示だけ、2件目はverboseの表示までを確認する。
        for (const auto* level : {"info", "verbose"}) {
            log_panel->SetLogLevel(level);
            const auto start_length = log_text->GetLastPosition();
            const auto start = std::chrono::steady_clock::now();
            wxGridEvent dump(grid->GetId(), wxEVT_GRID_CELL_LEFT_DCLICK, grid, 0, 0);
            grid->GetEventHandler()->ProcessEvent(dump);
            wxLog::FlushActive();
            wxTheApp->Yield(true);
            const auto added = log_text->GetRange(start_length, log_text->GetLastPosition());
            CHECK(added.Contains("Trace dump step="));
            CHECK(added.Contains("action_info.action=2:"));
            CHECK(added.Contains(wxString::FromUTF8("env=" + wxGetApp().GetConfigData().Get("env.class_id"))));
            CHECK(added.Contains("verbose") == (std::string_view(level) == "verbose"));
            const double elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            std::cout << "PRD105_REAL_UI config=" << config_path.string() << " level=" << level
                << " dump_to_log_ms=" << elapsed << " chars=" << added.size() << '\n';
        }
        show_trace(false);
        show_trace(true);
        CHECK(grid->GetNumberRows() == 0);
        CHECK(grid->GetSortingColumn() == 0);
        CHECK_FALSE(grid->IsSortOrderAscending());
        show_trace(false);
        const std::array<int, 4> arrows{WXK_UP, WXK_LEFT, WXK_DOWN, WXK_RIGHT};
        for (size_t action = 0; action < arrows.size(); ++action) {
            const auto prior = runner->GetCounts().exp_step;
            key(arrows[action]);
            CHECK(runner->GetCounts().exp_step == prior + 1);
            CHECK(frame->GetStatusBar()->GetStatusText(0).StartsWith(
                wxString::FromUTF8("Forced action: " + std::to_string(action) + ":")));
        }
    }
}

int main(int argc, char** argv)
{
    anet::test::PreparedTestArgs arguments;
    try { arguments = anet::test::PrepareTestArgs(argc, argv); }
    catch (const std::exception& error) { return anet::test::ReportTestArgsError(error); }
    anet::test::SetupTestFailureDialog(arguments.failure_dialog_enabled);
    // RunnerAppの工場だけを利用し、OnInit（Run起動）は呼ばない。
    int wxargc = 1;
    char* wxargv[] = {argv[0], nullptr};
    if (!wxEntryStart(wxargc, wxargv)) return 2;
    wxSetAssertHandler([](const wxString&, int, const wxString&, const wxString& condition, const wxString& message) {
        throw std::runtime_error((condition + ": " + message).ToStdString());
    });
    wxTheApp->SetAppearance(std::getenv("PRD094_UI_DARK") ? wxApp::Appearance::Dark : wxApp::Appearance::Light);
    // RunnerAppを起動する明示tagは別process専用。Env登録はOnInit側へ任せる。
    bool starts_runner_app = false;
    for (int i = 1; i < arguments.Argc(); ++i) {
        const std::string_view argument(arguments.Argv()[i]);
        starts_runner_app |= argument.find("eval_steps_menu") != std::string_view::npos
            || argument.find("prd105_ui_keys") != std::string_view::npos;
    }
    if (!starts_runner_app) {
        wxInitAllImageHandlers();
        anet::rl::env::InitCartPole();
    }
    anet::rl::AgentRepository::Instance().Register(std::make_shared<TraceUiAgentFactory>());
    int result;
    {
        anet::test::StderrLogGuard log;
        result = Catch::Session().run(arguments.Argc(), arguments.Argv());
    }
    wxEntryCleanup();
    return result;
}
