#include "anet/catch_test.hpp"
#include "anet/test_util.hpp"
#include "EvalTracePanel.hpp"
#include <wx/app.h>

#include "anet/agent.hpp"
#include "anet/env/CartPole.hpp"
#include "anet/metrics_logger.hpp"
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
class TraceUiAgent final : public anet::rl::Agent {
public:
    std::unique_ptr<anet::rl::AgentStepDiagnosticsProvider> CreateStepDiagnosticsProvider(int64_t) const override
    { return std::make_unique<TraceUiDiagnosticsProvider>(); }
    std::shared_ptr<anet::rl::Actor> CreateActor(const anet::rl::ActorRequest&) const override { return nullptr; }
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
    wxInitAllImageHandlers();
    anet::rl::env::InitCartPole();
    anet::rl::AgentRepository::Instance().Register(std::make_shared<TraceUiAgentFactory>());
    int result;
    {
        anet::test::StderrLogGuard log;
        result = Catch::Session().run(arguments.Argc(), arguments.Argv());
    }
    wxEntryCleanup();
    return result;
}
