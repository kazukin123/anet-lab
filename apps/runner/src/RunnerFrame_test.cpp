#include "anet/catch_test.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <tuple>
#include <utility>
#include <vector>
#include <wx/aui/floatpane.h>
#include <wx/dcbuffer.h>
#include <wx/dcclient.h>
#include <wx/dcmemory.h>
#include <wx/evtloop.h>
#include <wx/grid.h>
#include <wx/settings.h>
#include <wx/sizer.h>
#include <wx/weakref.h>
#include "EvalTracePanel.hpp"
#include "RunnerApp.hpp"

namespace {

class TrainPanelProbe final : public TrainPanel {
public:
    using TrainPanel::TrainPanel;
    static const wxTimer& Timer(const TrainPanel& panel)
    {
        // protected のメンバポインタなら、実Frameが所有する元のpanelも安全に読める。
        return panel.*(&TrainPanelProbe::update_timer_);
    }
};

struct MainPaneFixture {
    wxEventLoop loop;
    wxEventLoopActivator activator{&loop};
    const std::filesystem::path output = std::filesystem::temp_directory_path()
        / ("anet-prd110-" + std::to_string(wxGetProcessId()));
    wxWeakRef<wxFrame> frame;

    void Pump()
    {
        // timer / UPDATE_UI / CallAfter を実際のイベントループで配送する。
        while (loop.Pending()) loop.Dispatch();
        wxTheApp->ProcessPendingEvents();
        loop.ProcessIdle();
    }

    void Start(const std::string& overrides = "")
    {
        std::filesystem::create_directories(output);
        const auto config_path = output / "main_pane.txt";
        {
            std::ofstream config(config_path);
            config << "env.class_id = CartPoleEnv\nenv.worker_type = 1\nenv.device = cpu\n"
                "agent.class_id = TraceUiAgent\nagent.device = cpu\nrun.eval_device = cpu\n"
                "run.seed = 1\nrun.train.num_envs = 1\nrun.train.runner_type = serial\n"
                "run.eval.[panel].run_mode = eval\napp.eval_panel.eval_config_tag = panel\n"
                "app.train_auto_start = false\napp.save_agent_on_close = false\n"
                "app.show_error_dialog = false\napp.train_panel.fps = 10\n"
                "app.eval_panel.fps = 0\napp.eval_panel.auto_start = false\n"
                "backend.torch_num_threads = 1\napp.run_name = main_pane\napp.runs_dir = "
                << output.generic_string() << '\n' << overrides;
        }
        auto executable = wxTheApp->argv[0].ToStdWstring();
        auto path = config_path.wstring();
        wchar_t option[] = L"--config";
        wchar_t* arguments[] = {executable.data(), option, path.data()};
        wxTheApp->argc = 3;
        wxTheApp->argv.Init(3, arguments);
        wxTheApp->SetExitOnFrameDelete(false);
        REQUIRE(wxTheApp->OnInit());
        frame = wxGetApp().GetMainFrame();
        REQUIRE(frame);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!wxGetApp().IsRunInitialized() && std::chrono::steady_clock::now() < deadline) {
            Pump();
            wxMilliSleep(2);
        }
        REQUIRE(wxGetApp().IsRunInitialized());
    }

    wxAuiManager& Manager() { return *wxAuiManager::GetManager(frame); }
    wxAuiPaneInfo& Train() { return Manager().GetPane("TrainExperiencePanel"); }
    wxAuiPaneInfo& Eval() { return Manager().GetPane("EvalExperiencePanel"); }
    TrainPanel& TrainView() { return *dynamic_cast<TrainPanel*>(Train().window); }

    int MenuId(const char* label) { return frame->GetMenuBar()->FindMenuItem("View", label); }

    void Toggle(const char* label, bool show, bool toolbar = false)
    {
        const int id = MenuId(label);
        REQUIRE(id != wxNOT_FOUND);
        wxCommandEvent event(toolbar ? wxEVT_TOOL : wxEVT_MENU, id);
        event.SetInt(show);
        if (toolbar) {
            auto* tools = dynamic_cast<wxAuiToolBar*>(Manager().GetPane("PanelToolBar").window);
            REQUIRE(tools);
            tools->GetEventHandler()->ProcessEvent(event);
        } else {
            frame->GetEventHandler()->ProcessEvent(event);
        }
        Settle();
    }

    void Settle()
    {
        // UI の宣言と、その結果の CallAfter を明示的に一往復させる。
        for (int pass = 0; pass < 2; ++pass) {
            // wxMSW の menu は明示更新、AUI toolbar は idle 更新で同期する。
            frame->UpdateWindowUI(wxUPDATE_UI_RECURSE);
            Manager().GetPane("PanelToolBar").window->UpdateWindowUI(wxUPDATE_UI_FROMIDLE);
            Pump();
        }
    }

    void CheckToggle(const char* label, bool checked, bool enabled)
    {
        const int id = MenuId(label);
        REQUIRE(id != wxNOT_FOUND);
        auto* menu = frame->GetMenuBar()->FindItem(id);
        CHECK(menu->IsChecked() == checked);
        CHECK(menu->IsEnabled() == enabled);
        auto* tools = dynamic_cast<wxAuiToolBar*>(Manager().GetPane("PanelToolBar").window);
        REQUIRE(tools);
        REQUIRE(tools->FindTool(id));
        CHECK(tools->GetToolToggled(id) == checked);
        CHECK(tools->GetToolEnabled(id) == enabled);
    }

    void Button(wxWindow* window, int button, bool settle = true)
    {
        // wx の close イベント生成、基底の Veto、実際の Hide までを通す。
        wxAuiManagerEvent event(wxEVT_AUI_PANE_BUTTON);
        event.SetManager(&Manager());
        event.SetPane(&Manager().GetPane(window));
        event.SetButton(button);
        frame->GetEventHandler()->ProcessEvent(event);
        if (settle) Settle();
    }

    void Close(wxWindow* window, bool settle = true) { Button(window, wxAUI_BUTTON_CLOSE, settle); }

    void Reset()
    {
        wxCommandEvent event(wxEVT_MENU, MenuId("Reset Layout"));
        frame->GetEventHandler()->ProcessEvent(event);
        Settle();
    }

    void Fps(const char* label)
    {
        auto* menu = frame->GetMenuBar()->FindItem(MenuId("Train View FPS"))->GetSubMenu();
        const int id = menu->FindItem(label);
        REQUIRE(id != wxNOT_FOUND);
        menu->FindItem(id)->Check(true);
        wxCommandEvent event(wxEVT_MENU, id);
        frame->GetEventHandler()->ProcessEvent(event);
        Settle();
    }

    int DockSize(const wxString& name)
    {
        class SizeReader final : public wxAuiSerializer {
        public:
            std::vector<wxAuiPaneLayoutInfo> panes;
            void SavePane(const wxAuiPaneLayoutInfo& pane) override { panes.push_back(pane); }
            void BeforeSaveNotebook(const wxString&) override {}
            void SaveNotebookTabControl(const wxAuiTabLayoutInfo&) override {}
        } reader;
        Manager().SaveLayout(reader);
        for (const auto& pane : reader.panes) if (pane.name == name) return pane.dock_size;
        return 0;
    }

    void SetDockSize(const wxString& name, int size)
    {
        class Reader final : public wxAuiSerializer {
        public:
            std::vector<wxAuiPaneLayoutInfo> panes;
            void SavePane(const wxAuiPaneLayoutInfo& pane) override { panes.push_back(pane); }
            void BeforeSaveNotebook(const wxString&) override {}
            void SaveNotebookTabControl(const wxAuiTabLayoutInfo&) override {}
        } reader;
        Manager().SaveLayout(reader);
        const auto side = Manager().GetPane(name);
        for (auto& pane : reader.panes) {
            if (pane.dock_direction == side.dock_direction && pane.dock_layer == side.dock_layer
                && pane.dock_row == side.dock_row) pane.dock_size = size;
        }
        class Writer final : public wxAuiDeserializer {
        public:
            Writer(wxAuiManager& manager, std::vector<wxAuiPaneLayoutInfo> data)
                : wxAuiDeserializer(manager), panes(std::move(data)) {}
            std::vector<wxAuiPaneLayoutInfo> LoadPanes() override { return std::move(panes); }
            std::vector<wxAuiTabLayoutInfo> LoadNotebookTabs(const wxString&) override { return {}; }
            std::vector<wxAuiPaneLayoutInfo> panes;
        } writer(Manager(), std::move(reader.panes));
        // 置き場を変えず幅・高さだけを準備する。サッシ入力の再現ではない。
        Manager().LoadLayout(writer);
    }

    int MainWidth(const wxString& side_name)
    {
        const auto& side = Manager().GetPane(side_name);
        const auto side_key = std::tuple{side.dock_direction, side.dock_layer, side.dock_row};
        std::vector<std::tuple<int, int, int>> columns;
        int width = frame->GetClientSize().x;
        // 主paneの列以外に見える縦dockを、公開の実サイズで一度ずつ差し引く。
        for (const auto& pane : Manager().GetAllPanes()) {
            if (!pane.IsShown() || !pane.IsDocked()
                || (pane.dock_direction != wxAUI_DOCK_LEFT && pane.dock_direction != wxAUI_DOCK_RIGHT)) continue;
            const auto key = std::tuple{pane.dock_direction, pane.dock_layer, pane.dock_row};
            if (key == side_key || std::find(columns.begin(), columns.end(), key) != columns.end()) continue;
            columns.push_back(key);
            width -= DockSize(pane.name);
        }
        return width;
    }

    void CheckMainSplit()
    {
        const auto vertical = [](const wxAuiPaneInfo& pane) {
            return pane.dock_direction == wxAUI_DOCK_LEFT || pane.dock_direction == wxAUI_DOCK_RIGHT;
        };
        REQUIRE(Train().IsShown());
        REQUIRE(Eval().IsShown());
        REQUIRE(Train().IsDocked());
        REQUIRE(Eval().IsDocked());
        REQUIRE(((Train().dock_direction == wxAUI_DOCK_CENTER && vertical(Eval()))
            || (Eval().dock_direction == wxAUI_DOCK_CENTER && vertical(Train()))));
        const auto name = vertical(Train()) ? Train().name : Eval().name;
        CHECK(std::abs(DockSize(name) - MainWidth(name) / 2) <= 2);
    }

    void CheckMainHeightSplit()
    {
        REQUIRE(Train().IsShown());
        REQUIRE(Eval().IsShown());
        REQUIRE(Train().IsDocked());
        REQUIRE(Eval().IsDocked());
        const auto& centre = Train().dock_direction == wxAUI_DOCK_CENTER ? Train() : Eval();
        const auto& band = Train().dock_direction == wxAUI_DOCK_CENTER ? Eval() : Train();
        REQUIRE(centre.dock_direction == wxAUI_DOCK_CENTER);
        REQUIRE((band.dock_direction == wxAUI_DOCK_TOP || band.dock_direction == wxAUI_DOCK_BOTTOM));
        const int height = DockSize(band.name);
        CHECK(std::abs(height - (centre.rect.height + height) / 2) <= 2);
        CHECK(std::abs(band.rect.width - centre.rect.width) <= 2);
    }

    void CheckMainCaptions()
    {
        for (const auto& [window, body] : {std::pair{Train().window, "Train View"},
            std::pair{Eval().window, "Evaluation View"}}) {
            const auto& pane = Manager().GetPane(window);
            const bool centre = pane.IsShown() && pane.IsDocked() && pane.dock_direction == wxAUI_DOCK_CENTER;
            CHECK(pane.caption == wxString(body) + (centre ? " [Centre]" : ""));
            if (pane.IsFloating() && pane.IsShown() && pane.frame) CHECK(pane.frame->GetLabel() == pane.caption);
        }
        for (const char* name : {"LogPanel", "EvalQValuePanel", "EvalTracePanel"}) {
            CHECK_FALSE(Manager().GetPane(name).caption.Contains("[Centre]"));
        }
    }

    std::string Log()
    {
        wxGetApp().FlushRunOutputs();
        std::ifstream stream(output / "main_pane/main_pane.log");
        return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    }

    void CheckNoMainPaneFallback()
    {
        const auto log = Log();
        CHECK(log.find("No main pane occupies the centre.") == std::string::npos);
        CHECK(log.find("Both main panes occupy the centre.") == std::string::npos);
    }

    ~MainPaneFixture()
    {
        if (frame) frame->Close(true);
        wxTheApp->Yield(true);
        wxTheApp->OnExit();
    }
};

} // namespace

TEST_CASE("Main panes start with only Eval occupying the centre", "[.][prd110_main_pane][prd110_default]")
{
    MainPaneFixture ui;
    ui.Start();
    CHECK_FALSE(ui.Train().IsShown());
    CHECK(ui.Eval().IsShown());
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Shown main panes keep the toolbar selection outline while disabled", "[.][prd110_main_pane][prd110_toolbar]")
{
    MainPaneFixture ui;
    ui.Start();
    ui.Settle();
    const auto check_outline = [&](const char* label) {
        // 実toolbarの状態と公開描画経路で、操作不可でも表示中の枠が残ることを読む。
        ui.CheckToggle(label, true, false);
        auto* tools = dynamic_cast<wxAuiToolBar*>(ui.Manager().GetPane("PanelToolBar").window);
        REQUIRE(tools);
        const int id = ui.MenuId(label);
        const wxRect rect(wxPoint(0, 0), tools->GetToolRect(id).GetSize());
        REQUIRE(rect.width > 2);
        REQUIRE(rect.height > 2);
        wxBitmap bitmap(rect.GetSize());
        wxMemoryDC dc(bitmap);
        tools->GetArtProvider()->DrawBackground(dc, tools, rect);
        tools->GetArtProvider()->DrawButton(dc, tools, *tools->FindTool(id), rect);
        dc.SelectObject(wxNullBitmap);
        const auto image = bitmap.ConvertToImage();
        const auto highlight = wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT);
        int highlighted = 0;
        for (int x = 0; x < rect.width; ++x) {
            for (const int y : {0, rect.height - 1}) {
                if (image.GetRed(x, y) == highlight.Red()
                    && image.GetGreen(x, y) == highlight.Green()
                    && image.GetBlue(x, y) == highlight.Blue()) ++highlighted;
            }
        }
        CHECK(highlighted >= rect.width);
    };
    check_outline("Evaluation View");
    ui.Toggle("Train View", true);
    ui.Toggle("Evaluation View", false);
    check_outline("Train View");
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Main panes share guarded toggles and restore the previous Eval dock", "[.][prd110_main_pane][prd110_transitions]")
{
    MainPaneFixture ui;
    ui.Start();
    ui.Settle();
    ui.CheckToggle("Train View", false, true);
    ui.CheckToggle("Evaluation View", true, false);
    CHECK_FALSE(ui.Eval().IsFloatable());
    CHECK_FALSE(ui.Eval().HasCloseButton());
    CHECK_FALSE(ui.Eval().HasPinButton());
    ui.Toggle("Evaluation View", false);
    CHECK(ui.Eval().IsShown());
    const auto both = [&] {
        CHECK(ui.Train().IsShown());
        CHECK(ui.Train().dock_direction == wxAUI_DOCK_CENTER);
        CHECK(ui.Train().IsFloatable());
        CHECK(ui.Train().HasCloseButton());
        CHECK(ui.Train().HasPinButton());
        CHECK(ui.Eval().IsShown());
        CHECK(ui.Eval().dock_direction == wxAUI_DOCK_RIGHT);
        CHECK(ui.Eval().dock_layer == 0);
        CHECK(ui.Eval().IsFloatable());
        CHECK(ui.Eval().HasCloseButton());
        CHECK(ui.Eval().HasPinButton());
        ui.CheckToggle("Train View", true, true);
        ui.CheckToggle("Evaluation View", true, true);
        CHECK(std::abs(ui.DockSize(ui.Eval().name) - ui.frame->GetClientSize().x / 2) <= 2);
    };
    ui.Toggle("Train View", true);
    both();
    ui.Toggle("Train View", false, true);
    CHECK_FALSE(ui.Train().IsShown());
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    CHECK_FALSE(ui.Eval().HasPinButton());
    ui.CheckToggle("Evaluation View", true, false);
    ui.Toggle("Train View", true);
    both();
    ui.Close(ui.Train().window);
    CHECK_FALSE(ui.Train().IsShown());
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    ui.Toggle("Train View", true);
    both();

    // 定位置以外の4値は、昇格前に wx が確定した値を保存して比較する。
    ui.Eval().Dock().Left().Layer(2).Row(1).Position(2);
    ui.Manager().Update();
    const auto before = ui.Eval();
    ui.Toggle("Train View", false);
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    ui.Toggle("Log View", false);
    ui.Toggle("Train View", true);
    CHECK(ui.Eval().dock_direction == before.dock_direction);
    CHECK(ui.Eval().dock_layer == before.dock_layer);
    CHECK(ui.Eval().dock_row == before.dock_row);
    CHECK(ui.Eval().dock_pos == before.dock_pos);
    ui.CheckMainSplit();

    // 昇格中は直前の記憶幅を保ち、降格後は新しい予算で等幅にする。
    const int promoted_width = ui.DockSize(ui.Eval().name);
    ui.Toggle("Train View", false);
    // Q-Valuesの最小幅（＝既定幅）より広げた列を、主paneの200を残すところまで縮める。
    const int q_min = ui.Manager().GetPane("EvalQValuePanel").min_size.x;
    ui.Manager().GetPane("EvalQValuePanel").dock_size = q_min + 50;
    ui.Toggle("Evaluation QValue View", true);
    CHECK(ui.DockSize("EvalQValuePanel") == q_min + 50);
    ui.frame->SetClientSize(wxSize(q_min + 220, ui.frame->GetClientSize().y));
    ui.Settle();
    CHECK(ui.DockSize("EvalQValuePanel") == q_min + 20);
    CHECK(ui.Eval().dock_size == promoted_width);
    ui.frame->SetClientSize(wxSize(1800, ui.frame->GetClientSize().y));
    ui.Settle();
    ui.Toggle("Train View", true);
    CHECK(ui.Eval().dock_direction == before.dock_direction);
    ui.CheckMainSplit();
    ui.Reset();
    CHECK_FALSE(ui.Train().IsShown());
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    ui.Toggle("Train View", true);
    both();
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Hidden Train stops its rendering timer and retains the selected FPS", "[.][prd110_main_pane][prd110_timer]")
{
    MainPaneFixture ui;
    ui.Start();
    const auto& timer = TrainPanelProbe::Timer(ui.TrainView());
    REQUIRE_FALSE(timer.IsRunning());
    ui.Toggle("Train View", true);
    CHECK(timer.IsRunning());
    CHECK(timer.GetInterval() == 100);
    ui.Toggle("Train View", false);
    CHECK_FALSE(timer.IsRunning());
    ui.Fps("0 (Off)");
    ui.Toggle("Train View", true);
    CHECK_FALSE(timer.IsRunning());
    ui.Toggle("Train View", false);
    ui.Fps("10");
    CHECK_FALSE(timer.IsRunning());
    ui.Toggle("Train View", true);
    CHECK(timer.IsRunning());
    CHECK(timer.GetInterval() == 100);
    ui.Reset();
    CHECK_FALSE(timer.IsRunning());
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Train startup and Reset retain the configured main View", "[.][prd110_main_pane][prd110_train]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = train\n");
    ui.Settle();
    ui.CheckToggle("Train View", true, false);
    ui.CheckToggle("Evaluation View", false, true);
    CHECK_FALSE(ui.Train().IsFloatable());
    CHECK_FALSE(ui.Train().HasCloseButton());
    ui.Toggle("Train View", false);
    CHECK(ui.Train().IsShown());
    ui.Toggle("Evaluation View", true);
    ui.Toggle("Train View", false);
    ui.Toggle("Train View", true);
    ui.Reset();
    CHECK(ui.Train().IsShown());
    CHECK_FALSE(ui.Eval().IsShown());
    CHECK(ui.Train().dock_direction == wxAUI_DOCK_CENTER);
    CHECK(TrainPanelProbe::Timer(ui.TrainView()).IsRunning());
    CHECK(TrainPanelProbe::Timer(ui.TrainView()).GetInterval() == 100);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Both main Views start in the home split", "[.][prd110_main_pane][prd110_both]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.Settle();
    CHECK(ui.Train().dock_direction == wxAUI_DOCK_CENTER);
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_RIGHT);
    CHECK(ui.Eval().dock_layer == 0);
    ui.CheckToggle("Train View", true, true);
    ui.CheckToggle("Evaluation View", true, true);
    CHECK(std::abs(ui.DockSize(ui.Eval().name) - ui.frame->GetClientSize().x / 2) <= 2);
    ui.Toggle("Train View", false);
    ui.Reset();
    CHECK(ui.Train().IsShown());
    CHECK(ui.Eval().IsShown());
    CHECK(std::abs(ui.DockSize(ui.Eval().name) - ui.frame->GetClientSize().x / 2) <= 2);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Main View normalization accepts uppercase and surrounding spaces", "[.][prd110_main_pane][prd110_normalized]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view =   EVAL  \n");
    ui.Settle();
    CHECK_FALSE(ui.Train().IsShown());
    CHECK(ui.Eval().IsShown());
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    ui.CheckToggle("Evaluation View", true, false);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Train visibility and FPS are safe and retained before initialization", "[.][prd110_main_pane][prd110_preinit]")
{
    MainPaneFixture ui;
    ui.Start();
    auto manager = std::make_shared<anet::rl::RunManager>(wxGetApp().GetConfigData());
    TrainPanelProbe hidden(ui.frame, TrainPanelConfig{.fps=10});
    hidden.SetViewVisible(false);
    hidden.SetFps(5);
    hidden.SetViewVisible(true);
    CHECK_FALSE(TrainPanelProbe::Timer(hidden).IsRunning());
    hidden.SetViewVisible(false);
    hidden.Initialize(manager);
    CHECK_FALSE(TrainPanelProbe::Timer(hidden).IsRunning());
    hidden.SetViewVisible(true);
    CHECK(TrainPanelProbe::Timer(hidden).IsRunning());
    CHECK(TrainPanelProbe::Timer(hidden).GetInterval() == 200);

    TrainPanelProbe shown(ui.frame, TrainPanelConfig{.fps=10});
    shown.SetFps(4);
    shown.SetViewVisible(false);
    shown.SetViewVisible(true);
    CHECK_FALSE(TrainPanelProbe::Timer(shown).IsRunning());
    shown.Initialize(manager);
    CHECK(TrainPanelProbe::Timer(shown).IsRunning());
    CHECK(TrainPanelProbe::Timer(shown).GetInterval() == 250);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Floating Eval cannot leave Train without a docked partner", "[.][prd110_main_pane][prd110_float]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.Settle();
    REQUIRE(ui.Train().HasCloseButton());
    ui.Eval().Float();
    ui.Manager().Update();
    const auto saved = ui.Train().HasFlag(wxAuiPaneInfo::savedHiddenState);
    // 次のUPDATE_UIより前に、まだ描かれている ✕ を押した経路を検証する。
    ui.Close(ui.Train().window, false);
    ui.Button(ui.Train().window, wxAUI_BUTTON_PIN, false);
    CHECK(ui.Train().IsDocked());
    CHECK(ui.Train().IsShown());
    CHECK(ui.Train().HasFlag(wxAuiPaneInfo::savedHiddenState) == saved);
    CHECK(ui.Eval().IsFloating());
    ui.Settle();
    CHECK_FALSE(ui.Train().HasCloseButton());
    CHECK_FALSE(ui.Train().HasPinButton());
    ui.CheckToggle("Train View", true, false);
    ui.CheckToggle("Evaluation View", true, true);
    ui.Toggle("Train View", false);
    CHECK(ui.Train().IsShown());
    CHECK(ui.Eval().IsFloating());
    // float 中の Eval 自身は閉じられ、再表示でもfloatを保つ。
    REQUIRE(ui.Eval().frame);
    ui.Eval().frame->Close();
    ui.Settle();
    CHECK_FALSE(ui.Eval().IsShown());
    ui.Toggle("Evaluation View", true);
    CHECK(ui.Eval().IsFloating());
    ui.Eval().Dock().Right().Layer(0).Row(0).Position(0);
    ui.Manager().Update();
    ui.Settle();
    CHECK(ui.Train().HasCloseButton());
    CHECK(ui.Eval().IsFloatable());
    CHECK(ui.Train().HasPinButton());
    CHECK(ui.Eval().HasPinButton());
    ui.CheckToggle("Train View", true, true);
    ui.frame->SetSize(ui.frame->GetSize() + wxSize(2, 0));
    ui.Settle();
    CHECK(std::abs(ui.DockSize(ui.Eval().name) - ui.frame->GetClientSize().x / 2) <= 2);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("A floating Centre pane can close before the next UI declaration", "[.][prd110_main_pane][prd110_float_before_declaration]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.Settle();
    ui.Button(ui.Train().window, wxAUI_BUTTON_PIN, false);
    REQUIRE(ui.Train().IsFloating());
    REQUIRE(ui.Train().frame);
    // captionの宣言を待たずに浮動窓の✕を押す。
    ui.Train().frame->Close();
    ui.Settle();
    CHECK_FALSE(ui.Train().IsShown());
    ui.CheckMainCaptions();
    ui.Toggle("Train View", true);
    CHECK(ui.Train().IsShown());
    CHECK(ui.Train().IsFloating());
    ui.CheckMainCaptions();
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Floating Train promotes Eval on UI update and preserves its floating state", "[.][prd110_main_pane][prd110_train_float]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.Settle();
    REQUIRE(ui.Train().IsFloatable());
    REQUIRE(ui.Train().HasPinButton());
    REQUIRE(ui.Eval().HasCloseButton());
    ui.Button(ui.Train().window, wxAUI_BUTTON_PIN, false);
    const bool saved = ui.Eval().HasFlag(wxAuiPaneInfo::savedHiddenState);
    // 宣言が追いつく前でも、最後のdockedのEvalは閉じられない。
    ui.Close(ui.Eval().window, false);
    ui.Button(ui.Eval().window, wxAUI_BUTTON_PIN, false);
    CHECK(ui.Eval().IsDocked());
    CHECK(ui.Eval().IsShown());
    CHECK(ui.Eval().HasFlag(wxAuiPaneInfo::savedHiddenState) == saved);
    ui.Settle();
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    CHECK_FALSE(ui.Eval().IsFloatable());
    CHECK_FALSE(ui.Eval().HasCloseButton());
    CHECK_FALSE(ui.Eval().HasPinButton());
    ui.CheckToggle("Evaluation View", true, false);
    CHECK(ui.Train().IsFloating());
    CHECK(ui.Train().IsShown());
    CHECK(TrainPanelProbe::Timer(ui.TrainView()).IsRunning());
    ui.Toggle("Train View", false);
    CHECK_FALSE(ui.Train().IsShown());
    CHECK_FALSE(TrainPanelProbe::Timer(ui.TrainView()).IsRunning());
    ui.Toggle("Train View", true);
    CHECK(ui.Train().IsFloating());
    CHECK(ui.Train().IsShown());
    CHECK(TrainPanelProbe::Timer(ui.TrainView()).IsRunning());
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Stale pin buttons cannot float the last docked main pane", "[.][prd110_main_pane][prd110_pin_guard]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    for (bool float_train : {true, false}) {
        ui.Reset();
        auto* floating = float_train ? ui.Train().window : ui.Eval().window;
        auto* staying = float_train ? ui.Eval().window : ui.Train().window;
        REQUIRE(ui.Manager().GetPane(floating).HasPinButton());
        ui.Button(floating, wxAUI_BUTTON_PIN, false);
        REQUIRE(ui.Manager().GetPane(floating).IsFloating());
        const auto before = ui.Manager().GetPane(staying);
        REQUIRE(before.IsFloatable());
        REQUIRE(before.HasPinButton());
        // UPDATE_UI前の古いpinが届いても、配置と退避状態を変えない。
        ui.Button(staying, wxAUI_BUTTON_PIN, false);
        CHECK(ui.Manager().GetPane(staying).IsDocked());
        CHECK(ui.Manager().GetPane(staying).dock_direction == before.dock_direction);
        CHECK(ui.Manager().GetPane(staying).HasFlag(wxAuiPaneInfo::savedHiddenState)
            == before.HasFlag(wxAuiPaneInfo::savedHiddenState));
        ui.Settle();
        CHECK(ui.Manager().GetPane(staying).dock_direction == wxAUI_DOCK_CENTER);
        CHECK_FALSE(ui.Manager().GetPane(staying).HasPinButton());
        CHECK_FALSE(ui.Manager().GetPane(staying).HasCloseButton());
    }
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Pin respects a lone main pane and restores a maximized pane before floating", "[.][prd110_main_pane][prd110_pin_maximize]")
{
    MainPaneFixture ui;
    ui.Start();
    ui.Settle();
    REQUIRE_FALSE(ui.Eval().HasPinButton());
    ui.Button(ui.Eval().window, wxAUI_BUTTON_PIN);
    CHECK(ui.Eval().IsDocked());
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    ui.Toggle("Train View", true);
    ui.Button(ui.Train().window, wxAUI_BUTTON_MAXIMIZE_RESTORE);
    REQUIRE(ui.Train().IsMaximized());
    REQUIRE(ui.Train().HasPinButton());
    // maximizeの解除とfloatは、pinを受けたwx自身の順序で行う。
    ui.Button(ui.Train().window, wxAUI_BUTTON_PIN);
    CHECK_FALSE(ui.Train().IsMaximized());
    CHECK(ui.Train().IsFloating());
    CHECK(ui.Eval().IsShown());
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    CHECK_FALSE(ui.Eval().HasPinButton());
    CHECK_FALSE(ui.Eval().HasCloseButton());
    ui.CheckToggle("Evaluation View", true, false);
    CHECK(TrainPanelProbe::Timer(ui.TrainView()).IsRunning());
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Auxiliary pin buttons float panes without changing the main pane roles", "[.][prd110_main_pane][prd110_aux_pin]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.Toggle("Evaluation QValue View", true);
    ui.Toggle("Evaluation Trace", true);
    for (const char* name : {"LogPanel", "EvalQValuePanel", "EvalTracePanel"}) {
        REQUIRE(ui.Manager().GetPane(name).HasPinButton());
        ui.Button(ui.Manager().GetPane(name).window, wxAUI_BUTTON_PIN);
        CHECK(ui.Manager().GetPane(name).IsFloating());
        CHECK(ui.Train().dock_direction == wxAUI_DOCK_CENTER);
        CHECK(ui.Eval().dock_direction == wxAUI_DOCK_RIGHT);
        CHECK(ui.Train().IsFloatable());
        CHECK(ui.Eval().IsFloatable());
        CHECK(ui.Train().HasCloseButton());
        CHECK(ui.Eval().HasCloseButton());
        CHECK(ui.Train().HasPinButton());
        CHECK(ui.Eval().HasPinButton());
        ui.CheckToggle("Train View", true, true);
        ui.CheckToggle("Evaluation View", true, true);
    }
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Docked Train keeps its side position while Eval owns the centre", "[.][prd110_main_pane][prd110_side_owner]")
{
    MainPaneFixture ui;
    ui.Start();
    ui.Toggle("Train View", true);
    ui.Train().Float();
    ui.Manager().Update();
    ui.Settle();
    REQUIRE(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    // 縁へ落とした4値を実際のAUIから記録する。幅は宣言後に主領域を等分する。
    ui.Train().Dock().Left().Layer(2).Row(1).Position(3);
    ui.Train().dock_size = 420;
    ui.Manager().Update();
    const int layer = ui.Train().dock_layer;
    const int row = ui.Train().dock_row;
    const int position = ui.Train().dock_pos;
    ui.Settle();
    const auto side = [&] {
        CHECK(ui.Train().dock_direction == wxAUI_DOCK_LEFT);
        CHECK(ui.Train().dock_layer == layer);
        CHECK(ui.Train().dock_row == row);
        CHECK(ui.Train().dock_pos == position);
        ui.CheckMainSplit();
        CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
        CHECK(ui.Train().IsFloatable());
        CHECK(ui.Eval().IsFloatable());
        CHECK(ui.Train().HasCloseButton());
        CHECK(ui.Eval().HasCloseButton());
        ui.CheckToggle("Train View", true, true);
        ui.CheckToggle("Evaluation View", true, true);
    };
    side();
    ui.Toggle("Evaluation View", false);
    CHECK(ui.Train().dock_direction == wxAUI_DOCK_CENTER);
    CHECK_FALSE(ui.Train().IsFloatable());
    ui.Toggle("Evaluation View", true);
    side();
    ui.Toggle("Train View", false);
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    ui.Toggle("Train View", true);
    side();
    ui.Reset();
    CHECK_FALSE(ui.Train().IsShown());
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    ui.Toggle("Train View", true);
    CHECK(ui.Train().dock_direction == wxAUI_DOCK_CENTER);
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_RIGHT);
    CHECK(ui.Eval().dock_layer == 0);
    CHECK(ui.Eval().dock_row == 0);
    CHECK(ui.Eval().dock_pos == 0);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Eval Step and Resume reclaim its centre without a fallback", "[.][prd110_main_pane][prd110_eval_show]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.Train().Float();
    ui.Manager().Update();
    ui.Settle();
    ui.Train().Dock().Left().Layer(2).Row(1).Position(3);
    ui.Manager().Update();
    ui.Settle();
    const auto control = [&](const wxString& label) {
        auto* toolbar = dynamic_cast<wxAuiToolBar*>(ui.Manager().GetPane("RunControlToolBar").window);
        REQUIRE(toolbar);
        wxAuiToolBarItem* tool = nullptr;
        for (size_t i = 0; i < toolbar->GetToolCount(); ++i) {
            auto* candidate = toolbar->FindToolByIndex(static_cast<int>(i));
            if (candidate->GetLabel() == label) tool = candidate;
        }
        REQUIRE(tool);
        wxCommandEvent event(wxEVT_TOOL, tool->GetId());
        toolbar->GetEventHandler()->ProcessEvent(event);
        ui.Settle();
    };
    // メニューの表示トグル以外からの再表示も、元のCentreを取り戻す。
    ui.Toggle("Evaluation View", false);
    control("Step");
    CHECK(ui.Eval().IsShown());
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    CHECK(ui.Train().dock_direction == wxAUI_DOCK_LEFT);
    CHECK(ui.Train().dock_layer == 2);
    ui.Toggle("Evaluation View", false);
    control("Eval");
    CHECK(ui.Eval().IsShown());
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    CHECK(ui.Train().dock_direction == wxAUI_DOCK_LEFT);
    CHECK(ui.Train().dock_layer == 2);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Maximized main panes restore before hiding and preserve explicit closes", "[.][prd110_main_pane][prd110_maximize]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.Button(ui.Train().window, wxAUI_BUTTON_MAXIMIZE_RESTORE);
    REQUIRE(ui.Train().IsMaximized());
    ui.Toggle("Train View", false);
    CHECK_FALSE(ui.Train().IsMaximized());
    CHECK_FALSE(ui.Train().IsShown());
    CHECK(ui.Eval().IsShown());
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    CHECK_FALSE(ui.Eval().HasCloseButton());

    ui.Toggle("Train View", true);
    ui.Button(ui.Train().window, wxAUI_BUTTON_MAXIMIZE_RESTORE);
    ui.Toggle("Evaluation View", false);
    CHECK_FALSE(ui.Train().HasCloseButton());
    ui.Close(ui.Train().window);
    CHECK(ui.Train().IsShown());
    CHECK(ui.Train().IsMaximized());
    ui.Button(ui.Train().window, wxAUI_BUTTON_MAXIMIZE_RESTORE);
    CHECK(ui.Train().IsShown());
    CHECK_FALSE(ui.Eval().IsShown());

    ui.Toggle("Evaluation View", true);
    ui.Button(ui.Eval().window, wxAUI_BUTTON_MAXIMIZE_RESTORE);
    REQUIRE(ui.Eval().IsMaximized());
    CHECK_FALSE(TrainPanelProbe::Timer(ui.TrainView()).IsRunning());
    ui.Toggle("Evaluation View", false);
    CHECK_FALSE(ui.Eval().IsMaximized());
    CHECK_FALSE(ui.Eval().IsShown());
    CHECK(ui.Train().IsShown());
    CHECK(ui.Train().dock_direction == wxAUI_DOCK_CENTER);
    CHECK(TrainPanelProbe::Timer(ui.TrainView()).IsRunning());
    // 補助paneの最大化で主paneが一時非表示でも、保険で引き戻さない。
    auto* logs = ui.Manager().GetPane("LogPanel").window;
    ui.Button(logs, wxAUI_BUTTON_MAXIMIZE_RESTORE);
    CHECK_FALSE(ui.Train().IsShown());
    CHECK_FALSE(TrainPanelProbe::Timer(ui.TrainView()).IsRunning());
    ui.CheckToggle("Train View", true, false);
    ui.Button(logs, wxAUI_BUTTON_MAXIMIZE_RESTORE);
    CHECK(ui.Train().IsShown());
    CHECK(TrainPanelProbe::Timer(ui.TrainView()).IsRunning());
    // 補助pane最大化中の再表示でも、元の占有者を正しく選ぶ。
    ui.Toggle("Evaluation View", true);
    ui.Train().Float();
    ui.Manager().Update();
    ui.Settle();
    ui.Train().Dock().Left().Layer(2).Row(1).Position(3);
    ui.Manager().Update();
    ui.Settle();
    ui.Toggle("Evaluation View", false);
    ui.Button(logs, wxAUI_BUTTON_MAXIMIZE_RESTORE);
    ui.Toggle("Evaluation View", true);
    ui.Button(logs, wxAUI_BUTTON_MAXIMIZE_RESTORE);
    CHECK(ui.Train().dock_direction == wxAUI_DOCK_LEFT);
    CHECK(ui.Train().dock_layer == 2);
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    ui.CheckToggle("Train View", true, true);
    ui.CheckToggle("Evaluation View", true, true);

    ui.CheckNoMainPaneFallback();
}

TEST_CASE("An unknown direct hide repairs the centre and warns only once", "[.][prd110_main_pane][prd110_fallback]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.Train().Hide();
    ui.Eval().Hide();
    ui.Manager().Update();
    ui.frame->SetSize(ui.frame->GetSize() + wxSize(2, 0));
    ui.Settle();
    CHECK(ui.Train().IsShown());
    CHECK(ui.Train().dock_direction == wxAUI_DOCK_CENTER);
    CHECK_FALSE(ui.Train().IsFloatable());
    CHECK(TrainPanelProbe::Timer(ui.TrainView()).IsRunning());
    ui.CheckToggle("Train View", true, false);
    ui.frame->SetSize(ui.frame->GetSize() + wxSize(2, 0));
    ui.Settle();
    const auto log = ui.Log();
    constexpr char kWarning[] = "No main pane occupies the centre. Showing Train View as a fallback.";
    const auto first = log.find(kWarning);
    REQUIRE(first != std::string::npos);
    CHECK(log.find(kWarning, first + sizeof(kWarning) - 1) == std::string::npos);

    // 直接操作で両方Centreになった場合は、通常の復帰フックとは別の保険で直す。
    ui.Toggle("Evaluation View", true);
    ui.SetDockSize(ui.Eval().name, 330);
    ui.Eval().Centre();
    ui.Manager().Update();
    ui.frame->SetSize(ui.frame->GetSize() + wxSize(2, 0));
    ui.Settle();
    CHECK(ui.Train().IsShown());
    CHECK(ui.Train().dock_direction == wxAUI_DOCK_CENTER);
    CHECK(ui.Eval().IsShown());
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_RIGHT);
    CHECK(ui.Eval().dock_layer == 0);
    CHECK(ui.Eval().dock_row == 0);
    CHECK(ui.Eval().dock_pos == 0);
    ui.CheckMainSplit();
    ui.frame->SetSize(ui.frame->GetSize() + wxSize(2, 0));
    ui.Settle();
    constexpr char kCentresWarning[] = "Both main panes occupy the centre. Moving Evaluation View aside as a fallback.";
    const auto centres_log = ui.Log();
    const auto centre_first = centres_log.find(kCentresWarning);
    REQUIRE(centre_first != std::string::npos);
    CHECK(centres_log.find(kCentresWarning, centre_first + sizeof(kCentresWarning) - 1) == std::string::npos);
    CHECK(centres_log.find(kWarning, first + sizeof(kWarning) - 1) == std::string::npos);
}

TEST_CASE("A redundant show keeps unknown double centres on the fallback path", "[.][prd110_main_pane][prd110_redundant_show]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.Eval().Centre();
    ui.Manager().Update();
    // 表示済みへのONイベントを、隠れていた占有者の復帰と取り違えない。
    ui.Toggle("Evaluation View", true);
    CHECK(ui.Train().dock_direction == wxAUI_DOCK_CENTER);
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_RIGHT);
    ui.Toggle("Evaluation View", true);
    const auto log = ui.Log();
    constexpr char kWarning[] = "Both main panes occupy the centre. Moving Evaluation View aside as a fallback.";
    const auto first = log.find(kWarning);
    REQUIRE(first != std::string::npos);
    CHECK(log.find(kWarning, first + sizeof(kWarning) - 1) == std::string::npos);
    CHECK(log.find("No main pane occupies the centre.") == std::string::npos);
}

TEST_CASE("Train centre reserves its minimum and leaves a shared side column to wx", "[.][prd110_main_pane][prd110_train_columns]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = train\n");
    ui.frame->SetClientSize(wxSize(1800, 800));
    ui.Settle();
    ui.Toggle("Evaluation QValue View", true);
    ui.Toggle("Evaluation Trace", true);
    auto& trace = ui.Manager().GetPane("EvalTracePanel");
    const int trace_min = std::max(trace.min_size.x, trace.window->GetMinSize().x);
    ui.Toggle("Evaluation Trace", false);
    ui.Manager().GetPane("EvalTracePanel").dock_size = trace_min + 200;
    ui.Toggle("Evaluation Trace", true);
    const int q_width = ui.DockSize("EvalQValuePanel");
    const int trace_width = ui.DockSize("EvalTracePanel");
    REQUIRE(trace_width >= trace_min + 100);
    // 補助列に縮小余地があれば、Q-Valuesを保って単独Centreへ200を残す。
    ui.frame->SetClientSize(wxSize(q_width + trace_width + 150, 800));
    ui.Settle();
    CHECK(ui.DockSize("EvalQValuePanel") == q_width);
    CHECK(ui.DockSize("EvalTracePanel") <= trace_width - 50);
    CHECK(ui.frame->GetClientSize().x - ui.DockSize("EvalQValuePanel") - ui.DockSize("EvalTracePanel") >= 200);

    ui.frame->SetClientSize(wxSize(1800, 800));
    ui.Toggle("Evaluation View", true);
    ui.Train().Float();
    ui.Manager().Update();
    ui.Settle();
    ui.Train().Dock().Right().Layer(20).Row(0).Position(1);
    ui.Manager().Update();
    ui.Settle();
    const int side_width = ui.DockSize(ui.Train().name);
    ui.frame->SetClientSize(wxSize(ui.DockSize("EvalQValuePanel") + side_width + 150, 800));
    ui.Manager().Update();
    // 同じ列にTrainとTraceがある場合も、wxの配分をFrameが上書きしない。
    const int native_width = ui.DockSize(ui.Train().name);
    ui.Settle();
    CHECK(ui.Train().dock_direction == wxAUI_DOCK_RIGHT);
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    CHECK(ui.DockSize(ui.Train().name) == native_width);
    CHECK(ui.DockSize("EvalTracePanel") == native_width);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Main pane widths reserve the right columns and allow a single Eval minimum", "[.][prd110_main_pane][prd110_columns]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.frame->SetClientSize(wxSize(1800, 800));
    ui.Settle();
    ui.Toggle("Evaluation QValue View", true);
    ui.Toggle("Evaluation Trace", true);
    const int q_width = ui.DockSize("EvalQValuePanel");
    const int trace_width = ui.DockSize("EvalTracePanel");
    REQUIRE(q_width > 0);
    REQUIRE(trace_width > 0);
    const int main_width = ui.frame->GetClientSize().x - q_width - trace_width;
    CHECK(std::abs(ui.DockSize(ui.Eval().name) - main_width / 2) <= 2);
    const auto size = ui.frame->GetSize();
    ui.Toggle("Train View", false);
    CHECK(ui.frame->GetSize() == size);
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    // 単独Evalには300pxを残せば列を縮める必要がない（両方なら400pxが要る）。
    ui.frame->SetClientSize(wxSize(q_width + trace_width + 300, 800));
    ui.Settle();
    CHECK(ui.DockSize("EvalQValuePanel") == q_width);
    CHECK(ui.DockSize("EvalTracePanel") == trace_width);
    ui.Reset();
    CHECK(ui.frame->GetClientSize().x == q_width + trace_width + 300);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("A left main pane becomes equal width on placement change and resize", "[.][prd110_main_pane][prd110_width_left]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.frame->SetClientSize(wxSize(1800, 800));
    ui.Settle();
    ui.Button(ui.Train().window, wxAUI_BUTTON_PIN);
    ui.Train().Dock().Left().Layer(1).Row(0).Position(0);
    ui.Manager().Update();
    ui.Settle();
    CHECK(ui.Train().dock_direction == wxAUI_DOCK_LEFT);
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    ui.CheckMainSplit();
    ui.frame->SetClientSize(wxSize(1400, 800));
    ui.Settle();
    ui.CheckMainSplit();
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("An outer right main pane excludes the intervening and opposite columns", "[.][prd110_main_pane][prd110_width_outer]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.frame->SetClientSize(wxSize(1800, 800));
    ui.Toggle("Evaluation QValue View", true);
    const int q_width = ui.DockSize("EvalQValuePanel");
    ui.Button(ui.Train().window, wxAUI_BUTTON_PIN);
    ui.Train().Dock().Right().Layer(11).Row(0).Position(0);
    ui.Manager().Update();
    ui.Settle();
    ui.CheckMainSplit();
    CHECK(ui.DockSize("EvalQValuePanel") == q_width);

    // 左右に別の縦列があっても、その両方を主領域から除く。
    ui.Toggle("Evaluation Trace", true);
    ui.Manager().GetPane("EvalQValuePanel").Dock().Left().Layer(10).Row(0).Position(0);
    ui.Train().Layer(21);
    ui.Manager().Update();
    const int left_width = ui.DockSize("EvalQValuePanel");
    const int right_width = ui.DockSize("EvalTracePanel");
    ui.Settle();
    ui.CheckMainSplit();
    CHECK(ui.DockSize("EvalQValuePanel") == left_width);
    CHECK(ui.DockSize("EvalTracePanel") == right_width);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Outer Bottom and shared side main docks retain wx dimensions", "[.][prd110_main_pane][prd110_width_exclusions]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.frame->SetClientSize(wxSize(1800, 800));
    ui.Settle();
    ui.Toggle("Evaluation QValue View", true);
    ui.Button(ui.Train().window, wxAUI_BUTTON_PIN);
    ui.Train().Dock().Bottom().Layer(11).Row(0).Position(0);
    ui.Manager().Update();
    const auto size = ui.Train().rect.GetSize();
    const int height = ui.DockSize(ui.Train().name);
    ui.Settle();
    CHECK(ui.Train().rect.GetSize() == size);
    CHECK(ui.DockSize(ui.Train().name) == height);

    ui.Toggle("Evaluation Trace", true);
    ui.Train().Right().Layer(20).Row(0).Position(1);
    ui.Manager().Update();
    const int shared_width = ui.DockSize(ui.Train().name);
    ui.Settle();
    CHECK(ui.DockSize(ui.Train().name) == shared_width);
    CHECK(ui.DockSize("EvalTracePanel") == shared_width);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Width alone does not retrigger equalization and maximize keeps remembered width", "[.][prd110_main_pane][prd110_width_stability]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.frame->SetClientSize(wxSize(1600, 800));
    ui.Settle();
    ui.CheckMainSplit();
    const auto before = ui.Eval();
    ui.SetDockSize(ui.Eval().name, 330);
    REQUIRE(ui.DockSize(ui.Eval().name) == 330);
    CHECK(ui.Eval().dock_direction == before.dock_direction);
    CHECK(ui.Eval().dock_layer == before.dock_layer);
    CHECK(ui.Eval().dock_row == before.dock_row);
    CHECK(ui.Eval().dock_pos == before.dock_pos);
    for (int pass = 0; pass < 3; ++pass) {
        ui.Settle();
        CHECK(ui.DockSize(ui.Eval().name) == 330);
    }
    ui.Toggle("Log View", false);
    ui.CheckMainSplit();
    ui.SetDockSize(ui.Eval().name, 330);
    ui.Button(ui.Eval().window, wxAUI_BUTTON_MAXIMIZE_RESTORE);
    REQUIRE(ui.Eval().IsMaximized());
    CHECK(ui.Eval().dock_size == 330);
    ui.Settle();
    CHECK(ui.Eval().dock_size == 330);
    ui.Button(ui.Eval().window, wxAUI_BUTTON_MAXIMIZE_RESTORE);
    ui.CheckMainSplit();
    ui.Reset();
    ui.SetDockSize(ui.Eval().name, 310);
    ui.Settle();
    CHECK(ui.DockSize(ui.Eval().name) == 310);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Centre captions follow visible docked ownership and preserve Trace annotations", "[.][prd110_main_pane][prd110_centre_caption]")
{
    MainPaneFixture ui;
    ui.Start();
    ui.Settle();
    ui.CheckMainCaptions();
    ui.Toggle("Train View", true);
    ui.CheckMainCaptions();
    ui.Button(ui.Train().window, wxAUI_BUTTON_PIN);
    ui.CheckMainCaptions();
    ui.Train().Dock().Left().Layer(1).Row(0).Position(0);
    ui.Manager().Update();
    ui.Settle();
    ui.CheckMainCaptions();
    ui.Toggle("Evaluation View", false);
    ui.CheckMainCaptions();
    ui.Toggle("Evaluation View", true);
    ui.CheckMainCaptions();
    ui.Train().Bottom().Layer(1).Row(0).Position(0);
    ui.Manager().Update();
    ui.Settle();
    ui.CheckMainCaptions();
    ui.Button(ui.Eval().window, wxAUI_BUTTON_MAXIMIZE_RESTORE);
    ui.CheckMainCaptions();
    ui.Button(ui.Eval().window, wxAUI_BUTTON_MAXIMIZE_RESTORE);
    ui.CheckMainCaptions();
    ui.Toggle("Train View", false);
    ui.CheckMainCaptions();
    ui.Reset();
    ui.CheckMainCaptions();

    // Traceが登録時のcaptionに注記を足す入力も、自分の印だけ末尾へ正規化する。
    ui.Eval().Caption("Evaluation View [Centre] [Trace: step 7]");
    ui.Settle();
    CHECK(ui.Eval().caption == "Evaluation View [Trace: step 7] [Centre]");
    ui.Toggle("Train View", true);
    CHECK(ui.Eval().caption == "Evaluation View [Trace: step 7]");
    ui.Button(ui.Train().window, wxAUI_BUTTON_PIN);
    ui.Train().Dock().Left().Layer(1).Row(0).Position(0);
    ui.Manager().Update();
    ui.Settle();
    CHECK(ui.Eval().caption == "Evaluation View [Trace: step 7] [Centre]");
    ui.Button(ui.Eval().window, wxAUI_BUTTON_PIN);
    REQUIRE(ui.Eval().IsFloating());
    REQUIRE(ui.Eval().frame);
    CHECK(ui.Eval().caption == "Evaluation View [Trace: step 7]");
    CHECK(ui.Eval().frame->GetLabel() == ui.Eval().caption);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Trace row movement preserves Centre captions without rebuilding layouts", "[.][prd110_main_pane][prd110_trace_caption]")
{
    // 公開のroot sizerの破棄で全体更新を観測し、本体にtest-only APIを足さない。
    class LayoutRebuildProbe final : public wxBoxSizer {
    public:
        explicit LayoutRebuildProbe(int& count) : wxBoxSizer(wxVERTICAL), count_(count) {}
        ~LayoutRebuildProbe() override { ++count_; }
    private:
        int& count_; ///< frameより長く生きるテストスコープが所有
    };
    int rebuilds = 0;
    MainPaneFixture ui;
    ui.Start();
    ui.frame->SetClientSize(wxSize(1800, 900));
    ui.Toggle("Evaluation Trace", true);
    auto* trace = dynamic_cast<EvalTracePanel*>(ui.Manager().GetPane("EvalTracePanel").window);
    auto* eval = dynamic_cast<EvalPanel*>(ui.Eval().window);
    REQUIRE(trace);
    REQUIRE(eval);
    for (int step = 0; step < 8; ++step) eval->DoStep();

    const auto watch_layout = [&](wxWindow& window) {
        auto* original = window.GetSizer();
        REQUIRE(original);
        // 既存のsizerをそのまま包み、次のUpdateによる置換だけを数える。
        window.SetSizer(nullptr, false);
        auto* probe = new LayoutRebuildProbe(rebuilds);
        probe->Add(original, 1, wxEXPAND);
        window.SetSizer(probe);
    };
    for (bool floating : {false, true}) {
        if (floating) {
            ui.Toggle("Train View", true);
            ui.Button(ui.Eval().window, wxAUI_BUTTON_PIN);
            REQUIRE(ui.Eval().IsFloating());
            REQUIRE(ui.Eval().frame);
        }
        // 実Traceの選択開始の1回のUpdateは許し、通常の行移動だけを測る。
        trace->MoveSelection(-1);
        ui.Settle();
        REQUIRE(trace->IsTraceMode());
        rebuilds = 0;
        watch_layout(*ui.frame);
        if (floating) watch_layout(*ui.Eval().frame);
        for (int row = 1; row < 7; ++row) {
            const auto previous = ui.Eval().caption;
            trace->MoveSelection(1);
            ui.Settle();
            const auto& caption = ui.Eval().caption;
            CHECK(caption != previous);
            CHECK(caption.StartsWith("Evaluation View [Trace: step "));
            if (floating) {
                CHECK_FALSE(caption.Contains(" [Centre]"));
                CHECK(ui.Eval().frame->GetLabel() == caption);
            } else {
                CHECK(caption.EndsWith(" [Centre]"));
            }
            CHECK(rebuilds == 0);
        }
        trace->ReturnToLive();
        ui.Settle();
    }
    ui.CheckMainCaptions();
    ui.CheckNoMainPaneFallback();
}

namespace {

void CheckPausedViewResize(MainPaneFixture& ui)
{
    auto* eval = dynamic_cast<EvalPanel*>(ui.Eval().window);
    REQUIRE(eval);
    REQUIRE(eval->GetSizer());
    auto* canvas = eval->GetSizer()->GetItem(size_t{0})->GetWindow();
    REQUIRE(canvas);
    eval->DoStep();
    ui.Settle();
    const auto capture = [](wxWindow& window) {
        const auto size = window.GetClientSize();
        wxBitmap bitmap(size.x, size.y, 24);
        wxMemoryDC target(bitmap);
        wxClientDC source(&window);
        target.Blit(0, 0, size.x, size.y, &source, 0, 0);
        target.SelectObject(wxNullBitmap);
        return bitmap.ConvertToImage();
    };
    // 同じ停止snapshotを、通常のresize後と明示的な全面再描画後で比較する。
    for (bool split : {false, true}) {
        ui.Toggle("Train View", split);
        for (int direction = 0; direction < 3; ++direction) {
            const std::string axis = direction == 0 ? "height" : direction == 1 ? "width" : "both";
            const std::string name = (split ? "split_" : "eval_") + axis;
            INFO("condition=" << name);
            INFO("captures=" << ui.output.string());
            ui.frame->SetClientSize(900, 600);
            ui.Settle();
            canvas->Refresh(false);
            canvas->Update();
            ui.Pump();
            for (int step = 1; step <= 8; ++step) {
                ui.frame->SetClientSize(900 + (direction == 0 ? 0 : step * 12),
                    600 + (direction == 1 ? 0 : step * 12));
                ui.Settle();
            }
            const auto actual = capture(*canvas);
            canvas->Refresh(false);
            canvas->Update();
            ui.Pump();
            const auto expected = capture(*canvas);
            REQUIRE(actual.GetSize() == expected.GetSize());
            int different = 0;
            const auto bytes = actual.GetWidth() * actual.GetHeight() * 3;
            for (int byte = 0; byte < bytes; byte += 3) {
                if (!std::equal(actual.GetData() + byte, actual.GetData() + byte + 3,
                    expected.GetData() + byte)) ++different;
            }
            // 画像は別processごとの一時Runの隣へ保存し、失敗時にそのまま比較できる。
            actual.SaveFile((ui.output / (name + "_actual.png")).wstring(), wxBITMAP_TYPE_PNG);
            expected.SaveFile((ui.output / (name + "_expected.png")).wstring(), wxBITMAP_TYPE_PNG);
            INFO("different_pixels=" << different);
            CHECK(different == 0);
        }
    }
    ui.CheckNoMainPaneFallback();
}

} // namespace

TEST_CASE("Paused LunarLander View repaints cleanly after resizing", "[.][prd110_main_pane][prd110_resize_paint]")
{
    MainPaneFixture ui;
    ui.Start("env.class_id = LunarLanderEnv\napp.train_panel.fps = 0\n");
    CheckPausedViewResize(ui);
}

TEST_CASE("Paused CartPole View inherits clean resize painting", "[.][prd110_main_pane][prd110_cartpole_resize]")
{
    MainPaneFixture ui;
    ui.Start("app.train_panel.fps = 0\n");
    CheckPausedViewResize(ui);
}

TEST_CASE("A same-width main band splits the centre height after docking and resizing", "[.][prd110_main_pane][prd110_height_split]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.frame->SetClientSize(1200, 1000);
    ui.Toggle("Log View", false);
    ui.Button(ui.Train().window, wxAUI_BUTTON_PIN);
    // 公開AUIで等分規則の入力を作る。dropの実操作は人の受入に残す。
    ui.Train().Dock().Bottom().Layer(0).Row(0).Position(0);
    ui.Manager().Update();
    ui.Settle();
    ui.CheckMainHeightSplit();
    ui.CheckMainCaptions();
    ui.frame->SetClientSize(1200, 1200);
    ui.Settle();
    ui.CheckMainHeightSplit();
    // 上の帯も同じ規則で、方向以外の条件は変えない。
    ui.Train().Top();
    ui.Manager().Update();
    ui.Settle();
    ui.CheckMainHeightSplit();
    ui.CheckMainCaptions();
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Main height splitting preserves outer Q-Values and Trace column widths", "[.][prd110_main_pane][prd110_height_columns]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.frame->SetClientSize(1800, 1000);
    ui.Toggle("Log View", false);
    ui.Button(ui.Train().window, wxAUI_BUTTON_PIN);
    ui.Train().Dock().Bottom().Layer(0).Row(0).Position(0);
    ui.Manager().Update();
    ui.Settle();
    ui.CheckMainHeightSplit();
    ui.Toggle("Evaluation QValue View", true);
    const int q_width = ui.DockSize("EvalQValuePanel");
    ui.CheckMainHeightSplit();
    ui.Toggle("Evaluation Trace", true);
    const int trace_width = ui.DockSize("EvalTracePanel");
    ui.CheckMainHeightSplit();
    CHECK(ui.DockSize("EvalQValuePanel") == q_width);
    ui.frame->SetClientSize(1800, 1160);
    ui.Settle();
    ui.CheckMainHeightSplit();
    CHECK(ui.DockSize("EvalQValuePanel") == q_width);
    CHECK(ui.DockSize("EvalTracePanel") == trace_width);
    ui.CheckMainCaptions();
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Shared main bands stay unchanged and a separate row preserves Logs", "[.][prd110_main_pane][prd110_height_shared]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.frame->SetClientSize(1400, 1100);
    ui.Settle();
    ui.Button(ui.Train().window, wxAUI_BUTTON_PIN);
    ui.Train().Dock().Bottom().Layer(0).Row(0).Position(1);
    ui.Manager().Update();
    const auto shared_size = ui.Train().rect.GetSize();
    const int shared_height = ui.DockSize(ui.Train().name);
    ui.Settle();
    CHECK(ui.Train().rect.GetSize() == shared_size);
    CHECK(ui.DockSize(ui.Train().name) == shared_height);
    ui.CheckMainCaptions();

    // Logsとは別rowに分けた帯だけを等分し、Logsの高さを予算へ混ぜない。
    const int log_height = ui.DockSize("LogPanel");
    ui.Train().Row(1).Position(0);
    ui.Manager().Update();
    ui.Settle();
    ui.CheckMainHeightSplit();
    CHECK(ui.DockSize("LogPanel") == log_height);
    ui.CheckMainCaptions();

    // 同じlayerの左右の列を含む広い帯も、等分の対象から外す。
    ui.Toggle("Evaluation QValue View", true);
    ui.Manager().GetPane("EvalQValuePanel").Layer(0);
    ui.Manager().Update();
    const auto wide_size = ui.Train().rect.GetSize();
    const int wide_height = ui.DockSize(ui.Train().name);
    ui.Settle();
    CHECK(ui.Train().rect.GetSize() == wide_size);
    CHECK(ui.DockSize(ui.Train().name) == wide_height);
    CHECK(ui.DockSize("LogPanel") == log_height);
    ui.CheckMainCaptions();
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Height splitting restores placement and respects manual sizing and minimum space", "[.][prd110_main_pane][prd110_height_restore]")
{
    MainPaneFixture ui;
    ui.Start("app.main_view = both\n");
    ui.frame->SetClientSize(1400, 1100);
    ui.Toggle("Log View", false);
    ui.Button(ui.Train().window, wxAUI_BUTTON_PIN);
    ui.Train().Dock().Bottom().Layer(0).Row(0).Position(0);
    ui.Manager().Update();
    ui.Settle();
    ui.Toggle("Evaluation View", false);
    CHECK(ui.Train().dock_direction == wxAUI_DOCK_CENTER);
    ui.CheckMainCaptions();
    ui.Toggle("Evaluation View", true);
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    CHECK(ui.Train().dock_direction == wxAUI_DOCK_BOTTOM);
    CHECK(ui.Train().dock_layer == 0);
    CHECK(ui.Train().dock_row == 0);
    ui.CheckMainHeightSplit();
    ui.CheckMainCaptions();

    // 公開serializerで高さだけを準備し、宣言の繰り返しでは戻らないことを確認。
    ui.SetDockSize(ui.Train().name, 250);
    for (int pass = 0; pass < 3; ++pass) {
        ui.Settle();
        CHECK(ui.DockSize(ui.Train().name) == 250);
    }
    ui.Button(ui.Eval().window, wxAUI_BUTTON_MAXIMIZE_RESTORE);
    CHECK(ui.Train().dock_size == 250);
    ui.frame->SetClientSize(1400, 1200);
    ui.Settle();
    CHECK(ui.Train().dock_size == 250);
    ui.Button(ui.Eval().window, wxAUI_BUTTON_MAXIMIZE_RESTORE);
    ui.CheckMainHeightSplit();
    ui.SetDockSize(ui.Train().name, 250);
    ui.Settle();
    CHECK(ui.DockSize(ui.Train().name) == 250);
    ui.Toggle("Train View", false);
    ui.Toggle("Train View", true);
    ui.CheckMainHeightSplit();
    ui.CheckMainCaptions();

    // 縦の高さ不足は主paneの最小高で止め、別rowのLogsを縮めて補わない。
    ui.Train().Row(1);
    ui.Manager().Update();
    ui.Settle();
    ui.Toggle("Log View", true);
    const int log_height = ui.DockSize("LogPanel");
    // 公開LoadLayoutでwx自身の実最小高を得る。caption・枠の計算をテストへ写さない。
    ui.SetDockSize(ui.Train().name, 1);
    const int minimum = ui.DockSize(ui.Train().name);
    REQUIRE(minimum >= 200);
    ui.frame->SetClientSize(1400, 500);
    ui.Settle();
    REQUIRE(ui.Eval().rect.height + ui.DockSize(ui.Train().name) < 400);
    CHECK(ui.DockSize(ui.Train().name) == minimum);
    CHECK(ui.DockSize("LogPanel") == log_height);
    ui.CheckMainCaptions();
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Q-Values takes only its action rows in a shared column until Reset", "[.][runner_layout][q_value_height]")
{
    MainPaneFixture ui;
    ui.Start();
    ui.frame->SetClientSize(1800, 1000);
    ui.Toggle("Evaluation QValue View", true);
    const auto q_value = [&]() -> wxAuiPaneInfo& { return ui.Manager().GetPane("EvalQValuePanel"); };
    const auto trace = [&]() -> wxAuiPaneInfo& { return ui.Manager().GetPane("EvalTracePanel"); };
    auto* panel = dynamic_cast<QValuePanel*>(q_value().window);
    REQUIRE(panel);
    wxGrid* grid = nullptr;
    for (auto* child : panel->GetChildren()) {
        if (auto* candidate = dynamic_cast<wxGrid*>(child)) grid = candidate;
    }
    REQUIRE(grid);
    // CartPole の 2 行動と見出しが、縦スクロールなしで収まる高さ。float の窓もこの高さで開く。
    const int height = panel->GetPreferredDockHeight();
    CHECK(height >= grid->GetColLabelSize() + 2 * grid->GetDefaultRowSize());
    CHECK(q_value().best_size.y == height);
    CHECK(q_value().min_size.y == height);
    // ひとりの列では列いっぱい。
    CHECK(panel->GetSize().y > height * 2);

    // 同じ列のTraceへ余りを渡す。
    ui.Toggle("Evaluation Trace", true);
    trace().Dock().Right().Layer(q_value().dock_layer).Row(q_value().dock_row).Position(q_value().dock_pos + 1);
    ui.Manager().Update();
    ui.Settle();
    CHECK(panel->GetSize().y == height);
    CHECK(grid->GetClientSize().y >= grid->GetColLabelSize() + 2 * grid->GetDefaultRowSize());
    CHECK(trace().window->GetSize().y > height * 2);

    // サッシで広げた伸び率は wx のまま、閉じて開き直しても残る。
    q_value().dock_proportion = trace().dock_proportion;
    ui.Manager().Update();
    ui.Settle();
    CHECK(panel->GetSize().y > height * 2);
    const int widened = q_value().dock_proportion;
    ui.Toggle("Evaluation QValue View", false);
    ui.Toggle("Evaluation QValue View", true);
    CHECK(q_value().dock_proportion == widened);

    // Reset だけが行数ぶんの高さへ戻す。
    ui.Reset();
    CHECK(q_value().dock_proportion == 1);
    CHECK(q_value().best_size.y == height);
    CHECK(q_value().min_size.y == height);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Q-Values keeps its default width as the minimum and follows a wider table", "[.][runner_layout][q_value_width]")
{
    MainPaneFixture ui;
    ui.Start();
    ui.frame->SetClientSize(1400, 800);
    ui.Toggle("Evaluation QValue View", true);
    const auto q_value = [&]() -> wxAuiPaneInfo& { return ui.Manager().GetPane("EvalQValuePanel"); };
    auto* panel = dynamic_cast<QValuePanel*>(q_value().window);
    REQUIRE(panel);
    wxGrid* grid = nullptr;
    wxWindow* reset = nullptr;
    wxWindow* hist = nullptr;
    const auto find = [&](auto&& self, wxWindow* parent) -> void {
        for (auto* child : parent->GetChildren()) {
            if (auto* candidate = dynamic_cast<wxGrid*>(child)) grid = candidate;
            if (dynamic_cast<wxButton*>(child) && child->GetLabel() == "Reset Range") reset = child;
            if (dynamic_cast<wxCheckBox*>(child) && child->GetLabel() == "Hist") hist = child;
            self(self, child);
        }
    };
    find(find, panel);
    REQUIRE(grid);
    REQUIRE(reset);
    REQUIRE(hist);
    CHECK(grid->GetColLabelValue(1) == "Mean");
    CHECK(grid->GetColLabelValue(2) == "Max");
    CHECK(grid->GetColLabelValue(3) == "Min");
    CHECK(grid->GetColLabelValue(4) == "Std");
    // Reset Rangeはヒートマップが狭くても見える左端。
    CHECK(reset->GetPosition().x < hist->GetPosition().x);

    // 最小幅と既定幅は同じで、開いた直後の列はその幅。窓を狭めても縮まない。
    const int width = panel->GetPreferredDockWidth();
    CHECK(q_value().min_size.x == width);
    CHECK(q_value().best_size.x == width);
    CHECK(panel->GetSize().x == width);
    ui.frame->SetClientSize(600, 800);
    ui.Settle();
    CHECK(panel->GetSize().x == width);

    // 長い値で列が広がると、最小幅＝既定幅もその分だけ広がる。
    ui.frame->SetClientSize(1400, 800);
    ui.Settle();
    const int mean_width = grid->GetColSize(1);
    QValueData data;
    data.stats = {{-12345.678f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    data.selected_action = 0;
    data.width = 1;
    data.height = 2;
    data.vv = {-12345.678f, 0.0f};
    panel->ApplyData(data);
    ui.Settle();
    ui.Settle();
    REQUIRE(grid->GetColSize(1) > mean_width);
    CHECK(panel->GetPreferredDockWidth() == width + grid->GetColSize(1) - mean_width);
    CHECK(q_value().min_size.x == panel->GetPreferredDockWidth());
    CHECK(panel->GetSize().x == panel->GetPreferredDockWidth());
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Floating Q-Values can shrink to the table and one row", "[.][runner_layout][q_value_float]")
{
    MainPaneFixture ui;
    ui.Start();
    ui.frame->SetClientSize(1400, 900);
    ui.Toggle("Evaluation QValue View", true);
    const auto q_value = [&]() -> wxAuiPaneInfo& { return ui.Manager().GetPane("EvalQValuePanel"); };
    auto* panel = dynamic_cast<QValuePanel*>(q_value().window);
    REQUIRE(panel);
    const wxSize docked(panel->GetPreferredDockWidth(), panel->GetPreferredDockHeight());
    REQUIRE(q_value().min_size == docked);

    // floatした窓は、表が見える幅と見出し＋1行の高さまで縮められる。
    ui.Button(panel, wxAUI_BUTTON_PIN);
    REQUIRE(q_value().IsFloating());
    REQUIRE(q_value().frame);
    const wxSize minimum = panel->GetFloatingMinSize();
    CHECK(minimum.x < docked.x);
    CHECK(minimum.y < docked.y);
    CHECK(q_value().frame->GetMinClientSize() == minimum);
    q_value().frame->SetClientSize(minimum);
    ui.Settle();
    // 行が収まらず縦スクロールの幅が表に入っても、窓はその最小に合わせて広がる。
    CHECK(panel->GetSize() == panel->GetFloatingMinSize());
    CHECK(q_value().frame->GetClientSize() == panel->GetFloatingMinSize());

    // dockへ戻すと最小＝既定に戻る。
    q_value().Dock().Right().Layer(10).Row(0).Position(0);
    ui.Manager().Update();
    ui.Settle();
    CHECK(q_value().IsDocked());
    CHECK(q_value().min_size == docked);
    CHECK(panel->GetSize().x == docked.x);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Toolbars span the full width above the right columns", "[.][runner_layout][toolbar_full_width]")
{
    MainPaneFixture ui;
    ui.Start();
    ui.frame->SetClientSize(1200, 800);
    ui.Toggle("Evaluation QValue View", true);
    ui.Toggle("Evaluation Trace", true);
    const auto check = [&] {
        // 右の列は上端の帯の下から始まり、4本のtoolbarは全幅の帯に収まる。
        for (const char* column : {"EvalQValuePanel", "EvalTracePanel"}) {
            const auto& pane = ui.Manager().GetPane(column);
            REQUIRE(pane.IsShown());
            REQUIRE(pane.dock_direction == wxAUI_DOCK_RIGHT);
            for (const char* name : {"RunControlToolBar", "StepToolBar", "RunOpsToolBar", "PanelToolBar"}) {
                const auto& toolbar = ui.Manager().GetPane(name);
                INFO(column << " / " << name);
                CHECK(toolbar.rect.GetBottom() < pane.rect.GetTop());
                CHECK(toolbar.rect.GetRight() < ui.frame->GetClientSize().x);
            }
        }
    };
    check();
    ui.Reset();
    ui.Toggle("Evaluation QValue View", true);
    ui.Toggle("Evaluation Trace", true);
    check();
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Q-Values alone in a new band takes only its action rows", "[.][runner_layout][q_value_band_height]")
{
    MainPaneFixture ui;
    ui.Start();
    ui.frame->SetClientSize(1400, 1000);
    ui.Toggle("Log View", false);
    ui.Toggle("Evaluation QValue View", true);
    const auto q_value = [&]() -> wxAuiPaneInfo& { return ui.Manager().GetPane("EvalQValuePanel"); };
    auto* panel = dynamic_cast<QValuePanel*>(q_value().window);
    REQUIRE(panel);
    const int height = panel->GetPreferredDockHeight();
    // 列の幅を覚えたままdropした入力を公開AUIで作る。drop操作そのものは人の受入に残す。
    REQUIRE(q_value().dock_size > height * 2);
    q_value().Dock().Bottom().Row(0).Position(0);
    ui.Manager().Update();
    REQUIRE(ui.DockSize("EvalQValuePanel") > height * 2);
    ui.Settle();
    CHECK(panel->GetSize().y == height);
    CHECK(ui.Eval().dock_direction == wxAUI_DOCK_CENTER);
    CHECK(ui.Eval().window->GetSize().y > height * 2);

    // 置き場が変わらない間はサッシで変えた高さを保ち、閉じて開き直しても残す。
    ui.SetDockSize("EvalQValuePanel", 300);
    ui.Settle();
    CHECK(ui.DockSize("EvalQValuePanel") == 300);
    ui.Toggle("Evaluation QValue View", false);
    ui.Toggle("Evaluation QValue View", true);
    CHECK(ui.DockSize("EvalQValuePanel") == 300);

    // Logsと同居する帯はwxの高さのまま。
    ui.Toggle("Log View", true);
    const auto& logs = ui.Manager().GetPane("LogPanel");
    q_value().Dock().Bottom().Layer(logs.dock_layer).Row(logs.dock_row).Position(logs.dock_pos + 1);
    ui.Manager().Update();
    const int shared_height = ui.DockSize("LogPanel");
    ui.Settle();
    CHECK(ui.DockSize("EvalQValuePanel") == shared_height);
    CHECK(ui.DockSize("LogPanel") == shared_height);
    ui.CheckNoMainPaneFallback();
}

TEST_CASE("Common Panel redraws its whole stopped scene on resize with either style", "[.][prd110_main_pane][prd110_panel_repaint]")
{
    MainPaneFixture ui;
    ui.Start();
    for (bool explicit_style : {false, true}) {
        INFO("explicit_style=" << explicit_style);
        auto* frame = new wxFrame(nullptr, wxID_ANY, "Panel resize regression");
        auto* panel = explicit_style
            ? new anet::rl::gui::Panel(frame, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0)
            : new anet::rl::gui::Panel(frame);
        panel->SetBackgroundStyle(wxBG_STYLE_PAINT);
        panel->Bind(wxEVT_PAINT, [panel](wxPaintEvent&) {
            wxAutoBufferedPaintDC dc(panel);
            dc.SetBackground(*wxBLACK_BRUSH);
            dc.Clear();
            dc.SetPen(*wxWHITE_PEN);
            const auto size = panel->GetClientSize();
            dc.DrawLine(20, size.y - 25, size.x - 20, size.y - 25);
            dc.DrawLine(size.x - 25, 20, size.x - 25, size.y - 20);
        });
        frame->ShowWithoutActivating();
        ui.Pump();
        for (int direction = 0; direction < 3; ++direction) {
            INFO("axis=" << direction);
            frame->SetClientSize(400, 300);
            ui.Pump();
            panel->Refresh(false);
            panel->Update();
            for (int step = 1; step <= 8; ++step) {
                frame->SetClientSize(400 + (direction == 0 ? 0 : step * 12),
                    300 + (direction == 1 ? 0 : step * 12));
                ui.Pump();
            }
            const auto capture = [panel] {
                const auto size = panel->GetClientSize();
                wxBitmap bitmap(size.x, size.y, 24);
                wxMemoryDC target(bitmap);
                wxClientDC source(panel);
                target.Blit(0, 0, size.x, size.y, &source, 0, 0);
                target.SelectObject(wxNullBitmap);
                return bitmap.ConvertToImage();
            };
            const auto actual = capture();
            panel->Refresh(false);
            panel->Update();
            ui.Pump();
            const auto expected = capture();
            REQUIRE(actual.GetSize() == expected.GetSize());
            CHECK(std::equal(actual.GetData(), actual.GetData()
                + actual.GetWidth() * actual.GetHeight() * 3, expected.GetData()));
        }
        frame->Destroy();
        ui.Pump();
    }
}
