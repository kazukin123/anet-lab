#include "RunnerFrame.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <wx/artprov.h>
#include <wx/aui/floatpane.h>
#include <wx/dc.h>
#include <wx/display.h>
#include <wx/filedlg.h>
#include <wx/settings.h>
#include <wx/textdlg.h>
#include <wx/utils.h>
#include "anet/exception.hpp"
#include "anet/log.hpp"
#include "anet/observers.hpp"
#include "anet/str_util.hpp"
#include "RunnerApp.hpp"
#include "LogPanel.hpp"
#include "TrainPanel.hpp"
#include "HeatMapPanel.hpp"
#include "Conv2dPanel.hpp"
#include "ErrorDialog.hpp"

namespace LOG = anet::log;

// Runner 画面のレイアウト定数。
// dock サイズ制御の機構そのものは基底 anet::rl::gui::AuiLayoutFrame (docs/adr/0016) が担う。
namespace runner_layout {

constexpr int kTrainEvalLayer = 0;
constexpr int kQValueLayer = 10;
constexpr int kAuxLayer = 20;
// 上端のtoolbarは左右の列より外側に置き、全幅に並べる (wx の既定 10 では補助列が帯の幅を削る)
constexpr int kToolBarLayer = 30;
constexpr int kDefaultAuxDockWidth = 400;
constexpr int kDefaultLogDockHeight = 200;
constexpr int kMinTrainWidth = 200;
constexpr int kMinMainPaneHeight = 200;
constexpr int kMinEvalWidth = 200;
// 同じ列で Q-Values が余りを取らない伸び率 (wx の既定は 100000。0 は wx が既定へ戻す)
constexpr int kQValueDockProportion = 1;
constexpr int kMinAuxWidth = 100;
const wxString kCentreCaptionSuffix = wxS(" [Centre]");
const wxSize kToolBitmapSize(16, 16);

}  // namespace runner_layout

using namespace runner_layout;


enum {
    ID_ResetLayout = wxID_HIGHEST + 1,
    ID_LogView,
    ID_LogLevelInfo,
    ID_LogLevelVerbose,
    ID_LogLevelWarn,
    ID_LogLevelError,
    ID_AlwaysOnTopOff,
    ID_AlwaysOnTopAlways,
    ID_AlwaysOnTopWhileRunning,
    ID_TrainPanel,
    ID_EvalPanel,
    ID_QValuePanel,
    ID_EvalTracePanel,
    ID_HeatMap,
    ID_Conv2d,
    ID_TrainToggle,
    ID_EvalToggle,
    ID_EvalStep,
    ID_SaveAgent,
    ID_OpenRunFolder,
    ID_TrainFpsConfig,
    ID_TrainFpsOff,
    ID_TrainFps0_1,
    ID_TrainFps0_5,
    ID_TrainFps1,
    ID_TrainFps5,
    ID_TrainFps10,
    ID_TrainFps15,
    ID_TrainFps30,
    ID_TrainFps60,
    ID_EvalFpsConfig,
    ID_EvalFps0_1,
    ID_EvalFps0_5,
    ID_EvalFps1,
    ID_EvalFps5,
    ID_EvalFps10,
    ID_EvalFps15,
    ID_EvalFps30,
    ID_EvalFps60,
    ID_EvalFps120,
    ID_EvalStepsConfig,
    ID_EvalSteps1,
    ID_EvalSteps2,
    ID_EvalSteps4,
    ID_EvalSteps10,
    ID_EvalSteps20,
    ID_EvalSteps50,
    ID_EvalSteps100,
    ID_EvalStepsCustom,
};

namespace {

class PanelToolBarArtProvider final : public wxAuiDefaultToolBarArt {
public:
    wxAuiToolBarArt* Clone() override
    {
        return new PanelToolBarArtProvider(*this);
    }

    void DrawButton(wxDC& dc, wxWindow* window, const wxAuiToolBarItem& item, const wxRect& rect) override
    {
        // 標準の無効文字色を保ち、表示中のpaneを示す選択枠だけは操作可否と独立に描く。
        wxAuiDefaultToolBarArt::DrawButton(dc, window, item, rect);
        const int disabled_checked = wxAUI_BUTTON_STATE_DISABLED | wxAUI_BUTTON_STATE_CHECKED;
        if ((item.GetState() & disabled_checked) == disabled_checked) {
            wxDCPenChanger pen(dc, wxPen(wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT)));
            wxDCBrushChanger brush(dc, *wxTRANSPARENT_BRUSH);
            dc.DrawRectangle(rect);
        }
    }
};

} // namespace

namespace runner_frame_detail {

class RunnerDockArt final : public wxAuiDefaultDockArt {
public:
    RunnerDockArt()
    {
        UpdateColoursFromSystem();
    }

    wxAuiDockArt* Clone() override
    {
        return new RunnerDockArt(*this);
    }

    void UpdateColoursFromSystem() override
    {
        // 標準テーマを反映した後、非アクティブ pane 名だけを読みやすいシステム文字色へ揃える。
        wxAuiDefaultDockArt::UpdateColoursFromSystem();
        SetColour(wxAUI_DOCKART_INACTIVE_CAPTION_TEXT_COLOUR,
            wxSystemSettings::GetColour(wxSYS_COLOUR_BTNTEXT));
    }
};

wxString FormatFpsConfigLabel(float fps)
{
    return wxString::Format("Config (%g)", static_cast<double>(fps));
}

wxString FormatRate(const std::optional<float>& rate)
{
    if (!rate.has_value() || !std::isfinite(*rate) || *rate < 0.0f) {
        return "-";
    }
    return wxString::FromUTF8(anet::FormatWithCommas(std::llround(*rate)));
}

wxString FormatStepRates(const wxString& exp_rate, const wxString& train_rate)
{
    return "exp " + exp_rate + " steps/s    train " + train_rate + " steps/s";
}

wxString FormatElapsed(
    const std::optional<float>& elapsed_hour,
    std::chrono::steady_clock::time_point captured_at)
{
    if (!elapsed_hour.has_value() || !std::isfinite(*elapsed_hour) || *elapsed_hour < 0.0f) {
        return "--:--:--";
    }

    const double captured_seconds = static_cast<double>(*elapsed_hour) * 3600.0;
    const double extrapolated_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - captured_at).count();
    const int64_t total_seconds = static_cast<int64_t>(
        std::max(0.0, captured_seconds + extrapolated_seconds));
    const int64_t hours = total_seconds / 3600;
    const int64_t minutes = (total_seconds / 60) % 60;
    const int64_t seconds = total_seconds % 60;
    return wxString::Format("%lld:%02lld:%02lld",
        static_cast<long long>(hours), static_cast<long long>(minutes), static_cast<long long>(seconds));
}

void ShowUiOperationError(
    const wxString& operation,
    const std::filesystem::path& path,
    const wxString& reason,
    const wxString& detail = wxEmptyString)
{
    const wxString message = operation + " failed.\npath=" + wxString(path.wstring())
        + "\nreason=" + reason;
    ReportError(message, detail, wxGetApp().ShouldShowErrorDialog());
}

}  // namespace runner_frame_detail

using namespace runner_frame_detail;

// Logs ツールの説明。未確認の WARN/ERROR があるときは件数を後ろに足す。
constexpr const char* kLogViewHelp = "Show/hide Logs";


RunnerFrame::RunnerFrame(const wxString& title, const TrainPanelConfig& train_panel_config,
    const EvalPanelConfig& eval_panel_config, MainView main_view)
    : anet::rl::gui::AuiLayoutFrame(title, wxSize(1024, 1024)),
    conv2d_config_(wxGetApp().GetConfigData()),
    main_view_(main_view), train_config_fps_(train_panel_config.fps), eval_config_fps_(eval_panel_config.fps),
    eval_config_step_per_frame_(eval_panel_config.step_per_frame)
{
    // pane の dock/float とテーマ変更の両方で Runner 用 caption 配色を維持する。
    aui_mgr_.SetArtProvider(new RunnerDockArt());

    // フラグ設定
    aui_mgr_.SetFlags(wxAUI_MGR_ALLOW_FLOATING | wxAUI_MGR_TRANSPARENT_DRAG | wxAUI_MGR_TRANSPARENT_HINT);

    // 既定の30%制約だと右側の複数列表示が狭くなるため、補助列を含めて広げられるようにする。
    aui_mgr_.SetDockSizeConstraint(0.85, 0.3);

    // 初期サイズを作業領域 (タスクバーを除く) に収める。1920x1080 + タスクバーでは高さ 1024 がはみ出すため。
    // SetupPanes が client size から pane 幅を決めるので、それより前に確定させる。
    wxSize initial_size = GetSize();
    initial_size.DecTo(wxDisplay(this).GetClientArea().GetSize());
    SetSize(initial_size);

    // 画面レイアウトを作る (SetupPanes は client size を読むため、メニュー/ステータスバーより後)
    SetupMenuBar();
    CreateStatusBar();
    SetupToolBars();
    SetupPanes(train_panel_config, eval_panel_config);

    // イベントハンドラ登録
    SetupEvents();
    wxUpdateUIEvent::SetUpdateInterval(200);

    // AUIレイアウトを反映
    aui_mgr_.Update();
    ApplyLayoutPolicy();

    // 初期ログレベル（LogPanel が設定から読んだもの）に合わせてメニューのチェック状態を更新
    switch (log_panel_->GetLogLevel()) {
    case wxLOG_Info: GetMenuBar()->Check(ID_LogLevelVerbose, true); break;
    case wxLOG_Warning: GetMenuBar()->Check(ID_LogLevelWarn, true); break;
    case wxLOG_Error: GetMenuBar()->Check(ID_LogLevelError, true); break;
    default: GetMenuBar()->Check(ID_LogLevelInfo, true); break;
    }

    // ウィンドウ表示
    Centre();
}

void RunnerFrame::SetupMenuBar()
{
    wxMenuBar* menu_bar = new wxMenuBar;

    // File Menu
    wxMenu* file_menu = new wxMenu;
    file_menu->Append(wxID_EXIT);
    menu_bar->Append(file_menu, "&File");

    // View Menu
    wxMenu* view_menu = new wxMenu;
    view_menu->Append(ID_ResetLayout, "&Reset Layout", "Reset to default layout");

    // topmost モードは起動中だけ保持し、毎回 Off から開始する。
    wxMenu* always_on_top_menu = new wxMenu;
    always_on_top_menu->AppendRadioItem(ID_AlwaysOnTopOff, "&Off")->Check(true);
    always_on_top_menu->AppendRadioItem(ID_AlwaysOnTopAlways, "&Always");
    always_on_top_menu->AppendRadioItem(ID_AlwaysOnTopWhileRunning, "&While Running");
    view_menu->AppendSubMenu(always_on_top_menu, "Always on &Top");

    // 区切り
    view_menu->AppendSeparator();

    // ログViewメニューの追加
    view_menu->AppendCheckItem(ID_LogView, "&Log View")->Check(true);

    // ログレベルメニューの追加
    wxMenu* log_level_menu = new wxMenu;
    log_level_menu->AppendRadioItem(ID_LogLevelError, "&Error");
    log_level_menu->AppendRadioItem(ID_LogLevelWarn, "&Warn");
    log_level_menu->AppendRadioItem(ID_LogLevelInfo, "&Info");
    log_level_menu->AppendRadioItem(ID_LogLevelVerbose, "&Verbose");
    view_menu->AppendSubMenu(log_level_menu, "L&og Level");

    // 区切り
    view_menu->AppendSeparator();

    // Eval View関連メニュー項目
    view_menu->AppendCheckItem(ID_TrainPanel, "&Train View");
    view_menu->AppendCheckItem(ID_EvalPanel, "&Evaluation View")->Check(false);
    view_menu->AppendCheckItem(ID_QValuePanel, "&Evaluation QValue View")->Check(false);
    view_menu->AppendCheckItem(ID_EvalTracePanel, "Evaluation Trace")->Check(false);

    // 区切り
    view_menu->AppendSeparator();

    // Train View FPS
    // FPS は実行時 UI 操作であり、起動時 config 自体は変更しない。
    wxMenu* train_fps_menu = new wxMenu;
    train_fps_menu->AppendRadioItem(ID_TrainFpsConfig, FormatFpsConfigLabel(train_config_fps_))->Check(true);
    train_fps_menu->AppendRadioItem(ID_TrainFpsOff, "0 (Off)");
    train_fps_menu->AppendRadioItem(ID_TrainFps0_1, "0.1");
    train_fps_menu->AppendRadioItem(ID_TrainFps0_5, "0.5");
    train_fps_menu->AppendRadioItem(ID_TrainFps1, "1");
    train_fps_menu->AppendRadioItem(ID_TrainFps5, "5");
    train_fps_menu->AppendRadioItem(ID_TrainFps10, "10");
    train_fps_menu->AppendRadioItem(ID_TrainFps15, "15");
    train_fps_menu->AppendRadioItem(ID_TrainFps30, "30");
    train_fps_menu->AppendRadioItem(ID_TrainFps60, "60");
    view_menu->AppendSubMenu(train_fps_menu, "Train View FPS");

    // Eval View FPS
    wxMenu* eval_fps_menu = new wxMenu;
    eval_fps_menu->AppendRadioItem(ID_EvalFpsConfig, FormatFpsConfigLabel(eval_config_fps_))->Check(true);
    eval_fps_menu->AppendRadioItem(ID_EvalFps0_1, "0.1");
    eval_fps_menu->AppendRadioItem(ID_EvalFps0_5, "0.5");
    eval_fps_menu->AppendRadioItem(ID_EvalFps1, "1");
    eval_fps_menu->AppendRadioItem(ID_EvalFps5, "5");
    eval_fps_menu->AppendRadioItem(ID_EvalFps10, "10");
    eval_fps_menu->AppendRadioItem(ID_EvalFps15, "15");
    eval_fps_menu->AppendRadioItem(ID_EvalFps30, "30");
    eval_fps_menu->AppendRadioItem(ID_EvalFps60, "60");
    eval_fps_menu->AppendRadioItem(ID_EvalFps120, "120");
    view_menu->AppendSubMenu(eval_fps_menu, "Eval View FPS");

    // 起動時設定へ戻す項目と任意入力を含め、選択元を実行中だけ保持する。
    wxMenu* eval_steps_menu = new wxMenu;
    eval_steps_menu->AppendRadioItem(ID_EvalStepsConfig,
        wxString::Format("Config (%d)", eval_config_step_per_frame_))->Check(true);
    eval_steps_selection_ = ID_EvalStepsConfig;
    eval_steps_menu->AppendRadioItem(ID_EvalSteps1, "1");
    eval_steps_menu->AppendRadioItem(ID_EvalSteps2, "2");
    eval_steps_menu->AppendRadioItem(ID_EvalSteps4, "4");
    eval_steps_menu->AppendRadioItem(ID_EvalSteps10, "10");
    eval_steps_menu->AppendRadioItem(ID_EvalSteps20, "20");
    eval_steps_menu->AppendRadioItem(ID_EvalSteps50, "50");
    eval_steps_menu->AppendRadioItem(ID_EvalSteps100, "100");
    eval_steps_menu->AppendRadioItem(ID_EvalStepsCustom, "Custom...");
    view_menu->AppendSubMenu(eval_steps_menu, "Eval Steps per Frame");

    // 区切り
    view_menu->AppendSeparator();

    // その他Viewメニュー項目
    view_menu->Append(ID_HeatMap, "&HeatMap");
    view_menu->Append(ID_Conv2d, "&Conv2d");
    menu_bar->Append(view_menu, "&View");

    // Help Menu
    wxMenu* help_menu = new wxMenu;
    help_menu->Append(wxID_ABOUT);
    menu_bar->Append(help_menu, "&Help");

    SetMenuBar(menu_bar);
}

void RunnerFrame::CreateStatusBar()
{
    // ステータスバーを作成し、頻繁に変わる表示欄の幅を固定する。
    wxStatusBar* statusBar = wxFrame::CreateStatusBar(3);
    const int rate_width =
        statusBar->GetTextExtent(FormatStepRates("1,234,567", "1,234,567")).GetWidth() + 16;
    const int elapsed_width = statusBar->GetTextExtent("999:59:59").GetWidth() + 16;
    const int widths[] = {-1, rate_width, elapsed_width};
    statusBar->SetStatusWidths(3, widths);

    // wxAuiToolBar の hover は wxFrame::DoGiveHelp 経由で help pane を書き換え、tool から
    // 離れるときに hover 直前の文字列へ戻す (auibar.cpp:1519-1528)。この復元が toolbar 操作で
    // 出した pause/resume メッセージを消すため、help pane を無効化して field 0 を操作結果専用にする。
    SetStatusBarPane(-1);
    statusBar->SetStatusText("Ready", 0);
    statusBar->SetStatusText(FormatStepRates("-", "-"), 1);
    statusBar->SetStatusText("--:--:--", 2);
}

void RunnerFrame::SetupToolBars()
{
    // ToolbarPaneのgripperはAddPane時にwxAuiToolBar内蔵側へ移される。
    // toolbar artと各state色も含め、wxWidgets標準の表現に任せる。
    const long style = wxAUI_TB_DEFAULT_STYLE | wxAUI_TB_HORZ_TEXT;

    // Run 制御は 1 本にまとめ、Train と Eval 系 (toggle + 手動 step) を separator で分ける。
    run_control_toolbar_ = new wxAuiToolBar(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, style);
    run_control_toolbar_->SetToolBitmapSize(kToolBitmapSize);
    run_control_toolbar_->SetOverflowVisible(false);
    // 生成時は停止状態の再生アイコンを置き、実行状態に応じた差し替えは UpdateToggleBitmap が行う。
    const auto text_colour = wxSystemSettings::GetColour(wxSYS_COLOUR_BTNTEXT);
    const auto play_bitmap = anet::rl::gui::MakePlaybackBitmap(anet::rl::gui::PlaybackIcon::Play, text_colour);
    run_control_toolbar_->AddTool(ID_TrainToggle, "Train", play_bitmap,
        "Pause/resume training (Shift / left-click)", wxITEM_CHECK);
    run_control_toolbar_->AddSeparator();
    run_control_toolbar_->AddTool(ID_EvalToggle, "Eval", play_bitmap,
        "Pause/resume evaluation (Space / right-click)", wxITEM_CHECK);
    run_control_toolbar_->AddTool(ID_EvalStep, "Step", anet::rl::gui::MakePlaybackBitmap(anet::rl::gui::PlaybackIcon::StepForward, text_colour),
        "Advance evaluation by one step (Ctrl)");
    run_control_toolbar_->EnableTool(ID_TrainToggle, false);
    run_control_toolbar_->EnableTool(ID_EvalToggle, false);
    run_control_toolbar_->EnableTool(ID_EvalStep, false);
    run_control_toolbar_->Realize();

    // step名はtoolbar上のlabel、値は選択・コピー可能な標準read-only text controlで表示する。
    step_toolbar_ = new wxAuiToolBar(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, style);
    step_toolbar_->SetOverflowVisible(false);
    auto* exp_label = new wxStaticText(step_toolbar_, wxID_ANY, "exp");
    step_toolbar_->AddControl(exp_label);
    exp_step_text_ = new wxTextCtrl(step_toolbar_, wxID_ANY, "-",
        wxDefaultPosition, wxDefaultSize, wxTE_READONLY | wxTE_RIGHT);
    exp_step_text_->SetMinSize(wxSize(
        exp_step_text_->GetTextExtent("1,234,567,890").GetWidth() + 16, -1));
    step_toolbar_->AddControl(exp_step_text_);

    step_toolbar_->AddSeparator();
    auto* train_label = new wxStaticText(step_toolbar_, wxID_ANY, "train");
    step_toolbar_->AddControl(train_label);
    train_step_text_ = new wxTextCtrl(step_toolbar_, wxID_ANY, "-",
        wxDefaultPosition, wxDefaultSize, wxTE_READONLY | wxTE_RIGHT);
    train_step_text_->SetMinSize(wxSize(
        train_step_text_->GetTextExtent("1,234,567,890").GetWidth() + 16, -1));
    step_toolbar_->AddControl(train_step_text_);
    step_toolbar_->Realize();

    // Run 成果物の操作は標準 art を使い、テーマ側の表現に合わせる。
    run_ops_toolbar_ = new wxAuiToolBar(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, style);
    run_ops_toolbar_->SetToolBitmapSize(kToolBitmapSize);
    run_ops_toolbar_->SetOverflowVisible(false);
    run_ops_toolbar_->AddTool(ID_SaveAgent, wxEmptyString,
        wxArtProvider::GetBitmapBundle(wxART_FILE_SAVE, wxART_TOOLBAR, kToolBitmapSize), "Save Checkpoint");
    run_ops_toolbar_->AddTool(ID_OpenRunFolder, wxEmptyString,
        wxArtProvider::GetBitmapBundle(wxART_FOLDER_OPEN, wxART_TOOLBAR, kToolBitmapSize), "Open Run Folder");
    run_ops_toolbar_->EnableTool(ID_SaveAgent, false);
    run_ops_toolbar_->Realize();

    // Panel 表示ツールは View menu と同じ ID を共有し、同じ pane 操作へ到達させる。
    panel_toolbar_ = new wxAuiToolBar(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, style);
    panel_toolbar_->SetArtProvider(new PanelToolBarArtProvider());
    panel_toolbar_->SetOverflowVisible(false);
    panel_toolbar_->AddTool(ID_LogView, "Logs", wxBitmapBundle(), kLogViewHelp, wxITEM_CHECK);
    panel_toolbar_->AddTool(ID_TrainPanel, "Train View", wxBitmapBundle(), "Show/hide Train View", wxITEM_CHECK);
    panel_toolbar_->AddTool(ID_EvalPanel, "Eval View", wxBitmapBundle(), "Show/hide Eval View", wxITEM_CHECK);
    panel_toolbar_->AddTool(ID_QValuePanel, "Q-Values", wxBitmapBundle(), "Show/hide Q-Values", wxITEM_CHECK);
    panel_toolbar_->AddTool(ID_EvalTracePanel, "Trace", wxBitmapBundle(), "Show/hide Evaluation Trace", wxITEM_CHECK);
    panel_toolbar_->Realize();

    // float 時のミニフレームは pane caption を window title に使うため、name とは別に caption を与える。
    struct ToolBarPaneDef {
        wxAuiToolBar* toolbar;
        wxString name;
        wxString caption;
    };
    const ToolBarPaneDef toolbars[] = {
        {.toolbar = run_control_toolbar_, .name = "RunControlToolBar", .caption = "Run Control"},
        {.toolbar = step_toolbar_, .name = "StepToolBar", .caption = "Steps"},
        {.toolbar = run_ops_toolbar_, .name = "RunOpsToolBar", .caption = "Run Operations"},
        {.toolbar = panel_toolbar_, .name = "PanelToolBar", .caption = "Panels"},
    };
    for (int position = 0; position < static_cast<int>(std::size(toolbars)); ++position) {
        aui_mgr_.AddPane(toolbars[position].toolbar, wxAuiPaneInfo()
            .Name(toolbars[position].name).Caption(toolbars[position].caption)
            .ToolbarPane().Top().Layer(kToolBarLayer).Row(0).Position(position)
            .CloseButton(false));
    }
}

void RunnerFrame::SetupPanes(const TrainPanelConfig& train_panel_config, const EvalPanelConfig& eval_panel_config)
{
    const int initial_client_width = GetClientSize().GetWidth();
    const int main_pane_width = std::max(kMinTrainWidth, initial_client_width / 3);

    // Log View
    log_panel_ = new LogPanel(this, wxGetApp().GetConfigData());
    wxAuiPaneInfo log_info = wxAuiPaneInfo()
        .Name("LogPanel").Caption("Logs")
        .Bottom().Layer(kTrainEvalLayer).Row(0).Position(0)
        .BestSize(400, kDefaultLogDockHeight) // ドッキング時の推奨サイズ
        .FloatingSize(800, 400)     // 切り離したときのウィンドウサイズ
        .MinSize(200, 200)          // これ以上小さくならないようにする
        .CloseButton(true).MaximizeButton(true).MinimizeButton(false).PinButton(true);
    log_info.dock_size = kDefaultLogDockHeight;
    aui_mgr_.AddPane(log_panel_, log_info);

    // TrainExperienceView
    train_panel_ = new TrainPanel(this, train_panel_config);
    aui_mgr_.AddPane(train_panel_, wxAuiPaneInfo()
        .Name("TrainExperiencePanel").Caption("Train View")
        .Centre()
        .BestSize(main_pane_width, 400)
        .MinSize(200, 200)
        .CloseButton(false).MaximizeButton(true).MinimizeButton(false).PinButton(false)
        .Show(main_view_ != MainView::Eval)
    );

    // EvalExperienceView
    eval_panel_ = new EvalPanel(this, eval_panel_config, [this] {
        if (eval_trace_panel_) eval_trace_panel_->ReturnToLive();
    });
    wxAuiPaneInfo eval_info = wxAuiPaneInfo()
        .Name("EvalExperiencePanel").Caption("Evaluation View")
        .Right().Layer(kTrainEvalLayer).Row(0).Position(0)
        .BestSize(main_pane_width, 400)
        .MinSize(200, 200)
        .CloseButton(true).MaximizeButton(true).MinimizeButton(false).PinButton(false)
        .Show(main_view_ != MainView::Train);
    if (main_view_ == MainView::Eval) eval_info.Centre();
    eval_info.dock_size = main_pane_width;
    aui_mgr_.AddPane(eval_panel_, eval_info);

    // QValuePanel
    auto config_data = wxGetApp().GetConfigData();
    q_value_panel_ = new QValuePanel(this, config_data);
    wxAuiPaneInfo q_value_info = wxAuiPaneInfo()
        .Name("EvalQValuePanel").Caption("Evaluation Q-Values")
        .Right().Layer(kQValueLayer).Row(0).Position(0)
        .CloseButton(true).MaximizeButton(true).MinimizeButton(false).PinButton(true)
        //.DestroyOnClose(false) // デフォルトで、✕ボタンPanelを消しても非表示になるだけ
        .Hide();
    ApplyQValuePaneSize(q_value_info);
    q_value_info.dock_size = q_value_info.best_size.x;
    aui_mgr_.AddPane(q_value_panel_, q_value_info);

    // 履歴は右の補助列に一枚だけ置き、起動時は生成・記録とも休止する。
    eval_trace_panel_ = new EvalTracePanel(*this, aui_mgr_, *eval_panel_, config_data);
    AddAuxPane(eval_trace_panel_, wxAuiPaneInfo().Name("EvalTracePanel").Caption("Evaluation Trace")
        .CloseButton(true).MaximizeButton(true).PinButton(true).Hide());
}

void RunnerFrame::Initialize(std::shared_ptr<anet::rl::RunManager> run_manager)
{
    // TrainPanel初期化
    train_panel_->Initialize(run_manager);

    // EvalPanel初期化
    const auto& eval_config_tag = eval_panel_->GetConfig().eval_config_tag;
    eval_runner_ = run_manager->CreateEvalRunner(
        "EvalPanel", eval_config_tag);
    eval_panel_->Initialize(run_manager, eval_runner_);

    // QValuePanel初期化
    q_value_panel_->Initialize(run_manager, eval_runner_);
    q_value_panel_->SetActionHandler([this](int64_t action) {
        eval_panel_->DoStep(action);
        eval_panel_->Refresh();
    });
    // 行動数と行動名が決まったので、pane の大きさを表に合わせ直す
    auto& q_value_pane = aui_mgr_.GetPane(q_value_panel_);
    ApplyQValuePaneSize(q_value_pane);
    q_value_pane.dock_size = q_value_pane.best_size.x;
    if (q_value_pane.IsShown()) aui_mgr_.Update();
    // 長い値で表が広がったときも当て直す。panel の更新処理の中から呼ばれるので一拍置く
    q_value_panel_->SetDockSizeHandler([this] {
        CallAfter([this] {
            if (aui_mgr_.GetManagedWindow() == nullptr) return;
            auto& pane = aui_mgr_.GetPane(q_value_panel_);
            ApplyQValuePaneSize(pane);
            if (pane.IsShown()) aui_mgr_.Update();
        });
    });

    eval_trace_panel_->Initialize(run_manager, eval_runner_);
    eval_trace_panel_->RegisterFollower(eval_panel_);
    eval_trace_panel_->RegisterFollower(q_value_panel_);

    // Trainer thread 上で status snapshot を作り、UI から可変 Runner 状態を直読しない。
    AttachTrainStatusObserver(run_manager);
    initialized_ = true;
}

void RunnerFrame::AttachTrainStatusObserver(const std::shared_ptr<anet::rl::RunManager>& run_manager)
{
    auto train_runner = run_manager->GetTrainRunner();
    auto observer = std::make_shared<anet::rl::FunctionTrainObserver>(
        [this, train_runner](const anet::rl::TrainEvent& event) {
            if (!train_status_store_.ShouldUpdate()) return;

            const TrainStatusSnapshot snapshot{
                .counts = event.counts,
                .exp_sps = train_runner->GetScalar(anet::rl::Runner::EXP_STEP_PER_SEC),
                .train_sps = train_runner->GetScalar(anet::rl::Runner::TRAIN_STEP_PER_SEC),
                .elapsed_hour = train_runner->GetScalar(anet::rl::Runner::ELAPSE_HOUR),
                .captured_at = std::chrono::steady_clock::now(),
            };
            train_status_store_.Update(snapshot);
        },
        "RunnerToolBar");

    // AttachScoped() の返り値ではなく、Notifier が実際に保持する wrapper を保持して detach する。
    train_status_notifier_ = run_manager->GetNotifier();
    train_status_observer_ = std::make_shared<anet::rl::RunnerScopedTrainObserver>(observer, train_runner);
    train_status_notifier_->Attach(train_status_observer_);
}

void RunnerFrame::DetachTrainStatusObserver()
{
    if (train_status_notifier_ && train_status_observer_) {
        train_status_notifier_->Detach(train_status_observer_);
    }
    train_status_observer_.reset();
    train_status_notifier_.reset();
}

void RunnerFrame::UpdateTrainStatus()
{
    // 最新 snapshot を取り込み、同じ断面から toolbar と status bar を更新する。
    if (auto snapshot = train_status_store_.Get()) {
        latest_train_status_ = std::move(snapshot);
    }

    wxString exp_step_text = "-";
    wxString train_step_text = "-";
    wxString rate_text = FormatStepRates("-", "-");
    wxString elapsed_text = "--:--:--";
    if (latest_train_status_) {
        exp_step_text = wxString::FromUTF8(
            anet::FormatWithCommas(latest_train_status_->counts.exp_step));
        train_step_text = wxString::FromUTF8(
            anet::FormatWithCommas(latest_train_status_->counts.train_step));
        rate_text = FormatStepRates(
            FormatRate(latest_train_status_->exp_sps), FormatRate(latest_train_status_->train_sps));
        elapsed_text = FormatElapsed(
            latest_train_status_->elapsed_hour, latest_train_status_->captured_at);
    }

    // 値が変化した場合だけ描画更新を要求する。
    if (exp_step_text_ && exp_step_text_->GetValue() != exp_step_text) {
        exp_step_text_->ChangeValue(exp_step_text);
    }
    if (train_step_text_ && train_step_text_->GetValue() != train_step_text) {
        train_step_text_->ChangeValue(train_step_text);
    }
    if (GetStatusBar()->GetStatusText(1) != rate_text) {
        SetStatusText(rate_text, 1);
    }
    if (GetStatusBar()->GetStatusText(2) != elapsed_text) {
        SetStatusText(elapsed_text, 2);
    }
}

void RunnerFrame::UpdateToggleBitmap(
    wxAuiToolBar* toolbar, int tool_id, bool running, std::optional<bool>& shown_as_running)
{
    // 走行中は次に起きる操作 (一時停止)、停止中は再生を示す。無効時の淡色表示も wxAUI が
    // この bitmap から都度生成するため、差し替えはこの 1 箇所で足りる。
    if (toolbar == nullptr) return;

    // 200ms 周期で呼ばれるため、表示が実状態と一致している間は再描画を要求しない。
    if (shown_as_running.has_value() && *shown_as_running == running) return;

    const auto icon = running ? anet::rl::gui::PlaybackIcon::Pause : anet::rl::gui::PlaybackIcon::Play;
    toolbar->SetToolBitmap(tool_id, anet::rl::gui::MakePlaybackBitmap(icon,
        wxSystemSettings::GetColour(wxSYS_COLOUR_BTNTEXT)));
    shown_as_running = running;
    toolbar->Refresh(false);
}

void RunnerFrame::UpdateLogAlertBadge()
{
    // Logs pane の未確認の WARN/ERROR を、pane を閉じていても Panels toolbar の Logs に印で示す。
    if (panel_toolbar_ == nullptr || log_panel_ == nullptr) return;

    // 200ms 周期で呼ばれるため、件数が変わったときだけ差し替える。
    const auto alerts = log_panel_->GetUnviewedAlerts();
    if (shown_log_alerts_ == alerts) return;
    const bool had_badge = shown_log_alerts_.has_value() && shown_log_alerts_->Total() > 0;
    const bool has_badge = alerts.Total() > 0;
    shown_log_alerts_ = alerts;

    if (has_badge) {
        const wxArtID icon = alerts.errors > 0 ? wxART_ERROR : wxART_WARNING;
        panel_toolbar_->SetToolBitmap(ID_LogView, wxArtProvider::GetBitmapBundle(icon, wxART_TOOLBAR, kToolBitmapSize));
        panel_toolbar_->SetToolShortHelp(ID_LogView, wxString(kLogViewHelp) + " (" + alerts.Describe() + ")");
    } else {
        panel_toolbar_->SetToolBitmap(ID_LogView, wxBitmapBundle());
        panel_toolbar_->SetToolShortHelp(ID_LogView, kLogViewHelp);
    }
    if (had_badge == has_badge) {
        panel_toolbar_->Refresh(false);
        return;
    }

    // 印の有無で Logs の幅が変わる。toolbar の idle 更新の外で測り直し、pane を並べ直す。
    // OnClose が AUI の管理を外した後は、削除を待つだけの frame なので並べ直さない。
    CallAfter([this] {
        if (aui_mgr_.GetManagedWindow() == nullptr) return;
        panel_toolbar_->Realize();
        panel_toolbar_->InvalidateBestSize();
        auto& pane = aui_mgr_.GetPane(panel_toolbar_);
        if (!pane.IsOk()) return;
        SyncDockSizesToPanes();
        pane.BestSize(panel_toolbar_->GetBestSize());
        aui_mgr_.Update();
        ApplyLayoutPolicy();
    });
}

void RunnerFrame::SetAlwaysOnTopMode(AlwaysOnTopMode mode)
{
    // 選択を起動中の状態へ反映し、次回の定期更新を待たずに topmost を切り替える。
    always_on_top_mode_ = mode;
    ApplyAlwaysOnTopMode();
}

void RunnerFrame::ApplyAlwaysOnTopMode()
{
    // While Running は描画頻度や pane 表示ではなく、既存の再生/一時停止状態だけで判定する。
    const bool train_active = wxGetApp().IsTrainingRunning() && !wxGetApp().IsTrainingPaused();
    const bool eval_active = initialized_ && eval_panel_ != nullptr && !eval_panel_->IsPaused();
    const bool should_stay_on_top = always_on_top_mode_ == AlwaysOnTopMode::Always
        || (always_on_top_mode_ == AlwaysOnTopMode::WhileRunning && (train_active || eval_active));

    // wxWidgets の window style を必要な場合だけ更新し、フォーカスや表示状態は変更しない。
    const long current_style = GetWindowStyleFlag();
    const bool stays_on_top = (current_style & wxSTAY_ON_TOP) != 0;
    if (stays_on_top == should_stay_on_top) return;

    SetWindowStyleFlag(should_stay_on_top
        ? current_style | wxSTAY_ON_TOP
        : current_style & ~wxSTAY_ON_TOP);
}

void RunnerFrame::UpdateToolBarBitmaps()
{
    if (run_control_toolbar_ == nullptr) return;

    // 現在の実行状態を保ったまま bitmap を作り直す。
    const bool train_running = train_toggle_running_.value_or(false);
    const bool eval_running = eval_toggle_running_.value_or(false);
    train_toggle_running_.reset();
    eval_toggle_running_.reset();
    UpdateToggleBitmap(run_control_toolbar_, ID_TrainToggle, train_running, train_toggle_running_);
    UpdateToggleBitmap(run_control_toolbar_, ID_EvalToggle, eval_running, eval_toggle_running_);
    run_control_toolbar_->SetToolBitmap(ID_EvalStep, anet::rl::gui::MakePlaybackBitmap(anet::rl::gui::PlaybackIcon::StepForward, wxSystemSettings::GetColour(wxSYS_COLOUR_BTNTEXT)));
    run_control_toolbar_->Realize();
    run_control_toolbar_->Refresh(false);
}

void RunnerFrame::SetupEvents()
{
    // Train終了イベント
    Bind(wxEVT_TRAINER_EXIT, [this](wxCommandEvent&) {
        LOG::info() << "Stop training requested. Exiting.";
        Close(true);    // Frameを閉じる
        });

    // UI基本イベント
    Bind(wxEVT_CLOSE_WINDOW, &RunnerFrame::OnClose, this);
    Bind(wxEVT_SYS_COLOUR_CHANGED, &RunnerFrame::OnSystemColourChanged, this);
    Bind(wxEVT_UPDATE_UI, &RunnerFrame::OnUpdateTrainStatus, this, GetId());

    // メニューイベント
    Bind(wxEVT_MENU, &RunnerFrame::OnExit, this, wxID_EXIT);
    Bind(wxEVT_MENU, &RunnerFrame::OnAbout, this, wxID_ABOUT);
    Bind(wxEVT_MENU, &RunnerFrame::OnResetLayout, this, ID_ResetLayout);
    Bind(wxEVT_MENU, &RunnerFrame::OnHeatMap, this, ID_HeatMap);
    Bind(wxEVT_MENU, &RunnerFrame::OnConv2d, this, ID_Conv2d);
    Bind(wxEVT_MENU, &RunnerFrame::OnToggleTraining, this, ID_TrainToggle);
    Bind(wxEVT_MENU, &RunnerFrame::OnToggleEval, this, ID_EvalToggle);
    Bind(wxEVT_MENU, &RunnerFrame::OnEvalStep, this, ID_EvalStep);
    Bind(wxEVT_MENU, &RunnerFrame::OnSaveAgent, this, ID_SaveAgent);
    Bind(wxEVT_MENU, &RunnerFrame::OnOpenRunFolder, this, ID_OpenRunFolder);

    // topmost モードの選択と radio 表示を同じ一時状態へ同期する。
    const std::pair<int, AlwaysOnTopMode> always_on_top_items[] = {
        {ID_AlwaysOnTopOff, AlwaysOnTopMode::Off},
        {ID_AlwaysOnTopAlways, AlwaysOnTopMode::Always},
        {ID_AlwaysOnTopWhileRunning, AlwaysOnTopMode::WhileRunning},
    };
    for (const auto& [id, mode] : always_on_top_items) {
        Bind(wxEVT_MENU, [this, mode](wxCommandEvent&) { SetAlwaysOnTopMode(mode); }, id);
        Bind(wxEVT_UPDATE_UI, [this, mode](wxUpdateUIEvent& event) {
            event.Check(always_on_top_mode_ == mode);
        }, id);
    }

    // パネル表示/非表示メニュー連動 (トグル・✕・チェック同期は基底が処理)
    RegisterPaneMenu(ID_LogView, log_panel_);
    RegisterPaneMenu(ID_TrainPanel, train_panel_);
    RegisterPaneMenu(ID_EvalPanel, eval_panel_);
    RegisterPaneMenu(ID_QValuePanel, q_value_panel_);
    RegisterPaneMenu(ID_EvalTracePanel, eval_trace_panel_);

    // Update UI は操作経路に依存せず、実状態から menu と toolbar を同期する。
    Bind(wxEVT_UPDATE_UI, [this](wxUpdateUIEvent& event) {
        const bool running = wxGetApp().IsTrainingRunning();
        const bool active = running && !wxGetApp().IsTrainingPaused();
        event.Enable(running);
        event.Check(active);
        UpdateToggleBitmap(run_control_toolbar_, ID_TrainToggle, active, train_toggle_running_);
    }, ID_TrainToggle);
    Bind(wxEVT_UPDATE_UI, [this](wxUpdateUIEvent& event) {
        const bool available = initialized_ && eval_panel_ != nullptr;
        const bool active = available && !eval_panel_->IsPaused();
        event.Enable(available);
        event.Check(active);
        UpdateToggleBitmap(run_control_toolbar_, ID_EvalToggle, active, eval_toggle_running_);
    }, ID_EvalToggle);
    Bind(wxEVT_UPDATE_UI, [this](wxUpdateUIEvent& event) {
        event.Enable(initialized_ && eval_panel_ != nullptr);
    }, ID_EvalStep);
    Bind(wxEVT_UPDATE_UI, [this](wxUpdateUIEvent& event) {
        event.Enable(initialized_);
    }, ID_SaveAgent);
    Bind(wxEVT_UPDATE_UI, [this](wxUpdateUIEvent& event) {
        event.Enable(initialized_);
    }, ID_HeatMap);

    const std::pair<int, wxWindow*> panel_bindings[] = {
        {ID_LogView, log_panel_},
        {ID_TrainPanel, train_panel_},
        {ID_EvalPanel, eval_panel_},
        {ID_QValuePanel, q_value_panel_},
        {ID_EvalTracePanel, eval_trace_panel_},
    };
    for (const auto& [id, window] : panel_bindings) {
        Bind(wxEVT_UPDATE_UI, [this, window](wxUpdateUIEvent& event) {
            auto& pane = aui_mgr_.GetPane(window);
            const bool open = anet::rl::gui::IsPaneOpen(pane, HasMaximizedPane());
            event.Enable(pane.IsOk() && (!open || CanHidePane(pane)));
            event.Check(open);
        }, id);
    }
    // 後から Bind した handler が先に呼ばれるため、Logs の印を更新してから上の表示状態の同期へ回す。
    Bind(wxEVT_UPDATE_UI, [this](wxUpdateUIEvent& event) {
        UpdateLogAlertBadge();
        event.Skip();
    }, ID_LogView);

    const std::pair<int, float> train_fps_items[] = {
        {ID_TrainFpsConfig, train_config_fps_}, {ID_TrainFpsOff, 0.0f}, {ID_TrainFps0_1, 0.1f},
        {ID_TrainFps0_5, 0.5f}, {ID_TrainFps1, 1.0f}, {ID_TrainFps5, 5.0f},
        {ID_TrainFps10, 10.0f}, {ID_TrainFps15, 15.0f}, {ID_TrainFps30, 30.0f}, {ID_TrainFps60, 60.0f},
    };
    for (const auto& [id, fps] : train_fps_items) {
        Bind(wxEVT_MENU, [this, fps](wxCommandEvent&) {
            train_panel_->SetFps(fps);
            LOG::info() << "Train view FPS changed fps=" << fps;
        }, id);
        Bind(wxEVT_UPDATE_UI, [this](wxUpdateUIEvent& event) { event.Enable(initialized_); }, id);
    }

    const std::pair<int, float> eval_fps_items[] = {
        {ID_EvalFpsConfig, eval_config_fps_}, {ID_EvalFps0_1, 0.1f}, {ID_EvalFps0_5, 0.5f},
        {ID_EvalFps1, 1.0f}, {ID_EvalFps5, 5.0f}, {ID_EvalFps10, 10.0f},
        {ID_EvalFps15, 15.0f}, {ID_EvalFps30, 30.0f}, {ID_EvalFps60, 60.0f}, {ID_EvalFps120, 120.0f},
    };
    for (const auto& [id, fps] : eval_fps_items) {
        Bind(wxEVT_MENU, [this, fps](wxCommandEvent&) {
            eval_panel_->SetFps(fps);
            eval_panel_->ReportSpeedChange();
        }, id);
        Bind(wxEVT_UPDATE_UI, [this](wxUpdateUIEvent& event) { event.Enable(initialized_); }, id);
    }

    const std::pair<int, int> eval_steps_items[] = {
        {ID_EvalStepsConfig, eval_config_step_per_frame_}, {ID_EvalSteps1, 1}, {ID_EvalSteps2, 2},
        {ID_EvalSteps4, 4}, {ID_EvalSteps10, 10}, {ID_EvalSteps20, 20},
        {ID_EvalSteps50, 50}, {ID_EvalSteps100, 100},
    };
    for (const auto& [id, steps] : eval_steps_items) {
        Bind(wxEVT_MENU, [this, id, steps](wxCommandEvent&) {
            eval_panel_->SetStepsPerFrame(steps);
            eval_panel_->ReportSpeedChange();
            eval_steps_selection_ = id;
            GetMenuBar()->Check(id, true);
        }, id);
    }
    Bind(wxEVT_MENU, &RunnerFrame::OnEvalStepsCustom, this, ID_EvalStepsCustom);
    Bind(wxEVT_UPDATE_UI, [this](wxUpdateUIEvent& event) {
        event.Enable(initialized_);
        event.Check(event.GetId() == eval_steps_selection_);
    }, ID_EvalStepsConfig, ID_EvalStepsCustom);

    // ログレベルメニュー
    Bind(wxEVT_MENU, [this](wxCommandEvent&) { if (log_panel_) log_panel_->SetLogLevel("info"); }, ID_LogLevelInfo);
    Bind(wxEVT_MENU, [this](wxCommandEvent&) { if (log_panel_) log_panel_->SetLogLevel("verbose"); }, ID_LogLevelVerbose);
    Bind(wxEVT_MENU, [this](wxCommandEvent&) { if (log_panel_) log_panel_->SetLogLevel("warn"); }, ID_LogLevelWarn);
    Bind(wxEVT_MENU, [this](wxCommandEvent&) { if (log_panel_) log_panel_->SetLogLevel("error"); }, ID_LogLevelError);

    // きーまう
    Bind(anet::rl::gui::EVT_FORWARDED_MOUSE, &RunnerFrame::OnMouse, this);
    Bind(anet::rl::gui::EVT_FORWARDED_KEY, &RunnerFrame::OnKey, this);
}

bool RunnerFrame::IsAuxPane(const wxAuiPaneInfo& pane) const
{
    return pane.name == "EvalTracePanel" || pane.name.StartsWith("HeatMapPanel_") || pane.name.StartsWith("Conv2dPanel_");
}

std::vector<wxAuiPaneInfo*> RunnerFrame::GetAuxPanes()
{
    std::vector<wxAuiPaneInfo*> result;
    auto& panes = aui_mgr_.GetAllPanes();
    for (size_t i = 0; i < panes.GetCount(); ++i) {
        auto& pane = panes.Item(i);
        if (IsAuxPane(pane)) {
            result.push_back(&pane);
        }
    }
    return result;
}

void RunnerFrame::ApplyQValuePaneSize(wxAuiPaneInfo& pane) const
{
    // 幅は表＋ヒートマップで、最小幅と既定幅を同じにする。
    // 高さは行動数ぶんで、同じ列ではほかの pane へ余りを渡す。ひとりの列では伸び率に関係なく列いっぱいになる。
    // サッシで広げた後の伸び率は wx に任せる。
    const wxSize size(q_value_panel_->GetPreferredDockWidth(), q_value_panel_->GetPreferredDockHeight());
    pane.BestSize(size);
    pane.dock_proportion = kQValueDockProportion;
    if (!pane.IsFloating()) {
        pane.MinSize(size);
        return;
    }
    // float の窓は利用者が直接大きさを決めるので、表が見える幅と見出し＋1 行の高さまで縮められるようにする。
    // wx は float にした時点の最小を窓へ写すため、panel と窓の最小もここで直す。
    const wxSize minimum = q_value_panel_->GetFloatingMinSize();
    pane.MinSize(minimum);
    q_value_panel_->SetMinSize(minimum);
    if (pane.frame) {
        pane.frame->SetMinClientSize(minimum);
        wxSize client = pane.frame->GetClientSize();
        if (client.x < minimum.x || client.y < minimum.y) {
            client.IncTo(minimum);
            pane.frame->SetClientSize(client);
        }
    }
}

void RunnerFrame::ApplyQValuePlacement()
{
    // float と dock で違う最小を当て、ひとりで入った上下の帯は行数ぶんの高さにする
    auto& pane = aui_mgr_.GetPane(q_value_panel_);
    ApplyQValuePaneSize(pane);
    if (!pane.IsFloating()) {
        aui_mgr_.Update();
        ApplyQValueBandHeight();
        return;
    }
    // 全行の高さが画面より高いときは、画面に収まる高さにする
    if (!pane.frame) return;
    const wxRect area = wxDisplay(pane.frame).GetClientArea();
    const wxRect rect = pane.frame->GetRect();
    if (rect.GetBottom() > area.GetBottom()) {
        pane.frame->SetSize(rect.width, std::max(pane.frame->GetMinSize().y, area.GetBottom() - rect.y + 1));
    }
}

bool RunnerFrame::IsQValuePlacementChanged()
{
    // 隠している間は比べない。閉じて開き直しただけでは当て直さず、サッシで変えた高さを残す。
    const auto& pane = aui_mgr_.GetPane(q_value_panel_);
    return pane.IsShown() && GetPanePlacement(pane) != q_value_placement_;
}

void RunnerFrame::ApplyQValueBandHeight()
{
    // 新しい上下の帯はpaneが覚えたdock_size（左右の列にいたときの幅）の高さで作られる。
    // ひとりで入った帯だけ行数ぶんへ当て直し、同居帯と左右の列はwxの配分を保つ。
    const auto& q_value_pane = aui_mgr_.GetPane(q_value_panel_);
    if (!q_value_pane.IsDocked()
        || (q_value_pane.dock_direction != wxAUI_DOCK_TOP && q_value_pane.dock_direction != wxAUI_DOCK_BOTTOM)) return;
    for (const auto& pane : aui_mgr_.GetAllPanes()) {
        if (pane.window != q_value_pane.window && pane.IsShown() && pane.IsDocked()
            && pane.dock_direction == q_value_pane.dock_direction && pane.dock_layer == q_value_pane.dock_layer
            && pane.dock_row == q_value_pane.dock_row) return;
    }
    auto snapshot = TakeLayoutSnapshot();
    auto* band = FindPaneLayout(snapshot, q_value_pane.name);
    if (!band) return;
    // 最小の高さを渡せば、wxがcaptionと枠を足した帯の最小まで広げる。
    band->dock_size = q_value_pane.min_size.y;
    ApplyLayoutSnapshot(std::move(snapshot));
}

int RunnerFrame::ResolveAuxDockWidth()
{
    // 実 dock 幅を pane 側へ同期してから、表示中→非表示の順で記憶幅を拾う
    SyncDockSizesToPanes();

    int hidden_width = 0;
    for (auto* pane : GetAuxPanes()) {
        if (!pane) continue;
        if (pane->IsShown() && pane->IsDocked() && pane->dock_direction == wxAUI_DOCK_RIGHT
            && pane->dock_size > 0) {
            return std::max(kMinAuxWidth, pane->dock_size);
        }
        if (hidden_width == 0 && pane->dock_size > 0) {
            hidden_width = pane->dock_size;
        }
    }
    if (hidden_width > 0) return std::max(kMinAuxWidth, hidden_width);
    return kDefaultAuxDockWidth;
}

void RunnerFrame::OnApplyLayoutPolicy()
{
    // 実際の表示を渡すので、最大化による一時非表示でも描画timerを止める。
    train_panel_->SetViewVisible(aui_mgr_.GetPane(train_panel_).IsShown());
    // フラグの宣言は最大化中も行う。幅の往復とは独立させる。
    if (UpdateMainPaneControls()) ScheduleMainPaneUpdate();

    // maximize 中は LoadLayout 往復が復元情報 (savedHiddenState) を壊すため何もしない。
    // restore 後に基底の restore ハンドラ経由で再適用される。
    if (HasMaximizedPane()) return;

    // 開閉フックを通らない遷移でも、占有規則を冪等に当て直す。
    if (ApplyMainPaneLayout(aui_mgr_.GetPane(train_panel_).IsShown(), aui_mgr_.GetPane(eval_panel_).IsShown())) {
        aui_mgr_.Update();
    }
    ApplyMainPaneSizes();
    main_pane_placements_ = GetMainPanePlacements();

    // 既知の導線は判定で防ぐ。未知の経路だけを修復して知らせる。
    auto& train = aui_mgr_.GetPane(train_panel_);
    auto& eval = aui_mgr_.GetPane(eval_panel_);
    if (!(train.IsShown() && train.IsDocked()) && !(eval.IsShown() && eval.IsDocked())) {
        train.Show().Dock().Centre().Layer(kTrainEvalLayer).Row(0).Position(0);
        train.SetFlag(wxAuiPaneInfo::savedHiddenState, false);
        ApplyMainPaneLayout(true, eval.IsShown());
        UpdateMainPaneControls();
        aui_mgr_.Update();
        train_panel_->SetViewVisible(true);
        LOG::warn_once() << "No main pane occupies the centre. Showing Train View as a fallback.";
    } else if (train.IsShown() && train.IsDocked() && train.dock_direction == wxAUI_DOCK_CENTER
        && eval.IsShown() && eval.IsDocked() && eval.dock_direction == wxAUI_DOCK_CENTER) {
        // 表示前フックを通らない両Centreは、Trainを残してEvalだけを降格する。
        RestoreMainPanePosition(eval);
        UpdateMainPaneControls();
        aui_mgr_.Update();
        LOG::warn_once() << "Both main panes occupy the centre. Moving Evaluation View aside as a fallback.";
    }
}

bool RunnerFrame::CanHidePane(const wxAuiPaneInfo& pane)
{
    // maximize の一時非表示と、利用者が閉じた状態を区別する。
    wxWindow* other = nullptr;
    if (pane.window == train_panel_) other = eval_panel_;
    else if (pane.window == eval_panel_) other = train_panel_;
    else return true;
    const auto& partner = aui_mgr_.GetPane(other);
    return partner.IsDocked() && anet::rl::gui::IsPaneOpen(partner, HasMaximizedPane());
}

bool RunnerFrame::ApplyMainPaneLayout(bool train_open, bool eval_open, wxWindow* showing)
{
    auto& train = aui_mgr_.GetPane(train_panel_);
    auto& eval = aui_mgr_.GetPane(eval_panel_);
    const bool train_docked = train_open && train.IsDocked();
    const bool eval_docked = eval_open && eval.IsDocked();
    const bool train_centre = train_docked && train.dock_direction == wxAUI_DOCK_CENTER;
    const bool eval_centre = eval_docked && eval.dock_direction == wxAUI_DOCK_CENTER;
    if (train_centre && eval_centre) {
        // 隠れていた対象の復帰だけを処理し、表示済みへの冗長なONは保険へ渡す。
        if (showing == train_panel_ && !train.IsShown()) {
            RestoreMainPanePosition(eval);
            return true;
        }
        if (showing == eval_panel_ && !eval.IsShown()) {
            RestoreMainPanePosition(train);
            return true;
        }
        return false; // 通常のポリシーで見つかった両Centreは、最後の保険が扱う。
    }
    // 有効な占有者がいる間は、側面へdockした相方を移動しない。
    if (train_centre || eval_centre) return false;
    auto* candidate = train_docked ? &train : eval_docked ? &eval : nullptr;
    if (!candidate) return false;

    // 昇格で失う位置だけを1組覚え、幅は既存のpane.dock_sizeに任せる。
    promoted_dock_position_ = MainPaneDockPosition{.direction=candidate->dock_direction,
        .layer=candidate->dock_layer, .row=candidate->dock_row, .position=candidate->dock_pos};
    candidate->Centre().Layer(kTrainEvalLayer).Row(0).Position(0);
    return true;
}

void RunnerFrame::RestoreMainPanePosition(wxAuiPaneInfo& pane)
{
    // 起動時など未記憶ならRight定位置へ戻す。幅の記憶は変更しない。
    const auto position = promoted_dock_position_.value_or(MainPaneDockPosition{
        .direction=wxAUI_DOCK_RIGHT, .layer=kTrainEvalLayer, .row=0, .position=0});
    pane.Direction(position.direction).Layer(position.layer).Row(position.row).Position(position.position);
}

bool RunnerFrame::UpdateMainPaneControls()
{
    // UIに前回の判断を記憶せず、相方の開閉とdock状態から宣言し直す。
    bool changed = false;
    for (auto* panel : {static_cast<wxWindow*>(train_panel_), static_cast<wxWindow*>(eval_panel_)}) {
        auto& pane = aui_mgr_.GetPane(panel);
        const bool close = CanHidePane(pane);
        const bool floatable = close;
        changed |= pane.HasCloseButton() != close || pane.IsFloatable() != floatable
            || pane.HasPinButton() != floatable;
        pane.CloseButton(close).Floatable(floatable).PinButton(floatable);
        const bool centre = pane.IsShown() && pane.IsDocked() && pane.dock_direction == wxAUI_DOCK_CENTER;
        const auto marker = pane.caption.find(kCentreCaptionSuffix);
        const bool marked = marker != wxString::npos;
        const bool correct_marker = marked && marker + kCentreCaptionSuffix.length() == pane.caption.length();
        if (centre ? !correct_marker : marked) {
            // Traceの注記と本体を残し、自分の印だけを必要な末尾へ付け直す。
            wxString caption = pane.caption;
            caption.Replace(kCentreCaptionSuffix, wxEmptyString);
            if (centre) caption += kCentreCaptionSuffix;
            pane.Caption(caption);
            if (pane.frame) pane.frame->SetLabel(caption);
            // 注記だけの変化はdockを組み直さず、captionの再描画で反映する。
            Refresh(false);
        }
    }
    return changed;
}

RunnerFrame::PanePlacement RunnerFrame::GetPanePlacement(const wxAuiPaneInfo& pane)
{
    // 幅やgeometryは合図にせず、利用者が変えた置き場だけを比較する。
    return PanePlacement{.direction=pane.dock_direction, .layer=pane.dock_layer,
        .row=pane.dock_row, .position=pane.dock_pos, .floating=pane.IsFloating(), .shown=pane.IsShown()};
}

std::array<RunnerFrame::PanePlacement, 2> RunnerFrame::GetMainPanePlacements()
{
    return {GetPanePlacement(aui_mgr_.GetPane(train_panel_)), GetPanePlacement(aui_mgr_.GetPane(eval_panel_))};
}

void RunnerFrame::ScheduleMainPaneUpdate(bool update_layout)
{
    // UPDATE_UI 中はレイアウトを組み直さず、変化したときだけ一回に集約する。
    main_pane_layout_update_needed_ |= update_layout;
    if (main_pane_update_scheduled_) return;
    main_pane_update_scheduled_ = true;
    CallAfter([this] {
        main_pane_update_scheduled_ = false;
        // OnCloseがAUIの管理を外した後は、削除を待つだけのframeなので並べ直さない。
        if (aui_mgr_.GetManagedWindow() == nullptr) return;
        // 配置の比較だけならUpdate不要。閉じたfloatのframeを作り直さない。
        const bool update_layout = std::exchange(main_pane_layout_update_needed_, false);
        if (update_layout) aui_mgr_.Update();
        // 最新の配置で一度だけ幅を当てる。最大化中と幅だけの変化は対象外。
        if (!HasMaximizedPane() && GetMainPanePlacements() != main_pane_placements_) {
            ApplyMainPaneSizes();
            main_pane_placements_ = GetMainPanePlacements();
        }
        // Q-Valuesも同じ集約で、置き場が変わったときだけ大きさを当て直す。
        if (!HasMaximizedPane() && IsQValuePlacementChanged()) {
            ApplyQValuePlacement();
            q_value_placement_ = GetPanePlacement(aui_mgr_.GetPane(q_value_panel_));
        }
    });
}

void RunnerFrame::ApplyMainPaneSizes()
{
    ANET_PROFILE_SCOPE(layout);

    // Centreと相方を等分し、組に含まれないdockはその予算から除く。
    auto& train_pane = aui_mgr_.GetPane(train_panel_);
    auto& eval_pane = aui_mgr_.GetPane(eval_panel_);
    const bool train_centre = train_pane.IsShown() && train_pane.IsDocked()
        && train_pane.dock_direction == wxAUI_DOCK_CENTER;
    const bool eval_centre = eval_pane.IsShown() && eval_pane.IsDocked()
        && eval_pane.dock_direction == wxAUI_DOCK_CENTER;
    // Centreが一つでない状態は、占有規則と最後の保険へ任せる。
    if (train_centre == eval_centre) return;
    auto& side_pane = train_centre ? eval_pane : train_pane;

    auto snapshot = TakeLayoutSnapshot();

    // Centreと同じ幅の上下の帯だけを、高さで等分する。縦配置では列を縮めない。
    if (side_pane.IsShown() && side_pane.IsDocked()
        && (side_pane.dock_direction == wxAUI_DOCK_TOP || side_pane.dock_direction == wxAUI_DOCK_BOTTOM)) {
        for (const auto& pane : aui_mgr_.GetAllPanes()) {
            if (!pane.IsShown() || !pane.IsDocked() || pane.window == side_pane.window) continue;
            // 同居帯と、左右の列の下まで広がる帯はwxの配分をそのまま保つ。
            if (pane.dock_direction == side_pane.dock_direction && pane.dock_layer == side_pane.dock_layer
                && pane.dock_row == side_pane.dock_row) return;
            if ((pane.dock_direction == wxAUI_DOCK_LEFT || pane.dock_direction == wxAUI_DOCK_RIGHT)
                && pane.dock_layer <= side_pane.dock_layer) return;
        }
        auto* band = FindPaneLayout(snapshot, side_pane.name);
        if (!band || band->dock_size <= 0) return;
        const auto& centre = train_centre ? train_pane : eval_pane;
        // Logs・toolbarを含めず、組になる2paneの現在の高さだけを予算にする。
        const int available = centre.rect.height + band->dock_size;
        const int height = std::clamp(available / 2, kMinMainPaneHeight,
            std::max(kMinMainPaneHeight, available - kMinMainPaneHeight));
        if (height == band->dock_size) return;
        side_pane.best_size.SetHeight(height);
        band->dock_size = height;
        ApplyLayoutSnapshot(std::move(snapshot));
        return; // LoadLayout後のpane参照は使わない。
    }

    // 向きもキーに含め、左右の同じlayer/rowを別の列として数える。
    struct VerticalDock {
        int direction;
        int layer;
        int row;
        int width;
        int original_width;
        int min_width;
        bool has_q_value;
        bool has_aux;
        bool has_main;
        std::vector<wxString> pane_names;
    };
    std::vector<VerticalDock> docks;

    const auto& panes = aui_mgr_.GetAllPanes();
    for (size_t i = 0; i < panes.GetCount(); ++i) {
        const auto& pane = panes.Item(i);
        if (!pane.IsShown() || !pane.IsDocked()
            || (pane.dock_direction != wxAUI_DOCK_LEFT && pane.dock_direction != wxAUI_DOCK_RIGHT)) continue;

        VerticalDock* dock = nullptr;
        for (auto& d : docks) {
            if (d.direction == pane.dock_direction && d.layer == pane.dock_layer && d.row == pane.dock_row) {
                dock = &d;
                break;
            }
        }
        if (!dock) {
            const auto* layout = FindPaneLayout(snapshot, pane.name);
            const int width = (layout && layout->dock_size > 0) ? layout->dock_size : 0;
            docks.push_back({
                .direction = pane.dock_direction,
                .layer = pane.dock_layer,
                .row = pane.dock_row,
                .width = width,
                .original_width = width,
                .min_width = 0,
                .has_q_value = false,
                .has_aux = false,
                .has_main = false,
                .pane_names = {},
            });
            dock = &docks.back();
        }
        // 幅を融通する時も、同じ列に入る各paneの最小幅を守る。
        dock->min_width = std::max({dock->min_width, pane.min_size.x, pane.window->GetMinSize().x});
        dock->has_q_value = dock->has_q_value || pane.window == q_value_panel_;
        dock->has_aux = dock->has_aux || IsAuxPane(pane);
        dock->has_main = dock->has_main || pane.window == train_panel_ || pane.window == eval_panel_;
        dock->pane_names.push_back(pane.name);
    }

    // 主paneだけの側面列が等幅の対象。同居列・上下配置はwxの幅を保つ。
    VerticalDock* side_dock = nullptr;
    if (side_pane.IsShown() && side_pane.IsDocked()) {
        for (auto& dock : docks) {
            if (dock.direction == side_pane.dock_direction && dock.layer == side_pane.dock_layer
                && dock.row == side_pane.dock_row) side_dock = &dock;
        }
    }
    const bool split = side_dock && side_dock->pane_names.size() == 1;
    int other_width = 0;
    for (auto& dock : docks) {
        if ((split && &dock == side_dock) || (!dock.has_main && (dock.has_q_value || dock.has_aux))) {
            dock.width = std::max(dock.width, dock.min_width);
        }
        if (!split || &dock != side_dock) other_width += dock.width;
    }

    // 幅が足りない時は補助 pane 列 → QValue 列の順で縮めて主領域の最小幅を確保する
    const int client_width = GetClientSize().GetWidth();
    const int centre_min_width = train_centre ? kMinTrainWidth : kMinEvalWidth;
    const int side_min_width = train_centre ? kMinEvalWidth : kMinTrainWidth;
    const int min_main_width = split ? kMinTrainWidth + kMinEvalWidth : centre_min_width;
    int shortage = min_main_width - (client_width - other_width);
    if (shortage > 0) {
        for (auto& dock : docks) {
            if (shortage <= 0) break;
            if (dock.has_main || dock.has_q_value || !dock.has_aux) continue;
            const int reducible = std::max(0, dock.width - std::max(kMinAuxWidth, dock.min_width));
            const int reduction = std::min(reducible, shortage);
            dock.width -= reduction;
            other_width -= reduction;
            shortage -= reduction;
        }
        for (auto& dock : docks) {
            if (shortage <= 0) break;
            if (dock.has_main || !dock.has_q_value) continue;
            const int reducible = std::max(0, dock.width - dock.min_width);
            const int reduction = std::min(reducible, shortage);
            dock.width -= reduction;
            other_width -= reduction;
            shortage -= reduction;
        }
    }

    // 間の列や外側の列を除いた主領域を半分にし、両方の最小幅を守る。
    const int main_width = std::max(min_main_width, client_width - other_width);
    if (split) side_dock->width = std::clamp(main_width / 2, side_min_width,
        std::max(side_min_width, main_width - centre_min_width));

    bool changed = false;
    for (const auto& dock : docks) {
        changed = changed || (dock.width != dock.original_width);
    }
    if (!changed) return;

    // live pane への書き込みは往復前に行う (LoadLayout は pane 配列を差し替えるため)
    if (split) side_pane.best_size.SetWidth(side_dock->width);

    for (const auto& dock : docks) {
        if (dock.width == dock.original_width) continue;
        for (const auto& name : dock.pane_names) {
            if (auto* info = FindPaneLayout(snapshot, name)) {
                info->dock_size = dock.width;
            }
        }
    }
    ApplyLayoutSnapshot(std::move(snapshot));
    // 注意: ここから先、往復前に取得した pane 参照 (train_pane/eval_pane) は無効
}

void RunnerFrame::HideAuxPanes()
{
    for (auto* pane : GetAuxPanes()) {
        if (!pane) continue;
        pane->SetFlag(wxAuiPaneInfo::savedHiddenState, true);
        OnPaneHiding(*pane);
        pane->Show(false);
        pane->Dock().Right().Layer(kAuxLayer).Row(0).Position(0);
        pane->dock_size = kDefaultAuxDockWidth;   // 幅の記憶も既定へ戻す
    }
}

void RunnerFrame::RestoreDefaultPanes()
{
    // 最大化は wx の解除の口で解く。pane の印を消すだけでは wx 内部の最大化の状態が残り、サッシが作られない。
    aui_mgr_.RestoreMaximizedPane();
    // Reset の復帰先は起動時の設定。昇格前の位置の記憶もリセットする。
    promoted_dock_position_.reset();
    // toolbar はフロート中でも上端 Row 0 の既定順へ回収する。縁へのドラッグで wx が置く layer 10 からも外側へ戻す。
    const std::pair<wxAuiToolBar*, int> toolbars[] = {
        {run_control_toolbar_, 0}, {step_toolbar_, 1}, {run_ops_toolbar_, 2}, {panel_toolbar_, 3},
    };
    for (const auto& [toolbar, position] : toolbars) {
        auto& pane = aui_mgr_.GetPane(toolbar);
        if (!pane.IsOk()) continue;
        // ToolbarPane()は pane 側のgripperも再度有効化するため、AddPane後と同じく
        // wxAuiToolBar内蔵gripperだけが残る状態へ戻す。
        pane.Show(true).Dock().ToolbarPane().Gripper(false)
            .Top().Layer(kToolBarLayer).Row(0).Position(position).CloseButton(false);
    }

    auto& log_pane = aui_mgr_.GetPane(log_panel_);
    if (log_pane.IsOk()) {
        log_pane.Show(true).Dock().Bottom().Layer(kTrainEvalLayer).Row(0).Position(0);
        log_pane.BestSize(400, kDefaultLogDockHeight).MinSize(100, 100).FloatingSize(800, 400);
        log_pane.dock_size = kDefaultLogDockHeight;
    }

    auto& train_pane = aui_mgr_.GetPane(train_panel_);
    if (train_pane.IsOk()) {
        train_pane.Show(main_view_ != MainView::Eval).Dock().Centre().Layer(kTrainEvalLayer).Row(0).Position(0);
        train_pane.BestSize(kMinTrainWidth, 400).MinSize(200, 200);
        train_pane.dock_size = 0;
    }

    auto& eval_pane = aui_mgr_.GetPane(eval_panel_);
    if (eval_pane.IsOk()) {
        eval_pane.Show(main_view_ != MainView::Train).Dock().Right().Layer(kTrainEvalLayer).Row(0).Position(0);
        eval_pane.BestSize(kMinEvalWidth, 400).MinSize(200, 200);
        eval_pane.dock_size = kMinEvalWidth;
    }

    auto& q_value_pane = aui_mgr_.GetPane(q_value_panel_);
    if (q_value_pane.IsOk()) {
        q_value_pane.Show(false).Dock().Right().Layer(kQValueLayer).Row(0).Position(0);
        ApplyQValuePaneSize(q_value_pane);
        q_value_pane.dock_size = q_value_pane.best_size.x;
    }

    HideAuxPanes();
    ApplyMainPaneLayout(train_pane.IsShown(), eval_pane.IsShown());
    UpdateMainPaneControls();
    main_pane_placements_ = GetMainPanePlacements();
}

void RunnerFrame::AddAuxPane(wxWindow* window, const wxAuiPaneInfo& pane_info)
{
    // 既存の補助 pane 列に合わせた幅で、列の末尾へ追加する
    const int aux_width = ResolveAuxDockWidth();

    wxAuiPaneInfo info = pane_info;
    info.Dock().Right().Layer(kAuxLayer).Row(0).Position(static_cast<int>(GetAuxPanes().size()));
    const int min_width = std::max({kMinAuxWidth, info.min_size.x, window->GetMinSize().x});
    const int preferred_width = std::max(aux_width, min_width);
    info.BestSize(preferred_width, 400);      // ドッキング時の推奨サイズ
    info.FloatingSize(std::max(400, min_width), 400);        // 切り離したときのウィンドウサイズ
    info.MinSize(min_width, std::max({100, info.min_size.y, window->GetMinSize().y}));    // これ以上小さくならないようにする
    info.dock_size = preferred_width;

    aui_mgr_.AddPane(window, info);
    aui_mgr_.Update();
    ApplyLayoutPolicy();
}

void RunnerFrame::OnMouse(anet::rl::gui::ForwardedMouseEvent& event)
{
    auto mouse_event = event.GetMouseEvent();
    if (mouse_event.LeftDown())
        wxGetApp().ToggleTraining();    // 左クリック：Trainingトグル
    else
        ToggleEvalPause();              // 右クリック：Evalトグル (resume 時は pane も表示)
}

void RunnerFrame::OnKey(anet::rl::gui::ForwardedKeyEvent& event)
{
    auto key_event = event.GetKeyEvent();
    ANET_LOG_DEBUG("KeyDown: key=" << key_event.GetKeyCode());

    // Traceの選択がある間だけ、転送された矢印を履歴の操作として消費する。
    if (eval_trace_panel_ && eval_trace_panel_->IsTraceMode()) {
        switch (key_event.GetKeyCode()) {
        case WXK_UP: eval_trace_panel_->MoveSelection(-1); return;
        case WXK_DOWN: eval_trace_panel_->MoveSelection(1); return;
        case WXK_LEFT:
        case WXK_RIGHT: return;
        default: break;
        }
    }

    int64_t action;

    switch (key_event.GetKeyCode()) {
    case WXK_UP: action = 0; break;     // NOOP
    case WXK_DOWN: action = 2; break;   // MAIN ENGINE
    case WXK_LEFT: action = 1; break;   // LEFT ENGINE
    case WXK_RIGHT: action = 3; break;  // RIGHT ENGINE
    case WXK_NUMPAD0: action = 0; break;
    case WXK_NUMPAD1: action = 1; break;
    case WXK_NUMPAD2: action = 2; break;
    case WXK_NUMPAD3: action = 3; break;
    case WXK_NUMPAD4: action = 4; break;
    case WXK_NUMPAD5: action = 5; break;
    case WXK_NUMPAD6: action = 6; break;
    case WXK_NUMPAD7: action = 7; break;
    case WXK_NUMPAD8: action = 8; break;
    case WXK_NUMPAD9: action = 9; break;

    case WXK_SHIFT:
        // SHIFT：Trainningトグル
        wxGetApp().ToggleTraining();
        return;
    case WXK_SPACE:
        // スペース：Evalトグル
        eval_panel_->TogglePause();
        return;
    case WXK_CONTROL:
        // CTRL：Evalステップ実行
        eval_panel_->DoStep();
        eval_panel_->Refresh();
        return;
    default:
        event.Skip();
        return;
    }

    // 選択されたActionをEvalPanelで実行
    eval_panel_->DoStep(action);
    eval_panel_->Refresh();
    event.Skip();
}

void RunnerFrame::OnHeatMap(wxCommandEvent& event)
{
    if (!initialized_) return;
    auto train_runner = wxGetApp().GetRunManager().GetTrainRunner();
    auto env_spec = train_runner->GetBatchEnv()->GetSpec();

    // ダイアログ生成
    SweepHeatMapDialog dialog(this, env_spec);

    if (dialog.ShowModal() == wxID_OK) {
        // 代案で指定されたHeatMap設定を取得
        SweepHeatMapSettings s = dialog.GetSettings();

        // HeatMapパネルを生成 (サイズ・ドック位置は AddAuxPane 側で決まる)
        auto heatmap_panel = new SweepHeatMapPanel(this, s.tag, s, train_runner);
        AddAuxPane(heatmap_panel,
            MakeUniquePaneInfo("HeatMapPanel", "HeatMap", s.tag)
            .CloseButton(true).MaximizeButton(true).PinButton(true)
        );
    }
}

void RunnerFrame::OnPaneHiding(wxAuiPaneInfo& pane)
{
    // 基底が Hide / Update する前に、遷移後の Centre を決める。
    const bool maximized = HasMaximizedPane();
    ApplyMainPaneLayout(pane.window != train_panel_ && anet::rl::gui::IsPaneOpen(aui_mgr_.GetPane(train_panel_), maximized),
        pane.window != eval_panel_ && anet::rl::gui::IsPaneOpen(aui_mgr_.GetPane(eval_panel_), maximized));
    if (pane.window == eval_trace_panel_) eval_trace_panel_->SetActive(false);
}

void RunnerFrame::OnPaneShowing(wxAuiPaneInfo& pane)
{
    // 表示対象を明示し、最大化で実際には隠れた相方と取り違えない。
    const bool maximized = HasMaximizedPane();
    ApplyMainPaneLayout(pane.window == train_panel_ || anet::rl::gui::IsPaneOpen(aui_mgr_.GetPane(train_panel_), maximized),
        pane.window == eval_panel_ || anet::rl::gui::IsPaneOpen(aui_mgr_.GetPane(eval_panel_), maximized), pane.window);
    if (pane.window == eval_trace_panel_) eval_trace_panel_->SetActive(true);
}

void RunnerFrame::OnConv2d(wxCommandEvent& event)
{
    if (!eval_runner_) return;

    // Conv2dPanelを生成 (サイズ・ドック位置は AddAuxPane 側で決まる)
    auto& run_manager = wxGetApp().GetRunManager();
    auto conv2d_panel = new Conv2dPanel(this, "Conv2d", run_manager, eval_runner_, conv2d_config_);

    AddAuxPane(conv2d_panel,
        MakeUniquePaneInfo("Conv2dPanel", "EvalConv2d")        // name, caption, subcaption
        .CloseButton(true).MaximizeButton(true).PinButton(true)
    );
    eval_trace_panel_->RegisterFollower(conv2d_panel);
}

void RunnerFrame::OnResetLayout(wxCommandEvent& WXUNUSED(event))
{
    // live pane の dock 情報と記憶幅を既定値へ戻す
    RestoreDefaultPanes();

    // 既存 dock は古いサイズを保持し続けるため、pane 側の既定値で dock を再構築させる。
    // floating 系の情報は SaveLayout 由来 (DIP 変換込み) をそのまま使い、dock 系の情報
    // だけを live pane の値で上書きする。
    auto snapshot = TakeLayoutSnapshot();
    for (auto& info : snapshot) {
        auto& live = aui_mgr_.GetPane(info.name);
        if (!live.IsOk()) continue;
        info.dock_direction = live.dock_direction;
        info.dock_layer = live.dock_layer;
        info.dock_row = live.dock_row;
        info.dock_pos = live.dock_pos;
        info.dock_size = live.dock_size;
        info.is_hidden = !live.IsShown();
    }
    ApplyLayoutSnapshot(std::move(snapshot));

    ApplyLayoutPolicy();
    UpdatePaneMenuChecks();
}

void RunnerFrame::OnExit(wxCommandEvent& WXUNUSED(event))
{
    Close();
}

void RunnerFrame::OnToggleTraining(wxCommandEvent& WXUNUSED(event))
{
    if (wxGetApp().IsTrainingRunning()) {
        wxGetApp().ToggleTraining();
    }
}

void RunnerFrame::ShowEvalPaneIfHidden()
{
    // Eval を進める操作では、結果が見えるよう対象 pane を先に表示する。
    // メニュー・toolbar のチェック同期は UPDATE_UI と基底の pane 連動に任せる。
    auto& pane = aui_mgr_.GetPane(eval_panel_);
    if (!pane.IsOk() || pane.IsShown()) return;

    // 自動表示もメニューと同じ開く口を通す。最大化を先に解き、元の占有者の再表示を先に処理する。
    SetPaneOpen(pane, true);
}

void RunnerFrame::ToggleEvalPause()
{
    if (!eval_panel_) return;

    // resume 要求時だけ対象 pane を先に表示し、操作対象を画面上でも明示する。
    if (eval_panel_->IsPaused()) {
        ShowEvalPaneIfHidden();
    }
    eval_panel_->TogglePause();
}

void RunnerFrame::OnToggleEval(wxCommandEvent& WXUNUSED(event))
{
    if (!initialized_) return;
    ToggleEvalPause();
}

void RunnerFrame::OnEvalStep(wxCommandEvent& WXUNUSED(event))
{
    if (!initialized_ || !eval_panel_) return;

    // 手動 step も結果を見るための操作なので、Eval toggle と同じく pane を表示してから進める。
    ShowEvalPaneIfHidden();
    eval_panel_->DoStep();
    eval_panel_->Refresh();
}

void RunnerFrame::OnEvalStepsCustom(wxCommandEvent& WXUNUSED(event))
{
    // modal loop中も現在の選択を維持し、入力が確定してから値と選択元を更新する。
    GetMenuBar()->Check(eval_steps_selection_, true);
    constexpr int kMaxSteps = std::numeric_limits<int>::max();
    wxTextEntryDialog dialog(this,
        wxString::Format("Enter steps per frame (1-%d):", kMaxSteps),
        "Eval Steps per Frame", wxString::Format("%d", eval_panel_->GetStepsPerFrame()));
    while (dialog.ShowModal() == wxID_OK) {
        wxLongLong_t steps;
        if (dialog.GetValue().ToLongLong(&steps) && steps > 0 && steps <= kMaxSteps) {
            eval_panel_->SetStepsPerFrame(static_cast<int>(steps));
            eval_panel_->ReportSpeedChange();
            eval_steps_selection_ = ID_EvalStepsCustom;
            GetMenuBar()->SetLabel(ID_EvalStepsCustom, wxString::Format("Custom (%d)...", static_cast<int>(steps)));
            GetMenuBar()->Check(ID_EvalStepsCustom, true);
            return;
        }
        wxMessageBox(wxString::Format("Invalid step_per_frame: value=%s\nExpected an integer in [1, %d].",
            dialog.GetValue(), kMaxSteps), "Invalid Eval Steps per Frame", wxOK | wxICON_ERROR, this);
    }
    GetMenuBar()->Check(eval_steps_selection_, true);
}

void RunnerFrame::OnSaveAgent(wxCommandEvent& WXUNUSED(event))
{
    if (!initialized_) return;

    // dialog 表示中に step が進むと既定ファイル名と保存内容がずれるため、走行中なら先に pause する
    // (保存後・cancel 後の自動 resume はしない)。
    wxGetApp().PauseTraining();

    // 最新 snapshot の exp_step を既定名へ反映する。未取得時は 0 を使う。
    UpdateTrainStatus();
    const auto exp_step = latest_train_status_ ? latest_train_status_->counts.exp_step : 0;
    const wxString default_name = "agent_" + wxString::FromUTF8(std::to_string(exp_step)) + ".anet";
    const auto run_dir = wxGetApp().GetRunDir();
    wxFileDialog dialog(
        this,
        "Save Checkpoint",
        wxString(run_dir.wstring()),
        default_name,
        "ANET checkpoint (*.anet)|*.anet|All files (*.*)|*.*",
        wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
    if (dialog.ShowModal() != wxID_OK) return;

    TrySaveAgent(std::filesystem::path(dialog.GetPath().ToStdWstring()));
}

void RunnerFrame::OnOpenRunFolder(wxCommandEvent& WXUNUSED(event))
{
    const auto run_dir = wxGetApp().GetRunDir();
    if (!wxLaunchDefaultApplication(wxString(run_dir.wstring()))) {
        ShowUiOperationError(
            "Open Run Folder", run_dir, "The operating system could not open the folder.");
    }
}

void RunnerFrame::OnUpdateTrainStatus(wxUpdateUIEvent& WXUNUSED(event))
{
    UpdateTrainStatus();
    ApplyAlwaysOnTopMode();
    const bool maximized = HasMaximizedPane();
    // floatのpaneをdockへ落とすと、wxはrestoreのイベントを出さずに最大化を解く。
    // 解けたことを状態の変化で拾い、restoreボタンと同じくポリシー（Trainの描画timerを含む）を当て直す。
    if (std::exchange(had_maximized_pane_, maximized) && !maximized) SchedulePolicy();
    // ドラッグのfloatには遷移フックがないため、既存の更新で占有規則も当て直す。
    const bool layout_changed = !maximized
        && ApplyMainPaneLayout(aui_mgr_.GetPane(train_panel_).IsShown(), aui_mgr_.GetPane(eval_panel_).IsShown());
    const bool controls_changed = UpdateMainPaneControls();
    const bool placement_changed = !maximized
        && (GetMainPanePlacements() != main_pane_placements_ || IsQValuePlacementChanged());
    if (layout_changed || controls_changed || placement_changed)
        ScheduleMainPaneUpdate(layout_changed || controls_changed);
}

bool RunnerFrame::TrySaveAgent(const std::filesystem::path& file_path)
{
    // UI 操作の環境依存失敗はこの境界で通知し、Run と終了 cleanup へ伝播させない。
    const wxString incomplete_warning =
        "The target file may be incomplete. It was not deleted automatically.";
    try {
        wxGetApp().SaveAgent(file_path);
        return true;
    } catch (const anet::AnetException& e) {
        ShowUiOperationError("Save Checkpoint", file_path,
            wxString::FromUTF8(e.what()) + "\n" + incomplete_warning,
            wxString::FromUTF8(e.stack_trace()));
    } catch (const std::exception& e) {
        ShowUiOperationError("Save Checkpoint", file_path,
            wxString::FromUTF8(e.what()) + "\n" + incomplete_warning);
    } catch (...) {
        ShowUiOperationError("Save Checkpoint", file_path,
            "Unknown exception.\n" + incomplete_warning);
    }
    return false;
}

void RunnerFrame::OnSystemColourChanged(wxSysColourChangedEvent& event)
{
    UpdateToolBarBitmaps();
    event.Skip();
}

void RunnerFrame::OnAbout(wxCommandEvent& WXUNUSED(event))
{
    wxMessageBox("Anet RL Runner\nIntegrated Reinforcement Learning Environment",
        "About Anet RL Runner", wxOK | wxICON_INFORMATION);
}

void RunnerFrame::OnClose(wxCloseEvent& event)
{
    if (!wxGetApp().PrepareRunClose(event.CanVeto())) {
        event.Veto();
        return;
    }
    LOG::info() << "RunnerFrame::OnClose() called.";

    // train 側から新しい評価を投入できない状態にしてから、進行中評価を照会する。
    const bool was_paused = wxGetApp().IsTrainingPaused();
    wxGetApp().StopTraining();
    auto mode = anet::rl::ShutdownMode::WAIT;
    if (event.CanVeto() && wxGetApp().WillBlockOnDrain()) {
        wxMessageDialog dialog(
            this,
            "A background evaluation session is still running.\n"
            "Wait for it to finish (up to the drain timeout), cancel it and close now, or keep running?",
            "Close Run",
            wxYES_NO | wxCANCEL | wxICON_QUESTION);
        dialog.SetYesNoCancelLabels("Wait and close", "Cancel and close", "Keep running");
        switch (dialog.ShowModal()) {
        case wxID_YES:
            mode = anet::rl::ShutdownMode::WAIT;
            break;
        case wxID_NO:
            mode = anet::rl::ShutdownMode::CANCEL;
            break;
        default:
            wxGetApp().RestartTrainingAfterCloseVeto(was_paused);
            event.Veto();
            return;
        }
    }

    if (wxGetApp().ShouldSaveAgentOnClose()) {
        TrySaveAgent(wxGetApp().GetRunDir() / "agent_close.anet");
    }
    wxGetApp().DrainBackgroundObservers(mode);
    // 背景評価の worker も学習と同じ Notifier の train observer 列を走査するので、排水の後に外す。
    DetachTrainStatusObserver();
    wxGetApp().ShutdownRunLogging();

    if (eval_panel_) {
        eval_panel_->DoClose();
    }
    aui_mgr_.UnInit();
    event.Skip();
}
