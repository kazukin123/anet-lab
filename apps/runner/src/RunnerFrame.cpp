#include "RunnerFrame.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iterator>
#include <string>
#include <utility>
#include <wx/artprov.h>
#include <wx/filedlg.h>
#include <wx/settings.h>
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
constexpr int kDefaultAuxDockWidth = 400;
constexpr int kDefaultLogDockHeight = 200;
constexpr int kMinTrainWidth = 200;
constexpr int kMinEvalWidth = 200;
constexpr int kMinQValueWidth = 300;
constexpr int kMinAuxWidth = 100;
const wxSize kToolBitmapSize(16, 16);

constexpr const char* kPlaySvg =
    R"(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path fill="#000000" d="M4 2 L13 8 L4 14 Z"/></svg>)";
constexpr const char* kPauseSvg =
    R"(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect fill="#000000" x="4" y="2" width="3" height="12"/><rect fill="#000000" x="9" y="2" width="3" height="12"/></svg>)";
constexpr const char* kStepSvg =
    R"(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path fill="#000000" d="M3 2 L10 8 L3 14 Z"/><rect fill="#000000" x="11.5" y="2" width="2.5" height="12"/></svg>)";

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
    ID_HeatMap,
    ID_Conv2d,
    ID_TrainToggle,
    ID_EvalToggle,
    ID_EvalStep,
    ID_SaveAgent,
    ID_OpenRunFolder,
    ID_TrainFpsConfig,
    ID_TrainFpsOff,
    ID_TrainFps1,
    ID_TrainFps5,
    ID_TrainFps10,
    ID_TrainFps30,
    ID_TrainFps60,
    ID_EvalFpsConfig,
    ID_EvalFps1,
    ID_EvalFps5,
    ID_EvalFps10,
    ID_EvalFps30,
    ID_EvalFps60,
    ID_EvalFps120,
};

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

wxBitmapBundle MakeSystemTextSvg(const char* source)
{
    std::string svg(source);
    const auto colour = wxSystemSettings::GetColour(wxSYS_COLOUR_BTNTEXT)
        .GetAsString(wxC2S_HTML_SYNTAX).ToStdString();
    constexpr const char* token = "#000000";
    size_t offset = 0;
    while ((offset = svg.find(token, offset)) != std::string::npos) {
        svg.replace(offset, std::char_traits<char>::length(token), colour);
        offset += colour.size();
    }
    return wxBitmapBundle::FromSVG(svg.c_str(), kToolBitmapSize);
}

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
    ShowErrorDialog(message, detail);
}

}  // namespace runner_frame_detail

using namespace runner_frame_detail;


RunnerFrame::RunnerFrame(const wxString& title, const TrainPanelConfig& train_panel_config, const EvalPanelConfig& eval_panel_config)
    : anet::rl::gui::AuiLayoutFrame(title, wxSize(1024, 1024)),
    train_config_fps_(train_panel_config.fps), eval_config_fps_(eval_panel_config.fps)
{
    // pane の dock/float とテーマ変更の両方で Runner 用 caption 配色を維持する。
    aui_mgr_.SetArtProvider(new RunnerDockArt());

    // フラグ設定
    aui_mgr_.SetFlags(wxAUI_MGR_ALLOW_FLOATING | wxAUI_MGR_TRANSPARENT_DRAG | wxAUI_MGR_TRANSPARENT_HINT);

    // 既定の30%制約だと右側の複数列表示が狭くなるため、補助列を含めて広げられるようにする。
    aui_mgr_.SetDockSizeConstraint(0.85, 0.3);

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

    // 初期ログレベルに合わせてメニューのチェック状態を更新
    std::string init_log_level = wxGetApp().GetConfigData().Get("app.log_level", "info");
    if (init_log_level == "verbose") GetMenuBar()->Check(ID_LogLevelVerbose, true);
    else if (init_log_level == "warn") GetMenuBar()->Check(ID_LogLevelWarn, true);
    else if (init_log_level == "error") GetMenuBar()->Check(ID_LogLevelError, true);
    else GetMenuBar()->Check(ID_LogLevelInfo, true);

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
    view_menu->AppendCheckItem(ID_LogView, "&Log View")->Check(true);

    // topmost モードは起動中だけ保持し、毎回 Off から開始する。
    wxMenu* always_on_top_menu = new wxMenu;
    always_on_top_menu->AppendRadioItem(ID_AlwaysOnTopOff, "&Off")->Check(true);
    always_on_top_menu->AppendRadioItem(ID_AlwaysOnTopAlways, "&Always");
    always_on_top_menu->AppendRadioItem(ID_AlwaysOnTopWhileRunning, "&While Running");
    view_menu->AppendSubMenu(always_on_top_menu, "Always on &Top");

    // ログレベルメニューの追加
    wxMenu* log_level_menu = new wxMenu;
    log_level_menu->AppendRadioItem(ID_LogLevelError, "&Error");
    log_level_menu->AppendRadioItem(ID_LogLevelWarn, "&Warn");
    log_level_menu->AppendRadioItem(ID_LogLevelInfo, "&Info");
    log_level_menu->AppendRadioItem(ID_LogLevelVerbose, "&Verbose");
    view_menu->AppendSubMenu(log_level_menu, "L&og Level");

    view_menu->AppendSeparator();

    // その他Viewメニュー項目
    //view_menu->AppendCheckItem(ID_TrainPanel, "&Train View")->Check(true);
    view_menu->AppendCheckItem(ID_EvalPanel, "&Evaluation View")->Check(false);
    view_menu->AppendCheckItem(ID_QValuePanel, "&Evaluation QValue View")->Check(false);

    view_menu->AppendSeparator();

    // FPS は実行時 UI 操作であり、起動時 config 自体は変更しない。
    wxMenu* train_fps_menu = new wxMenu;
    train_fps_menu->AppendRadioItem(ID_TrainFpsConfig, FormatFpsConfigLabel(train_config_fps_))->Check(true);
    train_fps_menu->AppendRadioItem(ID_TrainFpsOff, "0 (Off)");
    train_fps_menu->AppendRadioItem(ID_TrainFps1, "1");
    train_fps_menu->AppendRadioItem(ID_TrainFps5, "5");
    train_fps_menu->AppendRadioItem(ID_TrainFps10, "10");
    train_fps_menu->AppendRadioItem(ID_TrainFps30, "30");
    train_fps_menu->AppendRadioItem(ID_TrainFps60, "60");
    view_menu->AppendSubMenu(train_fps_menu, "Train View FPS");

    wxMenu* eval_fps_menu = new wxMenu;
    eval_fps_menu->AppendRadioItem(ID_EvalFpsConfig, FormatFpsConfigLabel(eval_config_fps_))->Check(true);
    eval_fps_menu->AppendRadioItem(ID_EvalFps1, "1");
    eval_fps_menu->AppendRadioItem(ID_EvalFps5, "5");
    eval_fps_menu->AppendRadioItem(ID_EvalFps10, "10");
    eval_fps_menu->AppendRadioItem(ID_EvalFps30, "30");
    eval_fps_menu->AppendRadioItem(ID_EvalFps60, "60");
    eval_fps_menu->AppendRadioItem(ID_EvalFps120, "120");
    view_menu->AppendSubMenu(eval_fps_menu, "Eval View FPS");

    view_menu->AppendSeparator();

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
    run_control_toolbar_->AddTool(ID_TrainToggle, "Train", MakeSystemTextSvg(kPlaySvg),
        "Pause/resume training (Shift / left-click)", wxITEM_CHECK);
    run_control_toolbar_->AddSeparator();
    run_control_toolbar_->AddTool(ID_EvalToggle, "Eval", MakeSystemTextSvg(kPlaySvg),
        "Pause/resume evaluation (Space / right-click)", wxITEM_CHECK);
    run_control_toolbar_->AddTool(ID_EvalStep, "Step", MakeSystemTextSvg(kStepSvg),
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
    panel_toolbar_->SetOverflowVisible(false);
    panel_toolbar_->AddTool(ID_LogView, "Logs", wxBitmapBundle(), "Show/hide Logs", wxITEM_CHECK);
    panel_toolbar_->AddTool(ID_EvalPanel, "Eval View", wxBitmapBundle(), "Show/hide Eval View", wxITEM_CHECK);
    panel_toolbar_->AddTool(ID_QValuePanel, "Q-Values", wxBitmapBundle(), "Show/hide Q-Values", wxITEM_CHECK);
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
            .ToolbarPane().Top().Row(0).Position(position)
            .CloseButton(false));
    }
}

void RunnerFrame::SetupPanes(const TrainPanelConfig& train_panel_config, const EvalPanelConfig& eval_panel_config)
{
    const int initial_client_width = GetClientSize().GetWidth();
    const int initial_q_value_width = std::max(kMinQValueWidth, initial_client_width / 3);
    const int main_pane_width = std::max(kMinTrainWidth, (initial_client_width - initial_q_value_width) / 2);

    // Log View
    log_panel_ = new LogPanel(this);
    wxAuiPaneInfo log_info = wxAuiPaneInfo()
        .Name("LogPanel").Caption("Logs")
        .Bottom().Layer(kTrainEvalLayer).Row(0).Position(0)
        .BestSize(400, kDefaultLogDockHeight) // ドッキング時の推奨サイズ
        .FloatingSize(800, 400)     // 切り離したときのウィンドウサイズ
        .MinSize(200, 200)          // これ以上小さくならないようにする
        .CloseButton(true).MaximizeButton(true).MinimizeButton(true).PinButton(true);
    log_info.dock_size = kDefaultLogDockHeight;
    aui_mgr_.AddPane(log_panel_, log_info);

    // TrainExperienceView
    train_panel_ = new TrainPanel(this, train_panel_config);
    aui_mgr_.AddPane(train_panel_, wxAuiPaneInfo()
        .Name("TrainExperiencePanel").Caption("Train View")
        .Centre()
        .BestSize(main_pane_width, 400)
        .MinSize(200, 200)
        .CloseButton(false).MaximizeButton(true).MinimizeButton(true).PinButton(false)
    );

    // EvalExperienceView
    eval_panel_ = new EvalPanel(this, eval_panel_config);
    wxAuiPaneInfo eval_info = wxAuiPaneInfo()
        .Name("EvalExperiencePanel").Caption("Evaluation View")
        .Right().Layer(kTrainEvalLayer).Row(0).Position(0)
        .BestSize(main_pane_width, 400)
        .MinSize(200, 200)
        .CloseButton(true).MaximizeButton(true).MinimizeButton(true).PinButton(false)
        .Hide();
    eval_info.dock_size = main_pane_width;
    aui_mgr_.AddPane(eval_panel_, eval_info);

    // QValuePanel
    auto config_data = wxGetApp().GetConfigData();
    q_value_panel_ = new QValuePanel(this, config_data);
    const int q_value_width = GetDefaultQValueDockWidth();
    wxAuiPaneInfo q_value_info = wxAuiPaneInfo()
        .Name("EvalQValuePanel").Caption("Evaluation Q-Values")
        .Right().Layer(kQValueLayer).Row(0).Position(0)
        .BestSize(q_value_width, 800)
        .MinSize(300, 150)
        .CloseButton(true).MaximizeButton(true).MinimizeButton(true).PinButton(false)
        //.DestroyOnClose(false) // デフォルトで、✕ボタンPanelを消しても非表示になるだけ
        .Hide();
    q_value_info.dock_size = q_value_width;
    aui_mgr_.AddPane(q_value_panel_, q_value_info);
}

void RunnerFrame::Initialize(std::shared_ptr<anet::rl::RunManager> run_manager)
{
    // TrainPanel初期化
    train_panel_->Initialize(run_manager);

    // EvalPanel初期化
    const auto& eval_config_tag = eval_panel_->GetConfig().eval_config_tag;
    eval_runner_ = run_manager->CreateEvalRunner(
        "EvalPanel", anet::rl::RunMode::Eval1,
        eval_panel_->GetConfig().model_sync.UsesClonedModel(), std::nullopt, eval_config_tag);
    eval_panel_->Initialize(run_manager, eval_runner_);

    // QValuePanel初期化
    q_value_panel_->Initialize(run_manager, eval_runner_);
    q_value_panel_->SetActionHandler([this](int64_t action) {
        eval_panel_->DoStep(action);
        eval_panel_->Refresh();
    });

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

    toolbar->SetToolBitmap(tool_id, MakeSystemTextSvg(running ? kPauseSvg : kPlaySvg));
    shown_as_running = running;
    toolbar->Refresh(false);
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
    run_control_toolbar_->SetToolBitmap(ID_EvalStep, MakeSystemTextSvg(kStepSvg));
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

    const std::pair<int, wxWindow*> panel_bindings[] = {
        {ID_LogView, log_panel_},
        {ID_EvalPanel, eval_panel_},
        {ID_QValuePanel, q_value_panel_},
    };
    for (const auto& [id, window] : panel_bindings) {
        Bind(wxEVT_UPDATE_UI, [this, window](wxUpdateUIEvent& event) {
            auto& pane = aui_mgr_.GetPane(window);
            event.Enable(pane.IsOk());
            event.Check(pane.IsOk() && pane.IsShown());
        }, id);
    }

    const std::pair<int, float> train_fps_items[] = {
        {ID_TrainFpsConfig, train_config_fps_}, {ID_TrainFpsOff, 0.0f}, {ID_TrainFps1, 1.0f},
        {ID_TrainFps5, 5.0f}, {ID_TrainFps10, 10.0f}, {ID_TrainFps30, 30.0f}, {ID_TrainFps60, 60.0f},
    };
    for (const auto& [id, fps] : train_fps_items) {
        Bind(wxEVT_MENU, [this, fps](wxCommandEvent&) { train_panel_->SetFps(fps); }, id);
        Bind(wxEVT_UPDATE_UI, [this](wxUpdateUIEvent& event) { event.Enable(initialized_); }, id);
    }

    const std::pair<int, float> eval_fps_items[] = {
        {ID_EvalFpsConfig, eval_config_fps_}, {ID_EvalFps1, 1.0f}, {ID_EvalFps5, 5.0f},
        {ID_EvalFps10, 10.0f}, {ID_EvalFps30, 30.0f}, {ID_EvalFps60, 60.0f}, {ID_EvalFps120, 120.0f},
    };
    for (const auto& [id, fps] : eval_fps_items) {
        Bind(wxEVT_MENU, [this, fps](wxCommandEvent&) { eval_panel_->SetFps(fps); }, id);
        Bind(wxEVT_UPDATE_UI, [this](wxUpdateUIEvent& event) { event.Enable(initialized_); }, id);
    }

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
    return pane.name.StartsWith("HeatMapPanel_") || pane.name.StartsWith("Conv2dPanel_");
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

int RunnerFrame::GetDefaultQValueDockWidth() const
{
    if (!q_value_panel_) {
        return 640;
    }
    return std::max(kMinQValueWidth, q_value_panel_->GetPreferredDockWidth());
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
    // maximize 中は LoadLayout 往復が復元情報 (savedHiddenState) を壊すため何もしない。
    // restore 後に基底の restore ハンドラ経由で再適用される。
    if (HasMaximizedPane()) return;

    // 主領域ポリシー (Train/Eval 50:50) は両 pane が定位置にいる時だけ適用する
    auto& train_pane = aui_mgr_.GetPane(train_panel_);
    auto& eval_pane = aui_mgr_.GetPane(eval_panel_);
    if (!IsInHomeDock(eval_pane, wxAUI_DOCK_RIGHT, kTrainEvalLayer)) return;
    if (!train_pane.IsOk() || !train_pane.IsShown() || !train_pane.IsDocked()
        || train_pane.dock_direction != wxAUI_DOCK_CENTER) return;

    auto snapshot = TakeLayoutSnapshot();

    // 右側の dock (Eval の dock、QValue 列、補助 pane 列) を dock 単位で集計する
    struct RightDock {
        int layer;
        int row;
        int width;
        int original_width;
        bool has_q_value;
        std::vector<wxString> pane_names;
    };
    std::vector<RightDock> right_docks;

    const auto& panes = aui_mgr_.GetAllPanes();
    for (size_t i = 0; i < panes.GetCount(); ++i) {
        const auto& pane = panes.Item(i);
        if (!pane.IsShown() || !pane.IsDocked() || pane.dock_direction != wxAUI_DOCK_RIGHT) continue;
        const bool is_q_value = (pane.window == q_value_panel_);
        const bool is_main_layer = (pane.dock_layer == kTrainEvalLayer);
        if (!is_main_layer && !is_q_value && !IsAuxPane(pane)) continue;

        RightDock* dock = nullptr;
        for (auto& d : right_docks) {
            if (d.layer == pane.dock_layer && d.row == pane.dock_row) { dock = &d; break; }
        }
        if (!dock) {
            const auto* layout = FindPaneLayout(snapshot, pane.name);
            const int width = (layout && layout->dock_size > 0) ? layout->dock_size : 0;
            right_docks.push_back({
                .layer = pane.dock_layer,
                .row = pane.dock_row,
                .width = width,
                .original_width = width,
                .has_q_value = false,
                .pane_names = {},
            });
            dock = &right_docks.back();
        }
        dock->has_q_value = dock->has_q_value || is_q_value;
        dock->pane_names.push_back(pane.name);
    }

    // Eval が属する dock (主領域側) と、その外側の列 (QValue/補助) を分けて幅を集計
    RightDock* eval_dock = nullptr;
    int q_value_width = 0;
    int aux_width = 0;
    for (auto& dock : right_docks) {
        if (dock.layer == kTrainEvalLayer) {
            eval_dock = &dock;
        } else if (dock.has_q_value) {
            q_value_width += dock.width;
        } else {
            aux_width += dock.width;
        }
    }
    if (!eval_dock || eval_dock->width <= 0) return;

    // 幅が足りない時は補助 pane 列 → QValue 列の順で縮めて主領域の最小幅を確保する
    const int client_width = GetClientSize().GetWidth();
    const int min_main_width = kMinTrainWidth + kMinEvalWidth;
    int shortage = min_main_width - (client_width - q_value_width - aux_width);
    if (shortage > 0) {
        for (auto& dock : right_docks) {
            if (shortage <= 0) break;
            if (dock.layer == kTrainEvalLayer || dock.has_q_value) continue;
            const int reducible = std::max(0, dock.width - kMinAuxWidth);
            const int reduction = std::min(reducible, shortage);
            dock.width -= reduction;
            shortage -= reduction;
        }
        for (auto& dock : right_docks) {
            if (shortage <= 0) break;
            if (dock.layer == kTrainEvalLayer || !dock.has_q_value) continue;
            const int reducible = std::max(0, dock.width - kMinQValueWidth);
            const int reduction = std::min(reducible, shortage);
            dock.width -= reduction;
            shortage -= reduction;
        }
        q_value_width = 0;
        aux_width = 0;
        for (const auto& dock : right_docks) {
            if (dock.layer == kTrainEvalLayer) continue;
            (dock.has_q_value ? q_value_width : aux_width) += dock.width;
        }
    }

    // 主領域を Train:Eval = 50:50 に分割 (Train の最小幅は確保する)
    const int main_width = std::max(min_main_width, client_width - q_value_width - aux_width);
    int eval_width = main_width / 2;
    eval_width = std::clamp(eval_width, kMinEvalWidth, std::max(kMinEvalWidth, main_width - kMinTrainWidth));
    eval_dock->width = eval_width;

    bool changed = false;
    for (const auto& dock : right_docks) {
        changed = changed || (dock.width != dock.original_width);
    }
    if (!changed) return;

    // live pane への書き込みは往復前に行う (LoadLayout は pane 配列を差し替えるため)
    eval_pane.best_size.SetWidth(eval_width);

    for (const auto& dock : right_docks) {
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
        pane->Show(false).Restore();
        pane->Dock().Right().Layer(kAuxLayer).Row(0).Position(0);
        pane->dock_size = kDefaultAuxDockWidth;   // 幅の記憶も既定へ戻す
    }
}

void RunnerFrame::RestoreDefaultPanes()
{
    // toolbar はフロート中でも上端 Row 0 の既定順へ回収する。
    const std::pair<wxAuiToolBar*, int> toolbars[] = {
        {run_control_toolbar_, 0}, {step_toolbar_, 1}, {run_ops_toolbar_, 2}, {panel_toolbar_, 3},
    };
    for (const auto& [toolbar, position] : toolbars) {
        auto& pane = aui_mgr_.GetPane(toolbar);
        if (!pane.IsOk()) continue;
        // ToolbarPane()は pane 側のgripperも再度有効化するため、AddPane後と同じく
        // wxAuiToolBar内蔵gripperだけが残る状態へ戻す。
        pane.Show(true).Restore().Dock().ToolbarPane().Gripper(false)
            .Top().Row(0).Position(position).CloseButton(false);
    }

    auto& log_pane = aui_mgr_.GetPane(log_panel_);
    if (log_pane.IsOk()) {
        log_pane.Show(true).Dock().Bottom().Layer(kTrainEvalLayer).Row(0).Position(0).Restore();
        log_pane.BestSize(400, kDefaultLogDockHeight).MinSize(100, 100).FloatingSize(800, 400);
        log_pane.dock_size = kDefaultLogDockHeight;
    }

    auto& train_pane = aui_mgr_.GetPane(train_panel_);
    if (train_pane.IsOk()) {
        train_pane.Show(true).Dock().Centre().Layer(kTrainEvalLayer).Row(0).Position(0).Restore();
        train_pane.BestSize(kMinTrainWidth, 400).MinSize(200, 200);
        train_pane.dock_size = 0;
    }

    auto& eval_pane = aui_mgr_.GetPane(eval_panel_);
    if (eval_pane.IsOk()) {
        eval_pane.Show(false).Dock().Right().Layer(kTrainEvalLayer).Row(0).Position(0).Restore();
        eval_pane.BestSize(kMinEvalWidth, 400).MinSize(200, 200);
        eval_pane.dock_size = kMinEvalWidth;
    }

    auto& q_value_pane = aui_mgr_.GetPane(q_value_panel_);
    if (q_value_pane.IsOk()) {
        const int q_value_width = GetDefaultQValueDockWidth();
        q_value_pane.Show(false).Dock().Right().Layer(kQValueLayer).Row(0).Position(0).Restore();
        q_value_pane.BestSize(q_value_width, 800).MinSize(300, 150);
        q_value_pane.dock_size = q_value_width;
    }

    HideAuxPanes();
}

void RunnerFrame::AddAuxPane(wxWindow* window, const wxAuiPaneInfo& pane_info)
{
    // 既存の補助 pane 列に合わせた幅で、列の末尾へ追加する
    const int aux_width = ResolveAuxDockWidth();

    wxAuiPaneInfo info = pane_info;
    info.Dock().Right().Layer(kAuxLayer).Row(0).Position(static_cast<int>(GetAuxPanes().size()));
    info.BestSize(aux_width, 400);      // ドッキング時の推奨サイズ
    info.FloatingSize(400, 400);        // 切り離したときのウィンドウサイズ
    info.MinSize(kMinAuxWidth, 100);    // これ以上小さくならないようにする
    info.dock_size = aux_width;

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
            .CloseButton(true).MaximizeButton(true)
        );
    }
}

void RunnerFrame::OnConv2d(wxCommandEvent& event)
{
    if (!eval_runner_) return;

    // Conv2dPanelを生成 (サイズ・ドック位置は AddAuxPane 側で決まる)
    auto& run_manager = wxGetApp().GetRunManager();
    auto config_data = wxGetApp().GetConfigData();
    auto conv2d_panel = new Conv2dPanel(this, "Conv2d", run_manager, eval_runner_, config_data);

    AddAuxPane(conv2d_panel,
        MakeUniquePaneInfo("Conv2dPanel", "EvalConv2d")        // name, caption, subcaption
        .CloseButton(true).MaximizeButton(true)
    );
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
        info.is_maximized = false;    // RestoreDefaultPanes で Restore 済み
    }
    ApplyLayoutSnapshot(std::move(snapshot));

    ApplyLayoutPolicy();
    UpdatePaneMenuChecks();
}

void RunnerFrame::OnExit(wxCommandEvent& WXUNUSED(event))
{
    Close(true);
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

    pane.Show(true);
    aui_mgr_.Update();
    ApplyLayoutPolicy();
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
    LOG::info() << "RunnerFrame::OnClose() called.";

    wxGetApp().StopTraining();
    DetachTrainStatusObserver();
    TrySaveAgent(wxGetApp().GetRunDir() / "agent_close.anet");
    wxGetApp().ShutdownRunLogging();

    if (eval_panel_) {
        eval_panel_->DoClose();
    }
    aui_mgr_.UnInit();
    event.Skip();
}
