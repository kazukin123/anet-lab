#include "RunnerFrame.hpp"
#include <algorithm>
#include <wx/artprov.h>
#include "anet/log.hpp"
#include "RunnerApp.hpp"
#include "LogPanel.hpp"
#include "TrainPanel.hpp"
#include "HeatMapPanel.hpp"
#include "Conv2dPanel.hpp"

namespace LOG = anet::log;


/// @todo 仮枠
class ModuleBrowser : public wxPanel { public: using wxPanel::wxPanel; };
/// @todo 仮枠
class RunPanel : public wxPanel { public: using wxPanel::wxPanel; };


enum {
    ID_ResetLayout = wxID_HIGHEST + 1,
    ID_LogView,
    ID_LogLevelInfo,
    ID_LogLevelVerbose,
    ID_LogLevelWarn,
    ID_LogLevelError,
    ID_TrainPanel,
	ID_EvalPanel,
	ID_QValuePanel,
    ID_HeatMap,
    ID_Conv2d,
    //ID_ModuleBrowser,
    //ID_RunView,
};


RunnerFrame::RunnerFrame(const wxString& title, const TrainPanelConfig& train_panel_config, const EvalPanelConfig& eval_panel_config)
    : wxFrame(NULL, wxID_ANY, title, wxDefaultPosition, wxSize(1280, 800))
{
    // AUI Managerの管理下に置く
    aui_mgr_.SetManagedWindow(this);

    // フラグ設定
    aui_mgr_.SetFlags(wxAUI_MGR_ALLOW_FLOATING | wxAUI_MGR_TRANSPARENT_DRAG | wxAUI_MGR_TRANSPARENT_HINT);
    // 既定の30%制約だと右側のEval表示が狭くなるため、Train/Evalを左右半分程度まで広げる
    aui_mgr_.SetDockSizeConstraint(0.5, 0.3);

    // 画面レイアウトを作る
    SetupMenuBar();
    CreateStatusBar();
    SetupPanes(train_panel_config, eval_panel_config);

    // イベントハンドラ登録
    SetupEvents();

    // AUIレイアウトを反映
    aui_mgr_.Update();

    // 初期ログレベルに合わせてメニューのチェック状態を更新
    std::string init_log_level = wxGetApp().GetConfigData().Get("app.log_level", "info");
    if (init_log_level == "verbose") GetMenuBar()->Check(ID_LogLevelVerbose, true);
    else if (init_log_level == "warn") GetMenuBar()->Check(ID_LogLevelWarn, true);
    else if (init_log_level == "error") GetMenuBar()->Check(ID_LogLevelError, true);
    else GetMenuBar()->Check(ID_LogLevelInfo, true);

    // ウィンドウ表示
    Centre();
}

RunnerFrame::~RunnerFrame()
{
    aui_mgr_.UnInit();
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
    //view_menu->Append(ID_ResetLayout, "&Reset Layout", "Reset to default layout");
    view_menu->AppendCheckItem(ID_LogView, "&Log View")->Check(true);

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
    view_menu->AppendCheckItem(ID_EvalPanel, "&Evaluation View")->Check(true);
    view_menu->AppendCheckItem(ID_QValuePanel, "&Evaluation QValue View")->Check(true);

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
    // ステータスバーを作成
    wxStatusBar* statusBar = wxFrame::CreateStatusBar(3);
    statusBar->SetStatusText("Ready", 0);
    //statusBar->SetStatusWidths(3, (int[]) { -1, 100, 200 });
}

void RunnerFrame::SetupPanes(const TrainPanelConfig& train_panel_config, const EvalPanelConfig& eval_panel_config)
{
    const int main_pane_width = std::max(600, GetClientSize().GetWidth() / 2);

    // Log View
    log_panel_ = new LogPanel(this);
    aui_mgr_.AddPane(log_panel_, wxAuiPaneInfo()
        .Name("LogPanel").Caption("Logs")
        .Bottom().Layer(0).Position(0)
        .BestSize(400, 200)         // ドッキング時の推奨サイズ
        .FloatingSize(800, 400)     // 切り離したときのウィンドウサイズ
        .MinSize(100, 100)          // これ以上小さくならないようにする
        .CloseButton(true).MaximizeButton(true).MinimizeButton(true).PinButton(true)
    );

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
    aui_mgr_.AddPane(eval_panel_, wxAuiPaneInfo()
        .Name("EvalExperiencePanel").Caption("Evaluation View")
        .Right().Layer(10).Row(0).Position(0)
        .BestSize(main_pane_width, 400)
        .MinSize(200, 200)
        .CloseButton(true).MaximizeButton(true).MinimizeButton(true).PinButton(false)
    );

    // QValuePanel
    auto config_data = wxGetApp().GetConfigData();
    q_value_panel_ = new QValuePanel(this, config_data);
    aui_mgr_.AddPane(q_value_panel_, wxAuiPaneInfo()
        .Name("EvalQValuePanel").Caption("Evaluation Q-Values")
        .Right().Layer(10).Row(0).Position(1)
        .BestSize(main_pane_width, 800)
        .MinSize(300, 150)
        .CloseButton(true).MaximizeButton(true).MinimizeButton(true).PinButton(false)
        //.DestroyOnClose(false) // デフォルトで、✕ボタンPanelを消しても非表示になるだけ
    );
}

void RunnerFrame::Initialize(std::shared_ptr<anet::rl::RunManager> run_manager)
{
    // TrainPanel初期化
    train_panel_->Initialize(run_manager);

    // EvalPanel初期化
    eval_runner_ = run_manager->CreateEvalRunner(
        "EvalPanel", anet::rl::RunMode::Eval1, eval_panel_->GetConfig().model_sync.UsesClonedModel());
    eval_panel_->Initialize(run_manager, eval_runner_);

    // QValuePanel初期化
    q_value_panel_->Initialize(run_manager, eval_runner_);
    q_value_panel_->SetActionHandler([this](int64_t action) {
        eval_panel_->DoStep(action);
        eval_panel_->Refresh();
    });
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
    //Bind(wxEVT_MENU, [=](wxCommandEvent&) { Close(true); }, wxID_EXIT);

    // メニューイベント
    Bind(wxEVT_MENU, &RunnerFrame::OnExit, this, wxID_EXIT);
    Bind(wxEVT_MENU, &RunnerFrame::OnAbout, this, wxID_ABOUT);
    Bind(wxEVT_MENU, &RunnerFrame::OnResetLayout, this, ID_ResetLayout);
    Bind(wxEVT_MENU, &RunnerFrame::OnHeatMap, this, ID_HeatMap);
    Bind(wxEVT_MENU, &RunnerFrame::OnConv2d, this, ID_Conv2d);

	// パネル表示/非表示メニュー連動 (チェック状態に合わせてパネル表示切替)
    Bind(wxEVT_MENU, [this](wxCommandEvent& event) {
        auto& pane = aui_mgr_.GetPane(log_panel_);
        if (pane.IsOk()) {
            pane.Show(event.IsChecked());
            aui_mgr_.Update();
        }
        }, ID_LogView);
    Bind(wxEVT_MENU, [this](wxCommandEvent& event) {
        auto& pane = aui_mgr_.GetPane(train_panel_);
        if (pane.IsOk()) {
            pane.Show(event.IsChecked());
            aui_mgr_.Update();
        }
        }, ID_TrainPanel);
    Bind(wxEVT_MENU, [this](wxCommandEvent& event) {
        auto& pane = aui_mgr_.GetPane(eval_panel_);
        if (pane.IsOk()) {
            pane.Show(event.IsChecked());
            aui_mgr_.Update();
        }
        }, ID_EvalPanel);
    Bind(wxEVT_MENU, [this](wxCommandEvent& event) {
        auto& pane = aui_mgr_.GetPane(q_value_panel_);
        if (pane.IsOk()) {
            pane.Show(event.IsChecked());
            aui_mgr_.Update();
        }
        }, ID_QValuePanel);
        
    Bind(wxEVT_MENU, [this](wxCommandEvent&) { if (log_panel_) log_panel_->SetLogLevel("info"); }, ID_LogLevelInfo);
    Bind(wxEVT_MENU, [this](wxCommandEvent&) { if (log_panel_) log_panel_->SetLogLevel("verbose"); }, ID_LogLevelVerbose);
    Bind(wxEVT_MENU, [this](wxCommandEvent&) { if (log_panel_) log_panel_->SetLogLevel("warn"); }, ID_LogLevelWarn);
    Bind(wxEVT_MENU, [this](wxCommandEvent&) { if (log_panel_) log_panel_->SetLogLevel("error"); }, ID_LogLevelError);

	// ✕ボタンによるパネルクローズ時のメニュー連動
    Bind(wxEVT_AUI_PANE_CLOSE, [this](wxAuiManagerEvent& event) {
        if (event.GetPane()->window == log_panel_) {
            if (GetMenuBar()) {
                GetMenuBar()->Check(ID_LogView, false);
        }
    }
    event.Skip();
        });
    Bind(wxEVT_AUI_PANE_CLOSE, [this](wxAuiManagerEvent& event) {
        if (event.GetPane()->window == train_panel_) {
            if (GetMenuBar()) {
                GetMenuBar()->Check(ID_TrainPanel, false);
    }
        }
        event.Skip();
        });
    Bind(wxEVT_AUI_PANE_CLOSE, [this](wxAuiManagerEvent& event) {
        if (event.GetPane()->window == eval_panel_) {
            if (GetMenuBar()) {
                GetMenuBar()->Check(ID_EvalPanel, false);
        }
        }
        event.Skip();
        });
    Bind(wxEVT_AUI_PANE_CLOSE, [this](wxAuiManagerEvent& event) {
        if (event.GetPane()->window == q_value_panel_) {
            if (GetMenuBar()) {
                GetMenuBar()->Check(ID_QValuePanel, false);
        }
    }
    event.Skip();
    });

    // きーまう
    Bind(anet::rl::gui::EVT_FORWARDED_MOUSE, &RunnerFrame::OnMouse, this);
    Bind(anet::rl::gui::EVT_FORWARDED_KEY, &RunnerFrame::OnKey, this);
}

wxAuiPaneInfo RunnerFrame::PanelInfo(const wxString& name, const wxString& caption, const wxString& sub_caption)
{
    long long timestamp = wxGetLocalTimeMillis().GetValue();
    wxString unique_name = wxString::Format("%s_%lld", name, timestamp);
    auto new_cation = caption;
    if (!sub_caption.empty()) new_cation += " " + sub_caption;
    return wxAuiPaneInfo().Name(unique_name).Caption(new_cation);
}

void RunnerFrame::OnMouse(anet::rl::gui::ForwardedMouseEvent& event)
{
    auto mouse_event = event.GetMouseEvent();
    if (mouse_event.LeftDown())
        wxGetApp().ToggleTraining();    // 左クリック：Trainingトグル
    else
        eval_panel_->TogglePause();     // 右クリック：Evalトグル
}

void RunnerFrame::OnKey(anet::rl::gui::ForwardedKeyEvent& event)
{
    auto key_event = event.GetKeyEvent();
    //LOG::info() << "RunnerFrame::OnKey() key=" << key_event.GetKeyCode() << " eventType=" << key_event.GetEventType();
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

        // HeatMapパネルを生成
        auto heatmap_panel = new SweepHeatMapPanel(this, s.tag, s, train_runner);
        aui_mgr_.AddPane(heatmap_panel,
            PanelInfo("HeatMapPanel", "HeatMap", s.tag)
            .Right().Layer(20)          // Layer:大きいほど外側
            .BestSize(400, 400)          // ドッキング時の推奨サイズ 
            .FloatingSize(400, 400)     // 切り離したときのウィンドウサイズ
            .MinSize(100, 100)          // これ以上小さくならないようにする
            //.Float()
            .Dock()
            .CloseButton(true).MaximizeButton(true)
        );

        // レイアウト反映
        aui_mgr_.Update();
    }
}
void RunnerFrame::OnConv2d(wxCommandEvent& event)
{
    if (!eval_runner_) return;

    //Conv2dPanel(wxWindow * parent, const wxString & title, std::shared_ptr<anet::rl::RunManager> run_manager, std::shared_ptr<anet::rl::Runner> runner);

    // Conv2dPanelを生成
    auto& run_manager = wxGetApp().GetRunManager();
    auto config_data = wxGetApp().GetConfigData();
    auto conv2d_panel = new Conv2dPanel(this, "Conv2d", run_manager, eval_runner_, config_data);

    // AuiManagerに登録
    aui_mgr_.AddPane(conv2d_panel,
        PanelInfo("Conv2dPanel", "EvalConv2d")        // name, caption, subcaption
        .Right().Layer(20)          // Layer:大きいほど外側
        .BestSize(400, 400)          // ドッキング時の推奨サイズ 
        .FloatingSize(400, 400)     // 切り離したときのウィンドウサイズ
        .MinSize(100, 100)          // これ以上小さくならないようにする
        //.Float()
        .Dock()
        .CloseButton(true).MaximizeButton(true)
    );

    // レイアウト反映
    aui_mgr_.Update();
}

void RunnerFrame::OnResetLayout(wxCommandEvent& WXUNUSED(event))
{
    /// @todo impl.

    // 保存されたパースペクティブがあればロード、なければデフォルト設定
    // m_mgr_.LoadPerspective(default_perspective_);
}

void RunnerFrame::OnExit(wxCommandEvent& WXUNUSED(event))
{
    Close(true);
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
    wxGetApp().SaveAgent("agent_close.anet");
    wxGetApp().ShutdownRunLogging();

    if (eval_panel_) {
        eval_panel_->DoClose();
    }
    aui_mgr_.UnInit();
    event.Skip();
}
