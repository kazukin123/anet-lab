#include "RunnerFrame.hpp"
#include <wx/artprov.h>
#include "anet/log.hpp"
#include "RunnerApp.hpp"
#include "LogPanel.hpp"
#include "TrainPanel.hpp"
#include "HeatMapPanel.hpp"

namespace LOG = anet::log;


/// @todo 仮枠
class ModuleBrowser : public wxPanel { public: using wxPanel::wxPanel; };
/// @todo 仮枠
class RunPanel : public wxPanel { public: using wxPanel::wxPanel; };


enum {
    ID_ResetLayout = wxID_HIGHEST + 1,
    ID_LogView,
    ID_TrainPanel,
	ID_EvalPanel,
	ID_QValuePanel,
    ID_HeatMap,
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

    // 画面レイアウトを作る
    SetupMenuBar();
    CreateStatusBar();
    SetupPanes(train_panel_config, eval_panel_config);

    // イベントハンドラ登録
    SetupEvents();

    // AUIレイアウトを反映
    aui_mgr_.Update();

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
    view_menu->AppendSeparator();
    view_menu->AppendCheckItem(ID_TrainPanel, "&Train View")->Check(true);
    view_menu->AppendSeparator();
    //view_menu->AppendCheckItem(ID_EvalPanel, "&Evaluation View")->Check(true);
    view_menu->AppendCheckItem(ID_QValuePanel, "&Evaluation QValue View")->Check(true);
    view_menu->AppendSeparator();
    view_menu->Append(ID_HeatMap, "&HeatMap");
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
    // Log View
    log_panel_ = new LogPanel(this);
    aui_mgr_.AddPane(log_panel_, wxAuiPaneInfo()
        .Name("LogPanel").Caption("Logs")
        .Right().Layer(20)          // Layer:大きいほど外側
        //.Bottom().Layer(10)          // Layer:大きいほど外側
        .BestSize(200, 200)          // ドッキング時の推奨サイズ 
        .FloatingSize(800, 400)     // 切り離したときのウィンドウサイズ
        .MinSize(500, 100)          // これ以上小さくならないようにする
        .Position(1)
        .CloseButton(true).MaximizeButton(true).MinimizeButton(true).PinButton(true)
    );

    // Run View
    //run_panel_ = new RunPanel(this);
    //aui_mgr_.AddPane(run_panel_, wxAuiPaneInfo()
    //    .Name("RunPanel").Caption("Run Control")
    //    .Left().Layer(5).Position(0)
    //    .BestSize(250, 150)
    //    .MinSize(200, 100)
    //    .MaxSize(-1, 100)
    //    .CloseButton(false).MaximizeButton(true)
    //);

    // Module Browser
    //module_browser_ = new ModuleBrowser(this);
    //aui_mgr_.AddPane(module_browser_, wxAuiPaneInfo()
    //    .Name("ModuleBrowser").Caption("Modules")
    //    .Left().Layer(5).Position(1)
    //    .BestSize(250, 800)
    //    .MinSize(200, 200)
    //    .CloseButton(false).MaximizeButton(true)
    //);

    // TrainExperienceView
    train_panel_ = new TrainPanel(this, train_panel_config);
    aui_mgr_.AddPane(train_panel_, wxAuiPaneInfo()
        .Name("TrainExperiencePanel").Caption("Train View")
        .Right().Layer(20)          // Layer:大きいほど外側
        //.Centre()
        .BestSize(300, 300)
        .MinSize(200, 200)
        .CloseButton(true).MaximizeButton(true).MinimizeButton(true).PinButton(true)
    );

    // EvalExperienceView
    eval_panel_ = new EvalPanel(this, eval_panel_config);
    aui_mgr_.AddPane(eval_panel_, wxAuiPaneInfo()
        .Name("EvalExperiencePanel").Caption("Evaluation View")
        .Centre()
        //.Right().Layer(20)
        .BestSize(900, 400)
        .MinSize(200, 200)
        .CloseButton(false).MaximizeButton(true).MinimizeButton(true).PinButton(false)
    );

    // QValuePanel
    auto config_data = wxGetApp().GetConfigData();
    q_value_panel_ = new QValuePanel(this, config_data);
    aui_mgr_.AddPane(q_value_panel_, wxAuiPaneInfo()
        .Name("EvalQValuePanel").Caption("Q-Values")
        //.Right().Layer(20)
        .Bottom().Layer(15)
        //.Left().Layer(20)
        .BestSize(900, 500)
        .MinSize(700, 300)
        .CloseButton(true).MaximizeButton(true).MinimizeButton(true).PinButton(true)
        //.DestroyOnClose(false) // デフォルトで、✕ボタンPanelを消しても非表示になるだけ
    );
}

void RunnerFrame::Initialize(std::shared_ptr<anet::rl::RunManager> run_manager)
{
    // TrainPanel初期化
    train_panel_->Initialize(run_manager);

    // EvalPanel初期化
    auto eval_runner = run_manager->CreateEvalRunner("EvalPanel");
    eval_panel_->Initialize(run_manager, eval_runner);

    // QValuePanel初期化
    q_value_panel_->Initialize(run_manager, eval_runner);
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
        eval_panel_->TogglePause();     // 左クリック：Evalトグル
    else
        wxGetApp().ToggleTraining();    // 右クリック：Trainingトグル
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
    wxGetApp().StopTraining();
    aui_mgr_.UnInit();
    event.Skip();
}
