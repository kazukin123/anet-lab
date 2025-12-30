#include "RunnerFrame.hpp"
#include <wx/artprov.h>
#include "anet/log.hpp"
#include "RunnerApp.hpp"
#include "LogPanel.hpp"
#include "TrainPanel.hpp"
#include "HeatMapPanel.hpp"

namespace LOG = anet::log;

// 仮のViewクラス定義 (コンパイルを通すためのスタブ。本来は別ファイル)
class ModuleBrowser : public wxPanel { public: using wxPanel::wxPanel; };

class RunPanel : public wxPanel { public: using wxPanel::wxPanel; };


// --- ID Definitions ---
enum {
    ID_ResetLayout = wxID_HIGHEST + 1,
    ID_ModuleBrowser, // View Menu用
    ID_RunView,
    ID_LogView,
    ID_HeatMap
};

wxBEGIN_EVENT_TABLE(RunnerFrame, wxFrame)
    EVT_MENU(wxID_EXIT, RunnerFrame::OnExit)
    EVT_MENU(wxID_ABOUT, RunnerFrame::OnAbout)
    EVT_MENU(ID_ResetLayout, RunnerFrame::OnResetLayout)
    EVT_CLOSE(RunnerFrame::OnClose)
wxEND_EVENT_TABLE()

RunnerFrame::RunnerFrame(const wxString& title, const TrainPanelConfig& train_panel_config, const EvalPanelConfig& eval_panel_config)
    : wxFrame(NULL, wxID_ANY, title, wxDefaultPosition, wxSize(1280, 800))
{
    // AUI Managerの管理下に置く
    m_mgr_.SetManagedWindow(this);

    // 画面レイアウトを作る
    SetupMenuBar();
    CreateStatusBar();
    SetupPanes(train_panel_config, eval_panel_config);

    // イベントハンドラ登録
    SetupEvents();

    // 変更を反映
    m_mgr_.Update();

    // ウィンドウ表示
    Centre();
}

RunnerFrame::~RunnerFrame()
{
    m_mgr_.UnInit();
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
    m_mgr_.AddPane(log_panel_, wxAuiPaneInfo()
        .Name("LogPanel").Caption("Logs")
        .Bottom().Layer(10)          // Layer:大きいほど外側
        .BestSize(-1, 200)          // ドッキング時の推奨サイズ 
        .FloatingSize(800, 400)     // 切り離したときのウィンドウサイズ
        .MinSize(200, 100)          // これ以上小さくならないようにする
        .CloseButton(false).MaximizeButton(true)
    );

    // Run View
    //run_panel_ = new RunPanel(this);
    //m_mgr_.AddPane(run_panel_, wxAuiPaneInfo()
    //    .Name("RunPanel").Caption("Run Control")
    //    .Left().Layer(5).Position(0)
    //    .BestSize(250, 150)
    //    .MinSize(200, 100)
    //    .MaxSize(-1, 100)
    //    .CloseButton(false).MaximizeButton(true)
    //);

    // Module Browser
    //module_browser_ = new ModuleBrowser(this);
    //m_mgr_.AddPane(module_browser_, wxAuiPaneInfo()
    //    .Name("ModuleBrowser").Caption("Modules")
    //    .Left().Layer(5).Position(1)
    //    .BestSize(250, 800)
    //    .MinSize(200, 200)
    //    .CloseButton(false).MaximizeButton(true)
    //);

    // TrainExperienceView
    train_panel_ = new TrainPanel(this, train_panel_config);
    m_mgr_.AddPane(train_panel_, wxAuiPaneInfo()
        .Name("TrainExperiencePanel").Caption("Train View")
        //.Left().Layer(0)
        .Centre()
        .BestSize(400, 400)
        .CloseButton(false).MaximizeButton(true)
    );

    // EvalExperienceView
    eval_panel_ = new EvalPanel(this, eval_panel_config);
    m_mgr_.AddPane(eval_panel_, wxAuiPaneInfo()
        .Name("EvalExperiencePanel").Caption("Evaluation View")
        //.Centre()
        .Right().Layer(20)
        .BestSize(400, 400)
        .MinSize(200, 200)
        .CloseButton(true).MaximizeButton(true)
    );

    // 全てのペイン追加後に更新
    m_mgr_.Update();
}

void RunnerFrame::Initialize(std::shared_ptr<anet::rl::DefaultTrainer> trainer)
{
    train_panel_->Initialize(trainer);
    eval_panel_->Initialize(trainer);
    Layout();

    // 現在のENVに合ったTrainPanelを生成
    //auto train_view_panel = new TrainPanel(train_panel_, train_panel_config);

    //// TrainPanelを表示に反映
    //wxBoxSizer* train_sizer = new wxBoxSizer(wxVERTICAL);
    //train_sizer->Add(train_view_panel, 1, wxEXPAND | wxALL);
    //train_panel_->SetSizer(train_sizer);
    //train_panel_->Layout();

    // 現在のENVに合ったEvalPanelを生成
    //auto eval_view_panel = new EvalPanel(eval_panel_, eval_panel_config);

    // EvalPanelを表示に反映
    //wxBoxSizer* eval_sizer = new wxBoxSizer(wxVERTICAL);
    //eval_sizer->Add(eval_view_panel, 1, wxEXPAND | wxALL);
    //eval_panel_->SetSizer(eval_sizer);
    //eval_panel_->Layout();

}

void RunnerFrame::SetupEvents()
{
    // 基本イベント
    Bind(wxEVT_CLOSE_WINDOW, &RunnerFrame::OnClose, this);
    //Bind(wxEVT_MENU, [=](wxCommandEvent&) { Close(true); }, wxID_EXIT);

    // メニューイベント
    Bind(wxEVT_MENU, &RunnerFrame::OnExit, this, wxID_EXIT);
    Bind(wxEVT_MENU, &RunnerFrame::OnAbout, this, wxID_ABOUT);
    Bind(wxEVT_MENU, &RunnerFrame::OnResetLayout, this, ID_ResetLayout);
    Bind(wxEVT_MENU, &RunnerFrame::OnHeatMap, this, ID_HeatMap);
    //Bind(wxEVT_MENU, &LunarLanderFrame::OnViewHeatMap, this, ID_VIEW_HEAT_MAP, this);

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
        wxGetApp().ToggleTraining();
    else
        eval_panel_->TogglePause();
}

//void RunnerFrame::OnMouseRightClick(wxMouseEvent& event)
//{
//    //ToggleEval();
//}

//void RunnerFrame::ToggleEval()
//{
//    is_eval_pause_ = !is_eval_pause_;
//    LOG::info() << "Eval " << (is_eval_pause_ ? "paused." : " resumed.");
//}

void RunnerFrame::OnKey(anet::rl::gui::ForwardedKeyEvent& event)
{
    auto key_event = event.GetKeyEvent();
    //key_event.Get
    LOG::info() << "RunnerFrame::OnKey() key=" << key_event.GetKeyCode() << " eventType=" << key_event.GetEventType();
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
        wxGetApp().ToggleTraining();
        return;
    case WXK_SPACE:
        //ToggleEval();
        return;
    default:
        //eval_runner_->DoStep();
        //eval_canvas_->Refresh();
        event.Skip();
        return;
    }

    eval_panel_->DoStep(action);
    eval_panel_->Refresh();

    event.Skip();
}

void RunnerFrame::OnHeatMap(wxCommandEvent& event)
{
    auto trainer = wxGetApp().GetTrainer();
    auto env_spec = trainer->GetBatchEnv()->GetSpec();

    // ダイアログ生成
    SweepHeatMapDialog dialog(this, env_spec);

    if (dialog.ShowModal() == wxID_OK) {
        // 構造体でまとめて取得
        SweepHeatMapSettings s = dialog.GetSettings();

        auto heatmap_panel = new SweepHeatMapPanel(this, s.tag, s, trainer);
        m_mgr_.AddPane(heatmap_panel,
            PanelInfo("HeatMapPanel", "HeatMap", s.tag)
            .Right().Layer(20)          // Layer:大きいほど外側
            .BestSize(400, 400)          // ドッキング時の推奨サイズ 
            .FloatingSize(400, 400)     // 切り離したときのウィンドウサイズ
            .MinSize(100, 100)          // これ以上小さくならないようにする
            //.Float()
            .Dock()
            .CloseButton(true).MaximizeButton(true)
        );
        //heatmap_panel->Show();
        m_mgr_.Update();
    }
}

void RunnerFrame::OnResetLayout(wxCommandEvent& WXUNUSED(event))
{
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
    m_mgr_.UnInit();
    event.Skip();
}
