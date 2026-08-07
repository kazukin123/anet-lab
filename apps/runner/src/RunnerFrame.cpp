#include "RunnerFrame.hpp"
#include <algorithm>
#include "anet/log.hpp"
#include "RunnerApp.hpp"
#include "LogPanel.hpp"
#include "TrainPanel.hpp"
#include "HeatMapPanel.hpp"
#include "Conv2dPanel.hpp"

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

}  // namespace runner_layout

using namespace runner_layout;


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
};


RunnerFrame::RunnerFrame(const wxString& title, const TrainPanelConfig& train_panel_config, const EvalPanelConfig& eval_panel_config)
    : anet::rl::gui::AuiLayoutFrame(title, wxSize(800, 1024))
{
    // フラグ設定
    aui_mgr_.SetFlags(wxAUI_MGR_ALLOW_FLOATING | wxAUI_MGR_TRANSPARENT_DRAG | wxAUI_MGR_TRANSPARENT_HINT);
    // 既定の30%制約だと右側の複数列表示が狭くなるため、補助列を含めて広げられるようにする。
    aui_mgr_.SetDockSizeConstraint(0.85, 0.3);

    // 画面レイアウトを作る (SetupPanes は client size を読むため、メニュー/ステータスバーより後)
    SetupMenuBar();
    CreateStatusBar();
    SetupPanes(train_panel_config, eval_panel_config);

    // イベントハンドラ登録
    SetupEvents();

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

    // メニューイベント
    Bind(wxEVT_MENU, &RunnerFrame::OnExit, this, wxID_EXIT);
    Bind(wxEVT_MENU, &RunnerFrame::OnAbout, this, wxID_ABOUT);
    Bind(wxEVT_MENU, &RunnerFrame::OnResetLayout, this, ID_ResetLayout);
    Bind(wxEVT_MENU, &RunnerFrame::OnHeatMap, this, ID_HeatMap);
    Bind(wxEVT_MENU, &RunnerFrame::OnConv2d, this, ID_Conv2d);

    // パネル表示/非表示メニュー連動 (トグル・✕・チェック同期は基底が処理)
    RegisterPaneMenu(ID_LogView, log_panel_);
    RegisterPaneMenu(ID_TrainPanel, train_panel_);
    RegisterPaneMenu(ID_EvalPanel, eval_panel_);
    RegisterPaneMenu(ID_QValuePanel, q_value_panel_);

    // ログレベルメニュー
    Bind(wxEVT_MENU, [this](wxCommandEvent&) { if (log_panel_) log_panel_->SetLogLevel("info"); }, ID_LogLevelInfo);
    Bind(wxEVT_MENU, [this](wxCommandEvent&) { if (log_panel_) log_panel_->SetLogLevel("verbose"); }, ID_LogLevelVerbose);
    Bind(wxEVT_MENU, [this](wxCommandEvent&) { if (log_panel_) log_panel_->SetLogLevel("warn"); }, ID_LogLevelWarn);
    Bind(wxEVT_MENU, [this](wxCommandEvent&) { if (log_panel_) log_panel_->SetLogLevel("error"); }, ID_LogLevelError);

    // きーまう
    Bind(anet::rl::gui::EVT_FORWARDED_MOUSE, &RunnerFrame::OnMouse, this);
    Bind(anet::rl::gui::EVT_FORWARDED_KEY, &RunnerFrame::OnKey, this);
}

void RunnerFrame::OnPaneHiding(wxAuiPaneInfo& pane)
{
    // Eval を隠すときはフレーム縮退用に直前の幅を控える
    if (pane.window == eval_panel_) {
        pending_eval_compact_width_ = std::max(pending_eval_compact_width_, pane.rect.GetWidth());
    }
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

void RunnerFrame::RestoreFrameSizeIfNeeded()
{
    auto& eval_pane = aui_mgr_.GetPane(eval_panel_);
    if (!compact_restore_size_.has_value() || !eval_pane.IsOk() || !eval_pane.IsShown() || IsMaximized()) {
        return;
    }

    const wxSize restore_size = *compact_restore_size_;
    compact_restore_size_.reset();
    SetSize(wxDefaultCoord, wxDefaultCoord, restore_size.GetWidth(), restore_size.GetHeight(), wxSIZE_USE_EXISTING);
}

void RunnerFrame::CompactFrameForHiddenEval()
{
    auto& eval_pane = aui_mgr_.GetPane(eval_panel_);
    auto& train_pane = aui_mgr_.GetPane(train_panel_);
    if (!eval_pane.IsOk() || eval_pane.IsShown() || !train_pane.IsOk() || !train_pane.IsShown()) {
        pending_eval_compact_width_ = 0;
        return;
    }
    if (pending_eval_compact_width_ <= 0) {
        return;
    }
    if (compact_restore_size_.has_value() || IsMaximized()) {
        pending_eval_compact_width_ = 0;
        return;
    }

    int shrink_width = pending_eval_compact_width_;
    shrink_width = std::max(shrink_width, eval_pane.rect.GetWidth());
    shrink_width = std::max(shrink_width, eval_pane.best_size.GetWidth());
    shrink_width = std::max(shrink_width, kMinEvalWidth);
    pending_eval_compact_width_ = 0;

    const wxSize current_size = GetSize();
    const int min_width = std::max(480, GetMinSize().GetWidth());
    const int new_width = std::max(min_width, current_size.GetWidth() - shrink_width);
    if (new_width >= current_size.GetWidth()) return;

    compact_restore_size_ = current_size;
    SetSize(wxDefaultCoord, wxDefaultCoord, new_width, current_size.GetHeight(), wxSIZE_USE_EXISTING);
}

void RunnerFrame::OnApplyLayoutPolicy()
{
    // maximize 中は LoadLayout 往復が復元情報 (savedHiddenState) を壊すため何もしない。
    // restore 後に基底の restore ハンドラ経由で再適用される。
    if (HasMaximizedPane()) return;

    // Eval 非表示時のフレーム縮退と、再表示時の復元
    RestoreFrameSizeIfNeeded();
    CompactFrameForHiddenEval();

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
        eval_panel_->TogglePause();     // 右クリック：Evalトグル
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
    // Eval 非表示で縮退していた場合はフレームサイズを元へ戻す
    if (compact_restore_size_.has_value() && !IsMaximized()) {
        const wxSize restore_size = *compact_restore_size_;
        SetSize(wxDefaultCoord, wxDefaultCoord, restore_size.GetWidth(), restore_size.GetHeight(), wxSIZE_USE_EXISTING);
    }
    compact_restore_size_.reset();
    pending_eval_compact_width_ = 0;

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
