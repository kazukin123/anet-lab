#include "anet/gui.hpp"

namespace anet::rl::gui {
    // イベントIDの実体化
    wxDEFINE_EVENT(EVT_FORWARDED_MOUSE, anet::rl::gui::ForwardedMouseEvent);
    wxDEFINE_EVENT(EVT_FORWARDED_KEY, anet::rl::gui::ForwardedKeyEvent);
}

using namespace anet::rl::gui;
namespace LOG = anet::log;


// =============================================================
// ForwardedMouseEvent
// =============================================================

ForwardedMouseEvent::ForwardedMouseEvent(wxEventType commandType, int id)
    : wxCommandEvent(commandType, id), source_window_(nullptr)
{
    ;
}

ForwardedMouseEvent::ForwardedMouseEvent(const ForwardedMouseEvent& event)
    : wxCommandEvent(event), raw_event_(event.raw_event_), source_window_(event.source_window_)
{
    ;
}

wxEvent* ForwardedMouseEvent::Clone() const
{
    return new ForwardedMouseEvent(*this);
}


// =============================================================
// ForwardedKeyEvent
// =============================================================

ForwardedKeyEvent::ForwardedKeyEvent(wxEventType commandType, int id)
    : wxCommandEvent(commandType, id), source_window_(nullptr)
{
    ;
}

ForwardedKeyEvent::ForwardedKeyEvent(const ForwardedKeyEvent& event)
    : wxCommandEvent(event), raw_event_(event.raw_event_), source_window_(event.source_window_)
{
    ;
}

wxEvent* ForwardedKeyEvent::Clone() const
{
    return new ForwardedKeyEvent(*this);
}


// =============================================================
// Panel
// =============================================================

Panel::Panel(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style, const wxString& name)
    : wxPanel(parent, id, pos, size, style, name)
{
    Bind(wxEVT_LEFT_DOWN, &Panel::OnMouse, this);
    Bind(wxEVT_RIGHT_DOWN, &Panel::OnMouse, this);
    Bind(wxEVT_LEFT_DCLICK, &Panel::OnMouse, this);
    Bind(wxEVT_CHAR_HOOK, &Panel::OnKey, this);
}

void Panel::OnMouse(wxMouseEvent& event)
{
    SetFocus();
    ForwardedMouseEvent new_event(EVT_FORWARDED_MOUSE, GetId());
    new_event.SetEventObject(this);
    new_event.SetWindow(this);
    new_event.SetMouseEvent(event);
    ProcessEvent(new_event);
    event.Skip();
}

void Panel::OnKey(wxKeyEvent& event)
{
    //LOG::info() << "Panel::OnKey() type=" << event.GetEventType();
    ANET_LOG_DEBUG("KeyCode=" << event.GetKeyCode() << " EventType=" << event.GetEventType());

    // 多重発火防止： 自分が「最下層のPanel」なのか、それとも「すでに処理済みのイベントを受け取った親Panel」なのかを判定
    wxWindow* target = dynamic_cast<wxWindow*>(event.GetEventObject());
    bool childPanelExists = false;

    if (target && target != this) {
        // 発生元(target)自体が、自分(this)ではない別のPanelである場合をチェック
        bool is_panel = (dynamic_cast<Panel*>(target) != nullptr);
        if (is_panel) {
            childPanelExists = true;
        } else {
            // 発生元(target)から、自分(this)に辿り着くまでの親を順にチェック
            wxWindow* p = target->GetParent();
            while (p && p != this) {
                // もし途中にPanelクラスがあれば、その下位パネルがすでにイベントを発行しているはずなのでスルー
                if (dynamic_cast<Panel*>(p)) {
                    childPanelExists = true;
                    break;
                }
                p = p->GetParent();
            }
        }
    }

    // すでに下位のPanelが存在するなら、自分は何もせず標準動作(Skip)のみ行う
    if (childPanelExists) {
        event.Skip();
        return;
    }

    ForwardedKeyEvent new_event(EVT_FORWARDED_KEY, GetId());
    new_event.SetEventObject(this);
    new_event.SetWindow(this);
    new_event.SetKeyEvent(event);
    ProcessEvent(new_event);
    event.Skip();
}


// =============================================================
// AuiLayoutFrame
// =============================================================

// wxAuiManager::SaveLayout / LoadLayout と AuiLayoutFrame の間を仲介する serializer 実装。
namespace aui_layout_detail {

    // SaveLayout の結果を vector に貯めるだけの serializer。
    // dock_size には wxAUI が内部 dock の実サイズを入れてくれる。
    class LayoutSnapshot final : public wxAuiSerializer {
    public:
        void SavePane(const wxAuiPaneLayoutInfo& pane) override { panes_.push_back(pane); }

        // wxAuiNotebook は不使用のため空実装
        void BeforeSaveNotebook(const wxString&) override {}
        void SaveNotebookTabControl(const wxAuiTabLayoutInfo&) override {}

        std::vector<wxAuiPaneLayoutInfo>& Panes() { return panes_; }
    private:
        std::vector<wxAuiPaneLayoutInfo> panes_;
    };

    // 編集済み snapshot を wxAuiManager::LoadLayout へ流し込む deserializer。
    // AfterLoad の既定実装が Update() を 1 回呼ぶ。
    class LayoutRestorer final : public wxAuiDeserializer {
    public:
        LayoutRestorer(wxAuiManager& manager, std::vector<wxAuiPaneLayoutInfo> panes)
            : wxAuiDeserializer(manager), panes_(std::move(panes)) {}

        std::vector<wxAuiPaneLayoutInfo> LoadPanes() override { return std::move(panes_); }
        std::vector<wxAuiTabLayoutInfo> LoadNotebookTabs(const wxString&) override { return {}; }
    private:
        std::vector<wxAuiPaneLayoutInfo> panes_;
    };

}   // namespace aui_layout_detail

AuiLayoutFrame::AuiLayoutFrame(const wxString& title, const wxSize& size)
    : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, size)
{
    // AUI Managerの管理下に置く
    aui_mgr_.SetManagedWindow(this);

    // レイアウト機構が必要とするイベントを購読する (メニューは RegisterPaneMenu で個別に Bind)
    Bind(wxEVT_SIZE, &AuiLayoutFrame::OnFrameSize, this);
    Bind(wxEVT_AUI_PANE_CLOSE, &AuiLayoutFrame::OnAuiPaneClose, this);
    Bind(wxEVT_AUI_PANE_MAXIMIZE, &AuiLayoutFrame::OnAuiPaneMaximize, this);
    Bind(wxEVT_AUI_PANE_RESTORE, &AuiLayoutFrame::OnAuiPaneRestore, this);
}

AuiLayoutFrame::~AuiLayoutFrame()
{
    aui_mgr_.UnInit();
}

void AuiLayoutFrame::RegisterPaneMenu(int menu_id, wxWindow* window)
{
    pane_menu_bindings_.push_back({ .menu_id = menu_id, .window = window });
    Bind(wxEVT_MENU, &AuiLayoutFrame::OnTogglePaneMenu, this, menu_id);
}

void AuiLayoutFrame::UpdatePaneMenuChecks()
{
    for (const auto& binding : pane_menu_bindings_) {
        auto& pane = aui_mgr_.GetPane(binding.window);
        if (pane.IsOk()) {
            CheckPaneMenuItem(binding.menu_id, pane.IsShown());
        }
    }
}

void AuiLayoutFrame::SchedulePolicy()
{
    // リサイズ嵐などでの多重実行を 1 回に集約する
    if (layout_policy_scheduled_) return;

    layout_policy_scheduled_ = true;
    CallAfter([this]() {
        layout_policy_scheduled_ = false;
        ApplyLayoutPolicy();
    });
}

void AuiLayoutFrame::SyncDockSizesToPanes()
{
    // 表示中 pane が属する dock の実サイズを pane.dock_size へ書き戻す。
    // dock が消えても (hide/maximize)、dock 再生成時に pane.dock_size が
    // 読まれるため、サイズが復元される。
    aui_layout_detail::LayoutSnapshot snapshot;
    aui_mgr_.SaveLayout(snapshot);
    for (const auto& info : snapshot.Panes()) {
        if (info.dock_size <= 0) continue;   // 非表示・浮動 pane は記憶値を保持
        auto& live = aui_mgr_.GetPane(info.name);
        if (live.IsOk()) live.dock_size = info.dock_size;
    }
}

bool AuiLayoutFrame::HasMaximizedPane() const
{
    const auto& panes = aui_mgr_.GetAllPanes();
    for (size_t i = 0; i < panes.GetCount(); ++i) {
        if (panes.Item(i).IsMaximized()) return true;
    }
    return false;
}

std::vector<wxAuiPaneLayoutInfo> AuiLayoutFrame::TakeLayoutSnapshot()
{
    // SaveLayout は dock に属さない pane (非表示・浮動) の dock_size を 0 にするため、
    // そのまま LoadLayout へ戻すと pane 側の記憶幅が消える。0 の pane は live の
    // 記憶値 (pane.dock_size) で補完する。
    aui_layout_detail::LayoutSnapshot snapshot;
    aui_mgr_.SaveLayout(snapshot);
    for (auto& info : snapshot.Panes()) {
        if (info.dock_size > 0) continue;
        auto& live = aui_mgr_.GetPane(info.name);
        if (live.IsOk()) info.dock_size = live.dock_size;
    }
    return std::move(snapshot.Panes());
}

void AuiLayoutFrame::ApplyLayoutSnapshot(std::vector<wxAuiPaneLayoutInfo> panes)
{
    aui_layout_detail::LayoutRestorer restorer(aui_mgr_, std::move(panes));
    aui_mgr_.LoadLayout(restorer);
}

wxAuiPaneLayoutInfo* AuiLayoutFrame::FindPaneLayout(std::vector<wxAuiPaneLayoutInfo>& panes, const wxString& name)
{
    for (auto& info : panes) {
        if (info.name == name) return &info;
    }
    return nullptr;
}

bool AuiLayoutFrame::IsInHomeDock(const wxAuiPaneInfo& pane, int direction, int layer)
{
    return pane.IsOk() && pane.IsShown() && pane.IsDocked()
        && pane.dock_direction == direction
        && pane.dock_layer == layer;
}

wxAuiPaneInfo AuiLayoutFrame::MakeUniquePaneInfo(const wxString& name, const wxString& caption, const wxString& sub_caption)
{
    long long timestamp = wxGetLocalTimeMillis().GetValue();
    wxString unique_name = wxString::Format("%s_%lld", name, timestamp);
    auto new_caption = caption;
    if (!sub_caption.empty()) new_caption += " " + sub_caption;
    return wxAuiPaneInfo().Name(unique_name).Caption(new_caption);
}

void AuiLayoutFrame::OnFrameSize(wxSizeEvent& event)
{
    event.Skip();
    SchedulePolicy();
}

void AuiLayoutFrame::OnTogglePaneMenu(wxCommandEvent& event)
{
    const PaneMenuBinding* binding = FindBindingByMenuId(event.GetId());
    if (!binding) return;

    auto& pane = aui_mgr_.GetPane(binding->window);
    if (!pane.IsOk()) return;

    if (!event.IsChecked()) {
        // 非表示化で dock が消える前に、実 dock サイズを pane 側へ退避する
        SyncDockSizesToPanes();
        OnPaneHiding(pane);
    }

    pane.Show(event.IsChecked());
    aui_mgr_.Update();
    ApplyLayoutPolicy();
}

void AuiLayoutFrame::OnAuiPaneClose(wxAuiManagerEvent& event)
{
    wxAuiPaneInfo* pane = event.GetPane();
    if (!pane) {
        event.Skip();
        return;
    }

    // 実際の hide+Update は wxAUI 側がこの後で行うため、dock が実在するここで退避する
    SyncDockSizesToPanes();

    // メニューのチェックを外す
    if (const PaneMenuBinding* binding = FindBindingByWindow(pane->window)) {
        CheckPaneMenuItem(binding->menu_id, false);
    }
    OnPaneHiding(*pane);

    // ポリシー適用はクローズ処理の完了後に行う必要があるため遅延させる
    SchedulePolicy();
    event.Skip();
}

void AuiLayoutFrame::OnAuiPaneMaximize(wxAuiManagerEvent& event)
{
    // maximize で他 pane の dock が消える前に、実 dock サイズを退避する
    SyncDockSizesToPanes();
    event.Skip();
}

void AuiLayoutFrame::OnAuiPaneRestore(wxAuiManagerEvent& event)
{
    // restore によるレイアウト再構築の完了後にポリシーを再適用する
    SchedulePolicy();
    event.Skip();
}

const AuiLayoutFrame::PaneMenuBinding* AuiLayoutFrame::FindBindingByMenuId(int menu_id) const
{
    for (const auto& binding : pane_menu_bindings_) {
        if (binding.menu_id == menu_id) return &binding;
    }
    return nullptr;
}

const AuiLayoutFrame::PaneMenuBinding* AuiLayoutFrame::FindBindingByWindow(wxWindow* window) const
{
    for (const auto& binding : pane_menu_bindings_) {
        if (binding.window == window) return &binding;
    }
    return nullptr;
}

void AuiLayoutFrame::CheckPaneMenuItem(int menu_id, bool checked)
{
    wxMenuBar* menu_bar = GetMenuBar();
    if (!menu_bar) return;
    // メニュー項目を持たない pane 登録を許容する
    if (!menu_bar->FindItem(menu_id)) return;
    menu_bar->Check(menu_id, checked);
}


// =============================================================
// ViewRepository
// =============================================================

void ViewRepository::Register(std::shared_ptr<ViewCreator> creator)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto class_id = creator->GetTargetClassId();
    creators_[class_id] = creator;
}

std::shared_ptr<ViewCreator> ViewRepository::GetViewCreator(const std::string& class_id) const
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = creators_.find(class_id);
    if (it == creators_.end()) return nullptr;
    return it->second;
}


// =============================================================
// DefaultViewFactory
// =============================================================

DefaultViewFactory::DefaultViewFactory(const anet::ConfigData& config_data)
    : config_data_(config_data)
{
    ;
}

std::shared_ptr<View> DefaultViewFactory::CreateView(
    wxWindow* parent, const std::string& class_id, std::shared_ptr<Notifier> notifier) const
{
    auto creator = ViewRepository::Instance().GetViewCreator(class_id);
    if (creator == nullptr)
        return nullptr;

    auto view  = creator->CreateView(parent, config_data_, notifier);

    return view;
}
