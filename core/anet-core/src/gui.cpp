#include "anet/gui.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <sstream>
#include "anet/env.hpp"
#include "anet/metrics.hpp"

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

bool anet::rl::gui::IsPaneOpen(const wxAuiPaneInfo& pane, bool has_maximized_pane)
{
    // wxAUIが最大化時に退避するのはdockされた非toolbar paneの非表示フラグだけ。
    return pane.IsOk() && (pane.IsShown() || (has_maximized_pane && pane.IsDocked()
        && !pane.IsToolbar() && !pane.HasFlag(wxAuiPaneInfo::savedHiddenState)));
}

void AuiLayoutFrame::UpdatePaneMenuChecks()
{
    for (const auto& binding : pane_menu_bindings_) {
        auto& pane = aui_mgr_.GetPane(binding.window);
        if (pane.IsOk()) {
            CheckPaneMenuItem(binding.menu_id, IsPaneOpen(pane, HasMaximizedPane()));
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

    // 最大化解除で明示的な開閉が巻き戻されないよう、退避フラグも同期する。
    pane.SetFlag(wxAuiPaneInfo::savedHiddenState, !event.IsChecked());
    if (event.IsChecked()) {
        OnPaneShowing(pane);
    } else {
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
    pane->SetFlag(wxAuiPaneInfo::savedHiddenState, true);
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

// Evaluation Traceの表示・再生補助
wxBitmapBundle anet::rl::gui::MakePlaybackBitmap(PlaybackIcon icon, const wxColour& colour)
{
    // 形と色を分け、テーマが変わった時だけSVGを再生成できるようにする。
    const std::array<const char*, 7> shapes{
        R"(<path d="M4 2 L13 8 L4 14 Z"/>)",
        R"(<path d="M4 2 H7 V14 H4 Z M9 2 H12 V14 H9 Z"/>)",
        R"(<rect x="3" y="3" width="10" height="10"/>)",
        R"(<path d="M10 2 L2 8 L10 14 Z M12 2 H14 V14 H12 Z"/>)",
        R"(<path d="M2 2 H4 V14 H2 Z M6 2 L14 8 L6 14 Z"/>)",
        R"(<path d="M8 2 L1 8 L8 14 Z M15 2 L8 8 L15 14 Z"/>)",
        R"(<path d="M1 2 L8 8 L1 14 Z M8 2 L15 8 L8 14 Z"/>)"};
    const auto svg = wxString::Format(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 16 16\"><g fill=\"%s\">%s</g></svg>",
        colour.GetAsString(wxC2S_HTML_SYNTAX), wxString::FromUTF8(shapes.at(static_cast<size_t>(icon))));
    return wxBitmapBundle::FromSVG(svg.utf8_str(), wxSize(16, 16));
}

void TraceRowSorter::Reset(std::optional<size_t> step_column)
{
    // 開き直しでは保持値と並べ替えの両方を既定へ戻す。
    rows_.clear();
    step_column_ = step_column;
    column_ = step_column;
    ascending_ = false;
}

bool TraceRowSorter::ComesBefore(const Row& left, const Row& right) const
{
    // step列はfloatの丸めを避け、行が持つ実stepで比較する。
    if (column_ && column_ != step_column_) {
        const float a = left.values.at(*column_);
        const float b = right.values.at(*column_);
        const bool a_finite = std::isfinite(a);
        const bool b_finite = std::isfinite(b);
        if (a_finite != b_finite) return a_finite;
        if (a_finite && a != b) return ascending_ ? a < b : a > b;
    } else if (left.step != right.step) {
        return ascending_ ? left.step < right.step : left.step > right.step;
    }
    // 同値の中は向きにかかわらずstepの新しい順を保つ。
    if (left.step != right.step) return left.step > right.step;
    return left.id > right.id;
}

void TraceRowSorter::ToggleColumn(size_t column)
{
    ANET_PROFILE_FUNC();
    // 別の列は降順から始め、同じ列だけ向きを反転する。
    ascending_ = column_ == column ? !ascending_ : false;
    column_ = column;
    std::sort(rows_.begin(), rows_.end(), [this](const auto& a, const auto& b) { return ComesBefore(a, b); });
}

size_t TraceRowSorter::Upsert(Row row)
{
    ANET_PROFILE_FUNC();
    // 遅延確定した行だけを取り出し、新規行と同じ挿入経路へ戻す。
    const auto previous = std::find_if(rows_.begin(), rows_.end(), [&](const auto& value) { return value.id == row.id; });
    if (previous != rows_.end()) rows_.erase(previous);
    // 既定の新しい行は先頭へ足すだけにし、全行の値や画像を動かさない。
    const auto position = rows_.empty() || ComesBefore(row, rows_.front()) ? rows_.begin()
        : std::lower_bound(rows_.begin(), rows_.end(), row,
            [this](const auto& a, const auto& b) { return ComesBefore(a, b); });
    const auto index = static_cast<size_t>(std::distance(rows_.begin(), position));
    rows_.insert(position, std::move(row));
    return index;
}

std::optional<int64_t> TraceRowSorter::Trim(size_t max_rows)
{
    ANET_PROFILE_FUNC();
    ANET_CHECK_MSG(max_rows > 0, "Invalid trace row limit. max_rows=" << max_rows << " expected=integer >= 1.");
    if (rows_.size() <= max_rows) return std::nullopt;
    // 表示順と独立に最古を選ぶ。既定のstep降順では末尾だけで決まる。
    const auto oldest = !ascending_ && column_ == step_column_ ? std::prev(rows_.end())
        : std::min_element(rows_.begin(), rows_.end(), [](const auto& a, const auto& b) {
            return a.step != b.step ? a.step < b.step : a.id < b.id;
        });
    const int64_t id = oldest->id;
    rows_.erase(oldest);
    return id;
}

std::vector<int64_t> TraceRowSorter::GetDisplayOrder() const
{
    std::vector<int64_t> ids;
    ids.reserve(rows_.size());
    for (const auto& row : rows_) ids.push_back(row.id);
    return ids;
}

namespace {

std::string TraceDumpPrefix(const anet::rl::StepTrace& trace)
{
    return "Trace dump step=" + std::to_string(trace.step) + " lane=" + std::to_string(trace.lane) + ":";
}

torch::Tensor TraceDumpCpuTensor(const torch::Tensor& value)
{
    return value.defined() ? value.to(torch::kCPU) : value;
}

anet::TensorDict TraceDumpCpuDict(const anet::TensorDict& values)
{
    anet::TensorDict cpu;
    for (const auto& [key, value] : values) cpu.Set(key, TraceDumpCpuTensor(value));
    return cpu;
}

anet::rl::AuxData TraceDumpCpuAux(const anet::rl::AuxData& values)
{
    anet::rl::AuxData cpu;
    for (const auto& [key, value] : values) cpu.emplace(key, TraceDumpCpuTensor(value));
    return cpu;
}

anet::rl::BatchState TraceDumpCpuState(const anet::rl::BatchState& state)
{
    return {TraceDumpCpuDict(state.obs), TraceDumpCpuTensor(state.done),
        TraceDumpCpuTensor(state.truncated), TraceDumpCpuTensor(state.episode_start)};
}

void AppendTraceTensorSummary(std::ostream& output, const torch::Tensor& value)
{
    // 要素が一つのときだけ値を読む。大きなTensorはdeviceも元のまま知らせる。
    if (!value.defined()) output << "undefined";
    else if (value.numel() == 1) output << TraceDumpCpuTensor(value).item<double>();
    else output << value.sizes() << ' ' << value.scalar_type() << ' ' << value.device();
}

template<class Dictionary>
void AppendTraceTensorFields(std::ostream& output, const char* prefix, const Dictionary& values)
{
    // mapの実装によらず、同じ通知の項目は同じ順で読めるようにする。
    std::vector<std::pair<std::string, torch::Tensor>> ordered(values.begin(), values.end());
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [key, value] : ordered) {
        output << ' ' << prefix << key << '=';
        AppendTraceTensorSummary(output, value);
    }
}

} // namespace

std::string TraceDumpFormatter::FormatInfo(const Data& data, const anet::rl::StepTrace& trace,
    std::span<const anet::rl::StepTraceColumnSpec> columns, std::span<const float> displayed_values,
    const anet::rl::ActionSpec& actions)
{
    ANET_PROFILE_FUNC();
    ANET_CHECK_MSG(data.action_info && data.step_result,
        "Trace dump notification is unavailable. step=" << trace.step << " expected=action_info and step_result.");
    ANET_CHECK_MSG(columns.size() == displayed_values.size() && columns.size() <= trace.value_ready.size(),
        "Invalid trace dump columns. step=" << trace.step << " columns=" << columns.size()
        << " values=" << displayed_values.size() << " expected=matching column and value counts.");
    std::ostringstream output;
    output.precision(4);
    output << TraceDumpPrefix(trace);
    // 列の構成によらない通知項目は、列のあとへ必ず付ける。
    for (size_t i = 0; i < columns.size(); ++i) {
        output << ' ' << columns[i].label << '=';
        if (!trace.value_ready[i]) output << "pending";
        else output << FormatTraceValue(columns[i], displayed_values[i], actions);
    }
    output << " runner=" << data.runner_name << " env=" << data.env_name << " agent=" << data.agent_name
        << " counts.train_step=" << data.counts.train_step << " counts.exp_step=" << data.counts.exp_step
        << " counts.update_step=" << data.counts.update_step << " counts.learn_step=" << data.counts.learn_step
        << " counts.episode_count=" << data.counts.episode_count << " counts.sim_step=" << data.counts.sim_step
        << " action_info.action=";
    const auto action = data.action_info->GetAction();
    if (actions.is_discrete && action.defined() && action.dim() == 1) {
        const auto index = TraceDumpCpuTensor(action)[trace.lane].item<int64_t>();
        output << index << ':' << actions.value_labels.at(static_cast<size_t>(index));
    } else AppendTraceTensorSummary(output, action);
    output << " runner.forced=" << data.forced << " step_result.reward=";
    AppendTraceTensorSummary(output, data.step_result->reward);
    output << " step_result.done=" << trace.done << " step_result.truncated=" << trace.truncated;
    AppendTraceTensorFields(output, "action_info.info.", data.action_info->GetInfo());
    AppendTraceTensorFields(output, "action_info.aux.", data.action_info->GetAuxData());
    for (const auto& aux : data.step_result->GetAuxDataList(static_cast<int>(trace.lane))) {
        AppendTraceTensorFields(output, "step_result.aux.", aux);
    }
    auto text = output.str();
    std::replace(text.begin(), text.end(), '\r', ' ');
    std::replace(text.begin(), text.end(), '\n', ' ');
    return text;
}

std::string TraceDumpFormatter::FormatVerbose(const Data& data, const anet::rl::StepTrace& trace)
{
    ANET_PROFILE_FUNC();
    ANET_CHECK_MSG(data.action_info && data.step_result,
        "Trace dump notification is unavailable. step=" << trace.step << " expected=action_info and step_result.");
    // 保持中は転送せず、要求されたこの場でCPUの部品を作り、既存の文字列化へ渡す。
    const auto state = TraceDumpCpuState(data.state);
    const auto info = TraceDumpCpuDict(data.action_info->GetInfo());
    const anet::rl::BatchActionInfo action(TraceDumpCpuTensor(data.action_info->GetAction()), info,
        TraceDumpCpuAux(data.action_info->GetAuxData()));
    auto aux = data.step_result->GetAuxDataList();
    for (auto& values : aux) values = TraceDumpCpuAux(values);
    const anet::rl::PlainBatchStepResult result(TraceDumpCpuTensor(data.step_result->reward),
        TraceDumpCpuState(data.step_result->next_state), TraceDumpCpuState(data.step_result->continue_state),
        data.step_result->n_transitions, data.step_result->n_episode_end, std::move(aux));
    std::ostringstream output;
    output << TraceDumpPrefix(trace) << " verbose\nstate=" << state.ToString()
        << "\naction_info=" << action.ToString() << "\naction_info.info=" << info.ToString();
    if (const auto& hint = data.action_info->GetReplayInitialPriorityHint()) {
        output << "\naction_info.replay_initial_priority_hint=" << anet::ToString(TraceDumpCpuTensor(hint->GetPayload()));
    }
    output << "\nstep_result=" << result.ToString();
    return output.str();
}

void TracePlayer::SetRows(std::vector<int64_t> playback_order)
{
    // 表の下端から上端の再生順を受け、行IDで選択を追跡する。
    rows_ = std::move(playback_order);
    if (rows_.empty()) {
        selected_.reset();
        playing_ = false;
        progress_ = 0;
    } else if (selected_ && std::find(rows_.begin(), rows_.end(), *selected_) == rows_.end()) {
        selected_ = rows_.front();
        progress_ = 0;
    }
}

void TracePlayer::Select(int64_t id, bool keep_playing)
{
    ANET_CHECK_MSG(std::find(rows_.begin(), rows_.end(), id) != rows_.end(),
        "Trace selection is unavailable. id=" << id << " expected=a retained row.");
    selected_ = id;
    if (!keep_playing) playing_ = false;
    progress_ = 0;
}

bool TracePlayer::CanPlay(double base_steps_per_second) const
{
    ANET_CHECK_MSG(std::isfinite(base_steps_per_second) && base_steps_per_second >= 0,
        "Invalid trace playback rate. base_rate=" << base_steps_per_second << " expected=finite nonnegative value.");
    return !rows_.empty() && base_steps_per_second > 0;
}

std::string TracePlayer::Play()
{
    if (rows_.empty()) return {};
    // Liveと上端での再生開始だけを下端へ戻す。
    if (!selected_ || (!playing_ && *selected_ == rows_.back())) selected_ = rows_.front();
    playing_ = true;
    progress_ = 0;
    return "Trace: play";
}

std::string TracePlayer::Pause()
{
    if (rows_.empty()) return {};
    if (!selected_) selected_ = rows_.back();
    playing_ = false;
    progress_ = 0;
    return "Trace: pause";
}

std::string TracePlayer::Stop()
{
    selected_.reset();
    playing_ = false;
    progress_ = 0;
    return "Trace: stop (live)";
}

std::string TracePlayer::StepForward()
{
    if (rows_.empty()) return {};
    // Liveは表の上端を起点とし、通常の端判定へ進める。
    if (!selected_) selected_ = rows_.back();
    playing_ = false;
    progress_ = 0;
    const auto found = std::find(rows_.begin(), rows_.end(), *selected_);
    if (std::next(found) == rows_.end()) return "Trace: end";
    selected_ = *std::next(found);
    return "Trace: step";
}

std::string TracePlayer::StepBack()
{
    if (rows_.empty()) return {};
    // Liveも表の上端から一つ戻し、1件だけならstartになる。
    if (!selected_) selected_ = rows_.back();
    playing_ = false;
    progress_ = 0;
    const auto found = std::find(rows_.begin(), rows_.end(), *selected_);
    if (found == rows_.begin()) return "Trace: start";
    selected_ = *std::prev(found);
    return "Trace: step back";
}

double TracePlayer::GetSpeed() const
{
    return std::ldexp(1.0, speed_index_ - 3);
}

std::string TracePlayer::GetSpeedLabel() const
{
    const std::array<const char*, 7> labels{ "x0.125", "x0.25", "x0.5", "x1", "x2", "x4", "x8" };
    return labels[speed_index_];
}

std::string TracePlayer::Faster()
{
    const bool at_end = speed_index_ == 6;
    if (!at_end) ++speed_index_;
    return "Trace: speed " + GetSpeedLabel() + (at_end ? " (max)" : "");
}

std::string TracePlayer::Slower()
{
    const bool at_end = speed_index_ == 0;
    if (!at_end) --speed_index_;
    return "Trace: speed " + GetSpeedLabel() + (at_end ? " (min)" : "");
}

std::string TracePlayer::Tick(double seconds, double base_steps_per_second)
{
    ANET_CHECK_MSG(std::isfinite(seconds) && seconds >= 0 && std::isfinite(base_steps_per_second)
        && base_steps_per_second >= 0, "Invalid trace playback clock. seconds=" << seconds
        << " base_rate=" << base_steps_per_second << " expected=finite nonnegative values.");
    if (!playing_ || !selected_ || rows_.empty()) return {};
    // 端へ到達した瞬間に止め、実行環境のfps変更にも次のtickから追随する。
    progress_ += seconds * base_steps_per_second * GetSpeed();
    while (progress_ >= 1.0) {
        progress_ -= 1.0;
        auto found = std::find(rows_.begin(), rows_.end(), *selected_);
        if (std::next(found) != rows_.end()) selected_ = *std::next(found);
        if (*selected_ == rows_.back()) {
            playing_ = false;
            progress_ = 0;
            return "Trace: end";
        }
    }
    return {};
}

float anet::rl::gui::AppendTracePercentile(std::deque<float>& values, float value, size_t window_size)
{
    ANET_PROFILE_FUNC();
    ANET_CHECK_MSG(window_size > 0, "Trace percentile window must be positive. actual=" << window_size);
    // 不成立値は窓を進めず、有限値の直近窓へ自分を含めて中央順位を求める。
    if (!std::isfinite(value)) return std::numeric_limits<float>::quiet_NaN();
    values.push_back(value);
    if (values.size() > window_size) values.pop_front();
    size_t less = 0;
    size_t equal = 0;
    for (const float previous : values) {
        if (previous < value) ++less;
        else if (previous == value) ++equal;
    }
    return static_cast<float>(100.0 * (static_cast<double>(less) + 0.5 * equal) / values.size());
}

std::string anet::rl::gui::FormatTraceValue(const StepTraceColumnSpec& column, float value, const ActionSpec& actions)
{
    // 空欄は非有限値だけに使い、大きな整数を有効数字で丸めない。
    if (!std::isfinite(value)) return {};
    if (column.target == "$action_info" && column.key == "action") {
        ANET_CHECK_MSG(actions.is_discrete && std::trunc(value) == value && value >= 0
            && value < static_cast<float>(actions.value_labels.size()),
            "Trace action is out of range. column=" << column.number << " value=" << value
                << " expected=[0," << actions.value_labels.size() << ").");
        const auto index = static_cast<size_t>(value);
        if (column.display == "name") return actions.value_labels[index];
        if (column.display == "value_name") return std::to_string(index) + ":" + actions.value_labels[index];
    }
    return std::trunc(value) == value ? std::format("{:.0f}", value) : std::format("{:.4g}", value);
}
