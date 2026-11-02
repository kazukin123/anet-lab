// EvalTracePanel.cpp
#include "EvalTracePanel.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <wx/dcbuffer.h>
#include <wx/settings.h>
#include "anet/log.hpp"
#include "anet/observers.hpp"
#include "anet/profile.hpp"

using namespace anet::rl;
using anet::rl::gui::TracePlayer;

namespace {
// 見出しを中央寄せしても、右側のソート印と文字が重ならない余白を確保する。
constexpr int kColumnLabelPaddingDip = 28;

class TraceGrid final : public wxGrid {
public:
    using wxGrid::wxGrid;

    void DrawColLabel(wxDC& dc, int col) override
    {
        // ラベルと枠は標準描画に任せ、ソート印だけをシステム色で描く。
        wxGrid::DrawColLabel(dc, col);
        if (!IsSortingBy(col) || GetColSize(col) < FromDIP(16) || GetColLabelSize() < FromDIP(8)) return;
        const int x = GetColRight(col) - FromDIP(8);
        const int y = GetColLabelSize() / 2;
        const int half_width = FromDIP(4);
        const int half_height = FromDIP(3);
        const int direction = IsSortOrderAscending() ? -1 : 1;
        const std::array<wxPoint, 3> triangle{
            wxPoint(x - half_width, y - direction * half_height),
            wxPoint(x + half_width, y - direction * half_height),
            wxPoint(x, y + direction * half_height)};
        // 文字グリフを使わず、テーマやDPIの変更後もその場の色と寸法に従う。
        const auto colour = wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT);
        const wxDCPenChanger pen(dc, wxPen(colour));
        const wxDCBrushChanger brush(dc, wxBrush(colour));
        dc.DrawPolygon(static_cast<int>(triangle.size()), triangle.data());
    }
};
}

class EvalTracePanel::Table final : public wxGridTableBase {
public:
    explicit Table(EvalTracePanel& owner) : owner_(owner) {}
    int GetNumberRows() override { return static_cast<int>(owner_.rows_.size()); }
    int GetNumberCols() override { return static_cast<int>(owner_.columns_.size()); }
    bool IsEmptyCell(int row, int col) override { return GetValue(row, col).empty(); }
    void SetValue(int, int, const wxString&) override {}
    wxString GetColLabelValue(int col) override { return wxString::FromUTF8(owner_.columns_[col].label); }
    wxString GetValue(int row, int col) override
    {
        const auto& data = owner_.rows_.at(static_cast<size_t>(row));
        const auto& column = owner_.columns_.at(static_cast<size_t>(col));
        const float value = column.display == "pct" ? data.percentiles[col] : data.trace->values[col];
        return wxString::FromUTF8(gui::FormatTraceValue(column, value, owner_.env_spec_.action_spec));
    }
    wxGridCellAttr* GetAttr(int row, int, wxGridCellAttr::wxAttrKind) override
    {
        const auto& trace = *owner_.rows_.at(static_cast<size_t>(row)).trace;
        if (!trace.done && !trace.truncated && trace.reward == 0) return nullptr;
        auto* attr = new wxGridCellAttr;
        // 通常背景へ薄く重ねる。終端を優先し、文字と選択色はシステムに従う。
        const auto base = owner_.grid_->GetDefaultCellBackgroundColour();
        const auto tint = trace.done || trace.truncated ? wxColour(240, 140, 0) : wxColour(0, 170, 80);
        const auto blend = [](int background, int foreground) { return (84 * background + 16 * foreground) / 100; };
        attr->SetBackgroundColour(wxColour(blend(base.Red(), tint.Red()), blend(base.Green(), tint.Green()), blend(base.Blue(), tint.Blue())));
        attr->SetTextColour(owner_.grid_->GetDefaultCellTextColour());
        attr->SetReadOnly();
        return attr;
    }
private:
    EvalTracePanel& owner_; ///< このTableをGrid経由で所有するEvalTracePanelを借用
};

EvalTracePanel::EvalTracePanel(wxFrame& frame, wxAuiManager& manager, EvalPanel& eval, const anet::ConfigData& config)
    : wxPanel(&frame), frame_(frame), manager_(manager), eval_(eval), config_(config), timer_(this)
    , last_tick_(std::chrono::steady_clock::now())
{
    // 表示上限は非表示で起動した場合も構築時に検証する。
    max_rows_ = config.Get<int>("app.eval_trace_panel.max_rows", 1000);
    pct_window_ = config.Get<int>("app.eval_trace_panel.pct_window", 10000);
    if (max_rows_ < 1) ANET_SYSTEM_ERROR("Invalid app.eval_trace_panel.max_rows: value=" << max_rows_ << " expected=integer >= 1.");
    if (pct_window_ < 1) ANET_SYSTEM_ERROR("Invalid app.eval_trace_panel.pct_window: value=" << pct_window_ << " expected=integer >= 1.");
    auto* layout = new wxBoxSizer(wxVERTICAL);
    // 枠の領域を常時予約し、LiveとTraceの切替で操作位置を変えない。
    controls_panel_ = new wxPanel(this);
    controls_panel_->SetBackgroundStyle(wxBG_STYLE_PAINT);
    controls_panel_->Bind(wxEVT_PAINT, [this](wxPaintEvent&) {
        wxAutoBufferedPaintDC dc(controls_panel_);
        dc.SetBackground(wxBrush(controls_panel_->GetBackgroundColour()));
        dc.Clear();
        if (!player_.GetSelected()) return;
        dc.SetPen(wxPen(wxColour(220, 45, 45), controls_panel_->FromDIP(2)));
        dc.SetBrush(*wxTRANSPARENT_BRUSH);
        auto rect = controls_panel_->GetClientRect();
        rect.Deflate(controls_panel_->FromDIP(1));
        dc.DrawRectangle(rect);
    });
    auto* controls = new wxBoxSizer(wxHORIZONTAL);
    toolbar_ = new wxAuiToolBar(controls_panel_, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxAUI_TB_DEFAULT_STYLE);
    toolbar_->SetToolBitmapSize(wxSize(16, 16));
    toolbar_->SetOverflowVisible(false);
    struct Operation { const char* help; gui::PlaybackIcon icon; std::string (TracePlayer::*method)(); };
    const std::array<Operation, 6> operations{{
        {.help = "Play", .icon = gui::PlaybackIcon::Play, .method = &TracePlayer::Play},
        {.help = "Stop and return to Live", .icon = gui::PlaybackIcon::Stop, .method = &TracePlayer::Stop},
        {.help = "Step back", .icon = gui::PlaybackIcon::StepBack, .method = &TracePlayer::StepBack},
        {.help = "Step forward", .icon = gui::PlaybackIcon::StepForward, .method = &TracePlayer::StepForward},
        {.help = "Slower", .icon = gui::PlaybackIcon::Slower, .method = &TracePlayer::Slower},
        {.help = "Faster", .icon = gui::PlaybackIcon::Faster, .method = &TracePlayer::Faster}}};
    for (size_t i = 0; i < operations.size(); ++i) {
        const auto& operation = operations[i];
        const int id = tool_ids_[i] = wxWindow::NewControlId();
        toolbar_->AddTool(id, wxEmptyString, gui::MakePlaybackBitmap(operation.icon,
            wxSystemSettings::GetColour(wxSYS_COLOUR_BTNTEXT)), operation.help);
        toolbar_->Bind(wxEVT_TOOL, [this, method = operation.method](wxCommandEvent&) {
            auto action = method;
            if (method == &TracePlayer::Play) {
                if (player_.IsPlaying()) action = &TracePlayer::Pause;
                else if (!player_.CanPlay(eval_.GetPlaybackRate())) return;
            }
            const auto message = (player_.*action)();
            last_tick_ = std::chrono::steady_clock::now();
            ApplySelection(false, true);
            Notify(message);
            SyncControls();
        }, id);
    }
    toolbar_->Realize();
    controls->Add(toolbar_, 0, wxALIGN_CENTER_VERTICAL);
    slider_ = new wxSlider(controls_panel_, wxID_ANY, 0, 0, 1);
    controls->Add(slider_, 1, wxALIGN_CENTER_VERTICAL | wxALL, FromDIP(2));
    speed_label_ = new wxStaticText(controls_panel_, wxID_ANY, "x1");
    controls->Add(speed_label_, 0, wxALIGN_CENTER_VERTICAL | wxALL, FromDIP(4));
    auto* border = new wxBoxSizer(wxVERTICAL);
    border->Add(controls, 1, wxEXPAND | wxALL, FromDIP(3));
    controls_panel_->SetSizer(border);
    layout->Add(controls_panel_, 0, wxEXPAND);
    grid_ = new TraceGrid(this, wxID_ANY);
    layout->Add(grid_, 1, wxEXPAND);
    SetSizer(layout);
    slider_->Bind(wxEVT_SLIDER, [this](wxCommandEvent&) {
        if (!updating_controls_ && !rows_.empty()) SelectIndex(static_cast<int>(rows_.size()) - 1 - slider_->GetValue(), true);
    });
    Bind(wxEVT_SHOW, [this](wxShowEvent& event) {
        // HWND破棄時にも届くため、先に破棄されるmanagerや子コントロールへ触れない。
        if (IsBeingDeleted()) { event.Skip(); return; }
        SetActive(IsPaneOpen(manager_.GetPane(this)));
        event.Skip();
    });
    Bind(wxEVT_TIMER, &EvalTracePanel::OnTimer, this, timer_.GetId());
    Bind(wxEVT_SYS_COLOUR_CHANGED, [this](wxSysColourChangedEvent& event) { UpdateAppearance(); event.Skip(); });
    Bind(wxEVT_DPI_CHANGED, [this](wxDPIChangedEvent& event) { UpdateAppearance(); event.Skip(); });
    UpdateAppearance();
    timer_.Start(20);
    SyncControls();
}

EvalTracePanel::~EvalTracePanel()
{
    timer_.Stop();
}

void EvalTracePanel::Initialize(std::shared_ptr<RunManager> manager, std::shared_ptr<EvalRunner> runner)
{
    // 表の宣言と通知購読を一度だけ結び、非表示時はcallbackの先頭で止める。
    env_spec_ = runner->GetBatchEnv()->GetSpec();
    // 通知に名前のためのresourceを保持せず、評価Envの実効設定から一度だけ読む。
    const auto env_config = runner->GetBatchEnv()->GetConfigData();
    env_name_ = env_config ? env_config->Get<std::string>("env.class_id", runner->GetBatchEnv()->GetName())
        : runner->GetBatchEnv()->GetName();
    agent_name_ = config_.Get("agent.class_id");
    builder_ = std::make_unique<StepTraceBuilder>(config_, env_spec_, manager->GetAgent(), 1);
    columns_ = builder_->GetColumns();
    percentile_windows_.resize(columns_.size());
    table_ = new Table(*this);
    grid_->SetTable(table_, true, wxGrid::wxGridSelectRows);
    ResetSorting();
    grid_->EnableEditing(false);
    // 行選択は標準の選択色にし、列見出しへ強調を波及させない。
    grid_->DisableOverlaySelection();
    grid_->SetRowLabelSize(0);
    // 手動で変更した列は、次に明示的に開くまで自動調整しない。
    grid_->Bind(wxEVT_GRID_COL_SIZE, [this](wxGridSizeEvent& event) {
        if (!sizing_columns_) {
            manual_columns_.at(static_cast<size_t>(event.GetRowOrCol())) = true;
            UpdateMinimumSize();
        }
        event.Skip();
    });
    ResetColumnWidths();
    UpdateAppearance();
    builder_->SetHandlers([weak = wxWeakRef<EvalTracePanel>(this)](const auto& row, const TrainEvent& event) {
        if (weak && weak->active_) weak->AddRow(row, event);
    }, [weak = wxWeakRef<EvalTracePanel>(this)](const auto& row, std::span<const size_t> columns) {
        if (weak && weak->active_) weak->CompleteRow(row, columns);
    });
    observer_ = manager->GetNotifier()->AttachScoped<FunctionTrainObserver>(runner,
        [weak = wxWeakRef<EvalTracePanel>(this)](const TrainEvent& event) {
            if (!weak) return;
            auto& self = *weak;
            self.SetActive(self.IsPaneOpen(self.manager_.GetPane(&self)));
            if (!self.active_) return;
            self.builder_->OnTrain(event);
        }, "Evaluation Trace");
    grid_->Bind(wxEVT_GRID_CELL_LEFT_CLICK, [this](wxGridEvent& event) {
        if (!updating_controls_) SelectIndex(event.GetRow());
        event.Skip();
    });
    grid_->Bind(wxEVT_GRID_SELECT_CELL, [this](wxGridEvent& event) {
        if (!updating_controls_) SelectIndex(event.GetRow());
        event.Skip();
    });
    grid_->Bind(wxEVT_GRID_LABEL_LEFT_CLICK, [this](wxGridEvent& event) {
        SortByColumn(event.GetCol());
    });
    grid_->Bind(wxEVT_GRID_LABEL_LEFT_DCLICK, [this](wxGridEvent& event) {
        // nativeのダブルクリックは、最初のクリックに続く2回目の切替とする。
        SortByColumn(event.GetCol());
    });
    grid_->Bind(wxEVT_GRID_CELL_LEFT_DCLICK, [this](wxGridEvent& event) {
        DumpRow(event.GetRow());
    });
    grid_->Bind(wxEVT_CHAR_HOOK, [this](wxKeyEvent& event) {
        if (event.GetKeyCode() == WXK_ESCAPE) {
            ReturnToLive();
            Notify("Trace: stop (live)");
        } else if ((event.GetKeyCode() == WXK_UP || event.GetKeyCode() == WXK_DOWN) && !rows_.empty()) {
            MoveSelection(event.GetKeyCode() == WXK_UP ? -1 : 1);
        } else event.Skip();
    });
    SetActive(IsPaneOpen(manager_.GetPane(this)));
}

bool EvalTracePanel::IsPaneOpen(const wxAuiPaneInfo& pane) const
{
    // Trace本体と追従paneで、最大化による一時非表示を同じように扱う。
    const auto& panes = manager_.GetAllPanes();
    bool maximized = false;
    for (size_t i = 0; i < panes.GetCount(); ++i) maximized |= panes.Item(i).IsMaximized();
    return gui::IsPaneOpen(pane, maximized);
}

bool EvalTracePanel::IsFollowerShown(const Binding& binding) const
{
    if (!binding.window) return false;
    const auto& pane = manager_.GetPane(binding.window.get());
    return IsPaneOpen(pane);
}

void EvalTracePanel::SetActive(bool active)
{
    if (active == active_) return;
    active_ = active;
    // 明示的に閉じたときだけ消去し、最大化による一時非表示では状態を保持する。
    ReturnToLive();
    ClearRows();
    if (builder_) builder_->Reset();
    last_tick_ = std::chrono::steady_clock::now();
}

void EvalTracePanel::ClearRows()
{
    const int count = static_cast<int>(rows_.size());
    rows_.clear();
    for (auto& window : percentile_windows_) window.clear();
    player_.SetRows({});
    if (table_ && count > 0) {
        wxGridTableMessage message(table_, wxGRIDTABLE_NOTIFY_ROWS_DELETED, 0, count);
        grid_->ProcessTableMessage(message);
    }
    ResetSorting();
    ResetColumnWidths();
    SyncControls();
}

void EvalTracePanel::ApplyPercentiles(Row& row)
{
    // 各列が確定した時点で、NaNも含め一度だけ処理する。
    for (size_t col = 0; col < columns_.size(); ++col) {
        if (columns_[col].display != "pct" || row.percentile_ready[col]) continue;
        if (!row.trace->value_ready[col]) continue;
        row.percentiles[col] = gui::AppendTracePercentile(percentile_windows_[col], row.trace->values[col], pct_window_);
        row.percentile_ready[col] = true;
    }
}

std::vector<float> EvalTracePanel::GetDisplayedValues(const Row& row) const
{
    std::vector<float> values(columns_.size(), std::numeric_limits<float>::quiet_NaN());
    for (size_t col = 0; col < columns_.size(); ++col) {
        if (row.trace->value_ready[col]) {
            values[col] = columns_[col].display == "pct" ? row.percentiles[col] : row.trace->values[col];
        }
    }
    return values;
}

void EvalTracePanel::ResetSorting()
{
    // 見出しの表示名によらず、生のstep値を表示する列に既定の印を付ける。
    constexpr const char* kRunnerTarget = "$runner";
    std::optional<size_t> step_column;
    for (size_t i = 0; i < columns_.size(); ++i) {
        if (columns_[i].target == kRunnerTarget && columns_[i].key == Runner::EXP_STEP
            && columns_[i].display == "value") {
            step_column = i;
            break;
        }
    }
    sorter_.Reset(step_column);
    if (table_) {
        if (const auto column = sorter_.GetColumn()) grid_->SetSortingColumn(static_cast<int>(*column), false);
        else grid_->UnsetSortingColumn();
        grid_->GetGridColLabelWindow()->Refresh(false);
    }
}

void EvalTracePanel::SyncPlayerRows()
{
    auto ids = sorter_.GetDisplayOrder();
    std::reverse(ids.begin(), ids.end());
    player_.SetRows(std::move(ids));
}

void EvalTracePanel::SortByColumn(int column)
{
    if (column < 0 || column >= static_cast<int>(columns_.size())) return;
    ANET_PROFILE_FUNC();
    // 値の並べ替えと表示の移動を分け、行IDと再生の進み具合は変えない。
    sorter_.ToggleColumn(static_cast<size_t>(column));
    std::unordered_map<int64_t, Row> retained;
    retained.reserve(rows_.size());
    for (auto& row : rows_) retained.emplace(row.trace->id, std::move(row));
    rows_.clear();
    for (const auto id : sorter_.GetDisplayOrder()) rows_.push_back(std::move(retained.at(id)));
    SyncPlayerRows();
    grid_->SetSortingColumn(column, sorter_.IsAscending());
    grid_->GetGridColLabelWindow()->Refresh(false);
    if (IsTraceMode()) ApplySelection(false, true);
    else {
        int x = 0, y = 0;
        grid_->GetViewStart(&x, &y);
        grid_->Scroll(x, 0);
    }
    SyncControls();
    grid_->Refresh(false);
}

void EvalTracePanel::AddRow(const std::shared_ptr<StepTrace>& trace, const TrainEvent& event)
{
    ANET_PROFILE_FUNC();
    Row row;
    row.trace = trace;
    // experience自体は通知中の借用なので、状態のTensor参照と通知の共有部品だけを残す。
    row.dump_data = {.state=event.experience.state, .action_info=event.action_info, .step_result=event.step_result,
        .counts=event.counts, .runner_name=event.runner->GetName(), .env_name=env_name_, .agent_name=agent_name_,
        .forced=event.runner->GetScalar(EvalRunner::kKeyForced, trace->lane).value()};
    row.percentiles.assign(columns_.size(), std::numeric_limits<float>::quiet_NaN());
    row.percentile_ready.assign(columns_.size(), false);
    ApplyPercentiles(row);
    for (const auto& follower : followers_) {
        if (IsFollowerShown(follower)) row.snapshots.emplace(follower.id, follower.capture(event));
    }
    // 挿入前の選択とscroll位置を保存し、Trace中の表示を先頭へ引っ張らない。
    const auto previous_selection = player_.GetSelected();
    int previous_index = 0;
    if (previous_selection) {
        for (size_t i = 0; i < rows_.size(); ++i)
            if (rows_[i].trace->id == *previous_selection) previous_index = static_cast<int>(i);
    }
    int scroll_x = 0, scroll_y = 0;
    grid_->GetViewStart(&scroll_x, &scroll_y);
    updating_controls_ = true;
    const auto position = sorter_.Upsert({.id=trace->id, .step=trace->step, .values=GetDisplayedValues(row)});
    rows_.insert(rows_.begin() + static_cast<ptrdiff_t>(position), std::move(row));
    wxGridTableMessage inserted(table_, wxGRIDTABLE_NOTIFY_ROWS_INSERTED, static_cast<int>(position), 1);
    grid_->ProcessTableMessage(inserted);
    if (const auto evicted = sorter_.Trim(static_cast<size_t>(max_rows_))) {
        const auto oldest = std::find_if(rows_.begin(), rows_.end(), [&](const auto& value) { return value.trace->id == *evicted; });
        const int index = static_cast<int>(std::distance(rows_.begin(), oldest));
        rows_.erase(oldest);
        wxGridTableMessage removed(table_, wxGRIDTABLE_NOTIFY_ROWS_DELETED, index, 1);
        grid_->ProcessTableMessage(removed);
    }
    updating_controls_ = false;
    const auto added = std::find_if(rows_.begin(), rows_.end(), [&](const auto& value) { return value.trace->id == trace->id; });
    if (added != rows_.end()) GrowColumnWidths(static_cast<int>(std::distance(rows_.begin(), added)));
    SyncPlayerRows();
    if (player_.GetSelected()) {
        const bool changed = player_.GetSelected() != previous_selection;
        ApplySelection(false, changed);
        if (!changed) {
            int unit_x = 0, unit_y = 0;
            grid_->GetScrollPixelsPerUnit(&unit_x, &unit_y);
            int current_index = 0;
            for (size_t i = 0; i < rows_.size(); ++i)
                if (rows_[i].trace->id == *player_.GetSelected()) current_index = static_cast<int>(i);
            const int delta = current_index - previous_index;
            grid_->Scroll(scroll_x, std::max(0, scroll_y + delta * (unit_y > 0 ? (grid_->GetDefaultRowSize() + unit_y - 1) / unit_y : 0)));
        }
    } else grid_->Scroll(scroll_x, 0);
    SyncControls();
    grid_->Refresh(false);
}

void EvalTracePanel::CompleteRow(const std::shared_ptr<StepTrace>& trace, std::span<const size_t> resolved_columns)
{
    const auto found = std::find_if(rows_.begin(), rows_.end(), [&](const auto& row) { return row.trace->id == trace->id; });
    if (found != rows_.end()) {
        ApplyPercentiles(*found);
        // 更新した一行だけを入れ直し、百分位の窓へ二重投入しない。
        const auto previous = static_cast<size_t>(std::distance(rows_.begin(), found));
        const auto position = sorter_.Upsert({.id=trace->id, .step=trace->step, .values=GetDisplayedValues(*found)});
        if (previous != position) {
            Row moved = std::move(*found);
            rows_.erase(found);
            rows_.insert(rows_.begin() + static_cast<ptrdiff_t>(position), std::move(moved));
            // 表の順が変わったときだけ、再生順と選択の表示を更新する。
            SyncPlayerRows();
            if (IsTraceMode()) ApplySelection(false, true);
            SyncControls();
        }
        if (!resolved_columns.empty()) GrowColumnWidths(static_cast<int>(position), resolved_columns);
    }
    else {
        // 表の上限から消えていても、遅延値は百分位の長い窓へ含める。
        for (const auto col : resolved_columns) {
            if (columns_[col].display == "pct") {
                gui::AppendTracePercentile(percentile_windows_[col], trace->values[col], pct_window_);
            }
        }
    }
    grid_->Refresh(false);
}

void EvalTracePanel::SelectIndex(int index, bool keep_playing)
{
    if (index < 0 || index >= static_cast<int>(rows_.size())) return;
    player_.Select(rows_[index].trace->id, keep_playing);
    last_tick_ = std::chrono::steady_clock::now();
    ApplySelection(false, true);
    SyncControls();
}

void EvalTracePanel::MoveSelection(int delta)
{
    ANET_CHECK_MSG(delta == -1 || delta == 1,
        "Invalid trace row movement. delta=" << delta << " expected=-1 or 1.");
    if (rows_.empty()) return;
    // 表のキーと転送されたキーを、同じ表示順の一行移動へ集める。
    int index = 0;
    if (const auto selected = player_.GetSelected()) {
        for (size_t i = 0; i < rows_.size(); ++i)
            if (rows_[i].trace->id == *selected) index = static_cast<int>(i);
    }
    SelectIndex(std::clamp(index + delta, 0, static_cast<int>(rows_.size()) - 1));
}

void EvalTracePanel::DumpRow(int index)
{
    if (index < 0 || index >= static_cast<int>(rows_.size())) return;
    ANET_PROFILE_FUNC();
    // 行の選択やログ通知の間も、今回ダンプする一件の寿命を固定する。
    const auto trace = rows_[index].trace;
    const auto data = rows_[index].dump_data;
    const auto values = GetDisplayedValues(rows_[index]);
    SelectIndex(index);
    anet::log::info() << gui::TraceDumpFormatter::FormatInfo(data, *trace, columns_, values, env_spec_.action_spec);
    anet::log::verbose() << gui::TraceDumpFormatter::FormatVerbose(data, *trace);
}

void EvalTracePanel::ApplySelection(bool force, bool ensure_visible)
{
    ANET_PROFILE_FUNC();
    const auto selected = player_.GetSelected();
    if (!selected) { ReturnToLive(); return; }
    const auto found = std::find_if(rows_.begin(), rows_.end(), [&](const auto& row) { return row.trace->id == *selected; });
    if (found == rows_.end()) return;
    eval_.PauseForTrace();
    const bool entering_trace = !displayed_id_;
    if (force || displayed_id_ != selected) {
        for (const auto& follower : followers_) {
            if (!IsFollowerShown(follower)) continue;
            const auto snapshot = found->snapshots.find(follower.id);
            follower.show(snapshot == found->snapshots.end() ? nullptr : snapshot->second);
        }
        RefreshCaptions(found->trace->step);
        if (entering_trace || force) {
            // Trace開始と追従paneの開閉だけ、dock全体を再構築する。
            ANET_PROFILE_SCOPE(layout);
            manager_.Update();
        }
    }
    displayed_id_ = selected;
    const int index = static_cast<int>(std::distance(rows_.begin(), found));
    updating_controls_ = true;
    grid_->ClearSelection();
    if (!columns_.empty()) {
        grid_->SetGridCursor(index, 0);
        grid_->SelectRow(index);
        if (ensure_visible) grid_->MakeCellVisible(index, 0);
    }
    updating_controls_ = false;
}

void EvalTracePanel::RefreshCaptions(step_t step)
{
    ANET_PROFILE_FUNC();
    // 行移動はcaptionと描画要求だけに留め、浮動frameのタイトルも直接揃える。
    for (const auto& follower : followers_) {
        if (!IsFollowerShown(follower)) continue;
        auto& pane = manager_.GetPane(follower.window.get());
        pane.Caption(follower.caption + wxString::Format(" [Trace: step %llu]", static_cast<unsigned long long>(step)));
        if (pane.frame) pane.frame->SetLabel(pane.caption);
    }
    frame_.Refresh(false);
}

void EvalTracePanel::ReturnToLive()
{
    player_.Stop();
    const bool was_trace = displayed_id_.has_value();
    displayed_id_.reset();
    if (was_trace) {
        ANET_PROFILE_SCOPE(captions);
        for (const auto& follower : followers_) {
            if (!follower.window) continue;
            follower.live();
            auto& pane = manager_.GetPane(follower.window.get());
            if (pane.IsOk()) {
                pane.Caption(follower.caption);
                if (pane.frame) pane.frame->SetLabel(pane.caption);
            }
        }
        ANET_PROFILE_SCOPE_NEXT(layout);
        manager_.Update();
    }
    if (table_) grid_->ClearSelection();
    SyncControls();
}

void EvalTracePanel::SyncControls()
{
    const bool has_rows = !rows_.empty();
    for (const int id : tool_ids_) toolbar_->EnableTool(id, has_rows);
    toolbar_->EnableTool(tool_ids_[0], player_.IsPlaying() || player_.CanPlay(eval_.GetPlaybackRate()));
    const bool pause = player_.IsPlaying();
    if (pause != showing_pause_) {
        showing_pause_ = pause;
        toolbar_->SetToolBitmap(tool_ids_[0], icons_[static_cast<size_t>(pause ? gui::PlaybackIcon::Pause : gui::PlaybackIcon::Play)]);
        toolbar_->SetToolShortHelp(tool_ids_[0], pause ? "Pause" : "Play");
        toolbar_->Refresh(false);
    }
    if (const bool trace = player_.GetSelected().has_value(); trace != showing_trace_) {
        showing_trace_ = trace;
        controls_panel_->Refresh(false);
    }
    slider_->Enable(has_rows);
    speed_label_->SetLabel(wxString::FromUTF8(player_.GetSpeedLabel()));
    updating_controls_ = true;
    slider_->SetRange(0, std::max(0, static_cast<int>(rows_.size()) - 1));
    int position = has_rows ? static_cast<int>(rows_.size()) - 1 : 0;
    if (const auto selected = player_.GetSelected()) {
        for (size_t i = 0; i < rows_.size(); ++i) if (rows_[i].trace->id == *selected) position = static_cast<int>(rows_.size() - 1 - i);
    }
    slider_->SetValue(position);
    updating_controls_ = false;
}

void EvalTracePanel::UpdateIcons()
{
    // 通常のtickでは作り直さず、テーマとDPIが変わった時に共有の記号を更新する。
    for (size_t i = 0; i < icons_.size(); ++i)
        icons_[i] = gui::MakePlaybackBitmap(static_cast<gui::PlaybackIcon>(i), wxSystemSettings::GetColour(wxSYS_COLOUR_BTNTEXT));
    const std::array<gui::PlaybackIcon, 6> symbols{showing_pause_ ? gui::PlaybackIcon::Pause : gui::PlaybackIcon::Play,
        gui::PlaybackIcon::Stop, gui::PlaybackIcon::StepBack, gui::PlaybackIcon::StepForward,
        gui::PlaybackIcon::Slower, gui::PlaybackIcon::Faster};
    for (size_t i = 0; i < symbols.size(); ++i) toolbar_->SetToolBitmap(tool_ids_[i], icons_[static_cast<size_t>(symbols[i])]);
    toolbar_->Refresh(false);
    controls_panel_->Refresh(false);
}

bool EvalTracePanel::SetFont(const wxFont& font)
{
    const bool changed = wxPanel::SetFont(font);
    // DPI変更時は子ウィンドウの自動拡大後に測り、列幅の二重拡大を避ける。
    if (changed && grid_) CallAfter([weak = wxWeakRef<EvalTracePanel>(this)] { if (weak) weak->UpdateAppearance(); });
    return changed;
}

void EvalTracePanel::UpdateMinimumSize()
{
    ANET_PROFILE_FUNC();
    // 操作バー全体と最初の行動列までを守り、残りの列は横スクロールへ残す。
    const auto controls = controls_panel_->GetSizer()->CalcMin();
    int prefix_width = 0;
    bool has_action = false;
    if (table_) {
        for (size_t col = 0; col < columns_.size(); ++col) {
            prefix_width += grid_->GetColSize(static_cast<int>(col));
            const auto& column = columns_[col];
            if (column.target == "$action_info" && (column.key == "action" || column.key.starts_with("action.["))) {
                has_action = true;
                break;
            }
        }
    }
    const int grid_width = has_action ? prefix_width + wxSystemSettings::GetMetric(wxSYS_VSCROLL_X, grid_) + FromDIP(8) : 0;
    const wxSize minimum(std::max(controls.x, grid_width), FromDIP(100));
    const bool window_changed = GetMinSize() != minimum;
    if (window_changed) SetMinSize(minimum);
    auto& pane = manager_.GetPane(this);
    if (!pane.IsOk()) return;
    const bool pane_changed = pane.min_size != minimum;
    if (pane_changed) pane.MinSize(minimum);
    // 最小幅が変わり、現に不足している時だけ再配置する。通常stepでは行わない。
    if ((window_changed || pane_changed) && IsShownOnScreen() && GetClientSize().x < minimum.x) manager_.Update();
}

void EvalTracePanel::ResetColumnWidths()
{
    if (!table_) return;
    ANET_PROFILE_FUNC();
    // 開き直しでは手動指定も忘れ、見出しだけの幅から始める。
    manual_columns_.assign(columns_.size(), false);
    wxClientDC dc(grid_);
    dc.SetFont(grid_->GetLabelFont());
    sizing_columns_ = true;
    grid_->BeginBatch();
    for (size_t col = 0; col < columns_.size(); ++col)
        grid_->SetColSize(static_cast<int>(col), dc.GetTextExtent(wxString::FromUTF8(columns_[col].label)).x + FromDIP(kColumnLabelPaddingDip));
    grid_->EndBatch();
    sizing_columns_ = false;
    UpdateMinimumSize();
}

void EvalTracePanel::GrowColumnWidths(int row, std::span<const size_t> columns)
{
    ANET_PROFILE_FUNC();
    // 通常は新規行C列、遅延通知では確定したK列だけを測る。保持行を走査しない。
    wxClientDC dc(grid_);
    dc.SetFont(grid_->GetDefaultCellFont());
    const size_t count = columns.empty() ? columns_.size() : columns.size();
    std::vector<std::pair<int, int>> changes;
    for (size_t index = 0; index < count; ++index) {
        const auto col = columns.empty() ? index : columns[index];
        if (manual_columns_[col]) continue;
        const int width = dc.GetTextExtent(table_->GetValue(row, static_cast<int>(col))).x + FromDIP(12);
        if (width > grid_->GetColSize(static_cast<int>(col))) changes.emplace_back(static_cast<int>(col), width);
    }
    if (changes.empty()) return;
    // 幅が変わる場合だけGridの寸法を更新し、通常のstepで再配置しない。
    sizing_columns_ = true;
    grid_->BeginBatch();
    for (const auto& [col, width] : changes) grid_->SetColSize(col, width);
    grid_->EndBatch();
    sizing_columns_ = false;
    UpdateMinimumSize();
}

void EvalTracePanel::UpdateAppearance()
{
    // 外観の変更時だけ見出しと保持行を測り直す。通常のstep/tickには持ち込まない。
    UpdateIcons();
    controls_panel_->SetBackgroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_BTNFACE));
    wxClientDC speed_dc(speed_label_);
    speed_dc.SetFont(GetFont());
    speed_label_->SetMinSize(wxSize(speed_dc.GetTextExtent("x0.125").x, -1));
    if (!table_) { UpdateMinimumSize(); return; }
    grid_->SetDefaultCellBackgroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOW));
    grid_->SetDefaultCellTextColour(wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOWTEXT));
    grid_->SetLabelBackgroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_BTNFACE));
    grid_->SetLabelTextColour(wxSystemSettings::GetColour(wxSYS_COLOUR_BTNTEXT));
    grid_->SetGridLineColour(wxSystemSettings::GetColour(wxSYS_COLOUR_3DSHADOW));
    grid_->SetSelectionBackground(wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT));
    grid_->SetSelectionForeground(wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHTTEXT));
    grid_->SetDefaultCellFont(GetFont());
    grid_->SetLabelFont(GetFont());
    wxClientDC dc(grid_);
    dc.SetFont(GetFont());
    grid_->SetDefaultRowSize(std::max(FromDIP(24), dc.GetCharHeight() + FromDIP(6)), true);
    grid_->SetColLabelSize(std::max(FromDIP(24), dc.GetCharHeight() + FromDIP(8)));
    sizing_columns_ = true;
    for (size_t col = 0; col < columns_.size(); ++col) {
        if (manual_columns_[col]) continue;
        const int width = dc.GetTextExtent(wxString::FromUTF8(columns_[col].label)).x + FromDIP(kColumnLabelPaddingDip);
        grid_->SetColSize(static_cast<int>(col), std::max(width, grid_->GetColSize(static_cast<int>(col))));
    }
    sizing_columns_ = false;
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        for (int col = 0; col < static_cast<int>(columns_.size()); ++col) grid_->RefreshAttr(row, col);
        GrowColumnWidths(row);
    }
    controls_panel_->GetSizer()->GetItem(size_t{0})->SetBorder(FromDIP(3));
    controls_panel_->Layout();
    UpdateMinimumSize();
    grid_->Refresh(false);
    grid_->GetGridColLabelWindow()->Refresh(false);
}

void EvalTracePanel::Notify(const std::string& message)
{
    if (!message.empty()) frame_.SetStatusText(wxString::FromUTF8(message), 0);
}

void EvalTracePanel::OnTimer(wxTimerEvent&)
{
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - last_tick_).count();
    last_tick_ = now;
    const auto& pane = manager_.GetPane(this);
    if (pane.IsOk()) SetActive(IsPaneOpen(pane));
    if (!active_ || !builder_) return;
    // 行追加が止まっていても、実行時のFPS・step数変更を再生ボタンへ反映する。
    toolbar_->EnableTool(tool_ids_[0], player_.IsPlaying() || player_.CanPlay(eval_.GetPlaybackRate()));
    bool visibility_changed = false;
    for (auto& follower : followers_) {
        const bool shown = IsFollowerShown(follower);
        visibility_changed |= shown != follower.was_shown;
        follower.was_shown = shown;
    }
    const auto before = player_.GetSelected();
    const bool was_playing = player_.IsPlaying();
    const auto message = player_.Tick(seconds, eval_.GetPlaybackRate());
    const bool selection_changed = before != player_.GetSelected() || (visibility_changed && player_.GetSelected());
    if (selection_changed) ApplySelection(visibility_changed, true);
    if (selection_changed || was_playing != player_.IsPlaying()) SyncControls();
    Notify(message);
}
