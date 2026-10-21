// EvalTracePanel.cpp
#include "EvalTracePanel.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include "anet/observers.hpp"
#include "anet/profile.hpp"

using namespace anet::rl;
using anet::rl::gui::TracePlayer;

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
        attr->SetBackgroundColour(trace.done || trace.truncated ? wxColour(250, 225, 207) : wxColour(220, 242, 222));
        attr->SetTextColour(*wxBLACK);
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
    auto* controls = new wxBoxSizer(wxHORIZONTAL);
    struct Operation { const char* text; const char* help; std::string (TracePlayer::*method)(); };
    const std::array<Operation, 7> operations{{
        {.text = "<", .help = "Step back", .method = &TracePlayer::StepBack}, {.text = "Play", .help = "Play", .method = &TracePlayer::Play},
        {.text = "Pause", .help = "Pause", .method = &TracePlayer::Pause}, {.text = "Stop", .help = "Stop and return to Live", .method = &TracePlayer::Stop},
        {.text = ">", .help = "Step forward", .method = &TracePlayer::StepForward}, {.text = "-", .help = "Slower", .method = &TracePlayer::Slower},
        {.text = "+", .help = "Faster", .method = &TracePlayer::Faster} }};
    for (const auto& operation : operations) {
        auto* button = new wxButton(this, wxID_ANY, operation.text, wxDefaultPosition, wxDefaultSize, wxBU_EXACTFIT);
        button->SetToolTip(operation.help);
        controls->Add(button, 0, wxALL, 2);
        buttons_.push_back(button);
        if (operation.method == &TracePlayer::Play) play_button_ = button;
        button->Bind(wxEVT_BUTTON, [this, method = operation.method](wxCommandEvent&) {
            if (method == &TracePlayer::Play && !player_.CanPlay(eval_.GetPlaybackRate())) return;
            const auto message = (player_.*method)();
            last_tick_ = std::chrono::steady_clock::now();
            ApplySelection(false, true);
            Notify(message);
            SyncControls();
        });
    }
    slider_ = new wxSlider(this, wxID_ANY, 0, 0, 1);
    controls->Add(slider_, 1, wxALIGN_CENTER_VERTICAL | wxALL, 2);
    speed_label_ = new wxStaticText(this, wxID_ANY, "x1");
    controls->Add(speed_label_, 0, wxALIGN_CENTER_VERTICAL | wxALL, 4);
    layout->Add(controls, 0, wxEXPAND);
    grid_ = new wxGrid(this, wxID_ANY);
    layout->Add(grid_, 1, wxEXPAND);
    SetSizer(layout);
    slider_->Bind(wxEVT_SLIDER, [this](wxCommandEvent&) {
        if (!updating_controls_ && !rows_.empty()) SelectIndex(static_cast<int>(rows_.size()) - 1 - slider_->GetValue(), true);
    });
    Bind(wxEVT_SHOW, [this](wxShowEvent& event) { SetActive(IsPaneOpen(manager_.GetPane(this))); event.Skip(); });
    Bind(wxEVT_TIMER, &EvalTracePanel::OnTimer, this, timer_.GetId());
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
    builder_ = std::make_unique<StepTraceBuilder>(config_, env_spec_, manager->GetAgent(), 1);
    columns_ = builder_->GetColumns();
    percentile_windows_.resize(columns_.size());
    table_ = new Table(*this);
    grid_->SetTable(table_, true, wxGrid::wxGridSelectRows);
    grid_->EnableEditing(false);
    grid_->SetRowLabelSize(0);
    grid_->SetDefaultColSize(95);
    grid_->SetDefaultRowSize(24);
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
    grid_->Bind(wxEVT_CHAR_HOOK, [this](wxKeyEvent& event) {
        if (event.GetKeyCode() == WXK_ESCAPE) {
            ReturnToLive();
            Notify("Trace: stop (live)");
        } else if ((event.GetKeyCode() == WXK_UP || event.GetKeyCode() == WXK_DOWN) && !rows_.empty()) {
            int index = 0;
            if (const auto selected = player_.GetSelected()) {
                for (size_t i = 0; i < rows_.size(); ++i) if (rows_[i].trace->id == *selected) index = static_cast<int>(i);
            }
            index += event.GetKeyCode() == WXK_UP ? -1 : 1;
            SelectIndex(std::clamp(index, 0, static_cast<int>(rows_.size()) - 1));
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

void EvalTracePanel::AddRow(const std::shared_ptr<StepTrace>& trace, const TrainEvent& event)
{
    ANET_PROFILE_FUNC();
    Row row;
    row.trace = trace;
    row.percentiles.assign(columns_.size(), std::numeric_limits<float>::quiet_NaN());
    row.percentile_ready.assign(columns_.size(), false);
    ApplyPercentiles(row);
    for (const auto& follower : followers_) {
        if (IsFollowerShown(follower)) row.snapshots.emplace(follower.id, follower.capture(event));
    }
    // 挿入前の選択とscroll位置を保存し、Trace中の表示を先頭へ引っ張らない。
    const auto previous_selection = player_.GetSelected();
    int scroll_x = 0, scroll_y = 0;
    grid_->GetViewStart(&scroll_x, &scroll_y);
    updating_controls_ = true;
    rows_.push_front(std::move(row));
    wxGridTableMessage inserted(table_, wxGRIDTABLE_NOTIFY_ROWS_INSERTED, 0, 1);
    grid_->ProcessTableMessage(inserted);
    if (rows_.size() > static_cast<size_t>(max_rows_)) {
        rows_.pop_back();
        wxGridTableMessage removed(table_, wxGRIDTABLE_NOTIFY_ROWS_DELETED, max_rows_, 1);
        grid_->ProcessTableMessage(removed);
    }
    updating_controls_ = false;
    std::vector<int64_t> ids;
    ids.reserve(rows_.size());
    for (auto it = rows_.rbegin(); it != rows_.rend(); ++it) ids.push_back(it->trace->id);
    player_.SetRows(std::move(ids));
    if (player_.GetSelected()) {
        const bool changed = player_.GetSelected() != previous_selection;
        ApplySelection(false, changed);
        if (!changed) {
            int unit_x = 0, unit_y = 0;
            grid_->GetScrollPixelsPerUnit(&unit_x, &unit_y);
            grid_->Scroll(scroll_x, scroll_y + (unit_y > 0 ? (grid_->GetDefaultRowSize() + unit_y - 1) / unit_y : 0));
        }
    } else grid_->Scroll(scroll_x, 0);
    SyncControls();
    grid_->Refresh(false);
}

void EvalTracePanel::CompleteRow(const std::shared_ptr<StepTrace>& trace, std::span<const size_t> resolved_columns)
{
    const auto found = std::find_if(rows_.begin(), rows_.end(), [&](const auto& row) { return row.trace->id == trace->id; });
    if (found != rows_.end()) ApplyPercentiles(*found);
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
    for (auto* button : buttons_) button->Enable(has_rows);
    play_button_->Enable(player_.CanPlay(eval_.GetPlaybackRate()));
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
    // 行追加が止まっていても、実行時のFPS変更を再生ボタンへ反映する。
    play_button_->Enable(player_.CanPlay(eval_.GetPlaybackRate()));
    bool visibility_changed = false;
    for (auto& follower : followers_) {
        const bool shown = IsFollowerShown(follower);
        visibility_changed |= shown != follower.was_shown;
        follower.was_shown = shown;
    }
    const auto before = player_.GetSelected();
    const auto message = player_.Tick(seconds, eval_.GetPlaybackRate());
    if (before != player_.GetSelected() || (visibility_changed && player_.GetSelected())) {
        ApplySelection(visibility_changed, true);
        SyncControls();
    }
    Notify(message);
}
