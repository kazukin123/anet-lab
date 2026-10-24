// EvalTracePanel.hpp
#pragma once
#include <array>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <unordered_map>
#include <wx/aui/aui.h>
#include <wx/grid.h>
#include <wx/slider.h>
#include <wx/weakref.h>
#include "anet/metrics.hpp"
#include "anet/gui.hpp"
#include "anet/trainer.hpp"
#include "EvalPanel.hpp"

class EvalTracePanel final : public wxPanel {
public:
    EvalTracePanel(wxFrame& frame, wxAuiManager& manager, EvalPanel& eval, const anet::ConfigData& config);
    ~EvalTracePanel() override;
    void Initialize(std::shared_ptr<anet::rl::RunManager> manager, std::shared_ptr<anet::rl::EvalRunner> runner);
    void SetActive(bool active);
    void ReturnToLive();
    bool SetFont(const wxFont& font) override;

    template<class Pane>
    void RegisterFollower(Pane* pane)
    {
        // wxの追跡参照で寿命を確認し、破棄済みpaneを通知から参照しない。
        const wxWeakRef<Pane> weak(pane);
        Binding binding;
        binding.id = next_follower_id_++;
        binding.window = pane;
        binding.caption = manager_.GetPane(pane).caption;
        binding.capture = [weak](const anet::rl::TrainEvent& event) {
            return weak ? weak->CreateTraceSnapshot(event) : nullptr;
        };
        binding.show = [weak](std::shared_ptr<const anet::rl::gui::TraceSnapshot> snapshot) {
            if (weak) weak->ShowTraceSnapshot(std::move(snapshot));
        };
        binding.live = [weak] { if (weak) weak->ReturnToLive(); };
        followers_.push_back(std::move(binding));
        if (player_.GetSelected()) ApplySelection(true, false);
    }
private:
    class Table;
    using Snapshot = std::shared_ptr<const anet::rl::gui::TraceSnapshot>;
    struct Binding {
        uint64_t id = 0;
        wxWeakRef<wxWindow> window;
        wxString caption;
        bool was_shown = false;
        std::function<Snapshot(const anet::rl::TrainEvent&)> capture;
        std::function<void(Snapshot)> show;
        std::function<void()> live;
    };
    struct Row {
        std::shared_ptr<anet::rl::StepTrace> trace;
        std::vector<float> percentiles;
        std::vector<bool> percentile_ready;
        std::unordered_map<uint64_t, Snapshot> snapshots;
    };
    void AddRow(const std::shared_ptr<anet::rl::StepTrace>& trace, const anet::rl::TrainEvent& event);
    void CompleteRow(const std::shared_ptr<anet::rl::StepTrace>& trace, std::span<const size_t> resolved_columns);
    void ApplyPercentiles(Row& row);
    void ApplySelection(bool force, bool ensure_visible);
    void RefreshCaptions(anet::rl::step_t step);
    void SyncControls();
    void UpdateIcons();
    void UpdateAppearance();
    void UpdateMinimumSize();
    void ResetColumnWidths();
    void GrowColumnWidths(int row, std::span<const size_t> columns = {});
    void ClearRows();
    void SelectIndex(int index, bool keep_playing = false);
    void OnTimer(wxTimerEvent& event);
    bool IsFollowerShown(const Binding& binding) const;
    bool IsPaneOpen(const wxAuiPaneInfo& pane) const;
    void Notify(const std::string& message);

    wxFrame& frame_; ///< 親RunnerFrameが所有
    wxAuiManager& manager_; ///< 親RunnerFrameが所有
    EvalPanel& eval_; ///< 親RunnerFrameが所有
    anet::ConfigData config_;
    int max_rows_ = 1000;
    int pct_window_ = 10000;
    anet::rl::EnvSpec env_spec_;
    std::vector<anet::rl::StepTraceColumnSpec> columns_;
    std::vector<bool> manual_columns_;
    bool sizing_columns_ = false;
    std::unique_ptr<anet::rl::StepTraceBuilder> builder_;
    std::shared_ptr<anet::rl::TrainObserver> observer_;
    std::deque<Row> rows_;
    std::vector<std::deque<float>> percentile_windows_;
    std::vector<Binding> followers_;
    uint64_t next_follower_id_ = 0;
    anet::rl::gui::TracePlayer player_;
    std::optional<int64_t> displayed_id_;
    wxGrid* grid_ = nullptr;
    Table* table_ = nullptr;
    wxSlider* slider_ = nullptr;
    wxPanel* controls_panel_ = nullptr;
    wxAuiToolBar* toolbar_ = nullptr;
    std::array<int, 6> tool_ids_{};
    std::array<wxBitmapBundle, 7> icons_;
    bool showing_pause_ = false;
    bool showing_trace_ = false;
    wxStaticText* speed_label_ = nullptr;
    wxTimer timer_;
    std::chrono::steady_clock::time_point last_tick_;
    bool active_ = false;
    bool updating_controls_ = false;
};
