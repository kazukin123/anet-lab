// QValuePanel.hpp

#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <vector>
#include <optional>
#include <span>
#include <limits>
#include <wx/grid.h>
#include <wx/panel.h>
#include <wx/wx.h>
#include "anet/rl.hpp"
#include "anet/gui.hpp"
#include "anet/trainer.hpp"

class QValueHeatMapPanel : public anet::rl::gui::Panel {
public:
    explicit QValueHeatMapPanel(wxWindow* parent);
    ~QValueHeatMapPanel() override = default;

    // y_offset: 左側のGridのヘッダー高さに合わせるためのオフセット
    //void UpdateHeatMap(const wxImage& heatmap_image, int y_offset, int target_height);
    void UpdateHeatMap(const wxImage& heatmap_image, int y_offset, int target_height, float guide_line_pos = -1.0f);
    void SetScrollOffsetY(int scroll_y);
private:
    void OnPaint(wxPaintEvent& event);
    void OnEraseBackground(wxEraseEvent& event);
private:
    wxImage heatmap_image_;
    int y_offset_ = 0;
    int target_height_ = 0;
    int scroll_y_ = 0;
    float guide_line_pos_ = -1.0f; // 追加: 線のX座標割合
};

struct QValueData {
public:
    struct Stats {
        float mean;
        float std_dev;
        float max;
        float min;
    };
public:
    // 統計
    std::vector<Stats> stats;

    // 実際に選択された行動
    int64_t selected_action;

    // HeatMap (行優先 width×height の格子値)
    int width;
    int height;
    std::vector<float> vv;
    //wxImage heatmap_image;
};

// 同期呼び出しの指定。重みは従来どおりPanel側で一度作る。
struct QValueHeatMapRequest {
    bool histogram;
    bool advantage;
    bool log_scale;
    bool auto_range;
    int hist_bins;
    float hist_range_k;
    bool use_hist_smooth;
    float smooth_radius;
    std::span<const float> hist_weights; ///< 呼び出し元の重みを、この同期呼び出し中だけ借用する。
};

struct QValueHeatMapRangeState {
    float minimum = std::numeric_limits<float>::max();
    float maximum = std::numeric_limits<float>::lowest();
};

struct QValueHeatMapResult {
    int width;
    int height;
    std::vector<float> vv;
    float guide_line_ratio;
};

// 非空の格子値と有効なHistogram設定を受け、画面の部品を読まずに加工する。
QValueHeatMapResult ComputeQValueHeatMap(const QValueData& input,
    const QValueHeatMapRequest& request, QValueHeatMapRangeState& range);

class QValuePanel : public anet::rl::gui::Panel, public anet::rl::gui::TraceFollower {
public:
    explicit QValuePanel(wxWindow* parent, const anet::ConfigData& config_data);
    void Initialize(std::shared_ptr<anet::rl::RunManager> run_manager, std::shared_ptr<anet::rl::EvalRunner> runner);
    ~QValuePanel() noexcept override  = default;

    void SetActionHandler(std::function<void(int64_t)> action_handler);

    void ApplyData(QValueData data);
    void Update();

    int GetPreferredDockWidth() const;
    int GetPreferredDockHeight() const;
    wxSize GetFloatingMinSize() const;
    void SetDockSizeHandler(std::function<void()> handler);

    bool IsHistogram() const;
    bool IsAdvantage() const;
    bool IsLogScale() const;
public:	//TraceFollower
    std::shared_ptr<const anet::rl::gui::TraceSnapshot> CreateTraceSnapshot(const anet::rl::TrainEvent& event) override;
    void ShowTraceSnapshot(std::shared_ptr<const anet::rl::gui::TraceSnapshot> snapshot) override;
    void ReturnToLive() override;
private:
    void InitLayout();
    void SetupGrid();
    void UpdateGrid(const QValueData& data);
    void RenderHeatMap(const QValueHeatMapResult& data);
    std::optional<QValueData> CreateData(const anet::rl::TrainEvent& event);
    void ResetRange();
    void SyncHeatMapScroll(bool refresh_grid);
    void RefreshLayoutChildren();
    void BuildRows();
    int MeasureTextWidth(const wxString& text, const wxFont& font) const;
    int MeasureColumnWidth(const wxString& label, size_t chars) const;
    bool FitValueColumn(int col, const wxString& text);
    bool NeedsVerticalScroll() const;
    bool UpdateGridWidth();
    void ApplyGridWidth();
private:
    void OnSize(wxSizeEvent& event);
    void OnCloseWindow(wxCloseEvent& event);
    void OnCheck(wxCommandEvent& event); // 共通のチェックイベントハンドラ
    void OnResetRangeClick(wxCommandEvent& event);
    void OnGridActionClick(wxGridEvent& event);
    void OnGridScroll(wxScrollWinEvent& event);
    void OnGridMouseWheel(wxMouseEvent& event);
    void OnGridKeyUp(wxKeyEvent& event);
private:
    class Config;
    // 表の列。0 列目は行動名で、数値列は Mean・Max・Min・Std の順
    static constexpr int kActionCol = 0;
    static constexpr int kMeanCol = 1;
    static constexpr int kMaxCol = 2;
    static constexpr int kMinCol = 3;
    static constexpr int kStdCol = 4;
    static constexpr int kColumnCount = 5;
private:
    wxGrid* grid_;
    wxCheckBox* hist_check_;
    wxCheckBox* adv_check_;
    wxCheckBox* log_scale_check_;
    wxCheckBox* auto_range_check_;
    wxButton* reset_range_button_;
    wxPanel* header_panel_;
    QValueHeatMapPanel* heatmap_panel_;

    std::unique_ptr<Config> config_;
    std::vector<std::string> action_names_;
    std::shared_ptr<anet::rl::EvalRunner> runner_ = nullptr;
    std::function<void(int64_t)> action_handler_;
    std::shared_ptr<anet::rl::Notifier> notifier_ = nullptr;
    std::shared_ptr<anet::rl::TrainObserver> observer_ = nullptr;
    anet::rl::gui::UIDataStore<std::shared_ptr<const QValueData>> data_store_;
    std::shared_ptr<const QValueData> latest_data_;
    std::shared_ptr<const QValueData> trace_data_;
    std::optional<anet::rl::step_t> latest_step_;
    bool trace_mode_ = false;
    wxStaticText* no_data_label_ = nullptr;
    std::vector<float> hist_weights_;
    std::atomic<bool> update_pending_{ false };  ///< Update()のCallAfter予約中フラグ(多重予約防止)
    int last_selected_row_ = -1;                 ///< 前回ハイライト表示した行(差分更新用)
    std::function<void()> dock_size_handler_;    ///< 推奨の幅が変わったときの通知先
    std::array<size_t, kColumnCount> fitted_value_chars_{};  ///< 数値列の幅を合わせた文字数
    int grid_width_ = 0;                         ///< 固定した表の幅

    //  範囲記憶用
    QValueHeatMapRangeState heatmap_range_;

    // モード変更検知用
    bool last_is_adv_ = false;
    bool last_is_hist_ = false;
    bool last_is_log_ = false;
};
