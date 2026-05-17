// QValuePanel.hpp

#pragma once

#include <functional>
#include <vector>
#include <optional>
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

    // HeatMap
    int width;
    int height;
    std::vector<float> xv;
    std::vector<float> yv;
    std::vector<float> vv;
    //wxImage heatmap_image;
};

class QValuePanel : public anet::rl::gui::Panel {
public:
    explicit QValuePanel(wxWindow* parent, const anet::ConfigData& config_data);
    void Initialize(std::shared_ptr<anet::rl::RunManager> run_manager, std::shared_ptr<anet::rl::EvalRunner> runner);
    ~QValuePanel() noexcept override  = default;

    void SetActionHandler(std::function<void(int64_t)> action_handler);

    void ApplyData(const QValueData& data);
    void Update();

    bool IsHistogram() const;
    bool IsAdvantage() const;
    bool IsLogScale() const;
private:
    void InitLayout();
    void SetupGrid();
    std::optional<QValueData> CreateData(const anet::rl::TrainEvent& event);
    void ResetRange();
    void SyncHeatMapScroll(bool refresh_grid);
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
private:
    wxGrid* grid_;
    wxCheckBox* hist_check_;
    wxCheckBox* adv_check_;
    wxCheckBox* log_scale_check_;
    wxCheckBox* auto_range_check_;
    wxButton* reset_range_button_;
    QValueHeatMapPanel* heatmap_panel_;

    std::unique_ptr<Config> config_;
    std::vector<std::string> action_names_;
    std::shared_ptr<anet::rl::EvalRunner> runner_ = nullptr;
    std::function<void(int64_t)> action_handler_;
    std::shared_ptr<anet::rl::Notifier> notifier_ = nullptr;
    std::shared_ptr<anet::rl::TrainObserver> observer_ = nullptr;
    anet::rl::gui::UIDataStore<QValueData> data_store_;
    std::vector<float> hist_weights_;

    //  範囲記憶用
    float accumulated_min_ = std::numeric_limits<float>::max();
    float accumulated_max_ = std::numeric_limits<float>::lowest();

    // モード変更検知用
    bool last_is_adv_ = false;
    bool last_is_hist_ = false;
    bool last_is_log_ = false;
};
