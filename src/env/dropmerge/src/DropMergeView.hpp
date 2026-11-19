// DropMergeView.hpp
#pragma once

#include <optional>
#include <vector>
#include <memory>
#include <wx/wx.h>
#include <torch/torch.h>
#include "anet/rl.hpp"
#include "anet/gui.hpp"

// =============================================================
// DropMergeData
// =============================================================

struct DropMergeData {
    // Step info
    anet::rl::step_t step;

    // RL info
    anet::rl::SingleState state;
    std::int64_t action;
    float reward;

    // ENV info (AuxData)
    std::unordered_map<std::string, torch::Tensor> aux;

    // Display info
    std::optional<float> total_reward;

    static DropMergeData Create(anet::rl::TrainEvent event);
};

// =============================================================
// DropMergePanel
// =============================================================

class DropMergePanel : public anet::rl::gui::Panel {
public:
    explicit DropMergePanel(wxWindow* parent);

    void ApplyData(const DropMergeData& data);
protected:
    void OnPaint(wxPaintEvent& event);
    void OnMiddleClick(wxMouseEvent& event);
private:
    // 座標変換
    wxPoint WorldToScreen(float wx, float wy) const;
    int WorldScale(float size) const;

    // 描画メソッド
    void DrawBackground(wxDC& dc);
    void DrawContainer(wxDC& dc);
    void DrawGrid(wxDC& dc);
    void DrawFruits(wxDC& dc);
    void DrawDropper(wxDC& dc);
    void DrawSidePanel(wxDC& dc);

    // ヘルパ
    wxColour GetFruitColor(int rank) const;
private:
    bool has_snapshot_ = false;
    DropMergeData snapshot_;

    // Grid描画モード
    enum GridDrawMode {
        GRID_DRAW_MODE_DEFAULT = 0,
        GRID_DRAW_MODE_GRID_AND_FRUITS,
        GRID_DRAW_MODE_GRID_ONLY,
        GRID_DRAW_MODE_MAX
    } ;
    int grid_draw_mode_ = GRID_DRAW_MODE_DEFAULT;

    // レイアウト計算用
    wxRect main_view_rect_;
    wxRect side_view_rect_;

    // World範囲 (Configから取得して更新)
    float world_min_x_ = -1.5f;
    float world_max_x_ = 1.5f;
    float world_min_y_ = 0.5f;
    float world_max_y_ = 4.5f;
    float scale_factor_ = 1.0f;
};

// =============================================================
// DropMergeView
// =============================================================

class DropMergeView final : public anet::rl::gui::ViewBase<DropMergeData, DropMergePanel> {
public:
    explicit DropMergeView(wxWindow* parent);
    DropMergeData CreateViewData(const anet::rl::TrainEvent& event) const override;
};

// =============================================================
// DropMergeViewCreator
// =============================================================

class DropMergeViewCreator final : public anet::rl::gui::ViewCreator {
public:
    DropMergeViewCreator() = default;

    std::shared_ptr<anet::rl::gui::View> CreateView(
        wxWindow* parent, const anet::ConfigData& config_data, std::shared_ptr<anet::rl::Notifier> notifier) const
    {
        return std::make_shared<DropMergeView>(parent);
    }

    std::string GetTargetClassId() const override
    {
        return "DropMergeEnv";
    }
};
