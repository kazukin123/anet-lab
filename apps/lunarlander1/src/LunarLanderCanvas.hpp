#pragma once

#include <optional>
#include <vector>
#include <wx/wx.h>
#include <torch/torch.h>
#include "anet/rl.hpp"
#include "UISnapshot.hpp"

struct TerrainPoint;

class LunarLanderCanvas : public wxPanel {
public:
    explicit LunarLanderCanvas(wxWindow* parent);

    // UIデータ設定（Frame 側から呼ばれる）
    void SetUISnapshot(const UISnapshot& snapshot);

protected:
    void OnPaint(wxPaintEvent& event);
    void OnMouseLeftClick(wxMouseEvent& event);
    void OnMouseRightClick(wxMouseEvent& event);

private:
    // world座標→画面座標変換
    wxPoint WorldToScreen(float wx, float wy, int width, int height) const;

    // 描画ヘルパ
    void DrawTerrain(wxDC& dc, int width, int height);
    void DrawPad(wxDC& dc, int width, int height);
    void DrawLander(wxDC& dc, int width, int height);
    void DrawThrust(wxDC& dc,
        int width,
        int height,
        float world_x,
        float world_y,
        float angle,
        int64_t action);
    void DrawWind(wxDC& dc, int width, int height);

private:
    // 最新のUISnapshot
    bool has_snapshot_ = false;
    UISnapshot snapshot_;

    // 地形キャッシュ（world座標系）
    std::vector<TerrainPoint> terrain_points_;
    bool has_terrain_ = false;

    // world座標系の範囲
    float world_min_x_ = -10.0f;
    float world_max_x_ = 10.0f;
    float world_min_y_ = 0.0f;
    float world_max_y_ = 15.0f;

    wxDECLARE_EVENT_TABLE();
};
