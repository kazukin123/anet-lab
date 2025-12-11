#include "LunarLanderCanvas.hpp"

#include <cmath>
#include <wx/dcbuffer.h>
#include "LunarLanderFrame.hpp"
#include "LunarLanderApp.hpp"

#define _USE_MATH_DEFINES // for C++
#include <cmath>

wxBEGIN_EVENT_TABLE(LunarLanderCanvas, wxPanel)
EVT_PAINT(LunarLanderCanvas::OnPaint)
EVT_LEFT_DOWN(LunarLanderCanvas::OnMouseClick)
wxEND_EVENT_TABLE()

LunarLanderCanvas::LunarLanderCanvas(wxWindow* parent)
    : wxPanel(parent, wxID_ANY, wxDefaultPosition, wxSize(800, 400))
{
    SetBackgroundStyle(wxBG_STYLE_PAINT);
}

void LunarLanderCanvas::OnMouseClick(wxMouseEvent& event)
{
    wxGetApp().ToggleTraining();
}

void LunarLanderCanvas::SetUISnapshot(const UISnapshot& snapshot)
{
    has_snapshot_ = true;
    snapshot_ = snapshot;

    // world bounds
    world_min_x_ = snapshot_.world_min_x;
    world_max_x_ = snapshot_.world_max_x;
    world_min_y_ = snapshot_.world_min_y;
    world_max_y_ = snapshot_.world_max_y;

    //world_min_x_ = -10.5f;
    //world_max_x_ = 10.5f;
    //world_min_y_ = -0.5f;
    //world_max_y_ = 10.0f;

    // 地形キャッシュ更新
    if (snapshot_.terrain.has_value()) {
        terrain_points_.clear();
        terrain_points_.reserve(snapshot_.terrain->points.size());
        for (const auto& p : snapshot_.terrain->points) {
            terrain_points_.push_back(p);
        }
        has_terrain_ = !terrain_points_.empty();
    }

    Refresh(false);
}

wxPoint LunarLanderCanvas::WorldToScreen(float wx,
    float wy,
    int width,
    int height) const
{
    const int margin = 40;

    const float world_w = std::max(0.1f, world_max_x_ - world_min_x_);
    const float world_h = std::max(0.1f, world_max_y_ - world_min_y_);

    const float sx = static_cast<float>(width - 2 * margin) / world_w;
    const float sy = static_cast<float>(height - 2 * margin) / world_h;
    const float scale = std::min(sx, sy);

    // 正しい座標変換
    const float canvas_x = margin + (wx - world_min_x_) * scale;
    const float canvas_y = height - (margin + (wy - world_min_y_) * scale);

    return wxPoint(static_cast<int>(canvas_x),
        static_cast<int>(canvas_y));
}

void LunarLanderCanvas::DrawTerrain(wxDC& dc, int width, int height)
{
    if (!has_terrain_) {
        return;
    }

    if (terrain_points_.size() < 2) {
        return;
    }

    std::vector<wxPoint> pts;
    pts.reserve(terrain_points_.size());
    for (const auto& p : terrain_points_) {
        pts.push_back(WorldToScreen(p.x, p.y, width, height));
    }

    dc.SetPen(wxPen(*wxBLACK, 3));
    dc.DrawLines(static_cast<int>(pts.size()), pts.data());
}

void LunarLanderCanvas::DrawPad(wxDC& dc, int width, int height)
{
    const auto& pad = snapshot_.pad;

    const wxPoint p1 = WorldToScreen(pad.x1, pad.y, width, height);
    const wxPoint p2 = WorldToScreen(pad.x2, pad.y, width, height);

    dc.SetPen(wxPen(*wxGREEN, 3));
    dc.DrawLine(p1, p2);
}

void LunarLanderCanvas::DrawLander(wxDC& dc, int width, int height)
{
    const auto& exp = snapshot_.exp;

    if (!exp.state.obs.defined()) {
        return;
    }

    // obs = [x, y, vx, vy, angle, angular_vel, left_contact, right_contact]
    const auto obs = exp.state.obs;
    const float x_norm = obs[0].item<float>();
    const float y_norm = obs[1].item<float>();
    const float angle = obs[4].item<float>();

    // Canvas 用 world_y に変換
    const float world_x = x_norm;
    const float world_y = y_norm;

    const wxPoint center = WorldToScreen(world_x, world_y, width, height);

    // ランダー本体を円で描画
    const int base_radius = 30;
    dc.SetBrush(*wxBLUE_BRUSH);
    dc.SetPen(wxPen(*wxBLACK, 1));
    dc.DrawCircle(center, base_radius);

    // 機体の向きを線で示す
    const float dir_len = static_cast<float>(base_radius) * 1.5f;
    const float dx = std::sin(angle) * dir_len;
    const float dy = std::cos(angle) * dir_len;

    dc.SetPen(wxPen(*wxBLACK, 3));
    const wxPoint nose(center.x + static_cast<int>(dx),
        center.y - static_cast<int>(dy));
    dc.DrawLine(center, nose);

    // thrust 描画
    int64_t action = 0;
    if (exp.action.defined()) {
        action = exp.action.item<int64_t>();
    }
    DrawThrust(dc, width, height, world_x, world_y, angle, action);
}

void LunarLanderCanvas::DrawThrust(wxDC& dc,
    int width,
    int height,
    float world_x,
    float world_y,
    float angle,
    int64_t action)
{
    if (action == 0) return;

    const float R = 0.25f;      // 機体半径

    // ====== ノズル位置 ======
    float ox = 0.0f;
    float oy = -R;

    if (action == 2) { // left
        ox = -R * 0.55f;        // 機体と自然に見える比率（0.55R）
        oy = -R * 0.20f;
    }
    else if (action == 3) { // right
        ox = +R * 0.55f;
        oy = -R * 0.20f;
    }

    // ====== ローカル→ワールド ======
    float ca = std::cos(angle);
    float sa = std::sin(angle);

    float nx = world_x + ox * ca - oy * sa;
    float ny = world_y + ox * sa + oy * ca;

    wxPoint p0 = WorldToScreen(nx, ny, width, height);

    // ====== thrust 方向（本体の真下） ======
    // nose = (sin(a), cos(a)) → 上向き
    // down = (-sin(a), -cos(a))
    float dir_x = -std::sin(angle);
    float dir_y = -std::cos(angle);

    const float side_angle = 0.45f;

    // side engines
    if (action == 2) {
        float ca2 = std::cos(-side_angle);
        float sa2 = std::sin(-side_angle);
        float dx = dir_x * ca2 - dir_y * sa2;
        float dy = dir_x * sa2 + dir_y * ca2;
        dir_x = dx; dir_y = dy;
    }
    else if (action == 3) {
        float ca2 = std::cos(+side_angle);
        float sa2 = std::sin(+side_angle);
        float dx = dir_x * ca2 - dir_y * sa2;
        float dy = dir_x * sa2 + dir_y * ca2;
        dir_x = dx; dir_y = dy;
    }

    // ====== flame 長さ ======
    const float flame_len = 0.4f;

    float ex = nx + dir_x * flame_len;
    float ey = ny + dir_y * flame_len;

    wxPoint p1 = WorldToScreen(ex, ey, width, height);

    // ====== 矢印本体 ======
    dc.SetPen(wxPen(wxColour(250, 40, 40), 3));
    dc.DrawLine(p0, p1);

    // ====== arrowhead（細く、見やすく） ======
    // perpendicular ベクトル
    float px = -dir_y;
    float py = dir_x;

    const float head_len = 0.18f;      // 矢尻“長さ”
    const float head_width = 0.12f;    // 矢尻“横幅”（狭め）

    wxPoint h1 = WorldToScreen(
        ex - dir_x * head_len + px * head_width,
        ey - dir_y * head_len + py * head_width,
        width, height);

    wxPoint h2 = WorldToScreen(
        ex - dir_x * head_len - px * head_width,
        ey - dir_y * head_len - py * head_width,
        width, height);

    wxPoint headPts[3] = { p1, h1, h2 };

    dc.SetBrush(wxBrush(wxColour(250, 40, 40)));
    dc.DrawPolygon(3, headPts);
}


void LunarLanderCanvas::DrawWind(wxDC& dc, int width, int height)
{
    const float wind_x = snapshot_.wind_x;

    const int margin = 10;
    const int y = height - margin - 10;
    const int center_x = width - 120;

    dc.SetTextForeground(*wxBLACK);
    dc.DrawText(wxString::Format("Wind: %.2f", wind_x),
        center_x - 40,
        y - 20);

    dc.SetPen(wxPen(*wxBLACK, 2));

    const int base_len = 80;
    const float k = std::clamp(std::abs(wind_x) / 20.0f, 0.1f, 1.0f);
    const int len = static_cast<int>(base_len * k);

    int x1 = center_x - len / 2;
    int x2 = center_x + len / 2;
    if (wind_x < 0.0f) {
        std::swap(x1, x2);
    }

    dc.DrawLine(x1, y, x2, y);

    const int arrow = 6;
    if (wind_x > 0.0f) {
        dc.DrawLine(x2, y, x2 - arrow, y - arrow);
        dc.DrawLine(x2, y, x2 - arrow, y + arrow);
    }
    else if (wind_x < 0.0f) {
        dc.DrawLine(x2, y, x2 + arrow, y - arrow);
        dc.DrawLine(x2, y, x2 + arrow, y + arrow);
    }
}

void LunarLanderCanvas::OnPaint(wxPaintEvent& event)
{
    wxAutoBufferedPaintDC dc(this);
    dc.Clear();

    if (!has_snapshot_) {
        dc.DrawText("No snapshot", 10, 50);
        return;
    }

    const wxSize size = GetClientSize();
    const int width = size.GetWidth();
    const int height = size.GetHeight();

    dc.SetTextForeground(*wxBLACK);

    // Step
    dc.DrawText(wxString::Format("Step: %llu",
        static_cast<unsigned long long>(snapshot_.step)),
        10,
        10);

    // Reward
    dc.DrawText(wxString::Format("Reward: %.2f", snapshot_.exp.reward), 10, 30);

    // Action
    if (snapshot_.exp.action.defined()) {
        int64_t action = snapshot_.exp.action.item<int64_t>();
        dc.DrawText(wxString::Format("Action: %llu", action), 10, 50);
    }

    // pos
    if (snapshot_.exp.state.obs.defined()) {
        const float x = snapshot_.exp.state.obs[0].item<float>();
        const float y = snapshot_.exp.state.obs[1].item<float>();
        const float theta_deg = 180.0f / M_PI * snapshot_.exp.state.obs[4].item<float>();
        const float contact_left = snapshot_.exp.state.obs[6].item<float>();
        const float contact_right = snapshot_.exp.state.obs[7].item<float>();

        dc.DrawText(wxString::Format("(X Y θ°): (%.2f %.2f %.1f)", x, y, theta_deg), 10, 70);
        dc.DrawText(wxString::Format("Contact: (%.2f %.2f)", contact_left, contact_right), 10, 90);
    }

    DrawTerrain(dc, width, height);
    DrawPad(dc, width, height);
    DrawLander(dc, width, height);
    DrawWind(dc, width, height);
}
