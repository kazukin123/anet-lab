#include "LunarLanderCanvas.hpp"

#include <cmath>
#include <wx/dcbuffer.h>
#include "LunarLanderFrame.hpp"
#include "LunarLanderApp.hpp"

#define _USE_MATH_DEFINES // for C++
#include <cmath>

wxBEGIN_EVENT_TABLE(LunarLanderCanvas, wxPanel)
EVT_PAINT(LunarLanderCanvas::OnPaint)
EVT_LEFT_DOWN(LunarLanderCanvas::OnMouseLeftClick)
EVT_RIGHT_DOWN(LunarLanderCanvas::OnMouseRightClick)
wxEND_EVENT_TABLE()

using anet::ToBool;
using anet::ToFloat;


LunarLanderCanvas::LunarLanderCanvas(wxWindow* parent)
    : wxPanel(parent, wxID_ANY, wxDefaultPosition)//, wxSize(400, 400))
{
    SetBackgroundStyle(wxBG_STYLE_PAINT);
}

void LunarLanderCanvas::OnMouseLeftClick(wxMouseEvent& event)
{
    wxMouseEvent evt = event;
    GetParent()->GetEventHandler()->ProcessEvent(evt);
}

void LunarLanderCanvas::OnMouseRightClick(wxMouseEvent& event)
{
    wxMouseEvent evt = event;
    GetParent()->GetEventHandler()->ProcessEvent(evt);
}

void LunarLanderCanvas::SetUISnapshot(const UISnapshot& snapshot)
{
    //ANET_LOG_DEBUG("aux=" << anet::ToString(snapshot_.aux));

    has_snapshot_ = true;
    snapshot_ = snapshot;

    auto it = snapshot_.aux.find("world");
    if (it != snapshot_.aux.end()) {
        const auto w = it->second;
        world_min_x_ = w[0].item<float>();
        world_max_x_ = w[1].item<float>();
        world_min_y_ = w[2].item<float>();
        world_max_y_ = w[3].item<float>();

        auto it2 = snapshot_.aux.find("lander");
        if (it2 != snapshot_.aux.end()) {
            const auto v = it2->second;
            const float lander_y = v[1].item<float>();

            world_max_y_ = std::max(world_max_y_, lander_y + kLanderRadius);
        }
    }

    Refresh(false);
}

wxPoint LunarLanderCanvas::WorldToScreen(
    float wx, float wy, int width, int height) const
{
    const int margin = 40;

    const float world_w = std::max(0.1f, world_max_x_ - world_min_x_);
    const float world_h = std::max(0.1f, world_max_y_ - world_min_y_);

    const float sx = static_cast<float>(width - 2 * margin) / world_w;
    const float sy = static_cast<float>(height - 2 * margin) / world_h;
    const float scale = std::min(sx, sy);

    // world 中心
    const float world_cx = 0.5f * (world_min_x_ + world_max_x_);
    const float world_cy = 0.5f * (world_min_y_ + world_max_y_);

    // canvas 中心
    const float canvas_cx = 0.5f * width;
    const float canvas_cy = 0.5f * height;

    // 中心基準変換
    const float x = canvas_cx + (wx - world_cx) * scale;
    const float y = canvas_cy - (wy - world_cy) * scale;

    return wxPoint(static_cast<int>(x), static_cast<int>(y));
}

int LunarLanderCanvas::WorldToScreen(float size, int width, int height) const
{
    const int margin = 40;
    const float world_w = std::max(0.1f, world_max_x_ - world_min_x_);
    const float world_h = std::max(0.1f, world_max_y_ - world_min_y_);
    const float scale = std::min(
        (width - 2 * margin) / world_w,
        (height - 2 * margin) / world_h);

    return static_cast<int>(std::round(size * scale));
}

void LunarLanderCanvas::DrawWorldLine(wxDC& dc, int width, int height)
{
    dc.SetPen(wxPen(*wxLIGHT_GREY, 2));

    // 下
    const wxPoint p1 = WorldToScreen(world_min_x_, world_min_y_, width, height);
    const wxPoint p2 = WorldToScreen(world_max_x_, world_min_y_, width, height);
    dc.DrawLine(p1, p2);

    // 左
    const wxPoint p3 = WorldToScreen(world_min_x_, world_min_y_, width, height);
    const wxPoint p4 = WorldToScreen(world_min_x_, world_max_y_, width, height);
    dc.DrawLine(p3, p4);

    // 右
    const wxPoint p5 = WorldToScreen(world_max_x_, world_min_y_, width, height);
    const wxPoint p6 = WorldToScreen(world_max_x_, world_max_y_, width, height);
    dc.DrawLine(p5, p6);
}

void LunarLanderCanvas::DrawTerrain(wxDC& dc, int width, int height)
{
    auto it = snapshot_.aux.find("terrain");
    if (it == snapshot_.aux.end()) {
        return;
    }

    const auto& t = it->second;
    if (t.size(0) < 2) {
        return;
    }

    std::vector<wxPoint> pts;
    pts.reserve(t.size(0));

    for (int64_t i = 0; i < t.size(0); ++i) {
        pts.emplace_back(WorldToScreen(
                t[i][0].item<float>(), t[i][1].item<float>(),width, height));
    }

    dc.SetPen(wxPen(*wxBLACK, 3));
    dc.DrawLines(static_cast<int>(pts.size()), pts.data());
    dc.DrawLines(static_cast<int>(pts.size()), pts.data());
}

void LunarLanderCanvas::DrawPad(wxDC& dc, int width, int height)
{
    auto it = snapshot_.aux.find("pad");
    if (it == snapshot_.aux.end()) {
        return;
    }

    const auto p = it->second;
    const wxPoint p1 = WorldToScreen(p[0].item<float>(), p[2].item<float>(), width, height);
    const wxPoint p2 = WorldToScreen(p[1].item<float>(), p[2].item<float>(), width, height);

    dc.SetPen(wxPen(*wxGREEN, 3));
    dc.DrawLine(p1, p2);
}

void LunarLanderCanvas::DrawLander(wxDC& dc, int width, int height)
{
    auto it = snapshot_.aux.find("lander");
    if (it == snapshot_.aux.end()) {
        return;
    }

    // 生の位置・角度
    const auto v = it->second;
    const float x = v[0].item<float>();
    const float y = v[1].item<float>();
    const float angle = v[4].item<float>();

    // 描画用の位置・大きさ
    const wxPoint c = WorldToScreen(x, y, width, height);
    const int r = WorldToScreen(kLanderRadius, width, height);

    // 本体接触情報
    bool body_contact = false;
    auto it2 = snapshot_.aux.find("contacts");
    if (it2 != snapshot_.aux.end()) {
        body_contact = ToBool(it2->second[2]);
    }

    // 本体描画
    if (body_contact) dc.SetPen(wxPen(*wxRED, 2));
    else dc.SetPen(wxPen(*wxBLACK, 1));
    dc.SetBrush(*wxCYAN_BRUSH);
    dc.DrawCircle(c, r);

    // 本体角度線描画
    const float draw_angle = angle + static_cast<float>(M_PI) / 2.0f;
    const wxPoint nose(
        c.x + static_cast<int>(std::cos(draw_angle) * r * 1.5f),
        c.y - static_cast<int>(std::sin(draw_angle) * r * 1.5f));
    dc.SetPen(wxPen(*wxBLACK, 2));
    dc.DrawLine(c, nose);
}

void LunarLanderCanvas::DrawLegs(wxDC& dc, int width, int height)
{
    auto it = snapshot_.aux.find("legs");
    if (it == snapshot_.aux.end()) {
        return;
    }

    auto it2 = snapshot_.aux.find("contacts");
    bool left_contact = false;
    bool right_contact = false;
    if (it2 != snapshot_.aux.end()) {
        left_contact = ToBool(it2->second[0]);
        right_contact = ToBool(it2->second[1]);
    }

    //ANET_LOG_DEBUG("left_contact=" << left_contact);
    //ANET_LOG_DEBUG("right_contact=" << right_contact);

    const auto legs = it->second;

    {
        if (left_contact)
            dc.SetPen(wxPen(*wxGREEN, 2));
        else
            dc.SetPen(wxPen(*wxBLUE, 2));
        dc.DrawLine(
            WorldToScreen(ToFloat(legs[0]), ToFloat(legs[1]), width, height),
            WorldToScreen(ToFloat(legs[2]), ToFloat(legs[3]), width, height));
    }

    {
        if (right_contact)
            dc.SetPen(wxPen(*wxGREEN, 2));
        else
            dc.SetPen(wxPen(*wxBLUE, 2));
        dc.DrawLine(
            WorldToScreen(legs[4].item<float>(), legs[5].item<float>(), width, height),
            WorldToScreen(legs[6].item<float>(), legs[7].item<float>(), width, height));
    }
}

void LunarLanderCanvas::DrawThrust(wxDC& dc, int width, int height)
{
    const int64_t action = snapshot_.action;
    if (action == 0) {
        return;
    }

    auto it = snapshot_.aux.find("lander");
    if (it == snapshot_.aux.end()) {
        return;
    }

    const auto lander = it->second;
    const float x = lander[0].item<float>();
    const float y = lander[1].item<float>();
    const float angle = lander[4].item<float>();

    constexpr float kLanderRadius = 0.25f;

    float local_off_x = 0.0f;
    float local_off_y = 0.0f;
    float local_dir_x = 0.0f;
    float local_dir_y = 0.0f;
    float arrow_len = 0.0f;

    if (action == 1) {
        // --- Main Engine ---
        local_off_y = -kLanderRadius;
        local_dir_y = -1.0f;
        arrow_len = kMainEngineForce;
    } else if (action == 2 || action == 3) {
        // --- Side Engine ---
        const float h = kLanderRadius * 0.80f;
        const float r = kLanderRadius;
        const float dx = std::sqrt(std::max(0.0f, r * r - h * h));

        local_off_y = h;

        if (action == 2) {
            // Left Engine
            local_off_x = -dx;
            local_dir_x = -1.0f;
        } else {
            // Right Engine
            local_off_x = +dx;
            local_dir_x = +1.0f;
        }

        arrow_len = kSideEngineForce * 3;
    }
    else {
        return;
    }

    // --- 回転変換 ---
    const float ca = std::cos(angle);
    const float sa = std::sin(angle);

    const float off_x =
        local_off_x * ca - local_off_y * sa;
    const float off_y =
        local_off_x * sa + local_off_y * ca;

    const float dir_x =
        local_dir_x * ca - local_dir_y * sa;
    const float dir_y =
        local_dir_x * sa + local_dir_y * ca;

    const float start_x = x + off_x;
    const float start_y = y + off_y;

    const wxPoint p0 =
        WorldToScreen(start_x, start_y, width, height);

    const wxPoint p1(
        p0.x + static_cast<int>(std::round(dir_x * arrow_len)),
        p0.y - static_cast<int>(std::round(dir_y * arrow_len)));

    dc.SetPen(wxPen(wxColour(220, 0, 0), 2));
    dc.DrawLine(p0, p1);

    // --- Arrow head ---
    constexpr float head_len = 8.0f;
    constexpr float head_ang = 0.5f;

    const float back_x = -dir_x;
    const float back_y = -dir_y;

    const float hx1 =
        back_x * std::cos(head_ang) - back_y * std::sin(head_ang);
    const float hy1 =
        back_x * std::sin(head_ang) + back_y * std::cos(head_ang);

    const float hx2 =
        back_x * std::cos(-head_ang) - back_y * std::sin(-head_ang);
    const float hy2 =
        back_x * std::sin(-head_ang) + back_y * std::cos(-head_ang);

    dc.DrawLine(
        p1,
        wxPoint(
            p1.x + static_cast<int>(hx1 * head_len),
            p1.y - static_cast<int>(hy1 * head_len)));

    dc.DrawLine(
        p1,
        wxPoint(
            p1.x + static_cast<int>(hx2 * head_len),
            p1.y - static_cast<int>(hy2 * head_len)));
}

void LunarLanderCanvas::DrawWind(wxDC& dc, int width, int height)
{
    auto it = snapshot_.aux.find("forces");
    if (it == snapshot_.aux.end()) {
        return;
    }

    const float wind = it->second[0].item<float>();
    dc.DrawText(wxString::Format("Wind: %.2f", wind), 10, height - 20);
}

void LunarLanderCanvas::DrawRL(wxDC& dc)
{
    dc.SetTextForeground(*wxBLACK);

    // Step
    dc.DrawText(wxString::Format("Step: %llu",
        static_cast<unsigned long long>(snapshot_.step)),
        10,
        10);

    // Reward
    auto raw_reward = 0.0f;
    auto it = snapshot_.aux.find("rewards");
    if (it != snapshot_.aux.end()) raw_reward = ToFloat(it->second[1]);
    dc.DrawText(wxString::Format("Reward: %.2f (%.2f)", snapshot_.reward, raw_reward), 10, 30);

    // Action
    dc.DrawText(wxString::Format("Action: %llu", snapshot_.action), 10, 50);

    // pos
    if (snapshot_.state.obs.defined()) {
        const float x = snapshot_.state.obs[0].item<float>();
        const float y = snapshot_.state.obs[1].item<float>();
        const float theta = snapshot_.state.obs[4].item<float>();
        const float theta_deg = 180.0f / M_PI * theta;
        const float contact_left = snapshot_.state.obs[6].item<float>();
        const float contact_right = snapshot_.state.obs[7].item<float>();
        const float x_dot = snapshot_.state.obs[2].item<float>();
        const float y_dot = snapshot_.state.obs[3].item<float>();
        const float theta_dot = snapshot_.state.obs[5].item<float>();

        dc.DrawText(wxString::Format("(X Y θ): (%.2f %.2f %.2f[%.1f])", x, y, theta, theta_deg), 10, 70);
        dc.DrawText(wxString::Format("Contact: (%.2f %.2f)", contact_left, contact_right), 10, 90);
        dc.DrawText(wxString::Format("dot(X Y θ): (%.2f %.2f %.2f)", x_dot, y_dot, theta_dot), 10, 110);
    }

    // total_reward
    if (snapshot_.total_reward.has_value()) {
        dc.DrawText(wxString::Format("Total reward: %.3f", snapshot_.total_reward.value()), 10, 130);
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

    DrawWorldLine(dc, width, height);
    DrawTerrain(dc, width, height);
    DrawPad(dc, width, height);
    DrawLegs(dc, width, height);
    DrawLander(dc, width, height);
    DrawThrust(dc, width, height);
    DrawWind(dc, width, height);
    DrawRL(dc);
}
