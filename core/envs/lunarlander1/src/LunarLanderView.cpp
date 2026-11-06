#include "LunarLanderView.hpp"

#include <cmath>
#include <wx/dcbuffer.h>
#include "anet/profile.hpp"
#include "anet/str_util.hpp"
#include "LunarLanderEnv.hpp"

using namespace anet::rl::env;

using anet::ToBool;
using anet::ToFloat;

// サイドエンジンと横風の矢印の縮尺 [px/N]。同じ縮尺にして、風がサイドエンジンの何割かを長さで比べられるようにする
constexpr float kSideForceArrowScale = 3.0f;
// 回転力の円弧の縮尺 [度/(N·m)]
constexpr float kTorqueArcDegreesPerTorque = 60.0f;
// 運動の矢印と円弧が示す先の時間 [s]。速度の矢印の先と角速度の円弧の角度は、いまの値のままこの時間だけ進んだ量
constexpr float kMotionLookaheadSeconds = 0.5f;

// 矢印の先端に返しを描く。dir は world 座標系（y 上向き）の単位方向
static void DrawArrowHead(wxDC& dc, const wxPoint& tip, float dir_x, float dir_y)
{
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

    dc.DrawLine(tip, wxPoint(
            tip.x + static_cast<int>(hx1 * head_len),
            tip.y - static_cast<int>(hy1 * head_len)));
    dc.DrawLine(tip, wxPoint(
            tip.x + static_cast<int>(hx2 * head_len),
            tip.y - static_cast<int>(hy2 * head_len)));
}

// 中心 c の真上を中点とする半径 radius の円弧を sweep [rad] だけ描き、終点に返しを付ける。正の sweep は反時計回り
static void DrawArcArrow(wxDC& dc, const wxPoint& c, float radius, float sweep)
{
    // 1px に満たない円弧は描かない
    if (std::abs(sweep) * radius < 1.0f) {
        return;
    }

    constexpr int arc_segments = 16;
    const float start = static_cast<float>(M_PI) / 2.0f - sweep / 2.0f;
    std::vector<wxPoint> pts;
    pts.reserve(arc_segments + 1);
    for (int i = 0; i <= arc_segments; ++i) {
        const float a = start + sweep * static_cast<float>(i) / arc_segments;
        pts.emplace_back(
            c.x + static_cast<int>(std::round(radius * std::cos(a))),
            c.y - static_cast<int>(std::round(radius * std::sin(a))));
    }
    dc.DrawLines(static_cast<int>(pts.size()), pts.data());

    // 返しは円弧の終点の接線方向（回転の向き）に付ける
    const float end = start + sweep;
    const float sign = sweep > 0.0f ? 1.0f : -1.0f;
    DrawArrowHead(dc, pts.back(), -sign * std::sin(end), sign * std::cos(end));
}


// =============================================================
// LunarLanderData
// =============================================================

LunarLanderData LunarLanderData::Create(anet::rl::TrainEvent event)
{
    ANET_PROFILE_FUNC();

    //ANET_LOG_DEBUG("step_result=" << event.step_result->ToString());

    const int ENV_INDEX = 0;

    // RL由来情報
    auto exp_step = event.counts.exp_step;
    anet::rl::SingleState state = {
        .obs = { anet::rl::ObsKeys::kVector, event.step_result->next_state.obs.At(anet::rl::ObsKeys::kVector)[ENV_INDEX] },
        .done = event.step_result->next_state.done[ENV_INDEX].item<bool>(),
        .truncated = event.step_result->next_state.truncated[ENV_INDEX].item<bool>(),
        .episode_start = event.step_result->next_state.episode_start[ENV_INDEX].item<bool>(),
    };
    auto action = event.experience.action->GetAction(torch::kCPU)[ENV_INDEX].item<int64_t>();
    auto reward = event.experience.reward[ENV_INDEX].item<float>();

    // aux情報
    auto auxs = event.step_result->GetAuxDataList(ENV_INDEX);
    ANET_ASSERT(auxs.size() > 0);
    auto aux = auxs[0];

    // Snapshotを作る
    LunarLanderData snapshot{ exp_step, state, action, reward, aux };

    return snapshot;
}


// =============================================================
// LunarLanderCanvas
// =============================================================

LunarLanderPanel::LunarLanderPanel(wxWindow* parent)
    : anet::rl::gui::Panel(parent)
{
    SetBackgroundStyle(wxBG_STYLE_PAINT);

    Bind(wxEVT_PAINT, &LunarLanderPanel::OnPaint, this);
    Bind(wxEVT_MIDDLE_DOWN, &LunarLanderPanel::OnMiddleClick, this);
}

void LunarLanderPanel::OnMiddleClick(wxMouseEvent& event)
{
    // 段を 1 つ進める。保存済みのスナップショットで描き直すので、停止中も次のデータ更新を待たない
    switch (draw_level_) {
    case DrawLevel::kLander:
        draw_level_ = DrawLevel::kWind;
        break;
    case DrawLevel::kWind:
        draw_level_ = DrawLevel::kMotion;
        break;
    case DrawLevel::kMotion:
        draw_level_ = DrawLevel::kLander;
        break;
    }
    Refresh();
}

void LunarLanderPanel::ApplyData(const LunarLanderData& data)
{
    ANET_LOG_DEBUG("Data captured.");

    // データ（スナップショット）が取り出せたら取り込む
    auto snapshot = data;
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
}

wxPoint LunarLanderPanel::WorldToScreen(float wx, float wy, int width, int height) const
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

int LunarLanderPanel::WorldToScreen(float size, int width, int height) const
{
    const int margin = 40;
    const float world_w = std::max(0.1f, world_max_x_ - world_min_x_);
    const float world_h = std::max(0.1f, world_max_y_ - world_min_y_);
    const float scale = std::min(
        (width - 2 * margin) / world_w,
        (height - 2 * margin) / world_h);

    return static_cast<int>(std::round(size * scale));
}

void LunarLanderPanel::DrawWorldLine(wxDC& dc, int width, int height)
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

void LunarLanderPanel::DrawTerrain(wxDC& dc, int width, int height)
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

    wxColour ground_color(255, 140, 0);
    dc.SetPen(wxPen(ground_color, 3));
    dc.DrawLines(static_cast<int>(pts.size()), pts.data());
    dc.DrawLines(static_cast<int>(pts.size()), pts.data());
}

void LunarLanderPanel::DrawPad(wxDC& dc, int width, int height)
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

void LunarLanderPanel::DrawLander(wxDC& dc, int width, int height)
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

void LunarLanderPanel::DrawLegs(wxDC& dc, int width, int height)
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

void LunarLanderPanel::DrawThrust(wxDC& dc, int width, int height)
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

    if (action == kActionMain) {
        // --- Main Engine ---
        local_off_y = -kLanderRadius;
        local_dir_y = -1.0f;
        arrow_len = kMainEngineForce;
    } else if (action == kActionLeft || action == kActionRight) {
        // --- Side Engine ---
        const float h = kLanderRadius * 0.80f;
        const float r = kLanderRadius;
        const float dx = std::sqrt(std::max(0.0f, r * r - h * h));

        local_off_y = h;

        if (action == kActionLeft) {
            // Left Engine
            local_off_x = -dx;
            local_dir_x = -1.0f;
        } else {
            // Right Engine
            local_off_x = +dx;
            local_dir_x = +1.0f;
        }

        arrow_len = kSideEngineForce * kSideForceArrowScale;
    } else {
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

    const wxPoint p0 = WorldToScreen(start_x, start_y, width, height);

    const wxPoint p1(
        p0.x + static_cast<int>(std::round(dir_x * arrow_len)),
        p0.y - static_cast<int>(std::round(dir_y * arrow_len)));

    dc.SetPen(wxPen(wxColour(220, 0, 0), 2));
    dc.DrawLine(p0, p1);

    // --- Arrow head ---
    DrawArrowHead(dc, p1, dir_x, dir_y);
}

void LunarLanderPanel::DrawWind(wxDC& dc, int width, int height)
{
    auto it = snapshot_.aux.find("forces");
    if (it == snapshot_.aux.end()) {
        return;
    }

    const float wind_x = it->second[0].item<float>();
    const float wind_torque = it->second[1].item<float>();
    dc.DrawText(wxString::Format("Wind: %.2f %.2f", wind_x, wind_torque), 10, height - 20);

    // 数値はどの段でも出し、矢印と円弧は「＋風」の段から出す
    if (draw_level_ < DrawLevel::kWind) {
        return;
    }

    auto it2 = snapshot_.aux.find("lander");
    if (it2 == snapshot_.aux.end()) {
        return;
    }

    // 機体の画面上の位置と大きさ
    const auto lander = it2->second;
    const wxPoint c = WorldToScreen(lander[0].item<float>(), lander[1].item<float>(), width, height);
    const int r = WorldToScreen(kLanderRadius, width, height);

    const wxColour wind_color(127, 119, 221);

    // --- 横風 ---
    // 機体中心から風の向きへ水平の矢印を出す。長さは力に比例し、縮尺はサイドエンジンの矢印と同じ
    const int wind_len = static_cast<int>(std::round(wind_x * kSideForceArrowScale));
    if (wind_len != 0) {
        const wxPoint tip(c.x + wind_len, c.y);
        dc.SetPen(wxPen(wind_color, 3));
        dc.DrawLine(c, tip);
        DrawArrowHead(dc, tip, wind_len > 0 ? 1.0f : -1.0f, 0.0f);
    }

    // --- 回転力 ---
    // 機体を囲む円弧を回転の向きへ描く。角度は回転力に比例し、正は反時計回り（Box2D と同じ）
    const float sweep = wind_torque * kTorqueArcDegreesPerTorque * static_cast<float>(M_PI) / 180.0f;
    dc.SetPen(wxPen(wind_color, 2));
    DrawArcArrow(dc, c, 1.8f * static_cast<float>(r), sweep);
}

void LunarLanderPanel::DrawMotion(wxDC& dc, int width, int height)
{
    // 運動は「＋運動」の段だけで出す
    if (draw_level_ < DrawLevel::kMotion) {
        return;
    }

    auto it = snapshot_.aux.find("lander");
    if (it == snapshot_.aux.end()) {
        return;
    }

    // 機体の位置・速度・角速度
    const auto lander = it->second;
    const float x = lander[0].item<float>();
    const float y = lander[1].item<float>();
    const float vx = lander[2].item<float>();
    const float vy = lander[3].item<float>();
    const float angular_velocity = lander[5].item<float>();

    const wxPoint c = WorldToScreen(x, y, width, height);
    const int r = WorldToScreen(kLanderRadius, width, height);

    const wxColour motion_color(15, 110, 86);
    dc.SetPen(wxPen(motion_color, 2));

    // --- 速度 ---
    // 矢印の先は、いまの速度のまま kMotionLookaheadSeconds だけ進んだ位置。world 座標で描くので、画面の拡大に追従する
    const wxPoint tip = WorldToScreen(
        x + vx * kMotionLookaheadSeconds, y + vy * kMotionLookaheadSeconds, width, height);
    const float dx = static_cast<float>(tip.x - c.x);
    const float dy = static_cast<float>(tip.y - c.y);
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len >= 1.0f) {
        dc.DrawLine(c, tip);
        DrawArrowHead(dc, tip, dx / len, -dy / len);
    }

    // --- 角速度 ---
    // 回転力の円弧の外側に、いまの角速度のまま kMotionLookaheadSeconds で回る角度の円弧を描く。正は反時計回り
    DrawArcArrow(dc, c, 2.4f * static_cast<float>(r), angular_velocity * kMotionLookaheadSeconds);
}

void LunarLanderPanel::DrawRL(wxDC& dc)
{
    //dc.SetTextForeground(*wxBLACK);

    // Step
    auto step_str = anet::FormatWithCommas(snapshot_.step);
    dc.DrawText(wxString::Format("Step: %s", step_str), 10, 10);

    // Reward
    auto raw_reward = 0.0f;
    auto it = snapshot_.aux.find("rewards");
    if (it != snapshot_.aux.end()) raw_reward = ToFloat(it->second[1]);
    dc.DrawText(wxString::Format("Reward: %.2f (%.2f)", snapshot_.reward, raw_reward), 10, 30);

    // Action
    dc.DrawText(wxString::Format("Action: %llu", snapshot_.action), 10, 50);

    // pos
    auto obs = snapshot_.state.obs.At(anet::rl::ObsKeys::kVector);
    if (obs.defined()) {
        const float x = obs[0].item<float>();
        const float y = obs[1].item<float>();
        const float theta = obs[4].item<float>();
        const float theta_deg = 180.0f / M_PI * theta;
        const float contact_left = obs[6].item<float>();
        const float contact_right = obs[7].item<float>();
        const float x_dot = obs[2].item<float>();
        const float y_dot = obs[3].item<float>();
        const float theta_dot = obs[5].item<float>();

        dc.DrawText(wxString::Format("(X Y theta): (%.2f %.2f %.2f[%.1f])", x, y, theta, theta_deg), 10, 70);
        dc.DrawText(wxString::Format("Contact: (%.2f %.2f)", contact_left, contact_right), 10, 90);
        dc.DrawText(wxString::Format("dot(X Y theta): (%.2f %.2f %.2f)", x_dot, y_dot, theta_dot), 10, 110);
    }

    // total_reward
    if (snapshot_.total_reward.has_value()) {
        dc.DrawText(wxString::Format("Total reward: %.3f", snapshot_.total_reward.value()), 10, 130);
    }
}

void LunarLanderPanel::OnPaint(wxPaintEvent& event)
{
    wxAutoBufferedPaintDC dc(this);
    dc.Clear();

    if (!has_snapshot_) {
        dc.DrawText("No data", 10, 50);
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
    DrawMotion(dc, width, height);
    DrawRL(dc);
}


// =============================================================
// LunarLanderView
// =============================================================

LunarLanderView::LunarLanderView(wxWindow* parent)
    : ViewBaseType(parent)
{
    window_ = new LunarLanderPanel(parent_);
}

LunarLanderData LunarLanderView::CreateViewData(const anet::rl::TrainEvent& event) const
{
    return LunarLanderData::Create(event);
}
