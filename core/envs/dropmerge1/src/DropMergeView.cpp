// DropMergeView.cpp
#include "DropMergeView.hpp"
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <wx/dcbuffer.h>
#include <wx/dcgraph.h>
#include "anet/str_util.hpp"
#include "DropMergeEnv.hpp"

using namespace anet::rl::env;

static constexpr int kFruitAlpha = 200;

// =============================================================
// DropMergeData
// =============================================================

DropMergeData DropMergeData::Create(anet::rl::TrainEvent event)
{
    const int ENV_INDEX = 0;

    // RL由来情報
    auto exp_step = event.counts.exp_step;
    anet::rl::SingleState state = event.step_result->next_state.GetSingle(ENV_INDEX);

    auto action = event.action_info.GetAction(torch::kCPU)[ENV_INDEX].item<int64_t>();
    auto reward = event.step_result->reward[ENV_INDEX].item<float>();

    // Aux情報
    auto auxs = event.step_result->GetAuxDataList(ENV_INDEX);
    auto aux = (auxs.empty()) ? anet::rl::AuxData{} : auxs[0];

    DropMergeData snapshot;
    snapshot.step = exp_step;
    snapshot.state = state;
    snapshot.action = action;
    snapshot.reward = reward;
    snapshot.aux = aux;

    return snapshot;
}

// =============================================================
// DropMergePanel
// =============================================================

DropMergePanel::DropMergePanel(wxWindow* parent)
    : anet::rl::gui::Panel(parent, wxID_ANY, wxDefaultPosition)
{
    SetBackgroundStyle(wxBG_STYLE_PAINT);
    Bind(wxEVT_PAINT, &DropMergePanel::OnPaint, this);
    Bind(wxEVT_MIDDLE_DOWN, &DropMergePanel::OnMiddleClick, this);
}

void DropMergePanel::ApplyData(const DropMergeData& data)
{
    snapshot_ = data;
    has_snapshot_ = true;

    // World範囲更新
    if (snapshot_.aux.count("world_info")) {
        auto info = snapshot_.aux.at("world_info"); // width, height, ground_y
        float w = info[0].item<float>();
        float h = info[1].item<float>();
        float gy = info[2].item<float>();

        world_min_x_ = -w * 0.5f;
        world_max_x_ = w * 0.5f;
        world_min_y_ = gy;
        world_max_y_ = gy + h;
    }

    // 描画更新
    Refresh();
}

void DropMergePanel::OnMiddleClick(wxMouseEvent& event)
{
    // Grid描画モード切り替え
    grid_draw_mode_ = (grid_draw_mode_ + 1) % GRID_DRAW_MODE_MAX;
    Refresh();
}

wxColour DropMergePanel::GetFruitColor(int rank) const
{
    // Rank 1..10
    switch (rank) {
    case 1: return wxColour(255, 100, 100);  // Cherry (Pink/Red)
    case 2: return wxColour(255, 20, 20);    // Strawberry (Vivid Red)
    case 3: return wxColour(150, 50, 200);   // Grape (Purple)
    case 4: return wxColour(255, 175, 0);    // Decopon (Orange)
    case 5: return wxColour(255, 140, 0);    // Persimmon (Dark Orange)
    case 6: return wxColour(200, 10, 60);    // Apple (Crimson / Cool Red)
    case 7: return wxColour(200, 200, 50);   // Pear (Yellow)
    case 8: return wxColour(255, 200, 200);  // Peach (Pink)
    case 9: return wxColour(255, 255, 0);    // Pineapple (Yellow)
    case 10: return wxColour(154, 250, 154); // Melon (Light Green)
    case 11: return wxColour(0, 128, 0);     // Watermelon (Dark Green)
    default: return *wxLIGHT_GREY;
    }
}

// -------------------------------------------------------------
// Coordinate System
// -------------------------------------------------------------

wxPoint DropMergePanel::WorldToScreen(float wx, float wy) const
{
    // main_view_rect_ 内での座標変換
    // Y軸反転 (Worldは上が+Y)

    float world_w = world_max_x_ - world_min_x_;
    float world_h = world_max_y_ - world_min_y_;

    // マージン
    int margin_x = 20;
    int margin_y = 20;

    int draw_w = main_view_rect_.width - 2 * margin_x;
    int draw_h = main_view_rect_.height - 2 * margin_y;

    float sx = (float)draw_w / world_w;
    float sy = (float)draw_h / world_h;

    // アスペクト比維持
    float scale = std::min(sx, sy);

    int cx = main_view_rect_.x + main_view_rect_.width / 2;
    int cy = main_view_rect_.y + main_view_rect_.height - margin_y; // 底辺基準

    // 原点(0, ground_y) が画面下部中央に来るように
    // world_min_y_ (ground_y) -> cy

    int sx_out = cx + (int)((wx - 0.0f) * scale);
    int sy_out = cy - (int)((wy - world_min_y_) * scale);

    return wxPoint(sx_out, sy_out);
}

int DropMergePanel::WorldScale(float size) const
{
    float world_w = world_max_x_ - world_min_x_;
    float world_h = world_max_y_ - world_min_y_;
    int margin = 20;
    int draw_w = main_view_rect_.width - 2 * margin;
    int draw_h = main_view_rect_.height - 2 * margin;
    float scale = std::min((float)draw_w / world_w, (float)draw_h / world_h);

    return (int)(size * scale);
}

// -------------------------------------------------------------
// Drawing
// -------------------------------------------------------------

void DropMergePanel::DrawBackground(wxDC& dc)
{
    // 左側: ゲーム画面, 右側: 情報パネル
    wxSize size = GetClientSize();
    int side_panel_width = 250;

	// View Rects
    main_view_rect_ = wxRect(0, 0, size.x - side_panel_width, size.y);
    side_view_rect_ = wxRect(size.x - side_panel_width, 0, side_panel_width, size.y);

    wxColour bgColor = wxSystemSettings::GetColour(wxSYS_COLOUR_3DFACE);

    // Main Area (通常モードの背景色を変更)
    //dc.SetBrush(wxBrush(bgColor));
    //dc.SetPen(*wxTRANSPARENT_PEN);
    //dc.DrawRectangle(main_view_rect_);

    // Side Area (同じ色で塗る)
    dc.SetBrush(wxBrush(bgColor));
    dc.SetPen(*wxTRANSPARENT_PEN);
    dc.DrawRectangle(side_view_rect_);

    // 区切り線 (少し暗く/明るくして境界を目立たせる)
    dc.SetPen(wxPen(wxSystemSettings::GetColour(wxSYS_COLOUR_3DSHADOW)));
    dc.DrawLine(side_view_rect_.x, 0, side_view_rect_.x, size.y);
}

void DropMergePanel::DrawContainer(wxDC& dc)
{
    //  線の色：システムテキスト色（ダークモードなら白、ライトなら黒）
    wxColour lineColor = wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOWTEXT);
    dc.SetPen(wxPen(lineColor, 3));

    // 箱を描画

    // 左壁
    wxPoint p_bl = WorldToScreen(world_min_x_, world_min_y_);
    wxPoint p_tl = WorldToScreen(world_min_x_, world_max_y_);

    // 右壁
    wxPoint p_br = WorldToScreen(world_max_x_, world_min_y_);
    wxPoint p_tr = WorldToScreen(world_max_x_, world_max_y_);

    // 底
    dc.SetPen(wxPen(*wxBLACK, 3));
    dc.DrawLine(p_tl, p_bl); // 左
    dc.DrawLine(p_bl, p_br); // 底
    dc.DrawLine(p_br, p_tr); // 右

    // 上限ライン (破線)
    dc.SetPen(wxPen(*wxRED, 1, wxPENSTYLE_DOT));
    dc.DrawLine(p_tl, p_tr);
}

void DropMergePanel::DrawGrid(wxDC& dc)
{
    if (!snapshot_.aux.count("grid")) return;

    auto t = snapshot_.aux.at("grid"); // [rows, cols]
    int rows = t.size(0);
    int cols = t.size(1);

    float world_w = world_max_x_ - world_min_x_;
    float world_h = world_max_y_ - world_min_y_;
    float cell_w = world_w / cols;
    float cell_h = world_h / rows;


    // グリッド塗りつぶし
    {
        wxPoint p_tl = WorldToScreen(world_min_x_, world_max_y_); // 左上
        wxPoint p_br = WorldToScreen(world_max_x_, world_min_y_); // 右下
        wxRect boardRect(p_tl, p_br);

        wxColour bgColor = wxSystemSettings::GetColour(wxSYS_COLOUR_3DFACE);
        dc.SetBrush(wxBrush(bgColor));
        dc.SetPen(*wxTRANSPARENT_PEN);
        dc.DrawRectangle(boardRect);
    }

    dc.SetPen(wxPen(wxSystemSettings::GetColour(wxSYS_COLOUR_3DSHADOW), 1));
    dc.SetFont(wxFont(9, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));

    // ランク文字の色：Grid Viewなら黒、通常ならシステム色
    if (grid_draw_mode_ == GRID_DRAW_MODE_DEFAULT) dc.SetTextForeground(wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOWTEXT));
    else dc.SetTextForeground(*wxBLACK);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            // セル枠線
            float wx = world_min_x_ + c * cell_w;
            float wy = world_min_y_ + r * cell_h;
            wxPoint p1 = WorldToScreen(wx, wy); // 左下
            wxPoint p2 = WorldToScreen(wx + cell_w, wy + cell_h); // 右上 (Screen座標系ではYが小さい)

            // ランク取得
            float val = t[r][c].item<float>();
            int rank = 0;
            if (val > 0.001f) rank = (int)std::round(val);

            // Grid矩形
            wxRect rect(p1.x, p2.y, p2.x - p1.x, p1.y - p2.y);

            // Grid描画
            if (grid_draw_mode_ == GRID_DRAW_MODE_DEFAULT) {
                // 通常モード: 枠線のみ
                dc.SetBrush(*wxTRANSPARENT_BRUSH);
                dc.SetPen(wxPen(wxSystemSettings::GetColour(wxSYS_COLOUR_3DSHADOW), 1));
                dc.DrawRectangle(rect);
            } else {
                // Grid Viewモード: 塗りつぶし
                if (rank > 0) {
                    wxColour color = GetFruitColor(rank);
                    dc.SetBrush(wxBrush(color));
                    dc.SetPen(*wxTRANSPARENT_PEN); // 枠なし
                    dc.DrawRectangle(rect);
                }

                // グリッド線（薄く）
                dc.SetBrush(*wxTRANSPARENT_BRUSH);
                dc.SetPen(wxPen(wxColour(100, 100, 100), 1));
                dc.DrawRectangle(rect);
            }

            // ランク文字
            if (rank > 0) {
                wxString str = wxString::Format("%d", rank);
                dc.DrawText(str, p1.x + 2, p2.y + 2);
            }
        }
    }
}

void DropMergePanel::DrawFruits(wxDC& dc)
{
	if (grid_draw_mode_ == GRID_DRAW_MODE_GRID_ONLY) return;
    if (!snapshot_.aux.count("fruits")) return;

    auto t = snapshot_.aux.at("fruits"); // Nx5 [x, y, r, rank, angle]
    int n = t.size(0);

    for (int i = 0; i < n; ++i) {
		// データ取得
        float x = t[i][0].item<float>();
        float y = t[i][1].item<float>();
        float r = t[i][2].item<float>();
        int rank = (int)t[i][3].item<float>();
        float angle = t[i][4].item<float>();

        // 座標変換
        wxPoint center = WorldToScreen(x, y);
        int radius = WorldScale(r);

		// 果物本体（円）
        wxColour baseColor = GetFruitColor(rank);
        wxColour alphaColor(baseColor.Red(), baseColor.Green(), baseColor.Blue(), kFruitAlpha);
        dc.SetPen(wxPen(*wxBLACK, 1));
        dc.SetBrush(wxBrush(alphaColor));
        dc.DrawCircle(center, radius);

        // 回転がわかるように線を入れる
        int dx = (int)(std::cos(angle) * radius);
        int dy = (int)(-std::sin(angle) * radius); // Y反転
        dc.DrawLine(center, wxPoint(center.x + dx, center.y + dy));

        // 顔を描く
        dc.SetBrush(*wxBLACK_BRUSH);
        dc.DrawCircle(center.x - radius / 3, center.y - radius / 4, std::max(1, radius / 10)); // 左目
        dc.DrawCircle(center.x + radius / 3, center.y - radius / 4, std::max(1, radius / 10)); // 右目
        // 口 (円弧は面倒なので直線)
        dc.DrawLine(center.x - radius / 4, center.y + radius / 4, center.x + radius / 4, center.y + radius / 4);
    }
}

void DropMergePanel::DrawDropper(wxDC& dc)
{
    if (!snapshot_.aux.count("dropper")) return;

    auto t = snapshot_.aux.at("dropper"); // [x, current_rank, next_rank]
    float x = t[0].item<float>();
    int current_rank = (int)t[1].item<float>();
    float r_curr = t[3].item<float>();
    // int next_rank = (int)t[2].item<float>();

    // 雲（手）の位置
    float dropper_y = world_max_y_;// +0.2;// -0.2f;
    wxPoint pos = WorldToScreen(x, dropper_y);

    // 雲の描画
    dc.SetPen(wxPen(wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOWTEXT), 1));
    dc.SetBrush(*wxWHITE_BRUSH);
    dc.DrawCircle(pos, 10);

    // 持っている果物
    if (current_rank > 0) {
        // 手持ち果物の位置とサイズ
        //float r = 0.09f + (current_rank - 1) * 0.075f;
        //int radius = WorldScale(r);
        int radius = WorldScale(r_curr);
        wxPoint fruit_pos = WorldToScreen(x, dropper_y - 0.3f); // 少し下

        // 手持ち果物を描画
        dc.SetPen(wxPen(*wxBLACK, 1));
        dc.SetBrush(wxBrush(GetFruitColor(current_rank)));
        dc.DrawCircle(fruit_pos, radius);
    }
}

void DropMergePanel::DrawSidePanel(wxDC& dc)
{
    wxRect r = side_view_rect_;
    r.Deflate(10); // padding

    dc.SetFont(wxFont(12, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD));
    dc.SetTextForeground(wxSystemSettings::GetColour(wxSYS_COLOUR_BTNTEXT));

    int y = r.y;
    int line_h = 25;

    // --- NEXT Fruit ---
    dc.DrawText("NEXT:", r.x, y);
    y += line_h * 1.5;

    if (snapshot_.aux.count("dropper")) {
        auto t = snapshot_.aux.at("dropper");
        int next_rank = (int)t[2].item<float>();
        float r_next = t[4].item<float>();

        if (next_rank > 0) {
            int radius = std::min(40, WorldScale(r_next)); // サイドパネルからはみ出ないようキャップ

            // 果物の円
            wxPoint center(r.x + 50, y + 20);
            dc.SetPen(wxPen(*wxBLACK, 1));
            dc.SetBrush(wxBrush(GetFruitColor(next_rank)));
            dc.DrawCircle(center, radius);

            // 顔も描いておく
            dc.SetBrush(*wxBLACK_BRUSH);
            dc.DrawCircle(center.x - radius / 3, center.y - radius / 4, std::max(1, radius / 10));
            dc.DrawCircle(center.x + radius / 3, center.y - radius / 4, std::max(1, radius / 10));
            dc.SetPen(wxPen(*wxBLACK, 1));
            dc.DrawLine(center.x - radius / 4, center.y + radius / 4, center.x + radius / 4, center.y + radius / 4);
        }
    }
    y += 60;

    // スコア
    float score = 0.0f;
    if (snapshot_.aux.count("score")) {
        score = snapshot_.aux.at("score").item<float>();
    } else if (snapshot_.aux.count("rewards")) {
        score = snapshot_.aux.at("rewards")[1].item<float>();
    }
    dc.SetFont(wxFont(14, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD));
    wxString score_str = wxString::Format("Score: %.0f", score * 10.0f);
    dc.DrawText(score_str, r.x, y);

    // 前回のスコア (右側に少し小さく/色を変えて表示)
    if (snapshot_.aux.count("last_score")) {
        float last_score = snapshot_.aux.at("last_score").item<float>();

        if (last_score >= 0.0f) {
            wxSize textSize = dc.GetTextExtent(score_str);
            wxString lastStr = wxString::Format("(%.0f)", last_score * 10.0f);

            // フォントサイズを少し小さく
            dc.SetFont(wxFont(12, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
            dc.SetTextForeground(wxSystemSettings::GetColour(wxSYS_COLOUR_GRAYTEXT)); // グレー

            // "Score: 1230" の幅を測って、その右に描画
            dc.DrawText(lastStr, r.x + textSize.x + 5, y + 4); // 少し位置調整

            // 色とフォントを戻す
            dc.SetTextForeground(wxSystemSettings::GetColour(wxSYS_COLOUR_BTNTEXT));
        }
    }
    y += line_h;

    // エピソード内ステップ数
    int ep_step = 0;
    if (snapshot_.aux.count("step")) {
        ep_step = (int)snapshot_.aux.at("step").item<float>();
    }
    auto ep_step_str = anet::FormatWithCommas(ep_step);
    dc.SetFont(wxFont(14, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD));
    wxString step_str = wxString::Format("Step : %s", ep_step_str);
    dc.DrawText(step_str, r.x, y);

    // 前回のステップ数
    if (snapshot_.aux.count("last_step")) {
        int last_step = (int)snapshot_.aux.at("last_step").item<float>();

        if (last_step >= 0) {
            wxSize stepSize = dc.GetTextExtent(step_str);
            wxString lastStepStr = wxString::Format("(%s)", anet::FormatWithCommas(last_step));

            dc.SetFont(wxFont(12, wxFONTFAMILY_MODERN, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
            dc.SetTextForeground(wxSystemSettings::GetColour(wxSYS_COLOUR_GRAYTEXT));
            dc.DrawText(lastStepStr, r.x + stepSize.x + 5, y + 2);

            dc.SetTextForeground(wxSystemSettings::GetColour(wxSYS_COLOUR_BTNTEXT));
        }
    }
    y += line_h;

    // --- Details ---
    dc.SetFont(wxFont(10, wxFONTFAMILY_MODERN, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    y += line_h;

    // 通算ステップ (Exp Step)
    auto total_step_str = anet::FormatWithCommas(snapshot_.step);
    dc.DrawText(wxString::Format("Total Step: %s", total_step_str), r.x, y);
    y += line_h;

    // Reward
    float reward = snapshot_.reward;
    dc.DrawText(wxString::Format("Reward: %f", reward), r.x, y);
    y += line_h;

    // Action
    int action = (int)snapshot_.action;
    wxString act_str = "NOOP";
    if (action == 1) act_str = "LEFT";
    if (action == 2) act_str = "DROP";
    if (action == 3) act_str = "RIGHT";
    dc.DrawText(wxString::Format("Action: %s", act_str), r.x, y);
    y += line_h;

    // Dropper X
    if (snapshot_.aux.count("dropper")) {
        float dx = snapshot_.aux.at("dropper")[0].item<float>();
        dc.DrawText(wxString::Format("Dropper X: %.2f", dx), r.x, y);
        y += line_h;
    }
    //y += line_h;

    // --- Legend (色とランクの対応表) ---
    // 5列 x 2行 で表示
    {
        int legend_start_y = y;
        int circle_r = 12;
        int gap_x = 30;
        int gap_y = 30;

        dc.SetFont(wxFont(9, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD));
        dc.SetPen(wxPen(wxSystemSettings::GetColour(wxSYS_COLOUR_WINDOWTEXT), 1));

        for (int rank = 1; rank <= kFruitTypeCount; ++rank) {
            int col = (rank - 1) % 6;
            int row = (rank - 1) / 6;

            int cx = r.x + 20 + col * gap_x;
            int cy = legend_start_y + 10 + row * gap_y;

            // 色付き円
            dc.SetBrush(wxBrush(GetFruitColor(rank)));
            dc.DrawCircle(cx, cy, circle_r);

            // 数字 (黒文字で統一)
            dc.SetTextForeground(*wxBLACK);
            wxString str = wxString::Format("%d", rank);
            wxSize ts = dc.GetTextExtent(str);
            dc.DrawText(str, cx - ts.x / 2, cy - ts.y / 2);
        }

        // 描画位置を進める
        y += gap_y * 2 + 20;

        // フォントと色を戻す
        dc.SetFont(wxFont(14, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD));
        dc.SetTextForeground(wxSystemSettings::GetColour(wxSYS_COLOUR_BTNTEXT));
    }
}

void DropMergePanel::OnPaint(wxPaintEvent& event)
{
    // dc
    wxAutoBufferedPaintDC dc(this);
    dc.Clear();

	// GCDC使用可能ならGCDCを使う
    wxDC* pDC = &dc; // GCDCが使えない場合のフォールバック
    wxGCDC gdc(dc);
    if (gdc.IsOk()) {
        pDC = &gdc;
    }

    DrawBackground(*pDC);

    if (!has_snapshot_) {
        dc.DrawText("Waiting for data...", 20, 20);
        return;
    }

    DrawGrid(*pDC);
    DrawContainer(*pDC);
    DrawFruits(*pDC);
    DrawDropper(*pDC);
    DrawSidePanel(*pDC);
}

// =============================================================
// DropMergeView
// =============================================================

DropMergeView::DropMergeView(wxWindow* parent)
    : ViewBaseType(parent)
{
    window_ = new DropMergePanel(parent_);
}

DropMergeData DropMergeView::CreateViewData(const anet::rl::TrainEvent& event) const
{
    return DropMergeData::Create(event);
}

