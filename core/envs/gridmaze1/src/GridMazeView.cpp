// GridMazeView.cpp

#include "GridMazeView.hpp"
#include <wx/dcbuffer.h>
#include "anet/str_util.hpp"

using namespace anet::rl::env;


GridMazePanel::GridMazePanel(wxWindow* parent)
    : anet::rl::gui::Panel(parent)
{
    SetBackgroundStyle(wxBG_STYLE_PAINT);
    Bind(wxEVT_PAINT, &GridMazePanel::OnPaint, this);
}

void GridMazePanel::ApplyData(const GridMazeData& data)
{
    data_ = data;
}

void GridMazePanel::OnPaint(wxPaintEvent& event)
{
    wxAutoBufferedPaintDC dc(this);
    dc.Clear();

    if (!data_.map_tensor.defined()) return;

    const wxSize size = GetClientSize();
    const int width = size.GetWidth();
    const int height = size.GetHeight();

    int board_area_w = static_cast<int>(width * 0.7f);
    int side_area_x = board_area_w;

    int cols = data_.map_width;
    int rows = data_.map_height;
    if (cols == 0 || rows == 0) return;

    int cell_size = std::min(board_area_w / cols, height / rows) - 2;
    int offset_x = (board_area_w - cell_size * cols) / 2;
    int offset_y = (height - cell_size * rows) / 2;

    auto map_acc = data_.map_tensor.accessor<float, 2>();

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int cell_type = static_cast<int>(map_acc[y][x]);
            wxRect rect(offset_x + x * cell_size, offset_y + y * cell_size, cell_size, cell_size);

            dc.SetPen(*wxBLACK_PEN);
            if (cell_type == 0) { // Empty
                dc.SetBrush(*wxWHITE_BRUSH);
            } else if (cell_type == 1) { // Wall
                dc.SetBrush(*wxGREY_BRUSH);
            } else if (cell_type == 2) { // Hole
                dc.SetBrush(*wxBLACK_BRUSH);
            } else if (cell_type == 3) { // Goal
                dc.SetBrush(wxBrush(wxColour(255, 215, 0))); // Gold
            }
            dc.DrawRectangle(rect);
        }
    }

    wxRect agent_rect(offset_x + data_.pos_x * cell_size + 4,
        offset_y + data_.pos_y * cell_size + 4,
        cell_size - 8, cell_size - 8);
    dc.SetPen(*wxTRANSPARENT_PEN);
    dc.SetBrush(*wxBLUE_BRUSH);
    dc.DrawEllipse(agent_rect);

    dc.SetPen(wxPen(wxColour(200, 200, 200), 1));
    dc.DrawLine(side_area_x, 0, side_area_x, height);

    int text_x = side_area_x + 10;
    int text_y = 10;
    int line_h = 20;

    const int BATCH_POS = 0;
    float reward = data_.train_exp.reward[BATCH_POS].item<float>();
    int64_t action = data_.train_exp.action.GetAction(torch::kCPU)[BATCH_POS].item<int64_t>();
    std::string act_str = (action == 0) ? "Up" : (action == 1) ? "Left" : (action == 2) ? "Down" : "Right";

    dc.DrawText("--- GridMaze ---", text_x, text_y); text_y += line_h * 2;
    dc.DrawText(wxString::Format("Step: %s", anet::FormatWithCommas(data_.counts.exp_step)), text_x, text_y); text_y += line_h;
    dc.DrawText(wxString::Format("Seed: %lld", static_cast<long long>(data_.current_seed)), text_x, text_y); text_y += line_h;
    dc.DrawText(wxString::Format("Pos: (%d, %d)", data_.pos_x, data_.pos_y), text_x, text_y); text_y += line_h;
    dc.DrawText(wxString::Format("Action: %s", act_str), text_x, text_y); text_y += line_h;
    dc.DrawText(wxString::Format("Reward: %.2f", reward), text_x, text_y); text_y += line_h;
    dc.DrawText(wxString::Format("Total Reward: %.2f", data_.ep_reward_sum), text_x, text_y); text_y += line_h;
}

GridMazeView::GridMazeView(wxWindow* parent) : ViewBaseType(parent)
{
    window_ = new GridMazePanel(parent_);
}

GridMazeData GridMazeView::CreateViewData(const anet::rl::TrainEvent& event) const
{
    GridMazeData d;
    d.counts = event.counts;
    d.train_exp = event.experience;

    if (event.step_result) {
        auto aux_list = event.step_result->GetAuxDataList(0);
        if (!aux_list.empty()) {
            const auto& aux = aux_list[0];

            auto get_float = [&](const std::string& key, float def) {
                auto it = aux.find(key);
                return (it != aux.end()) ? it->second.item<float>() : def;
                };

            auto get_int64 = [&](const std::string& key, int64_t def) {
                auto it = aux.find(key);
                return (it != aux.end()) ? it->second.item<int64_t>() : def;
                };

            d.current_seed = get_int64("seed", 0);
            d.map_width = static_cast<int>(get_float("width", 5.0f));
            d.map_height = static_cast<int>(get_float("height", 5.0f));
            d.pos_x = static_cast<int>(get_float("pos_x", 0.0f));
            d.pos_y = static_cast<int>(get_float("pos_y", 0.0f));
            d.ep_reward_sum = get_float("ep_reward_sum", 0.0f);

            auto it = aux.find("map");
            if (it != aux.end()) {
                d.map_tensor = it->second;
            }
        }
    }
    return d;
}
