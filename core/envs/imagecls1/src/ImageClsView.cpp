// ImageClsView.cpp

#include "ImageClsView.hpp"
#include <wx/dcbuffer.h>
#include "anet/str_util.hpp"

using namespace anet::rl::env;


// ----------------------------------------------------
// ImageClsPanel
// ----------------------------------------------------

ImageClsPanel::ImageClsPanel(wxWindow* parent)
    : anet::rl::gui::Panel(parent)
{
    SetBackgroundStyle(wxBG_STYLE_PAINT);
    Bind(wxEVT_PAINT, &ImageClsPanel::OnPaint, this);
}

void ImageClsPanel::ApplyData(const ImageClsData& data)
{
    data_ = data;
    UpdateBitmap(); // 新しいデータが来たらBitmapを作り直す
}

void ImageClsPanel::UpdateBitmap()
{
    if (!data_.image_tensor.defined()) return;

    // GPUにある場合はCPUへ、かつ計算グラフから切り離す
    auto cpu_tensor = data_.image_tensor.cpu().detach();

    // PyTorch [C, H, W] -> wxWidgets [H, W, C] への変換
    cpu_tensor = cpu_tensor.permute({ 1, 2, 0 }).contiguous();

    // Float型（前処理で正規化済みなど）の場合は 0-255 の Byte 型へスケーリング
    if (cpu_tensor.dtype() == torch::kFloat32 || cpu_tensor.dtype() == torch::kFloat64) {
        float min_val = cpu_tensor.min().item<float>();
        float max_val = cpu_tensor.max().item<float>();
        if (max_val > min_val) {
            cpu_tensor = (cpu_tensor - min_val) / (max_val - min_val);
        }
        cpu_tensor = cpu_tensor.mul(255).toType(torch::kByte);
    } else {
        cpu_tensor = cpu_tensor.toType(torch::kByte);
    }

    int h = cpu_tensor.size(0);
    int w = cpu_tensor.size(1);

    // wxImageの構築 (ピクセルデータのコピー)
    wxImage img(w, h, false);
    unsigned char* dest = img.GetData();
    unsigned char* src = cpu_tensor.data_ptr<uint8_t>();
    std::memcpy(dest, src, w * h * 3);

    current_bitmap_ = wxBitmap(img);
}

void ImageClsPanel::OnPaint(wxPaintEvent& event)
{
    wxAutoBufferedPaintDC dc(this);
    dc.Clear();

    if (!data_.is_valid) return;

    const wxSize size = GetClientSize();
    const int width = size.GetWidth();
    const int height = size.GetHeight();

    int board_area_w = static_cast<int>(width * 0.6f); // 左60%を画像エリアに
    int text_x = board_area_w + 20;
    int text_y = 20;
    int line_h = 24;

    // 画像の描画（アスペクト比を維持して最大化）
    if (current_bitmap_.IsOk()) {
        double scale_x = (double)board_area_w / current_bitmap_.GetWidth();
        double scale_y = (double)height / current_bitmap_.GetHeight();
        double scale = std::min(scale_x, scale_y);

        wxImage img = current_bitmap_.ConvertToImage();
        wxBitmap scaled_bmp(img.Scale((int)(img.GetWidth() * scale), (int)(img.GetHeight() * scale), wxIMAGE_QUALITY_HIGH));

        // 中央寄せで描画
        int draw_x = (board_area_w - scaled_bmp.GetWidth()) / 2;
        int draw_y = (height - scaled_bmp.GetHeight()) / 2;
        dc.DrawBitmap(scaled_bmp, draw_x, draw_y, false);
    }

    // 右側のテキスト情報描画
    wxFont font(12, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD);
    dc.SetFont(font);

    dc.DrawText(wxString::Format("Step: %lld", data_.counts.exp_step), text_x, text_y); text_y += line_h;
    dc.DrawText(wxString::Format("True Label: %lld", data_.true_label), text_x, text_y); text_y += line_h;

    if (data_.predicted_label >= 0) {
        dc.DrawText(wxString::Format("Pred Label: %lld", data_.predicted_label), text_x, text_y); text_y += line_h;

        bool is_correct = (data_.true_label == data_.predicted_label);
        wxString result_str = is_correct ? "CORRECT" : "WRONG";

        // 正解なら青、不正解なら赤で強調表示
        dc.SetTextForeground(is_correct ? wxColour(0, 150, 0) : *wxRED);
        dc.DrawText(result_str, text_x, text_y); text_y += line_h;
        dc.SetTextForeground(*wxBLACK); // 戻す
    }
}


// ----------------------------------------------------
// ImageClsView
// ----------------------------------------------------

ImageClsView::ImageClsView(wxWindow* parent) : ViewBaseType(parent)
{
    window_ = new ImageClsPanel(parent_);
}

ImageClsData ImageClsView::CreateViewData(const anet::rl::TrainEvent& event) const
{
    ImageClsData d;
    d.counts = event.counts;
    d.is_valid = true;

    // N環境並列の場合、experience は [B, ...] のバッチになっているため、0番目の環境のデータを抽出
    if (event.experience.state.obs.Contains(anet::rl::ObsKeys::kGrid)) {
        auto img = event.experience.state.obs.At(anet::rl::ObsKeys::kGrid);
        d.image_tensor = img[0];
    }

    if (event.experience.state.obs.Contains(anet::rl::ObsKeys::kVector)) {
        auto tl = event.experience.state.obs.At(anet::rl::ObsKeys::kVector);
        d.true_label = tl[0].item<int64_t>();
    }

    if (event.experience.action) {
        auto pred = event.experience.action->GetAction();
        d.predicted_label = pred[0].item<int64_t>();
    }

    return d;
}

// ----------------------------------------------------
// ImageClsViewCreator
// ----------------------------------------------------

std::shared_ptr<anet::rl::gui::View> ImageClsViewCreator::CreateView(
    wxWindow* parent, const anet::ConfigData& config_data, std::shared_ptr<anet::rl::Notifier> notifier) const
{
    return std::make_shared<ImageClsView>(parent);
}
