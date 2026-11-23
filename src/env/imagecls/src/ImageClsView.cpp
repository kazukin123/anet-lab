// ImageClsView.cpp

#include "ImageClsView.hpp"
#include <algorithm>
#include <cstring>
#include <tuple>
#include <wx/dcbuffer.h>
#include "anet/str_util.hpp"

using namespace anet::rl::env;

namespace {

    constexpr int64_t kTopOutputCount = 10;

    std::string MakeLabelName(const std::vector<std::string>& label_names, int64_t label_id)
    {
        if (0 <= label_id && label_id < static_cast<int64_t>(label_names.size()) && !label_names[label_id].empty()) {
            return label_names[label_id];
        }
        return "#" + std::to_string(label_id);
    }

    wxString MakeLabelText(const std::string& label_name, int64_t label_id)
    {
        if (label_id < 0) return "N/A";
        return wxString::FromUTF8(label_name.c_str()) + wxString::Format(" (%lld)", static_cast<long long>(label_id));
    }

    wxString FitTextToWidth(wxDC& dc, const wxString& text, int max_width)
    {
        if (max_width <= 0) return wxString();
        if (dc.GetTextExtent(text).GetWidth() <= max_width) return text;

        const wxString ellipsis = "...";
        if (dc.GetTextExtent(ellipsis).GetWidth() >= max_width) return ellipsis;

        wxString fitted = text;
        while (!fitted.empty() && dc.GetTextExtent(fitted + ellipsis).GetWidth() > max_width) {
            fitted.RemoveLast();
        }
        return fitted.empty() ? ellipsis : fitted + ellipsis;
    }

    int64_t ReadFirstInt64(const torch::Tensor& tensor)
    {
        if (!tensor.defined() || tensor.numel() == 0) return -1;
        return tensor.reshape({ -1 })[0].item<int64_t>();
    }

    bool IsDarkColor(const wxColour& color)
    {
        if (!color.IsOk()) return false;
        const int luminance =
            (299 * color.Red() + 587 * color.Green() + 114 * color.Blue()) / 1000;
        return luminance < 128;
    }

    wxColour MakePredMarkerColor(bool is_dark)
    {
        return is_dark ? wxColour(90, 170, 255) : wxColour(0, 95, 190);
    }

    wxColour MakeTrueMarkerColor(bool is_dark)
    {
        return is_dark ? wxColour(90, 220, 130) : wxColour(0, 130, 70);
    }

} // namespace


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

    int board_area_w = std::max(1, static_cast<int>(width * 0.6f)); // 左60%を画像エリアに
    const int text_x = board_area_w + 20;
    const int text_width = std::max(0, width - text_x - 10);
    int text_y = 20;
    const int line_h = 24;

    auto draw_line = [&](const wxString& text) {
        dc.DrawText(FitTextToWidth(dc, text, text_width), text_x, text_y);
        text_y += line_h;
        };

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
    const wxColour normal_text_color = dc.GetTextForeground();
    const bool is_dark_background = IsDarkColor(GetBackgroundColour());
    const wxColour pred_marker_color = MakePredMarkerColor(is_dark_background);
    const wxColour true_marker_color = MakeTrueMarkerColor(is_dark_background);

    draw_line(wxString::Format("Step: %s", anet::FormatWithCommas(data_.counts.exp_step)));
    draw_line("True: " + MakeLabelText(data_.true_label_name, data_.true_label));

    if (data_.predicted_label >= 0) {
        draw_line("Pred: " + MakeLabelText(data_.predicted_label_name, data_.predicted_label));

        bool is_correct = (data_.true_label == data_.predicted_label);
        wxString result_str = is_correct ? "CORRECT" : "WRONG";

        // 正解なら青、不正解なら赤で強調表示
        dc.SetTextForeground(is_correct ? wxColour(0, 150, 0) : *wxRED);
        draw_line(result_str);
        dc.SetTextForeground(normal_text_color); // 通常の文字色に戻す
    }

    if (!data_.top_outputs.empty()) {
        text_y += line_h / 2;
        draw_line("Top probabilities");

        wxFont output_font(10, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL);
        dc.SetFont(output_font);

        auto draw_probability_line = [&](const wxString& base_text, bool is_predicted, bool is_true) {
            const wxString pred_marker = is_predicted ? " [pred]" : "";
            const wxString true_marker = is_true ? " [true]" : "";
            const int marker_width =
                dc.GetTextExtent(pred_marker).GetWidth() + dc.GetTextExtent(true_marker).GetWidth();

            int draw_x = text_x;
            const wxString fitted_base = FitTextToWidth(dc, base_text, std::max(0, text_width - marker_width));
            dc.SetTextForeground(normal_text_color);
            dc.DrawText(fitted_base, draw_x, text_y);
            draw_x += dc.GetTextExtent(fitted_base).GetWidth();

            if (!pred_marker.empty()) {
                dc.SetTextForeground(pred_marker_color);
                dc.DrawText(pred_marker, draw_x, text_y);
                draw_x += dc.GetTextExtent(pred_marker).GetWidth();
            }
            if (!true_marker.empty()) {
                dc.SetTextForeground(true_marker_color);
                dc.DrawText(true_marker, draw_x, text_y);
            }

            dc.SetTextForeground(normal_text_color);
            text_y += line_h;
            };

        for (const auto& output : data_.top_outputs) {
            const wxString line =
                wxString::Format("%lld. ", static_cast<long long>(output.rank))
                + MakeLabelText(output.label_name, output.label_id)
                + wxString::Format(" %.4f", output.probability);
            draw_probability_line(line, output.is_predicted, output.is_true);
        }
        dc.SetFont(font);
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

    std::vector<std::string> label_names;
    if (event.env) {
        label_names = event.env->GetSpec().action_spec.value_labels;
    }

    std::shared_ptr<const anet::rl::BatchActionInfo> action_info = event.action_info;
    if (!action_info) {
        action_info = event.experience.action;
    }

    // N環境並列の場合、experience は [B, ...] のバッチになっているため、0番目の環境のデータを抽出
    if (event.experience.state.obs.Contains(anet::rl::ObsKeys::kGrid)) {
        auto img = event.experience.state.obs.At(anet::rl::ObsKeys::kGrid);
        if (img.defined() && img.dim() >= 4 && img.size(0) > 0) {
            d.image_tensor = img[0];
        } else {
            d.image_tensor = img;
        }
    }

    if (event.experience.state.obs.Contains(anet::rl::ObsKeys::kVector)) {
        auto tl = event.experience.state.obs.At(anet::rl::ObsKeys::kVector);
        if (tl.defined() && tl.numel() > 0) {
            d.true_label = (tl.dim() >= 2 && tl.size(0) > 0) ? ReadFirstInt64(tl[0]) : ReadFirstInt64(tl);
            d.true_label_name = MakeLabelName(label_names, d.true_label);
        }
    }

    if (action_info) {
        auto pred = action_info->GetAction(torch::kCPU);
        if (pred.defined() && pred.numel() > 0) {
            d.predicted_label = ReadFirstInt64(pred);
            d.predicted_label_name = MakeLabelName(label_names, d.predicted_label);
        }

        const auto& info = action_info->GetInfo();
        if (info.Contains("probs")) {
            auto probs = info.At("probs");
            if (probs.defined() && probs.numel() > 0) {
                torch::Tensor first_probs;
                if (probs.dim() == 1) {
                    first_probs = probs.detach().to(torch::kCPU, torch::kFloat).contiguous();
                } else if (probs.dim() >= 2 && probs.size(0) > 0) {
                    first_probs = probs[0].detach().to(torch::kCPU, torch::kFloat).contiguous();
                }

                if (first_probs.defined() && first_probs.numel() > 0) {
                    first_probs = first_probs.reshape({ -1 }).contiguous();
                    const int64_t top_count = std::min(kTopOutputCount, first_probs.numel());
                    auto top = first_probs.topk(top_count);
                    auto top_probs = std::get<0>(top).contiguous();
                    auto top_indices = std::get<1>(top).contiguous();
                    const float* prob_ptr = top_probs.data_ptr<float>();
                    const int64_t* index_ptr = top_indices.data_ptr<int64_t>();

                    d.top_outputs.reserve(static_cast<size_t>(top_count));
                    for (int64_t i = 0; i < top_count; i++) {
                        const int64_t label_id = index_ptr[i];
                        d.top_outputs.push_back(ImageClsOutputValue{
                            .rank = i + 1,
                            .label_id = label_id,
                            .label_name = MakeLabelName(label_names, label_id),
                            .probability = prob_ptr[i],
                            .is_true = label_id == d.true_label,
                            .is_predicted = label_id == d.predicted_label,
                            });
                    }

                    const float* first_probs_ptr = first_probs.data_ptr<float>();
                    const bool has_true_label =
                        std::any_of(d.top_outputs.begin(), d.top_outputs.end(),
                            [&](const ImageClsOutputValue& output) {
                                return output.label_id == d.true_label;
                            });
                    if (!has_true_label && 0 <= d.true_label && d.true_label < first_probs.numel() && !d.top_outputs.empty()) {
                        const float true_probability = first_probs_ptr[d.true_label];
                        int64_t true_rank = 1;
                        for (int64_t i = 0; i < first_probs.numel(); i++) {
                            if (first_probs_ptr[i] > true_probability) {
                                true_rank++;
                            }
                        }
                        d.top_outputs.back() = ImageClsOutputValue{
                            .rank = true_rank,
                            .label_id = d.true_label,
                            .label_name = MakeLabelName(label_names, d.true_label),
                            .probability = true_probability,
                            .is_true = true,
                            .is_predicted = d.true_label == d.predicted_label,
                        };
                    }
                }
            }
        }
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
