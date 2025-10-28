#include "anet/HeatMap.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <filesystem>
#include <wx/log.h>
#include <random>

using namespace anet;

static void ValueToRGB_HSL(float v, unsigned char& r, unsigned char& g, unsigned char& b) {
    v = std::clamp(v, 0.0f, 1.0f);
    float h = (1.0f - v) * 240.0f;
    float s = 1.0f, l = 0.5f;
    float c = (1.0f - std::fabs(2.0f * l - 1.0f)) * s;
    float hp = h / 60.0f;
    float x = c * (1 - std::fabs(std::fmod(hp, 2.0f) - 1));
    float r1, g1, b1;
    if (hp < 1) { r1 = c; g1 = x; b1 = 0; }
    else if (hp < 2) { r1 = x; g1 = c; b1 = 0; }
    else if (hp < 3) { r1 = 0; g1 = c; b1 = x; }
    else if (hp < 4) { r1 = 0; g1 = x; b1 = c; }
    else if (hp < 5) { r1 = x; g1 = 0; b1 = c; }
    else { r1 = c; g1 = 0; b1 = x; }
    float m = l - c / 2;
    r = static_cast<unsigned char>((r1 + m) * 255);
    g = static_cast<unsigned char>((g1 + m) * 255);
    b = static_cast<unsigned char>((b1 + m) * 255);
}

static void ValueToRGB_Jet(float norm, unsigned char& r, unsigned char& g, unsigned char& b) {
    norm = std::clamp(norm, 0.0f, 1.0f);

    float r_f = 0.0f, g_f = 0.0f, b_f = 0.0f;

    // Jet colormap
    if (norm < 0.125f) {           // 黒〜青
        r_f = 0.0f;
        g_f = 0.0f;
        b_f = 0.5f + norm * 4.0f;
    }
    else if (norm < 0.375f) {    // 青〜シアン
        r_f = 0.0f;
        g_f = (norm - 0.125f) * 4.0f;
        b_f = 1.0f;
    }
    else if (norm < 0.625f) {    // シアン〜黄
        r_f = (norm - 0.375f) * 4.0f;
        g_f = 1.0f;
        b_f = 1.0f - (norm - 0.375f) * 4.0f;
    }
    else if (norm < 0.875f) {    // 黄〜赤
        r_f = 1.0f;
        g_f = 1.0f - (norm - 0.625f) * 4.0f;
        b_f = 0.0f;
    }
    else {                       // 赤〜暗赤
        r_f = 1.0f - (norm - 0.875f) * 4.0f;
        g_f = 0.0f;
        b_f = 0.0f;
    }

    r = static_cast<unsigned char>(std::clamp(r_f, 0.0f, 1.0f) * 255);
    g = static_cast<unsigned char>(std::clamp(g_f, 0.0f, 1.0f) * 255);
    b = static_cast<unsigned char>(std::clamp(b_f, 0.0f, 1.0f) * 255);
}

// ============================================================
// ImageSource
// ============================================================
void ImageSource::SavePng(const std::string& filename, int width, int height) const {
    std::filesystem::path path(filename);
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());

    auto img = Render(width, height);
    if (!img.IsOk()) {
        wxLogError("HeatMap::Render() returned invalid wxImage");
        return;
    }
    img.SaveFile(filename, wxBITMAP_TYPE_PNG);
}

// ============================================================
// HeatMap
// ============================================================
HeatMap::HeatMap(int width, int height,
    float x_min, float x_max,
    float y_min, float y_max,
    size_t max_points, uint32_t flags)
    : width_(width), height_(height),
    x_min_(x_min), x_max_(x_max),
    y_min_(y_min), y_max_(y_max),
    value_min_(std::numeric_limits<float>::max()),
    value_max_(-std::numeric_limits<float>::max()),
    max_points_(max_points),
    flags_(flags) {
}

void HeatMap::AddData(float x, float y, float value) {
    std::lock_guard<std::mutex> lock(mtx_);
    samples_.push_back({ x, y, value });
    if (max_points_ > 0 && samples_.size() > max_points_)
        samples_.pop_front();
    if (flags_ & HM_AutoNorm) UpdateMinMax(value);
}

void HeatMap::Reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    samples_.clear();
    value_min_ = std::numeric_limits<float>::max();
    value_max_ = -std::numeric_limits<float>::max();
}

void HeatMap::UpdateMinMax(float v) {
    value_min_ = std::min(value_min_, v);
    value_max_ = std::max(value_max_, v);
}

wxImage HeatMap::RenderRaw() const {
    std::vector<Sample> snapshot;
    { std::lock_guard<std::mutex> lock(mtx_); snapshot.assign(samples_.begin(), samples_.end()); }

    std::vector<float> buffer(width_ * height_, 0.0f);
    std::vector<int> count(width_ * height_, 0);

    // バッファと訪問カウントを構築
    for (const auto& p : snapshot) {
        int ix = static_cast<int>((p.x - x_min_) / (x_max_ - x_min_) * width_);
        int iy = static_cast<int>((p.y - y_min_) / (y_max_ - y_min_) * height_);
        if (ix < 0 || ix >= width_ || iy < 0 || iy >= height_) 
            continue;
        int idx = iy * width_ + ix;
        buffer[idx] += p.value;
        count[idx]++;
    }

    if (flags_ & HM_MeanMode)
        for (size_t i = 0; i < buffer.size(); ++i)
            if (count[i] > 0) buffer[i] /= count[i];

    return GenerateImage(buffer, count);
}

wxImage HeatMap::GenerateImage(const std::vector<float>& buf, const std::vector<int>& count) const {
    wxImage img(width_, height_);
    img.SetData(new unsigned char[width_ * height_ * 3]);
    unsigned char* data = img.GetData();

    float vmin = value_min_, vmax = value_max_;
    if (vmin == vmax) vmax = vmin + 1e-6f;

    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            int idx = y * width_ + x;
            float v = buf[idx];
            unsigned char r = 0, g = 0, b = 0;

            if (count[idx] == 0) {
                // 未訪問 → 黒
                r = g = b = 0;
            }
            else if (v == 0.0f) {
                // 値が0（訪問あり） → 灰
                r = g = b = 50;
            }
            else {
                float val = v;
                if (flags_ & HM_LogScale) {
                    float sign = (val >= 0.0f) ? 1.0f : -1.0f;
                    val = sign * std::log1p(std::abs(val));
                }
                float norm = (val - vmin) / std::max(vmax - vmin, 1e-6f);
                ValueToRGB_Jet(norm, r, g, b);
            }

            int di = ((height_ - 1 - y) * width_ + x) * 3;
            data[di] = r;
            data[di + 1] = g;
            data[di + 2] = b;
        }
    }
    return img;
}

TimeHeatMap::TimeHeatMap(int width_bins, int height,
    float min_val, float max_val,
    uint32_t flags,
    size_t max_points,
    TimeFrameMode mode)
    : HeatMap(width_bins, height, min_val, max_val, 0.0f, float(height - 1),
        max_points, flags),
    mode_(mode),
    cur_frame_(0),
    total_frames_(0)
{
}

void TimeHeatMap::AddData(float x, float value) {
    std::lock_guard<std::mutex> lock(mtx_);
    HeatMap::AddData(x, (float)cur_frame_, value);
}

void TimeHeatMap::Reset() {
    HeatMap::Reset();
    cur_frame_ = 0;        // 現在のy行
    total_frames_ = 0;     // 累計フレーム数（Unlimited時に増え続ける）
}

void TimeHeatMap::ScrollDown_()
{
    // y を 1 つ上へずらす（y=0 が最古 / y=height-1 が最新）
    for (auto& d : this->samples_) { // data_ は HeatMap 側の保持領域を想定
        d.y -= 1.0f;
    }
    // 画面上端を超えた要素を削除
    EraseRow_(0);
}

void TimeHeatMap::NextFrame() {
    std::lock_guard<std::mutex> lock(mtx_);

    if (mode_ == TimeFrameMode::Scroll) {
        ScrollDown_();
        cur_frame_ = height_ - 1; // 最新は常に下
    } else { // Unlimited
        cur_frame_++;
        total_frames_++;
    }
}

void TimeHeatMap::EraseRow_(int y_row) {
    auto it = samples_.begin();
    while (it != samples_.end()) {
        if (std::lround(it->y) == y_row) it = samples_.erase(it);
        else ++it;
    }
}

wxImage TimeHeatMap::RenderRaw() const
{
    std::lock_guard<std::mutex> lock(mtx_);
    return HeatMap::RenderRaw();
}

//wxImage TimeHeatMap::RenderRaw() const {
//    if (mode_ == TimeFrameMode::Unlimited && cur_frame_> height_) {
//        // 高さを超えるy座標も含む全サンプルのうち、
//        // 最後のheight_範囲だけを描画
//        float y_min = std::max(0.0f, float(cur_frame_ - height_ + 1));
//        float y_max = float(cur_frame_);
//
//        // 一時HeatMapを構築
//        HeatMap temp(width_, height_,
//            x_min_, x_max_,
//            y_min, y_max,
//            0, flags_);
//
//        {
//            std::lock_guard<std::mutex> lk(mtx_);
//            for (const auto& s : samples_) {
//                if (s.y >= y_min && s.y <= y_max)
//                    temp.AddData(s.x, s.y, s.value);
//            }
//        }
//
//        return temp.RenderRaw().Mirror(false);
//    }
//
//    // Overwrite / Scroll は通常描画
//    return HeatMap::RenderRaw();
//}

// ============================================================
// Histgram
// ============================================================
Histgram::Histgram(int bins, float min_val, float max_val, int width, int height)
    : bins_(bins), min_val_(min_val), max_val_(max_val), width_(width), height_(height),
    counts_(bins, 0) {
}

void Histgram::AddData(float value) {
    std::lock_guard<std::mutex> lock(mtx_);
    int idx = static_cast<int>((value - min_val_) / (max_val_ - min_val_) * bins_);
    if (idx >= 0 && idx < bins_) counts_[idx]++;
}

void Histgram::Reset() {
    std::fill(counts_.begin(), counts_.end(), 0);
}

wxImage Histgram::RenderRaw() const {
    std::lock_guard<std::mutex> lock(mtx_);  // ← 同期（任意）

    wxImage img(width_, height_);
    // バッファを明示確保して黒で初期化
    img.SetData(new unsigned char[width_ * height_ * 3]);
    std::fill_n(img.GetData(), width_ * height_ * 3, 0);

    unsigned char* data = img.GetData();
    int max_count = *std::max_element(counts_.begin(), counts_.end());
    if (max_count <= 0) return img; // 全0なら真っ黒のまま返す

    // 各binの横幅（x方向のピクセル範囲）を計算
    for (int i = 0; i < bins_; ++i) {
        // 高さは画面の高さでスケーリング（←修正）
        int h = static_cast<int>((float)counts_[i] / std::max(1, max_count) * height_);
        h = std::min(h, height_);

        // このbinが占めるx範囲 [x0, x1)
        int x0 = (i * width_) / bins_;
        int x1 = ((i + 1) * width_) / bins_;
        x0 = std::clamp(x0, 0, width_);
        x1 = std::clamp(x1, 0, width_);
        if (x1 <= x0) continue;

        // 白で塗る
        for (int x = x0; x < x1; ++x) {
            for (int y = 0; y < h; ++y) {
                int py = height_ - 1 - y;
                int idx = (py * width_ + x) * 3;
                data[idx + 0] = 255;
                data[idx + 1] = 255;
                data[idx + 2] = 255;
            }
        }
    }
    return img;
}

// ============================================================
// TimeHistogram
// ============================================================
    TimeHistogram::TimeHistogram(int bins, int max_frames, TimeFrameMode mode, bool auto_norm, bool auto_range, float min_val, float max_val) :
            thm_(bins, max_frames, 0, bins, HM_Default, 0, mode),bins_(bins),min_val_(min_val),max_val_(max_val),
            auto_norm_(auto_norm),buffer_(bins, 0.0f) {
    }

    void TimeHistogram::AddBatch(const std::vector<float>& values) {
    if (values.empty()) return;

    float cur_min = *std::min_element(values.begin(), values.end());
    float cur_max = *std::max_element(values.begin(), values.end());

    if (auto_range_) {
        if (cur_min < min_val_) min_val_ = cur_min;
        if (cur_max > max_val_) max_val_ = cur_max;

        // 緩やかなスムージング（指数平均）
        smooth_min_ = (1 - smooth_rate_) * smooth_min_ + smooth_rate_ * cur_min;
        smooth_max_ = (1 - smooth_rate_) * smooth_max_ + smooth_rate_ * cur_max;
    }

    float range = smooth_max_ - smooth_min_;
    if (range <= 1e-6f) range = 1.0f;

    for (float v : values) {
        int ix = static_cast<int>((v - smooth_min_) / range * bins_);
        ix = std::clamp(ix, 0, bins_ - 1);
        buffer_[ix] += 1.0f;
    }
}

void TimeHistogram::NextFrame() {
    // 正規化処理
    if (auto_norm_) {
        float maxv = *std::max_element(buffer_.begin(), buffer_.end());
        if (maxv > 0.0f) {
            for (auto& v : buffer_) v /= maxv;
        }
    }

    // TimeHeatMap に1ライン転送
    for (int x = 0; x < bins_; ++x)
        thm_.AddData(x, buffer_[x]);

    // 次フレーム用にリセット
    std::fill(buffer_.begin(), buffer_.end(), 0.0f);
    thm_.NextFrame();
}

void TimeHistogram::Reset() {
    thm_.Reset();
}

// ============================================================
// SweepedHeatMap
// ============================================================
SweepedHeatMap::SweepedHeatMap(int width, int height,
    float x_min, float x_max,
    float y_min, float y_max)
    : width_(width), height_(height),
    x_min_(x_min), x_max_(x_max),
    y_min_(y_min), y_max_(y_max),
    values_(width* height, 0.0f),
    value_min_(0.0f), value_max_(1.0f) {
}

void SweepedHeatMap::Evaluate(const std::function<float(float, float)>& func) {
    value_min_ = std::numeric_limits<float>::max();
    value_max_ = -std::numeric_limits<float>::max();
    for (int j = 0; j < height_; ++j) {
        for (int i = 0; i < width_; ++i) {
            float x = x_min_ + (x_max_ - x_min_) * i / (width_ - 1);
            float y = y_min_ + (y_max_ - y_min_) * j / (height_ - 1);
            float val = func(x, y);
            values_[j * width_ + i] = val;
            value_min_ = std::min(value_min_, val);
            value_max_ = std::max(value_max_, val);
        }
    }
}

void SweepedHeatMap::Normalize() {
    float range = std::max(value_max_ - value_min_, 1e-6f);
    for (float& v : values_) v = (v - value_min_) / range;
}

wxImage SweepedHeatMap::RenderRaw() const {
    wxImage img(width_, height_);
    unsigned char* data = img.GetData();
    float range = std::max(value_max_ - value_min_, 1e-6f);
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            float v = (values_[y * width_ + x] - value_min_) / range;
            unsigned char r, g, b;
            ValueToRGB_Jet(v, r, g, b);
            int idx = ((height_ - 1 - y) * width_ + x) * 3;
            data[idx] = r; data[idx + 1] = g; data[idx + 2] = b;
        }
    }
    return img;
}

void anet::SweepedHeatMap::Reset() {
    this->values_.clear();
    value_min_ = 0.0f;
    value_max_ = 1.0f;
}

// GPUバッチ対応版 Sweep
SweepedHeatMap SweepedHeatMap::EvaluateTensorFunction(
    int width, int height,
    float x_min, float x_max,
    float y_min, float y_max,
    const torch::Device& device,
    const std::function<torch::Tensor(const torch::Tensor&)>& nn_forward,
    const std::function<torch::Tensor(const torch::Tensor&)>& value_extractor)
{
    SweepedHeatMap map(width, height, x_min, x_max, y_min, y_max);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto x_t = torch::linspace(x_min, x_max, width, options).repeat({ height, 1 }).view({ -1, 1 });
    auto y_t = torch::linspace(y_min, y_max, height, options).repeat_interleave(width).view({ -1, 1 });
    auto input = torch::cat({ x_t, y_t }, 1);

    auto output = nn_forward(input);
    auto values = value_extractor(output)
        .detach()
        .to(torch::kCPU)
        .view({ -1 });

    auto acc = values.accessor<float, 1>();
    for (int idx = 0; idx < width * height; ++idx)
        map.values_[idx] = acc[idx];

    map.Normalize();
    return map;
}


void test_heatmap_and_histgram() {

    const int steps = 500;
    std::mt19937 rng(1234);

    // 疑似 TD誤差データ（時間とともに分布が狭まる）
    std::vector<float> td_errors;
    td_errors.reserve(steps);
    for (int t = 0; t < steps; ++t) {
        float sigma = 2.5f * std::exp(-0.01f * t);
        std::normal_distribution<float> dist(0.0f, sigma);
        td_errors.push_back(dist(rng));
    }

    // ============================================================
    // ① ヒストグラム — TD誤差分布（全期間まとめ）
    // ============================================================
    {
        anet::Histgram hist(100, -3.0f, 3.0f);
        for (float v : td_errors) hist.AddData(v);
        hist.SavePng("1_out_histogram_td.png");
    }

    // ============================================================
    // ② ヒートマップ — x=step, y=fixed (1ライン), 色=TD誤差値
    // ============================================================
    {
        anet::HeatMap map(steps, 1, 0, (float)steps, 0, 1, steps, anet::HM_AutoNorm);
        for (int t = 0; t < steps; ++t)
            map.AddData((float)t, 0.5f, td_errors[t]);
        map.SavePng("2_out_heatmap_td.png");
    }

    // ============================================================
    // ③ 時系列ヒストグラム
    // ============================================================
    {
        //TimeHistogram(int bins, int max_frames,
        //    TimeFrameMode mode = TimeFrameMode::Scroll,
        //    bool auto_norm = true, bool auto_range = true,
        //    float min_val = -1, float max_val = -1);

        TimeHistogram q_hist(64, 50, 
            anet::TimeFrameMode::Scroll,
            true, true,
            - 1.0f, 2.0f);

        std::normal_distribution<float> dist_center(0.0f, 0.4f);
        std::normal_distribution<float> dist_shift(0.0f, 0.02f);

        float center = 0.0f;

        for (int frame = 0; frame < 100; ++frame) {
            std::vector<float> batch;

            // フレームごとに少し中心を移動させる
            center += dist_shift(rng);
            std::normal_distribution<float> dist(center, 0.3f);

            // 乱数バッチ生成（128サンプル）
            for (int i = 0; i < 1280; ++i)
                batch.push_back(dist(rng));

            q_hist.AddBatch(batch);

            // 10フレームごとに1ライン追加
            if ((frame + 1) % 2 == 0)
                q_hist.NextFrame();
        }

        // 画像を生成
        q_hist.SavePng("4_out_timeheat.png");
    }

}
