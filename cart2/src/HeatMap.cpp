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
void ImageSource::SavePng(const std::string& filename) const {
    std::filesystem::path path(filename);
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());

    auto img = Render();
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

wxImage HeatMap::Render() const {
    std::vector<Sample> snapshot;
    { std::lock_guard<std::mutex> lock(mtx_); snapshot.assign(samples_.begin(), samples_.end()); }

    std::vector<float> buffer(width_ * height_, 0.0f);
    std::vector<int> count(width_ * height_, 0);

    // バッファと訪問カウントを構築
    for (const auto& p : snapshot) {
        int ix = static_cast<int>((p.x - x_min_) / (x_max_ - x_min_) * width_);
        int iy = static_cast<int>((p.y - y_min_) / (y_max_ - y_min_) * height_);
        if (ix < 0 || ix >= width_ || iy < 0 || iy >= height_) continue;
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
    float x_min, float x_max,
    uint32_t flags,
    size_t max_points,
    TimeFrameMode mode)
    : HeatMap(width_bins, height, x_min, x_max, 0.0f, float(height - 1),
        max_points, flags),
    mode_(mode),
    cur_frame_(0),
    total_frames_(0)
{
}

void TimeHeatMap::AddData(float x, float value) {
    HeatMap::AddData(x, float(cur_frame_), value);
}

void TimeHeatMap::Reset() {
    HeatMap::Reset();
    cur_frame_ = 0;        // 現在のy行
    total_frames_ = 0;     // 累計フレーム数（Unlimited時に増え続ける）
}

void TimeHeatMap::NextFrame() {
    std::lock_guard<std::mutex> lk(mtx_);
    total_frames_++;

    if (mode_ == TimeFrameMode::Unlimited) {
        // y座標を単調増加させ続ける
        cur_frame_++;
        // height_ は固定のまま。描画時にスライス。
        return;
    }
    else if (mode_ == TimeFrameMode::Overwrite) {
        cur_frame_ = (cur_frame_ + 1) % height_;
        PurgeRowUnchecked_(cur_frame_);
    }
    else if (mode_ == TimeFrameMode::Scroll) {
        if (cur_frame_ < height_ - 1) cur_frame_++;
        else { ScrollUp_(); cur_frame_ = height_ - 1; }
    }
}

void TimeHeatMap::PurgeRowUnchecked_(int y_row) {
    auto it = samples_.begin();
    while (it != samples_.end()) {
        if (std::lround(it->y) == y_row) it = samples_.erase(it);
        else ++it;
    }
}

void TimeHeatMap::ScrollUp_() {
    // y値を1行上げ、最上段(y==0)を削除
    for (auto it = samples_.begin(); it != samples_.end();) {
        if (std::lround(it->y) == 0) it = samples_.erase(it);
        else { it->y -= 1.0f; ++it; }
    }
}

wxImage TimeHeatMap::Render() const {
    if (mode_ == TimeFrameMode::Unlimited) {
        // 高さを超えるy座標も含む全サンプルのうち、
        // 最後のheight_範囲だけを描画
        float y_min = std::max(0.0f, float(cur_frame_ - height_ + 1));
        float y_max = float(cur_frame_);

        // 一時HeatMapを構築
        HeatMap temp(width_, height_,
            x_min_, x_max_,
            y_min, y_max,
            0, flags_);

        {
            std::lock_guard<std::mutex> lk(mtx_);
            for (const auto& s : samples_) {
                if (s.y >= y_min && s.y <= y_max)
                    temp.AddData(s.x, s.y, s.value);
            }
        }

        return temp.Render();
    }

    // Overwrite / Scroll は通常描画
    return HeatMap::Render();
}

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

wxImage Histgram::Render() const {
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
TimeHistogram::TimeHistogram(int bins, int max_frames, float min_val, float max_val, size_t max_points)
    : bins_(bins), max_frames_(max_frames),
    min_val_(min_val), max_val_(max_val),
    max_points_per_frame_(max_points) {
    frames_.emplace_back();
}

void TimeHistogram::AddData(float value) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (frames_.empty()) frames_.emplace_back();
    auto& cur = frames_.back();
    cur.values.push_back(value);
    if (max_points_per_frame_ > 0 && cur.values.size() > max_points_per_frame_)
        cur.values.pop_front();
}

void TimeHistogram::NextFrame() {
    std::lock_guard<std::mutex> lock(mtx_);
    frames_.emplace_back();
    if (frames_.size() > static_cast<size_t>(max_frames_))
        frames_.pop_front();
}

void TimeHistogram::Reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    frames_.clear();
    frames_.emplace_back();
}

wxImage TimeHistogram::Render() const {
    std::lock_guard<std::mutex> lock(mtx_);
    wxImage img(bins_, static_cast<int>(frames_.size()));
    unsigned char* data = img.GetData();
    std::vector<float> hist(bins_, 0.0f);

    for (size_t t = 0; t < frames_.size(); ++t) {
        std::fill(hist.begin(), hist.end(), 0.0f);
        for (float v : frames_[t].values) {
            int ix = static_cast<int>((v - min_val_) / (max_val_ - min_val_) * bins_);
            if (ix >= 0 && ix < bins_) hist[ix] += 1.0f;
        }
        float maxv = *std::max_element(hist.begin(), hist.end());
        for (int x = 0; x < bins_; ++x) {
            float norm = hist[x] / std::max(1e-6f, maxv);
            unsigned char r, g, b;
            ValueToRGB_Jet(norm, r, g, b);
            int idx = ((frames_.size() - 1 - t) * bins_ + x) * 3;
            data[idx] = r; data[idx + 1] = g; data[idx + 2] = b;
        }
    }
    return img;
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

wxImage SweepedHeatMap::Render() const {
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
    // ③ 時系列ヒストグラム — TD誤差分布の変化（TimeHistogram）
    // ============================================================
    {
        anet::TimeHistogram hist_series(100, 100, -3.0f, 3.0f, 256); // bins, frames
        for (int t = 0; t < steps; ++t) {
            hist_series.AddData(td_errors[t]);  // 現在の値をヒストグラムへ
            if (t % 5 == 0) hist_series.NextFrame(); // 5 stepごとに1フレーム
        }
        hist_series.SavePng("3_out_timehist_td.png");
    }

    // ============================================================
    // ④ 時系列ヒートマップ — x=value, y=time, 色=平均TD誤差値
    // ============================================================
    {
        const int width = 100;
        const int frames = 100;
        anet::TimeHistogram value_series(width, frames, -3.0f, 3.0f, 128);
        for (int t = 0; t < steps; ++t) {
            value_series.AddData(td_errors[t]);
            if (t % 5 == 0) value_series.NextFrame();
        }
        value_series.SavePng("4_out_timeheat_td.png");
    }
}
