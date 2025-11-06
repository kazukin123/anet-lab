#include "anet/heat_map.hpp"
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <wx/log.h>

using namespace anet;

// ============================================================
// Util: 値 → Jet RGB
// ============================================================
static void ValueToRGB_Jet(float norm, unsigned char& r, unsigned char& g, unsigned char& b) {
	norm = std::clamp(norm, 0.0f, 1.0f);
	float rf = 0.0f, gf = 0.0f, bf = 0.0f;
	if (norm < 0.125f) {
		rf = 0.0f; gf = 0.0f; bf = 0.5f + norm * 4.0f;
	}
	else if (norm < 0.375f) {
		rf = 0.0f; gf = (norm - 0.125f) * 4.0f; bf = 1.0f;
	}
	else if (norm < 0.625f) {
		rf = (norm - 0.375f) * 4.0f; gf = 1.0f; bf = 1.0f - (norm - 0.375f) * 4.0f;
	}
	else if (norm < 0.875f) {
		rf = 1.0f; gf = 1.0f - (norm - 0.625f) * 4.0f; bf = 0.0f;
	}
	else { rf = 1.0f - (norm - 0.875f) * 4.0f; gf = 0.0f; bf = 0.0f; }
	r = static_cast<unsigned char>(rf * 255);
	g = static_cast<unsigned char>(gf * 255);
	b = static_cast<unsigned char>(bf * 255);
}

static inline float safe_log1p_abs(float v) {
	return std::log1p(std::fabs(v));
}

// ============================================================
// ImageSource
// ============================================================
wxImage ImageSource::Render(int width, int height) const {
	wxImage src = RenderRaw();
	if (width < 0 && height < 0) return src;
	if (width < 0) width = src.GetWidth();
	if (height < 0) height = src.GetHeight();
	return src.Scale(width, height, wxIMAGE_QUALITY_HIGH);
}

void ImageSource::SavePng(const std::string& filename, int width, int height) const {
	std::filesystem::path path(filename);
	if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
	wxImage img = Render(width, height);
	if (!img.IsOk()) { wxLogError("Render() returned invalid wxImage"); return; }
	img.SaveFile(filename, wxBITMAP_TYPE_PNG);
}

// ============================================================
// HeatMap
// ============================================================
HeatMap::HeatMap(int width, int height, float x_min, float x_max, float y_min, float y_max,
	size_t max_points, uint32_t flags)
	: width_(width),
	height_(height),
	x_min_(x_min),
	x_max_(x_max),
	y_min_(y_min),
	y_max_(y_max),
	value_min_(std::numeric_limits<float>::max()),
	value_max_(-std::numeric_limits<float>::max()),
	max_points_(max_points),
	flags_(flags) {
}

void HeatMap::AddData(float x, float y, float value) {
	samples_.push_back({ x, y, value });
	if (max_points_ > 0 && samples_.size() > max_points_) samples_.pop_front();
	UpdateMinMax_(value);
}

void HeatMap::Reset() {
	samples_.clear();
	value_min_ = std::numeric_limits<float>::max();
	value_max_ = -std::numeric_limits<float>::max();
}

wxImage HeatMap::RenderRaw() const {
	// --- スナップショット取得 ---
	std::vector<Sample> snapshot(samples_.begin(), samples_.end());
	if (snapshot.empty()) {
		wxImage empty(1, 1);
		empty.SetData(new unsigned char[3] {0, 0, 0});
		return empty;
	}

	const int W = width_;
	const int H = height_;
	std::vector<float> buf(W * H, 0.0f);
	std::vector<int> cnt(W * H, 0);

	// --- 値レンジ決定（AutoNormValue有効時はゼロ値除外） ---
	float vmin = value_min_;
	float vmax = value_max_;
	if (flags_ & HM_AutoNormValue) {
		bool has_nonzero = false;
		vmin = std::numeric_limits<float>::max();
		vmax = -vmin;
		for (const auto& s : snapshot) {
			if (std::fabs(s.value) < 1e-8f) continue;  // 黒帯や未観測を除外
			has_nonzero = true;
			vmin = std::min(vmin, s.value);
			vmax = std::max(vmax, s.value);
		}
		if (!has_nonzero) { vmin = -1.0f; vmax = 1.0f; }
		if (std::fabs(vmax - vmin) < 1e-6f) vmax = vmin + 1e-6f;
	}

	// --- サンプル配置（サブピクセル補間＋ゼロスキップ） ---
	for (const auto& s : snapshot) {
		if (std::fabs(s.value) < 1e-8f) continue;  // ゼロ値は描画に寄与しない
		float fx = (s.x - x_min_) / (x_max_ - x_min_) * W;
		int ix = static_cast<int>(std::floor(fx));
		float frac = fx - ix;
		int iy = static_cast<int>((s.y - y_min_) / (y_max_ - y_min_) * H);
		if (iy < 0 || iy >= H) continue;

		if (ix >= 0 && ix < W) {
			int idx = iy * W + ix;
			buf[idx] += (1.0f - frac) * s.value;
			cnt[idx]++;
		}
		if (ix + 1 >= 0 && ix + 1 < W) {
			int idx2 = iy * W + (ix + 1);
			buf[idx2] += frac * s.value;
			cnt[idx2]++;
		}
	}

	// --- 平均化（MeanMode有効時） ---
	if (flags_ & HM_MeanMode) {
		for (size_t i = 0; i < buf.size(); ++i)
			if (cnt[i] > 0) buf[i] /= cnt[i];
	}

	// --- 画像生成 ---
	wxImage img(W, H);
	img.SetData(new unsigned char[W * H * 3]);
	unsigned char* data = img.GetData();

	for (int y = 0; y < H; ++y) {
		for (int x = 0; x < W; ++x) {
			int idx = y * W + x;
			unsigned char r = 0, g = 0, b = 0;
			float v = buf[idx];

			if (cnt[idx] == 0 || std::fabs(v) < 1e-8f) {
				// 未観測またはゼロ値 → 黒
				r = g = b = 0;
			}
			else {
				if (flags_ & HM_LogScaleValue)
					v = std::copysign(std::log1p(std::fabs(v)), v);
				float n = (v - vmin) / std::max(vmax - vmin, 1e-6f);
				ValueToRGB_Jet(n, r, g, b);
			}

			int di;
			if (flags_ & HM_FlipY)
				di = (y * W + x) * 3;
			else
				di = ((H - 1 - y) * W + x) * 3;

			data[di] = r;
			data[di + 1] = g;
			data[di + 2] = b;
		}
	}

	return img;
}

// ============================================================
// TimeHeatMap（Unlimitedモードもサブピクセル補間を適用）
// ============================================================
TimeHeatMap::TimeHeatMap(int width_frames, int height_bins, float in_min, float in_max,
	uint32_t flags, size_t max_points, TimeFrameMode mode)
	: HeatMap(width_frames, height_bins, 0.0f, float(width_frames - 1),
		in_min, in_max, 0, flags),
	mode_(mode),
	cur_frame_(0),
	total_frames_(0),
	max_frames_(width_frames * 4)
{
	//HeatMap::HeatMap(int width, int height, float x_min, float x_max, float y_min, float y_max,
	//    size_t max_points, uint32_t flags)
}

void TimeHeatMap::AddData(float in, float out) {
	HeatMap::AddData(static_cast<float>(cur_frame_), in, out);
}

void TimeHeatMap::Scroll_() {
	for (auto& s : samples_) s.x -= 1.0f;
	EraseCol_(0);
}

void TimeHeatMap::EraseCol_(int x_col) {
	auto it = samples_.begin();
	while (it != samples_.end()) {
		if (std::lround(it->x) == x_col) it = samples_.erase(it); else ++it;
	}
}

void TimeHeatMap::NextFrame() {
	if (mode_ == TimeFrameMode::Scroll) {
		Scroll_(); cur_frame_ = width_ - 1;
	}
	else if (mode_ == TimeFrameMode::Overwrite) {
		cur_frame_ = (cur_frame_ + 1) % width_;
	}
	else { // Unlimited
		cur_frame_++; total_frames_++;
	}
}

void TimeHeatMap::Reset() {
	HeatMap::Reset();
	cur_frame_ = 0;
	total_frames_ = 0;
}

wxImage TimeHeatMap::RenderRaw() const {
	if (mode_ != TimeFrameMode::Unlimited)
		return HeatMap::RenderRaw();

	std::vector<Sample> snapshot(samples_.begin(), samples_.end());
	if (snapshot.empty()) {
		wxImage empty(1, 1);
		empty.SetData(new unsigned char[3] {0, 0, 0});
		return empty;
	}

	const int N = static_cast<int>(total_frames_ > 0 ? total_frames_ : snapshot.back().x + 1);
	const int W = width_;
	const int H = height_;

	// --- X軸スケーリング（仕様：左端が最初・右端が最後、N<Wなら拡大・N>Wなら縮小）---
	const float scale_x = (N <= 1) ? 0.0f : static_cast<float>(W - 1) / static_cast<float>(N - 1);

	std::vector<float> buf(W * H, 0.0f);
	std::vector<int> cnt(W * H, 0);

	// --- 値レンジの決定（AutoNormValue有効時は非ゼロ値のみ対象）---
	float vmin = value_min_, vmax = value_max_;
	if (flags_ & HM_AutoNormValue) {
		bool has_nonzero = false;
		vmin = std::numeric_limits<float>::max();
		vmax = -vmin;
		for (const auto& s : snapshot) {
			if (std::fabs(s.value) < 1e-8f) continue;  // ゼロ値除外
			has_nonzero = true;
			vmin = std::min(vmin, s.value);
			vmax = std::max(vmax, s.value);
		}
		if (!has_nonzero) { vmin = -1.0f; vmax = 1.0f; }
		if (std::fabs(vmax - vmin) < 1e-6f) vmax = vmin + 1e-6f;
	}

	// --- サンプルを描画バッファへ反映（Xサブピクセル補間＋ゼロスキップ）---
	for (const auto& s : snapshot) {
		if (std::fabs(s.value) < 1e-8f) continue;  // ゼロ値は描画しない（時系列保持のみ）
		float fx = s.x * scale_x;
		int ix = static_cast<int>(std::floor(fx));
		float frac = fx - ix;
		int iy = static_cast<int>((s.y - y_min_) / (y_max_ - y_min_) * H);
		if (iy < 0 || iy >= H) continue;

		if (ix >= 0 && ix < W) {
			int idx = iy * W + ix;
			buf[idx] += (1.0f - frac) * s.value;
			cnt[idx]++;
		}
		if (ix + 1 >= 0 && ix + 1 < W) {
			int idx2 = iy * W + (ix + 1);
			buf[idx2] += frac * s.value;
			cnt[idx2]++;
		}
	}

	// --- 平均化（Unlimitedでは常にMean扱い）---
	for (size_t i = 0; i < buf.size(); ++i)
		if (cnt[i] > 0) buf[i] /= cnt[i];

	// --- イメージ化 ---
	wxImage img(W, H);
	img.SetData(new unsigned char[W * H * 3]);
	unsigned char* data = img.GetData();

	for (int y = 0; y < H; ++y) {
		for (int x = 0; x < W; ++x) {
			int idx = y * W + x;
			unsigned char r = 0, g = 0, b = 0;
			float v = buf[idx];

			if (cnt[idx] == 0) {
				r = g = b = 0;  // 未観測セルは黒
			}
			else {
				if (flags_ & HM_LogScaleValue) v = std::copysign(std::log1p(std::fabs(v)), v);
				float n = (v - vmin) / std::max(vmax - vmin, 1e-6f);
				ValueToRGB_Jet(n, r, g, b);
			}

			int di;
			if (flags_ & HM_FlipY)
				di = (y * W + x) * 3;
			else
				di = ((H - 1 - y) * W + x) * 3;

			data[di] = r;
			data[di + 1] = g;
			data[di + 2] = b;
		}
	}

	return img;
}

// ============================================================
// Histgram
// ============================================================
Histgram::Histgram(int bins, float min_val, float max_val, int width, int height)
	: bins_(bins),
	width_(width),
	height_(height),
	min_val_(min_val),
	max_val_(max_val),
	counts_(bins, 0) {
}

void Histgram::AddData(float value) {
	std::lock_guard<std::mutex> lock(mtx_);
	int idx = static_cast<int>((value - min_val_) / (max_val_ - min_val_) * bins_);
	if (idx >= 0 && idx < bins_) counts_[idx]++;
}

void Histgram::Reset() { std::fill(counts_.begin(), counts_.end(), 0); }

wxImage Histgram::RenderRaw() const {
	std::lock_guard<std::mutex> lock(mtx_);
	wxImage img(width_, height_);
	img.SetData(new unsigned char[width_ * height_ * 3]);
	std::fill_n(img.GetData(), width_ * height_ * 3, 0);
	int maxc = *std::max_element(counts_.begin(), counts_.end());
	if (maxc <= 0) return img;
	unsigned char* d = img.GetData();
	for (int i = 0; i < bins_; ++i) {
		int h = static_cast<int>((float)counts_[i] / std::max(1, maxc) * height_);
		h = std::min(h, height_);
		int x0 = (i * width_) / bins_;
		int x1 = ((i + 1) * width_) / bins_;
		for (int x = x0; x < x1; ++x) {
			for (int y = 0; y < h; ++y) {
				int py = height_ - 1 - y;
				int di = (py * width_ + x) * 3;
				d[di] = d[di + 1] = d[di + 2] = 255;
			}
		}
	}
	return img;
}

// ============================================================
// TimeHistogram
//   - thm_ は縦=bin(0..bins-1) 固定。値→bin の写像でレンジ/対数軸を実現。
// ============================================================
static inline bool is_nan(float v) { return std::isnan(v); }

TimeHistogram::TimeHistogram(int bins, int max_frames, TimeFrameMode mode,
	uint32_t flags, float base_min, float base_max, float alpha)
	: thm_(max_frames, bins, 0.0f, float(bins - 1), flags, 0, mode),
	bins_(bins),
	alpha_(alpha),
	flags_(flags),
	base_min_(base_min),
	base_max_(base_max),
	min_val_(std::numeric_limits<float>::quiet_NaN()),
	max_val_(std::numeric_limits<float>::quiet_NaN()),
	buffer_(bins, 0.0f) {
}

int TimeHistogram::MapToBinLinear_(float v) const {
	float a = min_val_, b = max_val_;
	float t = (v - a) / std::max(b - a, 1e-12f);
	int ix = static_cast<int>(t * bins_);
	return std::clamp(ix, 0, bins_ - 1);
}

int TimeHistogram::MapToBinLogAxis_(float v) const {
	// 符号を保持した log1p(|v|) を、min/max 側の log スケールで正規化
	float a = min_val_, b = max_val_;
	bool has_neg = a < 0.0f, has_pos = b > 0.0f;
	if (!has_neg && !has_pos) { return 0; }

	if (v >= 0.0f) {
		float vmax = std::max(b, 1e-12f);
		float t = safe_log1p_abs(v) / std::max(safe_log1p_abs(vmax), 1e-12f);
		int iy = static_cast<int>(t * (has_neg ? (bins_ / 2.0f - 1.0f) : (bins_ - 1.0f)));
		// 正だけなら最上段まで、正負混在なら中央より上だけを使用
		if (has_neg) iy += bins_ / 2;
		return std::clamp(iy, 0, bins_ - 1);
	}
	else {
		float vmin = std::min(a, -1e-12f);
		float t = safe_log1p_abs(-v) / std::max(safe_log1p_abs(-vmin), 1e-12f);
		int iy = static_cast<int>(t * (has_pos ? (bins_ / 2.0f - 1.0f) : (bins_ - 1.0f)));
		// 負だけなら最下段まで、正負混在なら中央より下だけを使用
		if (has_pos) iy = (bins_ / 2 - 1) - iy;
		else iy = (bins_ - 1) - iy;
		return std::clamp(iy, 0, bins_ - 1);
	}
}

int TimeHistogram::MapToBin_(float v) const {
	if (flags_ & HM_LogScaleAxis) return MapToBinLogAxis_(v);
	return MapToBinLinear_(v);
}

void TimeHistogram::AddBatch(const std::vector<float>& values) {
	if (values.empty()) return;

	// --- レンジ更新（軽量）：min/max のみ、NaN側だけEMA更新 ---
	float vmin = *std::min_element(values.begin(), values.end());
	float vmax = *std::max_element(values.begin(), values.end());

	if (!(flags_ & HM_AutoScaleAxis)) {
		// 固定モード：base_min_/base_max_ が有効ならそれを使用
		if (!is_nan(base_min_) && !is_nan(base_max_)) { min_val_ = base_min_; max_val_ = base_max_; }
		else {
			// 安全のため、初回は観測から決定（以降固定）
			if (is_nan(min_val_)) min_val_ = vmin;
			if (is_nan(max_val_)) max_val_ = vmax;
		}
	}
	else {
		// 自動：ベース指定を優先し、NaN側のみEMA追従
		const float a = std::clamp(alpha_, 0.001f, 1.0f);
		if (!is_nan(base_min_)) min_val_ = base_min_;
		else if (is_nan(min_val_)) min_val_ = vmin;
		else min_val_ = (1.0f - a) * min_val_ + a * vmin;

		if (!is_nan(base_max_)) max_val_ = base_max_;
		else if (is_nan(max_val_)) max_val_ = vmax;
		else max_val_ = (1.0f - a) * max_val_ + a * vmax;

		if (max_val_ - min_val_ < 1e-9f) { max_val_ = min_val_ + 1e-3f; }
	}

	// --- カウント蓄積 ---
	for (float v : values) {
		int ix = MapToBin_(v);
		buffer_[ix] += 1.0f;
	}
}

void TimeHistogram::NextFrame() {
	// 強度正規化（色方向）
	if (flags_ & HM_AutoNormValue) {
		float maxv = *std::max_element(buffer_.begin(), buffer_.end());
		if (maxv > 0.0f) for (auto& v : buffer_) v /= maxv;
	}
	// bin i は下から上に行くほど大きい値にしたいので反転して送る
	for (int i = 0; i < bins_; ++i) {
		int inv_i = bins_ - 1 - i;
		thm_.AddData(static_cast<float>(inv_i), buffer_[i]);
	}
	std::fill(buffer_.begin(), buffer_.end(), 0.0f);
	thm_.NextFrame();
}

void TimeHistogram::Reset() {
	thm_.Reset();
	std::fill(buffer_.begin(), buffer_.end(), 0.0f);
	min_val_ = std::numeric_limits<float>::quiet_NaN();
	max_val_ = std::numeric_limits<float>::quiet_NaN();
}

wxImage TimeHistogram::RenderRaw() const {
	wxImage img = thm_.RenderRaw();
	if (!img.IsOk()) return img;

	// ゼロライン描画
	if ((flags_ & HM_ShowZeroLine) && min_val_ < 0.0f && max_val_ > 0.0f) {
		int height = img.GetHeight(), width = img.GetWidth();
		int zero_bin = 0;
		if (flags_ & HM_LogScaleAxis) {
			// 中央をゼロに置く設計
			zero_bin = bins_ / 2;
		}
		else {
			// 線形スケールでのゼロ位置
			float t = (0.0f - min_val_) / std::max(max_val_ - min_val_, 1e-12f);
			zero_bin = static_cast<int>((1.0f - t) * bins_);
			zero_bin = std::clamp(zero_bin, 0, bins_ - 1);
		}
		int y = static_cast<int>((float)zero_bin / bins_ * height);
		y = std::clamp(y, 0, height - 1);
		unsigned char* d = img.GetData();
		for (int x = 0; x < width; ++x) {
			int di = (y * width + x) * 3; d[di] = 255; d[di + 1] = 255; d[di + 2] = 255;
		}
	}
	return img;
}

// ============================================================
// SweepedHeatMap
// ============================================================
SweepedHeatMap::SweepedHeatMap(int width, int height, float x_min, float x_max,
	float y_min, float y_max)
	: width_(width),
	height_(height),
	x_min_(x_min),
	x_max_(x_max),
	y_min_(y_min),
	y_max_(y_max),
	values_(width* height, 0.0f),
	value_min_(0.0f),
	value_max_(1.0f) {
}

void SweepedHeatMap::Evaluate(const std::function<float(float, float)>& func) {
	value_min_ = std::numeric_limits<float>::max();
	value_max_ = -std::numeric_limits<float>::max();
	for (int j = 0; j < height_; ++j) {
		for (int i = 0; i < width_; ++i) {
			float x = x_min_ + (x_max_ - x_min_) * i / std::max(1, width_ - 1);
			float y = y_min_ + (y_max_ - y_min_) * j / std::max(1, height_ - 1);
			float v = func(x, y);
			values_[j * width_ + i] = v;
			value_min_ = std::min(value_min_, v);
			value_max_ = std::max(value_max_, v);
		}
	}
}

wxImage SweepedHeatMap::RenderRaw()const {
	wxImage img(width_, height_);
	img.SetData(new unsigned char[width_ * height_ * 3]);
	unsigned char* d = img.GetData();
	float range = std::max(value_max_ - value_min_, 1e-6f);
	for (int y = 0; y < height_; ++y) {
		for (int x = 0; x < width_; ++x) {
			float v = (values_[y * width_ + x] - value_min_) / range;
			unsigned char r, g, b; ValueToRGB_Jet(v, r, g, b);
			int di = ((height_ - 1 - y) * width_ + x) * 3;
			d[di] = r; d[di + 1] = g; d[di + 2] = b;
		}
	}
	return img;
}

void SweepedHeatMap::Reset() {
	std::fill(values_.begin(), values_.end(), 0.0f);
	value_min_ = 0.0f; value_max_ = 1.0f;
}

SweepedHeatMap SweepedHeatMap::EvaluateTensorFunction(
	int width, int height, float x_min, float x_max, float y_min, float y_max,
	const torch::Device& device,
	const std::function<torch::Tensor(const torch::Tensor&)>& forward,
	const std::function<torch::Tensor(const torch::Tensor&)>& value_extractor) {

	SweepedHeatMap map(width, height, x_min, x_max, y_min, y_max);
	torch::NoGradGuard ng;
	auto xs = torch::linspace(x_min, x_max, width, device);
	auto ys = torch::linspace(y_min, y_max, height, device);

	map.value_min_ = std::numeric_limits<float>::max();
	map.value_max_ = -std::numeric_limits<float>::max();

	for (int j = 0; j < height; ++j) {
		float yv = ys[j].item<float>();
		auto grid = torch::stack({ xs, torch::full_like(xs, yv) }, 1); // (W,2)
		auto out = forward(grid);                                     // 任意形状
		auto val = value_extractor(out).to(torch::kCPU).contiguous(); // (W,)
		for (int i = 0; i < width; ++i) {
			float v = val[i].item<float>();
			map.values_[j * width + i] = v;
			map.value_min_ = std::min(map.value_min_, v);
			map.value_max_ = std::max(map.value_max_, v);
		}
	}
	return map;
}
