#include "anet/heat_map.hpp"
#include <cmath>
#include <algorithm>
#include <filesystem>
#include "anet/common.hpp"
#include "anet/log.hpp"
#include "anet/profile.hpp"

using namespace anet;
namespace LOG = anet::log;

const unsigned char BACKGROUND_LEVEL = 10;

// ============================================================
// Util: 値 → Jet RGB
// ============================================================
static void ValueToRGB_Jet(float norm, unsigned char& r, unsigned char& g, unsigned char& b)
{
	//ProfileRange r("ValueToRGB_Jet");

	norm = std::clamp(norm, 0.0f, 1.0f);
	float rf = 0.0f, gf = 0.0f, bf = 0.0f;
	if (norm < 0.125f) {
		rf = 0.0f; gf = 0.0f; bf = 0.5f + norm * 4.0f;
	} else if (norm < 0.375f) {
		rf = 0.0f; gf = (norm - 0.125f) * 4.0f; bf = 1.0f;
	} else if (norm < 0.625f) {
		rf = (norm - 0.375f) * 4.0f; gf = 1.0f; bf = 1.0f - (norm - 0.375f) * 4.0f;
	} else if (norm < 0.875f) {
		rf = 1.0f; gf = 1.0f - (norm - 0.625f) * 4.0f; bf = 0.0f;
	} else { rf = 1.0f - (norm - 0.875f) * 4.0f; gf = 0.0f; bf = 0.0f; }
	r = static_cast<unsigned char>(rf * 255);
	g = static_cast<unsigned char>(gf * 255);
	b = static_cast<unsigned char>(bf * 255);
}

static inline float safe_log1p_abs(float v) {
	return std::log1p(std::fabs(v));
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
	flags_(flags)
{
	is_fixed_ = (max_points > 0);
	if (is_fixed_) {
		buf_.resize(max_points);
	}
	//Reset();

	ANET_CHECK(width_ > 0);
	ANET_CHECK(height_ > 0);
}

void HeatMap::AddData(float x, float y, float value)
{
	ANET_PROFILE_FUNC();

	if (is_fixed_) {
		buf_[head_] = { x, y, value };
		head_ = (head_ + 1) % max_points_;
		if (size_ < max_points_) size_++;
	} else {
		buf_.push_back({ x, y, value });
		size_++;
	}

	UpdateMinMax_(value);
}

/// @brief 複数のデータを一括追加（AddDataの高速版）
/// SoA (xv, yv, vv) を受け取り、内部でAoS (buf_) に変換して格納する
void HeatMap::AddDataBatch(const std::vector<float>& xv, const std::vector<float>& yv, const std::vector<float>& vv)
{
	ANET_PROFILE_FUNC();

	// サイズチェック
	size_t n = xv.size();
	if (n == 0) return;
	// (呼び出し元でチェック済みなら省略可だが安全のため)
	if (yv.size() != n || vv.size() != n) return;

	std::lock_guard<std::mutex> lock(mtx_); // ロックは1回だけ

	// --- Min/Max の一括更新 (SIMD化されやすいループ) ---
	float local_min = std::numeric_limits<float>::max();
	float local_max = -std::numeric_limits<float>::max();

	// 値の範囲チェック用（AutoNormValueなどで使う場合）
	for (float v : vv) {
		if (v < local_min) local_min = v;
		if (v > local_max) local_max = v;
	}

	// グローバルなMin/Maxを更新
	if (local_min < value_min_) value_min_ = local_min;
	if (local_max > value_max_) value_max_ = local_max;

	// --- バッファへのデータ転送 ---
	if (!is_fixed_) {
		// --- 可変長モード (単純追加) ---
		// 必要なメモリを一度に確保
		buf_.reserve(size_ + n);

		for (size_t i = 0; i < n; ++i) {
			buf_.push_back({ xv[i], yv[i], vv[i] });
		}
		size_ += n;
	} else {
		// --- 固定長リングバッファモード ---
		// 剰余演算(%)をループ内でやると遅いので、2回のコピーに分ける

		size_t cap = buf_.size(); // max_points_ と同義と仮定
		if (cap == 0) return;

		size_t current_idx = head_;

		for (size_t i = 0; i < n; ++i) {
			buf_[current_idx] = { xv[i], yv[i], vv[i] };

			// インクリメントとラップアラウンド
			current_idx++;
			if (current_idx == cap) current_idx = 0;
		}

		// 状態更新
		head_ = current_idx;
		size_ = std::min(size_ + n, cap);
	}
}

/// @brief グリッドの値を一括設定（高速パス）
/// values は row-major (y * W + x)
/// サイズ = W * H
void HeatMap::SetGridValues(const float* values, int width, int height)
{
	ANET_PROFILE_FUNC();

	ANET_ASSERT(width == width_);
	ANET_ASSERT(height == height_);

	const size_t N = width * height;
	is_fixed_ = true;
	max_points_ = N;
	buf_.resize(N);

	float min_v = +std::numeric_limits<float>::infinity();
	float max_v = -std::numeric_limits<float>::infinity();

	for (size_t i = 0; i < N; ++i) {
		float v = values[i];
		buf_[i] = { float(i % width), float(i / width), v };
		min_v = std::min(min_v, v);
		max_v = std::max(max_v, v);
	}

	head_ = 0;
	size_ = N;
	value_min_ = min_v;
	value_max_ = max_v;
}

void HeatMap::Reset()
{
	if (!is_fixed_) {
		buf_.clear();
	}
	size_ = 0;
	head_ = 0;
	value_min_ = std::numeric_limits<float>::max();
	value_max_ = -std::numeric_limits<float>::max();
}

wxImage HeatMap::RenderRaw() const
{
	ANET_PROFILE_FUNC();

	std::lock_guard<std::mutex> lock(mtx_);

	if (size_ == 0) {
		wxImage empty(width_, height_);
		size_t buf_size = width_ * height_ * 3;
		unsigned char* d = (unsigned char*)malloc(buf_size);
		memset(d, BACKGROUND_LEVEL, buf_size);
		empty.SetData(d);
		return empty;
	}

	ANET_PROFILE_SCOPE(prepare);

	const int W = width_;
	const int H = height_;
	const size_t pixel_count = static_cast<size_t>(W) * H;

	// バッファ再利用 (サイズ変更時のみ確保発生)
	if (work_buf_.size() != pixel_count) {
		work_buf_.resize(pixel_count);
		work_cnt_.resize(pixel_count);
	}
	//std::fill(work_buf_.begin(), work_buf_.end(), 0.0f);
	//std::fill(work_cnt_.begin(), work_cnt_.end(), 0);
	std::memset(work_buf_.data(), 0, work_buf_.size() * sizeof(float));
	std::memset(work_cnt_.data(), 0, work_cnt_.size() * sizeof(int));

	// --- 座標変換係数の事前計算 (除算を排除) ---
	const float range_x = x_max_ - x_min_;
	const float range_y = y_max_ - y_min_;
	const float scale_x = (range_x > 1e-6f) ? (static_cast<float>(W) / range_x) : 0.0f;
	const float scale_y = (range_y > 1e-6f) ? (static_cast<float>(H) / range_y) : 0.0f;
	const float offset_x = -x_min_ * scale_x;
	const float offset_y = -y_min_ * scale_y;

	// --- サンプル走査ヘルパー ---
	auto traverse_samples = [&](auto process_func) {
		if (!is_fixed_) {
			for (size_t i = 0; i < size_; ++i) process_func(buf_[i]);
		} else {
			const size_t cap = buf_.size();
			if (cap == 0) return;
			size_t start = (head_ + cap - size_) % cap;
			size_t run1 = std::min(size_, cap - start);
			size_t run2 = size_ - run1;
			for (size_t i = 0; i < run1; ++i) process_func(buf_[start + i]);
			if (run2 > 0) {
				for (size_t i = 0; i < run2; ++i) process_func(buf_[i]);
			}
		}
		};

	// --- 値レンジ決定 ---
	ANET_PROFILE_SCOPE_NEXT(auto_norm_value, prepare);
	float vmin = value_min_;
	float vmax = value_max_;

	if (flags_ & HM_AutoNormValue) {
		float l_min = std::numeric_limits<float>::max();
		float l_max = -std::numeric_limits<float>::max();
		bool has_nonzero = false;
		traverse_samples([&](const Sample& s) {
			if (std::fabs(s.value) < 1e-8f) return;
			has_nonzero = true;
			if (s.value < l_min) l_min = s.value;
			if (s.value > l_max) l_max = s.value;
			});
		if (has_nonzero) {
			vmin = l_min; vmax = l_max;
			if (std::fabs(vmax - vmin) < 1e-6f) vmax = vmin + 1e-6f;
		} else {
			vmin = -1.0f; vmax = 1.0f;
		}
	}

	// --- サンプル配置 ---
	ANET_PROFILE_SCOPE_NEXT(normalize, auto_norm_value);

	float* p_buf = work_buf_.data();
	int* p_cnt = work_cnt_.data();

	traverse_samples([&](const Sample& s) {
		if (std::fabs(s.value) < 1e-8f) return;

		float fx = s.x * scale_x + offset_x;
		float fy = s.y * scale_y + offset_y;

		if (fy < 0.0f || fy >= static_cast<float>(H)) return;

		int ix = static_cast<int>(fx);
		int iy = static_cast<int>(fy);
		float frac = fx - static_cast<float>(ix);
		int base_idx = iy * W;

		// 最適化: 端以外の大部分のケースを高速パスへ
		// (ix >= 0 && ix < W-1)
		if (static_cast<unsigned int>(ix) < static_cast<unsigned int>(W - 1)) {
			int idx = base_idx + ix;
			p_buf[idx] += (1.0f - frac) * s.value;
			p_cnt[idx]++;

			p_buf[idx + 1] += frac * s.value;
			p_cnt[idx + 1]++;
		} else {
			// 境界チェックありパス
			if (ix >= 0 && ix < W) {
				int idx = base_idx + ix;
				p_buf[idx] += (1.0f - frac) * s.value;
				p_cnt[idx]++;
			}
			if (ix + 1 >= 0 && ix + 1 < W) {
				int idx2 = base_idx + ix + 1;
				p_buf[idx2] += frac * s.value;
				p_cnt[idx2]++;
			}
		}
		});

	// --- 平均化 ---
	if (flags_ & HM_MeanMode) {
		for (size_t i = 0; i < pixel_count; ++i) {
			if (p_cnt[i] > 0) p_buf[i] /= static_cast<float>(p_cnt[i]);
		}
	}

	// --- 画像生成 (ポインタアクセスによる高速化) ---
	ANET_PROFILE_SCOPE_NEXT(draw, normalize);
	wxImage img(W, H);
	unsigned char* data = (unsigned char*)malloc(W * H * 3);
	img.SetData(data);

	const float denom_inv = 1.0f / std::max(vmax - vmin, 1e-6f);
	const bool is_log = (flags_ & HM_LogScaleValue);
	const bool flip_y = (flags_ & HM_FlipY);

	// ★OpenMP有効化: 画像サイズが大きいほど効果大
#pragma omp parallel for schedule(static)
	for (int y = 0; y < H; ++y) {
		int out_y = flip_y ? y : (H - 1 - y);
		unsigned char* row_ptr = data + (out_y * W * 3);
		int src_idx = y * W;

		// 行単位でポインタを進める
		for (int x = 0; x < W; ++x, ++src_idx) {
			float v = p_buf[src_idx];
			int c = p_cnt[src_idx]; // int load

			unsigned char r, g, b;
			if (c == 0 || std::fabs(v) < 1e-8f) {
				r = g = b = BACKGROUND_LEVEL;
			} else {
				if (is_log) {
					v = std::copysign(std::log1p(std::fabs(v)), v);
				}
				float n = (v - vmin) * denom_inv;
				ValueToRGB_Jet(n, r, g, b);
			}

			*row_ptr++ = r;
			*row_ptr++ = g;
			*row_ptr++ = b;
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
	float dx = (x_max_ - x_min_) / float(total_frames_);
	x_min_ += dx;
	x_max_ += dx;
}

void TimeHeatMap::EraseCol_(int x_col) {
	//auto it = samples_.begin();
	//while (it != samples_.end()) {
	//	if (std::lround(it->x) == x_col) it = samples_.erase(it); else ++it;
	//}
}

void TimeHeatMap::NextFrame() {
	if (mode_ == TimeFrameMode::Scroll) {
		Scroll_(); cur_frame_ = width_ - 1;
	} else if (mode_ == TimeFrameMode::Overwrite) {
		cur_frame_ = (cur_frame_ + 1) % width_;
	} else { // Scale
		cur_frame_++; total_frames_++;
		x_max_ = float(cur_frame_);
	}
}

void TimeHeatMap::Reset() {
	HeatMap::Reset();
	cur_frame_ = 0;
	total_frames_ = 0;
}

wxImage TimeHeatMap::RenderRaw() const
{
	ANET_PROFILE_FUNC();

	if (mode_ != TimeFrameMode::Scale)
		return HeatMap::RenderRaw();

	if (size_ == 0) {
		wxImage empty(1, 1);
		empty.SetData(new unsigned char[3] {0, 0, 0});
		return empty;
	}

	// --- スナップショット取得 ---
	std::vector<Sample> snapshot;
	snapshot.reserve(size_);
	{
		std::lock_guard<std::mutex> lock(mtx_);

		if (!is_fixed_) {
			snapshot.assign(buf_.begin(), buf_.begin() + size_);
		} else {
			// ring buffer モード：最大2回のコピー
			size_t cap = buf_.size();
			size_t start = (head_ + cap - size_) % cap;

			size_t first_len = std::min(size_, cap - start);
			snapshot.insert(snapshot.end(),
				buf_.begin() + start,
				buf_.begin() + start + first_len);

			size_t remain = size_ - first_len;
			if (remain > 0) {
				snapshot.insert(snapshot.end(),
					buf_.begin(),
					buf_.begin() + remain);
			}
		}
	}

	ANET_ASSERT(size_ == snapshot.size());

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
				r = g = b = BACKGROUND_LEVEL;  // 未観測セルは黒
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

wxImage Histgram::RenderRaw() const
{
	ANET_PROFILE_FUNC();

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
	: thm_(max_frames, bins, 0.0f, float(bins - 1), flags & ~HM_AutoNormValue, 0, mode),
	bins_(bins),
	alpha_(alpha),
	flags_(flags),
	base_min_(base_min),
	base_max_(base_max),
	min_val_(std::numeric_limits<float>::quiet_NaN()),
	max_val_(std::numeric_limits<float>::quiet_NaN()),
	buffer_(bins, 0.0f)
{
	thm_.SetValueMinMax(0.0f, 1.0f);   // 明示的に固定
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

void TimeHistogram::AddBatch(const std::vector<float>& values)
{
	ANET_PROFILE_FUNC();

	if (values.empty()) return;

	float old_min = min_val_;
	float old_max = max_val_;

	float vmin = *std::min_element(values.begin(), values.end());
	float vmax = *std::max_element(values.begin(), values.end());
	float a = std::clamp(alpha_, 0.001f, 1.0f);

	// 初期 min/max（base_min/base_max 優先）
	if (is_nan(min_val_)) min_val_ = !is_nan(base_min_) ? base_min_ : vmin;
	if (is_nan(max_val_)) max_val_ = !is_nan(base_max_) ? base_max_ : vmax;

	// 拡大方向のみ追従
	if (vmin < min_val_) min_val_ = (1.0f - a) * min_val_ + a * vmin;
	if (vmax > max_val_) max_val_ = (1.0f - a) * max_val_ + a * vmax;

	if (max_val_ - min_val_ < 1e-9f) max_val_ = min_val_ + 1e-3f;

	// レンジ拡大が起きたら再構築が必要
	if (min_val_ < old_min || max_val_ > old_max)
		need_rebuild_ = true;

	// 生データ蓄積（後で再binning可能）
	for (float v : values) cur_raw_.push_back(v);
}

void TimeHistogram::NextFrame() {
	frames_raw_.push_back(cur_raw_);
	cur_raw_.clear();

	if (need_rebuild_) {
		RebuildFromRaw();
		need_rebuild_ = false;
	}
	else {
		AppendCurrentFrameOnly();
	}
}

void TimeHistogram::AppendCurrentFrameOnly() {
	const auto& vals = frames_raw_.back();
	std::vector<float> bins(bins_, 0.0f);

	// bin 割り当て
	for (float v : vals) {
		int ix = MapToBin_(v);
		bins[ix] += 1.0f;
	}

	// フレーム単位正規化
	if (flags_ & HM_AutoNormValue) {
		float mv = *std::max_element(bins.begin(), bins.end());
		if (mv > 0.0f) {
			for (auto& x : bins) x /= mv;
		}
	}

	// TimeHeatMap に投入
	for (int i = 0; i < bins_; ++i) {
		int inv = bins_ - 1 - i;
		thm_.AddData(float(inv), bins[i]);
	}

	thm_.NextFrame();
}

void TimeHistogram::RebuildFromRaw()
{
	ANET_PROFILE_FUNC();

	thm_.Reset();

	for (const auto& frame : frames_raw_) {

		std::vector<float> bins(bins_, 0.0f);

		for (float v : frame) {
			int ix = MapToBin_(v);
			bins[ix] += 1.0f;
		}

		// フレーム単位正規化
		if (flags_ & HM_AutoNormValue) {
			float mv = *std::max_element(bins.begin(), bins.end());
			if (mv > 0.0f) {
				for (auto& x : bins) x /= mv;
			}
		}

		for (int i = 0; i < bins_; ++i) {
			int inv = bins_ - 1 - i;
			thm_.AddData(float(inv), bins[i]);
		}

		thm_.NextFrame();
	}
}

void TimeHistogram::Reset() {
	thm_.Reset();
	frames_raw_.clear();
	cur_raw_.clear();
	min_val_ = std::numeric_limits<float>::quiet_NaN();
	max_val_ = std::numeric_limits<float>::quiet_NaN();
	need_rebuild_ = false;
}

wxImage TimeHistogram::RenderRaw() const
{
	ANET_PROFILE_FUNC();

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

wxImage SweepedHeatMap::RenderRaw() const
{
	ANET_PROFILE_FUNC();

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

void SweepedHeatMap::SetValues(const torch::Tensor& values) {
	// grid shape = (H, W)
	ANET_ASSERT(values.dim() == 2);
	ANET_ASSERT(values.size(0) == height_);
	ANET_ASSERT(values.size(1) == width_);

	values_.resize(width_ * height_);
	for (int j = 0; j < height_; ++j) {
		for (int i = 0; i < width_; ++i) {
			float v = values[j][i].item<float>();
			values_[j * width_ + i] = v;
			value_min_ = std::min(value_min_, v);
			value_max_ = std::max(value_max_, v);
		}
	}
}
