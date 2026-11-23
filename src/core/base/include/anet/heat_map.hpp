#pragma once
#include <vector>
#include <string>
#include <mutex>
#include <functional>
#include <cstdint>
#include <limits>
#include <torch/torch.h>
#include <wx/image.h>
#include "anet/image.hpp"


namespace anet {


    // ============================================================
    // HeatMapConfig
    // ============================================================

    enum HeatMapFlags : uint32_t {
        HM_None = 0,
        HM_LogScaleValue = 1 << 0,  // 値(色強度)の対数圧縮（既存: HM_LogScale）
        HM_AutoNormValue = 1 << 1,  // 値(色強度)の正規化
        HM_AutoNormX = 1 << 2,  // X軸の自動スケーリング
        HM_AutoNormY = 1 << 3,  // Y軸の自動スケーリング
        HM_MeanMode = 1 << 4,  // 平均モード
        HM_SumMode = 1 << 5,  // 合計モード

        // 拡張
        HM_AutoScaleAxis = 1 << 6,  // 値軸(データ軸)のレンジ自動追従（NaN指定側のみEMA更新）
        HM_LogScaleAxis = 1 << 7,  // 値軸を対数スケール表示（log1p(|v|)の符号付）
        HM_ShowZeroLine = 1 << 8,  // 値0の位置に水平ラインを描画
        HM_FlipY = 1 << 9,  // Y軸方向を反転して描画（上が大値）

        HM_Default = HM_AutoNormValue | HM_SumMode
    };

    enum class TimeFrameMode {
        Scale = 0,
        Overwrite,
        Scroll
    };


    // ============================================================
    // HeatMap : 任意 (x,y,value) の散布を2Dヒートマップ化
    // ============================================================

    class HeatMap : public ImageSource {
    public:
        HeatMap(int width, int height, float x_min = 0.0f, float x_max = 1.0f,
            float y_min = 0.0f, float y_max = 1.0f, size_t max_points = 0,
            uint32_t flags = HM_Default);

        std::string GetImageSubType() const override { return "heat_map"; }

        void AddData(float x, float y, float value);
        void AddDataBatch(const std::vector<float>& xv, const std::vector<float>& yv, const std::vector<float>& vv);
        void SetGridValues(const float* values, int width, int height);
        void Reset() override;
        wxImage RenderRaw() const override;
    protected:
        struct Sample { float x, y, value; };

        int width_, height_;
        float x_min_, x_max_, y_min_, y_max_;
        float value_min_, value_max_;
        size_t max_points_;
        uint32_t flags_;

        std::vector<Sample> buf_;
        size_t head_ = 0;
        size_t size_ = 0;
        bool is_fixed_ = false;

        mutable std::mutex mtx_;

        void UpdateMinMax_(float value) {
            value_min_ = std::min(value_min_, value);
            value_max_ = std::max(value_max_, value);
        }
    private:
        // ワークバッファ
        mutable std::vector<float> work_buf_;
        mutable std::vector<int> work_cnt_;
    };


    // ============================================================
    // 時系列ヒートマップ：横(右方向)に時間進行、縦は値軸
    //   AddData(in, out): in=縦方向値(=値軸), out=強度
    // ============================================================

    class TimeHeatMap : public HeatMap {
    public:
        TimeHeatMap(int width_frames, int height_bins, float in_min, float in_max,
            uint32_t flags = HM_Default, size_t max_points = 0,
            TimeFrameMode mode = TimeFrameMode::Scale);

        std::string GetImageSubType() const override { return "timed_heat_map"; }

        void AddData(float in, float out);
        void NextFrame();
        void Reset() override;
        wxImage RenderRaw() const override;

        int GetCurrentFrame() const { return cur_frame_; }
        int GetTotalFrames() const { return total_frames_; }

        void SetValueMinMax(float min, float max) {
            value_min_ = min;
            value_max_ = max;
        }
    private:
        const TimeFrameMode mode_;
        int cur_frame_;
        int total_frames_;
        int max_frames_;
        float smoothed_scale_x_;

        void Scroll_();
        void EraseCol_(int x_col);
    };


    // ============================================================
    // Histgram : 静的1Dヒストグラム
    // ============================================================
    class Histgram : public ImageSource {
    public:
        Histgram(int bins, float min_val, float max_val, int width = 256, int height = 128);
        std::string GetImageSubType() const override { return "histgram"; }

        void AddData(float value);
        void Reset() override;
        wxImage RenderRaw() const override;

    private:
        int bins_;
        int width_, height_;
        float min_val_, max_val_;
        std::vector<int> counts_;
        mutable std::mutex mtx_;
    };

    // ============================================================
    // TimeHistogram : 値分布の時間推移（X=時間、Y=値軸、色=頻度）
    //   - 固定/自動(EMA)の二択
    //   - 対数“軸”描画、ゼロライン描画
    //   - 値範囲が指定(NaN以外)されていればそれをベースとして尊重
    // ============================================================
    class TimeHistogram : public ImageSource {
    public:
        TimeHistogram(int bins, int width_frames,
            TimeFrameMode mode = TimeFrameMode::Scroll,
            uint32_t flags = HM_AutoScaleAxis | HM_AutoNormValue,
            float base_min = std::numeric_limits<float>::quiet_NaN(),
            float base_max = std::numeric_limits<float>::quiet_NaN(),
            float alpha = 1.0f
            );

        void AddBatch(const std::vector<float>& values);
        void NextFrame();
        void Reset() override;
        wxImage RenderRaw() const override;

        std::string GetImageSubType() const override { return "time_histgram"; }

        // 参照用（描画時のレンジ）
        float MinVal() const { return min_val_; }
        float MaxVal() const { return max_val_; }
    private:
        void RebuildFromRaw();
        void AppendCurrentFrameOnly();
    private:
        std::vector<std::vector<float>> frames_raw_;   ///< 全フレームの生データ
        std::vector<float> cur_raw_;                  ///< 今フレームの生データ

        TimeHeatMap thm_;          // 横:フレーム, 縦:bin(0..bins-1)
        int bins_;
        float alpha_;              ///< EMA係数
        uint32_t flags_;
        bool need_rebuild_ = false;

        // 固定ベース（NaNで未指定）
        float base_min_, base_max_;

        // 実効レンジ（描画に用いる）
        float min_val_, max_val_;

        // フレーム内カウントバッファ
        std::vector<float> buffer_;

        // 値→bin の写像
        int MapToBin_(float v) const;
        int MapToBinLinear_(float v) const;
        int MapToBinLogAxis_(float v) const;
    };


    // ============================================================
    // SweepedHeatMap : (x,y) 全域をスイープして値評価
    // ============================================================
    class SweepedHeatMap : public ImageSource {
    public:
        SweepedHeatMap(int width, int height, float x_min, float x_max,
            float y_min, float y_max);

        std::string GetImageSubType() const override { return "sweeped_map"; }

        void Evaluate(const std::function<float(float, float)>& func);
        wxImage RenderRaw() const override;
        void Reset() override;
        void SetValues(const torch::Tensor& grid);

    private:
        int width_, height_;
        float x_min_, x_max_, y_min_, y_max_;
        std::vector<float> values_;
        float value_min_, value_max_;
    };

}  // namespace anet
