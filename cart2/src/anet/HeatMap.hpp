#pragma once
#include <wx/image.h>
#include <vector>
#include <deque>
#include <string>
#include <mutex>
#include <functional>
#include <cstdint>
#include <torch/torch.h>

namespace anet {

    // ============================================================
    // HeatMapFlags : 設定フラグ
    // ============================================================
    enum HeatMapFlags : uint32_t {
        HM_None = 0,
        HM_LogScale = 1 << 0,   // 対数スケール
        HM_AutoNorm = 1 << 1,   // 自動正規化
        HM_MeanMode = 1 << 2,   // 平均値で集約
        HM_SumMode = 1 << 3,   // 合計値で集約
        HM_Default = HM_AutoNorm | HM_SumMode
    };

    class ImageSource {
    public:
        virtual void Reset() = 0;
        virtual wxImage RenderRaw() const = 0;
        virtual std::string GetImageSubType() const = 0; // 例: "heat_map", "time_heat_map", ...

        wxImage Render(int width = -1, int height = -1) const {
            wxImage src = RenderRaw(); // 元解像度で生成 (bins x frames)
            if (width < 0 && height < 0) return src;
            if (width < 0) width = src.GetWidth();
            if (height < 0) height = src.GetHeight();
            return src.Scale(width, height, wxIMAGE_QUALITY_NEAREST);
        }

        void SavePng(const std::string& filename, int width = -1, int height = -1) const;

        virtual ~ImageSource() = default;
    };

    // ============================================================
    // HeatMap : 任意座標データの2D可視化（加算/平均対応）
    // ============================================================
    class HeatMap : public ImageSource {
    public:
        HeatMap(int width, int height,
            float x_min = 0.0f, float x_max = 1.0f,
            float y_min = 0.0f, float y_max = 1.0f,
            size_t max_points = 0,
            uint32_t flags = HM_Default);

        std::string GetImageSubType() const override { return "heat_map"; }

        void AddData(float x, float y, float value);
        void Reset() override;
        wxImage RenderRaw() const override;
    protected:
        struct Sample {
            float x, y, value;
        };

        int width_, height_;
        float x_min_, x_max_, y_min_, y_max_;
        float value_min_, value_max_;
        size_t max_points_;
        uint32_t flags_;

        std::deque<Sample> samples_;
        mutable std::mutex mtx_;

        void UpdateMinMax(float value);
        wxImage GenerateImage(const std::vector<float>& buf, const std::vector<int>& count) const;
    };

    enum class TimeFrameMode {
        Unlimited = 0,  // 全履歴保持
        Overwrite,      // ラップ上書き
        Scroll          // 上詰めスクロール
    };

    class TimeHeatMap : public HeatMap {
    public:
        TimeHeatMap(int width_bins, int height,
            float x_min, float x_max,
            uint32_t flags = HM_Default,
            size_t max_points = 0,
            TimeFrameMode mode = TimeFrameMode::Unlimited);

        std::string GetImageSubType() const override { return "timed_heat_map"; }

        void AddData(float x, float value);
        void NextFrame();          // フレームを進める
        void Reset() override;
        wxImage RenderRaw() const override;    // 上から下へ時間描画（既存HeatMap::Renderを拡張）

        int GetCurrentFrame() const { return cur_frame_; }
        int GetTotalFrames() const { return total_frames_; }
    private:
        const TimeFrameMode mode_;
        int cur_frame_;        // 現在フレームカウント
        int total_frames_;     // 累計フレーム数（Unlimited時に増え続ける）
        mutable std::mutex mtx_;
    protected:
        void EraseRow_(int y_row);   // 行削除（Overwrite用）
        //void ScrollUp_();                     // スクロール（Scroll用）
        void ScrollDown_();      // 古い行を上へ押し上げ、最新を下に積む
    };


    // ============================================================
    // Histgram : 1次元分布可視化
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
    // TimeHistogram : 分布の時間変化（フレーム単位管理）
    // ============================================================
    class TimeHistogram : public ImageSource {
        public:
            TimeHistogram(int bins, int max_frames,
                TimeFrameMode mode = TimeFrameMode::Scroll,
                bool auto_norm = true, bool auto_range = true,
                float min_val = -1, float max_val = -1);

            void AddBatch(const std::vector<float>& values);
            void NextFrame();

            void Reset()override;
            wxImage RenderRaw() const override { return thm_.RenderRaw(); }

            std::string GetImageSubType() const override { return "time_histgram"; }
    private:
            TimeHeatMap thm_;
            int bins_;
            float min_val_;
            float max_val_;
            bool auto_norm_;
            bool auto_range_;
            std::vector<float> buffer_;
            float smooth_min_ = 0.0f;
            float smooth_max_ = 1.0f;
            float smooth_rate_ = 0.05f; // 5% ずつ追従
        };

    // ============================================================
    // SweepedHeatMap : xyスイープによるヒートマップ生成
    // ============================================================
    class SweepedHeatMap : public ImageSource {
    public:
        SweepedHeatMap(int width, int height,
            float x_min, float x_max,
            float y_min, float y_max);

        std::string GetImageSubType() const override { return "sweeped_map"; }

        void Evaluate(const std::function<float(float, float)>& func);
        wxImage RenderRaw() const override;
		void Reset() override;

		// Q値などのNN可視化（GPUバッチ対応）
        static SweepedHeatMap EvaluateTensorFunction(
            int width, int height,
            float x_min, float x_max,
            float y_min, float y_max,
            const torch::Device& device,
            const std::function<torch::Tensor(const torch::Tensor&)>& forward,              //  torch::Tensor input -> torch::Tensor output
			const std::function<torch::Tensor(const torch::Tensor&)>& value_extractor);     //  torch::Tensor output -> torch::Tensor value

        //auto q_map = anet::SweepedHeatMap::FromSweepFunctionNN(
        //    128, 128,
        //    -2.4f, 2.4f,   // x range
        //    -0.21f, 0.21f,  // theta range
        //    // forward関数: [x, theta] -> [x, x_dot, theta, theta_dot] のテンソルを作る
        //    [&](const torch::Tensor& xy) {
        //        // xy: (N, 2)
        //        auto x     = xy.index({ torch::indexing::Slice(), 0 }).unsqueeze(1); // (N,1)
        //        auto theta = xy.index({ torch::indexing::Slice(), 1 }).unsqueeze(1); // (N,1)
        //        // 固定値を作成（すべて0）
		//        auto x_dot = torch::zeros_like(x);          // (N,1)
		//        auto theta_dot = torch::zeros_like(theta);  // (N,1)
        //        // 4次元状態 [x, x_dot, theta, theta_dot]
        //        auto s = torch::cat({ x, x_dot, theta, theta_dot }, 1).to(device); // (N,4)
        //        return model->forward(s);
        //    },
        //    // value抽出関数: Q値のうちaction=0の値を抽出
        //    [&](const torch::Tensor& out) {
        //        return out.index({ torch::indexing::Slice(), 0 });
        //    }
        //);
    private:
        int width_, height_;
        float x_min_, x_max_, y_min_, y_max_;
        std::vector<float> values_;
        float value_min_, value_max_;

        void Normalize();
    };

} // namespace anet


void test_heatmap_and_histgram();

