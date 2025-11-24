#pragma once
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <unordered_map>
#include <torch/torch.h>
#include "anet/heat_map.hpp"
#include "anet/tensor_check.hpp"
#include "anet/rl.hpp"

namespace anet {

    //==============================================================
    // ■ IFloatProbe
    //==============================================================

    class IFloatProbe {
    public:
        /**
         * @brief 観測情報からFloat値を生成
         */
        virtual std::optional<float> GetFloat(
            int step,
            const anet::rl::Experience& experience,
            std::shared_ptr<const anet::rl::BatchUpdateResult> result) const = 0;

        /**
         * @brief 値の下限が決まっている場合に返す。
         */
        virtual std::optional<float> GetMin() const = 0;

        /**
         * @brief 値の上限が決まっている場合に返す。
         */
        virtual std::optional<float> GetMax() const = 0;

        virtual ~IFloatProbe() = default;
    };

    /**
     * @brief MetricsMap に格納された scalar を参照する Probe。
     */
    class MetricsScalarProbe : public IFloatProbe {
    public:
        explicit MetricsScalarProbe(std::string key)
            : key_(std::move(key)) {
        }

        std::optional<float> GetFloat(
            int step,
            const anet::rl::Experience& experience,
            std::shared_ptr<const anet::rl::BatchUpdateResult> result) const override;
        std::optional<float> GetMin() const override { return std::nullopt; }
        std::optional<float> GetMax() const override { return std::nullopt; }

    private:
        std::string key_;
    };

    /**
     * @brief 固定値を返す Probe。
     */
    class StaticScalarProbe : public IFloatProbe {
    public:
        explicit StaticScalarProbe(float value)
            : value_(value) {
        }

        std::optional<float> GetFloat(
            int step,
            const anet::rl::Experience& experience,
            std::shared_ptr<const anet::rl::BatchUpdateResult> result) const override;
        std::optional<float> GetMin() const override { return value_; }
        std::optional<float> GetMax() const override { return value_; }

    private:
        float value_;
    };

    /**
     * @brief 外部から設定された Tensor の特定 index を参照する Probe。
     */
    //class TensorInputProbe : public IFloatProbe {
    //public:
    //    explicit TensorInputProbe(int64_t index)
    //        : index_(index) {
    //    }

    //    void SetTensor(const torch::Tensor& t);

    //    std::optional<float> GetFloat(
    //        int step,
    //        const anet::rl::Experience& experience,
    //        std::shared_ptr<const anet::rl::BatchUpdateResult> result) const override;
    //    std::optional<float> GetMin() const override { return std::nullopt; }
    //    std::optional<float> GetMax() const override { return std::nullopt; }

    //private:
    //    int64_t index_;
    //    torch::Tensor latest_tensor_;
    //    std::optional<float> cached_value_;
    //};

    /**
     * @brief 関数オブジェクトから値を生成する汎用 Probe。
     */
    class FunctionFloatProbe : public IFloatProbe {
    public:
        using Fn = std::function<
            std::optional<float>(
                int step,
                const anet::rl::Experience& experience,
                std::shared_ptr<const anet::rl::BatchUpdateResult> result)>;

        FunctionFloatProbe(
            Fn fn,
            std::optional<float> min = std::nullopt,
            std::optional<float> max = std::nullopt)
            : fn_(std::move(fn)), min_(min), max_(max) {
        }

        std::optional<float> GetFloat(
            int step,
            const anet::rl::Experience& experience,
            std::shared_ptr<const anet::rl::BatchUpdateResult> result) const override;
        std::optional<float> GetMin() const override { return min_; }
        std::optional<float> GetMax() const override { return max_; }

    private:
        Fn fn_;
        std::optional<float> min_;
        std::optional<float> max_;
    };

    /**
     * @brief EnvSpec.state_spec と index を用いて state から float を抽出する Probe。
     */
    class StateAxisProbe : public IFloatProbe {
    public:
        /**
         * @param spec  値範囲抽出に使うSteteSpec。nullptrの場合は値範囲を定義しない。
         * @param index 抽出する state の次元（flatten 後の index）
         */
        StateAxisProbe(int state_index, const anet::rl::StateSpec* spec = nullptr, bool for_next_state = false);

        std::optional<float> GetFloat(
            int step,
            const anet::rl::Experience& experience,
            std::shared_ptr<const anet::rl::BatchUpdateResult> result) const override;
        std::optional<float> GetMin() const override { return min_; }
        std::optional<float> GetMax() const override { return max_; }
    private:
        int state_index_;
        bool for_next_state_;
        std::optional<float> min_;
        std::optional<float> max_;
    };

    class RewardProbe : public IFloatProbe {
    public:
        /**
         * @param spec 値範囲抽出に使うEnvSpec。nullの場合は値範囲を定義しない。
         */
        RewardProbe(const anet::rl::EnvSpec* spec = nullptr);

        std::optional<float> GetFloat(
            int step,
            const anet::rl::Experience& experience,
            std::shared_ptr<const anet::rl::BatchUpdateResult> result) const override;
        std::optional<float> GetMin() const override { return min_; }
        std::optional<float> GetMax() const override { return max_; }
    private:
        std::optional<float> min_;
        std::optional<float> max_;
    };

    //==============================================================
    // ■ IVectorProbe
    //==============================================================

    /**
     * @brief TimeHistogram 用の float ベクトル Probe
     */
    class IVectorProbe {
    public:
        virtual ~IVectorProbe() = default;

        /**
         * @brief 現在のベクトルデータを返す
         * @return true = out に有効値が入る
         */
        virtual bool TryGetVector(std::vector<float>& out) = 0;
    };

    class TensorVectorProbe : public IVectorProbe {
    public:
        void UpdateTensor(const torch::Tensor& t) {
            tensor_ = t.flatten().to(torch::kCPU);
            ANET_CHECK_DTYPE(tensor_, torch::kFloat32);
        }

        bool TryGetVector(std::vector<float>& out) override {
            if (!tensor_.defined()) return false;

            out.resize(tensor_.size(0));
            std::memcpy(out.data(), tensor_.data_ptr<float>(),
                tensor_.size(0) * sizeof(float));
            return true;
        }

    private:
        torch::Tensor tensor_;
    };

    class StaticVectorProbe : public IVectorProbe {
    public:
        void Set(const std::vector<float>& v) { values_ = v; }

        bool TryGetVector(std::vector<float>& out) override {
            if (values_.empty()) return false;
            out = values_;
            return true;
        }
    private:
        std::vector<float> values_;
    };

    //==============================================================
    // ■ CompositeProbe
    //==============================================================
    /**
     * @brief HeatMapBuilder などが利用する X/Y/Value の Probe の束。
     */
    struct CompositeProbe {
        std::shared_ptr<IFloatProbe> x;
        std::shared_ptr<IFloatProbe> y;
        std::shared_ptr<IFloatProbe> value;
    };

    /**
     * @brief SweepedHeatMap 用に (N*M, input_dim) の入力テンソルを生成する。
     *        base_input（1サンプル）が設定されている場合は flatten サイズを採用し、
     *        未設定の場合は input_dim のみを使用する。
     */
    class SweepInputGenerator {
    public:
        SweepInputGenerator() = default;
        explicit SweepInputGenerator(int64_t input_dim)
            : input_dim_(input_dim) {
        }

        /// @brief sweep の基準値となる 1サンプル入力
        void SetBaseInput(const torch::Tensor& t) {
            base_input_ = t.flatten();
            input_dim_ = base_input_.size(0);
        }

        /**
         * @brief X/Y sweep から入力バッチ (N*M, input_dim) を構築
         */
        torch::Tensor BuildBatchInput(
            const std::vector<float>& xs,
            const std::vector<float>& ys,
            int64_t x_index,
            int64_t y_index
        ) const;

    private:
        int64_t input_dim_ = -1;
        torch::Tensor base_input_;  ///< optional (flattened)
    };

    /**
     * @brief batched_output (N*M, D) から ValueGrid (N,M) を抽出する。
     *
     * @details
     *  - extract_fn_ は「1 行の出力テンソル → float」のカスタム抽出関数
     *  - これにより Q(s,a)、maxQ、entropy、argmax など柔軟に対応可能
     */
    class SweepOutputExtractor {
    public:
        using ExtractFn = std::function<float(const torch::Tensor& output)>;

        /// @brief 固定index抽出のコンストラクタ
        explicit SweepOutputExtractor(int64_t output_index)
            : extract_fn_(
                [=](const torch::Tensor& output) { return output[output_index].item<float>();}
            ) {
        }

        /// @brief カスタム抽出関数を与えるコンストラクタ
        explicit SweepOutputExtractor(ExtractFn fn) : extract_fn_(std::move(fn)) {
        }

        /**
         * @brief batched_output (N*M, D) → ValueGrid (N,M)
         */
        torch::Tensor ExtractValue(
            const torch::Tensor& batched_output,
            int64_t N,
            int64_t M
        ) const;
    private:
        ExtractFn extract_fn_;  ///< 抽出関数(Tensor → float)
    };

    //==============================================================
    // ■ Builder
    //==============================================================

    //class HeatMapBuilder {
    //public:
    //    HeatMapBuilder(std::shared_ptr<anet::HeatMap> target)
    //        : target_(std::move(target)) {
    //    }

    //    void SetProbes(
    //        std::shared_ptr<IFloatProbe> x,
    //        std::shared_ptr<IFloatProbe> y,
    //        std::shared_ptr<IFloatProbe> value
    //    ) {
    //        probes_.x = std::move(x);
    //        probes_.y = std::move(y);
    //        probes_.value = std::move(value);
    //    }

    //    /// @brief 本フレームのデータ1件を push（画像生成しない）
    //    void Push() {
    //        float fx, fy, fv;
    //        if (!probes_.x->TryGetFloat(fx)) return;
    //        if (!probes_.y->TryGetFloat(fy)) return;
    //        if (!probes_.value->TryGetFloat(fv)) return;
    //        target_->AddData(fx, fy, fv);
    //    }

    //private:
    //    std::shared_ptr<anet::HeatMap> target_;
    //    CompositeProbe probes_;
    //};

    //class TimeHeatMapBuilder {
    //public:
    //    TimeHeatMapBuilder(std::shared_ptr<anet::TimeHeatMap> target)
    //        : target_(std::move(target)) {
    //    }

    //    void SetProbes(
    //        std::shared_ptr<IFloatProbe> in,
    //        std::shared_ptr<IFloatProbe> value
    //    ) {
    //        in_ = std::move(in);
    //        value_ = std::move(value);
    //    }

    //    /// @brief 現在フレームに1点 push
    //    void Push() {
    //        float fin, fv;
    //        if (!in_->TryGetFloat(fin)) return;
    //        if (!value_->TryGetFloat(fv)) return;
    //        target_->AddData(fin, fv);
    //    }

    //    /// @brief フレーム終了→次のフレームへ
    //    void NextFrame() {
    //        target_->NextFrame();
    //    }

    //private:
    //    std::shared_ptr<anet::TimeHeatMap> target_;
    //    std::shared_ptr<IFloatProbe> in_, value_;
    //};

    //class TimeHistogramBuilder {
    //public:
    //    TimeHistogramBuilder(std::shared_ptr<anet::TimeHistogram> target)
    //        : target_(std::move(target)) {
    //    }

    //    void SetProbe(std::shared_ptr<IVectorProbe> vec) {
    //        vec_probe_ = std::move(vec);
    //    }

    //    /// このフレームに vector<float> を追加
    //    void Push() {
    //        std::vector<float> values;
    //        if (!vec_probe_->TryGetVector(values)) return;
    //        buffer_.insert(buffer_.end(), values.begin(), values.end());
    //    }

    //    /// フレーム完了
    //    void NextFrame() {
    //        if (!buffer_.empty())
    //            target_->AddBatch(buffer_);

    //        buffer_.clear();
    //        target_->NextFrame();
    //    }
    //private:
    //    std::shared_ptr<anet::TimeHistogram> target_;
    //    std::shared_ptr<IVectorProbe> vec_probe_;
    //    std::vector<float> buffer_;
    //};

    //class SweepedHeatMapBuilder {
    //public:
    //    SweepedHeatMapBuilder(
    //        std::shared_ptr<anet::SweepedHeatMap> target,
    //        std::shared_ptr<SweepInputGenerator> input_gen,
    //        std::shared_ptr<SweepOutputExtractor> output_ext
    //    ) :
    //        target_(std::move(target)),
    //        input_gen_(std::move(input_gen)),
    //        output_ext_(std::move(output_ext)) {
    //    }

    //    void SetSweepX(const std::vector<float>& xs) { xs_ = xs; }
    //    void SetSweepY(const std::vector<float>& ys) { ys_ = ys; }

    //    void SetAxisIndex(int64_t x_index, int64_t y_index) {
    //        x_index_ = x_index;
    //        y_index_ = y_index;
    //    }

    //    /// @brief forward_fn: (Tensor input) → Tensor output
    //    void SetForward(std::function<torch::Tensor(const torch::Tensor&)> fn) {
    //        forward_fn_ = std::move(fn);
    //    }

    //    /// @brief 全 sweep グリッドを評価 → target_->values_ に直接反映
    //    void Evaluate() {
    //        torch::Tensor batched_input =
    //            input_gen_->BuildBatchInput(xs_, ys_, x_index_, y_index_);

    //        torch::Tensor out = forward_fn_(batched_input);

    //        torch::Tensor values =
    //            output_ext_->ExtractValue(out, xs_.size(), ys_.size());

    //        target_->SetValues(values);
    //    }

    //private:
    //    std::shared_ptr<anet::SweepedHeatMap> target_;
    //    std::shared_ptr<SweepInputGenerator> input_gen_;
    //    std::shared_ptr<SweepOutputExtractor> output_ext_;

    //    std::vector<float> xs_;
    //    std::vector<float> ys_;
    //    int64_t x_index_ = -1;
    //    int64_t y_index_ = -1;

    //    std::function<torch::Tensor(const torch::Tensor&)> forward_fn_;
    //};

} // namespace anet::viz
