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
        virtual std::optional<std::vector<float>> GetVector(
            int step,
            const anet::rl::Experience& experience,
            std::shared_ptr<const anet::rl::BatchUpdateResult> result) const = 0;
    };

    class BatchUpdateResultVectorProbe : public IVectorProbe {
    public:
        BatchUpdateResultVectorProbe(const std::string& key) : key_(key) {}

        std::optional<std::vector<float>> GetVector(
            int step,
            const anet::rl::Experience& experience,
            std::shared_ptr<const anet::rl::BatchUpdateResult> result) const override {

            auto tensor = result->GetTensor(key_);
            if (!tensor.has_value()) return std::nullopt;
            if (!tensor->defined()) return std::nullopt;

            torch::Tensor flat = tensor->flatten().to(torch::kCPU);
            ANET_CHECK_SHAPE(flat, { ANET_SHAPE_ANY });
            ANET_CHECK_DTYPE(flat, torch::kFloat32);

            std::vector<float> out;
            out.resize(flat.size(0));
            std::memcpy(out.data(), flat.data_ptr<float>(), flat.size(0) * sizeof(float));
            return out;
        }
    private:
        std::string key_;
    };

    //class StaticVectorProbe : public IVectorProbe {
    //public:
    //    void Set(const std::vector<float>& v) { values_ = v; }

    //    bool TryGetVector(std::vector<float>& out) override {
    //        if (values_.empty()) return false;
    //        out = values_;
    //        return true;
    //    }
    //private:
    //    std::vector<float> values_;
    //};

    class ISweepInputGenerator {
    public:
        // nullopt の axis は Generator が決めてよい
        /// @brief Observer側からの希望Gridサイズを受け取る
        virtual void ApplyGridSize(int width, int height) = 0;

        /// @brief Generator が最終決定した Gridサイズ
        virtual std::pair<int, int> GetGridSize() const = 0;

        /// @brief (grid_x, grid_y) に対応する NN入力Tensor
        virtual torch::Tensor BuildInputTensor(int grid_x, int grid_y) = 0;
        
        /**
         * @brief flatten された state のサイズを返す
         */
        virtual int64_t GetFlattenSize() const = 0;

        virtual ~ISweepInputGenerator() = default;
    };

    /**
     * @brief SweepedHeatMap の出力処理側インタフェース。
     * batched NN 出力から HeatMap の 1セル値を抽出する役割を持つ。
     *
     * GridSize は InputGenerator が決定し、Observer が確定値を通知する。
     */
    class ISweepOutputExtractor {
    public:
        virtual ~ISweepOutputExtractor() = default;

        /**
         * @brief Observer から確定された GridSize を受け取る。
         * @param width  グリッド横方向 (W)
         * @param height グリッド縦方向 (H)
         */
        virtual void ApplyGridSize(int width, int height) = 0;

        /**
         * @brief OutputExtractor が保持している最終 GridSize を返す。
         * @return (width, height)
         */
        virtual std::pair<int, int> GetGridSize() const = 0;

        /**
         * @brief NN 出力 から HeatMap 1セルの値を取り出す。
         * @param output 例: shape = [W*H,n_actions] など
         * @param grid_x HeatMap の X 座標 (0 ≦ grid_x < W)
         * @param grid_y HeatMap の Y 座標 (0 ≦ grid_y < H)
         * @return value（スカラー）
         */
        virtual float ExtractValue(
            const torch::Tensor& output,
            int grid_x,
            int grid_y) = 0;
    };

    /**
     * @brief  
     * RL の State を 2軸で sweep し、NN 入力テンソルを生成する。
     * 同時に NN 出力から値抽出を行う。  
     * ISweepInputGenerator / ISweepOutputExtractor を統合した処理クラス。
     */
    class RLStateSweepProcessor
        : public ISweepInputGenerator
        , public ISweepOutputExtractor
    {
    public:
        using ValueExtractFn = std::function<float(const torch::Tensor&)>;

        static float MaxExtractor(const torch::Tensor& t)
        {
            auto r = t.max(0);
            return std::get<0>(r).item<float>();
        };

        /**
         * @param env_spec      状態仕様
         * @param x_index       X 軸に割り当てる state flatten index
         * @param y_index       Y 軸に割り当てる state flatten index
         * @param base_state    Sweep ベースとなる state(flatten)。未指定なら zero state。
         * @param x_min_override  X sweep 範囲の最小値（未指定時は StateSpec に従う）
         * @param x_max_override  X sweep 範囲の最大値
         * @param y_min_override  Y sweep 範囲の最小値
         * @param y_max_override  Y sweep 範囲の最大値
         * @param value_extractor 出力抽出関数。デフォルト max(sample)。
         */
        RLStateSweepProcessor(
            const anet::rl::EnvSpec& env_spec,
            int x_index,
            int y_index,
            ValueExtractFn value_extractor = &MaxExtractor,
            std::optional<torch::Tensor> base_state = std::nullopt,
            std::optional<float> x_min_override = std::nullopt,
            std::optional<float> x_max_override = std::nullopt,
            std::optional<float> y_min_override = std::nullopt,
            std::optional<float> y_max_override = std::nullopt
        );

        ~RLStateSweepProcessor() override = default;
    public:
        // ===========================================================
        // ISweepInputGenerator
        // ===========================================================
        void ApplyGridSize(int width, int height) override;
        std::pair<int, int> GetGridSize() const override;
        torch::Tensor BuildInputTensor(int grid_x, int grid_y) override;
        int64_t GetFlattenSize() const override;
    public:
        // ===========================================================
        // ISweepOutputExtractor
        // ===========================================================
        float ExtractValue(const torch::Tensor& batched_out,
            int grid_x, int grid_y) override;
    private:
        const anet::rl::EnvSpec env_spec_;
        const anet::rl::StateSpec state_spec_;

        int x_index_;
        int y_index_;

        // Sweep 範囲
        float x_min_;
        float x_max_;
        float y_min_;
        float y_max_;

        bool x_min_overridden_ = false;
        bool x_max_overridden_ = false;
        bool y_min_overridden_ = false;
        bool y_max_overridden_ = false;

        // Base(flatten) state
        torch::Tensor base_flatten_;

        // GridSize（InputGenerator が決定して OutputExtractor に伝える）
        int grid_w_ = 256;
        int grid_h_ = 256;

        // NN 出力 → 値抽出
        std::function<float(const torch::Tensor&)> value_extractor_;
    };

    ///**
    // * @brief SweepedHeatMap 用に (N*M, input_dim) の入力テンソルを生成する。
    // *        base_input（1サンプル）が設定されている場合は flatten サイズを採用し、
    // *        未設定の場合は input_dim のみを使用する。
    // */
    //class SweepInputGenerator {
    //public:
    //    SweepInputGenerator() = default;
    //    explicit SweepInputGenerator(int64_t input_dim)
    //        : input_dim_(input_dim) {
    //    }

    //    /// @brief sweep の基準値となる 1サンプル入力
    //    void SetBaseInput(const torch::Tensor& t) {
    //        base_input_ = t.flatten();
    //        input_dim_ = base_input_.size(0);
    //    }

    //    /**
    //     * @brief X/Y sweep から入力バッチ (N*M, input_dim) を構築
    //     */
    //    torch::Tensor BuildBatchInput(
    //        const std::vector<float>& xs,
    //        const std::vector<float>& ys,
    //        int64_t x_index,
    //        int64_t y_index
    //    ) const;

    //private:
    //    int64_t input_dim_ = -1;
    //    torch::Tensor base_input_;  ///< optional (flattened)
    //};

    ///**
    // * @brief batched_output (N*M, D) から ValueGrid (N,M) を抽出する。
    // *
    // * @details
    // *  - extract_fn_ は「1 行の出力テンソル → float」のカスタム抽出関数
    // *  - これにより Q(s,a)、maxQ、entropy、argmax など柔軟に対応可能
    // */
    //class SweepOutputExtractor {
    //public:
    //    using ExtractFn = std::function<float(const torch::Tensor& output)>;

    //    /// @brief 固定index抽出のコンストラクタ
    //    explicit SweepOutputExtractor(int64_t output_index)
    //        : extract_fn_(
    //            [=](const torch::Tensor& output) { return output[output_index].item<float>();}
    //        ) {
    //    }

    //    /// @brief カスタム抽出関数を与えるコンストラクタ
    //    explicit SweepOutputExtractor(ExtractFn fn) : extract_fn_(std::move(fn)) {
    //    }

    //    /**
    //     * @brief batched_output (N*M, D) → ValueGrid (N,M)
    //     */
    //    torch::Tensor ExtractValue(
    //        const torch::Tensor& batched_output,
    //        int64_t N,
    //        int64_t M
    //    ) const;
    //private:
    //    ExtractFn extract_fn_;  ///< 抽出関数(Tensor → float)
    //};


} // namespace anet