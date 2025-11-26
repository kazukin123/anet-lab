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
        /// @param grid_width gridの幅。-1は指定しないの意味
        /// @param grid_height gridの高さ。-1は指定しないの意味
        /// @brief Observer側からの希望Gridサイズを受け取る
        virtual void ApplyGridSize(int grid_width, int grid_height) = 0;

        /// @brief Generator が最終決定した Gridサイズ
        virtual std::pair<int, int> GetGridSize() const = 0;

        /// @return Tensor: [W*H, state_dim], Flatten済み、device上
        virtual torch::Tensor BuildInputTensor() = 0;

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
         * @param grid_width  グリッド幅 (W)
         * @param grid_height グリッド高さ (H)
         */
        virtual void ApplyGridSize(int grid_width, int grid_height) = 0;

        /**
         * @brief OutputExtractor が保持している最終 GridSize を返す。
         * @return (width, height)
         */
        virtual std::pair<int, int> GetGridSize() const = 0;

        /// @brief batched output から W*H 個の値を GPU 上で抽出
        /// @return shape = [W*H], dtype=float32, device=device
        virtual torch::Tensor ExtractValue(
            const torch::Tensor& output) = 0;
    };

    /**
     * @brief  
     * RL の State を 2軸で sweep し、NN 入力テンソルを生成する。
     * 同時に NN 出力から値抽出を行う。  
     * ISweepInputGenerator / ISweepOutputExtractor を統合した処理クラス。
     */
    class RLStateSweepProcessor : public ISweepInputGenerator, public ISweepOutputExtractor {
    public:
        using ValueExtractFn = std::function<torch::Tensor(const torch::Tensor&)>;

        static torch::Tensor MaxExtractor(const torch::Tensor& t) { // t: [W*H, out_dim]
            return std::get<0>(t.max(1)); // [W*H]
        }
        static torch::Tensor MeanExtractor(const torch::Tensor& t) {
            return t.mean(1); // [W*H]
        }
        static torch::Tensor IndexExtractor(const torch::Tensor& t, int idx) {
            return t.index({ torch::indexing::Slice(), idx });
        }
        static torch::Tensor DiffIndexExtractor(const torch::Tensor& t, int plus_idx, int minus_idx) {
            using namespace torch::indexing;
            auto plus_val = t.index({ Slice(), plus_idx });
            auto minus_va = t.index({ Slice(), minus_idx });
            return plus_val - minus_va;
        }
        static torch::Tensor PairDiffExtractor(const torch::Tensor& t, int n_actions) {
            using namespace torch::indexing;
            ANET_CHECK_SHAPE(t, { ANET_SHAPE_ANY, n_actions * 2 });
            auto q_online = t.index({ Slice(), Slice(0, n_actions) });               // [N, n_actions]
            auto q_target = t.index({ Slice(), Slice(n_actions, n_actions * 2) });   // [N, n_actions]
            auto diff = (q_online - q_target).abs().mean(1);                         // [N]
            return diff;
        }
        static torch::Tensor QdeltaQmaxCombined(
            const torch::Tensor& t,
            int n_actions,
            float qdelta_scale,      // ex: 0.5f
            float qmax_threshold)    // ex: 20.0f
        {
            using namespace torch::indexing;

            auto q_online = t.index({ Slice(), Slice(0, n_actions) });
            auto q_target = t.index({ Slice(), Slice(n_actions, 2 * n_actions) });

            // Qdelta = |online - target| の平均
            auto qdelta = (q_online - q_target).abs().mean(1);       // [N]

            // Qmax = |online| の最大値
            auto qmax = std::get<0>(q_online.abs().max(1));       // [N]

            // 正規化（0〜1）
            auto qdelta_norm = (qdelta / qdelta_scale).clamp(0.0f, 1.0f);
            auto qmax_norm = (qmax / qmax_threshold).clamp(0.0f, 1.0f);

            // 合成
            auto combined = qdelta_norm * qmax_norm;
            return combined;
        }
        // =============================
        // QdeltaQmaxCombinedAuto
        // -----------------
        //    Qdelta 高 × Qmax 高
        //    → 発散で地形が壊れて target 追従不能の領域
        //    Qdelta 高 × Qmax 低
        //    → target追従不足（でも発散ではない）
        //    Qdelta 低 × Qmax 高
        //    → Qの発散だけが起きている領域（target が遅れて青くなる前兆）
        //    両方低
        //    → 安定
        // -----------------
        //    発散領域（本当に最悪の赤）
        //    → 真っ赤に浮かび上がる
        //    （Qdelta_norm ≈ 1＆Qmax_norm ≈ 1）
        //    Qmax が高いのに Qdelta はまだ小さい（発散初期段階）
        //    → 暗オレンジ色に現れる
        //    → 崩壊の前兆が見える
        //    赤いけれど発散ではない Qdelta の赤
        //    → 黄色〜緑程度で止まる
        //    Qdelta が高いが Qmax はまだ青い（target追従だけ遅れ）
        //    → 緑〜黄に現れる
        // =============================
        static torch::Tensor QdeltaQmaxCombinedAuto(
            const torch::Tensor& t,
            int n_actions)
        {
            using namespace torch::indexing;

            auto q_online = t.index({ Slice(), Slice(0, n_actions) });
            auto q_target = t.index({ Slice(), Slice(n_actions, 2 * n_actions) });

            auto qdelta = (q_online - q_target).abs().mean(1);
            auto qmax = std::get<0>(q_online.abs().max(1));

            // GPU 上の max（scalar-tensor）
            auto qdelta_max = qdelta.max();  // shape [], device same as qdelta
            auto qmax_max = qmax.max();    // shape [], device same as qmax

            // EPS を GPU Tensor で作る
            auto eps = torch::full({}, 1e-6, qdelta.options());   // shape=[]

            // GPU同士の除算 → 完全GPU処理
            auto qdelta_norm = qdelta / (qdelta_max + eps);
            auto qmax_norm = qmax / (qmax_max + eps);

            return qdelta_norm * qmax_norm;
        }

        /// QDELTA × |Qdiff|
        static torch::Tensor QdeltaQdiffCombinedAuto(
            const torch::Tensor& t,
            int n_actions,
            int action_index_a,
            int action_index_b)
        {
            using namespace torch::indexing;

            ANET_ASSERT(t.dim() == 2);
            ANET_ASSERT(n_actions > 0);
            ANET_ASSERT(t.size(1) == n_actions * 2);
            ANET_ASSERT(action_index_a >= 0 && action_index_a < n_actions);
            ANET_ASSERT(action_index_b >= 0 && action_index_b < n_actions);

            // t: [N, 2 * n_actions] = [q_online, q_target]
            auto q_online = t.index({ Slice(), Slice(0, n_actions) });
            auto q_target = t.index({ Slice(), Slice(n_actions, 2 * n_actions) });

            // Qdelta(s) = mean_a |Q_online(s,a) - Q_target(s,a)|
            auto qdelta = (q_online - q_target).abs().mean(1);  // [N]

            // Qdiff(s) = Q_online(s, b) - Q_online(s, a)
            auto qdiff = q_online.index({ Slice(), action_index_b })
                - q_online.index({ Slice(), action_index_a }); // [N]
            auto qdiff_abs = qdiff.abs();                      // [N]

            // GPU 上での max（0 次元 Tensor）
            auto qdelta_max = qdelta.max();    // []
            auto qdiff_max = qdiff_abs.max();  // []

            // EPS を GPU Tensor で生成
            auto eps = torch::full({}, 1e-6f, t.options());

            // 自動正規化（0〜1）
            auto qdelta_norm = qdelta / (qdelta_max + eps);
            auto qdiff_norm = qdiff_abs / (qdiff_max + eps);

            // 合成: target 乖離 × 境界ゆらぎ
            auto combined = qdelta_norm * qdiff_norm;          // [N]

            return combined;
        }
        static torch::Tensor BoundaryMaskFromQdiffAuto(
            const torch::Tensor& t,
            int n_actions,
            int action_index_a,
            int action_index_b)
        {
            using namespace torch::indexing;

            ANET_ASSERT(t.dim() == 2);
            ANET_ASSERT(n_actions > 0);
            ANET_ASSERT(t.size(1) == n_actions);
            ANET_ASSERT(action_index_a >= 0 && action_index_a < n_actions);
            ANET_ASSERT(action_index_b >= 0 && action_index_b < n_actions);

            // t: [N, 2 * n_actions] = [q_online, q_target]
            auto q_online = t.index({ Slice(), Slice(0, n_actions) });

            // Qdiff(s) = Q_online(s, b) - Q_online(s, a)
            auto qdiff = q_online.index({ Slice(), action_index_b }) -
                q_online.index({ Slice(), action_index_a });  // [N]
            auto qdiff_abs = qdiff.abs();                             // [N]

            // GPU 上で max を取得（0 次元 Tensor）
            auto qdiff_max = qdiff_abs.max();                         // []

            // EPS を GPU Tensor として生成
            auto eps = torch::full({}, 1e-6f, t.options());

            // 正規化: 0〜1 （境界からの距離）
            auto qdiff_norm = qdiff_abs / (qdiff_max + eps);          // [N], 0〜1

            // 境界強度: 境界付近ほど 1.0、遠いほど 0.0
            auto boundary_strength = 1.0f - qdiff_norm;               // [N]

            return boundary_strength;
        }

        static torch::Tensor BoundaryMaskedQdeltaAuto(
            const torch::Tensor& t,
            int n_actions,
            int action_index_a,
            int action_index_b)
        {
            using namespace torch::indexing;

            ANET_ASSERT(t.dim() == 2);
            ANET_ASSERT(n_actions > 0);
            ANET_ASSERT(t.size(1) == n_actions * 2);
            ANET_ASSERT(action_index_a >= 0 && action_index_a < n_actions);
            ANET_ASSERT(action_index_b >= 0 && action_index_b < n_actions);

            // t: [N, 2 * n_actions] = [q_online, q_target]
            auto q_online = t.index({ Slice(), Slice(0, n_actions) });
            auto q_target = t.index({ Slice(), Slice(n_actions, 2 * n_actions) });

            // Qdelta(s) = mean_a |Q_online(s,a) - Q_target(s,a)|
            auto qdelta = (q_online - q_target).abs().mean(1);        // [N]

            // Qdiff(s) = Q_online(s, b) - Q_online(s, a)
            auto qdiff = q_online.index({ Slice(), action_index_b }) -
                q_online.index({ Slice(), action_index_a }); // [N]
            auto qdiff_abs = qdiff.abs();                             // [N]

            // max を GPU 上で取得
            auto qdelta_max = qdelta.max();                           // []
            auto qdiff_max = qdiff_abs.max();                        // []

            auto eps = torch::full({}, 1e-6f, t.options());

            // 0〜1 に自動正規化
            auto qdelta_norm = qdelta / (qdelta_max + eps);           // [N]
            auto qdiff_norm = qdiff_abs / (qdiff_max + eps);        // [N]

            // 境界強度: 境界付近ほど 1.0、遠いほど 0.0
            auto boundary_strength = 1.0f - qdiff_norm;               // [N]

            // 境界まわりでの Qdelta を強調
            auto combined = boundary_strength * qdelta_norm;          // [N]

            return combined;
        }
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
            const anet::rl::StateSpec& state_spec,
            int x_index,
            int y_index,
            ValueExtractFn value_extract_fn = &MaxExtractor,
            const torch::Device& device = torch::kCUDA,
            std::optional<torch::Tensor> base_state = std::nullopt,
            std::optional<float> x_min_override = std::nullopt,
            std::optional<float> x_max_override = std::nullopt,
            std::optional<float> y_min_override = std::nullopt,
            std::optional<float> y_max_override = std::nullopt
        );

        ~RLStateSweepProcessor() override = default;
    public:
        void ApplyGridSize(int width, int height) override;
        std::pair<int, int> GetGridSize() const override;
        torch::Tensor BuildInputTensor() override;
        int64_t GetFlattenSize() const override;
        torch::Tensor ExtractValue(const torch::Tensor& batched_out) override;
    private:
        const anet::rl::StateSpec state_spec_;
        const torch::Device device_;

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
        ValueExtractFn value_extract_fn_;
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