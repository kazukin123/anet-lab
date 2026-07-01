// scaler_impl.hpp
#pragma once

#include "anet/scaler.hpp"

namespace anet::rl {

    // =============================================================
    // RunningMeanStd
    // =============================================================

    /**
     * @brief 並列Welfordアルゴリズムを用いて、ストリームデータの平均と分散を逐次計算するクラス。
     * Reward Scaling (Scalar) と Observation Normalization (Vector) の両方で使用可能。
     */
    class RunningMeanStd {
    public:
        /**
         * @param shape 統計を取るデータの形状。
         * Reward用(Scalar)なら空のvector {}
         * Obs用(Vector)なら {feature_dim}
         * @param epsilon ゼロ除算防止用
         */
        explicit RunningMeanStd(const std::vector<int64_t>& shape = {}, double epsilon = 1e-4);

        virtual ~RunningMeanStd() = default;

        void Reset();

        /**
         * @brief バッチデータを入力して統計情報を更新する
         * @param batch_input
         * Rewardの場合: [Batch] or [Batch, 1] -> 全要素をスカラー統計として扱う
         * Obsの場合:    [Batch, Feature]      -> Batch次元(dim0)をつぶしてFeatureごとの統計を取る
         */
        void Update(const torch::Tensor& batch_input);

        // ---- 内部状態取得
        torch::Tensor GetMean() const { return mean_; }
        torch::Tensor GetVar() const { return var_; }
        torch::Tensor GetStd() const { return torch::sqrt(var_) + epsilon_; }
        int64_t GetCount() const { return count_; }

        // ---- メトリクス監視用
        double GetMeanMean() const { return mean_.mean().item<double>(); }      ///< 平均ベクトルの平均値（ドリフト検知用）
        double GetStdMean() const { return GetStd().mean().item<double>(); }    ///< 標準偏差ベクトルの平均値（探索範囲拡大/縮小の検知用）
        double GetMeanMax() const { return mean_.abs().max().item<double>(); }  ///< 平均の最大値（最も偏っている次元の検知）
    private:
        void updateFromBatchStats(const torch::Tensor& batch_mean, const torch::Tensor& batch_m2, int64_t batch_size);
    private:
        int64_t count_;
        double epsilon_;

        // 内部状態はすべて Double Tensor (Scalar or Vector)
        torch::Tensor mean_;
        torch::Tensor var_;
        torch::Tensor m2_;
    };


    // =============================================================
    // SingleTensorNormalizer (内部用インターフェース)
    // =============================================================

    class SingleTensorNormalizer : virtual public ModuleBase {
    public:
        virtual torch::Tensor Normalize(const torch::Tensor& obs) const = 0;
        virtual torch::Tensor NormalizeAndUpdateStats(const torch::Tensor& obs) = 0;
        virtual void Reset() = 0;
        virtual ~SingleTensorNormalizer() = default;
    };


    // =============================================================
    // ConstantSingleTensorNormalizer（内部用）
    // =============================================================

    class ConstantSingleTensorNormalizer : public SingleTensorNormalizer {
    public:
        ConstantSingleTensorNormalizer(bool pass_through,
            const std::vector<int64_t>& shape, const std::optional<float>& clip_range,
            const std::vector<float>& fixed_mean = {}, // 空なら 0.0
            const std::vector<float>& fixed_std = {}   // 空なら 1.0
        );
        virtual ~ConstantSingleTensorNormalizer() = default;

        torch::Tensor Normalize(const torch::Tensor& obs) const override;
        torch::Tensor NormalizeAndUpdateStats(const torch::Tensor& obs) override;
        void Reset() override;
    public:
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
    private:
        std::pair<torch::Tensor, float> normalizeInternal(const torch::Tensor& obs) const;
    private:
        // 設定情報
        const bool pass_through_;
        const std::vector<int64_t> shape_;
        const std::optional<float> clip_range_;
        torch::Tensor mean_;
        torch::Tensor std_;

        // 統計情報
        float last_clip_ratio_ = 0.0f;  ///< 直近のNormalizeでのクリップ率（メトリクス用）
    };


    // =============================================================
    // RunningStdSingleTensorNormalizer（内部用）
    // =============================================================

    class RunningStdSingleTensorNormalizer : public SingleTensorNormalizer {
    public:
        RunningStdSingleTensorNormalizer(
            const std::optional<float>& clip_range,
            const std::vector<int64_t>& shape, float epsilon,
            bool use_robust_update, int robust_warmup_count, float robust_std_threshold,
            int post_process_type, float post_process_threshold,
            bool use_centering);
        virtual ~RunningStdSingleTensorNormalizer() = default;

        torch::Tensor Normalize(const torch::Tensor& obs) const override;
        torch::Tensor NormalizeAndUpdateStats(const torch::Tensor& obs) override;
        void Reset() override;
    public:
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
    private:
        std::pair<torch::Tensor, float> normalizeInternal(const torch::Tensor& obs) const;
    private:
        // 設定情報
        const std::optional<float> clip_range_;
        const std::vector<int64_t> shape_; // 期待する形状 (e.g. {S}、{C, H, W})
        const bool use_robust_update_;
        const int robust_warmup_count_;
        const float robust_std_threshold_;
        const int post_process_type_;
        const float post_process_threshold_;
        const bool use_centering_;

        // 統計情報
        RunningMeanStd stats_;
        float last_clip_ratio_ = 0.0f;    ///< 最後のNormalizeでのクリップ率（メトリクス用）
        float last_outlier_ratio_ = 0.0f; ///< 最後のNormalizeでの異常値判定率（メトリクス用）
    };


    // =============================================================
    // DictObservationNormalizer (公開クラス)
    // =============================================================

    class DictObservationNormalizer : public ObservationNormalizer, virtual public ModuleBase {
    public:
        DictObservationNormalizer() = default;

        // 初期化時にKeyごとのNormalizerを登録する
        void AddNormalizer(const std::string& key, std::shared_ptr<SingleTensorNormalizer> normalizer);

        anet::TensorDict Normalize(const anet::TensorDict& obs) const override;
        anet::TensorDict NormalizeAndUpdateStats(const anet::TensorDict& obs) override;
        void Reset() override;

        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
    private:
        std::unordered_map<std::string, std::shared_ptr<SingleTensorNormalizer>> normalizers_;
    };


    // =============================================================
    // RewardScaler
    // =============================================================

    class ConstantRewardScaler : public RewardScaler, virtual public ModuleBase {
    public:
        ConstantRewardScaler(float scale_factor, const std::optional<float>& clip_range = std::nullopt);
        torch::Tensor Scale(const torch::Tensor& reward) override;
        void Reset() override {}
    public:
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
    private:
        float scale_factor_;
        const std::optional<float> clip_range_;
        float last_clip_ratio_;
    };

    class RunningStdRewardScaler : public RewardScaler, virtual public ModuleBase {
    public:
        RunningStdRewardScaler(const std::optional<float>& clip_range = std::nullopt, float epsilon = 1e-8f, float post_scale = 1.0f);

        torch::Tensor Scale(const torch::Tensor& reward) override;
        void Reset() override;
    public:
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
    private:
        // 設定
        const std::optional<float> clip_range_; ///< クリップ範囲
        const float epsilon_;             ///< ゼロ除算防止用の極小値
        const float post_scale_;

        // 統計
        RunningMeanStd stats_;
        float last_clip_ratio_ = 0.0f;  ///< 直近のNormalizeでのクリップ率（メトリクス用）
    };

} // namespace anet::rl
