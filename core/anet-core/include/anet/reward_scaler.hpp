// anet/reward_scaler.hpp
#pragma once

#include <cstdint>
#include <optional>
#include <torch/torch.h>
#include "anet/common.hpp"
#include "anet/config.hpp"

namespace anet::rl {

    // =============================================================
    // RewardScaler
    // =============================================================

    class RewardScaler : virtual public DataExporter {
    public:
        /**
         * @brief 報酬(Batch)を受け取り、統計を更新し、スケールされたTensorを返す
         * @param reward [N] の形状を持つ Tensor (CPU or CUDA、kFloat32)
         * @return [N] スケール済みの Tensor
         */
        virtual torch::Tensor Scale(const torch::Tensor& reward) = 0;
        virtual void Reset() = 0;

        virtual ~RewardScaler() = default;
    public:
        static constexpr const char* kKeyPrefix = "reward_scaler.";
        static constexpr const char* kKeyScale = "reward_scaler.scale";
        static constexpr const char* kKeyCount = "reward_scaler.count";
        static constexpr const char* kKeyMean = "reward_scaler.mean";
        static constexpr const char* kKeyM2 = "reward_scaler.m2";
        static constexpr const char* kKeyStd = "reward_scaler.std";
    };

    class ConstantRewardScaler : public RewardScaler, virtual public DataExporterBase {
    public:
        ConstantRewardScaler(float scale_factor, const std::optional<float>& clip = std::nullopt);
        torch::Tensor Scale(const torch::Tensor& reward) override;
        void Reset() override {}
    public:
        std::optional<float> GetScalar(const std::string& key, int index = -1) const override;
    private:
        float scale_factor_;
        const std::optional<float> clip_;
    };

    class RunningStdScaler : public RewardScaler, virtual public DataExporterBase {
    public:
        RunningStdScaler(const std::optional<float>& clip = std::nullopt, float epsilon = 1e-8f, float post_scale = 1.0f);

        torch::Tensor Scale(const torch::Tensor& reward) override;
        void Reset() override;
    public:
        std::optional<float> GetScalar(const std::string& key, int index = -1) const override;
    private:
        // 設定
        const std::optional<float> clip_; ///< クリップ範囲
        const float epsilon_;             ///< ゼロ除算防止用の極小値
        const float post_scale_;

        // --- 逐次更新のための内部状態 ---
        int64_t count_;     ///< 観測したデータの個数 (N)
        double mean_;        ///< 現在の平均値 (μ)
        double m2_;          ///< 平均からの偏差の二乗和 (Σ(x - μ)^2)
    };

    // =============================================================
    // RewardScalarFactory
    // =============================================================

    struct RewardScalerConfig {
        bool use_clip = false;
        float clip_threshold = 10;

        float constant_scale = 1.0f;

        bool use_dynamic_scaling = true;
        float scaling_epsilon = 1e-8f;

        bool use_auto_post_scale = true;     ///< gammaを使って自動設定するか？
        float reference_q_std = 30.0f;        ///< 自動設定時のQ値の目標スケール (推奨: 3.0 ~ 30.0)
        float manual_post_scale = 1.0f;

        // dynamicなScalerの種類が増えたらそのアルゴリズムを指定する設定を追加
    };

    class RewardScalerFactory {
    public:
        RewardScalerFactory(const RewardScalerConfig& config);
        std::unique_ptr<RewardScaler> CreateRewardScaler(float gamma) const;
    private:
        RewardScalerConfig config_;
    };


}	// namespace anet::rl