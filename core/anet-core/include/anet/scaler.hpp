// anet/scaler.hpp
#pragma once

#include <cstdint>
#include <optional>
#include <memory>
#include <vector>
#include <torch/torch.h>
#include "anet/common.hpp"
#include "anet/config.hpp"
#include "anet/rl.hpp"

namespace anet::rl {

    // =============================================================
    // ObservationNormalizer
    // =============================================================

    class ObservationNormalizer : virtual public DataExporter {
    public:
        virtual torch::Tensor Normalize(const torch::Tensor& obs) const = 0;
        virtual torch::Tensor NormalizeAndUpdateStats(const torch::Tensor& obs) = 0;
        virtual void Reset() = 0;

        virtual ~ObservationNormalizer() = default;
    public:
        static constexpr const char* kKeyPrefix = "obs_norm.";

        static constexpr const char* kKeyCount = "obs_norm.count";
        static constexpr const char* kKeyMeanMean = "obs_norm.mean_mean"; // 平均の平均（ドリフト監視）
        static constexpr const char* kKeyStdMean = "obs_norm.std_mean";   // 標準偏差の平均（探索範囲監視）
        static constexpr const char* kKeyClipRatio = "obs_norm.clip_ratio"; // クリップ率
    };

    struct ObservationNormalizerConfig {
        bool pass_through = false;
        bool use_clipping = true;
        float clip_range = 10.0f; // [-10, 10] にクリップ
        bool use_dynamic_scaling = true;
        float epsilon = 1e-4f;    // ゼロ除算防止
        std::vector<float> constant_mean; // 指定なし(空)なら 0
        std::vector<float> constant_std;  // 指定なし(空)なら 1
        bool use_raw_clipping = true;
        float raw_clip_range = 5.0f;

        // dynamicなScalerの種類が増えたらそのアルゴリズムを指定する設定を追加
    };

    class ObservationNormalizerFactory {
    public:
        ObservationNormalizerFactory(const ObservationNormalizerConfig& config);
        std::shared_ptr<ObservationNormalizer> CreateObservationNormalizer(const StateSpec& state_spec) const;
    private:
        ObservationNormalizerConfig config_;
    };

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

        static constexpr const char* kKeyCount = "reward_scaler.count";
        static constexpr const char* kKeyMean = "reward_scaler.mean";
        static constexpr const char* kKeyStd = "reward_scaler.std";
        static constexpr const char* kKeyScale = "reward_scaler.scale";
        static constexpr const char* kKeyClipRatio = "reward_scaler.clip_ratio";
    };

    // =============================================================
    // RewardScalarFactory
    // =============================================================

    struct RewardScalerConfig {
        bool use_clipping = false;
        float clip_range = 10;

        float constant_scale = 1.0f;

        bool use_dynamic_scaling = true;

        float epsilon = 1e-8f;
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