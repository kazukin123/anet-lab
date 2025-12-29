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

    class ObservationNormalizer : virtual public Module {
    public:
        virtual torch::Tensor Normalize(const torch::Tensor& obs) const = 0;
        virtual torch::Tensor NormalizeAndUpdateStats(const torch::Tensor& obs) = 0;
        virtual void Reset() = 0;

        virtual ~ObservationNormalizer() = default;
    public:
        static constexpr const char* kKeyPrefix = "obs_norm.";

        static constexpr const char* kKeyCount = "obs_norm.count";
        static constexpr const char* kKeyMeanMean = "obs_norm.mean_mean";   ///< 平均の平均（ドリフト監視）
        static constexpr const char* kKeyStdMean = "obs_norm.std_mean";     ///< 標準偏差の平均（探索範囲監視）
        static constexpr const char* kKeyMean = "obs_norm.mean";            ///< 平均の平均（ドリフト監視）
        static constexpr const char* kKeyStd = "obs_norm.std";              ///< 標準偏差の平均（探索範囲監視）
        static constexpr const char* kKeyClipRatio = "obs_norm.clip_ratio"; ///< クリップ率
        static constexpr const char* kKeyRobustOutlierRatio = "obs_norm.robust_outlier_ratio"; ///< Robust Updateで除外された割合
    };
    
    static constexpr const int kPostProcessNone = 0;
    static constexpr const int kPostProcessSymLog = 1;
    static constexpr const int kPostProcessTanh = 2;
    static constexpr const int kPostProcessSoftsign = 3;

    struct ObservationNormalizerConfig {
        bool pass_through = false;          ///< 生のObservationをそのまま使う。Clippingも無効。

        bool use_clipping = true;           ///< 最後にClippingするならtrue
        float clip_range = 10.0f;           ///

        std::vector<float> constant_mean;   ///< use_dynamic_scaling=falseの場合向けの次元毎の固定値。指定なし(空)なら0
        std::vector<float> constant_std;    ///< use_dynamic_scaling=falseの場合向けの次元毎の固定値。指定なし(空)なら1

        bool use_dynamic_scaling = true;    ///< 実行中にOBSの分散や平均の統計から動的にスケールするならtrue
        bool use_centering = false;         ///< 平均値を引く補正をするかどうか (零点に意味がある場合はfalse推奨）
        float epsilon = 1.0f;               ///< ゼロ除算防止＆最小分散値
        bool use_robust_update = true;      ///< 標準偏差が閾値より大きい異常値を統計から自動除外するならtrue
        int robust_warmup_count = 200;      ///< 開始直後から落ち着くまで無条件で統計に反映するサンプル数
        float robust_std_threshold = 5.0f;  ///< この分散値より大きい値を統計から自動除外
        int post_process_type = kPostProcessSymLog;
        float post_process_threshold = 10.0f;

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

    class RewardScaler : virtual public Module {
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