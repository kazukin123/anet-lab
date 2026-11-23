// anet/scaler.hpp
#pragma once

#include <cstdint>
#include <optional>
#include <memory>
#include <vector>
#include <torch/torch.h>
#include "anet/common.hpp"
#include "anet/config.hpp"
#include "anet/log.hpp"
#include "anet/rl.hpp"
#include "anet/serialize.hpp"

namespace anet::rl {

    // =============================================================
    // ObservationNormalizer
    // =============================================================

    class ObservationNormalizer : virtual public Module, public anet::Serializable {
    public:
        static constexpr const char* kKeyPrefix = "obs_norm.";

        static constexpr const char* kKeyCount = "obs_norm.count";
        static constexpr const char* kKeyMeanMean = "obs_norm.mean_mean";   ///< 平均の平均（ドリフト監視）
        static constexpr const char* kKeyStdMean = "obs_norm.std_mean";     ///< 標準偏差の平均（探索範囲監視）
        static constexpr const char* kKeyMean = "obs_norm.mean";            ///< 平均の平均（ドリフト監視）
        static constexpr const char* kKeyStd = "obs_norm.std";              ///< 標準偏差の平均（探索範囲監視）
        static constexpr const char* kKeyClipRatio = "obs_norm.clip_ratio"; ///< クリップ率
        static constexpr const char* kKeyRobustOutlierRatio = "obs_norm.robust_outlier_ratio"; ///< Robust Updateで除外された割合
    public:
        virtual anet::TensorDict Normalize(const anet::TensorDict& obs) const = 0;
        virtual anet::TensorDict NormalizeAndUpdateStats(const anet::TensorDict& obs) = 0;
        /// 統計だけを更新し、正規化と clip の割合の計測を行わない。
        virtual void UpdateStats(const anet::TensorDict& obs) = 0;
        virtual void Reset() = 0;

        virtual ~ObservationNormalizer() = default;
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
        ConfigFullKeyTable full_keys;

        // dynamicなScalerの種類が増えたらそのアルゴリズムを指定する設定を追加
    };

    /// 正規化設定が所有する値域を検証する。
    void ValidateObservationNormalizerConfig(const ObservationNormalizerConfig& config);

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

    class RewardScaler : virtual public Module, public anet::Serializable {
    public:
        static constexpr const char* kKeyPrefix = "reward_scaler.";

        static constexpr const char* kKeyCount = "reward_scaler.count";
        static constexpr const char* kKeyMean = "reward_scaler.mean";
        static constexpr const char* kKeyStd = "reward_scaler.std";
        static constexpr const char* kKeyScale = "reward_scaler.scale";
        static constexpr const char* kKeyClipRatio = "reward_scaler.clip_ratio";
    public:
        /**
         * @brief 報酬(Batch)を受け取り、統計を更新し、スケールされたTensorを返す
         * @param reward [N] の形状を持つ Tensor (CPU or CUDA、kFloat32)
         * @return [N] スケール済みの Tensor
         */
        virtual torch::Tensor Scale(const torch::Tensor& reward) = 0;
        /// 現在の統計で変換し、統計とclip率を更新しない。
        virtual torch::Tensor ScaleWithoutUpdate(const torch::Tensor& reward) const = 0;
        virtual void Reset() = 0;
        virtual std::string StateKind() const = 0;
        static constexpr char kConstantState[] = "constant";
        static constexpr char kRewardStdState[] = "reward_std";
        static constexpr char kReturnStdState[] = "return_std";

        virtual ~RewardScaler() = default;
    };

    // =============================================================
    // RewardScalarFactory
    // =============================================================

    struct RewardScalerConfig {
        bool use_clipping = false;
        float clip_range = 10;

        float constant_scale = 1.0f;

        bool use_dynamic_scaling = true;
        std::string dynamic_mode = RewardScaler::kRewardStdState;

        float epsilon = 1e-8f;
        bool use_auto_post_scale = true;     ///< gammaを使って自動設定するか？
        float reference_q_std = 30.0f;        ///< 自動設定時のQ値の目標スケール (推奨: 3.0 ~ 30.0)
        float manual_post_scale = 1.0f;
        ConfigFullKeyTable full_keys;

        // dynamicなScalerの種類が増えたらそのアルゴリズムを指定する設定を追加
    };

    class RewardScalerFactory {
    public:
        RewardScalerFactory(const RewardScalerConfig& config, std::string prefix = "reward_scaler");
        std::unique_ptr<RewardScaler> CreateRewardScaler(float gamma, int64_t num_lanes,
            const std::string& gamma_key) const;
    private:
        RewardScalerConfig config_;
        std::string prefix_;
    };

    void ValidateRewardScalerConfig(const RewardScalerConfig& config, bool used = true);


}	// namespace anet::rl

namespace anet {
    template<> struct ConfigReader<rl::RewardScalerConfig> {
        static constexpr bool kEnabled = true;
        static void Read(Config& owner, const ConfigData& data, const std::string& key, rl::RewardScalerConfig& value, bool used = true)
        {
            ReadImpl(owner, data, key, value, used, true);
        }
        /// 型と列挙を検証して読む。組み合わせは、全関連値が揃った時点でValidateRewardScalerConfigに渡す。
        static void ReadFields(Config& owner, const ConfigData& data, const std::string& key, rl::RewardScalerConfig& value, bool used = true)
        {
            ReadImpl(owner, data, key, value, used, false);
        }
    private:
        static void ReadImpl(Config& owner, const ConfigData& data, const std::string& key,
            rl::RewardScalerConfig& value, bool used, bool validate_combinations)
        {
            // 接頭辞だけが異なる外的・内的の設定を、同じ型の口で解釈する。
            const auto defaults = value;
            owner.ReadSubConfig(data, key, "use_clipping", value.use_clipping);
            owner.ReadSubConfig(data, key, "clip_range", value.clip_range);
            owner.ReadSubConfig(data, key, "constant_scale", value.constant_scale);
            owner.ReadSubConfig(data, key, "use_dynamic_scaling", value.use_dynamic_scaling);
            owner.ReadSubConfig(data, key, "dynamic_mode", value.dynamic_mode);
            owner.ReadSubConfig(data, key, "epsilon", value.epsilon);
            owner.ReadSubConfig(data, key, "use_auto_post_scale", value.use_auto_post_scale);
            owner.ReadSubConfig(data, key, "reference_q_std", value.reference_q_std);
            owner.ReadSubConfig(data, key, "manual_post_scale", value.manual_post_scale);
            value.full_keys = owner.GetFullKeys().Extract(key);
            // usedは休眠時の通知に保ち、段階読み込みでは組み合わせだけを後へ残す。
            rl::ValidateRewardScalerConfig(value, used && validate_combinations);
            const auto report = [&](const std::string& field, const auto& actual, const auto& default_value, bool field_used) {
                if (!field_used && owner.HasConfigValue(data, key + "." + field) && actual != default_value) {
                    const auto full_key = value.full_keys.GetFullKey(field);
                    anet::log::warn_once(full_key) << "Ignored " << full_key << "=" << anet::json(actual).dump()
                        << "; default=" << anet::json(default_value).dump()
                        << "; reason=" << (used ? value.full_keys.GetFullKey("use_dynamic_scaling") + "=false" : "reward scaler inactive") << ".";
                }
            };
            report("use_clipping", value.use_clipping, defaults.use_clipping, used);
            report("clip_range", value.clip_range, defaults.clip_range, used);
            report("constant_scale", value.constant_scale, defaults.constant_scale, used);
            report("use_dynamic_scaling", value.use_dynamic_scaling, defaults.use_dynamic_scaling, used);
            report("dynamic_mode", value.dynamic_mode, defaults.dynamic_mode, used && value.use_dynamic_scaling);
            report("epsilon", value.epsilon, defaults.epsilon, used);
            report("use_auto_post_scale", value.use_auto_post_scale, defaults.use_auto_post_scale, used);
            report("reference_q_std", value.reference_q_std, defaults.reference_q_std, used);
            report("manual_post_scale", value.manual_post_scale, defaults.manual_post_scale, used);
        }
    };
}
