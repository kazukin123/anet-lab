// reward_scaler.cpp

#include "anet/reward_scaler.hpp"
#include "anet/tensor_check.hpp"
#include "anet/profile.hpp"

using namespace anet::rl;


// =============================================================
// ConstantRewardScaler
// =============================================================

ConstantRewardScaler::ConstantRewardScaler(float scale_factor, const std::optional<float>& clip)
    : clip_(clip), scale_factor_(scale_factor)
{
    ;
}

torch::Tensor ConstantRewardScaler::Scale(const torch::Tensor& reward) 
{
    auto scaled = reward * scale_factor_;
    if (clip_.has_value())
        return torch::clamp(scaled, -*clip_, *clip_);
    return scaled;
}

std::optional<float> ConstantRewardScaler::GetScalar(const std::string& key, int index) const
{
    if (key == kKeyScale) {
        return scale_factor_;
    }
    if (key == kKeyCount) return std::numeric_limits<float>::quiet_NaN();
    if (key == kKeyMean) return std::numeric_limits<float>::quiet_NaN();
    if (key == kKeyM2) return std::numeric_limits<float>::quiet_NaN();
    if (key == kKeyStd) return std::numeric_limits<float>::quiet_NaN();

    return std::nullopt;
}

// =============================================================
// RunningStdScaler
// =============================================================

RunningStdScaler::RunningStdScaler(const std::optional<float>& clip, float epsilon, float post_scale)
    : clip_(clip), epsilon_(epsilon), post_scale_(post_scale)
{
    Reset();
}

void RunningStdScaler::Reset()
{
    count_ = 0;
    mean_ = 0.0f;
    m2_ = 0.0f;
}

torch::Tensor RunningStdScaler::Scale(const torch::Tensor& reward)
{
    ProfileRange r("RunningStdScaler::Scale");

    ANET_CHECK_SHAPE(reward, { ANET_SHAPE_ANY });
    ANET_CHECK_DTYPE(reward, torch::kFloat32);

    // reward: [N] (並列環境数分の報酬)

    // ---- バッチ内統計 ----

    // バッチ統計量
    int64_t batch_count = reward.numel();
    if (batch_count == 0) return reward;    // 内容がないようならそのまま返す

    // バッチ内の平均 (scalar)
    //  unbiased=false (標本分散ではなく、単純な二乗平均) を指定することで、M2 = Variance * N として直接復元できる
    auto [batch_var_t, batch_mean_t] = torch::var_mean(reward, /*unbiased=*/false);
    double batch_mean_val = batch_mean_t.item<double>();
    double batch_var_val = batch_var_t.item<double>();
    double batch_m2_val = batch_var_val * static_cast<double>(batch_count); // M2 (二乗和) の復元: Var = M2 / N  =>  M2 = Var * N

    // ---- 全体統計量へのマージ（double で計算することで桁落ちを防ぐ） ----
    // 既存の統計セットA (count_, mean_, m2_) と
    // 新しい統計セットB (batch_count, batch_mean, batch_m2) を結合する公式

    int64_t new_count = count_ + batch_count;

    // 平均の差分
    double delta = batch_mean_val - mean_;

    // 新しい平均の計算
    double new_mean = mean_ + delta * static_cast<double>(batch_count) / static_cast<double>(new_count);

    // 新しいM2の計算
    double new_m2 = m2_ + batch_m2_val + (delta * delta)
        * (static_cast<double>(count_) * static_cast<double>(batch_count) / static_cast<double>(new_count));

    // 内部状態更新
    count_ = new_count;
    mean_ = new_mean;
    m2_ = new_m2;

    // ---- スケーリング ----

    if (count_ < 2) return reward; // 分散定義不可

    // 不偏分散と標準偏差
    double var = m2_ / static_cast<double>(count_ - 1);
    double stdev = std::sqrt(var) + static_cast<double>(epsilon_);

    // 正規化 (Tensor / scalar)
    torch::Tensor scaled = reward / static_cast<float>(stdev);

    // ポストスケーリング
    if (post_scale_ != 1.0f)
        scaled *= post_scale_;

    // --- クリッピング ---
    if (clip_.has_value()) return torch::clamp(scaled, -*clip_, *clip_);
    return scaled;
}

std::optional<float> RunningStdScaler::GetScalar(const std::string& key, int index) const
{
    if (key == kKeyCount) return static_cast<float>(count_);
    if (key == kKeyMean) return static_cast<float>(mean_);
    if (key == kKeyM2) return static_cast<float>(m2_);
    if (key == kKeyStd) {
        if (count_ < 2) return 0.0f; // 分散定義不可
        double var = m2_ / static_cast<double>(count_ - 1);
        double stdev = std::sqrt(var) + static_cast<double>(epsilon_);
        return static_cast<float>(stdev);
    }
    if (key == kKeyScale) {
        if (count_ < 2) return 1.0f; // 分散定義不可
        double var = m2_ / static_cast<double>(count_ - 1);
        double stdev = std::sqrt(var) + static_cast<double>(epsilon_);
        float scale = 1.0f / static_cast<float>(stdev) * post_scale_;
        return scale;
    }

    return std::nullopt;
}


// =============================================================
// RewardScalarFactory
// =============================================================

RewardScalerFactory::RewardScalerFactory(const RewardScalerConfig& config)
    : config_(config)
{
    ;
}

std::unique_ptr<RewardScaler> RewardScalerFactory::CreateRewardScaler(float gamma) const
{
    std::optional<float> clip;
    if (config_.use_clip)
        clip = config_.clip_threshold;

    // post_scale の決定ロジック
    float final_post_scale = 1.0f;
    if (config_.use_auto_post_scale) {
        // 理論式: post_scale = Q_target * (1 - gamma)
        final_post_scale = config_.reference_q_std * (1.0f - gamma);
    } else {
        final_post_scale = config_.manual_post_scale;
    }

    // Scaler生成
    if (!config_.use_dynamic_scaling)
        return std::make_unique<ConstantRewardScaler>(config_.constant_scale * final_post_scale, clip);
    return std::make_unique<RunningStdScaler>(clip, config_.scaling_epsilon, final_post_scale);
}
