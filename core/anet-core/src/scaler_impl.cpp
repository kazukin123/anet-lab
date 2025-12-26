// scaler_impl.cpp
#include "scaler_impl.hpp"
#include "anet/scaler.hpp"
#include "anet/tensor_check.hpp"
#include "anet/profile.hpp"
#include "anet/log.hpp"

using namespace anet::rl;
namespace LOG = anet::log;


// =============================================================
// RunningMeanStd
// =============================================================

RunningMeanStd::RunningMeanStd(const std::vector<int64_t>& shape, double epsilon)
    : count_(0), epsilon_(epsilon)
{
    // double精度で初期化 (桁落ち防止)
    // shapeが空ならスカラ(0-dim tensor)、指定があればその形状
    mean_ = torch::zeros(shape, torch::TensorOptions().dtype(torch::kDouble));
    var_ = torch::ones(shape, torch::TensorOptions().dtype(torch::kDouble));

    // 内部計算用のM2 (平均からの二乗差和)
    m2_ = torch::zeros(shape, torch::TensorOptions().dtype(torch::kDouble));
}

void RunningMeanStd::Reset()
{
    count_ = 0;
    mean_.zero_();
    var_.fill_(1.0);
    m2_.zero_();
}

void RunningMeanStd::Update(const torch::Tensor& batch_input)
{
    // 入力が空なら何もしない
    if (batch_input.numel() == 0) return;

    // バッチ内の統計量を計算
    torch::Tensor batch_mean_t, batch_var_t;

    // 入力形状と内部形状を比較して、Reductionする次元を決定
    // 内部がScalar(0-dim)なら、入力の全次元をつぶす
    // 内部がVector(1-dim)なら、入力のdim0(Batch)をつぶす
    if (mean_.ndimension() == 0) {
        // Scalarモード (RewardScaler用)
        auto res = torch::var_mean(batch_input, /*unbiased=*/false);
        batch_var_t = std::get<0>(res);
        batch_mean_t = std::get<1>(res);
    } else {
        // Vectorモード (ObservationNormalizer用)
        // dim=0 (Batch方向) に沿って計算
        // batch_input: [B, F] -> mean: [F]
        auto res = torch::var_mean(batch_input, { 0 }, /*unbiased=*/false); // correction=0
        batch_var_t = std::get<0>(res);
        batch_mean_t = std::get<1>(res);
    }

    auto target_device = mean_.device(); // 通常はCPU

    // 計算精度確保のため double にキャスト
    batch_mean_t = batch_mean_t.to(target_device).to(torch::kDouble);
    batch_var_t = batch_var_t.to(target_device).to(torch::kDouble);

    // バッチサイズ (Obsの場合は行数)
    long long batch_count = 0;
    if (mean_.ndimension() == 0) {
        batch_count = batch_input.numel();
    } else {
        batch_count = batch_input.size(0);
    }

    // M2 (二乗和) の復元: Var = M2 / N => M2 = Var * N
    torch::Tensor batch_m2_t = batch_var_t * static_cast<double>(batch_count);

    // 全体統計量へのマージ (Parallel Welford Algorithm)
    updateFromBatchStats(batch_mean_t, batch_m2_t, batch_count);
}

void RunningMeanStd::updateFromBatchStats(const torch::Tensor& batch_mean, const torch::Tensor& batch_m2, int batch_count)
{
    int64_t new_count = count_ + batch_count;

    // 平均の差分 delta = batch_mean - mean_
    torch::Tensor delta = batch_mean - mean_;

    // 新しい平均
    // new_mean = mean_ + delta * (batch_count / new_count)
    torch::Tensor new_mean = mean_ + delta * (static_cast<double>(batch_count) / static_cast<double>(new_count));

    // 新しいM2
    // new_m2 = m2_ + batch_m2 + delta^2 * (count * batch_count / new_count)
    torch::Tensor new_m2 = m2_ + batch_m2 + delta.pow(2) * (static_cast<double>(count_) * static_cast<double>(batch_count) / static_cast<double>(new_count));

    // 新しい分散 (Population Variance)
    // count < 2 のときは分散定義不可だが、安全のため計算しておく
    torch::Tensor new_var = (new_count > 1) ? (new_m2 / static_cast<double>(new_count - 1)) : torch::ones_like(mean_);

    // 更新 (In-placeだと安全ではないのでコピー)
    count_ = new_count;
    mean_ = new_mean;
    m2_ = new_m2;
    var_ = new_var;
}


// =============================================================
// ObservationNormalizer
// =============================================================

ConstantObservationNormalizer::ConstantObservationNormalizer(bool pass_through,
    const std::vector<int64_t>& shape, const std::optional<float>& clip_range,
    const std::vector<float>& fixed_mean, const std::vector<float>& fixed_std)
    : pass_through_(pass_through), shape_(shape), clip_range_(clip_range)
{
    // 平均の初期化 (指定がなければ0)
    if (fixed_mean.empty()) {
        mean_ = torch::zeros(shape, torch::kFloat32);
    } else {
        mean_ = torch::tensor(fixed_mean, torch::kFloat32).view(shape);
    }

    // 分散の初期化 (指定がなければ1)
    if (fixed_std.empty()) {
        std_ = torch::ones(shape, torch::kFloat32);
    } else {
        std_ = torch::tensor(fixed_std, torch::kFloat32).view(shape);
    }
}

void ConstantObservationNormalizer::Reset()
{
    last_clip_ratio_ = 0.0f;
}

std::pair<torch::Tensor, float>
ConstantObservationNormalizer::normalizeInternal(const torch::Tensor& obs) const
{
    ProfileRange r("ConstantObservationNormalizer::Normalize");
    
    if (pass_through_) return { obs, 0.0f };

    // 形状チェック
    if (obs.ndimension() != static_cast<int64_t>(shape_.size() + 1)) {
        LOG::error() << "ConstantObsNorm: Dimension mismatch.";
        ANET_ASSERT_MSG(false, "ConstantObsNorm: Dimension mismatch.");
    }

    auto device = obs.device();

    // Device転送 (必要な時だけ)
    torch::Tensor mean_dev = mean_.to(device);
    torch::Tensor std_dev = std_.to(device);

    // 正規化: (x - mu) / sigma
    torch::Tensor normalized = (obs - mean_dev) / std_dev;

    // クリッピング
    float clipped = 0.0f;
    if (clip_range_.has_value()) {
        float clip_val = *clip_range_;
        auto out_of_range = (normalized.abs() > clip_val).to(torch::kFloat32);
        clipped = out_of_range.mean().item<float>();
        normalized = torch::clamp(normalized, -clip_val, clip_val);
    }

    return { normalized, clipped };
}

torch::Tensor ConstantObservationNormalizer::Normalize(const torch::Tensor& obs) const
{
    auto result = normalizeInternal(obs);
    return result.first;
}

torch::Tensor ConstantObservationNormalizer::NormalizeAndUpdateStats(const torch::Tensor& obs)
{
    auto result = normalizeInternal(obs);
    this->last_clip_ratio_ = result.second;
    return result.first;
}

std::optional<float> ConstantObservationNormalizer::GetScalar(const std::string& key, int index) const
{
    if (key == kKeyCount) return std::numeric_limits<float>::quiet_NaN();
    if (key == kKeyMeanMean) return std::numeric_limits<float>::quiet_NaN();
    if (key == kKeyStdMean) return std::numeric_limits<float>::quiet_NaN();
    if (key == kKeyClipRatio) return last_clip_ratio_;

    return std::nullopt;
}

std::optional<torch::Tensor> ConstantObservationNormalizer::GetTensor(const std::string& key, int index) const
{
    if (key == kKeyMean) return mean_;
    if (key == kKeyStd) return std_;

    return std::nullopt;
}

// =============================================================
// RunningStdObservationNormalizer
// =============================================================

RunningStdObservationNormalizer::RunningStdObservationNormalizer(
    const std::optional<float>& clip_range, const std::optional<float>& raw_clip_range,
    const std::vector<int64_t>& shape, float epsilon)
    : clip_range_(clip_range), raw_clip_range_(raw_clip_range), shape_(shape)
    , stats_(shape, epsilon) // Vectorモードで初期化
{
    LOG::info() << "ObservationNormalizer initialized with. shape=" << shape_;
}

void RunningStdObservationNormalizer::Reset()
{
    stats_.Reset();
    last_clip_ratio_ = 0.0f;
}

std::pair<torch::Tensor, float>
RunningStdObservationNormalizer::normalizeInternal(const torch::Tensor& obs) const
{
    ProfileRange r("RunningStdObservationNormalizer::Normalize");

    // 型チェック
    ANET_CHECK_DTYPE(obs, torch::kFloat32);

    // 次元数チェック: [Batch, state_dim...]
    if (obs.ndimension() != static_cast<int64_t>(shape_.size() + 1)) {
        LOG::error() << "ObservationNormalizer: Dimension mismatch. Expected "
            << (shape_.size() + 1) << " dims but got " << obs.ndimension();
        ANET_ASSERT_MSG(false, "Dimension mismatch in ObservationNormalizer");
        throw new std::runtime_error("Dimension mismatch in ObservationNormalizer");
    }

    // 各次元のサイズチェック (dim=0はBatchなのでスキップ)
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (obs.size(i + 1) != shape_[i]) {
            LOG::error() << "ObservationNormalizer: Shape mismatch at dim " << (i + 1)
                << ". Expected " << shape_[i] << " but got " << obs.size(i + 1);
            ANET_ASSERT_MSG(false, "Shape mismatch in ObservationNormalizer");
            throw new std::runtime_error("Shape mismatch in ObservationNormalizer");
        }
    }

    //  正規化処理
    auto device = obs.device();

    // 統計量取得 (Double CPU Tensor -> Float Device Tensor)
    torch::Tensor mean = stats_.GetMean().to(device).to(torch::kFloat32);
    torch::Tensor std = stats_.GetStd().to(device).to(torch::kFloat32);

    // 正規化 (Broadcasting: [B, state_dim...] -> [ state_dim...])
    torch::Tensor normalized = (obs - mean) / std;

    // クリッピング
    float clipped = 0.0f;
    if (clip_range_.has_value()) {
        float limit = *clip_range_;

        // 範囲外の要素数をカウント
        auto out_of_range = (normalized.abs() > limit).to(torch::kFloat32);
        clipped = out_of_range.mean().item<float>();

        normalized = torch::clamp(normalized, -limit, limit);
    }

    return { normalized, clipped };
}

torch::Tensor RunningStdObservationNormalizer::Normalize(const torch::Tensor& obs) const
{
    auto result = normalizeInternal(obs);
    return result.first;
}

torch::Tensor RunningStdObservationNormalizer::NormalizeAndUpdateStats(const torch::Tensor& obs)
{
    // 統計更新の前に、入力値を常識的な範囲にクリップする
    // これにより、暴発値で統計（分散）が爆発するのを防ぐ
    if (raw_clip_range_.has_value()) {
        float limit = *raw_clip_range_;
        auto clipped_obs = torch::clamp(obs, -limit, limit);
        stats_.Update(clipped_obs);
    } else {
        stats_.Update(obs);
    }
    auto result = normalizeInternal(obs);
    this->last_clip_ratio_ = result.second;
    return result.first;
}

std::optional<float> RunningStdObservationNormalizer::GetScalar(const std::string& key, int index) const
{
    if (key == kKeyCount) return static_cast<float>(stats_.GetCount());
    if (key == kKeyMeanMean) return static_cast<float>(stats_.GetMeanMean());
    if (key == kKeyStdMean) return static_cast<float>(stats_.GetStdMean());
    if (key == kKeyClipRatio) return last_clip_ratio_;

    return std::nullopt;
}

std::optional<torch::Tensor> RunningStdObservationNormalizer::GetTensor(const std::string& key, int index) const
{
    if (key == kKeyMean) return stats_.GetMean();
    if (key == kKeyStd) return stats_.GetStd();

    return std::nullopt;
}


// =============================================================
// ObservationNormalizerFactory
// =============================================================

ObservationNormalizerFactory::ObservationNormalizerFactory(const ObservationNormalizerConfig& config)
    : config_(config)
{
    ;
}

std::shared_ptr<ObservationNormalizer> ObservationNormalizerFactory::CreateObservationNormalizer(
    const StateSpec& state_spec) const
{
    auto shape = state_spec.shape;

    // Clip
    std::optional<float> clip;
    if (config_.use_clipping)
        clip = config_.clip_range;

    // Raw Clip
    std::optional<float> raw_clip;
    if (config_.use_raw_clipping) raw_clip = config_.raw_clip_range;

    // Scaler生成
    if (!config_.use_dynamic_scaling) {
        return std::make_shared<ConstantObservationNormalizer>(config_.pass_through, shape, clip, config_.constant_mean, config_.constant_std);
    } else {
        return std::make_shared<RunningStdObservationNormalizer>(clip, raw_clip, shape, config_.epsilon);
    }
}

// =============================================================
// ConstantRewardScaler
// =============================================================

ConstantRewardScaler::ConstantRewardScaler(float scale_factor, const std::optional<float>& clip_range)
    : clip_range_(clip_range), scale_factor_(scale_factor)
{
    ;
}

torch::Tensor ConstantRewardScaler::Scale(const torch::Tensor& reward)
{
    // スケーリング
    auto scaled = reward * scale_factor_;

    // クリッピング
    if (clip_range_.has_value()) {
        // クリップ判定
        auto out_of_range = (scaled.abs() > *clip_range_).to(torch::kFloat32);
        last_clip_ratio_ = out_of_range.mean().item<float>();

        // クリッピング実行
        return torch::clamp(scaled, -*clip_range_, *clip_range_);
    }
    return scaled;
}

std::optional<float> ConstantRewardScaler::GetScalar(const std::string& key, int index) const
{
    if (key == kKeyScale) return scale_factor_;
    if (key == kKeyCount) return std::numeric_limits<float>::quiet_NaN();
    if (key == kKeyMean) return std::numeric_limits<float>::quiet_NaN();
    if (key == kKeyStd) return std::numeric_limits<float>::quiet_NaN();
    if (key == kKeyClipRatio) return last_clip_ratio_;

    return std::nullopt;
}


// =============================================================
// RunningStdScaler
// =============================================================

RunningStdRewardScaler::RunningStdRewardScaler(const std::optional<float>& clip_range, float epsilon, float post_scale)
    : clip_range_(clip_range), epsilon_(epsilon), post_scale_(post_scale)
    , stats_({}, static_cast<double>(epsilon)) // shape={} でスカラーモード初期化
{
    Reset();
}

void RunningStdRewardScaler::Reset()
{
    stats_.Reset();
}

torch::Tensor RunningStdRewardScaler::Scale(const torch::Tensor& reward)
{
    ProfileRange r("RunningStdScaler::Scale");

    ANET_CHECK_SHAPE(reward, { ANET_SHAPE_ANY });
    ANET_CHECK_DTYPE(reward, torch::kFloat32);

    // reward: [N] (並列環境数分の報酬)

    // ---- 統計更新 (RunningMeanStdに丸投げ)
    stats_.Update(reward);

    // ---- スケーリング

    // RewardScalerでは、平均は引かずに「分散(std)」だけで割るのが一般的
    // (平均を引くと、疎な報酬環境で0の報酬がマイナスになったりして意味が変わるため)

    // 現在の標準偏差を取得 (Double Tensor)
    torch::Tensor stdev_t = stats_.GetStd();
    float stdev_val = static_cast<float>(stdev_t.item<double>());


    // 正規化
    torch::Tensor scaled = reward / stdev_val;

    // ポストスケーリング
    if (post_scale_ != 1.0f)
        scaled *= post_scale_;

    // クリッピング
    if (clip_range_.has_value()) {
        // クリップ判定
        auto out_of_range = (scaled.abs() > *clip_range_).to(torch::kFloat32);
        last_clip_ratio_ = out_of_range.mean().item<float>();

        // クリッピング実行
        return torch::clamp(scaled, -*clip_range_, *clip_range_);
    }
    return scaled;
}

std::optional<float> RunningStdRewardScaler::GetScalar(const std::string& key, int index) const
{
    // stats_ から情報を取得
    if (key == kKeyCount) return static_cast<float>(stats_.GetCount());
    if (key == kKeyMean)  return static_cast<float>(stats_.GetMeanMean());
    if (key == kKeyClipRatio) return last_clip_ratio_;

    if (key == kKeyStd) {
        if (stats_.GetCount() < 2) return 0.0f;
        return static_cast<float>(stats_.GetStd().item<double>());
    }
    if (key == kKeyScale) {
        if (stats_.GetCount() < 2) return 1.0f;
        double stdev = stats_.GetStd().item<double>();
        return static_cast<float>(1.0 / stdev * static_cast<double>(post_scale_));
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
    if (config_.use_clipping)
        clip = config_.clip_range;

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
    return std::make_unique<RunningStdRewardScaler>(clip, config_.epsilon, final_post_scale);
}
