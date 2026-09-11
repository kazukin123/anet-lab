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
    ANET_LOG_DEBUG("batch_input=" << anet::ToString(batch_input));

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
    int64_t batch_count = 0;
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

void RunningMeanStd::updateFromBatchStats(const torch::Tensor& batch_mean, const torch::Tensor& batch_m2, int64_t batch_count)
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

ConstantSingleTensorNormalizer::ConstantSingleTensorNormalizer(bool pass_through,
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

void ConstantSingleTensorNormalizer::Reset()
{
    last_clip_ratio_ = 0.0f;
}

std::pair<torch::Tensor, float>
ConstantSingleTensorNormalizer::normalizeInternal(const torch::Tensor& obs) const
{
    ANET_PROFILE_FUNC();
    
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

torch::Tensor ConstantSingleTensorNormalizer::Normalize(const torch::Tensor& obs) const
{
    auto result = normalizeInternal(obs);
    return result.first;
}

torch::Tensor ConstantSingleTensorNormalizer::NormalizeAndUpdateStats(const torch::Tensor& obs)
{
    auto result = normalizeInternal(obs);
    this->last_clip_ratio_ = result.second;
    return result.first;
}

std::optional<float> ConstantSingleTensorNormalizer::GetScalar(const std::string& key, int64_t index) const
{
    if (key == ObservationNormalizer::kKeyCount) return std::numeric_limits<float>::quiet_NaN();
    if (key == ObservationNormalizer::kKeyMeanMean) return std::numeric_limits<float>::quiet_NaN();
    if (key == ObservationNormalizer::kKeyStdMean) return std::numeric_limits<float>::quiet_NaN();
    if (key == ObservationNormalizer::kKeyClipRatio) return last_clip_ratio_;
    if (key == ObservationNormalizer::kKeyRobustOutlierRatio) return std::numeric_limits<float>::quiet_NaN();

    return std::nullopt;
}

std::optional<torch::Tensor> ConstantSingleTensorNormalizer::GetTensor(const std::string& key, int64_t index) const
{
    //if (key == kKeyMean) return mean_;
    //if (key == kKeyStd) return std_;

    return std::nullopt;
}

// =============================================================
// RunningStdSingleTensorNormalizer
// =============================================================

RunningStdSingleTensorNormalizer::RunningStdSingleTensorNormalizer(
    const std::optional<float>& clip_range,
    const std::vector<int64_t>& shape, float epsilon,
    bool use_robust_update, int robust_warmup_count, float robust_std_threshold,
    int post_process_type, float post_process_threshold, bool use_centering)
    : clip_range_(clip_range), shape_(shape)
    , use_robust_update_(use_robust_update)
    , use_centering_(use_centering)
    , robust_warmup_count_(robust_warmup_count)
    , robust_std_threshold_(robust_std_threshold)
    , post_process_type_(post_process_type)
    , post_process_threshold_(post_process_threshold)
    , stats_(shape, epsilon) // Vectorモードで初期化
{
    LOG::info() << "ObservationNormalizer initialized with. shape=" << shape_;
}

void RunningStdSingleTensorNormalizer::Reset()
{
    stats_.Reset();
    last_clip_ratio_ = 0.0f;
    last_outlier_ratio_ = 0.0f;
}

std::pair<torch::Tensor, float>
RunningStdSingleTensorNormalizer::normalizeInternal(const torch::Tensor& obs) const
{
    ANET_PROFILE_FUNC();

    // 型チェック
    ANET_ASSERT_DTYPE(obs, torch::kFloat32);

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
    torch::Tensor std = stats_.GetStd().to(device).to(torch::kFloat32);

    // 正規化 (Broadcasting: [B, state_dim...] -> [ state_dim...])
    torch::Tensor normalized;
    if (use_centering_) {
        // (x - mean) / std
        auto mean = stats_.GetMean().to(device).to(torch::kFloat32);
        normalized = (obs - mean) / std;
    } else {
        // x / std → 「スケール」だけが揃い、「原点」は保存される
        normalized = obs / std;
    }

    // 後処理
    switch (post_process_type_) {
    case kPostProcessSymLog: {
        // 式: sign(x) * ln(|x| + 1)

        // scaleがあれば適用: symlog(x / scale) * scale
        float scale = (post_process_threshold_ > 0) ? post_process_threshold_ : 1.0f;

        // 正規化後の値(normalized)自体がすでに「σ単位」になっているので、
        // そのまま SymLog に通しても良いが
        // 「10σまではリニアに扱いたい」なら scale=10.0 に設定

        // 実装: y = scale * symlog(x / scale)
        auto scaled_in = normalized / scale;
        normalized = torch::sign(scaled_in) * torch::log1p(torch::abs(scaled_in)) * scale;
        break;
    }
    case kPostProcessTanh: {
        float scale = (post_process_threshold_ > 0) ? post_process_threshold_ : 1.0f;
        normalized = torch::tanh(normalized / scale) * scale;
        break;
    }
    case kPostProcessSoftsign: {
        float limit = (post_process_threshold_ > 0) ? post_process_threshold_ : 1.0f;
        normalized = normalized / (1.0f + normalized.abs() / limit);
        break;
    }
    case kPostProcessNone:
        break;
    default:
        LOG::warn() << "Unknown squash_type.";
        break;
    }

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

torch::Tensor RunningStdSingleTensorNormalizer::Normalize(const torch::Tensor& obs) const
{
    auto result = normalizeInternal(obs);
    return result.first;
}

torch::Tensor RunningStdSingleTensorNormalizer::NormalizeAndUpdateStats(const torch::Tensor& obs)
{
    // 統計更新の前に、入力値を常識的な範囲にクリップする
    // これにより、暴発値で統計（分散）が爆発するのを防ぐ
    if (use_robust_update_) {
        // 現在の平均・標準偏差を取得 (デバイス合わせる)
        auto device = obs.device();
        auto mean = stats_.GetMean().to(device).to(torch::kFloat32);
        auto std = stats_.GetStd().to(device).to(torch::kFloat32);

        // 偏差 (z-score の分子)
        auto deviation = (obs - mean).abs();

        // 閾値: N * std
        auto threshold = robust_std_threshold_ * std;

        // マスク作成: 正常な値だけ true
        torch::Tensor mask;
        if (stats_.GetCount() < robust_warmup_count_) {
            // 初期は無条件で更新して統計を安定させる
            stats_.Update(obs);
            last_outlier_ratio_ = 0.0f;
        } else {
            // 異常値判定:thresholdより小さい(正常)、または thresholdが小さすぎる(初期)なら採用
            auto is_normal = (deviation <= threshold);

            // 正常なデータのみを抽出してUpdateに回す
            // ※ 部分更新の実装は RunningMeanStd 側で対応が必要だが、
            //    簡易的には「バッチ内の平均外れ値率」を見て、健全なバッチだけUpdateする手もある。
            //    厳密にやるなら以下のようにマスクしたデータを渡す。

            if (is_normal.all().item<bool>()) {
                // 全部正常ならそのまま更新
                stats_.Update(obs);
            } else {

                // 一部異常なら、正常な行だけ選んで更新（要: RunningMeanStdが可変長バッチ対応していること）
                // obs[is_normal] でフィルタリングすると形状が変わる(Flattenされる)が、
                // RunningMeanStd は [N, F] or [N] なので、フィルタリングして [M, F] にして渡せばOK

                // ただし、obsが [B, F] で、F次元のうち1つでも異常ならそのサンプルの更新を諦めるか、
                // 次元ごとに更新するかという問題がある。
                // ObservationNormalizerは通常「次元ごとに独立」なので、次元ごとにマスクして更新するのが正しいが実装コストが高い。

                // 【現実解】: バッチ内のサンプル単位（行単位）で、「全次元が正常な行」だけ更新に使う。
                // (obs - mean).abs() > threshold  -> [B, F]
                // (outliers).any(dim=1) -> [B] (異常を含む行)


                // 外れ値判定 (deviation > threshold)
                auto is_outlier = (deviation > threshold);

                // 外れ値率を記録 (全要素に対する割合)
                this->last_outlier_ratio_ = is_outlier.to(torch::kFloat32).mean().item<float>();

                // 全次元が正常な行だけを抽出
                auto is_outlier_row = is_outlier.view({ obs.size(0), -1 }).any(1); // [B] -> trueならその行に異常あり
                auto valid_indices = (~is_outlier_row).nonzero().squeeze(1);

                // DEBUG
                if (is_outlier_row.any().item<bool>()) {
                    std::stringstream ss;

                    // 異常行のインデックスを取得
                    auto row_indices = is_outlier_row.nonzero(); // [K, 1]
                    int num_logs = std::min((int)row_indices.size(0), 3); // 最大3件まで表示

                    ss << "Outlier Row Detected! (Showing " << num_logs << " samples):\n";

                    for (int i = 0; i < num_logs; ++i) {
                        int r_idx = row_indices[i][0].item<int>();

                        // 行全体の生データ
                        auto raw_row = obs[r_idx]; // [Feature]

                        // Tensor標準出力で見やすく表示
                        ss << " [Sample " << r_idx << "]\n";
                        ss << "  Full Obs: " << raw_row << "\n";

                        // どの次元が引っかかったか詳細を列挙
                        ss << "  Details:\n";
                        for (int d = 0; d < raw_row.size(0); ++d) {
                            float dev_val = deviation[r_idx][d].item<float>();
                            float th_val = threshold[d].item<float>();

                            if (dev_val > th_val) {
                                float val = raw_row[d].item<float>();
                                float mu = mean[d].item<float>();
                                float sigma = std[d].item<float>();

                                ss << "   - Dim[" << d << "] Val=" << val
                                    << " (Mean=" << mu << ", Std=" << sigma
                                    << ", Dev=" << dev_val << " > Th=" << th_val << ")\n";
                            }
                        }
                    }
                    LOG::warn() << ss.str();
                }

                // 全正常な行があればそれらで統計更新
                if (valid_indices.numel() > 0) {
                    stats_.Update(obs.index_select(0, valid_indices));
                }

                // 全部異常なら更新スキップ
            }
        }
    } else {
        stats_.Update(obs);
        last_outlier_ratio_ = 0.0f;
    }
    auto result = normalizeInternal(obs);
    this->last_clip_ratio_ = result.second;
    return result.first;
} 

std::optional<float> RunningStdSingleTensorNormalizer::GetScalar(const std::string& key, int64_t index) const
{
    if (key == ObservationNormalizer::kKeyCount) return static_cast<float>(stats_.GetCount());
    if (key == ObservationNormalizer::kKeyMeanMean) return static_cast<float>(stats_.GetMeanMean());
    if (key == ObservationNormalizer::kKeyStdMean) return static_cast<float>(stats_.GetStdMean());
    if (key == ObservationNormalizer::kKeyClipRatio) return last_clip_ratio_;
    if (key == ObservationNormalizer::kKeyRobustOutlierRatio) return last_outlier_ratio_;

    return std::nullopt;
}

std::optional<torch::Tensor> RunningStdSingleTensorNormalizer::GetTensor(const std::string& key, int64_t index) const
{
    if (key == ObservationNormalizer::kKeyMean) return stats_.GetMean();
    if (key == ObservationNormalizer::kKeyStd) return stats_.GetStd();

    return std::nullopt;
}


// =============================================================
// DictObservationNormalizer
// =============================================================

void DictObservationNormalizer::AddNormalizer(const std::string& key, std::shared_ptr<SingleTensorNormalizer> normalizer)
{
    normalizers_[key] = normalizer;
}

anet::TensorDict DictObservationNormalizer::Normalize(const anet::TensorDict& obs) const
{
    ANET_PROFILE_FUNC();

    anet::TensorDict result;
    for (const auto& kv : obs) {
        auto it = normalizers_.find(kv.first);
        if (it != normalizers_.end()) {
            // Normalizerが登録されているKeyは処理
            result.Set(kv.first, it->second->Normalize(kv.second));
        } else {
            // 登録されていないKey (画像やカテゴリIDなど) はそのままPass-through
            result.Set(kv.first, kv.second);
        }
    }
    return result;
}

anet::TensorDict DictObservationNormalizer::NormalizeAndUpdateStats(const anet::TensorDict& obs)
{
    ANET_PROFILE_FUNC();

    anet::TensorDict result;
    for (const auto& kv : obs) {
        auto it = normalizers_.find(kv.first);
        if (it != normalizers_.end()) {
            result.Set(kv.first, it->second->NormalizeAndUpdateStats(kv.second));
        } else {
            result.Set(kv.first, kv.second);
        }
    }
    return result;
}

void DictObservationNormalizer::Reset()
{
    ANET_PROFILE_FUNC();

    for (auto& kv : normalizers_) {
        kv.second->Reset();
    }
}

std::optional<float> DictObservationNormalizer::GetScalar(const std::string& key, int64_t index) const
{
	/// @todo 異なる観測空間の統計値を強引に平均せず、max.やmean.で集計方法を指定可能とする

    if (normalizers_.empty()) return std::nullopt;

    float sum = 0.0f;
    int count = 0;
    for (const auto& kv : normalizers_) {
        auto val = kv.second->GetScalar(key, index);

        // 子のNormalizerが1つでも「知らないKey」と判定したら、全体も nullopt を返す
        if (!val.has_value()) {
            return std::nullopt;
        }

        // 統計更新
        if (!std::isnan(*val)) {
            sum += *val;
            count++;
        }
    }

    // 全てのNormalizerが値を返したが、すべてNaNだった場合のフェイルセーフ
    if (count == 0) return 0.0f;

    return sum / static_cast<float>(count);
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
    // Keyごとに独立したコンテナを生成
    auto dict_normalizer = std::make_shared<DictObservationNormalizer>();

    for (const auto& kv : state_spec.obs_spec) {
        const auto& key = kv.first;
        const auto& t_spec = kv.second;

        // 離散値は正規化してはいけないため、Float系のみNormalizerを作る
        if (!torch::isFloatingType(t_spec.dtype) || t_spec.num_classes > 0) {
            continue;
        }

        std::optional<float> clip;
        if (config_.use_clipping) clip = config_.clip_range;

        std::shared_ptr<SingleTensorNormalizer> single_norm;
        if (config_.pass_through || !config_.use_dynamic_scaling) {
            single_norm = std::make_shared<ConstantSingleTensorNormalizer>(
                config_.pass_through, t_spec.shape, clip, config_.constant_mean, config_.constant_std);
        } else {
            single_norm = std::make_shared<RunningStdSingleTensorNormalizer>(
                clip, t_spec.shape, config_.epsilon, config_.use_robust_update, config_.robust_warmup_count,
                config_.robust_std_threshold, config_.post_process_type, config_.post_process_threshold, config_.use_centering);
        }

        dict_normalizer->AddNormalizer(key, single_norm);
    }

    return dict_normalizer;
}

// =============================================================
// ConstantRewardScaler
// =============================================================

ConstantRewardScaler::ConstantRewardScaler(float scale_factor, const std::optional<float>& clip_range)
    : scale_factor_(scale_factor), clip_range_(clip_range)
    , last_clip_ratio_(std::numeric_limits<float>::quiet_NaN())
{
    ;
}

torch::Tensor ConstantRewardScaler::Scale(const torch::Tensor& reward)
{
    ANET_PROFILE_FUNC();

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
    // クリップ無効時も今回の適用率を確定し、未初期化値をメトリクスへ流さない。
    last_clip_ratio_ = 0.0f;
    return scaled;
}

std::optional<float> ConstantRewardScaler::GetScalar(const std::string& key, int64_t index) const
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
    ANET_PROFILE_FUNC();

    ANET_ASSERT_SHAPE(reward, { ANET_SHAPE_ANY });
    ANET_ASSERT_DTYPE(reward, torch::kFloat32);

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

std::optional<float> RunningStdRewardScaler::GetScalar(const std::string& key, int64_t index) const
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
