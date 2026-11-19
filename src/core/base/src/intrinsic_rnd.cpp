#include "intrinsic_rnd.hpp"
#include <sstream>
#include <cmath>
#include <limits>
#include <ATen/CPUGeneratorImpl.h>
#include "anet/profile.hpp"
#include "anet/nn_head.hpp"

using namespace anet::rl::intrinsic;

constexpr char kTargetSeed[] = "intrinsic.target";
constexpr char kPredictorSeed[] = "intrinsic.predictor";
constexpr char kOutput[] = "novelty";
constexpr char kError[] = "error";
constexpr char kNormalizedError[] = "normalized_error";
constexpr char kErrorMean[] = "error_mean";
constexpr char kErrorStd[] = "error_std";
constexpr char kErrorStep[] = "error_step";
constexpr char kClipRatio[] = "clip_ratio";
constexpr char kPredictorLoss[] = "predictor_loss";
constexpr char kTargetFeatureVar[] = "target_feature_var";
constexpr char kTargetFeatureMax[] = "target_feature_max";
constexpr char kCheckpointFormat[] = "RndNoveltyEstimator-3";
constexpr char kEmaCheckpointFormat[] = "RndNoveltyEstimator-4";
constexpr char kErrorStatisticsUpdateProfile[] = "RndErrorStatistics::Update";
constexpr char kEstimateNormalizeProfile[] = "RndNoveltyEstimator::Estimate.normalize";
constexpr char kEstimateStatisticsProfile[] = "RndNoveltyEstimator::Estimate.statistics";

namespace {
    // LinearHeadの最終軸射影がlaneごとの1ベクトルになる条件を構築時に確定する。
    class RndHeadFactory final : public anet::nn::LinearHeadFactory {
    public:
        using anet::nn::LinearHeadFactory::LinearHeadFactory;
        std::shared_ptr<anet::nn::NetworkHead> CreateHead(const anet::TensorDict& features) const override
        {
            const auto& tensor = anet::GetOrFail(features, anet::nn::kKey_DefaultOutput,
                "Please configure RND net.body.output.[features].");
            if (tensor.dim() != 2)
                ANET_SYSTEM_ERROR("Invalid RND features shape=" << tensor.sizes()
                    << "; expected [B, F] before projection to [B, head.output_dim].");
            return anet::nn::LinearHeadFactory::CreateHead(features);
        }
    };

    // NN の通常の重み初期化は CPU の既定 generator を使う。構築時だけ seed を
    // 切り替え、例外時も元へ戻す。実行中のlane抽選はfacadeが専用乱数で行う。
    class ConstructionSeedScope {
    public:
        explicit ConstructionSeedScope(anet::seed_t seed)
            : generator_(at::detail::getDefaultCPUGenerator()), state_(generator_.get_state())
        {
            generator_.set_current_seed(seed);
        }
        ~ConstructionSeedScope() { generator_.set_state(state_); }
    private:
        at::Generator generator_;
        torch::Tensor state_;
    };
} // namespace

// 累積の計算と保存の並びは共通部品の既存契約をそのまま使う。
class RndNoveltyEstimator::CumulativeErrorStatistics final : public RndErrorStatistics {
public:
    void Update(const torch::Tensor& errors) override;
    bool HasData() const override { return cumulative_.GetCount() > 0; }
    float Mean() const override { return cumulative_.GetMean().item<float>(); }
    float Std() const override { return cumulative_.GetStd().item<float>(); }
    int64_t Save(anet::OutputArchive& archive) const override { return cumulative_.Save(archive); }
    int64_t Load(anet::InputArchive& archive) override { return cumulative_.Load(archive); }
    std::unique_ptr<RndErrorStatistics> Clone() const override;
private:
    RunningMeanStd cumulative_{{}, 0.0};
};

// EMA は CPU double の逐次の再帰と補正件数を所有する。
class RndNoveltyEstimator::EmaErrorStatistics final : public RndErrorStatistics {
public:
    explicit EmaErrorStatistics(int64_t ema_steps) : decay_(1.0 - 1.0 / static_cast<double>(ema_steps)) {}
public:
    void Update(const torch::Tensor& errors) override;
    bool HasData() const override { return weight_ > 0.0; }
    float Mean() const override { return static_cast<float>(mean_); }
    float Std() const override { return static_cast<float>(weight_ > 0.0 ? std::sqrt(m2_ / weight_) : 0.0); }
    int64_t Save(anet::OutputArchive& archive) const override;
    int64_t Load(anet::InputArchive& archive) override;
    std::unique_ptr<RndErrorStatistics> Clone() const override { return std::make_unique<EmaErrorStatistics>(*this); }
private:
    const double decay_;
    double weight_ = 0.0;
    double mean_ = 0.0;
    double m2_ = 0.0;
};

void RndNoveltyEstimator::CumulativeErrorStatistics::Update(const torch::Tensor& errors)
{
    ANET_PROFILE_SCOPE_FULL(update, kErrorStatisticsUpdateProfile);
    cumulative_.Update(errors);
}

std::unique_ptr<RndNoveltyEstimator::RndErrorStatistics>
RndNoveltyEstimator::CumulativeErrorStatistics::Clone() const
{
    // 共通統計の Tensor を保存・復元し、元との別名共有を避ける。
    auto result = std::make_unique<CumulativeErrorStatistics>();
    std::stringstream state(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(state);
    Save(output);
    anet::InputArchive input(state);
    result->Load(input);
    return result;
}

void RndNoveltyEstimator::EmaErrorStatistics::Update(const torch::Tensor& errors)
{
    ANET_PROFILE_SCOPE_FULL(update, kErrorStatisticsUpdateProfile);
    // 既に CPU に戻した誤差を double で一括変換する。lane ごとの転送は無い。
    const auto input = errors.to(torch::kDouble).contiguous();
    const auto* values = input.data_ptr<double>();
    for (int64_t lane = 0; lane < input.numel(); ++lane) {
        const auto previous_weight = weight_ * decay_;
        weight_ = previous_weight + 1.0;
        const auto difference = values[lane] - mean_;
        mean_ += difference / weight_;
        // 二乗和同士の減算を避け、微小な散らばりの桁落ちを抑える。
        m2_ = decay_ * m2_ + difference * difference * previous_weight / weight_;
    }
}

int64_t RndNoveltyEstimator::EmaErrorStatistics::Save(anet::OutputArchive& archive) const
{
    auto bytes = archive.Write(weight_);
    bytes += archive.Write(mean_);
    bytes += archive.Write(m2_);
    return bytes;
}

int64_t RndNoveltyEstimator::EmaErrorStatistics::Load(anet::InputArchive& archive)
{
    // 検証が終わるまで内部の統計へ反映しない。
    double weight, mean, m2;
    auto bytes = archive.Read(weight);
    bytes += archive.Read(mean);
    bytes += archive.Read(m2);
    if (!std::isfinite(weight) || weight < 0.0 || !std::isfinite(mean) || !std::isfinite(m2) || m2 < 0.0)
        ANET_SYSTEM_ERROR("Invalid RND EMA checkpoint weight=" << weight << ", mean=" << mean
            << ", m2=" << m2 << "; expected finite values with weight >= 0 and m2 >= 0.");
    weight_ = weight;
    mean_ = mean;
    m2_ = m2;
    return bytes;
}

// 観測の形だけbroadcastし、分散0の成分は0へ正規化する。呼出し側が倍率を1にする。
static torch::Tensor NormalizeFeedbackField(const torch::Tensor& value, const torch::Tensor& mean,
    const torch::Tensor& variance, float clip_range)
{
    const auto input = value.to(torch::kDouble);
    return torch::where(variance > 0, (input - mean) / variance, torch::zeros_like(input))
        .clamp(-clip_range, clip_range);
}

// 方式が選ぶ正規化の配置と統計の状態を、実行中の estimator から分離する。
class RndNoveltyEstimator::RndStatistics {
public:
    RndStatistics(const torch::Device& observation_device, const std::vector<std::string>& byte_keys,
        int64_t warmup_steps, bool skip_warmup_learning)
        : observation_device_(observation_device), byte_keys_(byte_keys), warmup_steps_(warmup_steps),
          skip_warmup_learning_(skip_warmup_learning) {}
    virtual ~RndStatistics() = default;
public:
    const torch::Device& GetObservationDevice() const { return observation_device_; }
    bool IsWarmingUp(int64_t count) const { return count < warmup_steps_; }
    bool ShouldSkipLearn(int64_t count) const
    {
        return skip_warmup_learning_ && warmup_steps_ > 0 && count <= warmup_steps_;
    }
    anet::TensorDict NormalizeObservations(anet::TensorDict observations) const
    {
        // 呼び出し元で選択済みの device へ転送したあと、連続 byte だけを変換する。
        ConvertBytes(observations);
        return Normalize(observations);
    }
public:
    // forward はこの呼び出しのあいだだけ借り、統計の状態には保存しない。
    virtual ScoreResult Estimate(anet::TensorDict observations, int64_t score_count,
        const ScoreFunction& forward, std::map<std::string, float>& diagnostics) = 0;
    virtual torch::Tensor StandardizeWithoutUpdate(const torch::Tensor& errors) const = 0;
    virtual void RestoreDiagnostics(std::map<std::string, float>& diagnostics) const = 0;
    virtual const char* GetCheckpointFormat() const = 0;
    virtual int64_t Save(anet::OutputArchive& archive) const = 0;
    virtual int64_t Load(anet::InputArchive& archive) = 0;
    virtual std::unique_ptr<RndStatistics> Clone() const = 0;
    virtual void ValidateScoreCount(int64_t) const {}
protected:
    virtual anet::TensorDict Normalize(const anet::TensorDict& observations) const = 0;
    int64_t GetWarmupSteps() const { return warmup_steps_; }
    void ConvertBytes(anet::TensorDict& observations) const
    {
        for (const auto& key : byte_keys_)
            observations.Set(key, observations.At(key).to(torch::kFloat32));
    }
private:
    const torch::Device observation_device_;
    const std::vector<std::string> byte_keys_;
    const int64_t warmup_steps_;
    const bool skip_warmup_learning_;
};

class RndNoveltyEstimator::WhitenStatistics final : public RndStatistics {
public:
    WhitenStatistics(const RndNoveltyEstimatorConfig& config, const anet::TensorSpecMap& observation_spec,
        const torch::Device& device, const std::vector<std::string>& byte_keys)
        : RndStatistics(device, byte_keys, config.warmup_steps, config.warmup_mode == kWarmupCollectErrors),
          normalizer_config_(config.obs_norm), use_centering_(config.error_norm.use_centering),
          collect_errors_(config.warmup_mode == kWarmupCollectErrors),
          checkpoint_format_(config.error_norm.average_mode == kErrorAverageEma ? kEmaCheckpointFormat : kCheckpointFormat),
          error_stats_(config.error_norm.average_mode == kErrorAverageEma
              ? std::unique_ptr<RndErrorStatistics>(std::make_unique<EmaErrorStatistics>(config.error_norm.ema_steps))
              : std::make_unique<CumulativeErrorStatistics>())
    {
        state_spec_.obs_spec = observation_spec;
        normalizer_ = anet::rl::ObservationNormalizerFactory(normalizer_config_).CreateObservationNormalizer(state_spec_);
    }
public:
    ScoreResult Estimate(anet::TensorDict observations, int64_t score_count,
        const ScoreFunction& forward, std::map<std::string, float>& diagnostics) override
    {
        // 採点は更新前の観測統計を読み、その結果だけを統計更新へ渡す。
        auto result = Score(observations, forward);
        result.alpha = UpdateStatistics(observations, result.errors, score_count, diagnostics);
        return result;
    }
    torch::Tensor StandardizeWithoutUpdate(const torch::Tensor& errors) const override
    {
        auto result = torch::full_like(errors, std::numeric_limits<float>::quiet_NaN());
        if (error_stats_->HasData() && error_stats_->Std() > 0.0f)
            result = (errors - (use_centering_ ? error_stats_->Mean() : 0.0f)) / error_stats_->Std();
        return result;
    }
    void RestoreDiagnostics(std::map<std::string, float>& diagnostics) const override
    {
        if (error_stats_->HasData()) {
            diagnostics[kErrorMean] = error_stats_->Mean();
            diagnostics[kErrorStd] = error_stats_->Std();
        }
    }
    const char* GetCheckpointFormat() const override { return checkpoint_format_.c_str(); }
    int64_t Save(anet::OutputArchive& archive) const override
    {
        auto bytes = normalizer_->Save(archive);
        bytes += error_stats_->Save(archive);
        return bytes;
    }
    int64_t Load(anet::InputArchive& archive) override
    {
        auto bytes = normalizer_->Load(archive);
        bytes += error_stats_->Load(archive);
        return bytes;
    }
    std::unique_ptr<RndStatistics> Clone() const override
    {
        return std::unique_ptr<RndStatistics>(new WhitenStatistics(*this));
    }
private:
    explicit WhitenStatistics(const WhitenStatistics& source)
        : RndStatistics(source), normalizer_config_(source.normalizer_config_), state_spec_(source.state_spec_),
          use_centering_(source.use_centering_), collect_errors_(source.collect_errors_),
          checkpoint_format_(source.checkpoint_format_),
          normalizer_(anet::rl::ObservationNormalizerFactory(normalizer_config_).CreateObservationNormalizer(state_spec_)),
          error_stats_(source.error_stats_->Clone())
    {
        // 正規化の Tensor は保存・復元し、読み取り用の写しも共有しない。
        std::stringstream state(std::ios::in | std::ios::out | std::ios::binary);
        anet::OutputArchive output(state);
        source.normalizer_->Save(output);
        anet::InputArchive input(state);
        normalizer_->Load(input);
    }

    anet::TensorDict Normalize(const anet::TensorDict& observations) const override
    {
        return normalizer_->Normalize(observations);
    }

    ScoreResult Score(anet::TensorDict& observations, const ScoreFunction& forward) const
    {
        ANET_PROFILE_SCOPE_FULL(normalize, kEstimateNormalizeProfile);
        ConvertBytes(observations);
        const auto normalized = Normalize(observations);
        ANET_PROFILE_SCOPE_END_NAMED(normalize);
        return forward(normalized);
    }
    torch::Tensor UpdateStatistics(const anet::TensorDict& observations, const torch::Tensor& errors,
        int64_t score_count, std::map<std::string, float>& diagnostics)
    {
        ANET_PROFILE_SCOPE_FULL(statistics, kEstimateStatisticsProfile);
        auto alpha = torch::ones_like(errors, torch::TensorOptions().dtype(torch::kFloat32));
        const auto std = error_stats_->Std();
        if (!IsWarmingUp(score_count) && error_stats_->HasData() && std > 0.0f)
            alpha = 1.0f + (errors - (use_centering_ ? error_stats_->Mean() : 0.0f)) / std;
        // predictor が慣れるまで誤差の統計を空に保つ。collect_errors のときだけ warm-up 中も集計する。
        if (!IsWarmingUp(score_count) || collect_errors_) error_stats_->Update(errors);
        // 誤差の統計を進めたあと、正規化せずに観測統計だけを更新する。
        ANET_PROFILE_SCOPE_FULL(observation_statistics, "RndNoveltyEstimator::Estimate.observation_statistics");
        normalizer_->UpdateStats(observations);
        RestoreDiagnostics(diagnostics);
        return alpha;
    }
private:
    const ObservationNormalizerConfig normalizer_config_;
    anet::rl::StateSpec state_spec_;
    const bool use_centering_;
    const bool collect_errors_;
    const std::string checkpoint_format_;
    std::shared_ptr<ObservationNormalizer> normalizer_;
    std::unique_ptr<RndErrorStatistics> error_stats_;
};

class RndNoveltyEstimator::VarianceFeedbackStatistics final : public RndStatistics {
public:
    VarianceFeedbackStatistics(const RndNoveltyEstimatorConfig& config,
        const anet::TensorSpecMap& observation_spec, const std::vector<std::string>& byte_keys);
public:
    ScoreResult Estimate(anet::TensorDict observations, int64_t score_count,
        const ScoreFunction& forward, std::map<std::string, float>& diagnostics) override
    {
        ANET_PROFILE_SCOPE_FULL(normalize, kEstimateNormalizeProfile);
        ConvertBytes(observations);
        // 観測の更新前の件数をここで確定し、誤差の更新時に逆算しない。
        const auto first_count = n_ + 1;
        auto [normalized, valid_lanes] = NormalizeAndUpdate(observations, clip_range_);
        ANET_PROFILE_SCOPE_END_NAMED(normalize);
        auto result = forward(normalized);
        ANET_PROFILE_SCOPE_FULL(statistics, kEstimateStatisticsProfile);
        result.alpha = torch::ones_like(result.errors, torch::TensorOptions().dtype(torch::kFloat32));
        result.alpha += UpdateErrors(result.errors, first_count);
        const auto counts = torch::arange(result.errors.size(0), torch::kInt64) + score_count;
        result.alpha.masked_fill_(~valid_lanes | (counts < GetWarmupSteps()), 1.0f);
        RestoreDiagnostics(diagnostics);
        return result;
    }
    torch::Tensor StandardizeWithoutUpdate(const torch::Tensor& errors) const override { return ErrorsWithoutUpdate(errors); }
    void RestoreDiagnostics(std::map<std::string, float>& diagnostics) const override
    {
        if (n_ > 0) {
            diagnostics[kErrorMean] = static_cast<float>(err_m_);
            diagnostics[kErrorStd] = static_cast<float>(std::sqrt(err_v_));
        }
    }
    const char* GetCheckpointFormat() const override { return kCheckpointFormat; }
    int64_t Save(anet::OutputArchive& archive) const override;
    int64_t Load(anet::InputArchive& archive) override;
    std::unique_ptr<RndStatistics> Clone() const override
    {
        return std::unique_ptr<RndStatistics>(new VarianceFeedbackStatistics(*this));
    }
    void ValidateScoreCount(int64_t count) const override
    {
        if (count != n_)
            ANET_SYSTEM_ERROR("Invalid RND checkpoint score_count=" << count
                << "; expected variance_feedback count=" << n_ << ".");
    }
private:
    explicit VarianceFeedbackStatistics(const VarianceFeedbackStatistics& source)
        : RndStatistics(source), clip_range_(source.clip_range_), err_m_(source.err_m_), err_v_(source.err_v_), n_(source.n_)
    {
        // 観測の平均・分散を device 内で複製し、元との更新を分離する。
        for (const auto& [key, mean] : source.obs_m_) {
            obs_m_.emplace(key, mean.clone());
            obs_v_.emplace(key, source.obs_v_.at(key).clone());
        }
    }
    anet::TensorDict Normalize(const anet::TensorDict& observations) const override;
    std::pair<anet::TensorDict, torch::Tensor> NormalizeAndUpdate(const anet::TensorDict& observations, float clip_range);
    torch::Tensor UpdateErrors(const torch::Tensor& errors, int64_t first_count);
    torch::Tensor ErrorsWithoutUpdate(const torch::Tensor& errors) const;
    void ValidateObservations(const anet::TensorDict& observations) const;
private:
    const float clip_range_;
    std::map<std::string, torch::Tensor> obs_m_, obs_v_;
    double err_m_ = 0.0;
    double err_v_ = 1.0;
    int64_t n_ = 0;
};

RndNoveltyEstimator::VarianceFeedbackStatistics::VarianceFeedbackStatistics(
    const RndNoveltyEstimatorConfig& config, const anet::TensorSpecMap& observation_spec,
    const std::vector<std::string>& byte_keys)
    : RndStatistics(torch::kCPU, byte_keys, config.warmup_steps, true), clip_range_(config.obs_norm.clip_range)
{
    // 連続観測と同じ形の平均0・分散1をCPU doubleで保持する。
    // カテゴリID・非浮動小数点は共通Normalizerと同じく素通しし、NN境界の変換に任せる。
    for (const auto& [key, spec] : observation_spec) {
        if (!torch::isFloatingType(spec.dtype) || spec.IsDiscrete()) continue;
        obs_m_.emplace(key, torch::zeros(spec.shape, torch::kDouble));
        obs_v_.emplace(key, torch::ones(spec.shape, torch::kDouble));
    }
}

void RndNoveltyEstimator::VarianceFeedbackStatistics::ValidateObservations(
    const anet::TensorDict& observations) const
{
    const auto batch_size = observations.Size(0);
    for (const auto& [key, mean] : obs_m_) {
        const auto& value = observations.At(key);
        if (value.dim() != mean.dim() + 1 || value.size(0) != batch_size
            || value.sizes().slice(1) != mean.sizes())
            ANET_SYSTEM_ERROR("Invalid RND observation key='" << key << "' shape=" << value.sizes()
                << "; expected batch=" << batch_size << " followed by shape=" << mean.sizes() << ".");
    }
}

anet::TensorDict RndNoveltyEstimator::VarianceFeedbackStatistics::Normalize(
    const anet::TensorDict& observations) const
{
    const auto clip_range = clip_range_;
    // Learnは現在の統計を読むだけで、nも平均・分散も更新しない。
    ValidateObservations(observations);
    anet::TensorDict normalized = observations;
    for (const auto& [key, mean] : obs_m_)
        normalized.Set(key, NormalizeFeedbackField(observations.At(key), mean, obs_v_.at(key), clip_range)
            .to(torch::kFloat32));
    return normalized;
}

std::pair<anet::TensorDict, torch::Tensor>
RndNoveltyEstimator::VarianceFeedbackStatistics::NormalizeAndUpdate(
    const anet::TensorDict& observations, float clip_range)
{
    // 各laneを、その直前までの統計で正規化してからフィードバックする。
    ValidateObservations(observations);
    const auto batch_size = observations.Size(0);
    anet::TensorDict normalized = observations;
    auto valid = torch::ones({batch_size}, torch::kBool);
    auto* valid_lanes = valid.data_ptr<bool>();
    for (const auto& [key, mean] : obs_m_)
        normalized.Set(key, torch::empty_like(observations.At(key), torch::TensorOptions().dtype(torch::kFloat32)));
    for (int64_t lane = 0; lane < batch_size; ++lane) {
        ++n_;
        const auto count = static_cast<double>(n_);
        for (auto& [key, mean] : obs_m_) {
            auto& variance = obs_v_.at(key);
            valid_lanes[lane] = valid_lanes[lane] && (variance > 0).all().item<bool>();
            const auto value = NormalizeFeedbackField(observations.At(key)[lane], mean, variance, clip_range);
            if (!torch::isfinite(value).all().item<bool>())
                ANET_SYSTEM_ERROR("Invalid RND normalized observation key='" << key << "' lane=" << lane
                    << " value=" << value.masked_select(~torch::isfinite(value))[0].item<double>()
                    << "; expected finite values.");
            normalized.At(key)[lane].copy_(value);
            const auto difference = value - mean;
            mean.add_(difference / count);
            variance.copy_((variance * count + difference.square() * count / (count + 1.0)) / (count + 1.0));
        }
    }
    return {normalized, valid};
}

torch::Tensor RndNoveltyEstimator::VarianceFeedbackStatistics::UpdateErrors(const torch::Tensor& errors, int64_t first_count)
{
    // forwardはまとめてよいが、誤差の再帰とzは観測と同じlane順で更新する。
    const auto input = errors.to(torch::kDouble).contiguous();
    const auto batch_size = input.size(0);
    const auto* values = input.data_ptr<double>();
    auto result = torch::zeros({batch_size}, torch::kFloat32);
    auto* standardized = result.data_ptr<float>();
    for (int64_t lane = 0; lane < batch_size; ++lane) {
        const auto count = static_cast<double>(first_count + lane);
        const auto difference = values[lane] - err_m_;
        err_m_ += difference / count;
        err_v_ = (err_v_ * count + difference * difference * count / (count + 1.0)) / (count + 1.0);
        standardized[lane] = err_v_ > 0 ? static_cast<float>((values[lane] - err_m_) / err_v_) : 0.0f;
    }
    return result;
}

torch::Tensor RndNoveltyEstimator::VarianceFeedbackStatistics::ErrorsWithoutUpdate(const torch::Tensor& errors) const
{
    // lane 同士は統計を共有更新せず、各 lane を独立に次の1件として扱う。
    for (const auto& [key, variance] : obs_v_)
        if (!(variance > 0).all().item<bool>())
            return torch::full_like(errors, std::numeric_limits<float>::quiet_NaN());
    const auto count = static_cast<double>(n_ + 1);
    const auto values = errors.to(torch::kDouble);
    const auto difference = values - err_m_;
    const auto mean = err_m_ + difference / count;
    const auto variance = (err_v_ * count + difference.square() * count / (count + 1.0)) / (count + 1.0);
    return torch::where(variance > 0, (values - mean) / variance,
        torch::full_like(values, std::numeric_limits<double>::quiet_NaN())).to(torch::kFloat32);
}

int64_t RndNoveltyEstimator::VarianceFeedbackStatistics::Save(anet::OutputArchive& archive) const
{
    auto bytes = archive.Write(n_);
    bytes += archive.Write(err_m_);
    bytes += archive.Write(err_v_);
    for (const auto& [key, mean] : obs_m_) {
        bytes += archive.Write(key);
        bytes += archive.Write(mean);
        bytes += archive.Write(obs_v_.at(key));
    }
    return bytes;
}

int64_t RndNoveltyEstimator::VarianceFeedbackStatistics::Load(anet::InputArchive& archive)
{
    // 現在の観測仕様と一致する有限のCPU doubleだけを復元する。
    int64_t count;
    double error_mean, error_variance;
    auto bytes = archive.Read(count);
    bytes += archive.Read(error_mean);
    bytes += archive.Read(error_variance);
    if (count < 0 || !std::isfinite(error_mean) || !std::isfinite(error_variance) || error_variance < 0)
        ANET_SYSTEM_ERROR("Invalid RND variance_feedback state: n=" << count << ", err_m=" << error_mean
            << ", err_v=" << error_variance << "; expected n >= 0, finite moments and err_v >= 0.");
    for (auto& [key, mean] : obs_m_) {
        std::string saved_key;
        torch::Tensor saved_mean, saved_variance;
        bytes += archive.Read(saved_key);
        bytes += archive.Read(saved_mean);
        bytes += archive.Read(saved_variance);
        if (saved_key != key)
            ANET_SYSTEM_ERROR("Invalid RND variance_feedback observation key='" << saved_key
                << "'; expected='" << key << "'.");
        for (const auto& value : {saved_mean, saved_variance}) {
            if (value.sizes() != mean.sizes() || value.scalar_type() != torch::kDouble
                || !value.device().is_cpu())
                ANET_SYSTEM_ERROR("Invalid RND variance_feedback tensor key='" << key
                    << "' shape=" << value.sizes() << " dtype=" << value.dtype() << " device=" << value.device()
                    << "; expected finite CPU double with shape=" << mean.sizes() << ".");
            const auto invalid_values = value.masked_select(~torch::isfinite(value));
            if (invalid_values.numel() > 0)
                ANET_SYSTEM_ERROR("Invalid RND variance_feedback tensor key='" << key
                    << "' value=" << invalid_values[0].item<double>() << "; expected finite values.");
        }
        if ((saved_variance < 0).any().item<bool>())
            ANET_SYSTEM_ERROR("Invalid RND variance_feedback variance key='" << key
                << "' value=" << saved_variance.masked_select(saved_variance < 0)[0].item<double>() << "; expected >= 0.");
        mean = saved_mean;
        obs_v_.at(key) = saved_variance;
    }
    n_ = count;
    err_m_ = error_mean;
    err_v_ = error_variance;
    return bytes;
}

RndNoveltyEstimatorConfig::RndNoveltyEstimatorConfig(
    const anet::ConfigData& data, const std::string& prefix)
    : anet::Config(data, prefix),
      target{.net = anet::nn::NetworkConfig(data, prefix + ".target.net")},
      predictor{.net = anet::nn::NetworkConfig(data, prefix + ".predictor.net")}
{
    ANET_READ_CONFIG(data, multiplier_mode);
    ANET_READ_CONFIG(data, statistics_mode);
    ANET_READ_CONFIG(data, multiplier_max);
    ANET_READ_CONFIG(data, warmup_steps);
    ANET_READ_CONFIG(data, warmup_mode);
    ANET_READ_CONFIG(data, error_norm.average_mode);
    ANET_READ_CONFIG(data, error_norm.ema_steps);
    ANET_READ_CONFIG(data, error_norm.use_centering);
    ANET_READ_CONFIG(data, learning_rate);
    ANET_READ_CONFIG(data, weight_decay);
    ANET_READ_CONFIG(data, adam_eps);
    ANET_READ_CONFIG(data, head.output_dim);
    ANET_READ_CONFIG(data, head.init.mode);
    ANET_READ_CONFIG(data, head.init.manual_gain);
    ANET_READ_CONFIG(data, head.init.nonlinearity);
    ANET_READ_CONFIG(data, head.init.constant_val);
    ANET_READ_CONFIG(data, head.init.trunc_std);
    ANET_READ_CONFIG(data, head.init.trunc_a);
    ANET_READ_CONFIG(data, head.init.trunc_b);
    // RND は平均を引いた白色化を使い、Q 入力用の既定とは分離する。
    obs_norm.use_centering = true;
    obs_norm.use_robust_update = false;
    obs_norm.post_process_type = anet::rl::kPostProcessNone;
    obs_norm.clip_range = 5.0f;
    const auto obs_norm_defaults = obs_norm;
    ANET_READ_CONFIG(data, obs_norm.pass_through);
    ANET_READ_CONFIG(data, obs_norm.use_clipping);
    ANET_READ_CONFIG(data, obs_norm.clip_range);
    ANET_READ_CONFIG(data, obs_norm.use_dynamic_scaling);
    ANET_READ_CONFIG(data, obs_norm.use_centering);
    ANET_READ_CONFIG(data, obs_norm.epsilon);
    ANET_READ_CONFIG(data, obs_norm.constant_mean);
    ANET_READ_CONFIG(data, obs_norm.constant_std);
    ANET_READ_CONFIG(data, obs_norm.use_robust_update);
    ANET_READ_CONFIG(data, obs_norm.robust_warmup_count);
    ANET_READ_CONFIG(data, obs_norm.robust_std_threshold);
    ANET_READ_CONFIG(data, obs_norm.post_process_type);
    ANET_READ_CONFIG(data, obs_norm.post_process_threshold);

    // Network や optimizer の資源を確保する前に全設定を検証する。
    if (head.output_dim < 1)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".head.output_dim=" << head.output_dim << "; expected >= 1.");
    if (warmup_steps < 0)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".warmup_steps=" << warmup_steps << "; expected >= 0.");
    if (!std::isfinite(multiplier_max) || multiplier_max <= 1.0f)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".multiplier_max=" << multiplier_max << "; expected finite > 1.");
    if (multiplier_mode != kMultiplierClipped && multiplier_mode != kMultiplierUnclipped)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".multiplier_mode='" << multiplier_mode
            << "'; expected clipped or unclipped.");
    if (multiplier_mode == kMultiplierUnclipped && multiplier_max != kDefaultMultiplierMax)
        anet::log::warn_once(prefix + ".multiplier_max") << "Ignored " << prefix << ".multiplier_max=" << multiplier_max
            << " with multiplier_mode=" << multiplier_mode << "; default=" << kDefaultMultiplierMax << ".";
    if (!std::isfinite(learning_rate) || learning_rate <= 0.0f)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".learning_rate=" << learning_rate << "; expected finite > 0.");
    if (!std::isfinite(weight_decay) || weight_decay < 0.0f)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".weight_decay=" << weight_decay << "; expected finite >= 0.");
    for (const auto& [role, network] : {std::pair("target", &target.net), std::pair("predictor", &predictor.net)})
        if (network->output_keys.empty())
            ANET_SYSTEM_ERROR("Invalid " << prefix << "." << role << ".net: no Body outputs; expected configured Network body.");
    anet::rl::ValidateObservationNormalizerConfig(obs_norm, prefix + ".obs_norm");
    anet::nn::ValidateWeightInitConfig(head.init, prefix + ".head.init");
    if (!std::isfinite(adam_eps) || adam_eps <= 0.0)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".adam_eps=" << adam_eps << "; expected finite > 0.");
    if (statistics_mode != kStatisticsWhiten && statistics_mode != kStatisticsVarianceFeedback)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".statistics_mode='" << statistics_mode
            << "'; expected whiten or variance_feedback.");
    if (warmup_mode != kWarmupLearnPredictor && warmup_mode != kWarmupCollectErrors)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".warmup_mode='" << warmup_mode
            << "'; expected learn_predictor or collect_errors.");
    if (error_norm.average_mode != kErrorAverageCumulative && error_norm.average_mode != kErrorAverageEma)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".error_norm.average_mode='" << error_norm.average_mode
            << "'; expected cumulative or ema.");
    if (error_norm.ema_steps < 1)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".error_norm.ema_steps=" << error_norm.ema_steps
            << "; expected integer >= 1.");

    // 値の検証後に、現在の組が使わない設定キーだけを既定との差で知らせる。
    const auto notify_unused = [&](const char* key, const auto& value, const auto& default_value, bool info = false) {
        if (value == default_value) return;
        const auto message = "Ignored " + prefix + "." + key + "=" + anet::json(value).dump()
            + " with statistics_mode=" + statistics_mode + ", error_norm.average_mode=" + error_norm.average_mode
            + ", warmup_steps=" + std::to_string(warmup_steps) + "; default=" + anet::json(default_value).dump() + ".";
        if (info) anet::log::info_once(prefix + "." + key) << message;
        else anet::log::warn_once(prefix + "." + key) << message;
    };
    if (statistics_mode == kStatisticsVarianceFeedback) {
        notify_unused("warmup_mode", warmup_mode, std::string(kWarmupLearnPredictor));
        notify_unused("error_norm.average_mode", error_norm.average_mode, std::string(kErrorAverageCumulative));
        notify_unused("error_norm.ema_steps", error_norm.ema_steps, kDefaultEmaSteps);
        notify_unused("error_norm.use_centering", error_norm.use_centering, kDefaultErrorNormUseCentering);
    } else {
        if (warmup_steps == 0)
            notify_unused("warmup_mode", warmup_mode, std::string(kWarmupLearnPredictor), true);
        if (error_norm.average_mode == kErrorAverageCumulative)
            notify_unused("error_norm.ema_steps", error_norm.ema_steps, kDefaultEmaSteps);
    }

    // 解決後の値だけで判断し、無効な設定キーも検証済みのうえで構築時に1回知らせる。
    if (statistics_mode == kStatisticsVarianceFeedback) {
        const auto warn_ignored = [&](const char* key, const auto& value, const auto& default_value) {
            if (value != default_value)
                anet::log::warn_once(prefix + ".obs_norm." + key) << "Ignored " << prefix << ".obs_norm." << key << "=" << anet::json(value).dump()
                    << " with statistics_mode=" << statistics_mode << "; default=" << anet::json(default_value).dump() << ".";
        };
        warn_ignored("pass_through", obs_norm.pass_through, obs_norm_defaults.pass_through);
        warn_ignored("use_clipping", obs_norm.use_clipping, obs_norm_defaults.use_clipping);
        warn_ignored("use_dynamic_scaling", obs_norm.use_dynamic_scaling, obs_norm_defaults.use_dynamic_scaling);
        warn_ignored("use_centering", obs_norm.use_centering, obs_norm_defaults.use_centering);
        warn_ignored("epsilon", obs_norm.epsilon, obs_norm_defaults.epsilon);
        warn_ignored("constant_mean", obs_norm.constant_mean, obs_norm_defaults.constant_mean);
        warn_ignored("constant_std", obs_norm.constant_std, obs_norm_defaults.constant_std);
        warn_ignored("use_robust_update", obs_norm.use_robust_update, obs_norm_defaults.use_robust_update);
        warn_ignored("robust_warmup_count", obs_norm.robust_warmup_count, obs_norm_defaults.robust_warmup_count);
        warn_ignored("robust_std_threshold", obs_norm.robust_std_threshold, obs_norm_defaults.robust_std_threshold);
        warn_ignored("post_process_type", obs_norm.post_process_type, obs_norm_defaults.post_process_type);
        warn_ignored("post_process_threshold", obs_norm.post_process_threshold, obs_norm_defaults.post_process_threshold);
    }
}

RndNoveltyEstimator::~RndNoveltyEstimator() = default;

RndNoveltyEstimator::RndNoveltyEstimator(const RndNoveltyEstimatorConfig& config,
    const anet::TensorSpecMap& observation_spec, const torch::Device& device,
    const anet::SeedMaker& seed_maker)
    : config_(config), device_(device), target_feature_stats_({config.head.output_dim}, 0.0)
{
    // 互いに独立した重みを作り、教師側は勾配と学習時の状態更新を止める。
    std::vector<std::string> byte_observation_keys;
    auto normalized_spec = observation_spec;
    for (auto& [key, spec] : normalized_spec) {
        if (spec.dtype == torch::kUInt8 && !spec.IsDiscrete()) {
            spec.dtype = torch::kFloat32;
            byte_observation_keys.push_back(key);
        }
    }
    auto head = std::make_shared<RndHeadFactory>(config.head.output_dim, kOutput,
        config.head.init);
    const auto build = [&](const char* name, const anet::nn::NetworkConfig& network_config) {
        const auto seed = seed_maker.MakeNamedSeed(name);
        ConstructionSeedScope scope(seed);
        return anet::nn::NetworkBuilder::BuildNetwork(network_config, normalized_spec, head, seed, torch::kCPU);
    };
    target_ = build(kTargetSeed, config.target.net);
    predictor_ = build(kPredictorSeed, config.predictor.net);
    target_->to(device);
    predictor_->to(device);
    target_->eval();
    for (auto& parameter : target_->parameters()) parameter.requires_grad_(false);
    optimizer_ = std::make_unique<torch::optim::AdamW>(predictor_->parameters(),
        torch::optim::AdamWOptions(config.learning_rate).weight_decay(config.weight_decay).eps(config.adam_eps));
    // 実行時の方式分岐は置かず、状態を所有する型をここで一度だけ選ぶ。
    if (config.statistics_mode == kStatisticsWhiten)
        statistics_ = std::make_unique<WhitenStatistics>(config, normalized_spec, device, byte_observation_keys);
    else
        statistics_ = std::make_unique<VarianceFeedbackStatistics>(config, normalized_spec, byte_observation_keys);
    diagnostics_ = LifelongNoveltyEstimatorFactory::InitialDiagnostics();
}

RndNoveltyEstimator::RndNoveltyEstimator(const RndNoveltyEstimator& source)
    : config_(source.config_), device_(source.device_), statistics_(source.statistics_->Clone()),
      target_feature_stats_({config_.head.output_dim}, 0.0), score_count_(source.score_count_), diagnostics_(source.diagnostics_)
{
    // Network の再構築が消費する既定乱数を外へ持ち越さず、重みは device 内で複製する。
    ConstructionSeedScope seed_scope(0);
    target_ = source.target_->Clone(device_);
    predictor_ = source.predictor_->Clone(device_);
    target_->eval();
    predictor_->train(source.predictor_->is_training());
    for (auto& parameter : target_->parameters()) parameter.requires_grad_(false);
    optimizer_ = std::make_unique<torch::optim::AdamW>(predictor_->parameters(),
        torch::optim::AdamWOptions(config_.learning_rate).weight_decay(config_.weight_decay).eps(config_.adam_eps));
    // Tensor を含む統計と optimizer の所有状態も深く複製し、元との別名共有を避ける。
    std::stringstream state(std::ios::in | std::ios::out | std::ios::binary);
    anet::OutputArchive output(state);
    source.target_feature_stats_.Save(output);
    output.WriteTorchObject(*source.optimizer_);
    anet::InputArchive input(state);
    target_feature_stats_.Load(input);
    input.ReadTorchObject(*optimizer_);
}

std::unique_ptr<LifelongNoveltyEstimator> RndNoveltyEstimator::Clone() const
{
    ANET_PROFILE_FUNC();
    return std::unique_ptr<LifelongNoveltyEstimator>(new RndNoveltyEstimator(*this));
}

template<RndNoveltyEstimator::ForwardUse use>
RndNoveltyEstimator::ScoreResult RndNoveltyEstimator::ComputePredictionErrors(const anet::TensorDict& normalized) const
{
    // 用途ごとの既存の計測名を維持し、target は常に勾配無しで呼ぶ。
    torch::NoGradGuard no_grad;
    ANET_PROFILE_SCOPE_FULL(input_transfer, use == ForwardUse::Learn ? "RndNoveltyEstimator::Learn.input_transfer"
        : use == ForwardUse::Estimate ? "RndNoveltyEstimator::Estimate.input_transfer"
        : "RndNoveltyEstimator::EstimateWithoutUpdate.input_transfer");
    const auto input = normalized.To(device_);
    ANET_PROFILE_SCOPE_END_NAMED(input_transfer);
    ANET_PROFILE_SCOPE_FULL(target_forward, use == ForwardUse::Learn ? "RndNoveltyEstimator::Learn.target_forward"
        : use == ForwardUse::Estimate ? "RndNoveltyEstimator::Estimate.target_forward"
        : "RndNoveltyEstimator::EstimateWithoutUpdate.target_forward");
    const auto target = target_->Forward(input).At(kOutput);
    ANET_PROFILE_SCOPE_END_NAMED(target_forward);
    ANET_PROFILE_SCOPE_FULL(predictor_forward, use == ForwardUse::Learn ? "RndNoveltyEstimator::Learn.predictor_forward"
        : use == ForwardUse::Estimate ? "RndNoveltyEstimator::Estimate.predictor_forward"
        : "RndNoveltyEstimator::EstimateWithoutUpdate.predictor_forward");
    torch::AutoGradMode predictor_grad(use == ForwardUse::Learn);
    const auto prediction = predictor_->Forward(input).At(kOutput);
    const auto prediction_errors = (prediction - target).square().mean(-1);
    ANET_PROFILE_SCOPE_END_NAMED(predictor_forward);
    // 誤差の読み戻しは一度だけ行う。読むだけの採点は呼び出し元で WARN と NaN にする。
    ANET_PROFILE_SCOPE_FULL(error_transfer, use == ForwardUse::Learn ? "RndNoveltyEstimator::Learn.error_transfer"
        : use == ForwardUse::Estimate ? "RndNoveltyEstimator::Estimate.error_transfer"
        : "RndNoveltyEstimator::EstimateWithoutUpdate.error_transfer");
    torch::Tensor errors;
    if constexpr (use == ForwardUse::Learn) errors = prediction_errors.detach().to(torch::kCPU);
    else if constexpr (use == ForwardUse::ReadOnly) errors = prediction_errors.to(torch::kCPU).contiguous();
    else errors = prediction_errors.to(torch::kCPU);
    torch::Tensor target_features;
    if constexpr (use == ForwardUse::Estimate) target_features = target.to(statistics_->GetObservationDevice());
    if constexpr (use != ForwardUse::ReadOnly) {
        // 非有限値で採点の累積統計や補助学習の重み・moment を汚さないよう、更新前に検査する。
        const auto invalid = (~torch::isfinite(errors)).nonzero();
        if (invalid.numel() != 0) {
            const auto lane = invalid[0][0].item<int64_t>();
            ANET_SYSTEM_ERROR("Invalid RND prediction_error at lane=" << lane
                << " value=" << errors[lane].item<float>() << "; expected finite value.");
        }
    }
    return {.target_features = std::move(target_features), .prediction_errors = prediction_errors,
        .errors = std::move(errors)};
}

torch::Tensor RndNoveltyEstimator::Estimate(const anet::TensorDict& observations)
{
    ANET_PROFILE_FUNC();
    torch::NoGradGuard no_grad;
    // 観測件数の境界だけを通知し、predictorの学習開始・終了とは区別する。
    const bool warming_up = statistics_->IsWarmingUp(score_count_);
    if (warming_up && score_count_ == 0)
        anet::log::info_once(config_.GetConfigPrefix()) << "RND warmup started: warmup_mode=" << config_.warmup_mode
            << " warmup_steps=" << config_.warmup_steps;
    ANET_PROFILE_SCOPE(observation_transfer);
    auto observations_for_scoring = observations.To(statistics_->GetObservationDevice());
    ANET_PROFILE_SCOPE_END(observation_transfer);
    auto result = statistics_->Estimate(std::move(observations_for_scoring), score_count_,
        [this](const anet::TensorDict& normalized) {
            return ComputePredictionErrors<ForwardUse::Estimate>(normalized);
        }, diagnostics_);
    auto alpha = std::move(result.alpha);
    const auto& errors = result.errors;
    const auto& target_features = result.target_features;
    if (config_.multiplier_mode == kMultiplierClipped)
        alpha = alpha.clamp(1.0f, config_.multiplier_max);
    score_count_ += observations.Size(0);
    if (warming_up && score_count_ >= config_.warmup_steps)
        anet::log::info_once(config_.GetConfigPrefix()) << "RND warmup finished: warmup_mode=" << config_.warmup_mode
            << " warmup_steps=" << config_.warmup_steps;
    // target の出力の統計と診断値の読み出しも、区間を分けて測る。
    ANET_PROFILE_SCOPE(target_feature_statistics);
    target_feature_stats_.Update(target_features);
    ANET_PROFILE_SCOPE_NEXT(diagnostics);
    diagnostics_[kTargetFeatureVar] = TargetFeatureVariance();
    diagnostics_[kTargetFeatureMax] = target_features.abs().max().item<float>();
    diagnostics_[kErrorStep] = errors.mean().item<float>();
    diagnostics_[kClipRatio] = config_.multiplier_mode == kMultiplierClipped
        ? (alpha == config_.multiplier_max).to(torch::kFloat32).mean().item<float>() : 0.0f;
    ANET_PROFILE_SCOPE_NEXT(result_transfer);
    return alpha.to(observations.device());
}

LifelongNoveltyEstimateResult RndNoveltyEstimator::EstimateWithoutUpdate(const anet::TensorDict& observations) const
{
    ANET_PROFILE_FUNC();
    torch::NoGradGuard no_grad;
    // 入力と誤差の転送は batch ごとにまとめ、統計は現在値を読むだけにする。
    ANET_PROFILE_SCOPE(normalize);
    // 採点と同じ配置・変換順に揃える。読むだけの正規化は写しも更新しない。
    const auto normalized = statistics_->NormalizeObservations(observations.To(statistics_->GetObservationDevice()));
    ANET_PROFILE_SCOPE_NEXT(forward);
    const auto result = ComputePredictionErrors<ForwardUse::ReadOnly>(normalized);
    const auto& errors = result.errors;
    ANET_PROFILE_SCOPE_NEXT(standardize);
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    auto standardized = statistics_->StandardizeWithoutUpdate(errors);
    // 未成立は NaN として公開し、学習が加える倍率だけは同じ条件で1にする。
    const auto invalid = ~torch::isfinite(errors);
    standardized.masked_fill_(invalid, nan);
    auto multipliers = torch::ones_like(errors);
    if (score_count_ >= config_.warmup_steps)
        multipliers = torch::where(torch::isnan(standardized), multipliers, 1.0f + standardized);
    if (config_.multiplier_mode == kMultiplierClipped)
        multipliers.clamp_(1.0f, config_.multiplier_max);
    const auto diagnostic_errors = errors.masked_fill(invalid, nan);
    if (invalid.any().item<bool>()) {
        const auto lane = invalid.nonzero()[0][0].item<int64_t>();
        anet::log::warn_once(config_.GetConfigPrefix() + ".read_only_prediction_error")
            << "Invalid RND read-only prediction_error at lane=" << lane << " value=" << errors[lane].item<float>()
            << "; expected finite value; lane diagnostics are unavailable.";
    }
    return {.multipliers = std::move(multipliers),
        .diagnostics = {{kError, diagnostic_errors}, {kNormalizedError, standardized}}};
}

bool RndNoveltyEstimator::Learn(const anet::TensorDict& observations)
{
    ANET_PROFILE_FUNC();
    torch::NoGradGuard no_grad;
    
    // 誤差を集める warm-up と variance_feedback だけは optimizer を休止する。
    if (statistics_->ShouldSkipLearn(score_count_)) return false;

    // 渡された観測すべてを正規化する。採点側で育てた統計は読むだけにする。
    // 採点とは独立した入力を、同じ計算 device と最新の統計で正規化する。
    ANET_PROFILE_SCOPE(observation_transfer);
    auto observations_for_learning = observations.To(statistics_->GetObservationDevice());
    ANET_PROFILE_SCOPE_NEXT(normalize);
    const auto normalized = statistics_->NormalizeObservations(std::move(observations_for_learning));
    ANET_PROFILE_SCOPE_END(normalize);
    const auto result = ComputePredictionErrors<ForwardUse::Learn>(normalized);
    const auto& errors = result.errors;
    // forward が返した計算グラフを使い、loss と optimizer の処理だけを続ける。
    torch::AutoGradMode enable_grad(true);
    diagnostics_[kPredictorLoss] = errors.mean().item<float>();
    ANET_PROFILE_SCOPE(backward);
    const auto loss = result.prediction_errors.mean();
    optimizer_->zero_grad();
    loss.backward();
    ANET_PROFILE_SCOPE_NEXT(optimizer_step);
    optimizer_->step();
    return true;
}

std::map<std::string, float> RndNoveltyEstimator::Diagnostics() const { return diagnostics_; }

float RndNoveltyEstimator::TargetFeatureVariance() const
{
    // 共通部品の不偏分散を母分散へ戻す。初回1観測の分散は0である。
    const auto count = target_feature_stats_.GetCount();
    return count > 1 ? static_cast<float>(target_feature_stats_.GetVar().mean().item<double>()
        * static_cast<double>(count - 1) / static_cast<double>(count)) : 0.0f;
}

int64_t RndNoveltyEstimator::Save(anet::OutputArchive& archive) const
{
    ANET_PROFILE_FUNC();
    // 固定教師を含む学習状態と、採点した観測の件数を保存する。
    auto bytes = archive.Write(std::string(statistics_->GetCheckpointFormat()));
    bytes += archive.Write(config_.statistics_mode);
    bytes += archive.WriteTorchObject(target_);
    bytes += archive.WriteTorchObject(predictor_);
    bytes += archive.WriteTorchObject(*optimizer_);
    bytes += statistics_->Save(archive);
    bytes += target_feature_stats_.Save(archive);
    bytes += archive.Write(score_count_);
    return bytes;
}

int64_t RndNoveltyEstimator::Load(anet::InputArchive& archive)
{
    ANET_PROFILE_FUNC();
    std::string format;
    auto bytes = archive.Read(format);
    const auto* expected = statistics_->GetCheckpointFormat();
    if (format != expected)
        ANET_SYSTEM_ERROR("Invalid RND checkpoint format='" << format << "'; expected='" << expected << "'.");
    std::string statistics_mode;
    bytes += archive.Read(statistics_mode);
    if (statistics_mode != config_.statistics_mode)
        ANET_SYSTEM_ERROR("RND checkpoint statistics_mode='" << statistics_mode
            << "' differs from configured statistics_mode='" << config_.statistics_mode << "'.");
    bytes += archive.ReadTorchObject(target_);
    bytes += archive.ReadTorchObject(predictor_);
    bytes += archive.ReadTorchObject(*optimizer_);

    // moment と step は保存値を継承し、options は現在設定を優先する。
    for (auto& group : optimizer_->param_groups()) {
        auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
        if (options.lr() != config_.learning_rate || options.weight_decay() != config_.weight_decay
            || options.eps() != config_.adam_eps)
            anet::log::info_once(config_.GetConfigPrefix()) << "RND checkpoint optimizer options replaced: learning_rate=" << options.lr()
                << " -> " << config_.learning_rate << ", weight_decay=" << options.weight_decay()
                << " -> " << config_.weight_decay << ", adam_eps=" << options.eps() << " -> " << config_.adam_eps;
        options = torch::optim::AdamWOptions(config_.learning_rate).weight_decay(config_.weight_decay).eps(config_.adam_eps);
    }
    bytes += statistics_->Load(archive);
    bytes += target_feature_stats_.Load(archive);
    bytes += archive.Read(score_count_);
    if (score_count_ < 0)
        ANET_SYSTEM_ERROR("Invalid RND checkpoint score_count=" << score_count_ << "; expected >= 0.");
    statistics_->ValidateScoreCount(score_count_);
    for (auto& [key, value] : diagnostics_) value = std::numeric_limits<float>::quiet_NaN();
    statistics_->RestoreDiagnostics(diagnostics_);
    if (target_feature_stats_.GetCount() > 0)
        diagnostics_[kTargetFeatureVar] = TargetFeatureVariance();
    return bytes;
}

std::vector<std::string> LifelongNoveltyEstimatorFactory::LaneDiagnosticKeys()
{
    return {kError, kNormalizedError};
}

std::map<std::string, float> LifelongNoveltyEstimatorFactory::InitialDiagnostics()
{
    std::map<std::string, float> diagnostics;
    for (const auto* key : {kErrorMean, kErrorStd, kErrorStep, kClipRatio, kPredictorLoss,
        kTargetFeatureVar, kTargetFeatureMax})
        diagnostics.emplace(key, std::numeric_limits<float>::quiet_NaN());
    return diagnostics;
}

std::unique_ptr<LifelongNoveltyEstimator> LifelongNoveltyEstimatorFactory::Create(
    const anet::ConfigData& data, const std::string& prefix, const anet::TensorSpecMap& observation_spec,
    const torch::Device& device, const anet::SeedMaker& seed_maker)
{
    return std::make_unique<RndNoveltyEstimator>(RndNoveltyEstimatorConfig(data, prefix),
        observation_spec, device, seed_maker);
}
