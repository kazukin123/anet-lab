#include "intrinsic_rnd.hpp"
#include <cmath>
#include <limits>
#include <ATen/CPUGeneratorImpl.h>
#include "anet/profile.hpp"
#include "nn_heads.hpp"

using namespace anet::rl::intrinsic;

constexpr char kTargetSeed[] = "intrinsic.target";
constexpr char kPredictorSeed[] = "intrinsic.predictor";
constexpr char kOutput[] = "novelty";
constexpr char kErrorMean[] = "error_mean";
constexpr char kErrorStd[] = "error_std";
constexpr char kErrorStep[] = "error_step";
constexpr char kClipRatio[] = "clip_ratio";
constexpr char kPredictorLoss[] = "predictor_loss";
constexpr char kTargetFeatureVar[] = "target_feature_var";
constexpr char kTargetFeatureMax[] = "target_feature_max";
constexpr char kCheckpointFormat[] = "RndNoveltyEstimator-3";

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
                    << "; expected [B, F] before projection to [B, output_dim].");
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

// 観測の形だけbroadcastし、分散0の成分は0へ正規化する。呼出し側が倍率を1にする。
static torch::Tensor NormalizeFeedbackField(const torch::Tensor& value, const torch::Tensor& mean,
    const torch::Tensor& variance, float clip_range)
{
    const auto input = value.to(torch::kDouble);
    return torch::where(variance > 0, (input - mean) / variance, torch::zeros_like(input))
        .clamp(-clip_range, clip_range);
}

RndNoveltyEstimator::VarianceFeedbackStatistics::VarianceFeedbackStatistics(
    const anet::TensorSpecMap& observation_spec)
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
    const anet::TensorDict& observations, float clip_range) const
{
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

torch::Tensor RndNoveltyEstimator::VarianceFeedbackStatistics::UpdateErrors(const torch::Tensor& errors)
{
    // forwardはまとめてよいが、誤差の再帰とzは観測と同じlane順で更新する。
    const auto input = errors.to(torch::kDouble).contiguous();
    const auto batch_size = input.size(0);
    const auto* values = input.data_ptr<double>();
    auto result = torch::zeros({batch_size}, torch::kFloat32);
    auto* standardized = result.data_ptr<float>();
    for (int64_t lane = 0; lane < batch_size; ++lane) {
        const auto count = static_cast<double>(n_ - batch_size + lane + 1);
        const auto difference = values[lane] - err_m_;
        err_m_ += difference / count;
        err_v_ = (err_v_ * count + difference * difference * count / (count + 1.0)) / (count + 1.0);
        standardized[lane] = err_v_ > 0 ? static_cast<float>((values[lane] - err_m_) / err_v_) : 0.0f;
    }
    return result;
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
    ANET_READ_CONFIG(data, output_dim);
    ANET_READ_CONFIG(data, multiplier_mode);
    ANET_READ_CONFIG(data, statistics_mode);
    ANET_READ_CONFIG(data, multiplier_max);
    ANET_READ_CONFIG(data, warmup_steps);
    ANET_READ_CONFIG(data, learning_rate);
    ANET_READ_CONFIG(data, weight_decay);
    ANET_READ_CONFIG(data, adam_eps);
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
    if (output_dim < 1)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".output_dim=" << output_dim << "; expected >= 1.");
    if (warmup_steps < 0)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".warmup_steps=" << warmup_steps << "; expected >= 0.");
    if (!std::isfinite(multiplier_max) || multiplier_max <= 1.0f)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".multiplier_max=" << multiplier_max << "; expected finite > 1.");
    if (multiplier_mode != kMultiplierClipped && multiplier_mode != kMultiplierUnclipped)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".multiplier_mode='" << multiplier_mode
            << "'; expected clipped or unclipped.");
    if (multiplier_mode == kMultiplierUnclipped && multiplier_max != kDefaultMultiplierMax)
        anet::log::warn() << "Ignored " << prefix << ".multiplier_max=" << multiplier_max
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

    // 解決後の値だけで判断し、無効な葉も検証済みのうえで構築時に1回知らせる。
    if (statistics_mode == kStatisticsVarianceFeedback) {
        const auto warn_ignored = [&](const char* key, const auto& value, const auto& default_value) {
            if (value != default_value)
                anet::log::warn() << "Ignored " << prefix << ".obs_norm." << key << "=" << anet::json(value).dump()
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

RndNoveltyEstimator::RndNoveltyEstimator(const RndNoveltyEstimatorConfig& config,
    const anet::TensorSpecMap& observation_spec, const torch::Device& device,
    const anet::SeedMaker& seed_maker)
    : config_(config), device_(device), error_stats_({}, 0.0), target_feature_stats_({config.output_dim}, 0.0)
{
    // 互いに独立した重みを作り、教師側は勾配と学習時の状態更新を止める。
    auto normalized_spec = observation_spec;
    for (auto& [key, spec] : normalized_spec) {
        if (spec.dtype == torch::kUInt8 && !spec.IsDiscrete()) {
            spec.dtype = torch::kFloat32;
            byte_observation_keys_.push_back(key);
        }
    }
    auto head = std::make_shared<RndHeadFactory>(config.output_dim, kOutput,
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
    anet::rl::StateSpec state_spec;
    state_spec.obs_spec = normalized_spec;
    if (config.statistics_mode == kStatisticsWhiten)
        normalizer_ = anet::rl::ObservationNormalizerFactory(config.obs_norm).CreateObservationNormalizer(state_spec);
    else
        feedback_stats_.emplace(normalized_spec);
    for (const auto key : {kErrorMean, kErrorStd, kErrorStep, kClipRatio, kPredictorLoss,
        kTargetFeatureVar, kTargetFeatureMax})
        diagnostics_.emplace(key, std::numeric_limits<float>::quiet_NaN());
}

torch::Tensor RndNoveltyEstimator::Estimate(const anet::TensorDict& observations)
{
    ANET_PROFILE_FUNC();
    torch::NoGradGuard no_grad;
    // 入力のdeviceによらず、観測の正規化と統計をCPUで扱う。
    ANET_PROFILE_SCOPE(observation_transfer);
    auto float_observations = observations.To(torch::kCPU);
    // 採点は更新前の観測統計を使う。入力転送は lane のループへ入れない。
    ANET_PROFILE_SCOPE_NEXT(normalize);
    for (const auto& key : byte_observation_keys_)
        float_observations.Set(key, float_observations.At(key).to(torch::kFloat32));
    anet::TensorDict normalized;
    torch::Tensor valid_lanes;
    if (feedback_stats_) {
        auto result = feedback_stats_->NormalizeAndUpdate(float_observations, config_.obs_norm.clip_range);
        normalized = std::move(result.first);
        valid_lanes = std::move(result.second);
    } else {
        normalized = normalizer_->Normalize(float_observations);
    }
    ANET_PROFILE_SCOPE_NEXT(input_transfer);
    const auto input = normalized.To(device_);
    ANET_PROFILE_SCOPE_NEXT(target_forward);
    const auto target = target_->Forward(input).At(kOutput);
    ANET_PROFILE_SCOPE_NEXT(predictor_forward);
    // 採点には勾配を使わず、次の操作へ計算グラフを持ち越さない。
    const auto prediction = predictor_->Forward(input).At(kOutput);
    const auto prediction_errors = (prediction - target).square().mean(-1);
    ANET_PROFILE_SCOPE_NEXT(error_transfer);
    const auto errors = prediction_errors.to(torch::kCPU);
    const auto target_features = target.to(torch::kCPU);
    // CPU へまとめて戻した結果を検証し、非有限値で累積統計を汚さない。
    const auto invalid = (~torch::isfinite(errors)).nonzero();
    if (invalid.numel() != 0) {
        const auto lane = invalid[0][0].item<int64_t>();
        ANET_SYSTEM_ERROR("Invalid RND prediction_error at lane=" << lane
            << " value=" << errors[lane].item<float>() << "; expected finite value.");
    }
    ANET_PROFILE_SCOPE_NEXT(statistics);
    auto alpha = torch::ones_like(errors, torch::TensorOptions().dtype(torch::kFloat32));
    if (feedback_stats_) {
        alpha += feedback_stats_->UpdateErrors(errors);
        const auto counts = torch::arange(errors.size(0), torch::kInt64) + score_count_;
        alpha.masked_fill_(~valid_lanes | (counts < config_.warmup_steps), 1.0f);
        diagnostics_[kErrorMean] = static_cast<float>(feedback_stats_->ErrorMean());
        diagnostics_[kErrorStd] = static_cast<float>(feedback_stats_->ErrorStd());
    } else {
        const auto std = error_stats_.GetStd().item<float>();
        if (score_count_ >= config_.warmup_steps && error_stats_.GetCount() > 0 && std > 0.0f)
            alpha = 1.0f + (errors - error_stats_.GetMean().item<float>()) / std;
        error_stats_.Update(errors);
        /// @todo 正規化結果を捨てる二重計算の費用を測定し、統計更新専用APIの追加を判断する。
        normalizer_->NormalizeAndUpdateStats(float_observations);
        diagnostics_[kErrorMean] = error_stats_.GetMean().item<float>();
        diagnostics_[kErrorStd] = error_stats_.GetStd().item<float>();
    }
    if (config_.multiplier_mode == kMultiplierClipped)
        alpha = alpha.clamp(1.0f, config_.multiplier_max);
    score_count_ += observations.Size(0);
    target_feature_stats_.Update(target_features);
    diagnostics_[kTargetFeatureVar] = TargetFeatureVariance();
    diagnostics_[kTargetFeatureMax] = target_features.abs().max().item<float>();
    diagnostics_[kErrorStep] = errors.mean().item<float>();
    diagnostics_[kClipRatio] = config_.multiplier_mode == kMultiplierClipped
        ? (alpha == config_.multiplier_max).to(torch::kFloat32).mean().item<float>() : 0.0f;
    ANET_PROFILE_SCOPE_NEXT(result_transfer);
    return alpha.to(observations.device());
}

bool RndNoveltyEstimator::Learn(const anet::TensorDict& observations)
{
    ANET_PROFILE_FUNC();
    torch::NoGradGuard no_grad;
    
    // warm-upではoptimizerを進めず、呼び出し側へ未実行を返す。
    if (config_.warmup_steps > 0 && score_count_ <= config_.warmup_steps) return false;

    // 渡された観測すべてを正規化する。採点側で育てた統計は読むだけにする。
    ANET_PROFILE_SCOPE(observation_transfer);
    auto float_observations = observations.To(torch::kCPU);
    ANET_PROFILE_SCOPE_NEXT(normalize);
    for (const auto& key : byte_observation_keys_)
        float_observations.Set(key, float_observations.At(key).to(torch::kFloat32));
    const auto normalized = feedback_stats_ ? feedback_stats_->Normalize(float_observations, config_.obs_norm.clip_range)
        : normalizer_->Normalize(float_observations);
    ANET_PROFILE_SCOPE_NEXT(input_transfer);
    const auto input = normalized.To(device_);
    ANET_PROFILE_SCOPE_NEXT(target_forward);
    const auto target = target_->Forward(input).At(kOutput);
    ANET_PROFILE_SCOPE_NEXT(predictor_forward);
    torch::AutoGradMode enable_grad(true);
    const auto prediction = predictor_->Forward(input).At(kOutput);
    const auto prediction_errors = (prediction - target).square().mean(-1);

    // 独立した学習入力も有限性を確かめ、異常な誤差で重みやmomentを汚さない。
    ANET_PROFILE_SCOPE_NEXT(error_transfer);
    const auto errors = prediction_errors.detach().to(torch::kCPU);
    const auto invalid = (~torch::isfinite(errors)).nonzero();
    if (invalid.numel() != 0) {
        const auto lane = invalid[0][0].item<int64_t>();
        ANET_SYSTEM_ERROR("Invalid RND prediction_error at lane=" << lane
            << " value=" << errors[lane].item<float>() << "; expected finite value.");
    }
    diagnostics_[kPredictorLoss] = errors.mean().item<float>();
    ANET_PROFILE_SCOPE_NEXT(backward);
    const auto loss = prediction_errors.mean();
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
    auto bytes = archive.Write(std::string(kCheckpointFormat));
    bytes += archive.Write(config_.statistics_mode);
    bytes += archive.WriteTorchObject(target_);
    bytes += archive.WriteTorchObject(predictor_);
    bytes += archive.WriteTorchObject(*optimizer_);
    if (feedback_stats_)
        bytes += feedback_stats_->Save(archive);
    else {
        bytes += normalizer_->Save(archive);
        bytes += error_stats_.Save(archive);
    }
    bytes += target_feature_stats_.Save(archive);
    bytes += archive.Write(score_count_);
    return bytes;
}

int64_t RndNoveltyEstimator::Load(anet::InputArchive& archive)
{
    ANET_PROFILE_FUNC();
    std::string format;
    auto bytes = archive.Read(format);
    if (format != kCheckpointFormat)
        ANET_SYSTEM_ERROR("Invalid RND checkpoint format='" << format << "'; expected='" << kCheckpointFormat << "'.");
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
            anet::log::warn() << "RND checkpoint optimizer options replaced: learning_rate=" << options.lr()
                << " -> " << config_.learning_rate << ", weight_decay=" << options.weight_decay()
                << " -> " << config_.weight_decay << ", adam_eps=" << options.eps() << " -> " << config_.adam_eps;
        options = torch::optim::AdamWOptions(config_.learning_rate).weight_decay(config_.weight_decay).eps(config_.adam_eps);
    }
    if (feedback_stats_)
        bytes += feedback_stats_->Load(archive);
    else {
        bytes += normalizer_->Load(archive);
        bytes += error_stats_.Load(archive);
    }
    bytes += target_feature_stats_.Load(archive);
    bytes += archive.Read(score_count_);
    if (score_count_ < 0)
        ANET_SYSTEM_ERROR("Invalid RND checkpoint score_count=" << score_count_ << "; expected >= 0.");
    if (feedback_stats_ && score_count_ != feedback_stats_->Count())
        ANET_SYSTEM_ERROR("Invalid RND checkpoint score_count=" << score_count_
            << "; expected variance_feedback count=" << feedback_stats_->Count() << ".");
    for (auto& [key, value] : diagnostics_) value = std::numeric_limits<float>::quiet_NaN();
    if (feedback_stats_ && feedback_stats_->Count() > 0) {
        diagnostics_[kErrorMean] = static_cast<float>(feedback_stats_->ErrorMean());
        diagnostics_[kErrorStd] = static_cast<float>(feedback_stats_->ErrorStd());
    } else if (!feedback_stats_ && error_stats_.GetCount() > 0) {
        diagnostics_[kErrorMean] = error_stats_.GetMean().item<float>();
        diagnostics_[kErrorStd] = error_stats_.GetStd().item<float>();
    }
    if (target_feature_stats_.GetCount() > 0)
        diagnostics_[kTargetFeatureVar] = TargetFeatureVariance();
    return bytes;
}

std::unique_ptr<LifelongNoveltyEstimator> LifelongNoveltyEstimatorFactory::Create(
    const anet::ConfigData& data, const std::string& prefix, const anet::TensorSpecMap& observation_spec,
    const torch::Device& device, const anet::SeedMaker& seed_maker)
{
    return std::make_unique<RndNoveltyEstimator>(RndNoveltyEstimatorConfig(data, prefix),
        observation_spec, device, seed_maker);
}
