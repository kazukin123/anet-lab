#include "intrinsic_rnd.hpp"
#include <cmath>
#include <limits>
#include <sstream>
#include <ATen/CPUGeneratorImpl.h>
#include "anet/profile.hpp"
#include "nn_heads.hpp"

using namespace anet::rl::intrinsic;

constexpr char kTargetSeed[] = "intrinsic.target";
constexpr char kPredictorSeed[] = "intrinsic.predictor";
constexpr char kTrainSeed[] = "intrinsic.train";
constexpr char kOutput[] = "novelty";
constexpr char kErrorMean[] = "error_mean";
constexpr char kErrorStd[] = "error_std";
constexpr char kErrorStep[] = "error_step";
constexpr char kClipRatio[] = "clip_ratio";
constexpr char kPredictorLoss[] = "predictor_loss";
constexpr char kCheckpointFormat[] = "RndNoveltyEstimator-1";

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
    // 切り替え、例外時も元へ戻す。実行中の抽選には専用 generator を使う。
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
}

RndNoveltyEstimatorConfig::RndNoveltyEstimatorConfig(
    const anet::ConfigData& data, const std::string& prefix)
    : anet::Config(data, prefix), net(data, prefix + ".net")
{
    ANET_READ_CONFIG(data, output_dim);
    ANET_READ_CONFIG(data, multiplier_max);
    ANET_READ_CONFIG(data, warmup_steps);
    ANET_READ_CONFIG(data, train_fraction);
    ANET_READ_CONFIG(data, learning_rate);
    ANET_READ_CONFIG(data, weight_decay);
    // RND は平均を引いた白色化を使い、Q 入力用の既定とは分離する。
    obs_norm.use_centering = true;
    obs_norm.use_robust_update = false;
    obs_norm.post_process_type = anet::rl::kPostProcessNone;
    obs_norm.clip_range = 5.0f;
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
    if (!std::isfinite(train_fraction) || train_fraction <= 0.0f || train_fraction > 1.0f)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".train_fraction=" << train_fraction << "; expected 0 < value <= 1.");
    if (!std::isfinite(learning_rate) || learning_rate <= 0.0f)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".learning_rate=" << learning_rate << "; expected finite > 0.");
    if (!std::isfinite(weight_decay) || weight_decay < 0.0f)
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".weight_decay=" << weight_decay << "; expected finite >= 0.");
    if (net.output_keys.empty())
        ANET_SYSTEM_ERROR("Invalid " << prefix << ".net: no Body outputs; expected configured Network body.");
    anet::rl::ValidateObservationNormalizerConfig(obs_norm, prefix + ".obs_norm");
}

RndNoveltyEstimator::RndNoveltyEstimator(const RndNoveltyEstimatorConfig& config,
    const anet::TensorSpecMap& observation_spec, const torch::Device& device,
    const anet::SeedMaker& seed_maker)
    : config_(config), device_(device), train_random_(seed_maker.MakeNamedSeed(kTrainSeed)), error_stats_({}, 0.0)
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
        anet::nn::WeightInitConfig{});
    const auto build = [&](const char* name) {
        const auto seed = seed_maker.MakeNamedSeed(name);
        ConstructionSeedScope scope(seed);
        return anet::nn::NetworkBuilder::BuildNetwork(config.net, normalized_spec, head, seed, torch::kCPU);
    };
    target_ = build(kTargetSeed);
    predictor_ = build(kPredictorSeed);
    target_->to(device);
    predictor_->to(device);
    target_->eval();
    for (auto& parameter : target_->parameters()) parameter.requires_grad_(false);
    optimizer_ = std::make_unique<torch::optim::AdamW>(predictor_->parameters(),
        torch::optim::AdamWOptions(config.learning_rate).weight_decay(config.weight_decay));
    anet::rl::StateSpec state_spec;
    state_spec.obs_spec = normalized_spec;
    normalizer_ = anet::rl::ObservationNormalizerFactory(config.obs_norm).CreateObservationNormalizer(state_spec);
    for (const auto key : {kErrorMean, kErrorStd, kErrorStep, kClipRatio, kPredictorLoss})
        diagnostics_.emplace(key, std::numeric_limits<float>::quiet_NaN());
}

torch::Tensor RndNoveltyEstimator::Multiplier(const anet::TensorDict& observations)
{
    ANET_PROFILE_FUNC();
    torch::NoGradGuard no_grad;
    // 採点は更新前の観測統計を使う。入力転送は lane のループへ入れない。
    ANET_PROFILE_SCOPE(normalize);
    auto float_observations = observations;
    for (const auto& key : byte_observation_keys_)
        float_observations.Set(key, observations.At(key).to(torch::kFloat32));
    const auto normalized = normalizer_->Normalize(float_observations);
    ANET_PROFILE_SCOPE_NEXT(input_transfer);
    const auto input = normalized.To(device_);
    ANET_PROFILE_SCOPE_NEXT(target_forward);
    const auto target = target_->Forward(input).At(kOutput);
    ANET_PROFILE_SCOPE_NEXT(predictor_forward);
    // 採点時のグラフを Update へ渡し、更新前統計での同じ入力を学習する。
    torch::AutoGradMode enable_grad(true);
    const auto prediction = predictor_->Forward(input).At(kOutput);
    pending_errors_ = (prediction - target).square().mean(-1);
    ANET_PROFILE_SCOPE_NEXT(error_transfer);
    const auto errors = pending_errors_.detach().to(torch::kCPU);
    // CPU へまとめて戻した結果を検証し、非有限値で累積統計を汚さない。
    const auto invalid = (~torch::isfinite(errors)).nonzero();
    if (invalid.numel() != 0) {
        const auto lane = invalid[0][0].item<int64_t>();
        ANET_SYSTEM_ERROR("Invalid RND prediction_error at lane=" << lane
            << " value=" << errors[lane].item<float>() << "; expected finite value.");
    }
    pending_errors_cpu_ = errors;
    ANET_PROFILE_SCOPE_NEXT(statistics);
    auto alpha = torch::ones_like(errors);
    const auto std = error_stats_.GetStd().item<float>();
    if (score_count_ >= config_.warmup_steps && error_stats_.GetCount() > 0 && std > 0.0f)
        alpha = (1.0f + (errors - error_stats_.GetMean().item<float>()) / std).clamp(1.0f, config_.multiplier_max);
    error_stats_.Update(errors);
    /// @todo 正規化結果を捨てる二重計算の費用を測定し、統計更新専用APIの追加を判断する。
    normalizer_->NormalizeAndUpdateStats(float_observations);
    ++score_count_;
    diagnostics_[kErrorMean] = error_stats_.GetMean().item<float>();
    diagnostics_[kErrorStd] = error_stats_.GetStd().item<float>();
    diagnostics_[kErrorStep] = errors.mean().item<float>();
    diagnostics_[kClipRatio] = (alpha == config_.multiplier_max).to(torch::kFloat32).mean().item<float>();
    return alpha;
}

void RndNoveltyEstimator::Update(const anet::TensorDict&)
{
    ANET_PROFILE_FUNC();
    // warm-up と空の抽選では、optimizer の moment と step も進めない。
    diagnostics_[kPredictorLoss] = std::numeric_limits<float>::quiet_NaN();
    if (score_count_ <= config_.warmup_steps || !pending_errors_.defined()) {
        pending_errors_ = torch::Tensor{};
        pending_errors_cpu_ = torch::Tensor{};
        return;
    }
    std::vector<int64_t> selected;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int64_t lane = 0; lane < pending_errors_.size(0); ++lane)
        if (uniform(train_random_) < config_.train_fraction) selected.push_back(lane);
    if (!selected.empty()) {
        torch::AutoGradMode enable_grad(true);
        ANET_PROFILE_SCOPE(selection_transfer);
        const auto indices = torch::tensor(selected, torch::kInt64);
        const auto device_indices = indices.to(device_);
        ANET_PROFILE_SCOPE_NEXT(backward);
        const auto loss = pending_errors_.index_select(0, device_indices).mean();
        diagnostics_[kPredictorLoss] = pending_errors_cpu_.index_select(0, indices).mean().item<float>();
        optimizer_->zero_grad();
        loss.backward();
        optimizer_->step();
    }
    pending_errors_ = torch::Tensor{};
    pending_errors_cpu_ = torch::Tensor{};
}

std::map<std::string, float> RndNoveltyEstimator::Diagnostics() const { return diagnostics_; }

int64_t RndNoveltyEstimator::Save(anet::OutputArchive& archive) const
{
    ANET_PROFILE_FUNC();
    // 固定教師を含む全学習状態を保存し、乱数の抽選列も再開する。
    auto bytes = archive.Write(std::string(kCheckpointFormat));
    bytes += archive.WriteTorchObject(target_);
    bytes += archive.WriteTorchObject(predictor_);
    bytes += archive.WriteTorchObject(*optimizer_);
    bytes += normalizer_->Save(archive);
    bytes += error_stats_.Save(archive);
    bytes += archive.Write(score_count_);
    std::ostringstream random_state;
    random_state << train_random_;
    bytes += archive.Write(random_state.str());
    return bytes;
}

int64_t RndNoveltyEstimator::Load(anet::InputArchive& archive)
{
    ANET_PROFILE_FUNC();
    std::string format;
    auto bytes = archive.Read(format);
    if (format != kCheckpointFormat)
        ANET_SYSTEM_ERROR("Invalid RND checkpoint format='" << format << "'; expected='" << kCheckpointFormat << "'.");
    bytes += archive.ReadTorchObject(target_);
    bytes += archive.ReadTorchObject(predictor_);
    bytes += archive.ReadTorchObject(*optimizer_);
    // moment と step は保存値を継承し、options は現在設定を優先する。
    for (auto& group : optimizer_->param_groups()) {
        auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
        if (options.lr() != config_.learning_rate || options.weight_decay() != config_.weight_decay)
            anet::log::warn() << "RND checkpoint optimizer options replaced: learning_rate=" << options.lr()
                << " -> " << config_.learning_rate << ", weight_decay=" << options.weight_decay()
                << " -> " << config_.weight_decay;
        options = torch::optim::AdamWOptions(config_.learning_rate).weight_decay(config_.weight_decay);
    }
    bytes += normalizer_->Load(archive);
    bytes += error_stats_.Load(archive);
    bytes += archive.Read(score_count_);
    if (score_count_ < 0)
        ANET_SYSTEM_ERROR("Invalid RND checkpoint score_count=" << score_count_ << "; expected >= 0.");
    std::string random_state;
    bytes += archive.Read(random_state);
    std::istringstream random_input(random_state);
    if (!(random_input >> train_random_)) ANET_SYSTEM_ERROR("Invalid RND checkpoint random state; expected mt19937_64 state.");
    pending_errors_ = torch::Tensor{};
    pending_errors_cpu_ = torch::Tensor{};
    for (auto& [key, value] : diagnostics_) value = std::numeric_limits<float>::quiet_NaN();
    if (error_stats_.GetCount() > 0) {
        diagnostics_[kErrorMean] = error_stats_.GetMean().item<float>();
        diagnostics_[kErrorStd] = error_stats_.GetStd().item<float>();
    }
    return bytes;
}

std::unique_ptr<LifelongNoveltyEstimator> LifelongNoveltyEstimatorFactory::Create(
    const anet::ConfigData& data, const std::string& prefix, const anet::TensorSpecMap& observation_spec,
    const torch::Device& device, const anet::SeedMaker& seed_maker)
{
    return std::make_unique<RndNoveltyEstimator>(RndNoveltyEstimatorConfig(data, prefix),
        observation_spec, device, seed_maker);
}
