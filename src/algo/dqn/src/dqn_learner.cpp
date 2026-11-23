#include "dqn_learner.hpp"
#include <algorithm>
#include <charconv>
#include <tuple>
#include <cmath>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <set>
#include <utility>
#include <unordered_set>
#include "anet/log.hpp"
#include "anet/profile.hpp"
#include "anet/tensor_check.hpp"
#include "anet/tensor_util.hpp"
#include "anet/str_util.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/replay_experience.hpp"
#include "anet/image.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/nn_head.hpp"
#include "dqn_heads.hpp"

using namespace anet::rl::dqn;
namespace LOG = anet::log;

constexpr const char* kIntrinsicOptimizerKeyPrefix = "intrinsic_optimizer.";

static constexpr std::string_view kPlasticityTargetPrefix = "plasticity_target_";

static constexpr std::string_view kPlasticityPrefix = "plasticity_";

// optimizerの保存契約は、roleの対応付け・Torch読込・診断で同じ値を使う。
constexpr const char* kOptimizerRolePrimary = "primary";

constexpr const char* kOptimizerRolePrimarySigma = "primary_sigma";

constexpr const char* kOptimizerRoleSkip = "skip";

constexpr const char* kOptimizerRoleSkipSigma = "skip_sigma";

constexpr const char* kOptimizerRolePrimarySkip = "primary_skip";

constexpr const char* kOptimizerSigmaRoleSuffix = "_sigma";

constexpr const char* kArchiveParamGroups = "param_groups";

constexpr const char* kArchiveParamGroupCount = "param_groups/size";

static std::vector<double> MakeAverageRanks(const std::vector<float>& values)
{
    std::vector<size_t> order(values.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        return values[lhs] < values[rhs];
    });

    std::vector<double> ranks(values.size());
    size_t begin = 0;
    while (begin < order.size()) {
        size_t end = begin + 1;
        while (end < order.size() && values[order[end]] == values[order[begin]]) {
            ++end;
        }
        const double average_rank = (static_cast<double>(begin + 1) + static_cast<double>(end)) / 2.0;
        for (size_t i = begin; i < end; ++i) {
            ranks[order[i]] = average_rank;
        }
        begin = end;
    }
    return ranks;
}

static float CalculateSpearmanCorrelation(const torch::Tensor& lhs_cpu, const torch::Tensor& rhs_cpu)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    if (!lhs_cpu.defined() || !rhs_cpu.defined()
        || lhs_cpu.dim() != 1 || rhs_cpu.sizes() != lhs_cpu.sizes()
        || lhs_cpu.size(0) < 2) {
        return nan;
    }

    // clipで生じた同値を平均順位へまとめ、minibatch内の順位相関だけをCPUで求める。
    const auto lhs = lhs_cpu.to(torch::kFloat32).contiguous();
    const auto rhs = rhs_cpu.to(torch::kFloat32).contiguous();
    if (!torch::isfinite(lhs).all().item<bool>() || !torch::isfinite(rhs).all().item<bool>()) {
        return nan;
    }
    std::vector<float> lhs_values(lhs.data_ptr<float>(), lhs.data_ptr<float>() + lhs.numel());
    std::vector<float> rhs_values(rhs.data_ptr<float>(), rhs.data_ptr<float>() + rhs.numel());
    const auto lhs_ranks = MakeAverageRanks(lhs_values);
    const auto rhs_ranks = MakeAverageRanks(rhs_values);
    const double lhs_mean = std::accumulate(lhs_ranks.begin(), lhs_ranks.end(), 0.0) / lhs_ranks.size();
    const double rhs_mean = std::accumulate(rhs_ranks.begin(), rhs_ranks.end(), 0.0) / rhs_ranks.size();

    double covariance = 0.0;
    double lhs_square_sum = 0.0;
    double rhs_square_sum = 0.0;
    for (size_t i = 0; i < lhs_ranks.size(); ++i) {
        const double lhs_centered = lhs_ranks[i] - lhs_mean;
        const double rhs_centered = rhs_ranks[i] - rhs_mean;
        covariance += lhs_centered * rhs_centered;
        lhs_square_sum += lhs_centered * lhs_centered;
        rhs_square_sum += rhs_centered * rhs_centered;
    }
    if (lhs_square_sum == 0.0 || rhs_square_sum == 0.0) {
        return nan;
    }
    return static_cast<float>(std::clamp(
        covariance / std::sqrt(lhs_square_sum * rhs_square_sum), -1.0, 1.0));
}

Learner::Learner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
    const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed, anet::seed_t agent_seed,
    std::shared_ptr<ActionPolicy> target_policy, const LearnerForwardResources& forward_resources,
    std::optional<StuckerConfig> stucker_config, std::optional<anet::seed_t> target_seed,
    std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe_config)
    : RandomHolder(target_seed), config_(config), learning_rate_(config.learning_rate), weight_decay_(config.weight_decay), per_beta_(config.per_beta), soft_update_tau_(model.GetSoftUpdateTauConfig()), gamma_(config.gamma), intrinsic_gamma_(config.intrinsic.gamma),
      munchausen_alpha_(config.munchausen.alpha), munchausen_entropy_tau_(config.munchausen.entropy_tau),
      munchausen_values_(config.munchausen.Resolve(munchausen_alpha_.Value(), munchausen_entropy_tau_.Value())),
      has_optimizer_schedule_(config.learning_rate.type != kConstantScheduleType || config.weight_decay.type != kConstantScheduleType), stucker_config_(stucker_config), model_(model), vars_(vars), obs_norm_(std::move(obs_norm))
    , num_envs_(batch_env_spec.num_envs)
    , n_actions_(env_spec.action_spec.GetNumActions())//, state_dim_(env_spec.state_spec.CalcFlattenDim())
    , device_(std::move(device))
    , target_policy_(std::move(target_policy))
    , plasticity_probe_random_(anet::SeedMaker(agent_seed).MakeNamedSeed("plasticity_probe"))
    , policy_churn_probe_random_(anet::SeedMaker(agent_seed).MakeNamedSeed("policy_churn_probe"))
    , replay_fit_probe_random_(anet::SeedMaker(agent_seed).MakeNamedSeed("replay_fit_probe"))
    , noisy_probe_random_(anet::SeedMaker(agent_seed).MakeNamedSeed("noisy_probe.sample"))
    , attention_entropy_probe_random_(anet::SeedMaker(agent_seed).MakeNamedSeed("attention_entropy_probe"))
    , noisy_probe_config_(std::move(noisy_probe_config))
{
    // 名前付き参照と受け取ったResourceの対応を、NNの準備や資源確保より前に確認する。
    config_.grad_noise.Validate(config_.full_keys, "grad_noise.");
    const auto validate_resource = [&](const std::string& purpose, const std::string& key,
        const anet::nn::NetworkForwardConfig& resource) {
        const bool matches = key.empty() ? resource.IsDefault()
            : !resource.IsDefault() && resource.key.ends_with(MakeNetworkForwardConfigKey(key));
        if (!matches) {
            ANET_SYSTEM_ERROR("Mismatched " << config_.full_keys.GetFullKey(purpose + ".nn_forward_key") << "=" << key
                << "; supplied resource=" << resource.key
                << "; expected " << (key.empty() ? "an explicit default resource" : "a named resource ending in " + MakeNetworkForwardConfigKey(key)) << ".");
        }
    };
    if (config_.enabled) {
        validate_resource("current", config_.current.nn_forward_key, forward_resources.current);
        validate_resource("target", config_.target.nn_forward_key, forward_resources.target);
    }

    // 用途ごとに設定を準備し、既存の学習/τ RNGとは独立した State を所有する。
    const anet::SeedMaker forward_seeds(GetSeed());
    const auto& current = config_.enabled ? forward_resources.current : anet::nn::NetworkForwardConfig::GetDefault();
    const auto& target = config_.enabled ? forward_resources.target : anet::nn::NetworkForwardConfig::GetDefault();
    const auto prepare = [](const std::shared_ptr<anet::nn::Network>& network,
        const anet::nn::NetworkForwardConfig& config, anet::seed_t seed) {
        return network->CreateForwardExecutor(config, anet::nn::NetworkForwardClock::kTransientRows, seed);
    };
    current_forward_ = prepare(model_.GetOnlineNetwork(), current, forward_seeds.MakeNamedSeed("network_forward.current"));
    if (model_.GetSkipOnlineNetwork()) {
        if (config_.use_amp && !config_.use_amp_bf16) {
            ANET_SYSTEM_ERROR("Skip learning requires " << config_.full_keys.GetFullKey("use_amp")
                << "=false or " << config_.full_keys.GetFullKey("use_amp_bf16") << "=true.");
        }
        skip_current_forward_ = prepare(model_.GetSkipOnlineNetwork(), current,
            forward_seeds.MakeNamedSeed("network_forward.skip_current"));
    }
    target_forwards_.target = prepare(model_.GetTargetNetwork(), target, forward_seeds.MakeNamedSeed("network_forward.target"));
    // target用途のonline側もNetwork別の箱を持ち、専用named seedで抽選する。
    target_forwards_.online = prepare(model_.GetOnlineNetwork(), target,
        forward_seeds.MakeNamedSeed("network_forward.target_online"));
    diagnostic_forwards_.online = prepare(model_.GetOnlineNetwork(), anet::nn::NetworkForwardConfig::GetDefault(), 0);
    diagnostic_forwards_.target = prepare(model_.GetTargetNetwork(), anet::nn::NetworkForwardConfig::GetDefault(), 0);
    if (config_.enabled && config_.munchausen.enabled && config_.munchausen.log_policy_mode == MunchausenValues::kLogPolicyOnlineReuse) {
        model_.GetOnlineNetwork()->ValidateForwardReuse(current, target);
    }

    // Credit計算
    if (config_.replay_ratio > 0) {
        // RRモード:  U = (N * RR) / B
        earned_credit_ = static_cast<float>(num_envs_) * config_.replay_ratio / config_.replay_batch_size;
    } else {
        // Intervalモード: U = 1.0 / Interval
        earned_credit_ = 1.0f / static_cast<float>(std::max(1, config_.update_interval));
    }

    LOG::info() << "Learner: U = " << earned_credit_;
    if (!config_.enabled) {
        LOG::info() << "Learner: learning is disabled (" << config_.full_keys.GetFullKey("enabled") << "=false)."
            << " No replay buffer and no gradient updates.";
    }

    if (device_.is_cuda()) {
        per_priority_copy_stream_ = at::cuda::getStreamFromPool(false);
    }
}

std::optional<float> Learner::GetScalar(const std::string& key, int64_t index) const
{
    for (size_t metric = 0; metric < kValueDecompositionMetricKeys.size(); ++metric) {
        if (key != kValueDecompositionMetricKeys[metric]) continue;
        if (!decomposition_metrics_.defined()) return std::numeric_limits<float>::quiet_NaN();
        if (!decomposition_metrics_cpu_.defined()) decomposition_metrics_cpu_ = decomposition_metrics_.to(torch::kCPU);
        return decomposition_metrics_cpu_.data_ptr<float>()[metric];
    }
    if (key == kLearnerScheduleMetricKeys[LearnerScheduleMetric::LearningRate]) return learning_rate_.Value();
    if (key == kLearnerScheduleMetricKeys[LearnerScheduleMetric::WeightDecay]) return weight_decay_.Value();
    if (key == kLearnerScheduleMetricKeys[LearnerScheduleMetric::SoftUpdateTau]) return soft_update_tau_.Value();
    if (key == kLearnerScheduleMetricKeys[LearnerScheduleMetric::Gamma]) return gamma_.Value();
    if (key == kLearnerScheduleMetricKeys[LearnerScheduleMetric::MunchausenAlpha]) return munchausen_alpha_.Value();
    if (key == kLearnerScheduleMetricKeys[LearnerScheduleMetric::MunchausenEntropyTau]) return munchausen_entropy_tau_.Value();
    if (const auto value = ReadDiagnosticScalar(ParseNoisyMetric(key),
        [&](size_t metric) -> std::optional<float> { return noisy_.values[metric]; })) return value;
    if (key.starts_with(kPlasticityProbePrefix)) {
        const auto suffix = key.substr(kPlasticityProbePrefix.size());
        return ReadDiagnosticScalar(anet::nn::ParsePlasticityMetricSuffix(suffix), [&](auto metric) -> std::optional<float> {
        if (!plasticity_.probe_request.Contains(metric)) {
            return std::nullopt;
        }
        if (!plasticity_.probe_features.defined()) return std::nullopt;
        if (!plasticity_.probe_metrics_cache.has_value()) {
            plasticity_.probe_metrics_cache = anet::nn::ComputePlasticityMetrics(
                plasticity_.probe_features, plasticity_.probe_request);
        }
        const auto value = plasticity_.probe_metrics_cache->Get(metric);
        if (!value.has_value()) {
            ANET_SYSTEM_ERROR("Learner plasticity probe cache is missing a requested metric.");
        }
        return value;
        });
    }
    if (key.find(ReplayBuffer::kKeyPrefix) == 0) {
        // ReplayBufferを持たないRun(learner.enabled=false)でも、この prefix は Learner が所有する
        // 既知の名前空間なので値未成立としてNaNを返す。nullopt は購読側で未知key扱いになり、
        // 他sourceを探索した末にupdateごとのWARNへ落ちる。
        if (replay_buffer_ == nullptr) return std::numeric_limits<float>::quiet_NaN();
        return replay_buffer_->GetScalar(key);
    }

    return std::nullopt;
}

void Learner::ConfigureScalarMetricSubscriptions(const std::vector<ScalarMetricSubscription>& subscriptions)
{
    DistributeDiagnosticSubscriptions(subscriptions);
    ValidateDiagnosticSubscriptions();
    PrepareDiagnosticResources();
}

void Learner::DistributeDiagnosticSubscriptions(const std::vector<ScalarMetricSubscription>& subscriptions)
{
    attention_ = AttentionEntropyState{};
    grad_noise_ = GradientNoiseState{};
    plasticity_ = PlasticityState{};
    noisy_ = NoisyDiagnosticState{};
    policy_churn_.demand.clear();
    replay_fit_demand_.clear();
    const auto update_min = [](int current, int candidate, bool was_enabled) {
        return was_enabled ? std::min(current, candidate) : candidate;
    };
    for (const auto& subscription : subscriptions) {
        if (subscription.scope != RunnerScope::TRAIN || subscription.event != EventType::LEARN) continue;
        const auto& key = subscription.source_key;
        if (subscription.target == EventField::UPDATE_RESULT) {
            if (const auto layer = ParseAttentionEntropyMetric(key)) {
                if (subscription.interval < 1) {
                    ANET_SYSTEM_ERROR("Invalid attention metric interval=" << subscription.interval << " key='" << key << "' expected >=1.");
                }
                if (config_.enabled) attention_.demand.emplace_back(*layer, subscription.interval);
                continue;
            }
            if (const auto metric = grad_noise_.demand.Add(key, subscription.interval)) {
                const auto i = *metric;
                if (i == GradientNoiseMetric::Scale || i == GradientNoiseMetric::SignalRaw || i == GradientNoiseMetric::VarianceRaw) grad_noise_.subscribed.raw = true;
                else if (i == GradientNoiseMetric::ScaleAdamFeature || i == GradientNoiseMetric::SignalAdamFeatureRaw || i == GradientNoiseMetric::VarianceAdamFeatureRaw) grad_noise_.subscribed.adam[1] = true;
                else if (i == GradientNoiseMetric::ScaleAdamReadout || i == GradientNoiseMetric::SignalAdamReadoutRaw || i == GradientNoiseMetric::VarianceAdamReadoutRaw) grad_noise_.subscribed.adam[2] = true;
                else grad_noise_.subscribed.adam[0] = true;
                continue;
            }
        }
        if (const auto metric = noisy_.demand.Add(key, subscription.interval)) {
            continue;
        }
        if (config_.enabled && subscription.target == EventField::UPDATE_RESULT) {
            if (const auto metric = replay_fit_demand_.Add(key, subscription.interval)) {
                continue;
            }
        }
        if (subscription.target == EventField::UPDATE_RESULT) {
            if (const auto metric = policy_churn_.demand.Add(key, subscription.interval)) {
                continue;
            }
        }
        if (ParseDiagnosticMetric(key, kPlasticityWeightNormMetricKeys)) {
            plasticity_.weight_norm_interval = update_min(
                plasticity_.weight_norm_interval, subscription.interval, plasticity_.weight_norm_enabled);
            plasticity_.weight_norm_enabled = true;
        } else if (key.starts_with(kPlasticityProbePrefix)) {
            const auto metric = anet::nn::ParsePlasticityMetricSuffix(
                key.substr(kPlasticityProbePrefix.size()));
            if (metric.has_value()) plasticity_.probe.Add(*metric, subscription.interval);
        } else if (key.starts_with(kPlasticityTargetPrefix)) {
            const auto metric = anet::nn::ParsePlasticityMetricSuffix(
                key.substr(kPlasticityTargetPrefix.size()));
            if (metric.has_value()) plasticity_.target.Add(*metric, subscription.interval);
        } else if (key.starts_with(kPlasticityPrefix)) {
            const auto metric = anet::nn::ParsePlasticityMetricSuffix(
                key.substr(kPlasticityPrefix.size()));
            if (metric.has_value()) plasticity_.online.Add(*metric, subscription.interval);
        }
    }

}

void Learner::ValidateDiagnosticSubscriptions()
{
    if (!replay_fit_demand_.empty() && target_policy_->GetConfig().IsThompsonSampling()) {
        ANET_SYSTEM_ERROR("Replay fit requires a deterministic target policy; " << target_policy_->GetConfig().full_keys.GetFullKey("policy_type") << "="
            << target_policy_->GetConfig().policy_type << " expected " << kActionPolicyTypeStr_Greedy << ", "
            << kActionPolicyTypeStr_EpsilonGreedy << ", or " << kActionPolicyTypeStr_UQE << ".");
    }
    // 購読のある構成だけで値域・精度と測定用parameterを確定する。
    const bool grad_noise_split = grad_noise_.subscribed.adam[1] || grad_noise_.subscribed.adam[2];
    if (!grad_noise_.demand.empty()) {
        const auto& setting = config_.grad_noise;
        const int batch = config_.replay_batch_size;
        if (batch <= 0 || batch % setting.num_micro_batches != 0) {
            ANET_SYSTEM_ERROR("Invalid " << config_.full_keys.GetFullKey("grad_noise.num_micro_batches") << "="
                              << setting.num_micro_batches << "; expected integer >=2 dividing " << config_.full_keys.GetFullKey("replay_batch_size") << "=" << batch);
        }
        if (config_.use_amp && !config_.use_amp_bf16) {
            ANET_SYSTEM_ERROR("Invalid " << config_.full_keys.GetFullKey("use_amp") << "=true "
                << config_.full_keys.GetFullKey("use_amp_bf16") << "=false for grad_noise metrics; expected FP32 or BF16");
        }
        for (auto& estimator : grad_noise_.estimators)
            estimator = anet::GradientNoiseEstimator(setting.ema_alpha);
        std::set<int> intervals;
        for (const auto& entry : grad_noise_.demand)
            intervals.insert(entry.interval);
        if (intervals.size() > 1) {
            std::ostringstream list;
            for (const auto interval : intervals) {
                if (list.tellp() > 0) list << ",";
                list << interval;
            }
            LOG::warn_once() << "Gradient noise subscription intervals differ: intervals=[" << list.str()
                        << "]; shared EMA updates at the union of measurement points.";
        }
        if (!grad_noise_split) {
            // 群別未購読時はfeatureフラグが結果に影響しないため、既定のfalseで全parameterを渡す。
            for (const auto& parameter : model_.GetOnlineParameters())
                grad_noise_.parameters.push_back({ .parameter = parameter });
        }
    }
}

void Learner::PrepareDiagnosticResources()
{
    const auto missing_config_hint = [this](const std::string& key) -> const char* {
        return config_.full_keys.Contains(key)
            ? ""
            : " This agent does not read this setting; remove the metric subscription that requires it.";
    };
    const bool grad_noise_split = grad_noise_.subscribed.adam[1] || grad_noise_.subscribed.adam[2];
    // Agent が解決した sample/call/batch 設定の検証は NN に委ねる。
    if (config_.enabled && std::ranges::any_of(noisy_.demand, [](const auto& entry) { return entry.metric >= NoisyMetric::QAbsDiff; })
        && !model_.GetOnlineNetwork()->GetNoisyParameterEntries().empty()) {
        if (!noisy_probe_config_) {
            ANET_SYSTEM_ERROR(config_.full_keys.GetFullKey("noisy_probe")
                << " requires a resolved sample forward config." << missing_config_hint("noisy_probe"));
        }
        noisy_.sample_forward = model_.GetOnlineNetwork()->CreateForwardExecutor(*noisy_probe_config_,
            anet::nn::NetworkForwardClock::kTransientRows,
            anet::SeedMaker(GetSeed()).MakeNamedSeed("network_forward.noisy_probe"));
    }

    // hard update と同一位相だけを観測する interval を、設定単位で一度だけ診断する。
    const int hard_update_interval = model_.GetHardUpdateInterval();
    if (hard_update_interval > 0) {
        for (const auto& entry : policy_churn_.demand) {
            const int metrics_interval = entry.interval;
            const int phase_count = hard_update_interval / std::gcd(hard_update_interval, metrics_interval);
            if (phase_count == 1) {
                LOG::warn_once(std::to_string(metrics_interval)) << "Policy churn metrics observe one hard-target phase: target_sync_age=0"
                    << " metrics_interval=" << metrics_interval
                    << " " << model_.GetHardUpdateIntervalKey() << "=" << hard_update_interval << ".";
            }
        }
    }

    if (!plasticity_.online.enabled && !plasticity_.target.enabled
        && !plasticity_.probe.enabled && !plasticity_.weight_norm_enabled
        && !std::ranges::any_of(noisy_.demand, [](const auto& entry) { return entry.metric < NoisyMetric::QAbsDiff; })
        && !grad_noise_split && attention_.demand.empty()) return;
    const auto branch_names = model_.GetOnlineNetwork()->GetBranchNames();
    std::ostringstream available;
    for (size_t i = 0; i < branch_names.size(); ++i) {
        if (i > 0) available << ", ";
        available << branch_names[i];
    }
    if (config_.final_feature_branch_key.empty()) {
        ANET_SYSTEM_ERROR(config_.full_keys.GetFullKey("final_feature_branch_key")
            << "='' is required when feature-boundary diagnostics are subscribed; available_branches=["
            << available.str() << "]" << missing_config_hint("final_feature_branch_key"));
    }
    if (std::ranges::find(branch_names, config_.final_feature_branch_key) == branch_names.end()) {
        ANET_SYSTEM_ERROR("Unknown " << config_.full_keys.GetFullKey("final_feature_branch_key") << "='"
            << config_.final_feature_branch_key << "'. available_branches=[" << available.str()
            << "]" << missing_config_hint("final_feature_branch_key"));
    }
    if (grad_noise_split) {
        const auto split = model_.GetOnlineNetwork()->GetParameterSplit(config_.final_feature_branch_key);
        for (const auto& parameter : split.feature) grad_noise_.parameters.push_back({ .parameter = parameter, .feature = true });
        for (const auto& parameter : split.readout) grad_noise_.parameters.push_back({ .parameter = parameter, .feature = false });
    }
    plasticity_.online_capture = anet::nn::NetworkForwardCapture({ config_.final_feature_branch_key });
    plasticity_.target_capture = anet::nn::NetworkForwardCapture({ config_.final_feature_branch_key });
}

ReplayFitRequirements anet::rl::dqn::ResolveReplayFitRequirements(
    const std::array<bool, kReplayFitMetricKeys.size()>& request, bool use_per)
{
    // 購読時のResource確保と測定回ごとの処理で、同じ指標依存を使う。
    const bool uniform = request[ReplayFitMetric::UniformTdMean] || (use_per && request[ReplayFitMetric::PerSelectivity]);
    return ReplayFitRequirements{
        .uniform = uniform,
        .per = use_per && (request[ReplayFitMetric::PerTdMean] || request[ReplayFitMetric::PerSelectivity]),
        .td_u = request[ReplayFitMetric::UnsampledTdMean] || request[ReplayFitMetric::UnsampledTdRatio] || uniform,
        .td_s = request[ReplayFitMetric::SampledTdMean] || request[ReplayFitMetric::UnsampledTdRatio] || uniform,
        .loss_u = request[ReplayFitMetric::UnsampledLossMean] || request[ReplayFitMetric::UnsampledLossRatio],
        .loss_s = request[ReplayFitMetric::SampledLossMean] || request[ReplayFitMetric::UnsampledLossRatio],
        .counts = request[ReplayFitMetric::UnsampledCount] || request[ReplayFitMetric::SampledCount] || request[ReplayFitMetric::UnsampledAgeMean] || request[ReplayFitMetric::SampledAgeMean] || uniform,
    };
}

std::optional<ReplayFitMetrics> Learner::CaptureReplayFit(const ExperienceSamples& samples)
{
    // 未購読では測定用配列やpackを作らず、cadence判定へも入らない。
    if (replay_fit_demand_.empty()) return std::nullopt;
    const auto request = replay_fit_demand_.MakeRequest(vars_.learn_step);
    if (!std::ranges::any_of(request, [](bool value) { return value; })) return std::nullopt;
    ANET_PROFILE_SCOPE(replay_fit);
    ReplayFitMetrics values;
    values.fill(std::numeric_limits<float>::quiet_NaN());
    // その回の出力の依存だけを合成し、PER無効時の比だけでは群を抽出しない。
    const auto [uniform, per, td_u, td_s, loss_u, loss_s, counts] = ResolveReplayFitRequirements(request, config_.use_per);
    SamplingHistoryProbeResult snapshot;
    if (counts || td_u || td_s || loss_u || loss_s) {
        snapshot = replay_buffer_->ProbeSamplingHistory(SamplingHistoryProbeRequest{
            .counts = counts,
            .unsampled_batch_size = td_u || loss_u ? std::optional<int64_t>(config_.replay_fit.probe.batch_size) : std::nullopt,
            .sampled_batch_size = td_s || loss_s ? std::optional<int64_t>(config_.replay_fit.probe.batch_size) : std::nullopt,
        }, &replay_fit_probe_random_, ComponentGammas());
        if (request[ReplayFitMetric::UnsampledCount]) values[ReplayFitMetric::UnsampledCount] = static_cast<float>(snapshot.unsampled_count);
        if (request[ReplayFitMetric::SampledCount]) values[ReplayFitMetric::SampledCount] = static_cast<float>(snapshot.sampled_count);
        if (request[ReplayFitMetric::UnsampledAgeMean]) values[ReplayFitMetric::UnsampledAgeMean] = snapshot.unsampled_age_mean;
        if (request[ReplayFitMetric::SampledAgeMean]) values[ReplayFitMetric::SampledAgeMean] = snapshot.sampled_age_mean;
    }
    if (!snapshot.unsampled && !snapshot.sampled && !per) return values;

    // 診断のforward全体でeval/NoGrad/FP32を保ち、既存captureを触らず例外時もmodeを戻す。
    torch::NoGradGuard no_grad;
    anet::TrainingModeGuard online_eval(*model_.GetOnlineNetwork(), false);
    anet::TrainingModeGuard target_eval(*model_.GetTargetNetwork(), false);
    anet::Autocast full_precision(device_, false, torch::kFloat32);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const auto u = snapshot.unsampled ? EvaluateReplayFit(*snapshot.unsampled, td_u, loss_u) : std::pair<float, float>{ nan, nan };
    const auto s = snapshot.sampled ? EvaluateReplayFit(*snapshot.sampled, td_s, loss_s) : std::pair<float, float>{ nan, nan };
    const auto ratio = [nan](float numerator, float denominator) {
        return denominator == 0.0f ? nan : numerator / denominator;
    };
    if (request[ReplayFitMetric::UnsampledTdMean]) values[ReplayFitMetric::UnsampledTdMean] = u.first;
    if (request[ReplayFitMetric::SampledTdMean]) values[ReplayFitMetric::SampledTdMean] = s.first;
    if (request[ReplayFitMetric::UnsampledTdRatio]) values[ReplayFitMetric::UnsampledTdRatio] = ratio(u.first, s.first);
    if (request[ReplayFitMetric::UnsampledLossMean]) values[ReplayFitMetric::UnsampledLossMean] = u.second;
    if (request[ReplayFitMetric::SampledLossMean]) values[ReplayFitMetric::SampledLossMean] = s.second;
    if (request[ReplayFitMetric::UnsampledLossRatio]) values[ReplayFitMetric::UnsampledLossRatio] = ratio(u.second, s.second);
    float uniform_td = nan;
    if (uniform) {
        const auto total = snapshot.unsampled_count + snapshot.sampled_count;
        if (total > 0) {
            const double sum = (snapshot.unsampled_count > 0 ? snapshot.unsampled_count * static_cast<double>(u.first) : 0.0)
                + (snapshot.sampled_count > 0 ? snapshot.sampled_count * static_cast<double>(s.first) : 0.0);
            uniform_td = static_cast<float>(sum / total);
        }
        if (request[ReplayFitMetric::UniformTdMean]) values[ReplayFitMetric::UniformTdMean] = uniform_td;
    }
    if (per) {
        const float per_td = EvaluateReplayFit(samples, true, false).first;
        if (request[ReplayFitMetric::PerTdMean]) values[ReplayFitMetric::PerTdMean] = per_td;
        if (request[ReplayFitMetric::PerSelectivity]) values[ReplayFitMetric::PerSelectivity] = ratio(per_td, uniform_td);
    }
    return values;
}

std::pair<float, float> Learner::EvaluateReplayFit(const ExperienceSamples& input, bool td, bool loss)
{
    const auto evaluation = MakeTargetEvaluation(true);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    ANET_PROFILE_SCOPE(transfer);
    const auto batch = input.To(device_);
    const auto normalized = NormalizeSampleObservations(batch);
    const int64_t size = batch.actions.size(0);
    auto current_input = normalized.obs;
    torch::Tensor taus;
    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn) {
        taus = evaluation.current_taus(size);
        current_input.Set(anet::rl::dqn::kKey_Taus, taus);
    }
    ANET_PROFILE_SCOPE_NEXT(forward);
    auto current = model_.GetOnlineNetwork()->Forward(current_input);
    if (config_.intrinsic.decomposed) DecomposedValueCombiner::Combine(current, intrinsic_beta_, config_.use_tbo, config_.tbo_epsilon);
    const bool distributional = config_.quantile_mode != ActionPolicyConfig::kQuantileModeNone;
    const auto all = current.At(distributional ? "q_dist" : "q");
    const auto selected = distributional
        ? all.gather(1, batch.actions.view({ size, 1, 1 }).expand({ size, 1, all.size(2) })).squeeze(1)
        : all.gather(1, batch.actions.unsqueeze(1)).squeeze(1);
    ANET_PROFILE_SCOPE_NEXT(error);
    ElementError error;
    if (config_.intrinsic.decomposed) {
        const CurrentForwardResult state{.input = current_input, .output = current, .all_values = all};
        const auto targets = MakeDecomposedTargets(batch, normalized.obs, normalized.next_obs, state, evaluation);
        const auto gather = [&](const std::string& key) {
            const auto& values = current.At(key);
            return distributional ? values.gather(1, batch.actions.view({size, 1, 1}).expand({size, 1, values.size(2)})).squeeze(1)
                : values.gather(1, batch.actions.unsqueeze(1)).squeeze(1);
        };
        if (loss && config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn) taus = taus.unsqueeze(2);
        const auto external = ComputeElementError(gather(distributional ? DecomposedHeadFactory::kExtrinsicQDistOutputKey : DecomposedHeadFactory::kExtrinsicQOutputKey),
            targets.primary, taus, ElementErrorRequest{.td = td, .loss = loss});
        const auto internal = ComputeElementError(gather(distributional ? DecomposedHeadFactory::kIntrinsicQDistOutputKey : DecomposedHeadFactory::kIntrinsicQOutputKey),
            targets.intrinsic, taus, ElementErrorRequest{.td = td, .loss = loss});
        if (td) error.td = external.td.abs() + config_.intrinsic.priority_lambda * internal.td.abs();
        if (loss) error.loss = external.loss + config_.intrinsic.loss_weight * internal.loss;
    } else {
        const auto [target, ignored] = MakeTarget(batch, normalized.obs, normalized.next_obs, all, evaluation);
        if (loss && config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn) taus = taus.unsqueeze(2);
        error = ComputeElementError(selected, target, taus, ElementErrorRequest{.td = td, .loss = loss});
    }
    // 必要な平均だけをまとめてCPUへ移し、IS weightや学習時の補助診断は適用しない。
    ANET_PROFILE_SCOPE_NEXT(aggregate);
    std::vector<torch::Tensor> means;
    if (td) means.push_back(error.td.abs().mean());
    if (loss) means.push_back(error.loss.mean());
    const auto cpu = torch::stack(means).to(torch::kCPU);
    const auto* data = cpu.data_ptr<float>();
    return std::pair<float, float>{ td ? data[0] : nan, loss ? data[td ? 1 : 0] : nan };
}

void Learner::CaptureNoisyDiagnostics()
{
    // 各更新で失効させ、購読 cadence が発火した項目だけを成立させる。
    noisy_.values.fill(std::numeric_limits<float>::quiet_NaN());
    const auto request = noisy_.demand.MakeRequest(vars_.learn_step);
    if (std::ranges::any_of(std::span(request).first(NoisyMetric::QAbsDiff), [](bool value) { return value; })) {
        ANET_PROFILE_SCOPE(statistics);
        const auto stats = model_.GetOnlineNetwork()->ComputeNoisyParameterStatistics(config_.final_feature_branch_key).cpu();
        const auto values = stats.accessor<float, 1>();
        for (size_t metric = 0; metric < NoisyMetric::QAbsDiff; ++metric) {
            if (request[metric]) noisy_.values[metric] = values[metric];
        }
    }
    if (!std::ranges::any_of(std::span(request).subspan(NoisyMetric::QAbsDiff), [](bool value) { return value; })) return;
    const auto network = model_.GetOnlineNetwork();
    if (network->GetNoisyParameterEntries().empty()) return;
    ANET_PROFILE_SCOPE(probe);
    if (!noisy_.sample_forward) {
        ANET_SYSTEM_ERROR("Noisy probe Resource is unavailable for subscribed metrics.");
    }
    ExperienceSamples cpu_samples;
    // 学習/PER の抽選を流用せず、一様非復元で B 状態が揃ったときだけ測る。
    if (!replay_buffer_->SampleUniqueUniform(cpu_samples, config_.noisy_probe.probe.batch_size, noisy_probe_random_, ComponentGammas())) return;
    const auto samples = cpu_samples.To(device_);
    ValidateDeviceSamples(samples, config_.noisy_probe.probe.batch_size);
    auto input = NormalizeSampleObservations(samples).obs;
    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn) {
        const auto midpoint = (torch::arange(config_.noisy_probe.iqn.num_taus,
            torch::TensorOptions().dtype(torch::kFloat32).device(device_)) + 0.5f) / config_.noisy_probe.iqn.num_taus;
        input.Set(anet::rl::dqn::kKey_Taus, midpoint.unsqueeze(0)
            .expand({config_.noisy_probe.probe.batch_size, config_.noisy_probe.iqn.num_taus}));
    }
    // 精度・mode・grad の変更はこの測定の lifetime に閉じ、通常 forward へ漏らさない。
    torch::NoGradGuard no_grad;
    anet::Autocast autocast_guard(device_, false, torch::kFloat32);
    anet::TrainingModeGuard training_guard(*network, false);
    auto mean_output = network->Forward(input);
    if (config_.intrinsic.decomposed) DecomposedValueCombiner::Combine(mean_output, intrinsic_beta_, config_.use_tbo, config_.tbo_epsilon);
    const auto mean_q = mean_output.At(HeadFactoryBase::kQOutputKey);
    const auto metrics = MeasureNoisyQProbe(mean_q, config_.noisy_probe.num_draws, [&](int) {
        const anet::nn::NetworkForwardCallInfo info({.clock = ++noisy_.forward_clock});
        auto output = noisy_.sample_forward->Forward(input, info);
        if (config_.intrinsic.decomposed) DecomposedValueCombiner::Combine(output, intrinsic_beta_, config_.use_tbo, config_.tbo_epsilon);
        return output.At(HeadFactoryBase::kQOutputKey);
    });
    for (size_t metric = NoisyMetric::QAbsDiff; metric < kNoisyMetricKeys.size(); ++metric) {
        if (request[metric]) noisy_.values[metric] = metrics[metric - NoisyMetric::QAbsDiff];
    }

}

torch::Tensor Learner::CaptureAttentionEntropy()
{
    bool due = false;
    for (auto& entry : attention_.demand) if (entry.gate.ShouldFire(vars_.learn_step)) due = true;
    if (!due) return {};
    ANET_PROFILE_FUNC();

    // 専用の非復元抽選と既存の正規化適用だけを使い、学習側の状態へ触れない。
    torch::NoGradGuard no_grad;
    ExperienceSamples cpu_samples;
    const bool sampled = replay_buffer_->SampleUniqueUniform(cpu_samples, config_.attention_entropy.probe.batch_size, attention_entropy_probe_random_, ComponentGammas());
    if (!sampled) return {};
    const auto samples = cpu_samples.To(device_);
    ValidateDeviceSamples(samples, config_.attention_entropy.probe.batch_size);
    const auto normalized = NormalizeSampleObservations(samples);
    const auto dtype = config_.use_amp_bf16 ? torch::kBFloat16 : torch::kHalf;
    anet::Autocast amp(device_, config_.use_amp, dtype);
    const auto prefix = std::string(anet::nn::kTransformerEncoderAttentionEntropyCaptureKey) + "/";
    anet::nn::NetworkForwardCapture capture({ prefix });
    model_.ForwardOnlineUpTo(normalized.obs, config_.final_feature_branch_key, &capture);

    // サイズは同期せず調べ、全moduleの層を実行順に連結して一度だけCPUへ転送する。
    std::vector<torch::Tensor> values;
    int64_t layer_count = 0;
    std::ostringstream written;
    for (const auto& [key, value] : capture.GetAll(prefix)) {
        if (value.dim() != 1 || value.scalar_type() != torch::kFloat32) {
            ANET_SYSTEM_ERROR("Attention capture key='" << key << "' shape=" << value.sizes() << " expected FP32 [layers].");
        }
        written << key << ":" << value.numel() << ", ";
        layer_count += value.numel();
        values.push_back(value);
    }
    for (const auto& entry : attention_.demand) {
        if (entry.layer >= layer_count) {
            ANET_SYSTEM_ERROR("Invalid scalar key='attention_entropy.[" << entry.layer << "]' layer_count=" << layer_count
                << " expected n in [0," << layer_count - 1 << "] written_keys_and_layers=[" << written.str() << "].");
        }
    }
    return torch::cat(values).to(torch::kCPU);
}

void Learner::CapturePlasticityProbe()
{
    ANET_PROFILE_SCOPE(plasticity_probe);

    ExperienceSamples cpu_samples;
    if (!replay_buffer_->SampleUniqueUniform(
        cpu_samples, config_.plasticity.probe.batch_size, plasticity_probe_random_, ComponentGammas())) return;
    auto samples = cpu_samples.To(device_);
    ValidateDeviceSamples(samples, config_.plasticity.probe.batch_size);
    const auto norm_samples = NormalizeSampleObservations(samples);
    const torch::ScalarType amp_dtype = config_.use_amp_bf16 ? torch::kBFloat16 : torch::kHalf;
    anet::Autocast amp_guard(device_, config_.use_amp, amp_dtype);
    const auto state = model_.ForwardOnlineUpTo(norm_samples.obs, config_.final_feature_branch_key);
    plasticity_.probe_features = state.At(config_.final_feature_branch_key).detach();
    if (plasticity_.probe_features.dim() != 2) {
        ANET_SYSTEM_ERROR("Plasticity probe feature must be rank-2 (N, D). feature_key='"
            << config_.final_feature_branch_key << "' shape=" << plasticity_.probe_features.sizes());
    }
    plasticity_.probe_metrics_cache.reset();
}

void Learner::PreparePolicyChurnRequest()
{
    // update を跨ぐ probe と Q を失効させ、現在 cadence の要求だけを再構築する。
    policy_churn_.request.fill(false);
    policy_churn_.probe_obs = {};
    policy_churn_.fixed_taus = torch::Tensor();
    policy_churn_.online_before = torch::Tensor();
    policy_churn_.online_after = torch::Tensor();
    policy_churn_.request = policy_churn_.demand.MakeRequest(vars_.learn_step);
}

void Learner::CapturePolicyChurnProbe()
{
    if (!policy_churn_.AnyQRequest()) return;
    ANET_PROFILE_SCOPE_FULL(sample, "Learner::PolicyChurn.sample");

    // 完全な一様非復元 batch を取得できる場合だけ、同一 update の比較入力を確定する。
    ExperienceSamples cpu_samples;
    if (!replay_buffer_->SampleUniqueUniform(
        cpu_samples, config_.policy_churn.probe.batch_size, policy_churn_probe_random_, ComponentGammas())) return;
    auto samples = cpu_samples.To(device_);
    ValidateDeviceSamples(samples, config_.policy_churn.probe.batch_size);
    policy_churn_.probe_obs = NormalizeSampleObservations(samples).obs;

    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn) {
        const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
        const auto midpoint = torch::arange(config_.policy_churn.iqn.num_taus, options) + 0.5f;
        policy_churn_.fixed_taus = midpoint.div(static_cast<float>(config_.policy_churn.iqn.num_taus))
            .unsqueeze(0)
            .expand({ config_.policy_churn.probe.batch_size, config_.policy_churn.iqn.num_taus });
    }
}

torch::Tensor Learner::ForwardPolicyChurnExpectedQ(bool target) const
{
    torch::NoGradGuard no_grad;
    anet::Autocast autocast_guard(device_, false, torch::kFloat32);
    auto network = target ? model_.GetTargetNetwork() : model_.GetOnlineNetwork();
    anet::TrainingModeGuard training_guard(*network, false);

    // IQN は 1 update 内で共有する固定 midpoint taus を shallow copy へ注入する。
    auto input = policy_churn_.probe_obs;
    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn) {
        input.Set(anet::rl::dqn::kKey_Taus, policy_churn_.fixed_taus);
    }
    auto output = network->Forward(input);
    if (config_.intrinsic.decomposed) DecomposedValueCombiner::Combine(output, intrinsic_beta_, config_.use_tbo, config_.tbo_epsilon);
    const auto q = output.At(HeadFactoryBase::kQOutputKey);
    if (q.scalar_type() != torch::kFloat32) {
        ANET_SYSTEM_ERROR("Policy churn forward must produce float32 Q values. dtype="
            << c10::toString(q.scalar_type()) << ".");
    }
    return q.detach();
}

void Learner::CapturePolicyChurnOnlineBefore()
{
    if (!policy_churn_.NeedsOnlineBefore() || !policy_churn_.probe_obs.Defined()) return;
    ANET_PROFILE_SCOPE_FULL(online_before, "Learner::PolicyChurn.online_before");
    policy_churn_.online_before = ForwardPolicyChurnExpectedQ(false);
}

void Learner::CapturePolicyChurnOnlineAfter()
{
    if (!policy_churn_.NeedsOnlineAfter() || !policy_churn_.probe_obs.Defined()) return;
    ANET_PROFILE_SCOPE_FULL(online_after, "Learner::PolicyChurn.online_after");
    policy_churn_.online_after = ForwardPolicyChurnExpectedQ(false);
}

void Learner::FinalizePolicyChurn(BatchUpdateResult& result)
{
    const bool any_request = std::ranges::any_of(policy_churn_.request, [](bool value) { return value; });
    if (!any_request) return;
    ANET_PROFILE_SCOPE_FULL(aggregate, "Learner::PolicyChurn.aggregate");

    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto values = torch::full(
        { static_cast<int64_t>(kPolicyChurnMetricKeys.size()) },
        std::numeric_limits<float>::quiet_NaN(), options);
    const auto set_value = [&](size_t index, const torch::Tensor& value) {
        if (policy_churn_.request[index]) values[static_cast<int64_t>(index)] = value.to(torch::kCPU);
    };

    // online update 差分は before/after の両方が成立した場合だけ集計する。
    if (policy_churn_.online_before.defined() && policy_churn_.online_after.defined()) {
        const auto delta = policy_churn_.online_after - policy_churn_.online_before;
        set_value(PolicyChurnMetric::ActionRatio, policy_churn_.online_before.argmax(1).ne(
            policy_churn_.online_after.argmax(1)).to(torch::kFloat32).mean());
        set_value(PolicyChurnMetric::QDeltaAbsMean, delta.abs().mean());
        const auto action_mean = delta.mean(0);
        set_value(PolicyChurnMetric::QDeltaSignedMax, action_mean.max());
        set_value(PolicyChurnMetric::QDeltaSignedMin, action_mean.min());
    }

    // target 指標は通常 target update 後の online/target を同一 probe 上で比較する。
    if (policy_churn_.NeedsTargetAfter() && policy_churn_.online_after.defined()) {
        ANET_PROFILE_SCOPE_FULL(target_after_profile, "Learner::PolicyChurn.target_after");
        const auto target_after = ForwardPolicyChurnExpectedQ(true);
        set_value(PolicyChurnMetric::TargetPolicyDisagreement, policy_churn_.online_after.argmax(1).ne(
            target_after.argmax(1)).to(torch::kFloat32).mean());
        set_value(PolicyChurnMetric::TargetQDeltaAbsMean, (policy_churn_.online_after - target_after).abs().mean());
    }

    if (policy_churn_.request[PolicyChurnMetric::TargetSyncAge] && model_.GetHardUpdateInterval() > 0) {
        values[PolicyChurnMetric::TargetSyncAge] = static_cast<float>(vars_.learn_step % model_.GetHardUpdateInterval());
    }
    result.policy_churn_metrics = std::move(values);
}

std::optional<torch::Tensor> Learner::GetTensor(const std::string& key, int64_t index) const
{
    if (key.find(ReplayBuffer::kKeyPrefix) == 0 && replay_buffer_ != nullptr)
        return replay_buffer_->GetTensor(key);

    return std::nullopt;
}

std::optional<std::vector<torch::Tensor>> Learner::GetTensorVector(const std::string& key, int64_t index) const
{
    if (key.find(ReplayBuffer::kKeyPrefix) == 0 && replay_buffer_ != nullptr)
        return replay_buffer_->GetTensorVector(key);

    return std::nullopt;
}

std::vector<std::string> Learner::OptimizerRoles() const
{
    if (model_.GetSkipOnlineNetwork()) return {kOptimizerRolePrimary, kOptimizerRolePrimarySigma, kOptimizerRoleSkip, kOptimizerRoleSkipSigma};
    return model_.GetOnlineNetwork()->GetNoisyParameterEntries().empty()
        ? std::vector<std::string>{kOptimizerRolePrimary} : std::vector<std::string>{kOptimizerRolePrimary, kOptimizerRolePrimarySigma};
}

std::vector<torch::optim::AdamWOptions> Learner::BuildOptimizerOptions() const
{
    // 構築と引き継ぎで同じ設定を使い、通常・sigma の group 順を保つ。
    std::vector<torch::optim::AdamWOptions> options{
        torch::optim::AdamWOptions(anet::ProfiledValue<float>(config_.learning_rate).Evaluate(0)).weight_decay(anet::ProfiledValue<float>(config_.weight_decay).Evaluate(0)).eps(config_.adam_eps)
    };
    if (model_.GetSkipOnlineNetwork()) {
        auto sigma = options.front();
        sigma.weight_decay(config_.noisy_weight_decay);
        return {options.front(), sigma, options.front(), sigma};
    }
    if (!model_.GetOnlineNetwork()->GetNoisyParameterEntries().empty()) {
        auto sigma_options = options.front();
        sigma_options.weight_decay(config_.noisy_weight_decay);
        options.push_back(sigma_options);
    }
    return options;
}

void Learner::SetupOptimizer()
{
    if (config_.intrinsic.decomposed) {
        std::unordered_set<const c10::TensorImpl*> sigma_ids;
        for (const auto& entry : model_.GetOnlineNetwork()->GetNoisyParameterEntries()) sigma_ids.insert(entry.sigma.unsafeGetTensorImpl());
        std::array<std::vector<torch::Tensor>, 4> parameters;
        for (const auto& entry : model_.GetOnlineNetwork()->named_parameters()) {
            const bool internal = entry.key().starts_with(DecomposedHeadFactory::kIntrinsicParameterPrefix);
            if (internal || entry.key().starts_with(DecomposedHeadFactory::kExtrinsicParameterPrefix)) value_head_parameters_[internal ? 1 : 0].push_back(entry.value());
            parameters[(internal ? 2 : 0) + (sigma_ids.contains(entry.value().unsafeGetTensorImpl()) ? 1 : 0)].push_back(entry.value());
        }
        if (config_.intrinsic.feature_gradient == LearnerConfig::kFeatureGradientSplit) {
            const auto options = BuildOptimizerOptions().front();
            std::vector<torch::optim::OptimizerParamGroup> groups;
            for (size_t i = 0; i < parameters.size(); ++i) {
                auto group_options = options;
                // 共有部分のdecayは外的の更新で1回だけ、内的Headは自分の更新で1回だけ掛ける。
                group_options.weight_decay(i < 2 ? 0.0 : i == 3 ? config_.noisy_weight_decay : options.weight_decay());
                groups.emplace_back(parameters[i], std::make_unique<torch::optim::AdamWOptions>(group_options));
            }
            if (config_.use_fused_optimizer) intrinsic_optimizer_ = std::make_unique<anet::FusedAdamW>(groups, options);
            else intrinsic_optimizer_ = std::make_unique<torch::optim::AdamW>(groups, options);
        }
    }
    optimizer_sigma_groups_.clear();
    for (const auto& role : OptimizerRoles()) optimizer_sigma_groups_.push_back(role.ends_with(kOptimizerSigmaRoleSuffix));
    const auto options = BuildOptimizerOptions();
    const auto& opts = options.front();
    LOG::verbose() << "Learner: lr=" << opts.lr() << " weight_decay=" << opts.weight_decay()
        << " eps=" << opts.eps() << " fused=" << config_.use_fused_optimizer;
    if (model_.GetSkipOnlineNetwork()) {
        std::vector<torch::optim::OptimizerParamGroup> groups;
        for (const auto& network : {model_.GetOnlineNetwork(), model_.GetSkipOnlineNetwork()}) {
            std::unordered_set<const c10::TensorImpl*> sigma_ids;
            for (const auto& entry : network->GetNoisyParameterEntries()) sigma_ids.insert(entry.sigma.unsafeGetTensorImpl());
            std::vector<torch::Tensor> regular, sigma;
            for (const auto& parameter : network->parameters()) {
                (sigma_ids.contains(parameter.unsafeGetTensorImpl()) ? sigma : regular).push_back(parameter);
            }
            groups.emplace_back(regular, std::make_unique<torch::optim::AdamWOptions>(options[groups.size()]));
            groups.emplace_back(sigma, std::make_unique<torch::optim::AdamWOptions>(options[groups.size()]));
        }
        if (config_.use_fused_optimizer) optimizer_ = std::make_unique<anet::FusedAdamW>(groups, opts);
        else optimizer_ = std::make_unique<torch::optim::AdamW>(groups, opts);
        if (config_.skip.feature_gradient == LearnerConfig::kFeatureGradientSplit) {
            // primaryのdecayはQ側で一度だけ掛け、sigmaを含むskip側の状態を別に持つ。
            auto skip_options = opts;
            skip_options.weight_decay(0);
            if (config_.use_fused_optimizer) skip_primary_optimizer_ = std::make_unique<anet::FusedAdamW>(model_.GetOnlineParameters(), skip_options);
            else skip_primary_optimizer_ = std::make_unique<torch::optim::AdamW>(model_.GetOnlineParameters(), skip_options);
        }
        return;
    }
    const auto entries = model_.GetOnlineNetwork()->GetNoisyParameterEntries();
    // Noisy OFF は既存の単一 group と parameter 順序をそのまま維持する。
    if (entries.empty()) {
        if (config_.use_fused_optimizer) {
            optimizer_ = std::make_unique<anet::FusedAdamW>(model_.GetOnlineParameters(), opts);
        } else {
            optimizer_ = std::make_unique<torch::optim::AdamW>(model_.GetOnlineParameters(), opts);
        }
        return;
    }
    // 計算部品が公開した対応だけから sigma を分類し、parameter 名には依存しない。
    std::unordered_set<const c10::TensorImpl*> sigma_ids;
    for (const auto& entry : entries) sigma_ids.insert(entry.sigma.unsafeGetTensorImpl());
    std::vector<torch::Tensor> ordinary_parameters;
    std::vector<torch::Tensor> sigma_parameters;
    for (const auto& parameter : model_.GetOnlineParameters()) {
        (sigma_ids.contains(parameter.unsafeGetTensorImpl()) ? sigma_parameters : ordinary_parameters).push_back(parameter);
    }
    std::vector<torch::optim::OptimizerParamGroup> groups;
    groups.emplace_back(ordinary_parameters, std::make_unique<torch::optim::AdamWOptions>(opts));
    groups.emplace_back(sigma_parameters, std::make_unique<torch::optim::AdamWOptions>(options[1]));
    // 両 optimizer は未定義勾配の parameter を skip し、mu-only では decay も適用しない。
    if (config_.use_fused_optimizer) {
        optimizer_ = std::make_unique<anet::FusedAdamW>(groups, opts);
    } else {
        optimizer_ = std::make_unique<torch::optim::AdamW>(groups, opts);
    }
}

void Learner::SetupReplayBuffer(const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec, seed_t seed)
{
    // 学習しないRunではcapacity分の確保が丸ごと無駄になるため構築しない。
    // 学習経路の無条件derefはUpdateFromBatchの早期returnで到達不能で、
    // metrics accessor側(GetScalar/GetTensor/GetTensorVector)は既にnullptrガード済み。
    if (!config_.enabled) return;

    const auto initial_priority_mode = ParseReplayInitialPriorityMode(config_);
    ValidateReplayPriorityConfig(config_, initial_priority_mode);
    anet::rl::ReplayBufferConfig rep_config{};
    rep_config.full_keys.CopyFieldFrom("capacity", config_.full_keys, "replay_capacity");
    rep_config.full_keys.CopyFieldFrom("n_step", config_.full_keys, "n_step");
    rep_config.capacity = config_.replay_capacity;
    rep_config.needs_return_recompute = {config_.gamma.type != kConstantScheduleType};
    if (config_.intrinsic.decomposed) {
        rep_config.reward_component_count = 2;
        rep_config.needs_return_recompute = {config_.gamma.type != kConstantScheduleType,
            config_.intrinsic.discount_mode == LearnerConfig::kDiscountShared ? config_.gamma.type != kConstantScheduleType : config_.intrinsic.gamma.type != kConstantScheduleType};
    }
    rep_config.n_step = config_.use_n_step ? config_.n_step : 1;
    if (model_.GetSkipOnlineNetwork()) {
        // 窓の抽出・抽選・転送は共通機構へ任せ、閉じる条件だけを選ぶ。
        const auto limit = model_.GetSkipConfig().max_length;
        if (config_.skip.target_window == LearnerConfig::kTargetWindowActionChange) {
            rep_config.window_rule = CreateActionChangeWindowRule(limit);
        } else if (config_.skip.target_window == LearnerConfig::kTargetWindowEpisodeEnd) {
            rep_config.window_rule = CreateEpisodeEndWindowRule(limit);
        } else {
            rep_config.window_rule = CreateDecisionWindowRule(limit);
        }
    }

    if (config_.use_per) {
        for (const auto* key : {"per_alpha", "per_initial_priority", "per_initial_priority_mode"}) {
            rep_config.full_keys.CopyFieldFrom(key, config_.full_keys, key);
        }
        rep_config.sampler_type = anet::rl::ReplaySamplerType::PRIORITIZED;
        rep_config.per_alpha = config_.per_alpha;
        rep_config.per_initial_priority = config_.per_initial_priority;
        rep_config.per_initial_priority_mode = initial_priority_mode;
        vars_.per_beta = per_beta_.Evaluate(0);
    } else {
        rep_config.sampler_type = ReplaySamplerType::UNIFORM;
    }
    if (stucker_config_.has_value()) {
        //rep_config.use_stacker = stucker_config_->use_stacker;
        rep_config.stack_count = stucker_config_->use_stacker ? stucker_config_->stack_count : 1;
        rep_config.stack_keys = stucker_config_->stack_keys;
    } else {
        rep_config.stack_count = 1;
    }
    
    //const ReplayBufferConfig& config, const EnvSpec& env_spec, int64_t num_envs, torch::Device storage_device, bool pin_memory, std::optional<uint64_t> seed)

    //this->replay_buffer_ = anet::rl::CreateReplayBuffer(rep_config, env_spec, batch_env_spec.num_envs, device_, false, seed);
    //this->replay_buffer_ = anet::rl::CreateReplayBuffer(rep_config, env_spec, batch_env_spec.num_envs, device_, true, seed);
    // PER無効時のmodeはPER有効化に備えた設定値でしかない(sampler=UNIFORMでReplayBufferへ渡さない)。
    // estimatorもActorヒントも作らず完全に不活性にする。
    std::unique_ptr<InitialPriorityEstimator> estimator;
    if (config_.use_per && initial_priority_mode == ReplayInitialPriorityMode::ACTOR_APPROX) {
        estimator = CreateInitialPriorityEstimator(config_);
    }
    this->replay_buffer_ = anet::rl::CreateReplayBuffer(
        rep_config, env_spec, batch_env_spec.num_envs, torch::kCPU, false, seed, std::move(estimator));
    if (config_.use_rb_prefetch) {
        this->replay_buffer_ = std::make_shared<anet::rl::PrefetchingReplayBuffer>(this->replay_buffer_, device_);
    }
}

NormalizedSampleObservations Learner::NormalizeSampleObservations(const anet::rl::ExperienceSamples& samples) const
{
    auto obs = samples.obs;
    auto next_obs = samples.next_state.next_obs;
    if (obs_norm_) {
        // 統計更新は Agent 側の収集フェーズで行うためここでは適用のみ
        obs = obs_norm_->Normalize(obs);
        next_obs = obs_norm_->Normalize(next_obs);
    }
    return { obs, next_obs };
}

std::optional<GradientNoiseMetrics> Learner::CaptureGradientNoise(const torch::Tensor& weighted_element_loss)
{
    // 未購読・未発火ではTensor、勾配、payloadを作らない。
    if (grad_noise_.demand.empty() || !std::ranges::any_of(grad_noise_.request, [](bool requested) { return requested; }))
        return std::nullopt;
    ANET_PROFILE_FUNC();
    // SetupOptimizerはAdamWまたはその派生のFusedAdamWだけを生成するため、このcastが成立する。
    const auto measurement =
        anet::MeasureGradientNoise(weighted_element_loss, grad_noise_.parameters, static_cast<const torch::optim::AdamW&>(*optimizer_),
            config_.grad_noise.num_micro_batches, config_.adam_eps, grad_noise_.subscribed);
    const auto raw_ema = grad_noise_.estimators[0].Update(measurement.raw);
    std::array<anet::GradientNoiseEstimate, 3> adam_ema;
    for (size_t group = 0; group < 3; ++group)
        adam_ema[group] = grad_noise_.estimators[group + 1].Update(measurement.adam[group]);
    const auto scale = [](const anet::GradientNoiseEstimate& estimate) {
        return estimate.signal > 0 ? estimate.variance / estimate.signal : std::numeric_limits<double>::quiet_NaN();
    };
    const auto values = MakeDiagnosticValues<GradientNoiseMetric::Count,
        GradientNoiseMetric::ScaleAdam, GradientNoiseMetric::ScaleAdamFeature, GradientNoiseMetric::ScaleAdamReadout,
        GradientNoiseMetric::RatioAdam, GradientNoiseMetric::Scale, GradientNoiseMetric::SignalAdam,
        GradientNoiseMetric::VarianceAdam, GradientNoiseMetric::SignalAdamRaw, GradientNoiseMetric::VarianceAdamRaw,
        GradientNoiseMetric::SignalRaw, GradientNoiseMetric::VarianceRaw, GradientNoiseMetric::SignalAdamFeatureRaw,
        GradientNoiseMetric::VarianceAdamFeatureRaw, GradientNoiseMetric::SignalAdamReadoutRaw, GradientNoiseMetric::VarianceAdamReadoutRaw>(scale(adam_ema[0]), scale(adam_ema[1]), scale(adam_ema[2]),
        scale(adam_ema[0]) / weighted_element_loss.size(0), scale(raw_ema), adam_ema[0].signal, adam_ema[0].variance,
        measurement.adam[0].signal, measurement.adam[0].variance, measurement.raw.signal, measurement.raw.variance,
        measurement.adam[1].signal, measurement.adam[1].variance, measurement.adam[2].signal, measurement.adam[2].variance);
    GradientNoiseMetrics result;
    result.fill(std::numeric_limits<float>::quiet_NaN());
    // EMA更新と公開を分離し、当該key自身が発火した値だけを確定する。
    for (size_t i = 0; i < values.size(); ++i) {
        if (grad_noise_.request[i] && std::isfinite(values[i]) && std::abs(values[i]) <= std::numeric_limits<float>::max()) {
            result[i] = static_cast<float>(values[i]);
        }
    }
    return result;
}

OptimizerStepResult Learner::Optimize(const torch::Tensor& primary_loss, const torch::Tensor& weighted_element_loss,
    const torch::Tensor& skip_loss, const torch::Tensor& intrinsic_loss)
{
	ANET_PROFILE_FUNC();
	
    auto parameters = model_.GetOnlineParameters();

    OptimizerStepResult result;
    result.grad_noise = CaptureGradientNoise(weighted_element_loss);
    result.grad_clip_tau = config_.use_grad_clip ? config_.grad_clip_tau : std::numeric_limits<float>::infinity();

    ANET_PROFILE_SCOPE(zero_grad);
    optimizer_->zero_grad();

    if (config_.intrinsic.decomposed) {
        ANET_PROFILE_SCOPE_NEXT(decomposed_prepare);
        const bool scaled = config_.use_amp && !config_.use_amp_bf16;
        const auto backward = [&](const torch::Tensor& loss, bool retain) {
            (scaled ? grad_scaler_.Scale(loss) : loss).backward({}, retain);
            if (scaled) grad_scaler_.Unscale_(*optimizer_);
        };
        const auto head_norm = [&](size_t index) {
            const auto grads = anet::CollectDefinedGrads(value_head_parameters_[index]);
            return grads.empty() ? torch::zeros({}, primary_loss.options().dtype(torch::kFloat32)) : anet::ForeachGradNorm(grads);
        };
        // 同じforwardから外的側の勾配を取り、split時だけ参照を保存する。
        ANET_PROFILE_SCOPE_NEXT(primary_backward);
        backward(primary_loss, intrinsic_optimizer_ != nullptr);
        ANET_PROFILE_SCOPE_NEXT(primary_grad_norm);
        const auto external_grads = anet::CollectDefinedGrads(parameters);
        result.grad_norm_tensor = anet::ForeachGradNorm(external_grads);
        result.extrinsic_head_grad_norm = head_norm(0);
        std::vector<torch::Tensor> external_gradients;
        if (intrinsic_optimizer_) {
            for (const auto& parameter : parameters) external_gradients.push_back(parameter.grad());
            ANET_PROFILE_SCOPE_NEXT(intrinsic_zero_grad);
            optimizer_->zero_grad(true);
            ANET_PROFILE_SCOPE_NEXT(intrinsic_backward);
            backward(config_.intrinsic.loss_weight * intrinsic_loss, false);
            ANET_PROFILE_SCOPE_NEXT(intrinsic_grad_norm);
            result.intrinsic_grad_norm = anet::ForeachGradNorm(anet::CollectDefinedGrads(parameters));
        }
        result.intrinsic_head_grad_norm = head_norm(1);
        const auto internal_grads = intrinsic_optimizer_ ? anet::CollectDefinedGrads(parameters) : std::vector<torch::Tensor>{};
        std::vector<torch::Tensor> internal_gradients;
        if (intrinsic_optimizer_) for (const auto& parameter : parameters) internal_gradients.push_back(parameter.grad());
        ANET_PROFILE_SCOPE_NEXT(grad_check);
#if ANET_ENABLE_TENSOR_NAN_CHECK
        // 両方の勾配を一度に検査し、片方のoptimizerさえ適用する前に止める。
        // FP16のoverflowは従来どおりGradScalerによる両更新のskipで扱う。
        if (!scaled) {
            const auto norms = intrinsic_optimizer_ ? torch::stack({result.grad_norm_tensor, result.intrinsic_grad_norm})
                : result.grad_norm_tensor;
            if (!torch::isfinite(norms).all().item<bool>())
                ANET_SYSTEM_ERROR("Nonfinite gradient norms before decomposed optimizer step: " << config_.full_keys.GetFullKey("intrinsic.feature_gradient") << "="
                    << config_.intrinsic.feature_gradient << " norms=" << norms << "; expected all finite.");
        }
#endif
        bool found_inf = false;
        if (scaled) {
            const auto norm = intrinsic_optimizer_ ? result.grad_norm_tensor + result.intrinsic_grad_norm : result.grad_norm_tensor;
            found_inf = !torch::isfinite(norm).item<bool>();
        }
        ANET_PROFILE_SCOPE_NEXT(grad_clip);
        if (config_.use_grad_clip) {
            anet::ForeachClipGradNorm_(external_grads, result.grad_norm_tensor, config_.grad_clip_tau);
            if (intrinsic_optimizer_) anet::ForeachClipGradNorm_(internal_grads, result.intrinsic_grad_norm, config_.grad_clip_tau);
        }
        ANET_PROFILE_SCOPE_NEXT(optimizer_step);
        CapturePolicyChurnOnlineBefore();
        if (intrinsic_optimizer_) for (size_t i = 0; i < parameters.size(); ++i) parameters[i].mutable_grad() = external_gradients[i];
        if (scaled) grad_scaler_.Step(*optimizer_, found_inf);
        else optimizer_->step();
        if (intrinsic_optimizer_) {
            for (size_t i = 0; i < parameters.size(); ++i) parameters[i].mutable_grad() = internal_gradients[i];
            if (scaled) grad_scaler_.Step(*intrinsic_optimizer_, found_inf);
            else intrinsic_optimizer_->step();
        }
        if (scaled) grad_scaler_.Update(found_inf);
        CapturePolicyChurnOnlineAfter();
        return result;
    }

    if (skip_primary_optimizer_) {
        // 同じforwardのgraphでQの勾配を先に取り、参照を保ったまま.gradだけを外す。
        ANET_PROFILE_SCOPE_NEXT(primary_backward);
        primary_loss.backward({}, true);
        std::vector<torch::Tensor> primary_gradients;
        for (const auto& parameter : parameters) primary_gradients.push_back(parameter.grad());
        const auto primary_grads = anet::CollectDefinedGrads(parameters);
        optimizer_->zero_grad(true);

        // パラメータを動かす前にskipの勾配も取り、Network別の3集合を確定する。
        ANET_PROFILE_SCOPE_NEXT(skip_backward);
        (config_.skip.loss_weight * skip_loss).backward();
        std::vector<torch::Tensor> skip_primary_gradients;
        for (const auto& parameter : parameters) skip_primary_gradients.push_back(parameter.grad());
        const auto skip_primary_grads = anet::CollectDefinedGrads(parameters);
        const auto skip_grads = anet::CollectDefinedGrads(model_.GetSkipOnlineNetwork()->parameters());
        ANET_PROFILE_SCOPE_NEXT(grad_norm);
        result.grad_norm_tensor = anet::ForeachGradNorm(primary_grads);
        result.skip_grad_norm = anet::ForeachGradNorm(skip_grads);
        result.skip_primary_grad_norm = skip_primary_grads.empty()
            ? torch::zeros_like(result.grad_norm_tensor) : anet::ForeachGradNorm(skip_primary_grads);
#if ANET_ENABLE_TENSOR_NAN_CHECK
        // 診断用の同期を既存経路と同じ一回にまとめ、どちらの更新よりも前に失敗させる。
        const auto norms = torch::stack({result.grad_norm_tensor, result.skip_primary_grad_norm, result.skip_grad_norm});
        if (!torch::isfinite(norms).all().item<bool>()) {
            ANET_SYSTEM_ERROR("Nonfinite gradient norms before optimizer step: [primary, skip_primary, skip]=" << norms);
        }
#endif
        ANET_PROFILE_SCOPE_NEXT(grad_clip);
        if (config_.use_grad_clip) {
            anet::ForeachClipGradNorm_(primary_grads, result.grad_norm_tensor, config_.grad_clip_tau);
            anet::ForeachClipGradNorm_(skip_primary_grads, result.skip_primary_grad_norm, config_.grad_clip_tau);
            anet::ForeachClipGradNorm_(skip_grads, result.skip_grad_norm, config_.grad_clip_tau);
        }

        // Q側とskip Networkを更新し、primaryへskip由来の更新を足す。churnは和の前後で測る。
        ANET_PROFILE_SCOPE_NEXT(optimizer_step);
        for (size_t i = 0; i < parameters.size(); ++i) parameters[i].mutable_grad() = primary_gradients[i];
        CapturePolicyChurnOnlineBefore();
        optimizer_->step();
        for (size_t i = 0; i < parameters.size(); ++i) parameters[i].mutable_grad() = skip_primary_gradients[i];
        skip_primary_optimizer_->step();
        CapturePolicyChurnOnlineAfter();
        return result;
    }

    // 既定のstop/mergeは従来の合算式と一回のbackwardを保つ。
    const auto loss = skip_loss.defined() ? primary_loss + config_.skip.loss_weight * skip_loss : primary_loss;
    if (model_.GetSkipOnlineNetwork()) {
        ANET_PROFILE_SCOPE_NEXT(backward);
        loss.backward();
        ANET_PROFILE_SCOPE_NEXT(grad_norm);
        const auto primary_grads = anet::CollectDefinedGrads(parameters);
        const auto skip_grads = anet::CollectDefinedGrads(model_.GetSkipOnlineNetwork()->parameters());
        result.grad_norm_tensor = anet::ForeachGradNorm(primary_grads);
        result.skip_grad_norm = anet::ForeachGradNorm(skip_grads);
#if ANET_ENABLE_TENSOR_NAN_CHECK
        // 両Networkの勾配の非有限値はnormへ伝播する。まとめて1回だけ同期し、適用前に止める。
        ANET_PROFILE_SCOPE_NEXT(nan_check);
        const auto gradient_norms = torch::stack({result.grad_norm_tensor, result.skip_grad_norm});
        if (!torch::isfinite(gradient_norms).all().item<bool>()) {
            ANET_SYSTEM_ERROR("Nonfinite gradient norms before optimizer step: [primary, skip]=" << gradient_norms);
        }
        ANET_PROFILE_SCOPE_NEXT(grad_clip);
#endif
        if (config_.use_grad_clip) {
            anet::ForeachClipGradNorm_(primary_grads, result.grad_norm_tensor, config_.grad_clip_tau);
            anet::ForeachClipGradNorm_(skip_grads, result.skip_grad_norm, config_.grad_clip_tau);
        }
        ANET_PROFILE_SCOPE_NEXT(optimizer_step);
        CapturePolicyChurnOnlineBefore();
        optimizer_->step();
        CapturePolicyChurnOnlineAfter();
        return result;
    }

    // ------------------------------------------------------------
    // AMP + BF16: scalerを使わず、旧QRLearnerと同じ順序で更新する
    // ------------------------------------------------------------
    if (config_.use_amp && config_.use_amp_bf16) {
        ANET_PROFILE_SCOPE_NEXT(backward);
        loss.backward();

        // 勾配ノルムとclipをforeach化し、CPU同期をoptimizer step後まで遅延する
        ANET_PROFILE_SCOPE_NEXT(grad_norm);
        auto grads = anet::CollectDefinedGrads(parameters);
        result.grad_norm = std::nullopt;
        result.grad_norm_tensor = anet::ForeachGradNorm(grads);

        if (config_.use_grad_clip) {
            ANET_PROFILE_SCOPE_NEXT(grad_clip);
            anet::ForeachClipGradNorm_(grads, result.grad_norm_tensor, config_.grad_clip_tau);

            ANET_PROFILE_SCOPE_NEXT(optimizer_step);
            CapturePolicyChurnOnlineBefore();
            optimizer_->step();
            CapturePolicyChurnOnlineAfter();
            return result;
        } else {
            ANET_PROFILE_SCOPE_NEXT(optimizer_step);
            CapturePolicyChurnOnlineBefore();
            optimizer_->step();
            CapturePolicyChurnOnlineAfter();
            return result;
        }
    }

    // ------------------------------------------------------------
    // AMP + FP16: scale -> backward -> unscale -> clip/norm -> step/update
    // ------------------------------------------------------------
    if (config_.use_amp) {
        ANET_PROFILE_SCOPE_NEXT(backward);
        grad_scaler_.Scale(loss).backward();

        ANET_PROFILE_SCOPE_NEXT(unscale);
        grad_scaler_.Unscale_(*optimizer_);

        // Unscale後の勾配に対してノルム計算とclipを行う
        auto grads = anet::CollectDefinedGrads(parameters);
        ANET_PROFILE_SCOPE_NEXT(grad_norm);
        result.grad_norm = std::nullopt;
        result.grad_norm_tensor = anet::ForeachGradNorm(grads);

        if (config_.use_grad_clip) {
            ANET_PROFILE_SCOPE_NEXT(grad_clip);
            anet::ForeachClipGradNorm_(grads, result.grad_norm_tensor, config_.grad_clip_tau);

            ANET_PROFILE_SCOPE_NEXT(scaler_step);
            CapturePolicyChurnOnlineBefore();
            grad_scaler_.Step(*optimizer_);
            CapturePolicyChurnOnlineAfter();

            ANET_PROFILE_SCOPE_NEXT(scaler_update);
            grad_scaler_.Update();
            return result;
        } else {
            ANET_PROFILE_SCOPE_NEXT(scaler_step);
            CapturePolicyChurnOnlineBefore();
            grad_scaler_.Step(*optimizer_);
            CapturePolicyChurnOnlineAfter();

            ANET_PROFILE_SCOPE_NEXT(scaler_update);
            grad_scaler_.Update();
            return result;
        }
    }

    // ------------------------------------------------------------
    // FP32: grad_norm_tensorをTensorのまま保持し、CPU同期を避けて手動clipする
    // ------------------------------------------------------------
    ANET_PROFILE_SCOPE_NEXT(backward);
    loss.backward();

#if ANET_ENABLE_TENSOR_NAN_CHECK
    ANET_PROFILE_SCOPE_NEXT(nan_check);
    for (auto& param : parameters) {
        if (param.grad().defined()) {
            ANET_ASSERT_NAN(param.grad());
        }
    }
    ANET_PROFILE_SCOPE_NEXT(grad_norm);
#else
    ANET_PROFILE_SCOPE_NEXT(grad_norm);
#endif

    auto grads = anet::CollectDefinedGrads(parameters);

    // BatchUpdateResult側でgrad_norm_tensorへフォールバックさせるため、FP32ではgrad_normを値化しない
    result.grad_norm = std::nullopt;
    result.grad_norm_tensor = anet::ForeachGradNorm(grads);

    if (config_.use_grad_clip) {
        ANET_PROFILE_SCOPE_NEXT(grad_clip);

        // clip_grad_norm_のCPU同期を避けるため、foreach演算のままスケールを適用する
        anet::ForeachClipGradNorm_(grads, result.grad_norm_tensor, config_.grad_clip_tau);

        ANET_PROFILE_SCOPE_NEXT(optimizer_step);
        CapturePolicyChurnOnlineBefore();
        optimizer_->step();
        CapturePolicyChurnOnlineAfter();
        return result;
    }

    ANET_PROFILE_SCOPE_NEXT(optimizer_step);
    CapturePolicyChurnOnlineBefore();
    optimizer_->step();
    CapturePolicyChurnOnlineAfter();
    return result;
}

// 初回 priority（固定・最大・Actor 近似）由来の行をbool [B] のフラグに。
// source は ReplayBuffer の CPU metadata なので、device へ送らずに CPU で判定する。
static torch::Tensor MakeInitialPrioritySourceFlags(const torch::Tensor& per_priority_sources)
{
    using anet::rl::ReplayPrioritySource;
    const auto sources = per_priority_sources.to(torch::kInt8).to(torch::kCPU).contiguous();
    return sources.eq(static_cast<int8_t>(ReplayPrioritySource::FIXED_INITIAL))
        | sources.eq(static_cast<int8_t>(ReplayPrioritySource::MAX_INITIAL))
        | sources.eq(static_cast<int8_t>(ReplayPrioritySource::ACTOR_INITIAL));
}

PerPriorityUpdatePending Learner::PreparePerPriorityUpdate(
    const anet::rl::ExperienceSamples& samples,
    const torch::Tensor& td_error,
    const torch::Tensor& iqn_diagnostics,
    const torch::Tensor& upper_tail_std,
    const torch::Tensor& munchausen_diagnostics)
{
    PerPriorityUpdatePending pending;

    if (!config_.use_per && !iqn_diagnostics.defined() && !munchausen_diagnostics.defined()) {
        return pending;
    }

    torch::NoGradGuard grad_guard;
    const int64_t batch_size = td_error.size(0);
    pending.enabled = true;
    pending.per_enabled = config_.use_per;
    pending.per_minibatch_size = batch_size;
    pending.iqn_diagnostics_count = iqn_diagnostics.defined() ? iqn_diagnostics.numel() : 0;
    pending.munchausen_diagnostics_count = munchausen_diagnostics.defined() ? munchausen_diagnostics.numel() : 0;
    pending.upper_tail_std_count = config_.use_per && upper_tail_std.defined() ? upper_tail_std.numel() : 0;
    if (samples.is_weights.defined()) {
        pending.per_is_weights = samples.is_weights;
    }

    // source未指定またはPER無効時も、初回行数は明示的な0として公開する。
    const auto count_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    pending.per_sample_initial_count = torch::zeros({}, count_options);
    pending.per_sample_fixed_initial_count = torch::zeros({}, count_options);
    pending.per_sample_max_initial_count = torch::zeros({}, count_options);
    pending.per_sample_actor_initial_count = torch::zeros({}, count_options);
    if (config_.use_per && samples.per_priority_sources.defined()) {
        auto sources = samples.per_priority_sources.to(torch::kInt8).to(torch::kCPU).contiguous();
        auto initial_flags = MakeInitialPrioritySourceFlags(sources);
        ANET_ASSERT_SHAPE(initial_flags, { batch_size });
        pending.per_sample_initial_count = initial_flags.to(torch::kFloat32).sum();
        pending.per_sample_fixed_initial_count = sources.eq(
            static_cast<int8_t>(ReplayPrioritySource::FIXED_INITIAL)).to(torch::kFloat32).sum();
        pending.per_sample_max_initial_count = sources.eq(
            static_cast<int8_t>(ReplayPrioritySource::MAX_INITIAL)).to(torch::kFloat32).sum();
        pending.per_sample_actor_initial_count = sources.eq(
            static_cast<int8_t>(ReplayPrioritySource::ACTOR_INITIAL)).to(torch::kFloat32).sum();
    }

    std::vector<torch::Tensor> readback_parts;
    if (config_.use_per) {
        ANET_PROFILE_SCOPE_FULL(indices_cpu, "Learner::UpdatePerPriorities.indices_cpu");

        if (!samples.replay_item_keys.device().is_cpu()) {
            ANET_SYSTEM_ERROR("Learner::PreparePerPriorityUpdate expected samples.replay_item_keys on CPU, actual="
                << samples.replay_item_keys.device());
        }
        ANET_ASSERT_DEVICE_CPU(samples.replay_item_keys);
        ANET_ASSERT_SHAPE(samples.replay_item_keys, { batch_size });
        ANET_ASSERT_DTYPE(samples.replay_item_keys, torch::kInt64);
        auto indices_tensor_cpu = samples.replay_item_keys.contiguous();
        auto indices_ptr = indices_tensor_cpu.data_ptr<int64_t>();
        pending.indices.assign(indices_ptr, indices_ptr + batch_size);

        // TD error由来のpriorityとclip件数を先頭へ置き、ReplayBuffer更新契約を維持する。
        auto priority_result = MakePerRawPriorityBatch(
            td_error.detach(), config_.per_eps, config_.use_per_prio_clip, config_.per_prio_clip_value);
        auto raw_priorities = priority_result.priorities.to(torch::kFloat32).contiguous();
        ANET_ASSERT_SHAPE(raw_priorities, { batch_size });
        ANET_ASSERT_NAN(raw_priorities);
        readback_parts.push_back(raw_priorities);
        readback_parts.push_back(priority_result.clipped_count.to(raw_priorities.options()).reshape({ 1 }));
    }
    if (iqn_diagnostics.defined()) {
        if (iqn_diagnostics.dim() != 1) {
            ANET_SYSTEM_ERROR("Learner::PreparePerPriorityUpdate expected rank-1 IQN diagnostics. actual="
                << iqn_diagnostics.sizes());
        }
        readback_parts.push_back(iqn_diagnostics.detach().to(
            torch::TensorOptions().dtype(torch::kFloat32).device(td_error.device())).contiguous());
    }
    if (munchausen_diagnostics.defined()) {
        // PERなしでも診断だけを既存の単一readbackへ載せる。
        if (munchausen_diagnostics.dim() != 1 || munchausen_diagnostics.numel() != MunchausenMetric::Count) {
            ANET_SYSTEM_ERROR("Invalid Munchausen diagnostics shape=" << munchausen_diagnostics.sizes()
                << "; expected [" << MunchausenMetric::Count << "].");
        }
        readback_parts.push_back(munchausen_diagnostics.detach().to(
            torch::TensorOptions().dtype(torch::kFloat32).device(td_error.device())).contiguous());
    }
    if (pending.upper_tail_std_count > 0) {
        if (upper_tail_std.dim() != 1 || upper_tail_std.size(0) != batch_size) {
            ANET_SYSTEM_ERROR("Learner::PreparePerPriorityUpdate expected upper_tail_std shape ["
                << batch_size << "], actual=" << upper_tail_std.sizes());
        }
        readback_parts.push_back(upper_tail_std.detach().to(
            torch::TensorOptions().dtype(torch::kFloat32).device(td_error.device())).contiguous());
    }
    auto packed_readback = torch::cat(readback_parts).contiguous();
    const int64_t priority_value_count = config_.use_per ? batch_size + 1 : 0;
    ANET_ASSERT_SHAPE(packed_readback, {
        priority_value_count + pending.iqn_diagnostics_count + pending.munchausen_diagnostics_count + pending.upper_tail_std_count });

    if (packed_readback.is_cuda() && per_priority_copy_stream_.has_value()) {
        ANET_PROFILE_SCOPE_FULL(launch, "Learner::PerPriorityD2H.launch");
        const auto producer_stream = at::cuda::getCurrentCUDAStream();
        pending.priority_readback = anet::transfer::HostReadback(
            std::move(packed_readback),
            per_priority_copy_stream_.value(),
            producer_stream,
            per_priority_event_recycler_);
    } else {
        pending.priority_readback = anet::transfer::HostReadback(
            packed_readback.device().is_cpu() ? std::move(packed_readback) : packed_readback.cpu());
    }

    return pending;
}

PerPriorityUpdateInfo Learner::ApplyPerPriorityUpdate(PerPriorityUpdatePending pending)
{
    PerPriorityUpdateInfo info;
    if (!pending.enabled) {
        return info;
    }

    const int64_t batch_size = pending.per_minibatch_size;
    info.per_minibatch_size = batch_size;
    info.per_is_weights = pending.per_is_weights;
    info.per_sample_initial_count = pending.per_sample_initial_count;
    info.per_sample_fixed_initial_count = pending.per_sample_fixed_initial_count;
    info.per_sample_max_initial_count = pending.per_sample_max_initial_count;
    info.per_sample_actor_initial_count = pending.per_sample_actor_initial_count;

    {
        ANET_PROFILE_SCOPE_FULL(wait, "Learner::PerPriorityD2H.wait");
        pending.priority_readback.Wait();
    }

    std::vector<float> priorities_vec;
    {
        ANET_PROFILE_SCOPE_FULL(vector_copy, "Learner::PerPriorityD2H.vector_copy");

        auto packed_cpu = pending.priority_readback.pinned_result.to(torch::kFloat32).contiguous();
        ANET_ASSERT_DEVICE_CPU(packed_cpu);
        const int64_t priority_value_count = pending.per_enabled ? batch_size + 1 : 0;
        ANET_ASSERT_SHAPE(packed_cpu, {
            priority_value_count + pending.iqn_diagnostics_count + pending.munchausen_diagnostics_count + pending.upper_tail_std_count });

        torch::Tensor priorities_cpu;
        if (pending.per_enabled) {
            // pack先頭のpriorityとclip件数だけをReplayBuffer更新へ渡す。
            const int64_t clipped_count = static_cast<int64_t>(packed_cpu[batch_size].item<float>());
            info.per_clipped_count = torch::tensor(
                clipped_count, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
            priorities_cpu = packed_cpu.narrow(0, 0, batch_size).contiguous();
            info.per_priorities = priorities_cpu;
            auto priorities_ptr = priorities_cpu.data_ptr<float>();
            priorities_vec.assign(priorities_ptr, priorities_ptr + batch_size);
        }
        if (pending.iqn_diagnostics_count > 0) {
            info.iqn_diagnostics = packed_cpu.narrow(
                0, priority_value_count, pending.iqn_diagnostics_count).contiguous();
        }
        if (pending.munchausen_diagnostics_count > 0) {
            info.munchausen_diagnostics = packed_cpu.narrow(
                0, priority_value_count + pending.iqn_diagnostics_count, pending.munchausen_diagnostics_count).contiguous();
        }
        if (pending.upper_tail_std_count > 0) {
            const auto upper_tail_std_cpu = packed_cpu.narrow(
                0,
                priority_value_count + pending.iqn_diagnostics_count + pending.munchausen_diagnostics_count,
                pending.upper_tail_std_count).contiguous();
            info.upper_tail_priority_spearman = CalculateSpearmanCorrelation(
                upper_tail_std_cpu, priorities_cpu);
        }
    }

    pending.priority_readback.RetireEvents(per_priority_event_recycler_);

    if (pending.per_enabled) {
        ANET_PROFILE_SCOPE_FULL(update_tree, "Learner::UpdatePerPriorities.update_tree");
        info.per_update_result = replay_buffer_->UpdatePriorities(pending.indices, priorities_vec);
    }

    return info;
}

PerPriorityUpdateInfo Learner::UpdatePerPriorities(const anet::rl::ExperienceSamples& samples, const torch::Tensor& td_error)
{
    auto pending = PreparePerPriorityUpdate(samples, td_error);
    return ApplyPerPriorityUpdate(std::move(pending));
}

torch::Tensor Learner::TransformH(const torch::Tensor& x) const
{
    // h 空間へ圧縮し、大きなBellmanターゲットの影響を抑える。
    return dqn::TransformH(x, config_.tbo_epsilon);
}

torch::Tensor Learner::TransformHInv(const torch::Tensor& x) const
{
    // h 空間の値を、報酬と足し合わせる前に実空間へ戻す。
    return config_.intrinsic.decomposed ? InverseDecomposedValue(x, config_.tbo_epsilon) : dqn::TransformHInv(x, config_.tbo_epsilon);
}

std::shared_ptr<anet::rl::dqn::BatchUpdateResult> Learner::MakeBatchUpdateResult(
    const torch::Tensor& loss,
    const torch::Tensor& td_error,
    const OptimizerStepResult& opt_result,
    const torch::Tensor& max_q,
    const torch::Tensor& q_sa,
    const PerPriorityUpdateInfo& per_info,
    const torch::Tensor& q_std,
    const torch::Tensor& q_gap,
    const torch::Tensor& q_gap_rel, const SkipLearningResult& skip) const
{
    auto result = std::make_shared<BatchUpdateResult>();
    if (skip.loss.defined()) {
        const auto skip_primary_norm = opt_result.skip_primary_grad_norm.defined()
            ? opt_result.skip_primary_grad_norm
            : torch::full({}, std::numeric_limits<float>::quiet_NaN(), skip.loss.options());
        const auto skip_primary_clip = opt_result.skip_primary_grad_norm.defined()
            ? skip_primary_norm.gt(opt_result.grad_clip_tau).to(torch::kFloat32) : skip_primary_norm;
        result->skip_metrics = StackDiagnosticValues<SkipUpdateMetric::Count,
        SkipUpdateMetric::Loss, SkipUpdateMetric::TdAbsMean, SkipUpdateMetric::SampleKMean,
        SkipUpdateMetric::KmaxMean, SkipUpdateMetric::GradNorm, SkipUpdateMetric::PrimaryGradNorm,
        SkipUpdateMetric::PrimaryClipRatio>(skip.loss.detach(), skip.td.detach().abs().mean(),
            skip.k_mean, skip.kmax_mean, opt_result.skip_grad_norm, skip_primary_norm, skip_primary_clip).detach();
    }
    result->grad_noise = opt_result.grad_noise;
    result->loss = loss.detach();
    result->td_error = td_error.detach();
    result->grad_norm = opt_result.grad_norm;
    result->grad_norm_tensor = opt_result.grad_norm_tensor;
    result->grad_clip_ratio = opt_result.grad_clip_ratio;
    result->grad_clip_tau = opt_result.grad_clip_tau;
    result->max_q = max_q;
    result->q_sa = q_sa;
    if (config_.use_tbo) {
        result->max_q_real = TransformHInv(max_q);
        result->q_sa_real = TransformHInv(q_sa);
    }
    result->q_std = q_std;
    result->q_gap = q_gap;
    result->q_gap_rel = q_gap_rel;
    result->iqn_diagnostics = per_info.iqn_diagnostics;
    result->munchausen_diagnostics = per_info.munchausen_diagnostics;
    result->upper_tail_priority_spearman = per_info.upper_tail_priority_spearman;
    if (const auto features = plasticity_.online_capture.Get(config_.final_feature_branch_key); features.defined()) {
        if (features.dim() != 2) {
            ANET_SYSTEM_ERROR("Plasticity feature must be rank-2 (N, D). " << config_.full_keys.GetFullKey("final_feature_branch_key") << "='"
                << config_.final_feature_branch_key << "' shape=" << features.sizes());
        }
        result->plasticity_features = features;
    }
    result->plasticity_request = plasticity_.online_request;
    if (const auto features = plasticity_.target_capture.Get(config_.final_feature_branch_key); features.defined()) {
        if (features.dim() != 2) {
            ANET_SYSTEM_ERROR("Target plasticity feature must be rank-2 (N, D). " << config_.full_keys.GetFullKey("final_feature_branch_key") << "='"
                << config_.final_feature_branch_key << "' shape=" << features.sizes());
        }
        // captureの鍵は一度だけ書き、joint targetの切出しは利用側の結果で行う。
        result->plasticity_target_features = plasticity_.target_capture_batch_size
            ? features.narrow(0, plasticity_.target_capture_offset, *plasticity_.target_capture_batch_size)
            : features;
    }
    result->plasticity_target_request = plasticity_.target_request;
    if (plasticity_.weight_norms.defined()) {
        result->plasticity_weight_norms = plasticity_.weight_norms;
        result->plasticity_online_network = model_.GetOnlineNetwork();
        result->plasticity_target_network = model_.GetTargetNetwork();
    }
    if (per_info.per_minibatch_size > 0) {
        result->per_minibatch_size = per_info.per_minibatch_size;
        result->per_clipped_count = per_info.per_clipped_count;
        result->per_priorities = per_info.per_priorities;
        result->per_is_weights = per_info.per_is_weights;
        result->per_sample_initial_count = per_info.per_sample_initial_count;
        result->per_sample_fixed_initial_count = per_info.per_sample_fixed_initial_count;
        result->per_sample_max_initial_count = per_info.per_sample_max_initial_count;
        result->per_sample_actor_initial_count = per_info.per_sample_actor_initial_count;
        result->per_update_result = per_info.per_update_result;
    }
    return result;
}

void Learner::UpdateTargetNetwork(step_t step)
{
    ANET_PROFILE_FUNC();
    model_.UpdateTarget(step, soft_update_tau_.Value());
}

void Learner::UpdatePerBeta(step_t step)
{
    ANET_PROFILE_FUNC();

    if (config_.use_per) vars_.per_beta = per_beta_.Evaluate(step);
}

bool Learner::CanUpdate(step_t exp_step)
{
    // warmupの境界だけを通知し、Replay件数の成立や最初の学習更新とは区別する。
    if (config_.update_warmup_steps > 0 && exp_step < config_.update_warmup_steps) {
        if (!warmup_active_) {
            LOG::info_once() << "Learner warmup started: " << config_.full_keys.GetFullKey("update_warmup_steps") << "=" << config_.update_warmup_steps;
            warmup_active_ = true;
        }
        return false;
    }
    if (warmup_active_) {
        LOG::info_once() << "Learner warmup finished: " << config_.full_keys.GetFullKey("update_warmup_steps") << "=" << config_.update_warmup_steps;
        warmup_active_ = false;
    }

    // ReplayBufferのサイズがminibatchサイズに満たない場合はスキップ（N-STEPであり得る）
    // 一度 minibatch サイズに到達した後は sampleable count が閾値未満へ戻らない前提で、
    // 高コストな Size() 再走査を避ける。
    if (!has_enough_replay_samples_) {
        if (replay_buffer_->Size() < config_.replay_batch_size)
            return false;
        has_enough_replay_samples_ = true;
    }

    return true;
}

void Learner::ValidateDeviceSamples(const anet::rl::ExperienceSamples& samples, int64_t batch_size) const
{
    ANET_ASSERT_DEVICE(samples.obs, device_);
    ANET_ASSERT_DEVICE(samples.actions, device_);
    ANET_ASSERT_DEVICE(samples.target_returns, device_);
    ANET_ASSERT_DEVICE(samples.next_state.next_obs, device_);
    ANET_ASSERT_DEVICE(samples.next_state.terminals, device_);
    ANET_ASSERT_DEVICE(samples.n_steps, device_);
    ANET_ASSERT_DEVICE_CPU(samples.replay_item_keys);
    ANET_ASSERT_SHAPE(samples.actions, { batch_size });    // 離散アクション
    if (config_.intrinsic.decomposed) ANET_ASSERT_SHAPE(samples.target_returns, {batch_size, 2});
    else ANET_ASSERT_SHAPE(samples.target_returns, { batch_size });
    ANET_ASSERT_SHAPE(samples.next_state.terminals, { batch_size });
    ANET_ASSERT_SHAPE(samples.n_steps, { batch_size });
    ANET_ASSERT_SHAPE(samples.replay_item_keys, { batch_size });
    ANET_ASSERT_DTYPE(samples.actions, torch::kInt64);    // 離散アクション
    ANET_ASSERT_DTYPE(samples.target_returns, torch::kFloat32);
    ANET_ASSERT_DTYPE(samples.next_state.terminals, torch::kBool);
    ANET_ASSERT_DTYPE(samples.n_steps, torch::kInt64);
    ANET_ASSERT_DTYPE(samples.replay_item_keys, torch::kInt64);
}

anet::rl::BatchUpdateResultList
Learner::UpdateFromBatch(const anet::rl::StepCounts& counts, const anet::rl::BatchExperience& experiences)
{
    ANET_PROFILE_FUNC();

    learning_rate_.Update(counts.exp_step);
    weight_decay_.Update(counts.exp_step);
    soft_update_tau_.Update(counts.exp_step);
    gamma_.Update(counts.exp_step);
    intrinsic_gamma_.Update(counts.exp_step);
    munchausen_alpha_.Update(counts.exp_step);
    munchausen_entropy_tau_.Update(counts.exp_step);
    munchausen_values_ = config_.munchausen.Resolve(munchausen_alpha_.Value(), munchausen_entropy_tau_.Value());
    // 定数optionsは構築・読込時に反映済み。時間変化する構成だけを書き戻す。
    if (optimizer_ && has_optimizer_schedule_) {
        auto& groups = optimizer_->param_groups();
        ANET_ASSERT(groups.size() == optimizer_sigma_groups_.size());
        for (size_t i = 0; i < groups.size(); ++i) {
            auto& options = static_cast<torch::optim::AdamWOptions&>(groups[i].options());
            options.lr(learning_rate_.Value());
            if (!optimizer_sigma_groups_[i]) options.weight_decay(weight_decay_.Value());
        }
        if (skip_primary_optimizer_) {
            static_cast<torch::optim::AdamWOptions&>(skip_primary_optimizer_->param_groups().front().options()).lr(learning_rate_.Value());
        }
        if (intrinsic_optimizer_) {
            for (size_t i = 0; i < intrinsic_optimizer_->param_groups().size(); ++i) {
                auto& options = static_cast<torch::optim::AdamWOptions&>(intrinsic_optimizer_->param_groups()[i].options());
                options.lr(learning_rate_.Value());
                if (i == 2) options.weight_decay(weight_decay_.Value());
            }
        }
    }

    // 学習停止時。空listを返すとLearnEventが発火せず(trainer側の発火条件が非空)、
    // evalの唯一の駆動源であるEpisodeEvalObserver::OnLearnが呼ばれない。
    // 全診断値が未成立(=NaN)の結果を1件だけ返し、trainer側にlearn_stepを1進めさせる。
    if (!config_.enabled) {
        return { std::make_shared<BatchUpdateResult>() };
    }

    // ReplayBuffer へ push
    replay_buffer_->Push(experiences, ComponentGammas());

    // 戻り値のUpdateResultを準備
    BatchUpdateResultList result_list;

    // Agent source の probe 値は、この LearnEvent に対応する update 群だけで共有する。
    // 群の途中で cadence が発火した値は後続 update で消さず、次の event 開始時に失効させる。
    plasticity_.probe_request = {};
    plasticity_.probe_features = torch::Tensor();
    plasticity_.probe_metrics_cache.reset();

    // Update不可なら空の結果を返す
    if (!CanUpdate(counts.exp_step)) {
        return result_list;  // 空配列
    }

    // Credit加算
    update_credit_ += earned_credit_;

    // update_credit が十分な間、学習ループを回す
    while (update_credit_ >= 1.0f) {
        const int64_t B = config_.replay_batch_size;
        //const int S = state_dim_;

        // Sample
        float current_beta = config_.use_per ? vars_.per_beta : 0.0f;
        ExperienceSamples samples;
        replay_buffer_->Sample(samples, B, current_beta, ComponentGammas());
        ExperienceSamples dev_samples = samples.To(device_);
//LOG::info() << "\n========\nUpdateFromBatch() exp_step=" << counts.exp_step << "\n========";
//LOG::verbose() << "indexes=" << anet::ToString(samples.indices);
//LOG::verbose() << "samples=" << samples.ToString();

        //ANET_LOG_DEBUG("raw_samples.obs=" << anet::ToDefString(raw_samples.obs));
        //ANET_LOG_DEBUG("raw_samples.next_states.obs=" << anet::ToDefString(raw_samples.next_states.obs));
//LOG::info() << "[ " << counts.exp_step << "] samples=\n" << samples.ToString() << "\n[" << counts.exp_step << "] ---- END OF SAMPLES ----";
//if (counts.exp_step > 120) {
//    LOG::info() << "DEUBG";
//}

        // Check shapes & dtypes
        ValidateDeviceSamples(dev_samples, B);

        // 現在の learn_step に対応する actual/probe 測定要求を準備
        if (!grad_noise_.demand.empty()) {
            grad_noise_.request = grad_noise_.demand.MakeRequest(vars_.learn_step);
        }
        PreparePolicyChurnRequest();
        CapturePolicyChurnProbe();
        plasticity_.online_request = plasticity_.online.MakeRequest(vars_.learn_step);
        plasticity_.target_request = plasticity_.target.MakeRequest(vars_.learn_step);
        const auto probe_request = plasticity_.probe.MakeRequest(vars_.learn_step);
        plasticity_.online_capture.Reset();
        plasticity_.target_capture.Reset();
        plasticity_.target_capture_batch_size.reset();
        plasticity_.target_capture_offset = 0;
        plasticity_.weight_norms = torch::Tensor();
        if (probe_request.Any()) {
            plasticity_.probe_request = probe_request;
            plasticity_.probe_features = torch::Tensor();
            plasticity_.probe_metrics_cache.reset();
            CapturePlasticityProbe();
        }

        // このupdateが適用される直前のonline parameterを、購読cadenceでのみ集計する。
        if (plasticity_.weight_norm_enabled
            && vars_.learn_step % plasticity_.weight_norm_interval == 0) {
            ANET_PROFILE_SCOPE(plasticity_weight_norm);
            const auto norms = model_.GetOnlineNetwork()->ComputeParameterNormSplit(
                config_.final_feature_branch_key);
            const auto target_invalid = model_.GetTargetNetwork()->ComputeSpectralNormValidity();
            plasticity_.weight_norms = StackDiagnosticValues<PlasticityWeightNormMetric::PackCount,
        PlasticityWeightNormMetric::Feature, PlasticityWeightNormMetric::Readout, PlasticityWeightNormMetric::FeatureEffective,
        PlasticityWeightNormMetric::ReadoutEffective, PlasticityWeightNormMetric::SpectralSigmaFeature, PlasticityWeightNormMetric::SpectralSigmaReadout,
        PlasticityWeightNormMetric::OnlineInvalidCount, PlasticityWeightNormMetric::TargetInvalidCount>(
                norms.feature,
                norms.readout,
                norms.feature_effective,
                norms.readout_effective,
                norms.sigma_feature_max,
                norms.sigma_readout_max,
                norms.invalid_count,
                target_invalid
            );
        }

        // 固有処理呼び出し
        //auto samples = raw_samples.FlattenStates();
        std::shared_ptr<anet::rl::dqn::BatchUpdateResult> result;
        CaptureNoisyDiagnostics();
        const auto replay_fit = CaptureReplayFit(dev_samples);
        const auto attention_entropy = CaptureAttentionEntropy();
        result = UpdateFromSamples(dev_samples);
        if (attention_entropy.defined()) result->attention_entropy = attention_entropy;
        if (replay_fit) result->replay_fit_metrics = replay_fit;
        if (!noisy_.demand.empty()) result->noisy_metrics = noisy_.values;

        // 更新後処理
        UpdateTargetNetwork(vars_.learn_step);
        FinalizePolicyChurn(*result);
        result_list.push_back(result);
        UpdatePerBeta(counts.exp_step);

        // カウント系更新
        vars_.learn_step++;
        update_credit_ -= 1.0f;
    }

    return result_list;
}

int64_t Learner::Save(OutputArchive& archive) const
{
    int64_t size = 0;
    if (model_.GetSkipOnlineNetwork()) {
        auto roles = OptimizerRoles();
        if (skip_primary_optimizer_) roles.push_back(kOptimizerRolePrimarySkip);
        size += archive.Write(static_cast<int64_t>(roles.size()));
        for (const auto& role : roles) size += archive.Write(role);
    }
    size += archive.WriteTorchObject(*optimizer_);
    if (skip_primary_optimizer_) size += archive.WriteTorchObject(*skip_primary_optimizer_);
    if (intrinsic_optimizer_) size += archive.WriteTorchObject(*intrinsic_optimizer_);
    return size;
}

int64_t Learner::Load(InputArchive& archive)
{
    return LoadCheckpoint(archive, model_.GetSkipOnlineNetwork() != nullptr);
}

namespace {


    // OFF checkpointの既存Torch形式を読み、stateを復元する前にprimaryの構成差を診断する。
    class PrimaryOptimizerCheckpointReader final : public torch::optim::AdamW {
    public:
        using torch::optim::AdamW::AdamW;

        void load(torch::serialize::InputArchive& archive) override
        {
            torch::serialize::InputArchive groups;
            archive.read(kArchiveParamGroups, groups);
            torch::Tensor count;
            groups.read(kArchiveParamGroupCount, count);
            const auto saved = count.item<int64_t>();
            const auto expected = static_cast<int64_t>(param_groups().size());
            if (saved != expected) {
                ANET_SYSTEM_ERROR("Invalid primary optimizer checkpoint: checkpoint_groups=" << saved
                    << " expected_primary_groups=" << expected
                    << "; primary NoisyNet configuration must match the checkpoint.");
            }
            torch::optim::AdamW::load(archive);
        }
    };

} // namespace

int64_t Learner::LoadCheckpoint(InputArchive& archive, bool has_skip)
{
    ANET_PROFILE_FUNC();
    int64_t size = 0;
    auto roles = OptimizerRoles();
    bool has_primary_skip = false;
    if (has_skip) {
        if (!model_.GetSkipOnlineNetwork()) ANET_SYSTEM_ERROR("Checkpoint contains skip optimizer groups but "
            << model_.GetSkipConfig().full_keys.GetFullKey("enabled") << "=false.");
        int64_t count = 0;
        size += archive.Read(count);
        if (count != static_cast<int64_t>(roles.size()) && count != static_cast<int64_t>(roles.size() + 1)) {
            ANET_SYSTEM_ERROR("Invalid skip optimizer role count=" << count << " expected=4 or 5.");
        }
        std::vector<std::string> saved_roles;
        std::vector<torch::optim::OptimizerParamGroup> ordered;
        for (int64_t i = 0; i < count; ++i) {
            std::string role;
            size += archive.Read(role);
            if (role == kOptimizerRolePrimarySkip) {
                if (!skip_primary_optimizer_ || has_primary_skip) {
                    ANET_SYSTEM_ERROR("Invalid optimizer role='" << role
                        << "'; expected exactly one " << kOptimizerRolePrimarySkip << " role only for "
                        << config_.full_keys.GetFullKey("skip.feature_gradient") << "="
                        << LearnerConfig::kFeatureGradientSplit << ".");
                }
                has_primary_skip = true;
                continue;
            }
            const auto it = std::find(roles.begin(), roles.end(), role);
            if (it == roles.end() || std::find(saved_roles.begin(), saved_roles.end(), role) != saved_roles.end()) {
                ANET_SYSTEM_ERROR("Invalid optimizer role='" << role << "'; expected each " << kOptimizerRolePrimary << "/" << kOptimizerRolePrimarySigma
                    << "/" << kOptimizerRoleSkip << "/" << kOptimizerRoleSkipSigma << " role once.");
            }
            saved_roles.push_back(role);
            ordered.emplace_back(optimizer_->param_groups()[std::distance(roles.begin(), it)]);
        }
        if (ordered.size() != roles.size()) {
            ANET_SYSTEM_ERROR("Incomplete skip optimizer roles: count=" << ordered.size() << " expected=" << roles.size());
        }
        // 保存順へ同じ役割のparameterを並べ、Torchのstate読込後は構築時の役割順へ戻す。
        optimizer_->param_groups() = std::move(ordered);
        size += archive.ReadTorchObject(*optimizer_);
        std::vector<torch::optim::OptimizerParamGroup> restored;
        for (const auto& role : roles) {
            const auto it = std::find(saved_roles.begin(), saved_roles.end(), role);
            restored.emplace_back(std::move(optimizer_->param_groups()[std::distance(saved_roles.begin(), it)]));
        }
        optimizer_->param_groups() = std::move(restored);
        if (has_primary_skip) size += archive.ReadTorchObject(*skip_primary_optimizer_);
    } else if (model_.GetSkipOnlineNetwork()) {
        // 無効checkpointの1/2 groupをprimaryの同じTensorへ読み、skipのstateは初期状態にする。
        std::vector<torch::optim::OptimizerParamGroup> primary_groups;
        primary_groups.emplace_back(optimizer_->param_groups()[0]);
        if (!model_.GetOnlineNetwork()->GetNoisyParameterEntries().empty()) primary_groups.emplace_back(optimizer_->param_groups()[1]);
        PrimaryOptimizerCheckpointReader primary(primary_groups, BuildOptimizerOptions().front());
        size += archive.ReadTorchObject(primary);
        optimizer_->state().clear();
        for (auto& [parameter, state] : primary.state()) optimizer_->state()[parameter] = std::move(state);
        for (size_t i = 0; i < primary.param_groups().size(); ++i) {
            static_cast<torch::optim::AdamWOptions&>(optimizer_->param_groups()[i].options())
                = static_cast<const torch::optim::AdamWOptions&>(primary.param_groups()[i].options());
        }
    } else {
        size += archive.ReadTorchObject(*optimizer_);
    }
    if (intrinsic_optimizer_) {
        std::vector<torch::optim::AdamWOptions> current;
        for (const auto& group : intrinsic_optimizer_->param_groups())
            current.push_back(static_cast<const torch::optim::AdamWOptions&>(group.options()));
        size += archive.ReadTorchObject(*intrinsic_optimizer_);
        for (size_t i = 0; i < current.size(); ++i) {
            auto& loaded = static_cast<torch::optim::AdamWOptions&>(intrinsic_optimizer_->param_groups()[i].options());
            if (loaded.lr() != current[i].lr() || loaded.weight_decay() != current[i].weight_decay() || loaded.eps() != current[i].eps()) {
                LOG::info_once(kIntrinsicOptimizerKeyPrefix + std::to_string(i))
                    << "Learner::Load: intrinsic optimizer options replaced from config. group=" << i
                    << " checkpoint_lr=" << loaded.lr() << " config_lr=" << current[i].lr()
                    << " checkpoint_weight_decay=" << loaded.weight_decay() << " config_weight_decay=" << current[i].weight_decay()
                    << " checkpoint_eps=" << loaded.eps() << " config_eps=" << current[i].eps();
            }
            loaded = current[i];
        }
    }
    if (skip_primary_optimizer_ && !has_primary_skip) {
        skip_primary_optimizer_->state().clear();
        LOG::warn_once(kOptimizerRolePrimarySkip) << "Learner::Load: checkpoint has no " << kOptimizerRolePrimarySkip
            << " optimizer role; initialized moments for " << config_.full_keys.GetFullKey("skip.feature_gradient")
            << "=" << LearnerConfig::kFeatureGradientSplit << ".";
    }
    // 診断EMAはcheckpointの一部ではなく、引き継ぎ時は未初期化へ戻す。
    if (!grad_noise_.demand.empty()) {
        for (auto& estimator : grad_noise_.estimators) estimator = anet::GradientNoiseEstimator(config_.grad_noise.ema_alpha);
    }
    // moment と step の復元後、保存時との差を通知して group の options だけを置き換える。
    auto options = BuildOptimizerOptions();
    const auto learning_rate_key = GetInitialProfiledValueKey(config_.learning_rate);
    const auto weight_decay_key = GetInitialProfiledValueKey(config_.weight_decay);
    const auto noisy_weight_decay_key = config_.full_keys.GetFullKey("noisy_weight_decay");
    const auto adam_eps_key = config_.full_keys.GetFullKey("adam_eps");
    const auto main_group_count = options.size();
    if (skip_primary_optimizer_) {
        options.push_back(torch::optim::AdamWOptions(options.front()).weight_decay(0.0));
        roles.push_back(kOptimizerRolePrimarySkip);
    }
    for (size_t i = 0; i < options.size(); ++i) {
        auto& group = i < main_group_count ? optimizer_->param_groups()[i] : skip_primary_optimizer_->param_groups().front();
        auto& loaded = static_cast<torch::optim::AdamWOptions&>(group.options());
        const auto& current = options[i];
        const std::string* group_decay_key = nullptr;
        if (roles[i] != kOptimizerRolePrimarySkip) {
            group_decay_key = roles[i].ends_with(kOptimizerSigmaRoleSuffix) ? &noisy_weight_decay_key : &weight_decay_key;
        }
        const auto report_difference = [i, &roles](const char* option, const std::string* key,
            const auto& saved_value, const auto& applied_value, bool fixed = false) {
            auto notification = LOG::info_once(roles[i] + ":" + option);
            notification << "Learner::Load: optimizer option differs from checkpoint; "
                << (fixed ? "fixed value applied" : "config applied") << ". group=" << i
                << " role=" << roles[i] << " option=" << option;
            if (key != nullptr) notification << " key=" << *key;
            notification << " checkpoint=" << saved_value << (fixed ? " fixed=" : " config=") << applied_value;
        };
        if (loaded.lr() != current.lr()) {
            report_difference("lr", &learning_rate_key, loaded.lr(), current.lr());
        }
        if (loaded.weight_decay() != current.weight_decay()) {
            report_difference("weight_decay", group_decay_key, loaded.weight_decay(), current.weight_decay(), group_decay_key == nullptr);
        }
        if (loaded.eps() != current.eps()) {
            report_difference("eps", &adam_eps_key, loaded.eps(), current.eps());
        }
        if (loaded.betas() != current.betas()) {
            // 比較は丸める前の値で行い、ログでは2値を1つの option として示す。
            std::ostringstream saved_betas, config_betas;
            saved_betas << "(" << std::get<0>(loaded.betas()) << "," << std::get<1>(loaded.betas()) << ")";
            config_betas << "(" << std::get<0>(current.betas()) << "," << std::get<1>(current.betas()) << ")";
            report_difference("betas", nullptr, saved_betas.str(), config_betas.str());
        }
        if (loaded.amsgrad() != current.amsgrad()) {
            report_difference("amsgrad", nullptr, loaded.amsgrad(), current.amsgrad());
        }
        // state はそのままに全 options を代入し、実際に適用された値を記録する。
        loaded = current;
        auto applied = LOG::info();
        applied << "Learner::Load: optimizer options applied from config. group=" << i
            << " " << learning_rate_key << "=" << loaded.lr();
        if (group_decay_key != nullptr) {
            applied << " " << *group_decay_key << "=" << loaded.weight_decay();
        } else {
            applied << " fixed weight decay " << loaded.weight_decay();
        }
        applied << " " << adam_eps_key << "=" << loaded.eps();
    }
    return size;
}

torch::Tensor Learner::MakeBellmanTarget(const ExperienceSamples& samples,
    const torch::Tensor& bootstrap_real, const torch::Tensor& bonus, std::optional<float> gamma) const
{
    torch::NoGradGuard no_grad;
    ANET_PROFILE_FUNC();
    const int64_t B = samples.target_returns.size(0);
    auto gamma_n = torch::pow(gamma.value_or(gamma_.Value()), samples.n_steps.to(torch::kFloat32));
    auto mask = 1.0f - samples.next_state.terminals.to(torch::kFloat32);
    auto returns = samples.target_returns.to(torch::kFloat32);
    ANET_ASSERT_SHAPE(returns, { B });
    ANET_ASSERT_SHAPE(mask, { B });
    // bonus は先頭 return に一度だけ加え、終端でも残す。
    if (bonus.defined()) {
        ANET_ASSERT_SHAPE(bonus, { B });
        returns = returns + bonus;
    }
    // サンプル軸を保ち、分位点の軸だけへ同じ return と割引を広げる。
    if (bootstrap_real.dim() == 2) {
        returns = returns.view({ B, 1 });
        gamma_n = gamma_n.view({ B, 1 });
        mask = mask.view({ B, 1 });
    }
    const auto raw_target = returns + gamma_n * mask * bootstrap_real;
    // 実空間で完成させた target にだけ h を一度掛ける。
    const auto target = config_.use_tbo ? TransformH(raw_target) : raw_target;
    if (bootstrap_real.dim() == 2) {
        ANET_ASSERT_SHAPE(target, { B, bootstrap_real.size(1) });
    } else {
        ANET_ASSERT_SHAPE(target, { B });
    }
    ANET_ASSERT_NAN(target);
    return target;
}

MunchausenForwardResult Learner::ForwardMunchausenValues(
    const ExperienceSamples& samples, const anet::TensorDict& obs,
    const anet::TensorDict& next_obs, const torch::Tensor& current_all,
    const TargetEvaluation& evaluation, const std::string& output_key, bool use_taus)
{
    torch::NoGradGuard no_grad;
    const int64_t B = samples.actions.size(0);
    const int64_t C = obs.Size(0);
    const bool target_mode = config_.munchausen.log_policy_mode == MunchausenValues::kLogPolicyTarget;
    // 正規化済み入力を連結し、target規則のtauをcurrent生成の後に消費する。
    ANET_PROFILE_SCOPE(forward_target);
    anet::TensorDict target_input = next_obs;
    if (target_mode) {
        for (const auto& [key, value] : obs) target_input.Set(key, torch::cat({ value, next_obs.At(key) }, 0));
    }
    if (use_taus) {
        target_input.Set(anet::rl::dqn::kKey_Taus, evaluation.target_taus(target_mode ? C + B : B));
    }
    // 結合した current/next の行を元の B 遷移へ対応付け、同じ箱・時計で共有する。
    auto row_mapping = evaluation.info.GetRowMapping();
    if (target_mode) {
        row_mapping.emplace(
            torch::arange(C, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)).repeat({1 + B / C}),
            C, target_input.device());
    }
    const auto target_info = evaluation.info.WithRowMapping(std::move(row_mapping));
    const auto target_values = (evaluation.forwards->target->Forward(target_input, target_info, evaluation.target_capture)).At(output_key);
    torch::Tensor current_values = target_mode ? target_values.narrow(0, 0, C) : current_all.detach();
    const auto next_values = target_mode ? target_values.narrow(0, C, B) : target_values;
    if (!evaluation.diagnostic && target_mode && plasticity_.target_capture.Get(config_.final_feature_branch_key).defined()) {
        const auto capture = plasticity_.target_capture.Get(config_.final_feature_branch_key);
        if (capture.dim() != 2 || capture.size(0) != C + B) {
            ANET_SYSTEM_ERROR("Munchausen target capture expected rows=" << C + B << ", actual=" << capture.sizes());
        }
        plasticity_.target_capture_batch_size = C;
        plasticity_.target_capture_offset = C;
    }
    ANET_PROFILE_SCOPE_END(forward_target);

    // fresh onlineだけevalで追加評価する。capture済みtrain特徴と既存tau集合は上書きしない。
    if (config_.munchausen.log_policy_mode == MunchausenValues::kLogPolicyOnline) {
        ANET_PROFILE_SCOPE(forward_munchausen_online);
        anet::TensorDict current_input = obs;
        if (use_taus) {
            current_input.Set(anet::rl::dqn::kKey_Taus, evaluation.current_taus(C));
        }
        anet::TrainingModeGuard eval_guard(*model_.GetOnlineNetwork(), false);
        current_values = evaluation.forwards->online->Forward(current_input,
            C == B ? evaluation.info : evaluation.info.WithRowMapping(std::nullopt)).At(output_key);
    }
    if (C != B) current_values = torch::cat({current_values, current_values});

    return MunchausenForwardResult{ .current_values = current_values, .next_values = next_values };
}

QuantileLearnerBase::QuantileLearnerBase(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
    const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, seed_t replay_seed, seed_t agent_seed,
    std::shared_ptr<ActionPolicy> target_policy, const LearnerForwardResources& forward_resources,
    std::optional<StuckerConfig> stucker_config, std::optional<anet::seed_t> target_seed,
    std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe_config)
    : Learner(config, model, vars, std::move(obs_norm), batch_env_spec, env_spec, std::move(device), replay_seed, agent_seed,
        std::move(target_policy), forward_resources, stucker_config, target_seed, std::move(noisy_probe_config))
{
}

CurrentForwardResult QuantileLearnerBase::ForwardCurrent(const ExperienceSamples& samples,
    const anet::TensorDict& obs, const anet::nn::NetworkForwardCallInfo& info)
{
    return ForwardCurrentQuantiles(samples, obs, info, {});
}

CurrentForwardResult QuantileLearnerBase::ForwardCurrentQuantiles(const ExperienceSamples& samples,
    const anet::TensorDict& input, const anet::nn::NetworkForwardCallInfo& info,
    const torch::Tensor& current_taus)
{
    ANET_PROFILE_FUNC();
    const int B = config_.replay_batch_size;
    const int A = n_actions_;
    const int64_t N = current_taus.defined() ? current_taus.size(1) : config_.num_quantiles;
    auto output = model_.ForwardOnlineWithTrain(input, *current_forward_, info, OnlineBranchCapture());
    if (config_.intrinsic.decomposed) DecomposedValueCombiner::Combine(output, intrinsic_beta_, config_.use_tbo, config_.tbo_epsilon);
    const auto current_dist_all = output.At(HeadFactoryBase::kQDistOutputKey);
    ANET_ASSERT_SHAPE(current_dist_all, { B, A, N });
    ANET_ASSERT_NAN(current_dist_all);
    const auto current_dist = GatherActionQuantiles(current_dist_all, samples.actions);

    // IQN では値の大小でなく tau の順へ並べ、上側の広がりを測る。
    auto ordered_dist = current_dist;
    if (current_taus.defined()) {
        const auto order = std::get<1>(current_taus.sort(/*dim=*/1));
        ordered_dist = current_dist.gather(1, order);
    }
    const auto upper_tail_std = CalculateQuantileTailStatistics(ordered_dist).upper_truncated_std;
    auto metrics = MakeQuantileMetrics(current_dist, output.At(HeadFactoryBase::kQOutputKey));
    return CurrentForwardResult{
        .input = input, .output = std::move(output), .all_values = current_dist_all,
        .action_values = current_dist,
        .taus = current_taus.defined() ? current_taus.unsqueeze(2) : torch::Tensor(),
        .current_mean = metrics.q_sa,
        .error_request = ElementErrorRequest{ .auxiliary = current_taus.defined() },
        .metrics = std::move(metrics), .upper_tail_std = upper_tail_std
    };
}

torch::Tensor QuantileLearnerBase::GatherActionQuantiles(const torch::Tensor& quantiles, const torch::Tensor& actions) const
{
    const int64_t B = actions.size(0);
    const int64_t N = quantiles.size(2);

    // actions(B)をquantile次元にbroadcastし、Z(s, a)だけを取り出す
    torch::Tensor idx_actions = actions.view({ B, 1, 1 }).expand({ B, 1, N });
    ANET_ASSERT_SHAPE(idx_actions, { B, 1, N });
    ANET_ASSERT_NAN(idx_actions);

    auto selected = quantiles.gather(1, idx_actions).squeeze(1);
    ANET_ASSERT_SHAPE(selected, { B, N });
    ANET_ASSERT_NAN(selected);
    return selected;
}

torch::Tensor QuantileLearnerBase::SelectTargetActions(const anet::TensorDict& next_obs,
    const TargetEvaluation& evaluation)
{
    // Double DQN有効時はPolicyNetで行動選択し、価値評価は呼び出し側でTargetNetを使う
    const auto& forward = config_.use_double_dqn ? evaluation.forwards->online : evaluation.forwards->target;
    const DiagnosticActionEvaluation action_evaluation{ config_.replay_fit.iqn.num_taus };
    auto target_action_info = target_policy_->SelectAction(next_obs, /*greedy_only=*/true, *forward, this->GetRandomGenerator(), evaluation.info,
        {}, evaluation.diagnostic ? &action_evaluation : nullptr);
    return target_action_info->GetAction(device_);
}

std::pair<torch::Tensor, torch::Tensor> QuantileLearnerBase::MakeMunchausenTarget(
    const anet::rl::ExperienceSamples& samples, const anet::TensorDict& obs,
    const anet::TensorDict& next_obs, const torch::Tensor& current_dist_all,
    const TargetEvaluation& evaluation)
{
    torch::NoGradGuard no_grad;
    const auto values = ForwardMunchausenValues(samples, obs, next_obs, current_dist_all, evaluation,
        HeadFactoryBase::kQDistOutputKey, config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn);

    // 分位点を個別に実空間へ戻し、方策と全分布混合をFP32で計算する。
    ANET_PROFILE_SCOPE(munchausen_target);
    const auto current_real = config_.use_tbo ? TransformHInv(values.current_values.to(torch::kFloat32)) : values.current_values.to(torch::kFloat32);
    const auto next_real = config_.use_tbo ? TransformHInv(values.next_values.to(torch::kFloat32)) : values.next_values.to(torch::kFloat32);
    const auto next_mean = next_real.mean(2);
    // tauの減衰StateはPolicyが所有する。IQNでは既存tau集合の経験分位を近似スコアとして使う。
    const auto risk = target_policy_->GetRiskScoreSpec();
    const auto current_score = risk ? MakeRiskBiasedScore(risk->tau, current_real, risk->use_tail_mean) : current_real.mean(2);
    const auto next_score = risk ? MakeRiskBiasedScore(risk->tau, next_real, risk->use_tail_mean) : next_mean;
    const auto terms = MakeMunchausenTargetTerms(current_score, next_score, next_mean, samples.actions, munchausen_values_, !evaluation.diagnostic);
    const auto soft_dist = (terms.next_policy.unsqueeze(2)
        * (next_real - terms.next_scaled_log_policy.unsqueeze(2))).sum(1);

    return { MakeBellmanTarget(samples, soft_dist, terms.bonus), terms.diagnostics };
}

QuantileMetrics QuantileLearnerBase::MakeQuantileMetrics(const torch::Tensor& current_dist, const torch::Tensor& q_values_mean) const
{
    QuantileMetrics metrics;
    const int64_t B = current_dist.size(0);
    auto current_dist_metrics = current_dist.detach();
    auto q_values_metrics = q_values_mean.detach();

    // 実際に選択された行動の期待値Qと、全行動中の最大期待値Qを記録する
    metrics.q_sa = current_dist_metrics.mean(1);
    metrics.max_q = std::get<0>(q_values_metrics.max(1));
    ANET_ASSERT_SHAPE(metrics.max_q, { B });

    // 分布の広がりをQ stdとして保持する。N=1は不偏分散が未定義なので明示的に0とする。
    if (current_dist_metrics.size(1) == 1) {
        metrics.q_std = torch::zeros({}, current_dist_metrics.options());
    } else {
        metrics.q_std = current_dist_metrics.std(1).mean();
    }
    ANET_ASSERT_SHAPE(metrics.q_std, {});

    if (n_actions_ >= 2) {
        // top-1/top-2の差を、絶対値と相対値の両方で監視する
        auto top2 = std::get<0>(q_values_metrics.topk(2, 1));
        auto q_best = top2.select(1, 0);
        auto q_second = top2.select(1, 1);
        auto gap_batch = q_best - q_second;
        metrics.q_gap = gap_batch.mean();

        auto denom = q_best.abs() + 1e-6f;
        metrics.q_gap_rel = (gap_batch / denom).mean();
    }

    return metrics;
}

torch::Tensor QuantileLearnerBase::ComputeQuantileHuberLoss(
    const torch::Tensor& current_dist, const torch::Tensor& target_dist, const torch::Tensor& taus, float kappa)
{
    ANET_PROFILE_FUNC();

    const auto B = current_dist.size(0);
    const auto N = current_dist.size(1);
    const auto M = target_dist.size(1);

    ANET_ASSERT_SHAPE(current_dist, { B, N });
    ANET_ASSERT_SHAPE(target_dist, { B, M });

    // current: (B, N) -> (B, N, 1), target: (B, M) -> (B, 1, M)
    // QR-DQNではN==M、IQNではtarget側のサンプル数Mが異なる可能性を許容する
    auto cur = current_dist.unsqueeze(2);
    auto tgt = target_dist.unsqueeze(1);
    ANET_ASSERT_SHAPE(cur, { B, N, 1 });
    ANET_ASSERT_SHAPE(tgt, { B, 1, M });

    // pair-wise差分 δ_ij = z'_j - z_i
    auto diff = tgt - cur;
    ANET_ASSERT_SHAPE(diff, { B, N, M });

    // Huber loss。|δ| < κ では二乗、外側では線形にする
    auto abs_diff = diff.abs();
    auto huber = torch::where(abs_diff < kappa, 0.5f * diff.pow(2), kappa * (abs_diff - 0.5f * kappa));
    ANET_ASSERT_SHAPE(huber, { B, N, M });

    // quantile regressionの重み |τ - I(δ < 0)| を掛ける
    auto indicator = (diff.detach() < 0).to(torch::kFloat);
    auto quantile_weight = torch::abs(taus - indicator);
    ANET_ASSERT_SHAPE(quantile_weight, { B, N, M });

    // target quantile方向を合計し、current quantile方向を平均してサンプル単位のlossにする
    auto loss_per_pair = quantile_weight * huber;
    auto element_wise_loss = loss_per_pair.sum(2).mean(1);
    ANET_ASSERT_SHAPE(element_wise_loss, { B });

    return element_wise_loss;
}

torch::Tensor QuantileLearnerBase::ComputeQuantileHuberLoss(
    const torch::Tensor& current_dist, const torch::Tensor& target_dist, const torch::Tensor& taus) const
{
    return ComputeQuantileHuberLoss(current_dist, target_dist, taus, config_.quantile_huber_kappa);
}

IqnLossResult QuantileLearnerBase::ComputeIqnQuantileHuberLoss(
    const torch::Tensor& current_dist, const torch::Tensor& target_dist, const torch::Tensor& taus, float kappa, bool auxiliary)
{
    ANET_PROFILE_FUNC();

    const auto B = current_dist.size(0);
    const auto N = current_dist.size(1);
    const auto M = target_dist.size(1);

    ANET_ASSERT_SHAPE(current_dist, { B, N });
    ANET_ASSERT_SHAPE(target_dist, { B, M });
    ANET_ASSERT_SHAPE(taus, { B, N, 1 });

    // Eq. 3 の全 (current tau_i, target tau'_j) ペアを作る。
    auto diff = target_dist.unsqueeze(1) - current_dist.unsqueeze(2);
    ANET_ASSERT_SHAPE(diff, { B, N, M });

    // IQN論文どおりHuberをkappaで正規化し、quantile regression重みを掛ける。
    auto abs_diff = diff.abs();
    auto huber = torch::where(
        abs_diff < kappa,
        0.5f * diff.pow(2),
        kappa * (abs_diff - 0.5f * kappa));
    auto indicator = (diff.detach() < 0).to(torch::kFloat);
    auto loss_per_pair = torch::abs(taus - indicator) * huber / kappa;
    ANET_ASSERT_SHAPE(loss_per_pair, { B, N, M });

    // current側Nをsum、target側Mをmeanし、サンプル単位のlossを返す。
    auto element_wise_loss = loss_per_pair.sum(1).mean(1);
    ANET_ASSERT_SHAPE(element_wise_loss, { B });

    if (!auxiliary) return IqnLossResult{ .element_loss = element_wise_loss };

    // 同じdeltaから相殺前の大きさと相殺率をfloat32で集約し、pairwise Tensorを再生成しない。
    const auto diff_float = diff.detach().to(torch::kFloat32);
    const auto pair_abs_td = diff_float.abs().mean({ 1, 2 });
    const auto pair_mean_td = diff_float.mean({ 1, 2 });
    const auto cancellation_ratio = (1.0f - pair_mean_td.abs() / (pair_abs_td + 1.0e-6f))
        .clamp(0.0f, 1.0f);

    return IqnLossResult{
        .element_loss = element_wise_loss,
        .pair_abs_td = pair_abs_td,
        .cancellation_ratio = cancellation_ratio,
    };
}

TDLearner::TDLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
    const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, seed_t replay_seed, seed_t agent_seed,
    std::shared_ptr<ActionPolicy> target_policy, const LearnerForwardResources& forward_resources,
    std::optional<StuckerConfig> stucker_config, std::optional<anet::seed_t> target_seed,
    std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe_config)
    : Learner(config, model, vars, std::move(obs_norm), batch_env_spec, env_spec, device, replay_seed, agent_seed,
        std::move(target_policy), forward_resources, stucker_config, target_seed, std::move(noisy_probe_config))
{
    SetupReplayBuffer(batch_env_spec, env_spec, replay_seed);
    SetupOptimizer();
}

anet::nn::NetworkForwardCallInfo Learner::MakeCurrentForwardInfo() const
{
    // 実更新の learn_step をそのまま使い、再 forward でも用途と時計を変えない。
    return anet::nn::NetworkForwardCallInfo({.clock = vars_.learn_step});
}

anet::nn::NetworkForwardCapture* Learner::OnlineBranchCapture()
{
    // 今回の更新で要求されたときだけ、Learner所有の受け皿を同期forwardへ渡す。
    return plasticity_.online_request.Any() ? &plasticity_.online_capture : nullptr;
}

anet::nn::NetworkForwardCapture* Learner::TargetBranchCapture()
{
    // targetの測定要求はonlineとは独立し、診断用forwardでは利用しない。
    return plasticity_.target_request.Any() ? &plasticity_.target_capture : nullptr;
}

TargetEvaluation Learner::MakeTargetEvaluation(bool diagnostic)
{
    // 生成関数はforward順序を動かさず、従来の各tau生成位置から呼ぶ。
    const auto make = [this, diagnostic](int64_t batch, bool current) {
        const auto& rule = current ? config_.iqn.current_taus : config_.iqn.target_taus;
        if (diagnostic) {
            const int64_t count = config_.replay_fit.iqn.num_taus;
            return MakeTauMidpointPositions(count, torch::TensorOptions().dtype(torch::kFloat32).device(device_))
                .unsqueeze(0).expand({ batch, count });
        }
        return GenerateTaus(batch, rule.num_taus, rule.sample_mode, 0.0f, 1.0f, device_, *GetRandomGenerator());
    };
    return TargetEvaluation{
        .diagnostic = diagnostic,
        .info = anet::nn::NetworkForwardCallInfo({.clock = diagnostic ? 0 : vars_.learn_step}),
        .forwards = diagnostic ? &diagnostic_forwards_ : &target_forwards_,
        .target_capture = diagnostic ? nullptr : TargetBranchCapture(),
        .current_taus = [make](int64_t batch) { return make(batch, true); },
        .target_taus = [make](int64_t batch) { return make(batch, false); },
    };
}

LearningTargets Learner::MakeLearningTargets(const ExperienceSamples& samples, const anet::TensorDict& obs,
    const anet::TensorDict& next_obs, const torch::Tensor& current_all, const TargetEvaluation& evaluation)
{
    if (!skip_current_forward_) {
        auto [target, diagnostics] = MakeTarget(samples, obs, next_obs, current_all, evaluation);
        return {.primary = target, .diagnostics = diagnostics};
    }
    ANET_PROFILE_FUNC();
    const auto batch = samples.actions.size(0);
    if (!samples.window.lengths.defined()) ANET_SYSTEM_ERROR("Skip learner requires replay window lengths.");
    // 通常の n-step と、同じサンプル起点から抽選した k-step 窓の R・k・終端を並べ、同じ target 規則を適用する。
    auto window_obs = obs_norm_ ? obs_norm_->Normalize(samples.window.next_obs) : samples.window.next_obs;
    auto combined = samples;
    combined.actions = torch::cat({samples.actions, samples.actions});
    combined.target_returns = torch::cat({samples.target_returns, samples.window.returns});
    combined.n_steps = torch::cat({samples.n_steps, samples.window.lengths});
    combined.next_state.terminals = torch::cat({samples.next_state.terminals, samples.window.terminals});
    anet::TensorDict endpoints;
    for (const auto& [key, value] : next_obs) endpoints.Set(key, torch::cat({value, window_obs.At(key)}));
    const TargetEvaluation paired{
        .diagnostic = evaluation.diagnostic,
        .info = evaluation.info.WithRowMapping(anet::nn::NetworkForwardRowMapping(
            torch::arange(batch, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)).repeat({2}), batch, endpoints.device())),
        .forwards = evaluation.forwards, .target_capture = evaluation.target_capture,
        .current_taus = evaluation.current_taus, .target_taus = evaluation.target_taus};
    // currentはBのまま、終端を2Bで渡す。Munchausen target方式はcurrentを1回だけ含む3Bになる。
    auto [target, diagnostics] = MakeTarget(combined, obs, endpoints, current_all, paired);
    if (evaluation.target_capture) {
        plasticity_.target_capture_batch_size = batch;
        plasticity_.target_capture_offset = config_.munchausen.enabled && config_.munchausen.log_policy_mode == MunchausenValues::kLogPolicyTarget ? batch : 0;
    }
    return {.primary = target.narrow(0, 0, batch), .window = target.narrow(0, batch, batch), .diagnostics = diagnostics};
}

SkipLearningResult Learner::ComputeSkipLoss(const ExperienceSamples& samples, const anet::TensorDict& input,
    const anet::TensorDict& primary, const torch::Tensor& target,
    const anet::nn::NetworkForwardCallInfo& info, const torch::Tensor& taus)
{
    if (!skip_current_forward_) return {};
    ANET_PROFILE_FUNC();
    // stop は公開特徴を detach、merge は同じ forward graph を通じて primary Body にも勾配を戻す。
    const auto skip_input = MakeSkipInput(input, primary, samples.actions, config_.skip.feature_gradient == LearnerConfig::kFeatureGradientStop);
    anet::TrainingModeGuard train(*model_.GetSkipOnlineNetwork(), true);
    const auto output = skip_current_forward_->Forward(skip_input, info);
    const auto values = ReadSkipValues(output, primary, samples.actions, model_.GetSkipConfig());
    const bool distributed = values.Contains("q_dist");
    const auto all = values.At(distributed ? "q_dist" : "q");
    // 実際に抽選した窓長 k の Q だけを窓の target に回帰する。
    const auto lengths = samples.window.lengths.to(torch::kInt64) - 1;
    const auto batch = lengths.size(0);
    const auto index = distributed ? lengths.view({batch, 1, 1}).expand({batch, 1, all.size(-1)}) : lengths.view({batch, 1});
    const auto selected = all.gather(1, index).squeeze(1);
    const auto error = ComputeElementError(selected, target, taus, {});
    auto weighted = config_.use_per ? error.loss * samples.is_weights : error.loss;
    if (config_.skip.sample_weight == LearnerConfig::kSampleWeightKmax) {
        const auto maximum = samples.window.max_lengths.to(torch::kFloat32);
        weighted = weighted * (maximum / maximum.mean());
    }
    return {.loss = weighted.mean(), .td = error.td,
        .k_mean = samples.window.lengths.to(torch::kFloat32).mean(),
        .kmax_mean = samples.window.max_lengths.to(torch::kFloat32).mean()};
}

torch::Tensor Learner::CombinePriorityErrors(const torch::Tensor& primary, const SkipLearningResult& skip) const
{
    // lambda > 0 のときだけ、primary の絶対 TD に skip の絶対 TD を足して PER へ返す。
    return skip.td.defined() && config_.skip.priority_lambda > 0.0f
        ? primary.abs() + config_.skip.priority_lambda * skip.td.abs() : primary;
}

ElementError Learner::ComputeElementError(const torch::Tensor& current,
    const torch::Tensor& target, const torch::Tensor& taus, const ElementErrorRequest& request,
    const torch::Tensor& current_mean) const
{
    ElementError result{ .td = current - target };
    if (!request.loss) return result;
    ANET_PROFILE_SCOPE(element_loss);
    // clipは損失にのみ適用し、観測TDとPERの符号付き残差を保持する。
    const auto residual = config_.use_td_clip && config_.td_clip_value > 0.0f
        ? result.td.clamp(-config_.td_clip_value, config_.td_clip_value) : result.td;
    result.loss = torch::nn::functional::smooth_l1_loss(residual, torch::zeros_like(residual),
        torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kNone));
    return result;
}

std::pair<torch::Tensor, torch::Tensor> QuantileLearnerBase::MakeTarget(
    const ExperienceSamples& samples, const anet::TensorDict& obs,
    const anet::TensorDict& next_obs, const torch::Tensor& current_all,
    const TargetEvaluation& evaluation)
{
    if (config_.munchausen.enabled) return MakeMunchausenTarget(samples, obs, next_obs, current_all, evaluation);
    torch::NoGradGuard guard;
    // hard targetは行動選択後にtarget tausを生成する。IQNのRNG消費順序を保持する。
    ANET_PROFILE_SCOPE(select_target_action);
    const auto actions = SelectTargetActions(next_obs, evaluation);
    ANET_PROFILE_SCOPE_NEXT(forward_target);
    auto input = next_obs;
    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn) input.Set(anet::rl::dqn::kKey_Taus, evaluation.target_taus(samples.actions.size(0)));
    const auto output = evaluation.forwards->target->Forward(input, evaluation.info, evaluation.target_capture);
    const auto next_dist = GatherActionQuantiles(output.At(HeadFactoryBase::kQDistOutputKey), actions);
    const auto bootstrap = config_.use_tbo ? TransformHInv(next_dist) : next_dist;
    return { MakeBellmanTarget(samples, bootstrap), {} };
}

ElementError QuantileLearnerBase::ComputeElementError(const torch::Tensor& current,
    const torch::Tensor& target, const torch::Tensor& taus, const ElementErrorRequest& request,
    const torch::Tensor& current_mean) const
{
    ElementError result;
    if (request.td) result.td = (current_mean.defined() ? current_mean : current.mean(1)) - target.mean(1).detach();
    if (!request.loss) return result;
    // TDのみの観測では全quantileペアを生成しない。IQNの補助統計は学習側だけが要求する。
    if (config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn) {
        const auto error = ComputeIqnQuantileHuberLoss(current, target, taus, config_.quantile_huber_kappa, request.auxiliary);
        result.loss = error.element_loss;
        result.pair_abs_td = error.pair_abs_td;
        result.cancellation_ratio = error.cancellation_ratio;
    } else {
        result.loss = ComputeQuantileHuberLoss(current, target, taus);
    }
    return result;
}

std::pair<torch::Tensor, torch::Tensor> Learner::MakeTarget(
    const ExperienceSamples& samples, const anet::TensorDict& obs,
    const anet::TensorDict& next_obs, const torch::Tensor& q_all,
    const TargetEvaluation& evaluation)
{
    const int64_t B = samples.actions.size(0);
    const int A = n_actions_;
    const DiagnosticActionEvaluation action_evaluation{ config_.replay_fit.iqn.num_taus };
    torch::Tensor munchausen_diagnostics;
    torch::Tensor max_next_q;
    torch::Tensor td_target;
    if (config_.munchausen.enabled) {
        torch::NoGradGuard no_grad;
        const auto values = ForwardMunchausenValues(samples, obs, next_obs, q_all, evaluation, HeadFactoryBase::kQOutputKey, false);

        ANET_PROFILE_SCOPE(munchausen_target);
        const auto current_q = config_.use_tbo ? TransformHInv(values.current_values.to(torch::kFloat32)) : values.current_values.to(torch::kFloat32);
        const auto next_q = config_.use_tbo ? TransformHInv(values.next_values.to(torch::kFloat32)) : values.next_values.to(torch::kFloat32);
        const auto terms = MakeMunchausenTargetTerms(current_q, next_q, next_q, samples.actions, munchausen_values_, !evaluation.diagnostic);
        max_next_q = (terms.next_policy * (next_q - terms.next_scaled_log_policy)).sum(1);
        munchausen_diagnostics = terms.diagnostics;
        td_target = MakeBellmanTarget(samples, max_next_q, terms.bonus);
    } else {
        torch::NoGradGuard no_grad;

        // 行動 a' を決定
        // DDQN有効の場合は PolicyNetで行動選択、DDQN無効の場合はTargetNetで行動選択
        const auto& forward = config_.use_double_dqn ? evaluation.forwards->online : evaluation.forwards->target;

        // target_policy_ に行動を選ばせる
        // greedy_only=trueはepsilonだけを無効化し、UQEのrisk選択は維持する。
        auto target_action_info = target_policy_->SelectAction(next_obs, /*greedy_only=*/true, *forward, this->GetRandomGenerator(), evaluation.info, {}, evaluation.diagnostic ? &action_evaluation : nullptr);
        torch::Tensor next_actions = target_action_info->GetAction(device_);

        // 選んだ行動の価値を TargetNet で評価する (価値評価は常に TargetNet)
        auto next_q_out = (evaluation.forwards->target->Forward(next_obs, evaluation.info, evaluation.target_capture));
        auto next_q_target = next_q_out.At("q"); // (B, A)
        ANET_ASSERT_SHAPE(next_q_target, { B, A });

        torch::Tensor next_actions_b = next_actions.view({ B, 1 }); // (B,1)
        max_next_q = next_q_target.gather(1, next_actions_b).squeeze(1);
    }
    ANET_ASSERT_SHAPE(max_next_q, { B });
    ANET_ASSERT_DTYPE(max_next_q, torch::kFloat32);

    // ------------------------------------------------------------
    // TD target & TD Error
    // ------------------------------------------------------------
    if (!config_.munchausen.enabled) {
        const auto bootstrap = config_.use_tbo ? TransformHInv(max_next_q.detach()) : max_next_q.detach();
        td_target = MakeBellmanTarget(samples, bootstrap);
    }
    return { td_target, munchausen_diagnostics };
}

std::vector<float> Learner::ComponentGammas() const
{
    if (!config_.intrinsic.decomposed) return {gamma_.Value()};
    return {gamma_.Value(), config_.intrinsic.discount_mode == LearnerConfig::kDiscountShared ? gamma_.Value() : intrinsic_gamma_.Value()};
}

anet::rl::BatchUpdateResultList Learner::UpdateFromBatch(const StepCounts& counts, const BatchExperience& experiences, float beta)
{
    // βは取り込みの入口で評価した今回の値を受け取る。診断用にも最後の値を保持する。
    intrinsic_beta_ = beta;
    return UpdateFromBatch(counts, experiences);
}

LearningTargets Learner::MakeDecomposedTargets(const ExperienceSamples& samples, const anet::TensorDict& obs,
    const anet::TensorDict& next_obs, const CurrentForwardResult& current, const TargetEvaluation& evaluation)
{
    torch::NoGradGuard guard;
    ANET_PROFILE_FUNC();
    const int64_t batch = samples.actions.size(0);
    const bool quantiles = config_.quantile_mode != ActionPolicyConfig::kQuantileModeNone;
    const bool iqn = config_.quantile_mode == ActionPolicyConfig::kQuantileModeIqn;
    const auto& external_key = quantiles ? DecomposedHeadFactory::kExtrinsicQDistOutputKey : DecomposedHeadFactory::kExtrinsicQOutputKey;
    const auto& internal_key = quantiles ? DecomposedHeadFactory::kIntrinsicQDistOutputKey : DecomposedHeadFactory::kIntrinsicQOutputKey;
    const auto real = [&](const torch::Tensor& value) {
        return config_.use_tbo ? TransformHInv(value.to(torch::kFloat32)) : value.to(torch::kFloat32);
    };
    const auto mean = [&](const torch::Tensor& value) { return quantiles ? value.mean(2) : value; };
    const auto gather = [&](const torch::Tensor& value, const torch::Tensor& action) {
        const auto index = quantiles ? action.view({batch, 1, 1}).expand({batch, 1, value.size(2)}) : action.view({batch, 1});
        return value.gather(1, index).squeeze(1);
    };
    torch::Tensor external_bootstrap, internal_bootstrap, bonus, diagnostics;
    if (config_.munchausen.enabled) {
        // 今と同じtarget/online/online_reuseのforward数・順序で、2成分を一緒に取得する。
        const bool target_mode = config_.munchausen.log_policy_mode == MunchausenValues::kLogPolicyTarget;
        auto input = next_obs;
        if (target_mode) for (const auto& [key, value] : obs) input.Set(key, torch::cat({value, next_obs.At(key)}, 0));
        if (iqn) input.Set(anet::rl::dqn::kKey_Taus, evaluation.target_taus(target_mode ? 2 * batch : batch));
        auto mapping = evaluation.info.GetRowMapping();
        if (target_mode) mapping.emplace(torch::arange(batch, torch::kInt64).repeat({2}), batch, input.device());
        const auto target = evaluation.forwards->target->Forward(input, evaluation.info.WithRowMapping(std::move(mapping)), evaluation.target_capture);
        if (!evaluation.diagnostic && target_mode) {
            const auto capture = plasticity_.target_capture.Get(config_.final_feature_branch_key);
            if (capture.defined()) {
                if (capture.dim() != 2 || capture.size(0) != 2 * batch)
                    ANET_SYSTEM_ERROR("Decomposed Munchausen target capture expected rows=" << 2 * batch
                        << ", actual=" << capture.sizes());
                plasticity_.target_capture_batch_size = batch;
                plasticity_.target_capture_offset = batch;
            }
        }
        auto external_current = current.output.At(external_key);
        auto internal_current = current.output.At(internal_key);
        auto external_next = target.At(external_key);
        auto internal_next = target.At(internal_key);
        if (target_mode) {
            external_current = external_next.narrow(0, 0, batch);
            internal_current = internal_next.narrow(0, 0, batch);
            external_next = external_next.narrow(0, batch, batch);
            internal_next = internal_next.narrow(0, batch, batch);
        } else if (config_.munchausen.log_policy_mode == MunchausenValues::kLogPolicyOnline) {
            auto current_input = obs;
            if (iqn) current_input.Set(anet::rl::dqn::kKey_Taus, evaluation.current_taus(batch));
            anet::TrainingModeGuard evaluation_guard(*model_.GetOnlineNetwork(), false);
            const auto online = evaluation.forwards->online->Forward(current_input, evaluation.info);
            external_current = online.At(external_key);
            internal_current = online.At(internal_key);
        }
        const auto external_current_real = real(external_current), internal_current_real = real(internal_current);
        const auto external_next_real = real(external_next), internal_next_real = real(internal_next);
        const auto score = [&](const torch::Tensor& value) {
            const auto risk = target_policy_->GetRiskScoreSpec();
            return quantiles && risk ? MakeRiskBiasedScore(risk->tau, value, risk->use_tail_mean) : mean(value);
        };
        const auto combined_current = external_current_real + intrinsic_beta_ * internal_current_real;
        const auto combined_next = external_next_real + intrinsic_beta_ * internal_next_real;
        const auto combined_terms = MakeMunchausenTargetTerms(score(combined_current), score(combined_next), mean(combined_next), samples.actions, munchausen_values_, !evaluation.diagnostic);
        const auto external_terms = config_.intrinsic.extrinsic_bootstrap_mode == LearnerConfig::kBootstrapExtrinsic
            ? MakeMunchausenTargetTerms(score(external_current_real), score(external_next_real), mean(external_next_real), samples.actions, munchausen_values_, !evaluation.diagnostic)
            : combined_terms;
        const auto external_policy = quantiles ? external_terms.next_policy.unsqueeze(2) : external_terms.next_policy;
        const auto internal_policy = quantiles ? combined_terms.next_policy.unsqueeze(2) : combined_terms.next_policy;
        const auto entropy = quantiles ? external_terms.next_scaled_log_policy.unsqueeze(2) : external_terms.next_scaled_log_policy;
        external_bootstrap = (external_policy * (external_next_real - entropy)).sum(1);
        internal_bootstrap = (internal_policy * internal_next_real).sum(1);
        bonus = external_terms.bonus;
        diagnostics = external_terms.diagnostics;
    } else {
        const auto& source = config_.use_double_dqn ? evaluation.forwards->online : evaluation.forwards->target;
        const DqnForwardExecutor selection(*source, intrinsic_beta_, config_.use_tbo, config_.tbo_epsilon);
        const DiagnosticActionEvaluation diagnostic{config_.replay_fit.iqn.num_taus};
        const auto action_info = target_policy_->SelectAction(next_obs, true, selection, GetRandomGenerator(), evaluation.info, {}, evaluation.diagnostic ? &diagnostic : nullptr);
        const auto action = action_info->GetAction(device_);
        auto external_action = action;
        if (config_.intrinsic.extrinsic_bootstrap_mode == LearnerConfig::kBootstrapExtrinsic) {
            auto distribution = quantiles ? action_info->forward_output.At(DecomposedHeadFactory::kExtrinsicQDistOutputKey) : torch::Tensor();
            if (iqn && action_info->risk_num_taus > 0) distribution = distribution.narrow(2, 0, action_info->risk_num_taus);
            const auto values = quantiles ? distribution.mean(2) : action_info->forward_output.At(DecomposedHeadFactory::kExtrinsicQOutputKey);
            // 同じτとforwardへ方策を当て、探索を除く。Thompsonも抽選済みτを使い直す。
            external_action = target_policy_->SelectFromValues(values, distribution, GetRandomGenerator(), iqn,
                /*greedy_only=*/true, action_info->selection_taus);
        }
        auto input = next_obs;
        if (iqn) input.Set(anet::rl::dqn::kKey_Taus, evaluation.target_taus(batch));
        const auto target = evaluation.forwards->target->Forward(input, evaluation.info, evaluation.target_capture);
        external_bootstrap = real(gather(target.At(external_key), external_action));
        internal_bootstrap = real(gather(target.At(internal_key), action));
    }
    auto component = samples;
    component.target_returns = samples.target_returns.select(1, 0);
    auto external_target = MakeBellmanTarget(component, external_bootstrap, bonus);
    component.target_returns = samples.target_returns.select(1, 1);
    if (config_.intrinsic.terminal_mode == LearnerConfig::kTerminalNonEpisodic)
        component.next_state.terminals = samples.next_state.terminals & samples.next_state.truncated;
    const auto internal_target = MakeBellmanTarget(component, internal_bootstrap, {}, ComponentGammas()[1]);
    return {.primary = std::move(external_target), .diagnostics = std::move(diagnostics), .intrinsic = internal_target};
}

std::shared_ptr<anet::rl::dqn::BatchUpdateResult>
Learner::UpdateFromSamples(const anet::rl::ExperienceSamples& samples)
{
    ANET_PROFILE_FUNC();
    const int B = config_.replay_batch_size;
    const torch::ScalarType amp_dtype = config_.use_amp_bf16 ? torch::kBFloat16 : torch::kHalf;

    // ReplayBuffer の batch 契約は、直接この更新を呼ぶ経路でも検査する。
    ANET_ASSERT_SHAPE(samples.actions, { B });
    if (config_.intrinsic.decomposed) ANET_ASSERT_SHAPE(samples.target_returns, {B, 2});
    else ANET_ASSERT_SHAPE(samples.target_returns, { B });
    ANET_ASSERT_SHAPE(samples.next_state.terminals, { B });
    //ANET_LOG_DEBUG("obs=" << samples.obs.ToString());

    ANET_PROFILE_SCOPE(normalize);
    auto norm_samples = NormalizeSampleObservations(samples);
    const auto& obs = norm_samples.obs;
    const auto& next_obs = norm_samples.next_obs;
    ANET_ASSERT_NAN(obs);
    ANET_ASSERT_NAN(next_obs);
    ANET_PROFILE_SCOPE_END(normalize);

    torch::Tensor loss;
    torch::Tensor weighted_element_loss;
    torch::Tensor td_error;
    torch::Tensor iqn_diagnostics;
    torch::Tensor upper_tail_std;
    torch::Tensor munchausen_diagnostics;
    QuantileMetrics metrics;
    SkipLearningResult skip_result;
    torch::Tensor extrinsic_loss, intrinsic_loss, head_values;
    {
        // current の forward から PER の重み付けと平均までを同じ AMP 範囲に置く。
        anet::Autocast amp_guard(device_, config_.use_amp, amp_dtype);
        ANET_PROFILE_SCOPE(forward_current);
        const auto forward_info = MakeCurrentForwardInfo();
        auto current = ForwardCurrent(samples, obs, forward_info);
        metrics = current.metrics;
        upper_tail_std = current.upper_tail_std;
        if (config_.intrinsic.decomposed) {
            const auto gather = [&](const torch::Tensor& values) {
                const auto indices = values.dim() == 3
                    ? samples.actions.view({B, 1, 1}).expand({B, 1, values.size(2)}) : samples.actions.view({B, 1});
                return values.gather(1, indices).squeeze(1);
            };
            const bool distribution = config_.quantile_mode != ActionPolicyConfig::kQuantileModeNone;
            current.action_values = gather(current.output.At(distribution ? DecomposedHeadFactory::kExtrinsicQDistOutputKey : DecomposedHeadFactory::kExtrinsicQOutputKey));
            current.intrinsic_action_values = gather(current.output.At(distribution ? DecomposedHeadFactory::kIntrinsicQDistOutputKey : DecomposedHeadFactory::kIntrinsicQOutputKey));
            current.current_mean = distribution ? current.action_values.mean(1) : current.action_values;
            const auto internal_mean = distribution ? current.intrinsic_action_values.mean(1) : current.intrinsic_action_values;
            // 表示するHead平均は出力空間のまま、割合だけは分位点ごとに逆変換した実空間で作る。
            const auto external_real = config_.use_tbo ? TransformHInv(current.action_values.detach()) : current.action_values.detach();
            const auto internal_real = config_.use_tbo ? TransformHInv(current.intrinsic_action_values.detach()) : current.intrinsic_action_values.detach();
            const auto external_real_mean = distribution ? external_real.mean(1) : external_real;
            const auto internal_real_mean = distribution ? internal_real.mean(1) : internal_real;
            const auto weighted_internal = intrinsic_beta_ * internal_real_mean;
            const auto denominator = external_real_mean.abs() + weighted_internal.abs();
            head_values = torch::stack({current.current_mean.detach().mean(), internal_mean.detach().mean(),
                torch::where(denominator > 0, weighted_internal.abs() / denominator.clamp_min(std::numeric_limits<float>::min()), torch::zeros_like(denominator)).mean()});
        }
        ANET_PROFILE_SCOPE_END(forward_current);

        const auto evaluation = MakeTargetEvaluation(false);
        auto targets = config_.intrinsic.decomposed ? MakeDecomposedTargets(samples, obs, next_obs, current, evaluation)
            : MakeLearningTargets(samples, obs, next_obs, current.all_values, evaluation);
        munchausen_diagnostics = targets.diagnostics;
        // primary と同じ batch から窓の skip loss を計算し、後段で一度だけ最適化する。
        skip_result = ComputeSkipLoss(samples, current.input, current.output, targets.window, forward_info, current.taus);

        ANET_PROFILE_SCOPE(loss);
        const auto error = ComputeElementError(current.action_values, targets.primary,
            current.taus, current.error_request, current.current_mean);
        td_error = error.td;
        ANET_ASSERT_SHAPE(td_error, { B });
        ANET_ASSERT_DTYPE(td_error, torch::kFloat32);
        ANET_ASSERT_NAN(td_error);
        auto element_loss = error.loss;
        if (config_.intrinsic.decomposed) {
            const auto internal = ComputeElementError(current.intrinsic_action_values, targets.intrinsic, current.taus, current.error_request);
            const auto weights = config_.use_per ? samples.is_weights : torch::ones_like(element_loss);
            extrinsic_loss = (element_loss * weights).mean();
            intrinsic_loss = (internal.loss * weights).mean();
            element_loss = element_loss + config_.intrinsic.loss_weight * internal.loss;
            td_error = td_error.abs() + config_.intrinsic.priority_lambda * internal.td.abs();
        }
        ANET_LOG_DEBUG("element_loss=" << anet::ToString(element_loss));
        ANET_ASSERT_SHAPE(element_loss, { B });
        ANET_ASSERT_NAN(element_loss);

        iqn_diagnostics = MakeUpdateDiagnostics(samples, current, targets.primary, error);
        // PER 無効時は重みの Tensor を確保せず、要素 loss をそのまま平均する。
        if (config_.use_per) {
            const auto& weights = samples.is_weights;
            ANET_LOG_DEBUG("weights=" << anet::ToString(weights));
            ANET_ASSERT_SHAPE(weights, { B });
            ANET_ASSERT_NAN(weights);
            weighted_element_loss = element_loss * weights;
        } else {
            weighted_element_loss = element_loss;
        }
        loss = weighted_element_loss.mean();
        ANET_ASSERT_SHAPE(loss, {});
        ANET_ASSERT_NAN(loss);
    }

    // backward 前の誤差で優先度を準備し、optimizer の完了後に反映する。
    ANET_PROFILE_SCOPE(prepare_per);
    auto per_pending = PreparePerPriorityUpdate(samples, CombinePriorityErrors(td_error, skip_result),
        iqn_diagnostics, upper_tail_std, munchausen_diagnostics);
    ANET_PROFILE_SCOPE_NEXT(optimize);
    auto opt_result = Optimize(intrinsic_optimizer_ ? extrinsic_loss : loss, weighted_element_loss, skip_result.loss, intrinsic_loss);
    if (config_.intrinsic.decomposed) {
        // 更新後の診断の読出しへ、device上の値をひとまとめで渡す。
        const auto nan = torch::full({}, std::numeric_limits<float>::quiet_NaN(), head_values.options());
        decomposition_metrics_ = torch::stack({extrinsic_loss.detach(), intrinsic_loss.detach(), head_values[0], head_values[1],
            opt_result.extrinsic_head_grad_norm, opt_result.intrinsic_head_grad_norm,
            intrinsic_optimizer_ ? opt_result.grad_norm_tensor : nan,
            intrinsic_optimizer_ ? opt_result.intrinsic_grad_norm : nan, head_values[2]});
        decomposition_metrics_cpu_ = torch::Tensor();
    }
    ANET_PROFILE_SCOPE_NEXT(update_per);
    auto per_result = ApplyPerPriorityUpdate(std::move(per_pending));
    ANET_PROFILE_SCOPE_NEXT(make_result);
    return MakeBatchUpdateResult(loss, td_error, opt_result, metrics.max_q, metrics.q_sa, per_result,
        metrics.q_std, metrics.q_gap, metrics.q_gap_rel, skip_result);
}

CurrentForwardResult Learner::ForwardCurrent(const ExperienceSamples& samples,
    const anet::TensorDict& obs, const anet::nn::NetworkForwardCallInfo& info)
{
    ANET_PROFILE_FUNC();
    const int B = config_.replay_batch_size;
    const int A = n_actions_;
    auto output = model_.ForwardOnlineWithTrain(obs, *current_forward_, info, OnlineBranchCapture());
    if (config_.intrinsic.decomposed) DecomposedValueCombiner::Combine(output, intrinsic_beta_, config_.use_tbo, config_.tbo_epsilon);
    const auto q_all = output.At(HeadFactoryBase::kQOutputKey);
    ANET_ASSERT_SHAPE(q_all, { B, A });
    const auto idx_actions = samples.actions.view({ B, 1 });
    ANET_ASSERT_SHAPE(idx_actions, { B, 1 });
    ANET_ASSERT_DTYPE(idx_actions, torch::kInt64);
    const auto q_sa = q_all.gather(1, idx_actions).squeeze(1);
    ANET_ASSERT_SHAPE(q_sa, { B });
    ANET_ASSERT_DTYPE(q_sa, torch::kFloat32);

    // 最大 Q と top-1/top-2 の差は、学習グラフから切り離して集約する。
    const auto q_metrics = q_all.detach();
    QuantileMetrics metrics{ .q_sa = q_sa.detach(), .max_q = std::get<0>(q_metrics.max(1)) };
    if (n_actions_ >= 2) {
        const auto top2 = std::get<0>(q_metrics.topk(2, 1));
        const auto q_best = top2.select(1, 0);
        const auto q_second = top2.select(1, 1);
        const auto gap_batch = q_best - q_second;
        metrics.q_gap = gap_batch.mean();
        // Q は負にもなるので絶対値を使い、初期値 0 での除算を避ける。
        const auto denom = q_best.abs() + 1e-6f;
        metrics.q_gap_rel = (gap_batch / denom).mean();
    }
    return CurrentForwardResult{
        .input = obs, .output = std::move(output), .all_values = q_all, .action_values = q_sa,
        .metrics = std::move(metrics)
    };
}

torch::Tensor Learner::MakeUpdateDiagnostics(const ExperienceSamples&,
    const CurrentForwardResult&, const torch::Tensor&, const ElementError&) const
{
    // 診断を持たない分布は未定義のまま返し、readback の有無を保つ。
    return {};
}

QRLearner::QRLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
    const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, seed_t replay_seed, seed_t agent_seed,
    std::shared_ptr<ActionPolicy> target_policy, const LearnerForwardResources& forward_resources,
    std::optional<StuckerConfig> stucker_config, std::optional<anet::seed_t> target_seed,
    std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe_config)
    : QuantileLearnerBase(config, model, vars, std::move(obs_norm), batch_env_spec, env_spec, std::move(device), replay_seed, agent_seed,
        std::move(target_policy), forward_resources, stucker_config, target_seed, std::move(noisy_probe_config))
{
    SetupReplayBuffer(batch_env_spec, env_spec, replay_seed);
    SetupOptimizer();

    // tau_iの事前計算
    const int N = config_.num_quantiles;
    tau_i_ = torch::arange(0.5f / N, 1.0f, 1.0f / N, device).view({ 1, N, 1 });
    ANET_ASSERT_SHAPE(tau_i_, { 1, N, 1 });
}

ElementError QRLearner::ComputeElementError(const torch::Tensor& current,
    const torch::Tensor& target, const torch::Tensor&, const ElementErrorRequest& request,
    const torch::Tensor& current_mean) const
{
    // QRの固定分位点はLearnerが所有し、学習・診断の両経路で共有する。
    return QuantileLearnerBase::ComputeElementError(current, target, tau_i_, request, current_mean);
}

IQNLearner::IQNLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
    const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, seed_t replay_seed, seed_t agent_seed,
    std::shared_ptr<ActionPolicy> target_policy, const LearnerForwardResources& forward_resources,
    std::optional<StuckerConfig> stucker_config, std::optional<anet::seed_t> target_seed,
    std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe_config)
    : QuantileLearnerBase(config, model, vars, std::move(obs_norm), batch_env_spec, env_spec, std::move(device), replay_seed, agent_seed,
        std::move(target_policy), forward_resources, stucker_config, target_seed, std::move(noisy_probe_config))
{
    SetupReplayBuffer(batch_env_spec, env_spec, replay_seed);
    SetupOptimizer();
}

CurrentForwardResult IQNLearner::ForwardCurrent(const ExperienceSamples& samples,
    const anet::TensorDict& obs, const anet::nn::NetworkForwardCallInfo& info)
{
    const int B = config_.replay_batch_size;
    const int N = config_.iqn.current_taus.num_taus;
    ANET_PROFILE_SCOPE(generate_current_taus);
    // current の tau は forward より前に消費し、ReplayBuffer の観測は変えない。
    const auto current_taus = GenerateTaus(B, N, config_.iqn.current_taus.sample_mode, 0.0f, 1.0f, device_, *GetRandomGenerator());
    ANET_ASSERT_SHAPE(current_taus, { B, N });
    anet::TensorDict input = obs;
    input.Set(anet::rl::dqn::kKey_Taus, current_taus);
    ANET_PROFILE_SCOPE_END(generate_current_taus);
    return ForwardCurrentQuantiles(samples, input, info, current_taus);
}

torch::Tensor IQNLearner::MakeUpdateDiagnostics(const ExperienceSamples& samples,
    const CurrentForwardResult& current, const torch::Tensor& target_dist, const ElementError& error) const
{
    ANET_PROFILE_SCOPE(iqn_diagnostics);
    torch::NoGradGuard grad_guard;
    const int B = config_.replay_batch_size;
    const int N = config_.iqn.current_taus.num_taus;
    const int M = config_.iqn.target_taus.num_taus;
    const auto& current_dist = current.action_values;
    // current・target分布のMonte Carlo標準誤差をfloat32で求め、priorityを合成標準誤差で正規化する。
    const auto float_options = current_dist.options().dtype(torch::kFloat32);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    torch::Tensor current_scale = torch::full({ B }, nan, float_options);
    torch::Tensor target_scale = torch::full({ B }, nan, float_options);
    if (N >= 2) {
        current_scale = current_dist.detach().to(torch::kFloat32).std(/*dim=*/1, /*unbiased=*/true)
            / std::sqrt(static_cast<float>(N));
    }
    if (M >= 2) {
        target_scale = target_dist.detach().to(torch::kFloat32).std(/*dim=*/1, /*unbiased=*/true)
            / std::sqrt(static_cast<float>(M));
    }
    const auto priority_ratio = error.td.detach().to(torch::kFloat32).abs()
        / ((current_scale.square() + target_scale.square()).sqrt() + 1.0e-6f);

    // 初回priority由来の行だけを抽出し、対象行がない場合は初回系診断をNaNにする。
    auto initial_mask = torch::zeros(
        { B }, torch::TensorOptions().dtype(torch::kBool).device(device_));
    if (config_.use_per && samples.per_priority_sources.defined()) {
        // 旗は CPU で作り、CUDA では pinned から非同期に送る。同期コピーにすると、
        // ここで CPU がそれまでに積んだ forward の完了を待ち、後続の演算を積めなくなるため。
        const auto flags = MakeInitialPrioritySourceFlags(samples.per_priority_sources);
        initial_mask = device_.is_cuda()
            ? flags.pin_memory().to(device_, /*non_blocking=*/true)
            : flags.to(device_);
    }
    const auto initial_count = initial_mask.to(torch::kFloat32).sum();
    const auto masked_mean = [&](const torch::Tensor& values) {
        const auto value_sum = torch::where(initial_mask, values, torch::zeros_like(values)).sum();
        return torch::where(initial_count.gt(0.0f), value_sum / initial_count.clamp_min(1.0f),
            torch::full({}, nan, float_options));
    };

    // 全batch診断と初回行診断を、既存priority readbackへ同梱する固定長packにまとめる。
    return StackDiagnosticValues<IqnMetric::Count,
        IqnMetric::CurrentMcScale, IqnMetric::TargetMcScale, IqnMetric::PriorityMcRatio,
        IqnMetric::FirstPriorityMcRatio, IqnMetric::FirstPairAbsTd, IqnMetric::FirstCancellationRatio,
        IqnMetric::FirstQuantileLossNorm>(
        current_scale.mean(),
        target_scale.mean(),
        priority_ratio.mean(),
        masked_mean(priority_ratio),
        masked_mean(error.pair_abs_td),
        masked_mean(error.cancellation_ratio),
        masked_mean(error.loss.detach().to(torch::kFloat32) / static_cast<float>(N))
    ).detach();
}

std::optional<float> BatchUpdateResult::GetScalar(const std::string& key, int64_t index) const
{
    ANET_PROFILE_FUNC();
    // 名前の解析や固定のキー配列を使う群は、従来の順に先に引く。
    if (const auto value = GetAttentionEntropyScalar(key)) return value;
    if (const auto value = GetSkipScalar(key)) return value;
    if (const auto value = GetGradientNoiseScalar(key)) return value;
    if (const auto value = GetNoisyScalar(key)) return value;
    if (const auto value = GetReplayFitScalar(key)) return value;
    if (const auto value = GetMunchausenScalar(key)) return value;
    if (const auto value = GetPolicyChurnScalar(key)) return value;
    if (const auto value = GetPlasticityScalar(key)) return value;

    if (const auto value = ReadDiagnosticScalar(ParseDiagnosticMetric(key, kIqnMetricKeys),
        [&](size_t metric) -> std::optional<float> {
            return iqn_diagnostics.defined()
                ? std::optional<float>(iqn_diagnostics[static_cast<int64_t>(metric)].item<float>()) : std::nullopt;
        })) return value;

    // 固定名は1つの表に並べ、値の計算と遅延転送は当たった行だけで行う。
    struct ScalarReaderEntry {
        std::string_view key;
        std::optional<float> (*read)(const BatchUpdateResult&);
    };
    static constexpr std::array<ScalarReaderEntry, 36> kDirectScalarReaders{{
        { .key = "loss", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (!result.loss.defined()) return std::numeric_limits<float>::quiet_NaN();
            return result.loss.item<float>();
        } },
        { .key = "td_mean", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (!result.td_error.defined()) return std::numeric_limits<float>::quiet_NaN();
            if (!result.td_error_abs_cpu.defined())
                result.td_error_abs_cpu = result.td_error.abs().cpu();
            return result.td_error_abs_cpu.mean().item<float>();
        } },
        { .key = "td_std", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (!result.td_error.defined()) return std::numeric_limits<float>::quiet_NaN();
            if (!result.td_error_abs_cpu.defined())
                result.td_error_abs_cpu = result.td_error.abs().cpu();
            return result.td_error_abs_cpu.std().item<float>();
        } },
        { .key = "grad_norm", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (result.grad_norm.has_value())
                return *result.grad_norm;
            if (result.grad_norm_tensor.defined())
                return result.grad_norm_tensor.item<float>();
            // 既知keyなのでnulloptではなくNaN。nulloptは購読側で未知key扱いになり、
            // 値未成立のupdateごとにWARNを出させてしまう。
            return std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "grad_clip_ratio", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (!result.grad_norm_tensor.defined()) return std::numeric_limits<float>::quiet_NaN();
            return (result.grad_norm_tensor.item<float>() > result.grad_clip_tau) ? 1.0f : 0.0f;
        } },
        { .key = "q_max_max", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            result.TransQToCpu();
            return result.max_q_cpu.defined() ? std::optional<float>(result.max_q_cpu.max().item<float>()) : std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "q_max_mean", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            result.TransQToCpu();
            return result.max_q_cpu.defined() ? std::optional<float>(result.max_q_cpu.mean().item<float>()) : std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "q_max_std", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            result.TransQToCpu();
            return result.max_q_cpu.defined() ? std::optional<float>(result.max_q_cpu.std(false).item<float>()) : std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "q_sa_mean", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (!result.q_sa_cpu.defined() && result.q_sa.defined()) result.q_sa_cpu = result.q_sa.cpu();
            return result.q_sa_cpu.defined() ? std::optional<float>(result.q_sa_cpu.mean().item<float>()) : std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "q_max_real_max", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            result.TransRealQToCpu();
            return result.max_q_real_cpu.defined() ? std::optional<float>(result.max_q_real_cpu.max().item<float>()) : std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "q_max_real_mean", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            result.TransRealQToCpu();
            return result.max_q_real_cpu.defined() ? std::optional<float>(result.max_q_real_cpu.mean().item<float>()) : std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "q_max_real_std", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            result.TransRealQToCpu();
            return result.max_q_real_cpu.defined() ? std::optional<float>(result.max_q_real_cpu.std(false).item<float>()) : std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "q_sa_real_mean", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (!result.q_sa_real_cpu.defined() && result.q_sa_real.defined()) result.q_sa_real_cpu = result.q_sa_real.cpu();
            return result.q_sa_real_cpu.defined() ? std::optional<float>(result.q_sa_real_cpu.mean().item<float>()) : std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "q_std", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (result.q_std.defined()) return anet::ToFloat(result.q_std);
            return std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "q_gap", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (result.q_gap.defined()) return anet::ToFloat(result.q_gap);
            return std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "q_gap_rel", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (result.q_gap_rel.defined()) return anet::ToFloat(result.q_gap_rel);
            return std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "upper_tail_priority_spearman", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            return result.upper_tail_priority_spearman;
        } },
        { .key = "per_td_error_abs_max", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (result.td_error.defined())
                return result.td_error.abs().max().item<float>();
            return std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "per_prio_clip_ratio", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (result.per_clipped_count.defined() && result.per_minibatch_size > 0)
                return result.per_clipped_count.item<float>() / static_cast<float>(result.per_minibatch_size);
            return std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "per_sample_initial_ratio", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (result.per_sample_initial_count.defined() && result.per_minibatch_size > 0)
                return result.per_sample_initial_count.item<float>() / static_cast<float>(result.per_minibatch_size);
            return std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "per_sample_initial_count", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            return result.per_sample_initial_count.defined()
                ? result.per_sample_initial_count.item<float>()
                : 0.0f;
        } },
        { .key = "per_sample_fixed_initial_ratio", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            return result.per_minibatch_size > 0 && result.per_sample_fixed_initial_count.defined()
                ? result.per_sample_fixed_initial_count.item<float>() / static_cast<float>(result.per_minibatch_size)
                : std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "per_sample_max_initial_ratio", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            return result.per_minibatch_size > 0 && result.per_sample_max_initial_count.defined()
                ? result.per_sample_max_initial_count.item<float>() / static_cast<float>(result.per_minibatch_size)
                : std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "per_sample_actor_initial_ratio", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            return result.per_minibatch_size > 0 && result.per_sample_actor_initial_count.defined()
                ? result.per_sample_actor_initial_count.item<float>() / static_cast<float>(result.per_minibatch_size)
                : std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "per_priority_update_stale_ratio", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            const int64_t total = result.per_update_result.applied_count + result.per_update_result.stale_count;
            return total > 0 ? static_cast<float>(result.per_update_result.stale_count) / static_cast<float>(total)
                : std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "per_actor_learner_pair_count", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            return static_cast<float>(result.per_update_result.actor_learner_pair_count);
        } },
        { .key = "per_actor_learner_positive_pair_ratio", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            return result.per_update_result.actor_learner_positive_pair_ratio;
        } },
        { .key = "per_actor_learner_ratio_median", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            return result.per_update_result.actor_learner_ratio_median;
        } },
        { .key = "per_actor_learner_log_ratio_mean", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            return result.per_update_result.actor_learner_log_ratio_mean;
        } },
        { .key = "per_actor_learner_spearman", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            return result.per_update_result.actor_learner_spearman;
        } },
        { .key = "per_prio_max", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (result.per_priorities.defined())
                return result.per_priorities.max().item<float>();
            return std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "per_batch_prio_mean", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (result.per_priorities.defined())
                return result.per_priorities.mean().item<float>();
            return std::numeric_limits<float>::quiet_NaN();
        } },
        { .key = "per_is_weight_mean", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (result.per_is_weights.defined())
                return result.per_is_weights.mean().item<float>();
            return std::numeric_limits<float>::quiet_NaN();
        } },
        // CV = 標準偏差 / 平均
        //  高: 特定の経験に優先度が集中しており、PERが「選別」を強く行っている状態。
        //  低(0に近い) : すべての経験が似たようなTD誤差を持っており、一様サンプリングに近い状態。
        // 対策: per_alpha を調整
        { .key = "per_prio_cv", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (result.per_priorities.defined()) {
                auto mean = result.per_priorities.mean();
                auto std = result.per_priorities.std();
                return (std / (mean + 1e-9)).item<float>();
            }
            return std::numeric_limits<float>::quiet_NaN();
        } },
        // 勾配更新の偏り(IS Weightsベース)
        { .key = "per_is_ess_ratio", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (result.per_is_weights.defined() && result.per_minibatch_size > 0) {
                auto w = result.per_is_weights;
                auto sum_w = w.sum();
                auto sum_w2 = (w * w).sum();
                // ESS = (Σw)^2 / (Σw^2) / B
                return ((sum_w * sum_w) / (static_cast<float>(result.per_minibatch_size) * sum_w2 + 1e-9)).item<float>();
            }
            return std::numeric_limits<float>::quiet_NaN();
        } },
        // 有効サンプルサイズ比率 (ESS Ratio) 実質的にバッチ内の何割のデータが学習に寄与しているか (0.0 ~ 1.0) 公式: (Σp)^2 / (B * Σp^2)
        //   1.0に近い: バッチ内のデータが均等に重要。
        //   0に近い : バッチ内の極一部のデータ（外れ値など）が支配的で、実質的な学習効率が低下している警告信号。 対策: per_beta を上げる（初期値を 0.6 -> 0.8 にするなど）か、per_alpha を下げる。
        // 理想: 0.5 〜 0.8 付近
        { .key = "per_prio_ess_ratio", .read = [](const BatchUpdateResult& result) -> std::optional<float> {
            if (result.per_priorities.defined() && result.per_minibatch_size > 0) {
                auto p = result.per_priorities;
                auto sum_p = p.sum();
                auto sum_p2 = (p * p).sum();
                auto ess = (sum_p * sum_p) / (static_cast<float>(result.per_minibatch_size) * sum_p2 + 1e-9);
                return ess.item<float>();
            }
            return std::numeric_limits<float>::quiet_NaN();
        } },
    }};
    static_assert([] {
        for (size_t i = 0; i < kDirectScalarReaders.size(); ++i) {
            for (size_t j = i + 1; j < kDirectScalarReaders.size(); ++j) {
                if (kDirectScalarReaders[i].key == kDirectScalarReaders[j].key) return false;
            }
        }
        return true;
    }(), "Duplicate DQN update scalar key");
    for (const auto& entry : kDirectScalarReaders) {
        if (key == entry.key) return entry.read(*this);
    }
    return std::nullopt;
}

std::optional<float> BatchUpdateResult::GetAttentionEntropyScalar(const std::string& key) const
{
    return ReadDiagnosticScalar(ParseAttentionEntropyMetric(key), [&](int64_t layer) -> std::optional<float> {
        if (!attention_entropy.defined() || layer >= attention_entropy.numel()) return std::nullopt;
        return attention_entropy.data_ptr<float>()[layer];
    });
}

std::optional<float> BatchUpdateResult::GetSkipScalar(const std::string& key) const
{
    return ReadDiagnosticScalar(ParseDiagnosticMetric(key, kSkipUpdateMetricKeys), [&](size_t metric) -> std::optional<float> {
        if (!skip_metrics.defined()) return std::nullopt;
        if (!skip_metrics_cpu.defined()) skip_metrics_cpu = skip_metrics.cpu();
        return skip_metrics_cpu[static_cast<int64_t>(metric)].item<float>();
    });
}

std::optional<float> BatchUpdateResult::GetGradientNoiseScalar(const std::string& key) const
{
    return ReadDiagnosticScalar(ParseGradientNoiseMetric(key), [&](size_t metric) -> std::optional<float> {
        return grad_noise ? std::optional<float>((*grad_noise)[metric]) : std::nullopt;
    });
}

std::optional<float> BatchUpdateResult::GetNoisyScalar(const std::string& key) const
{
    return ReadDiagnosticScalar(ParseNoisyMetric(key), [&](size_t metric) -> std::optional<float> {
        return noisy_metrics ? std::optional<float>((*noisy_metrics)[metric]) : std::nullopt;
    });
}

std::optional<float> BatchUpdateResult::GetReplayFitScalar(const std::string& key) const
{
    return ReadDiagnosticScalar(ParseReplayFitMetric(key), [&](size_t metric) -> std::optional<float> {
        return replay_fit_metrics ? std::optional<float>((*replay_fit_metrics)[metric]) : std::nullopt;
    });
}

std::optional<float> BatchUpdateResult::GetMunchausenScalar(const std::string& key) const
{
    return ReadDiagnosticScalar(ParseDiagnosticMetric(key, kMunchausenMetricKeys), [&](size_t metric) -> std::optional<float> {
        if (!munchausen_diagnostics.defined()) return std::nullopt;
        if (munchausen_diagnostics.dim() != 1 || munchausen_diagnostics.numel() != MunchausenMetric::Count) {
            ANET_SYSTEM_ERROR("Invalid Munchausen diagnostics shape=" << munchausen_diagnostics.sizes()
                << "; expected [" << MunchausenMetric::Count << "].");
        }
        return munchausen_diagnostics[static_cast<int64_t>(metric)].item<float>();
    });
}

std::optional<float> BatchUpdateResult::GetPolicyChurnScalar(const std::string& key) const
{
    return ReadDiagnosticScalar(ParsePolicyChurnMetric(key), [&](size_t metric) -> std::optional<float> {
        if (!policy_churn_metrics.defined()) return std::nullopt;
        if (policy_churn_metrics.numel() != static_cast<int64_t>(kPolicyChurnMetricKeys.size())) {
            ANET_SYSTEM_ERROR("DQN policy churn scalar pack has invalid size="
                << policy_churn_metrics.numel() << " expected=" << kPolicyChurnMetricKeys.size() << ".");
        }
        return policy_churn_metrics[static_cast<int64_t>(metric)].item<float>();
    });
}

std::optional<float> BatchUpdateResult::GetPlasticityScalar(const std::string& key) const
{
    if (const auto value = ReadDiagnosticScalar(ParseDiagnosticMetric(key, kPlasticityWeightNormMetricKeys),
        [&](size_t metric) -> std::optional<float> {
            if (!plasticity_weight_norms.defined()) return std::nullopt;
            if (!plasticity_weight_norms_cpu.defined()) plasticity_weight_norms_cpu = plasticity_weight_norms.cpu();
            if (plasticity_weight_norms_cpu.numel() != PlasticityWeightNormMetric::PackCount) {
                ANET_SYSTEM_ERROR("DQN plasticity weight norm pack has invalid size="
                    << plasticity_weight_norms_cpu.numel() << " expected=" << PlasticityWeightNormMetric::PackCount << ".");
            }
            ValidateSpectralNormSentinel(
                plasticity_weight_norms_cpu[PlasticityWeightNormMetric::OnlineInvalidCount].item<float>(), "online", plasticity_online_network);
            ValidateSpectralNormSentinel(
                plasticity_weight_norms_cpu[PlasticityWeightNormMetric::TargetInvalidCount].item<float>(), "target", plasticity_target_network);
            return plasticity_weight_norms_cpu[static_cast<int64_t>(metric)].item<float>();
        })) return value;

    const torch::Tensor* features = nullptr;
    const anet::nn::PlasticityMetricRequest* request = nullptr;
    std::optional<anet::nn::PlasticityMetrics>* cache = nullptr;
    std::string suffix;
    if (key.starts_with(kPlasticityTargetPrefix)) {
        features = &plasticity_target_features;
        request = &plasticity_target_request;
        cache = &plasticity_target_metrics_cache;
        suffix = key.substr(kPlasticityTargetPrefix.size());
    } else if (key.starts_with(kPlasticityPrefix)) {
        features = &plasticity_features;
        request = &plasticity_request;
        cache = &plasticity_metrics_cache;
        suffix = key.substr(kPlasticityPrefix.size());
    } else {
        return std::nullopt;
    }

    return ReadDiagnosticScalar(anet::nn::ParsePlasticityMetricSuffix(suffix), [&](auto metric) -> std::optional<float> {
    if (!request->Contains(metric)) return std::nullopt;
    if (!features->defined()) return std::nullopt;
    if (!cache->has_value()) {
        *cache = anet::nn::ComputePlasticityMetrics(*features, *request);
    }
    const auto value = cache->value().Get(metric);
    if (!value.has_value()) {
        ANET_SYSTEM_ERROR("BatchUpdateResult plasticity cache is missing a requested metric.");
    }
    return value;
    });
}

void BatchUpdateResult::ValidateSpectralNormSentinel(
    float invalid_count,
    const std::string& network_name,
    const std::shared_ptr<anet::nn::Network>& network)
{
    if (invalid_count < 1.0f) return;
    if (!network) {
        ANET_SYSTEM_ERROR("DQN spectral normalization sentinel failed: network="
            << network_name << " invalid_count=" << invalid_count << " network handle is unavailable.");
    }
    for (const auto& entry : network->GetSpectralNormEntries()) {
        const auto sigma = anet::nn::ComputeSpectralSigma(
            entry.weight.reshape({ entry.weight.size(0), -1 }), entry.u, entry.v).item<float>();
        const bool valid = std::isfinite(sigma)
            && (entry.mode == anet::nn::WeightNormMode::kSpectralCap || sigma > 0.0f);
        if (!valid) {
            ANET_SYSTEM_ERROR("DQN spectral normalization is invalid: network="
                << network_name << " layer=" << entry.name << " sigma=" << sigma << ".");
        }
    }
    ANET_SYSTEM_ERROR("DQN spectral normalization sentinel mismatch: network="
        << network_name << " invalid_count=" << invalid_count << ".");
}

void BatchUpdateResult::TransQToCpu() const
{
    if (max_q_cpu.defined()) return;
    // 未測定のupdateではmax_qがundefinedなので、real側と同じくCPU転送を試みない。
    if (max_q.defined()) max_q_cpu = max_q.cpu();
}

void BatchUpdateResult::TransRealQToCpu() const
{
    if (max_q_real_cpu.defined()) return;
    if (max_q_real.defined()) max_q_real_cpu = max_q_real.cpu();
}
