// dqn_based_agent.hpp
#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <limits>
#include <span>
#include <string_view>
#include <unordered_set>
#include <utility>
#include "anet/random.hpp"
#include "anet/stacker.hpp"
#include "anet/schedule.hpp"
#include "anet/nn_util.hpp"
#include "anet/nn.hpp"
#include "anet/rl.hpp"
#include "anet/scaler.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/agent.hpp"
#include "anet/transfer.hpp"


namespace anet::rl::dqn {

    /// 未知keyはnullopt、認識したindexed keyの不正番号はfail-fastにする。
    std::optional<int64_t> ParseAttentionEntropyMetric(std::string_view key);

    struct DiagnosticMetricEntry {
        size_t index;
        const char* key;
    };

    template<size_t Count>
    consteval std::array<const char*, Count> MakeDiagnosticMetricKeys(
        const std::array<DiagnosticMetricEntry, Count>& entries)
    {
        std::array<const char*, Count> keys{};
        for (size_t i = 0; i < Count; ++i) {
            if (entries[i].index != i) throw "DQN diagnostic metric order does not match its named index";
            keys[i] = entries[i].key;
        }
        return keys;
    }

    template<size_t Count>
    std::optional<size_t> ParseDiagnosticMetric(
        const std::string& key, const std::array<const char*, Count>& keys)
    {
        for (size_t i = 0; i < Count; ++i) if (key == keys[i]) return i;
        return std::nullopt;
    }

    template<size_t Count>
    class DiagnosticDemandTable {
    public:
        struct Entry {
            size_t metric;
            int interval;
            anet::IntervalGate gate;
            Entry(size_t metric, int interval)
                : metric(metric), interval(interval), gate(static_cast<uint64_t>(interval)) {}
        };
        explicit DiagnosticDemandTable(const std::array<const char*, Count>& keys) : keys_(keys) {}
        std::optional<size_t> Add(const std::string& key, int interval) {
            const auto metric = ParseDiagnosticMetric(key, keys_);
            if (metric) entries_.emplace_back(*metric, interval);
            return metric;
        }
        std::array<bool, Count> MakeRequest(uint64_t step) {
            std::array<bool, Count> request{};
            for (auto& entry : entries_) if (entry.gate.ShouldFire(step)) request[entry.metric] = true;
            return request;
        }
        void clear() { entries_.clear(); }
        bool empty() const { return entries_.empty(); }
        auto begin() { return entries_.begin(); }
        auto end() { return entries_.end(); }
        auto begin() const { return entries_.begin(); }
        auto end() const { return entries_.end(); }
    private:
        std::array<const char*, Count> keys_;
        std::vector<Entry> entries_;
    };

    template<size_t Count, size_t... Indices, class... Values>
    auto MakeDiagnosticValues(Values&&... values)
    {
        static_assert(sizeof...(Indices) == Count && sizeof...(Values) == Count);
        static_assert([] {
            constexpr std::array<size_t, sizeof...(Indices)> indices{Indices...};
            for (size_t i = 0; i < indices.size(); ++i) if (indices[i] != i) return false;
            return true;
        }(), "DQN diagnostic values do not match their named order");
        return std::array{std::forward<Values>(values)...};
    }

    template<size_t Count, size_t... Indices, class... Values>
    torch::Tensor StackDiagnosticValues(Values&&... values)
    {
        const auto ordered = MakeDiagnosticValues<Count, Indices...>(std::forward<Values>(values)...);
        return torch::stack(ordered);
    }

    // 先にキーを解析し、未知キーでは遅延転送や診断の計算を始めない。
    template<class Index, class Read>
    std::optional<float> ReadDiagnosticScalar(std::optional<Index> metric, Read&& read)
    {
        if (!metric) return std::nullopt;
        return read(*metric).value_or(std::numeric_limits<float>::quiet_NaN());
    }

    struct IqnMetric {
        enum : size_t { CurrentMcScale, TargetMcScale, PriorityMcRatio, FirstPriorityMcRatio, FirstPairAbsTd, FirstCancellationRatio, FirstQuantileLossNorm, Count };
    };
    inline constexpr auto kIqnMetricKeys = MakeDiagnosticMetricKeys<IqnMetric::Count>({{
        { .index = IqnMetric::CurrentMcScale, .key = "iqn_current_mc_scale" },
        { .index = IqnMetric::TargetMcScale, .key = "iqn_target_mc_scale" },
        { .index = IqnMetric::PriorityMcRatio, .key = "iqn_priority_mc_ratio" },
        { .index = IqnMetric::FirstPriorityMcRatio, .key = "iqn_first_priority_mc_ratio" },
        { .index = IqnMetric::FirstPairAbsTd, .key = "iqn_first_pair_abs_td" },
        { .index = IqnMetric::FirstCancellationRatio, .key = "iqn_first_cancellation_ratio" },
        { .index = IqnMetric::FirstQuantileLossNorm, .key = "iqn_first_quantile_loss_norm" },
    }});

    struct PlasticityWeightNormMetric {
        enum : size_t { Feature, Readout, FeatureEffective, ReadoutEffective, SpectralSigmaFeature, SpectralSigmaReadout, Count, OnlineInvalidCount = Count, TargetInvalidCount, PackCount };
    };
    inline constexpr auto kPlasticityWeightNormMetricKeys = MakeDiagnosticMetricKeys<PlasticityWeightNormMetric::Count>({{
        { .index = PlasticityWeightNormMetric::Feature, .key = "plasticity_weight_norm_feature" },
        { .index = PlasticityWeightNormMetric::Readout, .key = "plasticity_weight_norm_readout" },
        { .index = PlasticityWeightNormMetric::FeatureEffective, .key = "plasticity_weight_norm_feature_effective" },
        { .index = PlasticityWeightNormMetric::ReadoutEffective, .key = "plasticity_weight_norm_readout_effective" },
        { .index = PlasticityWeightNormMetric::SpectralSigmaFeature, .key = "plasticity_spectral_sigma_feature" },
        { .index = PlasticityWeightNormMetric::SpectralSigmaReadout, .key = "plasticity_spectral_sigma_readout" },
    }});

    struct LearnerScheduleMetric {
        enum : size_t { LearningRate, WeightDecay, SoftUpdateTau, Gamma, MunchausenAlpha, MunchausenEntropyTau, Count };
    };
    inline constexpr auto kLearnerScheduleMetricKeys = MakeDiagnosticMetricKeys<LearnerScheduleMetric::Count>({{
        { .index = LearnerScheduleMetric::LearningRate, .key = "learning_rate" },
        { .index = LearnerScheduleMetric::WeightDecay, .key = "weight_decay" },
        { .index = LearnerScheduleMetric::SoftUpdateTau, .key = "soft_update_tau" },
        { .index = LearnerScheduleMetric::Gamma, .key = "gamma" },
        { .index = LearnerScheduleMetric::MunchausenAlpha, .key = "munchausen_alpha" },
        { .index = LearnerScheduleMetric::MunchausenEntropyTau, .key = "munchausen_entropy_tau" },
    }});
    struct MunchausenMetric {
        enum : size_t { ScaledLogPolicyMean, ClipRatio, BonusMean, NextEntropy, SoftGap, Count };
    };
    inline constexpr auto kMunchausenMetricKeys = MakeDiagnosticMetricKeys<MunchausenMetric::Count>({{
        { .index = MunchausenMetric::ScaledLogPolicyMean, .key = "munchausen_scaled_log_policy_mean" },
        { .index = MunchausenMetric::ClipRatio, .key = "munchausen_clip_ratio" },
        { .index = MunchausenMetric::BonusMean, .key = "munchausen_bonus_mean" },
        { .index = MunchausenMetric::NextEntropy, .key = "munchausen_next_entropy" },
        { .index = MunchausenMetric::SoftGap, .key = "munchausen_soft_gap" },
    }});
    struct SkipUpdateMetric {
        enum : size_t { Loss, TdAbsMean, SampleKMean, KmaxMean, GradNorm, PrimaryGradNorm, PrimaryClipRatio, Count };
    };
    inline constexpr auto kSkipUpdateMetricKeys = MakeDiagnosticMetricKeys<SkipUpdateMetric::Count>({{
        { .index = SkipUpdateMetric::Loss, .key = "skip_loss" },
        { .index = SkipUpdateMetric::TdAbsMean, .key = "skip_td_abs_mean" },
        { .index = SkipUpdateMetric::SampleKMean, .key = "skip_sample_k_mean" },
        { .index = SkipUpdateMetric::KmaxMean, .key = "skip_kmax_mean" },
        { .index = SkipUpdateMetric::GradNorm, .key = "skip_grad_norm" },
        { .index = SkipUpdateMetric::PrimaryGradNorm, .key = "skip_primary_grad_norm" },
        { .index = SkipUpdateMetric::PrimaryClipRatio, .key = "skip_primary_clip_ratio" },
    }});

    torch::Tensor MakeRiskBiasedScore(float tau, const torch::Tensor& q_quantiles, bool use_tail_mean);

    struct MunchausenTargetTerms {
        torch::Tensor bonus;
        torch::Tensor next_policy;
        torch::Tensor next_scaled_log_policy;
        torch::Tensor diagnostics;
    };

    MunchausenTargetTerms MakeMunchausenTargetTerms(
        const torch::Tensor& current_score, const torch::Tensor& next_score,
        const torch::Tensor& next_mean_q, const torch::Tensor& actions,
        const MunchausenValues& config, bool collect_diagnostics = true);

    struct PolicyChurnMetric {
        enum : size_t { ActionRatio, QDeltaAbsMean, QDeltaSignedMax, QDeltaSignedMin, TargetPolicyDisagreement, TargetQDeltaAbsMean, TargetSyncAge, Count };
    };
    inline constexpr auto kPolicyChurnMetricKeys = MakeDiagnosticMetricKeys<PolicyChurnMetric::Count>({{
        { .index = PolicyChurnMetric::ActionRatio, .key = "policy_churn_action_ratio" },
        { .index = PolicyChurnMetric::QDeltaAbsMean, .key = "policy_churn_q_delta_abs_mean" },
        { .index = PolicyChurnMetric::QDeltaSignedMax, .key = "policy_churn_q_delta_signed_max" },
        { .index = PolicyChurnMetric::QDeltaSignedMin, .key = "policy_churn_q_delta_signed_min" },
        { .index = PolicyChurnMetric::TargetPolicyDisagreement, .key = "policy_churn_target_policy_disagreement" },
        { .index = PolicyChurnMetric::TargetQDeltaAbsMean, .key = "policy_churn_target_q_delta_abs_mean" },
        { .index = PolicyChurnMetric::TargetSyncAge, .key = "policy_churn_target_sync_age" },
    }});

    struct GradientNoiseMetric {
        enum : size_t { ScaleAdam, ScaleAdamFeature, ScaleAdamReadout, RatioAdam, Scale, SignalAdam, VarianceAdam, SignalAdamRaw, VarianceAdamRaw, SignalRaw, VarianceRaw, SignalAdamFeatureRaw, VarianceAdamFeatureRaw, SignalAdamReadoutRaw, VarianceAdamReadoutRaw, Count };
    };
    inline constexpr auto kGradientNoiseMetricKeys = MakeDiagnosticMetricKeys<GradientNoiseMetric::Count>({{
        { .index = GradientNoiseMetric::ScaleAdam, .key = "grad_noise_scale_adam" },
        { .index = GradientNoiseMetric::ScaleAdamFeature, .key = "grad_noise_scale_adam_feature" },
        { .index = GradientNoiseMetric::ScaleAdamReadout, .key = "grad_noise_scale_adam_readout" },
        { .index = GradientNoiseMetric::RatioAdam, .key = "grad_noise_ratio_adam" },
        { .index = GradientNoiseMetric::Scale, .key = "grad_noise_scale" },
        { .index = GradientNoiseMetric::SignalAdam, .key = "grad_noise_signal_adam" },
        { .index = GradientNoiseMetric::VarianceAdam, .key = "grad_noise_variance_adam" },
        { .index = GradientNoiseMetric::SignalAdamRaw, .key = "grad_noise_signal_adam_raw" },
        { .index = GradientNoiseMetric::VarianceAdamRaw, .key = "grad_noise_variance_adam_raw" },
        { .index = GradientNoiseMetric::SignalRaw, .key = "grad_noise_signal_raw" },
        { .index = GradientNoiseMetric::VarianceRaw, .key = "grad_noise_variance_raw" },
        { .index = GradientNoiseMetric::SignalAdamFeatureRaw, .key = "grad_noise_signal_adam_feature_raw" },
        { .index = GradientNoiseMetric::VarianceAdamFeatureRaw, .key = "grad_noise_variance_adam_feature_raw" },
        { .index = GradientNoiseMetric::SignalAdamReadoutRaw, .key = "grad_noise_signal_adam_readout_raw" },
        { .index = GradientNoiseMetric::VarianceAdamReadoutRaw, .key = "grad_noise_variance_adam_readout_raw" },
    }});
    
    using GradientNoiseMetrics = std::array<float, kGradientNoiseMetricKeys.size()>;
    inline std::optional<size_t> ParseGradientNoiseMetric(const std::string& key)
    {
        return ParseDiagnosticMetric(key, kGradientNoiseMetricKeys);
    }

    struct NoisyMetric {
        enum : size_t { SigmaMeanAbsFeatureWeight, SigmaMeanAbsFeatureBias, SigmaMeanAbsReadoutWeight, SigmaMeanAbsReadoutBias, SigmaL2RatioFeatureWeight, SigmaL2RatioFeatureBias, SigmaL2RatioReadoutWeight, SigmaL2RatioReadoutBias, QAbsDiff, ActionDisagreementRatio, ActionEntropy, Count };
    };
    inline constexpr auto kNoisyMetricKeys = MakeDiagnosticMetricKeys<NoisyMetric::Count>({{
        { .index = NoisyMetric::SigmaMeanAbsFeatureWeight, .key = "noisy_sigma_mean_abs_feature_weight" },
        { .index = NoisyMetric::SigmaMeanAbsFeatureBias, .key = "noisy_sigma_mean_abs_feature_bias" },
        { .index = NoisyMetric::SigmaMeanAbsReadoutWeight, .key = "noisy_sigma_mean_abs_readout_weight" },
        { .index = NoisyMetric::SigmaMeanAbsReadoutBias, .key = "noisy_sigma_mean_abs_readout_bias" },
        { .index = NoisyMetric::SigmaL2RatioFeatureWeight, .key = "noisy_sigma_l2_ratio_feature_weight" },
        { .index = NoisyMetric::SigmaL2RatioFeatureBias, .key = "noisy_sigma_l2_ratio_feature_bias" },
        { .index = NoisyMetric::SigmaL2RatioReadoutWeight, .key = "noisy_sigma_l2_ratio_readout_weight" },
        { .index = NoisyMetric::SigmaL2RatioReadoutBias, .key = "noisy_sigma_l2_ratio_readout_bias" },
        { .index = NoisyMetric::QAbsDiff, .key = "noisy_q_abs_diff" },
        { .index = NoisyMetric::ActionDisagreementRatio, .key = "noisy_action_disagreement_ratio" },
        { .index = NoisyMetric::ActionEntropy, .key = "noisy_action_entropy" },
    }});
    using NoisyMetrics = std::array<float, kNoisyMetricKeys.size()>;
    std::array<float, 3> MeasureNoisyQProbe(const torch::Tensor& mean_q, int num_draws,
        const std::function<torch::Tensor(int)>& draw);


    inline std::optional<size_t> ParseNoisyMetric(const std::string& key)
    {
        return ParseDiagnosticMetric(key, kNoisyMetricKeys);
    }

    struct ReplayFitMetric {
        enum : size_t { UnsampledTdMean, SampledTdMean, UnsampledTdRatio, UnsampledLossMean, SampledLossMean, UnsampledLossRatio, UnsampledCount, SampledCount, UnsampledAgeMean, SampledAgeMean, UniformTdMean, PerTdMean, PerSelectivity, Count };
    };
    inline constexpr auto kReplayFitMetricKeys = MakeDiagnosticMetricKeys<ReplayFitMetric::Count>({{
        { .index = ReplayFitMetric::UnsampledTdMean, .key = "replay_fit_unsampled_td_mean" },
        { .index = ReplayFitMetric::SampledTdMean, .key = "replay_fit_sampled_td_mean" },
        { .index = ReplayFitMetric::UnsampledTdRatio, .key = "replay_fit_unsampled_td_ratio" },
        { .index = ReplayFitMetric::UnsampledLossMean, .key = "replay_fit_unsampled_loss_mean" },
        { .index = ReplayFitMetric::SampledLossMean, .key = "replay_fit_sampled_loss_mean" },
        { .index = ReplayFitMetric::UnsampledLossRatio, .key = "replay_fit_unsampled_loss_ratio" },
        { .index = ReplayFitMetric::UnsampledCount, .key = "replay_fit_unsampled_count" },
        { .index = ReplayFitMetric::SampledCount, .key = "replay_fit_sampled_count" },
        { .index = ReplayFitMetric::UnsampledAgeMean, .key = "replay_fit_unsampled_age_mean" },
        { .index = ReplayFitMetric::SampledAgeMean, .key = "replay_fit_sampled_age_mean" },
        { .index = ReplayFitMetric::UniformTdMean, .key = "replay_fit_uniform_td_mean" },
        { .index = ReplayFitMetric::PerTdMean, .key = "replay_fit_per_td_mean" },
        { .index = ReplayFitMetric::PerSelectivity, .key = "replay_fit_per_selectivity" },
    }});
    using ReplayFitMetrics = std::array<float, kReplayFitMetricKeys.size()>;

    struct ReplayFitRequirements {
        bool uniform;
        bool per;
        bool td_u;
        bool td_s;
        bool loss_u;
        bool loss_s;
        bool counts;

        bool NeedsSampling() const { return td_u || td_s || loss_u || loss_s; }
    };

    ReplayFitRequirements ResolveReplayFitRequirements(
        const std::array<bool, kReplayFitMetricKeys.size()>& request, bool use_per);

    inline std::optional<size_t> ParseReplayFitMetric(const std::string& key)
    {
        return ParseDiagnosticMetric(key, kReplayFitMetricKeys);
    }

    inline std::optional<size_t> ParsePolicyChurnMetric(const std::string& key)
    {
        return ParseDiagnosticMetric(key, kPolicyChurnMetricKeys);
    }

    ReplayInitialPriorityMode ParseReplayInitialPriorityMode(const LearnerConfig& config);
    void ValidateReplayPriorityConfig(const LearnerConfig& config, ReplayInitialPriorityMode initial_priority_mode);
    torch::Tensor TransformH(const torch::Tensor& x, float epsilon);
    float TransformH(float x, float epsilon);
    torch::Tensor TransformHInv(const torch::Tensor& x, float epsilon);
    float TransformHInv(float x, float epsilon);

    torch::Tensor GenerateTaus(
        int64_t batch_size,
        int64_t num_taus,
        const std::string& sample_mode,
        float tau_min,
        float tau_max,
        const torch::Device& device,
        anet::RandomGenerator& rnd);

    torch::Tensor GenerateTaus(
        const torch::Tensor& tau_min_per_env,
        float tau_max,
        int64_t num_taus,
        const std::string& sample_mode,
        anet::RandomGenerator& rnd);

    struct PerRawPriorityBatchResult {
        torch::Tensor priorities;    ///< clip適用後のfloat32 priority [B]
        torch::Tensor clipped_count; ///< clip前priorityが上限を超えた件数のscalar
    };

    PerRawPriorityBatchResult MakePerRawPriorityBatch(
        const torch::Tensor& td_error, float per_eps, bool use_clip, float clip_value);
    torch::Tensor MakePerRawPriority(
        const torch::Tensor& td_error, float per_eps, bool use_clip, float clip_value);
    float MakePerRawPriority(
        float td_error, float per_eps, bool use_clip, float clip_value);
    std::unique_ptr<InitialPriorityEstimator> CreateInitialPriorityEstimator(const LearnerConfig& config);

    inline constexpr int64_t kActorQHintColumnCount = 5;
    inline constexpr int64_t kActorQSaColumn = 0;
    inline constexpr int64_t kActorStateValueColumn = 1;
    inline constexpr int64_t kActorMunchausenTermColumn = 2;
    inline constexpr int64_t kActorIntrinsicQSaColumn = 3;
    inline constexpr int64_t kActorIntrinsicStateValueColumn = 4;

    struct ActorQHintBatch {
        torch::Tensor actor_q_sa;
        torch::Tensor actor_state_value;
        torch::Tensor munchausen_term;
        torch::Tensor intrinsic_q_sa;
        torch::Tensor intrinsic_state_value;
    };

    struct ActorQHintRow {
        float actor_q_sa = 0.0f;
        float actor_state_value = 0.0f;
        float munchausen_term = 0.0f;
        float intrinsic_q_sa = 0.0f;
        float intrinsic_state_value = 0.0f;
    };

    torch::Tensor PackActorQHint(
        const torch::Tensor& actor_q_sa, const torch::Tensor& actor_state_value, const torch::Tensor& munchausen_term,
        const torch::Tensor& intrinsic_q_sa, const torch::Tensor& intrinsic_state_value);
    ActorQHintBatch DecodeActorQHint(const torch::Tensor& payload);
    ActorQHintRow DecodeActorQHint(std::span<const float> payload);

    const float MET_EMA_DECAY = 0.001f;  // 平滑化係数(メトリクス用)
    const float MET_EMA_DECAY_ACT = 0.0005f;  // 平滑化係数(メトリクス用)action_ema用


    // ======================================================
    //  データ構造
    // ======================================================

    /// ランタイム変数
    struct anet::rl::dqn::RuntimeVars {
        float uqe_tau = 0.9f;   // 互換性のために残す
        anet::rl::step_t learn_step = 0;
        float per_beta = 0.0f;  ///< PER用beta
    };


    // ======================================================
    // BatchUpdateResult 
    // ======================================================

    class BatchUpdateResult : public anet::rl::BatchUpdateResult {
    public:
        // grad/loss/td
        std::optional<GradientNoiseMetrics> grad_noise;
        torch::Tensor grad_norm_tensor;
        torch::Tensor skip_metrics;
        mutable torch::Tensor skip_metrics_cpu;
        std::optional<float> grad_norm;
        float grad_clip_ratio = 0.0f;
        torch::Tensor loss;
        torch::Tensor td_error;
        mutable torch::Tensor td_error_abs_cpu;
        float grad_clip_tau;

		// Q Value Metrics Source Tensors
        torch::Tensor max_q;
        mutable torch::Tensor max_q_cpu;
        torch::Tensor max_q_real;
        mutable torch::Tensor max_q_real_cpu;
        torch::Tensor q_sa;
        mutable torch::Tensor q_sa_cpu;
        torch::Tensor q_sa_real;
        mutable torch::Tensor q_sa_real_cpu;
        torch::Tensor q_gap;
        torch::Tensor q_gap_rel;

        // PER Metrics Source Tensors
        torch::Tensor per_is_weights;      ///< IS Weights (B,)
        torch::Tensor per_priorities;      ///< Updated Priorities (B,)
        torch::Tensor per_clipped_count;   ///< Clipped Count (scalar tensor)
        torch::Tensor per_sample_initial_count; ///< 初期優先度のままサンプルされた件数
        torch::Tensor per_sample_fixed_initial_count; ///< fixed_initial sourceのサンプル件数
        torch::Tensor per_sample_max_initial_count;   ///< max_initial sourceのサンプル件数
        torch::Tensor per_sample_actor_initial_count; ///< actor_initial sourceのサンプル件数
        ReplayPriorityUpdateResult per_update_result; ///< 対応するLearner minibatchの優先度更新結果
        long per_minibatch_size = 0;       ///< Minibatch Size

        // QR-DQN Metrics
        torch::Tensor q_std; // 分布の標準偏差

        // IQN診断（CPU scalar pack）
        torch::Tensor iqn_diagnostics;
        torch::Tensor munchausen_diagnostics; ///< CPU上の固定長5値。未成立時はundefined。
        float upper_tail_priority_spearman = std::numeric_limits<float>::quiet_NaN();

        // 可塑性メトリクス元特徴。CPU統計は購読側が初めて読む時だけ計算する。
        torch::Tensor plasticity_features;
        torch::Tensor plasticity_target_features;
        torch::Tensor plasticity_weight_norms;
        std::shared_ptr<anet::nn::Network> plasticity_online_network;
        std::shared_ptr<anet::nn::Network> plasticity_target_network;
        anet::nn::PlasticityMetricRequest plasticity_request;
        anet::nn::PlasticityMetricRequest plasticity_target_request;
        mutable torch::Tensor plasticity_weight_norms_cpu;
        mutable std::optional<anet::nn::PlasticityMetrics> plasticity_metrics_cache;
        mutable std::optional<anet::nn::PlasticityMetrics> plasticity_target_metrics_cache;

        torch::Tensor attention_entropy; ///< 測定した更新だけに属する全層のCPU FP32値。
        // policy churn の 7 scalar を key 順に保持する CPU float32 pack。
        torch::Tensor policy_churn_metrics;
        std::optional<NoisyMetrics> noisy_metrics; ///< 購読した更新だけに属する診断値。
        std::optional<ReplayFitMetrics> replay_fit_metrics; ///< 測定回だけ生成するCPU結果。

    public:
        BatchUpdateResult() = default;

        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;

        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index) const override
        {
            if (key == "max_q") return max_q;
            return std::nullopt;
        }

        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index) const override
        {
            return std::nullopt;
        }
    private:
        std::optional<float> GetAttentionEntropyScalar(const std::string& key) const;
        std::optional<float> GetSkipScalar(const std::string& key) const;
        std::optional<float> GetGradientNoiseScalar(const std::string& key) const;
        std::optional<float> GetNoisyScalar(const std::string& key) const;
        std::optional<float> GetReplayFitScalar(const std::string& key) const;
        std::optional<float> GetMunchausenScalar(const std::string& key) const;
        std::optional<float> GetPolicyChurnScalar(const std::string& key) const;

        std::optional<float> GetPlasticityScalar(const std::string& key) const;

        static void ValidateSpectralNormSentinel(
            float invalid_count,
            const std::string& network_name,
            const std::shared_ptr<anet::nn::Network>& network);

        void TransQToCpu() const;
        void TransRealQToCpu() const;
    };

    struct NormalizedSampleObservations {
        anet::TensorDict obs;
        anet::TensorDict next_obs;
    };

    struct OptimizerStepResult {
        std::optional<GradientNoiseMetrics> grad_noise;
        torch::Tensor grad_norm_tensor;
        torch::Tensor skip_grad_norm;
        torch::Tensor skip_primary_grad_norm;
        torch::Tensor extrinsic_head_grad_norm;
        torch::Tensor intrinsic_head_grad_norm;
        torch::Tensor intrinsic_grad_norm;
        std::optional<float> grad_norm;
        float grad_clip_ratio = 0.0f;
        float grad_clip_tau = std::numeric_limits<float>::infinity();
    };

    struct LearningTargets {
        torch::Tensor primary;
        torch::Tensor window;
        torch::Tensor diagnostics;
        torch::Tensor intrinsic;
    };

    struct SkipLearningResult {
        torch::Tensor loss;
        torch::Tensor td;
        torch::Tensor k_mean;
        torch::Tensor kmax_mean;
    };

    struct PerPriorityUpdateInfo {
        torch::Tensor per_clipped_count;
        torch::Tensor per_priorities;
        torch::Tensor per_is_weights;
        torch::Tensor per_sample_initial_count;
        torch::Tensor per_sample_fixed_initial_count; ///< fixed_initial sourceのサンプル件数
        torch::Tensor per_sample_max_initial_count;   ///< max_initial sourceのサンプル件数
        torch::Tensor per_sample_actor_initial_count; ///< actor_initial sourceのサンプル件数
        ReplayPriorityUpdateResult per_update_result; ///< ReplayBufferへ適用済みの更新結果
        long per_minibatch_size = 0;                  ///< source比率の分母となるminibatch size
        torch::Tensor iqn_diagnostics;                ///< IQN診断scalarのCPU pack
        torch::Tensor munchausen_diagnostics;         ///< Munchausen診断scalarのCPU pack
        float upper_tail_priority_spearman = std::numeric_limits<float>::quiet_NaN();
    };

    struct PerPriorityUpdatePending {
        std::vector<int64_t> indices;                   ///< CPU上のgeneration付きreplay item key
        anet::transfer::HostReadback priority_readback; ///< priority、clip件数、診断scalarをまとめた遅延D2H結果
        torch::Tensor per_is_weights;                   ///< 対応minibatchのIS weight
        torch::Tensor per_sample_initial_count;         ///< 全initial sourceのサンプル件数
        torch::Tensor per_sample_fixed_initial_count;   ///< fixed_initial sourceのサンプル件数
        torch::Tensor per_sample_max_initial_count;     ///< max_initial sourceのサンプル件数
        torch::Tensor per_sample_actor_initial_count;   ///< actor_initial sourceのサンプル件数
        long per_minibatch_size = 0;                    ///< source比率の分母となるminibatch size
        int64_t iqn_diagnostics_count = 0;              ///< pack末尾のIQN診断scalar数
        int64_t munchausen_diagnostics_count = 0;       ///< IQN診断に続くMunchausen診断scalar数
        int64_t upper_tail_std_count = 0;               ///< pack末尾のsample単位upper-tail幅数
        bool per_enabled = false;                       ///< ReplayBuffer priority更新を行うか
        bool enabled = false;                           ///< packed readbackが必要なminibatchか
    };

    struct QuantileMetrics {
        torch::Tensor q_sa;
        torch::Tensor max_q;
        torch::Tensor q_std;
        torch::Tensor q_gap;
        torch::Tensor q_gap_rel;
    };

    struct IqnLossResult {
        torch::Tensor element_loss;
        torch::Tensor pair_abs_td;
        torch::Tensor cancellation_ratio;
    };


    // ======================================================
    //  NetworkModel
    // ======================================================

    struct SkipNetworkDefinition {
        SkipConfig config;
        anet::nn::NetworkConfig network_config;
        std::shared_ptr<anet::nn::NetworkHeadFactory> head_factory;
        anet::seed_t seed;
    };

    class NetworkModel : public anet::Serializable {
    public:
        static constexpr const char* kKeyActionInput = "action";

        NetworkModel(
            const NetworkModelConfig& config,
            const torch::Device device,
            const anet::nn::NetworkConfig& network_config,
            const anet::TensorSpecMap& obs_spec,
            int64_t n_actions,
            std::shared_ptr<anet::nn::NetworkHeadFactory> head_factory,
            bool distributional,
            anet::seed_t network_seed,
            const SkipNetworkDefinition* skip_definition = nullptr);
    protected:
        NetworkModel(
            const NetworkModelConfig& config,
            std::shared_ptr<anet::nn::Network> online_net,
            std::shared_ptr<anet::nn::Network> target_net,
            int64_t n_actions,
            bool distributional);
    public:

        /// 訓練モードを一時的に有効にし、復元して返す。
        anet::TensorDict ForwardOnlineWithTrain(const anet::TensorDict& obs, anet::nn::NetworkForwardExecutor& forward, const anet::nn::NetworkForwardCallInfo& info,
            anet::nn::NetworkForwardCapture* capture = nullptr) const;
        anet::TensorDict ForwardOnlineUpTo(const anet::TensorDict& obs, const std::string& branch_key,
            anet::nn::NetworkForwardCapture* capture = nullptr) const;

        // Network取得
        std::shared_ptr<anet::nn::Network> GetOnlineNetwork() { return online_net_; }
        std::shared_ptr<anet::nn::Network> GetTargetNetwork() { return target_net_; }
        std::shared_ptr<anet::nn::Network> GetSkipOnlineNetwork() { return skip_online_net_; }
        std::shared_ptr<anet::nn::Network> GetSkipTargetNetwork() { return skip_target_net_; }
        const SkipConfig& GetSkipConfig() const { return skip_config_; }

        /// 分布型Head判定
        bool IsDistributional() const;

        /// online_netのパラメータ取得
        std::vector<torch::Tensor> GetOnlineParameters() const;

        torch::OrderedDict<std::string, torch::Tensor> GetOnlineNamedParameters() const;

        /// target network 同期
        void UpdateTarget(anet::rl::step_t learn_step, float tau);
        const anet::ProfiledValueConfig<float>& GetSoftUpdateTauConfig() const { return config_.soft_update_tau; }
        int GetHardUpdateInterval() const { return config_.hard_update_interval; }

        /// メトリクス用：TensorDict
        std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key, const torch::Device& device);
    public:
        int64_t Save(OutputArchive& archive) const override;
        int64_t Load(InputArchive& archive) override;
        int64_t LoadCheckpoint(InputArchive& archive, bool has_skip);
    private:
        void ValidateSpectralNormSoftUpdateConfig() const;
        void SoftUpdate(float tau);
        void HardUpdate();
    private:
        const anet::rl::dqn::NetworkModelConfig config_;
        std::shared_ptr<anet::nn::Network> online_net_;
        std::shared_ptr<anet::nn::Network> target_net_;
        SkipConfig skip_config_;
        std::shared_ptr<anet::nn::Network> skip_online_net_;
        std::shared_ptr<anet::nn::Network> skip_target_net_;

        int64_t n_actions_;
        bool distributional_;
    };


    // ======================================================
    // ActionPolicy
    // ======================================================

    struct TrainActorSnapshotMetrics {
        float interval = std::numeric_limits<float>::quiet_NaN();
        float age = std::numeric_limits<float>::quiet_NaN();
    };

    class DQNActionInfo final : public anet::rl::BatchActionInfo {
    public:
        static constexpr const char* kSkipDecision = "skip_decision";
        static constexpr const char* kSkipLength = "skip_length";
        static constexpr const char* kSkipQValues = "skip_q_values";
        static constexpr const char* kSkipDecisionRate = "skip_decision_rate";
        static constexpr const char* kSkipLengthMean = "skip_length_mean";
        static constexpr const char* kSkipQGap = "skip_q_gap";
        static constexpr const char* kSkipLengthRatePrefix = "skip_length_rate.";
        static constexpr const char* kSkipLengthRateIndexPrefix = "skip_length_rate.[";
        static constexpr const char* kSkipQPrefix = "skip_q.";
        static constexpr const char* kSkipQIndexPrefix = "skip_q.[";
        static constexpr const char* kIntrinsicGamma = "intrinsic_gamma";

        using anet::rl::BatchActionInfo::BatchActionInfo;
        anet::TensorDict forward_output; ///< Actor が skip 入力を組むまで保持する一時出力。
        int64_t risk_num_taus = 0;
        torch::Tensor selection_taus; ///< 同じforwardの別の価値へ方策を当て直すときに使う抽選済みτ。
        float effective_gamma = std::numeric_limits<float>::quiet_NaN();
        float effective_intrinsic_gamma = std::numeric_limits<float>::quiet_NaN();
        float munchausen_alpha = std::numeric_limits<float>::quiet_NaN();
        float munchausen_entropy_tau = std::numeric_limits<float>::quiet_NaN();

        std::shared_ptr<anet::rl::BatchActionInfo> To(torch::Device device) const override;
        std::shared_ptr<anet::rl::BatchActionInfo> WithAction(torch::Tensor action) const override;
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        void SetTrainActorSnapshotMetrics(TrainActorSnapshotMetrics metrics)
        {
            train_actor_snapshot_metrics_ = metrics;
        }
    private:
        std::optional<TrainActorSnapshotMetrics> train_actor_snapshot_metrics_;
        mutable torch::Tensor iqn_policy_diagnostics_cpu_;
        mutable torch::Tensor quantile_tail_diagnostics_cpu_;
        mutable torch::Tensor skip_diagnostics_cpu_;
        mutable torch::Tensor value_selection_diagnostics_cpu_;
    };

    struct DiagnosticActionEvaluation {
        int64_t num_taus;
    };

    /// target計算で使うonline用とtarget用の実行器。
    struct TargetForwards {
        std::shared_ptr<anet::nn::NetworkForwardExecutor> online; ///< Double DQNの行動選択とMunchausenのonline方式
        std::shared_ptr<anet::nn::NetworkForwardExecutor> target; ///< 価値の評価とMunchausenのtarget方式
    };

    struct TargetEvaluation {
        bool diagnostic = false;
        anet::nn::NetworkForwardCallInfo info{{}};
        const TargetForwards* forwards = nullptr; ///< 選択された用途ペア。同期target計算中だけ参照する。
        anet::nn::NetworkForwardCapture* target_capture = nullptr; ///< 通常更新のtarget取得要求。診断ではnullptr。
        std::function<torch::Tensor(int64_t)> current_taus;
        std::function<torch::Tensor(int64_t)> target_taus;
    };

    struct ElementErrorRequest {
        bool td = true;
        bool loss = true;
        bool auxiliary = false;
    };

    struct ElementError {
        torch::Tensor td; ///< 符号付き期待TD。診断の絶対値化は集約側で行う。
        torch::Tensor loss;
        torch::Tensor pair_abs_td;
        torch::Tensor cancellation_ratio;
    };

    struct MunchausenForwardResult {
        torch::Tensor current_values;
        torch::Tensor next_values;
    };

    struct CurrentForwardResult {
        anet::TensorDict input;
        anet::TensorDict output;
        torch::Tensor all_values;
        torch::Tensor action_values;
        torch::Tensor taus;
        torch::Tensor current_mean;
        ElementErrorRequest error_request;
        QuantileMetrics metrics;
        torch::Tensor upper_tail_std;
        torch::Tensor intrinsic_action_values;
    };

    struct RiskScoreSpec {
        float tau;
        bool use_tail_mean;
    };

    class SpatialLadderSchedule {
    public:
        SpatialLadderSchedule(const ActionPolicyConfig::LadderConfig& config,
            std::string scale, int64_t lanes, torch::Device device);
        void Update(step_t step);
        const torch::Tensor& Tensor() const { return tensor_; }
        float Mean() const { return mean_; }
        float Min() const { return minimum_; }
        float Max() const { return maximum_; }
    private:
        anet::ProfiledValue<float> first_, last_;
        std::string scale_;
        int64_t lanes_;
        torch::Device device_;
        torch::Tensor tensor_;
        float current_first_ = std::numeric_limits<float>::quiet_NaN();
        float current_last_ = std::numeric_limits<float>::quiet_NaN();
        float mean_ = 0.0f, minimum_ = 0.0f, maximum_ = 0.0f;
    };

    class DecomposedValueCombiner {
    public:
        static void Combine(anet::TensorDict& output, float beta, bool use_tbo, float epsilon);
    };

    /// DQN側だけで結合する同期呼び出し。NNの出力契約へβを持ち込まない。
    class DqnForwardExecutor {
    public:
        DqnForwardExecutor(anet::nn::NetworkForwardExecutor& forward,
            std::optional<float> beta = std::nullopt, bool use_tbo = false, float epsilon = 0.01f)
            : forward_(forward), beta_(beta), use_tbo_(use_tbo), epsilon_(epsilon) {}
        anet::TensorDict Forward(const anet::TensorDict& input, const anet::nn::NetworkForwardCallInfo& info,
            anet::nn::NetworkForwardCapture* capture = nullptr) const;
    private:
        anet::nn::NetworkForwardExecutor& forward_; ///< 同期呼び出しの間、ActorまたはLearner所有の実行器を借りる。
        std::optional<float> beta_;
        bool use_tbo_;
        float epsilon_;
    };

    class ActionPolicy : virtual public anet::ModuleBase {
    public:
        ActionPolicy(const ActionPolicyConfig& config,
        	bool enable_spatial_exploration = false, int64_t num_envs = 0,
            const torch::Device& device = torch::Device(torch::kCPU));

        virtual std::shared_ptr<DQNActionInfo> SelectAction(const anet::TensorDict& obs, bool greedy_only,
            const DqnForwardExecutor& forward, std::shared_ptr<anet::RandomGenerator> rnd, const anet::nn::NetworkForwardCallInfo& info,
            anet::nn::NetworkForwardCapture* capture = nullptr,
            const DiagnosticActionEvaluation* evaluation = nullptr, const torch::Tensor& additional_taus = {}) const = 0;
        virtual void UpdateSchedule(const StepCounts& counts) { }
        virtual std::optional<RiskScoreSpec> GetRiskScoreSpec() const { return std::nullopt; }
        /// 同じforwardと抽選済みτでscoreを作り、乱数と探索を追加しない。
        virtual torch::Tensor MakeActionScores(const torch::Tensor& q, const torch::Tensor& distribution,
            bool iqn_risk_query, const torch::Tensor& selection_taus) const;
        virtual torch::Tensor SelectFromValues(const torch::Tensor& q, const torch::Tensor& distribution,
            std::shared_ptr<anet::RandomGenerator> random, bool iqn_risk_query = true,
            bool greedy_only = false, const torch::Tensor& selection_taus = {}) const;
        virtual torch::Tensor MakeValueQueryTaus(int64_t batch, const torch::Device& device, anet::RandomGenerator& random) const;

        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;

        const ActionPolicyConfig& GetConfig() const { return config_; }
        virtual ~ActionPolicy() = default;
    protected:
        anet::TensorDict ForwardForAction(const anet::TensorDict& obs, const DqnForwardExecutor& forward, const anet::nn::NetworkForwardCallInfo& info, anet::nn::NetworkForwardCapture* capture) const;
        anet::TensorDict ForwardForActionWithTaus(
            const anet::TensorDict& obs, const torch::Tensor& taus,
            const DqnForwardExecutor& forward, const anet::nn::NetworkForwardCallInfo& info, anet::nn::NetworkForwardCapture* capture,
            const torch::Tensor& additional_taus = {}) const;
        torch::Tensor MakeEpsilonGreedyAction(const torch::Tensor& greedy_action, float epsilon, int64_t num_envs, int64_t n_actions, std::shared_ptr<anet::RandomGenerator> rnd) const;
        torch::Tensor MakeEpsilonGreedyAction(const torch::Tensor& greedy_action, const torch::Tensor& epsilon_tensor, int64_t num_envs, int64_t n_actions, std::shared_ptr<anet::RandomGenerator> rnd) const;
        std::shared_ptr<DQNActionInfo> MakeActionInfo(const torch::Tensor& action_values, const torch::Tensor& q_values, const torch::Tensor& q_quantiles) const;
        //torch::Tensor GetQuantiles(const torch::Tensor& obs, bool use_target) const;
        void UpdateEpsilon(step_t step, bool is_uqe = false);
        bool IsSpatialExplorationEnabled() const { return spatial_enabled_; }
        torch::Tensor GetSpatialEpsilonTensor(int64_t num_envs, const torch::Device& device, bool is_uqe) const;
        torch::Tensor GetSpatialTauTensor(int64_t num_envs, const torch::Device& device) const;
    protected:
        const ActionPolicyConfig config_;
        bool spatial_enabled_ = false;
        bool uses_uqe_ = false;
        std::optional<anet::ProfiledValue<float>> epsilon_schedule_, tau_schedule_;
        bool thompson_ = false;
        std::unique_ptr<SpatialLadderSchedule> epsilon_ladder_, uqe_epsilon_ladder_, tau_ladder_;
        int64_t spatial_num_envs_ = 0;
        torch::Tensor spatial_eps_tensor_;
        torch::Tensor spatial_uqe_eps_tensor_;
        torch::Tensor spatial_tau_tensor_;
        float current_epsilon_ = 0.0f;
        float current_uqe_tau_ = 0.0f;
    };

    class EpsilonGreedyActionPolicy final : public ActionPolicy {
    public:
        EpsilonGreedyActionPolicy(const ActionPolicyConfig& config,
            bool enable_spatial_exploration = false,
            int64_t num_envs = 0,
            const torch::Device& device = torch::Device(torch::kCPU));

        std::shared_ptr<DQNActionInfo> SelectAction(const anet::TensorDict& obs, bool greedy_only,
            const DqnForwardExecutor& forward, std::shared_ptr<anet::RandomGenerator> rnd, const anet::nn::NetworkForwardCallInfo& info,
            anet::nn::NetworkForwardCapture* capture,
            const DiagnosticActionEvaluation* evaluation = nullptr, const torch::Tensor& additional_taus = {}) const override;
        void UpdateSchedule(const StepCounts& counts) override;
    };

    /**
     * UQE (Upper Quantile Exploration)
     * 分布型RLにおける楽観的探索 (Optimistic Exploration) を行うクラス。
     * 分布の右裾（上位Quantile）の情報を用いて、不確実性の高い行動を積極的に探索する。
     * * uqe.use_tail_meanにより、以下2つの標準的な手法に対応：
     * 1. Q-UCB: 特定の上位分位点(tau)の値を使用 (falseの場合)
     * 2. Upper CVaR: 上位tauから1.0までの平均値を使用 (trueの場合)
     */
    class UQEActionPolicy : public ActionPolicy {
    public:
        UQEActionPolicy(const ActionPolicyConfig& config,
            bool enable_spatial_exploration = false,
            int64_t num_envs = 0,
            const torch::Device& device = torch::Device(torch::kCPU));

        std::shared_ptr<DQNActionInfo> SelectAction(const anet::TensorDict& obs, bool greedy_only,
            const DqnForwardExecutor& forward, std::shared_ptr<anet::RandomGenerator> rnd, const anet::nn::NetworkForwardCallInfo& info,
            anet::nn::NetworkForwardCapture* capture,
            const DiagnosticActionEvaluation* evaluation = nullptr, const torch::Tensor& additional_taus = {}) const override;
        void UpdateSchedule(const StepCounts& counts) override;
        torch::Tensor MakeActionScores(const torch::Tensor& q, const torch::Tensor& distribution,
            bool iqn_risk_query, const torch::Tensor& selection_taus) const override;
        torch::Tensor SelectFromValues(const torch::Tensor& q, const torch::Tensor& distribution,
            std::shared_ptr<anet::RandomGenerator> random, bool iqn_risk_query = true,
            bool greedy_only = false, const torch::Tensor& selection_taus = {}) const override;
        torch::Tensor MakeValueQueryTaus(int64_t batch, const torch::Device& device, anet::RandomGenerator& random) const override;

        virtual ~UQEActionPolicy() = default;
        std::optional<RiskScoreSpec> GetRiskScoreSpec() const override
        {
            return RiskScoreSpec{ .tau = current_uqe_tau_, .use_tail_mean = config_.uqe.use_tail_mean };
        }
    protected:
        std::shared_ptr<DQNActionInfo> MakeUQEActionInfo(float tau, const torch::Tensor& tau_tensor, const anet::TensorDict& obs, bool greedy_only,
            const DqnForwardExecutor& forward, std::shared_ptr<anet::RandomGenerator> rnd, const anet::nn::NetworkForwardCallInfo& info, anet::nn::NetworkForwardCapture* capture,
            bool iqn_use_full_range = false, const DiagnosticActionEvaluation* evaluation = nullptr,
            const torch::Tensor& additional_taus = {}) const;
        void UpdateTau(step_t step);
    private:
        torch::Tensor MakeUQEValues(float tau, const torch::Tensor& q_quantiles) const;
        torch::Tensor MakeVectorizedUQEValues(const torch::Tensor& tau_tensor, const torch::Tensor& q_quantiles) const;
    };

    class ThompsonSamplingActionPolicy final : public UQEActionPolicy {
    public:
        std::optional<RiskScoreSpec> GetRiskScoreSpec() const override { return std::nullopt; }
        ThompsonSamplingActionPolicy(const ActionPolicyConfig& config,
            bool enable_spatial_exploration = false,
            int64_t num_envs = 0,
            const torch::Device& device = torch::Device(torch::kCPU));

        std::shared_ptr<DQNActionInfo> SelectAction(const anet::TensorDict& obs, bool greedy_only,
            const DqnForwardExecutor& forward, std::shared_ptr<anet::RandomGenerator> rnd, const anet::nn::NetworkForwardCallInfo& info,
            anet::nn::NetworkForwardCapture* capture,
            const DiagnosticActionEvaluation* evaluation = nullptr, const torch::Tensor& additional_taus = {}) const override;
        void UpdateSchedule(const StepCounts& counts) override;
    };


    // ======================================================
    // Actor
    // ======================================================

    struct ActorQHintConfig {
        MunchausenConfig munchausen;
        bool use_tbo = false;
        float tbo_epsilon = 0.01f;
        anet::ProfiledValueConfig<float> gamma{ .value = 0.99f, .min_value = 0.0f, .max_value = 1.0f };
        bool decomposed = false;
        anet::ProfiledValueConfig<float> beta{.value = 0.1f, .min_value = 0.0f};
        bool extrinsic_bootstrap = false;
        anet::ProfiledValueConfig<float> intrinsic_gamma{.value = 0.99f, .min_value = 0.0f, .max_value = 1.0f};
    };

    struct ActorQHintValues {
        MunchausenValues munchausen;
        bool use_tbo = false;
        float tbo_epsilon = 0.01f;
        torch::Tensor extrinsic_q;
        torch::Tensor intrinsic_q;
        torch::Tensor extrinsic_scores;
        torch::Tensor combined_scores;
        bool extrinsic_bootstrap = false;
    };

    struct ActorQHintData {
        torch::Tensor payload;
        torch::Tensor munchausen_terms;
    };
    ActorQHintData MakeActorQHint(const torch::Tensor& q_values,
        const torch::Tensor& action, const ActorQHintValues& config);
    std::unique_ptr<AgentStepDiagnosticsProvider> CreateStepDiagnosticsProvider(const LearnerConfig& config,
        int64_t num_lanes, std::function<torch::Tensor(const torch::Tensor&)> scale_reward, bool needs_step_priority = true);

    anet::TensorDict MakeSkipInput(const anet::TensorDict& obs, const anet::TensorDict& primary,
        const torch::Tensor& actions, bool stop_gradient);
    anet::TensorDict ReadSkipValues(const anet::TensorDict& output, const anet::TensorDict& primary,
        const torch::Tensor& actions, const SkipConfig& config);

    struct SkipActorResources {
        SkipConfig config;
        std::shared_ptr<ActionPolicy> policy;
        std::shared_ptr<anet::nn::Network> network;
        std::shared_ptr<anet::nn::Network> source;
        std::shared_ptr<const anet::nn::NetworkForwardConfig> forward_config;
        bool use_amp = false;
        bool use_amp_bf16 = true;
        bool shared_policy = true;
        std::string max_length_mode = DQNActorConfig::kMaxLengthFull;
        std::optional<anet::ProfiledValueConfig<int>> max_length;
    };

    class Actor : public anet::rl::Actor, public anet::RandomHolder {
    public:
        Actor(std::shared_ptr<ActionPolicy> policy,
            std::shared_ptr<anet::rl::ObservationNormalizer> obs_norm,
            std::unique_ptr<FrameStacker> stacker,
            torch::Device device,
            std::optional<seed_t> seed,
            std::shared_ptr<std::shared_mutex> mutex,
            std::shared_ptr<anet::nn::Network> network,
            std::shared_ptr<anet::nn::Network> src_network,
            const anet::nn::NetworkForwardConfig& forward_config,
            bool emit_actor_q_hint = false,
            std::optional<anet::ProfiledValueConfig<step_t>> snapshot_sync_interval = std::nullopt,
            bool emit_snapshot_metrics = false,
            ActorQHintConfig actor_q_hint_config = {},
            std::optional<SkipActorResources> skip = std::nullopt);
        std::shared_ptr<BatchActionInfo> MakeAction(const StepCounts& step, const anet::rl::BatchState& state, const anet::rl::MakeActionOptions& options = {}) const override;
        void Sync() override;
        bool AttachSyncFollower(std::weak_ptr<ActorSyncFollower> follower) override;
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
    private:
        void ResetSessionDiagnostics() const;
        void CopySourceNetwork() const;
        void UpdateSnapshot(const StepCounts& step) const;
        std::shared_ptr<DQNActionInfo> ApplySkip(std::shared_ptr<DQNActionInfo> proposal,
            const anet::TensorDict& obs, const torch::Tensor& episode_start,
            const anet::nn::NetworkForwardCallInfo& info, int64_t dedicated_tau_count) const;
    private:
        std::shared_ptr<ActionPolicy> policy_;
        std::shared_ptr<anet::rl::ObservationNormalizer> obs_norm_;
        std::unique_ptr<FrameStacker> stacker_;
        torch::Device device_;
        std::shared_ptr<std::shared_mutex> mutex_;
        std::shared_ptr<anet::nn::Network> network_;
        std::shared_ptr<anet::nn::Network> src_network_;
        std::shared_ptr<anet::nn::NetworkForwardExecutor> actor_forward_;
        mutable std::vector<std::weak_ptr<ActorSyncFollower>> sync_followers_;
        mutable uint64_t forward_clock_ = 0; ///< 行動選択ごとの時計。episodeやsnapshot同期でリセットしない。
        bool emit_actor_q_hint_ = false; ///< 学習Actorから初期優先度推定用Qヒントを生成するか
        const ActorQHintConfig actor_q_hint_config_;
        anet::ProfiledValue<float> gamma_;
        anet::ProfiledValue<float> intrinsic_gamma_;
        anet::ProfiledValue<float> intrinsic_beta_;
        anet::ProfiledValue<float> munchausen_alpha_, munchausen_entropy_tau_;
        mutable std::optional<anet::ProfiledValue<step_t>> snapshot_sync_interval_;
        mutable step_t last_snapshot_sync_train_step_ = 0;
        mutable bool reset_snapshot_age_on_next_action_ = false;
        bool emit_snapshot_metrics_ = false;
        const std::optional<SkipActorResources> skip_;
        std::shared_ptr<anet::nn::NetworkForwardExecutor> skip_forward_;
        std::shared_ptr<anet::RandomGenerator> skip_random_;
        mutable torch::Tensor skip_remaining_;
        mutable std::optional<anet::ProfiledValue<int>> skip_max_length_;
        mutable torch::Tensor skip_session_counts_; ///< 長さ別の判断回数。Actorと同じdeviceで加算する。
        mutable torch::Tensor skip_session_counts_cpu_;
        mutable int64_t skip_session_lane_steps_ = 0;
        mutable torch::Tensor value_selection_session_sums_;
        mutable torch::Tensor value_selection_session_cpu_;
        mutable int64_t value_selection_session_lane_steps_ = 0;
        mutable torch::Tensor skip_actions_;
        mutable torch::Tensor skip_lengths_;
    };

    // ======================================================
    // Learner
    // ======================================================

    struct LearnerForwardResources {
        const anet::nn::NetworkForwardConfig current = anet::nn::NetworkForwardConfig::GetDefault();
        const anet::nn::NetworkForwardConfig target = anet::nn::NetworkForwardConfig::GetDefault();
    };

    class Learner : public anet::rl::Learner, public anet::Module, public anet::Serializable, public anet::RandomHolder {
    public:
        Learner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec,
            torch::Device device, anet::seed_t replay_seed, anet::seed_t agent_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            const LearnerForwardResources& forward_resources,
            std::optional<StuckerConfig> stucker_config = std::nullopt,
            std::optional<anet::seed_t> target_seed = std::nullopt,
            std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe_config = nullptr);

        BatchUpdateResultList UpdateFromBatch(const StepCounts& step, const BatchExperience& expriences) override;
        BatchUpdateResultList UpdateFromBatch(const StepCounts& step, const BatchExperience& experiences, float beta);
        void ConfigureScalarMetricSubscriptions(const std::vector<ScalarMetricSubscription>& subscriptions);

        virtual ~Learner() = default;
    public:
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override;
    public:
        int64_t Save(OutputArchive& archive) const override;
        int64_t Load(InputArchive& archive) override;
        int64_t LoadCheckpoint(InputArchive& archive, bool has_skip);
    public:
        // 1 batch の進行を共有し、既存のテスト用派生は全体を差し替えられる。
        virtual std::shared_ptr<anet::rl::dqn::BatchUpdateResult> UpdateFromSamples(
            const anet::rl::ExperienceSamples& samples);
    protected:
        virtual CurrentForwardResult ForwardCurrent(const ExperienceSamples& samples,
            const anet::TensorDict& obs, const anet::nn::NetworkForwardCallInfo& info);
        virtual torch::Tensor MakeUpdateDiagnostics(const ExperienceSamples& samples,
            const CurrentForwardResult& current, const torch::Tensor& target, const ElementError& error) const;
        void SetupOptimizer();                  ///< 共通初期化処理（Optimizer生成など）
        void SetupReplayBuffer(const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec, anet::seed_t seed);
        anet::nn::NetworkForwardCapture* OnlineBranchCapture();
        anet::nn::NetworkForwardCapture* TargetBranchCapture();
        TargetEvaluation MakeTargetEvaluation(bool diagnostic);
        LearningTargets MakeLearningTargets(const ExperienceSamples& samples, const anet::TensorDict& obs,
            const anet::TensorDict& next_obs, const torch::Tensor& current_all, const TargetEvaluation& evaluation);
        LearningTargets MakeDecomposedTargets(const ExperienceSamples& samples, const anet::TensorDict& obs,
            const anet::TensorDict& next_obs, const CurrentForwardResult& current, const TargetEvaluation& evaluation);
        SkipLearningResult ComputeSkipLoss(const ExperienceSamples& samples, const anet::TensorDict& input,
            const anet::TensorDict& primary, const torch::Tensor& target,
            const anet::nn::NetworkForwardCallInfo& info, const torch::Tensor& taus = {});
        torch::Tensor CombinePriorityErrors(const torch::Tensor& primary, const SkipLearningResult& skip) const;
        anet::nn::NetworkForwardCallInfo MakeCurrentForwardInfo() const;
        virtual std::pair<torch::Tensor, torch::Tensor> MakeTarget(
            const ExperienceSamples& samples, const anet::TensorDict& obs,
            const anet::TensorDict& next_obs, const torch::Tensor& current_all,
            const TargetEvaluation& evaluation);
        MunchausenForwardResult ForwardMunchausenValues(const ExperienceSamples& samples,
            const anet::TensorDict& obs, const anet::TensorDict& next_obs,
            const torch::Tensor& current_all, const TargetEvaluation& evaluation,
            const std::string& output_key, bool use_taus);
        virtual ElementError ComputeElementError(const torch::Tensor& current,
            const torch::Tensor& target, const torch::Tensor& taus, const ElementErrorRequest& request,
            const torch::Tensor& current_mean = {}) const;
        NormalizedSampleObservations NormalizeSampleObservations(const anet::rl::ExperienceSamples& samples) const;
        OptimizerStepResult Optimize(const torch::Tensor& primary_loss, const torch::Tensor& weighted_element_loss,
            const torch::Tensor& skip_loss = {}, const torch::Tensor& intrinsic_loss = {});
    protected:
        PerPriorityUpdatePending PreparePerPriorityUpdate(
            const anet::rl::ExperienceSamples& samples,
            const torch::Tensor& td_error,
            const torch::Tensor& iqn_diagnostics = torch::Tensor(),
            const torch::Tensor& upper_tail_std = torch::Tensor(),
            const torch::Tensor& munchausen_diagnostics = torch::Tensor());
        PerPriorityUpdateInfo ApplyPerPriorityUpdate(PerPriorityUpdatePending pending);
        PerPriorityUpdateInfo UpdatePerPriorities(const anet::rl::ExperienceSamples& samples, const torch::Tensor& td_error);
    protected:
        torch::Tensor TransformH(const torch::Tensor& x) const;
        torch::Tensor TransformHInv(const torch::Tensor& x) const;
        torch::Tensor MakeBellmanTarget(const ExperienceSamples& samples,
            const torch::Tensor& bootstrap_real, const torch::Tensor& bonus = {}, std::optional<float> gamma = std::nullopt) const;
        std::shared_ptr<anet::rl::dqn::BatchUpdateResult> MakeBatchUpdateResult(
            const torch::Tensor& loss,
            const torch::Tensor& td_error,
            const OptimizerStepResult& opt_result,
            const torch::Tensor& max_q,
            const torch::Tensor& q_sa,
            const PerPriorityUpdateInfo& per_info,
            const torch::Tensor& q_std = torch::Tensor(),
            const torch::Tensor& q_gap = torch::Tensor(),
            const torch::Tensor& q_gap_rel = torch::Tensor(), const SkipLearningResult& skip = {}) const;
    private:
        void DistributeDiagnosticSubscriptions(const std::vector<ScalarMetricSubscription>& subscriptions);
        void ValidateDiagnosticSubscriptions();
        void PrepareDiagnosticResources();
        std::vector<torch::optim::AdamWOptions> BuildOptimizerOptions() const;
        std::vector<std::string> OptimizerRoles() const;
        bool CanUpdate(step_t exp_step);
        void UpdatePerBeta(step_t step);
        void UpdateTargetNetwork(step_t step);
        void ValidateDeviceSamples(const anet::rl::ExperienceSamples& samples, int64_t batch_size) const;
        torch::Tensor CaptureAttentionEntropy();
        void CaptureNoisyDiagnostics();
        std::optional<GradientNoiseMetrics> CaptureGradientNoise(const torch::Tensor& weighted_element_loss);
        void CapturePlasticityProbe();
        void PreparePolicyChurnRequest();
        void CapturePolicyChurnProbe();
        void CapturePolicyChurnOnlineBefore();
        void CapturePolicyChurnOnlineAfter();
        void FinalizePolicyChurn(BatchUpdateResult& result);
        torch::Tensor ForwardPolicyChurnExpectedQ(bool target) const;
        std::optional<ReplayFitMetrics> CaptureReplayFit(const ExperienceSamples& samples);
        std::pair<float, float> EvaluateReplayFit(const ExperienceSamples& samples, bool td, bool loss);
    protected:
        const torch::Device device_;
        int num_envs_;
        int n_actions_;
        float earned_credit_;
        LearnerConfig config_;
        anet::ProfiledValue<float> learning_rate_;
        anet::ProfiledValue<float> weight_decay_;
        anet::ProfiledValue<float> per_beta_;
        anet::ProfiledValue<float> soft_update_tau_;
        anet::ProfiledValue<float> gamma_;
        anet::ProfiledValue<float> intrinsic_gamma_;
        float intrinsic_beta_ = 0.0f;
        torch::Tensor decomposition_metrics_;
        mutable torch::Tensor decomposition_metrics_cpu_;
        std::vector<float> ComponentGammas() const;
        anet::ProfiledValue<float> munchausen_alpha_, munchausen_entropy_tau_;
        MunchausenValues munchausen_values_;
        const bool has_optimizer_schedule_;
        std::vector<bool> optimizer_sigma_groups_;
        std::shared_ptr<anet::nn::NetworkForwardExecutor> current_forward_;
        std::shared_ptr<anet::nn::NetworkForwardExecutor> skip_current_forward_;
        TargetForwards target_forwards_;
        TargetForwards diagnostic_forwards_;
        std::optional<StuckerConfig> stucker_config_;
        std::shared_ptr<ActionPolicy> target_policy_;
        NetworkModel& model_;
        RuntimeVars& vars_;
        std::shared_ptr<ObservationNormalizer> obs_norm_;
        std::shared_ptr<anet::rl::ReplayBuffer> replay_buffer_;
        std::unique_ptr<torch::optim::Optimizer> optimizer_;
        std::unique_ptr<torch::optim::Optimizer> intrinsic_optimizer_;
        std::array<std::vector<torch::Tensor>, 2> value_head_parameters_;
        std::unique_ptr<torch::optim::Optimizer> skip_primary_optimizer_; ///< primaryへ流れるskip勾配だけの独立したモーメント。
        anet::GradScaler grad_scaler_;
        anet::RandomGenerator plasticity_probe_random_;
        anet::RandomGenerator policy_churn_probe_random_;
        anet::RandomGenerator replay_fit_probe_random_;
        anet::RandomGenerator noisy_probe_random_;
        anet::RandomGenerator attention_entropy_probe_random_;
        std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe_config_;
        std::optional<at::cuda::CUDAStream> per_priority_copy_stream_;
        anet::transfer::EventRecycler<torch::Tensor> per_priority_event_recycler_;
        struct GradientNoiseState {
            DiagnosticDemandTable<kGradientNoiseMetricKeys.size()> demand{kGradientNoiseMetricKeys};
            std::array<bool, kGradientNoiseMetricKeys.size()> request{};
            anet::GradientNoiseRequest subscribed;
            std::array<anet::GradientNoiseEstimator, 4> estimators;
            std::vector<anet::GradientNoiseParameter> parameters;
        } grad_noise_;
        struct NoisyDiagnosticState {
            DiagnosticDemandTable<kNoisyMetricKeys.size()> demand{kNoisyMetricKeys};
            std::shared_ptr<anet::nn::NetworkForwardExecutor> sample_forward;
            uint64_t forward_clock = 0;
            NoisyMetrics values;
            NoisyDiagnosticState() { values.fill(std::numeric_limits<float>::quiet_NaN()); }
        } noisy_;
        struct AttentionEntropyState {
            struct DemandEntry {
                int64_t layer;
                anet::IntervalGate gate;
                DemandEntry(int64_t layer, int interval) : layer(layer), gate(static_cast<uint64_t>(interval)) {}
            };
            std::vector<DemandEntry> demand;
        } attention_;
        struct PlasticityState {
            struct DemandEntry {
                anet::nn::PlasticityMetric metric;
                anet::IntervalGate gate;

                DemandEntry(anet::nn::PlasticityMetric metric, int interval)
                    : metric(metric), gate(static_cast<uint64_t>(interval)) {}
            };

            struct ChannelDemand {
                bool enabled = false;
                int interval = 1;
                std::vector<DemandEntry> entries;

                void Add(anet::nn::PlasticityMetric metric, int candidate_interval) {
                    interval = enabled ? std::min(interval, candidate_interval) : candidate_interval;
                    enabled = true;
                    entries.emplace_back(metric, candidate_interval);
                }

                anet::nn::PlasticityMetricRequest MakeRequest(uint64_t step) {
                    anet::nn::PlasticityMetricRequest request;
                    if (!enabled || step % static_cast<uint64_t>(interval) != 0) return request;
                    for (auto& entry : entries) {
                        if (entry.gate.ShouldFire(step)) request.Add(entry.metric);
                    }
                    return request;
                }
            };

            ChannelDemand online;
            ChannelDemand target;
            ChannelDemand probe;
            bool weight_norm_enabled = false;
            int weight_norm_interval = 1;
            anet::nn::NetworkForwardCapture online_capture;
            anet::nn::NetworkForwardCapture target_capture;
            std::optional<int64_t> target_capture_batch_size; ///< joint targetの後半だけを結果へ載せる。
            int64_t target_capture_offset = 0;
            anet::nn::PlasticityMetricRequest online_request;
            anet::nn::PlasticityMetricRequest target_request;
            anet::nn::PlasticityMetricRequest probe_request;
            torch::Tensor probe_features;
            torch::Tensor weight_norms;
            mutable std::optional<anet::nn::PlasticityMetrics> probe_metrics_cache;
        } plasticity_;
        struct PolicyChurnState {
            DiagnosticDemandTable<kPolicyChurnMetricKeys.size()> demand{kPolicyChurnMetricKeys};
            std::array<bool, kPolicyChurnMetricKeys.size()> request{};
            anet::TensorDict probe_obs;
            torch::Tensor fixed_taus;
            torch::Tensor online_before;
            torch::Tensor online_after;

            bool AnyQRequest() const
            {
                for (size_t i = 0; i < PolicyChurnMetric::TargetSyncAge; ++i) {
                    if (request[i]) return true;
                }
                return false;
            }

            bool NeedsOnlineBefore() const
            {
                return request[PolicyChurnMetric::ActionRatio] || request[PolicyChurnMetric::QDeltaAbsMean] || request[PolicyChurnMetric::QDeltaSignedMax] || request[PolicyChurnMetric::QDeltaSignedMin];
            }

            bool NeedsOnlineAfter() const
            {
                return AnyQRequest();
            }

            bool NeedsTargetAfter() const
            {
                return request[PolicyChurnMetric::TargetPolicyDisagreement] || request[PolicyChurnMetric::TargetQDeltaAbsMean];
            }
        } policy_churn_;
        DiagnosticDemandTable<kReplayFitMetricKeys.size()> replay_fit_demand_{kReplayFitMetricKeys};
    protected:
        float update_credit_ = 0.0f;
    private:
        bool warmup_active_ = false; ///< 開始を通知したウォームアップの終了を一度だけ通知する。
        bool has_enough_replay_samples_ = false;
    };

    class QuantileLearnerBase : public anet::rl::dqn::Learner {
    public:
        explicit QuantileLearnerBase(
        	const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed, anet::seed_t agent_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            const LearnerForwardResources& forward_resources,
            std::optional<StuckerConfig> stucker_config = std::nullopt,
            std::optional<anet::seed_t> target_seed = std::nullopt,
            std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe_config = nullptr);

        virtual ~QuantileLearnerBase() = default;
    protected:
        CurrentForwardResult ForwardCurrent(const ExperienceSamples& samples,
            const anet::TensorDict& obs, const anet::nn::NetworkForwardCallInfo& info) override;
        CurrentForwardResult ForwardCurrentQuantiles(const ExperienceSamples& samples,
            const anet::TensorDict& input, const anet::nn::NetworkForwardCallInfo& info,
            const torch::Tensor& current_taus);
        torch::Tensor GatherActionQuantiles(const torch::Tensor& quantiles, const torch::Tensor& actions) const;
        torch::Tensor SelectTargetActions(const anet::TensorDict& next_obs,
            const TargetEvaluation& evaluation);
        std::pair<torch::Tensor, torch::Tensor> MakeTarget(
            const ExperienceSamples& samples, const anet::TensorDict& obs,
            const anet::TensorDict& next_obs, const torch::Tensor& current_all,
            const TargetEvaluation& evaluation) override;
        ElementError ComputeElementError(const torch::Tensor& current,
            const torch::Tensor& target, const torch::Tensor& taus, const ElementErrorRequest& request,
            const torch::Tensor& current_mean = {}) const override;
        std::pair<torch::Tensor, torch::Tensor> MakeMunchausenTarget(
            const anet::rl::ExperienceSamples& samples, const anet::TensorDict& obs,
            const anet::TensorDict& next_obs, const torch::Tensor& current_dist_all,
            const TargetEvaluation& evaluation);
        QuantileMetrics MakeQuantileMetrics(const torch::Tensor& current_dist, const torch::Tensor& q_values_mean) const;
        torch::Tensor ComputeQuantileHuberLoss(
        	const torch::Tensor& current_dist, const torch::Tensor& target_dist, const torch::Tensor& taus) const;
        static torch::Tensor ComputeQuantileHuberLoss(
            const torch::Tensor& current_dist, const torch::Tensor& target_dist, const torch::Tensor& taus, float kappa);
        static IqnLossResult ComputeIqnQuantileHuberLoss(
            const torch::Tensor& current_dist, const torch::Tensor& target_dist, const torch::Tensor& taus, float kappa,
            bool auxiliary = true);
    };

    class TDLearner final : public anet::rl::dqn::Learner {
    public:
        explicit TDLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed, anet::seed_t agent_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            const LearnerForwardResources& forward_resources,
            std::optional<StuckerConfig> stucker_config = std::nullopt,
            std::optional<anet::seed_t> target_seed = std::nullopt,
            std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe_config = nullptr);

    };

    class QRLearner final : public anet::rl::dqn::QuantileLearnerBase {
    public:
        explicit QRLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed, anet::seed_t agent_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            const LearnerForwardResources& forward_resources,
            std::optional<StuckerConfig> stucker_config = std::nullopt,
            std::optional<anet::seed_t> target_seed = std::nullopt,
            std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe_config = nullptr);

    private:
        ElementError ComputeElementError(const torch::Tensor& current,
            const torch::Tensor& target, const torch::Tensor& taus, const ElementErrorRequest& request,
            const torch::Tensor& current_mean = {}) const override;
        torch::Tensor tau_i_; // QuantileHuberLoss 算出用
    };

    class IQNLearner final : public anet::rl::dqn::QuantileLearnerBase {
    public:
        explicit IQNLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed, anet::seed_t agent_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            const LearnerForwardResources& forward_resources,
            std::optional<StuckerConfig> stucker_config = std::nullopt,
            std::optional<anet::seed_t> target_seed = std::nullopt,
            std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe_config = nullptr);

    private:
        CurrentForwardResult ForwardCurrent(const ExperienceSamples& samples,
            const anet::TensorDict& obs, const anet::nn::NetworkForwardCallInfo& info) override;
        torch::Tensor MakeUpdateDiagnostics(const ExperienceSamples& samples,
            const CurrentForwardResult& current, const torch::Tensor& target, const ElementError& error) const override;
    };


    class DqnComponentAssembler {
    public:
        /// Agentのstep診断で要求・完了を対応付ける鍵。
        static constexpr char kStepApproxPriorityKey[] = "approx_priority";

        static std::shared_ptr<anet::nn::NetworkHeadFactory> CreateHeadFactory(
            const std::string& quantile_mode, bool dueling, int64_t num_actions, int num_quantiles,
            const anet::nn::WeightInitConfig& init,
            const anet::ConfigData& projection = anet::EmptyConfigData);
        static std::shared_ptr<ActionPolicy> CreateActionPolicy(
            const ActionPolicyConfig& config, const std::string& quantile_mode,
            bool spatial, int64_t num_envs, const torch::Device& device);
        static std::shared_ptr<Learner> CreateLearner(
            const std::string& quantile_mode, int num_quantiles, const LearnerConfig& config,
            NetworkModel& model, RuntimeVars& vars,
            const BatchEnvSpec& batch, const EnvSpec& env, const torch::Device& device,
            anet::seed_t replay_seed, anet::seed_t agent_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            std::shared_ptr<ObservationNormalizer> normalizer = nullptr,
            const LearnerForwardResources& forwards = {},
            std::optional<StuckerConfig> stacker = std::nullopt,
            std::optional<anet::seed_t> learner_seed = std::nullopt,
            std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe = nullptr,
            const std::string& log_suffix = "");
        static std::shared_ptr<Actor> CreateActor(
            const DQNActorConfig& config, const LearnerConfig& learner_config,
            const ActorRequest& request, NetworkModel& model,
            const std::shared_ptr<std::shared_mutex>& mutex, std::shared_ptr<ActionPolicy> policy,
            const anet::nn::NetworkForwardConfig& forward = anet::nn::NetworkForwardConfig::GetDefault(),
            std::shared_ptr<ObservationNormalizer> normalizer = nullptr,
            std::unique_ptr<FrameStacker> stacker = nullptr,
            bool emit_snapshot_metrics = false, ActorQHintConfig hint = {},
            std::optional<anet::ProfiledValueConfig<step_t>> sync_interval = std::nullopt,
            std::optional<SkipActorResources> skip = std::nullopt);
        static std::optional<float> GetScalar(const std::string& key, int64_t index,
            const Learner& learner, const RuntimeVars& vars, const ActionPolicy& target_policy,
            std::shared_mutex& mutex);
        static std::optional<torch::Tensor> GetTensor(
            const std::string& key, const Learner& learner, std::shared_mutex& mutex);
        static std::optional<std::vector<torch::Tensor>> GetTensorVector(
            const std::string& key, const Learner& learner, std::shared_mutex& mutex);
    };

} // namespace anet::rl::dqn
