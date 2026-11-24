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
#include "anet/forwarder.hpp"
#include "anet/schedule.hpp"
#include "anet/nn_util.hpp"
#include "anet/nn.hpp"
#include "anet/rl.hpp"
#include "anet/scaler.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/agent.hpp"
#include "anet/transfer.hpp"
#include "dqn_actor.hpp"
#include "dqn_network_model.hpp"
#include "dqn_math.hpp"

namespace anet::rl::dqn {

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

    struct SamplePerception {
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

    /// target計算で使うonline用とtarget用の実行器。
    struct TargetForwards {
        std::shared_ptr<anet::nn::NetworkForwardExecutor> online; ///< Double DQNの行動選択とMunchausenのonline方式
        std::shared_ptr<anet::nn::NetworkForwardExecutor> target; ///< 価値の評価とMunchausenのtarget方式
    };

    struct TargetEvaluation {
        bool diagnostic = false;
        anet::nn::NetworkForwardCallInfo info{{}};
        const TargetForwards* forwards = nullptr; ///< 選択された用途ペア。同期target計算中だけ参照する。
        anet::ForwardCapture* target_capture = nullptr; ///< 通常更新のtarget取得要求。診断ではnullptr。
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

    struct LearnerForwardResources {
        const anet::nn::NetworkForwardConfig current = anet::nn::NetworkForwardConfig::GetDefault();
        const anet::nn::NetworkForwardConfig target = anet::nn::NetworkForwardConfig::GetDefault();
    };

    struct LearnerPerceptionSetup {
        const anet::ForwarderBuilder& builder;
        const anet::TensorSpecMap& expected_specs;
    };

    std::shared_ptr<anet::SpecForwarder> CompletePerception(std::shared_ptr<anet::SpecForwarder> forwarder,
        const std::vector<std::string>& stack_keys, const std::string& start_key = {});
    void RequirePerceptionSpecs(const anet::TensorSpecMap& expected, const anet::TensorSpecMap& actual,
        const anet::ForwardKind& kind);

    class Learner : public anet::rl::Learner, public anet::Module, public anet::Serializable, public anet::RandomHolder {
    public:
        Learner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars,
            const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec,
            torch::Device device, anet::seed_t replay_seed, anet::seed_t agent_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            const LearnerForwardResources& forward_resources,
            std::optional<ReplayFrameStack> frame_stack = std::nullopt,
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
        void SetupReplayBuffer(const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec, anet::seed_t seed,
            const LearnerPerceptionSetup* perception_setup = nullptr);
        anet::ForwardCapture* OnlineBranchCapture();
        anet::ForwardCapture* TargetBranchCapture();
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
        SamplePerception MakeSamplePerception(const anet::rl::ExperienceSamples& samples) const;
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
        std::optional<ReplayFrameStack> frame_stack_;
        std::shared_ptr<ActionPolicy> target_policy_;
        NetworkModel& model_;
        RuntimeVars& vars_;
        std::shared_ptr<anet::SpecForwarder> sample_perception_;
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
            anet::ForwardCapture online_capture;
            anet::ForwardCapture target_capture;
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
            const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed, anet::seed_t agent_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            const LearnerForwardResources& forward_resources,
            std::optional<ReplayFrameStack> frame_stack = std::nullopt,
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
        explicit TDLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, const LearnerPerceptionSetup* perception_setup,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed, anet::seed_t agent_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            const LearnerForwardResources& forward_resources,
            std::optional<ReplayFrameStack> frame_stack = std::nullopt,
            std::optional<anet::seed_t> target_seed = std::nullopt,
            std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe_config = nullptr);

    };

    class QRLearner final : public anet::rl::dqn::QuantileLearnerBase {
    public:
        explicit QRLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, const LearnerPerceptionSetup* perception_setup,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed, anet::seed_t agent_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            const LearnerForwardResources& forward_resources,
            std::optional<ReplayFrameStack> frame_stack = std::nullopt,
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
        explicit IQNLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, const LearnerPerceptionSetup* perception_setup,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed, anet::seed_t agent_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            const LearnerForwardResources& forward_resources,
            std::optional<ReplayFrameStack> frame_stack = std::nullopt,
            std::optional<anet::seed_t> target_seed = std::nullopt,
            std::shared_ptr<const anet::nn::NetworkForwardConfig> noisy_probe_config = nullptr);

    private:
        CurrentForwardResult ForwardCurrent(const ExperienceSamples& samples,
            const anet::TensorDict& obs, const anet::nn::NetworkForwardCallInfo& info) override;
        torch::Tensor MakeUpdateDiagnostics(const ExperienceSamples& samples,
            const CurrentForwardResult& current, const torch::Tensor& target, const ElementError& error) const override;
    };

} // namespace anet::rl::dqn
