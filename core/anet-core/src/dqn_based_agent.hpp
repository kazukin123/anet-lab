// rainbow_agent_impl.hpp

#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <limits>
#include <span>
#include <unordered_set>
#include "anet/agent.hpp"
#include "anet/rl.hpp"
#include "anet/scaler.hpp"
#include "anet/nn.hpp"
#include "anet/nn_util.hpp"
#include "anet/transfer.hpp"
#include "anet/replay_buffer.hpp"
#include "anet/random.hpp"
#include "anet/schedule.hpp"


namespace anet::rl::dqn {

    inline constexpr std::array<const char*, 7> kPolicyChurnMetricKeys = {
        "policy_churn_action_ratio",
        "policy_churn_q_delta_abs_mean",
        "policy_churn_q_delta_signed_max",
        "policy_churn_q_delta_signed_min",
        "policy_churn_target_policy_disagreement",
        "policy_churn_target_q_delta_abs_mean",
        "policy_churn_target_sync_age",
    };

    inline std::optional<size_t> ParsePolicyChurnMetric(const std::string& key)
    {
        for (size_t i = 0; i < kPolicyChurnMetricKeys.size(); ++i) {
            if (key == kPolicyChurnMetricKeys[i]) return i;
        }
        return std::nullopt;
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

    inline constexpr int64_t kActorQHintColumnCount = 2;
    inline constexpr int64_t kActorQSaColumn = 0;
    inline constexpr int64_t kActorStateValueColumn = 1;

    struct ActorQHintBatch {
        torch::Tensor actor_q_sa;
        torch::Tensor actor_state_value;
    };

    struct ActorQHintRow {
        float actor_q_sa = 0.0f;
        float actor_state_value = 0.0f;
    };

    torch::Tensor PackActorQHint(
        const torch::Tensor& actor_q_sa, const torch::Tensor& actor_state_value);
    ActorQHintBatch DecodeActorQHint(const torch::Tensor& payload);
    ActorQHintRow DecodeActorQHint(std::span<const float> payload);

    const float MET_EMA_DECAY = 0.001f;  // 平滑化係数(メトリクス用)
    const float MET_EMA_DECAY_ACT = 0.0005f;  // 平滑化係数(メトリクス用)action_ema用


    // ======================================================
    //  データ構造
    // ======================================================

    /// ランタイム変数
    struct anet::rl::dqn::RuntimeVars {
        float epsilon = 1.0f;   // 互換性のために残す
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
        torch::Tensor grad_norm_tensor;
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

        // policy churn の 7 scalar を key 順に保持する CPU float32 pack。
        torch::Tensor policy_churn_metrics;

    public:
        BatchUpdateResult() = default;

        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override
        {
            // 必要になって初めてCPUに転送する

            if (const auto metric = ParsePolicyChurnMetric(key)) {
                if (!policy_churn_metrics.defined()) {
                    return std::numeric_limits<float>::quiet_NaN();
                }
                if (policy_churn_metrics.numel() != static_cast<int64_t>(kPolicyChurnMetricKeys.size())) {
                    ANET_SYSTEM_ERROR("DQN policy churn scalar pack has invalid size="
                        << policy_churn_metrics.numel() << " expected=" << kPolicyChurnMetricKeys.size() << ".");
                }
                return policy_churn_metrics[static_cast<int64_t>(*metric)].item<float>();
            }

            if (auto value = GetPlasticityScalar(key)) return value;

            // loss/td/grad
            if (key == "loss") return loss.item<float>();
            if (key == "td_mean") {
                if (!td_error_abs_cpu.defined())
                    td_error_abs_cpu = td_error.abs().cpu();
                return td_error_abs_cpu.mean().item<float>();
            }
            if (key == "td_std") {
                if (!td_error_abs_cpu.defined())
                    td_error_abs_cpu = td_error.abs().cpu();
                return td_error_abs_cpu.std().item<float>();
            }
            if (key == "grad_norm") {
                if (grad_norm.has_value())
                    return *grad_norm;
                if (grad_norm_tensor.defined())
                    return grad_norm_tensor.item<float>();
                return std::nullopt;
            }
            if (key == "grad_clip_ratio") {
                if (!grad_norm_tensor.defined()) return 0.0f;
                return (grad_norm_tensor.item<float>() > grad_clip_tau) ? 1.0f : 0.0f;
            }

			// Q Values
            if (key == "q_max_max") {
                TransQToCpu();
                return max_q_cpu.defined() ? std::optional<float>(max_q_cpu.max().item<float>()) : std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "q_max_mean") {
                TransQToCpu();
                return max_q_cpu.defined() ? std::optional<float>(max_q_cpu.mean().item<float>()) : std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "q_max_std") {
                TransQToCpu();
                return max_q_cpu.defined() ? std::optional<float>(max_q_cpu.std(false).item<float>()) : std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "q_sa_mean") {
                if (!q_sa_cpu.defined() && q_sa.defined()) q_sa_cpu = q_sa.cpu();
                return q_sa_cpu.defined() ? std::optional<float>(q_sa_cpu.mean().item<float>()) : std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "q_max_real_max") {
                TransRealQToCpu();
                return max_q_real_cpu.defined() ? std::optional<float>(max_q_real_cpu.max().item<float>()) : std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "q_max_real_mean") {
                TransRealQToCpu();
                return max_q_real_cpu.defined() ? std::optional<float>(max_q_real_cpu.mean().item<float>()) : std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "q_max_real_std") {
                TransRealQToCpu();
                return max_q_real_cpu.defined() ? std::optional<float>(max_q_real_cpu.std(false).item<float>()) : std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "q_sa_real_mean") {
                if (!q_sa_real_cpu.defined() && q_sa_real.defined()) q_sa_real_cpu = q_sa_real.cpu();
                return q_sa_real_cpu.defined() ? std::optional<float>(q_sa_real_cpu.mean().item<float>()) : std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "q_std") {
                if (q_std.defined()) return anet::ToFloat(q_std);
                return 0.0f;
            }
            if (key == "q_gap") {
                if (q_gap.defined()) return anet::ToFloat(q_gap);
                return 0.0f;
            }
            if (key == "q_gap_rel") {
                if (q_gap_rel.defined()) return anet::ToFloat(q_gap_rel);
                return 0.0f;
            }

            int64_t iqn_diagnostic_index = -1;
            if (key == "iqn_current_mc_scale") iqn_diagnostic_index = 0;
            else if (key == "iqn_target_mc_scale") iqn_diagnostic_index = 1;
            else if (key == "iqn_priority_mc_ratio") iqn_diagnostic_index = 2;
            else if (key == "iqn_first_priority_mc_ratio") iqn_diagnostic_index = 3;
            else if (key == "iqn_first_pair_abs_td") iqn_diagnostic_index = 4;
            else if (key == "iqn_first_cancellation_ratio") iqn_diagnostic_index = 5;
            else if (key == "iqn_first_quantile_loss_norm") iqn_diagnostic_index = 6;
            if (iqn_diagnostic_index >= 0) {
                return iqn_diagnostics.defined()
                    ? iqn_diagnostics[iqn_diagnostic_index].item<float>()
                    : std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "upper_tail_priority_spearman") {
                return upper_tail_priority_spearman;
            }

            // PER Metrics
            if (key == "per_td_error_abs_max") {
                if (td_error.defined())
                    return td_error.abs().max().item<float>();
                return std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "per_prio_clip_ratio") {
                if (per_clipped_count.defined() && per_minibatch_size > 0)
                    return per_clipped_count.item<float>() / static_cast<float>(per_minibatch_size);
                return std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "per_sample_initial_ratio") {
                if (per_sample_initial_count.defined() && per_minibatch_size > 0)
                    return per_sample_initial_count.item<float>() / static_cast<float>(per_minibatch_size);
                return std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "per_sample_initial_count") {
                return per_sample_initial_count.defined()
                    ? per_sample_initial_count.item<float>()
                    : 0.0f;
            }
            if (key == "per_sample_fixed_initial_ratio") {
                return per_minibatch_size > 0 && per_sample_fixed_initial_count.defined()
                    ? per_sample_fixed_initial_count.item<float>() / static_cast<float>(per_minibatch_size)
                    : std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "per_sample_max_initial_ratio") {
                return per_minibatch_size > 0 && per_sample_max_initial_count.defined()
                    ? per_sample_max_initial_count.item<float>() / static_cast<float>(per_minibatch_size)
                    : std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "per_sample_actor_initial_ratio") {
                return per_minibatch_size > 0 && per_sample_actor_initial_count.defined()
                    ? per_sample_actor_initial_count.item<float>() / static_cast<float>(per_minibatch_size)
                    : std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "per_priority_update_stale_ratio") {
                const int64_t total = per_update_result.applied_count + per_update_result.stale_count;
                return total > 0 ? static_cast<float>(per_update_result.stale_count) / static_cast<float>(total)
                    : std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "per_actor_learner_pair_count") return static_cast<float>(per_update_result.actor_learner_pair_count);
            if (key == "per_actor_learner_positive_pair_ratio") return per_update_result.actor_learner_positive_pair_ratio;
            if (key == "per_actor_learner_ratio_median") return per_update_result.actor_learner_ratio_median;
            if (key == "per_actor_learner_log_ratio_mean") return per_update_result.actor_learner_log_ratio_mean;
            if (key == "per_actor_learner_spearman") return per_update_result.actor_learner_spearman;
            if (key == "per_prio_max") {
                if (per_priorities.defined())
                    return per_priorities.max().item<float>();
                return std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "per_batch_prio_mean") {
                if (per_priorities.defined())
                    return per_priorities.mean().item<float>();
                return std::numeric_limits<float>::quiet_NaN();
            }
            if (key == "per_is_weight_mean") {
                if (per_is_weights.defined())
                    return per_is_weights.mean().item<float>();
                return std::numeric_limits<float>::quiet_NaN();
            }
            // CV = 標準偏差 / 平均
            //  高: 特定の経験に優先度が集中しており、PERが「選別」を強く行っている状態。
            //  低(0に近い) : すべての経験が似たようなTD誤差を持っており、一様サンプリングに近い状態。
            // 対策: per_alpha を調整
            if (key == "per_prio_cv") {
                if (per_priorities.defined()) {
                    auto mean = per_priorities.mean();
                    auto std = per_priorities.std();
                    return (std / (mean + 1e-9)).item<float>();
                }
                return std::numeric_limits<float>::quiet_NaN();
            }
            // 勾配更新の偏り(IS Weightsベース)
            if (key == "per_is_ess_ratio") {
                if (per_is_weights.defined() && per_minibatch_size > 0) {
                    auto w = per_is_weights;
                    auto sum_w = w.sum();
                    auto sum_w2 = (w * w).sum();
                    // ESS = (Σw)^2 / (Σw^2) / B
                    return ((sum_w * sum_w) / (static_cast<float>(per_minibatch_size) * sum_w2 + 1e-9)).item<float>();
                }
                return std::numeric_limits<float>::quiet_NaN();
            }
            // 有効サンプルサイズ比率 (ESS Ratio) 実質的にバッチ内の何割のデータが学習に寄与しているか (0.0 ~ 1.0) 公式: (Σp)^2 / (B * Σp^2)
            //   1.0に近い: バッチ内のデータが均等に重要。
            //   0に近い : バッチ内の極一部のデータ（外れ値など）が支配的で、実質的な学習効率が低下している警告信号。 対策: per_beta を上げる（初期値を 0.6 -> 0.8 にするなど）か、per_alpha を下げる。
            // 理想: 0.5 〜 0.8 付近
            if (key == "per_prio_ess_ratio") {
                if (per_priorities.defined() && per_minibatch_size > 0) {
                    auto p = per_priorities;
                    auto sum_p = p.sum();
                    auto sum_p2 = (p * p).sum();
                    auto ess = (sum_p * sum_p) / (static_cast<float>(per_minibatch_size) * sum_p2 + 1e-9);
                    return ess.item<float>();
                }
                return std::numeric_limits<float>::quiet_NaN();
            }
            return std::nullopt;
        }

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
        std::optional<float> GetPlasticityScalar(const std::string& key) const
        {
            int64_t weight_norm_index = -1;
            if (key == "plasticity_weight_norm_feature") weight_norm_index = 0;
            else if (key == "plasticity_weight_norm_readout") weight_norm_index = 1;
            else if (key == "plasticity_weight_norm_feature_effective") weight_norm_index = 2;
            else if (key == "plasticity_weight_norm_readout_effective") weight_norm_index = 3;
            else if (key == "plasticity_spectral_sigma_feature") weight_norm_index = 4;
            else if (key == "plasticity_spectral_sigma_readout") weight_norm_index = 5;
            if (weight_norm_index >= 0) {
                if (!plasticity_weight_norms.defined()) {
                    return std::numeric_limits<float>::quiet_NaN();
                }
                if (!plasticity_weight_norms_cpu.defined()) {
                    plasticity_weight_norms_cpu = plasticity_weight_norms.cpu();
                }
                if (plasticity_weight_norms_cpu.numel() != 8) {
                    ANET_SYSTEM_ERROR("DQN plasticity weight norm pack has invalid size="
                        << plasticity_weight_norms_cpu.numel() << " expected=8.");
                }
                ValidateSpectralNormSentinel(
                    plasticity_weight_norms_cpu[6].item<float>(), "online", plasticity_online_network);
                ValidateSpectralNormSentinel(
                    plasticity_weight_norms_cpu[7].item<float>(), "target", plasticity_target_network);
                return plasticity_weight_norms_cpu[weight_norm_index].item<float>();
            }

            const torch::Tensor* features = nullptr;
            const anet::nn::PlasticityMetricRequest* request = nullptr;
            std::optional<anet::nn::PlasticityMetrics>* cache = nullptr;
            std::string suffix;
            if (key.starts_with("plasticity_target_")) {
                features = &plasticity_target_features;
                request = &plasticity_target_request;
                cache = &plasticity_target_metrics_cache;
                suffix = key.substr(std::string("plasticity_target_").size());
            } else if (key.starts_with("plasticity_")) {
                features = &plasticity_features;
                request = &plasticity_request;
                cache = &plasticity_metrics_cache;
                suffix = key.substr(std::string("plasticity_").size());
            } else {
                return std::nullopt;
            }

            const auto metric = anet::nn::ParsePlasticityMetricSuffix(suffix);
            if (!metric.has_value()) return std::nullopt;
            if (!request->Contains(*metric)) return std::numeric_limits<float>::quiet_NaN();
            if (!features->defined()) return std::numeric_limits<float>::quiet_NaN();
            if (!cache->has_value()) {
                *cache = anet::nn::ComputePlasticityMetrics(*features, *request);
            }
            const auto value = cache->value().Get(*metric);
            if (!value.has_value()) {
                ANET_SYSTEM_ERROR("BatchUpdateResult plasticity cache is missing a requested metric.");
            }
            return value;
        }

        static void ValidateSpectralNormSentinel(
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

        void TransQToCpu() const
        {
            if (max_q_cpu.defined()) return;
            max_q_cpu = max_q.cpu();
        }
        void TransRealQToCpu() const
        {
            if (max_q_real_cpu.defined()) return;
            if (max_q_real.defined()) max_q_real_cpu = max_q_real.cpu();
        }
    };

    struct NormalizedSampleObservations {
        anet::TensorDict obs;
        anet::TensorDict next_obs;
    };

    struct OptimizerStepResult {
        torch::Tensor grad_norm_tensor;
        std::optional<float> grad_norm;
        float grad_clip_ratio = 0.0f;
        float grad_clip_tau = std::numeric_limits<float>::infinity();
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
        int64_t upper_tail_std_count = 0;                ///< pack末尾のsample単位upper-tail幅数
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

    class NetworkModel : public anet::Serializable {
    public:
        NetworkModel(
            const NetworkModelConfig& config,
            const torch::Device device,
            const anet::nn::NetworkConfig& network_config,
            const anet::TensorSpecMap& obs_spec,
            int64_t n_actions,
            std::shared_ptr<anet::nn::NetworkHeadFactory> head_factory,
            bool distributional,
            anet::seed_t network_seed);
    protected:
        NetworkModel(
            const NetworkModelConfig& config,
            std::shared_ptr<anet::nn::Network> online_net,
            std::shared_ptr<anet::nn::Network> target_net,
            int64_t n_actions,
            bool distributional);
    public:

        /// 行動選択・学習用：期待値Q (B, A) を返す
        /// QR-DQNの場合は分布の平均を計算して返す
        anet::TensorDict ForwardOnline(const anet::TensorDict& obs) const;
        anet::TensorDict ForwardOnlineWithTrain(const anet::TensorDict& obs) const;
        anet::TensorDict ForwardTarget(const anet::TensorDict& obs) const;
        anet::TensorDict ForwardOnlineUpTo(const anet::TensorDict& obs, const std::string& branch_key) const;
        void SetBranchCaptureRequests(
            anet::nn::NetworkBranchCapture* online_capture,
            anet::nn::NetworkBranchCapture* target_capture);
        void ClearBranchCaptureRequests();

        // Network取得
        std::shared_ptr<anet::nn::Network> GetOnlineNetwork() { return online_net_; }
        std::shared_ptr<anet::nn::Network> GetTargetNetwork() { return target_net_; }

        /// 分布型Head判定
        bool IsDistributional() const;

        /// online_netのパラメータ取得
        std::vector<torch::Tensor> GetOnlineParameters() const;

        torch::OrderedDict<std::string, torch::Tensor> GetOnlineNamedParameters() const;

        /// target network 同期
        void UpdateTarget(anet::rl::step_t learn_step);
        int GetHardUpdateInterval() const { return config_.hard_update_interval; }

        /// メトリクス用：TensorDict
        std::optional<anet::TensorDictFunction> GetTensorDictFunction(const std::string& key, const torch::Device& device);
    public:
        int64_t Save(OutputArchive& archive) const override;
        int64_t Load(InputArchive& archive) override;
    private:
        void ValidateSpectralNormSoftUpdateConfig() const;
        void SoftUpdate();
        void HardUpdate();
    private:
        const anet::rl::dqn::NetworkModelConfig config_;
        std::shared_ptr<anet::nn::Network> online_net_;
        std::shared_ptr<anet::nn::Network> target_net_;

        int64_t n_actions_;
        bool distributional_;
        anet::nn::NetworkBranchCapture* online_capture_ = nullptr;
        anet::nn::NetworkBranchCapture* target_capture_ = nullptr;
    };


    // ======================================================
    // ActionPolicy
    // ======================================================

    struct TrainActorSnapshotMetrics {
        float interval = std::numeric_limits<float>::quiet_NaN();
        float age = std::numeric_limits<float>::quiet_NaN();
    };

    class DQNActionInfo final : public anet::rl::BatchActionInfo, public anet::ModuleBase {
    public:
        using anet::rl::BatchActionInfo::BatchActionInfo;

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
    };

    class ActionPolicy : virtual public anet::ModuleBase {
    public:
        ActionPolicy(const ActionPolicyConfig& config,
        	bool enable_spatial_exploration = false, int64_t num_envs = 0,
            const torch::Device& device = torch::Device(torch::kCPU));

        virtual std::shared_ptr<DQNActionInfo> SelectAction(const anet::TensorDict& obs, bool greedy_only,
            std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd,
            const anet::TraceCallback& callback = {}) const = 0;
        virtual void OnLearn(const StepCounts& counts) { }

        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;

        virtual ~ActionPolicy() = default;
    protected:
        anet::TensorDict ForwardForAction(const anet::TensorDict& obs, std::shared_ptr<anet::nn::Network> network, const anet::TraceCallback& callback) const;
        anet::TensorDict ForwardForActionWithTaus(
            const anet::TensorDict& obs, const torch::Tensor& taus,
            std::shared_ptr<anet::nn::Network> network, const anet::TraceCallback& callback) const;
        torch::Tensor MakeEpsilonGreedyAction(const torch::Tensor& greedy_action, float epsilon, int64_t num_envs, int64_t n_actions, std::shared_ptr<anet::RandomGenerator> rnd) const;
        torch::Tensor MakeEpsilonGreedyAction(const torch::Tensor& greedy_action, const torch::Tensor& epsilon_tensor, int64_t num_envs, int64_t n_actions, std::shared_ptr<anet::RandomGenerator> rnd) const;
        std::shared_ptr<DQNActionInfo> MakeActionInfo(const torch::Tensor& action_values, const torch::Tensor& q_values, const torch::Tensor& q_quantiles) const;
        //torch::Tensor GetQuantiles(const torch::Tensor& obs, bool use_target) const;
        void UpdateEpsilon(step_t step, bool is_uqe = false);
        bool IsSpatialExplorationEnabled() const { return use_spatial_exploration_; }
        static torch::Tensor CreateSpatialTensor(int64_t num_envs, float start_val, float end_val, const std::string& scale_type, const torch::Device& device);
        static torch::Tensor CreateSpatialLaneTensor(int64_t num_envs, float start_val, float end_val, const std::string& scale_type, const torch::Device& device);
        torch::Tensor GetSpatialEpsilonTensor(int64_t num_envs, const torch::Device& device, bool is_uqe) const;
        torch::Tensor GetSpatialTauTensor(int64_t num_envs, const torch::Device& device) const;
    protected:
        const ActionPolicyConfig config_;
        bool use_spatial_exploration_ = false;
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
            std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd,
            const anet::TraceCallback& callback) const;
        void OnLearn(const StepCounts& counts) override;
    };

    /**
     * UQE (Upper Quantile Exploration)
     * 分布型RLにおける楽観的探索 (Optimistic Exploration) を行うクラス。
     * 分布の右裾（上位Quantile）の情報を用いて、不確実性の高い行動を積極的に探索する。
     * * uqe_use_tail_meanにより、以下2つの標準的な手法に対応：
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
            std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd,
            const anet::TraceCallback& callback) const;
        void OnLearn(const StepCounts& counts) override;

        virtual ~UQEActionPolicy() = default;
    protected:
        std::shared_ptr<DQNActionInfo> MakeUQEActionInfo(float tau, const torch::Tensor& tau_tensor, const anet::TensorDict& obs, bool greedy_only,
            std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd, const anet::TraceCallback& callback,
            bool iqn_use_full_range = false) const;
        void UpdateTau(step_t step);
    private:
        torch::Tensor MakeUQEValues(float tau, const torch::Tensor& q_quantiles) const;
        torch::Tensor MakeVectorizedUQEValues(const torch::Tensor& tau_tensor, const torch::Tensor& q_quantiles) const;
    };

    class ThompsonSamplingActionPolicy final : public UQEActionPolicy {
    public:
        ThompsonSamplingActionPolicy(const ActionPolicyConfig& config,
            bool enable_spatial_exploration = false,
            int64_t num_envs = 0,
            const torch::Device& device = torch::Device(torch::kCPU));

        std::shared_ptr<DQNActionInfo> SelectAction(const anet::TensorDict& obs, bool greedy_only,
            std::shared_ptr<anet::nn::Network> network, std::shared_ptr<anet::RandomGenerator> rnd,
            const anet::TraceCallback& callback) const;
        void OnLearn(const StepCounts& counts) override;
    };


    // ======================================================
    // Actor
    // ======================================================

    class Actor : public anet::rl::Actor {
    public:
        Actor(std::shared_ptr<ActionPolicy> policy,
            std::shared_ptr<anet::rl::ObservationNormalizer> obs_norm,
            std::shared_ptr<ActionContext> context,
            std::shared_ptr<std::shared_mutex> mutex,
            std::shared_ptr<anet::nn::Network> network,
            std::shared_ptr<anet::nn::Network> src_network,
            bool emit_actor_q_hint = false,
            std::optional<anet::ProfiledValueConfig<step_t>> snapshot_sync_interval = std::nullopt,
            bool emit_snapshot_metrics = false);
        std::shared_ptr<BatchActionInfo> MakeAction(const StepCounts& step, const anet::rl::BatchState& state) const override;
        void Sync() override;
    private:
        void CopySourceNetwork() const;
        void UpdateSnapshot(const StepCounts& step) const;
    private:
        std::shared_ptr<ActionPolicy> policy_;
        std::shared_ptr<anet::rl::ObservationNormalizer> obs_norm_;
        std::shared_ptr<ActionContext> context_;
        std::shared_ptr<std::shared_mutex> mutex_;
        std::shared_ptr<anet::nn::Network> network_;
        std::shared_ptr<anet::nn::Network> src_network_;
        bool emit_actor_q_hint_ = false; ///< 学習Actorから初期優先度推定用Qヒントを生成するか
        mutable std::optional<anet::ProfiledValue<step_t>> snapshot_sync_interval_;
        mutable step_t last_snapshot_sync_train_step_ = 0;
        mutable bool reset_snapshot_age_on_next_action_ = false;
        bool emit_snapshot_metrics_ = false;
    };

    // ======================================================
    // Learner
    // ======================================================

    class Learner : public anet::rl::Learner, public anet::Module, public anet::Serializable, public anet::RandomHolder {
    public:
        Learner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec,
            torch::Device device, anet::seed_t replay_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            std::optional<StuckerConfig> stucker_config = std::nullopt,
            std::optional<anet::seed_t> target_seed = std::nullopt,
            anet::RandomGenerator* plasticity_probe_random = nullptr,
            anet::RandomGenerator* policy_churn_probe_random = nullptr);

        BatchUpdateResultList UpdateFromBatch(const StepCounts& step, const BatchExperience& expriences) override;
        void ConfigureScalarMetricSubscriptions(
            const std::vector<ScalarMetricSubscription>& subscriptions);

        virtual ~Learner() = default;
    public:
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override;
    public:
        int64_t Save(OutputArchive& archive) const override;
        int64_t Load(InputArchive& archive) override;
    protected:
        // アルゴリズム固有の更新処理 (Loss計算, Backprop, Priority更新)
        virtual std::shared_ptr<anet::rl::dqn::BatchUpdateResult> UpdateFromSamples(
            const anet::rl::ExperienceSamples& samples) = 0;
    protected:
        void SetupOptimizer();                  ///< 共通初期化処理（Optimizer生成など）
        void SetupReplayBuffer(const BatchEnvSpec batch_env_spec, const EnvSpec& env_spec, anet::seed_t seed);
        NormalizedSampleObservations NormalizeSampleObservations(const anet::rl::ExperienceSamples& samples) const;
        OptimizerStepResult Optimize(const torch::Tensor& loss);
    protected:
        PerPriorityUpdatePending PreparePerPriorityUpdate(
            const anet::rl::ExperienceSamples& samples,
            const torch::Tensor& td_error,
            const torch::Tensor& iqn_diagnostics = torch::Tensor(),
            const torch::Tensor& upper_tail_std = torch::Tensor());
        PerPriorityUpdateInfo ApplyPerPriorityUpdate(PerPriorityUpdatePending pending);
        PerPriorityUpdateInfo UpdatePerPriorities(const anet::rl::ExperienceSamples& samples, const torch::Tensor& td_error);
    protected:
        torch::Tensor TransformH(const torch::Tensor& x) const;
        torch::Tensor TransformHInv(const torch::Tensor& x) const;
        std::shared_ptr<anet::rl::dqn::BatchUpdateResult> MakeBatchUpdateResult(
            const torch::Tensor& loss,
            const torch::Tensor& td_error,
            const OptimizerStepResult& opt_result,
            const torch::Tensor& max_q,
            const torch::Tensor& q_sa,
            const PerPriorityUpdateInfo& per_info,
            const torch::Tensor& q_std = torch::Tensor(),
            const torch::Tensor& q_gap = torch::Tensor(),
            const torch::Tensor& q_gap_rel = torch::Tensor()) const;
    private:
        bool CanUpdate(step_t exp_step);
        void UpdatePerBeta(step_t step);
        void UpdateTargetNetwork(step_t step);
        void ValidateDeviceSamples(const anet::rl::ExperienceSamples& samples, int64_t batch_size) const;
        void CapturePlasticityProbe();
        void PreparePolicyChurnRequest();
        void CapturePolicyChurnProbe();
        void CapturePolicyChurnOnlineBefore();
        void CapturePolicyChurnOnlineAfter();
        void FinalizePolicyChurn(BatchUpdateResult& result);
        torch::Tensor ForwardPolicyChurnExpectedQ(bool target) const;
    protected:
        const torch::Device device_;
        int num_envs_;
        int n_actions_;
        float earned_credit_;
        LearnerConfig config_;
        std::optional<StuckerConfig> stucker_config_;
        std::shared_ptr<ActionPolicy> target_policy_;
        NetworkModel& model_;
        RuntimeVars& vars_;
        std::shared_ptr<ObservationNormalizer> obs_norm_;
        std::shared_ptr<anet::rl::ReplayBuffer> replay_buffer_;
        std::unique_ptr<torch::optim::Optimizer> optimizer_;
        anet::GradScaler grad_scaler_;
        anet::RandomGenerator* plasticity_probe_random_ = nullptr;
        anet::RandomGenerator* policy_churn_probe_random_ = nullptr;
        std::optional<at::cuda::CUDAStream> per_priority_copy_stream_;
        anet::transfer::EventRecycler<torch::Tensor> per_priority_event_recycler_;
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
            anet::nn::NetworkBranchCapture online_capture;
            anet::nn::NetworkBranchCapture target_capture;
            anet::nn::PlasticityMetricRequest online_request;
            anet::nn::PlasticityMetricRequest target_request;
            anet::nn::PlasticityMetricRequest probe_request;
            torch::Tensor probe_features;
            torch::Tensor weight_norms;
            mutable std::optional<anet::nn::PlasticityMetrics> probe_metrics_cache;
        } plasticity_;
        struct PolicyChurnState {
            struct DemandEntry {
                size_t metric_index;
                int interval;
                anet::IntervalGate gate;

                DemandEntry(size_t metric_index, int interval)
                    : metric_index(metric_index), interval(interval), gate(static_cast<uint64_t>(interval)) {}
            };

            std::vector<DemandEntry> demand;
            std::array<bool, kPolicyChurnMetricKeys.size()> request{};
            anet::TensorDict probe_obs;
            torch::Tensor fixed_taus;
            torch::Tensor online_before;
            torch::Tensor online_after;

            bool AnyQRequest() const
            {
                for (size_t i = 0; i < request.size() - 1; ++i) {
                    if (request[i]) return true;
                }
                return false;
            }

            bool NeedsOnlineBefore() const
            {
                return request[0] || request[1] || request[2] || request[3];
            }

            bool NeedsOnlineAfter() const
            {
                return AnyQRequest();
            }

            bool NeedsTargetAfter() const
            {
                return request[4] || request[5];
            }
        } policy_churn_;
        std::unordered_set<int> policy_churn_warned_intervals_;
    protected:
        float update_credit_ = 0.0f;
    private:
        bool has_enough_replay_samples_ = false;
    };

    class QuantileLearnerBase : public anet::rl::dqn::Learner {
    public:
        explicit QuantileLearnerBase(
        	const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            std::optional<StuckerConfig> stucker_config = std::nullopt,
            std::optional<anet::seed_t> target_seed = std::nullopt,
            anet::RandomGenerator* plasticity_probe_random = nullptr,
            anet::RandomGenerator* policy_churn_probe_random = nullptr);

        virtual ~QuantileLearnerBase() = default;
    protected:
        torch::Tensor GatherActionQuantiles(const torch::Tensor& quantiles, const torch::Tensor& actions) const;
        torch::Tensor SelectTargetActions(const anet::TensorDict& next_obs);
        torch::Tensor CalcTargetQuantiles(const anet::rl::ExperienceSamples& samples, const torch::Tensor& next_dist) const;
        QuantileMetrics MakeQuantileMetrics(const torch::Tensor& current_dist, const torch::Tensor& q_values_mean) const;
        torch::Tensor ComputeQuantileHuberLoss(
        	const torch::Tensor& current_dist, const torch::Tensor& target_dist, const torch::Tensor& taus) const;
        static torch::Tensor ComputeQuantileHuberLoss(
            const torch::Tensor& current_dist, const torch::Tensor& target_dist, const torch::Tensor& taus, float kappa);
        static IqnLossResult ComputeIqnQuantileHuberLoss(
            const torch::Tensor& current_dist, const torch::Tensor& target_dist, const torch::Tensor& taus, float kappa);
    };

    class TDLearner final : public anet::rl::dqn::Learner {
    public:
        explicit TDLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            std::optional<StuckerConfig> stucker_config = std::nullopt,
            std::optional<anet::seed_t> target_seed = std::nullopt,
            anet::RandomGenerator* plasticity_probe_random = nullptr,
            anet::RandomGenerator* policy_churn_probe_random = nullptr);

        std::shared_ptr<anet::rl::dqn::BatchUpdateResult> UpdateFromSamples(
            const anet::rl::ExperienceSamples& samples) override;
    };

    class QRLearner final : public anet::rl::dqn::QuantileLearnerBase {
    public:
        explicit QRLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            std::optional<StuckerConfig> stucker_config = std::nullopt,
            std::optional<anet::seed_t> target_seed = std::nullopt,
            anet::RandomGenerator* plasticity_probe_random = nullptr,
            anet::RandomGenerator* policy_churn_probe_random = nullptr);

        std::shared_ptr<anet::rl::dqn::BatchUpdateResult> UpdateFromSamples(
            const anet::rl::ExperienceSamples& samples) override;
    private:
        torch::Tensor tau_i_; // QuantileHuberLoss 算出用
    };

    class IQNLearner final : public anet::rl::dqn::QuantileLearnerBase {
    public:
        explicit IQNLearner(const LearnerConfig& config, NetworkModel& model, RuntimeVars& vars, std::shared_ptr<ObservationNormalizer> obs_norm,
            const BatchEnvSpec& batch_env_spec, const EnvSpec& env_spec, torch::Device device, anet::seed_t replay_seed,
            std::shared_ptr<ActionPolicy> target_policy,
            std::optional<StuckerConfig> stucker_config = std::nullopt,
            std::optional<anet::seed_t> target_seed = std::nullopt,
            anet::RandomGenerator* plasticity_probe_random = nullptr,
            anet::RandomGenerator* policy_churn_probe_random = nullptr);

        std::shared_ptr<anet::rl::dqn::BatchUpdateResult> UpdateFromSamples(
            const anet::rl::ExperienceSamples& samples) override;
    };


} // namespace anet::rl::dqn
