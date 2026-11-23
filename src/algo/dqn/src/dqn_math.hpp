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
#include "dqn_config.hpp"
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

    std::array<float, 3> MeasureNoisyQProbe(const torch::Tensor& mean_q, int num_draws,
        const std::function<torch::Tensor(int)>& draw);

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

    struct ActorQHintRow {
        float actor_q_sa = 0.0f;
        float actor_state_value = 0.0f;
        float munchausen_term = 0.0f;
        float intrinsic_q_sa = 0.0f;
        float intrinsic_state_value = 0.0f;
    };

    ActorQHintRow DecodeActorQHint(std::span<const float> payload);

    const float MET_EMA_DECAY = 0.001f;  // 平滑化係数(メトリクス用)

    const float MET_EMA_DECAY_ACT = 0.0005f;  // 平滑化係数(メトリクス用)action_ema用

    /// ランタイム変数
    struct anet::rl::dqn::RuntimeVars {
        float uqe_tau = 0.9f;   // 互換性のために残す
        anet::rl::step_t learn_step = 0;
        float per_beta = 0.0f;  ///< PER用beta
    };

    struct DiagnosticActionEvaluation {
        int64_t num_taus;
    };

    struct RiskScoreSpec {
        float tau;
        bool use_tail_mean;
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
            anet::ForwardCapture* capture = nullptr) const;
    private:
        anet::nn::NetworkForwardExecutor& forward_; ///< 同期呼び出しの間、ActorまたはLearner所有の実行器を借りる。
        std::optional<float> beta_;
        bool use_tbo_;
        float epsilon_;
    };

    inline constexpr std::string_view kPlasticityProbePrefix = "plasticity_probe_";

    inline constexpr char kExtrinsicQValues[] = "extrinsic_q_values";

    inline constexpr char kIntrinsicQValues[] = "intrinsic_q_values";

    inline constexpr char kConstantScheduleType[] = "constant";

    inline constexpr char kStepIntrinsicQSaKey[] = "intrinsic_q_sa";

    inline constexpr char kStepIntrinsicStateValueKey[] = "intrinsic_state_value";

    inline constexpr std::array<const char*, 9> kValueDecompositionMetricKeys{
    "extrinsic_loss", "intrinsic_loss", "extrinsic_q_sa_mean", "intrinsic_q_sa_mean",
    "extrinsic_head_grad_norm", "intrinsic_head_grad_norm", "extrinsic_grad_norm", "intrinsic_grad_norm", "intrinsic_q_share"};

    inline constexpr char kExtrinsicActionScores[] = "extrinsic_action_scores";

    inline constexpr char kCombinedActionScores[] = "combined_action_scores";

    torch::Tensor MakeScaledLogPolicy(const torch::Tensor& score, float entropy_tau);

    torch::Tensor MakeTauMidpointPositions(int64_t num_taus, const torch::TensorOptions& options);

    // 分解の和と差は桁落ちに敏感なので、同じ逆関数を引き算なしの形で評価する。
// combinedの既存系列は従来の逆変換を保つ。
torch::Tensor InverseDecomposedValue(const torch::Tensor& value, float epsilon);

    float InverseDecomposedValue(float value, float epsilon);

    struct QuantileTailStatistics {
        torch::Tensor upper_truncated_std;
        torch::Tensor lower_truncated_std;
        torch::Tensor crossing_ratio;
    };

    QuantileTailStatistics CalculateQuantileTailStatistics(const torch::Tensor& tau_ordered_quantiles);

} // namespace anet::rl::dqn
