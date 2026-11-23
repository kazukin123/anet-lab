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
#include "dqn_network_model.hpp"
#include "dqn_math.hpp"

namespace anet::rl::dqn {

    struct ActorQHintBatch {
        torch::Tensor actor_q_sa;
        torch::Tensor actor_state_value;
        torch::Tensor munchausen_term;
        torch::Tensor intrinsic_q_sa;
        torch::Tensor intrinsic_state_value;
    };

    torch::Tensor PackActorQHint(
        const torch::Tensor& actor_q_sa, const torch::Tensor& actor_state_value, const torch::Tensor& munchausen_term,
        const torch::Tensor& intrinsic_q_sa, const torch::Tensor& intrinsic_state_value);

    ActorQHintBatch DecodeActorQHint(const torch::Tensor& payload);

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
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        void SetTrainActorSnapshotMetrics(TrainActorSnapshotMetrics metrics)
        {
            train_actor_snapshot_metrics_ = metrics;
        }
    private:
        friend class Actor;
        void ApplyForcedActions(const anet::rl::ForcedActions& forced_actions);

        std::optional<TrainActorSnapshotMetrics> train_actor_snapshot_metrics_;
        mutable torch::Tensor iqn_policy_diagnostics_cpu_;
        mutable torch::Tensor quantile_tail_diagnostics_cpu_;
        mutable torch::Tensor skip_diagnostics_cpu_;
        mutable torch::Tensor value_selection_diagnostics_cpu_;
    };

    class SpatialLadderSchedule {
    public:
        SpatialLadderSchedule(const ActionPolicyConfig::LadderConfig& config,
            std::string scale, int64_t lanes, torch::Device device,
            const std::string& scale_key);
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

    class ActionPolicy : virtual public anet::ModuleBase {
    public:
        ActionPolicy(const ActionPolicyConfig& config,
        	bool enable_spatial_exploration = false, int64_t num_envs = 0,
            const torch::Device& device = torch::Device(torch::kCPU));

        virtual std::shared_ptr<DQNActionInfo> SelectAction(const anet::TensorDict& obs, bool greedy_only,
            const DqnForwardExecutor& forward, std::shared_ptr<anet::RandomGenerator> rnd, const anet::nn::NetworkForwardCallInfo& info,
            anet::ForwardCapture* capture = nullptr,
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
        anet::TensorDict ForwardForAction(const anet::TensorDict& obs, const DqnForwardExecutor& forward, const anet::nn::NetworkForwardCallInfo& info, anet::ForwardCapture* capture) const;
        anet::TensorDict ForwardForActionWithTaus(
            const anet::TensorDict& obs, const torch::Tensor& taus,
            const DqnForwardExecutor& forward, const anet::nn::NetworkForwardCallInfo& info, anet::ForwardCapture* capture,
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
            anet::ForwardCapture* capture,
            const DiagnosticActionEvaluation* evaluation = nullptr, const torch::Tensor& additional_taus = {}) const override;
        void UpdateSchedule(const StepCounts& counts) override;
    };

    class UQEActionPolicy : public ActionPolicy {
    public:
        UQEActionPolicy(const ActionPolicyConfig& config,
            bool enable_spatial_exploration = false,
            int64_t num_envs = 0,
            const torch::Device& device = torch::Device(torch::kCPU));

        std::shared_ptr<DQNActionInfo> SelectAction(const anet::TensorDict& obs, bool greedy_only,
            const DqnForwardExecutor& forward, std::shared_ptr<anet::RandomGenerator> rnd, const anet::nn::NetworkForwardCallInfo& info,
            anet::ForwardCapture* capture,
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
            const DqnForwardExecutor& forward, std::shared_ptr<anet::RandomGenerator> rnd, const anet::nn::NetworkForwardCallInfo& info, anet::ForwardCapture* capture,
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
            anet::ForwardCapture* capture,
            const DiagnosticActionEvaluation* evaluation = nullptr, const torch::Tensor& additional_taus = {}) const override;
        void UpdateSchedule(const StepCounts& counts) override;
    };

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

    torch::Tensor MakeActorQHint(const torch::Tensor& q_values,
        const torch::Tensor& action, const ActorQHintValues& config);

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
        ConfigFullKeyTable actor_keys;
    };

    class Actor : public anet::rl::Actor, public anet::RandomHolder {
    public:
        Actor(std::shared_ptr<ActionPolicy> policy,
            std::unique_ptr<anet::ForwardExecutor> perception,
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
            const anet::nn::NetworkForwardCallInfo& info, int64_t dedicated_tau_count,
            const std::optional<anet::rl::ForcedActions>& forced_actions) const;
    private:
        std::shared_ptr<ActionPolicy> policy_;
        std::unique_ptr<anet::ForwardExecutor> perception_;
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

} // namespace anet::rl::dqn
