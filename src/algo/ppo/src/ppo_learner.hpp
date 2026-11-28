#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>
#include "anet/random.hpp"
#include "anet/tensor_store.hpp"
#include "anet/serialize.hpp"
#include "anet/schedule.hpp"
#include "anet/rl.hpp"
#include "ppo_config.hpp"
#include "ppo_network_model.hpp"

namespace anet::rl::ppo {

    /// optimizer の step 1 回（ミニバッチ 1 つ）の更新結果。
    /// 統計は rollout の単位でまとめて持ち、最初に読まれたときに 1 回だけ CPU へ写す。
    class UpdateResult final : public anet::rl::BatchUpdateResult {
    public:
        /// ミニバッチごとの統計の並び。行動ごとの確率と頻度は、この並びのあとに行動数ぶんずつ続く。
        enum MinibatchStat : size_t {
            kLossTotal, kLossPolicy, kLossValue, kEntropy, kEntropyRatio, kMaxProbMean, kApproxKl, kRatioClipFraction,
            kValueClipFraction, kRatioMean, kRatioStd, kRatioMax, kAdvantageNormAbsMax, kGradNorm, kGradNormShared,
            kGradNormPolicy, kGradNormValue, kGradClipped, kMinibatchStatCount
        };
        /// rollout ごとの統計のうち、device の上で作るものの並び。
        enum RolloutStat : size_t {
            kKlFirstEpoch, kKlLastEpoch, kFirstRatioDeviation, kExplainedVariance, kValueMean, kValueStd,
            kValueTargetMean, kValueTargetStd, kValueErrorStd, kBootstrapValueMean, kAdvantageMean, kAdvantageStd,
            kTerminatedRatio, kTruncatedRatio, kEpisodeEndCount, kRolloutStatCount
        };
        /// rollout ごとの統計のうち、CPU で分かっているものの並び。
        enum HostStat : size_t {
            kLearningRate, kRatioClipRange, kEntropyCoef, kPolicyLag, kEpochsRun, kHostStatCount
        };

        /// 1 つの rollout の更新結果が分け合う統計。
        struct RolloutStatistics {
        public:
            /// device の上の統計を、最初の呼び出しで 1 回だけ CPU へ写して返す。
            const std::vector<float>& GetDeviceValuesOnHost() const;
        public:
            /// device の上の統計を 1 本にまとめたもの。ミニバッチの行（件数 × 幅）のあとに、rollout の値が続く。
            /// 統計の購読が無い rollout では未定義で、device の統計の鍵は NaN を返す。
            torch::Tensor device_values;
            int64_t row_count = 0;    ///< 実行した optimizer の step の数。
            int64_t action_count = 0; ///< 行動数。行動ごとの鍵の範囲を決める。
            std::array<float, kHostStatCount> host_values{};
        private:
            void ReadBackDeviceValues() const;
        private:
            mutable std::once_flag readback_once_;
            mutable std::vector<float> readback_;
        };
    public:
        static constexpr char kActionProbKeyPrefix[] = "action_prob_";
        static constexpr char kActionFreqKeyPrefix[] = "action_freq_";
    public:
        UpdateResult(std::shared_ptr<const RolloutStatistics> statistics, int64_t row);

        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override
            { return std::nullopt; }
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override
            { return std::nullopt; }

        /// 統計の鍵の名前を、並びの順で返す。
        static const std::array<const char*, kMinibatchStatCount>& GetMinibatchStatKeys();
        static const std::array<const char*, kRolloutStatCount>& GetRolloutStatKeys();
        static const std::array<const char*, kHostStatCount>& GetHostStatKeys();
        /// device の上で計算する統計の鍵（ミニバッチ・rollout・行動ごと）かを返す。
        static bool IsDeviceStatKey(const std::string& key);
        /// 1 ミニバッチの行の幅（統計の数に、行動ごとの確率と頻度を足した数）。
        static int64_t GetRowWidth(int64_t action_count) { return kMinibatchStatCount + 2 * action_count; }
    private:
        std::shared_ptr<const RolloutStatistics> statistics_;
        int64_t row_;
    };

    /// rollout をためて、たまったら GAE と複数 epoch のミニバッチ更新で、方策と価値の Network を更新する。
    class Learner final : public anet::rl::Learner {
    public:
        Learner(const LearnerConfig& config, int rollout_length, int64_t lane_count, int64_t action_count,
            std::shared_ptr<std::shared_mutex> mutex, std::shared_ptr<NetworkModel> model, torch::Device device,
            anet::seed_t seed);

        anet::rl::BatchUpdateResultList UpdateFromBatch(const StepCounts& step,
            const anet::rl::BatchExperience& experience) override;

        /// 統計の購読を受け取る。購読の無い統計は計算しない。
        void ConfigureScalarMetricSubscriptions(const std::vector<ScalarMetricSubscription>& subscriptions);
        int64_t Save(anet::OutputArchive& archive) const;
        int64_t Load(anet::InputArchive& archive);
    private:
        struct Minibatch;
        struct MinibatchOutcome;
    private:
        void AppendIntake(const anet::rl::BatchExperience& experience);
        anet::rl::BatchUpdateResultList UpdateFromRollout(const StepCounts& step,
            const anet::rl::BatchExperience& last_experience);
        MinibatchOutcome UpdateMinibatch(const Minibatch& minibatch, float ratio_clip_range, float entropy_coef);
        torch::Tensor ComputeValues(const anet::TensorDict& obs) const;
        void SetLearningRate(double learning_rate);
    private:
        const LearnerConfig config_;
        const int rollout_length_;
        const int64_t lane_count_;
        const int64_t action_count_;
        std::shared_ptr<std::shared_mutex> mutex_; ///< Agent が持つ Network のロック。
        std::shared_ptr<NetworkModel> model_;      ///< Agent が所有する Resource。
        const torch::Device device_;
        anet::RandomGenerator random_; ///< ミニバッチの並べ替えにだけ使う乱数。
        std::vector<torch::Tensor> parameters_; ///< 学習する parameter。
        NetworkModel::ParameterGroups parameter_groups_;
        std::unique_ptr<torch::optim::Optimizer> optimizer_;
        anet::ProfiledValue<double> learning_rate_;
        anet::ProfiledValue<float> ratio_clip_range_;
        anet::ProfiledValue<float> entropy_coef_;
        bool statistics_enabled_ = false;      ///< 統計の購読があるか。
        bool gradient_groups_enabled_ = false; ///< 勾配の内訳の購読があるか。

        // rollout の State。Learner だけが更新する。
        anet::LaneTimeTensorStore device_store_; ///< Network の入力・行動・log 確率・value。Learner の device に置く。
        anet::LaneTimeTensorStore host_store_;   ///< 報酬・終端・打ち切り。CPU に置き、rollout の終わりに 1 回で送る。
        torch::Tensor truncation_values_;        ///< [時刻, lane]。打ち切りの step の、Reset の前の観測の value。無ければ未定義。
        std::vector<uint64_t> policy_versions_;  ///< 時刻ごとの、行動を選んだ方策の世代。
    };

} // namespace anet::rl::ppo
