#pragma once

#include "anet/test/catch.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "anet/config.hpp"
#include "anet/rl.hpp"
#include "anet/env.hpp"
#include "anet/test/nn.hpp"
#include "ppo_agent.hpp"

namespace anet::rl::ppo::test {

    /// テストの設定（MakeConfigData）が持つ Actor キー。
    inline constexpr char kTrainActorKey[] = "train"; ///< Network を共有して抽選する。
    inline constexpr char kEvalActorKey[] = "eval";   ///< 写しを持って、確率が最大の行動を選ぶ。

    /// 何も書かない MetricsLogger を、生きている間だけ置く。Agent は構築時に設定と Network の構造を記録する。
    class ScopedNoopMetricsLogger final {
    public:
        ScopedNoopMetricsLogger();
        ~ScopedNoopMetricsLogger();
        ScopedNoopMetricsLogger(const ScopedNoopMetricsLogger&) = delete;
        ScopedNoopMetricsLogger& operator=(const ScopedNoopMetricsLogger&) = delete;
    };

    /// Agent・Actor・Learner を、Runner と同じ口で作った組。
    struct Rig {
        std::shared_ptr<ScopedNoopMetricsLogger> metrics_logger; ///< Agent より先に作り、あとに壊す。
        anet::rl::EnvSpec env_spec;
        anet::rl::BatchEnvSpec batch_env_spec;
        std::shared_ptr<PPOAgent> agent;
        std::shared_ptr<anet::rl::Actor> actor;
        std::shared_ptr<anet::rl::Learner> learner;
    };

    /// 一本道の課題。位置の one-hot を観測し、右端に着くと報酬 1 で終端、上限の step で打ち切りになる。
    class CorridorEnv {
    public:
        struct StepResult {
            torch::Tensor reward;
            anet::rl::BatchState next_state;     ///< Reset の前の観測と、終端・打ち切りの印。
            anet::rl::BatchState continue_state; ///< 終わった lane を Reset したあとの状態。
            std::vector<float> finished_returns; ///< この step で終わった episode の return。
        };
    public:
        static constexpr int kLength = 8;
        static constexpr int kLimitStep = 20;
        static constexpr int kActionCount = 2; ///< 0 は左、1 は右。
        static constexpr float kStepReward = -0.01f;
        static constexpr float kGoalReward = 1.0f;
        static constexpr float kOptimalReturn = 0.94f; ///< 右へ 7 回進んだときの return（−0.01 × 6 ＋ 1）。
    public:
        explicit CorridorEnv(int lane_count);
        anet::rl::BatchState Reset();
        StepResult Step(const torch::Tensor& actions);
    private:
        torch::Tensor MakeObs() const;
    private:
        std::vector<int> positions_;
        std::vector<int> steps_;
        std::vector<float> returns_;
    };

    /// 一本道の課題を、Train Runner が使う BatchEnv の口で出す。
    class CorridorBatchEnv final : public anet::rl::BatchEnvBase {
    public:
        explicit CorridorBatchEnv(int lane_count);

        anet::rl::EnvSpec GetSpec() const override;
        anet::rl::BatchEnvSpec GetBatchSpec() const override;
        torch::Device GetDevice() const override;
        std::shared_ptr<const anet::rl::BatchResetResult> Reset() override;
        std::shared_ptr<const anet::rl::BatchStepResult> Step(
            std::shared_ptr<anet::rl::BatchActionInfo> action_info) override;
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override
            { return std::nullopt; }
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override
            { return std::nullopt; }
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override
            { return std::nullopt; }
    private:
        CorridorEnv env_;
        const int lane_count_;
    };

    /// 観測キー vector を読む小さな Network の設定を作る。shared_body が真なら、方策と価値が Body を共有する。
    anet::ConfigData MakeConfigData(bool shared_body = false);
    anet::rl::EnvSpec MakeEnvSpec(int64_t obs_size, int action_count);
    Rig MakeRig(const anet::ConfigData& config_data, const anet::rl::EnvSpec& env_spec, int lane_count,
        torch::Device device, anet::seed_t seed = 123);
    /// vector の観測 [lane, 要素] から、終端でも開始でもない状態を作る。
    anet::rl::BatchState MakeState(const torch::Tensor& vector);
    /// vector の観測 [lane, 要素] と、lane ごとの終端・打ち切りの印から、step のあとの状態を作る。
    anet::rl::BatchState MakeState(const torch::Tensor& vector, const std::vector<bool>& done,
        const std::vector<bool>& truncated);
    /// Actor が今の Network で出す value [lane] を、CPU で返す。
    torch::Tensor ReadValues(const anet::rl::Actor& actor, const torch::Tensor& vector);
    /// 全部の lane の行動を強制する、行動選択の要求を作る。actions は [lane] の int64。
    anet::rl::MakeActionOptions MakeForcedActionOptions(const torch::Tensor& actions);
    /// 行動を強制して、Actor が今の Network で出す、その行動の log 確率 [lane] を、CPU で返す。
    torch::Tensor ReadLogProbs(const anet::rl::Actor& actor, const torch::Tensor& vector, const torch::Tensor& actions);
    /// 更新結果の統計の鍵の全部（行動ごとの鍵を含む）を、Train の LEARN の購読として作る。
    std::vector<anet::rl::ScalarMetricSubscription> MakeAllSubscriptions(int action_count);
    /// Runner の 1 step と同じ順で、行動を選び、与えた結果で経験を作って Learner に渡す。
    anet::rl::BatchUpdateResultList StepOnce(Rig& rig, const anet::rl::BatchState& state,
        const torch::Tensor& reward, const anet::rl::BatchState& next_state,
        const anet::rl::StepCounts& counts = {}, const anet::rl::MakeActionOptions& options = {});

} // namespace anet::rl::ppo::test
