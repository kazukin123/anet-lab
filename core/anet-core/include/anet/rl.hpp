#pragma once
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <random>
#include <map>
#include <unordered_map>
#include <cstdint>
#include <nlohmann/json.hpp>
#include "anet/heat_map.hpp"
#include "anet/tensor_utils.hpp"
#include "anet/tensor_check.hpp"
#include "anet/random.hpp"

namespace anet::rl {

    /*
        Environment
          ↓  (BatchState)
        Runer(Agent)
          ↓ （BatchActionInfo）
        Environment
          ↓  (BatchReward, BatchNextState) → BatchStepResult
        Trainer
          ↓  Batch(s, a ,r, s) ⇒ vector<Experience> ⇒ ReplayBuffer
        Sampler(Agent)
          ↓  ReplayBuffer ⇒ Update
        Learner(Agent)
    */

    // =============================================================
    // RunMode
    // =============================================================

    enum class RunMode { Train, Eval1, Eval2 };
    inline bool IsTrain(RunMode mode) { return mode == RunMode::Train; }
    inline bool IsEval(RunMode mode) { return mode == RunMode::Eval1 || mode == RunMode::Eval2; }

    // =============================================================
    // Environment 定義クラス
    // =============================================================

    // 観測次元情報
    struct StateDimInfo {
        std::vector<int64_t> coords;  ///< 対象の位置情報。 例: {0}, {2}, {0,10,20} など
        float min_value = std::numeric_limits<float>::lowest();  ///< 最小値
        float max_value = std::numeric_limits<float>::max();     ///< 最大値
        std::string name;             ///< 名前（任意）
        std::string description;      ///< 説明（任意）

        nlohmann::json ToJson() const;
        std::string ToString() const;
    };

    // 観測仕様
    struct StateSpec {
        std::vector<int64_t> shape;        // 任意次元対応
        std::vector<StateDimInfo> dims;    // 必要な位置だけ登録（配列）
        std::map<std::string, std::string> options;

        int64_t CalcStateDim() const;
        const StateDimInfo* FindDim(const std::vector<int64_t>& coords) const;
        bool MatchesShape(const torch::Tensor& obs) const;
        bool MatchesRange(const torch::Tensor& obs) const;
        bool MatchesRangeFlat(const torch::Tensor& flat_obs) const;
        nlohmann::json ToJson() const;
        std::string ToString() const;
    };

    // 連続アクション次元情報
    struct ActionDimInfo {
        float min_value;
        float max_value;
        std::string name;
        std::string description;

        nlohmann::json ToJson() const;
        std::string ToString() const;
    };

    // 行動仕様（離散/連続を一本化）
    struct ActionSpec {
        bool is_discrete;
        std::vector<std::string> value_labels; // 離散アクションの場合のみ使用
        std::vector<ActionDimInfo> dims; // 連続アクションの場合のみ使用
        std::map<std::string, std::string> options;

        int ActionCount() const {
            if (is_discrete) {
                return (int)value_labels.size();
            }

            // continuous
            ANET_ASSERT_MSG(!dims.empty(),
                "ActionSpec::ActionCount(): continuous action must define dims.");
            return (int)dims.size();
        }

        nlohmann::json ToJson() const;
        std::string ToString() const;
    };

    // 環境仕様
    struct EnvSpec {
        StateSpec state;
        ActionSpec action;
        std::pair<float, float> reward_range;
        std::map<std::string, std::string> options;

        nlohmann::json ToJson() const;
        std::string ToString() const;
    };

    // ------------------------------------------------------------
    // 実データ
    // ------------------------------------------------------------

    // 状態
    struct BatchState {
        torch::Tensor obs;              ///< 行動前の観測 (N,state_dim) kFloat32
        torch::Tensor done;             ///< 自然終端     (N) kBool
        torch::Tensor truncated;        ///< 人工終端     (N) kBool
        torch::Tensor episode_start;    ///< reset直後    (N) kBool

        BatchState Clone() {
            return { obs.clone(), done.clone(), truncated.clone(), episode_start.clone() };
        }

        /// obs を (N, state_dim) にフラット化
        BatchState Flatten() const {
            ANET_CHECK_DTYPE(obs, torch::kFloat32);
            int64_t N = obs.size(0);
            int64_t flat_dim = obs.numel() / N;
            auto f = obs.reshape({ N, flat_dim });
            return { f, done, truncated, episode_start };
        }

        BatchState to(torch::Device device) const {
            ANET_CHECK_SHAPE(obs, { ANET_SHAPE_ANY, ANET_SHAPE_ANY });
            ANET_CHECK_SHAPE(done, { ANET_SHAPE_ANY });
            ANET_CHECK_SHAPE(truncated, { ANET_SHAPE_ANY });
            ANET_CHECK_SHAPE(episode_start, { ANET_SHAPE_ANY });
            ANET_CHECK_DTYPE(obs, torch::kFloat32);
            ANET_CHECK_DTYPE(done, torch::kBool);
            ANET_CHECK_DTYPE(truncated, torch::kBool);
            ANET_CHECK_DTYPE(episode_start, torch::kBool);

            return {
                obs.to(device), done.to(device),
                truncated.to(device), episode_start.to(device)
            };
        }
        bool IsDone() const {   // 1環境専用
            ANET_CHECK_SHAPE(done, { ANET_SHAPE_ANY });
            ANET_ASSERT(done.size(0) == 1);
            return done.item<bool>();
        }
        bool IsTruncated() const {  // 1環境専用
            ANET_CHECK_SHAPE(done, { ANET_SHAPE_ANY });
            ANET_ASSERT(done.size(0) == 1);
            return truncated.item<bool>();
        }
        bool IsEpisodeStart() const {   // 1環境専用
            ANET_CHECK_SHAPE(episode_start, { ANET_SHAPE_ANY });
            ANET_ASSERT(episode_start.size(0) == 1);
            return episode_start.item<bool>();
        }
        std::string ToString() const;
    };

    // 行動選択時のメタ情報
    struct BatchActionInfo {
        torch::Tensor action;       ///< 実際に選択された行動値      (N, action_dim) kFloat32 or kInt64
        torch::Tensor is_random;    ///< ε-greedy のランダム選択か  (N) kBool

        BatchActionInfo to(torch::Device device) const {
            ANET_CHECK_SHAPE(action, { ANET_SHAPE_ANY, ANET_SHAPE_ANY });
            ANET_CHECK_SHAPE(is_random, { ANET_SHAPE_ANY });
            ANET_ASSERT(action.dtype() == torch::kFloat32 || action.dtype() == torch::kInt64);
            ANET_CHECK_DTYPE(is_random, torch::kBool);

            return BatchActionInfo{ action.to(device), is_random.to(device) };
        }
        std::string ToString() const;
    };

    // Env::DoStep() の結果
    struct BatchStepResult {
        BatchState next_state;           ///< 遷移後の観測  (N,)
        torch::Tensor reward;       ///< 報酬          (N) kFloat32

        BatchStepResult to(torch::Device device) const {
            return { next_state.to(device), reward.to(device) };
        }
        std::string ToString() const;
    };

    // ------------------------------------------------------------
    // 経験情報関連
    // ------------------------------------------------------------

    struct SingleState {
        torch::Tensor obs;          // (state_dim,...)
        bool done;
        bool truncated;
        bool episode_start;

        /// 状態テンソルを 1D に変換する
        torch::Tensor Flattened() const {
            ANET_CHECK_DTYPE(obs, torch::kFloat32);
            return obs.reshape({ obs.numel() });
        }
        SingleState to(torch::Device device) const {
            ANET_CHECK_SHAPE(obs, { ANET_SHAPE_ANY });
            ANET_CHECK_DTYPE(obs, torch::kFloat32);
            return { obs.to(device), done, truncated, episode_start };
        }
        std::string ToString() const;
    };

    // 経験情報（Updateの入力情報、ReplayBufferに入る）
    struct Experience {
        SingleState state;
        torch::Tensor action;       // (action_dim)
        float reward;
        SingleState next_state;

        Experience to(torch::Device device) const {
            ANET_CHECK_SHAPE(action, { });
            ANET_ASSERT(action.dtype() == torch::kInt64 || action.dtype() == torch::kFloat32);
            return {
                state.to(device), action.to(device), reward, next_state.to(device)
            };
        }
        std::string ToString() const;
    };

    struct BatchExperience {
        BatchState state;
        BatchActionInfo action;
        torch::Tensor reward;
        BatchState next_state;

        BatchExperience to(torch::Device d) const {
            return { state.to(d), action.to(d), reward.to(d), next_state.to(d) };
        }
        std::vector<Experience> ToExperienceList() const;
        std::string ToString() const;
    };

    // =============================================================
    // Environment
    // =============================================================

    class BatchEnvironment {
    public:
        virtual EnvSpec GetSpec() const = 0;
        virtual BatchState Reset(RunMode mode = RunMode::Train) = 0;
        virtual BatchStepResult DoStep(const torch::Tensor& action, RunMode mode = RunMode::Train) = 0;
        virtual BatchState GetState() const = 0;

        virtual ~BatchEnvironment() = default;
    };

    // =============================================================
    // Agent
    // =============================================================

    using MetricsMap = std::unordered_map<std::string, float>;

    class BatchUpdateResult {
    public:
        virtual MetricsMap GetMetricsMap() const = 0;
        virtual ~BatchUpdateResult() = default;
    };

    class Runner {
    public:
        virtual BatchActionInfo MakeAction(const anet::rl::BatchState& state, RunMode mode = RunMode::Train) = 0;
        virtual ~Runner() = default;
    };

    class Sampler {
    public:
        virtual void ObserveFirst(const BatchStepResult& step_result) = 0;
        virtual void Observe(const BatchExperience& exprience) = 0;
        virtual ~Sampler() = default;
    };

    class Learner {
    public:
        virtual std::shared_ptr<const BatchUpdateResult> UpdateFromBatch(const BatchExperience& expriences) = 0;
        virtual ~Learner() = default;
    };

    class PostUpdateObserver {
    public:
        virtual void OnPostUpdate(
            std::shared_ptr<const BatchUpdateResult> result,
            const BatchExperience& expriences,
            size_t step
        ) = 0;
        virtual ~PostUpdateObserver() = default;
    };

    class Agent : public Runner, public Learner {
    public:
        virtual ~Agent() = default;
    };

    // 環境のステップに同期して更新する Agent 基底クラス
    template<typename ConfigT>
    class StepBasedAgent : public Agent, public anet::RandomHolder {
    public:
        StepBasedAgent(ConfigT config, torch::Device device, std::shared_ptr<anet::RandomGenerator> rnd = nullptr)
            : config_(config), device_(device), RandomHolder(rnd) { }
        virtual ~StepBasedAgent() = default;

        size_t GetStepCount() const { return step_count_; }
    protected:
        std::shared_mutex mutex_;
    protected:
        // Resource（Agentが管理すべき領域）
        ConfigT config_;
        torch::Device device_;
    protected:
        // InternalState
        size_t step_count_ = 0;
    };

    // 複数ステップ（軌跡）収集後に更新する Agent 基底クラス（PPO, TRPO など）
    class TrajectoryBasedLearner : public Learner {
    public:
        // TODO: define
        virtual ~TrajectoryBasedLearner() = default;
    };

    class RunnerFactory {
    public:
        virtual std::shared_ptr<Runner> CreateRunner() = 0;
        virtual ~RunnerFactory() = default;
    };

    class Notifier {
    public:
        void AddObserver(PostUpdateObserver* obs) {
            observers_.push_back(obs);
        }

        void Notify(
            const std::shared_ptr<const BatchUpdateResult>& result,
            const BatchExperience& expriences, size_t step)
        {
            for (PostUpdateObserver* o : observers_) {
                o->OnPostUpdate(result, expriences, step);
            }
        }
    private:
        std::vector<PostUpdateObserver*> observers_;
    };

    // =============================================================
    // HeatMap 関連
    // =============================================================

    //std::unique_ptr<HeatMap> MakeStateHeatMapPtr(
    //    const StateSpaceInfo& info,
    //    int idx_x,
    //    int idx_y,
    //    int width = 256,
    //    int height = 256,
    //    size_t max_points = 10000,
    //    uint32_t flags = HM_Default);

    //std::unique_ptr<TimeHeatMap> MakeStateTimeHeatMapPtr(
    //    const StateSpaceInfo& info,
    //    int idx_x,
    //    int width = 256,
    //    int height = 2560,
    //    size_t max_points = 0,
    //    uint32_t flags = HM_Default,
    //    TimeFrameMode mode = TimeFrameMode::Unlimited);

} // namespace anet::rl
