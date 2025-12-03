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
#include <optional>
#include <cstdint>
#include <nlohmann/json.hpp>
#include "anet/heat_map.hpp"
#include "anet/tensor_utils.hpp"
#include "anet/tensor_check.hpp"
#include "anet/random.hpp"

namespace anet::rl {

    /*
     ========================

    CPU(ENV)
    │ state
    ▼ ①
    GPU(NN) ←──────────┐
    │                            │
    │ ② produce action          │
    ▼                            │
    CPU(ENV) ←───────③─┘
    │ next_state, reward
    ▼ ⑤
    CPU(ReplayBuffer)
    │ minibatch
    ▼ ⑥
    GPU(NN learner)
    │
    ▼ ⑦ training

     ========================

     ① CPU State → GPU（NN入力）
        ENV が CPU 上で状態を生成
        → Agent に渡すときに GPU へ転送
        → NN へ入力

     ② NNが GPU 上で Action を計算
        Action は GPU Tensor として得られる

     ③ GPU Action → CPU（ENV入力）
        ENV に渡すため Action を CPU に転送

     ④ ENV が next_state / reward を生成（CPU）
        step を実行し、結果も CPU 上

     ⑤ （CPU上の State, Action, Reward, 次State）→ ReplayBuffer(CPU)
        ReplayBuffer は CPU 常駐
        データを CPU のまま保存

     ⑥ ReplayBuffer (CPU) → ミニバッチ → GPU
        ミニバッチをサンプリングしたら GPU へ転送

     ⑦ NN（GPU）で学習
        GPU 上の NN にミニバッチを渡して
        forward → loss → backward → optimizer

    ========================
    */

    // =============================================================
    // RunMode
    // =============================================================

    enum class RunMode { Train, Eval, Eval1, Eval2 };
    inline bool IsTrain(RunMode mode) { return mode == RunMode::Train; }
    inline bool IsEval(RunMode mode) { return mode == RunMode::Eval || mode == RunMode::Eval1 || mode == RunMode::Eval2; }

    // =============================================================
    // Environment 定義クラス
    // =============================================================

    // 観測次元情報
    struct StateDimInfo {
        std::vector<std::int64_t> coords;  ///< 対象の位置情報。 例: {0}, {2}, {0,10,20} など
        float min_value = std::numeric_limits<float>::lowest();  ///< 最小値
        float max_value = std::numeric_limits<float>::max();     ///< 最大値
        std::string name;             ///< 名前（任意）
        std::string description;      ///< 説明（任意）

        nlohmann::json ToJson() const;
        std::string ToString() const;
    };

    // 観測仕様
    struct StateSpec {
        std::vector<std::int64_t> shape;        // 任意次元対応
        std::vector<StateDimInfo> dims;    // 必要な位置だけ登録（配列）
        std::map<std::string, std::string> info;

        std::int64_t CalcFlattenSize() const;
        const StateDimInfo* FindDim(const std::vector<std::int64_t>& coords) const;
        const StateDimInfo* FindDim(std::int64_t flatten_index) const;
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
        std::map<std::string, std::string> info;

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
        StateSpec state_spec;
        ActionSpec action_spec;
        std::pair<float, float> reward_range;
        std::map<std::string, std::string> info;

        /// @todo RewardSpec

        nlohmann::json ToJson() const;
        std::string ToString() const;
    };

    struct BatchEnvSpec {
        int batch_size;
        int num_threads;

        nlohmann::json ToJson() const;
        std::string ToString() const;
    };

    // ------------------------------------------------------------
    // 実データ
    // ------------------------------------------------------------

    // 「Batch～」はN環境対応版の意味

    // 状態
    struct BatchState {
        torch::Tensor obs;              ///< 行動前の観測 (N,state_dim) kFloat32
        torch::Tensor done;             ///< 自然終端     (N) kBool
        torch::Tensor truncated;        ///< 人工終端     (N) kBool
        torch::Tensor episode_start;    ///< reset直後    (N) kBool

        BatchState Clone() const {
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
        torch::Tensor action;       ///< 実際に選択された行動値      (N, action_dim...) kFloat32 or kInt64
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
        torch::Tensor reward;       ///< 報酬          (N) kFloat32
        BatchState next_state;      ///< 遷移後の観測  (N, state_dim...)
        BatchState continue_state;  ///< 実行継続用（Reset 後の状態も含む）

        BatchStepResult to(torch::Device device) const {
            return { reward.to(device), next_state.to(device), continue_state.to(device)};
        }
        std::string ToString() const;
    };

    using MetricsMap = std::unordered_map<std::string, float>;

    class BatchUpdateResult : public DataExporter {
    public:
        virtual MetricsMap GetMetricsMap() const = 0;
        virtual ~BatchUpdateResult() = default;
    };

    // ------------------------------------------------------------
    // 経験情報関連
    // ------------------------------------------------------------

    struct SingleState {
        torch::Tensor obs;          // (state_dim,...)
        bool done;
        bool truncated;
        bool episode_start;

        BatchState toBatchState() const{
            return {
                obs.unsqueeze(0),
                torch::tensor(done, torch::kBool).unsqueeze(0),
                torch::tensor(truncated, torch::kBool).unsqueeze(0),
                torch::tensor(episode_start, torch::kBool).unsqueeze(0),
            };
        }

        /// 状態テンソルを 1D に変換する
        torch::Tensor Flatten() const {
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

    struct SingleDiscreteActionInfo {
        std::int64_t action;  ///< 実際に選択された行動値      (action_dim...) kFloat32 or kInt64
        bool is_random;       ///< ε-greedy のランダム選択か
        
        std::string ToString() const;
    };

    struct SingleStepResult {
        float reward;              ///< 報酬         
        SingleState next_state;    ///< 遷移後の観測  (state_dim...)

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

    struct BatchExperience : public DataExporter {
        BatchState state;
        BatchActionInfo action;
        torch::Tensor reward;
        BatchState next_state;

        explicit BatchExperience() {}
        BatchExperience(
            const BatchState& state__,
            const BatchActionInfo& action__,
            const torch::Tensor& reward__,
            const BatchState& next_state__
        ) : state(state__), action(action__), reward(reward__), next_state(next_state__) { }

        std::optional<float> GetScalar(const std::string& key) const { return std::nullopt; }
        std::optional<torch::Tensor> GetTensor(const std::string& key) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key) const override;

        BatchExperience to(torch::Device d) const;
        std::vector<Experience> ToExperienceList() const;
        std::string ToString() const;
    public:
        static constexpr const char* STATE_OBS = "experience.state.obs";
        static constexpr const char* STATE_DONE = "experience.state.done";
        static constexpr const char* STATE_TRUNCATED = "experience.state.truncated";
        static constexpr const char* STATE_EPISODE_START = "experience.state.episode_start";
        static constexpr const char* ACTION_ACTION = "experience.action.action";
        static constexpr const char* ACTION_IS_RANDOM = "experience.action.is_random";
        static constexpr const char* REWARD = "experience.reward";
        static constexpr const char* NEXT_STATE_OBS = "experience.next_state.obs";
        static constexpr const char* NEXT_STATE_DONE = "experience.next_state.done";
        static constexpr const char* NEXT_STATE_TRUNCATED = "experience.next_state.truncated";
        static constexpr const char* NEXT_STATE_EPISODE_START = "experience.next_state.episode_start";
    };

    // =============================================================
    // Environment
    // =============================================================

    /// not-thread-safe
    class SingleDiscreteEnv {
    public:
        virtual EnvSpec GetSpec() const = 0;
        virtual SingleState Reset(RunMode mode) = 0;
        virtual SingleStepResult Step(int64_t action, RunMode mode) = 0;

        virtual ~SingleDiscreteEnv() = default;
    };

    class SingleDiscreteEnvFactory {
    public:
        virtual std::unique_ptr<SingleDiscreteEnv> Create(std::optional<anet::seed_t> seed = std::nullopt) = 0;

        virtual ~SingleDiscreteEnvFactory() = default;
    };

    class BatchEnv {
    public:
        virtual EnvSpec GetSpec() const = 0;
        virtual BatchEnvSpec GetBatchSpec() const = 0;
        virtual BatchState Reset(RunMode mode = RunMode::Train) = 0;
        virtual BatchStepResult Step(const torch::Tensor& action, RunMode mode = RunMode::Train) = 0;
        virtual BatchState GetState() const = 0;

        virtual ~BatchEnv() = default;
    };

    // =============================================================
    // Agent
    // =============================================================

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

    class Agent : public Runner, public Learner, public DataExporter {
    public:
        virtual TensorFunction GetTensorFunction(const std::string& key) const = 0;
        virtual ~Agent() = default;
    };

    // 環境のステップに同期して更新する Agent 基底クラス
    template<typename ConfigT>
    class StepBasedAgent : public Agent, public anet::RandomHolder {
    public:
        StepBasedAgent(ConfigT config, torch::Device device, std::optional<seed_t> seed = std::nullopt)
            : config_(config), device_(device), RandomHolder(seed) {
        }
        virtual ~StepBasedAgent() = default;

        int GetStepCount() const { return step_count_; }
    protected:
        mutable std::shared_mutex mutex_;
    protected:
        // Resource（Agentが管理すべき領域）
        ConfigT config_;
        torch::Device device_;
    protected:
        // InternalState
        int step_count_ = 0;
    };

    /// @todo 複数ステップ（軌跡）収集後に更新する Agent 基底クラス（PPO, TRPO など）
    //class TrajectoryBasedLearner : public Learner {
    //public:
    //    virtual ~TrajectoryBasedLearner() = default;
    //};

    // =============================================================

    struct PostUpdateEvent {
        int step;
        std::shared_ptr<Agent> agent;
        const BatchExperience& batch_exp;
        std::shared_ptr<const BatchUpdateResult> update_result;
    };

    class PostUpdateObserver {
    public:
        /// @todo PostUpdateEventを組み込み
        virtual void OnPostUpdate(
            int step,
            std::shared_ptr<Agent> agent,
            const BatchExperience& batch_exp,
            std::shared_ptr<const BatchUpdateResult> update_result
        ) = 0;
        virtual ~PostUpdateObserver() = default;
    };

    class Notifier {
    public:
        Notifier();

        void Attach(std::shared_ptr<PostUpdateObserver> observer);
        void Detach(std::shared_ptr<PostUpdateObserver> observer);
        void Notify(
            size_t step,
            std::shared_ptr<Agent> agent,
            const BatchExperience& batch_exp,
            const std::shared_ptr<const BatchUpdateResult>& update_result
        );
    public:
        template <class T, class... Args>
        std::shared_ptr<T> Attach(Args&&... args)
        {
            auto obs = std::make_shared<T>(std::forward<Args>(args)...);
            Attach(obs);
            return obs;
        }
    private:
        std::vector<std::shared_ptr<PostUpdateObserver>> observers_;
    };

    // =============================================================

    /// @todo class NNFactory

} // namespace anet::rl
