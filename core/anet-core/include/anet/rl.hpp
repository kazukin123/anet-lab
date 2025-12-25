#pragma once
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <stdexcept>
#include <optional>
#include <cstdint>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include "anet/tensor_check.hpp"
#include "anet/random.hpp"
#include "anet/config.hpp"
#include "anet/log.hpp"
#include "anet/tensor_util.hpp"

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
    // Step / StepAxis
    // =============================================================

    using step_t = uint64_t;

    enum class StepAxis {
        TRAIN,
        EXP,
        UPDATE, ///< @todo 不要？
        LEARN,
        EPISODE,
        SIM
    };

    struct StepCounts {
        step_t train_step = 0;
        step_t exp_step = 0;
        step_t update_step = 0;
        step_t learn_step = 0;
        step_t episode_count = 0;
        step_t sim_step = 0;

        step_t& GetByAxis(StepAxis axis)
        {
            switch (axis) {
            case StepAxis::TRAIN: return train_step;
            case StepAxis::EXP: return exp_step;
            case StepAxis::UPDATE: return update_step;
            case StepAxis::LEARN: return learn_step;
            case StepAxis::EPISODE: return episode_count;
            case StepAxis::SIM: return sim_step;
            }
            throw std::runtime_error("StepCounts::GetByAxis(): invalid StepAxis");
		}

        step_t GetByAxis(StepAxis axis) const
        {
            switch (axis) {
            case StepAxis::TRAIN: return train_step;
            case StepAxis::EXP: return exp_step;
            case StepAxis::UPDATE: return update_step;
            case StepAxis::LEARN: return learn_step;
            case StepAxis::EPISODE: return episode_count;
            case StepAxis::SIM: return sim_step;
            }
            throw std::runtime_error("StepCounts::GetByAxis(): invalid StepAxis");
        }
    };

    static std::string toString(StepAxis step_axis)
    {
        switch (step_axis) {
        case StepAxis::TRAIN: return "train";
        case StepAxis::EXP: return "exp";
        case StepAxis::UPDATE: return "update";
        case StepAxis::LEARN: return "learn";
        case StepAxis::EPISODE: return "episode";
        case StepAxis::SIM: return "sim";
        }
        ANET_ASSERT_MSG(false, "Invalid StepAxis.");
        return "";
    }

    // =============================================================
    // RunMode
    // =============================================================

    enum class RunMode { Train, Eval, Eval1, Eval2 };

    inline bool IsTrain(RunMode mode) { return mode == RunMode::Train; }
    inline bool IsEval(RunMode mode) { return mode == RunMode::Eval || mode == RunMode::Eval1 || mode == RunMode::Eval2; }
    inline std::string ToString(RunMode mode) {
        switch (mode) {
        case RunMode::Train: return "Train";
        case RunMode::Eval: return "Eval";
        case RunMode::Eval1: return "Eval1";
        case RunMode::Eval2: return "Eval2";
        }
        ANET_ASSERT_MSG(false, "Invalid RunMode.");
        return "";
    }

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

        std::int64_t CalcFlattenDim() const;
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

        int GetNumActions() const {
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

    // =============================================================
    // Single系データ
    // =============================================================

    struct SingleState {
        torch::Tensor obs;          // (state_dim,...)
        bool done;
        bool truncated;
        bool episode_start;

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

        /// @todo is_random廃止？

        std::string ToString() const;
    };

    using AuxData = std::unordered_map<std::string, torch::Tensor>; ///< 任意の追加情報（UI描画用の非可観測情報を含む）


    class SingleEnvResult {
    public:
        virtual AuxData GetAuxData() const { return AuxData(); }
        virtual ~SingleEnvResult() = default;
    };

    class SingleResetResult : virtual public SingleEnvResult {
    public:
        explicit SingleResetResult(SingleState state_in)
            : state(std::move(state_in)) {}

        virtual ~SingleResetResult() = default;
    public:
        SingleState state;
    };

    class SingleStepResult : virtual public SingleEnvResult {
    public:
        SingleStepResult(float reward_in, SingleState next_state_in)
            : reward(reward_in), next_state(std::move(next_state_in)) { } 

        virtual ~SingleStepResult() = default;
        std::string ToString() const;
    public:
        float reward;              ///< 報酬         
        SingleState next_state;    ///< 遷移後の観測  (state_dim...)
    };

    class DefaultSingleStepResult : public SingleStepResult {
    public:
        DefaultSingleStepResult(float reward_in, const SingleState& next_state_in)
            : SingleStepResult(reward_in, next_state_in) {}
        AuxData GetAuxData() const override { return AuxData(); }

        virtual ~DefaultSingleStepResult() = default;
    };

    // 経験情報（Updateの入力情報、ReplayBufferに入る）
    struct SingleExperience {
        SingleState state;
        torch::Tensor action;       // (action_dim)
        float reward;
        SingleState next_state;

        SingleExperience to(torch::Device device) const {
            ANET_CHECK_SHAPE(action, { });
            ANET_ASSERT(action.dtype() == torch::kInt64 || action.dtype() == torch::kFloat32);
            return {
                state.to(device), action.to(device), reward, next_state.to(device)
            };
        }
        std::string ToString() const;
    };

    // =============================================================
    // Batch系データ
    // =============================================================

    // 「Batch～」はN環境対応版の意味

    // 状態
    struct BatchState {
        torch::Tensor obs;              ///< 行動前の観測 (N,state_dim) kFloat32
        torch::Tensor done;             ///< 自然終端     (N) kBool
        torch::Tensor truncated;        ///< 人工終端     (N) kBool
        torch::Tensor episode_start;    ///< reset直後    (N) kBool

        BatchState() {}

        BatchState(torch::Tensor o, torch::Tensor d, torch::Tensor t, torch::Tensor e)
            : obs(std::move(o)), done(std::move(d)), truncated(std::move(t)), episode_start(std::move(e)) { }

        BatchState Clone() const {
            return { obs.clone(), done.clone(), truncated.clone(), episode_start.clone() };
        }

        SingleState GetSingle(int64_t index) const
        {
            return {
                obs[index],
                done[index].item<bool>(),
                truncated[index].item<bool>(),
                episode_start[index].item<bool>()
            };
        }

        /// obs を (N, state_dim) にフラット化
        BatchState Flatten() const {
            ANET_CHECK_DTYPE(obs, torch::kFloat32);
            int64_t N = obs.size(0);
            int64_t flat_dim = obs.numel() / N;
            auto f = obs.reshape({ N, flat_dim });
            return { f, done, truncated, episode_start };
        }

        BatchState To(torch::Device device, bool non_blocking = true) const {
            ANET_CHECK_SHAPE(obs, { ANET_SHAPE_ANY, ANET_SHAPE_ANY });
            ANET_CHECK_SHAPE(done, { ANET_SHAPE_ANY });
            ANET_CHECK_SHAPE(truncated, { ANET_SHAPE_ANY });
            ANET_CHECK_SHAPE(episode_start, { ANET_SHAPE_ANY });
            ANET_CHECK_DTYPE(obs, torch::kFloat32);
            ANET_CHECK_DTYPE(done, torch::kBool);
            ANET_CHECK_DTYPE(truncated, torch::kBool);
            ANET_CHECK_DTYPE(episode_start, torch::kBool);

            return {
                obs.to(device, non_blocking), done.to(device, non_blocking),
                truncated.to(device,non_blocking), episode_start.to(device, non_blocking)
            };
        }
        bool IsDone() const {
            ANET_CHECK_SHAPE(done, { ANET_SHAPE_ANY });
            ANET_ASSERT(done.size(0) >= 1);
            return done[0].item<bool>();
        }
        bool IsTruncated() const {
            ANET_CHECK_SHAPE(done, { ANET_SHAPE_ANY });
            ANET_ASSERT(done.size(0) >= 1);
            return truncated[0].item<bool>();
        }
        bool IsEpisodeStart() const {
            ANET_CHECK_SHAPE(episode_start, { ANET_SHAPE_ANY });
            ANET_ASSERT(episode_start.size(0) >= 1);
            return episode_start[0].item<bool>();
        }
        std::string ToString() const;
    };

    // 行動選択時の情報
    struct BatchActionInfo {
    public:
        BatchActionInfo() {}

        BatchActionInfo(const torch::Tensor action, const AuxData& aux)
            : action_cpu_(action.device().is_cpu() ? std::move(action) : torch::Tensor())
            , aux_(aux)
        {
            if (action.device().is_cuda())
                gpu_ = std::pair(action.device(), std::move(action));
        }

        BatchActionInfo(const torch::Tensor action)
            : BatchActionInfo(action, AuxData{})
        {
        }

        torch::Tensor GetAction() const
        {
            if (action_cpu_.defined())
                return action_cpu_;
            if (gpu_.has_value())
                return gpu_->second;
            return torch::Tensor();
        }

        torch::Tensor GetAction(torch::Device device) const
        {
        	/// @todo GPU→CPUのno_blockingは同期が必要で面倒なので使わない
        	
            // --- CPU requested ---
            if (device.is_cpu()) {
                if (!action_cpu_.defined()) {
                    if (!gpu_)
                        throw std::logic_error("BatchActionInfo: no tensor available to create CPU action");
                    action_cpu_ = gpu_->second.to(torch::kCPU);
                }
                return action_cpu_;
            }

            // --- GPU requested ---
            if (!device.is_cuda()) {
                throw std::logic_error("BatchActionInfo: unsupported device type");
            }
            if (!gpu_) {
                if (!action_cpu_.defined())
                    throw std::logic_error("BatchActionInfo: no tensor available to create GPU action");
                gpu_ = std::make_pair(device, action_cpu_.to(device));
                return gpu_->second;
            }
            if (gpu_->first != device) {
                throw std::logic_error(
                    "BatchActionInfo: GPU device mismatch (cached="
                    + gpu_->first.str() + ", requested=" + device.str() + ")"
                );
            }

            return gpu_->second;
        }
        BatchActionInfo To(torch::Device device) const {
            return BatchActionInfo{ GetAction(device), aux_};
        }
        const AuxData& GetAuxData() const { return aux_; }
        AuxData& GetAuxData() { return aux_; }

        std::string ToString() const;
    private:
        mutable torch::Tensor action_cpu_;
        mutable std::optional<std::pair<torch::Device, torch::Tensor>> gpu_;
        AuxData aux_;
    };

    class BatchEnvResult {
    public:
        virtual std::vector<AuxData> GetAuxDataList(int env_index = -1) const = 0;
        virtual ~BatchEnvResult() = default;
    };

    class BatchResetResult : virtual public BatchEnvResult {
    public:
        BatchResetResult(BatchState state_in) : state(std::move(state_in)) { }
        std::string ToString() const;

        virtual ~BatchResetResult() = default;
    public:
        BatchState state;      ///<初期の観測情報  (N, state_dim...)
    };

    class BatchStepResult : virtual public BatchEnvResult {
    public:
        BatchStepResult(torch::Tensor reward_in, BatchState next_state_in, BatchState continue_state_in, uint32_t n_transitions_in, uint32_t n_done_in);
        std::string ToString() const;

        virtual ~BatchStepResult() = default;
    public:
        torch::Tensor reward;       ///< 報酬          (N) kFloat32
        BatchState next_state;      ///< 遷移後の観測  (N, state_dim...)
        BatchState continue_state;  ///< 実行継続用（Reset 後の状態も含む）
        uint32_t n_transitions;     ///< 今回のStepにおける状態遷移カウント数
        uint32_t n_done;            ///< 今回のStepにおけるエピソード終了カウント数
    };

    using MetricsMap = std::unordered_map<std::string, float>;

    class BatchUpdateResult : public DataExporter {
    public:
        BatchUpdateResult(uint32_t learn_step_diff) : learn_step_diff_(learn_step_diff) {}

        uint32_t GetLearnStepDiff() const { return learn_step_diff_; }

        virtual ~BatchUpdateResult() = default;
    protected:
        void SetLearnStepDiff(uint32_t diff) { learn_step_diff_ = diff; }
    protected:
        uint32_t learn_step_diff_;   ///< この batch で増えた learn_step数
    };

    struct BatchExperience : public DataExporter {
        BatchState state;
        BatchActionInfo action;
        torch::Tensor reward;
        BatchState next_state;

        BatchExperience() {}
        BatchExperience(
            const BatchState& state__,
            const BatchActionInfo& action__,
            torch::Tensor reward__,
            const BatchState& next_state__
        ) : state(state__), action(action__), reward(std::move(reward__)), next_state(next_state__) { }

        std::optional<float> GetScalar(const std::string& key, int index = -1) const { return std::nullopt; }
        std::optional<torch::Tensor> GetTensor(const std::string& key, int index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int index = -1) const override;

        BatchExperience To(torch::Device d, bool non_blocking = true) const;
        std::vector<SingleExperience> ToExperienceList() const;
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
    class SingleDiscreteEnv : public DataExporter {
    public:
        virtual EnvSpec GetSpec() const = 0;
        virtual std::shared_ptr<const SingleResetResult> Reset(RunMode mode) = 0;
        virtual std::shared_ptr<const SingleStepResult> Step(int64_t action, RunMode mode) = 0;

        virtual ~SingleDiscreteEnv() = default;
    };

    class SingleDiscreteEnvFactory {
    public:
        virtual std::shared_ptr<SingleDiscreteEnv> CreateSingleEnv(
            const anet::ConfigData& config_data,
            const torch::Device& device,
            std::optional<anet::seed_t> seed = std::nullopt) = 0;
        virtual std::string GetTargetEnvClassId() const = 0;

        virtual ~SingleDiscreteEnvFactory() = default;
    };

    class BatchEnv : public DataExporter {
    public:
        virtual EnvSpec GetSpec() const = 0;
        virtual BatchEnvSpec GetBatchSpec() const = 0;

        virtual std::shared_ptr<const BatchResetResult> Reset(RunMode mode = RunMode::Train) = 0;    ///< BatchStateは使い回されるので必要に応じてDeepCopy必須
        virtual std::shared_ptr<const BatchStepResult> Step(const BatchActionInfo& action_info, RunMode mode = RunMode::Train) = 0; ///< BatchStepResultは使い回されるので必要に応じてDeepCopy必須
        //virtual std::shared_ptr<BatchEnv> Clone() const = 0;

        virtual ~BatchEnv() = default;
    };

    class BatchEnvFactory {
    public:
        virtual std::shared_ptr<BatchEnv> CreateBatchEnv(std::optional<seed_t> seed = std::nullopt, int batch_size = -1) = 0;	///< batch_size=-1でbatch_size自動
        virtual ~BatchEnvFactory() = default;
    };

    // =============================================================
    // Agent
    // =============================================================

    class Runner;

    class Sampler {
    public:
        virtual void ObserveFirst(const BatchStepResult& step_result) = 0;
        virtual void Observe(const BatchExperience& exprience) = 0;
        virtual ~Sampler() = default;
    };

    class ActionPolicy {
    public:
        virtual BatchActionInfo MakeAction(
            const StepCounts& counts, const anet::rl::BatchState& state, RunMode mode = RunMode::Train) const = 0;
        virtual ~ActionPolicy() = default;
    };

    class Learner {
    public:
        virtual std::shared_ptr<const BatchUpdateResult> UpdateFromBatch(
            const StepCounts& step, const BatchExperience& expriences, const Runner& trainer
        ) = 0;  ///< @todo 戻りをstd::vector化?
        virtual ~Learner() = default;
    };

    class Agent : public ActionPolicy, public Learner, public DataExporter {
    public:
        virtual std::optional<anet::TensorFunction> GetTensorFunction(const std::string& key) = 0;
        virtual ~Agent() = default;
    };

    // =============================================================
    // ReplayBuffer 
    // =============================================================

    /// ReplayBufferから取り出したB個のサンプルデータ（「N環境」ではなく「Bサンプル」である事に注意）
    struct ExperienceSamples {
        torch::Tensor obs;            // (B, state_dim...)
        torch::Tensor actions;        // (B, action_dim...)
        torch::Tensor target_values;  // (B,)
        struct {
            torch::Tensor obs;             // (B, state_dim...)
            //torch::Tensor dones;         // (B,)
            //torch::Tensor truncateds;    // (B,)
            //torch::Tensor episode_start; // (B,)
            torch::Tensor terminals;       // (B,) bool
        } next_states;
        torch::Tensor n_steps;       // (B,) int

        torch::Tensor indices;          // (B,) kInt64
        torch::Tensor sampling_prob;    // (B,) kFloat32
        torch::Tensor is_weights;       // (B,) kFloat32

        ExperienceSamples FlattenStates() const;
        ExperienceSamples To(torch::Device device, bool non_blocking) const;
        std::string ToString() const;
    };

    class ReplayPriorityController : public anet::DataExporter {
    public:
        virtual void UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities) = 0;

        ~ReplayPriorityController() = default;
    };

    class ReplayBuffer : public ReplayPriorityController {
    public:
        virtual void Push(const BatchExperience& batch_exp) = 0;
        virtual void Push(const std::vector<SingleExperience>& exps) = 0;
        virtual ExperienceSamples Sample(int64_t minibatch_size, torch::Device device, float beta = -1) const = 0;
        virtual int64_t Size() const = 0;

        virtual ~ReplayBuffer() = default;
    public:
        static constexpr const char* kKeyPrefix = "replaybuffer.";

        static constexpr const char* STATE_OBS = "replaybuffer.storage.state";
        static constexpr const char* ACTION = "replaybuffer.storage.action";
        static constexpr const char* REWARD = "replaybuffer.storage.reward";
        static constexpr const char* NEXT_STATE_OBS = "replaybuffer.storage.next_state";
        static constexpr const char* NEXT_STATE_TERMINAL = "replaybuffer.storage.terminal";
        static constexpr const char* N_STEP = "replaybuffer.storage.n_step";

        static constexpr const char* PER_TOTAL = "replaybuffer.per.total";
        static constexpr const char* PER_VALUES = "replaybuffer.per.values";
        static constexpr const char* PER_DIST = "replaybuffer.per.distribution";
    };

    // =============================================================
    // Observer
    // =============================================================

    enum class EventType {
        TRAIN,
        LEARN
        /// @todo EPISODE_END（ENV由来）を追加
    };

    enum class EventField {
        BATCH_EXPERIENCE,
        AGENT,
		BATCH_UPDATE_RESULT,
        RUNNER
    };

    struct BeforeStepEvent {
        const Runner& runner;
        const StepCounts counts;
        std::shared_ptr<const Agent> agent;
        std::shared_ptr<const BatchEnv> env;
    };

    struct UpdateEvent {
        const BatchExperience& batch_exp;
		const Runner& runner;
        const StepCounts counts;
        std::shared_ptr<const Agent> agent;
        std::shared_ptr<const BatchUpdateResult> update_result;
    };

    struct TrainEvent : public UpdateEvent {
        std::shared_ptr<const BatchEnv> env;
        std::shared_ptr<const BatchStepResult> batch_step_result;
        const BatchActionInfo& action_info;
    };

    struct LearnEvent : public UpdateEvent {
    };

    class BeforeStepObserver {
    public:
        virtual void OnBeforeStep(const BeforeStepEvent& event) = 0;
        virtual std::string ToString() const = 0;
        virtual ~BeforeStepObserver() = default;
    };

    class TrainObserver {
    public:
        virtual void OnTrain(const TrainEvent& event) = 0;
        virtual std::string ToString() const = 0;
        virtual ~TrainObserver() = default;
    };

    class LearnObserver {
    public:
        virtual void OnLearn(const LearnEvent& event) = 0;
        virtual std::string ToString() const = 0;
        virtual ~LearnObserver() = default;
    };

    // =============================================================
    // Notifier
    // =============================================================

    class Notifier {
    public:
        Notifier();

        std::shared_ptr<BeforeStepObserver> Attach(std::shared_ptr<BeforeStepObserver> observer);
        void Detach(std::shared_ptr<BeforeStepObserver> observer);
        void Notify(const BeforeStepEvent& event);

        std::shared_ptr<TrainObserver> Attach(std::shared_ptr<TrainObserver> observer);
        void Detach(std::shared_ptr<TrainObserver> observer);
        void Notify(const TrainEvent& event);

        std::shared_ptr<LearnObserver> Attach(std::shared_ptr<LearnObserver> observer);
        void Detach(std::shared_ptr<LearnObserver> observer);
        void Notify(const LearnEvent& event);

        void LogObservers() const;
    public:
        template <class T, class... Args>
        std::shared_ptr<T> Attach(Args&&... args)
        {
            auto obs = std::make_shared<T>(std::forward<Args>(args)...);
            Attach(obs);
            return obs;
        }
    private:
        std::vector<std::shared_ptr<BeforeStepObserver>> before_step_observers_;
        std::vector<std::shared_ptr<TrainObserver>> train_observers_;
        std::vector<std::shared_ptr<LearnObserver>> learn_observers_;
    };

    // =============================================================
    // Trainer
    // =============================================================

    enum class RunnerStatus {
        NOT_INITIALIZED, ///< 未初期化
        RUNNING,         ///< 実行中
        COMPLETED        ///< 終了済み
    };

    enum class ControlSignal {
        CONTINUE,///< 処理継続
        BREAK,   ///< フレーム内のループを中断して抜ける。Trainer自体は実行継続。
        STOP     ///< TrainerのStatusをCOMPLETEDに変更
    };

    class Runner : public DataExporter {
    public:
        using ControlFunction = std::function<ControlSignal(const StepCounts& counts)>; ///< 学習ステップ制御処理(戻り値trueで処理中断)
        static ControlSignal noop(const StepCounts& counts) { return ControlSignal::CONTINUE; }
    public:
        virtual RunnerStatus Initialize(const ConfigData& config_data) = 0;
        virtual StepCounts DoStep() = 0;
        virtual StepCounts DoUpdateFrame(int max_steps,
            ControlFunction pre_step_func = noop, ControlFunction post_step_func = noop) = 0;
        virtual RunnerStatus GetStatus() const = 0;
    public:
        virtual StepCounts GetCounts() const = 0;
        virtual std::shared_ptr<anet::rl::BatchEnv> GetBatchEnv()const = 0;
        virtual std::shared_ptr<anet::rl::Agent> GetAgent() const = 0;
        virtual std::shared_ptr<anet::rl::Notifier> GetNotifier() const = 0;
    public:
        virtual ~Runner() = default;
    public:
        static constexpr const char* TRAIN_REWARD = "train_reward";
        static constexpr const char* TRAIN_REWARD_EMA = "train_reward_ema";
        static constexpr const char* TRAIN_EPISODE_REWARD = "train_episode_reward";
        static constexpr const char* TARGET_EVAL_REWARD = "target_eval_reward";
        static constexpr const char* POLICY_EVAL_REWARD = "policy_eval_reward";
        static constexpr const char* TRAIN_STEP = "train_step";
        static constexpr const char* EXP_STEP = "exp_step";
        static constexpr const char* LEARN_STEP = "learn_step";
        static constexpr const char* EPISODE_COUNT = "episode_count";
        static constexpr const char* SIM_STEP = "sim_step";
        static constexpr const char* TRAIN_STEP_PER_SEC = "train_step_per_sec";
        static constexpr const char* EXP_STEP_PER_SEC = "exp_step_per_sec";
        static constexpr const char* ELAPSE_HOUR = "elapse_hour";
    };

    // =============================================================

    class AgentFactory {
    public:
        virtual std::shared_ptr<Agent> CreateAgent(
            const EnvSpec& env_spec,
            const BatchEnvSpec& batch_env_spec,
            const torch::Device& device,
            const anet::ConfigData& config_data = EmptyConfigData,
            std::shared_ptr<anet::rl::Notifier> notifier = nullptr,
            std::optional<anet::seed_t> seed = std::nullopt
        ) const = 0;
        virtual std::string GetTargetAgentClassId() const = 0;

        virtual ~AgentFactory() = default;
    };

    // =============================================================

    /// @todo class NNFactory

} // namespace anet::rl
