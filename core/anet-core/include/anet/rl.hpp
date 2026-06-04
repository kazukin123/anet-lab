// anet/rl.hpp

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <string_view>
#include <map>
#include <unordered_map>
#include <stdexcept>
#include <optional>
#include <cstdint>
#include <cctype>
#include <torch/torch.h>
#include "anet/common.hpp"
#include "anet/tensor_check.hpp"
#include "anet/random.hpp"
#include "anet/config.hpp"
#include "anet/log.hpp"
#include "anet/tensor_util.hpp"
#include "anet/serialize.hpp"
#include "anet/graphviz.hpp"

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

    /// ステップ数型
    using step_t = uint64_t;

    /// Step軸
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

    enum class RunMode {
        Train,
        Eval,
        Eval1,
        Eval2,
        //Collect
    };

    inline bool IsTrain(RunMode mode) { return mode == RunMode::Train; }
    inline bool IsEval(RunMode mode) { return mode == RunMode::Eval || mode == RunMode::Eval1 || mode == RunMode::Eval2; }
    inline std::string ToString(RunMode mode)
    {
        switch (mode) {
        case RunMode::Train: return "Train";
        case RunMode::Eval: return "Eval";
        case RunMode::Eval1: return "Eval1";
        case RunMode::Eval2: return "Eval2";
        }
        ANET_SYSTEM_ERROR("RunModeFromString(): RunMode:" << static_cast<int>(mode));
        return "";
    }
    inline RunMode RunModeFromString(const std::string& str)
    {
        auto mode_str = anet::ToLower(str);

        if (mode_str == "train") return RunMode::Train;
        if (mode_str == "eval") return RunMode::Eval;
        if (mode_str == "eval1") return RunMode::Eval1;
        if (mode_str == "eval2") return RunMode::Eval2;
        ANET_SYSTEM_ERROR("RunModeFromString(): invalid string: " <<  str);
		return RunMode::Train;
	}

    // =============================================================
    // Environment 定義クラス
    // =============================================================

    struct ObsKeys
    {
        // --- 空間トポロジによる使い分けデフォルトキー ---
        static constexpr const char* kVector = "vector";
        static constexpr const char* kGrid = "grid";

        // --- メタデータ・特殊用途 ---
        static constexpr const char* kActionMask = "action_mask";
    };

    // 観測仕様
    struct StateSpec {
        anet::TensorSpecMap obs_spec;
        std::map<std::string, std::string> info;

        const anet::TensorSpec& GetSpace(const std::string& key) const
        {
            return obs_spec.at(key);
        }

        std::vector<std::int64_t> GetShape(const std::string& key) const
        {
            return obs_spec.at(key).shape;
        }
         
        // ---------------------------------------------------------
        // 整合性検証 (Assert用)
        // ---------------------------------------------------------
 
        /// 定義されたTensorSpec自体に矛盾がないかを検証する
        void AssertSanity() const;

        /// 観測データ(TensorDict)が仕様(存在, Dtype, Shape, Range)を全て満たしているか一括検証する。
        bool ValidateObservation(const anet::TensorDict& obs, bool is_batched = true) const;

        anet::json ToJson() const;
        std::string ToString() const;
    };

    // 連続アクション次元情報
    struct ActionDimInfo {
        float min_value;
        float max_value;
        std::string name;
        std::string description;

        anet::json ToJson() const;
        std::string ToString() const;
    };

    // 行動仕様（離散/連続を一本化）
    struct ActionSpec {
        bool is_discrete;
        std::vector<std::string> value_labels; // 離散アクションの場合のみ使用
        std::vector<ActionDimInfo> dims; // 連続アクションの場合のみ使用
        std::map<std::string, std::string> info;

        std::vector<int64_t> GetShape() const
        {
            if (is_discrete) {
                return { }; // 離散アクションではスカラー
            } else {
                std::vector<int64_t> shape;
                for (const auto& dim : dims) {
                    shape.push_back(1); // 連続アクションは各次元がスカラーなので、shapeは(dims.size(),)となる
                }
                return shape;
            }
		}

        torch::ScalarType GetDataType() const
        {
            if (is_discrete) {
                return torch::kInt64;
            } else {
                return torch::kFloat32;
            }
        }

        int GetNumActions() const
        {
            if (is_discrete) {
                return (int)value_labels.size();
            }

            // continuous
            ANET_ASSERT_MSG(!dims.empty(),
                "ActionSpec::ActionCount(): continuous action must define dims.");
            return (int)dims.size();
        }

        anet::json ToJson() const;
        std::string ToString() const;
    };

    // 環境仕様
    struct EnvSpec {
        StateSpec state_spec;
        ActionSpec action_spec;
        std::pair<float, float> reward_range;
        std::map<std::string, std::string> info;

        /// @todo RewardSpec

        anet::json ToJson() const;
        std::string ToString() const;
    };

    struct BatchEnvSpec {
        int num_envs;
        int num_threads;

        anet::json ToJson() const;
        std::string ToString() const;
    };


    // =============================================================
    // 暫定
    // =============================================================

    torch::Tensor ToUnifiedObservation(const TensorDict& obs, bool is_batched = true);


    // =============================================================
    // Single系データ
    // =============================================================

    struct SingleState {
        TensorDict obs;
        bool done;                  ///< Gymnasiumのterminated相当。真の終了（ゲームオーバー、クリア）。未来の価値は 0。doneとtruncatedは独立。
		bool truncated;             ///< Gymnasiumのtruncated相当。時間切れなどの人工終了。未来の価値は 0。doneとtruncatedは独立。
        bool episode_start;

        SingleState To(torch::Device device) const
        {
            //ANET_ASSERT_SHAPE(obs, { ANET_SHAPE_ANY });
            //ANET_ASSERT_DTYPE(obs, torch::kFloat32);
            return { obs.To(device), done, truncated, episode_start };
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

    inline constexpr std::string_view kNnTracePrefix = "nn_trace/";
    anet::TraceSink MakeActionTraceSink(anet::TensorDict& trace);
    void AppendTraceAux(AuxData& aux, const anet::TensorDict& trace);
    anet::TensorDict ExtractNnTrace(const AuxData& aux);


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

        SingleExperience To(torch::Device device) const
        {
            ANET_ASSERT_SHAPE(action, { });
            ANET_ASSERT(action.dtype() == torch::kInt64 || action.dtype() == torch::kFloat32);
            return {
                state.To(device), action.to(device), reward, next_state.To(device)
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
        TensorDict obs;              ///< 行動前の観測 (N,state_dim) kFloat32
        torch::Tensor done;             ///< (N) kBool Gymnasiumのterminated相当。真の終了（ゲームオーバー、クリア）。未来の価値は 0。doneとtruncatedは独立。
        torch::Tensor truncated;        ///< (N) kBool Gymnasiumのtruncated相当。時間切れなどの人工終了。未来の価値は 0。doneとtruncatedは独立。
        torch::Tensor episode_start;    ///< reset直後    (N) kBool

        BatchState() {}

        BatchState(const TensorDict& o, torch::Tensor d, torch::Tensor t, torch::Tensor e)
            : obs(std::move(o)), done(std::move(d)), truncated(std::move(t)), episode_start(std::move(e)) { }

        BatchState Clone() const {
            return { obs.Clone(), done.clone(), truncated.clone(), episode_start.clone() };
        }

        SingleState GetSingle(int64_t index) const
        {
            return {
                .obs = obs[index],
                .done = done[index].item<bool>(),
                .truncated = truncated[index].item<bool>(),
                .episode_start = episode_start[index].item<bool>()
            };
        }

        /// obs を (N, state_dim) にフラット化
        //BatchState Flatten() const
        //{
        //    ANET_ASSERT_DTYPE(obs, torch::kFloat32);
        //    int64_t N = obs.size(0);
        //    int64_t flat_dim = obs.numel() / N;
        //    auto f = obs.reshape({ N, flat_dim });
        //    return { f, done, truncated, episode_start };
        //}

        BatchState To(torch::Device device, bool non_blocking = false) const
        {
            //ANET_ASSERT_SHAPE(obs, { ANET_SHAPE_ANY, ANET_SHAPE_ANY });
            ANET_ASSERT_SHAPE(done, { ANET_SHAPE_ANY });
            ANET_ASSERT_SHAPE(truncated, { ANET_SHAPE_ANY });
            ANET_ASSERT_SHAPE(episode_start, { ANET_SHAPE_ANY });
            //ANET_ASSERT_DTYPE(obs, torch::kFloat32);
            ANET_ASSERT_DTYPE(done, torch::kBool);
            ANET_ASSERT_DTYPE(truncated, torch::kBool);
            ANET_ASSERT_DTYPE(episode_start, torch::kBool);

            return {
                obs.To(device, non_blocking), done.to(device, non_blocking),
                truncated.to(device,non_blocking), episode_start.to(device, non_blocking)
            };
        }
        bool IsDone() const
        {
            ANET_ASSERT_SHAPE(done, { ANET_SHAPE_ANY });
            ANET_ASSERT(done.size(0) >= 1);
            return done[0].item<bool>();
        }
        bool IsTruncated() const
        {
            ANET_ASSERT_SHAPE(truncated, { ANET_SHAPE_ANY });
            ANET_ASSERT(truncated.size(0) >= 1);
            return truncated[0].item<bool>();
        }
        bool IsEpisodeStart() const
        {
            ANET_ASSERT_SHAPE(episode_start, { ANET_SHAPE_ANY });
            ANET_ASSERT(episode_start.size(0) >= 1);
            return episode_start[0].item<bool>();
        }
        std::string ToString() const;
    };

    // 行動選択時の情報
    class BatchActionInfo : public anet::graphviz::GraphVizProvider {
    public:
        BatchActionInfo() {}
        virtual ~BatchActionInfo() = default;

        BatchActionInfo(const torch::Tensor action, const anet::TensorDict& info = {}, const AuxData& aux = {})
            : action_cpu_(action.device().is_cpu() ? std::move(action) : torch::Tensor())
            , info_(info)
            , aux_(aux)
        {
            if (action.device().is_cuda())
                gpu_ = std::pair(action.device(), std::move(action));
        }

        torch::Tensor GetAction() const
        {
            if (action_cpu_.defined())
                return action_cpu_;
            if (gpu_.has_value())
                return gpu_->second;
            return torch::Tensor();
        }

        torch::Tensor GetAction(torch::Device device) const;
        virtual std::shared_ptr<BatchActionInfo> To(torch::Device device) const
        {
            return std::make_shared<BatchActionInfo>(GetAction(device), info_.To(device), aux_ );
        }
        virtual std::shared_ptr<BatchActionInfo> WithAction(torch::Tensor action) const
        {
            return std::make_shared<BatchActionInfo>(std::move(action), info_, aux_);
        }
        const AuxData& GetAuxData() const { return aux_; }
        AuxData& GetAuxData() { return aux_; }
        const anet::TensorDict& GetInfo() const { return info_; }
        anet::TensorDict& GetInfo() { return info_; }

        std::string ToString() const;
    protected:
        mutable torch::Tensor action_cpu_;
        mutable std::optional<std::pair<torch::Device, torch::Tensor>> gpu_;
        anet::TensorDict info_;
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
        BatchStepResult(torch::Tensor reward_in, BatchState next_state_in, BatchState continue_state_in, uint32_t n_transitions_in, uint32_t n_episode_end_in);
        std::string ToString() const;

        virtual ~BatchStepResult() = default;
    public:
        torch::Tensor reward;       ///< 報酬          (N) kFloat32
        BatchState next_state;      ///< 遷移後の観測  (N, state_dim...)
        BatchState continue_state;  ///< 実行継続用（Reset 後の状態も含む）
        uint32_t n_transitions;     ///< 今回のStepにおける状態遷移カウント数
        uint32_t n_episode_end;     ///< 今回のStepにおけるエピソード終了カウント数
    };

    using MetricsMap = std::unordered_map<std::string, float>;

    class BatchUpdateResult : public Module {
    public:
        BatchUpdateResult() {}

        virtual ~BatchUpdateResult() = default;
    };

    struct BatchExperience : public Module {
        BatchState state;
        std::shared_ptr<BatchActionInfo> action;
        torch::Tensor reward;
        BatchState next_state;

        BatchExperience() {}
        BatchExperience(
            const BatchState& state__,
            std::shared_ptr<BatchActionInfo> action__,
            torch::Tensor reward__,
            const BatchState& next_state__
        ) : state(state__), action(action__), reward(std::move(reward__)), next_state(next_state__) { }

        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const { return std::nullopt; }
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override;

        BatchExperience To(torch::Device d, bool non_blocking = false) const;
        std::vector<SingleExperience> ToExperienceList() const;
        std::string ToString() const;
    public:
        static constexpr const char* STATE_OBS = "experience.state.obs";
        static constexpr const char* STATE_DONE = "experience.state.done";
        static constexpr const char* STATE_TRUNCATED = "experience.state.truncated";
        static constexpr const char* STATE_EPISODE_START = "experience.state.episode_start";
        static constexpr const char* ACTION_ACTION = "experience.action.action";
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
    class SingleDiscreteEnv : public Module {
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
            std::optional<anet::seed_t> seed = std::nullopt,
            const std::string& config_prefix = "") = 0;

        virtual std::string GetTargetEnvClassId() const = 0;
        virtual std::string GetDefaultConfigPrefix() const
        {
            return GetTargetEnvClassId();
        }

        virtual ~SingleDiscreteEnvFactory() = default;
    };

    class BatchEnv : public Module {
    public:
        virtual EnvSpec GetSpec() const = 0;
        virtual BatchEnvSpec GetBatchSpec() const = 0;
        virtual torch::Device GetDevice() const = 0;

        virtual std::shared_ptr<const BatchResetResult> Reset(RunMode mode = RunMode::Train) = 0;    ///< BatchStateは使い回されるので必要に応じてDeepCopy必須
        virtual std::shared_ptr<const BatchStepResult> Step(std::shared_ptr<BatchActionInfo> action_info, RunMode mode = RunMode::Train) = 0; ///< BatchStepResultは使い回されるので必要に応じてDeepCopy必須
        //virtual std::shared_ptr<BatchEnv> Clone() const = 0;

        virtual void Shutdown() { }

        virtual ~BatchEnv() = default;
    };

    class BatchEnvFactory {
    public:
        virtual std::shared_ptr<BatchEnv> CreateBatchEnv(std::optional<seed_t> seed = std::nullopt, int num_envs = -1) = 0;	///< num_envs=-1でnum_envs自動
        virtual ~BatchEnvFactory() = default;
    };


    // =============================================================
    // Policy APIs
    // =============================================================

    class Actor {
    public:
        virtual std::shared_ptr<BatchActionInfo> MakeAction(const StepCounts& step, const anet::rl::BatchState& state) const = 0;
        virtual void Sync() = 0;
        virtual ~Actor() = default;
    };

    using BatchUpdateResultList = std::vector<std::shared_ptr<const anet::rl::BatchUpdateResult>>;

    class Runner;

    class Learner {
    public:
        virtual BatchUpdateResultList UpdateFromBatch(const StepCounts& step, const BatchExperience& expriences) = 0;
        virtual ~Learner() = default;
    };


    // =============================================================
    // Agent
    // =============================================================

    class Agent : public Module, public TensorFunctionProvider, public TensorDictFunctionProvider , public Serializable {
    public:
        virtual std::shared_ptr<Actor> CreateActor(const BatchEnvSpec& batch_env_spec, RunMode run_mode, bool clone_model, std::optional<torch::Device> device = std::nullopt) const = 0;
        virtual std::shared_ptr<Learner> CreateLearner() = 0;
    public:
        virtual int64_t Save(anet::OutputArchive& archive) const override { return 0;  }
        virtual int64_t Load(anet::InputArchive& archive) override { return 0; }
        virtual ~Agent() = default;
    };


    // =============================================================
    // ReplayBuffer 
    // =============================================================

    /// サンプリングされた経験のミニバッチ（再利用可能オブジェクト）
    struct ExperienceSamples {
        // --- 観測データ ---
        anet::TensorDict obs;           ///< [B, Stack, C, H, W] 等

        // --- コア遷移データ ---
        torch::Tensor actions;          ///< [B] または [B, Unroll]
        torch::Tensor target_returns;   ///< [B] (N-Step割引累積報酬) または [B, Unroll]

        struct {
            anet::TensorDict next_obs;      ///< [B, Stack, C, H, W] 等 (Rainbow等の N-Step 次状態)
            torch::Tensor terminals;        ///< [B] (真の終了フラグ)
        } next_state;

        // --- N-Step & PER メタデータ ---
        torch::Tensor n_steps;          ///< [B] 実際に進んだステップ数 (終端到達でNより短くなるケース用)
        torch::Tensor indices;          ///< [B] PER優先度更新用の1Dインデックス
        torch::Tensor is_weights;       ///< [B] Importance Sampling の重み

        // --- その他 ---
        anet::TensorDict info;          ///< アルゴリズム固有データ (MuZeroの target_values 等)

        // ユーティリティ
        ExperienceSamples To(torch::Device device, bool non_blocking = true) const;
        std::string ToString() const;
    };


    class ReplayPriorityController {
    public:
        virtual void UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities) = 0;
        //virtual void UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities) override = 0;
        //virtual void UpdatePriorities(const torch::Tensor& indices, const torch::Tensor& priorities) = 0;
        ~ReplayPriorityController() = default;
    };

    class ReplayBuffer : public anet::Module, public ReplayPriorityController {
    public:
        virtual void Push(const BatchExperience& batch_exp) = 0;
        virtual void Sample(ExperienceSamples& out_samples, int64_t minibatch_size, float beta) const = 0;
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
        LEARN,
        EPISODE_END
    };

    enum class RunnerScope {
        TRAIN,
        EVAL
    };

    enum class EventField {
        EXPERIENCE,
        AGENT,
        ENV,
        UPDATE_RESULT,
        RUNNER,
        ACTION_INFO,
    };

    struct UpdateEvent {
        const BatchExperience& experience;
        const std::shared_ptr<const Runner> runner;
        const StepCounts counts;
        const std::shared_ptr<const Agent> agent;
        const BatchUpdateResultList update_result_list;
    };

    struct TrainEvent : public UpdateEvent {
        const std::shared_ptr<const BatchEnv> env;
        const std::shared_ptr<const BatchStepResult> step_result;
        const std::shared_ptr<const BatchActionInfo> action_info;
    };

    struct LearnEvent : public UpdateEvent {
    };

    struct EpisodeEndEvent {
        const std::shared_ptr<const Runner> runner;
        const StepCounts counts;
        const std::shared_ptr<const Agent> agent;
        const std::shared_ptr<const BatchEnv> env;
        int env_index;
        float eps_total_reward;
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

    class EpisodeEndObserver {
    public:
        virtual void OnEpisodeEnd(const EpisodeEndEvent& event) = 0;
        virtual std::string ToString() const = 0;
        virtual ~EpisodeEndObserver() = default;
    };

    // -----------------------------------------------------------------
    // RunnerScopedTrainObserver
    // -----------------------------------------------------------------
    class RunnerScopedTrainObserver : public TrainObserver {
    public:
        RunnerScopedTrainObserver(std::shared_ptr<TrainObserver> real_observer, std::shared_ptr<const Runner> target_runner);

        void OnTrain(const TrainEvent& event) override;
        std::string ToString() const override;
    private:
        std::shared_ptr<TrainObserver> real_observer_;
        std::shared_ptr<const Runner> target_runner_;
    };

    // -----------------------------------------------------------------
    // RunnerScopedLearnObserver
    // -----------------------------------------------------------------
    class RunnerScopedLearnObserver : public LearnObserver {
    public:
        RunnerScopedLearnObserver(std::shared_ptr<LearnObserver> real_observer, std::shared_ptr<const Runner> target_runner);
        void OnLearn(const LearnEvent& event) override;
        std::string ToString() const override;
    private:
        std::shared_ptr<LearnObserver> real_observer_;
        std::shared_ptr<const Runner> target_runner_;
    };

    // -----------------------------------------------------------------
    // RunnerScopedEpisodeEndObserver
    // -----------------------------------------------------------------
    class RunnerScopedEpisodeEndObserver : public EpisodeEndObserver {
    public:
        RunnerScopedEpisodeEndObserver(std::shared_ptr<EpisodeEndObserver> real_observer, std::shared_ptr<const Runner> target_runner);
        void OnEpisodeEnd(const EpisodeEndEvent& event) override;
        std::string ToString() const override;
    private:
        std::shared_ptr<EpisodeEndObserver> real_observer_;
        std::shared_ptr<const Runner> target_runner_;
    };

    // =============================================================
    // Notifier
    // =============================================================

    class Notifier {
    public:
        Notifier();

        std::shared_ptr<TrainObserver> Attach(std::shared_ptr<TrainObserver> observer);
        void Detach(std::shared_ptr<TrainObserver> observer);
        void Detach(const TrainObserver* observer);
        void Notify(const TrainEvent& event);

        std::shared_ptr<LearnObserver> Attach(std::shared_ptr<LearnObserver> observer);
        void Detach(std::shared_ptr<LearnObserver> observer);
        void Detach(const LearnObserver* observer);
        void Notify(const LearnEvent& event);

        std::shared_ptr<EpisodeEndObserver> Attach(std::shared_ptr<EpisodeEndObserver> observer);
        void Detach(std::shared_ptr<EpisodeEndObserver> observer);
        void Detach(const EpisodeEndObserver* observer);
        void Notify(const EpisodeEndEvent& event);

        void Clear();

        void LogObservers() const;
    public:
        template <class T, class... Args>
        std::shared_ptr<T> Attach(Args&&... args)
        {
            auto obs = std::make_shared<T>(std::forward<Args>(args)...);
            Attach(obs);
            return obs;
        }
        template <class T, class... Args>
        std::shared_ptr<T> AttachScoped(std::shared_ptr<Runner> target_runner, Args&&... args)
        {
            auto obs = std::make_shared<T>(std::forward<Args>(args)...);
            if constexpr (std::is_base_of_v<TrainObserver, T>) {
                auto wrapper = std::make_shared<RunnerScopedTrainObserver>(obs, target_runner);
                this->Attach(wrapper);
            }
            if constexpr (std::is_base_of_v<LearnObserver, T>) {
                auto wrapper = std::make_shared<RunnerScopedLearnObserver>(obs, target_runner);
                this->Attach(wrapper);
            }
            if constexpr (std::is_base_of_v<EpisodeEndObserver, T>) {
                auto wrapper = std::make_shared<RunnerScopedEpisodeEndObserver>(obs, target_runner);
                this->Attach(wrapper);
            }
            return obs;
        }
        std::shared_ptr<TrainObserver> AttachScoped(std::shared_ptr<TrainObserver> observer, std::shared_ptr<const Runner> target_runner)
        {
            auto wrapper = std::make_shared<RunnerScopedTrainObserver>(observer, target_runner);
            this->Attach(wrapper);
            return observer;
        }
        std::shared_ptr<LearnObserver> AttachScoped(std::shared_ptr<LearnObserver> observer, std::shared_ptr<const Runner> target_runner)
        {
            auto wrapper = std::make_shared<RunnerScopedLearnObserver>(observer, target_runner);
            this->Attach(wrapper);
            return observer;
        }
        std::shared_ptr<EpisodeEndObserver> AttachScoped(std::shared_ptr<EpisodeEndObserver> observer, std::shared_ptr<const Runner> target_runner)
        {
            auto wrapper = std::make_shared<RunnerScopedEpisodeEndObserver>(observer, target_runner);
            this->Attach(wrapper);
            return observer;
        }
    private:
        std::vector<std::shared_ptr<TrainObserver>> train_observers_;
        std::vector<std::shared_ptr<LearnObserver>> learn_observers_;
        std::vector<std::shared_ptr<EpisodeEndObserver>> episode_end_observers_;
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

    class Runner : public Module {
    public:
        using ControlFunction = std::function<ControlSignal(const StepCounts& counts)>; ///< 学習ステップ制御処理(戻り値trueで処理中断)
    public:
        //virtual RunnerStatus Initialize(const ConfigData& config_data) = 0;
        virtual StepCounts DoStep() = 0;
        virtual StepCounts DoUpdateFrame(int max_steps,
            ControlFunction pre_step_func = nullptr, ControlFunction post_step_func = nullptr) = 0;
        virtual RunnerStatus GetStatus() const = 0;

        virtual void Shutdown() = 0;
    public:
        virtual StepCounts GetCounts() const = 0;
        virtual const std::string& GetName() const = 0;
        virtual std::shared_ptr<anet::rl::BatchEnv> GetBatchEnv()const = 0;
        virtual std::shared_ptr<anet::rl::Agent> GetAgent() const = 0;
        virtual std::shared_ptr<anet::rl::Notifier> GetNotifier() const = 0;
    public:
        virtual ~Runner() = default;
    public:
        static constexpr const char* TRAIN_REWARD = "train_reward";
        static constexpr const char* TRAIN_REWARD_EMA = "train_reward_ema";
        static constexpr const char* TRAIN_EPISODE_REWARD = "train_episode_reward";
        static constexpr const char* EPS_TOTAL_REWARD = "eps_total_reward";
        static constexpr const char* TRAIN_STEP = "train_step";
        static constexpr const char* EXP_STEP = "exp_step";
        static constexpr const char* LEARN_STEP = "learn_step";
        static constexpr const char* EPISODE_COUNT = "episode_count";
        static constexpr const char* SIM_STEP = "sim_step";
        static constexpr const char* TRAIN_STEP_PER_SEC = "train_step_per_sec";
        static constexpr const char* EXP_STEP_PER_SEC = "exp_step_per_sec";
        static constexpr const char* ELAPSE_HOUR = "elapse_hour";

        static constexpr const char* REWARD = "reward";
        static constexpr const char* REWARD_EMA= "reward_ema";
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
