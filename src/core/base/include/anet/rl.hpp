// anet/rl.hpp

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <string_view>
#include <type_traits>
#include <map>
#include <unordered_map>
#include <stdexcept>
#include <optional>
#include <variant>
#include <span>
#include <mutex>
#include <cstdint>
#include <cctype>
#include <chrono>
#include <torch/torch.h>
#include "anet/common.hpp"
#include "anet/tensor_check.hpp"
#include "anet/random.hpp"
#include "anet/config.hpp"
#include "anet/log.hpp"
#include "anet/tensor_util.hpp"
#include "anet/serialize.hpp"
#include "anet/graphviz.hpp"

namespace anet { class ForwardCapture; }

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

    enum class ShutdownMode {
        WAIT,
        CANCEL,
    };

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

        void CheckSameStateActionSpec(const EnvSpec& actual) const;
        anet::json ToJson() const;
        std::string ToString() const;
    };

    enum class EpisodeScope {
        PER_LANE,  ///< 各laneを独立したepisode groupとして扱う。
        SHARED,    ///< 全laneを1つの共有episode groupとして扱う。
    };

    struct BatchEnvSpec {
        int num_envs;
        int num_threads;
        EpisodeScope episode_scope = EpisodeScope::PER_LANE;

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

    /// 呼び出し側が指定する lane ごとの離散行動。
    class ForcedActions {
    public:
        static ForcedActions ByLane(int64_t lane, int64_t action);
        static ForcedActions ByMask(torch::Tensor mask, torch::Tensor actions);

        void Validate(const ActionSpec& spec, int64_t lane_count) const;
        torch::Tensor Apply(const torch::Tensor& policy_actions, int64_t action_count) const;
        /// Validate または Apply が同じ lane 数で成功した指定の印を返す。
        torch::Tensor MakeMask(int64_t lane_count, torch::Device device) const;
    private:
        struct Lane {
            int64_t index;
            int64_t action;
        };
        struct Mask {
            torch::Tensor lanes;
            torch::Tensor actions;
        };
    private:
        explicit ForcedActions(std::variant<Lane, Mask> selection) : selection_(std::move(selection)) {}
        void ValidateDiscrete(int64_t lane_count, int64_t action_count) const;
    private:
        std::variant<Lane, Mask> selection_;
    };

    /// その回の行動選択に渡す要求。要求は呼び出しの間だけ有効。
    struct MakeActionOptions {
        bool nn_trace = false;
        std::optional<ForcedActions> forced_actions;
    };

    void AppendNnTraceAux(AuxData& aux, anet::ForwardCapture& capture);
    anet::TensorDict ExtractNnTrace(const AuxData& aux);


    class SingleEnvResult {
    public:
        explicit SingleEnvResult(AuxData aux = {}) : aux_(std::move(aux)) {}
        virtual AuxData GetAuxData() const { return aux_; }
        virtual std::optional<float> GetScalar(const std::string& key) const;
        virtual ~SingleEnvResult() = default;
    protected:
        AuxData aux_;  ///< 結果の時点の値を所有し、scalarの読出しでは複製しない。
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

    /// 学習Actorが既存forwardから取り出す、初期優先度推定用の不透明なpayload。
    /// 同じ実体の GetPayloadCpu と読み取り・コピーを別スレッドで同時に行わない。
    /// スレッドをまたぐ場合は、受け渡しが済んでから受け取った側で呼ぶ。
    class ReplayInitialPriorityHint {
    public:
        explicit ReplayInitialPriorityHint(torch::Tensor payload);

        const torch::Tensor& GetPayload() const { return payload_; }
        const torch::Tensor& GetPayloadCpu() const;

    private:
        torch::Tensor payload_;             ///< float32 [B,K]。列の意味は生成元Agentが所有する
        mutable torch::Tensor payload_cpu_; ///< ReplayBufferへ渡す際に再利用するCPU cache
    };

    /// 行動選択時の情報。同じ実体の GetAction(device) や action を読む GetScalar と、
    /// その実体の読み取り・コピーを別スレッドで同時に行わない。
    /// スレッドをまたぐ場合は、受け渡しが済んでから受け取った側で呼ぶ。
    class BatchActionInfo : public anet::ModuleBase, public anet::graphviz::GraphVizProvider {
    public:
        /// 判断の境界をActorからReplayへ渡す共有info鍵。
        static constexpr const char* kKeyDecision = "decision";

        BatchActionInfo() {}
        virtual ~BatchActionInfo() = default;

        BatchActionInfo(const torch::Tensor action, const anet::TensorDict& info = {}, const AuxData& aux = {},
            std::optional<ReplayInitialPriorityHint> replay_initial_priority_hint = std::nullopt)
            : action_cpu_(action.device().is_cpu() ? std::move(action) : torch::Tensor())
            , info_(info)
            , aux_(aux)
            , replay_initial_priority_hint_(std::move(replay_initial_priority_hint))
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
        std::optional<float> GetScalar(const std::string& key, int64_t lane = -1) const override;
        virtual std::shared_ptr<BatchActionInfo> To(torch::Device device) const
        {
            return std::make_shared<BatchActionInfo>(
                GetAction(device), info_.To(device), aux_, replay_initial_priority_hint_);
        }
        const AuxData& GetAuxData() const { return aux_; }
        AuxData& GetAuxData() { return aux_; }
        const anet::TensorDict& GetInfo() const { return info_; }
        anet::TensorDict& GetInfo() { return info_; }
        const std::optional<ReplayInitialPriorityHint>& GetReplayInitialPriorityHint() const
        {
            return replay_initial_priority_hint_;
        }
        void SetReplayInitialPriorityHint(ReplayInitialPriorityHint hint)
        {
            replay_initial_priority_hint_ = std::move(hint);
        }

        std::string ToString() const;
    protected:
        mutable torch::Tensor action_cpu_;
        mutable std::optional<std::pair<torch::Device, torch::Tensor>> gpu_;
        anet::TensorDict info_;
        AuxData aux_;
        std::optional<ReplayInitialPriorityHint> replay_initial_priority_hint_; ///< 初期優先度推定用の不透明なpayload
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
        virtual std::optional<float> GetScalar(const std::string& key, int64_t lane = -1) const;
        /// 同じ実体に対して複数のスレッドから同時に呼べる。
        const torch::Tensor& GetEpisodeEndMask() const;

        virtual ~BatchStepResult() = default;
    public:
        torch::Tensor reward;       ///< 報酬          (N) kFloat32
        BatchState next_state;      ///< 遷移後の観測  (N, state_dim...)
        BatchState continue_state;  ///< 実行継続用（Reset 後の状態も含む）
        uint32_t n_transitions;     ///< 今回のStepにおける状態遷移カウント数
        uint32_t n_episode_end;     ///< 今回のStepにおけるエピソード終了カウント数
    private:
        mutable std::once_flag episode_end_once_;
        mutable torch::Tensor episode_end_cpu_;
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
        virtual const std::string& GetName() const = 0;
        virtual RunMode GetRunMode() const = 0;
        virtual EnvSpec GetSpec() const = 0;
        virtual std::shared_ptr<const SingleResetResult> Reset() = 0;
        virtual std::shared_ptr<const SingleStepResult> Step(int64_t action) = 0;

        virtual ~SingleDiscreteEnv() = default;
    };

    class SingleDiscreteEnvFactory {
    public:
        virtual void ValidateConfig(const anet::ConfigData&, RunMode, const std::string&) const {}
        virtual std::shared_ptr<SingleDiscreteEnv> CreateSingleEnv(
            const anet::ConfigData& config_data,
            const torch::Device& device,
            const std::string& name,
            std::optional<anet::seed_t> seed = std::nullopt,
            RunMode run_mode = RunMode::Train,
            const std::string& config_prefix = "") = 0;

        virtual std::string GetTargetEnvClassId() const = 0;
        virtual std::string GetDefaultConfigPrefix() const
        {
            return GetTargetEnvClassId();
        }
        virtual std::vector<std::string> GetConfigPrefixes() const { return {GetDefaultConfigPrefix()}; }

        virtual ~SingleDiscreteEnvFactory() = default;
    };

    class BatchEnv : public Module {
    public:
        virtual const std::string& GetName() const = 0;
        virtual const std::string& GetEnvName(int64_t lane_index) const = 0;
        virtual RunMode GetRunMode() const = 0;
        virtual EnvSpec GetSpec() const = 0;
        virtual BatchEnvSpec GetBatchSpec() const = 0;
        virtual torch::Device GetDevice() const = 0;

        virtual std::shared_ptr<const BatchResetResult> Reset() = 0;    ///< BatchStateは使い回されるので必要に応じてDeepCopy必須
        virtual std::shared_ptr<const BatchStepResult> Step(std::shared_ptr<BatchActionInfo> action_info) = 0; ///< 遷移の結果を新規に返し、後のStepやResetで値を書き換えない
        //virtual std::shared_ptr<BatchEnv> Clone() const = 0;

        virtual void Shutdown() { }

        virtual ~BatchEnv() = default;
    };

    class BatchEnvFactory {
    public:
        virtual void ValidateConfig(const anet::ConfigData&, RunMode, const std::string&) const {}
        virtual std::shared_ptr<BatchEnv> CreateBatchEnv(
            const anet::ConfigData& config_data,
            const torch::Device& device,
            const std::string& name,
            std::optional<seed_t> seed = std::nullopt,
            int num_envs = 1,
            RunMode run_mode = RunMode::Train,
            const std::string& config_prefix = "") = 0;
        virtual std::string GetTargetEnvClassId() const = 0;
        virtual std::vector<std::string> GetConfigPrefixes() const { return {GetTargetEnvClassId()}; }
        virtual ~BatchEnvFactory() = default;
    };


    // =============================================================
    // Policy APIs
    // =============================================================

    struct ActorRequest {
        BatchEnvSpec batch_env_spec;
        EnvSpec env_spec;
        torch::Device device;
        seed_t seed;
        std::string actor_key;
    };

    /// source を守るロックの中で、Actor の写し直しに従う。
    class ActorSyncFollower {
    public:
        virtual void OnActorSync() = 0;
        virtual ~ActorSyncFollower() = default;
    };

    class Actor : public ModuleBase {
    public:
        /// @brief BatchStateから行動を生成する。
        /// @note 強制行動がある lane の指定を反映できない Actor はエラーを返す。
        /// @note 同一Actor instanceのMakeActionまたはSyncと並行呼び出ししてはならない。
        virtual std::shared_ptr<BatchActionInfo> MakeAction(const StepCounts& step, const anet::rl::BatchState& state, const MakeActionOptions& options = {}) const = 0;

        /// @brief Actor固有のsourceから推論resourceを強制同期する。
        /// @note 同一Actor instanceのMakeActionまたはSyncと並行呼び出ししてはならない。
        virtual void Sync() = 0;

        /// 同期の相手を所有せずにつなぐ。MakeAction / Sync と並行に呼ばない。
        virtual bool AttachSyncFollower(std::weak_ptr<ActorSyncFollower>) { return false; }

        /// 仮想デストラクタ
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

    struct ScalarMetricSubscription;

    struct AgentStepDiagnosticsRequest {
        int64_t num_lanes;
        std::vector<std::string> keys;
        std::shared_ptr<Actor> actor;
    };

    struct AgentStepDiagnosticsCompletion {
        int64_t lane = 0;
        int64_t steps_back = 0;
        std::vector<std::string> resolved_keys;
        std::unique_ptr<const anet::Module> values; ///< 対象1行をGetScalar(key, 0)で読む所有値。
    };

    struct AgentStepDiagnosticsResult {
        std::unique_ptr<const anet::Module> current; ///< batchをGetScalar(key, lane)で読む所有値。
        std::vector<std::vector<std::string>> pending_keys; ///< lane別。NaNとは独立に遅延確定を示す。
        std::vector<AgentStepDiagnosticsCompletion> completed;
    };

    /// Agent固有のstep診断値を生成し、laneごとの系列窓を所有する。
    /// pendingに示す鍵もcurrentでは既知の鍵としてNaNを返す。
    /// pendingの鍵は値が成立しなかった場合もNaNで確定を返し、必ず待機を終える。
    class AgentStepDiagnosticsProvider {
    public:
        virtual AgentStepDiagnosticsResult Observe(const BatchExperience& experience) = 0;
        virtual void Reset() = 0;
        virtual void Suspend() {}
        virtual bool Resume() { return false; }
        virtual ~AgentStepDiagnosticsProvider() = default;
    };

    class Agent : public Module, public TensorDictFunctionProvider , public Serializable {
    public:
        virtual std::shared_ptr<Actor> CreateActor(const ActorRequest& request) const = 0;
        virtual std::shared_ptr<Learner> CreateLearner() = 0;
        virtual std::unique_ptr<AgentStepDiagnosticsProvider> CreateStepDiagnosticsProvider(const AgentStepDiagnosticsRequest& request) const
        {
            return nullptr;
        }
        virtual torch::Device GetDevice() const = 0;
        virtual void ConfigureScalarMetricSubscriptions(
            const std::vector<ScalarMetricSubscription>& subscriptions)
        {
            (void)subscriptions;
        }
    public:
        virtual int64_t Save(anet::OutputArchive& archive) const override { return 0;  }
        virtual int64_t Load(anet::InputArchive& archive) override { return 0; }
        virtual ~Agent() = default;
    };


    // =============================================================
    // ReplayBuffer
    // =============================================================

    enum class ReplayPrioritySource : int8_t {
        NONE = 0,            ///< 未初期化または無効化済み
        FIXED_INITIAL = 1,   ///< 固定値による初期優先度
        MAX_INITIAL = 2,     ///< Learner更新済み最大値による初期優先度
        ACTOR_INITIAL = 3,   ///< Actor近似による初期優先度
        LEARNER_UPDATED = 4, ///< LearnerのTD誤差で更新済み
    };

    struct ReplayPriorityUpdateResult {
        int64_t applied_count = 0; ///< 現generationへ適用した更新数
        int64_t stale_count = 0;   ///< 上書き済みgenerationとして棄却した更新数
        int64_t actor_learner_pair_count = 0; ///< Actor初期値とLearner更新値を比較できた件数
        float actor_learner_positive_pair_ratio = std::numeric_limits<float>::quiet_NaN(); ///< 両方が正値の比較ペア率
        float actor_learner_ratio_median = std::numeric_limits<float>::quiet_NaN();        ///< 正値ペアのActor/Learner比の中央値
        float actor_learner_log_ratio_mean = std::numeric_limits<float>::quiet_NaN();      ///< 正値ペアのlog(Actor/Learner)平均
        float actor_learner_spearman = std::numeric_limits<float>::quiet_NaN();            ///< 全finiteペアのSpearman順位相関
    };

    /// サンプリングされた経験のミニバッチ（再利用可能オブジェクト）
    struct ExperienceSamples {
        // --- 観測データ ---
        anet::TensorDict obs;           ///< [B, Stack, C, H, W] 等

        // --- コア遷移データ ---
        torch::Tensor actions;          ///< [B] または [B, Unroll]
        torch::Tensor target_returns;   ///< n-stepの収益。[B]、複数成分なら[B,C]、unrollなら[B,Unroll]。

        struct {
            anet::TensorDict next_obs;      ///< [B, Stack, C, H, W] 等 (Rainbow等の N-Step 次状態)
            torch::Tensor terminals;        ///< [B] (真の終了フラグ)
            torch::Tensor truncated;        ///< [B] 末尾のstepの打ち切り。分解のbootstrap判定に使う。
        } next_state;

        // --- N-Step & PER メタデータ ---
        torch::Tensor n_steps;          ///< [B] 実際に進んだステップ数 (終端到達でNより短くなるケース用)

        struct {
            torch::Tensor lengths;     ///< [B] 抽選した窓の長さ。規則未指定なら未定義。
            torch::Tensor max_lengths; ///< [B] 起点から取れる窓の最大長
            torch::Tensor returns;     ///< [B] 窓の収益
            torch::Tensor terminals;   ///< [B] 真の終端
            torch::Tensor truncated;   ///< [B] 打ち切り
            anet::TensorDict next_obs; ///< k先の観測（履歴を含む）
        } window;
        
        torch::Tensor replay_item_keys; ///< [B] generation付きopaque key (CPU int64)
        torch::Tensor is_weights;       ///< [B] Importance Sampling の重み
        torch::Tensor per_priority_sources; ///< [B] ReplayPrioritySource (CPU int8)

        // --- その他 ---
        anet::TensorDict info;          ///< アルゴリズム固有データ (MuZeroの target_values 等)

        // schedule gamma用の抽選時snapshot。定数構成では未定義。
        torch::Tensor sampled_rewards;        ///< step報酬。[B,N]、複数成分なら[B,N,C]。実長以後は0。
        torch::Tensor needs_return_recompute; ///< [C]のbool。真の成分はFinalizeReturnsがsampled_rewardsから消費時のγで収益を作り直す。
        torch::Tensor sampled_window_rewards; ///< [B,J] 実長以後は0
        void FinalizeReturns(std::span<const float> gammas);

        // ユーティリティ
        ExperienceSamples To(torch::Device device, bool non_blocking = true) const;
        template <class F>
        void ForEachTensor(F&& fn)
        {
            auto visit = [&fn](torch::Tensor& tensor) {
                if (tensor.defined()) {
                    fn(tensor);
                }
            };

            VisitFields(*this, *this, [&](auto& value, auto&, const char*, bool) {
                if constexpr (std::is_same_v<std::remove_cvref_t<decltype(value)>, anet::TensorDict>)
                    value.ForEachTensor(visit);
                else
                    visit(value);
            });
        }
        std::string ToString() const;
    private:
        template <class Source, class Destination, class F>
        static void VisitFields(Source& source, Destination& destination, F&& fn)
        {
            // 転送先を伴う処理と、その場の走査で同じフィールドの並びを使う。
            // opaque な Replay metadata だけは CPU に置き、学習 device へ送らない。
            fn(source.obs, destination.obs, "obs", false);
            fn(source.actions, destination.actions, "actions", false);
            fn(source.target_returns, destination.target_returns, "target_returns", false);
            fn(source.next_state.next_obs, destination.next_state.next_obs, "next_state.next_obs", false);
            fn(source.next_state.terminals, destination.next_state.terminals, "next_state.terminals", false);
            fn(source.next_state.truncated, destination.next_state.truncated, "next_state.truncated", false);
            fn(source.n_steps, destination.n_steps, "n_steps", false);
            fn(source.window.lengths, destination.window.lengths, "window.lengths", false);
            fn(source.window.max_lengths, destination.window.max_lengths, "window.max_lengths", false);
            fn(source.window.returns, destination.window.returns, "window.returns", false);
            fn(source.window.terminals, destination.window.terminals, "window.terminals", false);
            fn(source.window.truncated, destination.window.truncated, "window.truncated", false);
            fn(source.window.next_obs, destination.window.next_obs, "window.next_obs", false);
            fn(source.replay_item_keys, destination.replay_item_keys, "replay_item_keys", true);
            fn(source.is_weights, destination.is_weights, "is_weights", false);
            fn(source.per_priority_sources, destination.per_priority_sources, "per_priority_sources", true);
            fn(source.info, destination.info, "info", false);
            fn(source.sampled_rewards, destination.sampled_rewards, "sampled_rewards", false);
            fn(source.needs_return_recompute, destination.needs_return_recompute, "needs_return_recompute", false);
            fn(source.sampled_window_rewards, destination.sampled_window_rewards, "sampled_window_rewards", false);
        }
    };


    /// 同一sampleable snapshotに対する履歴別の集計・抽出要求。
    struct SamplingHistoryProbeRequest {
        bool counts = false;
        std::optional<int64_t> unsampled_batch_size;
        std::optional<int64_t> sampled_batch_size;
    };

    struct SamplingHistoryProbeResult {
        int64_t unsampled_count = 0;
        int64_t sampled_count = 0;
        float unsampled_age_mean = std::numeric_limits<float>::quiet_NaN();
        float sampled_age_mean = std::numeric_limits<float>::quiet_NaN();
        std::optional<ExperienceSamples> unsampled;
        std::optional<ExperienceSamples> sampled;
    };

    class ReplayPriorityController {
    public:
        virtual ReplayPriorityUpdateResult UpdatePriorities(
            const std::vector<int64_t>& item_keys, const std::vector<float>& priorities) = 0;
        //virtual void UpdatePriorities(const std::vector<int64_t>& indices, const std::vector<float>& priorities) override = 0;
        //virtual void UpdatePriorities(const torch::Tensor& indices, const torch::Tensor& priorities) = 0;
        ~ReplayPriorityController() = default;
    };

    class ReplayBuffer : public anet::Module, public ReplayPriorityController {
    public:
        virtual void Push(const BatchExperience& batch_exp, std::span<const float> gammas) = 0;
        virtual void Sample(ExperienceSamples& out_samples, int64_t minibatch_size, float beta, std::span<const float> gammas) const = 0;
        virtual bool SampleUniqueUniform(
            ExperienceSamples& out_samples, int64_t batch_size, anet::RandomGenerator& random, std::span<const float> gammas) const = 0;
        virtual SamplingHistoryProbeResult ProbeSamplingHistory(
            const SamplingHistoryProbeRequest& request, anet::RandomGenerator* random, std::span<const float> gammas) const = 0;
        virtual int64_t Size() const = 0;

        virtual ~ReplayBuffer() = default;
    public:
        static constexpr const char* kKeyPrefix = "replaybuffer.";

        static constexpr const char* STATE_OBS = "replaybuffer.storage.state";
        static constexpr const char* ACTION = "replaybuffer.storage.action";
        static constexpr const char* TARGET_RETURN = "replaybuffer.storage.target_return";
        static constexpr const char* NEXT_STATE_OBS = "replaybuffer.storage.next_state";
        static constexpr const char* NEXT_STATE_TERMINAL = "replaybuffer.storage.terminal";
        static constexpr const char* N_STEP = "replaybuffer.storage.n_step";

        static constexpr const char* PER_TOTAL = "replaybuffer.per.total";
        static constexpr const char* PER_VALUES = "replaybuffer.per.values";
        static constexpr const char* PER_DIST = "replaybuffer.per.distribution";
        static constexpr const char* PER_INITIAL_MASS_RATIO = "replaybuffer.per.initial_mass_ratio";
        static constexpr const char* PER_FIXED_INITIAL_MASS_RATIO = "replaybuffer.per.fixed_initial_mass_ratio";
        static constexpr const char* PER_MAX_INITIAL_MASS_RATIO = "replaybuffer.per.max_initial_mass_ratio";
        static constexpr const char* PER_ACTOR_INITIAL_MASS_RATIO = "replaybuffer.per.actor_initial_mass_ratio";
        static constexpr const char* PER_ACTOR_COMPLETION_ATTEMPT_COUNT = "replaybuffer.per.actor_completion_attempt_count";
        static constexpr const char* PER_ACTOR_COMPLETION_SUCCESS_COUNT = "replaybuffer.per.actor_completion_success_count";
        static constexpr const char* PER_ACTOR_COMPLETION_SUCCESS_RATIO = "replaybuffer.per.actor_completion_success_ratio";
        static constexpr const char* PER_ACTOR_TRUNCATION_FALLBACK_COUNT = "replaybuffer.per.actor_truncation_fallback_count";
        static constexpr const char* PER_ACTOR_TRUNCATION_FALLBACK_RATIO = "replaybuffer.per.actor_truncation_fallback_ratio";
        static constexpr const char* PER_ACTOR_NONFINITE_FALLBACK_COUNT = "replaybuffer.per.actor_nonfinite_fallback_count";
        static constexpr const char* PER_ACTOR_NONFINITE_FALLBACK_RATIO = "replaybuffer.per.actor_nonfinite_fallback_ratio";
        static constexpr const char* PER_PRIORITY_UPDATE_STALE_DROP_COUNT = "replaybuffer.per.priority_update_stale_drop_count";
        static constexpr const char* PER_LAST_EVICTED_NEVER_SAMPLED_RATIO = "replaybuffer.per.last_evicted_never_sampled_ratio";
    };


    // =============================================================
    // Observer
    // =============================================================

    enum class EventType {
        TRAIN,
        LEARN,
        EPISODE_END,
        SESSION_END
    };

    enum class RunnerScope {
        TRAIN,
        EVAL
    };

    enum class EventField {
        EXPERIENCE,
        AGENT,
        ACTOR,
        ENV,
        ENV_STEP,
        UPDATE_RESULT,
        RUNNER,
        ACTION_INFO,
        AGENT_STEP,
    };

    struct ScalarMetricSubscription {
        std::string source_key;
        EventType event = EventType::TRAIN;
        std::optional<EventField> target;
        int interval = 1;
        RunnerScope scope = RunnerScope::TRAIN;
        std::string eval_name;
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
        const std::shared_ptr<const BatchStepResult> step_result;
    };

    struct SessionEndEvent {
        const std::shared_ptr<const Runner> runner;
        const StepCounts counts;
        const std::shared_ptr<const Agent> agent;
        const std::shared_ptr<const BatchEnv> env;
    };

    class TrainObserver {
    public:
        virtual void OnTrain(const TrainEvent& event) = 0;
        virtual void Shutdown(std::chrono::steady_clock::time_point, ShutdownMode) {}
        virtual bool WillBlockOnShutdown() const { return false; }
        virtual std::string ToString() const = 0;
        virtual ~TrainObserver() = default;
    };

    class LearnObserver {
    public:
        virtual void OnLearn(const LearnEvent& event) = 0;
        virtual void Shutdown(std::chrono::steady_clock::time_point, ShutdownMode) {}
        virtual bool WillBlockOnShutdown() const { return false; }
        virtual std::string ToString() const = 0;
        virtual ~LearnObserver() = default;
    };

    class EpisodeEndObserver {
    public:
        virtual void OnEpisodeEnd(const EpisodeEndEvent& event) = 0;
        virtual void Shutdown(std::chrono::steady_clock::time_point, ShutdownMode) {}
        virtual bool WillBlockOnShutdown() const { return false; }
        virtual std::string ToString() const = 0;
        virtual ~EpisodeEndObserver() = default;
    };

    class SessionEndObserver {
    public:
        virtual void OnSessionEnd(const SessionEndEvent& event) = 0;
        virtual void Shutdown(std::chrono::steady_clock::time_point, ShutdownMode) {}
        virtual bool WillBlockOnShutdown() const { return false; }
        virtual std::string ToString() const = 0;
        virtual ~SessionEndObserver() = default;
    };

    // -----------------------------------------------------------------
    // RunnerScopedTrainObserver
    // -----------------------------------------------------------------
    class RunnerScopedTrainObserver : public TrainObserver {
    public:
        RunnerScopedTrainObserver(std::shared_ptr<TrainObserver> real_observer, std::shared_ptr<const Runner> target_runner);

        void OnTrain(const TrainEvent& event) override;
        void Shutdown(std::chrono::steady_clock::time_point deadline, ShutdownMode mode) override;
        bool WillBlockOnShutdown() const override;
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
        void Shutdown(std::chrono::steady_clock::time_point deadline, ShutdownMode mode) override;
        bool WillBlockOnShutdown() const override;
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
        void Shutdown(std::chrono::steady_clock::time_point deadline, ShutdownMode mode) override;
        bool WillBlockOnShutdown() const override;
        std::string ToString() const override;
    private:
        std::shared_ptr<EpisodeEndObserver> real_observer_;
        std::shared_ptr<const Runner> target_runner_;
    };

    class RunnerScopedSessionEndObserver : public SessionEndObserver {
    public:
        RunnerScopedSessionEndObserver(std::shared_ptr<SessionEndObserver> real_observer, std::shared_ptr<const Runner> target_runner);
        void OnSessionEnd(const SessionEndEvent& event) override;
        void Shutdown(std::chrono::steady_clock::time_point deadline, ShutdownMode mode) override;
        bool WillBlockOnShutdown() const override;
        std::string ToString() const override;
    private:
        std::shared_ptr<SessionEndObserver> real_observer_;
        std::shared_ptr<const Runner> target_runner_;
    };

    // =============================================================
    // Notifier
    // =============================================================

    /// 登録・解除・通知・全解除・終了・一覧の操作は同時に呼べる。
    /// Observer は登録順に呼び出し元のスレッドで動き、その間 Notifier のロックを持たない。
    /// Observer から登録・解除・通知を呼べる。登録・解除は次に始まる通知から効く。
    /// Detach と Clear は進行中の通知を待たず、その通知が終わるまで取得済みの Observer は生きる。
    /// Observer が触る相手の寿命は登録側で守る。Observer は最後の参照を手放したスレッドで破棄される。
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

        std::shared_ptr<SessionEndObserver> Attach(std::shared_ptr<SessionEndObserver> observer);
        void Detach(std::shared_ptr<SessionEndObserver> observer);
        void Detach(const SessionEndObserver* observer);
        void Notify(const SessionEndEvent& event);

        void Clear();

        void Shutdown(std::chrono::steady_clock::time_point deadline, ShutdownMode mode);
        bool WillBlockOnShutdown() const;

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
            if constexpr (std::is_base_of_v<SessionEndObserver, T>) {
                auto wrapper = std::make_shared<RunnerScopedSessionEndObserver>(obs, target_runner);
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
        std::shared_ptr<SessionEndObserver> AttachScoped(std::shared_ptr<SessionEndObserver> observer, std::shared_ptr<const Runner> target_runner)
        {
            auto wrapper = std::make_shared<RunnerScopedSessionEndObserver>(observer, target_runner);
            this->Attach(wrapper);
            return observer;
        }
    private:
        template <class Observer>
        using ObserverList = std::shared_ptr<const std::vector<std::shared_ptr<Observer>>>;

        struct ObserverLists {
            ObserverList<TrainObserver> train = std::make_shared<const std::vector<std::shared_ptr<TrainObserver>>>();
            ObserverList<LearnObserver> learn = std::make_shared<const std::vector<std::shared_ptr<LearnObserver>>>();
            ObserverList<EpisodeEndObserver> episode_end = std::make_shared<const std::vector<std::shared_ptr<EpisodeEndObserver>>>();
            ObserverList<SessionEndObserver> session_end = std::make_shared<const std::vector<std::shared_ptr<SessionEndObserver>>>();
        };

        template <class Observer>
        ObserverList<Observer> GetObserverSnapshot(const ObserverList<Observer>& observers) const;
        ObserverLists GetObserverSnapshot() const;
        template <class Observer>
        std::shared_ptr<Observer> AttachObserver(ObserverList<Observer>& observers, std::shared_ptr<Observer> observer);
        template <class Observer>
        void DetachObserver(ObserverList<Observer>& observers, const Observer* observer);

        mutable std::mutex observers_mutex_;
        ObserverLists observers_;
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
        virtual StepCounts DoStep(const MakeActionOptions& options = {}) = 0;
        virtual StepCounts DoUpdateFrame(int max_steps,
            ControlFunction pre_step_func = nullptr, ControlFunction post_step_func = nullptr, const MakeActionOptions& options = {}) = 0;
        virtual RunnerStatus GetStatus() const = 0;

        virtual void Shutdown() = 0;
    public:
        virtual StepCounts GetCounts() const = 0;
        virtual const std::string& GetName() const = 0;
        virtual std::shared_ptr<anet::rl::BatchEnv> GetBatchEnv()const = 0;
        virtual std::shared_ptr<anet::rl::Agent> GetAgent() const = 0;
        virtual std::shared_ptr<anet::rl::Actor> GetActor() const = 0;
        virtual std::shared_ptr<anet::rl::Notifier> GetNotifier() const = 0;
    public:
        virtual ~Runner() = default;
    public:
        static constexpr const char* TRAIN_REWARD = "train_reward";
        static constexpr const char* TRAIN_REWARD_EMA = "train_reward_ema";
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
        virtual std::vector<std::string> GetConfigPrefixes() const { return {GetTargetAgentClassId()}; }

        virtual ~AgentFactory() = default;
    };

    // =============================================================

    /// @todo class NNFactory

} // namespace anet::rl
