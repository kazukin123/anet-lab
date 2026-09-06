// anet/env.hpp

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include "anet/config.hpp"
#include "anet/log.hpp"
#include "anet/thread.hpp"
#include "anet/rl.hpp"
#include "anet/util.hpp"

namespace anet::rl {

    // ==============

    class SingleDiscreteEnvBase : public SingleDiscreteEnv {
    public:
        explicit SingleDiscreteEnvBase(
            std::string name,
            RunMode run_mode = RunMode::Train,
            std::optional<ConfigData> config_data = std::nullopt);

        const std::string& GetName() const override final { return name_; }
        RunMode GetRunMode() const override final { return run_mode_; }
        std::optional<ConfigData> GetConfigData() const override final { return config_data_; }
    private:
        std::string name_;
        RunMode run_mode_;
        std::optional<ConfigData> config_data_;
    protected:
        log::Logger log;
    };

    class BatchEnvBase : public BatchEnv {
    public:
        BatchEnvBase(
            std::string name,
            int num_envs,
            RunMode run_mode = RunMode::Train,
            std::optional<ConfigData> config_data = std::nullopt);

        const std::string& GetName() const override final { return name_; }
        const std::string& GetEnvName(int64_t lane_index) const override final;
        RunMode GetRunMode() const override final { return run_mode_; }
        std::optional<ConfigData> GetConfigData() const override final { return config_data_; }
    protected:
        void SetConfigData(ConfigData config_data);
    private:
        std::string name_;
        std::vector<std::string> lane_names_;
        RunMode run_mode_;
        std::optional<ConfigData> config_data_;
    protected:
        log::Logger log;
    };

    /// batch-native Env が追加情報をそのまま保持する標準 Reset result。
    class PlainBatchResetResult final : public BatchResetResult {
    public:
        PlainBatchResetResult(BatchState state, std::vector<AuxData> aux_data)
            : BatchResetResult(std::move(state)), aux_data_(std::move(aux_data)) {}

        std::vector<AuxData> GetAuxDataList(int env_index = -1) const override
        {
            if (env_index < 0) return aux_data_;
            if (env_index >= static_cast<int>(aux_data_.size())) return {};
            return { aux_data_[static_cast<size_t>(env_index)] };
        }

    private:
        std::vector<AuxData> aux_data_;
    };

    /// batch-native Env が追加情報をそのまま保持する標準 Step result。
    class PlainBatchStepResult final : public BatchStepResult {
    public:
        PlainBatchStepResult(
            torch::Tensor reward, BatchState next_state, BatchState continue_state,
            uint32_t n_transitions, uint32_t n_episode_end, std::vector<AuxData> aux_data)
            : BatchStepResult(
                std::move(reward), std::move(next_state), std::move(continue_state),
                n_transitions, n_episode_end)
            , aux_data_(std::move(aux_data)) {}

        std::vector<AuxData> GetAuxDataList(int env_index = -1) const override
        {
            if (env_index < 0) return aux_data_;
            if (env_index >= static_cast<int>(aux_data_.size())) return {};
            return { aux_data_[static_cast<size_t>(env_index)] };
        }

    private:
        std::vector<AuxData> aux_data_;
    };

    // ==============

    struct CompletedEpisodeReturn {
        int64_t group_index;
        float episode_return;

        bool operator==(const CompletedEpisodeReturn&) const = default;
    };

    /// episode group 単位で step reward を累積する。
    class EpisodeReturnAccumulator {
    public:
        explicit EpisodeReturnAccumulator(BatchEnvSpec batch_spec);

        void Reset();
        std::vector<CompletedEpisodeReturn> Add(const BatchStepResult& result);

    private:
        BatchEnvSpec batch_spec_;
        std::vector<float> current_returns_;
    };

    void ValidateEpisodeStructure(
        const std::string& env_name,
        const BatchEnvSpec& batch_spec,
        const BatchResetResult& result);
    std::vector<int64_t> ValidateEpisodeStructure(
        const std::string& env_name,
        const BatchEnvSpec& batch_spec,
        const BatchStepResult& result);

    struct EvalSessionResult {
        std::vector<float> episode_returns;
    };

    /// configured Eval の採用 episode と scalar snapshot を集約する decorator。
    class EvalSessionEnv final : public BatchEnvBase {
    public:
        EvalSessionEnv(
            std::shared_ptr<BatchEnv> inner,
            int eval_episodes,
            std::vector<std::string> subscribed_env_keys);

        EnvSpec GetSpec() const override { return inner_->GetSpec(); }
        BatchEnvSpec GetBatchSpec() const override { return batch_spec_; }
        torch::Device GetDevice() const override { return inner_->GetDevice(); }

        std::shared_ptr<const BatchResetResult> Reset() override;
        std::shared_ptr<const BatchStepResult> Step(
            std::shared_ptr<BatchActionInfo> action_info) override;
        void Shutdown() override { inner_->Shutdown(); }

        std::optional<float> GetScalar(
            const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(
            const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(
            const std::string& key, int64_t index = -1) const override;

        std::optional<EvalSessionResult> GetSessionResult() const { return session_result_; }
        /// 直前 Step で完了した採用 episode の group index（昇順）。次の Step / Reset まで有効。
        const std::vector<int64_t>& LastAdoptedGroups() const { return last_adopted_groups_; }

    private:
        struct ScalarSubscription {
            std::string source_key;
            std::string base_key;
            anet::ScalarAggregation aggregation;
            anet::ScalarSampleAccumulator accumulator;
        };

        bool HasAllFreshGroups(const BatchState& state) const;
        void BeginSession();
        void CaptureScalars(int64_t group_index);

        std::shared_ptr<BatchEnv> inner_;
        BatchEnvSpec batch_spec_;
        int eval_episodes_;
        int64_t group_count_;
        EpisodeReturnAccumulator return_accumulator_;
        std::vector<ScalarSubscription> subscriptions_;
        std::vector<bool> adopted_groups_;
        std::vector<int64_t> last_adopted_groups_;
        int issued_episodes_ = 0;
        int completed_episodes_ = 0;
        bool session_started_ = false;
        std::vector<float> captured_episode_returns_;
        std::shared_ptr<const BatchStepResult> cached_step_result_;
        std::optional<EvalSessionResult> session_result_;
    };

    // ==============

    class DiscreteBatchEnvBase : public BatchEnvBase, public RandomHolder {
    public:
        DiscreteBatchEnvBase(
            const ConfigData& config_data,
            std::shared_ptr<SingleDiscreteEnvFactory> factory,
            const std::string& name,
            int num_envs,
            const torch::Device& device,
            std::optional<seed_t> seed,
            RunMode run_mode,
            const std::string& config_prefix);

        EnvSpec GetSpec() const override;
        BatchEnvSpec GetBatchSpec() const override;
		torch::Device GetDevice() const override { return device_; }
    public:
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override;
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override;
    public:
        class Result;
        class ResetResult;
        class StepResult;
    protected:
        std::shared_ptr<DiscreteBatchEnvBase::ResetResult> createEmptyResetResult() const;
        std::shared_ptr<DiscreteBatchEnvBase::StepResult> createEmptyStepResult() const;
        std::shared_ptr<DiscreteBatchEnvBase::ResetResult> getResetResult() const;
        std::shared_ptr<DiscreteBatchEnvBase::StepResult> getStepResult() const;
    private:
        anet::TensorDict createEmptyObsDict() const;
    protected:
        int64_t num_envs_;
        std::vector<std::shared_ptr<SingleDiscreteEnv>> envs_;
        std::unique_ptr<EnvSpec> spec_;
        BatchEnvSpec batch_spec_;
        torch::Device device_;

        torch::TensorOptions float_opt_;
        torch::TensorOptions bool_opt_;
    protected:
        std::shared_ptr<ResetResult> reset_result_;
        std::shared_ptr<StepResult> step_result_;
    };

    // ==============

    class VectorizedDiscreteBatchEnv : public DiscreteBatchEnvBase {
    public:
        VectorizedDiscreteBatchEnv(
            const ConfigData& configData,
            std::shared_ptr<SingleDiscreteEnvFactory> factory,
            const std::string& name,
            int num_envs,
            const torch::Device& device,
            std::optional<seed_t> seed = std::nullopt,
            RunMode run_mode = RunMode::Train,
            const std::string& config_prefix = "");

        std::shared_ptr<const BatchResetResult> Reset() override;
        std::shared_ptr<const BatchStepResult> Step(std::shared_ptr<BatchActionInfo> action_info) override;
    };

    class ThreadPoolDiscreteEnv : public DiscreteBatchEnvBase {
    public:
        ThreadPoolDiscreteEnv(
            const ConfigData& configData,
            std::shared_ptr<SingleDiscreteEnvFactory> factory,
            const std::string& name,
            int num_envs,
            const torch::Device& device,
            std::shared_ptr<ThreadPool> pool,
            std::optional<seed_t> seed = std::nullopt,
            RunMode run_mode = RunMode::Train,
            const std::string& config_prefix = "");

        virtual ~ThreadPoolDiscreteEnv();

        std::shared_ptr<const BatchResetResult> Reset() override;
        std::shared_ptr<const BatchStepResult> Step(std::shared_ptr<BatchActionInfo> action_info) override;
        void Shutdown() override;
    private:
        std::shared_ptr<ThreadPool> pool_;
    };

    // ==============

    class EnvRepository {
    public:
        static EnvRepository& Instance()
        {
            static EnvRepository inst;
            return inst;
        }

        /// @todo SingleDiscreteEnvFactory → SingleDiscreteEnvCreator

        using Factory = std::variant<std::shared_ptr<SingleDiscreteEnvFactory>, std::shared_ptr<BatchEnvFactory>>;

        void Regist(std::shared_ptr<SingleDiscreteEnvFactory> factory);
        void Regist(std::shared_ptr<BatchEnvFactory> factory);
        std::shared_ptr<SingleDiscreteEnvFactory> GetSingleDiscreteEnvFactory(const std::string& id) const;
        std::shared_ptr<BatchEnvFactory> GetBatchEnvFactory(const std::string& id) const;
    private:
        EnvRepository() = default;

        mutable std::mutex mtx_;
        std::unordered_map<std::string, Factory> factories_;
    };

    inline void RegistEnvFactory(std::shared_ptr<SingleDiscreteEnvFactory> factory)
    {
        EnvRepository::Instance().Regist(factory);
    }

    inline void RegistEnvFactory(std::shared_ptr<BatchEnvFactory> factory)
    {
        EnvRepository::Instance().Regist(std::move(factory));
    }

    // ==============

    namespace WorkerThreadAuto
    {
        static constexpr int INVALID =0;
        static constexpr int AUTO = -1;                ///< 自動＝min(EnvCount, 論理コア数 - 2)
        static constexpr int ENV_COUNT = -2;           ///< num_envs 固定
        static constexpr int LOGICAL_CORES = -3;       ///< 論理コア数（HT込み） - 2
        static constexpr int LOGICAL_CORES_EXACT = -4; ///< 論理コア数（HT込み）
        //static constexpr int PHYSICAL_EXACT = -5;      ///< 物理コア数そのまま
    }
    namespace WorkerType {
        static constexpr int AUTO = 0;
        static constexpr int SINGLE_THREAD = 1;
        static constexpr int THREAD_POOL = 2;
    }

    class WorkerThreadResolver {
    public:
        static int Resolve(int worker_threads, int work_items);
    private:
        static int GetLogicalCores();
    };

    struct BatchEnvBuilderConfig : public anet::Config
    {
        std::string class_id;
        int worker_threads = WorkerThreadAuto::AUTO; ///< 負値は自動設定
        int device_type = 0;   ///< 0=cpu 1=cuda
        int device_index = -1; ///< GPU index -1=current device
        int worker_type = WorkerType::AUTO;

        BatchEnvBuilderConfig(const ConfigData& config_data = EmptyConfigData)
            : anet::Config(config_data, "env")
        {
            ANET_READ_CONFIG(config_data, class_id);
            ANET_READ_CONFIG(config_data, worker_threads);
            ANET_READ_CONFIG(config_data, device_type);
            ANET_READ_CONFIG(config_data, device_index);
            ANET_READ_CONFIG(config_data, worker_type);
        }
    };

    class BatchEnvBuilder {
    public:
        BatchEnvBuilder(
            const BatchEnvBuilderConfig& config,
            const ConfigData& config_data,
            int num_envs = 1,
            std::optional<const torch::Device> device = std::nullopt);

        std::shared_ptr<BatchEnv> CreateBatchEnv(
            const std::string& name,
            std::optional<seed_t> seed = std::nullopt,
            int num_envs = -1,
            RunMode run_mode = RunMode::Train,
            const std::string& config_prefix = "");
        void ValidateConfig(RunMode run_mode, const std::string& config_prefix = "") const;
        std::shared_ptr<SingleDiscreteEnvFactory> GetSingleFactory() const;
        torch::Device GetDevice() const { return device_; }
    public:
    private:
        std::shared_ptr<ThreadPool> CreatePool(int worker_threads) const;
        int ResolveWorkerThreads(int num_envs) const;
    private:
        BatchEnvBuilderConfig config_;
        ConfigData config_data_;
        int num_envs_;
        torch::Device device_;
    };
}
