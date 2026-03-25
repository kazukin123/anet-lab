#pragma once
#include <functional>
#include <optional>
#include "anet/config.hpp"
#include "anet/thread.hpp"
#include "anet/rl.hpp"

namespace anet::rl {

    // ==============

    class DiscreteBatchEnvBase : public BatchEnv, public RandomHolder {
    public:
        DiscreteBatchEnvBase(
            const ConfigData& config_data,
            std::shared_ptr<SingleDiscreteEnvFactory> factory,
            int batch_size,
            const torch::Device& device,
            std::optional<seed_t> seed,
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
        std::shared_ptr<DiscreteBatchEnvBase::ResetResult> createEmptyResetResult() const;
        std::shared_ptr<DiscreteBatchEnvBase::StepResult> createEmptyStepResult() const;
    protected:
        std::shared_ptr<DiscreteBatchEnvBase::ResetResult> getResetResult() const;
        std::shared_ptr<DiscreteBatchEnvBase::StepResult> getStepResult() const;
    protected:
        int64_t batch_size_;
        std::vector<std::shared_ptr<SingleDiscreteEnv>> envs_;
        std::unique_ptr<EnvSpec> spec_;
        BatchEnvSpec batch_spec_;
        torch::Device device_;

        std::vector<int64_t> obs_dims_;
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
            int batch_size,
            const torch::Device& device,
            std::optional<seed_t> seed = std::nullopt,
            const std::string& config_prefix = "");

        std::shared_ptr<const BatchResetResult> Reset(RunMode mode) override;
        std::shared_ptr<const BatchStepResult> Step(std::shared_ptr<BatchActionInfo> action_info, RunMode mode) override;
    };

    class ThreadPoolDiscreteEnv : public DiscreteBatchEnvBase {
    public:
        ThreadPoolDiscreteEnv(
            const ConfigData& configData,
            std::shared_ptr<SingleDiscreteEnvFactory> factory,
            int batch_size,
            const torch::Device& device,
            std::shared_ptr<ThreadPool> pool,
            std::optional<seed_t> seed = std::nullopt,
            const std::string& config_prefix = "");

        virtual ~ThreadPoolDiscreteEnv();

        std::shared_ptr<const BatchResetResult> Reset(RunMode mode) override;
        std::shared_ptr<const BatchStepResult> Step(std::shared_ptr<BatchActionInfo> action_info, RunMode mode) override;
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

        void Regist(std::shared_ptr<SingleDiscreteEnvFactory> factory);
        std::shared_ptr<SingleDiscreteEnvFactory> GetSingleDiscreteEnvFactory(const std::string& id) const;
    private:
        EnvRepository() = default;

        mutable std::mutex mtx_;
        std::unordered_map<std::string, std::shared_ptr<SingleDiscreteEnvFactory>> factories_;
    };

    inline void RegistEnvFactory(std::shared_ptr<SingleDiscreteEnvFactory> factory)
    {
        EnvRepository::Instance().Regist(factory);
    }

    // ==============

    namespace WorkerThreadAuto
    {
        static constexpr int INVALID =0;
        static constexpr int AUTO = -1;                ///< 自動＝min(EnvCount, 論理コア数 - 2)
        static constexpr int ENV_COUNT = -2;           ///< batch_size 固定
        static constexpr int LOGICAL_CORES = -3;       ///< 論理コア数（HT込み） - 2
        static constexpr int LOGICAL_CORES_EXACT = -4; ///< 論理コア数（HT込み）
        //static constexpr int PHYSICAL_EXACT = -5;      ///< 物理コア数そのまま
    }
    namespace WorkerType {
        static constexpr int AUTO = 0;
        static constexpr int SINGLE_THREAD = 1;
        static constexpr int THREAD_POOL = 2;
    }

    struct DefaultBatchEnvFactoryConfig : public anet::Config
    {
        std::string class_id;
        int worker_threads = WorkerThreadAuto::AUTO; ///< 負値は自動設定
        int device_type = 0;   ///< 0=cpu 1=cuda
        int device_index = -1; ///< GPU index -1=current device
        int worker_type = WorkerType::AUTO;
 
        DefaultBatchEnvFactoryConfig(const ConfigData& config_data = EmptyConfigData)
            : anet::Config(config_data, "env")
        {
            ANET_READ_CONFIG(config_data, class_id);
            ANET_READ_CONFIG(config_data, worker_threads);
            ANET_READ_CONFIG(config_data, device_type);
            ANET_READ_CONFIG(config_data, device_index);
            ANET_READ_CONFIG(config_data, worker_type);
        }
    };

    class DefaultBatchEnvFactory : public BatchEnvFactory {
    public:
        DefaultBatchEnvFactory(
            const DefaultBatchEnvFactoryConfig& config,
            const ConfigData& config_data,
            int batch_size = 1,
            std::optional<const torch::Device> device = std::nullopt);

        std::shared_ptr<BatchEnv> CreateBatchEnv(std::optional<seed_t> seed = std::nullopt, int batch_size = -1) override;
        std::shared_ptr<SingleDiscreteEnvFactory> GetSingleFactory() const;
        torch::Device GetDevice() const { return device_; }
    public:
    private:
        std::shared_ptr<ThreadPool> CreatePool(int worker_threads) const;
        int GetLogicalCores() const;
        int ResolveWorkerThreads(int batch) const;
    private:
        DefaultBatchEnvFactoryConfig config_;
        ConfigData config_data_;
        int batch_size_;
        torch::Device device_;
    };
}

#define ANET_REGISTER_ENV_FACTORY(FactoryType) \
    namespace { \
        struct FactoryType##AutoRegister { \
            FactoryType##AutoRegister() { \
                anet::rl::RegistEnvFactory(std::make_shared<FactoryType>()); \
            } \
        }; \
        static FactoryType##AutoRegister global_##FactoryType##_auto_register; \
    }


