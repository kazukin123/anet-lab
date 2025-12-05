#pragma once
#include <functional>
#include <optional>
#include "anet/config.hpp"
#include "anet/rl.hpp"

namespace anet::rl {

    class DiscreteBatchEnvBase : public BatchEnv, public RandomHolder {
    public:
        DiscreteBatchEnvBase(
            std::shared_ptr<SingleDiscreteEnvFactory> factory,
            int batch_size,
            const torch::Device& device,
            std::optional<seed_t> seed);

        EnvSpec GetSpec() const override;
        BatchEnvSpec GetBatchSpec() const override;
    protected:
        anet::rl::BatchState createEmptyState() const;
        anet::rl::BatchStepResult createEmptyStepResult() const;
    protected:
        int64_t batch_size_;
        std::vector<std::unique_ptr<SingleDiscreteEnv>> envs_;
        std::unique_ptr<EnvSpec> spec_;
        BatchEnvSpec batch_spec_;
        torch::Device device_;

        std::vector<int64_t> obs_dims_;
        torch::TensorOptions float_opt_;
        torch::TensorOptions bool_opt_;
    };

    class VectorizedDiscreteBatchEnv : public DiscreteBatchEnvBase {
    public:
        VectorizedDiscreteBatchEnv(
            std::shared_ptr<SingleDiscreteEnvFactory> factory,
            int batch_size,
            const torch::Device& device,
            std::optional<seed_t> seed = std::nullopt);

        BatchState Reset(RunMode mode) override;
        BatchStepResult Step(const torch::Tensor& actions, RunMode mode) override;
    };

    class ThreadPool {
    public:
        using TaskFunction = std::function<void()>;

        virtual void Enqueue(int worker_id, TaskFunction fn) = 0;
        virtual void WaitAll() = 0;
        virtual int GetWorkerCount() const = 0;

        virtual ~ThreadPool() = default;
    };

    class PinnedThreadPool : public ThreadPool {
    public:
        explicit PinnedThreadPool(int worker_count);
        ~PinnedThreadPool();

        int GetWorkerCount() const override { return worker_count_; }
        void Enqueue(int worker_id, TaskFunction fn) override;
        void WaitAll() override;
    public:
        // コピー不可
        PinnedThreadPool(const PinnedThreadPool&) = delete;
        PinnedThreadPool(PinnedThreadPool&&) = delete;
        PinnedThreadPool& operator=(const PinnedThreadPool&) = delete;
        PinnedThreadPool& operator=(PinnedThreadPool&&) = delete;
    private:
        int worker_count_;
        std::unique_ptr<std::thread[]> workers_;

        std::unique_ptr<std::mutex[]> mutexes_;                 ///< タスクキューの排他用
        std::unique_ptr<std::condition_variable[]> cvs_;        ///< タスクキューとstop_flagの待ち合わせ用
        std::unique_ptr<std::deque<TaskFunction>[]> queues_;    ///< 実行待ちタスクキュー

        std::atomic<bool> stop_flag_;       ///< 全スレッド終了フラグ（デストラクト用）
        std::atomic<int> pending_tasks_;    ///< PinnedThreadPool全体の実行待ち/実行中タスク数
    private:
        void WorkerLoop(int wid);
        void Stop();
    };

    class ThreadPoolDiscreteEnv : public DiscreteBatchEnvBase {
    public:
        ThreadPoolDiscreteEnv(
            std::shared_ptr<SingleDiscreteEnvFactory> factory,
            int batch_size,
            const torch::Device& device,
            std::shared_ptr<ThreadPool> pool,
            std::optional<seed_t> seed = std::nullopt);

        BatchState Reset(RunMode mode) override;
        BatchStepResult Step(const torch::Tensor& a, RunMode mode) override;
    private:
        std::shared_ptr<ThreadPool> pool_;
    };

    namespace WorkerThreadAuto
    {
        static constexpr int INVALID =0;
        static constexpr int AUTO = -1;                ///< 自動＝min(EnvCount, 論理コア数 - 2)
        static constexpr int ENV_COUNT = -2;           ///< batch_size 固定
        static constexpr int LOGICAL_CORES = -3;       ///< 論理コア数（HT込み） - 2
        static constexpr int LOGICAL_CORES_EXACT = -4; ///< 論理コア数（HT込み）
        //static constexpr int PHYSICAL_EXACT = -5;      ///< 物理コア数そのまま
    }

    struct DefaultBatchEnvFactoryConfig : public anet::Config
    {
        std::string env_class_id;
        int worker_threads = WorkerThreadAuto::AUTO; ///< 負値は自動設定
        int device_type = 0;   ///< 0=cpu 1=cuda
        int device_index = -1; ///< GPU index -1=current device

        DefaultBatchEnvFactoryConfig() {}
        DefaultBatchEnvFactoryConfig(const ConfigData& configData)
            : anet::Config(configData, "BatchEnvFactory" "DefaultBatchEnvFactory")
        {
            ANET_APPLY_CONFIG(configData, env_class_id);
            ANET_APPLY_CONFIG(configData, worker_threads);
            ANET_APPLY_CONFIG(configData, device_type);
            ANET_APPLY_CONFIG(configData, device_index);
        }
    };

    class EnvRepository {
    public:
        static EnvRepository& Instance() {
            static EnvRepository inst;
            return inst;
        }

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

    class DefaultBatchEnvFactory : public BatchEnvFactory {
    public:
    public:
        DefaultBatchEnvFactory(
            const DefaultBatchEnvFactoryConfig& config,
            int batch_size = 1,
            std::optional<seed_t> seed = std::nullopt,
            std::optional<const torch::Device> device = std::nullopt);

        std::shared_ptr<BatchEnv> CreateBatchEnv() override;
        std::shared_ptr<SingleDiscreteEnvFactory> GetSingleFactory() const;
        torch::Device GetDevice() const { return device_; }
    public:
    private:
        std::shared_ptr<ThreadPool> createPool(int worker_threads) const;
        int getLogicalCores() const;
        int getPhysicalCores() const;
        int resolveWorkerThreads(int batch) const;
    private:
        DefaultBatchEnvFactoryConfig config_;
        int batch_size_;
        std::optional<seed_t> seed_;
        torch::Device device_;
    };
}

#define ANET_REGIST_ENV_FACTORY(FactoryType) \
    namespace { \
        struct FactoryType##AutoRegister { \
            FactoryType##AutoRegister() { \
                anet::rl::RegistEnvFactory(std::make_shared<FactoryType>()); \
            } \
        }; \
        static FactoryType##AutoRegister global_##FactoryType##_auto_register; \
    }


