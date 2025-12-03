#pragma once

#include "anet/rl.hpp"

namespace anet::rl {

    class VectorizedDiscreteBatchEnv : public BatchEnv, public RandomHolder {
    public:
        VectorizedDiscreteBatchEnv(
            std::shared_ptr<SingleDiscreteEnvFactory> factory, int batch_size, std::optional<seed_t> seed = std::nullopt);

        EnvSpec GetSpec() const override;
        BatchEnvSpec GetBatchSpec() const override;
        BatchState Reset(RunMode mode) override;
        BatchStepResult Step(const torch::Tensor& actions, RunMode mode) override;
        BatchState GetState() const override;
    private:
        anet::rl::BatchState createEmptyState() const;
        anet::rl::BatchStepResult createEmptyStepResult() const;
    private:
        int64_t batch_size_;
        std::vector<std::unique_ptr<SingleDiscreteEnv>> envs_;
        std::unique_ptr<EnvSpec> spec_;
        BatchEnvSpec batch_spec_;

        std::vector<int64_t> obs_dims_;
        torch::TensorOptions float_opts_;
        torch::TensorOptions bool_opts_;

        BatchState state_;   ///< 現在のState
    };

    class ThreadPoolDiscreteEnv : public BatchEnv {
    public:
        ThreadPoolDiscreteEnv(std::shared_ptr<SingleDiscreteEnvFactory> factory,
            int batch_size,
            int worker_threads);

        EnvSpec GetSpec() const override;
        BatchEnvSpec GetBatchSpec() const override;
        BatchState Reset(RunMode mode) override;
        BatchStepResult Step(const torch::Tensor& actions, RunMode mode) override;
    private:
        struct StepTask {
            SingleDiscreteEnv* env;
            int64_t action;
            SingleStepResult result;
        };

        std::vector<std::shared_ptr<SingleDiscreteEnv>> envs_;
        EnvSpec spec_;
        BatchEnvSpec batch_spec_;

        // @todo: ThreadPool 実装 or 外部注入
        // ThreadPool pool_;
    };

}
