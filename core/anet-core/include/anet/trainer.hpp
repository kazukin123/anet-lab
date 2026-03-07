// anet/trainer.hpp
#pragma once

#include <memory>
#include <vector>
#include <chrono>
#include "anet/util.hpp"
#include "anet/thread.hpp"
#include "anet/env.hpp"
#include "anet/rl.hpp"

namespace anet::rl {

    class RunnerBase : public Runner {
    public:
        RunnerBase(
            std::shared_ptr<anet::rl::BatchEnv> env,
            std::shared_ptr<anet::rl::Agent> agent,
            std::shared_ptr<anet::rl::Notifier> notifier,
            RunMode runmode);

        virtual StepCounts DoStep() = 0;

        StepCounts DoUpdateFrame(
            int max_steps, ControlFunction pre_step_func = nullptr, ControlFunction post_step_func = nullptr) override;

        virtual ~RunnerBase() = default;
    public:
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override { return std::nullopt; }
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override { return std::nullopt; }
    public:
        RunnerStatus GetStatus() const override { return status_; }
        StepCounts GetCounts() const override { return step_counts_; }
        std::shared_ptr<anet::rl::BatchEnv> GetBatchEnv() const override { return env_; }
        std::shared_ptr<anet::rl::Agent> GetAgent() const override { return agent_; }
        std::shared_ptr<anet::rl::Notifier> GetNotifier() const override { return notifier_; }
    protected:
        void InitializeMetrics();
        void UpdateMetrics(std::shared_ptr<const BatchStepResult> result);
    protected:
        // 内部状態
        RunnerStatus status_ = RunnerStatus::NOT_INITIALIZED;
        bool env_initialized_ = false;

        // 強化学習関連
        anet::rl::StepCounts step_counts_;
        std::shared_ptr<anet::rl::BatchEnv> env_;
        std::shared_ptr<anet::rl::Agent> agent_;
        std::shared_ptr<anet::rl::Notifier> notifier_;
        anet::rl::BatchState state_;
        std::shared_ptr<ActionContext> action_context_ = nullptr;
        RunMode runmode_;

        // メトリクス
        //std::chrono::high_resolution_clock::time_point start_time_;
        //std::chrono::high_resolution_clock::time_point last_time_;
        float last_reward_ = 0.0f;
        anet::EmaFilter<float> reward_ema_;
        torch::Tensor episode_total_reward_cur_;        ///< エピソード単位総報酬を集計するために現在値
        torch::Tensor episode_total_reward_comp_;       ///< エピソード単位総報酬
    };


    class EvalRunner final : public RunnerBase, public std::enable_shared_from_this<EvalRunner> {
    public:
        EvalRunner(
            std::shared_ptr<anet::rl::BatchEnv> env,
            std::shared_ptr<anet::rl::Agent> agent,
            std::shared_ptr<anet::rl::Notifier> notifier,
            RunMode runmode = RunMode::Eval);

        void Shutdown() override { }

        //RunnerStatus Initialize(const ConfigData& config_data);
        StepCounts DoStep(int64_t action);
        StepCounts DoStep() override;
    };

    class TrainRunner final : public RunnerBase, public std::enable_shared_from_this<TrainRunner> {
    public:
        TrainRunner(
            std::shared_ptr<anet::rl::BatchEnv> env,
            std::shared_ptr<anet::rl::Agent> agent,
            std::shared_ptr<anet::rl::Notifier> notifier);

        RunnerStatus Initialize(const ConfigData& config_data);
        StepCounts DoStep() override;
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::string GetEnvClassId() const { return env_class_id_; }
        void SetEvalLastReward(const std::string& name, float val);

        void Shutdown() override;
    private:
        // 乱数
    private:
        // Trainer情報
        std::string env_class_id_;

        // メトリクス
        std::chrono::high_resolution_clock::time_point start_time_;
        std::chrono::high_resolution_clock::time_point last_time_;
		std::unordered_map<std::string, float> eval_last_rewards_;

        step_t acc_train_steps_ = 0;
        step_t acc_exp_steps_ = 0;
        anet::rl::step_t last_train_step_ = 0;
        anet::rl::step_t last_exp_step_ = 0;
        float last_train_step_per_sec_ = std::numeric_limits<float>::quiet_NaN();
        float last_exp_step_per_sec_ = std::numeric_limits<float>::quiet_NaN();
    };


    // ----------------------------------------------------------------------
    // RunManager
    // ----------------------------------------------------------------------

    class RunManager {
    public:
        //RunManager(const ConfigData& config_data, const std::string& config_prefix = "train");
        RunManager(const ConfigData& config_data);
        ~RunManager();

        bool Initialize(const ConfigData& config_data);

        std::shared_ptr<EvalRunner> CreateEvalRunner(const std::string& name, RunMode runmode = RunMode::Eval);

        // アクセサ
        //std::shared_ptr<anet::rl::BatchEnv> GetBatchEnv() const { return env_; }  // EnvはRunnerインスタンス別なので隠蔽
        std::string GetEnvClassId() const { return env_class_id_; }
        std::shared_ptr<anet::rl::Agent> GetAgent() const { return agent_; }
        std::shared_ptr<anet::rl::Notifier> GetNotifier() const { return notifier_; }
        std::shared_ptr<TrainRunner> GetTrainRunner() { return train_runner_; }
        //std::shared_ptr<EvalRunner> GetEvalRunner(const std::string& name) { return eval_runners[name]; }
        anet::rl::RunnerStatus GetStatus() { return status_; }
    private:
        // パラメータ
        struct Config;
        std::unique_ptr<Config> config_;
        std::string env_class_id_;
        seed_t eval_env_seed_;

        // インスタンス(管理系)
        anet::rl::RunnerStatus status_ = RunnerStatus::NOT_INITIALIZED;
        std::unique_ptr<anet::rl::DefaultBatchEnvFactory> env_factory_;
        std::unique_ptr<anet::MasterSeedManager> master_seed_;
        std::shared_ptr<TrainRunner> train_runner_;
        std::unordered_map<std::string, std::shared_ptr<EvalRunner>> eval_runners;

        // インスタンス(共有)
        std::shared_ptr<anet::rl::BatchEnv> env_;
        std::shared_ptr<anet::rl::Agent> agent_;
        std::shared_ptr<anet::rl::Notifier> notifier_;

        // メトリクス
        //std::unordered_map<std::string, float> eval_last_rewards_;
    };


    // ----------------------------------------------------------------------
    // RunnerThread
    // ----------------------------------------------------------------------

    class RunnerThread : public anet::ThreadBase {
    public:
        using ExceptionFunction = std::function<void()>;
        static void noop(void) {}
    public:
        explicit RunnerThread(
            const std::string& name,std::shared_ptr<anet::rl::Runner> runner,
            anet::rl::Runner::ControlFunction pre_func = nullptr, anet::rl::Runner::ControlFunction post_func = nullptr, ExceptionFunction exception_func = nullptr);

        ~RunnerThread() override;
    protected:
        bool ProcessStep() override;
        void OnException() override;

    private:
        std::shared_ptr<anet::rl::Runner> runner_;
        anet::rl::Runner::ControlFunction pre_func_;
        anet::rl::Runner::ControlFunction post_func_;
        ExceptionFunction exception_func_;
    };

} // namespace anet::rl
