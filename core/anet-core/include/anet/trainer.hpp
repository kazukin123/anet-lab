// anet/trainer.hpp
#pragma once

#include <memory>
#include <vector>
#include <chrono>
#include <limits>
#include "anet/util.hpp"
#include "anet/thread.hpp"
#include "anet/env.hpp"
#include "anet/rl.hpp"

namespace anet::rl {


    // ======================================================
    // RunnerBase
    // ======================================================
    
    class RunnerBase : public Runner {
    public:
        RunnerBase(
            std::shared_ptr<anet::rl::BatchEnv> env,
            std::shared_ptr<anet::rl::Agent> agent,
            std::shared_ptr<anet::rl::Notifier> notifier,
            RunMode run_mode,
            std::optional<bool> clone_model_override,
            std::optional<torch::Device> device,
            std::string name);

        virtual StepCounts DoStep() = 0;

        StepCounts DoUpdateFrame(
            int max_steps, ControlFunction pre_step_func = nullptr, ControlFunction post_step_func = nullptr) override;

        virtual ~RunnerBase() = default;
    public:
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int64_t index = -1) const override { return std::nullopt; }
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int64_t index = -1) const override { return std::nullopt; }
        bool LastStepHadEpisodeEnd() const { return last_step_had_episode_end_; }
    public:
        RunnerStatus GetStatus() const override { return status_; }
        StepCounts GetCounts() const override { return step_counts_; }
        const std::string& GetName() const override { return name_; }
        std::shared_ptr<anet::rl::BatchEnv> GetBatchEnv() const override { return env_; }
        std::shared_ptr<anet::rl::Agent> GetAgent() const override { return agent_; }
        std::shared_ptr<anet::rl::Notifier> GetNotifier() const override { return notifier_; }
    protected:
        void InitializeMetrics();
        void UpdateMetrics(std::shared_ptr<const BatchStepResult> result);
        bool AccumulateAndNotifyEpisodeEnd(
            std::shared_ptr<const Runner> self,
            std::shared_ptr<const BatchStepResult> result,
            const StepCounts& event_counts);
    protected:
        // 内部状態
        std::string name_;
        RunnerStatus status_ = RunnerStatus::NOT_INITIALIZED;
        bool env_initialized_ = false;

        // 強化学習関連
        anet::rl::StepCounts step_counts_;
        std::shared_ptr<anet::rl::BatchEnv> env_;
        std::shared_ptr<anet::rl::Agent> agent_;
        std::shared_ptr<anet::rl::Notifier> notifier_;
        anet::rl::BatchState state_;
        std::shared_ptr<Actor> actor_ = nullptr;
        RunMode run_mode_;

        // メトリクス
        //std::chrono::high_resolution_clock::time_point start_time_;
        //std::chrono::high_resolution_clock::time_point last_time_;
        float last_reward_ = 0.0f;
        anet::EmaFilter<float> reward_ema_;
        torch::Tensor episode_total_reward_cur_;        ///< エピソード単位総報酬を集計するために現在値
        torch::Tensor episode_total_reward_comp_;       ///< エピソード単位総報酬
        std::vector<float> eps_total_reward_per_env_;   ///< EpisodeEndEvent用のENV単位総報酬累積
        float last_episode_total_reward_ = std::numeric_limits<float>::quiet_NaN();
        bool last_step_had_episode_end_ = false;
    };


    // ======================================================
    // EvalRunner (評価用Runner）
    // ======================================================

    class EvalRunner final : public RunnerBase, public std::enable_shared_from_this<EvalRunner> {
    public:
        EvalRunner(
            std::shared_ptr<anet::rl::BatchEnv> env,
            std::shared_ptr<anet::rl::Agent> agent,
            std::shared_ptr<anet::rl::Notifier> notifier,
            RunMode run_mode = RunMode::Eval,
            bool clone_model = false,
            std::optional<torch::Device> device = std::nullopt,
            std::string name = "eval");

        void Sync();
        void Shutdown() override { }

        //RunnerStatus Initialize(const ConfigData& config_data);
        StepCounts DoStep(int64_t action);
        StepCounts DoStep(int64_t action, const StepCounts& event_counts);
        StepCounts DoStep(const StepCounts& event_counts);
        StepCounts DoStep() override;
    };


    // ======================================================
    // TrainRunner (Train系Runnerの基底クラス)
    // ======================================================

    class TrainRunner : public RunnerBase, public std::enable_shared_from_this<TrainRunner> {
    public:
        TrainRunner(
            std::shared_ptr<anet::rl::BatchEnv> env,
            std::shared_ptr<anet::rl::Agent> agent,
            std::shared_ptr<anet::rl::Notifier> notifier);

        virtual StepCounts DoStep() override = 0;
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::string GetEnvClassId() const { return env_class_id_; }

        void Shutdown() override;

        virtual ~TrainRunner() = default;
    protected:
        void CalcPerformanceMetrics();
    protected:
        // Learner
        std::shared_ptr<Learner> learner_ = nullptr;

        // Trainer情報
        std::string env_class_id_;

        // メトリクス
        std::chrono::high_resolution_clock::time_point start_time_;
        std::chrono::high_resolution_clock::time_point last_time_;
        step_t acc_train_steps_ = 0;
        step_t acc_exp_steps_ = 0;
        anet::rl::step_t last_train_step_ = 0;
        anet::rl::step_t last_exp_step_ = 0;
        float last_train_step_per_sec_ = std::numeric_limits<float>::quiet_NaN();
        float last_exp_step_per_sec_ = std::numeric_limits<float>::quiet_NaN();
    };


    // ======================================================
    // SerialTrainRunner (直列実行)
    // ======================================================
    class SerialTrainRunner final : public TrainRunner {
    public:
        SerialTrainRunner(
            std::shared_ptr<anet::rl::BatchEnv> env,
            std::shared_ptr<anet::rl::Agent> agent,
            std::shared_ptr<anet::rl::Notifier> notifier);

        StepCounts DoStep() override;
    };


    // ======================================================
    // PipelineTrainRunner (並列実行)
    // ======================================================
    class PipelineTrainRunner final : public TrainRunner {
    public:
        PipelineTrainRunner(
            std::shared_ptr<anet::rl::BatchEnv> env,
            std::shared_ptr<anet::rl::Agent> agent,
            std::shared_ptr<anet::rl::Notifier> notifier);

        StepCounts DoStep() override;
        void Shutdown() override;
    private:
        std::unique_ptr<anet::PinnedThreadPool> learn_pool_;
        std::future<anet::rl::BatchUpdateResultList> learn_future_;

        // 1ステップ遅れで学習を投げるための状態保持
        bool has_prev_data_ = false;
        anet::rl::BatchExperience prev_exp_;
        std::shared_ptr<const BatchStepResult> prev_result_;
        std::shared_ptr<const anet::rl::BatchActionInfo> prev_action_info_;
        anet::rl::StepCounts prev_counts_;
    };


    // ======================================================
    // RunnerFactory
    // ======================================================
    class RunnerFactory {
    public:
        static std::shared_ptr<TrainRunner> CreateMainRunner(
            const std::string& type,
            std::shared_ptr<anet::rl::BatchEnv> env,
            std::shared_ptr<anet::rl::Agent> agent,
            std::shared_ptr<anet::rl::Notifier> notifier);
    };


    // ----------------------------------------------------------------------
    // RunManager
    // ----------------------------------------------------------------------

    class RunManager {
    public:
        //RunManager(const ConfigData& config_data, const std::string& config_prefix = "train");
        RunManager(const ConfigData& config_data);
        ~RunManager();

        std::shared_ptr<EvalRunner> CreateEvalRunner(const std::string& name, RunMode runmode = RunMode::Eval, bool clone_model = false, std::optional<torch::Device> device = std::nullopt);

        // アクセサ
        //std::shared_ptr<anet::rl::BatchEnv> GetBatchEnv() const { return env_; }  // EnvはRunnerインスタンス別なので隠蔽
        std::string GetEnvClassId() const { return env_class_id_; }
        std::shared_ptr<anet::rl::Agent> GetAgent() const { return agent_; }
        std::shared_ptr<anet::rl::Notifier> GetNotifier() const { return notifier_; }
        std::shared_ptr<TrainRunner> GetTrainRunner() { return train_runner_; }
        std::shared_ptr<EvalRunner> GetEvalRunner(const std::string& name) { return eval_runners.at(name); }
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
