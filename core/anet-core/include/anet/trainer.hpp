// anet/trainer.hpp
#pragma once

#include <memory>
#include <chrono>
#include "anet/util.hpp"
#include "anet/env.hpp"
#include "anet/rl.hpp"

namespace anet::rl {

    class RunnerBase : public Runner {
    public:
        RunnerBase();
        RunnerBase(std::shared_ptr<BatchEnv> env);
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
        std::shared_ptr<anet::rl::BatchEnv> GetBatchEnv()const override { return env_; }
        std::shared_ptr<anet::rl::Agent> GetAgent() const override { return agent_; }
        std::shared_ptr<anet::rl::Notifier> GetNotifier() const override { return notifier_; }
    protected:
        // 内部状態
        RunnerStatus status_ = RunnerStatus::NOT_INITIALIZED;
        bool env_initialized_ = false;

        // 強化学習関連
        anet::rl::StepCounts step_counts_;
        std::shared_ptr<anet::rl::BatchEnv> env_;
        std::shared_ptr<anet::rl::Agent> agent_;
        anet::rl::BatchState state_;
        std::shared_ptr<anet::rl::Notifier> notifier_;
    };


    class EvalRunner : public RunnerBase {
    public:
        EvalRunner(std::shared_ptr<BatchEnv> env, std::shared_ptr<const Agent> agent, RunMode runmode = RunMode::Eval);

        RunnerStatus Initialize(const ConfigData& config_data);
        StepCounts DoStep(int64_t action);
        StepCounts DoStep() override;

        ~EvalRunner() = default;
    private:
        std::shared_ptr<const anet::rl::Agent> agent_;
        RunMode runmode_;
    };


    class DefaultTrainer : public RunnerBase {
    public:
        DefaultTrainer(const ConfigData& config_data, const std::string& config_prefix = "train");

        RunnerStatus Initialize(const ConfigData& config_data);
        StepCounts DoStep() override;
        std::shared_ptr<EvalRunner> CreateEvalRunner(RunMode runmode = RunMode::Eval) const;
        std::optional<float> GetScalar(const std::string& key, int64_t index = -1) const override;
        std::string GetEnvClassId() const { return env_class_id_; }

        virtual ~DefaultTrainer() = default;
    private:
        // パラメータ
        struct Config;
        std::unique_ptr<Config> config_;

        // 乱数
        std::unique_ptr<anet::rl::DefaultBatchEnvFactory> env_factory_;
        std::unique_ptr<anet::MasterSeedManager> master_seed_;
        seed_t eval_env_seed_;
    private:
        // Trainer情報
        std::string env_class_id_;

        // メトリクス
        std::chrono::high_resolution_clock::time_point start_time_;
        std::chrono::high_resolution_clock::time_point last_time_;
        float last_train_reward_ = 0.0f;
        anet::EmaFilter<float> train_reward_ema_;
        float last_target_eval_reward_ = 0.0f;
        float last_policy_eval_reward_ = 0.0f;
        torch::Tensor episode_total_reward_cur_;        ///< Trainのエピソード単位総報酬を集計するために現在値
        torch::Tensor episode_total_reward_comp_;       ///< Trainのエピソード単位総報酬

        step_t acc_train_steps_ = 0;
        step_t acc_exp_steps_ = 0;
        anet::rl::step_t last_train_step_ = 0;
        anet::rl::step_t last_exp_step_ = 0;
        float last_train_step_per_sec_ = std::numeric_limits<float>::quiet_NaN();
        float last_exp_step_per_sec_ = std::numeric_limits<float>::quiet_NaN();
    };

    class RunnerThread {
    public:
        using ExceptionFunction = std::function<void()>;
        static void noop(void) { }
    public:
        explicit RunnerThread(std::shared_ptr<anet::rl::Runner> runner,
            anet::rl::Runner::ControlFunction pre_func = nullptr,
            anet::rl::Runner::ControlFunction post_func = nullptr,
            ExceptionFunction exception_func = nullptr);
        ~RunnerThread();

        void Start();

        /// Trainerスレッド停止＆停止待ち合わせ
        void Stop();

        // フラグ取得/設定
        bool IsRunning() const { return running_.load(); }
        bool IsPaused() const { return paused_.load(); }
        void Pause() { paused_.store(true); }
        void Resume() { paused_.store(false); }
    private:
        void ThreadMain();
    private:
        std::atomic<bool> running_{ false }; ///< thread実行中フラグ
        std::atomic<bool> paused_{ false }; ///< thread実行中フラグ
        std::shared_ptr<anet::rl::Runner> runner_;
        std::thread worker_;
        anet::rl::Runner::ControlFunction pre_func_;
        anet::rl::Runner::ControlFunction post_func_;
        ExceptionFunction exception_func_;
    };

} // namespace anet::rl
