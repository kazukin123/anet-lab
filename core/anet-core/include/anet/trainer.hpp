// anet/trainer.hpp
#pragma once

#include <memory>
#include <chrono>
#include "anet/util.hpp"
#include "anet/rl.hpp"

namespace anet::rl {

    class DefaultTrainer : public Trainer {
    public:
        DefaultTrainer(const ConfigData& config_data);

        TrainerStatus Initialize(const ConfigData& config_data);
        StepCounts DoStep();
        StepCounts DoUpdateFrame(int max_steps,
            ControlFunction pre_step_func = noop, ControlFunction post_step_func = noop);

        virtual ~DefaultTrainer() = default;
    public:
        std::optional<float> GetScalar(const std::string& key, int index = -1) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key, int index = -1) const override { return std::nullopt; }
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key, int index = -1) const override { return std::nullopt; }
    public:
        TrainerStatus GetStatus() const override { return status_; }
        StepCounts GetCounts() const override { return step_counts_; }
        std::shared_ptr<anet::rl::BatchEnv> GetBatchEnv()const override { return env_; }
        std::shared_ptr<anet::rl::Agent> GetAgent() const override { return agent_; }
        std::shared_ptr<anet::rl::Notifier> GetNotifier() const override { return notifier_; }
    private:
        // 内部状態
        TrainerStatus status_ = TrainerStatus::NOT_INITIALIZED;
        bool env_initialized_ = false;

        // パラメータ
        struct Config;
        std::unique_ptr<Config> config_;

        // 乱数
        std::unique_ptr<anet::MasterSeedManager> master_seed_;

        // 強化学習関連
        anet::rl::StepCounts step_counts_;
        std::shared_ptr<anet::rl::BatchEnv> env_;
        std::shared_ptr<anet::rl::Agent> agent_;
        anet::rl::BatchState state_;

        // メトリクス
        std::shared_ptr<anet::rl::Notifier> notifier_;
        std::chrono::high_resolution_clock::time_point start_time_;
        std::chrono::high_resolution_clock::time_point last_time_;
        anet::rl::step_t last_exp_step_ = 0;
        float last_train_reward_ = 0.0f;
        anet::EmaFilter<float> train_reward_ema_;
        float last_target_eval_reward_ = 0.0f;
        float last_policy_eval_reward_ = 0.0f;
        float last_train_step_per_sec_ = std::numeric_limits<float>::quiet_NaN();
        float last_exp_step_per_sec_ = std::numeric_limits<float>::quiet_NaN();
    };

    class AsyncTrainerRunner {
    public:
        explicit AsyncTrainerRunner(std::shared_ptr<anet::rl::Trainer> trainer,
            anet::rl::Trainer::ControlFunction pre_func = anet::rl::Trainer::noop,
            anet::rl::Trainer::ControlFunction post_func = anet::rl::Trainer::noop);
        ~AsyncTrainerRunner();

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
        std::shared_ptr<anet::rl::Trainer> trainer_;
        std::thread worker_;
        anet::rl::Trainer::ControlFunction pre_func_;
        anet::rl::Trainer::ControlFunction post_func_;
    };

} // namespace anet::rl
