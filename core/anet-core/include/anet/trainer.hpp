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
        StepCounts DoUpdateFrame(int max_steps,
            ControlFunction pre_step_func = noop, ControlFunction post_step_func = noop);
    public:
        std::optional<float> GetScalar(const std::string& key) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key) const override { return std::nullopt; }
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key) const override { return std::nullopt; }
    public:
        TrainerStatus GetStatus() const override { return status_; }
        StepCounts GetCounts() const override { return step_counts_; }
        std::shared_ptr<anet::rl::BatchEnv> GetBatchEnv()const override { return env_; }
        std::shared_ptr<anet::rl::Agent> GetAgent() const override { return agent_; }
        std::shared_ptr<anet::rl::Notifier> GetNotifier() const override { return notifier_; }
    private:
        // 内部状態
        TrainerStatus status_ = TrainerStatus::NOT_INITIALIZED;

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

} // namespace anet::rl
