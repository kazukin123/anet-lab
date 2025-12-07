// anet/trainer.hpp
#pragma once

#include <memory>
#include <chrono>
#include "anet/util.hpp"
#include "anet/rl.hpp"

namespace anet::rl {

    class DefaultTrainer : public Trainer {
    public:
		using ControlFunction = std::function<bool()>;
        static bool noop_function() { return false; };
    public:
        DefaultTrainer(const ConfigData& config_data);

        void DoUpdateFrame(int max_step,
            ControlFunction pre_step_func = noop_function,     ///< 学習ステップ実行前処理(bool戻り値trueで中断要求)
            ControlFunction post_step_func = noop_function);   ///< 学習ステップ実行後処理(bool戻り値trueで中断要求)

        std::optional<float> GetScalar(const std::string& key) const override;
        std::optional<torch::Tensor> GetTensor(const std::string& key) const override { return std::nullopt; }
        std::optional<std::vector<torch::Tensor>> GetTensorVector(const std::string& key) const override { return std::nullopt; }

        const StepCounts& GetCounts() const { return step_counts_; }
        std::shared_ptr<anet::rl::BatchEnv> GetBatchEnv() { return env_; }
        std::shared_ptr<anet::rl::Agent> GetAgent() { return agent_; }
        std::shared_ptr<anet::rl::Notifier> GetNotifier() { return notifier_; }
    private:
        void Initialize(const ConfigData& config_data);
    private:
        // パラメータ
        struct Config;
        std::unique_ptr<Config> config_;

        // デバイス
        torch::Device device_agent_;

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
