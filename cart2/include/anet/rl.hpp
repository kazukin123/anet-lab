#pragma once
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <random>

namespace anet::rl {

    // =============================================================
    // 列挙・構造体（RunModeは学習/推論の切替）
    // =============================================================

    enum class RunMode { Train, Eval1, Eval2 };
    inline bool IsTrain(RunMode mode) { return mode == RunMode::Train; }
    inline bool IsEval(RunMode mode) { return mode == RunMode::Eval1 || mode == RunMode::Eval2; }

    /**
     * @brief 環境が返すステップ応答。
     * next_state, reward, done, truncated の基本情報を保持。
     */
    struct EnvResponse {
        torch::Tensor next_state;
        float reward;
        bool done;
        bool truncated;
    };

    /**
     * @brief エージェントの学習に使う「1回の経験」。
     * 状態・行動と、その結果としてのEnvResponseを内包。
     */
    struct Experience {
        torch::Tensor state;
        torch::Tensor action;
        EnvResponse response;
    };

    // =============================================================
    // Logger Interface（最小構成）
    // =============================================================

    class LoggerInterface {
    public:
        virtual void LogScalar(const std::string& tag, float value, int step) = 0;
        virtual void LogText(const std::string& tag, const std::string& msg) = 0;
        virtual void Flush() {}
        virtual ~LoggerInterface() = default;
    };

    class ConsoleLogger : public LoggerInterface {
    public:
        void LogScalar(const std::string& tag, float value, int step) override {
            std::cout << "[" << step << "] " << tag << " = " << value << "\n";
        }
        void LogText(const std::string& tag, const std::string& msg) override {
            std::cout << "[LOG] " << tag << ": " << msg << "\n";
        }
    };

    // =============================================================
    // Environment 抽象クラス（Gym風API）
    // =============================================================

    class Environment {
    public:
        virtual torch::Tensor Reset(anet::rl::RunMode mode = anet::rl::RunMode::Train) = 0;
        virtual EnvResponse DoStep(const torch::Tensor& action, anet::rl::RunMode mode = anet::rl::RunMode::Train) = 0;
        virtual torch::Tensor GetState() const = 0;
        virtual ~Environment() = default;
    };

    // =============================================================
    // BatchData / ReplayBuffer（学習データ格納）
    // =============================================================

    class BatchData {
    public:
        void Add(const Experience& e) { experiences_.push_back(e); }

        void ToDevice(torch::Device device) {
            for (auto& e : experiences_) {
                e.state = e.state.to(device);
                e.action = e.action.to(device);
                e.response.next_state = e.response.next_state.to(device);
            }
        }

        const std::vector<Experience>& Data() const { return experiences_; }
        size_t Size() const { return experiences_.size(); }
        void Clear() { experiences_.clear(); }

    private:
        std::vector<Experience> experiences_;
    };

    class ReplayBuffer {
    public:
        explicit ReplayBuffer(size_t capacity = 10000) : capacity_(capacity) {}

        void Push(const Experience& e) {
            if (buffer_.size() >= capacity_) buffer_.erase(buffer_.begin());
            buffer_.push_back(e);
        }

        std::vector<Experience> Sample(size_t n) const {
            n = std::min(n, buffer_.size());
            std::vector<Experience> batch;
            batch.reserve(n);
            for (size_t i = 0; i < n; ++i)
                batch.push_back(buffer_[rand() % buffer_.size()]);
            return batch;
        }

        size_t Size() const { return buffer_.size(); }

    private:
        size_t capacity_;
        std::vector<Experience> buffer_;
    };

    // =============================================================
    // Agent 抽象クラス
    // =============================================================

    class Agent {
    public:
        /**
         * @brief 状態から行動を選択する（学習・推論共通）。
         * @return (action, log_prob, value)
         */
        virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
            SelectAction(const torch::Tensor& state, RunMode mode = RunMode::Train) = 0;

        /**
         * @brief 1経験を更新・保存（DQN系はReplayBufferにPushなど）。
         */
        virtual void Update(const Experience&) {}

        /**
         * @brief バッチ単位の学習更新（フル／ミニ両対応）。
         */
        virtual void UpdateBatch(const BatchData&) = 0;

        virtual ~Agent() = default;
    };

    // =============================================================
    // OnPolicySession（PPO/A2Cなど用）
    // =============================================================

    class OnPolicySession {
    public:
        explicit OnPolicySession(Agent& agent) : agent_(agent) {}

        void AddExperience(const Experience& e) { batch_.Add(e); }

        void Finalize() {
            if (finalized_) return;
            agent_.UpdateBatch(batch_);
            finalized_ = true;
            batch_.Clear();
        }

        const BatchData& GetBatch() const { return batch_; }

    private:
        Agent& agent_;
        BatchData batch_;
        bool finalized_ = false;
    };

} // namespace anet::rl
