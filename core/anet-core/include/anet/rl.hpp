#pragma once
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <random>
#include <unordered_map>
#include <cstdint>
#include "anet/heat_map.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/tensor_utils.hpp"

namespace anet::rl {

    /*
        Environment
          ↓   (state)
        Runer(Agent)
          ↓  （action）
        Environment
          ↓   (reward, next_state) → EnvResponse  → Experience
        Trainer
          ↓   Experience ⇒ ReplayBuffer
        Sampler(Agent)
          ↓   ReplayBuffer ⇒ Update
        Learner(Agent)
    */

    // =============================================================
    // RunMode
    // =============================================================

    enum class RunMode { Train, Eval1, Eval2 };
    inline bool IsTrain(RunMode mode) { return mode == RunMode::Train; }
    inline bool IsEval(RunMode mode) { return mode == RunMode::Eval1 || mode == RunMode::Eval2; }

    /**
     * @brief 環境が返すステップ応答。
     */
    struct EnvResponse {
        torch::Tensor next_state;      // (N, state_dim)
        float reward;
        bool done;
        bool truncated;
    };

    /**
     * @brief エージェントの学習に使う「1回の経験」。
     */
    struct Experience {
        torch::Tensor state;        // (N, state_dim)
        torch::Tensor action;       // (N, action_dim)
        EnvResponse response;
    };

    struct StateSpaceInfo {
        torch::Tensor shape;
        torch::Tensor low;
        torch::Tensor high;
    };

    // =============================================================
    // Environment 抽象クラス
    // =============================================================

    class Environment {
    public:
        virtual StateSpaceInfo GetStateSpaceInfo() const = 0;

        virtual torch::Tensor Reset(RunMode mode = RunMode::Train) = 0;
        virtual EnvResponse DoStep(const torch::Tensor& action, RunMode mode = RunMode::Train) = 0;
        virtual torch::Tensor GetState() const = 0;

        virtual ~Environment() = default;
    };

    // =============================================================
    // BatchData（Learner 抽象用の汎用バッチ）
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

    struct ExperienceBatch {
        torch::Tensor states;       // (B, state_dim...)
        torch::Tensor actions;      // (B, action_dim...) or (B,)
        torch::Tensor next_states;  // (B, state_dim...)
        torch::Tensor rewards;      // (B,)
        torch::Tensor dones;        // (B,)
        torch::Tensor truncateds;   // (B,)
    };

    class ReplayBuffer {
    public:
        explicit ReplayBuffer(size_t capacity = 10000, bool is_discrete = true)
            : capacity_(capacity), is_discrete_(is_discrete) {}

        void Push(const Experience& e);
        std::vector<Experience> Sample(size_t n) const;
        ExperienceBatch SampleBatch(size_t n, torch::Device device) const;

        size_t Size() const { return size_; }
    private:
        void PushSingle_(const Experience& e);
    private:
        size_t capacity_;
        size_t size_ = 0;
        size_t write_index_ = 0;
        bool is_discrete_;
        bool initialized_ = false;

        torch::Tensor states_;
        torch::Tensor actions_;
        torch::Tensor next_states_;
        torch::Tensor rewards_;
        torch::Tensor dones_;
        torch::Tensor truncateds_;

        mutable std::mt19937 engine_{ std::random_device{}() };
    };

    // =============================================================
    // Runner / Sampler
    // =============================================================

    struct ActionResult {
        torch::Tensor action;
        torch::Tensor log_prob;
        torch::Tensor value;
    };

    class UpdateResult {
    public:
        virtual ~UpdateResult() = default;

        virtual void SyncMetrics() = 0;        // 派生クラスが metrics_ を埋めるためのフック
        const std::unordered_map<std::string, float>& GetMetricsMap() const { return metrics_; }
    protected:
        void SetMetric(const std::string& key, float v) { metrics_[key] = v; }
    private:
        std::unordered_map<std::string, float> metrics_;
    };

    class Runner {
    public:
        virtual ActionResult SelectAction(const torch::Tensor& state, RunMode mode = RunMode::Train) = 0;
        virtual ~Runner() = default;
    };

    class Sampler {
    public:
        virtual void ObserveFirst(const EnvResponse& envResponse) = 0;
        virtual void Observe(const Experience& exprience) = 0;
        virtual ~Sampler() = default;
    };

    class Learner {
    public:
        virtual std::shared_ptr<UpdateResult> UpdateStep(const Experience& exprience) = 0;
        virtual std::shared_ptr<UpdateResult> UpdateBatch(const BatchData&) = 0;
        virtual ~Learner() = default;
    };

    class PostUpdateObserver {
    public:
        virtual void OnPostUpdate(const std::shared_ptr<UpdateResult>& result) = 0;
        virtual ~PostUpdateObserver() = default;
    };

    class Agent : public Runner, public Learner, public PostUpdateObserver {
    public:
        virtual ~Agent() = default;
    };

    // =============================================================
    // StepBasedLearner / TrajectoryBasedLearner
    // =============================================================

    // 環境のステップに同期して更新する Agent 基底クラス（DQN, DDPG, SAC, A2C など）
    class StepBasedLearner : public Learner {
    public:
        StepBasedLearner() = default;
        virtual ~StepBasedLearner() = default;

        // ① pure virtual — 各 Agent が実装
        virtual std::shared_ptr<UpdateResult> UpdateStep(const Experience& expr) override = 0;

        // ② 共通ラッパ（BatchData を Experience 単位で回す）
        virtual std::shared_ptr<UpdateResult> UpdateBatch(const BatchData& batch) override {
            std::shared_ptr<UpdateResult> last;
            for (auto& e : batch.Data()) {
                last = UpdateStep(e);
            }
            return last;
        }

        virtual void OnPostUpdate(const std::shared_ptr<UpdateResult>& /*result*/) {
            // 必要に応じて派生クラスで利用
        }

        size_t GetStepCount() const { return step_count_; }
    protected:
        void IncrementStep() { step_count_++; }

    private:
        size_t step_count_ = 0;
    };

    // 複数ステップ（軌跡）収集後に更新する Agent 基底クラス（PPO, TRPO など）
    class TrajectoryBasedLearner : public Learner {
    public:
        // TODO: define
        virtual ~TrajectoryBasedLearner() = default;
    };

    // =============================================================
    // MetricsObserver / Notifier
    // =============================================================
    class MetricsObserver : public PostUpdateObserver {
    public:
        explicit MetricsObserver(const StepBasedLearner* learner)
            : learner_(learner) {
        }

        void OnPostUpdate(const std::shared_ptr<UpdateResult>& r) override {
            if (!learner_) return;

            size_t step = learner_->GetStepCount();
            r->SyncMetrics();
            auto map = r->GetMetricsMap();

            for (const auto& [tag, value] : map) {
                MetricsLogger::Instance()->log_scalar(tag, step, value);
            }
        }
    private:
        const StepBasedLearner* learner_;
    };

    class Notifier {
    public:
        void AddObserver(PostUpdateObserver* obs) {
            observers_.push_back(obs);
        }

        void Notify(const std::shared_ptr<UpdateResult>& r) {
            for (PostUpdateObserver* o : observers_) {
                o->OnPostUpdate(r);
            }
        }
    private:
        std::vector<PostUpdateObserver*> observers_;
    };

    // =============================================================
    // HeatMap 関連
    // =============================================================

    std::unique_ptr<HeatMap> MakeStateHeatMapPtr(
        const StateSpaceInfo& info,
        int idx_x,
        int idx_y,
        int width = 256,
        int height = 256,
        size_t max_points = 10000,
        uint32_t flags = HM_Default);

    std::unique_ptr<TimeHeatMap> MakeStateTimeHeatMapPtr(
        const StateSpaceInfo& info,
        int idx_x,
        int width = 256,
        int height = 2560,
        size_t max_points = 0,
        uint32_t flags = HM_Default,
        TimeFrameMode mode = TimeFrameMode::Unlimited);

} // namespace anet::rl
