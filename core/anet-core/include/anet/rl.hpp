#pragma once
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <random>
#include <map>
#include <unordered_map>
#include <cstdint>
#include "anet/heat_map.hpp"
#include "anet/metrics_logger.hpp"
#include "anet/tensor_utils.hpp"
#include "anet/random.hpp"

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
    struct ActionResult {
        torch::Tensor next_state; // (N, state_dim)
        torch::Tensor reward;     // (N,) 
        torch::Tensor done;       // (N,)
        torch::Tensor truncated;  // (N,)

        bool IsDone() const {
            return done.item<bool>();
        }
    };

    /**
     * @brief エージェントの学習に使う「1回の経験」。
     */
    struct Experience {
        torch::Tensor state;        // (N, state_dim)
        torch::Tensor action;       // (N, action_dim)
        ActionResult result;
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
        virtual ActionResult DoStep(const torch::Tensor& action, RunMode mode = RunMode::Train) = 0;
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
                e.result.next_state = e.result.next_state.to(device);
            }
        }

        const std::vector<Experience>& Data() const { return experiences_; }
        size_t Size() const { return experiences_.size(); }
        void Clear() { experiences_.clear(); }

    private:
        std::vector<Experience> experiences_;
    };

    struct ActionInfo {
        torch::Tensor action;        // (N, action_dim)  or (N,) for discrete
        torch::Tensor is_randomized; // (N,)  bool tensor  (or uint8 for C++)
        //torch::Tensor raw_action;
        //torch::Tensor noise;
    };

    using MetricsMap = std::unordered_map<std::string, float>;

    class UpdateResult {
    public:
        virtual ~UpdateResult() = default;
        virtual MetricsMap GetMetricsMap() const = 0;
    };

    class Runner {
    public:
        virtual ActionInfo MakeAction(const torch::Tensor& state, RunMode mode = RunMode::Train) = 0;
        virtual ~Runner() = default;
    };

    class Sampler {
    public:
        virtual void ObserveFirst(const ActionResult& result) = 0;
        virtual void Observe(const Experience& exprience) = 0;
        virtual ~Sampler() = default;
    };

    class Learner {
    public:
        virtual std::shared_ptr<const UpdateResult> UpdateStep(const Experience& exprience) = 0;
        virtual std::shared_ptr<const UpdateResult> UpdateBatch(const BatchData& batch) = 0;
        virtual ~Learner() = default;
    };

    class PostUpdateObserver {
    public:
        //virtual void OnPostUpdate(const std::shared_ptr<UpdateResult>& result) = 0;
        virtual void OnPostUpdate(
            std::shared_ptr<const UpdateResult> result,
            const Experience& exprience,
            const ActionInfo& action_info,
            size_t step
        ) = 0;
        virtual ~PostUpdateObserver() = default;
    };

    class Agent : public Runner, public Learner {
    public:
        virtual ~Agent() = default;
    };

    // =============================================================
    // StepBasedLearner / TrajectoryBasedLearner
    // =============================================================

    // 環境のステップに同期して更新する Agent 基底クラス（DQN, DDPG, SAC, A2C など）
    template<typename ConfigT>
    class StepBasedAgent : public Agent {
    public:
        StepBasedAgent(ConfigT config, torch::Device device) : config_(config), device_(device) { }
        virtual ~StepBasedAgent() = default;

        // ① pure virtual — 各 Agent が実装
        virtual std::shared_ptr<const UpdateResult> UpdateStep(const Experience& expr) override = 0;

        // ② 共通ラッパ（BatchData を Experience 単位で回す）
        virtual std::shared_ptr<const UpdateResult> UpdateBatch(const BatchData& batch) override {
            std::shared_ptr<const UpdateResult> last;
            for (auto& e : batch.Data()) {
                last = UpdateStep(e);
            }
            return last;
        }

        size_t GetStepCount() const { return step_count_; }
    protected:
        // Resource（Agentが管理すべき領域）
        ConfigT config_;
        anet::RandomGenerator* rnd = &anet::RandomGenerator::Default();
        torch::Device device_;
    protected:
        // InternalState
        size_t step_count_ = 0;
    };

    // 複数ステップ（軌跡）収集後に更新する Agent 基底クラス（PPO, TRPO など）
    class TrajectoryBasedLearner : public Learner {
    public:
        // TODO: define
        virtual ~TrajectoryBasedLearner() = default;
    };

    class RunnerFactory {
    public:
        virtual std::shared_ptr<Runner> CreateRunner() = 0;
        virtual ~RunnerFactory() = default;
    };

    class Notifier {
    public:
        void AddObserver(PostUpdateObserver* obs) {
            observers_.push_back(obs);
        }

        void Notify(
            const std::shared_ptr<const UpdateResult>& result,
            const Experience& exprience, const ActionInfo& action_info,size_t step)
        {
            for (PostUpdateObserver* o : observers_) {
                o->OnPostUpdate(result, exprience, action_info, step);
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
