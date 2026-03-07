// anet/thread.hpp

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <deque>

namespace anet {


    class ThreadBase {
    public:
        explicit ThreadBase(const std::string& name);
        virtual ~ThreadBase();

        // コピー不可
        ThreadBase(const ThreadBase&) = delete;
        ThreadBase& operator=(const ThreadBase&) = delete;

        void Start();
        void Stop();
        void Pause() { paused_.store(true); }
        void Resume() { paused_.store(false); }

        bool IsRunning() const { return running_.load(); }
        bool IsPaused() const { return paused_.load(); }

    protected:
        /// 1ステップ分の処理。trueで継続、falseでスレッド終了
        virtual bool ProcessStep() = 0;

        // スレッド実行前後のフック
        virtual void OnStart() {}
        virtual void OnStop() {}
        virtual void OnException() {}

    private:
        void ThreadMain();
    private:
        std::string name_;
        std::atomic<bool> running_{ false };
        std::atomic<bool> paused_{ false };
        std::thread worker_;
    };


    class ThreadPool {
    public:
        using TaskFunction = std::function<void()>;

        virtual void Enqueue(int worker_id, TaskFunction fn) = 0;
        virtual void WaitAll() = 0;
        virtual int GetWorkerCount() const = 0;
        virtual void Stop() = 0;

        virtual ~ThreadPool() = default;
    };

    class PinnedThreadPool final : public ThreadPool {
    public:
        explicit PinnedThreadPool(int worker_count);
        ~PinnedThreadPool();

        int GetWorkerCount() const override { return worker_count_; }
        void Enqueue(int worker_id, TaskFunction fn) override;
        void WaitAll() override;
        void Stop();
    public:
        // コピー不可
        PinnedThreadPool(const PinnedThreadPool&) = delete;
        PinnedThreadPool(PinnedThreadPool&&) = delete;
        PinnedThreadPool& operator=(const PinnedThreadPool&) = delete;
        PinnedThreadPool& operator=(PinnedThreadPool&&) = delete;
    private:
        int worker_count_;
        std::unique_ptr<std::thread[]> workers_;

        std::unique_ptr<std::mutex[]> mutexes_;                 ///< タスクキューの排他用
        std::unique_ptr<std::condition_variable[]> cvs_;        ///< タスクキューとstop_flagの待ち合わせ用
        std::unique_ptr<std::deque<TaskFunction>[]> queues_;    ///< 実行待ちタスクキュー

        std::atomic<bool> stop_flag_;       ///< 全スレッド終了フラグ（デストラクト用）
        std::atomic<int> pending_tasks_;    ///< PinnedThreadPool全体の実行待ち/実行中タスク数
    private:
        void WorkerLoop(int wid);
    };
}