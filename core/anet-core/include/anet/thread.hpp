// anet/thread.hpp

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <deque>

namespace anet {

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