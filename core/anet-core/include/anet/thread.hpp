// anet/thread.hpp

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <mutex>
#include <deque>
#include <future>
#include <type_traits>


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
        explicit PinnedThreadPool(int worker_count, const std::string& name = "PinnedThreadPool");
        ~PinnedThreadPool();

        int GetWorkerCount() const override { return worker_count_; }
        void Enqueue(int worker_id, TaskFunction fn) override;
        void WaitAll() override;
        void Stop();
    public:
        /// 任意の戻り値を持つ関数をキューに積み、futureを返す
        template<class F, class... Args>
        auto EnqueueFuture(int worker_id, F&& f, Args&&... args)
            -> std::future<typename std::invoke_result_t<F, Args...>>
        {
            // 関数の戻り値の型を推論
            using return_type = typename std::invoke_result_t<F, Args...>;

            // タスクを std::packaged_task でラップし、shared_ptr で寿命管理する
            auto task = std::make_shared<std::packaged_task<return_type()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...)
            );

            // 呼び出し元に返す future を取得
            std::future<return_type> res = task->get_future();

            // 既存の Enqueue(std::function<void()>) にラップして投げる
            this->Enqueue(worker_id, [task]() {
                (*task)();
                });

            return res;
        }
    public:
        // コピー不可
        PinnedThreadPool(const PinnedThreadPool&) = delete;
        PinnedThreadPool(PinnedThreadPool&&) = delete;
        PinnedThreadPool& operator=(const PinnedThreadPool&) = delete;
        PinnedThreadPool& operator=(PinnedThreadPool&&) = delete;
    private:
        std::string name_;
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