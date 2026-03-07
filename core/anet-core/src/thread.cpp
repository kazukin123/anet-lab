#include "anet/thread.hpp"
#include "anet/common.hpp"
#include "anet/profile.hpp"
#include "anet/log.hpp"

using namespace anet;
namespace LOG = anet::log;

//----------------------------------------------
// ThreadBase
//----------------------------------------------

ThreadBase::ThreadBase(const std::string& name)
    : name_(name)
{
}

ThreadBase::~ThreadBase()
{
    Stop();
}

void ThreadBase::Start()
{
    if (running_.load()) return;
    running_.store(true);
    paused_.store(false);

    worker_ = std::thread([this]() { ThreadMain(); });
}

void ThreadBase::Stop()
{
    if (running_.load()) {
        running_.store(false);
    }

    if (worker_.joinable()) {
        worker_.join();
    }
}

void ThreadBase::ThreadMain()
{
    anet::ProfileThreadName th(name_.c_str());
    ANET_LOG_DEBUG("BEGIN name=" << name_);

    try {
        OnStart();

        while (running_.load()) {
            if (paused_.load()) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                continue;
            }

            if (!ProcessStep()) {
                break;
            }
        }
    } catch (...) {
        LOG::error() << "Thrad [" << name_ << "]: Exception caught.";
        OnException();
    }

    OnStop();
    running_.store(false);

    ANET_LOG_DEBUG("END name=" << name_);
}


//----------------------------------------------
// PinnedThreadPool
//----------------------------------------------

PinnedThreadPool::PinnedThreadPool(int worker_count)
    : worker_count_(worker_count)
    , stop_flag_(false)
    , pending_tasks_(0)
{
    queues_ = std::make_unique<std::deque<TaskFunction>[]>(worker_count);
    mutexes_ = std::make_unique<std::mutex[]>(worker_count);
    cvs_ = std::make_unique<std::condition_variable[]>(worker_count);
    workers_ = std::make_unique<std::thread[]>(worker_count);

    for (int i = 0; i < worker_count_; ++i) {
        workers_[i] = std::thread([this, i] { WorkerLoop(i); });
    }
}

PinnedThreadPool::~PinnedThreadPool()
{
    Stop();
}

void PinnedThreadPool::Enqueue(int worker_id, TaskFunction fn)
{
    ANET_ASSERT(worker_id >= 0 && worker_id < worker_count_);

    // queueにtaskを積む
    {
        std::lock_guard<std::mutex> lock(mutexes_[worker_id]);
        queues_[worker_id].push_back(std::move(fn));
    }

    // 積んだので待ちタスク数を増やす
    pending_tasks_.fetch_add(1, std::memory_order_relaxed);
    cvs_[worker_id].notify_one();
}

void PinnedThreadPool::WaitAll()
{
    // 処理待ちタスクが捌けるまで待つ
    while (pending_tasks_.load(std::memory_order_acquire) != 0) {
        std::this_thread::yield();
    }

    // busy wait を避けるため軽い sleep を入れてもよいが、
    // Env のタスクは比較的重いためスピンで十分安定。
}

void PinnedThreadPool::WorkerLoop(int wid)
{
    ProfileThreadName thr_name("PinnedThreadPool::Worker", wid);

    ANET_LOG_DEBUG("BEGIN");

    // 各スレッドのメインループ
    while (true) {
        TaskFunction task;
        {
            // worker(wid)のタスクキューをロック
            std::unique_lock<std::mutex> lock(mutexes_[wid]);

            // stop_flagがtrueになる、またはqueues_[wid]が非空になるまでwaitに入る
            cvs_[wid].wait(lock, [&] {
                // 偽起床の場合はsleep継続
                return stop_flag_.load(std::memory_order_acquire)
                    || !queues_[wid].empty();
                });

            // queue が空かどうかを先に処理する
            if (!queues_[wid].empty()) {
                // キューからTaskを取り出す
                task = std::move(queues_[wid].front());
                queues_[wid].pop_front();
            }
            else if (stop_flag_.load(std::memory_order_acquire)) {
                // stop_flagがtrueの場合はループを抜けてスレッドを終了させる
                break;
            }
        }

        if (task) {
            // 取り出したTaskを実行
            task();

            // PinnedThreadPool全体の実行待ち/実行中タスク数をアトミックにカウントダウン
            pending_tasks_.fetch_sub(1, std::memory_order_release);
        }
    }

    ANET_LOG_DEBUG("END");
}

void PinnedThreadPool::Stop()
{
    // stop_flag_をロックフリーで一度だけtrueに変更
    bool expected = false;
    if (!stop_flag_.compare_exchange_strong(expected, true, std::memory_order_release))
        return; // 同時操作で既にstop_flagが立ってたら抜ける

    // 全てのworkerスレッドを起こす
    for (int i = 0; i < worker_count_; ++i)
        cvs_[i].notify_all();

    // 全てのworkerスレッドを終了待ち
    for (int i = 0; i < worker_count_; ++i)
        if (workers_[i].joinable())
            workers_[i].join();
}