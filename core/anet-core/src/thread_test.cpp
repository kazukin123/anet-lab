#include "anet/catch_test.hpp"

#include "anet/thread.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

TEST_CASE("ThreadPool ParallelFor executes every work index exactly once", "[thread][parallel_for]")
{
    anet::PinnedThreadPool pool(3, "parallel-for-test");
    std::vector<std::atomic<int>> counts(17);

    pool.ParallelFor(counts.size(), [&](size_t index) {
        counts[index].fetch_add(1, std::memory_order_relaxed);
    });

    for (const auto& count : counts) {
        CHECK(count.load(std::memory_order_relaxed) == 1);
    }
}

TEST_CASE("ThreadPool ParallelFor rethrows a worker failure on the caller", "[thread][parallel_for]")
{
    anet::PinnedThreadPool pool(3, "parallel-for-failure-test");

    CHECK_THROWS_WITH(
        pool.ParallelFor(17, [](size_t index) {
            if (index == 0) throw std::runtime_error("parallel-for-error");
        }),
        "parallel-for-error");
}

TEST_CASE("ThreadPool ParallelFor handles empty and undersized work ranges", "[thread][parallel_for]")
{
    anet::PinnedThreadPool pool(5, "parallel-for-boundary-test");
    std::atomic<int> calls = 0;

    pool.ParallelFor(0, [&](size_t) {
        calls.fetch_add(1, std::memory_order_relaxed);
    });
    CHECK(calls.load(std::memory_order_relaxed) == 0);

    pool.ParallelFor(2, [&](size_t) {
        calls.fetch_add(1, std::memory_order_relaxed);
    });
    CHECK(calls.load(std::memory_order_relaxed) == 2);
}

TEST_CASE("ThreadPool ParallelFor keeps work off the caller thread", "[thread][parallel_for]")
{
    anet::PinnedThreadPool pool(3, "parallel-for-caller-test");
    const auto caller_thread = std::this_thread::get_id();
    std::mutex mutex;
    std::vector<std::thread::id> work_threads;

    pool.ParallelFor(11, [&](size_t) {
        std::lock_guard lock(mutex);
        work_threads.push_back(std::this_thread::get_id());
    });

    REQUIRE(work_threads.size() == 11);
    for (const auto work_thread : work_threads) {
        CHECK(work_thread != caller_thread);
    }
}

TEST_CASE("ThreadPool ParallelFor stops claiming work after a failure", "[thread][parallel_for]")
{
    anet::PinnedThreadPool pool(3, "parallel-for-cancel-test");
    std::atomic<int> calls = 0;

    CHECK_THROWS_AS(
        pool.ParallelFor(100, [&](size_t index) {
            calls.fetch_add(1, std::memory_order_relaxed);
            if (index == 0) throw std::runtime_error("parallel-for-cancel");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }),
        std::runtime_error);

    CHECK(calls.load(std::memory_order_relaxed) < 100);
}
