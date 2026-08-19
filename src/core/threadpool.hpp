// Persistent thread pool for data-parallel loops.
//
// Why persistent: a decode step runs ~170 linear() calls; spawning threads
// per call would cost more than the math. Workers are created once and park
// on a condition variable between calls.
//
// Why not std::async/std::function: the decode hot loop must not allocate
// (F024). A task here is a plain function pointer plus a context pointer to
// the caller's stack — nothing is copied or heap-allocated per call.
//
// parallel_for(n, body) splits [0, n) into one contiguous chunk per thread
// (workers + the calling thread, which does its own share instead of just
// waiting). Each index is processed by exactly one thread with the same
// arithmetic as the serial loop, so results are bit-identical to
// single-threaded execution regardless of thread count.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace nano {

class ThreadPool {
public:
    /// Creates `threads - 1` workers (the caller is the remaining thread).
    /// threads <= 0 picks std::thread::hardware_concurrency().
    explicit ThreadPool(int threads = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int threads() const { return static_cast<int>(workers_.size()) + 1; }

    /// Runs body(begin, end) on disjoint chunks covering [0, n), blocking
    /// until every chunk is done. `body` must be safe to call concurrently
    /// on disjoint ranges. Allocation-free after construction.
    template <typename Body>
    void parallel_for(int64_t n, const Body& body) {
        // Type-erase through a function pointer instead of std::function so
        // nothing is heap-allocated; `body` stays on the caller's stack.
        const auto trampoline = [](const void* ctx, int64_t begin, int64_t end) {
            (*static_cast<const Body*>(ctx))(begin, end);
        };
        run(n, trampoline, &body);
    }

private:
    using TaskFn = void (*)(const void* ctx, int64_t begin, int64_t end);

    void run(int64_t n, TaskFn fn, const void* ctx);
    void worker_loop(int index);

    /// Chunk `index` of `total` over [0, n): balanced to within one element.
    static std::pair<int64_t, int64_t> chunk(int64_t n, int index, int total) {
        const int64_t begin = n * index / total;
        const int64_t end = n * (index + 1) / total;
        return {begin, end};
    }

    std::vector<std::thread> workers_;

    // One outstanding task at a time, guarded by mutex_. generation_ bumps
    // once per run(); workers wake when it changes, so a slow waker can never
    // run the same task twice or miss one.
    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable done_;
    uint64_t generation_ = 0;
    int64_t n_ = 0;
    TaskFn fn_ = nullptr;
    const void* ctx_ = nullptr;
    int pending_ = 0;  // workers still running the current task
    bool stopping_ = false;
};

}  // namespace nano
