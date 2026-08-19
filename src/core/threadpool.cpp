#include "core/threadpool.hpp"

namespace nano {

ThreadPool::ThreadPool(int threads) {
    if (threads <= 0) {
        threads = static_cast<int>(std::thread::hardware_concurrency());
        if (threads <= 0) {
            threads = 1;  // hardware_concurrency may report 0
        }
    }
    workers_.reserve(static_cast<size_t>(threads - 1));
    for (int i = 0; i < threads - 1; ++i) {
        // Worker i handles chunk i; the caller handles the last chunk.
        workers_.emplace_back([this, i] { worker_loop(i); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    start_.notify_all();
    for (std::thread& w : workers_) {
        w.join();
    }
}

void ThreadPool::run(int64_t n, TaskFn fn, const void* ctx) {
    const int total = threads();
    if (n <= 0) {
        return;
    }
    if (total == 1 || n == 1) {
        fn(ctx, 0, n);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        n_ = n;
        fn_ = fn;
        ctx_ = ctx;
        pending_ = static_cast<int>(workers_.size());
        ++generation_;
    }
    start_.notify_all();

    // The caller works too — its chunk is the last one.
    const auto [begin, end] = chunk(n, total - 1, total);
    fn(ctx, begin, end);

    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return pending_ == 0; });
}

void ThreadPool::worker_loop(int index) {
    uint64_t seen = 0;
    while (true) {
        TaskFn fn = nullptr;
        const void* ctx = nullptr;
        int64_t n = 0;
        int total = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            start_.wait(lock, [&] { return stopping_ || generation_ != seen; });
            if (stopping_) {
                return;
            }
            seen = generation_;
            fn = fn_;
            ctx = ctx_;
            n = n_;
            total = threads();
        }
        const auto [begin, end] = chunk(n, index, total);
        if (begin < end) {
            fn(ctx, begin, end);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --pending_;
        }
        done_.notify_one();
    }
}

}  // namespace nano
