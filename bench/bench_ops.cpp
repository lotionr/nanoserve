// Microbenchmark for the linear() kernel (F025 threading, later F026 SIMD).
//
// Times the real decode-step GEMV shapes (Qwen2.5-0.5B) single-threaded vs
// pooled, on synthetic data resident in RAM. This is NOT the benchmark
// harness (F030) and its numbers never go in the README — they are recorded
// in claude-progress.txt to justify each optimization.
//
// Usage: ./build/bench_ops [threads]   (default: hardware concurrency)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "core/ops.hpp"

namespace {

struct Shape {
    const char* name;
    int64_t tokens, d_in, d_out;
};

// Decode shapes (tokens = 1) are GEMV: each weight is read once, so they are
// memory-bandwidth bound. Prefill shapes (tokens = 41, a real chat prompt)
// reuse each weight across all rows, making them compute bound — the two
// regimes scale very differently with threads, which is the point of showing
// both.
constexpr Shape kShapes[] = {
    {"decode  lm_head  [151936]", 1, 896, 151936},
    {"decode  mlp gate [4864]  ", 1, 896, 4864},
    {"decode  q_proj   [896]   ", 1, 896, 896},
    {"prefill mlp gate [4864]  ", 41, 896, 4864},
    {"prefill q_proj   [896]   ", 41, 896, 896},
};

/// Median wall time of `reps` calls, in milliseconds.
double time_linear(const float* x, const float* w, float* y, const Shape& s, int reps) {
    std::vector<double> ms;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        nano::ops::linear(x, w, nullptr, y, s.tokens, s.d_in, s.d_out);
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 0;

    std::mt19937 gen(20260818);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::printf("%-28s %12s %12s %9s %10s\n", "shape", "1 thread", "pooled", "speedup",
                "GFLOP/s");
    for (const Shape& s : kShapes) {
        std::vector<float> x(static_cast<size_t>(s.tokens * s.d_in));
        std::vector<float> w(static_cast<size_t>(s.d_out * s.d_in));
        std::vector<float> y(static_cast<size_t>(s.tokens * s.d_out));
        for (float& v : x) {
            v = dist(gen);
        }
        for (float& v : w) {
            v = dist(gen);
        }

        constexpr int kReps = 20;
        nano::ops::set_num_threads(1);
        time_linear(x.data(), w.data(), y.data(), s, 3);  // warmup
        const double serial_ms = time_linear(x.data(), w.data(), y.data(), s, kReps);

        nano::ops::set_num_threads(threads);
        time_linear(x.data(), w.data(), y.data(), s, 3);  // warmup (spawns pool)
        const double pooled_ms = time_linear(x.data(), w.data(), y.data(), s, kReps);
        const int nthreads = nano::ops::num_threads();

        // One multiply-add per (token, d_in, d_out) triple = 2 flops.
        const double gflop =
            2.0 * static_cast<double>(s.tokens * s.d_in * s.d_out) / 1e9;
        std::printf("%-28s %9.3f ms %9.3f ms %8.2fx %10.1f\n", s.name, serial_ms,
                    pooled_ms, serial_ms / pooled_ms, gflop / (pooled_ms / 1e3));
        (void)nthreads;
    }
    std::printf("pooled = %d threads (median of 20 reps)\n", nano::ops::num_threads());
    return 0;
}
