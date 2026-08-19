#include "core/ops.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

#include "core/threadpool.hpp"

namespace nano::ops {

namespace {

int g_requested_threads = 0;  // 0 = hardware concurrency
std::unique_ptr<ThreadPool> g_pool;

/// The process-wide pool, created on first use (so short CLI commands like
/// `tokenize` never spawn threads).
ThreadPool& pool() {
    if (!g_pool) {
        g_pool = std::make_unique<ThreadPool>(g_requested_threads);
    }
    return *g_pool;
}

/// Below this many multiply-adds a call runs serial: waking the pool costs
/// a few microseconds, which only pays for itself on the bigger projections
/// (decode-step q/o/MLP/lm_head), not on tiny per-token k/v rows.
constexpr int64_t kParallelThreshold = 1 << 19;

/// The serial kernel, over output columns [o_begin, o_end). Both the serial
/// and the threaded path funnel through this one loop, so they cannot drift.
void linear_range(const float* x, const float* w, const float* bias, float* y,
                  int64_t tokens, int64_t d_in, int64_t d_out, int64_t o_begin,
                  int64_t o_end) {
    for (int64_t t = 0; t < tokens; ++t) {
        const float* row = x + t * d_in;
        for (int64_t o = o_begin; o < o_end; ++o) {
            const float* wrow = w + o * d_in;
            float acc = 0.0f;
            for (int64_t i = 0; i < d_in; ++i) {
                acc += row[i] * wrow[i];
            }
            y[t * d_out + o] = bias ? acc + bias[o] : acc;
        }
    }
}

}  // namespace

void set_num_threads(int n) {
    g_requested_threads = n;
    g_pool.reset();  // rebuilt at the new size on next use
}

int num_threads() { return pool().threads(); }

void linear(const float* x, const float* w, const float* bias, float* y,
            int64_t tokens, int64_t d_in, int64_t d_out) {
    if (tokens * d_in * d_out < kParallelThreshold) {
        linear_range(x, w, bias, y, tokens, d_in, d_out, 0, d_out);
        return;
    }
    // Each thread computes a disjoint slice of output columns; every y value
    // is produced by exactly one thread running the identical serial loop,
    // so the result is bit-exact equal to the single-threaded one.
    pool().parallel_for(d_out, [&](int64_t o_begin, int64_t o_end) {
        linear_range(x, w, bias, y, tokens, d_in, d_out, o_begin, o_end);
    });
}

void matmul(const float* a, const float* b, float* c, int64_t m, int64_t k, int64_t n) {
    // Row i of the output is independent of every other row; parallel_for
    // hands each thread a contiguous block of rows (serial when small).
    const auto rows = [&](int64_t i_begin, int64_t i_end) {
        for (int64_t i = i_begin; i < i_end; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (int64_t p = 0; p < k; ++p) {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
    };
    if (m * k * n < kParallelThreshold) {
        rows(0, m);
        return;
    }
    pool().parallel_for(m, rows);
}

void rmsnorm(const float* x, const float* weight, float* y, int64_t tokens,
             int64_t dim, float eps) {
    for (int64_t t = 0; t < tokens; ++t) {
        const float* row = x + t * dim;
        double sum_sq = 0.0;
        for (int64_t i = 0; i < dim; ++i) {
            sum_sq += static_cast<double>(row[i]) * static_cast<double>(row[i]);
        }
        const float inv_rms =
            1.0f / std::sqrt(static_cast<float>(sum_sq / static_cast<double>(dim)) + eps);
        for (int64_t i = 0; i < dim; ++i) {
            y[t * dim + i] = row[i] * inv_rms * weight[i];
        }
    }
}

void softmax(float* x, int64_t n) {
    float max_val = x[0];
    for (int64_t i = 1; i < n; ++i) {
        max_val = std::max(max_val, x[i]);
    }
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += static_cast<double>(x[i]);
    }
    const float inv_sum = static_cast<float>(1.0 / sum);
    for (int64_t i = 0; i < n; ++i) {
        x[i] *= inv_sum;
    }
}

void silu(float* x, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

void add(const float* a, const float* b, float* y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = a[i] + b[i];
    }
}

void mul(const float* a, const float* b, float* y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = a[i] * b[i];
    }
}

void rope(float* x, int64_t n_heads, int64_t head_dim, int64_t position, float theta) {
    const int64_t half = head_dim / 2;
    for (int64_t i = 0; i < half; ++i) {
        // Mirror HF's float32 arithmetic: inv_freq = 1 / theta^(2i/dim),
        // angle = position * inv_freq, all in float32, so the golden test can
        // hold a tight tolerance.
        const float exponent =
            static_cast<float>(2 * i) / static_cast<float>(head_dim);
        const float inv_freq = 1.0f / std::pow(theta, exponent);
        const float angle = static_cast<float>(position) * inv_freq;
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        for (int64_t h = 0; h < n_heads; ++h) {
            float* head = x + h * head_dim;
            const float lo = head[i];         // pairs with dimension i + half
            const float hi = head[i + half];
            head[i] = lo * c - hi * s;
            head[i + half] = hi * c + lo * s;
        }
    }
}

}  // namespace nano::ops
