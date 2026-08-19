#include "core/ops.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

#include "core/threadpool.hpp"

// NEON on arm64 (every Apple silicon Mac); everything else takes the scalar
// path. NANO_FORCE_SCALAR builds the fallback even on arm64, so this machine
// can run the scalar path through the same golden tests that CI's x86 runner
// will (targets test_ops_scalar / bench_ops_scalar).
#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && !defined(NANO_FORCE_SCALAR)
#define NANO_HAS_NEON 1
#include <arm_neon.h>
#else
#define NANO_HAS_NEON 0
#endif

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

/// Dot product of two contiguous fp32 vectors — the inner loop of every
/// projection, and where essentially all decode time goes.
///
/// The NEON version keeps four independent accumulators (16 floats per
/// iteration): NEON has no horizontal-add-into-scalar dependency chain that
/// way, so the four fmla's issue back to back instead of serializing on one
/// accumulator's latency. Summing the four at the end changes the reduction
/// order versus the scalar loop, so results differ by a few ulp — well
/// inside the 1e-5 tolerance the goldens hold, and test_ops compares the two
/// paths directly.
float dot_f32(const float* a, const float* b, int64_t n) {
#if NANO_HAS_NEON
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= n; i += 4) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < n; ++i) {  // tail: up to 3 elements
        sum += a[i] * b[i];
    }
    return sum;
#else
    float acc = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
#endif
}

/// The serial kernel, over output columns [o_begin, o_end). Both the serial
/// and the threaded path funnel through this one loop, so they cannot drift.
void linear_range(const float* x, const float* w, const float* bias, float* y,
                  int64_t tokens, int64_t d_in, int64_t d_out, int64_t o_begin,
                  int64_t o_end) {
    for (int64_t t = 0; t < tokens; ++t) {
        const float* row = x + t * d_in;
        for (int64_t o = o_begin; o < o_end; ++o) {
            const float acc = dot_f32(row, w + o * d_in, d_in);
            y[t * d_out + o] = bias ? acc + bias[o] : acc;
        }
    }
}

}  // namespace

float dot(const float* a, const float* b, int64_t n) { return dot_f32(a, b, n); }

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
        // The sum of squares stays in double (as before SIMD): over 896
        // values fp32 accumulation drifts enough to matter, and this
        // reduction is a rounding error's walk away from the golden.
        double sum_sq = 0.0;
        int64_t i = 0;
#if NANO_HAS_NEON
        float64x2_t acc0 = vdupq_n_f64(0.0);
        float64x2_t acc1 = vdupq_n_f64(0.0);
        for (; i + 4 <= dim; i += 4) {
            const float32x4_t v = vld1q_f32(row + i);
            const float64x2_t lo = vcvt_f64_f32(vget_low_f32(v));
            const float64x2_t hi = vcvt_high_f64_f32(v);
            acc0 = vfmaq_f64(acc0, lo, lo);
            acc1 = vfmaq_f64(acc1, hi, hi);
        }
        sum_sq = vaddvq_f64(vaddq_f64(acc0, acc1));
#endif
        for (; i < dim; ++i) {
            sum_sq += static_cast<double>(row[i]) * static_cast<double>(row[i]);
        }

        const float inv_rms =
            1.0f / std::sqrt(static_cast<float>(sum_sq / static_cast<double>(dim)) + eps);
        float* out = y + t * dim;
        i = 0;
#if NANO_HAS_NEON
        const float32x4_t scale = vdupq_n_f32(inv_rms);
        for (; i + 4 <= dim; i += 4) {
            vst1q_f32(out + i,
                      vmulq_f32(vmulq_f32(vld1q_f32(row + i), scale),
                                vld1q_f32(weight + i)));
        }
#endif
        for (; i < dim; ++i) {
            out[i] = row[i] * inv_rms * weight[i];
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
