#include "core/ops.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "core/metal.hpp"
#include "core/quant.hpp"
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

// The int8 dot-product instruction (sdot). Every Apple silicon chip has it
// (ARMv8.2 dotprod extension, baseline since the M1); older arm64 or a
// NANO_FORCE_SCALAR build falls back to the plain integer loop below —
// which computes the SAME exact int32 sum, just slower.
#if NANO_HAS_NEON && defined(__ARM_FEATURE_DOTPROD)
#define NANO_HAS_SDOT 1
#else
#define NANO_HAS_SDOT 0
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

Backend g_backend = Backend::cpu;

/// Below this many multiply-adds a call stays on the CPU even when the
/// metal backend is selected — the GPU analog of kParallelThreshold, for a
/// much bigger fixed cost: one encode + commit + waitUntilCompleted
/// round-trip (tens of microseconds at best) instead of a pool wakeup.
/// Same value as kParallelThreshold: it keeps the tiny decode-step k/v
/// projections (1 x 896 x 128 ≈ 115K MACs) on the CPU while everything
/// from a decode q_proj (1 x 896 x 896 ≈ 800K) up goes to the GPU. This is
/// a floor for correctness of the COST model, not a tuning knob — the
/// bench measures the backend as configured here.
constexpr int64_t kMetalMinMacs = 1 << 19;

/// True when this call should take the GPU path.
bool use_metal(int64_t macs) {
    return g_backend == Backend::metal && macs >= kMetalMinMacs;
}

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
///
/// Loop order: weight row OUTER, token INNER (F036). A weight row is read
/// from memory once and dotted against every token row while it is hot; the
/// token-outer order streamed the whole weight matrix once PER TOKEN, which
/// costs nothing at tokens=1 (decode) but throws away the entire point of
/// batching. Measured on the M3 Pro (test_batch): a 4-sequence batched
/// decode step went from 58.2 ms to 31.3 ms on this swap alone — 1.47x to
/// 2.85x aggregate throughput over single-sequence decode. Each y value is
/// still the identical dot product, so outputs are bit-identical in either
/// order (test_ops asserts GEMM-row vs GEMV parity).
void linear_range(const float* x, const float* w, const float* bias, float* y,
                  int64_t tokens, int64_t d_in, int64_t d_out, int64_t o_begin,
                  int64_t o_end) {
    for (int64_t o = o_begin; o < o_end; ++o) {
        const float* w_row = w + o * d_in;
        for (int64_t t = 0; t < tokens; ++t) {
            const float acc = dot_f32(x + t * d_in, w_row, d_in);
            y[t * d_out + o] = bias ? acc + bias[o] : acc;
        }
    }
}

/// Integer-core dot product for linear_q8: activations quantized int8 per
/// 32-value block (`a` + one fp32 scale per block in `a_scales`), weights
/// int8 (`w`, whose per-ROW scale the caller applies afterwards). Returns
///     sum_b  a_scales[b] * ( sum_{i in block b} a[i] * w[i] )
/// i.e. each block's dot product is computed EXACTLY in int32 and only the
/// per-block results meet floating point.
///
/// Why integer: one sdot instruction (vdotq_s32) multiplies 16 int8 pairs
/// and adds them into 4 int32 lanes. The previous widening kernel needed
/// ~11 instructions for the same 16 weights (load + 2x vmovl to int16 + 4x
/// (vmovl to int32 + vcvt to fp32) + 4x fmla), so decode was still
/// instruction-throughput limited instead of memory-bandwidth limited.
/// Per 32-weight block this path is 2 sdot + 1 cvt + 1 fma — the same
/// structure as llama.cpp's Q8_0 vec_dot.
///
/// Why per-block activation scales (not one per row): activation rows carry
/// outliers, and a row-wide absmax scale rounded typical values coarsely
/// enough to flip greedy tokens on the golden prompts (measured — see the
/// session log). Blocks confine an outlier's damage to its own 32 values.
///
/// No overflow: |q| <= 127 so each product is <= 16129, and a block sums 32
/// of them -> |block sum| <= 516k, far inside int32. (sdot widens each
/// 4-product group straight into an int32 lane; no int16 stage to saturate.)
///
/// Determinism: within a block the int32 sum is exact in any order, so the
/// NEON and scalar builds agree bit-for-bit per block; the fp32 summation
/// ACROSS blocks differs in order between the two builds (4 SIMD lanes vs
/// sequential), so cross-build comparisons hold to ~1e-5 like the other fp
/// kernels — but within one build, results are bit-identical at any thread
/// count (each output value is computed whole by exactly one thread).
///
/// Four independent accumulators (4 blocks = 128 int8 per iteration), same
/// shape and reason as dot_f32: independent fma chains instead of one
/// serialized accumulator.
float dot_q8_blocks(const int8_t* a, const float* a_scales, const int8_t* w,
                    int64_t n) {
    const int64_t full_blocks = n / kQ8Block;  // kQ8Block == 32
#if NANO_HAS_SDOT
    // One block's exact integer dot: two 16-wide sdot into the same int32x4.
    const auto block_dot = [](const int8_t* pa, const int8_t* pw) -> int32x4_t {
        const int32x4_t d0 =
            vdotq_s32(vdupq_n_s32(0), vld1q_s8(pa), vld1q_s8(pw));
        return vdotq_s32(d0, vld1q_s8(pa + 16), vld1q_s8(pw + 16));
    };
    float32x4_t facc0 = vdupq_n_f32(0.0f);
    float32x4_t facc1 = vdupq_n_f32(0.0f);
    float32x4_t facc2 = vdupq_n_f32(0.0f);
    float32x4_t facc3 = vdupq_n_f32(0.0f);
    int64_t b = 0;
    for (; b + 4 <= full_blocks; b += 4) {
        const int64_t i = b * kQ8Block;
        // facc lanes hold partial sums; scaling the whole int32x4 by the
        // block's one scale is valid because every lane belongs to the block.
        facc0 = vfmaq_n_f32(facc0, vcvtq_f32_s32(block_dot(a + i, w + i)),
                            a_scales[b]);
        facc1 = vfmaq_n_f32(facc1, vcvtq_f32_s32(block_dot(a + i + 32, w + i + 32)),
                            a_scales[b + 1]);
        facc2 = vfmaq_n_f32(facc2, vcvtq_f32_s32(block_dot(a + i + 64, w + i + 64)),
                            a_scales[b + 2]);
        facc3 = vfmaq_n_f32(facc3, vcvtq_f32_s32(block_dot(a + i + 96, w + i + 96)),
                            a_scales[b + 3]);
    }
    for (; b < full_blocks; ++b) {
        facc0 = vfmaq_n_f32(facc0, vcvtq_f32_s32(block_dot(a + b * kQ8Block,
                                                           w + b * kQ8Block)),
                            a_scales[b]);
    }
    float sum =
        vaddvq_f32(vaddq_f32(vaddq_f32(facc0, facc1), vaddq_f32(facc2, facc3)));
#else
    float sum = 0.0f;
    for (int64_t b = 0; b < full_blocks; ++b) {
        int32_t isum = 0;
        for (int64_t i = b * kQ8Block; i < (b + 1) * kQ8Block; ++i) {
            isum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(w[i]);
        }
        sum += a_scales[b] * static_cast<float>(isum);
    }
#endif
    if (const int64_t tail = n % kQ8Block; tail != 0) {  // partial last block
        int32_t isum = 0;
        for (int64_t i = full_blocks * kQ8Block; i < n; ++i) {
            isum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(w[i]);
        }
        sum += a_scales[full_blocks] * static_cast<float>(isum);
    }
    return sum;
}

/// Quantized-activation scratch for linear_q8, reused across calls. Grow-only
/// (the Engine::Scratch pattern, F024): the prefill call is the largest, so
/// every later 1-token decode step finds it already big enough and the decode
/// hot loop stays allocation-free. ops entry points run on the single
/// orchestrating thread (see set_num_threads), so one buffer suffices — the
/// pool workers only READ it inside linear_q8's parallel_for.
std::vector<int8_t> g_act_q;      // [tokens * d_in] int8 activations
std::vector<float> g_act_scales;  // [tokens * blocks_per_row] block scales

/// linear_q8's serial kernel over output columns [o_begin, o_end) — the q8
/// twin of linear_range, and funnel for both its serial and threaded paths.
/// `xq`/`x_scales` are the pre-quantized activation rows (g_act_*);
/// `x_blocks` is the number of 32-value blocks per activation row.
///
/// The math, spelled out (the "accuracy model" for the int8 path):
///   weight row:  w[o,i] ~= w_scale[o] * qw[o,i],    |err| <= w_scale[o]/2
///   activations: x[t,i] ~= x_scale[t,b] * qa[t,i],  |err| <= x_scale[t,b]/2
///                (b = i/32: one scale per 32-value block)
///   y[t,o] = w_scale[o] * sum_b x_scale[t,b] * (sum_{i in b} qw[o,i]*qa[t,i])
///            (+ bias[o])
/// Each block's int32 sum is EXACT; the only error sources are the two
/// quantization roundings, the per-block int->fp32 conversion + scale fma,
/// and the final w_scale multiply. Whether that budget is acceptable is not
/// argued, it is measured: test_quant checks top-1 logits and greedy
/// continuations against the fp32 goldens.
void linear_q8_range(const int8_t* xq, const float* x_scales, int64_t x_blocks,
                     const int8_t* w, const float* scales, const float* bias,
                     float* y, int64_t tokens, int64_t d_in, int64_t d_out,
                     int64_t o_begin, int64_t o_end) {
    // Weight row outer, token inner — same reuse argument as linear_range.
    for (int64_t o = o_begin; o < o_end; ++o) {
        const int8_t* w_row = w + o * d_in;
        for (int64_t t = 0; t < tokens; ++t) {
            const float acc =
                scales[o] * dot_q8_blocks(xq + t * d_in, x_scales + t * x_blocks,
                                          w_row, d_in);
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

bool set_backend(Backend b) {
    if (b == Backend::metal && !metal::available()) {
        return false;
    }
    g_backend = b;
    return true;
}

Backend backend() { return g_backend; }

void linear(const float* x, const float* w, const float* bias, float* y,
            int64_t tokens, int64_t d_in, int64_t d_out) {
    if (use_metal(tokens * d_in * d_out)) {
        metal::linear_f32(x, w, bias, y, tokens, d_in, d_out);
        return;
    }
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

void linear_q8(const float* x, const int8_t* w, const float* scales,
               const float* bias, float* y, int64_t tokens, int64_t d_in,
               int64_t d_out) {
    // GPU path first: it works from the fp32 activations directly (no
    // activation quantization — see metal.hpp for why), so it must branch
    // BEFORE the CPU path's quantize step below.
    if (use_metal(tokens * d_in * d_out)) {
        metal::linear_q8(x, w, scales, bias, y, tokens, d_in, d_out);
        return;
    }
    // Quantize each activation row once, up front and serial: O(tokens*d_in)
    // work that unlocks O(tokens*d_in*d_out) integer dots. In decode
    // (tokens=1) this is one absmax pass + one rounding pass over d_in
    // values — well under 1% of the projection it feeds.
    const int64_t x_blocks = (d_in + kQ8Block - 1) / kQ8Block;
    const size_t need = static_cast<size_t>(tokens * d_in);
    if (g_act_q.size() < need) {
        g_act_q.resize(need);
    }
    if (g_act_scales.size() < static_cast<size_t>(tokens * x_blocks)) {
        g_act_scales.resize(static_cast<size_t>(tokens * x_blocks));
    }
    for (int64_t t = 0; t < tokens; ++t) {
        quantize_row_q8_blocks(x + t * d_in, g_act_q.data() + t * d_in,
                               g_act_scales.data() + t * x_blocks, d_in);
    }

    if (tokens * d_in * d_out < kParallelThreshold) {
        linear_q8_range(g_act_q.data(), g_act_scales.data(), x_blocks, w, scales,
                        bias, y, tokens, d_in, d_out, 0, d_out);
        return;
    }
    // Same split as linear(): disjoint output columns, one thread per slice,
    // identical serial arithmetic — bit-exact at any thread count.
    pool().parallel_for(d_out, [&](int64_t o_begin, int64_t o_end) {
        linear_q8_range(g_act_q.data(), g_act_scales.data(), x_blocks, w, scales,
                        bias, y, tokens, d_in, d_out, o_begin, o_end);
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
