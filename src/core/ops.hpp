// Core fp32 tensor ops for the forward pass.
//
// Everything is row-major over raw float pointers — no tensor class, no
// broadcasting. Each op is the textbook implementation; the perf features
// (threading F025, SIMD F026) later replace the insides while these tests
// keep the outputs honest.
//
// Convention: `tokens` counts rows (one per sequence position), `dim`/`d_in`/
// `d_out` count columns.
#pragma once

#include <cstdint>

namespace nano::ops {

/// Sets the thread count for the matmul path (F025): 1 = serial, 0 = one per
/// hardware thread (the default). linear()/matmul() split their output rows
/// across a persistent pool when the call is large enough to amortize the
/// wakeup; every element is still computed by exactly one thread with the
/// serial arithmetic, so results are bit-identical at any thread count.
/// Not thread-safe itself — call it from the orchestrating thread only.
void set_num_threads(int n);
int num_threads();

/// Dot product of two contiguous fp32 vectors (NEON-accelerated on arm64,
/// scalar elsewhere). Exposed because attention scores are dot products too.
float dot(const float* a, const float* b, int64_t n);

/// y[t, d_out] = x[t, d_in] @ W^T + bias.
/// W is stored [d_out, d_in] exactly as in the safetensors file (Hugging Face
/// nn.Linear layout), so each output value is a dot product of an x row with
/// a W row — both contiguous in memory.
/// `bias` may be nullptr (most Qwen2 projections have no bias).
void linear(const float* x, const float* w, const float* bias, float* y,
            int64_t tokens, int64_t d_in, int64_t d_out);

/// linear() with an int8 weight matrix (F027): W holds per-row-quantized
/// int8 values ([d_out, d_in], same layout as fp32), `scales` one fp32 scale
/// per output row. Each weight is widened to fp32 in-register and the row's
/// dot product is scaled once at the end:
///   y[t, o] = scales[o] * sum_i x[t, i] * (float)w[o, i]  (+ bias[o]).
/// Activations stay fp32 — the point is reading 4x fewer weight bytes, which
/// is what bounds decode speed, not the multiply-adds.
void linear_q8(const float* x, const int8_t* w, const float* scales,
               const float* bias, float* y, int64_t tokens, int64_t d_in,
               int64_t d_out);

/// c[m, n] = a[m, k] @ b[k, n]. Plain row-major matmul.
void matmul(const float* a, const float* b, float* c, int64_t m, int64_t k, int64_t n);

/// RMSNorm (Qwen2/Llama style): y = x / sqrt(mean(x^2) + eps) * weight,
/// normalizing each row of x independently. The mean accumulates in double so
/// the reduction over `dim` values doesn't drift.
void rmsnorm(const float* x, const float* weight, float* y, int64_t tokens,
             int64_t dim, float eps);

/// In-place softmax over one row of n values (max-subtracted for stability).
void softmax(float* x, int64_t n);

/// In-place SiLU: x = x * sigmoid(x). The activation inside Qwen2's MLP.
void silu(float* x, int64_t n);

/// y = a + b, elementwise (residual connections).
void add(const float* a, const float* b, float* y, int64_t n);

/// y = a * b, elementwise (SwiGLU gate).
void mul(const float* a, const float* b, float* y, int64_t n);

/// Rotary position embedding (RoPE), Qwen2/Llama "half rotation" layout,
/// applied in place to one token's q or k laid out [n_heads, head_dim].
///
/// For each head, dimensions i and i+head_dim/2 form a 2D pair rotated by
/// angle = position / theta^(2i/head_dim). The angles are computed in float32
/// to mirror the HF reference implementation bit-for-bit closely.
void rope(float* x, int64_t n_heads, int64_t head_dim, int64_t position, float theta);

}  // namespace nano::ops
