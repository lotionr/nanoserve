// int8 weight quantization (F027).
//
// Scheme: symmetric per-row absmax. For each row r of a [rows, cols] weight
// matrix, scale[r] = max(|w|) / 127 and q = round(w / scale), clamped to
// [-127, 127]. Dequantization is q * scale, so the worst-case error per
// value is scale/2 — about 0.4% of the row's largest weight.
//
// Why per-row: linear() computes each output value as a dot product of an
// activation row with one weight ROW, so a per-row scale factors cleanly out
// of the whole dot product — the kernel does int8->f32 math and multiplies
// by the scale once at the end. (llama.cpp's Q8_0 is the same idea at finer
// granularity: one scale per 32-value block instead of per row.)
//
// The same scheme quantizes ACTIVATIONS at runtime (inside ops::linear_q8),
// so the kernel can do its dot products in int8 x int8 -> int32 integer
// SIMD (sdot). Activations get one scale per 32-VALUE BLOCK, not per row:
// activation rows carry outliers (a handful of values 10-100x the typical
// magnitude), and one row-wide absmax scale rounds everything else so
// coarsely that greedy decoding visibly diverged — measured, not guessed
// (see the session log; the FIRST greedy token flipped on 1 of the 4 golden
// prompts). Per-32-block is exactly the granularity llama.cpp's Q8_0 uses
// for activations; with it, top-1 holds on every logits golden and every
// greedy token whose fp32 top-2 margin exceeds 0.12 is preserved — the one
// remaining flip sits at a 0.03-margin near-tie (details in test_quant and
// the session log). Weights keep per-row scales (they are smooth, and it
// keeps the file format unchanged).
// quantize_row_q8 below is the single quantizer both paths share — a
// "block" is just a 32-value row.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/dtype.hpp"

namespace nano {

/// One quantized matrix: int8 values in the same [rows, cols] row-major
/// layout as the fp32 original, plus one fp32 scale per row.
struct QuantMatrix {
    int64_t rows = 0;
    int64_t cols = 0;
    std::vector<int8_t> q;       // [rows * cols]
    std::vector<float> scales;   // [rows]
};

/// Quantizes one row of n fp32 values into `out`, symmetric absmax, and
/// returns the scale (absmax / 127; 0 for an all-zero row, with all-zero
/// values so it dequantizes exactly). This is THE quantizer: quantize_rows
/// calls it per weight row offline, quantize_row_q8_blocks calls it per
/// 32-value activation block at runtime.
float quantize_row_q8(const float* src, int8_t* out, int64_t n);

/// Activation-block granularity for the runtime side of linear_q8.
/// 32 matches llama.cpp's Q8_0 and divides every width in this model
/// (hidden 896, kv 128, intermediate 4864) — but a trailing partial block
/// is still handled, with its own scale over the remainder.
constexpr int64_t kQ8Block = 32;

/// Quantizes one row of n values in blocks of kQ8Block: block b covers
/// values [b*32, min((b+1)*32, n)) and writes its scale to scales[b].
/// scales must hold ceil(n / 32) floats.
void quantize_row_q8_blocks(const float* src, int8_t* out, float* scales, int64_t n);

/// Quantizes a row-major [rows, cols] fp32 matrix, per-row absmax symmetric
/// (quantize_row_q8 applied to each row).
QuantMatrix quantize_rows(const float* w, int64_t rows, int64_t cols);

/// Dequantized value at (row, col) — the reference for error checks.
inline float dequant_at(const QuantMatrix& m, int64_t row, int64_t col) {
    return static_cast<float>(m.q[static_cast<size_t>(row * m.cols + col)]) *
           m.scales[static_cast<size_t>(row)];
}

/// One tensor to be written to a safetensors file. `data` must stay alive
/// until write_safetensors returns.
struct SaveTensor {
    std::string name;
    DType dtype = DType::F32;
    std::vector<int64_t> shape;
    const void* data = nullptr;
    size_t nbytes = 0;
};

/// Writes a safetensors file (8-byte header length, JSON header, raw data).
/// Tensor names must not need JSON escaping (ours are plain ASCII paths).
/// `metadata` becomes the __metadata__ entry (string -> string).
void write_safetensors(const std::string& path, const std::vector<SaveTensor>& tensors,
                       const std::vector<std::pair<std::string, std::string>>& metadata);

}  // namespace nano
