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
// Weights only; activations stay fp32. The win this targets is memory
// bandwidth — decode reads every weight once per token, and int8 reads 4x
// fewer bytes than fp32 — not integer arithmetic throughput.
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

/// Quantizes a row-major [rows, cols] fp32 matrix, per-row absmax symmetric.
/// An all-zero row gets scale 0 and all-zero values (dequantizes exactly).
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
