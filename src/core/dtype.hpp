// Tensor element types as they appear in safetensors files, plus the scalar
// conversions the engine needs. Qwen2.5 checkpoints ship bf16; the fp32
// pipeline converts on load.
#pragma once

#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace nano {

enum class DType : uint8_t {
    F32,
    F16,
    BF16,
    I64,
    I32,
    I8,
    U8,
};

inline size_t dtype_size(DType t) {
    switch (t) {
        case DType::F32:
            return 4;
        case DType::F16:
            return 2;
        case DType::BF16:
            return 2;
        case DType::I64:
            return 8;
        case DType::I32:
            return 4;
        case DType::I8:
        case DType::U8:
            return 1;
    }
    throw std::runtime_error("dtype_size: unknown dtype");
}

inline std::string_view dtype_name(DType t) {
    switch (t) {
        case DType::F32:
            return "F32";
        case DType::F16:
            return "F16";
        case DType::BF16:
            return "BF16";
        case DType::I64:
            return "I64";
        case DType::I32:
            return "I32";
        case DType::I8:
            return "I8";
        case DType::U8:
            return "U8";
    }
    throw std::runtime_error("dtype_name: unknown dtype");
}

inline DType dtype_from_string(std::string_view s) {
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "I64") return DType::I64;
    if (s == "I32") return DType::I32;
    if (s == "I8") return DType::I8;
    if (s == "U8") return DType::U8;
    throw std::runtime_error("unsupported dtype: " + std::string(s));
}

/// bf16 is just the top 16 bits of an fp32 — same exponent range, truncated
/// mantissa — so widening is exact: shift back into place.
inline float bf16_to_f32(uint16_t bits) {
    return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

/// IEEE fp16 -> fp32, handling subnormals and inf/nan explicitly.
inline float f16_to_f32(uint16_t bits) {
    uint32_t sign = static_cast<uint32_t>(bits >> 15) & 1u;
    uint32_t exp = static_cast<uint32_t>(bits >> 10) & 0x1Fu;
    uint32_t frac = static_cast<uint32_t>(bits) & 0x3FFu;

    uint32_t out;
    if (exp == 0) {
        if (frac == 0) {
            out = sign << 31;  // +/- zero
        } else {
            // Subnormal half: normalize into fp32.
            int shift = 0;
            while ((frac & 0x400u) == 0) {
                frac <<= 1;
                ++shift;
            }
            frac &= 0x3FFu;
            uint32_t out_exp = static_cast<uint32_t>(127 - 15 - shift + 1);
            out = (sign << 31) | (out_exp << 23) | (frac << 13);
        }
    } else if (exp == 0x1F) {
        out = (sign << 31) | 0x7F800000u | (frac << 13);  // inf / nan
    } else {
        out = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    return std::bit_cast<float>(out);
}

}  // namespace nano
