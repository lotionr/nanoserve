// Metal GPU backend for the matmul path (F033).
//
// Scope: ONLY linear projections (fp32 and int8-weight) run on the GPU —
// the rest of the forward pass (norms, RoPE, attention, sampling) stays on
// the CPU. ops::linear / ops::linear_q8 route here when the metal backend
// is selected (ops::set_backend) and the call is big enough to amortize a
// GPU dispatch; everything below that floor keeps taking the CPU kernels.
// The CPU path is therefore always available and always the reference.
//
// For a reader who knows CUDA but not Metal, the dictionary used throughout:
//   MTLDevice          = the GPU (cudaGetDevice)
//   MTLCommandQueue    = a stream (cudaStream_t)
//   MTLCommandBuffer   = one batch of work submitted to the stream
//   MTLComputePipelineState = a compiled kernel (CUfunction)
//   threadgroup        = block;  simdgroup = warp (width 32 on Apple GPUs)
//   waitUntilCompleted = cudaStreamSynchronize
// The big difference from a discrete GPU: Apple silicon memory is UNIFIED.
// There is no PCIe bus and no cudaMemcpy — an MTLBuffer with "shared"
// storage is ordinary process memory that both CPU and GPU read at full
// bandwidth. "Uploading" a weight is either a one-time memcpy into such a
// buffer or, when the source allocation is page-aligned, a zero-copy wrap
// of the very pages the std::vector already owns.
#pragma once

#include <cstddef>
#include <cstdint>

namespace nano::metal {

/// True when a Metal device exists AND the kernels compiled. First call
/// initializes the backend (device + runtime-compiled pipelines, ~50 ms);
/// later calls are a flag read. False on non-Apple builds (see
/// metal_stub.cpp) or if anything failed — the failure reason is printed
/// to stderr once.
bool available();

/// GPU name for logs/bench provenance ("Apple M3 Pro"); "" if unavailable.
const char* device_name();

/// Weight residency. linear_f32/linear_q8 look their weight pointer up in
/// a registry: registered weights bind their long-lived MTLBuffer directly;
/// unregistered pointers are copied into a staging buffer EVERY call (still
/// correct — just slow for big matrices, it's what the parity test uses).
/// The Engine registers all weight matrices/scales/biases at construction
/// (only when the metal backend is already selected) and unregisters them
/// in its destructor, so the registry never holds a dangling pointer.
/// Registration is keyed by pointer: the caller guarantees the memory is
/// immutable and outlives the registration — nothing here re-checks the
/// contents, which is exactly why unregistering on free is mandatory.
void register_weights(const void* ptr, size_t bytes);
void unregister_weights(const void* ptr);

/// ops::linear on the GPU: y[t, o] = x[t, :] . w[o, :] (+ bias[o]).
/// Same [d_out, d_in] row-major weight layout as the CPU kernel. `bias`
/// may be nullptr. Synchronous: returns with y fully written. Float
/// caveat: the GPU reduces each dot product in a different order than
/// NEON/scalar (lane-strided partials + simd_sum tree), so results match
/// the CPU within ~1e-5 relative, not bit-exactly — the same class of gap
/// the NEON and scalar CPU builds already have between each other.
void linear_f32(const float* x, const float* w, const float* bias, float* y,
                int64_t tokens, int64_t d_in, int64_t d_out);

/// ops::linear_q8 on the GPU: int8 weights + one fp32 scale per output row.
/// DELIBERATE semantic difference from the CPU sdot kernel: activations are
/// NOT quantized. Each weight is dequantized in-register and multiplied by
/// the fp32 activation (y = scales[o] * sum_i float(w[o,i]) * x[t,i]).
/// The CPU path quantizes activations because that unlocks the sdot
/// instruction — a CPU-ISA trick that turns 16 multiply-adds into one op.
/// A GPU is not instruction-starved on GEMV, it is bandwidth-bound, and the
/// bandwidth win of int8 is in the WEIGHT bytes (read every call), not the
/// activation bytes (tiny). Skipping activation quantization is therefore
/// free speed-wise and strictly MORE accurate: this kernel computes exactly
/// what the pre-sdot F027 kernel computed, which matched every fp32 greedy
/// golden. test_metal holds it to the same bar.
void linear_q8(const float* x, const int8_t* w, const float* scales,
               const float* bias, float* y, int64_t tokens, int64_t d_in,
               int64_t d_out);

}  // namespace nano::metal
