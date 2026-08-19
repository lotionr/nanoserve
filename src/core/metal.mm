// Metal backend implementation (F033). See metal.hpp for the scope and the
// CUDA-to-Metal dictionary; this file has the mechanics.
//
// Design choices, and why:
//  - The MSL kernel source is embedded as a string and compiled at first
//    use (newLibraryWithSource, ~50 ms once). The alternative — compiling
//    .metal -> .metallib at build time — is faster to load but adds build
//    machinery and a file the binary must locate at runtime. llama.cpp
//    shipped the embedded-source route for years for the same reason.
//  - One kernel shape for GEMV and GEMM: a simdgroup (= warp, 32 lanes)
//    owns one output column o and a tile of up to 8 tokens. Each lane
//    strides the weight row (lane, lane+32, ...), so consecutive lanes
//    read consecutive floats — coalesced, in CUDA terms. The token tile is
//    what makes prefill viable: the weight row is read ONCE per tile
//    instead of once per token, cutting weight traffic 8x for a 41-token
//    prefill. Decode (tokens == 1) is simply tile size 1.
//  - Every call is synchronous (commit + waitUntilCompleted). The caller
//    consumes y on the CPU immediately (RoPE, softmax, silu...), so there
//    is nothing to overlap with. This is the honest cost of offloading
//    ONLY the matmuls: one full submit/wait round-trip per projection —
//    measured, not hidden (see the bench discussion in the README).
#include "core/metal.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace nano::metal {

namespace {

/// Mirrored 1:1 by the MSL `Params` struct below — plain uint32s so the
/// C++ and MSL layouts cannot disagree about padding.
struct Params {
    uint32_t tokens = 0;
    uint32_t d_in = 0;
    uint32_t d_out = 0;
    uint32_t has_bias = 0;
};

/// Threadgroup geometry, shared by host and kernels. 4 simdgroups per
/// threadgroup (128 threads — a common sweet spot; one lone simdgroup per
/// threadgroup underfills the GPU's threadgroup slots) and 8 tokens per
/// tile (8 fp32 accumulators per lane is cheap in registers, and 8x weight
/// reuse already moves prefill from weight-bound toward compute).
constexpr uint32_t kRowsPerThreadgroup = 4;
constexpr uint32_t kTokenTile = 8;

/// The kernels. MSL is C++14-ish; the [[...]] attributes bind arguments to
/// buffer indices and expose the thread's coordinates (blockIdx/laneid).
constexpr const char* kKernelSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint tokens;
    uint d_in;
    uint d_out;
    uint has_bias;
};

constant uint kRowsPerThreadgroup = 4;   // simdgroups (warps) per threadgroup
constant uint kTokenTile = 8;            // tokens each simdgroup accumulates

// y[t, o] = x[t, :] . w[o, :] (+ bias[o])
//
// Grid: threadgroup (tg.x, tg.y) covers output columns
// [tg.x * 4, tg.x * 4 + 3] and tokens [tg.y * 8, tg.y * 8 + 7]. Within the
// threadgroup, simdgroup `sg` owns column o, and its 32 lanes stride the
// d_in-long dot product; simd_sum() is the warp tree-reduction
// (__reduce_add_sync in CUDA terms).
kernel void linear_f32(device const float* x      [[buffer(0)]],
                       device const float* w      [[buffer(1)]],
                       device const float* bias   [[buffer(2)]],
                       device float*       y      [[buffer(3)]],
                       constant Params&    p      [[buffer(4)]],
                       uint2 tg   [[threadgroup_position_in_grid]],
                       uint  lane [[thread_index_in_simdgroup]],
                       uint  sg   [[simdgroup_index_in_threadgroup]]) {
    const uint o = tg.x * kRowsPerThreadgroup + sg;
    if (o >= p.d_out) {
        return;   // ragged last threadgroup when d_out % 4 != 0
    }
    const uint t0 = tg.y * kTokenTile;
    const uint n_t = min(kTokenTile, p.tokens - t0);

    // One pass over the weight row feeds ALL n_t tokens: w is read once per
    // 8-token tile (the whole point — weights are the dominant traffic).
    float acc[kTokenTile] = {0.0f};
    device const float* wrow = w + ulong(o) * p.d_in;
    for (uint i = lane; i < p.d_in; i += 32) {
        const float wv = wrow[i];
        for (uint t = 0; t < n_t; ++t) {
            acc[t] = fma(wv, x[ulong(t0 + t) * p.d_in + i], acc[t]);
        }
    }
    for (uint t = 0; t < n_t; ++t) {
        const float sum = simd_sum(acc[t]);   // reduce 32 lane-partials
        if (lane == 0) {
            y[ulong(t0 + t) * p.d_out + o] = p.has_bias != 0 ? sum + bias[o] : sum;
        }
    }
}

// int8-weight variant: y[t, o] = scales[o] * (x[t, :] . float(w[o, :]))
// (+ bias[o]). Weights are dequantized in-register — see metal.hpp for why
// activations deliberately stay fp32 here (unlike the CPU sdot kernel).
// char == int8_t; the float() conversion is exact for [-127, 127].
kernel void linear_q8(device const float* x      [[buffer(0)]],
                      device const char*  w      [[buffer(1)]],
                      device const float* scales [[buffer(2)]],
                      device const float* bias   [[buffer(3)]],
                      device float*       y      [[buffer(4)]],
                      constant Params&    p      [[buffer(5)]],
                      uint2 tg   [[threadgroup_position_in_grid]],
                      uint  lane [[thread_index_in_simdgroup]],
                      uint  sg   [[simdgroup_index_in_threadgroup]]) {
    const uint o = tg.x * kRowsPerThreadgroup + sg;
    if (o >= p.d_out) {
        return;
    }
    const uint t0 = tg.y * kTokenTile;
    const uint n_t = min(kTokenTile, p.tokens - t0);

    float acc[kTokenTile] = {0.0f};
    device const char* wrow = w + ulong(o) * p.d_in;
    for (uint i = lane; i < p.d_in; i += 32) {
        const float wv = float(wrow[i]);
        for (uint t = 0; t < n_t; ++t) {
            acc[t] = fma(wv, x[ulong(t0 + t) * p.d_in + i], acc[t]);
        }
    }
    for (uint t = 0; t < n_t; ++t) {
        // The per-row scale multiplies the WHOLE dot product once, after
        // the reduction — same factoring the CPU kernel uses.
        const float sum = scales[o] * simd_sum(acc[t]);
        if (lane == 0) {
            y[ulong(t0 + t) * p.d_out + o] = p.has_bias != 0 ? sum + bias[o] : sum;
        }
    }
}
)MSL";

/// Everything the backend owns. Created once, lazily; never destroyed
/// (process-lifetime, like the CPU thread pool).
struct Ctx {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pso_f32 = nil;
    id<MTLComputePipelineState> pso_q8 = nil;
    std::string device_name;

    // Registered long-lived weights, keyed by their CPU pointer.
    struct Resident {
        id<MTLBuffer> buf = nil;
        size_t bytes = 0;
    };
    std::unordered_map<const void*, Resident> resident;

    // Grow-only staging buffers (the Scratch pattern, F024): activations in,
    // results out, plus fallbacks for unregistered weights/scales/bias. After
    // the first (largest) call, decode steps allocate nothing.
    id<MTLBuffer> x_buf = nil;
    id<MTLBuffer> y_buf = nil;
    id<MTLBuffer> w_stage = nil;
    id<MTLBuffer> s_stage = nil;
    id<MTLBuffer> b_stage = nil;
    id<MTLBuffer> dummy = nil;   // bound in the bias slot when bias == nullptr

    // ops entry points run on the one orchestrating thread (see
    // ops::set_num_threads' contract), but register/unregister can race a
    // test's direct calls in principle — one mutex keeps the registry and
    // staging buffers coherent either way.
    std::mutex mu;
};

/// Wraps `ptr` as an MTLBuffer with zero copies when possible.
///
/// Unified memory makes this the interesting call: newBufferWithBytesNoCopy
/// hands the GPU the SAME physical pages the std::vector owns — no second
/// copy of a 0.5-1.9 GB model. Requirements: page-aligned base and
/// page-multiple length (macOS malloc satisfies both for large allocations,
/// which weight matrices always are — they come from vm_allocate under the
/// hood). Rounding the length up to a page is safe because the underlying
/// vm region is page-granular. Small allocations (biases, scales) fail the
/// alignment check and take the one-time-copy path instead — a few KB each,
/// irrelevant. On a discrete GPU none of this would work; every byte would
/// cross PCIe into device memory.
id<MTLBuffer> wrap_or_copy(id<MTLDevice> device, const void* ptr, size_t bytes) {
    const size_t page = static_cast<size_t>(getpagesize());
    if (reinterpret_cast<uintptr_t>(ptr) % page == 0) {
        const size_t rounded = (bytes + page - 1) / page * page;
        id<MTLBuffer> buf =
            [device newBufferWithBytesNoCopy:const_cast<void*>(ptr)
                                      length:rounded
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
        if (buf != nil) {
            return buf;   // zero-copy
        }
    }
    return [device newBufferWithBytes:ptr
                               length:bytes
                              options:MTLResourceStorageModeShared];
}

/// Grows `buf` to at least `bytes` (never shrinks). Returns the buffer.
/// (`__strong &`: under ARC a plain `id&` parameter would be
/// __autoreleasing and refuse to bind to the Ctx's __strong members.)
id<MTLBuffer> ensure_capacity(id<MTLDevice> device, id<MTLBuffer> __strong& buf,
                              size_t bytes) {
    if (buf == nil || [buf length] < bytes) {
        buf = [device newBufferWithLength:bytes
                                  options:MTLResourceStorageModeShared];
    }
    return buf;
}

/// One-time init. Returns nullptr (after one stderr line) if there is no
/// GPU or the kernels fail to compile — callers then report unavailable and
/// everything stays on the CPU.
Ctx* init_ctx() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        std::fprintf(stderr, "metal: no GPU device — backend unavailable\n");
        return nullptr;
    }

    NSError* error = nil;
    id<MTLLibrary> lib =
        [device newLibraryWithSource:[NSString stringWithUTF8String:kKernelSource]
                             options:nil
                               error:&error];
    if (lib == nil) {
        std::fprintf(stderr, "metal: kernel compile failed: %s\n",
                     error.localizedDescription.UTF8String);
        return nullptr;
    }

    auto make_pso = [&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        if (fn == nil) {
            return nil;
        }
        return [device newComputePipelineStateWithFunction:fn error:&error];
    };
    id<MTLComputePipelineState> pso_f32 = make_pso(@"linear_f32");
    id<MTLComputePipelineState> pso_q8 = make_pso(@"linear_q8");
    if (pso_f32 == nil || pso_q8 == nil) {
        std::fprintf(stderr, "metal: pipeline creation failed: %s\n",
                     error ? error.localizedDescription.UTF8String : "unknown");
        return nullptr;
    }
    // The kernels' lane-strided loops and simd_sum reductions assume a
    // 32-wide simdgroup. True on every Apple GPU; refuse rather than
    // miscompute if a future device disagrees.
    if ([pso_f32 threadExecutionWidth] != 32) {
        std::fprintf(stderr, "metal: simdgroup width %lu != 32 — backend disabled\n",
                     static_cast<unsigned long>([pso_f32 threadExecutionWidth]));
        return nullptr;
    }

    Ctx* ctx = new Ctx();
    ctx->device = device;
    ctx->queue = [device newCommandQueue];
    ctx->pso_f32 = pso_f32;
    ctx->pso_q8 = pso_q8;
    ctx->device_name = [[device name] UTF8String];
    ctx->dummy = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    return ctx;
}

Ctx* ctx() {
    static Ctx* c = init_ctx();   // thread-safe magic static
    return c;
}

/// Shared body of linear_f32/linear_q8: stage inputs, encode one dispatch,
/// wait, copy the result out. `w_bytes_per_value` is 4 (fp32) or 1 (int8).
void dispatch_linear(bool q8, const float* x, const void* w, const float* scales,
                     const float* bias, float* y, int64_t tokens, int64_t d_in,
                     int64_t d_out) {
    Ctx* c = ctx();
    if (c == nullptr) {
        throw std::runtime_error("metal backend unavailable");
    }
    std::lock_guard<std::mutex> lock(c->mu);

    const size_t x_bytes = static_cast<size_t>(tokens * d_in) * sizeof(float);
    const size_t y_bytes = static_cast<size_t>(tokens * d_out) * sizeof(float);
    const size_t w_bytes =
        static_cast<size_t>(d_out * d_in) * (q8 ? sizeof(int8_t) : sizeof(float));

    // Activations in: a memcpy into shared memory, not a bus transfer.
    // Decode staging is ~4 KB; even prefill is ~150 KB — noise next to the
    // weight reads the kernel then does.
    id<MTLBuffer> xb = ensure_capacity(c->device, c->x_buf, x_bytes);
    std::memcpy([xb contents], x, x_bytes);
    id<MTLBuffer> yb = ensure_capacity(c->device, c->y_buf, y_bytes);

    // Weights: registered -> bind the resident buffer (zero work here);
    // unregistered -> stage a copy (correct but slow; tests only).
    const auto bind_or_stage = [&](const void* ptr, size_t bytes,
                                   id<MTLBuffer> __strong& stage) -> id<MTLBuffer> {
        const auto it = c->resident.find(ptr);
        if (it != c->resident.end()) {
            return it->second.buf;
        }
        id<MTLBuffer> buf = ensure_capacity(c->device, stage, bytes);
        std::memcpy([buf contents], ptr, bytes);
        return buf;
    };
    id<MTLBuffer> wb = bind_or_stage(w, w_bytes, c->w_stage);
    id<MTLBuffer> sb = nil;
    if (q8) {
        sb = bind_or_stage(scales, static_cast<size_t>(d_out) * sizeof(float),
                           c->s_stage);
    }
    id<MTLBuffer> bb = c->dummy;   // never read when has_bias == 0
    if (bias != nullptr) {
        bb = bind_or_stage(bias, static_cast<size_t>(d_out) * sizeof(float),
                           c->b_stage);
    }

    Params p;
    p.tokens = static_cast<uint32_t>(tokens);
    p.d_in = static_cast<uint32_t>(d_in);
    p.d_out = static_cast<uint32_t>(d_out);
    p.has_bias = bias != nullptr ? 1 : 0;

    id<MTLCommandBuffer> cmd = [c->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:(q8 ? c->pso_q8 : c->pso_f32)];
    NSUInteger idx = 0;
    [enc setBuffer:xb offset:0 atIndex:idx++];
    [enc setBuffer:wb offset:0 atIndex:idx++];
    if (q8) {
        [enc setBuffer:sb offset:0 atIndex:idx++];
    }
    [enc setBuffer:bb offset:0 atIndex:idx++];
    [enc setBuffer:yb offset:0 atIndex:idx++];
    // setBytes: tiny constants ride along in the command stream — no
    // MTLBuffer needed (like CUDA kernel parameters).
    [enc setBytes:&p length:sizeof(p) atIndex:idx];

    // Grid: one threadgroup covers 4 output columns x 8 tokens (see the
    // geometry constants). dispatchThreadgroups is <<<grid, block>>>.
    const auto ceil_div = [](int64_t a, int64_t b) {
        return static_cast<NSUInteger>((a + b - 1) / b);
    };
    const MTLSize grid = MTLSizeMake(ceil_div(d_out, kRowsPerThreadgroup),
                                     ceil_div(tokens, kTokenTile), 1);
    const MTLSize block = MTLSizeMake(32 * kRowsPerThreadgroup, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:block];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];   // the synchronous round-trip, by design
    if ([cmd status] != MTLCommandBufferStatusCompleted) {
        NSError* err = [cmd error];
        throw std::runtime_error(
            std::string("metal: command buffer failed: ") +
            (err != nil ? err.localizedDescription.UTF8String : "unknown"));
    }

    std::memcpy(y, [yb contents], y_bytes);
}

}  // namespace

bool available() { return ctx() != nullptr; }

const char* device_name() {
    Ctx* c = ctx();
    return c != nullptr ? c->device_name.c_str() : "";
}

void register_weights(const void* ptr, size_t bytes) {
    Ctx* c = ctx();
    if (c == nullptr || ptr == nullptr || bytes == 0) {
        return;   // no GPU: registering is a harmless no-op
    }
    std::lock_guard<std::mutex> lock(c->mu);
    if (c->resident.count(ptr) != 0) {
        return;
    }
    c->resident[ptr] = {wrap_or_copy(c->device, ptr, bytes), bytes};
}

void unregister_weights(const void* ptr) {
    Ctx* c = ctx();
    if (c == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(c->mu);
    c->resident.erase(ptr);   // ARC releases the MTLBuffer
}

void linear_f32(const float* x, const float* w, const float* bias, float* y,
                int64_t tokens, int64_t d_in, int64_t d_out) {
    @autoreleasepool {
        dispatch_linear(false, x, w, nullptr, bias, y, tokens, d_in, d_out);
    }
}

void linear_q8(const float* x, const int8_t* w, const float* scales,
               const float* bias, float* y, int64_t tokens, int64_t d_in,
               int64_t d_out) {
    @autoreleasepool {
        dispatch_linear(true, x, w, scales, bias, y, tokens, d_in, d_out);
    }
}

}  // namespace nano::metal
