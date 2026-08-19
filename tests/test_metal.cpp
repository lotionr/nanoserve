// F033: Metal GPU backend for the matmul path.
//
// Four layers of checks (skips entirely — exit 77 — without a Metal device,
// e.g. on the x86 CI runner):
//   1. Kernel parity: metal::linear_f32 against the CPU ops::linear on a
//      shape sweep (GEMV, GEMM, ragged tails, sub-simdgroup widths) — every
//      element within 1e-3 (combined abs/rel), worst error printed. These
//      direct calls use the backend's staging path (no registration), so
//      they also prove correctness for unregistered weights.
//   2. int8 kernel parity: metal::linear_q8 against the exact reference it
//      claims to compute — the CPU fp32 linear over DEQUANTIZED weights
//      (the GPU deliberately does not quantize activations; see metal.hpp).
//   3. Backend routing: ops::set_backend(metal) sends big calls to the GPU
//      and keeps sub-floor calls on the (bit-identical) CPU path.
//   4. The real model (needs downloaded weights): with the metal backend,
//      fp32 greedy continuations must be IDENTICAL to the HF goldens and
//      top-1 logits must match on every logits golden; the CPU-vs-GPU
//      logit divergence is measured and printed. The int8+metal engine is
//      held to test_quant's bar: identical continuations, or a divergence
//      only at a genuine near-tie (fp32 top-2 margin < 0.1).
#include "core/metal.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "core/ops.hpp"
#include "core/quant.hpp"
#include "model/qwen2.hpp"
#include "testing.hpp"

namespace {

std::vector<int32_t> ints(const nano::json::Value& v) {
    std::vector<int32_t> out;
    for (const auto& x : v.items()) {
        out.push_back(static_cast<int32_t>(x.as_int()));
    }
    return out;
}

/// max over elements of |got - want| / max(1, |want|) — the combined
/// abs/rel error the 1e-3 acceptance bound applies to.
float worst_error(const std::vector<float>& got, const std::vector<float>& want) {
    float worst = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        const float denom = std::max(1.0f, std::fabs(want[i]));
        worst = std::max(worst, std::fabs(got[i] - want[i]) / denom);
    }
    return worst;
}

void fill(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : v) {
        x = dist(rng);
    }
}

void test_linear_f32_parity() {
    std::mt19937 rng(33);
    // {tokens, d_in, d_out, bias}: decode GEMVs (q/o, k/v, MLP shapes), a
    // prefill GEMM, transposed-ish GEMM, ragged sizes that don't divide the
    // 4-row/8-token threadgroup tile, and d_in < 32 (idle simd lanes).
    struct Case {
        int64_t tokens, d_in, d_out;
        bool bias;
    };
    const Case cases[] = {
        {1, 896, 896, true},     // decode q_proj (with bias, like Qwen2)
        {1, 896, 128, true},     // decode k/v_proj shape
        {1, 896, 4864, false},   // decode gate/up
        {1, 4864, 896, false},   // decode down
        {1, 896, 8192, false},   // a slice of the lm_head GEMV
        {41, 896, 4864, false},  // prefill GEMM (exercises the token tile)
        {41, 4864, 896, false},  // prefill down-proj GEMM
        {9, 100, 130, true},     // nothing divides the tile sizes
        {3, 33, 7, true},        // tiny + ragged
        {2, 31, 5, false},       // d_in < simdgroup width
    };
    float worst = 0.0f;
    for (const Case& c : cases) {
        std::vector<float> x(static_cast<size_t>(c.tokens * c.d_in));
        std::vector<float> w(static_cast<size_t>(c.d_out * c.d_in));
        std::vector<float> bias(static_cast<size_t>(c.d_out));
        fill(x, rng);
        fill(w, rng);
        fill(bias, rng);
        const float* b = c.bias ? bias.data() : nullptr;

        std::vector<float> cpu(static_cast<size_t>(c.tokens * c.d_out));
        std::vector<float> gpu(cpu.size());
        nano::ops::linear(x.data(), w.data(), b, cpu.data(), c.tokens, c.d_in,
                          c.d_out);
        nano::metal::linear_f32(x.data(), w.data(), b, gpu.data(), c.tokens,
                                c.d_in, c.d_out);

        const float err = worst_error(gpu, cpu);
        worst = std::max(worst, err);
        NANO_CHECK_MSG(err <= 1e-3f,
                       "linear_f32 %lldx%lld->%lld: worst err %g > 1e-3",
                       static_cast<long long>(c.tokens),
                       static_cast<long long>(c.d_in),
                       static_cast<long long>(c.d_out), static_cast<double>(err));
    }
    std::printf("linear_f32 GPU vs CPU: worst combined abs/rel error %.3g "
                "(bound 1e-3)\n",
                static_cast<double>(worst));
}

void test_linear_q8_parity() {
    std::mt19937 rng(44);
    struct Case {
        int64_t tokens, d_in, d_out;
        bool bias;
    };
    const Case cases[] = {
        {1, 896, 896, true},     // decode projection
        {1, 896, 4864, false},   // decode MLP
        {41, 896, 4864, false},  // prefill GEMM
        {5, 67, 11, true},       // ragged
    };
    float worst = 0.0f;
    for (const Case& c : cases) {
        std::vector<float> x(static_cast<size_t>(c.tokens * c.d_in));
        std::vector<float> w(static_cast<size_t>(c.d_out * c.d_in));
        std::vector<float> bias(static_cast<size_t>(c.d_out));
        fill(x, rng);
        fill(w, rng);
        fill(bias, rng);
        const float* b = c.bias ? bias.data() : nullptr;
        const nano::QuantMatrix q = nano::quantize_rows(w.data(), c.d_out, c.d_in);

        // The exact reference for the GPU's documented math: fp32 linear
        // over the dequantized weights. (The CPU linear_q8 is NOT the
        // reference here — it additionally quantizes activations.)
        std::vector<float> dequant(w.size());
        for (int64_t r = 0; r < c.d_out; ++r) {
            for (int64_t col = 0; col < c.d_in; ++col) {
                dequant[static_cast<size_t>(r * c.d_in + col)] =
                    nano::dequant_at(q, r, col);
            }
        }
        std::vector<float> ref(static_cast<size_t>(c.tokens * c.d_out));
        std::vector<float> gpu(ref.size());
        nano::ops::linear(x.data(), dequant.data(), b, ref.data(), c.tokens,
                          c.d_in, c.d_out);
        nano::metal::linear_q8(x.data(), q.q.data(), q.scales.data(), b,
                               gpu.data(), c.tokens, c.d_in, c.d_out);

        const float err = worst_error(gpu, ref);
        worst = std::max(worst, err);
        NANO_CHECK_MSG(err <= 1e-3f,
                       "linear_q8 %lldx%lld->%lld: worst err %g > 1e-3",
                       static_cast<long long>(c.tokens),
                       static_cast<long long>(c.d_in),
                       static_cast<long long>(c.d_out), static_cast<double>(err));
    }
    std::printf("linear_q8 GPU vs dequantized-fp32 reference: worst error %.3g "
                "(bound 1e-3)\n",
                static_cast<double>(worst));
}

void test_backend_routing() {
    NANO_CHECK(nano::ops::backend() == nano::ops::Backend::cpu);
    NANO_CHECK(nano::ops::set_backend(nano::ops::Backend::metal));
    NANO_CHECK(nano::ops::backend() == nano::ops::Backend::metal);

    std::mt19937 rng(55);

    // Above the floor (1 * 896 * 896 MACs): ops::linear must produce the
    // GPU kernel's output. Same kernel, same inputs — GPU dispatch is
    // deterministic, so the two results are byte-identical.
    {
        const int64_t d = 896;
        std::vector<float> x(static_cast<size_t>(d)), w(static_cast<size_t>(d * d));
        fill(x, rng);
        fill(w, rng);
        std::vector<float> routed(static_cast<size_t>(d)), direct(static_cast<size_t>(d));
        nano::ops::linear(x.data(), w.data(), nullptr, routed.data(), 1, d, d);
        nano::metal::linear_f32(x.data(), w.data(), nullptr, direct.data(), 1, d, d);
        NANO_CHECK_MSG(std::memcmp(routed.data(), direct.data(),
                                   routed.size() * sizeof(float)) == 0,
                       "big call did not route to the GPU kernel");
    }

    // Below the floor (1 * 64 * 64 MACs): must stay on the CPU path —
    // bit-identical to the result with the cpu backend selected.
    {
        const int64_t d = 64;
        std::vector<float> x(static_cast<size_t>(d)), w(static_cast<size_t>(d * d));
        fill(x, rng);
        fill(w, rng);
        std::vector<float> with_metal(static_cast<size_t>(d)),
            with_cpu(static_cast<size_t>(d));
        nano::ops::linear(x.data(), w.data(), nullptr, with_metal.data(), 1, d, d);
        NANO_CHECK(nano::ops::set_backend(nano::ops::Backend::cpu));
        nano::ops::linear(x.data(), w.data(), nullptr, with_cpu.data(), 1, d, d);
        NANO_CHECK_MSG(std::memcmp(with_metal.data(), with_cpu.data(),
                                   with_cpu.size() * sizeof(float)) == 0,
                       "sub-floor call did not stay on the CPU path");
    }
}

/// Finds the top-2 margin of a logits vector (for the near-tie frame).
float top2_margin(std::span<const float> logits) {
    float v1 = logits[0], v2 = -1e30f;
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > v1) {
            v2 = v1;
            v1 = logits[i];
        } else if (logits[i] > v2) {
            v2 = logits[i];
        }
    }
    return v1 - v2;
}

void test_model_goldens() {
    const std::string dir = "models/qwen2.5-0.5b-instruct";
    if (!nano::testing::file_exists(dir + "/config.json")) {
        std::printf("model not downloaded — kernel parity checked, golden "
                    "checks skipped\n");
        return;
    }
    const std::vector<int32_t> stop_ids = {151645, 151643};  // <|im_end|>, <|endoftext|>

    // Select metal BEFORE construction so the Engine registers its weights
    // (zero-copy on unified memory) — the intended production sequence.
    NANO_CHECK(nano::ops::set_backend(nano::ops::Backend::metal));
    nano::Engine engine(dir, 2048);

    // Top-1 must match every HF logits golden, and the GPU-vs-CPU logit
    // divergence is measured by running the SAME engine's forward under
    // both backends (routing is global; registration only affects speed).
    const nano::json::Value lg =
        nano::json::parse(nano::json::read_file("tests/data/logits_golden.json"));
    for (const auto& c : lg.at("cases").items()) {
        const std::vector<int32_t> ids = ints(c.at("ids"));
        const int32_t want = static_cast<int32_t>(c.at("argmax").as_int());

        engine.reset();
        const std::span<const float> logits = engine.forward(ids);
        const std::vector<float> gpu_logits(logits.begin(), logits.end());
        NANO_CHECK_MSG(nano::argmax(logits) == want,
                       "metal top-1 %d != golden %d (prompt: %s)",
                       nano::argmax(logits), want,
                       c.at("prompt").as_string().c_str());

        NANO_CHECK(nano::ops::set_backend(nano::ops::Backend::cpu));
        engine.reset();
        const std::span<const float> cpu_logits = engine.forward(ids);
        float max_diff = 0.0f;
        for (size_t i = 0; i < cpu_logits.size(); ++i) {
            max_diff = std::max(max_diff, std::fabs(gpu_logits[i] - cpu_logits[i]));
        }
        std::printf("fp32 metal vs cpu logits (%s): max |delta| = %.3g\n",
                    c.at("prompt").as_string().c_str(),
                    static_cast<double>(max_diff));
        NANO_CHECK(nano::ops::set_backend(nano::ops::Backend::metal));
    }

    // fp32 + metal greedy continuations: IDENTICAL to the HF goldens — the
    // same bar the CPU fp32 engine passes in test_generate. fp32 GPU noise
    // is ~1e-5 logits (printed above), two orders under the tightest greedy
    // margin in these goldens (0.007), so exactness is the honest demand.
    const nano::json::Value gg =
        nano::json::parse(nano::json::read_file("tests/data/generate_golden.json"));
    for (const auto& c : gg.at("cases").items()) {
        const std::vector<int32_t> prompt_ids = ints(c.at("prompt_ids"));
        const std::vector<int32_t> want = ints(c.at("generated_ids"));
        engine.reset();
        const std::vector<int32_t> got =
            nano::greedy_generate(engine, prompt_ids, 32, stop_ids);
        NANO_CHECK_MSG(got == want, "fp32 metal greedy diverges (prompt: %s)",
                       c.at("prompt").as_string().c_str());
    }
    std::printf("fp32 metal greedy: identical to HF goldens on all %zu prompts\n",
                gg.at("cases").items().size());

    // int8 + metal: held to test_quant's frame. The GPU int8 kernel is
    // MORE faithful than the CPU sdot kernel (no activation quantization —
    // it computes what the original F027 kernel computed, which matched all
    // goldens), so expect identity; tolerate a flip only at a genuine
    // near-tie, measured by teacher-forcing the fp32 engine on the CPU.
    const std::string int8_file = dir + "/model.int8.safetensors";
    if (!nano::testing::file_exists(int8_file)) {
        std::printf("int8 weights missing (run test_quant or `nanoserve "
                    "quantize`) — int8 metal golden checks skipped\n");
        nano::ops::set_backend(nano::ops::Backend::cpu);
        return;
    }
    constexpr float kTieMargin = 0.1f;
    nano::Engine int8_engine(dir, 2048, int8_file);
    for (const auto& c : gg.at("cases").items()) {
        const std::vector<int32_t> prompt_ids = ints(c.at("prompt_ids"));
        const std::vector<int32_t> want = ints(c.at("generated_ids"));
        int8_engine.reset();
        const std::vector<int32_t> got =
            nano::greedy_generate(int8_engine, prompt_ids, 32, stop_ids);

        size_t match = 0;
        while (match < got.size() && match < want.size() && got[match] == want[match]) {
            ++match;
        }
        if (match == got.size() && match == want.size()) {
            std::printf("int8 metal greedy vs fp32 golden (%s): IDENTICAL "
                        "(%zu/%zu)\n",
                        c.at("prompt").as_string().c_str(), match, want.size());
            continue;
        }
        // Diverged: how contested was that step, per the fp32 reference?
        nano::ops::set_backend(nano::ops::Backend::cpu);
        engine.reset();
        std::vector<int32_t> prefix = prompt_ids;
        prefix.insert(prefix.end(), want.begin(),
                      want.begin() + static_cast<int64_t>(match));
        const float margin = top2_margin(engine.forward(prefix));
        nano::ops::set_backend(nano::ops::Backend::metal);
        std::printf("int8 metal greedy vs fp32 golden (%s): diverges at step "
                    "%zu/%zu, fp32 top-2 margin %.4f\n",
                    c.at("prompt").as_string().c_str(), match,
                    std::max(want.size(), got.size()), static_cast<double>(margin));
        NANO_CHECK_MSG(margin < kTieMargin,
                       "int8 metal flipped a decisive token: margin %.4f >= %.2f "
                       "(prompt: %s)",
                       static_cast<double>(margin), static_cast<double>(kTieMargin),
                       c.at("prompt").as_string().c_str());
    }

    nano::ops::set_backend(nano::ops::Backend::cpu);
}

}  // namespace

int main() {
    if (!nano::metal::available()) {
        nano::testing::skip("no Metal device (or kernels failed to compile)");
    }
    std::printf("metal device: %s\n", nano::metal::device_name());

    // A metal request must succeed here; an unavailable backend was the
    // skip above. Also prove the refusal path can't be reached silently:
    // backend starts as cpu.
    NANO_CHECK(nano::ops::backend() == nano::ops::Backend::cpu);

    test_linear_f32_parity();
    test_linear_q8_parity();
    test_backend_routing();
    test_model_goldens();
    return nano::testing::finish("test_metal");
}
