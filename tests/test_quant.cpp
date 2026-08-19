// F027: int8 weight quantization.
//
// Three layers of checks:
//   1. quantize_rows math on hand-built and random matrices: scale =
//      absmax/127, every dequantized value within scale/2 of the original,
//      exact round-trip when the values are representable, zero-row safety.
//   2. The int8 kernel: linear_q8 matches a dequantize-then-linear reference
//      within fp tolerance, and is bit-identical across thread counts.
//      The safetensors writer round-trips through our own reader.
//   3. The real model (skips without weights): `quantize_model_file` output
//      loads, and the quantized engine's top-1 token matches the HF golden
//      argmax on all 3 logits prompts. Greedy 32-token continuations are
//      compared against the fp32 goldens and any divergence is PRINTED
//      honestly (quantization is lossy; the acceptance bar is top-1
//      unchanged, with divergence documented — see feature_list.json).
#include "core/quant.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "core/ops.hpp"
#include "core/safetensors.hpp"
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

void test_quantize_rows() {
    // scale 0.5 exactly (63.5 / 127), so every value here is representable
    // and the round trip must be bit-exact.
    const std::vector<float> exact = {63.5f, -32.0f, 1.5f, 0.0f};
    nano::QuantMatrix m = nano::quantize_rows(exact.data(), 1, 4);
    NANO_CHECK(m.scales[0] == 0.5f);
    NANO_CHECK(m.q[0] == 127 && m.q[1] == -64 && m.q[2] == 3 && m.q[3] == 0);
    for (int64_t c = 0; c < 4; ++c) {
        NANO_CHECK(nano::dequant_at(m, 0, c) == exact[static_cast<size_t>(c)]);
    }

    // A negative absmax value must map to -127, not overflow.
    const std::vector<float> neg = {-2.0f, 1.0f};
    m = nano::quantize_rows(neg.data(), 1, 2);
    NANO_CHECK(m.q[0] == -127);
    NANO_CHECK(m.scales[0] == 2.0f / 127.0f);

    // All-zero row: scale 0, values 0, dequantizes to exactly 0.
    const std::vector<float> zeros(8, 0.0f);
    m = nano::quantize_rows(zeros.data(), 1, 8);
    NANO_CHECK(m.scales[0] == 0.0f);
    for (int64_t c = 0; c < 8; ++c) {
        NANO_CHECK(nano::dequant_at(m, 0, c) == 0.0f);
    }

    // Random rows: per-row scale is absmax/127 and every dequantized value
    // sits within half a quantization step of the original.
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    const int64_t rows = 7;
    const int64_t cols = 100;
    std::vector<float> w(static_cast<size_t>(rows * cols));
    for (float& v : w) {
        v = dist(rng);
    }
    m = nano::quantize_rows(w.data(), rows, cols);
    for (int64_t r = 0; r < rows; ++r) {
        float absmax = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            absmax = std::max(absmax, std::fabs(w[static_cast<size_t>(r * cols + c)]));
        }
        NANO_CHECK(m.scales[static_cast<size_t>(r)] == absmax / 127.0f);
        // Half a step, plus fp32 slack: the quantizer computes w * (127 /
        // absmax), and rounding that product can push a value sitting
        // EXACTLY on a .5 tie one product-ulp (~127 * 2^-24 ≈ 8e-6 steps)
        // past the mathematical half-step. Observed on the real q_proj:
        // w/scale = -63.5 exactly, quantized to -63, rel error 1.000007.
        const float bound = m.scales[static_cast<size_t>(r)] * 0.5f * 1.0001f;
        for (int64_t c = 0; c < cols; ++c) {
            const float err = std::fabs(nano::dequant_at(m, r, c) -
                                        w[static_cast<size_t>(r * cols + c)]);
            NANO_CHECK_MSG(err <= bound, "row %lld col %lld: err %g > bound %g",
                           static_cast<long long>(r), static_cast<long long>(c),
                           static_cast<double>(err), static_cast<double>(bound));
        }
    }
}

void test_linear_q8_kernel() {
    // Shapes chosen to straddle the SIMD unroll widths (16/4/tail) and, for
    // the threaded check, to exceed the parallel threshold.
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (const int64_t d_in : {13, 64, 896}) {
        const int64_t tokens = 3;
        const int64_t d_out = 5;
        std::vector<float> x(static_cast<size_t>(tokens * d_in));
        std::vector<float> w(static_cast<size_t>(d_out * d_in));
        std::vector<float> bias(static_cast<size_t>(d_out));
        for (float& v : x) v = dist(rng);
        for (float& v : w) v = dist(rng);
        for (float& v : bias) v = dist(rng);

        const nano::QuantMatrix q = nano::quantize_rows(w.data(), d_out, d_in);

        // Reference: dequantize, then the plain scalar linear. linear_q8
        // must equal this up to fp reduction order (NEON sums in a
        // different order), hence tolerance, not equality.
        std::vector<float> ref(static_cast<size_t>(tokens * d_out));
        for (int64_t t = 0; t < tokens; ++t) {
            for (int64_t o = 0; o < d_out; ++o) {
                double acc = 0.0;
                for (int64_t i = 0; i < d_in; ++i) {
                    acc += static_cast<double>(x[static_cast<size_t>(t * d_in + i)]) *
                           static_cast<double>(nano::dequant_at(q, o, i));
                }
                ref[static_cast<size_t>(t * d_out + o)] =
                    static_cast<float>(acc) + bias[static_cast<size_t>(o)];
            }
        }

        std::vector<float> y(static_cast<size_t>(tokens * d_out));
        nano::ops::linear_q8(x.data(), q.q.data(), q.scales.data(), bias.data(),
                             y.data(), tokens, d_in, d_out);
        for (size_t i = 0; i < y.size(); ++i) {
            const float tol = 1e-4f * std::max(1.0f, std::fabs(ref[i]));
            NANO_CHECK_MSG(std::fabs(y[i] - ref[i]) <= tol,
                           "d_in=%lld i=%zu: got %g want %g",
                           static_cast<long long>(d_in), i,
                           static_cast<double>(y[i]), static_cast<double>(ref[i]));
        }
    }

    // Threaded == serial, bit-exact: disjoint column slices, same arithmetic.
    const int64_t tokens = 2;
    const int64_t d_in = 896;
    const int64_t d_out = 1024;  // 2 * 896 * 1024 > the parallel threshold
    std::vector<float> x(static_cast<size_t>(tokens * d_in));
    std::vector<float> w(static_cast<size_t>(d_out * d_in));
    for (float& v : x) v = dist(rng);
    for (float& v : w) v = dist(rng);
    const nano::QuantMatrix q = nano::quantize_rows(w.data(), d_out, d_in);

    std::vector<float> serial(static_cast<size_t>(tokens * d_out));
    nano::ops::set_num_threads(1);
    nano::ops::linear_q8(x.data(), q.q.data(), q.scales.data(), nullptr,
                         serial.data(), tokens, d_in, d_out);
    for (const int threads : {2, 5, 0}) {
        std::vector<float> threaded(static_cast<size_t>(tokens * d_out));
        nano::ops::set_num_threads(threads);
        nano::ops::linear_q8(x.data(), q.q.data(), q.scales.data(), nullptr,
                             threaded.data(), tokens, d_in, d_out);
        NANO_CHECK_MSG(std::memcmp(serial.data(), threaded.data(),
                                   serial.size() * sizeof(float)) == 0,
                       "threads=%d differs from serial", threads);
    }
    nano::ops::set_num_threads(0);
}

void test_safetensors_writer() {
    const std::string path =
        (std::filesystem::temp_directory_path() / "nanoserve_test_quant.safetensors")
            .string();

    const std::vector<int8_t> qdata = {-127, -1, 0, 1, 127, 42};
    const std::vector<float> fdata = {0.5f, -1.25f};
    const std::vector<nano::SaveTensor> tensors = {
        {"w", nano::DType::I8, {2, 3}, qdata.data(), qdata.size()},
        {"w.scale", nano::DType::F32, {2}, fdata.data(), fdata.size() * sizeof(float)},
    };
    nano::write_safetensors(path, tensors, {{"format", "test"}});

    const nano::SafeTensors st(path);
    NANO_CHECK(st.tensors().size() == 2);
    const nano::TensorInfo& w = st.require("w");
    NANO_CHECK(w.dtype == nano::DType::I8);
    NANO_CHECK(w.shape == (std::vector<int64_t>{2, 3}));
    const std::span<const std::byte> raw = st.raw(w);
    NANO_CHECK(raw.size() == qdata.size());
    NANO_CHECK(std::memcmp(raw.data(), qdata.data(), qdata.size()) == 0);
    const std::vector<float> scales = st.to_f32(st.require("w.scale"));
    NANO_CHECK(scales == fdata);

    // A shape/bytes mismatch must be rejected before anything is written.
    const std::vector<nano::SaveTensor> bad = {
        {"w", nano::DType::I8, {2, 4}, qdata.data(), qdata.size()}};
    NANO_CHECK_THROWS(nano::write_safetensors(path + ".bad", bad, {}));

    std::filesystem::remove(path);
}

void test_real_model() {
    const std::string dir = nano::testing::require_model_dir();
    const std::string int8_file = dir + "/model.int8.safetensors";

    // Always regenerate: the test must exercise today's quantizer, not a
    // stale file from an older format.
    const int n = nano::quantize_model_file(dir, int8_file);
    NANO_CHECK_MSG(n == 169, "Qwen2.5-0.5B has 169 2D matrices, quantized %d", n);

    // Spot-check dequantization error on real weights (layer 0 q_proj):
    // every value within half a step of the bf16-widened original.
    {
        const nano::SafeTensors fp32(dir + "/model.safetensors");
        const nano::TensorInfo& info =
            fp32.require("model.layers.0.self_attn.q_proj.weight");
        const std::vector<float> w = fp32.to_f32(info);
        const nano::QuantMatrix q =
            nano::quantize_rows(w.data(), info.shape[0], info.shape[1]);
        float max_err = 0.0f;
        bool ok = true;
        for (int64_t r = 0; r < q.rows && ok; ++r) {
            // Same fp32 tie slack as the synthetic check above.
            const float bound = q.scales[static_cast<size_t>(r)] * 0.5f * 1.0001f;
            for (int64_t c = 0; c < q.cols; ++c) {
                const float err =
                    std::fabs(nano::dequant_at(q, r, c) -
                              w[static_cast<size_t>(r * q.cols + c)]);
                max_err = std::max(max_err, err);
                ok = ok && err <= bound;
            }
        }
        NANO_CHECK_MSG(ok, "q_proj dequant error exceeds scale/2");
        std::printf("q_proj [%lld x %lld]: max dequant error %.3g\n",
                    static_cast<long long>(q.rows), static_cast<long long>(q.cols),
                    static_cast<double>(max_err));
    }

    nano::Engine engine(dir, 2048, int8_file);
    NANO_CHECK(engine.model().quantized());

    // Top-1 on the HF logits goldens must survive quantization (the F027
    // acceptance bar).
    const nano::json::Value lg =
        nano::json::parse(nano::json::read_file("tests/data/logits_golden.json"));
    for (const auto& c : lg.at("cases").items()) {
        const std::vector<int32_t> ids = ints(c.at("ids"));
        const int32_t want = static_cast<int32_t>(c.at("argmax").as_int());
        engine.reset();
        const std::span<const float> logits = engine.forward(ids);
        const int32_t got = nano::argmax(logits);
        NANO_CHECK_MSG(got == want, "int8 top-1 %d != golden %d (prompt: %s)", got,
                       want, c.at("prompt").as_string().c_str());

        // Quantify the damage: max |int8 logit - HF fp32 logit| over the
        // golden top-10. Printed, not asserted — this is the honest
        // "how lossy is it" number for the session log. (For reference, the
        // fp32 engine itself sits within ~1e-3 of HF.)
        const auto& top_ids = c.at("top10_ids").items();
        const auto& top_logits = c.at("top10_logits").items();
        float max_diff = 0.0f;
        for (size_t i = 0; i < top_ids.size(); ++i) {
            const auto id = static_cast<size_t>(top_ids[i].as_int());
            const float hf = static_cast<float>(top_logits[i].as_double());
            max_diff = std::max(max_diff, std::fabs(logits[id] - hf));
        }
        std::printf("int8 vs HF fp32 top-10 logits (%s): max |delta| = %.4f\n",
                    c.at("prompt").as_string().c_str(),
                    static_cast<double>(max_diff));
    }

    // Greedy continuations vs the fp32 goldens: count how far each matches
    // and print it. Quantization MAY legitimately diverge mid-sequence; the
    // hard assertion is only that the first token (== the argmax check
    // above, on fresh state) matches. Whatever happens gets printed, and the
    // session log records the measured result.
    const std::vector<int32_t> stop_ids = {151645, 151643};  // <|im_end|>, <|endoftext|>
    const nano::json::Value gg =
        nano::json::parse(nano::json::read_file("tests/data/generate_golden.json"));
    for (const auto& c : gg.at("cases").items()) {
        const std::vector<int32_t> prompt_ids = ints(c.at("prompt_ids"));
        const std::vector<int32_t> want = ints(c.at("generated_ids"));
        engine.reset();
        const std::vector<int32_t> got =
            nano::greedy_generate(engine, prompt_ids, 32, stop_ids);

        size_t match = 0;
        while (match < got.size() && match < want.size() &&
               got[match] == want[match]) {
            ++match;
        }
        const bool identical = match == got.size() && match == want.size();
        NANO_CHECK_MSG(!got.empty() && got[0] == want[0],
                       "int8 first greedy token differs (prompt: %s)",
                       c.at("prompt").as_string().c_str());
        std::printf("int8 greedy vs fp32 golden (%s): %s (%zu/%zu tokens match)\n",
                    c.at("prompt").as_string().c_str(),
                    identical ? "IDENTICAL" : "diverges", match,
                    std::max(want.size(), got.size()));
    }
}

}  // namespace

int main() {
    test_quantize_rows();
    test_linear_q8_kernel();
    test_safetensors_writer();
    test_real_model();
    return nano::testing::finish("test_quant");
}
