// F017: transformer layer 0 output vs HF fp32 reference activations.
// F018: full forward pass — last-position logits and argmax vs HF fp32.
//
// Goldens come from scripts/gen_golden.py running the real HF model in fp32.
// Requires the downloaded model (skips otherwise).
#include "model/qwen2.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "testing.hpp"

namespace {

std::vector<float> floats(const nano::json::Value& v) {
    std::vector<float> out;
    for (const auto& x : v.items()) {
        out.push_back(static_cast<float>(x.as_double()));
    }
    return out;
}

std::vector<int32_t> ints(const nano::json::Value& v) {
    std::vector<int32_t> out;
    for (const auto& x : v.items()) {
        out.push_back(static_cast<int32_t>(x.as_int()));
    }
    return out;
}

void check_close(const std::vector<float>& got, const std::vector<float>& want,
                 float rtol, float atol, const char* what) {
    NANO_CHECK_MSG(got.size() == want.size(), "%s: size %zu vs %zu", what, got.size(),
                   want.size());
    if (got.size() != want.size()) {
        return;
    }
    float worst = 0.0f;
    size_t worst_i = 0;
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
        const float err = std::fabs(got[i] - want[i]);
        if (err > atol + rtol * std::fabs(want[i])) {
            ok = false;
        }
        if (err > worst) {
            worst = err;
            worst_i = i;
        }
    }
    NANO_CHECK_MSG(ok, "%s: worst |diff| %.3g at [%zu] (got %.8g, want %.8g)", what,
                   static_cast<double>(worst), worst_i,
                   static_cast<double>(got[worst_i]),
                   static_cast<double>(want[worst_i]));
}

}  // namespace

int main() {
    const std::string dir = nano::testing::require_model_dir();
    nano::Engine engine(dir);
    const nano::ModelConfig& cfg = engine.model().config;

    // ---- F017: layer 0 in isolation --------------------------------------
    {
        const nano::json::Value g =
            nano::json::parse(nano::json::read_file("tests/data/layer_golden.json"));
        const std::vector<int32_t> ids = ints(g.at("ids"));
        const std::vector<float> want_in = floats(g.at("layer0_in"));
        const std::vector<float> want_out = floats(g.at("layer0_out"));
        const int64_t tokens = static_cast<int64_t>(ids.size());

        // Embedding lookup must reproduce HF's layer-0 input exactly: both
        // sides are the same bf16 weights widened to fp32, no arithmetic.
        std::vector<float> hidden(static_cast<size_t>(tokens * cfg.hidden_size));
        engine.embed(ids, hidden.data());
        bool embed_exact = hidden.size() == want_in.size();
        for (size_t i = 0; embed_exact && i < hidden.size(); ++i) {
            embed_exact = hidden[i] == want_in[i];
        }
        NANO_CHECK_MSG(embed_exact, "embedding lookup differs from HF layer-0 input");

        // Drive layer 0 alone, starting from the golden input, with a fresh
        // cache — isolates the block from the embedding path.
        //
        // Tolerance: rtol 1e-4 with an atol floor of 1e-4. The floor is not
        // slack — it is HF's own noise: recomputing this layer in float64
        // shows the HF fp32 golden itself deviates by up to 4.1e-5 absolute
        // (accumulation order), and our fp32 result lands within 4.5e-5 of
        // the golden, i.e. at the same fp32 noise floor.
        nano::KvCache cache(cfg, tokens);
        nano::layer_forward(engine.model(), 0, hidden.data(), tokens, 0, cache);
        check_close(hidden, want_out, 1e-4f, 1e-4f, "layer0 forward");
    }

    // ---- F018: full forward, last-position logits ------------------------
    {
        const nano::json::Value g =
            nano::json::parse(nano::json::read_file("tests/data/logits_golden.json"));
        for (const auto& c : g.at("cases").items()) {
            const std::string& prompt = c.at("prompt").as_string();
            const std::vector<int32_t> ids = ints(c.at("ids"));

            engine.reset();
            const std::span<const float> logits = engine.forward(ids);

            // Top-1 must match exactly — this is what greedy decoding emits.
            int64_t argmax = 0;
            for (int64_t i = 1; i < cfg.vocab_size; ++i) {
                if (logits[static_cast<size_t>(i)] > logits[static_cast<size_t>(argmax)]) {
                    argmax = i;
                }
            }
            NANO_CHECK_MSG(argmax == c.at("argmax").as_int(),
                           "argmax %lld vs golden %lld (prompt: %s)",
                           static_cast<long long>(argmax),
                           static_cast<long long>(c.at("argmax").as_int()),
                           prompt.c_str());

            // Logit values at the golden top-10 ids and at a 512-point stride
            // across the vocab (the full 151936-logit vector is too large to
            // commit; this samples it densely enough to catch real bugs).
            const std::vector<int32_t> top_ids = ints(c.at("top10_ids"));
            const std::vector<float> top_want = floats(c.at("top10_logits"));
            std::vector<float> top_got;
            for (int32_t id : top_ids) {
                top_got.push_back(logits[static_cast<size_t>(id)]);
            }
            check_close(top_got, top_want, 1e-3f, 1e-3f,
                        (std::string("top10 logits: ") + prompt).c_str());

            const int64_t stride = c.at("sample_stride").as_int();
            const std::vector<float> sample_want = floats(c.at("sample_logits"));
            std::vector<float> sample_got;
            for (int64_t i = 0; i < cfg.vocab_size; i += stride) {
                sample_got.push_back(logits[static_cast<size_t>(i)]);
            }
            check_close(sample_got, sample_want, 1e-3f, 1e-3f,
                        (std::string("strided logits: ") + prompt).c_str());
        }
    }

    return nano::testing::finish("test_forward");
}
