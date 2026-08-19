// F020: the KV cache neither changes results nor fails to pay for itself.
//
// Part A — correctness: decoding token-by-token through the cache must give
// BIT-IDENTICAL logits to re-running the whole sequence through forward() in
// one call. Every op is deterministic and each position's computation only
// depends on cache rows 0..pos, so even fp32 must agree exactly; any diff
// means the cache path and the full path compute different math.
//
// Part B — speed: with the cache, one decode step costs O(seq) attention but
// O(1) matmuls; without it, every step re-runs the whole sequence. We time
// 128 cached decode steps, then time full-sequence forwards at two
// representative lengths (the true per-token cost of cache-less decoding),
// and assert cached < uncached. Numbers are printed for the log.
#include "model/qwen2.hpp"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "model/chat_template.hpp"
#include "model/tokenizer.hpp"
#include "testing.hpp"

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                     t0)
        .count();
}

}  // namespace

int main() {
    const std::string dir = nano::testing::require_model_dir();
    nano::Engine engine(dir);
    const nano::Tokenizer tok = nano::Tokenizer::from_dir(dir);

    const std::vector<nano::ChatMessage> messages = {
        {"user", "Explain what a KV cache does in an LLM."}};
    const std::vector<int32_t> prompt =
        nano::apply_chat_template(tok, messages, /*add_generation_prompt=*/true);

    // ---- Part A: cached logits == full-recompute logits, bit for bit ------
    constexpr int64_t kExactSteps = 5;
    std::vector<std::vector<float>> cached_logits;
    std::vector<int32_t> generated;
    {
        engine.reset();
        std::span<const float> logits = engine.forward(prompt);
        for (int64_t i = 0; i < kExactSteps; ++i) {
            const int32_t next = nano::argmax(logits);
            generated.push_back(next);
            const int32_t fed[] = {next};
            logits = engine.forward(fed);
            cached_logits.emplace_back(logits.begin(), logits.end());
        }
    }
    for (int64_t i = 0; i < kExactSteps; ++i) {
        std::vector<int32_t> full_ids = prompt;
        full_ids.insert(full_ids.end(), generated.begin(),
                        generated.begin() + i + 1);
        engine.reset();
        const std::span<const float> logits = engine.forward(full_ids);

        const std::vector<float>& want = cached_logits[static_cast<size_t>(i)];
        size_t diffs = 0;
        for (size_t j = 0; j < want.size(); ++j) {
            diffs += logits[j] != want[j] ? 1 : 0;
        }
        NANO_CHECK_MSG(diffs == 0,
                       "step %lld: %zu/%zu logits differ between cached and "
                       "full recompute",
                       static_cast<long long>(i), diffs, want.size());
    }

    // ---- Part B: cached decode is faster per token than recompute --------
    constexpr int64_t kDecodeSteps = 128;
    double cached_ms_per_token = 0.0;
    {
        engine.reset();
        std::span<const float> logits = engine.forward(prompt);
        const auto t0 = std::chrono::steady_clock::now();
        for (int64_t i = 0; i < kDecodeSteps; ++i) {
            const int32_t fed[] = {nano::argmax(logits)};
            logits = engine.forward(fed);  // eos ignored: this measures cost
        }
        cached_ms_per_token = ms_since(t0) / kDecodeSteps;
    }

    // Without a cache, producing the next token at sequence length L means a
    // full L-token forward. Sample that cost mid-run and at the end.
    double uncached_ms_per_token = 0.0;
    {
        const int64_t plen = static_cast<int64_t>(prompt.size());
        const int64_t lengths[] = {plen + kDecodeSteps / 2, plen + kDecodeSteps};
        // Build a plausible id sequence of the needed length by repeating the
        // prompt body (values don't matter for timing, shapes do).
        std::vector<int32_t> ids;
        while (static_cast<int64_t>(ids.size()) < lengths[1]) {
            ids.insert(ids.end(), prompt.begin(), prompt.end());
        }
        for (const int64_t len : lengths) {
            engine.reset();
            const auto t0 = std::chrono::steady_clock::now();
            (void)engine.forward(std::span<const int32_t>(ids.data(),
                                                          static_cast<size_t>(len)));
            uncached_ms_per_token += ms_since(t0);
        }
        uncached_ms_per_token /= 2.0;  // mean cost of ONE cache-less step
    }

    std::printf("decode per token: %.1f ms with KV cache, %.1f ms without "
                "(full recompute)\n",
                cached_ms_per_token, uncached_ms_per_token);
    NANO_CHECK_MSG(cached_ms_per_token < uncached_ms_per_token,
                   "cache not faster: %.1f ms vs %.1f ms", cached_ms_per_token,
                   uncached_ms_per_token);

    return nano::testing::finish("test_kvcache");
}
