// F024: the decode hot loop performs zero heap allocations.
//
// This TU replaces the global operator new/delete with counting versions
// (a program-wide replacement the linker applies to every translation unit,
// std::vector included). After warmup — prefill plus one decode step, which
// sizes the engine's scratch buffers — 16 decode steps through forward() +
// sampler must not allocate once, on both the greedy and the sampling path.
// A canary check proves the counter actually intercepts allocations, so the
// zero-assert can't pass vacuously. Finally the F019 golden re-runs to show
// the refactor changed no output.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "model/qwen2.hpp"
#include "model/sampler.hpp"
#include "model/tokenizer.hpp"
#include "testing.hpp"

// --- counting global allocator ----------------------------------------------

namespace {
std::atomic<int64_t> g_alloc_count{0};
}  // namespace

void* operator new(std::size_t size) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(size)) {
        return p;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) { return ::operator new(size); }

void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

// --- test --------------------------------------------------------------------

namespace {

std::vector<int32_t> ints(const nano::json::Value& v) {
    std::vector<int32_t> out;
    for (const auto& x : v.items()) {
        out.push_back(static_cast<int32_t>(x.as_int()));
    }
    return out;
}

/// Allocations across `steps` decode iterations (sample -> forward one token).
int64_t count_decode_allocs(nano::Engine& engine, nano::Sampler& sampler,
                            std::span<const float>& logits, int steps,
                            double* ms_per_token) {
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t before = g_alloc_count.load(std::memory_order_relaxed);
    for (int i = 0; i < steps; ++i) {
        const int32_t fed[] = {sampler.sample(logits)};
        logits = engine.forward(fed);
    }
    const int64_t allocs = g_alloc_count.load(std::memory_order_relaxed) - before;
    *ms_per_token = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count() /
                    steps;
    return allocs;
}

}  // namespace

int main() {
    const std::string dir = nano::testing::require_model_dir();

    // Canary: the replaced operator new must actually be the one counting.
    {
        const int64_t before = g_alloc_count.load();
        std::vector<int> canary(1000);
        NANO_CHECK_MSG(g_alloc_count.load() > before,
                       "allocation counter is not intercepting operator new");
    }

    const nano::json::Value golden =
        nano::json::parse(nano::json::read_file("tests/data/generate_golden.json"));
    const auto& c = golden.at("cases").items().front();
    const std::vector<int32_t> prompt_ids = ints(c.at("prompt_ids"));

    nano::Engine engine(dir);
    nano::Sampler greedy({.temperature = 0.0f, .top_k = 0, .top_p = 1.0f, .seed = 0},
                         engine.model().config.vocab_size);
    nano::Sampler sampling({.temperature = 0.8f, .top_k = 20, .top_p = 0.95f, .seed = 1},
                           engine.model().config.vocab_size);

    // Warmup: prefill sizes hidden_ and the scratch buffers; one decode step
    // settles anything that only allocates on its first use.
    std::span<const float> logits = engine.forward(prompt_ids);
    const int32_t warm[] = {nano::argmax(logits)};
    logits = engine.forward(warm);

    double greedy_ms = 0.0;
    const int64_t greedy_allocs =
        count_decode_allocs(engine, greedy, logits, 16, &greedy_ms);
    NANO_CHECK_MSG(greedy_allocs == 0, "%lld allocations in 16 greedy decode steps",
                   static_cast<long long>(greedy_allocs));

    double sampling_ms = 0.0;
    const int64_t sampling_allocs =
        count_decode_allocs(engine, sampling, logits, 16, &sampling_ms);
    NANO_CHECK_MSG(sampling_allocs == 0, "%lld allocations in 16 sampling decode steps",
                   static_cast<long long>(sampling_allocs));

    std::printf("decode: 0 allocations over 16+16 steps "
                "(greedy %.1f ms/token, sampling %.1f ms/token)\n",
                greedy_ms, sampling_ms);

    // The refactor must not have changed a single token: re-check the first
    // HF greedy golden end to end.
    engine.reset();
    const nano::Tokenizer tok = nano::Tokenizer::from_dir(dir);
    const std::vector<int32_t> stop_ids = {tok.special_id("<|im_end|>"),
                                           tok.special_id("<|endoftext|>")};
    const std::vector<int32_t> got =
        nano::greedy_generate(engine, prompt_ids, 32, stop_ids);
    NANO_CHECK_MSG(got == ints(c.at("generated_ids")),
                   "greedy output changed after the scratch-buffer refactor");

    return nano::testing::finish("test_alloc");
}
