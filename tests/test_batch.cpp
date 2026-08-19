// F036, engine level: batched decode must be indistinguishable from solo
// decode in its OUTPUTS and distinguishable in its THROUGHPUT.
//
//   A. Tie to the validated engine: greedy tokens from BatchEngine running
//      one sequence alone are bit-identical to Engine (paged) — the engine
//      whose outputs test_forward/test_generate tie to the HF goldens.
//   B. Cross-request isolation, adversarially: four different prompts are
//      admitted at DIFFERENT decode steps, retired at different steps, and
//      one is admitted only after another's pages went back to the pool —
//      so every sequence's pages interleave with strangers' and one runs
//      entirely on recycled pages (LIFO reuse). If attention ever read
//      another sequence's rows — block-table aliasing, stale pages, a wrong
//      page id — these tokens could not stay bit-identical to the solo
//      runs. This is the contamination test the feature list asks for:
//      it fails on construction bugs instead of hoping one shows up.
//   C. The numbers: n=1 decode latency within 20% of Engine's (the
//      feature's regression bound), and a 4-sequence step decodes 4 tokens
//      in far less than 4x a 1-sequence step (the throughput win).
//
// Needs the model; skips (77) without it.
#include "model/batch.hpp"

#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "model/qwen2.hpp"
#include "model/tokenizer.hpp"
#include "testing.hpp"

namespace {

constexpr int64_t kNewTokens = 24;  // per sequence, everywhere in this test

/// Greedy continuation via the single-sequence Engine: always exactly
/// `kNewTokens` tokens (no early stop — determinism comparisons want fixed
/// lengths, and greedy at these lengths doesn't hit eos anyway).
std::vector<int32_t> engine_greedy(nano::Engine& engine,
                                   const std::vector<int32_t>& prompt) {
    engine.reset();
    std::vector<int32_t> out;
    int32_t tok = nano::argmax(engine.forward(prompt));
    out.push_back(tok);
    while (static_cast<int64_t>(out.size()) < kNewTokens) {
        const int32_t fed[] = {tok};
        tok = nano::argmax(engine.forward(fed));
        out.push_back(tok);
    }
    return out;
}

/// One in-flight sequence in the hand-rolled schedules below: its cache
/// handle, everything it has produced, and the token it feeds next.
struct Runner {
    std::unique_ptr<nano::Sequence> seq;
    std::vector<int32_t> got;
    int32_t last = 0;

    bool done() const { return static_cast<int64_t>(got.size()) >= kNewTokens; }
};

void admit(nano::BatchEngine& eng, Runner& r, const std::vector<int32_t>& prompt) {
    r.seq = eng.new_sequence();
    r.last = nano::argmax(eng.prefill(*r.seq, prompt));
    r.got.push_back(r.last);
}

/// One decode step over whichever runners are live right now.
void step(nano::BatchEngine& eng, const std::vector<Runner*>& live) {
    std::vector<nano::Sequence*> seqs;
    std::vector<int32_t> toks;
    for (Runner* r : live) {
        seqs.push_back(r->seq.get());
        toks.push_back(r->last);
    }
    const std::span<const float> logits = eng.decode_step(seqs, toks);
    const int64_t vocab = eng.model().config.vocab_size;
    for (size_t i = 0; i < live.size(); ++i) {
        live[i]->last = nano::argmax(
            logits.subspan(i * static_cast<size_t>(vocab), static_cast<size_t>(vocab)));
        live[i]->got.push_back(live[i]->last);
    }
}

std::vector<int32_t> solo_greedy(nano::BatchEngine& eng,
                                 const std::vector<int32_t>& prompt) {
    Runner r;
    admit(eng, r, prompt);
    while (!r.done()) {
        step(eng, {&r});
    }
    return r.got;
}

/// Milliseconds per call of `fn`, after a short warmup.
double ms_per_call(const std::function<void()>& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    const auto dt = std::chrono::steady_clock::now() - t0;
    return std::chrono::duration<double, std::milli>(dt).count() / iters;
}

bool same_tokens(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    return a == b;
}

}  // namespace

int main() {
    const std::string dir = nano::testing::require_model_dir();
    const nano::Tokenizer tok = nano::Tokenizer::from_dir(dir);

    const std::vector<std::string> prompts = {
        "The capital of France is",
        "In C++, a std::vector is",
        "The theory of relativity says that",
        "Deep learning is",
    };
    std::vector<std::vector<int32_t>> prompt_ids;
    for (const std::string& p : prompts) {
        prompt_ids.push_back(tok.encode(p));
    }

    // Both engines stay alive for the whole test (~2 GB of fp32 weights
    // each) so part C can time them back to back — timing the Engine in an
    // early phase and the BatchEngine minutes of hot loops later would let
    // clock/thermal drift masquerade as a latency regression.
    nano::Engine engine(dir, /*max_seq=*/2048, "", nano::KvLayout::paged);
    nano::BatchEngine eng(dir, /*max_seq=*/2048, "", /*max_seqs=*/4);

    // ---- Engine goldens (the engine test_generate ties to the HF goldens) --
    std::vector<std::vector<int32_t>> engine_golden;
    for (const auto& ids : prompt_ids) {
        engine_golden.push_back(engine_greedy(engine, ids));
    }

    // ---- A: BatchEngine solo == Engine, token for token --------------------
    std::vector<std::vector<int32_t>> solo;
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
        solo.push_back(solo_greedy(eng, prompt_ids[i]));
        NANO_CHECK_MSG(same_tokens(solo[i], engine_golden[i]),
                       "prompt %zu: BatchEngine solo diverged from Engine", i);
    }
    std::printf("A: 4 solo runs match Engine (paged) token-for-token\n");

    // ---- B: staggered batch, adversarial page traffic ----------------------
    // Admission order S0, S1, S2 (interleaves everyone's pages in the pool);
    // S0 finishes FIRST and S3 is admitted only after S0's pages are freed,
    // so S3 runs on S0's recycled pages (LIFO pool) while S1/S2 are mid-
    // flight. Every sequence must still reproduce its solo tokens exactly.
    {
        Runner r0, r1, r2, r3;
        admit(eng, r0, prompt_ids[0]);
        for (int i = 0; i < 4; ++i) {
            step(eng, {&r0});  // S0 alone: 5 tokens
        }
        admit(eng, r1, prompt_ids[1]);  // prefill between S0's decode steps
        for (int i = 0; i < 4; ++i) {
            step(eng, {&r0, &r1});  // S0: 9, S1: 5
        }
        admit(eng, r2, prompt_ids[2]);
        while (!r0.done()) {
            step(eng, {&r0, &r1, &r2});  // until S0 hits 24 (S1: 20, S2: 16)
        }
        r0.seq.reset();  // S0's pages go back to the pool (LIFO)...
        admit(eng, r3, prompt_ids[3]);  // ...and S3 reuses them immediately
        while (!r1.done() || !r2.done() || !r3.done()) {
            std::vector<Runner*> live;
            for (Runner* r : {&r1, &r2, &r3}) {
                if (!r->done()) {
                    live.push_back(r);
                }
            }
            step(eng, live);
        }

        const Runner* runners[] = {&r0, &r1, &r2, &r3};
        for (size_t i = 0; i < 4; ++i) {
            NANO_CHECK_MSG(same_tokens(runners[i]->got, solo[i]),
                           "prompt %zu: batched tokens != solo tokens "
                           "(cross-request contamination or position bug)",
                           i);
        }
        std::printf(
            "B: staggered 4-way batch (admissions mid-flight, page reuse) matches "
            "all 4 solo runs; KV high-water %.2f MiB\n",
            static_cast<double>(eng.kv_bytes_allocated()) / (1024.0 * 1024.0));
    }

    // ---- C: latency + throughput -------------------------------------------
    // All three timings run back to back, at the same sequence positions,
    // in one already-hot process — the only variable left is the code path.
    {
        engine.reset();
        int32_t tok_id = nano::argmax(engine.forward(prompt_ids[0]));
        const double engine_ms = ms_per_call(
            [&] {
                const int32_t fed[] = {tok_id};
                tok_id = nano::argmax(engine.forward(fed));
            },
            /*warmup=*/5, /*iters=*/30);

        // n=1 latency through the batched path (feature bound: <20%
        // regression; expected: near-identical, it is the same math).
        Runner r;
        admit(eng, r, prompt_ids[0]);
        const double batch1_ms = ms_per_call([&] { step(eng, {&r}); }, 5, 30);
        r.seq.reset();

        // 4-sequence step at the same positions: 4 tokens per step.
        Runner rs[4];
        for (size_t i = 0; i < 4; ++i) {
            admit(eng, rs[i], prompt_ids[0]);
        }
        const double batch4_ms = ms_per_call(
            [&] { step(eng, {&rs[0], &rs[1], &rs[2], &rs[3]}); }, 5, 30);

        // Aggregate throughput vs the pre-F036 serial baseline: 4 sequences
        // served by one Engine take 4 steps where the batch takes 1.
        const double speedup = 4.0 * engine_ms / batch4_ms;
        std::printf(
            "C: decode ms/step — Engine %.2f, BatchEngine n=1 %.2f, n=4 %.2f "
            "(aggregate %.1f tok/s batched vs %.1f serial; speedup %.2fx)\n",
            engine_ms, batch1_ms, batch4_ms, 4000.0 / batch4_ms,
            1000.0 / engine_ms, speedup);
        NANO_CHECK_MSG(batch1_ms <= engine_ms * 1.20,
                       "n=1 decode regressed >20%% vs Engine: %.2f vs %.2f ms",
                       batch1_ms, engine_ms);
        // The whole point of the feature. Measured ~2.7x on the M3 Pro; the
        // bound is loose so background load can't flake the build.
        NANO_CHECK_MSG(speedup > 1.5,
                       "batching 4 sequences gained only %.2fx aggregate throughput",
                       speedup);
    }

    return nano::testing::finish("test_batch");
}
