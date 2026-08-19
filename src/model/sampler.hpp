// Token sampling: temperature, top-k, top-p, with a seeded RNG.
//
// Semantics mirror Hugging Face `generate()` exactly, in its order:
//   1. logits are divided by `temperature`
//   2. top-k keeps the k highest logits
//   3. top-p keeps the smallest set of tokens (highest probability first)
//      whose cumulative probability reaches top_p — walking tokens in
//      descending order, a token is kept while the cumulative probability
//      *before* it is still < top_p, so the token that crosses the
//      threshold is included (verified against HF's TopPLogitsWarper)
//   4. the survivors are renormalized and one is drawn
//
// temperature == 0 short-circuits to argmax (greedy) and consumes no
// randomness, so seeded runs stay reproducible even with warmup calls.
//
// Determinism: mt19937_64 produces a bit-exact stream on every platform (the
// C++ standard fixes the algorithm), and the uniform [0,1) draw is built
// directly from its 64-bit output rather than std::uniform_real_distribution,
// whose mapping is implementation-defined. Same seed => same tokens anywhere.
#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace nano {

struct SamplerOptions {
    float temperature = 1.0f;  // 0 = greedy (argmax)
    int64_t top_k = 0;         // 0 = disabled (consider all tokens)
    float top_p = 1.0f;        // 1 = disabled
    uint64_t seed = 0;
};

class Sampler {
public:
    /// Buffers are sized for `vocab_size` once, here — sample() itself never
    /// allocates (the decode loop must stay allocation-free, F024).
    Sampler(const SamplerOptions& opts, int64_t vocab_size);

    /// Draws the next token id from `logits` (length vocab_size).
    int32_t sample(std::span<const float> logits);

    /// Test hook: the exact distribution sample() draws from, scattered to
    /// full vocab size (filtered-out tokens are 0). Consumes no randomness.
    void probabilities(std::span<const float> logits, std::span<float> out);

private:
    /// Applies temperature/top-k/top-p. On return, candidate_[0, n) holds
    /// token ids in descending probability order and prob_[0, n) their
    /// renormalized probabilities; returns n.
    int64_t compute_candidates(std::span<const float> logits);

    SamplerOptions opts_;
    int64_t vocab_size_ = 0;
    std::mt19937_64 rng_;
    std::vector<int32_t> candidate_;  // [vocab] token ids, filled per call
    std::vector<float> prob_;         // [vocab] matching probabilities
};

}  // namespace nano
