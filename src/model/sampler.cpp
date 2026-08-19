#include "model/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "model/qwen2.hpp"  // argmax

namespace nano {

Sampler::Sampler(const SamplerOptions& opts, int64_t vocab_size)
    : opts_(opts),
      vocab_size_(vocab_size),
      rng_(opts.seed),
      candidate_(static_cast<size_t>(vocab_size)),
      prob_(static_cast<size_t>(vocab_size)) {}

int64_t Sampler::compute_candidates(std::span<const float> logits) {
    const int64_t v = vocab_size_;
    for (int64_t i = 0; i < v; ++i) {
        candidate_[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }

    // Sort candidate ids by logit, descending. Ties break toward the lower
    // token id so the order (and therefore the top-k/top-p cut) is fully
    // deterministic. std::sort/partial_sort allocate nothing, unlike
    // stable_sort. Dividing by temperature doesn't change the order, so
    // sorting raw logits is fine.
    const auto by_logit_desc = [&](int32_t a, int32_t b) {
        const float la = logits[static_cast<size_t>(a)];
        const float lb = logits[static_cast<size_t>(b)];
        return la != lb ? la > lb : a < b;
    };
    int64_t n = (opts_.top_k > 0 && opts_.top_k < v) ? opts_.top_k : v;
    if (n < v) {
        // top-k enabled: only the k best need to be in order.
        std::partial_sort(candidate_.begin(), candidate_.begin() + n, candidate_.end(),
                          by_logit_desc);
    } else if (opts_.top_p < 1.0f) {
        // top-p walks tokens best-first, so it needs the full order.
        std::sort(candidate_.begin(), candidate_.end(), by_logit_desc);
    }
    // else: pure temperature sampling — every token survives and the draw
    // doesn't care about order, so skip sorting entirely.

    // Softmax over the surviving candidates at the given temperature
    // (max-subtracted for stability; the sum accumulates in double).
    const float t = opts_.temperature;
    float max_scaled = logits[static_cast<size_t>(candidate_[0])] / t;
    for (int64_t i = 1; i < n; ++i) {
        max_scaled = std::max(max_scaled, logits[static_cast<size_t>(candidate_[static_cast<size_t>(i)])] / t);
    }
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const float l = logits[static_cast<size_t>(candidate_[static_cast<size_t>(i)])];
        const float p = std::exp(l / t - max_scaled);
        prob_[static_cast<size_t>(i)] = p;
        sum += static_cast<double>(p);
    }
    float inv_sum = static_cast<float>(1.0 / sum);
    for (int64_t i = 0; i < n; ++i) {
        prob_[static_cast<size_t>(i)] *= inv_sum;
    }

    // top-p: keep tokens best-first while the cumulative probability before
    // this token is still < top_p (the crossing token is included).
    if (opts_.top_p < 1.0f) {
        double cum_before = 0.0;
        int64_t kept = 0;
        while (kept < n && cum_before < static_cast<double>(opts_.top_p)) {
            cum_before += static_cast<double>(prob_[static_cast<size_t>(kept)]);
            ++kept;
        }
        n = kept;
        // Renormalize the survivors.
        double kept_sum = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            kept_sum += static_cast<double>(prob_[static_cast<size_t>(i)]);
        }
        inv_sum = static_cast<float>(1.0 / kept_sum);
        for (int64_t i = 0; i < n; ++i) {
            prob_[static_cast<size_t>(i)] *= inv_sum;
        }
    }
    return n;
}

int32_t Sampler::sample(std::span<const float> logits) {
    if (opts_.temperature <= 0.0f) {
        return argmax(logits);
    }
    const int64_t n = compute_candidates(logits);

    // Uniform double in [0,1) from the top 53 bits of one mt19937_64 output —
    // the exact construction std::generate_canonical is allowed but not
    // required to use, done by hand so every platform draws the same number.
    const double u = static_cast<double>(rng_() >> 11) * 0x1.0p-53;

    // Invert the CDF: the first candidate whose cumulative probability
    // exceeds u wins. Falls through to the last candidate if float rounding
    // leaves the total a hair under 1.
    double cum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        cum += static_cast<double>(prob_[static_cast<size_t>(i)]);
        if (u < cum) {
            return candidate_[static_cast<size_t>(i)];
        }
    }
    return candidate_[static_cast<size_t>(n - 1)];
}

void Sampler::probabilities(std::span<const float> logits, std::span<float> out) {
    const int64_t n = compute_candidates(logits);
    std::memset(out.data(), 0, sizeof(float) * out.size());
    for (int64_t i = 0; i < n; ++i) {
        out[static_cast<size_t>(candidate_[static_cast<size_t>(i)])] =
            prob_[static_cast<size_t>(i)];
    }
}

}  // namespace nano
