// F022: sampling — temperature / top-k / top-p with a seeded RNG.
//
// Two layers of verification, neither needing model files:
//   1. The filtered distribution matches the HF logits warpers exactly
//      (tests/data/sampling_golden.json: random logits -> temperature ->
//      top-k -> top-p -> softmax, computed by transformers).
//   2. Hand-built distributions with known probabilities pin down the
//      candidate-set edges (top-k cut, top-p crossing token, ties) plus
//      determinism and draw-frequency sanity.
#include "model/sampler.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "model/qwen2.hpp"  // argmax
#include "testing.hpp"

namespace {

std::vector<float> floats(const nano::json::Value& v) {
    std::vector<float> out;
    for (const auto& x : v.items()) {
        out.push_back(static_cast<float>(x.as_double()));
    }
    return out;
}

/// Distribution the sampler would draw from, full vocab length.
std::vector<float> probs_of(const nano::SamplerOptions& opts,
                            const std::vector<float>& logits) {
    nano::Sampler s(opts, static_cast<int64_t>(logits.size()));
    std::vector<float> out(logits.size());
    s.probabilities(logits, out);
    return out;
}

/// logits = log(p): softmax recovers p (up to fp32 rounding), so tests can
/// state distributions directly as probabilities.
std::vector<float> log_probs(const std::vector<float>& p) {
    std::vector<float> out;
    for (float x : p) {
        out.push_back(std::log(x));
    }
    return out;
}

void check_golden_cases() {
    const nano::json::Value g =
        nano::json::parse(nano::json::read_file("tests/data/sampling_golden.json"));
    for (const auto& c : g.at("cases").items()) {
        const std::string& name = c.at("name").as_string();
        const nano::SamplerOptions opts = {
            .temperature = static_cast<float>(c.at("temperature").as_double()),
            .top_k = c.at("top_k").as_int(),
            .top_p = static_cast<float>(c.at("top_p").as_double()),
            .seed = 0,
        };
        const std::vector<float> logits = floats(c.at("logits"));
        const std::vector<float> want = floats(c.at("probs"));
        const std::vector<float> got = probs_of(opts, logits);

        // The surviving candidate set must match HF exactly...
        int64_t kept = 0;
        bool same_support = true;
        for (size_t i = 0; i < want.size(); ++i) {
            same_support = same_support && ((got[i] > 0.0f) == (want[i] > 0.0f));
            kept += got[i] > 0.0f ? 1 : 0;
        }
        NANO_CHECK_MSG(same_support, "candidate set diverges from HF (%s)", name.c_str());
        NANO_CHECK_MSG(kept == c.at("kept").as_int(), "kept %lld tokens, HF kept %lld (%s)",
                       static_cast<long long>(kept),
                       static_cast<long long>(c.at("kept").as_int()), name.c_str());

        // ...and the renormalized probabilities to fp32 tolerance.
        float max_err = 0.0f;
        for (size_t i = 0; i < want.size(); ++i) {
            max_err = std::max(max_err, std::abs(got[i] - want[i]));
        }
        NANO_CHECK_MSG(max_err < 1e-6f, "probs diverge from HF by %g (%s)",
                       static_cast<double>(max_err), name.c_str());
    }
}

void check_greedy_is_argmax() {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 3.0f);
    nano::Sampler greedy({.temperature = 0.0f, .top_k = 0, .top_p = 1.0f, .seed = 7}, 64);
    for (int rep = 0; rep < 20; ++rep) {
        std::vector<float> logits(64);
        for (float& x : logits) {
            x = dist(gen);
        }
        NANO_CHECK(greedy.sample(logits) == nano::argmax(logits));
    }
}

void check_top_k_cut() {
    // Descending logits: top_k=3 must keep exactly ids {0,1,2}, with the
    // softmax renormalized over just those three.
    std::vector<float> logits;
    for (int i = 0; i < 10; ++i) {
        logits.push_back(static_cast<float>(9 - i));
    }
    const auto p = probs_of({.temperature = 1.0f, .top_k = 3, .top_p = 1.0f, .seed = 0},
                            logits);
    for (size_t i = 0; i < p.size(); ++i) {
        NANO_CHECK((p[i] > 0.0f) == (i < 3));
    }
    // softmax over logits {9,8,7} = softmax over {2,1,0}
    const float z = std::exp(2.0f) + std::exp(1.0f) + 1.0f;
    NANO_CHECK(std::abs(p[0] - std::exp(2.0f) / z) < 1e-6f);
    NANO_CHECK(std::abs(p[1] - std::exp(1.0f) / z) < 1e-6f);
    NANO_CHECK(std::abs(p[2] - 1.0f / z) < 1e-6f);
}

void check_top_p_crossing() {
    // p = [0.6, 0.3, 0.1]: cumulative-before is 0, 0.6, 0.9 — so the kept
    // count as top_p rises is 1 (crossing token included), 2, then 3.
    // Thresholds sit safely between cumulative values; fp32 log/exp rounding
    // can't flip them.
    const std::vector<float> logits = log_probs({0.6f, 0.3f, 0.1f});
    auto kept = [&](float top_p) {
        const auto p =
            probs_of({.temperature = 1.0f, .top_k = 0, .top_p = top_p, .seed = 0}, logits);
        int n = 0;
        double sum = 0.0;
        for (float x : p) {
            n += x > 0.0f ? 1 : 0;
            sum += static_cast<double>(x);
        }
        NANO_CHECK(std::abs(sum - 1.0) < 1e-6);  // renormalized
        return n;
    };
    NANO_CHECK(kept(0.05f) == 1);  // always at least one token
    NANO_CHECK(kept(0.50f) == 1);
    NANO_CHECK(kept(0.65f) == 2);
    NANO_CHECK(kept(0.95f) == 3);

    // The kept-2 case renormalizes to [2/3, 1/3].
    const auto p =
        probs_of({.temperature = 1.0f, .top_k = 0, .top_p = 0.65f, .seed = 0}, logits);
    NANO_CHECK(std::abs(p[0] - 2.0f / 3.0f) < 1e-6f);
    NANO_CHECK(std::abs(p[1] - 1.0f / 3.0f) < 1e-6f);
}

void check_tie_break() {
    // Four equal logits, top_k=2: our documented tie rule keeps the lower
    // token ids. (HF's unstable sort makes no promise here — this pins OUR
    // behavior so seeded runs can't silently change.)
    const std::vector<float> logits = {1.0f, 1.0f, 1.0f, 1.0f};
    const auto p = probs_of({.temperature = 1.0f, .top_k = 2, .top_p = 1.0f, .seed = 0},
                            logits);
    NANO_CHECK(p[0] > 0.0f && p[1] > 0.0f && p[2] == 0.0f && p[3] == 0.0f);
}

void check_determinism() {
    const nano::SamplerOptions opts = {
        .temperature = 0.9f, .top_k = 8, .top_p = 0.95f, .seed = 123};
    nano::SamplerOptions other_seed = opts;
    other_seed.seed = 124;

    // Same seed -> identical 64-draw sequence; different seed -> diverges.
    // Logits vary per step so this exercises the full pipeline, not one draw.
    auto run = [](const nano::SamplerOptions& o) {
        nano::Sampler s(o, 32);
        std::mt19937 gen(1);
        std::normal_distribution<float> dist(0.0f, 2.0f);
        std::vector<int32_t> out;
        for (int step = 0; step < 64; ++step) {
            std::vector<float> logits(32);
            for (float& x : logits) {
                x = dist(gen);
            }
            out.push_back(s.sample(logits));
        }
        return out;
    };
    const auto a = run(opts);
    const auto b = run(opts);
    const auto c = run(other_seed);
    NANO_CHECK(a == b);
    NANO_CHECK(a != c);
}

void check_draw_frequencies() {
    // 200k draws from p = [0.5, 0.3, 0.2]. Empirical frequency must sit
    // within 0.01 of each probability (~9 sigma — and the run is fully
    // deterministic under seed 99, so this cannot flake).
    const std::vector<float> logits = log_probs({0.5f, 0.3f, 0.2f});
    const std::vector<float> expect = {0.5f, 0.3f, 0.2f};
    nano::Sampler s({.temperature = 1.0f, .top_k = 0, .top_p = 1.0f, .seed = 99}, 3);
    std::vector<int64_t> count(3, 0);
    const int64_t draws = 200000;
    for (int64_t i = 0; i < draws; ++i) {
        ++count[static_cast<size_t>(s.sample(logits))];
    }
    for (size_t i = 0; i < 3; ++i) {
        const double freq =
            static_cast<double>(count[i]) / static_cast<double>(draws);
        NANO_CHECK_MSG(std::abs(freq - static_cast<double>(expect[i])) < 0.01,
                       "token %zu drawn at %.4f, expected %.2f", i, freq,
                       static_cast<double>(expect[i]));
    }

    // Filtered tokens are never drawn: top_k=3 over 10 logits, 1000 draws.
    std::vector<float> wide;
    for (int i = 0; i < 10; ++i) {
        wide.push_back(static_cast<float>(9 - i) * 0.5f);
    }
    nano::Sampler topk({.temperature = 1.0f, .top_k = 3, .top_p = 1.0f, .seed = 5}, 10);
    for (int i = 0; i < 1000; ++i) {
        NANO_CHECK_MSG(topk.sample(wide) < 3, "drew a token outside top-k");
        if (nano::testing::failures > 0) {
            break;  // don't spam 1000 failures
        }
    }
}

}  // namespace

int main() {
    check_golden_cases();
    check_greedy_is_argmax();
    check_top_k_cut();
    check_top_p_crossing();
    check_tie_break();
    check_determinism();
    check_draw_frequencies();
    return nano::testing::finish("test_sampling");
}
