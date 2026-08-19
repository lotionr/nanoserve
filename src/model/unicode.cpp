#include "model/unicode.hpp"

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

#include "model/unicode_tables.hpp"

namespace nano::unicode {

namespace {

// Invalid UTF-8 bytes travel through normalization untouched. They are
// represented as sentinel "code points" above the Unicode range (0x110000 +
// byte) so the pipeline stays a flat vector; ccc/decompose/compose all treat
// them as inert, and encoding turns them back into the original byte.
constexpr uint32_t kByteSentinel = 0x110000;

uint8_t combining_class(uint32_t cp) {
    const auto begin = std::begin(kCombiningClasses);
    const auto end = std::end(kCombiningClasses);
    auto it = std::lower_bound(begin, end, cp,
                               [](const CombiningClass& e, uint32_t c) { return e.cp < c; });
    return (it != end && it->cp == cp) ? it->ccc : 0;
}

std::span<const uint32_t> decomposition(uint32_t cp) {
    const auto begin = std::begin(kDecompositions);
    const auto end = std::end(kDecompositions);
    auto it = std::lower_bound(begin, end, cp,
                               [](const Decomposition& e, uint32_t c) { return e.cp < c; });
    if (it == end || it->cp != cp) {
        return {};
    }
    return {kDecompositionData + it->offset, it->len};
}

// Hangul syllables decompose and compose arithmetically (UAX #15 §3.12).
constexpr uint32_t kHangulSBase = 0xAC00, kHangulLBase = 0x1100;
constexpr uint32_t kHangulVBase = 0x1161, kHangulTBase = 0x11A7;
constexpr uint32_t kHangulLCount = 19, kHangulVCount = 21, kHangulTCount = 28;
constexpr uint32_t kHangulNCount = kHangulVCount * kHangulTCount;  // 588
constexpr uint32_t kHangulSCount = kHangulLCount * kHangulNCount;  // 11172

bool is_hangul_syllable(uint32_t cp) {
    return cp >= kHangulSBase && cp < kHangulSBase + kHangulSCount;
}

void decompose_into(uint32_t cp, std::vector<uint32_t>& out) {
    if (is_hangul_syllable(cp)) {
        const uint32_t s = cp - kHangulSBase;
        out.push_back(kHangulLBase + s / kHangulNCount);
        out.push_back(kHangulVBase + (s % kHangulNCount) / kHangulTCount);
        if (s % kHangulTCount != 0) {
            out.push_back(kHangulTBase + s % kHangulTCount);
        }
        return;
    }
    const std::span<const uint32_t> d = decomposition(cp);
    if (d.empty()) {
        out.push_back(cp);
    } else {
        out.insert(out.end(), d.begin(), d.end());  // table is pre-expanded
    }
}

/// Primary composite of (a, b), or 0 if the pair doesn't compose.
uint32_t compose_pair(uint32_t a, uint32_t b) {
    // Hangul L+V -> LV and LV+T -> LVT.
    if (a >= kHangulLBase && a < kHangulLBase + kHangulLCount && b >= kHangulVBase &&
        b < kHangulVBase + kHangulVCount) {
        return kHangulSBase + ((a - kHangulLBase) * kHangulVCount + (b - kHangulVBase)) *
                                  kHangulTCount;
    }
    if (is_hangul_syllable(a) && (a - kHangulSBase) % kHangulTCount == 0 &&
        b > kHangulTBase && b < kHangulTBase + kHangulTCount) {
        return a + (b - kHangulTBase);
    }
    const auto begin = std::begin(kCompositionPairs);
    const auto end = std::end(kCompositionPairs);
    auto it = std::lower_bound(begin, end, std::pair{a, b},
                               [](const CompositionPair& e, const std::pair<uint32_t, uint32_t>& k) {
                                   return e.first != k.first ? e.first < k.first
                                                             : e.second < k.second;
                               });
    return (it != end && it->first == a && it->second == b) ? it->composed : 0;
}

/// Strict UTF-8 decode of one sequence at s[i]. Returns the code point and
/// advances i, or returns a byte sentinel (advancing by 1) for anything
/// malformed: bad continuations, overlong forms, surrogates, > U+10FFFF.
uint32_t next_codepoint_strict(std::string_view s, size_t& i) {
    const uint8_t b0 = static_cast<uint8_t>(s[i]);
    size_t len = 0;
    uint32_t cp = 0;
    if (b0 < 0x80) {
        ++i;
        return b0;
    } else if ((b0 & 0xE0u) == 0xC0u) {
        len = 2;
        cp = b0 & 0x1Fu;
    } else if ((b0 & 0xF0u) == 0xE0u) {
        len = 3;
        cp = b0 & 0x0Fu;
    } else if ((b0 & 0xF8u) == 0xF0u) {
        len = 4;
        cp = b0 & 0x07u;
    } else {
        ++i;
        return kByteSentinel + b0;
    }
    if (i + len > s.size()) {
        ++i;
        return kByteSentinel + b0;
    }
    for (size_t k = 1; k < len; ++k) {
        const uint8_t bk = static_cast<uint8_t>(s[i + k]);
        if ((bk & 0xC0u) != 0x80u) {
            ++i;
            return kByteSentinel + b0;
        }
        cp = (cp << 6) | (bk & 0x3Fu);
    }
    const bool overlong = (len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) ||
                          (len == 4 && cp < 0x10000);
    const bool surrogate = cp >= 0xD800 && cp <= 0xDFFF;
    if (overlong || surrogate || cp > 0x10FFFF) {
        ++i;
        return kByteSentinel + b0;
    }
    i += len;
    return cp;
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp >= kByteSentinel) {  // invalid input byte: emit verbatim
        out.push_back(static_cast<char>(cp - kByteSentinel));
    } else if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

uint8_t ccc_of(uint32_t cp) {
    return cp >= kByteSentinel ? 0 : combining_class(cp);
}

}  // namespace

std::string nfc(std::string_view text) {
    // Phase 1: decode + full canonical decomposition.
    std::vector<uint32_t> cps;
    cps.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        const uint32_t cp = next_codepoint_strict(text, i);
        if (cp >= kByteSentinel) {
            cps.push_back(cp);
        } else {
            decompose_into(cp, cps);
        }
    }

    // Phase 2: canonical ordering — each maximal run of nonzero-ccc marks is
    // stably sorted by ccc (stability preserves the order of equal marks).
    size_t run = 0;
    while (run < cps.size()) {
        if (ccc_of(cps[run]) == 0) {
            ++run;
            continue;
        }
        size_t run_end = run;
        while (run_end < cps.size() && ccc_of(cps[run_end]) != 0) {
            ++run_end;
        }
        std::stable_sort(cps.begin() + static_cast<int64_t>(run),
                         cps.begin() + static_cast<int64_t>(run_end),
                         [](uint32_t a, uint32_t b) { return ccc_of(a) < ccc_of(b); });
        run = run_end;
    }

    // Phase 3: canonical composition (UAX #15): a character combines with the
    // last starter when nothing of equal-or-higher ccc blocks the way.
    std::vector<uint32_t> out;
    out.reserve(cps.size());
    int64_t last_starter = -1;
    for (const uint32_t cp : cps) {
        const uint8_t cc = ccc_of(cp);
        if (last_starter >= 0) {
            const bool adjacent =
                static_cast<int64_t>(out.size()) - 1 == last_starter;
            const bool blocked = !adjacent && ccc_of(out.back()) >= cc;
            if (!blocked && cp < kByteSentinel && out[static_cast<size_t>(last_starter)] < kByteSentinel) {
                const uint32_t composed =
                    compose_pair(out[static_cast<size_t>(last_starter)], cp);
                if (composed != 0) {
                    out[static_cast<size_t>(last_starter)] = composed;
                    continue;
                }
            }
        }
        if (cc == 0) {
            last_starter = static_cast<int64_t>(out.size());
        }
        out.push_back(cp);
    }

    std::string result;
    result.reserve(text.size());
    for (const uint32_t cp : out) {
        append_utf8(result, cp);
    }
    return result;
}

}  // namespace nano::unicode
