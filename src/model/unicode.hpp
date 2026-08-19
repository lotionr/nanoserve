// NFC normalization (Unicode canonical composition).
//
// Qwen's tokenizer.json declares {"normalizer": {"type": "NFC"}} — HF runs
// NFC over the input before pretokenization, so e.g. "e" + COMBINING ACUTE
// becomes the single code point "é" before BPE ever sees it. Matching HF
// token ids exactly therefore requires matching this step too.
//
// Standard three-phase algorithm (UAX #15):
//   1. decompose  — full canonical decomposition (table + Hangul arithmetic)
//   2. reorder    — canonical ordering of combining marks (by ccc)
//   3. compose    — recombine starter+mark / starter+starter primary pairs
#pragma once

#include <string>
#include <string_view>

namespace nano::unicode {

/// Returns the NFC form of UTF-8 `text`. Invalid byte sequences pass through
/// unchanged (mirroring the tokenizer's lenient UTF-8 handling).
std::string nfc(std::string_view text);

}  // namespace nano::unicode
