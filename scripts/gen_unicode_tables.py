#!/usr/bin/env python3
"""Generates src/model/unicode_tables.hpp: codepoint ranges for \\p{L} and \\p{N}.

The pretokenizer regex needs exact Unicode general categories (letters L*,
numbers N*). Hand-maintaining those tables is hopeless (~750 ranges), so —
llama.cpp style — this script derives them from Python's unicodedata and the
generated header is committed. Rerun only when bumping the Unicode version.

    .venv/bin/python scripts/gen_unicode_tables.py
"""

import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "src" / "model" / "unicode_tables.hpp"


def ranges_for(first_letters: str) -> list:
    """Merged [lo, hi] codepoint ranges whose category starts with one of
    `first_letters` (e.g. "L" matches Lu, Ll, Lt, Lm, Lo)."""
    out = []
    start = None
    for cp in range(0x110000):
        match = unicodedata.category(chr(cp))[0] in first_letters
        if match and start is None:
            start = cp
        elif not match and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def emit(name: str, ranges: list) -> str:
    lines = [f"inline constexpr CodepointRange {name}[] = {{"]
    row = []
    for lo, hi in ranges:
        row.append(f"{{0x{lo:X}, 0x{hi:X}}},")
        if len(row) == 4:
            lines.append("    " + " ".join(row))
            row = []
    if row:
        lines.append("    " + " ".join(row))
    lines.append("};")
    return "\n".join(lines)


HANGUL_FIRST, HANGUL_LAST = 0xAC00, 0xD7A3  # algorithmic, excluded from tables


def canonical_decomposition(cp: int) -> list:
    """Single-level canonical decomposition (no compatibility <tags>)."""
    d = unicodedata.decomposition(chr(cp))
    if not d or d.startswith("<"):
        return []
    return [int(x, 16) for x in d.split()]


def nfc_tables():
    """Tables for NFC normalization (the tokenizer.json normalizer).

    - ccc: nonzero canonical combining classes (for canonical ordering)
    - decomp: FULL canonical decompositions (recursively expanded offline so
      the C++ side decomposes in one lookup)
    - pairs: primary composites (a, b) -> composed. Derived empirically: a
      length-2 canonical decomposition is a primary composite iff NFC of the
      decomposed pair re-composes to the original — this sidesteps needing
      the composition-exclusions list.
    """
    ccc = [(cp, unicodedata.combining(chr(cp)))
           for cp in range(0x110000) if unicodedata.combining(chr(cp))]

    decomp = {}
    for cp in range(0x110000):
        if HANGUL_FIRST <= cp <= HANGUL_LAST:
            continue
        if canonical_decomposition(cp):
            # full expansion: NFD of the single codepoint
            decomp[cp] = [ord(c) for c in unicodedata.normalize("NFD", chr(cp))]

    pairs = []
    for cp in range(0x110000):
        if HANGUL_FIRST <= cp <= HANGUL_LAST:
            continue
        d = canonical_decomposition(cp)
        if len(d) == 2 and unicodedata.normalize("NFC", chr(d[0]) + chr(d[1])) == chr(cp):
            pairs.append((d[0], d[1], cp))
    pairs.sort()
    return ccc, decomp, pairs


def emit_nfc(ccc, decomp, pairs) -> str:
    parts = []

    lines = ["inline constexpr CombiningClass kCombiningClasses[] = {"]
    row = []
    for cp, cls in ccc:
        row.append(f"{{0x{cp:X}, {cls}}},")
        if len(row) == 6:
            lines.append("    " + " ".join(row))
            row = []
    if row:
        lines.append("    " + " ".join(row))
    lines.append("};")
    parts.append("\n".join(lines))

    # decompositions: sorted index into a flat data array
    data = []
    lines = ["inline constexpr Decomposition kDecompositions[] = {"]
    row = []
    for cp in sorted(decomp):
        seq = decomp[cp]
        row.append(f"{{0x{cp:X}, {len(data)}, {len(seq)}}},")
        data.extend(seq)
        if len(row) == 4:
            lines.append("    " + " ".join(row))
            row = []
    if row:
        lines.append("    " + " ".join(row))
    lines.append("};")
    parts.append("\n".join(lines))

    lines = ["inline constexpr uint32_t kDecompositionData[] = {"]
    row = []
    for cp in data:
        row.append(f"0x{cp:X},")
        if len(row) == 10:
            lines.append("    " + " ".join(row))
            row = []
    if row:
        lines.append("    " + " ".join(row))
    lines.append("};")
    parts.append("\n".join(lines))

    lines = ["inline constexpr CompositionPair kCompositionPairs[] = {"]
    row = []
    for a, b, c in pairs:
        row.append(f"{{0x{a:X}, 0x{b:X}, 0x{c:X}}},")
        if len(row) == 4:
            lines.append("    " + " ".join(row))
            row = []
    if row:
        lines.append("    " + " ".join(row))
    lines.append("};")
    parts.append("\n".join(lines))

    return "\n\n".join(parts)


def main() -> None:
    letters = ranges_for("L")
    numbers = ranges_for("N")
    ccc, decomp, pairs = nfc_tables()
    py = ".".join(map(str, sys.version_info[:2]))
    header = f"""\
// GENERATED FILE — do not edit by hand.
// Regenerate with: .venv/bin/python scripts/gen_unicode_tables.py
//
// Unicode data backing the tokenizer. Source: Python {py} unicodedata
// (Unicode {unicodedata.unidata_version}). All tables sorted for binary search.
//
//  - kLetterRanges / kNumberRanges: the pretokenizer's \\p{{L}} / \\p{{N}}
//    ({len(letters)} / {len(numbers)} ranges)
//  - NFC tables (the tokenizer.json normalizer): {len(ccc)} combining
//    classes, {len(decomp)} full canonical decompositions, {len(pairs)}
//    primary composition pairs (Hangul is algorithmic, not tabled)
#pragma once

#include <cstdint>

namespace nano::unicode {{

struct CodepointRange {{
    uint32_t lo;
    uint32_t hi;
}};

struct CombiningClass {{
    uint32_t cp;
    uint8_t ccc;
}};

/// `len` codepoints starting at kDecompositionData[offset].
struct Decomposition {{
    uint32_t cp;
    uint32_t offset;
    uint32_t len;
}};

struct CompositionPair {{
    uint32_t first;
    uint32_t second;
    uint32_t composed;
}};

{emit("kLetterRanges", letters)}

{emit("kNumberRanges", numbers)}

{emit_nfc(ccc, decomp, pairs)}

}}  // namespace nano::unicode
"""
    DEST.write_text(header)
    print(f"wrote {DEST} ({len(letters)}L/{len(numbers)}N ranges, "
          f"{len(ccc)} ccc, {len(decomp)} decomps, {len(pairs)} pairs)")


if __name__ == "__main__":
    main()
