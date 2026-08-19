#include "model/tokenizer.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "core/json.hpp"

namespace nano {

namespace {

// ---------------------------------------------------------------------------
// UTF-8 <-> code points
// ---------------------------------------------------------------------------

void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
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

/// Decodes UTF-8 into code points, recording the byte offset where each code
/// point starts (plus one final end offset). Invalid bytes are treated as
/// single Latin-1 code points so tokenization never fails on odd input.
void decode_utf8(std::string_view s, std::vector<uint32_t>& cps, std::vector<size_t>& offsets) {
    cps.clear();
    offsets.clear();
    size_t i = 0;
    while (i < s.size()) {
        offsets.push_back(i);
        const uint8_t b0 = static_cast<uint8_t>(s[i]);
        uint32_t cp = b0;
        size_t len = 1;
        if (b0 >= 0xF0) {
            len = 4;
            cp = b0 & 0x07u;
        } else if (b0 >= 0xE0) {
            len = 3;
            cp = b0 & 0x0Fu;
        } else if (b0 >= 0xC0) {
            len = 2;
            cp = b0 & 0x1Fu;
        }
        if (len > 1) {
            if (i + len > s.size()) {
                len = 1;  // truncated sequence: fall back to the single byte
                cp = b0;
            } else {
                for (size_t k = 1; k < len; ++k) {
                    const uint8_t bk = static_cast<uint8_t>(s[i + k]);
                    if ((bk & 0xC0u) != 0x80u) {
                        len = 1;  // broken continuation: fall back
                        cp = b0;
                        break;
                    }
                    cp = (cp << 6) | (bk & 0x3Fu);
                }
            }
        }
        cps.push_back(cp);
        i += len;
    }
    offsets.push_back(s.size());
}

/// Reads one UTF-8 code point starting at s[i]; advances i.
uint32_t next_codepoint(std::string_view s, size_t& i) {
    std::vector<uint32_t> cps;
    std::vector<size_t> offs;
    decode_utf8(s.substr(i, 4), cps, offs);
    i += offs.size() > 1 ? offs[1] : 1;
    return cps.empty() ? 0u : cps[0];
}

// ---------------------------------------------------------------------------
// Character classes for the pretokenizer.
//
// ASCII is exact. Non-ASCII uses coarse block ranges covering the common
// scripts; this is enough for exact HF parity on the English golden corpus.
// Full Unicode category tables (exact \p{L}/\p{N} parity on CJK/emoji/mixed
// text) are tracked as feature F014.
// ---------------------------------------------------------------------------

bool is_whitespace(uint32_t c) {
    switch (c) {
        case '\t':
        case '\n':
        case 0x0B:
        case 0x0C:
        case '\r':
        case ' ':
        case 0x85:
        case 0xA0:
        case 0x1680:
        case 0x2028:
        case 0x2029:
        case 0x202F:
        case 0x205F:
        case 0x3000:
            return true;
        default:
            return (c >= 0x2000 && c <= 0x200A);
    }
}

bool is_letter(uint32_t c) {
    if (c < 0x80) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    }
    static constexpr std::pair<uint32_t, uint32_t> kRanges[] = {
        {0xAA, 0xAA},      {0xB5, 0xB5},      {0xBA, 0xBA},      {0xC0, 0xD6},
        {0xD8, 0xF6},      {0xF8, 0x2C1},     {0x370, 0x3FF},    {0x400, 0x52F},
        {0x531, 0x58F},    {0x5D0, 0x5EA},    {0x620, 0x64A},    {0x900, 0x97F},
        {0xE01, 0xE5B},    {0x1E00, 0x1FFF},  {0x3041, 0x30FF},  {0x3400, 0x4DBF},
        {0x4E00, 0x9FFF},  {0xA000, 0xA48F},  {0xAC00, 0xD7A3},  {0xF900, 0xFAFF},
        {0x1F00, 0x1FFF},
    };
    for (const auto& [lo, hi] : kRanges) {
        if (c >= lo && c <= hi) {
            return true;
        }
    }
    return false;
}

bool is_number(uint32_t c) {
    if (c < 0x80) {
        return c >= '0' && c <= '9';
    }
    static constexpr std::pair<uint32_t, uint32_t> kRanges[] = {
        {0x660, 0x669},    // Arabic-Indic
        {0x6F0, 0x6F9},    // Extended Arabic-Indic
        {0x966, 0x96F},    // Devanagari
        {0xFF10, 0xFF19},  // Fullwidth
    };
    for (const auto& [lo, hi] : kRanges) {
        if (c >= lo && c <= hi) {
            return true;
        }
    }
    return false;
}

uint32_t ascii_lower(uint32_t c) {
    return (c >= 'A' && c <= 'Z') ? c + 32 : c;
}

}  // namespace

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

Tokenizer Tokenizer::from_dir(const std::string& model_dir) {
    Tokenizer tok;

    // GPT-2 bytes-to-unicode: printable Latin-1 bytes map to themselves;
    // the remaining 68 bytes map to 0x100, 0x101, ... in byte order.
    {
        auto printable = [](uint32_t b) {
            return (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        };
        uint32_t next_free = 0x100;
        for (uint32_t b = 0; b < 256; ++b) {
            const uint32_t cp = printable(b) ? b : next_free++;
            append_utf8(tok.byte_to_unicode_[b], cp);
            tok.unicode_to_byte_.emplace(cp, static_cast<uint8_t>(b));
        }
    }

    // vocab.json: { "token-in-unicode-space": id, ... }
    {
        const std::string path = model_dir + "/vocab.json";
        const json::Value root = json::parse(json::read_file(path));
        int64_t max_id = -1;
        for (const auto& [token, id] : root.members()) {
            max_id = std::max(max_id, id.as_int());
        }
        tok.id_to_token_.resize(static_cast<size_t>(max_id + 1));
        tok.vocab_.reserve(root.members().size());
        for (const auto& [token, id] : root.members()) {
            const int32_t i = static_cast<int32_t>(id.as_int());
            tok.vocab_.emplace(token, i);
            tok.id_to_token_[static_cast<size_t>(i)] = token;
        }
    }

    // merges.txt: one "left right" pair per line, rank = line order.
    // A leading "#version" comment is standard.
    {
        const std::string text = json::read_file(model_dir + "/merges.txt");
        std::istringstream lines(text);
        std::string line;
        int32_t rank = 0;
        bool first_line = true;
        while (std::getline(lines, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            // Only the leading "#version: ..." header is a comment; "#" is a
            // perfectly valid merge symbol ("# $" is a real Qwen merge).
            const bool is_version_header = first_line && line.starts_with("#version");
            first_line = false;
            if (line.empty() || is_version_header) {
                continue;
            }
            tok.merge_rank_.emplace(line, rank++);
        }
        if (tok.merge_rank_.empty()) {
            throw std::runtime_error(model_dir + "/merges.txt: no merges found");
        }
    }

    // tokenizer_config.json: added_tokens_decoder maps id -> {"content": ...}.
    // These are the specials (<|endoftext|>, <|im_start|>, ...) that live
    // outside vocab.json.
    {
        const std::string path = model_dir + "/tokenizer_config.json";
        const json::Value root = json::parse(json::read_file(path));
        const json::Value* added = root.find("added_tokens_decoder");
        if (added != nullptr && added->is_object()) {
            for (const auto& [id_str, entry] : added->members()) {
                const int32_t id = static_cast<int32_t>(std::stoll(id_str));
                const std::string& content = entry.at("content").as_string();
                tok.specials_.emplace_back(content, id);
                tok.special_by_id_.emplace(id, content);
            }
        }
        // Longest-first so overlapping specials match greedily.
        std::sort(tok.specials_.begin(), tok.specials_.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    }

    return tok;
}

int32_t Tokenizer::special_id(std::string_view token) const {
    for (const auto& [text, id] : specials_) {
        if (text == token) {
            return id;
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Pretokenizer
//
// Mirrors Qwen2.5's pretokenizer regex, alternative by alternative, as a
// scanner over code points (std::regex has no \p{L}/\p{N}, and a hand-rolled
// scanner is both faster and explainable):
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)        contractions stay attached
//   | [^\r\n\p{L}\p{N}]?\p{L}+          a word, optionally with one prefix
//                                       char (usually the leading space)
//   | \p{N}                             each digit alone
//   |  ?[^\s\p{L}\p{N}]+[\r\n]*         punctuation run (+ trailing newlines)
//   | \s*[\r\n]+                        whitespace ending in newlines
//   | \s+(?!\S)                         trailing whitespace, minus one char
//                                       if more text follows
//   | \s+                               any other whitespace run
//
// Alternatives are tried in order at each position, exactly like the regex.
// ---------------------------------------------------------------------------

std::vector<std::pair<size_t, size_t>> Tokenizer::pretokenize(std::string_view text) const {
    std::vector<uint32_t> cp;
    std::vector<size_t> off;  // off[i] = byte offset of cp[i]; off[n] = text.size()
    decode_utf8(text, cp, off);
    const size_t n = cp.size();

    std::vector<std::pair<size_t, size_t>> chunks;
    auto emit = [&](size_t from, size_t to) { chunks.emplace_back(off[from], off[to]); };

    size_t i = 0;
    while (i < n) {
        // 1. Contractions: 's 't 're 've 'm 'll 'd (case-insensitive).
        if (cp[i] == '\'' && i + 1 < n) {
            const uint32_t a = ascii_lower(cp[i + 1]);
            const uint32_t b = i + 2 < n ? ascii_lower(cp[i + 2]) : 0;
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
                emit(i, i + 3);
                i += 3;
                continue;
            }
            if (a == 's' || a == 't' || a == 'm' || a == 'd') {
                emit(i, i + 2);
                i += 2;
                continue;
            }
        }

        // 2. Optional single prefix char + letters.
        if (is_letter(cp[i])) {
            size_t j = i + 1;
            while (j < n && is_letter(cp[j])) {
                ++j;
            }
            emit(i, j);
            i = j;
            continue;
        }
        if (cp[i] != '\r' && cp[i] != '\n' && !is_number(cp[i]) && i + 1 < n &&
            is_letter(cp[i + 1])) {
            size_t j = i + 2;
            while (j < n && is_letter(cp[j])) {
                ++j;
            }
            emit(i, j);
            i = j;
            continue;
        }

        // 3. A single digit.
        if (is_number(cp[i])) {
            emit(i, i + 1);
            ++i;
            continue;
        }

        // 4. Optional space + run of symbols + trailing newlines.
        {
            size_t j = i;
            if (cp[j] == ' ') {
                ++j;
            }
            size_t k = j;
            while (k < n && !is_whitespace(cp[k]) && !is_letter(cp[k]) && !is_number(cp[k])) {
                ++k;
            }
            if (k > j) {
                while (k < n && (cp[k] == '\r' || cp[k] == '\n')) {
                    ++k;
                }
                emit(i, k);
                i = k;
                continue;
            }
        }

        // Alternatives 5-7 all start with whitespace.
        size_t j = i;
        while (j < n && is_whitespace(cp[j])) {
            ++j;
        }
        if (j == i) {
            // Not whitespace and nothing above matched: lone unclassified
            // code point (rare; approximate Unicode tables). Emit it alone.
            emit(i, i + 1);
            ++i;
            continue;
        }

        // 5. Whitespace run ending in newlines: match through the last newline.
        size_t last_newline = n;  // sentinel for "none"
        for (size_t k = i; k < j; ++k) {
            if (cp[k] == '\r' || cp[k] == '\n') {
                last_newline = k;
            }
        }
        if (last_newline != n) {
            emit(i, last_newline + 1);
            i = last_newline + 1;
            continue;
        }

        // 6. Whitespace at end of text; otherwise leave one char to prefix
        //    the next token (the regex's (?!\S) backtrack).
        if (j == n) {
            emit(i, j);
            i = j;
        } else if (j - i >= 2) {
            emit(i, j - 1);
            i = j - 1;
        } else {
            // 7. Single whitespace char followed by text.
            emit(i, j);
            i = j;
        }
    }
    return chunks;
}

// ---------------------------------------------------------------------------
// BPE
// ---------------------------------------------------------------------------

void Tokenizer::bpe(std::string_view chunk, std::vector<int32_t>& out) const {
    // Map each raw byte into unicode space; these are the initial symbols.
    std::vector<std::string> word;
    word.reserve(chunk.size());
    for (char c : chunk) {
        word.push_back(byte_to_unicode_[static_cast<uint8_t>(c)]);
    }

    // Repeatedly find the adjacent pair with the lowest merge rank and merge
    // every occurrence of it (reference GPT-2 semantics). Chunks are short,
    // so the linear rescan is simple and fast enough.
    std::string key;
    while (word.size() >= 2) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best = word.size();
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            key.assign(word[i]);
            key.push_back(' ');
            key.append(word[i + 1]);
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best = i;
            }
        }
        if (best == word.size()) {
            break;  // no mergeable pair left
        }

        const std::string first = word[best];
        const std::string second = word[best + 1];
        std::vector<std::string> merged;
        merged.reserve(word.size());
        for (size_t i = 0; i < word.size();) {
            if (i + 1 < word.size() && word[i] == first && word[i + 1] == second) {
                merged.push_back(first + second);
                i += 2;
            } else {
                merged.push_back(std::move(word[i]));
                i += 1;
            }
        }
        word = std::move(merged);
    }

    for (const std::string& symbol : word) {
        auto it = vocab_.find(symbol);
        if (it == vocab_.end()) {
            // Cannot happen with a well-formed byte-level vocab (all 256
            // single bytes present; merges only produce vocab entries).
            throw std::runtime_error("tokenizer: symbol not in vocab: " + symbol);
        }
        out.push_back(it->second);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
    std::vector<int32_t> ids;
    for (const auto& [begin, end] : pretokenize(text)) {
        bpe(text.substr(begin, end - begin), ids);
    }
    return ids;
}

std::vector<int32_t> Tokenizer::encode_with_specials(std::string_view text) const {
    std::vector<int32_t> ids;
    size_t pos = 0;
    while (pos < text.size()) {
        // Earliest special occurrence; specials_ is longest-first, so ties at
        // the same position go to the longest match.
        size_t match_pos = std::string_view::npos;
        size_t match_len = 0;
        int32_t match_id = -1;
        for (const auto& [tok, id] : specials_) {
            const size_t p = text.find(tok, pos);
            if (p != std::string_view::npos && p < match_pos) {
                match_pos = p;
                match_len = tok.size();
                match_id = id;
            }
        }
        if (match_pos == std::string_view::npos) {
            break;
        }
        if (match_pos > pos) {
            for (const auto& [begin, end] : pretokenize(text.substr(pos, match_pos - pos))) {
                bpe(text.substr(pos + begin, end - begin), ids);
            }
        }
        ids.push_back(match_id);
        pos = match_pos + match_len;
    }
    if (pos < text.size()) {
        for (const auto& [begin, end] : pretokenize(text.substr(pos))) {
            bpe(text.substr(pos + begin, end - begin), ids);
        }
    }
    return ids;
}

std::string Tokenizer::decode(std::span<const int32_t> ids) const {
    std::string out;
    for (int32_t id : ids) {
        if (auto it = special_by_id_.find(id); it != special_by_id_.end()) {
            out += it->second;
            continue;
        }
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) {
            throw std::runtime_error("decode: id out of range: " + std::to_string(id));
        }
        const std::string& token = id_to_token_[static_cast<size_t>(id)];
        // Token is in unicode space; map each code point back to its byte.
        size_t i = 0;
        while (i < token.size()) {
            const uint32_t cp = next_codepoint(token, i);
            auto it = unicode_to_byte_.find(cp);
            if (it == unicode_to_byte_.end()) {
                throw std::runtime_error("decode: unmapped code point in token");
            }
            out.push_back(static_cast<char>(it->second));
        }
    }
    return out;
}

}  // namespace nano
