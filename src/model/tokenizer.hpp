// Byte-level BPE tokenizer (GPT-2 lineage, as used by Qwen2.5).
//
// Pipeline for encode():
//   text --pretokenize--> chunks --bytes-to-unicode--> symbols --BPE merges--> vocab ids
//
// 1. Pretokenize: split text into chunks with a hand-rolled scanner that
//    mirrors the model's pretokenizer regex (see tokenizer.cpp). BPE merges
//    never cross chunk boundaries — this is what keeps "hello," from fusing
//    the comma into the word.
// 2. Bytes-to-unicode: each raw byte maps to a printable code point
//    (GPT-2's trick so the vocab file never contains control bytes).
// 3. BPE: repeatedly merge the adjacent symbol pair with the lowest merge
//    rank (from merges.txt) until no listed pair remains, then look up ids.
//
// decode() inverts the mapping exactly; roundtrip is byte-identical.
//
// Loaded from a Hugging Face model dir: vocab.json + merges.txt +
// tokenizer_config.json (special tokens such as <|im_start|>).
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace nano {

class Tokenizer {
public:
    /// Loads vocab.json, merges.txt, and tokenizer_config.json from a model
    /// directory. Throws on missing/malformed files.
    static Tokenizer from_dir(const std::string& model_dir);

    /// Text -> ids. Special-token strings in the input are treated as plain
    /// text (encoded byte by byte), matching HF's add_special_tokens=False.
    std::vector<int32_t> encode(std::string_view text) const;

    /// Like encode(), but exact occurrences of special-token strings
    /// (e.g. "<|im_start|>") become their single reserved id.
    std::vector<int32_t> encode_with_specials(std::string_view text) const;

    /// Ids -> text. Exact inverse of encode(); special ids decode to their
    /// literal string.
    std::string decode(std::span<const int32_t> ids) const;

    size_t vocab_size() const { return id_to_token_.size(); }
    size_t num_merges() const { return merge_rank_.size(); }

    /// Id for a special token string, or -1 if not a known special.
    int32_t special_id(std::string_view token) const;

private:
    Tokenizer() = default;

    /// Byte offsets [begin, end) of each pretokenized chunk.
    std::vector<std::pair<size_t, size_t>> pretokenize(std::string_view text) const;

    /// BPE-encode one chunk, appending ids to `out`.
    void bpe(std::string_view chunk, std::vector<int32_t>& out) const;

    std::unordered_map<std::string, int32_t> vocab_;   // token (unicode space) -> id
    std::vector<std::string> id_to_token_;             // id -> token (unicode space)
    std::unordered_map<std::string, int32_t> merge_rank_;  // "left right" -> rank

    // GPT-2 byte<->unicode mapping. byte_to_unicode_[b] is the UTF-8 encoding
    // of the code point that byte b maps to.
    std::array<std::string, 256> byte_to_unicode_;
    std::unordered_map<uint32_t, uint8_t> unicode_to_byte_;

    // Special tokens, longest-first for greedy matching in encode_with_specials.
    std::vector<std::pair<std::string, int32_t>> specials_;
    std::unordered_map<int32_t, std::string> special_by_id_;
};

/// Incremental detokenizer for streaming output.
///
/// decode() is a pure per-token byte concatenation, so decoding token by
/// token yields exactly the batch result — except that byte-level BPE can
/// split one multi-byte UTF-8 character across two tokens (e.g. a 3-byte CJK
/// character whose first byte ends one token). push() holds such an
/// incomplete trailing sequence back until its continuation bytes arrive, so
/// every returned chunk is printable valid UTF-8 and the concatenation of
/// all chunks (plus flush()) is byte-identical to decode(all ids).
class StreamDecoder {
public:
    /// The tokenizer must outlive the decoder.
    explicit StreamDecoder(const Tokenizer& tokenizer) : tokenizer_(tokenizer) {}

    /// Feeds one generated token; returns the bytes now safe to print
    /// (possibly empty while a character is still incomplete).
    std::string push(int32_t id);

    /// Returns any still-held bytes. Only non-empty if the stream ended in
    /// the middle of a character (e.g. generation cut off by max_tokens).
    std::string flush();

private:
    const Tokenizer& tokenizer_;
    std::string pending_;  // bytes withheld because they end mid-character
};

}  // namespace nano
