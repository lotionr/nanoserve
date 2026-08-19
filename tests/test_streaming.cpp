// F023: streaming — per-token callback + UTF-8-safe incremental detokenizing.
//
// StreamDecoder checks run over every committed corpus (English, Unicode,
// specials, and the HF greedy continuations): pushing ids one at a time must
// reproduce the batch decode() byte-for-byte, and every intermediate chunk
// must be complete UTF-8 — the Unicode corpus makes sure the hold-back path
// (a character split across tokens) actually fires. The engine section then
// verifies the generate() callback: one call per token, in order, before the
// function returns its full result.
#include "model/tokenizer.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "model/qwen2.hpp"
#include "model/sampler.hpp"
#include "testing.hpp"

namespace {

std::vector<int32_t> ints(const nano::json::Value& v) {
    std::vector<int32_t> out;
    for (const auto& x : v.items()) {
        out.push_back(static_cast<int32_t>(x.as_int()));
    }
    return out;
}

/// Structurally valid UTF-8: every lead byte is followed by exactly its
/// continuation bytes, and no stray continuation bytes appear. (Printing a
/// chunk that fails this would garble a terminal mid-character.)
bool is_complete_utf8(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        const auto b = static_cast<unsigned char>(s[i]);
        size_t len = 0;
        if (b < 0x80) {
            len = 1;
        } else if ((b & 0xE0) == 0xC0) {
            len = 2;
        } else if ((b & 0xF0) == 0xE0) {
            len = 3;
        } else if ((b & 0xF8) == 0xF0) {
            len = 4;
        } else {
            return false;  // continuation byte with no lead, or invalid lead
        }
        if (i + len > s.size()) {
            return false;  // sequence runs off the end
        }
        for (size_t j = 1; j < len; ++j) {
            if ((static_cast<unsigned char>(s[i + j]) & 0xC0) != 0x80) {
                return false;
            }
        }
        i += len;
    }
    return true;
}

/// Streams `ids` one at a time; checks chunk validity and byte-exactness
/// against the batch decode. Returns how often bytes were held back, so the
/// caller can assert the interesting path was exercised.
int64_t check_stream(const nano::Tokenizer& tok, const std::vector<int32_t>& ids,
                     const char* label) {
    nano::StreamDecoder stream(tok);
    std::string streamed;
    int64_t holds = 0;
    bool chunks_valid = true;
    for (size_t i = 0; i < ids.size(); ++i) {
        const std::string chunk = stream.push(ids[i]);
        chunks_valid = chunks_valid && is_complete_utf8(chunk);
        // Held-back bytes show up as the streamed text lagging the batch
        // decode of the ids seen so far.
        streamed += chunk;
        std::span<const int32_t> so_far(ids.data(), i + 1);
        if (streamed != tok.decode(so_far)) {
            ++holds;
        }
    }
    streamed += stream.flush();
    NANO_CHECK_MSG(chunks_valid, "chunk with partial UTF-8 (%s)", label);
    NANO_CHECK_MSG(streamed == tok.decode(ids), "streamed text != batch decode (%s)",
                   label);
    return holds;
}

}  // namespace

int main() {
    const std::string dir = nano::testing::require_model_dir();
    const nano::Tokenizer tok = nano::Tokenizer::from_dir(dir);

    // --- StreamDecoder vs batch decode on every committed corpus ---
    const nano::json::Value g =
        nano::json::parse(nano::json::read_file("tests/data/tokenizer_golden.json"));
    int64_t unicode_holds = 0;
    for (const char* section : {"cases", "unicode_cases", "specials"}) {
        for (const auto& c : g.at(section).items()) {
            const int64_t holds =
                check_stream(tok, ints(c.at("ids")), c.at("text").as_string().c_str());
            if (std::string(section) == "unicode_cases") {
                unicode_holds += holds;
            }
        }
    }
    // The Unicode corpus must actually split characters across tokens —
    // otherwise the hold-back logic was never tested.
    NANO_CHECK_MSG(unicode_holds > 0,
                   "no multi-byte character ever crossed a token boundary");

    // Adversarial case: a 4-byte emoji fed byte by byte. Encoding "🚀" yields
    // one token per byte (byte-level BPE fallback), so the decoder must stay
    // silent for three pushes and emit the full character on the fourth.
    {
        const std::vector<int32_t> rocket = tok.encode("🚀");
        if (rocket.size() >= 2) {  // stays meaningful even if merges change
            nano::StreamDecoder stream(tok);
            std::string chunks;
            for (size_t i = 0; i + 1 < rocket.size(); ++i) {
                chunks += stream.push(rocket[i]);
            }
            NANO_CHECK_MSG(chunks.empty(), "emitted a partial emoji");
            chunks += stream.push(rocket.back());
            NANO_CHECK(chunks == "\xF0\x9F\x9A\x80");
            NANO_CHECK(stream.flush().empty());
        }
    }

    // flush() returns the incomplete tail when a stream is cut mid-character.
    {
        const std::vector<int32_t> rocket = tok.encode("🚀");
        if (rocket.size() >= 2) {
            nano::StreamDecoder stream(tok);
            std::string chunks = stream.push(rocket[0]);
            NANO_CHECK(chunks.empty());
            NANO_CHECK(!stream.flush().empty());  // the orphaned lead byte(s)
        }
    }

    // --- generate() callback ordering, against the greedy golden ---
    const nano::json::Value gg =
        nano::json::parse(nano::json::read_file("tests/data/generate_golden.json"));
    const auto& c = gg.at("cases").items().front();
    const std::vector<int32_t> prompt_ids = ints(c.at("prompt_ids"));

    nano::Engine engine(dir);
    nano::Sampler greedy({.temperature = 0.0f, .top_k = 0, .top_p = 1.0f, .seed = 0},
                         engine.model().config.vocab_size);
    const std::vector<int32_t> stop_ids = {tok.special_id("<|im_end|>"),
                                           tok.special_id("<|endoftext|>")};

    std::vector<int32_t> seen;
    nano::StreamDecoder stream(tok);
    std::string live_text;
    const std::vector<int32_t> got = nano::generate(
        engine, prompt_ids, 32, stop_ids, greedy, nullptr, [&](int32_t id) {
            seen.push_back(id);
            live_text += stream.push(id);
        });
    live_text += stream.flush();

    NANO_CHECK_MSG(seen == got, "callback tokens differ from returned tokens");
    NANO_CHECK_MSG(live_text == tok.decode(got),
                   "streamed generation text != batch decode");
    NANO_CHECK_MSG(got == ints(c.at("generated_ids")),
                   "generation diverged from the HF golden");

    return nano::testing::finish("test_streaming");
}
