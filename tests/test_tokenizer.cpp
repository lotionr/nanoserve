#include "model/tokenizer.hpp"

#include <string>
#include <vector>

#include "core/json.hpp"
#include "model/chat_template.hpp"
#include "testing.hpp"

int main() {
    const std::string dir = nano::testing::require_model_dir();
    const nano::Tokenizer tok = nano::Tokenizer::from_dir(dir);

    // Qwen2.5 base vocab (before added specials) and BPE merge count.
    NANO_CHECK_MSG(tok.vocab_size() == 151643, "vocab_size = %zu", tok.vocab_size());
    NANO_CHECK(tok.num_merges() > 100000);

    const nano::json::Value golden =
        nano::json::parse(nano::json::read_file("tests/data/tokenizer_golden.json"));

    // F011: encode() matches HF `tokenizers` exactly on the golden corpus.
    // F012: decode() roundtrips byte-identically.
    // F014: same checks over the non-ASCII corpus (CJK, Cyrillic, emoji, ...)
    //       — exercises the generated \p{L}/\p{N} tables.
    for (const char* section : {"cases", "unicode_cases"}) {
        for (const auto& c : golden.at(section).items()) {
            const std::string& text = c.at("text").as_string();
            const auto& want = c.at("ids").items();

            const std::vector<int32_t> got = tok.encode(text);
            bool match = got.size() == want.size();
            if (match) {
                for (size_t i = 0; i < got.size(); ++i) {
                    match = match && got[i] == static_cast<int32_t>(want[i].as_int());
                }
            }
            if (!match) {
                std::fprintf(stderr, "  text: %s\n  got :", text.c_str());
                for (int32_t id : got) {
                    std::fprintf(stderr, " %d", id);
                }
                std::fprintf(stderr, "\n  want:");
                for (const auto& id : want) {
                    std::fprintf(stderr, " %lld", static_cast<long long>(id.as_int()));
                }
                std::fprintf(stderr, "\n");
            }
            NANO_CHECK_MSG(match, "encode mismatch on: %s", text.c_str());
            // Roundtrip target is HF's own decode: identical to `text` except
            // when NFC normalization changed the input before tokenization.
            const std::string& want_decoded = c.at("decoded").as_string();
            NANO_CHECK_MSG(tok.decode(got) == want_decoded, "roundtrip mismatch on: %s",
                           text.c_str());
        }
    }

    // Special tokens are known and atomic (never split into bytes).
    NANO_CHECK(tok.special_id("<|endoftext|>") == 151643);
    NANO_CHECK(tok.special_id("<|im_start|>") == 151644);
    NANO_CHECK(tok.special_id("<|im_end|>") == 151645);
    NANO_CHECK(tok.special_id("not_a_special") == -1);

    for (const auto& c : golden.at("specials").items()) {
        const std::string& text = c.at("text").as_string();
        const auto& want = c.at("ids").items();
        const std::vector<int32_t> got = tok.encode_with_specials(text);
        bool match = got.size() == want.size();
        if (match) {
            for (size_t i = 0; i < got.size(); ++i) {
                match = match && got[i] == static_cast<int32_t>(want[i].as_int());
            }
        }
        NANO_CHECK_MSG(match, "encode_with_specials mismatch on: %s", text.c_str());
        NANO_CHECK_MSG(tok.decode(got) == text, "specials roundtrip mismatch on: %s",
                       text.c_str());
    }

    // F013: chat-template helper matches HF apply_chat_template token ids.
    const nano::json::Value chat_golden =
        nano::json::parse(nano::json::read_file("tests/data/chat_golden.json"));
    for (const auto& c : chat_golden.at("cases").items()) {
        const std::string& name = c.at("name").as_string();

        std::vector<nano::ChatMessage> messages;
        for (const auto& m : c.at("messages").items()) {
            messages.push_back({m.at("role").as_string(), m.at("content").as_string()});
        }
        const bool gen_prompt = c.at("add_generation_prompt").as_bool();

        const std::vector<int32_t> got = nano::apply_chat_template(tok, messages, gen_prompt);
        const auto& want = c.at("ids").items();
        bool match = got.size() == want.size();
        if (match) {
            for (size_t i = 0; i < got.size(); ++i) {
                match = match && got[i] == static_cast<int32_t>(want[i].as_int());
            }
        }
        if (!match) {
            std::fprintf(stderr, "  got :");
            for (int32_t id : got) {
                std::fprintf(stderr, " %d", id);
            }
            std::fprintf(stderr, "\n  want:");
            for (const auto& id : want) {
                std::fprintf(stderr, " %lld", static_cast<long long>(id.as_int()));
            }
            std::fprintf(stderr, "\n");
        }
        NANO_CHECK_MSG(match, "chat template mismatch on: %s", name.c_str());
    }

    return nano::testing::finish("test_tokenizer");
}
