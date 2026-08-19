// F019: greedy decoding end-to-end vs HF fp32 greedy continuations.
//
// The golden stores, for 3 chat prompts, the exact token ids HF `generate()`
// produced with do_sample=False (up to 32 new tokens, stopping at eos). The
// engine must reproduce every id — one wrong logit anywhere in 24 layers
// shows up here as a diverging token.
#include "model/qwen2.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "model/tokenizer.hpp"
#include "testing.hpp"

namespace {

std::vector<int32_t> ints(const nano::json::Value& v) {
    std::vector<int32_t> out;
    for (const auto& x : v.items()) {
        out.push_back(static_cast<int32_t>(x.as_int()));
    }
    return out;
}

}  // namespace

int main() {
    const std::string dir = nano::testing::require_model_dir();
    nano::Engine engine(dir);
    const nano::Tokenizer tok = nano::Tokenizer::from_dir(dir);

    // Same stop set as the model's generation_config.json.
    const std::vector<int32_t> stop_ids = {tok.special_id("<|im_end|>"),
                                           tok.special_id("<|endoftext|>")};

    const nano::json::Value g =
        nano::json::parse(nano::json::read_file("tests/data/generate_golden.json"));
    for (const auto& c : g.at("cases").items()) {
        const std::string& prompt = c.at("prompt").as_string();
        const std::vector<int32_t> prompt_ids = ints(c.at("prompt_ids"));
        const std::vector<int32_t> want = ints(c.at("generated_ids"));

        engine.reset();
        const std::vector<int32_t> got =
            nano::greedy_generate(engine, prompt_ids, 32, stop_ids);

        bool match = got.size() == want.size();
        for (size_t i = 0; match && i < got.size(); ++i) {
            match = got[i] == want[i];
        }
        if (!match) {
            std::fprintf(stderr, "  prompt: %s\n  got :", prompt.c_str());
            for (int32_t id : got) {
                std::fprintf(stderr, " %d", id);
            }
            std::fprintf(stderr, "\n  want:");
            for (int32_t id : want) {
                std::fprintf(stderr, " %d", id);
            }
            std::fprintf(stderr, "\n  got text: %s\n", tok.decode(got).c_str());
        }
        NANO_CHECK_MSG(match, "greedy continuation diverges from HF (prompt: %s)",
                       prompt.c_str());
    }

    return nano::testing::finish("test_generate");
}
