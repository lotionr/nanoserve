// Qwen2.5 chat template: turns a list of chat messages into prompt token ids.
//
// The reference template lives in tokenizer_config.json as a jinja program;
// this helper re-implements its no-tools branch in plain C++:
//
//   <|im_start|>system\n{system or default}<|im_end|>\n
//   <|im_start|>{role}\n{content}<|im_end|>\n        (per non-system message)
//   <|im_start|>assistant\n                          (if add_generation_prompt)
//
// We render the prompt as a string first and then tokenize it with specials
// enabled — exactly what HF's apply_chat_template does — so the golden test
// can compare token ids directly against the HF reference.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "model/tokenizer.hpp"

namespace nano {

struct ChatMessage {
    std::string role;     // "system", "user", or "assistant"
    std::string content;  // raw text, rendered verbatim into the template
};

/// Renders messages into the Qwen2.5 prompt string (no tokenization).
/// A leading system message replaces the default Qwen system prompt.
std::string render_chat_template(std::span<const ChatMessage> messages,
                                 bool add_generation_prompt);

/// render_chat_template + encode_with_specials: the ids to feed the model.
std::vector<int32_t> apply_chat_template(const Tokenizer& tokenizer,
                                         std::span<const ChatMessage> messages,
                                         bool add_generation_prompt);

}  // namespace nano
