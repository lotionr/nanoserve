#include "model/chat_template.hpp"

namespace nano {

namespace {

constexpr const char* kDefaultSystem =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

}  // namespace

std::string render_chat_template(std::span<const ChatMessage> messages,
                                 bool add_generation_prompt) {
    std::string out;

    // The template always opens with a system block. A leading system message
    // supplies the text; otherwise Qwen's default system prompt is used.
    const bool has_system = !messages.empty() && messages.front().role == "system";
    out += "<|im_start|>system\n";
    out += has_system ? messages.front().content : kDefaultSystem;
    out += "<|im_end|>\n";

    // Every message after that leading system block is emitted verbatim.
    for (size_t i = has_system ? 1 : 0; i < messages.size(); ++i) {
        out += "<|im_start|>";
        out += messages[i].role;
        out += "\n";
        out += messages[i].content;
        out += "<|im_end|>\n";
    }

    // Open the assistant turn the model is supposed to complete.
    if (add_generation_prompt) {
        out += "<|im_start|>assistant\n";
    }
    return out;
}

std::vector<int32_t> apply_chat_template(const Tokenizer& tokenizer,
                                         std::span<const ChatMessage> messages,
                                         bool add_generation_prompt) {
    const std::string prompt = render_chat_template(messages, add_generation_prompt);
    return tokenizer.encode_with_specials(prompt);
}

}  // namespace nano
