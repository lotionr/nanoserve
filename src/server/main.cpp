// nanoserve CLI.
//
//   nanoserve inspect  <file.safetensors>            tensor table + totals
//   nanoserve tokenize <model_dir> <text>            encode/decode roundtrip
//   nanoserve generate <model_dir> -p <prompt>       greedy chat completion
//   nanoserve version
//
// `serve` arrives with the HTTP endpoint (see feature_list.json).
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

#include "core/safetensors.hpp"
#include "model/chat_template.hpp"
#include "model/config.hpp"
#include "model/qwen2.hpp"
#include "model/tokenizer.hpp"

namespace {

constexpr std::string_view kVersion = "0.1.0";

int usage() {
    std::fprintf(stderr,
                 "nanoserve %.*s — a from-scratch LLM inference engine\n"
                 "\n"
                 "usage:\n"
                 "  nanoserve inspect  <file.safetensors>\n"
                 "  nanoserve tokenize <model_dir> <text>\n"
                 "  nanoserve generate <model_dir> -p <prompt> [-n max_tokens] [--greedy]\n"
                 "  nanoserve version\n",
                 static_cast<int>(kVersion.size()), kVersion.data());
    return 2;
}

std::string shape_string(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            s += ", ";
        }
        s += std::to_string(shape[i]);
    }
    return s + "]";
}

int cmd_inspect(const std::string& path) {
    const nano::SafeTensors st(path);

    std::printf("%-48s %-6s %-20s %12s\n", "tensor", "dtype", "shape", "params");
    std::printf("%s\n", std::string(90, '-').c_str());
    for (const auto& t : st.tensors()) {
        std::printf("%-48s %-6.*s %-20s %12lld\n", t.name.c_str(),
                    static_cast<int>(nano::dtype_name(t.dtype).size()),
                    nano::dtype_name(t.dtype).data(), shape_string(t.shape).c_str(),
                    static_cast<long long>(t.numel()));
    }
    std::printf("%s\n", std::string(90, '-').c_str());
    std::printf("%zu tensors, %lld parameters, %.1f MiB on disk\n", st.tensors().size(),
                static_cast<long long>(st.total_params()),
                static_cast<double>(st.file_size()) / (1024.0 * 1024.0));
    return 0;
}

int cmd_tokenize(const std::string& model_dir, const std::string& text) {
    const nano::Tokenizer tok = nano::Tokenizer::from_dir(model_dir);
    const std::vector<int32_t> ids = tok.encode(text);

    std::printf("%zu tokens:", ids.size());
    for (int32_t id : ids) {
        std::printf(" %d", id);
    }
    std::printf("\n");

    const std::string roundtrip = tok.decode(ids);
    std::printf("roundtrip: %s\n", roundtrip == text ? "exact" : "MISMATCH");
    return roundtrip == text ? 0 : 1;
}

int cmd_generate(const std::vector<std::string>& args) {
    // args: <model_dir> then flags. Greedy is the only decoding mode so far
    // (--greedy is accepted for forward compatibility; sampling is F022).
    std::string model_dir;
    std::string prompt;
    int64_t max_tokens = 32;
    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "-p" && i + 1 < args.size()) {
            prompt = args[++i];
        } else if (args[i] == "-n" && i + 1 < args.size()) {
            max_tokens = std::atoll(args[++i].c_str());
        } else if (args[i] == "--greedy") {
            // default; nothing to do
        } else if (model_dir.empty() && args[i][0] != '-') {
            model_dir = args[i];
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", args[i].c_str());
            return usage();
        }
    }
    if (model_dir.empty() || prompt.empty() || max_tokens <= 0) {
        return usage();
    }

    const nano::Tokenizer tok = nano::Tokenizer::from_dir(model_dir);
    const std::vector<nano::ChatMessage> messages = {{"user", prompt}};
    const std::vector<int32_t> prompt_ids =
        nano::apply_chat_template(tok, messages, /*add_generation_prompt=*/true);

    std::fprintf(stderr, "loading model (fp32)...\n");
    nano::Engine engine(model_dir);
    const std::vector<int32_t> stop_ids = {tok.special_id("<|im_end|>"),
                                           tok.special_id("<|endoftext|>")};
    const std::vector<int32_t> generated =
        nano::greedy_generate(engine, prompt_ids, max_tokens, stop_ids);

    std::printf("prompt tokens: %zu, generated tokens: %zu\n", prompt_ids.size(),
                generated.size());
    std::printf("ids:");
    for (int32_t id : generated) {
        std::printf(" %d", id);
    }
    std::printf("\n---\n%s\n", tok.decode(generated).c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    try {
        if (args.empty()) {
            return usage();
        }
        if (args[0] == "version") {
            std::printf("nanoserve %.*s\n", static_cast<int>(kVersion.size()), kVersion.data());
            return 0;
        }
        if (args[0] == "inspect" && args.size() == 2) {
            return cmd_inspect(args[1]);
        }
        if (args[0] == "tokenize" && args.size() == 3) {
            return cmd_tokenize(args[1], args[2]);
        }
        if (args[0] == "generate" && args.size() >= 2) {
            return cmd_generate(args);
        }
        return usage();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
