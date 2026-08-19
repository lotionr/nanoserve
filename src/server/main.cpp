// nanoserve CLI.
//
//   nanoserve inspect  <file.safetensors>            tensor table + totals
//   nanoserve tokenize <model_dir> <text>            encode/decode roundtrip
//   nanoserve generate <model_dir> -p <prompt>       greedy chat completion
//   nanoserve version
//
// `serve` arrives with the HTTP endpoint (see feature_list.json).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

#include "core/ops.hpp"
#include "core/safetensors.hpp"
#include "model/chat_template.hpp"
#include "model/config.hpp"
#include "model/qwen2.hpp"
#include "model/sampler.hpp"
#include "model/tokenizer.hpp"
#include "model/unicode.hpp"

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
                 "                     [--temp T] [--top-k K] [--top-p P] [--seed S]\n"
                 "                     [--threads N] [--stats] [--int8]\n"
                 "                     [--backend cpu|metal]  (default: cpu)\n"
                 "  nanoserve quantize <model_dir> [-o out.safetensors]\n"
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

    // Tokenization NFC-normalizes first (tokenizer.json declares it), so the
    // roundtrip target is the NFC form — identical to the input except for
    // decomposed accents and other non-NFC sequences.
    const std::string roundtrip = tok.decode(ids);
    const std::string normalized = nano::unicode::nfc(text);
    std::printf("roundtrip: %s\n", roundtrip == normalized ? "exact" : "MISMATCH");
    return roundtrip == normalized ? 0 : 1;
}

/// Default location for the int8 weights, next to the fp32 checkpoint.
std::string int8_path(const std::string& model_dir) {
    return model_dir + "/model.int8.safetensors";
}

int cmd_quantize(const std::vector<std::string>& args) {
    std::string model_dir;
    std::string out_path;
    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "-o" && i + 1 < args.size()) {
            out_path = args[++i];
        } else if (model_dir.empty() && args[i][0] != '-') {
            model_dir = args[i];
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", args[i].c_str());
            return usage();
        }
    }
    if (model_dir.empty()) {
        return usage();
    }
    if (out_path.empty()) {
        out_path = int8_path(model_dir);
    }

    std::fprintf(stderr, "quantizing %s/model.safetensors -> %s ...\n",
                 model_dir.c_str(), out_path.c_str());
    const int n = nano::quantize_model_file(model_dir, out_path);
    const nano::SafeTensors check(out_path);  // re-open: proves it parses
    std::printf("quantized %d matrices to int8 (per-row scales); %zu tensors, %.1f MiB\n",
                n, check.tensors().size(),
                static_cast<double>(check.file_size()) / (1024.0 * 1024.0));
    return 0;
}

int cmd_generate(const std::vector<std::string>& args) {
    // args: <model_dir> then flags. Greedy (temperature 0) is the default;
    // --temp enables sampling, shaped by --top-k/--top-p, seeded by --seed.
    std::string model_dir;
    std::string prompt;
    int64_t max_tokens = 32;
    bool stats = false;
    bool int8 = false;
    std::string backend = "cpu";
    nano::SamplerOptions sampling = {
        .temperature = 0.0f, .top_k = 0, .top_p = 1.0f, .seed = 0};
    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "-p" && i + 1 < args.size()) {
            prompt = args[++i];
        } else if (args[i] == "-n" && i + 1 < args.size()) {
            max_tokens = std::atoll(args[++i].c_str());
        } else if (args[i] == "--greedy") {
            sampling.temperature = 0.0f;
        } else if (args[i] == "--temp" && i + 1 < args.size()) {
            sampling.temperature = std::strtof(args[++i].c_str(), nullptr);
        } else if (args[i] == "--top-k" && i + 1 < args.size()) {
            sampling.top_k = std::atoll(args[++i].c_str());
        } else if (args[i] == "--top-p" && i + 1 < args.size()) {
            sampling.top_p = std::strtof(args[++i].c_str(), nullptr);
        } else if (args[i] == "--seed" && i + 1 < args.size()) {
            sampling.seed = std::strtoull(args[++i].c_str(), nullptr, 10);
        } else if (args[i] == "--threads" && i + 1 < args.size()) {
            nano::ops::set_num_threads(std::atoi(args[++i].c_str()));
        } else if (args[i] == "--stats") {
            stats = true;
        } else if (args[i] == "--int8") {
            int8 = true;
        } else if (args[i] == "--backend" && i + 1 < args.size()) {
            backend = args[++i];
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

    std::string weights_file;  // empty = fp32 model.safetensors
    if (int8) {
        weights_file = int8_path(model_dir);
        if (std::FILE* f = std::fopen(weights_file.c_str(), "rb"); f != nullptr) {
            std::fclose(f);
        } else {
            std::fprintf(stderr,
                         "error: %s not found — run `nanoserve quantize %s` first\n",
                         weights_file.c_str(), model_dir.c_str());
            return 1;
        }
    }
    // Backend BEFORE the Engine: construction registers the weights with
    // the GPU (zero-copy on unified memory) only if metal is already
    // selected. CPU is the default; the README bench table records how the
    // two compare on this machine.
    if (backend == "metal") {
        if (!nano::ops::set_backend(nano::ops::Backend::metal)) {
            std::fprintf(stderr, "error: --backend metal: no usable Metal device\n");
            return 1;
        }
    } else if (backend != "cpu") {
        std::fprintf(stderr, "unknown backend: %s (cpu|metal)\n", backend.c_str());
        return usage();
    }
    std::fprintf(stderr, "loading model (%s, %s)...\n", int8 ? "int8" : "fp32",
                 backend.c_str());
    nano::Engine engine(model_dir, 2048, weights_file);
    const std::vector<int32_t> stop_ids = {tok.special_id("<|im_end|>"),
                                           tok.special_id("<|endoftext|>")};
    nano::Sampler sampler(sampling, engine.model().config.vocab_size);
    nano::GenerateStats gen_stats;

    // Tokens stream to stdout as they are generated (flushed per token so
    // pipes see them live). Stop tokens like <|im_end|> are not printed.
    nano::StreamDecoder stream(tok);
    const auto print_token = [&](int32_t id) {
        for (int32_t s : stop_ids) {
            if (id == s) {
                return;
            }
        }
        const std::string chunk = stream.push(id);
        std::fwrite(chunk.data(), 1, chunk.size(), stdout);
        std::fflush(stdout);
    };

    std::printf("---\n");
    const std::vector<int32_t> generated =
        nano::generate(engine, prompt_ids, max_tokens, stop_ids, sampler,
                       stats ? &gen_stats : nullptr, print_token);
    const std::string tail = stream.flush();
    std::fwrite(tail.data(), 1, tail.size(), stdout);
    std::printf("\n---\n");

    std::printf("prompt tokens: %zu, generated tokens: %zu\n", prompt_ids.size(),
                generated.size());
    std::printf("ids:");
    for (int32_t id : generated) {
        std::printf(" %d", id);
    }
    std::printf("\n");

    if (stats && !generated.empty()) {
        std::vector<double> steps = gen_stats.step_ms;
        std::sort(steps.begin(), steps.end());
        double sum = 0.0;
        for (double ms : steps) {
            sum += ms;
        }
        const double mean = steps.empty() ? 0.0 : sum / static_cast<double>(steps.size());
        const double median = steps.empty() ? 0.0 : steps[steps.size() / 2];
        std::printf("---\nttft (prefill %zu tokens): %.1f ms\n", prompt_ids.size(),
                    gen_stats.ttft_ms);
        std::printf("decode: %zu steps, mean %.1f ms/token, median %.1f ms/token "
                    "(%.2f tok/s)\n",
                    steps.size(), mean, median, mean > 0.0 ? 1000.0 / mean : 0.0);
    }
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
        if (args[0] == "quantize" && args.size() >= 2) {
            return cmd_quantize(args);
        }
        return usage();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
