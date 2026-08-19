// nanoserve CLI.
//
//   nanoserve inspect  <file.safetensors>       tensor table + totals
//   nanoserve tokenize <model_dir> <text>       encode/decode roundtrip
//   nanoserve version
//
// `generate` and `serve` arrive with the forward pass (see feature_list.json).
#include <cstdio>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

#include "core/safetensors.hpp"
#include "model/config.hpp"
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
        return usage();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
