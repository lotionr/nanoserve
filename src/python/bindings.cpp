// pybind11 bindings (F028-F029).
//
// The C++ module stays thin: an Engine that owns the nano::Engine plus its
// tokenizer, a GenerateResult value type, and a TokenStream iterator that
// runs ONE decode step per __next__ — streaming needs no threads or queues,
// the Python for-loop IS the decode loop. The ergonomic keyword API
// (generate(..., stream=True)) lives in python/nanoserve/__init__.py.
//
// The GIL is released around every forward pass, so a generation running in
// one thread doesn't freeze the rest of a Python process.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/metal.hpp"
#include "core/ops.hpp"
#include "model/chat_template.hpp"
#include "model/qwen2.hpp"
#include "model/sampler.hpp"
#include "model/tokenizer.hpp"

namespace py = pybind11;

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                     t0)
        .count();
}

/// What a finished (or in-flight, for streams) generation looks like from
/// Python. tokens_per_second covers decode steps only — the prefill cost is
/// reported separately as ttft_ms, the same split the CLI --stats prints.
struct GenerateResult {
    std::string text;                // stop tokens excluded
    std::vector<int32_t> token_ids;  // stop token included (matches HF/CLI)
    int64_t prompt_tokens = 0;       // chat-template ids fed as prefill
    double ttft_ms = 0.0;
    double tokens_per_second = 0.0;
};

class TokenStream;

class Engine {
public:
    Engine(const std::string& model_dir, int64_t max_seq, bool int8)
        : engine_(model_dir, max_seq,
                  int8 ? model_dir + "/model.int8.safetensors" : std::string()),
          tokenizer_(nano::Tokenizer::from_dir(model_dir)),
          stop_ids_{tokenizer_.special_id("<|im_end|>"),
                    tokenizer_.special_id("<|endoftext|>")} {}

    GenerateResult generate(const std::string& prompt, int64_t max_tokens,
                            float temperature, int64_t top_k, float top_p,
                            uint64_t seed);

    // Defined after TokenStream so it can construct one.
    TokenStream stream(const std::string& prompt, int64_t max_tokens,
                       float temperature, int64_t top_k, float top_p, uint64_t seed);

    bool quantized() const { return engine_.model().quantized(); }

private:
    friend class TokenStream;

    /// Same prompt construction as the CLI: one user message through the
    /// Qwen chat template, with the generation prompt appended.
    std::vector<int32_t> prompt_ids(const std::string& prompt) const {
        const std::vector<nano::ChatMessage> messages = {{"user", prompt}};
        return nano::apply_chat_template(tokenizer_, messages,
                                         /*add_generation_prompt=*/true);
    }

    bool is_stop(int32_t id) const {
        return id == stop_ids_[0] || id == stop_ids_[1];
    }

    static void check_max_tokens(int64_t max_tokens) {
        if (max_tokens <= 0) {
            throw std::invalid_argument("max_tokens must be positive");
        }
    }

    nano::Engine engine_;
    nano::Tokenizer tokenizer_;
    std::vector<int32_t> stop_ids_;
};

/// Iterator over one generation: each __next__ runs forward passes until a
/// printable text chunk exists (StreamDecoder may withhold bytes while a
/// multi-byte character is split across tokens). Stats accumulate as it
/// goes and are final once the loop ends.
///
/// The stream drives the engine's one KV cache, so only the most recently
/// created stream of an Engine may be consumed; creating any new generation
/// resets the cache.
class TokenStream {
public:
    TokenStream(Engine& owner, std::vector<int32_t> prompt_ids, int64_t max_tokens,
                const nano::SamplerOptions& opts)
        : owner_(owner),
          prompt_ids_(std::move(prompt_ids)),
          max_tokens_(max_tokens),
          sampler_(opts, owner.engine_.model().config.vocab_size),
          decoder_(owner.tokenizer_) {}

    std::string next() {
        while (true) {
            if (done_) {
                if (!flushed_) {
                    flushed_ = true;
                    const std::string tail = decoder_.flush();
                    text_ += tail;
                    if (!tail.empty()) {
                        return tail;
                    }
                }
                throw py::stop_iteration();
            }

            int32_t token = 0;
            {
                // The forward pass is the expensive part — let other Python
                // threads run while C++ crunches.
                py::gil_scoped_release release;
                const auto t0 = std::chrono::steady_clock::now();
                std::span<const float> logits;
                if (produced_ == 0) {
                    owner_.engine_.reset();
                    logits = owner_.engine_.forward(prompt_ids_);
                } else {
                    const int32_t fed[] = {last_token_};
                    logits = owner_.engine_.forward(fed);
                }
                token = sampler_.sample(logits);
                const double ms = ms_since(t0);
                if (produced_ == 0) {
                    ttft_ms_ = ms;
                } else {
                    decode_ms_ += ms;
                    ++decode_steps_;
                }
            }

            ++produced_;
            last_token_ = token;
            token_ids_.push_back(token);
            if (owner_.is_stop(token) || produced_ >= max_tokens_) {
                done_ = true;
            }
            if (!owner_.is_stop(token)) {
                const std::string chunk = decoder_.push(token);
                text_ += chunk;
                if (!chunk.empty()) {
                    return chunk;
                }
            }
            // Withheld bytes or a stop token: run another step (or fall
            // into the flush/stop path) instead of yielding "".
        }
    }

    const std::string& text() const { return text_; }
    const std::vector<int32_t>& token_ids() const { return token_ids_; }
    int64_t prompt_tokens() const { return static_cast<int64_t>(prompt_ids_.size()); }
    double ttft_ms() const { return ttft_ms_; }
    double tokens_per_second() const {
        return decode_ms_ > 0.0 ? 1000.0 * static_cast<double>(decode_steps_) / decode_ms_
                                : 0.0;
    }

private:
    Engine& owner_;
    std::vector<int32_t> prompt_ids_;
    int64_t max_tokens_;
    nano::Sampler sampler_;
    nano::StreamDecoder decoder_;

    int64_t produced_ = 0;
    int32_t last_token_ = 0;
    bool done_ = false;
    bool flushed_ = false;

    std::string text_;
    std::vector<int32_t> token_ids_;
    double ttft_ms_ = 0.0;
    double decode_ms_ = 0.0;
    int64_t decode_steps_ = 0;
};

GenerateResult Engine::generate(const std::string& prompt, int64_t max_tokens,
                                float temperature, int64_t top_k, float top_p,
                                uint64_t seed) {
    check_max_tokens(max_tokens);
    const std::vector<int32_t> ids = prompt_ids(prompt);
    nano::Sampler sampler({temperature, top_k, top_p, seed},
                          engine_.model().config.vocab_size);
    nano::GenerateStats stats;
    std::vector<int32_t> out;
    {
        py::gil_scoped_release release;
        engine_.reset();
        out = nano::generate(engine_, ids, max_tokens, stop_ids_, sampler, &stats);
    }

    GenerateResult r;
    r.token_ids = out;
    r.prompt_tokens = static_cast<int64_t>(ids.size());
    std::vector<int32_t> printable;
    printable.reserve(out.size());
    for (int32_t id : out) {
        if (!is_stop(id)) {
            printable.push_back(id);
        }
    }
    r.text = tokenizer_.decode(printable);
    r.ttft_ms = stats.ttft_ms;
    double decode_ms = 0.0;
    for (double ms : stats.step_ms) {
        decode_ms += ms;
    }
    r.tokens_per_second =
        decode_ms > 0.0
            ? 1000.0 * static_cast<double>(stats.step_ms.size()) / decode_ms
            : 0.0;
    return r;
}

TokenStream Engine::stream(const std::string& prompt, int64_t max_tokens,
                           float temperature, int64_t top_k, float top_p,
                           uint64_t seed) {
    check_max_tokens(max_tokens);
    return TokenStream(*this, prompt_ids(prompt), max_tokens,
                       {temperature, top_k, top_p, seed});
}

}  // namespace

PYBIND11_MODULE(_nanoserve, m) {
    m.doc() = "nanoserve: from-scratch LLM inference engine (C++ core)";
    m.attr("__version__") = "0.1.0";

    m.def("set_num_threads", &nano::ops::set_num_threads, py::arg("n"),
          "Thread count for matmuls: 1 = serial, 0 = one per hardware thread.");

    // Backend selection (F033). Raises on an impossible request instead of
    // returning False: a benchmark that silently measured the CPU while
    // labeling it 'metal' would be worse than a crash. Select the backend
    // BEFORE constructing an Engine (weights register with the GPU at
    // construction).
    m.def(
        "set_backend",
        [](const std::string& name) {
            if (name == "cpu") {
                nano::ops::set_backend(nano::ops::Backend::cpu);
            } else if (name == "metal") {
                if (!nano::ops::set_backend(nano::ops::Backend::metal)) {
                    throw std::runtime_error("no usable Metal device");
                }
            } else {
                throw std::invalid_argument("backend must be 'cpu' or 'metal'");
            }
        },
        py::arg("name"),
        "Where matmuls run: 'cpu' (default) or 'metal' (M-series GPU).");
    m.def(
        "metal_available", [] { return nano::metal::available(); },
        "True if a Metal device exists and the GPU kernels compiled.");

    py::class_<GenerateResult>(m, "GenerateResult")
        .def_readonly("text", &GenerateResult::text)
        .def_readonly("token_ids", &GenerateResult::token_ids)
        .def_readonly("prompt_tokens", &GenerateResult::prompt_tokens)
        .def_readonly("ttft_ms", &GenerateResult::ttft_ms)
        .def_readonly("tokens_per_second", &GenerateResult::tokens_per_second)
        .def("__repr__", [](const GenerateResult& r) {
            return "GenerateResult(text=" + py::repr(py::str(r.text)).cast<std::string>() +
                   ", tokens=" + std::to_string(r.token_ids.size()) +
                   ", ttft_ms=" + std::to_string(r.ttft_ms) + ")";
        });

    py::class_<TokenStream>(m, "TokenStream")
        .def("__iter__", [](TokenStream& s) -> TokenStream& { return s; },
             py::keep_alive<0, 1>())
        .def("__next__", &TokenStream::next)
        .def_property_readonly("text", &TokenStream::text)
        .def_property_readonly("token_ids", &TokenStream::token_ids)
        .def_property_readonly("prompt_tokens", &TokenStream::prompt_tokens)
        .def_property_readonly("ttft_ms", &TokenStream::ttft_ms)
        .def_property_readonly("tokens_per_second", &TokenStream::tokens_per_second);

    py::class_<Engine>(m, "Engine")
        .def(py::init<const std::string&, int64_t, bool>(), py::arg("model_dir"),
             py::arg("max_seq") = 2048, py::arg("int8") = false,
             py::call_guard<py::gil_scoped_release>(),
             "Load the model. int8=True loads model.int8.safetensors "
             "(produce it with `nanoserve quantize`).")
        .def("generate", &Engine::generate, py::arg("prompt"),
             py::arg("max_tokens") = 32, py::arg("temperature") = 0.0f,
             py::arg("top_k") = 0, py::arg("top_p") = 1.0f, py::arg("seed") = 0,
             "Run a full generation; temperature 0 (default) is greedy.")
        .def("stream", &Engine::stream, py::arg("prompt"), py::arg("max_tokens") = 32,
             py::arg("temperature") = 0.0f, py::arg("top_k") = 0,
             py::arg("top_p") = 1.0f, py::arg("seed") = 0,
             py::keep_alive<0, 1>(),  // the stream keeps the engine alive
             "Like generate(), but returns an iterator of text chunks.")
        .def_property_readonly("quantized", &Engine::quantized);
}
