// POST /v1/completions (F035): the classic OpenAI completions schema, the
// subset an inference demo honestly implements:
//
//   request:  prompt (string, required), max_tokens, temperature, top_k,
//             top_p, seed, stream; n must be 1 if present. The prompt is
//             completed RAW — no chat template — matching the endpoint's
//             "continue this text" contract (the CLI's `generate` is the
//             chat-formatted interface).
//   response: text_completion object with one choice and usage counts;
//             completion_tokens counts every sampled token, including the
//             stop token that ends the text but is never shown in it.
//   stream:   Server-Sent Events, one chunk per detokenized piece the moment
//             it is sampled, a final empty chunk carrying finish_reason,
//             then `data: [DONE]`.
//
// The engine runs on the paged KV cache (F034): a server is exactly the
// workload paging exists for — memory tracks the live request, not max_seq.
// One request at a time end to end; continuous batching is F036.
#include "server/serve.hpp"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "core/ops.hpp"
#include "model/qwen2.hpp"
#include "model/sampler.hpp"
#include "model/tokenizer.hpp"
#include "server/http.hpp"

namespace nano {

namespace {

/// Client-caused failures: reported as HTTP 400 with the message, unlike
/// engine bugs, which the http layer turns into a 500.
struct BadRequest : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct CompletionRequest {
    std::string prompt;
    int64_t max_tokens = 16;
    SamplerOptions sampling;  // defaults: temperature 1.0, no top-k/top-p
    bool stream = false;
};

CompletionRequest parse_completion(const std::string& body) {
    json::Value doc;
    try {
        doc = json::parse(body);
    } catch (const std::exception& e) {
        throw BadRequest(std::string("body is not valid JSON: ") + e.what());
    }
    if (!doc.is_object()) {
        throw BadRequest("body must be a JSON object");
    }
    const json::Value* prompt = doc.find("prompt");
    if (prompt == nullptr || !prompt->is_string()) {
        throw BadRequest("\"prompt\" (string) is required");
    }
    CompletionRequest req;
    req.prompt = prompt->as_string();
    try {
        if (const json::Value* v = doc.find("max_tokens")) {
            req.max_tokens = v->as_int();
        }
        if (const json::Value* v = doc.find("temperature")) {
            req.sampling.temperature = static_cast<float>(v->as_double());
        }
        if (const json::Value* v = doc.find("top_k")) {
            req.sampling.top_k = v->as_int();
        }
        if (const json::Value* v = doc.find("top_p")) {
            req.sampling.top_p = static_cast<float>(v->as_double());
        }
        if (const json::Value* v = doc.find("seed")) {
            req.sampling.seed = static_cast<uint64_t>(v->as_int());
        }
        if (const json::Value* v = doc.find("stream")) {
            req.stream = v->as_bool();
        }
        if (const json::Value* v = doc.find("n"); v != nullptr && v->as_int() != 1) {
            throw BadRequest("only n=1 is supported");
        }
    } catch (const BadRequest&) {
        throw;
    } catch (const std::exception& e) {
        throw BadRequest(std::string("bad parameter type: ") + e.what());
    }
    if (req.max_tokens <= 0) {
        throw BadRequest("\"max_tokens\" must be positive");
    }
    if (req.sampling.temperature < 0.0f) {
        throw BadRequest("\"temperature\" must be >= 0");
    }
    return req;
}

/// OpenAI-shaped error body.
std::string error_json(std::string_view message) {
    return "{\"error\":{\"message\":" + json::quote(message) +
           ",\"type\":\"invalid_request_error\"}}";
}

/// The model, tokenizer, and endpoint logic behind /v1/completions.
class CompletionService {
public:
    CompletionService(const std::string& model_dir, const std::string& weights_file)
        : tok_(Tokenizer::from_dir(model_dir)),
          engine_(model_dir, /*max_seq=*/2048, weights_file, KvLayout::paged),
          stop_ids_{tok_.special_id("<|im_end|>"), tok_.special_id("<|endoftext|>")},
          model_name_(basename_of(model_dir)) {}

    void handle(const http::Request& req, http::ResponseWriter& out) {
        if (req.path != "/v1/completions") {
            out.send(404, "application/json", error_json("no such endpoint: " + req.path));
            return;
        }
        if (req.method != "POST") {
            out.send(405, "application/json", error_json("use POST"));
            return;
        }
        CompletionRequest completion;
        try {
            completion = parse_completion(req.body);
            complete(completion, out);
        } catch (const BadRequest& e) {
            out.send(400, "application/json", error_json(e.what()));
        }
    }

private:
    static std::string basename_of(std::string dir) {
        while (!dir.empty() && dir.back() == '/') {
            dir.pop_back();
        }
        const size_t slash = dir.rfind('/');
        return slash == std::string::npos ? dir : dir.substr(slash + 1);
    }

    bool is_stop(int32_t id) const {
        for (const int32_t s : stop_ids_) {
            if (id == s) {
                return true;
            }
        }
        return false;
    }

    /// The shared preamble of every response object for one completion.
    std::string object_prefix(const std::string& id, int64_t created) const {
        return "{\"id\":" + json::quote(id) +
               ",\"object\":\"text_completion\",\"created\":" +
               std::to_string(created) + ",\"model\":" + json::quote(model_name_);
    }

    static std::string choice_json(std::string_view text, const char* finish_reason) {
        std::string choice = "{\"index\":0,\"text\":" + json::quote(text) +
                             ",\"logprobs\":null,\"finish_reason\":";
        choice += finish_reason == nullptr
                      ? "null"
                      : "\"" + std::string(finish_reason) + "\"";
        return choice + "}";
    }

    void complete(const CompletionRequest& req, http::ResponseWriter& out) {
        const std::vector<int32_t> prompt_ids = tok_.encode(req.prompt);
        if (prompt_ids.empty()) {
            throw BadRequest("prompt produced no tokens");
        }
        if (static_cast<int64_t>(prompt_ids.size()) + req.max_tokens >
            engine_.max_seq()) {
            throw BadRequest("prompt + max_tokens exceeds the " +
                             std::to_string(engine_.max_seq()) + "-token context");
        }
        engine_.reset();  // paged: the previous request's pages go back
        Sampler sampler(req.sampling, engine_.model().config.vocab_size);
        const std::string id = "cmpl-" + std::to_string(++served_);
        const int64_t created = static_cast<int64_t>(std::time(nullptr));
        const std::string prefix = object_prefix(id, created);

        if (!req.stream) {
            std::vector<int32_t> ids =
                generate(engine_, prompt_ids, req.max_tokens, stop_ids_, sampler);
            const int64_t completion_tokens = static_cast<int64_t>(ids.size());
            const bool hit_stop = !ids.empty() && is_stop(ids.back());
            if (hit_stop) {
                ids.pop_back();  // the stop token ends, but never joins, the text
            }
            std::string body = prefix;
            body += ",\"choices\":[";
            body += choice_json(tok_.decode(ids), hit_stop ? "stop" : "length");
            body += "],\"usage\":{\"prompt_tokens\":" +
                    std::to_string(prompt_ids.size()) +
                    ",\"completion_tokens\":" + std::to_string(completion_tokens) +
                    ",\"total_tokens\":" +
                    std::to_string(static_cast<int64_t>(prompt_ids.size()) +
                                   completion_tokens) +
                    "}}";
            out.send(200, "application/json", body);
            return;
        }

        // Streaming: each detokenized piece leaves the process before the
        // next forward pass runs (generate() calls on_token pre-forward).
        out.begin_sse();
        StreamDecoder stream(tok_);
        bool hit_stop = false;
        const auto emit = [&](std::string_view text, const char* finish_reason) {
            out.send_event(prefix + ",\"choices\":[" +
                           choice_json(text, finish_reason) + "]}");
        };
        const TokenCallback on_token = [&](int32_t token) {
            if (is_stop(token)) {
                hit_stop = true;
                return;
            }
            const std::string piece = stream.push(token);
            if (!piece.empty()) {  // empty = mid-UTF-8-sequence, keep buffering
                emit(piece, nullptr);
            }
        };
        generate(engine_, prompt_ids, req.max_tokens, stop_ids_, sampler,
                 /*stats=*/nullptr, on_token);
        const std::string tail = stream.flush();
        if (!tail.empty()) {
            emit(tail, nullptr);
        }
        emit("", hit_stop ? "stop" : "length");
        out.send_event("[DONE]");
    }

    Tokenizer tok_;
    Engine engine_;
    std::vector<int32_t> stop_ids_;
    std::string model_name_;
    int64_t served_ = 0;
};

}  // namespace

int cmd_serve(const std::vector<std::string>& args) {
    // args: serve <model_dir> [--port P] [--int8] [--threads N]
    std::string model_dir;
    long port = 8080;
    bool int8 = false;
    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--port" && i + 1 < args.size()) {
            port = std::atol(args[++i].c_str());
        } else if (args[i] == "--int8") {
            int8 = true;
        } else if (args[i] == "--threads" && i + 1 < args.size()) {
            ops::set_num_threads(std::atoi(args[++i].c_str()));
        } else if (model_dir.empty() && args[i][0] != '-') {
            model_dir = args[i];
        } else {
            std::fprintf(stderr, "serve: unknown argument: %s\n", args[i].c_str());
            return 2;
        }
    }
    if (model_dir.empty() || port < 0 || port > 65535) {
        std::fprintf(stderr,
                     "usage: nanoserve serve <model_dir> [--port P] [--int8] "
                     "[--threads N]\n       (--port 0 = OS-assigned)\n");
        return 2;
    }

    std::string weights_file;  // empty = fp32 model.safetensors
    if (int8) {
        weights_file = model_dir + "/model.int8.safetensors";
    }
    std::fprintf(stderr, "loading model (%s)...\n", int8 ? "int8" : "fp32");
    CompletionService service(model_dir, weights_file);
    http::Server server(static_cast<uint16_t>(port));

    // stdout, after the model is loaded: "when this line appears, requests
    // will be answered" — test_server waits for exactly this.
    std::printf("nanoserve listening on http://127.0.0.1:%u\n",
                static_cast<unsigned>(server.port()));
    std::fflush(stdout);
    server.run([&service](const http::Request& req, http::ResponseWriter& out) {
        service.handle(req, out);
    });
}

}  // namespace nano
