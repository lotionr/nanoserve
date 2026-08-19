// POST /v1/completions (F035 + F036): the classic OpenAI completions schema,
// the subset an inference demo honestly implements:
//
//   request:  prompt (string, required), max_tokens, temperature, top_k,
//             top_p, seed, stream; n must be 1 if present. The prompt is
//             completed RAW — no chat template — matching the endpoint's
//             "continue this text" contract (the CLI's `generate` is the
//             chat-formatted interface).
//   response: text_completion object with one choice and usage counts;
//             completion_tokens counts every sampled token, including the
//             stop token that ends the text but is never shown in it.
//   stream:   Server-Sent Events, one chunk per detokenized piece, a final
//             empty chunk carrying finish_reason, then `data: [DONE]`.
//
// Continuous batching (F036): requests are decoded TOGETHER, not queued
// behind each other. The accept thread validates each request and moves its
// connection into a FIFO queue; one worker thread owns the BatchEngine and
// loops { admit -> prefill newcomers -> one decode_step for every live
// request }, so a request joins the batch at the next step boundary instead
// of waiting for earlier requests to finish, and each weight read serves
// every live request (see batch.hpp for why that is the throughput win).
// Admission is capped at --max-batch live requests — exactly what the
// engine's page pool is sized for — and the queue simply holds the rest.
//
// Per-request state that must not be shared (Sampler RNG, StreamDecoder
// UTF-8 buffer, the KV cache pages) lives in one Live struct per request;
// the only cross-request state is the engine itself.
#include "server/serve.hpp"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/json.hpp"
#include "core/ops.hpp"
#include "model/batch.hpp"
#include "model/sampler.hpp"
#include "model/tokenizer.hpp"
#include "server/http.hpp"

namespace nano {

namespace {

/// Client-caused failures: reported as HTTP 400 with the message.
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
/// handle() runs on the accept thread; everything that touches the engine or
/// the tokenizer runs on the single worker thread.
class CompletionService {
public:
    CompletionService(const std::string& model_dir, const std::string& weights_file,
                      int64_t max_batch)
        : tok_(Tokenizer::from_dir(model_dir)),
          engine_(model_dir, /*max_seq=*/2048, weights_file, max_batch),
          stop_ids_{tok_.special_id("<|im_end|>"), tok_.special_id("<|endoftext|>")},
          model_name_(basename_of(model_dir)),
          worker_([this] { worker_loop(); }) {}

    ~CompletionService() {
        {
            const std::lock_guard<std::mutex> lock(mu_);
            stopping_ = true;
        }
        cv_.notify_all();
        worker_.join();
    }

    /// Accept-thread side: answer what can be answered without the model
    /// (routing and schema errors), queue everything real for the worker.
    /// Never throws — this connection is ours to answer (http.hpp contract).
    void handle(http::Connection conn) {
        const http::Request& req = conn.request();
        try {
            if (req.path != "/v1/completions") {
                conn.writer().send(404, "application/json",
                                   error_json("no such endpoint: " + req.path));
                return;
            }
            if (req.method != "POST") {
                conn.writer().send(405, "application/json", error_json("use POST"));
                return;
            }
            CompletionRequest completion;
            try {
                completion = parse_completion(req.body);
            } catch (const BadRequest& e) {
                conn.writer().send(400, "application/json", error_json(e.what()));
                return;
            }
            {
                const std::lock_guard<std::mutex> lock(mu_);
                queue_.push_back({std::move(conn), std::move(completion)});
            }
            cv_.notify_one();
        } catch (...) {
            // Writing the error failed (client gone): drop the connection.
        }
    }

private:
    struct Pending {
        http::Connection conn;
        CompletionRequest req;
    };

    /// Everything one in-flight request owns. Created at admission on the
    /// worker thread, destroyed when its response is finished (or its client
    /// goes away) — destroying seq returns the request's pages to the pool.
    struct Live {
        Live(Pending&& p, std::string id_, int64_t created_)
            : conn(std::move(p.conn)), req(std::move(p.req)), id(std::move(id_)),
              created(created_) {}

        http::Connection conn;
        CompletionRequest req;
        std::string id;
        int64_t created = 0;
        std::unique_ptr<Sequence> seq;
        std::unique_ptr<Sampler> sampler;
        std::unique_ptr<StreamDecoder> stream;  // stream mode only
        int64_t prompt_tokens = 0;
        std::vector<int32_t> generated;  // every sampled id, stop included
        int32_t last_token = 0;          // fed to the next decode step
    };

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
    std::string object_prefix(const Live& l) const {
        return "{\"id\":" + json::quote(l.id) +
               ",\"object\":\"text_completion\",\"created\":" +
               std::to_string(l.created) + ",\"model\":" + json::quote(model_name_);
    }

    static std::string choice_json(std::string_view text, const char* finish_reason) {
        std::string choice = "{\"index\":0,\"text\":" + json::quote(text) +
                             ",\"logprobs\":null,\"finish_reason\":";
        choice += finish_reason == nullptr
                      ? "null"
                      : "\"" + std::string(finish_reason) + "\"";
        return choice + "}";
    }

    void send_chunk(Live& l, std::string_view text, const char* finish_reason) {
        l.conn.writer().send_event(object_prefix(l) + ",\"choices\":[" +
                                   choice_json(text, finish_reason) + "]}");
    }

    /// The full non-streaming response, or the stream's tail. Runs when the
    /// request finishes normally; write failures propagate to the caller,
    /// which retires the request either way.
    void send_final(Live& l, bool hit_stop) {
        const char* finish = hit_stop ? "stop" : "length";
        if (l.req.stream) {
            const std::string tail = l.stream->flush();
            if (!tail.empty()) {
                send_chunk(l, tail, nullptr);
            }
            send_chunk(l, "", finish);
            l.conn.writer().send_event("[DONE]");
            return;
        }
        std::vector<int32_t> shown = l.generated;
        if (hit_stop) {
            shown.pop_back();  // the stop token ends, but never joins, the text
        }
        const int64_t completion_tokens = static_cast<int64_t>(l.generated.size());
        std::string body = object_prefix(l);
        body += ",\"choices\":[";
        body += choice_json(tok_.decode(shown), finish);
        body += "],\"usage\":{\"prompt_tokens\":" + std::to_string(l.prompt_tokens) +
                ",\"completion_tokens\":" + std::to_string(completion_tokens) +
                ",\"total_tokens\":" +
                std::to_string(l.prompt_tokens + completion_tokens) + "}}";
        l.conn.writer().send(200, "application/json", body);
    }

    /// Feeds one freshly sampled token into a request's state and streams it
    /// out if streaming. Returns true when the request is FINISHED (stop
    /// token, max_tokens reached, or its client went away) — the caller then
    /// drops the Live, which frees the sequence's pages.
    bool deliver(Live& l, int32_t token) {
        l.generated.push_back(token);
        const bool hit_stop = is_stop(token);
        const bool at_cap = static_cast<int64_t>(l.generated.size()) >= l.req.max_tokens;
        try {
            if (l.req.stream && !hit_stop) {
                const std::string piece = l.stream->push(token);
                if (!piece.empty()) {  // empty = mid-UTF-8-sequence, buffering
                    send_chunk(l, piece, nullptr);
                }
            }
            if (hit_stop || at_cap) {
                send_final(l, hit_stop);
                return true;
            }
        } catch (const std::exception&) {
            return true;  // client hung up: retire this request, serve the rest
        }
        l.last_token = token;
        return false;
    }

    /// Admission (worker thread): tokenize, prefill the whole prompt in one
    /// pass, sample the first token. Returns true if the request already
    /// finished (validation failure, 1-token completion, client gone).
    bool start(Live& l) {
        std::vector<int32_t> prompt_ids;
        try {
            prompt_ids = tok_.encode(l.req.prompt);
            if (prompt_ids.empty()) {
                throw BadRequest("prompt produced no tokens");
            }
            if (static_cast<int64_t>(prompt_ids.size()) + l.req.max_tokens >
                engine_.max_seq()) {
                throw BadRequest("prompt + max_tokens exceeds the " +
                                 std::to_string(engine_.max_seq()) + "-token context");
            }
        } catch (const BadRequest& e) {
            try {
                l.conn.writer().send(400, "application/json", error_json(e.what()));
            } catch (...) {
            }
            return true;
        }
        l.prompt_tokens = static_cast<int64_t>(prompt_ids.size());
        l.seq = engine_.new_sequence();
        l.sampler = std::make_unique<Sampler>(l.req.sampling,
                                              engine_.model().config.vocab_size);
        if (l.req.stream) {
            l.stream = std::make_unique<StreamDecoder>(tok_);
            try {
                l.conn.writer().begin_sse();
            } catch (const std::exception&) {
                return true;  // client gone before the first byte
            }
        }
        const std::span<const float> logits = engine_.prefill(*l.seq, prompt_ids);
        return deliver(l, l.sampler->sample(logits));
    }

    /// The scheduler. One thread, one loop: admit whatever fits, prefill the
    /// newcomers, then run ONE decode step for every live request. New
    /// requests therefore join between decode steps (continuous batching),
    /// never behind a whole earlier request.
    void worker_loop() {
        std::vector<std::unique_ptr<Live>> live;
        for (;;) {
            // Admit up to max_seqs live requests, FIFO. Block only when
            // there is nothing to decode AND nothing to admit.
            std::vector<std::unique_ptr<Live>> admitted;
            {
                std::unique_lock<std::mutex> lock(mu_);
                if (live.empty()) {
                    cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                }
                if (stopping_) {
                    return;
                }
                while (live.size() + admitted.size() <
                           static_cast<size_t>(engine_.max_seqs()) &&
                       !queue_.empty()) {
                    admitted.push_back(std::make_unique<Live>(
                        std::move(queue_.front()), "cmpl-" + std::to_string(++served_),
                        static_cast<int64_t>(std::time(nullptr))));
                    queue_.pop_front();
                }
            }
            for (std::unique_ptr<Live>& l : admitted) {
                if (!start(*l)) {
                    live.push_back(std::move(l));
                }
                // else: finished during admission; dropping it frees its pages
            }
            if (live.empty()) {
                continue;
            }

            // One decode step for the whole batch.
            seqs_.clear();
            tokens_.clear();
            for (const std::unique_ptr<Live>& l : live) {
                seqs_.push_back(l->seq.get());
                tokens_.push_back(l->last_token);
            }
            const std::span<const float> logits = engine_.decode_step(seqs_, tokens_);
            const int64_t vocab = engine_.model().config.vocab_size;
            size_t kept = 0;
            for (size_t i = 0; i < live.size(); ++i) {
                Live& l = *live[i];
                const std::span<const float> row =
                    logits.subspan(i * static_cast<size_t>(vocab),
                                   static_cast<size_t>(vocab));
                if (!deliver(l, l.sampler->sample(row))) {
                    live[kept++] = std::move(live[i]);  // still running
                }
            }
            live.resize(kept);
        }
    }

    Tokenizer tok_;
    BatchEngine engine_;
    std::vector<int32_t> stop_ids_;
    std::string model_name_;
    int64_t served_ = 0;  // worker thread only

    std::mutex mu_;                // guards queue_ + stopping_
    std::condition_variable cv_;
    std::deque<Pending> queue_;
    bool stopping_ = false;
    std::vector<Sequence*> seqs_;    // decode-step scratch (worker only)
    std::vector<int32_t> tokens_;
    std::thread worker_;  // last member: joins before the rest is destroyed
};

}  // namespace

int cmd_serve(const std::vector<std::string>& args) {
    // args: serve <model_dir> [--port P] [--int8] [--threads N] [--max-batch B]
    std::string model_dir;
    long port = 8080;
    bool int8 = false;
    long max_batch = 4;
    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--port" && i + 1 < args.size()) {
            port = std::atol(args[++i].c_str());
        } else if (args[i] == "--int8") {
            int8 = true;
        } else if (args[i] == "--threads" && i + 1 < args.size()) {
            ops::set_num_threads(std::atoi(args[++i].c_str()));
        } else if (args[i] == "--max-batch" && i + 1 < args.size()) {
            max_batch = std::atol(args[++i].c_str());
        } else if (model_dir.empty() && args[i][0] != '-') {
            model_dir = args[i];
        } else {
            std::fprintf(stderr, "serve: unknown argument: %s\n", args[i].c_str());
            return 2;
        }
    }
    if (model_dir.empty() || port < 0 || port > 65535 || max_batch < 1 ||
        max_batch > 64) {
        std::fprintf(stderr,
                     "usage: nanoserve serve <model_dir> [--port P] [--int8] "
                     "[--threads N] [--max-batch B]\n"
                     "       (--port 0 = OS-assigned; --max-batch: 1..64 requests "
                     "decoded together, default 4)\n");
        return 2;
    }

    std::string weights_file;  // empty = fp32 model.safetensors
    if (int8) {
        weights_file = model_dir + "/model.int8.safetensors";
    }
    std::fprintf(stderr, "loading model (%s, max batch %ld)...\n",
                 int8 ? "int8" : "fp32", max_batch);
    CompletionService service(model_dir, weights_file, max_batch);
    http::Server server(static_cast<uint16_t>(port));

    // stdout, after the model is loaded: "when this line appears, requests
    // will be answered" — test_server waits for exactly this.
    std::printf("nanoserve listening on http://127.0.0.1:%u\n",
                static_cast<unsigned>(server.port()));
    std::fflush(stdout);
    server.run([&service](http::Connection conn) {
        service.handle(std::move(conn));
    });
}

}  // namespace nano
