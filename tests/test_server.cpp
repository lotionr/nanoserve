// F035, end to end: spawn the real `nanoserve serve` binary and speak real
// HTTP to it over a real socket — no mocked transport, because the feature
// IS the transport. ctest passes the binary's path as argv[1].
//
// Checks: a well-formed non-streaming completion (schema + usage counts
// against a locally-loaded tokenizer), greedy determinism across requests,
// SSE streaming whose concatenated chunks equal the non-streamed text and
// which terminates with finish_reason + [DONE], and the error paths
// (malformed JSON -> 400, wrong method -> 405, unknown path -> 404).
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "model/tokenizer.hpp"
#include "testing.hpp"

namespace {

/// Infra failure (can't spawn/connect): abort the test, this is not a check.
[[noreturn]] void die(const std::string& why) {
    std::fprintf(stderr, "test_server: %s (errno: %s)\n", why.c_str(),
                 std::strerror(errno));
    std::exit(1);
}

struct ServerProc {
    pid_t pid = -1;
    int out_fd = -1;  // server's stdout
    uint16_t port = 0;

    ~ServerProc() {
        if (pid > 0) {
            ::kill(pid, SIGTERM);
            int status = 0;
            ::waitpid(pid, &status, 0);
        }
        if (out_fd >= 0) {
            ::close(out_fd);
        }
    }
};

/// Spawns `binary serve model_dir --port 0` and waits (up to 120 s) for the
/// "listening on http://127.0.0.1:<port>" line that means the model is
/// loaded and the socket is accepting.
ServerProc start_server(const char* binary, const std::string& model_dir) {
    int fds[2];
    if (::pipe(fds) != 0) {
        die("pipe() failed");
    }
    ServerProc server;
    server.pid = ::fork();
    if (server.pid < 0) {
        die("fork() failed");
    }
    if (server.pid == 0) {
        ::dup2(fds[1], STDOUT_FILENO);
        ::close(fds[0]);
        ::close(fds[1]);
        ::execl(binary, binary, "serve", model_dir.c_str(), "--port", "0",
                static_cast<char*>(nullptr));
        std::perror("execl");
        ::_exit(127);
    }
    ::close(fds[1]);
    server.out_fd = fds[0];

    const std::string marker = "listening on http://127.0.0.1:";
    std::string seen;
    for (;;) {
        pollfd pfd = {server.out_fd, POLLIN, 0};
        const int r = ::poll(&pfd, 1, 120 * 1000);
        if (r <= 0) {
            die("server did not report listening within 120 s");
        }
        char buf[512];
        const ssize_t n = ::read(server.out_fd, buf, sizeof(buf));
        if (n <= 0) {
            die("server exited before listening (is the binary path right?)");
        }
        seen.append(buf, static_cast<size_t>(n));
        const size_t at = seen.find(marker);
        if (at != std::string::npos && seen.find('\n', at) != std::string::npos) {
            server.port = static_cast<uint16_t>(
                std::atoi(seen.c_str() + at + marker.size()));
            break;
        }
    }
    if (server.port == 0) {
        die("could not parse the server port");
    }
    return server;
}

/// One full HTTP exchange: connect, send, read to EOF (the server closes).
std::string http_exchange(uint16_t port, const std::string& request) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        die("socket() failed");
    }
    timeval timeout = {120, 0};  // generous: covers a full 16-token decode
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    if (::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
        die("connect() failed");
    }
    size_t sent = 0;
    while (sent < request.size()) {
        const ssize_t n = ::send(fd, request.data() + sent, request.size() - sent, 0);
        if (n <= 0) {
            die("send() failed");
        }
        sent += static_cast<size_t>(n);
    }
    std::string response;
    char buf[4096];
    for (;;) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n < 0) {
            die("recv() failed or timed out");
        }
        if (n == 0) {
            break;
        }
        response.append(buf, static_cast<size_t>(n));
    }
    ::close(fd);
    return response;
}

std::string post_completions(const std::string& body) {
    return "POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
           "Content-Type: application/json\r\nContent-Length: " +
           std::to_string(body.size()) + "\r\n\r\n" + body;
}

int status_of(const std::string& response) {
    // "HTTP/1.1 200 OK" — the code sits after the first space.
    const size_t sp = response.find(' ');
    return sp == std::string::npos ? -1 : std::atoi(response.c_str() + sp + 1);
}

std::string body_of(const std::string& response) {
    const size_t at = response.find("\r\n\r\n");
    return at == std::string::npos ? std::string() : response.substr(at + 4);
}

std::string headers_of(const std::string& response) {
    const size_t at = response.find("\r\n\r\n");
    return at == std::string::npos ? response : response.substr(0, at);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: test_server <path-to-nanoserve-binary>\n");
        return 1;
    }
    const std::string dir = nano::testing::require_model_dir();
    const nano::Tokenizer tok = nano::Tokenizer::from_dir(dir);
    const ServerProc server = start_server(argv[1], dir);
    std::printf("server up on port %u\n", static_cast<unsigned>(server.port));

    const std::string prompt = "The capital of France is";
    const std::string request_json =
        "{\"prompt\":\"" + prompt + "\",\"max_tokens\":8,\"temperature\":0}";

    // ---- A: well-formed non-streaming completion --------------------------
    std::string first_text;
    {
        const std::string response = http_exchange(server.port, post_completions(request_json));
        NANO_CHECK_MSG(status_of(response) == 200, "status %d, want 200",
                       status_of(response));
        const nano::json::Value doc = nano::json::parse(body_of(response));
        NANO_CHECK(doc.at("object").as_string() == "text_completion");
        NANO_CHECK(!doc.at("id").as_string().empty());
        const auto& choices = doc.at("choices").items();
        NANO_CHECK_MSG(choices.size() == 1, "%zu choices, want 1", choices.size());
        first_text = choices[0].at("text").as_string();
        NANO_CHECK_MSG(!first_text.empty(), "empty completion text");
        const std::string finish = choices[0].at("finish_reason").as_string();
        NANO_CHECK_MSG(finish == "stop" || finish == "length",
                       "finish_reason \"%s\"", finish.c_str());

        // usage must agree with a locally computed tokenization.
        const auto& usage = doc.at("usage");
        const int64_t plen = static_cast<int64_t>(tok.encode(prompt).size());
        NANO_CHECK_MSG(usage.at("prompt_tokens").as_int() == plen,
                       "prompt_tokens %lld, local tokenizer says %lld",
                       static_cast<long long>(usage.at("prompt_tokens").as_int()),
                       static_cast<long long>(plen));
        const int64_t completion = usage.at("completion_tokens").as_int();
        NANO_CHECK(completion >= 1 && completion <= 8);
        NANO_CHECK(usage.at("total_tokens").as_int() == plen + completion);
        std::printf("completion: \"%s\"\n", first_text.c_str());
    }

    // ---- B: greedy determinism across requests ----------------------------
    {
        const std::string response = http_exchange(server.port, post_completions(request_json));
        const nano::json::Value doc = nano::json::parse(body_of(response));
        const std::string text = doc.at("choices").items()[0].at("text").as_string();
        NANO_CHECK_MSG(text == first_text,
                       "greedy completion changed between requests: \"%s\" vs \"%s\"",
                       text.c_str(), first_text.c_str());
    }

    // ---- C: SSE streaming --------------------------------------------------
    {
        const std::string body =
            "{\"prompt\":\"" + prompt + "\",\"max_tokens\":8,\"temperature\":0,"
            "\"stream\":true}";
        const std::string response = http_exchange(server.port, post_completions(body));
        NANO_CHECK_MSG(status_of(response) == 200, "stream status %d", status_of(response));
        NANO_CHECK_MSG(headers_of(response).find("text/event-stream") != std::string::npos,
                       "missing text/event-stream content type");

        // Split the SSE payload into events on the blank-line separator.
        std::vector<std::string> events;
        std::string payload = body_of(response);
        size_t start = 0;
        while (start < payload.size()) {
            size_t end = payload.find("\n\n", start);
            if (end == std::string::npos) {
                end = payload.size();
            }
            if (end > start) {
                events.push_back(payload.substr(start, end - start));
            }
            start = end + 2;
        }
        NANO_CHECK_MSG(events.size() >= 3, "only %zu SSE events", events.size());

        std::string streamed_text;
        bool saw_finish = false;
        for (size_t i = 0; i < events.size(); ++i) {
            NANO_CHECK_MSG(events[i].rfind("data: ", 0) == 0,
                           "event %zu missing \"data: \" prefix: %s", i,
                           events[i].c_str());
            const std::string data = events[i].substr(6);
            if (i + 1 == events.size()) {
                NANO_CHECK_MSG(data == "[DONE]", "last event is \"%s\", want [DONE]",
                               data.c_str());
                break;
            }
            const nano::json::Value chunk = nano::json::parse(data);
            NANO_CHECK(chunk.at("object").as_string() == "text_completion");
            const auto& choice = chunk.at("choices").items()[0];
            streamed_text += choice.at("text").as_string();
            if (!choice.at("finish_reason").is_null()) {
                // Only the final chunk carries it, with empty text.
                NANO_CHECK(i + 2 == events.size());
                NANO_CHECK(choice.at("text").as_string().empty());
                saw_finish = true;
            }
        }
        NANO_CHECK_MSG(saw_finish, "no chunk carried a finish_reason");
        NANO_CHECK_MSG(streamed_text == first_text,
                       "streamed \"%s\" != non-streamed \"%s\"", streamed_text.c_str(),
                       first_text.c_str());
    }

    // ---- D: error paths ----------------------------------------------------
    {
        const std::string bad = http_exchange(server.port, post_completions("{not json"));
        NANO_CHECK_MSG(status_of(bad) == 400, "malformed JSON gave %d, want 400",
                       status_of(bad));

        const std::string missing = http_exchange(
            server.port, post_completions("{\"max_tokens\":4}"));
        NANO_CHECK_MSG(status_of(missing) == 400, "missing prompt gave %d, want 400",
                       status_of(missing));

        const std::string get = http_exchange(
            server.port, "GET /v1/completions HTTP/1.1\r\nHost: x\r\n\r\n");
        NANO_CHECK_MSG(status_of(get) == 405, "GET gave %d, want 405", status_of(get));

        const std::string nope = http_exchange(
            server.port,
            "POST /nope HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}");
        NANO_CHECK_MSG(status_of(nope) == 404, "unknown path gave %d, want 404",
                       status_of(nope));
    }

    return nano::testing::finish("test_server");
}
