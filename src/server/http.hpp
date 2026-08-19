// Minimal HTTP/1.1 server (F035): POSIX sockets, blocking, one connection at
// a time, Connection: close on every response. That is not a toy shortcut so
// much as the honest shape of the current engine — decode is single-sequence,
// so overlapping requests couldn't share it anyway. Continuous batching
// (F036) is the feature that changes this, and it will change this file.
//
// From scratch (no third-party deps) like the JSON/BPE/safetensors code:
// parsing the protocol the tokens travel over is part of the serving story.
#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace nano::http {

struct Request {
    std::string method;  // "POST", uppercased as received
    std::string path;    // "/v1/completions" (query strings not split off)
    std::string body;
};

/// Thrown by request parsing when the bytes on the wire are not a request we
/// can serve; run() turns it into a bare error response with this status.
struct ParseError : std::runtime_error {
    ParseError(int status_, const std::string& message)
        : std::runtime_error(message), status(status_) {}
    int status;
};

/// Writes exactly one response on a connected socket. Two modes:
///   - send(...):    complete response with Content-Length (JSON bodies)
///   - begin_sse() + send_event(...): streaming Server-Sent Events; the
///     response ends when the connection closes (we send Connection: close,
///     so no chunked framing is needed)
/// Write failures (client went away) throw; the caller drops the connection.
class ResponseWriter {
public:
    explicit ResponseWriter(int fd) : fd_(fd) {}

    void send(int status, std::string_view content_type, std::string_view body);
    void begin_sse();
    /// One "data: <data>\n\n" frame, flushed to the socket immediately —
    /// this is the "token appears as soon as it is sampled" path.
    void send_event(std::string_view data);

    bool headers_sent() const { return headers_sent_; }

private:
    void write_all(std::string_view bytes);

    int fd_;
    bool headers_sent_ = false;
};

using Handler = std::function<void(const Request&, ResponseWriter&)>;

class Server {
public:
    /// Binds and listens on 127.0.0.1:`port` (loopback only — this is a demo
    /// server, not an internet-facing one). Port 0 asks the OS for a free
    /// port; port() reports the real one either way. Throws on bind failure.
    explicit Server(uint16_t port);
    ~Server();
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    uint16_t port() const { return port_; }

    /// Accept loop, forever: parse one request, call `handler`, close. A
    /// connection-level error (bad request, client hangup mid-write) drops
    /// that connection and keeps serving; handler exceptions become a 500 if
    /// the response hasn't started yet.
    [[noreturn]] void run(const Handler& handler);

private:
    int listen_fd_ = -1;
    uint16_t port_ = 0;
};

/// "OK", "Bad Request", ... for the handful of statuses this server emits.
std::string_view status_text(int status);

}  // namespace nano::http
