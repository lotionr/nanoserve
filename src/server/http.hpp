// Minimal HTTP/1.1 server (F035/F036): POSIX sockets, `Connection: close` on
// every response. The accept loop reads and parses one request at a time,
// then hands the whole connection — socket and parsed request — to the
// handler as an owning Connection object. The handler may answer inline and
// let it die, or move it somewhere longer-lived; that move is what lets the
// serving layer (serve.cpp) keep many responses in flight at once while this
// file stays a single-threaded parser. Continuous batching (F036) is built
// on exactly that handoff.
//
// From scratch (no third-party deps) like the JSON/BPE/safetensors code:
// parsing the protocol the tokens travel over is part of the serving story.
#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

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

/// One accepted connection: the parsed request plus the socket to answer on.
/// Owns the socket — whoever holds the Connection holds the duty to answer,
/// and destroying it closes the connection (which, with Connection: close,
/// IS the end-of-response signal). Move-only, so that duty can be handed to
/// a queue (serve.cpp's scheduler) without any shared ownership.
class Connection {
public:
    Connection(int fd, Request req)
        : fd_(fd), req_(std::move(req)), writer_(fd) {}
    ~Connection() { close_now(); }
    Connection(Connection&& other) noexcept
        : fd_(other.fd_), req_(std::move(other.req_)), writer_(other.fd_) {
        other.fd_ = -1;
    }
    Connection& operator=(Connection&&) = delete;  // members would need re-seating
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    const Request& request() const { return req_; }
    ResponseWriter& writer() { return writer_; }

private:
    void close_now();

    int fd_ = -1;  // -1 after being moved from
    Request req_;
    ResponseWriter writer_;
};

/// Takes ownership of the connection. Must not throw — by the time the
/// handler runs, only it can still answer the client, so it must turn its
/// own failures into responses (or drop the connection) itself.
using Handler = std::function<void(Connection)>;

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

    /// Accept loop, forever: parse one request, hand the connection to
    /// `handler`. A parse error (bad request, client hangup mid-read) is
    /// answered here and that connection dropped; the loop keeps serving.
    [[noreturn]] void run(const Handler& handler);

private:
    int listen_fd_ = -1;
    uint16_t port_ = 0;
};

/// "OK", "Bad Request", ... for the handful of statuses this server emits.
std::string_view status_text(int status);

}  // namespace nano::http
