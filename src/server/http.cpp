#include "server/http.hpp"

#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <csignal>
#include <cstring>

namespace nano::http {

namespace {

// A request bigger than this is not something an inference demo needs to
// accept; refuse instead of buffering unboundedly.
constexpr size_t kMaxHeaderBytes = 64 * 1024;
constexpr size_t kMaxBodyBytes = 1024 * 1024;

/// ASCII case-insensitive equality (header names).
bool iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) !=
            std::tolower(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

/// Reads until the blank line ending the headers, then Content-Length more
/// bytes. Throws ParseError on anything malformed or oversized.
Request read_request(int fd) {
    std::string buf;
    size_t header_end = std::string::npos;
    char chunk[4096];
    while (header_end == std::string::npos) {
        if (buf.size() > kMaxHeaderBytes) {
            throw ParseError(431, "request headers too large");
        }
        const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) {
            throw ParseError(400, "connection closed before headers finished");
        }
        buf.append(chunk, static_cast<size_t>(n));
        header_end = buf.find("\r\n\r\n");
    }
    const std::string_view head = std::string_view(buf).substr(0, header_end);

    // Request line: METHOD SP PATH SP VERSION
    const size_t line_end = head.find("\r\n");
    const std::string_view line = head.substr(0, line_end);
    const size_t sp1 = line.find(' ');
    const size_t sp2 = line.rfind(' ');
    if (sp1 == std::string_view::npos || sp2 == sp1) {
        throw ParseError(400, "malformed request line");
    }
    Request req;
    req.method = std::string(line.substr(0, sp1));
    req.path = std::string(line.substr(sp1 + 1, sp2 - sp1 - 1));

    // Headers: only Content-Length matters to us.
    size_t content_length = 0;
    std::string_view rest = line_end == std::string_view::npos
                                ? std::string_view()
                                : head.substr(line_end + 2);
    while (!rest.empty()) {
        const size_t eol = rest.find("\r\n");
        const std::string_view header = rest.substr(0, eol);
        rest = eol == std::string_view::npos ? std::string_view() : rest.substr(eol + 2);
        const size_t colon = header.find(':');
        if (colon == std::string_view::npos) {
            continue;
        }
        const std::string_view name = header.substr(0, colon);
        std::string_view value = header.substr(colon + 1);
        while (!value.empty() && value.front() == ' ') {
            value.remove_prefix(1);
        }
        if (iequals(name, "content-length")) {
            content_length = 0;
            for (const char c : value) {
                if (c < '0' || c > '9') {
                    throw ParseError(400, "bad Content-Length");
                }
                content_length = content_length * 10 + static_cast<size_t>(c - '0');
                if (content_length > kMaxBodyBytes) {
                    throw ParseError(413, "request body too large");
                }
            }
        } else if (iequals(name, "transfer-encoding")) {
            throw ParseError(400, "chunked request bodies not supported");
        }
    }

    req.body = buf.substr(header_end + 4);
    while (req.body.size() < content_length) {
        const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) {
            throw ParseError(400, "connection closed before body finished");
        }
        req.body.append(chunk, static_cast<size_t>(n));
    }
    req.body.resize(content_length);  // ignore pipelined extra bytes
    return req;
}

}  // namespace

std::string_view status_text(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        default:  return "Error";
    }
}

void ResponseWriter::write_all(std::string_view bytes) {
    while (!bytes.empty()) {
        const ssize_t n = ::send(fd_, bytes.data(), bytes.size(), 0);
        if (n <= 0) {
            throw std::runtime_error("client closed the connection");
        }
        bytes.remove_prefix(static_cast<size_t>(n));
    }
}

void ResponseWriter::send(int status, std::string_view content_type,
                          std::string_view body) {
    std::string head = "HTTP/1.1 " + std::to_string(status) + " ";
    head += status_text(status);
    head += "\r\nContent-Type: ";
    head += content_type;
    head += "\r\nContent-Length: " + std::to_string(body.size());
    head += "\r\nConnection: close\r\n\r\n";
    headers_sent_ = true;
    write_all(head);
    write_all(body);
}

void ResponseWriter::begin_sse() {
    headers_sent_ = true;
    write_all(
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n\r\n");
}

void ResponseWriter::send_event(std::string_view data) {
    std::string frame = "data: ";
    frame += data;
    frame += "\n\n";
    write_all(frame);
}

Server::Server(uint16_t port) {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        throw std::runtime_error("socket(): " + std::string(std::strerror(errno)));
    }
    const int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    if (::bind(listen_fd_, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) < 0 ||
        ::listen(listen_fd_, 16) < 0) {
        const std::string why = std::strerror(errno);
        ::close(listen_fd_);
        throw std::runtime_error("bind/listen on port " + std::to_string(port) + ": " +
                                 why);
    }
    // Port 0 = OS-assigned; read back what we actually got.
    socklen_t len = sizeof(addr);
    ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len);
    port_ = ntohs(addr.sin_port);
}

Server::~Server() {
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
    }
}

void Server::run(const Handler& handler) {
    // A client that disconnects mid-stream must surface as a write error we
    // can catch, not a process-killing SIGPIPE.
    std::signal(SIGPIPE, SIG_IGN);

    for (;;) {
        const int fd = ::accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) {
            continue;  // interrupted accept; nothing to clean up
        }
        ResponseWriter out(fd);
        try {
            const Request req = read_request(fd);
            handler(req, out);
        } catch (const ParseError& e) {
            try {
                out.send(e.status, "text/plain", std::string(e.what()) + "\n");
            } catch (...) {
            }
        } catch (const std::exception& e) {
            // Handler failed. If the response already started (mid-SSE),
            // closing the socket is all we can honestly do.
            if (!out.headers_sent()) {
                try {
                    out.send(500, "text/plain", std::string(e.what()) + "\n");
                } catch (...) {
                }
            }
        }
        ::close(fd);
    }
}

}  // namespace nano::http
