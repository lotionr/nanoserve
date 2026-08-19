#include "core/json.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace nano::json {

const Value* Value::find(std::string_view key) const {
    if (type_ != Type::Object) {
        return nullptr;
    }
    for (const auto& [k, v] : object_) {
        if (k == key) {
            return &v;
        }
    }
    return nullptr;
}

const Value& Value::at(std::string_view key) const {
    const Value* v = find(key);
    if (v == nullptr) {
        throw std::runtime_error("json: missing key '" + std::string(key) + "'");
    }
    return *v;
}

namespace {

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    Value parse_document() {
        Value v = parse_value();
        skip_whitespace();
        if (pos_ != text_.size()) {
            fail("trailing characters after document");
        }
        return v;
    }

private:
    [[noreturn]] void fail(const std::string& why) const {
        throw std::runtime_error("json: " + why + " at byte " + std::to_string(pos_));
    }

    bool at_end() const { return pos_ >= text_.size(); }

    char peek() const {
        if (at_end()) {
            fail("unexpected end of input");
        }
        return text_[pos_];
    }

    char next() {
        char c = peek();
        ++pos_;
        return c;
    }

    void skip_whitespace() {
        while (!at_end()) {
            char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    void expect_literal(std::string_view lit) {
        if (text_.substr(pos_, lit.size()) != lit) {
            fail("invalid literal");
        }
        pos_ += lit.size();
    }

    Value parse_value() {
        skip_whitespace();
        switch (peek()) {
            case '{':
                return parse_object();
            case '[':
                return parse_array();
            case '"':
                return Value(parse_string());
            case 't':
                expect_literal("true");
                return Value(true);
            case 'f':
                expect_literal("false");
                return Value(false);
            case 'n':
                expect_literal("null");
                return Value();
            default:
                return parse_number();
        }
    }

    Value parse_object() {
        next();  // '{'
        Value obj = Value::make_object();
        skip_whitespace();
        if (peek() == '}') {
            next();
            return obj;
        }
        while (true) {
            skip_whitespace();
            std::string key = parse_string();
            skip_whitespace();
            if (next() != ':') {
                fail("expected ':' in object");
            }
            obj.mutable_members().emplace_back(std::move(key), parse_value());
            skip_whitespace();
            char c = next();
            if (c == '}') {
                return obj;
            }
            if (c != ',') {
                fail("expected ',' or '}' in object");
            }
        }
    }

    Value parse_array() {
        next();  // '['
        Value arr = Value::make_array();
        skip_whitespace();
        if (peek() == ']') {
            next();
            return arr;
        }
        while (true) {
            arr.mutable_items().push_back(parse_value());
            skip_whitespace();
            char c = next();
            if (c == ']') {
                return arr;
            }
            if (c != ',') {
                fail("expected ',' or ']' in array");
            }
        }
    }

    Value parse_number() {
        size_t start = pos_;
        bool is_double = false;
        if (!at_end() && peek() == '-') {
            ++pos_;
        }
        while (!at_end()) {
            char c = text_[pos_];
            if (c >= '0' && c <= '9') {
                ++pos_;
            } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
                is_double = true;
                ++pos_;
            } else {
                break;
            }
        }
        if (pos_ == start) {
            fail("expected a value");
        }
        // Copy for null-termination; number tokens are tiny.
        std::string token(text_.substr(start, pos_ - start));
        errno = 0;
        if (!is_double) {
            char* end = nullptr;
            long long i = std::strtoll(token.c_str(), &end, 10);
            if (end == token.c_str() + token.size() && errno == 0) {
                return Value(static_cast<int64_t>(i));
            }
            // Integer overflow: fall through to double.
        }
        char* end = nullptr;
        errno = 0;
        double d = std::strtod(token.c_str(), &end);
        if (end != token.c_str() + token.size()) {
            fail("malformed number '" + token + "'");
        }
        return Value(d);
    }

    uint32_t parse_hex4() {
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            char c = next();
            value <<= 4;
            if (c >= '0' && c <= '9') {
                value |= static_cast<uint32_t>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                value |= static_cast<uint32_t>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                value |= static_cast<uint32_t>(c - 'A' + 10);
            } else {
                fail("invalid \\u escape");
            }
        }
        return value;
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    std::string parse_string() {
        if (next() != '"') {
            fail("expected string");
        }
        std::string out;
        while (true) {
            char c = next();
            if (c == '"') {
                return out;
            }
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            char esc = next();
            switch (esc) {
                case '"':
                    out.push_back('"');
                    break;
                case '\\':
                    out.push_back('\\');
                    break;
                case '/':
                    out.push_back('/');
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'u': {
                    uint32_t cp = parse_hex4();
                    // UTF-16 surrogate pair -> single code point.
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (next() != '\\' || next() != 'u') {
                            fail("unpaired high surrogate");
                        }
                        uint32_t low = parse_hex4();
                        if (low < 0xDC00 || low > 0xDFFF) {
                            fail("invalid low surrogate");
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        fail("unpaired low surrogate");
                    }
                    append_utf8(out, cp);
                    break;
                }
                default:
                    fail("invalid escape character");
            }
        }
    }

    std::string_view text_;
    size_t pos_ = 0;
};

}  // namespace

Value parse(std::string_view text) {
    return Parser(text).parse_document();
}

std::string quote(std::string_view text) {
    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(text.size() + 2);
    out += '"';
    for (const char c : text) {
        const auto u = static_cast<unsigned char>(c);
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (u < 0x20) {  // remaining control chars need \u00XX
                    out += "\\u00";
                    out += hex[u >> 4];
                    out += hex[u & 0xf];
                } else {
                    out += c;  // UTF-8 bytes pass through untouched
                }
        }
    }
    out += '"';
    return out;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open file: " + path);
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    return std::move(buf).str();
}

}  // namespace nano::json
