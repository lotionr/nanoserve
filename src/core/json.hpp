// Minimal JSON parser — just enough for what nanoserve reads:
// safetensors headers, model config.json, vocab.json, tokenizer_config.json.
//
// Written from scratch (no third-party deps) on purpose: parsing the actual
// file formats is part of understanding the serving stack. Strict where it
// matters (bounds, escapes, surrogate pairs), simple everywhere else.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nano::json {

class Value;

/// Object members as an ordered vector of (key, value). Lookup is linear —
/// fine for the small objects we read; the tokenizer builds its own hash maps.
using Member = std::pair<std::string, Value>;

class Value {
public:
    enum class Type { Null, Bool, Int, Double, String, Array, Object };

    Value() = default;
    explicit Value(bool b) : type_(Type::Bool), bool_(b) {}
    explicit Value(int64_t i) : type_(Type::Int), int_(i) {}
    explicit Value(double d) : type_(Type::Double), double_(d) {}
    explicit Value(std::string s) : type_(Type::String), string_(std::move(s)) {}

    static Value make_array() {
        Value v;
        v.type_ = Type::Array;
        return v;
    }
    static Value make_object() {
        Value v;
        v.type_ = Type::Object;
        return v;
    }

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_object() const { return type_ == Type::Object; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_string() const { return type_ == Type::String; }
    bool is_number() const { return type_ == Type::Int || type_ == Type::Double; }

    bool as_bool() const {
        expect(Type::Bool, "bool");
        return bool_;
    }
    int64_t as_int() const {
        expect(Type::Int, "integer");
        return int_;
    }
    /// Numeric accessor that accepts both 3 and 3.0 (config files mix them).
    double as_double() const {
        if (type_ == Type::Int) {
            return static_cast<double>(int_);
        }
        expect(Type::Double, "number");
        return double_;
    }
    const std::string& as_string() const {
        expect(Type::String, "string");
        return string_;
    }
    const std::vector<Value>& items() const {
        expect(Type::Array, "array");
        return array_;
    }
    const std::vector<Member>& members() const {
        expect(Type::Object, "object");
        return object_;
    }

    /// Returns nullptr if `key` is absent (or this is not an object).
    const Value* find(std::string_view key) const;
    /// Throws with a readable message if `key` is absent.
    const Value& at(std::string_view key) const;

    // Mutation used by the parser only.
    std::vector<Value>& mutable_items() { return array_; }
    std::vector<Member>& mutable_members() { return object_; }

private:
    void expect(Type t, const char* name) const {
        if (type_ != t) {
            throw std::runtime_error(std::string("json: expected ") + name);
        }
    }

    Type type_ = Type::Null;
    bool bool_ = false;
    int64_t int_ = 0;
    double double_ = 0.0;
    std::string string_;
    std::vector<Value> array_;
    std::vector<Member> object_;
};

/// Parses a complete JSON document. Throws std::runtime_error with a byte
/// offset on malformed input. Trailing non-whitespace is an error.
Value parse(std::string_view text);

/// Reads an entire file into a string. Throws on failure.
std::string read_file(const std::string& path);

}  // namespace nano::json
