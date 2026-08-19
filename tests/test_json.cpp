#include "core/json.hpp"

#include "testing.hpp"

using nano::json::Value;
using nano::json::parse;

int main() {
    // Scalars.
    NANO_CHECK(parse("null").is_null());
    NANO_CHECK(parse("true").as_bool() == true);
    NANO_CHECK(parse("false").as_bool() == false);
    NANO_CHECK(parse("42").as_int() == 42);
    NANO_CHECK(parse("-7").as_int() == -7);
    NANO_CHECK(parse("3.5").as_double() == 3.5);
    NANO_CHECK(parse("1e-6").as_double() == 1e-6);
    NANO_CHECK(parse("1000000.0").as_double() == 1000000.0);
    NANO_CHECK(parse("7").as_double() == 7.0);  // int promotes to double
    NANO_CHECK(parse("\"hi\"").as_string() == "hi");

    // Structures and lookup.
    {
        const Value v = parse(R"({"a": [1, 2, 3], "b": {"c": "d"}, "e": null})");
        NANO_CHECK(v.is_object());
        NANO_CHECK(v.at("a").items().size() == 3);
        NANO_CHECK(v.at("a").items()[2].as_int() == 3);
        NANO_CHECK(v.at("b").at("c").as_string() == "d");
        NANO_CHECK(v.at("e").is_null());
        NANO_CHECK(v.find("missing") == nullptr);
        NANO_CHECK_THROWS(v.at("missing"));
    }

    // Member order is preserved (safetensors tensor order matters for inspect).
    {
        const Value v = parse(R"({"z": 1, "a": 2, "m": 3})");
        NANO_CHECK(v.members()[0].first == "z");
        NANO_CHECK(v.members()[1].first == "a");
        NANO_CHECK(v.members()[2].first == "m");
    }

    // String escapes.
    NANO_CHECK(parse(R"("a\nb\t\"c\"\\")").as_string() == "a\nb\t\"c\"\\");
    NANO_CHECK(parse(R"("A")").as_string() == "A");
    NANO_CHECK(parse(R"("é")").as_string() == "\xC3\xA9");            // é
    NANO_CHECK(parse(R"("中")").as_string() == "\xE4\xB8\xAD");        // 中
    NANO_CHECK(parse(R"("😀")").as_string() == "\xF0\x9F\x98\x80");  // 😀 (surrogates)

    // Whitespace tolerance.
    NANO_CHECK(parse(" \n\t{ \"k\" : [ ] } \r\n").at("k").items().empty());

    // Malformed input is rejected, not crashed on.
    NANO_CHECK_THROWS(parse(""));
    NANO_CHECK_THROWS(parse("{"));
    NANO_CHECK_THROWS(parse("{\"a\": }"));
    NANO_CHECK_THROWS(parse("[1, 2"));
    NANO_CHECK_THROWS(parse("[1, 2] junk"));
    NANO_CHECK_THROWS(parse("\"unterminated"));
    NANO_CHECK_THROWS(parse("{\"a\" 1}"));
    NANO_CHECK_THROWS(parse("tru"));
    NANO_CHECK_THROWS(parse("nul"));
    NANO_CHECK_THROWS(parse("--5"));
    NANO_CHECK_THROWS(parse("1.2.3"));
    NANO_CHECK_THROWS(parse(R"("\u12g4")"));
    NANO_CHECK_THROWS(parse(R"("\ud83d")"));  // unpaired high surrogate
    NANO_CHECK_THROWS(parse(R"("\x41")"));

    // Type errors are loud.
    NANO_CHECK_THROWS(parse("42").as_string());
    NANO_CHECK_THROWS(parse("\"s\"").as_int());

    return nano::testing::finish("test_json");
}
