// Tiny test support for plain-executable tests (llama.cpp style).
// Exit codes: 0 = pass, 77 = skipped (ctest SKIP_RETURN_CODE), else = fail.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>

namespace nano::testing {

inline int failures = 0;
inline int checks = 0;

#define NANO_CHECK(cond)                                                              \
    do {                                                                              \
        ++nano::testing::checks;                                                      \
        if (!(cond)) {                                                                \
            ++nano::testing::failures;                                                \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
        }                                                                             \
    } while (0)

#define NANO_CHECK_MSG(cond, ...)                                                     \
    do {                                                                              \
        ++nano::testing::checks;                                                      \
        if (!(cond)) {                                                                \
            ++nano::testing::failures;                                                \
            std::fprintf(stderr, "FAIL %s:%d: %s — ", __FILE__, __LINE__, #cond);     \
            std::fprintf(stderr, __VA_ARGS__);                                        \
            std::fprintf(stderr, "\n");                                               \
        }                                                                             \
    } while (0)

/// Expects `fn()` to throw; the whole point of strict parsers is that they
/// reject garbage loudly.
#define NANO_CHECK_THROWS(fn)                                                         \
    do {                                                                              \
        ++nano::testing::checks;                                                      \
        bool threw = false;                                                           \
        try {                                                                         \
            (void)(fn);                                                               \
        } catch (const std::exception&) {                                             \
            threw = true;                                                             \
        }                                                                             \
        if (!threw) {                                                                 \
            ++nano::testing::failures;                                                \
            std::fprintf(stderr, "FAIL %s:%d: expected throw: %s\n", __FILE__,        \
                         __LINE__, #fn);                                              \
        }                                                                             \
    } while (0)

[[noreturn]] inline void skip(const char* why) {
    std::fprintf(stderr, "SKIP: %s\n", why);
    std::exit(77);
}

inline bool file_exists(const std::string& path) {
    struct stat st = {};
    return ::stat(path.c_str(), &st) == 0 && st.st_size > 0;
}

/// Skips the test (exit 77) unless the model has been downloaded. Keeps CI —
/// which never downloads the ~1 GB weights — green but honest.
inline std::string require_model_dir() {
    const std::string dir = "models/qwen2.5-0.5b-instruct";
    if (!file_exists(dir + "/config.json")) {
        skip("model not downloaded — run scripts/download_model.sh");
    }
    return dir;
}

inline int finish(const char* name) {
    if (failures == 0) {
        std::printf("%s: %d checks passed\n", name, checks);
        return 0;
    }
    std::fprintf(stderr, "%s: %d/%d checks FAILED\n", name, failures, checks);
    return 1;
}

}  // namespace nano::testing
