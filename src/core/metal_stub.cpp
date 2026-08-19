// Non-Apple stand-in for metal.mm: the backend simply does not exist.
// ops::set_backend(Backend::metal) sees available() == false and refuses,
// so the linear entry points below are unreachable in normal use — they
// throw rather than return garbage if someone calls them anyway.
#include "core/metal.hpp"

#include <stdexcept>

namespace nano::metal {

bool available() { return false; }

const char* device_name() { return ""; }

void register_weights(const void*, size_t) {}

void unregister_weights(const void*) {}

void linear_f32(const float*, const float*, const float*, float*, int64_t,
                int64_t, int64_t) {
    throw std::runtime_error("metal backend not built on this platform");
}

void linear_q8(const float*, const int8_t*, const float*, const float*, float*,
               int64_t, int64_t, int64_t) {
    throw std::runtime_error("metal backend not built on this platform");
}

}  // namespace nano::metal
