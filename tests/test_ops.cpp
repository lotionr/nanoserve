// F015/F016: core fp32 ops vs numpy goldens, RoPE vs the HF Qwen2 reference.
//
// Goldens are committed (tests/data/ops_golden.json, rope_golden.json), so
// this test needs no model download and no Python — it always runs.
#include "core/ops.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "testing.hpp"

namespace {

std::vector<float> floats(const nano::json::Value& v) {
    std::vector<float> out;
    for (const auto& x : v.items()) {
        out.push_back(static_cast<float>(x.as_double()));
    }
    return out;
}

/// rel-tolerance compare against the golden value, with an absolute floor for
/// values near zero. Reports the worst element on failure.
void check_close(const std::vector<float>& got, const std::vector<float>& want,
                 float rtol, float atol, const char* what) {
    NANO_CHECK_MSG(got.size() == want.size(), "%s: size %zu vs %zu", what, got.size(),
                   want.size());
    if (got.size() != want.size()) {
        return;
    }
    float worst = 0.0f;
    size_t worst_i = 0;
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
        const float err = std::fabs(got[i] - want[i]);
        const float bound = atol + rtol * std::fabs(want[i]);
        if (err > bound) {
            ok = false;
        }
        if (err > worst) {
            worst = err;
            worst_i = i;
        }
    }
    NANO_CHECK_MSG(ok, "%s: worst |diff| %.3g at [%zu] (got %.8g, want %.8g)", what,
                   static_cast<double>(worst), worst_i,
                   static_cast<double>(got[worst_i]),
                   static_cast<double>(want[worst_i]));
}

constexpr float kRtol = 1e-5f;
constexpr float kAtol = 1e-6f;

}  // namespace

int main() {
    const nano::json::Value g =
        nano::json::parse(nano::json::read_file("tests/data/ops_golden.json"));

    {  // linear: y = x @ W^T + bias
        const auto& c = g.at("linear");
        const auto x = floats(c.at("x")), w = floats(c.at("w")), b = floats(c.at("bias"));
        const auto want = floats(c.at("y"));
        const int64_t tokens = c.at("tokens").as_int();
        const int64_t d_in = c.at("d_in").as_int();
        const int64_t d_out = c.at("d_out").as_int();

        std::vector<float> y(static_cast<size_t>(tokens * d_out));
        nano::ops::linear(x.data(), w.data(), b.data(), y.data(), tokens, d_in, d_out);
        check_close(y, want, kRtol, kAtol, "linear");

        // bias == nullptr path: same result minus the bias
        nano::ops::linear(x.data(), w.data(), nullptr, y.data(), tokens, d_in, d_out);
        std::vector<float> want_nobias = want;
        for (int64_t t = 0; t < tokens; ++t) {
            for (int64_t o = 0; o < d_out; ++o) {
                want_nobias[static_cast<size_t>(t * d_out + o)] -=
                    b[static_cast<size_t>(o)];
            }
        }
        check_close(y, want_nobias, 1e-4f, 1e-5f, "linear(no bias)");
    }

    {  // matmul: c = a @ b
        const auto& c = g.at("matmul");
        const auto a = floats(c.at("a")), b = floats(c.at("b"));
        const auto want = floats(c.at("c"));
        const int64_t m = c.at("m").as_int(), k = c.at("k").as_int(),
                      n = c.at("n").as_int();
        std::vector<float> out(static_cast<size_t>(m * n));
        nano::ops::matmul(a.data(), b.data(), out.data(), m, k, n);
        check_close(out, want, kRtol, kAtol, "matmul");
    }

    {  // rmsnorm at model width
        const auto& c = g.at("rmsnorm");
        const auto x = floats(c.at("x")), w = floats(c.at("weight"));
        const auto want = floats(c.at("y"));
        const int64_t tokens = c.at("tokens").as_int(), dim = c.at("dim").as_int();
        const float eps = static_cast<float>(c.at("eps").as_double());
        std::vector<float> y(x.size());
        nano::ops::rmsnorm(x.data(), w.data(), y.data(), tokens, dim, eps);
        check_close(y, want, kRtol, kAtol, "rmsnorm");
    }

    {  // softmax (in place)
        const auto& c = g.at("softmax");
        auto x = floats(c.at("x"));
        const auto want = floats(c.at("y"));
        nano::ops::softmax(x.data(), static_cast<int64_t>(x.size()));
        check_close(x, want, kRtol, kAtol, "softmax");
        double sum = 0.0;
        for (float v : x) {
            sum += static_cast<double>(v);
        }
        NANO_CHECK_MSG(std::fabs(sum - 1.0) < 1e-5, "softmax sum = %.8f", sum);
    }

    {  // silu (in place)
        const auto& c = g.at("silu");
        auto x = floats(c.at("x"));
        const auto want = floats(c.at("y"));
        nano::ops::silu(x.data(), static_cast<int64_t>(x.size()));
        check_close(x, want, kRtol, kAtol, "silu");
    }

    {  // elementwise add / mul
        const auto a = floats(g.at("add").at("a")), b = floats(g.at("add").at("b"));
        std::vector<float> y(a.size());
        nano::ops::add(a.data(), b.data(), y.data(), static_cast<int64_t>(a.size()));
        check_close(y, floats(g.at("add").at("y")), kRtol, kAtol, "add");
        nano::ops::mul(a.data(), b.data(), y.data(), static_cast<int64_t>(a.size()));
        check_close(y, floats(g.at("mul").at("y")), kRtol, kAtol, "mul");
    }

    {  // F016: RoPE vs the HF Qwen2 implementation, q (14 heads) and k (2 heads)
        const nano::json::Value r =
            nano::json::parse(nano::json::read_file("tests/data/rope_golden.json"));
        const float theta = static_cast<float>(r.at("rope_theta").as_double());
        const int64_t head_dim = r.at("head_dim").as_int();
        const auto& positions = r.at("positions").items();

        const struct {
            const char* in;
            const char* out;
            const char* heads_key;
        } cases[] = {{"q_in", "q_out", "n_heads"}, {"k_in", "k_out", "n_kv_heads"}};
        for (const auto& tc : cases) {
            const int64_t n_heads = r.at(tc.heads_key).as_int();
            for (size_t t = 0; t < positions.size(); ++t) {
                auto x = floats(r.at(tc.in).items()[t]);
                const auto want = floats(r.at(tc.out).items()[t]);
                const int64_t pos = positions[t].as_int();
                nano::ops::rope(x.data(), n_heads, head_dim, pos, theta);
                std::string what = std::string("rope ") + tc.in + " pos " +
                                   std::to_string(pos);
                check_close(x, want, kRtol, kAtol, what.c_str());
            }
        }
    }

    {  // F025: threaded linear/matmul are bit-identical to single-threaded.
        // Shapes above the parallel threshold, with a d_out that doesn't
        // divide evenly by any thread count, so chunk boundaries are odd.
        std::mt19937 gen(20260818);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        const int64_t tokens = 3, d_in = 896, d_out = 2047;
        std::vector<float> x(static_cast<size_t>(tokens * d_in));
        std::vector<float> w(static_cast<size_t>(d_out * d_in));
        std::vector<float> bias(static_cast<size_t>(d_out));
        for (auto* v : {&x, &w, &bias}) {
            for (float& f : *v) {
                f = dist(gen);
            }
        }

        std::vector<float> serial(static_cast<size_t>(tokens * d_out));
        nano::ops::set_num_threads(1);
        nano::ops::linear(x.data(), w.data(), bias.data(), serial.data(), tokens, d_in,
                          d_out);

        for (int threads : {2, 5, 0}) {  // 0 = hardware concurrency
            std::vector<float> pooled(static_cast<size_t>(tokens * d_out), -1.0f);
            nano::ops::set_num_threads(threads);
            nano::ops::linear(x.data(), w.data(), bias.data(), pooled.data(), tokens,
                              d_in, d_out);
            NANO_CHECK_MSG(std::memcmp(serial.data(), pooled.data(),
                                       serial.size() * sizeof(float)) == 0,
                           "threaded linear (threads=%d) not bit-identical", threads);
        }

        // matmul splits over rows; 131 rows won't divide evenly either.
        const int64_t m = 131, k = 64, n = 96;
        std::vector<float> a(static_cast<size_t>(m * k));
        std::vector<float> b(static_cast<size_t>(k * n));
        for (auto* v : {&a, &b}) {
            for (float& f : *v) {
                f = dist(gen);
            }
        }
        std::vector<float> c_serial(static_cast<size_t>(m * n));
        std::vector<float> c_pooled(static_cast<size_t>(m * n), -1.0f);
        nano::ops::set_num_threads(1);
        nano::ops::matmul(a.data(), b.data(), c_serial.data(), m, k, n);
        nano::ops::set_num_threads(0);
        nano::ops::matmul(a.data(), b.data(), c_pooled.data(), m, k, n);
        NANO_CHECK_MSG(std::memcmp(c_serial.data(), c_pooled.data(),
                                   c_serial.size() * sizeof(float)) == 0,
                       "threaded matmul not bit-identical");
    }

    return nano::testing::finish("test_ops");
}
