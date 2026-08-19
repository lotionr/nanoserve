// F034, allocator half: the page allocator survives alloc/free churn.
//
// The invariant under attack is the one that breaks first in real paged-KV
// systems (it is exactly the failure mode of kvcached issue #437): an
// allocator that hands out a page id with no memory behind it, or hands the
// same page to two live sequences. Neither corrupts anything immediately —
// the damage surfaces later as silently wrong attention over someone else's
// K/V rows. So this test doesn't just exercise the API; after every phase it
// re-reads EVERY row every live sequence owns and checks the exact pattern
// that sequence wrote, which fails loudly on aliasing, stale block tables,
// and unbacked pages alike.
//
// Runs without model weights (synthetic config), so CI always executes it.
#include "model/paged_kv.hpp"

#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "model/config.hpp"
#include "testing.hpp"

namespace {

// Small on purpose: a 6-page pool fills up fast, so the churn loop spends
// most of its time in the interesting regime (reuse and exhaustion).
constexpr int64_t kPageSize = 4;
constexpr int64_t kMaxPages = 6;
constexpr int64_t kMaxSeqPerCache = 16;  // 4 pages if fully grown

nano::ModelConfig tiny_config() {
    nano::ModelConfig c;
    c.num_layers = 3;
    c.num_kv_heads = 2;
    c.head_dim = 4;  // kv_dim = 8 floats per row
    return c;
}

/// The pattern sequence `seq_id` writes at (layer, pos, dim). Any collision —
/// two sequences on one page, a block table pointing at a freed-and-reused
/// page — makes some read disagree with the owner's formula.
float pattern(int seq_id, int64_t layer, int64_t pos, int64_t d, bool v_side) {
    return static_cast<float>(seq_id * 100000 + layer * 10000 + pos * 100 +
                              d * 2 + (v_side ? 1 : 0));
}

/// One simulated sequence: a PagedKvCache plus the length it has "written".
struct Seq {
    explicit Seq(nano::PagePool& pool) : cache(pool, kMaxSeqPerCache) {}
    nano::PagedKvCache cache;
    int64_t len = 0;
};

void write_rows(Seq& s, int seq_id, const nano::ModelConfig& cfg, int64_t pos0,
                int64_t tokens) {
    s.cache.prepare(pos0, tokens);  // may throw: caller handles exhaustion
    const int64_t kv_dim = cfg.num_kv_heads * cfg.head_dim;
    for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
        for (int64_t pos = pos0; pos < pos0 + tokens; ++pos) {
            float* k = s.cache.k_row(layer, pos);
            float* v = s.cache.v_row(layer, pos);
            for (int64_t d = 0; d < kv_dim; ++d) {
                k[d] = pattern(seq_id, layer, pos, d, /*v_side=*/false);
                v[d] = pattern(seq_id, layer, pos, d, /*v_side=*/true);
            }
        }
    }
    s.len = pos0 + tokens;
}

/// Re-reads everything `seq_id` ever wrote. Returns mismatch count.
int64_t verify_rows(Seq& s, int seq_id, const nano::ModelConfig& cfg) {
    const int64_t kv_dim = cfg.num_kv_heads * cfg.head_dim;
    int64_t bad = 0;
    for (int64_t layer = 0; layer < cfg.num_layers; ++layer) {
        for (int64_t pos = 0; pos < s.len; ++pos) {
            const float* k = s.cache.k_row(layer, pos);
            const float* v = s.cache.v_row(layer, pos);
            for (int64_t d = 0; d < kv_dim; ++d) {
                bad += k[d] != pattern(seq_id, layer, pos, d, false) ? 1 : 0;
                bad += v[d] != pattern(seq_id, layer, pos, d, true) ? 1 : 0;
            }
        }
    }
    return bad;
}

}  // namespace

int main() {
    const nano::ModelConfig cfg = tiny_config();

    // ---- Part A: the allocator's basic contract ---------------------------
    {
        nano::PagePool pool(cfg, kPageSize, kMaxPages);
        std::vector<int32_t> ids;
        for (int64_t i = 0; i < kMaxPages; ++i) {
            const int32_t page = pool.alloc();
            // THE invariant: an id is only ever handed out with memory
            // already behind it (never beyond the backed watermark).
            NANO_CHECK_MSG(page < pool.pages_backed(),
                           "alloc returned unbacked page %d (backed: %lld)", page,
                           static_cast<long long>(pool.pages_backed()));
            for (const int32_t seen : ids) {
                NANO_CHECK_MSG(page != seen, "page %d handed out twice", page);
            }
            ids.push_back(page);
        }
        NANO_CHECK(pool.pages_live() == kMaxPages);
        NANO_CHECK_THROWS(pool.alloc());  // exhausted, must refuse loudly

        // Free two, realloc two: memory is REUSED (backed stays flat), and
        // LIFO order means the most recently freed page comes back first.
        pool.free_page(ids[1]);
        pool.free_page(ids[4]);
        NANO_CHECK(pool.pages_live() == kMaxPages - 2);
        NANO_CHECK(pool.alloc() == ids[4]);
        NANO_CHECK(pool.alloc() == ids[1]);
        NANO_CHECK_MSG(pool.pages_backed() == kMaxPages,
                       "free/realloc grew the pool: %lld pages backed",
                       static_cast<long long>(pool.pages_backed()));
        NANO_CHECK_THROWS(pool.free_page(-1));
        NANO_CHECK_THROWS(pool.free_page(static_cast<int32_t>(kMaxPages)));
    }

    // ---- Part B: block-table mechanics on one sequence --------------------
    {
        nano::PagePool pool(cfg, kPageSize, kMaxPages);
        Seq s(pool);
        NANO_CHECK(s.cache.pages_held() == 0);  // nothing preallocated
        write_rows(s, /*seq_id=*/7, cfg, 0, 3);  // inside page 0
        NANO_CHECK(s.cache.pages_held() == 1);
        write_rows(s, 7, cfg, 3, 2);  // crosses into page 1
        NANO_CHECK(s.cache.pages_held() == 2);
        NANO_CHECK(verify_rows(s, 7, cfg) == 0);
        NANO_CHECK_THROWS(s.cache.prepare(0, kMaxSeqPerCache + 1));  // > max_seq
        s.cache.release();
        NANO_CHECK(s.cache.pages_held() == 0);
        NANO_CHECK(pool.pages_live() == 0);  // everything came back
    }

    // ---- Part C: multi-sequence churn (the kvcached #437 shape) -----------
    // Random interleaving of grow / reset across 3 sequences sharing one
    // 6-page pool: 3 fully-grown sequences need 12 pages, so the pool is
    // constantly exhausted and every grow races against reuse of just-freed
    // pages. After every operation, every live sequence's every row must
    // still read back as its own pattern.
    {
        nano::PagePool pool(cfg, kPageSize, kMaxPages);
        Seq s0(pool);
        Seq s1(pool);
        Seq s2(pool);
        Seq* const seqs[3] = {&s0, &s1, &s2};

        std::mt19937 rng(1234);  // fixed seed: failures must reproduce
        int64_t grows = 0;
        int64_t resets = 0;
        int64_t exhaustions = 0;
        for (int64_t step = 0; step < 2000; ++step) {
            const int i = static_cast<int>(rng() % 3);
            Seq& s = *seqs[i];
            const bool reset = (rng() % 8) == 0 && s.len > 0;
            if (reset) {
                s.cache.release();
                s.len = 0;
                ++resets;
            } else {
                const int64_t tokens =
                    std::min<int64_t>(1 + static_cast<int64_t>(rng() % 5),
                                      kMaxSeqPerCache - s.len);
                if (tokens == 0) {  // this sequence is full: recycle it
                    s.cache.release();
                    s.len = 0;
                    ++resets;
                    continue;
                }
                try {
                    write_rows(s, i, cfg, s.len, tokens);
                    ++grows;
                } catch (const std::runtime_error&) {
                    // Pool exhausted mid-grow: "evict" this sequence (free its
                    // pages) and keep churning — the recovery a scheduler
                    // would do. Rows written before the throw are dropped
                    // with the sequence, so no partial state leaks.
                    s.cache.release();
                    s.len = 0;
                    ++exhaustions;
                }
            }

            // Pool accounting must balance every single step.
            int64_t held = 0;
            for (Seq* q : seqs) {
                held += q->cache.pages_held();
            }
            NANO_CHECK_MSG(pool.pages_live() == held,
                           "step %lld: pool says %lld live, sequences hold %lld",
                           static_cast<long long>(step),
                           static_cast<long long>(pool.pages_live()),
                           static_cast<long long>(held));
            NANO_CHECK(pool.pages_backed() <= kMaxPages);

            // Every live row still belongs to whoever wrote it.
            for (int j = 0; j < 3; ++j) {
                const int64_t bad = verify_rows(*seqs[j], j, cfg);
                NANO_CHECK_MSG(bad == 0,
                               "step %lld: seq %d has %lld corrupted values",
                               static_cast<long long>(step), j,
                               static_cast<long long>(bad));
                if (bad != 0) {
                    return nano::testing::finish("test_paged_alloc");  // stop the flood
                }
            }
        }
        std::printf("churn: 2000 steps, %lld grows, %lld resets, %lld pool "
                    "exhaustions recovered; backed high-water %lld/%lld pages\n",
                    static_cast<long long>(grows), static_cast<long long>(resets),
                    static_cast<long long>(exhaustions),
                    static_cast<long long>(pool.pages_backed()),
                    static_cast<long long>(kMaxPages));
        NANO_CHECK_MSG(exhaustions > 0,
                       "churn never exhausted the pool — the test is too gentle "
                       "to mean anything");
    }

    return nano::testing::finish("test_paged_alloc");
}
