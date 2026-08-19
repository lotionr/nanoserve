// Paged KV cache (F034): fixed-size pages + a free-list allocator, so a
// sequence's KV memory grows with its actual length instead of being
// preallocated at max context.
//
// ---------------------------------------------------------------------------
// Design note: mapping onto vLLM's PagedAttention vocabulary
// ---------------------------------------------------------------------------
// This is a deliberately small CPU re-derivation of the PagedAttention memory
// model (Kwon et al., SOSP 2023). Name-for-name:
//
//   vLLM                          nanoserve
//   ----------------------------  ------------------------------------------
//   KV block (block_size tokens)  page (kPageSizeDefault = 16 tokens, same
//                                 default as vLLM's block_size)
//   block allocator / KV pool     PagePool — owns the physical pages and a
//                                 LIFO free list, shared by every sequence
//   block table (per sequence)    PagedKvCache::block_table_ — logical block
//                                 index -> physical page id
//   slot mapping                  k_row()/v_row(): page = table[pos / P],
//                                 row = pos % P — the same address arithmetic
//                                 vLLM's reshape_and_cache kernel computes
//   physical KV tensor            one heap buffer per page holding all
//                                 layers' K and V rows for its 16 positions
//                                 (vLLM shapes this per layer instead; the
//                                 block id is shared across layers in both)
//
// The point of the indirection is identical to the paper's: (a) no
// max-context preallocation per sequence — memory high-water tracks actual
// tokens, (b) freed pages are immediately reusable by other sequences, so a
// serving workload's fragmentation is bounded by page_size, not max_seq.
//
// Deliberate simplifications (each is real machinery in vLLM that this repo
// doesn't need yet):
//   - No prefix sharing / copy-on-write: every page has exactly one owner,
//     so there are no reference counts and freeing is unconditional.
//   - No eviction policy: the free list is plain LIFO. vLLM orders evictable
//     blocks (LRU) because cached prefixes might be reused; with no prefix
//     cache there is nothing to rank.
//   - No preemption or CPU swap: when the pool is exhausted, alloc() throws.
//     A scheduler that recomputes or swaps victims is a serving-layer
//     concern (F036 territory), not a cache concern.
//   - Pages are eagerly backed: alloc() returns a page id only after the
//     memory behind it exists, so "allocated but unbacked" states are
//     unrepresentable by construction. Systems that reserve virtual address
//     space up front and back pages on demand (e.g. Sky Computing's kvcached)
//     must instead enforce this as a runtime invariant — the exact failure
//     mode where an allocator hands out block ids past the backed watermark.
//     test_paged_alloc churns alloc/free across interleaved sequences and
//     asserts the invariant on every allocation.
//
// Correctness argument: attention reads K/V one position-row at a time, and a
// row's contents never depend on where it is stored. Paging only changes the
// addresses of rows, not the values or the order they are combined in, so
// paged logits must be BIT-IDENTICAL to the contiguous cache's — and
// test_kvcache asserts exactly that, across page boundaries.
#pragma once

#include <cstdint>
#include <vector>

#include "model/config.hpp"

namespace nano {

/// vLLM's default block_size; small enough that a short sequence wastes at
/// most 15 rows per layer, big enough that the block table stays tiny.
inline constexpr int64_t kPageSizeDefault = 16;

/// Owns the physical KV pages and the free list. Shared by every sequence
/// (PagedKvCache) that lives on it; must outlive all of them.
///
/// One page = one heap buffer holding `page_size` K rows and `page_size` V
/// rows for EVERY layer: [layer][K|V][page_size * kv_dim] floats. Buffers
/// are backed lazily (first time a page id is minted) but never freed until
/// the pool dies — a page returned to the free list keeps its memory and is
/// handed to the next alloc() caller. pages_backed() is therefore the
/// all-time high-water mark of pages in use.
class PagePool {
public:
    PagePool(const ModelConfig& config, int64_t page_size, int64_t max_pages);

    /// Returns a page id whose memory is already backed. Prefers the free
    /// list (reuse); mints and backs a new page otherwise. Throws when
    /// max_pages are all live — exhaustion is the caller's problem to
    /// schedule around, never a silent out-of-bounds page.
    int32_t alloc();

    /// Returns a page to the free list. The caller promises it holds no
    /// other reference (single-owner pages; no refcounts — see header note).
    void free_page(int32_t page);

    float* k_page(int64_t layer, int32_t page) {
        return pages_[static_cast<size_t>(page)].data() + layer * layer_stride_;
    }
    float* v_page(int64_t layer, int32_t page) {
        return k_page(layer, page) + page_size_ * kv_dim_;
    }

    int64_t page_size() const { return page_size_; }
    int64_t kv_dim() const { return kv_dim_; }
    int64_t max_pages() const { return max_pages_; }
    int64_t pages_live() const { return pages_backed() - pages_free(); }
    int64_t pages_free() const { return static_cast<int64_t>(free_.size()); }
    /// High-water mark: pages that have real memory behind them.
    int64_t pages_backed() const { return static_cast<int64_t>(pages_.size()); }
    /// Bytes actually allocated (the number the memory test prints).
    int64_t bytes_backed() const {
        return pages_backed() * floats_per_page_ * static_cast<int64_t>(sizeof(float));
    }

private:
    int64_t page_size_ = 0;      // tokens per page
    int64_t kv_dim_ = 0;         // floats per K (or V) row
    int64_t layer_stride_ = 0;   // floats per layer within a page (K + V)
    int64_t floats_per_page_ = 0;
    int64_t max_pages_ = 0;
    // Inner buffers never resize after creation, so row pointers stay valid
    // even while push_back moves the (empty-capacity-preserving) outer slots.
    std::vector<std::vector<float>> pages_;
    std::vector<int32_t> free_;  // LIFO: most-recently-freed page reused first
};

/// One sequence's view of the pool: a block table from logical position to
/// physical page. Presents the same k_row/v_row/max_seq/prepare interface as
/// the contiguous KvCache, so layer_forward is templated over the two rather
/// than paying a virtual call in the attention inner loop.
class PagedKvCache {
public:
    /// `max_seq` caps this sequence (checked in prepare); the pool caps
    /// total memory across all sequences.
    PagedKvCache(PagePool& pool, int64_t max_seq);

    /// Pages go back to the pool when the sequence dies.
    ~PagedKvCache() { release(); }
    PagedKvCache(const PagedKvCache&) = delete;  // pages are single-owner
    PagedKvCache& operator=(const PagedKvCache&) = delete;

    /// Ensures pages exist for positions [0, pos0 + tokens). The engine only
    /// ever appends, so this just extends the block table; it throws if the
    /// sequence would exceed max_seq or the pool is exhausted.
    void prepare(int64_t pos0, int64_t tokens);

    /// Returns every page to the pool (Engine::reset for the paged path).
    void release();

    int64_t max_seq() const { return max_seq_; }
    int64_t pages_held() const { return static_cast<int64_t>(block_table_.size()); }

    float* k_row(int64_t layer, int64_t pos) {
        const int32_t page = block_table_[static_cast<size_t>(pos / page_size_)];
        return pool_.k_page(layer, page) + (pos % page_size_) * pool_.kv_dim();
    }
    float* v_row(int64_t layer, int64_t pos) {
        const int32_t page = block_table_[static_cast<size_t>(pos / page_size_)];
        return pool_.v_page(layer, page) + (pos % page_size_) * pool_.kv_dim();
    }

private:
    PagePool& pool_;
    int64_t page_size_ = 0;  // copied from the pool: hot-path locality
    int64_t max_seq_ = 0;
    std::vector<int32_t> block_table_;  // logical block index -> page id
};

}  // namespace nano
