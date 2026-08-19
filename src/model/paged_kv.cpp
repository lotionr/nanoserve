#include "model/paged_kv.hpp"

#include <stdexcept>
#include <string>

namespace nano {

PagePool::PagePool(const ModelConfig& config, int64_t page_size, int64_t max_pages)
    : page_size_(page_size),
      kv_dim_(config.num_kv_heads * config.head_dim),
      layer_stride_(2 * page_size * kv_dim_),  // K rows then V rows
      floats_per_page_(config.num_layers * layer_stride_),
      max_pages_(max_pages) {
    if (page_size <= 0 || max_pages <= 0) {
        throw std::runtime_error("PagePool: page_size and max_pages must be positive");
    }
    pages_.reserve(static_cast<size_t>(max_pages));
    free_.reserve(static_cast<size_t>(max_pages));
}

int32_t PagePool::alloc() {
    if (!free_.empty()) {
        const int32_t page = free_.back();
        free_.pop_back();
        return page;
    }
    if (pages_backed() == max_pages_) {
        throw std::runtime_error("PagePool: exhausted (" + std::to_string(max_pages_) +
                                 " pages of " + std::to_string(page_size_) +
                                 " tokens all live)");
    }
    // Mint a new id. The buffer is allocated BEFORE the id is returned, so a
    // page id in the wild always has memory behind it (see header note).
    pages_.emplace_back(static_cast<size_t>(floats_per_page_));
    return static_cast<int32_t>(pages_.size() - 1);
}

void PagePool::free_page(int32_t page) {
    if (page < 0 || static_cast<int64_t>(page) >= pages_backed()) {
        throw std::runtime_error("PagePool: free of unknown page " + std::to_string(page));
    }
    free_.push_back(page);
}

PagedKvCache::PagedKvCache(PagePool& pool, int64_t max_seq)
    : pool_(pool), page_size_(pool.page_size()), max_seq_(max_seq) {
    block_table_.reserve(
        static_cast<size_t>((max_seq + page_size_ - 1) / page_size_));
}

void PagedKvCache::prepare(int64_t pos0, int64_t tokens) {
    const int64_t end = pos0 + tokens;  // first position NOT written
    if (end > max_seq_) {
        throw std::runtime_error("PagedKvCache: sequence exceeds max_seq (" +
                                 std::to_string(max_seq_) + " tokens)");
    }
    const int64_t blocks_needed = (end + page_size_ - 1) / page_size_;
    while (static_cast<int64_t>(block_table_.size()) < blocks_needed) {
        block_table_.push_back(pool_.alloc());
    }
}

void PagedKvCache::release() {
    for (const int32_t page : block_table_) {
        pool_.free_page(page);
    }
    block_table_.clear();
}

}  // namespace nano
