// Continuous batching (F036): several sequences decode together in one
// forward pass, so every weight matrix is read once per step instead of once
// per sequence.
//
// Why this is where serving throughput comes from: a single-sequence decode
// step is a chain of GEMVs — for each projection, every weight byte is read
// to produce one output row, so the step is memory-bandwidth-bound on weight
// traffic (the F029 bench measured exactly that). Stack the current token of
// n live sequences into an [n, hidden] matrix and those GEMVs become one
// GEMM: the same weight read now serves n rows, so aggregate tokens/s rises
// roughly with n while per-step latency barely moves. That conversion is the
// entire trick; everything else in this file is bookkeeping to keep each
// sequence's state separate:
//
//   - Sequence      = one request's PagedKvCache + its length. All sequences
//                     draw pages from the engine's shared PagePool — the
//                     pool was built for exactly this (F034).
//   - prefill()     = one sequence's whole prompt, alone. Same code path as
//                     the single-sequence Engine (layer_forward), so a
//                     prompt's KV and first-token logits are bit-identical
//                     to what Engine would produce.
//   - decode_step() = ONE token for EACH live sequence. The projections and
//                     the lm_head batch [n, ...] through the GEMM path; RoPE
//                     and attention run per row, each against its own
//                     sequence's cache at its own position.
//
// Determinism contract: a sequence decoded in a batch produces bit-identical
// tokens to the same sequence decoded alone. It holds because ops::linear
// computes each output row with the identical serial loop whatever `tokens`
// is (asserted bit-for-bit in test_ops), rmsnorm/rope are row-local, and
// attention reads only the row's own cache. test_batch asserts the
// end-to-end consequence: batched greedy tokens == solo greedy tokens.
//
// What this deliberately is not (vLLM has all of it; F036 doesn't need it):
// no chunked/interleaved prefill (a new request's prompt runs whole, between
// decode steps — bounded by max_seq, and honest about favoring simplicity
// over tail latency), no scheduling policy beyond FIFO admission, and no
// preemption — the server admits at most max_seqs sequences, which the pool
// is sized to fit, so pool exhaustion is unreachable rather than handled.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "model/paged_kv.hpp"
#include "model/qwen2.hpp"

namespace nano {

class BatchEngine;

/// One row of a batched decode step: whose cache, and which position the
/// row's token occupies there. (batch.cpp's layer_forward_batch — the
/// batched twin of layer_forward — takes a span of these.)
struct DecodeRow {
    PagedKvCache* cache = nullptr;
    int64_t pos = 0;
};

/// One live sequence: its paged KV cache plus how many tokens it holds.
/// Created by BatchEngine::new_sequence(); destroying it returns its pages
/// to the engine's shared pool (that is the whole "free a finished request"
/// story). Must not outlive its engine.
class Sequence {
public:
    ~Sequence() = default;
    Sequence(const Sequence&) = delete;  // pages are single-owner
    Sequence& operator=(const Sequence&) = delete;

    int64_t seq_len() const { return len_; }

private:
    friend class BatchEngine;
    Sequence(PagePool& pool, int64_t max_seq) : cache_(pool, max_seq) {}

    PagedKvCache cache_;
    int64_t len_ = 0;  // tokens already in the cache
};

/// The model plus a shared page pool sized for `max_seqs` full-length
/// sequences. CPU-only by design: the F033 result is that the GPU loses
/// decode at this model size, and batching multiplies exactly the CPU GEMM
/// path's strength. (With the metal backend selected the math would still be
/// correct — weights just stage per call, slowly, because no Engine
/// registered them.)
class BatchEngine {
public:
    explicit BatchEngine(const std::string& model_dir, int64_t max_seq = 2048,
                         const std::string& weights_file = "",
                         int64_t max_seqs = 4);

    /// A fresh, empty sequence drawing pages from the shared pool. The
    /// engine does not track it — the caller decides which sequences join
    /// each decode_step() and drops the pointer to free the pages.
    std::unique_ptr<Sequence> new_sequence();

    /// Appends `ids` to one sequence and runs the forward pass over them —
    /// the whole prompt in one call, exactly like Engine::forward. Returns
    /// the last token's logits ([vocab_size]), valid until the next
    /// prefill/decode_step call.
    std::span<const float> prefill(Sequence& seq, std::span<const int32_t> ids);

    /// One decode step for the whole batch: appends tokens[i] to seqs[i] and
    /// returns logits for EVERY row ([seqs.size() * vocab_size], row i at
    /// i * vocab_size) — unlike prefill, every sequence needs its next-token
    /// distribution. Valid until the next prefill/decode_step call.
    std::span<const float> decode_step(std::span<Sequence* const> seqs,
                                       std::span<const int32_t> tokens);

    const Qwen2Model& model() const { return model_; }
    int64_t max_seq() const { return max_seq_; }
    int64_t max_seqs() const { return max_seqs_; }
    /// Backed-pages high-water mark across all sequences (see PagePool).
    int64_t kv_bytes_allocated() const { return pool_.bytes_backed(); }

private:
    Qwen2Model model_;
    int64_t max_seq_ = 0;
    int64_t max_seqs_ = 0;
    PagePool pool_;
    Scratch scratch_;             // shared per-layer work buffers (F024 style)
    std::vector<DecodeRow> rows_;  // [n] per decode_step; reused, never shrinks
    std::vector<float> hidden_;   // [rows, hidden] for the current call
    std::vector<float> logits_;   // [max_seqs * vocab]
};

}  // namespace nano
