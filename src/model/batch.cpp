#include "model/batch.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "core/ops.hpp"

namespace nano {

namespace {

/// The batched twin of layer_forward (qwen2.cpp): the same math over
/// [n, hidden], but each row is one decode token from a DIFFERENT sequence
/// at its own position, instead of consecutive positions of one sequence.
///
/// Reading the two side by side: everywhere layer_forward derives a row's
/// position as pos0 + t and touches `cache`, this derives it as rows[i].pos
/// and touches rows[i].cache. The projections and the MLP batch over all n
/// rows (the GEMM path — the point of F036); k/v stay one GEMV per row
/// because each result lands in a different sequence's cache (and they are
/// the small matrices — batching pays on q/o/gate/up/down and the lm_head).
void layer_forward_batch(const Qwen2Model& model, int64_t layer_idx, float* hidden,
                         std::span<const DecodeRow> rows, Scratch& scratch) {
    const ModelConfig& c = model.config;
    const LayerWeights& w = model.layers[static_cast<size_t>(layer_idx)];
    const int64_t n = static_cast<int64_t>(rows.size());
    const int64_t H = c.hidden_size;
    const int64_t D = c.head_dim;
    const int64_t q_dim = c.num_heads * D;
    // GQA: this many consecutive query heads share one k/v head.
    const int64_t group = c.num_heads / c.num_kv_heads;
    const float eps = static_cast<float>(c.rms_norm_eps);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const float theta = static_cast<float>(c.rope_theta);

    // scores must fit the longest row's context; every cache shares max_seq.
    scratch.ensure(c, n, rows[0].cache->max_seq());
    std::vector<float>& normed = scratch.normed;
    std::vector<float>& q = scratch.q;
    std::vector<float>& attn = scratch.attn;
    std::vector<float>& proj = scratch.proj;
    std::vector<float>& scores = scratch.scores;

    // --- attention block: hidden += o_proj(attend(rope(qkv(norm(hidden))))) ---
    ops::rmsnorm(hidden, w.input_norm.data(), normed.data(), n, H, eps);
    weight_linear(w.q_w, normed.data(), w.q_b.data(), q.data(), n);  // one GEMM
    for (int64_t i = 0; i < n; ++i) {
        // K and V go straight into row i's own cache at its own position.
        const float* in = normed.data() + i * H;
        PagedKvCache& cache = *rows[static_cast<size_t>(i)].cache;
        const int64_t pos = rows[static_cast<size_t>(i)].pos;
        weight_linear(w.k_w, in, w.k_b.data(), cache.k_row(layer_idx, pos), 1);
        weight_linear(w.v_w, in, w.v_b.data(), cache.v_row(layer_idx, pos), 1);
        ops::rope(q.data() + i * q_dim, c.num_heads, D, pos, theta);
        ops::rope(cache.k_row(layer_idx, pos), c.num_kv_heads, D, pos, theta);
    }

    for (int64_t i = 0; i < n; ++i) {
        // Row i attends over ITS sequence's positions [0, pos] and nothing
        // else — cross-request isolation is this loop reading only
        // rows[i].cache (test_batch's contamination check targets it).
        PagedKvCache& cache = *rows[static_cast<size_t>(i)].cache;
        const int64_t n_ctx = rows[static_cast<size_t>(i)].pos + 1;
        for (int64_t h = 0; h < c.num_heads; ++h) {
            const float* q_head = q.data() + i * q_dim + h * D;
            const int64_t kv_off = (h / group) * D;  // shared k/v head (GQA)

            for (int64_t s = 0; s < n_ctx; ++s) {
                scores[static_cast<size_t>(s)] =
                    ops::dot(q_head, cache.k_row(layer_idx, s) + kv_off, D) * scale;
            }
            ops::softmax(scores.data(), n_ctx);

            float* out = attn.data() + i * q_dim + h * D;
            std::memset(out, 0, sizeof(float) * static_cast<size_t>(D));
            for (int64_t s = 0; s < n_ctx; ++s) {
                const float* v_head = cache.v_row(layer_idx, s) + kv_off;
                const float weight = scores[static_cast<size_t>(s)];
                for (int64_t d = 0; d < D; ++d) {
                    out[d] += weight * v_head[d];
                }
            }
        }
    }
    weight_linear(w.o_w, attn.data(), nullptr, proj.data(), n);
    ops::add(hidden, proj.data(), hidden, n * H);

    // --- MLP block: hidden += down(silu(gate(norm(hidden))) * up(norm(hidden))) ---
    const int64_t I = c.intermediate_size;
    std::vector<float>& gate = scratch.gate;
    std::vector<float>& up = scratch.up;
    ops::rmsnorm(hidden, w.post_norm.data(), normed.data(), n, H, eps);
    weight_linear(w.gate_w, normed.data(), nullptr, gate.data(), n);
    weight_linear(w.up_w, normed.data(), nullptr, up.data(), n);
    ops::silu(gate.data(), n * I);
    ops::mul(gate.data(), up.data(), gate.data(), n * I);
    weight_linear(w.down_w, gate.data(), nullptr, proj.data(), n);
    ops::add(hidden, proj.data(), hidden, n * H);
}

}  // namespace

BatchEngine::BatchEngine(const std::string& model_dir, int64_t max_seq,
                         const std::string& weights_file, int64_t max_seqs)
    : model_(Qwen2Model::load(model_dir, weights_file)),
      max_seq_(max_seq),
      max_seqs_(max_seqs),
      // Sized so max_seqs sequences can all reach max_seq: a caller that caps
      // live sequences at max_seqs can never exhaust the pool. Pages are
      // backed lazily, so this worst case costs address space, not memory.
      pool_(model_.config, kPageSizeDefault,
            max_seqs * ((max_seq + kPageSizeDefault - 1) / kPageSizeDefault)) {
    if (max_seqs <= 0) {
        throw std::runtime_error("BatchEngine: max_seqs must be positive");
    }
    rows_.reserve(static_cast<size_t>(max_seqs));
    logits_.reserve(static_cast<size_t>(max_seqs * model_.config.vocab_size));
}

std::unique_ptr<Sequence> BatchEngine::new_sequence() {
    // Not make_unique: the constructor is private and BatchEngine is the
    // friend, not std::make_unique.
    return std::unique_ptr<Sequence>(new Sequence(pool_, max_seq_));
}

std::span<const float> BatchEngine::prefill(Sequence& seq,
                                            std::span<const int32_t> ids) {
    const ModelConfig& c = model_.config;
    const int64_t tokens = static_cast<int64_t>(ids.size());
    if (tokens == 0) {
        throw std::runtime_error("prefill() needs at least one token");
    }
    if (seq.len_ + tokens > max_seq_) {
        throw std::runtime_error("sequence exceeds KV cache capacity (" +
                                 std::to_string(max_seq_) + " tokens)");
    }
    seq.cache_.prepare(seq.len_, tokens);
    const int64_t H = c.hidden_size;

    hidden_.resize(static_cast<size_t>(tokens * H));
    embed_rows(model_, ids, hidden_.data());
    for (int64_t layer = 0; layer < c.num_layers; ++layer) {
        layer_forward(model_, layer, hidden_.data(), tokens, seq.len_, seq.cache_,
                      scratch_);
    }
    seq.len_ += tokens;

    // Only the last prompt token's hidden state becomes logits, exactly like
    // Engine::forward_with. lm_head is the embedding matrix (tied).
    float* normed = scratch_.final_normed.data();
    ops::rmsnorm(hidden_.data() + (tokens - 1) * H, model_.final_norm.data(),
                 normed, 1, H, static_cast<float>(c.rms_norm_eps));
    logits_.resize(static_cast<size_t>(c.vocab_size));
    weight_linear(model_.embed_tokens, normed, nullptr, logits_.data(), 1);
    return logits_;
}

std::span<const float> BatchEngine::decode_step(std::span<Sequence* const> seqs,
                                                std::span<const int32_t> tokens) {
    const ModelConfig& c = model_.config;
    const int64_t n = static_cast<int64_t>(seqs.size());
    if (n == 0 || tokens.size() != seqs.size()) {
        throw std::runtime_error("decode_step() needs one token per sequence");
    }
    if (n > max_seqs_) {
        throw std::runtime_error("decode_step: " + std::to_string(n) +
                                 " sequences, engine sized for " +
                                 std::to_string(max_seqs_));
    }
    const int64_t H = c.hidden_size;

    // Pages first (they can throw), then math — mirrors Engine::forward.
    rows_.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        Sequence& s = *seqs[static_cast<size_t>(i)];
        if (s.len_ + 1 > max_seq_) {
            throw std::runtime_error("sequence exceeds KV cache capacity (" +
                                     std::to_string(max_seq_) + " tokens)");
        }
        s.cache_.prepare(s.len_, 1);
        rows_[static_cast<size_t>(i)] = {&s.cache_, s.len_};
    }

    hidden_.resize(static_cast<size_t>(n * H));
    embed_rows(model_, tokens, hidden_.data());
    for (int64_t layer = 0; layer < c.num_layers; ++layer) {
        layer_forward_batch(model_, layer, hidden_.data(), rows_, scratch_);
    }
    for (Sequence* s : seqs) {
        s->len_ += 1;
    }

    // Unlike prefill, EVERY row's next token is wanted: final-norm all n rows
    // (rmsnorm is row-local) and run ONE lm_head GEMM. At [151936, 896] the
    // lm_head is the largest weight read of the step, so batching it is the
    // single biggest win of the whole feature.
    scratch_.ensure(c, n, max_seq_);
    ops::rmsnorm(hidden_.data(), model_.final_norm.data(), scratch_.normed.data(), n,
                 H, static_cast<float>(c.rms_norm_eps));
    logits_.resize(static_cast<size_t>(n * c.vocab_size));
    weight_linear(model_.embed_tokens, scratch_.normed.data(), nullptr,
                  logits_.data(), n);
    return logits_;
}

}  // namespace nano
