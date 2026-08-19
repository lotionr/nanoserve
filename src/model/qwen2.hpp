// Qwen2.5 model: weights, KV cache, and the fp32 forward pass.
//
// The design is deliberately plain (llama2.c lineage):
//   - Qwen2Model  = every weight widened to fp32 up front (~2 GB for 0.5B).
//   - KvCache     = per-layer K/V rows, contiguous [max_seq, kv_dim]
//                   (or PagedKvCache, paged_kv.hpp — grown page by page).
//   - layer_forward() = one transformer block over a span of new tokens.
//   - Engine      = embed -> 24 x layer_forward -> final norm -> lm_head.
//
// One forward() call handles both phases of inference: the first call feeds
// the whole prompt (prefill), later calls feed one token each (decode). Both
// go through the same code path — the KV cache is what makes decode O(1)
// per token instead of re-running the whole sequence.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "model/config.hpp"
#include "model/paged_kv.hpp"

namespace nano {

/// One weight matrix for linear(): fp32, or int8 with one fp32 scale per
/// output row (F027). Exactly one of f32 / q8 is populated; rows is d_out
/// and cols is d_in, matching the [out, in] safetensors layout.
struct Weight {
    int64_t rows = 0;
    int64_t cols = 0;
    std::vector<float> f32;      // fp32 mode: [rows * cols]
    std::vector<int8_t> q8;      // int8 mode: [rows * cols]
    std::vector<float> scales;   // int8 mode: [rows]

    bool quantized() const { return !q8.empty(); }
};

/// One transformer block's weights. Matrices may be fp32 or int8 (Weight);
/// norms and biases are tiny and always stay fp32.
/// Qwen2 quirk: q/k/v projections have biases; o_proj and the MLP do not.
struct LayerWeights {
    std::vector<float> input_norm;  // [hidden]
    Weight q_w;                     // [n_heads*head_dim, hidden]
    std::vector<float> q_b;         // [n_heads*head_dim]
    Weight k_w;                     // [n_kv_heads*head_dim, hidden]
    std::vector<float> k_b;
    Weight v_w;
    std::vector<float> v_b;
    Weight o_w;                     // [hidden, n_heads*head_dim]
    std::vector<float> post_norm;   // [hidden]
    Weight gate_w;                  // [intermediate, hidden]
    Weight up_w;                    // [intermediate, hidden]
    Weight down_w;                  // [hidden, intermediate]
};

/// All model weights. The fp32 checkpoint loads bf16 -> fp32 (the honest
/// baseline); a file produced by `nanoserve quantize` loads the same tensors
/// as int8 + per-row scales (F027) through the identical code path.
struct Qwen2Model {
    ModelConfig config;
    Weight embed_tokens;              // [vocab, hidden]; doubles as lm_head (tied)
    std::vector<LayerWeights> layers;
    std::vector<float> final_norm;    // [hidden]

    bool quantized() const { return embed_tokens.quantized(); }

    /// Loads config.json plus a weights file — model.safetensors when
    /// `weights_file` is empty, else that file (e.g. model.int8.safetensors).
    /// Throws on missing tensors or shape mismatches — a model that
    /// half-loads must not half-run.
    static Qwen2Model load(const std::string& model_dir,
                           const std::string& weights_file = "");
};

/// Offline quantization pass (F027): reads model_dir/model.safetensors,
/// quantizes every 2D weight matrix to int8 with per-row scales
/// (quantize_rows), keeps norms/biases fp32, and writes a safetensors file
/// to `out_path` (each quantized tensor keeps its name with dtype I8, plus a
/// "<name>.scale" F32 tensor). Returns the number of quantized tensors.
int quantize_model_file(const std::string& model_dir, const std::string& out_path);

/// K/V storage: for each layer, `max_seq` rows of `n_kv_heads * head_dim`
/// floats. Rows are written once per position and never move (contiguous
/// preallocation; the paged alternative is PagedKvCache, F034).
class KvCache {
public:
    KvCache(const ModelConfig& config, int64_t max_seq);

    /// No-op: every row exists from construction. Present so KvCache and
    /// PagedKvCache satisfy the same interface for the templated forward.
    void prepare(int64_t /*pos0*/, int64_t /*tokens*/) {}

    int64_t max_seq() const { return max_seq_; }
    /// The preallocation cost, for comparison against PagePool::bytes_backed.
    int64_t bytes_allocated() const;
    float* k_row(int64_t layer, int64_t pos) {
        return k_[static_cast<size_t>(layer)].data() + pos * kv_dim_;
    }
    float* v_row(int64_t layer, int64_t pos) {
        return v_[static_cast<size_t>(layer)].data() + pos * kv_dim_;
    }

private:
    int64_t max_seq_ = 0;
    int64_t kv_dim_ = 0;
    std::vector<std::vector<float>> k_;  // [layer][max_seq * kv_dim]
    std::vector<std::vector<float>> v_;
};

/// Scratch buffers for the forward pass, owned by the caller (the Engine)
/// and reused across calls so the decode hot loop allocates nothing (F024).
/// ensure() only ever grows a buffer — after the first, largest call (the
/// prefill), every 1-token decode step finds the buffers already big enough.
struct Scratch {
    std::vector<float> normed;        // [tokens, hidden]
    std::vector<float> q;             // [tokens, n_heads * head_dim]
    std::vector<float> attn;          // [tokens, n_heads * head_dim]
    std::vector<float> proj;          // [tokens, hidden]
    std::vector<float> gate;          // [tokens, intermediate]
    std::vector<float> up;            // [tokens, intermediate]
    std::vector<float> scores;        // [max_seq]: one attention row, sized
                                      //   for the full context up front so it
                                      //   never regrows as the sequence does
    std::vector<float> final_normed;  // [hidden]: lm_head input

    void ensure(const ModelConfig& config, int64_t tokens, int64_t max_seq);
};

/// Runs transformer layer `layer_idx` in place over `tokens` rows of `hidden`
/// ([tokens, hidden_size]) whose sequence positions are pos0, pos0+1, ....
/// Writes this range's K/V into the cache and attends over positions
/// [0, pos0+t] for each row t — the causal mask, expressed as a loop bound.
/// Free function (not an Engine private) so the layer golden test can drive
/// a single layer in isolation.
///
/// Templated over the cache (KvCache or PagedKvCache — defined for exactly
/// those two in qwen2.cpp) instead of virtual: k_row/v_row sit in the
/// attention inner loop, and the two storage layouts share every line of
/// math. The caller must have prepare()d the cache for [pos0, pos0+tokens).
template <class Cache>
void layer_forward(const Qwen2Model& model, int64_t layer_idx, float* hidden,
                   int64_t tokens, int64_t pos0, Cache& cache, Scratch& scratch);

/// Which KV cache layout an Engine runs on (F034). Contiguous preallocates
/// max_seq rows up front; paged allocates 16-token pages as the sequence
/// grows (see paged_kv.hpp for the design note). Outputs are bit-identical.
enum class KvLayout {
    contiguous,
    paged,
};

/// Owns the model, the cache, and the scratch buffers.
class Engine {
public:
    /// `weights_file` empty = model.safetensors (fp32); pass an int8 file
    /// from `nanoserve quantize` to run quantized (F027).
    /// If the metal backend is selected (ops::set_backend, F033) when this
    /// runs, every weight buffer is registered with the GPU here — on
    /// unified memory that is a zero-copy page wrap for the big matrices,
    /// so "upload" costs nothing and happens once, at load time.
    explicit Engine(const std::string& model_dir, int64_t max_seq = 2048,
                    const std::string& weights_file = "",
                    KvLayout kv_layout = KvLayout::contiguous);

    /// Unregisters the GPU-resident weights (registered iff the metal
    /// backend was selected at construction) BEFORE the vectors that back
    /// them are freed — the registry must never hold a dangling pointer.
    ~Engine();
    Engine(const Engine&) = delete;             // registry holds our pointers
    Engine& operator=(const Engine&) = delete;

    /// Appends `ids` to the sequence, runs the forward pass over them, and
    /// returns the logits ([vocab_size]) for the last token fed.
    std::span<const float> forward(std::span<const int32_t> ids);

    /// Forgets the sequence. Contiguous: rows are simply overwritten.
    /// Paged: the sequence's pages go back to the pool as well.
    void reset();

    int64_t seq_len() const { return seq_len_; }
    int64_t max_seq() const { return max_seq_; }
    const Qwen2Model& model() const { return model_; }

    /// KV memory actually allocated so far: the full preallocation for the
    /// contiguous layout, the backed-pages high-water mark for the paged one
    /// (pages keep their memory across reset(); see PagePool).
    int64_t kv_bytes_allocated() const;

    /// Embedding lookup only (exposed for the golden test).
    void embed(std::span<const int32_t> ids, float* out) const;

private:
    /// forward() after the layout dispatch; Cache is KvCache or PagedKvCache.
    template <class Cache>
    std::span<const float> forward_with(Cache& cache, std::span<const int32_t> ids);

    Qwen2Model model_;
    int64_t max_seq_ = 0;
    // Exactly one layout is live: contig_cache_, or pool_ + paged_cache_
    // (constructed in that order — the cache borrows the pool).
    std::unique_ptr<KvCache> contig_cache_;
    std::unique_ptr<PagePool> pool_;
    std::unique_ptr<PagedKvCache> paged_cache_;
    bool gpu_registered_ = false;  // weights registered with the metal backend
    int64_t seq_len_ = 0;         // tokens already in the cache
    Scratch scratch_;             // per-layer work buffers, reused every call
    std::vector<float> hidden_;   // [tokens, hidden] for the current call
    std::vector<float> logits_;   // [vocab]
};

/// Index of the largest logit — the greedy choice. First index wins ties,
/// matching torch.argmax, so greedy runs are comparable token by token.
int32_t argmax(std::span<const float> logits);

/// Timing split of one generation: prefill (time-to-first-token) vs decode
/// (per-token latency). The two phases have very different cost profiles —
/// prefill is O(prompt) work before anything appears, decode is one token's
/// work per step — which is why serving systems report them separately.
struct GenerateStats {
    double ttft_ms = 0.0;          // prompt forward + first token pick
    std::vector<double> step_ms;   // each subsequent decode step
};

class Sampler;  // model/sampler.hpp

/// Called once per generated token, in generation order, as soon as the
/// token is picked — before the next forward pass runs. This is what makes
/// streaming output possible: the caller sees token i while token i+1 is
/// still being computed.
using TokenCallback = std::function<void(int32_t token)>;

/// Decoding loop: prefill `prompt_ids`, then repeatedly append the token the
/// sampler picks, up to `max_new_tokens`. A generated stop id (eos) IS
/// included in the returned tokens (and reported to `on_token`) and ends the
/// loop — the same contract as HF `generate()`, so outputs compare 1:1
/// against the goldens.
/// If `stats` is non-null it is filled with prefill/decode timings.
std::vector<int32_t> generate(Engine& engine, std::span<const int32_t> prompt_ids,
                              int64_t max_new_tokens,
                              std::span<const int32_t> stop_ids, Sampler& sampler,
                              GenerateStats* stats = nullptr,
                              const TokenCallback& on_token = nullptr);

/// generate() with a temperature-0 sampler — pure argmax decoding.
std::vector<int32_t> greedy_generate(Engine& engine,
                                     std::span<const int32_t> prompt_ids,
                                     int64_t max_new_tokens,
                                     std::span<const int32_t> stop_ids,
                                     GenerateStats* stats = nullptr);

}  // namespace nano
