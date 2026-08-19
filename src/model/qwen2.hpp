// Qwen2.5 model: weights, KV cache, and the fp32 forward pass.
//
// The design is deliberately plain (llama2.c lineage):
//   - Qwen2Model  = every weight widened to fp32 up front (~2 GB for 0.5B).
//   - KvCache     = per-layer K/V rows, contiguous [max_seq, kv_dim].
//   - layer_forward() = one transformer block over a span of new tokens.
//   - Engine      = embed -> 24 x layer_forward -> final norm -> lm_head.
//
// One forward() call handles both phases of inference: the first call feeds
// the whole prompt (prefill), later calls feed one token each (decode). Both
// go through the same code path — the KV cache is what makes decode O(1)
// per token instead of re-running the whole sequence.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "model/config.hpp"

namespace nano {

/// One transformer block's weights, fp32.
/// Qwen2 quirk: q/k/v projections have biases; o_proj and the MLP do not.
struct LayerWeights {
    std::vector<float> input_norm;  // [hidden]
    std::vector<float> q_w, q_b;    // [n_heads*head_dim, hidden], [n_heads*head_dim]
    std::vector<float> k_w, k_b;    // [n_kv_heads*head_dim, hidden], ...
    std::vector<float> v_w, v_b;
    std::vector<float> o_w;         // [hidden, n_heads*head_dim]
    std::vector<float> post_norm;   // [hidden]
    std::vector<float> gate_w;      // [intermediate, hidden]
    std::vector<float> up_w;        // [intermediate, hidden]
    std::vector<float> down_w;      // [hidden, intermediate]
};

/// All model weights, loaded from safetensors and widened bf16 -> fp32.
/// Keeping everything fp32 is the honest baseline; quantization is F027.
struct Qwen2Model {
    ModelConfig config;
    std::vector<float> embed_tokens;  // [vocab, hidden]; doubles as lm_head (tied)
    std::vector<LayerWeights> layers;
    std::vector<float> final_norm;    // [hidden]

    /// Loads config.json + model.safetensors. Throws on missing tensors or
    /// shape mismatches — a model that half-loads must not half-run.
    static Qwen2Model load(const std::string& model_dir);
};

/// K/V storage: for each layer, `max_seq` rows of `n_kv_heads * head_dim`
/// floats. Rows are written once per position and never move (contiguous
/// preallocation; the paged version is F034).
class KvCache {
public:
    KvCache(const ModelConfig& config, int64_t max_seq);

    int64_t max_seq() const { return max_seq_; }
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

/// Runs transformer layer `layer_idx` in place over `tokens` rows of `hidden`
/// ([tokens, hidden_size]) whose sequence positions are pos0, pos0+1, ....
/// Writes this range's K/V into the cache and attends over positions
/// [0, pos0+t] for each row t — the causal mask, expressed as a loop bound.
/// Free function (not an Engine private) so the layer golden test can drive
/// a single layer in isolation.
void layer_forward(const Qwen2Model& model, int64_t layer_idx, float* hidden,
                   int64_t tokens, int64_t pos0, KvCache& cache);

/// Owns the model, the cache, and the scratch buffers.
class Engine {
public:
    explicit Engine(const std::string& model_dir, int64_t max_seq = 2048);

    /// Appends `ids` to the sequence, runs the forward pass over them, and
    /// returns the logits ([vocab_size]) for the last token fed.
    std::span<const float> forward(std::span<const int32_t> ids);

    /// Forgets the sequence (cache rows are simply overwritten).
    void reset() { seq_len_ = 0; }

    int64_t seq_len() const { return seq_len_; }
    const Qwen2Model& model() const { return model_; }

    /// Embedding lookup only (exposed for the golden test).
    void embed(std::span<const int32_t> ids, float* out) const;

private:
    Qwen2Model model_;
    KvCache cache_;
    int64_t seq_len_ = 0;         // tokens already in the cache
    std::vector<float> hidden_;   // [tokens, hidden] for the current call
    std::vector<float> logits_;   // [vocab]
};

/// Index of the largest logit — the greedy choice. First index wins ties,
/// matching torch.argmax, so greedy runs are comparable token by token.
int32_t argmax(std::span<const float> logits);

/// Greedy decoding: prefill `prompt_ids`, then repeatedly append the argmax
/// token, up to `max_new_tokens`. A generated stop id (eos) IS included in
/// the returned tokens and ends the loop — the same contract as HF
/// `generate()`, so outputs compare 1:1 against the golden.
std::vector<int32_t> greedy_generate(Engine& engine,
                                     std::span<const int32_t> prompt_ids,
                                     int64_t max_new_tokens,
                                     std::span<const int32_t> stop_ids);

}  // namespace nano
