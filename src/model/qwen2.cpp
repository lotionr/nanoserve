#include "model/qwen2.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "core/ops.hpp"
#include "core/safetensors.hpp"

namespace nano {

namespace {

/// Fetches one tensor as fp32 and checks its element count. A wrong shape
/// here would otherwise surface as garbage activations much later.
std::vector<float> load_tensor(const SafeTensors& st, const std::string& name,
                               int64_t expected_numel) {
    const TensorInfo& info = st.require(name);
    if (info.numel() != expected_numel) {
        throw std::runtime_error("tensor " + name + ": expected " +
                                 std::to_string(expected_numel) + " values, file has " +
                                 std::to_string(info.numel()));
    }
    return st.to_f32(info);
}

float dot(const float* a, const float* b, int64_t n) {
    float acc = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

}  // namespace

Qwen2Model Qwen2Model::load(const std::string& model_dir) {
    Qwen2Model m;
    m.config = ModelConfig::from_dir(model_dir);
    const ModelConfig& c = m.config;
    const SafeTensors st(model_dir + "/model.safetensors");

    const int64_t hidden = c.hidden_size;
    const int64_t q_dim = c.num_heads * c.head_dim;
    const int64_t kv_dim = c.num_kv_heads * c.head_dim;

    m.embed_tokens = load_tensor(st, "model.embed_tokens.weight", c.vocab_size * hidden);
    m.final_norm = load_tensor(st, "model.norm.weight", hidden);
    if (!c.tie_word_embeddings) {
        throw std::runtime_error("untied lm_head not supported yet (Qwen2.5-0.5B ties it)");
    }

    m.layers.reserve(static_cast<size_t>(c.num_layers));
    for (int64_t i = 0; i < c.num_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        LayerWeights w;
        w.input_norm = load_tensor(st, p + "input_layernorm.weight", hidden);
        w.q_w = load_tensor(st, p + "self_attn.q_proj.weight", q_dim * hidden);
        w.q_b = load_tensor(st, p + "self_attn.q_proj.bias", q_dim);
        w.k_w = load_tensor(st, p + "self_attn.k_proj.weight", kv_dim * hidden);
        w.k_b = load_tensor(st, p + "self_attn.k_proj.bias", kv_dim);
        w.v_w = load_tensor(st, p + "self_attn.v_proj.weight", kv_dim * hidden);
        w.v_b = load_tensor(st, p + "self_attn.v_proj.bias", kv_dim);
        w.o_w = load_tensor(st, p + "self_attn.o_proj.weight", hidden * q_dim);
        w.post_norm = load_tensor(st, p + "post_attention_layernorm.weight", hidden);
        w.gate_w = load_tensor(st, p + "mlp.gate_proj.weight", c.intermediate_size * hidden);
        w.up_w = load_tensor(st, p + "mlp.up_proj.weight", c.intermediate_size * hidden);
        w.down_w = load_tensor(st, p + "mlp.down_proj.weight", hidden * c.intermediate_size);
        m.layers.push_back(std::move(w));
    }
    return m;
}

KvCache::KvCache(const ModelConfig& config, int64_t max_seq)
    : max_seq_(max_seq), kv_dim_(config.num_kv_heads * config.head_dim) {
    const size_t bytes_per_layer = static_cast<size_t>(max_seq_ * kv_dim_);
    k_.assign(static_cast<size_t>(config.num_layers), std::vector<float>(bytes_per_layer));
    v_.assign(static_cast<size_t>(config.num_layers), std::vector<float>(bytes_per_layer));
}

void layer_forward(const Qwen2Model& model, int64_t layer_idx, float* hidden,
                   int64_t tokens, int64_t pos0, KvCache& cache) {
    const ModelConfig& c = model.config;
    const LayerWeights& w = model.layers[static_cast<size_t>(layer_idx)];
    const int64_t H = c.hidden_size;
    const int64_t D = c.head_dim;
    const int64_t q_dim = c.num_heads * D;
    const int64_t kv_dim = c.num_kv_heads * D;
    // GQA: this many consecutive query heads share one k/v head.
    const int64_t group = c.num_heads / c.num_kv_heads;
    const float eps = static_cast<float>(c.rms_norm_eps);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const float theta = static_cast<float>(c.rope_theta);

    // Scratch (allocation-free decode is F024; clarity first).
    std::vector<float> normed(static_cast<size_t>(tokens * H));
    std::vector<float> q(static_cast<size_t>(tokens * q_dim));
    std::vector<float> attn(static_cast<size_t>(tokens * q_dim));
    std::vector<float> proj(static_cast<size_t>(tokens * H));
    std::vector<float> scores(static_cast<size_t>(pos0 + tokens));

    // --- attention block: hidden += o_proj(attend(rope(qkv(norm(hidden))))) ---
    ops::rmsnorm(hidden, w.input_norm.data(), normed.data(), tokens, H, eps);
    ops::linear(normed.data(), w.q_w.data(), w.q_b.data(), q.data(), tokens, H, q_dim);
    for (int64_t t = 0; t < tokens; ++t) {
        // K and V rows are computed straight into the cache — after this
        // loop the cache holds positions [0, pos0 + tokens).
        const float* in = normed.data() + t * H;
        ops::linear(in, w.k_w.data(), w.k_b.data(), cache.k_row(layer_idx, pos0 + t), 1,
                    H, kv_dim);
        ops::linear(in, w.v_w.data(), w.v_b.data(), cache.v_row(layer_idx, pos0 + t), 1,
                    H, kv_dim);
        ops::rope(q.data() + t * q_dim, c.num_heads, D, pos0 + t, theta);
        ops::rope(cache.k_row(layer_idx, pos0 + t), c.num_kv_heads, D, pos0 + t, theta);
    }

    for (int64_t t = 0; t < tokens; ++t) {
        const int64_t n_ctx = pos0 + t + 1;  // causal: attend to [0, pos0+t]
        for (int64_t h = 0; h < c.num_heads; ++h) {
            const float* q_head = q.data() + t * q_dim + h * D;
            const int64_t kv_off = (h / group) * D;  // shared k/v head (GQA)

            for (int64_t s = 0; s < n_ctx; ++s) {
                scores[static_cast<size_t>(s)] =
                    dot(q_head, cache.k_row(layer_idx, s) + kv_off, D) * scale;
            }
            ops::softmax(scores.data(), n_ctx);

            float* out = attn.data() + t * q_dim + h * D;
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
    ops::linear(attn.data(), w.o_w.data(), nullptr, proj.data(), tokens, q_dim, H);
    ops::add(hidden, proj.data(), hidden, tokens * H);

    // --- MLP block: hidden += down(silu(gate(norm(hidden))) * up(norm(hidden))) ---
    const int64_t I = c.intermediate_size;
    std::vector<float> gate(static_cast<size_t>(tokens * I));
    std::vector<float> up(static_cast<size_t>(tokens * I));
    ops::rmsnorm(hidden, w.post_norm.data(), normed.data(), tokens, H, eps);
    ops::linear(normed.data(), w.gate_w.data(), nullptr, gate.data(), tokens, H, I);
    ops::linear(normed.data(), w.up_w.data(), nullptr, up.data(), tokens, H, I);
    ops::silu(gate.data(), tokens * I);
    ops::mul(gate.data(), up.data(), gate.data(), tokens * I);
    ops::linear(gate.data(), w.down_w.data(), nullptr, proj.data(), tokens, I, H);
    ops::add(hidden, proj.data(), hidden, tokens * H);
}

Engine::Engine(const std::string& model_dir, int64_t max_seq)
    : model_(Qwen2Model::load(model_dir)), cache_(model_.config, max_seq) {
    logits_.resize(static_cast<size_t>(model_.config.vocab_size));
}

void Engine::embed(std::span<const int32_t> ids, float* out) const {
    const int64_t H = model_.config.hidden_size;
    for (size_t i = 0; i < ids.size(); ++i) {
        const float* row = model_.embed_tokens.data() + int64_t{ids[i]} * H;
        std::memcpy(out + static_cast<int64_t>(i) * H, row,
                    sizeof(float) * static_cast<size_t>(H));
    }
}

std::span<const float> Engine::forward(std::span<const int32_t> ids) {
    const ModelConfig& c = model_.config;
    const int64_t tokens = static_cast<int64_t>(ids.size());
    if (tokens == 0) {
        throw std::runtime_error("forward() needs at least one token");
    }
    if (seq_len_ + tokens > cache_.max_seq()) {
        throw std::runtime_error("sequence exceeds KV cache capacity (" +
                                 std::to_string(cache_.max_seq()) + " tokens)");
    }
    const int64_t H = c.hidden_size;

    hidden_.resize(static_cast<size_t>(tokens * H));
    embed(ids, hidden_.data());
    for (int64_t layer = 0; layer < c.num_layers; ++layer) {
        layer_forward(model_, layer, hidden_.data(), tokens, seq_len_, cache_);
    }
    seq_len_ += tokens;

    // Only the last token's hidden state becomes logits (that's the one whose
    // next token we're predicting). lm_head is the embedding matrix (tied).
    std::vector<float> normed(static_cast<size_t>(H));
    ops::rmsnorm(hidden_.data() + (tokens - 1) * H, model_.final_norm.data(),
                 normed.data(), 1, H, static_cast<float>(c.rms_norm_eps));
    ops::linear(normed.data(), model_.embed_tokens.data(), nullptr, logits_.data(), 1, H,
                c.vocab_size);
    return logits_;
}

int32_t argmax(std::span<const float> logits) {
    size_t best = 0;
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return static_cast<int32_t>(best);
}

std::vector<int32_t> greedy_generate(Engine& engine,
                                     std::span<const int32_t> prompt_ids,
                                     int64_t max_new_tokens,
                                     std::span<const int32_t> stop_ids) {
    std::vector<int32_t> out;
    std::span<const float> logits = engine.forward(prompt_ids);
    for (int64_t i = 0; i < max_new_tokens; ++i) {
        const int32_t next = argmax(logits);
        out.push_back(next);
        bool stop = false;
        for (int32_t s : stop_ids) {
            stop = stop || next == s;
        }
        if (stop) {
            break;
        }
        const int32_t fed[] = {next};
        logits = engine.forward(fed);
    }
    return out;
}

}  // namespace nano
