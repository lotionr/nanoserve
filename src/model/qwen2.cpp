#include "model/qwen2.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "core/metal.hpp"
#include "core/ops.hpp"
#include "core/quant.hpp"
#include "core/safetensors.hpp"
#include "model/sampler.hpp"

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

/// Fetches one weight matrix, in whichever precision the file stores it:
/// I8 tensors come with a "<name>.scale" companion (written by `nanoserve
/// quantize`); anything else is widened to fp32. Loading is the ONLY place
/// the two formats differ — the forward pass just dispatches on Weight.
Weight load_weight(const SafeTensors& st, const std::string& name, int64_t rows,
                   int64_t cols) {
    const TensorInfo& info = st.require(name);
    if (info.numel() != rows * cols) {
        throw std::runtime_error("tensor " + name + ": expected " +
                                 std::to_string(rows * cols) + " values, file has " +
                                 std::to_string(info.numel()));
    }
    Weight w;
    w.rows = rows;
    w.cols = cols;
    if (info.dtype == DType::I8) {
        const std::span<const std::byte> bytes = st.raw(info);
        w.q8.resize(bytes.size());
        std::memcpy(w.q8.data(), bytes.data(), bytes.size());
        w.scales = load_tensor(st, name + ".scale", rows);
    } else {
        w.f32 = st.to_f32(info);
    }
    return w;
}

/// Walks every buffer the GPU kernels might bind — weight matrices (fp32 or
/// int8 + scales) and the qkv biases — and registers or unregisters each
/// with the metal backend. Norms are skipped: rmsnorm always runs on the
/// CPU. Registering is idempotent and unregistering unknown pointers is a
/// no-op, so the walk doesn't need to know which calls the size floor will
/// actually send to the GPU.
void register_model_weights(const Qwen2Model& m, bool reg) {
    const auto one = [&](const void* ptr, size_t bytes) {
        if (reg) {
            metal::register_weights(ptr, bytes);
        } else {
            metal::unregister_weights(ptr);
        }
    };
    const auto weight = [&](const Weight& w) {
        if (w.quantized()) {
            one(w.q8.data(), w.q8.size());
            one(w.scales.data(), w.scales.size() * sizeof(float));
        } else {
            one(w.f32.data(), w.f32.size() * sizeof(float));
        }
    };
    weight(m.embed_tokens);   // also the lm_head (tied)
    for (const LayerWeights& l : m.layers) {
        weight(l.q_w);
        weight(l.k_w);
        weight(l.v_w);
        weight(l.o_w);
        weight(l.gate_w);
        weight(l.up_w);
        weight(l.down_w);
        one(l.q_b.data(), l.q_b.size() * sizeof(float));
        one(l.k_b.data(), l.k_b.size() * sizeof(float));
        one(l.v_b.data(), l.v_b.size() * sizeof(float));
    }
}

}  // namespace

// Declared in qwen2.hpp; kept here next to the ops:: kernels it dispatches to.
void weight_linear(const Weight& w, const float* x, const float* bias, float* y,
                   int64_t tokens) {
    if (w.quantized()) {
        ops::linear_q8(x, w.q8.data(), w.scales.data(), bias, y, tokens, w.cols, w.rows);
    } else {
        ops::linear(x, w.f32.data(), bias, y, tokens, w.cols, w.rows);
    }
}

void embed_rows(const Qwen2Model& model, std::span<const int32_t> ids, float* out) {
    const int64_t H = model.config.hidden_size;
    const Weight& e = model.embed_tokens;
    for (size_t i = 0; i < ids.size(); ++i) {
        float* dst = out + static_cast<int64_t>(i) * H;
        if (e.quantized()) {
            // Dequantize the one looked-up row: q * its row scale. This is
            // the only quantization error the embedding lookup introduces.
            const int8_t* row = e.q8.data() + int64_t{ids[i]} * H;
            const float scale = e.scales[static_cast<size_t>(ids[i])];
            for (int64_t j = 0; j < H; ++j) {
                dst[j] = static_cast<float>(row[j]) * scale;
            }
        } else {
            std::memcpy(dst, e.f32.data() + int64_t{ids[i]} * H,
                        sizeof(float) * static_cast<size_t>(H));
        }
    }
}

Qwen2Model Qwen2Model::load(const std::string& model_dir,
                            const std::string& weights_file) {
    Qwen2Model m;
    m.config = ModelConfig::from_dir(model_dir);
    const ModelConfig& c = m.config;
    const std::string path =
        weights_file.empty() ? model_dir + "/model.safetensors" : weights_file;
    const SafeTensors st(path);

    const int64_t hidden = c.hidden_size;
    const int64_t q_dim = c.num_heads * c.head_dim;
    const int64_t kv_dim = c.num_kv_heads * c.head_dim;

    m.embed_tokens = load_weight(st, "model.embed_tokens.weight", c.vocab_size, hidden);
    m.final_norm = load_tensor(st, "model.norm.weight", hidden);
    if (!c.tie_word_embeddings) {
        throw std::runtime_error("untied lm_head not supported yet (Qwen2.5-0.5B ties it)");
    }

    m.layers.reserve(static_cast<size_t>(c.num_layers));
    for (int64_t i = 0; i < c.num_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        LayerWeights w;
        w.input_norm = load_tensor(st, p + "input_layernorm.weight", hidden);
        w.q_w = load_weight(st, p + "self_attn.q_proj.weight", q_dim, hidden);
        w.q_b = load_tensor(st, p + "self_attn.q_proj.bias", q_dim);
        w.k_w = load_weight(st, p + "self_attn.k_proj.weight", kv_dim, hidden);
        w.k_b = load_tensor(st, p + "self_attn.k_proj.bias", kv_dim);
        w.v_w = load_weight(st, p + "self_attn.v_proj.weight", kv_dim, hidden);
        w.v_b = load_tensor(st, p + "self_attn.v_proj.bias", kv_dim);
        w.o_w = load_weight(st, p + "self_attn.o_proj.weight", hidden, q_dim);
        w.post_norm = load_tensor(st, p + "post_attention_layernorm.weight", hidden);
        w.gate_w = load_weight(st, p + "mlp.gate_proj.weight", c.intermediate_size, hidden);
        w.up_w = load_weight(st, p + "mlp.up_proj.weight", c.intermediate_size, hidden);
        w.down_w = load_weight(st, p + "mlp.down_proj.weight", hidden, c.intermediate_size);
        m.layers.push_back(std::move(w));
    }
    return m;
}

int quantize_model_file(const std::string& model_dir, const std::string& out_path) {
    const SafeTensors st(model_dir + "/model.safetensors");

    // Everything 2D in a Qwen2 checkpoint is a projection matrix (or the
    // embedding, which doubles as lm_head) — exactly the tensors worth
    // quantizing. Norms and biases are 1D, tiny, and stay fp32.
    std::vector<QuantMatrix> quantized;      // owns int8 data until write
    std::vector<std::vector<float>> kept;    // owns fp32 data until write
    std::vector<SaveTensor> out;
    int n_quantized = 0;

    // Reserve so the data pointers taken below never move on push_back.
    quantized.reserve(st.tensors().size());
    kept.reserve(st.tensors().size());

    for (const TensorInfo& t : st.tensors()) {
        const std::vector<float> f32 = st.to_f32(t);
        if (t.shape.size() == 2) {
            quantized.push_back(quantize_rows(f32.data(), t.shape[0], t.shape[1]));
            const QuantMatrix& q = quantized.back();
            out.push_back({t.name, DType::I8, t.shape, q.q.data(), q.q.size()});
            out.push_back({t.name + ".scale", DType::F32, {q.rows}, q.scales.data(),
                           q.scales.size() * sizeof(float)});
            ++n_quantized;
        } else {
            kept.push_back(f32);
            out.push_back({t.name, DType::F32, t.shape, kept.back().data(),
                           kept.back().size() * sizeof(float)});
        }
    }

    write_safetensors(out_path, out,
                      {{"format", "nanoserve int8 per-row (F027)"},
                       {"source", model_dir + "/model.safetensors"}});
    return n_quantized;
}

KvCache::KvCache(const ModelConfig& config, int64_t max_seq)
    : max_seq_(max_seq), kv_dim_(config.num_kv_heads * config.head_dim) {
    const size_t bytes_per_layer = static_cast<size_t>(max_seq_ * kv_dim_);
    k_.assign(static_cast<size_t>(config.num_layers), std::vector<float>(bytes_per_layer));
    v_.assign(static_cast<size_t>(config.num_layers), std::vector<float>(bytes_per_layer));
}

int64_t KvCache::bytes_allocated() const {
    return static_cast<int64_t>(k_.size()) * 2 * max_seq_ * kv_dim_ *
           static_cast<int64_t>(sizeof(float));
}

void Scratch::ensure(const ModelConfig& config, int64_t tokens, int64_t max_seq) {
    const auto grow = [](std::vector<float>& v, int64_t needed) {
        if (v.size() < static_cast<size_t>(needed)) {
            v.resize(static_cast<size_t>(needed));
        }
    };
    grow(normed, tokens * config.hidden_size);
    grow(q, tokens * config.num_heads * config.head_dim);
    grow(attn, tokens * config.num_heads * config.head_dim);
    grow(proj, tokens * config.hidden_size);
    grow(gate, tokens * config.intermediate_size);
    grow(up, tokens * config.intermediate_size);
    grow(scores, max_seq);
    grow(final_normed, config.hidden_size);
}

template <class Cache>
void layer_forward(const Qwen2Model& model, int64_t layer_idx, float* hidden,
                   int64_t tokens, int64_t pos0, Cache& cache, Scratch& scratch) {
    const ModelConfig& c = model.config;
    const LayerWeights& w = model.layers[static_cast<size_t>(layer_idx)];
    const int64_t H = c.hidden_size;
    const int64_t D = c.head_dim;
    const int64_t q_dim = c.num_heads * D;
    // GQA: this many consecutive query heads share one k/v head.
    const int64_t group = c.num_heads / c.num_kv_heads;
    const float eps = static_cast<float>(c.rms_norm_eps);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const float theta = static_cast<float>(c.rope_theta);

    // No-op after the first (largest) call — decode steps allocate nothing.
    scratch.ensure(c, tokens, cache.max_seq());
    std::vector<float>& normed = scratch.normed;
    std::vector<float>& q = scratch.q;
    std::vector<float>& attn = scratch.attn;
    std::vector<float>& proj = scratch.proj;
    std::vector<float>& scores = scratch.scores;

    // --- attention block: hidden += o_proj(attend(rope(qkv(norm(hidden))))) ---
    ops::rmsnorm(hidden, w.input_norm.data(), normed.data(), tokens, H, eps);
    weight_linear(w.q_w, normed.data(), w.q_b.data(), q.data(), tokens);
    for (int64_t t = 0; t < tokens; ++t) {
        // K and V rows are computed straight into the cache — after this
        // loop the cache holds positions [0, pos0 + tokens).
        const float* in = normed.data() + t * H;
        weight_linear(w.k_w, in, w.k_b.data(), cache.k_row(layer_idx, pos0 + t), 1);
        weight_linear(w.v_w, in, w.v_b.data(), cache.v_row(layer_idx, pos0 + t), 1);
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
                    ops::dot(q_head, cache.k_row(layer_idx, s) + kv_off, D) * scale;
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
    weight_linear(w.o_w, attn.data(), nullptr, proj.data(), tokens);
    ops::add(hidden, proj.data(), hidden, tokens * H);

    // --- MLP block: hidden += down(silu(gate(norm(hidden))) * up(norm(hidden))) ---
    const int64_t I = c.intermediate_size;
    std::vector<float>& gate = scratch.gate;
    std::vector<float>& up = scratch.up;
    ops::rmsnorm(hidden, w.post_norm.data(), normed.data(), tokens, H, eps);
    weight_linear(w.gate_w, normed.data(), nullptr, gate.data(), tokens);
    weight_linear(w.up_w, normed.data(), nullptr, up.data(), tokens);
    ops::silu(gate.data(), tokens * I);
    ops::mul(gate.data(), up.data(), gate.data(), tokens * I);
    weight_linear(w.down_w, gate.data(), nullptr, proj.data(), tokens);
    ops::add(hidden, proj.data(), hidden, tokens * H);
}

// The only two cache layouts (qwen2.hpp's KvLayout). Instantiated here so
// the template definition can live in this .cpp next to the math it runs.
template void layer_forward<KvCache>(const Qwen2Model&, int64_t, float*, int64_t,
                                     int64_t, KvCache&, Scratch&);
template void layer_forward<PagedKvCache>(const Qwen2Model&, int64_t, float*,
                                          int64_t, int64_t, PagedKvCache&, Scratch&);

Engine::Engine(const std::string& model_dir, int64_t max_seq,
               const std::string& weights_file, KvLayout kv_layout)
    : model_(Qwen2Model::load(model_dir, weights_file)), max_seq_(max_seq) {
    if (kv_layout == KvLayout::paged) {
        // Pool capacity = exactly the pages a max_seq sequence could need, so
        // the "sequence exceeds capacity" error fires before pool exhaustion.
        const int64_t max_pages =
            (max_seq + kPageSizeDefault - 1) / kPageSizeDefault;
        pool_ = std::make_unique<PagePool>(model_.config, kPageSizeDefault, max_pages);
        paged_cache_ = std::make_unique<PagedKvCache>(*pool_, max_seq);
    } else {
        contig_cache_ = std::make_unique<KvCache>(model_.config, max_seq);
    }
    logits_.resize(static_cast<size_t>(model_.config.vocab_size));
    if (ops::backend() == ops::Backend::metal) {
        register_model_weights(model_, /*reg=*/true);
        gpu_registered_ = true;
    }
}

Engine::~Engine() {
    if (gpu_registered_) {
        // Runs before the member destructors free the weight vectors, so
        // the GPU registry never points at freed memory.
        register_model_weights(model_, /*reg=*/false);
    }
}

void Engine::embed(std::span<const int32_t> ids, float* out) const {
    embed_rows(model_, ids, out);
}

void Engine::reset() {
    seq_len_ = 0;
    if (paged_cache_) {
        paged_cache_->release();  // pages back to the pool
    }
}

int64_t Engine::kv_bytes_allocated() const {
    return pool_ ? pool_->bytes_backed() : contig_cache_->bytes_allocated();
}

std::span<const float> Engine::forward(std::span<const int32_t> ids) {
    // One dispatch per forward() call; every loop below it is fully typed.
    if (paged_cache_) {
        return forward_with(*paged_cache_, ids);
    }
    return forward_with(*contig_cache_, ids);
}

template <class Cache>
std::span<const float> Engine::forward_with(Cache& cache, std::span<const int32_t> ids) {
    const ModelConfig& c = model_.config;
    const int64_t tokens = static_cast<int64_t>(ids.size());
    if (tokens == 0) {
        throw std::runtime_error("forward() needs at least one token");
    }
    if (seq_len_ + tokens > cache.max_seq()) {
        throw std::runtime_error("sequence exceeds KV cache capacity (" +
                                 std::to_string(cache.max_seq()) + " tokens)");
    }
    // Contiguous: no-op. Paged: allocate pages covering the new positions.
    cache.prepare(seq_len_, tokens);
    const int64_t H = c.hidden_size;

    // resize() only allocates when it grows past capacity — a 1-token decode
    // step after a longer prefill reuses the prefill's storage.
    hidden_.resize(static_cast<size_t>(tokens * H));
    embed(ids, hidden_.data());
    for (int64_t layer = 0; layer < c.num_layers; ++layer) {
        layer_forward(model_, layer, hidden_.data(), tokens, seq_len_, cache, scratch_);
    }
    seq_len_ += tokens;

    // Only the last token's hidden state becomes logits (that's the one whose
    // next token we're predicting). lm_head is the embedding matrix (tied).
    float* normed = scratch_.final_normed.data();
    ops::rmsnorm(hidden_.data() + (tokens - 1) * H, model_.final_norm.data(),
                 normed, 1, H, static_cast<float>(c.rms_norm_eps));
    weight_linear(model_.embed_tokens, normed, nullptr, logits_.data(), 1);
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

std::vector<int32_t> generate(Engine& engine, std::span<const int32_t> prompt_ids,
                              int64_t max_new_tokens,
                              std::span<const int32_t> stop_ids, Sampler& sampler,
                              GenerateStats* stats, const TokenCallback& on_token) {
    const auto now = std::chrono::steady_clock::now;
    const auto to_ms = [](auto d) {
        return std::chrono::duration<double, std::milli>(d).count();
    };

    // Reserve up front so the loop's push_backs never reallocate (F024).
    std::vector<int32_t> out;
    out.reserve(static_cast<size_t>(max_new_tokens));
    if (stats) {
        stats->step_ms.reserve(static_cast<size_t>(max_new_tokens));
    }
    auto t0 = now();
    std::span<const float> logits = engine.forward(prompt_ids);
    for (int64_t i = 0; i < max_new_tokens; ++i) {
        const int32_t next = sampler.sample(logits);
        if (stats) {
            // First token closes the prefill phase; the rest are decode steps.
            const double ms = to_ms(now() - t0);
            if (i == 0) {
                stats->ttft_ms = ms;
            } else {
                stats->step_ms.push_back(ms);
            }
        }
        out.push_back(next);
        if (on_token) {
            on_token(next);
        }
        bool stop = false;
        for (int32_t s : stop_ids) {
            stop = stop || next == s;
        }
        if (stop) {
            break;
        }
        t0 = now();
        const int32_t fed[] = {next};
        logits = engine.forward(fed);
    }
    return out;
}

std::vector<int32_t> greedy_generate(Engine& engine,
                                     std::span<const int32_t> prompt_ids,
                                     int64_t max_new_tokens,
                                     std::span<const int32_t> stop_ids,
                                     GenerateStats* stats) {
    Sampler greedy({.temperature = 0.0f, .top_k = 0, .top_p = 1.0f, .seed = 0},
                   engine.model().config.vocab_size);
    return generate(engine, prompt_ids, max_new_tokens, stop_ids, greedy, stats);
}

}  // namespace nano
