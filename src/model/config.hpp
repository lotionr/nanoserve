// Typed view of a Hugging Face config.json for Qwen2-family models.
// Loading this into a struct up front means the rest of the engine never
// touches JSON — and missing fields fail loudly at startup, not mid-forward.
#pragma once

#include <cstdint>
#include <string>

namespace nano {

struct ModelConfig {
    int64_t hidden_size = 0;         // model width (896 for Qwen2.5-0.5B)
    int64_t num_layers = 0;          // transformer blocks (24)
    int64_t num_heads = 0;           // query heads (14)
    int64_t num_kv_heads = 0;        // key/value heads — GQA when < num_heads (2)
    int64_t head_dim = 0;            // hidden_size / num_heads (64)
    int64_t intermediate_size = 0;   // MLP width (4864)
    int64_t vocab_size = 0;          // 151936
    int64_t max_position_embeddings = 0;
    double rope_theta = 0.0;         // RoPE base frequency (1e6)
    double rms_norm_eps = 0.0;       // 1e-6
    bool tie_word_embeddings = false;  // lm_head shares the embedding matrix

    /// Parses `<model_dir>/config.json`. Throws with the missing key's name
    /// if a required field is absent.
    static ModelConfig from_dir(const std::string& model_dir);
};

}  // namespace nano
