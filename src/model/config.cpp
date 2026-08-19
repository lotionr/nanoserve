#include "model/config.hpp"

#include <stdexcept>

#include "core/json.hpp"

namespace nano {

ModelConfig ModelConfig::from_dir(const std::string& model_dir) {
    const std::string path = model_dir + "/config.json";
    const json::Value root = json::parse(json::read_file(path));

    auto require_int = [&](const char* key) {
        const json::Value* v = root.find(key);
        if (v == nullptr) {
            throw std::runtime_error(path + ": missing required field '" + key + "'");
        }
        return v->as_int();
    };
    auto require_double = [&](const char* key) {
        const json::Value* v = root.find(key);
        if (v == nullptr) {
            throw std::runtime_error(path + ": missing required field '" + key + "'");
        }
        return v->as_double();
    };

    ModelConfig cfg;
    cfg.hidden_size = require_int("hidden_size");
    cfg.num_layers = require_int("num_hidden_layers");
    cfg.num_heads = require_int("num_attention_heads");
    cfg.num_kv_heads = require_int("num_key_value_heads");
    cfg.intermediate_size = require_int("intermediate_size");
    cfg.vocab_size = require_int("vocab_size");
    cfg.max_position_embeddings = require_int("max_position_embeddings");
    cfg.rope_theta = require_double("rope_theta");
    cfg.rms_norm_eps = require_double("rms_norm_eps");

    const json::Value* tie = root.find("tie_word_embeddings");
    cfg.tie_word_embeddings = tie != nullptr && tie->as_bool();

    if (cfg.num_heads <= 0 || cfg.hidden_size % cfg.num_heads != 0) {
        throw std::runtime_error(path + ": hidden_size must be divisible by num_attention_heads");
    }
    cfg.head_dim = cfg.hidden_size / cfg.num_heads;

    return cfg;
}

}  // namespace nano
