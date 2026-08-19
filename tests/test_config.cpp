#include "model/config.hpp"

#include <cstdio>
#include <string>

#include "core/json.hpp"
#include "testing.hpp"

int main() {
    const std::string dir = nano::testing::require_model_dir();

    // Known Qwen2.5-0.5B-Instruct architecture values.
    const nano::ModelConfig cfg = nano::ModelConfig::from_dir(dir);
    NANO_CHECK(cfg.hidden_size == 896);
    NANO_CHECK(cfg.num_layers == 24);
    NANO_CHECK(cfg.num_heads == 14);
    NANO_CHECK(cfg.num_kv_heads == 2);  // GQA: 7 query heads per kv head
    NANO_CHECK(cfg.head_dim == 64);
    NANO_CHECK(cfg.intermediate_size == 4864);
    NANO_CHECK(cfg.vocab_size == 151936);
    NANO_CHECK(cfg.max_position_embeddings == 32768);
    NANO_CHECK(cfg.rope_theta == 1000000.0);
    NANO_CHECK(cfg.rms_norm_eps == 1e-6);
    NANO_CHECK(cfg.tie_word_embeddings == true);

    // A config missing a required field fails with the key's name in the
    // message. Build a broken config.json in a temp dir to prove it.
    {
        const std::string tmp_dir = "build/test_config_tmp";
        std::string cmd = "mkdir -p " + tmp_dir;
        NANO_CHECK(std::system(cmd.c_str()) == 0);
        std::FILE* f = std::fopen((tmp_dir + "/config.json").c_str(), "w");
        NANO_CHECK(f != nullptr);
        if (f != nullptr) {
            std::fputs(R"({"hidden_size": 896, "num_hidden_layers": 24})", f);
            std::fclose(f);
        }
        bool threw = false;
        std::string message;
        try {
            (void)nano::ModelConfig::from_dir(tmp_dir);
        } catch (const std::exception& e) {
            threw = true;
            message = e.what();
        }
        NANO_CHECK(threw);
        NANO_CHECK_MSG(message.find("num_attention_heads") != std::string::npos,
                       "error message was: %s", message.c_str());
    }

    return nano::testing::finish("test_config");
}
