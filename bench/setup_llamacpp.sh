#!/usr/bin/env bash
# Sets up the llama.cpp comparison baseline (F031):
#   1. clones llama.cpp at the commit pinned in bench/llamacpp.lock
#   2. builds llama-bench + llama-quantize (Metal off: nanoserve is
#      CPU-only, so the comparison must be CPU vs CPU)
#   3. converts the local Qwen2.5-0.5B-Instruct safetensors to an f32 GGUF
#      and quantizes a Q8_0 from it
#
# Everything lands under bench/llama.cpp/ and bench/models/ (git-ignored).
# Idempotent: each step is skipped if its output already exists.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

LOCK_FILE=bench/llamacpp.lock
LLAMA_DIR=bench/llama.cpp
MODEL_DIR=models/qwen2.5-0.5b-instruct
GGUF_DIR=bench/models
PYTHON=.venv/bin/python

COMMIT=$(cat "$LOCK_FILE")
echo "== llama.cpp pinned commit: $COMMIT"

if [ ! -d "$LLAMA_DIR/.git" ]; then
    echo "== cloning llama.cpp"
    git init -q "$LLAMA_DIR"
    git -C "$LLAMA_DIR" remote add origin https://github.com/ggml-org/llama.cpp
fi
if [ "$(git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null || true)" != "$COMMIT" ]; then
    echo "== fetching $COMMIT"
    git -C "$LLAMA_DIR" fetch -q --depth 1 origin "$COMMIT"
    git -C "$LLAMA_DIR" checkout -q FETCH_HEAD
fi

if [ ! -x "$LLAMA_DIR/build/bin/llama-bench" ]; then
    echo "== building llama.cpp (CPU only)"
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_TOOLS=ON > /dev/null
    cmake --build "$LLAMA_DIR/build" -j --target llama-bench llama-quantize > /dev/null
fi
"$LLAMA_DIR/build/bin/llama-bench" --version 2>&1 | head -1 || true

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "error: download the model first (scripts/download_model.sh)" >&2
    exit 1
fi

mkdir -p "$GGUF_DIR"
F32_GGUF="$GGUF_DIR/qwen2.5-0.5b-instruct-f32.gguf"
Q8_GGUF="$GGUF_DIR/qwen2.5-0.5b-instruct-q8_0.gguf"

if [ ! -f "$F32_GGUF" ]; then
    echo "== converting safetensors -> f32 GGUF (same fp32 weights nanoserve runs)"
    # sentencepiece: the converter imports it while PROBING for a
    # sentencepiece vocab before falling back to Qwen's BPE one.
    "$PYTHON" -c "import gguf, sentencepiece" 2>/dev/null || \
        "$PYTHON" -m pip install -q gguf sentencepiece
    # The converter globs every *.safetensors in the directory, which would
    # pick up nanoserve's model.int8.safetensors too — convert from a clean
    # directory of symlinks to just the HF checkpoint files.
    CLEAN_DIR="$GGUF_DIR/hf-clean"
    rm -rf "$CLEAN_DIR" && mkdir -p "$CLEAN_DIR"
    for f in config.json generation_config.json tokenizer.json \
             tokenizer_config.json vocab.json merges.txt model.safetensors; do
        [ -f "$MODEL_DIR/$f" ] && ln -s "$(cd "$MODEL_DIR" && pwd)/$f" "$CLEAN_DIR/$f"
    done
    "$PYTHON" "$LLAMA_DIR/convert_hf_to_gguf.py" "$CLEAN_DIR" \
        --outfile "$F32_GGUF" --outtype f32 > /dev/null
    rm -rf "$CLEAN_DIR"
fi

if [ ! -f "$Q8_GGUF" ]; then
    echo "== quantizing f32 GGUF -> Q8_0"
    "$LLAMA_DIR/build/bin/llama-quantize" "$F32_GGUF" "$Q8_GGUF" Q8_0 > /dev/null
fi

ls -lh "$GGUF_DIR"
echo "== setup complete"
