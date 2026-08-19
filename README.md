# nanoserve

A small LLM inference engine, written from scratch to understand how serving systems
actually work — weight loading, tokenization, the transformer forward pass, KV caching,
quantization, and benchmarking — with no ML framework underneath. The core is plain
C++20 with zero third-party dependencies; the benchmark harness and user-facing API
are Python.

This is a learning-in-public project. Everything here is measured, not claimed:
there are **no benchmark numbers in this README until the benchmark suite exists and
has been run reproducibly**, and every feature listed as done has a test behind it
(`feature_list.json` tracks exactly what passes today).

## Why

Understanding inference serving at the level of "I built each piece myself" — how a
safetensors file is laid out, why decode is memory-bound while prefill is compute-bound,
what a KV cache actually stores and why paging it matters, what int8 quantization does
to quality and to bandwidth — rather than at the level of calling a library. The target
is a single-batch CPU engine for one small open model, benchmarked honestly against
llama.cpp on the same machine, then extended layer by layer (threading, SIMD, int8,
Metal) with the effect of each layer measured.

## Model

[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) —
Apache-2.0 licensed, ungated (no account needed), ~1 GB of bf16 safetensors, and small
enough to iterate on quickly while still being a real, modern transformer
(GQA, RoPE, RMSNorm, SwiGLU, tied embeddings).

```sh
scripts/download_model.sh   # -> models/qwen2.5-0.5b-instruct/ (git-ignored)
```

## Status

Current state (kept honest; see `feature_list.json` for the full sequenced list):

- [x] safetensors loader — mmap, header parsing, bounds validation, bf16→fp32
- [x] Model config parsing (`config.json` → typed struct)
- [x] Byte-level BPE tokenizer (encode/decode), verified against HF `tokenizers`
      golden vectors for English text; full Unicode pretokenizer parity is a
      tracked follow-up
- [x] `nanoserve` CLI: `inspect` (tensor table for a safetensors file), `tokenize`
- [ ] fp32 forward pass (RMSNorm / RoPE / GQA attention / SwiGLU)
- [ ] KV cache + greedy decoding
- [ ] Sampling (temperature, top-k, top-p)
- [ ] Threaded + SIMD matmul, int8 quantization
- [ ] Python bindings (`import nanoserve; nanoserve.generate(...)`)
- [ ] Benchmark harness (Python): tok/s + time-to-first-token vs llama.cpp,
      reproducible script, results table generated into this README
- [ ] Stretch: Metal backend, paged KV cache, HTTP serving endpoint

## Layout

```
src/core/     tensor, dtype, mmap'd safetensors loader, minimal JSON parser
src/model/    model config, tokenizer (later: transformer forward, KV cache)
src/server/   nanoserve CLI (later: HTTP endpoint)
tests/        plain-executable ctest tests (llama.cpp style), golden vectors in tests/data/
bench/        Python benchmark harness (tok/s, TTFT, llama.cpp comparison)
scripts/      model download, golden-vector generation
```

## Build

Requires CMake ≥ 3.24 and a C++20 compiler (tested: Apple clang 17, GCC on Linux CI).
No third-party C++ dependencies.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The build is warnings-as-errors (`-Wall -Wextra -Wpedantic -Werror`). Tests that need
model weights skip cleanly when `models/` is absent, so CI runs without downloading
anything.

```sh
./build/nanoserve inspect models/qwen2.5-0.5b-instruct/model.safetensors
./build/nanoserve tokenize models/qwen2.5-0.5b-instruct "Hello, world"
```

## Design notes

- **Readability over cleverness.** This codebase is meant to be walked through line
  by line. Where a trick would save 10% and cost clarity, the trick loses — until the
  benchmark phase, where any optimization has to justify itself with a measured number.
- **No dependencies in the core.** The JSON parser, BPE tokenizer, and safetensors
  reader are written here, because writing them is the point.
- **Tests are executables.** Each test is a small `main()` that returns non-zero on
  failure (exit 77 = skipped), registered with ctest — the same pattern llama.cpp uses.

## License

MIT. Model weights are downloaded separately from Hugging Face under their own
license (Qwen2.5-0.5B-Instruct: Apache-2.0) and are never redistributed here.
