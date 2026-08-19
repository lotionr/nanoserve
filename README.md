# nanoserve

A small LLM inference engine, written from scratch to understand how serving systems
actually work — weight loading, tokenization, the transformer forward pass, KV caching,
quantization, and benchmarking — with no ML framework underneath. The core is plain
C++20 with zero third-party dependencies; the benchmark harness and user-facing API
are Python.

This is a learning-in-public project. Everything here is measured, not claimed:
**every benchmark number in this README is machine-generated from a run of the
reproducible suite in `bench/` — none are typed by hand**, and every feature listed
as done has a test behind it (`feature_list.json` tracks exactly what passes today).

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
- [x] Byte-level BPE tokenizer (encode/decode) with full Unicode pretokenizer
      parity and NFC normalization, verified against HF `tokenizers` golden
      vectors; Qwen chat-template helper verified against `apply_chat_template`
- [x] `nanoserve` CLI: `inspect` (tensor table for a safetensors file), `tokenize`,
      `generate`
- [x] fp32 forward pass (RMSNorm / RoPE / GQA attention / SwiGLU), verified
      layer-by-layer and logits-level against HF transformers
- [x] KV cache + greedy decoding, matching HF greedy continuations token for token
- [x] Sampling (temperature, top-k, top-p) with a seeded RNG, matching the HF
      logits warpers; streaming output with UTF-8-safe incremental detokenization
- [x] Allocation-free decode loop (verified with an allocation counter)
- [x] Threaded matmul (persistent thread pool) and NEON SIMD kernels with a
      scalar fallback, both tested for parity against the reference path
- [x] int8 weight quantization (per-row scales, offline `nanoserve quantize` pass)
      with an integer NEON sdot kernel: activations quantized per 32-value block
      at runtime, dot products in int8 x int8 -> int32 (llama.cpp Q8_0 style);
      quality gated against the fp32 goldens — top-1 logits unchanged, greedy
      continuations identical on 3 of 4 golden prompts, the one divergence
      verified to sit on a 0.03-logit-margin near-tie (test enforces this)
- [x] Python bindings (pybind11 + scikit-build-core): `nanoserve.Engine(...)`
      with `generate()` and streaming, verified to match the CLI token for token
- [x] Benchmark harness (Python): tok/s + time-to-first-token vs llama.cpp,
      reproducible script, results table generated into this README
- [x] Metal GPU backend for the matmul path (`--backend metal`, off by default):
      runtime-compiled MSL kernels, zero-copy weight residency via unified memory,
      parity-tested against the CPU kernels (worst error 2.9e-5 against a 1e-3
      bound); fp32 greedy output identical to the HF goldens, int8 greedy
      bit-identical to the fp32 goldens on all 4 prompts (the GPU int8 kernel
      skips activation quantization, so it is slightly *more* faithful than the
      CPU sdot path). The honest headline: at 0.5B, GPU decode is SLOWER than the
      CPU (each projection pays a synchronous dispatch round-trip, ~121 per
      token), so the CPU stays the default — see the benchmark table
- [x] Paged KV cache: 16-token pages, per-sequence block tables, and a shared
      free-list page pool — the vLLM PagedAttention memory model, minus prefix
      sharing (the design note in `src/model/paged_kv.hpp` maps the two
      name-for-name and lists what was deliberately left out). Paged logits are
      bit-identical to the contiguous cache across page boundaries; a short
      generation's KV memory high-water is ~20x below the max-context
      preallocation (test prints both); allocator invariants (no unbacked or
      double-allocated page, exact reuse after free) survive a model-free
      alloc/free churn test across interleaved sequences (`test_paged_alloc`)
- [x] HTTP serving endpoint: `nanoserve serve <model_dir>` exposes
      `POST /v1/completions` (the classic OpenAI completions schema — prompt,
      max_tokens, temperature/top_k/top_p/seed, usage counts) with `stream:true`
      delivering Server-Sent Events token by token; HTTP/1.1 server written on
      raw POSIX sockets like everything else here. The engine serves from the
      paged KV cache. End-to-end test spawns the real binary and speaks real
      HTTP: schema and usage checked against a locally loaded tokenizer, greedy
      determinism across requests, streamed chunks reassembling to the
      non-streamed text, and the 400/404/405 error paths
- [ ] Stretch: continuous batching in the server

## Benchmarks

All numbers below are generated by `bench/report.py` from JSON produced by
`bench/run.py` on the machine described in the table — never typed by hand.
Both engines run on the same weights; the backend column says where nanoserve's
matmuls run (llama.cpp is CPU-only here — details and caveats below the table).

<!-- BENCH:BEGIN (generated by bench/report.py — do not edit by hand) -->

Workload: 41-token chat prompt ("Explain what a KV cache does in an LLM."), 64 greedy tokens, 1 warmup run.
Hardware: Apple M3 Pro (5P + 6E cores, 18 GB), macOS 15.5.

| engine | precision | backend | threads | prefill tok/s | decode tok/s | TTFT ms |
|---|---|---|---:|---:|---:|---:|
| nanoserve (`cea0581`) | fp32 | cpu | 5 | 144.5 ± 0.4 | 50.8 ± 3.2 | 284 |
| nanoserve (`cea0581`) | fp32 | metal | 5 | 144.2 ± 3.0 | 30.9 ± 0.9 | 284 |
| llama.cpp (`5112b97`) | f32 | cpu | 5 | 560.3 ± 2.5 | 55.4 ± 2.5 | 73 |
| nanoserve (`cea0581`) | int8 | cpu | 5 | 546.5 ± 2.2 | 147.0 ± 0.4 | 75 |
| nanoserve (`cea0581`) | int8 | metal | 5 | 150.4 ± 0.4 | 38.2 ± 0.2 | 273 |
| llama.cpp (`5112b97`) | q8_0 | cpu | 5 | 1428.1 ± 33.5 | 195.6 ± 7.3 | 29 |

- **fp32 vs f32 decode:** llama.cpp is **1.09x faster** (55.4 vs 50.8 tok/s).
- **int8 vs Q8_0 decode:** llama.cpp is **1.33x faster** (195.6 vs 147.0 tok/s).
- **fp32 metal decode:** 1.64x slower than our CPU path (30.9 vs 50.8 tok/s).
- **fp32 metal prefill:** at parity with our CPU path (144.2 vs 144.5 tok/s).
- **int8 metal decode:** 3.85x slower than our CPU path (38.2 vs 147.0 tok/s).
- **int8 metal prefill:** 3.63x slower than our CPU path (150.4 vs 546.5 tok/s).

Caveats, stated plainly: the f32 GGUF is converted from the exact safetensors file nanoserve loads, so the fp32 row is like-for-like; the int8 WEIGHT formats are NOT identical (nanoserve: per-row scales, 8.0 bits/weight; llama.cpp Q8_0: per-32-block scales, 8.5 bits/weight — finer-grained and slightly larger). On the CPU, both engines quantize activations to int8 per 32-value block at runtime and compute the dot products with integer SIMD (sdot). nanoserve numbers come from the engine's internal prefill/decode timers on a real prompt; llama.cpp numbers come from `llama-bench` with the same token counts, threads, and repeat count (its decode test starts from an empty context — at 0.5B the KV-read difference is ~3 MB/token vs ~0.5-2 GB/token of weights, i.e. under the noise floor). The thread count was chosen by measuring both engines at 5 (performance cores) and 11 (all hardware threads): decode is fastest at 5 for BOTH engines on this chip (the sweep files sit in bench/results/). llama.cpp is its default macOS CPU build, which uses Accelerate BLAS for prompt processing — much of its prefill lead is Apple's GEMM, not ggml kernels.

The metal rows (F033) offload ONLY the linear projections above a dispatch-amortization floor to the M-series GPU: attention, norms, RoPE, and the small per-token k/v projections stay on the CPU, and each GPU projection is one synchronous command-buffer round-trip (the CPU consumes every result immediately, so there is nothing to pipeline without restructuring the engine). Decode therefore pays that round-trip ~121 times per token, which is the number to watch in the decode column. Weights are GPU-resident zero-copy (unified memory; registered once at load). The GPU int8 kernel dequantizes weights in-register against fp32 activations — no activation quantization, unlike the CPU sdot path, so it is slightly MORE accurate (bit-identical to the fp32 goldens on all 4 greedy prompts, where the CPU int8 path has one documented near-tie flip). llama.cpp stays CPU-only in this table.

Reproduce:

```sh
bash bench/setup_llamacpp.sh   # clone+build llama.cpp (pinned), convert GGUFs
.venv/bin/python bench/run.py --engine all --repeats 5
.venv/bin/python bench/report.py
```

<!-- BENCH:END -->

## Python

```sh
.venv/bin/pip install -e .    # builds the C++ core + pybind11 module (scikit-build-core)
```

```python
import nanoserve

engine = nanoserve.Engine("models/qwen2.5-0.5b-instruct", int8=True)
result = engine.generate("What does a KV cache store?", max_tokens=64)
print(result.text, result.ttft_ms, result.tokens_per_second)

for chunk in engine.generate("Tell me a story.", max_tokens=64, stream=True):
    print(chunk, end="", flush=True)
```

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
