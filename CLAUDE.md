# nanoserve — Project Context for Claude Code

This file is read automatically by Claude Code at the start of every session.

## What is this project?

**nanoserve** is a from-scratch LLM inference engine: C++20 core (no third-party
deps), Python bindings and benchmark harness. It exists to turn inference-systems
understanding (weight formats, KV caching, quantization, serving, benchmarking) into
a public, tested, benchmarked artifact. Target model: Qwen2.5-0.5B-Instruct
(Apache-2.0, ungated).

## How to work on this project

**Always start every session by running:**
```bash
bash init.sh
```
It verifies the toolchain, builds warnings-as-errors, and runs ctest.

## Key files

| File | Purpose |
|------|---------|
| `init.sh` | Env check + clean build + ctest. Always run first. |
| `scripts/download_model.sh` | Fetches Qwen2.5-0.5B-Instruct into `models/` (git-ignored). |
| `scripts/gen_golden.py` | Regenerates golden test vectors in `tests/data/` (needs `.venv`). |

## Critical rules

- **Never mark a feature as done without its verification steps actually passing**
- Always end a session with **git commits** (small, conventional)
- The build must be **green (ctest passing) at the end of every session**
- **No benchmark numbers anywhere (README, comments, commits) unless produced by the
  bench harness on this machine** — placeholders like [X] until then
- **Readability over cleverness** in the C++: the author must be able to explain every
  line in an interview. Prefer the obvious implementation until the perf phase, then
  justify each optimization with a measured number.
- The story this repo tells is *inference systems understanding* (KV cache, quantization,
  serving, benchmarking) — not "C++ expertise". Keep README/docs framed that way.

## Tech stack

- C++20, CMake ≥ 3.24, zero third-party C++ deps (JSON/BPE/safetensors hand-written)
- Tests: plain executables + ctest (exit 77 = skip when model files absent)
- Python 3.13 (`.venv/`): golden-vector generation (`tokenizers`, `numpy`,
  `safetensors`), later pybind11 bindings + benchmark harness
- Model: Qwen2.5-0.5B-Instruct bf16 safetensors in `models/` (git-ignored)
