#!/usr/bin/env python3
"""nanoserve benchmark harness (F030/F031).

Measures prefill throughput (and TTFT) plus decode throughput for nanoserve
and/or llama.cpp on this machine, with warmup, N repeats, and dispersion,
then writes JSON to bench/results/ and prints a table.

    .venv/bin/python bench/run.py --engine nanoserve --repeats 5
    .venv/bin/python bench/run.py --engine llama.cpp --repeats 5
    .venv/bin/python bench/run.py --engine all --repeats 5

Methodology (and its honest caveats):
- nanoserve runs in-process through the pybind11 bindings: model loaded
  once, one warmup generation, then N timed generations of a fixed pinned
  prompt (41 chat-template tokens) x 64 greedy tokens. TTFT and per-step
  decode times come from the engine's internal clocks (same ones the CLI
  --stats prints); prefill tok/s = prompt_tokens / ttft.
- llama.cpp runs via llama-bench (its own standard benchmarking tool, one
  warmup pass per test) with the SAME token counts, thread count, and
  repeat count: -p 41 measures prefill, -n 64 measures decode. llama-bench
  generates from an empty context rather than after a 41-token prompt; at
  0.5B scale the KV-cache read this skips is ~3 MB/token vs ~0.5-2 GB/token
  of weight traffic, well under measurement noise.
- Quantization formats are NOT identical: nanoserve int8 uses per-row
  scales, llama.cpp Q8_0 uses per-32-block scales (finer, slightly more
  bytes/weight). fp32 vs f32-GGUF is exact like-for-like — the GGUF is
  converted from the very same safetensors file.
- Both engines run CPU-only (the llama.cpp build has Metal disabled).
"""

import argparse
import datetime
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "qwen2.5-0.5b-instruct"
RESULTS_DIR = ROOT / "bench" / "results"
LLAMA_BENCH = ROOT / "bench" / "llama.cpp" / "build" / "bin" / "llama-bench"
GGUF = {
    "f32": ROOT / "bench" / "models" / "qwen2.5-0.5b-instruct-f32.gguf",
    "q8_0": ROOT / "bench" / "models" / "qwen2.5-0.5b-instruct-q8_0.gguf",
}

# Pinned workload: this prompt renders to 41 chat-template tokens (asserted
# at runtime) and greedily decodes past 64 tokens without hitting eos.
PROMPT = "Explain what a KV cache does in an LLM."
PROMPT_TOKENS = 41
GEN_TOKENS = 64

# Dispersion above this fraction of the mean gets flagged in the table (F030).
VARIANCE_LIMIT = 0.10


def sysctl(name: str) -> str:
    out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def machine_info() -> dict:
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }
    if platform.system() == "Darwin":
        info.update(
            chip=sysctl("machdep.cpu.brand_string"),
            performance_cores=sysctl("hw.perflevel0.physicalcpu"),
            efficiency_cores=sysctl("hw.perflevel1.physicalcpu"),
            hardware_threads=sysctl("hw.logicalcpu"),
            ram_gb=round(int(sysctl("hw.memsize")) / 2**30),
            macos=platform.mac_ver()[0],
        )
    return info


def mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def nanoserve_version() -> dict:
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"engine_version": f"nanoserve@{commit}"}


def bench_nanoserve(precision: str, repeats: int, threads: int) -> dict:
    import nanoserve

    nanoserve.set_num_threads(threads)
    engine = nanoserve.Engine(MODEL_DIR, int8=(precision == "int8"))

    engine.generate(PROMPT, max_tokens=GEN_TOKENS)  # warmup (page-in, pool spinup)

    ttft, prefill_tps, decode_tps = [], [], []
    generated = None
    for _ in range(repeats):
        r = engine.generate(PROMPT, max_tokens=GEN_TOKENS)
        assert r.prompt_tokens == PROMPT_TOKENS, (
            f"pinned prompt changed: {r.prompt_tokens} tokens != {PROMPT_TOKENS}"
        )
        assert len(r.token_ids) == GEN_TOKENS, (
            f"generation stopped early ({len(r.token_ids)} tokens) — "
            "pick a prompt that decodes past GEN_TOKENS"
        )
        ttft.append(r.ttft_ms)
        prefill_tps.append(r.prompt_tokens / r.ttft_ms * 1000.0)
        decode_tps.append(r.tokens_per_second)
        generated = len(r.token_ids)

    return {
        "engine": "nanoserve",
        "precision": precision,
        "threads": threads,
        "prompt_tokens": PROMPT_TOKENS,
        "gen_tokens": generated,
        "repeats": repeats,
        "ttft_ms": mean_std(ttft),
        "prefill_tps": mean_std(prefill_tps),
        "decode_tps": mean_std(decode_tps),
        **nanoserve_version(),
    }


def bench_llamacpp(precision: str, repeats: int, threads: int) -> dict:
    gguf = GGUF[precision]
    if not LLAMA_BENCH.exists() or not gguf.exists():
        sys.exit("llama.cpp baseline missing — run bench/setup_llamacpp.sh first")

    cmd = [
        str(LLAMA_BENCH),
        "-m", str(gguf),
        "-p", str(PROMPT_TOKENS),
        "-n", str(GEN_TOKENS),
        "-r", str(repeats),
        "-t", str(threads),
        "-o", "json",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1800)
    tests = json.loads(out.stdout)

    def pick(n_prompt: int, n_gen: int) -> dict:
        for t in tests:
            if t["n_prompt"] == n_prompt and t["n_gen"] == n_gen:
                return t
        raise KeyError(f"llama-bench produced no (p={n_prompt}, n={n_gen}) test")

    pp = pick(PROMPT_TOKENS, 0)   # prefill
    tg = pick(0, GEN_TOKENS)      # decode
    prefill_tps = {"mean": pp["avg_ts"], "std": pp["stddev_ts"], "n": repeats}
    return {
        "engine": "llama.cpp",
        "precision": precision,
        "threads": threads,
        "prompt_tokens": PROMPT_TOKENS,
        "gen_tokens": GEN_TOKENS,
        "repeats": repeats,
        # TTFT derived from prefill throughput (llama-bench reports tok/s).
        "ttft_ms": {
            "mean": PROMPT_TOKENS / pp["avg_ts"] * 1000.0,
            "std": 0.0,
            "n": repeats,
            "derived": "prompt_tokens / prefill_tps",
        },
        "prefill_tps": prefill_tps,
        "decode_tps": {"mean": tg["avg_ts"], "std": tg["stddev_ts"], "n": repeats},
        "engine_version": f"llama.cpp@{pp.get('build_commit', 'unknown')}",
        "gguf_type": pp.get("model_type", ""),
        # Build/settings provenance, straight from llama-bench's own report.
        "llama_build_number": pp.get("build_number"),
        "llama_backends": pp.get("backends"),
        "llama_cpu_info": pp.get("cpu_info"),
        "llama_n_batch": pp.get("n_batch"),
        "model_size_bytes": pp.get("model_size"),
    }


def flag(stat: dict) -> str:
    high = stat["mean"] > 0 and stat["std"] / stat["mean"] > VARIANCE_LIMIT
    return " HIGH-VARIANCE" if high else ""


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'engine':<12} {'precision':<10} {'threads':>7} "
        f"{'prefill tok/s':>18} {'decode tok/s':>18} {'ttft ms':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        pf, dc = r["prefill_tps"], r["decode_tps"]
        print(
            f"{r['engine']:<12} {r['precision']:<10} {r['threads']:>7} "
            f"{pf['mean']:>10.1f} ± {pf['std']:>5.1f} "
            f"{dc['mean']:>10.1f} ± {dc['std']:>5.1f} "
            f"{r['ttft_ms']['mean']:>10.1f}"
            f"{flag(pf)}{flag(dc)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", choices=["nanoserve", "llama.cpp", "all"],
                        default="nanoserve")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=0,
                        help="0 = all hardware threads (both engines)")
    args = parser.parse_args()

    threads = args.threads or int(sysctl("hw.logicalcpu"))

    plan = []
    if args.engine in ("nanoserve", "all"):
        plan += [("nanoserve", "fp32"), ("nanoserve", "int8")]
    if args.engine in ("llama.cpp", "all"):
        plan += [("llama.cpp", "f32"), ("llama.cpp", "q8_0")]

    rows = []
    for engine, precision in plan:
        print(f"benchmarking {engine} {precision} "
              f"(threads={threads}, repeats={args.repeats}) ...", flush=True)
        fn = bench_nanoserve if engine == "nanoserve" else bench_llamacpp
        rows.append(fn(precision, args.repeats, threads))

    result = {
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "machine": machine_info(),
        "workload": {
            "prompt": PROMPT,
            "prompt_tokens": PROMPT_TOKENS,
            "gen_tokens": GEN_TOKENS,
            "sampling": "greedy",
            "warmup_runs": 1,
        },
        "rows": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = result["created_utc"].replace(":", "").replace("-", "").replace("+0000", "Z")
    out_path = RESULTS_DIR / f"results-{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out_path.relative_to(ROOT)}\n")
    print_table(rows)


if __name__ == "__main__":
    main()
