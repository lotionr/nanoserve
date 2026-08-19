# F028/F029: pybind11 bindings.
#
# The load-bearing check is the CLI round trip: the Python module and the
# C++ CLI must produce identical token ids for the same greedy settings —
# same chat template, same engine, same stop handling. Streaming checks
# that chunks really arrive incrementally (per-chunk timestamps) and that
# the stream's final text/ids/stats agree with the non-streaming path.
#
# Skips (like the C++ tests) when the model isn't downloaded.
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "qwen2.5-0.5b-instruct"
CLI = ROOT / "build" / "nanoserve"

nanoserve = pytest.importorskip("nanoserve")

pytestmark = pytest.mark.skipif(
    not (MODEL / "config.json").exists(),
    reason="model not downloaded — run scripts/download_model.sh",
)


@pytest.fixture(scope="module")
def engine():
    return nanoserve.Engine(MODEL)


def cli_generate(prompt: str, n: int) -> tuple[list[int], str]:
    """Runs the C++ CLI and returns (generated ids, generated text)."""
    out = subprocess.run(
        [str(CLI), "generate", str(MODEL), "-p", prompt, "-n", str(n), "--greedy"],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    lines = out.stdout.splitlines()
    ids_line = next(line for line in lines if line.startswith("ids:"))
    ids = [int(x) for x in ids_line.split()[1:]]
    # Generated text sits between the two "---" marker lines.
    first, last = lines.index("---"), len(lines) - 1 - lines[::-1].index("---")
    text = "\n".join(lines[first + 1 : last])
    return ids, text


def test_generate_matches_cli(engine):
    if not CLI.exists():
        pytest.skip("CLI not built — run init.sh")
    prompt = "What is 2+2?"
    cli_ids, cli_text = cli_generate(prompt, 16)
    result = engine.generate(prompt, max_tokens=16)  # temperature=0 -> greedy
    assert result.token_ids == cli_ids
    assert result.text == cli_text


def test_result_stats_populated(engine):
    result = engine.generate("Name the capital of France.", max_tokens=16)
    assert result.token_ids, "no tokens generated"
    assert result.ttft_ms > 0
    assert result.tokens_per_second > 0
    # Sanity, not benchmarks: an M-series CPU decodes this model somewhere
    # between 1 and 1000 tok/s; anything outside that is a stats bug.
    assert 1 < result.tokens_per_second < 1000


def test_sampling_deterministic_by_seed(engine):
    kwargs = dict(max_tokens=12, temperature=0.8, top_k=50, top_p=0.9)
    a = engine.generate("Write one short sentence about GPUs.", seed=7, **kwargs)
    b = engine.generate("Write one short sentence about GPUs.", seed=7, **kwargs)
    c = engine.generate("Write one short sentence about GPUs.", seed=8, **kwargs)
    assert a.token_ids == b.token_ids
    assert a.token_ids != c.token_ids


def test_stream_yields_incrementally(engine):
    stream = engine.generate("Write a haiku about caches.", max_tokens=24, stream=True)
    chunks, stamps = [], []
    for chunk in stream:
        chunks.append(chunk)
        stamps.append(time.perf_counter())
    assert len(chunks) >= 4, "expected several chunks, got one blob"
    assert all(isinstance(c, str) and c for c in chunks)
    # Incremental means spread over time: with >= 4 decode steps at real
    # model speed, first-to-last must span at least a few milliseconds.
    assert stamps[-1] - stamps[0] > 0.005
    assert "".join(chunks) == stream.text


def test_stream_matches_generate(engine):
    prompt = "Explain what a KV cache does in an LLM."
    stream = engine.generate(prompt, max_tokens=24, stream=True)
    streamed_text = "".join(stream)
    result = engine.generate(prompt, max_tokens=24)
    assert stream.token_ids == result.token_ids
    assert streamed_text == result.text
    assert stream.ttft_ms > 0
    assert stream.tokens_per_second > 0


def test_invalid_max_tokens(engine):
    with pytest.raises(ValueError):
        engine.generate("hi", max_tokens=0)
