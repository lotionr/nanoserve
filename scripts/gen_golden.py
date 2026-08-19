#!/usr/bin/env python3
"""Regenerates golden test vectors in tests/data/ from the real model files.

Golden files are committed, so C++ tests never need Python — this script only
runs when the corpus changes. Requires the venv:

    python3 -m venv .venv
    .venv/bin/pip install tokenizers numpy
    .venv/bin/python scripts/gen_golden.py
"""

import json
import struct
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "qwen2.5-0.5b-instruct"
DATA_DIR = ROOT / "tests" / "data"

# English/ASCII corpus for F011/F012. Non-ASCII parity is feature F014 and
# gets its own corpus when the Unicode tables land.
TOKENIZER_CORPUS = [
    "Hello, world!",
    "hello world",
    " leading space",
    "trailing space ",
    "The quick brown fox jumps over the lazy dog.",
    "I'm sure it's fine — we've seen worse, they'll manage, you'd agree.",
    "DON'T SHOUT'RE WEIRD 'll 'D",
    "x = (a + b) * c / d - e;",
    "def f(x):\n    return x * 2\n",
    "line one\nline two\r\nline three\n\n\nend",
    "tabs\tand\tspaces   mixed \t \n ok",
    "1234567890",
    "In 2026, 3.14159 is still pi; 1,000,000 > 999999.",
    "version v2.5.0-rc.1+build.42",
    "email@example.com and https://example.com/path?q=1&r=2#frag",
    "!@#$%^&*()_+-=[]{}|;':\",./<>?",
    "...ellipsis... and -- dashes --- everywhere",
    "  double  spaces   triple   ",
    "a",
    " ",
    "\n",
    "snake_case camelCase PascalCase SCREAMING_SNAKE kebab-case",
    "C++ isn't C, and C# isn't C++ either.",
    "wait... what?! (really?) [yes] {no} <maybe>",
    "The KV cache stores keys/values per layer: 24 layers x 2 heads x 64 dims.",
]

SPECIALS_CASES = [
    "<|im_start|>user\nHello<|im_end|>\n",
    "<|endoftext|>",
    "before<|im_start|>after",
]

# Chat-template cases for F013. Reference ids come from HF transformers'
# apply_chat_template on the real tokenizer_config.json jinja template; the
# C++ helper re-implements only the no-tools branch of that template.
CHAT_CASES = [
    {
        "name": "single user turn, default system",
        "messages": [{"role": "user", "content": "Hello"}],
        "add_generation_prompt": True,
    },
    {
        "name": "explicit system message",
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "What is a KV cache?"},
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "multi-turn with assistant history",
        "messages": [
            {"role": "user", "content": "Name a prime number."},
            {"role": "assistant", "content": "7"},
            {"role": "user", "content": "Another one, please!"},
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "no generation prompt",
        "messages": [{"role": "user", "content": "Hi"}],
        "add_generation_prompt": False,
    },
    {
        "name": "content with newlines and punctuation",
        "messages": [
            {"role": "user", "content": "def f(x):\n    return x * 2\n\nExplain, briefly?"},
        ],
        "add_generation_prompt": True,
    },
]


def gen_tokenizer_golden() -> None:
    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    cases = []
    for text in TOKENIZER_CORPUS:
        ids = tok.encode(text, add_special_tokens=False).ids
        cases.append({"text": text, "ids": ids})
    specials = []
    for text in SPECIALS_CASES:
        ids = tok.encode(text, add_special_tokens=False).ids
        specials.append({"text": text, "ids": ids})
    out = {
        "source": "Qwen/Qwen2.5-0.5B-Instruct tokenizer.json via HF tokenizers",
        "cases": cases,
        "specials": specials,
    }
    path = DATA_DIR / "tokenizer_golden.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path} ({len(cases)} cases, {len(specials)} specials)")


def gen_chat_golden() -> None:
    """Reference prompt ids from HF apply_chat_template (F013)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    cases = []
    for case in CHAT_CASES:
        ids = tok.apply_chat_template(
            case["messages"],
            tokenize=True,
            add_generation_prompt=case["add_generation_prompt"],
        )["input_ids"]
        cases.append(
            {
                "name": case["name"],
                "messages": case["messages"],
                "add_generation_prompt": case["add_generation_prompt"],
                "ids": ids,
            }
        )
    out = {
        "source": "Qwen/Qwen2.5-0.5B-Instruct via transformers apply_chat_template",
        "cases": cases,
    }
    path = DATA_DIR / "chat_golden.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path} ({len(cases)} cases)")


def gen_embed_golden() -> None:
    """First N bf16 values of model.embed_tokens.weight, widened to fp32.

    bf16 -> fp32 is exact (shift into the top 16 bits), and every fp32 value
    round-trips exactly through a JSON double, so the C++ test compares with ==.
    """
    path = MODEL_DIR / "model.safetensors"
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        info = header["model.embed_tokens.weight"]
        assert info["dtype"] == "BF16", info["dtype"]
        begin, _end = info["data_offsets"]
        n_values = 64
        f.seek(8 + header_len + begin)
        raw = f.read(2 * n_values)
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    f32 = u16.view(np.float32)

    tensor_count = sum(1 for k in header if k != "__metadata__")
    total_params = sum(
        int(np.prod(v["shape"])) for k, v in header.items() if k != "__metadata__"
    )
    out = {
        "tensor": "model.embed_tokens.weight",
        "dtype": "BF16",
        "shape": info["shape"],
        "tensor_count": tensor_count,
        "total_params": total_params,
        "first_values_f32": [float(x) for x in f32],
    }
    dest = DATA_DIR / "embed_golden.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest} ({tensor_count} tensors, {total_params} params)")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gen_tokenizer_golden()
    gen_chat_golden()
    gen_embed_golden()


if __name__ == "__main__":
    main()
