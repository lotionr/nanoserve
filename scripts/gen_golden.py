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


def _floats(a) -> list:
    """np array -> JSON list. fp32 values roundtrip exactly through doubles."""
    return [float(x) for x in np.asarray(a, dtype=np.float32).ravel()]


def gen_ops_golden() -> None:
    """numpy reference outputs for the core fp32 ops (F015).

    Shapes are small enough to commit as JSON but representative of the model
    (dim 896 rmsnorm rows, head_dim 64 attention rows). Odd matmul sizes catch
    indexing bugs that square shapes hide.
    """
    rng = np.random.default_rng(20260818)
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    out = {"source": "numpy " + np.__version__}

    # linear: y = x @ W^T + bias, W stored [d_out, d_in] (HF nn.Linear layout)
    t, d_in, d_out = 3, 64, 48
    x = f32(rng.standard_normal((t, d_in)))
    w = f32(rng.standard_normal((d_out, d_in)))
    b = f32(rng.standard_normal(d_out))
    y = f32(x.astype(np.float64) @ w.astype(np.float64).T + b)
    out["linear"] = {
        "tokens": t, "d_in": d_in, "d_out": d_out,
        "x": _floats(x), "w": _floats(w), "bias": _floats(b), "y": _floats(y),
    }

    # matmul: c = a @ b, deliberately odd sizes
    m, k, n = 5, 33, 17
    a = f32(rng.standard_normal((m, k)))
    bb = f32(rng.standard_normal((k, n)))
    c = f32(a.astype(np.float64) @ bb.astype(np.float64))
    out["matmul"] = {
        "m": m, "k": k, "n": n,
        "a": _floats(a), "b": _floats(bb), "c": _floats(c),
    }

    # rmsnorm at the real model width and eps
    t, dim, eps = 3, 896, 1e-6
    x = f32(rng.standard_normal((t, dim)) * 3.0)
    w = f32(rng.standard_normal(dim))
    var = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    y = f32(x.astype(np.float64) / np.sqrt(var + eps) * w.astype(np.float64))
    out["rmsnorm"] = {
        "tokens": t, "dim": dim, "eps": eps,
        "x": _floats(x), "weight": _floats(w), "y": _floats(y),
    }

    # softmax over one attention-score row (head_dim-ish length, wide range)
    x = f32(rng.standard_normal(64) * 10.0)
    e = np.exp(x.astype(np.float64) - x.max())
    out["softmax"] = {"x": _floats(x), "y": _floats(e / e.sum())}

    # silu across the active range
    x = f32(np.linspace(-8.0, 8.0, 101))
    xd = x.astype(np.float64)
    out["silu"] = {"x": _floats(x), "y": _floats(xd / (1.0 + np.exp(-xd)))}

    # elementwise add / mul (residual + SwiGLU gate)
    a = f32(rng.standard_normal(64))
    b = f32(rng.standard_normal(64))
    out["add"] = {"a": _floats(a), "b": _floats(b), "y": _floats(a + b)}
    out["mul"] = {"a": _floats(a), "b": _floats(b), "y": _floats(a * b)}

    path = DATA_DIR / "ops_golden.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path}")


def gen_rope_golden() -> None:
    """RoPE reference from the actual HF Qwen2 implementation (F016)."""
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2RotaryEmbedding,
        apply_rotary_pos_emb,
    )

    cfg = AutoConfig.from_pretrained(str(MODEL_DIR))
    rot = Qwen2RotaryEmbedding(config=cfg)

    torch.manual_seed(20260818)
    positions = [0, 1, 2, 3, 17, 100, 1000]
    n_heads, n_kv_heads, head_dim = 14, 2, 64
    seq = len(positions)
    q = torch.randn(1, n_heads, seq, head_dim, dtype=torch.float32)
    k = torch.randn(1, n_kv_heads, seq, head_dim, dtype=torch.float32)
    pos_ids = torch.tensor([positions])
    cos, sin = rot(torch.zeros(1, seq, head_dim), pos_ids)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    def per_token(x: torch.Tensor) -> list:
        # [1, H, T, D] -> per token [H, D], the layout the C++ engine uses.
        return [_floats(x[0, :, t, :].numpy()) for t in range(seq)]

    out = {
        "source": "transformers Qwen2RotaryEmbedding + apply_rotary_pos_emb",
        "rope_theta": cfg.rope_parameters["rope_theta"],
        "head_dim": head_dim,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "positions": positions,
        "q_in": per_token(q), "q_out": per_token(q_rot),
        "k_in": per_token(k), "k_out": per_token(k_rot),
    }
    path = DATA_DIR / "rope_golden.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path}")


# Chat prompts for the forward-pass goldens (F017-F019). Fixed forever so the
# committed goldens stay valid; the C++ tests rebuild the same prompt ids with
# the chat-template helper (F013).
FORWARD_PROMPTS = [
    "What is 2+2?",
    "Write a haiku about caches.",
    "Name the capital of France.",
]


def _load_fp32_model():
    import torch
    from transformers import AutoModelForCausalLM

    # fp32, not bf16: the C++ engine computes in fp32, and greedy argmax must
    # compare against the same-precision reference.
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype=torch.float32)
    model.eval()
    return model


def _chat_ids(prompt: str) -> list:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )["input_ids"]


def gen_layer_golden(model=None) -> None:
    """F017: hidden states into and out of transformer layer 0 (HF fp32)."""
    import torch

    model = model or _load_fp32_model()
    ids = _chat_ids(FORWARD_PROMPTS[0])
    with torch.no_grad():
        out = model(torch.tensor([ids]), output_hidden_states=True)
    hs = out.hidden_states  # [0] = embeddings (layer 0 input), [1] = layer 0 output
    golden = {
        "source": "HF transformers fp32 forward, output_hidden_states",
        "prompt": FORWARD_PROMPTS[0],
        "ids": ids,
        "hidden_size": hs[0].shape[-1],
        "layer0_in": _floats(hs[0][0].numpy()),
        "layer0_out": _floats(hs[1][0].numpy()),
    }
    path = DATA_DIR / "layer_golden.json"
    path.write_text(json.dumps(golden))
    print(f"wrote {path} ({len(ids)} tokens)")


def gen_logits_golden(model=None) -> None:
    """F018: final-position logits for 3 prompts (HF fp32).

    The full vocab is 151936 logits per prompt — too large to commit as JSON —
    so the golden stores the argmax, the top-10 (ids + values), and a strided
    sample of 512 logits across the vocab. The C++ test checks all of these.
    """
    import numpy as np_
    import torch

    model = model or _load_fp32_model()
    cases = []
    for prompt in FORWARD_PROMPTS:
        ids = _chat_ids(prompt)
        with torch.no_grad():
            logits = model(torch.tensor([ids])).logits[0, -1].numpy()
        top10 = np_.argsort(logits)[::-1][:10]
        stride = len(logits) // 512
        sample_idx = list(range(0, len(logits), stride))
        cases.append(
            {
                "prompt": prompt,
                "ids": ids,
                "argmax": int(np_.argmax(logits)),
                "top10_ids": [int(i) for i in top10],
                "top10_logits": _floats(logits[top10]),
                "sample_stride": stride,
                "sample_logits": _floats(logits[sample_idx]),
            }
        )
    golden = {"source": "HF transformers fp32 forward, last-position logits",
              "cases": cases}
    path = DATA_DIR / "logits_golden.json"
    path.write_text(json.dumps(golden))
    print(f"wrote {path} ({len(cases)} prompts)")


# The three FORWARD_PROMPTS all hit <|im_end|> before 32 tokens under pure
# greedy; this one keeps going, so the golden also covers a full-length
# (no-eos) 32-token decode.
GENERATE_LONG_PROMPT = "Explain what a KV cache does in an LLM."


def gen_generate_golden(model=None) -> None:
    """F019: greedy 32-token continuations for chat prompts (HF fp32)."""
    import torch

    model = model or _load_fp32_model()
    cases = []
    for prompt in FORWARD_PROMPTS + [GENERATE_LONG_PROMPT]:
        ids = _chat_ids(prompt)
        input_ids = torch.tensor([ids])
        with torch.no_grad():
            # repetition_penalty=1.0 matters: the model's generation_config
            # ships 1.1, and HF applies it even with do_sample=False, which
            # is NOT pure greedy. The engine implements pure argmax decoding,
            # so the reference must too. (Found when the golden test diverged
            # at tokens that had already appeared in the prompt.)
            out = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                do_sample=False,
                repetition_penalty=1.0,
                max_new_tokens=32,
            )
        new_ids = out[0][len(ids):].tolist()
        cases.append({"prompt": prompt, "prompt_ids": ids, "generated_ids": new_ids})
    golden = {"source": "HF transformers fp32 greedy generate, max_new_tokens=32",
              "cases": cases}
    path = DATA_DIR / "generate_golden.json"
    path.write_text(json.dumps(golden))
    print(f"wrote {path} ({len(cases)} prompts)")


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
    gen_ops_golden()
    gen_rope_golden()
    model = _load_fp32_model()
    gen_layer_golden(model)
    gen_logits_golden(model)
    gen_generate_golden(model)


if __name__ == "__main__":
    main()
