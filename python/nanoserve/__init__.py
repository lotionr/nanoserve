"""nanoserve: a from-scratch LLM inference engine (Qwen2.5-0.5B-Instruct).

    import nanoserve

    engine = nanoserve.Engine("models/qwen2.5-0.5b-instruct")
    result = engine.generate("What is a KV cache?", max_tokens=64)
    print(result.text, result.ttft_ms, result.tokens_per_second)

    for chunk in engine.generate("Tell me a story.", max_tokens=64, stream=True):
        print(chunk, end="", flush=True)
"""

from nanoserve._nanoserve import (
    GenerateResult,
    TokenStream,
    __version__,
    set_num_threads,
)
from nanoserve._nanoserve import Engine as _CEngine

__all__ = ["Engine", "GenerateResult", "TokenStream", "set_num_threads", "__version__"]


class Engine:
    """Owns the model weights and the KV cache.

    One generation at a time: each generate()/stream resets the cache, so
    consume a stream before starting the next generation.
    """

    def __init__(self, model_dir, *, max_seq=2048, int8=False):
        self._engine = _CEngine(str(model_dir), max_seq, int8)

    @property
    def quantized(self):
        """True when running int8 weights (Engine(..., int8=True))."""
        return self._engine.quantized

    def generate(
        self,
        prompt,
        *,
        max_tokens=32,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        seed=0,
        stream=False,
    ):
        """Chat-complete `prompt` (rendered through the Qwen chat template).

        temperature=0 (default) decodes greedily; otherwise sampling uses
        temperature -> top-k -> top-p in Hugging Face's order, seeded and
        deterministic. Returns a GenerateResult, or with stream=True a
        TokenStream iterator of text chunks whose .text/.token_ids/.ttft_ms/
        .tokens_per_second are populated as it is consumed.
        """
        args = (prompt, max_tokens, temperature, top_k, top_p, seed)
        if stream:
            return self._engine.stream(*args)
        return self._engine.generate(*args)
