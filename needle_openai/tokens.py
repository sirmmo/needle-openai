"""Token accounting for the ``usage`` block.

The native engine reports throughput (``prefill_tps`` / ``decode_tps``) but not
token counts, so usage is reconstructed here. With ``sentencepiece`` installed
the model's own tokenizer gives exact counts for the text we control; otherwise
a character heuristic is used and the response says so.

Either way the count covers the text this server sends and receives, not the
engine's internal chat-template framing (``<|im_start|>``, ``<tools>`` and
friends), so treat it as a close lower bound rather than a billing figure.
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4.0


class TokenCounter:
    """Counts tokens exactly when possible, heuristically otherwise."""

    def __init__(self, exact: bool = True) -> None:
        self._tokenizer: Any = None
        self.exact = False
        if exact:
            self._try_load()

    def _try_load(self) -> None:
        try:
            import sentencepiece  # noqa: F401  (probe before the HF download)
            from needle.model.tokenizer import get_tokenizer

            self._tokenizer = get_tokenizer()
            self.exact = True
            log.info("token counting: exact (needle sentencepiece tokenizer)")
        except Exception as exc:
            log.info("token counting: estimated (%s)", exc.__class__.__name__)

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass
        return max(1, int(len(text) / _CHARS_PER_TOKEN + 0.5))

    def usage(
        self,
        system: str,
        tools: list[dict[str, Any]],
        turn_texts: list[str],
        raw: dict[str, Any],
    ) -> dict[str, Any]:
        prompt_text = "\n".join(
            [system, json.dumps(tools, separators=(",", ":")) if tools else "", *turn_texts]
        )
        completion_parts = [str(raw.get("reasoning") or "")]
        if raw.get("function_calls"):
            completion_parts.append(json.dumps(raw["function_calls"], separators=(",", ":")))
        prompt_tokens = self.count(prompt_text)
        completion_tokens = self.count("\n".join(p for p in completion_parts if p))
        usage: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if not self.exact:
            usage["estimated"] = True
        return usage
