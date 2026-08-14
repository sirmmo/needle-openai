"""Server-sent-event rendering of a completed response.

Needle's FFI returns one finished JSON object -- ``needle_complete`` has no
token callback -- so there is no genuine token stream to forward. When a client
asks for ``stream: true`` the finished result is cut into the chunk sequence an
OpenAI client expects. Time-to-first-chunk therefore equals full generation
time; at ~150 tok/s on a 45M model that is a fraction of a second.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

from .translate import build_tool_calls, needle_extras, new_id, zero_usage

#: Characters per synthetic content chunk.
_CHUNK_SIZE = 24


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def iter_chat_chunks(
    result: Any,
    model: str,
    *,
    extraction_mode: bool = False,
    include_extras: bool = True,
    include_usage: bool = False,
    usage: dict[str, int] | None = None,
    warnings: list[str] | None = None,
) -> Iterator[str]:
    """Yield the SSE lines for one ``chat.completion.chunk`` stream."""
    raw = result.raw
    calls = result.function_calls
    reasoning = result.reasoning
    completion_id = new_id("chatcmpl")
    created = int(time.time())

    def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
        return _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}
                ],
            }
        )

    yield chunk({"role": "assistant", "content": None})

    if extraction_mode:
        if calls:
            text = json.dumps(calls[0].get("arguments") or {})
            for piece in _split(text):
                yield chunk({"content": piece})
            finish_reason = "stop"
        else:
            yield chunk({"refusal": reasoning or "the model did not produce a matching object"})
            finish_reason = "content_filter" if raw.get("type") == "refuse" else "stop"
    elif calls:
        for index, call in enumerate(build_tool_calls(calls)):
            # Name and id arrive first, then the arguments, as real streams do.
            yield chunk(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["function"]["name"], "arguments": ""},
                        }
                    ]
                }
            )
            for piece in _split(call["function"]["arguments"]):
                yield chunk({"tool_calls": [{"index": index, "function": {"arguments": piece}}]})
        finish_reason = "tool_calls"
    else:
        for piece in _split(reasoning):
            yield chunk({"content": piece})
        finish_reason = "stop"

    final: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": finish_reason}],
    }
    if include_extras:
        final["x_needle"] = needle_extras(
            raw,
            replayed_turns=result.replayed_turns,
            queue_wait_seconds=round(result.queue_wait_seconds, 4),
            compute_seconds=round(result.compute_seconds, 4),
            warnings=warnings or None,
        )
    yield _sse(final)

    if include_usage:
        yield _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage or zero_usage(),
            }
        )

    yield "data: [DONE]\n\n"


def _split(text: str, size: int = _CHUNK_SIZE) -> Iterator[str]:
    for start in range(0, len(text or ""), size):
        yield text[start : start + size]
