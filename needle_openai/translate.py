"""Translation between the OpenAI wire format and Needle's native shapes.

Everything here is a pure function so the mapping can be tested without the
native engine.

Needle's response vocabulary is narrow. The model emits *tool calls or nothing*
-- it has no free-form generation head -- and reports:

    type            "call" | "respond" | "refuse"
    function_calls  [{"name": str, "arguments": {...}}]
    reasoning       a short natural-language trace
    confidence      calibrated 0..1 score for gating
    validation      {"ungrounded": [...], "negation": bool}

The mapping consequences are spelled out in the README under "Fidelity notes".
"""

from __future__ import annotations

import json
import secrets
import time
from typing import Any

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TranslationError(ValueError):
    """The request cannot be expressed in Needle's terms."""

    def __init__(self, message: str, param: str | None = None, code: str | None = None):
        super().__init__(message)
        self.message = message
        self.param = param
        self.code = code or "invalid_request_error"


# ---------------------------------------------------------------------------
# Request -> Needle
# ---------------------------------------------------------------------------


def flatten_content(content: Any) -> str:
    """Reduce OpenAI message content to plain text.

    Content may be a string or a list of parts. Only text parts survive --
    Needle 2 has no image or audio input.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                if part.get("type") in (None, "text", "input_text"):
                    chunks.append(str(part.get("text") or ""))
                else:
                    raise TranslationError(
                        f"content part of type {part.get('type')!r} is not supported; "
                        "needle-2 accepts text only",
                        param="messages",
                    )
        return "\n".join(c for c in chunks if c)
    return str(content)


def convert_tools(
    tools: list[dict[str, Any]] | None,
    functions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert OpenAI tool definitions to Needle's flat schema list.

    OpenAI nests under ``{"type": "function", "function": {...}}``; Needle wants
    ``{"name", "description", "parameters"}`` at the top level. The deprecated
    ``functions`` parameter is already in Needle's shape.
    """
    out: list[dict[str, Any]] = []
    for entry in tools or []:
        if not isinstance(entry, dict):
            raise TranslationError("each tool must be an object", param="tools")
        kind = entry.get("type", "function")
        if kind != "function":
            raise TranslationError(
                f"tool type {kind!r} is not supported; needle-2 supports function tools only",
                param="tools",
            )
        fn = entry.get("function") or {}
        out.append(_tool_schema(fn, param="tools"))
    for fn in functions or []:
        out.append(_tool_schema(fn, param="functions"))
    return out


def _tool_schema(fn: dict[str, Any], param: str) -> dict[str, Any]:
    name = fn.get("name")
    if not name:
        raise TranslationError("tool is missing 'name'", param=param)
    schema: dict[str, Any] = {"name": str(name)}
    if fn.get("description"):
        schema["description"] = str(fn["description"])
    parameters = fn.get("parameters")
    if parameters is None:
        parameters = {"type": "object", "properties": {}}
    if not isinstance(parameters, dict):
        raise TranslationError(f"tool {name!r} has non-object 'parameters'", param=param)
    schema["parameters"] = parameters
    return schema


def apply_tool_choice(
    tools: list[dict[str, Any]], tool_choice: Any
) -> tuple[list[dict[str, Any]], bool]:
    """Narrow the declared tool set according to ``tool_choice``.

    Needle has no decode-time tool forcing, but the choice can be honoured by
    changing *what is declared*:

    ``"none"``      declare nothing, so no call is possible.
    ``"auto"``      declare everything (the default).
    ``"required"``  declare everything; cannot be enforced -- see the returned flag.
    a named tool    declare only that tool.
    """
    if tool_choice in (None, "auto"):
        return tools, False
    if tool_choice == "none":
        return [], False
    if tool_choice == "required":
        return tools, True
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
        if not name:
            raise TranslationError("tool_choice object must name a function", param="tool_choice")
        picked = [t for t in tools if t.get("name") == name]
        if not picked:
            raise TranslationError(
                f"tool_choice names {name!r}, which is not in 'tools'", param="tool_choice"
            )
        return picked, True
    raise TranslationError(f"unsupported tool_choice {tool_choice!r}", param="tool_choice")


def schema_from_response_format(response_format: Any) -> dict[str, Any] | None:
    """Turn ``response_format`` into a Needle tool schema for extraction mode.

    ``{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}``
    becomes a single tool; the model's "call" to it *is* the structured object.
    ``{"type": "text"}`` returns None. ``{"type": "json_object"}`` is rejected:
    Needle is schema-driven and cannot emit free-form JSON.
    """
    if not response_format:
        return None
    if isinstance(response_format, str):
        response_format = {"type": response_format}
    kind = response_format.get("type")
    if kind in (None, "text"):
        return None
    if kind == "json_object":
        raise TranslationError(
            "response_format 'json_object' is not supported; needle-2 needs an explicit "
            'schema -- use {"type": "json_schema", "json_schema": {...}}',
            param="response_format",
        )
    if kind != "json_schema":
        raise TranslationError(
            f"unsupported response_format type {kind!r}", param="response_format"
        )
    spec = response_format.get("json_schema") or {}
    schema = spec.get("schema")
    if not isinstance(schema, dict):
        raise TranslationError(
            "response_format.json_schema.schema must be a JSON Schema object",
            param="response_format",
        )
    tool: dict[str, Any] = {"name": str(spec.get("name") or "extract"), "parameters": schema}
    if spec.get("description"):
        tool["description"] = str(spec["description"])
    return tool


def build_turns(messages: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    """Split messages into a system prompt and the turns to feed the engine.

    Returns ``(system, turns)`` where each turn is fed with one
    ``needle_complete`` call. Assistant messages are *dropped*: the native
    session holds whatever it generated itself and offers no way to inject a
    different assistant turn. Consecutive ``role: tool`` messages collapse into
    one JSON array, mirroring ``needle.Needle.run``.
    """
    from .engine import Turn

    system_parts: list[str] = []
    turns: list[Turn] = []
    pending_results: list[Any] = []

    def flush_results() -> None:
        if pending_results:
            turns.append(
                Turn(
                    text=json.dumps(pending_results, default=str, separators=(",", ":")),
                    kind="tool_results",
                )
            )
            pending_results.clear()

    for index, message in enumerate(messages or []):
        if not isinstance(message, dict):
            raise TranslationError("each message must be an object", param="messages")
        role = message.get("role")
        if role in ("system", "developer"):
            flush_results()
            text = flatten_content(message.get("content"))
            if text:
                system_parts.append(text)
        elif role == "user":
            flush_results()
            text = flatten_content(message.get("content"))
            turns.append(Turn(text=text, kind="user"))
        elif role in ("tool", "function"):
            pending_results.append(_tool_payload(message))
        elif role == "assistant":
            # Intentionally not fed; see the docstring.
            flush_results()
        else:
            raise TranslationError(
                f"messages[{index}] has unsupported role {role!r}", param="messages"
            )

    flush_results()
    return "\n\n".join(system_parts), turns


def _tool_payload(message: dict[str, Any]) -> Any:
    """Decode a tool result, preferring structured JSON over a bare string."""
    text = flatten_content(message.get("content"))
    try:
        return json.loads(text) if text else None
    except (TypeError, ValueError):
        return text


# ---------------------------------------------------------------------------
# Needle -> Response
# ---------------------------------------------------------------------------


def new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(12)}"


def build_tool_calls(function_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render Needle's calls as OpenAI tool calls (arguments JSON-encoded)."""
    calls = []
    for call in function_calls:
        arguments = call.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {})
        calls.append(
            {
                "id": f"call_{secrets.token_hex(12)}",
                "type": "function",
                "function": {"name": str(call.get("name") or ""), "arguments": arguments},
            }
        )
    return calls


def needle_extras(raw: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Collect Needle-native signals that have no OpenAI equivalent."""
    extras = {
        key: raw[key]
        for key in (
            "type",
            "confidence",
            "reasoning",
            "validation",
            "prefill_tps",
            "decode_tps",
            "peak_ram_mb",
            "reason",
        )
        if raw.get(key) is not None
    }
    extras.update({k: v for k, v in overrides.items() if v is not None})
    return extras


def build_chat_completion(
    result: Any,
    model: str,
    *,
    extraction_mode: bool = False,
    include_extras: bool = True,
    usage: dict[str, int] | None = None,
    warnings: list[str] | None = None,
    created: int | None = None,
) -> dict[str, Any]:
    """Assemble a ``chat.completion`` object from an :class:`EngineResult`."""
    raw = result.raw
    calls = result.function_calls
    reasoning = result.reasoning

    message: dict[str, Any] = {"role": "assistant", "content": None, "refusal": None}

    if extraction_mode:
        # The call's arguments *are* the requested object.
        if calls:
            message["content"] = json.dumps(calls[0].get("arguments") or {})
            finish_reason = "stop"
        else:
            message["content"] = None
            message["refusal"] = reasoning or "the model did not produce a matching object"
            finish_reason = "content_filter" if raw.get("type") == "refuse" else "stop"
    elif calls:
        message["tool_calls"] = build_tool_calls(calls)
        finish_reason = "tool_calls"
    else:
        # No call and no free-form head: the reasoning trace is the only text
        # the model produced, so it becomes the assistant's content.
        message["content"] = reasoning
        finish_reason = "stop"

    if reasoning:
        # Convention shared with reasoning models (DeepSeek/vLLM); additive.
        message["reasoning_content"] = reasoning

    body: dict[str, Any] = {
        "id": new_id("chatcmpl"),
        "object": "chat.completion",
        "created": created if created is not None else int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": message, "logprobs": None, "finish_reason": finish_reason}
        ],
        "usage": usage or zero_usage(),
    }
    if include_extras:
        body["x_needle"] = needle_extras(
            raw,
            replayed_turns=result.replayed_turns,
            reinitialized=result.reinitialized,
            queue_wait_seconds=round(result.queue_wait_seconds, 4),
            compute_seconds=round(result.compute_seconds, 4),
            warnings=warnings or None,
        )
    return body


def build_text_completion(
    result: Any,
    model: str,
    *,
    include_extras: bool = True,
    usage: dict[str, int] | None = None,
    created: int | None = None,
) -> dict[str, Any]:
    """Assemble a legacy ``text_completion`` object."""
    raw = result.raw
    calls = result.function_calls
    text = json.dumps(calls) if calls else result.reasoning
    body: dict[str, Any] = {
        "id": new_id("cmpl"),
        "object": "text_completion",
        "created": created if created is not None else int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": "stop"}],
        "usage": usage or zero_usage(),
    }
    if include_extras:
        body["x_needle"] = needle_extras(raw)
    return body


def zero_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
