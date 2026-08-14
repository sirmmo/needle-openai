"""Request models for the OpenAI-compatible surface.

Models are deliberately permissive (``extra="allow"``): OpenAI clients send a
long tail of parameters, and rejecting unknown fields would break them for no
benefit. Parameters Needle cannot honour are collected by
:func:`unsupported_parameters` and reported back as warnings instead of errors.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

#: Accepted, then ignored. Needle decodes greedily with grammar constraints --
#: the FFI exposes no temperature, top-p or seed, and output is deterministic.
IGNORED_SAMPLING_PARAMS = (
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "stop",
    "min_p",
    "repetition_penalty",
)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    functions: list[dict[str, Any]] | None = None
    function_call: Any = None
    response_format: Any = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    n: int | None = None
    user: str | None = None

    @property
    def token_budget(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens


class CompletionRequest(BaseModel):
    """Legacy ``/v1/completions``."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    prompt: Any = ""
    tools: list[dict[str, Any]] | None = None
    max_tokens: int | None = None
    stream: bool = False
    suffix: str | None = None
    user: str | None = None


class ExtractRequest(BaseModel):
    """Needle-native structured extraction (not part of the OpenAI spec)."""

    model_config = ConfigDict(extra="allow")

    text: str
    schema_: dict[str, Any] = Field(alias="schema")
    name: str | None = None
    description: str | None = None
    max_tokens: int | None = None


class NeedleCompleteRequest(BaseModel):
    """Raw passthrough to ``needle_complete``, returning the native response."""

    model_config = ConfigDict(extra="allow")

    query: str = ""
    tools: Any = None
    system: str | None = None
    messages: list[ChatMessage] | None = None
    max_tokens: int | None = None


def unsupported_parameters(payload: dict[str, Any]) -> list[str]:
    """Name the request fields that were accepted but had no effect."""
    warnings: list[str] = []
    for key in IGNORED_SAMPLING_PARAMS:
        value = payload.get(key)
        if value in (None, False, [], {}, 0):
            continue
        warnings.append(
            f"{key!r} ignored: needle-2 decodes deterministically and exposes no sampling controls"
        )
    n = payload.get("n")
    if isinstance(n, int) and n > 1:
        warnings.append(f"'n'={n} ignored: only one deterministic choice is produced")
    if payload.get("tool_choice") == "required":
        warnings.append(
            "tool_choice 'required' cannot be enforced: needle-2 decides whether to call a tool"
        )
    return warnings


def error_body(
    message: str,
    kind: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    """An OpenAI-shaped error envelope."""
    return {"error": {"message": message, "type": kind, "param": param, "code": code}}


ObjectType = Literal["model", "list"]
