"""Runtime configuration, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


@dataclass
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000

    #: Optional tuned ``.cact`` archive. ``None`` serves the base needle-2.
    weights: str | None = None

    #: Model id advertised by ``GET /v1/models``.
    model_id: str = "needle-2"

    #: Reject requests naming a model we do not serve. Off by default because
    #: many OpenAI clients hard-code names like "gpt-4o-mini".
    strict_model: bool = False

    #: When set, require ``Authorization: Bearer <key>``.
    api_key: str | None = None

    #: Output buffer handed to ``needle_complete``. The native side truncates
    #: silently, which surfaces as a JSON decode error, so keep this generous.
    buffer_size: int = 1 << 20

    #: Default cap on generated tokens when the request omits ``max_tokens``.
    max_new_tokens: int = 256

    #: Reject a request outright once this many are already waiting for the
    #: engine. Every request serializes through one thread, so an unbounded
    #: queue just converts load into timeouts.
    max_queue_depth: int = 32

    #: Seconds a single request may spend queued + running before giving up.
    request_timeout: float = 120.0

    #: Number of ``complete()`` calls one request may trigger during history
    #: replay. Guards against a client posting a thousand-message transcript.
    max_replay_steps: int = 32

    #: Include Needle-native signals (confidence, reasoning, timings) as an
    #: ``x_needle`` object on responses.
    expose_needle_extras: bool = True

    #: Use sentencepiece for exact prompt-token counts when available.
    exact_token_counts: bool = True

    allowed_origins: list[str] = field(default_factory=lambda: ["*"])

    @classmethod
    def from_env(cls) -> Settings:
        origins = os.environ.get("NEEDLE_ALLOWED_ORIGINS", "*")
        return cls(
            host=os.environ.get("NEEDLE_HOST", "0.0.0.0"),
            port=_env_int("NEEDLE_PORT", 8000),
            weights=os.environ.get("NEEDLE_WEIGHTS") or None,
            model_id=os.environ.get("NEEDLE_MODEL_ID", "needle-2"),
            strict_model=_env_bool("NEEDLE_STRICT_MODEL", False),
            api_key=os.environ.get("NEEDLE_API_KEY") or None,
            buffer_size=_env_int("NEEDLE_BUFFER_SIZE", 1 << 20),
            max_new_tokens=_env_int("NEEDLE_MAX_NEW_TOKENS", 256),
            max_queue_depth=_env_int("NEEDLE_MAX_QUEUE_DEPTH", 32),
            request_timeout=_env_float("NEEDLE_REQUEST_TIMEOUT", 120.0),
            max_replay_steps=_env_int("NEEDLE_MAX_REPLAY_STEPS", 32),
            expose_needle_extras=_env_bool("NEEDLE_EXPOSE_EXTRAS", True),
            exact_token_counts=_env_bool("NEEDLE_EXACT_TOKEN_COUNTS", True),
            allowed_origins=[o.strip() for o in origins.split(",") if o.strip()],
        )
