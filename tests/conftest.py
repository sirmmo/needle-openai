"""Shared fixtures: a fake engine so the HTTP surface is testable offline."""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any

import pytest
from fastapi.testclient import TestClient

from needle_openai.config import Settings
from needle_openai.engine import EngineOverloaded, EngineResult
from needle_openai.server import create_app
from needle_openai.tokens import TokenCounter


def needle_response(
    type_: str = "call",
    function_calls: list[dict[str, Any]] | None = None,
    reasoning: str = "because the user asked",
    confidence: float = 0.87,
    **extra: Any,
) -> dict[str, Any]:
    """A response shaped exactly like the ones the native engine returns."""
    body = {
        "type": type_,
        "success": True,
        "error": None,
        "error_code": None,
        "reason": None,
        "function_calls": function_calls if function_calls is not None else [],
        "reasoning": reasoning,
        "confidence": confidence,
        "prefill_tps": 240.1,
        "decode_tps": 157.7,
        "peak_ram_mb": 34.2,
    }
    body.update(extra)
    return body


class FakeEngine:
    """Stands in for :class:`NeedleEngine`, recording what it was asked to do."""

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.ready = True
        self.model_name = "needle-2 (base)"
        self.queue_depth = 0
        self.response = response or needle_response(
            function_calls=[{"name": "get_weather", "arguments": {"city": "Lagos"}}]
        )
        self.calls: list[dict[str, Any]] = []
        self.overloaded = False
        self.raise_on_run: Exception | None = None

    def submit(self, fn, *args, **kwargs) -> Future:
        if self.overloaded:
            raise EngineOverloaded("engine queue is full (32 requests waiting); retry shortly")
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # surfaced to the caller via the future
            future.set_exception(exc)
        return future

    def run_conversation(self, system, tools, turns, max_new_tokens=None) -> EngineResult:
        self.calls.append(
            {
                "system": system,
                "tools": tools,
                "turns": [(t.kind, t.text) for t in turns],
                "max_new_tokens": max_new_tokens,
            }
        )
        if self.raise_on_run is not None:
            raise self.raise_on_run
        return EngineResult(
            raw=dict(self.response), replayed_turns=len(turns), compute_seconds=0.01
        )

    def shutdown(self) -> None:
        pass


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def settings() -> Settings:
    # Deterministic settings, independent of the caller's environment.
    return Settings(api_key=None, expose_needle_extras=True, exact_token_counts=False)


@pytest.fixture
def client(settings: Settings, engine: FakeEngine):
    app = create_app(settings=settings, engine=engine, counter=TokenCounter(exact=False))
    with TestClient(app) as test_client:
        test_client.engine = engine  # type: ignore[attr-defined]
        yield test_client
