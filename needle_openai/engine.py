"""Serialized access to the Needle native engine.

``needle.Needle`` is a thin ctypes shim over ``libneedle.so``, and the native
side keeps **one** global session: ``needle_init`` re-binds it, ``needle_reset``
clears its history, and the ``needle`` module tracks the current owner in module
globals. Calling it from two threads at once segfaults the process (verified),
so every request is funnelled through a single dedicated worker thread.

That thread also owns the initial library load and weight upload, keeping all
native state confined to one thread for its whole lifetime.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


class EngineError(RuntimeError):
    """The engine could not serve a request."""


class EngineOverloaded(EngineError):
    """Too many requests are already waiting for the engine."""


@dataclass
class Turn:
    """One ``needle_complete`` call's worth of input.

    ``text`` is fed verbatim to the engine. A user message contributes its
    content; a run of ``role: tool`` messages contributes a JSON array of their
    payloads, matching how ``needle.Needle.run`` feeds tool results back.
    """

    text: str
    kind: str = "user"  # "user" | "tool_results"


@dataclass
class EngineResult:
    """Raw native response plus bookkeeping the HTTP layer needs."""

    raw: dict[str, Any]
    replayed_turns: int = 1
    reinitialized: bool = False
    queue_wait_seconds: float = 0.0
    compute_seconds: float = 0.0

    @property
    def type(self) -> str:
        return str(self.raw.get("type") or "")

    @property
    def function_calls(self) -> list[dict[str, Any]]:
        calls = self.raw.get("function_calls")
        return list(calls) if isinstance(calls, list) else []

    @property
    def reasoning(self) -> str:
        return str(self.raw.get("reasoning") or "")

    @property
    def error(self) -> str | None:
        err = self.raw.get("error")
        return str(err) if err else None


@dataclass
class _Session:
    """The engine configuration currently bound natively."""

    system: str = ""
    tools_json: str = "[]"
    bound: bool = False
    #: Turns already fed since the last reset, so an extended conversation can
    #: skip re-feeding the prefix it already holds.
    fed: list[str] = field(default_factory=list)

    def matches(self, system: str, tools_json: str) -> bool:
        return self.bound and self.system == system and self.tools_json == tools_json


class NeedleEngine:
    """Thread-confined wrapper around the Needle native session."""

    def __init__(
        self,
        weights: str | None = None,
        buffer_size: int = 1 << 20,
        max_queue_depth: int = 32,
        default_max_new_tokens: int = 256,
        max_replay_steps: int = 32,
    ) -> None:
        self.weights = weights
        self.buffer_size = buffer_size
        self.max_queue_depth = max_queue_depth
        self.default_max_new_tokens = default_max_new_tokens
        self.max_replay_steps = max_replay_steps

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="needle-engine")
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._session = _Session()
        self._agent: Any = None
        self._needle: Any = None
        self._ready = False
        self._model_name = "needle-2 (base)"

    # -- lifecycle ---------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def model_name(self) -> str:
        return self._model_name

    def start(self, timeout: float = 600.0) -> None:
        """Load the native library and weights on the worker thread.

        Blocking here means the first HTTP request does not pay the download
        and warm-up cost.
        """
        self._executor.submit(self._load).result(timeout=timeout)

    def _load(self) -> None:
        try:
            import needle  # imported on the worker thread that will own it
        except ImportError as exc:
            # cactus-needle is not a hard dependency (it would drag in jax), so
            # a missing engine is a normal setup mistake worth explaining.
            raise EngineError(
                "the needle engine is not installed. Install it without its "
                "training dependencies:\n"
                "    pip install --no-deps cactus-needle==2.0.3\n"
                "or, accepting the jax install, `pip install 'needle-openai[engine]'`"
            ) from exc

        self._needle = needle
        # Binding an empty tool set forces the library load, weight upload and
        # first `needle_init` to happen now rather than on first request.
        self._agent = needle.Needle(tools="[]", weights=self.weights, buffer_size=self.buffer_size)
        self._session = _Session(system="", tools_json="[]", bound=True, fed=[])
        if self.weights:
            import os

            self._model_name = os.path.basename(self.weights)
        self._ready = True
        log.info("needle engine ready (%s)", self._model_name)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- submission --------------------------------------------------------

    def submit(self, fn, *args, **kwargs) -> Future:
        """Queue work on the engine thread, rejecting past the depth limit."""
        with self._pending_lock:
            if self._pending >= self.max_queue_depth:
                raise EngineOverloaded(
                    f"engine queue is full ({self._pending} requests waiting); retry shortly"
                )
            self._pending += 1

        def runner():
            try:
                return fn(*args, **kwargs)
            finally:
                with self._pending_lock:
                    self._pending -= 1

        try:
            return self._executor.submit(runner)
        except RuntimeError as exc:  # executor already shut down
            with self._pending_lock:
                self._pending -= 1
            raise EngineError("engine is shutting down") from exc

    @property
    def queue_depth(self) -> int:
        with self._pending_lock:
            return self._pending

    # -- the actual work ---------------------------------------------------

    def run_conversation(
        self,
        system: str,
        tools: list[dict[str, Any]],
        turns: list[Turn],
        max_new_tokens: int | None = None,
    ) -> EngineResult:
        """Replay ``turns`` against the engine and return the final response.

        Runs on the engine thread. Because the native session is stateful but
        cannot be *told* what it previously said, replay feeds only the inputs
        (user text and tool results); the engine regenerates its own
        intermediate assistant turns. See README "Fidelity notes".
        """
        import time

        if not self._ready:
            raise EngineError("engine is not ready")
        if not turns:
            raise EngineError("no content to send to the model")
        if len(turns) > self.max_replay_steps:
            raise EngineError(
                f"conversation needs {len(turns)} engine calls, over the "
                f"limit of {self.max_replay_steps}; send a shorter history"
            )

        started = time.monotonic()
        tools_json = json.dumps(tools, separators=(",", ":"), sort_keys=True)
        budget = int(max_new_tokens or self.default_max_new_tokens)
        texts = [t.text for t in turns]

        reinitialized = self._prepare(system, tools_json, texts)
        to_feed = texts[len(self._session.fed) :]

        raw: dict[str, Any] | None = None
        for text in to_feed:
            raw = self._agent.complete(text, max_new_tokens=budget)
            self._session.fed.append(text)
        assert raw is not None  # _prepare guarantees at least one turn remains

        return EngineResult(
            raw=raw,
            replayed_turns=len(to_feed),
            reinitialized=reinitialized,
            compute_seconds=time.monotonic() - started,
        )

    def _prepare(self, system: str, tools_json: str, texts: list[str]) -> bool:
        """Bind the requested config, reusing session state when it is a prefix.

        Returns True when the native session had to be re-initialized.
        """
        reinitialized = False
        if not self._session.matches(system, tools_json):
            self._agent = self._needle.Needle(
                tools=tools_json,
                system=system,
                weights=self.weights,
                buffer_size=self.buffer_size,
            )
            self._session = _Session(system=system, tools_json=tools_json, bound=True, fed=[])
            reinitialized = True

        # A continuing conversation (previous turns unchanged, new ones
        # appended) can keep the session's history and feed only the tail.
        # Anything else starts clean -- reset() restores a virgin session.
        fed = self._session.fed
        continues = len(texts) > len(fed) and texts[: len(fed)] == fed
        if not continues:
            self._needle._lib().needle_reset()
            self._session.fed = []
        return reinitialized

    def extract(
        self, text: str, schema: dict[str, Any], max_new_tokens: int | None = None
    ) -> EngineResult:
        """One-shot structured extraction: declare ``schema`` as the only tool."""
        return self.run_conversation(
            system="", tools=[schema], turns=[Turn(text=text)], max_new_tokens=max_new_tokens
        )
