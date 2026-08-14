"""FastAPI application exposing Needle 2 over an OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from . import translate
from .config import Settings
from .engine import (
    EngineError,
    EngineOverloaded,
    NeedleEngine,
    Turn,
)
from .schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    ExtractRequest,
    NeedleCompleteRequest,
    error_body,
    unsupported_parameters,
)
from .streaming import iter_chat_chunks
from .tokens import TokenCounter
from .translate import TranslationError

log = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    engine: NeedleEngine | None = None,
    counter: TokenCounter | None = None,
) -> FastAPI:
    """Build the app. ``engine``/``counter`` may be injected for testing."""
    settings = settings or Settings.from_env()
    owns_engine = engine is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if owns_engine:
            app.state.engine = NeedleEngine(
                weights=settings.weights,
                buffer_size=settings.buffer_size,
                max_queue_depth=settings.max_queue_depth,
                default_max_new_tokens=settings.max_new_tokens,
                max_replay_steps=settings.max_replay_steps,
            )
            log.info("loading needle engine (weights=%s)...", settings.weights or "base")
            await asyncio.to_thread(app.state.engine.start)
        else:
            app.state.engine = engine
        app.state.counter = counter or TokenCounter(exact=settings.exact_token_counts)
        try:
            yield
        finally:
            if owns_engine:
                app.state.engine.shutdown()

    app = FastAPI(
        title="needle-openai",
        version="0.1.0",
        description="OpenAI-compatible API for the Needle 2 tool-calling model.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # -- plumbing ----------------------------------------------------------

    async def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not settings.api_key:
            return
        expected = f"Bearer {settings.api_key}"
        if authorization != expected:
            raise HTTPException(
                status_code=401,
                detail=error_body(
                    "Incorrect API key provided.", "invalid_request_error", code="invalid_api_key"
                ),
            )

    def get_engine(request: Request) -> NeedleEngine:
        eng: NeedleEngine = request.app.state.engine
        if not eng.ready:
            raise HTTPException(
                status_code=503,
                detail=error_body(
                    "The model is still loading; retry shortly.", "server_error", code="not_ready"
                ),
            )
        return eng

    def resolve_model(name: str | None) -> str:
        served = {settings.model_id, "needle-2", "needle2"}
        if name and settings.strict_model and name not in served:
            raise HTTPException(
                status_code=404,
                detail=error_body(
                    f"The model {name!r} does not exist.",
                    "invalid_request_error",
                    param="model",
                    code="model_not_found",
                ),
            )
        return name or settings.model_id

    async def run_engine(
        eng: NeedleEngine,
        system: str,
        tools: list[dict[str, Any]],
        turns: list[Turn],
        budget: int | None,
    ):
        """Submit to the engine thread and await the result with a deadline."""
        started = time.monotonic()
        try:
            future = eng.submit(eng.run_conversation, system, tools, turns, budget)
        except EngineOverloaded as exc:
            raise HTTPException(
                status_code=429,
                detail=error_body(str(exc), "rate_limit_error", code="engine_busy"),
            ) from exc
        except EngineError as exc:
            raise HTTPException(
                status_code=503, detail=error_body(str(exc), "server_error")
            ) from exc

        try:
            result = await asyncio.wait_for(
                asyncio.wrap_future(future), timeout=settings.request_timeout
            )
        except asyncio.TimeoutError as exc:
            future.cancel()
            raise HTTPException(
                status_code=504,
                detail=error_body(
                    f"The request exceeded {settings.request_timeout:g}s waiting for the engine.",
                    "server_error",
                    code="engine_timeout",
                ),
            ) from exc
        except EngineError as exc:
            raise HTTPException(
                status_code=503, detail=error_body(str(exc), "server_error")
            ) from exc
        except TranslationError as exc:
            raise HTTPException(
                status_code=400, detail=error_body(exc.message, param=exc.param, code=exc.code)
            ) from exc

        elapsed = time.monotonic() - started
        result.queue_wait_seconds = max(0.0, elapsed - result.compute_seconds)

        if result.error:
            raise HTTPException(
                status_code=502,
                detail=error_body(
                    f"needle engine error: {result.error}",
                    "server_error",
                    code=str(result.raw.get("error_code") or "engine_error"),
                ),
            )
        return result

    # -- OpenAI surface ----------------------------------------------------

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    async def list_models(request: Request) -> dict[str, Any]:
        return {"object": "list", "data": [_model_card(request, settings)]}

    @app.get("/v1/models/{model_id:path}", dependencies=[Depends(require_auth)])
    async def get_model(model_id: str, request: Request) -> dict[str, Any]:
        resolve_model(model_id)
        card = _model_card(request, settings)
        card["id"] = model_id
        card["root"] = model_id
        return card

    @app.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
    async def chat_completions(request: Request):
        payload = await _json_body(request)
        body = ChatCompletionRequest.model_validate(payload)
        eng = get_engine(request)
        model = resolve_model(body.model)
        warnings = unsupported_parameters(payload)

        try:
            extraction_tool = translate.schema_from_response_format(body.response_format)
            tools = translate.convert_tools(body.tools, body.functions)
            choice = body.tool_choice if body.tool_choice is not None else body.function_call
            tools, _forced = translate.apply_tool_choice(tools, choice)
            system, turns = translate.build_turns(
                [m.model_dump(exclude_none=False) for m in body.messages]
            )
            if extraction_tool is not None:
                if tools:
                    warnings.append(
                        "'tools' ignored: response_format json_schema puts the model in "
                        "extraction mode, where the schema is the only declared tool"
                    )
                tools = [extraction_tool]
            if not turns:
                raise TranslationError(
                    "no user or tool message to act on; needle-2 needs at least one",
                    param="messages",
                )
        except TranslationError as exc:
            return _error_response(400, exc)

        result = await run_engine(eng, system, tools, turns, body.token_budget)
        usage = request.app.state.counter.usage(system, tools, [t.text for t in turns], result.raw)
        extraction_mode = extraction_tool is not None

        if body.stream:
            include_usage = bool((body.stream_options or {}).get("include_usage"))
            chunks = iter_chat_chunks(
                result,
                model,
                extraction_mode=extraction_mode,
                include_extras=settings.expose_needle_extras,
                include_usage=include_usage,
                usage=usage,
                warnings=warnings,
            )
            return StreamingResponse(
                _as_async(chunks),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        return translate.build_chat_completion(
            result,
            model,
            extraction_mode=extraction_mode,
            include_extras=settings.expose_needle_extras,
            usage=usage,
            warnings=warnings,
        )

    @app.post("/v1/completions", dependencies=[Depends(require_auth)])
    async def completions(request: Request):
        payload = await _json_body(request)
        body = CompletionRequest.model_validate(payload)
        eng = get_engine(request)
        model = resolve_model(body.model)

        prompt = body.prompt
        if isinstance(prompt, list):
            if prompt and isinstance(prompt[0], (int, float)):
                return _error_response(
                    400,
                    TranslationError(
                        "token-array prompts are not supported; send text", param="prompt"
                    ),
                )
            prompt = "\n".join(str(p) for p in prompt)
        prompt = str(prompt or "")
        if not prompt:
            return _error_response(
                400, TranslationError("'prompt' must not be empty", param="prompt")
            )

        try:
            tools = translate.convert_tools(body.tools)
        except TranslationError as exc:
            return _error_response(400, exc)

        result = await run_engine(eng, "", tools, [Turn(text=prompt)], body.max_tokens)
        usage = request.app.state.counter.usage("", tools, [prompt], result.raw)
        return translate.build_text_completion(
            result, model, include_extras=settings.expose_needle_extras, usage=usage
        )

    # -- Needle-native surface --------------------------------------------

    @app.post("/v1/needle/extract", dependencies=[Depends(require_auth)])
    async def extract(request: Request):
        payload = await _json_body(request)
        body = ExtractRequest.model_validate(payload)
        eng = get_engine(request)
        tool: dict[str, Any] = {
            "name": body.name or body.schema_.get("title") or "extract",
            "parameters": body.schema_,
        }
        if body.description:
            tool["description"] = body.description
        result = await run_engine(eng, "", [tool], [Turn(text=body.text)], body.max_tokens)
        calls = result.function_calls
        return {
            "object": "needle.extraction",
            "data": (calls[0].get("arguments") if calls else None),
            "matched": bool(calls),
            "x_needle": translate.needle_extras(result.raw),
        }

    @app.post("/v1/needle/complete", dependencies=[Depends(require_auth)])
    async def needle_complete(request: Request):
        """Passthrough returning the engine's response verbatim."""
        payload = await _json_body(request)
        body = NeedleCompleteRequest.model_validate(payload)
        eng = get_engine(request)

        raw_tools = body.tools
        if isinstance(raw_tools, str):
            try:
                raw_tools = json.loads(raw_tools)
            except ValueError as exc:
                return _error_response(
                    400, TranslationError(f"'tools' is not valid JSON: {exc}", param="tools")
                )
        tools = list(raw_tools or [])

        try:
            if body.messages:
                system, turns = translate.build_turns(
                    [m.model_dump(exclude_none=False) for m in body.messages]
                )
            else:
                system, turns = (body.system or ""), [Turn(text=body.query)]
            if not turns:
                raise TranslationError("nothing to send to the model", param="messages")
        except TranslationError as exc:
            return _error_response(400, exc)

        result = await run_engine(eng, system, tools, turns, body.max_tokens)
        return result.raw

    # -- ops ---------------------------------------------------------------

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        eng: NeedleEngine = request.app.state.engine
        return {
            "status": "ok" if eng.ready else "loading",
            "model": settings.model_id,
            "weights": eng.model_name,
            "queue_depth": eng.queue_depth,
            "max_queue_depth": settings.max_queue_depth,
            "exact_token_counts": request.app.state.counter.exact,
        }

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "name": "needle-openai",
            "model": settings.model_id,
            "endpoints": [
                "/v1/models",
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/needle/extract",
                "/v1/needle/complete",
                "/health",
            ],
        }

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        detail = exc.detail
        if not (isinstance(detail, dict) and "error" in detail):
            detail = error_body(
                str(detail), "invalid_request_error" if exc.status_code < 500 else "server_error"
            )
        return JSONResponse(status_code=exc.status_code, content=detail, headers=exc.headers)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception):
        log.exception("unhandled error serving %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content=error_body(f"{exc.__class__.__name__}: {exc}", "server_error"),
        )

    return app


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _model_card(request: Request, settings: Settings) -> dict[str, Any]:
    eng: NeedleEngine = request.app.state.engine
    return {
        "id": settings.model_id,
        "object": "model",
        "created": 0,
        "owned_by": "cactus-compute",
        "root": settings.model_id,
        "parent": None,
        "permission": [],
        "x_needle": {
            "weights": eng.model_name,
            "parameters": "45M",
            "capabilities": ["tool_calling", "structured_extraction"],
            "supports_free_form_text": False,
            "deterministic": True,
        },
    }


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=error_body(f"Invalid JSON body: {exc}")
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail=error_body("Request body must be an object"))
    return payload


def _error_response(status: int, exc: TranslationError) -> JSONResponse:
    return JSONResponse(
        status_code=status, content=error_body(exc.message, param=exc.param, code=exc.code)
    )


async def _as_async(iterator):
    for item in iterator:
        yield item


app = None  # populated by __main__ / uvicorn factory


def get_app() -> FastAPI:
    """Factory for ``uvicorn needle_openai.server:get_app --factory``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    return create_app()
