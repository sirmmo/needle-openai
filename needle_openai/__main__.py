"""Command-line entry point: ``needle-openai`` / ``python -m needle_openai``."""

from __future__ import annotations

import argparse
import logging

from .config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="needle-openai",
        description="Serve the Needle 2 model behind an OpenAI-compatible API.",
    )
    parser.add_argument("--host", default=None, help="bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default 8000)")
    parser.add_argument(
        "--weights", default=None, help="tuned .cact archive to serve instead of base needle-2"
    )
    parser.add_argument("--model-id", default=None, help="model id to advertise")
    parser.add_argument(
        "--api-key", default=None, help="require this bearer token on every request"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None, help="default generation cap (default 256)"
    )
    parser.add_argument(
        "--max-queue-depth",
        type=int,
        default=None,
        help="reject requests once this many are queued (default 32)",
    )
    parser.add_argument(
        "--request-timeout", type=float, default=None, help="per-request deadline in seconds"
    )
    parser.add_argument(
        "--strict-model",
        action="store_true",
        help="404 on requests naming a model other than the one served",
    )
    parser.add_argument(
        "--no-extras",
        action="store_true",
        help="omit the x_needle block (confidence, reasoning, timings) from responses",
    )
    parser.add_argument("--log-level", default="info")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = Settings.from_env()
    for attr, value in (
        ("host", args.host),
        ("port", args.port),
        ("weights", args.weights),
        ("model_id", args.model_id),
        ("api_key", args.api_key),
        ("max_new_tokens", args.max_new_tokens),
        ("max_queue_depth", args.max_queue_depth),
        ("request_timeout", args.request_timeout),
    ):
        if value is not None:
            setattr(settings, attr, value)
    if args.strict_model:
        settings.strict_model = True
    if args.no_extras:
        settings.expose_needle_extras = False

    import uvicorn

    from .server import create_app

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=args.log_level,
        # One worker only: the native engine is a single global session and
        # multiple processes would each load their own copy of the weights.
        workers=1,
    )


if __name__ == "__main__":
    main()
