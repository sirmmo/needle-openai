"""OpenAI-compatible HTTP API for the Needle 2 tool-calling model."""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__", "create_app", "Settings", "NeedleEngine"]


def __getattr__(name: str):
    # Lazy so `import needle_openai` stays cheap and does not pull FastAPI.
    if name == "create_app":
        from .server import create_app

        return create_app
    if name == "Settings":
        from .config import Settings

        return Settings
    if name == "NeedleEngine":
        from .engine import NeedleEngine

        return NeedleEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
