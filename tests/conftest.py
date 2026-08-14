"""Shared fixtures.

Every import of ``needle_openai`` or ``fastapi`` happens *inside* a fixture.
``conftest.py`` is loaded for every session, including the live suite, which runs
against a container with only ``pytest`` and ``httpx`` available -- a module-level
import here would make the server's dependencies mandatory for that suite too.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def engine():
    from fakes import FakeEngine

    return FakeEngine()


@pytest.fixture
def settings():
    from needle_openai.config import Settings

    # Deterministic settings, independent of the caller's environment.
    return Settings(api_key=None, expose_needle_extras=True, exact_token_counts=False)


@pytest.fixture
def client(settings, engine):
    from fastapi.testclient import TestClient

    from needle_openai.server import create_app
    from needle_openai.tokens import TokenCounter

    app = create_app(settings=settings, engine=engine, counter=TokenCounter(exact=False))
    with TestClient(app) as test_client:
        test_client.engine = engine  # type: ignore[attr-defined]
        yield test_client
