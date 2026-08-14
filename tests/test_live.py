"""End-to-end tests against a running server with the real Needle engine.

These are skipped unless a server URL is provided, because they download the
engine and weights from HuggingFace:

    docker compose up -d
    NEEDLE_TEST_BASE_URL=http://127.0.0.1:8000 pytest -m live

Run them after any change to engine.py -- the single-session serialization they
exercise is the part unit tests with a fake engine cannot cover.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = pytest.mark.live

BASE_URL = os.environ.get("NEEDLE_TEST_BASE_URL")

if not BASE_URL:
    pytest.skip("set NEEDLE_TEST_BASE_URL to run live tests", allow_module_level=True)

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


@pytest.fixture(scope="module")
def http():
    import httpx

    with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
        yield client


def post_chat(http, **payload):
    body = {"model": "needle-2", **payload}
    response = http.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_health_is_ok(http):
    body = http.get("/health").json()
    assert body["status"] == "ok"


def test_real_tool_call(http):
    body = post_chat(
        http,
        messages=[{"role": "user", "content": "what's it like in Lagos right now?"}],
        tools=[WEATHER],
    )
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Lagos"}
    assert 0.0 <= body["x_needle"]["confidence"] <= 1.0


def test_real_tool_result_round_trip(http):
    first = post_chat(
        http,
        messages=[{"role": "user", "content": "what's it like in Lagos right now?"}],
        tools=[WEATHER],
    )
    call = first["choices"][0]["message"]["tool_calls"][0]
    second = post_chat(
        http,
        messages=[
            {"role": "user", "content": "what's it like in Lagos right now?"},
            {"role": "assistant", "tool_calls": [call]},
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps({"city": "Lagos", "temp_c": 27, "sky": "clear"}),
            },
        ],
        tools=[WEATHER],
    )
    # Having seen the result the model stops calling and answers.
    assert second["choices"][0]["finish_reason"] == "stop"
    assert second["x_needle"]["type"] == "respond"


def test_real_structured_extraction(http):
    body = post_chat(
        http,
        messages=[{"role": "user", "content": "Invoice from Acme Corp, $1,200.00, due 2026-09-01"}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Invoice",
                "schema": {
                    "type": "object",
                    "properties": {
                        "vendor": {"type": "string"},
                        "total": {"type": "number"},
                        "due_date": {"type": "string"},
                    },
                    "required": ["vendor", "total", "due_date"],
                },
            },
        },
    )
    parsed = json.loads(body["choices"][0]["message"]["content"])
    assert parsed == {"vendor": "Acme Corp", "total": 1200.0, "due_date": "2026-09-01"}


def test_real_streaming(http):
    payload = {
        "model": "needle-2",
        "messages": [{"role": "user", "content": "weather in Tokyo?"}],
        "tools": [WEATHER],
        "stream": True,
    }
    arguments, names = "", []
    with http.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[len("data: ") :])
            for choice in event.get("choices", []):
                for tc in choice["delta"].get("tool_calls") or []:
                    if tc.get("function", {}).get("name"):
                        names.append(tc["function"]["name"])
                    arguments += tc.get("function", {}).get("arguments") or ""
    assert names == ["get_weather"]
    assert json.loads(arguments) == {"city": "Tokyo"}


def test_deterministic_across_requests(http):
    def confidence():
        body = post_chat(
            http, messages=[{"role": "user", "content": "weather in Lagos?"}], tools=[WEATHER]
        )
        return body["x_needle"]["confidence"], body["choices"][0]["message"]["tool_calls"][0][
            "function"
        ]["arguments"]

    assert confidence() == confidence()


def test_concurrent_requests_are_serialized_safely(http):
    """The native engine is one global session and crashes under concurrent use.

    Twenty-four simultaneous requests must all come back correct, which only
    holds if every one of them was funnelled through the engine thread.
    """
    cities = ["Lagos", "Tokyo", "Paris", "Berlin", "Lima", "Cairo"]

    def ask(index: int) -> tuple[str, str | None]:
        city = cities[index % len(cities)]
        body = post_chat(
            http, messages=[{"role": "user", "content": f"weather in {city}?"}], tools=[WEATHER]
        )
        calls = body["choices"][0]["message"].get("tool_calls")
        got = json.loads(calls[0]["function"]["arguments"]).get("city") if calls else None
        return city, got

    with ThreadPoolExecutor(max_workers=24) as pool:
        results = list(pool.map(ask, range(24)))

    assert [want for want, got in results if want != got] == []
    assert http.get("/health").json()["status"] == "ok"


def test_needle_extract_endpoint(http):
    response = http.post(
        "/v1/needle/extract",
        json={
            "text": "Invoice from Acme Corp, $1,200.00, due 2026-09-01",
            "name": "Invoice",
            "schema": {
                "type": "object",
                "properties": {
                    "vendor": {"type": "string"},
                    "total": {"type": "number"},
                    "due_date": {"type": "string"},
                },
                "required": ["vendor", "total", "due_date"],
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["matched"] is True
    assert body["data"]["vendor"] == "Acme Corp"
