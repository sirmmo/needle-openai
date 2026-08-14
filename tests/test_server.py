"""HTTP-level tests, driven through a fake engine."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from fakes import FakeEngine, needle_response
from needle_openai.config import Settings
from needle_openai.server import create_app
from needle_openai.tokens import TokenCounter

WEATHER_TOOL = {
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


def chat(client, **overrides):
    payload = {
        "model": "needle-2",
        "messages": [{"role": "user", "content": "weather in Lagos?"}],
        "tools": [WEATHER_TOOL],
    }
    payload.update(overrides)
    return client.post("/v1/chat/completions", json=payload)


# -- discovery -------------------------------------------------------------


def test_list_models(client):
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    card = body["data"][0]
    assert card["id"] == "needle-2"
    assert card["x_needle"]["supports_free_form_text"] is False
    assert card["x_needle"]["deterministic"] is True


def test_health_reports_engine_state(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["queue_depth"] == 0


def test_root_lists_endpoints(client):
    assert "/v1/chat/completions" in client.get("/").json()["endpoints"]


# -- chat completions ------------------------------------------------------


def test_tool_call_round_trip(client):
    body = chat(client).json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Lagos"}
    # The engine saw the flattened Needle schema, not OpenAI's nesting.
    assert client.engine.calls[0]["tools"] == [
        {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]


def test_system_prompt_is_forwarded_separately(client):
    chat(
        client,
        messages=[
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "weather in Lagos?"},
        ],
    )
    call = client.engine.calls[0]
    assert call["system"] == "Be terse."
    assert call["turns"] == [("user", "weather in Lagos?")]


def test_tool_result_history_is_replayed_as_turns(client):
    chat(
        client,
        messages=[
            {"role": "user", "content": "weather in Lagos?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Lagos"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 27}'},
        ],
    )
    kinds = [kind for kind, _ in client.engine.calls[0]["turns"]]
    assert kinds == ["user", "tool_results"]


def test_max_tokens_is_forwarded(client):
    chat(client, max_tokens=64)
    assert client.engine.calls[0]["max_new_tokens"] == 64
    chat(client, max_completion_tokens=32)
    assert client.engine.calls[1]["max_new_tokens"] == 32


def test_tool_choice_none_declares_no_tools(client):
    chat(client, tool_choice="none")
    assert client.engine.calls[0]["tools"] == []


def test_tool_choice_named_narrows_the_declared_tools(client):
    chat(
        client,
        tools=[WEATHER_TOOL, {"type": "function", "function": {"name": "send_email"}}],
        tool_choice={"type": "function", "function": {"name": "send_email"}},
    )
    assert [t["name"] for t in client.engine.calls[0]["tools"]] == ["send_email"]


def test_ignored_sampling_params_are_reported_not_rejected(client):
    body = chat(client, temperature=0.7, top_p=0.9, n=3).json()
    warnings = body["x_needle"]["warnings"]
    assert any("temperature" in w for w in warnings)
    assert any("'n'=3" in w for w in warnings)
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_usage_is_populated_and_flagged_as_estimated(client):
    usage = chat(client).json()["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["estimated"] is True


def test_no_tool_call_returns_reasoning_as_content(settings):
    engine = FakeEngine(
        needle_response(type_="respond", function_calls=[], reasoning="No tool available.")
    )
    app = create_app(settings=settings, engine=engine, counter=TokenCounter(exact=False))
    with TestClient(app) as client:
        choice = chat(client).json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "No tool available."


# -- structured output ---------------------------------------------------


def test_response_format_json_schema_enters_extraction_mode(settings):
    engine = FakeEngine(
        needle_response(
            function_calls=[{"name": "Invoice", "arguments": {"vendor": "Acme", "total": 1200.0}}]
        )
    )
    app = create_app(settings=settings, engine=engine, counter=TokenCounter(exact=False))
    schema = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}, "total": {"type": "number"}},
        "required": ["vendor", "total"],
    }
    with TestClient(app) as client:
        body = client.post(
            "/v1/chat/completions",
            json={
                "model": "needle-2",
                "messages": [{"role": "user", "content": "Invoice from Acme Corp, $1,200.00"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "Invoice", "schema": schema},
                },
            },
        ).json()
    # The schema became the only declared tool...
    assert engine.calls[0]["tools"] == [{"name": "Invoice", "parameters": schema}]
    # ...and its arguments came back as JSON content, not a tool call.
    assert json.loads(body["choices"][0]["message"]["content"]) == {
        "vendor": "Acme",
        "total": 1200.0,
    }
    assert body["choices"][0]["finish_reason"] == "stop"


def test_json_object_response_format_returns_a_clear_error(client):
    response = chat(client, response_format={"type": "json_object"})
    assert response.status_code == 400
    assert "explicit schema" in response.json()["error"]["message"]
    assert response.json()["error"]["param"] == "response_format"


def test_needle_extract_endpoint(settings):
    engine = FakeEngine(
        needle_response(function_calls=[{"name": "Invoice", "arguments": {"vendor": "Acme"}}])
    )
    app = create_app(settings=settings, engine=engine, counter=TokenCounter(exact=False))
    with TestClient(app) as client:
        body = client.post(
            "/v1/needle/extract",
            json={
                "text": "Invoice from Acme Corp",
                "schema": {"type": "object", "properties": {"vendor": {"type": "string"}}},
                "name": "Invoice",
            },
        ).json()
    assert body["matched"] is True
    assert body["data"] == {"vendor": "Acme"}
    assert body["x_needle"]["confidence"] == 0.87


def test_needle_complete_passthrough_returns_raw_response(client):
    body = client.post(
        "/v1/needle/complete",
        json={"query": "weather in Lagos?", "tools": [{"name": "get_weather", "parameters": {}}]},
    ).json()
    # Verbatim native shape, including fields OpenAI has no room for.
    assert body["type"] == "call"
    assert body["confidence"] == 0.87
    assert "peak_ram_mb" in body


def test_needle_complete_accepts_tools_as_a_json_string(client):
    response = client.post(
        "/v1/needle/complete",
        json={"query": "hi", "tools": '[{"name": "ping", "parameters": {}}]'},
    )
    assert response.status_code == 200
    assert client.engine.calls[0]["tools"] == [{"name": "ping", "parameters": {}}]


# -- streaming -----------------------------------------------------------


def _events(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            out.append(json.loads(line[len("data: ") :]))
    return out


def test_streaming_emits_a_valid_chunk_sequence(client):
    response = chat(client, stream=True)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.rstrip().endswith("data: [DONE]")

    events = _events(response.text)
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert all(e["object"] == "chat.completion.chunk" for e in events)
    # One id for the whole stream.
    assert len({e["id"] for e in events}) == 1

    # Tool call name arrives before its arguments, which stream in fragments.
    tool_deltas = [
        d for e in events for d in [e["choices"][0]["delta"]] if e["choices"] and "tool_calls" in d
    ]
    assert tool_deltas[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    joined = "".join(d["tool_calls"][0]["function"].get("arguments", "") for d in tool_deltas)
    assert json.loads(joined) == {"city": "Lagos"}
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_streaming_include_usage_appends_a_usage_chunk(client):
    response = chat(client, stream=True, stream_options={"include_usage": True})
    events = _events(response.text)
    assert events[-1]["choices"] == []
    assert events[-1]["usage"]["total_tokens"] > 0


def test_streaming_content_reassembles_the_reasoning(settings):
    long_reason = "No tool is available for this request, so nothing was called at all."
    engine = FakeEngine(needle_response(type_="respond", function_calls=[], reasoning=long_reason))
    app = create_app(settings=settings, engine=engine, counter=TokenCounter(exact=False))
    with TestClient(app) as client:
        events = _events(chat(client, stream=True).text)
    text = "".join(
        e["choices"][0]["delta"].get("content") or ""
        for e in events
        if e["choices"] and "content" in e["choices"][0]["delta"]
    )
    assert text == long_reason


# -- legacy completions --------------------------------------------------


def test_legacy_completions_endpoint(client):
    body = client.post(
        "/v1/completions", json={"model": "needle-2", "prompt": "weather in Lagos?"}
    ).json()
    assert body["object"] == "text_completion"
    assert json.loads(body["choices"][0]["text"])[0]["name"] == "get_weather"


def test_legacy_completions_rejects_token_arrays(client):
    response = client.post("/v1/completions", json={"prompt": [1, 2, 3]})
    assert response.status_code == 400
    assert "token-array" in response.json()["error"]["message"]


# -- errors and limits ---------------------------------------------------


def test_empty_messages_is_a_clear_400(client):
    response = chat(client, messages=[])
    assert response.status_code == 400
    assert "at least one" in response.json()["error"]["message"]


def test_system_only_conversation_is_a_400(client):
    response = chat(client, messages=[{"role": "system", "content": "Be terse."}])
    assert response.status_code == 400


def test_invalid_json_body_is_a_400(client):
    response = client.post(
        "/v1/chat/completions",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_engine_overload_maps_to_429(client):
    client.engine.overloaded = True
    response = chat(client)
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "engine_busy"


def test_engine_error_response_maps_to_502(settings):
    engine = FakeEngine(needle_response(error="decode failed", error_code="E_DECODE"))
    app = create_app(settings=settings, engine=engine, counter=TokenCounter(exact=False))
    with TestClient(app) as client:
        response = chat(client)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "E_DECODE"


def test_engine_not_ready_maps_to_503(settings, engine):
    engine.ready = False
    app = create_app(settings=settings, engine=engine, counter=TokenCounter(exact=False))
    with TestClient(app) as client:
        response = chat(client)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "not_ready"


def test_unsupported_tool_type_is_a_400(client):
    response = chat(client, tools=[{"type": "file_search"}])
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "tools"


# -- auth and model gating ------------------------------------------------


def test_api_key_is_enforced_when_configured(engine):
    app = create_app(
        settings=Settings(api_key="secret", exact_token_counts=False),
        engine=engine,
        counter=TokenCounter(exact=False),
    )
    with TestClient(app) as client:
        assert client.get("/v1/models").status_code == 401
        assert (
            client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
        )
        ok = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
        assert ok.status_code == 200
        # /health stays open so orchestrators can probe it.
        assert client.get("/health").status_code == 200


def test_unknown_model_is_accepted_by_default(client):
    assert chat(client, model="gpt-4o-mini").status_code == 200


def test_strict_model_rejects_unknown_names(engine):
    app = create_app(
        settings=Settings(strict_model=True, exact_token_counts=False),
        engine=engine,
        counter=TokenCounter(exact=False),
    )
    with TestClient(app) as client:
        response = chat(client, model="gpt-4o-mini")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "model_not_found"
        assert chat(client, model="needle-2").status_code == 200
