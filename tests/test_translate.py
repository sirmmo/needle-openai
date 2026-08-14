"""Unit tests for the OpenAI <-> Needle mapping."""

from __future__ import annotations

import json

import pytest

from conftest import needle_response
from needle_openai import translate
from needle_openai.engine import EngineResult
from needle_openai.translate import TranslationError

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


# -- tools -----------------------------------------------------------------


def test_convert_tools_flattens_openai_nesting():
    assert translate.convert_tools([WEATHER]) == [
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


def test_convert_tools_defaults_missing_parameters():
    tools = translate.convert_tools([{"type": "function", "function": {"name": "ping"}}])
    assert tools[0]["parameters"] == {"type": "object", "properties": {}}


def test_convert_tools_accepts_legacy_functions():
    tools = translate.convert_tools(None, functions=[{"name": "ping", "parameters": {}}])
    assert tools[0]["name"] == "ping"


def test_convert_tools_rejects_non_function_tool():
    with pytest.raises(TranslationError, match="not supported"):
        translate.convert_tools([{"type": "file_search"}])


def test_convert_tools_rejects_unnamed_tool():
    with pytest.raises(TranslationError, match="missing 'name'"):
        translate.convert_tools([{"type": "function", "function": {}}])


# -- tool_choice -----------------------------------------------------------


def test_tool_choice_none_drops_all_tools():
    tools = translate.convert_tools([WEATHER])
    assert translate.apply_tool_choice(tools, "none") == ([], False)


def test_tool_choice_auto_keeps_all_tools():
    tools = translate.convert_tools([WEATHER])
    assert translate.apply_tool_choice(tools, "auto") == (tools, False)


def test_tool_choice_named_declares_only_that_tool():
    tools = translate.convert_tools(
        [WEATHER, {"type": "function", "function": {"name": "send_email"}}]
    )
    picked, forced = translate.apply_tool_choice(
        tools, {"type": "function", "function": {"name": "send_email"}}
    )
    assert [t["name"] for t in picked] == ["send_email"]
    assert forced is True


def test_tool_choice_named_unknown_tool_errors():
    tools = translate.convert_tools([WEATHER])
    with pytest.raises(TranslationError, match="not in 'tools'"):
        translate.apply_tool_choice(tools, {"type": "function", "function": {"name": "nope"}})


# -- response_format -----------------------------------------------------


def test_json_schema_response_format_becomes_a_tool():
    tool = translate.schema_from_response_format(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "Invoice",
                "schema": {"type": "object", "properties": {"vendor": {"type": "string"}}},
            },
        }
    )
    assert tool == {
        "name": "Invoice",
        "parameters": {"type": "object", "properties": {"vendor": {"type": "string"}}},
    }


def test_text_response_format_is_not_extraction():
    assert translate.schema_from_response_format({"type": "text"}) is None
    assert translate.schema_from_response_format(None) is None


def test_json_object_response_format_is_rejected():
    with pytest.raises(TranslationError, match="explicit schema"):
        translate.schema_from_response_format({"type": "json_object"})


def test_json_schema_without_schema_is_rejected():
    with pytest.raises(TranslationError, match="must be a JSON Schema"):
        translate.schema_from_response_format({"type": "json_schema", "json_schema": {}})


# -- messages -> turns ---------------------------------------------------


def test_system_messages_are_joined_and_separated_from_turns():
    system, turns = translate.build_turns(
        [
            {"role": "system", "content": "Be terse."},
            {"role": "developer", "content": "Never email anyone."},
            {"role": "user", "content": "weather in Lagos?"},
        ]
    )
    assert system == "Be terse.\n\nNever email anyone."
    assert [(t.kind, t.text) for t in turns] == [("user", "weather in Lagos?")]


def test_assistant_messages_are_dropped_and_tool_results_collapse():
    system, turns = translate.build_turns(
        [
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
            {"role": "tool", "tool_call_id": "call_2", "content": '{"temp_c": 18}'},
        ]
    )
    assert system == ""
    assert [t.kind for t in turns] == ["user", "tool_results"]
    # Consecutive tool results become one JSON array, as Needle.run feeds them.
    assert json.loads(turns[1].text) == [{"temp_c": 27}, {"temp_c": 18}]


def test_non_json_tool_result_is_passed_through_as_text():
    _, turns = translate.build_turns(
        [{"role": "user", "content": "hi"}, {"role": "tool", "content": "it is sunny"}]
    )
    assert json.loads(turns[1].text) == ["it is sunny"]


def test_multimodal_text_parts_are_flattened():
    _, turns = translate.build_turns(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "weather in"},
                    {"type": "text", "text": "Lagos"},
                ],
            }
        ]
    )
    assert turns[0].text == "weather in\nLagos"


def test_image_content_part_is_rejected():
    with pytest.raises(TranslationError, match="text only"):
        translate.build_turns(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
        )


def test_unknown_role_is_rejected():
    with pytest.raises(TranslationError, match="unsupported role"):
        translate.build_turns([{"role": "assistantt", "content": "hi"}])


# -- needle -> OpenAI ----------------------------------------------------


def _result(**kwargs) -> EngineResult:
    return EngineResult(raw=needle_response(**kwargs))


def test_function_calls_become_openai_tool_calls():
    body = translate.build_chat_completion(
        _result(function_calls=[{"name": "get_weather", "arguments": {"city": "Lagos"}}]),
        "needle-2",
    )
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["id"].startswith("call_")
    assert call["function"]["name"] == "get_weather"
    # Arguments are a JSON *string*, per the OpenAI schema.
    assert json.loads(call["function"]["arguments"]) == {"city": "Lagos"}


def test_no_call_falls_back_to_the_reasoning_trace_as_content():
    body = translate.build_chat_completion(
        _result(type_="respond", function_calls=[], reasoning="No tool available."), "needle-2"
    )
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "No tool available."
    assert choice["message"]["reasoning_content"] == "No tool available."
    assert "tool_calls" not in choice["message"]


def test_extraction_mode_returns_the_arguments_as_content():
    body = translate.build_chat_completion(
        _result(
            function_calls=[{"name": "Invoice", "arguments": {"vendor": "Acme", "total": 1200.0}}]
        ),
        "needle-2",
        extraction_mode=True,
    )
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert json.loads(choice["message"]["content"]) == {"vendor": "Acme", "total": 1200.0}
    assert "tool_calls" not in choice["message"]


def test_extraction_mode_with_no_match_sets_refusal():
    body = translate.build_chat_completion(
        _result(function_calls=[], reasoning="nothing to extract"),
        "needle-2",
        extraction_mode=True,
    )
    choice = body["choices"][0]
    assert choice["message"]["content"] is None
    assert choice["message"]["refusal"] == "nothing to extract"


def test_extras_carry_confidence_and_can_be_disabled():
    result = _result(
        function_calls=[], confidence=0.42, validation={"ungrounded": [], "negation": False}
    )
    with_extras = translate.build_chat_completion(result, "needle-2", include_extras=True)
    assert with_extras["x_needle"]["confidence"] == 0.42
    assert with_extras["x_needle"]["validation"] == {"ungrounded": [], "negation": False}
    without = translate.build_chat_completion(result, "needle-2", include_extras=False)
    assert "x_needle" not in without


def test_completion_shape_is_openai_compatible():
    body = translate.build_chat_completion(_result(), "needle-2")
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert set(body) >= {"id", "object", "created", "model", "choices", "usage"}
