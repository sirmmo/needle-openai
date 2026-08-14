"""Drive needle-openai with the official OpenAI Python SDK.

pip install openai
python examples/openai_client.py
"""

from __future__ import annotations

import json
import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("NEEDLE_BASE_URL", "http://127.0.0.1:8000/v1"),
    # Only checked if the server was started with NEEDLE_API_KEY.
    api_key=os.environ.get("NEEDLE_API_KEY", "not-needed"),
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]


def tool_call() -> None:
    print("--- tool call ---")
    response = client.chat.completions.create(
        model="needle-2",
        messages=[{"role": "user", "content": "what's it like in Lagos right now?"}],
        tools=TOOLS,
    )
    message = response.choices[0].message
    for call in message.tool_calls or []:
        print(f"{call.function.name}({call.function.arguments})")

    # Needle reports a calibrated confidence score with every response. Gating
    # on it is the intended way to use the model: act on high-confidence calls,
    # escalate the rest to a larger model.
    extras = response.model_extra.get("x_needle", {})
    print(f"confidence={extras.get('confidence')}  reasoning={extras.get('reasoning')!r}")


def full_loop() -> None:
    """The standard OpenAI agent loop, unchanged."""
    print("\n--- tool loop ---")
    messages: list[dict] = [{"role": "user", "content": "what's it like in Lagos right now?"}]

    for _ in range(4):
        response = client.chat.completions.create(model="needle-2", messages=messages, tools=TOOLS)
        message = response.choices[0].message
        if not message.tool_calls:
            print("final:", message.content)
            break

        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            result = {"city": args.get("city"), "temp_c": 27, "sky": "clear"}  # your tool here
            print(f"  -> {call.function.name}({args}) = {result}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
            )


def structured_output() -> None:
    print("\n--- structured output ---")
    response = client.chat.completions.create(
        model="needle-2",
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
    print(json.loads(response.choices[0].message.content))


def streaming() -> None:
    print("\n--- streaming ---")
    stream = client.chat.completions.create(
        model="needle-2",
        messages=[{"role": "user", "content": "weather in Tokyo?"}],
        tools=TOOLS,
        stream=True,
    )
    for chunk in stream:
        for choice in chunk.choices:
            for call in choice.delta.tool_calls or []:
                if call.function and call.function.name:
                    print(f"{call.function.name}(", end="", flush=True)
                if call.function and call.function.arguments:
                    print(call.function.arguments, end="", flush=True)
    print(")")


if __name__ == "__main__":
    tool_call()
    full_loop()
    structured_output()
    streaming()
