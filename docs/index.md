# needle-openai

An OpenAI-compatible HTTP API in front of [Needle 2](https://github.com/cactus-compute/needle) —
Cactus Compute's 45M-parameter model for tool calling, device use and structured
extraction. Point any OpenAI client at it and get function calls back from a
14MB model that runs on CPU in ~100MB of RAM.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="needle-2",
    messages=[{"role": "user", "content": "what's it like in Lagos right now?"}],
    tools=[
        {
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
    ],
)
print(response.choices[0].message.tool_calls[0].function)
# Function(arguments='{"city": "Lagos"}', name='get_weather')
```

!!! warning "Read the [fidelity notes](fidelity.md) before building on this"
    Needle 2 is **not a chat model.** It emits tool calls or nothing at all — it
    has no free-form generation head. Several parts of the OpenAI surface map
    onto it imperfectly, and that page says exactly where and why.

## What it is good at

Needle 2 is a specialist. Within its niche it is remarkable: a 14MB binary that
picks the right function and fills its arguments in about 200ms on a CPU, with a
calibrated confidence score attached so you can decide whether to trust it.

- **Tool / function calling** — the OpenAI `tools` parameter, unchanged.
- **Structured extraction** — `response_format` with a JSON Schema pulls typed
  objects out of text.
- **On-device and edge deployment** — no GPU, ~100MB peak RAM, deterministic
  output.
- **Confidence gating** — route low-confidence requests to a larger model
  instead of guessing.

## What it is not

Do not reach for this as a general assistant. Asked `"hello, who are you?"` with
no relevant tools, it replies with its own reasoning trace — *"No tool available
to query location or contact info."* — because that is the only text it can
produce. Use a full chat model for conversation and this for the narrow, fast,
cheap job of deciding which function to call.

## Where to go next

<div class="grid cards" markdown>

- **[Quickstart](quickstart.md)** — run it in Docker and make your first call.
- **[API reference](api-reference.md)** — endpoints, parameter support matrix,
  the `x_needle` extension block.
- **[Fidelity notes](fidelity.md)** — every place the OpenAI mapping is lossy,
  and the two failure modes worth guarding against.
- **[Configuration](configuration.md)** — environment variables and CLI flags.
- **[Architecture](architecture.md)** — why everything serializes through one
  thread, and why that is a correctness requirement.
- **[Development](development.md)** — running the tests, project layout.

</div>

## License

MIT. Needle 2 and its weights are licensed by
[Cactus Compute](https://github.com/cactus-compute/needle).
