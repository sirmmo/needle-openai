# Quickstart

## Run the server

=== "Docker Compose"

    ```bash
    git clone https://github.com/sirmmo/needle-openai
    cd needle-openai
    docker compose up -d
    curl -s localhost:8000/health
    ```

=== "Prebuilt container"

    ```bash
    docker run -d -p 8000:8000 -v needle-cache:/cache \
      ghcr.io/sirmmo/needle-openai:latest
    ```

=== "Docker (local build)"

    ```bash
    docker build -t needle-openai .
    docker run -d -p 8000:8000 -v needle-cache:/cache needle-openai
    ```

=== "Python (no Docker)"

    Needs Python ≥3.10.

    ```bash
    pip install needle-openai
    pip install --no-deps cactus-needle==2.0.3   # see "Why --no-deps"
    needle-openai --port 8000
    ```

The first start downloads the native engine and weights from HuggingFace into
`/cache` — a few tens of MB, about 6 seconds on a warm connection, and cached
across restarts if you mount the volume. `GET /health` reports `loading` until
the model is resident, then `ok`:

```json
{"status": "ok", "model": "needle-2", "weights": "needle-2 (base)",
 "queue_depth": 0, "max_queue_depth": 32, "exact_token_counts": true}
```

!!! tip "Set `HF_TOKEN`"
    Anonymous HuggingFace downloads are rate-limited. Exporting `HF_TOKEN`
    avoids a slow or failed first start on shared CI runners and NAT'd networks.

## Make a tool call

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

TOOLS = [
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
]

response = client.chat.completions.create(
    model="needle-2",
    messages=[{"role": "user", "content": "what's it like in Lagos right now?"}],
    tools=TOOLS,
)
call = response.choices[0].message.tool_calls[0]
print(call.function.name, call.function.arguments)
# get_weather {"city": "Lagos"}

# Needle attaches a calibrated confidence score to every response.
print(response.model_extra["x_needle"]["confidence"])  # 0.957
```

## The agent loop

The standard OpenAI tool loop works unchanged — feed results back as
`role: "tool"` messages and the model stops calling once it has what it needs.

```python
import json

messages = [{"role": "user", "content": "what's it like in Lagos right now?"}]

for _ in range(4):
    response = client.chat.completions.create(model="needle-2", messages=messages, tools=TOOLS)
    message = response.choices[0].message
    if not message.tool_calls:
        print("final:", message.content)
        break

    messages.append(message.model_dump(exclude_none=True))
    for call in message.tool_calls:
        args = json.loads(call.function.arguments)
        result = {"city": args["city"], "temp_c": 27, "sky": "clear"}  # your tool
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
```

Consecutive requests that share a prefix reuse the engine's session, so each
loop iteration costs one model call rather than a full replay. See
[Architecture](architecture.md#session-reuse).

## Extract structured data

A JSON Schema in `response_format` switches the server into extraction mode.

```python
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
# {'vendor': 'Acme Corp', 'total': 1200.0, 'due_date': '2026-09-01'}
```

## Gate on confidence

This is the intended way to use a model this small: act on confident calls, and
escalate the rest to something bigger.

```python
THRESHOLD = 0.6  # calibrate against your own tools -- see below

response = client.chat.completions.create(model="needle-2", messages=messages, tools=TOOLS)
extras = response.model_extra["x_needle"]

if extras["confidence"] >= THRESHOLD and response.choices[0].message.tool_calls:
    dispatch(response.choices[0].message.tool_calls)
else:
    escalate_to_larger_model(messages)
```

!!! warning "Calibrate the threshold yourself"
    Confidence moves over a wide range with context. The same `get_weather`
    request scored **0.957** with one tool declared and **0.030** with two. There
    is no universally good default — measure it against your own tool set and
    prompts.

`x_needle.validation.ungrounded` complements this: it names arguments the model
could not trace back to the input, which is a direct hallucination check.

## More examples

The repository ships runnable examples:

- [`examples/openai_client.py`](https://github.com/sirmmo/needle-openai/blob/main/examples/openai_client.py)
  — tool calls, the agent loop, structured output and streaming via the OpenAI SDK.
- [`examples/curl.sh`](https://github.com/sirmmo/needle-openai/blob/main/examples/curl.sh)
  — every endpoint over plain `curl`.
