# needle-openai

[![CI](https://github.com/sirmmo/needle-openai/actions/workflows/ci.yml/badge.svg)](https://github.com/sirmmo/needle-openai/actions/workflows/ci.yml)
[![Docs](https://github.com/sirmmo/needle-openai/actions/workflows/docs.yml/badge.svg)](https://sirmmo.github.io/needle-openai/)
[![PyPI](https://img.shields.io/pypi/v/needle-openai)](https://pypi.org/project/needle-openai/)
[![Python](https://img.shields.io/pypi/pyversions/needle-openai)](https://pypi.org/project/needle-openai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An OpenAI-compatible HTTP API in front of [Needle 2](https://github.com/cactus-compute/needle) —
Cactus Compute's 45M-parameter model for tool calling, device use and structured
extraction. Point any OpenAI client at it and get function calls back from a
14MB model that runs on CPU in ~100MB of RAM.

📖 **[Full documentation](https://sirmmo.github.io/needle-openai/)**

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

> [!IMPORTANT]
> Needle 2 is **not a chat model.** It emits tool calls or nothing at all — it has
> no free-form generation head. Read the
> [fidelity notes](https://sirmmo.github.io/needle-openai/fidelity/) before
> building on this; they list every place the OpenAI mapping is lossy and the two
> failure modes worth guarding against.

## Quickstart

```bash
docker run -d -p 8000:8000 -v needle-cache:/cache ghcr.io/sirmmo/needle-openai:latest
curl -s localhost:8000/health
```

Or from a clone:

```bash
docker compose up -d
```

Or as a Python package (needs Python ≥3.10):

```bash
pip install needle-openai
pip install --no-deps cactus-needle==2.0.3   # see "Why --no-deps" in the docs
needle-openai --port 8000
```

First start downloads the native engine and weights from HuggingFace into
`/cache` — a few tens of MB, ~6s on a warm connection, cached across restarts.

Runnable examples: [`examples/openai_client.py`](examples/openai_client.py) and
[`examples/curl.sh`](examples/curl.sh).

## What works

| | |
| --- | --- |
| **Tool calling** | `tools` / `functions`, parallel calls, the full agent loop. |
| **Structured extraction** | `response_format` with a JSON Schema → typed objects. |
| **Streaming** | SSE chunks, synthesized from the finished result. |
| **Confidence gating** | Calibrated 0–1 score on every response via `x_needle`. |
| **Tuned weights** | Serve a fine-tuned `.cact` with `NEEDLE_WEIGHTS`. |
| **Auth** | Optional bearer token; `/health` stays open. |

## Endpoints

| Endpoint | Notes |
| --- | --- |
| `POST /v1/chat/completions` | Tools, streaming, structured output. |
| `POST /v1/completions` | Legacy text completion. |
| `GET /v1/models`, `GET /v1/models/{id}` | Model discovery. |
| `POST /v1/needle/extract` | Native extraction: `text` + JSON Schema → object. |
| `POST /v1/needle/complete` | Passthrough returning the engine's raw response. |
| `GET /health` | Readiness, queue depth, token-count mode. |

Full details in the [API reference](https://sirmmo.github.io/needle-openai/api-reference/).

## Key things to know

A short version of the [fidelity notes](https://sirmmo.github.io/needle-openai/fidelity/):

- **No free-form text.** When no tool is called, the model's `reasoning` trace is
  returned as `content`. Asked `"hello, who are you?"` it replies *"No tool
  available to query location or contact info."*
- **Deterministic.** `temperature`, `top_p`, `seed` and `n` are accepted, ignored,
  and reported in `x_needle.warnings`.
- **One request at a time per container.** The native engine is a single global
  session that segfaults under concurrent use, so everything serializes through
  one thread. 24 concurrent requests complete correctly in ~4.2s; scale by
  running more containers.
- **Long inputs truncate silently** and return a confident-looking wrong answer
  with `confidence: 0.0`. Gate on `confidence`.
- **`response_format: {"type": "json_object"}` returns 400** — Needle needs an
  explicit schema.

## Development

```bash
pip install -r requirements.txt pytest httpx ruff
pytest                    # 57 tests, ~1s, no model download (fake engine)
```

```bash
docker compose up -d      # live suite against the real engine
NEEDLE_TEST_BASE_URL=http://127.0.0.1:8000 pytest -m live
```

See [Development](https://sirmmo.github.io/needle-openai/development/) and
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE). Needle 2 and its weights are licensed by
[Cactus Compute](https://github.com/cactus-compute/needle).
