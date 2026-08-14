# API reference

Base URL is `/v1`. Set `NEEDLE_API_KEY` to require
`Authorization: Bearer <key>` on every endpoint except `/health` and `/`.

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/v1/chat/completions` | POST | Tools, streaming, structured output. The main endpoint. |
| `/v1/completions` | POST | Legacy text completion. |
| `/v1/models` | GET | List the served model. |
| `/v1/models/{id}` | GET | Fetch one model card. |
| `/v1/needle/extract` | POST | Native extraction: text + JSON Schema → object. |
| `/v1/needle/complete` | POST | Passthrough returning the engine's raw response. |
| `/health` | GET | Readiness, queue depth, token-count mode. |
| `/` | GET | Endpoint index. |

## POST /v1/chat/completions

### Supported parameters

| Parameter | Support | Notes |
| --- | --- | --- |
| `messages` | ✅ | `system`, `developer`, `user`, `tool`, `function`. Assistant messages are dropped — see [fidelity notes](fidelity.md#assistant-messages-in-your-history-are-dropped). |
| `tools` | ✅ | `type: "function"` only. |
| `functions` | ✅ | Deprecated form, accepted. |
| `tool_choice` | ⚠️ | Honoured by narrowing what is declared. `"required"` cannot be enforced. |
| `response_format` | ⚠️ | `json_schema` → extraction mode. `json_object` is rejected. |
| `stream` | ⚠️ | Synthesized from the finished result, not true token streaming. |
| `stream_options.include_usage` | ✅ | Appends the usage chunk. |
| `max_tokens` / `max_completion_tokens` | ✅ | Maps to the engine's `max_new_tokens`. |
| `model` | ✅ | Any name accepted unless `NEEDLE_STRICT_MODEL=true`. |
| `temperature`, `top_p`, `seed`, `stop`, `logprobs`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `n` | ❌ | Accepted, ignored, and reported in `x_needle.warnings`. Decoding is deterministic. |
| Image / audio content parts | ❌ | Rejected with 400. Text only. |

### Response

Standard `chat.completion`, plus an `x_needle` block. A tool call:

```json
{
  "id": "chatcmpl-751cf77b6deea9564ca661d1",
  "object": "chat.completion",
  "created": 1786719243,
  "model": "needle-2",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "refusal": null,
      "tool_calls": [{
        "id": "call_e00c85cfc7c8c9e729c55b7e",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Lagos\"}"}
      }],
      "reasoning_content": "User asks for current weather in Lagos. get_weather with city 'Lagos' from query."
    },
    "logprobs": null,
    "finish_reason": "tool_calls"
  }],
  "usage": {"prompt_tokens": 34, "completion_tokens": 18, "total_tokens": 52},
  "x_needle": { "...": "see below" }
}
```

`finish_reason` is `tool_calls` when the model called something, `stop`
otherwise. When nothing is called, `content` carries the model's reasoning trace,
because that is its only textual output.

### The `x_needle` block

Needle reports signals the OpenAI schema has nowhere to put. Disable the block
with `--no-extras` / `NEEDLE_EXPOSE_EXTRAS=false`.

```json
{
  "type": "call",
  "confidence": 0.9568,
  "reasoning": "User asks for current weather in Lagos. get_weather with city 'Lagos' from query.",
  "validation": {"ungrounded": [], "negation": false},
  "prefill_tps": 299.3,
  "decode_tps": 180.0,
  "peak_ram_mb": 103.1,
  "replayed_turns": 1,
  "reinitialized": false,
  "queue_wait_seconds": 0.0004,
  "compute_seconds": 0.2433,
  "warnings": []
}
```

| Field | Meaning |
| --- | --- |
| `type` | `call` (tools invoked), `respond` (done, answering), `refuse` (declined). |
| `confidence` | Calibrated 0–1 score. **The field to gate on.** Context-sensitive — calibrate your own threshold. |
| `reasoning` | Short natural-language trace. Also mirrored to `message.reasoning_content`. |
| `validation.ungrounded` | Arguments the model could not trace back to the input — a hallucination check. |
| `validation.negation` | Whether a negation was detected in the request. |
| `prefill_tps` / `decode_tps` | Throughput, tokens/second. |
| `peak_ram_mb` | Peak resident memory of the engine. |
| `replayed_turns` | Engine calls this request needed. `1` means the session prefix was reused. |
| `reinitialized` | Whether the tool set or system prompt forced a fresh session. |
| `queue_wait_seconds` | Time spent waiting for the single engine thread. |
| `warnings` | Parameters that were accepted but had no effect. |

### Streaming

`stream: true` returns `text/event-stream` with `chat.completion.chunk` events
terminated by `data: [DONE]`. The sequence is: a role delta, then tool-call
deltas (name and id first, then argument fragments) or content deltas, then a
finish-reason chunk carrying `x_needle`, then optionally a usage chunk.

The chunks are well-formed and reassemble correctly, but they are cut from an
already-finished result — see
[fidelity notes](fidelity.md#streaming-is-synthesized).

## POST /v1/needle/extract

Native extraction without the chat envelope.

```bash
curl -s localhost:8000/v1/needle/extract -H 'Content-Type: application/json' -d '{
  "text": "Invoice from Acme Corp, $1,200.00, due 2026-09-01",
  "name": "Invoice",
  "schema": {"type": "object",
             "properties": {"vendor": {"type": "string"},
                            "total": {"type": "number"},
                            "due_date": {"type": "string"}},
             "required": ["vendor", "total", "due_date"]}
}'
```

```json
{
  "object": "needle.extraction",
  "data": {"vendor": "Acme Corp", "total": 1200.0, "due_date": "2026-09-01"},
  "matched": true,
  "x_needle": {"type": "call", "confidence": 0.3794, "...": "..."}
}
```

`matched` is `false` and `data` is `null` when the model declined to produce an
object.

## POST /v1/needle/complete

Passthrough that returns the engine's response verbatim — useful for debugging
or when you want the native shape rather than the OpenAI translation.

```bash
curl -s localhost:8000/v1/needle/complete -H 'Content-Type: application/json' -d '{
  "query": "what is it like in Lagos right now?",
  "tools": [{"name": "get_weather",
             "parameters": {"type": "object",
                            "properties": {"city": {"type": "string"}}}}]
}'
```

Accepts either `query` + `system`, or a full `messages` array. `tools` may be a
JSON array or a JSON string, and uses Needle's flat schema shape
(`{name, description, parameters}`) rather than OpenAI's nesting.

## POST /v1/completions

Legacy endpoint. Text `prompt` becomes a single user turn; the response `text` is
the tool calls as JSON, or the reasoning trace if nothing was called. Token-array
prompts are rejected with 400.

## GET /health

Open even when an API key is configured, so orchestrators can probe it.

```json
{"status": "ok", "model": "needle-2", "weights": "needle-2 (base)",
 "queue_depth": 0, "max_queue_depth": 32, "exact_token_counts": true}
```

`status` is `loading` until the engine finishes its first load. `queue_depth`
shows requests waiting for the engine thread; requests are rejected with 429 once
it reaches `max_queue_depth`.

## Errors

OpenAI-shaped envelopes throughout:

```json
{"error": {"message": "...", "type": "invalid_request_error",
           "param": "response_format", "code": "invalid_request_error"}}
```

| Status | When |
| --- | --- |
| 400 | Unmappable request — unsupported tool type, image content, `json_object`, empty conversation. |
| 401 | Missing or wrong bearer token. |
| 404 | Unknown model, with `NEEDLE_STRICT_MODEL=true`. |
| 429 | Engine queue full (`code: engine_busy`). Retry with backoff. |
| 502 | The engine itself reported an error. |
| 503 | Model still loading (`code: not_ready`), or shutting down. |
| 504 | Request exceeded `NEEDLE_REQUEST_TIMEOUT`. |

## Usage / token counts

The engine reports throughput but not token counts, so `usage` is reconstructed.
With `sentencepiece` installed (the `full` image target) the model's own
tokenizer gives exact counts; otherwise a character heuristic is used and
`usage.estimated: true` is set. Either figure excludes the engine's internal
chat-template framing, so treat it as a close lower bound rather than a billing
number.
