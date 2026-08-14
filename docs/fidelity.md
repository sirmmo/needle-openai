# Fidelity notes

Needle 2 is a 45M-parameter tool-caller, and this wrapper cannot paper over what
the model does not do. Every limitation below is a property of the model or its
FFI, not something a future version of this server will fix. They are documented
here rather than discovered in production.

Each one was verified against the real engine while building the wrapper.

## There is no free-form text

The model has no general generation head. It returns tool calls or an empty list;
its only natural-language output is a short `reasoning` trace.

When no tool is called, that trace becomes the assistant's `content` (and is
mirrored to `message.reasoning_content`), because a response with `content: null`
and no `tool_calls` breaks most clients. The consequence is that a bare
conversational prompt gets an odd answer:

```python
client.chat.completions.create(
    model="needle-2",
    messages=[{"role": "user", "content": "hello, who are you?"}],
    tools=TOOLS,
)
# content: "No tool available to query location or contact info."
```

!!! danger "Do not use this as a chatbot"
    Pair it with a full chat model: let Needle decide *which function to call*,
    and let the larger model do the talking.

## Assistant messages in your history are dropped

The native session is stateful but write-only. `needle_init`, `needle_complete`
and `needle_reset` offer no way to inject an assistant turn — the engine only
knows what it generated itself.

Replaying a conversation therefore feeds only user text and tool results, and the
engine regenerates its own intermediate turns. Because decoding is deterministic
this reproduces what it originally said, **as long as the tools and system prompt
are unchanged.** Rewriting history to put different words in the assistant's
mouth will not work: those messages are discarded.

Normal agent loops are unaffected — you append the assistant message for your own
bookkeeping and the server ignores it.

## Output is deterministic, and sampling parameters do nothing

The FFI exposes no temperature, top-p or seed. Identical requests return
identical bytes, confidence included — verified across process restarts.

`temperature`, `top_p`, `seed`, `stop`, `logprobs`, `top_logprobs`,
`presence_penalty`, `frequency_penalty`, `logit_bias` and `n` are accepted so
existing clients keep working, then ignored. Each one that had no effect is named
in `x_needle.warnings`, so the silence is visible rather than mysterious:

```json
"warnings": [
  "'temperature' ignored: needle-2 decodes deterministically and exposes no sampling controls"
]
```

There is no retry-with-higher-temperature strategy available here. If a call is
wrong, it will be wrong the same way every time — which at least makes it
reproducible.

## Streaming is synthesized

`needle_complete` returns one finished JSON object and offers no token callback,
so `stream: true` cuts the chunk sequence from the completed result.

The chunks are well-formed, carry a single stream id, and reassemble correctly
through the OpenAI SDK. What you do not get is progressive generation:
time-to-first-chunk equals full generation time. At ~150–180 tok/s on a model
this size that is a fraction of a second, so the practical difference is small —
but it is not real streaming, and a client measuring inter-token latency will see
it.

## `tool_choice` is honoured by narrowing what is declared

There is no decode-time tool forcing, so the choice is applied by changing what
the model is shown:

| Value | Behaviour |
| --- | --- |
| `"auto"` (default) | All tools declared. |
| `"none"` | No tools declared, so no call is possible. |
| `{"type": "function", "function": {"name": "x"}}` | Only that tool declared. |
| `"required"` | **Cannot be enforced.** All tools declared; the model still decides whether to call one. A warning says so. |

## `response_format: {"type": "json_object"}` is rejected

Needle is schema-driven and cannot emit free-form JSON. The request fails with
400 and a message pointing at the alternative:

```json
{"error": {"message": "response_format 'json_object' is not supported; needle-2 needs an explicit schema -- use {\"type\": \"json_schema\", \"json_schema\": {...}}",
           "type": "invalid_request_error", "param": "response_format"}}
```

`json_schema` works well — see [Quickstart](quickstart.md#extract-structured-data).

## Text input only

Image and audio content parts are rejected with 400. Needle 2 has no vision or
audio encoder.

## Long inputs truncate silently and confidently wrong

This is the failure mode most likely to bite you. A prompt past the context
window does **not** raise an error. In testing, a padded prompt asking about
Lagos returned a call for **Madrid** — with `confidence: 0.0`.

!!! warning "Check `confidence`"
    It is the only signal that truncation happened. There is no length error, no
    `finish_reason: "length"`, and the response looks structurally valid.

Keep prompts short — this is a 45M model designed for terse device-level requests,
not document analysis.

## Large tool sets degrade

With 60 tools declared, the model failed to call the one that was explicitly
asked for. Keep the declared set small and task-specific; use `tool_choice` to
narrow it per request when you have a large catalogue.

Needle supports a tool index (`tool_index_path`) for larger catalogues. This
server does not expose it yet.

## Token counts are reconstructed

The engine reports throughput (`prefill_tps`, `decode_tps`) but not token counts.

With `sentencepiece` installed — the `full` image target — the model's own
tokenizer gives exact counts for the text this server sends and receives.
Otherwise a character heuristic is used and `usage.estimated: true` is set. The
difference is material: for one request the heuristic reported 104 prompt tokens
where the real tokenizer counted 34.

Either figure excludes the engine's internal chat-template framing
(`<|im_start|>`, `<tools>` and friends), so treat `usage` as a close lower bound
rather than a billing number.

## One request at a time

Not a mapping limitation but worth stating here: the native engine is a single
global session and **crashes if called concurrently**, so all requests serialize
through one thread. Throughput is one request at a time per container, at roughly
200ms each. See [Architecture](architecture.md) for the details and how to scale.
