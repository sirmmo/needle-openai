# Architecture

```
HTTP request ──> FastAPI (async) ──> queue ──> engine thread ──> libneedle.so
                     │                              │
              translate.py                    one global session
         (OpenAI <-> needle shapes)        (init / complete / reset)
```

## Everything serializes through one thread

This is not a performance compromise. It is a correctness requirement.

`needle.Needle` is a thin ctypes shim over `libneedle.so`, and the native side
keeps a **single global session**: `needle_init` rebinds it, `needle_reset` clears
its history, and the Python module tracks the current owner in module-level
globals (`_active`, `_active_weights`). Two threads calling it concurrently
**segfaults the process** — verified while building this wrapper, with two threads
each constructing a `Needle` with different tool sets.

So the engine is confined to a single `ThreadPoolExecutor(max_workers=1)`, which
also performs the initial library load and weight upload. All native state stays
on one thread for the process lifetime. FastAPI handlers submit work and await the
future, so the async event loop stays responsive while requests queue.

Consequences:

- **One uvicorn worker.** Multiple processes would each load their own copy of
  the weights and gain nothing.
- **Bounded queue.** Past `NEEDLE_MAX_QUEUE_DEPTH` requests get a 429 rather than
  piling up into timeouts.
- **Measured throughput.** 24 concurrent requests complete correctly in ~4.2s,
  roughly 200ms each, serialized.
- **Scale horizontally** — see [deployment](configuration.md#deployment).

## Session reuse

The native session is stateful, which the wrapper exploits rather than fights.

When consecutive requests share a system prompt, tool set **and** message prefix,
the engine keeps its history and only the new turns are fed. A typical agent loop
— where each request is the previous one plus a tool result — therefore costs one
`complete()` call per step instead of replaying the whole conversation.

Anything else resets to a clean session first. `needle_reset()` restores a virgin
session, verified by comparing confidence scores against a freshly started
process.

`x_needle.replayed_turns` reports how many engine calls a request actually
needed, and `x_needle.reinitialized` whether the tool set or system prompt forced
a fresh session:

| Situation | `replayed_turns` | `reinitialized` |
| --- | --- | --- |
| First request | all turns | `true` |
| Agent loop continuation | `1` | `false` |
| Same prefix, different tools | all turns | `true` |
| Unrelated conversation | all turns | `false` (reset, not re-init) |

Prefix reuse is safe because a matching prefix means the history is byte-identical
and decoding is deterministic — the engine's own generated turns are exactly what
a replay would reproduce.

## History replay

Because the session cannot be told what it previously said, replaying a
conversation means feeding only the inputs:

- `system` / `developer` messages are joined and passed to `needle_init`.
- `user` messages are fed verbatim, one `complete()` call each.
- Consecutive `tool` messages collapse into a single JSON array — mirroring how
  upstream `needle.Needle.run` feeds tool results back.
- `assistant` messages are **dropped**; the engine regenerates its own. See
  [fidelity notes](fidelity.md#assistant-messages-in-your-history-are-dropped).

`NEEDLE_MAX_REPLAY_STEPS` caps how many engine calls one request may trigger, so
a client posting a thousand-message transcript gets a clear error instead of
monopolizing the engine.

## Response mapping

Needle's response vocabulary is narrow — `type` is `call`, `respond` or `refuse`,
with `function_calls`, `reasoning`, `confidence` and `validation` alongside. The
mapping:

| Needle result | OpenAI response |
| --- | --- |
| `function_calls` non-empty | `finish_reason: "tool_calls"`, `content: null`, `tool_calls` populated |
| `function_calls` empty | `finish_reason: "stop"`, `content` = the reasoning trace |
| Extraction mode, call present | `finish_reason: "stop"`, `content` = the arguments as JSON |
| Extraction mode, no call | `content: null`, `refusal` = the reasoning trace |
| `error` set | HTTP 502 with the engine's `error_code` |

All of it lives in `translate.py` as pure functions, so the mapping is unit-tested
without the native library.

## The output buffer

The native side writes JSON into a fixed-size buffer and truncates silently when
it is too small — the failure surfaces as a `json.JSONDecodeError` deep in the
ctypes shim rather than anything diagnostic. The default here is 1MB
(`NEEDLE_BUFFER_SIZE`), well above the few hundred bytes a typical response needs.

## Why `--no-deps`

`cactus-needle` declares `jax`, `jaxlib`, `flax`, `optax` and `sentencepiece`, but
those are for training, LoRA fine-tuning and `.cact` export. Inference is a ctypes
call into `libneedle.so` and needs only `huggingface_hub` to fetch the engine and
weights.

Installing with `--no-deps` keeps the image at **183MB** instead of well over a
gigabyte. The `full` build target adds `sentencepiece` alone (186MB) for exact
token counts.

The engine binary itself is fetched at runtime from the
[Cactus-Compute/needle2](https://huggingface.co/Cactus-Compute/needle2)
HuggingFace repo, matched to the platform (`manylinux2014_x86_64`,
`manylinux2014_aarch64`, musl and macOS variants), and cached under `/cache`.

## Module layout

| File | Responsibility |
| --- | --- |
| `engine.py` | Thread-confined native session; queueing, session reuse, replay. |
| `translate.py` | Pure OpenAI ↔ Needle mapping. All the interesting decisions, no I/O. |
| `server.py` | FastAPI routes, auth, error shapes. |
| `streaming.py` | SSE rendering of a finished result. |
| `tokens.py` | Exact-or-estimated token accounting. |
| `schemas.py` | Permissive request models; names the ignored parameters. |
| `config.py` | Settings from the environment. |
