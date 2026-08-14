# Configuration

Every setting is an environment variable. The common ones also have CLI flags —
run `needle-openai --help` (or `python -m needle_openai --help`) to list them.
CLI flags win over environment variables.

The repository ships a
[`.env.example`](https://github.com/sirmmo/needle-openai/blob/main/.env.example)
you can copy to `.env` for Docker Compose.

## Reference

| Variable | Flag | Default | Purpose |
| --- | --- | --- | --- |
| `NEEDLE_HOST` | `--host` | `0.0.0.0` | Bind address. |
| `NEEDLE_PORT` | `--port` | `8000` | Bind port. |
| `NEEDLE_API_KEY` | `--api-key` | unset | Require `Authorization: Bearer <key>`. Unset serves openly. |
| `NEEDLE_WEIGHTS` | `--weights` | unset | Serve a tuned `.cact` instead of base needle-2. |
| `NEEDLE_MODEL_ID` | `--model-id` | `needle-2` | Id advertised by `/v1/models`. |
| `NEEDLE_STRICT_MODEL` | `--strict-model` | `false` | 404 on other model names. |
| `NEEDLE_MAX_NEW_TOKENS` | `--max-new-tokens` | `256` | Generation cap when the request omits `max_tokens`. |
| `NEEDLE_MAX_QUEUE_DEPTH` | `--max-queue-depth` | `32` | Reject with 429 past this backlog. |
| `NEEDLE_REQUEST_TIMEOUT` | `--request-timeout` | `120` | Per-request deadline, seconds. |
| `NEEDLE_MAX_REPLAY_STEPS` | — | `32` | Cap on engine calls used to replay one conversation. |
| `NEEDLE_EXPOSE_EXTRAS` | `--no-extras` | `true` | Include the `x_needle` block on responses. |
| `NEEDLE_EXACT_TOKEN_COUNTS` | — | `true` | Use the sentencepiece tokenizer for `usage` when available. |
| `NEEDLE_BUFFER_SIZE` | — | `1048576` | Output buffer handed to the native engine. |
| `NEEDLE_ALLOWED_ORIGINS` | — | `*` | CORS origins, comma-separated. |
| `HF_TOKEN` | — | unset | Avoids anonymous HuggingFace rate limits on first download. |

## Notes on specific settings

### `NEEDLE_STRICT_MODEL`

Off by default. Many OpenAI clients and frameworks hard-code model names like
`gpt-4o-mini`, and rejecting them would break drop-in use for no benefit — there
is only one model being served either way. Turn it on when you want an explicit
error instead of silent substitution.

### `NEEDLE_MAX_QUEUE_DEPTH`

Requests serialize through a single engine thread, so an unbounded queue converts
load into timeouts. Past this depth the server returns 429 with
`code: engine_busy`, which a client can back off on. Watch `queue_depth` on
`/health` and `x_needle.queue_wait_seconds` on responses to size it.

### `NEEDLE_BUFFER_SIZE`

The native engine writes its JSON response into a fixed buffer and **truncates
silently** when it is too small, which surfaces as a JSON decode error rather
than anything diagnostic. The 1MB default is generous; raise it only if you
declare very large tool sets and see decode failures.

### `NEEDLE_EXACT_TOKEN_COUNTS`

Requires `sentencepiece`, which the `full` image target includes and the `base`
target does not. When unavailable the server falls back to a character heuristic
and flags `usage.estimated: true`. See
[fidelity notes](fidelity.md#token-counts-are-reconstructed).

## Tuned weights

Fine-tune with `needle finetune` / `needle build` from the upstream
[cactus-needle](https://github.com/cactus-compute/needle) package, then serve the
result:

```bash
cp my_tuned.cact weights/
NEEDLE_WEIGHTS=/weights/my_tuned.cact docker compose up -d
```

With Compose, `./weights` is mounted read-only at `/weights`.

!!! note "`.cact` archives are engine-version-locked"
    An archive exported by a different `cactus-needle` version will not load, and
    fails loudly at startup rather than serving a broken model. Re-run
    `needle build` on your checkpoint with the matching package version.

## Deployment

The server runs **one uvicorn worker** by design — the native engine is a single
global session (see [Architecture](architecture.md)). Scale horizontally:

```yaml
services:
  needle:
    image: ghcr.io/sirmmo/needle-openai:latest
    deploy:
      replicas: 4
    volumes:
      - needle-cache:/cache
```

Each replica holds its own copy of the model, which is the cheap thing about a
14MB model. Put any load balancer in front; requests are independent apart from
the [session-reuse optimization](architecture.md#session-reuse), which is a
latency nicety rather than a correctness requirement — sticky routing improves
agent-loop performance but nothing breaks without it.

Resource sizing: the model reported 31–103MB peak RAM across the requests tested.
The Compose file sets a 1GB limit, which is generous headroom for Python and the
HTTP stack.
