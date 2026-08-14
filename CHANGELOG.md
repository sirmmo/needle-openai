# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-08-14

Packaging and release-workflow fixes. No changes to server behaviour.

### Added

- PyPI project metadata: documentation, source, issue and changelog links, plus
  keywords and trove classifiers. The 0.1.0 project page had no links at all, and
  PyPI metadata cannot be edited after upload.

### Fixed

- The container's `latest` tag was gated on `{{is_default_branch}}`, which never
  matches on a tag ref — so no `latest` would ever have been published, while the
  README tells you to pull it. It now tracks the newest stable tag, excluding
  prereleases.
- Publishing jobs are gated on tag refs. A manual `workflow_dispatch` run would
  otherwise have created a GitHub release named "main"; it is now a safe dry run
  that builds and checks the distributions only.
- `pytest` works when invoked bare, not just as `python -m pytest`
  (`pythonpath = ["."]`).
- The live test suite no longer requires the server's dependencies. `conftest.py`
  imported FastAPI at module scope, which made it a hard requirement of the live
  job even though that job deliberately talks to the container over HTTP only.

## [0.1.0] - 2026-08-14

Initial release: an OpenAI-compatible HTTP API for Needle 2 (`cactus-needle`
2.0.3, engine 2.0.1).

### Added

- **`POST /v1/chat/completions`** with tool calling, synthesized SSE streaming,
  and structured output via `response_format: json_schema`.
- **`POST /v1/completions`** — legacy text completion.
- **`GET /v1/models`, `GET /v1/models/{id}`** — model discovery.
- **`POST /v1/needle/extract`** — native extraction from text plus a JSON Schema.
- **`POST /v1/needle/complete`** — passthrough returning the engine's raw response.
- **`GET /health`** — readiness, engine queue depth, token-count mode.
- **`x_needle` response block** exposing Needle's calibrated `confidence`,
  `reasoning` trace, `validation.ungrounded` hallucination check, throughput and
  request bookkeeping. Disable with `--no-extras`.
- **`tool_choice` support** by narrowing the declared tool set: `"none"` declares
  nothing, a named function declares only that one.
- **Optional bearer-token auth** via `NEEDLE_API_KEY`, with `/health` left open.
- **Tuned weight support** — serve a `.cact` archive with `NEEDLE_WEIGHTS`.
- **Exact token counts** when `sentencepiece` is available (the `full` image
  target), falling back to a character heuristic flagged as
  `usage.estimated: true`.
- Docker image (183MB base / 186MB with the tokenizer), Compose file, MkDocs
  documentation site, and CI covering lint, Python 3.10–3.13, and a live suite
  against the real engine.

### Notes on behaviour

These follow from the model and its FFI rather than from choices made here; see
[the fidelity notes](https://ingmmo.com/needle-openai/fidelity/) for the
full list.

- **Requests serialize through one thread.** The native engine is a single global
  session that segfaults when called concurrently, so it is confined to one
  worker thread with a bounded queue (429 past `NEEDLE_MAX_QUEUE_DEPTH`).
- **Needle 2 emits tool calls or nothing** — it has no free-form generation head.
  When no tool is called, its `reasoning` trace is returned as the assistant's
  `content`. It is not a chat model.
- **Output is deterministic.** `temperature`, `top_p`, `seed`, `n` and friends are
  accepted, ignored, and reported in `x_needle.warnings`.
- **Streaming is synthesized** from the finished result; the FFI has no token
  callback.
- **Assistant messages in a posted history are dropped** — the native session
  cannot be told what it previously said, so it regenerates its own turns.
- **`response_format: {"type": "json_object"}` returns 400.** Needle needs an
  explicit schema.
- **Long inputs truncate silently**, returning a confident-looking wrong call with
  `confidence: 0.0`. Gate on `confidence`.

[Unreleased]: https://github.com/sirmmo/needle-openai/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/sirmmo/needle-openai/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/sirmmo/needle-openai/releases/tag/v0.1.0
