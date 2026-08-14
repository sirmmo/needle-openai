# Development

```bash
git clone https://github.com/sirmmo/needle-openai
cd needle-openai
pip install -r requirements.txt pytest httpx ruff
```

## Tests

Two suites, split by whether they need the real model.

### Unit suite — no engine, runs anywhere

```bash
pytest
# 57 passed, 1 skipped in ~1s
```

A fake engine drives the whole HTTP surface, so these need no native library and
no model download. They cover the OpenAI ↔ Needle mapping, every endpoint,
streaming chunk sequences, auth, and each error path.

### Live suite — the real engine

Skipped unless you point it at a running server:

```bash
docker compose up -d
NEEDLE_TEST_BASE_URL=http://127.0.0.1:8000 pytest -m live
# 8 passed
```

These exercise real tool calls, the tool-result round trip, structured
extraction, streaming, determinism across requests, and — most importantly — 24
concurrent requests to prove the single-session serialization holds.

!!! important "Run the live suite after touching `engine.py`"
    The thread confinement is exactly what a fake engine cannot verify. Without
    it, a regression there is a process segfault under load rather than a test
    failure.

## Lint

```bash
ruff check .
ruff format --check .
```

100-column lines, with `E`, `F`, `W`, `I`, `UP`, `B`, `C4` and `SIM` rules
enabled. CI runs both and fails on either.

## Probing the model directly

When you need to know what the engine actually does — rather than what the docs
claim — the fastest path is a throwaway container:

```bash
docker run --rm -v needle-cache:/root/.cache python:3.12-slim sh -c '
  pip install -q --no-deps cactus-needle==2.0.3 && pip install -q huggingface_hub && python -u -c "
import json, needle
tools = [{\"name\": \"get_weather\",
          \"parameters\": {\"type\": \"object\",
                         \"properties\": {\"city\": {\"type\": \"string\"}}}}]
agent = needle.Needle(tools=tools)
print(json.dumps(agent.complete(\"weather in Lagos?\"), indent=1))
"'
```

Most of this wrapper's design came from doing exactly that. Two cautions learned
the hard way: run with `python -u`, because a segfault loses buffered stdout and
leaves you with an empty log; and never call the engine from two threads at once
unless you are deliberately testing that it crashes.

## CI

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `ci.yml` | push to `main`, PRs | Ruff lint, unit tests on Python 3.10–3.13, then builds the image and runs the live suite against it. |
| `docs.yml` | changes to `docs/`, `mkdocs.yml` | Builds the site with `--strict`; deploys to GitHub Pages from `main`. |
| `release.yml` | `v*` tags | Builds sdist/wheel → PyPI (trusted publishing), multi-arch image → GHCR, and a GitHub release. |

The live job in `ci.yml` downloads the engine from HuggingFace. Adding an
`HF_TOKEN` repository secret avoids anonymous rate limits, but it is optional.

## Docs

```bash
pip install -r docs/requirements.txt
mkdocs serve          # http://127.0.0.1:8000
mkdocs build --strict # what CI runs; fails on broken internal links
```

## Releasing

1. Update `CHANGELOG.md` and the version in `pyproject.toml`.
2. Tag and push:

   ```bash
   git tag v0.1.1 && git push origin v0.1.1
   ```

3. `release.yml` handles PyPI, GHCR and the GitHub release.

PyPI publishing uses trusted publishing (OIDC), so there is no token secret to
rotate — but the publisher must be configured once at
`https://pypi.org/manage/project/needle-openai/settings/publishing/` with
owner `sirmmo`, repo `needle-openai`, workflow `release.yml`, environment `pypi`.

## Contributing

See
[CONTRIBUTING.md](https://github.com/sirmmo/needle-openai/blob/main/CONTRIBUTING.md).
The short version: keep `translate.py` pure and unit-tested, and if you change
engine behaviour, say which live test proves it.
