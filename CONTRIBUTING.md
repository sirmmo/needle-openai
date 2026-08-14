# Contributing

Thanks for looking. This is a small, focused project: an OpenAI-compatible
translation layer over Needle 2. Contributions that keep it small and honest are
very welcome.

## Setup

```bash
pip install -r requirements.txt pytest httpx ruff
pytest          # 57 tests, no model download needed
ruff check . && ruff format --check .
```

## Ground rules

**Keep `translate.py` pure.** Every OpenAI ↔ Needle mapping decision lives there
as a function with no I/O, which is why the whole HTTP surface can be tested
without the native library. New mapping logic belongs there with a unit test, not
inline in a route handler.

**Run the live suite when you touch `engine.py`.**

```bash
docker compose up -d
NEEDLE_TEST_BASE_URL=http://127.0.0.1:8000 pytest -m live
```

The engine is a single global native session that segfaults under concurrent use.
The unit tests use a fake engine and cannot catch a regression in that
confinement — the symptom would be a crash under load in production instead.

**Document limitations rather than hiding them.** If Needle cannot do something
an OpenAI client expects, the right outcome is a clear 400 or a warning in
`x_needle.warnings`, plus an entry in
[`docs/fidelity.md`](https://sirmmo.github.io/needle-openai/fidelity/). Silently
accepting a parameter that does nothing is the thing we are trying to avoid.

**Verify claims against the real engine.** Most of this project's design came
from probing `libneedle.so` directly rather than trusting documentation. If you
assert the model behaves some way, please include how you checked — see
[Probing the model directly](https://sirmmo.github.io/needle-openai/development/#probing-the-model-directly).

## Pull requests

- One concern per PR.
- Lint and unit tests must pass; CI runs them on Python 3.10–3.13.
- Update `CHANGELOG.md` under "Unreleased".
- If you change or add an endpoint, update `docs/api-reference.md`.

## Reporting bugs

Please include the request body, the response (including the `x_needle` block if
present), and the output of `GET /health`. For anything involving load or
crashes, `docker logs` matters — a segfault in the native engine leaves very
little behind, so the surrounding log lines are often the only evidence.
