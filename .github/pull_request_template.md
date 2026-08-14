## What and why

<!-- What changes, and what problem it solves. -->

## Checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `pytest` passes (unit suite, no model needed)
- [ ] `CHANGELOG.md` updated under "Unreleased"
- [ ] Docs updated if an endpoint, parameter or limitation changed

## If this touches `engine.py`

The native engine is a single global session that segfaults under concurrent
use, and the unit tests run against a fake engine — so they cannot catch a
regression there.

- [ ] Live suite run: `NEEDLE_TEST_BASE_URL=... pytest -m live`
- [ ] Which live test covers this change: <!-- test name -->

## If this changes the OpenAI mapping

- [ ] Logic lives in `translate.py` as a pure function, with a unit test
- [ ] Anything Needle cannot honour returns a clear 400 or lands in
      `x_needle.warnings`, and is documented in `docs/fidelity.md`
