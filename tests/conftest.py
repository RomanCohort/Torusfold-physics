"""Pytest configuration for the TorusFold-Hybrid test suite.

WHY THIS FILE EXISTS -- it fixes a defect that made the whole suite unrunnable.

`tests/test_checkpoint_store.py` is a self-contained script: it asserts at module
level, prints PASS/FAIL lines as it goes, and ends with `sys.exit(1 if fails else 0)`.
Its own docstring says so ("Run: python tests/test_checkpoint_store.py"). That is a
legitimate shape, but it is not a pytest shape, and pytest's collection phase IMPORTS
every candidate module before deciding what to collect. The import therefore ran the
entire script, and the trailing `sys.exit(0)` surfaced as

    INTERNALERROR> SystemExit: 0
    no tests ran in 1.65s

Measured on 2026-10-02 with `python -m pytest -q tests` -- the exact command README.md
recommends a judge run to check the published numbers. The repository advertises 145
tests; this collected ZERO of them and reported an internal error instead. Every
other file in tests/ is a normal pytest module (33 of the 34 define `def test_*`).

Setting `__test__ = False` inside that file does NOT work, and the failure is
worth recording because it looks like it should: pytest must still import the
module to read that attribute, so the module-level side effect fires first and the
flag is never consulted. `collect_ignore` is the mechanism that skips the file
without importing it.

Both callers keep working:
    python -m pytest -q tests              -> collects the real suite, skips this file
    python tests/test_checkpoint_store.py  -> still runs as the standalone check it is
"""

collect_ignore = ["test_checkpoint_store.py"]
