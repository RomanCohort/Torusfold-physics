"""Repo smoke tests.

Run with pytest (or `python -m pytest -q tests`). No heavy dependencies are
required beyond numpy: this proves every Python file at least compiles, and
that the `torusfold.scheme2` package imports.

Heavy numerical paths (OpenMM MD, external predictors) need a full install and
are intentionally NOT exercised here — see docs/NOTES.md and
docs/DEPLOY_EXTERNAL.md for those.
"""
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_all_python_compiles():
    import compileall

    ok = compileall.compile_dir(str(REPO / "src"), quiet=1, force=True)
    assert ok, "compileall failed under src/"
    for f in ("serve.py", "run_2013nt.py",):
        r = subprocess.run([sys.executable, "-m", "py_compile", str(REPO / f)])
        assert r.returncode == 0, f"py_compile failed on {f}"


def test_scheme2_package_imports():
    pytest = __import__("pytest")
    np = pytest.importorskip("numpy")  # numpy is the only hard import at package load
    sys.path.insert(0, str(REPO / "src"))
    import torusfold.scheme2  # noqa: F401

    assert torusfold.scheme2 is not None
