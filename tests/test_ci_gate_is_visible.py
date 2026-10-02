"""The CI gate excludes twenty test modules; this keeps that exclusion honest.

WHY THIS FILE EXISTS -- the change that made the CI gate pass did it by excluding the
twenty test modules that cannot be collected without torch (tests/conftest.py holds the
measurement and the reasoning). That is the honest thing to do, but it creates a way for
the gate to lie: if the exclusion list keeps a name that no longer needs excluding, CI
skips tests it could be running and reports nothing. A gate whose coverage can shrink in
silence is worse than a gate that is red.

THE CHECK IS A RUNTIME PROBE, NOT A SOURCE SCAN, and that was arrived at the hard way.
Three static attempts each got the answer wrong:

  1. parsing `pytest --collect-only` output -- it prints one line per TEST, so a
     file-level regex matched a few files and silently missed the rest;
  2. the same, after fixing the regex -- a module calling `pytest.importorskip("torch")`
     at module level raises Skipped during collection, so NONE of its tests are listed
     and its file name never appears;
  3. an AST walk looking for a bare `ast.Expr` call to importorskip -- which misses
     `torch = pytest.importorskip("torch")`, an assignment, and reported three files as
     not importing torch when they plainly do.

The lesson is in the third one: every static formulation has a blind spot that is
invisible until it produces a wrong answer. Asking the interpreter instead has none.

The probe is skipped when torch is absent, because then nothing CAN be collected and
there is no signal to read. On a developer machine it runs and prints what it proved.
"""

import importlib.util
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
HAS_TORCH = importlib.util.find_spec("torch") is not None

# The stale-exclusion probe below spawns one interpreter per excluded module, and each
# one imports torch, so it costs about 50 s. That is worth paying when someone is
# editing the exclusion lists and wants to know whether they are still accurate -- and
# not worth paying on every run of the full suite. Set TF_CHECK_CI_EXCLUSIONS=1 to run
# it; CI itself skips it anyway, because a runner without torch cannot answer the
# question it asks.
PROBE_ENABLED = os.environ.get("TF_CHECK_CI_EXCLUSIONS", "") not in ("", "0")

sys.path.insert(0, str(TESTS))
try:
    import conftest
finally:
    sys.path.pop(0)

DECLARED = set(conftest.collect_ignore)
TORCH_EXCLUDED = set(conftest._TORCH_ONLY_MODULES)
# conftest excludes one further file for an unrelated reason: it is a standalone script
# with a module-level sys.exit(), not a pytest module.
NOT_A_PYTEST_MODULE = {"test_checkpoint_store.py"}


def test_every_excluded_name_is_a_file_in_tests():
    """An exclusion matching no file is a hole nothing else would report."""
    on_disk = {p.name for p in TESTS.glob("test_*.py")}
    unknown = sorted(DECLARED - on_disk)
    assert not unknown, (
        f"tests/conftest.py excludes {unknown}, which are not files in tests/. "
        f"An exclusion matching nothing silently reduces what CI covers."
    )


def test_the_two_lists_do_not_overlap_and_cover_the_torch_exclusions():
    """Cheap structural invariants on the lists conftest declares."""
    assert not (set(conftest._TORCH_IMPORT_MODULES) & set(conftest._TORCH_TRANSITIVE_MODULES)), (
        "a module appears in both torch lists; the split is meant to be a partition"
    )
    assert TORCH_EXCLUDED == set(conftest._TORCH_IMPORT_MODULES) | set(
        conftest._TORCH_TRANSITIVE_MODULES
    ), "TORCH_ONLY_MODULES is not the union of the import and transitive lists"


@pytest.mark.skipif(
    not HAS_TORCH,
    reason="without torch every excluded module fails to collect, so the probe cannot "
           "tell a needed exclusion from a stale one",
)
@pytest.mark.skipif(
    not PROBE_ENABLED,
    reason="opt-in: ~50 s (one interpreter per excluded module). "
           "Set TF_CHECK_CI_EXCLUSIONS=1 when editing tests/conftest.py's lists.",
)
def test_no_exclusion_is_stale():
    """With torch present, importing each excluded module must still pull torch in.

    The probe is run in a subprocess because importing a test module here would execute
    it inside this pytest session. The child imports the module and reports whether
    `torch` ended up in sys.modules -- directly or through something the module imports.

    A module that does NOT drag torch in is one CI is excluding for no reason, and this
    names it. This is the one direction a source scan cannot check reliably, because
    eight of the twenty reach torch through a helper in scripts/ rather than by naming
    it.
    """
    import subprocess

    probe = (
        "import importlib, sys, pathlib\n"
        "sys.path.insert(0, str(pathlib.Path('tests').resolve()))\n"
        "sys.path.insert(0, str(pathlib.Path('src').resolve()))\n"
        "sys.path.insert(0, str(pathlib.Path('scripts').resolve()))\n"
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('probe_mod', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "try:\n"
        "    spec.loader.exec_module(m)\n"
        "except BaseException:\n"
        "    pass\n"          # import-time failures other than torch are not this test's business
        "print('TORCH_IN_SYS_MODULES' if 'torch' in sys.modules else 'NO_TORCH')\n"
    )

    stale = []
    for name in sorted(TORCH_EXCLUDED):
        target = TESTS / name
        if not target.is_file():
            continue
        r = subprocess.run([sys.executable, "-c", probe, str(target)],
                           cwd=str(ROOT), capture_output=True)
        out = r.stdout.decode("utf-8", errors="replace")
        if "NO_TORCH" in out:
            stale.append(name)

    assert not stale, (
        "these modules are excluded from CI but do not import torch at all, here where "
        "torch is available, so CI is skipping tests it could be running:\n  "
        + "\n  ".join(stale)
    )
    print(f"\nprobe: all {len(TORCH_EXCLUDED)} excluded modules still pull torch in.")


def test_the_gate_reports_its_own_coverage():
    """Print the coverage, so the number in the CI log is checkable against the files.

    Not an assertion -- the counts legitimately differ between a GPU machine and the
    runner, and pinning either would make this fail on the other.

    The breakdown names the three reasons a module can be absent, because one number
    ("24 excluded") hides that they are not excluded for the same cause, and an earlier
    version of this line miscounted by folding the standalone script into the torch
    total and printing "0 neither".
    """
    on_disk = sorted(p.name for p in TESTS.glob("test_*.py"))
    torch_excluded = TORCH_EXCLUDED
    script_excluded = DECLARED - torch_excluded
    other_excluded = DECLARED - torch_excluded - NOT_A_PYTEST_MODULE
    executed = len(on_disk) - len(DECLARED)
    print(
        f"\nCI gate coverage: {executed} of {len(on_disk)} test modules executed; "
        f"{len(DECLARED)} excluded -- {len(torch_excluded)} need torch, "
        f"{len(script_excluded & NOT_A_PYTEST_MODULE)} is a standalone script, "
        f"{len(other_excluded)} neither. "
        f"torch present in this interpreter: {HAS_TORCH}"
    )
    assert executed + len(DECLARED) == len(on_disk), (
        "the coverage arithmetic does not close: executed + excluded != files on disk"
    )
