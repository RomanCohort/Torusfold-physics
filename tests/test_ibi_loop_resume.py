"""The resume path: what a restarted loop must not quietly change.

--start-round=N exists because a 21-hour round was lost to a machine that went down between
rounds (867 chains, round 0 finished, round 1 killed at its first task). The two ways a resume
can be wrong are both SILENT, which is why they are pinned here rather than left to the run:

  * an empty divergence history. plan_update refuses a correction that has risen for PATIENCE
    consecutive rounds and is GROWTH times its value at the start of that window; with fewer
    than PATIENCE+1 values the rule cannot fire at all, so a resumed run would wave through
    exactly the rounds after a rise that a continuous run would have refused.
  * a table on bins that are not the reference's. The update compares the histogram against
    p_ref bin by bin, and two different bin layouts do not raise -- they compare.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import ibi_loop as L          # noqa: E402


@pytest.fixture
def scratch_dir():
    """A directory these tests may write round jsons into.

    tmp_path is the idiomatic fixture and it is unusable under a sandbox that denies the system
    temp directory: measured, tmp_path errors at SETUP (os.scandir on its own base dir is
    denied), which reads in the summary exactly like the test failing. This one lives inside the
    repository, where the same sandbox allows mkdir/write/rmtree, and it is removed on the way
    out so a run leaves nothing behind.
    """
    d = REPO / "results" / f"_test_ibi_loop_resume_{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _table(lo=0.0, hi=1.0, nbins=100):
    binw = (hi - lo) / nbins
    return {"lo": lo, "hi": hi, "binw": binw, "U": [0.0] * nbins,
            "centre": [lo + 0.5 * binw] * nbins, "sigma": 0.05}


def _round_json(path, values):
    """A round json carrying only what a resume reads out of it."""
    path.write_text(json.dumps({"updates": {c: {"max_abs_dU": v}
                                            for c, v in values.items()}}), encoding="utf-8")


def test_start_round_flag_is_read_and_typed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ibi_loop.py", "867", "4", "1", "32500", "5",
                                      "--start-round=2"])
    assert L._flag_int("start-round", 0) == 2
    monkeypatch.setattr(sys, "argv", ["ibi_loop.py", "867", "4"])
    assert L._flag_int("start-round", 7) == 7
    monkeypatch.setattr(sys, "argv", ["ibi_loop.py", "--start-round=x"])
    with pytest.raises(SystemExit):
        L._flag_int("start-round", 0)


def test_history_is_rebuilt_per_coordinate_oldest_first(scratch_dir):
    _round_json(scratch_dir / "round0.json", {"bb_bond": 7.94, "angle": 4.85, "dihedral": 5.52})
    _round_json(scratch_dir / "round1.json", {"bb_bond": 2.78, "angle": 5.56, "dihedral": 1.22})
    hist = L.history_from_rounds(scratch_dir, 2)
    assert hist["bb_bond"] == [7.94, 2.78]
    assert hist["angle"] == [4.85, 5.56]
    assert hist["dihedral"] == [5.52, 1.22]
    # a fresh run has no history, and that is not the same thing as a missing file
    assert all(v == [] for v in L.history_from_rounds(scratch_dir, 0).values())


def test_history_refuses_a_hole_instead_of_starting_empty(scratch_dir):
    _round_json(scratch_dir / "round0.json", {"bb_bond": 1.0, "angle": 1.0, "dihedral": 1.0})
    with pytest.raises(SystemExit):
        L.history_from_rounds(scratch_dir, 2)      # round1.json absent


def test_bins_must_match_the_reference():
    ref = {"bb_bond": _table(), "angle": _table(), "dihedral": _table()}
    same = {c: dict(t) for c, t in ref.items()}
    L.check_bins_agree(same, ref, "same")                     # identical: no complaint
    for key in ("lo", "hi", "binw", "sigma"):
        off = {c: dict(t) for c, t in ref.items()}
        off["dihedral"][key] = off["dihedral"][key] * 1.01 + 0.01
        with pytest.raises(SystemExit):
            L.check_bins_agree(off, ref, f"{key} off")
    short = {c: dict(t) for c, t in ref.items()}
    short["angle"] = _table(nbins=64)
    with pytest.raises(SystemExit):
        L.check_bins_agree(short, ref, "half the bins")


def test_start_round_outside_the_run_is_refused(monkeypatch):
    # n_rounds=4 means rounds 0..3, so a resume at 4 names a round this run will not do
    monkeypatch.setattr(sys, "argv", ["ibi_loop.py", "7", "4", "16", "32500", "25",
                                      "--start-round=4"])
    with pytest.raises(SystemExit):
        L.main()


def test_resume_round_is_where_sampling_starts(monkeypatch):
    """The loop's round range is a function of start_round, not always range(n_rounds)."""
    src = (REPO / "scripts" / "ibi_loop.py").read_text(encoding="utf-8")
    assert "for rnd in range(start_round, n_rounds):" in src
    assert "tables_r{start_round}.npz" in src
