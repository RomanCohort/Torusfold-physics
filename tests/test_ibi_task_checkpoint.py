"""Per-task checkpoints: a round that dies must not take every finished chain with it.

Measured, 2026-09-16: four workers of the 867-chain loop crashed (VCRUNTIME140.dll at 23:04:18 x3,
ucrtbase.dll 0xc0000409 at 23:08:32), multiprocessing.Pool respawned them AT THOSE SAME SECONDS,
and the tasks they were holding were gone. pool.map then waited for results that could never
arrive -- the round never finished, and the completed chains' histograms were only ever in the
parent's memory. These tests pin the three pieces that make that impossible now: a result that
round-trips through disk, a heartbeat that can be stale, and an existing result being the resume
condition.
"""
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import boltzmann_bonded as B          # noqa: E402
import ibi_loop as L                  # noqa: E402


def _fake_result(joint_J=0.1716):
    return {"counts": {c: np.arange(64, dtype=np.int64) + i for i, c in enumerate(B.COORDS)},
            "n_outside": {c: i for i, c in enumerate(B.COORDS)},
            "n_total": {c: 1000 + i for i, c in enumerate(B.COORDS)},
            "joint_J": joint_J, "j_coords": [4, 4], "residues": 27, "seconds": 512.5,
            "entry": {"energy_0": 8908.52, "max_force_0": 4175.14, "at_cap_0": False},
            "relax": {"accepted": 1500, "rejected": 0, "steps": 1500, "evals": 1501,
                      "energy_start": 186901.1, "energy_end": 92512.1,
                      "max_force_start": 5000.0, "max_force_end": 5000.0,
                      "hit_cap": True, "left_cap": False}}


def test_a_result_round_trips_through_disk(tmp_path_less=None):
    import shutil
    d = REPO / "results" / f"_test_checkpoint_{int(time.time() * 1000)}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        r = _fake_result()
        L.save_task_result(d, 7, r)
        assert L.task_npz(d, 7).exists(), "the result file is the resume condition; it must exist"
        back = L.load_task_result(d, 7)
        for c in B.COORDS:
            assert np.array_equal(back["counts"][c], r["counts"][c]), c
        assert back["n_outside"] == r["n_outside"] and back["n_total"] == r["n_total"]
        assert back["joint_J"] == r["joint_J"] and back["j_coords"] == [4, 4]
        assert back["residues"] == 27 and abs(back["seconds"] - 512.5) < 1e-9
        # entry and relax are the two nested dicts the round json carries; they survive as JSON
        assert back["entry"] == r["entry"]
        assert back["relax"]["accepted"] == 1500 and back["relax"]["hit_cap"] is True
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_a_nan_joint_j_comes_back_as_none():
    """A chain whose J cannot be scored must not come back as a NaN that propagates into a mean."""
    import shutil
    d = REPO / "results" / f"_test_checkpoint_nan_{int(time.time() * 1000)}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        L.save_task_result(d, 0, _fake_result(joint_J=None))
        assert L.load_task_result(d, 0)["joint_J"] is None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_dead_tasks_separates_never_started_from_crashed():
    """The distinction the whole watchdog rests on.

    With 867 tasks and 33 workers, a task with no heartbeat file is simply still in the queue. A
    task that BEAT and then went silent is a worker that died holding it -- and Pool will never
    re-issue it, which is what cost round 0 its 21 hours.
    """
    import os
    import shutil
    d = REPO / "results" / f"_test_checkpoint_hb_{int(time.time() * 1000)}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        fresh, stale, never = (d / "0.hb"), (d / "1.hb"), (d / "2.hb")
        fresh.write_text("x", encoding="utf-8")
        stale.write_text("x", encoding="utf-8")
        old = time.time() - 3600
        os.utime(stale, (old, old))
        tasks = [(0, str(fresh)), (1, str(stale)), (2, str(never))]
        dead = L.dead_tasks(tasks, 420.0)
        assert [t[0] for t in dead] == [1], "only the task that beat and stopped is dead"
        assert L.dead_tasks([], 420.0) == []
        # A beat from BEFORE this attempt is not evidence. On a resume every file on disk is older
        # than STALE_S, and reading those as dead burned a pool cycle at every restart (measured
        # 2026-09-17: "DEAD WORKER: 25 task(s)" printed before a single step of the new attempt).
        assert L.dead_tasks(tasks, 420.0, since=time.time()) == [], (
            "heartbeats from a previous attempt must not kill this attempt's tasks")
        assert [t[0] for t in L.dead_tasks(tasks, 420.0, since=0.0)] == [1], (
            "with no cutoff the stale beat is still found")
        # a suspend freezes the parent too, so on wake nothing is dead -- it is all just old
        L.touch_heartbeats(tasks)
        assert L.dead_tasks(tasks, 420.0) == [], "a refreshed heartbeat must not read as death"
        assert not never.exists(), "a task that never started has no heartbeat to touch"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_an_existing_result_is_skipped():
    """The inline resume condition of main(): nothing to do for an index already on disk."""
    import shutil
    d = REPO / "results" / f"_test_checkpoint_skip_{int(time.time() * 1000)}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        tasks = [(i, str(d / f"{i}.hb")) for i in range(4)]
        L.save_task_result(d, 1, _fake_result())
        L.save_task_result(d, 3, _fake_result())
        remaining = [t for t in tasks if not L.task_npz(d, t[0]).exists()]
        assert [t[0] for t in remaining] == [0, 2]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_dispatch_is_longest_first_and_keeps_the_indices():
    """LPT scheduling, and the task index still names the checkpoint file.

    Measured on 2026-09-19: with the loader's alphabetical order, round 0's largest chain
    (8FMW_24, 2929 residues) was task 536 of 867, started late, and finished last -- the final
    stretch of the round ran on one core while thirty-one sat idle (19.9 CPU-seconds per 20 s of
    wall). Longest first puts that chain in the first wave instead.
    """
    tasks = [(0, "a"), (1, "b"), (2, "c"), (3, "d")]
    size = {0: 50, 1: 2000, 2: 7, 3: 300}
    order = L.longest_first(tasks, lambda i: size[i])
    assert [t[0] for t in order] == [1, 3, 0, 2], "longest chain first"
    assert [t[0] for t in tasks] == [0, 1, 2, 3], (
        "the input list must not be reordered: a renumbering would make a resume read one chain's "
        "result as another's, silently, because the files would still be valid")
    assert L.longest_first([], lambda i: 0) == []
