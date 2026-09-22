"""Where a 2929-residue chain's 18 hours go, measured piece by piece.

WHY THIS EXISTS, measured 2026-09-22. A round of the production loop is ~450 core-hours over 867
chains, which is a 14 h floor on 32 workers -- and the last three rounds took 18.2, 16.6 and 15.9
h. Every one of them ends with the same chain still running: 8FMW_24, L=2929, alone on one core
for 15.9 / 9.2 / 18.2 / 17.9 h. It is 4 percent of the round's work and all of its tail: for the
last 2-4 h of a round it is the only thing running and 31 of 32 cores are idle.

Two explanations call for opposite fixes and this script separates them:

  A. THE PER-BEAD COST IS FLAT. Then the chain is simply 139x the 21-residue chain the throughput
     numbers in ibi_loop's docstring were measured on, the tail is irreducible arithmetic, and the
     only lever left is the number of rounds.
  B. THE PER-BEAD COST GROWS WITH L. Then there is a superlinear term -- and the candidates are
     visible in the code: the per-frame clash metric is a chunked torch.cdist, O(n^2) in beads,
     and the clash neighbour list is rebuilt on a step counter.

So: time the pieces of one production step at three chain sizes and print per-bead cost against L.
A flat column says B is dead; a rising column says where the 18 hours actually are.

Read-only with respect to the production run: it writes no heartbeat, no task file and no table;
it runs one process, one torch thread, and is meant to be run WHILE the loop is in its tail.
"""
import inspect
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Same environment the scheduled launcher sets, because the potentials resolve their data through
# TORUSFOLD_RSRNASP and the table through IBI_LOOP_REF's siblings.
os.environ.setdefault("TORUSFOLD_RSRNASP", str(REPO / "_cgdata" / "combined"))
os.environ.setdefault("IBI_LOOP_OUT", str(REPO / "results" / "ibi_relax"))
os.environ.setdefault("IBI_LOOP_REF", str(REPO / "results" / "refit_smooth5.npz"))
# BEFORE torch: the reservation this saves was measured at 776 MB per process (957 -> 181 MB).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_BLOCKTIME"] = "0"

sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np                      # noqa: E402
import torch                            # noqa: E402

torch.set_num_threads(1)

import boltzmann_bonded as B            # noqa: E402
import ibi_core as IC                   # noqa: E402
import ibi_loop as IL                   # noqa: E402
import torusfold.scheme2.torch_cgsim as C  # noqa: E402

ROUND_NPZ = REPO / "results" / "ibi_relax" / "tables_r4.npz"
WARM = 5
REPS = 30


def timeit(fn, reps=REPS):
    """(best seconds, median seconds) over reps calls, best first so a noisy box cannot inflate."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def main():
    pool = [s for s in B.load_structures(limit=5000) if len(s["pairs"]) >= 8][:867]
    by_len = sorted(pool, key=lambda s: len(s["pos"]))
    picks = [by_len[0], by_len[len(by_len) // 2], by_len[-1]]
    print(f"pool {len(pool)} chains; timing {[ (s['name'], len(s['pos'])) for s in picks ]}")
    print(f"table {ROUND_NPZ.name}, {WARM} warmup then {REPS} timed calls, 1 torch thread")
    print()

    # ONE _build_potentials CALL: the closure over the table cannot be pickled, so this is the
    # same call a worker makes inside its own process (ibi_loop._build_potentials).
    pots, pot_kw = IL._build_potentials(ROUND_NPZ)
    print("potential kwargs:", {k: (type(v).__name__ if not isinstance(v, (int, float, str)) else v)
                                for k, v in pot_kw.items()})
    print()

    hdr = (f"{'chain':10s} {'L':>5s} {'beads':>6s} | {'cellbuild':>10s} {'field':>10s} "
           f"{'clashpair':>10s} | {'us/step/bead':>12s} {'field+clash':>12s}")
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for s in picks:
        L = len(s["pos"])
        tab = IC.load_tables(str(ROUND_NPZ))
        pos = torch.tensor(np.asarray(s["pos"], dtype=np.float64).reshape(1, 3 * L, 3))
        ij = torch.tensor(list(s["pairs"]), dtype=torch.long).reshape(-1, 2)
        pw = torch.ones(len(s["pairs"]), dtype=torch.float32)

        cl = C.GPUCellList(cell_size=1.5)
        cl.build(pos)

        def field():
            with torch.no_grad():
                C.cg_energy_forces(pos, ij, pw, cell_list=cl, force_cap=5000.0, **pot_kw)

        def build():
            cl.build(pos)

        def clash():
            IC.closest_bead_pair(pos.reshape(1, -1, 3))

        for _ in range(WARM):
            build(); field(); clash()

        b_best, b_med = timeit(build)
        f_best, f_med = timeit(field)
        c_best, c_med = timeit(clash)

        # A production step is one field call plus, every stride-th step, one clash metric. The
        # sampler's window is 2500 frames at stride 5 over 12500 steps, so the clash cost is
        # amortised over 5 steps.
        per_step = f_best + c_best / 5.0
        beads = 3 * L
        rows.append((s["name"], L, beads, b_best, f_best, c_best, per_step))
        print(f"{s['name']:10s} {L:5d} {beads:6d} | {b_best*1e3:10.2f} {f_best*1e3:10.2f} "
              f"{c_best*1e3:10.2f} | {per_step/beads*1e6:12.3f} {per_step*1e3:12.2f}")

    print()
    print("(milliseconds per call; us/step/bead is the production step divided by 3L)")
    print()
    # WHAT IT WOULD TAKE: the production chain's own number, and what the flat hypothesis predicts.
    gi = rows[-1]
    sm = rows[0]
    ratio_L = gi[1] / sm[1]
    ratio_cost = gi[6] / sm[6]
    print(f"L ratio {ratio_L:.1f}x, step-cost ratio {ratio_cost:.1f}x, "
          f"per-bead ratio {gi[6]/gi[2] / (sm[6]/sm[2]):.2f}x")
    secs = gi[6] * 32500
    print(f"{gi[0]} at this cost: {secs/3600:.1f} h for 32500 steps "
          f"(production measured 9.2-18.2 h)")
    flat = sm[6] * (gi[2] / sm[2]) * 32500
    print(f"flat-per-bead prediction: {flat/3600:.1f} h -> "
          f"{'the tail is arithmetic' if abs(flat - secs) < 0.25 * secs else 'there is a superlinear term'}")
    if len(rows) > 1:
        mid = rows[1]
        print(f"  per-bead cost us: " + ", ".join(
            f"L={r[1]} {r[6]/r[2]*1e6:.3f}" for r in rows))


if __name__ == "__main__":
    main()
