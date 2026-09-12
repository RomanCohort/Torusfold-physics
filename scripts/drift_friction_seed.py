"""Does the block-to-block drift depend on friction? Is it reproducible across seeds?

The zero-cost analysis (scripts/analyze_drift_trend.py) showed the drift is NOT independent
random walks: bb_bond and stack move monotonically downward in all five E2 chains, while the
other four coordinates wander per-chain. That is a common-direction signature, but it does not
yet say WHAT is moving slowly. Two clean controls separate the candidates:

  * FRICTION. The Langevin stationary distribution is Boltzmann at every friction, so friction
    sets the relaxation RATE, never the equilibrium. If the block spread (and the per-coordinate
    drift slope) depends on friction, then the drift is a relaxation transient -- the system is
    not equilibrated at 200 ps and the "slow mode" is the time it takes to get there. If it does
    not depend on friction, the drift is not a dynamical transient and the cause is elsewhere.

  * SEED. All eight replicas start from the same deposited structure with zero velocity; the only
    seed dependence is the Langevin noise. If a different seed reproduces the same downward drift
    in bb_bond/stack, the drift is the deterministic relaxation of the ensemble mean (the noise
    has been averaged down by 8 replicas x 400 frames). If it does not, the drift is stochastic.

This does not re-implement the sampler. It runs scripts/ibi_round0.py through runpy with
torch.manual_seed monkeypatched (the same bootstrap as scripts/scan_k_pair.py), one subprocess
per arm, and parses the "=== stationarity ===" block table each arm prints. The friction is a
positional argument of ibi_round0, so no field constant is touched. Each arm gets its own log in
results/, and the reference friction 0.1 / seed 20260218 arm is read from the existing
results/E1_stationarity.log rather than re-run.

Arms run SEQUENTIALLY, each subprocess capped at 4 torch threads, so the run does not
oversubscribe the shared machine (the first version ran three 16-thread subprocesses in
parallel and was killed). A completed arm's log is on disk before the next starts, so an
interruption loses at most the arm in flight, and re-running skips arms already finished.

Run: python scripts/drift_friction_seed.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"

NREP = "8"
NSTEPS = "100000"
IDX = "0"
STRIDE = "25"
BURN = "20000"          # 40 ps, the E1/E2 window
BLOCKS = "8"
DEFAULT_SEED = 20260218
NTHREADS = "4"

COORDS = ("bb_bond", "intra_pc", "intra_cn", "angle", "dihedral", "stack")

# (label, friction, seed) -- run in this order. The 0.1/default arm is E1, already on disk.
ARMS = [
    ("fric5.0_seed20260218", "5.0", DEFAULT_SEED),
    ("fric1.0_seed20260218", "1.0", DEFAULT_SEED),
    ("fric0.1_seed7",        "0.1", 7),
]

BOOT = """
import runpy, sys, torch
torch.set_num_threads(%(nthreads)s)
_real = torch.manual_seed
torch.manual_seed = lambda s, _r=_real, _o=%(seed)r: _r(_o)
sys.path.insert(0, r'%(src)s')
sys.argv = ['ibi_round0.py', '%(nrep)s', '%(nsteps)s', '%(idx)s', '%(fric)s', '%(stride)s',
            '%(burn)s', '--blocks=%(blocks)s']
runpy.run_path(r'%(script)s', run_name='__main__')
"""

ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)-\s*([\d.]+)\s+(\d+)\s+"
                 r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
SPREAD = re.compile(r"block J spread:\s*([\d.]+) to ([\d.]+)\s*\(([\d.]+) percent")


def parse_blocks(text):
    blocks = {c: [] for c in COORDS}
    blocks["J"] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("=== stationarity"):
            in_table = True
            continue
        if not in_table:
            continue
        m = ROW.match(line)
        if not m:
            if blocks["J"]:
                break
            continue
        g = m.groups()
        blocks["J"].append(float(g[4]))
        for i, c in enumerate(COORDS):
            blocks[c].append(float(g[5 + i]))
    return blocks


def slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    return float(((x - xm) * (y - ym)).sum() / denom) if denom else 0.0


def arm_stats(blocks):
    n = len(blocks["J"])
    if n < 3:
        return None
    x = np.arange(n)
    js = np.array(blocks["J"])
    lo, hi = js.min(), js.max()
    rel = (hi - lo) / lo * 100.0 if lo > 0 else float("nan")
    per = {c: slope(x, blocks[c]) for c in COORDS}
    return {"spread": (lo, hi, rel), "J_slope": slope(x, js), "coord_slope": per,
            "J": js.tolist()}


def run_arm(label, fric, seed):
    logpath = RESULTS / f"F_{label}.log"
    if logpath.exists():
        print(f"  {label}: already done, reading {logpath.name}")
        text = logpath.read_text(encoding="utf-8", errors="replace")
        st = arm_stats(parse_blocks(text))
        return st
    boot = BOOT % {"seed": seed, "src": REPO / "src", "nrep": NREP, "nsteps": NSTEPS,
                   "idx": IDX, "fric": fric, "stride": STRIDE, "burn": BURN,
                   "blocks": BLOCKS, "script": REPO / "scripts" / "ibi_round0.py",
                   "nthreads": NTHREADS}
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = NTHREADS
    env["MKL_NUM_THREADS"] = NTHREADS
    env["OPENBLAS_NUM_THREADS"] = NTHREADS
    print(f"  {label}: running (friction {fric}, seed {seed}, {NTHREADS} threads)", flush=True)
    p = subprocess.run([sys.executable, "-c", boot],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=env)
    logpath.write_text(p.stdout + p.stderr, encoding="utf-8")
    if p.returncode != 0:
        print(f"  {label}: FAILED, see results/F_{label}.log")
        print(p.stderr[-1500:])
        return None
    b = parse_blocks(p.stdout)
    st = arm_stats(b)
    if st is None:
        print(f"  {label}: no block table parsed")
        return None
    return st


def main():
    results = {}
    ref_path = RESULTS / "E1_stationarity.log"
    if ref_path.exists():
        b = parse_blocks(ref_path.read_text(encoding="utf-8", errors="replace"))
        results["fric0.1_seed20260218"] = arm_stats(b)

    for label, fric, seed in ARMS:
        st = run_arm(label, fric, seed)
        if st is None:
            continue
        results[label] = st
        print(f"  {label}: block J spread {st['spread'][2]:.1f}% ({st['spread'][0]:.4f} to "
              f"{st['spread'][1]:.4f});  J slope {st['J_slope']:+.4f}")
        print(f"      coord slopes (sim/ref per block): "
              + "  ".join(f"{c}={st['coord_slope'][c]:+.4f}" for c in COORDS), flush=True)

    print()
    print("=" * 96)
    print("summary: friction/seed controls for the block-to-block drift "
          f"({NSTEPS} steps, burn {BURN}, {BLOCKS} blocks, idx {IDX})")
    print("=" * 96)
    order = ["fric0.1_seed20260218", "fric1.0_seed20260218", "fric5.0_seed20260218",
             "fric0.1_seed7"]
    print(f"{'arm':>24s}  {'spread %':>9s} {'J slope':>8s}  " +
          "  ".join(f"{c:>8s}" for c in COORDS))
    print("-" * 96)
    for label in order:
        if label not in results or results[label] is None:
            continue
        st = results[label]
        print(f"{label:>24s}  {st['spread'][2]:9.1f} {st['J_slope']:+8.4f}  " +
              "  ".join(f"{st['coord_slope'][c]:+8.4f}" for c in COORDS))
    print()
    print("Read the spread % column against friction: if it moves, the drift is a relaxation")
    print("transient (friction-dependent rate). Read the coord-slope columns against seed: if the")
    print("fric0.1_seed7 row reproduces the 20260218 row's signs, the drift is deterministic.")


if __name__ == "__main__":
    main()
