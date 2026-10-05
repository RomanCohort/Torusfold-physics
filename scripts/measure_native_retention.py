"""Does the field keep a chain near the geometry it was started from?

THE QUESTION PLAN C CANNOT ANSWER WITHOUT THIS. Native retention is C's acceptance test -- "a chain
started from its deposited geometry stays near it" -- and none of the existing artifacts can decide
it: the task results carry histograms, moments, joint_J and the entry state, but NOT coordinates.
Measured before building this (docs/archive/plan_c_selfconsistent_target.md, first free check): the entry
state does not predict the residual (Spearman(J, E0) = +0.04, started-at-cap 0.1746 against
below-cap 0.1735), so "which chains fail to stay" is not readable off the diagnostics. It needs
geometry.

WHAT IT MEASURES, per chain, all after a Kabsch alignment:

  rmsd_dep_relaxed    deposited -> the relaxed start (what the 5000-step relaxation moved)
  rmsd_dep_mean       deposited -> the MEAN of the sampled frames (where the chain lives now)
  rmsd_dep_frame_mean deposited -> each sampled frame, averaged (the ensemble's own spread)
  rmsd_dep_frame_max  the worst sampled frame
  J, entry, relax diagnostics, closest bead approach -- so retention can be correlated with the
                      numbers the loop already reports.

A chain that "stays" has rmsd_dep_mean of a few Angstrom and a small spread; one that melts has
tens of Angstrom and a spread to match. Both are printed, and the threshold is not hard-coded: the
distribution is the result.

CHAINS ARE LENGTH-STRATIFIED AND CAPPED. The pool runs to 2929 residues; the instrument takes
quantiles of the length distribution up to MAX_L because a 2929-residue chain would spend an hour
per arm on its own, and the point here is the shape of the distribution, not its tail. Say so in
the output rather than silently.

Run: python scripts/measure_native_retention.py [n_chains] [nsteps] [relax_steps] [max_L]
     NATIVE_RETENTION_SERIAL=1 ...   # no process pool (a sandbox that denies pipes)
"""
import glob
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import boltzmann_bonded as B              # noqa: E402
import cg_potentials as P                 # noqa: E402
import ibi_core as IC                     # noqa: E402
import torusfold.scheme2.torch_cgsim as C  # noqa: E402

# THE CAMPAIGN RECORD MOVED (2026-10-01) and this path used to be hard-coded to it; the resolver in
# ibi_core knows the candidates. `TABLES` can point the instrument at any table file directly, which is
# what the base-level comparison below does: it runs the same protocol with and without the stacking
# term, so both arms must be on the SAME tables.
RUN_DIR = REPO / "results" / "ibi_relax"       # kept for the historical default; resolved at run time
TABLES_OVERRIDE = os.environ.get("NATIVE_RETENTION_TABLES", "")
# The base-level term, if the caller wants it: "eps" or "eps:w_d,w_r,w_t". Shared names with ibi_loop's
# IBI_LOOP_BASE_STACK so the loop and this instrument cannot be running different fields.
BASE_STACK_SPEC = os.environ.get("NATIVE_RETENTION_BASE_STACK", "")


def base_stack_kwargs():
    """The base_stacking parameters from NATIVE_RETENTION_BASE_STACK, or None."""
    if not BASE_STACK_SPEC:
        return None
    parts = BASE_STACK_SPEC.split(":")
    w = [float(x) for x in parts[1].split(",")] if len(parts) > 1 else [1.0, 1.0, 1.0]
    return {"eps": float(parts[0]), "form": os.environ.get("NATIVE_RETENTION_BASE_STACK_FORM", "sum"),
            "w_d": w[0], "w_r": w[1], "w_t": w[2]}
OUT_DIR = REPO / "results" / "native_retention"
WALL_K = 2000.0
FRICTION = 1.0
FORCE_CAP = 5000.0


def newest_tables():
    """The field as it stands: NATIVE_RETENTION_TABLES if given, else the newest tables_r<N>.npz.

    The campaign record was moved out of results/ on 2026-10-01 (see ibi_core.campaign_root), so the
    directory is resolved rather than hard-coded -- and a caller comparing two fields wants to name the
    exact file instead, which is what the base-level comparison does.
    """
    if TABLES_OVERRIDE:
        if not Path(TABLES_OVERRIDE).exists():
            raise SystemExit(f"NATIVE_RETENTION_TABLES={TABLES_OVERRIDE} does not exist")
        return TABLES_OVERRIDE
    global RUN_DIR
    try:
        RUN_DIR = IC.campaign_root()
    except FileNotFoundError:
        pass
    found = []
    for p in glob.glob(str(RUN_DIR / "tables_r*.npz")):
        m = re.search(r"tables_r(\d+)\.npz$", p)
        if m:
            found.append((int(m.group(1)), p))
    if not found:
        raise SystemExit(f"no tables_r*.npz under {RUN_DIR}")
    return max(found)[1]


def kabsch_rmsd(a, b):
    """RMSD between two (N, 3) coordinate sets after optimal superposition."""
    a = np.asarray(a, dtype=np.float64) - np.asarray(a, dtype=np.float64).mean(0)
    b = np.asarray(b, dtype=np.float64) - np.asarray(b, dtype=np.float64).mean(0)
    u, _s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt(((a @ rot.T - b) ** 2).sum(axis=1).mean()))


def _one(task):
    idx, name, L, pos_np, pairs, tables, nsteps, relax, seed = task
    torch.set_num_threads(1)
    t0 = time.time()
    # ONE PLACE builds the field, so this instrument and the loop cannot disagree about which one ran.
    _pots, pot_kw = P.build_potential_kwargs(str(tables), wall_k=WALL_K,
                                             coords=("bb_bond", "angle", "dihedral"),
                                             base_stack=base_stack_kwargs())
    tab = IC.load_tables(str(tables))

    dep = np.asarray(pos_np, dtype=np.float64).reshape(-1, 3)
    ij = torch.tensor(np.asarray(pairs, dtype=np.int64).reshape(-1, 2))
    pos0 = torch.tensor(np.asarray(pos_np, dtype=np.float64).reshape(1, 3 * L, 3))
    res = IC.run_round(pos=pos0, vel=torch.zeros_like(pos0), ij=ij,
                       pw=torch.ones(len(ij), dtype=torch.float32),
                       temps=torch.full((1,), 300.0, dtype=torch.float64), tab=tab,
                       nsteps=nsteps, burn=0, stride=25, blocks=4, friction=FRICTION,
                       force_cap=FORCE_CAP, pot_kw=pot_kw, seed=seed, nrep=1, progress=False,
                       constraints=C.make_intra_constraints(L), relax=relax,
                       collect_positions=True, log=lambda *a, **k: None)

    f0 = dep
    fR = torch.stack([p[0].reshape(-1, 3) for p in [res.pos]]).numpy() if False else None
    # the relaxed start is the first sampled frame's predecessor: recover it from res.pos only if
    # the caller asked for relax; otherwise the deposited geometry IS the start.
    frames = res.positions.numpy()                      # (F, 1, 3L, 3)
    flat = frames[:, 0].reshape(frames.shape[0], -1, 3)
    mean = flat.mean(axis=0)
    vals, j = IC.simref(res.acc, tab, skip=res.skip)
    return {
        "idx": idx, "name": name, "L": L,
        "rmsd_dep_mean": kabsch_rmsd(dep, mean),
        "rmsd_dep_frame_mean": float(np.mean([kabsch_rmsd(dep, f) for f in flat])),
        "rmsd_dep_frame_max": float(np.max([kabsch_rmsd(dep, f) for f in flat])),
        "rmsd_frame_spread": float(np.mean([kabsch_rmsd(mean, f) for f in flat])),
        "J": None if j != j else float(j),
        "entry": res.relax["energy_start"] if res.relax else None,
        "relax_accepted": res.relax["accepted"] if res.relax else 0,
        "relax_end": res.relax["energy_end"] if res.relax else None,
        "clash_min": min(res.clash_min) if res.clash_min else float("nan"),
        "frames": int(flat.shape[0]), "seconds": time.time() - t0,
    }


def main():
    n_chains = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    relax = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    max_l = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    serial = os.environ.get("NATIVE_RETENTION_SERIAL", "") == "1"
    tables = newest_tables()
    structs = B.load_structures(limit=5000)
    cand = [(i, s) for i, s in enumerate(structs) if len(s["pos"]) <= max_l]
    if len(cand) < n_chains:
        raise SystemExit(f"only {len(cand)} chains at L <= {max_l}, asked for {n_chains}")
    # length-stratified: quantiles of the length distribution up to the cap
    cand.sort(key=lambda t: len(t[1]["pos"]))
    picks = [cand[min(len(cand) - 1, int((k + 0.5) * len(cand) / n_chains))] for k in range(n_chains)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"tables     {tables}")
    print(f"{n_chains} length-stratified chains, L <= {max_l}, {nsteps} steps "
          f"({nsteps * 0.002:.0f} ps), relax {relax}, friction {FRICTION}")
    tasks = [(i, s["name"], len(s["pos"]), np.asarray(s["pos"], dtype=np.float64), s["pairs"],
              tables, nsteps, relax, 4242 + i) for i, s in picks]
    if serial:
        rows = [_one(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(processes=min(len(tasks), os.cpu_count() or 1)) as procs:
            rows = list(procs.imap_unordered(_one, tasks))
    rows.sort(key=lambda r: r["L"])
    print()
    print(f"{'chain':>10} {'L':>5} {'dep->mean':>10} {'dep->frame':>11} {'frame max':>10} "
          f"{'spread':>8} {'J':>7} {'clash':>7} {'relax acc':>9} {'s':>5}")
    for r in rows:
        print(f"{r['name']:>10} {r['L']:>5} {r['rmsd_dep_mean']:>10.2f} {r['rmsd_dep_frame_mean']:>11.2f} "
              f"{r['rmsd_dep_frame_max']:>10.2f} {r['rmsd_frame_spread']:>8.2f} "
              f"{(r['J'] if r['J'] is not None else float('nan')):>7.3f} {r['clash_min']:>7.3f} "
              f"{r['relax_accepted']:>9} {r['seconds']:>5.0f}")
    moved = [r for r in rows if r["rmsd_dep_mean"] > 10.0]
    print()
    print(f"chains whose mean moved more than 10 A: {len(moved)} of {len(rows)}"
          + (": " + ", ".join(f"{r['name']}({r['rmsd_dep_mean']:.0f} A)" for r in moved) if moved else ""))
    print(f"median dep->mean {np.median([r['rmsd_dep_mean'] for r in rows]):.2f} A   "
          f"median spread {np.median([r['rmsd_frame_spread'] for r in rows]):.2f} A")
    out = OUT_DIR / f"retention_{Path(tables).stem}.json"
    out.write_text(json.dumps({"tables": str(tables), "nsteps": nsteps, "relax": relax,
                               "max_L": max_l, "rows": rows}, indent=1, default=float),
                   encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
