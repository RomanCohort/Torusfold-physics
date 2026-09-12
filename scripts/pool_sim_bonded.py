"""Sample ONE chain's six bonded coordinates and SAVE the per-coordinate moments.

ibi_round0.py answers "does the shipped field reproduce the reference marginal?" but throws
away the samples: it prints sigma and sim/ref, and the per-chain first/second moments needed to
pool several chains into one like-for-like statistic against the pooled reference are gone once
the process exits. This script runs the SAME trajectory protocol -- same seed, same symplectic
Langevin integrator with the recomputed tail kick, same force cap, same 40-200 ps window -- and
additionally writes the per-coordinate (sum, sum-of-squares, count) to an npz keyed by config
and structure, so a pooled comparison can be built afterwards without re-sampling.

It does not print the pooled comparison. That lives in pool_sim_report.py, kept separate so the
expensive sampling can be farmed across chains (parallel processes) and the cheap arithmetic is
done once everything has landed.

The optional --set=K_NAME=VALUE,... patch is applied to the module globals of torch_cgsim BEFORE
the run and echoed to the log, exactly as ibi_patched_field.py does, so a patched run (the
README-conformant linear-chain field: BSJ trio and K_BPP zero) and an unpatched run share one
code path and one provenance format.

Run: python scripts/pool_sim_bonded.py <idx> <tag> [n_rep] [n_steps] [friction] [stride] [burn]
                                          [--set=K_BSJ=0,K_BSJ_GUIDE=0,K_BSJ_CONTACT=0,K_BPP=0]

The moments land in results/pooled_sim_<tag>_idx<idx>.npz.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

THREADS = int(os.environ.get("POOL_THREADS", "0") or 0)
if THREADS > 0:
    torch.set_num_threads(THREADS)

# ---- patch any --set=K=V,K=V before anything reads the constants ---------------------------
_set = {}
_positional = []
for a in sys.argv[1:]:
    if a.startswith("--set="):
        for kv in a[len("--set="):].split(","):
            kv = kv.strip()
            if not kv:
                continue
            name, _, val = kv.partition("=")
            _set[name.strip()] = float(val)
    else:
        _positional.append(a)
for _n, _v in _set.items():
    if not hasattr(C, _n):
        raise SystemExit(f"no such constant in torch_cgsim: {_n!r}; refusing to run, because a "
                         f"silently ignored name would produce a run of the unpatched field")
    setattr(C, _n, _v)
if _set:
    print("PATCHED FIELD: " + (", ".join(f"{k}={v:g}" for k, v in _set.items())))
    print("Every constant below is read after this patch, so the fingerprint reflects it.")

# ---- positional args, same defaults and meanings as ibi_round0.py --------------------------
IDX = int(_positional[0]) if len(_positional) > 0 else 0
TAG = _positional[1] if len(_positional) > 1 else "full"
NREP = int(_positional[2]) if len(_positional) > 2 else 8
NSTEPS = int(_positional[3]) if len(_positional) > 3 else 100000
FRICTION = float(_positional[4]) if len(_positional) > 4 else 0.1
STRIDE = int(_positional[5]) if len(_positional) > 5 else 25
_burn_arg = int(_positional[6]) if len(_positional) > 6 else 0
burn = _burn_arg if _burn_arg > 0 else max(NSTEPS // 5, 1)

NPZ = REPO / "results" / "boltzmann_tables_clean.npz"
SEED = 20260218

z = np.load(NPZ)
TAB = {}
for name in B.COORDS:
    TAB[name] = {"centre": z[f"{name}__centre"], "U": z[f"{name}__U"],
                 "binw": float(z[f"{name}__binw"]), "sigma": float(z[f"{name}__sigma"]),
                 "lo": float(z[f"{name}__lo"]), "hi": float(z[f"{name}__hi"])}

pool = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s0 = pool[IDX]
L = len(s0["pos"])
print(f"structure {s0['name']}  L={L}  pairs={len(s0['pairs'])}  tag={TAG}")
print(f"{NREP} replicas, {NSTEPS} steps of 0.002 ps = {NSTEPS * 0.002:.1f} ps per replica")
print(f"300 K, mass 110 Da, friction {FRICTION}/ps, sampling every {STRIDE} steps, "
      f"burn {burn * 0.002:.1f} ps")
_FINGERPRINT = ("K_BB", "K_INTRA_PC", "K_INTRA_CN", "K_INTRA_PN",
                "K_LINK_CP", "K_LINK_NP", "K_LINK_NC",
                "K_PAIR", "K_ANGLE", "K_DIH", "K_BPP", "K_STACK",
                "K_CLASH", "CLASH_SIGMA", "K_BSJ", "K_BSJ_GUIDE")
print("field: " + "  ".join(f"{n}={getattr(C, n)}" for n in _FINGERPRINT))
import inspect as _inspect
_cap = _inspect.signature(C.cg_energy_forces).parameters["force_cap"].default
print(f"force_cap={_cap}  mass=110.0 Da  dt=0.002 ps  friction={FRICTION}/ps  threads={torch.get_num_threads()}")

pos = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64).repeat(NREP, 1, 1)
vel = torch.zeros_like(pos)
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
temps = torch.full((NREP,), 300.0, dtype=torch.float64)

torch.manual_seed(SEED)

# Whole-window per-coordinate moments. sum / sumsq / n are the minimal sufficient statistics
# for the pooled variance across structures (pooled var needs the grand mean, which needs the
# per-chain sums, so saving just sigma would be lossy).
acc = {c: [0.0, 0.0, 0] for c in B.COORDS}

clash_min = []
clash_below = 0
clash_below_live = 0


def _forces_at(p):
    cl2 = C.GPUCellList(cell_size=1.5)
    cl2.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl2)[1]


t0 = time.time()
for step in range(NSTEPS):
    with torch.no_grad():
        cl = C.GPUCellList(cell_size=1.5)
        cl.build(pos)
        e, f = C.cg_energy_forces(pos, ij, pw, cell_list=cl)
        pos, vel = C.batch_langevin_step(pos, vel, f, temps,
                                         dt_ps=0.002, mass_amu=110.0,
                                         friction=FRICTION, force_fn=_forces_at)
    if step == 0:
        print(f"first step {time.time() - t0:.3f} s")
    if step >= burn and step % STRIDE == 0:
        with torch.no_grad():
            for c in B.COORDS:
                q = B.coords_of(pos, c).reshape(-1).numpy().astype(np.float64)
                a = acc[c]
                a[0] += q.sum()
                a[1] += (q ** 2).sum()
                a[2] += q.size
            beads = pos.reshape(NREP, -1, 3)
            dd = torch.cdist(beads, beads)
            dd = dd + torch.eye(dd.shape[-1], device=dd.device) * 10.0
            clash_min.append(float(dd.min()))
            clash_below += int((dd < C.CLASH_DIST).sum())
            clash_below_live += int((dd < C.CLASH_SIGMA).sum())
    if (step + 1) % max(NSTEPS // 10, 1) == 0:
        el = time.time() - t0
        parts = []
        for c in B.COORDS:
            s1, s2, n = acc[c]
            if n:
                mm = s1 / n
                ss = float(np.sqrt(max(s2 / n - mm * mm, 0.0)))
                parts.append(f"{c[:5]} {ss / TAB[c]['sigma']:5.3f}")
            else:
                parts.append(f"{c[:5]}   --")
        print(f"  {step + 1:>7d} {el:6.0f}s  " + "  ".join(parts))

el = time.time() - t0
print(f"done in {el:.0f} s, {NSTEPS / el:.1f} steps/s")
print(f"clash watch: closest bead pair ever {min(clash_min):.4f} nm; "
      f"below live {C.CLASH_SIGMA:.4f}: {clash_below_live}; below old 0.300: {clash_below}")

# ---- save moments + provenance ------------------------------------------------------------
out = REPO / "results" / f"pooled_sim_{TAG}_idx{IDX}.npz"
moments = {}
for c in B.COORDS:
    moments[f"{c}__sum"] = np.asarray(acc[c][0])
    moments[f"{c}__sumsq"] = np.asarray(acc[c][1])
    moments[f"{c}__n"] = np.asarray(acc[c][2], dtype=np.int64)
meta = {"name": s0["name"], "L": L, "pairs": len(s0["pairs"]), "tag": TAG, "idx": IDX,
        "n_rep": NREP, "n_steps": NSTEPS, "friction": FRICTION, "stride": STRIDE,
        "burn": burn, "seed": SEED,
        "force_cap": float(_cap),
        "patch": ",".join(f"{k}={v:g}" for k, v in _set.items()) or "(none)"}
np.savez(out, **moments, **{f"meta__{k}": np.asarray(v) for k, v in meta.items()})

# ---- self-contained per-coordinate table, matching ibi_round0's final block -----------------
print()
print(f"{'coordinate':10s} {'ref sig':>8s} {'sim sig':>8s} {'sim/ref':>8s} {'n':>8s}")
print("-" * 44)
for c in B.COORDS:
    s1, s2, n = acc[c]
    m = s1 / n
    sig = float(np.sqrt(max(s2 / n - m * m, 0.0)))
    print(f"{c:10s} {TAB[c]['sigma']:8.4f} {sig:8.4f} {sig / TAB[c]['sigma']:8.3f} {n:8d}")
print(f"\nsaved {out}")
