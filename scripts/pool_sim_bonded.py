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
import cg_potentials as P              # noqa: E402
import ibi_core as IC                  # noqa: E402

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


def _opt(name, default=""):
    """--name=VALUE out of argv as a string, leaving the positional arguments untouched."""
    pre = f"--{name}="
    for a in sys.argv[1:]:
        if a.startswith(pre):
            return a[len(pre):]
    return default


NPZ = REPO / "results" / "boltzmann_tables_clean.npz"
SEED = 20260218
import inspect as _inspect
_cap = _inspect.signature(C.cg_energy_forces).parameters["force_cap"].default

# ── optional substitutes for the two backbone angular terms ──
# Same flags and same construction as ibi_round0.py, because the point of this script is to run
# the SAME field on several chains: a potential that changes one and not the other would make the
# pooled comparison measure the difference between two protocols.
#
# Note this is the flag a POOLED run needs and --set= is not: --set replaces a CONSTANT's value,
# while a potential replaces a term's SHAPE and leaves every constant alone -- so --set cannot
# express it, and the constant fingerprint cannot show it.
_TABLE_PATH = Path(_opt("table", str(NPZ)))
P.use_table_file(_TABLE_PATH)
TAB = IC.load_tables(_TABLE_PATH)

_POTS = []          # [(coord, spec, potential)]
for _coord in ("angle", "dihedral"):
    _text = _opt(_coord, "")
    if _text:
        _spec = P.resolve_spec(_text, _coord)
        _POTS.append((_coord, _spec, P.make_potential(_coord, _spec)))
_POT_KW = {f"{_c}_potential": _pot for _c, _s, _pot in _POTS}

_cap_text = _opt("cap", "auto")
if _cap_text == "auto":
    CAP = float(_cap)
elif _cap_text == "none":
    CAP = None
else:
    CAP = float(_cap_text)

if _POTS:
    print("potentials: " + "  ".join(f"{_c}={P.describe(_s)}" for _c, _s, _ in _POTS)
          + f"  cap={'none' if CAP is None else f'{CAP:g}'}")
    print("            NOTE: the constant fingerprint below does NOT reflect this -- it names "
          "constants,")
    print("            and a potential replaces the SHAPE of a term while leaving every "
          "constant alone.")

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
print(f"force_cap={_cap}  mass=110.0 Da  dt=0.002 ps  friction={FRICTION}/ps  threads={torch.get_num_threads()}")

pos = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64).repeat(NREP, 1, 1)
vel = torch.zeros_like(pos)
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
temps = torch.full((NREP,), 300.0, dtype=torch.float64)

# The loop lives in ibi_core, NOT here. This script used to carry its own copy of it, which is
# what docs/statistical_potentials_as_forces.md:2754 forbids -- this repository has already
# drifted three times from duplicated logic, and a second sampling loop means a change to one
# silently leaves the other behind. The stakes are higher here than usual: the whole point of
# this script is to run the SAME protocol on several chains, so a divergence between two copies
# would be measured as a difference between two FIELDS.
#
# What stays here is the accumulator's use. sum / sumsq / n are the minimal sufficient statistics
# for the pooled variance across structures (the pooled variance needs the grand mean, which
# needs the per-chain sums, so saving sigma alone would be lossy). run_round returns the same
# accumulator it always did, so this stays a one-line binding.
_res = IC.run_round(pos=pos, vel=vel, ij=ij, pw=pw, temps=temps, tab=TAB,
                    nsteps=NSTEPS, burn=burn, stride=STRIDE, blocks=1,
                    friction=FRICTION, force_cap=CAP, pot_kw=_POT_KW, seed=SEED, nrep=NREP)
acc = _res.acc
clash_min = _res.clash_min
clash_below, clash_below_live = _res.clash_below, _res.clash_below_live

el = _res.seconds
print(f"done in {el:.0f} s, {_res.steps_per_s:.1f} steps/s")
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
