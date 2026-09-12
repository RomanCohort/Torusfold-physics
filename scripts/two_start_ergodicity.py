"""Two-start ergodicity test: do two conformations of ONE chain relax to the same marginals?

The drift attribution established that the block-to-block drift is a deterministic relaxation of
the ensemble mean from the deposited (native) start -- not a shifted equilibrium. The next
question that follows is: relax TO WHERE? If two different starting conformations of the same
chain do not reach the same terminal marginals, the sampler is not ergodic on this time scale,
and IBI's input is start-dependent -- which is more fundamental than "can REMD accelerate it".

This script runs the SECOND start. The first start (the native deposited structure) is already on
disk: results/F_noBSJ_idx0.log (ibi_round0 with K_BSJ=K_BSJ_GUIDE=K_BSJ_CONTACT=0, seed
20260218, 8 replicas x 100000 steps, burn 20000, blocks 8, 1L2X). Both starts must use the same
field and the same scale, and this one does.

Field: the BSJ trio is ZERO throughout, for both the anneal and the production run. 1L2X is a
LINEAR deposited chain and the three BSJ terms act on P(0)-P(L-1), which a linear chain does not
have; with them on, the chain is compacted by a 1263-4159 kJ/mol/nm force (attribute_compaction
measurement). Measuring an ergodicity/relaxation endpoint under the wrong field would measure the
artifact, not the sampler.

Start B is built by HIGH-TEMPERATURE annealing the native structure in the same field (BSJ=0):
8 replicas at 600 K for ANNEAL_STEPS, then one replica's final conformation is taken as start B
and replicated x8 for the 300 K production run. Annealing changes the INTERNAL coordinates (a
rigid-body transform would not -- bond/angle/dihedral/stack are translation/rotation invariant,
so a rotated native start is the SAME start for every marginal this test scores).

The production run re-uses the exact ibi_round0.py sampling loop (same symplectic integrator,
same seed 20260218, same 8x100000 / burn 20000 / stride 25 / blocks 8), so its block table is
directly comparable to results/F_noBSJ_idx0.log. The only difference from that run is the initial
positions.

Run: python scripts/two_start_ergodicity.py [n_anneal] [anneal_temp]
"""
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

NPZ = REPO / "results" / "boltzmann_tables_clean.npz"
SEED = 20260218
NREP = 8
NSTEPS = 100000
IDX = 0
FRICTION = 0.1
STRIDE = 25
BURN = 20000
NB = 8
# Positional args are (n_anneal, anneal_temp); flags may appear anywhere, so filter them out
# before reading positionally -- `--anneal-only` would otherwise land in argv[1] and int() it.
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
ANNEAL_ONLY = "--anneal-only" in sys.argv
ANNEAL_STEPS = int(_args[0]) if len(_args) > 0 else 20000
ANNEAL_TEMP = float(_args[1]) if len(_args) > 1 else 600.0

# The correct field for a linear chain: the three BSJ terms are zero.
C.K_BSJ = 0.0
C.K_BSJ_GUIDE = 0.0
C.K_BSJ_CONTACT = 0.0

z = np.load(NPZ)
TAB = {}
for name in B.COORDS:
    TAB[name] = {"centre": z[f"{name}__centre"], "U": z[f"{name}__U"],
                 "binw": float(z[f"{name}__binw"]), "sigma": float(z[f"{name}__sigma"]),
                 "lo": float(z[f"{name}__lo"]), "hi": float(z[f"{name}__hi"])}

pool = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s0 = pool[IDX]
L = len(s0["pos"])
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
native = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
CKPT = REPO / "results" / "two_start_B_anneal.npz"

print(f"structure {s0['name']}  L={L}  pairs={len(ij)}")
print(f"field: K_BSJ=0, K_BSJ_GUIDE=0, K_BSJ_CONTACT=0 (linear chain, per README)")
print(f"anneal: {ANNEAL_STEPS} steps at {ANNEAL_TEMP:.0f} K, friction {FRICTION}/ps, "
      f"then quench to 300 K")
print(f"production: {NREP} x {NSTEPS} steps, friction {FRICTION}/ps, seed {SEED}, "
      f"burn {BURN}, blocks {NB} -- same protocol as results/F_noBSJ_idx0.log")
print()


def _forces_at(p):
    cl2 = C.GPUCellList(cell_size=1.5)
    cl2.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl2)[1]


def rg(p):
    c = p.mean(dim=1, keepdim=True)
    return torch.linalg.norm(p - c, dim=-1).pow(2).mean(dim=1).sqrt()


# ---- anneal: produce start B from native under the same (BSJ=0) field ----
# The anneal is deterministic (seed SEED), so a saved checkpoint is only a time-saver: rerunning
# from scratch gives the identical start B. If the production run is killed, the anneal need not
# be redone. The checkpoint records (L, steps, temp, seed) so a mismatched call re-anneals.
startB = None
if CKPT.exists():
    ck = np.load(CKPT, allow_pickle=False)
    if (int(ck["L"]) == L and int(ck["anneal_steps"]) == ANNEAL_STEPS
            and float(ck["anneal_temp"]) == ANNEAL_TEMP and int(ck["seed"]) == SEED):
        startB = torch.tensor(ck["startB"], dtype=torch.float64)
        print(f"loaded anneal checkpoint {CKPT.name} "
              f"(anneal {ANNEAL_STEPS} @ {ANNEAL_TEMP:.0f} K, seed {SEED} already done)", flush=True)
    else:
        print(f"checkpoint {CKPT.name} params differ from this call; re-annealing", flush=True)

if startB is None:
    pos = native.repeat(NREP, 1, 1)
    vel = torch.zeros_like(pos)
    temps = torch.full((NREP,), ANNEAL_TEMP, dtype=torch.float64)
    torch.manual_seed(SEED)
    t0 = time.time()
    for step in range(ANNEAL_STEPS):
        with torch.no_grad():
            cl = C.GPUCellList(cell_size=1.5)
            cl.build(pos)
            _e, f = C.cg_energy_forces(pos, ij, pw, cell_list=cl)
            pos, vel = C.batch_langevin_step(pos, vel, f, temps,
                                             dt_ps=0.002, mass_amu=110.0,
                                             friction=FRICTION, force_fn=_forces_at)
        if (step + 1) % 5000 == 0:
            print(f"  anneal {step + 1}/{ANNEAL_STEPS} steps ({time.time() - t0:.0f} s)", flush=True)
    startB = pos[0:1].clone()   # one replica's annealed conformation, replicated x8 below
    np.savez(CKPT, startB=startB.numpy(), L=L, anneal_steps=ANNEAL_STEPS,
             anneal_temp=ANNEAL_TEMP, seed=SEED)
    print(f"anneal done in {time.time() - t0:.0f} s; start B saved to {CKPT.name}", flush=True)

rg_native = float(rg(native)[0])
rg_B = float(rg(startB)[0])
disp = float(torch.linalg.norm(startB[0] - native[0], dim=-1).mean())
print(f"native Rg {rg_native:.3f} nm -> start B Rg {rg_B:.3f} nm "
      f"({rg_B - rg_native:+.3f}); mean per-bead displacement {disp:.3f} nm", flush=True)
print("(Rg and the raw displacement are orientation-dependent and are NOT the test: bond, angle, "
      "dihedral and stack are translation/rotation invariant, so a rotated native start is the "
      "same start for every marginal scored below. What decides whether start B is a second "
      "conformation is this:)", flush=True)
print()

# The six internal coordinates are what the two-start comparison ultimately scores, so they are
# what "a second start" has to mean. A start B that has not moved them IS the native start, and
# the comparison would then report "the two starts agree" for a reason that has nothing to do
# with ergodicity -- the two runs would be measuring the same trajectory from the same place.
print("internal coordinates, native -> start B, shift in units of the reference sigma:")
print(f"{'coordinate':10s} {'d mean':>9s} {'d std':>9s} {'sigma':>9s}  moved?")
print("-" * 50)
_n_moved = 0
for c in B.COORDS:
    qa = B.coords_of(native, c).reshape(-1).numpy().astype(np.float64)
    qb = B.coords_of(startB, c).reshape(-1).numpy().astype(np.float64)
    sig = TAB[c]["sigma"]
    d_mean = (qb.mean() - qa.mean()) / sig
    d_std = (qb.std() - qa.std()) / sig
    # "moved" = mean or width shifted by more than a tenth of the reference sigma. That is an
    # order-of-magnitude flag, not a test: the per-block tables in this repository move by 2-10
    # percent of sigma between windows, so a tenth sits at the top of that band. The per-
    # coordinate numbers are printed so that the verdict does not rest on the threshold alone --
    # no single scalar in this table can tell a second conformation from a long soak.
    moved = abs(d_mean) > 0.1 or abs(d_std) > 0.1
    _n_moved += int(moved)
    print(f"{c:10s} {d_mean:+9.3f} {d_std:+9.3f} {sig:9.4f}  {'yes' if moved else 'no'}")
print(f"  {_n_moved}/{len(B.COORDS)} coordinates moved by more than 0.1 sigma")
if _n_moved == 0:
    print("  WARNING: start B is not measurably distinct from native in any scored coordinate. "
          "The production run would compare a start against itself; raise the anneal length or "
          "temperature until this table moves.", flush=True)
print()

if ANNEAL_ONLY:
    print("--anneal-only: stopping here. Re-run the same command without the flag to resume from "
          "the checkpoint and run production.", flush=True)
    sys.exit(0)

# ---- production run from start B, identical to ibi_round0 with BSJ=0 ----
pos = startB.repeat(NREP, 1, 1)
vel = torch.zeros_like(pos)
temps = torch.full((NREP,), 300.0, dtype=torch.float64)
torch.manual_seed(SEED)   # same noise sequence as the start-A run; only the start differs

b_counts = {b: {c: np.zeros(len(TAB[c]["U"]), dtype=np.int64) for c in B.COORDS}
            for b in range(NB)}
b_acc = {b: {c: [0.0, 0.0, 0] for c in B.COORDS} for b in range(NB)}
b_frames = [0] * NB

# running cumulative accumulator, printed every 10000 steps, so a killed run still leaves its
# partial cumulative sim/ref up to the last checkpoint (the same reason ibi_round0 prints its
# progress table). With python -u these land in the log as they happen.
run_counts = {c: np.zeros(len(TAB[c]["U"]), dtype=np.int64) for c in B.COORDS}
run_acc = {c: [0.0, 0.0, 0] for c in B.COORDS}


def _simref(acc, c):
    s1, s2, n = acc[c]
    if not n:
        return float("nan")
    mm = s1 / n
    ss = float(np.sqrt(max(s2 / n - mm * mm, 0.0)))
    return ss / TAB[c]["sigma"] if TAB[c]["sigma"] > 0 else float("nan")


t0 = time.time()
for step in range(NSTEPS):
    with torch.no_grad():
        cl = C.GPUCellList(cell_size=1.5)
        cl.build(pos)
        e, f = C.cg_energy_forces(pos, ij, pw, cell_list=cl)
        pos, vel = C.batch_langevin_step(pos, vel, f, temps,
                                         dt_ps=0.002, mass_amu=110.0,
                                         friction=FRICTION, force_fn=_forces_at)
    if step >= BURN and step % STRIDE == 0:
        blk = min(((step - BURN) * NB) // max(NSTEPS - BURN, 1), NB - 1)
        b_frames[blk] += 1
        with torch.no_grad():
            for c in B.COORDS:
                q = B.coords_of(pos, c).reshape(-1).numpy().astype(np.float64)
                a = b_acc[blk][c]
                a[0] += q.sum()
                a[1] += (q ** 2).sum()
                a[2] += q.size
                ra = run_acc[c]
                ra[0] += q.sum()
                ra[1] += (q ** 2).sum()
                ra[2] += q.size
                t = TAB[c]
                k = np.round((q - t["centre"][0]) / t["binw"]).astype(np.int64)
                ok = (k >= 0) & (k < len(t["U"]))
                bc = np.bincount(k[ok], minlength=len(t["U"]))
                b_counts[blk][c] += bc
                run_counts[c] += bc
    if step % 10000 == 0:
        vals = [_simref(run_acc, c) for c in B.COORDS]
        j = float(np.mean([abs(float(np.log(v))) for v in vals if v == v and v > 0])) \
            if any(v == v and v > 0 for v in vals) else float("nan")
        print(f"  {step:6d}  " + "  ".join(f"{v:6.3f}" for v in vals)
              + f"   J {j:.4f}  ({time.time() - t0:.0f} s)", flush=True)

print(f"production done in {time.time() - t0:.0f} s, {NSTEPS / (time.time() - t0):.1f} steps/s",
      flush=True)
print()

# whole-window sim/ref and the block table, same format as ibi_round0
counts = {c: np.zeros(len(TAB[c]["U"]), dtype=np.int64) for c in B.COORDS}
acc = {c: [0.0, 0.0, 0] for c in B.COORDS}
for b in range(NB):
    for c in B.COORDS:
        counts[c] += b_counts[b][c]
        a = b_acc[b][c]
        acc[c][0] += a[0]
        acc[c][1] += a[1]
        acc[c][2] += a[2]

print("start B terminal sim/ref:")
print(f"{'coordinate':10s} {'sim/ref':>8s}")
print("-" * 20)
for c in B.COORDS:
    s1, s2, n = acc[c]
    mm = s1 / n
    ss = float(np.sqrt(max(s2 / n - mm * mm, 0.0)))
    r = ss / TAB[c]["sigma"] if TAB[c]["sigma"] > 0 else float("nan")
    print(f"{c:10s} {r:8.3f}")

print()
print(f"=== stationarity: {NB} disjoint blocks of the sampling window (start B) ===")
print(f"{'block':>5s} {'window (ps)':>15s} {'frames':>7s} {'J':>7s}  "
      + "  ".join(f"{c[:5]:>5s}" for c in B.COORDS))
print("-" * (37 + 7 * len(B.COORDS)))
_Jb = []
for b in range(NB):
    lo = BURN + (NSTEPS - BURN) * b // NB
    hi = BURN + (NSTEPS - BURN) * (b + 1) // NB
    parts, js = [], []
    for c in B.COORDS:
        s1, s2, n = b_acc[b][c]
        if n:
            mm = s1 / n
            ss = float(np.sqrt(max(s2 / n - mm * mm, 0.0)))
            r = ss / TAB[c]["sigma"] if TAB[c]["sigma"] > 0 else float("nan")
            parts.append(f"{r:5.3f}")
            if r == r and r > 0:
                js.append(abs(float(np.log(r))))
        else:
            parts.append("   --")
    j = float(np.mean(js)) if js else float("nan")
    _Jb.append(j)
    print(f"{b + 1:5d} {lo * 0.002:6.1f}-{hi * 0.002:6.1f} {b_frames[b]:7d} {j:7.4f}  "
          + "  ".join(parts))
_lo, _hi = min(_Jb), max(_Jb)
print(f"  block J spread: {_lo:.4f} to {_hi:.4f}  ({(_hi - _lo) / _lo * 100:.1f} percent)")
