"""IBI round 0: does the shipped full field reproduce the reference bonded marginals?

scripts/sample_bonded_chain.py removed the nonbonded terms, and the chain unfolded -- correctly,
since a bonded-only chain has no reason to stay folded. But that made the acceptance test fail
for a reason that has nothing to do with the potentials, and it left the actual question
unanswered.

The variance decomposition says only 2.6 to 10.5 percent of the pooled sigma is between
structures; the rest is residue-to-residue spread inside each chain. So the pooled reference
is close to a single chain's own distribution, and a single chain sampled with the FULL field
is a fair stand-in. That is the run this script does.

What it measures is the IBI residual, before any update:

    dU(q) = kBT * ln( P_sim(q) / P_ref(q) )

If the shipped field already matches the reference, dU is flat and IBI has nothing to do. If it
is not flat, this is the correction IBI would apply on its first round, and its size says
whether the rest of the loop is worth running.

The field is used exactly as shipped, including its force cap (force_cap's default, read from the
signature below rather than written here), because that is what the pipeline runs.

Run: python scripts/ibi_round0.py [n_rep] [n_steps] [struct_idx] [friction] [stride] [burn]

The BURN argument is separate from NSTEPS on purpose. It used to be NSTEPS // 5, so asking for a
longer run moved the sampling window as well as lengthening it, and a residual that fell could
not be told apart from a window that had merely slid past a transient. Pass it explicitly to keep
the window fixed while the run grows. 0 or omitted means NSTEPS // 5, the old behaviour.

The progress line reports each coordinate's sim/ref and the joint mean |ln(sim/ref)| over the
window SO FAR, so one long run shows whether the estimate has stopped moving. Before this it
printed sim/1D, which is the coupling correction and not the acceptance number.

BUT a cumulative number cannot establish stationarity on its own: it is one average over
everything seen so far, so a good early stretch hides a bad later one. The final report splits
the sampling window into --blocks=N disjoint equal-TIME blocks and gives each its own J. Read
the spread of those, not the movement of the cumulative line.
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 32
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
IDX = int(sys.argv[3]) if len(sys.argv) > 3 else 0
FRICTION = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
STRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 25


def _opt_int(name, default):
    """--name=N out of argv, leaving the positional arguments untouched."""
    pre = f"--{name}="
    for a in sys.argv[1:]:
        if a.startswith(pre):
            return int(a[len(pre):])
    return default
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
print(f"structure {s0['name']}  L={L}  pairs={len(s0['pairs'])}")
print(f"{NREP} replicas, {NSTEPS} steps of 0.002 ps = {NSTEPS * 0.002:.1f} ps per replica")
print(f"full field as shipped, 300 K, mass 110 Da, friction {FRICTION}/ps, "
      f"sampling every {STRIDE} steps")
# Provenance. Three earlier runs of this experiment were invalidated by a force-field defect
# found after they started -- a thermostat at 0.4 T, an effective mass 100x too large, and a
# K_INTRA 52x too soft -- and none of them recorded which field they had actually run against,
# so each result had to be judged by its numbers alone. Print the field's fingerprint instead.
# The list has to name EVERY constant whose value changes the sampled distribution. The four
# backbone terms were added after this script was written, and without them here a run under the
# old field and a run under the new one print the same fingerprint -- which is the provenance hole
# the paragraph above says invalidated three earlier runs.
_FINGERPRINT = ("K_BB", "K_INTRA_PC", "K_INTRA_CN", "K_INTRA_PN",
                "K_LINK_CP", "K_LINK_NP", "K_LINK_NC",
                "K_PAIR", "K_ANGLE", "K_DIH", "K_BPP", "K_STACK",
                "K_CLASH", "CLASH_SIGMA", "K_BSJ", "K_BSJ_GUIDE")
_missing = [n for n in _FINGERPRINT if not hasattr(C, n)]
if _missing:
    raise RuntimeError(f"the fingerprint names {_missing}, which this module does not define; a "
                       f"renamed constant would silently drop out of the provenance record")
print("field: " + "  ".join(f"{n}={getattr(C, n)}" for n in _FINGERPRINT))
import inspect as _inspect
_cap = _inspect.signature(C.cg_energy_forces).parameters["force_cap"].default
print(f"force_cap={_cap}  mass=110.0 Da  dt=0.002 ps  friction={FRICTION}/ps")
# The SHAPE of the guide is not a constant, so the fingerprint above cannot see it: a run under
# the short-range-reward form and a run under the long-range form print the SAME fingerprint.
# That is the same provenance hole this script's own paragraph says invalidated three earlier
# runs, so probe the shape directly instead of trusting a list of numbers. With r0 = 1.0,
# w = 0.2, k = 1: the long-range form gives ~0 at 0.5 nm and ~10 at 3.0 nm, the short-range
# reward gives ~10 then ~0.
_gs = C._sigmoid_f(torch.tensor([0.5, 3.0]), 1.0, 1.0, 0.2)[0]
_gnear, _gfar = float(_gs[0]), float(_gs[1])
print(f"guide shape: E(0.5 nm)={_gnear:.3f}  E(3.0 nm)={_gfar:.3f}  ("
      + ("long-range: zero below r0, bounded pull above" if _gnear < _gfar
         else "SHORT-RANGE REWARD -- the pre-6e7a44b form") + ")")
# The second B half-kick takes force_fn, so this run is symplectic. It has to be: the
# non-symplectic fallback pumps energy at dt*omega^2/(4*gamma) of the drag per step, and after
# K_INTRA was split into its two measured values the stiffest coordinate is C4'-N at
# 36399.2 kJ/mol/nm^2 over a 55 amu reduced mass, omega = sqrt(36399.2/55) = 25.73 /ps. At
# gamma = 0.1 and dt = 0.002 that ratio is dt*omega^2/(4*gamma) = 3.31 -- the pump is more than
# three times the drag, so the fallback would heat the intra-bead bonds rather than merely
# perturb them. The extra force evaluation doubles the cost and is worth it here.
print(f"second B half-kick: symplectic (force_fn recomputes at the post-update coordinates). "
      f"Without it the stiffest coordinate would pump at "
      f"{0.002 * (36399.2 / 55.0) / (4 * FRICTION):.2f}x the drag per step.")
print()

pos = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64).repeat(NREP, 1, 1)
vel = torch.zeros_like(pos)
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
temps = torch.full((NREP,), 300.0, dtype=torch.float64)

torch.manual_seed(SEED)
# The sampling window, decoupled from the run length. See the docstring.
_burn_arg = int(sys.argv[6]) if len(sys.argv) > 6 else 0
burn = _burn_arg if _burn_arg > 0 else max(NSTEPS // 5, 1)
print(f"burn = {burn} steps = {burn * 0.002:.1f} ps; sampling window "
      f"{burn * 0.002:.1f}-{NSTEPS * 0.002:.1f} ps")
print("  progress: each column is sim/ref; J is the joint mean |ln(sim/ref)| over the window so far")
# hoisted so the progress line can print the coupling ratio while the run is going
K_SHIPPED = {"bb_bond": C.K_BB, "intra_pc": C.K_INTRA_PC, "intra_cn": C.K_INTRA_CN,
             "angle": C.K_ANGLE, "dihedral": C.K_DIH, "stack": C.K_STACK}
# Stationarity needs DISJOINT blocks, not a cumulative average. A cumulative J over [burn, t]
# cannot separate "the window is settling" from "the early part of the window happened to look
# good": it is one number computed over everything seen so far. That is exactly how the
# 40-200 ps run read 0.0911 and then climbed to 0.0968 -- the second half was worse and the
# cumulative average hid it behind the first. So the accumulators are per block and every
# whole-window number below is the sum over blocks.
NB = max(_opt_int("blocks", 4), 1)
b_counts = {b: {c: np.zeros(len(TAB[c]["U"]), dtype=np.int64) for c in B.COORDS}
            for b in range(NB)}
b_acc = {b: {c: [0.0, 0.0, 0] for c in B.COORDS} for b in range(NB)}
b_frames = [0] * NB


def _summed():
    """Whole-window counts and moments, as the sum over the blocks."""
    ct = {c: np.zeros(len(TAB[c]["U"]), dtype=np.int64) for c in B.COORDS}
    ac = {c: [0.0, 0.0, 0] for c in B.COORDS}
    for b in range(NB):
        for c in B.COORDS:
            ct[c] += b_counts[b][c]
            a = b_acc[b][c]
            ac[c][0] += a[0]
            ac[c][1] += a[1]
            ac[c][2] += a[2]
    return ct, ac
# Clash watch. The analytical claim about the intra-bead bonds assumes the repulsion never
# fires, and C4'-N sits at 0.335 nm against a 0.300 nm cutoff, so that is not free.
clash_min = []
# The live range is CLASH_SIGMA. This used to count against CLASH_DIST, which is retired: it is
# 0.30 and nothing reads it, so the watch was blind to every pair between 0.30 and 0.3975 that the
# shipped wall does act on. Both are counted now, so a comparison against the old number stays
# possible without mistaking it for the live one.
clash_below = 0
clash_below_live = 0
import time
t0 = time.time()
def _forces_at(p):
    """Fresh forces at the post-update coordinates, for the symplectic tail kick.

    The cell list has to be rebuilt here rather than reused: it is built from positions.
    """
    cl2 = C.GPUCellList(cell_size=1.5)
    cl2.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl2)[1]


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
        # which disjoint block this sample falls in; blocks are equal in TIME, not in count
        blk = min(((step - burn) * NB) // max(NSTEPS - burn, 1), NB - 1)
        b_frames[blk] += 1
        with torch.no_grad():
            for c in B.COORDS:
                q = B.coords_of(pos, c).reshape(-1).numpy().astype(np.float64)
                a = b_acc[blk][c]
                a[0] += q.sum()
                a[1] += (q ** 2).sum()
                a[2] += q.size
                t = TAB[c]
                k = np.round((q - t["centre"][0]) / t["binw"]).astype(np.int64)
                ok = (k >= 0) & (k < len(t["U"]))
                b_counts[blk][c] += np.bincount(k[ok], minlength=len(t["U"]))
            beads = pos.reshape(NREP, -1, 3)
            dd = torch.cdist(beads, beads)
            dd = dd + torch.eye(dd.shape[-1], device=dd.device) * 10.0
            clash_min.append(float(dd.min()))
            clash_below += int((dd < C.CLASH_DIST).sum())
            clash_below_live += int((dd < C.CLASH_SIGMA).sum())
    if (step + 1) % max(NSTEPS // 10, 1) == 0:
        el = time.time() - t0
        parts, _joint = [], []
        _ct, _ac = _summed()
        for c in B.COORDS:
            s1, s2, n = _ac[c]
            if n:
                mm = s1 / n
                ss = float(np.sqrt(max(s2 / n - mm * mm, 0.0)))
                rs = TAB[c]["sigma"]
                r = ss / rs if rs > 0 else float("nan")
                parts.append(f"{c[:5]} {r:5.3f}")
                if r == r and r > 0:
                    _joint.append(abs(float(np.log(r))))
            else:
                parts.append(f"{c[:5]}   --")
        _j = float(np.mean(_joint)) if _joint else float("nan")
        print(f"  {step+1:>7d} {el:6.0f}s  " + "  ".join(parts) + f"   J {_j:.4f}")

el = time.time() - t0
print(f"done in {el:.0f} s, {NSTEPS / el:.1f} steps/s")
print()
print(f"clash watch: live range {C.CLASH_SIGMA:.4f} nm, retired cutoff {C.CLASH_DIST:.3f} nm; "
      f"closest bead pair ever {min(clash_min):.4f} nm")
print(f"  pair instances below the live range : {clash_below_live}")
print(f"  pair instances below the old 0.300 : {clash_below}  "
      f"(the number an earlier version of this script reported as if it were the live one)")
print()

# reference density from the stored table, on the same bins
def ref_p(c):
    U = TAB[c]["U"]
    p = np.exp(-(U - U.min()) / B.KBT)
    return p / p.sum()

# The single-coordinate prediction. Whatever the coupling does, a harmonic restraint of
# stiffness k on one coordinate has variance kBT/k on its own. If the shipped k was chosen as
# kBT/sigma_ref^2, then sigma_1D equals sigma_ref by construction and any gap between the
# sampled sigma and sigma_1D is the coupling, which is the only thing IBI can address.
print(f"{'coordinate':10s} {'ref sig':>8s} {'1-D sig':>8s} {'sim sig':>8s} "
      f"{'sim/ref':>8s} {'sim/1D':>7s} {'dU min':>8s} {'dU max':>8s} {'|dU|>1kBT':>10s}")
print("-" * 86)
rows = {}
counts, acc = _summed()
for c in B.COORDS:
    t = TAB[c]
    s1, s2, n = acc[c]
    m = s1 / n
    sig = float(np.sqrt(max(s2 / n - m * m, 0.0)))
    ps = (counts[c] + 1.0) / (counts[c].sum() + len(counts[c]))
    pr = ref_p(c)
    dU = B.KBT * np.log(ps / pr)
    ref_m = float((t["centre"] * pr).sum())
    frac = float((np.abs(dU) > B.KBT).mean())
    k = K_SHIPPED[c]
    s1d = float(np.sqrt(B.KBT / k)) if k > 0 else float("nan")
    rows[c] = (m, sig, dU)
    print(f"{c:10s} {t['sigma']:8.4f} {s1d:8.4f} {sig:8.4f} "
          f"{sig / t['sigma']:8.3f} {sig / s1d:7.3f} {dU.min():8.2f} {dU.max():8.2f} "
          f"{frac:10.3f}")
print()
# ── stationarity: disjoint blocks of the same sampling window, each with its own J ──
# This is the instrument the cumulative line cannot replace. If the blocks agree, the window is
# a sample of one distribution and a cumulative J over it means something. If they do not, the
# run has not reached the stationary distribution and no residual taken over it is an
# equilibrium number -- which is what the 40-200 ps run showed when its cumulative J dipped to
# 0.0911 and then climbed back to 0.0968.
if NB > 1:
    print(f"=== stationarity: {NB} disjoint blocks of the sampling window ===")
    print(f"{'block':>5s} {'window (ps)':>15s} {'frames':>7s} {'J':>7s}  "
          + "  ".join(f"{c[:5]:>5s}" for c in B.COORDS))
    print("-" * (37 + 7 * len(B.COORDS)))
    _Jb = []
    for b in range(NB):
        lo = burn + (NSTEPS - burn) * b // NB
        hi = burn + (NSTEPS - burn) * (b + 1) // NB
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
    _rel = (_hi - _lo) / _lo if _lo > 0 else float("nan")
    print(f"  block J spread: {_lo:.4f} to {_hi:.4f}  ({_rel * 100:.1f} percent of the lowest)")
    print("  Rule of thumb: within a few percent the blocks are consistent and the window is a"
          " sample of one distribution. Tens of percent means the run is still drifting and any"
          " residual over it is a mixture, not an equilibrium value. The per-coordinate columns"
          " say WHICH coordinate is moving.")
    print()
print("sim/1D is the coupling correction, and it is the whole content of the IBI step:")
print("where it is 1.0 the coupled chain already reproduces the single-coordinate result.")
print("sim/ref mixes that with the choice of k itself, which is a separate question.")
print()
print("=== the correction at the reference mode, and its curvature ===")
print(f"{'coordinate':10s} {'mode q':>9s} {'dU(mode)':>10s} {'dU at +1 sig':>13s} "
      f"{'dU at -1 sig':>13s}")
print("-" * 60)
for c in B.COORDS:
    t = TAB[c]
    m, sig, dU = rows[c]
    i0 = int(np.argmin(t["U"]))
    idx = lambda q: int(np.clip(np.round((q - t["centre"][0]) / t["binw"]), 0, len(dU) - 1))
    q0 = float(t["centre"][i0])
    print(f"{c:10s} {q0:9.4f} {dU[i0]:10.2f} {dU[idx(q0 + t['sigma'])]:13.2f} "
          f"{dU[idx(q0 - t['sigma'])]:13.2f}")
