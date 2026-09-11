"""After the four backbone terms: does a long run settle, and where does the chain reach?

The short diagnostic (scripts/identify_penetrating_pair.py) minimises with a step-halving scheme
that stalls once the step size underflows, and then runs 3 ps. On the stiffer field that stall
happens at 778 iterations with the force still well above the harmonic floor, so the 3 ps window
is measuring relaxation rather than equilibrium, and the mean over it (569 K) is not a property of
the thermostat.

This does it properly: restart the descent when the step stalls so the structure actually reaches a
minimum, then run long enough that the reported temperature is not the transient.

Baseline to compare against, from the same measurement on the field before the four terms existed
(scripts/test_backbone_13_terms.py, 15 ps, 2 replicas, seed 99):
    shipped field:      mean T 326.6 K   last quarter 292.5 K   drift -1174.4   deepest 0.0058 nm
    with three terms:   mean T 300.4 K   last quarter 300.8 K   drift  -383.4   deepest 0.3086 nm

Run: python scripts/check_field_after_fix.py [ps] [nrep] [max_iter]
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

PS = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
NREP = int(sys.argv[2]) if len(sys.argv) > 2 else 2
MAXIT = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
MASS = 110.0
KB_INT = 0.008314462618
TARGET = 300.0
DT = 0.002
NSTEPS = int(round(PS / DT))
EQUIL = NSTEPS // 4          # discard the first quarter before reporting anything
ATOM = ("P", "C4'", "N9/N1")
SHORT = {(0, 2, 0): "P(i)-N9(i)", (1, 0, 1): "C4'(i)-P(i+1)",
         (2, 0, 1): "N9(i)-P(i+1)", (2, 1, 1): "N9(i)-C4'(i+1)"}
BONDED = {(0, 1, 0): "P(i)-C4'(i) bonded", (1, 2, 0): "C4'(i)-N9(i) bonded",
          (0, 0, 1): "P(i)-P(i+1) bonded"}
NATIVE_MIN = {"P(i)-N9(i)": 0.3801, "C4'(i)-P(i+1)": 0.2810,
              "N9(i)-P(i+1)": 0.4731, "N9(i)-C4'(i+1)": 0.3799}

pool = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s0 = pool[0]
L = len(s0["pos"])
NB = 3 * L
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
xs = torch.tensor(s0["pos"].reshape(1, NB, 3), dtype=torch.float64)


def classify(i, j):
    a, b = (i, j) if i < j else (j, i)
    ra, ai = divmod(a, 3)
    rb, aj = divmod(b, 3)
    return (ai, aj, rb - ra)


def label_of(i, j):
    """Name a bead pair, and never through a bare default.

    The first version of this used SHORT.get(cl, "wall-applied pair"), which silently absorbed the
    two BONDED classes: a P(i)-C4' contact at 0.3022 nm was reported as a wall-applied pair and
    read as the excluded volume being penetrated, when the wall's own nearest approach was 0.3411.
    Same shape as the hand-written audit table that missed the fourth unguarded pair.

    The default now fires only when the pair really has |i-j| >= 3, which is the set the excluded
    volume acts on; anything else is returned as UNCLASSIFIED so it is visible instead of absorbed.
    """
    cl = classify(i, j)
    if cl in SHORT:
        return SHORT[cl]
    if cl in BONDED:
        return BONDED[cl]
    if abs(i - j) >= 3:
        return "wall-applied pair"
    return f"UNCLASSIFIED {cl}"


def field(p, cap=None):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl, force_cap=cap)


def maxf(p):
    with torch.no_grad():
        _e, f = field(p)
    return float(torch.linalg.norm(f.reshape(-1, 3), dim=-1).max())


# ── minimise, restarting the descent when the step underflows ──
x = xs.clone()
e_cur = float(field(x)[0].reshape(-1)[0])
s = 1e-5
resets, total = 0, 0
for k in range(MAXIT):
    with torch.no_grad():
        _e, f = field(x)
        tr = x + s * f
        e_tr = float(field(tr)[0].reshape(-1)[0])
    total = k + 1
    if e_tr < e_cur:
        x, e_cur, s = tr, e_tr, min(s * 1.2, 1e-2)
    else:
        s *= 0.5
        if s < 1e-12:
            if resets >= 6:
                break
            resets += 1
            s = 1e-4                      # restart, so a stall is not the end of the descent
x_min = x.detach().clone()
print(f"{s0['name']}  {L} residues / {NB} beads / {len(ij)} WC pairs")
print(f"minimised: {total} iterations, {resets} restarts, E = {e_cur:.2f} kJ/mol, "
      f"max|F| = {maxf(x_min):.2f} kJ/mol/nm")
print(f"  (the harmonic force floor is sqrt(k*kBT) = 236.3 for P-C4', so a residual at that scale"
      f" is as low as this criterion goes)")
print()

xr = x_min.clone().repeat(NREP, 1, 1)
v = torch.zeros_like(xr)
temps = torch.full((NREP,), TARGET, dtype=torch.float64)
torch.manual_seed(99)


def ff(p):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl, force_cap=5000.0)[1]


Ts, Es, deepest, close = [], [], {}, 0
for step in range(NSTEPS):
    with torch.no_grad():
        f = ff(xr)
        xr, v = C.batch_langevin_step(xr, v, f, temps, dt_ps=DT, mass_amu=MASS,
                                      friction=1.0, force_fn=ff)
        if step % 10 == 0:
            Ts.append(float((MASS * (v ** 2).sum(dim=-1) / (3.0 * KB_INT)).mean()))
            if step % (NSTEPS // 10) == 0:
                Es.append(float(field(xr)[0].mean()))
            if step >= EQUIL:
                b = xr.reshape(NREP, NB, 3)
                d = torch.cdist(b, b) + torch.eye(NB)[None] * 1e6
                for r in range(NREP):
                    i, j = divmod(int(d[r].argmin()), NB)
                    dm = float(d[r, i, j])
                    lab = label_of(i, j)
                    if dm < deepest.get(lab, 1e9):
                        deepest[lab] = dm
                    if dm < 0.30:
                        close += 1

Ts = np.array(Ts)
Es = np.array(Es)
eq = Ts[int(len(Ts) * 0.75):]
print(f"{PS} ps Langevin, {NREP} replicas, cap 5000, target {TARGET:.0f} K, "
      f"reporting after {EQUIL * DT:.1f} ps")
print(f"mean T {Ts.mean():.1f} K ({Ts.mean()/TARGET:.2f}x)   "
      f"last quarter {eq.mean():.1f} K ({eq.mean()/TARGET:.2f}x)")
print(f"potential energy {Es[0]:.1f} -> {Es[-1]:.1f} kJ/mol, drift {Es[-1] - Es[0]:.1f}")
print(f"T(t): " + " ".join(f"{t:.0f}" for t in Ts[::max(1, len(Ts)//14)]))
print(f"E(t): " + " ".join(f"{e:.0f}" for e in Es))
print()
print("deepest approach after equilibration, by pair class:")
for lab, dm in sorted(deepest.items(), key=lambda kv: kv[1]):
    if lab in NATIVE_MIN:
        print(f"    {lab:22s} {dm:.4f} nm   (database minimum for this class {NATIVE_MIN[lab]:.4f})")
    else:
        print(f"    {lab:22s} {dm:.4f} nm")
print()
print(f"frames after equilibration with the closest pair below 0.30 nm: {close} "
      f"of {len(range(EQUIL, NSTEPS, 10)) * NREP} sampled")
