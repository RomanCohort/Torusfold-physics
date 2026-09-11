"""Where does the chain reach, now that the four short backbone pairs have terms?

History, because the number only means something next to it. Before the four terms existed, a 3 ps
Langevin run from a minimised structure brought a bead pair to 0.0058 nm -- through each other --
and 811 of 900 sampled frames had their closest contact on one of the four pairs the excluded
volume skips (|i-j| <= 2) and no bonded term owned. The wall was not at fault: unit-tested it
returns 608203 kJ/mol/nm at 0.0863 nm and the cell list contains the pair
(scripts/unit_test_clash_reach.py).

This reruns that measurement against the fixed field and reports the closest approach per pair
class, so the same script answers both "did it become physical" and "which class is still the
tightest".

Bead indexing is flat: bead 3i+0 = P(i), 3i+1 = C4'(i), 3i+2 = N9/N1(i), bead 3i+3 = P(i+1), so a
pair is classified by (atom_i, atom_j, residue_j - residue_i).

Run: python scripts/identify_penetrating_pair.py [max_iter] [ps] [nrep]
"""
import sys
import collections
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

MAXIT = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
PS = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
NREP = int(sys.argv[3]) if len(sys.argv) > 3 else 4
MASS = 110.0
KB_INT = 0.008314462618
TARGET = 300.0
DT = 0.002
NSTEPS = int(round(PS / DT))
ATOM = ("P", "C4'", "N9/N1")

# The four pairs that had no term, keyed by (atom_i, atom_j, residue_j - residue_i). They carry
# K_INTRA_PN / K_LINK_CP / K_LINK_NP / K_LINK_NC now, and the excluded volume still skips them.
SHORT = {(0, 2, 0): "P(i)-N9(i)      short", (1, 0, 1): "C4'(i)-P(i+1)   short",
         (2, 0, 1): "N9(i)-P(i+1)    short", (2, 1, 1): "N9(i)-C4'(i+1)  short"}
BONDED = {(0, 1, 0): "P(i)-C4'(i)     bonded", (1, 2, 0): "C4'(i)-N9(i)    bonded",
          (0, 0, 1): "P(i)-P(i+1)     walled"}
SHORT_CLASSES = set(SHORT)


def classify(i, j):
    a, b = (i, j) if i < j else (j, i)
    ra, ai = divmod(a, 3)
    rb, aj = divmod(b, 3)
    return (ai, aj, rb - ra)


def name_of(cl):
    return SHORT.get(cl) or BONDED.get(cl) or "far   wall applies"


pool = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s0 = pool[0]
L = len(s0["pos"])
NB = 3 * L
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)   # residue indices
pw = torch.ones(len(ij), dtype=torch.float32)
xs = torch.tensor(s0["pos"].reshape(1, NB, 3), dtype=torch.float64)


def field(p, cap=None):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl, force_cap=cap)


def report(b, tag):
    d = torch.cdist(b, b) + torch.eye(NB) * 1e6
    i, j = divmod(int(d.argmin()), NB)
    cl = classify(i, j)
    print(f"{tag}: closest {float(d[i, j]):.4f} nm  bead ({i},{j})  {ATOM[cl[0]]}({i//3})"
          f"-{ATOM[cl[1]]}({j//3})   {name_of(cl)}")
    allow = d.clone()
    ix = torch.arange(NB)
    allow[(ix[None, :] - ix[:, None]).abs() <= 2] = 1e6
    i2, j2 = divmod(int(allow.argmin()), NB)
    print(f"{tag}: closest with the wall allowed (|i-j|>=3) {float(allow[i2, j2]):.4f} nm")


x = xs.clone()
e_cur = float(field(x)[0].reshape(-1)[0])
s = 1e-5
for k in range(MAXIT):
    with torch.no_grad():
        _e, f = field(x)
        tr = x + s * f
        e_tr = float(field(tr)[0].reshape(-1)[0])
    if e_tr < e_cur:
        x, e_cur, s = tr, e_tr, min(s * 1.2, 1e-2)
    else:
        s *= 0.5
        if s < 1e-12:
            break
x_min = x.detach().clone()
print(f"{s0['name']}  {L} residues / {NB} beads / {len(ij)} pairs   "
      f"minimised {k+1} iters to E = {e_cur:.2f} kJ/mol")
report(xs[0], "start  ")
report(x_min[0], "minimum")
print()

xr = x_min.clone().repeat(NREP, 1, 1)
v = torch.zeros_like(xr)
temps = torch.full((NREP,), TARGET, dtype=torch.float64)
torch.manual_seed(99)


def ff(p):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl, force_cap=5000.0)[1]


Ts, deepest, close = [], {}, []
for step in range(NSTEPS):
    with torch.no_grad():
        f = ff(xr)
        xr, v = C.batch_langevin_step(xr, v, f, temps, dt_ps=DT, mass_amu=MASS,
                                      friction=1.0, force_fn=ff)
        if step >= NSTEPS // 4 and step % 5 == 0:
            Ts.append(float((MASS * (v ** 2).sum(dim=-1) / (3.0 * KB_INT)).mean()))
            b = xr.reshape(NREP, NB, 3)
            d = torch.cdist(b, b) + torch.eye(NB)[None] * 1e6
            for r in range(NREP):
                i, j = divmod(int(d[r].argmin()), NB)
                dm = float(d[r, i, j])
                cl = name_of(classify(i, j))
                if dm < deepest.get(cl, 1e9):
                    deepest[cl] = dm
                if dm < 0.30:
                    close.append((cl, dm, i, j, r, step))

Ts = np.array(Ts)
print(f"dynamics from the minimum: {PS} ps x {NREP} replicas, cap 5000, target {TARGET:.0f} K")
print(f"mean kinetic T {Ts.mean():.1f} K   ratio {Ts.mean()/TARGET:.2f}")
print(f"T over time: " + " ".join(f"{t:.0f}" for t in Ts))
print()
print("deepest approach reached, by pair class:")
for cl, dm in sorted(deepest.items(), key=lambda kv: kv[1]):
    print(f"    {cl:26s} {dm:.4f} nm")
print()
print(f"frames with the closest pair below 0.30 nm: {len(close)} of {len(Ts) * NREP}")
if close:
    close.sort(key=lambda h: h[1])
    print("  deepest 15:")
    for cl, dm, i, j, r, st in close[:15]:
        print(f"    step {st:6d} rep {r}  bead ({i:3d},{j:3d})  {cl:26s} {dm:.4f} nm")
    print()
    print("  count by class:", dict(collections.Counter(c[0] for c in close)))
print()
print("The four short classes are compared against their own database minima, which are 0.3801 for")
print("P-N9, 0.2810 for C4'-P(next), 0.4731 for N9-P(next) and 0.3799 for N9-C4'(next) nm.")
print("Contact near those numbers is native geometry. Contact at 0.01 nm is beads overlapping.")
