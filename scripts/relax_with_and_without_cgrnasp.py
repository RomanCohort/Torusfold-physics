"""Does adding the cgRNASP term move a relaxation closer to the crystal?

Start from our own CG -> all-atom reconstruction of the 1EHZ P trace, relax by damped
gradient descent, and measure the bead RMSD against the crystal after P-superposition.

A previous version of this script used a normalised ("sign") step, which discards the
relative magnitudes of the forces and tore the structure apart; it is not a fair test of
any potential. Here the step is -lr * F with a per-bead displacement cap, and lr for each
potential is calibrated from its own force magnitude so the two are equally stable.

The absolute scale between the two potentials is not known and is not claimed here: the
cgRNASP weight is scanned, and "what weight should it carry" is left open.

Caveat: the start is built from the crystal's own P trace, so this is not an independent
test of prediction quality, only of which direction each term pushes.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import truth_1ehz
import cg_rnasp_evaluator as CG
from torusfold.scheme2.aform_from_template import real_cg_beads
from torusfold.scheme2.torch_cgsim import GPUCellList, cg_energy_forces

order, res, base_of, ps, partner, meta = truth_1ehz.load()
L = len(order)
seq = "".join(base_of)
pairs = [(a, b) for a, b in partner.items() if a < b]
types = CG.bead_types(base_of)
PA, PB, PC = CG.bead_pair_arrays(L)

# kJ/mol/nm (CG) and kBT/A (cgRNASP) are different units; convert the latter so both are
# kJ/mol/nm. 1 kBT at 300 K = 2.494 kJ/mol, 1/A = 10/nm.
KBT = 2.494
TRI_TO_KJMOL_NM = KBT * 10.0


def crystal_beads():
    out = np.zeros((L, 3, 3))
    for i in range(L):
        r = res[order[i]]
        gly = "N9" if base_of[i] in "AG" else "N1"
        out[i, 0] = r["P"]; out[i, 1] = r["C4'"]; out[i, 2] = r[gly]
    return out


def force_cg(beads):
    x = torch.tensor(beads.reshape(1, 3 * L, 3) / 10.0, dtype=torch.float64)
    pi = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pj = torch.tensor([b for _, b in pairs], dtype=torch.long)
    pw = torch.ones(len(pairs), dtype=torch.float64)
    cl = GPUCellList(cell_size=1.5)
    cl.build(x)
    _, f = cg_energy_forces(x, torch.stack([pi, pj], 1), pw, cell_list=cl)
    return f.reshape(3 * L, 3).numpy() * 10.0            # nm -> A


def force_tri(beads):
    _, f = CG.energy_forces_interp(
        beads.reshape(3 * L, 3), types, PA, PB, PC, L, want_forces=True)
    return f * TRI_TO_KJMOL_NM                          # kBT/A -> kJ/mol/A


def rmsd_to(beads, ref):
    a = beads.reshape(3 * L, 3); b = ref.reshape(3 * L, 3)
    pa, pb = a[0::3], b[0::3]
    ca, cb = pa.mean(0), pb.mean(0)
    H = (pa - ca).T @ (pb - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    a = (a - ca) @ R.T + cb
    diff = np.linalg.norm(a - b, axis=1)
    return diff.mean(), diff[0::3].mean(), diff[1::3].mean(), diff[2::3].mean()


ref = crystal_beads()
start = real_cg_beads(ps, seq, pairs=pairs)
r0 = rmsd_to(start, ref)
print(f"start (our reconstruction) vs crystal: all {r0[0]:.3f} A  "
      f"(P {r0[1]:.3f}, C4' {r0[2]:.3f}, N {r0[3]:.3f})")

f_cg0 = force_cg(start)
f_tr0 = force_tri(start)
mcg = np.median(np.linalg.norm(f_cg0, axis=1))
mtr = np.median(np.linalg.norm(f_tr0, axis=1))
print(f"median |F| at the start: CG {mcg:,.1f} kJ/mol/A   cgRNASP {mtr:,.1f} kJ/mol/A")
print()

TARGET = 0.005                       # A per step, median
LR_CG = TARGET / max(mcg, 1e-9)
LR_TRI = TARGET / max(mtr, 1e-9)
CAP = 0.05                           # A per bead per step
NSTEP = 400
print(f"gradient descent: {NSTEP} steps, median step {TARGET} A, cap {CAP} A")
print(f"{'case':24s} {'all':>8s} {'P':>8s} {'C4*':>8s} {'N':>8s}   {'E_cgrnasp':>11s}")
CASES = [("CG only", 1.0, 0.0),
         ("CG + 0.1 cgRNASP", 1.0, 0.1),
         ("CG + 0.5 cgRNASP", 1.0, 0.5),
         ("CG + 2.0 cgRNASP", 1.0, 2.0),
         ("cgRNASP only", 0.0, 1.0)]
for name, wc, wt in CASES:
    beads = start.copy()
    for step in range(NSTEP):
        f = wc * LR_CG * force_cg(beads) + wt * LR_TRI * force_tri(beads)
        n = np.linalg.norm(f, axis=1, keepdims=True)
        scale = np.minimum(1.0, CAP / np.maximum(n, 1e-12))
        beads = (beads.reshape(3 * L, 3) + f * scale).reshape(L, 3, 3)
    al, pp, cc, nn = rmsd_to(beads, ref)
    e = CG.energy_forces_interp(
        beads.reshape(3 * L, 3), types, PA, PB, PC, L, want_forces=False)[0]
    print(f"{name:24s} {al:8.3f} {pp:8.3f} {cc:8.3f} {nn:8.3f}   {e:11.2f}")
# ---------------------------------------------------------------------------
# Control: the same machinery with the sequence information destroyed.
#
# The concern is that any smooth field of the right magnitude could nudge a structure,
# and that "improvement" would then not be evidence that the table's distance
# dependence carries usable information. The fair null keeps everything -- geometry,
# pair list, table values, interpolator, step calibration -- and permutes only WHICH
# residue gets which base type. Smoothness and force-magnitude statistics are
# therefore preserved; the sequence-specific information is not.
#
# A bin-shuffle control would additionally break the table's smoothness, which changes
# the force-magnitude distribution and makes the step sizes incomparable; not used.
# ---------------------------------------------------------------------------
N_SHUFFLE = 12
rng = np.random.default_rng(20260212)


def relax(types_used, wc, wt, lr_cg, lr_tri):
    beads = start.copy()
    for _ in range(NSTEP):
        f = wc * lr_cg * force_cg(beads)
        _, ft = CG.energy_forces_interp(
            beads.reshape(3 * L, 3), types_used, PA, PB, PC, L, want_forces=True)
        f = f + wt * lr_tri * ft * TRI_TO_KJMOL_NM
        n = np.linalg.norm(f, axis=1, keepdims=True)
        scale = np.minimum(1.0, CAP / np.maximum(n, 1e-12))
        beads = (beads.reshape(3 * L, 3) + f * scale).reshape(L, 3, 3)
    return rmsd_to(beads, ref)[0]


print()
print("=" * 78)
print(f"control: {N_SHUFFLE} random type permutations (sequence information destroyed)")
print(f"{'case':26s} {'real':>8s} {'null mean':>10s} {'null sd':>8s} {'p':>6s}")
for name, wc, wt in [("cgRNASP only", 0.0, 1.0), ("CG + 2.0 cgRNASP", 1.0, 2.0)]:
    real = relax(types, wc, wt, LR_CG, LR_TRI)
    nulls = np.array([relax(CG.bead_types([base_of[p] for p in rng.permutation(L)]),
                            wc, wt, LR_CG, LR_TRI) for _ in range(N_SHUFFLE)])
    p = float((nulls <= real).sum() + 1) / (N_SHUFFLE + 1)
    print(f"{name:26s} {real:8.3f} {nulls.mean():10.3f} {nulls.std():8.3f} {p:6.3f}")
print()
print("p = fraction of null runs at least as close to the crystal (one-sided).")
print(f"start = {r0[0]:.3f} A; smaller is closer to the crystal.")

