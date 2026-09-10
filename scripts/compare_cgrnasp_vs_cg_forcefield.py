"""Does the cgRNASP term carry information our CG force field does not express?

Two energies on the same structures:
  E_cgrnasp  - the differentiable cgRNASP evaluator (scripts/cg_rnasp_evaluator.py)
  E_cg       - our own CG force field, cg_energy_forces, pair restraints left out because
               the pairing is external input rather than a property of the structure

Structures: the 1EHZ crystal; our CG -> all-atom reconstruction from the crystal P trace
alone; five decoys that rotate only the base bead about the P-C4' axis (which leaves both
intra-bead distances exactly intact, so the force field's bond terms cannot see it); and a
planar ring P trace put through the same reconstruction.

If cgRNASP separates the native from these and E_cg does not, the term carries marginal
information. If both separate them equally, it is largely redundant with what the force
field already encodes.
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
BP = CG.bead_pairs(L)


def crystal_beads():
    out = np.zeros((L, 3, 3))
    for i in range(L):
        r = res[order[i]]
        gly = "N9" if base_of[i] in "AG" else "N1"
        out[i, 0] = r["P"]; out[i, 1] = r["C4'"]; out[i, 2] = r[gly]
    return out


def rotate_about(point, axis, vec, angle):
    a = axis / np.linalg.norm(axis)
    v = vec - point
    c, s = np.cos(angle), np.sin(angle)
    return point + v * c + np.cross(a, v) * s + a * np.dot(a, v) * (1 - c)


def base_rotated(beads, seed):
    rng = np.random.default_rng(seed)
    out = beads.copy()
    for i in range(L):
        P, C4, N = out[i]
        ax = C4 - P
        if np.linalg.norm(ax) < 1e-6:
            continue
        out[i, 2] = rotate_about(P, ax, N, rng.uniform(0, 2 * np.pi))
    return out


def ring_beads():
    Rr = L * 5.9 / (2 * np.pi)
    ang = np.linspace(0, 2 * np.pi, L, endpoint=False)
    pr = np.stack([Rr * np.cos(ang), Rr * np.sin(ang), np.zeros(L)], axis=1)
    return real_cg_beads(pr, seq, pairs=pairs)


def e_cgrnasp(beads):
    x = beads.reshape(3 * L, 3)
    return CG.energy_forces_interp(x, types, BP, L, want_forces=False)[0]


def e_cg(beads):
    """Our CG force field, clash term included.

    Without the cell list the clash term is skipped (cg_energy_forces only calls it
    'if cell_list is not None'), and every remaining term is a function of the P beads
    and the two intra-bead distances. Rotating the base bead about the P-C4' axis
    preserves both, so the comparison would be against a crippled force field.
    """
    x = torch.tensor(beads.reshape(1, 3 * L, 3) / 10.0, dtype=torch.float64)
    empty = torch.zeros((0, 2), dtype=torch.long)
    cl = GPUCellList(cell_size=1.5)
    cl.build(x)
    e, _ = cg_energy_forces(x, empty, None, cell_list=cl)
    return float(e.reshape(-1)[0])


def e_cg_paired(beads):
    """The same, with the WC pairing restraint on. That N-N term is the only part of
    the force field that rewards a correct base position."""
    x = torch.tensor(beads.reshape(1, 3 * L, 3) / 10.0, dtype=torch.float64)
    pi = torch.tensor([a for a, _ in pairs], dtype=torch.long)
    pj = torch.tensor([b for _, b in pairs], dtype=torch.long)
    pw = torch.ones(len(pairs), dtype=torch.float64)
    cl = GPUCellList(cell_size=1.5)
    cl.build(x)
    e, _ = cg_energy_forces(x, torch.stack([pi, pj], 1), pw, cell_list=cl)
    return float(e.reshape(-1)[0])


native = crystal_beads()
cases = [("native (1EHZ crystal)", native),
         ("reconstruction from P", real_cg_beads(ps, seq, pairs=pairs))]
for k in range(5):
    cases.append((f"base-rotated decoy {k + 1}", base_rotated(native, 100 + k)))
cases.append(("planar ring decoy", ring_beads()))

print(f"L = {L}, cgRNASP bead pairs = {len(BP)}")
print()
print(f"{'structure':26s} {'E_cgrnasp':>12s} {'E_cg':>12s} {'E_cg+pairs':>12s}")
rows = []
for name, b in cases:
    a, c, cp = e_cgrnasp(b), e_cg(b), e_cg_paired(b)
    rows.append((name, a, c, cp))
    print(f"{name:26s} {a:12.2f} {c:12.1f} {cp:12.1f}")

nat_a, nat_c, nat_cp = rows[0][1], rows[0][2], rows[0][3]
print()
print("gaps against the native (positive = native is favoured)")
print(f"{'structure':26s} {'dE_cgrnasp':>12s} {'dE_cg':>12s} {'dE_cg+pairs':>12s}")
for name, a, c, cp in rows[1:]:
    print(f"{name:26s} {a - nat_a:+12.2f} {c - nat_c:+12.1f} {cp - nat_cp:+12.1f}")

dec = rows[1:]
print()
print(f"native ranked first by cgRNASP    : {sum(1 for _, a, _, _ in dec if a > nat_a)}/{len(dec)}")
print(f"native ranked first by E_cg       : {sum(1 for _, _, c, _ in dec if c > nat_c)}/{len(dec)}")
print(f"native ranked first by E_cg+pairs : {sum(1 for _, _, _, cp in dec if cp > nat_cp)}/{len(dec)}")
