"""Do the force field's internal-coordinate targets match native RNA geometry?

diagnose_cg_force_terms.py showed which terms carry the force at a near-native geometry.
The four largest (dihedral, bpp, angle, stacking) are all "restraint toward a target", so
the next question is whether the targets match native RNA.

real_cg_beads returns ANGSTROM; the library's constants are in nm. An earlier version of
this script compared the two directly and produced nonsense (a 5.969 "P-P bond" against a
0.590 target). Everything is converted to nm before comparing.

Run: python scripts/check_ff_targets_vs_native.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import truth_1ehz
from torusfold.scheme2.aform_from_template import real_cg_beads
import torusfold.scheme2.torch_cgsim as C

order, res, base_of, ps, partner, meta = truth_1ehz.load()
L = len(order)
seq = "".join(base_of)
pairs = [(a, b) for a, b in partner.items() if a < b]
beads = real_cg_beads(ps, seq, pairs=pairs) / 10.0    # Angstrom -> nm

P, C4, NN = beads[:, 0, :], beads[:, 1, :], beads[:, 2, :]
pi = np.array([a for a, _ in pairs])
pj = np.array([b for _, b in pairs])


def d(a, b):
    return np.linalg.norm(a - b, axis=-1)


def cos_angle(a, b, c):
    v1, v2 = a - b, c - b
    return (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))


def cos_dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n0 = np.cross(b0, b1)
    n1 = np.cross(b1, b2)
    n0 /= np.linalg.norm(n0, axis=-1, keepdims=True)
    n1 /= np.linalg.norm(n1, axis=-1, keepdims=True)
    return (n0 * n1).sum(-1)


rows = []


def row(name, target, measured, const, unit="nm", angle=False):
    rows.append((name, float(target), measured, const, unit, angle))


row("P-P backbone bond", C.BOND_P_NEXT, d(P[:-1], P[1:]), "K_BB 500")
row("P-C4' intra-bead", C.BOND_P_C4, d(P, C4), "K_INTRA 400")
row("C4'-N intra-bead", C.BOND_C4_N, d(C4, NN), "K_INTRA 400")
row("N-N WC pair (harmonic)", C.PAIR_NN, d(NN[pi], NN[pj]), "K_PAIR 600")
row("N-N WC pair (bpp)", C.PAIR_NN, d(NN[pi], NN[pj]), "K_BPP/w = 2000")
row("P-P WC pair (guide)", C.PAIR_NN, d(P[pi], P[pj]), "K_PAIR_GUIDE/0.2 = 500")
row("P(i)-P(i+2) stacking", C.STACK_R0, d(P[:-2], P[2:]), "K_STACK 500")
row("P-P-P angle", math.degrees(C.ANGLE_PPP),
    np.degrees(np.arccos(np.clip(cos_angle(P[:-2], P[1:-1], P[2:]), -1, 1))),
    "K_ANGLE 600", "deg", True)
row("P-P-P-P dihedral", math.degrees(C.DIH_PPPP),
    np.degrees(np.arccos(np.clip(cos_dihedral(P[:-3], P[1:-2], P[2:-1], P[3:]), -1, 1))),
    "K_DIH 500", "deg", True)

print(f"force-field targets vs 1EHZ native geometry ({L} residues, {len(pairs)} WC pairs)")
print("beads converted to nm; the library's constants are nm.")
print()
print(f"{'coordinate':26s} {'target':>8s} {'native':>8s} {'sd':>7s} {'min':>8s} "
      f"{'max':>8s} {'off by':>9s}  {'constant':>18s}")
print("-" * 104)
for name, tgt, m, const, unit, is_ang in rows:
    unit_scale = 1.0 if is_ang else 1.0
    print(f"{name:26s} {tgt*unit_scale:8.3f} {m.mean():8.3f} {m.std():7.3f} "
          f"{m.min():8.3f} {m.max():8.3f} {m.mean()-tgt:+9.3f}  {const:>18s}")
print()

# The code restrains cos(angle), not the angle, so report the comparison in cosines too.
ca_n = cos_angle(P[:-2], P[1:-1], P[2:])
cd_n = cos_dihedral(P[:-3], P[1:-2], P[2:-1], P[3:])
print("the code restrains cos, so the same comparison as cosines (what the force uses):")
print(f"  P-P-P     target cos = {math.cos(C.ANGLE_PPP):+.3f}   native cos = "
      f"{ca_n.mean():+.3f} +/- {ca_n.std():.3f}   off by {ca_n.mean()-math.cos(C.ANGLE_PPP):+.3f}")
print(f"  P-P-P-P   target cos = {math.cos(C.DIH_PPPP):+.3f}   native cos = "
      f"{cd_n.mean():+.3f} +/- {cd_n.std():.3f}   off by {cd_n.mean()-math.cos(C.DIH_PPPP):+.3f}")
print()

print("what those offsets cost, using the term as coded")
nn_off = float(d(NN[pi], NN[pj]).mean()) - C.PAIR_NN
st_off = float(d(P[:-2], P[2:]).mean()) - C.STACK_R0
print(f"  harmonic N-N   K_PAIR*|off|         = {C.K_PAIR*abs(nn_off):8.0f} kJ/mol/nm"
      f"   (off {nn_off:+.3f} nm)")
print(f"  bpp            K_BPP/w * sigmoid    = {C.K_BPP/0.3*1/(1+math.exp(-1)):8.0f} kJ/mol/nm"
      f"   at the mean offset, capped by the field at {C.K_BPP/0.3:.0f}")
print(f"  stacking       K_STACK*|off|        = {C.K_STACK*abs(st_off):8.0f} kJ/mol/nm"
      f"   (off {st_off:+.3f} nm)")
print(f"  dihedral       K_DIH*|off in cos|   = {C.K_DIH*abs(float(cd_n.mean())-math.cos(C.DIH_PPPP)):8.0f}"
      f"   (off {float(cd_n.mean())-math.cos(C.DIH_PPPP):+.3f} in cos)")
print()
print("for scale: 200 kJ/mol/nm is the cap cg_energy_forces applies; the production path")
print("cg_forces_explicit_batched applies none.")
