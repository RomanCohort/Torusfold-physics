"""The mode, not the mean: what value is the bonded coordinate actually AT?

check_bonded_distributions.py showed the pseudo-torsion distribution is unimodal but
heavily skewed -- 63.9 percent of observations land in the top cosine bin. For a skewed
distribution the mean minimises the mean squared deviation but is not a state the molecule
occupies, so a restraint to the mean pins the structure somewhere it never goes.

This reports the modes, both pooled and per structure, so the two can be compared.

Run: python scripts/check_bonded_modes.py
"""
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torusfold.scheme2.torch_cgsim as C

DATA = Path(r"D:\torusfold-cgdata\rsRNASP\Training_set")


def p_chains(pdb):
    ch = collections.OrderedDict()
    for line in open(pdb):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if line[16] not in (" ", "A") or line[17:20].strip() not in ("A", "C", "G", "U"):
            continue
        if line[12:16].strip() != "P":
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        ch.setdefault(line[21], collections.OrderedDict())[line[22:27].strip()] = xyz
    return [np.array(list(v.values())) / 10.0 for v in ch.values() if len(v) >= 12]


def cos_angle(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    return (v1 * v2).sum(-1) / np.clip(n, 1e-9, None)


def cos_dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n0, n1 = np.cross(b0, b1), np.cross(b1, b2)
    n0 /= np.clip(np.linalg.norm(n0, axis=-1, keepdims=True), 1e-9, None)
    n1 /= np.clip(np.linalg.norm(n1, axis=-1, keepdims=True), 1e-9, None)
    return (n0 * n1).sum(-1)


D, A, S = [], [], []
per_mode_d, per_mode_a, per_mode_s = [], [], []
for f in sorted(DATA.glob("*.pdb")):
    for P in p_chains(f):
        if len(P) > 5:
            v = cos_dihedral(P[:-3], P[1:-2], P[2:-1], P[3:])
            D.append(v)
            h, e = np.histogram(v, bins=np.linspace(-1, 1, 41))
            per_mode_d.append(0.5 * (e[h.argmax()] + e[h.argmax() + 1]))
        if len(P) > 4:
            v = np.linalg.norm(P[2:] - P[:-2], axis=-1)
            S.append(v)
            h, e = np.histogram(v, bins=np.linspace(0.4, 2.0, 33))
            per_mode_s.append(0.5 * (e[h.argmax()] + e[h.argmax() + 1]))
        if len(P) > 3:
            v = cos_angle(P[:-2], P[1:-1], P[2:])
            A.append(v)
            h, e = np.histogram(v, bins=np.linspace(-1, 1, 41))
            per_mode_a.append(0.5 * (e[h.argmax()] + e[h.argmax() + 1]))

D, A, S = np.concatenate(D), np.concatenate(A), np.concatenate(S)


def mode(x, lo, hi, n=40):
    h, e = np.histogram(x, bins=np.linspace(lo, hi, n + 1))
    i = h.argmax()
    return 0.5 * (e[i] + e[i + 1]), h[i] / h.sum()


d_mode, d_frac = mode(D, -1, 1)
a_mode, a_frac = mode(A, -1, 1)
s_mode, s_frac = mode(S, 0.4, 2.0, 32)

print(f"pooled n = {len(D)} pseudo-torsions, {len(S)} i,i+2 distances")
print()
print(f"{'coordinate':26s} {'ff target':>10s} {'mean':>8s} {'MODE':>8s} {'at mode':>9s}")
print("-" * 66)
print(f"{'P-P-P-P cos':26s} {np.cos(C.DIH_PPPP):10.3f} {D.mean():8.3f} {d_mode:8.3f} "
      f"{d_frac*100:8.1f}%")
print(f"{'P-P-P cos':26s} {np.cos(C.ANGLE_PPP):10.3f} {A.mean():8.3f} {a_mode:8.3f} "
      f"{a_frac*100:8.1f}%")
print(f"{'P(i)-P(i+2) (nm)':26s} {C.STACK_R0:10.3f} {S.mean():8.3f} {s_mode:8.3f} "
      f"{s_frac*100:8.1f}%")
print()
pm = np.array(per_mode_d)
print(f"per-structure mode of the pseudo-torsion cos: mean {pm.mean():+.3f}, "
      f"sd {pm.std():.3f}")
print(f"  structures whose own mode is below cos -0.8 (near the ff target): "
      f"{int((pm < -0.8).sum())} / {len(pm)}")
print(f"  structures whose own mode is above cos +0.8: {int((pm > 0.8).sum())} / {len(pm)}")
pma = np.array(per_mode_a)
print(f"per-structure mode of the P-P-P cos: mean {pma.mean():+.3f}, sd {pma.std():.3f}")
print(f"  structures whose own mode is below cos -0.8: {int((pma < -0.8).sum())} / {len(pma)}")
pms = np.array(per_mode_s)
print(f"per-structure mode of the i,i+2 distance: mean {pms.mean():.3f}, sd {pms.std():.3f} nm")
print(f"  structures whose own mode is below 0.7 nm: {int((pms < 0.7).sum())} / {len(pms)}")
print()
print(f"if a single harmonic restraint is kept, the target that native structures actually")
print(f"satisfy is the mode, not the mean:")
print(f"  P-P-P-P cos  {-1.0:+.3f}  ->  {d_mode:+.3f}   (angle {np.degrees(np.arccos(d_mode)):.1f} deg)")
print(f"  P-P-P cos    {np.cos(C.ANGLE_PPP):+.3f}  ->  {a_mode:+.3f}   "
      f"(angle {np.degrees(np.arccos(a_mode)):.1f} deg) -- already close as shipped")
print(f"  i,i+2 nm     {C.STACK_R0:.3f}  ->  {s_mode:.3f}")
