"""Are the bonded coordinates unimodal? The recalibration depends on it.

recalibrate_ff_targets.py set each target to the MEAN of the observable, which is the
value that minimises the mean squared restraint energy. That argument holds only for a
unimodal distribution. The P-P-P-P pseudo-torsion cosine has a pooled spread of 0.584,
which is nearly the full range of a cosine, so "the mean is +0.577" may be an average of
two populations rather than a state the molecule is ever in -- and a restraint to the mean
of a bimodal distribution pins the structure to a place it never visits.

This plots the distributions instead of summarising them. P atoms come straight from the
deposited coordinates.

Run: python scripts/check_bonded_distributions.py
"""
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torusfold.scheme2.torch_cgsim as C

import _cgdata
DATA = _cgdata.rsrnasp()


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


dih, st, ang, per_mode = [], [], [], []
for f in sorted(DATA.glob("*.pdb")):
    for P in p_chains(f):
        if len(P) > 5:
            d = cos_dihedral(P[:-3], P[1:-2], P[2:-1], P[3:])
            dih.append(d)
        if len(P) > 4:
            st.append(np.linalg.norm(P[2:] - P[:-2], axis=-1))
        if len(P) > 3:
            ang.append(cos_angle(P[:-2], P[1:-1], P[2:]))

dih = np.concatenate(dih)
st = np.concatenate(st)
ang = np.concatenate(ang)


def hist(name, x, target, lo, hi, nb=10):
    print(f"--- {name}: n = {len(x)}, mean {x.mean():+.3f}, sd {x.std():.3f}  "
          f"(ff target {target:+.3f})")
    edges = np.linspace(lo, hi, nb + 1)
    counts, _ = np.histogram(x, bins=edges)
    frac = counts / counts.sum()
    for i in range(nb):
        bar = "#" * int(round(frac[i] * 100))
        print(f"   [{edges[i]:+.2f},{edges[i+1]:+.2f})  {frac[i]*100:5.1f}%  {bar}")
    print()


hist("P-P-P-P pseudo-torsion cos", dih, np.cos(C.DIH_PPPP), -1.0, 1.0)
hist("P-P-P cos", ang, np.cos(C.ANGLE_PPP), -1.0, 1.0)
hist("P(i)-P(i+2) distance (nm)", st, C.STACK_R0, 0.4, 2.0, 8)

print(f"fraction of pseudo-torsions within 0.1 of the ff target cos = "
      f"{float((np.abs(dih - np.cos(C.DIH_PPPP)) < 0.1).mean())*100:.1f}%")
print(f"fraction within 0.1 of the recalibrated mean cos = "
      f"{float((np.abs(dih - 0.577) < 0.1).mean())*100:.1f}%")
print(f"fraction within 0.1 nm of the ff stacking target = "
      f"{float((np.abs(st - C.STACK_R0) < 0.1).mean())*100:.1f}%")
print(f"fraction within 0.1 nm of the recalibrated stacking target = "
      f"{float((np.abs(st - 1.152) < 0.1).mean())*100:.1f}%")
