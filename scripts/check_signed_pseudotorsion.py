"""Signed pseudo-torsion, so the geometry can be described without ambiguity.

check_bonded_modes.py used the angle between two plane normals, which is the ABSOLUTE
dihedral. Near 0 degrees it cannot distinguish +13 from -13, and for describing what the
backbone actually does that distinction matters. This uses the atan2 form.

Run: python scripts/check_signed_pseudotorsion.py
"""
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
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


def signed_dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n0, n1 = np.cross(b0, b1), np.cross(b1, b2)
    b1u = b1 / np.clip(np.linalg.norm(b1, axis=-1, keepdims=True), 1e-12, None)
    m = np.cross(n0, b1u)
    return np.degrees(np.arctan2((m * n1).sum(-1), (n0 * n1).sum(-1)))


vals = []
for f in sorted(DATA.glob("*.pdb")):
    for P in p_chains(f):
        if len(P) > 5:
            vals.append(signed_dihedral(P[:-3], P[1:-2], P[2:-1], P[3:]))
v = np.concatenate(vals)
print(f"signed P-P-P-P pseudo-torsion, n = {len(v)}")
print(f"mean {v.mean():+.2f} deg, sd {v.std():.2f}, median {np.median(v):+.2f}")
print()
h, e = np.histogram(v, bins=np.arange(-180, 185, 15))
for i in range(len(h)):
    print(f"  [{e[i]:+6.0f},{e[i+1]:+6.0f})  {h[i]/h.sum()*100:5.1f}%  "
          f"{'#' * int(round(h[i]/h.sum()*100))}")
print()
print(f"fraction within 30 deg of 0 (cis-like):  {float((np.abs(v) < 30).mean())*100:.1f}%")
print(f"fraction within 30 deg of 180 (trans):   {float((np.abs(np.abs(v) - 180) < 30).mean())*100:.1f}%")
print(f"fraction with v < 0: {float((v < 0).mean())*100:.1f}%   v > 0: {float((v > 0).mean())*100:.1f}%")
