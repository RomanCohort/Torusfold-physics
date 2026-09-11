"""Are the P-only force-field targets wrong on native RNA in general, or only on 1EHZ?

check_ff_targets_vs_native.py measured the targets against 1EHZ and found the stacking
target (0.505 nm) and the pseudo-dihedral target (180 deg, i.e. cos = -1) far from native.
1EHZ is a tRNA, so that could be a quirk of one fold.

This repeats the P-only part on rsRNASP's training set -- 191 experimentally determined RNA
structures, unrelated to our reconstruction -- measuring the P atoms straight from the
deposited coordinates. No reconstruction is involved, so nothing here depends on our own
code.

Run: python scripts/measure_targets_on_training_set.py
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
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 191


def p_traces(pdb):
    """Per-chain ordered P atoms; only nucleotides; SEQADV maps modified -> parent."""
    parent = {}
    for line in open(pdb):
        if line.startswith("SEQADV"):
            p = line.split()
            if len(p) > 8 and p[-2:] == ["MODIFIED", "RESIDUE"]:
                parent[p[2]] = p[7]
    chains = collections.OrderedDict()
    that_chain = None
    for line in open(pdb):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if line[16] not in (" ", "A"):
            continue
        rname = line[17:20].strip()
        if rname not in ("A", "C", "G", "U") and rname not in parent:
            continue
        if line[12:16].strip() != "P":
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        key = (line[21], line[22:27].strip())
        chains.setdefault(line[21], {})[key] = xyz
    out = []
    for ch, resd in chains.items():
        if len(resd) >= 8:
            out.append(np.array(list(resd.values())))
    return out


def cos_angle(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    return (v1 * v2).sum(-1) / np.clip(n, 1e-9, None)


def cos_dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n0 = np.cross(b0, b1)
    n1 = np.cross(b1, b2)
    n0 /= np.clip(np.linalg.norm(n0, axis=-1, keepdims=True), 1e-9, None)
    n1 /= np.clip(np.linalg.norm(n1, axis=-1, keepdims=True), 1e-9, None)
    return (n0 * n1).sum(-1)


files = sorted(DATA.glob("*.pdb"))[:LIMIT]
acc = collections.defaultdict(list)
per_struct = collections.defaultdict(list)
used = 0
for f in files:
    traces = p_traces(f)
    if not traces:
        continue
    used += 1
    for P in traces:
        P = P / 10.0                                   # A -> nm
        if len(P) < 5:
            continue
        v = {
            "P-P bond": np.linalg.norm(P[1:] - P[:-1], axis=-1),
            "P(i)-P(i+2)": np.linalg.norm(P[2:] - P[:-2], axis=-1),
            "P(i)-P(i+3)": np.linalg.norm(P[3:] - P[:-3], axis=-1),
        }
        for k, arr in v.items():
            acc[k].append(arr)
            per_struct[k].append(arr.mean())
        a2 = cos_angle(P[:-2], P[1:-1], P[2:])
        acc["P-P-P cos"].append(a2)
        per_struct["P-P-P cos"].append(a2.mean())
        a3 = cos_dihedral(P[:-3], P[1:-2], P[2:-1], P[3:])
        acc["P-P-P-P cos"].append(a3)
        per_struct["P-P-P-P cos"].append(a3.mean())

print(f"{used} structures from rsRNASP/Training_set, {sum(len(v) for v in acc.values())} "
      f"coordinate values")
print("P atoms taken straight from the deposited coordinates; no reconstruction involved.")
print()
print(f"{'coordinate':18s} {'ff target':>10s} {'native mean':>12s} {'sd':>8s} "
      f"{'per-struct sd':>14s} {'5th pct':>9s} {'95th pct':>9s}")
print("-" * 86)
allv = np.concatenate(acc["P-P bond"]); m1 = np.asarray(per_struct["P-P bond"])
print(f"{'P-P bond (nm)':18s} {C.BOND_P_NEXT:10.3f} {allv.mean():12.3f} {allv.std():8.3f} "
      f"{m1.std():14.3f} {np.percentile(m1,5):9.3f} {np.percentile(m1,95):9.3f}")
allv = np.concatenate(acc["P(i)-P(i+2)"]); m1 = np.asarray(per_struct["P(i)-P(i+2)"])
print(f"{'P(i)-P(i+2) (nm)':18s} {C.STACK_R0:10.3f} {allv.mean():12.3f} {allv.std():8.3f} "
      f"{m1.std():14.3f} {np.percentile(m1,5):9.3f} {np.percentile(m1,95):9.3f}   <- STACK_R0")
allv = np.concatenate(acc["P(i)-P(i+3)"]); m1 = np.asarray(per_struct["P(i)-P(i+3)"])
print(f"{'P(i)-P(i+3) (nm)':18s} {'-':>10s} {allv.mean():12.3f} {allv.std():8.3f} "
      f"{m1.std():14.3f} {np.percentile(m1,5):9.3f} {np.percentile(m1,95):9.3f}")
allv = np.concatenate(acc["P-P-P cos"]); m1 = np.asarray(per_struct["P-P-P cos"])
print(f"{'P-P-P cos':18s} {np.cos(C.ANGLE_PPP):10.3f} {allv.mean():12.3f} {allv.std():8.3f} "
      f"{m1.std():14.3f} {np.percentile(m1,5):9.3f} {np.percentile(m1,95):9.3f}   <- ANGLE_PPP")
allv = np.concatenate(acc["P-P-P-P cos"]); m1 = np.asarray(per_struct["P-P-P-P cos"])
print(f"{'P-P-P-P cos':18s} {np.cos(C.DIH_PPPP):10.3f} {allv.mean():12.3f} {allv.std():8.3f} "
      f"{m1.std():14.3f} {np.percentile(m1,5):9.3f} {np.percentile(m1,95):9.3f}   <- DIH_PPPP")
print()
print("'per-struct sd' is the spread of the per-structure means -- it answers whether a")
print("target fitted to one structure would transfer to another.")
