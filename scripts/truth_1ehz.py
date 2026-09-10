"""Shared 1EHZ truth loader.

1EHZ records its 14 modified nucleotides as HETATM, not ATOM. A parser that reads only
ATOM silently drops them, leaving a chain with numbering gaps: every local frame built
across one of those gaps gets a wrong backbone step b = P[i+1] - P[i]. That bug
contaminated an earlier round of measurements; this loader includes HETATM and keeps
only nucleotides (HETATM also carries 169 waters and ions).

SEQADV lines give the authoritative modified -> parent mapping.
"""
import collections
from pathlib import Path

import numpy as np

PDB = Path(__file__).resolve().parent.parent / "_truth" / "1ehz.pdb"
if not PDB.exists():
    PDB = Path(r"D:\Torusfold-Templates\_tools\1ehz.pdb")

WCP = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
GLY = {"A": "N9", "G": "N9", "C": "N1", "U": "N1"}


def load(pdb=PDB, pair_lo=9.0, pair_hi=11.5):
    parent = {}
    for line in open(pdb):
        if line.startswith("SEQADV"):
            p = line.split()
            if len(p) > 8 and p[-2:] == ["MODIFIED", "RESIDUE"]:
                parent[p[2]] = p[7]

    order, res = [], collections.OrderedDict()
    for line in open(pdb):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if line[16] not in (" ", "A"):
            continue
        rname = line[17:20].strip()
        if rname not in ("A", "C", "G", "U") and rname not in parent:
            continue
        key = (line[21], line[22:27].strip(), rname)
        try:
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        if key not in res:
            res[key] = {}
            order.append(key)
        res[key].setdefault(line[12:16].strip(), xyz)

    base_of = [parent.get(k[2], k[2]) for k in order]
    ps = np.array([res[k]["P"] for k in order])
    partner = {}
    for a in range(len(order)):
        for b in range(a + 3, len(order)):
            if (base_of[a], base_of[b]) not in WCP:
                continue
            if 9.0 <= np.linalg.norm(res[order[a]]["C1'"] - res[order[b]]["C1'"]) <= pair_hi:
                partner.setdefault(a, b)
                partner.setdefault(b, a)

    meta = {
        "parent": parent,
        "n_residues": len(order),
        "n_atoms": sum(len(v) for v in res.values()),
        "contiguous": [int(k[1]) for k in order] == list(range(1, len(order) + 1)),
        "n_paired": len(partner),
    }
    return order, res, base_of, ps, partner, meta
