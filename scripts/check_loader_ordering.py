"""Is the 6.37 degree P-P-P angle real, or an artefact of how the loader orders residues?

diagnose_dihedral_large_force.py found the largest dihedral forces all sharing one atom,
and that atom sits in a window whose P-P-P angle is 6.37 degrees -- a backbone folding back
on itself. That is not a plausible geometry, and the same loader built the distributions the
Boltzmann tables were fitted to, so it has to be checked before anything is concluded from
those tables.

boltzmann_bonded._chain_residues orders residues by the order they appear in the file and
keeps only A/C/G/U. A modified nucleotide in the middle is therefore dropped, which would
leave two entries that are consecutive in the list but not in the chain. This looks for that
by printing the residue numbering of the worst window's four residues.

Run: python scripts/check_loader_ordering.py
"""
import collections
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B
import torusfold.scheme2.torch_cgsim as C

DATA = B.DATA


def chains_with_ids(pdb):
    """Same filter as the real loader, but keeping residue ids so gaps are visible."""
    ch = collections.OrderedDict()
    for line in open(pdb):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if line[16] not in (" ", "A"):
            continue
        rname = line[17:20].strip()
        if rname not in ("A", "C", "G", "U"):
            continue
        aname = line[12:16].strip()
        gly = "N9" if rname in ("A", "G") else "N1"
        if aname not in ("P", "C4'", gly, "C1'"):
            continue
        try:
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError:
            continue
        rec = ch.setdefault(line[21], collections.OrderedDict())
        rec.setdefault((line[22:27].strip(), rname), {})[aname] = xyz
    out = []
    for chain, rec in ch.items():
        lst = []
        for (rid, rname), at in rec.items():
            gly = "N9" if rname in ("A", "G") else "N1"
            if not all(k in at for k in ("P", "C4'", gly, "C1'")):
                continue
            lst.append((rid, rname, at))
        if B.MIN_L < len(lst) <= B.MAX_L:
            out.append((chain, lst))
    return out


def num(s):
    try:
        return int(s)
    except ValueError:
        return None


worst = []
for f in sorted(DATA.glob("*.pdb")):
    for chain, lst in chains_with_ids(f):
        pos = torch.tensor(
            np.array([[a["P"], a["C4'"], a["N9" if r in ("A", "G") else "N1"]]
                      for _i, r, a in lst]).reshape(1, -1, 3) / 10.0,
            dtype=torch.float64)
        L = len(lst)
        _, F = C._dihedral_f(pos, C.K_DIH, np.cos(C.DIH_PPPP))
        P = torch.arange(L) * 3
        for i in range(L - 3):
            idx = [P[i + k] for k in range(4)]
            fmag = float(torch.linalg.norm(F[0, idx], dim=-1).max())
            worst.append((fmag, f.stem, chain, i, lst, pos))
worst.sort(key=lambda t: -t[0])

print("the three largest dihedral forces, with the residue numbering behind them")
for fmag, name, chain, i, lst, pos in worst[:3]:
    print()
    print(f"{name} chain {chain!r}, window starting at index {i}, |F| = {fmag:.1f} kJ/mol/nm")
    rids = [lst[i + k][0] for k in range(4)]
    print(f"  residue ids    {rids}")
    nums = [num(r) for r in rids]
    if all(n is not None for n in nums):
        print(f"  numeric gaps   {[nums[k+1] - nums[k] for k in range(3)]}")
    print(f"  residue names  {[lst[i + k][1] for k in range(4)]}")
    P = torch.arange(len(lst)) * 3
    for k in range(4):
        p = pos[0, P[i + k]]
        print(f"    {rids[k]:>5s} {lst[i+k][1]}  P = "
              f"({float(p[0]):7.3f}, {float(p[1]):7.3f}, {float(p[2]):7.3f}) nm")

# how common are gaps overall
print()
gap_hist = collections.Counter()
n_win = 0
for f in sorted(DATA.glob("*.pdb")):
    for chain, lst in chains_with_ids(f):
        nums = [num(r) for r, _n, _a in lst]
        for k in range(len(nums) - 1):
            if nums[k] is not None and nums[k + 1] is not None:
                gap_hist[nums[k + 1] - nums[k]] += 1
                n_win += 1
print(f"residue-number steps between consecutive kept entries, over {n_win} pairs:")
for step, n in sorted(gap_hist.items()):
    tag = "   <- a gap: residues were dropped between these two" if step != 1 else ""
    print(f"  step {step:>3d}: {n:6d}{tag}")
