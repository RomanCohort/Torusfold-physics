"""Extend the target measurement to the base beads, across the training set.

measure_targets_on_training_set.py already covers every P-only quantity. This adds the
base-bead quantities that PAIR_NN is supposed to target: N9(purine)/N1(pyrimidine) and
C4', read straight from the deposited coordinates.

Base pairs are identified with the same criterion the rest of the project uses -- C1'-C1'
between 9.0 and 11.5 A plus Watson-Crick complementarity. That is a definition, not a
measurement, and the resulting N-N distribution depends on it; it is stated here so the
number is not read as definition-free.

Only unmodified A/C/G/U residues contribute a base bead, because modified nucleotides name
their glycosidic nitrogen differently and would need the SEQADV parent map per atom.

Run: python scripts/measure_pair_targets_on_training_set.py
"""
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torusfold.scheme2.torch_cgsim as C
from truth_1ehz import WCP

DATA = Path(r"D:\torusfold-cgdata\rsRNASP\Training_set")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 191
WANT = {"P", "C4'", "C1'", "N9", "N1"}


def chains_of(pdb):
    """chain -> ordered [(resid, resname, {atom: xyz})] for A/C/G/U residues."""
    parent = {}
    for line in open(pdb):
        if line.startswith("SEQADV"):
            p = line.split()
            if len(p) > 8 and p[-2:] == ["MODIFIED", "RESIDUE"]:
                parent[p[2]] = p[7]
    ch = collections.OrderedDict()
    for line in open(pdb):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if line[16] not in (" ", "A"):
            continue
        rname = line[17:20].strip()
        if rname in parent or rname not in ("A", "C", "G", "U"):
            continue                               # unmodified A/C/G/U only
        aname = line[12:16].strip()
        if aname not in WANT:
            continue
        try:
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        key = line[22:27].strip()
        rec = ch.setdefault(line[21], collections.OrderedDict())
        rec.setdefault(key, {"res": rname, "atoms": {}})
        rec[key]["atoms"].setdefault(aname, xyz)
    out = []
    for c, rec in ch.items():
        # Require every anchor a bead triplet needs. A few deposited chains start at the
        # 5' position without a phosphate, and a few residues are missing the glycosidic
        # nitrogen; neither can contribute beads, so drop them rather than key on them.
        def complete(v):
            gly = "N9" if v["res"] in ("A", "G") else "N1"
            return all(a in v["atoms"] for a in ("C1'", "P", gly))

        lst = [v for v in rec.values() if complete(v)]
        if len(lst) >= 8:
            out.append(lst)
    return out


def pairs_of(lst):
    """WC pairs by C1'-C1' 9-11.5 A + complementarity, matching the project's criterion."""
    out = []
    for a in range(len(lst)):
        for b in range(a + 3, len(lst)):
            if (lst[a]["res"], lst[b]["res"]) not in WCP:
                continue
            d = np.linalg.norm(lst[a]["atoms"]["C1'"] - lst[b]["atoms"]["C1'"])
            if 9.0 <= d <= 11.5:
                out.append((a, b))
    return out


files = sorted(DATA.glob("*.pdb"))[:LIMIT]
NN, PP = [], []
per_nn, per_pp = [], []
n_str = n_chain = n_pair = 0
for f in files:
    chs = chains_of(f)
    if not chs:
        continue
    n_str += 1
    for lst in chs:
        n_chain += 1
        pr = pairs_of(lst)
        if not pr:
            continue
        n_pair += len(pr)
        gly = lambda r: "N9" if r in ("A", "G") else "N1"
        nn = [np.linalg.norm(lst[a]["atoms"][gly(lst[a]["res"])]
                             - lst[b]["atoms"][gly(lst[b]["res"])]) for a, b in pr]
        pp = [np.linalg.norm(lst[a]["atoms"]["P"] - lst[b]["atoms"]["P"]) for a, b in pr]
        NN.append(np.array(nn) / 10.0)
        PP.append(np.array(pp) / 10.0)
        per_nn.append(np.mean(nn) / 10.0)
        per_pp.append(np.mean(pp) / 10.0)

nn = np.concatenate(NN)
pp = np.concatenate(PP)
print(f"{n_str} structures, {n_chain} chains, {n_pair} Watson-Crick pairs identified")
print("criterion: C1'-C1' 9.0-11.5 A + A-U/G-C/G-U complementarity; unmodified residues only")
print()
print(f"{'quantity':22s} {'ff target':>10s} {'native mean':>12s} {'sd':>8s} "
      f"{'per-struct sd':>14s} {'5th':>7s} {'95th':>7s}")
print("-" * 84)
for name, arr, per, tgt in [("N-N (nm)", nn, per_nn, C.PAIR_NN),
                            ("P-P (nm)", pp, per_pp, C.PAIR_NN)]:
    per = np.asarray(per)
    print(f"{name:22s} {tgt:10.3f} {arr.mean():12.3f} {arr.std():8.3f} "
          f"{per.std():14.3f} {np.percentile(per,5):7.3f} {np.percentile(per,95):7.3f}")
print()
print(f"PAIR_NN = {C.PAIR_NN:.3f} nm is used for BOTH rows. It matches neither.")
print(f"  N-N off by {nn.mean()-C.PAIR_NN:+.3f} nm -> K_BPP/w * sigma gives the bpp force)")
print(f"  P-P off by {pp.mean()-C.PAIR_NN:+.3f} nm -> the pair guide term")
