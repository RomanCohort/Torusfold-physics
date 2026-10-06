"""Reconstruct ring atoms from the CG trace, then run the pair cross-check.

Why this is the step that was missing. Every all-atom product this run wrote came
back without base ring atoms -- cg2aa/*.pdb, final_allatom.pdb and remd_r0.pdb all
measure 12.0 atoms/residue with names P OP1 OP2 O5' C5' C4' O4' C3' O2' C2' C1'
plus one base representative. My pair detector needs the glycosidic N and three
ring atoms to fit a base plane, so on those files every candidate failed with
reason "missing" -- 809 of 809, 980 of 980. That is a measurement wall, not a
result, and it is why the delivered 21.3-atom model matters as a comparison.

`allatom_reconstruct.reconstruct_all_atom()` exists for exactly this: it takes the
(L, 3) P coordinates the CG solver produces and expands each residue into a full
all-atom template including the base rings. This script runs it on the run's own
CG trace and then repeats the cross-check against the checkpoint pairing.

What it can and cannot settle:
  CAN  -- whether ring atoms make the checkpoint's 82 pairs readable, and what
          C1'--C1' distances those pairs actually have.
  CANNOT -- whether the reconstruction's geometry is right. It is a template
          placement from A-form geometry, so a pair it reports as "formed" is a
          statement about the model's P trace, not an independent crystallographic
          measurement.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(r"C:\baidunetdiskdownload\torusfold-hybrid")
sys.path.insert(0, str(REPO / "src"))

from torusfold.immuno.checkpoint_pairs import load_checkpoint_pairing  # noqa: E402
from torusfold.immuno.pair_graph_from_coords import build_pair_graph  # noqa: E402
from torusfold.scheme2.allatom_reconstruct import (  # noqa: E402
    get_atom_xyzs,
    reconstruct_all_atom,
)

RUN = REPO / "results" / "immuno_full"
RING_N = {"N1", "N2", "N3", "N4", "N6", "N7", "N9", "O2", "O4", "O6"}

src = "".join((REPO / "artifacts" / "2013nt" / "sequence.txt").read_text(
    encoding="utf-8").split()).upper()
seq_all = "".join(c for c in src if c in "ACGU")

print("=" * 78)
print("reconstruct from the run's own CG trace")
print("=" * 78)

# The pipeline names the CG structure it hands to the all-atom step. `latest_cg.pdb`
# is a rolling snapshot for the viewer and is cleaned up, so it is not a reliable
# input; _final_cg_for_aa.pdb is the one written for this purpose.
cg_candidates = [RUN / "_final_cg_for_aa.pdb", RUN / "cg2aa" / "seg_0_cg.pdb",
                 RUN / "latest_cg.pdb"]
cg_path = next((p for p in cg_candidates if p.is_file()), None)
if cg_path is None:
    print(f"  no CG structure found among {[p.name for p in cg_candidates]}")
    sys.exit(1)
atoms = [l for l in cg_path.read_text(encoding="utf-8", errors="replace").splitlines()
         if l.startswith(("ATOM", "HETATM"))]
P = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in atoms])
L = len(P)
seq = seq_all[:L]
print(f"  input      : {cg_path.name}  {L} P atoms")
print(f"  sequence   : {len(seq)} nt")

try:
    structure = reconstruct_all_atom(P, seq)
except Exception as e:  # noqa: BLE001
    print(f"  reconstruct_all_atom failed: {type(e).__name__}: {e}")
    sys.exit(1)

names = [a.atom_name for a in structure.atoms]
resnames = sorted({a.res_name for a in structure.atoms})
print(f"  output     : {len(structure.atoms)} atoms, "
      f"{len(structure.atoms)/max(1,L):.1f} atoms/residue")
print(f"  res names  : {resnames}")
print(f"  atom names : {sorted(set(names))}")
ring = sorted(set(names) & RING_N)
print(f"  RING ATOMS : {ring if ring else 'none'}")
print(f"  => pair cross-check possible: {'YES' if ring else 'NO'}")

# --- write it out in PDB form for the detector -----------------------------
out = RUN / "_reconstructed_allatom.pdb"
lines = []
for i, a in enumerate(structure.atoms):
    lines.append(
        f"ATOM  {i+1:5d} {a.atom_name:<4s} {a.res_name:>3s} A{a.res_seq:4d}    "
        f"{a.xyz[0]:8.3f}{a.xyz[1]:8.3f}{a.xyz[2]:8.3f}  1.00  0.00          "
        f"{a.element:>2s}"
    )
lines.append("END")
out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(f"  written    : {out.relative_to(RUN)}")

# --- the cross-check ------------------------------------------------------
print()
print("=" * 78)
print("pair cross-check on the reconstruction")
print("=" * 78)

ck = load_checkpoint_pairing(RUN)
g = build_pair_graph(out.read_text(encoding="utf-8"), is_circular=True)
ck_set = {(min(i, j), max(i, j)) for i, j, _ in ck.pairs}
co_set = {(min(p.key, p.partner), max(p.key, p.partner)) for p in g.pairs}
reasons = collections.Counter(r["reason"].split(" ")[0] for r in g.rejected)

print(f"  residues parsed          : {len(g.residues)}")
print(f"  checkpoint pairs         : {len(ck_set)}")
print(f"  coordinate pairs         : {len(co_set)}")
print(f"  in both                  : {len(ck_set & co_set)}")
print(f"  candidates considered    : {len(g.rejected)}")
print(f"  rejection reasons        : {dict(reasons)}")
print(f"  helix runs               : {len(g.helix_runs())}  "
      f"lengths {[r['n_pairs'] for r in g.helix_runs()][:8]}")

if ck_set:
    print(f"  recall on checkpoint pairs: {(len(ck_set & co_set)/len(ck_set)):.1%}")

miss = sorted(ck_set - co_set)
if miss and g.residues:
    ds = []
    for i, j in miss:
        ri, rj = g.residues.get(i), g.residues.get(j)
        if ri is not None and rj is not None:
            ds.append(float(np.linalg.norm(rj.c1p - ri.c1p)))
    if ds:
        a = np.asarray(ds)
        print(f"  predicted-but-not-formed : {len(miss)} pairs, "
              f"C1'--C1' {a.min():.1f}-{a.max():.1f} A (a pair is ~10.4 A)")

print()
print("=" * 78)
print("compare against the run's own 12-atom products and the delivered model")
print("=" * 78)
print(f"  {'file':<44} {'atoms/res':>10} {'ring':>6}")
for p in [RUN / "final_allatom.pdb", RUN / "cg2aa" / "merged_aa.pdb", out,
          REPO / "artifacts" / "2013nt" / "isrnaclong_final.pdb"]:
    if not p.is_file():
        continue
    a = [l for l in p.read_text(encoding="utf-8", errors="replace").splitlines()
         if l.startswith(("ATOM", "HETATM"))]
    res: dict = collections.OrderedDict()
    nm = set()
    for l in a:
        res.setdefault((l[21], l[22:27]), l[17:20].strip())
        nm.add(l[12:16].strip())
    r = len(a) / max(1, len(res))
    label = str(p.relative_to(REPO)) if REPO in p.parents else p.name
    print(f"  {label:<44} {r:>10.1f} {'yes' if nm & RING_N else 'no':>6}")
