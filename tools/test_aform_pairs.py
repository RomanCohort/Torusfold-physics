"""Does the pairing form when the reconstruction is TOLD about the pairs?

Why this is the test that matters. Two reconstruction paths exist:

  allatom_reconstruct.reconstruct_all_atom(p, seq)
      hand-derived templates, fixed orientation per residue. The docstring of
      aform_from_template says the hand-built geometry "deviates from the amber14
      OL3 force-field equilibrium, so after minimization amber_field stayed
      positive (+70,000 kJ/mol, which is physically unreasonable)".

  aform_from_template.reconstruct_all_atom(p, seq, pairs=None)
      real 1EHZ crystal residues, Kabsch-aligned on (P, C1', C4'), and -- the part
      that matters here -- when `pairs` is given, "a paired residue's
      perpendicular anchor axis points at its partner, so the base faces the base
      it pairs with instead of radiating from the centroid".

The second one already implements what I was about to propose adding: a
per-residue orientation degree of freedom, driven by the pairing. The question
this script answers is whether that is enough for the pair detector to recover
the checkpoint's pairs.

It also carries the honest bound from aform_from_template's own docstring, so
the result is read against it: rotating a base cannot close a POSITIONAL gap --
"a base whose partner sits eight Angstroms away cannot be paired by any twist."

Read-only: writes one file under results/immuno_full/ and nothing else.
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
from torusfold.scheme2.aform_from_template import (  # noqa: E402
    reconstruct_all_atom as reconstruct_aform,
)

RUN = REPO / "results" / "immuno_full"
RING_N = {"N1", "N2", "N3", "N4", "N6", "N7", "N9", "O2", "O4", "O6"}

src = "".join((REPO / "artifacts" / "2013nt" / "sequence.txt").read_text(
    encoding="utf-8").split()).upper()
seq_all = "".join(c for c in src if c in "ACGU")

cg_path = RUN / "_final_cg_for_aa.pdb"
atoms = [l for l in cg_path.read_text(encoding="utf-8", errors="replace").splitlines()
         if l.startswith(("ATOM", "HETATM"))]
P = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in atoms])
L = len(P)
seq = seq_all[:L]
print(f"input : {cg_path.name}  {L} P atoms, {len(seq)} nt")

ck = load_checkpoint_pairing(RUN)
# The checkpoint is 0-based; aform_from_template indexes into the sequence, which
# is 0-based, so these are passed through as-is -- but state the convention
# rather than leaving it implicit.
pairs0 = [(int(p[0]), int(p[1])) for p in ck.pairs]
print(f"pairs : {len(pairs0)} from {ck.source.name} "
      f"(0-based, indices used directly against the sequence)")

out = RUN / "_aform_with_pairs.pdb"

for label, use_pairs in (("pairs=None (radial fallback)", None),
                         (f"pairs=<{len(pairs0)} checkpoint pairs>", pairs0)):
    print()
    print("=" * 78)
    print(f"reconstruct with {label}")
    print("=" * 78)
    try:
        s = reconstruct_aform(P, seq, pairs=use_pairs)
    except Exception as e:  # noqa: BLE001
        print(f"  reconstruct failed: {type(e).__name__}: {e}")
        continue

    names = {a.atom_name for a in s.atoms}
    ring = sorted(names & RING_N)
    print(f"  atoms {len(s.atoms)}  ({len(s.atoms)/L:.1f}/residue)  "
          f"ring atoms {'yes' if ring else 'NO'}")

    if not ring:
        print("  no ring atoms; pair detection cannot run")
        continue

    if use_pairs is not None:
        lines = []
        for i, a in enumerate(s.atoms, 1):
            lines.append(
                f"ATOM  {i:5d} {a.atom_name:<4s} {a.res_name:>3s} A{a.res_seq:4d}    "
                f"{a.xyz[0]:8.3f}{a.xyz[1]:8.3f}{a.xyz[2]:8.3f}  1.00  0.00          "
                f"{a.element:>2s}")
        lines.append("END")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    # pair detection needs a PDB-ish text; build it in memory for either case
    lines = []
    for i, a in enumerate(s.atoms, 1):
        lines.append(
            f"ATOM  {i:5d} {a.atom_name:<4s} {a.res_name:>3s} A{a.res_seq:4d}    "
            f"{a.xyz[0]:8.3f}{a.xyz[1]:8.3f}{a.xyz[2]:8.3f}  1.00  0.00          "
            f"{a.element:>2s}")
    lines.append("END")
    g = build_pair_graph("\n".join(lines) + "\n", is_circular=True)
    ck_set = {(min(i, j), max(i, j)) for i, j in pairs0}
    co_set = {(min(p.key, p.partner), max(p.key, p.partner)) for p in g.pairs}
    reasons = collections.Counter(r["reason"].split(" ")[0] for r in g.rejected)
    print(f"  checkpoint pairs {len(ck_set)}   coordinate pairs {len(co_set)}   "
          f"in both {len(ck_set & co_set)}")
    print(f"  rejection reasons {dict(reasons)}")
    print(f"  helix runs {len(g.helix_runs())}  "
          f"lengths {[r['n_pairs'] for r in g.helix_runs()][:8]}")
    if ck_set:
        print(f"  RECALL on checkpoint pairs: {(len(ck_set & co_set)/len(ck_set)):.1%}")

    # positional gap: the bound aform_from_template says chi cannot close
    miss = sorted(ck_set - co_set)
    if miss:
        ds = []
        for i, j in miss:
            ri, rj = g.residues.get(i), g.residues.get(j)
            if ri is not None and rj is not None:
                ds.append(float(np.linalg.norm(rj.c1p - ri.c1p)))
        if ds:
            a = np.asarray(ds)
            print(f"  predicted-but-not-formed: {len(miss)} pairs, "
                  f"C1'--C1' {a.min():.1f}-{a.max():.1f} A  "
                  f"(median {np.median(a):.1f}; a pair is ~10.4 A)")
            print(f"     of these, within 12 A (a twist could in principle help): "
                  f"{int((a <= 12).sum())}")
            print(f"     beyond 12 A (positional gap; no twist closes it):        "
                  f"{int((a > 12).sum())}")

if out.is_file():
    print()
    print(f"written: {out.relative_to(RUN)}")
