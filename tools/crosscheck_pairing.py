"""Cross-check the coordinate-derived pair graph against the pipeline's own pairing.

The two are independent statements about the same molecule:

  checkpoint   what the secondary-structure stage DECIDED the pairing is, from
               sequence and ViennaRNA, before any 3D work.
  coordinates  what the final structure ACTUALLY does, read geometrically.

They will not agree exactly, and the disagreement is the interesting part. A pair
the checkpoint asserts but the coordinates do not form is a restraint that did not
take. A pair the coordinates form but the checkpoint never listed is a contact the
sequence predictor missed. Either is a real observation about the model.

Both sides are 1-based here: the checkpoint reader converts on load, and the pair
graph uses PDB author numbering.
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(r"C:\baidunetdiskdownload\torusfold-hybrid")
os.environ.setdefault("ISRNACIRC_BIN_DIR", str(REPO / "_isrnacirc_ascii" / "bin"))
os.environ.setdefault("ISRNACIRC_ROOT", str(REPO / "_isrnacirc_ascii"))
sys.path.insert(0, str(REPO / "src"))

from torusfold.immuno.checkpoint_pairs import load_checkpoint_pairing  # noqa: E402
from torusfold.immuno.pair_graph_from_coords import build_pair_graph  # noqa: E402

RUN = REPO / "results" / "immuno_run"
AA = RUN / "cg2aa" / "_test_aa.pdb"

src = "".join((REPO / "artifacts" / "2013nt" / "sequence.txt").read_text(
    encoding="utf-8").split()).upper()
seq = "".join(c for c in src if c in "ACGU")[:200]

ck = load_checkpoint_pairing(RUN, sequence=seq)
print("=== checkpoint (its own prediction) ===")
print(json.dumps(ck.summary(), indent=2, ensure_ascii=False))
print()

g = build_pair_graph(AA.read_text(encoding="utf-8"), is_circular=True)
print("=== coordinate detector on the all-atom structure ===")
print(json.dumps(g.summary(), indent=2, ensure_ascii=False))
print()

ck_set = {(min(i, j), max(i, j)) for i, j, _ in ck.pairs}
co_set = {(min(p.key, p.partner), max(p.key, p.partner)) for p in g.pairs}
both = ck_set & co_set
only_ck = ck_set - co_set
only_co = co_set - ck_set

print("=== agreement ===")
print(f"  checkpoint pairs            : {len(ck_set)}")
print(f"  coordinate pairs            : {len(co_set)}")
print(f"  in both                     : {len(both)}")
print(f"  checkpoint only (no contact): {len(only_ck)}")
print(f"  coordinates only (unlisted) : {len(only_co)}")
if ck_set:
    print(f"  recall                      : {len(both)/len(ck_set):.1%}")
if co_set:
    print(f"  precision                   : {len(both)/len(co_set):.1%}")
print()

print("  examples, checkpoint only:")
for i, j in sorted(only_ck)[:8]:
    ri, rj = g.residues.get(i), g.residues.get(j)
    if ri is None or rj is None:
        print(f"    {i:4d}-{j:<4d}  C1'--C1' = n/a (residue absent)")
        continue
    d = float(np.linalg.norm(rj.c1p - ri.c1p))
    print(f"    {i:4d}-{j:<4d}  C1'--C1' = {d:6.2f} A")
print()
print("  examples, coordinates only:")
for i, j in sorted(only_co)[:8]:
    print(f"    {i:4d}-{j:<4d}  {seq[i-1]}-{seq[j-1]}")
print()

# helices: the feature that actually matters
print("=== helices ===")
print(f"  checkpoint stem_blocks : {len(ck.stem_blocks)}  "
      f"lengths {ck.helix_lengths()}")
runs = g.helix_runs()
print(f"  geometric helix runs   : {len(runs)}  "
      f"lengths {[r['n_pairs'] for r in runs]}")
print()
print("  longest geometric run:")
if runs:
    r = runs[0]
    print(f"    {r['n_pairs']} pairs  key {r['key_start']}..{r['key_end']}  "
          f"partner {r['partner_start']}..{r['partner_end']}")
    print(f"    (PKR needs ~30 bp; anything here is far below)")
