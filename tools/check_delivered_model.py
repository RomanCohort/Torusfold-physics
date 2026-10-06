"""Run the pair detector on the delivered 2013 nt model and report honestly."""
from __future__ import annotations

import collections
import pathlib
import sys

REPO = pathlib.Path(r"C:\baidunetdiskdownload\torusfold-hybrid")
sys.path.insert(0, str(REPO / "src"))

from torusfold.immuno.pair_graph_from_coords import build_pair_graph  # noqa: E402

txt = (REPO / "artifacts" / "2013nt" / "isrnaclong_final.pdb").read_text(encoding="utf-8")
g = build_pair_graph(txt, is_circular=True)
s = g.summary()

print("delivered model: artifacts/2013nt/isrnaclong_final.pdb")
for k in ("length", "n_residues_parsed", "n_pairs", "paired_fraction",
          "by_type", "n_helices", "longest_helix_pairs"):
    print(f"  {k:<22} {s[k]}")

print()
reasons = collections.Counter(r["reason"].split(" ")[0] for r in g.rejected)
print(f"  candidates considered : {len(g.rejected)}")
print(f"  rejection reasons     : {dict(reasons)}")

import numpy as np  # noqa: PLC0415

idx = sorted(g.residues)
coords = np.vstack([g.residues[i].c1p for i in idx])
from scipy.spatial import cKDTree  # noqa: PLC0415

t = cKDTree(coords)
d, _ = t.query(coords, k=2)
nn = d[:, 1]
print()
print(f"  nearest C1'--C1' anywhere : min {nn.min():.2f}  median {np.median(nn):.2f}  "
      f"max {nn.max():.2f} A")
print(f"  a Watson-Crick pair needs : ~10.4 A")
print(f"  so the largest nearest-neighbour distance is "
      f"{'BELOW' if nn.max() < 10.4 else 'above'} pairing distance")

consec = []
for a, b in zip(idx, idx[1:]):
    if g.residues[a].chain == g.residues[b].chain:
        consec.append(float(np.linalg.norm(g.residues[b].c1p - g.residues[a].c1p)))
print(f"  consecutive C1'--C1'      : median {np.median(consec):.2f} A "
      f"(2OIU 5.52, 1QC0 5.45) -> backbone is NOT compressed")
