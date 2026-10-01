"""Precompute 2OIU's sequence, secondary structure and base-pair list for the A/B benchmark.

Kept separate because the ROCm torch environment (which has the GPU) does not have ViennaRNA.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from openmm.app import PDBFile

pdb = PDBFile(str(REPO / "artifacts" / "2oiu" / "2OIU.pdb"))
positions = pdb.positions
seq_chars, p_coords = [], []
for res in pdb.topology.residues():
    if res.chain.id != "P":
        continue
    base = {"RA": "A", "RU": "U", "RG": "G", "RC": "C"}.get(res.name.strip(), res.name.strip()[:1])
    if base not in "ACGU":
        continue
    seq_chars.append(base)
    for atom in res.atoms():
        if atom.name.strip() == "P":
            q = positions[atom.index]
            p_coords.append([q.x, q.y, q.z])
seq = "".join(seq_chars)
p_coords = np.asarray(p_coords, dtype=float)

import ViennaRNA
fc = ViennaRNA.fold_compound(seq)
ss = fc.mfe()[0]
fc.pf()
bpp = fc.bpp()
pairs = [[i - 1, j - 1, float(bpp[i][j])] for i in range(1, len(seq) + 1)
         for j in range(i + 1, len(seq) + 1) if bpp[i][j] > 0.1]

out = REPO / "results" / "plan_c" / "_2oiu_input.json"
out.write_text(json.dumps({"seq": seq, "ss": ss, "pairs": pairs,
                           "p_coords_nm": p_coords.tolist()}), encoding="utf-8")
print("2OIU:", len(seq), "nt, ss", ss, ", pairs", len(pairs), "->", out.name)
print("BSJ |P0 - P_last| =", round(float(np.linalg.norm(p_coords[0] - p_coords[-1])), 4), "nm")
