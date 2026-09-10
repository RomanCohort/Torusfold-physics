"""Build aform_template.npz from 1EHZ with documented provenance.

The shipped repo has no aform_template.npz and it was never committed
(git log --all -- '*aform_template*' is empty). aform_from_template._load_templates()
needs keys "{base}_names" and "{base}_coords" for base in "AUGC".

Representative residue per base: the FIRST standard A/U/G/C residue in chain A whose
heavy-atom set contains P, C1', C4', O3' and the glycosidic N (N9 for A/G, N1 for C/U).
Which instance is taken matters little for shape: |C4'-P| is 3.90 +/- 0.04 A across all
61 usable residues.
"""
import collections, hashlib, json
from pathlib import Path
import numpy as np

PDB = Path(r"D:\Torusfold-Templates\_tools\1ehz.pdb")
OUT = Path(r"D:\torusfold-hybrid\src\torusfold\scheme2\aform_template.npz")

res = collections.OrderedDict()
order = []
for line in open(PDB):
    if not line.startswith("ATOM"):
        continue
    if line[16] not in (" ", "A"):
        continue
    name = line[12:16].strip()
    key = (line[21], line[22:27].strip(), line[17:20].strip())
    try:
        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    except ValueError:
        continue
    if key not in res:
        res[key] = []
        order.append(key)
    res[key].append((name, xyz))

need = {"P", "C1'", "C4'", "O3'"}
chosen, arrays, found = {}, {}, set()
for base, glyco in (("A", "N9"), ("U", "N1"), ("G", "N9"), ("C", "N1")):
    for key in order:
        chain, seq, rname = key
        if chain != "A" or rname != base:
            continue
        names = [n for n, _ in res[key]]
        if not need.issubset(set(names)) or glyco not in names:
            continue
        chosen[base] = {"chain": chain, "resSeq": seq, "n_atoms": len(names)}
        arrays[base + "_names"] = np.array(names)
        arrays[base + "_coords"] = np.array([c for _, c in res[key]], dtype=np.float32)
        found.add(base)
        break

missing = [b for b in "AUGC" if b not in found]
if missing:
    raise SystemExit("missing bases: " + str(missing))
np.savez(OUT, **arrays)
print("written:", OUT)
print("bytes  :", OUT.stat().st_size)
print("sha256 :", hashlib.sha256(OUT.read_bytes()).hexdigest())
print("chosen :", json.dumps(chosen, ensure_ascii=False))
for b in "AUGC":
    print("  " + b + ": " + str(len(arrays[b + "_names"])) + " atoms")
