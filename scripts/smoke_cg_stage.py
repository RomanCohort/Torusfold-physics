"""Does the CG stage the pipeline actually calls still run, with real weights?

Every measurement in this arc has been on scripts. The pipeline calls BatchedREMD2D
(isrnaclong.py:2024), which builds its pair tensor and pair weights at torch_cgsim.py:1942,
so this drives that same entry point with the same shape of input the pipeline passes: P-only
coordinates in Angstrom, and a list of (i, j, w) triples.

Reduced to a few replicas and a few hundred steps, because the point is whether it runs,
returns finite coordinates and reports diagnostics, not whether it folds.

Run: python scripts/smoke_cg_stage.py [n_steps]
"""
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B
from torusfold.scheme2.torch_cgsim import BatchedREMD2D

N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
DATA = Path(r"D:\torusfold-cgdata\rsRNASP\Training_set")

s = next(x for x in B.load_structures(limit=200) if len(x["pairs"]) >= 8)

# The sequence is required, not optional: real_cg_beads raises rather than fabricating beads
# when it is missing, and BatchedREMD2D calls it on the first run. isrnaclong.py passes
# sequence=... so the pipeline is fine, but any caller that omits it fails here.
ch = collections.OrderedDict()
for line in open(DATA / (s["name"] + ".pdb")):
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        continue
    if line[16] not in (" ", "A") or line[17:20].strip() not in ("A", "C", "G", "U"):
        continue
    ch.setdefault(line[21], collections.OrderedDict())[line[22:27].strip()] = \
        line[17:20].strip()
seq = max(("".join(v.values()) for v in ch.values()), key=len)
print(f"structure {s['name']}, L = {len(s['pos'])}, {len(s['pairs'])} WC pairs, "
      f"sequence length {len(seq)}")
assert len(seq) == len(s["pos"]), "sequence and bead count disagree"

coords_A = s["pos"][:, 0, :] * 10.0          # nm -> Angstrom, P only
rng = np.random.default_rng(7)
pairs = [(i, j, float(rng.choice([0.0, 0.25, 0.8, 1.0], p=[0.4, 0.3, 0.2, 0.1])))
         for i, j in s["pairs"]]
print(f"weights in the rough shape the RCM measurement gave: "
      f"{sorted(set(round(p[2], 2) for p in pairs))}")
print()

remd = BatchedREMD2D(
    n_t=2, t_lo=300.0, t_hi=600.0,
    lambdas=(1.0, 0.7),
    exchange_interval=100,
    sequence=seq,
    force_refresh_freq=100,
    relax_bond_k=500.0, relax_angle_k=200.0,
    relax_pair_k=500.0, restraint_k=500.0,
)
print("BatchedREMD2D constructed")
out = remd.run(coords_A, pairs, n_steps=N_STEPS, verbose=True)
print()
print(f"run returned a {len(out)}-tuple")
coords = np.asarray(out[0])
print(f"  coords shape {coords.shape}, dtype {coords.dtype}")
print(f"  all finite: {bool(np.all(np.isfinite(coords)))}")
print(f"  energy: {out[1]}")
if len(out) > 2 and isinstance(out[2], dict):
    for k in list(out[2])[:14]:
        print(f"  diag[{k}] = {out[2][k]}")
