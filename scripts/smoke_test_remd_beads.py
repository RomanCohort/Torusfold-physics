"""Smoke test: does BatchedREMD2D still initialise and step after the bead change?"""
import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import truth_1ehz
from torusfold.scheme2.torch_cgsim import BatchedREMD2D

order, res, base_of, ps, partner, meta = truth_1ehz.load()
n = 24
seq = "".join(base_of[:n])
p = ps[:n]
pairs = [(a, b, 1.0) for a, b in partner.items() if a < n and b < n and a < b]
print(f"test system: {n} residues, {len(pairs)} pairs")

sig = inspect.signature(BatchedREMD2D.__init__)
print("init params:", ", ".join(list(sig.parameters)[1:]))
print()
try:
    remd = BatchedREMD2D(n_t=2, t_lo=300.0, t_hi=1000.0, lambdas=(1.0, 0.8),
                         exchange_interval=5, sequence=seq)
    print("constructed ok")
    out = remd.run(p, pairs, n_steps=5, verbose=False)
    coords, e, diag = out[0], out[1], out[2]
    print(f"ran 5 steps ok; energy {float(e):.1f}")
    if isinstance(diag, dict):
        for k in ("tri_force_injected", "tri_scores"):
            if k in diag:
                print(f"  diag[{k!r}] = {diag[k]}")
    print(f"best coords shape {np.asarray(coords).shape}")
except Exception as exc:
    print(f"FAILED: {type(exc).__name__}: {exc}")
    raise
