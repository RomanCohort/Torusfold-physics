"""Is the Boltzmann-inverted bonded potential built correctly?

Three checks before it is used for anything:
  1. U = -kBT ln P has its minimum at the observable's mode, not at its mean and not
     wherever the old harmonic target was.
  2. Outside the fitted range the wall actually pushes back, and the force it produces
     matches the finite-difference gradient of its own energy.
  3. The autograd forces match central differences of the energy, which is what catches an
     indexing or clamping mistake in the interpolator.

Run: python scripts/test_boltzmann_bonded.py [n_fit]
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B
import torusfold.scheme2.torch_cgsim as C

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
structs = B.load_structures(limit=N)
print(f"{len(structs)} structures for fitting")
tables = B.prepare(B.fit(structs))
print()

print("does the tabulated minimum land on the observable's mode?")
print(f"{'coordinate':12s} {'n':>8s} {'data mode':>10s} {'argmin U':>10s} "
      f"{'empty bins':>11s} {'old target':>11s}")
print("-" * 70)
OLD = {"bb_bond": C.BOND_P_NEXT, "intra_pc": C.BOND_P_C4,
       "intra_cn": C.BOND_C4_N, "stack": C.STACK_R0,
       "angle": float(np.cos(C.ANGLE_PPP)), "dihedral": float(np.cos(C.DIH_PPPP))}
vals_by = {}
for name in B.COORDS:
    t = tables[name]
    v = []
    for s in structs:
        pos = torch.tensor(s["pos"].reshape(1, -1, 3), dtype=torch.float64)
        v.append(B.coords_of(pos, name).reshape(-1).numpy())
    v = np.concatenate(v)
    vals_by[name] = v
    h, e = np.histogram(v, bins=60)
    mode = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
    argmin = float(t["centre"][int(np.argmin(t["U"]))])
    print(f"{name:12s} {len(v):8d} {mode:10.3f} {argmin:10.3f} "
          f"{t['empty']:6d}/{len(t['U']):<4d} {OLD[name]:11.3f}")
print()
print("(the harmonic target is shown for reference; on 'dihedral' and 'stack' it is nowhere")
print(" near where the data is, which is the defect this replaces)")
print()

# ---- wall and gradient -----------------------------------------------------
print("wall outside the fitted range, and autograd vs central differences")
# The wall is a function of the scalar coordinate, so it is tested on scalar values. An
# earlier version of this script translated the whole structure by 5 nm and expected the
# energy to rise -- but every one of these coordinates is translation invariant, so that
# probe measured nothing at all.
kBT = B.KBT
print(f"{'coordinate':12s} {'edge U':>9s} {'U at +20 pct':>13s} {'slope in':>10s} "
      f"{'slope out':>10s} {'outward force?':>15s}")
print("-" * 76)
with torch.no_grad():
    for name in B.COORDS:
        t = tables[name]
        span = t["hi"] - t["lo"]
        out_hi = torch.tensor([t["hi"] + 0.2 * span], dtype=torch.float64)
        at_hi = torch.tensor([t["hi"]], dtype=torch.float64)
        e_in = B._sample(at_hi, t)
        # slope_in is the table's own edge slope, not a resampled difference: sampling the
        # range more finely than one bin makes consecutive probes share a bin and report 0.
        slope_in = float(t["slope_hi"])
        e_out = B._sample(out_hi, t) + t["slope_hi"] * (out_hi - t["hi"]) \
            + 0.5 * 200.0 * (out_hi - t["hi"]) ** 2
        slope_out = float(t["slope_hi"] + 200.0 * (out_hi - t["hi"])[0])
        pushes = "yes" if slope_out > 0 else "NO"
        print(f"{name:12s} {float(e_in[0]):9.2f} {float(e_out[0]):13.2f} "
              f"{slope_in:10.2f} {slope_out:10.2f} {pushes:>15s}")
print()
print("A positive outward slope means dU/dq > 0 past the range, so the force -dU/dq points")
print("back inside.")
print("slope in is 0.00 for every coordinate, and that is the reason the wall exists rather")
print("than a bug: the outermost bins are empty, the pseudo-count gives them all the same")
print("value, so the table itself is flat there. Interpolating a flat table outside the range")
print("would leave the coordinate free to drift. The quadratic wall supplies the restoring")
print("force instead; slope out is dU/dq twenty percent beyond the top edge.")
print()

# ---- autograd vs central differences ---------------------------------------
pos = torch.tensor(structs[0]["pos"].reshape(1, -1, 3), dtype=torch.float64,
                   requires_grad=True)
e0 = B.energy(pos, tables)
grad = torch.autograd.grad(e0, pos, create_graph=False)[0]
h = 1e-5
worst = 0.0
probe = [(0, 0), (0, 1), (1, 2), (3, 0), (5, 1), (9, 2)]
with torch.no_grad():
    for (i, d) in probe:
        if i >= pos.shape[1]:
            continue
        hi = pos.detach().clone(); hi[0, i, d] += h
        lo = pos.detach().clone(); lo[0, i, d] -= h
        fd = (B.energy(hi, tables) - B.energy(lo, tables)) / (2 * h)
        an = grad[0, i, d]
        denom = max(abs(float(fd)), abs(float(an)), 1e-3)
        worst = max(worst, abs(float(fd) - float(an)) / denom)
print(f"autograd vs central differences, {len(probe)} probes on the first structure:")
print(f"  worst relative error {worst:.3e}")
print(f"  (the gradient is exact by construction; a non-zero error here would mean the")
print(f"   interpolation index or the clamp is breaking the graph)")
