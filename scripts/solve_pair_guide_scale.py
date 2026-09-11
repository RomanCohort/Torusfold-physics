"""The largest K_PAIR_GUIDE that does not move the base-pair minimum by more than one spread."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torusfold.scheme2.torch_cgsim as C

rr = torch.linspace(0.05, 2.5, 20001, dtype=torch.float64)
r = rr.reshape(1, -1, 1)
SD = 0.1103          # nm, the measured within-chain spread of the N-N coordinate


def shift(k_guide):
    e = 0.5 * C.K_PAIR * (rr - C.PAIR_NN) ** 2 + C._sigmoid_f(r, C.PAIR_NN, k_guide, 0.2)[0].reshape(-1)
    return C.PAIR_NN - float(rr[int(torch.argmin(e))])


lo, hi = 0.0, 100.0
for _ in range(80):
    mid = 0.5 * (lo + hi)
    if shift(mid) <= SD:
        lo = mid
    else:
        hi = mid
print(f"criterion: the pair minimum must stay within one measured spread ({SD} nm) of PAIR_NN")
print(f"solves to K_PAIR_GUIDE <= {lo:.3f}")
for k in (lo, 100.0):
    print(f"    K = {k:8.3f} -> minimum at {C.PAIR_NN - shift(k):.4f} nm, "
          f"shift {shift(k) / SD:.3f} spreads")
