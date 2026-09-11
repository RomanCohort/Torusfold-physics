"""Where is the base pair's effective equilibrium once K_PAIR_GUIDE is switched on?

_sigmoid_f(dist, r0, k, width) is E = -k * softplus((r0 - dist)/w). Its gradient is

    dE/dr = +k * sigmoid((r0 - r)/w) / w

which is POSITIVE for every r, i.e. the force is inward (toward smaller r) at every separation.
At r = r0 it is k/(2w), and as r falls below r0 it saturates at k/w. So this is not a term that
helps a distant pair find itself and then gets out of the way: it pulls hardest exactly when the
pair is already too close, and it does not vanish anywhere.

The base-pair restraint is the harmonic 0.5 * K_PAIR * (r - PAIR_NN)^2, whose own gradient is
K_PAIR * (r - r0). Where the two are summed the minimum is no longer at PAIR_NN. This solves for
it numerically rather than by hand, for several K_PAIR_GUIDE.

The same scan is done with the Library's own two functions rather than with a transcription, so a
future edit to _sigmoid_f cannot leave this measurement describing something else.

Run: python scripts/measure_pair_equilibrium.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

rr = torch.linspace(0.05, 2.5, 4001, dtype=torch.float64)      # (M,)
r = rr.reshape(1, -1, 1)                                        # (1, M, 1) for _sigmoid_f


def total(k_guide):
    """The pair coordinate's energy, restraint + guide, on the same grid."""
    e_wc = 0.5 * C.K_PAIR * (rr - C.PAIR_NN) ** 2
    e_g, _sig = C._sigmoid_f(r, C.PAIR_NN, k_guide, 0.2)
    return e_wc + e_g.reshape(-1)


def r_min(k_guide):
    return float(rr[int(torch.argmin(total(k_guide)))])

print(f"PAIR_NN = {C.PAIR_NN} nm, K_PAIR = {C.K_PAIR}, width = 0.2, shipped K_PAIR_GUIDE = "
      f"{C.K_PAIR_GUIDE}")
print()
print(f"{'K_PAIR_GUIDE':>13s} {'E minimum at r':>15s} {'shift from PAIR_NN':>19s} "
      f"{'E at the minimum':>18s} {'E at PAIR_NN':>14s}")
print("-" * 86)
for kg in (0.0, 1.0, 5.0, 13.4, 25.0, 50.0, 100.0):
    e = total(kg)
    rmin = r_min(kg)
    j = int(torch.argmin(torch.abs(rr - C.PAIR_NN)))
    tag = "   <- shipped" if kg == C.K_PAIR_GUIDE else ""
    print(f"{kg:13.2f} {rmin:15.4f} {rmin - C.PAIR_NN:+19.4f} {float(e.min()):18.2f} "
          f"{float(e[j]):14.2f}{tag}")
print()
print("K_PAIR_GUIDE = 0 recovers PAIR_NN exactly, which is the check that this scan is measuring")
print("the pair coordinate and nothing else. Every nonzero value moves the minimum, and the shipped")
print("100 moves it further than any of the values between.")
print()
# where does the shipped setting put it, in units of the database's own spread
sd = 0.1103
print(f"for scale: the measured within-chain spread of this coordinate is {sd} nm")
for kg in (0.0, 100.0):
    rmin = r_min(kg)
    print(f"    K_PAIR_GUIDE {kg:6.1f}: minimum at {rmin:.4f} nm, "
          f"{(C.PAIR_NN - rmin) / sd:.2f} database spreads from PAIR_NN")
print()
print("The guide's own force at the minimum, against the restraint's:")
for kg in (13.4, 100.0):
    rmin = r_min(kg)
    f_wc = abs(C.K_PAIR * (rmin - C.PAIR_NN))
    f_g = kg / 0.2 * float(C._sigmoid_f(torch.tensor([[[rmin]]], dtype=torch.float64),
                                        C.PAIR_NN, 1.0, 0.2)[1])
    print(f"    K_PAIR_GUIDE {kg:6.1f} at r = {rmin:.4f}: restraint {f_wc:8.1f} outward, "
          f"guide {f_g:8.1f} inward, net {f_g - f_wc:+8.1f}")
