"""Where does the P(0)-P(L-1) coordinate actually sit? Two shipped targets disagree about it."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torusfold.scheme2.torch_cgsim as C

rr = torch.linspace(0.0, 2.5, 25001, dtype=torch.float64)
r = rr.reshape(1, -1, 1)
print(f"K_BSJ {C.K_BSJ} at target BOND_P_NEXT {C.BOND_P_NEXT}; "
      f"K_BSJ_GUIDE {C.K_BSJ_GUIDE} with r0 = PAIR_NN {C.PAIR_NN}")


def curve(k_bsj, k_guide):
    e = 0.5 * k_bsj * (rr - C.BOND_P_NEXT) ** 2
    if k_guide:
        e = e + C._sigmoid_f(r, C.PAIR_NN, k_guide, 0.2)[0].reshape(-1)
    return e


print()
print(f"{'K_BSJ':>8s} {'K_BSJ_GUIDE':>12s} {'minimum at r':>13s} {'E there':>10s} "
      f"{'gradient at 0.59':>17s}")
print("-" * 68)
for kb, kg in ((600.0, 0.0), (600.0, 100.0), (1122.4, 0.0), (1122.4, 100.0), (1122.4, 20.0)):
    e = curve(kb, kg)
    i = int(torch.argmin(e))
    j = int(torch.argmin(torch.abs(rr - C.BOND_P_NEXT)))
    grad = (e[j + 1] - e[j - 1]) / (rr[j + 1] - rr[j - 1])
    print(f"{kb:8.1f} {kg:12.1f} {float(rr[i]):13.4f} {float(e[i]):10.2f} {float(grad):17.2f}")
print()
print("A minimum at r = 0 means the closure is driven to a point overlap: the guide's pull")
print("saturates at K/0.2 away but never vanishes, while the harmonic's restoring force is what")
print("would have to grow to hold it at BOND_P_NEXT.")
print()
print("For reference, what the equilibrated runs actually reach: the diagnostics report the")
print("deepest bead pair, not this coordinate, so the value here is the LANDSCAPE, not a sample.")
