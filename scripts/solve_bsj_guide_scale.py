"""Set K_BSJ_GUIDE by the one-spread criterion, and K_BSJ by transferability."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torusfold.scheme2.torch_cgsim as C

rr = torch.linspace(0.0, 2.5, 50001, dtype=torch.float64)
r = rr.reshape(1, -1, 1)
SPREAD = 0.0470        # nm, the measured spread of a P-P phosphodiester bond


def r_min(k_bsj, k_guide):
    e = 0.5 * k_bsj * (rr - C.BOND_P_NEXT) ** 2
    if k_guide:
        e = e + C._sigmoid_f(r, C.PAIR_NN, k_guide, 0.2)[0].reshape(-1)
    return float(rr[int(torch.argmin(e))])


K_BSJ_NEW = 1122.4
lo, hi = 0.0, 100.0
for _ in range(70):
    mid = 0.5 * (lo + hi)
    if abs(r_min(K_BSJ_NEW, mid) - C.BOND_P_NEXT) <= SPREAD:
        lo = mid
    else:
        hi = mid
print(f"K_BSJ = {K_BSJ_NEW} (transferability: the BSJ IS a 3'-5' phosphodiester bond)")
print(f"criterion: the closure minimum must stay within one bond spread ({SPREAD} nm) of "
      f"BOND_P_NEXT = {C.BOND_P_NEXT}")
print(f"solves to K_BSJ_GUIDE <= {lo:.4f}")
print()
for kg in (100.0, lo, 0.0):
    m = r_min(K_BSJ_NEW, kg)
    print(f"    K_BSJ_GUIDE {kg:8.3f} -> minimum {m:.4f} nm, shift {abs(m - C.BOND_P_NEXT):.4f} nm "
          f"= {abs(m - C.BOND_P_NEXT) / SPREAD:.2f} spreads")
print()
print("the control: with the guide off the minimum is exactly BOND_P_NEXT for either K_BSJ")
for kb in (600.0, K_BSJ_NEW):
    print(f"    K_BSJ {kb:8.1f}, guide 0 -> {r_min(kb, 0.0):.4f} nm")
