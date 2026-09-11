"""What the guide sign fix changed: the force curve, and where the pair minimum sits now."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torusfold.scheme2.torch_cgsim as C

K = C.K_PAIR_GUIDE
W = 0.2
rr = torch.linspace(0.05, 4.0, 4001, dtype=torch.float64)
r = rr.reshape(1, -1, 1)

print(f"K_PAIR_GUIDE = {K}, width = {W}, saturation force = K/width = {K / W:.1f} kJ/mol/nm")
print()
print("the guide's own force, new shape vs the old closed form (old = K/width * sigmoid((r0-r)/w))")
print(f"{'r (nm)':>8s} {'new':>10s} {'old':>10s}   ratio")
print("-" * 44)
for x in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0):
    new = K / W * torch.sigmoid(torch.tensor((x - C.PAIR_NN) / W, dtype=torch.float64)).item()
    old = K / W * torch.sigmoid(torch.tensor((C.PAIR_NN - x) / W, dtype=torch.float64)).item()
    print(f"{x:8.1f} {new:10.2f} {old:10.2f}   {new / old if old else float('inf'):.4g}")
print()
print("the old shape is strongest where the pair is too close and does nothing far away; the new one")
print("is the reverse. That is the whole content of the fix.")
print()
print("where the base-pair minimum sits, restraint + guide, new shape:")
print(f"{'K_PAIR_GUIDE':>13s} {'minimum (nm)':>13s} {'shift (nm)':>11s} {'spreads':>8s}")
print("-" * 50)
SD = 0.1103
for kg in (0.0, K, 13.4, 100.0):
    e = (0.5 * C.K_PAIR * (rr - C.PAIR_NN) ** 2
         + C._sigmoid_f(r, C.PAIR_NN, kg, W)[0].reshape(-1))
    m = float(rr[int(torch.argmin(e))])
    print(f"{kg:13.2f} {m:13.4f} {C.PAIR_NN - m:11.4f} {(C.PAIR_NN - m) / SD:8.3f}")
print()
print("the K = 0 row is the control: without the guide the minimum is exactly PAIR_NN.")
print()
print("and the closure coordinate, which has its own guide:")
for kg in (0.0, C.K_BSJ_GUIDE, 100.0):
    e = (0.5 * C.K_BSJ * (rr - C.BOND_P_NEXT) ** 2
         + C._sigmoid_f(r, C.PAIR_NN, kg, W)[0].reshape(-1))
    m = float(rr[int(torch.argmin(e))])
    print(f"    K_BSJ_GUIDE {kg:8.2f} -> {m:.4f} nm, shift {abs(m - C.BOND_P_NEXT):.4f} nm")
