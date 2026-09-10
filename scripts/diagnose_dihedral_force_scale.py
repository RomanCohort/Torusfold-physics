"""How large does the exact dihedral gradient get, and why?

After _dihedral_f's force was replaced with the exact gradient, the per-term check passed
at 2.9e-07 but the term's largest force on a 43-residue structure jumped from 1729.60 to
31132.49 kJ/mol/nm. The energy expression did not change, so the new force is the true
gradient of the old energy, and the old force was simply not it. The question is whether
31132 is one pathological window or a systematic property.

The suspicion is systematic, and it is about the functional form rather than the gradient.
E = 0.5*k*(cos(phi) - c)^2 has gradient magnitude going like k*|cos(phi) - c| / (b*sin(phi)),
which diverges as phi approaches 0 or 180 degrees. The recalibration in 3m moved c from
-1.0 to +0.975, i.e. the target from one singular point (180 degrees) to the other (0
degrees) -- which is where the data actually sits.

Run: python scripts/diagnose_dihedral_force_scale.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import boltzmann_bonded as B
import torusfold.scheme2.torch_cgsim as C

structs = B.load_structures(limit=8)


def windows(pos_raw):
    """Per four-atom window: the signed pseudo-torsion and the force it carries."""
    p = torch.tensor(pos_raw.reshape(1, -1, 3), dtype=torch.float64)
    L = p.shape[1] // 3
    P = torch.arange(L) * 3
    a = p[:, P[1:-2]] - p[:, P[:-3]]
    b = p[:, P[2:-1]] - p[:, P[1:-2]]
    c = p[:, P[3:]] - p[:, P[2:-1]]
    n0 = torch.linalg.cross(a, b, dim=-1)
    n1 = torch.linalg.cross(b, c, dim=-1)
    cos = (n0 * n1).sum(-1) / (torch.linalg.norm(n0, dim=-1) * torch.linalg.norm(n1, dim=-1))
    _, F = C._dihedral_f(p, C.K_DIH, np.cos(C.DIH_PPPP))
    Lc = L
    fw = []
    for i in range(L - 3):
        idx = [3 * (i + k) for k in range(4)]
        fw.append(float(torch.linalg.norm(F[0, idx], dim=-1).max()))
    return cos[0].numpy(), np.array(fw)


print(f"target cos(DIH_PPPP) = {np.cos(C.DIH_PPPP):+.3f}  "
      f"(the old target was {np.cos(math.pi):+.3f})" if False else
      f"target cos(DIH_PPPP) = {np.cos(C.DIH_PPPP):+.3f}")
print()
print(f"{'structure':10s} {'L':>4s} {'|cos| min':>10s} {'|F| p50':>10s} {'|F| p95':>10s} "
      f"{'|F| max':>11s} {'max/sin':>10s}")
print("-" * 72)
allcos, allf = [], []
for s in structs:
    cos, f = windows(s["pos"])
    if len(f) == 0:
        continue
    allcos.append(cos)
    allf.append(f)
    worst = int(np.argmax(f))
    sing = 1.0 / max(np.sqrt(max(1.0 - cos[worst] ** 2, 0.0)), 1e-6)
    print(f"{s['name']:10s} {len(s['pos']):4d} {np.abs(np.abs(cos)-1).min():10.4f} "
          f"{np.percentile(f,50):10.1f} {np.percentile(f,95):10.1f} {f.max():11.1f} "
          f"{sing:10.1f}")
allcos = np.concatenate(allcos)
allf = np.concatenate(allf)
print()
print(f"across {len(structs)} structures and {len(allf)} dihedral windows:")
print(f"  |cos| within 1e-4 of 1 (phi near 0 or 180): {int((np.abs(np.abs(allcos)-1) < 1e-4).sum())}")
print(f"  |cos| within 1e-3 of 1:                     {int((np.abs(np.abs(allcos)-1) < 1e-3).sum())}")
print(f"  |F| over the 200 kJ/mol/nm cap:             {int((allf > 200).sum())} "
      f"({(allf > 200).mean()*100:.1f} percent)")
print(f"  |F| over 1000:                              {int((allf > 1000).sum())}")
print(f"  |F| median {np.median(allf):.1f}, max {allf.max():.1f}")
print()
print("For a restraint quadratic in cos(phi), |dE/dx| goes like k|cos(phi)-c|/(b sin(phi)).")
print("A restraint written on the angle phi instead does not carry that 1/sin factor.")
