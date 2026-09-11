"""Does the integrator survive with the cap off?

The cap was raised in effect by the K_INTRA split: a matched harmonic has an equilibrium force
scale of sqrt(k*kBT), which is 227.5 kJ/mol/nm for P-C4' and 301.3 for C4'-N, both above the cap
of 200. Measured with the cap off, 40 percent of native beads and 63 percent of trajectory
samples exceed it. So the cap is clipping most of the field again.

Before it can be raised or removed, the integrator has to survive without it. Both half-kicks use
the uncapped force here; the earlier headroom measurement used the capped force for the first
kick, which is a flaw in that script rather than in the answer.

Run: python scripts/check_uncapped_stability.py [n_steps] [friction]
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

NSTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
FRICTION = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
DT = 0.002

pool = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8]
s0 = pool[0]
L = len(s0["pos"])
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
NREP = 4
pos = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64).repeat(NREP, 1, 1)
vel = torch.zeros_like(pos)
temps = torch.full((NREP,), 300.0, dtype=torch.float64)
torch.manual_seed(20260221)


def _forces_at(p, cap):
    cl2 = C.GPUCellList(cell_size=1.5)
    cl2.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl2, force_cap=cap)[1]


def run(cap, label):
    x = pos.clone()
    v = torch.zeros_like(x)
    fmax = 0.0
    vmax = 0.0
    dmin = 1e9
    blow = False
    for step in range(NSTEPS):
        try:
            with torch.no_grad():
                f = _forces_at(x, cap)
                x, v = C.batch_langevin_step(x, v, f, temps, dt_ps=DT,
                                             mass_amu=110.0, friction=FRICTION,
                                             force_fn=lambda p: _forces_at(p, cap))
        except Exception as exc:
            blow = True
            print(f"  {label}: raised {type(exc).__name__} at step {step}: "
                  f"{str(exc)[:90]}")
            break
        if step % 20 == 0:
            with torch.no_grad():
                fmax = max(fmax, float(torch.linalg.norm(f.reshape(-1, 3), dim=-1).max()))
                vmax = max(vmax, float(torch.linalg.norm(v.reshape(-1, 3), dim=-1).max()))
                beads = x.reshape(NREP, -1, 3)
                n = beads.shape[1]
                dd = torch.cdist(beads, beads) + torch.eye(n, device=beads.device) * 10.0
                dmin = min(dmin, float(dd.min()))
    print(f"  {label:22s} steps {NSTEPS}  blown up {blow}  "
          f"max|F| {fmax:9.2f}  max|v| {vmax:7.4f} nm/ps  min bead {dmin:.4f} nm")
    return not blow


print(f"{s0['name']}, L={L}, {NREP} replicas, {NSTEPS} steps of {DT} ps, friction {FRICTION}")
print(f"thermal |v| for 110 amu at 300 K = {np.sqrt(2.494 / 100 / 110):.4f} nm/ps")
print()
print("=== cap at 200 (as shipped) vs no cap ===")
run(200.0, "cap 200")
run(None, "no cap")
run(20000.0, "cap 20000")
print()
print("A run that stays finite with no cap means the cap can be a blow-up guard rather than a")
print("force law. A run that dies without it says the cap is load-bearing and the field has to")
print("be brought under it instead -- which for the intra bonds would mean abandoning the")
print("measured kBT/sigma^2 value, so that is a decision and not a tweak.")
