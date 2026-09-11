"""Is the chain collapsing, or is the integrator heating it?

Every explanation for the collapse has been excluded by measurement: the old linear repulsion, the
new diverging one, K_CLASH set to zero, the force cap at 200 and at 5000 and off, the BSJ closure
and guide terms zeroed, the pair guide zeroed. In all of them the closest approach ends up between
0.01 and 0.1 nm where real structures never go below 0.3975.

One number has been staring the whole time and never read properly: with the cap off the fastest
bead moves at 4.03 nm/ps against a thermal 0.0151, which is 267 times. That is a MAX over every
bead and every step, so it can be one bad event. The mean kinetic temperature is the number that
separates the two possibilities:

  mean T near 300 K and a few beads overlapping  -> the field pulls beads together
  mean T in the thousands                        -> the integrator is pumping and the
                                                    "collapse" is a numerical artefact

Run: python scripts/check_kinetic_temperature.py [n_steps] [friction]
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
MASS = 110.0
KB_INT = 0.008314462618          # amu nm^2 / (ps^2 K), numerically the same as kJ/(mol K)
TARGET = 300.0

pool = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s0 = pool[0]
L = len(s0["pos"])
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
NREP = 4
pos = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64).repeat(NREP, 1, 1)
vel = torch.zeros_like(pos)
temps = torch.full((NREP,), TARGET, dtype=torch.float64)
torch.manual_seed(20260222)


def ff(p, cap):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl, force_cap=cap)[1]


print(f"{s0['name']} L={L}, {NREP} replicas, dt 0.002 ps, friction {FRICTION}")
print(f"thermal |v| for {MASS:.0f} amu at {TARGET:.0f} K = "
      f"{np.sqrt(2.494 / 100 / MASS):.4f} nm/ps")
print()
for cap in (200.0, 5000.0):
    x = pos.clone()
    v = torch.zeros_like(x)
    ts, vs, dmins = [], [], []
    for step in range(NSTEPS):
        with torch.no_grad():
            f = ff(x, cap)
            x, v = C.batch_langevin_step(x, v, f, temps, dt_ps=0.002, mass_amu=MASS,
                                         friction=FRICTION, force_fn=lambda p: ff(p, cap))
        if step % 20 == 0 and step > 0:
            with torch.no_grad():
                v2 = (v ** 2).sum(dim=-1)                      # (B, N)
                Tkin = MASS * v2 / (3.0 * KB_INT)
                ts.append(float(Tkin.mean()))
                b = x.reshape(NREP, -1, 3)
                n = b.shape[1]
                d = torch.cdist(b, b) + torch.eye(n, device=b.device) * 10.0
                dmins.append(float(d.min()))
    ts = np.array(ts)
    dmins = np.array(dmins)
    print(f"force_cap = {cap}")
    print(f"  mean kinetic T: first {ts[0]:10.1f} K   median {np.median(ts):12.1f} K   "
          f"last {ts[-1]:12.1f} K")
    print(f"  closest bead pair: median {np.median(dmins):.4f} nm   minimum {dmins.min():.4f} nm")
    print(f"  ratio of final to target temperature: {ts[-1] / TARGET:.3g}")
    print()
print("A median near 300 K says the thermostat holds and the structure is being pulled apart by")
print("the field. A median in the thousands says the integrator is pumping, and every 'collapse'")
print("measured so far has been that.")
