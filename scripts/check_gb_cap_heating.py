"""Does the inner GB force cap cause the heating?

cg_energy_forces clips only the GB/SA + Manning gradient, at 50 kJ/mol/nm, and adds the same
block's ENERGY in full. So the force is not -dE/dx wherever that fires, and it fires when the
energy is largest: a bead is pushed back with 50 while the energy keeps climbing, and when
something else pulls it free the difference comes out as kinetic energy.

It predicts an observation made before it was found: raising the OUTER cap from 200 to 5000 raised
the mean kinetic temperature from 440.8 to 606.6 K, because every other term then delivers more of
its true force while this one stays clipped.

Run: python scripts/check_gb_cap_heating.py [ps]
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

PS = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
MASS = 110.0
KB_INT = 0.008314462618
TARGET = 300.0
NREP = 2
DT = 0.002
N = int(round(PS / DT))

pool = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s0 = pool[0]
L = len(s0["pos"])
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
x0 = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64).repeat(NREP, 1, 1)
temps = torch.full((NREP,), TARGET, dtype=torch.float64)

print(f"{s0['name']} L={L}, {NREP} replicas, {PS} ps, dt {DT}, cap 5000 (outer), friction 1.0")
print(f"thermal |v| at {TARGET:.0f} K = {np.sqrt(2.494 / 100 / MASS):.4f} nm/ps")
print()
print(f"{'GB_FORCE_CAP':>13s} {'mean T (K)':>12s} {'T/target':>9s} "
      f"{'closest median (nm)':>20s} {'finite':>8s}")
print("-" * 68)
for gbcap in (50.0, 500.0, 1.0e9):
    C.GB_FORCE_CAP = gbcap
    x = x0.clone()
    v = torch.zeros_like(x)
    torch.manual_seed(4242)

    def ff(p):
        cl = C.GPUCellList(cell_size=1.5)
        cl.build(p)
        return C.cg_energy_forces(p, ij, pw, cell_list=cl, force_cap=5000.0)[1]

    Ts, dmins, ok = [], [], True
    for step in range(N):
        try:
            with torch.no_grad():
                f = ff(x)
                x, v = C.batch_langevin_step(x, v, f, temps, dt_ps=DT, mass_amu=MASS,
                                             friction=1.0, force_fn=ff)
        except Exception as exc:
            ok = False
            print(f"{gbcap:13.1f}  raised {type(exc).__name__} at step {step}: {str(exc)[:60]}")
            break
        if step % max(1, N // 40) == 0 and step >= N // 4:
            with torch.no_grad():
                Ts.append(float((MASS * (v ** 2).sum(dim=-1) / (3.0 * KB_INT)).mean()))
                b = x.reshape(NREP, -1, 3)
                m = b.shape[1]
                d = torch.cdist(b, b) + torch.eye(m, device=b.device) * 10.0
                dmins.append(float(d.min()))
    if ok:
        Ts = np.array(Ts)
        print(f"{gbcap:13.1f} {Ts.mean():12.1f} {Ts.mean() / TARGET:9.2f} "
              f"{np.median(dmins):20.4f} {'yes':>8s}")
print()
print("A mean temperature that falls toward 300 K as the cap is removed says this clip is the")
print("heater. A mean that does not move says it is not, and the search continues.")
