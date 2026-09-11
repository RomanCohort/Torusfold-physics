"""Is the timestep too large for this field, and has every collapse measurement been that?

The kinetic temperature settles at 440 K with the cap at 200 and 607 K with it at 5000, against a
300 K target. That is a real overshoot and it scales with the cap, which points at the cap
injecting energy -- but there is a suspect that has never been checked.

The field now produces up to 4103 kJ/mol/nm on undamaged native geometry. On a 110 amu bead that
is an acceleration of 36 nm/ps^2, so one half-kick of dt = 0.002 ps adds 0.073 nm/ps of velocity,
five times the thermal 0.0151. If that is happening often, the integrator is the thing heating the
system, and every collapse measured so far is contaminated by integration error rather than by the
force field.

The test is dt convergence at FIXED SIMULATED TIME. Fixing the step count instead would compare
2 ps against 0.5 ps and mean nothing, which is a mistake already made once in this project.

Run: python scripts/check_dt_convergence.py [ps]
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

pool = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s0 = pool[0]
L = len(s0["pos"])
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
x0 = torch.tensor(s0["pos"].reshape(1, 3 * L, 3), dtype=torch.float64).repeat(NREP, 1, 1)
temps = torch.full((NREP,), TARGET, dtype=torch.float64)

print(f"{s0['name']} L={L}, {NREP} replicas, {PS} ps of SIMULATED TIME per row, friction 1.0,")
print(f"force_cap 5000, symplectic. Thermal |v| at {TARGET:.0f} K = "
      f"{np.sqrt(2.494 / 100 / MASS):.4f} nm/ps")
print()
print(f"{'dt (ps)':>9s} {'steps':>8s} {'mean T (K)':>12s} {'T/target':>9s} "
      f"{'closest median (nm)':>20s} {'max |v|':>9s}")
print("-" * 72)
for dt in (0.002, 0.001, 0.0005):
    n = int(round(PS / dt))
    x = x0.clone()
    v = torch.zeros_like(x)
    torch.manual_seed(31337)

    def ff(p):
        cl = C.GPUCellList(cell_size=1.5)
        cl.build(p)
        return C.cg_energy_forces(p, ij, pw, cell_list=cl, force_cap=5000.0)[1]

    Ts, dmins, vmax = [], [], 0.0
    for step in range(n):
        with torch.no_grad():
            f = ff(x)
            x, v = C.batch_langevin_step(x, v, f, temps, dt_ps=dt, mass_amu=MASS,
                                         friction=1.0, force_fn=ff)
        if step % max(1, n // 40) == 0 and step >= n // 4:
            with torch.no_grad():
                Tk = MASS * (v ** 2).sum(dim=-1) / (3.0 * KB_INT)
                Ts.append(float(Tk.mean()))
                vmax = max(vmax, float(torch.linalg.norm(v.reshape(-1, 3), dim=-1).max()))
                b = x.reshape(NREP, -1, 3)
                m = b.shape[1]
                d = torch.cdist(b, b) + torch.eye(m, device=b.device) * 10.0
                dmins.append(float(d.min()))
    Ts = np.array(Ts)
    print(f"{dt:9.4f} {n:8d} {Ts.mean():12.1f} {Ts.mean() / TARGET:9.2f} "
          f"{np.median(dmins):20.4f} {vmax:9.4f}")
print()
print("If the mean temperature falls toward 300 K as dt shrinks, the integrator is the heater and")
print("the collapse measurements are contaminated. If it stays at 440-600 K, the field or the cap")
print("is, and dt is not the problem.")
