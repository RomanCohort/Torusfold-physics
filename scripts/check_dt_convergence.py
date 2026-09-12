"""Is the timestep too large for this field, and has every collapse measurement been that?

The kinetic temperature settles at 440 K with the cap at 200 and 607 K with it at 5000, against a
300 K target. That is a real overshoot and it scales with the cap, which points at the cap
injecting energy -- but there is a suspect that has never been checked.

The first version of this docstring argued the suspect was the integrator, from this arithmetic:

    a half-kick of dt = 0.002 ps adds 0.073 nm/ps of velocity, five times the thermal 0.0151.

**That was off by ten, and the cause is a stray /100 in the print statement below.** With mass in
amu and velocity in nm/ps the thermal speed is sqrt(kBT/m) with kBT in kJ/mol, i.e.
sqrt(2.494/110) = 0.1506 nm/ps per component (the 3D magnitude, 0.2608, is what the T formula in
this script uses). The print read sqrt(2.494 / 100 / MASS), which is kBT at 3 K, not 300 K.

Corrected: the half-kick is 0.0746 against a per-component thermal 0.1506 -- a ratio of 0.49, not
5. A half-kick equal to the thermal speed would need 8284 kJ/mol/nm, twice the 4103 this field
produces. So the half-kick is not evidence that the integrator is the heater.

The timestep is comfortable for a second reason. omega = sqrt(k/mu) over the field, at the 55 amu
reduced mass this script already uses for C4'-N:

    K_INTRA_CN   36399.2   omega 25.73 /ps   omega*dt = 0.051  at dt = 0.002
    K_INTRA_PC   20752.7   omega 19.42 /ps   0.039
    K_CLASH      20000.0   omega 19.07 /ps   0.038
    K_LINK_CP     9574.4   omega 13.19 /ps   0.026
    K_LINK_NP     5477.7   omega  9.98 /ps   0.020
    K_INTRA_PN    1785.9   omega  5.70 /ps   0.011
    K_BB          1122.4   omega  4.52 /ps   0.009
    K_PAIR         600.0   omega  3.30 /ps   0.007
    K_ANGLE         28.1   omega  0.71 /ps   0.001
    K_DIH            7.2   omega  0.36 /ps   0.001

omega*dt maxes at 0.051, so the Verlet integration error is order (omega*dt)^2/4 = 6.5e-4. That is
not a heater either.

And the reduction is small: constraining the bonds with SHAKE/LINCS -- the obvious way to buy a
bigger dt -- removes 25.73 and 19.42 but NOT K_CLASH at 19.07, which is a repulsive wall and is
not constrainable. So the ceiling is 25.73/19.07 = 1.35x, about 2 fs to 2.7 fs. The ten-fold
timestep is not on the table, and it is worth knowing that before paying for constraints, because
constraining the intra bonds would set the width of intra_pc and intra_cn to zero -- the two
coordinates this field currently reproduces exactly (1.010 and 1.000 of reference).

None of that settles where the 440/607 K comes from. It only removes one candidate. The test is dt
convergence at FIXED SIMULATED TIME, which is what this script does: fixing the step count
instead would compare 2 ps against 0.5 ps and mean nothing, a mistake already made once here.

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
      f"{np.sqrt(2.494 / MASS):.4f} nm/ps (per component)")
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
