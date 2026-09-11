"""What is the integrator's effective mass? A period measurement, not a dimensional argument.

Every velocity-update force kick in torch_cgsim divides by unit_conv = 100, with the comment
"kJ/mol/nm = amu*nm/ps^2 x 100". Whether that 100 belongs there decides what the stated
dt_ps = 0.002 ps means, and no existing test can see it: a stationary distribution does not
depend on the mass at all, so getting the mass wrong by any factor leaves every equilibrium
average correct and only the clock wrong.

Two harmonic beads of 110 amu each have a reduced mass of 55 amu. On a spring of
500 kJ/mol/nm^2 that is a physical oscillation, and its period in ps is arithmetic:

    k = 500 kJ/mol/nm^2 = 0.83027 N/m,  mu = 55 amu = 9.1333e-26 kg
    omega = sqrt(k/mu) = 3.0151e12 rad/s   ->   T = 2.084 ps

If the code's force kick is right, the simulated bond oscillates once every 2.084 ps. If the
kick is 100x too weak, the effective mass is 100*mu and the period is 10 times longer.

Friction is zero and the thermostat is off, so this is pure classical mechanics.

Run: python scripts/integrator_mass_probe.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

K = 500.0          # kJ/mol/nm^2
R0 = 0.590         # nm
MASS = 110.0       # amu per bead, so mu = 55
DT = 0.002         # ps
NSTEPS = 40000     # 80 ps, long enough for dozens of periods either way

K_SI = K * 1.660539e-21 / 1e-18            # J/m^2
MU_SI = 0.5 * MASS * 1.660539e-27          # kg
OMEGA = math.sqrt(K_SI / MU_SI)            # rad/s
PERIOD = 2 * math.pi / OMEGA * 1e12        # ps
print(f"physical period for mu = {0.5 * MASS:.0f} amu on k = {K} kJ/mol/nm^2: {PERIOD:.4f} ps")
print(f"the same bond with an effective mass 100x larger would take "
      f"{PERIOD * 10:.3f} ps")
print()

pos = torch.zeros((1, 3, 3), dtype=torch.float64)
pos[0, 1, 0] = R0 + 0.010          # a 0.010 nm displacement, small enough to stay harmonic
vel = torch.zeros_like(pos)
zero_f = torch.zeros_like(pos)
temps = torch.full((1,), 300.0, dtype=torch.float64)
pi = torch.tensor([0])
pj = torch.tensor([1])

r = []
for step in range(NSTEPS):
    _e, f = C._bond_f(pos, pi, pj, K, R0)
    pos, vel = C.batch_langevin_step(pos, vel, f, temps, dt_ps=DT,
                                     mass_amu=MASS, friction=0.0)
    # bead 0 sits at x = 0 and bead 1 at x = r0, so take the difference the other way round;
    # writing it backwards gave a constant -0.59 and the crossing count below saw nothing
    r.append(float(pos[0, 1, 0] - pos[0, 0, 0]))
r = np.array(r)
t = np.arange(NSTEPS) * DT

# count sign changes of (r - R0) to get the period without any fitting
dev = r - R0
crossings = np.nonzero((dev[:-1] > 0) != (dev[1:] > 0))[0]
print(f"{NSTEPS} steps at dt = {DT} ps = {NSTEPS * DT:.1f} ps")
print(f"sign changes of (r - r0): {len(crossings)}")
if len(crossings) >= 3:
    half = np.diff(t[crossings])
    print(f"half-period from consecutive crossings: mean {half.mean():.4f} ps  "
          f"min {half.min():.4f}  max {half.max():.4f}")
    print(f"implied period {2 * half.mean():.4f} ps")
    print()
    print(f"measured / physical = {2 * half.mean() / PERIOD:.3f}")
    print("  1 -> unit_conv = 100 leaves the force kick alone")
    print(" 10 -> the force kick is 100x too weak, i.e. m is effectively 100*mass_amu")
else:
    print(f"too few crossings in {NSTEPS * DT:.1f} ps; the bond has not completed a period")
    print(f"r went from {r[0]:.5f} to {r[-1]:.5f}, min {r.min():.5f}, max {r.max():.5f}")
print()
print("=== energy sanity: is the amplitude conserved at all? ===")
print(f"r range over the run: [{r.min():.5f}, {r.max():.5f}] around r0 = {R0}")
print(f"started at {R0 + 0.010:.5f}, so the amplitude should stay near 0.010 nm")
