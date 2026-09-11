"""After the thermostat fix, what is left of the bond's width deficit, and where does it come from?

With the O-step noise corrected, a free particle settles at kB*T/m to within half a percent. The
lone harmonic bond, however, still comes in 7 to 14 percent narrow at the shipped timestep. That
is a separate claim and it needs its own evidence.

The candidate is the second B step, which reuses the forces from the start of the step. The code
says so in a comment: "using the same forces; a simplified version - the exact one would require
recomputing the forces". A stale kick is not the BAOAB splitting, so it perturbs the stationary
distribution by some power of dt. A bias in dt shows up as a deficit that SHRINKS as dt shrinks.
A deficit that plateaus is structural instead, and would need a different explanation.

Time is held fixed across rows, so only dt varies.

Run: python scripts/stale_force_bias.py [ps] [n_rep]
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

PS = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
NREP = int(sys.argv[2]) if len(sys.argv) > 2 else 16
K = 500.0
R0 = 0.590
MASS = 110.0
T = 300.0
GAMMA = 1.0
EXACT = math.sqrt(C.KB_KJ * T / K)
JACOBIAN = 2.0 * C.KB_KJ * T / (K * R0)

print(f"gamma = {GAMMA} /ps (the shipped value), {PS} ps per row, {NREP} replicas")
print(f"exact sigma = {EXACT:.6f} nm; the radial measure shifts the mean out by {JACOBIAN:.5f} nm")
print()
print(f"{'dt (ps)':>9s} {'steps':>8s} {'mean r':>9s} {'sigma':>10s} {'sigma/exact':>12s} "
      f"{'implied T (K)':>14s}")
print("-" * 68)
rows = []
for dt in (0.0005, 0.001, 0.002, 0.004, 0.010):
    n = int(round(PS / dt))
    pos = torch.zeros((NREP, 3, 3), dtype=torch.float64)
    pos[:, 1, 0] = R0
    vel = torch.zeros_like(pos)
    temps = torch.full((NREP,), T, dtype=torch.float64)
    pi = torch.tensor([0])
    pj = torch.tensor([1])
    torch.manual_seed(31337)
    s1 = s2 = 0.0
    cnt = 0
    for step in range(n):
        _e, f = C._bond_f(pos, pi, pj, K, R0)
        pos, vel = C.batch_langevin_step(pos, vel, f, temps, dt_ps=dt,
                                         mass_amu=MASS, friction=GAMMA)
        if step >= n // 2:
            r = torch.linalg.norm(pos[:, 0] - pos[:, 1], dim=-1)
            s1 += float(r.sum())
            s2 += float((r ** 2).sum())
            cnt += int(r.numel())
    m = s1 / cnt
    sig = math.sqrt(max(s2 / cnt - m * m, 0.0))
    ratio = sig / EXACT
    rows.append((dt, ratio))
    print(f"{dt:9.4f} {n:8d} {m:9.5f} {sig:10.6f} {ratio:12.4f} "
          f"{T * ratio ** 2:14.1f}")
print()
print("=== does the deficit scale with dt? ===")
print(f"{'dt (ps)':>9s} {'1 - sigma/exact':>17s} {'deficit / dt':>14s} "
      f"{'deficit / dt^2':>16s}")
print("-" * 62)
for dt, ratio in rows:
    d = 1.0 - ratio
    print(f"{dt:9.4f} {d:17.5f} {d / dt:14.2f} {d / dt ** 2:16.2f}")
print()
print("A constant deficit/dt means the error is O(dt); constant deficit/dt^2 means O(dt^2).")
print("Either way a deficit that keeps falling is the stale-force splitting. A plateau is not.")
