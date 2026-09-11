"""Two assumptions about the sampler, measured instead of derived.

1. Isolated O step. With no forces at all, velocity is a pure Ornstein-Uhlenbeck process and
   its stationary second moment must be kBT/m whatever the potential does. batch_langevin_step
   writes the noise as sqrt(2*gamma*kB/m) * sqrt(T) * sqrt(dt) * c2, while the standard OU step
   wants sqrt(kBT/m * (1 - c1^2)). The two differ by sqrt(2*gamma*dt). This measures which is
   right, with no potential to hide behind.

2. The bonded control, redone. The first version held the STEP COUNT fixed while varying dt, so
   dt = 0.2 ps ran 100 times longer than dt = 0.002 ps and the comparison was meaningless. This
   one holds the simulated TIME fixed, which is the only way the dt dependence means anything.

Run: python scripts/integrator_control.py [ps]
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

PS = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
K = 500.0
R0 = 0.590
MASS = 110.0
NREP = 32
# amu*nm^2/ps^2 is the integrator's energy unit: unit_conv = 100 turns kJ/mol/nm into amu*nm/ps^2
KBT_INT = C.KB_KJ * 300.0 / 100.0
V2_EXACT = KBT_INT / MASS
print(f"target <v^2> = kBT/m = {V2_EXACT:.6e} (nm/ps)^2")
print(f"exact sigma for the bond = sqrt(kBT/k) = {np.sqrt(C.KB_KJ * 300.0 / K):.6f} nm")
print(f"{NREP} replicas, {PS} ps of simulated time per row")
print()


def free_particle(gamma, dt):
    """No forces. Only the O step acts, so <v^2> is the O step's stationary value."""
    n = int(round(PS / dt))
    torch.manual_seed(99)
    vel = torch.zeros((NREP, 3, 3), dtype=torch.float64)
    pos = torch.zeros_like(vel)
    forces = torch.zeros_like(vel)
    temps = torch.full((NREP,), 300.0, dtype=torch.float64)
    s2 = 0.0
    cnt = 0
    for step in range(n):
        pos, vel = C.batch_langevin_step(pos, vel, forces, temps, dt_ps=dt,
                                         mass_amu=MASS, friction=gamma)
        if step >= n // 2:
            s2 += float((vel ** 2).sum())
            cnt += vel.numel()
    return s2 / cnt


def bonded(gamma, dt):
    n = int(round(PS / dt))
    pos = torch.zeros((NREP, 3, 3), dtype=torch.float64)
    pos[:, 1, 0] = R0
    vel = torch.zeros_like(pos)
    temps = torch.full((NREP,), 300.0, dtype=torch.float64)
    pi = torch.tensor([0])
    pj = torch.tensor([1])
    torch.manual_seed(4242)
    s1 = s2 = 0.0
    cnt = 0
    for step in range(n):
        _e, f = C._bond_f(pos, pi, pj, K, R0)
        pos, vel = C.batch_langevin_step(pos, vel, f, temps, dt_ps=dt,
                                         mass_amu=MASS, friction=gamma)
        if step >= n // 2:
            r = torch.linalg.norm(pos[:, 0] - pos[:, 1], dim=-1)
            s1 += float(r.sum())
            s2 += float((r ** 2).sum())
            cnt += int(r.numel())
    m = s1 / cnt
    return m, float(np.sqrt(max(s2 / cnt - m * m, 0.0)))


print("=== 1. free particle: is the thermostat at the right temperature? ===")
print(f"{'gamma':>7s} {'dt (ps)':>9s} {'steps':>8s} {'<v^2>/exact':>13s} "
      f"{'sqrt(2*g*dt)':>13s}")
print("-" * 56)
for gamma, dt in ((0.1, 0.002), (1.0, 0.002), (0.1, 0.02), (5.0, 0.002)):
    ratio = free_particle(gamma, dt) / V2_EXACT
    print(f"{gamma:7.2f} {dt:9.4f} {int(round(PS / dt)):8d} {ratio:13.6f} "
          f"{np.sqrt(2 * gamma * dt):13.6f}")
print()
print("  A ratio well below 1 that tracks sqrt(2*gamma*dt) is the missing factor, not")
print("  slow relaxation: this is the stationary value of the O step alone.")
print()

print("=== 2. bonded control at fixed simulated time ===")
print(f"{'gamma':>7s} {'dt (ps)':>9s} {'steps':>8s} {'mean r':>10s} {'sigma':>10s} "
      f"{'sigma/exact':>12s}")
print("-" * 66)
exact = np.sqrt(C.KB_KJ * 300.0 / K)
for gamma, dt in ((0.1, 0.002), (0.1, 0.005), (0.1, 0.02), (1.0, 0.002), (0.1, 0.05)):
    m, sig = bonded(gamma, dt)
    print(f"{gamma:7.2f} {dt:9.4f} {int(round(PS / dt)):8d} {m:10.5f} {sig:10.6f} "
          f"{sig / exact:12.4f}")
