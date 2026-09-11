"""The Langevin thermostat's temperature, tested against moments whose exact value is known.

Why this exists. batch_langevin_step wrote its O-step noise as

    sqrt(2*gamma*kB/m) * sqrt(T*dt) * c2

while the stationary variance of v <- c1*v + A*xi is A^2/(1-c1^2), and the integrator works in
nm/ps, so the target is kB*T/(mass*unit_conv). The two differ by sqrt(2*gamma*dt*unit_conv).
With the shipped parameters, gamma = 1.0 /ps and dt = 0.002 ps, the effective temperature was
0.4*T: the GPU REST2/REMD path labelled 300-1000 K actually ran at 120-400 K.

Nothing in the field could have caught this. Every force, energy and gradient was correct; only
the thermostat was wrong, and a wrong thermostat makes a wrong ensemble while looking healthy.
The REMD exchange criterion at torch_cgsim.py:1720 computes its Metropolis test with
beta = 1/(kB*T) at the NOMINAL temperature, so it was also inconsistent with the dynamics.

So the tests below assert on quantities whose value is fixed by physics and cannot be tuned:

  1. one O step from rest must produce the exact noise variance, for any gamma and dt
  2. a free particle, with no forces at all, must reach <v^2> = kB*T/m
  3. a lone harmonic bond must reach sigma = sqrt(kB*T/k)

Test 1 is exact and costs one call. Tests 2 and 3 evolve in time, so they use a coarse dt to
reach the same stationary state in few steps; the stationary values do not depend on dt.

Run: python -m pytest tests/test_integrator_thermostat.py
"""
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
C = pytest.importorskip("torusfold.scheme2.torch_cgsim")

MASS = 110.0
T = 300.0
UNIT_CONV = 100.0
# kB*T/m already converted into the integrator's (nm/ps)^2
KBT_OVER_M = C.KB_KJ * T / (UNIT_CONV * MASS)


def _step(pos, vel, forces, temps, gamma, dt):
    return C.batch_langevin_step(pos, vel, forces, temps, dt_ps=dt,
                                 mass_amu=MASS, friction=gamma)


def test_one_step_from_rest_has_the_exact_noise_variance():
    """With v = 0 and no forces, the O step leaves v = A*xi, so Var(v) = A^2 exactly."""
    gamma, dt = 1.0, 0.002
    c1 = math.exp(-gamma * dt)
    exact = KBT_OVER_M * (1.0 - c1 * c1)

    B = 50000
    vel = torch.zeros((B, 3, 3), dtype=torch.float64)
    pos = torch.zeros_like(vel)
    forces = torch.zeros_like(vel)
    temps = torch.full((B,), T, dtype=torch.float64)
    _pos, vel = _step(pos, vel, forces, temps, gamma, dt)

    got = float(vel.var())
    # 450k samples puts the sampling error of a variance near 0.2 percent
    assert got == pytest.approx(exact, rel=0.02), (
        f"one-step noise variance {got:.6e} against the exact {exact:.6e} "
        f"(ratio {got / exact:.4f}); the pre-fix form was off by 1/(2*gamma*dt*{UNIT_CONV})")


@pytest.mark.parametrize("gamma,dt", [(0.1, 0.002), (1.0, 0.002), (0.1, 0.02), (5.0, 0.002)])
def test_noise_variance_does_not_depend_on_gamma_or_dt(gamma, dt):
    """The one-step variance is kBT/m * (1 - c1^2) for every (gamma, dt).

    A thermostat is allowed to reach the same temperature at different rates, but the stationary
    state must not depend on how it got there. The pre-fix form multiplied this by
    100*2*gamma*dt, so it failed at every one of these four points -- with the four factors
    0.04, 0.4, 0.4 and 2.0, which is what made the bug invisible: two of them are within a
    factor of three of the truth.
    """
    c1 = math.exp(-gamma * dt)
    exact = KBT_OVER_M * (1.0 - c1 * c1)

    B = 50000
    vel = torch.zeros((B, 3, 3), dtype=torch.float64)
    pos = torch.zeros_like(vel)
    forces = torch.zeros_like(vel)
    temps = torch.full((B,), T, dtype=torch.float64)
    _pos, vel = _step(pos, vel, forces, temps, gamma, dt)

    assert float(vel.var()) == pytest.approx(exact, rel=0.02)


def test_free_particle_reaches_kBT_over_m():
    """No forces at all, so only the O step acts. <v^2> must settle at kB*T/m."""
    gamma, dt = 1.0, 0.02
    n, burn, B = 1000, 200, 64
    vel = torch.zeros((B, 3, 3), dtype=torch.float64)
    pos = torch.zeros_like(vel)
    forces = torch.zeros_like(vel)
    temps = torch.full((B,), T, dtype=torch.float64)
    torch.manual_seed(20260218)

    s2, n_samp = 0.0, 0
    for step in range(n):
        pos, vel = _step(pos, vel, forces, temps, gamma, dt)
        if step >= burn:
            s2 += float((vel ** 2).sum())
            n_samp += vel.numel()
    got = s2 / n_samp
    assert got == pytest.approx(KBT_OVER_M, rel=0.10), (
        f"free-particle <v^2> {got:.6e} against kB*T/m {KBT_OVER_M:.6e} "
        f"(ratio {got / KBT_OVER_M:.4f}, implied T = {T * got / KBT_OVER_M:.1f} K)")


def test_lone_harmonic_bond_reaches_its_exact_width():
    """One bond, nothing else. sigma must be sqrt(kB*T/k), which is the same claim as test 2."""
    k, r0 = 500.0, 0.590
    gamma, dt = 1.0, 0.02
    n, burn, B = 4000, 500, 32
    pos = torch.zeros((B, 3, 3), dtype=torch.float64)
    pos[:, 1, 0] = r0
    vel = torch.zeros_like(pos)
    temps = torch.full((B,), T, dtype=torch.float64)
    pi = torch.tensor([0])
    pj = torch.tensor([1])
    torch.manual_seed(4242)

    s1 = s2 = 0.0
    n_samp = 0
    for step in range(n):
        _e, f = C._bond_f(pos, pi, pj, k, r0)
        pos, vel = _step(pos, vel, f, temps, gamma, dt)
        if step >= burn:
            r = torch.linalg.norm(pos[:, 0] - pos[:, 1], dim=-1)
            s1 += float(r.sum())
            s2 += float((r ** 2).sum())
            n_samp += int(r.numel())
    m = s1 / n_samp
    sig = math.sqrt(max(s2 / n_samp - m * m, 0.0))
    kbt = C.KB_KJ * T
    exact = math.sqrt(kbt / k)
    # The potential depends only on |r|, so the equilibrium density carries the radial measure
    # r^2 and the mean is NOT r0: it sits 2*kBT/(k*r0) further out. Asserting m == r0 failed at
    # 0.6073 against 0.590, which is this shift to within two percent; the test was wrong, not
    # the integrator. Checking the shift makes the test stronger, not weaker.
    mean_exact = r0 + 2.0 * kbt / (k * r0)

    assert m == pytest.approx(mean_exact, rel=0.01), (
        f"bond mean {m:.5f} nm against the radial expectation "
        f"{mean_exact:.5f} nm (r0 shifted out by 2*kBT/(k*r0) = "
        f"{2.0 * kbt / (k * r0):.5f})")
    assert sig == pytest.approx(exact, rel=0.10), (
        f"bond sigma {sig:.6f} nm against sqrt(kB*T/k) = {exact:.6f} "
        f"(ratio {sig / exact:.4f}, implied T = {T * (sig / exact) ** 2:.1f} K)")
