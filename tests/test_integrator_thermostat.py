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

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
C = pytest.importorskip("torusfold.scheme2.torch_cgsim")

MASS = 110.0
T = 300.0
# In nm/ps/amu/kJ/mol the conversion is 1, and kB*T in amu nm^2/ps^2 is numerically the same
# number as kB*T in kJ/mol. So this is the physical moment, with no factor to hide behind.
KBT_OVER_M = C.KB_KJ * T / MASS


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
        f"(ratio {got / exact:.4f}); the pre-fix form was off by 1/(2*gamma*dt)")


@pytest.mark.parametrize("gamma,dt", [(0.1, 0.002), (1.0, 0.002), (0.1, 0.02), (5.0, 0.002)])
def test_noise_variance_does_not_depend_on_gamma_or_dt(gamma, dt):
    """The one-step variance is kBT/m * (1 - c1^2) for every (gamma, dt).

    A thermostat is allowed to reach the same temperature at different rates, but the stationary
    state must not depend on how it got there. The pre-fix form multiplied this by
    100*2*gamma*dt, so it failed at every one of these four points -- with the four factors
    0.04, 0.4, 0.4 and 2.0, which is what made the bug invisible: two of them are within a
    factor of three of the truth.
    """  # noqa: D401
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

def test_force_kick_uses_the_stated_mass_not_a_hundredth_of_it():
    """An oscillation period, which is the only thing that can see a wrong mass.

    A stationary distribution does not depend on the mass at all, so getting the mass wrong by
    any factor leaves every equilibrium average correct and only the clock wrong -- which is why
    no energy, force or gradient test ever caught this. Two 110 amu beads on a
    500 kJ/mol/nm^2 spring have a reduced mass of 55 amu and a closed-form period of 2.0839 ps.

    The pre-fix code divided both force kicks by unit_conv = 100, a factor that belongs to force
    constants in kJ/mol/angstrom^2. It measured 20.8389 ps, a ratio of exactly 10.000, i.e. an
    effective mass of 100 * mass_amu. The thermostat fix alone did not catch it either: the
    noise had been written to be consistent with that heavier mass, so the ensemble looked
    right while every timescale was ten times too long.
    """
    k, r0, dt, n = 500.0, 0.590, 0.002, 3000
    pos = torch.zeros((1, 3, 3), dtype=torch.float64)
    pos[0, 1, 0] = r0 + 0.010
    vel = torch.zeros_like(pos)
    temps = torch.full((1,), T, dtype=torch.float64)
    pi = torch.tensor([0])
    pj = torch.tensor([1])

    k_si = k * 1.660539e-21 / 1e-18          # J/m^2
    mu_si = 0.5 * MASS * 1.660539e-27        # kg
    period = 2.0 * math.pi / math.sqrt(k_si / mu_si) * 1e12

    dev = []
    for _ in range(n):
        _e, f = C._bond_f(pos, pi, pj, k, r0)
        # friction 0 turns the thermostat off: c1 = 1 and c2 = 0, so this is classical mechanics
        pos, vel = _step(pos, vel, f, temps, gamma=0.0, dt=dt)
        dev.append(float(pos[0, 1, 0] - pos[0, 0, 0]) - r0)
    dev = np.asarray(dev)
    t = np.arange(n) * dt
    cross = np.nonzero((dev[:-1] > 0) != (dev[1:] > 0))[0]
    assert len(cross) >= 3, (
        f"only {len(cross)} crossings in {n * dt:.1f} ps; a 10x heavy mass would show 2")
    half = float(np.diff(t[cross]).mean())
    assert 2.0 * half == pytest.approx(period, rel=0.02), (
        f"period {2 * half:.4f} ps against the closed-form {period:.4f} ps "
        f"(ratio {2 * half / period:.4f}; the pre-fix code gave 10.000)")

def test_free_oscillation_survives_when_the_tail_kick_uses_the_new_positions():
    """A symplectic integrator conserves the amplitude; the shipped B-A-O-A-B map did not.

    Friction 0 and no noise turn batch_langevin_step into a deterministic map. The two B
    half-kicks used to share one force tensor, f(x_n), even though the second one acts at
    x_{n+1}: that map has Jacobian determinant 1 - (dt^2/2)*a'(x), which for this bond is
    1 + (dt*omega)^2/2 = 1 + 1.8e-5 > 1, so phase space and energy grow at every step.
    Over 40000 steps the predicted energy factor is (1 + 1.8e-5)^40000 = 2.05, against
    2.04 measured on the amplitude (0.0143 nm against the starting 0.010 nm).

    Passing force_fn makes the last kick f(x'), which is what turns the deterministic part
    into a composition of exact Hamiltonian shears. The amplitude then stays at the
    starting 0.010 nm up to the bounded shadow-Hamiltonian oscillation, about 1e-5
    relative at dt*omega = 0.0060 -- three orders inside the 2 percent asserted here, so
    this bound is a statement about symplecticity, not about the step size.
    """
    k, r0, dt, n = 500.0, 0.590, 0.002, 40000
    pos = torch.zeros((1, 3, 3), dtype=torch.float64)
    pos[0, 1, 0] = r0 + 0.010
    vel = torch.zeros_like(pos)
    temps = torch.full((1,), T, dtype=torch.float64)
    pi = torch.tensor([0])
    pj = torch.tensor([1])

    def force_fn(p):
        return C._bond_f(p, pi, pj, k, r0)[1]

    r = np.empty(n)
    for step in range(n):
        _e, f = C._bond_f(pos, pi, pj, k, r0)
        # friction 0: c1 = 1 and c2 = 0, so only the deterministic B/A/B part acts
        pos, vel = C.batch_langevin_step(pos, vel, f, temps, dt_ps=dt,
                                         mass_amu=MASS, friction=0.0,
                                         force_fn=force_fn)
        r[step] = float(pos[0, 1, 0] - pos[0, 0, 0])
    amp = float(np.abs(r - r0).max())
    assert amp == pytest.approx(0.010, rel=0.02), (
        f"amplitude {amp:.5f} nm against the initial displacement 0.010 nm "
        f"(ratio {amp / 0.010:.4f}); without force_fn the same run grows to 0.0143 nm")
