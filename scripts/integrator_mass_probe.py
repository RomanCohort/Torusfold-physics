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

The last three sections run the same bond through batch_langevin_step's force_fn path,
where the final half-kick sees the forces at the post-update coordinates: the amplitude
there must not drift, and the two paths are timed against each other at the end.

Run: python scripts/integrator_mass_probe.py
"""
import math
import sys
import time
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

# ═══════════════════════════════════════════════════════════════════════════════
# The same experiment through the integrator's force_fn path -- the symplectic one.
#
# batch_langevin_step reuses the caller's force tensor for BOTH B half-kicks when
# force_fn is None. The section above measures that fallback: the second kick acts at
# x + dt*v but sees f(x), its Jacobian determinant is 1 - (dt^2/2)*a', and the energy
# grows by the factor (1 + (dt*omega)^2/2)^n. Passing force_fn makes the final kick
# f(x') instead, so the deterministic part is a composition of exact Hamiltonian
# shears and the amplitude cannot drift.
# ═══════════════════════════════════════════════════════════════════════════════


def run_bond(force_fn, n_steps):
    """One friction-0 bond trajectory; returns r(t), bead 0 held at x = 0 nm."""
    p = torch.zeros((1, 3, 3), dtype=torch.float64)
    p[0, 1, 0] = R0 + 0.010
    v = torch.zeros_like(p)
    t = torch.full((1,), 300.0, dtype=torch.float64)
    out = np.empty(n_steps)
    for i in range(n_steps):
        _e, f = C._bond_f(p, pi, pj, K, R0)
        p, v = C.batch_langevin_step(p, v, f, t, dt_ps=DT, mass_amu=MASS,
                                     friction=0.0, force_fn=force_fn)
        out[i] = float(p[0, 1, 0] - p[0, 0, 0])
    return out


print()
print("=== the same run with force_fn (the symplectic path) ===")
r_fixed = run_bond(lambda q: C._bond_f(q, pi, pj, K, R0)[1], NSTEPS)
amp_fallback = float(np.abs(r - R0).max())
amp_fixed = float(np.abs(r_fixed - R0).max())
print(f"force_fn=None : r range [{r.min():.5f}, {r.max():.5f}], amplitude {amp_fallback:.5f} nm")
print(f"force_fn set  : r range [{r_fixed.min():.5f}, {r_fixed.max():.5f}], amplitude {amp_fixed:.5f} nm")
print("started at 0.010 nm; the acceptance bound is 2 percent of 0.010 = 0.0002 nm")


def pair_times(run_none, run_fn, n_steps, reps=7):
    """Time both paths alternately and return the two lists of per-run seconds.

    This box is running a long background job, and a plain before/after pair is not
    trustworthy here: measured that way the same two paths came out anywhere from 1.07x
    to 2.9x. Alternating the paths makes the machine load hit both, and the minimum is
    the run least disturbed by it.
    """
    none_t, fn_t = [], []
    for _ in range(reps):
        t0 = time.perf_counter(); run_none(); none_t.append((time.perf_counter() - t0) / n_steps)
        t0 = time.perf_counter(); run_fn(); fn_t.append((time.perf_counter() - t0) / n_steps)
    return none_t, fn_t


def report(label, none_t, fn_t, unit, unit_name):
    """min and median per-step cost for each path, and the ratio of the minima."""
    m0, m1 = min(none_t), min(fn_t)
    q0, q1 = sorted(none_t)[len(none_t) // 2], sorted(fn_t)[len(fn_t) // 2]
    print(label)
    print(f"  force_fn=None : min {m0 * unit:.2f}  median {q0 * unit:.2f}  {unit_name}")
    print(f"  force_fn set  : min {m1 * unit:.2f}  median {q1 * unit:.2f}  {unit_name}")
    print(f"  extra cost    : {100 * (m1 / m0 - 1):.1f} percent per step (minima), {m1 / m0:.2f}x")


print()
print("=== what the extra force evaluation costs: 3-bead integrator, 1000 steps x 7 ===")
print("(here the force is the analytic 1-bond call, so this isolates the integrator side)")
bond_fn = lambda q: C._bond_f(q, pi, pj, K, R0)[1]
bond_none, bond_fn_t = pair_times(lambda: run_bond(None, 1000),
                                  lambda: run_bond(bond_fn, 1000), 1000)
report("3-bead, one 500 kJ/mol/nm^2 bond:", bond_none, bond_fn_t, 1e6, "us/step")


def cg_loop(use_force_fn, n_steps, L=24, B=8):
    """A realistic REMD shard: B replicas x 3L CG beads, one full force call per step."""
    rng = np.random.default_rng(0)
    coords = rng.normal(0, 1.0, size=(L, 3)).cumsum(axis=0) / 10.0
    p = torch.tensor(coords, dtype=torch.float32)[None].repeat(B, 1, 1)
    v = torch.zeros_like(p)
    pairs = torch.zeros((0, 2), dtype=torch.long)
    lams = torch.ones(B)
    t = torch.full((B,), 300.0)
    # same cell-list object as BatchedREMD2D.run: built once, reused by every force call
    cell_list = C.GPUCellList(cell_size=1.5)
    counter = [0]

    def force_fn(q):
        counter[0] += 1
        return C.cg_energy_forces(q, pairs, None, lams=lams, cell_list=cell_list)[1]

    for _ in range(n_steps):
        _e, f = C.cg_energy_forces(p, pairs, None, lams=lams, cell_list=cell_list)
        p, v = C.batch_langevin_step(p, v, f, t, dt_ps=0.002, friction=1.0,
                                     force_fn=force_fn if use_force_fn else None)
    return counter[0]


print()
print("=== the cost that matters: 8 replicas x 24 residues, 50 steps x 7 ===")
cg_loop(False, 3); cg_loop(True, 3)   # warm up, including the neighbour list
cg_none, cg_fn_t = pair_times(lambda: cg_loop(False, 50),
                              lambda: cg_loop(True, 50), 50)
report("8 replicas x 24 residues, full cg_energy_forces:", cg_none, cg_fn_t, 1e3, "ms/step")
calls = cg_loop(True, 7)
print(f"force_fn invocations over 7 steps: {calls} (1 per step => the extra call is real)")
