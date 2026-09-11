"""The GPU metadynamics loop must hand the integrator a force at the new positions.

Why this exists. batch_langevin_step runs B, A, O, A, B, and the two B half-kicks act at
different coordinates: f(x_n) and f(x_{n+1}). A caller that holds only one force tensor
cannot supply the second, so the integrator falls back to reusing f(x_n) and the
deterministic part of the step stops being symplectic -- its Jacobian determinant is
1 + (dt*omega)^2/2 for a harmonic coordinate, so phase-space volume and energy grow at
every step. metadynamics_gpu.run used to be such a caller; these tests keep it from
becoming one again.

For every call run() makes to batch_langevin_step this file asserts:

  1. force_fn is present. This is the regression under test.
  2. the integrator invokes it, exactly once per step (the tail B half-kick).
  3. force_fn(x) at the coordinates the loop integrated reproduces the force tensor the
     loop integrated with, bit for bit. That tensor is cg_energy_forces(...) + the
     accumulated-hill bias -- the bias is part of the Hamiltonian the integrator
     propagates -- so a force_fn that recomputed only the physical part fails here.

Point 3 is evaluated inside the spy, at the moment of the step, because the closure reads
hills_centers at call time: calling it again after the run would let it see hills that did
not exist during that step.

The second test makes the bias unmistakable instead of rounding-level: it replaces
_compute_bias_forces with a function that returns zeros while there are no hills and a
constant 50 kJ/mol/nm once a hill exists. A physical-only force_fn then misses by 50, not
by the ~1e-6 a four-step run deposits.

The run is deliberately tiny (6 residues, 1 replica, 4 steps). run() ends with 200 Adam
minimisation steps over the CG energy, which is why each test still costs a few seconds;
that part is not under test.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

M = pytest.importorskip("torusfold.scheme2.metadynamics_gpu")

L = 6
N_REP = 1
HILL_FREQ = 2
N_STEPS = 2 * HILL_FREQ   # two blocks, so the second one runs with a deposited hill
COORDS = np.random.default_rng(0).normal(0.0, 5.0, (L, 3))
PAIRS = [(0, 3, 1.0), (1, 4, 1.0)]


def _run_with_spy(monkeypatch):
    """Run BatchedMetadynamics once and record every integrator call it makes.

    Each record holds the force tensor handed in, the force_fn kwarg, the force_fn's own
    answer at those coordinates (with the hill state of that step), and the arguments the
    integrator passed to the force_fn for the tail kick.
    """
    real = M.batch_langevin_step
    calls = []

    def spy(pos, vel, forces, temperatures, **kw):
        record = {"force_fn": kw.get("force_fn"), "tail_args": []}
        fn = record["force_fn"]
        if fn is not None:
            record["forces"] = forces.detach().clone()
            record["recomputed"] = fn(pos.detach().clone()).detach().clone()

            def tail(p):
                record["tail_args"].append(p.detach().clone())
                return fn(p)
            kw["force_fn"] = tail
        calls.append(record)
        return real(pos, vel, forces, temperatures, **kw)

    monkeypatch.setattr(M, "batch_langevin_step", spy)
    meta = M.BatchedMetadynamics(n_replicas=N_REP, sequence="A" * L, device="cpu")
    meta.run(COORDS, PAIRS, n_steps=N_STEPS, hill_height=1.0, hill_sigma=1.0,
             hill_freq=HILL_FREQ, max_hills=10, well_tempered=True, bias_factor=5.0)
    return calls


def test_every_step_gets_the_force_at_its_post_update_positions(monkeypatch):
    calls = _run_with_spy(monkeypatch)
    assert len(calls) == N_STEPS

    for i, c in enumerate(calls):
        assert c["force_fn"] is not None, (
            f"step {i}: batch_langevin_step was called without force_fn, so its tail B "
            f"half-kick reused the pre-step force tensor and the step is not symplectic")
        assert len(c["tail_args"]) == 1, (
            f"step {i}: the integrator invoked force_fn {len(c['tail_args'])} times, "
            f"expected exactly one tail kick")
        diff = float((c["recomputed"] - c["forces"]).abs().max())
        assert torch.equal(c["recomputed"], c["forces"]), (
            f"step {i}: force_fn at the incoming coordinates differs from the force "
            f"tensor it was integrated with by {diff:.3e} (max |force| "
            f"{float(c['forces'].abs().max()):.3e}); the metadynamics bias must be part "
            f"of the recomputed force")


def test_the_force_fn_carries_the_metadynamics_bias(monkeypatch):
    seen = {"with_hills": 0}

    def loud_bias(self, pos, hills_centers, hills_heights, sigma):
        if not hills_centers:
            return torch.zeros_like(pos)
        seen["with_hills"] += 1
        return torch.full_like(pos, 50.0)

    monkeypatch.setattr(M.BatchedMetadynamics, "_compute_bias_forces", loud_bias)
    calls = _run_with_spy(monkeypatch)

    assert seen["with_hills"] > 0, (
        "no hill was ever visible to _compute_bias_forces, so this test would be vacuous")

    for i, c in enumerate(calls):
        assert c["force_fn"] is not None, (
            f"step {i}: batch_langevin_step was called without force_fn")
        diff = float((c["recomputed"] - c["forces"]).abs().max())
        assert torch.equal(c["recomputed"], c["forces"]), (
            f"step {i}: force_fn reproduces the physical force only; it differs from the "
            f"force tensor by {diff:.3e}, and the injected bias is 50.0 kJ/mol/nm")
