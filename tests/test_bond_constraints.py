"""SHAKE/RATTLE distance constraints: src/torusfold/scheme2/rigid_bonds.py.

Why this exists. Two of the three intra-residue distances in the 3-bead model have springs
fitted to sigma = 0.13 A and 0.08 A, which is BELOW the coordinate-error floor of the PDB data
they came from. They measure refinement restraints, not thermal motion, and no dataset can
calibrate them better. A constraint has no width and therefore no constant to get wrong.

What is load-bearing here, and why each check is the shape it is:

  * the ASSEMBLED Jacobian. A 2x2 written out by hand for P-C4'/C4'-N is correct for exactly
    this topology and silently becomes Jacobi iteration -- still convergent, just slower -- the
    day a constraint spans two residues. test_the_assembled_jacobian_matches_a_finite_difference
    checks the assembled G against a central difference of g itself, which knows nothing about
    how G was built.
  * RATTLE IN ONE SOLVE. The velocity constraint is linear in v, so one Newton step lands on the
    manifold and the residual afterwards is roundoff. The test asserts that at the roundoff
    floor AND that the second call changes nothing, rather than "the residual got small".
  * constraints=None IS BIT-IDENTICAL. Every number in this repository's record was produced by
    batch_langevin_step, and the None branch must add no operation at all. The test seeds the
    generator so the O-step noise matches; without that it would compare two different random
    draws and pass while checking nothing.
  * THE NAIVE 3N TEMPERATURE IS PINNED AT 0.778x, not just the corrected one at 300 K. If the
    two ever agree, the correction was not applied -- and a run whose thermostat overshoots at
    440 K would "improve" to 342 K purely on bookkeeping.

Run: python tests/test_bond_constraints.py     (also collectable by pytest)
"""
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch                                          # noqa: E402
import torusfold.scheme2.torch_cgsim as C             # noqa: E402
import torusfold.scheme2.rigid_bonds as R             # noqa: E402

MASS = 110.0
T_K = 300.0
# In nm/ps/amu/kJ/mol the conversion is 1, so kB*T/m is numerically the same number here as it is
# in kJ/mol per amu. Same identity tests/test_integrator_thermostat.py uses.
KBT_OVER_M = C.KB_KJ * T_K / MASS


def _random_chain(L=3, B=2, dtype=torch.float64, seed=21):
    """A jittered chain projected onto the constraint manifold, so no bond starts satisfied."""
    g = torch.Generator().manual_seed(seed)
    con = C.make_intra_constraints(L, mass_amu=MASS)
    return con.shake(torch.randn(B, 3 * L, 3, dtype=dtype, generator=g)), con


def _thermal_vel(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    return math.sqrt(KBT_OVER_M) * torch.randn(*shape, dtype=dtype, generator=g)


# ── the Jacobian ───────────────────────────────────────────────────────────────────────────
def test_the_assembled_jacobian_matches_a_finite_difference():
    """G must be the derivative of g, checked against a central difference of g itself.

    White-box on purpose: _geometry is the only thing that exposes G, and re-deriving it from
    the public API would re-derive it from the same assumption being tested. The finite
    difference knows nothing about scattering, components, or the 2x2 structure.
    """
    con = C.make_intra_constraints(3, mass_amu=MASS)
    pos, _ = _random_chain(L=3, B=2)
    bead_idx, loc_i, loc_j, sigma = con._idx(pos.device, pos.dtype)
    x = con._local(pos, bead_idx)
    _r, _u, G, _g = con._geometry(x, loc_i, loc_j, sigma)

    h = 1e-6
    G_fd = torch.zeros_like(G)
    for k in range(con.n_components):
        for lb in range(con.n_local_beads):
            for c in range(3):
                pp, pm = x.clone(), x.clone()
                pp[:, k, lb, c] += h
                pm[:, k, lb, c] -= h
                gp = con._geometry(pp, loc_i, loc_j, sigma)[3][:, k, :]
                gm = con._geometry(pm, loc_i, loc_j, sigma)[3][:, k, :]
                G_fd[:, k, :, lb * 3 + c] = (gp - gm) / (2.0 * h)

    worst = float((G.double() - G_fd.double()).abs().max())
    assert worst < 1e-8, (
        f"the assembled Jacobian differs from d(g)/dx by {worst:.3e}; the scatter is no longer "
        f"the derivative of the constraint functions")
    # And it must not be trivially zero or trivially full: each row has exactly +u on its first
    # endpoint and -u on its second, so exactly 6 of every 9 columns are nonzero.
    nz = (G_fd.abs() > 1e-12).sum(-1)
    assert bool((nz == 6).all()), f"expected 6 nonzero columns per row, got {nz.flatten()[:8]}"


def test_the_jacobian_carries_the_shared_bead_coupling():
    """The diagonal-only Jacobian is the failure mode the assembly exists to prevent.

    For one residue with unit bond vectors u0 (P->C4') and u1 (C4'->N) sharing C4',
    J = (1/m)[[2, -u0.u1], [-u0.u1, 2]], so det = (4 - (u0.u1)^2)/m^2, which is >= 3/m^2.
    The system is uniformly well conditioned -- there is no geometry, however stretched, at which
    the two constraints become dependent. Asserting the off-diagonal is nonzero is what
    distinguishes Newton from the coupled-blind version.
    """
    con = C.make_intra_constraints(4, mass_amu=MASS)
    pos, _ = _random_chain(L=4, B=8, seed=5)
    bead_idx, loc_i, loc_j, sigma = con._idx(pos.device, pos.dtype)
    x = con._local(pos, bead_idx)
    _r, u, G, _g = con._geometry(x, loc_i, loc_j, sigma)
    Gf = G.reshape(*G.shape[:3], -1)
    J = (Gf @ Gf.transpose(-1, -2)) / MASS

    offdiag_want = -(u[:, :, 0] * u[:, :, 1]).sum(-1) / MASS
    assert float(J[..., 0, 0].sub(2.0 / MASS).abs().max()) < 1e-12, "J00 must be 2/m"
    assert float(J[..., 1, 1].sub(2.0 / MASS).abs().max()) < 1e-12, "J11 must be 2/m"
    assert float((J[..., 0, 1] - offdiag_want).abs().max()) < 1e-12, (
        "the off-diagonal is not -u0.u1/m, so the shared C4' bead is not coupling the two "
        "constraints -- the solve would be Jacobi, not Newton")
    # A vacuity guard, checked on the MAX and deliberately not the min. u0.u1 goes through zero
    # whenever a residue happens to sit near a 90-degree P-C4'-N angle, which across 32 samples
    # is routine -- asserting a floor on the minimum would fail on a perfectly good geometry.
    assert float(J[..., 0, 1].abs().max()) > 1e-3, (
        "the coupling came out numerically zero everywhere, so the comparison above is vacuous")

    det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    assert float(det.min()) >= 3.0 / MASS ** 2 - 1e-15, (
        f"det(J) fell below 3/m^2 = {3.0 / MASS ** 2:.3e}. The uniform conditioning claim in "
        f"rigid_bonds._solve is wrong and the singularity handling needs revisiting")


# ── the two projections ────────────────────────────────────────────────────────────────────
def test_rattle_is_exact_in_one_solve():
    """One call must land on Gv = 0, and a second call must find nothing left to do.

    Not a claim that the residual is small: the constraint is LINEAR in v, so the Newton step is
    exact and what remains is roundoff from the dot products, scale-free in the constraint count.
    """
    con = C.make_intra_constraints(3, mass_amu=MASS)
    for dtype, floor in ((torch.float64, 1e-12), (torch.float32, 1e-5)):
        pos, _ = _random_chain(L=3, B=16, dtype=dtype, seed=31)
        vel = _thermal_vel((16, 9, 3), dtype, 32) * 50.0     # deliberately far off the manifold

        before = con.velocity_residual(pos, vel)
        once = con.rattle(pos, vel)
        after = con.velocity_residual(pos, once)
        twice = con.rattle(pos, once)

        assert float(before.min()) > 1.0, "the test velocity is already constraint-consistent"
        assert float(after.max()) < floor, (
            f"one RATTLE left |Gv| = {float(after.max()):.3e} (floor {floor:.0e}); the velocity "
            f"projection is no longer a single exact solve")
        # The idempotence tolerance is the SAME dtype-aware floor, scaled by the 50x the test
        # velocities were multiplied by. A single fixed 1e-12 here passes at float64 and fails at
        # float32 for the honest reason -- float32 cannot resolve 1e-12 of a 7 nm/ps velocity.
        moved = float((twice.double() - once.double()).abs().max())
        assert moved < floor * 50.0, (
            f"a second RATTLE moved the velocity by {moved:.3e}, so the first one did not land "
            f"on the manifold")


def test_shake_holds_the_bonds_over_a_long_run():
    """3000 constrained steps: the bond lengths must not drift, and the caps must not be hit.

    A tolerance that is quietly exceeded on one step and re-satisfied on the next is invisible in
    a final snapshot, which is why this checks every step's worst residual rather than the last.
    """
    con = C.make_intra_constraints(4, mass_amu=MASS)
    pos, _ = _random_chain(L=4, B=8, seed=41)
    vel = _thermal_vel(pos.shape, torch.float64, 42)
    temps = torch.full((8,), T_K, dtype=torch.float64)

    def spring(p):
        return -50.0 * p                      # a real pull, so SHAKE has work to do every step

    worst = 0.0
    for _ in range(3000):
        pos, vel = C.batch_langevin_step(pos, vel, spring(pos), temps, dt_ps=0.002,
                                         friction=1.0, mass_amu=MASS, force_fn=spring,
                                         constraints=con)
        worst = max(worst, float((con.residual(pos).abs() / con.sigma.view(-1)).max()))

    assert worst <= R._TOL_REL[torch.float64], (
        f"worst relative bond error over 3000 steps was {worst:.3e}, above the {1e-9:.0e} the "
        f"solver claims to hold")
    assert con.last_shake_iters < con.max_iter, (
        "SHAKE ran to its iteration cap; it is not converging, it is being stopped")


# ── the integrator's None path ─────────────────────────────────────────────────────────────
def test_the_none_path_is_bit_identical():
    """constraints=None must add no operation to batch_langevin_step.

    The O step draws from the global RNG, so the two calls have to be seeded identically or the
    test compares two different noise draws and passes without checking anything.
    """
    for dtype in (torch.float64, torch.float32):
        g = torch.Generator().manual_seed(51)
        pos = torch.randn(4, 12, 3, dtype=dtype, generator=g)
        vel = torch.randn(4, 12, 3, dtype=dtype, generator=g)
        forces = torch.randn(4, 12, 3, dtype=dtype, generator=g)
        temps = torch.full((4,), T_K, dtype=dtype)

        torch.manual_seed(20260218)
        p_a, v_a = C.batch_langevin_step(pos.clone(), vel.clone(), forces, temps)
        torch.manual_seed(20260218)
        p_b, v_b = C.batch_langevin_step(pos.clone(), vel.clone(), forces, temps,
                                         constraints=None)
        assert torch.equal(p_a, p_b) and torch.equal(v_a, v_b), (
            f"constraints=None changed the trajectory at {dtype}; the None branch must perform "
            f"no tensor operation at all")


def test_the_constraints_do_not_touch_the_forces_tensor():
    """A constrained step must not write into forces, and must return the same shapes.

    A constraint is a projection on x and v, not a term in the Hamiltonian. If it ever starts
    contributing to F, every energy printed alongside a constrained run silently becomes a
    different quantity than the one the exchange criterion uses.
    """
    con = C.make_intra_constraints(3, mass_amu=MASS)
    pos, _ = _random_chain(L=3, B=4, seed=61)
    vel = _thermal_vel(pos.shape, torch.float64, 62)
    forces = torch.randn_like(pos)
    snapshot = forces.clone()
    temps = torch.full((4,), T_K, dtype=torch.float64)

    p2, v2 = C.batch_langevin_step(pos, vel, forces, temps, constraints=con)

    assert torch.equal(forces, snapshot), "batch_langevin_step mutated the caller's forces"
    assert p2.shape == pos.shape and v2.shape == vel.shape
    assert p2 is not pos and v2 is not vel, (
        "the constrained path returned its inputs; batch_langevin_step's result must be usable "
        "by a caller that still holds the previous step")


# ── tolerances, refusal, ordering ──────────────────────────────────────────────────────────
def test_the_dtype_tolerance_is_not_interchangeable():
    """A float64 tolerance on the float32 path raises on a step the float32 path handles.

    This is the claim in rigid_bonds._TOL_REL made falsifiable. Without it, "dtype-aware" is a
    comment, and the first person to unify the two constants gets a solver that refuses every
    production step.
    """
    assert R._TOL_REL[torch.float32] > R._TOL_REL[torch.float64] * 1e3, (
        "the two defaults are within 1e3 of each other, which is not a dtype-aware pair")

    L = 8
    g = torch.Generator().manual_seed(71)
    pos32 = torch.randn(4, 3 * L, 3, dtype=torch.float32, generator=g)

    strict = C.make_intra_constraints(L, mass_amu=MASS, tol_rel=R._TOL_REL[torch.float64])
    try:
        strict.shake(pos32)
    except RuntimeError as exc:
        assert "did not converge" in str(exc), f"raised for the wrong reason: {exc}"
    else:
        raise AssertionError(
            "a float64-grade tolerance converged on the float32 path; either the tolerance is "
            "no longer dtype-aware, or float32 is doing better than float32 can")

    lenient = C.make_intra_constraints(L, mass_amu=MASS)      # the float32 default
    lenient.shake(pos32)                                      # must not raise
    assert lenient.last_shake_iters < lenient.max_iter


def test_a_non_convergent_shake_raises_about_convergence():
    """max_iter=1 on an unsatisfied start must raise, and say so in those words."""
    con = C.make_intra_constraints(2, mass_amu=MASS, max_iter=1)
    g = torch.Generator().manual_seed(81)
    pos = torch.randn(2, 6, 3, dtype=torch.float64, generator=g)   # nowhere near satisfied
    try:
        con.shake(pos)
    except RuntimeError as exc:
        assert "did not converge" in str(exc) and "max_iter" not in str(exc), (
            f"non-convergence raised a message that does not name the condition: {exc}")
    else:
        raise AssertionError("SHAKE returned a silent result after hitting its iteration cap")


def test_the_residual_comes_back_in_the_callers_order():
    """The solver groups constraints by component; residual() must undo that.

    "constraint 7" has to mean the same thing to the caller and to the solver, or a diagnostic
    that reports the worst bond names the wrong atom pair.
    """
    pairs = torch.tensor([[3, 4], [4, 5], [0, 1], [1, 2]], dtype=torch.long)   # component order
    targets = torch.tensor([0.30, 0.31, 0.40, 0.41], dtype=torch.float64)      # is 1, 0 here
    con = R.DistanceConstraints(pairs, targets, mass_amu=MASS, name="ordering")
    assert con.n_components == 2 and con.n_constraints == 4

    # Every literal gets an explicit dtype. torch.tensor([0.30, ...]) builds a FLOAT32 tensor
    # and only then casts it, so the float64 this ends up in carries float32's rounding of 0.30
    # -- a residual of 1.2e-8 on a distance that is exact, which reads exactly like a solver bug.
    pos = torch.zeros(1, 6, 3, dtype=torch.float64)
    rows = ((0, [0.0, 0.0, 0.0]),
            (1, [0.40, 0.0, 0.0]),          # g = 0.40 - 0.40 = 0
            (2, [1.41, 0.0, 0.0]),          # g = 1.01 - 0.41 = 0.60
            (3, [0.0, 5.0, 0.0]),
            (4, [0.30, 5.0, 0.0]),          # g = 0.30 - 0.30 = 0
            (5, [0.30, 5.31, 0.0]))         # g = 0.31 - 0.31 = 0
    for i, row in rows:
        pos[0, i] = torch.tensor(row, dtype=torch.float64)

    g = con.residual(pos)[0]
    assert abs(float(g[0])) < 1e-12, f"constraint 0 is the (3,4) pair, got g = {float(g[0])}"
    assert abs(float(g[2])) < 1e-12, f"constraint 2 is the (0,1) pair, got g = {float(g[2])}"
    assert abs(float(g[3]) - 0.60) < 1e-12, f"constraint 3 is the (1,2) pair, got g = {float(g[3])}"


def test_a_degenerate_constraint_set_is_refused():
    """Each refusal by name, because each one silently produces a plausible number instead."""
    cases = (
        ("self pair", torch.tensor([[0, 0]]), torch.tensor([0.4])),
        ("no constraints", torch.zeros((0, 2), dtype=torch.long), torch.zeros(0)),
        ("target count", torch.tensor([[0, 1], [1, 2]]), torch.tensor([0.4])),
        ("negative index", torch.tensor([[-1, 1]]), torch.tensor([0.4])),
        ("zero target", torch.tensor([[0, 1]]), torch.tensor([0.0])),
    )
    for what, pairs, targets in cases:
        try:
            R.DistanceConstraints(pairs, targets, mass_amu=MASS)
        except (ValueError, NotImplementedError):
            continue
        raise AssertionError(f"{what} was accepted by DistanceConstraints")

    # A ragged constraint set would need padding, and padded beads are constraints on a geometry
    # that does not exist. Two residues and a third constraint on a fourth bead is that shape.
    try:
        R.DistanceConstraints(torch.tensor([[0, 1], [1, 2], [3, 4]]),
                              torch.tensor([0.4, 0.3, 0.5]), mass_amu=MASS)
    except NotImplementedError as exc:
        assert "not all the same shape" in str(exc), exc
    else:
        raise AssertionError("a ragged constraint set was accepted")


# ── the numbers the scheme has to reproduce ────────────────────────────────────────────────
def test_constrained_equipartition_gives_300k_and_the_naive_3n_reading_is_0778():
    """The thermostat must reach 300 K on the constrained manifold, read with 3N - C.

    Free replicas are the cheapest exact case: with no forces the only dynamics is the Langevin
    O step plus the two projections, so the stationary kinetic energy per remaining degree of
    freedom is fixed by the fluctuation-dissipation balance and nothing else.

    Two details are load-bearing, and both were arrived at by getting them wrong first.

    Starting velocities are drawn at the thermal scale. From rest, or from unit-variance noise
    (44x too hot), the run relaxes toward 300 K with a 1/friction = 500-step time constant, and
    a short test measures the transient: it reads 6111 K at step 200 and 408 K at step 1200, which
    looks exactly like a broken RATTLE and is not.

    The mean is taken over FOUR independent batches, not one. Measured over 12 seeds at B=2048,
    N=500, the per-batch spread is sd 2.65 K (0.88%, matching sqrt(2/14)*300/sqrt(2048) = 0.83%
    for 14 degrees of freedom over 2048 replicas) with a batch mean of 299.50 +- 0.76 K. So a
    single batch sits 2.4 sigma away from 300 roughly one time in sixty -- and a test that pins
    one batch at +-2% is a coin flip, not a measurement. Averaging four brings the standard error
    to ~0.17%, which is what makes the +-1% below a statement about the scheme rather than about
    the seed.
    """
    L, B, N, N_BATCH = 2, 2048, 300, 4
    per_batch = []
    for seed in range(1, N_BATCH + 1):
        con = C.make_intra_constraints(L, mass_amu=MASS)
        g = torch.Generator().manual_seed(seed)
        pos = con.shake(torch.randn(B, 3 * L, 3, dtype=torch.float32, generator=g))
        vel = con.rattle(pos, math.sqrt(KBT_OVER_M) * torch.randn(
            B, 3 * L, 3, dtype=torch.float32, generator=g))
        temps = torch.full((B,), T_K, dtype=torch.float32)
        zero = lambda p: torch.zeros_like(p)                               # noqa: E731

        for _ in range(N):
            pos, vel = C.batch_langevin_step(pos, vel, torch.zeros_like(pos), temps, dt_ps=0.002,
                                             friction=1.0, mass_amu=MASS, force_fn=zero,
                                             constraints=con)

        T_fixed = C.kinetic_temperature(vel, MASS, n_constraints=con.n_constraints)
        T_naive = C.kinetic_temperature(vel, MASS, n_constraints=0)
        per_batch.append(float(T_fixed.mean()))

        # The ratio is an algebraic identity and must hold to roundoff on every batch, including
        # the ones that would otherwise be discarded -- it is the correction, not the physics.
        want = (3 * 3 * L - con.n_constraints) / (3 * 3 * L)
        assert abs(want - 7.0 / 9.0) < 1e-12, f"2L on 3L should be 7/9, got {want}"
        ratio = float(T_naive.mean()) / float(T_fixed.mean())
        assert abs(ratio - want) < 1e-3, (
            f"the naive 3N reading is {ratio:.4f} of the corrected one, not the {want:.4f} the "
            f"degree-of-freedom count demands -- the correction was dropped or double applied")

    T_mean = sum(per_batch) / len(per_batch)
    assert abs(T_mean / T_K - 1.0) < 0.01, (
        f"constrained equipartition gave {T_mean:.2f} K over {N_BATCH} batches against a 300 K "
        f"target ({T_mean / T_K - 1:+.2%}, per batch {[round(t, 2) for t in per_batch]}); the "
        f"constrained Langevin scheme is not sampling the canonical distribution")


def test_the_factory_reads_the_constants_at_call_time():
    """A frozen copy of BOND_P_C4 would hold a stale target while printing the new value.

    tests/test_ff_bonded_targets.py found three of those in torch_cgsim; the factory is new code
    with the same exposure, so it gets the same check.
    """
    before = C.make_intra_constraints(2, mass_amu=MASS)
    assert float(before.targets[0]) == float(C.BOND_P_C4)
    assert float(before.targets[1]) == float(C.BOND_C4_N)

    saved_pc, saved_cn = C.BOND_P_C4, C.BOND_C4_N
    try:
        C.BOND_P_C4, C.BOND_C4_N = 0.401, 0.349
        after = C.make_intra_constraints(2, mass_amu=MASS)
        assert float(after.targets[0]) == 0.401 and float(after.targets[1]) == 0.349, (
            f"the factory kept a stale target: {after.targets[:2].tolist()}")
        assert float(before.targets[0]) != 0.401, (
            "the already-built set moved with the constant, so it holds a reference rather than "
            "a value and cannot be snapshotted")
    finally:
        C.BOND_P_C4, C.BOND_C4_N = saved_pc, saved_cn

    assert float(C.make_intra_constraints(2, mass_amu=MASS).targets[0]) == saved_pc


if __name__ == "__main__":
    # Same runner tests/test_table_potential_injection.py uses, so this file runs under pytest
    # and standalone alike. The equipartition case is the slow one at ~2 s.
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    _failed = 0
    for _fn in _tests:
        try:
            _fn()
            print(f"  PASS  {_fn.__name__}")
        except Exception as _exc:                     # noqa: BLE001
            _failed += 1
            print(f"  FAIL  {_fn.__name__}: {_exc!r}")
    print(f"\n{len(_tests) - _failed}/{len(_tests)} passed")
    sys.exit(1 if _failed else 0)
