"""The injected-potential path of cg_energy_forces.

cg_energy_forces gained two keyword-only parameters, angle_potential and dihedral_potential, so
that a sampler can run the full production field with the two backbone angular terms replaced --
by tabulated potentials, or by a fitted Fourier series. That is the sampler loop IBI has been
missing; docs/statistical_potentials_as_forces.md section 3ba fixes the ordering it belongs to
("block J flat -> write the sampler loop -> then talk about updates", line 2763).

The whole safety argument for the change is the default. With both parameters None the function
must be BIT-IDENTICAL to what it was before they existed, because around twenty callers across
scripts/ and src/ pass no potentials and every one of them produced a number in the record.

This file states that as an identity, not as a comparison against reference values: hand the
injection a callable that IS the shipped expression, and require torch.equal on both the energy
and the force. A tolerance would also pass if both sides were wrong the same way, and it would
miss the failure mode that actually threatens this code -- the accumulation. The running total is
float32 in the production dtype, float addition is not associative, so "total_E += e_a;
total_F += f_a" has to stay on one line and in that order. tests/test_backbone_13_terms.py is
sensitive at exactly that scale.
"""
import inspect
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import torch                                  # noqa: E402
import torusfold.scheme2.torch_cgsim as C     # noqa: E402
import cg_potentials as P                     # noqa: E402

# Deliberately no `import pytest`: this machine has no pytest installed, and the cases are loops
# inside the test functions rather than @parametrize so that `python tests/test_<name>.py` works
# as well as `pytest tests/`. Same convention as tests/test_ibi_bonded.py, which is the one file
# in tests/ that does not import pytest.


def _chain(L=9, B=1, dtype=torch.float64):
    """A straight, deliberately unclosed chain: P(i) at 0.59 i along x, C4' and N9/N1 off-axis.

    The same construction tests/test_backbone_13_terms.py uses. Unclosed matters for the same
    reason it does there: residue 0 sits far from residue L-1, so the BSJ term is a real number
    instead of one hiding at its own minimum.
    """
    pos = torch.zeros(1, 3 * L, 3, dtype=dtype)
    for i in range(L):
        base = torch.tensor([0.59 * i, 0.0, 0.0], dtype=dtype)
        pos[0, 3 * i] = base
        pos[0, 3 * i + 1] = base + torch.tensor([0.12, 0.32, 0.0], dtype=dtype)
        pos[0, 3 * i + 2] = pos[0, 3 * i + 1] + torch.tensor([0.07, 0.26, 0.13], dtype=dtype)
    if B > 1:
        # Different internal geometry per replica, so a batch-axis mix-up inside the injection
        # has something to mix up. A rigid offset would not do: it leaves every internal
        # coordinate identical across the batch.
        g = torch.Generator().manual_seed(20260218)
        pos = pos.repeat(B, 1, 1) + 0.05 * torch.randn((B,) + tuple(pos.shape[1:]),
                                                       generator=g, dtype=dtype)
        return pos
    return pos


def _pairs():
    pairs = torch.tensor([[0, 4], [1, 6]], dtype=torch.long)
    return pairs, torch.ones(pairs.shape[0], dtype=torch.float64)


def _cell(p):
    c = C.GPUCellList(cell_size=1.5)
    c.build(p)
    return c


SHIPPED_ANGLE = lambda p: C._angle_f(p, C.K_ANGLE, math.cos(C.ANGLE_PPP))
SHIPPED_DIHEDRAL = lambda p: C._dihedral_f(p, C.K_DIH, math.cos(C.DIH_PPPP))


def SHIPPED_BOND(p):
    """The field's own P(i)-P(i+1) term, on the same L-1 windows the bond section builds.

    _bond_f takes explicit bead indices rather than deriving them, so the window set has to be
    spelled out here. No closure window: the circular P(0)-P(L-1) link is K_BSJ's term, and
    including it would make this differ from the shipped path for a reason that is not the
    injection's fault.
    """
    L = p.shape[1] // 3
    idx = torch.arange(L - 1, device=p.device)
    return C._bond_f(p, 3 * idx, 3 * (idx + 1), C.K_BB, C.BOND_P_NEXT)


def test_injected_shipped_harmonic_is_bit_identical():
    """Handing the injection the shipped expression must reproduce the shipped path exactly.

    This is the test that makes the feature safe to land: it exercises the plumbing -- batch
    axis, dtype, device, the no-grad scope -- against a known answer, without needing a table,
    a reference number, or a decision about which potential to use.

    float32 matters as much as float64: the production dtype is float32 on the batched path, and
    it is the one where a reordered accumulation would show up.
    """
    for dtype in (torch.float64, torch.float32):
        for B in (1, 8):
            for which in ("angle", "dihedral", "bond", "both", "all"):
                pos = _chain(B=B, dtype=dtype)
                pairs, w = _pairs()

                kw = {}
                if which in ("angle", "both", "all"):
                    kw["angle_potential"] = SHIPPED_ANGLE
                if which in ("dihedral", "both", "all"):
                    kw["dihedral_potential"] = SHIPPED_DIHEDRAL
                if which in ("bond", "all"):
                    kw["bond_potential"] = SHIPPED_BOND

                e_ref, f_ref = C.cg_energy_forces(pos, pairs, w, cell_list=_cell(pos),
                                                  force_cap=None)
                e_inj, f_inj = C.cg_energy_forces(pos, pairs, w, cell_list=_cell(pos),
                                                  force_cap=None, **kw)

                tag = f"{dtype} B={B} {which}"
                assert torch.equal(e_ref, e_inj), (
                    f"energy differs between the shipped path and the injected shipped "
                    f"expression [{tag}] -- the default branch is no longer inert")
                assert torch.equal(f_ref, f_inj), (
                    f"force differs between the shipped path and the injected shipped "
                    f"expression [{tag}] -- check that total_E += e_a; total_F += f_a was not "
                    f"reordered, split, or moved")


def test_injection_survives_the_samplers_no_grad_scope():
    """The sampler evaluates forces inside torch.no_grad(); the injection must still return force.

    _dihedral_f already carries its own enable_grad block for this reason. A potential built on
    autograd inherits the same requirement, so this pins it rather than assuming it.
    """
    pos = _chain(B=4)
    pairs, w = _pairs()
    with torch.no_grad():
        _e, f = C.cg_energy_forces(pos, pairs, w, cell_list=_cell(pos), force_cap=None,
                                   angle_potential=SHIPPED_ANGLE,
                                   dihedral_potential=SHIPPED_DIHEDRAL)
    assert torch.isfinite(f).all()
    assert float(f.abs().amax()) > 0.0


def test_the_two_parameters_are_keyword_only():
    """Keyword-only is what keeps the ~20 existing positional call sites unbreakable.

    Every caller in scripts/ and src/ passes pos, pairs, w positionally and the rest by keyword.
    Adding these two after a bare "*" means no future reordering can silently shift an argument
    into a potential slot -- which would not raise, it would sample a different field.
    """
    sig = inspect.signature(C.cg_energy_forces)
    for name in ("angle_potential", "dihedral_potential", "bond_potential"):
        assert name in sig.parameters, f"{name} is missing from cg_energy_forces"
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name} must be keyword-only; a positional slot can be filled by accident")
        assert sig.parameters[name].default is None, (
            f"{name} must default to None -- a non-None default would change every existing call")


def test_the_wrapped_bond_harmonic_reproduces_the_field_exactly():
    """The gate that makes the bond injection landable, stated end to end.

    resolve_spec -> make_potential -> cg_energy_forces must reproduce the field's own analytic
    bond term with NO difference at all. Not "close": the analytic _bond_f and an autograd
    potential computing the same expression can part company in the last bit, and if they do
    then every before/after comparison across this change carries an offset that is an artefact
    of the plumbing rather than of the physics.

    This is also the one test that would catch the spec resolving to the WRONG coordinate's
    force function -- an angle table interpolated at a distance still returns finite numbers.

    Why the force is not asserted bit-identical here, unlike the angle/dihedral case:
    SHIPPED_ANGLE is literally C._angle_f, the same function the default branch calls, so
    equality is structural. force_reference's bond harmonic is a SECOND implementation of the
    same expression, differentiated by autograd, and `a*(b/c)` and `(a*b)/c` round differently.
    Measured on this geometry the difference is exactly one ULP, on 6 of 648 components at
    float64 and 8 of 648 at float32, with the energy identical everywhere. So the energy gets
    torch.equal and the force gets a one-ULP tolerance.

    Use test_injected_shipped_harmonic_is_bit_identical's "bond" case for the structural check:
    that one injects _bond_f itself and does require exact equality.
    """
    spec = P.resolve_spec(f"harmonic:K={C.K_BB},r0={C.BOND_P_NEXT}", "bb_bond")
    tol = {torch.float64: 1e-14, torch.float32: 1e-6}
    for dtype in (torch.float64, torch.float32):
        for B in (1, 8):
            pos = _chain(B=B, dtype=dtype)
            pairs, w = _pairs()
            e_ref, f_ref = C.cg_energy_forces(pos, pairs, w, cell_list=_cell(pos), force_cap=None)
            e_inj, f_inj = C.cg_energy_forces(pos, pairs, w, cell_list=_cell(pos), force_cap=None,
                                              bond_potential=P.make_potential("bb_bond", spec))
            tag = f"{dtype} B={B}"
            assert torch.equal(e_ref, e_inj), f"energy differs from the shipped bond term [{tag}]"
            scale = max(float(f_ref.abs().max()), 1.0)
            worst = float((f_ref.double() - f_inj.double()).abs().max()) / scale
            assert worst < tol[dtype], (
                f"force differs from the shipped bond term by {worst:.3e} [{tag}], more than the "
                f"one ULP that autograd-vs-analytic ordering accounts for")


def test_an_injected_bond_wins_over_relax_bond_k():
    """relax_bond_k must not be able to override an injected bond potential.

    isrnaclong.py passes relax_bond_k into BatchedREMD2D and no call site there forwards it, so
    the plumbing is currently dead. The day someone fixes that, an injected U(r) -- which has no
    single spring constant to scale -- would be silently rescaled by a harmonic that nothing
    else in the run uses. Pin the precedence now rather than discover it then.
    """
    pos = _chain(B=2)
    pairs, w = _pairs()
    spec = P.resolve_spec(f"harmonic:K={C.K_BB},r0={C.BOND_P_NEXT}", "bb_bond")
    pot = P.make_potential("bb_bond", spec)
    e_a, f_a = C.cg_energy_forces(pos, pairs, w, cell_list=_cell(pos), force_cap=None,
                                  bond_potential=pot, relax_bond_k=500.0)
    e_b, f_b = C.cg_energy_forces(pos, pairs, w, cell_list=_cell(pos), force_cap=None,
                                  bond_potential=pot, relax_bond_k=5000.0)
    assert torch.equal(e_a, e_b), (
        "relax_bond_k changed the energy while a bond_potential was injected -- the harmonic is "
        "still live underneath the injected term")
    assert torch.equal(f_a, f_b), "relax_bond_k changed the force under an injected bond"


def test_the_bond_specs_that_do_not_transfer_are_refused():
    """A distance is unbounded; three of the four cosine specs are meaningless for it.

    Each is refused by name. Silently evaluating them would return plausible finite numbers --
    table_jac would take ln(1-q^2) of a quantity that can exceed 1 and clamp to a constant.
    """
    for bad in ("shipped_harmonic", "table_jac", "table_jac:0.05", "fourier:K1=1.0"):
        try:
            P.resolve_spec(bad, "bb_bond")
        except SystemExit:
            continue
        raise AssertionError(f"{bad!r} was accepted for bb_bond, where it is not defined")
    # and the ones that do transfer
    for good, want in (("table", "table"),
                       ("table_wall", ("table_wall", 200.0)),
                       ("table_wall:500", ("table_wall", 500.0))):
        assert P.resolve_spec(good, "bb_bond") == want, good


if __name__ == "__main__":
    # Same runner tests/test_ibi_bonded.py uses. pytest is not installed on this machine, and a
    # regression that cannot be run is not a regression.
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
