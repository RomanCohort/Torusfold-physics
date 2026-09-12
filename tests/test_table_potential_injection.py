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

import torch                                  # noqa: E402
import torusfold.scheme2.torch_cgsim as C     # noqa: E402

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
            for which in ("angle", "dihedral", "both"):
                pos = _chain(B=B, dtype=dtype)
                pairs, w = _pairs()

                kw = {}
                if which in ("angle", "both"):
                    kw["angle_potential"] = SHIPPED_ANGLE
                if which in ("dihedral", "both"):
                    kw["dihedral_potential"] = SHIPPED_DIHEDRAL

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
    for name in ("angle_potential", "dihedral_potential"):
        assert name in sig.parameters, f"{name} is missing from cg_energy_forces"
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name} must be keyword-only; a positional slot can be filled by accident")
        assert sig.parameters[name].default is None, (
            f"{name} must default to None -- a non-None default would change every existing call")


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
