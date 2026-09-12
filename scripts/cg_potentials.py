"""Spec string -> potential callable, for injection into cg_energy_forces.

cg_energy_forces takes angle_potential / dihedral_potential, keyword-only, defaulting to None
(which selects the shipped harmonic and is bit-identical to before they existed). This module is
what a sampler hands it: it parses a spec string off the command line, resolves it to the form
scripts/force_reference.py already computes, and wraps that in the (pos) -> (E, F) contract.

It deliberately contains no energy expression of its own. The dihedral's four candidate energies
live in force_reference._v_fn and are cited as numbers throughout docs/dihedral_table_decision.md;
a second copy here would be the fifth place a force could drift from its own energy, which is the
defect scripts/audit_field_state.py exists to catch. The angle's expressions were added to
force_reference alongside them (angle_force), for the same reason.

Why the angle needs its own spec vocabulary
-------------------------------------------
"shipped_harmonic" is not a well-defined spec for the angle. The module constant is K_ANGLE=28.1,
but every production caller passes relax_angle_k=200.0 (isrnaclong.py:1950, :2087) -- a factor of
7 -- so a sampler that inherited "the shipped angle" could mean either one and would print neither.
force_reference._v_fn refuses that spec for the angle rather than choosing. Use harmonic:K=,q0=
and say which; the sampler echoes the resolved spec into its provenance line.

Specs (after resolve_spec; the bracket form is what force_reference._v_fn takes)
-------------------------------------------------------------------------------
    table                     the stored table, interpolated at bin centres. Measure-naive.
    table_jac                 table(q) - 0.5*kBT*ln(1-q^2), the torsion Jacobian. Exact form;
                              its force diverges as q -> +-1.
    table_jac:<eps>           the same with 1-q^2 floored at eps^2. The floor depends on q^2, so
                              it is symmetric: for |q| > sqrt(1-eps^2) the Jacobian force is
                              identically zero on both the cis and trans sides.
    harmonic:K=<k>,q0=<q0>    0.5*k*(q - q0)^2
    fourier:K1=..,K2=..       sum_n K_n * T_n(q), T_n the Chebyshev polynomial = cos(n*phi).
                              Coefficients are NAMED by their n, never listed positionally: an
                              off-by-one on n=0 silently gives a potential with a different
                              period and no error anywhere.
    shipped_harmonic          dihedral only; refused for the angle (see above)

Run: python scripts/cg_potentials.py [--check]
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import force_reference as FR                  # noqa: E402

COORDS = ("angle", "dihedral")
_FORCE_FN = {"angle": FR.angle_force, "dihedral": FR.dihedral_force}


def _kv(s):
    """`K1=-2.78,K2=-1.33` -> {"K1": -2.78, "K2": -1.33}, refusing a bare number."""
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        name, eq, val = part.partition("=")
        if not eq:
            raise SystemExit(
                f"expected name=value in {part!r}; coefficients are named (K1=.., K2=..) so an "
                f"off-by-one on n cannot pass silently as a different potential")
        try:
            out[name.strip()] = float(val)
        except ValueError:
            raise SystemExit(f"{val!r} is not a number in {part!r}") from None
    return out


def resolve_spec(text, coord=None):
    """Parse a CLI spec string into what force_reference._v_fn expects."""
    text = (text or "").strip()
    if text == "shipped_harmonic":
        if coord == "angle":
            raise SystemExit(
                "shipped_harmonic is not defined for the angle: K_ANGLE=28.1 and the production "
                "relax_angle_k=200.0 are 7x apart. Use harmonic:K=<k>,q0=<q0> and say which.")
        return text
    if text == "table":
        return text
    if text == "table_jac":
        return ("table_jac", None)
    if text.startswith("table_jac:"):
        return ("table_jac", float(text.split(":", 1)[1]))
    if text.startswith("harmonic:"):
        kv = _kv(text.split(":", 1)[1])
        for need in ("K", "q0"):
            if need not in kv:
                raise SystemExit(f"harmonic: needs {need}= (got {text!r})")
        return ("harmonic", kv["K"], kv["q0"])
    if text.startswith("fourier:"):
        kv = _kv(text.split(":", 1)[1])
        if not kv:
            raise SystemExit("fourier: needs at least K1=..")
        n_max = 0
        for name in kv:
            if not (name.startswith("K") and name[1:].isdigit()):
                raise SystemExit(f"fourier coefficients are named K<n>, got {name!r}")
            n_max = max(n_max, int(name[1:]))
        K = [0.0] * (n_max + 1)
        for name, val in kv.items():
            K[int(name[1:])] = val
        return ("fourier", K)
    raise SystemExit(
        f"unknown potential spec {text!r}; expected table | table_jac[:eps] | "
        f"harmonic:K=..,q0=.. | fourier:K1=..,K2=.. | shipped_harmonic")


def describe(spec):
    """One provenance line for a resolved spec. The sampler prints this.

    It exists because the constant fingerprint cannot see a shape change: _FINGERPRINT names
    constants, and a potential replaces the SHAPE of a term while leaving every constant alone.
    ibi_round0.py's own docstring records that exactly this hole invalidated three earlier runs.
    """
    if spec == "shipped_harmonic":
        return "shipped_harmonic"
    if spec == "table":
        return "table"
    if isinstance(spec, tuple) and spec[0] == "table_jac":
        return "table_jac(exact)" if spec[1] is None else f"table_jac(eps={spec[1]:g})"
    if isinstance(spec, tuple) and spec[0] == "harmonic":
        return f"harmonic(K={spec[1]:g},q0={spec[2]:g})"
    if isinstance(spec, tuple) and spec[0] == "fourier":
        K = spec[1]
        body = ", ".join(f"K{n}={K[n]:g}" for n in range(len(K)) if K[n] != 0)
        return f"fourier({body})"
    return repr(spec)


def use_table_file(path, coord=None):
    """Point the table specs at a specific file instead of the shipped one.

    force_reference loads the shipped tables once and caches them on its module, which is right
    for a reference implementation always asked about the same field. An IBI round is not: round
    N has to SAMPLE under round N-1's table, which is a different file. Without this call,
    passing a table file to the sampler would move the BINNING and leave the POTENTIAL on round
    0 -- a round that looks converged because the thing being updated and the thing being
    sampled were never the same object.

    coord=None sets both; the file only needs the keys for the coordinates actually asked about.
    """
    z = np.load(path)
    for c, attr in (("angle", "_ANGLE_TABLE"), ("dihedral", "_TABLE")):
        if coord is not None and c != coord:
            continue
        if f"{c}__U" not in z.files:
            continue
        setattr(FR, attr, {"lo": float(z[f"{c}__lo"]),
                           "binw": float(z[f"{c}__binw"]),
                           "U": torch.tensor(z[f"{c}__U"], dtype=torch.float64)})


def make_potential(coord, spec):
    """Return pot(pos_nm: (B, N, 3)) -> (E (B,), F (B, N, 3)), the contract cg_energy_forces wants.

    The returned callable is force_reference's function with a dtype guard, not a re-derivation:
    E and F are cast back to pos.dtype, because the tables are float64 and the batched production
    path is float32, and `total_E += e_a` on a float32 running total raises on a float64 operand
    rather than promoting silently.

    The table this closes over is whichever use_table_file() last installed; call that first if
    the round is not running the shipped tables. _v_fn resolves the table when the spec is
    resolved, so building the potential before pointing it at a file would freeze the old one.
    """
    if coord not in _FORCE_FN:
        raise SystemExit(f"coord must be one of {COORDS}, got {coord!r}")
    fn = _FORCE_FN[coord]

    def potential(pos):
        if pos.device.type != "cpu":
            raise NotImplementedError(
                "the tabulated potentials live on cpu -- force_reference loads them once, on the "
                "device it is imported from -- and the IBI sampler runs on cpu. Supporting cuda "
                "means moving the table tensors and the interpolation together, not just this "
                "call, so it is left until something needs it.")
        e, f = fn(spec, pos)
        return e.to(pos.dtype), f.to(pos.dtype)

    return potential


def check(verbose=True):
    """The wrapped potential must BE force_reference's, at both dtypes and both coordinates.

    Not a comparison of two derivations: make_potential calls that function, so this is an
    identity check on the wrapper -- it catches a dtype cast that changes a bit, a closure that
    captured the wrong spec, or a coord that resolved to the wrong force function. The energy
    and force of the underlying expression are verified in force_reference's own report and in
    tests/test_table_potential_injection.py.
    """
    L, B = 9, 3
    g = torch.Generator().manual_seed(20260218)
    # A real, non-degenerate geometry: a jittered straight chain, so no bond or angle is
    # singular and the dihedral is not near collinear. Same construction the injection test uses.
    base = torch.zeros(1, 3 * L, 3, dtype=torch.float64)
    for i in range(L):
        b = torch.tensor([0.59 * i, 0.0, 0.0], dtype=torch.float64)
        base[0, 3 * i] = b
        base[0, 3 * i + 1] = b + torch.tensor([0.12, 0.32, 0.0], dtype=torch.float64)
        base[0, 3 * i + 2] = base[0, 3 * i + 1] + torch.tensor([0.07, 0.26, 0.13],
                                                               dtype=torch.float64)
    pos = base.repeat(B, 1, 1) + 0.05 * torch.randn((B, 3 * L, 3), generator=g,
                                                    dtype=torch.float64)

    specs = {
        "angle": ["table", ("table_jac", 0.05), ("harmonic", 200.0, 0.975),
                  ("harmonic", 28.1, -0.866)],
        "dihedral": ["shipped_harmonic", "table", ("table_jac", None), ("table_jac", 0.05),
                     ("fourier", [0.0, -2.78, -1.33])],
    }
    worst = 0.0
    for coord, spec_list in specs.items():
        for spec in spec_list:
            for dt in (torch.float64, torch.float32):
                p = pos.to(dt)
                e_w, f_w = make_potential(coord, spec)(p)
                # Compare against force_reference CAST THE SAME WAY. At float32 the wrapper's
                # E.to(pos.dtype) is a real rounding, so comparing it against the uncast float64
                # would report that rounding as a wrapper error.
                e_r, f_r = _FORCE_FN[coord](spec, p)
                e_r, f_r = e_r.to(dt), f_r.to(dt)
                de = float((e_w.double() - e_r.double()).abs().max()) / max(
                    float(e_r.abs().max()), 1.0)
                df = float((f_w.double() - f_r.double()).abs().max())
                scale = max(float(f_r.abs().max()), 1.0)
                worst = max(worst, de, df / scale)
                if verbose:
                    print(f"  {coord:8s} {describe(spec):28s} {str(dt):16s} "
                          f"dE/|E|max {de:.3e}  dF/|F|max {df / scale:.3e}  "
                          f"|F|max {float(f_r.abs().max()):.2f}")

    # The scope the sampler actually uses. This is not a hypothetical: ibi_round0.py evaluates
    # every force inside torch.no_grad(), and inside that scope requires_grad_(True) does not
    # re-enable tracking -- so an autograd-based potential without an enable_grad block of its
    # own raises "element 0 of tensors does not require grad". force_reference.dihedral_force
    # and angle_force carry that block; this is the check that would notice if it were removed.
    with torch.no_grad():
        for coord in COORDS:
            _e, _f = make_potential(coord, "table")(pos)
            if not bool(torch.isfinite(_f).all()):
                raise AssertionError(f"{coord}: non-finite force under torch.no_grad()")
    return worst


if __name__ == "__main__":
    print("wrapped potential vs force_reference, same spec (identity check on the wrapper):")
    worst = check()
    print(f"\nworst relative discrepancy: {worst:.3e}")
    print("PASS" if worst < 1e-12 else "FAIL -- the wrapper is not the identity")
    sys.exit(0 if worst < 1e-12 else 1)
