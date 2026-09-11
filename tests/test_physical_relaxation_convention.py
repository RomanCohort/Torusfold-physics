"""Pin physical_relaxation.py's numbers to its own documented table, and pin its
energy convention (E = K*(x-x0)^2, no 1/2) so neither can drift silently.

Why this exists.  physical_relaxation.py declares force-field numerals of the
same magnitude as the rest of the repo, but applies them WITHOUT the 1/2 that
torch_cgsim.py, openmm_gpu_refiner.py and OpenMM itself all carry, so each of
its numerals means twice the stiffness.  It also used to carry a docstring
listing K_BB=500 / K_ANGLE=600 / K_DIH=800 / K_PAIR=1500 / K_STACK=500 while
the code below used 5000 / 200 / 200 / 500 / 0 -- stale numerals belonging to
two other files.  That is how this module came to be described, in a report,
with constants its code did not implement.  The first test makes that failure
mode impossible: the set of numeric constants the torch path defines must equal
the set named in the module header's table, with the same values.

The second test locks the functional form.  If a 1/2 ever appears in one of the
five harmonic terms, every constant silently halves in effect and the header's
convention statement becomes false, so the test fails and points at the header.
It matches the five expressions as literally as it can, so a pure reformat also
fails -- update the expected strings if that is all you did.

Run: python -m pytest tests/test_physical_relaxation_convention.py
"""
import ast
import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torusfold.scheme2.physical_relaxation as P

_SRC = Path(P.__file__).read_text(encoding="utf-8")
_TREE = ast.parse(_SRC)
_RELAX = next(n for n in _TREE.body
              if isinstance(n, ast.FunctionDef) and n.name == "_relax_torch_gpu")

# The header calls the convention SINGLE-SIDED; that word is the contract.
_CONVENTION_WORD = "SINGLE-SIDED"

# The five harmonic terms and the return, as the code spells them.  Whitespace is
# normalised before the comparison, so only tokens matter.
_EXPECTED_FORMS = [
    "e_bb = (K_BB * (dist_bb - BOND_R0) ** 2).sum()",
    "e_bsj = K_BSJ * (d_bsj - BOND_R0) ** 2",
    "e_angle = (K_ANGLE * (angles - ANGLE_0) ** 2).sum()",
    "e_dih = (K_DIH * (dihedral - DIH_0) ** 2).sum()",
    "e_pair = (K_PAIR * pr_w * (dist_pr - PAIR_R0) ** 2).sum()",
    "return e_bb + e_bsj + e_angle + e_dih + e_pair",
]

_ROW = re.compile(
    r"^\s*\|\s*([A-Za-z_][A-Za-z0-9_]*)\s*\|\s*"
    r"([-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)\s*\|",
    re.MULTILINE,
)


def _fold(node):
    """Fold the numeric forms the constant block uses; None if not a number."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Attribute) and node.attr == "pi":
        return math.pi
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _fold(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.BinOp):
        left, right = _fold(node.left), _fold(node.right)
        if left is None or right is None:
            return None
        op = type(node.op)
        if op is ast.Mult:
            return left * right
        if op is ast.Div:
            return left / right
        if op is ast.Add:
            return left + right
        if op is ast.Sub:
            return left - right
    return None


def _code_constants():
    """name -> value for every numeric assignment in _relax_torch_gpu."""
    found = {}
    for node in ast.walk(_RELAX):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            value = _fold(node.value)
            if value is not None:
                found[node.targets[0].id] = value
    return found


def _table_constants():
    """name -> value for every header table row whose value cell is a number."""
    return {name: float(value) for name, value in _ROW.findall(P.__doc__ or "")}


def test_header_table_is_exactly_the_torch_paths_constants():
    code = _code_constants()
    table = _table_constants()
    assert code, "no numeric constants found in _relax_torch_gpu; did it move?"
    assert set(table) == set(code), (
        f"module header table and _relax_torch_gpu disagree.\n"
        f"  in code only:  {sorted(set(code) - set(table))}\n"
        f"  in table only: {sorted(set(table) - set(code))}\n"
        "The table is the file's own statement of its numbers and units; a stale "
        "table is how this module's constants were once reported with values its "
        "code did not implement.")
    for name, value in sorted(code.items()):
        assert table[name] == pytest.approx(value, rel=1e-6), (
            f"{name}: code has {value}, header table says {table[name]}")


def test_torch_harmonics_are_single_sided():
    body = re.sub(r"\s+", " ", ast.get_source_segment(_SRC, _RELAX) or "")
    assert _CONVENTION_WORD in (P.__doc__ or ""), (
        "the module header no longer calls this file's convention "
        f"{_CONVENTION_WORD!r}; that word is what the next assertion relies on")
    for form in _EXPECTED_FORMS:
        assert re.sub(r"\s+", " ", form) in body, (
            f"physical_relaxation._relax_torch_gpu no longer contains {form!r}.\n"
            "If you added the 1/2 (or removed it), every constant in the header "
            "table changes meaning by a factor of two -- update the header's "
            "convention section and say why. If you only reformatted, update "
            "_EXPECTED_FORMS in this test.")
