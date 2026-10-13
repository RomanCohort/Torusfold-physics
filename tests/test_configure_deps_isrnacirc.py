"""The isRNAcirc dependency check called a working install BLOCKING.

WHAT WENT WRONG, measured on this machine before the fix.

`tools/configure_deps.py` ended every scan with

    tools found   : 4 / 8
    BLOCKING      : isrnacirc

on a machine where the CG -> all-atom conversion demonstrably works: a 200 nt
input converted to 2400 atoms through `isrnacirc_wrapper.cg_to_allatom()`, using
the environment that same tool had written. Two separate bugs produced the line.

BUG 1 -- the search resolved to `bin/`, not to the tree.
    isRNAcirc is the one tool in the list with `probe: None`, because no single
    filename identifies its root. `_looks_like_tool` therefore falls through to
    "any non-empty directory", and `bin/` -- containing one exe and four DLLs --
    passes. Every piece was then looked for *from inside bin/*, where the sibling
    `Data/` and the five `AA_*.dat` files are not visible. The report said
    `Data/ : not found` about a directory that exists one level up.

BUG 2 -- "installed" required a binary that nothing calls.
    The refinement binary (`IsRNAcirc.exe` / `IsRNAcirc.out`) is not in the
    distributed `bin/` at all, and `isrnacirc_refine()` -- the only thing that
    would run it -- is defined in `isrnacirc_wrapper.py` and never called; level-2
    refinement goes through the Torch field or OpenMM. Requiring it made a
    complete conversion install report as missing.

The tests below cover the rule and the path arithmetic, which need no scan. A
full `discover()` is deliberately not exercised: it probes four interpreters with
a 180 s timeout each, so a test that called it would be minutes of subprocess on
a CI runner and would assert something about the machine rather than about this
code.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import configure_deps as cd  # noqa: E402


def _fake_tree(tmp_path, *, exe=True, coefficients=True, data=True,
               refinement=False):
    """Build the shape of a real isRNAcirc package, without any binaries.

    The layout matches the distributed package as it actually is: `bin/` holds
    `CG_to_allatom.exe` and its DLLs, and the coefficient templates are in
    `Data/data/IsRNA2/` -- not in `Data/`, which is what the binary's own
    documented default assumes.
    """
    tree = tmp_path / "isRNAcirc"
    if exe:
        b = tree / "bin"
        b.mkdir(parents=True)
        (b / "CG_to_allatom.exe").write_bytes(b"\x00")   # never executed here
        (b / "libgomp-1.dll").write_bytes(b"\x00")
    if refinement:
        (tree / "bin").mkdir(parents=True, exist_ok=True)
        (tree / "bin" / "IsRNAcirc.exe").write_bytes(b"\x00")
    if data:
        (tree / "Data" / "data" / "IsRNA2").mkdir(parents=True)
    if coefficients:
        coeff = tree / "Data" / "data" / "IsRNA2"
        coeff.mkdir(parents=True, exist_ok=True)
        for name in cd.CG_COEFF_FILES:
            (coeff / name).write_text("0.0\n")
    return tree


# ── bug 1: the path arithmetic ───────────────────────────────────────────────

def test_a_bin_directory_is_widened_to_the_tree_that_holds_data(tmp_path):
    tree = _fake_tree(tmp_path)
    assert cd._isrnacirc_tree_root(str(tree / "bin")) == str(tree)


def test_widening_is_a_fixed_point_on_the_tree_root(tmp_path):
    """Widening twice must not keep climbing: `Data/data/IsRNA2` is a component
    name too, and climbing from there would leave the tree entirely."""
    tree = _fake_tree(tmp_path)
    once = cd._isrnacirc_tree_root(str(tree))
    assert once == str(tree)
    assert cd._isrnacirc_tree_root(once) == str(tree)


def test_an_unrelated_directory_is_not_promoted_to_a_tree_root(tmp_path):
    """The widening key is the directory NAME, so it cannot promote a directory
    that merely happens to sit above something isRNAcirc-shaped."""
    (tmp_path / "projects" / "isRNAcirc" / "bin").mkdir(parents=True)
    (tmp_path / "projects" / "isRNAcirc" / "bin" / "CG_to_allatom.exe").write_bytes(b"\x00")
    assert cd._isrnacirc_tree_root(str(tmp_path / "projects")) == str(tmp_path / "projects")


def test_the_component_directory_alone_cannot_see_the_sibling_data(tmp_path):
    """This is the bug, pinned as a property of the layout function.

    Not an assertion that the old behaviour was fine -- it is the reason
    `discover()` has to widen before calling this. If a later edit removes the
    widening, `layout["data"]` goes back to None and the coefficient search below
    returns None, which is exactly the state that produced BLOCKING.
    """
    tree = _fake_tree(tmp_path)
    assert cd._isrnacirc_layout(str(tree / "bin"))["data"] is None
    assert cd.find_cg_coeff_dir(str(tree / "bin")) is None


def test_the_widened_tree_resolves_all_three_pieces(tmp_path):
    tree = _fake_tree(tmp_path)
    wide = cd._isrnacirc_tree_root(str(tree / "bin"))
    assert cd._isrnacirc_layout(wide) == {
        "bin_dir": str(tree / "bin"),
        "cg_to_aa": str(tree / "bin" / "CG_to_allatom.exe"),
        "isrnacirc_exe": None,
        "data": str(tree / "Data"),
    }


def test_the_coefficient_files_are_found_under_data_data_isrna2(tmp_path):
    """The check that decides whether a conversion can run at all."""
    tree = _fake_tree(tmp_path)
    assert cd.find_cg_coeff_dir(str(tree)) == str(tree / "Data" / "data" / "IsRNA2")


def test_four_of_five_coefficient_files_is_not_enough(tmp_path):
    """CG_to_allatom.exe opens AA_baseA.dat first and dies with "Wrong coeffDIR"
    if any of the five is missing, so a partial set must not count."""
    tree = _fake_tree(tmp_path)
    (tree / "Data" / "data" / "IsRNA2" / cd.CG_COEFF_FILES[-1]).unlink()
    assert cd.find_cg_coeff_dir(str(tree)) is None


def test_coefficients_directly_in_the_tree_root_are_accepted(tmp_path):
    """A rearranged copy is still a usable copy."""
    tree = _fake_tree(tmp_path, data=False)
    for name in cd.CG_COEFF_FILES:
        (tree / name).write_text("0.0\n")
    assert cd.find_cg_coeff_dir(str(tree)) == str(tree)


# ── bug 2: the rule that decides BLOCKING ────────────────────────────────────

def test_a_working_conversion_with_no_refinement_binary_is_installed():
    """The regression. This combination is a real, complete install on this
    machine, and it used to print `BLOCKING : isrnacirc`."""
    ok, reason = cd._isrnacirc_verdict(
        cg_ok=True, coeff=r"C:\t\Data\data\IsRNA2", refinement_ok=False,
        has_data=True, cg_why="runs (exit 1)",
        exe_why="no IsRNAcirc.exe/.out in this tree",
        bin_dir=r"C:\t\bin")
    assert ok is True
    assert "no IsRNAcirc.exe" in reason
    assert "no code path invokes" in reason


def test_a_conversion_that_does_not_load_is_not_installed():
    ok, reason = cd._isrnacirc_verdict(
        cg_ok=False, coeff=r"C:\t\Data\data\IsRNA2", refinement_ok=False,
        has_data=True, cg_why="STATUS_DLL_NOT_FOUND — a required DLL is missing",
        exe_why="unusable", bin_dir=r"C:\t\bin")
    assert ok is False
    assert "DLL_NOT_FOUND" in reason


def test_launching_without_coefficients_is_not_installed():
    """The case that used to be reported as present-and-usable while every
    conversion failed."""
    ok, reason = cd._isrnacirc_verdict(
        cg_ok=True, coeff=None, refinement_ok=True, has_data=True,
        cg_why="runs (exit 1)", exe_why="runs (exit 0)", bin_dir=r"C:\t\bin")
    assert ok is False
    assert "Wrong coeffDIR" in reason
    assert "Data/data/IsRNA2/" in reason


def test_missing_data_is_a_note_when_the_conversion_still_works():
    ok, reason = cd._isrnacirc_verdict(
        cg_ok=True, coeff=r"C:\t\IsRNA2", refinement_ok=True, has_data=False,
        cg_why="runs (exit 1)", exe_why="runs (exit 0)", bin_dir=r"C:\t\bin")
    assert ok is True
    assert "Data/ is absent" in reason


def test_a_full_install_says_so():
    ok, reason = cd._isrnacirc_verdict(
        cg_ok=True, coeff=r"C:\t\Data\data\IsRNA2", refinement_ok=True,
        has_data=True, cg_why="runs (exit 1)", exe_why="runs (exit 0)",
        bin_dir=r"C:\t\bin")
    assert ok is True
    assert reason == "CG->all-atom and refinement both usable"


@pytest.mark.parametrize("missing", ["exe", "data", "coefficients"])
def test_a_tree_missing_one_piece_still_resolves_the_pieces_it_has(tmp_path, missing):
    """So the report can say which sub-step fails instead of collapsing to
    MISSING, which is the whole reason the layout is resolved piece by piece."""
    tree = _fake_tree(
        tmp_path,
        exe=missing != "exe",
        data=missing != "data",
        coefficients=missing != "coefficients",
    )
    layout = cd._isrnacirc_layout(str(tree))
    if missing == "exe":
        assert layout["cg_to_aa"] is None
    else:
        assert layout["cg_to_aa"] is not None
