"""No energy term is computed and then left unread anywhere in src/.

This is a regression guard for one shape of silent failure. cg_energy_3bead computed e_intra -- the
P-C4', C4'-N9/N1 and backbone-link terms -- and left it out of its own return expression, so that
path reported an energy with none of them in it. Nothing caught it: the only caller differences two
runs of the same function, and a term missing from both cancels out of the difference.

The scan itself is scripts/scan_unreturned_energies.py; it reports rather than fails, because some
of what it finds is legitimately dropped. Both surviving candidates are the same deliberate,
documented omission, in physical_relaxation.py:

    # Stacking.  NOTE: e_stack is built and then NOT returned -- it is absent from
    # the sum below, which is why K_STACK = 0.0 is dead twice over.  Keeping the
    # omission visible matters: switching K_STACK back on would otherwise look like
    # it re-enables a restraint while changing no energy at all.

Anything else appearing in this set means a path computes an energy and forgets to add it, which is
the failure this file exists to prevent.
"""
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

scan = pytest.importorskip("scan_unreturned_energies")

# (file name, function name, variable), with the reason each is allowed
KNOWN = {
    ("physical_relaxation.py", "_relax_torch_gpu", "e_stack"):
        "deliberate: K_STACK = 0.0 and the omission is documented at the assignment",
    ("physical_relaxation.py", "_energy", "e_stack"):
        "same omission, reached through the nested _energy closure",
}


def test_only_the_documented_dropped_energies_remain():
    found, unparsable = scan.scan_tree()
    # the scan is only meaningful over the whole tree; an empty result must not pass
    assert found, (
        "the scanner found no candidates at all, which means it scanned nothing -- check that "
        "scan.DEFAULT_ROOT resolves to src/ and not to whatever sys.argv happened to hold")
    assert not unparsable, (
        f"the scanner could not parse {[str(p) for p, _ in unparsable]}, so it is not looking at "
        f"the whole tree and this test is weaker than it reads")
    unknown = {k: v for k, v in found.items() if k not in KNOWN}
    assert not unknown, (
        f"these names are assigned and never read anywhere in their function: {unknown}. A term "
        f"the field computes and never adds is invisible to every test that differences two runs "
        f"of the same function, which is how cg_energy_3bead's e_intra survived. Either wire it up "
        f"or add it to KNOWN in this file with the reason.")
    missing = [k for k in KNOWN if k not in found]
    assert not missing, (
        f"{missing} is in KNOWN but the scanner no longer finds it. If the term was wired up, "
        f"remove it from KNOWN; if the code moved, update the name.")


def test_the_scanner_reports_a_deliberately_broken_function():
    """The scanner has to be able to fail, or the test above is vacuous.

    Two earlier versions of it could not: one walked the data-flow closure the wrong way and
    flagged every term a function collects into a returned total, the other required a read to
    reach a Return and so flagged values written into arrays and diagnostics. Both were caught by
    running them on cg_energy_3bead, and this pins the behaviour so a third version cannot quietly
    stop finding anything.
    """
    import ast
    import textwrap
    src = textwrap.dedent("""
        def field():
            e_kept = 1.0
            e_dropped = 2.0
            total = e_kept
            return total
    """)
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef))
    assigned, read_free = scan.scan_function(fn)
    dead = {k for k in assigned if k not in read_free}
    assert dead == {"e_dropped"}, (
        f"the scanner reports {dead} for a function that drops e_dropped and returns e_kept; it "
        f"must find exactly the dropped one")
