"""Find energy terms that are computed and then never read, anywhere under src/.

cg_energy_3bead computed e_intra and left it out of its own return expression, so that path reported
an energy with no P-C4', no C4'-N9/N1 and no backbone-link term in it. Nothing caught it for months
because the only caller differences two runs of the same function, and a term missing from both
cancels out of the difference. This is the static check for that shape.

The rule is one line: a candidate is USED if it is read anywhere other than the right-hand side of
an assignment to itself. That is deliberately looser than "reaches a return", because a value that
only feeds a diagnostic string or a numpy slot array is still doing work.

It still catches the original bug. Before the fix, e_intra's only reads were inside
"e_intra = e_intra + (...)" -- its own assignment -- so every read was excluded and it was flagged.
Two earlier versions of this script failed on that: one went the wrong way round the data-flow
closure and flagged every term cg_energy_3bead collects into "energy" (including e_intra, which was
right but for the wrong reason and buried), the other required a read to reach a Return or a
total_* accumulation and therefore flagged e_slot_np (written into energies[slots] = ...), e_cg_min
(read by a ratio) and e_r (read by .backward()).

Candidate: a local name matching ^e_ or exactly "energy" that is assigned in the function. Names
starting with an underscore are skipped, because "_e, f = ..." is the deliberate discard convention
and flagging it would bury the real hits.

Deliberately NOT a lint pass: it reports, it does not fail. Some of these are intermediates that are
legitimately dropped, and the point is to make a human say which.

Run: python scripts/scan_unreturned_energies.py [root]
"""
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# NOT `Path(sys.argv[1]) if len(sys.argv) > 1 else ...` at module level. Under pytest sys.argv[1] is
# the test file, so importing this module set the root to a file, rglob walked nothing, the scan
# found zero candidates and tests/test_no_unreturned_energies.py passed its first two assertions on
# an empty set. The argv is read in main() only.
DEFAULT_ROOT = REPO / "src"

CAND = re.compile(r"^(e_|energy$)")


def scan_function(node):
    """(assigned name -> first lineno, name -> whether it is read outside its own assignment)."""
    assigned = {}
    own_rhs = {}
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign):
            tgts = [t.id for t in sub.targets if isinstance(t, ast.Name)]
            for t in tgts:
                if CAND.match(t) and not t.startswith("_"):
                    assigned.setdefault(t, sub.lineno)
            for name_node in ast.walk(sub.value):
                if isinstance(name_node, ast.Name):
                    for t in tgts:
                        own_rhs.setdefault(t, set()).add(id(name_node))

    read_free = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            if id(sub) not in own_rhs.get(sub.id, ()):
                read_free.add(sub.id)
    return assigned, read_free


def report(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assigned, read_free = scan_function(node)
        dead = {k: v for k, v in assigned.items() if k not in read_free}
        if dead:
            hits.append((node.name, dead))
    return hits


def scan_tree(root=None):
    """{(file name, function name, variable): line number} over every parsable .py under root."""
    root = DEFAULT_ROOT if root is None else Path(root)
    found = {}
    unparsable = []
    for p in sorted(p for p in root.rglob("*.py") if "__pycache__" not in str(p)):
        try:
            hits = report(p)
        except SyntaxError as exc:
            unparsable.append((p, exc))
            continue
        for fname, dead in hits:
            for var, vline in dead.items():
                found[(p.name, fname, var)] = vline
    return found, unparsable


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    found, unparsable = scan_tree(root)
    for p, exc in unparsable:
        print(f"{p.relative_to(REPO)}: cannot parse ({exc})")
    for (fname, fn, var), vline in sorted(found.items(), key=lambda kv: (kv[0][0], kv[1])):
        print(f"{fname}:{vline}  {fn}(): {var} is assigned and never read")
    print()
    print(f"{len(found)} candidate(s) under {root.relative_to(REPO)}")
    print()
    print("Each one needs a human answer: an intermediate that is legitimately dropped, or a term")
    print("the path computes and then forgets to add? cg_energy_3bead's e_intra was the second.")
    print("tests/test_no_unreturned_energies.py pins the current answer.")


if __name__ == "__main__":
    main()
