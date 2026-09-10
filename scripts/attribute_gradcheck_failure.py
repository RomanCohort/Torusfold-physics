"""Did the bonded-target change break the force/energy consistency gradcheck?

Running pytest after the change shows tests/test_force_gradcheck.py failing with
max_rel 4.23 and 11.42 against a 0.2 tolerance. That has to be attributed before anything
else is done, so this runs the test's own helpers against the revision before the change and
the revision after, on the identical system the test builds.

A script in tests/ so the helpers are the test's own, not a reimplementation.

Run: python scripts/attribute_gradcheck_failure.py [git_ref]
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
REL = "src/torusfold/scheme2/torch_cgsim.py"
REF = sys.argv[1] if len(sys.argv) > 1 else "HEAD"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))


def load_from_git(ref, name):
    src = subprocess.run(["git", "show", f"{ref}:{REL}"], cwd=REPO, capture_output=True,
                         text=True, encoding="utf-8", check=True).stdout
    tmp = Path(tempfile.gettempdir()) / f"{name}.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import test_force_gradcheck as G          # noqa: E402  the test's own helpers
import torusfold.scheme2.torch_cgsim as NEW  # noqa: E402

OLD = load_from_git(REF, "gradcheck_old_torch_cgsim")

FNS = ("cg_forces_explicit_batched", "cg_energy_forces")
print(f"git ref {REF}: STACK_R0 {OLD.STACK_R0}, cos(DIH_PPPP) {torch.cos(torch.tensor(OLD.DIH_PPPP)).item():+.3f}")
print(f"working tree  : STACK_R0 {NEW.STACK_R0}, cos(DIH_PPPP) {torch.cos(torch.tensor(NEW.DIH_PPPP)).item():+.3f}")
print()
print(f"{'force function':28s} {'before':>22s} {'after':>22s}")
print(f"{'':28s} {'(max_rel / max_abs)':>22s} {'(max_rel / max_abs)':>22s}")
print("-" * 74)
for fn_name in FNS:
    row = []
    for mod in (OLD, NEW):
        pos, pairs, pair_w = G._make_system()
        row.append(G._fd_max_errors(getattr(mod, fn_name), pos, pairs, pair_w))
    print(f"{fn_name:28s} {row[0][1]:10.4f} /{row[0][0]:10.2f} "
          f"{row[1][1]:10.4f} /{row[1][0]:10.2f}")
print()
print("tolerance in the test is max_rel < 0.2")
