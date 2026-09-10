"""Prove the frozen-snapshot bug was real, and that the regression test detects it.

tests/test_ff_bonded_targets.py asserts that both force paths follow a change to
STACK_R0. A test that can never fail is worth nothing, so this replays the same check
against the version of torch_cgsim.py from before the fix, which is taken straight out of
git rather than retyped.

The module is self-contained (stdlib + numpy + torch), so the old revision can be imported
by path under a different name without disturbing the installed package.

Run: python scripts/prove_snapshot_bug_was_real.py [git_ref]
     git_ref defaults to HEAD, i.e. the last commit before the working-tree change.
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
REL = "src/torusfold/scheme2/torch_cgsim.py"
REF = sys.argv[1] if len(sys.argv) > 1 else "HEAD"


def load_from_git(ref, name):
    # encoding matters: the source is UTF-8 and this shell's default is GBK on Windows
    src = subprocess.run(["git", "show", f"{ref}:{REL}"], cwd=REPO, capture_output=True,
                         text=True, encoding="utf-8", check=True).stdout
    tmp = Path(tempfile.gettempdir()) / f"{name}.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def chain(L=8):
    pos = np.zeros((L, 3, 3))
    for i in range(L):
        base = np.array([0.59 * i, 0.0, 0.0])
        pos[i, 0] = base
        pos[i, 1] = base + np.array([0.39 * 0.3, 0.39 * 0.95, 0.0])
        pos[i, 2] = pos[i, 1] + np.array([0.335 * 0.2, 0.335 * 0.9, 0.335 * 0.4])
    return torch.tensor(pos.reshape(1, 3 * L, 3), dtype=torch.float64)


def responds(mod, target_now, target_new):
    """Does each path change its energy when the live constant changes?"""
    pos = chain()
    pairs = torch.zeros((0, 2), dtype=torch.long)
    cl = mod.GPUCellList(cell_size=1.5)
    cl.build(pos)

    def energies():
        return (float(mod.cg_energy_forces(pos, pairs, None, cell_list=cl)[0]),
                float(mod.cg_forces_explicit_batched(pos, pairs, None, cell_list=cl)[0]))

    mod.STACK_R0 = target_now
    a = energies()
    mod.STACK_R0 = target_new
    b = energies()
    return a[0] != b[0], a[1] != b[1]


name = "old_torch_cgsim_probe"
old = load_from_git(REF, name)
old_stack0 = old.STACK_R0
print(f"loaded {REF}:{REL}")
print(f"  old STACK_R0 = {old_stack0}")
print(f"  has _R0_STACK = {hasattr(old, '_R0_STACK')}"
      + (f" (frozen at {old._R0_STACK})" if hasattr(old, "_R0_STACK") else ""))
print()

u, b = responds(old, old_stack0, 1.125)
print("old revision, changing STACK_R0:")
print(f"  cg_energy_forces            responds: {u}")
print(f"  cg_forces_explicit_batched  responds: {b}")
print()
if b:
    print("UNEXPECTED: the old batched path did follow STACK_R0, so the snapshot was not")
    print("actually load-bearing and the regression test proves nothing.")
    sys.exit(2)
print("CONFIRMED: on the old revision the batched path silently ignored the change -- it was")
print("reading the import-time copy. The two force paths used different stacking targets.")
print("The regression test fails on exactly this, so it has detection power.")
