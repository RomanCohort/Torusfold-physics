"""Is the live path's gradcheck failure the acknowledged dihedral approximation?

tests/test_force_gradcheck.py fails on cg_energy_forces with max_rel 11.42 at HEAD. The
suspect is _dihedral_f, whose own comment says "The full formula is too complex; use an
analytic approximation of the numerical gradient" and which distributes the force over
b0 and b2 with arbitrary 0.25 factors.

This asks the question directly: zero each term's constant in turn and see which one
accounts for the discrepancy. The difference from the previous diagnostic is that this is
about cg_energy_forces, the path that is actually called, not the abandoned batched one.

Run: python scripts/attribute_gradcheck_to_term.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import torusfold.scheme2.torch_cgsim as C
import test_force_gradcheck as G


def max_rel(**zero):
    """max_rel of cg_energy_forces with the named constants zeroed."""
    saved = {k: getattr(C, k) for k in zero}
    try:
        for k, v in zero.items():
            setattr(C, k, 0.0 if v else getattr(C, k))
        pos, pairs, pair_w = G._make_system()
        a, r = G._fd_max_errors(C.cg_energy_forces, pos, pairs, pair_w)
        return r, a
    finally:
        for k, v in saved.items():
            setattr(C, k, v)


base_r, base_a = max_rel()
print(f"cg_energy_forces, all terms on            max_rel {base_r:9.4f}  "
      f"max_abs {base_a:10.2f}")
print()
print("zeroing one term at a time (a large drop means that term was carrying the error):")
print(f"{'term zeroed':18s} {'max_rel':>10s} {'max_abs':>12s} {'drop':>8s}")
print("-" * 52)
for label, name in [("dihedral points", "K_DIH"),
                    ("angle points", "K_ANGLE"),
                    ("bb bonds", "K_BB"),
                    ("intra-bead", "K_INTRA"),
                    ("stacking", "K_STACK"),
                    ("WC pair", "K_PAIR"),
                    ("BSJ closure", "K_BSJ"),
                    ("bpp", "K_BPP"),
                    ("pair guide", "K_PAIR_GUIDE"),
                    ("BSJ contact", "K_BSJ_CONTACT")]:
    # K_PAIR/K_BPP/K_PAIR_GUIDE only act through the pair list, which _make_system provides
    if not hasattr(C, name):
        print(f"{label:18s} {'(no such constant)':>10s}")
        continue
    r, a = max_rel(**{name: 1})
    print(f"{label:18s} {r:10.4f} {a:12.2f} {base_r - r:8.4f}")
print()
print("the test threshold is max_rel < 0.2")
