"""E3: settle section 3ay -- run the PRE-6e7a44b guide shape on the same window.

3ay measured what correcting `_sigmoid_f`'s sign cost: joint residual 0.2937 -> 0.3263, an
11 percent regression. That was measured on the DEFAULT window (burn = NSTEPS // 5 = 3.2 ps of
a 16 ps run), which section 3ba later showed is a transient. So the 11 percent is a transient
difference, and whether the corrected sign is better or worse on a stationary window has never
been measured. That is the whole of E3.

It patches rather than checking out, which differs from the handoff's recipe on purpose:

    git checkout 424e1d7 -- src/torusfold/scheme2/torch_cgsim.py

That recipe mutates the working tree while other runs are in flight. Those runs already
imported the module and are unaffected, but any process that starts during the swap -- a retry,
a crash restart, another experiment -- silently reads the wrong field, and a failed restore
leaves the tree dirty in a way nothing in the output records. The patch cannot do either.

It is also self-verifying: ibi_round0.py probes the guide shape directly (it cannot see shape
through the constant fingerprint, which is why that probe exists). A patched run must print
`SHORT-RANGE REWARD -- the pre-6e7a44b form`. If it prints `long-range`, the patch did not take
and the run is not E3.

Scope: `cg_energy_forces` reaches the guide through `_sigmoid_f` at two call sites, resolved as
a module global at call time, so replacing the attribute reproduces the old field on the
production path. The other three writing points 3ay lists are on the two explicit paths and
`cg_energy_3bead`, all of which `_alternate_field` guards and this run never enters.

Run: python scripts/e3_old_field.py [n_rep] [n_steps] [idx] [friction] [stride] [burn]
"""
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import torch                                    # noqa: E402
import torusfold.scheme2.torch_cgsim as C       # noqa: E402

_args = sys.argv[1:] or ["8", "100000", "0", "0.1", "25", "20000"]

_softplus = C._stable_softplus


def _old_sigmoid_f(dist, r0, k, width):
    """The pre-6e7a44b form, verbatim from `git show 424e1d7:src/torusfold/scheme2/torch_cgsim.py`.

        x = (r0 - dist) / width;  e = -k * softplus(x)

    Pulls hardest when the pair is already inside r0 and does nothing at long range -- the
    opposite of the term's stated purpose. Curvature at the well is -k/(4w^2).
    """
    x = (r0 - dist) / width
    sig = torch.sigmoid(x)
    e = -k * _softplus(x)
    return e, sig


C._sigmoid_f = _old_sigmoid_f
print("E3: _sigmoid_f patched to the pre-6e7a44b form (no working-tree change).")
print("The guide-shape probe below must print SHORT-RANGE REWARD or this run is not E3.")

sys.argv = ["ibi_round0.py"] + _args
runpy.run_path(str(REPO / "scripts" / "ibi_round0.py"), run_name="__main__")
