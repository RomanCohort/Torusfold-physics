"""Is the stalled minimiser a property of the field, or of the descent scheme?

README's acceptance table has one row that currently fails:

    | minimiser | 4427 iterations, 6 restarts, max abs F 690.19 | this row currently fails |

check_field_after_fix.py descends with a halving step: accept x + s*f and grow s by 1.2, else
halve it, and restart below 1e-12. Six restarts say THAT METHOD stalls on this landscape; it does
not say a 690 kJ/mol/nm force is unbalanced. The residual is still above the P-C4' force floor of
236.3 kJ/mol/nm, so the run's starting point is not a true minimum and its -171.2 drift is the
relaxation of that start.

The experiment is one-variable: SAME start, SAME potential, SAME analytic force, only the
optimiser differs.

L-BFGS-B comes from scipy, not hand-rolled. An earlier version of this script rolled its own
two-loop recursion with Armijo backtracking; it converged to max|F| = 5000.00, which is exactly
force_cap, so it walked into the cap and its number says nothing about L-BFGS. A hand-written line
search is not evidence about a method that has a maintained implementation.

The cap is why the last column exists. force_cap clips the force vector, and a clipped gradient is
not the gradient of the energy -- so an optimiser given a capped force is not minimising this
potential. Both optimisers here run UNCAPPED, matching check_field_after_fix.py; the last column
reports how many beads of the final structure the shipped cap would clip anyway.

Run: python scripts/check_minimizer_method.py [max_iter ...]   (default: 1500 6000)
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402
import torusfold.scheme2.torch_cgsim as C   # noqa: E402

MAXITS = [int(a) for a in sys.argv[1:]] or [1500, 6000]
FLOOR = 236.3                          # sqrt(K_INTRA_PC * kBT), the stated force floor
CAP = 5000.0                           # = inspect.signature(C.cg_energy_forces)['force_cap'].default

pool = [s for s in B.load_structures(limit=400) if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
s0 = pool[0]
L = len(s0["pos"])
NB = 3 * L
ij = torch.tensor(s0["pairs"], dtype=torch.long).reshape(-1, 2)
pw = torch.ones(len(ij), dtype=torch.float32)
XS = torch.tensor(s0["pos"].reshape(1, NB, 3), dtype=torch.float64)


def field(p, cap=None):
    """cap=None by default, because check_field_after_fix.py minimises with cap=None.

    That is not a detail. Its `field()` signature is `def field(p, cap=None)`, so the shipped
    minimisation descends on the UNCAPPED potential. An earlier version of this script passed
    force_cap=5000 here and reported 3772 iterations / max|F| 311.52 for the reference row, where
    the acceptance table says 4427 / 690.19 -- the cap was the whole difference. Reproducing a
    row means using the field that row was measured on.
    """
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(p)
    return C.cg_energy_forces(p, ij, pw, cell_list=cl, force_cap=cap)


def ef(p, cap=None):
    with torch.no_grad():
        e, f = field(p, cap)
    return float(e.reshape(-1)[0]), f


def report(x):
    """max|F| on the uncapped potential, and how many beads the SHIPPED cap would clip."""
    _e, f_true = ef(x, None)
    mag = torch.linalg.norm(f_true.reshape(-1, 3), dim=-1)
    return float(mag.max()), int((mag > CAP).sum())


def steepest(xs, maxit):
    """check_field_after_fix.py's scheme, verbatim, as the reference row of the table."""
    x = xs.clone()
    e_cur, _ = ef(x)
    s = 1e-5
    resets, total = 0, 0
    for k in range(maxit):
        _e, f = ef(x)
        tr = x + s * f
        e_tr, _ = ef(tr)
        total = k + 1
        if e_tr < e_cur:
            x, e_cur, s = tr, e_tr, min(s * 1.2, 1e-2)
        else:
            s *= 0.5
            if s < 1e-12:
                if resets >= 6:
                    break
                resets += 1
                s = 1e-4
    return x, e_cur, total, resets


def scipy_lbfgs(xs, maxit, ftol=1e-12, gtol=1e-10):
    """L-BFGS-B on the analytic force. jac=True: fun returns (E, -F) together, exactly as the
    production entry point produces them, so the gradient is never recomputed or differenced.

    The two termination criteria are run separately because they answer different questions.
    The default (ftol small, so factr = ftol/eps is finite) stops when f stops decreasing --
    which on this landscape happens while |grad|_inf is still 2.4e4, so it reports convergence
    at a point that is not stationary. factr is ftol/eps, so ftol=1e-16 puts it at 0.45 and
    leaves gtol in charge, which is the honest test: it asks whether the gradient is small,
    not whether the descent ran out of change. ftol=0.0 is NOT that -- scipy maps it back to
    its default, and an earlier version of this script reported the f-stop row twice.
    """
    x0 = xs.reshape(-1).numpy().astype(np.float64)

    def fun(x):
        t = torch.tensor(x, dtype=torch.float64).reshape(1, NB, 3)
        e, f = ef(t)
        return e, (-f).reshape(-1).numpy().astype(np.float64)

    res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxit, "maxfun": 40 * maxit,
                            "ftol": ftol, "gtol": gtol, "maxcor": 20})
    x = torch.tensor(res.x, dtype=torch.float64).reshape(1, NB, 3)
    return x, float(res.fun), int(res.nit), res


e0, f0 = ef(XS)
print(f"{s0['name']}  L={L} / {NB} beads / {len(ij)} WC pairs")
print(f"start: E = {e0:.2f} kJ/mol, max|F| = {float(torch.linalg.norm(f0.reshape(-1,3),dim=-1).max()):.2f}")
print(f"minimising on the uncapped potential (as check_field_after_fix.py does); the shipped cap is "
      f"{CAP:.0f}")
print(f"harmonic force floor sqrt(K_INTRA_PC*kBT) = {FLOOR} kJ/mol/nm")
print()
print(f"{'method':>30s} {'max_iter':>8s} {'iters':>7s} {'restart':>7s} {'E':>10s} "
      f"{'max|F|':>9s} {'vs floor':>9s} {'would clip':>10s}")
print("-" * 100)

for maxit in MAXITS:
    x, e, total, resets = steepest(XS, maxit)
    mf, ncap = report(x)
    print(f"{'steepest + halving (as shipped)':>30s} {maxit:8d} {total:7d} {resets:7d} {e:10.2f} "
          f"{mf:9.2f} {mf / FLOOR:8.2f}x {ncap:10d}")

for maxit in MAXITS:
    for label, kw in (("L-BFGS-B (f-stop)", dict(ftol=1e-12, gtol=1e-10)),
                      ("L-BFGS-B (grad-stop)", dict(ftol=1e-16, gtol=1e-9))):
        x, e, nit, res = scipy_lbfgs(XS, maxit, **kw)
        mf, ncap = report(x)
        print(f"{'scipy ' + label:>30s} {maxit:8d} {nit:7d} {'':7s} {e:10.2f} "
              f"{mf:9.2f} {mf / FLOOR:8.2f}x {ncap:10d}")
        print(f"{'':>30s} scipy status {res.status} ({res.message.strip()}), "
              f"|grad|_inf = {np.abs(res.jac).max():.3e}, {res.nfev} evals")

print()
print("A residual at or below the floor means the optimiser is the limit, not the field.")
print("'would clip' counts beads whose force exceeds the SHIPPED cap. Minimisation above runs")
print("uncapped, so that column is a property of the result, not of the run: if it is non-zero,")
print("the pipeline's own field would hand any optimiser a direction that is not -dE/dx there.")
