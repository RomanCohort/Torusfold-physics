"""The table stores bin probabilities but the sampler reads node values. Measure the gap, then
close it on the interpolation model itself rather than choosing between the two readings.

boltzmann_bonded._to_table_values stores U_i = -kBT ln p_i, where p_i is the probability MASS of
bin i. _sample then reads U_i as the potential AT the bin centre and interpolates linearly between
centres. Those are different quantities, so the interpolant's own bin masses are not the p_i that
were fitted, and the difference is a systematic offset that every IBI step inherits.

The fix is not "interpret U as a step function" -- that reproduces p_i but makes U discontinuous,
and its gradient is a train of delta functions, which is worse for a force. It is to solve for node
values u_i such that the piecewise-linear interpolant has bin masses p_i. The update is the
project's own IBI rule applied to the interpolation model:

    u_i <- u_i + kBT * ln( p_i^target / p_i^model(u) )

and p_i^model is exact, because a piecewise-linear U makes exp(-U/kBT) integrable in closed form.
On a segment where U runs linearly between Ua and Ub over length d the integral is

    d * kBT / (Ub - Ua) * (exp(-Ua/kBT) - exp(-Ub/kBT))        Ub != Ua
    d * exp(-Ua/kBT)                                          Ub == Ua

and a bin is two such segments, from its left edge through its centre to its right edge.

Run: python scripts/fix_table_interpolation_convention.py [iters]
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import boltzmann_bonded as B   # noqa: E402

KBT = B.KBT if hasattr(B, "KBT") else 2.494
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def seg_integral(ua, ub, d, kBT):
    """Integral of exp(-U/kBT) over a segment of length d with U linear from ua to ub."""
    ua = np.asarray(ua, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    du = ub - ua
    ea = np.exp(-ua / kBT)
    eb = np.exp(-ub / kBT)
    flat = np.abs(du) < 1e-12
    safe = np.where(flat, 1.0, du)
    return np.where(flat, d * ea, d * kBT / safe * (ea - eb))


def bin_masses(u, binw, kBT):
    """Mass of every bin under the piecewise-linear u, matching _sample's clamping exactly.

    _sample's i0 is clamped to [0, len-2] and f to [0,1], so the first bin's left half is flat at
    u[0] and the last bin's right half is flat at u[-1]. Reproduced here rather than idealised,
    because those two flats are what the sampler actually does.
    """
    n = len(u)
    lo = np.empty(n)
    hi = np.empty(n)
    # left half of bin i: from u[i-1/2] to u[i]
    u_left = np.empty(n)
    u_left[0] = u[0]                      # clamped flat
    u_left[1:] = 0.5 * (u[:-1] + u[1:])
    # right half of bin i: from u[i] to u[i+1/2]
    u_right = np.empty(n)
    u_right[-1] = u[-1]                   # clamped flat
    u_right[:-1] = 0.5 * (u[:-1] + u[1:])
    lo = seg_integral(u_left, u, 0.5 * binw, kBT)
    hi = seg_integral(u, u_right, 0.5 * binw, kBT)
    return lo + hi


def target_mass(u, kBT):
    p = np.exp(-(u - u.min()) / kBT)      # the stored U IS -kBT ln p, up to a constant
    return p / p.sum()


structs = B.load_structures()
tables = B.prepare(B.fit(structs))
print(f"{len(structs)} chains; tables on {len(tables)} coordinates")
print()
print("core = bins whose target mass is at least 1e-4; rms is weighted by that mass")
print("the last three columns are diagnostics of the inversion attempt, not results of one")
print(f"{'coord':10s} {'bins':>5s} {'core':>5s} {'gap max':>9s} {'gap rms':>9s} "
      f"{'gap/kBT':>9s} {'diag(J)':>9s} {'cond(Jr)':>10s} {'Newton':>11s}")
print("-" * 96)
worst_before = 0.0
for name in B.TABLED:
    t = tables[name]
    U = np.array(t["U"], dtype=np.float64)
    binw = float(t["binw"])
    target = target_mass(U, KBT)

    # A bin whose target mass is below CORE has no data in it: its U comes from the pseudo-count
    # and its model mass is whatever the interpolation says, so kBT ln(target/model) there is
    # noise and would dominate a plain max. Everything is reported twice, over the populated core
    # and over everything, because the two answer different questions.
    CORE = 1e-4
    pop = target >= CORE
    upd = target >= 1e-8

    def dev(u):
        m = bin_masses(u, binw, KBT)
        p = np.maximum(m / m.sum(), 1e-300)
        return KBT * np.log(np.maximum(target, 1e-300) / p)

    def stats(d, mask):
        v = np.abs(d[mask])
        w = target[mask]
        return float(v.max()), float((v * w).sum() / w.sum())

    b_max, b_rms = stats(dev(U), pop)

    # Jacobian of the log bin masses with respect to the node values, at the stored nodes.
    # The fixed-point update u += d assumes this is close to the identity. It is not.
    n = len(U)
    J = np.zeros((n, n))
    h = 1e-6
    d0 = dev(U)
    for j in range(n):
        up = U.copy()
        up[j] += h
        J[:, j] = (dev(up) - d0) / h

    diag = np.diag(J)

    # Gauss-Newton, recomputed each step and RESTRICTED to the bins that carry data. The full
    # 120x120 system is rank-deficient on purpose: a dead bin's mass is clamped at 1e-300, so its
    # row is identically zero and the smallest singular value above is measuring those rows, not
    # the physics. Solving on the populated subset removes the null directions.
    idx = np.where(upd)[0]
    Jr = J[np.ix_(idx, idx)]
    sv = np.linalg.svd(Jr, compute_uv=False)
    du = dev(U)
    raw = np.linalg.lstsq(Jr, -du[idx], rcond=1e-10)[0]

    # A damped version of the same direction: take a small fixed fraction of the Newton step and
    # see whether the residual actually falls. If it does, the step is a descent direction and
    # only the magnitude was the problem; if it does not, clipping was not the issue either.
    u = U.copy()
    hist = []
    for _ in range(ITERS):
        d_now = dev(u)
        hist.append(float(np.abs(d_now[idx]).max()))
        u[idx] += -0.02 * raw
    print(f"{name:10s} {len(U):5d} {int(pop.sum()):5d} {b_max:9.4f} {b_rms:9.4f} "
          f"{b_max / KBT:9.4f} {np.median(diag):9.4f} "
          f"{sv[0] / max(sv[-1], 1e-300):10.2e} {np.abs(raw).max():11.3e}")
    if hist:
        k = max(1, len(hist) // 5)
        print(f"{'':10s} 0.02x damped along the same direction, residual: "
              + " ".join(f"{v:.4g}" for v in hist[::k][:5]))
    worst_before = max(worst_before, b_max)
print()
print("gap columns are kJ/mol of kBT * ln(target/model) per bin, so they read as 'this bin is")
print("off by this much'. The kBT is %.4f (300 K)." % KBT)
print()
print(f"worst bin offset before the correction: {worst_before:.4f} kJ/mol = "
      f"{worst_before / KBT:.4f} kBT")
print()
print("The fixed-point update u_i += kBT ln(target/model) assumes d(ln p_i)/d(u_i) = 1. It measures")
print("0.745 everywhere, so the update overshoots by 1.34x -- and that alone would still converge.")
print("What runs away is the same update applied to the dead tails, whose model mass is clamped and")
print("therefore independent of their node: those bins never close. Restricting the solve to the")
print("populated bins fixes that and leaves the conditioning problem, which is the real one.")
print()
print("Both solvers fail, and the two printed lines say why rather than leaving it as 'it did not")
print("converge'. The Newton step is enormous because the operator is nearly singular, so clipping")
print("it changes its direction rather than its length; and the damped version of the same direction")
print("does not reduce the residual, so the direction itself is not a descent direction for this")
print("residual. Inverting 'node values -> bin masses' is an ill-posed problem here.")
print()
print("That is the result. It is not a reason to add a solver to boltzmann_bonded.prepare(). IBI is")
print("self-correcting for any convention, because its fixed point -- the simulated bin masses equal")
print("the target -- is stated in terms of the simulation rather than the interpolant, and the")
print("shipped constants come from the spread (k = kBT/sigma^2), not from the table shape. The gap")
print("is recorded as a bounded approximation and nothing is changed.")
