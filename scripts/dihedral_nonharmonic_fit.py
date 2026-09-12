"""Fit a non-harmonic (periodic / double-well) form for the P-P-P-P dihedral, evaluate a
tabulated Boltzmann alternative, and measure the forces each one produces on reference
geometry. This is a PROTOTYPE: it reads the reference tables and the production constants but
never writes to src/ (the force field is not modified).

Why. The shipped dihedral E = 0.5*K_DIH*(cos(phi) - cos(12.84 deg))**2 is a single cis well.
The reference distribution P_ref(q), q = cos(phi), inverted from results/boltzmann_tables_clean.npz
(U = -kBT ln P), is bimodal plus a wide tail: cis (q ~ +0.9967) ~63%, trans (q ~ -0.9967) ~6.5%,
middle (-0.8..+0.8) ~30%. A single well cannot produce the trans mass or the tail whatever K is,
so the defect is the form, not the constant. This script fits two candidate forms and checks them
item by item against the reference:

  A. low-order Fourier, reduced to its even part. The production field reads the dihedral only
     through q = cos(phi) (the normal-vector dot product in _dihedral_f), so it cannot see the
     sign of sin(phi): the field-compatible Fourier is the even part V(phi) = sum_n K_n cos(n phi)
     = sum_n K_n T_n(q), a degree-4 polynomial in q for n <= 4. This is the IsRNA2 "quadruple
     Fourier" in the coordinate the field actually has. Fitted by MAXIMUM LIKELIHOOD: minimise
     -sum_i m_i ln pmodel_i over K, where the model is exp(-V(phi)/kBT) under the flat-phi
     (torsion) measure and m_i is the reference bin mass. ML is used rather than least-squares on
     -kBT ln P - 0.5 kBT ln(1-q^2) because that target carries a logarithmic Jacobian divergence
     at q = +-1 that no finite polynomial can follow and that mis-weights the very bins where the
     cis mass lives.
  B. tabulated Boltzmann potential (DBI). Two variants:
     B1  U(q) = -kBT ln P_ref(q)  -- exactly what scripts/boltzmann_bonded.py energy()/mixed_energy()
         does today (measure-naive: it treats the q-table as if q had flat measure).
     B2  U(q) = -kBT ln P_ref(q) - 0.5 kBT ln(1-q^2)  -- the measure-correct DBI for a torsion.

Degradability. Candidate A is re-fit to a "washed" reference (the trans peak, q < -0.8, zeroed and
renormalised) so the same Fourier family produces a single-cis well; this is the one-switch path if
the trans peak turns out to be database contamination and the fix is to wash the reference, not
change the field. Candidate B degrades the same way by rebuilding the table from the washed
reference.

The angle (P-P-P, flat-cos measure, no Jacobian) is run through the tabulated form, plus a cubic
analytic alternative, and checked against its mode / sigma / skew.

Forces. For each candidate the per-bead dihedral force |F| = |dV/dq| * |dq/dx| is computed by torch
autograd on real P(i)-P(i+1)-P(i+2)-P(i+3) windows from the rsRNASP Training_set, exactly the way
_dihedral_f differentiates, and compared to force_cap = 5000 kJ/mol/nm.

Run: python scripts/dihedral_nonharmonic_fit.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402  (KBT, load_structures, coords_of)
import torusfold.scheme2.torch_cgsim as C   # noqa: E402  (K_DIH, DIH_PPPP, K_ANGLE, ANGLE_PPP)

KBT = B.KBT
CIS_LO = 0.8          # q > CIS_LO is the cis peak
TRANS_HI = -0.8       # q < TRANS_HI is the trans peak
FORCE_CAP = 5000.0
NPHI = 2_000_000      # phi grid points for the final model q-distribution
NPHI_FIT = 100_000    # phi grid points for the ML objective (cheap enough for scipy)


# ------------------------------------------------------------------ reference tables
def load_table(name):
    z = np.load(REPO / "results" / "boltzmann_tables_clean.npz")
    return {f: np.asarray(z[f"{name}__{f}"]) for f in ("lo", "hi", "binw", "U", "centre")}


def table_edges(t):
    return t["lo"] + np.arange(len(t["U"]) + 1) * t["binw"]


def ref_centres_masses(name):
    """Normalised per-bin mass over the physical domain q in [-1, 1]."""
    t = load_table(name)
    c = t["centre"]
    m = np.exp(-t["U"] / KBT)
    mask = (c >= -1.0) & (c <= 1.0)
    c, m = c[mask], m[mask]
    m = m / m.sum()
    return c, m, t


def dist_stats(c, m):
    """cis/trans/middle masses, peak positions, mean, sigma, skew, quantiles, from bin masses.

    cis/trans are the dihedral's two wells; for a single-peaked coordinate the cis_* and
    trans_* fields are reported as nan. trans_depth_kbt is ln(cis_peak/trans_peak), the cis
    well depth relative to the trans well, the field's key energy gap.
    """
    mean = float((m * c).sum())
    sd = float(np.sqrt((m * (c - mean) ** 2).sum()))
    skew = float((m * (c - mean) ** 3).sum() / sd ** 3)
    cs = np.cumsum(m)

    def q(p):
        return float(c[np.searchsorted(cs, p)])

    cis = c > CIS_LO
    trans = c < TRANS_HI
    mid = ~(cis | trans)

    def _peak(mask):
        if not mask.any():
            return float("nan"), float("nan")
        j = int(np.argmax(m[mask]))
        pos = float(c[mask][j])
        peak = float(m[mask][j])
        return pos, peak

    cis_pos, cis_peak = _peak(cis)
    trans_pos, trans_peak = _peak(trans)
    depth = float(np.log(cis_peak / trans_peak)) if (cis_peak > 0 and trans_peak > 0) else float("nan")
    return {
        "cis_pos": cis_pos, "cis_mass": float(m[cis].sum()),
        "trans_pos": trans_pos, "trans_mass": float(m[trans].sum()),
        "mid_mass": float(m[mid].sum()),
        "cis_peak": cis_peak, "trans_peak": trans_peak, "trans_depth_kbt": depth,
        "mean": mean, "sd": sd, "skew": skew,
        "q01": q(0.01), "q25": q(0.25), "q50": q(0.50), "q75": q(0.75), "q99": q(0.99),
        "mode": float(c[int(np.argmax(m))]),
    }


def print_stats(label, st, ref=None, indent="  "):
    keys = ["cis_pos", "cis_mass", "trans_pos", "trans_mass", "mid_mass", "trans_depth_kbt",
            "mean", "sd", "skew", "mode"]
    for k in keys:
        v = st[k]
        if v != v:  # nan
            continue
        extra = f"   (ref {ref[k]:.4f})" if (ref is not None and k in ref and ref[k] == ref[k]) else ""
        print(f"{indent}{k:14s} {v:.4f}{extra}")


# ------------------------------------------------------------------ model distribution
def cheb(q, n):
    """Chebyshev polynomial of the first kind T_n(q) = cos(n arccos q), vectorised."""
    q = np.asarray(q, dtype=float)
    if n == 0:
        return np.ones_like(q)
    if n == 1:
        return q
    t0, t1 = np.ones_like(q), q
    for _ in range(2, n + 1):
        t0, t1 = t1, 2.0 * q * t1 - t0
    return t1


def cheb_torch(q, n):
    if n == 0:
        return torch.ones_like(q)
    if n == 1:
        return q
    t0, t1 = torch.ones_like(q), q
    for _ in range(2, n + 1):
        t0, t1 = t1, 2.0 * q * t1 - t0
    return t1


def bin_index_of(q, t):
    """Bin index of q in the table's binning (same rule as the histogram below)."""
    return np.floor((q - float(t["lo"])) / float(t["binw"])).astype(int)


def model_bin_masses(Vq, t, nphi=NPHI):
    """P(q) propto exp(-V(cos phi)/kBT) under flat phi, binned into the reference bins,
    normalised over q in [-1, 1]. This is the TORSION measure: q = cos(phi) with flat dphi."""
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    q = np.cos(phi)
    w = np.exp(-Vq(q) / KBT)
    edges = table_edges(t)
    hist, _ = np.histogram(q, bins=edges, weights=w)
    c = t["centre"]
    mask = (c >= -1.0) & (c <= 1.0)
    m = hist[mask]
    m = m / m.sum()
    return c[mask], m


def model_bin_masses_flatq(Vq, t, nq=400_000):
    """P(q) propto exp(-V(q)/kBT) under FLAT q (the bond-angle measure, q = cos(theta) uniform).
    Used for the angle coordinate, which carries flat-cos measure, not flat-phi."""
    q = np.linspace(-1.0, 1.0, nq, endpoint=True)
    w = np.exp(-Vq(q) / KBT)
    edges = table_edges(t)
    hist, _ = np.histogram(q, bins=edges, weights=w)
    c = t["centre"]
    mask = (c >= -1.0) & (c <= 1.0)
    m = hist[mask]
    m = m / m.sum()
    return c[mask], m


# ------------------------------------------------------------------ candidate potentials
def v_fourier(K):
    def V(q):
        q = np.asarray(q, dtype=float)
        return sum(K[n - 1] * cheb(q, n) for n in range(1, len(K) + 1))
    return V


def v_tabulated(t, jacobian):
    """Interpolate the stored table U(q) in q, optionally adding the Jacobian term
    -0.5 kBT ln(1-q^2). Linear in q, matching boltzmann_bonded._sample's spirit."""
    centre = t["centre"]
    U = t["U"]
    binw = float(t["binw"])

    def V(q):
        q = np.asarray(q, dtype=float)
        u = (q - float(t["lo"])) / binw - 0.5
        i0 = np.floor(u).astype(int).clip(0, len(U) - 2)
        f = (u - i0).clip(0.0, 1.0)
        out = U[i0] + f * (U[i0 + 1] - U[i0])
        if jacobian:
            out = out - 0.5 * KBT * np.log(np.maximum(1.0 - q * q, 1e-8))
        return out
    return V


# ------------------------------------------------------------------ Fourier ML fit
def fourier_ml_fit(c, m, order, washed=False, nphi=NPHI_FIT, restarts=12, kbound=200.0):
    """Maximum-likelihood fit of V(phi) = sum_{n=1..order} K_n cos(n phi) to the reference.

    Objective: -sum_i m_i ln pmodel_i, pmodel = exp(-V/kBT)/Z on a fixed uniform phi grid,
    binned into the reference bins. m is the reference bin mass over q in [-1,1]. The exponent
    is shifted by its per-evaluation minimum so exp never overflows (the optimisation may probe
    coefficients up to kbound kJ/mol).

    washed=True zeroes the trans peak (q < TRANS_HI) first, the single-cis degradation.
    """
    if washed:
        m = m.copy()
        m[c < TRANS_HI] = 0.0
        m = m / m.sum()
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    qgrid = np.cos(phi)
    basis = np.stack([cheb(qgrid, n) for n in range(1, order + 1)])
    idx = bin_index_of(qgrid, load_table("dihedral"))
    nbins = len(m)
    valid = (idx >= 0) & (idx < nbins)
    idx = idx[valid]
    basis = basis[:, valid]
    m_target = m.astype(float)
    keep = m_target > 0
    lnm = np.log(m_target[keep])
    keep_bins = np.where(keep)[0]

    def nll(K):
        K = np.asarray(K, dtype=float)
        E = K @ basis                      # V(phi)/kBT is E/KBT
        E = E - E.min()                    # shift so w <= 1, no overflow
        w = np.exp(-E / KBT)
        Z = w.sum()
        pm = np.zeros(nbins)
        np.add.at(pm, idx, w)
        pm = pm / Z
        pm = np.maximum(pm[keep_bins], 1e-12)
        return float(-(lnm @ np.log(pm)))

    bounds = [(-kbound, kbound)] * order
    best = None
    rng = np.random.default_rng(20260218)
    for trial in range(restarts):
        if trial == 0:
            K0 = np.zeros(order)
        elif trial == 1:
            # hand-picked double-well guess: deep cis, shallow trans
            K0 = np.zeros(order)
            K0[0] = 3.0   # -cos phi lowers cis (phi=0) relative to trans (phi=pi)
        else:
            K0 = rng.normal(0.0, 5.0, size=order)
        res = minimize(nll, K0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 800, "ftol": 1e-14})
        if best is None or res.fun < best.fun:
            best = res
    return best.x, best.fun


# ------------------------------------------------------------------ candidate C: Gaussian mixture in phi
def gauss_mixture_potential(params, washed=False):
    """V(phi) = -kBT ln f(phi), where f is an even Gaussian mixture + uniform tail:

        f(phi) = w1 [G(phi;+phi1,s1) + G(phi;-phi1,s1)]
               + w2 [G(phi;+phi2,s2) + G(phi;-phi2,s2)] + w3

    phi2 = pi - phi1 (the trans partner of cis), and the +- pairs enforce f(phi)=f(2pi-phi),
    which is the symmetry the field sees (it only reads q = cos(phi)). G is a wrapped Gaussian
    (period 2pi) so the components stay smooth across the branch cut. w = softmax(logw). If
    washed, the trans Gaussian (w2) is forced to zero -- the one-switch degradation to a single
    cis well plus tail.

    params = [logw1, logw2, logw3, phi1, logs1, logs2].
    """
    logw1, logw2, logw3, phi1, logs1, logs2 = params
    if washed:
        logw2 = -np.inf
    lw = np.array([logw1, logw2, logw3])
    lw = lw - lw.max()
    w = np.exp(lw)
    w = w / w.sum()
    s1, s2 = np.exp(logs1), np.exp(logs2)
    phi2 = np.pi - phi1

    def V(q):
        q = np.asarray(q, dtype=float)
        u = np.arccos(np.clip(q, -1, 1))
        return -KBT * np.log(
            w[0] * (gauss_wrap(u, phi1, s1) + gauss_wrap(u, -phi1, s1))
            + w[1] * (gauss_wrap(u, phi2, s2) + gauss_wrap(u, -phi2, s2))
            + w[2] * (2.0 / np.pi)                       # uniform on [0, pi] == flat phi
        )
    return V


def gauss_wrap(u, u0, s, nwrap=2):
    """Periodic (period 2pi) wrapped Gaussian evaluated at u in [0, pi]."""
    total = np.zeros_like(u)
    for k in range(-nwrap, nwrap + 1):
        d = u - (u0 + 2.0 * np.pi * k)
        total += np.exp(-0.5 * (d / s) ** 2) / (s * np.sqrt(2.0 * np.pi))
    return total


def gauss_mixture_ml_fit(c, m, washed=False, restarts=8):
    """ML fit of the Gaussian-mixture potential to the reference bin masses (flat-phi measure).

    The NLL is computed on a uniform phi grid exactly as in fourier_ml_fit, so the two
    candidates share the same objective and the same measure."""
    if washed:
        m = m.copy()
        m[c < TRANS_HI] = 0.0
        m = m / m.sum()
    phi = np.linspace(0.0, 2.0 * np.pi, NPHI_FIT, endpoint=False)
    qgrid = np.cos(phi)
    idx = bin_index_of(qgrid, load_table("dihedral"))
    nbins = len(m)
    valid = (idx >= 0) & (idx < nbins)
    idx = idx[valid]
    qvalid = qgrid[valid]
    m_target = m.astype(float)
    keep = m_target > 0
    lnm = np.log(m_target[keep])
    keep_bins = np.where(keep)[0]

    def nll(params):
        V = gauss_mixture_potential(params, washed=washed)
        E = V(qvalid)
        E = E - E.min()
        w = np.exp(-E / KBT)
        Z = w.sum()
        pm = np.zeros(nbins)
        np.add.at(pm, idx, w)
        pm = pm / Z
        pm = np.maximum(pm[keep_bins], 1e-12)
        return float(-(lnm @ np.log(pm)))

    # bounds: logw in [-20, 20], phi1 in [0.01, pi/2-0.01], logs in [log(0.01), log(1.5)]
    bounds = [(-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0),
              (0.01, np.pi / 2 - 0.01), (np.log(0.01), np.log(1.5)), (np.log(0.01), np.log(1.5))]
    best = None
    rng = np.random.default_rng(20260218)
    guesses = []
    if not washed:
        guesses.append([2.0, -1.0, -2.0, 0.081, np.log(0.05), np.log(0.3)])   # cis, trans, tail
        guesses.append([2.0, 0.0, -1.0, 0.081, np.log(0.05), np.log(0.3)])
    else:
        guesses.append([2.0, -20.0, -1.0, 0.081, np.log(0.05), np.log(0.3)])
    for _ in range(restarts):
        guesses.append([rng.uniform(-2, 3), rng.uniform(-3, 2), rng.uniform(-3, 2),
                        rng.uniform(0.03, 0.5), rng.uniform(np.log(0.02), np.log(0.8)),
                        rng.uniform(np.log(0.05), np.log(1.0))])
    for g in guesses:
        res = minimize(nll, g, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 800, "ftol": 1e-14})
        if best is None or res.fun < best.fun:
            best = res
    return best.x, best.fun


# ------------------------------------------------------------------ torch forms for force
def v_torch(kind, K=None, t=None):
    """Return a torch callable V(q) -> scalar energy for autograd force evaluation."""
    if kind == "harmonic":
        c0 = math.cos(C.DIH_PPPP)
        return lambda q: 0.5 * C.K_DIH * (q - c0) ** 2
    if kind == "fourier":
        return lambda q: sum(K[n - 1] * cheb_torch(q, n) for n in range(1, len(K) + 1))
    if kind == "tab":
        centre = torch.tensor(t["centre"], dtype=torch.float64)
        U = torch.tensor(t["U"], dtype=torch.float64)
        lo = float(t["lo"]); binw = float(t["binw"])
        return lambda q: _interp(q, centre, U, lo, binw)
    if kind == "tab_jac":
        centre = torch.tensor(t["centre"], dtype=torch.float64)
        U = torch.tensor(t["U"], dtype=torch.float64)
        lo = float(t["lo"]); binw = float(t["binw"])
        return lambda q: (_interp(q, centre, U, lo, binw)
                          - 0.5 * KBT * torch.log(torch.clamp(1.0 - q * q, min=1e-8)))
    if kind == "gauss":
        return lambda q: _gauss_torch(q, K)
    if kind == "gauss_cis":
        return lambda q: _gauss_torch(q, K, washed=True)
    raise ValueError(kind)


def _gauss_torch(q, params, washed=False):
    """Torch, differentiable Gaussian-mixture potential V(q)."""
    logw1, logw2, logw3, phi1, logs1, logs2 = [torch.tensor(x, dtype=torch.float64) for x in params]
    if washed:
        logw2 = torch.tensor(-float("inf"), dtype=torch.float64)
    lw = torch.stack([logw1, logw2, logw3])
    lw = lw - lw.max()
    w = torch.exp(lw)
    w = w / w.sum()
    s1, s2 = torch.exp(logs1), torch.exp(logs2)
    phi2 = np.pi - phi1
    u = torch.arccos(torch.clamp(q, -1.0, 1.0))

    def gw(u0, s):
        total = torch.zeros_like(u)
        for k in range(-3, 4):
            d = u - (u0 + 2.0 * np.pi * k)
            total = total + torch.exp(-0.5 * (d / s) ** 2) / (s * np.sqrt(2.0 * np.pi))
        return total

    f = (w[0] * (gw(phi1, s1) + gw(-phi1, s1))
         + w[1] * (gw(phi2, s2) + gw(-phi2, s2))
         + w[2] * (2.0 / np.pi))
    return -KBT * torch.log(f)


def _interp(q, centre, U, lo, binw):
    u = (q - lo) / binw - 0.5
    i0 = torch.floor(u).long().clamp(0, len(U) - 2)
    f = (u - i0.double()).clamp(0.0, 1.0)
    return U[i0] + f * (U[i0 + 1] - U[i0])


def _cos_dih(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n0 = torch.linalg.cross(b0, b1)
    n1 = torch.linalg.cross(b1, b2)
    n0 = n0 / torch.clamp(torch.norm(n0), min=1e-6)
    n1 = n1 / torch.clamp(torch.norm(n1), min=1e-6)
    return (n0 * n1).sum()


def _cos_angle(p0, p1, p2):
    v1, v2 = p0 - p1, p2 - p1
    n1 = torch.clamp(torch.norm(v1), min=1e-6)
    n2 = torch.clamp(torch.norm(v2), min=1e-6)
    return ((v1 * v2).sum() / (n1 * n2)).clamp(-1 + 1e-6, 1 - 1e-6)


def dihedral_force(pos4, Vq):
    """Per-bead force magnitude (max over the 4 beads) of V(q(x)) by autograd, as _dihedral_f."""
    p = pos4.detach().clone().requires_grad_(True)
    q = _cos_dih(p[0], p[1], p[2], p[3])
    E = Vq(q)
    E.backward()
    F = -p.grad
    return q.item(), float(F.norm(dim=-1).max().item())


def angle_force(pos3, Efn):
    """Per-bead force magnitude (max over the 3 beads) of E(q(x)) by autograd, as _angle_f."""
    p = pos3.detach().clone().requires_grad_(True)
    q = _cos_angle(p[0], p[1], p[2])
    E = Efn(q)
    E.backward()
    F = -p.grad
    return q.item(), float(F.norm(dim=-1).max().item())


# ------------------------------------------------------------------ force on real geometry
def collect_forces(kinds):
    """For every real P-P-P-P window, evaluate q and the max per-bead force under each candidate."""
    structs = B.load_structures(limit=8)
    out = {k: [] for k in kinds}
    qs = []
    for s in structs:
        pos = torch.tensor(s["pos"], dtype=torch.float64)   # (L,3)
        L = pos.shape[0]
        if L < 4:
            continue
        for i in range(L - 3):
            win = pos[i:i + 4]
            q0, _ = dihedral_force(win, lambda q: q * 0.0)  # q only
            qs.append(q0)
            for k in kinds:
                _, fm = dihedral_force(win, kinds[k])
                out[k].append(fm)
    return np.array(qs), {k: np.array(v) for k, v in out.items()}


# ------------------------------------------------------------------ angle
def fit_cubic_angle(c, m):
    """Analytic non-harmonic angle: U(c) = k/2 (c-c0)^2 + a (c-c0)^3, fit to -kBT ln P_ref(c).
    The cubic term carries the skew; c0 is the reference mode. Flat-cos measure, no Jacobian."""
    i0 = int(np.argmax(m))
    c0 = float(c[i0])
    dc = c - c0
    p = np.maximum(m, 1e-12)
    u = -KBT * np.log(p)
    u -= u.min()
    A = np.column_stack([0.5 * dc ** 2, dc ** 3])
    sol, *_ = np.linalg.lstsq(A, u, rcond=None)
    k, a = sol

    def U(q):
        q = np.asarray(q, dtype=float)
        d = q - c0
        return 0.5 * k * d ** 2 + a * d ** 3
    return c0, k, a, U


# ------------------------------------------------------------------ main
def main():
    print(f"KBT = {KBT} kJ/mol   force_cap = {FORCE_CAP} kJ/mol/nm")
    print(f"cis boundary q > {CIS_LO}, trans boundary q < {TRANS_HI}")
    print()

    c, m, t = ref_centres_masses("dihedral")
    ref = dist_stats(c, m)
    print("=== reference (dihedral, from boltzmann_tables_clean.npz) ===")
    print_stats("ref", ref, indent="  ")
    print()

    # ---- candidate A: Fourier (even) full fit, orders 4 (IsRNA2) and 8 for the sufficiency test
    K_full4 = None
    for order in (4, 8):
        K_full, nll = fourier_ml_fit(c, m, order=order, washed=False)
        if order == 4:
            K_full4 = K_full
        cm_A, mm_A = model_bin_masses(v_fourier(K_full), t)
        st_A = dist_stats(cm_A, mm_A)
        print(f"candidate A (Fourier/even, order {order}, ML fit; NLL = {nll:.4f}):")
        print(f"  K = [{', '.join(f'{k:+.4f}' for k in K_full)}]")
        print_stats("A", st_A, ref)
        print()

    # ---- candidate A-degraded: Fourier single cis (washed reference)
    K_cis, nll_cis = fourier_ml_fit(c, m, order=4, washed=True)
    cm_Ac, mm_Ac = model_bin_masses(v_fourier(K_cis), t)
    st_Ac = dist_stats(cm_Ac, mm_Ac)
    print("candidate A-degraded (Fourier, order 4, single cis; trans peak washed out):")
    print(f"  K = [{', '.join(f'{k:+.4f}' for k in K_cis)}]")
    print_stats("A_cis", st_Ac, indent="  ")
    print()

    # ---- candidate B: tabulated DBI, two variants
    print("candidate B1 (tabulated DBI, measure-naive: U = -kBT ln P_ref(q), as shipped):")
    cm_B1, mm_B1 = model_bin_masses(v_tabulated(t, jacobian=False), t)
    st_B1 = dist_stats(cm_B1, mm_B1)
    print_stats("B1", st_B1, ref)
    print()

    print("candidate B2 (tabulated DBI, measure-correct: U = -kBT ln P_ref(q) - 0.5 kBT ln(1-q^2)):")
    cm_B2, mm_B2 = model_bin_masses(v_tabulated(t, jacobian=True), t)
    st_B2 = dist_stats(cm_B2, mm_B2)
    print_stats("B2", st_B2, ref)
    print()

    # ---- candidate C: Gaussian mixture (parametric double well + tail), degradable
    g_params, g_nll = gauss_mixture_ml_fit(c, m, washed=False)
    st_C = dist_stats(*model_bin_masses(gauss_mixture_potential(g_params), t))
    print(f"candidate C (Gaussian mixture in phi, even; ML fit; NLL = {g_nll:.4f}):")
    print(f"  params [logw1,logw2,logw3,phi1,logs1,logs2] = [{', '.join(f'{x:+.4f}' for x in g_params)}]")
    lw = np.array(g_params[:3]); lw = lw - lw.max(); wmix = np.exp(lw); wmix = wmix / wmix.sum()
    print(f"  weights (cis,trans,tail) = {wmix[0]:.4f}, {wmix[1]:.4f}, {wmix[2]:.4f}  "
          f"phi1 = {g_params[3]*180/np.pi:.2f} deg, s1 = {np.exp(g_params[4])*180/np.pi:.2f} deg, "
          f"s2 = {np.exp(g_params[5])*180/np.pi:.2f} deg")
    print_stats("C", st_C, ref)
    print()

    # ---- candidate C-degraded: trans weight zeroed (single cis + tail)
    g_cis_params, _ = gauss_mixture_ml_fit(c, m, washed=True)
    st_Cc = dist_stats(*model_bin_masses(gauss_mixture_potential(g_cis_params, washed=True), t))
    print("candidate C-degraded (Gaussian mixture, trans weight forced to 0):")
    print_stats("C_cis", st_Cc, indent="  ")
    print()

    # ---- forces on real geometry
    print("=== forces (max per-bead |F| over real P-P-P-P windows, 8 chains) ===")
    kinds = {
        "shipped harmonic": v_torch("harmonic"),
        "A Fourier order4": v_torch("fourier", K=K_full4),
        "A Fourier cis": v_torch("fourier", K=K_cis),
        "B1 tab naive": v_torch("tab", t=t),
        "B2 tab jac": v_torch("tab_jac", t=t),
        "C gauss mix": v_torch("gauss", K=g_params),
        "C gauss cis": v_torch("gauss_cis", K=g_cis_params),
    }
    qs, forces = collect_forces(kinds)
    print(f"  {len(qs)} dihedral windows; q in [{qs.min():.3f}, {qs.max():.3f}]")
    print(f"  {'form':20s} {'median':>9s} {'p99':>9s} {'max':>9s} {'>5000':>7s}")
    for k, v in forces.items():
        print(f"  {k:20s} {np.median(v):9.2f} {np.percentile(v, 99):9.2f} "
              f"{v.max():9.2f} {int((v > FORCE_CAP).sum()):7d}")
    print()

    # ---- angle
    print("=== angle (P-P-P, flat-cos measure) ===")
    ca, ma, ta = ref_centres_masses("angle")
    ra = dist_stats(ca, ma)
    print("  reference:")
    print_stats("angle", ra, indent="    ")
    cm_a, mm_a = model_bin_masses_flatq(v_tabulated(ta, jacobian=False), ta)
    st_a = dist_stats(cm_a, mm_a)
    print("  tabulated DBI (flat-cos measure, the correct one for a bond angle):")
    print_stats("angle-tab", st_a, ra)
    c0, k_a, a_a, U_cubic = fit_cubic_angle(ca, ma)
    cm_c, mm_c = model_bin_masses_flatq(U_cubic, ta)
    st_c = dist_stats(cm_c, mm_c)
    print(f"  cubic analytic U(c) = 0.5*{k_a:.1f}*(c-{c0:.3f})^2 + {a_a:+.1f}*(c-{c0:.3f})^3:")
    print_stats("angle-cubic", st_c, ra)
    print(f"  (shipped K_ANGLE={C.K_ANGLE}; naive kBT/sigma^2 = {KBT/ra['sd']**2:.2f}; the "
          f"truncation-corrected harmonic k is 12.16, from the prior dihedral_measure work)")

    # ---- angle forces
    print()
    print("=== angle forces (max per-bead |F| over real P-P-P windows, 8 chains) ===")
    ta_t = {k: torch.tensor(ta[k], dtype=torch.float64) for k in ("centre", "U")}

    def e_shipped(q):
        return 0.5 * C.K_ANGLE * (q - math.cos(C.ANGLE_PPP)) ** 2

    def e_tab(q):
        return _interp(q, ta_t["centre"], ta_t["U"], float(ta["lo"]), float(ta["binw"]))

    def e_cubic(q):
        d = q - c0
        return 0.5 * k_a * d ** 2 + a_a * d ** 3

    af = {"shipped harmonic": [], "tabulated DBI": [], "cubic analytic": []}
    structs = B.load_structures(limit=8)
    for s in structs:
        pos = torch.tensor(s["pos"], dtype=torch.float64)
        L = pos.shape[0]
        if L < 3:
            continue
        for i in range(L - 2):
            for key, fn in (("shipped harmonic", e_shipped),
                            ("tabulated DBI", e_tab),
                            ("cubic analytic", e_cubic)):
                _, fm = angle_force(pos[i:i + 3], fn)
                af[key].append(fm)
    print(f"  {'form':20s} {'median':>9s} {'p99':>9s} {'max':>9s} {'>5000':>7s}")
    for key in af:
        v = np.array(af[key])
        print(f"  {key:20s} {np.median(v):9.2f} {np.percentile(v, 99):9.2f} "
              f"{v.max():9.2f} {int((v > FORCE_CAP).sum()):7d}")


if __name__ == "__main__":
    main()
