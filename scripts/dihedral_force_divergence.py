"""Force magnitudes of every non-harmonic dihedral candidate, and the divergence that rules
out the naive "exact" form.

Prototype. Never writes to src/. Reads results/boltzmann_tables_clean.npz and the first 8
chains of _cgdata/rsRNASP/Training_set.

What this measures, and why:

  * The measure-correct tabulated form V(q) = U_ref(q) - 0.5 kBT ln(1-q^2) (this is candidate
    B2 of scripts/dihedral_nonharmonic_fit.py) is the ONLY form that reproduced the reference
    item-by-item. Its second term is the entropic Jacobian d(cos phi)/dphi = -sin(phi):
    -0.5 kBT ln(1-q^2) = -kBT ln(sin phi). Its force contribution is
        d/dq [-0.5 kBT ln(1-q^2)] = kBT q / (1-q^2),
    which DIVERGES as q -> +-1 (phi -> 0, pi). The reference cis and trans peaks sit at
    q = +-0.9967, i.e. right next to those poles, so the "exact" form has an unbounded force
    exactly where the mass is. This script quantifies that divergence, then proposes and
    measures a regularized form whose force is bounded, and checks whether regularization
    costs anything on the distribution criteria.

  * A smooth finite Fourier V(phi) = sum_{n=1..N} K_n cos(n phi) = sum K_n T_n(q) is fitted to
    the flat-phi reference potential U_ref(q) - kBT ln(sin phi) by weighted least squares
    (weights = reference bin mass), N = 1..6. Its force is BOUNDED (a polynomial in q), so the
    question is whether it can both match the distribution and stay under force_cap = 5000.
    The N = 1..6 series also shows where sigma/skew converge.

The trans-well depth reported here is the WELL-TO-WELL depth: ln(peak mass cis / peak mass trans)
= (U at trans mode - U at cis mode)/kBT, i.e. the energy gap from the global minimum (cis mode)
down to the trans mode, not a peak-to-saddle barrier.

Run: python scripts/dihedral_force_divergence.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402  (KBT, load_structures)
import torusfold.scheme2.torch_cgsim as C   # noqa: E402  (K_DIH, DIH_PPPP)

KBT = B.KBT
CIS_LO = 0.8
TRANS_HI = -0.8
FORCE_CAP = 5000.0
NPHI = 2_000_000


# ------------------------------------------------------------------ reference
def ref_table():
    z = np.load(REPO / "results" / "boltzmann_tables_clean.npz")
    return {f: np.asarray(z[f"dihedral__{f}"]) for f in ("lo", "hi", "binw", "U", "centre")}


def ref_masses():
    t = ref_table()
    c = t["centre"]
    m = np.exp(-t["U"] / KBT)
    mask = (c >= -1.0) & (c <= 1.0)
    c, m = c[mask], m[mask]
    return c, m / m.sum(), t


def stats(c, m):
    mean = float((m * c).sum())
    sd = float(np.sqrt((m * (c - mean) ** 2).sum()))
    skew = float((m * (c - mean) ** 3).sum() / sd ** 3)
    cis = c > CIS_LO
    trans = c < TRANS_HI
    mid = ~(cis | trans)
    cis_peak = float(m[cis].max())
    trans_peak = float(m[trans].max())
    depth = float(np.log(cis_peak / trans_peak)) if trans_peak > 0 else float("nan")
    return {"cis_mass": float(m[cis].sum()), "trans_mass": float(m[trans].sum()),
            "mid_mass": float(m[mid].sum()), "mean": mean, "sd": sd, "skew": skew,
            "trans_depth_kbt": depth, "mode": float(c[int(np.argmax(m))])}


def bin_masses(Vq, t, nphi=NPHI):
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    q = np.cos(phi)
    w = np.exp(-Vq(q) / KBT)
    edges = t["lo"] + np.arange(len(t["U"]) + 1) * t["binw"]
    hist, _ = np.histogram(q, bins=edges, weights=w)
    c = t["centre"]
    mask = (c >= -1.0) & (c <= 1.0)
    m = hist[mask]
    return c[mask], m / m.sum()


# ------------------------------------------------------------------ Chebyshev
def T(q, n):
    q = np.asarray(q, dtype=float)
    if n == 0:
        return np.ones_like(q)
    if n == 1:
        return q
    t0, t1 = np.ones_like(q), q
    for _ in range(2, n + 1):
        t0, t1 = t1, 2.0 * q * t1 - t0
    return t1


def U2(q, n):
    """Chebyshev of the second kind U_n(q)."""
    q = np.asarray(q, dtype=float)
    if n == 0:
        return np.ones_like(q)
    if n == 1:
        return 2.0 * q
    u0, u1 = np.ones_like(q), 2.0 * q
    for _ in range(2, n + 1):
        u0, u1 = u1, 2.0 * q * u1 - u0
    return u1


def dTdq(q, n):
    """d/dq T_n(q) = n U_{n-1}(q)."""
    return n * U2(q, n - 1)


# ------------------------------------------------------------------ Fourier fit
def fourier_fit(c, m, t, N):
    """Weighted least squares of V_N(phi) = sum_{n=0..N} K_n cos(n phi) to the flat-phi
    reference potential U_ref(q) - kBT ln(sin phi), evaluated at the reference bin centres.
    Weight = reference bin mass, so the fit pins the potential where the mass is."""
    q = c
    sinphi = np.sqrt(np.clip(1.0 - q * q, 1e-12, None))
    U_phi = (-KBT * np.log(np.maximum(m, 1e-12))) - KBT * np.log(sinphi)
    w = m.copy()
    A = np.column_stack([np.sqrt(w) * T(q, n) for n in range(0, N + 1)])
    y = np.sqrt(w) * U_phi
    K, *_ = np.linalg.lstsq(A, y, rcond=None)
    return K   # [K0, K1, ..., KN]


def v_fourier(K):
    K = np.asarray(K, dtype=float)

    def V(q):
        return sum(K[n] * T(q, n) for n in range(len(K)))
    return V


def dv_fourier(K):
    K = np.asarray(K, dtype=float)

    def dV(q):
        return sum(K[n] * dTdq(q, n) for n in range(1, len(K)))
    return dV


# ------------------------------------------------------------------ table derivatives
def uref_interp(t):
    centre = t["centre"]
    U = t["U"]
    binw = float(t["binw"])

    def V(q):
        q = np.asarray(q, dtype=float)
        u = (q - float(t["lo"])) / binw - 0.5
        i0 = np.floor(u).astype(int).clip(0, len(U) - 2)
        f = (u - i0).clip(0.0, 1.0)
        return U[i0] + f * (U[i0 + 1] - U[i0])
    return V


def uref_deriv(t):
    """Piecewise-constant derivative of the linear-interpolated table."""
    centre = t["centre"]
    U = t["U"]
    binw = float(t["binw"])
    dU = (U[1:] - U[:-1]) / binw       # slope on each interval [centre[i], centre[i+1]]

    def dV(q):
        q = np.asarray(q, dtype=float)
        u = (q - float(t["lo"])) / binw - 0.5
        i0 = np.floor(u).astype(int).clip(0, len(dU) - 1)
        return dU[i0]
    return dV


def dv_exact(t, eps=None):
    """dV/dq of V = U_ref(q) - 0.5 kBT ln(1-q^2), optionally regularized with
    1-q^2 -> max(1-q^2, eps^2). eps=None is the exact (divergent) form."""
    dU = uref_deriv(t)

    def dV(q):
        q = np.asarray(q, dtype=float)
        base = dU(q)
        if eps is None:
            jac = KBT * q / (1.0 - q * q)
        else:
            denom = np.maximum(1.0 - q * q, eps * eps)
            jac = np.where(1.0 - q * q > eps * eps, KBT * q / denom, 0.0)
        return base + jac
    return dV


def v_exact(t, eps=None):
    U = uref_interp(t)

    def V(q):
        q = np.asarray(q, dtype=float)
        if eps is None:
            return U(q) - 0.5 * KBT * np.log(np.maximum(1.0 - q * q, 1e-8))
        return U(q) - 0.5 * KBT * np.log(np.maximum(1.0 - q * q, eps * eps))
    return V


# ------------------------------------------------------------------ torch force on real geometry
def _cos_dih(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n0 = torch.linalg.cross(b0, b1)
    n1 = torch.linalg.cross(b1, b2)
    n0 = n0 / torch.clamp(torch.norm(n0), min=1e-6)
    n1 = n1 / torch.clamp(torch.norm(n1), min=1e-6)
    return (n0 * n1).sum()


def v_torch(kind, K=None, t=None, eps=None):
    if kind == "harmonic":
        c0 = math.cos(C.DIH_PPPP)
        return lambda q: 0.5 * C.K_DIH * (q - c0) ** 2
    if kind == "fourier":
        Kt = [torch.tensor(x, dtype=torch.float64) for x in K]

        def V(q):
            return sum(Kt[n] * _T_torch(q, n) for n in range(len(Kt)))
        return V
    if kind == "exact" or kind == "reg":
        centre = torch.tensor(t["centre"], dtype=torch.float64)
        U = torch.tensor(t["U"], dtype=torch.float64)
        lo = float(t["lo"]); binw = float(t["binw"])
        e2 = None if eps is None else eps * eps

        def V(q):
            u = (q - lo) / binw - 0.5
            i0 = torch.floor(u).long().clamp(0, len(U) - 2)
            f = (u - i0.double()).clamp(0.0, 1.0)
            base = U[i0] + f * (U[i0 + 1] - U[i0])
            if e2 is None:
                return base - 0.5 * KBT * torch.log(torch.clamp(1.0 - q * q, min=1e-8))
            return base - 0.5 * KBT * torch.log(torch.clamp(1.0 - q * q, min=e2))
        return V
    raise ValueError(kind)


def _T_torch(q, n):
    if n == 0:
        return torch.ones_like(q)
    if n == 1:
        return q
    t0, t1 = torch.ones_like(q), q
    for _ in range(2, n + 1):
        t0, t1 = t1, 2.0 * q * t1 - t0
    return t1


def force_on_geometry(kinds):
    structs = B.load_structures(limit=8)
    out = {k: [] for k in kinds}
    qs = []
    for s in structs:
        pos = torch.tensor(s["pos"], dtype=torch.float64)
        L = pos.shape[0]
        if L < 4:
            continue
        for i in range(L - 3):
            win = pos[i:i + 4]
            for k in kinds:
                p = win.detach().clone().requires_grad_(True)
                q = _cos_dih(p[0], p[1], p[2], p[3])
                E = kinds[k](q)
                E.backward()
                F = -p.grad
                out[k].append(float(F.norm(dim=-1).max().item()))
                if k == "shipped harmonic":
                    qs.append(q.item())
    return np.array(qs), {k: np.array(v) for k, v in out.items()}


# ------------------------------------------------------------------ main
def main():
    print(f"KBT = {KBT}, force_cap = {FORCE_CAP} kJ/mol/nm")
    c, m, t = ref_masses()
    ref = stats(c, m)
    print("reference: mode %.4f, sd %.4f, skew %.4f, cis %.4f, trans %.4f, mid %.4f, trans-depth %.2f kBT"
          % (ref["mode"], ref["sd"], ref["skew"], ref["cis_mass"], ref["trans_mass"],
             ref["mid_mass"], ref["trans_depth_kbt"]))
    print()

    # ---- Fourier N=1..6: coefficients, distribution stats, dV/dq at the two peaks
    print("=== Fourier V(phi)=sum K_n cos(n phi), weighted-LS to U_ref - kBT ln sin(phi) ===")
    print(f"{'N':>2s} {'sd':>7s} {'skew':>7s} {'mean':>7s} {'trans-depth':>12s} "
          f"{'|dV/dq|@cis':>12s} {'|dV/dq|@trans':>13s}  K")
    Ks = {}
    for N in range(1, 7):
        K = fourier_fit(c, m, t, N)
        Ks[N] = K
        cm, mm = bin_masses(v_fourier(K), t)
        st = stats(cm, mm)
        dV = dv_fourier(K)
        dcis = abs(dV(np.array([0.9967]))[0])
        dtrans = abs(dV(np.array([-0.9967]))[0])
        Kstr = " ".join(f"{K[n]:+.2f}" for n in range(1, N + 1))
        print(f"{N:2d} {st['sd']:7.4f} {st['skew']:7.3f} {st['mean']:7.4f} {st['trans_depth_kbt']:12.2f} "
              f"{dcis:12.2f} {dtrans:13.2f}  {Kstr}")
    print()
    print("  full distribution stats per N (reference: cis 0.6317, trans 0.0651, mid 0.3032, "
          "mode 0.9967):")
    print(f"  {'N':>2s} {'cis_mass':>9s} {'trans_mass':>11s} {'mid_mass':>9s} {'mode':>7s}")
    for N in range(1, 7):
        cm, mm = bin_masses(v_fourier(Ks[N]), t)
        st = stats(cm, mm)
        print(f"  {N:2d} {st['cis_mass']:9.4f} {st['trans_mass']:11.4f} {st['mid_mass']:9.4f} "
              f"{st['mode']:7.4f}")
    print()

    # ---- exact form divergence (dV/dq analytic)
    dV_ex = dv_exact(t, eps=None)
    print("=== exact form V = U_ref(q) - 0.5 kBT ln(1-q^2): dV/dq as q -> +-1 ===")
    print(f"  {'q':>9s} {'dV/dq':>12s}")
    for q in (0.9967, 0.999, 0.9999, 0.99999, -0.9967, -0.999, -0.9999):
        print(f"  {q:9.5f} {float(dV_ex(np.array([q]))[0]):12.1f}")
    print()

    # ---- regularized form: bounded dV/dq
    print("=== regularized form (1-q^2 floored at eps^2): max |dV/dq| over q ===")
    for deg in (2.0, 3.0, 5.0):
        eps = math.sin(math.radians(deg))
        dV_reg = dv_exact(t, eps=eps)
        qgrid = np.linspace(-0.99999, 0.99999, 2000001)
        d = dV_reg(qgrid)
        print(f"  eps=sin({deg:.0f} deg)={eps:.4f}: max|dV/dq| = {np.abs(d).max():.1f}")
    print()

    # ---- regularized table: does eps cost anything on the distribution?
    print("=== regularized table form: distribution stats vs exact (eps floors 1-q^2) ===")
    print(f"  {'form':18s} {'sd':>7s} {'skew':>7s} {'cis_mass':>9s} {'trans_mass':>11s} "
          f"{'mid_mass':>9s} {'mode':>7s}")
    for name, eps in (("exact B2", None), ("reg eps=2deg", math.sin(math.radians(2.0))),
                      ("reg eps=3deg", math.sin(math.radians(3.0))),
                      ("reg eps=5deg", math.sin(math.radians(5.0)))):
        cm, mm = bin_masses(v_exact(t, eps=eps), t)
        st = stats(cm, mm)
        print(f"  {name:18s} {st['sd']:7.4f} {st['skew']:7.3f} {st['cis_mass']:9.4f} "
              f"{st['trans_mass']:11.4f} {st['mid_mass']:9.4f} {st['mode']:7.4f}")
    print("  reference:           0.5916  -1.599    0.6317       0.0651    0.3032  0.9967")
    print()

    # ---- per-bead force on real reference geometry, overall and split by q-region
    print("=== per-bead |F| on real P-P-P-P windows (8 chains) ===")
    kinds = {
        "shipped harmonic": v_torch("harmonic"),
        "Fourier N=2": v_torch("fourier", K=Ks[2]),
        "Fourier N=3": v_torch("fourier", K=Ks[3]),
        "Fourier N=4": v_torch("fourier", K=Ks[4]),
        "Fourier N=6": v_torch("fourier", K=Ks[6]),
        "exact B2": v_torch("exact", t=t),
        "reg eps=3deg": v_torch("reg", t=t, eps=math.sin(math.radians(3.0))),
    }
    qs, forces = force_on_geometry(kinds)
    print(f"  {len(qs)} windows; q in [{qs.min():.4f}, {qs.max():.4f}]")
    print(f"  {'form':18s} {'median':>9s} {'p99':>9s} {'max':>9s} {'>5000':>6s}")
    for k, v in forces.items():
        print(f"  {k:18s} {np.median(v):9.2f} {np.percentile(v, 99):9.2f} "
              f"{v.max():9.2f} {int((v > FORCE_CAP).sum()):6d}")
    print()

    # per-region breakdown: cis peak (q>0.8), trans peak (q<-0.8), middle
    print("  max |F| split by q-region (real windows):")
    print(f"  {'form':18s} {'cis(q>0.8)':>11s} {'trans(q<-0.8)':>14s} {'mid':>9s}")
    cis_mask = qs > 0.8
    trans_mask = qs < -0.8
    mid_mask = ~(cis_mask | trans_mask)
    for k, v in forces.items():
        def mx(mask):
            return v[mask].max() if mask.any() else float("nan")
        print(f"  {k:18s} {mx(cis_mask):11.2f} {mx(trans_mask):14.2f} {mx(mid_mask):9.2f}")
    print()

    # ---- degradation: Fourier re-fit to the washed reference (single cis)
    print("=== degradation switch (trans well off) ===")
    m_washed = m.copy()
    m_washed[c < TRANS_HI] = 0.0
    m_washed = m_washed / m_washed.sum()
    for N in (2, 3, 4):
        K_w = fourier_fit(c, m_washed, t, N)
        cm, mm = bin_masses(v_fourier(K_w), t)
        st = stats(cm, mm)
        Kstr = " ".join(f"{K_w[n]:+.2f}" for n in range(1, N + 1))
        print(f"  Fourier N={N}, washed ref: K = {Kstr}")
        print(f"      -> cis {st['cis_mass']:.4f}, trans {st['trans_mass']:.4f}, "
              f"mid {st['mid_mass']:.4f}, sd {st['sd']:.4f}, mode {st['mode']:.4f}")
    print("  (a washed single-cis reference has cis ~0.67, trans 0, mid ~0.33 by construction)")
    print("  table form switch: wash the reference (zero q < -0.8, renormalise, rebuild the table).")
    print("  Fourier form switch: re-fit the LS to the washed reference -- the coefficients above.")
    print("  For N=2 the trans well depth is V(pi)-V(0) = -2*K1, so K1 IS the trans-depth knob.")
    print()


if __name__ == "__main__":
    main()
