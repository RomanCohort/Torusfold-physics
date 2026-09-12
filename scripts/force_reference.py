"""Reference implementation of the dihedral force. ONE function, used by every caller, so that
the force of a given candidate energy is computed exactly once and never drifts between scripts.

The dihedral is the P-P-P-P pseudo-torsion. Its observable in this force field is q = cos(phi),
the normal-vector dot product, exactly as torch_cgsim._dihedral_f computes it (cross products of
consecutive bond vectors). The force is -dE/dpos by autograd on THE SAME energy expression that is
returned: no post-hoc terms, no detach of any piece of the energy, no analytic gradient shortcut.

Function
--------
    dihedral_force(spec, pos, ij_like=None) -> (E: (B,), F: (B, N, 3))

    spec     explicit candidate (see SPECS below). The energy expression is fully determined by
             spec; nothing is inferred from the calling context.
    pos      (B, N, 3) float64, bead coordinates, N = 3L beads in P/C4'/N order per residue.
    ij_like  None -> consecutive P atoms (beads 3i, 3i+3, 3i+6, 3i+9 for i = 0..L-4 inclusive,
             i.e. the L-3 windows a linear backbone carries; L residues give L-3 P-P-P-P windows).
    E        (B,) total energy over all windows (sum of the per-window V(q)).
    F        (B, N, 3) force = -dE/dpos. Only the 4 atoms of each window carry force.

SPECS (spec is a string or a tuple; the tuple form carries parameters):
    "shipped_harmonic"           V(q) = 0.5*K_DIH*(q - cos(DIH_PPPP))^2,  K_DIH=7.2, DIH_PPPP=acos(0.975)
    "table"                      V(q) = linear interpolation of the stored dihedral table
                                 (results/boltzmann_tables_clean.npz), i.e. boltzmann_bonded._sample
                                 and the energy()/mixed_energy(which=["dihedral"]) path. Measure-naive.
    ("table_jac", None)          V(q) = table(q) - 0.5*kBT*ln(1-q^2).  This is the "exact" measure-correct
                                 form (candidate B2). The second term is the torsion Jacobian
                                 -kBT*ln(sin phi); its force kBT*q/(1-q^2) DIVERGES as q -> +-1.
    ("table_jac", eps)           V(q) = table(q) - 0.5*kBT*ln(max(1-q^2, eps^2)).  Regularized B2.
                                 The floor is on 1-q^2, which depends on q^2 only, so the
                                 regularisation is SYMMETRIC: for |q| > sqrt(1-eps^2) the Jacobian
                                 force is zeroed identically on the cis (q -> +1) and trans (q -> -1)
                                 sides; for |q| <= sqrt(1-eps^2) it is the exact kBT*q/(1-q^2).
    ("fourier", K)               V(q) = sum_{n=0..N} K[n]*T_n(q), T_n the Chebyshev polynomial of the
                                 first kind (= sum K[n] cos(n phi)). K is a list/array; K[0] is n=0.

The force is defined per window and then summed onto the beads, matching how the integrator sees it
(each bead's total force is the sum over all windows it participates in). per-WINDOW isolated force
is available separately (see window_max_force).

Run: python scripts/force_reference.py
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
import torusfold.scheme2.torch_cgsim as C   # noqa: E402  (K_DIH, DIH_PPPP -- read-only)

KBT = float(B.KBT)                     # 2.494 kJ/mol at 300 K
K_DIH = float(C.K_DIH)                 # 7.2
COS_DIH = float(math.cos(C.DIH_PPPP))  # 0.975
FORCE_CAP = 5000.0


# ------------------------------------------------------------------ table (loaded once)
_TABLE = None
_ANGLE_TABLE = None


def dihedral_table():
    """The stored dihedral table as torch tensors {lo, binw, U}. U[i] sits at bin centre i."""
    global _TABLE
    if _TABLE is None:
        z = np.load(REPO / "results" / "boltzmann_tables_clean.npz")
        _TABLE = {
            "lo": float(z["dihedral__lo"]),
            "binw": float(z["dihedral__binw"]),
            "U": torch.tensor(z["dihedral__U"], dtype=torch.float64),
        }
    return _TABLE


def angle_table():
    """The stored angle table, same shape of record as dihedral_table().

    The support is NOT the whole cosine range: hi = 0.7998, so q above that belongs to the wall
    term rather than to the table (see boltzmann_bonded.mixed_energy, which adds
    slope_hi*d + 0.5*k_wall*d^2 there for the same reason). A probe or a finite difference that
    walks past hi measures the wall's kink, not the potential this table describes.
    """
    global _ANGLE_TABLE
    if _ANGLE_TABLE is None:
        z = np.load(REPO / "results" / "boltzmann_tables_clean.npz")
        _ANGLE_TABLE = {
            "lo": float(z["angle__lo"]),
            "binw": float(z["angle__binw"]),
            "U": torch.tensor(z["angle__U"], dtype=torch.float64),
        }
    return _ANGLE_TABLE


def _interp(q, t):
    """Linear interpolation of the table, matching boltzmann_bonded._sample (bin centres)."""
    u = (q - t["lo"]) / t["binw"] - 0.5
    i0 = torch.floor(u).long().clamp(0, len(t["U"]) - 2)
    f = (u - i0.double()).clamp(0.0, 1.0)
    return t["U"][i0] + f * (t["U"][i0 + 1] - t["U"][i0])


def _T(q, n):
    if n == 0:
        return torch.ones_like(q)
    if n == 1:
        return q
    t0, t1 = torch.ones_like(q), q
    for _ in range(2, n + 1):
        t0, t1 = t1, 2.0 * q * t1 - t0
    return t1


def _v_fn(spec, coord="dihedral"):
    """Return V(q) -> (...,) energy callable for the given spec. q is a torch tensor.

    coord selects which table "table" and ("table_jac", eps) resolve to, and it REFUSES
    "shipped_harmonic" for the angle rather than guessing at it: the shipped constant is
    K_ANGLE=28.1, but every production caller passes relax_angle_k=200.0 (isrnaclong.py:1950,
    :2087) -- a factor of 7 -- so "the shipped angle harmonic" does not name one potential.
    ("harmonic", K, q0) makes the choice visible at the call site instead of inheriting it, and
    it is the spec an angle comparison must use for the field to be reproducible.

    The measure term in ("table_jac", eps) is the torsion Jacobian; it is written in terms of q
    alone, so the expression is the same for either coordinate. Whether it is the RIGHT
    correction for the angle is a separate question -- the angle's measure is not the torsion's
    -- so callers that know which coordinate they mean should name their spec rather than rely
    on a default in here.
    """
    tbl = dihedral_table if coord == "dihedral" else angle_table
    if spec == "shipped_harmonic":
        if coord != "dihedral":
            raise ValueError(
                f"shipped_harmonic is not defined for {coord!r}: K_ANGLE=28.1 and the production "
                f"relax_angle_k=200.0 are 7x apart. Use ('harmonic', K, q0) and say which.")
        return lambda q: 0.5 * K_DIH * (q - COS_DIH) ** 2
    if isinstance(spec, tuple) and spec[0] == "harmonic":
        k, q0 = float(spec[1]), float(spec[2])
        return lambda q: 0.5 * k * (q - q0) ** 2
    if spec == "table":
        t = tbl()
        return lambda q: _interp(q, t)
    if isinstance(spec, tuple) and spec[0] == "table_jac":
        t = tbl()
        eps = spec[1]
        if eps is None:
            return lambda q: _interp(q, t) - 0.5 * KBT * torch.log(torch.clamp(1.0 - q * q, min=1e-8))
        e2 = eps * eps
        return lambda q: _interp(q, t) - 0.5 * KBT * torch.log(torch.clamp(1.0 - q * q, min=e2))
    if isinstance(spec, tuple) and spec[0] == "fourier":
        K = [torch.tensor(float(x), dtype=torch.float64) for x in spec[1]]
        return lambda q: sum(K[n] * _T(q, n) for n in range(len(K)))
    raise ValueError(f"unknown spec {spec!r}")


# ------------------------------------------------------------------ the reference function
def _q_dihedral(p):
    """q = cos(phi) for a (B, M, 4, 3) stack of 4-atom windows, same convention as _dihedral_f."""
    p0, p1, p2, p3 = p[..., 0, :], p[..., 1, :], p[..., 2, :], p[..., 3, :]
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n0 = torch.linalg.cross(b0, b1)
    n1 = torch.linalg.cross(b1, b2)
    n0 = n0 / torch.clamp(torch.linalg.norm(n0, dim=-1, keepdim=True), min=1e-6)
    n1 = n1 / torch.clamp(torch.linalg.norm(n1, dim=-1, keepdim=True), min=1e-6)
    return (n0 * n1).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)


def _q_angle(p):
    """q = cos(theta) for a (B, M, 3, 3) stack of 3-atom windows, same convention as _angle_f.

    _angle_f builds v1 = p0 - p1 and v2 = p2 - p1 -- both pointing AWAY from the central atom --
    and takes their normalised dot product, clamped to +-1-1e-6. Reproduced exactly here rather
    than re-derived: a v1 = p1 - p0 convention would give the same cosine but flip the sign of
    dcos/dx, and that sign is what scripts/diagnose_angle_gradient.py measured at 1.927 relative
    error in the previous hand-derived force.
    """
    p0, p1, p2 = p[..., 0, :], p[..., 1, :], p[..., 2, :]
    v1, v2 = p0 - p1, p2 - p1
    n1 = torch.clamp(torch.linalg.norm(v1, dim=-1, keepdim=True), min=1e-6)
    n2 = torch.clamp(torch.linalg.norm(v2, dim=-1, keepdim=True), min=1e-6)
    return ((v1 * v2).sum(-1, keepdim=True) / (n1 * n2)).squeeze(-1).clamp(-1 + 1e-6, 1 - 1e-6)


def dihedral_force(spec, pos, ij_like=None):
    """Reference dihedral energy and force. See module docstring for the contract."""
    B_, N, _ = pos.shape
    dev = pos.device
    if ij_like is None:
        L = N // 3
        if L < 4:
            return torch.zeros(B_, device=dev), torch.zeros_like(pos)
        a = torch.arange(L - 3, device=dev)
        ij = torch.stack([3 * a, 3 * a + 3, 3 * a + 6, 3 * a + 9], dim=-1)   # (M, 4)
    else:
        ij = ij_like
    if ij.shape[0] == 0:
        return torch.zeros(B_, device=dev), torch.zeros_like(pos)

    # differentiable copy: the energy and the force below come from THIS graph, nothing detached.
    #
    # enable_grad is load-bearing, not decoration. The sampler evaluates forces inside
    # torch.no_grad() (ibi_round0.py's loop), and inside that scope requires_grad_(True) on a
    # fresh tensor does NOT switch tracking back on -- backward() then raises "element 0 of
    # tensors does not require grad". Without this block the function works from main() and fails
    # from the one caller it was written for. torch_cgsim._dihedral_f carries the same block at
    # line 1094 for the same reason.
    with torch.enable_grad():
        pos_g = pos.detach().clone().requires_grad_(True)
        p = pos_g[:, ij]                                 # (B, M, 4, 3)
        q = _q_dihedral(p)                               # (B, M)
        V = _v_fn(spec)(q)                               # (B, M)
        E = V.sum(dim=-1)                                # (B,)
        E.sum().backward()
    F = -pos_g.grad                                     # (B, N, 3)
    return E.detach(), F


def angle_force(spec, pos, ij_like=None):
    """Reference angle energy and force. Same contract and construction as dihedral_force.

    A sibling rather than a shared body: dihedral_force's printed numbers are cited throughout
    docs/dihedral_table_decision.md, and one body parameterised by coordinate would let a change
    made for the angle move them. The two differ only in the window (3 atoms vs 4), the
    coordinate function, and the default coordinate for the table specs.

    ij_like: None -> consecutive P atoms (3i, 3i+3, 3i+6 for i = 0..L-3), the L-2 windows a
    linear backbone carries. Note the angle's table support stops at hi = 0.7998, so q above it
    is outside what this spec describes.
    """
    B_, N, _ = pos.shape
    dev = pos.device
    if ij_like is None:
        L = N // 3
        if L < 3:
            return torch.zeros(B_, device=dev), torch.zeros_like(pos)
        a = torch.arange(L - 2, device=dev)
        ij = torch.stack([3 * a, 3 * a + 3, 3 * a + 6], dim=-1)   # (M, 3)
    else:
        ij = ij_like
    if ij.shape[0] == 0:
        return torch.zeros(B_, device=dev), torch.zeros_like(pos)

    # enable_grad for the same reason dihedral_force carries it: the sampler runs inside
    # torch.no_grad(), where requires_grad_(True) does not re-enable tracking and backward()
    # raises. See the longer note in dihedral_force.
    with torch.enable_grad():
        pos_g = pos.detach().clone().requires_grad_(True)
        p = pos_g[:, ij]                                 # (B, M, 3, 3)
        q = _q_angle(p)                                  # (B, M)
        V = _v_fn(spec, "angle")(q)                      # (B, M)
        E = V.sum(dim=-1)                                # (B,)
        E.sum().backward()
    F = -pos_g.grad                                     # (B, N, 3)
    return E.detach(), F


def window_max_force(spec, pos, ij_like=None):
    """Per-WINDOW isolated max per-bead |F| (each window differentiated on its own 4 atoms,
    not summed). Returns (q_values (M,), per_window_max_force (M,))."""
    B_, N, _ = pos.shape
    dev = pos.device
    if ij_like is None:
        L = N // 3
        a = torch.arange(L - 3, device=dev)
        ij = torch.stack([3 * a, 3 * a + 3, 3 * a + 6, 3 * a + 9], dim=-1)
    else:
        ij = ij_like
    qs, fmax = [], []
    for row in ij:
        p = pos[:, row].detach().clone().requires_grad_(True)
        q = _q_dihedral(p)                               # (B, 1)
        E = _v_fn(spec)(q).sum()
        E.backward()
        F = -p.grad                                      # (B, 1, 4, 3)
        qs.append(q.detach().item())
        fmax.append(float(F.norm(dim=-1).max().item()))
    return np.array(qs), np.array(fmax)


# ------------------------------------------------------------------ Fourier fit (deterministic)
def cheb(q, n):
    q = np.asarray(q, dtype=float)
    if n == 0:
        return np.ones_like(q)
    if n == 1:
        return q
    t0, t1 = np.ones_like(q), q
    for _ in range(2, n + 1):
        t0, t1 = t1, 2.0 * q * t1 - t0
    return t1


def fourier_fit(N):
    """Weighted least squares of V(phi)=sum_{n=0..N} K_n cos(n phi) to the flat-phi reference
    potential U_ref(q) - kBT ln(sin phi), weight = reference bin mass. Deterministic. Returns K
    (length N+1) usable as ("fourier", K)."""
    z = np.load(REPO / "results" / "boltzmann_tables_clean.npz")
    c = z["dihedral__centre"]
    U = z["dihedral__U"]
    m = np.exp(-U / KBT)
    mask = (c >= -1.0) & (c <= 1.0)
    c, m = c[mask], m[mask]
    m = m / m.sum()
    q = c
    sinphi = np.sqrt(np.clip(1.0 - q * q, 1e-12, None))
    U_phi = -KBT * np.log(np.maximum(m, 1e-12)) - KBT * np.log(sinphi)
    w = m.copy()
    A = np.column_stack([np.sqrt(w) * cheb(q, n) for n in range(0, N + 1)])
    y = np.sqrt(w) * U_phi
    K, *_ = np.linalg.lstsq(A, y, rcond=None)
    return K


# ------------------------------------------------------------------ samples
def sample_chains(which="8chains"):
    """Explicit geometry collections. '8chains' = load_structures(limit=8) (does NOT contain
    2OIU); '2OIU' = the single 2OIU chain (has a q=0.9999 dihedral)."""
    structs = B.load_structures()
    if which == "2OIU":
        return [s for s in structs if s["name"] == "2OIU"]
    return structs[:8]


def chain_tensor(s):
    """(1, 3L, 3) float64 bead tensor for a struct record."""
    return torch.tensor(s["pos"], dtype=torch.float64).reshape(1, -1, 3)


def q_distribution(structs):
    """min/max q and counts of |q|>0.99, |q|>0.999 over all dihedral windows of the chains."""
    qs = []
    for s in structs:
        pos = chain_tensor(s)
        ij = None
        q, _ = window_max_force("shipped_harmonic", pos)   # q does not depend on the spec
        qs.append(q)
    q = np.concatenate(qs)
    return {"n": len(q), "min": q.min(), "max": q.max(),
            "gt099": int((np.abs(q) > 0.99).sum()),
            "gt0999": int((np.abs(q) > 0.999).sum())}


def chain_summed_force(spec, structs):
    """Per-bead |F| over every bead of every chain, with the dihedral force SUMMED over the
    chain's windows (what the integrator / mixed_energy sees). Returns (beads, 1) array."""
    vals = []
    for s in structs:
        pos = chain_tensor(s)
        E, F = dihedral_force(spec, pos)
        vals.append(F.reshape(-1, 3).norm(dim=-1).numpy())
    return np.concatenate(vals)


def window_isolated_force(spec, structs):
    """Per-WINDOW isolated max per-bead |F| (each window on its own), the definition my earlier
    519 number used."""
    qs, fmax = [], []
    for s in structs:
        pos = chain_tensor(s)
        q, f = window_max_force(spec, pos)
        qs.append(q)
        fmax.append(f)
    return np.concatenate(qs), np.concatenate(fmax)


def report(forces):
    if len(forces) == 0:
        return {"median": float("nan"), "p99": float("nan"), "max": float("nan"),
                "gt5000": 0}
    return {"median": float(np.median(forces)), "p99": float(np.percentile(forces, 99)),
            "max": float(forces.max()), "gt5000": int((forces > FORCE_CAP).sum())}


# ------------------------------------------------------------------ main
def main():
    print(f"KBT = {KBT}, K_DIH = {K_DIH}, cos(DIH_PPPP) = {COS_DIH}, force_cap = {FORCE_CAP}")
    print()

    samples = {"8chains": sample_chains("8chains"), "2OIU": sample_chains("2OIU")}
    for name, ss in samples.items():
        qd = q_distribution(ss)
        print(f"sample {name}: {len(ss)} chain(s), {qd['n']} windows, "
              f"q in [{qd['min']:.4f}, {qd['max']:.4f}], "
              f"|q|>0.99: {qd['gt099']}, |q|>0.999: {qd['gt0999']}")
    print()

    specs = [
        ("shipped_harmonic", "shipped_harmonic"),
        ("table", "table"),
        ("exact_b2", ("table_jac", None)),
        ("reg eps=3deg", ("table_jac", math.sin(math.radians(3.0)))),
        ("fourier N=2", ("fourier", fourier_fit(2))),
        ("fourier N=4", ("fourier", fourier_fit(4))),
    ]

    print("=== chain-summed per-bead |F| (what the integrator caps) ===")
    print(f"  {'spec':18s} {'sample':8s} {'median':>9s} {'p99':>9s} {'max':>9s} {'>5000':>6s}")
    for sname, ss in samples.items():
        for label, spec in specs:
            r = report(chain_summed_force(spec, ss))
            print(f"  {label:18s} {sname:8s} {r['median']:9.2f} {r['p99']:9.2f} "
                  f"{r['max']:9.2f} {r['gt5000']:6d}")
    print()

    print("=== per-WINDOW isolated max per-bead |F| (my earlier 519 used this) ===")
    print(f"  {'spec':18s} {'sample':8s} {'median':>9s} {'p99':>9s} {'max':>9s} {'>5000':>6s}")
    for sname, ss in samples.items():
        for label, spec in specs:
            _, f = window_isolated_force(spec, ss)
            r = report(f)
            print(f"  {label:18s} {sname:8s} {r['median']:9.2f} {r['p99']:9.2f} "
                  f"{r['max']:9.2f} {r['gt5000']:6d}")
    print()

    # ---- geometric factor max|dq/dx| on the 2OIU q=0.9999 window (diagnostic)
    print("=== 2OIU window with |q|>0.999: q, and the force at that window ===")
    s2 = sample_chains("2OIU")[0]
    pos = chain_tensor(s2)
    q, f = window_max_force("table", pos)
    big = np.where(np.abs(q) > 0.999)[0]
    for i in big:
        # max|dq/dx| = force of V(q)=q, i.e. spec that returns E=q
        L = s2["pos"].shape[0]
        a = torch.arange(L - 3)
        ij = torch.stack([3 * a, 3 * a + 3, 3 * a + 6, 3 * a + 9], dim=-1)
        row = ij[i]
        p = pos[:, row].detach().clone().requires_grad_(True)
        qq = _q_dihedral(p)
        qq.sum().backward()
        dqdx = float((-p.grad).norm(dim=-1).max().item())
        # table dV/dq at that q (numeric, piecewise-constant derivative)
        dU = (dihedral_table()["U"][1:] - dihedral_table()["U"][:-1]) / dihedral_table()["binw"]
        u = (qq.item() - dihedral_table()["lo"]) / dihedral_table()["binw"] - 0.5
        i0 = int(np.clip(np.floor(u), 0, len(dU) - 1))
        dv = float(dU[i0].item())
        jac = KBT * qq.item() / (1 - qq.item() ** 2)
        print(f"  window {i}: q={q[i]:.6f}, max|dq/dx|={dqdx:.3f} /nm, "
              f"table dV/dq={dv:.1f}, jac dV/dq={jac:.1f}")
        print(f"      -> table force ~ |{dv}|*{dqdx:.3f} = {abs(dv)*dqdx:.1f}; "
              f"exact_b2 force ~ |{dv+jac}|*{dqdx:.3f} = {abs(dv+jac)*dqdx:.1f}")
    print()

    # ---- regularization symmetry check
    print("=== regularization symmetry: dV/dq of the Jacobian term, exact vs reg(3deg) ===")
    for qq in (0.999, -0.999, 0.99, -0.99):
        q = torch.tensor(qq, dtype=torch.float64, requires_grad=True)
        E = -0.5 * KBT * torch.log(torch.clamp(1.0 - q * q, min=1e-8))
        E.backward()
        gex = q.grad.item()
        q2 = torch.tensor(qq, dtype=torch.float64, requires_grad=True)
        E2 = -0.5 * KBT * torch.log(torch.clamp(1.0 - q2 * q2, min=math.sin(math.radians(3.0)) ** 2))
        E2.backward()
        greg = q2.grad.item()
        print(f"  q={qq:+.3f}: exact dV/dq={gex:10.1f}, reg(3deg) dV/dq={greg:10.1f}")
    print("  (reg floors 1-q^2 symmetrically in q^2; both sides zero identically for |q|>sqrt(1-eps^2))")


if __name__ == "__main__":
    main()
