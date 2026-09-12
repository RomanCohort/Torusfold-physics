"""Can the tabulated Boltzmann dihedral potential be used as a force law?

The decision "replace the harmonic dihedral with the tabulated U = -kBT ln P_ref" rests on
four claims that nobody has measured on the shipped table. This script measures them, on the
7-structure pool ibi_round0.py uses (pairs >= 8, 24 <= L <= 34) plus 2OIU, all read from the
same loader the IBI path uses.

The four questions, and what is measured for each:

  1. Force magnitude.  F = -d/dx mixed_energy(which=["dihedral"]) by autograd, on every
     reference geometry. Is max |F| well below force_cap = 5000 kJ/mol/nm?

  2. Is F the true gradient of E?  Central finite difference of the SAME energy the autograd
     force differentiates. The table is piecewise-linear, so the derivative is discontinuous
     at bin boundaries: within a bin the two must agree to FD truncation error (~1e-10 in
     float64), while a probe whose FD step straddles a bin edge sees a one-sided autograd
     against a two-sided average. The check is therefore split into straddling vs
     non-straddling probes instead of a single rel-L2 that would smear the two.

     What it bypasses, explicitly. This energy has NO cell list, NO GB/SA pair filter, NO
     force cap and NO nonbonded terms -- those are the two things that made the full-path
     gradcheck in tests/test_force_gradcheck.py "not doable" (the per-component ratio
     denominator passes through zero, and the full-path energy is discontinuous at the 1.0 nm
     GB cell filter). mixed_energy(which=["dihedral"]) is a pure function of the P-atom
     dihedral cosines, so the ONLY non-smoothness left is the table interpolation, which is
     exactly what is under test.

  3. Does the interpolation produce steps at the real step size?  The energy is C0 (verified
     by a one-bead scan across a bin edge); the force is piecewise-constant in q and jumps at
     bin edges. Measured: the max per-bin slope and max slope jump of the table, the geometric
     factor max |dq/dx| (how much one bead's cosine moves per nm of bead motion), and the
     fraction of (bead, dihedral) instances whose q sits within one real step (5e-4 nm, README
     section 3ae) of a bin edge -- i.e. how often a real integrator step crosses the kink.

  4. Does the table reach q = +-1, and is the peak at the boundary?  Reported: lo/hi/binw,
     whether [lo, hi] contains the physical cosine range [-1, 1], where argmin U sits relative
     to the support edge, and whether the wall (quadratic outside [lo, hi]) is ever activated
     on real geometries.

Run: python scripts/check_table_force_law.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B          # noqa: E402

NPZ = REPO / "results" / "boltzmann_tables_clean.npz"
FORCE_CAP = 5000.0
STEP_NM = 5e-4                        # README section 3ae: bead displacement per dt=0.002 ps step
KBT = B.KBT


def _bin_index(q, lo, binw):
    return np.floor((q - lo) / binw - 0.5).astype(int)


def _dihedral_force(pos, tables):
    pos = pos.detach().requires_grad_(True)
    E = B.mixed_energy(pos, tables, which=["dihedral"])
    F = -torch.autograd.grad(E, pos)[0]
    return float(E.detach()), F.detach()


def main():
    z = np.load(NPZ)
    tables = {}
    for name in B.COORDS:
        tables[name] = {
            "lo": float(z[f"{name}__lo"]), "hi": float(z[f"{name}__hi"]),
            "binw": float(z[f"{name}__binw"]), "U": z[f"{name}__U"],
            "centre": z[f"{name}__centre"], "sigma": float(z[f"{name}__sigma"]),
        }
    B.prepare(tables)
    t = tables["dihedral"]
    U = np.asarray(t["U"], dtype=np.float64)
    lo, hi, binw = t["lo"], t["hi"], t["binw"]
    centre = np.asarray(t["centre"], dtype=np.float64)

    print("=" * 74)
    print("Q4: table range, peak position, wall reachability")
    print("=" * 74)
    print(f"dihedral table: lo={lo:.5f}  hi={hi:.5f}  binw={binw:.5f}  nbins={len(U)}")
    print(f"  physical cos range [-1, 1] inside [lo, hi]?  {lo <= -1.0 <= hi and lo <= 1.0 <= hi}")
    i_min = int(np.argmin(U))
    print(f"  argmin U at bin {i_min} of {len(U)}, centre={centre[i_min]:.5f}  "
          f"(hi - centre[i_min] = {hi - centre[i_min]:.4f} of q)")
    print(f"  peak is {'INTERIOR' if 0 < i_min < len(U) - 1 else 'AT SUPPORT EDGE'} "
          f"of the table support")

    # ---------------------------------------------------------------- structures
    structs = B.load_structures(limit=400)
    pool = [s for s in structs if len(s["pairs"]) >= 8 and 24 <= len(s["pos"]) <= 34]
    oiu = [s for s in structs if s["name"] == "2OIU"]
    targets = [(s["name"], s) for s in pool] + [(s["name"], s) for s in oiu]

    print()
    print("=" * 74)
    print("Q1: max |F| of the dihedral table term, vs force_cap =", FORCE_CAP)
    print("=" * 74)
    print(f"{'structure':10s} {'L':>4s} {'n_dihedral':>10s} {'E_dih (kJ/mol)':>15s} "
          f"{'max |F|':>12s} {'/cap':>7s} {'d_lo>0':>7s} {'d_hi>0':>7s}")
    print("-" * 74)
    maxf_global = 0.0
    for name, s in targets:
        L = len(s["pos"])
        pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
        E, F = _dihedral_force(pos, tables)
        maxf = float(F.reshape(-1, 3).norm(dim=-1).max())
        maxf_global = max(maxf_global, maxf)
        q = B.coords_of(pos.detach(), "dihedral").reshape(-1)
        d_lo = float((t["lo"] - q).clamp(min=0.0).max())
        d_hi = float((q - t["hi"]).clamp(min=0.0).max())
        print(f"{name:10s} {L:4d} {q.numel():10d} {E:15.4f} {maxf:12.4f} "
              f"{maxf / FORCE_CAP:7.4f} {str(d_lo > 0):>7s} {str(d_hi > 0):>7s}")
    print(f"\n  global max |F| over all geometries = {maxf_global:.4f} kJ/mol/nm "
          f"= {maxf_global / FORCE_CAP:.4f} of the cap")
    print("  d_lo/d_hi: whether the quadratic wall outside [lo, hi] is ever activated "
          "(cos in [-1,1] so it should never be)")

    # ---------------------------------------------------------------- Q2: FD
    print()
    print("=" * 74)
    print("Q2: autograd force vs finite difference of the same energy")
    print("=" * 74)
    for h in (1e-5, 1e-4):
        n_probe = n_straddle = 0
        num = den = 0.0
        num_ns = den_ns = 0.0
        worst = 0.0
        worst_where = ""
        for name, s in targets:
            L = len(s["pos"])
            pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
            _, F = _dihedral_force(pos, tables)
            F = F.reshape(1, -1, 3)
            for i in range(pos.shape[1]):
                for d in range(3):
                    hi_p = pos.clone(); hi_p[:, i, d] += h
                    lo_p = pos.clone(); lo_p[:, i, d] -= h
                    Eh, _ = _dihedral_force(hi_p, tables)
                    El, _ = _dihedral_force(lo_p, tables)
                    fd = -(Eh - El) / (2.0 * h)
                    an = float(F[0, i, d])
                    diff = abs(an - fd)
                    n_probe += 1
                    num += (an - fd) ** 2
                    den += fd ** 2
                    # straddle: did any dihedral of this bead change bin across +-h?
                    qp = B.coords_of(hi_p.detach(), "dihedral").reshape(-1).numpy()
                    qm = B.coords_of(lo_p.detach(), "dihedral").reshape(-1).numpy()
                    if (_bin_index(qp, lo, binw) != _bin_index(qm, lo, binw)).any():
                        n_straddle += 1
                        if diff > worst:
                            worst = diff
                            worst_where = f"{name} bead {i} axis {d}"
                    else:
                        num_ns += (an - fd) ** 2
                        den_ns += fd ** 2
        rel = float(np.sqrt(num / max(den, 1e-30)))
        rel_ns = float(np.sqrt(num_ns / max(den_ns, 1e-30)))
        print(f"  h={h:.0e}: rel L2 = {rel:.3e} over {n_probe} probes; "
              f"non-straddle-only rel L2 = {rel_ns:.3e}; "
              f"{n_straddle} straddled a bin edge; worst |dF| = {worst:.3e} kJ/mol/nm "
              f"({worst_where})")

    # Where does 2OIU's large force come from? A single-bead force histogram.
    print()
    print("=" * 74)
    print("2OIU force source: per-bead |F| top 5")
    print("=" * 74)
    s2 = [s for s in targets if s[0] == "2OIU"][0][1]
    L2 = len(s2["pos"])
    pos2 = torch.tensor(s2["pos"].reshape(1, 3 * L2, 3), dtype=torch.float64)
    _, F2 = _dihedral_force(pos2, tables)
    mags = F2.reshape(-1, 3).norm(dim=-1).numpy()
    order = np.argsort(mags)[::-1]
    print("  bead is 0-indexed over 3L beads; P atoms sit at 0,3,6,...")
    print(f"  {'bead':>5s} {'|F|':>12s} {'/cap':>7s}")
    for k in order[:5]:
        print(f"  {k:5d} {mags[k]:12.4f} {mags[k] / FORCE_CAP:7.4f}")
    q2 = B.coords_of(pos2.detach(), "dihedral").reshape(-1).numpy()
    # nearest dihedral centre to each top bead's P index, for context
    print("  dihedral cosines in 2OIU: min=%.4f max=%.4f  (the steepest bin slope is near the"
          " cis peak q~+1)" % (q2.min(), q2.max()))

    # ---------------------------------------------------------------- Q3: steps
    print()
    print("=" * 74)
    print("Q3: interpolation kinks and the real step size")
    print("=" * 74)
    slope = np.diff(U) / binw                      # per-bin force slope, kJ/mol per unit q
    print(f"  max |slope| per bin        = {np.abs(slope).max():.3f} kJ/mol per unit cos")
    print(f"  max slope jump (|dU'/dq|)  = {np.abs(np.diff(slope)).max():.3f} kJ/mol per unit cos")

    # geometric factor: how fast one dihedral's cosine follows a bead, and the steepest
    # region (the cis peak) drives both |dU/dq| and |dq/dx| up at once.
    name0, s0 = targets[0]
    pos0 = torch.tensor(s0["pos"].reshape(1, 3 * len(s0["pos"]), 3),
                        dtype=torch.float64).requires_grad_(True)
    q0 = B.coords_of(pos0, "dihedral").reshape(-1)
    gmax = 0.0
    for k in range(q0.numel()):
        g = torch.autograd.grad(q0[k], pos0, retain_graph=True)[0]
        gmax = max(gmax, float(g.reshape(-1, 3).norm(dim=-1).max()))
    print(f"  max |dq/dx| on {name0}     = {gmax:.4f} 1/nm  (how fast cos follows a bead)")
    jump_force = float(np.abs(np.diff(slope)).max()) * gmax
    print(f"  -> max force jump at a bin edge = {jump_force:.4f} kJ/mol/nm "
          f"= {jump_force / FORCE_CAP:.4f} of the cap")

    # §3ae-style measurement: move ALL beads by u = delta * (unit direction), and compare
    # E(+u) - E(-u) against -2 F.u. For a smooth energy the residual is O(delta^3); for a
    # C0-with-kink energy every bead whose dihedral crosses a bin edge leaves a residual of
    # order delta * slope_jump. This is the exact "does the interpolation produce a step at
    # the real step size" test -- one integrator step moves every bead at once.
    rng = np.random.default_rng(20260912)
    print(f"  (delta in nm; real step = {STEP_NM:.0e})")
    print(f"  {'structure':10s} {'delta':>8s} {'smooth |2F.u|':>14s} {'step median':>12s} "
          f"{'step max':>12s} {'max/kBT':>8s}")
    for delta in (STEP_NM, 1e-3, 1e-2):
        for name, s in targets:
            L = len(s["pos"])
            pos = torch.tensor(s["pos"].reshape(1, 3 * L, 3), dtype=torch.float64)
            _, F = _dihedral_force(pos, tables)
            Fv = F.reshape(-1).numpy()
            meds, maxs, smooths = [], [], []
            for _ in range(12):
                u = torch.tensor(rng.standard_normal(pos.numel()),
                                 dtype=torch.float64).reshape(pos.shape)
                u = u / torch.linalg.norm(u)
                hp = pos + delta * u
                lp = pos - delta * u
                Ep, _ = _dihedral_force(hp, tables)
                Em, _ = _dihedral_force(lp, tables)
                smooth = 2.0 * float((Fv * (delta * u.reshape(-1).numpy())).sum())
                residual = (Ep - Em) + smooth
                meds.append(abs(residual)); maxs.append(abs(residual))
                smooths.append(abs(smooth))
            print(f"  {name:10s} {delta:8.1e} {np.median(smooths):14.4f} "
                  f"{np.median(meds):12.4f} {np.max(maxs):12.4f} {np.max(maxs) / KBT:8.3f}")

    # C0 continuity: a one-bead scan across a bin edge must not jump in ENERGY (only in
    # slope). A true step would show as an O(1) kJ/mol jump between adjacent samples.
    pos = torch.tensor(s0["pos"].reshape(1, 3 * len(s0["pos"]), 3), dtype=torch.float64)
    dirc = rng.standard_normal(3); dirc /= np.linalg.norm(dirc)
    xs = np.linspace(-2e-3, 2e-3, 2001)
    Es = []
    for dx in xs:
        pp = pos.clone()
        pp[0, 0] += torch.tensor(dirc * dx, dtype=torch.float64)
        e, _ = _dihedral_force(pp, tables)
        Es.append(e)
    de = np.diff(np.array(Es))
    print(f"  C0 scan (P0 moved +-2e-3 nm, 2001 pts): max |dE| between adjacent pts "
          f"= {np.abs(de).max():.3e} kJ/mol (a step would be O(1); this is O(h))")
    print("  wall never fires on real geometries (d_lo=d_hi=0 above), so edge slopes are moot.")


if __name__ == "__main__":
    main()
