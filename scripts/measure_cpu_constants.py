"""Measure the CPU-path (OpenMM) 3-bead CG force constants from the rsRNASP training set.

Why this file exists
--------------------
openmm_gpu_refiner.py declares its bonded constants in kJ/mol/angstrom^2 (distance
terms) and kJ/mol/rad^2 (angle, dihedral) and converts the distance ones at use
(bb_k = K_BB * 100.0).  torch_cgsim.py declares the same numerals in kJ/mol/nm^2.
The two files therefore mean different things by "500" and "400", and the CPU file's
functional forms differ as well: OpenMM HarmonicAngleForce acts on the angle in
radians, CustomTorsionForce acts on the torsion in radians, and the CPU stacking
coordinate is N(i)-N(i+1), not P(i)-P(i+2).

So the CPU constants have to be measured for the CPU's own coordinates and units.
Criterion: a harmonic term E = 0.5*k*(q-q0)^2 has thermal width sigma = sqrt(kBT/k),
hence k = kBT/sigma^2 with kBT = 2.494 kJ/mol at 300 K.

Honesty about what the database can do
--------------------------------------
Each structure in the database is ONE conformation.  A pooled sigma over 191 files
mixes residue-to-residue and conformation-to-conformation variation into the same
number as thermal fluctuation, so pooled sigma OVERSTATES the thermal width and
kBT/sigma^2 is a LOWER BOUND on the stiffness, never the stiffness itself.  This
script reports two more numbers next to it, both of which are still bounds:
  * mean within-chain sigma  -- drops the chain-to-chain variation, keeps the
    sequence variation inside one deposited conformation;
  * local sigma at the CPU target -- the spread of the subset of observations within
    +/-0.5 rad of the target, which is the curvature the spring actually feels where
    it sits.

Coordinates measured (all in nm / rad):
  bb_bond    P(i)-P(i+1)          CPU: HarmonicBondForce,  r0 = 5.90 A
  intra_pc   P(i)-C4'(i)          CPU: HarmonicBondForce,  r0 = 3.90 A
  intra_cn   C4'(i)-N(i)          CPU: HarmonicBondForce,  r0 = 3.35 A
  stack_nn   N(i)-N(i+1)          CPU: CustomBondForce,    r0 = 5.05 A
  stack_pp2  P(i)-P(i+2)          GPU: torch_cgsim only    r0 = 1.125 nm
  angle_rad  P(i)-P(i+1)-P(i+2)   CPU: HarmonicAngleForce, theta0 = 150 deg
  dihedral   P(i)..P(i+3)         CPU: CustomTorsionForce, theta0 = +33 deg in
                                  OpenMM's convention

Sign convention: OpenMM's CustomTorsionForce "theta" was measured to be the NEGATED
IUPAC atan2 dihedral (max |theta_omm + iupac| = 8.9e-16 rad over 300 random quads,
Reference platform, OpenMM 8.5.2).  The dihedral below is therefore reported as
-atan2(...), i.e. in the same convention the CPU force uses.  The cpu target then is
DIH_PPPP = +33 deg in that convention.

Run:  python scripts/measure_cpu_constants.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
KBT = 2.494  # kJ/mol at 300 K
import _cgdata
DATA = _cgdata.rsrnasp()
NPZ = REPO / "results" / "boltzmann_tables_clean.npz"

# The CPU's own geometric targets, read from openmm_gpu_refiner.py (Angstrom / deg).
CPU = {
    "bb_bond": 5.90, "intra_pc": 3.90, "intra_cn": 3.35, "stack_nn": 5.05,
    "angle_deg": 150.0, "dihedral_deg": 33.0,
}
LOCAL_WIN = 0.5  # rad, half-window for the local-curvature sigma


def _load_boltzmann_bonded():
    spec = importlib.util.spec_from_file_location(
        "boltzmann_bonded", REPO / "scripts" / "boltzmann_bonded.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def coords_of_chain(beads):
    """beads: (L,3,3) nm.  Returns {name: 1-D np.array} for one chain."""
    P, C4, N = beads[:, 0], beads[:, 1], beads[:, 2]
    out = {}
    out["bb_bond"] = np.linalg.norm(P[1:] - P[:-1], axis=1)
    out["intra_pc"] = np.linalg.norm(C4 - P, axis=1)
    out["intra_cn"] = np.linalg.norm(N - C4, axis=1)
    out["stack_nn"] = np.linalg.norm(N[1:] - N[:-1], axis=1)
    out["stack_pp2"] = np.linalg.norm(P[2:] - P[:-2], axis=1)

    v1, v2 = P[:-2] - P[1:-1], P[2:] - P[1:-1]
    c = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    out["angle_rad"] = np.arccos(np.clip(c, -1.0, 1.0))
    out["angle_cos"] = np.clip(c, -1.0, 1.0)

    b0, b1, b2 = P[1:-2] - P[:-3], P[2:-1] - P[1:-2], P[3:] - P[2:-1]
    n0, n1 = np.cross(b0, b1), np.cross(b1, b2)
    m1 = np.cross(n0, b1 / np.linalg.norm(b1, axis=1)[:, None])
    dih = -np.arctan2((m1 * n1).sum(1), (n0 * n1).sum(1))  # OpenMM convention
    out["dihedral_rad"] = dih
    out["dihedral_cos"] = np.cos(dih)
    return out


def describe(v, nbins=120, local=None):
    lo, hi = float(v.min()), float(v.max())
    counts, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    centre = 0.5 * (edges[:-1] + edges[1:])
    mode = float(centre[int(np.argmin(-np.log(counts + 0.5)))])
    out = {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std()),
           "mode": mode, "min": lo, "max": hi}
    if local is not None:
        sub = v[np.abs(v - local) <= LOCAL_WIN]
        out["n_local"] = int(sub.size)
        out["sd_local"] = float(sub.std()) if sub.size > 1 else float("nan")
    return out


def curvature_k(v, target, window, nbins=240):
    """Curvature of U(q) = -kBT ln p(q) at q = target, quadratic fit over |q-target|<=window.

    Returns (k, n_in_window, linear_slope) with U ~ c0 + c1*d + c2*d^2, d = q - target,
    k = 2*c2.  A non-zero c1 means the CPU's pinned target is off the data mode and the
    spring therefore exerts a systematic torque at native geometry.
    """
    lo, hi = float(v.min()), float(v.max())
    counts, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    centre = 0.5 * (edges[:-1] + edges[1:])
    p = (counts + 0.5) / (counts.sum() + 0.5 * nbins)
    U = -KBT * np.log(p)
    sel = np.abs(centre - target) <= window
    d = centre[sel] - target
    c = np.polyfit(d, U[sel], 2)          # c[0]*d^2 + c[1]*d + c[2]
    return 2.0 * float(c[0]), int(counts[sel].sum()), float(c[1])


def main():
    bb = _load_boltzmann_bonded()
    structs = bb.load_structures()
    # with_names=True gives the same chains; used only to report base composition
    names = []
    for f in sorted(DATA.glob("*.pdb")):
        for _, _, nm in bb._chain_residues(f, with_names=True):
            names.append(nm)
    print(f"database: {DATA}")
    print(f"chains kept by boltzmann_bonded.load_structures(): {len(structs)}")
    print(f"residues in those chains: {sum(len(s['pos']) for s in structs)}")
    print()

    pools = {k: [] for k in coords_of_chain(structs[0]["pos"])}
    per_chain = {k: [] for k in pools}
    for s in structs:
        d = coords_of_chain(s["pos"])
        for k, v in d.items():
            pools[k].append(v)
            if v.size:
                per_chain[k].append(float(v.std()))

    stats = {}
    print("=" * 108)
    print("MEASURED DISTRIBUTIONS  (n = observations, sd = pooled std, wsd = mean within-chain std)")
    print("=" * 108)
    print(f"{'coord':<14}{'n':>7}{'mean':>12}{'sd':>12}{'wsd':>12}{'mode':>12}"
          f"{'min':>10}{'max':>10}")
    local_target = {"angle_rad": np.radians(CPU["angle_deg"]),
                    "dihedral_rad": np.radians(CPU["dihedral_deg"])}
    for k in ("bb_bond", "intra_pc", "intra_cn", "stack_nn", "stack_pp2",
              "angle_rad", "angle_cos", "dihedral_rad", "dihedral_cos"):
        v = np.concatenate(pools[k])
        st = describe(v, local=local_target.get(k))
        st["wsd"] = float(np.mean(per_chain[k]))
        stats[k] = st
        print(f"{k:<14}{st['n']:>7}{st['mean']:>12.5f}{st['sd']:>12.5f}"
              f"{st['wsd']:>12.5f}{st['mode']:>12.5f}{st['min']:>10.4f}{st['max']:>10.4f}")

    # ---- self-check against the shipped npz (same machinery, same subset) ----
    if NPZ.exists():
        z = np.load(NPZ)
        print()
        print("=" * 108)
        print("SELF-CHECK vs results/boltzmann_tables_clean.npz  (|this - npz| should be ~0)")
        print("=" * 108)
        pairs = [("bb_bond", "bb_bond__sigma"), ("intra_pc", "intra_pc__sigma"),
                 ("intra_cn", "intra_cn__sigma"), ("stack_pp2", "stack__sigma"),
                 ("angle_cos", "angle__sigma"), ("dihedral_cos", "dihedral__sigma")]
        for mine, key in pairs:
            zs = float(z[key])
            print(f"  {mine:<12} this={stats[mine]['sd']:.6f}  npz={zs:.6f}  "
                  f"diff={stats[mine]['sd'] - zs:+.2e}")

    # ---- derived constants ----
    print()
    print("=" * 108)
    print("DERIVED CPU CONSTANTS   k = kBT / sigma^2   (kBT = 2.494 kJ/mol)")
    print("=" * 108)
    print("distance terms: k in kJ/mol/nm^2 and in kJ/mol/angstrom^2 (the CPU's declared unit,")
    print("                because the CPU file multiplies the declared value by 100 to get nm^-2)")
    print()
    print(f"{'coord':<12}{'sigma':>11}{'k [kJ/mol/nm^2]':>18}{'k [kJ/mol/A^2]':>16}"
          f"{'target':>9}{'mode':>10}{'k(r0)':>16}")
    dist = [("bb_bond", CPU["bb_bond"], "A"), ("intra_pc", CPU["intra_pc"], "A"),
            ("intra_cn", CPU["intra_cn"], "A"), ("stack_nn", CPU["stack_nn"], "A")]
    for name, r0_A, _ in dist:
        sig_nm, sig_A = stats[name]["sd"], stats[name]["sd"] * 10.0
        k_nm = KBT / sig_nm ** 2
        k_A = KBT / sig_A ** 2
        print(f"{name:<12}{sig_A:>10.4f}A{k_nm:>18.1f}{k_A:>16.4f}"
              f"{r0_A:>9.2f}{stats[name]['mode'] * 10:>10.3f}A{'-':>16}")
    for name, tgt in (("angle_rad", CPU["angle_deg"]), ("dihedral_rad", CPU["dihedral_deg"])):
        sig = stats[name]["sd"]
        k = KBT / sig ** 2
        print(f"{name:<12}{sig:>10.4f}r{k:>18.4f}{'-':>16}{tgt:>8.0f}d"
              f"{np.degrees(stats[name]['mode']):>10.2f}d{k:>16.4f}")

    print()
    print("same table with the conservative alternatives (all three are still BOUNDS, not values):")
    print(f"{'coord':<12}{'k(pooled sd)':>16}{'k(within-chain)':>18}{'k(local +-0.5rad)':>20}")
    for name in ("bb_bond", "intra_pc", "intra_cn", "stack_nn"):
        soff = stats[name]["sd"] * 10.0 if name != "bb_bond" else stats[name]["sd"] * 10.0
        print(f"{name:<12}{KBT/soff**2:>16.4f}{KBT/(stats[name]['wsd']*10)**2:>18.4f}{'-':>20}")
    for name in ("angle_rad", "dihedral_rad"):
        print(f"{name:<12}{KBT/stats[name]['sd']**2:>16.4f}"
              f"{KBT/stats[name]['wsd']**2:>18.4f}{KBT/stats[name]['sd_local']**2:>20.4f}")


    # ---- curvature of the measured free energy at the CPU's own target ----
    print()
    print("=" * 108)
    print("LOCAL CURVATURE of U(q) = -kBT ln p(q) at the CPU's target (quadratic fit)")
    print("=" * 108)
    print(f"{'coord':<14}{'target':>10}{'window':>9}{'n_win':>8}{'slope c1':>12}"
          f"{'k=2c2':>12}")
    for name, tgt_deg in (("angle_rad", CPU["angle_deg"]),
                          ("dihedral_rad", CPU["dihedral_deg"])):
        target = np.radians(tgt_deg)
        for win in (0.35, 0.6):
            k, n_win, c1 = curvature_k(np.concatenate(pools[name]), target, win)
            print(f"{name:<14}{tgt_deg:>9.0f}d{np.degrees(win):>8.1f}d{n_win:>8d}"
                  f"{c1:>12.3f}{k:>12.4f}")

    print()
    print("CPU constants as the file declares them (distance terms in kJ/mol/A^2):")
    print(f"  K_BB       = {KBT/(stats['bb_bond']['sd']*10)**2:.4f}")
    print(f"  K_INTRA_PC = {KBT/(stats['intra_pc']['sd']*10)**2:.4f}")
    print(f"  K_INTRA_CN = {KBT/(stats['intra_cn']['sd']*10)**2:.4f}")
    print(f"  K_ANGLE    = {KBT/stats['angle_rad']['sd']**2:.4f}  kJ/mol/rad^2")
    print(f"  K_DIHEDRAL = {KBT/stats['dihedral_rad']['sd']**2:.4f}  kJ/mol/rad^2")
    print(f"  K_STACK    = {KBT/(stats['stack_nn']['sd']*10)**2:.4f}")
    return stats


if __name__ == "__main__":
    main()
