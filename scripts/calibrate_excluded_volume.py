"""Calibrate the excluded-volume stiffness against the structure database, on a grid.

The range is not swept.  CLASH_SIGMA is the database's own lower edge: over 191 PDB files /
126 gap-free chains the 213732 non-bonded P-P pairs (residue index gap >= 3) have a minimum of
0.3975 nm, with 0 pairs below 0.36 nm and 1 below 0.40 nm.  What is not in the database is the
stiffness, because the old range (0.300 nm) sat below every distance the database contains, so
the old term was identically zero over all of it.

Criterion, fixed before the sweep was run:

  1. sigma = CLASH_SIGMA = the minimum of the non-bonded P-P distribution, recomputed here.
     The potential is exactly zero at and beyond sigma, so it cannot touch a pair the database
     populates.
  2. k = Boltzmann inversion of the two lowest populated bins, the only place the database has
     an opinion.  Per unit shell volume the bins read

         [0.36,0.40)  n0 / V0      [0.40,0.45)  n1 / V1

     with V = 4*pi/3*(b^3 - a^3).  The upper bin is outside sigma, where the potential is
     identically zero, so the deficit in the lower bin is the potential's own Boltzmann weight:
     U(mid0) = kBT * ln((n1/V1)/(n0/V0)), and U(mid0) = k * 0.5*(sigma-mid0)^2*(sigma/mid0)^2
     gives k.  Both bin counts enter as measured, no trend is extrapolated.
  3. The sweep below is the CHECK on that number, not the definition of it.  If the simulated
     distribution reproduces the database's lower edge only at a k far from (2), or at every k
     in the grid, that is the result and it is reported as one.

What each grid point measures is one short full-field run at 300 K (every term shipped, the
same Langevin seed for every k, replicas of the same native structure):

  * the closest non-bonded bead pair, over all beads and over P-P, and the median per-frame
    closest -- reported against the database's own 0.3333 nm (any beads, gap >= 3) and
    0.3975 nm (P-P, gap >= 3);
  * the fraction of replica-frames in which any non-bonded pair is inside sigma, and the
    fraction of pair instances below sigma and below 0.36 nm -- the database has 0 of 213732
    below either;
  * the same bins the database histogram reports, per frame.

Run:
  python scripts/calibrate_excluded_volume.py
  python scripts/calibrate_excluded_volume.py --k 0,1e3,1e4,2e4,1e5 --steps 1000 --nrep 4
  python scripts/calibrate_excluded_volume.py --module <other torch_cgsim.py> --k 500 --range 0.30
"""
import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import boltzmann_bonded as B          # noqa: E402

KBT = 2.494
BIN_EDGES = (0.36, 0.40, 0.45, 0.50, 0.60)
SEED = 20260219
DT_PS = 0.002
MASS_AMU = 110.0
FRICTION = 0.1
DB_ANY_MIN = 0.3333     # nm, closest bead pair of any type, residue gap >= 3
DB_PP_MIN = 0.3975      # nm, closest non-bonded P-P pair, residue gap >= 3


def load_module(path):
    """The force field under test: the repo's, or an explicit other file (for A/B)."""
    if path is None:
        sys.path.insert(0, str(REPO / "src"))
        import torusfold.scheme2.torch_cgsim as C
        return C
    spec = importlib.util.spec_from_file_location("torch_cgsim_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["torch_cgsim_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def database_distribution():
    """Every non-bonded P-P distance with residue index gap >= 3, over the whole database."""
    rows = []
    nchain = 0
    for f in sorted(B.DATA.glob("*.pdb")):
        for beads, _pairs, _names in B._chain_residues(str(f), with_names=True):
            nchain += 1
            P = beads[:, 0, :]
            n = len(P)
            d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
            i, j = np.triu_indices(n)
            keep = np.abs(i - j) >= 3
            rows.append(d[i[keep], j[keep]])
    return np.concatenate(rows), nchain


def criterion_k(sigma, v):
    """The stiffness the two lowest populated bins imply. Returns (k, text of the arithmetic)."""
    a0, a1, a2 = BIN_EDGES[0], BIN_EDGES[1], BIN_EDGES[2]
    n0 = int(((v >= a0) & (v < a1)).sum())
    n1 = int(((v >= a1) & (v < a2)).sum())
    v0 = 4.0 / 3.0 * math.pi * (a1 ** 3 - a0 ** 3)
    v1 = 4.0 / 3.0 * math.pi * (a2 ** 3 - a1 ** 3)
    d0, d1 = n0 / v0, n1 / v1
    du = KBT * math.log(d1 / d0)
    mid = 0.5 * (a0 + a1)
    shape = 0.5 * (sigma - mid) ** 2 * (sigma / mid) ** 2       # U(mid) = k * shape
    lines = [
        "  [%.2f,%.2f)  %5d pairs / %.6f nm^3 = %8.3f nm^-3" % (a0, a1, n0, v0, d0),
        "  [%.2f,%.2f)  %5d pairs / %.6f nm^3 = %8.3f nm^-3" % (a1, a2, n1, v1, d1),
        "  U(%.3f) = kBT * ln(%.3f/%.3f) = %s * %.4f = %.4f kJ/mol"
        % (mid, d1, d0, KBT, math.log(d1 / d0), du),
        "  U(%.3f) = k * 0.5*(sigma-%.3f)^2*(sigma/%.3f)^2 = k * %.6e" % (mid, mid, mid, shape),
        "  k = %.4f / %.6e = %.4f kJ/mol/nm^2" % (du, shape, du / shape),
    ]
    return du / shape, lines


def pair_index(L):
    """(i, j, is_pp) for every bead pair with residue index gap >= 3."""
    n = 3 * L
    i, j = torch.triu_indices(n, n, offset=1)
    keep = ((i // 3) - (j // 3)).abs() >= 3
    i, j = i[keep], j[keep]
    return i, j, (i % 3 == 0) & (j % 3 == 0)


def measure(pos, i, j, is_pp, sigma):
    """Non-bonded pair statistics for one frame of every replica."""
    nrep = pos.shape[0]
    flat = pos.reshape(nrep, -1, 3)
    d = torch.linalg.norm(flat[:, i] - flat[:, j], dim=-1)        # (nrep, M)
    dpp = d[:, is_pp]
    per_frame_min = d.min(dim=-1).values
    below_sigma = (d < sigma)
    return {
        "closest": float(d.min()),
        "closest_pp": float(dpp.min()),
        "frame_min_median": float(per_frame_min.median()),
        "frames_below_sigma": int(below_sigma.any(dim=-1).sum()),
        "instances_below_sigma": int(below_sigma.sum()),
        "instances": int(d.numel()),
        "frames_below_036": int((d < 0.36).any(dim=-1).sum()),
        "nbins": [int(((d >= a) & (d < b)).sum())
                  for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:])],
    }


def run_one(C, struct, k, nrep, steps, stride, cap, sigma, dt_ps=DT_PS):
    """One short full-field 300 K run at stiffness k, sampled every stride steps."""
    L = len(struct["pos"])
    pos = torch.tensor(struct["pos"].reshape(1, 3 * L, 3), dtype=torch.float64).repeat(nrep, 1, 1)
    vel = torch.zeros_like(pos)
    ij = torch.tensor(struct["pairs"], dtype=torch.long).reshape(-1, 2)
    pw = torch.ones(len(ij), dtype=torch.float32)
    temps = torch.full((nrep,), 300.0, dtype=torch.float64)
    i, j, is_pp = pair_index(L)
    torch.manual_seed(SEED)
    C.K_CLASH = float(k)

    def forces_at(p):
        cl2 = C.GPUCellList(cell_size=1.5)
        cl2.build(p)
        return C.cg_energy_forces(p, ij, pw, cell_list=cl2, force_cap=cap)[1]

    frames = []
    for step in range(steps):
        with torch.no_grad():
            cl = C.GPUCellList(cell_size=1.5)
            cl.build(pos)
            _e, f = C.cg_energy_forces(pos, ij, pw, cell_list=cl, force_cap=cap)
            pos, vel = C.batch_langevin_step(pos, vel, f, temps, dt_ps=dt_ps,
                                             mass_amu=MASS_AMU, friction=FRICTION,
                                             force_fn=forces_at)
        if step % stride == 0 or step == steps - 1:
            with torch.no_grad():
                if not bool(torch.isfinite(pos).all()):
                    frames.append(None)
                    continue
                frames.append(measure(pos, i, j, is_pp, sigma))
    return frames


def summarise(frames, nrep):
    ok = [f for f in frames if f is not None]
    out = {
        "sampled": len(frames),
        "non_finite": len(frames) - len(ok),
        "replica_frames": len(ok) * nrep,
    }
    if not ok:
        return out
    out["closest"] = min(f["closest"] for f in ok)
    out["closest_pp"] = min(f["closest_pp"] for f in ok)
    out["frame_min_median"] = float(np.median([f["frame_min_median"] for f in ok]))
    out["frames_below_sigma"] = sum(f["frames_below_sigma"] for f in ok)
    out["instances_below_sigma"] = sum(f["instances_below_sigma"] for f in ok)
    out["instances"] = sum(f["instances"] for f in ok)
    out["frames_below_036"] = sum(f["frames_below_036"] for f in ok)
    out["nbins"] = [sum(f["nbins"][b] for f in ok) for b in range(len(BIN_EDGES) - 1)]
    out["first"] = ok[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="0,1e3,3e3,1e4,2e4,4e4,1e5",
                    help="stiffness grid, kJ/mol/nm^2 (default: log grid around the criterion)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--nrep", type=int, default=8)
    ap.add_argument("--structs", type=int, default=1)
    ap.add_argument("--min-pairs", type=int, default=8)
    ap.add_argument("--lmin", type=int, default=24)
    ap.add_argument("--lmax", type=int, default=34)
    ap.add_argument("--cap", default="200", help="force cap, or none")
    ap.add_argument("--dt", type=float, default=DT_PS,
                    help="timestep in ps (the shipped cg_energy_forces cap, not the "
                         "potential, sets the closest approach while it binds; --cap none "
                         "is what shows the potential and --dt is what shows the timestep)")
    ap.add_argument("--module", default=None, help="path to another torch_cgsim.py to measure")
    ap.add_argument("--range", dest="clash_range", type=float, default=None,
                    help="the range this module's potential uses (default: its CLASH_SIGMA)")
    args = ap.parse_args()
    cap = None if str(args.cap).lower() == "none" else float(args.cap)
    dt = float(args.dt)
    ks = [float(x) for x in str(args.k).split(",") if x.strip()]

    C = load_module(args.module)
    sigma = args.clash_range if args.clash_range is not None else float(C.CLASH_SIGMA)

    v, nchain = database_distribution()
    print("=== the database criterion (recomputed here, not quoted) ===")
    print("  source            %s" % B.DATA)
    print("  chains             %d" % nchain)
    print("  non-bonded P-P     %d pairs, residue index gap >= 3" % v.size)
    print("  minimum            %.4f nm" % v.min())
    for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        print("    [%.2f,%.2f)  %6d" % (a, b, int(((v >= a) & (v < b)).sum())))
    print("  below 0.36         %d" % int((v < 0.36).sum()))
    print("  below 0.40         %d" % int((v < 0.40).sum()))
    k_crit, lines = criterion_k(sigma, v)
    print()
    print("=== criterion k, Boltzmann inversion of the two lowest bins (sigma = %s) ===" % sigma)
    for ln in lines:
        print(ln)
    print()

    pool = [s for s in B.load_structures(limit=400)
            if len(s["pairs"]) >= args.min_pairs and args.lmin <= len(s["pos"]) <= args.lmax]
    structs = pool[:args.structs]
    print("=== the runs ===  %d structure(s) x %d replicas x %d steps (%.3f ps) at 300 K, "
          "dt %s ps, mass %s amu, friction %s, force_cap %s"
          % (len(structs), args.nrep, args.steps, args.steps * dt, dt, MASS_AMU,
             FRICTION, cap))
    for s in structs:
        L = len(s["pos"])
        print("  %s  L=%d  wc pairs=%d  non-bonded bead pairs=%d"
              % (s["name"], L, len(s["pairs"]), len(pair_index(L)[0])))
    print()

    results = []
    for k in ks:
        frames = run_one(C, structs[0], k, args.nrep, args.steps, args.stride, cap, sigma, dt)
        for s in structs[1:]:
            frames = frames + run_one(C, s, k, args.nrep, args.steps, args.stride, cap, sigma, dt)
        r = summarise(frames, args.nrep)
        r["k"] = k
        results.append(r)
        print("k = %g   (%+.2f%% from the criterion k = %.1f)" % (k, 100 * (k / k_crit - 1), k_crit))
        if r.get("closest") is None:
            print("    every frame was non-finite -- the run blew up at this stiffness")
            print()
            continue
        print("    closest non-bonded bead pair (any)  %.4f nm   (database %s)"
              % (r["closest"], DB_ANY_MIN))
        print("    closest non-bonded P-P              %.4f nm   (database %s)"
              % (r["closest_pp"], DB_PP_MIN))
        print("    per-frame closest: median %.4f, first frame %.4f nm"
              % (r["frame_min_median"], r["first"]["frame_min_median"]))
        print("    replica-frames sampled              %d%s"
              % (r["replica_frames"],
                 "  (%d non-finite, dropped)" % r["non_finite"] if r["non_finite"] else ""))
        print("    frames with any pair < sigma        %d / %d  (%.4f%%)"
              % (r["frames_below_sigma"], r["replica_frames"],
                 100.0 * r["frames_below_sigma"] / max(r["replica_frames"], 1)))
        print("    pair instances < sigma              %d / %d  (database 0 / 213732)"
              % (r["instances_below_sigma"], r["instances"]))
        print("    frames with any pair < 0.36         %d / %d  (database 0 / 213732)"
              % (r["frames_below_036"], r["replica_frames"]))
        print("    bins per frame  " + "  ".join(
            "[%.2f,%.2f) %.3f" % (a, b, r["nbins"][q] / max(r["replica_frames"], 1))
            for q, (a, b) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:]))))
        print()

    print("=== the grid, side by side ===")
    print("%10s %12s %12s %15s %18s %14s"
          % ("k", "closest any", "closest P-P", "frames<sigma", "instances<sigma", "frames<0.36"))
    for r in results:
        if r.get("closest") is None:
            print("%10g %12s" % (r["k"], "non-finite"))
            continue
        print("%10g %12.4f %12.4f %8d/%-6d %11d/%-6d %8d/%-6d"
              % (r["k"], r["closest"], r["closest_pp"],
                 r["frames_below_sigma"], r["replica_frames"],
                 r["instances_below_sigma"], r["instances"],
                 r["frames_below_036"], r["replica_frames"]))
    print()
    print("the criterion k is %.1f kJ/mol/nm^2; the shipped K_CLASH is %g" % (k_crit, C.K_CLASH))
    print("Reading the grid: a k whose closest pair and below-sigma count match the database's")
    print("zero is reproducing the criterion; a k that leaves pair instances inside sigma is a")
    print("repulsion that the field's own driving force exceeds. If the whole grid reads the")
    print("same, the answer is insensitive over it and that is the finding.")
    print()
    print("One thing to check before reading a flat grid as insensitivity: cg_energy_forces")
    print("caps the SUM of the forces on a bead (force_cap, 200 kJ/mol/nm as shipped) AFTER")
    print("every term including this one, so a stiff repulsion is clipped along with everything")
    print("else. Run the same grid with --cap none to see what the potential itself does; if the")
    print("capped grid is flat and the uncapped one is not, the cap is what set the closest")
    print("approach, not the stiffness.")


if __name__ == "__main__":
    main()
